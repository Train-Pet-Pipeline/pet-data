# pet-data 设计文档

## 概述

pet-data 是 Train-Pet-Pipeline 数据管线的核心仓库，负责数据采集、清洗、增强和弱监督。上游依赖 pet-schema v1.0（Schema 合同），下游消费方为 pet-annotation（读取 frames 表 + 帧图像文件）。

本文档基于 `pet-infra/docs/DEVELOPMENT_GUIDE.md` 第 4-5.2 章的规范，记录经过讨论确认的设计决策和调整。

## 设计决策

### 决策 1：数据源采用 BaseSource + FrameExtractor 策略模式

**背景**：文档原版每个数据源是独立脚本（`oxford_pet.py`、`selfshot/ingest.py` 等），无公共抽象。数据源会持续新增，重复的入库/去重/日志逻辑散落在每个脚本中。

**调整**：采用基类 + 策略注入模式，两个独立变化维度分离：

- **BaseSource ABC**：定义 `download()` + `validate_metadata()` 抽象方法，`ingest()` 模板方法编排完整流程
- **FrameExtractor 策略族**：VideoExtractor / ImageExtractor / AutoExtractor，负责"怎么从原始资源提取帧"

数据源和抽帧是两个独立变化维度——数据源决定"从哪拿"，extractor 决定"怎么抽"。新增数据源只需继承 BaseSource 实现两个方法 + 选一个 extractor。

**备选方案（已排除）**：

| 方案 | 排除理由 |
|---|---|
| 三层继承（BaseSource → ImageSource/VideoSource → 具体源） | hospital 等混合源不好归类 |
| 基类 + Mixin | extract 逻辑散落在每个子类，不够统一 |

### 决策 2：frames 表新增 phash 列

**背景**：去重需要计算 pHash 并与已入库帧对比。如果不持久化 pHash，每次去重都要重新加载图片计算哈希，数据量增长后不可接受。

**调整**：frames 表新增 `phash BLOB` 列，入库时一次计算后存入。`store.get_phashes()` 提供批量查询接口供去重使用。

### 决策 3：统一 CLI 入口配合 DVC

**背景**：文档原版各脚本独立运行。DVC stages 需要统一的命令格式，日常使用也需要一致的入口。

**调整**：新增 `src/pet_data/cli.py`，子命令包括 ingest / dedup / quality / augment / train-ae / score-anomaly。`dvc.yaml` 的 stages 通过 CLI 调用。pyproject.toml 注册 `pet-data` 命令。

### 决策 4：yt-dlp / praw 改为 optional-dependencies

**背景**：不是所有开发者都需要 YouTube 和 Reddit 数据源。这些依赖较重且有平台限制。

**调整**：移入 `[project.optional-dependencies]`，按需安装 `pip install -e ".[youtube]"` 或 `pip install -e ".[community]"`。

### 决策 5：视频生成后端可插拔

**背景**：Wan2.1 I2V 服务可能不可用（本地开发、CI 环境等），不应阻塞其他流程。未来可能替换为其他生成模型。

**调整**：定义 `VideoGenerator ABC`，Wan21Generator 和 NullGenerator 两个实现。无 endpoint 配置时自动降级为 NullGenerator。

### 决策 6：params.yaml 扩展

**背景**：文档只定义了 6 个参数字段。训练超参（max_epochs、batch_size、learning_rate）和传统增强参数也需要可配置，不硬编码。

**调整**：扩展 params.yaml，新增 `augmentation.traditional`、`weak_supervision.max_epochs/batch_size/learning_rate`、`dvc` 配置段。

## 项目结构

```
pet-data/
├── src/
│   └── pet_data/
│       ├── __init__.py
│       ├── cli.py                          # 统一 CLI 入口
│       ├── sources/                        # 数据源（基类 + 子类）
│       │   ├── __init__.py
│       │   ├── base.py                     # BaseSource ABC + RawItem + SourceMetadata
│       │   ├── extractors.py               # FrameExtractor 策略族
│       │   ├── selfshot.py                 # SelfShotSource
│       │   ├── oxford_pet.py               # OxfordPetSource
│       │   ├── coco_pet.py                 # CocoPetSource
│       │   ├── youtube.py                  # YoutubeSource
│       │   ├── community.py                # CommunitySource (Reddit)
│       │   └── hospital.py                 # HospitalSource
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── dedup.py                    # pHash 去重
│       │   └── quality_filter.py           # 模糊/曝光检测
│       ├── augmentation/
│       │   ├── __init__.py
│       │   ├── video_gen.py                # Wan2.1 I2V（可插拔）
│       │   ├── distortion_filter.py        # YOLO-nano 过滤失真帧
│       │   └── traditional_aug.py          # albumentations 光线/色温/噪声
│       ├── weak_supervision/
│       │   ├── __init__.py
│       │   ├── train_autoencoder.py        # 卷积 AE 训练
│       │   └── score_anomaly.py            # 重建误差打分
│       └── storage/
│           ├── __init__.py
│           ├── migrations/
│           │   └── 001_init.py             # Alembic 初始迁移
│           ├── schema.sql                  # frames 表 DDL
│           └── store.py                    # 唯一数据库访问接口
├── tests/
│   ├── conftest.py                         # 共享 fixtures
│   ├── test_sources/
│   │   ├── test_base.py                    # 基类契约测试
│   │   ├── test_extractors.py
│   │   ├── test_selfshot.py
│   │   └── test_oxford_pet.py
│   ├── test_processing/
│   │   ├── test_dedup.py
│   │   └── test_quality_filter.py
│   ├── test_augmentation/
│   │   ├── test_video_gen.py
│   │   ├── test_distortion_filter.py
│   │   └── test_traditional_aug.py
│   ├── test_weak_supervision/
│   │   ├── test_autoencoder.py
│   │   └── test_score_anomaly.py
│   └── test_storage/
│       ├── test_store.py
│       └── test_migrations.py
├── dvc.yaml
├── params.yaml
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── .gitignore
└── .dvcignore
```

## 数据源架构

### RawItem 与 SourceMetadata

```python
@dataclass
class SourceMetadata:
    species: str | None                  # "cat" / "dog" / None
    breed: str | None
    lighting: str | None                 # "bright" / "dim" / "infrared_night" / "unknown"
    bowl_type: str | None
    device_model: str | None             # selfshot 必填，其他可选
    video_id: str                        # 来源唯一标识

@dataclass
class RawItem:
    source: str                          # "selfshot" / "oxford_pet" / "coco" / ...
    resource_path: Path
    resource_type: Literal["video", "image"]
    metadata: SourceMetadata
```

### BaseSource ABC

```python
class BaseSource(ABC):
    source_name: str
    extractor: FrameExtractor

    def __init__(self, store: FrameStore, params: dict):
        self.store = store
        self.params = params

    def ingest(self) -> IngestReport:
        """模板方法：download → extract → dedup → quality → store。子类不覆盖。"""
        report = IngestReport()
        # 批量加载已有 phash，避免逐帧查库（N 次查询 → 1 次）
        existing_phashes = self.store.get_phashes()
        for item in self.download():
            if not self.validate_metadata(item):
                report.skipped += 1
                continue
            frames = self.extractor.extract(item, self.params)
            for frame_path in frames:
                dedup_result = dedup_check(frame_path, existing_phashes, self.params)
                if dedup_result.is_duplicate:
                    report.duplicates += 1
                    continue
                quality = assess_quality(frame_path, self.params)
                self.store.insert_frame(...)
                existing_phashes[frame_id] = dedup_result.phash  # 增量更新内存缓存
                report.inserted += 1
        return report

    @abstractmethod
    def download(self) -> Iterator[RawItem]:
        """拉取原始资源。子类实现。"""

    @abstractmethod
    def validate_metadata(self, item: RawItem) -> bool:
        """校验元数据完整性。子类实现。"""
```

### FrameExtractor 策略族

```python
class FrameExtractor(ABC):
    @abstractmethod
    def extract(self, item: RawItem, params: dict) -> list[Path]:
        """从 RawItem 提取帧，返回帧文件路径列表。"""

class VideoExtractor(FrameExtractor):
    """用 decord 按 params["frames"]["extract_fps"] 抽帧。"""

class ImageExtractor(FrameExtractor):
    """图片源，复制/转换到标准格式，返回 [path]。"""

class AutoExtractor(FrameExtractor):
    """按 item.resource_type 自动分发到 VideoExtractor 或 ImageExtractor。"""
```

### 各数据源子类

| 子类 | extractor | download 逻辑 | validate_metadata 特殊要求 |
|---|---|---|---|
| SelfShotSource | VideoExtractor | 扫描本地目录 | device_model 必填 |
| OxfordPetSource | ImageExtractor | torchvision / 直接下载 | species + breed 必填 |
| CocoPetSource | ImageExtractor | COCO API 过滤 cat/dog | species 必填 |
| YoutubeSource | VideoExtractor | yt-dlp，robots.txt 合规 | video_id 必填 |
| CommunitySource | AutoExtractor | PRAW 公开帖子，限速 | source URL 必填 |
| HospitalSource | AutoExtractor | 本地目录，入库时 PII 脱敏 | species 必填 |

## 存储层

### FrameStore 接口

```python
class FrameStore:
    def __init__(self, db_path: Path): ...

    # 写入
    def insert_frame(self, frame: FrameRecord) -> str: ...
    def bulk_insert_frames(self, frames: list[FrameRecord]) -> int: ...

    # 查询
    def get_frame(self, frame_id: str) -> FrameRecord | None: ...
    def query_frames(self, filters: FrameFilter) -> list[FrameRecord]: ...
    def get_phashes(self, source: str | None = None) -> dict[str, bytes]: ...

    # 更新
    def update_quality(self, frame_id: str, quality_flag: str, blur_score: float) -> None: ...
    def update_anomaly(self, frame_id: str, is_candidate: bool, score: float) -> None: ...
    def update_annotation_status(self, frame_id: str, status: str) -> None: ...
    def update_augmentation(self, frame_id: str, aug_quality: str, parent_frame_id: str) -> None: ...

    # 统计
    def count_by_source(self) -> dict[str, int]: ...
    def count_by_status(self) -> dict[str, int]: ...
    def count_normal_frames(self) -> int: ...
```

### frames 表（在文档基础上新增 phash 列）

```sql
CREATE TABLE frames (
    frame_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    source          TEXT NOT NULL,
    frame_path      TEXT NOT NULL,
    data_root       TEXT NOT NULL,
    timestamp_ms    INTEGER,
    species         TEXT,
    breed           TEXT,
    lighting        TEXT CHECK(lighting IN ('bright','dim','infrared_night','unknown')),
    bowl_type       TEXT,
    quality_flag    TEXT NOT NULL DEFAULT 'normal'
                    CHECK(quality_flag IN ('normal','low','failed')),
    blur_score      REAL,
    phash           BLOB,                -- 新增：pHash 二进制，去重用
    aug_quality     TEXT CHECK(aug_quality IN ('ok','failed') OR aug_quality IS NULL),
    aug_seed        INTEGER,
    parent_frame_id TEXT,
    is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
    anomaly_score   REAL,
    annotation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(annotation_status IN ('pending','annotating','auto_checked',
                                    'approved','needs_review','reviewed','rejected','exported')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_frames_status    ON frames(annotation_status);
CREATE INDEX idx_frames_source    ON frames(source);
CREATE INDEX idx_frames_quality   ON frames(quality_flag);
CREATE INDEX idx_frames_anomaly   ON frames(is_anomaly_candidate, anomaly_score DESC);
-- 注意：phash 不建 B-tree 索引，汉明距离比较需在 Python 内存中完成
-- 批量加载 phash 由 store.get_phashes() 实现
```

### FrameRecord 与 FrameFilter

```python
@dataclass
class FrameRecord:
    frame_id: str
    video_id: str
    source: str
    frame_path: str                      # 相对于 data_root
    data_root: str
    timestamp_ms: int | None
    species: str | None
    breed: str | None
    lighting: str | None
    bowl_type: str | None
    quality_flag: str = "normal"
    blur_score: float | None = None
    phash: bytes | None = None
    aug_quality: str | None = None
    aug_seed: int | None = None
    parent_frame_id: str | None = None
    is_anomaly_candidate: bool = False
    anomaly_score: float | None = None
    annotation_status: str = "pending"

@dataclass
class FrameFilter:
    source: str | None = None
    quality_flag: str | None = None
    annotation_status: str | None = None
    is_anomaly_candidate: bool | None = None
    limit: int = 1000
    offset: int = 0
```

## Processing 模块

### dedup.py

```python
@dataclass
class DedupResult:
    is_duplicate: bool
    phash: bytes
    duplicate_of: str | None = None

def compute_phash(image_path: Path) -> bytes: ...
def hamming_distance(hash_a: bytes, hash_b: bytes) -> int: ...
def dedup_check(image_path: Path, existing_phashes: dict[str, bytes],
                params: dict) -> DedupResult:
    """跨来源全局去重。接收内存中的 phash 字典，避免逐帧查库。
    阈值从 params["frames"]["dedup_hamming_threshold"] 读取。"""
```

### quality_filter.py

```python
@dataclass
class QualityResult:
    quality_flag: str                    # "normal" / "low" / "failed"
    blur_score: float

def assess_quality(image_path: Path, params: dict) -> QualityResult:
    """Laplacian 方差评估。打标不删除。"""
```

## Augmentation 模块

### video_gen.py（可插拔）

```python
class VideoGenerator(ABC):
    @abstractmethod
    def generate(self, seed_frame: Path, prompt: str, seed: int) -> Path | None: ...

class Wan21Generator(VideoGenerator):
    """Wan2.1 I2V。endpoint 从环境变量/params 读取。tenacity 重试。"""

class NullGenerator(VideoGenerator):
    """空实现，服务不可用时降级。"""

def run_augmentation(store: FrameStore, params: dict,
                     generator: VideoGenerator | None = None) -> AugmentReport:
    """查种子帧 → 生成变体 → 抽帧 → distortion_filter → 入库。"""
```

### distortion_filter.py

```python
def filter_distortion(frame_paths: list[Path], params: dict) -> list[tuple[Path, str]]:
    """YOLO-nano 检测。模型不可用时降级全部 ok。"""
```

### traditional_aug.py

```python
def augment_frame(image_path: Path, output_dir: Path, params: dict) -> list[Path]:
    """albumentations 管线：亮度/色温/噪声/旋转。"""
```

## Weak Supervision 模块

### train_autoencoder.py

```python
class FeedingAutoencoder(nn.Module):
    """文档定义的卷积 AE。输入 (B,3,224,224) → 瓶颈 (B,32,14,14) → 重建。"""
    def forward(self, x: Tensor) -> Tensor: ...
    def anomaly_score(self, x: Tensor) -> Tensor: ...

@dataclass
class TrainReport:
    num_frames: int
    epochs: int
    final_train_loss: float
    final_val_loss: float
    model_path: Path

def train(store: FrameStore, params: dict, output_dir: Path) -> TrainReport:
    """训练 AE。帧数 < min_normal_frames 时报错。保存到 output_dir/autoencoder.pt。"""
```

### score_anomaly.py

```python
@dataclass
class ScoreReport:
    total_scored: int
    anomaly_candidates: int
    mean_score: float
    threshold: float

def score_frames(store: FrameStore, model_path: Path, params: dict) -> ScoreReport:
    """加载 AE → 批量推理 → score > threshold 标记为 anomaly candidate。"""
```

## DVC 管线

```yaml
stages:
  ingest:
    cmd: python -m pet_data.cli ingest --source ${source}
    deps: [src/pet_data/sources/, src/pet_data/storage/store.py]
    params: [frames]
    outs: [{data/frames/: {cache: true}}]

  dedup:
    cmd: python -m pet_data.cli dedup
    deps: [src/pet_data/processing/dedup.py, data/frames/]
    params: [frames.dedup_hamming_threshold]

  quality:
    cmd: python -m pet_data.cli quality
    deps: [src/pet_data/processing/quality_filter.py, data/frames/]

  augment:
    cmd: python -m pet_data.cli augment
    deps: [src/pet_data/augmentation/, data/frames/]
    params: [augmentation]
    outs: [{data/augmented/: {cache: true}}]

  train_ae:
    cmd: python -m pet_data.cli train-ae
    deps: [src/pet_data/weak_supervision/train_autoencoder.py]
    params: [weak_supervision]
    outs: [{models/autoencoder.pt: {cache: true}}]

  score_anomaly:
    cmd: python -m pet_data.cli score-anomaly
    deps: [src/pet_data/weak_supervision/score_anomaly.py, models/autoencoder.pt]
    params: [weak_supervision.anomaly_score_threshold]
```

## params.yaml

```yaml
data_root: "/data/pet-data"

frames:
  extract_fps: 1.0
  dedup_hamming_threshold: 10
  quality_blur_threshold: 100.0

augmentation:
  video_gen_count_per_seed: 10
  distortion_conf_threshold: 0.5
  traditional:
    brightness_limit: 0.2
    noise_var_limit: 0.02

weak_supervision:
  anomaly_score_threshold: 0.07
  min_normal_frames: 2000
  max_epochs: 100
  batch_size: 32
  learning_rate: 0.001

dvc:
  remote: "local"
  remote_path: "/data/dvc-cache"
```

## 测试策略

### 共享 fixtures (conftest.py)

- `tmp_store` — 临时 SQLite，测试后自动清理
- `tmp_data_root` — 临时数据目录
- `sample_image` — 224x224 测试图片
- `sample_video` — 3 秒测试视频
- `default_params` — params.yaml 默认值
- `seeded_store` — 预插入 10 条帧记录

### 覆盖范围

| 模块 | 测试文件 | 关键用例 |
|---|---|---|
| BaseSource 契约 | test_base.py | ingest 全流程、空迭代、validate 失败跳过、重复跳过 |
| FrameExtractor | test_extractors.py | Video/Image/Auto 三种策略 |
| SelfShotSource | test_selfshot.py | 目录扫描、device_model 校验 |
| OxfordPetSource | test_oxford_pet.py | mock 下载、metadata 校验 |
| dedup | test_dedup.py | 相同/不同/微修改图片、跨 source 去重 |
| quality_filter | test_quality_filter.py | 清晰/模糊图片判定 |
| video_gen | test_video_gen.py | NullGenerator 降级、Wan21 mock、重试 |
| distortion_filter | test_distortion_filter.py | 高/低置信度、模型不可用降级 |
| traditional_aug | test_traditional_aug.py | 输出数量、尺寸不变 |
| autoencoder | test_autoencoder.py | 前向形状、anomaly_score、帧不足报错、训练保存 |
| score_anomaly | test_score_anomaly.py | 全低分/部分高分、store 更新验证 |
| store | test_store.py | CRUD、bulk insert 回滚、filter 组合、统计 |
| migrations | test_migrations.py | 空库初始化、幂等性 |

## 工程配置

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pet-data"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "pet-schema @ git+https://github.com/Train-Pet-Pipeline/pet-schema.git@v1.0.0",
    "Pillow>=10.0,<11.0",
    "imagehash>=4.3,<5.0",
    "decord>=0.6,<1.0",
    "albumentations>=1.3,<2.0",
    "torch>=2.1,<3.0",
    "torchvision>=0.16,<1.0",
    "alembic>=1.13,<2.0",
    "tenacity>=8.0,<9.0",
    "pyyaml>=6.0,<7.0",
    "dvc>=3.40,<4.0",
    "requests>=2.31,<3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff", "mypy", "pip-tools", "opencv-python-headless>=4.8"]
youtube = ["yt-dlp>=2024.1"]
community = ["praw>=7.7,<8.0"]

[project.scripts]
pet-data = "pet_data.cli:main"
```

### Makefile

四个必须 target：setup / test / lint / clean。

## 与 DEVELOPMENT_GUIDE 文档的差异汇总

| # | 项目 | 文档原版 | 本设计 | 理由 |
|---|---|---|---|---|
| 1 | frames 表 | 无 phash 列 | 新增 `phash BLOB` + 索引 | 去重性能，避免每次重算 |
| 2 | sources/ 架构 | 独立脚本 | BaseSource ABC + FrameExtractor 策略 | 可扩展，消除重复逻辑 |
| 3 | CLI 入口 | 无 | `pet_data.cli` 统一入口 | DVC stages 和日常使用需要 |
| 4 | params.yaml | 6 个字段 | 扩展训练超参和增强参数 | 不硬编码 |
| 5 | yt-dlp / praw | 主依赖 | optional-dependencies | 按需安装 |
| 6 | 视频生成 | 直接调用 | VideoGenerator ABC 可插拔 | 服务不可用时降级 |

这些差异确认后将回写到 DEVELOPMENT_GUIDE.md。
