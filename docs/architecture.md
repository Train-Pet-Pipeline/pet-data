# pet-data 技术设计文档

> 维护说明：代码变更影响本文档任一章节时，与代码在同 PR 内更新
> 最后对齐：pet-data v1.3.0 / 2026-04-23 (Phase 3 ecosystem optimization)

## 1. 仓库职责

- 数据采集 / 清洗 / 增强 / 弱监督 — pipeline 源头
- pipeline 位置：pet-schema → **pet-data** → pet-annotation → pet-train → pet-eval → pet-quantize → pet-ota
- 做什么：从外部源 ingest → dedup → quality filter → anomaly scoring → 进入标注阶段 (via pet-annotation) 或直接 export dataset (via DATASETS registry)
- 不做什么：不做标注 (pet-annotation)，不训练 (pet-train)，不做 evaluator/gate 决策 (pet-eval)；DB 只通过 `store.py` 操作不直连；不 skip dedup

## 2. 输入输出契约

- **上游**: pet-schema (peer-dep β — v3.1.0)；pet-infra (peer-dep β — v2.6.0)
- **下游消费**: pet-annotation (Annotation producer reads Samples via DATASETS plugin)；pet-train (Trainer 的 DATASETS plugin 读 VisionSample/AudioSample)
- **Sample 类型**:
  - `VisionSample`：从 `frames` 表 → `frame_row_to_vision_sample()` (`storage/adapter.py`)
  - `AudioSample`：从 `audio_samples` 表 → `audio_row_to_audio_sample()` (`storage/adapter.py`)
- **契约变更流程**: pet-schema 改 → `.github/workflows/ci.yml` `repository_dispatch: schema-updated` 触发 pet-data CI → 本仓需 pass

## 3. 架构总览

```
src/pet_data/
├── sources/              ← 7 ingester 类（BaseSource 子类）
│   ├── base.py           ← BaseSource + ingester_name + default_provenance ClassVar
│   ├── youtube.py        ← YouTubeIngester (provenance=youtube)
│   ├── community.py      ← CommunityIngester (provenance=community)
│   ├── selfshot.py       ← SelfshotIngester (provenance=community)
│   ├── oxford_pet.py     ← OxfordPetIngester (provenance=academic_dataset)
│   ├── coco_pet.py       ← CocoIngester (provenance=academic_dataset)
│   ├── hospital.py       ← HospitalIngester (provenance=device)
│   ├── local_dir.py      ← LocalDirIngester (provenance=device)
│   └── extractors.py     ← FrameExtractor (Video / Image dispatchers)
├── storage/
│   ├── store.py          ← FrameStore / AudioStore — 唯一 DB 写入口
│   ├── adapter.py        ← DB row → Pydantic Sample 转换
│   ├── schema.sql        ← 初始 frames 表 DDL (读到 001_init migration 中)
│   └── migrations/       ← Alembic style 迁移 (001-004)；已提交不可改
├── processing/           ← quality_filter / dedup (phash)
├── augmentation/         ← traditional_aug (params from params.yaml) + video_gen + distortion_filter
├── datasets/             ← DATASETS plugin 入口 (为 pet-infra REGISTRIES 注册)
├── weak_supervision/     ← autoencoder scoring
├── cli.py / cli_legacy.py ← CLI 入口
└── _register.py          ← plugin discovery entry point
tests/
.github/workflows/ci.yml  ← peer-dep 5 步装序
params.yaml               ← 所有数值配置 (no-hardcode policy)
```

关键数据流：`ingester.ingest()` → `dedup_check` → `store.insert_frame()` → `quality_filter` → `store.update_anomaly()` → `adapter.frame_row_to_vision_sample()` → downstream consumer

## 4. 核心模块详解

### 4.1 `sources/base.py` — BaseSource abstract + concept separation

- **Why**: 每个 source 是独立 ingest 实现（YouTube / community / academic datasets / first-party device）。`BaseSource.ingest()` 是 template method，封装公共 pipeline（download → extract → dedup → insert），subclass 只实现 `download()` 和 `validate_metadata()`。
- **Tradeoff**: `ingester_name`（implementation 标识，如 `"oxford_pet"`）和 `default_provenance`（provenance category，如 `"academic_dataset"`，必须是 `SourceType` 6 literals 之一）通过 ClassVar 分别声明。Phase 3 之前老代码把 `ingester_name` 直接当 `source_type` 传给 `SourceInfo`，导致 `oxford_pet` / `coco` / `hospital` / `local_dir` 等 ingester 在构造 `VisionSample` 时 ValidationError（finding F1）。ClassVar 而非 instance attr — 每类固定 provenance，不支持运行时 per-call override（详见 §9.2 followup）。
- **Pitfall**: 新增 ingester 时必须同时声明 `ingester_name: str` 和 `default_provenance: ClassVar[SourceType]`；missing 任一在实例化时 `AttributeError`。`default_provenance` 必须是 v3.1.0 `SourceType` 的 6 literal 之一（`youtube / community / device / synthetic / academic_dataset / commercial_licensed`）。
- **代码路径**: `base.py:71-72`（ClassVar 声明）；`base.py:139`（`record` 构造时读 `self.default_provenance`）

### 4.2 `storage/store.py` — FrameStore 唯一 DB 写入口

- **Why**: `CLAUDE.md` 强制"DB 只通过 `store.py` 操作不直连"。集中 insert / update / query 入口便于加 validation + 结构化 logging + migration safety。`FrameStore.__init__()` 打开 SQLite 连接后立即跑 `_apply_subsequent_migrations()`（`store.py:114`），确保任何读写前 schema 已 up-to-date。
- **Tradeoff**: dataset plugin（`datasets/audio_clips.py:46`，`datasets/vision_frames.py:41`）直连 `sqlite3` 做只读迭代 — 只读例外（§8.1）。这是性能取舍：`FrameStore` CRUD 对百万样本 streaming consumer 过重。
- **Pitfall**: `FrameRecord.provenance_type` 默认值 `"device"`（`store.py:67`）是 fallback，只在 migration 004 未跑的老 DB 生效；正常路径 `FrameStore.__init__` 总跑 pending migrations，不会踩此 fallback。`_row_to_record()` 对 `provenance_type` 有 `key in row.keys()` 防御（`store.py:575`），同理。添加新字段：先加新 Alembic migration（新文件，不改旧），再加 `FrameRecord` 字段，再改 `INSERT` / `_row_to_record` / 测试。

### 4.3 `storage/adapter.py` — DB row → Pydantic Sample

- **Why**: DB schema（SQLite 列名）≠ Pydantic contract（pet-schema `SourceInfo`）。`adapter.py` 是概念隔离层：DB 里 `source` 列存 `ingester_name`，`provenance_type` 列存 provenance；输出 `SourceInfo.source_type = row["provenance_type"]`，`SourceInfo.ingester = row["source"]`（`adapter.py:65-68`）。
- **Tradeoff**: 可以把概念分离完全推到 DB schema（rename `source` 列 → `ingester_name`）；但 rename 是 breaking migration，成本/收益不相当。保持 DB 列名 `source` + adapter 做映射，blast radius 小（详见 §9.6 followup）。
- **Pitfall**: `frame_row_to_vision_sample()` 直接访问 `row["provenance_type"]`（`adapter.py:66`），没有 fallback。`store.py` 有防御式 `key in row.keys()` fallback，`adapter.py` 没有——当前安全，因为 `FrameStore.__init__` 总跑 pending migrations，正常路径不会踩 `KeyError`（详见 §9.4 followup）。lighting 归一化走 `_LIGHTING_MAP`（`adapter.py:30-35`），未知 DB 值 raise `ValueError`。

### 4.4 `storage/migrations/004_add_provenance_type.py` — 概念分离第 3 步

- **Why**: SQLite 无 `ALTER TABLE ADD CONSTRAINT` 语法，想加 CHECK 必须 rebuild table。`upgrade()` 分 3 步：① 加 nullable `provenance_type` 列；② backfill（ingester → provenance 映射表 `_INGESTER_TO_PROVENANCE`，未知 source 降 fallback `"device"` + warning log）；③ table rebuild 带 `NOT NULL DEFAULT 'device' CHECK(...)` 约束（`004:107-163`）。
- **Tradeoff**: Table rebuild 是重操作但 data integrity 保证。`upgrade()` 以 `"duplicate column name"` 守护幂等性（`004:76-78`）——FrameStore 的 `_apply_subsequent_migrations()` 总对所有 migration 文件调用 `upgrade()`，幂等守护必不可少。
- **Pitfall**: `downgrade()` 必须用**显式 `CREATE TABLE` with full schema**，不能用 `CREATE TABLE x AS SELECT ... FROM y`（SQLite 后者创建 zero-constraint 表！）。`downgrade()` 已用显式 DDL 重建（`004:178-239`），包括 schema.sql + migration 002 的所有 CHECK / PRIMARY KEY / NOT NULL / DEFAULT 约束，并有 regression test `test_004_downgrade_preserves_pre_004_check_constraints` 守护。未来扩 `SourceType` literals：新 migration（005+），不改本文件。

## 5. 扩展点

### 5.1 添加新 ingester（如 GithubRepoIngester）

1. 在 `src/pet_data/sources/` 加新 Python 文件（如 `github_repo.py`）
2. 继承 `BaseSource`；声明 `ingester_name = "github_repo"`；`default_provenance = "<适当 SourceType literal>"`；`extractor = VideoExtractor()` 或 `ImageExtractor()`（视数据类型）
3. 实现 `download()` (yields `RawItem`) 和 `validate_metadata()` abstractmethod
4. 在 `sources/__init__.py` 加 import 让类被发现（若有 dynamic discovery，不需要）
5. 加 unit test 覆盖：(a) `ingester_name` 和 `default_provenance` 正确；(b) ingest flow 不出错（integration test w/ fixture）
6. **Note**: sources 当前不是 pet-infra plugin（无 SOURCES registry）；Phase 5+ followup 会做（§9.1）

### 5.2 支持新的 SourceType provenance literal

1. 在 pet-schema 加新 literal（需 pet-schema minor bump）
2. 在 pet-data 新建 `005_extend_provenance_literals.py` migration（不改 004；rebuild table + 新 CHECK 约束）
3. 更新 `004._INGESTER_TO_PROVENANCE` 等映射逻辑（若 ingester → 新 literal）
4. 新 migration 有 regression test 覆盖 upgrade + downgrade

### 5.3 添加新 augmentation 参数

1. 在 `params.yaml` `augmentation.<namespace>.*` 加 sub-key
2. `traditional_aug.py` 或其他 augmentation 模块从 params dict 读（**严禁硬编码**）
3. 加 unit test：(a) param override 生效；(b) missing param raises `KeyError`（防 fallback drift）

## 6. 依赖管理

- **对上游依赖**:
  - `pet-schema`（peer-dep β；无 pin；`_register.py` 中 `RuntimeError` fail-fast guard；CI 装序先装 `pet-schema@v3.1.0`）
  - `pet-infra`（peer-dep β；同模式，`@v2.6.0`）
- **CI 装序**（5 步，`.github/workflows/ci.yml`）:
  1. `pip install 'pet-schema @ git+...@v3.1.0'`
  2. `pip install 'pet-infra @ git+...@v2.6.0'`
  3. `pip install -e ".[dev]" --no-deps`（no-deps over peer-deps）
  4. `pip install -e ".[dev]"`（resolve remaining deps）
  5. version assert：`SCHEMA_VERSION == '3.1.0'` & `pet_infra.__version__.startswith('2.')` & `pet_data.__version__ == '1.3.0'`
- **第三方依赖关键约束**: `albumentations`（augmentation）；`imagehash`（dedup phash）；`torch`（weak supervision autoencoder）；`ffmpeg` + `yt-dlp`（video extraction）

## 7. 本地开发与测试

- `conda activate pet-pipeline`（共享 env）
- `make setup` = pip install peer-deps + editable pet-data
- `make test` = pytest（130 passed，Phase 3 baseline）
- `make lint` = ruff + mypy
- **Test 环境 pitfall**:
  - 跑 `pytest` 前 conda env 必须有最新 pet-schema v3.1.0 + pet-infra v2.6.0 安装（否则 `_register.py` guard `RuntimeError`）
  - `pytest-env` 不用；env var 在 `conftest.py` 里 set
- **migration test pitfall**: 测试迁移时用 `:memory:` db 而不是 `tmp_path` 里的文件，避免文件权限问题

## 8. 已知复杂点（复杂但必要）

### 8.1 Dataset plugin 直连 sqlite 绕过 store.py（只读）

- **位置**: `datasets/audio_clips.py:46`，`datasets/vision_frames.py:41`
- **保留理由**: streaming read-only 迭代；无写路径 → 无 data integrity risk。`FrameStore` CRUD 对 streaming consumer 过重
- **删了会损失什么**: dataset plugins 性能退化（CRUD wrapper overhead for 百万样本迭代）；若强制走 `store.py` 可能 OOM
- **重新审视触发条件**: 若 dataset plugin 开始需要 mutate 数据（加字段 / update 状态）就必须走 `store.py` — 此时评估是否给 store 加专用 iterator API

### 8.2 DVC `dedup` stage 用目录 hash 依赖

- **位置**: `dvc.yaml dedup` stage `deps: [data/frames/]`
- **保留理由**: DVC 目录 hash 自动跟踪内容变化；简单 + 正确；显式 stage-name deps 要求改 `dvc.yaml` 配合 upstream 重命名
- **删了会损失什么**: 自动增量感知；改用 stage-name deps 要更精细地手动管理 stage 之间的依赖图
- **重新审视触发条件**: 若 DVC pipeline 变得更复杂（10+ stages，非线性 DAG），显式依赖可能更好

## 9. Phase 5+ Followups

### 9.1 Sources 插件化（SOURCES registry）

- **触发条件**: 当需要接入 3+ 自定义 ingester（如 `GithubRepoIngester`，`S3BucketIngester`）时
- **描述**: 把 `BaseSource` 移到 pet-infra + 加 `SOURCES` registry（pet-infra 第 7 个 registry，和 `TRAINERS` / `EVALUATORS` 并列）；`@SOURCES.register_module` 装饰器替代 class tree；用户写 plugin 不动 pet-data 源树
- **ROI 评估**: 当前 7 ingester 固定，插件化 ROI 中等。若有 2+ 外部 source contributor 就立即做
- **blast radius**: 跨 pet-infra + pet-data + pet-schema 三仓 ~500 行重构；需独立 brainstorm/spec/plan

### 9.2 Provenance 运行时 override

- **触发条件**: 当 `local_dir.py` 被用于非 device 数据（如学术数据集或商业许可数据下载到本地目录）
- **描述**: `BaseSource.ingest()` 当前读 `self.default_provenance`（ClassVar，固定，`base.py:139`）。改为 `self.params.get("provenance_type", self.default_provenance)` 允许 config override per-run
- **ROI**: 单行改动 + 1 测试；触发条件出现后立即做

### 9.3 DataRequirement 模型-数据契约

- **触发条件**: 训练新模型时需要验证数据集满足模型要求（字段 / 数量 / modality / provenance 限制）
- **描述**: pet-schema 加 `DataRequirement` 模型（`required_fields / min_samples / allowed_modalities / allowed_provenance`）；trainer plugin 声明 `required_data: ClassVar[DataRequirement]`；`pet validate --recipe` preflight check
- **位置**: 跨 pet-schema + pet-train + pet-infra；规划在 Phase 10 closeout 或独立 phase

### 9.4 adapter.py `provenance_type` 防御式 fallback

- **触发条件**: 若发现绕过 `FrameStore` 初始化的代码路径
- **描述**: `frame_row_to_vision_sample()` 直接访问 `row["provenance_type"]`（`adapter.py:66`）；`store.py` 有 fallback（`store.py:575`），`adapter.py` 没有。当前 `FrameStore.__init__` 总跑 pending migrations 所以正常路径不会踩 `KeyError`；防御式 fallback 会让契约更明显（一行改动）

### 9.5 audio_samples CHECK constraint 扩展（pre-existing）

- **触发条件**: 若 audio pipeline 需要接入 `academic_dataset` 或 `commercial_licensed` provenance 的音频数据
- **描述**: migration `003_add_audio_samples.py` CHECK 仍限制为原 4 literals（`youtube/community/device/synthetic`）。需要新 migration（005+ audio equivalent）扩到 6 literals — 和 frames 对齐

### 9.6 DB column `source` → `ingester_name` 重命名

- **触发条件**: 当下游所有消费方都更新完、不再对 `source` 列名有外部期望时
- **描述**: 当前 DB 列名仍是 `source` 但语义已是 `ingester_name`（存 7 ingester 名）。新 migration 重命名列 + 下游 `adapter.py` 更新。non-trivial 因为 index 也要重命名（`idx_frames_source`）
