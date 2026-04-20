"""Shared test fixtures for pet-data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from pet_data.storage.store import FrameRecord, FrameStore


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """Temporary data directory simulating DATA_ROOT."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def default_params() -> dict:
    """Load default params.yaml values."""
    params_path = Path(__file__).parent.parent / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_image(tmp_data_root: Path) -> Path:
    """Generate a 224x224 test PNG image with random pixels."""
    img_path = tmp_data_root / "test_frame.png"
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


@pytest.fixture
def sample_image_duplicate(tmp_data_root: Path, sample_image: Path) -> Path:
    """A copy of sample_image at a different path (same pixels = same phash)."""
    dup_path = tmp_data_root / "test_frame_dup.png"
    img = Image.open(sample_image)
    img.save(dup_path)
    return dup_path


@pytest.fixture
def sample_image_different(tmp_data_root: Path) -> Path:
    """A completely different test image."""
    img_path = tmp_data_root / "test_frame_diff.png"
    rng = np.random.default_rng(999)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


@pytest.fixture
def fresh_db_with_frames(tmp_path: Path) -> Path:
    """FrameStore DB with 3 VisionSample-compliant rows (all required fields populated)."""
    db_path = tmp_path / "db.sqlite"
    store = FrameStore(db_path)
    for i in range(3):
        store.insert_frame(
            FrameRecord(
                frame_id=f"sha256:f{i}",
                video_id=f"v{i}",
                source="youtube",
                frame_path=f"frames/{i}.jpg",
                data_root="/data",
                timestamp_ms=1000 * i,
                species="dog",
                lighting="bright",
                bowl_type="ceramic",
                quality_flag="normal",
                blur_score=120.0,
                modality="vision",
                storage_uri=f"local:///data/frames/{i}.jpg",
                frame_width=1920,
                frame_height=1080,
                brightness_score=0.6,
            )
        )
    store.close()
    return db_path
