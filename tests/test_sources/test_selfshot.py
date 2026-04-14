"""Tests for SelfShotSource."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pet_data.sources.base import RawItem, SourceMetadata
from pet_data.sources.selfshot import SelfShotSource
from pet_data.storage.store import FrameStore


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore."""
    return FrameStore(tmp_path / "test.db")


@pytest.fixture
def selfshot_dir(tmp_data_root: Path) -> Path:
    """Create a selfshot directory with a test video."""
    d = tmp_data_root / "selfshot"
    d.mkdir()
    meta_dir = d / "meta"
    meta_dir.mkdir()
    return d


@pytest.fixture
def selfshot_with_video(selfshot_dir: Path) -> Path:
    """Add a test video to the selfshot directory."""
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv not installed")
    video_path = selfshot_dir / "cat_eating_001.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (224, 224))
    rng = np.random.default_rng(42)
    for _ in range(30):
        writer.write(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    writer.release()

    import yaml
    meta_path = selfshot_dir / "meta" / "cat_eating_001.yaml"
    with open(meta_path, "w") as f:
        yaml.dump({
            "species": "cat",
            "breed": "british_shorthair",
            "lighting": "bright",
            "bowl_type": "round",
            "device_model": "RK3576-proto-v1",
        }, f)

    return selfshot_dir


class TestSelfShotSource:
    """Tests for SelfShotSource."""

    def test_download_yields_videos(
        self, store: FrameStore, default_params: dict, selfshot_with_video: Path,
        tmp_data_root: Path,
    ) -> None:
        """download() yields RawItems for each video in the directory."""
        default_params["selfshot_dir"] = str(selfshot_with_video)
        default_params["data_root"] = str(tmp_data_root)
        src = SelfShotSource(store, default_params)
        items = list(src.download())
        assert len(items) >= 1
        assert items[0].resource_type == "video"

    def test_validate_metadata_requires_device_model(
        self, store: FrameStore, default_params: dict, selfshot_with_video: Path,
        tmp_data_root: Path,
    ) -> None:
        """validate_metadata returns False if device_model is missing."""
        default_params["selfshot_dir"] = str(selfshot_with_video)
        default_params["data_root"] = str(tmp_data_root)
        src = SelfShotSource(store, default_params)
        item = RawItem(
            source="selfshot",
            resource_path=Path("fake.mp4"),
            resource_type="video",
            metadata=SourceMetadata(
                species="cat", breed="tabby", lighting="bright",
                bowl_type=None, device_model=None, video_id="vid",
            ),
        )
        assert src.validate_metadata(item) is False

    def test_validate_metadata_passes_with_device_model(
        self, store: FrameStore, default_params: dict, selfshot_with_video: Path,
        tmp_data_root: Path,
    ) -> None:
        """validate_metadata returns True when device_model is set."""
        default_params["selfshot_dir"] = str(selfshot_with_video)
        default_params["data_root"] = str(tmp_data_root)
        src = SelfShotSource(store, default_params)
        item = RawItem(
            source="selfshot",
            resource_path=Path("fake.mp4"),
            resource_type="video",
            metadata=SourceMetadata(
                species="cat", breed="tabby", lighting="bright",
                bowl_type=None, device_model="RK3576-proto-v1", video_id="vid",
            ),
        )
        assert src.validate_metadata(item) is True
