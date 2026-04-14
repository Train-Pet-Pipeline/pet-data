"""Tests for FrameExtractor strategy classes."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from PIL import Image

from pet_data.sources.base import RawItem, SourceMetadata
from pet_data.sources.extractors import AutoExtractor, ImageExtractor, VideoExtractor


def _make_raw_item(
    path: Path, resource_type: Literal["video", "image"], source: str = "test"
) -> RawItem:
    """Create a RawItem for testing."""
    return RawItem(
        source=source,
        resource_path=path,
        resource_type=resource_type,
        metadata=SourceMetadata(
            species=None, breed=None, lighting=None,
            bowl_type=None, device_model=None, video_id="test-vid",
        ),
    )


class TestImageExtractor:
    """Tests for ImageExtractor."""

    def test_extract_returns_single_path(
        self, sample_image: Path, default_params: dict, tmp_data_root: Path
    ) -> None:
        """ImageExtractor returns a list with one path for a single image."""
        item = _make_raw_item(sample_image, "image")
        extractor = ImageExtractor(output_dir=tmp_data_root / "extracted")
        result = extractor.extract(item, default_params)
        assert len(result) == 1
        assert result[0].exists()

    def test_extracted_image_is_valid(
        self, sample_image: Path, default_params: dict, tmp_data_root: Path
    ) -> None:
        """Extracted image can be opened by PIL."""
        item = _make_raw_item(sample_image, "image")
        extractor = ImageExtractor(output_dir=tmp_data_root / "extracted")
        result = extractor.extract(item, default_params)
        img = Image.open(result[0])
        assert img.size == (224, 224)


class TestVideoExtractor:
    """Tests for VideoExtractor."""

    def test_extract_returns_frames(
        self, tmp_data_root: Path, default_params: dict
    ) -> None:
        """VideoExtractor extracts frames from a video at configured fps."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv not installed")

        video_path = tmp_data_root / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (224, 224))
        rng = np.random.default_rng(42)
        for _ in range(30):  # 3 seconds at 10fps
            frame = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        item = _make_raw_item(video_path, "video")
        extractor = VideoExtractor(output_dir=tmp_data_root / "extracted")
        # fps=1.0 → 3 seconds → 3 frames
        result = extractor.extract(item, default_params)
        assert len(result) >= 2  # at least 2 frames from 3s video at 1fps
        for p in result:
            assert p.exists()


class TestAutoExtractor:
    """Tests for AutoExtractor."""

    def test_routes_image_to_image_extractor(
        self, sample_image: Path, default_params: dict, tmp_data_root: Path
    ) -> None:
        """AutoExtractor delegates images to ImageExtractor."""
        item = _make_raw_item(sample_image, "image")
        extractor = AutoExtractor(output_dir=tmp_data_root / "extracted")
        result = extractor.extract(item, default_params)
        assert len(result) == 1

    def test_routes_video_to_video_extractor(
        self, tmp_data_root: Path, default_params: dict
    ) -> None:
        """AutoExtractor delegates videos to VideoExtractor."""
        try:
            import cv2
        except ImportError:
            pytest.skip("opencv not installed")

        video_path = tmp_data_root / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (224, 224))
        rng = np.random.default_rng(42)
        for _ in range(30):
            writer.write(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        writer.release()

        item = _make_raw_item(video_path, "video")
        extractor = AutoExtractor(output_dir=tmp_data_root / "extracted")
        result = extractor.extract(item, default_params)
        assert len(result) >= 2
