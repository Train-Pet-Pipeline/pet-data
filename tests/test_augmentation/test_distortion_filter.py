"""Tests for distortion filter."""
from __future__ import annotations

from pathlib import Path

from pet_data.augmentation.distortion_filter import filter_distortion


class TestFilterDistortion:
    def test_model_unavailable_returns_all_ok(
        self, sample_image: Path, default_params: dict
    ) -> None:
        """When YOLO model is not available, all frames pass (degraded mode)."""
        result = filter_distortion([sample_image], default_params)
        assert len(result) == 1
        assert result[0][1] == "ok"

    def test_returns_tuple_of_path_and_status(
        self, sample_image: Path, default_params: dict
    ) -> None:
        """Returns list of (Path, status) tuples."""
        result = filter_distortion([sample_image], default_params)
        path, status = result[0]
        assert isinstance(path, Path)
        assert status in ("ok", "failed")
