"""Tests for quality filter module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from pet_data.processing.quality_filter import QualityResult, assess_quality


@pytest.fixture
def blurry_image(tmp_data_root: Path) -> Path:
    """A heavily blurred image that should fail quality check."""
    img_path = tmp_data_root / "blurry.png"
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=20))
    img.save(img_path)
    return img_path


class TestAssessQuality:
    def test_clear_image_is_normal(self, sample_image: Path, default_params: dict) -> None:
        """A random-pixel image (high-frequency) should be quality=normal."""
        result = assess_quality(sample_image, default_params)
        assert result.quality_flag == "normal"
        assert result.blur_score > default_params["frames"]["quality_blur_threshold"]

    def test_blurry_image_is_low(self, blurry_image: Path, default_params: dict) -> None:
        """A heavily blurred image should be quality=low."""
        result = assess_quality(blurry_image, default_params)
        assert result.quality_flag == "low"
        assert result.blur_score < default_params["frames"]["quality_blur_threshold"]

    def test_returns_quality_result(self, sample_image: Path, default_params: dict) -> None:
        """assess_quality returns a QualityResult dataclass."""
        result = assess_quality(sample_image, default_params)
        assert isinstance(result, QualityResult)
        assert isinstance(result.blur_score, float)
