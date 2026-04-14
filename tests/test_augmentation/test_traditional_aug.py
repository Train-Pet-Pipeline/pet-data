"""Tests for traditional augmentation."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from pet_data.augmentation.traditional_aug import augment_frame


class TestAugmentFrame:
    def test_outputs_images(
        self, sample_image: Path, tmp_data_root: Path, default_params: dict
    ) -> None:
        """augment_frame produces at least one output image."""
        output_dir = tmp_data_root / "aug_out"
        results = augment_frame(sample_image, output_dir, default_params)
        assert len(results) >= 1
        for p in results:
            assert p.exists()

    def test_output_size_matches_input(
        self, sample_image: Path, tmp_data_root: Path, default_params: dict
    ) -> None:
        """Augmented images have the same dimensions."""
        output_dir = tmp_data_root / "aug_out"
        results = augment_frame(sample_image, output_dir, default_params)
        original = Image.open(sample_image)
        for p in results:
            aug = Image.open(p)
            assert aug.size == original.size
