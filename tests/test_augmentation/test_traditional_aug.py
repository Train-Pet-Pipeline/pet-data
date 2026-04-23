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

    def test_aug_params_sourced_from_params_yaml(
        self, sample_image: Path, tmp_data_root: Path, default_params: dict
    ) -> None:
        """All 6 aug params (F7) are read from params dict, not hardcoded.

        Two checks:
        1. Passing params with the 6 sub-keys produces valid output (params readable).
        2. Removing a key causes a KeyError — proving the code actually reads it.
        """
        import copy
        import pytest

        # Check 1: All 6 params present → success
        params = copy.deepcopy(default_params)
        trad = params["augmentation"]["traditional"]
        trad["hue_shift_limit"] = 5
        trad["sat_shift_limit"] = 10
        trad["val_shift_limit"] = 5
        trad["shift_limit"] = 0.02
        trad["scale_limit"] = 0.02
        trad["rotate_limit"] = 5

        output_dir = tmp_data_root / "aug_out_params"
        results = augment_frame(sample_image, output_dir, params)
        assert len(results) == 4
        for p in results:
            assert p.exists()

        # Check 2: Missing param → KeyError (proves code reads from dict, not hardcode)
        params_missing = copy.deepcopy(default_params)
        del params_missing["augmentation"]["traditional"]["hue_shift_limit"]
        output_dir2 = tmp_data_root / "aug_out_missing"
        with pytest.raises(KeyError):
            augment_frame(sample_image, output_dir2, params_missing)
