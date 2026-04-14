"""Tests for pHash dedup module."""
from __future__ import annotations

from pathlib import Path

from pet_data.processing.dedup import compute_phash, dedup_check, hamming_distance


class TestComputePhash:
    def test_returns_bytes(self, sample_image: Path) -> None:
        """compute_phash returns 8-byte hash."""
        result = compute_phash(sample_image)
        assert isinstance(result, bytes)
        assert len(result) == 8

    def test_same_image_same_hash(self, sample_image: Path, sample_image_duplicate: Path) -> None:
        """Identical images produce identical phash."""
        h1 = compute_phash(sample_image)
        h2 = compute_phash(sample_image_duplicate)
        assert h1 == h2

    def test_different_image_different_hash(
        self, sample_image: Path, sample_image_different: Path
    ) -> None:
        """Completely different images produce different phash."""
        h1 = compute_phash(sample_image)
        h2 = compute_phash(sample_image_different)
        assert h1 != h2


class TestHammingDistance:
    def test_identical_hashes(self) -> None:
        """Identical hashes have distance 0."""
        h = b"\xff\x00\xff\x00\xff\x00\xff\x00"
        assert hamming_distance(h, h) == 0

    def test_one_bit_difference(self) -> None:
        """One bit flip = distance 1."""
        h1 = b"\x00\x00\x00\x00\x00\x00\x00\x00"
        h2 = b"\x01\x00\x00\x00\x00\x00\x00\x00"
        assert hamming_distance(h1, h2) == 1

    def test_all_bits_different(self) -> None:
        """All bits different = distance 64."""
        h1 = b"\x00" * 8
        h2 = b"\xff" * 8
        assert hamming_distance(h1, h2) == 64


class TestDedupCheck:
    def test_no_existing_hashes_not_duplicate(
        self, sample_image: Path, default_params: dict
    ) -> None:
        """Empty DB means nothing is a duplicate."""
        result = dedup_check(sample_image, {}, default_params)
        assert result.is_duplicate is False
        assert result.phash is not None

    def test_identical_image_is_duplicate(
        self, sample_image: Path, sample_image_duplicate: Path, default_params: dict
    ) -> None:
        """Same image detected as duplicate."""
        first = dedup_check(sample_image, {}, default_params)
        existing = {"frame-001": first.phash}
        second = dedup_check(sample_image_duplicate, existing, default_params)
        assert second.is_duplicate is True
        assert second.duplicate_of == "frame-001"

    def test_different_image_not_duplicate(
        self, sample_image: Path, sample_image_different: Path, default_params: dict
    ) -> None:
        """Different image not detected as duplicate."""
        first = dedup_check(sample_image, {}, default_params)
        existing = {"frame-001": first.phash}
        second = dedup_check(sample_image_different, existing, default_params)
        assert second.is_duplicate is False
