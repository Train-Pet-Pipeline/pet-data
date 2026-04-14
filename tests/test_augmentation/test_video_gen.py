"""Tests for video generation module."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pet_data.augmentation.video_gen import (
    AugmentReport,
    NullGenerator,
    Wan21Generator,
    run_augmentation,
)
from pet_data.storage.store import FrameStore


class TestNullGenerator:
    def test_returns_none(self, tmp_path: Path) -> None:
        """NullGenerator always returns None."""
        gen = NullGenerator()
        result = gen.generate(tmp_path / "seed.png", "a cat eating", seed=42)
        assert result is None


class TestWan21Generator:
    def test_generate_calls_api(self, sample_image: Path) -> None:
        """Wan21Generator calls the configured endpoint."""
        gen = Wan21Generator(endpoint="http://fake:8080/generate", timeout=10)
        with patch("pet_data.augmentation.video_gen.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b"fake_video_data"
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp
            gen.generate(sample_image, "a cat eating dry food", seed=42)
            mock_post.assert_called_once()

    def test_generate_returns_none_on_failure(self, sample_image: Path) -> None:
        """Wan21Generator returns None after retries exhausted."""
        gen = Wan21Generator(endpoint="http://fake:8080/generate", timeout=1)
        with patch("pet_data.augmentation.video_gen.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError("refused")
            result = gen.generate(sample_image, "a cat", seed=42)
            assert result is None


class TestRunAugmentation:
    def test_with_null_generator(self, tmp_path: Path, default_params: dict) -> None:
        """run_augmentation with NullGenerator produces zero augmented frames."""
        store = FrameStore(tmp_path / "test.db")
        report = run_augmentation(store, default_params, generator=NullGenerator())
        assert isinstance(report, AugmentReport)
        assert report.generated == 0
