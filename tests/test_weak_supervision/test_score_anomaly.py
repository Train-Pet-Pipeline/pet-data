"""Tests for anomaly scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pet_data.storage.store import FrameRecord, FrameStore
from pet_data.weak_supervision.score_anomaly import ScoreReport, score_frames
from pet_data.weak_supervision.train_autoencoder import FeedingAutoencoder


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """Save a fresh (untrained) autoencoder for testing."""
    model = FeedingAutoencoder()
    path = tmp_path / "autoencoder.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def store_with_frames(tmp_path: Path) -> tuple[FrameStore, Path]:
    """Store with frames that have images but no anomaly scores."""
    store = FrameStore(tmp_path / "test.db")
    data_dir = tmp_path / "frames"
    data_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(5):
        img_path = data_dir / f"{i}.png"
        arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)
        store.insert_frame(
            FrameRecord(
                frame_id=f"f-{i}", video_id="v", source="s",
                frame_path=str(img_path), data_root=str(tmp_path),
                quality_flag="normal",
            )
        )
    return store, tmp_path


class TestScoreFrames:
    def test_scores_all_unscored_frames(
        self, store_with_frames: tuple, model_path: Path, default_params: dict
    ) -> None:
        """All frames with anomaly_score=None get scored."""
        store, _ = store_with_frames
        report = score_frames(store, model_path, default_params)
        assert isinstance(report, ScoreReport)
        assert report.total_scored == 5

    def test_already_scored_frames_skipped(
        self, store_with_frames: tuple, model_path: Path, default_params: dict
    ) -> None:
        """Frames that already have anomaly_score are not re-scored."""
        store, _ = store_with_frames
        store.update_anomaly("f-0", is_candidate=False, score=0.01)
        report = score_frames(store, model_path, default_params)
        assert report.total_scored == 4

    def test_report_has_threshold(
        self, store_with_frames: tuple, model_path: Path, default_params: dict
    ) -> None:
        """Report includes the threshold used."""
        store, _ = store_with_frames
        report = score_frames(store, model_path, default_params)
        assert report.threshold == default_params["weak_supervision"]["anomaly_score_threshold"]
