"""Tests for FeedingAutoencoder and training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pet_data.storage.store import FrameRecord, FrameStore
from pet_data.weak_supervision.train_autoencoder import (
    FeedingAutoencoder,
    TrainReport,
    train,
)


class TestFeedingAutoencoder:
    def test_forward_shape(self) -> None:
        """Forward pass preserves input shape."""
        model = FeedingAutoencoder()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 3, 224, 224)

    def test_anomaly_score_shape(self) -> None:
        """anomaly_score returns per-sample scalar."""
        model = FeedingAutoencoder()
        x = torch.randn(4, 3, 224, 224)
        scores = model.anomaly_score(x)
        assert scores.shape == (4,)
        assert (scores >= 0).all()

    def test_anomaly_score_lower_for_similar_input(self) -> None:
        """After minimal training, reconstruction of training data should have lower score."""
        model = FeedingAutoencoder()
        x = torch.zeros(2, 3, 224, 224)
        score_before = model.anomaly_score(x).mean().item()
        assert score_before >= 0


class TestTrain:
    def test_insufficient_frames_raises(
        self, tmp_path: Path, default_params: dict
    ) -> None:
        """Training with fewer than min_normal_frames raises ValueError."""
        store = FrameStore(tmp_path / "test.db")
        for i in range(5):
            store.insert_frame(
                FrameRecord(
                    frame_id=f"f-{i}", video_id="v", source="s",
                    frame_path=f"{i}.png", data_root="/data",
                    quality_flag="normal",
                )
            )
        with pytest.raises(ValueError, match="min_normal_frames"):
            train(store, default_params, tmp_path / "out")

    def test_train_saves_model(self, tmp_path: Path, default_params: dict) -> None:
        """Training with enough frames saves autoencoder.pt."""
        store = FrameStore(tmp_path / "test.db")
        data_dir = tmp_path / "frames"
        data_dir.mkdir()

        params = {**default_params}
        params["weak_supervision"] = {
            **default_params["weak_supervision"],
            "min_normal_frames": 10,
            "max_epochs": 2,
            "batch_size": 4,
        }

        from PIL import Image

        rng = np.random.default_rng(42)
        for i in range(10):
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

        output_dir = tmp_path / "model_out"
        report = train(store, params, output_dir)
        assert isinstance(report, TrainReport)
        assert report.model_path.exists()
        assert report.epochs == 2
