"""Anomaly scoring for frames using a trained FeedingAutoencoder."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pet_data.storage.store import FrameFilter, FrameStore
from pet_data.weak_supervision.train_autoencoder import FeedingAutoencoder

logger = logging.getLogger(__name__)


@dataclass
class ScoreReport:
    """Summary of an anomaly scoring run.

    Attributes:
        total_scored: Number of frames that were scored in this run.
        anomalies_found: Number of frames flagged as anomaly candidates.
        threshold: The anomaly score threshold used for flagging.
    """

    total_scored: int
    anomalies_found: int
    threshold: float


def _load_image_tensor(frame_path: str) -> torch.Tensor:
    """Load an image and normalize to [-1, 1] as a (1, 3, 224, 224) tensor.

    Args:
        frame_path: Absolute path to the image file.

    Returns:
        Tensor of shape (1, 3, 224, 224) normalized to [-1, 1].
    """
    img = Image.open(frame_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def score_frames(store: FrameStore, model_path: Path, params: dict) -> ScoreReport:
    """Score all unscored frames with the trained autoencoder.

    Loads the model from model_path, queries the store for frames with
    anomaly_score IS NULL, and writes back scores and candidate flags.

    Args:
        store: FrameStore to query and update.
        model_path: Path to a saved FeedingAutoencoder state dict.
        params: Full params dict (reads from params["weak_supervision"]).

    Returns:
        ScoreReport with total frames scored, anomalies found, and threshold.
    """
    threshold: float = params["weak_supervision"]["anomaly_score_threshold"]

    model = FeedingAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    logger.info('{"event": "score_frames_start", "model_path": "%s"}', model_path)

    # FrameFilter doesn't support anomaly_score IS NULL, so fetch all and filter in Python
    all_records = store.query_frames(FrameFilter(limit=10_000_000))
    unscored = [r for r in all_records if r.anomaly_score is None]

    logger.info('{"event": "score_frames_unscored", "count": %d}', len(unscored))

    total_scored = 0
    anomalies_found = 0

    with torch.no_grad():
        for record in unscored:
            tensor = _load_image_tensor(record.frame_path)
            score_val = model.anomaly_score(tensor).item()
            is_candidate = score_val > threshold
            store.update_anomaly(record.frame_id, is_candidate=is_candidate, score=score_val)
            total_scored += 1
            if is_candidate:
                anomalies_found += 1

    logger.info(
        '{"event": "score_frames_done", "total_scored": %d, "anomalies_found": %d,'
        ' "threshold": %f}',
        total_scored,
        anomalies_found,
        threshold,
    )

    return ScoreReport(
        total_scored=total_scored,
        anomalies_found=anomalies_found,
        threshold=threshold,
    )
