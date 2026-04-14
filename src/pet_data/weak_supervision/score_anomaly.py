"""Anomaly scoring for frames using a trained FeedingAutoencoder."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from pet_data.storage.store import FrameStore
from pet_data.weak_supervision._image_util import load_and_normalize
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

    unscored = store.query_unscored_frames()

    logger.info('{"event": "score_frames_unscored", "count": %d}', len(unscored))

    total_scored = 0
    anomalies_found = 0

    with torch.no_grad():
        for record in unscored:
            try:
                tensor = load_and_normalize(record.frame_path).unsqueeze(0)
            except (OSError, SyntaxError) as exc:
                logger.warning(
                    '{"event": "score_frame_skip", "frame_id": "%s", "error": "%s"}',
                    record.frame_id,
                    exc,
                )
                continue
            score_val = model.anomaly_score(tensor).item()
            is_candidate = score_val > threshold
            store.update_anomaly(
                record.frame_id, is_candidate=is_candidate, score=score_val
            )
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
