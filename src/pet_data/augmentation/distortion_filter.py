"""Distortion filter using YOLO-based detection with graceful degradation."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def filter_distortion(
    frame_paths: list[Path],
    params: dict,
) -> list[tuple[Path, str]]:
    """Filter frames by checking for visual distortions using a YOLO model.

    When the YOLO model is available, each frame is run through detection and
    the confidence is compared against the configured threshold.  When the model
    is not available (no path set or file missing), the function operates in
    degraded mode and marks every frame as ``"ok"``.

    Args:
        frame_paths: List of image file paths to check.
        params: Pipeline parameters containing
            ``augmentation.distortion_conf_threshold``.

    Returns:
        List of ``(path, status)`` tuples where status is ``"ok"`` or
        ``"failed"``.
    """
    model_path = os.environ.get("YOLO_MODEL_PATH") or params.get("yolo_model_path")
    conf_threshold = params["augmentation"]["distortion_conf_threshold"]

    if model_path and Path(model_path).exists():
        return _filter_with_model(frame_paths, model_path, conf_threshold)

    logger.warning(
        '{"event": "distortion_filter_degraded", "reason": "model_not_found"}'
    )
    return [(p, "ok") for p in frame_paths]


def _filter_with_model(
    frame_paths: list[Path],
    model_path: str,
    conf_threshold: float,
) -> list[tuple[Path, str]]:
    """Run YOLO detection and filter by confidence threshold.

    Args:
        frame_paths: Image paths to evaluate.
        model_path: Filesystem path to the YOLO model weights.
        conf_threshold: Minimum detection confidence to pass.

    Returns:
        List of ``(path, status)`` tuples.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model = YOLO(model_path)
    results: list[tuple[Path, str]] = []
    for path in frame_paths:
        detections = model(str(path), verbose=False)
        if detections and len(detections[0].boxes) > 0:
            max_conf = float(detections[0].boxes.conf.max())
            status = "ok" if max_conf >= conf_threshold else "failed"
        else:
            status = "failed"
        results.append((path, status))
    return results
