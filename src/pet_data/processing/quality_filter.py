"""Frame quality assessment using Laplacian variance."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality assessment."""

    quality_flag: str  # "normal" / "low" / "failed"
    blur_score: float  # Laplacian variance — lower = blurrier


def assess_quality(image_path: Path, params: dict) -> QualityResult:
    """
    Assess frame quality using Laplacian variance.

    Flags as 'low' if blur_score < threshold. Does NOT delete frames —
    quality_flag is informational for downstream filtering.

    Args:
        image_path: Path to image file.
        params: Must contain frames.quality_blur_threshold.

    Returns:
        QualityResult with quality_flag and blur_score.
    """
    threshold = params["frames"]["quality_blur_threshold"]

    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float64)

    # Laplacian kernel convolution via numpy
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    filtered = convolve2d(arr, laplacian, mode="valid")
    blur_score = float(filtered.var())

    quality_flag = "normal" if blur_score >= threshold else "low"

    logger.info(
        "Quality assessed: %s — blur_score=%.2f flag=%s",
        image_path.name,
        blur_score,
        quality_flag,
    )

    return QualityResult(quality_flag=quality_flag, blur_score=blur_score)
