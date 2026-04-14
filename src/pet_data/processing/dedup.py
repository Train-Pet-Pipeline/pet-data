"""pHash-based frame deduplication."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DedupResult:
    """Result of a dedup check."""

    is_duplicate: bool
    phash: bytes
    duplicate_of: str | None = None


def compute_phash(image_path: Path) -> bytes:
    """Compute perceptual hash of an image, return 8-byte binary."""
    img = Image.open(image_path)
    h = imagehash.phash(img)
    # h.hash is an 8x8 bool ndarray; packbits converts to 8 bytes
    return np.packbits(h.hash.flatten()).tobytes()


def hamming_distance(hash_a: bytes, hash_b: bytes) -> int:
    """Compute hamming distance between two phash byte strings."""
    xor = int.from_bytes(hash_a, "big") ^ int.from_bytes(hash_b, "big")
    return bin(xor).count("1")


def dedup_check(
    image_path: Path,
    existing_phashes: dict[str, bytes],
    params: dict,
) -> DedupResult:
    """
    Check if image is a duplicate of any existing frame.

    Args:
        image_path: Path to the image to check.
        existing_phashes: {frame_id: phash_bytes} mapping loaded from store.
        params: Must contain frames.dedup_hamming_threshold.

    Returns:
        DedupResult with is_duplicate flag and computed phash.
    """
    threshold = params["frames"]["dedup_hamming_threshold"]
    phash = compute_phash(image_path)

    for frame_id, existing_hash in existing_phashes.items():
        if hamming_distance(phash, existing_hash) < threshold:
            logger.info(
                "Duplicate detected: %s matches %s", image_path.name, frame_id
            )
            return DedupResult(is_duplicate=True, phash=phash, duplicate_of=frame_id)

    return DedupResult(is_duplicate=False, phash=phash)
