"""Shared image loading and normalization for weak supervision modules."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def load_and_normalize(path: str) -> torch.Tensor:
    """Load an image, resize to 224x224, normalize to [-1, 1].

    Args:
        path: Absolute path to the image file.

    Returns:
        Tensor of shape (3, 224, 224) normalized to [-1, 1].
    """
    with Image.open(path) as img:
        img: Image.Image = img.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)
    arr = arr / 127.5 - 1.0
    return torch.from_numpy(arr.transpose(2, 0, 1))
