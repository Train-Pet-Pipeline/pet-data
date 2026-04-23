"""Traditional image augmentations using albumentations."""
from __future__ import annotations

import logging
from pathlib import Path

import albumentations as alb
import cv2

logger = logging.getLogger(__name__)


def augment_frame(
    image_path: Path,
    output_dir: Path,
    params: dict,
) -> list[Path]:
    """Apply four traditional augmentation variants to a single image.

    Each variant uses a different albumentations transform controlled by
    ``params["augmentation"]["traditional"]``.

    Args:
        image_path: Path to the source image.
        output_dir: Directory to write augmented images into (created if absent).
        params: Pipeline parameters containing augmentation.traditional settings.

    Returns:
        List of paths to the four augmented images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        msg = f"Failed to read image: {image_path}"
        raise FileNotFoundError(msg)

    trad_params = params["augmentation"]["traditional"]
    brightness_limit = trad_params["brightness_limit"]
    noise_var_limit = trad_params["noise_var_limit"]
    hue_shift_limit = trad_params["hue_shift_limit"]
    sat_shift_limit = trad_params["sat_shift_limit"]
    val_shift_limit = trad_params["val_shift_limit"]
    shift_limit = trad_params["shift_limit"]
    scale_limit = trad_params["scale_limit"]
    rotate_limit = trad_params["rotate_limit"]

    transforms: list[tuple[str, alb.BasicTransform]] = [
        (
            "brightness",
            alb.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=brightness_limit,
                p=1.0,
            ),
        ),
        (
            "noise",
            alb.GaussNoise(
                std_range=(noise_var_limit, noise_var_limit * 5),
                p=1.0,
            ),
        ),
        (
            "hue",
            alb.HueSaturationValue(
                hue_shift_limit=hue_shift_limit,
                sat_shift_limit=sat_shift_limit,
                val_shift_limit=val_shift_limit,
                p=1.0,
            ),
        ),
        (
            "shift",
            alb.ShiftScaleRotate(
                shift_limit=shift_limit,
                scale_limit=scale_limit,
                rotate_limit=rotate_limit,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
        ),
    ]

    stem = image_path.stem
    suffix = image_path.suffix or ".png"
    output_paths: list[Path] = []

    for name, transform in transforms:
        pipeline = alb.Compose([transform])
        augmented = pipeline(image=img)["image"]
        out_path = output_dir / f"{stem}_{name}{suffix}"
        cv2.imwrite(str(out_path), augmented)
        output_paths.append(out_path)
        logger.info(
            '{"event": "traditional_aug", "variant": "%s", "output": "%s"}',
            name,
            out_path,
        )

    return output_paths
