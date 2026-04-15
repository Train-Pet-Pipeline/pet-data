"""Hospital veterinary data source — PII scrubbing at ingest time."""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, cast

import yaml
from PIL import Image

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import AutoExtractor

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),      # phone numbers
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # email
    re.compile(r"\b\d{6,}\b"),                             # long numeric IDs (patient IDs)
]


def scrub_pii_from_string(text: str) -> str:
    """Replace PII patterns in a string with '[REDACTED]'."""
    for pattern in PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def scrub_exif(image_path: Path) -> None:
    """Remove all EXIF data from an image file (in-place)."""
    try:
        img = Image.open(image_path)
        if hasattr(img, "info") and "exif" in img.info:
            data = list(img.getdata())
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(data)
            clean_img.save(image_path)
            logger.info("Scrubbed EXIF from %s", image_path.name)
    except Exception:
        logger.exception("Failed to scrub EXIF: %s", image_path)


def sanitize_filename(original_path: Path) -> str:
    """Generate an anonymous filename from hash of original path."""
    h = hashlib.sha256(str(original_path).encode()).hexdigest()[:16]
    return f"hospital_{h}{original_path.suffix.lower()}"


class HospitalSource(BaseSource):
    """Ingest veterinary hospital data with mandatory PII scrubbing.

    CRITICAL: All PII scrubbing happens at ingest time, not query time.
    """

    source_name = "hospital"

    def __init__(self, store, params: dict) -> None:
        """Initialize with AutoExtractor."""
        super().__init__(store, params)
        hospital_dir = Path(params.get("hospital_dir", ""))
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "hospital"
        self.extractor = AutoExtractor(output_dir=output_dir)
        self.hospital_dir = hospital_dir

    def download(self) -> Iterator[RawItem]:
        """Scan hospital directory, scrub PII, yield sanitized RawItems."""
        if not self.hospital_dir.exists():
            logger.warning("Hospital directory not found: %s", self.hospital_dir)
            return

        meta_dir = self.hospital_dir / "meta"

        for file_path in sorted(self.hospital_dir.iterdir()):
            suffix = file_path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                resource_type = "image"
                scrub_exif(file_path)
            elif suffix in VIDEO_EXTENSIONS:
                resource_type = "video"
            else:
                continue

            meta_path = meta_dir / f"{file_path.stem}.yaml"
            if meta_path.exists():
                with open(meta_path) as f:
                    raw_meta = yaml.safe_load(f) or {}
                meta = {
                    k: scrub_pii_from_string(str(v)) if isinstance(v, str) else v
                    for k, v in raw_meta.items()
                }
            else:
                meta = {}

            safe_vid = sanitize_filename(file_path)

            yield RawItem(
                source=self.source_name,
                resource_path=file_path,
                resource_type=cast(Literal["video", "image"], resource_type),
                metadata=SourceMetadata(
                    species=meta.get("species"),
                    breed=meta.get("breed"),
                    lighting=meta.get("lighting", "unknown"),
                    bowl_type=None,
                    device_model=None,
                    video_id=safe_vid,
                ),
            )

    def validate_metadata(self, item: RawItem) -> bool:
        """Hospital data requires species."""
        if not item.metadata.species:
            logger.warning("Missing species for hospital item: %s", item.metadata.video_id)
            return False
        return True
