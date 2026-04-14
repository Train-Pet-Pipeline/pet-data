"""SelfShot data source — ingests videos from local directory with YAML metadata."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import yaml

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import VideoExtractor

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class SelfShotSource(BaseSource):
    """Ingest self-shot videos from a local directory.

    Expects:
    - Videos in params["selfshot_dir"]
    - YAML metadata in params["selfshot_dir"]/meta/<video_stem>.yaml
    - device_model is mandatory for selfshot data.
    """

    source_name = "selfshot"

    def __init__(self, store, params: dict) -> None:
        """Initialize with VideoExtractor."""
        super().__init__(store, params)
        selfshot_dir = Path(params.get("selfshot_dir", ""))
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "selfshot"
        self.extractor = VideoExtractor(output_dir=output_dir)
        self.selfshot_dir = selfshot_dir

    def download(self) -> Iterator[RawItem]:
        """Scan selfshot directory for videos, load metadata from YAML."""
        if not self.selfshot_dir.exists():
            logger.warning("Selfshot directory not found: %s", self.selfshot_dir)
            return

        meta_dir = self.selfshot_dir / "meta"

        for video_path in sorted(self.selfshot_dir.iterdir()):
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            meta_path = meta_dir / f"{video_path.stem}.yaml"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = yaml.safe_load(f) or {}
            else:
                logger.warning("No metadata for %s", video_path.name)
                meta = {}

            yield RawItem(
                source=self.source_name,
                resource_path=video_path,
                resource_type="video",
                metadata=SourceMetadata(
                    species=meta.get("species"),
                    breed=meta.get("breed"),
                    lighting=meta.get("lighting"),
                    bowl_type=meta.get("bowl_type"),
                    device_model=meta.get("device_model"),
                    video_id=video_path.stem,
                ),
            )

    def validate_metadata(self, item: RawItem) -> bool:
        """Selfshot data requires device_model."""
        if not item.metadata.device_model:
            logger.warning(
                "Selfshot item missing device_model: %s", item.resource_path.name
            )
            return False
        return True
