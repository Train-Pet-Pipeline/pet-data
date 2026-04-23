"""Local directory source — imports images/videos from a structured local directory.

Expected directory layout:
    {root_dir}/
        cat_images/     ← species inferred from prefix
        dog_images/
        cat_video/
        dog_video/

Each subdirectory name is parsed as ``{species}_{media_type}``.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, cast

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import AutoExtractor

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class LocalDirSource(BaseSource):
    """Ingest images and videos from a local directory tree.

    Configure via ``params["local_dir"]`` — the root directory containing
    species-prefixed subdirectories (e.g. ``cat_images/``, ``dog_video/``).

    An optional ``params["local_dir_limit"]`` caps the number of items
    ingested per subdirectory, useful for quick integration tests.
    """

    ingester_name = "local_dir"

    def __init__(self, store, params: dict) -> None:
        """Initialize with AutoExtractor for mixed image/video content."""
        super().__init__(store, params)
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "local_dir"
        self.extractor = AutoExtractor(output_dir=output_dir)
        self.root_dir = Path(params.get("local_dir", ""))
        self.limit_per_subdir = params.get("local_dir_limit", 0)

    def download(self) -> Iterator[RawItem]:
        """Scan subdirectories and yield RawItems with inferred metadata."""
        if not self.root_dir.exists():
            logger.warning("Local dir not found: %s", self.root_dir)
            return

        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue

            parts = subdir.name.split("_", 1)
            if len(parts) != 2:
                logger.warning("Skipping unrecognized subdir: %s", subdir.name)
                continue

            species = parts[0]

            count = 0
            for file_path in sorted(subdir.iterdir()):
                if not file_path.is_file():
                    continue

                suffix = file_path.suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    resource_type = "image"
                elif suffix in VIDEO_EXTENSIONS:
                    resource_type = "video"
                else:
                    continue

                yield RawItem(
                    source=self.ingester_name,
                    resource_path=file_path,
                    resource_type=cast(Literal["video", "image"], resource_type),
                    metadata=SourceMetadata(
                        species=species,
                        breed=None,
                        lighting="unknown",
                        bowl_type=None,
                        device_model=None,
                        video_id=f"{subdir.name}_{file_path.stem}",
                    ),
                )

                count += 1
                if self.limit_per_subdir and count >= self.limit_per_subdir:
                    logger.info(
                        "Reached limit %d for subdir %s",
                        self.limit_per_subdir, subdir.name,
                    )
                    break

    def validate_metadata(self, item: RawItem) -> bool:
        """Local dir items require species."""
        if not item.metadata.species:
            logger.warning("Missing species for: %s", item.resource_path.name)
            return False
        return True
