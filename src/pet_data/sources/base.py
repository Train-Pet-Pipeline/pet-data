"""Base classes for data sources."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

from pet_schema.enums import SourceType

from pet_data.processing.dedup import dedup_check
from pet_data.processing.quality_filter import assess_quality

if TYPE_CHECKING:
    from pet_data.sources.extractors import FrameExtractor
    from pet_data.storage.store import FrameStore

logger = logging.getLogger(__name__)


@dataclass
class SourceMetadata:
    """Metadata attached to each raw data item."""

    species: str | None
    breed: str | None
    lighting: str | None
    bowl_type: str | None
    device_model: str | None
    video_id: str


@dataclass
class RawItem:
    """A single raw resource (video or image) from a data source."""

    source: str
    resource_path: Path
    resource_type: Literal["video", "image"]
    metadata: SourceMetadata


@dataclass
class IngestReport:
    """Summary of an ingest run."""

    inserted: int = 0
    skipped: int = 0
    duplicates: int = 0
    errors: int = 0


class BaseSource(ABC):
    """Abstract base for all data sources.

    Subclasses must:
    - Set ingester_name (str) and extractor (FrameExtractor) as class attributes
    - Implement download() and validate_metadata()

    The ingest() template method handles the full pipeline:
    download → extract → dedup → quality → store.

    Concept separation (Phase 3):
    - ingester_name: which code produced this sample (implementation identity)
    - default_provenance: legal/compliance category (declared in subclasses)
    """

    ingester_name: str
    default_provenance: ClassVar[SourceType]
    extractor: FrameExtractor

    def __init__(self, store: FrameStore, params: dict) -> None:
        """Initialize with a FrameStore and params dict."""
        self.store = store
        self.params = params

    def ingest(self) -> IngestReport:
        """Run the full ingest pipeline. Subclasses should not override."""
        import uuid

        from pet_data.storage.store import FrameRecord

        report = IngestReport()
        existing_phashes = self.store.get_phashes()

        for item in self.download():
            if not self.validate_metadata(item):
                logger.warning("Metadata validation failed: %s", item.resource_path)
                report.skipped += 1
                continue

            try:
                frames = self.extractor.extract(item, self.params)
            except Exception:
                logger.exception("Extract failed: %s", item.resource_path)
                report.errors += 1
                continue

            for frame_path in frames:
                try:
                    dedup_result = dedup_check(frame_path, existing_phashes, self.params)
                    if dedup_result.is_duplicate:
                        report.duplicates += 1
                        continue

                    quality = assess_quality(frame_path, self.params)
                    frame_id = str(uuid.uuid4())
                    data_root = self.params.get("data_root", "")

                    rel_path = (
                        str(frame_path.relative_to(Path(data_root)))
                        if data_root and frame_path.is_relative_to(Path(data_root))
                        else str(frame_path)
                    )
                    storage_uri = (
                        f"local://{data_root}/{rel_path}" if data_root else f"local://{rel_path}"
                    )

                    record = FrameRecord(
                        frame_id=frame_id,
                        video_id=item.metadata.video_id,
                        source=self.ingester_name,
                        frame_path=rel_path,
                        data_root=data_root,
                        species=item.metadata.species,
                        breed=item.metadata.breed,
                        lighting=item.metadata.lighting,
                        bowl_type=item.metadata.bowl_type,
                        quality_flag=quality.quality_flag,
                        blur_score=quality.blur_score,
                        phash=dedup_result.phash,
                        modality="vision",
                        storage_uri=storage_uri,
                        frame_width=quality.width,
                        frame_height=quality.height,
                        brightness_score=quality.brightness_score,
                    )

                    self.store.insert_frame(record)
                    existing_phashes[frame_id] = dedup_result.phash
                    report.inserted += 1
                except Exception:
                    logger.exception("Failed to process frame: %s", frame_path)
                    report.errors += 1

        return report

    @abstractmethod
    def download(self) -> Iterator[RawItem]:
        """Yield raw items from the data source."""

    @abstractmethod
    def validate_metadata(self, item: RawItem) -> bool:
        """Validate that the item has required metadata fields."""
