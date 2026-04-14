"""Tests for BaseSource contract."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import ImageExtractor
from pet_data.storage.store import FrameStore


class DummySource(BaseSource):
    """Minimal BaseSource implementation for testing."""

    source_name = "dummy"

    def __init__(
        self,
        store: FrameStore,
        params: dict,
        items: list[RawItem] | None = None,
        fail_validation: bool = False,
    ) -> None:
        """Initialize with optional items and validation behavior."""
        super().__init__(store, params)
        self._items = items or []
        self._fail_validation = fail_validation

    def download(self) -> Iterator[RawItem]:
        """Yield pre-configured items."""
        yield from self._items

    def validate_metadata(self, item: RawItem) -> bool:
        """Return False if fail_validation was set."""
        return not self._fail_validation


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore for testing."""
    return FrameStore(tmp_path / "test.db")


class TestIngestPipeline:
    """Tests for the BaseSource.ingest() template method."""

    def test_empty_download_returns_zero_report(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path
    ) -> None:
        """Empty download iterator produces zero-count report."""
        src = DummySource(store, default_params, items=[])
        src.extractor = ImageExtractor(output_dir=tmp_data_root / "out")
        report = src.ingest()
        assert report.inserted == 0
        assert report.skipped == 0
        assert report.duplicates == 0

    def test_valid_item_gets_inserted(
        self,
        store: FrameStore,
        default_params: dict,
        sample_image: Path,
        tmp_data_root: Path,
    ) -> None:
        """A valid item passes through the full pipeline and gets inserted."""
        item = RawItem(
            source="dummy",
            resource_path=sample_image,
            resource_type="image",
            metadata=SourceMetadata(
                species="cat", breed="tabby", lighting="bright",
                bowl_type=None, device_model=None, video_id="vid-001",
            ),
        )
        src = DummySource(store, default_params, items=[item])
        src.extractor = ImageExtractor(output_dir=tmp_data_root / "out")
        report = src.ingest()
        assert report.inserted == 1
        assert store.count_by_source()["dummy"] == 1

    def test_failed_validation_skips_item(
        self,
        store: FrameStore,
        default_params: dict,
        sample_image: Path,
        tmp_data_root: Path,
    ) -> None:
        """Item failing validate_metadata is skipped."""
        item = RawItem(
            source="dummy",
            resource_path=sample_image,
            resource_type="image",
            metadata=SourceMetadata(
                species=None, breed=None, lighting=None,
                bowl_type=None, device_model=None, video_id="vid-001",
            ),
        )
        src = DummySource(store, default_params, items=[item], fail_validation=True)
        src.extractor = ImageExtractor(output_dir=tmp_data_root / "out")
        report = src.ingest()
        assert report.skipped == 1
        assert report.inserted == 0

    def test_duplicate_frame_counted(
        self,
        store: FrameStore,
        default_params: dict,
        sample_image: Path,
        sample_image_duplicate: Path,
        tmp_data_root: Path,
    ) -> None:
        """Duplicate frames are detected and counted."""
        items = [
            RawItem(
                source="dummy",
                resource_path=sample_image,
                resource_type="image",
                metadata=SourceMetadata(
                    species="cat", breed=None, lighting=None,
                    bowl_type=None, device_model=None, video_id="vid-001",
                ),
            ),
            RawItem(
                source="dummy",
                resource_path=sample_image_duplicate,
                resource_type="image",
                metadata=SourceMetadata(
                    species="cat", breed=None, lighting=None,
                    bowl_type=None, device_model=None, video_id="vid-002",
                ),
            ),
        ]
        src = DummySource(store, default_params, items=items)
        src.extractor = ImageExtractor(output_dir=tmp_data_root / "out")
        report = src.ingest()
        assert report.inserted == 1
        assert report.duplicates == 1
