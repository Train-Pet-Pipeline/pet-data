"""Tests for FrameStore CRUD operations."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from pet_data.storage.store import FrameFilter, FrameRecord, FrameStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Return a FrameStore backed by an in-memory SQLite database."""
    return FrameStore(Path(":memory:"))


def _make_record(frame_id: str = "fr_001", source: str = "youtube") -> FrameRecord:
    """Build a minimal FrameRecord for use in tests."""
    return FrameRecord(
        frame_id=frame_id,
        video_id="vid_001",
        source=source,
        frame_path=f"frames/{frame_id}.jpg",
        data_root="/data",
    )


# ---------------------------------------------------------------------------
# TestInsertAndGet
# ---------------------------------------------------------------------------


class TestInsertAndGet:
    """Tests for insert_frame and get_frame."""

    def test_insert_and_get_frame(self, store: FrameStore) -> None:
        """Inserting a frame and retrieving it by ID returns the same record."""
        record = _make_record()
        store.insert_frame(record)
        retrieved = store.get_frame("fr_001")
        assert retrieved is not None
        assert retrieved.frame_id == "fr_001"
        assert retrieved.video_id == "vid_001"
        assert retrieved.source == "youtube"
        assert retrieved.quality_flag == "normal"
        assert retrieved.annotation_status == "pending"
        assert retrieved.is_anomaly_candidate is False

    def test_get_nonexistent_returns_none(self, store: FrameStore) -> None:
        """get_frame returns None when the frame_id does not exist."""
        assert store.get_frame("nonexistent") is None

    def test_insert_duplicate_id_raises(self, store: FrameStore) -> None:
        """Inserting a record with a duplicate frame_id raises IntegrityError."""
        record = _make_record()
        store.insert_frame(record)
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_frame(record)


# ---------------------------------------------------------------------------
# TestBulkInsert
# ---------------------------------------------------------------------------


class TestBulkInsert:
    """Tests for bulk_insert_frames."""

    def test_bulk_insert_success(self, store: FrameStore) -> None:
        """Bulk-inserting 5 records returns 5 and all rows are queryable."""
        records = [_make_record(f"fr_{i:03d}") for i in range(5)]
        count = store.bulk_insert_frames(records)
        assert count == 5
        for i in range(5):
            assert store.get_frame(f"fr_{i:03d}") is not None

    def test_bulk_insert_rollback_on_duplicate(self, store: FrameStore) -> None:
        """A duplicate within a bulk batch causes a full rollback."""
        records = [_make_record(f"fr_{i:03d}") for i in range(3)]
        store.bulk_insert_frames(records)

        # Batch containing one record that already exists → must roll back entirely
        new_batch = [_make_record("fr_010"), _make_record("fr_000")]  # fr_000 is duplicate
        with pytest.raises(sqlite3.IntegrityError):
            store.bulk_insert_frames(new_batch)

        # fr_010 must NOT have been committed (full rollback)
        assert store.get_frame("fr_010") is None


# ---------------------------------------------------------------------------
# TestQueryFrames
# ---------------------------------------------------------------------------


class TestQueryFrames:
    """Tests for query_frames with various filter combinations."""

    def test_filter_by_source(self, store: FrameStore) -> None:
        """Filtering by source returns only frames from that source."""
        store.insert_frame(_make_record("fr_001", source="youtube"))
        store.insert_frame(_make_record("fr_002", source="community"))
        results = store.query_frames(FrameFilter(source="youtube"))
        assert len(results) == 1
        assert results[0].frame_id == "fr_001"

    def test_filter_by_annotation_status(self, store: FrameStore) -> None:
        """Filtering by annotation_status returns matching frames only."""
        store.insert_frame(_make_record("fr_001"))
        store.insert_frame(_make_record("fr_002"))
        store.update_annotation_status("fr_002", "approved")
        results = store.query_frames(FrameFilter(annotation_status="approved"))
        assert len(results) == 1
        assert results[0].frame_id == "fr_002"

    def test_limit_and_offset(self, store: FrameStore) -> None:
        """limit and offset correctly page through results."""
        records = [_make_record(f"fr_{i:03d}") for i in range(10)]
        store.bulk_insert_frames(records)

        page1 = store.query_frames(FrameFilter(limit=3, offset=0))
        page2 = store.query_frames(FrameFilter(limit=3, offset=3))
        assert len(page1) == 3
        assert len(page2) == 3
        ids1 = {r.frame_id for r in page1}
        ids2 = {r.frame_id for r in page2}
        assert ids1.isdisjoint(ids2)


# ---------------------------------------------------------------------------
# TestUpdateMethods
# ---------------------------------------------------------------------------


class TestUpdateMethods:
    """Tests for individual update_* methods."""

    def test_update_quality(self, store: FrameStore) -> None:
        """update_quality persists quality_flag and blur_score."""
        store.insert_frame(_make_record())
        store.update_quality("fr_001", "low", 12.5)
        record = store.get_frame("fr_001")
        assert record is not None
        assert record.quality_flag == "low"
        assert record.blur_score == pytest.approx(12.5)

    def test_update_anomaly(self, store: FrameStore) -> None:
        """update_anomaly persists is_anomaly_candidate and anomaly_score."""
        store.insert_frame(_make_record())
        store.update_anomaly("fr_001", True, 0.95)
        record = store.get_frame("fr_001")
        assert record is not None
        assert record.is_anomaly_candidate is True
        assert record.anomaly_score == pytest.approx(0.95)

    def test_update_annotation_status(self, store: FrameStore) -> None:
        """update_annotation_status persists the new status."""
        store.insert_frame(_make_record())
        store.update_annotation_status("fr_001", "approved")
        record = store.get_frame("fr_001")
        assert record is not None
        assert record.annotation_status == "approved"

    def test_update_augmentation(self, store: FrameStore) -> None:
        """update_augmentation persists aug_quality and parent_frame_id."""
        store.insert_frame(_make_record("fr_parent"))
        store.insert_frame(_make_record("fr_aug"))
        store.update_augmentation("fr_aug", "ok", "fr_parent")
        record = store.get_frame("fr_aug")
        assert record is not None
        assert record.aug_quality == "ok"
        assert record.parent_frame_id == "fr_parent"


# ---------------------------------------------------------------------------
# TestGetPhashes
# ---------------------------------------------------------------------------


class TestGetPhashes:
    """Tests for get_phashes."""

    def test_get_phashes_returns_mapping(self, store: FrameStore) -> None:
        """get_phashes returns only frames that have a non-NULL phash."""
        store.insert_frame(_make_record("fr_001"))
        store.insert_frame(_make_record("fr_002"))
        store.update_phash("fr_001", b"\xde\xad\xbe\xef")
        # fr_002 has no phash

        result = store.get_phashes()
        assert "fr_001" in result
        assert result["fr_001"] == b"\xde\xad\xbe\xef"
        assert "fr_002" not in result

    def test_get_phashes_filter_by_source(self, store: FrameStore) -> None:
        """get_phashes with source argument filters to that source only."""
        store.insert_frame(_make_record("fr_yt", source="youtube"))
        store.insert_frame(_make_record("fr_cm", source="community"))
        store.update_phash("fr_yt", b"\x01\x02")
        store.update_phash("fr_cm", b"\x03\x04")

        result = store.get_phashes(source="youtube")
        assert "fr_yt" in result
        assert "fr_cm" not in result


# ---------------------------------------------------------------------------
# TestStatistics
# ---------------------------------------------------------------------------


class TestStatistics:
    """Tests for aggregate count methods."""

    def test_count_by_source(self, store: FrameStore) -> None:
        """count_by_source returns correct per-source counts."""
        store.insert_frame(_make_record("fr_001", source="youtube"))
        store.insert_frame(_make_record("fr_002", source="youtube"))
        store.insert_frame(_make_record("fr_003", source="community"))
        counts = store.count_by_source()
        assert counts["youtube"] == 2
        assert counts["community"] == 1

    def test_count_by_status(self, store: FrameStore) -> None:
        """count_by_status returns correct per-status counts."""
        store.insert_frame(_make_record("fr_001"))
        store.insert_frame(_make_record("fr_002"))
        store.update_annotation_status("fr_001", "approved")
        counts = store.count_by_status()
        assert counts["pending"] == 1
        assert counts["approved"] == 1

    def test_count_normal_frames(self, store: FrameStore) -> None:
        """count_normal_frames excludes low-quality and anomaly-candidate frames."""
        store.insert_frame(_make_record("fr_001"))  # normal + not anomaly → counted
        store.insert_frame(_make_record("fr_002"))  # normal + anomaly     → excluded
        store.insert_frame(_make_record("fr_003"))  # low quality          → excluded
        store.update_anomaly("fr_002", is_candidate=True, score=0.9)
        store.update_quality("fr_003", "low", 5.0)
        assert store.count_normal_frames() == 1
