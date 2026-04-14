"""Tests for the 001_init migration."""
from __future__ import annotations

from pathlib import Path

from pet_data.storage.store import FrameStore


class TestMigration001:
    """Tests that verify the 001_init migration produces the expected schema."""

    def test_fresh_db_has_frames_table(self, tmp_path: Path) -> None:
        """A freshly initialised FrameStore contains all 20 expected columns."""
        store = FrameStore(Path(":memory:"))
        # Access the underlying connection directly for schema inspection
        conn = store._conn
        cursor = conn.execute("PRAGMA table_info(frames)")
        columns = {row[1] for row in cursor.fetchall()}
        expected_columns = {
            "frame_id",
            "video_id",
            "source",
            "frame_path",
            "data_root",
            "timestamp_ms",
            "species",
            "breed",
            "lighting",
            "bowl_type",
            "quality_flag",
            "blur_score",
            "phash",
            "aug_quality",
            "aug_seed",
            "parent_frame_id",
            "is_anomaly_candidate",
            "anomaly_score",
            "annotation_status",
            "created_at",
        }
        assert columns == expected_columns

    def test_fresh_db_has_indexes(self, tmp_path: Path) -> None:
        """A freshly initialised FrameStore has exactly 4 expected indexes."""
        store = FrameStore(Path(":memory:"))
        conn = store._conn
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='frames'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        expected_indexes = {
            "idx_frames_status",
            "idx_frames_source",
            "idx_frames_quality",
            "idx_frames_anomaly",
        }
        assert expected_indexes.issubset(index_names)
        # SQLite also creates sqlite_autoindex_frames_1 for the PRIMARY KEY
        explicit_indexes = {n for n in index_names if not n.startswith("sqlite_autoindex_")}
        assert len(explicit_indexes) == 4

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """Creating two FrameStore instances pointing at the same DB file does not error."""
        db_path = tmp_path / "test.db"
        store1 = FrameStore(db_path)
        # Second init must not raise (CREATE TABLE IF NOT EXISTS)
        store2 = FrameStore(db_path)
        assert store1 is not store2
