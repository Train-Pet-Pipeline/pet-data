"""New Phase 2 tests for FrameStore modality columns and migration driver."""
from __future__ import annotations

from pathlib import Path

import pytest

from pet_data.storage.store import FrameFilter, FrameRecord, FrameStore


def test_framestore_insert_and_query_with_modality(tmp_path: Path) -> None:
    """Insert a FrameRecord with all 5 new fields, query by modality, verify round-trip."""
    store = FrameStore(tmp_path / "db.sqlite")
    record = FrameRecord(
        frame_id="fr_modality_001",
        video_id="vid_001",
        source="youtube",
        frame_path="frames/fr_modality_001.jpg",
        data_root="/data",
        modality="vision",
        storage_uri="local:///data/frames/fr_modality_001.jpg",
        frame_width=1920,
        frame_height=1080,
        brightness_score=0.75,
    )
    store.insert_frame(record)

    results = store.query_frames(FrameFilter(modality="vision"))
    assert len(results) == 1
    got = results[0]
    assert got.frame_id == "fr_modality_001"
    assert got.modality == "vision"
    assert got.storage_uri == "local:///data/frames/fr_modality_001.jpg"
    assert got.frame_width == 1920
    assert got.frame_height == 1080
    assert got.brightness_score == pytest.approx(0.75)
    store.close()


def test_framestore_production_db_has_phase2_columns(tmp_path: Path) -> None:
    """FrameStore on a new DB must have phase2 columns AND audio_samples table."""
    store = FrameStore(tmp_path / "prod.sqlite")

    # Check phase 2 columns exist in frames table
    cols = {
        row[1]
        for row in store._conn.execute("PRAGMA table_info(frames)").fetchall()
    }
    phase2_cols = ("modality", "storage_uri", "frame_width", "frame_height", "brightness_score")
    for expected_col in phase2_cols:
        assert expected_col in cols, f"Missing column: {expected_col}"

    # Check audio_samples table exists
    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "audio_samples" in tables, "audio_samples table missing"
    store.close()


def test_framestore_insert_populates_provenance_type(tmp_path: Path) -> None:
    """FrameRecord with provenance_type is stored and retrieved correctly."""
    store = FrameStore(tmp_path / "db.sqlite")
    record = FrameRecord(
        frame_id="fr_prov_001",
        video_id="vid_001",
        source="oxford_pet",
        frame_path="frames/fr_prov_001.jpg",
        data_root="/data",
        modality="vision",
        storage_uri="local:///data/frames/fr_prov_001.jpg",
        frame_width=224,
        frame_height=224,
        brightness_score=0.5,
        provenance_type="academic_dataset",
    )
    store.insert_frame(record)

    result = store.get_frame("fr_prov_001")
    assert result is not None
    assert result.provenance_type == "academic_dataset"
    assert result.source == "oxford_pet"  # ingester_name preserved in source column
    store.close()
