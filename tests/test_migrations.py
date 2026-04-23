"""Tests for database migrations."""
import sqlite3

import pytest

from tests.conftest_migrations import load_migration


def test_002_upgrade_adds_columns(tmp_path):
    """Test that migration 002 adds modality, storage_uri, and vision columns."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(frames)").fetchall()}
    assert {"modality", "storage_uri", "frame_width", "frame_height", "brightness_score"} <= cols
    conn.close()


def test_002_downgrade_drops_columns(tmp_path):
    """Test that migration 002 downgrade removes the added columns."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    m001 = load_migration(1)
    m002 = load_migration(2)
    m001.upgrade(conn)
    m002.upgrade(conn)
    m002.downgrade(conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(frames)").fetchall()}
    assert "modality" not in cols and "storage_uri" not in cols
    conn.close()


def test_002_backfills_storage_uri_for_existing_rows(tmp_path):
    """Test that migration 002 backfills storage_uri for rows created before the migration."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) "
        "VALUES ('f1', 'v1', 'youtube', 'frames/a.jpg', '/data')"
    )
    conn.commit()
    load_migration(2).upgrade(conn)
    row = conn.execute("SELECT modality, storage_uri FROM frames WHERE frame_id='f1'").fetchone()
    assert row[0] == "vision"
    assert row[1] == "local:///data/frames/a.jpg"
    conn.close()


def test_003_upgrade_creates_audio_table(tmp_path):
    """Test that migration 003 creates audio_samples table with correct columns."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "audio_samples" in tables
    cols = {r[1] for r in conn.execute("PRAGMA table_info(audio_samples)").fetchall()}
    required_cols = {
        "sample_id",
        "modality",
        "storage_uri",
        "duration_s",
        "sample_rate",
        "num_channels",
        "clip_type",
    }
    assert required_cols <= cols
    conn.close()


def test_003_downgrade_drops_audio_table(tmp_path):
    """Test that migration 003 downgrade removes the audio_samples table."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    m1 = load_migration(1)
    m2 = load_migration(2)
    m3 = load_migration(3)
    m1.upgrade(conn)
    m2.upgrade(conn)
    m3.upgrade(conn)
    m3.downgrade(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "audio_samples" not in tables
    conn.close()


def test_004_upgrade_adds_provenance_type_column(tmp_path):
    """Migration 004 must add provenance_type column to frames table."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    load_migration(4).upgrade(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(frames)").fetchall()}
    assert "provenance_type" in cols
    conn.close()


def test_004_provenance_type_backfill_uses_mapping(tmp_path):
    """Migration 004 backfill maps ingester names to valid SourceType literals."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)

    # Insert rows with various source (ingester_name) values before migration
    rows = [
        ("f1", "youtube"),
        ("f2", "community"),
        ("f3", "selfshot"),
        ("f4", "oxford_pet"),
        ("f5", "coco"),
        ("f6", "hospital"),
        ("f7", "local_dir"),
        ("f8", "unknown_ingester"),  # fallback to "device"
    ]
    for frame_id, source in rows:
        conn.execute(
            "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) "
            "VALUES (?, 'v1', ?, 'f.jpg', '/data')",
            (frame_id, source),
        )
    conn.commit()

    load_migration(4).upgrade(conn)

    expected = {
        "f1": "youtube",
        "f2": "community",
        "f3": "community",
        "f4": "academic_dataset",
        "f5": "academic_dataset",
        "f6": "device",
        "f7": "device",
        "f8": "device",  # unknown → device fallback
    }
    for frame_id, provenance in expected.items():
        row = conn.execute(
            "SELECT provenance_type FROM frames WHERE frame_id=?", (frame_id,)
        ).fetchone()
        assert row is not None, f"Row {frame_id} missing"
        assert row[0] == provenance, (
            f"frame_id={frame_id}: expected {provenance!r}, got {row[0]!r}"
        )
    conn.close()


def test_004_check_constraint_rejects_invalid_provenance(tmp_path):
    """provenance_type CHECK constraint prevents invalid values."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    load_migration(4).upgrade(conn)

    conn.execute(
        "INSERT INTO frames (frame_id, video_id, source, frame_path, data_root) "
        "VALUES ('f_valid', 'v1', 'youtube', 'f.jpg', '/data')"
    )
    conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE frames SET provenance_type='invalid_value' WHERE frame_id='f_valid'"
        )
        conn.commit()
    conn.close()


def test_004_downgrade_removes_provenance_type_column(tmp_path):
    """Migration 004 downgrade removes the provenance_type column."""
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    load_migration(4).upgrade(conn)
    load_migration(4).downgrade(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(frames)").fetchall()}
    assert "provenance_type" not in cols
    conn.close()


def test_004_downgrade_preserves_pre_004_check_constraints(tmp_path):
    """Migration 004 downgrade preserves all pre-004 CHECK constraints.

    Regression guard: the naive downgrade `CREATE TABLE x AS SELECT ... FROM y`
    would silently strip all CHECK constraints (data integrity risk). Explicit
    CREATE TABLE with full schema recreation keeps them.
    """
    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    load_migration(4).upgrade(conn)
    load_migration(4).downgrade(conn)

    # Verify PRIMARY KEY preserved
    pk_info = [r for r in conn.execute("PRAGMA table_info(frames)").fetchall() if r[5] > 0]
    pk_cols = [r[1] for r in pk_info]
    assert pk_cols == ["frame_id"], f"PRIMARY KEY lost after downgrade: {pk_cols}"

    # Verify lighting CHECK constraint preserved (invalid value rejected)
    conn.execute(
        """INSERT INTO frames (frame_id, video_id, source, frame_path, data_root)
           VALUES ('f1', 'v1', 'youtube', 'path/to/frame.jpg', '/data')"""
    )
    conn.commit()
    try:
        conn.execute("UPDATE frames SET lighting = 'invalid_value' WHERE frame_id = 'f1'")
        conn.commit()
        raise AssertionError("lighting CHECK constraint lost — 'invalid_value' accepted")
    except sqlite3.IntegrityError:
        pass  # expected — CHECK constraint correctly rejected invalid value

    # Verify quality_flag CHECK preserved
    try:
        conn.execute("UPDATE frames SET quality_flag = 'invalid' WHERE frame_id = 'f1'")
        conn.commit()
        raise AssertionError("quality_flag CHECK constraint lost after downgrade")
    except sqlite3.IntegrityError:
        pass

    # Verify modality CHECK preserved (from migration 002)
    try:
        conn.execute("UPDATE frames SET modality = 'invalid' WHERE frame_id = 'f1'")
        conn.commit()
        raise AssertionError("modality CHECK constraint lost after downgrade")
    except sqlite3.IntegrityError:
        pass

    # Verify NOT NULL preserved (on source column for example)
    try:
        conn.execute(
            """INSERT INTO frames (frame_id, video_id, frame_path, data_root)
               VALUES ('f2', 'v2', 'path', '/data')"""  # missing source (NOT NULL)
        )
        conn.commit()
        raise AssertionError("NOT NULL on source lost after downgrade")
    except sqlite3.IntegrityError:
        pass

    conn.close()
