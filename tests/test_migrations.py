"""Tests for database migrations."""
import sqlite3

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
