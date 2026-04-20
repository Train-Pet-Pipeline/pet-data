"""Tests for AudioStore insert, query, and count."""
from __future__ import annotations

import sqlite3

from pet_data.storage.store import AudioSampleRow, AudioStore
from tests.conftest_migrations import load_migration


def _conn(tmp_path):
    """Create a connection with migrations 001-003 applied."""
    c = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(c)
    load_migration(2).upgrade(c)
    load_migration(3).upgrade(c)
    c.row_factory = sqlite3.Row
    return c


def test_audio_store_insert_roundtrip(tmp_path) -> None:
    """Insert an AudioSampleRow and verify round-trip via query."""
    conn = _conn(tmp_path)
    store = AudioStore(conn)
    row = AudioSampleRow(
        sample_id="sha256:a",
        storage_uri="local:///audio/a.wav",
        captured_at="2026-04-21T12:00:00+00:00",
        source_type="community",
        source_id="esc50",
        source_license="CC-BY",
        pet_species="dog",
        duration_s=5.0,
        sample_rate=44100,
        num_channels=1,
        snr_db=22.5,
        clip_type="bark",
    )
    store.insert(row)
    got = store.query(clip_type="bark")
    assert len(got) == 1 and got[0].sample_id == "sha256:a"


def test_audio_store_count(tmp_path) -> None:
    """AudioStore.count() returns 0 on empty table."""
    conn = _conn(tmp_path)
    store = AudioStore(conn)
    assert store.count() == 0
