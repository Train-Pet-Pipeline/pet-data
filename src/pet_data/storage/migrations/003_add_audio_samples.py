"""Create audio_samples table for Phase 2 multi-model refactor.

Introduces a new audio_samples table with modality='audio', storing
audio metadata (duration, sample rate, channels, clip type) and references
to audio files via storage_uri.
"""
from __future__ import annotations

import sqlite3

AUDIO_SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_samples (
    sample_id     TEXT PRIMARY KEY,
    modality      TEXT NOT NULL DEFAULT 'audio' CHECK (modality = 'audio'),
    storage_uri   TEXT NOT NULL,
    captured_at   TIMESTAMP NOT NULL,
    source_type   TEXT NOT NULL CHECK (source_type IN ('youtube','community','device','synthetic')),
    source_id     TEXT NOT NULL,
    source_license TEXT,
    pet_species   TEXT,
    duration_s    REAL NOT NULL,
    sample_rate   INTEGER NOT NULL,
    num_channels  INTEGER NOT NULL,
    snr_db        REAL,
    clip_type     TEXT CHECK (clip_type IN ('bark','meow','purr','silence','ambient')),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_audio_source ON audio_samples(source_id);
CREATE INDEX IF NOT EXISTS idx_audio_clip_type ON audio_samples(clip_type);
"""


def upgrade(conn: sqlite3.Connection) -> None:
    """Create audio_samples table with Phase 2 schema.

    Creates the audio_samples table with columns for audio metadata, indexes
    on source_id and clip_type for efficient querying.

    Args:
        conn: An open sqlite3.Connection.
    """
    conn.executescript(AUDIO_SCHEMA)
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Drop audio_samples table and associated indexes.

    Args:
        conn: An open sqlite3.Connection.
    """
    conn.executescript(
        "DROP INDEX IF EXISTS idx_audio_clip_type; "
        "DROP INDEX IF EXISTS idx_audio_source; "
        "DROP TABLE IF EXISTS audio_samples;"
    )
    conn.commit()
