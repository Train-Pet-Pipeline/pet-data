"""Add modality + storage_uri + vision-specific columns to `frames`.

Part of Phase 2 multi-model refactor. Existing rows get modality='vision'
and a backfilled storage_uri derived from data_root + frame_path.
"""
from __future__ import annotations

import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    """Add Phase 2 columns to frames table.

    Adds:
        - modality: TEXT, defaults to 'vision', CHECK constraint on valid values
        - storage_uri: TEXT, defaults to empty string (backfilled from data_root + frame_path)
        - frame_width, frame_height: INTEGER, vision-specific dimensions
        - brightness_score: REAL, vision-specific quality metric

    Args:
        conn: An open sqlite3.Connection.
    """
    conn.executescript(
        """
        ALTER TABLE frames ADD COLUMN modality TEXT NOT NULL DEFAULT 'vision'
            CHECK (modality IN ('vision', 'audio', 'sensor', 'multimodal'));
        ALTER TABLE frames ADD COLUMN storage_uri TEXT NOT NULL DEFAULT '';
        ALTER TABLE frames ADD COLUMN frame_width INTEGER;
        ALTER TABLE frames ADD COLUMN frame_height INTEGER;
        ALTER TABLE frames ADD COLUMN brightness_score REAL;

        UPDATE frames
           SET storage_uri = 'local://' || data_root || '/' || frame_path
         WHERE storage_uri = '';

        CREATE INDEX IF NOT EXISTS idx_frames_modality ON frames(modality);
        """
    )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Remove Phase 2 columns via table rebuild (SQLite < 3.35 doesn't support DROP COLUMN).

    Rebuilds the frames table, preserving all pre-Phase-2 columns.

    Args:
        conn: An open sqlite3.Connection.
    """
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_frames_modality;
        CREATE TABLE frames_tmp AS SELECT
            frame_id, video_id, source, frame_path, data_root, timestamp_ms,
            species, breed, lighting, bowl_type, quality_flag, blur_score,
            phash, aug_quality, aug_seed, parent_frame_id, is_anomaly_candidate,
            anomaly_score, annotation_status, created_at
        FROM frames;
        DROP TABLE frames;
        ALTER TABLE frames_tmp RENAME TO frames;
        """
    )
    conn.commit()
