"""Add provenance_type column to frames table.

Concept separation step 3 (Phase 3). Distinguishes ingester identity
(the existing ``source`` column, which holds the ingester_name value) from
legal/compliance provenance (the new ``provenance_type`` column).

Backfill maps current ``source`` values to the correct v3.1.0 SourceType
literals using the user-approved mapping (2026-04-23):

  youtube      → youtube
  community    → community
  selfshot     → community   (user-contributed; no separate literal)
  oxford_pet   → academic_dataset
  coco         → academic_dataset
  hospital     → device       (partner hospital hardware captures)
  local_dir    → device       (dev/internal default)
  <unknown>    → device       (safe fallback; logged as warning)

The existing ``source`` column is preserved (it now semantically holds the
ingester_name).  A future migration may rename it for clarity once downstream
consumers are updated.

**Extending SourceType in the future:** If pet-schema's ``SourceType`` gains
a new literal (e.g., ``restricted_medical``), create a NEW migration file
(e.g., ``005_extend_provenance_literals.py``) that rebuilds the frames table
with the updated CHECK constraint. DO NOT modify this file — committed
Alembic migrations are immutable (CLAUDE.md rule). Same policy applies to
audio_samples table (currently pinned to the original 4 literals in
``003_add_audio_samples.py``).
"""
from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# User-approved ingester → provenance mapping (2026-04-23 Phase 3 brainstorm)
_INGESTER_TO_PROVENANCE: dict[str, str] = {
    "youtube": "youtube",
    "community": "community",
    "selfshot": "community",
    "oxford_pet": "academic_dataset",
    "coco": "academic_dataset",
    "hospital": "device",
    "local_dir": "device",
}

_FALLBACK_PROVENANCE = "device"

_VALID_PROVENANCE = frozenset(
    ["youtube", "community", "device", "synthetic", "academic_dataset", "commercial_licensed"]
)


def upgrade(conn: sqlite3.Connection) -> None:
    """Add provenance_type column to frames, backfill from source, add CHECK constraint.

    Steps:
    1. Add provenance_type TEXT column (nullable initially for backfill)
    2. Backfill all rows using the ingester→provenance mapping
    3. Set NOT NULL DEFAULT 'device' after backfill
    4. Recreate column with CHECK constraint via table rebuild

    SQLite does not support ADD COLUMN with CHECK constraints, so we add the
    column, backfill, then rebuild the table with the constraint.

    Args:
        conn: An open sqlite3.Connection.
    """
    # Step 1: Add column (nullable, no constraint yet)
    try:
        conn.execute("ALTER TABLE frames ADD COLUMN provenance_type TEXT")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            return  # already migrated; idempotent
        raise

    # Step 2: Backfill existing rows
    rows = conn.execute(
        "SELECT frame_id, source FROM frames WHERE provenance_type IS NULL"
    ).fetchall()
    unknown_sources: list[str] = []
    for frame_id, source in rows:
        provenance = _INGESTER_TO_PROVENANCE.get(source)
        if provenance is None:
            provenance = _FALLBACK_PROVENANCE
            unknown_sources.append(source)
        conn.execute(
            "UPDATE frames SET provenance_type=? WHERE frame_id=?",
            (provenance, frame_id),
        )
    conn.commit()

    if unknown_sources:
        logger.warning(
            '{"event": "migration_004_unknown_sources", '
            '"count": %d, "examples": %s, '
            '"action": "fallback_to_device"}',
            len(unknown_sources),
            str(list(set(unknown_sources))[:5]),
        )

    # Step 3: Rebuild table to enforce NOT NULL + CHECK constraint
    # (SQLite does not support ALTER COLUMN or ADD CONSTRAINT)
    conn.executescript(
        """
        CREATE TABLE frames_tmp AS SELECT * FROM frames;

        DROP TABLE frames;

        CREATE TABLE frames (
            frame_id        TEXT PRIMARY KEY,
            video_id        TEXT NOT NULL,
            source          TEXT NOT NULL,
            frame_path      TEXT NOT NULL,
            data_root       TEXT NOT NULL,
            timestamp_ms    INTEGER,
            species         TEXT,
            breed           TEXT,
            lighting        TEXT CHECK(lighting IN ('bright','dim','infrared_night','unknown')),
            bowl_type       TEXT,
            quality_flag    TEXT NOT NULL DEFAULT 'normal'
                            CHECK(quality_flag IN ('normal','low','failed')),
            blur_score      REAL,
            phash           BLOB,
            aug_quality     TEXT CHECK(aug_quality IN ('ok','failed') OR aug_quality IS NULL),
            aug_seed        INTEGER,
            parent_frame_id TEXT,
            is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
            anomaly_score   REAL,
            annotation_status TEXT NOT NULL DEFAULT 'pending'
                CHECK(annotation_status IN ('pending','annotating','auto_checked',
                                            'approved','needs_review','reviewed','rejected','exported')),
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            modality        TEXT NOT NULL DEFAULT 'vision'
                            CHECK (modality IN ('vision', 'audio', 'sensor', 'multimodal')),
            storage_uri     TEXT NOT NULL DEFAULT '',
            frame_width     INTEGER,
            frame_height    INTEGER,
            brightness_score REAL,
            provenance_type TEXT NOT NULL DEFAULT 'device'
                CHECK(provenance_type IN (
                    'youtube','community','device','synthetic',
                    'academic_dataset','commercial_licensed'
                ))
        );

        INSERT INTO frames SELECT * FROM frames_tmp;

        DROP TABLE frames_tmp;

        CREATE INDEX IF NOT EXISTS idx_frames_status    ON frames(annotation_status);
        CREATE INDEX IF NOT EXISTS idx_frames_source    ON frames(source);
        CREATE INDEX IF NOT EXISTS idx_frames_quality   ON frames(quality_flag);
        CREATE INDEX IF NOT EXISTS idx_frames_anomaly
            ON frames(is_anomaly_candidate, anomaly_score DESC);
        CREATE INDEX IF NOT EXISTS idx_frames_modality  ON frames(modality);
        CREATE INDEX IF NOT EXISTS idx_frames_provenance ON frames(provenance_type);
        """
    )
    conn.commit()


def downgrade(conn: sqlite3.Connection) -> None:
    """Remove provenance_type column via table rebuild.

    **CRITICAL**: SQLite's ``CREATE TABLE x AS SELECT ... FROM y`` creates a table
    with NO constraints (no PRIMARY KEY, no NOT NULL, no CHECK, no DEFAULT).
    To preserve all pre-004 constraints on downgrade, this function explicitly
    recreates the frames table matching the combined post-002 / pre-004 schema
    (schema.sql + migration 002 additions), then re-populates from a temp copy.

    Args:
        conn: An open sqlite3.Connection.
    """
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_frames_provenance;

        -- Step 1: stash current rows into an unconstrained temp table.
        CREATE TABLE frames_tmp AS SELECT
            frame_id, video_id, source, frame_path, data_root, timestamp_ms,
            species, breed, lighting, bowl_type, quality_flag, blur_score,
            phash, aug_quality, aug_seed, parent_frame_id, is_anomaly_candidate,
            anomaly_score, annotation_status, created_at,
            modality, storage_uri, frame_width, frame_height, brightness_score
        FROM frames;

        DROP TABLE frames;

        -- Step 2: recreate frames with full pre-004 schema (schema.sql + migration 002).
        -- Matches exactly: same columns, types, PRIMARY KEY, NOT NULL, DEFAULT, CHECK.
        CREATE TABLE frames (
            frame_id        TEXT PRIMARY KEY,
            video_id        TEXT NOT NULL,
            source          TEXT NOT NULL,
            frame_path      TEXT NOT NULL,
            data_root       TEXT NOT NULL,
            timestamp_ms    INTEGER,
            species         TEXT,
            breed           TEXT,
            lighting        TEXT CHECK(lighting IN ('bright','dim','infrared_night','unknown')),
            bowl_type       TEXT,
            quality_flag    TEXT NOT NULL DEFAULT 'normal'
                            CHECK(quality_flag IN ('normal','low','failed')),
            blur_score      REAL,
            phash           BLOB,
            aug_quality     TEXT CHECK(aug_quality IN ('ok','failed') OR aug_quality IS NULL),
            aug_seed        INTEGER,
            parent_frame_id TEXT,
            is_anomaly_candidate INTEGER NOT NULL DEFAULT 0,
            anomaly_score   REAL,
            annotation_status TEXT NOT NULL DEFAULT 'pending'
                CHECK(annotation_status IN ('pending','annotating','auto_checked',
                                            'approved','needs_review','reviewed','rejected','exported')),
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            -- Migration 002 additions:
            modality        TEXT NOT NULL DEFAULT 'vision'
                            CHECK(modality IN ('vision','audio','sensor','multimodal')),
            storage_uri     TEXT NOT NULL DEFAULT '',
            frame_width     INTEGER,
            frame_height    INTEGER,
            brightness_score REAL
        );

        -- Step 3: repopulate from temp, drop temp.
        INSERT INTO frames SELECT * FROM frames_tmp;
        DROP TABLE frames_tmp;

        CREATE INDEX IF NOT EXISTS idx_frames_status    ON frames(annotation_status);
        CREATE INDEX IF NOT EXISTS idx_frames_source    ON frames(source);
        CREATE INDEX IF NOT EXISTS idx_frames_quality   ON frames(quality_flag);
        CREATE INDEX IF NOT EXISTS idx_frames_anomaly
            ON frames(is_anomaly_candidate, anomaly_score DESC);
        CREATE INDEX IF NOT EXISTS idx_frames_modality  ON frames(modality);
        """
    )
    conn.commit()
