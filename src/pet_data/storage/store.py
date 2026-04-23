"""Storage layer for pet-data: FrameStore with full CRUD over SQLite."""
from __future__ import annotations

import importlib.util
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from pet_infra.logging import get_logger

logger = get_logger("pet-data")


@dataclass
class FrameRecord:
    """Represents a single frame record stored in the frames table.

    Attributes:
        frame_id: Unique identifier for this frame.
        video_id: Identifier of the source video this frame belongs to.
        source: Data source label (e.g. 'youtube', 'community').
        frame_path: Relative path to the frame image file.
        data_root: Absolute base directory under which frame_path resolves.
        timestamp_ms: Position in the source video, in milliseconds.
        species: Pet species label (e.g. 'cat', 'dog').
        breed: Breed label for the animal.
        lighting: Lighting condition of the frame.
        bowl_type: Type of feeding bowl visible in the frame.
        quality_flag: Quality tier — 'normal', 'low', or 'failed'.
        blur_score: Laplacian variance blur score (lower = blurrier).
        phash: Perceptual hash bytes for deduplication.
        aug_quality: Augmentation outcome — 'ok' or 'failed'.
        aug_seed: RNG seed used for augmentation reproducibility.
        parent_frame_id: frame_id of the original frame this was augmented from.
        is_anomaly_candidate: Whether this frame has been flagged for anomaly review.
        anomaly_score: Anomaly detector confidence score.
        annotation_status: Current annotation lifecycle state.
        provenance_type: Legal/compliance provenance category (SourceType literal).
            Separate from source (ingester_name). Added Phase 3 concept separation.
    """

    frame_id: str
    video_id: str
    source: str
    frame_path: str
    data_root: str
    timestamp_ms: int | None = None
    species: str | None = None
    breed: str | None = None
    lighting: str | None = None
    bowl_type: str | None = None
    quality_flag: str = "normal"
    blur_score: float | None = None
    phash: bytes | None = None
    aug_quality: str | None = None
    aug_seed: int | None = None
    parent_frame_id: str | None = None
    is_anomaly_candidate: bool = False
    anomaly_score: float | None = None
    annotation_status: str = "pending"
    modality: str = "vision"
    storage_uri: str = ""
    frame_width: int | None = None
    frame_height: int | None = None
    brightness_score: float | None = None
    provenance_type: str = "device"


@dataclass
class FrameFilter:
    """Filter criteria for querying the frames table.

    Attributes:
        source: Filter by data source label.
        quality_flag: Filter by quality tier.
        annotation_status: Filter by annotation lifecycle state.
        is_anomaly_candidate: Filter by anomaly candidate flag.
        limit: Maximum number of rows to return.
        offset: Number of rows to skip before returning results.
    """

    source: str | None = None
    quality_flag: str | None = None
    annotation_status: str | None = None
    is_anomaly_candidate: bool | None = None
    modality: str | None = None
    limit: int = 1000
    offset: int = 0


class FrameStore:
    """SQLite-backed store for frame records.

    All database access goes through this class; no code outside this module
    should hold a direct connection handle.
    """

    def __init__(self, db_path: Path) -> None:
        """Open (or create) the SQLite database and ensure the schema exists.

        Args:
            db_path: Filesystem path for the SQLite file.  Pass
                ``Path(":memory:")`` for an in-memory database in tests.
        """
        str_path = ":memory:" if db_path == Path(":memory:") else str(db_path)
        self._conn = sqlite3.connect(str_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        schema_path = Path(__file__).parent / "schema.sql"
        self._conn.executescript(schema_path.read_text())
        self._conn.commit()
        self._apply_subsequent_migrations()
        logger.info(
            '{"event": "frame_store_init", "db_path": "%s"}',
            str_path,
        )

    def _apply_subsequent_migrations(self) -> None:
        """Run all migrations 002+ from the migrations directory (idempotent).

        Uses ``importlib.util.spec_from_file_location`` because migration filenames
        start with digits and cannot be imported via dotted names.  Tolerates
        ``OperationalError: duplicate column name`` so that re-running is safe.
        """
        migrations_dir = Path(__file__).parent / "migrations"
        migration_files = sorted(migrations_dir.glob("[0-9][0-9][0-9]_*.py"))
        for path in migration_files:
            number = int(path.stem[:3])
            if number < 2:
                continue
            mod = self._load_migration_module(path)
            try:
                mod.upgrade(self._conn)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" in str(exc):
                    continue
                raise

    @staticmethod
    def _load_migration_module(path: Path) -> ModuleType:
        """Load a migration module from a filesystem path.

        Args:
            path: Absolute path to the migration ``.py`` file.

        Returns:
            The loaded module with ``upgrade`` and ``downgrade`` callables.
        """
        spec = importlib.util.spec_from_file_location(f"migration_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> FrameStore:
        """Support ``with FrameStore(...) as store:`` usage."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close connection on context-manager exit."""
        self.close()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _record_to_params(frame: FrameRecord) -> dict:
        """Convert a FrameRecord to a dict of SQL parameters."""
        return {
            "frame_id": frame.frame_id,
            "video_id": frame.video_id,
            "source": frame.source,
            "frame_path": frame.frame_path,
            "data_root": frame.data_root,
            "timestamp_ms": frame.timestamp_ms,
            "species": frame.species,
            "breed": frame.breed,
            "lighting": frame.lighting,
            "bowl_type": frame.bowl_type,
            "quality_flag": frame.quality_flag,
            "blur_score": frame.blur_score,
            "phash": frame.phash,
            "aug_quality": frame.aug_quality,
            "aug_seed": frame.aug_seed,
            "parent_frame_id": frame.parent_frame_id,
            "is_anomaly_candidate": int(frame.is_anomaly_candidate),
            "anomaly_score": frame.anomaly_score,
            "annotation_status": frame.annotation_status,
            "modality": frame.modality,
            "storage_uri": frame.storage_uri,
            "frame_width": frame.frame_width,
            "frame_height": frame.frame_height,
            "brightness_score": frame.brightness_score,
            "provenance_type": frame.provenance_type,
        }

    def insert_frame(self, frame: FrameRecord) -> str:
        """Insert a single frame record and return its frame_id.

        Args:
            frame: The :class:`FrameRecord` to persist.

        Returns:
            The ``frame_id`` of the inserted record.

        Raises:
            sqlite3.IntegrityError: If ``frame_id`` already exists.
        """
        self._conn.execute(
            """
            INSERT INTO frames (
                frame_id, video_id, source, frame_path, data_root,
                timestamp_ms, species, breed, lighting, bowl_type,
                quality_flag, blur_score, phash, aug_quality, aug_seed,
                parent_frame_id, is_anomaly_candidate, anomaly_score,
                annotation_status,
                modality, storage_uri, frame_width, frame_height, brightness_score,
                provenance_type
            ) VALUES (
                :frame_id, :video_id, :source, :frame_path, :data_root,
                :timestamp_ms, :species, :breed, :lighting, :bowl_type,
                :quality_flag, :blur_score, :phash, :aug_quality, :aug_seed,
                :parent_frame_id, :is_anomaly_candidate, :anomaly_score,
                :annotation_status,
                :modality, :storage_uri, :frame_width, :frame_height, :brightness_score,
                :provenance_type
            )
            """,
            self._record_to_params(frame),
        )
        self._conn.commit()
        logger.info('{"event": "insert_frame", "frame_id": "%s"}', frame.frame_id)
        return frame.frame_id

    def get_frame(self, frame_id: str) -> FrameRecord | None:
        """Retrieve a single frame by primary key.

        Args:
            frame_id: The primary key to look up.

        Returns:
            A :class:`FrameRecord` if found, otherwise ``None``.
        """
        row = self._conn.execute(
            "SELECT * FROM frames WHERE frame_id = ?", (frame_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def bulk_insert_frames(self, frames: list[FrameRecord]) -> int:
        """Insert multiple frame records in a single transaction.

        The entire batch is rolled back if any single insert fails (e.g. duplicate key).

        Args:
            frames: List of :class:`FrameRecord` objects to insert.

        Returns:
            Number of rows successfully inserted.

        Raises:
            sqlite3.IntegrityError: Propagated after rollback if a duplicate exists.
        """
        rows = [self._record_to_params(f) for f in frames]
        try:
            self._conn.executemany(
                """
                INSERT INTO frames (
                    frame_id, video_id, source, frame_path, data_root,
                    timestamp_ms, species, breed, lighting, bowl_type,
                    quality_flag, blur_score, phash, aug_quality, aug_seed,
                    parent_frame_id, is_anomaly_candidate, anomaly_score,
                    annotation_status,
                    modality, storage_uri, frame_width, frame_height, brightness_score,
                    provenance_type
                ) VALUES (
                    :frame_id, :video_id, :source, :frame_path, :data_root,
                    :timestamp_ms, :species, :breed, :lighting, :bowl_type,
                    :quality_flag, :blur_score, :phash, :aug_quality, :aug_seed,
                    :parent_frame_id, :is_anomaly_candidate, :anomaly_score,
                    :annotation_status,
                    :modality, :storage_uri, :frame_width, :frame_height, :brightness_score,
                    :provenance_type
                )
                """,
                rows,
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            self._conn.rollback()
            raise
        count = len(frames)
        logger.info('{"event": "bulk_insert_frames", "count": %d}', count)
        return count

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_frames(self, filters: FrameFilter) -> list[FrameRecord]:
        """Return frames matching the given filter criteria.

        Only non-``None`` filter fields generate WHERE clauses.

        Args:
            filters: A :class:`FrameFilter` describing the query constraints.

        Returns:
            List of matching :class:`FrameRecord` objects.
        """
        clauses: list[str] = []
        params: list[object] = []

        if filters.source is not None:
            clauses.append("source = ?")
            params.append(filters.source)
        if filters.quality_flag is not None:
            clauses.append("quality_flag = ?")
            params.append(filters.quality_flag)
        if filters.annotation_status is not None:
            clauses.append("annotation_status = ?")
            params.append(filters.annotation_status)
        if filters.is_anomaly_candidate is not None:
            clauses.append("is_anomaly_candidate = ?")
            params.append(int(filters.is_anomaly_candidate))
        if filters.modality is not None:
            clauses.append("modality = ?")
            params.append(filters.modality)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.extend([filters.limit, filters.offset])
        sql = f"SELECT * FROM frames {where} LIMIT ? OFFSET ?"  # noqa: S608
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_phashes(self, source: str | None = None) -> dict[str, bytes]:
        """Return a mapping of frame_id to phash for frames that have a phash.

        Args:
            source: Optional source filter.  When given, only frames from that
                source are included.

        Returns:
            Dictionary mapping ``frame_id`` to raw phash bytes.
        """
        if source is not None:
            rows = self._conn.execute(
                "SELECT frame_id, phash FROM frames WHERE phash IS NOT NULL AND source = ?",
                (source,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT frame_id, phash FROM frames WHERE phash IS NOT NULL"
            ).fetchall()
        return {row["frame_id"]: bytes(row["phash"]) for row in rows}

    # ------------------------------------------------------------------
    # Update helpers
    # ------------------------------------------------------------------

    def update_quality(self, frame_id: str, quality_flag: str, blur_score: float) -> None:
        """Update the quality_flag and blur_score for a frame.

        Args:
            frame_id: Primary key of the frame to update.
            quality_flag: New quality tier ('normal', 'low', or 'failed').
            blur_score: New Laplacian variance blur score.
        """
        cursor = self._conn.execute(
            "UPDATE frames SET quality_flag = ?, blur_score = ? WHERE frame_id = ?",
            (quality_flag, blur_score, frame_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Frame not found: {frame_id}")
        self._conn.commit()
        logger.info(
            '{"event": "update_quality", "frame_id": "%s", "quality_flag": "%s"}',
            frame_id,
            quality_flag,
        )

    def update_anomaly(self, frame_id: str, is_candidate: bool, score: float) -> None:
        """Update anomaly detection results for a frame.

        Args:
            frame_id: Primary key of the frame to update.
            is_candidate: Whether the frame is an anomaly candidate.
            score: Anomaly detector confidence score.

        Raises:
            ValueError: If ``frame_id`` does not exist.
        """
        cursor = self._conn.execute(
            "UPDATE frames SET is_anomaly_candidate = ?, anomaly_score = ? WHERE frame_id = ?",
            (int(is_candidate), score, frame_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Frame not found: {frame_id}")
        self._conn.commit()
        logger.info(
            '{"event": "update_anomaly", "frame_id": "%s", "is_candidate": %s}',
            frame_id,
            is_candidate,
        )

    def update_annotation_status(self, frame_id: str, status: str) -> None:
        """Update the annotation lifecycle status for a frame.

        Args:
            frame_id: Primary key of the frame to update.
            status: New annotation status value (must satisfy the column CHECK constraint).

        Raises:
            ValueError: If ``frame_id`` does not exist.
        """
        cursor = self._conn.execute(
            "UPDATE frames SET annotation_status = ? WHERE frame_id = ?",
            (status, frame_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Frame not found: {frame_id}")
        self._conn.commit()
        logger.info(
            '{"event": "update_annotation_status", "frame_id": "%s", "status": "%s"}',
            frame_id,
            status,
        )

    def update_augmentation(
        self, frame_id: str, aug_quality: str, parent_frame_id: str
    ) -> None:
        """Record augmentation outcome for a frame.

        Args:
            frame_id: Primary key of the augmented frame.
            aug_quality: Augmentation result — 'ok' or 'failed'.
            parent_frame_id: frame_id of the original frame this was derived from.

        Raises:
            ValueError: If ``frame_id`` does not exist.
        """
        cursor = self._conn.execute(
            "UPDATE frames SET aug_quality = ?, parent_frame_id = ? WHERE frame_id = ?",
            (aug_quality, parent_frame_id, frame_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Frame not found: {frame_id}")
        self._conn.commit()
        logger.info(
            '{"event": "update_augmentation", "frame_id": "%s", "aug_quality": "%s"}',
            frame_id,
            aug_quality,
        )

    def update_phash(self, frame_id: str, phash: bytes) -> None:
        """Store the perceptual hash for a frame.

        Args:
            frame_id: Primary key of the frame to update.
            phash: Raw bytes of the perceptual hash.

        Raises:
            ValueError: If ``frame_id`` does not exist.
        """
        cursor = self._conn.execute(
            "UPDATE frames SET phash = ? WHERE frame_id = ?",
            (phash, frame_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Frame not found: {frame_id}")
        self._conn.commit()
        logger.info('{"event": "update_phash", "frame_id": "%s"}', frame_id)

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def count_by_source(self) -> dict[str, int]:
        """Return the number of frames grouped by source label.

        Returns:
            Dictionary mapping source name to frame count.
        """
        rows = self._conn.execute(
            "SELECT source, COUNT(*) AS cnt FROM frames GROUP BY source"
        ).fetchall()
        return {row["source"]: row["cnt"] for row in rows}

    def count_by_status(self) -> dict[str, int]:
        """Return the number of frames grouped by annotation_status.

        Returns:
            Dictionary mapping annotation status to frame count.
        """
        rows = self._conn.execute(
            "SELECT annotation_status, COUNT(*) AS cnt FROM frames GROUP BY annotation_status"
        ).fetchall()
        return {row["annotation_status"]: row["cnt"] for row in rows}

    def query_unscored_frames(self) -> list[FrameRecord]:
        """Return all frames that have not been scored (anomaly_score IS NULL).

        Returns:
            List of :class:`FrameRecord` objects with no anomaly score.
        """
        rows = self._conn.execute(
            "SELECT * FROM frames WHERE anomaly_score IS NULL"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_normal_frames(self) -> int:
        """Count frames suitable for autoencoder training.

        Returns:
            Number of frames with quality_flag='normal' and is_anomaly_candidate=0.
        """
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM frames
            WHERE quality_flag = 'normal'
              AND is_anomaly_candidate = 0
            """
        ).fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> FrameRecord:
        """Convert a sqlite3.Row to a FrameRecord, handling type coercions.

        SQLite stores booleans as INTEGER (0/1); this method converts
        ``is_anomaly_candidate`` back to Python ``bool``.

        Args:
            row: A :class:`sqlite3.Row` fetched from the frames table.

        Returns:
            A populated :class:`FrameRecord`.
        """
        return FrameRecord(
            frame_id=row["frame_id"],
            video_id=row["video_id"],
            source=row["source"],
            frame_path=row["frame_path"],
            data_root=row["data_root"],
            timestamp_ms=row["timestamp_ms"],
            species=row["species"],
            breed=row["breed"],
            lighting=row["lighting"],
            bowl_type=row["bowl_type"],
            quality_flag=row["quality_flag"],
            blur_score=row["blur_score"],
            phash=bytes(row["phash"]) if row["phash"] is not None else None,
            aug_quality=row["aug_quality"],
            aug_seed=row["aug_seed"],
            parent_frame_id=row["parent_frame_id"],
            is_anomaly_candidate=bool(row["is_anomaly_candidate"]),
            anomaly_score=row["anomaly_score"],
            annotation_status=row["annotation_status"],
            modality=row["modality"] if "modality" in row.keys() else "vision",
            storage_uri=row["storage_uri"] if "storage_uri" in row.keys() else "",
            frame_width=row["frame_width"] if "frame_width" in row.keys() else None,
            frame_height=row["frame_height"] if "frame_height" in row.keys() else None,
            brightness_score=row["brightness_score"] if "brightness_score" in row.keys() else None,
            provenance_type=row["provenance_type"] if "provenance_type" in row.keys() else "device",
        )


# ---------------------------------------------------------------------------
# AudioSampleRow + AudioStore
# ---------------------------------------------------------------------------


@dataclass
class AudioSampleRow:
    """Represents a single audio sample stored in the audio_samples table.

    Attributes:
        sample_id: Unique identifier (e.g. sha256 digest) for this sample.
        storage_uri: URI to the audio file (e.g. 'local:///audio/a.wav').
        captured_at: ISO-8601 timestamp of when the audio was captured.
        source_type: Provenance category — 'youtube', 'community', 'device', or 'synthetic'.
        source_id: Dataset or collection identifier (e.g. 'esc50').
        source_license: SPDX licence string (e.g. 'CC-BY') or None.
        pet_species: Species label ('dog', 'cat', …) or None.
        duration_s: Audio clip length in seconds.
        sample_rate: Audio sample rate in Hz.
        num_channels: Number of audio channels.
        snr_db: Signal-to-noise ratio in decibels, or None if unknown.
        clip_type: Semantic label — 'bark', 'meow', 'purr', 'silence', 'ambient', or None.
    """

    sample_id: str
    storage_uri: str
    captured_at: str
    source_type: str
    source_id: str
    source_license: str | None = None
    pet_species: str | None = None
    duration_s: float = 0.0
    sample_rate: int = 0
    num_channels: int = 1
    snr_db: float | None = None
    clip_type: str | None = None


class AudioStore:
    """SQLite-backed store for audio sample records.

    Operates on the ``audio_samples`` table created by migration 003.
    Accepts an **open** :class:`sqlite3.Connection` (unlike :class:`FrameStore`
    which takes a path).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialise AudioStore with an open database connection.

        Args:
            conn: An open :class:`sqlite3.Connection` with ``row_factory`` set
                to ``sqlite3.Row``.
        """
        self._conn = conn

    def insert(self, row: AudioSampleRow) -> str:
        """Insert a single audio sample record and return its sample_id.

        Args:
            row: The :class:`AudioSampleRow` to persist.

        Returns:
            The ``sample_id`` of the inserted record.

        Raises:
            sqlite3.IntegrityError: If ``sample_id`` already exists.
        """
        self._conn.execute(
            """
            INSERT INTO audio_samples (
                sample_id, storage_uri, captured_at,
                source_type, source_id, source_license,
                pet_species, duration_s, sample_rate, num_channels,
                snr_db, clip_type
            ) VALUES (
                :sample_id, :storage_uri, :captured_at,
                :source_type, :source_id, :source_license,
                :pet_species, :duration_s, :sample_rate, :num_channels,
                :snr_db, :clip_type
            )
            """,
            {
                "sample_id": row.sample_id,
                "storage_uri": row.storage_uri,
                "captured_at": row.captured_at,
                "source_type": row.source_type,
                "source_id": row.source_id,
                "source_license": row.source_license,
                "pet_species": row.pet_species,
                "duration_s": row.duration_s,
                "sample_rate": row.sample_rate,
                "num_channels": row.num_channels,
                "snr_db": row.snr_db,
                "clip_type": row.clip_type,
            },
        )
        self._conn.commit()
        logger.info('{"event": "audio_store_insert", "sample_id": "%s"}', row.sample_id)
        return row.sample_id

    def query(self, clip_type: str | None = None) -> list[AudioSampleRow]:
        """Return audio samples optionally filtered by clip_type.

        Args:
            clip_type: When given, restrict results to this clip type.

        Returns:
            List of :class:`AudioSampleRow` objects matching the filter.
        """
        _fields = AudioSampleRow.__dataclass_fields__
        if clip_type is not None:
            db_rows = self._conn.execute(
                "SELECT * FROM audio_samples WHERE clip_type = ?", (clip_type,)
            ).fetchall()
        else:
            db_rows = self._conn.execute("SELECT * FROM audio_samples").fetchall()
        return [
            AudioSampleRow(**{k: r[k] for k in r.keys() if k in _fields})
            for r in db_rows
        ]

    def count(self) -> int:
        """Return the total number of audio sample records.

        Returns:
            Integer count of rows in the audio_samples table.
        """
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM audio_samples").fetchone()
        return row["cnt"]
