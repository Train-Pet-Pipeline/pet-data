"""Storage layer for pet-data: FrameStore with full CRUD over SQLite."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


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
        logger.info(
            '{"event": "frame_store_init", "db_path": "%s"}',
            str_path,
        )

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

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
                annotation_status
            ) VALUES (
                :frame_id, :video_id, :source, :frame_path, :data_root,
                :timestamp_ms, :species, :breed, :lighting, :bowl_type,
                :quality_flag, :blur_score, :phash, :aug_quality, :aug_seed,
                :parent_frame_id, :is_anomaly_candidate, :anomaly_score,
                :annotation_status
            )
            """,
            {
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
            },
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
        rows = [
            {
                "frame_id": f.frame_id,
                "video_id": f.video_id,
                "source": f.source,
                "frame_path": f.frame_path,
                "data_root": f.data_root,
                "timestamp_ms": f.timestamp_ms,
                "species": f.species,
                "breed": f.breed,
                "lighting": f.lighting,
                "bowl_type": f.bowl_type,
                "quality_flag": f.quality_flag,
                "blur_score": f.blur_score,
                "phash": f.phash,
                "aug_quality": f.aug_quality,
                "aug_seed": f.aug_seed,
                "parent_frame_id": f.parent_frame_id,
                "is_anomaly_candidate": int(f.is_anomaly_candidate),
                "anomaly_score": f.anomaly_score,
                "annotation_status": f.annotation_status,
            }
            for f in frames
        ]
        try:
            self._conn.executemany(
                """
                INSERT INTO frames (
                    frame_id, video_id, source, frame_path, data_root,
                    timestamp_ms, species, breed, lighting, bowl_type,
                    quality_flag, blur_score, phash, aug_quality, aug_seed,
                    parent_frame_id, is_anomaly_candidate, anomaly_score,
                    annotation_status
                ) VALUES (
                    :frame_id, :video_id, :source, :frame_path, :data_root,
                    :timestamp_ms, :species, :breed, :lighting, :bowl_type,
                    :quality_flag, :blur_score, :phash, :aug_quality, :aug_seed,
                    :parent_frame_id, :is_anomaly_candidate, :anomaly_score,
                    :annotation_status
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
        self._conn.execute(
            "UPDATE frames SET quality_flag = ?, blur_score = ? WHERE frame_id = ?",
            (quality_flag, blur_score, frame_id),
        )
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
        """
        self._conn.execute(
            "UPDATE frames SET is_anomaly_candidate = ?, anomaly_score = ? WHERE frame_id = ?",
            (int(is_candidate), score, frame_id),
        )
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
        """
        self._conn.execute(
            "UPDATE frames SET annotation_status = ? WHERE frame_id = ?",
            (status, frame_id),
        )
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
        """
        self._conn.execute(
            "UPDATE frames SET aug_quality = ?, parent_frame_id = ? WHERE frame_id = ?",
            (aug_quality, parent_frame_id, frame_id),
        )
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
        """
        self._conn.execute(
            "UPDATE frames SET phash = ? WHERE frame_id = ?",
            (phash, frame_id),
        )
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

    def count_normal_frames(self) -> int:
        """Count frames with quality_flag='normal' that have not been rejected.

        Returns:
            Number of normal, non-rejected frames.
        """
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM frames
            WHERE quality_flag = 'normal'
              AND annotation_status NOT IN ('rejected')
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
        )
