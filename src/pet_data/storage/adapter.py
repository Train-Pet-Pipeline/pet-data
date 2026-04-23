"""Pure data mappers: sqlite rows -> pet_schema discriminated union types.

Keeping all mapping in one file ensures future schema changes touch a single adapter.

Lighting normalization policy (DB value -> pet_schema.Lighting):
    bright           -> bright
    dim              -> dim
    infrared_night   -> dark
    unknown          -> dim   (conservative default; matches "neither bright nor dark")

If the mapping needs to change, update _LIGHTING_MAP and the Task A4 policy table in
docs/superpowers/plans/2026-04-20-phase-2-data-annotation-plan.md in lockstep.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pet_schema.samples import AudioSample, SourceInfo, VisionSample


def _parse_timestamp_ms(ts_ms: int | None) -> datetime:
    """Convert milliseconds since epoch to a UTC datetime."""
    if ts_ms is None:
        raise ValueError("timestamp_ms is NULL; cannot derive captured_at")
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)  # noqa: UP017


_LIGHTING_MAP = {
    "bright": "bright",
    "dim": "dim",
    "infrared_night": "dark",
    "unknown": "dim",
}


def _normalize_lighting(db_value: str) -> str:
    """Map DB lighting values to pet_schema.Lighting enum names."""
    try:
        return _LIGHTING_MAP[db_value]
    except KeyError as e:
        raise ValueError(
            f"lighting={db_value!r} has no mapping to pet_schema.Lighting; "
            f"update _LIGHTING_MAP in adapter.py and document in Task A4 table"
        ) from e


def frame_row_to_vision_sample(row: Mapping[str, Any]) -> VisionSample:
    """Convert a video frame row from sqlite to VisionSample.

    Concept separation (Phase 3):
    - source_type comes from row["provenance_type"] (legal/compliance category)
    - ingester comes from row["source"] (which code produced this sample)

    Before this fix, row["source"] (the ingester_name) was passed directly as
    source_type, which would cause a ValidationError for non-youtube/community
    ingesters (oxford_pet, coco, selfshot, hospital, local_dir) because those
    strings are not valid SourceType literals.
    """
    return VisionSample(
        sample_id=row["frame_id"],
        storage_uri=row["storage_uri"],
        captured_at=_parse_timestamp_ms(row["timestamp_ms"]),
        source=SourceInfo(
            source_type=row["provenance_type"],
            ingester=row["source"],
            source_id=row.get("video_id") or row["frame_id"],
            license=None,
        ),
        pet_species=row.get("species"),
        frame_width=row["frame_width"],
        frame_height=row["frame_height"],
        lighting=_normalize_lighting(row["lighting"]),
        bowl_type=row.get("bowl_type"),
        blur_score=row["blur_score"],
        brightness_score=row["brightness_score"],
    )


def audio_row_to_audio_sample(row: Mapping[str, Any]) -> AudioSample:
    """Convert an audio clip row from sqlite to AudioSample."""
    captured = row["captured_at"]
    if isinstance(captured, str):
        captured = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    return AudioSample(
        sample_id=row["sample_id"],
        storage_uri=row["storage_uri"],
        captured_at=captured,
        source=SourceInfo(
            source_type=row["source_type"],
            source_id=row["source_id"],
            license=row.get("source_license"),
        ),
        pet_species=row.get("pet_species"),
        duration_s=row["duration_s"],
        sample_rate=row["sample_rate"],
        num_channels=row["num_channels"],
        snr_db=row.get("snr_db"),
        clip_type=row.get("clip_type"),
    )
