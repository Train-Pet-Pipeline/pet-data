"""Tests for storage adapter: DB rows -> pet_schema Sample types."""

import pytest
from pet_schema.samples import AudioSample, VisionSample

from pet_data.storage.adapter import audio_row_to_audio_sample, frame_row_to_vision_sample


def test_frame_row_to_vision_sample_roundtrip():
    row = {
        "frame_id": "sha256:abc",
        "source": "youtube",
        "video_id": "vid1",
        "frame_path": "frames/a.jpg",
        "data_root": "/data",
        "storage_uri": "local:///data/frames/a.jpg",
        "timestamp_ms": 123456,
        "species": "dog",
        "lighting": "bright",
        "bowl_type": "ceramic",
        "blur_score": 120.5,
        "brightness_score": 0.6,
        "frame_width": 1920,
        "frame_height": 1080,
    }
    vs = frame_row_to_vision_sample(row)
    assert isinstance(vs, VisionSample)
    assert vs.sample_id == "sha256:abc"
    assert vs.storage_uri.startswith("local://")
    assert vs.modality == "vision"
    assert VisionSample.model_validate(vs.model_dump()) == vs


def test_audio_row_to_audio_sample_roundtrip():
    row = {
        "sample_id": "sha256:def",
        "storage_uri": "local:///data/audio/bark.wav",
        "captured_at": "2026-04-21T12:00:00Z",
        "source_type": "community",
        "source_id": "esc50",
        "source_license": "CC-BY",
        "pet_species": "dog",
        "duration_s": 5.0,
        "sample_rate": 44100,
        "num_channels": 1,
        "snr_db": 22.5,
        "clip_type": "bark",
    }
    a = audio_row_to_audio_sample(row)
    assert isinstance(a, AudioSample)
    assert a.modality == "audio"
    assert AudioSample.model_validate(a.model_dump()) == a


def _frame_row(**overrides):
    """Helper to create a minimal valid frame row with overrides."""
    base = {
        "frame_id": "sha256:abc",
        "source": "youtube",
        "video_id": "vid1",
        "frame_path": "a.jpg",
        "data_root": "/d",
        "storage_uri": "local:///d/a.jpg",
        "timestamp_ms": 1,
        "species": "dog",
        "lighting": "bright",
        "bowl_type": "ceramic",
        "blur_score": 120.0,
        "brightness_score": 0.5,
        "frame_width": 1920,
        "frame_height": 1080,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "db_val,schema_val",
    [
        ("bright", "bright"),
        ("dim", "dim"),
        ("infrared_night", "dark"),
        ("unknown", "dim"),
    ],
)
def test_lighting_normalization(db_val, schema_val):
    """Verify lighting enum mapping policy from DB to schema."""
    vs = frame_row_to_vision_sample(_frame_row(lighting=db_val))
    assert vs.lighting == schema_val


def test_lighting_unknown_raises():
    """Unmapped lighting values should raise ValueError with clear message."""
    with pytest.raises(ValueError, match="no mapping to pet_schema.Lighting"):
        frame_row_to_vision_sample(_frame_row(lighting="strobe"))


def test_null_timestamp_raises():
    """NULL timestamp_ms on a frame row must raise ValueError, not crash opaquely."""
    with pytest.raises(ValueError, match="timestamp_ms is NULL"):
        frame_row_to_vision_sample(_frame_row(timestamp_ms=None))
