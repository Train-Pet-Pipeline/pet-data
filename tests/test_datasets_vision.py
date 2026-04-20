"""Tests for VisionFramesDataset plugin registration and behaviour."""
from __future__ import annotations

from pet_infra.registry import DATASETS

import pet_data.datasets.vision_frames  # noqa: F401  (trigger registration)


def test_vision_frames_dataset_registered():
    assert "pet_data.vision_frames" in DATASETS.module_dict


def test_vision_frames_build_yields_vision_samples(fresh_db_with_frames):
    from pet_schema.samples import VisionSample
    cls = DATASETS.module_dict["pet_data.vision_frames"]
    ds = cls()
    samples = list(ds.build({"db_path": str(fresh_db_with_frames), "modality_filter": "vision"}))
    assert len(samples) == 3
    assert all(isinstance(s, VisionSample) for s in samples)


def test_vision_frames_modality_method(fresh_db_with_frames):
    cls = DATASETS.module_dict["pet_data.vision_frames"]
    ds = cls()
    assert ds.modality() == "vision"


def test_vision_frames_skips_rows_missing_required_fields(tmp_path):
    """Rows missing frame_width/height/brightness_score (NULL) must be skipped, not raise."""
    from pet_data.storage.store import FrameRecord, FrameStore
    store = FrameStore(tmp_path / "db.sqlite")
    store.insert_frame(FrameRecord(
        frame_id="incomplete", video_id="v", source="youtube",
        frame_path="x.jpg", data_root="/data",
        timestamp_ms=0, species="dog", lighting="bright",
        bowl_type=None, quality_flag="normal", blur_score=50.0,
        modality="vision", storage_uri="local:///data/x.jpg",
        frame_width=None, frame_height=None, brightness_score=None,
    ))
    store.close()
    cls = DATASETS.module_dict["pet_data.vision_frames"]
    ds = cls()
    assert list(ds.build({"db_path": str(tmp_path/"db.sqlite")})) == []
