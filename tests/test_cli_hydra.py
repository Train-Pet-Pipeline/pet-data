"""Hydra config composition tests for pet-data config group."""
from pathlib import Path

from hydra import compose, initialize_config_dir

CFG_DIR = str((Path(__file__).parent.parent / "src" / "pet_data" / "configs").resolve())


def test_compose_dataset_vision_frames():
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_data_ingest",
            overrides=["dataset=vision_frames"],
        )
    assert cfg.dataset.type == "pet_data.vision_frames"
    assert cfg.dataset.modality == "vision"


def test_compose_override_audio():
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_data_ingest",
            overrides=["dataset=audio_clips"],
        )
    assert cfg.dataset.type == "pet_data.audio_clips"
    assert cfg.dataset.modality == "audio"
