import importlib.metadata as md

from pet_infra.registry import DATASETS


def test_pet_data_entry_point_discoverable():
    eps = md.entry_points(group="pet_infra.plugins")
    names = {ep.name for ep in eps}
    assert "pet_data" in names


def test_register_all_registers_both_datasets():
    """Simulate pet-infra discovery invoking register_all() on a fresh registry state."""
    DATASETS.module_dict.pop("pet_data.vision_frames", None)
    DATASETS.module_dict.pop("pet_data.audio_clips", None)
    from pet_data import _register

    _register.register_all()
    assert "pet_data.vision_frames" in DATASETS.module_dict
    assert "pet_data.audio_clips" in DATASETS.module_dict
