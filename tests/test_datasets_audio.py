import sqlite3

import pytest
from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS
from pet_schema.samples import AudioSample

from tests.conftest_migrations import load_migration


@pytest.fixture
def fresh_db_with_audio(tmp_path):
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(str(db))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    from pet_data.storage.store import AudioSampleRow, AudioStore
    store = AudioStore(conn)
    store.insert(AudioSampleRow(
        sample_id="a1", storage_uri="local:///tmp/a1.wav",
        captured_at="2026-04-21T12:00:00+00:00",
        source_type="community", source_id="esc50", source_license="CC-BY",
        pet_species="dog", duration_s=2.5, sample_rate=16000, num_channels=1,
        snr_db=20.0, clip_type="bark",
    ))
    conn.close()
    return db


def test_audio_clips_registered_and_ABC():  # noqa: N802
    from pet_data.datasets import audio_clips  # noqa: F401
    assert "pet_data.audio_clips" in DATASETS.module_dict
    cls = DATASETS.module_dict["pet_data.audio_clips"]
    inst = cls()
    assert isinstance(inst, BaseDataset)
    assert inst.modality() == "audio"


def test_audio_clips_build_yields_AudioSample(fresh_db_with_audio):  # noqa: N802
    from pet_data.datasets import audio_clips  # noqa: F401
    cls = DATASETS.module_dict["pet_data.audio_clips"]
    ds = cls()
    samples = list(ds.build({"db_path": str(fresh_db_with_audio)}))
    assert len(samples) == 1
    assert isinstance(samples[0], AudioSample)
    assert samples[0].modality == "audio"
    assert samples[0].duration_s == 2.5
