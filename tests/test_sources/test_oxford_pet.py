"""Tests for OxfordPetSource."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pet_data.sources.base import RawItem, SourceMetadata
from pet_data.sources.oxford_pet import OxfordPetSource
from pet_data.storage.store import FrameStore


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore."""
    return FrameStore(tmp_path / "test.db")


@pytest.fixture
def oxford_dir(tmp_data_root: Path) -> Path:
    """Create a fake Oxford Pet dataset directory."""
    d = tmp_data_root / "oxford_pet" / "images"
    d.mkdir(parents=True)
    rng = np.random.default_rng(42)
    for name in ["Abyssinian_001.jpg", "Bengal_002.jpg", "pug_003.jpg"]:
        arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / name)
    return tmp_data_root / "oxford_pet"


class TestOxfordPetSource:
    """Tests for OxfordPetSource."""

    def test_download_yields_images(
        self, store: FrameStore, default_params: dict, oxford_dir: Path, tmp_data_root: Path
    ) -> None:
        """download() yields RawItems for each image."""
        default_params["oxford_pet_dir"] = str(oxford_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = OxfordPetSource(store, default_params)
        items = list(src.download())
        assert len(items) == 3
        assert all(i.resource_type == "image" for i in items)

    def test_species_inferred_from_filename(
        self, store: FrameStore, default_params: dict, oxford_dir: Path, tmp_data_root: Path
    ) -> None:
        """Species is inferred: capitalized first letter = cat, lowercase = dog."""
        default_params["oxford_pet_dir"] = str(oxford_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = OxfordPetSource(store, default_params)
        items = list(src.download())
        species_map = {i.metadata.breed: i.metadata.species for i in items}
        assert species_map["Abyssinian"] == "cat"
        assert species_map["pug"] == "dog"

    def test_validate_requires_species_and_breed(
        self, store: FrameStore, default_params: dict, oxford_dir: Path, tmp_data_root: Path
    ) -> None:
        """validate_metadata returns False if species or breed is missing."""
        default_params["oxford_pet_dir"] = str(oxford_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = OxfordPetSource(store, default_params)
        item = RawItem(
            source="oxford_pet",
            resource_path=Path("fake.jpg"),
            resource_type="image",
            metadata=SourceMetadata(
                species=None, breed=None, lighting=None,
                bowl_type=None, device_model=None, video_id="fake",
            ),
        )
        assert src.validate_metadata(item) is False
