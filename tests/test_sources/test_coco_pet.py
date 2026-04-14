"""Tests for CocoPetSource."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pet_data.sources.base import RawItem, SourceMetadata
from pet_data.sources.coco_pet import CocoPetSource
from pet_data.storage.store import FrameStore


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore."""
    return FrameStore(tmp_path / "test.db")


@pytest.fixture
def coco_dir(tmp_data_root: Path) -> Path:
    """Create a fake COCO directory with minimal annotations."""
    d = tmp_data_root / "coco"
    (d / "annotations").mkdir(parents=True)
    (d / "images" / "train2017").mkdir(parents=True)

    annotations = {
        "images": [
            {"id": 1, "file_name": "000001.jpg"},
            {"id": 2, "file_name": "000002.jpg"},
            {"id": 3, "file_name": "000003.jpg"},
        ],
        "annotations": [
            {"image_id": 1, "category_id": 17},  # cat
            {"image_id": 2, "category_id": 18},  # dog
            {"image_id": 3, "category_id": 1},   # not pet
        ],
    }
    with open(d / "annotations" / "instances_train2017.json", "w") as f:
        json.dump(annotations, f)

    rng = np.random.default_rng(42)
    for name in ["000001.jpg", "000002.jpg", "000003.jpg"]:
        arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / "images" / "train2017" / name)

    return d


class TestCocoPetSource:
    """Tests for CocoPetSource."""

    def test_download_yields_only_pet_images(
        self, store: FrameStore, default_params: dict, coco_dir: Path,
        tmp_data_root: Path,
    ) -> None:
        """download() only yields cat/dog images, not other categories."""
        default_params["coco_dir"] = str(coco_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = CocoPetSource(store, default_params)
        items = list(src.download())
        assert len(items) == 2

    def test_species_correctly_assigned(
        self, store: FrameStore, default_params: dict, coco_dir: Path,
        tmp_data_root: Path,
    ) -> None:
        """Species is correctly assigned based on category ID."""
        default_params["coco_dir"] = str(coco_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = CocoPetSource(store, default_params)
        items = list(src.download())
        species_set = {i.metadata.species for i in items}
        assert species_set == {"cat", "dog"}

    def test_validate_requires_species(
        self, store: FrameStore, default_params: dict, coco_dir: Path,
        tmp_data_root: Path,
    ) -> None:
        """validate_metadata returns False if species missing."""
        default_params["coco_dir"] = str(coco_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = CocoPetSource(store, default_params)
        item = RawItem(
            source="coco",
            resource_path=Path("f.jpg"),
            resource_type="image",
            metadata=SourceMetadata(
                species=None,
                breed=None,
                lighting=None,
                bowl_type=None,
                device_model=None,
                video_id="x",
            ),
        )
        assert src.validate_metadata(item) is False
