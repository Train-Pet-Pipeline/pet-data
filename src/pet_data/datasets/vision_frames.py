"""Dataset plugin exposing pet_data's vision frames as pet_schema.VisionSample iterator."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS
from pet_schema import BaseSample

from pet_data.storage.adapter import frame_row_to_vision_sample

if TYPE_CHECKING:
    import datasets as hf_datasets


@DATASETS.register_module(name="pet_data.vision_frames", force=True)
class VisionFramesDataset(BaseDataset):
    """VisionSample iterator over the pet-data frames table.

    `dataset_config` keys:
        db_path: str — required, path to pet-data sqlite file
        modality_filter: str — optional, defaults to "vision"

    Registered as the flat key ``pet_data.vision_frames`` in
    :data:`pet_infra.registry.DATASETS`. Lookup via ``DATASETS.module_dict``
    (mmengine's ``.get()`` parses dots as ``scope.module`` and returns None
    for flat dotted names — this matches the preflight lookup pattern in
    pet-infra v2.0.0).
    """

    def modality(self) -> Literal["vision", "audio", "sensor", "multimodal"]:
        """Return the modality handled by this dataset plugin."""
        return "vision"

    def build(self, dataset_config: dict) -> Iterable[BaseSample]:
        """Yield VisionSample objects from the frames table."""
        db_path = dataset_config["db_path"]
        mfilter = dataset_config.get("modality_filter", "vision")
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT * FROM frames WHERE modality = ? "
                "AND frame_width IS NOT NULL "
                "AND frame_height IS NOT NULL "
                "AND brightness_score IS NOT NULL",
                (mfilter,),
            )
            for row in cur.fetchall():
                yield frame_row_to_vision_sample(dict(row))
        finally:
            conn.close()

    def to_hf_dataset(self, dataset_config: dict) -> hf_datasets.Dataset:
        """Return a HuggingFace Dataset materialised from :meth:`build`."""
        import datasets as hf_datasets

        records = [s.model_dump(mode="json") for s in self.build(dataset_config)]
        return hf_datasets.Dataset.from_list(records)
