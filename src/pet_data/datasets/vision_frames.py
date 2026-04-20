"""Dataset plugin exposing pet_data's vision frames as pet_schema.VisionSample iterator."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from mmengine.registry import Registry
from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS
from pet_schema import BaseSample

from pet_data.storage.adapter import frame_row_to_vision_sample

if TYPE_CHECKING:
    import datasets as hf_datasets

# Child registry with scope "pet_data" so DATASETS.get("pet_data.vision_frames") resolves.
_PET_DATA_DATASETS = Registry("dataset", scope="pet_data", parent=DATASETS)


@_PET_DATA_DATASETS.register_module(name="vision_frames", force=True)
class VisionFramesDataset(BaseDataset):
    """VisionSample iterator over the pet-data frames table.

    `dataset_config` keys:
        db_path: str — required, path to pet-data sqlite file
        modality_filter: str — optional, defaults to "vision"
    """

    def modality(self) -> Literal["vision", "audio", "sensor", "multimodal"]:
        """Return the modality handled by this dataset plugin."""
        return "vision"

    def build(self, dataset_config: dict) -> Iterable[BaseSample]:
        """Yield VisionSample objects from the frames table.

        Args:
            dataset_config: Must contain 'db_path'; optionally 'modality_filter'.

        Yields:
            VisionSample instances for rows with non-null required fields.
        """
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
        """Return a HuggingFace Dataset from the vision frames.

        Args:
            dataset_config: Passed directly to build().

        Returns:
            hf_datasets.Dataset with one row per VisionSample.
        """
        import datasets as hf_datasets

        records = [s.model_dump(mode="json") for s in self.build(dataset_config)]
        return hf_datasets.Dataset.from_list(records)
