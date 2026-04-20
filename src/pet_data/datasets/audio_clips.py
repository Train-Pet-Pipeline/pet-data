"""AudioClipsDataset plugin — mirrors VisionFramesDataset for the audio modality."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

from pet_infra.base.dataset import BaseDataset
from pet_infra.registry import DATASETS
from pet_schema.samples import AudioSample

from pet_data.storage.adapter import audio_row_to_audio_sample
from pet_data.storage.store import AudioStore

if TYPE_CHECKING:
    import datasets as hf_datasets


@DATASETS.register_module(name="pet_data.audio_clips", force=True)
class AudioClipsDataset(BaseDataset):
    """Iterate the pet-data ``audio_samples`` table and yield ``pet_schema.AudioSample``.

    Registered under the flat-dotted key ``pet_data.audio_clips`` so that
    ``DATASETS.module_dict[...]`` (the lookup used by pet-infra preflight)
    finds it directly. Do NOT rely on ``DATASETS.get(...)`` — mmengine parses
    the dot as ``scope.module`` and returns ``None`` for flat dotted keys.
    """

    def modality(self) -> Literal["audio"]:
        """Return the modality handled by this plugin."""
        return "audio"

    def build(self, dataset_config: dict) -> Iterable[AudioSample]:
        """Yield :class:`AudioSample` objects from the ``audio_samples`` table.

        Args:
            dataset_config: Must contain ``db_path``.

        Yields:
            :class:`AudioSample` per row. ``AudioStore.query()`` returns
            :class:`AudioSampleRow` dataclass instances; the adapter expects
            a ``Mapping``, so convert via :func:`dataclasses.asdict`.
        """
        db_path = dataset_config["db_path"]
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            store = AudioStore(conn)
            for row in store.query():
                yield audio_row_to_audio_sample(asdict(row))
        finally:
            conn.close()

    def to_hf_dataset(self, dataset_config: dict) -> hf_datasets.Dataset:
        """Return a HuggingFace Dataset materialised from :meth:`build`."""
        import datasets as hf_datasets

        rows = list(self.build(dataset_config))
        return hf_datasets.Dataset.from_list([r.model_dump(mode="json") for r in rows])
