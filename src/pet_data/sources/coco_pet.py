"""COCO dataset source — filters cat/dog images from local COCO directory."""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import ImageExtractor

logger = logging.getLogger(__name__)

COCO_CAT_ID = 17
COCO_DOG_ID = 18
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class CocoPetSource(BaseSource):
    """Ingest pet images from a local COCO dataset.

    Expects:
    - params["coco_dir"] pointing to COCO root (with images/ and annotations/)
    - Reads annotations/instances_train2017.json for cat (id=17) / dog (id=18)
    """

    ingester_name = "coco"
    default_provenance = "academic_dataset"

    def __init__(self, store, params: dict) -> None:
        """Initialize with ImageExtractor."""
        super().__init__(store, params)
        coco_dir = Path(params.get("coco_dir", ""))
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "coco"
        self.extractor = ImageExtractor(output_dir=output_dir)
        self.coco_dir = coco_dir

    def download(self) -> Iterator[RawItem]:
        """Parse COCO annotations, yield images containing cat or dog."""
        ann_path = self.coco_dir / "annotations" / "instances_train2017.json"
        if not ann_path.exists():
            logger.warning("COCO annotations not found: %s", ann_path)
            return

        with open(ann_path) as f:
            coco = json.load(f)

        pet_images: dict[int, str] = {}
        for ann in coco.get("annotations", []):
            cat_id = ann["category_id"]
            if cat_id == COCO_CAT_ID:
                pet_images[ann["image_id"]] = "cat"
            elif cat_id == COCO_DOG_ID:
                pet_images[ann["image_id"]] = "dog"

        id_to_file = {img["id"]: img["file_name"] for img in coco.get("images", [])}

        images_dir = self.coco_dir / "images" / "train2017"
        for image_id, species in pet_images.items():
            filename = id_to_file.get(image_id)
            if not filename:
                continue
            img_path = images_dir / filename
            if not img_path.exists():
                continue

            yield RawItem(
                source=self.ingester_name,
                resource_path=img_path,
                resource_type="image",
                metadata=SourceMetadata(
                    species=species,
                    breed=None,
                    lighting="unknown",
                    bowl_type=None,
                    device_model=None,
                    video_id=f"coco_{image_id}",
                ),
            )

    def validate_metadata(self, item: RawItem) -> bool:
        """COCO data requires species."""
        if not item.metadata.species:
            logger.warning("Missing species: %s", item.resource_path.name)
            return False
        return True
