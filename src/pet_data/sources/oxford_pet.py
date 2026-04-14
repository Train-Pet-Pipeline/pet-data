"""Oxford-IIIT Pet Dataset source."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import ImageExtractor

logger = logging.getLogger(__name__)

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class OxfordPetSource(BaseSource):
    """Ingest images from Oxford-IIIT Pet Dataset directory.

    Expects images in params["oxford_pet_dir"]/images/.
    Breed is parsed from filename (e.g. 'Abyssinian_001.jpg' -> breed='Abyssinian').
    Species inferred: first word capitalized -> cat, lowercase -> dog.
    """

    source_name = "oxford_pet"

    def __init__(self, store, params: dict) -> None:
        """Initialize with ImageExtractor."""
        super().__init__(store, params)
        oxford_dir = Path(params.get("oxford_pet_dir", ""))
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "oxford_pet"
        self.extractor = ImageExtractor(output_dir=output_dir)
        self.images_dir = oxford_dir / "images"

    def download(self) -> Iterator[RawItem]:
        """Scan images directory and yield RawItems with parsed metadata."""
        if not self.images_dir.exists():
            logger.warning("Oxford Pet images dir not found: %s", self.images_dir)
            return

        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            parts = img_path.stem.rsplit("_", 1)
            breed = parts[0] if len(parts) == 2 else img_path.stem
            species = "cat" if breed[0].isupper() else "dog"

            yield RawItem(
                source=self.source_name,
                resource_path=img_path,
                resource_type="image",
                metadata=SourceMetadata(
                    species=species,
                    breed=breed,
                    lighting="unknown",
                    bowl_type=None,
                    device_model=None,
                    video_id=img_path.stem,
                ),
            )

    def validate_metadata(self, item: RawItem) -> bool:
        """Oxford Pet data requires species and breed."""
        if not item.metadata.species or not item.metadata.breed:
            logger.warning("Missing species/breed: %s", item.resource_path.name)
            return False
        return True
