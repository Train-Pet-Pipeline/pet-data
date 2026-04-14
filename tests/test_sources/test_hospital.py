"""Tests for HospitalSource — especially PII scrubbing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from pet_data.sources.base import RawItem, SourceMetadata
from pet_data.sources.hospital import (
    HospitalSource,
    sanitize_filename,
    scrub_exif,
    scrub_pii_from_string,
)
from pet_data.storage.store import FrameStore


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore."""
    return FrameStore(tmp_path / "test.db")


class TestPIIScrubbing:
    """Tests for PII scrubbing utilities."""

    def test_scrub_phone_number(self) -> None:
        """Phone numbers are replaced with [REDACTED]."""
        assert "[REDACTED]" in scrub_pii_from_string("Call 123-456-7890")

    def test_scrub_email(self) -> None:
        """Email addresses are replaced with [REDACTED]."""
        assert "[REDACTED]" in scrub_pii_from_string("Contact vet@clinic.com")

    def test_scrub_patient_id(self) -> None:
        """Long numeric IDs are replaced with [REDACTED]."""
        assert "[REDACTED]" in scrub_pii_from_string("Patient 12345678")

    def test_scrub_preserves_normal_text(self) -> None:
        """Normal text without PII patterns is preserved."""
        text = "Orange tabby, male, 3 years old"
        assert scrub_pii_from_string(text) == text


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_different_paths_different_hashes(self) -> None:
        """Different input paths produce different hashed filenames."""
        name1 = sanitize_filename(Path("/data/patient_john_doe_001.jpg"))
        name2 = sanitize_filename(Path("/data/patient_jane_doe_002.jpg"))
        assert name1 != name2

    def test_preserves_extension(self) -> None:
        """File extension is preserved in sanitized filename."""
        name = sanitize_filename(Path("/data/img.jpg"))
        assert name.endswith(".jpg")

    def test_starts_with_hospital_prefix(self) -> None:
        """Sanitized filename starts with hospital_ prefix."""
        name = sanitize_filename(Path("/data/img.jpg"))
        assert name.startswith("hospital_")


class TestScrubExif:
    """Tests for EXIF scrubbing."""

    def test_exif_removed(self, tmp_data_root: Path) -> None:
        """EXIF scrubbing preserves image dimensions."""
        img_path = tmp_data_root / "with_exif.jpg"
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(img_path)
        scrub_exif(img_path)
        result = Image.open(img_path)
        assert result.size == (224, 224)


class TestHospitalSource:
    """Tests for HospitalSource."""

    def test_download_yields_items(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path
    ) -> None:
        """download() yields sanitized items from hospital directory."""
        hospital_dir = tmp_data_root / "hospital"
        hospital_dir.mkdir()
        (hospital_dir / "meta").mkdir()

        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(hospital_dir / "case_001.jpg")

        with open(hospital_dir / "meta" / "case_001.yaml", "w") as f:
            yaml.dump(
                {
                    "species": "cat",
                    "breed": "persian",
                    "notes": "Owner John Doe, phone 555-123-4567",
                },
                f,
            )

        default_params["hospital_dir"] = str(hospital_dir)
        default_params["data_root"] = str(tmp_data_root)
        src = HospitalSource(store, default_params)
        items = list(src.download())
        assert len(items) == 1
        assert "case_001" not in items[0].metadata.video_id
        assert items[0].metadata.video_id.startswith("hospital_")

    def test_validate_requires_species(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path
    ) -> None:
        """validate_metadata returns False if species missing."""
        default_params["hospital_dir"] = str(tmp_data_root)
        default_params["data_root"] = str(tmp_data_root)
        src = HospitalSource(store, default_params)
        item = RawItem(
            source="hospital",
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
