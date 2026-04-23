"""Tests for BaseSource.default_provenance ClassVar (concept separation step 2)."""
from __future__ import annotations

import typing

import pytest

from pet_data.sources.base import BaseSource
from pet_data.sources.coco_pet import CocoPetSource
from pet_data.sources.community import CommunitySource
from pet_data.sources.hospital import HospitalSource
from pet_data.sources.local_dir import LocalDirSource
from pet_data.sources.oxford_pet import OxfordPetSource
from pet_data.sources.selfshot import SelfShotSource
from pet_data.sources.youtube import YoutubeSource

VALID_LITERALS = frozenset(
    typing.get_args(
        # SourceType is a type alias for Literal[...]; get the actual string values
        __import__("pet_schema.enums", fromlist=["SourceType"]).SourceType
    )
)

ALL_INGESTERS: list[type[BaseSource]] = [
    YoutubeSource,
    CommunitySource,
    SelfShotSource,
    OxfordPetSource,
    CocoPetSource,
    HospitalSource,
    LocalDirSource,
]

EXPECTED_MAPPING = {
    "youtube": "youtube",
    "community": "community",
    "selfshot": "community",
    "oxford_pet": "academic_dataset",
    "coco": "academic_dataset",
    "hospital": "device",
    "local_dir": "device",
}


def test_each_ingester_declares_valid_provenance():
    """Every BaseSource subclass must declare default_provenance as a valid SourceType literal."""
    for cls in ALL_INGESTERS:
        assert hasattr(cls, "default_provenance"), (
            f"{cls.__name__} missing 'default_provenance' ClassVar"
        )
        val = cls.default_provenance
        assert val in VALID_LITERALS, (
            f"{cls.__name__}.default_provenance={val!r} not in valid SourceType literals "
            f"{sorted(VALID_LITERALS)}"
        )


def test_provenance_mapping_matches_user_approved_spec():
    """Verify the user-approved ingester → default_provenance mapping (2026-04-23)."""
    for cls in ALL_INGESTERS:
        expected = EXPECTED_MAPPING[cls.ingester_name]
        actual = cls.default_provenance
        assert actual == expected, (
            f"{cls.__name__}: expected default_provenance={expected!r}, got {actual!r}"
        )
