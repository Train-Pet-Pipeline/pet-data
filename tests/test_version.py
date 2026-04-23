import importlib.metadata

import pet_infra
import pet_schema.version

import pet_data


def test_pet_data_version():
    assert pet_data.__version__ == "1.2.0"


def test_pet_data_pins_phase1_foundation():
    assert pet_schema.version.SCHEMA_VERSION == "3.1.0"
    assert pet_infra.__version__.startswith("2.")


def test_pet_data_version_matches_pyproject():
    """__version__ in __init__.py must match the version in pyproject.toml.

    Uses importlib.metadata.version() which reads from the installed package
    metadata, ensuring pyproject.toml and __init__.py stay in sync.
    Mirrors the parity test pattern used in pet-schema Phase 1 and pet-infra Phase 2.
    """
    installed = importlib.metadata.version("pet-data")
    assert pet_data.__version__ == installed
