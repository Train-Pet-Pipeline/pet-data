"""Tests for peer-dep fail-fast guards in _register.py."""
from __future__ import annotations

import sys
import types

import pytest


def test_register_all_raises_when_pet_schema_version_missing(monkeypatch):
    """register_all() must raise RuntimeError with 'pet-schema' when
    pet_schema.version.SCHEMA_VERSION is inaccessible.

    Simulates a broken/missing pet-schema version sub-module by patching
    sys.modules["pet_schema.version"] to None, which causes any import of
    that sub-module to raise ImportError.  register_all()'s Mode-B guard
    must catch this and re-raise as RuntimeError("pet-schema ...").

    We avoid patching the top-level pet_schema module because pet_infra
    depends on it at import time and would break unrelated tests.
    """
    import pet_data._register as reg

    monkeypatch.setitem(sys.modules, "pet_schema.version", None)

    with pytest.raises(RuntimeError, match="pet-schema"):
        reg.register_all()
