import importlib
import sys

import pytest


def test_register_raises_friendly_error_if_pet_infra_missing(monkeypatch):
    """没装 pet-infra 时 import pet_data._register 必须抛带安装指引的 ImportError。"""
    monkeypatch.setitem(sys.modules, "pet_infra", None)
    if "pet_data._register" in sys.modules:
        del sys.modules["pet_data._register"]

    with pytest.raises(ImportError) as excinfo:
        importlib.import_module("pet_data._register")

    msg = str(excinfo.value)
    assert "pet-infra" in msg
    assert "git+https://github.com/Train-Pet-Pipeline/pet-infra" in msg
    assert "compatibility_matrix" in msg
