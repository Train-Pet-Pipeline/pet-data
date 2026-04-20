"""Ensure params.yaml carries the Phase 2 sample defaults."""
from pathlib import Path

import yaml


def test_params_has_sample_defaults():
    params = yaml.safe_load(
        (Path(__file__).parent.parent / "params.yaml").read_text()
    )
    assert params["sample"]["default_modality"] == "vision"
    assert params["sample"]["storage_scheme"] == "local"
