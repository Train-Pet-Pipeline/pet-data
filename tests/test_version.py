import pet_data
import pet_schema
import pet_infra


def test_pet_data_version():
    assert pet_data.__version__ == "1.1.0"


def test_pet_data_pins_phase1_foundation():
    assert pet_schema.version.SCHEMA_VERSION == "2.0.0"
    assert pet_infra.__version__.startswith("2.")
