"""Hydra config composition tests for pet-data config group."""
from pathlib import Path

from hydra import compose, initialize_config_dir

CFG_DIR = str((Path(__file__).parent.parent / "src" / "pet_data" / "configs").resolve())


def test_compose_dataset_vision_frames():
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_data_ingest",
            overrides=["dataset=vision_frames"],
        )
    assert cfg.dataset.type == "pet_data.vision_frames"
    assert cfg.dataset.modality == "vision"


def test_compose_override_audio():
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="experiment/pet_data_ingest",
            overrides=["dataset=audio_clips"],
        )
    assert cfg.dataset.type == "pet_data.audio_clips"
    assert cfg.dataset.modality == "audio"


def test_cli_run_smokes_without_error(tmp_path, monkeypatch):
    """run --dry-run must compose config and exit 0."""
    monkeypatch.setenv("PET_DATA_DB", str(tmp_path / "db.sqlite"))
    import sqlite3

    from tests.conftest_migrations import load_migration

    conn = sqlite3.connect(str(tmp_path / "db.sqlite"))
    load_migration(1).upgrade(conn)
    load_migration(2).upgrade(conn)
    load_migration(3).upgrade(conn)
    conn.close()

    from click.testing import CliRunner

    from pet_data.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, ["run", "--config-name=experiment/pet_data_ingest", "--dry-run"]
    )
    assert result.exit_code == 0, result.output


def test_cli_ingest_help_still_works():
    """Legacy subcommand must still be invocable (DVC back-compat)."""
    from click.testing import CliRunner

    from pet_data.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "source" in result.output.lower()
