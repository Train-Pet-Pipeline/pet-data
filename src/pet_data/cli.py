"""pet-data CLI.

Legacy subcommands (ingest/dedup/quality/augment/train-ae/score-anomaly) are
retained for DVC compatibility. New ``run`` command dispatches via Hydra +
pet-infra's Recipe resolver.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
from hydra import compose, initialize_config_dir

from pet_data import cli_legacy

CFG_DIR = str((Path(__file__).parent / "configs").resolve())


@click.group()
def cli() -> None:
    """pet-data CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}',
    )


@cli.command()
@click.option("--source", required=True, help="Source name")
@click.option("--params", "params_path", type=click.Path(path_type=Path), default=None)
def ingest(source: str, params_path: Path | None) -> None:
    """Ingest data from a source."""
    cli_legacy.run_ingest(source=source, params_path=params_path)


@cli.command()
@click.option("--params", "params_path", type=click.Path(path_type=Path), default=None)
def dedup(params_path: Path | None) -> None:
    """Run dedup pass."""
    cli_legacy.run_dedup(params_path=params_path)


@cli.command()
@click.option("--params", "params_path", type=click.Path(path_type=Path), default=None)
def quality(params_path: Path | None) -> None:
    """Run quality assessment."""
    cli_legacy.run_quality(params_path=params_path)


@cli.command()
@click.option("--params", "params_path", type=click.Path(path_type=Path), default=None)
def augment(params_path: Path | None) -> None:
    """Run augmentation pipeline."""
    cli_legacy.run_augment(params_path=params_path)


@cli.command("train-ae")
@click.option("--params", "params_path", type=click.Path(path_type=Path), default=None)
def train_ae(params_path: Path | None) -> None:
    """Train the anomaly detection autoencoder."""
    cli_legacy.run_train_ae(params_path=params_path)


@cli.command("score-anomaly")
@click.option("--params", "params_path", type=click.Path(path_type=Path), default=None)
def score_anomaly(params_path: Path | None) -> None:
    """Score frames for anomaly detection."""
    cli_legacy.run_score_anomaly(params_path=params_path)


@cli.command()
@click.option(
    "--config-name", required=True, help="Hydra config name, e.g. experiment/pet_data_ingest"
)
@click.option("--dry-run/--no-dry-run", default=True, help="Compose + validate, skip execution")
@click.argument("overrides", nargs=-1)
def run(config_name: str, dry_run: bool, overrides: tuple[str, ...]) -> None:
    """Hydra-composed run.

    --dry-run composes and echoes the recipe id only. Real execution lives in
    pet-infra's ``pet run recipe=...`` — this command is a compose-time smoke test.
    """
    with initialize_config_dir(CFG_DIR, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    # config_name may be "experiment/pet_data_ingest" — Hydra nests _self_ under the
    # sub-directory key, so recipe lives at cfg.experiment.recipe when the name has a
    # path component.  Fall back gracefully if recipe is at root (flat config names).
    recipe_node = cfg
    for part in config_name.split("/")[:-1]:
        recipe_node = recipe_node[part]
    click.echo(f"Composed: {recipe_node.recipe.recipe_id}")
    if dry_run:
        return
    raise click.UsageError(
        "Non-dry-run execution is owned by pet-infra's `pet run`. "
        "Use: pet run recipe=<recipe-name>"
    )


def main() -> None:
    """Entry point for pet-data console script."""
    cli()


if __name__ == "__main__":
    main()
