"""Shared helper to load numbered migration modules by filename."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

MIGRATIONS_DIR = Path(__file__).parent.parent / "src" / "pet_data" / "storage" / "migrations"


def load_migration(n: int) -> ModuleType:
    """Load migrations/NNN_*.py by its leading 3-digit number.

    Args:
        n: The migration number (1, 2, etc.).

    Returns:
        The loaded migration module with upgrade() and downgrade() functions.

    Raises:
        FileNotFoundError: If exactly one matching file is not found.
    """
    matches = list(MIGRATIONS_DIR.glob(f"{n:03d}_*.py"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one migration file for {n:03d}, got {matches}"
        )
    path = matches[0]
    spec = importlib.util.spec_from_file_location(f"migration_{n:03d}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
