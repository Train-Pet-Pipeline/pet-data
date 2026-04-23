"""Entry-point target for pet-infra's plugin discovery.

pet-infra scans ``[project.entry-points."pet_infra.plugins"]`` and calls the
registered callable (named ``register_all``, matching pet-infra's own
convention — see ``pet_infra._register``) at CLI startup to trigger the
``@DATASETS.register_module`` side-effects in plugin modules.
"""
from __future__ import annotations

try:
    import pet_infra  # noqa: F401
except ImportError as e:
    raise ImportError(
        "pet-data requires pet-infra to be installed first. "
        "Install via 'pip install pet-infra @ "
        "git+https://github.com/Train-Pet-Pipeline/pet-infra@<tag>' "
        "using the tag pinned in pet-infra/docs/compatibility_matrix.yaml."
    ) from e


def register_all() -> None:
    """Import (or reload) pet-data plugin modules to trigger registration side-effects.

    Raises:
        RuntimeError: If pet-schema is not installed or its version module is missing.
            pet-schema is a required peer dependency; install it before calling
            register_all().  See pet-infra/docs/compatibility_matrix.yaml for the
            correct version tag to pin.
    """
    import importlib

    # Mode B (DEV_GUIDE §11.3): delayed fail-fast guard — checked at call time,
    # not at import time, to avoid circular import issues at startup.
    try:
        import pet_schema.version as _psv  # noqa: F401
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "pet-data requires pet-schema to be installed as a peer dependency. "
            "Install via: pip install 'pet-schema @ "
            "git+https://github.com/Train-Pet-Pipeline/pet-schema@<tag>' "
            "using the tag pinned in pet-infra/docs/compatibility_matrix.yaml. "
            f"Original error: {e}"
        ) from e

    from pet_data.datasets import audio_clips, vision_frames

    importlib.reload(vision_frames)
    importlib.reload(audio_clips)
