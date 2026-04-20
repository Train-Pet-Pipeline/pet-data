"""Legacy CLI handlers extracted from cli.py for Click-wrapped delegation.

Bodies preserved verbatim from the pre-Hydra argparse CLI so DVC stage
contracts stay byte-compatible.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_params(params_path: Path | None = None) -> dict:
    """Load params.yaml from project root or specified path.

    Args:
        params_path: Optional explicit path to params.yaml.

    Returns:
        Parsed params dict.
    """
    if params_path is None:
        params_path = Path(__file__).parent.parent.parent / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def run_ingest(source: str, params_path: Path | None) -> None:
    """Run ingest for a specific source.

    Args:
        source: Source name key (e.g. 'selfshot', 'oxford_pet').
        params_path: Optional path to params.yaml.
    """
    from pet_data.sources.coco_pet import CocoPetSource
    from pet_data.sources.community import CommunitySource
    from pet_data.sources.hospital import HospitalSource
    from pet_data.sources.local_dir import LocalDirSource
    from pet_data.sources.oxford_pet import OxfordPetSource
    from pet_data.sources.selfshot import SelfShotSource
    from pet_data.sources.youtube import YoutubeSource
    from pet_data.storage.store import FrameStore

    params = load_params(params_path)
    source_map = {
        "selfshot": SelfShotSource,
        "oxford_pet": OxfordPetSource,
        "coco": CocoPetSource,
        "youtube": YoutubeSource,
        "community": CommunitySource,
        "hospital": HospitalSource,
        "local_dir": LocalDirSource,
    }
    if source not in source_map:
        logger.error("Unknown source: %s. Available: %s", source, list(source_map.keys()))
        sys.exit(1)
    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        src = source_map[source](store, params)  # type: ignore[abstract]
        report = src.ingest()
    logger.info("Ingest complete: %s", report)


def run_dedup(params_path: Path | None) -> None:
    """Run dedup pass — compute missing phashes and detect cross-source duplicates.

    Args:
        params_path: Optional path to params.yaml.
    """
    from pet_data.processing.dedup import compute_phash, dedup_check
    from pet_data.storage.store import FrameFilter, FrameStore

    params = load_params(params_path)
    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        all_frames = store.query_frames(FrameFilter(limit=100000))
        existing_phashes = store.get_phashes()
        updated = 0
        duplicates = 0
        for frame in all_frames:
            frame_path = Path(frame.data_root) / frame.frame_path
            if not frame_path.exists():
                continue
            if frame.frame_id not in existing_phashes:
                phash = compute_phash(frame_path)
                store.update_phash(frame.frame_id, phash)
                existing_phashes[frame.frame_id] = phash
                updated += 1
                result = dedup_check(frame_path, existing_phashes, params)
                if result.is_duplicate and result.duplicate_of != frame.frame_id:
                    store.update_quality(frame.frame_id, "low", frame.blur_score or 0.0)
                    duplicates += 1
    logger.info('{"event": "dedup_pass", "updated": %d, "duplicates": %d}', updated, duplicates)


def run_quality(params_path: Path | None) -> None:
    """Run quality assessment on frames with blur_score=NULL.

    Args:
        params_path: Optional path to params.yaml.
    """
    from pet_data.processing.quality_filter import assess_quality
    from pet_data.storage.store import FrameFilter, FrameStore

    params = load_params(params_path)
    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        all_frames = store.query_frames(FrameFilter(limit=100000))
        assessed = 0
        for frame in all_frames:
            if frame.blur_score is not None:
                continue
            frame_path = Path(frame.data_root) / frame.frame_path
            if not frame_path.exists():
                continue
            result = assess_quality(frame_path, params)
            store.update_quality(frame.frame_id, result.quality_flag, result.blur_score)
            assessed += 1
    logger.info("Quality pass: assessed %d frames", assessed)


def run_augment(params_path: Path | None) -> None:
    """Run augmentation pipeline.

    Args:
        params_path: Optional path to params.yaml.
    """
    import os

    from pet_data.augmentation.video_gen import NullGenerator, Wan21Generator, run_augmentation
    from pet_data.storage.store import FrameStore

    params = load_params(params_path)
    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        endpoint = os.environ.get("WAN21_ENDPOINT")
        generator = Wan21Generator(endpoint=endpoint) if endpoint else NullGenerator()
        report = run_augmentation(store, params, generator=generator)
    logger.info("Augmentation complete: %s", report)


def run_train_ae(params_path: Path | None) -> None:
    """Train the anomaly detection autoencoder.

    Args:
        params_path: Optional path to params.yaml.
    """
    from pet_data.storage.store import FrameStore
    from pet_data.weak_supervision.train_autoencoder import train

    params = load_params(params_path)
    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        output_dir = Path(params["data_root"]) / "models"
        report = train(store, params, output_dir)
    logger.info("Training complete: %s", report)


def run_score_anomaly(params_path: Path | None) -> None:
    """Score frames for anomaly detection.

    Args:
        params_path: Optional path to params.yaml.
    """
    from pet_data.storage.store import FrameStore
    from pet_data.weak_supervision.score_anomaly import score_frames

    params = load_params(params_path)
    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        model_path = Path(params["data_root"]) / "models" / "autoencoder.pt"
        report = score_frames(store, model_path, params)
    logger.info("Scoring complete: %s", report)
