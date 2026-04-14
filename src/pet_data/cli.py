"""Unified CLI entry point for pet-data pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_params(params_path: Path | None = None) -> dict:
    """Load params.yaml from project root or specified path."""
    if params_path is None:
        params_path = Path(__file__).parent.parent.parent / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run ingest for a specific source."""
    from pet_data.sources.coco_pet import CocoPetSource
    from pet_data.sources.community import CommunitySource
    from pet_data.sources.hospital import HospitalSource
    from pet_data.sources.oxford_pet import OxfordPetSource
    from pet_data.sources.selfshot import SelfShotSource
    from pet_data.sources.youtube import YoutubeSource
    from pet_data.storage.store import FrameStore

    params = load_params(args.params)

    source_map = {
        "selfshot": SelfShotSource,
        "oxford_pet": OxfordPetSource,
        "coco": CocoPetSource,
        "youtube": YoutubeSource,
        "community": CommunitySource,
        "hospital": HospitalSource,
    }

    if args.source not in source_map:
        logger.error("Unknown source: %s. Available: %s", args.source, list(source_map.keys()))
        sys.exit(1)

    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        src = source_map[args.source](store, params)
        report = src.ingest()
    logger.info("Ingest complete: %s", report)


def cmd_dedup(args: argparse.Namespace) -> None:
    """Run dedup pass — compute missing phashes and detect cross-source duplicates."""
    from pet_data.processing.dedup import compute_phash, dedup_check
    from pet_data.storage.store import FrameFilter, FrameStore

    params = load_params(args.params)

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

                # Check newly hashed frame against all existing phashes
                result = dedup_check(frame_path, existing_phashes, params)
                if result.is_duplicate and result.duplicate_of != frame.frame_id:
                    store.update_quality(frame.frame_id, "low", frame.blur_score or 0.0)
                    duplicates += 1

    logger.info(
        '{"event": "dedup_pass", "updated": %d, "duplicates": %d}', updated, duplicates
    )


def cmd_quality(args: argparse.Namespace) -> None:
    """Run quality assessment on frames with blur_score=NULL."""
    from pet_data.processing.quality_filter import assess_quality
    from pet_data.storage.store import FrameFilter, FrameStore

    params = load_params(args.params)

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


def cmd_augment(args: argparse.Namespace) -> None:
    """Run augmentation pipeline."""
    import os

    from pet_data.augmentation.video_gen import NullGenerator, Wan21Generator, run_augmentation
    from pet_data.storage.store import FrameStore

    params = load_params(args.params)

    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        endpoint = os.environ.get("WAN21_ENDPOINT")
        generator = Wan21Generator(endpoint=endpoint) if endpoint else NullGenerator()
        report = run_augmentation(store, params, generator=generator)
    logger.info("Augmentation complete: %s", report)


def cmd_train_ae(args: argparse.Namespace) -> None:
    """Train the anomaly detection autoencoder."""
    from pet_data.storage.store import FrameStore
    from pet_data.weak_supervision.train_autoencoder import train

    params = load_params(args.params)

    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        output_dir = Path(params["data_root"]) / "models"
        report = train(store, params, output_dir)
    logger.info("Training complete: %s", report)


def cmd_score_anomaly(args: argparse.Namespace) -> None:
    """Score frames for anomaly detection."""
    from pet_data.storage.store import FrameStore
    from pet_data.weak_supervision.score_anomaly import score_frames

    params = load_params(args.params)

    with FrameStore(Path(params["data_root"]) / "frames.db") as store:
        model_path = Path(params["data_root"]) / "models" / "autoencoder.pt"
        report = score_frames(store, model_path, params)
    logger.info("Scoring complete: %s", report)


def main() -> None:
    """Main CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(name)s","msg":"%(message)s"}',
    )

    parser = argparse.ArgumentParser(prog="pet-data", description="Pet data pipeline CLI")
    parser.add_argument("--params", type=Path, default=None, help="Path to params.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest data from a source")
    p_ingest.add_argument("--source", required=True, help="Source name")
    p_ingest.set_defaults(func=cmd_ingest)

    p_dedup = sub.add_parser("dedup", help="Run dedup pass")
    p_dedup.set_defaults(func=cmd_dedup)

    p_quality = sub.add_parser("quality", help="Run quality assessment")
    p_quality.set_defaults(func=cmd_quality)

    p_augment = sub.add_parser("augment", help="Run augmentation pipeline")
    p_augment.set_defaults(func=cmd_augment)

    p_train = sub.add_parser("train-ae", help="Train autoencoder")
    p_train.set_defaults(func=cmd_train_ae)

    p_score = sub.add_parser("score-anomaly", help="Score anomaly candidates")
    p_score.set_defaults(func=cmd_score_anomaly)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
