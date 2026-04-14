"""Video generation for data augmentation via REST-based generative models."""
from __future__ import annotations

import abc
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from pet_data.storage.store import FrameFilter, FrameStore

logger = logging.getLogger(__name__)


class VideoGenerator(abc.ABC):
    """Abstract base class for video generation backends."""

    @abc.abstractmethod
    def generate(self, seed_image: Path, prompt: str, seed: int) -> Path | None:
        """Generate a video from a seed image and text prompt.

        Args:
            seed_image: Path to the seed frame image.
            prompt: Text prompt describing the desired video content.
            seed: RNG seed for reproducibility.

        Returns:
            Path to the generated video file, or None on failure.
        """


class NullGenerator(VideoGenerator):
    """No-op generator that always returns None.  Useful for testing."""

    def generate(self, seed_image: Path, prompt: str, seed: int) -> Path | None:
        """Return None unconditionally.

        Args:
            seed_image: Ignored.
            prompt: Ignored.
            seed: Ignored.

        Returns:
            Always None.
        """
        return None


class Wan21Generator(VideoGenerator):
    """Video generator that calls a Wan2.1 REST endpoint."""

    def __init__(self, endpoint: str, timeout: int = 60) -> None:
        """Initialise with endpoint URL and request timeout.

        Args:
            endpoint: Full URL of the generation API.
            timeout: HTTP request timeout in seconds.
        """
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, seed_image: Path, prompt: str, seed: int) -> Path | None:
        """Call the Wan2.1 API to generate a video.

        Args:
            seed_image: Path to the seed frame image.
            prompt: Text prompt for video generation.
            seed: RNG seed for reproducibility.

        Returns:
            Path to the saved video file, or None after retries are exhausted.
        """
        try:
            return self._call_api(seed_image, prompt, seed)
        except RetryError:
            logger.warning(
                '{"event": "video_gen_exhausted", "seed_image": "%s"}',
                seed_image,
            )
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1))
    def _call_api(self, seed_image: Path, prompt: str, seed: int) -> Path:
        """Send the generation request with tenacity retry.

        Args:
            seed_image: Path to the seed frame image.
            prompt: Text prompt for video generation.
            seed: RNG seed for reproducibility.

        Returns:
            Path to the saved video file.

        Raises:
            requests.HTTPError: On non-2xx response status.
            ConnectionError: On network-level failures.
        """
        with open(seed_image, "rb") as f:
            resp = requests.post(
                self.endpoint,
                files={"image": f},
                data={"prompt": prompt, "seed": str(seed)},
                timeout=self.timeout,
            )
        resp.raise_for_status()

        suffix = ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(resp.content)
        tmp.close()
        result_path = Path(tmp.name)
        logger.info(
            '{"event": "video_generated", "path": "%s"}',
            result_path,
        )
        return result_path


@dataclass
class AugmentReport:
    """Summary statistics for an augmentation run.

    Attributes:
        generated: Number of videos successfully generated.
        failed: Number of generation attempts that failed.
    """

    generated: int = field(default=0)
    failed: int = field(default=0)


def run_augmentation(
    store: FrameStore,
    params: dict,
    generator: VideoGenerator,
) -> AugmentReport:
    """Orchestrate video-based augmentation over seed frames in the store.

    Queries the store for seed frames (quality_flag=normal, annotation_status=pending),
    then calls the generator for each seed the configured number of times.

    Args:
        store: FrameStore instance for querying seed frames.
        params: Pipeline parameters (must contain augmentation.video_gen_count_per_seed).
        generator: VideoGenerator backend to use.

    Returns:
        AugmentReport summarising generation outcomes.
    """
    report = AugmentReport()
    count_per_seed = params["augmentation"]["video_gen_count_per_seed"]

    seeds = store.query_frames(
        FrameFilter(quality_flag="normal", annotation_status="pending")
    )
    logger.info('{"event": "run_augmentation_start", "seed_count": %d}', len(seeds))

    for frame in seeds:
        frame_path = Path(frame.data_root) / frame.frame_path
        for i in range(count_per_seed):
            result = generator.generate(frame_path, prompt="pet feeding scene", seed=i)
            if result is not None:
                report.generated += 1
            else:
                report.failed += 1

    logger.info(
        '{"event": "run_augmentation_done", "generated": %d, "failed": %d}',
        report.generated,
        report.failed,
    )
    return report
