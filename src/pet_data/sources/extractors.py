"""Frame extraction strategies."""
from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from pet_data.sources.base import RawItem

logger = logging.getLogger(__name__)


class FrameExtractor(ABC):
    """Abstract frame extraction strategy."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with output directory for extracted frames."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract(self, item: RawItem, params: dict) -> list[Path]:
        """Extract frames from a RawItem, return list of frame file paths."""


class ImageExtractor(FrameExtractor):
    """Extract frames from image sources — copy/convert to standard format."""

    def extract(self, item: RawItem, params: dict) -> list[Path]:
        """Copy image to output directory as PNG, return [path]."""
        dest = self.output_dir / f"{item.metadata.video_id}_{item.resource_path.stem}.png"
        if item.resource_path.suffix.lower() == ".png":
            shutil.copy2(item.resource_path, dest)
        else:
            from PIL import Image

            with Image.open(item.resource_path) as img:
                img.save(dest, format="PNG")
        logger.info("Extracted image: %s → %s", item.resource_path.name, dest.name)
        return [dest]


def _extract_with_decord(
    video_path: Path, fps: float, output_dir: Path, video_id: str,
) -> list[Path]:
    """Extract frames using decord (preferred, faster).

    Args:
        video_path: Path to the video file.
        fps: Target frames per second.
        output_dir: Directory to save extracted frames.
        video_id: Video identifier for filenames.

    Returns:
        List of extracted frame paths.
    """
    import numpy as np
    from decord import VideoReader
    from PIL import Image

    vr = VideoReader(str(video_path))
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration_sec = total_frames / video_fps

    num_frames = max(1, int(duration_sec * fps))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    extracted: list[Path] = []
    for i, idx in enumerate(indices):
        frame = vr[int(idx)].asnumpy()
        dest = output_dir / f"{video_id}_f{i:04d}.png"
        Image.fromarray(frame).save(dest)
        extracted.append(dest)

    return extracted


def _extract_with_av(video_path: Path, fps: float, output_dir: Path, video_id: str) -> list[Path]:
    """Extract frames using PyAV (fallback for environments where decord fails).

    Args:
        video_path: Path to the video file.
        fps: Target frames per second.
        output_dir: Directory to save extracted frames.
        video_id: Video identifier for filenames.

    Returns:
        List of extracted frame paths.
    """
    import av
    from PIL import Image

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate or stream.guessed_rate or 25)
    total_frames = stream.frames or 0

    # If frame count unknown, decode all frames
    if total_frames == 0:
        frames_list = []
        for frame in container.decode(video=0):
            frames_list.append(frame.to_ndarray(format="rgb24"))
        total_frames = len(frames_list)
        container.close()
    else:
        frames_list = None

    duration_sec = total_frames / video_fps
    num_frames = max(1, int(duration_sec * fps))

    import numpy as np

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    extracted: list[Path] = []
    if frames_list is not None:
        # Already decoded all frames
        for i, idx in enumerate(indices):
            dest = output_dir / f"{video_id}_f{i:04d}.png"
            Image.fromarray(frames_list[int(idx)]).save(dest)
            extracted.append(dest)
    else:
        # Decode on-the-fly, collecting only needed indices
        container = av.open(str(video_path))
        target_set = set(int(x) for x in indices)
        frame_idx = 0
        out_idx = 0
        for frame in container.decode(video=0):
            if frame_idx in target_set:
                arr = frame.to_ndarray(format="rgb24")
                dest = output_dir / f"{video_id}_f{out_idx:04d}.png"
                Image.fromarray(arr).save(dest)
                extracted.append(dest)
                out_idx += 1
            frame_idx += 1
        container.close()

    return extracted


class VideoExtractor(FrameExtractor):
    """Extract frames from video at configured fps.

    Tries decord first (faster), falls back to PyAV if decord is unavailable
    or fails (e.g. ffmpeg compatibility issues on macOS arm64).
    """

    def extract(self, item: RawItem, params: dict) -> list[Path]:
        """Extract frames at params['frames']['extract_fps'] rate."""
        fps = params["frames"]["extract_fps"]
        video_id = item.metadata.video_id

        # Try decord first
        try:
            extracted = _extract_with_decord(
                item.resource_path, fps, self.output_dir, video_id
            )
        except Exception as decord_err:
            logger.warning(
                "decord failed (%s), falling back to PyAV for %s",
                type(decord_err).__name__, item.resource_path.name,
            )
            try:
                extracted = _extract_with_av(
                    item.resource_path, fps, self.output_dir, video_id
                )
            except ImportError:
                raise RuntimeError(
                    "Neither decord nor av (PyAV) could extract video. "
                    "Install one of: pip install decord  OR  pip install av"
                ) from decord_err

        duration_sec = len(extracted) / fps if fps > 0 else 0
        logger.info(
            "Extracted %d frames from %s (%.1fs at %.1f fps)",
            len(extracted), item.resource_path.name, duration_sec, fps,
        )
        return extracted


class AutoExtractor(FrameExtractor):
    """Auto-dispatch to VideoExtractor or ImageExtractor based on resource_type."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with sub-extractors."""
        super().__init__(output_dir)
        self._video = VideoExtractor(output_dir)
        self._image = ImageExtractor(output_dir)

    def extract(self, item: RawItem, params: dict) -> list[Path]:
        """Route to appropriate extractor based on item.resource_type."""
        if item.resource_type == "video":
            return self._video.extract(item, params)
        return self._image.extract(item, params)
