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
        """Copy image to output directory, return [path]."""
        dest = self.output_dir / f"{item.metadata.video_id}_{item.resource_path.stem}.png"
        if item.resource_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            shutil.copy2(item.resource_path, dest)
        else:
            from PIL import Image

            img = Image.open(item.resource_path)
            img.save(dest, format="PNG")
        logger.info("Extracted image: %s → %s", item.resource_path.name, dest.name)
        return [dest]


class VideoExtractor(FrameExtractor):
    """Extract frames from video using decord at configured fps."""

    def extract(self, item: RawItem, params: dict) -> list[Path]:
        """Extract frames at params['frames']['extract_fps'] rate."""
        import numpy as np
        from decord import VideoReader
        from PIL import Image

        fps = params["frames"]["extract_fps"]
        vr = VideoReader(str(item.resource_path))
        video_fps = vr.get_avg_fps()
        total_frames = len(vr)
        duration_sec = total_frames / video_fps

        # Calculate frame indices at desired fps
        num_frames = max(1, int(duration_sec * fps))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        extracted: list[Path] = []
        for i, idx in enumerate(indices):
            frame = vr[int(idx)].asnumpy()
            dest = self.output_dir / f"{item.metadata.video_id}_f{i:04d}.png"
            Image.fromarray(frame).save(dest)
            extracted.append(dest)

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
