"""YouTube video source — downloads pet videos via yt-dlp."""
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import VideoExtractor

logger = logging.getLogger(__name__)


class YoutubeSource(BaseSource):
    """Ingest pet videos from YouTube URLs.

    Expects:
    - params["youtube_urls_file"]: path to text file, one URL per line
    - yt-dlp installed (optional dependency: pip install pet-data[youtube])
    """

    source_name = "youtube"

    def __init__(self, store, params: dict) -> None:
        """Initialize with VideoExtractor."""
        super().__init__(store, params)
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "youtube"
        self.extractor = VideoExtractor(output_dir=output_dir)
        self.urls_file = Path(params.get("youtube_urls_file", ""))
        self.download_dir = Path(params.get("data_root", "/tmp")) / "raw" / "youtube"

    def download(self) -> Iterator[RawItem]:
        """Download videos via yt-dlp and yield RawItems."""
        try:
            import yt_dlp
        except ImportError:
            logger.error("yt-dlp not installed. Run: pip install pet-data[youtube]")
            return

        if not self.urls_file.exists():
            logger.warning("URLs file not found: %s", self.urls_file)
            return

        self.download_dir.mkdir(parents=True, exist_ok=True)
        urls = [line.strip() for line in self.urls_file.read_text().splitlines() if line.strip()]

        for url in urls:
            video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url.split("/")[-1]
            output_path = self.download_dir / f"{video_id}.mp4"

            if not output_path.exists():
                ydl_opts = {
                    "outtmpl": str(output_path),
                    "format": "best[height<=720]",
                    "quiet": True,
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([url])
                except Exception:
                    logger.exception("Failed to download: %s", url)
                    continue

            if output_path.exists():
                yield RawItem(
                    source=self.source_name,
                    resource_path=output_path,
                    resource_type="video",
                    metadata=SourceMetadata(
                        species=None,
                        breed=None,
                        lighting="unknown",
                        bowl_type=None,
                        device_model=None,
                        video_id=video_id,
                    ),
                )

    def validate_metadata(self, item: RawItem) -> bool:
        """YouTube data requires video_id."""
        if not item.metadata.video_id:
            return False
        return True
