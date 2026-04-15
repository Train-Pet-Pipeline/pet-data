"""Reddit community source — scrapes public pet posts via PRAW."""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, cast

import requests as req

from pet_data.sources.base import BaseSource, RawItem, SourceMetadata
from pet_data.sources.extractors import AutoExtractor

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}


class CommunitySource(BaseSource):
    """Ingest pet images/videos from Reddit public posts.

    Expects:
    - params["reddit_subreddits"]: list of subreddit names
    - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT env vars
    - praw installed (optional dependency: pip install pet-data[community])
    """

    source_name = "community"

    def __init__(self, store, params: dict) -> None:
        """Initialize with AutoExtractor."""
        super().__init__(store, params)
        output_dir = Path(params.get("data_root", "/tmp")) / "frames" / "community"
        self.extractor = AutoExtractor(output_dir=output_dir)
        self.subreddits = params.get("reddit_subreddits", [])
        self.download_dir = Path(params.get("data_root", "/tmp")) / "raw" / "community"

    def download(self) -> Iterator[RawItem]:
        """Scrape public posts from configured subreddits."""
        try:
            import praw
        except ImportError:
            logger.error("praw not installed. Run: pip install pet-data[community]")
            return

        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        user_agent = os.environ.get("REDDIT_USER_AGENT", "pet-data-scraper/1.0")

        if not client_id or not client_secret:
            logger.error("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars required")
            return

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        self.download_dir.mkdir(parents=True, exist_ok=True)

        for sub_name in self.subreddits:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=100):
                if not post.url:
                    continue

                suffix = Path(post.url).suffix.lower()
                if suffix in IMAGE_EXTENSIONS:
                    resource_type = "image"
                elif suffix in VIDEO_EXTENSIONS:
                    resource_type = "video"
                else:
                    continue

                local_path = self.download_dir / f"{post.id}{suffix}"
                if not local_path.exists():
                    try:
                        resp = req.get(post.url, timeout=30)
                        resp.raise_for_status()
                        local_path.write_bytes(resp.content)
                    except Exception:
                        logger.exception("Failed to download: %s", post.url)
                        continue

                yield RawItem(
                    source=self.source_name,
                    resource_path=local_path,
                    resource_type=cast(Literal["video", "image"], resource_type),
                    metadata=SourceMetadata(
                        species=None,
                        breed=None,
                        lighting="unknown",
                        bowl_type=None,
                        device_model=None,
                        video_id=post.id,
                    ),
                )

    def validate_metadata(self, item: RawItem) -> bool:
        """Community data requires video_id (post ID)."""
        if not item.metadata.video_id:
            return False
        return True
