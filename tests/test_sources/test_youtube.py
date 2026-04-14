"""Tests for YoutubeSource."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from pet_data.sources.base import RawItem, SourceMetadata
from pet_data.sources.youtube import YoutubeSource
from pet_data.storage.store import FrameStore


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore."""
    return FrameStore(tmp_path / "test.db")


class TestYoutubeSource:
    """Tests for YoutubeSource."""

    def test_missing_ytdlp_logs_error(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path
    ) -> None:
        """download() returns empty if yt-dlp not installed."""
        default_params["youtube_urls_file"] = "/nonexistent"
        default_params["data_root"] = str(tmp_data_root)
        with patch.dict("sys.modules", {"yt_dlp": None}):
            src = YoutubeSource(store, default_params)
            items = list(src.download())
            assert items == []

    def test_validate_requires_video_id(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path
    ) -> None:
        """validate_metadata returns False without video_id."""
        default_params["data_root"] = str(tmp_data_root)
        src = YoutubeSource(store, default_params)
        item = RawItem(
            source="youtube",
            resource_path=Path("f.mp4"),
            resource_type="video",
            metadata=SourceMetadata(
                species=None,
                breed=None,
                lighting=None,
                bowl_type=None,
                device_model=None,
                video_id="",
            ),
        )
        assert src.validate_metadata(item) is False
