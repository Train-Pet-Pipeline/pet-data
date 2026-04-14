"""Tests for CommunitySource."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pet_data.sources.community import CommunitySource
from pet_data.storage.store import FrameStore


@pytest.fixture
def store(tmp_path: Path) -> FrameStore:
    """Create a temporary FrameStore."""
    return FrameStore(tmp_path / "test.db")


class TestCommunitySource:
    """Tests for CommunitySource."""

    def test_missing_praw_logs_error(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path
    ) -> None:
        """download() returns empty if praw not installed."""
        default_params["reddit_subreddits"] = ["cats"]
        default_params["data_root"] = str(tmp_data_root)
        with patch.dict("sys.modules", {"praw": None}):
            src = CommunitySource(store, default_params)
            items = list(src.download())
            assert items == []

    def test_missing_credentials_logs_error(
        self, store: FrameStore, default_params: dict, tmp_data_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """download() returns empty without Reddit credentials."""
        default_params["reddit_subreddits"] = ["cats"]
        default_params["data_root"] = str(tmp_data_root)
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        mock_praw = MagicMock()
        with patch.dict("sys.modules", {"praw": mock_praw}):
            src = CommunitySource(store, default_params)
            items = list(src.download())
            assert items == []
