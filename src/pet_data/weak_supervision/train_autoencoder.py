"""Autoencoder training for weak supervision anomaly detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pet_data.storage.store import FrameFilter, FrameStore
from pet_data.weak_supervision._image_util import load_and_normalize

logger = logging.getLogger(__name__)


class FeedingAutoencoder(nn.Module):
    """Convolutional autoencoder trained on normal feeding frames.

    Input: (B, 3, 224, 224) normalized to [-1,1].
    Only trained on normal feeding frames. High reconstruction error = anomaly candidate.
    """

    def __init__(self) -> None:
        """Initialize the encoder and decoder networks."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # -> 112x112
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # -> 56x56
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # -> 28x28
            nn.ReLU(),
            nn.Conv2d(128, 32, 4, stride=2, padding=1),  # -> 14x14 (bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run full encode-decode pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224) normalized to [-1, 1].

        Returns:
            Reconstructed tensor of same shape as input.
        """
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction MSE as anomaly score.

        Args:
            x: Input tensor of shape (B, 3, 224, 224) normalized to [-1, 1].

        Returns:
            1D tensor of shape (B,) with per-sample MSE scores.
        """
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2, 3))  # per-sample MSE


class _FrameDataset(Dataset):
    """Dataset loading frame images from absolute paths, normalized to [-1, 1]."""

    def __init__(self, paths: list[str]) -> None:
        """Initialize dataset with list of absolute image paths.

        Args:
            paths: List of absolute file paths to frame images.
        """
        self._paths = paths

    def __len__(self) -> int:
        """Return number of frames in the dataset."""
        return len(self._paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and normalize a single frame.

        Args:
            idx: Index of the frame to load.

        Returns:
            Tensor of shape (3, 224, 224) normalized to [-1, 1].
        """
        return load_and_normalize(self._paths[idx])


@dataclass
class TrainReport:
    """Summary of an autoencoder training run.

    Attributes:
        model_path: Path to the saved model state dict.
        epochs: Number of epochs actually trained.
        final_train_loss: MSE loss on the training set at the last epoch.
        final_val_loss: MSE loss on the validation set at the last epoch.
    """

    model_path: Path
    epochs: int
    final_train_loss: float
    final_val_loss: float


def train(store: FrameStore, params: dict, output_dir: Path) -> TrainReport:
    """Train a FeedingAutoencoder on normal frames from the store.

    Args:
        store: FrameStore to query for normal frames.
        params: Full params dict (reads from params["weak_supervision"]).
        output_dir: Directory to save the trained model.

    Returns:
        TrainReport with path, epoch count and final losses.

    Raises:
        ValueError: If the number of normal frames is below min_normal_frames.
    """
    ws_params = params["weak_supervision"]
    min_frames: int = ws_params["min_normal_frames"]
    max_epochs: int = ws_params["max_epochs"]
    batch_size: int = ws_params["batch_size"]
    learning_rate: float = ws_params["learning_rate"]

    count = store.count_normal_frames()
    logger.info('{"event": "train_autoencoder_start", "normal_frame_count": %d}', count)

    if count < min_frames:
        raise ValueError(
            f"min_normal_frames: need {min_frames} normal frames, found {count}"
        )

    records = store.query_frames(
        FrameFilter(quality_flag="normal", is_anomaly_candidate=False, limit=count + 1)
    )
    paths = [r.frame_path for r in records]

    # Shuffle deterministically before 80/20 split to avoid temporal bias
    rng = np.random.default_rng(42)
    rng.shuffle(paths)
    torch.manual_seed(42)
    split = int(len(paths) * 0.8)
    train_paths = paths[:split]
    val_paths = paths[split:]

    train_loader = DataLoader(
        _FrameDataset(train_paths),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        _FrameDataset(val_paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = FeedingAutoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    final_train_loss = 0.0
    final_val_loss = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        final_train_loss = train_loss / len(train_paths)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * len(batch)
        final_val_loss = val_loss / max(len(val_paths), 1)

        logger.info(
            '{"event": "train_epoch", "epoch": %d, "train_loss": %.6f, "val_loss": %.6f}',
            epoch,
            final_train_loss,
            final_val_loss,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "autoencoder.pt"
    torch.save(model.state_dict(), model_path)

    logger.info(
        '{"event": "train_autoencoder_done", "model_path": "%s", "epochs": %d}',
        model_path,
        max_epochs,
    )

    return TrainReport(
        model_path=model_path,
        epochs=max_epochs,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
    )
