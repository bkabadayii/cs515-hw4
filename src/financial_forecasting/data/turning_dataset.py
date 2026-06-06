"""PyTorch Dataset for turning-point binary classification."""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


class TurningPointDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    """PyTorch Dataset for the turning-point classification task.

    Loads X (inputs), y (binary labels), max_future_return, tickers, and
    anchor_dates from a compressed NPZ file produced by
    build_turning_point_windows.py.

    Attributes
    ----------
    X:
        Feature tensor of shape (num_samples, T, num_features).
    y:
        Binary label tensor of shape (num_samples,) with dtype float32
        (required by BCEWithLogitsLoss).
    max_future_return:
        Float tensor of shape (num_samples,) holding the max observed
        max-price return over d = 1..5 (for analysis; not fed to model).
    tickers:
        List of ticker strings per sample.
    anchor_dates:
        List of anchor-date strings per sample.
    """

    def __init__(self, npz_path: str | pathlib.Path) -> None:
        """Load and convert data from compressed NPZ archive.

        Parameters
        ----------
        npz_path:
            Path to the turning-point NPZ split file.
        """
        data = np.load(npz_path, allow_pickle=True)

        self.X = torch.from_numpy(data["X"].astype(np.float32))
        # Labels stored as float32 for BCEWithLogitsLoss
        self.y = torch.from_numpy(data["y"].astype(np.float32))
        self.max_future_return = torch.from_numpy(
            data["max_future_return"].astype(np.float32)
        )
        self.tickers: list[str] = [str(t) for t in data["tickers"]]
        self.anchor_dates: list[str] = [str(d) for d in data["anchor_dates"]]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Get a single sample.

        Parameters
        ----------
        idx:
            Sample index.

        Returns
        -------
        tuple:
            (X_tensor, label_tensor, ticker, anchor_date)
        """
        return self.X[idx], self.y[idx], self.tickers[idx], self.anchor_dates[idx]
