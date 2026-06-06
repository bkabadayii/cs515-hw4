"""PyTorch dataset implementation for financial sequences."""

from __future__ import annotations

import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    """PyTorch Dataset loading inputs, targets, tickers, and anchor dates from NPZ.

    Attributes
    ----------
    X:
        Feature input tensor of shape (num_samples, T, num_features).
    y:
        Forecasting targets of shape (num_samples, output_size).
    tickers:
        List of ticker strings per sequence.
    anchor_dates:
        List of anchor dates (strings) per sequence.
    """

    def __init__(self, npz_path: str | pathlib.Path) -> None:
        """Load and convert data from compressed NPZ archive.

        Parameters
        ----------
        npz_path:
            Path to the saved NPZ split file.
        """
        # Set allow_pickle=True since ticker/date arrays are object arrays
        data = np.load(npz_path, allow_pickle=True)

        # Convert feature and target arrays to float32 tensors
        self.X = torch.from_numpy(data["X"].astype(np.float32))
        self.y = torch.from_numpy(data["y"].astype(np.float32))

        # Convert object arrays back to list of strings
        self.tickers = [str(t) for t in data["tickers"]]
        self.anchor_dates = [str(d) for d in data["anchor_dates"]]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Get a single data sample.

        Parameters
        ----------
        idx:
            Index of the sample to fetch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, str, str]:
            (X, y, ticker, anchor_date)
        """
        return self.X[idx], self.y[idx], self.tickers[idx], self.anchor_dates[idx]
