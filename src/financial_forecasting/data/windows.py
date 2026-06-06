"""Sliding window generator for time-series datasets.

Supports input sequence lookback windows and future exact-return target extraction.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Define a strict type for metadata values
MetadataVal = Union[str, float, list[str], list[float]]


def build_sliding_windows(
    df: pd.DataFrame,
    features: list[str],
    lookback_window: int = 20,
    horizons: list[int] = [1, 2, 3, 4, 5],
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, MetadataVal]]]:
    """Construct sequence features (X), return targets (y), and metadata.

    Parameters
    ----------
    df:
        DataFrame containing Date, Ticker, Close, and all feature columns.
    features:
        List of feature columns to include in the lookback window.
    lookback_window:
        The length of each lookback sequence (T = 20).
    horizons:
        Trading day prediction horizons (d = 1..5).

    Returns
    -------
    X:
        Numpy float32 array of shape (num_samples, T, num_features).
    y:
        Numpy float32 array of shape (num_samples, D) where D = len(horizons).
    metadata:
        List of sample descriptions including ticker and dates.
    """
    all_X: list[NDArray[np.float32]] = []
    all_y: list[NDArray[np.float32]] = []
    all_metadata: list[dict[str, MetadataVal]] = []

    # Group by ticker to maintain independence and chronology
    tickers = sorted(df["Ticker"].unique())
    for ticker in tickers:
        ticker_df = (
            df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        )
        n_rows = len(ticker_df)
        max_h = max(horizons)

        if n_rows < lookback_window + max_h:
            continue

        f_matrix = ticker_df[features].to_numpy(dtype=np.float32)
        close_col = "Raw_Close" if "Raw_Close" in ticker_df.columns else "Close"
        close_prices = ticker_df[close_col].to_numpy(dtype=np.float32)
        dates = ticker_df["Date"].tolist()

        start_idx = lookback_window - 1
        end_idx = n_rows - max_h

        for idx in range(start_idx, end_idx):
            X_sample = f_matrix[idx - lookback_window + 1 : idx + 1]
            all_X.append(X_sample)

            p_t = close_prices[idx]
            y_sample = np.zeros(len(horizons), dtype=np.float32)
            target_dates: list[str] = []
            target_closes: list[float] = []

            for i, h in enumerate(horizons):
                p_td = close_prices[idx + h]
                y_sample[i] = (p_td - p_t) / p_t
                target_dates.append(
                    pd.to_datetime(dates[idx + h]).strftime("%Y-%m-%d")
                )
                target_closes.append(float(p_td))

            all_y.append(y_sample)

            anchor_date_str = pd.to_datetime(dates[idx]).strftime("%Y-%m-%d")
            all_metadata.append(
                {
                    "ticker": ticker,
                    "anchor_date": anchor_date_str,
                    "target_dates": target_dates,
                    "raw_anchor_close": float(p_t),
                    "raw_target_closes": target_closes,
                }
            )

    if not all_X:
        X = np.empty((0, lookback_window, len(features)), dtype=np.float32)
        y = np.empty((0, len(horizons)), dtype=np.float32)
    else:
        X = np.stack(all_X, axis=0)
        y = np.stack(all_y, axis=0)

    return X, y, all_metadata
