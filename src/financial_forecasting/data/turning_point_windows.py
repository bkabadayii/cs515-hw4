"""Turning-point window builder.

Constructs binary buy/pass labels and sliding-window inputs for the
turning-point detection task (Phase 5).

Label definition (from the assignment)
---------------------------------------
    r_{t+d} = (p_max_{t+d} - p_t) / p_t

where
    p_max_{t+d} = High price at day t+d
    p_t         = Close price at anchor day t

    label = 1  (buy)   if  max_{d=1..5}  r_{t+d} > gamma
    label = 0  (pass)  otherwise

With gamma = 1.1 this requires a 110%+ gain in 5 trading days, so the
positive class is extremely rare in real data.  The builder faithfully
implements the formula and records the class balance statistics.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MetadataVal = Union[str, float, int, list[str], list[float]]


def build_turning_point_windows(
    df: pd.DataFrame,
    features: list[str],
    gamma: float = 1.1,
    horizons: list[int] | None = None,
    lookback_window: int = 20,
    use_high_price: bool = True,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int8],
    NDArray[np.float32],
    list[str],
    list[str],
]:
    """Construct sliding-window inputs and binary turning-point labels.

    Parameters
    ----------
    df:
        DataFrame with columns Date, Ticker, Open, High, Low, Close and
        all requested feature columns.  Must contain a Raw_Close column
        (unscaled Close) and, when use_high_price=True, a Raw_High column
        (unscaled High).
    features:
        Ordered list of feature column names to include in each window.
    gamma:
        Assignment threshold (default 1.1).
    horizons:
        Look-ahead horizons (default [1, 2, 3, 4, 5]).
    lookback_window:
        Length T of each input sequence (default 20).
    use_high_price:
        If True, use Raw_High for p_max; otherwise use Raw_Close.

    Returns
    -------
    X:
        Float32 array of shape (num_samples, lookback_window, num_features).
    y:
        Int8 binary label array of shape (num_samples,).
    max_future_return:
        Float32 array of shape (num_samples,) holding the maximum
        r_{t+d} observed over d = 1..5 (for analysis purposes).
    tickers:
        Ticker string per sample.
    anchor_dates:
        Anchor-date string per sample.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5]

    max_h = max(horizons)

    all_X: list[NDArray[np.float32]] = []
    all_y: list[int] = []
    all_max_r: list[float] = []
    all_tickers: list[str] = []
    all_dates: list[str] = []

    for ticker in sorted(df["Ticker"].unique()):
        ticker_df = (
            df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        )
        n = len(ticker_df)

        if n < lookback_window + max_h:
            continue

        f_matrix = ticker_df[features].to_numpy(dtype=np.float32)

        close_prices = ticker_df["Raw_Close"].to_numpy(dtype=np.float64)
        if use_high_price:
            high_col = "Raw_High"
        else:
            high_col = "Raw_Close"
        high_prices = ticker_df[high_col].to_numpy(dtype=np.float64)

        dates = ticker_df["Date"].tolist()

        start_idx = lookback_window - 1
        end_idx = n - max_h

        for idx in range(start_idx, end_idx):
            X_sample = f_matrix[idx - lookback_window + 1 : idx + 1]
            all_X.append(X_sample)

            p_t = close_prices[idx]

            # Compute max-price return for each horizon
            max_r = -np.inf
            for h in horizons:
                p_max = high_prices[idx + h]
                r = (p_max - p_t) / p_t
                if r > max_r:
                    max_r = r

            label = 1 if max_r > gamma else 0
            all_y.append(label)
            all_max_r.append(float(max_r))

            anchor_str = pd.to_datetime(dates[idx]).strftime("%Y-%m-%d")
            all_tickers.append(ticker)
            all_dates.append(anchor_str)

    if not all_X:
        X = np.empty((0, lookback_window, len(features)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int8)
        max_future_return = np.empty((0,), dtype=np.float32)
    else:
        X = np.stack(all_X, axis=0)
        y = np.array(all_y, dtype=np.int8)
        max_future_return = np.array(all_max_r, dtype=np.float32)

    return X, y, max_future_return, all_tickers, all_dates


def compute_class_balance(y: NDArray[np.int8]) -> dict[str, float]:
    """Compute binary class balance statistics.

    Parameters
    ----------
    y:
        Binary label array (0 or 1).

    Returns
    -------
    dict:
        Keys: total_samples, num_positive, num_negative, positive_rate,
        negative_rate.
    """
    total = len(y)
    num_pos = int(np.sum(y == 1))
    num_neg = total - num_pos
    pos_rate = num_pos / total if total > 0 else 0.0
    return {
        "total_samples": total,
        "num_positive": num_pos,
        "num_negative": num_neg,
        "positive_rate": pos_rate,
        "negative_rate": 1.0 - pos_rate,
    }
