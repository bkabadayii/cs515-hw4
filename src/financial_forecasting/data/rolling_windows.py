"""Rolling-average sliding window generator for time-series datasets.

Computes rolling-average future return targets instead of exact-return targets.
The lookback input windows (X) are identical to those used for exact-return
forecasting; only the target (y) computation changes.

Rolling target convention (Phase 4):
    rolling_window = 3, weights = [1/3, 1/3, 1/3]

    For anchor t and horizon d:
        r_roll_{t+d} = sum_{j=0}^{rolling_window-1} weights[j] * (close[t+d-j] - close[t]) / close[t]

    This requires close prices at indices t+d, t+d-1, ..., t+d-(rolling_window-1).
    For d=1 and rolling_window=3: indices t+1, t, t-1.
    Since lookback_window >= 20, close[t-1] is always available in the ticker series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Metadata value types (must be JSON-serializable scalars or lists)
MetadataVal = str | float | list[str] | list[float]


def build_rolling_windows(
    df: pd.DataFrame,
    features: list[str],
    lookback_window: int = 20,
    horizons: list[int] = [1, 2, 3, 4, 5],
    rolling_window: int = 3,
    weights: list[float] = [1 / 3, 1 / 3, 1 / 3],
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, MetadataVal]]]:
    """Construct input windows (X) and rolling-average return targets (y).

    The input lookback window X is identical to the exact-return version.
    The target y is a weighted rolling average of exact returns ending at each
    horizon d, using only prices from the same ticker's contiguous series.

    Parameters
    ----------
    df:
        DataFrame containing Date, Ticker, Raw_Close (or Close), and feature columns.
        Must be sorted by Ticker then Date.
    features:
        Ordered list of input feature column names.
    lookback_window:
        Number of past trading days per input sample (T = 20).
    horizons:
        Prediction horizons d = 1..5.
    rolling_window:
        Number of terms in the rolling average (l = 3).
    weights:
        Per-term weights; len(weights) == rolling_window and sum(weights) == 1.0.

    Returns
    -------
    X:
        Input array of shape (num_samples, lookback_window, num_features), float32.
    y:
        Rolling-average target array of shape (num_samples, len(horizons)), float32.
    metadata:
        List of sample dicts with ticker, anchor_date, target_dates, raw_anchor_close,
        raw_target_closes, rolling_window, weights.

    Raises
    ------
    ValueError:
        If len(weights) != rolling_window or weights do not sum to approximately 1.0.
    """
    if len(weights) != rolling_window:
        raise ValueError(
            f"len(weights)={len(weights)} must equal rolling_window={rolling_window}"
        )
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0, got {weight_sum:.6f}"
        )

    all_X: list[NDArray[np.float32]] = []
    all_y: list[NDArray[np.float32]] = []
    all_metadata: list[dict[str, MetadataVal]] = []

    weights_arr = np.array(weights, dtype=np.float64)
    tickers = sorted(df["Ticker"].unique())

    for ticker in tickers:
        ticker_df = (
            df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        )
        n_rows = len(ticker_df)
        max_h = max(horizons)

        # Need at least lookback_window + max_h rows
        if n_rows < lookback_window + max_h:
            continue

        f_matrix = ticker_df[features].to_numpy(dtype=np.float32)
        close_col = "Raw_Close" if "Raw_Close" in ticker_df.columns else "Close"
        close_prices = ticker_df[close_col].to_numpy(dtype=np.float64)
        dates = ticker_df["Date"].tolist()

        # anchor index range: need t >= lookback_window - 1 (for lookback)
        # and t + max_h < n_rows (for future close prices)
        # and t - (rolling_window - 1) >= 0 (for rolling window going back)
        # -> t >= max(lookback_window - 1, rolling_window - 1)
        start_idx = max(lookback_window - 1, rolling_window - 1)
        end_idx = n_rows - max_h

        for idx in range(start_idx, end_idx):
            # Input lookback window (same as exact-return)
            X_sample = f_matrix[idx - lookback_window + 1 : idx + 1]
            all_X.append(X_sample)

            p_t = close_prices[idx]
            y_sample = np.zeros(len(horizons), dtype=np.float32)
            target_dates: list[str] = []
            target_closes: list[float] = []

            for i, h in enumerate(horizons):
                # Rolling average: average of exact returns at (t+h, t+h-1, ..., t+h-rolling_window+1)
                r_roll = 0.0
                for j in range(rolling_window):
                    future_idx = idx + h - j
                    # Safety guard: future_idx must be valid
                    if future_idx < 0 or future_idx >= n_rows:
                        r_roll = float("nan")
                        break
                    p_future = close_prices[future_idx]
                    r_roll += float(weights_arr[j]) * (p_future - p_t) / p_t

                y_sample[i] = np.float32(r_roll)
                target_dates.append(
                    pd.to_datetime(dates[idx + h]).strftime("%Y-%m-%d")
                )
                target_closes.append(float(close_prices[idx + h]))

            all_y.append(y_sample)

            anchor_date_str = pd.to_datetime(dates[idx]).strftime("%Y-%m-%d")
            all_metadata.append(
                {
                    "ticker": ticker,
                    "anchor_date": anchor_date_str,
                    "target_dates": target_dates,
                    "raw_anchor_close": float(p_t),
                    "raw_target_closes": target_closes,
                    "rolling_window": float(rolling_window),
                    "weights": list(weights),
                }
            )

    if not all_X:
        X = np.empty((0, lookback_window, len(features)), dtype=np.float32)
        y = np.empty((0, len(horizons)), dtype=np.float32)
    else:
        X = np.stack(all_X, axis=0)
        y = np.stack(all_y, axis=0)

    return X, y, all_metadata
