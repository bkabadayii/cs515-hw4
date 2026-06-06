"""Auxiliary feature engineering for stock return forecasting.

All features are derived strictly from data available at or before the
anchor day t.  No future values are used.  Rolling window features
(SMA ratios, volatility) are computed per-ticker on the sorted raw
DataFrame before scaling.

Functions in this module accept and return pandas DataFrames and are
designed to be called before scaler fitting so that derived features
can themselves be scaled on the training split only.
"""

from __future__ import annotations

import pandas as pd


def compute_auxiliary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute auxiliary features from raw OHLC data per ticker.

    Operations performed per ticker (in chronological order):
    1. close_to_close_return  = (Close[t] - Close[t-1]) / Close[t-1]
    2. open_to_close_return   = (Close[t] - Open[t])  / Open[t]
    3. high_low_range_pct     = (High[t]  - Low[t])   / Low[t]
    4. close_sma_5_ratio      = Close[t] / rolling_mean(Close, 5)
    5. close_sma_10_ratio     = Close[t] / rolling_mean(Close, 10)
    6. rolling_volatility_5   = rolling_std(close_to_close_return, 5)

    Parameters
    ----------
    df:
        DataFrame with columns: Date, Ticker, Open, High, Low, Close.
        Must be sorted chronologically within each ticker.

    Returns
    -------
    pd.DataFrame:
        Copy of df with 6 additional columns.  Rows where any rolling
        feature is NaN are dropped (these are the first few rows per
        ticker where the window is not yet full).
    """
    df = df.copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    parts: list[pd.DataFrame] = []

    for ticker, group in df.groupby("Ticker", sort=True):
        g = group.copy().reset_index(drop=True)

        # 1. Close-to-close 1-day return
        g["close_to_close_return"] = g["Close"].pct_change()

        # 2. Open-to-close intraday return
        g["open_to_close_return"] = (g["Close"] - g["Open"]) / g["Open"]

        # 3. High-low range percentage
        g["high_low_range_pct"] = (g["High"] - g["Low"]) / g["Low"]

        # 4. 5-day SMA ratio (close / sma_5)
        sma5 = g["Close"].rolling(window=5, min_periods=5).mean()
        g["close_sma_5_ratio"] = g["Close"] / sma5

        # 5. 10-day SMA ratio (close / sma_10)
        sma10 = g["Close"].rolling(window=10, min_periods=10).mean()
        g["close_sma_10_ratio"] = g["Close"] / sma10

        # 6. 5-day rolling volatility of close-to-close return
        g["rolling_volatility_5"] = g["close_to_close_return"].rolling(
            window=5, min_periods=5
        ).std()

        parts.append(g)

    df_out = pd.concat(parts, ignore_index=True)

    # Drop rows where any auxiliary feature is NaN (start-of-window rows).
    aux_cols = [
        "close_to_close_return",
        "open_to_close_return",
        "high_low_range_pct",
        "close_sma_5_ratio",
        "close_sma_10_ratio",
        "rolling_volatility_5",
    ]
    df_out = df_out.dropna(subset=aux_cols).reset_index(drop=True)

    return df_out


def get_auxiliary_feature_columns() -> list[str]:
    """Return the ordered list of auxiliary feature column names.

    Returns
    -------
    list[str]:
        Column names in the order they will appear in the feature matrix.
    """
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "close_to_close_return",
        "open_to_close_return",
        "high_low_range_pct",
        "close_sma_5_ratio",
        "close_sma_10_ratio",
        "rolling_volatility_5",
    ]
