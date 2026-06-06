"""Utilities for saving and formatting prediction records with reconstructed target dates."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def build_date_lookup(
    ohlc_csv_path: pathlib.Path,
) -> dict[tuple[str, str], list[str]]:
    """Build a lookup map from (ticker, anchor_date) to its 5 future trading dates.

    Parameters
    ----------
    ohlc_csv_path:
        Path to the raw combined CSV containing Date and Ticker.

    Returns
    -------
    dict:
        Mapping from (ticker, anchor_date) -> [target_date_h1, ..., target_date_h5]
    """
    df = pd.read_csv(ohlc_csv_path)
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["Date_Str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    lookup: dict[tuple[str, str], list[str]] = {}
    tickers = df["Ticker"].unique()

    for ticker in tickers:
        ticker_df = df[df["Ticker"] == ticker]
        dates = ticker_df["Date_Str"].tolist()
        n = len(dates)
        for idx in range(n):
            anchor = dates[idx]
            # Store next 5 trading days
            future_days: list[str] = []
            for h in [1, 2, 3, 4, 5]:
                if idx + h < n:
                    future_days.append(dates[idx + h])
                else:
                    future_days.append("")
            lookup[(str(ticker), anchor)] = future_days

    return lookup


def save_prediction_records(
    save_path: pathlib.Path,
    split_name: str,
    tickers: list[str],
    anchor_dates: list[str],
    actuals: NDArray[np.float32],
    predictions: NDArray[np.float32],
    ohlc_csv_path: pathlib.Path,
) -> None:
    """Reconstruct target dates and save the predictions to a structured CSV.

    Parameters
    ----------
    save_path:
        Destination CSV path.
    split_name:
        Name of the data split (e.g. "train", "val", "test").
    tickers:
        List of ticker strings per sequence.
    anchor_dates:
        List of anchor date strings per sequence.
    actuals:
        Array of actual returns, shape (N, 5).
    predictions:
        Array of predicted returns, shape (N, 5).
    ohlc_csv_path:
        Path to the combined OHLC CSV to resolve chronological trading days.
    """
    lookup = build_date_lookup(ohlc_csv_path)
    residuals = actuals - predictions

    records: list[dict[str, float | str]] = []
    n_samples = len(tickers)

    for j in range(n_samples):
        ticker = tickers[j]
        anchor = anchor_dates[j]
        target_days = lookup.get((ticker, anchor), ["", "", "", "", ""])

        record: dict[str, float | str] = {
            "split": split_name,
            "ticker": ticker,
            "anchor_date": anchor,
            "target_date_h1": target_days[0],
            "target_date_h2": target_days[1],
            "target_date_h3": target_days[2],
            "target_date_h4": target_days[3],
            "target_date_h5": target_days[4],
            "actual_h1": float(actuals[j, 0]),
            "actual_h2": float(actuals[j, 1]),
            "actual_h3": float(actuals[j, 2]),
            "actual_h4": float(actuals[j, 3]),
            "actual_h5": float(actuals[j, 4]),
            "pred_h1": float(predictions[j, 0]),
            "pred_h2": float(predictions[j, 1]),
            "pred_h3": float(predictions[j, 2]),
            "pred_h4": float(predictions[j, 3]),
            "pred_h5": float(predictions[j, 4]),
            "residual_h1": float(residuals[j, 0]),
            "residual_h2": float(residuals[j, 1]),
            "residual_h3": float(residuals[j, 2]),
            "residual_h4": float(residuals[j, 3]),
            "residual_h5": float(residuals[j, 4]),
        }
        records.append(record)

    df_out = pd.DataFrame(records)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(save_path, index=False)
    print(f"Saved prediction records to: {save_path}")
