"""Utilities for chronological splitting of financial time-series data.

This ensures zero lookahead leakage by strictly partitioning dates.
"""
from __future__ import annotations

import pandas as pd

from financial_forecasting.parameters.feature_params import SplitParams


def get_split_name(date: pd.Timestamp | str, params: SplitParams) -> str | None:
    """Determine the split name for a given date.

    Parameters
    ----------
    date:
        The date to classify.
    params:
        Chronological split boundary parameters.

    Returns
    -------
    str | None:
        "train", "val", "test", or None if the date does not fall into any split.
    """
    ts = pd.to_datetime(date)
    if pd.to_datetime(params.train_start) <= ts <= pd.to_datetime(params.train_end):
        return "train"
    if pd.to_datetime(params.val_start) <= ts <= pd.to_datetime(params.val_end):
        return "val"
    if pd.to_datetime(params.test_start) <= ts <= pd.to_datetime(params.test_end):
        return "test"
    return None


def filter_by_split(
    df: pd.DataFrame, split_name: str, params: SplitParams
) -> pd.DataFrame:
    """Filter rows of a DataFrame based on split boundary dates.

    Parameters
    ----------
    df:
        DataFrame containing a 'Date' column.
    split_name:
        One of 'train', 'val', or 'test'.
    params:
        The chronological split boundaries.

    Returns
    -------
    pd.DataFrame:
        A subset of the input DataFrame containing only the rows matching the split.
    """
    dates = pd.to_datetime(df["Date"])
    if split_name == "train":
        mask = (dates >= pd.to_datetime(params.train_start)) & (
            dates <= pd.to_datetime(params.train_end)
        )
    elif split_name == "val":
        mask = (dates >= pd.to_datetime(params.val_start)) & (
            dates <= pd.to_datetime(params.val_end)
        )
    elif split_name == "test":
        mask = (dates >= pd.to_datetime(params.test_start)) & (
            dates <= pd.to_datetime(params.test_end)
        )
    else:
        raise ValueError(f"Unknown split name: {split_name}")
    return df[mask].copy()
