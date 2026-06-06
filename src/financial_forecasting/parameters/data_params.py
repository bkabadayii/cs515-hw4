"""Data download parameters for yfinance.

``DataDownloadParams`` is the single source of truth for every
setting that controls what data is fetched, where it is saved, and
how it is labelled.  All scripts that download or validate raw data
must accept an instance of this dataclass rather than individual
keyword arguments.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field


@dataclass
class DataDownloadParams:
    """Parameters controlling the yfinance data download.

    Attributes
    ----------
    tickers:
        List of S&P 500 ticker symbols to download.
    start_date:
        Inclusive start date for the download, in ``YYYY-MM-DD`` format.
    end_date:
        **Exclusive** end date passed to yfinance.  Set to one day after
        the last desired date.  To include all of December 2025 use
        ``"2026-01-01"``.
    interval:
        yfinance data interval string.  ``"1d"`` gives daily OHLCV bars.
    raw_data_dir:
        Directory where individual per-ticker CSVs are saved.
    combined_raw_path:
        File path for the single combined CSV that contains all tickers.
    metadata_path:
        File path for the JSON metadata record saved after download.
    auto_adjust:
        Whether to apply yfinance split/dividend auto-adjustments to the
        Close price.  ``True`` means the Close column is already adjusted.
    threads:
        Whether yfinance should use multi-threaded downloads when fetching
        multiple tickers in one call.
    required_columns:
        Column names that must be present after download.  Validated during
        the schema check.
    """

    tickers: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "JPM"]
    )
    start_date: str = "2020-01-01"
    # Exclusive end date: yfinance does not include this day in the result.
    # Setting it to 2026-01-01 ensures the full December 2025 is captured.
    end_date: str = "2026-01-01"
    interval: str = "1d"
    raw_data_dir: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("data/raw")
    )
    combined_raw_path: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("data/raw/all_tickers_ohlc.csv")
    )
    metadata_path: pathlib.Path = field(
        default_factory=lambda: pathlib.Path(
            "data/metadata/raw_download_metadata.json"
        )
    )
    auto_adjust: bool = True
    threads: bool = True
    required_columns: list[str] = field(
        default_factory=lambda: ["Date", "Ticker", "Open", "High", "Low", "Close"]
    )
