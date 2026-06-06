"""Metadata builder and JSON serialisation helpers.

:func:`build_download_metadata` collects all information that should be
preserved about a yfinance download run (parameters, library versions,
timing, per-ticker summaries) and returns a plain ``dict`` that is safe
to serialise with :mod:`json`.

:func:`save_metadata` writes that dict to the JSON file specified in the
download parameters.
"""
from __future__ import annotations

import json
import pathlib
import platform
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from financial_forecasting.data.raw_schema import ValidationSummary
from financial_forecasting.parameters.data_params import DataDownloadParams


def build_download_metadata(
    params: DataDownloadParams,
    validation_summary: ValidationSummary,
    download_start_utc: datetime,
    download_end_utc: datetime,
) -> dict[str, object]:
    """Assemble a metadata record for a completed download.

    Parameters
    ----------
    params:
        The download configuration dataclass used for this run.
    validation_summary:
        Dict returned by :func:`~raw_schema.validate_raw_data`.
    download_start_utc:
        UTC timestamp recorded just before the yfinance call.
    download_end_utc:
        UTC timestamp recorded just after saving the combined CSV.

    Returns
    -------
    dict[str, object]
        JSON-serialisable metadata dict.
    """
    elapsed_seconds = (download_end_utc - download_start_utc).total_seconds()

    return {
        "schema_version": "1.0",
        "download_timestamp_utc": download_start_utc.isoformat(),
        "download_elapsed_seconds": round(elapsed_seconds, 2),
        "environment": {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "yfinance_version": yf.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "tickers": params.tickers,
            "start_date": params.start_date,
            "end_date": params.end_date,
            "interval": params.interval,
            "auto_adjust": params.auto_adjust,
            "threads": params.threads,
            "required_columns": params.required_columns,
        },
        "output_files": {
            "raw_data_dir": str(params.raw_data_dir),
            "combined_raw_path": str(params.combined_raw_path),
            "metadata_path": str(params.metadata_path),
            "per_ticker_csvs": [
                str(params.raw_data_dir / f"{t}.csv") for t in params.tickers
            ],
        },
        "validation": validation_summary,
    }


def save_metadata(
    metadata: dict[str, object], path: pathlib.Path
) -> None:
    """Write a metadata dict to a JSON file.

    Parameters
    ----------
    metadata:
        JSON-serialisable dict (e.g. from :func:`build_download_metadata`).
    path:
        Destination file path.  Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: object) -> object:
        """Fallback serialiser for types that json cannot handle natively."""
        if isinstance(obj, pathlib.Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON-serialisable")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=_default)
    print(f"  Saved metadata -> {path}")
