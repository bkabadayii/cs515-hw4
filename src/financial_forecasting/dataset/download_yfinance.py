"""dataset/download_yfinance.py - Phase 1 entry-point script.

Downloads daily OHLC data for the default tickers (AAPL, MSFT, JPM)
from 2020-01-01 to 2025-12-31 (exclusive end date 2026-01-01), saves
per-ticker CSVs and a combined CSV, validates the result, and writes a
metadata JSON.

Usage
-----
    uv run python src/financial_forecasting/dataset/download_yfinance.py

All tunable settings live in
``src/financial_forecasting/parameters/data_params.py``.  Edit that
file to change tickers, date range, or output paths.
"""
from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from financial_forecasting.data.metadata_builder import (
    build_download_metadata,
    save_metadata,
)
from financial_forecasting.data.raw_schema import validate_raw_data
from financial_forecasting.data.yfinance_client import download_tickers
from financial_forecasting.parameters.data_params import DataDownloadParams

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def main() -> None:
    """Run the full Phase 1 download + validation + metadata pipeline."""
    # ------------------------------------------------------------------ #
    # 1. Build parameters                                                 #
    # ------------------------------------------------------------------ #
    # Resolve paths relative to the repository root so that the script
    # works regardless of the working directory from which it is invoked.
    params = DataDownloadParams(
        raw_data_dir=_REPO_ROOT / "data" / "raw",
        combined_raw_path=_REPO_ROOT / "data" / "raw" / "all_tickers_ohlc.csv",
        metadata_path=_REPO_ROOT / "data" / "metadata" / "raw_download_metadata.json",
    )

    print("=" * 60)
    print("Phase 1 - Dataset download and validation")
    print("=" * 60)
    print(f"Tickers  : {params.tickers}")
    print(f"Start    : {params.start_date}  (inclusive)")
    print(f"End      : {params.end_date}  (exclusive - yfinance convention)")
    print(f"Interval : {params.interval}")
    print()

    # ------------------------------------------------------------------ #
    # 2. Download                                                         #
    # ------------------------------------------------------------------ #
    t_start = datetime.now(tz=UTC)
    combined_df = download_tickers(params)
    t_end = datetime.now(tz=UTC)

    print()

    # ------------------------------------------------------------------ #
    # 3. Validate                                                         #
    # ------------------------------------------------------------------ #
    print("Validating raw data ...")
    validation_summary = validate_raw_data(combined_df, params)

    print("  [PASS] Schema check passed.")
    print(f"  [PASS] Total rows   : {validation_summary['total_rows']}")
    print(f"  [PASS] Tickers found: {validation_summary['tickers_found']}")
    for ticker, info in validation_summary["ticker_summaries"].items():
        print(
            f"         {ticker}: {info['rows']} rows  "
            f"{info['min_date']} -> {info['max_date']}"
        )

    warnings = validation_summary.get("warnings", [])
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    [WARN] {w}")

    print()

    # ------------------------------------------------------------------ #
    # 4. Save metadata                                                    #
    # ------------------------------------------------------------------ #
    print("Saving metadata ...")
    metadata = build_download_metadata(
        params=params,
        validation_summary=validation_summary,
        download_start_utc=t_start,
        download_end_utc=t_end,
    )
    save_metadata(metadata, params.metadata_path)

    print()
    print("=" * 60)
    print("Phase 1 complete.  Output files:")
    for ticker in params.tickers:
        csv_path = params.raw_data_dir / f"{ticker}.csv"
        print(f"  {csv_path}")
    print(f"  {params.combined_raw_path}")
    print(f"  {params.metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
