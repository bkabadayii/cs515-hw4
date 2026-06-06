"""dataset/validate_raw_data.py - Standalone Phase 1 validation script.

Re-runs all validation checks on an already-downloaded combined CSV
without re-downloading from Yahoo Finance.  Useful for CI or spot-checks.

Usage
-----
    uv run python src/financial_forecasting/dataset/validate_raw_data.py
    uv run python src/financial_forecasting/dataset/validate_raw_data.py --csv data/raw/all_tickers_ohlc.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

from financial_forecasting.data.raw_schema import validate_raw_data
from financial_forecasting.parameters.data_params import DataDownloadParams

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate a previously downloaded raw OHLC CSV."
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=_REPO_ROOT / "data" / "raw" / "all_tickers_ohlc.csv",
        help="Path to the combined OHLC CSV (default: data/raw/all_tickers_ohlc.csv).",
    )
    return parser.parse_args()


def main() -> None:
    """Load combined CSV and run all validation checks."""
    args = _parse_args()
    csv_path: pathlib.Path = args.csv

    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        print("Run:  uv run python src/financial_forecasting/dataset/download_yfinance.py")
        sys.exit(1)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    params = DataDownloadParams(
        raw_data_dir=_REPO_ROOT / "data" / "raw",
        combined_raw_path=csv_path,
        metadata_path=_REPO_ROOT / "data" / "metadata" / "raw_download_metadata.json",
    )

    print("Running validation checks ...")
    try:
        summary = validate_raw_data(df, params)
    except ValueError as exc:
        print(f"\n[FAIL] {exc}")
        sys.exit(1)

    print("\n[PASS] All validation checks passed.")
    print(f"  Total rows   : {summary['total_rows']}")
    print(f"  Tickers found: {summary['tickers_found']}")
    for ticker, info in summary["ticker_summaries"].items():
        print(
            f"  {ticker}: {info['rows']} rows  "
            f"{info['min_date']} -> {info['max_date']}"
        )
    warnings = summary.get("warnings", [])
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  [WARN] {w}")


if __name__ == "__main__":
    main()
