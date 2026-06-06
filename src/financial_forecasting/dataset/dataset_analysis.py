"""Phase 1A analysis orchestrator.

Runs all three Phase 1A dataset plots in sequence:
  1. Close-price timeline with split regions
  2. Missing-value heatmap
  3. Close-price distribution (raw + normalised)

Usage
-----
    uv run python src/financial_forecasting/dataset/dataset_analysis.py

All figures are saved under ``figures/dataset/``.
All data is read from ``data/raw/all_tickers_ohlc.csv`` -- no re-download.
"""

from __future__ import annotations

import pathlib

import pandas as pd

from financial_forecasting.plotting.dataset.plot_close_distribution import (
    plot_close_distribution,
)
from financial_forecasting.plotting.dataset.plot_dataset_timeline import (
    plot_close_timeline,
)
from financial_forecasting.plotting.dataset.plot_missing_values import (
    plot_missing_values,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def main() -> None:
    """Run all Phase 1A plots."""
    combined_csv = _REPO_ROOT / "data" / "raw" / "all_tickers_ohlc.csv"
    out_dir = _REPO_ROOT / "figures" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Combined CSV not found: {combined_csv}\n"
            "Run: uv run python src/financial_forecasting/dataset/download_yfinance.py"
        )

    print("=" * 60)
    print("Phase 1A - Dataset Analysis Plots")
    print("=" * 60)

    df = pd.read_csv(combined_csv, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows from {combined_csv}\n")

    print("[1/3] Close-price timeline ...")
    plot_close_timeline(df, out_dir)

    print("\n[2/3] Missing-value summary ...")
    plot_missing_values(df, out_dir)

    print("\n[3/3] Close-price distribution ...")
    plot_close_distribution(df, out_dir)

    print("\n" + "=" * 60)
    print("Phase 1A complete.  Figures saved under:", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
