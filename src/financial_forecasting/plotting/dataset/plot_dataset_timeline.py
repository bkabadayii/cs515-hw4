"""Plot 1A-1: Close-price timeline with chronological split regions.

Produces ``fig01_close_timeline.{png,pdf}``.

The figure shows:
- Adjusted Close price for AAPL, MSFT, JPM on a shared x-axis (date).
- Shaded vertical bands for Train / Validation / Test splits.
- A compact legend identifying tickers and split regions.

Why this visual form?
    A line plot over time lets the reader see absolute price scale
    differences between tickers and verify that the chronological split
    boundaries land on reasonable market periods (no look-ahead leakage).
"""
from __future__ import annotations

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from financial_forecasting.plotting.style import (
    SPLIT_ALPHA,
    SPLIT_COLOURS,
    TICKER_COLOURS,
    apply_style,
    save_figure,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Split boundaries (must match Phase 2 SplitParams exactly)
# ---------------------------------------------------------------------------
TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-07-31"
VAL_START   = "2024-08-01"
VAL_END     = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2025-12-31"


def plot_close_timeline(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
) -> tuple[str, str]:
    """Create and save the close-price timeline figure.

    Parameters
    ----------
    df:
        Combined long-format OHLC DataFrame with columns
        ``Date, Ticker, Open, High, Low, Close``.
    out_dir:
        Directory where PNG, PDF, and caption markdown will be saved.

    Returns
    -------
    tuple[str, str]
        PNG and PDF absolute paths.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 5))

    # Plot one line per ticker
    for ticker in ["AAPL", "MSFT", "JPM"]:
        sub = df[df["Ticker"] == ticker].sort_values("Date")
        ax.plot(
            sub["Date"],
            sub["Close"],
            color=TICKER_COLOURS[ticker],
            linewidth=1.4,
            label=ticker,
            zorder=3,
        )

    # Shaded split regions
    splits = [
        ("train",      TRAIN_START, TRAIN_END),
        ("validation", VAL_START,   VAL_END),
        ("test",       TEST_START,  TEST_END),
    ]
    for split_name, s, e in splits:
        ax.axvspan(
            pd.Timestamp(s),  # type: ignore[arg-type]
            pd.Timestamp(e),  # type: ignore[arg-type]
            alpha=SPLIT_ALPHA,
            color=SPLIT_COLOURS[split_name],
            zorder=1,
            label=f"_{split_name}",  # leading _ hides from legend
        )
        # Add a subtle boundary line at each split start
        ax.axvline(
            pd.Timestamp(s),  # type: ignore[arg-type]
            color=SPLIT_COLOURS[split_name],
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            zorder=2,
        )

    # Build a clean legend: tickers + split patches
    ticker_handles = [
        mpatches.Patch(color=TICKER_COLOURS[t], label=t)
        for t in ["AAPL", "MSFT", "JPM"]
    ]
    split_handles = [
        mpatches.Patch(
            color=SPLIT_COLOURS[n],
            alpha=0.4,
            label=f"{n.capitalize()} ({lbl})",
        )
        for n, lbl in [
            ("train",      "Jan 2020 - Jul 2024"),
            ("validation", "Aug - Dec 2024"),
            ("test",       "Jan - Dec 2025"),
        ]
    ]
    ax.legend(
        handles=ticker_handles + split_handles,
        loc="upper left",
        framealpha=0.25,
        ncol=2,
        fontsize=9,
    )

    ax.set_title("Adjusted Close Price - AAPL / MSFT / JPM (2020-2025)", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close (USD)")
    ax.set_xlim(
        pd.Timestamp(TRAIN_START),  # type: ignore[arg-type]
        pd.Timestamp("2026-01-01"),  # type: ignore[arg-type]
    )

    fig.tight_layout()

    stem = str(out_dir / "fig01_close_timeline")
    png, pdf = save_figure(fig, stem)

    print(f"  Saved {png}")
    print(f"  Saved {pdf}")
    return png, pdf


def main() -> None:
    """Entry point: load data and produce figure 1A-1."""
    combined_csv = _REPO_ROOT / "data" / "raw" / "all_tickers_ohlc.csv"
    out_dir = _REPO_ROOT / "figures" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Combined CSV not found: {combined_csv}\n"
            "Run: uv run python src/financial_forecasting/dataset/download_yfinance.py"
        )

    df = pd.read_csv(combined_csv, parse_dates=["Date"])
    print("Plotting close-price timeline ...")
    plot_close_timeline(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
