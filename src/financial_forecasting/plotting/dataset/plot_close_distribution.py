"""Plot 1A-3: Normalised close distribution (min-max per ticker).

Produces ``fig03_close_distribution.{png,pdf}``.

The figure has two panels:
  Panel A (left)  -- Raw Close KDE per ticker.
    Shows the absolute price distribution.  Reveals that MSFT is on a
    completely different scale from AAPL and JPM, motivating normalisation.
  Panel B (right) -- Min-max normalised Close KDE per ticker.
    After normalisation all three tickers share the [0, 1] range.
    This is the distribution that the scaler will produce in Phase 2.

Why two panels?
    Showing both raw and normalised side-by-side makes the argument for
    normalisation explicit without requiring the reader to look at any
    other figure.  The panels have different visual forms (absolute vs
    normalised) and ask different questions, so they are not duplicates.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from financial_forecasting.plotting.style import (
    TICKER_COLOURS,
    apply_style,
    save_figure,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

TICKERS = ["AAPL", "MSFT", "JPM"]

def _kde(values: NDArray[np.float64], n_pts: int = 300) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a simple Gaussian KDE over *values*.

    Uses Scott's bandwidth rule.  Returns (x_grid, density).
    """
    std = float(np.std(values))
    bw = 1.06 * std * len(values) ** (-0.2)
    x = np.linspace(values.min() - 2 * bw, values.max() + 2 * bw, n_pts)
    density = np.zeros_like(x)
    for v in values:
        density += np.exp(-0.5 * ((x - v) / bw) ** 2)
    density /= density.sum() * (x[1] - x[0])
    return x, density


def plot_close_distribution(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
) -> tuple[str, str]:
    """Create and save the close-distribution figure (two panels).

    Parameters
    ----------
    df:
        Combined long-format OHLC DataFrame.
    out_dir:
        Output directory.

    Returns
    -------
    tuple[str, str]
        PNG and PDF absolute paths.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ticker in TICKERS:
        raw_vals = df[df["Ticker"] == ticker]["Close"].dropna().to_numpy(dtype=float)
        norm_vals = (raw_vals - raw_vals.min()) / (raw_vals.max() - raw_vals.min())
        colour = TICKER_COLOURS[ticker]

        # Panel A: raw
        x_raw, d_raw = _kde(raw_vals)
        axes[0].plot(x_raw, d_raw, color=colour, linewidth=2.0, label=ticker)
        axes[0].fill_between(x_raw, d_raw, alpha=0.15, color=colour)

        # Panel B: normalised
        x_norm, d_norm = _kde(norm_vals)
        axes[1].plot(x_norm, d_norm, color=colour, linewidth=2.0, label=ticker)
        axes[1].fill_between(x_norm, d_norm, alpha=0.15, color=colour)

    for ax, title, xlabel in [
        (axes[0], "Raw Adjusted Close (USD)", "Price (USD)"),
        (axes[1], "Min-Max Normalised Close [0, 1]", "Normalised price"),
    ]:
        ax.set_title(title, pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.25, fontsize=9)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Close Price Distribution -- Raw vs Normalised (2020-2025)",
        y=1.01,
        fontsize=13,
    )
    fig.tight_layout()

    stem = str(out_dir / "fig03_close_distribution")
    png, pdf = save_figure(fig, stem)

    print(f"  Saved {png}")
    print(f"  Saved {pdf}")
    return png, pdf


def main() -> None:
    """Entry point: load data and produce figure 1A-3."""
    combined_csv = _REPO_ROOT / "data" / "raw" / "all_tickers_ohlc.csv"
    out_dir = _REPO_ROOT / "figures" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Combined CSV not found: {combined_csv}\n"
            "Run: uv run python src/financial_forecasting/dataset/download_yfinance.py"
        )

    df = pd.read_csv(combined_csv, parse_dates=["Date"])
    print("Plotting close-price distribution ...")
    plot_close_distribution(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
