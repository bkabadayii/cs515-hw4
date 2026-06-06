"""Plot 1A-2: Missing-value summary heatmap and table.

Produces ``fig02_missing_values.{png,pdf}``.

The figure shows:
- A colour-coded grid: one column per (Ticker, OHLC field), one row per
  calendar year, cell value = number of missing trading days in that year.
- A text annotation inside each cell when any value is non-zero.

Why this visual form?
    A matrix heatmap makes it immediately obvious whether any field or any
    ticker has systematic gaps.  For Phase 1A the dataset has zero NaNs, so
    the heatmap will be uniformly ``0`` -- a useful sanity-check artefact.
    If a re-download ever introduces gaps, the colour scale changes visibly.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from financial_forecasting.plotting.style import apply_style, save_figure

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


OHLC_COLS = ["Open", "High", "Low", "Close"]
TICKERS   = ["AAPL", "MSFT", "JPM"]


def plot_missing_values(
    df: pd.DataFrame,
    out_dir: pathlib.Path,
) -> tuple[str, str]:
    """Create and save the missing-value summary figure.

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

    # Build matrix: rows = years, columns = (ticker, field) pairs
    df["Year"] = df["Date"].dt.year
    years = sorted(df["Year"].unique())
    col_labels = [f"{t}\n{c}" for t in TICKERS for c in OHLC_COLS]

    matrix = np.zeros((len(years), len(col_labels)), dtype=int)
    col_idx = 0
    for ticker in TICKERS:
        sub = df[df["Ticker"] == ticker]
        for col in OHLC_COLS:
            for row_idx, yr in enumerate(years):
                yr_sub = sub[sub["Year"] == yr]
                matrix[row_idx, col_idx] = int(yr_sub[col].isna().sum())
            col_idx += 1

    # Determine colour scale: if all zero, show green; if any non-zero, use red scale.
    vmax = max(1, int(matrix.max()))

    fig, ax = plt.subplots(figsize=(13, max(4, len(years) * 0.7)))

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=vmax,
        alpha=0.85,
    )

    # Annotate cells
    for r in range(len(years)):
        for c in range(len(col_labels)):
            val = matrix[r, c]
            txt = str(val) if val > 0 else "0"
            ax.text(
                c, r, txt,
                ha="center", va="center",
                fontsize=8,
                color="#1A1A1A",
            )

    # Axes labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=9)
    ax.set_xlabel("(Ticker, OHLC Field)")
    ax.set_ylabel("Year")
    ax.set_title("Missing Values per (Ticker, Field, Year)", pad=10)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Missing count", fontsize=9)
    # Import FG_COLOUR and GRID_COLOUR locally or resolve them
    from financial_forecasting.plotting.style import FG_COLOUR, GRID_COLOUR
    cbar.ax.yaxis.set_tick_params(color=FG_COLOUR)

    # Separator lines between ticker groups
    for sep in [3.5, 7.5]:
        ax.axvline(sep, color=GRID_COLOUR, linewidth=1.5)

    fig.tight_layout()

    stem = str(out_dir / "fig02_missing_values")
    png, pdf = save_figure(fig, stem)

    print(f"  Saved {png}")
    print(f"  Saved {pdf}")
    return png, pdf


def main() -> None:
    """Entry point: load data and produce figure 1A-2."""
    combined_csv = _REPO_ROOT / "data" / "raw" / "all_tickers_ohlc.csv"
    out_dir = _REPO_ROOT / "figures" / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Combined CSV not found: {combined_csv}\n"
            "Run: uv run python src/financial_forecasting/dataset/download_yfinance.py"
        )

    df = pd.read_csv(combined_csv, parse_dates=["Date"])
    print("Plotting missing-value summary ...")
    plot_missing_values(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
