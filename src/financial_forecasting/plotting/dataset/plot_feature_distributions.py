"""Plotting script for normalized feature distributions.

Produces `figures/dataset/feature_distributions.{png,pdf}`.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.parameters.feature_params import (
    FeatureParams,
    SplitParams,
)
from financial_forecasting.plotting.style import (
    TICKER_COLOURS,
    apply_style,
    save_figure,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
TICKERS = ["AAPL", "MSFT", "JPM"]


def _kde(
    values: NDArray[np.float32], n_pts: int = 300
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute a simple Gaussian KDE over *values*.

    Uses Scott's bandwidth rule. Returns (x_grid, density).
    """
    std = float(np.std(values))
    if std == 0.0:
        std = 1e-5
    bw = 1.06 * std * len(values) ** (-0.2)
    x = np.linspace(
        float(values.min()) - 2 * bw, float(values.max()) + 2 * bw, n_pts
    )
    density = np.zeros_like(x)
    for v in values:
        density += np.exp(-0.5 * ((x - v) / bw) ** 2)
    density /= density.sum() * (x[1] - x[0])
    return x.astype(np.float32), density.astype(np.float32)


def main() -> None:
    """Load scaled features and plot their distributions for the training split."""
    apply_style()

    paths = ProjectPaths()
    scaled_csv = paths.data_processed / "features_scaled.csv"

    if not scaled_csv.exists():
        raise FileNotFoundError(
            f"Scaled features CSV not found: {scaled_csv}\n"
            "Please run build_exact_return_windows.py first."
        )

    df = pd.read_csv(scaled_csv, parse_dates=["Date"])

    # Filter to training split only for showing normalized distributions
    split_params = SplitParams()
    dates = pd.to_datetime(df["Date"])
    train_mask = (dates >= pd.to_datetime(split_params.train_start)) & (
        dates <= pd.to_datetime(split_params.train_end)
    )
    df_train = df[train_mask].copy()

    feat_params = FeatureParams()
    features = feat_params.columns

    # Set up subplots (2x2 grid for Open, High, Low, Close)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes_flat = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes_flat[idx]
        ax.set_title(f"Normalized {feature} Distribution", pad=8)

        for ticker in TICKERS:
            ticker_df = df_train[df_train["Ticker"] == ticker]
            if ticker_df.empty:
                continue

            vals = ticker_df[feature].dropna().to_numpy(dtype=np.float32)
            if len(vals) < 2:
                continue

            x_grid, density = _kde(vals)
            colour = TICKER_COLOURS[ticker]

            ax.plot(
                x_grid,
                density,
                color=colour,
                linewidth=2.0,
                label=ticker,
            )
            ax.fill_between(x_grid, density, alpha=0.10, color=colour)

        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Density")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(bottom=0)
        ax.legend(framealpha=0.25, fontsize=9)

    fig.suptitle(
        "Normalized Feature Distributions (Training Split: Jan 2020 - Jul 2024)",
        y=0.98,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = paths.figures_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(out_dir / "feature_distributions")

    png, pdf = save_figure(fig, stem)
    print(f"Saved feature distributions plot to: {png} and {pdf}")


if __name__ == "__main__":
    main()
