"""Plotting script for exact return target distributions and variances.

Produces `figures/dataset/exact_return_distributions.{png,pdf}`.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.plotting.style import (
    TICKER_COLOURS,
    apply_style,
    save_figure,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
TICKERS = ["AAPL", "MSFT", "JPM"]
HORIZONS = [1, 2, 3, 4, 5]


def _kde(
    values: NDArray[np.float32], n_pts: int = 300
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute a simple Gaussian KDE over *values*.

    Uses Scott's bandwidth rule.
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
    """Load train NPZ targets and plot return distributions and variance."""
    apply_style()

    paths = ProjectPaths()
    train_npz_path = paths.data_splits / "exact_train.npz"

    if not train_npz_path.exists():
        raise FileNotFoundError(
            f"Training NPZ file not found: {train_npz_path}\n"
            "Please run build_exact_return_windows.py first."
        )

    # Load targets and tickers
    npz = np.load(train_npz_path, allow_pickle=True)
    y_train = npz["y"]  # (num_samples, 5)
    tickers_train = npz["tickers"]  # (num_samples,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Return distributions by horizon
    ax_kde = axes[0]
    ax_kde.set_title("Exact Return Distribution by Horizon (Train)", pad=10)

    # A color map/gradient from light blue to deep pink/red for horizons
    horizon_colours = ["#26C6DA", "#42A5F5", "#5C6BC0", "#AB47BC", "#EC407A"]

    for idx, d in enumerate(HORIZONS):
        h_vals = y_train[:, idx]
        x_grid, density = _kde(h_vals)
        ax_kde.plot(
            x_grid,
            density,
            color=horizon_colours[idx],
            linewidth=2.0,
            label=f"d = {d} day{'s' if d > 1 else ''}",
        )
        ax_kde.fill_between(
            x_grid, density, alpha=0.05, color=horizon_colours[idx]
        )

    ax_kde.set_xlabel("Return Ratio")
    ax_kde.set_ylabel("Density")
    ax_kde.set_xlim(-0.15, 0.15)
    ax_kde.set_ylim(bottom=0)
    ax_kde.legend(framealpha=0.25, fontsize=10)

    # Panel B: Return variance by horizon per ticker
    ax_var = axes[1]
    ax_var.set_title("Return Variance by Horizon and Ticker", pad=10)

    # Grouped bar chart calculations
    x = np.arange(len(HORIZONS))  # label locations
    width = 0.25  # width of the bars

    for idx, ticker in enumerate(TICKERS):
        ticker_mask = tickers_train == ticker
        ticker_y = y_train[ticker_mask]

        variances: list[float] = []
        for h_idx in range(len(HORIZONS)):
            var_val = float(np.var(ticker_y[:, h_idx]))
            variances.append(var_val)

        offset = (idx - 1) * width
        ax_var.bar(
            x + offset,
            variances,
            width,
            label=ticker,
            color=TICKER_COLOURS[ticker],
        )

    ax_var.set_xlabel("Forecasting Horizon (d)")
    ax_var.set_ylabel("Variance")
    ax_var.set_xticks(x)
    ax_var.set_xticklabels([f"d={d}" for d in HORIZONS])
    ax_var.legend(framealpha=0.25, fontsize=10)

    fig.suptitle(
        "Return Target Analysis (Training Split: Jan 2020 - Jul 2024)",
        y=0.98,
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = paths.figures_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(out_dir / "exact_return_distributions")

    png, pdf = save_figure(fig, stem)
    print(f"Saved return target distributions plot to: {png} and {pdf}")


if __name__ == "__main__":
    main()
