"""Phase 2A analysis orchestrator.

Runs all Phase 2A plots and prints split stats:
  1. Normalized feature distributions (KDE per ticker)
  2. Return target distributions by horizon and variance.
"""
from __future__ import annotations

import json

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.plotting.dataset.plot_feature_distributions import (
    main as plot_features,
)
from financial_forecasting.plotting.dataset.plot_return_distributions import (
    main as plot_returns,
)


def main() -> None:
    """Orchestrate Phase 2A analysis and plotting."""
    print("=" * 60)
    print("Phase 2A - Windowing and Target Distribution Analysis")
    print("=" * 60)

    paths = ProjectPaths()
    metadata_json_path = paths.data_metadata / "exact_window_metadata.json"

    if not metadata_json_path.exists():
        raise FileNotFoundError(
            f"Exact window metadata JSON not found: {metadata_json_path}\n"
            "Please run: uv run python src/financial_forecasting/dataset/build_exact_return_windows.py"
        )

    # Load and print metadata summaries
    with open(metadata_json_path, encoding="utf-8") as f:
        meta = json.load(f)

    print("Dataset Splits Sample Summary:")
    splits_summary = meta["splits_summary"]
    for split_name, summary in splits_summary.items():
        print(f"  {split_name.capitalize()} split:")
        print(f"    Total samples: {summary['total_samples']}")
        print(f"    Ticker counts: {summary['ticker_counts']}")

    print("\n[1/2] Plotting feature distributions ...")
    plot_features()

    print("\n[2/2] Plotting return target distributions and variances ...")
    plot_returns()

    print("\n" + "=" * 60)
    print("Phase 2A analysis complete. Figures saved under:", paths.figures_dataset)
    print("=" * 60)


if __name__ == "__main__":
    main()
