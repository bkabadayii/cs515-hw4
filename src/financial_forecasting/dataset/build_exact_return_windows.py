"""Phase 2 dataset build pipeline.

Prepares chronological splits, feature scaling (per ticker), and window/target construction.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.data.scaling import TickerMinMaxScaler
from financial_forecasting.data.splits import filter_by_split, get_split_name
from financial_forecasting.data.windows import build_sliding_windows
from financial_forecasting.parameters.feature_params import (
    ExactReturnTargetParams,
    FeatureParams,
    ScalingParams,
    SplitParams,
    WindowParams,
)


def main() -> None:
    """Orchestrate Phase 2 dataset scaling and window building."""
    print("=" * 60)
    print("Phase 2 - Splits, Features, Scaling, and Window Construction")
    print("=" * 60)

    # Initialize configuration
    paths = ProjectPaths()
    paths.ensure_all()

    # Construct the path to the combined raw OHLC data
    combined_csv = paths.data_raw / "all_tickers_ohlc.csv"

    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Combined raw CSV not found at: {combined_csv}\n"
            "Please run download_yfinance.py first."
        )

    # Load raw data
    print(f"Loading raw data from: {combined_csv}")
    df_raw = pd.read_csv(combined_csv, parse_dates=["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Instantiating parameters
    split_params = SplitParams()
    feat_params = FeatureParams()
    scale_params = ScalingParams()
    win_params = WindowParams()
    target_params = ExactReturnTargetParams()

    # Log split dates
    print("Chronological Splits:")
    print(f"  Train:      {split_params.train_start} to {split_params.train_end}")
    print(f"  Validation: {split_params.val_start} to {split_params.val_end}")
    print(f"  Test:       {split_params.test_start} to {split_params.test_end}")

    # Step 1: Filter training rows to fit the scaler
    print("\n[1/5] Fitting scaler on training split only...")
    df_train_only = filter_by_split(df_raw, "train", split_params)

    # Initialize and fit per-ticker MinMaxScaler
    scaler = TickerMinMaxScaler(
        features=feat_params.columns,
        feature_range=(
            scale_params.feature_range_min,
            scale_params.feature_range_max,
        ),
    )
    scaler.fit(df_train_only)

    # Save the scaler config
    scaler_save_path = paths.root / scale_params.scaler_save_path
    scaler.save(scaler_save_path)
    print(f"Scaler parameters saved to: {scaler_save_path}")

    # Step 2: Transform all data with training scaler
    print("\n[2/5] Scaling full dataset features...")
    df_scaled = scaler.transform(df_raw)

    # Save scaled features
    scaled_csv_path = paths.data_processed / "features_scaled.csv"
    df_scaled.to_csv(scaled_csv_path, index=False)
    print(f"Scaled features saved to: {scaled_csv_path}")

    # Add Raw_Close column to avoid division by zero and scaling distortion during target construction
    df_scaled["Raw_Close"] = df_raw["Close"]

    # Step 3: Build sliding windows
    print(
        f"\n[3/5] Constructing lookback windows (T={win_params.lookback_window}) and return targets (D={len(target_params.horizons)})..."
    )
    X, y, all_metadata = build_sliding_windows(
        df=df_scaled,
        features=feat_params.columns,
        lookback_window=win_params.lookback_window,
        horizons=target_params.horizons,
    )
    print(f"Total sliding windows generated: {len(X)}")

    # Step 4: Split sliding windows into Train, Validation, and Test
    print("\n[4/5] Partitioning samples into chronological splits...")

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for idx, sample in enumerate(all_metadata):
        anchor_date = sample["anchor_date"]
        assert isinstance(anchor_date, str)
        # Ensure we get the split name properly
        split = get_split_name(anchor_date, split_params)
        if split == "train":
            train_indices.append(idx)
        elif split == "val":
            val_indices.append(idx)
        elif split == "test":
            test_indices.append(idx)

    # Filter NPZ arrays and metadata lists
    splits_data = {
        "train": (
            X[train_indices],
            y[train_indices],
            [all_metadata[i] for i in train_indices],
        ),
        "val": (
            X[val_indices],
            y[val_indices],
            [all_metadata[i] for i in val_indices],
        ),
        "test": (
            X[test_indices],
            y[test_indices],
            [all_metadata[i] for i in test_indices],
        ),
    }

    # Step 5: Save NPZ files and summarize
    print("\n[5/5] Saving split NPZ files and metadata...")
    split_summaries: dict[str, object] = {}

    for split_name, (X_split, y_split, meta_split) in splits_data.items():
        # Save NPZ file
        npz_filename = f"exact_{split_name}.npz"
        npz_path = paths.data_splits / npz_filename

        # Prepare arrays for saving
        tickers_arr = np.array([m["ticker"] for m in meta_split], dtype=object)
        dates_arr = np.array([m["anchor_date"] for m in meta_split], dtype=object)

        np.savez_compressed(
            npz_path,
            X=X_split,
            y=y_split,
            tickers=tickers_arr,
            anchor_dates=dates_arr,
        )

        # Get count per ticker
        ticker_counts: dict[str, int] = {}
        for m in meta_split:
            tick = str(m["ticker"])
            ticker_counts[tick] = ticker_counts.get(tick, 0) + 1

        print(f"Saved {split_name} split to: {npz_path}")
        print(f"  Shape X: {X_split.shape}, Shape y: {y_split.shape}")
        print(f"  Ticker counts: {ticker_counts}")

        split_summaries[split_name] = {
            "total_samples": len(X_split),
            "ticker_counts": ticker_counts,
        }

    # Prepare windowing metadata JSON
    metadata_json = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "parameters": {
            "splits": {
                "train_start": split_params.train_start,
                "train_end": split_params.train_end,
                "val_start": split_params.val_start,
                "val_end": split_params.val_end,
                "test_start": split_params.test_start,
                "test_end": split_params.test_end,
            },
            "features": {
                "columns": feat_params.columns,
            },
            "scaling": {
                "scaler_type": scale_params.scaler_type,
                "feature_range_min": scale_params.feature_range_min,
                "feature_range_max": scale_params.feature_range_max,
                "scaler_save_path": str(scale_params.scaler_save_path),
            },
            "windowing": {
                "lookback_window": win_params.lookback_window,
            },
            "targets": {
                "horizons": target_params.horizons,
            },
        },
        "splits_summary": split_summaries,
    }

    metadata_save_path = paths.data_metadata / "exact_window_metadata.json"
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    print(f"\nMetadata saved to: {metadata_save_path}")

    print("\n" + "=" * 60)
    print("Phase 2 window building complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
