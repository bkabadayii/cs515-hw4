"""Phase 3C - Build auxiliary OHLC feature windows.

Computes the auxiliary_ohlc feature set (10 features), scales it using a
per-ticker MinMaxScaler fit on the training split only, then constructs
sliding windows and saves NPZ files for train/val/test splits.

Outputs:
    data/processed/features_auxiliary_ohlc_scaled.csv
    data/splits/exact_auxiliary_ohlc_train.npz
    data/splits/exact_auxiliary_ohlc_val.npz
    data/splits/exact_auxiliary_ohlc_test.npz
    data/metadata/exact_auxiliary_ohlc_window_metadata.json
    data/metadata/scaler_auxiliary_ohlc_config.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.data.auxiliary_features import (
    compute_auxiliary_features,
    get_auxiliary_feature_columns,
)
from financial_forecasting.data.scaling import TickerMinMaxScaler
from financial_forecasting.data.splits import filter_by_split, get_split_name
from financial_forecasting.data.windows import build_sliding_windows
from financial_forecasting.parameters.feature_params import (
    ExactReturnTargetParams,
    SplitParams,
    WindowParams,
)


def main() -> None:
    """Orchestrate auxiliary feature generation and window building."""
    print("=" * 60)
    print("Phase 3C - Auxiliary OHLC Feature Window Construction")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    combined_csv = paths.data_raw / "all_tickers_ohlc.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Combined raw CSV not found at: {combined_csv}\n"
            "Please run download_yfinance.py first."
        )

    # Load and sort raw data
    print(f"\nLoading raw OHLC data from: {combined_csv}")
    df_raw = pd.read_csv(combined_csv, parse_dates=["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    split_params = SplitParams()
    win_params = WindowParams()
    target_params = ExactReturnTargetParams()

    # Compute auxiliary features (drops early NaN rows)
    print("\n[1/6] Computing auxiliary features per ticker...")
    df_aux = compute_auxiliary_features(df_raw)
    aux_feature_cols = get_auxiliary_feature_columns()
    print(f"Auxiliary feature columns ({len(aux_feature_cols)}): {aux_feature_cols}")
    print(f"Rows after NaN drop: {len(df_aux)} (was {len(df_raw)})")

    # Filter training rows to fit scaler (leakage prevention)
    print("\n[2/6] Fitting auxiliary scaler on training split only...")
    df_train_only = filter_by_split(df_aux, "train", split_params)
    print(f"Training rows available for scaler fit: {len(df_train_only)}")

    scaler = TickerMinMaxScaler(
        features=aux_feature_cols,
        feature_range=(0.0, 1.0),
    )
    scaler.fit(df_train_only)

    scaler_path = paths.data_metadata / "scaler_auxiliary_ohlc_config.json"
    scaler.save(scaler_path)
    print(f"Auxiliary scaler saved to: {scaler_path}")

    # Scale all rows
    print("\n[3/6] Scaling auxiliary features across all splits...")
    df_scaled = scaler.transform(df_aux)

    # Preserve Raw_Close for target computation
    df_scaled["Raw_Close"] = df_aux["Close"].values

    scaled_csv_path = paths.data_processed / "features_auxiliary_ohlc_scaled.csv"
    df_scaled.to_csv(scaled_csv_path, index=False)
    print(f"Scaled auxiliary features saved to: {scaled_csv_path}")

    # Build sliding windows over full scaled dataset
    print(
        f"\n[4/6] Building sliding windows (T={win_params.lookback_window}, "
        f"D={len(target_params.horizons)})..."
    )
    X, y, all_metadata = build_sliding_windows(
        df=df_scaled,
        features=aux_feature_cols,
        lookback_window=win_params.lookback_window,
        horizons=target_params.horizons,
    )
    print(f"Total sliding windows: {len(X)}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Partition into splits
    print("\n[5/6] Partitioning windows into chronological splits...")
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for idx, sample in enumerate(all_metadata):
        anchor_date = sample["anchor_date"]
        assert isinstance(anchor_date, str)
        split = get_split_name(anchor_date, split_params)
        if split == "train":
            train_indices.append(idx)
        elif split == "val":
            val_indices.append(idx)
        elif split == "test":
            test_indices.append(idx)

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

    # Save NPZ files
    print("\n[6/6] Saving auxiliary OHLC split NPZ files...")
    split_summaries: dict[str, object] = {}

    for split_name, (X_split, y_split, meta_split) in splits_data.items():
        npz_filename = f"exact_auxiliary_ohlc_{split_name}.npz"
        npz_path = paths.data_splits / npz_filename

        tickers_arr = np.array([m["ticker"] for m in meta_split], dtype=object)
        dates_arr = np.array([m["anchor_date"] for m in meta_split], dtype=object)

        np.savez_compressed(
            npz_path,
            X=X_split,
            y=y_split,
            tickers=tickers_arr,
            anchor_dates=dates_arr,
        )

        ticker_counts: dict[str, int] = {}
        for m in meta_split:
            tick = str(m["ticker"])
            ticker_counts[tick] = ticker_counts.get(tick, 0) + 1

        print(f"Saved {split_name}: {npz_path}")
        print(f"  X shape: {X_split.shape}, y shape: {y_split.shape}")
        print(f"  Ticker counts: {ticker_counts}")

        split_summaries[split_name] = {
            "total_samples": len(X_split),
            "ticker_counts": ticker_counts,
        }

    # Save metadata
    metadata_json = {
        "schema_version": "1.0",
        "feature_set": "auxiliary_ohlc",
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
                "columns": aux_feature_cols,
                "num_features": len(aux_feature_cols),
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

    meta_path = paths.data_metadata / "exact_auxiliary_ohlc_window_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    print("\n" + "=" * 60)
    print("Phase 3C auxiliary window building complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
