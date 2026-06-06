"""Phase 4 dataset build pipeline - rolling-average return windows.

Builds rolling-average future return targets (rolling_window=3, uniform weights)
for both the original_ohlc and auxiliary_ohlc feature sets, and saves NPZ files
for train/validation/test splits.

The same scaled feature data produced in Phase 2 and Phase 3C is reused.
Only the target (y) computation changes from exact to rolling-average.

Convention:
    rolling_window = 3
    weights = [1/3, 1/3, 1/3]
    r_roll_{t+d} = (1/3) * sum_{j=0}^{2} (close[t+d-j] - close[t]) / close[t]

Usage:
    uv run python src/financial_forecasting/dataset/build_rolling_return_windows.py
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
from financial_forecasting.data.rolling_windows import build_rolling_windows
from financial_forecasting.data.scaling import TickerMinMaxScaler
from financial_forecasting.data.splits import filter_by_split, get_split_name
from financial_forecasting.parameters.feature_params import (
    FeatureParams,
    RollingReturnTargetParams,
    ScalingParams,
    SplitParams,
    WindowParams,
)


def _save_split_npz(
    paths: ProjectPaths,
    prefix: str,
    feature_set: str,
    splits_data: dict[str, tuple[object, object, list[dict[str, object]]]],
) -> dict[str, object]:
    """Save split NPZ files and return a summaries dict.

    Parameters
    ----------
    paths:
        Project path container.
    prefix:
        Target type prefix, e.g. "rolling".
    feature_set:
        Feature set identifier, e.g. "original_ohlc".
    splits_data:
        Dict mapping split name -> (X, y, metadata_list).

    Returns
    -------
    split_summaries:
        Dict with per-split sample counts and ticker counts.
    """
    split_summaries: dict[str, object] = {}

    for split_name, (X_split, y_split, meta_split) in splits_data.items():
        assert isinstance(X_split, np.ndarray)
        assert isinstance(y_split, np.ndarray)
        assert isinstance(meta_split, list)

        npz_filename = f"{prefix}_{feature_set}_{split_name}.npz"
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

        print(f"  Saved {split_name}: {npz_path}")
        print(f"    X shape: {X_split.shape}, y shape: {y_split.shape}")
        print(f"    Ticker counts: {ticker_counts}")

        split_summaries[split_name] = {
            "total_samples": len(X_split),
            "ticker_counts": ticker_counts,
        }

    return split_summaries


def build_original_ohlc_rolling(
    paths: ProjectPaths,
    split_params: SplitParams,
    feat_params: FeatureParams,
    scale_params: ScalingParams,
    win_params: WindowParams,
    target_params: RollingReturnTargetParams,
) -> None:
    """Build rolling-average windows for the original OHLC feature set.

    Reuses the per-ticker MinMaxScaler fit during Phase 2. The same scaled
    features are used; only the target values differ (rolling average vs exact).
    """
    print("\n" + "=" * 60)
    print("Building rolling windows: original_ohlc")
    print("=" * 60)

    combined_csv = paths.data_raw / "all_tickers_ohlc.csv"
    print(f"Loading raw data from: {combined_csv}")
    df_raw = pd.read_csv(combined_csv, parse_dates=["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Fit scaler on training split only (leakage-safe)
    print("[1/4] Fitting scaler on training split (original_ohlc)...")
    df_train_only = filter_by_split(df_raw, "train", split_params)
    scaler = TickerMinMaxScaler(
        features=feat_params.columns,
        feature_range=(
            scale_params.feature_range_min,
            scale_params.feature_range_max,
        ),
    )
    scaler.fit(df_train_only)

    # Scale all rows
    print("[2/4] Scaling full dataset...")
    df_scaled = scaler.transform(df_raw)
    df_scaled["Raw_Close"] = df_raw["Close"]

    # Build rolling windows
    print(
        f"[3/4] Building rolling windows (T={win_params.lookback_window}, "
        f"rolling_window={target_params.rolling_window}, "
        f"weights={target_params.weights})..."
    )
    X, y, all_metadata = build_rolling_windows(
        df=df_scaled,
        features=feat_params.columns,
        lookback_window=win_params.lookback_window,
        horizons=target_params.horizons,
        rolling_window=target_params.rolling_window,
        weights=target_params.weights,
    )
    print(f"Total windows: {len(X)}, X shape: {X.shape}, y shape: {y.shape}")

    # Partition into splits
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
    print("[4/4] Saving rolling_original_ohlc NPZ files...")
    split_summaries = _save_split_npz(
        paths, "rolling", "original_ohlc", splits_data  # type: ignore[arg-type]
    )

    # Save metadata JSON
    metadata_json = {
        "schema_version": "1.0",
        "feature_set": "original_ohlc",
        "target_type": "rolling",
        "rolling_window": target_params.rolling_window,
        "weights": target_params.weights,
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
            "features": {"columns": feat_params.columns},
            "windowing": {"lookback_window": win_params.lookback_window},
            "targets": {
                "horizons": target_params.horizons,
                "rolling_window": target_params.rolling_window,
                "weights": target_params.weights,
            },
        },
        "splits_summary": split_summaries,
    }
    meta_path = paths.data_metadata / "rolling_original_ohlc_window_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


def build_auxiliary_ohlc_rolling(
    paths: ProjectPaths,
    split_params: SplitParams,
    win_params: WindowParams,
    target_params: RollingReturnTargetParams,
) -> None:
    """Build rolling-average windows for the auxiliary OHLC feature set.

    Recomputes auxiliary features from raw OHLC, fits a fresh scaler on training
    split only, then builds rolling-average windows.
    """
    print("\n" + "=" * 60)
    print("Building rolling windows: auxiliary_ohlc")
    print("=" * 60)

    combined_csv = paths.data_raw / "all_tickers_ohlc.csv"
    print(f"Loading raw data from: {combined_csv}")
    df_raw = pd.read_csv(combined_csv, parse_dates=["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Compute auxiliary features
    print("[1/4] Computing auxiliary features...")
    df_aux = compute_auxiliary_features(df_raw)
    aux_feature_cols = get_auxiliary_feature_columns()
    print(f"Auxiliary features ({len(aux_feature_cols)}): {aux_feature_cols}")
    print(f"Rows after NaN drop: {len(df_aux)}")

    # Fit scaler on training split only
    print("[2/4] Fitting auxiliary scaler on training split...")
    df_train_only = filter_by_split(df_aux, "train", split_params)
    scaler = TickerMinMaxScaler(
        features=aux_feature_cols,
        feature_range=(0.0, 1.0),
    )
    scaler.fit(df_train_only)

    # Scale all rows
    df_scaled = scaler.transform(df_aux)
    df_scaled["Raw_Close"] = df_aux["Close"].values

    # Build rolling windows
    print(
        f"[3/4] Building rolling windows (T={win_params.lookback_window}, "
        f"rolling_window={target_params.rolling_window})..."
    )
    X, y, all_metadata = build_rolling_windows(
        df=df_scaled,
        features=aux_feature_cols,
        lookback_window=win_params.lookback_window,
        horizons=target_params.horizons,
        rolling_window=target_params.rolling_window,
        weights=target_params.weights,
    )
    print(f"Total windows: {len(X)}, X shape: {X.shape}, y shape: {y.shape}")

    # Partition into splits
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
    print("[4/4] Saving rolling_auxiliary_ohlc NPZ files...")
    split_summaries = _save_split_npz(
        paths, "rolling", "auxiliary_ohlc", splits_data  # type: ignore[arg-type]
    )

    # Save metadata JSON
    metadata_json = {
        "schema_version": "1.0",
        "feature_set": "auxiliary_ohlc",
        "target_type": "rolling",
        "rolling_window": target_params.rolling_window,
        "weights": target_params.weights,
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
            "windowing": {"lookback_window": win_params.lookback_window},
            "targets": {
                "horizons": target_params.horizons,
                "rolling_window": target_params.rolling_window,
                "weights": target_params.weights,
            },
        },
        "splits_summary": split_summaries,
    }
    meta_path = paths.data_metadata / "rolling_auxiliary_ohlc_window_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    print(f"Metadata saved to: {meta_path}")


def main() -> None:
    """Orchestrate Phase 4 rolling window construction for both feature sets."""
    print("=" * 60)
    print("Phase 4 - Rolling-Average Return Window Construction")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    split_params = SplitParams()
    feat_params = FeatureParams()
    scale_params = ScalingParams()
    win_params = WindowParams()
    target_params = RollingReturnTargetParams()

    print(
        f"\nRolling target convention: window={target_params.rolling_window}, "
        f"weights={target_params.weights}"
    )

    # 1. original_ohlc
    build_original_ohlc_rolling(
        paths, split_params, feat_params, scale_params, win_params, target_params
    )

    # 2. auxiliary_ohlc
    build_auxiliary_ohlc_rolling(
        paths, split_params, win_params, target_params
    )

    print("\n" + "=" * 60)
    print("Phase 4 rolling window building complete!")
    print("  data/splits/rolling_original_ohlc_{train,val,test}.npz")
    print("  data/splits/rolling_auxiliary_ohlc_{train,val,test}.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
