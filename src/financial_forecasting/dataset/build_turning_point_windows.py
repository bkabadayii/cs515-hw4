"""Phase 5 dataset builder: turning-point windows.

Builds binary buy/pass labels for both original_ohlc and auxiliary_ohlc
feature sets and saves compressed NPZ files.

Usage:
    uv run python src/financial_forecasting/dataset/build_turning_point_windows.py

Label definition
----------------
    r_{t+d} = (High[t+d] - Close[t]) / Close[t]
    label   = 1  if  max_{d=1..5}  r_{t+d} > gamma  (gamma = 1.1)
            = 0  otherwise

Because gamma = 1.1 requires a 110%+ gain in 5 trading days, positive
labels are extremely rare in real S&P 500 data.  This script reports
the exact class balance for each split so the effect can be documented.
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
from financial_forecasting.data.turning_point_windows import (
    build_turning_point_windows,
    compute_class_balance,
)
from financial_forecasting.parameters.auxiliary_feature_params import (
    OriginalFeatureParams,
)
from financial_forecasting.parameters.feature_params import (
    ScalingParams,
    SplitParams,
    WindowParams,
)
from financial_forecasting.parameters.turning_point_params import (
    TurningPointLabelParams,
)


def _build_and_save(
    df_scaled: pd.DataFrame,
    
    features: list[str],
    feature_set_name: str,
    split_params: SplitParams,
    label_params: TurningPointLabelParams,
    win_params: WindowParams,
    paths: ProjectPaths,
) -> dict[str, object]:
    """Build turning-point windows for one feature set and save NPZs.

    Parameters
    ----------
    df_scaled:
        Scaled DataFrame with all feature columns, Date, Ticker, Raw_Close,
        and Raw_High.
    df_raw_high:
        Raw High price series aligned to df_scaled index.
    features:
        Ordered feature column names.
    feature_set_name:
        Name used in output filenames (e.g. 'original_ohlc').
    split_params:
        Chronological split boundaries.
    label_params:
        Turning-point label configuration (gamma, horizons, etc.).
    win_params:
        Window construction parameters (lookback_window).
    paths:
        Project path helper.

    Returns
    -------
    dict:
        Summary of class balance per split for metadata.
    """
    print(f"\n  Building windows for feature set: {feature_set_name}")
    print(f"  Features ({len(features)}): {features}")

    # Build all windows across the full dataset
    X, y, max_future_return, tickers, anchor_dates = build_turning_point_windows(
        df=df_scaled,
        features=features,
        gamma=label_params.gamma,
        horizons=label_params.horizons,
        lookback_window=win_params.lookback_window,
        use_high_price=label_params.use_high_price,
    )

    print(f"  Total windows generated: {len(X)}")

    # Partition by anchor date
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for i, ad in enumerate(anchor_dates):
        split = get_split_name(ad, split_params)
        if split == "train":
            train_idx.append(i)
        elif split == "val":
            val_idx.append(i)
        elif split == "test":
            test_idx.append(i)

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    summary: dict[str, object] = {}

    for split_name, indices in split_indices.items():
        idx_arr = np.array(indices, dtype=np.int64)
        X_split = X[idx_arr]
        y_split = y[idx_arr]
        mfr_split = max_future_return[idx_arr]
        tickers_arr = np.array([tickers[i] for i in indices], dtype=object)
        dates_arr = np.array([anchor_dates[i] for i in indices], dtype=object)

        balance = compute_class_balance(y_split)

        npz_path = paths.data_splits / f"turning_{feature_set_name}_{split_name}.npz"
        np.savez_compressed(
            npz_path,
            X=X_split,
            y=y_split,
            max_future_return=mfr_split,
            tickers=tickers_arr,
            anchor_dates=dates_arr,
        )

        print(
            f"  Saved {split_name:5s}: {len(X_split):5d} samples  "
            f"| pos={balance['num_positive']:4d}  "
            f"| pos_rate={balance['positive_rate']:.4f}  "
            f"-> {npz_path}"
        )

        summary[split_name] = balance

    return summary


def main() -> None:
    """Orchestrate Phase 5 turning-point window building."""
    print("=" * 65)
    print("Phase 5 - Build Turning-Point Windows")
    print("=" * 65)

    paths = ProjectPaths()
    paths.ensure_all()

    # Load raw OHLC
    combined_csv = paths.data_raw / "all_tickers_ohlc.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(
            f"Raw OHLC not found at {combined_csv}. Run download_yfinance.py first."
        )

    df_raw = pd.read_csv(combined_csv, parse_dates=["Date"])
    df_raw = df_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    print(f"Loaded raw OHLC: {len(df_raw)} rows, {df_raw['Ticker'].nunique()} tickers")

    # Parameters
    split_params = SplitParams()
    scale_params = ScalingParams()
    win_params = WindowParams()
    label_params = TurningPointLabelParams()
    orig_params = OriginalFeatureParams()

    print("\nLabel parameters:")
    print(f"  gamma          = {label_params.gamma}")
    print(f"  horizons       = {label_params.horizons}")
    print(f"  use_high_price = {label_params.use_high_price}")
    print(
        f"\n  NOTE: gamma={label_params.gamma} means r > 1.1, i.e. High > 2.1 * Close."
    )
    print(
        "  This requires a >110% gain in 5 trading days. Positive labels will be"
    )
    print(
        "  extremely rare (likely 0) in real data. Class balance is reported below."
    )

    # ---- ORIGINAL OHLC ----------------------------------------
    print("\n[1/2] Original OHLC feature set")
    print("-" * 50)

    orig_features = orig_params.columns  # ["Open", "High", "Low", "Close"]

    # Fit scaler on training rows only
    df_train_only = filter_by_split(df_raw, "train", split_params)
    scaler_orig = TickerMinMaxScaler(
        features=orig_features,
        feature_range=(scale_params.feature_range_min, scale_params.feature_range_max),
    )
    scaler_orig.fit(df_train_only)
    df_orig_scaled = scaler_orig.transform(df_raw)
    df_orig_scaled["Raw_Close"] = df_raw["Close"].values
    df_orig_scaled["Raw_High"] = df_raw["High"].values

    summary_orig = _build_and_save(
        df_scaled=df_orig_scaled,
        
        features=orig_features,
        feature_set_name="original_ohlc",
        split_params=split_params,
        label_params=label_params,
        win_params=win_params,
        paths=paths,
    )

    # ---- AUXILIARY OHLC ----------------------------------------
    print("\n[2/2] Auxiliary OHLC feature set")
    print("-" * 50)

    df_aux_raw = compute_auxiliary_features(df_raw)
    df_aux_raw = df_aux_raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    aux_features = get_auxiliary_feature_columns()  # 10 features

    df_aux_train_only = filter_by_split(df_aux_raw, "train", split_params)
    scaler_aux = TickerMinMaxScaler(
        features=aux_features,
        feature_range=(scale_params.feature_range_min, scale_params.feature_range_max),
    )
    scaler_aux.fit(df_aux_train_only)
    df_aux_scaled = scaler_aux.transform(df_aux_raw)
    df_aux_scaled["Raw_Close"] = df_aux_raw["Close"].values
    df_aux_scaled["Raw_High"] = df_aux_raw["High"].values

    summary_aux = _build_and_save(
        df_scaled=df_aux_scaled,
        
        features=aux_features,
        feature_set_name="auxiliary_ohlc",
        split_params=split_params,
        label_params=label_params,
        win_params=win_params,
        paths=paths,
    )

    # ---- Save metadata ----------------------------------------
    metadata = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "label_params": {
            "gamma": label_params.gamma,
            "horizons": label_params.horizons,
            "use_high_price": label_params.use_high_price,
        },
        "window_params": {
            "lookback_window": win_params.lookback_window,
        },
        "split_params": {
            "train_start": split_params.train_start,
            "train_end": split_params.train_end,
            "val_start": split_params.val_start,
            "val_end": split_params.val_end,
            "test_start": split_params.test_start,
            "test_end": split_params.test_end,
        },
        "class_balance": {
            "original_ohlc": summary_orig,
            "auxiliary_ohlc": summary_aux,
        },
    }

    meta_path = paths.data_metadata / "turning_point_window_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {meta_path}")

    print("\n" + "=" * 65)
    print("Phase 5 window building complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
