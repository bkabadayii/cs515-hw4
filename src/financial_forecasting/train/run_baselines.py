"""Phase 3B - Run all exact-return regression baselines.

Evaluates four non-neural baseline predictors against the existing
exact-return train/val/test windows and saves results in run-folder
format compatible with the aggregate analysis pipeline.

Baselines:
    1. zero_return           - predicts 0.0 for all horizons
    2. train_mean            - per-horizon training mean
    3. per_stock_train_mean  - per-ticker per-horizon training mean
    4. previous_return       - last close-to-close return propagated

Usage:
    uv run python src/financial_forecasting/train/run_baselines.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from financial_forecasting.baselines.regression_baselines import (
    PerStockTrainMeanBaseline,
    PreviousReturnBaseline,
    TrainMeanBaseline,
    ZeroReturnBaseline,
)
from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.data.torch_dataset import StockDataset
from financial_forecasting.evaluation.prediction_records import (
    save_prediction_records,
)
from financial_forecasting.evaluation.regression_metrics import (
    compute_regression_metrics,
)
from financial_forecasting.parameters.feature_params import (
    ExactReturnTargetParams,
    SplitParams,
)


def run_one_baseline(
    name: str,
    tag: str,
    predictor: (
        ZeroReturnBaseline
        | TrainMeanBaseline
        | PerStockTrainMeanBaseline
        | PreviousReturnBaseline
    ),
    X_train: NDArray[np.float32],
    y_train: NDArray[np.float32],
    tickers_train: list[str],
    X_val: NDArray[np.float32],
    y_val: NDArray[np.float32],
    tickers_val: list[str],
    X_test: NDArray[np.float32],
    y_test: NDArray[np.float32],
    tickers_test: list[str],
    dates_train: list[str],
    dates_val: list[str],
    dates_test: list[str],
    paths: ProjectPaths,
    split_params: SplitParams,
    target_params: ExactReturnTargetParams,
    raw_ohlc_path: object,
) -> str:
    """Fit and evaluate one baseline, save run folder, return run_id."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_exact_{tag}_baseline"
    run_dir = paths.results_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Baseline: {name}")
    print(f"Run ID: {run_id}")

    # Fit on training data
    predictor.fit(X_train, y_train, tickers_train)

    # Predict all splits
    pred_train = predictor.predict(X_train, tickers_train)
    pred_val = predictor.predict(X_val, tickers_val)
    pred_test = predictor.predict(X_test, tickers_test)

    # Compute metrics
    metrics_train = compute_regression_metrics(
        y_train, pred_train, tickers_train, target_params.horizons
    )
    metrics_val = compute_regression_metrics(
        y_val, pred_val, tickers_val, target_params.horizons
    )
    metrics_test = compute_regression_metrics(
        y_test, pred_test, tickers_test, target_params.horizons
    )

    print(
        f"  Test MSE: {metrics_test['overall_mse']:.6f} | "
        f"RMSE: {metrics_test['overall_rmse']:.6f} | "
        f"MAE: {metrics_test['overall_mae']:.6f}"
    )

    # Save metrics
    for split_name, metrics in [
        ("train", metrics_train),
        ("val", metrics_val),
        ("test", metrics_test),
    ]:
        with open(run_dir / f"metrics_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # Save prediction records
    assert isinstance(raw_ohlc_path, object)
    save_prediction_records(
        run_dir / "predictions_train.csv",
        "train",
        tickers_train,
        dates_train,
        y_train,
        pred_train,
        raw_ohlc_path,  # type: ignore[arg-type]
    )
    save_prediction_records(
        run_dir / "predictions_val.csv",
        "val",
        tickers_val,
        dates_val,
        y_val,
        pred_val,
        raw_ohlc_path,  # type: ignore[arg-type]
    )
    save_prediction_records(
        run_dir / "predictions_test.csv",
        "test",
        tickers_test,
        dates_test,
        y_test,
        pred_test,
        raw_ohlc_path,  # type: ignore[arg-type]
    )

    # Save config.json
    config_json = {
        "baseline_name": name,
        "baseline_tag": tag,
        "feature_set": "original_ohlc",
        "target_type": "exact",
        "splits": {
            "train_start": split_params.train_start,
            "train_end": split_params.train_end,
            "val_start": split_params.val_start,
            "val_end": split_params.val_end,
            "test_start": split_params.test_start,
            "test_end": split_params.test_end,
        },
        "horizons": target_params.horizons,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    # Save run_metadata.json
    metadata_json = {
        "experiment_id": run_id,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "python_version": sys.version,
        "seed": "N/A (deterministic baseline)",
        "script_name": __file__,
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "sample_counts": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)

    # Save implementation_summary.md
    summary_md = f"""# Baseline Run Summary - {run_id}

## Baseline: {name}

### Definition

{_get_baseline_definition(tag)}

### Evaluation Results

| Split | MSE | RMSE | MAE | R2 | Dir. Acc |
|---|---|---|---|---|---|
| Train | {metrics_train['overall_mse']:.6f} | {metrics_train['overall_rmse']:.6f} | {metrics_train['overall_mae']:.6f} | {metrics_train['overall_r2']:.4f} | {metrics_train['overall_directional_accuracy']:.4f} |
| Val   | {metrics_val['overall_mse']:.6f} | {metrics_val['overall_rmse']:.6f} | {metrics_val['overall_mae']:.6f} | {metrics_val['overall_r2']:.4f} | {metrics_val['overall_directional_accuracy']:.4f} |
| Test  | {metrics_test['overall_mse']:.6f} | {metrics_test['overall_rmse']:.6f} | {metrics_test['overall_mae']:.6f} | {metrics_test['overall_r2']:.4f} | {metrics_test['overall_directional_accuracy']:.4f} |

### Saved Files

- `config.json`, `run_metadata.json`, `metrics_*.json`, `predictions_*.csv`
"""
    with open(run_dir / "implementation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    # Placeholder analysis_summary.md (will be written after joint analysis)
    with open(run_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Analysis Summary - {run_id}\n\nSee docs/03_exact_forecasting_workbench_analysis.md for full analysis.\n")

    print(f"  Run saved to: {run_dir}")
    return run_id


def _get_baseline_definition(tag: str) -> str:
    """Return a human-readable definition string for each baseline tag."""
    defs = {
        "zero_return": (
            "Predicts 0.0 for all horizons. "
            "Tests whether the neural model beats the simplest finance prior (random walk)."
        ),
        "train_mean": (
            "Predicts the per-horizon mean return computed from the training split. "
            "Tests whether the neural model beats the global conditional mean."
        ),
        "per_stock_train_mean": (
            "Predicts the per-ticker, per-horizon mean return from the training split. "
            "Tests whether stock-specific average behavior explains performance."
        ),
        "previous_return": (
            "Predicts the most recent scaled close-to-close return (from the lookback window) "
            "for all horizons. Tests whether the recurrent model beats a simple temporal heuristic."
        ),
    }
    return defs.get(tag, "No definition available.")


def main() -> None:
    """Orchestrate all four exact-return baseline evaluations."""
    print("=" * 60)
    print("Phase 3B - Exact-Return Regression Baselines")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    split_params = SplitParams()
    target_params = ExactReturnTargetParams()
    raw_ohlc_path = paths.data_raw / "all_tickers_ohlc.csv"

    # Load the original OHLC splits
    # Try canonical name first, fall back to legacy name
    def _load_npz(split: str) -> StockDataset:
        canonical = paths.data_splits / f"exact_original_ohlc_{split}.npz"
        legacy = paths.data_splits / f"exact_{split}.npz"
        if canonical.exists():
            return StockDataset(canonical)
        elif legacy.exists():
            return StockDataset(legacy)
        else:
            raise FileNotFoundError(
                f"NPZ file not found: {canonical} or {legacy}\n"
                "Please run build_exact_return_windows.py first."
            )

    print("\nLoading split datasets...")
    train_ds = _load_npz("train")
    val_ds = _load_npz("val")
    test_ds = _load_npz("test")

    X_train = train_ds.X.numpy()
    y_train = train_ds.y.numpy()
    tickers_train = train_ds.tickers
    dates_train = train_ds.anchor_dates

    X_val = val_ds.X.numpy()
    y_val = val_ds.y.numpy()
    tickers_val = val_ds.tickers
    dates_val = val_ds.anchor_dates

    X_test = test_ds.X.numpy()
    y_test = test_ds.y.numpy()
    tickers_test = test_ds.tickers
    dates_test = test_ds.anchor_dates

    print(f"  Train: {len(X_train)} samples, X shape: {X_train.shape}")
    print(f"  Val:   {len(X_val)} samples,  X shape: {X_val.shape}")
    print(f"  Test:  {len(X_test)} samples, X shape: {X_test.shape}")

    # Define baselines
    baselines: list[
        tuple[
            str,
            str,
            ZeroReturnBaseline
            | TrainMeanBaseline
            | PerStockTrainMeanBaseline
            | PreviousReturnBaseline,
        ]
    ] = [
        ("Zero Return", "zero_return", ZeroReturnBaseline()),
        ("Train Mean", "train_mean", TrainMeanBaseline()),
        ("Per-Stock Train Mean", "per_stock_train_mean", PerStockTrainMeanBaseline()),
        ("Previous Return", "previous_return", PreviousReturnBaseline()),
    ]

    run_ids: list[str] = []

    for name, tag, predictor in baselines:
        run_id = run_one_baseline(
            name=name,
            tag=tag,
            predictor=predictor,
            X_train=X_train,
            y_train=y_train,
            tickers_train=tickers_train,
            X_val=X_val,
            y_val=y_val,
            tickers_val=tickers_val,
            X_test=X_test,
            y_test=y_test,
            tickers_test=tickers_test,
            dates_train=dates_train,
            dates_val=dates_val,
            dates_test=dates_test,
            paths=paths,
            split_params=split_params,
            target_params=target_params,
            raw_ohlc_path=raw_ohlc_path,
        )
        run_ids.append(run_id)

    # Save baseline run IDs for aggregation
    baseline_registry = paths.results_aggregate / "baseline_run_ids.json"
    with open(baseline_registry, "w", encoding="utf-8") as f:
        json.dump({"exact_baseline_run_ids": run_ids}, f, indent=2)
    print(f"\nBaseline run IDs saved to: {baseline_registry}")

    print("\n" + "=" * 60)
    print("Phase 3B baselines complete!")
    for rid in run_ids:
        print(f"  results/runs/{rid}")
    print("=" * 60)


if __name__ == "__main__":
    main()
