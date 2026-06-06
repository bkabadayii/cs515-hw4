"""Phase 4 training script for rolling-average StockGRU.

Trains the GRU model on rolling-average return targets with early stopping
on the validation split and saves all history, checkpoints, metrics, and
prediction records.

Usage:
    uv run python src/financial_forecasting/train/train_rolling_gru.py
    uv run python src/financial_forecasting/train/train_rolling_gru.py --feature-set auxiliary_ohlc
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.data.torch_dataset import StockDataset
from financial_forecasting.evaluation.prediction_records import (
    save_prediction_records,
)
from financial_forecasting.evaluation.regression_metrics import (
    compute_regression_metrics,
)
from financial_forecasting.models.stock_gru import StockGRU
from financial_forecasting.parameters.feature_params import (
    RollingReturnTargetParams,
    SplitParams,
    WindowParams,
)
from financial_forecasting.parameters.rolling_gru_params import (
    RollingGRUModelParams,
    RollingGRUTrainingParams,
)
from financial_forecasting.training.loops import run_training_pipeline
from financial_forecasting.training.reproducibility import set_seeds
from financial_forecasting.utils.device import get_device

# Type alias for DataLoader yields
BatchType = tuple[torch.Tensor, torch.Tensor, list[str], list[str]]


def get_git_commit() -> str | None:
    """Retrieve the current git commit hash, if available."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate_and_predict(
    model: nn.Module,
    loader: DataLoader[BatchType],
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str], list[str]]:
    """Predict outputs and gather actual targets, tickers, and dates.

    Parameters
    ----------
    model:
        Model to evaluate.
    loader:
        DataLoader containing evaluation samples.
    device:
        Selected compute device.

    Returns
    -------
    tuple:
        (predictions, actuals, tickers, anchor_dates)
    """
    model.eval()
    all_preds: list[NDArray[np.float32]] = []
    all_trues: list[NDArray[np.float32]] = []
    all_tickers: list[str] = []
    all_dates: list[str] = []

    for X_batch, y_batch, tickers, dates in loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(y_batch.numpy())
        all_tickers.extend(list(tickers))
        all_dates.extend(list(dates))

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_trues, axis=0),
        all_tickers,
        all_dates,
    )


def main() -> None:
    """Orchestrate training of rolling-average StockGRU."""
    parser = argparse.ArgumentParser(
        description="Train rolling-average StockGRU."
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="original_ohlc",
        choices=["original_ohlc", "auxiliary_ohlc"],
        help="Feature set to use for training (default: original_ohlc).",
    )
    args = parser.parse_args()
    feature_set = args.feature_set

    print("=" * 60)
    print(f"Phase 4 - Rolling-Average StockGRU Training [{feature_set}]")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    model_params = RollingGRUModelParams()
    train_params = RollingGRUTrainingParams()
    split_params = SplitParams()
    win_params = WindowParams()
    target_params = RollingReturnTargetParams()

    set_seeds(train_params.seed)
    print(f"Random seed: {train_params.seed}")
    print(f"Feature set: {feature_set}")
    print(
        f"Rolling target: window={target_params.rolling_window}, "
        f"weights={target_params.weights}"
    )

    # Resolve NPZ paths
    def _resolve_npz(split: str) -> object:
        npz_path = paths.data_splits / f"rolling_{feature_set}_{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Rolling NPZ not found: {npz_path}\n"
                "Run build_rolling_return_windows.py first."
            )
        return npz_path

    train_npz = _resolve_npz("train")
    val_npz = _resolve_npz("val")
    test_npz = _resolve_npz("test")

    train_dataset = StockDataset(train_npz)  # type: ignore[arg-type]
    val_dataset = StockDataset(val_npz)  # type: ignore[arg-type]
    test_dataset = StockDataset(test_npz)  # type: ignore[arg-type]

    train_loader = cast(
        DataLoader[BatchType],
        DataLoader(
            train_dataset, batch_size=train_params.batch_size, shuffle=True
        ),
    )
    val_loader = cast(
        DataLoader[BatchType],
        DataLoader(
            val_dataset, batch_size=train_params.batch_size, shuffle=False
        ),
    )
    test_loader = cast(
        DataLoader[BatchType],
        DataLoader(
            test_dataset, batch_size=train_params.batch_size, shuffle=False
        ),
    )

    print("Dataset Split Sizes:")
    print(f"  Train:      {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test:       {len(test_dataset)} samples")

    # Unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_rolling_stock_gru_{feature_set}_seed{train_params.seed}"
    run_dir = paths.results_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    print(f"Run ID: {run_id}")

    # Instantiate model
    input_size = train_dataset.X.shape[2]
    output_size = train_dataset.y.shape[1]

    model = StockGRU(
        input_size=input_size,
        hidden_size=model_params.hidden_size,
        num_layers=model_params.num_layers,
        dropout=model_params.dropout,
        output_size=output_size,
    )

    device = get_device(train_params.device)
    model = model.to(device)
    print(f"StockGRU on device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params.lr,
        weight_decay=train_params.weight_decay,
    )
    criterion = nn.MSELoss()

    print("\nStarting model training...")
    history = run_training_pipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        params=train_params,
        device=device,
        run_dir=run_dir,
    )

    # Load best checkpoint
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["val_loss"])
        print(
            f"\nBest checkpoint: epoch {best_epoch}, val_loss={best_val_loss:.6f}"
        )
    else:
        best_epoch = len(history)
        best_val_loss = history[-1]["val_loss"]

    # Evaluate all splits
    print("\nEvaluating all splits...")
    train_preds, train_trues, train_tickers, train_dates = evaluate_and_predict(
        model, train_loader, device
    )
    val_preds, val_trues, val_tickers, val_dates = evaluate_and_predict(
        model, val_loader, device
    )
    test_preds, test_trues, test_tickers, test_dates = evaluate_and_predict(
        model, test_loader, device
    )

    # Compute metrics
    metrics_train = compute_regression_metrics(
        train_trues, train_preds, train_tickers, target_params.horizons
    )
    metrics_val = compute_regression_metrics(
        val_trues, val_preds, val_tickers, target_params.horizons
    )
    metrics_test = compute_regression_metrics(
        test_trues, test_preds, test_tickers, target_params.horizons
    )

    # Save metrics JSONs
    for split_name, metrics in [
        ("train", metrics_train),
        ("val", metrics_val),
        ("test", metrics_test),
    ]:
        with open(run_dir / f"metrics_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {split_name} metrics.")

    # Save prediction records
    raw_ohlc_path = paths.data_raw / "all_tickers_ohlc.csv"
    save_prediction_records(
        run_dir / "predictions_train.csv", "train",
        train_tickers, train_dates, train_trues, train_preds, raw_ohlc_path,
    )
    save_prediction_records(
        run_dir / "predictions_val.csv", "val",
        val_tickers, val_dates, val_trues, val_preds, raw_ohlc_path,
    )
    save_prediction_records(
        run_dir / "predictions_test.csv", "test",
        test_tickers, test_dates, test_trues, test_preds, raw_ohlc_path,
    )

    # Save history CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(run_dir / "history.csv", index=False)

    # Save config.json
    config_json = {
        "model_type": "StockGRU",
        "feature_set": feature_set,
        "target_type": "rolling",
        "rolling_window": target_params.rolling_window,
        "rolling_weights": target_params.weights,
        "model": {
            "input_size": input_size,
            "hidden_size": model_params.hidden_size,
            "num_layers": model_params.num_layers,
            "dropout": model_params.dropout,
            "output_size": output_size,
        },
        "training": {
            "batch_size": train_params.batch_size,
            "epochs": train_params.epochs,
            "lr": train_params.lr,
            "weight_decay": train_params.weight_decay,
            "patience": train_params.patience,
            "seed": train_params.seed,
            "device_setting": train_params.device,
        },
        "splits": {
            "train_start": split_params.train_start,
            "train_end": split_params.train_end,
            "val_start": split_params.val_start,
            "val_end": split_params.val_end,
            "test_start": split_params.test_start,
            "test_end": split_params.test_end,
        },
        "window": {"lookback_window": win_params.lookback_window},
        "targets": {"horizons": target_params.horizons},
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    # Save run_metadata.json
    metadata_json = {
        "experiment_id": run_id,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "seed": train_params.seed,
        "script_name": __file__,
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "sample_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)

    # Save implementation_summary.md
    summary_md = f"""# Run Implementation Summary - {run_id}

## Experiment Configuration

- **Model Architecture:** Stacked GRU (StockGRU) [{feature_set}]
  - Hidden Size: {model_params.hidden_size}
  - Layers: {model_params.num_layers}
  - Dropout: {model_params.dropout}
  - Input Features: {input_size}
- **Target Type:** Rolling-average (window={target_params.rolling_window}, weights={target_params.weights})
- **Optimization:**
  - Optimizer: AdamW
  - LR: {train_params.lr}
  - Weight Decay: {train_params.weight_decay}
  - Loss: MSE
- **Training:**
  - Batch Size: {train_params.batch_size}
  - Max Epochs: {train_params.epochs}
  - Patience: {train_params.patience}
  - Seed: {train_params.seed}
  - Device: {device}

## Training Summary

- **Epochs Completed:** {len(history)}
- **Best Epoch:** {best_epoch}
- **Best Val Loss:** {best_val_loss:.6f}
- **Test MSE:** {metrics_test['overall_mse']:.6f}
- **Test RMSE:** {metrics_test['overall_rmse']:.6f}
- **Test MAE:** {metrics_test['overall_mae']:.6f}
- **Test R2:** {metrics_test['overall_r2']:.4f}

## Saved Deliverables

- `config.json`, `run_metadata.json`, `history.csv`
- `metrics_train.json`, `metrics_val.json`, `metrics_test.json`
- `predictions_train.csv`, `predictions_val.csv`, `predictions_test.csv`
- `checkpoint_best.pt`, `checkpoint_last.pt`
"""
    with open(run_dir / "implementation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    with open(run_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(
            f"# Analysis Summary - {run_id}\n\n"
            "See docs/04_rolling_forecasting_workbench_analysis.md for full analysis.\n"
        )

    print("\n" + "=" * 60)
    print(f"Phase 4 rolling GRU training complete! Run ID: {run_id}")
    print(f"Test MSE: {metrics_test['overall_mse']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
