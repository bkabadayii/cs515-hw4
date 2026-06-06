"""Phase 3 training script for exact-return StockLSTM.

Trains the LSTM model with early stopping on validation split and saves all
history, checkpoints, test metrics, and prediction records.

Usage:
    uv run python src/financial_forecasting/train/train_exact_lstm.py
    uv run python src/financial_forecasting/train/train_exact_lstm.py --feature-set auxiliary_ohlc
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
from financial_forecasting.models.stock_lstm import StockLSTM
from financial_forecasting.parameters.exact_lstm_params import (
    ExactLSTMModelParams,
    ExactLSTMTrainingParams,
)
from financial_forecasting.parameters.feature_params import (
    ExactReturnTargetParams,
    SplitParams,
    WindowParams,
)
from financial_forecasting.training.loops import run_training_pipeline
from financial_forecasting.training.reproducibility import set_seeds
from financial_forecasting.utils.device import get_device


def get_git_commit() -> str | None:
    """Retrieve the current git commit hash, if available."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return None


# Type alias for DataLoader yields
BatchType = tuple[torch.Tensor, torch.Tensor, list[str], list[str]]


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
    tuple[NDArray[np.float32], NDArray[np.float32], list[str], list[str]]:
        - predicted returns: shape (num_samples, 5)
        - actual targets: shape (num_samples, 5)
        - ticker list: length num_samples
        - anchor date list: length num_samples
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
    """Orchestrate training of Exact-return StockLSTM."""
    parser = argparse.ArgumentParser(
        description="Train exact-return StockLSTM."
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
    print(f"Phase 3 - Exact-return StockLSTM Training [{feature_set}]")
    print("=" * 60)

    # Initialize configuration and paths
    paths = ProjectPaths()
    paths.ensure_all()

    # Load parameters
    model_params = ExactLSTMModelParams()
    train_params = ExactLSTMTrainingParams()
    split_params = SplitParams()
    win_params = WindowParams()
    target_params = ExactReturnTargetParams()

    # Set random seeds for reproducibility
    set_seeds(train_params.seed)
    print(f"Random seed locked to: {train_params.seed}")
    print(f"Feature set: {feature_set}")

    # Resolve NPZ paths for selected feature set
    def _resolve_npz(split: str) -> object:
        canonical = paths.data_splits / f"exact_{feature_set}_{split}.npz"
        legacy = paths.data_splits / f"exact_{split}.npz"
        if canonical.exists():
            return canonical
        elif feature_set == "original_ohlc" and legacy.exists():
            return legacy
        else:
            raise FileNotFoundError(
                f"NPZ not found: {canonical}\nRun the build script first."
            )

    train_npz = _resolve_npz("train")
    val_npz = _resolve_npz("val")
    test_npz = _resolve_npz("test")

    train_dataset = StockDataset(train_npz)  # type: ignore[arg-type]
    val_dataset = StockDataset(val_npz)  # type: ignore[arg-type]
    test_dataset = StockDataset(test_npz)  # type: ignore[arg-type]

    # Create DataLoaders
    # Shuffle training data to break sequence correlation across tickers,
    # but keep val/test ordered for evaluation.
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

    # Set up unique run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_exact_stock_lstm_{feature_set}_seed{train_params.seed}"
    run_dir = paths.results_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    print(f"Experiment Run ID: {run_id}")
    print(f"Run folder created at: {run_dir}")

    # Instantiate model
    input_size = train_dataset.X.shape[2]
    output_size = train_dataset.y.shape[1]

    model = StockLSTM(
        input_size=input_size,
        hidden_size=model_params.hidden_size,
        num_layers=model_params.num_layers,
        dropout=model_params.dropout,
        output_size=output_size,
    )

    device = get_device(train_params.device)
    model = model.to(device)
    print(f"Initialized StockLSTM model on device: {device}")
    print(f"Model Architecture:\n{model}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params.lr,
        weight_decay=train_params.weight_decay,
    )
    criterion = nn.MSELoss()

    # Train model
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

    # Re-load the best validation model checkpoint for final evaluations
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["val_loss"])
        print(
            f"\nLoaded best validation model checkpoint from epoch {best_epoch} "
            f"with Val Loss: {best_val_loss:.6f}"
        )
    else:
        best_epoch = len(history)
        best_val_loss = history[-1]["val_loss"]
        print(
            "\nWarning: best checkpoint not found. Using final weights instead."
        )

    # Evaluate on all splits
    print("\nEvaluating and saving prediction records...")
    train_preds, train_trues, train_tickers, train_dates = evaluate_and_predict(
        model, train_loader, device
    )
    val_preds, val_trues, val_tickers, val_dates = evaluate_and_predict(
        model, val_loader, device
    )
    test_preds, test_trues, test_tickers, test_dates = evaluate_and_predict(
        model, test_loader, device
    )

    # Compute regression metrics
    metrics_train = compute_regression_metrics(
        train_trues, train_preds, train_tickers, target_params.horizons
    )
    metrics_val = compute_regression_metrics(
        val_trues, val_preds, val_tickers, target_params.horizons
    )
    metrics_test = compute_regression_metrics(
        test_trues, test_preds, test_tickers, target_params.horizons
    )

    # Reconstruct target dates and save predictions CSVs
    raw_ohlc_path = paths.data_raw / "all_tickers_ohlc.csv"
    save_prediction_records(
        run_dir / "predictions_train.csv",
        "train",
        train_tickers,
        train_dates,
        train_trues,
        train_preds,
        raw_ohlc_path,
    )
    save_prediction_records(
        run_dir / "predictions_val.csv",
        "val",
        val_tickers,
        val_dates,
        val_trues,
        val_preds,
        raw_ohlc_path,
    )
    save_prediction_records(
        run_dir / "predictions_test.csv",
        "test",
        test_tickers,
        test_dates,
        test_trues,
        test_preds,
        raw_ohlc_path,
    )

    # Save metrics JSONs
    for split_name, metrics in [
        ("train", metrics_train),
        ("val", metrics_val),
        ("test", metrics_test),
    ]:
        save_path = run_dir / f"metrics_{split_name}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {split_name} metrics to: {save_path}")

    # Save training history CSV
    df_hist = pd.DataFrame(history)
    history_csv_path = run_dir / "history.csv"
    df_hist.to_csv(history_csv_path, index=False)
    print(f"Saved history CSV to: {history_csv_path}")

    # Save config.json
    config_json = {
        "model_type": "StockLSTM",
        "feature_set": feature_set,
        "target_type": "exact",
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
        "window": {
            "lookback_window": win_params.lookback_window,
        },
        "targets": {
            "horizons": target_params.horizons,
        },
    }
    config_save_path = run_dir / "config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)
    print(f"Saved config parameters to: {config_save_path}")

    # Save metadata JSON
    git_commit = get_git_commit()
    metadata_json = {
        "experiment_id": run_id,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "git_commit": git_commit,
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
    metadata_save_path = run_dir / "run_metadata.json"
    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)
    print(f"Saved run metadata to: {metadata_save_path}")

    # Save implementation_summary.md in the run folder
    summary_md = f"""# Run Implementation Summary - {run_id}

## Experiment Configuration

- **Model Architecture:** Stacked LSTM (StockLSTM) [{feature_set}]
  - Hidden Size: {model_params.hidden_size}
  - Layers: {model_params.num_layers}
  - Dropout: {model_params.dropout}
  - Input Features: {input_size}
- **Optimization settings:**
  - Optimizer: AdamW
  - Learning Rate: {train_params.lr}
  - Weight Decay: {train_params.weight_decay}
  - Loss Function: Mean Squared Error (MSE)
- **Training specifications:**
  - Batch Size: {train_params.batch_size}
  - Maximum Epochs: {train_params.epochs}
  - Patience (Early Stopping): {train_params.patience}
  - Random Seed: {train_params.seed}
  - Compute Device: {device} (requested: {train_params.device})

## Training Summary

- **Total Epochs Completed:** {len(history)}
- **Best Validation Epoch:** {best_epoch}
- **Best Validation Loss:** {best_val_loss:.6f}
- **Final Test Loss (MSE):** {metrics_test["overall_mse"]:.6f}
- **Final Test RMSE:** {metrics_test["overall_rmse"]:.6f}
- **Final Test MAE:** {metrics_test["overall_mae"]:.6f}
- **Final Test R2:** {metrics_test["overall_r2"]:.4f}

## Saved Deliverables

All outputs are preserved in this immutable run directory:
- `config.json`: Run configurations.
- `run_metadata.json`: Environmental execution context and git reference.
- `history.csv`: Per-epoch training and validation loss log.
- `metrics_train.json` / `metrics_val.json` / `metrics_test.json`: Standard evaluation summaries.
- `predictions_train.csv` / `predictions_val.csv` / `predictions_test.csv`: Forecast records.
- `checkpoint_best.pt` / `checkpoint_last.pt`: Model weights.
"""
    summary_save_path = run_dir / "implementation_summary.md"
    with open(summary_save_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Saved run implementation summary markdown to: {summary_save_path}")

    with open(run_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(
            f"# Analysis Summary - {run_id}\n\n"
            "See docs/03_exact_forecasting_workbench_analysis.md for full analysis.\n"
        )

    print("\n" + "=" * 60)
    print(f"Phase 3 training complete! Run Folder ID: {run_id}")
    print(f"Test MSE: {metrics_test['overall_mse']:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
