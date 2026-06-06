"""Phase 5 training script for bidirectional GRU turning-point classifier.

Mirrors train_turning_bilstm.py but uses StockBiGRUClassifier.

Usage:
    uv run python src/financial_forecasting/train/train_turning_bigru.py
    uv run python src/financial_forecasting/train/train_turning_bigru.py --feature-set auxiliary_ohlc
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Sized
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.data.turning_dataset import TurningPointDataset
from financial_forecasting.evaluation.classification_metrics import (
    compute_classification_metrics,
    compute_threshold_sweep,
)
from financial_forecasting.models.stock_bigru_classifier import (
    StockBiGRUClassifier,
)
from financial_forecasting.parameters.feature_params import WindowParams
from financial_forecasting.parameters.turning_point_params import (
    TurningBiGRUModelParams,
    TurningPointLabelParams,
    TurningTrainingParams,
)
from financial_forecasting.training.reproducibility import set_seeds
from financial_forecasting.utils.device import get_device


def get_git_commit() -> str | None:
    """Retrieve the current git commit hash, if available."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except Exception:
        return None


BatchType = tuple[torch.Tensor, torch.Tensor, str, str]


def train_epoch_clf(
    model: nn.Module,
    loader: DataLoader[BatchType],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train classifier for one epoch.

    Parameters
    ----------
    model:
        The classifier model.
    loader:
        DataLoader providing (X, label, ticker, date) tuples.
    optimizer:
        The optimizer.
    criterion:
        BCEWithLogitsLoss.
    device:
        Compute device.

    Returns
    -------
    float:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch, _, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)

    dataset = loader.dataset
    assert isinstance(dataset, Sized)
    return float(total_loss / len(dataset))


@torch.no_grad()
def validate_epoch_clf(
    model: nn.Module,
    loader: DataLoader[BatchType],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate on validation set and return average BCE loss.

    Parameters
    ----------
    model:
        The classifier model.
    loader:
        DataLoader for validation.
    criterion:
        BCEWithLogitsLoss.
    device:
        Compute device.

    Returns
    -------
    float:
        Average validation BCE loss.
    """
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch, _, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)

    dataset = loader.dataset
    assert isinstance(dataset, Sized)
    return float(total_loss / len(dataset))


@torch.no_grad()
def predict_split(
    model: nn.Module,
    loader: DataLoader[BatchType],
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[str], list[str]]:
    """Collect predicted probabilities and ground-truth labels.

    Parameters
    ----------
    model:
        Trained classifier.
    loader:
        DataLoader for the target split.
    device:
        Compute device.

    Returns
    -------
    tuple:
        (probabilities, true_labels, tickers, anchor_dates)
    """
    model.eval()
    all_probs: list[NDArray[np.float32]] = []
    all_labels: list[NDArray[np.float32]] = []
    all_tickers: list[str] = []
    all_dates: list[str] = []

    for X_batch, y_batch, tickers, dates in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        all_probs.append(probs)
        all_labels.append(y_batch.numpy().astype(np.float32))
        all_tickers.extend(list(tickers))
        all_dates.extend(list(dates))

    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels, axis=0),
        all_tickers,
        all_dates,
    )


def save_predictions_csv(
    out_path: object,
    split: str,
    tickers: list[str],
    anchor_dates: list[str],
    y_true: NDArray[np.float32],
    y_prob: NDArray[np.float32],
    threshold: float,
) -> None:
    """Save turning-point predictions CSV.

    Parameters
    ----------
    out_path:
        Destination path.
    split:
        Split name.
    tickers:
        Per-sample ticker labels.
    anchor_dates:
        Per-sample anchor dates.
    y_true:
        Ground-truth labels.
    y_prob:
        Predicted probabilities.
    threshold:
        Decision threshold for predicted labels.
    """
    y_pred = (y_prob >= threshold).astype(np.int32)
    df = pd.DataFrame(
        {
            "split": split,
            "ticker": tickers,
            "anchor_date": anchor_dates,
            "actual_label": y_true.astype(np.int32),
            "predicted_label": y_pred,
            "predicted_probability": y_prob,
        }
    )
    df.to_csv(str(out_path), index=False)


def main() -> None:
    """Orchestrate training of BiGRU turning-point classifier."""
    parser = argparse.ArgumentParser(
        description="Train bidirectional GRU turning-point classifier."
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="original_ohlc",
        choices=["original_ohlc", "auxiliary_ohlc"],
        help="Feature set to use (default: original_ohlc).",
    )
    args = parser.parse_args()
    feature_set = args.feature_set

    print("=" * 65)
    print(f"Phase 5 - BiGRU Turning-Point Classifier [{feature_set}]")
    print("=" * 65)

    paths = ProjectPaths()
    paths.ensure_all()

    model_params = TurningBiGRUModelParams()
    train_params = TurningTrainingParams()
    label_params = TurningPointLabelParams()
    win_params = WindowParams()

    set_seeds(train_params.seed)
    print(f"Seed: {train_params.seed}  |  Feature set: {feature_set}")

    def _npz(split_name: str) -> object:
        p = paths.data_splits / f"turning_{feature_set}_{split_name}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"NPZ not found: {p}\n"
                "Run build_turning_point_windows.py first."
            )
        return p

    train_ds = TurningPointDataset(_npz("train"))  # type: ignore[arg-type]
    val_ds = TurningPointDataset(_npz("val"))       # type: ignore[arg-type]
    test_ds = TurningPointDataset(_npz("test"))     # type: ignore[arg-type]

    train_loader = cast(
        DataLoader[BatchType],
        DataLoader(train_ds, batch_size=train_params.batch_size, shuffle=True),
    )
    val_loader = cast(
        DataLoader[BatchType],
        DataLoader(val_ds, batch_size=train_params.batch_size, shuffle=False),
    )
    test_loader = cast(
        DataLoader[BatchType],
        DataLoader(test_ds, batch_size=train_params.batch_size, shuffle=False),
    )

    print(f"\nDataset: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    y_train_all = train_ds.y.numpy()
    n_pos = int(np.sum(y_train_all == 1))
    n_neg = int(np.sum(y_train_all == 0))
    pos_rate = n_pos / len(y_train_all) if len(y_train_all) > 0 else 0.0
    print(f"Train class balance: pos={n_pos}  neg={n_neg}  pos_rate={pos_rate:.4f}")

    if pos_rate == 0.0:
        print(
            "  WARNING: No positive labels.  gamma=1.1 requires >110% gain; "
            "model will learn all-zero predictions."
        )

    pos_weight_val: float | None = None
    if train_params.use_pos_weight and n_pos > 0 and n_neg > 0:
        pos_weight_val = n_neg / n_pos

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_turning_bigru_{feature_set}_seed{train_params.seed}"
    run_dir = paths.results_runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    print(f"\nRun ID: {run_id}")

    input_size = train_ds.X.shape[2]
    model = StockBiGRUClassifier(
        input_size=input_size,
        hidden_size=model_params.hidden_size,
        num_layers=model_params.num_layers,
        dropout=model_params.dropout,
        output_size=model_params.output_size,
    )
    device = get_device(train_params.device)
    model = model.to(device)
    print(f"Model on {device}:\n{model}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_params.lr, weight_decay=train_params.weight_decay
    )
    if pos_weight_val is not None:
        pw = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_val_loss = math.inf
    patience_counter = 0
    best_ckpt = run_dir / "checkpoint_best.pt"
    last_ckpt = run_dir / "checkpoint_last.pt"

    print("\nStarting training...")
    for epoch in range(1, train_params.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch_clf(model, train_loader, optimizer, criterion, device)
        vl_loss = validate_epoch_clf(model, val_loader, criterion, device)
        dur = time.time() - t0
        lr_now = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": tr_loss,
                "val_loss": vl_loss,
                "learning_rate": lr_now,
                "epoch_seconds": dur,
            }
        )
        print(
            f"Epoch {epoch:03d}/{train_params.epochs} | "
            f"Train BCE: {tr_loss:.6f} | Val BCE: {vl_loss:.6f} | {dur:.2f}s"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "val_loss": vl_loss,
        }
        torch.save(ckpt, last_ckpt)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save(ckpt, best_ckpt)
            print(f"  --> Best at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= train_params.patience:
                print(f"\n[Early Stopping] after {train_params.patience} no-improve epochs.")
                break

    best_epoch = len(history)
    if best_ckpt.exists():
        ckpt_data = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt_data["model_state_dict"])
        best_epoch = int(ckpt_data["epoch"])
        best_val_loss = float(ckpt_data["val_loss"])
        print(f"\nLoaded best checkpoint from epoch {best_epoch}")

    train_prob, train_true, train_tickers, train_dates = predict_split(
        model, train_loader, device
    )
    val_prob, val_true, val_tickers, val_dates = predict_split(
        model, val_loader, device
    )
    test_prob, test_true, test_tickers, test_dates = predict_split(
        model, test_loader, device
    )

    thr = train_params.classification_threshold

    metrics_train = compute_classification_metrics(
        train_true.astype(np.int8), train_prob, train_tickers, thr
    )
    metrics_val = compute_classification_metrics(
        val_true.astype(np.int8), val_prob, val_tickers, thr
    )
    metrics_test = compute_classification_metrics(
        test_true.astype(np.int8), test_prob, test_tickers, thr
    )

    save_predictions_csv(
        run_dir / "predictions_train.csv",
        "train", train_tickers, train_dates, train_true, train_prob, thr,
    )
    save_predictions_csv(
        run_dir / "predictions_val.csv",
        "val", val_tickers, val_dates, val_true, val_prob, thr,
    )
    save_predictions_csv(
        run_dir / "predictions_test.csv",
        "test", test_tickers, test_dates, test_true, test_prob, thr,
    )

    for split_name, mets in [
        ("train", metrics_train),
        ("val", metrics_val),
        ("test", metrics_test),
    ]:
        p = run_dir / f"metrics_{split_name}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(mets, f, indent=2, default=float)

    class_balance_json = {
        "train": {"pos_rate": float(pos_rate), "n_pos": n_pos, "n_neg": n_neg, "total": len(y_train_all)},
        "val": {"pos_rate": float(np.mean(val_true == 1)), "n_pos": int(np.sum(val_true == 1)), "n_neg": int(np.sum(val_true == 0)), "total": len(val_true)},
        "test": {"pos_rate": float(np.mean(test_true == 1)), "n_pos": int(np.sum(test_true == 1)), "n_neg": int(np.sum(test_true == 0)), "total": len(test_true)},
    }
    with open(run_dir / "class_balance.json", "w", encoding="utf-8") as f:
        json.dump(class_balance_json, f, indent=2)

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    sweep = compute_threshold_sweep(test_true.astype(np.int8), test_prob)
    with open(run_dir / "threshold_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)

    config_json = {
        "model_type": "StockBiGRUClassifier",
        "feature_set": feature_set,
        "target_type": "turning_point",
        "model": {
            "input_size": input_size,
            "hidden_size": model_params.hidden_size,
            "num_layers": model_params.num_layers,
            "dropout": model_params.dropout,
            "bidirectional": model_params.bidirectional,
            "output_size": model_params.output_size,
        },
        "training": {
            "batch_size": train_params.batch_size,
            "epochs": train_params.epochs,
            "lr": train_params.lr,
            "weight_decay": train_params.weight_decay,
            "patience": train_params.patience,
            "seed": train_params.seed,
            "classification_threshold": thr,
            "pos_weight_applied": pos_weight_val,
        },
        "label": {
            "gamma": label_params.gamma,
            "horizons": label_params.horizons,
            "use_high_price": label_params.use_high_price,
        },
        "window": {"lookback_window": win_params.lookback_window},
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

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
        "sample_counts": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2)

    summary_md = f"""# Run Implementation Summary - {run_id}

## Experiment Configuration

- **Model:** StockBiGRUClassifier (bidirectional GRU) [{feature_set}]
  - Hidden size: {model_params.hidden_size} (x2 = {model_params.hidden_size*2} bidirectional)
  - Layers: {model_params.num_layers}  |  Dropout: {model_params.dropout}
  - Input features: {input_size}

## Class Balance

| Split | Total | Positives | Pos Rate |
|-------|-------|-----------|----------|
| Train | {class_balance_json['train']['total']} | {class_balance_json['train']['n_pos']} | {class_balance_json['train']['pos_rate']:.4f} |
| Val   | {class_balance_json['val']['total']} | {class_balance_json['val']['n_pos']} | {class_balance_json['val']['pos_rate']:.4f} |
| Test  | {class_balance_json['test']['total']} | {class_balance_json['test']['n_pos']} | {class_balance_json['test']['pos_rate']:.4f} |

## Training Summary

- Epochs: {len(history)}  |  Best epoch: {best_epoch}  |  Best val BCE: {best_val_loss:.6f}
- Test accuracy: {metrics_test.get('accuracy', float('nan')):.4f}  |  Test F1: {metrics_test.get('f1', float('nan')):.4f}
"""
    with open(run_dir / "implementation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
    with open(run_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Analysis Summary - {run_id}\n\nSee docs/05_turning_point_analysis.md.\n")

    print("\n" + "=" * 65)
    print(f"Phase 5 BiGRU training complete!  Run ID: {run_id}")
    print(f"Test accuracy: {metrics_test.get('accuracy', float('nan')):.4f}")
    print(f"Test F1:       {metrics_test.get('f1', float('nan')):.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
