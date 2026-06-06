"""Training loop implementations with early stopping and checkpointing.

The training pipeline is parameterized via a TrainingParamsProtocol so that
any dataclass implementing the required attributes (epochs, patience,
batch_size) can be used.  Both ExactLSTMTrainingParams and ExactGRUTrainingParams
satisfy this protocol.
"""

from __future__ import annotations

import pathlib
import time
from collections.abc import Sized
from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class TrainingParamsProtocol(Protocol):
    """Protocol for training parameter dataclasses.

    Any frozen dataclass that provides these attributes is compatible
    with run_training_pipeline.
    """

    epochs: int
    patience: int
    batch_size: int


# Type alias for the loader yield type to avoid Any
BatchType = tuple[torch.Tensor, torch.Tensor, list[str], list[str]]


def train_epoch(
    model: nn.Module,
    loader: DataLoader[BatchType],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch.

    Parameters
    ----------
    model:
        The model to train.
    loader:
        DataLoader containing training sequences.
    optimizer:
        The optimizer.
    criterion:
        The loss function.
    device:
        The PyTorch device.

    Returns
    -------
    float:
        The average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch, _, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    dataset = loader.dataset
    assert isinstance(dataset, Sized)
    return float(total_loss / len(dataset))


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader[BatchType],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on a validation set.

    Parameters
    ----------
    model:
        The model to evaluate.
    loader:
        DataLoader containing validation sequences.
    criterion:
        The loss function.
    device:
        The PyTorch device.

    Returns
    -------
    float:
        The average validation loss.
    """
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch, _, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        total_loss += loss.item() * len(X_batch)

    dataset = loader.dataset
    assert isinstance(dataset, Sized)
    return float(total_loss / len(dataset))


def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader[BatchType],
    val_loader: DataLoader[BatchType],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    params: TrainingParamsProtocol,
    device: torch.device,
    run_dir: pathlib.Path,
) -> list[dict[str, float]]:
    """Execute the training and validation loop with early stopping.

    Saves checkpoint_best.pt and checkpoint_last.pt in run_dir.

    Parameters
    ----------
    model:
        The model to train.
    train_loader:
        DataLoader for the training split.
    val_loader:
        DataLoader for the validation split.
    optimizer:
        PyTorch optimizer.
    criterion:
        Loss function (MSE).
    params:
        Training parameters satisfying TrainingParamsProtocol.
    device:
        Device to run computations on.
    run_dir:
        Directory path to save checkpoints.

    Returns
    -------
    list[dict[str, float]]:
        Training history with loss, learning rate, and time per epoch.
    """
    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    patience_counter = 0

    run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    last_checkpoint_path = run_dir / "checkpoint_last.pt"

    for epoch in range(1, params.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate_epoch(model, val_loader, criterion, device)

        duration = time.time() - start_time
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "epoch_seconds": duration,
            }
        )

        print(
            f"Epoch {epoch:03d}/{params.epochs:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {duration:.2f}s"
        )

        # Checkpoint last epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            last_checkpoint_path,
        )

        # Early stopping and best checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= params.patience:
                print(
                    f"\n[Early Stopping] No improvement for {params.patience} epochs. Stopping."
                )
                break

    return history
