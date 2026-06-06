"""Parameters for exact-return StockGRU model and training loop.

Uses the same hyperparameter defaults as ExactLSTMTrainingParams to ensure
fair comparison between LSTM and GRU architectures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExactGRUModelParams:
    """Hyperparameters defining the StockGRU architecture.

    Attributes
    ----------
    input_size:
        Number of input features per time step (default = 4: Open, High, Low, Close).
    hidden_size:
        Dimension of the GRU hidden state.
    num_layers:
        Number of stacked GRU layers.
    dropout:
        Dropout probability applied between stacked GRU layers.
    output_size:
        Number of forecasting horizons (default = 5: d = 1..5).
    """

    input_size: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 5


@dataclass(frozen=True)
class ExactGRUTrainingParams:
    """Hyperparameters and configuration settings for GRU training.

    Attributes
    ----------
    batch_size:
        Size of mini-batches for optimization.
    epochs:
        Maximum number of epochs to train.
    lr:
        Learning rate for the optimizer.
    weight_decay:
        L2 regularization penalty.
    patience:
        Early stopping validation patience.
    seed:
        Random state seed for reproducibility.
    device:
        Selected compute device ("cpu", "cuda", "mps", "auto").
    """

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 42
    device: str = "auto"
