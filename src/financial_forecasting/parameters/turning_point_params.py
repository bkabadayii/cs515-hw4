"""Parameter definitions for turning-point detection (Phase 5).

All configurations are frozen dataclasses; no typing.Any is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TurningPointLabelParams:
    """Parameters controlling binary buy/pass label construction.

    Attributes
    ----------
    gamma:
        The assignment threshold.  A window is labelled 'buy' (1) if
        any max-price return over d = 1..5 exceeds gamma.
        Formula: r_{t+d} = (High[t+d] - Close[t]) / Close[t] > gamma.
        With gamma = 1.1 this requires a 110% gain, so positive labels
        are extremely rare in real data.  The value is kept at 1.1 as
        required by the assignment.
    horizons:
        Look-ahead horizons (d = 1..5).
    use_high_price:
        When True, use the High column for p_max (as recommended by the
        plan).  When False, use Close.
    """

    gamma: float = 1.1
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    use_high_price: bool = True


@dataclass(frozen=True)
class TurningBiLSTMModelParams:
    """Hyperparameters for the bidirectional LSTM classifier.

    Attributes
    ----------
    input_size:
        Number of input features per timestep.
    hidden_size:
        Per-direction hidden-state size.  The final FC layer receives
        hidden_size * 2 because of the bidirectional concatenation.
    num_layers:
        Number of stacked BiLSTM layers.
    dropout:
        Dropout between stacked layers (ignored when num_layers == 1).
    bidirectional:
        Always True for this classifier.  Kept explicit for clarity.
    output_size:
        Output logit dimension (1 for binary classification).
    """

    input_size: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    output_size: int = 1


@dataclass(frozen=True)
class TurningBiGRUModelParams:
    """Hyperparameters for the bidirectional GRU classifier.

    Mirrors TurningBiLSTMModelParams but for GRU cells.
    """

    input_size: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    output_size: int = 1


@dataclass(frozen=True)
class TurningTrainingParams:
    """Hyperparameters and configuration settings for classifier training.

    Attributes
    ----------
    batch_size:
        Mini-batch size.
    epochs:
        Maximum training epochs.
    lr:
        Learning rate for AdamW.
    weight_decay:
        L2 regularisation strength.
    patience:
        Early-stopping patience (validation BCE loss).
    seed:
        Random seed for reproducibility.
    device:
        Compute device ('cpu', 'cuda', 'mps', or 'auto').
    classification_threshold:
        Decision threshold applied to sigmoid probabilities.
    use_pos_weight:
        When True, compute a pos_weight for BCEWithLogitsLoss from the
        training-label distribution to partially compensate for class
        imbalance.  Has no effect when there are zero positive labels.
    """

    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    seed: int = 42
    device: str = "auto"
    classification_threshold: float = 0.5
    use_pos_weight: bool = True
