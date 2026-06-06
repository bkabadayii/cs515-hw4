"""Parameter definitions for feature construction, splits, and windowing.

All configurations must be explicit dataclasses with no typing.Any.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SplitParams:
    """Date boundaries for chronological splits.

    All dates should be in YYYY-MM-DD format.
    """

    train_start: str = "2020-01-01"
    train_end: str = "2024-07-31"
    val_start: str = "2024-08-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31"


@dataclass(frozen=True)
class FeatureParams:
    """Parameters defining what features to construct from raw OHLC data."""

    columns: list[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close"]
    )


@dataclass(frozen=True)
class ScalingParams:
    """Scaling parameters for input features."""

    scaler_type: str = "minmax"
    feature_range_min: float = 0.0
    feature_range_max: float = 1.0
    scaler_save_path: pathlib.Path = field(
        default_factory=lambda: pathlib.Path("data/metadata/scaler_config.json")
    )


@dataclass(frozen=True)
class WindowParams:
    """Sliding window parameters."""

    lookback_window: int = 20  # T = 20


@dataclass(frozen=True)
class ExactReturnTargetParams:
    """Target parameterization for exact return forecasting."""

    horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])


@dataclass(frozen=True)
class RollingReturnTargetParams:
    """Target parameterization for rolling-average return forecasting.

    Convention (Phase 4):
        rolling_window = 3  (l = 3, num_terms = l, NOT l+1)
        weights = [1/3, 1/3, 1/3]   (uniform)

    For horizon d, anchor t:
        r_roll_{t+d} = (1/3) * sum_{j=0}^{2} (close[t+d-j] - close[t]) / close[t]
                     = mean of exact returns for t+d, t+d-1, t+d-2

    This smooths the target by averaging the last rolling_window exact returns
    ending at t+d. For d=1, the average spans (t+1, t, t-1) where close[t] and
    close[t-1] are always available inside the lookback window (T=20 >= 2).

    Attributes
    ----------
    rolling_window:
        Number of future close prices to average (l = 3).
    weights:
        Per-term averaging weights (must sum to 1.0).
    horizons:
        Prediction horizons d = 1..5.
    """

    rolling_window: int = 3
    weights: list[float] = field(
        default_factory=lambda: [1 / 3, 1 / 3, 1 / 3]
    )
    horizons: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
