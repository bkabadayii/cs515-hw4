"""Parameter definitions for auxiliary OHLC feature set.

Auxiliary features extend the original 4-column OHLC set with derived
technical indicators that may improve predictive performance.
All features are computed from historical data only (no future leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuxiliaryFeatureParams:
    """Parameters defining the auxiliary OHLC feature set.

    Attributes
    ----------
    base_columns:
        Raw OHLC columns from the dataset (not scaled; used in ratio computation).
    scaled_columns:
        Scaled OHLC column names produced after scaler application.
    sma_windows:
        Rolling windows used for simple moving average ratio features.
    volatility_window:
        Rolling window for close-to-close return standard deviation.
    feature_set_name:
        Canonical name for this feature set.
    """

    base_columns: list[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close"]
    )
    scaled_columns: list[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close"]
    )
    sma_windows: list[int] = field(default_factory=lambda: [5, 10])
    volatility_window: int = 5
    feature_set_name: str = "auxiliary_ohlc"

    @property
    def all_feature_names(self) -> list[str]:
        """Return the full ordered list of auxiliary feature column names."""
        return [
            "Open",
            "High",
            "Low",
            "Close",
            "close_to_close_return",
            "open_to_close_return",
            "high_low_range_pct",
            "close_sma_5_ratio",
            "close_sma_10_ratio",
            "rolling_volatility_5",
        ]


@dataclass(frozen=True)
class OriginalFeatureParams:
    """Parameters for the original OHLC-only feature set.

    Attributes
    ----------
    feature_set_name:
        Canonical name for this feature set.
    columns:
        Feature column names.
    """

    feature_set_name: str = "original_ohlc"
    columns: list[str] = field(
        default_factory=lambda: ["Open", "High", "Low", "Close"]
    )
