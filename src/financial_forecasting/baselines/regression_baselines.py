"""Regression baseline predictors for exact and rolling return forecasting.

Implements four non-neural baseline predictors that serve as lower-bound
comparisons for LSTM/GRU models:

1. ZeroReturnBaseline      - predicts 0.0 for all horizons.
2. TrainMeanBaseline       - predicts per-horizon training mean.
3. PerStockTrainMeanBaseline - predicts per-ticker per-horizon training mean.
4. PreviousReturnBaseline  - predicts last observed close-to-close return.

Each baseline exposes:
    fit(X_train, y_train, tickers_train)
    predict(X, tickers) -> NDArray[np.float32]  shape (num_samples, num_horizons)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class ZeroReturnBaseline:
    """Baseline that always predicts zero return for all horizons.

    This tests whether neural models beat the simplest finance prior:
    that tomorrow's return is zero (i.e. market is a random walk).
    """

    num_horizons: int = 5
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        tickers_train: list[str] | None = None,
    ) -> ZeroReturnBaseline:
        """Fit the baseline (no-op: zero baseline needs no training data).

        Parameters
        ----------
        X_train:
            Training input sequences of shape (num_samples, T, F).
        y_train:
            Training targets of shape (num_samples, num_horizons).
        tickers_train:
            Ticker labels per sample (unused).

        Returns
        -------
        ZeroReturnBaseline:
            Self.
        """
        self.num_horizons = y_train.shape[1]
        self._fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float32],
        tickers: list[str] | None = None,
    ) -> NDArray[np.float32]:
        """Return zeros for every sample.

        Parameters
        ----------
        X:
            Input sequences of shape (num_samples, T, F).
        tickers:
            Ticker labels per sample (unused).

        Returns
        -------
        NDArray[np.float32]:
            Zero array of shape (num_samples, num_horizons).
        """
        return np.zeros((len(X), self.num_horizons), dtype=np.float32)


@dataclass
class TrainMeanBaseline:
    """Baseline that predicts the training-set mean return per horizon.

    Tests whether neural models beat the global conditional mean predictor.
    """

    num_horizons: int = 5
    _mean_per_horizon: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32),
        init=False,
        repr=False,
    )
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        tickers_train: list[str] | None = None,
    ) -> TrainMeanBaseline:
        """Compute per-horizon mean from training targets.

        Parameters
        ----------
        X_train:
            Training input sequences (unused by this baseline).
        y_train:
            Training targets of shape (num_samples, num_horizons).
        tickers_train:
            Ticker labels (unused).

        Returns
        -------
        TrainMeanBaseline:
            Self.
        """
        self.num_horizons = y_train.shape[1]
        self._mean_per_horizon = np.mean(y_train, axis=0).astype(np.float32)
        self._fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float32],
        tickers: list[str] | None = None,
    ) -> NDArray[np.float32]:
        """Broadcast per-horizon mean to all samples.

        Parameters
        ----------
        X:
            Input sequences of shape (num_samples, T, F).
        tickers:
            Ticker labels (unused).

        Returns
        -------
        NDArray[np.float32]:
            Predictions of shape (num_samples, num_horizons).
        """
        n = len(X)
        return np.tile(self._mean_per_horizon, (n, 1))


@dataclass
class PerStockTrainMeanBaseline:
    """Baseline that predicts per-ticker per-horizon training mean.

    Tests whether stock-specific average behavior explains performance.
    """

    num_horizons: int = 5
    _means: dict[str, NDArray[np.float32]] = field(
        default_factory=dict, init=False, repr=False
    )
    _global_mean: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(5, dtype=np.float32),
        init=False,
        repr=False,
    )
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        tickers_train: list[str] | None = None,
    ) -> PerStockTrainMeanBaseline:
        """Compute per-ticker per-horizon means from training targets.

        Parameters
        ----------
        X_train:
            Training input sequences (unused).
        y_train:
            Training targets of shape (num_samples, num_horizons).
        tickers_train:
            Ticker labels per sample (required).

        Returns
        -------
        PerStockTrainMeanBaseline:
            Self.
        """
        self.num_horizons = y_train.shape[1]
        self._global_mean = np.mean(y_train, axis=0).astype(np.float32)

        if tickers_train is None:
            self._means = {}
            self._fitted = True
            return self

        tickers_arr = np.array(tickers_train)
        unique_tickers = sorted(set(tickers_train))
        for ticker in unique_tickers:
            mask = tickers_arr == ticker
            if np.any(mask):
                self._means[ticker] = np.mean(y_train[mask], axis=0).astype(
                    np.float32
                )
        self._fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float32],
        tickers: list[str] | None = None,
    ) -> NDArray[np.float32]:
        """Return per-ticker mean for each sample.

        Parameters
        ----------
        X:
            Input sequences of shape (num_samples, T, F).
        tickers:
            Ticker labels per sample (required for stock-specific means).

        Returns
        -------
        NDArray[np.float32]:
            Predictions of shape (num_samples, num_horizons).
        """
        n = len(X)
        out = np.tile(self._global_mean, (n, 1))

        if tickers is not None:
            for i, ticker in enumerate(tickers):
                if ticker in self._means:
                    out[i] = self._means[ticker]

        return out.astype(np.float32)


@dataclass
class PreviousReturnBaseline:
    """Baseline that predicts the most recent 1-day close-to-close return.

    For anchor day t, predicts r_{t} = (Close[t] - Close[t-1]) / Close[t-1]
    for all horizons.  This tests whether the recurrent model beats a simple
    momentum / carry heuristic.

    The previous return is extracted from the last row of the lookback window:
        X[:, -1, close_feature_idx] - X[:, -2, close_feature_idx]
    divided by X[:, -2, close_feature_idx] (approximation using scaled values).

    Because the OHLC features are min-max scaled per ticker, this ratio
    approximates the close-to-close return in scaled space.  The exact
    raw return cannot be trivially recovered without the scaler.  However,
    this still provides a useful non-trivial temporal heuristic that any
    reasonable model should be able to beat.
    """

    close_feature_idx: int = 3  # Index of Close in feature matrix (0=Open,1=High,2=Low,3=Close)
    num_horizons: int = 5
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.float32],
        tickers_train: list[str] | None = None,
    ) -> PreviousReturnBaseline:
        """Fit the baseline (records horizon count from training targets).

        Parameters
        ----------
        X_train:
            Training inputs (shape verification only).
        y_train:
            Training targets to record num_horizons.
        tickers_train:
            Ticker labels (unused).

        Returns
        -------
        PreviousReturnBaseline:
            Self.
        """
        self.num_horizons = y_train.shape[1]
        self._fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float32],
        tickers: list[str] | None = None,
    ) -> NDArray[np.float32]:
        """Compute scaled previous-return and broadcast to all horizons.

        Parameters
        ----------
        X:
            Input sequences of shape (num_samples, T, F).
        tickers:
            Ticker labels (unused).

        Returns
        -------
        NDArray[np.float32]:
            Predictions of shape (num_samples, num_horizons).
        """
        # Last close (anchor day t)
        close_t = X[:, -1, self.close_feature_idx]
        # Second-to-last close (anchor day t-1)
        close_t_minus1 = X[:, -2, self.close_feature_idx]

        # Avoid division by zero for edge cases
        denom = np.where(
            np.abs(close_t_minus1) > 1e-8, close_t_minus1, 1e-8
        )
        prev_return = (close_t - denom) / denom  # shape (num_samples,)

        # Broadcast to all horizons
        pred = np.tile(prev_return[:, np.newaxis], (1, self.num_horizons))
        return pred.astype(np.float32)
