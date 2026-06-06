"""Feature scaling utilities.

Provides per-ticker MinMaxScaler to avoid cross-ticker scaling bias and
leakage across chronological boundaries.
"""
from __future__ import annotations

import json
import pathlib
import pandas as pd


class TickerMinMaxScaler:
    """Min-max scaler that fits and transforms features independently for each ticker.

    This ensures that absolute price differences between tickers do not dominate
    the neural network inputs, while preventing leakage from validation/test
    periods if fit on the training split only.
    """

    def __init__(
        self,
        features: list[str],
        feature_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Initialize the TickerMinMaxScaler.

        Parameters
        ----------
        features:
            The column names in the DataFrame to scale.
        feature_range:
            The target minimum and maximum values for scaling.
        """
        self.features = features
        self.feature_range = feature_range
        # Maps ticker -> feature_name -> (min, max)
        self.scales: dict[str, dict[str, tuple[float, float]]] = {}

    def fit(self, df: pd.DataFrame) -> TickerMinMaxScaler:
        """Calculate min/max for each feature and ticker combination.

        Parameters
        ----------
        df:
            The training DataFrame containing 'Ticker' and feature columns.

        Returns
        -------
        TickerMinMaxScaler:
            Self.
        """
        self.scales = {}
        tickers = df["Ticker"].unique()
        for ticker in tickers:
            ticker_df = df[df["Ticker"] == ticker]
            self.scales[ticker] = {}
            for col in self.features:
                if col not in ticker_df.columns:
                    raise ValueError(f"Feature column {col} not found in DataFrame.")
                col_min = float(ticker_df[col].min())
                col_max = float(ticker_df[col].max())
                self.scales[ticker][col] = (col_min, col_max)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max scaling to the features in the DataFrame.

        Parameters
        ----------
        df:
            The DataFrame to scale.

        Returns
        -------
        pd.DataFrame:
            A copy of the input DataFrame with scaled features.
        """
        if not self.scales:
            raise ValueError("Scaler must be fit before transform is called.")

        df_out = df.copy()
        min_out, max_out = self.feature_range

        for ticker, ticker_scales in self.scales.items():
            mask = df_out["Ticker"] == ticker
            if not mask.any():
                continue

            for col, (col_min, col_max) in ticker_scales.items():
                if col not in df_out.columns:
                    raise ValueError(f"Feature column {col} not found in DataFrame.")

                denom = col_max - col_min
                if denom == 0.0:
                    df_out.loc[mask, col] = min_out
                else:
                    scaled = (df_out.loc[mask, col] - col_min) / denom
                    df_out.loc[mask, col] = (
                        scaled * (max_out - min_out) + min_out
                    )

        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on a DataFrame and immediately transform it.

        Parameters
        ----------
        df:
            The training DataFrame.

        Returns
        -------
        pd.DataFrame:
            Scaled training DataFrame.
        """
        return self.fit(df).transform(df)

    def save(self, path: pathlib.Path | str) -> None:
        """Serialize scaling statistics to JSON.

        Parameters
        ----------
        path:
            File path to save the scaling config.
        """
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        serializable = {
            "features": self.features,
            "feature_range": list(self.feature_range),
            "scales": {
                ticker: {col: [val[0], val[1]] for col, val in col_scales.items()}
                for ticker, col_scales in self.scales.items()
            },
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path | str) -> TickerMinMaxScaler:
        """Deserialize scaler statistics from JSON.

        Parameters
        ----------
        path:
            File path to load scaling config from.

        Returns
        -------
        TickerMinMaxScaler:
            A fitted scaler.
        """
        p = pathlib.Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        scaler = cls(
            features=data["features"],
            feature_range=(data["feature_range"][0], data["feature_range"][1]),
        )
        scaler.scales = {
            ticker: {col: (val[0], val[1]) for col, val in col_scales.items()}
            for ticker, col_scales in data["scales"].items()
        }
        return scaler
