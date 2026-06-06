"""Regression evaluation metrics for return forecasting.

Provides overall, per-horizon, and per-stock metrics including MSE, RMSE,
MAE, R-squared, Pearson correlation, directional accuracy, and prediction
vs target standard deviation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats


def compute_regression_metrics(
    y_true: NDArray[np.float32],
    y_pred: NDArray[np.float32],
    tickers: list[str] | None = None,
    horizons: list[int] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute overall, per-horizon, and per-stock regression metrics.

    Parameters
    ----------
    y_true:
        Numpy array of actual returns, shape (num_samples, num_horizons).
    y_pred:
        Numpy array of predicted returns, shape (num_samples, num_horizons).
    tickers:
        List of tickers for each sample. Required for per-stock calculations.
    horizons:
        List of integer prediction horizons (e.g., [1, 2, 3, 4, 5]).

    Returns
    -------
    dict:
        A dictionary containing overall metrics, per-horizon metrics, and
        per-stock metrics.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5]

    # Flatten for overall computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Overall MSE / RMSE / MAE
    mse = float(np.mean((y_true_flat - y_pred_flat) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))

    # Overall R-squared
    ss_res = float(np.sum((y_true_flat - y_pred_flat) ** 2))
    ss_tot = float(np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    # Overall Pearson correlation
    if np.std(y_true_flat) > 0.0 and np.std(y_pred_flat) > 0.0:
        corr_result = scipy_stats.pearsonr(y_true_flat, y_pred_flat)
        overall_pearson = float(corr_result[0])
    else:
        overall_pearson = 0.0

    # Overall directional accuracy
    overall_dir_acc = float(np.mean(np.sign(y_true_flat) == np.sign(y_pred_flat)))

    # Target and prediction std
    target_std_overall = float(np.std(y_true_flat))
    pred_std_overall = float(np.std(y_pred_flat))

    # Per-horizon metrics
    per_horizon_mse: dict[str, float] = {}
    per_horizon_rmse: dict[str, float] = {}
    per_horizon_mae: dict[str, float] = {}
    per_horizon_dir_acc: dict[str, float] = {}
    per_horizon_pearson: dict[str, float] = {}
    target_std_by_horizon: dict[str, float] = {}
    prediction_std_by_horizon: dict[str, float] = {}

    for i, h in enumerate(horizons):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        h_mse = float(np.mean((yt - yp) ** 2))
        h_rmse = float(np.sqrt(h_mse))
        h_mae = float(np.mean(np.abs(yt - yp)))
        h_dir_acc = float(np.mean(np.sign(yt) == np.sign(yp)))
        h_target_std = float(np.std(yt))
        h_pred_std = float(np.std(yp))

        if np.std(yt) > 0.0 and np.std(yp) > 0.0:
            h_corr = float(scipy_stats.pearsonr(yt, yp)[0])
        else:
            h_corr = 0.0

        per_horizon_mse[f"horizon_{h}"] = h_mse
        per_horizon_rmse[f"horizon_{h}"] = h_rmse
        per_horizon_mae[f"horizon_{h}"] = h_mae
        per_horizon_dir_acc[f"horizon_{h}"] = h_dir_acc
        per_horizon_pearson[f"horizon_{h}"] = h_corr
        target_std_by_horizon[f"horizon_{h}"] = h_target_std
        prediction_std_by_horizon[f"horizon_{h}"] = h_pred_std

    # Per-stock metrics
    per_stock_mse: dict[str, float] = {}
    per_stock_rmse: dict[str, float] = {}
    per_stock_mae: dict[str, float] = {}
    per_stock_per_horizon_mse: dict[str, dict[str, float]] = {}

    if tickers is not None:
        unique_tickers = sorted(list(set(tickers)))
        tickers_arr = np.array(tickers)
        for ticker in unique_tickers:
            mask = tickers_arr == ticker
            if np.any(mask):
                t_mse = float(np.mean((y_true[mask] - y_pred[mask]) ** 2))
                t_rmse = float(np.sqrt(t_mse))
                t_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

                per_stock_mse[ticker] = t_mse
                per_stock_rmse[ticker] = t_rmse
                per_stock_mae[ticker] = t_mae

                # Per-stock per-horizon MSE
                per_stock_per_horizon_mse[ticker] = {}
                for i, h in enumerate(horizons):
                    ph_mse = float(
                        np.mean((y_true[mask, i] - y_pred[mask, i]) ** 2)
                    )
                    per_stock_per_horizon_mse[ticker][f"horizon_{h}"] = ph_mse

    return {
        "overall_mse": mse,
        "overall_rmse": rmse,
        "overall_mae": mae,
        "overall_r2": r2,
        "overall_pearson_corr": overall_pearson,
        "overall_directional_accuracy": overall_dir_acc,
        "target_std_overall": target_std_overall,
        "prediction_std_overall": pred_std_overall,
        "per_horizon_mse": per_horizon_mse,
        "per_horizon_rmse": per_horizon_rmse,
        "per_horizon_mae": per_horizon_mae,
        "per_horizon_directional_accuracy": per_horizon_dir_acc,
        "per_horizon_pearson_corr": per_horizon_pearson,
        "target_std_by_horizon": target_std_by_horizon,
        "prediction_std_by_horizon": prediction_std_by_horizon,
        "per_stock_mse": per_stock_mse,
        "per_stock_rmse": per_stock_rmse,
        "per_stock_mae": per_stock_mae,
        "per_stock_per_horizon_mse": per_stock_per_horizon_mse,
    }
