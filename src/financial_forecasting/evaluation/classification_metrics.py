"""Classification metrics for turning-point detection (Phase 5).

Computes accuracy, precision, recall, F1, confusion matrix, class balance
statistics, ROC-AUC, and PR-AUC.  All functions handle degenerate cases
(e.g. all-negative predictions or only one class present in y_true).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 if denominator is zero."""
    return numerator / denominator if denominator != 0.0 else 0.0


def compute_classification_metrics(
    y_true: NDArray[np.int8 | np.float32 | np.int64],
    y_prob: NDArray[np.float32 | np.float64],
    tickers: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, float | int | list[list[int]] | dict[str, float]]:
    """Compute full classification metric suite.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 or 1), shape (N,).
    y_prob:
        Predicted probabilities (after sigmoid), shape (N,).
    tickers:
        Optional ticker labels per sample for per-stock metrics.
    threshold:
        Decision boundary for converting probabilities to labels.

    Returns
    -------
    dict:
        accuracy, precision, recall, f1, confusion_matrix,
        positive_class_rate, predicted_positive_rate,
        roc_auc (if computable), pr_auc (if computable),
        per_stock_metrics (if tickers provided).
    """
    y_true_int = np.asarray(y_true, dtype=np.int32)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob_f >= threshold).astype(np.int32)

    n = len(y_true_int)
    tp = int(np.sum((y_pred == 1) & (y_true_int == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true_int == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true_int == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true_int == 1)))

    accuracy = _safe_divide(tp + tn, n)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    positive_class_rate = _safe_divide(int(np.sum(y_true_int == 1)), n)
    predicted_positive_rate = _safe_divide(int(np.sum(y_pred == 1)), n)

    confusion_matrix: list[list[int]] = [[tn, fp], [fn, tp]]

    result: dict[str, float | int | list[list[int]] | dict[str, float]] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix,
        "positive_class_rate": positive_class_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "threshold_used": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    # ROC-AUC and PR-AUC: only computable if both classes present
    unique_classes = np.unique(y_true_int)
    if len(unique_classes) == 2:
        result["roc_auc"] = _compute_roc_auc(y_true_int, y_prob_f)
        result["pr_auc"] = _compute_pr_auc(y_true_int, y_prob_f)
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")
        result["roc_pr_note"] = (
            "Only one class present in y_true; ROC-AUC and PR-AUC are undefined."
        )

    # Per-stock metrics
    if tickers is not None:
        per_stock = _per_stock_metrics(y_true_int, y_pred, y_prob_f, tickers, threshold)
        result["per_stock_metrics"] = per_stock

    return result


def _compute_roc_auc(
    y_true: NDArray[np.int32], y_prob: NDArray[np.float64]
) -> float:
    """Compute ROC-AUC via trapezoidal rule.

    Uses a threshold sweep rather than importing sklearn so there is no
    additional dependency.
    """
    thresholds = np.sort(np.unique(y_prob))[::-1]
    tprs: list[float] = []
    fprs: list[float] = []

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    for thr in thresholds:
        pred = (y_prob >= thr).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tprs.append(tp / n_pos)
        fprs.append(fp / n_neg)

    # Add corner points
    tprs = [0.0] + tprs + [1.0]
    fprs = [0.0] + fprs + [1.0]

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fprs)):
        dx = fprs[i] - fprs[i - 1]
        auc += dx * (tprs[i] + tprs[i - 1]) / 2.0
    return float(abs(auc))


def _compute_pr_auc(
    y_true: NDArray[np.int32], y_prob: NDArray[np.float64]
) -> float:
    """Compute PR-AUC (area under the precision-recall curve)."""
    thresholds = np.sort(np.unique(y_prob))[::-1]
    precisions: list[float] = []
    recalls: list[float] = []

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return float("nan")

    for thr in thresholds:
        pred = (y_prob >= thr).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        prec = _safe_divide(tp, tp + fp)
        rec = _safe_divide(tp, tp + fn)
        precisions.append(prec)
        recalls.append(rec)

    # Add endpoint
    recalls = recalls + [0.0]
    precisions = precisions + [1.0]

    auc = 0.0
    for i in range(1, len(recalls)):
        dr = abs(recalls[i] - recalls[i - 1])
        auc += dr * (precisions[i] + precisions[i - 1]) / 2.0
    return float(auc)


def _per_stock_metrics(
    y_true: NDArray[np.int32],
    y_pred: NDArray[np.int32],
    y_prob: NDArray[np.float64],
    tickers: list[str],
    threshold: float,
) -> dict[str, float]:
    """Compute per-ticker classification accuracy."""
    tickers_arr = np.array(tickers)
    result: dict[str, float] = {}
    for ticker in sorted(set(tickers)):
        mask = tickers_arr == ticker
        if not np.any(mask):
            continue
        n = int(np.sum(mask))
        tp_i = int(np.sum((y_pred[mask] == 1) & (y_true[mask] == 1)))
        tn_i = int(np.sum((y_pred[mask] == 0) & (y_true[mask] == 0)))
        result[f"{ticker}_accuracy"] = _safe_divide(tp_i + tn_i, n)
        result[f"{ticker}_positive_rate"] = float(
            np.mean(y_true[mask].astype(np.float64))
        )
        result[f"{ticker}_predicted_positive_rate"] = float(
            np.mean(y_pred[mask].astype(np.float64))
        )
    return result


def compute_threshold_sweep(
    y_true: NDArray[np.int8 | np.float32 | np.int64],
    y_prob: NDArray[np.float32 | np.float64],
    num_thresholds: int = 101,
) -> dict[str, list[float]]:
    """Sweep classification threshold and record precision, recall, F1.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities.
    num_thresholds:
        Number of evenly spaced thresholds between 0 and 1.

    Returns
    -------
    dict:
        Keys: thresholds, precision, recall, f1  (all lists of length
        num_thresholds).
    """
    y_true_int = np.asarray(y_true, dtype=np.int32)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)

    thresholds = np.linspace(0.0, 1.0, num_thresholds).tolist()
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for thr in thresholds:
        y_pred = (y_prob_f >= thr).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true_int == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_int == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_int == 1)))
        prec = _safe_divide(tp, tp + fp)
        rec = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * prec * rec, prec + rec)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "thresholds": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
    }


def compute_pr_curve_points(
    y_true: NDArray[np.int8 | np.float32 | np.int64],
    y_prob: NDArray[np.float32 | np.float64],
) -> tuple[list[float], list[float], list[float]]:
    """Compute precision-recall curve data points.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted probabilities.

    Returns
    -------
    tuple:
        (precision_list, recall_list, threshold_list)
    """
    y_true_int = np.asarray(y_true, dtype=np.int32)
    y_prob_f = np.asarray(y_prob, dtype=np.float64)

    thresholds = np.sort(np.unique(y_prob_f))[::-1]
    precisions: list[float] = []
    recalls: list[float] = []

    for thr in thresholds:
        pred = (y_prob_f >= thr).astype(np.int32)
        tp = int(np.sum((pred == 1) & (y_true_int == 1)))
        fp = int(np.sum((pred == 1) & (y_true_int == 0)))
        fn = int(np.sum((pred == 0) & (y_true_int == 1)))
        precisions.append(_safe_divide(tp, tp + fp))
        recalls.append(_safe_divide(tp, tp + fn))

    return precisions, recalls, thresholds.tolist()
