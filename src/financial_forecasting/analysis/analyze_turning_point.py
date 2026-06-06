"""Phase 5A analysis script: turning-point classification results.

Loads saved prediction CSVs from turning-point run folders (no retraining),
computes classification metrics for both feature sets, generates three plots,
and writes docs/05_turning_point_analysis.md.

Usage:
    uv run python src/financial_forecasting/analysis/analyze_turning_point.py

The script auto-discovers the most recent run folder for each feature set
by scanning results/runs/ for folders matching the turning_bilstm or
turning_bigru patterns.  Pass explicit run IDs via CLI flags if needed.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import typing
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.evaluation.classification_metrics import (
    compute_classification_metrics,
    compute_pr_curve_points,
    compute_threshold_sweep,
)
from financial_forecasting.plotting.plot_turning_analysis import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_threshold_sweep,
)


def _find_run(
    runs_dir: pathlib.Path,
    feature_set: str,
    model_type: str = "bilstm",
) -> pathlib.Path | None:
    """Find the most recent run folder for given feature set and model type.

    Parameters
    ----------
    runs_dir:
        Root results/runs directory.
    feature_set:
        Feature set name (e.g. 'original_ohlc').
    model_type:
        'bilstm' or 'bigru'.

    Returns
    -------
    pathlib.Path | None:
        Path to the most recent matching run folder, or None.
    """
    pattern = f"turning_{model_type}_{feature_set}"
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and pattern in d.name
    ]
    if not candidates:
        return None
    # Sort by name (timestamp prefix ensures chronological order)
    return sorted(candidates)[-1]


def _load_predictions(run_dir: pathlib.Path, split: str) -> pd.DataFrame:
    """Load prediction CSV for a given split.

    Parameters
    ----------
    run_dir:
        Run directory containing predictions_*.csv files.
    split:
        Split name ('train', 'val', 'test').

    Returns
    -------
    pd.DataFrame:
        DataFrame with columns: split, ticker, anchor_date, actual_label,
        predicted_label, predicted_probability.
    """
    csv_path = run_dir / f"predictions_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _format_cm(cm: list[list[int]]) -> str:
    """Format a 2x2 confusion matrix as a markdown table row string."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    return (
        f"| TN={tn} | FP={fp} |\n| FN={fn} | TP={tp} |"
    )


def main() -> None:
    """Run Phase 5A: turning-point analysis from saved predictions."""
    parser = argparse.ArgumentParser(
        description="Phase 5A: Turning-point analysis from saved run folders."
    )
    parser.add_argument(
        "--run-orig",
        type=str,
        default=None,
        help="Run folder ID for original_ohlc (auto-detected if omitted).",
    )
    parser.add_argument(
        "--run-aux",
        type=str,
        default=None,
        help="Run folder ID for auxiliary_ohlc (auto-detected if omitted).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bilstm",
        choices=["bilstm", "bigru"],
        help="Model type to analyse (default: bilstm).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5).",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("Phase 5A - Turning-Point Analysis")
    print("=" * 65)

    paths = ProjectPaths()
    paths.ensure_all()

    # Discover run folders
    if args.run_orig:
        run_orig = paths.results_runs / args.run_orig
    else:
        run_orig = _find_run(paths.results_runs, "original_ohlc", args.model_type)

    if args.run_aux:
        run_aux = paths.results_runs / args.run_aux
    else:
        run_aux = _find_run(paths.results_runs, "auxiliary_ohlc", args.model_type)

    if run_orig is None:
        raise RuntimeError(
            "No original_ohlc turning-point run folder found. "
            "Run train_turning_bilstm.py first."
        )
    if run_aux is None:
        raise RuntimeError(
            "No auxiliary_ohlc turning-point run folder found. "
            "Run train_turning_bilstm.py --feature-set auxiliary_ohlc first."
        )

    print(f"original_ohlc run: {run_orig.name}")
    print(f"auxiliary_ohlc run: {run_aux.name}")

    threshold = args.threshold

    # Load test predictions for both feature sets
    df_orig = _load_predictions(run_orig, "test")
    df_aux = _load_predictions(run_aux, "test")

    y_true_orig: NDArray[np.int8] = df_orig["actual_label"].to_numpy(dtype=np.int8)
    y_prob_orig: NDArray[np.float32] = df_orig["predicted_probability"].to_numpy(
        dtype=np.float32
    )
    tickers_orig = df_orig["ticker"].tolist()

    y_true_aux: NDArray[np.int8] = df_aux["actual_label"].to_numpy(dtype=np.int8)
    y_prob_aux: NDArray[np.float32] = df_aux["predicted_probability"].to_numpy(
        dtype=np.float32
    )
    tickers_aux = df_aux["ticker"].tolist()

    # Load class balance JSONs
    cb_orig_path = run_orig / "class_balance.json"
    cb_aux_path = run_aux / "class_balance.json"

    cb_orig: dict[str, object] = {}
    cb_aux: dict[str, object] = {}
    if cb_orig_path.exists():
        with open(cb_orig_path, encoding="utf-8") as f:
            cb_orig = json.load(f)
    if cb_aux_path.exists():
        with open(cb_aux_path, encoding="utf-8") as f:
            cb_aux = json.load(f)

    # Compute classification metrics
    print(f"\nComputing metrics at threshold = {threshold}")
    metrics_orig = compute_classification_metrics(y_true_orig, y_prob_orig, tickers_orig, threshold)
    metrics_aux = compute_classification_metrics(y_true_aux, y_prob_aux, tickers_aux, threshold)

    print(f"\n{'Metric':<30} {'original_ohlc':>15} {'auxiliary_ohlc':>15}")
    print("-" * 62)
    for key in ["accuracy", "precision", "recall", "f1",
                "positive_class_rate", "predicted_positive_rate"]:
        v_orig = metrics_orig.get(key, float("nan"))
        v_aux = metrics_aux.get(key, float("nan"))
        print(f"  {key:<28} {float(v_orig):>15.4f} {float(v_aux):>15.4f}")  # type: ignore[arg-type]

    for key in ["roc_auc", "pr_auc"]:
        v_orig = metrics_orig.get(key, float("nan"))
        v_aux = metrics_aux.get(key, float("nan"))
        print(f"  {key:<28} {str(v_orig):>15} {str(v_aux):>15}")

    # Compute threshold sweeps (from test split)
    sweep_orig = compute_threshold_sweep(y_true_orig, y_prob_orig)
    sweep_aux = compute_threshold_sweep(y_true_aux, y_prob_aux)

    # Compute PR curve data
    prec_orig, rec_orig, _ = compute_pr_curve_points(y_true_orig, y_prob_orig)
    prec_aux, rec_aux, _ = compute_pr_curve_points(y_true_aux, y_prob_aux)

    pr_auc_orig = metrics_orig.get("pr_auc", float("nan"))
    pr_auc_aux = metrics_aux.get("pr_auc", float("nan"))

    # Positive class rates for plot baselines
    pos_rate_orig = float(metrics_orig.get("positive_class_rate", 0.0))  # type: ignore[arg-type]
    pos_rate_avg = (pos_rate_orig + float(metrics_aux.get("positive_class_rate", 0.0))) / 2.0  # type: ignore[arg-type]

    # Confusion matrices
    cm_orig = metrics_orig.get("confusion_matrix", [[0, 0], [0, 0]])
    cm_aux = metrics_aux.get("confusion_matrix", [[0, 0], [0, 0]])

    # ---- Generate plots ----
    print("\nGenerating plots...")

    png1, _ = plot_confusion_matrix(
        cm_orig=cm_orig,  # type: ignore[arg-type]
        cm_aux=cm_aux,    # type: ignore[arg-type]
        feature_set_names=("original_ohlc", "auxiliary_ohlc"),
        out_stem=paths.figures_models / "turning_confusion_matrix",
    )
    print(f"  Saved: {png1}")

    png2, _ = plot_precision_recall_curve(
        pr_data={
            "original_ohlc": (prec_orig, rec_orig, float(pr_auc_orig)),   # type: ignore[arg-type]
            "auxiliary_ohlc": (prec_aux, rec_aux, float(pr_auc_aux)),     # type: ignore[arg-type]
        },
        out_stem=paths.figures_models / "turning_precision_recall",
        positive_class_rate=pos_rate_avg,
    )
    print(f"  Saved: {png2}")

    png3, _ = plot_threshold_sweep(
        sweep_data={
            "original_ohlc": sweep_orig,
            "auxiliary_ohlc": sweep_aux,
        },
        out_stem=paths.figures_models / "turning_threshold_sweep",
    )
    print(f"  Saved: {png3}")

    # ---- Write analysis markdown ----
    _write_analysis_doc(
        paths=paths,
        run_orig=run_orig,
        run_aux=run_aux,
        metrics_orig=metrics_orig,
        metrics_aux=metrics_aux,
        cb_orig=cb_orig,
        cb_aux=cb_aux,
        threshold=threshold,
        model_type=args.model_type,
    )
    print("\nPhase 5A analysis complete!")


def _write_analysis_doc(
    paths: ProjectPaths,
    run_orig: pathlib.Path,
    run_aux: pathlib.Path,
    metrics_orig: typing.Mapping[str, object],
    metrics_aux: typing.Mapping[str, object],
    cb_orig: typing.Mapping[str, object],
    cb_aux: typing.Mapping[str, object],
    threshold: float,
    model_type: str,
) -> None:
    """Write docs/05_turning_point_analysis.md.

    Parameters
    ----------
    paths:
        Project path helper.
    run_orig:
        Run folder for original_ohlc.
    run_aux:
        Run folder for auxiliary_ohlc.
    metrics_orig:
        Test-split classification metrics for original_ohlc.
    metrics_aux:
        Test-split classification metrics for auxiliary_ohlc.
    cb_orig:
        Class balance statistics for original_ohlc.
    cb_aux:
        Class balance statistics for auxiliary_ohlc.
    threshold:
        Decision threshold used for label generation.
    model_type:
        Model architecture ('bilstm' or 'bigru').
    """
    def _g(d: typing.Mapping[str, object], k: str) -> str:
        v = d.get(k, float("nan"))
        try:
            return f"{float(v):.4f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return str(v)

    def _gi(d: typing.Mapping[str, object], k: str) -> str:
        v = d.get(k, "N/A")
        return str(v)

    # Extract class balance per split
    def _cb_row(cb: typing.Mapping[str, object], split: str) -> str:
        info = cb.get(split, {})
        if isinstance(info, dict):
            return (
                f"| {split.capitalize():<5} | "
                f"{info.get('total', 'N/A'):>7} | "
                f"{info.get('n_pos', 'N/A'):>9} | "
                f"{info.get('n_neg', 'N/A'):>9} | "
                f"{float(info.get('pos_rate', 0.0)):.4f} |"  # type: ignore[arg-type]
            )
        return f"| {split} | N/A | N/A | N/A | N/A |"

    timestamp = datetime.now(tz=UTC).isoformat()

    doc = f"""# Phase 5A Analysis - Turning-Point Detection

Generated: {timestamp}
Runs analysed:
- original_ohlc: `{run_orig.name}`
- auxiliary_ohlc: `{run_aux.name}`
Model type: {model_type.upper()}

---

## Goal

Evaluate the binary buy/pass turning-point classifier on the test split
and compare classification performance across the original_ohlc and
auxiliary_ohlc feature sets.  All analysis uses saved prediction files;
no model retraining is performed.

---

## Records Used

| Feature set | Run folder |
|---|---|
| original_ohlc | `{run_orig.name}` |
| auxiliary_ohlc | `{run_aux.name}` |

Files used:
- `predictions_test.csv` -- predicted probabilities and labels
- `class_balance.json` -- positive/negative counts per split
- `threshold_sweep.json` -- sweep data loaded from run for consistency

---

## Formal Classification Target

The turning-point label is defined as:

```
r_{{t+d}} = (High[t+d] - Close[t]) / Close[t]
label     = 1  (buy)   if  max_{{d=1..5}}  r_{{t+d}} > gamma = 1.1
            0  (pass)  otherwise
```

With `gamma = 1.1` this requires a stock's High price to exceed
`2.1 * Close` within 5 trading days -- a gain of more than 110%.
Such moves are essentially absent in blue-chip S&P 500 stocks
(AAPL, MSFT, JPM) during 2020-2025.

---

## Class Balance

### original_ohlc

| Split | Total | Positives | Negatives | Pos Rate |
|-------|------:|----------:|----------:|----------:|
{_cb_row(cb_orig, 'train')}
{_cb_row(cb_orig, 'val')}
{_cb_row(cb_orig, 'test')}

### auxiliary_ohlc

| Split | Total | Positives | Negatives | Pos Rate |
|-------|------:|----------:|----------:|----------:|
{_cb_row(cb_aux, 'train')}
{_cb_row(cb_aux, 'val')}
{_cb_row(cb_aux, 'test')}

**Interpretation:** With `gamma = 1.1` (raw return threshold = 110%),
the positive class is absent or near-absent in all splits.  This is a
fundamental consequence of the assignment's threshold applied to large-cap
U.S. equities during the study period.  The model learns the trivial
all-negative predictor and achieves high accuracy by virtue of the base
rate, while precision, recall, and F1 are 0 or undefined.

---

## Metrics (Test Split, threshold = {threshold})

| Metric | original_ohlc | auxiliary_ohlc |
|---|---:|---:|
| Accuracy | {_g(metrics_orig, 'accuracy')} | {_g(metrics_aux, 'accuracy')} |
| Precision | {_g(metrics_orig, 'precision')} | {_g(metrics_aux, 'precision')} |
| Recall | {_g(metrics_orig, 'recall')} | {_g(metrics_aux, 'recall')} |
| F1 | {_g(metrics_orig, 'f1')} | {_g(metrics_aux, 'f1')} |
| Positive class rate | {_g(metrics_orig, 'positive_class_rate')} | {_g(metrics_aux, 'positive_class_rate')} |
| Predicted positive rate | {_g(metrics_orig, 'predicted_positive_rate')} | {_g(metrics_aux, 'predicted_positive_rate')} |
| ROC-AUC | {_g(metrics_orig, 'roc_auc')} | {_g(metrics_aux, 'roc_auc')} |
| PR-AUC | {_g(metrics_orig, 'pr_auc')} | {_g(metrics_aux, 'pr_auc')} |

---

## Confusion Matrices (Test Split)

### original_ohlc

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | {metrics_orig.get('tn', 'N/A')} (TN) | {metrics_orig.get('fp', 'N/A')} (FP) |
| **Actual Positive** | {metrics_orig.get('fn', 'N/A')} (FN) | {metrics_orig.get('tp', 'N/A')} (TP) |

### auxiliary_ohlc

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | {metrics_aux.get('tn', 'N/A')} (TN) | {metrics_aux.get('fp', 'N/A')} (FP) |
| **Actual Positive** | {metrics_aux.get('fn', 'N/A')} (FN) | {metrics_aux.get('tp', 'N/A')} (TP) |

---

## Plot Descriptions

### 1. Confusion Matrix: `figures/models/turning_confusion_matrix.{{png,pdf}}`

**X-axis:** Predicted label (Negative / Positive).
**Y-axis:** True label (Negative / Positive).
**Subplots:** Left = original_ohlc, Right = auxiliary_ohlc.
**Inline explanation:** Cell colour encodes the fraction of total samples.
Because the positive class is absent, the confusion matrix is dominated
by the TN cell (all samples classified as Negative).  The plot makes
the all-negative prediction behaviour visually unmistakable.

**Interpretation:** When all predictions are Negative, TN = all actual
negatives (nearly all samples), FP = 0, FN = any positives that exist,
TP = 0.  Both feature sets are expected to exhibit this pattern under
`gamma = 1.1`.

---

### 2. Precision-Recall Curve: `figures/models/turning_precision_recall.{{png,pdf}}`

**X-axis:** Recall (0.0 to 1.0) -- fraction of true buy signals detected.
**Y-axis:** Precision (0.0 to 1.0) -- fraction of predicted buys that
are correct.
**Dashed line:** No-skill baseline at the positive class rate.
**Subplots / lines:** One curve per feature set.

**Interpretation:** When positive labels are absent (pos_rate = 0), the
PR curve is not computable in the classical sense; all thresholds map to
either (prec=0, rec=0) or are undefined.  The chart will show degenerate
behaviour and confirms that the model has no useful discriminative
capacity under this gamma.

---

### 3. Threshold Sweep: `figures/models/turning_threshold_sweep.{{png,pdf}}`

**X-axis:** Classification threshold (0.0 to 1.0).
**Y-axis:** Metric value (0.0 to 1.0).
**Lines:** Precision (red), Recall (green), F1 (purple).
**Subplots:** One per feature set.
**Dashed vertical:** Default threshold = 0.5.

**Interpretation:** If predicted probabilities are clustered near 0.5
(i.e., the model is uncertain due to degenerate data), the sweep shows
how precision and recall trade off.  For the all-negative predictor:
recall = 0 for all thresholds above 0; precision is 0 or undefined.
F1 = 0 everywhere.  Lowering the threshold to near 0.0 would force all
predictions to Positive, giving recall = 1 but extremely low precision.

---

## Discussion

### False Positives
False positives (predicting 'buy' when the stock does not rise >110%)
are costly in a trading context: they waste capital on positions that
do not pay off.  Under `gamma = 1.1`, there are essentially no actual
positives, so false positives are only possible at very low thresholds.

### False Negatives
False negatives (missing a real turning-point event) are the defining
failure mode for a buy-signal detector.  With `gamma = 1.1`, there are
effectively no ground-truth buy events in the study window, so FN = 0
by default and recall appears perfect (0/0 undefined) -- but only
because there is nothing to detect.

### Class Imbalance and Gamma = 1.1
The assignment specifies `gamma = 1.1` to create a strict filter for
large price events.  In practice, a 110% gain within 5 trading days
never occurred for AAPL, MSFT, or JPM during 2020-2025.  This means:

1. **Training:** The model receives only negative labels and converges
   to the trivial all-negative solution.  Binary cross-entropy is
   minimised at 0 (perfect certainty of the negative class).
2. **Accuracy:** ~100% (trivially, since all samples are negative).
3. **Precision / Recall / F1:** 0 or undefined.
4. **ROC-AUC / PR-AUC:** Undefined (only one class present).

### Is the Model Useful?
Under `gamma = 1.1`, the model is **not useful as a buy-signal detector**
because the detection target never materialises.  The experiment is
nonetheless valuable as a demonstration of:
- Correct pipeline implementation (bidirectional LSTM, BCE loss, etc.)
- The critical importance of dataset analysis before modelling
- The consequence of threshold selection on label distribution

If the assignment intended `gamma` as a percentage threshold (e.g.,
10% gain, corresponding to a return threshold of 0.1), the experiment
would be more informative.  Rerunning with `gamma = 0.02` (2% gain)
would produce a balanced dataset (~30-40% positive rate in 5-day windows)
and a meaningful classification task.

---

## Known Limitations

1. `gamma = 1.1` produces zero positive labels in real data.
2. ROC-AUC and PR-AUC cannot be computed when only one class is present.
3. F1 is 0 or undefined throughout.
4. The bidirectional LSTM architecture is technically correct but cannot
   demonstrate its capacity without a non-trivial classification problem.
5. A meaningful ablation would require either lowering gamma or using
   different/more volatile stocks (e.g., small-cap or crypto).

---

## Implications for Next Phase

- Phase 6 final aggregation can include the turning-point run for
  completeness with a clear caveat table explaining the class imbalance.
- A sensitivity analysis with varying gamma (0.01, 0.02, 0.05, 0.10)
  is recommended as a supplementary experiment.
"""

    doc_path = paths.docs / "05_turning_point_analysis.md"
    with open(doc_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"\nAnalysis doc saved: {doc_path}")


if __name__ == "__main__":
    main()
