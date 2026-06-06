"""Phase 3E - Aggregate exact-return experiment results.

Reads all experiment run folders under results/runs/ and builds a unified
comparison CSV at results/aggregate/exact_forecasting_matrix.csv.

Each run must have:
    config.json          (includes model_type, feature_set, target_type)
    metrics_test.json    (includes overall_mse, overall_rmse, etc.)
    history.csv          (optional: for stability metrics)

Usage:
    uv run python src/financial_forecasting/analysis/aggregate_exact.py
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

from financial_forecasting.config.base import ProjectPaths


def _load_json(path: pathlib.Path) -> dict[str, object]:
    """Load a JSON file and return as a dict."""
    with open(path, encoding="utf-8") as f:
        result: dict[str, object] = json.load(f)
    return result


def _extract_row(run_dir: pathlib.Path) -> dict[str, object] | None:
    """Extract a flat summary row from one run folder.

    Parameters
    ----------
    run_dir:
        Path to the run folder.

    Returns
    -------
    dict or None:
        Flat summary row, or None if required files are missing.
    """
    config_path = run_dir / "config.json"
    metrics_test_path = run_dir / "metrics_test.json"
    metrics_val_path = run_dir / "metrics_val.json"
    history_path = run_dir / "history.csv"

    if not config_path.exists() or not metrics_test_path.exists():
        return None

    config = _load_json(config_path)
    metrics_test = _load_json(metrics_test_path)

    # Identify model / predictor name
    model_type = str(config.get("model_type", config.get("baseline_name", "unknown")))
    feature_set = str(config.get("feature_set", "original_ohlc"))
    target_type = str(config.get("target_type", "exact"))
    baseline_tag = str(config.get("baseline_tag", ""))

    # Build predictor label for the matrix
    if baseline_tag:
        predictor_label = f"Baseline: {model_type}"
    else:
        predictor_label = model_type

    # --- Test metrics ---
    row: dict[str, object] = {
        "run_id": run_dir.name,
        "predictor": predictor_label,
        "feature_set": feature_set,
        "target_type": target_type,
        "test_mse": metrics_test.get("overall_mse", float("nan")),
        "test_rmse": metrics_test.get("overall_rmse", float("nan")),
        "test_mae": metrics_test.get("overall_mae", float("nan")),
        "test_r2": metrics_test.get("overall_r2", float("nan")),
        "test_pearson": metrics_test.get("overall_pearson_corr", float("nan")),
        "test_dir_acc": metrics_test.get("overall_directional_accuracy", float("nan")),
        "test_target_std": metrics_test.get("target_std_overall", float("nan")),
        "test_pred_std": metrics_test.get("prediction_std_overall", float("nan")),
    }

    # Per-horizon test MSE
    ph_mse = metrics_test.get("per_horizon_mse", {})
    if isinstance(ph_mse, dict):
        for h in range(1, 6):
            key = f"horizon_{h}"
            row[f"test_mse_h{h}"] = ph_mse.get(key, float("nan"))

    # Per-horizon test RMSE
    ph_rmse = metrics_test.get("per_horizon_rmse", {})
    if isinstance(ph_rmse, dict):
        for h in range(1, 6):
            key = f"horizon_{h}"
            row[f"test_rmse_h{h}"] = ph_rmse.get(key, float("nan"))

    # Val metrics (for stability)
    if metrics_val_path.exists():
        metrics_val = _load_json(metrics_val_path)
        row["val_mse"] = metrics_val.get("overall_mse", float("nan"))
        row["val_rmse"] = metrics_val.get("overall_rmse", float("nan"))

    # History-based stability metrics (neural models only)
    if history_path.exists():
        try:
            df_hist = pd.read_csv(history_path)
            row["best_epoch"] = int(df_hist.loc[df_hist["val_loss"].idxmin(), "epoch"])
            row["total_epochs"] = len(df_hist)
            row["best_val_mse"] = float(df_hist["val_loss"].min())
            # Last-k validation variance (k = 10)
            k = 10
            last_k = df_hist["val_loss"].tail(k)
            row["last_k_val_variance"] = float(last_k.var()) if len(last_k) > 1 else float("nan")
            # Mean absolute epoch-to-epoch val change
            row["mean_abs_val_change"] = float(df_hist["val_loss"].diff().abs().mean())
        except Exception:
            row["best_epoch"] = float("nan")
            row["total_epochs"] = float("nan")
            row["best_val_mse"] = float("nan")
            row["last_k_val_variance"] = float("nan")
            row["mean_abs_val_change"] = float("nan")
    else:
        row["best_epoch"] = float("nan")
        row["total_epochs"] = float("nan")
        row["best_val_mse"] = float("nan")
        row["last_k_val_variance"] = float("nan")
        row["mean_abs_val_change"] = float("nan")

    return row


def main() -> None:
    """Aggregate all exact-return run folders into a comparison matrix."""
    print("=" * 60)
    print("Phase 3E - Aggregate Exact-Return Experiment Matrix")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    runs_dir = paths.results_runs
    rows: list[dict[str, object]] = []

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    print(f"\nScanning {len(run_dirs)} run folders in: {runs_dir}")

    for run_dir in run_dirs:
        row = _extract_row(run_dir)
        if row is None:
            print(f"  Skipping (missing files): {run_dir.name}")
            continue
        # Filter to exact-return experiments only
        if str(row.get("target_type", "")) != "exact":
            continue
        rows.append(row)
        print(f"  Loaded: {run_dir.name}")

    if not rows:
        print("\nNo exact-return runs found. Run training scripts first.")
        return

    df = pd.DataFrame(rows)

    # Sort by: feature_set, predictor name
    df = df.sort_values(["feature_set", "predictor"]).reset_index(drop=True)

    # Save matrix CSV
    matrix_path = paths.results_aggregate / "exact_forecasting_matrix.csv"
    df.to_csv(matrix_path, index=False)
    print(f"\nAggregation matrix saved to: {matrix_path}")
    print(f"Total rows: {len(df)}")

    # Print summary table
    print("\n=== Exact-Return Comparison Matrix ===")
    summary_cols = ["predictor", "feature_set", "test_mse", "test_rmse", "test_mae", "test_r2", "test_dir_acc"]
    print(df[summary_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("Aggregation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
