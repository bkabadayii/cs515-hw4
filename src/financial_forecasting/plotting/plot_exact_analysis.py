"""Phase 3E - Publication-ready plots for exact-return forecasting workbench.

Reads results/aggregate/exact_forecasting_matrix.csv and generates four plots:

    1. exact_baseline_comparison_rmse.{png,pdf}
       Grouped bar chart: predictor vs test RMSE per horizon.

    2. exact_prediction_std_collapse.{png,pdf}
       Target vs prediction std per horizon for each neural model.

    3. exact_model_horizon_heatmap.{png,pdf}
       Heatmap: predictor x horizon -> test MSE.

    4. exact_feature_set_comparison.{png,pdf}
       Dumbbell plot: original vs auxiliary features per horizon.

Usage:
    uv run python src/financial_forecasting/plotting/plot_exact_analysis.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.plotting.style import apply_style

matplotlib.use("Agg")


HORIZONS = [1, 2, 3, 4, 5]
HORIZON_LABELS = [f"d={h}" for h in HORIZONS]


def _save_fig(fig: plt.Figure, base_path: pathlib.Path) -> None:
    """Save figure in both PNG and PDF formats."""
    fig.savefig(str(base_path) + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(str(base_path) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base_path}.png / .pdf")


def plot_baseline_comparison(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Grouped bar chart: test RMSE per horizon, grouped by predictor.

    X-axis: prediction horizon d=1..5
    Y-axis: test RMSE
    Bars: one group per predictor (baselines + neural models)
    """
    apply_style()

    rmse_cols = [f"test_rmse_h{h}" for h in HORIZONS]
    # Use original_ohlc rows only for the baseline comparison
    df_orig = df[df["feature_set"] == "original_ohlc"].copy()

    predictors = df_orig["predictor"].tolist()
    n_pred = len(predictors)
    x = np.arange(len(HORIZONS))
    width = 0.8 / max(n_pred, 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    cmap = plt.get_cmap("tab10")
    for i, (_, row) in enumerate(df_orig.iterrows()):
        vals = [row[c] for c in rmse_cols]
        offset = (i - n_pred / 2.0 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=row["predictor"],
            color=cmap(i),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Test RMSE")
    ax.set_title(
        "Exact-Return Forecasting: Test RMSE by Predictor and Horizon\n"
        "(feature set: original_ohlc)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    _save_fig(fig, out_dir / "exact_baseline_comparison_rmse")


def plot_prediction_std_collapse(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Actual vs predicted std per horizon for all predictors.

    X-axis: prediction horizon d=1..5
    Y-axis: standard deviation of returns
    Lines: actual target std and predicted std per predictor
    """
    apply_style()

    # We need per-horizon std from metrics_test.json.
    # These are already stored in the matrix as: target_std = test_target_std,
    # prediction_std = test_pred_std (only overall, not per-horizon).
    # For this plot we use the overall values across horizons from the aggregate CSV.
    # If per-horizon std columns are present, use them; otherwise display overall.

    # Load individual run metrics for per-horizon std if available
    paths = ProjectPaths()
    runs_dir = paths.results_runs

    # Collect per-horizon target and pred std from individual run metrics
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        run_id = str(row["run_id"])
        run_dir = runs_dir / run_id
        metrics_path = run_dir / "metrics_test.json"
        if not metrics_path.exists():
            continue
        import json
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)

        target_std = m.get("target_std_by_horizon", {})
        pred_std = m.get("prediction_std_by_horizon", {})

        for h in HORIZONS:
            key = f"horizon_{h}"
            records.append(
                {
                    "predictor": row["predictor"],
                    "feature_set": row["feature_set"],
                    "horizon": h,
                    "target_std": target_std.get(key, float("nan")),
                    "pred_std": pred_std.get(key, float("nan")),
                }
            )

    if not records:
        print("  No per-horizon std data available, skipping std collapse plot.")
        return

    df_std = pd.DataFrame(records)
    df_orig = df_std[df_std["feature_set"] == "original_ohlc"]

    predictors = df_orig["predictor"].unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot target std once (shared across predictors)
    # Use first predictor's target std as reference
    first_pred = predictors[0]
    df_first = df_orig[df_orig["predictor"] == first_pred].sort_values("horizon")
    ax.plot(
        df_first["horizon"],
        df_first["target_std"],
        "k--",
        linewidth=2,
        label="Actual Target Std",
        marker="D",
        markersize=5,
    )

    cmap = plt.get_cmap("tab10")
    for i, pred in enumerate(predictors):
        df_p = df_orig[df_orig["predictor"] == pred].sort_values("horizon")
        ax.plot(
            df_p["horizon"],
            df_p["pred_std"],
            color=cmap(i),
            linewidth=1.5,
            label=f"Pred Std: {pred}",
            marker="o",
            markersize=4,
        )

    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("Standard Deviation of Returns")
    ax.set_title(
        "Prediction Spread: Actual Target Std vs Predicted Std by Horizon\n"
        "(feature set: original_ohlc)"
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    _save_fig(fig, out_dir / "exact_prediction_std_collapse")


def plot_model_horizon_heatmap(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Heatmap: predictor x horizon -> test MSE.

    X-axis: prediction horizon d=1..5
    Y-axis: predictor name (model + feature set)
    Cell: test MSE (lower is better)
    """
    apply_style()

    mse_cols = [f"test_mse_h{h}" for h in HORIZONS]

    # Create row labels combining predictor and feature set
    df = df.copy()
    df["label"] = df["predictor"] + " [" + df["feature_set"] + "]"

    pivot = df.set_index("label")[mse_cols].rename(
        columns={f"test_mse_h{h}": f"d={h}" for h in HORIZONS}
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.65)))

    mat = pivot.values.astype(float)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"d={h}" for h in HORIZONS])
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_xlabel("Prediction Horizon")
    ax.set_title("Exact-Return Test MSE: Model × Horizon Heatmap\n(lower = better)")

    # Annotate cells
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            if not np.isnan(val):
                ax.text(
                    c, r, f"{val:.5f}",
                    ha="center", va="center", fontsize=7,
                    color="black",
                )

    plt.colorbar(im, ax=ax, label="Test MSE")
    fig.tight_layout()

    _save_fig(fig, out_dir / "exact_model_horizon_heatmap")


def plot_feature_set_comparison(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Dumbbell plot: original vs auxiliary features per horizon.

    X-axis: test RMSE
    Y-axis: predictor name
    Points: original_ohlc (circle) and auxiliary_ohlc (triangle) per horizon
    Each subplot = one horizon
    """
    apply_style()

    df_orig = df[df["feature_set"] == "original_ohlc"].set_index("predictor")
    df_aux = df[df["feature_set"] == "auxiliary_ohlc"].set_index("predictor")

    # Find predictors present in both feature sets (neural models only)
    shared = list(set(df_orig.index) & set(df_aux.index))
    if not shared:
        print("  No shared predictors between feature sets. Skipping dumbbell plot.")
        return

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, max(3, len(shared) * 0.7 + 2)), sharey=True)
    if len(HORIZONS) == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        col = f"test_rmse_h{h}"

        for i, pred in enumerate(shared):
            orig_val = df_orig.loc[pred, col] if pred in df_orig.index else float("nan")
            aux_val = df_aux.loc[pred, col] if pred in df_aux.index else float("nan")

            if not (np.isnan(orig_val) or np.isnan(aux_val)):
                # Draw line segment
                ax.plot([orig_val, aux_val], [i, i], color="grey", linewidth=1.2, zorder=1)

            ax.scatter(
                [orig_val], [i],
                color="#1f77b4", marker="o", s=55, zorder=2,
                label="original" if i == 0 else "_nolegend_",
            )
            ax.scatter(
                [aux_val], [i],
                color="#ff7f0e", marker="^", s=55, zorder=2,
                label="auxiliary" if i == 0 else "_nolegend_",
            )

        ax.set_title(f"d={h}", fontsize=10)
        ax.set_xlabel("Test RMSE", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        if h == HORIZONS[0]:
            ax.set_yticks(range(len(shared)))
            ax.set_yticklabels(shared, fontsize=8)
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "Exact-Return Test RMSE: Original OHLC vs Auxiliary OHLC Features\n"
        "(circle = original, triangle = auxiliary)",
        fontsize=11,
    )
    fig.tight_layout()

    _save_fig(fig, out_dir / "exact_feature_set_comparison")


def main() -> None:
    """Generate all Phase 3E analysis plots."""
    print("=" * 60)
    print("Phase 3E - Exact-Return Analysis Plots")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()

    matrix_path = paths.results_aggregate / "exact_forecasting_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Aggregation matrix not found: {matrix_path}\n"
            "Please run aggregate_exact.py first."
        )

    df = pd.read_csv(matrix_path)
    print(f"Loaded {len(df)} rows from: {matrix_path}")

    out_dir = paths.figures_models
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Plot 1: Baseline comparison RMSE bar chart...")
    plot_baseline_comparison(df, out_dir)

    print("Generating Plot 2: Prediction std collapse...")
    plot_prediction_std_collapse(df, out_dir)

    print("Generating Plot 3: Model × horizon heatmap...")
    plot_model_horizon_heatmap(df, out_dir)

    print("Generating Plot 4: Feature set comparison dumbbell...")
    plot_feature_set_comparison(df, out_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
