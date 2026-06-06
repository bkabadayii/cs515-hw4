"""Phase 4F - Compare exact-return vs rolling-average experiments.

Reads both aggregation matrices (exact and rolling) and produces:
1. exact_vs_rolling_matrix.csv       - combined comparison table
2. rolling_model_horizon_heatmap     - rolling experiments heatmap
3. exact_vs_rolling_stability        - validation loss curves comparison
4. exact_vs_rolling_dumbbell_mse     - test MSE dumbbell plot per horizon
5. exact_vs_rolling_feature_matrix   - 2x2 heatmap over target x feature-set

Usage:
    uv run python src/financial_forecasting/analysis/compare_exact_vs_rolling.py
"""

from __future__ import annotations

import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from financial_forecasting.config.base import ProjectPaths
from financial_forecasting.plotting.style import apply_style, save_figure

# ------------------------------------------------------------------ #
#  Helper utilities                                                    #
# ------------------------------------------------------------------ #

def _load_matrix(path: pathlib.Path) -> pd.DataFrame:
    """Load a comparison matrix CSV, raising if it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Matrix CSV not found: {path}\n"
            "Run the relevant aggregation script first."
        )
    return pd.read_csv(path)


def _load_history(run_dir: pathlib.Path) -> pd.DataFrame | None:
    """Load history.csv from a run folder, returning None if missing."""
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return None
    try:
        return pd.read_csv(history_path)
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Plot 1 - Validation loss stability curves                          #
# ------------------------------------------------------------------ #

def plot_exact_vs_rolling_stability(
    runs_dir: pathlib.Path,
    df_exact: pd.DataFrame,
    df_rolling: pd.DataFrame,
    save_stem: pathlib.Path,
) -> None:
    """Plot validation loss curves for exact vs rolling neural models.

    X-axis: epoch
    Y-axis: validation MSE
    Lines: one per (target_type, model, feature_set) combination
    Purpose: inspect whether rolling targets create smoother training curves.

    Parameters
    ----------
    runs_dir:
        Path to results/runs/.
    df_exact:
        Exact-return aggregation matrix.
    df_rolling:
        Rolling-average aggregation matrix.
    save_stem:
        Output file stem without extension.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    feature_sets = ["original_ohlc", "auxiliary_ohlc"]
    target_colors = {"exact": "#1F77B4", "rolling": "#D62728"}
    model_styles = {"StockLSTM": "-", "StockGRU": "--"}
    fs_labels = {"original_ohlc": "Original OHLC", "auxiliary_ohlc": "Auxiliary OHLC"}

    for ax_idx, fs in enumerate(feature_sets):
        ax = axes[ax_idx]
        ax.set_title(f"{fs_labels[fs]}", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MSE")
        ax.grid(True, linestyle=":", alpha=0.5)

        for df, target_type in [(df_exact, "exact"), (df_rolling, "rolling")]:
            subset = df[
                (df["feature_set"] == fs)
                & (df["predictor"].isin(["StockLSTM", "StockGRU"]))
            ]
            for _, row in subset.iterrows():
                run_id = str(row["run_id"])
                model = str(row["predictor"])
                hist = _load_history(runs_dir / run_id)
                if hist is None:
                    continue
                label = f"{model} ({target_type})"
                ax.plot(
                    hist["epoch"],
                    hist["val_loss"],
                    color=target_colors[target_type],
                    linestyle=model_styles.get(model, "-"),
                    linewidth=1.5,
                    alpha=0.85,
                    label=label,
                )

        ax.legend(fontsize=8, loc="upper right")

    # Legend patches for colors
    legend_patches = [
        mpatches.Patch(color=target_colors["exact"], label="Exact targets"),
        mpatches.Patch(color=target_colors["rolling"], label="Rolling targets"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Validation Loss: Exact vs Rolling Targets by Feature Set",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, str(save_stem))
    print(f"Stability plot saved to {save_stem}.{{png,pdf}}")


# ------------------------------------------------------------------ #
#  Plot 2 - Test MSE dumbbell per horizon                             #
# ------------------------------------------------------------------ #

def plot_exact_vs_rolling_dumbbell(
    df_exact: pd.DataFrame,
    df_rolling: pd.DataFrame,
    save_stem: pathlib.Path,
) -> None:
    """Dumbbell plot comparing exact vs rolling test MSE across horizons.

    X-axis: test MSE
    Y-axis: horizon d=1..5 (one row per horizon)
    Points: exact (circle) and rolling (diamond), connected by a horizontal line
    Grouping: one panel per (model, feature_set) combination
    Purpose: show per-horizon improvement or degradation when switching to rolling targets.

    Parameters
    ----------
    df_exact:
        Exact-return aggregation matrix.
    df_rolling:
        Rolling-average aggregation matrix.
    save_stem:
        Output file stem without extension.
    """
    apply_style()

    models = ["StockLSTM", "StockGRU"]
    feature_sets = ["original_ohlc", "auxiliary_ohlc"]
    horizons = [1, 2, 3, 4, 5]
    fs_labels = {"original_ohlc": "Original OHLC", "auxiliary_ohlc": "Auxiliary OHLC"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    panel_idx = 0
    for model in models:
        for fs in feature_sets:
            ax = axes_flat[panel_idx]
            panel_idx += 1

            # Get exact row
            exact_row = df_exact[
                (df_exact["predictor"] == model) & (df_exact["feature_set"] == fs)
            ]
            rolling_row = df_rolling[
                (df_rolling["predictor"] == model) & (df_rolling["feature_set"] == fs)
            ]

            if exact_row.empty or rolling_row.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{model} | {fs_labels[fs]}", fontsize=10)
                continue

            for h in horizons:
                col = f"test_mse_h{h}"
                exact_val = float(exact_row[col].values[0]) if col in exact_row.columns else float("nan")
                rolling_val = float(rolling_row[col].values[0]) if col in rolling_row.columns else float("nan")

                y_pos = h
                if not (np.isnan(exact_val) or np.isnan(rolling_val)):
                    ax.plot(
                        [exact_val, rolling_val],
                        [y_pos, y_pos],
                        color="#AAAAAA",
                        linewidth=1.0,
                        zorder=1,
                    )
                ax.scatter(
                    [exact_val],
                    [y_pos],
                    color="#1F77B4",
                    marker="o",
                    s=60,
                    zorder=3,
                    label="Exact" if h == 1 else "",
                )
                ax.scatter(
                    [rolling_val],
                    [y_pos],
                    color="#D62728",
                    marker="D",
                    s=50,
                    zorder=3,
                    label="Rolling" if h == 1 else "",
                )

            ax.set_yticks(horizons)
            ax.set_yticklabels([f"d={h}" for h in horizons])
            ax.set_xlabel("Test MSE")
            ax.set_title(f"{model} | {fs_labels[fs]}", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5, axis="x")
            if panel_idx == 2:
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Exact vs Rolling Test MSE per Horizon (Dumbbell Plot)",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, str(save_stem))
    print(f"Dumbbell plot saved to {save_stem}.{{png,pdf}}")


# ------------------------------------------------------------------ #
#  Plot 3 - Rolling experiment horizon heatmap                        #
# ------------------------------------------------------------------ #

def plot_rolling_horizon_heatmap(
    df_rolling: pd.DataFrame,
    save_stem: pathlib.Path,
) -> None:
    """Heatmap of rolling-target test MSE by experiment and horizon.

    X-axis: horizon d=1..5
    Y-axis: experiment (predictor x feature-set combination)
    Cell value: test MSE
    Purpose: compact rolling-only comparison across models and feature sets.

    Parameters
    ----------
    df_rolling:
        Rolling-average aggregation matrix.
    save_stem:
        Output file stem without extension.
    """
    apply_style()

    horizon_cols = [f"test_mse_h{h}" for h in range(1, 6)]
    horizon_labels = [f"d={h}" for h in range(1, 6)]

    # Build label column
    df = df_rolling.copy()
    df["label"] = df["predictor"] + "\n[" + df["feature_set"].str.replace("_ohlc", "").str.replace("_", " ").str.title() + "]"

    # Sort: baselines first, then neural models
    df["_is_baseline"] = df["predictor"].str.startswith("Baseline")
    df = df.sort_values(["_is_baseline", "predictor", "feature_set"], ascending=[False, True, True])

    matrix = df[horizon_cols].to_numpy(dtype=float)
    row_labels = df["label"].tolist()

    fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(row_labels) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(horizon_cols)))
    ax.set_xticklabels(horizon_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Forecast Horizon")
    ax.set_title("Rolling-Average Experiment: Test MSE by Horizon", fontsize=12)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(horizon_cols)):
            val = matrix[i, j]
            text_color = "white" if val > np.nanpercentile(matrix, 70) else "black"
            ax.text(
                j, i, f"{val:.5f}",
                ha="center", va="center",
                fontsize=7.5,
                color=text_color,
            )

    plt.colorbar(im, ax=ax, label="Test MSE", pad=0.01)
    fig.tight_layout()
    save_figure(fig, str(save_stem))
    print(f"Rolling heatmap saved to {save_stem}.{{png,pdf}}")


# ------------------------------------------------------------------ #
#  Plot 4 - Exact/rolling x original/auxiliary MSE matrix heatmap    #
# ------------------------------------------------------------------ #

def plot_exact_vs_rolling_feature_matrix(
    df_exact: pd.DataFrame,
    df_rolling: pd.DataFrame,
    save_stem: pathlib.Path,
) -> None:
    """2x2 heatmap: rows = target type, columns = feature set, cells = test MSE.

    Separate panels for LSTM and GRU.
    X-axis: feature set (original, auxiliary)
    Y-axis: target type (exact, rolling)
    Cell value: overall test MSE
    Purpose: show joint impact of target type and feature set.

    Parameters
    ----------
    df_exact:
        Exact-return aggregation matrix.
    df_rolling:
        Rolling-average aggregation matrix.
    save_stem:
        Output file stem without extension.
    """
    apply_style()

    models = ["StockLSTM", "StockGRU"]
    feature_sets = ["original_ohlc", "auxiliary_ohlc"]
    target_types = ["exact", "rolling"]
    fs_labels = ["Original OHLC", "Auxiliary OHLC"]
    tt_labels = ["Exact", "Rolling"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    combined = pd.concat([
        df_exact.assign(target_type="exact"),
        df_rolling.assign(target_type="rolling"),
    ], ignore_index=True)

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]

        grid = np.full((len(target_types), len(feature_sets)), float("nan"))
        for i, tt in enumerate(target_types):
            for j, fs in enumerate(feature_sets):
                row = combined[
                    (combined["predictor"] == model)
                    & (combined["feature_set"] == fs)
                    & (combined["target_type"] == tt)
                ]
                if not row.empty:
                    grid[i, j] = float(row["test_mse"].values[0])

        vmin = np.nanmin(grid) * 0.98
        vmax = np.nanmax(grid) * 1.02
        im = ax.imshow(grid, cmap="coolwarm_r", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(feature_sets)))
        ax.set_xticklabels(fs_labels, fontsize=9)
        ax.set_yticks(range(len(target_types)))
        ax.set_yticklabels(tt_labels, fontsize=9)
        ax.set_title(f"{model}: Test MSE (Target x Feature Set)", fontsize=10)
        ax.set_xlabel("Feature Set")
        ax.set_ylabel("Target Type")

        for i in range(len(target_types)):
            for j in range(len(feature_sets)):
                val = grid[i, j]
                if not np.isnan(val):
                    text_color = "white" if val < vmin + (vmax - vmin) * 0.4 else "black"
                    ax.text(
                        j, i, f"{val:.5f}",
                        ha="center", va="center",
                        fontsize=9,
                        color=text_color,
                        fontweight="bold",
                    )

        plt.colorbar(im, ax=ax, label="Test MSE", shrink=0.85)

    fig.suptitle(
        "Exact vs Rolling x Original vs Auxiliary Feature Set Matrix (Test MSE)",
        fontsize=12,
    )
    fig.tight_layout()
    save_figure(fig, str(save_stem))
    print(f"Feature matrix heatmap saved to {save_stem}.{{png,pdf}}")


# ------------------------------------------------------------------ #
#  Stability metric table                                             #
# ------------------------------------------------------------------ #

def build_stability_table(
    df_exact: pd.DataFrame,
    df_rolling: pd.DataFrame,
) -> pd.DataFrame:
    """Build a stability metric comparison table for neural models.

    Parameters
    ----------
    df_exact:
        Exact-return aggregation matrix with stability columns.
    df_rolling:
        Rolling-average aggregation matrix with stability columns.

    Returns
    -------
    pd.DataFrame:
        Combined stability table with columns for both target types.
    """
    stability_cols = [
        "predictor", "feature_set",
        "best_val_mse", "test_mse",
        "last_k_val_variance", "mean_abs_val_change",
        "best_epoch", "total_epochs",
    ]
    models = ["StockLSTM", "StockGRU"]

    rows: list[dict[str, object]] = []
    for df, target in [(df_exact, "exact"), (df_rolling, "rolling")]:
        subset = df[df["predictor"].isin(models)]
        for _, row in subset.iterrows():
            entry: dict[str, object] = {"target_type": target}
            for col in stability_cols:
                entry[col] = row.get(col, float("nan"))
            rows.append(entry)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["predictor", "feature_set", "target_type"]
    ).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Build combined CSV                                                  #
# ------------------------------------------------------------------ #

def build_exact_vs_rolling_matrix(
    df_exact: pd.DataFrame,
    df_rolling: pd.DataFrame,
    out_path: pathlib.Path,
) -> pd.DataFrame:
    """Concatenate exact and rolling matrices into one CSV.

    Parameters
    ----------
    df_exact:
        Exact-return aggregation matrix.
    df_rolling:
        Rolling-average aggregation matrix.
    out_path:
        Output CSV path.

    Returns
    -------
    pd.DataFrame:
        Combined dataframe saved to out_path.
    """
    df_exact_tagged = df_exact.copy()
    df_exact_tagged["target_type"] = "exact"

    df_rolling_tagged = df_rolling.copy()
    df_rolling_tagged["target_type"] = "rolling"

    combined = pd.concat([df_exact_tagged, df_rolling_tagged], ignore_index=True)
    combined = combined.sort_values(
        ["target_type", "feature_set", "predictor"]
    ).reset_index(drop=True)

    combined.to_csv(out_path, index=False)
    print(f"Exact vs rolling combined matrix saved to: {out_path}")
    return combined


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    """Orchestrate Phase 4 exact-vs-rolling analysis and plotting."""
    print("=" * 60)
    print("Phase 4F - Exact vs Rolling Comparison Analysis")
    print("=" * 60)

    paths = ProjectPaths()
    paths.ensure_all()
    paths.figures_models.mkdir(parents=True, exist_ok=True)

    apply_style()

    # Load both matrices
    exact_matrix_path = paths.results_aggregate / "exact_forecasting_matrix.csv"
    rolling_matrix_path = paths.results_aggregate / "rolling_forecasting_matrix.csv"

    print(f"\nLoading exact matrix from:   {exact_matrix_path}")
    print(f"Loading rolling matrix from: {rolling_matrix_path}")

    df_exact = _load_matrix(exact_matrix_path)
    df_rolling = _load_matrix(rolling_matrix_path)

    print(f"Exact rows:   {len(df_exact)}")
    print(f"Rolling rows: {len(df_rolling)}")

    # Step 1: Build combined CSV
    print("\n[1/5] Building exact_vs_rolling_matrix.csv...")
    combined_path = paths.results_aggregate / "exact_vs_rolling_matrix.csv"
    build_exact_vs_rolling_matrix(df_exact, df_rolling, combined_path)

    # Step 2: Print stability table to console and save to CSV
    print("\n[2/5] Building stability metric table...")
    df_stability = build_stability_table(df_exact, df_rolling)
    if not df_stability.empty:
        stability_path = paths.results_aggregate / "stability_metrics.csv"
        df_stability.to_csv(stability_path, index=False)
        print(f"Stability table saved to: {stability_path}")
        print("\n=== Stability Metric Table ===")
        print(df_stability.to_string(index=False))

    # Step 3: Exact vs rolling validation stability plot
    print("\n[3/5] Plotting validation loss stability curves...")
    stability_stem = paths.figures_models / "exact_vs_rolling_stability"
    plot_exact_vs_rolling_stability(
        paths.results_runs, df_exact, df_rolling, stability_stem
    )

    # Step 4: Dumbbell MSE plot per horizon
    print("\n[4/5] Plotting exact vs rolling dumbbell MSE chart...")
    dumbbell_stem = paths.figures_models / "exact_vs_rolling_dumbbell_mse"
    plot_exact_vs_rolling_dumbbell(df_exact, df_rolling, dumbbell_stem)

    # Step 5: Rolling heatmap
    print("\n[5/5] Plotting rolling experiment horizon heatmap...")
    heatmap_stem = paths.figures_models / "rolling_model_horizon_heatmap"
    plot_rolling_horizon_heatmap(df_rolling, heatmap_stem)

    # Also produce exact/rolling x feature-set matrix (bonus)
    matrix_stem = paths.figures_models / "exact_vs_rolling_feature_matrix"
    plot_exact_vs_rolling_feature_matrix(df_exact, df_rolling, matrix_stem)

    print("\n" + "=" * 60)
    print("Phase 4F analysis complete!")
    print("Outputs:")
    print(f"  {combined_path}")
    print(f"  {stability_stem}.{{png,pdf}}")
    print(f"  {dumbbell_stem}.{{png,pdf}}")
    print(f"  {heatmap_stem}.{{png,pdf}}")
    print(f"  {matrix_stem}.{{png,pdf}}")
    print("=" * 60)


if __name__ == "__main__":
    main()
