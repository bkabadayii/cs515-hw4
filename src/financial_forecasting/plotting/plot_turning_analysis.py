"""Turning-point analysis plots for Phase 5A.

Three plotting functions:
    1. plot_confusion_matrix   -- side-by-side for original vs auxiliary
    2. plot_precision_recall   -- PR curve with chance baseline
    3. plot_threshold_sweep    -- precision, recall, F1 vs threshold

All functions accept precomputed data so they can be called without
re-running any model training.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

from financial_forecasting.plotting.style import apply_style, save_figure

# Consistent colours for the two feature sets
COLOUR_ORIG = "#1F77B4"   # blue
COLOUR_AUX = "#FF7F0E"    # orange
COLOUR_CHANCE = "#AAAAAA"  # grey


def plot_confusion_matrix(
    cm_orig: list[list[int]],
    cm_aux: list[list[int]],
    feature_set_names: tuple[str, str],
    out_stem: str | pathlib.Path,
) -> tuple[str, str]:
    """Plot side-by-side confusion matrices for two feature sets.

    X-axis: Predicted label (Negative / Positive).
    Y-axis: True label (Negative / Positive).
    Cell values are raw counts; percentage of total is shown in parentheses.

    Parameters
    ----------
    cm_orig:
        Confusion matrix [[TN, FP], [FN, TP]] for original_ohlc.
    cm_aux:
        Confusion matrix [[TN, FP], [FN, TP]] for auxiliary_ohlc.
    feature_set_names:
        Display names for the two feature sets.
    out_stem:
        Output file path without extension (PNG and PDF saved).

    Returns
    -------
    tuple[str, str]:
        Paths to the saved PNG and PDF files.
    """
    apply_style()
    labels = ["Negative\n(Pass)", "Positive\n(Buy)"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Turning-Point Classifier: Confusion Matrices\n"
        "(gamma = 1.1,  threshold = 0.5)",
        fontsize=13,
        y=1.02,
    )

    cms = [cm_orig, cm_aux]
    titles = [f"{feature_set_names[0]}", f"{feature_set_names[1]}"]
    colours = [COLOUR_ORIG, COLOUR_AUX]

    for ax, cm, title, colour in zip(axes, cms, titles, colours):
        cm_arr = np.array(cm, dtype=np.int64)
        total = cm_arr.sum()
        # Normalised for colour intensity
        cm_norm = cm_arr.astype(float) / (total + 1e-9)

        im = ax.imshow(
            cm_norm, cmap="Blues", vmin=0, vmax=1.0, aspect="auto"
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(2):
            for j in range(2):
                count = int(cm_arr[i, j])
                pct = 100.0 * cm_arr[i, j] / (total + 1e-9)
                text_colour = "white" if cm_norm[i, j] > 0.5 else "#1A1A1A"
                ax.text(
                    j,
                    i,
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_colour,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        ax.set_title(f"{title}", fontsize=12, color=colour, fontweight="bold")

    fig.tight_layout()
    return save_figure(fig, str(out_stem))


def plot_precision_recall_curve(
    pr_data: dict[str, tuple[list[float], list[float], float]],
    out_stem: str | pathlib.Path,
    positive_class_rate: float = 0.0,
) -> tuple[str, str]:
    """Plot precision-recall curves for each feature set.

    X-axis: Recall (sensitivity, true-positive rate).
    Y-axis: Precision (positive predictive value).
    The dashed horizontal line marks the positive class rate (no-skill
    classifier baseline).

    Parameters
    ----------
    pr_data:
        Mapping from feature set name to (precision_list, recall_list, pr_auc).
    out_stem:
        Output path stem.
    positive_class_rate:
        Fraction of positive labels (shown as no-skill baseline).

    Returns
    -------
    tuple[str, str]:
        Paths to PNG and PDF.
    """
    apply_style()
    colours = [COLOUR_ORIG, COLOUR_AUX]

    fig, ax = plt.subplots(figsize=(7, 5))

    for (name, (prec, rec, auc)), colour in zip(pr_data.items(), colours):
        if len(prec) > 0 and len(rec) > 0:
            # Sort by recall for a clean curve
            pairs = sorted(zip(rec, prec))
            rec_sorted = [p[0] for p in pairs]
            prec_sorted = [p[1] for p in pairs]
            label_str = f"{name}  (PR-AUC = {auc:.4f})"
            ax.plot(rec_sorted, prec_sorted, color=colour, linewidth=2, label=label_str)
        else:
            ax.plot([], [], color=colour, label=f"{name}  (no PR curve)")

    # No-skill baseline
    if positive_class_rate > 0:
        ax.axhline(
            y=positive_class_rate,
            color=COLOUR_CHANCE,
            linestyle="--",
            linewidth=1.2,
            label=f"No-skill (pos rate = {positive_class_rate:.4f})",
        )
    else:
        ax.axhline(
            y=0.0,
            color=COLOUR_CHANCE,
            linestyle="--",
            linewidth=1.2,
            label="No-skill (pos rate = 0.000 -- no positives)",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curve\n"
        "Turning-point classifier (gamma = 1.1)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_figure(fig, str(out_stem))


def plot_threshold_sweep(
    sweep_data: dict[str, dict[str, list[float]]],
    out_stem: str | pathlib.Path,
) -> tuple[str, str]:
    """Plot precision, recall, and F1 as a function of classification threshold.

    X-axis: Classification threshold (0.0 to 1.0).
    Y-axis: Metric value (0.0 to 1.0).
    Each feature set is shown in one subplot row.

    Parameters
    ----------
    sweep_data:
        Mapping from feature set name to threshold sweep dict with keys
        'thresholds', 'precision', 'recall', 'f1'.
    out_stem:
        Output path stem.

    Returns
    -------
    tuple[str, str]:
        Paths to PNG and PDF.
    """
    apply_style()

    n_sets = len(sweep_data)
    fig, axes = plt.subplots(1, n_sets, figsize=(6.5 * n_sets, 4.5), sharey=True)
    if n_sets == 1:
        axes = [axes]

    colours_map = {
        "precision": "#E74C3C",
        "recall": "#2ECC71",
        "f1": "#8E44AD",
    }

    for ax, (name, sweep) in zip(axes, sweep_data.items()):
        thresholds = sweep["thresholds"]
        for metric in ("precision", "recall", "f1"):
            ax.plot(
                thresholds,
                sweep[metric],
                color=colours_map[metric],
                linewidth=1.8,
                label=metric.capitalize(),
            )

        ax.axvline(x=0.5, color="#888888", linestyle=":", linewidth=1.0, label="thr=0.5")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Threshold", fontsize=11)
        ax.set_ylabel("Metric value", fontsize=11)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Threshold Sweep: Precision / Recall / F1\n"
        "Turning-point classifier (gamma = 1.1, test split)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return save_figure(fig, str(out_stem))
