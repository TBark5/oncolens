"""Model-evaluation figures: CV comparison, ROC/PR overlays, calibration, thresholds,
confusion matrices, learning curve, and the error profile."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from oncolens.models import DISPLAY_NAMES, MODEL_NAMES
from oncolens.style import (
    BENIGN_COLOR,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MALIGNANT_COLOR,
    SEQUENTIAL,
    SERIES,
    SURFACE,
)

# Each model keeps the same color and line style in every chart (style = secondary encoding).
MODEL_COLORS = dict(zip(MODEL_NAMES, SERIES))
MODEL_STYLES = dict(zip(MODEL_NAMES, ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]))
THRESHOLD_COLOR = SERIES[6]


def plot_cv_comparison(nested: pd.DataFrame, default: pd.DataFrame) -> plt.Figure:
    """Mean +/- std across the 5 outer folds for ROC AUC and recall, tuned vs default."""
    metrics = [("roc_auc", "ROC AUC"), ("recall", "Recall (malignant) at p >= 0.5")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.0 * len(MODEL_NAMES) + 0.8), sharey=True)
    y_pos = np.arange(len(MODEL_NAMES))[::-1]
    for ax, (key, label) in zip(axes, metrics):
        for offset, frame, marker in ((0.14, nested, "o"), (-0.14, default, "s")):
            for y, model in zip(y_pos, MODEL_NAMES):
                mean, std = frame.loc[model, f"{key}_mean"], frame.loc[model, f"{key}_std"]
                ax.errorbar(mean, y + offset, xerr=std, fmt=marker, ms=8, capsize=3, lw=1.5,
                            color=MODEL_COLORS[model], mfc=MODEL_COLORS[model] if marker == "o" else SURFACE,
                            mew=1.8)
        if key == "roc_auc":
            for y, model in zip(y_pos, MODEL_NAMES):
                ax.text(1.0012, y + 0.14, f"{nested.loc[model, 'roc_auc_mean']:.4f}", va="center",
                        fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel(f"{label} (mean +/- std over 5 folds)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y_pos, [DISPLAY_NAMES[m] for m in MODEL_NAMES])
    axes[0].set_xlim(right=1.0045)
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=INK_SECONDARY, label="Tuned (nested CV)"),
               Line2D([], [], marker="s", ls="", ms=8, mfc=SURFACE, mew=1.8, color=INK_SECONDARY,
                      label="Default hyperparameters")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("Model comparison, stratified 5-fold CV on the training split", x=0.01, ha="left",
                 fontweight="semibold", y=1.0)
    fig.tight_layout()
    return fig


def _overlay(ax: plt.Axes, oof: pd.DataFrame, kind: str) -> None:
    """Draw ROC or PR curves for every model on one axis."""
    y = oof["y_true"].to_numpy()
    for model in MODEL_NAMES:
        p = oof[model].to_numpy()
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, linestyle=MODEL_STYLES[model], color=MODEL_COLORS[model],
                    label=f"{DISPLAY_NAMES[model]} (AUC {roc_auc_score(y, p):.4f})")
        else:
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, linestyle=MODEL_STYLES[model], color=MODEL_COLORS[model], drawstyle="steps-post",
                    label=f"{DISPLAY_NAMES[model]} (AP {average_precision_score(y, p):.4f})")


def plot_roc_overlay(oof: pd.DataFrame) -> plt.Figure:
    """Overlaid ROC curves from out-of-fold predictions, with a zoomed inset of the top-left corner."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _overlay(ax, oof, "roc")
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=":", label="Chance")
    ax.set_xlabel("False positive rate (benign flagged as malignant)")
    ax.set_ylabel("True positive rate (recall, malignant)")
    ax.set_title("ROC curves (out-of-fold, training split)")
    ax.legend(loc="lower right")
    inset = ax.inset_axes([0.38, 0.36, 0.42, 0.36])
    _overlay(inset, oof, "roc")
    inset.set_xlim(0, 0.15)
    inset.set_ylim(0.85, 1.005)
    inset.set_title("zoom", fontsize=9, color=INK_MUTED)
    inset.tick_params(labelsize=8)
    return fig


def plot_pr_overlay(oof: pd.DataFrame) -> plt.Figure:
    """Overlaid precision-recall curves from out-of-fold predictions."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    _overlay(ax, oof, "pr")
    base_rate = oof["y_true"].mean()
    ax.axhline(base_rate, color=INK_MUTED, lw=1, ls=":", label=f"Chance (prevalence {base_rate:.2f})")
    ax.set_xlabel("Recall (malignant cases caught)")
    ax.set_ylabel("Precision (flagged cases that are malignant)")
    ax.set_ylim(0, 1.03)
    ax.set_title("Precision-recall curves (out-of-fold, training split)")
    ax.legend(loc="lower left")
    inset = ax.inset_axes([0.12, 0.36, 0.42, 0.34])
    _overlay(inset, oof, "pr")
    inset.set_xlim(0.85, 1.005)
    inset.set_ylim(0.8, 1.005)
    inset.set_title("zoom", fontsize=9, color=INK_MUTED)
    inset.tick_params(labelsize=8)
    return fig


def plot_calibration(oof: pd.DataFrame, calib: pd.DataFrame, chosen: str, n_bins: int = 10) -> plt.Figure:
    """Reliability diagram for every model plus a histogram of the chosen model's predictions."""
    fig, (ax, hist_ax) = plt.subplots(2, 1, figsize=(7.5, 7.5), height_ratios=[3, 1], sharex=True)
    y = oof["y_true"].to_numpy()
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=":", label="Perfect calibration")
    for model in MODEL_NAMES:
        frac, mean_pred = calibration_curve(y, oof[model], n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac, linestyle=MODEL_STYLES[model], marker="o", ms=5, color=MODEL_COLORS[model],
                label=f"{DISPLAY_NAMES[model]} (Brier {calib.loc[model, 'brier']:.3f}, "
                      f"ECE {calib.loc[model, 'ece']:.3f})")
    ax.set_ylabel("Observed fraction malignant")
    ax.set_title("Calibration (out-of-fold, training split, 10 bins)")
    ax.text(0.99, 0.03, "Middle bins hold only a few samples each (see histogram),\nso their points are noisy.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=INK_MUTED)
    ax.legend(loc="upper left", fontsize=9)
    hist_ax.hist(oof[chosen], bins=20, range=(0, 1), color=MODEL_COLORS[chosen],
                 edgecolor=SURFACE)
    hist_ax.set_ylabel("Count")
    hist_ax.set_xlabel("Predicted probability of malignancy")
    hist_ax.set_title(f"{DISPLAY_NAMES[chosen]} (chosen model): most predictions are near 0 or 1", fontsize=10)
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(sweep: pd.DataFrame, threshold: float) -> plt.Figure:
    """Recall, precision, and specificity as the decision threshold moves."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for col, color, style in (("recall", MALIGNANT_COLOR, "-"), ("precision", SERIES[2], "--"),
                              ("specificity", BENIGN_COLOR, "-.")):
        ax.plot(sweep["threshold"], sweep[col], style, color=color, label=col.capitalize())
    row = sweep.loc[np.isclose(sweep["threshold"], threshold)].iloc[0]
    ax.axvline(threshold, color=THRESHOLD_COLOR, lw=1.4)
    ax.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
    ax.text(threshold + 0.01, 0.62, f"chosen t = {threshold:g}\nrecall {row['recall']:.3f}\n"
            f"precision {row['precision']:.3f}", color=THRESHOLD_COLOR, fontsize=9.5)
    ax.text(0.51, 0.62, "default 0.5", color=INK_MUTED, fontsize=9.5)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Decision threshold on p(malignant)")
    ax.set_ylabel("Metric value (out-of-fold, training split)")
    ax.set_title("Threshold trade-off: lowering the threshold catches more malignant cases")
    ax.legend(loc="lower right")
    return fig


def _draw_confusion(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    """One confusion-matrix panel with counts and row percentages."""
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n{share:.1%}", ha="center", va="center", fontsize=12,
                    color=SURFACE if share > 0.5 else INK)
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_matrices(panels: list[tuple[str, np.ndarray]]) -> plt.Figure:
    """A 2x2 grid of confusion matrices (e.g. train-OOF/test by default/chosen threshold)."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    for ax, (title, cm) in zip(axes.ravel(), panels):
        _draw_confusion(ax, cm, title)
    fig.suptitle("Confusion matrices, logistic regression (row % = share of the actual class)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return fig


def plot_learning_curve(curve: pd.DataFrame, metric_label: str) -> plt.Figure:
    """Training vs validation score as the number of training samples grows."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix, color, style, name in (("train", SERIES[0], "--", "Training folds"),
                                       ("val", SERIES[1], "-", "Validation folds")):
        mean, std = curve[f"{prefix}_mean"], curve[f"{prefix}_std"]
        ax.plot(curve["n_samples"], mean, style, marker="o", ms=5, color=color, label=name)
        ax.fill_between(curve["n_samples"], mean - std, mean + std, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel(metric_label)
    ax.set_title("Learning curve, logistic regression (stratified 5-fold CV)")
    ax.legend(loc="lower right")
    return fig


def plot_error_profile(z_profile: pd.DataFrame) -> plt.Figure:
    """Dot plot of mean z-scores per outcome group for the top SHAP features.

    Filled markers = correctly classified, hollow = errors; color = true class.
    """
    styles = {"TN": (BENIGN_COLOR, True, "o", "Benign, correct (TN)"),
              "FP": (BENIGN_COLOR, False, "o", "Benign, flagged malignant (FP)"),
              "FN": (MALIGNANT_COLOR, False, "D", "Malignant, missed (FN)"),
              "TP": (MALIGNANT_COLOR, True, "D", "Malignant, correct (TP)")}
    features = list(z_profile.index)[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(features) + 1.8))
    for y, feature in enumerate(features):
        ax.plot(z_profile.loc[feature].agg(["min", "max"]), [y, y], color=INK_MUTED, lw=1, alpha=0.5)
    for group, (color, filled, marker, label) in styles.items():
        if group in z_profile:
            ax.scatter(z_profile.loc[features, group], range(len(features)), s=80, marker=marker,
                       facecolor=color if filled else SURFACE, edgecolor=color, linewidth=2, label=label,
                       zorder=3)
    ax.axvline(0, color=INK_MUTED, lw=1, ls=":")
    ax.set_yticks(range(len(features)), features)
    ax.set_xlabel("Mean z-score (0 = average training tumor)")
    ax.set_title("Error profile: misclassified vs correctly classified tumors (out-of-fold)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax.grid(axis="y", visible=False)
    return fig
