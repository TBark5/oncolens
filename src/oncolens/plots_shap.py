"""SHAP figures: global importance bar chart, beeswarm summary, and per-patient waterfall."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap

from oncolens.config import RANDOM_STATE
from oncolens.explain import sigmoid
from oncolens.style import BENIGN_COLOR, INK_MUTED, INK_SECONDARY, MALIGNANT_COLOR, SERIES, SURFACE

# Feature value low -> high, blue -> gray -> orange (matches the class colors).
VALUE_CMAP = LinearSegmentedColormap.from_list("low_high", [BENIGN_COLOR, "#c3c2b7", MALIGNANT_COLOR])
THRESHOLD_COLOR = SERIES[6]


def plot_global_importance(importance: pd.DataFrame, unit: str, top_k: int = 15) -> plt.Figure:
    """Horizontal bar chart of mean |SHAP| for the most important features."""
    top = importance.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * top_k + 1.2))
    ax.barh(top["feature"], top["mean_abs_shap"], color=SERIES[0], height=0.65,
            edgecolor=SURFACE, linewidth=2)
    ax.set_xlabel(f"Mean |SHAP value| ({unit})")
    ax.set_title(f"Global feature importance (top {top_k}, training set)")
    ax.grid(axis="y", visible=False)
    return fig


def plot_beeswarm(explanation: shap.Explanation, max_display: int = 15) -> plt.Figure:
    """SHAP beeswarm: each dot is one patient; x = contribution, color = feature value."""
    np.random.seed(RANDOM_STATE)  # shap jitters overlapping dots with numpy global randomness
    shap.plots.beeswarm(explanation, max_display=max_display, show=False, color=VALUE_CMAP,
                        axis_color=INK_MUTED, s=14, plot_size=(9, 0.4 * max_display + 1.5))
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_title("SHAP summary: how each feature pushes predictions (training set)", loc="left")
    ax.set_xlabel("SHAP value (log-odds of malignancy; > 0 pushes toward malignant)")
    return fig


def waterfall_rows(values: np.ndarray, data: np.ndarray, names: list[str], k: int) -> list[tuple[str, float]]:
    """Top-k contributions (largest first) plus one aggregated row for the remaining features."""
    order = np.argsort(-np.abs(values))
    rows = [(f"{names[i]} = {data[i]:.4g}", float(values[i])) for i in order[:k]]
    rest = order[k:]
    if len(rest):
        rows.append((f"{len(rest)} other features", float(values[rest].sum())))
    return rows


def plot_waterfall(values: np.ndarray, data: np.ndarray, names: list[str], base: float, title: str,
                   threshold: float | None = None, k: int = 10) -> plt.Figure:
    """Per-patient waterfall in log-odds: base value -> contributions -> final prediction.

    Orange bars push toward malignant, blue bars toward benign. Bars accumulate from the
    bottom (base value) up to the top (the patient's prediction).
    """
    rows = waterfall_rows(values, data, names, k)[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2.0))
    ends = base + np.cumsum([v for _, v in rows])
    final = float(ends[-1])
    cut = float(np.log(threshold / (1 - threshold))) if threshold is not None else None
    lo = min(base, float(ends.min()), cut if cut is not None else base)
    hi = max(base, float(ends.max()), cut if cut is not None else base)
    pad = 0.12 * (hi - lo or 1.0)
    ax.axvline(base, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)
    ax.axvline(final, color=INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=1)
    position = base
    for y, (label, value) in enumerate(rows):
        color = MALIGNANT_COLOR if value > 0 else BENIGN_COLOR
        ax.barh(y, value, left=position, color=color, height=0.62, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        nudge = 0.015 * (hi - lo) * (1 if value >= 0 else -1)
        ax.text(position + value + nudge, y, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right",
                fontsize=9, color=INK_SECONDARY, zorder=3,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5})
        position += value
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=9.5)
    top = len(rows) - 0.4
    if cut is not None:
        ax.axvline(cut, color=THRESHOLD_COLOR, linewidth=1.4, zorder=1)
        ax.text(cut, top + 0.1, f" decision threshold (p = {threshold:g})", color=THRESHOLD_COLOR,
                fontsize=9, va="bottom")
    ax.text(0.99, 0.01, f"base value (mean log-odds) {base:.2f}   ->   "
            f"this patient {final:.2f} (p = {sigmoid(final):.3f})", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.5, color=INK_SECONDARY, bbox={"facecolor": SURFACE, "edgecolor": "none"})
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-1.3, top + 0.9)
    ax.set_xlabel("Log-odds of malignancy (orange = toward malignant, blue = toward benign)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return fig
