"""Exploratory figures: class balance, correlation heatmap, PCA scatter, feature distributions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oncolens import config
from oncolens.data import feature_columns
from oncolens.style import CLASS_COLORS, DIVERGING, INK_SECONDARY, SURFACE


def plot_class_balance(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of how many samples fall in each class."""
    counts = df[config.TARGET_COL].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = [config.CLASS_NAMES[i] for i in counts.index]
    bars = ax.barh(labels, counts.values, color=[CLASS_COLORS[i] for i in counts.index],
                   height=0.55, edgecolor=SURFACE, linewidth=2)
    total = counts.sum()
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n} ({n / total:.1%})", va="center", color=INK_SECONDARY)
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_xlabel("Number of samples")
    ax.set_title(f"Class balance (n = {total})")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Lower-triangle Pearson correlation heatmap of the 30 features."""
    corr = df[feature_columns(df)].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(np.ma.masked_where(mask, corr.values), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Pearson correlation")
    cbar.outline.set_visible(False)
    ax.set_title("Feature correlations (training set): many size features are near-duplicates")
    return fig


def plot_pca_scatter(df: pd.DataFrame) -> plt.Figure:
    """2-D PCA projection of standardized features, colored by class."""
    features = feature_columns(df)
    pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=config.RANDOM_STATE))
    coords = pca_pipe.fit_transform(df[features])
    explained = pca_pipe.named_steps["pca"].explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label in (0, 1):
        sel = df[config.TARGET_COL].to_numpy() == label
        ax.scatter(coords[sel, 0], coords[sel, 1], s=28, alpha=0.75, color=CLASS_COLORS[label],
                   edgecolor=SURFACE, linewidth=0.6, label=config.CLASS_NAMES[label])
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.set_title("PCA projection (training set): classes separate mostly along PC1")
    ax.legend(loc="upper right")
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: list[str]) -> plt.Figure:
    """Overlaid per-class histograms for a handful of informative features."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4))
    for ax, feature in zip(np.atleast_1d(axes), features):
        for label in (0, 1):
            values = df.loc[df[config.TARGET_COL] == label, feature]
            ax.hist(values, bins=25, alpha=0.6, color=CLASS_COLORS[label],
                    label=config.CLASS_NAMES[label], edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel("Value")
    np.atleast_1d(axes)[0].set_ylabel("Count")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle("Distributions of selected features by class (training set)",
                 x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return fig
