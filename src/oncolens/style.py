"""Shared figure style: colorblind-safe palette, fonts, and a save helper.

Colors come from a validated categorical palette (checked for color-vision
deficiency separation). Class colors are fixed: benign = blue, malignant = orange.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: scripts never open windows

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

DPI: int = 300

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots in fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BENIGN_COLOR = SERIES[0]
MALIGNANT_COLOR = SERIES[1]
CLASS_COLORS = {0: BENIGN_COLOR, 1: MALIGNANT_COLOR}

# Sequential (magnitude) and diverging (polarity) colormaps.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "oncolens_blue", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "oncolens_div", ["#1c5cab", "#6da7ec", "#f0efec", "#ec7a78", "#b3302f"]
)


def apply_style() -> None:
    """Set matplotlib rcParams used by every figure in the project."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.labelsize": 11,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "axes.prop_cycle": plt.cycler(color=SERIES),
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a figure at 300 dpi with a tight bounding box, then close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
