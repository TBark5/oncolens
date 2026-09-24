"""Decision-threshold selection that prioritizes catching malignant cases.

Missing a malignant tumor (false negative) is far more costly than a false alarm
(false positive), which in practice leads to a follow-up test. So instead of the default
0.5 cut-off we pick the *highest* threshold that still reaches a target recall on
out-of-fold training predictions. "Highest" keeps as many false alarms away as possible
while meeting the recall target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oncolens.metrics import classification_metrics

THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.01, 1.00, 0.01), 2)


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> pd.DataFrame:
    """Precision, recall, specificity, and error counts at every candidate threshold."""
    rows = []
    for t in grid:
        m = classification_metrics(y_true, proba, float(t))
        rows.append({k: m[k] for k in ("threshold", "precision", "recall", "specificity", "f1", "fp", "fn")})
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame, target_recall: float) -> float:
    """Highest threshold whose recall meets the target; falls back to the lowest threshold."""
    ok = sweep[sweep["recall"] >= target_recall]
    if ok.empty:
        return float(sweep["threshold"].min())
    return float(ok["threshold"].max())
