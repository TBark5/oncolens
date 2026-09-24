"""Probability calibration diagnostics: reliability-curve bins, ECE, and Brier score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

N_BINS: int = 10
# Pre-registered rule: recalibrate the chosen model only if its expected calibration
# error on out-of-fold predictions is worse than this.
ECE_RECALIBRATION_LIMIT: float = 0.05


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = N_BINS) -> float:
    """Weighted mean gap between predicted probability and observed frequency.

    Uses equal-width bins over [0, 1]; each bin's gap is weighted by its share of samples.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    bin_ids = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_ids == b
        if in_bin.any():
            ece += in_bin.mean() * abs(proba[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def calibration_table(y_true: np.ndarray, proba: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Observed fraction of malignant cases versus mean predicted probability, per bin."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="uniform")
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_malignant": frac_pos})


def calibration_summary(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Brier score and ECE for one model's probabilities."""
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "ece": expected_calibration_error(y_true, proba),
    }
