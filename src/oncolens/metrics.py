"""Classification metrics computed from labels and predicted probabilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Scorers passed to cross_validate; "recall" etc. refer to the positive class (malignant = 1).
CV_SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "brier": "neg_brier_score",
}


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute threshold-dependent and threshold-free metrics for the malignant class.

    Args:
        y_true: true labels (1 = malignant).
        proba: predicted probability of malignancy.
        threshold: probability at or above which a sample is called malignant.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def summarize_cv(cv_results: dict[str, np.ndarray]) -> dict[str, float]:
    """Turn sklearn ``cross_validate`` output into mean/std per metric.

    Brier comes back negated from sklearn (higher-is-better convention); we flip it back.
    """
    summary: dict[str, float] = {}
    for metric in CV_SCORING:
        scores = np.asarray(cv_results[f"test_{metric}"], dtype=float)
        if metric == "brier":
            scores = -scores
        summary[f"{metric}_mean"] = float(scores.mean())
        summary[f"{metric}_std"] = float(scores.std())
    return summary
