"""Error analysis: find misclassified samples and describe how they differ from the rest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_outcomes(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> pd.Series:
    """Tag each sample as TP, TN, FP (false alarm), or FN (missed malignant)."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    outcome = np.where(y_true == 1, np.where(y_pred == 1, "TP", "FN"), np.where(y_pred == 1, "FP", "TN"))
    return pd.Series(outcome, name="outcome")


def misclassified_table(X: pd.DataFrame, y: pd.Series, proba: np.ndarray, threshold: float,
                        split: str) -> pd.DataFrame:
    """One row per misclassified sample with its probability and raw feature values."""
    proba = np.asarray(proba)
    table = X.copy()
    table.insert(0, "split", split)
    table.insert(1, "row_in_split", np.arange(len(X)))
    table.insert(2, "true_label", np.asarray(y))
    table.insert(3, "outcome", label_outcomes(y, proba, threshold).to_numpy())
    table.insert(4, "p_malignant", proba)
    table.insert(5, "distance_to_threshold", np.abs(proba - threshold))
    return table[table["outcome"].isin(["FP", "FN"])].reset_index(drop=True)


def outcome_profile(X: pd.DataFrame, outcome: pd.Series, features: list[str]) -> pd.DataFrame:
    """Mean of each feature per outcome group, columns ordered TN, FP, FN, TP."""
    frame = X[features].copy()
    frame["outcome"] = outcome.to_numpy()
    order = [o for o in ("TN", "FP", "FN", "TP") if o in set(frame["outcome"])]
    return frame.groupby("outcome")[features].mean().T[order]


def standardized_profile(X: pd.DataFrame, outcome: pd.Series, features: list[str]) -> pd.DataFrame:
    """Like ``outcome_profile`` but in z-scores relative to the whole training split.

    Positive means "larger than the average tumor". Malignant tumors sit on the positive
    side for most size and shape features.
    """
    z = (X[features] - X[features].mean()) / X[features].std()
    return outcome_profile(z, outcome, features)


def relative_position(profile: pd.DataFrame) -> pd.DataFrame:
    """Where each error group sits between the class averages, per feature.

    0 means "same as the average correctly classified benign tumor (TN)" and 1 means
    "same as the average correctly classified malignant tumor (TP)". Values between 0 and 1
    indicate the error group sits in the overlap region between the two classes.
    """
    span = profile["TP"] - profile["TN"]
    cols = [c for c in ("FN", "FP") if c in profile.columns]
    return profile[cols].sub(profile["TN"], axis=0).div(span, axis=0)


def borderline_share(errors: pd.DataFrame, margin: float = 0.15) -> float:
    """Fraction of errors whose probability was within ``margin`` of the threshold."""
    if errors.empty:
        return 0.0
    return float((errors["distance_to_threshold"] <= margin).mean())
