"""Model comparison, hyperparameter tuning, and model selection.

Three steps, all on the training split only:

1. ``compare_default_models``: stratified 5-fold CV of each model with default settings.
2. ``nested_cv_tuned``: nested CV (GridSearchCV inside each outer fold). This gives an
   honest estimate of "tune, then predict", because the outer validation fold never
   influences which hyperparameters are chosen.
3. ``tune_model``: one final GridSearchCV on the whole training split to pick the
   hyperparameters that are actually used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from oncolens.metrics import CV_SCORING, summarize_cv
from oncolens.models import MODEL_NAMES, make_cv, make_pipeline, param_grid

SELECTION_METRIC: str = "roc_auc"
# If logistic regression is this close to the best model, prefer it: it is the easiest
# model to explain and its SHAP values are exact.
SIMPLICITY_MARGIN: float = 0.005


def compare_default_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Cross-validate every model with default hyperparameters; one row per model."""
    rows = []
    for name in MODEL_NAMES:
        scores = cross_validate(make_pipeline(name), X, y, cv=make_cv(), scoring=CV_SCORING)
        rows.append({"model": name, **summarize_cv(scores)})
    return pd.DataFrame(rows).set_index("model")


def make_grid_search(name: str) -> GridSearchCV:
    """GridSearchCV over the model's grid, scored by ROC AUC with stratified 5-fold CV."""
    return GridSearchCV(
        make_pipeline(name),
        param_grid(name),
        scoring=SELECTION_METRIC,
        cv=make_cv(),
        n_jobs=-1,
    )


def nested_cv_tuned(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Nested CV of the full tuning procedure for every model; one row per model."""
    rows = []
    for name in MODEL_NAMES:
        scores = cross_validate(make_grid_search(name), X, y, cv=make_cv(), scoring=CV_SCORING)
        rows.append({"model": name, **summarize_cv(scores)})
    return pd.DataFrame(rows).set_index("model")


def tune_model(name: str, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    """Run GridSearchCV on the whole training split and return the fitted search."""
    search = make_grid_search(name)
    search.fit(X, y)
    return search


def choose_model(nested: pd.DataFrame) -> tuple[str, str]:
    """Pick the final model from nested-CV results and explain why in one sentence."""
    col = f"{SELECTION_METRIC}_mean"
    best = str(nested[col].idxmax())
    best_score = float(nested.loc[best, col])
    lr_score = float(nested.loc["logistic_regression", col])
    if best != "logistic_regression" and best_score - lr_score <= SIMPLICITY_MARGIN:
        return "logistic_regression", (
            f"{best} had the highest nested-CV ROC AUC ({best_score:.4f}) but logistic regression "
            f"({lr_score:.4f}) is within {SIMPLICITY_MARGIN}, so the simpler, more explainable model was kept."
        )
    return best, f"{best} had the highest nested-CV ROC AUC ({best_score:.4f})."


def out_of_fold_proba(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold predicted malignancy probabilities for every training row.

    Each row's probability comes from a model that never saw that row, so these can be
    used for calibration checks and threshold selection without touching the test set.
    """
    return cross_val_predict(pipeline, X, y, cv=make_cv(), method="predict_proba")[:, 1]
