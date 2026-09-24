"""Model pipelines and hyperparameter grids.

Every model is wrapped in a Pipeline whose first step is a StandardScaler. Because the
scaler lives inside the pipeline, cross-validation re-fits it on each training fold, so
no statistics from a validation fold ever leak into training.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from oncolens import config

MODEL_NAMES: tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting", "svm")

DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "svm": "SVM (RBF kernel)",
}


def make_estimator(name: str) -> ClassifierMixin:
    """Return an untuned classifier with sensible defaults and a fixed random seed."""
    seed = config.RANDOM_STATE
    estimators: dict[str, ClassifierMixin] = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "svm": SVC(kernel="rbf", probability=True, random_state=seed),
    }
    if name not in estimators:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(estimators)}")
    return estimators[name]


def make_pipeline(name: str) -> Pipeline:
    """Build a scaler + classifier pipeline for the given model name."""
    return Pipeline([("scaler", StandardScaler()), ("model", make_estimator(name))])


def param_grid(name: str) -> dict[str, list[Any]]:
    """Small, conventional hyperparameter grids for GridSearchCV (keys use the pipeline prefix)."""
    grids: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]},
        "random_forest": {
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.3],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
        "svm": {
            "model__C": [0.1, 1.0, 10.0, 100.0],
            "model__gamma": ["scale", 0.001, 0.01, 0.1],
        },
    }
    return grids[name]


def make_cv() -> StratifiedKFold:
    """Stratified 5-fold splitter with a fixed seed, shared by every experiment."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
