"""Helpers for the Streamlit app: load artifacts, slider ranges, presets, and predictions.

Kept separate from app.py so the logic can be unit-tested without Streamlit. Heavy modules
(shap, xgboost via the model-selection code) are imported only when needed, so the hosted
app stays well inside a small memory limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from oncolens import config
from oncolens.data import load_splits, split_features_target
from oncolens.io_utils import read_json

DEFAULT_THRESHOLD: float = 0.5
FEATURE_GROUPS: tuple[str, ...] = ("mean", "error", "worst")


@dataclass
class Prediction:
    """The app's output for one set of measurements."""

    p_malignant: float
    threshold: float

    @property
    def is_malignant(self) -> bool:
        """True when the probability reaches the decision threshold."""
        return self.p_malignant >= self.threshold

    @property
    def label(self) -> str:
        """Predicted class name."""
        return config.CLASS_NAMES[int(self.is_malignant)]

    @property
    def confidence(self) -> float:
        """Model probability of the predicted class (not a clinical certainty)."""
        return self.p_malignant if self.is_malignant else 1.0 - self.p_malignant


def load_or_train_model() -> Pipeline:
    """Load the saved final model, or re-fit logistic regression if it is missing.

    The re-fit runs in a single process: parallel workers each load their own copy of
    numpy/scikit-learn, which is enough to exceed a small hosting memory limit.
    """
    if config.FINAL_MODEL_FILE.exists():
        return joblib.load(config.FINAL_MODEL_FILE)
    from oncolens.selection import tune_model  # imports xgboost; only needed for this fallback

    train, _ = load_splits()
    X, y = split_features_target(train)
    model = tune_model("logistic_regression", X, y, n_jobs=1).best_estimator_
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.FINAL_MODEL_FILE)
    return model


def load_threshold() -> float:
    """Decision threshold chosen in Phase 3 (falls back to 0.5 if results are missing)."""
    path = config.RESULTS_DIR / "threshold_selection.json"
    return float(read_json(path)["chosen_threshold"]) if path.exists() else DEFAULT_THRESHOLD


def training_features() -> pd.DataFrame:
    """Raw training-split features (used for slider ranges and as the SHAP background)."""
    train, _ = load_splits()
    return split_features_target(train)[0]


def slider_ranges(X: pd.DataFrame) -> pd.DataFrame:
    """Min, max, median, and a sensible step for each feature, from the training split."""
    ranges = pd.DataFrame({"min": X.min(), "max": X.max(), "median": X.median()})
    ranges["step"] = ((ranges["max"] - ranges["min"]) / 200).apply(lambda s: float(f"{s:.1g}"))
    return ranges


def feature_group(name: str) -> str:
    """Which of the three measurement groups ("mean", "error", "worst") a feature belongs to."""
    if name.startswith("mean "):
        return "mean"
    if name.startswith("worst "):
        return "worst"
    return "error"


def presets(X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series]:
    """Named starting points for the sliders: class medians and the overall median."""
    return {
        "Training median (all tumors)": X.median(),
        "Typical benign (median of benign)": X[y == 0].median(),
        "Typical malignant (median of malignant)": X[y == 1].median(),
    }


def predict(model: Pipeline, features: pd.DataFrame, threshold: float) -> Prediction:
    """Probability of malignancy for a single-row DataFrame."""
    p = float(model.predict_proba(features)[:, 1][0])
    return Prediction(p_malignant=p, threshold=threshold)


@dataclass
class LinearShap:
    """Exact SHAP values for a scaler + logistic regression pipeline, without importing shap.

    With independent features, the SHAP value of feature j is coef_j * (x_j - mean_j) on the
    scaled inputs, and the base value is the model's log-odds at the background mean. This is
    what shap.LinearExplainer computes; a test checks the two agree.
    """

    pipeline: Pipeline
    coef: np.ndarray
    background_mean: np.ndarray
    base_value: float

    def __call__(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        scaled = self.pipeline[:-1].transform(features)
        values = (np.asarray(scaled) - self.background_mean) * self.coef
        return values[0], features.to_numpy()[0], self.base_value


def make_explainer(model: Pipeline, X_background: pd.DataFrame) -> Any:
    """A lightweight exact explainer for logistic regression, else the general shap-based one."""
    estimator = model.named_steps["model"]
    if isinstance(estimator, LogisticRegression):
        mean = np.asarray(model[:-1].transform(X_background)).mean(axis=0)
        coef = estimator.coef_[0]
        return LinearShap(model, coef, mean, float(estimator.intercept_[0] + coef @ mean))
    from oncolens.explain import build_explainer  # imports shap (~100 MB); only for other models

    return build_explainer(model, X_background)


def explain_one(explainer: Any, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """SHAP values, raw feature values, and base value for a single-row DataFrame."""
    if isinstance(explainer, LinearShap):
        return explainer(features)
    from oncolens.explain import explain_rows

    expl = explain_rows(explainer, features)
    return expl.values[0], expl.data[0], float(np.ravel(expl.base_values)[0])


def waterfall_rows(values: np.ndarray, data: np.ndarray, names: list[str], k: int) -> list[tuple[str, float]]:
    """Top-k contributions (largest first) plus one aggregated row for the remaining features."""
    order = np.argsort(-np.abs(values))
    rows = [(f"{names[i]} = {data[i]:.4g}", float(values[i])) for i in order[:k]]
    rest = order[k:]
    if len(rest):
        rows.append((f"{len(rest)} other features", float(values[rest].sum())))
    return rows


def format_probability(p: float) -> str:
    """Percent string that never rounds a non-certain probability to 0% or 100%."""
    if p > 0.999:
        return "> 99.9%"
    if p < 0.001:
        return "< 0.1%"
    return f"{p:.1%}"
