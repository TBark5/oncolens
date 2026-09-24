"""Helpers for the Streamlit app: load artifacts, slider ranges, presets, and predictions.

Kept separate from app.py so the logic can be unit-tested without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from oncolens import config
from oncolens.data import load_splits, split_features_target
from oncolens.explain import PipelineExplainer, build_explainer, explain_rows
from oncolens.io_utils import read_json
from oncolens.selection import tune_model

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
    """Load the saved final model, or re-fit logistic regression if it is missing."""
    if config.FINAL_MODEL_FILE.exists():
        return joblib.load(config.FINAL_MODEL_FILE)
    train, _ = load_splits()
    X, y = split_features_target(train)
    model = tune_model("logistic_regression", X, y).best_estimator_
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


def explain_one(explainer: PipelineExplainer, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """SHAP values, raw feature values, and base value for a single-row DataFrame."""
    expl = explain_rows(explainer, features)
    return expl.values[0], expl.data[0], float(np.ravel(expl.base_values)[0])


def make_explainer(model: Pipeline, X_background: pd.DataFrame) -> PipelineExplainer:
    """Thin wrapper so the app does not import explain internals directly."""
    return build_explainer(model, X_background)


def format_probability(p: float) -> str:
    """Percent string that never rounds a non-certain probability to 0% or 100%."""
    if p > 0.999:
        return "> 99.9%"
    if p < 0.001:
        return "< 0.1%"
    return f"{p:.1%}"
