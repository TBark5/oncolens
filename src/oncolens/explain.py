"""SHAP explanations for a fitted scaler + classifier pipeline.

For logistic regression we use SHAP's exact LinearExplainer on the standardized features.
Contributions are in log-odds of malignancy: they add up exactly from the base value
(average prediction on the background data) to the model's output for that patient.
For any other model we fall back to the model-agnostic permutation explainer on the
predicted probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from oncolens import config


@dataclass
class PipelineExplainer:
    """A SHAP explainer bound to a pipeline, plus the unit its values are expressed in."""

    pipeline: Pipeline
    explainer: shap.Explainer
    is_linear: bool

    @property
    def output_unit(self) -> str:
        """Human-readable name of the scale that SHAP values are expressed on."""
        return "log-odds of malignancy" if self.is_linear else "probability of malignancy"


def build_explainer(pipeline: Pipeline, X_background: pd.DataFrame) -> PipelineExplainer:
    """Create the right SHAP explainer for the pipeline's final estimator.

    Args:
        pipeline: fitted Pipeline whose last step is named "model".
        X_background: raw (unscaled) training features used as the SHAP background.
    """
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        background = pipeline[:-1].transform(X_background)
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(background, max_samples=1000))
        return PipelineExplainer(pipeline, explainer, is_linear=True)

    def predict_malignant(x: np.ndarray) -> np.ndarray:
        return pipeline.predict_proba(pd.DataFrame(x, columns=X_background.columns))[:, 1]

    background = shap.sample(X_background, 100, random_state=config.RANDOM_STATE)
    explainer = shap.PermutationExplainer(predict_malignant, shap.maskers.Independent(background),
                                          seed=config.RANDOM_STATE)
    return PipelineExplainer(pipeline, explainer, is_linear=False)


def explain_rows(pe: PipelineExplainer, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values for each row of X, reported alongside the raw (unscaled) feature values."""
    if pe.is_linear:
        raw = pe.explainer(pe.pipeline[:-1].transform(X))
    else:
        raw = pe.explainer(X.to_numpy())
    return shap.Explanation(
        values=np.asarray(raw.values),
        base_values=np.asarray(raw.base_values),
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )


def global_importance(explanation: shap.Explanation) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, sorted from most to least important."""
    importance = np.abs(explanation.values).mean(axis=0)
    df = pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": importance})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def top_contributions(explanation: shap.Explanation, row: int, k: int = 5) -> pd.DataFrame:
    """The k features with the largest absolute contribution for a single row."""
    values = explanation.values[row]
    order = np.argsort(-np.abs(values))[:k]
    return pd.DataFrame(
        {
            "feature": [explanation.feature_names[i] for i in order],
            "value": explanation.data[row][order],
            "shap": values[order],
        }
    )


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Convert log-odds to probability."""
    return 1.0 / (1.0 + np.exp(-z))
