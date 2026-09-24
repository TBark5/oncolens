"""Tests for SHAP explanations and error-analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oncolens.errors import label_outcomes, misclassified_table, relative_position
from oncolens.explain import build_explainer, explain_rows, global_importance, sigmoid, top_contributions
from oncolens.models import make_pipeline


@pytest.fixture(scope="module")
def fitted_lr(train_xy):
    X, y = train_xy
    return make_pipeline("logistic_regression").fit(X, y)


def test_shap_values_add_up_to_model_probability(fitted_lr, train_xy) -> None:
    X, _ = train_xy
    expl = explain_rows(build_explainer(fitted_lr, X), X.head(20))
    reconstructed = sigmoid(expl.values.sum(axis=1) + expl.base_values)
    np.testing.assert_allclose(reconstructed, fitted_lr.predict_proba(X.head(20))[:, 1], atol=1e-8)


def test_explanation_reports_raw_feature_values(fitted_lr, train_xy) -> None:
    X, _ = train_xy
    expl = explain_rows(build_explainer(fitted_lr, X), X.head(3))
    np.testing.assert_allclose(expl.data, X.head(3).to_numpy())


def test_global_importance_sorted_and_complete(fitted_lr, train_xy) -> None:
    X, _ = train_xy
    imp = global_importance(explain_rows(build_explainer(fitted_lr, X), X.head(50)))
    assert len(imp) == X.shape[1]
    assert imp["mean_abs_shap"].is_monotonic_decreasing


def test_top_contributions_largest_first(fitted_lr, train_xy) -> None:
    X, _ = train_xy
    top = top_contributions(explain_rows(build_explainer(fitted_lr, X), X.head(1)), row=0, k=4)
    assert len(top) == 4 and top["shap"].abs().is_monotonic_decreasing


def test_label_outcomes() -> None:
    out = label_outcomes(np.array([1, 1, 0, 0]), np.array([0.9, 0.1, 0.9, 0.1]), 0.5)
    assert out.tolist() == ["TP", "FN", "FP", "TN"]


def test_misclassified_table_keeps_only_errors() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    table = misclassified_table(X, pd.Series([1, 1, 0, 0]), np.array([0.9, 0.1, 0.9, 0.1]), 0.5, "t")
    assert sorted(table["outcome"]) == ["FN", "FP"]


def test_relative_position_scale() -> None:
    profile = pd.DataFrame({"TN": [0.0], "FP": [2.5], "FN": [7.5], "TP": [10.0]}, index=["f"])
    pos = relative_position(profile)
    assert pos.loc["f", "FP"] == pytest.approx(0.25) and pos.loc["f", "FN"] == pytest.approx(0.75)
