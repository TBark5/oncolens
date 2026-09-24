"""Tests for pipelines, grids, and the model-selection rule."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from oncolens.models import MODEL_NAMES, make_pipeline, param_grid
from oncolens.selection import SIMPLICITY_MARGIN, choose_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_pipeline_scales_inside_the_pipeline(name: str) -> None:
    pipe = make_pipeline(name)
    assert isinstance(pipe.steps[0][1], StandardScaler)  # so the scaler is re-fit in each CV fold


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_grid_keys_are_valid_pipeline_params(name: str) -> None:
    pipe = make_pipeline(name)
    first_values = {key: values[0] for key, values in param_grid(name).items()}
    pipe.set_params(**first_values)  # raises ValueError on an invalid key


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        make_pipeline("deep_magic")


def _nested(scores: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"roc_auc_mean": scores})


def test_choose_model_prefers_lr_within_margin() -> None:
    chosen, _ = choose_model(_nested({"logistic_regression": 0.990, "svm": 0.990 + SIMPLICITY_MARGIN / 2,
                                      "random_forest": 0.98, "gradient_boosting": 0.98}))
    assert chosen == "logistic_regression"


def test_choose_model_takes_clear_winner() -> None:
    chosen, _ = choose_model(_nested({"logistic_regression": 0.95, "svm": 0.99,
                                      "random_forest": 0.98, "gradient_boosting": 0.97}))
    assert chosen == "svm"
