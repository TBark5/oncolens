"""Tests for the app helpers and a headless run of the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from oncolens.app_support import (
    Prediction,
    feature_group,
    format_probability,
    load_or_train_model,
    predict,
    presets,
    slider_ranges,
)

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_prediction_label_and_confidence() -> None:
    assert Prediction(0.3, threshold=0.2).label == "Malignant"
    assert Prediction(0.1, threshold=0.2).confidence == pytest.approx(0.9)


def test_format_probability_never_claims_certainty() -> None:
    assert format_probability(0.99999) == "> 99.9%"
    assert format_probability(0.00001) == "< 0.1%"
    assert format_probability(0.5) == "50.0%"


def test_feature_groups() -> None:
    assert feature_group("mean radius") == "mean"
    assert feature_group("worst area") == "worst"
    assert feature_group("radius error") == "error"


def test_presets_lie_within_slider_ranges(train_xy) -> None:
    X, y = train_xy
    ranges = slider_ranges(X)
    for values in presets(X, y).values():
        assert ((values >= ranges["min"]) & (values <= ranges["max"])).all()


def test_typical_malignant_preset_is_predicted_malignant(train_xy) -> None:
    X, y = train_xy
    model = load_or_train_model()
    row = pd.DataFrame([presets(X, y)["Typical malignant (median of malignant)"]])
    assert predict(model, row, threshold=0.2).is_malignant


def test_app_runs_without_exceptions() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception
    assert len(at.slider) == 30
    assert at.metric[0].value in ("Benign", "Malignant")
    texture = at.slider(key="worst texture")
    texture.set_value(float(texture.max)).run()
    assert not at.exception
