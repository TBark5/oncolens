"""Tests for metrics, threshold selection, and calibration helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oncolens.calibration import calibration_summary, expected_calibration_error
from oncolens.metrics import classification_metrics, summarize_cv
from oncolens.threshold import choose_threshold, threshold_sweep

Y = np.array([0, 0, 0, 1, 1, 1])
P = np.array([0.1, 0.3, 0.6, 0.4, 0.8, 0.9])


def test_classification_metrics_counts_and_rates() -> None:
    m = classification_metrics(Y, P, threshold=0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)


def test_lower_threshold_never_lowers_recall() -> None:
    assert classification_metrics(Y, P, 0.35)["recall"] >= classification_metrics(Y, P, 0.5)["recall"]


def test_summarize_cv_flips_negated_brier() -> None:
    fake = {f"test_{m}": np.array([0.9, 1.0]) for m in
            ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")}
    fake["test_brier"] = np.array([-0.02, -0.04])
    assert summarize_cv(fake)["brier_mean"] == pytest.approx(0.03)


def test_choose_threshold_picks_highest_meeting_target() -> None:
    sweep = threshold_sweep(Y, P, grid=np.array([0.2, 0.35, 0.5, 0.7]))
    # recall is 1.0 at 0.2 and 0.35 but 2/3 at 0.5, so the highest threshold meeting 1.0 is 0.35
    assert choose_threshold(sweep, target_recall=1.0) == 0.35


def test_choose_threshold_falls_back_to_lowest() -> None:
    sweep = pd.DataFrame({"threshold": [0.3, 0.6], "recall": [0.5, 0.4]})
    assert choose_threshold(sweep, target_recall=0.99) == 0.3


def test_ece_zero_when_frequencies_match() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_large_when_confidently_wrong() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.95, 0.95, 0.05, 0.05])
    assert expected_calibration_error(y, p) > 0.9
    assert calibration_summary(y, p)["brier"] > 0.8
