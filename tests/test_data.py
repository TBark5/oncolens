"""Tests for loading, validating, and splitting the data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oncolens import config
from oncolens.data import (
    feature_columns,
    load_splits,
    make_train_test_split,
    summary_table,
    validate_dataframe,
)


def test_target_is_flipped_so_malignant_is_one(raw_df: pd.DataFrame) -> None:
    assert raw_df.shape == (569, 31)
    assert int(raw_df[config.TARGET_COL].sum()) == 212  # 212 malignant in the original dataset


def test_clean_data_passes_validation(raw_df: pd.DataFrame) -> None:
    assert validate_dataframe(raw_df) == []


def test_validation_catches_missing_and_negative_values(raw_df: pd.DataFrame) -> None:
    broken = raw_df.copy()
    broken.iloc[0, 0] = np.nan
    broken.iloc[1, 1] = -5.0
    problems = " ".join(validate_dataframe(broken))
    assert "missing" in problems and "negative" in problems


def test_split_is_disjoint_stratified_and_reproducible(raw_df: pd.DataFrame) -> None:
    train, test = make_train_test_split(raw_df)
    assert len(train) + len(test) == len(raw_df)
    assert len(pd.merge(train, test, how="inner")) == 0  # no row appears in both
    assert abs(train[config.TARGET_COL].mean() - test[config.TARGET_COL].mean()) < 0.01
    train_again, _ = make_train_test_split(raw_df)
    pd.testing.assert_frame_equal(train, train_again)


def test_cached_split_matches_fresh_split(raw_df: pd.DataFrame) -> None:
    cached_train, cached_test = load_splits()
    fresh_train, fresh_test = make_train_test_split(raw_df)
    np.testing.assert_allclose(cached_train.to_numpy(), fresh_train.to_numpy())
    np.testing.assert_allclose(cached_test.to_numpy(), fresh_test.to_numpy())


def test_summary_table_has_one_row_per_feature(raw_df: pd.DataFrame) -> None:
    table = summary_table(raw_df)
    assert list(table.index) == feature_columns(raw_df)
    assert {"mean_benign", "mean_malignant", "malignant_to_benign_ratio"} <= set(table.columns)
