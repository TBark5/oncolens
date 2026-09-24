"""Shared fixtures: the real (small) dataset and its cached train/test split."""

from __future__ import annotations

import pandas as pd
import pytest

from oncolens.data import load_raw_dataframe, load_splits, split_features_target


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Full 569-row dataset with the malignant = 1 target."""
    return load_raw_dataframe()


@pytest.fixture(scope="session")
def train_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels of the training split."""
    train, _ = load_splits()
    return split_features_target(train)
