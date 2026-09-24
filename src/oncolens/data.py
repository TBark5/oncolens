"""Load, validate, cache, and split the Wisconsin diagnostic breast cancer data."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from oncolens import config

EXPECTED_ROWS: int = 569
EXPECTED_FEATURES: int = 30


def load_raw_dataframe() -> pd.DataFrame:
    """Load the dataset from the local CSV cache, creating the cache on first use.

    The dataset ships inside scikit-learn, so no network access is needed. The CSV
    copy in ``data/`` makes the exact input visible and easy to inspect.

    sklearn encodes 0 = malignant, 1 = benign. We flip this so that the positive
    class (1) is malignant, which makes "recall" mean "share of malignant tumors caught".
    """
    if config.RAW_DATA_FILE.exists():
        return pd.read_csv(config.RAW_DATA_FILE)

    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    df[config.TARGET_COL] = (df.pop("target") == 0).astype(int)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_DATA_FILE, index=False)
    return df


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    """Run basic sanity checks and return a list of problems (empty if clean)."""
    problems: list[str] = []
    if len(df) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    features = feature_columns(df)
    if len(features) != EXPECTED_FEATURES:
        problems.append(f"expected {EXPECTED_FEATURES} features, got {len(features)}")
    if df.isna().any().any():
        problems.append("dataset contains missing values")
    if not set(df[config.TARGET_COL].unique()) <= {0, 1}:
        problems.append("target is not binary 0/1")
    if (df[features] < 0).any().any():
        problems.append("negative values found in physical measurements")
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        problems.append(f"{n_dupes} duplicate rows")
    return problems


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of the 30 numeric feature columns (everything but the target)."""
    return [c for c in df.columns if c != config.TARGET_COL]


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a dataframe into the feature matrix X and target vector y."""
    return df[feature_columns(df)], df[config.TARGET_COL]


def make_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split. The test part is held out until the final evaluation."""
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET_COL],
        random_state=config.RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cached train/test split, creating it from the raw data if missing."""
    if config.TRAIN_FILE.exists() and config.TEST_FILE.exists():
        return pd.read_csv(config.TRAIN_FILE), pd.read_csv(config.TEST_FILE)
    train, test = make_train_test_split(load_raw_dataframe())
    train.to_csv(config.TRAIN_FILE, index=False)
    test.to_csv(config.TEST_FILE, index=False)
    return train, test


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature descriptive statistics, split by class (mean for each class)."""
    features = feature_columns(df)
    summary = df[features].describe().T[["mean", "std", "min", "50%", "max"]]
    summary = summary.rename(columns={"50%": "median"})
    by_class = df.groupby(config.TARGET_COL)[features].mean().T
    summary["mean_benign"] = by_class[0]
    summary["mean_malignant"] = by_class[1]
    summary["malignant_to_benign_ratio"] = by_class[1] / by_class[0]
    return summary.round(4)
