"""Project-wide constants: paths, random seed, split sizes, and label names."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Recall target for the malignant class when choosing the decision threshold.
TARGET_RECALL: float = 0.98

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
MODELS_DIR: Path = PROJECT_ROOT / "models"

RAW_DATA_FILE: Path = DATA_DIR / "breast_cancer.csv"
TRAIN_FILE: Path = DATA_DIR / "train.csv"
TEST_FILE: Path = DATA_DIR / "test.csv"
FINAL_MODEL_FILE: Path = MODELS_DIR / "final_model.joblib"

TARGET_COL: str = "malignant"
# After remapping, 1 = malignant (the positive class we want to catch), 0 = benign.
CLASS_NAMES: dict[int, str] = {0: "Benign", 1: "Malignant"}


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
