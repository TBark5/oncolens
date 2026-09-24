"""Reproduce every result, figure, and model in this project from scratch.

Usage:
    python run_all.py            # re-run all steps (reuses the cached data CSVs)
    python run_all.py --clean    # delete generated data/results/figures/models first
    python run_all.py --tests    # also run the pytest suite at the end
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # lets run_all.py work even before `pip install -e .`

from oncolens import config  # noqa: E402

STEPS: list[tuple[str, str]] = [
    ("01_data.py", "Load, validate, split data; EDA figures"),
    ("02_baseline.py", "Logistic regression baseline"),
    ("03_model_selection.py", "Model comparison, nested CV, tuning"),
    ("04_calibration_threshold.py", "Calibration check and threshold selection"),
    ("05_final_test.py", "One-time held-out test evaluation"),
    ("06_explain_errors.py", "SHAP explanations and error analysis"),
    ("07_figures.py", "Evaluation, SHAP and error figures"),
]


def clean_outputs() -> None:
    """Remove everything the pipeline generates (never touches code, docs, or captions)."""
    for pattern_dir, pattern in ((config.DATA_DIR, "*.csv"), (config.RESULTS_DIR, "*.*"),
                                 (config.FIGURES_DIR, "*.png")):
        for path in pattern_dir.glob(pattern):
            path.unlink()
    shutil.rmtree(config.MODELS_DIR, ignore_errors=True)


def main() -> None:
    """Run each pipeline step in order and report timings."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true", help="delete generated outputs first")
    parser.add_argument("--tests", action="store_true", help="run pytest after the pipeline")
    args = parser.parse_args()

    if args.clean:
        clean_outputs()
        print("Cleaned data/, results/, figures/*.png and models/.")
    config.ensure_dirs()
    total = time.time()
    for script, description in STEPS:
        start = time.time()
        print(f"\n=== {script}: {description} ===", flush=True)
        runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
        print(f"--- done in {time.time() - start:.1f}s", flush=True)
    print(f"\nPipeline finished in {time.time() - total:.0f}s. Results in results/, figures in figures/.")

    if args.tests:
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))


if __name__ == "__main__":
    main()
