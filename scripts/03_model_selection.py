"""Phase 3a: compare four models, tune them, choose one, and save out-of-fold predictions."""

from __future__ import annotations

import time

import joblib
import pandas as pd

from oncolens import config
from oncolens.data import load_splits, split_features_target
from oncolens.io_utils import write_json
from oncolens.models import MODEL_NAMES
from oncolens.selection import (
    SELECTION_METRIC,
    choose_model,
    compare_default_models,
    nested_cv_tuned,
    out_of_fold_proba,
    tune_model,
)


def main() -> None:
    """Run model comparison and tuning on the training split only."""
    config.ensure_dirs()
    train, _ = load_splits()
    X, y = split_features_target(train)
    start = time.time()

    default_cv = compare_default_models(X, y)
    default_cv.round(4).to_csv(config.RESULTS_DIR / "cv_default_models.csv")
    print("Default-hyperparameter CV:\n", default_cv[["roc_auc_mean", "recall_mean", "accuracy_mean"]].round(4))

    nested = nested_cv_tuned(X, y)
    nested.round(4).to_csv(config.RESULTS_DIR / "cv_nested_tuned_models.csv")
    print("Nested CV (tuned):\n", nested[["roc_auc_mean", "recall_mean", "accuracy_mean"]].round(4))

    tuning: dict[str, dict] = {}
    oof = pd.DataFrame({"y_true": y})
    for name in MODEL_NAMES:
        search = tune_model(name, X, y)
        tuning[name] = {
            "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
            f"best_inner_cv_{SELECTION_METRIC}": round(float(search.best_score_), 4),
        }
        joblib.dump(search.best_estimator_, config.MODELS_DIR / f"{name}.joblib")
        oof[name] = out_of_fold_proba(search.best_estimator_, X, y)
    write_json(tuning, config.RESULTS_DIR / "tuning_best_params.json")
    oof.to_csv(config.RESULTS_DIR / "oof_probabilities_train.csv", index=False)

    chosen, reason = choose_model(nested)
    joblib.dump(joblib.load(config.MODELS_DIR / f"{chosen}.joblib"), config.FINAL_MODEL_FILE)
    write_json(
        {
            "chosen_model": chosen,
            "reason": reason,
            "selection_metric": f"nested-CV mean {SELECTION_METRIC}",
            "best_params": tuning[chosen]["best_params"],
        },
        config.RESULTS_DIR / "model_selection.json",
    )
    print(f"Chosen model: {chosen}. {reason}  ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
