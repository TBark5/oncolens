"""Phase 3c: the single, final evaluation on the held-out test set.

All decisions (model, hyperparameters, threshold) were made earlier using only the
training split. This script applies them to the test set once and records the result.
The other three tuned models are scored too, for context only; they are not used to
change any decision.
"""

from __future__ import annotations

import joblib
import pandas as pd

from oncolens import config
from oncolens.data import load_splits, split_features_target
from oncolens.io_utils import read_json, write_json
from oncolens.metrics import classification_metrics
from oncolens.models import MODEL_NAMES


def main() -> None:
    """Score every tuned model on the test set and save metrics and predictions."""
    _, test = load_splits()
    X_test, y_test = split_features_target(test)
    selection = read_json(config.RESULTS_DIR / "model_selection.json")
    threshold = read_json(config.RESULTS_DIR / "threshold_selection.json")["chosen_threshold"]
    chosen = selection["chosen_model"]

    preds = pd.DataFrame({"y_true": y_test})
    rows = []
    for name in MODEL_NAMES:
        model = joblib.load(config.MODELS_DIR / f"{name}.joblib")
        preds[name] = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, **classification_metrics(y_test, preds[name], 0.5)})
    preds.to_csv(config.RESULTS_DIR / "test_probabilities.csv", index=False)
    pd.DataFrame(rows).set_index("model").round(4).to_csv(config.RESULTS_DIR / "test_all_models_at_0.5.csv")

    final = {
        "model": chosen,
        "n_test": len(y_test),
        "n_test_malignant": int(y_test.sum()),
        "threshold": threshold,
        "metrics_at_chosen_threshold": classification_metrics(y_test, preds[chosen], threshold),
        "metrics_at_0.5": classification_metrics(y_test, preds[chosen], 0.5),
    }
    write_json(final, config.RESULTS_DIR / "final_test_metrics.json")
    m = final["metrics_at_chosen_threshold"]
    print(f"TEST ({chosen}, t={threshold}): recall {m['recall']:.4f} precision {m['precision']:.4f} "
          f"accuracy {m['accuracy']:.4f} ROC AUC {m['roc_auc']:.4f}  FN={m['fn']} FP={m['fp']}")


if __name__ == "__main__":
    main()
