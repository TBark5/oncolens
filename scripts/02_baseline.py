"""Phase 2: a plain logistic regression baseline, evaluated with stratified 5-fold CV.

Only the training split is used here. The held-out test set is not touched.
"""

from __future__ import annotations

from sklearn.model_selection import cross_val_predict, cross_validate

from oncolens import config
from oncolens.data import load_splits, split_features_target
from oncolens.io_utils import write_json
from oncolens.metrics import CV_SCORING, classification_metrics, summarize_cv
from oncolens.models import make_cv, make_pipeline


def main() -> None:
    """Fit and cross-validate the baseline; save metrics to results/baseline_cv.json."""
    config.ensure_dirs()
    train, _ = load_splits()
    X, y = split_features_target(train)

    pipeline = make_pipeline("logistic_regression")
    cv = make_cv()
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=CV_SCORING)
    oof_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    # End-to-end smoke check: the pipeline fits on all training rows and predicts probabilities.
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((proba >= 0) & (proba <= 1)).all()

    results = {
        "model": "logistic_regression (default C=1.0, standardized features)",
        "cv_folds": config.CV_FOLDS,
        "n_train": len(X),
        "cv_fold_mean_std": summarize_cv(scores),
        "pooled_out_of_fold_at_0.5": classification_metrics(y, oof_proba, 0.5),
    }
    write_json(results, config.RESULTS_DIR / "baseline_cv.json")
    s = results["cv_fold_mean_std"]
    print(f"Baseline LR  ROC AUC {s['roc_auc_mean']:.4f} +/- {s['roc_auc_std']:.4f}  "
          f"recall {s['recall_mean']:.4f}  accuracy {s['accuracy_mean']:.4f}")


if __name__ == "__main__":
    main()
