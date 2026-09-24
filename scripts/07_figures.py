"""Phase 4: build every model, SHAP, and error-analysis figure from the saved results.

EDA figures (01-04) are made by 01_data.py. This script also computes and saves the
learning-curve table, because the learning curve is only used for a figure.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from oncolens import config
from oncolens.data import load_splits, split_features_target
from oncolens.errors import label_outcomes
from oncolens.explain import build_explainer, explain_rows, global_importance
from oncolens.io_utils import read_json
from oncolens.models import make_cv
from oncolens.plots_model import (
    plot_calibration,
    plot_confusion_matrices,
    plot_cv_comparison,
    plot_error_profile,
    plot_learning_curve,
    plot_pr_overlay,
    plot_roc_overlay,
    plot_threshold_tradeoff,
)
from oncolens.plots_shap import plot_beeswarm, plot_global_importance, plot_waterfall
from oncolens.style import apply_style, save_figure

EXAMPLE_TITLES: dict[str, str] = {
    "confident_malignant": "Confidently malignant (correct)",
    "confident_benign": "Confidently benign (correct)",
    "false_alarm_highest_p": "False alarm: benign flagged as malignant",
}


def compute_learning_curve(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Accuracy of the final pipeline on growing training subsets (stratified 5-fold CV)."""
    pipeline = joblib.load(config.FINAL_MODEL_FILE)
    sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=make_cv(), train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy",
        shuffle=True, random_state=config.RANDOM_STATE,
    )
    return pd.DataFrame({
        "n_samples": sizes,
        "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1), "val_std": val_scores.std(axis=1),
    })


def confusion_panels(chosen: str, threshold: float) -> list[tuple[str, np.ndarray]]:
    """Confusion matrices for train-OOF and test, at the default and chosen thresholds."""
    oof = pd.read_csv(config.RESULTS_DIR / "oof_probabilities_train.csv")
    test = pd.read_csv(config.RESULTS_DIR / "test_probabilities.csv")
    panels = []
    for split_name, frame in (("Training, out-of-fold", oof), ("Held-out test", test)):
        for t in (0.5, threshold):
            cm = confusion_matrix(frame["y_true"], (frame[chosen] >= t).astype(int), labels=[0, 1])
            panels.append((f"{split_name}, threshold {t:g}", cm))
    return panels


def main() -> None:
    """Render figures 05-15 into figures/."""
    apply_style()
    fig_dir = config.FIGURES_DIR
    res = config.RESULTS_DIR
    chosen = read_json(res / "model_selection.json")["chosen_model"]
    threshold = read_json(res / "threshold_selection.json")["chosen_threshold"]
    oof = pd.read_csv(res / "oof_probabilities_train.csv")

    nested = pd.read_csv(res / "cv_nested_tuned_models.csv", index_col="model")
    default = pd.read_csv(res / "cv_default_models.csv", index_col="model")
    save_figure(plot_cv_comparison(nested, default), fig_dir / "05_model_comparison_cv.png")
    save_figure(plot_roc_overlay(oof), fig_dir / "06_roc_curves.png")
    save_figure(plot_pr_overlay(oof), fig_dir / "07_precision_recall_curves.png")
    calib = pd.read_csv(res / "calibration_oof.csv", index_col="model")
    save_figure(plot_calibration(oof, calib, chosen), fig_dir / "08_calibration_curve.png")
    sweep = pd.read_csv(res / "threshold_sweep_oof.csv")
    save_figure(plot_threshold_tradeoff(sweep, threshold), fig_dir / "09_threshold_tradeoff.png")
    save_figure(plot_confusion_matrices(confusion_panels(chosen, threshold)),
                fig_dir / "10_confusion_matrices.png")

    train, test = load_splits()
    X, y = split_features_target(train)
    X_test, _ = split_features_target(test)
    curve = compute_learning_curve(X, y)
    curve.round(4).to_csv(res / "learning_curve.csv", index=False)
    save_figure(plot_learning_curve(curve, "Accuracy"), fig_dir / "11_learning_curve.png")

    pe = build_explainer(joblib.load(config.FINAL_MODEL_FILE), X)
    train_expl = explain_rows(pe, X)
    save_figure(plot_global_importance(global_importance(train_expl), pe.output_unit),
                fig_dir / "12_shap_global_importance.png")
    save_figure(plot_beeswarm(train_expl), fig_dir / "13_shap_summary_beeswarm.png")

    test_expl = explain_rows(pe, X_test)
    for tag, row in read_json(res / "error_analysis.json")["example_patients"].items():
        title = EXAMPLE_TITLES.get(tag, "Missed malignant tumor (false negative)")
        p = float(pd.read_csv(res / "test_probabilities.csv")[chosen].iloc[row])
        fig = plot_waterfall(test_expl.values[row], test_expl.data[row], test_expl.feature_names,
                             float(test_expl.base_values[row]), f"{title}: test patient #{row}, p = {p:.3f}",
                             threshold=threshold)
        save_figure(fig, fig_dir / f"14_shap_waterfall_{tag}.png")

    z_profile = pd.read_csv(res / "error_profile_zscore.csv", index_col="feature")
    save_figure(plot_error_profile(z_profile), fig_dir / "15_error_profile.png")
    outcome_counts = label_outcomes(y, oof[chosen], threshold).value_counts().to_dict()
    print(f"Figures written to {fig_dir} (OOF outcomes {outcome_counts})")


if __name__ == "__main__":
    main()
