"""Phase 7 helper: build the Markdown results tables used in README.md from results/.

The README embeds this file's output verbatim between marker comments, and
tests/test_docs.py fails if the two ever drift apart.
"""

from __future__ import annotations

import pandas as pd

from oncolens import config
from oncolens.io_utils import read_json
from oncolens.models import DISPLAY_NAMES, MODEL_NAMES

SUMMARY_FILE = config.RESULTS_DIR / "summary_tables.md"


def cv_table() -> str:
    """Nested-CV comparison of the four tuned models, plus the default-settings ROC AUC."""
    nested = pd.read_csv(config.RESULTS_DIR / "cv_nested_tuned_models.csv", index_col="model")
    default = pd.read_csv(config.RESULTS_DIR / "cv_default_models.csv", index_col="model")
    chosen = read_json(config.RESULTS_DIR / "model_selection.json")["chosen_model"]
    lines = [
        "| Model | ROC AUC (tuned, nested CV) | Recall at 0.5 | Accuracy at 0.5 | Brier | ROC AUC (default settings) |",
        "|---|---|---|---|---|---|",
    ]
    for m in MODEL_NAMES:
        r = nested.loc[m]
        name = f"**{DISPLAY_NAMES[m]}** (chosen)" if m == chosen else DISPLAY_NAMES[m]
        lines.append(
            f"| {name} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | {r['recall_mean']:.4f} ± "
            f"{r['recall_std']:.4f} | {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
            f"{r['brier_mean']:.4f} | {default.loc[m, 'roc_auc_mean']:.4f} ± {default.loc[m, 'roc_auc_std']:.4f} |"
        )
    return "\n".join(lines)


def test_table() -> str:
    """Final model on the held-out test set at the default and the chosen threshold."""
    final = read_json(config.RESULTS_DIR / "final_test_metrics.json")
    lines = [
        "| Threshold | Recall (malignant) | Precision | Specificity | Accuracy | ROC AUC | Missed malignant (FN) | False alarms (FP) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, label in (("metrics_at_0.5", "0.5 (default)"),
                       ("metrics_at_chosen_threshold", f"{final['threshold']:g} (chosen)")):
        m = final[key]
        lines.append(
            f"| {label} | {m['recall']:.3f} ({m['tp']}/{m['tp'] + m['fn']}) | {m['precision']:.3f} | "
            f"{m['specificity']:.3f} | {m['accuracy']:.3f} | {m['roc_auc']:.3f} | {m['fn']} | {m['fp']} |"
        )
    return "\n".join(lines)


def build_summary() -> str:
    """Both tables with short headings, as one Markdown block."""
    final = read_json(config.RESULTS_DIR / "final_test_metrics.json")
    return (
        "**Model comparison** (training split, 455 samples, stratified 5-fold CV, mean ± std across folds):\n\n"
        f"{cv_table()}\n\n"
        f"**Held-out test set** ({final['n_test']} samples, {final['n_test_malignant']} malignant, "
        f"used once), {DISPLAY_NAMES[final['model']].lower()}:\n\n"
        f"{test_table()}\n"
    )


def main() -> None:
    """Write results/summary_tables.md."""
    SUMMARY_FILE.write_text(build_summary(), encoding="utf-8")
    print(f"Wrote {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
