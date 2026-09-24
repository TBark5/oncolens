"""Phase 3b: check probability calibration and choose a high-recall decision threshold.

Uses only out-of-fold predictions on the training split (no test data).
"""

from __future__ import annotations

import pandas as pd

from oncolens import config
from oncolens.calibration import ECE_RECALIBRATION_LIMIT, calibration_summary, calibration_table
from oncolens.io_utils import read_json, write_json
from oncolens.metrics import classification_metrics
from oncolens.models import MODEL_NAMES
from oncolens.threshold import choose_threshold, threshold_sweep


def main() -> None:
    """Write calibration and threshold results for the chosen model."""
    oof = pd.read_csv(config.RESULTS_DIR / "oof_probabilities_train.csv")
    y = oof["y_true"].to_numpy()
    chosen = read_json(config.RESULTS_DIR / "model_selection.json")["chosen_model"]

    calib = {name: calibration_summary(y, oof[name].to_numpy()) for name in MODEL_NAMES}
    pd.DataFrame(calib).T.round(4).to_csv(config.RESULTS_DIR / "calibration_oof.csv", index_label="model")
    calibration_table(y, oof[chosen].to_numpy()).round(4).to_csv(
        config.RESULTS_DIR / "calibration_bins_chosen_model.csv", index=False
    )
    needs_recalibration = calib[chosen]["ece"] > ECE_RECALIBRATION_LIMIT

    sweep = threshold_sweep(y, oof[chosen].to_numpy())
    sweep.round(4).to_csv(config.RESULTS_DIR / "threshold_sweep_oof.csv", index=False)
    threshold = choose_threshold(sweep, config.TARGET_RECALL)

    result = {
        "model": chosen,
        "target_recall": config.TARGET_RECALL,
        "chosen_threshold": threshold,
        "calibration_oof": calib[chosen],
        "ece_limit_for_recalibration": ECE_RECALIBRATION_LIMIT,
        "recalibrated": bool(needs_recalibration),
        "oof_metrics_at_0.5": classification_metrics(y, oof[chosen], 0.5),
        "oof_metrics_at_chosen_threshold": classification_metrics(y, oof[chosen], threshold),
    }
    write_json(result, config.RESULTS_DIR / "threshold_selection.json")
    print(f"{chosen}: ECE {calib[chosen]['ece']:.4f}, Brier {calib[chosen]['brier']:.4f}, "
          f"threshold {threshold} -> OOF recall {result['oof_metrics_at_chosen_threshold']['recall']:.4f}")
    if needs_recalibration:
        raise RuntimeError("Chosen model is poorly calibrated; add a calibration step before continuing.")


if __name__ == "__main__":
    main()
