# Resume bullets

Every number below comes from `results/` (see the README results table). Pick one version; do not mix
claims across versions without checking them.

## Short (one line)

- Built **OncoLens**, an explainable tumor-classification dashboard (scikit-learn, SHAP, Streamlit) that
  catches 41 of 42 malignant cases on a held-out test set by tuning the decision threshold for recall.

## Medium (two or three bullets)

- Built an end-to-end ML pipeline on the Wisconsin breast cancer dataset (569 samples), comparing logistic
  regression, random forest, gradient boosting and SVM with stratified 5-fold and nested cross-validation;
  selected logistic regression (nested-CV ROC AUC 0.995).
- Chose a recall-first decision threshold (0.2) from out-of-fold predictions, reducing missed malignant
  cases on the held-out test set from 3 to 1 of 42 at the cost of 3 extra false alarms.
- Shipped a Streamlit app that returns a prediction and a per-input SHAP explanation, backed by 46 pytest
  tests and a one-command, byte-for-byte reproducible pipeline.

## Technical (for ML-focused roles)

- Designed a leakage-safe evaluation: scaler inside every `Pipeline`, stratified 80/20 split with the test set
  used once, and nested CV (GridSearchCV inside each outer fold) to compare four tuned classifiers without
  selection bias; logistic regression won (ROC AUC 0.9949 ± 0.0051).
- Checked probability calibration on out-of-fold predictions (Brier 0.019, ECE 0.017) and selected the highest
  threshold meeting 0.98 out-of-fold recall (t = 0.2); held-out test recall 0.976, precision 0.911,
  ROC AUC 0.996.
- Explained predictions with exact SHAP linear attributions (verified to sum to the model output) and ran an
  error analysis showing that misclassified tumors sit between the class averages (median relative
  position 0.29 for missed cancers, 0.38 for false alarms).
- Engineered for reproducibility: typed, documented `src/` package, 46 pytest tests (including a headless
  Streamlit `AppTest` run and a check that README numbers match generated results), and a fresh-clone /
  fresh-venv verification script.

## Wording to avoid

- Do not write "diagnoses cancer", "clinical-grade", "99.6% accurate", or "outperforms doctors". The project
  has no clinical validation, and ROC AUC is not accuracy.
