# Decisions Log

Choices made without asking, with the reason for each. Newest last.

| # | Decision | Why |
|---|----------|-----|
| 1 | Use the existing `.venv` (Python 3.14.5). | It was already in the folder; every dependency installed cleanly from wheels. |
| 2 | Top-level deps in `requirements.in`, exact pins in `requirements.txt`. | Human-readable intent plus reproducible installs. |
| 3 | Removed PyCharm's template `main.py`. | Boilerplate, unrelated to the project. |
| 4 | Layout: `src/oncolens/` package, `scripts/` one script per phase, `run_all.py` runs them in order. | Hard rule 9: small modules, package plus scripts. |
| 5 | Flipped the target so 1 = malignant (sklearn uses 0 = malignant). | Malignant is the class we want to catch, so "recall" and "positive" mean the intuitive thing. |
| 6 | Stratified 80/20 train/test split (seed 42), cached as CSV in `data/`. | The test set is held out and used exactly once, in the final evaluation. |
| 7 | Per-feature summary, correlation heatmap, PCA and histograms use the training split only. Class balance uses all 569 rows (counting labels involves no fitting). | Keeps every modelling-relevant look at the data away from the test set. |
| 8 | Colors follow a validated colorblind-safe categorical palette; benign = blue, malignant = orange everywhere. Diverging blue/red colormap with a gray midpoint for correlations. | Consistent, accessible visuals (Phase 4 requirement). |
| 9 | Package installed in editable mode (`pip install -e .`, listed in requirements.txt). | Scripts, tests and the app can all `import oncolens` without path hacks. |
| 10 | Model comparison uses nested CV (GridSearchCV inside each outer fold) as the selection criterion, plus a default-hyperparameter CV table for reference. | Selecting on GridSearchCV's own best score is optimistically biased; nested CV is the standard fix and is cheap here. |
| 11 | Selection metric = nested-CV mean ROC AUC; logistic regression preferred if within 0.005 of the best (rule fixed before seeing results). | ROC AUC is threshold-free (threshold is chosen separately); LR is easiest to explain. LR won outright, so the tie-break was not needed. |
| 12 | SVM probabilities via `CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False)`. | `SVC(probability=True)` is deprecated in scikit-learn 1.9; this is the documented replacement. |
| 13 | Calibration judged on out-of-fold probabilities (ECE, 10 equal-width bins, and Brier). Rule: recalibrate only if ECE > 0.05. | LR had ECE 0.017, so no recalibration; keeps SHAP exact and the pipeline simple. |
| 14 | Threshold = highest value on a 0.01 grid whose out-of-fold recall >= 0.98. | Missing a malignant case is costlier than a false alarm; "highest" keeps false alarms as low as possible under that constraint. Chosen on training data only. |
| 15 | Test set scored once for all four tuned models; only the pre-chosen model/threshold are the headline result. | Transparency without letting test results influence any choice. |
| 16 | SHAP via exact `LinearExplainer` (interventional, training background) in log-odds units; permutation explainer fallback for non-linear models. | Exact and fast; contributions sum to the model's log-odds (verified to 1e-15). |
| 17 | Error analysis on out-of-fold training errors (17) plus test errors (5), described by a 0 (avg benign) to 1 (avg malignant) "relative position" per feature. All report text is generated from the numbers. | More errors to study than the 5 test errors alone; avoids hand-written claims drifting from the data. |
| 18 | App sliders: the 8 most influential features (by global SHAP) up front, the other 22 in collapsible groups; ranges = training min/max; presets = class medians. | 30 sliders at once is overwhelming; ranges from training data keep inputs realistic. |
| 19 | App shows "probability of the predicted class" instead of the word "confidence", and never rounds to 0% or 100%. | Avoids implying clinical certainty. |
| 20 | Screenshots via Playwright driving the locally installed Chrome (`requirements-dev.txt`). | Headless Chrome's one-shot `--screenshot` only captures Streamlit's loading skeleton; Playwright can wait for content. No browser download needed. |
