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
