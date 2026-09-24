# PROGRESS (handoff file)

## Done
- Phase 0: structure, venv (Python 3.14.5), pinned requirements, git, docs skeleton.
- Phase 1: data load/validate/cache (`data/`), stratified 80/20 split, summary table, 4 EDA figures.
- Phase 2: logistic regression baseline (`results/baseline_cv.json`).
- Phase 3: default CV + nested-CV comparison of 4 models, GridSearchCV tuning, calibration check,
  recall-first threshold (0.20, from out-of-fold train predictions), one-time test evaluation,
  SHAP (exact LinearExplainer), error analysis (`results/error_analysis.md`).

## In progress
- Phase 4: model/SHAP/error figures + figures/CAPTIONS.md.

## Next
- Phase 5 Streamlit app, Phase 6 tests + run_all.py + fresh-venv check, Phase 7 docs.

## Known bugs
- None.

## How to run
```bash
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/01_data.py   # ... through scripts/06_explain_errors.py, in order
```

## Key files
- `src/oncolens/` package (config, data, models, metrics, selection, calibration, threshold, explain, errors, plots_*).
- `scripts/0N_*.py` one script per pipeline step.
- `results/` all metrics (CSV/JSON); `models/` fitted pipelines (git-ignored, regenerated).
