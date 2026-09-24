# Morning report

Good morning. Everything in the task list is done and committed. Details below.

## What works (verified tonight)

- **All 7 phases** are complete, each committed separately (`git log --oneline`).
- **`python run_all.py --clean --tests`** rebuilds data, results, 18 figures and models from scratch in about
  1 minute, then runs the tests. Re-running gives **byte-identical** results files.
- **Fresh-environment check passed:** `scripts/check_fresh_env.sh` cloned the repo, created a new venv, installed
  `requirements.txt`, ran the clean pipeline plus the tests, and confirmed the results match the committed ones.
- **Tests:** 48 pytest tests pass. They include a headless run of the Streamlit app and a check that the README
  tables equal `results/summary_tables.md`.
- **App:** `streamlit run app.py` launches. I checked it with Streamlit's `AppTest`, a live server health check,
  and real browser screenshots (`docs/screenshots/`, `docs/demo.gif`).
- **Stretch goals:** XGBoost added to the comparison; GitHub Actions workflow; Dockerfile; demo-GIF script;
  independent self-review with fixes applied.

## What doesn't (or is unverified)

- **Dockerfile was never built.** Docker is not installed on this machine. It follows standard practice, but
  run `docker build -t oncolens .` once before you mention it.
- **GitHub Actions CI passes** on Linux with Python 3.12 and 3.13 (tests plus a clean pipeline rebuild):
  https://github.com/TBark5/oncolens/actions
- Developed and fully verified locally on Windows / Python 3.14.5. CI confirms the tests and pipeline also run on Linux
  with Python 3.12 and 3.13; the last digits of some results may differ slightly across platforms.
- Nothing is BLOCKED and there are no known bugs.

## Final numbers (from `results/`)

| What | Value |
|---|---|
| Data | 569 samples (212 malignant); 455 train / 114 test (42 malignant in test) |
| Chosen model | Logistic regression, `C = 1.0` |
| Nested-CV ROC AUC (train) | 0.9949 ± 0.0051 (SVM 0.9938, XGBoost 0.9936, GB 0.9914, RF 0.9863) |
| Decision threshold | 0.2 (highest threshold with out-of-fold recall >= 0.98) |
| **Test at 0.2** | recall 0.976 (41/42), precision 0.911, specificity 0.944, accuracy 0.956, ROC AUC 0.996, 1 FN, 4 FP |
| Test at 0.5 (for comparison) | recall 0.929 (39/42), precision 0.975, 3 FN, 1 FP |
| Calibration (out-of-fold) | Brier 0.019, ECE 0.017 (no recalibration needed) |
| Top SHAP features | worst texture, radius error, mean concave points |

## Exact commands

```bash
cd C:\Users\jhy61\PycharmProjects\Oncolens
.venv\Scripts\activate
python run_all.py --clean --tests      # reproduce everything + run tests (~1 min)
streamlit run app.py                   # http://localhost:8501
pytest -q                              # tests only
bash scripts/check_fresh_env.sh        # full from-scratch check in a new venv (~5 min, needs internet for pip)
python scripts/capture_screenshots.py --gif   # refresh screenshots and GIF (needs requirements-dev.txt)
```

Repository: https://github.com/TBark5/oncolens (public).

## 5 things to understand before presenting this

1. **Why the threshold is 0.2, not 0.5.** A missed cancer (false negative) costs far more than a false alarm.
   The threshold was chosen on out-of-fold training predictions only, as the highest value reaching 98%
   recall. On the test set it turned 3 misses and 1 false alarm into 1 miss and 4 false alarms. Accuracy dropped
   slightly, and that is expected. The 98% target is your judgment call, not a clinical standard. Say so.

2. **Why nested CV, and what "no leakage" means here.** The scaler sits inside the pipeline, so it is re-fitted on
   every training fold. GridSearchCV's best score is optimistically biased, so tuned models were compared with
   nested CV. The test set was touched exactly once, after every decision was made.

3. **The models are basically tied.** The top four ROC AUCs are within one standard deviation of each other.
   Logistic regression was chosen because it scored highest *and* is the simplest, is about as well calibrated
   as the others, and gives exact SHAP values. Don't claim it "beat" XGBoost.

4. **SHAP explains the model, not biology.** The values are exact log-odds contributions for the linear model.
   Because radius, perimeter and area are nearly the same feature, the model spreads weight across them
   oddly. For example, a higher "compactness error" *lowers* predicted risk. Be ready to explain collinearity.

5. **Limits.** 569 samples from one institution in the early 1990s, pre-extracted features, and 42 malignant
   test cases (one tumor = 2.4 points of recall). This is an educational project, not a diagnostic tool. Never
   say "diagnoses cancer" or "99.6% accurate" (0.996 is ROC AUC, not accuracy).

Useful files: `INTERVIEW_PREP.md` (15 Q&As), `RESUME_BULLETS.md`, `LINKEDIN_POST.md`, `DECISIONS.md`
(every judgment call and why), `PROGRESS.md` (handoff status).
