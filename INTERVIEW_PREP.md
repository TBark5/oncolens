# Interview prep: 15 likely questions

Short, accurate answers. Numbers come from `results/`. If you are unsure of a number in the interview, say
"about" and name where it lives rather than guessing.

---

**1. Give me the 30-second summary.**
I built a classifier for the Wisconsin breast cancer dataset (569 tumors, 30 cell-nucleus measurements).
I compared four models with nested cross-validation, picked logistic regression, lowered the decision threshold
to 0.2 so it catches more malignant cases, explained each prediction with SHAP, and wrapped it in a Streamlit
app. On a held-out test set it caught 41 of 42 malignant tumors with 4 false alarms among 72 benign ones.

**2. Why did you choose that threshold?**
Because missing a cancer is far worse than a false alarm. A false alarm leads to another test; a miss can
delay treatment. So I set a target of at least 98% recall and picked the *highest* threshold meeting it, which
keeps false alarms as low as possible. I chose it on out-of-fold training predictions, never on the test set.
It came out at 0.20. On the training folds that moved us from 8 misses and 4 false alarms (at 0.5) to 3 misses
and 14 false alarms. The 98% target itself is a judgment call; in real life clinicians would set it from the
actual costs.

**3. Why logistic regression and not a fancier model?**
It had the highest nested-CV ROC AUC (0.9949), although the top three models are within one standard deviation
of each other. So the tie-breakers mattered: it is the simplest, its probabilities were the best calibrated,
and its SHAP values are exact and fast. With 455 training samples and features that are mostly linearly
separable, a more complex model has little to gain.

**4. What is nested cross-validation and why use it?**
GridSearchCV reports the best score over many hyperparameter settings. That score is optimistically biased
because you picked the maximum. Nested CV runs the whole grid search inside each outer training fold and
scores it on an outer fold it never saw. That estimates how well "tune, then predict" works, which is the
fair way to compare tuned models.

**5. How did you prevent data leakage?**
Three ways. The scaler sits inside a scikit-learn `Pipeline`, so in cross-validation it is fitted on the
training folds only. The 20% test split was set aside at the start and used once, at the end. And every
choice (model, hyperparameters, threshold, calibration check) used only training data, mostly out-of-fold
predictions. Even the exploratory plots used the training split.

**6. What does "calibrated" mean, and was your model calibrated?**
A model is calibrated if, among cases it gives 30%, about 30% are really malignant. I checked it with a
reliability curve on out-of-fold predictions, the Brier score (0.019) and the expected calibration error
(0.017). That was good enough, below the 0.05 limit I set in advance, so I did not add a recalibration step.
One caveat: most predictions are near 0 or 1, so the middle bins have very few samples and are noisy.

**7. Explain SHAP in plain words.**
SHAP splits a single prediction into contributions from each feature, starting from the average prediction.
For a linear model it is exact: each contribution is the coefficient times how far the feature is from its
average (on the scaled data). The bars in the waterfall add up to the model's log-odds for that patient.
I checked in a test that they reproduce the predicted probability.

**8. Which features matter most?**
By mean absolute SHAP value on the training set: worst texture, radius error, and mean concave points,
followed by worst symmetry, worst concavity and the "worst" size features. Size and shape irregularity push
toward malignant.

**9. Anything surprising in the SHAP results?**
Yes. A higher "compactness error" *lowers* the predicted risk, which makes little biological sense. The
reason is collinearity: many features are near-duplicates (radius, perimeter and area correlate above 0.9),
and a linear model can give one correlated feature a negative weight to balance the others. So SHAP explains
the model, not the biology. Stronger regularization or dropping redundant features would make the weights
easier to read.

**10. What did the error analysis show?**
Out of fold on the training data there were 3 missed cancers and 14 false alarms. Both groups sit between the
class averages. False alarms are big for benign tumors (mean worst radius 16.2 vs 13.3 for correct benign), and
missed cancers are small for malignant ones (15.8 vs 21.4). So the errors are in the overlap region, and more
features or data would probably help more than a different algorithm. There are only 3 missed cancers, though,
so I would not over-read those averages.

**11. How confident are you in the test numbers?**
Not very precise. There are 42 malignant test cases, so one tumor moves recall by 2.4 points. The CV standard
deviations (for example recall ±0.04) are a better sense of the spread. I would add bootstrap confidence
intervals next.

**12. Why is test accuracy lower at your threshold than at 0.5?**
Because the threshold trades accuracy for recall on purpose: 3 extra false alarms, 2 fewer misses. Accuracy
counts both errors equally, which is exactly the assumption I rejected.

**13. How is the project engineered?**
A small `src/` package with type hints and docstrings, one script per pipeline step, and `run_all.py` that
rebuilds everything in about a minute. Seeds are fixed, and a rerun produces byte-identical results files.
There are 46 pytest tests, including a headless run of the Streamlit app and a test that the README tables
match the generated results. A script clones the repo into a fresh virtual environment and checks the whole
thing end to end.

**14. What would you improve?**
- A bigger, newer, multi-site dataset, and external validation on a different hospital's data.
- Bootstrap confidence intervals for every reported metric.
- Choosing the threshold from explicit costs (or a decision curve) worked out with clinicians.
- Handling correlated features (grouped SHAP, or dropping redundant ones) so explanations are easier to read.
- Keeping app inputs realistic, for example by warning when a slider combination is far from the training data.

**15. Could this be used in a hospital?**
No. It is a learning project on a small, 30-year-old, single-site dataset of pre-computed features. It has had
no clinical validation, no regulatory review, and no testing on other populations. The README has a
Limitations section that says this.

---

## Numbers cheat sheet

| What | Value | File |
|---|---|---|
| Samples (train / test) | 569 (455 / 114) | `results/data_overview.json` |
| Malignant share | 37.3% | `results/data_overview.json` |
| Nested-CV ROC AUC, logistic regression | 0.9949 ± 0.0051 | `results/cv_nested_tuned_models.csv` |
| Chosen threshold | 0.2 | `results/threshold_selection.json` |
| Test recall / precision at 0.2 | 0.976 / 0.911 | `results/final_test_metrics.json` |
| Test misses / false alarms, 0.5 vs 0.2 | 3 / 1 vs 1 / 4 | `results/final_test_metrics.json` |
| Out-of-fold Brier / ECE | 0.019 / 0.017 | `results/calibration_oof.csv` |
