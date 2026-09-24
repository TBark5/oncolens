**Model comparison** (training split, 455 samples, stratified 5-fold CV, mean ± std across folds):

| Model | ROC AUC (tuned, nested CV) | Recall at 0.5 | Accuracy at 0.5 | Brier | ROC AUC (default settings) |
|---|---|---|---|---|---|
| **Logistic regression** (chosen) | 0.9949 ± 0.0051 | 0.9353 ± 0.0432 | 0.9714 ± 0.0149 | 0.0232 | 0.9958 ± 0.0047 |
| Random forest | 0.9863 ± 0.0075 | 0.9412 ± 0.0372 | 0.9582 ± 0.0162 | 0.0351 | 0.9880 ± 0.0073 |
| Gradient boosting | 0.9914 ± 0.0057 | 0.9353 ± 0.0343 | 0.9648 ± 0.0162 | 0.0251 | 0.9908 ± 0.0054 |
| SVM (RBF kernel) | 0.9938 ± 0.0067 | 0.9471 ± 0.0432 | 0.9670 ± 0.0098 | 0.0217 | 0.9949 ± 0.0050 |
| XGBoost | 0.9936 ± 0.0037 | 0.9529 ± 0.0235 | 0.9758 ± 0.0082 | 0.0224 | 0.9942 ± 0.0039 |

**Held-out test set** (114 samples, 42 malignant, used once), logistic regression:

| Threshold | Recall (malignant) | Precision | Specificity | Accuracy | ROC AUC | Missed malignant (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|---|
| 0.5 (default) | 0.929 (39/42) | 0.975 | 0.986 | 0.965 | 0.996 | 3 | 1 |
| 0.2 (chosen) | 0.976 (41/42) | 0.911 | 0.944 | 0.956 | 0.996 | 1 | 4 |
