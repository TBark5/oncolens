# Figure captions

All figures are 300 dpi PNGs produced by `scripts/01_data.py` (01-04) and `scripts/07_figures.py` (05-15).
Colors: benign = blue, malignant = orange; each model keeps one color and line style throughout.

| File | Caption |
|---|---|
| `01_class_balance.png` | The dataset has 357 benign and 212 malignant samples (569 total), a mild imbalance that motivates stratified splits. |
| `02_correlation_heatmap.png` | Pearson correlations among the 30 features on the training split; the radius, perimeter, and area features are almost perfectly correlated with each other. |
| `03_pca_scatter.png` | A two-component PCA of the standardized training features shows the two classes separating mainly along the first component, with a narrow region of overlap. |
| `04_feature_distributions.png` | Per-class histograms of three features on the training split: worst concave points and worst perimeter separate the classes well, mean texture much less so. |
| `05_model_comparison_cv.png` | Mean and standard deviation of ROC AUC and recall across 5 stratified folds for all four models, with tuned (nested CV) and default hyperparameters. |
| `06_roc_curves.png` | Out-of-fold ROC curves on the training split for the four tuned models, with a zoom on the top-left corner where they differ. |
| `07_precision_recall_curves.png` | Out-of-fold precision-recall curves on the training split for the four tuned models, with a zoom on the high-recall region. |
| `08_calibration_curve.png` | Reliability diagram of out-of-fold probabilities (top) and the logistic regression's prediction histogram (bottom); most predictions are near 0 or 1, so the middle bins are noisy. |
| `09_threshold_tradeoff.png` | Recall, precision, and specificity of logistic regression as the decision threshold changes, with the chosen threshold of 0.2 and the default of 0.5 marked. |
| `10_confusion_matrices.png` | Confusion matrices for logistic regression on out-of-fold training predictions and on the held-out test set, at thresholds 0.5 and 0.2. |
| `11_learning_curve.png` | Training and validation accuracy of logistic regression as the training set grows; the small, stable gap suggests low variance at the full training size. |
| `12_shap_global_importance.png` | Mean absolute SHAP value (log-odds) of the 15 most influential features for logistic regression, computed on the training split. |
| `13_shap_summary_beeswarm.png` | SHAP beeswarm on the training split: each dot is one sample, its position is the feature's push on the log-odds, and its color is the feature value. |
| `14_shap_waterfall_confident_malignant.png` | SHAP waterfall for the test patient with the highest malignancy probability, showing size-related features driving the prediction. |
| `14_shap_waterfall_confident_benign.png` | SHAP waterfall for the test patient with the lowest malignancy probability. |
| `14_shap_waterfall_false_alarm_highest_p.png` | SHAP waterfall for the benign test patient with the highest malignancy probability, pushed over the threshold mainly by a very high worst texture value. |
| `14_shap_waterfall_missed_malignant_1.png` | SHAP waterfall for the one malignant test patient the model missed at threshold 0.2 (p = 0.061), pulled toward benign mainly by low worst texture. |
| `15_error_profile.png` | Mean z-scores of the top SHAP features for correct and incorrect out-of-fold predictions; the misclassified groups mostly sit between the two class averages. |
