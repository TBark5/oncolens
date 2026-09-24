# LinkedIn post (draft)

I just finished a portfolio project called OncoLens: an explainable classifier for the Wisconsin breast
cancer dataset, with a small Streamlit app on top.

The dataset is a classic. Most models score well on it, so accuracy was not the interesting part. These
questions were:

1. How do you compare models honestly? I used nested cross-validation, so the hyperparameter search could not
   quietly inflate the scores. Four models (logistic regression, random forest, gradient boosting, SVM) all
   landed between 0.986 and 0.995 ROC AUC. The simplest one, logistic regression, came out on top.

2. What should the decision threshold be? The default cut-off of 0.5 treats a missed cancer and a false alarm
   as equally bad. They are not. I picked the threshold on training data only, aiming for at least 98% recall.
   On the untouched test set, missed malignant cases dropped from 3 to 1 out of 42, and false alarms rose
   from 1 to 4.

3. Why did the model decide that? Every prediction in the app comes with a SHAP breakdown of which
   measurements pushed it toward benign or malignant. Looking at the mistakes was the most useful part. They
   were mostly "in-between" tumors: large benign ones and small malignant ones.

What I'd do differently: get a larger, more recent, multi-site dataset, report confidence intervals, and
talk to clinicians about what the real costs of each error type are.

To be clear, this is a learning project on a 569-sample public dataset. It is not a medical tool.

The code, figures and write-up are on my GitHub (link in the first comment).

#MachineLearning #DataScience #ExplainableAI #Python
