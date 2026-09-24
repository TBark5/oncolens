"""OncoLens Streamlit app: adjust cell measurements, get a prediction and its SHAP explanation.

Run with:  streamlit run app.py
Educational demo on a public dataset. Not a medical device.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from oncolens import config
from oncolens.app_support import (
    FEATURE_GROUPS,
    explain_one,
    format_probability,
    feature_group,
    load_or_train_model,
    load_threshold,
    make_explainer,
    predict,
    presets,
    slider_ranges,
    training_features,
)
from oncolens.data import load_splits, split_features_target
from oncolens.io_utils import read_json
from oncolens.plots_shap import plot_waterfall
from oncolens.style import apply_style

N_TOP_SLIDERS = 8
GROUP_TITLES = {"mean": "Mean values", "error": "Standard errors", "worst": "Worst (largest) values"}

st.set_page_config(page_title="OncoLens", layout="wide")
apply_style()


@st.cache_resource
def get_model_and_explainer():
    """Load the final pipeline and build its SHAP explainer once per session."""
    model = load_or_train_model()
    return model, make_explainer(model, training_features())


@st.cache_data
def get_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Training-split features and labels (for slider ranges and presets)."""
    train, _ = load_splits()
    return split_features_target(train)


def feature_order(columns: list[str]) -> list[str]:
    """Features sorted by global SHAP importance when available, else in dataset order."""
    path = config.RESULTS_DIR / "shap_global_importance.csv"
    if not path.exists():
        return columns
    return pd.read_csv(path)["feature"].tolist()


def apply_preset(values: pd.Series) -> None:
    """Copy a preset's values into the slider state."""
    for name, value in values.items():
        st.session_state[name] = float(value)


def render_sliders(ranges: pd.DataFrame, ordered: list[str]) -> pd.DataFrame:
    """Draw every feature slider in the sidebar and return the current input as one row."""
    def slider(name: str) -> None:
        r = ranges.loc[name]
        st.slider(name, float(r["min"]), float(r["max"]), step=float(r["step"]), key=name, format="%.4g")

    with st.sidebar:
        st.subheader(f"Top {N_TOP_SLIDERS} most influential")
        for name in ordered[:N_TOP_SLIDERS]:
            slider(name)
        for group in FEATURE_GROUPS:
            rest = [f for f in ordered[N_TOP_SLIDERS:] if feature_group(f) == group]
            with st.expander(f"{GROUP_TITLES[group]} ({len(rest)} more)"):
                for name in rest:
                    slider(name)
    return pd.DataFrame([{name: st.session_state[name] for name in ranges.index}])


def render_prediction(model, explainer, row: pd.DataFrame, threshold: float) -> None:
    """Prediction metrics, the SHAP waterfall, and a plain-language reading of it."""
    pred = predict(model, row, threshold)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model output", pred.label)
    c2.metric("P(malignant)", format_probability(pred.p_malignant))
    c3.metric("Probability of the predicted class", format_probability(pred.confidence))
    st.progress(min(max(pred.p_malignant, 0.0), 1.0))
    st.caption(f"Flagged as malignant when P(malignant) >= {threshold:g}. The threshold was lowered from "
               "0.5 to catch more malignant cases, at the cost of more false alarms.")

    values, data, base = explain_one(explainer, row)
    fig = plot_waterfall(values, data, list(row.columns), base, "Why this prediction? (SHAP waterfall)",
                         threshold=threshold)
    st.pyplot(fig, clear_figure=True)
    top = pd.DataFrame({"feature": row.columns, "value": data, "shap": values})
    top = top.reindex(top["shap"].abs().sort_values(ascending=False).index).head(5)
    lines = [f"- **{r.feature}** = {r.value:.4g} pushes toward "
             f"{'malignant' if r.shap > 0 else 'benign'} ({r.shap:+.2f} log-odds)" for r in top.itertuples()]
    st.markdown("**Biggest contributions for this input**\n" + "\n".join(lines))


def render_performance() -> None:
    """Held-out test metrics and key figures saved by the pipeline."""
    path = config.RESULTS_DIR / "final_test_metrics.json"
    if not path.exists():
        st.info("Run `python run_all.py` to generate results.")
        return
    final = read_json(path)
    rows = {"threshold " + str(final["threshold"]): final["metrics_at_chosen_threshold"],
            "threshold 0.5": final["metrics_at_0.5"]}
    table = pd.DataFrame(rows).T[["recall", "precision", "specificity", "accuracy", "roc_auc", "fn", "fp"]]
    st.markdown(f"Held-out test set: {final['n_test']} tumors ({final['n_test_malignant']} malignant), "
                f"model = {final['model'].replace('_', ' ')}.")
    table[["fn", "fp"]] = table[["fn", "fp"]].astype(int)
    table = table.rename(columns={"roc_auc": "ROC AUC", "fn": "missed malignant (FN)", "fp": "false alarms (FP)"})
    st.dataframe(table.style.format({c: "{:.3f}" for c in ("recall", "precision", "specificity", "accuracy", "ROC AUC")}))
    col1, col2 = st.columns(2)
    for col, name in ((col1, "10_confusion_matrices.png"), (col2, "09_threshold_tradeoff.png")):
        fig_path = config.FIGURES_DIR / name
        if fig_path.exists():
            col.image(str(fig_path))


def main() -> None:
    """Page layout."""
    st.title("OncoLens: explainable tumor classification")
    st.warning("Educational demo on the public Wisconsin Diagnostic Breast Cancer dataset. "
               "Not a medical device and not for diagnosis.")
    X, y = get_training_data()
    ranges = slider_ranges(X)
    ordered = feature_order(list(X.columns))
    preset_values = presets(X, y)
    if ordered[0] not in st.session_state:
        apply_preset(preset_values["Training median (all tumors)"])

    st.sidebar.header("Cell measurements")
    choice = st.sidebar.selectbox("Start from", list(preset_values))
    if st.sidebar.button("Load these values"):
        apply_preset(preset_values[choice])
    row = render_sliders(ranges, ordered)

    model, explainer = get_model_and_explainer()
    threshold = load_threshold()
    tab_pred, tab_perf, tab_about = st.tabs(["Prediction", "Model performance", "About"])
    with tab_pred:
        render_prediction(model, explainer, row, threshold)
    with tab_perf:
        render_performance()
    with tab_about:
        st.markdown(
            "- **Model:** logistic regression on standardized features, chosen by nested cross-validation "
            "against random forest, gradient boosting, and an RBF SVM.\n"
            "- **Explanation:** exact SHAP values from a linear explainer, in log-odds. Bars add up from the "
            "average prediction (base value) to this input's prediction.\n"
            "- **Caveat:** the 30 inputs are summary statistics of cell nuclei from one historical dataset of "
            "569 samples. Correlated inputs share credit, so a single SHAP bar is not a causal effect.\n"
            "- Code, figures and limitations: see README.md."
        )


main()
