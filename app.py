"""OncoLens Streamlit app: adjust cell measurements, get a prediction and its SHAP explanation.

Run with:  streamlit run app.py
Educational demo on a public dataset. Not a medical device.
"""

from __future__ import annotations

import altair as alt
import numpy as np
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
from oncolens.explain import sigmoid
from oncolens.io_utils import read_json
from oncolens.plots_shap import waterfall_rows
from oncolens.style import BENIGN_COLOR, MALIGNANT_COLOR, apply_style

N_TOP_SLIDERS = 8
N_SHAP_ROWS = 10
GROUP_TITLES = {"mean": "Mean values", "error": "Standard errors", "worst": "Worst (largest) values"}
PRESET_LABELS = {
    "Median": "Training median (all tumors)",
    "Benign": "Typical benign (median of benign)",
    "Malignant": "Typical malignant (median of malignant)",
}
# Darker shades of the class colors, for text on white (4.5:1 contrast).
BENIGN_INK = "#1c5cab"
MALIGNANT_INK = "#b8481c"

st.set_page_config(page_title="OncoLens", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
apply_style()

CSS = """
<style>
:root {
  --ol-ink: #1a1a18; --ol-ink-2: #45443f; --ol-muted: #6b6a64;
  --ol-line: #e4e2da; --ol-ground: #f6f5f1; --ol-card: #ffffff;
  --ol-benign: #1c5cab; --ol-malignant: #b8481c;
}
.stApp { background: var(--ol-ground); }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; font-weight: 600 !important; letter-spacing: -0.01em; color: var(--ol-ink); }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--ol-card); border-right: 1px solid var(--ol-line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] [data-testid="stSlider"] label p { font-size: 0.85rem; color: var(--ol-ink-2); }
section[data-testid="stSidebar"] [role="radiogroup"] { flex-wrap: nowrap; width: 100%; }
section[data-testid="stSidebar"] [role="radiogroup"] button { flex: 1 1 0; min-height: 40px; padding-left: 4px; padding-right: 4px; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { border-radius: 10px; border-color: var(--ol-line); }

/* Brand + labels */
.ol-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 0.4rem; }
.ol-brand-name { font-family: 'Fraunces', Georgia, serif; font-size: 1.45rem; font-weight: 600; line-height: 1.1; color: var(--ol-ink); }
.ol-brand-sub { font-size: 0.78rem; color: var(--ol-muted); }
.ol-eyebrow { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ol-muted); margin: 0.9rem 0 0.35rem; }

/* Page header */
.ol-hero { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 0.4rem; }
.ol-hero h1 { font-size: 2.3rem; margin: 0; padding: 0; line-height: 1.15; }
.ol-hero p { margin: 0.35rem 0 0; color: var(--ol-ink-2); font-size: 1rem; max-width: 640px; }
.ol-pill { display: inline-block; padding: 7px 14px; border-radius: 999px; background: #fdf3e2; color: #7a4a00; font-size: 0.78rem; font-weight: 500; white-space: nowrap; }

/* Tabs */
.stTabs [data-baseweb="tab"] p { font-size: 0.95rem; }
.stTabs [aria-selected="true"] p { font-weight: 600; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

/* Cards: bordered containers created with card() */
[class*="st-key-card-"] { background: var(--ol-card); border-radius: 18px !important; border: 1px solid var(--ol-line) !important;
  padding: 1.25rem 1.4rem !important; box-shadow: 0 1px 2px rgba(26, 26, 24, 0.04); }

/* Metrics */
[data-testid="stMetricLabel"] p { font-size: 0.7rem !important; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ol-muted); }
[data-testid="stMetricValue"] { font-family: 'Fraunces', Georgia, serif; font-weight: 600; color: var(--ol-ink); }
.st-key-ol-verdict [data-testid="stMetricValue"] { font-size: 3rem; line-height: 1.1; }

/* Probability bar */
.ol-bar { position: relative; height: 14px; border-radius: 7px; background: #eceae4; margin: 6px 0 4px; }
.ol-bar-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; }
.ol-bar-cut { position: absolute; top: -6px; bottom: -6px; width: 2px; background: var(--ol-ink); }
.ol-bar-scale { position: relative; height: 20px; font-size: 0.75rem; color: var(--ol-muted); }
.ol-bar-scale span { position: absolute; }
.ol-note { font-size: 0.88rem; line-height: 1.55; color: var(--ol-ink-2); margin: 0.4rem 0 0; }

/* Contribution list */
.ol-drivers { display: flex; flex-direction: column; gap: 8px; }
.ol-driver { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 10px 12px; border-radius: 10px; background: var(--ol-ground); font-size: 0.88rem; }
.ol-driver b { font-weight: 600; color: var(--ol-ink); }
.ol-driver .v { color: var(--ol-muted); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.ol-driver .s { font-family: 'IBM Plex Mono', monospace; font-weight: 500; white-space: nowrap; }
.ol-legend { display: flex; gap: 16px; font-size: 0.78rem; color: var(--ol-ink-2); }
.ol-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }

/* About cards */
.ol-about h4 { font-family: 'Fraunces', Georgia, serif; font-size: 1.1rem; margin: 0 0 0.3rem; }
.ol-about p { margin: 0; font-size: 0.92rem; line-height: 1.55; color: var(--ol-ink-2); }
.ol-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--ol-line); font-size: 0.78rem; color: var(--ol-muted); }
</style>
"""

LOGO = """
<svg width="36" height="36" viewBox="0 0 36 36" fill="none" aria-hidden="true">
  <circle cx="18" cy="18" r="16" stroke="#1a1a18" stroke-width="2"/>
  <circle cx="18" cy="18" r="8" stroke="#eb6834" stroke-width="2"/>
  <circle cx="18" cy="18" r="2.5" fill="#2a78d6"/>
</svg>
"""


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


@st.cache_data
def get_final_metrics() -> dict | None:
    """Held-out test metrics saved by the pipeline, if present."""
    path = config.RESULTS_DIR / "final_test_metrics.json"
    return read_json(path) if path.exists() else None


def card(parent, name: str):
    """A white, rounded card (a keyed container; the key hooks the CSS)."""
    return parent.container(key=f"card-{name}")


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


def render_sidebar(ranges: pd.DataFrame, ordered: list[str], preset_values: dict[str, pd.Series]) -> pd.DataFrame:
    """Brand, preset picker, and every feature slider; returns the current input as one row."""
    def on_preset() -> None:
        choice = st.session_state.get("preset")
        if choice:
            apply_preset(preset_values[PRESET_LABELS[choice]])

    def slider(name: str) -> None:
        r = ranges.loc[name]
        st.slider(name, float(r["min"]), float(r["max"]), step=float(r["step"]), key=name, format="%.4g")

    with st.sidebar:
        st.markdown(f'<div class="ol-brand">{LOGO}<div><div class="ol-brand-name">OncoLens</div>'
                    '<div class="ol-brand-sub">Explainable tumor classification</div></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ol-eyebrow">Start from</div>', unsafe_allow_html=True)
        st.segmented_control("Start from", list(PRESET_LABELS), key="preset", on_change=on_preset,
                             label_visibility="collapsed", width="stretch")
        st.markdown('<div class="ol-eyebrow">Most influential measurements</div>', unsafe_allow_html=True)
        for name in ordered[:N_TOP_SLIDERS]:
            slider(name)
        st.markdown('<div class="ol-eyebrow">All other measurements</div>', unsafe_allow_html=True)
        for group in FEATURE_GROUPS:
            rest = [f for f in ordered[N_TOP_SLIDERS:] if feature_group(f) == group]
            with st.expander(f"{GROUP_TITLES[group]} · {len(rest)}"):
                for name in rest:
                    slider(name)
    return pd.DataFrame([{name: st.session_state[name] for name in ranges.index}])


def probability_bar(p: float, threshold: float, color: str) -> str:
    """HTML bar for P(malignant) with a marker at the decision threshold."""
    fill = min(max(p, 0.0), 1.0) * 100
    cut = threshold * 100
    return (f'<div class="ol-bar"><div class="ol-bar-fill" style="width:{fill:.1f}%;background:{color}"></div>'
            f'<div class="ol-bar-cut" style="left:{cut:.1f}%"></div></div>'
            f'<div class="ol-bar-scale"><span style="left:0">0%</span>'
            f'<span style="left:{cut:.1f}%;transform:translateX(-50%);color:var(--ol-ink);font-weight:500">'
            f'threshold {threshold:.0%}</span><span style="right:0">100%</span></div>')


def shap_chart(values: np.ndarray, data: np.ndarray, names: list[str], base: float, threshold: float) -> alt.Chart:
    """Interactive SHAP waterfall in log-odds: top contributions plus one row for the rest."""
    rows = waterfall_rows(values, data, names, N_SHAP_ROWS)
    ends = base + np.cumsum([v for _, v in rows[::-1]])[::-1]  # accumulate bottom-up
    df = pd.DataFrame({
        "feature": [label for label, _ in rows],
        "shap": [v for _, v in rows],
        "end": ends,
    })
    df["start"] = df["end"] - df["shap"]
    df["direction"] = np.where(df["shap"] > 0, "toward malignant", "toward benign")
    df["label"] = df["shap"].map(lambda v: f"{v:+.2f}")
    df["label_x"] = np.where(df["shap"] > 0, df[["start", "end"]].max(axis=1), df[["start", "end"]].min(axis=1))
    df["align"] = np.where(df["shap"] > 0, "left", "right")
    order = df["feature"].tolist()
    final = float(ends[0])
    cut = float(np.log(threshold / (1 - threshold)))

    y = alt.Y("feature:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12,
                                                                 labelColor="#45443f", ticks=False, domain=False))
    color = alt.Color("direction:N", legend=None, scale=alt.Scale(
        domain=["toward malignant", "toward benign"], range=[MALIGNANT_COLOR, BENIGN_COLOR]))
    bars = alt.Chart(df).mark_bar(cornerRadius=3, height=18).encode(
        x=alt.X("start:Q", title="Log-odds of malignancy",
                axis=alt.Axis(grid=True, gridColor="#eceae4", domain=False, labelColor="#6b6a64",
                              titleColor="#6b6a64", titleFontWeight="normal")),
        x2="end:Q", y=y, color=color,
        tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("shap:Q", title="SHAP (log-odds)", format="+.3f"),
                 alt.Tooltip("direction:N", title="Pushes")],
    )
    text_pos = alt.Chart(df[df["shap"] > 0]).mark_text(align="left", dx=5, fontSize=11, color="#45443f").encode(
        x="label_x:Q", y=y, text="label:N")
    text_neg = alt.Chart(df[df["shap"] <= 0]).mark_text(align="right", dx=-5, fontSize=11, color="#45443f").encode(
        x="label_x:Q", y=y, text="label:N")
    refs = pd.DataFrame({
        "x": [base, final, cut],
        "what": [f"base value {base:.2f}", f"this input {final:.2f} (p = {sigmoid(final):.3f})",
                 f"decision threshold (p = {threshold:g})"],
        "dash": ["dot", "dash", "solid"],
    })
    rules = alt.Chart(refs).mark_rule(strokeWidth=1.4).encode(
        x="x:Q",
        strokeDash=alt.StrokeDash("dash:N", legend=None, scale=alt.Scale(
            domain=["dot", "dash", "solid"], range=[[2, 3], [6, 4], [1, 0]])),
        color=alt.Color("dash:N", legend=None, scale=alt.Scale(
            domain=["dot", "dash", "solid"], range=["#898781", "#45443f", "#4a3aa7"])),
        tooltip=[alt.Tooltip("what:N", title="Reference")],
    )
    return (rules + bars + text_pos + text_neg).resolve_scale(color="independent").properties(
        height=34 * len(df) + 30).configure_view(strokeWidth=0).configure(
        background="transparent", font="'IBM Plex Sans', system-ui, sans-serif")


def render_prediction(model, explainer, row: pd.DataFrame, threshold: float) -> None:
    """Verdict card, headline test metrics, and the SHAP explanation for the current input."""
    pred = predict(model, row, threshold)
    accent = MALIGNANT_COLOR if pred.is_malignant else BENIGN_COLOR
    ink = MALIGNANT_INK if pred.is_malignant else BENIGN_INK
    final = get_final_metrics()

    left, right = st.columns([2, 1], gap="medium")
    with card(left, "verdict"):
        st.markdown(f"<style>.st-key-ol-verdict [data-testid='stMetricValue'] {{ color: {ink}; }}</style>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns([3, 2])
        with c1.container(key="ol-verdict"):
            st.metric("Model output", pred.label)
        c2.metric("P(malignant)", format_probability(pred.p_malignant))
        st.markdown(probability_bar(pred.p_malignant, threshold, accent), unsafe_allow_html=True)
        st.markdown(f'<p class="ol-note">Flagged as malignant when P(malignant) ≥ {threshold:g}. The threshold was '
                    "lowered from 0.5 to catch more malignant cases, at the cost of more false alarms. "
                    f"Probability of the predicted class: <b>{format_probability(pred.confidence)}</b>.</p>",
                    unsafe_allow_html=True)
    with right:
        if final:
            m = final["metrics_at_chosen_threshold"]
            with card(st, "recall"):
                st.metric("Test recall (malignant)", f"{m['recall']:.1%}")
                st.caption(f"{m['tp']} of {m['tp'] + m['fn']} caught on held-out data")
            with card(st, "auc"):
                st.metric("Test ROC AUC", f"{m['roc_auc']:.3f}")
                st.caption(f"{final['model'].replace('_', ' ').capitalize()}, {final['n_test']} tumors")

    values, data, base = explain_one(explainer, row)
    chart_col, list_col = st.columns([2, 1], gap="medium")
    with card(chart_col, "shap"):
        st.markdown("### Why this prediction?")
        st.markdown(f'<div class="ol-legend"><span><i style="background:{MALIGNANT_COLOR}"></i>toward malignant</span>'
                    f'<span><i style="background:{BENIGN_COLOR}"></i>toward benign</span></div>',
                    unsafe_allow_html=True)
        st.altair_chart(shap_chart(values, data, list(row.columns), base, threshold), width="stretch")
        st.caption("SHAP contributions in log-odds, accumulating from the base value (average over the training "
                   "data) to this input. Hover a bar for details. Correlated inputs share credit, so one bar is "
                   "not a causal effect.")
    with card(list_col, "drivers"):
        st.markdown("### Top drivers")
        top = pd.DataFrame({"feature": row.columns, "value": data, "shap": values})
        top = top.reindex(top["shap"].abs().sort_values(ascending=False).index).head(5)
        items = "".join(
            f'<div class="ol-driver"><div><b>{r.feature}</b><br><span class="v">= {r.value:.4g}</span></div>'
            f'<span class="s" style="color:{MALIGNANT_INK if r.shap > 0 else BENIGN_INK}">{r.shap:+.2f}</span></div>'
            for r in top.itertuples())
        st.markdown(f'<div class="ol-drivers">{items}</div>', unsafe_allow_html=True)
        st.caption("Positive values push toward malignant, negative toward benign (log-odds).")


def render_performance() -> None:
    """Held-out test metrics and key figures saved by the pipeline."""
    final = get_final_metrics()
    if final is None:
        st.info("Run `python run_all.py` to generate results.")
        return
    chosen, default = final["metrics_at_chosen_threshold"], final["metrics_at_0.5"]
    st.markdown(f"### Held-out test set · {final['n_test']} tumors, {final['n_test_malignant']} malignant")
    cols = st.columns(4)
    tiles = [("Recall", "recall", "{:.1%}"), ("Precision", "precision", "{:.1%}"),
             ("Specificity", "specificity", "{:.1%}"), ("Missed malignant", "fn", "{:d}")]
    for col, (label, key, fmt) in zip(cols, tiles):
        with card(col, f"tile-{key}"):
            delta = chosen[key] - default[key]
            delta_text = (f"{delta:+d} vs 0.5" if key == "fn" else f"{delta:+.1%} vs 0.5") if delta else "same as 0.5"
            st.metric(f"{label} @ {final['threshold']:g}", fmt.format(chosen[key]), delta_text,
                      delta_color="inverse" if key == "fn" else "normal")

    rows = {f"Threshold {final['threshold']:g} (chosen)": chosen, "Threshold 0.5 (default)": default}
    table = pd.DataFrame(rows).T[["recall", "precision", "specificity", "accuracy", "roc_auc", "fn", "fp"]]
    table[["fn", "fp"]] = table[["fn", "fp"]].astype(int)
    table = table.rename(columns={"recall": "Recall", "precision": "Precision", "specificity": "Specificity",
                                  "accuracy": "Accuracy", "roc_auc": "ROC AUC", "fn": "Missed malignant (FN)",
                                  "fp": "False alarms (FP)"})
    with card(st, "table"):
        st.dataframe(table.style.format({c: "{:.3f}" for c in ("Recall", "Precision", "Specificity", "Accuracy",
                                                                "ROC AUC")}), width="stretch")
    col1, col2 = st.columns(2, gap="medium")
    for col, name, title in ((col1, "10_confusion_matrices.png", "Confusion matrices"),
                             (col2, "09_threshold_tradeoff.png", "Threshold trade-off")):
        fig_path = config.FIGURES_DIR / name
        if fig_path.exists():
            with card(col, name.split('_')[0]):
                st.markdown(f"#### {title}")
                st.image(str(fig_path), width="stretch")


def render_about() -> None:
    """Short explanation cards."""
    cards = [
        ("The model", "Logistic regression on standardized features, chosen by nested cross-validation against "
                      "random forest, gradient boosting, an RBF SVM, and XGBoost."),
        ("The explanation", "Exact SHAP values from a linear explainer, in log-odds. Bars add up from the base "
                            "value (average log-odds over the training data) to this input's log-odds."),
        ("The data", "The 30 inputs are summary statistics of cell nuclei from the public Wisconsin Diagnostic "
                     "Breast Cancer dataset: 569 samples from one historical study."),
        ("The caveats", "Sliders move independently, so some combinations are unrealistic. Correlated inputs "
                        "share credit, so a single SHAP bar is not a causal effect. See README.md for limitations."),
    ]
    for i in range(0, len(cards), 2):
        cols = st.columns(2, gap="medium")
        for col, (title, body) in zip(cols, cards[i:i + 2]):
            with card(col, f"about-{title.split()[-1]}"):
                st.markdown(f'<div class="ol-about"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def main() -> None:
    """Page layout."""
    st.markdown(CSS, unsafe_allow_html=True)
    X, y = get_training_data()
    ranges = slider_ranges(X)
    ordered = feature_order(list(X.columns))
    preset_values = presets(X, y)
    if ordered[0] not in st.session_state:
        apply_preset(preset_values[PRESET_LABELS["Median"]])
        st.session_state["preset"] = "Median"
    row = render_sidebar(ranges, ordered, preset_values)

    st.markdown('<div class="ol-hero"><div><h1>Explainable tumor classification</h1>'
                "<p>Adjust the cell-nucleus measurements in the sidebar. The model predicts benign or malignant "
                "and shows which measurements drove that call.</p></div>"
                '<span class="ol-pill">Educational demo · not for diagnosis</span></div>',
                unsafe_allow_html=True)

    model, explainer = get_model_and_explainer()
    threshold = load_threshold()
    tab_pred, tab_perf, tab_about = st.tabs(["Prediction", "Model performance", "About"])
    with tab_pred:
        render_prediction(model, explainer, row, threshold)
    with tab_perf:
        render_performance()
    with tab_about:
        render_about()
    st.markdown('<div class="ol-footer">Educational project on the public Wisconsin Diagnostic Breast Cancer '
                "dataset. Not a medical device, not clinically validated, and not for any health decision."
                "</div>", unsafe_allow_html=True)


main()
