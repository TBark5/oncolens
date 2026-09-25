"""OncoLens Streamlit app: adjust cell measurements, get a prediction and its SHAP explanation.

Run with:  streamlit run app.py
Educational demo on a public dataset. Not a medical device.
"""

from __future__ import annotations

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
    waterfall_rows,
)
from oncolens.data import load_splits, split_features_target
from oncolens.io_utils import read_json

N_TOP_SLIDERS = 8
N_SHAP_ROWS = 7
GROUP_TITLES = {"mean": "Mean values", "error": "Standard errors", "worst": "Worst (largest) values"}
PRESET_LABELS = {
    "Median": "Training median (all tumors)",
    "Benign": "Typical benign (median of benign)",
    "Malignant": "Typical malignant (median of malignant)",
}
# Class colors, matching the figures (benign = blue, malignant = orange).
BENIGN_COLOR = "#2a78d6"
MALIGNANT_COLOR = "#eb6834"
# Darker shades of the class colors, for text on white (4.5:1 contrast).
BENIGN_INK = "#1c5cab"
MALIGNANT_INK = "#b8481c"

st.set_page_config(page_title="OncoLens", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
  --ol-ink: #1a1a18; --ol-ink-2: #45443f; --ol-muted: #6b6a64;
  --ol-line: #e4e2da; --ol-ground: #f6f5f1; --ol-card: #ffffff; --ol-track: #eceae4;
  --ol-serif: 'Fraunces', Georgia, serif; --ol-mono: 'IBM Plex Mono', ui-monospace, monospace;
}
.stApp { background: var(--ol-ground); }
.block-container { padding-top: 2rem; padding-left: 3rem; padding-right: 3rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { font-family: var(--ol-serif) !important; font-weight: 600 !important; letter-spacing: -0.01em; color: var(--ol-ink); }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--ol-card); border-right: 1px solid var(--ol-line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] [data-testid="stSlider"] label p { font-size: 0.82rem; color: var(--ol-ink); }
section[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] p { font-family: var(--ol-mono); font-size: 0.78rem; color: #1c5cab; }
section[data-testid="stSidebar"] [role="radiogroup"] { flex-wrap: nowrap; width: 100%; gap: 4px; padding: 4px; background: #f0efea; border-radius: 10px; }
section[data-testid="stSidebar"] [role="radiogroup"] button { flex: 1 1 0; min-height: 36px; padding: 0 4px; border: 0 !important; border-radius: 7px !important; background: transparent; color: var(--ol-ink-2); }
section[data-testid="stSidebar"] [role="radiogroup"] button[aria-checked="true"] { background: var(--ol-card) !important; box-shadow: 0 1px 2px rgba(0,0,0,0.08); color: var(--ol-ink) !important; }
section[data-testid="stSidebar"] [role="radiogroup"] button[aria-checked="true"] p { font-weight: 600; color: var(--ol-ink); }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { border-radius: 10px; border-color: var(--ol-line); }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: 0.82rem; }

.ol-brand { display: flex; align-items: center; gap: 12px; margin-bottom: 0.6rem; }
.ol-brand-name { font-family: var(--ol-serif); font-size: 1.4rem; font-weight: 600; line-height: 1.1; color: var(--ol-ink); letter-spacing: -0.01em; }
.ol-brand-sub { font-size: 0.75rem; color: var(--ol-muted); }
.ol-eyebrow { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ol-muted); margin: 1rem 0 0.4rem; }

/* Top bar: pill tabs on the left, disclaimer on the right */
.ol-topbar { position: relative; height: 0; z-index: 2; }
.ol-pill { position: absolute; right: 0; top: 8px; padding: 8px 14px; border-radius: 999px; background: #fdf3e2; color: #7a4a00; font-size: 0.75rem; font-weight: 500; white-space: nowrap; }
.stTabs [role="tablist"] { gap: 4px; padding: 4px; background: #ebe9e3; border-radius: 12px; width: fit-content; border: 0; }
.stTabs [data-testid="stTab"] { height: 40px; padding: 0 18px; border-radius: 9px; background: transparent; }
.stTabs [data-testid="stTab"] p { font-size: 0.9rem; color: var(--ol-ink-2); }
.stTabs [data-testid="stTab"][aria-selected="true"] { background: var(--ol-card); box-shadow: 0 1px 2px rgba(0,0,0,0.08); }
.stTabs [data-testid="stTab"][aria-selected="true"] p { font-weight: 600; color: var(--ol-ink); }
.stTabs .react-aria-SelectionIndicator { display: none; }
.stTabs [data-testid="stTabPanel"] { padding-top: 1.4rem; }

/* Cards: keyed containers created with card() */
[class*="st-key-card-"] { background: var(--ol-card); border-radius: 18px !important; border: 1px solid var(--ol-line) !important; padding: 1.4rem 1.6rem !important; }
[data-testid="stMetricLabel"] p { font-size: 0.68rem !important; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ol-muted); }
[data-testid="stMetricValue"] { font-family: var(--ol-serif); font-weight: 600; color: var(--ol-ink); }

/* Prediction tab (plain HTML, mirrors the design mockup) */
.ol-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; margin-bottom: 20px; }
.ol-card { background: var(--ol-card); border: 1px solid var(--ol-line); border-radius: 18px; padding: 28px 32px; box-sizing: border-box; }
.ol-verdict { grid-column: span 2; display: flex; flex-direction: column; gap: 22px; }
.ol-stack { display: flex; flex-direction: column; gap: 20px; }
.ol-stat { flex: 1; display: flex; flex-direction: column; justify-content: center; gap: 6px; padding: 22px 24px; }
.ol-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--ol-muted); }
.ol-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.ol-col { display: flex; flex-direction: column; gap: 6px; }
.ol-big { font-family: var(--ol-serif); font-size: 3.5rem; font-weight: 600; line-height: 1; letter-spacing: -0.01em; }
.ol-prob { font-family: var(--ol-mono); font-size: 2.5rem; font-weight: 500; line-height: 1; color: var(--ol-ink); text-align: right; }
.ol-stat-value { font-family: var(--ol-serif); font-size: 2.1rem; font-weight: 600; line-height: 1.1; color: var(--ol-ink); }
.ol-stat-note { font-size: 0.82rem; color: var(--ol-ink-2); }
.ol-bar { position: relative; height: 14px; border-radius: 7px; background: var(--ol-track); }
.ol-bar-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; }
.ol-bar-cut { position: absolute; top: -6px; bottom: -6px; width: 2px; background: var(--ol-ink); }
.ol-bar-scale { position: relative; height: 18px; margin-top: 8px; font-size: 0.75rem; color: var(--ol-muted); }
.ol-bar-scale span { position: absolute; white-space: nowrap; }
.ol-note { margin: 0; font-size: 0.88rem; line-height: 1.55; color: var(--ol-ink-2); }
.ol-shap { display: flex; flex-direction: column; gap: 20px; }
.ol-shap-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; }
.ol-shap-head h2 { margin: 0; padding: 0; font-size: 1.5rem; }
.ol-legend { display: flex; gap: 18px; font-size: 0.75rem; color: var(--ol-ink-2); }
.ol-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
.ol-bars { display: flex; flex-direction: column; gap: 10px; }
.ol-bar-row { display: grid; grid-template-columns: 260px minmax(0, 1fr) 64px; gap: 16px; align-items: center; font-size: 0.82rem; }
.ol-bar-row > span:first-child { text-align: right; color: var(--ol-ink-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ol-track { position: relative; height: 22px; background: var(--ol-ground); border-radius: 4px; }
.ol-track > div { position: absolute; left: 0; top: 3px; bottom: 3px; border-radius: 3px; }
.ol-val { font-family: var(--ol-mono); color: var(--ol-ink); }
.ol-foot { margin: 0; font-size: 0.82rem; color: var(--ol-muted); }
@media (max-width: 900px) {
  .ol-grid { grid-template-columns: minmax(0, 1fr); }
  .ol-verdict { grid-column: auto; }
  .ol-bar-row { grid-template-columns: 140px minmax(0, 1fr) 52px; }
  .ol-big { font-size: 2.6rem; } .ol-prob { font-size: 1.8rem; }
  .ol-pill { position: static; display: inline-block; margin-bottom: 12px; }
  .ol-topbar { height: auto; }
}

.ol-about h4 { font-size: 1.1rem; margin: 0 0 0.3rem; }
.ol-about p { margin: 0; font-size: 0.92rem; line-height: 1.55; color: var(--ol-ink-2); }
.ol-footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--ol-line); font-size: 0.75rem; color: var(--ol-muted); }
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
        for group in FEATURE_GROUPS:
            rest = [f for f in ordered[N_TOP_SLIDERS:] if feature_group(f) == group]
            with st.expander(f"{GROUP_TITLES[group]} · {len(rest)} more"):
                for name in rest:
                    slider(name)
    return pd.DataFrame([{name: st.session_state[name] for name in ranges.index}])


def probability_bar(p: float, threshold: float, color: str) -> str:
    """HTML bar for P(malignant) with a marker at the decision threshold."""
    fill = min(max(p, 0.0), 1.0) * 100
    cut = threshold * 100
    return (f'<div><div class="ol-bar"><div class="ol-bar-fill" style="width:{fill:.1f}%;background:{color}"></div>'
            f'<div class="ol-bar-cut" style="left:{cut:.1f}%"></div></div>'
            f'<div class="ol-bar-scale"><span style="left:0">0%</span>'
            f'<span style="left:{cut:.1f}%;transform:translateX(-50%);color:var(--ol-ink);font-weight:500">'
            f'threshold {threshold:.0%}</span><span style="right:0">100%</span></div></div>')


def shap_bars(values: np.ndarray, data: np.ndarray, names: list[str]) -> str:
    """HTML rows: one bar per top contribution, length = |SHAP|, color = direction."""
    rows = waterfall_rows(values, data, names, N_SHAP_ROWS)
    scale = max(abs(v) for _, v in rows) or 1.0
    html = []
    for label, v in rows:
        color = MALIGNANT_COLOR if v > 0 else BENIGN_COLOR
        direction = "toward malignant" if v > 0 else "toward benign"
        html.append(f'<div class="ol-bar-row" title="{label}: {v:+.3f} log-odds, {direction}"><span>{label}</span>'
                    f'<div class="ol-track"><div style="width:{abs(v) / scale * 100:.1f}%;background:{color}"></div></div>'
                    f'<span class="ol-val">{v:+.2f}</span></div>')
    return "".join(html)


def render_prediction(model, explainer, row: pd.DataFrame, threshold: float) -> None:
    """Verdict card, headline test metrics, and the SHAP explanation for the current input."""
    pred = predict(model, row, threshold)
    accent = MALIGNANT_COLOR if pred.is_malignant else BENIGN_COLOR
    ink = MALIGNANT_INK if pred.is_malignant else BENIGN_INK
    final = get_final_metrics()

    stats = ""
    if final:
        m = final["metrics_at_chosen_threshold"]
        stats = (
            '<div class="ol-stack">'
            '<div class="ol-card ol-stat"><span class="ol-label">Test recall (malignant)</span>'
            f'<span class="ol-stat-value">{m["recall"]:.1%}</span>'
            f'<span class="ol-stat-note">{m["tp"]} of {m["tp"] + m["fn"]} caught on held-out data</span></div>'
            '<div class="ol-card ol-stat"><span class="ol-label">Test ROC AUC</span>'
            f'<span class="ol-stat-value">{m["roc_auc"]:.3f}</span>'
            f'<span class="ol-stat-note">{final["model"].replace("_", " ").capitalize()}, '
            f'{final["n_test"]} tumors</span></div></div>'
        )
    st.markdown(
        '<div class="ol-grid"><div class="ol-card ol-verdict"><div class="ol-row">'
        f'<div class="ol-col"><span class="ol-label">Model output</span>'
        f'<span class="ol-big" data-verdict="{pred.label}" style="color:{ink}">{pred.label}</span></div>'
        f'<div class="ol-col" style="align-items:flex-end"><span class="ol-label">P(malignant)</span>'
        f'<span class="ol-prob">{format_probability(pred.p_malignant)}</span></div></div>'
        f'{probability_bar(pred.p_malignant, threshold, accent)}'
        f'<p class="ol-note">Flagged as malignant when P(malignant) ≥ {threshold:g}. The threshold was lowered from '
        "0.5 to catch more malignant cases, at the cost of more false alarms. Probability of the predicted class: "
        f"<b>{format_probability(pred.confidence)}</b>.</p></div>{stats}</div>",
        unsafe_allow_html=True,
    )

    values, data, base = explain_one(explainer, row)
    final_logit = base + float(values.sum())
    st.markdown(
        '<div class="ol-card ol-shap"><div class="ol-shap-head"><h2>Why this prediction?</h2>'
        f'<div class="ol-legend"><span><i style="background:{MALIGNANT_COLOR}"></i>toward malignant</span>'
        f'<span><i style="background:{BENIGN_COLOR}"></i>toward benign</span></div></div>'
        f'<div class="ol-bars">{shap_bars(values, data, list(row.columns))}</div>'
        f'<p class="ol-foot">SHAP contributions in log-odds, from the base value {base:.2f} (average over the '
        f"training data) to this input's {final_logit:.2f}. Hover a row for details. Correlated inputs share "
        "credit, so one bar is not a causal effect.</p></div>",
        unsafe_allow_html=True,
    )


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

    st.markdown('<div class="ol-topbar"><span class="ol-pill">Educational demo · not for diagnosis</span>'
                '</div>', unsafe_allow_html=True)

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
