# -*- coding: utf-8 -*-
"""
SepsisDPi — 早期脓毒症风险预测网页工具
10 特征 XGBoost，训练于 MIMIC-IV (2008-2019)，时间外验 (2020-2022) AUC 0.820
"""
import os
import json
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="SepsisDPi — Sepsis Risk", page_icon="🩺", layout="wide")

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base, "model", "model.joblib"))
    with open(os.path.join(base, "model", "metadata.json")) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()
FEATURES = meta["features"]
MEDIANS = meta["medians"]
THRESHOLD = meta["threshold"]

st.title("SepsisDPi — Early Sepsis Risk Prediction")
st.caption(
    "A parsimonious 10-feature gradient-boosted model for early sepsis risk in ICU patients. "
    "Trained on MIMIC-IV (2008–2019, n = 83,661 ICU stays) with a unified Sepsis-3 outcome definition "
    "temporally separated from the predictor window. Temporal external validation (2020–2022) AUC = 0.820."
)

st.markdown("### Patient measurements (first 12 hours of ICU admission)")
st.markdown("Missing values are imputed with the training-set median (shown as the default).")

# 默认值（中位数，四舍五入到合理精度）
DEFAULTS = {
    'crea': 0.95, 'plt': 191, 'ph': 7.37, 'lact': 1.8, 'resp': 18,
    'wbc': 10.6, 'mbp': 77, 'bun': 19, 'rdw': 14.5, 'bicar': 23,
}

col1, col2 = st.columns(2)
inputs = {}
half = 5
for i, f in enumerate(FEATURES):
    col = col1 if i < half else col2
    with col:
        label = f"{meta['labels'][f]}{' (' + meta['units'][f] + ')' if meta['units'][f] else ''}"
        inputs[f] = st.number_input(label, value=float(DEFAULTS[f]),
                                    step=0.01 if f in ('ph','lact','crea','bicar','k') else 1.0,
                                    format="%.2f" if f in ('ph','lact','crea','bicar') else "%.1f")

def predict():
    x = np.array([[inputs[f] for f in FEATURES]], dtype=float)
    # 中位数插补（与训练一致）
    x = np.where(np.isnan(x), np.array([MEDIANS[f] for f in FEATURES]), x)
    proba = model.predict_proba(x)[0, 1]
    return proba

st.markdown("---")
if st.button("Predict sepsis risk", type="primary"):
    p = predict()
    st.markdown("### Result")
    if p >= THRESHOLD:
        st.error(f"**High risk of sepsis** — predicted probability **{p*100:.1f}%** (threshold {THRESHOLD*100:.0f}%)")
    elif p >= 0.3:
        st.warning(f"**Moderate risk** — predicted probability **{p*100:.1f}%** (threshold {THRESHOLD*100:.0f}%)")
    else:
        st.success(f"**Low risk** — predicted probability **{p*100:.1f}%** (threshold {THRESHOLD*100:.0f}%)")
    st.progress(min(int(p*100), 100))

st.markdown("---")
st.caption(
    "⚠️ Disclaimer: This tool is a research prototype for screening/decision support and is not a replacement "
    "for clinical judgement or diagnostic testing. It was developed and validated on the MIMIC-IV database "
    "of US ICU patients; performance in other populations has not been established. The model reports "
    "discrimination (AUC), precision–recall, and calibration together with modest positive predictive value; "
    "use the probability as one input among many clinical signals."
)
