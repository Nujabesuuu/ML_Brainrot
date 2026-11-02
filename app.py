"""Streamlit demo: upload an image, get the brainrot probability.

Run: streamlit run app.py
"""
import json

import streamlit as st
from PIL import Image
from tensorflow import keras

from src.config import METRICS_PATH, MODEL_PATH
from src.model import pil_to_array

st.set_page_config(page_title="Brainrot Detector", page_icon="🦈", layout="centered")

st.markdown("""
<style>
.verdict {
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
    border: 2px solid var(--c);
    background: color-mix(in srgb, var(--c) 12%, transparent);
}
.verdict .label { font-size: 1.6rem; font-weight: 800; color: var(--c); letter-spacing: .02em; }
.verdict .prob { font-size: .95rem; opacity: .8; }
.brainrot { --c: #e0457b; }
.normal { --c: #1f9e89; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


model = load_model()
metrics = load_metrics()
default_threshold = metrics["threshold"] if metrics else 0.5

# ---------- sidebar: what the model is and how good it is ----------
with st.sidebar:
    st.header("About the model")
    st.write("MobileNetV2 pretrained on ImageNet, fine-tuned on ~7 000 images: "
             "Italian brainrot characters vs. ordinary photos and ordinary AI-generated images.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, round(default_threshold, 2), 0.01,
                          help="Chosen on the validation set. Higher = fewer false alarms, more misses.")
    if metrics:
        test = metrics["test"]["fine-tuned MobileNetV2"]
        st.subheader("Held-out test set")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
        c2.metric("Balanced acc.", f"{test['bal_acc']:.3f}")
        errors = metrics["test_error_by_source"]["fine-tuned MobileNetV2"]
        st.caption("Error rate by image type")
        for name, value in errors.items():
            st.write(f"{name.replace('error on ', '')}: **{value:.1%}**")
    st.info("Limits: trained on one meme style (2025 Italian brainrot). "
            "Other memes, drawings or unusual AI art can be misclassified.")

# ---------- main page ----------
st.title("🦈 Brainrot Detector")
st.write("Upload a picture and the network will tell whether it looks like an **Italian brainrot** character.")

files = st.file_uploader("Images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                         label_visibility="collapsed")

for file in files or []:
    img = Image.open(file)
    prob = float(model.predict(pil_to_array(img), verbose=0)[0, 0])
    is_brainrot = prob >= threshold

    left, right = st.columns([1, 1.2], vertical_alignment="center")
    left.image(img, width="stretch")
    with right:
        css, label = ("brainrot", "BRAINROT") if is_brainrot else ("normal", "NOT BRAINROT")
        st.markdown(
            f'<div class="verdict {css}"><div class="label">{label}</div>'
            f'<div class="prob">brainrot probability {prob:.1%} · threshold {threshold:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress(prob)
        st.caption(file.name)
    st.divider()

if not files:
    st.caption("Tip: try a Tralalero Tralala screenshot, a holiday photo and a random AI picture.")

