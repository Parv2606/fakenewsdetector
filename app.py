import streamlit as st
from transformers import pipeline
import torch
from datetime import datetime

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Avoid CPU overload on Streamlit Cloud
torch.set_num_threads(1)

@st.cache_resource(show_spinner=True)
@st.cache_resource(show_spinner=True)
def load_models():
    detector = pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-fake-news",
        return_all_scores=False,
        truncation=True
    )

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",
        truncation=True
    )

    return detector, summarizer

detector, summarizer = load_models()

# --- UI HEADER ---
st.markdown("""
<div style="text-align:center;">
<h1>📰 Fake News Detector</h1>
<p style="font-size:18px;">Analyze text credibility using AI-powered NLP models.</p>
</div>
""", unsafe_allow_html=True)

# Maintain History
if "history" not in st.session_state:
    st.session_state.history = []


# --- INPUT AREA SECTION ---
st.markdown("### ✍️ Enter News Article or Statement")
text = st.text_area("Paste text here:", height=230, placeholder="Type or paste text to analyze...")

col1, col2 = st.columns(2)
use_summary = col1.checkbox("Use AI-generated summary before detection", value=False)
show_history = col2.checkbox("Show analysis history", value=False)


def analyze_text(text):
    if use_summary:
        summary = summarizer(text, max_length=120, min_length=40)[0]["summary_text"]
        processed_text = summary
    else:
        summary = None
        processed_text = text

    result = detector(processed_text)[0]
    label = result["label"]
    confidence = float(result["score"])

    return label, confidence, summary


# --- PROCESS BUTTON ---
if st.button("🚀 Analyze", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        label, confidence, summary = analyze_text(text)

        # Color-coded result box
        if label == "FAKE":
            st.markdown(f"""
            <div style="padding:18px;border-radius:10px;border-left:8px solid #ff4b4b;background:#ffecec;">
            <h3>⚠️ Fake News Detected</h3>
            <p><b>Confidence:</b> {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:18px;border-radius:10px;border-left:8px solid #32cd32;background:#eaffea;">
            <h3>✅ Likely Real News</h3>
            <p><b>Confidence:</b> {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Confidence Level")
        st.progress(confidence)

        if summary:
            st.markdown("### 📝 AI Summary")
            st.info(summary)

        # Save to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "label": label,
            "confidence": confidence,
            "text": text if len(text) <= 300 else text[:300] + "…",
            "summary": summary
        })

# --- HISTORY SECTION ---
if show_history:
    st.markdown("---")
    st.markdown("### 🗂️ Analysis History")

    if len(st.session_state.history) == 0:
        st.info("No past analyses yet.")
    else:
        for item in reversed(st.session_state.history):
            with st.expander(f"{item['timestamp']} — {item['label']} ({item['confidence']*100:.1f}%)"):
                if item["summary"]:
                    st.write("**Summary:**", item["summary"])
                st.write("**Text Snippet:**", item["text"])

        if st.button("🧹 Clear History"):
            st.session_state.history = []
            st.experimental_rerun()
