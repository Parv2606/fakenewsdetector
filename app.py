import streamlit as st
from transformers import pipeline
import torch
import json
from io import StringIO
import csv
from datetime import datetime

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Prevent CPU overload on Streamlit Cloud
torch.set_num_threads(1)

def get_device():
    return 0 if torch.cuda.is_available() else -1

@st.cache_resource(show_spinner=True)
def load_models():
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=get_device()
    )

    detector = pipeline(
        "text-classification",
        model="taltech-cs/fake-news-detection-bert",
        return_all_scores=True,
        truncation=True,
        device=get_device()
    )
    return summarizer, detector


def chunk_text(text, size=180):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])


def detect_fake_news(text, detector, chunk_size=180):
    scores = {"FAKE": 0.0, "REAL": 0.0}
    chunks = 0

    for chunk in chunk_text(text, chunk_size):
        result = detector(chunk)[0]
        for item in result:
            lbl = item["label"].upper()
            scr = float(item["score"])
            if "FAKE" in lbl:
                scores["FAKE"] += scr
            if "REAL" in lbl or "TRUE" in lbl:
                scores["REAL"] += scr
        chunks += 1

    label = "FAKE" if scores["FAKE"] > scores["REAL"] else "REAL"
    confidence = max(scores["FAKE"], scores["REAL"]) / (scores["FAKE"] + scores["REAL"] + 1e-8)
    return label, confidence, scores, chunks


summarizer, detector = load_models()

st.title("📰 Fake News Detector")
st.write("Analyze news articles for credibility & get a clean summary.")

if "history" not in st.session_state:
    st.session_state["history"] = []

text = st.text_area("Paste the article here:", height=220)

use_summary = st.checkbox("Detect using summary instead of full text", value=False)
chunk_size = st.slider("Words per chunk for detection", 120, 240, 180)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        summary = summarizer(text, max_length=120, min_length=40)[0]["summary_text"]
        input_text = summary if use_summary else text

        label, conf, scores, chunks = detect_fake_news(input_text, detector, chunk_size)

        st.subheader("🧾 Summary")
        st.write(summary)

        st.subheader("🔍 Credibility Result")
        if label == "FAKE":
            st.error(f"⚠️ The article appears FAKE — Confidence: {conf*100:.1f}% (Chunks analyzed: {chunks})")
        else:
            st.success(f"✅ The article appears REAL — Confidence: {conf*100:.1f}% (Chunks analyzed: {chunks})")

        st.write({"FAKE Score": scores["FAKE"], "REAL Score": scores["REAL"]})

        timestamp = datetime.utcnow().isoformat()
        result = {
            "timestamp": timestamp,
            "label": label,
            "confidence": conf,
            "summary": summary,
            "snippet": text[:200] + "…" if len(text) > 200 else text,
        }
        st.session_state["history"].append(result)

        st.download_button("⬇️ Download Result (JSON)", json.dumps(result, indent=2), file_name="result.json")

st.divider()
st.subheader("History")

if st.session_state["history"]:
    st.write(st.session_state["history"])
    if st.button("Clear history"):
        st.session_state["history"] = []
else:
    st.info("No history yet.")
