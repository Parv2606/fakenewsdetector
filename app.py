import streamlit as st
from transformers import pipeline
import torch
import json
from io import StringIO
import csv
from datetime import datetime

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# ---------- Helpers ----------
def get_device():
    # Use GPU if available; otherwise CPU
    return 0 if torch.cuda.is_available() else -1

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        # Summarizer (fast + reliable)
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=get_device()
        )

        # Stronger fake-news model (full BERT)
        detector = pipeline(
            "text-classification",
            model="taltech-cs/fake-news-detection-bert",
            return_all_scores=True,
            truncation=True,
            device=get_device()
        )
        return summarizer, detector
    except Exception as e:
        st.error("⚠️ Model loading failed. Check your internet or try again.")
        st.write(e)
        return None, None

def word_chunks(text: str, max_words: int = 180):
    """
    Split long text into ~max_words chunks.
    Keeps punctuation reasonably intact without extra libraries.
    """
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def analyze_article_in_chunks(text: str, detector, max_words_per_chunk: int = 180):
    """
    Run the detector on multiple chunks and aggregate scores.
    Returns (label, confidence, {"FAKE": total, "REAL": total}, chunk_count)
    """
    scores = {"FAKE": 0.0, "REAL": 0.0}
    chunk_count = 0

    for chunk in word_chunks(text, max_words=max_words_per_chunk):
        # Each call returns list with one element (the sample), which is list of scores per label
        result = detector(chunk)[0]
        # Normalize labels and accumulate
        for item in result:
            label = str(item["label"]).upper()
            score = float(item["score"])
            if "FAKE" in label:
                scores["FAKE"] += score
            elif "REAL" in label or "TRUE" in label:
                scores["REAL"] += score
        chunk_count += 1

    if scores["FAKE"] >= scores["REAL"]:
        final_label = "FAKE"
        confidence = scores["FAKE"] / (scores["FAKE"] + scores["REAL"] + 1e-9)
    else:
        final_label = "REAL"
        confidence = scores["REAL"] / (scores["FAKE"] + scores["REAL"] + 1e-9)

    return final_label, confidence, scores, chunk_count


# ---------- Load models ----------
summarizer, detector = load_models()

# ---------- UI ----------
st.title("📰 Fake News Detector for Students")
st.markdown("Analyze articles, assess credibility, and summarize them to prevent misinformation.")

# Initialize history
if "history" not in st.session_state:
    st.session_state["history"] = []

article = st.text_area("Paste a news article or paragraph:", height=220)

# Controls
colA, colB = st.columns(2)
with colA:
    use_summary_for_detection = st.checkbox("Detect on summary instead of full article", value=False)
with colB:
    chunk_words = st.slider("Words per chunk (for full article detection)", 120, 240, 180, 10)

if st.button("Analyze Article"):
    if not article.strip():
        st.warning("Please enter some text.")
    elif summarizer is None or detector is None:
        st.error("Model not loaded. Refresh and try again.")
    else:
        with st.spinner("Analyzing..."):
            # Summarize for display
            summary = summarizer(article, max_length=120, min_length=40, do_sample=False)[0]['summary_text']

            # Choose text for detection: summary or full article
            detection_text = summary if use_summary_for_detection else article

            # Chunked detection
            label, score, agg_scores, n_chunks = analyze_article_in_chunks(
                detection_text, detector, max_words_per_chunk=chunk_words
            )

        st.subheader("🧾 Summary")
        st.write(summary)

        st.subheader("🔍 Credibility Check")
        if label == "FAKE":
            st.error(f"⚠️ Appears **FAKE** with confidence {score*100:.1f}%  •  (evaluated over {n_chunks} chunk(s))")
        else:
            st.success(f"✅ Appears **REAL** with confidence {score*100:.1f}%  •  (evaluated over {n_chunks} chunk(s))")

        # Show both aggregated probabilities
        total = max(agg_scores["FAKE"] + agg_scores["REAL"], 1e-9)
        st.markdown("**Aggregated class scores (sum across chunks):**")
        st.write({
            "FAKE (sum)": f"{agg_scores['FAKE']:.4f}",
            "REAL (sum)": f"{agg_scores['REAL']:.4f}",
            "FAKE (%)": f"{(agg_scores['FAKE']/total)*100:.1f}%",
            "REAL (%)": f"{(agg_scores['REAL']/total)*100:.1f}%"
        })

        # Prepare result payload
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        snippet = (article[:200] + "…") if len(article) > 200 else article
        result_payload = {
            "timestamp": timestamp,
            "prediction": label,
            "confidence": round(float(score), 4),
            "prob_fake_sum": round(float(agg_scores["FAKE"]), 4),
            "prob_real_sum": round(float(agg_scores["REAL"]), 4),
            "chunks_evaluated": n_chunks,
            "used_summary_for_detection": use_summary_for_detection,
            "summary": summary,
            "text_snippet": snippet,
        }

        # Save to history
        st.session_state["history"].append(result_payload)

        # Download current result (JSON)
        st.download_button(
            label="⬇️ Download this result (JSON)",
            data=json.dumps(result_payload, ensure_ascii=False, indent=2),
            file_name=f"fake_news_result_{timestamp}.json",
            mime="application/json",
        )

# History section
st.divider()
st.subheader("🗂️ Analysis History")

if len(st.session_state["history"]) == 0:
    st.info("No analyses yet. Run an analysis to build history.")
else:
    st.write([
        {
            "time": h["timestamp"],
            "prediction": h["prediction"],
            "conf%": f"{h['confidence']*100:.1f}",
            "fake_sum": f"{h['prob_fake_sum']:.3f}",
            "real_sum": f"{h['prob_real_sum']:.3f}",
            "chunks": h["chunks_evaluated"],
            "summary_det?": h["used_summary_for_detection"],
            "snippet": h["text_snippet"],
        }
        for h in st.session_state["history"]
    ])

    col1, col2, col3 = st.columns(3)

    # Download history as JSON
    with col1:
        st.download_button(
            label="⬇️ Download history (JSON)",
            data=json.dumps(st.session_state["history"], ensure_ascii=False, indent=2),
            file_name="fake_news_history.json",
            mime="application/json",
        )

    # Download history as CSV
    with col2:
        csv_buffer = StringIO()
        writer = csv.DictWriter(
            csv_buffer,
            fieldnames=[
                "timestamp",
                "prediction",
                "confidence",
                "prob_fake_sum",
                "prob_real_sum",
                "chunks_evaluated",
                "used_summary_for_detection",
                "summary",
                "text_snippet",
            ],
        )
        writer.writeheader()
        for row in st.session_state["history"]:
            writer.writerow(row)
        st.download_button(
            label="⬇️ Download history (CSV)",
            data=csv_buffer.getvalue(),
            file_name="fake_news_history.csv",
            mime="text/csv",
        )

    # Clear history
    with col3:
        if st.button("🧹 Clear history"):
            st.session_state["history"] = []
            st.success("History cleared.")
