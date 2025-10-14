import streamlit as st
from transformers import pipeline
import json
from io import StringIO
import csv
from datetime import datetime

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

@st.cache_resource
def load_models():
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        detector = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        return summarizer, detector
    except Exception as e:
        st.error("⚠️ Model loading failed. Please try again later.")
        st.write(e)
        return None, None

summarizer, detector = load_models()

st.title("📰 Fake News Detector for Students")
st.markdown("Analyze articles, assess credibility, and summarize them to prevent misinformation.")

# Initialize session state (non-ML)
if "history" not in st.session_state:
	st.session_state["history"] = []
if "analysis_count" not in st.session_state:
	st.session_state["analysis_count"] = 0
if "last_submit_ts" not in st.session_state:
	st.session_state["last_submit_ts"] = None
if "last_input_hash" not in st.session_state:
	st.session_state["last_input_hash"] = None
if "cooldown_seconds" not in st.session_state:
	st.session_state["cooldown_seconds"] = 3

# Sidebar: app info and status (non-ML)
with st.sidebar:
	st.header("About")
	st.write("This app analyzes news articles for credibility and provides a concise summary.")
	status_ok = summarizer is not None and detector is not None
	if status_ok:
		st.success("Models loaded")
	else:
		st.warning("Models unavailable")
	st.markdown("---")
	st.subheader("Session Stats")
	st.metric("Analyses this session", st.session_state["analysis_count"])
	if len(st.session_state["history"]) > 0:
		st.caption(f"Last result at {st.session_state['history'][-1]['timestamp']}")
	st.markdown("---")
	st.subheader("Settings")
	max_len = st.slider("Max input length (chars)", min_value=500, max_value=10000, value=5000, step=500)
	st.caption(f"Cooldown between analyses: {st.session_state['cooldown_seconds']}s")

# Main input with live counter
article = st.text_area("Paste a news article or paragraph:", key="article_input", height=200)
st.caption(f"Characters: {len(article)} / {max_len}")
if len(article) > max_len:
	st.warning("Input exceeds the maximum length. Please shorten the text.")

col_run, col_clear = st.columns([3,1])
run_clicked = col_run.button("Analyze Article")
clear_clicked = col_clear.button("Clear Input")

if clear_clicked:
	st.session_state["article_input"] = ""
	st.info("Input cleared.")

if run_clicked:
    if article.strip() == "":
        st.warning("Please enter some text.")
	elif len(article) > max_len:
		st.warning("Text too long. Please shorten and try again.")
    elif summarizer is None or detector is None:
        st.error("Model not loaded. Please refresh the page and try again.")
    else:
		# Duplicate submission prevention and cooldown
		import hashlib, time
		text_hash = hashlib.sha256(article.encode("utf-8")).hexdigest()
		now = time.time()
		if st.session_state["last_input_hash"] == text_hash:
			st.info("You already analyzed this exact text.")
			cooldown_left = 0
			if st.session_state["last_submit_ts"] is not None:
				elapsed = now - st.session_state["last_submit_ts"]
				cooldown_left = max(0, st.session_state["cooldown_seconds"] - int(elapsed))
			if cooldown_left > 0:
				st.caption(f"Please wait {cooldown_left}s before re-analyzing.")
			# Proceed anyway to allow re-run; message above informs the user.
		st.session_state["last_input_hash"] = text_hash
		st.session_state["last_submit_ts"] = now
        with st.spinner("Analyzing..."):
            # Summarize
            summary = summarizer(article, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

            # Detect with all scores to present more realistic accuracy
            prob_fake = None
            prob_real = None
            label = "UNKNOWN"
            score = 0.0
            all_scores = detector(article, return_all_scores=True)
            all_scores = all_scores[0] if isinstance(all_scores, list) else all_scores
            for item in all_scores:
                lbl = str(item.get("label", "")).upper()
                scr = float(item.get("score", 0.0))
                if "FAKE" in lbl:
                    prob_fake = scr
                if "REAL" in lbl or "TRUE" in lbl:
                    prob_real = scr
            if prob_fake is None or prob_real is None:
                result = detector(article)[0]
                top_label = result.get("label", "").upper()
                top_score = float(result.get("score", 0.0))
                if "FAKE" in top_label:
                    prob_fake = top_score
                    prob_real = 1.0 - top_score
                else:
                    prob_real = top_score
                    prob_fake = 1.0 - top_score
            if prob_fake >= prob_real:
                label, score = "FAKE", prob_fake
            else:
                label, score = "REAL", prob_real

        st.subheader("🧾 Summary:")
        st.write(summary)

        st.subheader("🔍 Credibility Check:")
        if label.lower() == "fake":
            st.error(f"⚠️ This article appears **FAKE** with confidence {score*100:.1f}%")
        elif label.lower() == "real":
            st.success(f"✅ This article appears **REAL** with confidence {score*100:.1f}%")
        else:
            st.info(f"This article's credibility is unclear. Confidence {score*100:.1f}%")

        # Show both probabilities for transparency
        st.markdown("**Class probabilities:**")
        st.write({"FAKE": f"{(prob_fake or 0.0)*100:.1f}%", "REAL": f"{(prob_real or 0.0)*100:.1f}%"})

        # Prepare result payload
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        snippet = (article[:200] + "…") if len(article) > 200 else article
        result_payload = {
            "timestamp": timestamp,
            "prediction": label,
            "confidence": round(float(score), 4),
            "prob_fake": round(float(prob_fake or 0.0), 4),
            "prob_real": round(float(prob_real or 0.0), 4),
            "summary": summary,
            "text_snippet": snippet,
        }

        # Save to history
        st.session_state["history"].append(result_payload)
		st.session_state["analysis_count"] += 1

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
    # Show concise table
    st.write([
        {
            "time": h["timestamp"],
            "prediction": h["prediction"],
            "conf%": f"{h['confidence']*100:.1f}",
            "fake%": f"{h['prob_fake']*100:.1f}",
            "real%": f"{h['prob_real']*100:.1f}",
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
                "prob_fake",
                "prob_real",
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
