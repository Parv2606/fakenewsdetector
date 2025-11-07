import streamlit as st
from transformers import pipeline
import json
from io import StringIO
import csv
from datetime import datetime
import pandas as pd

# --- Streamlit Configuration and Styling ---
st.set_page_config(
    page_title="Credibility Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stAlert {
    border-radius: 12px;
}
.stMetric > div {
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stMetricLabel {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
}
.stMetricValue {
    font-size: 2.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_models():
    """
    Loads high-quality models for summarization and text classification.
    Using publicly available, robust models from Hugging Face.
    """
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        detector = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
        return summarizer, detector
    except Exception as e:
        st.error(f"⚠️ Model Loading Error! Could not load required models. Please check your connection or try again later.")
        st.code(e, language='python')
        return None, None

summarizer, detector = load_models()

# --- Application Title and Description ---
st.title("🧭 Credibility Compass: AI-Powered News Analysis")
st.markdown("A tool for students to analyze text credibility and get fast summaries. Uses top-tier AI models for reliable results.")

# Initialize history storage
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Main Input Area ---
st.subheader("1. Paste Your Article Here")
article = st.text_area(
    "Paste a news article or paragraph for analysis:", 
    height=250,
    placeholder="Start typing or paste a long news article..."
)

col_analyze, _ = st.columns([1, 4])
with col_analyze:
    analyze_button = st.button("🚀 Analyze Text", use_container_width=True)

if analyze_button:
    if not article.strip():
        st.warning("Please enter some text in the box above to analyze.")
    elif summarizer is None or detector is None:
        st.error("AI Models are not loaded. Analysis cannot proceed.")
    else:
        with st.spinner("Analyzing text and checking credibility..."):
            try:
                summary = summarizer(
                    article, 
                    max_length=150, 
                    min_length=40, 
                    do_sample=False, 
                    truncation=True
                )[0]['summary_text']

                # Get model output and map labels if needed
                all_scores = detector(article, return_all_scores=True)[0]
                # Map LABEL_0 and LABEL_1 depending on the model (for mrm8488/bert-tiny-finetuned-fake-news-detection):
                # LABEL_0 is FAKE, LABEL_1 is REAL
                label_map = {"LABEL_0": "FAKE", "LABEL_1": "REAL"}
                prob_fake = prob_real = 0.0
                for item in all_scores:
                    lbl = label_map.get(item["label"], item["label"])
                    if lbl == "FAKE":
                        prob_fake = item["score"]
                    elif lbl == "REAL":
                        prob_real = item["score"]
                if prob_fake > prob_real:
                    label, score = "FAKE", prob_fake
                else:
                    label, score = "REAL", prob_real

                # Display results
                st.divider()
                st.subheader("2. Analysis Results")
                col_pred, col_fake, col_real = st.columns(3)

                with col_pred:
                    delta_str = f"{score*100:.1f}% Confidence"
                    st.metric(
                        label="Credibility Prediction",
                        value=label,
                        delta=delta_str,
                        delta_color="inverse" if label == "FAKE" else "normal"
                    )

                with col_fake:
                    st.metric("Fake Probability", f"{prob_fake*100:.1f}%", delta_color="off")
                with col_real:
                    st.metric("Real Probability", f"{prob_real*100:.1f}%", delta_color="off")

                st.markdown("---")
                st.markdown("### 📜 Summary")
                st.info(summary)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                snippet = (article[:250].replace('\n', ' ') + "...") if len(article) > 250 else article.replace('\n', ' ')
                result_payload = {
                    "timestamp": timestamp,
                    "prediction": label,
                    "confidence": round(float(score), 4),
                    "prob_fake": round(float(prob_fake), 4),
                    "prob_real": round(float(prob_real), 4),
                    "summary": summary,
                    "text_snippet": snippet,
                }
                st.session_state["history"].append(result_payload)
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# --- History Section ---
st.divider()
st.subheader("3. Analysis History")

if len(st.session_state["history"]) == 0:
    st.info("Your analysis history will appear here once you run your first check.")
else:
    history_reversed = st.session_state["history"][::-1]
    df = pd.DataFrame(history_reversed)
    df = df.rename(columns={
        "timestamp": "Time",
        "prediction": "Result",
        "confidence": "Conf.",
        "prob_fake": "Fake %",
        "prob_real": "Real %",
        "text_snippet": "Text Snippet"
    })
    df["Conf."] = (df["Conf."] * 100).map('{:.1f}%'.format)
    df["Fake %"] = (df["Fake %"] * 100).map('{:.1f}%'.format)
    df["Real %"] = (df["Real %"] * 100).map('{:.1f}%'.format)
    df_display = df[["Time", "Result", "Conf.", "Fake %", "Real %", "Text Snippet"]]

    with st.expander(f"**View Last {len(df_display)} Analyses**", expanded=True):
        st.dataframe(
            df_display, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Text Snippet": st.column_config.TextColumn("Text Snippet", width="medium"),
                "Time": st.column_config.TextColumn("Time", width="small"),
            }
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            json_data = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Download All (JSON)",
                data=json_data,
                file_name="credibility_compass_history.json",
                mime="application/json",
                use_container_width=True
            )
        with col2:
            csv_buffer = StringIO()
            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=st.session_state["history"][0].keys()
            )
            writer.writeheader()
            for row in st.session_state["history"]:
                writer.writerow(row)
            st.download_button(
                label="⬇️ Download All (CSV)",
                data=csv_buffer.getvalue(),
                file_name="credibility_compass_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()
