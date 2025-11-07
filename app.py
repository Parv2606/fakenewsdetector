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


# Customizing the look and feel
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
    Using better models ensures higher accuracy and relevance.
    """
    try:
        # High-Quality Summarizer: BART large CNN (standard for abstractive summarization)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Enhanced Detector: A BERT model fine-tuned specifically for Fake News (Credibility)
        # This model is generally more robust than the tiny BERT version.
        detector = pipeline("text-classification", model="Hello-SimpleAI/bert-base-cased-fake-news-detection")
        
        return summarizer, detector
    except Exception as e:
        st.error(f"⚠️ **Model Loading Error!** Could not load required models. Please check your connection or try again later.")
        st.code(e, language='python')
        return None, None


summarizer, detector = load_models()


# --- Application Title and Description ---
st.title("🧭 Credibility Compass: AI-Powered News Analysis")
st.markdown("A tool for students to analyze text credibility and get fast summaries. Use better AI models for more accurate results.")


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


# Use a container for the button for better centering/layout
col_analyze, col_spacer = st.columns([1, 4])
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
                # --- Summarization ---
                summary = summarizer(
                    article, 
                    max_length=150, 
                    min_length=40, 
                    do_sample=False, 
                    truncation=True
                )[0]['summary_text']


                # --- Detection ---
                # Retrieve all scores for both "FAKE" and "REAL" labels
                all_scores = detector(article, return_all_scores=True)[0]
                
                # Normalize and find the probabilities for explicit labels
                prob_map = {item['label'].upper(): item['score'] for item in all_scores}
                
                prob_fake = prob_map.get("FAKE", 0.0)
                prob_real = prob_map.get("REAL", 0.0)
                
                # Determine the final prediction
                if prob_fake > prob_real:
                    label, score = "FAKE", prob_fake
                else:
                    label, score = "REAL", prob_real
                
                # --- Display Results ---
                st.divider()
                st.subheader("2. Analysis Results")
                
                # Result Metrics using columns
                col_pred, col_fake, col_real = st.columns(3)


                with col_pred:
                    icon = "🚨" if label == "FAKE" else "✅"
                    delta_str = f"{score*100:.1f}% Confidence"
                    st.metric(
                        label="Credibility Prediction",
                        value=label,
                        delta=delta_str,
                        delta_color="inverse" if label == "FAKE" else "normal"
                    )


                with col_fake:
                    st.metric("Fake Probability", f"{(prob_fake)*100:.1f}%", delta_color="off")


                with col_real:
                    st.metric("Real Probability", f"{(prob_real)*100:.1f}%", delta_color="off")
                    
                st.markdown("---")
                
                st.markdown("### 📜 Summary")
                st.info(summary)


                # --- History Storage ---
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
    # Reverse history for newest first display
    history_reversed = st.session_state["history"][::-1]


    # Convert history list of dicts to a Pandas DataFrame for better visualization
    df = pd.DataFrame(history_reversed)
    df = df.rename(columns={
        "timestamp": "Time",
        "prediction": "Result",
        "confidence": "Conf.",
        "prob_fake": "Fake %",
        "prob_real": "Real %",
        "text_snippet": "Text Snippet"
    })
    
    # Format percentages
    df["Conf."] = (df["Conf."] * 100).map('{:.1f}%'.format)
    df["Fake %"] = (df["Fake %"] * 100).map('{:.1f}%'.format)
    df["Real %"] = (df["Real %"] * 100).map('{:.1f}%'.format)
    
    # Select columns to display in the table
    df_display = df[["Time", "Result", "Conf.", "Fake %", "Real %", "Text Snippet"]]


    # Use an expander to keep the UI clean
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


        # Download history as JSON
        with col1:
            json_data = json.dumps(st.session_state["history"], ensure_ascii=False, indent=2)
            st.download_button(
                label="⬇️ Download All (JSON)",
                data=json_data,
                file_name="credibility_compass_history.json",
                mime="application/json",
                use_container_width=True
            )


        # Download history as CSV
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


        # Clear history
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["history"] = []
                st.rerun() # Refresh the app to clear the display
