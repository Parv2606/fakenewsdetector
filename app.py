import streamlit as st
import joblib
from datetime import datetime

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

@st.cache_resource
def load_model():
    vectorizer = joblib.load("model/vectorizer.pkl")
    classifier = joblib.load("model/classifier.pkl")
    return vectorizer, classifier

vectorizer, classifier = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<div style="text-align:center;">
<h1>📰 Fake News Detector</h1>
<p>Check credibility of any text using machine learning.</p>
</div>
""", unsafe_allow_html=True)

text = st.text_area("Paste news or statement here:", height=230)

if st.button("🚀 Analyze", use_container_width=True):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        X = vectorizer.transform([text])
        prediction = classifier.predict(X)[0]
        confidence = max(classifier.predict_proba(X)[0])

        if prediction == 1:
            st.success(f"✅ Likely Real — Confidence: {confidence*100:.2f}%")
        else:
            st.error(f"⚠️ Fake News Detected — Confidence: {confidence*100:.2f}%")

        # Save history
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "text": text[:150] + "…" if len(text) > 150 else text,
            "result": "REAL" if prediction == 1 else "FAKE",
            "confidence": f"{confidence*100:.1f}%"
        })

st.markdown("---")
st.markdown("### 🗂️ Analysis History")
for record in reversed(st.session_state.history):
    with st.expander(f"{record['time']} — {record['result']} ({record['confidence']})"):
        st.write(record["text"])

if st.button("🧹 Clear History"):
    st.session_state.history = []
    st.experimental_rerun()
