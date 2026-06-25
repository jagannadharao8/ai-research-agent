import streamlit as st
import sys
import os
import tempfile
import hashlib
import pandas as pd

# --------------------------------------------------
# FIX PYTHON PATH FOR STREAMLIT CLOUD
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag_pipeline import prepare_rag, stream_llm, evaluate_answer
from export.pdf_report import generate_pdf_report
from core.database import log_query, get_analytics
from core.cache import check_cache, add_to_cache
from core.logger import logger

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Autonomous AI Research Agent",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS AESTHETICS (GLASSMORPHISM)
# --------------------------------------------------
st.markdown("""
<style>
/* Base Theme */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f1f2f6;
    font-family: 'Inter', sans-serif;
}
/* Main Card Styles */
.block-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 3rem !important;
    margin-top: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
/* Inputs */
.stTextInput>div>div>input {
    background-color: rgba(0, 0, 0, 0.2);
    color: white;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.2);
}
/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
    color: #000;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(0, 201, 255, 0.5);
}
/* Metrics */
[data-testid="stMetricValue"] {
    color: #92FE9D !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("# 🧠 Autonomous AI Research Agent V2")
st.markdown("### *Next-Gen Research Assistant Powered by Groq + Llama*")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Research", "📊 Real-Time Analytics"])

with tab1:
    # --------------------------------------------------
    # SESSION STATE INIT
    # --------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    col_run, col_clear = st.columns([1, 5])
    with col_clear:
        if st.button("Clear History 🗑️"):
            st.session_state.messages = []
            st.rerun()

    # Sidebar for PDF Upload
    with st.sidebar:
        st.header("Document Context")
        uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metrics" in msg:
                m = msg["metrics"]
                cols = st.columns(3)
                cols[0].caption(f"Score: {m['score']}%")
                cols[1].caption(f"Confidence: {m['confidence']}%")
                cols[2].caption(f"Risk: {m['risk']}")

    # Chat Input
    if query := st.chat_input("Ask a research question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            pdf_path = None
            if uploaded_pdf is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    pdf_path = tmp.name
                    
            logger.info(f"Processing query: {query}")
            
            # Check Semantic Cache
            cached = check_cache(query)
            if cached:
                st.info("⚡ Served from Semantic Cache")
                st.markdown(cached["answer"])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": cached["answer"],
                    "metrics": {"score": cached["score"], "confidence": cached["confidence"], "risk": cached["risk"]}
                })
            else:
                with st.spinner("Analyzing web sources and preparing context..."):
                    prompt, sources, mode = prepare_rag(query, pdf_path, chat_history=st.session_state.messages[:-1])
                
                st.info(f"**Execution Mode:** {mode}")
                
                # Streaming
                stream = stream_llm(prompt)
                answer = st.write_stream(stream)
                
                with st.spinner("Calculating reliability metrics..."):
                    score, risk, confidence = evaluate_answer(answer, sources)
                    log_query(query, mode, score, confidence, risk)
                    
                    # Add to cache
                    add_to_cache(query, mode, prompt, sources, answer, score, risk, confidence)
                
                cols = st.columns(3)
                cols[0].caption(f"Score: {score}%")
                cols[1].caption(f"Confidence: {confidence}%")
                cols[2].caption(f"Risk: {risk}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metrics": {"score": score, "confidence": confidence, "risk": risk}
                })

            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass

with tab2:
    st.subheader("Live System Analytics")
    data = get_analytics()
    if not data:
        st.info("No queries logged yet. Run a research query to see analytics.")
    else:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", len(df))
        col2.metric("Average Confidence", f"{df['confidence'].mean():.2f}%")
        col3.metric("Average Hallucination Risk", f"{df['hallucination_score'].mean():.2f}%")
        
        st.markdown("### Confidence Trend")
        st.line_chart(df.set_index('timestamp')['confidence'])
        
        st.markdown("### Hallucination Score Trend")
        st.area_chart(df.set_index('timestamp')['hallucination_score'])
        
        st.markdown("### Recent Queries")
        st.dataframe(df[['timestamp', 'query', 'mode', 'confidence', 'risk_level']].head(10))