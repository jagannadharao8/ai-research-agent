import streamlit as st
import sys
import os
import tempfile
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
st.markdown("# 🧠 Autonomous AI Research Agent")
st.markdown("### *Next-Gen Research Assistant Powered by Groq + Llama*")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Research", "📊 Real-Time Analytics"])

with tab1:
    # --------------------------------------------------
    # USER INPUT
    # --------------------------------------------------
    query = st.text_input("Enter your research question:")
    use_pdf = st.checkbox("Add a PDF document for context")

    pdf_path = None
    if use_pdf:
        pdf_path = st.text_input("Enter full PDF path (or relative to project root):")

    # --------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------
    if st.button("Run Research 🚀"):

        if not query.strip():
            st.warning("Please enter a research question.")
        else:
            with st.spinner("Analyzing web sources and preparing context..."):
                prompt, sources, mode = prepare_rag(query, pdf_path)
            
            st.success("Context Prepared! Generating response...")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.info(f"**Execution Mode:** {mode}")
                st.subheader("AI Response")
                
                # Real-time Streaming
                stream = stream_llm(prompt)
                answer = st.write_stream(stream)

                # Sources
                if sources:
                    st.subheader("Sources Referenced")
                    for doc in sources:
                        citation = doc.get("citation", "")
                        title = doc.get("title", "Untitled")
                        url = doc.get("url", "")
                        if url:
                            st.markdown(f"- **[{citation}]** [{title}]({url})")
                        else:
                            st.markdown(f"- **[{citation}]** {title}")

            with col2:
                with st.spinner("Calculating reliability metrics..."):
                    score, risk, confidence = evaluate_answer(answer, sources)
                    
                    # Log to database for real-time analytics
                    log_query(query, mode, score, confidence, risk)

                st.subheader("Reliability Metrics")
                st.metric("Hallucination Score", f"{score:.2f}%")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.metric("Risk Level", risk)
                
                # PDF Generation
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_pdf_path = tmp_file.name
                    
                try:
                    generate_pdf_report(
                        filename=tmp_pdf_path,
                        query=query,
                        answer=answer,
                        score=score,
                        confidence=confidence,
                        risk=risk,
                        sources=sources
                    )
                    
                    with open(tmp_pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        
                    st.markdown("---")
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name="Research_Report.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")
                finally:
                    if os.path.exists(tmp_pdf_path):
                        os.remove(tmp_pdf_path)

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