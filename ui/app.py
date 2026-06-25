import streamlit as st
import sys
import os
import tempfile

# --------------------------------------------------
# FIX PYTHON PATH FOR STREAMLIT CLOUD
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag_pipeline import run_rag
from export.pdf_report import generate_pdf_report

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
        with st.spinner("Analyzing web sources and formulating response..."):
            answer, sources, score, risk, confidence, mode = run_rag(query, pdf_path)

        st.success("Research Completed!")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.info(f"**Execution Mode:** {mode}")
            st.subheader("AI Response")
            st.markdown(answer)

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