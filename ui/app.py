import streamlit as st
import sys
import os

# --------------------------------------------------
# FIX PYTHON PATH FOR STREAMLIT CLOUD
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag_pipeline import run_rag

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Autonomous AI Research Agent",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
# 🧠 Autonomous AI Research Agent
### Created by Jagannadharao
---
""")

st.caption("Powered by Groq + Llama")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
query = st.text_input("Enter your research question:")
use_pdf = st.checkbox("Add a PDF document")

pdf_path = None
if use_pdf:
    pdf_path = st.text_input("Enter full PDF path:")

# --------------------------------------------------
# RUN PIPELINE
# --------------------------------------------------
if st.button("Run Research"):

    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Running AI Research Pipeline..."):
            answer, sources, score, risk, confidence, mode = run_rag(query, pdf_path)

        st.success("Research Completed")

        # Mode Display
        st.info(f"Mode: {mode}")

        # Response
        st.subheader("AI Response")
        st.markdown(answer)

        # Sidebar Metrics
        st.sidebar.header("Reliability Metrics")
        st.sidebar.metric("Hallucination Score", f"{score:.2f}%")
        st.sidebar.metric("Confidence", f"{confidence:.2f}%")
        st.sidebar.metric("Risk Level", risk)

        # Sources
        if sources:
            st.subheader("Sources")
            for doc in sources:
                citation = doc.get("citation", "")
                title = doc.get("title", "Untitled")
                url = doc.get("url", "")

                if url:
                    st.markdown(f"[{citation}] [{title}]({url})")
                else:
                    st.markdown(f"[{citation}] {title}")