import streamlit as st
import sys
import os

# --------------------------------------------------
# FIX PYTHON PATH FOR STREAMLIT CLOUD
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# IMPORT PIPELINE
# --------------------------------------------------
from app.rag_pipeline import run_rag

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Autonomous AI Research Agent",
    layout="wide",
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>
        🧠 Autonomous AI Research Agent
    </h1>
    <h4 style='text-align:center; color:gray;'>
        Created by Jagannadharao
    </h4>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Powered by badge
st.markdown(
    "<div style='text-align:center; font-size:13px; color:gray;'>Powered by Groq + Llama</div>",
    unsafe_allow_html=True,
)

st.markdown("")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
query = st.text_input("🔎 Enter your research question:")

use_pdf = st.checkbox("Add a PDF document")

pdf_path = None
if use_pdf:
    pdf_path = st.text_input("Enter full PDF path:")

# --------------------------------------------------
# RUN BUTTON
# --------------------------------------------------
if st.button("Run Research"):

    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Running AI Research Pipeline..."):
            answer, sources, score, risk = run_rag(query, pdf_path)

        st.success("Research Completed ✔")

        # -------------------------------
        # MAIN RESPONSE
        # -------------------------------
        st.subheader("📄 AI Response")
        st.markdown(answer)

        # -------------------------------
        # SIDEBAR METRICS
        # -------------------------------
        st.sidebar.title("🔍 Hallucination Analysis")
        st.sidebar.metric("Hallucination Score", f"{score:.2f}%")
        st.sidebar.metric("Risk Level", risk)

        # -------------------------------
        # SOURCES SECTION
        # -------------------------------
        if sources:
            st.subheader("📚 Sources")

            for doc in sources:
                title = doc.get("title", "Source")
                url = doc.get("url", "")
                citation = doc.get("citation", "")

                if url:
                    st.markdown(f"[{citation}] {title}  \n🔗 {url}")
                else:
                    st.markdown(f"[{citation}] {title}")