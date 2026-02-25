import streamlit as st
import sys
import os
import json
import tempfile
from streamlit_lottie import st_lottie

# -----------------------------
# FIX PYTHON PATH
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag_pipeline import run_rag
from export.pdf_report import generate_pdf_report

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Autonomous AI Research Agent",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# DARK / LIGHT THEME TOGGLE
# -----------------------------
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)

if theme:
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------
# LOAD LOTTIE ANIMATION
# -----------------------------
def load_lottie(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

BASE_DIR = os.path.dirname(__file__)
animation_path = os.path.join(BASE_DIR, "..", "assets", "ai_animation.json")
animation = load_lottie(animation_path)

# -----------------------------
# HEADER SECTION
# -----------------------------
st.markdown("""
    <h1 style='text-align: center;'>🧠 Autonomous AI Research Agent</h1>
    <p style='text-align: center; font-size:18px;'>
    Created by Jagannadharao
    </p>
""", unsafe_allow_html=True)

st.markdown(
    "<div style='text-align:center; font-size:14px;'>⚡ Powered by Llama3</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# INTRO ANIMATION
# -----------------------------
if animation:
    st_lottie(animation, height=200, key="intro_anim")

# -----------------------------
# INPUT SECTION
# -----------------------------
query = st.text_input("🔍 Enter your research question:")

use_pdf = st.checkbox("📄 Add a PDF document")

pdf_path = None
if use_pdf:
    pdf_path = st.text_input("Enter full PDF path:")

st.markdown("")

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("🚀 Run Research"):

    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Running AI Research Pipeline..."):

            answer, sources, score, risk, confidence = run_rag(query, pdf_path)

        st.success("Research Completed ✔")

        # -----------------------------
        # AI RESPONSE
        # -----------------------------
        st.markdown("## 📄 AI Response")
        st.markdown(answer)

        # -----------------------------
        # SIDEBAR METRICS
        # -----------------------------
        st.sidebar.markdown("## 📊 Metrics")
        st.sidebar.metric("Hallucination Score", f"{score:.2f}%")
        st.sidebar.metric("Confidence", f"{confidence:.2f}%")
        st.sidebar.metric("Risk Level", risk)

        if sources:
            st.sidebar.success("Mode: RAG")
        else:
            st.sidebar.info("Mode: Direct LLM")

        # -----------------------------
        # SOURCES SECTION
        # -----------------------------
        if sources:
            st.markdown("## 📚 Sources")

            for doc in sources:
                citation = doc.get("citation", "")
                title = doc.get("title", "Untitled")
                url = doc.get("url", "")
                source_type = doc.get("source", "web")

                if url:
                    st.markdown(
                        f"[{citation}] **{title}** ({source_type})  \n🔗 {url}"
                    )
                else:
                    st.markdown(
                        f"[{citation}] **{title}** ({source_type})"
                    )

        # -----------------------------
        # PDF EXPORT SECTION
        # -----------------------------
        if answer and answer.strip():

            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

            generate_pdf_report(
                temp_pdf.name,
                query,
                answer,
                score,
                confidence,
                risk,
                sources
            )

            with open(temp_pdf.name, "rb") as f:
                st.download_button(
                    label="📥 Download Research Report (PDF)",
                    data=f,
                    file_name="AI_Research_Report.pdf",
                    mime="application/pdf"
                )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; font-size:12px;'>Autonomous AI Research Agent v1.0</div>",
            unsafe_allow_html=True
        )