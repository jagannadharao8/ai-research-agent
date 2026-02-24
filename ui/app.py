import streamlit as st
import sys
import os
import json
import time
from streamlit_lottie import st_lottie

# ---------------------------------------------------
# FIX PYTHON PATH
# ---------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag_pipeline import run_rag

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Autonomous AI Research Agent",
    layout="wide"
)

# ---------------------------------------------------
# LOAD LOTTIE (UTF-8 SAFE)
# ---------------------------------------------------
def load_lottie(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
animation_path = os.path.join(BASE_DIR, "..", "assets", "ai_animation.json")
ai_anim = load_lottie(animation_path)

# ---------------------------------------------------
# INTRO SCREEN (ONLY ONCE)
# ---------------------------------------------------
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        text-align: center;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🚀 Autonomous AI Research Agent")
    st.markdown("### Initializing Intelligence Engine...")
    st.caption("Created by Jagannadharao")

    if ai_anim:
        st_lottie(ai_anim, height=250)

    time.sleep(2)
    st.session_state.intro_done = True
    st.rerun()

# ---------------------------------------------------
# PREMIUM UI STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.gradient-text {
    font-size: 44px;
    font-weight: bold;
    background: linear-gradient(90deg, #00dbde, #fc00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown('<div class="gradient-text">Autonomous AI Research Agent</div>', unsafe_allow_html=True)
st.markdown("### Hybrid RAG • Query Refinement • Semantic Ranking • Reliability Scoring")
st.caption("Created by Jagannadharao")

st.markdown("""
<div style="
    display: inline-block;
    padding: 6px 14px;
    margin-top: 6px;
    font-size: 13px;
    border-radius: 20px;
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(8px);
    color: #00e0ff;
    font-weight: 500;
">
⚡ Powered by Llama3 (Local via Ollama)
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------
# INPUT CARD
# ---------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

query = st.text_input("Enter your research question:")
use_pdf = st.checkbox("Add a PDF document")

pdf_path = None
if use_pdf:
    pdf_path = st.text_input("Enter full PDF path:")

run_button = st.button("Run Research")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# RUN PIPELINE
# ---------------------------------------------------
if run_button:

    if not query.strip():
        st.warning("Please enter a research question.")
    else:

        with st.spinner("Running AI Research Pipeline..."):
            answer, sources, hallucination_score, risk, confidence, mode = run_rag(query, pdf_path)

        st.success("Research Completed ✔")

        # Mode Badge
        if mode == "RAG":
            st.success(f"Mode: {mode}")
        elif mode == "Direct":
            st.info(f"Mode: {mode}")
        elif mode == "Fallback":
            st.warning(f"Mode: {mode}")
        else:
            st.error(f"Mode: {mode}")

        # ---------------------------------------------------
        # RESPONSE CARD
        # ---------------------------------------------------
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📄 AI Response")
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------------------------------
        # SIDEBAR METRICS
        # ---------------------------------------------------
        st.sidebar.header("🔍 Reliability Metrics")
        st.sidebar.metric("Hallucination Score", f"{hallucination_score:.2f}%")
        st.sidebar.metric("Risk Level", risk)
        st.sidebar.metric("Confidence", f"{confidence:.2f}%")

        # ---------------------------------------------------
        # SOURCES
        # ---------------------------------------------------
        if sources:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📚 Sources")

            for doc in sources:
                citation = doc.get("citation", "")
                title = doc.get("title", "Untitled")
                url = doc.get("url", "")

                if url:
                    st.markdown(f"[{citation}] [{title}]({url})")
                else:
                    st.markdown(f"[{citation}] {title}")

            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()