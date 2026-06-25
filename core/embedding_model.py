import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def get_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")