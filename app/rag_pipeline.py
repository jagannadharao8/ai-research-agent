import os
import time
import numpy as np
import faiss
import ollama

from core.embedding_model import embed_model
from tools.web_search import search_web
from tools.pdf_loader import load_pdf_as_documents
from evaluation.hallucination_checker import hallucination_check


LLM_MODEL = "llama3:8b"
SIMILARITY_THRESHOLD = 0.35


# =========================
# QUERY ROUTING
# =========================

def is_general_query(query: str) -> bool:
    research_keywords = [
        "research", "paper", "study", "latest",
        "2024", "2025", "report", "statistics"
    ]
    q = query.lower()
    return not any(word in q for word in research_keywords)


def generate_direct_answer(query):
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": f"Provide a clear and professional answer.\n\nQuestion:\n{query}"
        }],
        options={"num_predict": 400}
    )
    return response["message"]["content"]


# =========================
# VECTOR STORE
# =========================

def build_vector_store(documents):
    texts = [doc["content"] for doc in documents]

    embeddings = embed_model.encode(texts)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, documents


# =========================
# RETRIEVAL
# =========================

def retrieve_context(query, index, documents, k=5):
    query_vector = embed_model.encode([query])
    D, I = index.search(np.array(query_vector), k)

    retrieved_docs = []
    context_chunks = []

    for citation_number, idx in enumerate(I[0], start=1):
        doc = documents[idx]

        doc_with_citation = {
            "citation": citation_number,
            "source": doc.get("source", "web"),
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "content": doc["content"]
        }

        retrieved_docs.append(doc_with_citation)

        context_chunks.append(
            f"[{citation_number}] {doc['content']}"
        )

    context_text = "\n\n".join(context_chunks)

    return context_text, retrieved_docs


# =========================
# RAG GENERATION
# =========================

def generate_answer(query, context):

    prompt = f"""
You are a professional AI research assistant.

STRICT RULES:
1. Use ONLY information explicitly present in the Retrieved Context.
2. Reference citations like [1], [2].
3. Do NOT invent information.
4. If insufficient data exists, clearly state it.

FORMAT:

=== MAIN ANSWER ===
...

=== MODEL INFERENCE ===
...

=== RECOMMENDATIONS ===
...

Retrieved Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 600}
    )

    return response["message"]["content"]


# =========================
# RISK + CONFIDENCE
# =========================

def classify_risk(score):
    if score < 20:
        return "LOW"
    elif score < 40:
        return "MEDIUM"
    else:
        return "HIGH"


def calculate_confidence(score):
    return round(max(0, 100 - score), 2)


# =========================
# MAIN PIPELINE
# =========================

def run_rag(query, pdf_path=None):

    try:

        documents = []

        # Direct Mode
        if is_general_query(query):
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 95.0, "Direct"

        # Web Search
        web_docs = search_web(query)
        if web_docs:
            documents.extend(web_docs)

        # PDF Load
        if pdf_path:
            pdf_docs = load_pdf_as_documents(pdf_path)
            documents.extend(pdf_docs)

        if not documents:
            return "No documents found.", [], 0.0, "LOW", 100.0, "Fallback"

        # Vector Store
        index, documents = build_vector_store(documents)

        # Retrieval
        context, retrieved_docs = retrieve_context(query, index, documents)

        if not context.strip():
            return "Retrieved context is empty.", [], 0.0, "MEDIUM", 50.0, "Fallback"

        # Generate Answer
        answer = generate_answer(query, context)

        # Hallucination Check
        flagged, score, total = hallucination_check(answer, retrieved_docs)

        risk = classify_risk(score)
        confidence = calculate_confidence(score)

        return answer, retrieved_docs, score, risk, confidence, "RAG"

    except Exception as e:
        return f"Pipeline failed: {e}", [], 0.0, "HIGH", 0.0, "Error"