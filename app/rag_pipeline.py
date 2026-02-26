import os
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv

from core.embedding_model import embed_model
from tools.web_search import search_web
from tools.pdf_loader import load_pdf_as_documents
from evaluation.hallucination_checker import hallucination_check

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"


# =========================
# LLM CALL (GROQ)
# =========================

def call_llm(prompt, max_tokens=600):
    if not GROQ_API_KEY:
        return "GROQ_API_KEY not configured."

    try:
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {e}"


# =========================
# QUERY ROUTING
# =========================

def is_general_query(query: str) -> bool:
    research_keywords = [
        "research", "paper", "study",
        "latest", "2024", "2025",
        "report", "statistics"
    ]
    q = query.lower()
    return not any(word in q for word in research_keywords)


def generate_direct_answer(query):
    prompt = f"Provide a clear and professional answer.\n\nQuestion:\n{query}"
    return call_llm(prompt, 400)


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
        context_chunks.append(f"[{citation_number}] {doc['content']}")

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

    return call_llm(prompt, 700)


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

        # Direct mode
        if is_general_query(query):
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 95.0, "Direct"

        # Web Search
        web_docs = search_web(query)
        if web_docs:
            documents.extend(web_docs)

        # PDF
        if pdf_path:
            pdf_docs = load_pdf_as_documents(pdf_path)
            documents.extend(pdf_docs)

        if not documents:
            return "No documents found.", [], 0.0, "LOW", 100.0, "Fallback"

        index, documents = build_vector_store(documents)
        context, retrieved_docs = retrieve_context(query, index, documents)

        if not context.strip():
            return "Retrieved context is empty.", [], 0.0, "MEDIUM", 50.0, "Fallback"

        answer = generate_answer(query, context)

        flagged, score, total = hallucination_check(answer, retrieved_docs)

        risk = classify_risk(score)
        confidence = calculate_confidence(score)

        return answer, retrieved_docs, score, risk, confidence, "RAG"

    except Exception as e:
        return f"Pipeline failed: {e}", [], 0.0, "HIGH", 0.0, "Error"