import os
import time
import numpy as np
import faiss
from groq import Groq
from dotenv import load_dotenv

from core.embedding_model import embed_model
from tools.web_search import search_web
from tools.pdf_loader import load_pdf_as_documents
from evaluation.hallucination_checker import hallucination_check


# ============================================================
# LOAD ENV VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# QUERY ROUTING
# ============================================================

def is_exam_query(query: str) -> bool:
    exam_keywords = [
        "pgecet",
        "gate previous papers",
        "jee papers",
        "exam papers",
        "previous year papers",
        "question papers"
    ]
    q = query.lower()
    return any(word in q for word in exam_keywords)


def is_general_query(query: str) -> bool:
    research_keywords = [
        "latest", "research", "2024", "2025",
        "paper", "study", "report",
        "statistics", "trend", "analysis"
    ]
    q = query.lower()
    return not any(word in q for word in research_keywords)


def needs_academic_boost(query: str) -> bool:
    academic_terms = [
        "research paper",
        "latest research",
        "arxiv",
        "journal",
        "conference paper",
        "icml",
        "neurips",
        "iclr"
    ]
    q = query.lower()
    return any(term in q for term in academic_terms)


# ============================================================
# STABLE GROQ CALL (CURRENT MODEL)
# ============================================================

def call_llm(prompt, max_tokens=600):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "GROQ_API_KEY not configured properly."

    for attempt in range(3):
        try:
            client = Groq(api_key=api_key)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # ✅ CURRENT ACTIVE GROQ MODEL
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Groq attempt {attempt+1} failed:", e)
            time.sleep(2)

    return "LLM connection failed after retries."


# ============================================================
# DIRECT MODE
# ============================================================

def generate_direct_answer(query):
    prompt = f"""
Provide a clear, concise and professional answer.

Question:
{query}
"""
    return call_llm(prompt, max_tokens=400)


# ============================================================
# VECTOR STORE
# ============================================================

def build_vector_store(documents):
    texts = [doc["content"] for doc in documents]

    embeddings = embed_model.encode(texts)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, documents


# ============================================================
# RETRIEVAL WITH CITATIONS
# ============================================================

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


# ============================================================
# RAG GENERATION
# ============================================================

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

    return call_llm(prompt, max_tokens=700)


# ============================================================
# CONFIDENCE & RISK
# ============================================================

def calculate_confidence(score):
    return round(max(0, 100 - score), 2)


def classify_risk(score):
    if score < 20:
        return "LOW"
    elif score < 40:
        return "MEDIUM"
    else:
        return "HIGH"


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_rag(query, pdf_path=None):

    documents = []

    try:

        # -------------------------
        # EXAM ROUTING
        # -------------------------
        if is_exam_query(query):
            return (
                "For latest PGECET papers, please visit the official APSCHE website "
                "or university portals for downloadable PDFs.",
                [],
                0.0,
                "LOW",
                100.0
            )

        # -------------------------
        # DIRECT MODE
        # -------------------------
        if is_general_query(query):
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 100.0

        # -------------------------
        # ACADEMIC BOOST
        # -------------------------
        if needs_academic_boost(query):
            query += " site:arxiv.org"

        # -------------------------
        # WEB SEARCH
        # -------------------------
        web_docs = search_web(query)
        if web_docs:
            documents.extend(web_docs)

        # -------------------------
        # PDF LOAD
        # -------------------------
        if pdf_path:
            pdf_docs = load_pdf_as_documents(pdf_path)
            documents.extend(pdf_docs)

        if not documents:
            return "No documents found.", [], 0.0, "LOW", 100.0

        # -------------------------
        # VECTOR STORE
        # -------------------------
        index, documents = build_vector_store(documents)

        # -------------------------
        # RETRIEVAL
        # -------------------------
        context, retrieved_docs = retrieve_context(query, index, documents)

        if not context.strip():
            return "Retrieved context is empty.", [], 0.0, "MEDIUM", 50.0

        # -------------------------
        # GENERATE ANSWER
        # -------------------------
        answer = generate_answer(query, context)

        flagged, score, total = hallucination_check(answer, retrieved_docs)

        risk = classify_risk(score)
        confidence = calculate_confidence(score)

        return answer, retrieved_docs, score, risk, confidence

    except Exception as e:
        print("Pipeline error:", e)
        return f"Pipeline failed: {e}", [], 0.0, "HIGH", 0.0