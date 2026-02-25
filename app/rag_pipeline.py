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

# =====================================================
# CONFIG
# =====================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

RETRIEVAL_K = 5

# =====================================================
# GROQ CALL
# =====================================================

def call_llm(prompt, max_tokens=700):

    if not GROQ_API_KEY:
        return "GROQ_API_KEY not configured."

    for attempt in range(3):
        try:
            client = Groq(api_key=GROQ_API_KEY)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Groq attempt {attempt+1} failed:", e)
            time.sleep(2)

    return "LLM connection failed after retries."

# =====================================================
# QUERY ROUTING
# =====================================================

def is_general_query(query: str) -> bool:
    research_keywords = [
        "paper", "research", "study",
        "journal", "arxiv",
        "2024", "2025"
    ]
    q = query.lower()
    return not any(k in q for k in research_keywords)

def needs_academic_boost(query: str) -> bool:
    academic_terms = [
        "research paper",
        "latest research",
        "arxiv",
        "journal",
        "conference",
        "icml",
        "neurips",
        "iclr"
    ]
    q = query.lower()
    return any(term in q for term in academic_terms)

# =====================================================
# DIRECT MODE
# =====================================================

def generate_direct_answer(query):
    prompt = f"""
Provide a clear, professional and concise answer.

Question:
{query}
"""
    return call_llm(prompt, max_tokens=500)

# =====================================================
# VECTOR STORE
# =====================================================

def build_vector_store(documents):
    texts = [doc["content"] for doc in documents]
    embeddings = embed_model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, documents

# =====================================================
# RETRIEVAL
# =====================================================

def retrieve_context(query, index, documents, k=RETRIEVAL_K):

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

# =====================================================
# RAG GENERATION
# =====================================================

def generate_rag_answer(query, context):

    prompt = f"""
You are a professional AI research assistant.

STRICT RULES:
1. Use ONLY information explicitly present in the Retrieved Context.
2. Reference citations like [1], [2].
3. Do NOT invent information.
4. If insufficient data exists, clearly state it.

=== MAIN ANSWER ===

Retrieved Context:
{context}

Question:
{query}
"""

    return call_llm(prompt)

# =====================================================
# CONFIDENCE
# =====================================================

def calculate_confidence(hallucination_score):
    return round(max(0, 100 - hallucination_score), 2)

def classify_risk(score):
    if score < 20:
        return "LOW"
    elif score < 40:
        return "MEDIUM"
    else:
        return "HIGH"

# =====================================================
# MAIN PIPELINE
# =====================================================

def run_rag(query, pdf_path=None):

    try:

        # DIRECT MODE
        if is_general_query(query):
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 95.0, "Direct"

        # Academic boost
        if needs_academic_boost(query):
            query += " site:arxiv.org"

        documents = []

        web_docs = search_web(query)
        if web_docs:
            documents.extend(web_docs)

        if pdf_path:
            pdf_docs = load_pdf_as_documents(pdf_path)
            documents.extend(pdf_docs)

        if not documents:
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 80.0, "Fallback"

        # Vector store
        index, documents = build_vector_store(documents)

        # Retrieval
        context, retrieved_docs = retrieve_context(query, index, documents)

        if not context.strip():
            return "No relevant context found.", [], 0.0, "MEDIUM", 50.0, "Fallback"

        # Generate answer
        answer = generate_rag_answer(query, context)

        # Hallucination check
        flagged, score, total = hallucination_check(answer, retrieved_docs)

        risk = classify_risk(score)
        confidence = calculate_confidence(score)

        return answer, retrieved_docs, score, risk, confidence, "RAG"

    except Exception as e:
        print("Pipeline error:", e)
        return f"Pipeline failed: {e}", [], 0.0, "HIGH", 0.0, "Error"