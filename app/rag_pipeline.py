import ollama
import numpy as np
import faiss

from core.embedding_model import embed_model
from tools.web_search import search_web
from tools.pdf_loader import load_pdf_as_documents
from evaluation.hallucination_checker import hallucination_check


# =====================================================
# CONFIGURATION
# =====================================================

LLM_MODEL = "llama3:8b"

RETRIEVAL_SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_LENGTH = 4000


# =====================================================
# QUERY TYPE DETECTION
# =====================================================

def is_general_query(query: str) -> bool:
    query = query.lower()

    research_keywords = [
        "latest", "research", "2024", "2025",
        "study", "paper", "analysis", "impact",
        "statistics", "report"
    ]

    return not any(k in query for k in research_keywords)


def is_research_query(query: str) -> bool:
    query = query.lower()

    research_keywords = [
        "paper", "research", "study",
        "journal", "survey",
        "arxiv", "ieee", "acm",
        "2024", "2025"
    ]

    return any(k in query for k in research_keywords)


# =====================================================
# QUERY REFINEMENT
# =====================================================

def refine_query(query: str) -> str:
    prompt = f"""
Rewrite the following query to optimize it for web retrieval.
Make it specific and structured.

Query:
{query}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 60}
    )

    return response["message"]["content"].strip()


# =====================================================
# DIRECT MODE
# =====================================================

def generate_direct_answer(query):

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": f"Provide a clear, professional answer.\n\nQuestion:\n{query}"
        }],
        options={"num_predict": 400}
    )

    return response["message"]["content"]


# =====================================================
# SEARCH RANKING
# =====================================================

def rank_documents(query, documents, top_n=6):

    if not documents:
        return documents

    query_vec = embed_model.encode([query])[0]

    scored_docs = []

    for doc in documents:
        doc_vec = embed_model.encode([doc["content"]])[0]

        similarity = np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        )

        scored_docs.append((similarity, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored_docs[:top_n]]


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
# RETRIEVAL WITH DYNAMIC K
# =====================================================

def retrieve_context(query, index, documents, top_k):

    query_vector = embed_model.encode([query])
    D, I = index.search(np.array(query_vector), top_k)

    retrieved_docs = []
    similarities = []

    for position, idx in enumerate(I[0]):

        # Convert L2 distance to similarity proxy
        similarity = 1 - D[0][position]
        similarities.append(similarity)

        doc = documents[idx]

        retrieved_docs.append({
            "citation": position + 1,
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "content": doc["content"]
        })

    avg_similarity = float(np.mean(similarities)) if similarities else 0

    return retrieved_docs, avg_similarity


# =====================================================
# RAG GENERATION
# =====================================================

def generate_answer(query, retrieved_docs):

    context = ""

    for doc in retrieved_docs:
        context += f"[{doc['citation']}] {doc['content']}\n\n"

    # Context trimming
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    prompt = f"""
Use the retrieved context primarily.
Cite using [number].
Apply reasoning only if necessary.

Context:
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


# =====================================================
# CONFIDENCE CALCULATION
# =====================================================

def compute_confidence(hallucination_score,
                       retrieval_similarity,
                       critic_score=80):

    hallucination_component = (100 - hallucination_score) * 0.4
    retrieval_component = (retrieval_similarity * 100) * 0.3
    critic_component = critic_score * 0.3

    confidence = (
        hallucination_component +
        retrieval_component +
        critic_component
    )

    return max(0, min(100, confidence))


# =====================================================
# MAIN PIPELINE
# =====================================================

def run_rag(query, pdf_path=None):

    try:

        # -------------------------------------------------
        # DIRECT MODE FOR GENERAL QUESTIONS
        # -------------------------------------------------

        if is_general_query(query):
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 95.0, "Direct"

        # -------------------------------------------------
        # QUERY REFINEMENT
        # -------------------------------------------------

        refined_query = refine_query(query)

        if is_research_query(query):
            refined_query += " site:arxiv.org OR site:ieee.org OR site:acm.org"

        # -------------------------------------------------
        # WEB SEARCH
        # -------------------------------------------------

        web_docs = search_web(refined_query)

        if not web_docs:
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 85.0, "Fallback"

        # -------------------------------------------------
        # RANK DOCUMENTS
        # -------------------------------------------------

        ranked_docs = rank_documents(query, web_docs)

        # -------------------------------------------------
        # BUILD VECTOR STORE
        # -------------------------------------------------

        index, documents = build_vector_store(ranked_docs)

        # -------------------------------------------------
        # DYNAMIC TOP-K
        # -------------------------------------------------

        if len(query.split()) > 10:
            top_k = 6
        else:
            top_k = 3

        retrieved_docs, retrieval_similarity = retrieve_context(
            query, index, documents, top_k
        )

        # -------------------------------------------------
        # SIMILARITY THRESHOLD CHECK
        # -------------------------------------------------

        if retrieval_similarity < RETRIEVAL_SIMILARITY_THRESHOLD:
            answer = generate_direct_answer(query)
            return answer, [], 0.0, "LOW", 80.0, "Fallback"

        # -------------------------------------------------
        # GENERATE RAG ANSWER
        # -------------------------------------------------

        answer = generate_answer(query, retrieved_docs)

        # -------------------------------------------------
        # HALLUCINATION CHECK
        # -------------------------------------------------

        flagged, hallucination_score, _ = hallucination_check(
            answer, retrieved_docs
        )

        # -------------------------------------------------
        # CONFIDENCE
        # -------------------------------------------------

        confidence = compute_confidence(
            hallucination_score,
            retrieval_similarity
        )

        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if hallucination_score > 40:
            risk = "HIGH"
        elif hallucination_score > 20:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return answer, retrieved_docs, hallucination_score, risk, confidence, "RAG"

    except Exception as e:
        return f"Pipeline failed: {e}", [], 0.0, "HIGH", 0.0, "Error"