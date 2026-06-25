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
    prompt = f"Decide if the following query requires real-time web search or searching external documents to answer correctly (e.g., current events, specific facts, recent news, detailed research). Answer only 'SEARCH' or 'DIRECT'.\n\nQuery: {query}"
    response = call_llm(prompt, max_tokens=10).strip().upper()
    # If the LLM determines it needs search, it is NOT a general query
    return "DIRECT" in response and "SEARCH" not in response


# =========================
# VECTOR STORE & RETRIEVAL
# =========================

def build_vector_store(documents):
    texts = [doc["content"] for doc in documents]
    embeddings = embed_model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, documents

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
# LLM STREAMING
# =========================

def stream_llm(prompt, max_tokens=700):
    if not GROQ_API_KEY:
        yield "GROQ_API_KEY not configured."
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"LLM Error: {e}"


# =========================
# PREPARE RAG
# =========================

def prepare_rag(query, pdf_path=None):
    """
    Returns (prompt, retrieved_docs, mode)
    """
    try:
        documents = []

        # Direct mode
        if is_general_query(query):
            prompt = f"Provide a clear and professional answer.\n\nQuestion:\n{query}"
            return prompt, [], "Direct"

        # Web Search
        web_docs = search_web(query)
        if web_docs:
            documents.extend(web_docs)

        # PDF
        if pdf_path:
            pdf_docs = load_pdf_as_documents(pdf_path)
            documents.extend(pdf_docs)

        if not documents:
            prompt = f"Provide a clear and professional answer indicating no documents were found.\n\nQuestion:\n{query}"
            return prompt, [], "Fallback"

        index, documents = build_vector_store(documents)
        context, retrieved_docs = retrieve_context(query, index, documents)

        if not context.strip():
            prompt = f"Provide a clear and professional answer indicating context was empty.\n\nQuestion:\n{query}"
            return prompt, [], "Fallback"

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
        return prompt, retrieved_docs, "RAG"

    except Exception as e:
        return f"Error occurred: {e}", [], "Error"


# =========================
# EVALUATE
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

def evaluate_answer(answer, retrieved_docs):
    if not retrieved_docs or answer.startswith("LLM Error:"):
        return 0.0, "LOW", 100.0

    flagged, score, total = hallucination_check(answer, retrieved_docs)
    risk = classify_risk(score)
    confidence = calculate_confidence(score)
    
    return score, risk, confidence