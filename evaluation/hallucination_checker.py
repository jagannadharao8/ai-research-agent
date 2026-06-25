import numpy as np
import re
from core.embedding_model import get_embed_model


def extract_main_answer(text):
    """
    Extract text inside === MAIN ANSWER === section with fallbacks.
    """
    # Standard extraction
    pattern = r"=== MAIN ANSWER ===(.*?)=== MODEL INFERENCE ==="
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    
    # Fallback 1: Missing model inference block
    pattern_fallback = r"=== MAIN ANSWER ===(.*)"
    match_fallback = re.search(pattern_fallback, text, re.DOTALL)
    
    if match_fallback:
        return match_fallback.group(1).strip()
        
    return text.strip()


def split_into_sentences(text):
    """
    Improved sentence splitter avoiding common abbreviations
    """
    # A slightly better regex avoiding single letter abbreviations like U.S. or U.K.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def safe_cosine_similarity(vec_a, vec_b):
    """
    Safe cosine similarity computation
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_similarity(sentence, context_embeddings):
    """
    Compute maximum cosine similarity between sentence and retrieved docs
    """
    sentence_embedding = get_embed_model().encode([sentence])[0]

    similarities = [
        safe_cosine_similarity(sentence_embedding, ctx_emb)
        for ctx_emb in context_embeddings
    ]

    return max(similarities) if similarities else 0.0


def hallucination_check(full_answer, retrieved_docs, threshold=0.50):
    """
    Returns:
    - flagged sentences with similarity
    - hallucination score (%)
    - total sentences evaluated
    """

    main_answer = extract_main_answer(full_answer)
    sentences = split_into_sentences(main_answer)

    if not sentences:
        return [], 0.0, 0

    context_texts = [doc["content"] for doc in retrieved_docs]

    if not context_texts:
        return [], 0.0, len(sentences)

    context_embeddings = get_embed_model().encode(context_texts)

    flagged = []

    for sentence in sentences:
        similarity = compute_similarity(sentence, context_embeddings)

        if similarity < threshold:
            flagged.append((sentence, similarity))

    total_sentences = len(sentences)
    hallucination_score = (len(flagged) / total_sentences) * 100

    return flagged, hallucination_score, total_sentences