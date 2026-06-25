import os
import json
import numpy as np
import faiss
from core.embedding_model import get_embed_model

CACHE_FILE = "semantic_cache.json"
CACHE_THRESHOLD = 0.95

_cache_index = None
_cache_data = []

def init_cache():
    global _cache_index, _cache_data
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _cache_data = json.load(f)
            
        if _cache_data:
            texts = [item["query"] for item in _cache_data]
            embeddings = get_embed_model().encode(texts)
            dimension = embeddings.shape[1]
            _cache_index = faiss.IndexFlatIP(dimension) # Inner product for cosine sim if normalized
            faiss.normalize_L2(embeddings)
            _cache_index.add(np.array(embeddings, dtype=np.float32))
            return
            
    # Empty cache
    dimension = 384 # all-MiniLM-L6-v2 dimension
    _cache_index = faiss.IndexFlatIP(dimension)
    _cache_data = []

def check_cache(query):
    global _cache_index, _cache_data
    if _cache_index is None:
        init_cache()
        
    if not _cache_data:
        return None
        
    query_vector = get_embed_model().encode([query])
    faiss.normalize_L2(query_vector)
    
    D, I = _cache_index.search(np.array(query_vector, dtype=np.float32), 1)
    
    if D[0][0] >= CACHE_THRESHOLD:
        idx = I[0][0]
        if idx != -1 and idx < len(_cache_data):
            return _cache_data[idx]
            
    return None

def add_to_cache(query, mode, prompt, sources, answer, score, risk, confidence):
    global _cache_index, _cache_data
    if _cache_index is None:
        init_cache()
        
    entry = {
        "query": query,
        "mode": mode,
        "prompt": prompt,
        "sources": sources,
        "answer": answer,
        "score": score,
        "risk": risk,
        "confidence": confidence
    }
    
    query_vector = get_embed_model().encode([query])
    faiss.normalize_L2(query_vector)
    
    _cache_data.append(entry)
    _cache_index.add(np.array(query_vector, dtype=np.float32))
    
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache_data, f, indent=2)
