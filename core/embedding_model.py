from sentence_transformers import SentenceTransformer

_model = None

def get_embed_model():
    """Returns a singleton SentenceTransformer model instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model