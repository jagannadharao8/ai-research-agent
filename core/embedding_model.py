from sentence_transformers import SentenceTransformer

# Load model only once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")