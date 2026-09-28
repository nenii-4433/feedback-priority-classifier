"""
embedder.py - turns sentences into meaning vectors using a pretrained model.

First run downloads the model (about 90 MB) and caches it; after that it
works offline. Needs:  pip install sentence-transformers
"""

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def load_embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    return load_embedder().encode(
        list(texts), show_progress_bar=False, normalize_embeddings=True
    )