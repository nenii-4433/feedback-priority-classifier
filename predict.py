"""
predict.py - classify feedback as HIGH / MEDIUM / LOW priority.

Three ways to use it:

  1. Single text:
       python predict.py "the app crashes when I log in"

  2. Interactive (type feedback, press Enter; type q to quit):
       python predict.py

  3. Batch - a CSV file with a column named "text":
       python predict.py --file feedback.csv
     Writes feedback_predicted.csv with priority + confidence columns.
"""

import os
import re
import sys
import joblib
from nltk.stem import PorterStemmer

VECTORIZER_PATH = "vectorizer.joblib"
MODEL_PATH = "model.joblib"
EMBED_MODEL_PATH = "embedding_model.joblib"

_vectorizer = None
_model = None
_stemmer = PorterStemmer()

SAFE_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at"}


def preprocess_text(text):
    """Must exactly match the preprocessing used in train_model.py."""
    markers = []
    if re.search(r"!{2,}", text):
        markers.append("manyexclaimmarker")

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 8:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.5:
            markers.append("allcapsmarker")

    words = re.findall(r"[a-zA-Z']+", text.lower())
    stemmed = [_stemmer.stem(w) for w in words if w not in SAFE_STOPWORDS]
    return " ".join(stemmed + markers)


_backend = None  # "embedding" or "tfidf"
_embed_clf = None


def _load():
    """Use the embedding model if available, otherwise fall back to TF-IDF."""
    global _vectorizer, _model, _backend, _embed_clf
    if _backend is not None:
        return
    if os.path.exists(EMBED_MODEL_PATH):
        try:
            import embedder
            embedder.load_embedder()  # fails fast if sentence-transformers is missing
            _embed_clf = joblib.load(EMBED_MODEL_PATH)
            _backend = "embedding"
            return
        except Exception as e:
            print(f"(embedding model unavailable: {e}; using TF-IDF)", file=sys.stderr)
    _vectorizer = joblib.load(VECTORIZER_PATH)
    _model = joblib.load(MODEL_PATH)
    _backend = "tfidf"


def predict_priority(text):
    """Returns (label, {label: probability})."""
    _load()
    if _backend == "embedding":
        import embedder
        vec = embedder.embed([text])
        clf = _embed_clf
    else:
        vec = _vectorizer.transform([preprocess_text(text)])
        clf = _model
    prediction = clf.predict(vec)[0]
    probabilities = clf.predict_proba(vec)[0]
    return prediction, dict(zip(clf.classes_, probabilities))


def print_result(text):
    priority, confidence = predict_priority(text)
    print(f"Text: {text}")
    print(f"Predicted priority: {priority.upper()}  [model: {_backend}]")
    print("Confidence scores:")
    for label, score in sorted(confidence.items(), key=lambda x: -x[1]):
        print(f"  {label}: {score:.2%}")


def run_batch(path):
    import pandas as pd

    df = pd.read_csv(path)
    column = "text" if "text" in df.columns else df.columns[0]
    labels, scores = [], []
    for text in df[column].fillna("").astype(str):
        label, conf = predict_priority(text)
        labels.append(label)
        scores.append(round(conf[label], 4))

    df["predicted_priority"] = labels
    df["confidence"] = scores
    out_path = re.sub(r"\.csv$", "", path) + "_predicted.csv"
    df.to_csv(out_path, index=False)
    print(f"Classified {len(df)} rows using column '{column}'.")
    print(df["predicted_priority"].value_counts().to_string())
    print(f"Saved to {out_path}")


def run_interactive():
    print("Feedback priority classifier. Type feedback and press Enter (q to quit).")
    while True:
        text = input("\n> ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        if text:
            print_result(text)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        run_interactive()
    elif args[0] == "--file" and len(args) > 1:
        run_batch(args[1])
    else:
        print_result(" ".join(args))