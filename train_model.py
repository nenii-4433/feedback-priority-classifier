"""
train_model.py - trains the feedback priority classifier.

Trains TWO models from feedback_dataset.csv (+ real_feedback.csv if present):
  1. Sentence-embedding model  -> embedding_model.joblib   (more accurate)
  2. TF-IDF word model         -> model.joblib + vectorizer.joblib (fallback)

predict.py uses the embedding model when it is available, otherwise TF-IDF.
"""

import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from nltk.stem import PorterStemmer
import joblib

DATA_PATH = "feedback_dataset.csv"
REAL_PATH = "real_feedback.csv"
VECTORIZER_PATH = "vectorizer.joblib"
MODEL_PATH = "model.joblib"
EMBED_MODEL_PATH = "embedding_model.joblib"

_stemmer = PorterStemmer()

# sklearn's default stopword list deletes "no", "not", "nothing", "never"...
# which carry the meaning we need, so we only drop harmless filler words.
SAFE_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at"}


def preprocess_text(text):
    """TF-IDF path only: style markers + stemming (must match predict.py)."""
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


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "priority"])
    df["priority"] = df["priority"].str.lower().str.strip()
    df = df[df["priority"].isin(["high", "medium", "low"])]
    df["raw_text"] = df["text"].astype(str)          # embeddings use natural text
    df["text"] = df["raw_text"].apply(preprocess_text)  # TF-IDF uses stemmed text
    return df


def report(y_test, preds, labels):
    print(classification_report(y_test, preds))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("Labels order:", labels)
    print(confusion_matrix(y_test, preds, labels=labels))


def train_tfidf(df):
    print("\n=== TF-IDF model (fallback) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["priority"], test_size=0.2, random_state=42,
        stratify=df["priority"],
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)
    report(y_test, clf.predict(vectorizer.transform(X_test)),
           sorted(df["priority"].unique()))
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved {VECTORIZER_PATH} and {MODEL_PATH}")


def train_embedding(df):
    import embedder

    print("\n=== Sentence-embedding model ===")
    print("Loading embedding model (first run downloads ~90 MB)...")
    X = embedder.embed(df["raw_text"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["priority"], test_size=0.2, random_state=42,
        stratify=df["priority"],
    )
    clf = LogisticRegression(max_iter=2000, C=10, class_weight="balanced")
    clf.fit(X_train, y_train)
    report(y_test, clf.predict(X_test), sorted(df["priority"].unique()))
    joblib.dump(clf, EMBED_MODEL_PATH)
    print(f"Saved {EMBED_MODEL_PATH}")


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} labeled examples")

    # Optional: your own REAL labeled feedback (columns: text,priority).
    if os.path.exists(REAL_PATH):
        real = load_data(REAL_PATH)
        print(f"Adding {len(real)} real examples from {REAL_PATH}")
        df = pd.concat([df, real], ignore_index=True)

    print(df["priority"].value_counts())

    train_tfidf(df)

    try:
        train_embedding(df)
    except Exception as e:
        print(f"\nCould not train the embedding model: {e}")
        print("predict.py will use the TF-IDF model instead.")
        print("To enable embeddings:  pip install sentence-transformers")
        if os.path.exists(EMBED_MODEL_PATH):
            os.remove(EMBED_MODEL_PATH)  # avoid using a stale embedding model


if __name__ == "__main__":
    main()