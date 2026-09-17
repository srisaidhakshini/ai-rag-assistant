"""Pluggable embedding backends, auto-selected by what's available
(PRD requirement #4): OpenAI -> Sentence-Transformers -> TF-IDF.
"""
import os

import numpy as np

import config


def _l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return vecs / norms


class EmbeddingBackend:
    name = "base"
    # True for backends whose vector space depends on the whole corpus
    # (TF-IDF's vocabulary), meaning a full re-embed is needed on every
    # new document rather than an incremental append.
    stateful = False

    def embed(self, texts):
        raise NotImplementedError

    def fit(self, texts):
        return self.embed(texts)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = config.OPENAI_EMBEDDING_MODEL

    def embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        return _l2_normalize(vecs)


class SentenceTransformerBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)

    def embed(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs.astype("float32")


class TfidfEmbeddingBackend(EmbeddingBackend):
    name = "tfidf"
    stateful = True

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def fit(self, texts):
        vecs = self.vectorizer.fit_transform(texts).toarray().astype("float32")
        self._fitted = True
        return _l2_normalize(vecs)

    def embed(self, texts):
        if not self._fitted:
            return self.fit(texts)
        vecs = self.vectorizer.transform(texts).toarray().astype("float32")
        return _l2_normalize(vecs)


def get_embedding_backend():
    """Auto-select the best available backend. A missing key or an
    uninstalled optional dependency silently falls back to the next tier -
    the app must always be runnable with zero configuration."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingBackend()
        except Exception:
            pass
    try:
        return SentenceTransformerBackend()
    except ImportError:
        pass
    return TfidfEmbeddingBackend()


def downgrade_embedding_backend(current):
    """Called when a live API call fails at runtime (invalid key, rate
    limit, outage) - degrades to the next local-capable tier so indexing
    can keep going instead of hard-failing."""
    if current.name == "openai":
        try:
            return SentenceTransformerBackend()
        except ImportError:
            pass
    return TfidfEmbeddingBackend()
