import numpy as np

from embeddings import TfidfEmbeddingBackend


def test_tfidf_fit_produces_normalized_vectors():
    backend = TfidfEmbeddingBackend()
    vecs = backend.fit(["cat dog animal", "dog bird animal", "quarterly finance report"])
    assert vecs.shape[0] == 3
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_tfidf_query_transform_uses_existing_vocab_after_fit():
    backend = TfidfEmbeddingBackend()
    backend.fit(["cat dog animal", "dog bird animal"])
    qvec = backend.embed(["cat"])
    assert qvec.shape == (1, backend.fit(["cat dog animal", "dog bird animal"]).shape[1])


def test_tfidf_unrelated_query_has_low_similarity_to_unrelated_corpus():
    backend = TfidfEmbeddingBackend()
    corpus_vecs = backend.fit(["employee leave and paid time off policy"])
    query_vec = backend.embed(["what is the capital of France"])
    similarity = float(np.dot(corpus_vecs[0], query_vec[0]))
    assert similarity < 0.1
