"""Ties ingestion -> embedding -> retrieval -> generation together, and
owns the cross-cutting v1 behaviors: dedup, relevance thresholding,
conversation history, and graceful degradation on backend failures.
"""
import os

import config
from embeddings import downgrade_embedding_backend, get_embedding_backend
from ingestion import DocumentError, chunk_document, hash_file, load_document
from llm_providers import ExtractiveProvider, get_llm_provider
from vectorstore import VectorStore


def _dedupe_by_overlap(results, threshold):
    """Greedily drops chunks whose word overlap with an already-kept, higher
    scoring chunk exceeds `threshold`. CHUNK_OVERLAP means two adjacent
    chunks from the same document are often near-duplicates once both score
    highly - this keeps the retained set diverse instead of redundant."""
    kept = []
    kept_word_sets = []
    for r in results:
        words = set(r["text"].lower().split())
        if any(
            words and kw and len(words & kw) / min(len(words), len(kw)) >= threshold
            for kw in kept_word_sets
        ):
            continue
        kept.append(r)
        kept_word_sets.append(words)
    return kept


class RagPipeline:
    def __init__(self, embedding_backend=None, llm_provider=None):
        self.embedding_backend = embedding_backend or get_embedding_backend()
        self.llm_provider = llm_provider or get_llm_provider()
        self.vectorstore = None
        self.all_chunks = []  # raw chunk dicts, backend-agnostic - lets us
        # rebuild the index from scratch if the embedding backend changes.
        self.indexed_hashes = set()
        self.indexed_files = []
        self.history = []  # list of (question, answer), capped at HISTORY_TURNS

    # -- indexing ----------------------------------------------------------

    def index_document(self, path):
        filename = os.path.basename(path)
        file_hash = hash_file(path)
        if file_hash in self.indexed_hashes:
            return f"'{filename}' was already indexed - skipping duplicate."

        pages = load_document(path)  # raises DocumentError on bad input
        chunks = chunk_document(pages, filename)
        if not chunks:
            raise DocumentError(f"No extractable text found in '{filename}'.")

        self.all_chunks.extend(chunks)

        if self.embedding_backend.stateful:
            self._rebuild_index()
        else:
            try:
                vectors = self.embedding_backend.embed([c["text"] for c in chunks])
            except Exception:
                old_name = self.embedding_backend.name
                self.embedding_backend = downgrade_embedding_backend(self.embedding_backend)
                self._rebuild_index()
                self.indexed_hashes.add(file_hash)
                self.indexed_files.append(filename)
                return (
                    f"'{old_name}' embedding API failed - switched to local "
                    f"'{self.embedding_backend.name}' backend and re-indexed everything. "
                    f"Indexed '{filename}' ({len(chunks)} chunks)."
                )
            if self.vectorstore is None:
                self.vectorstore = VectorStore(vectors.shape[1])
            self.vectorstore.add(vectors, chunks)

        self.indexed_hashes.add(file_hash)
        self.indexed_files.append(filename)
        return f"Indexed '{filename}' - {len(chunks)} chunks added. Total documents: {len(self.indexed_files)}."

    def _rebuild_index(self):
        texts = [c["text"] for c in self.all_chunks]
        vectors = self.embedding_backend.fit(texts) if self.embedding_backend.stateful else self.embedding_backend.embed(texts)
        self.vectorstore = VectorStore(vectors.shape[1])
        self.vectorstore.rebuild(vectors, self.all_chunks)

    # -- querying ------------------------------------------------------------

    def ask(self, question):
        if not question or not question.strip():
            return {"answer": "Please enter a question.", "citations": []}
        if self.vectorstore is None or self.vectorstore.ntotal == 0:
            return {"answer": "No documents indexed yet. Please upload a document first.", "citations": []}

        query_vec = self.embedding_backend.embed([question])[0]
        candidates = self.vectorstore.search(query_vec, config.TOP_K * config.RETRIEVAL_FANOUT)
        threshold = config.RELEVANCE_THRESHOLDS.get(self.embedding_backend.name, config.DEFAULT_RELEVANCE_THRESHOLD)
        passing = [r for r in candidates if r["score"] >= threshold]
        relevant = _dedupe_by_overlap(passing, config.DEDUP_OVERLAP_THRESHOLD)[: config.TOP_K]

        if not relevant:
            answer = (
                "I couldn't find anything in the indexed documents that covers this "
                "question, so I won't guess. Try rephrasing, or upload a document "
                "that covers this topic."
            )
            self._update_history(question, answer)
            return {"answer": answer, "citations": []}

        try:
            answer = self.llm_provider.generate(question, relevant, self.history)
        except Exception:
            self.llm_provider = ExtractiveProvider()
            answer = self.llm_provider.generate(question, relevant, self.history)

        citations = [
            {"source": r["source"], "page": r.get("page"), "score": round(r["score"], 3)}
            for r in relevant
        ]
        self._update_history(question, answer)
        return {"answer": answer, "citations": citations}

    def _update_history(self, question, answer):
        self.history.append((question, answer))
        self.history = self.history[-config.HISTORY_TURNS:]

    # -- lifecycle -----------------------------------------------------------

    def reset(self):
        self.vectorstore = None
        self.all_chunks = []
        self.indexed_hashes = set()
        self.indexed_files = []
        self.history = []
        if self.embedding_backend.stateful:
            self.embedding_backend = type(self.embedding_backend)()

    def status(self):
        return {
            "embedding_backend": self.embedding_backend.name,
            "llm_backend": self.llm_provider.name,
            "num_documents": len(self.indexed_files),
            "num_chunks": len(self.all_chunks),
        }
