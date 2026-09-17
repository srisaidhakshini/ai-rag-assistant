"""FAISS-backed vector store using exact cosine similarity
(normalized vectors + IndexFlatIP), per PRD requirement #5."""
import pickle

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []  # list of dicts, aligned by row with the index

    def add(self, vectors, metadatas):
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.metadata.extend(metadatas)

    def rebuild(self, vectors, metadatas):
        """Replaces the index wholesale - used after a stateful re-embed
        (TF-IDF vocab changed) or an embedding-backend downgrade."""
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.metadata = list(metadatas)

    def search(self, query_vec, top_k):
        if self.index.ntotal == 0:
            return []
        query_vec = np.ascontiguousarray(query_vec, dtype="float32").reshape(1, -1)
        scores, idxs = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results

    def save(self, path):
        """Scaffolded persistence (PRD Future Work #3) - not wired into
        the app's runtime by default, which stays in-memory per session."""
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump({"metadata": self.metadata, "dim": self.dim}, f)

    def load(self, path):
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".meta.pkl", "rb") as f:
            data = pickle.load(f)
        self.metadata = data["metadata"]
        self.dim = data["dim"]

    @property
    def ntotal(self):
        return self.index.ntotal
