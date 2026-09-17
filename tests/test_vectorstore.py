import numpy as np

from vectorstore import VectorStore


def test_add_and_search_returns_best_match_first():
    vs = VectorStore(dim=3)
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    metas = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    vs.add(vecs, metas)

    results = vs.search(np.array([1, 0, 0], dtype="float32"), top_k=2)
    assert results[0]["text"] == "a"
    assert results[0]["score"] > results[1]["score"]


def test_rebuild_replaces_index_and_metadata():
    vs = VectorStore(dim=3)
    vs.add(np.array([[1, 0, 0]], dtype="float32"), [{"text": "old"}])

    vs.rebuild(np.array([[0, 1]], dtype="float32"), [{"text": "new"}])
    assert vs.ntotal == 1
    assert vs.metadata[0]["text"] == "new"
    assert vs.dim == 2


def test_search_on_empty_store_returns_empty_list():
    vs = VectorStore(dim=3)
    assert vs.search(np.array([1, 0, 0], dtype="float32"), top_k=3) == []


def test_save_and_load_roundtrip(tmp_path):
    vs = VectorStore(dim=2)
    vs.add(np.array([[1, 0]], dtype="float32"), [{"text": "x"}])
    path = str(tmp_path / "idx")
    vs.save(path)

    vs2 = VectorStore(dim=2)
    vs2.load(path)
    assert vs2.ntotal == 1
    assert vs2.metadata[0]["text"] == "x"
