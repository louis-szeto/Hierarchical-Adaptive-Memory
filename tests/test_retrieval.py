import numpy as np

from ham.memory import retrieval
from ham.memory.store import MemoryRecord


def _records(vectors, texts):
    return [MemoryRecord(id=i, text=texts[i], embedding=np.asarray(v, dtype=np.float32))
            for i, v in enumerate(vectors)]


def test_cosine_topk_orders_by_similarity():
    mat = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
    q = np.array([1, 0], dtype=np.float32)
    hits = retrieval.cosine_topk(q, mat, k=2)
    assert [i for i, _ in hits] == [0, 1]


def test_cosine_and_faiss_parity_when_available():
    faiss = None
    try:
        import faiss  # noqa: F401
    except Exception:
        return  # FAISS optional: parity check only runs when installed
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((50, 16)).astype(np.float32)
    q = rng.standard_normal(16).astype(np.float32)
    a = [i for i, _ in retrieval.cosine_topk(q, mat, 5)]
    b = [i for i, _ in retrieval.faiss_topk(q, mat, 5)]
    assert a == b


def test_lexical_retrieval_prefers_token_overlap():
    recs = _records(
        [[1, 0], [0, 1], [1, 1]],
        ["the capital of aurora is verona",
         "the mascot of basalt is otter",
         "aurora capital verona city"],
    )
    hits = retrieval.retrieve("what is the capital of aurora", np.zeros(2, dtype=np.float32),
                              recs, k=1, method="lexical")
    assert hits[0][0].text.startswith("the capital of aurora")


def test_retrieve_empty_is_safe():
    assert retrieval.retrieve("q", np.zeros(2, dtype=np.float32), [], k=3) == []
