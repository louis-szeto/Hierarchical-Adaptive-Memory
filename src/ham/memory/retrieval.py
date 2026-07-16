"""Retrieval over stored records: cosine (numpy), FAISS, or lexical.

The cosine path is the reference implementation and requires only numpy. The
FAISS path is used when available and requested; it must produce parity with the
numpy path on exact (flat inner-product) search. The lexical path is a
BM25-lite token-overlap retriever for the lexical-only ablation.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .store import MemoryRecord

_TOK = re.compile(r"[a-z0-9]+")


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return mat / norms


def cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    if matrix.shape[0] == 0:
        return []
    q = query.astype(np.float32)
    qn = np.linalg.norm(q)
    if qn > 1e-12:
        q = q / qn
    m = _normalize_rows(matrix.astype(np.float32))
    sims = m @ q
    k = min(k, sims.shape[0])
    idx = np.argsort(-sims, kind="stable")[:k]
    return [(int(i), float(sims[i])) for i in idx]


def faiss_topk(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    import faiss

    if matrix.shape[0] == 0:
        return []
    m = _normalize_rows(matrix.astype(np.float32))
    q = query.astype(np.float32)[None, :]
    qn = np.linalg.norm(q)
    if qn > 1e-12:
        q = q / qn
    index = faiss.IndexFlatIP(m.shape[1])
    index.add(np.ascontiguousarray(m))
    k = min(k, m.shape[0])
    sims, idx = index.search(np.ascontiguousarray(q), k)
    return [(int(i), float(s)) for i, s in zip(idx[0], sims[0]) if i >= 0]


def lexical_topk(query_text: str, records: list[MemoryRecord], k: int) -> list[tuple[int, float]]:
    """BM25-lite token-overlap retrieval (for the lexical-only ablation)."""
    if not records:
        return []
    docs = [Counter(_TOK.findall(r.text.lower())) for r in records]
    df: Counter = Counter()
    for d in docs:
        for term in d:
            df[term] += 1
    n = len(records)
    q_terms = _TOK.findall(query_text.lower())
    scores = np.zeros(n, dtype=np.float32)
    avgdl = np.mean([sum(d.values()) for d in docs]) or 1.0
    k1, b = 1.5, 0.75
    for j, d in enumerate(docs):
        dl = sum(d.values()) or 1
        s = 0.0
        for term in q_terms:
            if term not in d:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = d[term]
            s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores[j] = s
    k = min(k, n)
    idx = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in idx]


def retrieve(
    query_text: str,
    query_emb: np.ndarray,
    records: list[MemoryRecord],
    k: int,
    method: str = "cosine",
) -> list[tuple[MemoryRecord, float]]:
    if not records:
        return []
    if method == "lexical":
        hits = lexical_topk(query_text, records, k)
    else:
        matrix = np.vstack([r.embedding for r in records]).astype(np.float32)
        if method == "faiss":
            hits = faiss_topk(query_emb, matrix, k)
        elif method == "cosine":
            hits = cosine_topk(query_emb, matrix, k)
        else:
            raise ValueError(f"unknown retrieval method: {method!r}")
    return [(records[i], score) for i, score in hits]
