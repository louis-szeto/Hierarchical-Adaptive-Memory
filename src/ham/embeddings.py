"""Embedding backends: a deterministic hash embedder (no dependencies) and a
real sentence-transformers embedder. Both support optional Matryoshka-style
dimension truncation.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from .config import EmbeddingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return mat / norm


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder.

    Each lowercased alphanumeric token is hashed (blake2b) to a bucket and a
    sign; features are summed and L2-normalized. This is fully reproducible
    across machines and requires no model download, so it is the CI default and
    a graceful fallback when sentence-transformers is unavailable.
    """

    kind = "hash"

    def __init__(self, dim: int = 256, seed: int = 0, normalize: bool = True):
        self.dim = dim
        self.seed = seed
        self.normalize = normalize

    def _hash(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        val = int.from_bytes(h, "little")
        bucket = val % self.dim
        sign = 1.0 if (val >> 63) & 1 else -1.0
        return bucket, sign

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _TOKEN_RE.findall(text.lower()):
                b, s = self._hash(tok)
                out[i, b] += s
        if self.normalize:
            out = _l2_normalize(out)
        return out


class SentenceTransformerEmbedder:
    """Real embeddings via sentence-transformers (optional dependency)."""

    kind = "sentence-transformers"

    def __init__(self, model_id: str, normalize: bool = True):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)
        self.model_id = model_id
        self.normalize = normalize
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(cfg: EmbeddingConfig):
    if cfg.kind == "hash":
        emb = HashEmbedder(dim=cfg.dim, seed=cfg.seed, normalize=cfg.normalize)
    elif cfg.kind == "sentence-transformers":
        emb = SentenceTransformerEmbedder(cfg.model_id, normalize=cfg.normalize)
    else:
        raise ValueError(f"unknown embedding kind: {cfg.kind!r}")
    return _MaybeTruncated(emb, cfg.matryoshka_dim, cfg.normalize)


class _MaybeTruncated:
    """Wraps an embedder to optionally truncate to a Matryoshka prefix dim."""

    def __init__(self, base, matryoshka_dim: int | None, normalize: bool):
        self.base = base
        self.matryoshka_dim = matryoshka_dim
        self.normalize = normalize
        self.dim = min(base.dim, matryoshka_dim) if matryoshka_dim else base.dim
        self.kind = base.kind

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.base.encode(texts)
        if self.matryoshka_dim and vecs.shape[1] > self.matryoshka_dim:
            vecs = vecs[:, : self.matryoshka_dim]
            if self.normalize:
                vecs = _l2_normalize(vecs)
        return np.asarray(vecs, dtype=np.float32)
