"""The HAM memory system: ingest -> tier -> consolidate -> retrieve -> serialize.

One :class:`HAMemory` instance serves one example (one multi-session history).
It is parameterized by a :class:`~ham.conditions.ConditionSpec` so that the same
code path produces HAM, the uncompressed-retrieval baseline, and every ablation.
The frozen LLM is never touched here -- memory is the only independent variable.
"""

from __future__ import annotations

import os
import time

import numpy as np

from ..compression import serialize
from ..conditions import ConditionSpec
from ..config import MemoryConfig
from . import importance as imp
from .consolidation import Consolidator
from .retrieval import retrieve
from .store import EPISODIC, EVICTED, SEMANTIC, MemoryRecord, MemoryStore


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split a turn into <= max_chars chunks on sentence-ish boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks


class HAMemory:
    def __init__(self, cfg: MemoryConfig, spec: ConditionSpec, embedder, seed: int = 0):
        self.cfg = cfg
        self.spec = spec
        self.embedder = embedder
        self.seed = seed
        self.store = MemoryStore()
        self.consolidator = Consolidator(
            store=self.store, radius=cfg.consolidation_radius,
            mode=getattr(spec, "consolidation_mode", "adaptive"))
        self.working: list[MemoryRecord] = []
        self.full_history_chunks: list[str] = []
        self._clock = 0
        self._rng = np.random.default_rng(seed)

    # -- ingestion -----------------------------------------------------------

    def _novelty(self, emb: np.ndarray) -> float:
        existing = self.store.retrievable()
        if not existing:
            return 1.0
        mat = np.vstack([r.embedding for r in existing]).astype(np.float32)
        q = emb / (np.linalg.norm(emb) + 1e-12)
        m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sim = float(np.max(m @ q))
        return float(min(max(1.0 - sim, 0.0), 1.0))

    def ingest_turn(self, text: str, session_id: int, role: str = "user") -> None:
        self._clock += 1
        self.full_history_chunks.append(text)
        if not self.spec.use_memory or self.spec.mode == "full_history":
            return

        chunks = chunk_text(text, self.cfg.chunk_max_chars)
        if not chunks:
            return
        embs = self.embedder.encode(chunks)
        for chunk, emb in zip(chunks, embs):
            emb = np.asarray(emb, dtype=np.float32)
            nov = self._novelty(emb)
            rec = MemoryRecord(
                id=self.store.new_id(),
                text=chunk,
                embedding=emb,
                session_id=session_id,
                timestamp=self._clock,
                last_access=self._clock,
                tier=EPISODIC,
                novelty=nov,
            )
            rec.importance = imp.compute_importance(
                rec, self.cfg, self._clock,
                use_recency=self.spec.use_recency,
                use_novelty=self.spec.use_novelty,
                use_reuse=self.spec.use_reuse,
            )
            self._assign_tier(rec)
            rec.bits = imp.assign_bits(
                rec.importance, self.spec.allocation,
                base_bits=8 if self.spec.vector_quant == "int8" else 4,
                seed_val=self.seed,
                precision_threshold=self.cfg.precision_threshold,
            )
            self.store.add(rec)
            self._update_working(rec)
            if self.spec.consolidation and rec.tier == SEMANTIC:
                # Fold the semantic-tier record into a prototype (MDL two-part
                # code); the original is no longer stored independently.
                self.consolidator.consolidate(rec, self._clock)
                rec.folded = True
            if self.spec.eviction == "fifo":
                self._evict_fifo()
            elif self.spec.eviction == "utility":
                self._maintain_utility()

    def _evict_fifo(self) -> None:
        """Recency/FIFO forgetting: keep only the most recent ``retention_capacity``
        retrievable items, evicting the oldest by timestamp regardless of utility.
        This is the ablation of HAM's utility-driven promotion/forgetting."""
        cap = self.cfg.retention_capacity
        if cap is None or cap <= 0:
            return
        live = [r for r in self.store.retrievable()]
        if len(live) <= cap:
            return
        live.sort(key=lambda r: (r.timestamp, r.id))
        for r in live[:len(live) - cap]:
            r.evicted = True

    def _maintain_utility(self) -> None:
        """Utility-driven forgetting (paper Eq 6): re-score every retrievable item
        at the current logical time and drop those whose utility fell below the
        episodic threshold. Runs between turns as asynchronous maintenance.

        Gated on retrieval history: eviction needs the reuse signal from prior
        retrievals, so it is a no-op until at least one item has been retrieved.
        This avoids uninformed over-eviction during the initial ingest of a fresh
        history (e.g. the proof-of-concept's single-pass ingest, where no
        retrieval has yet occurred) and lets eviction act in a realistic
        multi-session lifetime where retrievals accumulate reuse between turns."""
        if not any(r.reuse > 0 for r in self.store.retrievable()):
            return
        for r in self.store.retrievable():
            u = imp.compute_importance(
                r, self.cfg, self._clock,
                use_recency=self.spec.use_recency,
                use_novelty=self.spec.use_novelty,
                use_reuse=self.spec.use_reuse)
            if u < self.cfg.episodic_threshold:
                r.evicted = True
                r.tier = EVICTED

    def _assign_tier(self, rec: MemoryRecord) -> None:
        if self.spec.tiering == "random":
            rec.tier = self._rng.choice([EPISODIC, SEMANTIC])
        else:
            tier = imp.assign_tier(rec.importance, self.cfg)
            if tier == EVICTED:
                rec.evicted = True
                rec.tier = EVICTED
            else:
                rec.tier = tier

    def _update_working(self, rec: MemoryRecord) -> None:
        self.working.append(rec)
        if len(self.working) > self.cfg.working_capacity:
            self.working = self.working[-self.cfg.working_capacity:]

    # -- retrieval / context build ------------------------------------------

    def build_context(self, query: str) -> tuple[str, dict]:
        """Return (context_string, diagnostics) for a query, with timers."""
        diag: dict = {"retrieval_latency_s": 0.0, "context_build_latency_s": 0.0,
                      "n_retrieved": 0, "retrieved_ids": [], "retrieved_texts": []}
        if not self.spec.use_memory or self.spec.mode == "none":
            return "", diag

        t_build0 = time.perf_counter()
        if self.spec.mode == "full_history":
            ctx = self._budget_join(self.full_history_chunks)
            diag["context_build_latency_s"] = time.perf_counter() - t_build0
            diag["n_retrieved"] = len(self.full_history_chunks)
            return ctx, diag

        # retrieval mode
        candidates = self.store.retrievable()
        q_emb = self.embedder.encode([query])[0]
        t_ret0 = time.perf_counter()
        hits = retrieve(query, np.asarray(q_emb, dtype=np.float32), candidates,
                        self.cfg.retrieval_k, method=self.spec.retrieval_method)
        diag["retrieval_latency_s"] = time.perf_counter() - t_ret0

        texts = []
        for rec, _score in hits:
            rec.reuse += 1
            rec.last_access = self._clock
            texts.append(rec.text)
            diag["retrieved_ids"].append(rec.id)
            diag["retrieved_texts"].append(rec.text)
        ctx = self._budget_join(texts)
        diag["context_build_latency_s"] = time.perf_counter() - t_build0
        diag["n_retrieved"] = len(texts)
        self._last_retrieved = [rec for rec, _ in hits]
        return ctx, diag

    def _budget_join(self, texts: list[str]) -> str:
        """Join text chunks, truncating to the configured token budget using the
        embedder-agnostic whitespace proxy (the backend does exact counting)."""
        budget = self.cfg.token_budget
        out, used = [], 0
        for t in texts:
            n = len(t.split())
            if used + n > budget and out:
                break
            out.append(t)
            used += n
        return "\n".join(out)

    def record_feedback(self, correct: bool) -> None:
        """Online predictive-utility update: retrieved items that co-occur with a
        correct answer gain utility (IB-style relevance to task success)."""
        for rec in getattr(self, "_last_retrieved", []):
            if correct:
                rec.helpful_hits += 1
            denom = rec.reuse + 2
            rec.predictive_utility = (rec.helpful_hits + 1) / denom

    # -- serialization / diagnostics ----------------------------------------

    def serialize(self, out_dir: str) -> serialize.ByteAccounting:
        """Physically write the store grouped by per-item precision and return
        real byte accounting summed across groups."""
        os.makedirs(out_dir, exist_ok=True)
        if self.spec.mode == "full_history" or not self.spec.use_memory:
            texts = list(self.full_history_chunks)
            metadata = [{"i": i} for i in range(len(texts))]
            return serialize.serialize_snapshot(
                out_dir, texts, None, metadata,
                text_codec_name=self.spec.text_codec, vector_quant_name="none",
                n_facts=max(len(texts), 1),
            )

        records = self.store.retrievable()
        if not records:
            return serialize.serialize_snapshot(out_dir, [], None, [], n_facts=1)

        # Group by assigned precision for mixed-precision (ham) or single group.
        groups: dict[str, list[MemoryRecord]] = {}
        for r in records:
            if self.spec.vector_quant in ("none", "pq"):
                key = self.spec.vector_quant
            else:
                key = f"int{r.bits}"
            groups.setdefault(key, []).append(r)

        total = serialize.ByteAccounting(0, 0, 0, 0, 0, 0, 0, "", self.spec.vector_quant)
        n_facts = sum(r.n_atomic_facts for r in records)
        for key, recs in sorted(groups.items()):
            texts = [r.text for r in recs]
            embs = np.vstack([r.embedding for r in recs]).astype(np.float32)
            meta = [r.to_meta() for r in recs]
            acc = serialize.serialize_snapshot(
                os.path.join(out_dir, key), texts, embs, meta,
                text_codec_name=self.spec.text_codec,
                vector_quant_name=("none" if key == "none" else ("pq" if key == "pq" else key)),
                n_facts=n_facts,
            )
            # Record per-item vector reconstruction error (paper Eq 8) on the
            # memory records. None for 'pq' (FAISS owns dequantization); 0.0
            # for 'none' (no quantization applied so x_hat == x). The field is
            # a pure diagnostic -- it is never read back into the bytes/quality
            # computation.
            if acc.per_item_quantization_error is not None:
                for r, e in zip(recs, acc.per_item_quantization_error):
                    r.quantization_error = float(e)
            elif key == "none":
                for r in recs:
                    r.quantization_error = 0.0
            total.logical_text_bytes += acc.logical_text_bytes
            total.logical_vector_bytes += acc.logical_vector_bytes
            total.physical_text_bytes += acc.physical_text_bytes
            total.physical_vector_bytes += acc.physical_vector_bytes
            total.physical_meta_bytes += acc.physical_meta_bytes
            total.n_items += acc.n_items
            total.text_codec = acc.text_codec
        total.n_facts = max(n_facts, 1)
        return total

    def diagnostics(self) -> dict:
        occ = self.store.tier_occupancy()
        return {
            "tier_working": len(self.working),
            "tier_episodic": occ.get(EPISODIC, 0),
            "tier_semantic": occ.get(SEMANTIC, 0),
            "n_records": len(self.store),
            "n_prototypes": self.consolidator.prototype_count(),
            "n_retrievable": len(self.store.retrievable()),
        }
