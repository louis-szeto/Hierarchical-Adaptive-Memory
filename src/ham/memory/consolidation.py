"""Online consolidation of episodic records into semantic prototypes.

We use streaming leader clustering: each incoming embedding either joins the
nearest existing prototype (if within ``consolidation_radius`` cosine distance)
or seeds a new prototype. A prototype stores the running mean embedding and the
list of member ids -- the MDL two-part-code view: the prototype is the shared
hypothesis H and per-episode residuals are D|H (the residual norm is tracked as
a stability signal).

This is *inspired by* systems consolidation (episodic -> semantic), not a model
of the brain.
"""

from __future__ import annotations

import numpy as np

from .store import SEMANTIC, MemoryRecord, MemoryStore


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


class Consolidator:
    """Online (adaptive) or static consolidation of episodic records.

    ``mode='adaptive'`` is HAM's default: prototypes update their running-mean
    embedding, accumulate members, and grow more stable as evidence accrues.
    ``mode='static'`` freezes each prototype at creation (no running-mean update,
    no stability growth, no membership merging) -- the *static_prototype* baseline
    isolating the value of HAM's *adaptive* consolidation over time.
    """

    def __init__(self, store: MemoryStore, radius: float = 0.25,
                 mode: str = "adaptive") -> None:
        self.store = store
        self.radius = radius
        self.mode = mode
        self.prototypes: list[MemoryRecord] = []

    def _nearest(self, emb: np.ndarray) -> tuple[MemoryRecord | None, float]:
        best, best_d = None, 2.0
        for proto in self.prototypes:
            d = _cosine_distance(emb, proto.embedding)
            if d < best_d:
                best, best_d = proto, d
        return best, best_d

    def consolidate(self, record: MemoryRecord, now: int) -> MemoryRecord:
        """Fold ``record`` into a prototype (creating one if needed). Returns the
        prototype it was assigned to."""
        proto, dist = self._nearest(record.embedding)
        # Static mode: never merge into an existing prototype; each consolidated
        # item seeds a frozen prototype that is never updated afterward.
        if self.mode == "static" or proto is None or dist > self.radius:
            proto = MemoryRecord(
                id=self.store.new_id(),
                text=record.text,
                embedding=record.embedding.astype(np.float32).copy(),
                session_id=record.session_id,
                timestamp=now,
                last_access=now,
                tier=SEMANTIC,
                frequency=record.frequency,
                novelty=record.novelty,
                stability=0.6,
                predictive_utility=record.predictive_utility,
                is_prototype=True,
                members=[record.id],
                n_atomic_facts=record.n_atomic_facts,
            )
            self.prototypes.append(proto)
            self.store.add(proto)
            return proto

        # Merge: running mean, accumulate frequency/stability, keep exemplar text.
        k = len(proto.members)
        proto.embedding = ((proto.embedding * k + record.embedding) / (k + 1)).astype(np.float32)
        proto.members.append(record.id)
        proto.frequency += record.frequency
        proto.n_atomic_facts += record.n_atomic_facts
        proto.last_access = now
        # More members => more confirmed => more stable (saturating).
        proto.stability = min(1.0, 0.6 + 0.1 * len(proto.members))
        # Keep the longest member text as the exemplar (most information).
        if len(record.text) > len(proto.text):
            proto.text = record.text
        return proto

    def prototype_count(self) -> int:
        return len(self.prototypes)
