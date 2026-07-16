"""Memory record model and the tiered store (working / episodic / semantic)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WORKING = "working"
EPISODIC = "episodic"
SEMANTIC = "semantic"
TIERS = (WORKING, EPISODIC, SEMANTIC)


@dataclass
class MemoryRecord:
    id: int
    text: str
    embedding: np.ndarray
    session_id: int = 0
    timestamp: int = 0  # turn index at insertion / last update
    last_access: int = 0
    tier: str = EPISODIC
    frequency: int = 1  # times this content (or a near-duplicate) was observed
    reuse: int = 0  # times retrieved into a prompt
    helpful_hits: int = 0  # retrievals that co-occurred with a correct answer
    novelty: float = 1.0  # 1 - max cosine sim to prior records at insertion
    stability: float = 0.5  # grows with confirmation / consolidation
    predictive_utility: float = 0.5  # IB-style relevance proxy, updated online
    importance: float = 0.0  # normalized [0, 1]
    bits: int = 8  # per-item vector precision assigned by allocation policy
    is_prototype: bool = False
    folded: bool = False  # consolidated into a prototype; no longer independently stored
    evicted: bool = False  # removed by an eviction policy (recency_fifo forgetting)
    members: list[int] = field(default_factory=list)  # episodic ids under a prototype
    n_atomic_facts: int = 1

    def to_meta(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "tier": self.tier,
            "frequency": self.frequency,
            "reuse": self.reuse,
            "novelty": round(self.novelty, 6),
            "stability": round(self.stability, 6),
            "predictive_utility": round(self.predictive_utility, 6),
            "importance": round(self.importance, 6),
            "bits": self.bits,
            "is_prototype": self.is_prototype,
            "members": self.members,
            "n_atomic_facts": self.n_atomic_facts,
        }


class MemoryStore:
    """Holds all records; provides tier views and matrix assembly for retrieval."""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []
        self._next_id = 0

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)

    def by_tier(self, tier: str) -> list[MemoryRecord]:
        return [r for r in self.records if r.tier == tier]

    def tier_occupancy(self) -> dict[str, int]:
        """Occupancy of the *retained* store (folded originals excluded)."""
        occ = {t: 0 for t in TIERS}
        for r in self.records:
            if r.folded or r.evicted:
                continue
            occ[r.tier] = occ.get(r.tier, 0) + 1
        return occ

    def retrievable(self) -> list[MemoryRecord]:
        """Episodic + semantic records participate in vector retrieval. Records
        folded into a prototype or evicted by a forgetting policy are excluded."""
        return [r for r in self.records
                if r.tier in (EPISODIC, SEMANTIC) and not r.folded and not r.evicted]

    def embedding_matrix(self, records: list[MemoryRecord]) -> np.ndarray:
        if not records:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([r.embedding for r in records]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.records)
