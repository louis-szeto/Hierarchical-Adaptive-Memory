"""Utility-driven eviction (paper Eq 6): gated on retrieval history.

Eviction must not fire before any retrieval has occurred (no reuse signal), so a
fresh ingest is never over-evicted; it acts once retrievals accumulate reuse
between sessions, the regime Eq 6 assumes. No torch required.
"""
from ham.config import CompressionConfig, MemoryConfig
from ham.conditions import build_condition
from ham.embeddings import HashEmbedder
from ham.memory.ham import HAMemory


def test_utility_eviction_gated_until_retrieval():
    cfg = MemoryConfig()
    spec = build_condition("ham_memory", CompressionConfig())
    mem = HAMemory(cfg, spec, HashEmbedder(dim=32), seed=0)
    for i in range(6):
        mem.ingest_turn(f"fact number {i} about entity {i}", session_id=0)
    recs = mem.store.retrievable()
    assert recs and all(r.reuse == 0 for r in recs)  # no retrieval yet
    n = len(recs)
    mem._maintain_utility()  # gated: no reuse -> no-op
    assert len(mem.store.retrievable()) == n  # nothing evicted pre-retrieval
