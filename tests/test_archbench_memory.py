"""Archbench memory stores (per-item window + prototypes): byte-honest accounting
and consolidation behaviour. Requires torch (skipped when torch is unavailable)."""

import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from ham.archbench.memory import (FlatMemory, HamMemory, _int4_prototype_bytes,
                                  build_memory_store)


def test_flat_memory_bytes_and_fifo():
    m = FlatMemory(dim=8, capacity=10)
    m.write_batch(torch.randn(5, 8))
    assert m.byte_size() == 5 * 8 * 4
    assert m.read_kv().shape == (5, 8)
    m.write_batch(torch.randn(30, 8))  # exceed capacity -> FIFO evict
    assert len(m.window) <= 10
    assert m.byte_size() == len(m.window) * 8 * 4


def test_int4_prototype_bytes():
    assert _int4_prototype_bytes(10, 8) == 40 + 80   # ceil(80/2) + 10*8 metadata
    assert _int4_prototype_bytes(0, 8) == 0


def test_ham_consolidates_frequent_items():
    # Many near-identical (frequent) tokens -> one prototype.
    m = HamMemory(dim=8, capacity=100, radius=0.3, semantic_bits=4,
                  mode="ham", seed=0)
    base = torch.tensor([1.0] * 8)
    m.write_batch(base.unsqueeze(0).repeat(20, 1) + 0.01 * torch.randn(20, 8))
    n = m.consolidate()
    occ = m.occupancy()
    assert n >= 1
    assert occ["prototypes"] >= 1
    # Compressed bytes far below the flat 20 * dim * 4.
    assert m.byte_size() < 20 * 8 * 4


def test_distinct_items_form_more_prototypes():
    # Distinct (rare) tokens -> many prototypes; this is the redundancy lever.
    m = HamMemory(dim=8, capacity=100, radius=0.05, semantic_bits=4,
                  mode="ham", seed=0)
    m.write_batch(torch.randn(20, 8) * 5)  # well-separated items
    m.consolidate()
    many = m.occupancy()["prototypes"]
    m2 = HamMemory(dim=8, capacity=100, radius=0.3, semantic_bits=4,
                   mode="ham", seed=0)
    m2.write_batch(torch.ones(20, 8) + 0.01 * torch.randn(20, 8))  # all similar
    m2.consolidate()
    few = m2.occupancy()["prototypes"]
    assert many > few


def test_ham_uniform_uses_float32_prototypes():
    m = HamMemory(dim=8, capacity=100, radius=0.3, semantic_bits=4,
                  mode="uniform", seed=0)
    m.write_batch(torch.ones(10, 8) + 0.01 * torch.randn(10, 8))
    m.consolidate()
    n = m.occupancy()["prototypes"]
    assert m.byte_size() == n * 8 * 4   # float32 prototypes, not int4


def test_no_consolidation_keeps_window():
    m = HamMemory(dim=8, capacity=100, radius=0.3, semantic_bits=4,
                  mode="no_consolidation", seed=0)
    m.write_batch(torch.randn(10, 8))
    assert m.consolidate() == 0
    assert m.occupancy()["prototypes"] == 0
    assert m.occupancy()["window"] == 10
    assert m.byte_size() == 10 * 8 * 4   # = standard (no compression)


def test_build_store_factory():
    assert build_memory_store("no_memory", dim=8, capacity=10, radius=0.2,
                              semantic_bits=4, seed=0) is None
    assert build_memory_store("standard_memory", dim=8, capacity=10, radius=0.2,
                              semantic_bits=4, seed=0).kind == "standard_memory"
    for cond in ("ham_memory", "ham_uniform", "ham_no_consolidation",
                 "ham_random_alloc"):
        s = build_memory_store(cond, dim=8, capacity=10, radius=0.2,
                               semantic_bits=4, seed=0)
        assert s is not None and s.kind == cond
