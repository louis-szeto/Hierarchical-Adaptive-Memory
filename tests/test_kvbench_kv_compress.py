"""KV compressor: byte-accounting, representative selection, cross-layer
consistency. Requires torch (skipped when torch is unavailable)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from ham.kvbench.kv_compress import _float_bytes, compress_cache


def _legacy(n_layers=3, n_kv_heads=2, seq=16, head_dim=8):
    torch.manual_seed(0)
    return [(torch.randn(1, n_kv_heads, seq, head_dim), torch.randn(1, n_kv_heads, seq, head_dim))
            for _ in range(n_layers)]


def _cfg(**o):
    return SimpleNamespace(cluster_radius=o.get("cluster_radius", 0.25), kv_bits=4)


def test_full_kv_float_bytes():
    legacy = _legacy()
    comp, b, n = compress_cache(legacy, "full_kv", _cfg(), 0, keep_ratio=1.0)
    assert n == 16 and b == _float_bytes(16, 2, 8, 3)
    assert all(c[0].shape[2] == 16 for c in comp)   # all layers full seq


def test_uniform_quant_smaller_than_full():
    legacy = _legacy()
    _, fb, _ = compress_cache(legacy, "full_kv", _cfg(), 0, keep_ratio=1.0)
    _, ub, _ = compress_cache(legacy, "uniform_quant_kv", _cfg(), 0, keep_ratio=1.0)
    assert ub < fb


def test_ham_kv_keeps_budget_and_is_consistent():
    legacy = _legacy(seq=24)
    comp, b, n = compress_cache(legacy, "ham_kv", _cfg(cluster_radius=0.1), 0, keep_ratio=0.5)
    # n_target = 0.5 * 24 = 12; consistent per-layer count (DynamicCache rebuild)
    assert n <= 12
    assert all(c[0].shape[2] == n for c in comp)
    assert b < _float_bytes(24, 2, 8, 3)        # fewer positions + int4


def test_ham_kv_prioritizes_frequent_cluster():
    # One dominant cluster of 12 positions; kr=0.5 -> keep 6, filled from it.
    base = torch.randn(1, 2, 1, 8)
    block = base.repeat(1, 1, 12, 1) + 0.01 * torch.randn(1, 2, 12, 8)
    legacy = [(block.clone(), block.clone()) for _ in range(3)]
    _, _, n = compress_cache(legacy, "ham_kv", _cfg(cluster_radius=0.3), 0, keep_ratio=0.5)
    assert n == 6                                # n_target, from the frequent cluster


def test_ham_kv_lower_keep_ratio_keeps_fewer():
    legacy = _legacy(seq=40)
    _, _, n_hi = compress_cache(legacy, "ham_kv", _cfg(cluster_radius=0.1), 0, keep_ratio=0.75)
    _, _, n_lo = compress_cache(legacy, "ham_kv", _cfg(cluster_radius=0.1), 0, keep_ratio=0.25)
    assert n_lo <= n_hi


def test_eviction_consistent_and_keeps_fraction():
    legacy = _legacy(seq=20)
    comp, b, n = compress_cache(legacy, "random_evict_kv", _cfg(), 0, keep_ratio=0.5)
    assert n == 10                                   # 0.5 * 20
    assert all(c[0].shape[2] == 10 for c in comp)
    assert b == _float_bytes(10, 2, 8, 3)            # float32 eviction


def test_unknown_condition_raises():
    legacy = _legacy()
    try:
        compress_cache(legacy, "bogus", _cfg(), 0, keep_ratio=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass
