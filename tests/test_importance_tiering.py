import numpy as np

from ham.config import MemoryConfig
from ham.memory import importance as imp
from ham.memory.store import EPISODIC, EVICTED, SEMANTIC, MemoryRecord


def _rec(**kw):
    base = dict(id=0, text="x", embedding=np.ones(4, dtype=np.float32))
    base.update(kw)
    return MemoryRecord(**base)


def test_importance_is_deterministic_and_bounded():
    cfg = MemoryConfig()
    r = _rec(frequency=3, reuse=2, novelty=0.8, stability=0.7, predictive_utility=0.6,
             last_access=5)
    a = imp.compute_importance(r, cfg, now=5)
    b = imp.compute_importance(r, cfg, now=5)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_recency_kernel_monotonic():
    # More elapsed time => lower recency.
    r1 = imp.recency_score(now=10, last_access=10, halflife=5)
    r2 = imp.recency_score(now=10, last_access=5, halflife=5)
    assert r1 > r2
    assert 0.0 <= r2 <= r1 <= 1.0


def test_tier_thresholds_monotone():
    cfg = MemoryConfig(semantic_threshold=0.6, episodic_threshold=0.3)
    assert imp.assign_tier(0.9, cfg) == SEMANTIC
    assert imp.assign_tier(0.45, cfg) == EPISODIC
    assert imp.assign_tier(0.1, cfg) == EVICTED  # u < episodic_threshold -> evicted (Eq 6)


def test_ablation_flags_change_score():
    cfg = MemoryConfig()
    r = _rec(frequency=1, reuse=0, novelty=1.0, last_access=0)
    full = imp.compute_importance(r, cfg, now=100, use_recency=True)
    no_rec = imp.compute_importance(r, cfg, now=100, use_recency=False)
    # With large elapsed time recency ~ 0; dropping it should raise the average.
    assert no_rec != full


def test_bit_allocation_policies():
    assert imp.assign_bits(0.9, "ham", base_bits=8) == 8
    assert imp.assign_bits(0.1, "ham", base_bits=8) == 4
    assert imp.assign_bits(0.1, "uniform", base_bits=8) == 8
    assert imp.assign_bits(0.9, "uniform", base_bits=4) == 4
    # Random is deterministic given the same inputs.
    assert imp.assign_bits(0.5, "random", base_bits=8, seed_val=1) == \
        imp.assign_bits(0.5, "random", base_bits=8, seed_val=1)
