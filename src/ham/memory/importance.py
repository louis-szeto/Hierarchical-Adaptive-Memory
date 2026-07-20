"""Normalized importance scoring and deterministic tier / bit-budget assignment.

Importance combines six signals motivated by the evidence report:

    u = w_freq*frequency + w_reuse*reuse + w_recency*recency
        + w_novelty*novelty + w_pu*predictive_utility + w_stability*stability

Each signal is squashed into [0, 1] so weights are comparable and the final
importance is in [0, 1]. Recency uses the forgetting kernel R = exp(-t/S).
This is a rate-distortion-style governor: higher importance earns more storage
bits (see :func:`assign_bits`). It is a heuristic proxy, not a Shannon-optimal
allocation.
"""

from __future__ import annotations

import math

from ..config import MemoryConfig
from .store import EPISODIC, EVICTED, SEMANTIC, WORKING, MemoryRecord


def _squash_count(x: float, k: float = 3.0) -> float:
    """Map a non-negative count to [0, 1) with diminishing returns.

    ``k`` is the paper's squash kappa (Eq 5). The default of 3.0 reproduces the
    previous hardcoded behavior and is overridden by ``MemoryConfig.squash_kappa``
    when called via :func:`compute_importance`.
    """
    return 1.0 - math.exp(-x / k)


def recency_score(now: int, last_access: int, halflife: float) -> float:
    t = max(now - last_access, 0)
    if halflife <= 0:
        return 1.0 if t == 0 else 0.0
    return math.exp(-t / halflife)


def compute_importance(
    record: MemoryRecord, cfg: MemoryConfig, now: int, *, use_recency: bool = True,
    use_novelty: bool = True, use_reuse: bool = True
) -> float:
    """Deterministic normalized importance in [0, 1].

    The ``use_*`` flags drop individual signals (for the no_recency / no_novelty /
    no_reuse ablations) by zeroing their contribution *and* renormalizing weights
    so remaining signals still span [0, 1]. The squash kappa for frequency/reuse
    comes from ``cfg.squash_kappa`` (paper Eq 5).
    """
    kappa = cfg.squash_kappa
    freq = _squash_count(record.frequency, k=kappa)
    reuse = _squash_count(record.reuse, k=kappa) if use_reuse else 0.0
    rec = recency_score(now, record.last_access, cfg.recency_halflife) if use_recency else 0.0
    nov = record.novelty if use_novelty else 0.0
    pu = record.predictive_utility
    stab = record.stability

    terms = [
        (cfg.w_frequency, freq, True),
        (cfg.w_reuse, reuse, use_reuse),
        (cfg.w_recency, rec, use_recency),
        (cfg.w_novelty, nov, use_novelty),
        (cfg.w_predictive_utility, pu, True),
        (cfg.w_stability, stab, True),
    ]
    active_weight = sum(w for w, _, on in terms if on)
    if active_weight <= 0:
        return 0.0
    score = sum(w * v for w, v, on in terms if on) / active_weight
    return float(min(max(score, 0.0), 1.0))


def assign_tier(importance: float, cfg: MemoryConfig) -> str:
    """Deterministic tier from normalized importance."""
    if importance >= cfg.semantic_threshold:
        return SEMANTIC
    if importance >= cfg.episodic_threshold:
        return EPISODIC
    return EVICTED  # u < episodic_threshold -> dropped (paper Eq 6, utility-driven forgetting)


def assign_bits(
    importance: float, allocation: str, base_bits: int, seed_val: int = 0,
    precision_threshold: float = 0.66,
) -> int:
    """Map importance to per-item vector precision (a rate-distortion allocation).

    - ``ham``: importance >= ``precision_threshold`` -> 8 bits, else 4 bits. The
      threshold defaults to 0.66 (the previous hardcoded cutoff, paper Eq 7's
      rho); callers pass ``cfg.memory.precision_threshold`` to override.
    - ``uniform``: every item gets ``base_bits`` (isolates whether *variable*
      allocation, not mere quantization, is what matters).
    - ``random``: bits chosen pseudo-deterministically from importance hash
      (isolates whether *utility-driven* allocation beats arbitrary tiering).
    """
    if allocation == "uniform":
        return base_bits
    if allocation == "random":
        # Deterministic pseudo-random in {4, 8} seeded by the item.
        h = (hash((round(importance, 4), seed_val)) & 0xFFFF) / 0xFFFF
        return 8 if h > 0.5 else 4
    # ham
    if importance >= precision_threshold:
        return 8
    return 4
