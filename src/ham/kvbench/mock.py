"""Deterministic mock KV trainer: synthetic Pareto curves with NO torch.

Encodes the *expected* behavior so tests/CI and ``kvbench_smoke`` exercise the
plumbing deterministically: each position-reducing condition is evaluated across
the ``keep_ratios`` sweep (iso-quality Pareto). HAM's advantage grows with
redundancy (fewer representatives for the same budget) and its curve dominates on
the quality-vs-bytes Pareto. Outputs are ``is_smoke`` / ``SMOKE TEST`` -- never
scientific results.
"""

from __future__ import annotations

from ..config import KVBenchExperimentConfig
from .protocol import KVResult


def _npos_factor(cond: str, r: float, kr: float) -> float:
    if cond == "full_kv":
        return 1.0
    if cond == "uniform_quant_kv":
        return 1.0                       # all positions
    if cond == "ham_kv":
        return kr * (1.0 - 0.5 * r)      # redundant KV -> fewer reps for the budget
    return kr                            # h2o / random / ham_no_cluster: budget = kr


def _precision_factor(cond: str) -> float:
    # int4 conditions ~1/8 per-element of float32 (K+V) -> ~0.25 overall vs float
    return 0.25 if cond in ("ham_kv", "uniform_quant_kv", "ham_no_cluster") else 1.0


def _agreement(cond: str, r: float, kr: float) -> float:
    base = {"full_kv": 1.0, "ham_kv": 0.985, "uniform_quant_kv": 0.975,
            "h2o_kv": 0.95, "random_evict_kv": 0.90, "ham_no_cluster": 0.92}[cond]
    pen = 0.0 if cond in ("full_kv", "uniform_quant_kv") else 0.25 * (1.0 - kr)
    if cond == "ham_kv":                 # HAM quality rises with redundancy (lossless dedup)
        pen -= 0.08 * r
    return max(0.0, min(1.0, base - max(0.0, pen)))


def _accuracy(cond: str, r: float, kr: float) -> float:
    base = {"full_kv": 0.90, "ham_kv": 0.90, "uniform_quant_kv": 0.895,
            "h2o_kv": 0.88, "random_evict_kv": 0.83, "ham_no_cluster": 0.85}[cond]
    pen = 0.0 if cond in ("full_kv", "uniform_quant_kv") else 0.30 * (1.0 - kr)
    if cond == "ham_kv":
        pen -= 0.10 * r
    return max(0.0, min(1.0, base - max(0.0, pen)))


class MockKVTrainer:
    def __init__(self, cfg: KVBenchExperimentConfig):
        self.cfg = cfg
        self.kb = cfg.kvbench

    def run(self) -> list[KVResult]:
        kb = self.kb
        full_bytes = int(kb.mock_full_bytes_per_position * kb.context_len)
        results: list[KVResult] = []
        for r in kb.redundancy_levels:
            for ci in range(min(kb.n_contexts, 4)):
                for cond in kb.conditions:
                    kr_list = [1.0] if cond in ("full_kv", "uniform_quant_kv") else list(kb.keep_ratios)
                    for kr in kr_list:
                        nf = _npos_factor(cond, r, kr)
                        bf = nf * _precision_factor(cond)
                        results.append(KVResult(
                            condition=cond, redundancy=r, keep_ratio=kr, context_id=ci,
                            kv_bytes=int(full_bytes * bf),
                            n_positions=max(1, int(kb.context_len * nf)),
                            quality_agreement=_agreement(cond, r, kr),
                            quality_accuracy=_accuracy(cond, r, kr)))
        return results
