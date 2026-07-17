"""Deterministic mock archbench trainer: synthetic curves with NO torch.

Encodes the *expected* behavior so tests/CI and the ``archbench_smoke`` config
exercise the full plumbing deterministically. Outputs are flagged ``is_smoke`` and
watermarked ``SMOKE TEST`` -- never scientific results (mirrors the mock-trainer
discipline of the other experiments).

The synthetic model bakes in the thesis so the plumbing is meaningfully
non-degenerate:
- quality climbs to a per-condition ceiling (no_memory < memory conditions);
  standard/ham are iso-quality at a given redundancy.
- memory bytes / latency: HAM conditions compress relative to ``standard_memory``
  by a factor that GROWS with redundancy ``r`` (0 = no compression, ->1 = max).
  ``ham_no_consolidation`` does not compress (= standard); ``ham_uniform`` lacks
  int4 so compresses less than ``ham_memory``; ``ham_random_alloc`` clusters weakly.
"""

from __future__ import annotations

import math

from ..config import ArchBenchExperimentConfig
from .protocol import ArchCheckpoint, checkpoint_steps

_CEILING = {
    "no_memory": 0.75, "standard_memory": 0.95, "ham_memory": 0.95,
    "ham_uniform": 0.95, "ham_no_consolidation": 0.95, "ham_random_alloc": 0.94,
}


def _bytes_factor(condition: str, r: float) -> float:
    """ham_bytes / standard_bytes as a function of redundancy r in [0,1]."""
    if condition in ("no_memory", "standard_memory", "ham_no_consolidation"):
        return 1.0
    if condition == "ham_memory":
        return 1.0 - 0.6 * r            # item-reduction + int4
    if condition == "ham_uniform":
        return 1.0 - 0.3 * r            # item-reduction only (float32 prototypes)
    if condition == "ham_random_alloc":
        return 1.0 - 0.2 * r            # weak/lossy clustering
    return 1.0


def _items_factor(condition: str, r: float) -> float:
    """ham_items / standard_items (drives latency). Consolidation reduces item
    count independent of precision."""
    if condition in ("no_memory", "standard_memory", "ham_no_consolidation"):
        return 1.0
    if condition in ("ham_memory", "ham_uniform"):
        return 1.0 - 0.5 * r
    if condition == "ham_random_alloc":
        return 1.0 - 0.15 * r
    return 1.0


class MockArchTrainer:
    """No-torch synthetic trainer for one (condition x redundancy) cell."""

    def __init__(self, cfg: ArchBenchExperimentConfig, condition: str,
                 redundancy: float):
        self.ab = cfg.archbench
        self.condition = condition
        self.r = redundancy

    def run(self) -> list[ArchCheckpoint]:
        ab = self.ab
        steps = checkpoint_steps(ab.max_steps, ab.checkpoint_every)
        ceiling = _CEILING.get(self.condition, 0.9)
        cb = _bytes_factor(self.condition, self.r)
        ci = _items_factor(self.condition, self.r)
        rate = 3.0 / max(1, ab.max_steps)
        curve: list[ArchCheckpoint] = []
        for s in steps:
            tokens = s * ab.batch_size * ab.seq_len
            wall = s * 0.05
            loss = max(0.05, 3.0 * math.exp(-rate * s * 1.5)) if s else None
            quality = ceiling * (1.0 - math.exp(-rate * s))
            std_items = min(ab.capacity, s)
            std_bytes = std_items * ab.dim * 4
            if self.condition == "no_memory":
                mem_bytes, latency = 0, 1.0e-4
            else:
                mem_bytes = int(std_bytes * cb)
                latency = (std_items * ci) * 1.0e-4 + 1.0e-4
            curve.append(ArchCheckpoint(
                step=s, tokens_seen=tokens, wall_clock_s=wall, train_loss=loss,
                quality=quality, memory_bytes=mem_bytes, inference_latency_s=latency,
                redundancy=self.r, condition=self.condition, regime="pretrain"))
        return curve
