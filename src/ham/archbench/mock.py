"""Deterministic mock archbench trainer: synthetic curves with NO torch.

Encodes the *expected* behavior so tests/CI and the ``archbench_smoke`` config
exercise the full plumbing deterministically. Outputs are flagged ``is_smoke`` and
watermarked ``SMOKE TEST`` -- never scientific results (mirrors the mock-trainer
discipline of the other experiments).

The synthetic model bakes in the thesis so the plumbing is meaningfully
non-degenerate:
- quality climbs to a per-condition ceiling (no_memory < memory conditions);
  standard/ham are iso-quality at a given redundancy.
- memory bytes: HAM conditions compress relative to ``standard_memory`` by a
  factor that GROWS with redundancy ``r`` (0 = no compression, ->1 = max).
  ``ham_no_consolidation`` does not compress (= standard); ``ham_uniform`` lacks
  int4 so compresses less than ``ham_memory``; ``ham_random_alloc`` clusters weakly.
- drift_rms: a synthetic sqrt(step) curve scaled per condition (HAM slightly
  higher than standard, mirroring the real torch toy result). Clearly synthetic;
  only the torch trainer records real measured drift.
"""

from __future__ import annotations

import math

from ..config import ArchBenchExperimentConfig
from .protocol import ArchCheckpoint, checkpoint_steps

_CEILING = {
    "no_memory": 0.75, "standard_memory": 0.95, "ham_memory": 0.95,
    "ham_uniform": 0.95, "ham_no_consolidation": 0.95, "ham_random_alloc": 0.94,
}

# Per-condition multiplicative scale on the synthetic drift curve. The toy
# HAM memory block has more learnable parameters (router + fusion + memory
# encoding) than the standard one, so its drift is slightly higher at any given
# step. Ordered so the post-hoc ratios are finite and meaningful.
_DRIFT_SCALE = {
    "no_memory": 0.85, "standard_memory": 1.00, "ham_memory": 1.01,
    "ham_uniform": 1.02, "ham_no_consolidation": 1.00, "ham_random_alloc": 1.03,
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
        drift_scale = _DRIFT_SCALE.get(self.condition, 1.0)
        rate = 3.0 / max(1, ab.max_steps)
        curve: list[ArchCheckpoint] = []
        for s in steps:
            tokens = s * ab.batch_size * ab.seq_len
            loss = max(0.05, 3.0 * math.exp(-rate * s * 1.5)) if s else None
            quality = ceiling * (1.0 - math.exp(-rate * s))
            std_items = min(ab.capacity, s)
            std_bytes = std_items * ab.dim * 4
            mem_bytes = 0 if self.condition == "no_memory" else int(std_bytes * cb)
            # SYNTHETIC L2 weight drift: a smooth sqrt(step) curve scaled per
            # condition so the post-hoc ratios are finite and ordered (HAM
            # slightly higher than standard, mirroring the real toy result).
            # Clearly synthetic -- only the torch trainer records real drift.
            drift_rms = drift_scale * (0.5 * math.sqrt(s)) if s else 0.0
            curve.append(ArchCheckpoint(
                step=s, tokens_seen=tokens, train_loss=loss,
                quality=quality, memory_bytes=mem_bytes,
                redundancy=self.r, condition=self.condition, regime="pretrain",
                drift_rms=drift_rms))
        return curve
