"""Stage-F architecture memory-block experiment: types and constants.

A toy language model is trained (pre-training and/or fine-tuning) under different
memory-block policies. The memory policy is the SOLE independent variable; the
redundancy of the corpus is the lever that isolates 'frequency' as the mechanism.
See ``docs/ARCHBENCH_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Memory-block policies (conditions). The memory policy is the sole variable.
CONDITIONS = [
    "no_memory",           # plain toy LM, no memory adapter (floor)
    "standard_memory",     # FlatMemory: append-only float32, FIFO (reference)
    "ham_memory",          # tiered + cosine consolidation + int4 prototypes (treatment)
    "ham_uniform",         # consolidation but float32 prototypes (isolates precision)
    "ham_no_consolidation",  # episodic FIFO only (isolates consolidation / item-reduction)
    "ham_random_alloc",    # consolidation but random item->prototype assignment (isolates frequency clustering)
]
BASELINE = "standard_memory"
TREATMENT = "ham_memory"
REGIMES = ["pretrain", "finetune"]


@dataclass
class ArchCheckpoint:
    """One checkpoint of one (condition x redundancy x regime) training run."""

    step: int
    tokens_seen: int
    train_loss: float | None
    quality: float            # recall accuracy or next-token accuracy in [0, 1]
    memory_bytes: int         # byte-honest peak memory-block size during the eval stream
    redundancy: float         # corpus redundancy level (0 = uniform/low, ->1 = high)
    condition: str
    regime: str


def checkpoint_steps(max_steps: int, checkpoint_every: int) -> list[int]:
    """Steps at which the model is evaluated: 0 (untrained), every
    ``checkpoint_every`` steps, and always ``max_steps`` (final)."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    steps = set(range(0, max_steps + 1, checkpoint_every))
    steps.add(0)
    steps.add(max_steps)
    return sorted(steps)
