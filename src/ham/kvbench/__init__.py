"""Stage-D KV-cache-compression experiment (real frozen model).

A frozen model's KV cache is compressed under different policies; the policy is
the sole independent variable and context redundancy is the lever isolating
'frequency'. Headline: HAM/full bytes- and latency-ratios vs redundancy (the slope
is the proof). See ``docs/KVBENCH_PROTOCOL.md``.
"""

from __future__ import annotations

from ..config import KVBenchExperimentConfig
from .mock import MockKVTrainer
from .protocol import CONDITIONS, KVResult

__all__ = ["CONDITIONS", "KVResult", "MockKVTrainer", "TorchKVTrainer", "build_trainer"]


def build_trainer(cfg: KVBenchExperimentConfig):
    """Dispatch on ``cfg.kvbench.trainer``. Mock needs no model; torch needs the
    HF backend (lazy torch, fails loudly)."""
    if cfg.kvbench.trainer == "mock":
        return MockKVTrainer(cfg)
    if cfg.kvbench.trainer == "torch":
        from .trainer import TorchKVTrainer
        return TorchKVTrainer(cfg)
    raise ValueError(f"unknown trainer {cfg.kvbench.trainer!r}")
