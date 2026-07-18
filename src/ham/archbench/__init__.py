"""Stage-F architecture memory-block compression experiment (toy model).

Identical toy language models are trained under different memory-block policies
(standard FlatMemory vs HAM-compressed). The memory policy is the SOLE
independent variable; corpus redundancy is the lever that isolates 'frequency'.
Headline: HAM/standard bytes-ratio vs redundancy (the slope is the
proof). See ``docs/ARCHBENCH_PROTOCOL.md``.
"""

from __future__ import annotations

from ..config import ArchBenchExperimentConfig
from .memory import FlatMemory, HamMemory, build_memory_store
from .mock import MockArchTrainer
from .protocol import CONDITIONS, ArchCheckpoint, checkpoint_steps
from .trainer import TorchArchTrainer

__all__ = [
    "CONDITIONS", "ArchCheckpoint", "checkpoint_steps", "FlatMemory", "HamMemory",
    "build_memory_store", "MockArchTrainer", "TorchArchTrainer", "build_trainer",
]


def build_trainer(cfg: ArchBenchExperimentConfig, condition: str, redundancy: float,
                  corpus=None, device: str = "cpu"):
    """Dispatch on ``cfg.archbench.trainer``. The mock trainer needs no corpus;
    the torch trainer requires torch + a corpus."""
    if cfg.archbench.trainer == "mock":
        return MockArchTrainer(cfg, condition, redundancy)
    if cfg.archbench.trainer == "torch":
        if corpus is None:
            raise ValueError("the torch archbench trainer requires a corpus")
        return TorchArchTrainer(cfg, condition, redundancy, corpus, device)
    raise ValueError(f"unknown trainer {cfg.archbench.trainer!r}")
