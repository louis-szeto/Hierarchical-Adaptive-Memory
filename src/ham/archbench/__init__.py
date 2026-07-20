"""Stage-F architecture memory-block compression experiment (toy model).

Identical toy language models are trained under different memory-block policies
(standard FlatMemory vs HAM-compressed). The memory policy is the SOLE
independent variable; corpus redundancy is the lever that isolates 'frequency'.
Headline: HAM/standard bytes-ratio vs redundancy (the slope is the
proof). See ``docs/ARCHBENCH_PROTOCOL.md``.

PyTorch is an optional dependency (CI installs only the ``zstd,dev`` extras, no
torch). Importing this package and any torch-free submodule (``task``, ``runner``,
``report``, ``mock``, ``protocol``) is safe without torch; touching a
torch-dependent symbol (the memory stores, the torch trainer) fails loudly with
install guidance, never a silent no-op. Mirrors ``ham.architecture``.
"""

from __future__ import annotations

import importlib.util

from ..config import ArchBenchExperimentConfig
from .mock import MockArchTrainer
from .protocol import CONDITIONS, ArchCheckpoint, checkpoint_steps

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

_INSTALL_HINT = (
    "PyTorch is required for the stage-F archbench torch components but is not "
    "installed. Install it with:  pip install torch  "
    "(or: pip install -e \".[hf]\" for the full research stack)."
)


def require_torch():
    """Raise loudly with install guidance if torch is unavailable."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(_INSTALL_HINT)


__all__ = [
    "CONDITIONS", "ArchCheckpoint", "checkpoint_steps", "FlatMemory", "HamMemory",
    "build_memory_store", "MockArchTrainer", "TorchArchTrainer", "build_trainer",
    "TORCH_AVAILABLE", "require_torch",
]


def build_trainer(cfg: ArchBenchExperimentConfig, condition: str, redundancy: float,
                  corpus=None, device: str = "cpu"):
    """Dispatch on ``cfg.archbench.trainer``. The mock trainer needs no corpus;
    the torch trainer requires torch + a corpus (lazy torch, fails loudly)."""
    if cfg.archbench.trainer == "mock":
        return MockArchTrainer(cfg, condition, redundancy)
    if cfg.archbench.trainer == "torch":
        if corpus is None:
            raise ValueError("the torch archbench trainer requires a corpus")
        require_torch()
        from .trainer import TorchArchTrainer
        return TorchArchTrainer(cfg, condition, redundancy, corpus, device)
    raise ValueError(f"unknown trainer {cfg.archbench.trainer!r}")


def __getattr__(name: str):
    # Lazily expose the torch-dependent symbols so `import ham.archbench` (and
    # the torch-free submodules the mock-path tests need) is safe without torch,
    # while touching a real component still fails loudly via require_torch().
    if name in {"FlatMemory", "HamMemory", "build_memory_store"}:
        require_torch()
        from . import memory
        return getattr(memory, name)
    if name == "TorchArchTrainer":
        require_torch()
        from . import trainer
        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
