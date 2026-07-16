"""Optional PyTorch prototype of the *architecture-level* HAM layer (stage F).

This package is the concrete skeleton for the "full HAM architecture" described
in the research addendum §6 -- a memory read/fusion/write path that attaches to a
Transformer or Mamba-style reasoning block, with a memory router, cross-attention
and gated-residual fusion, working/episodic/semantic tier interfaces, a post-block
write/update hook, frozen (stop-gradient) vs trainable modes, and an
asynchronous-consolidation interface.

Scope and honesty (must be preserved):
- This is a *prototype* exercised only by unit tests and a minimal toy integration
  (``ham.architecture.toy``). It is **not** evaluated on publication benchmarks and
  is explicitly separate from the runnable ``external_context`` PoC.
- We do **not** implement generic hidden-state injection into arbitrary Hugging
  Face models. The toy attaches HAM to *self-contained toy blocks* whose shapes and
  gradient behavior we can verify; wiring HAM into a specific pretrained model's
  internals is left as future work.
- PyTorch is an optional dependency. Importing anything that needs torch when torch
  is not installed fails loudly with install guidance (never a silent no-op).
"""

from __future__ import annotations

import importlib.util

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

_INSTALL_HINT = (
    "PyTorch is required for the HAM architecture prototype but is not installed. "
    "Install the optional extra:  pip install -e \".[hf]\"  (or: pip install torch)."
)


def require_torch():
    """Return the imported ``torch`` module or raise loudly if unavailable."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(_INSTALL_HINT)
    import torch

    return torch


__all__ = ["TORCH_AVAILABLE", "require_torch"]


def __getattr__(name: str):
    # Lazily expose the torch-dependent symbols so `import ham.architecture` is
    # safe without torch, but touching a real component fails loudly.
    if name in {
        "MemoryRouter", "CrossAttentionFusion", "GatedResidualFusion",
        "TierState", "HAMBlock", "ToyTransformerBlock", "ToyRecurrentBlock",
        "AsyncConsolidationInterface",
    }:
        require_torch()
        from . import layer

        return getattr(layer, name)
    if name in {"run_toy_demo"}:
        require_torch()
        from . import toy

        return getattr(toy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
