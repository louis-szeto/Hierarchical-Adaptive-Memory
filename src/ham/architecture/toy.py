"""Minimal executable toy integration for the architecture-level HAM layer.

Demonstrates -- on self-contained toy blocks, NOT a pretrained model -- the three
things requirement 2 asks a reader to be able to verify:

  1. shapes: a HAM-wrapped block preserves ``(B, T, D)`` end to end;
  2. gradient / no-grad behavior: ``frozen`` mode produces a detached output and
     leaves all parameters grad-free; ``trainable`` mode routes gradients to the
     router/fusion parameters while the (frozen) base block receives none;
  3. the memory read/write lifecycle: writes land in the working+episodic tiers,
     the async interface consolidates episodic -> semantic between "turns", and a
     later read sees the consolidated prototypes.

``run_toy_demo`` returns a JSON-serializable dict so the CLI (`ham arch-demo`) and
the unit tests can assert on the same evidence. Nothing here touches the runnable
``external_context`` PoC or any benchmark.
"""

from __future__ import annotations

import torch

from .layer import (
    AsyncConsolidationInterface,
    HAMBlock,
    ToyRecurrentBlock,
    ToyTransformerBlock,
    TierState,
)


def _block(kind: str, dim: int) -> torch.nn.Module:
    if kind == "transformer":
        return ToyTransformerBlock(dim)
    if kind == "recurrent":
        return ToyRecurrentBlock(dim)
    raise ValueError(f"unknown toy block kind {kind!r}")


def run_toy_demo(*, dim: int = 32, batch: int = 2, seq: int = 5,
                 block: str = "transformer", fusion: str = "cross_attention",
                 seed: int = 0) -> dict:
    """Exercise the HAM layer end to end and return an evidence dict.

    The dict is JSON-serializable and is the single source of truth for both the
    ``arch-demo`` CLI output and the architecture unit tests.
    """
    torch.manual_seed(seed)
    base = _block(block, dim)
    ham = HAMBlock(base, dim, fusion=fusion)
    tiers = TierState(dim)
    consolidator = AsyncConsolidationInterface(tiers)

    x = torch.randn(batch, seq, dim)

    # --- Frozen (read-only PoC-style) path: no grads anywhere. ------------- #
    ham.set_mode("frozen")
    out_frozen = ham(x, tiers)
    frozen_shape_ok = tuple(out_frozen.shape) == (batch, seq, dim)
    frozen_requires_grad = bool(out_frozen.requires_grad)
    occ_after_frozen = tiers.occupancy()

    # A first write with no memory present, then consolidate between "turns".
    consolidator.schedule()
    consolidated = consolidator.run_pending()
    occ_after_consolidate = tiers.occupancy()

    # A second frozen forward now reads back the consolidated prototypes.
    out_frozen2 = ham(x, tiers)
    read_saw_memory = tiers.read_kv().shape[0] > 0

    # --- Trainable path: router/fusion get grads, frozen base does not. ---- #
    x2 = torch.randn(batch, seq, dim, requires_grad=False)
    ham.set_mode("trainable", train_base=False)
    out_trainable = ham(x2, tiers)
    trainable_shape_ok = tuple(out_trainable.shape) == (batch, seq, dim)
    loss = out_trainable.pow(2).mean()
    loss.backward()

    def _grad_present(module: torch.nn.Module) -> bool:
        return any(p.grad is not None and torch.any(p.grad != 0) for p in module.parameters())

    router_has_grad = _grad_present(ham.router)
    fusion_has_grad = _grad_present(ham.fusion)
    base_has_grad = any(p.grad is not None for p in ham.base_block.parameters())

    return {
        "config": {"dim": dim, "batch": batch, "seq": seq, "block": block,
                   "fusion": fusion, "seed": seed},
        "shapes": {
            "input": [batch, seq, dim],
            "frozen_output": list(out_frozen.shape),
            "trainable_output": list(out_trainable.shape),
            "frozen_shape_preserved": frozen_shape_ok,
            "trainable_shape_preserved": trainable_shape_ok,
        },
        "gradients": {
            "frozen_output_requires_grad": frozen_requires_grad,
            "router_received_grad": router_has_grad,
            "fusion_received_grad": fusion_has_grad,
            "frozen_base_received_grad": base_has_grad,
        },
        "memory_lifecycle": {
            "occupancy_after_first_forward": occ_after_frozen,
            "items_consolidated": consolidated,
            "occupancy_after_consolidate": occ_after_consolidate,
            "read_saw_memory_after_consolidate": read_saw_memory,
            "event_count": len(tiers.events),
        },
        "invariants_ok": (
            frozen_shape_ok and trainable_shape_ok
            and not frozen_requires_grad
            and router_has_grad and fusion_has_grad and not base_has_grad
        ),
    }


__all__ = ["run_toy_demo"]
