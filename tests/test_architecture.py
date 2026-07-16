"""Architecture-level HAM prototype: import safety, shapes, grad/no-grad, lifecycle.

Torch is an optional dependency; every torch-dependent test is skipped (not failed)
when torch is unavailable, but importing the package must always be safe.
"""

import importlib.util

import pytest

TORCH = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not TORCH, reason="torch not installed")


def test_package_imports_without_touching_torch():
    import ham.architecture as arch

    assert isinstance(arch.TORCH_AVAILABLE, bool)


def test_touching_component_without_torch_fails_loudly(monkeypatch):
    import ham.architecture as arch

    if arch.TORCH_AVAILABLE:
        pytest.skip("torch is installed; loud-failure path exercised only when absent")
    with pytest.raises(RuntimeError):
        _ = arch.HAMBlock


@requires_torch
@pytest.mark.parametrize("block", ["transformer", "recurrent"])
@pytest.mark.parametrize("fusion", ["cross_attention", "gated_residual"])
def test_toy_demo_invariants(block, fusion):
    from ham.architecture.toy import run_toy_demo

    d = run_toy_demo(block=block, fusion=fusion, dim=32, batch=2, seq=5)
    # Shapes preserved end to end.
    assert d["shapes"]["frozen_shape_preserved"]
    assert d["shapes"]["trainable_shape_preserved"]
    # Frozen path: detached output, no grad.
    assert d["gradients"]["frozen_output_requires_grad"] is False
    # Trainable path: router + fusion get grads; frozen base does not.
    assert d["gradients"]["router_received_grad"] is True
    assert d["gradients"]["fusion_received_grad"] is True
    assert d["gradients"]["frozen_base_received_grad"] is False
    assert d["invariants_ok"] is True


@requires_torch
def test_memory_read_write_consolidate_lifecycle():
    import torch

    from ham.architecture.layer import (
        AsyncConsolidationInterface,
        HAMBlock,
        TierState,
        ToyTransformerBlock,
    )

    dim = 16
    ham = HAMBlock(ToyTransformerBlock(dim), dim)
    ham.set_mode("frozen")
    tiers = TierState(dim)
    consolidator = AsyncConsolidationInterface(tiers)

    assert tiers.read_kv().shape[0] == 0  # empty at start
    x = torch.randn(2, 4, dim)
    ham(x, tiers)
    occ = tiers.occupancy()
    assert occ["episodic"] == 1 and occ["working"] == 1 and occ["semantic"] == 0

    moved = consolidator.run_pending()
    assert moved == 0  # nothing scheduled yet
    consolidator.schedule()
    moved = consolidator.run_pending()
    assert moved == 1
    assert tiers.occupancy()["semantic"] == 1
    assert tiers.occupancy()["episodic"] == 0
    # A subsequent read now sees the consolidated prototype.
    assert tiers.read_kv().shape[0] == 1


@requires_torch
def test_router_topk_and_shapes():
    import torch

    from ham.architecture.layer import MemoryRouter

    router = MemoryRouter(dim=8, top_k=3)
    hidden = torch.randn(2, 5, 8)
    memory = torch.randn(10, 8)
    selected, weights, idx = router(hidden, memory)
    assert selected.shape == (2, 3, 8)
    assert weights.shape == (2, 3)
    assert idx.shape == (2, 3)
    # Empty memory => empty selection, no crash.
    sel0, w0, i0 = router(hidden, torch.zeros(0, 8))
    assert sel0.shape == (2, 0, 8)
