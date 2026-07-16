"""PyTorch components for the architecture-level HAM read/fusion/write path.

All modules operate on hidden states of shape ``(B, T, D)`` and an external memory
of shape ``(M, D)`` assembled from the tier interfaces. See the package docstring
for scope/honesty caveats. Import requires torch (guarded by the package).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Tier interfaces (working / episodic / semantic)
# --------------------------------------------------------------------------- #
@dataclass
class MemoryItem:
    value: torch.Tensor  # (D,)
    provenance: dict = field(default_factory=dict)  # turn/session/precision, etc.
    tier: str = "episodic"


class TierState:
    """Holds the three HAM tiers as tensors + provenance, with read/write and a
    (leader-clustering) consolidation used by the async interface.

    - Tier 0 (working): a bounded buffer of the most recent written vectors.
    - Tier 1 (episodic): high-precision mutable items with provenance.
    - Tier 2 (semantic): consolidated prototypes (running-mean vectors).
    """

    def __init__(self, dim: int, working_capacity: int = 8,
                 consolidation_radius: float = 0.25):
        self.dim = dim
        self.working_capacity = working_capacity
        self.consolidation_radius = consolidation_radius
        self.working: list[MemoryItem] = []
        self.episodic: list[MemoryItem] = []
        self.semantic: list[MemoryItem] = []
        self.events: list[dict] = []  # auditable write/promote/evict log

    def occupancy(self) -> dict:
        return {"working": len(self.working), "episodic": len(self.episodic),
                "semantic": len(self.semantic)}

    def read_kv(self) -> torch.Tensor:
        """Return the retrievable memory matrix (episodic + semantic), (M, D)."""
        items = self.episodic + self.semantic
        if not items:
            return torch.zeros((0, self.dim))
        return torch.stack([it.value for it in items], dim=0)

    def write(self, value: torch.Tensor, provenance: dict | None = None) -> None:
        """Post-block write hook: append a candidate item to episodic + working."""
        v = value.detach().reshape(-1)
        item = MemoryItem(value=v, provenance=provenance or {}, tier="episodic")
        self.episodic.append(item)
        self.working.append(item)
        if len(self.working) > self.working_capacity:
            self.working = self.working[-self.working_capacity:]
        self.events.append({"event": "write", "tier": "episodic",
                            "provenance": item.provenance})

    def consolidate(self) -> int:
        """Fold episodic items into semantic prototypes via leader clustering.
        Returns the number of items consolidated. Off the critical path."""
        moved = 0
        for it in list(self.episodic):
            proto = self._nearest_prototype(it.value)
            if proto is None:
                proto = MemoryItem(value=it.value.clone(),
                                   provenance={"members": 1}, tier="semantic")
                self.semantic.append(proto)
                self.events.append({"event": "promote", "tier": "semantic",
                                    "new_prototype": True})
            else:
                k = proto.provenance.get("members", 1)
                proto.value = (proto.value * k + it.value) / (k + 1)
                proto.provenance["members"] = k + 1
                self.events.append({"event": "promote", "tier": "semantic",
                                    "new_prototype": False})
            moved += 1
        self.episodic.clear()
        return moved

    def _nearest_prototype(self, v: torch.Tensor) -> MemoryItem | None:
        best, best_d = None, 2.0
        for p in self.semantic:
            d = 1.0 - float(F.cosine_similarity(v, p.value, dim=0))
            if d < best_d:
                best, best_d = p, d
        return best if (best is not None and best_d <= self.consolidation_radius) else None


# --------------------------------------------------------------------------- #
# Memory router
# --------------------------------------------------------------------------- #
class MemoryRouter(nn.Module):
    """Scores external memory against the current hidden state and selects top-k.

    A learned query/key projection makes the router *optionally trainable*; in the
    frozen PoC path the selection is detached (stop-gradient into the base model),
    analogous to the non-differentiable memory of Memorizing Transformers.
    """

    def __init__(self, dim: int, top_k: int = 4):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor):
        """hidden: (B, T, D); memory: (M, D).
        Returns (selected (B, k, D), weights (B, k), indices (B, k))."""
        B = hidden.shape[0]
        M = memory.shape[0]
        if M == 0:
            empty = hidden.new_zeros((B, 0, self.dim))
            return empty, hidden.new_zeros((B, 0)), \
                torch.zeros((B, 0), dtype=torch.long)
        q = self.q_proj(hidden.mean(dim=1))          # (B, D)
        k = self.k_proj(memory)                       # (M, D)
        scores = q @ k.t()                            # (B, M)
        kk = min(self.top_k, M)
        top_scores, top_idx = torch.topk(scores, kk, dim=1)   # (B, kk)
        weights = torch.softmax(top_scores, dim=1)            # (B, kk)
        selected = memory[top_idx]                            # (B, kk, D)
        return selected, weights, top_idx


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
class CrossAttentionFusion(nn.Module):
    """h' = h + CrossAttn(Q=h, K=V=serialize(m_retrieved)). RETRO / Memorizing-
    Transformers mechanism family (implemented analogue, not a reproduction)."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, hidden: torch.Tensor, memory_sel: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        if memory_sel.shape[1] == 0:
            return hidden
        # Scale the retrieved keys/values by the router's soft weights so the
        # (otherwise non-differentiable top-k) router still receives a gradient.
        if weights is not None:
            memory_sel = memory_sel * weights.unsqueeze(-1)
        out, _ = self.attn(hidden, memory_sel, memory_sel, need_weights=False)
        return hidden + out


class GatedResidualFusion(nn.Module):
    """h' = h + g ⊙ (W_m m), with g = σ(W_g[h; m]). The natural interface for a
    Mamba-style block, which has no attention sub-layer to attach cross-attn to."""

    def __init__(self, dim: int):
        super().__init__()
        self.w_m = nn.Linear(dim, dim)
        self.w_g = nn.Linear(2 * dim, dim)

    def forward(self, hidden: torch.Tensor, memory_sel: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        if memory_sel.shape[1] == 0:
            return hidden
        if weights is None:
            pooled = memory_sel.mean(dim=1)                       # (B, D)
        else:
            pooled = (memory_sel * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        pooled = pooled.unsqueeze(1).expand(-1, hidden.shape[1], -1)  # (B, T, D)
        g = torch.sigmoid(self.w_g(torch.cat([hidden, pooled], dim=-1)))
        return hidden + g * self.w_m(pooled)


# --------------------------------------------------------------------------- #
# Toy reasoning blocks (Transformer + Mamba-style recurrent)
# --------------------------------------------------------------------------- #
class ToyTransformerBlock(nn.Module):
    """A minimal self-attention + MLP block standing in for a reasoning block.
    NOT a real pretrained model -- used only to verify HAM's shapes/gradients."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class ToyRecurrentBlock(nn.Module):
    """A minimal gated recurrent block as a Mamba-style stand-in (Tier 0 = its
    recurrent state). NOT the Mamba architecture -- a shape/gradient stand-in."""

    def __init__(self, dim: int):
        super().__init__()
        self.cell = nn.GRUCell(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        state = x.new_zeros((B, D))
        outs = []
        for t in range(T):
            state = self.cell(x[:, t, :], state)
            outs.append(state)
        return self.out(torch.stack(outs, dim=1))


# --------------------------------------------------------------------------- #
# Asynchronous consolidation interface
# --------------------------------------------------------------------------- #
class AsyncConsolidationInterface:
    """Off-critical-path consolidation. In this prototype it is a synchronous
    stub with an explicit queue: ``schedule`` marks work; ``run_pending`` performs
    Tier1->Tier2 consolidation between turns so inference latency is unaffected."""

    def __init__(self, tiers: TierState):
        self.tiers = tiers
        self._pending = False

    def schedule(self) -> None:
        self._pending = True

    def run_pending(self) -> int:
        if not self._pending:
            return 0
        n = self.tiers.consolidate()
        self._pending = False
        return n


# --------------------------------------------------------------------------- #
# HAM block: read -> fuse -> base compute -> write
# --------------------------------------------------------------------------- #
class HAMBlock(nn.Module):
    """Wraps a base reasoning block with the HAM read/fusion/write path.

    modes:
      - ``frozen``: read-only PoC path. The whole forward runs under ``no_grad``
        and the memory read is detached (stop-gradient into the base model);
        no parameter receives gradients. Mirrors the frozen-weight PoC.
      - ``trainable``: the router/fusion params are trainable; the base block's
        parameters are frozen (``requires_grad=False``) unless ``train_base=True``.
        Gradients flow to the router/fusion (optionally-trainable stage-F variant).
    """

    def __init__(self, base_block: nn.Module, dim: int, *, fusion: str = "cross_attention",
                 top_k: int = 4, num_heads: int = 4):
        super().__init__()
        self.base_block = base_block
        self.dim = dim
        self.router = MemoryRouter(dim, top_k=top_k)
        if fusion == "cross_attention":
            self.fusion = CrossAttentionFusion(dim, num_heads=num_heads)
        elif fusion == "gated_residual":
            self.fusion = GatedResidualFusion(dim)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        self.fusion_kind = fusion

    def set_mode(self, mode: str, *, train_base: bool = False) -> None:
        if mode not in ("frozen", "trainable"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        base_req = (mode == "trainable" and train_base)
        for p in self.base_block.parameters():
            p.requires_grad_(base_req)
        ham_req = (mode == "trainable")
        for p in list(self.router.parameters()) + list(self.fusion.parameters()):
            p.requires_grad_(ham_req)

    def _forward_impl(self, hidden: torch.Tensor, tiers: TierState) -> torch.Tensor:
        memory = tiers.read_kv()
        if getattr(self, "mode", "frozen") == "frozen" and memory.shape[0] > 0:
            memory = memory.detach()  # non-differentiable memory (stop-gradient)
        selected, weights, idx = self.router(hidden, memory)
        fused = self.fusion(hidden, selected, weights)
        out = self.base_block(fused)
        # Post-block write hook: pool the block output as a candidate memory value.
        tiers.write(out.mean(dim=(0, 1)),
                    provenance={"fusion": self.fusion_kind,
                                "n_selected": int(selected.shape[1])})
        return out

    def forward(self, hidden: torch.Tensor, tiers: TierState) -> torch.Tensor:
        if getattr(self, "mode", "frozen") == "frozen":
            with torch.no_grad():
                return self._forward_impl(hidden, tiers)
        return self._forward_impl(hidden, tiers)
