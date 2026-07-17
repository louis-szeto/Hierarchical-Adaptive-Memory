"""Toy language model for the stage-F memory-block experiment.

A small transformer LM (embedding -> N ToyTransformerBlocks -> LM head). At one
configurable layer, if the condition has a memory block, the model does the
HAM read -> fuse -> base -> write path inline (reusing ``MemoryRouter`` +
``CrossAttentionFusion`` from ``ham.architecture.layer``), driving an external
``MemoryStore``. The memory policy (FlatMemory vs HamMemory) is the SOLE
difference between conditions; the base architecture, data, and optimizer are
identical.

The memory accumulates one pooled item per forward across the training run (FIFO
at capacity, consolidated periodically by the trainer), so its size is a
meaningful, byte-honest quantity we can compare across conditions. Importing
requires torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..architecture.layer import CrossAttentionFusion, MemoryRouter, ToyTransformerBlock
from .memory import build_memory_store
from .task import PAD


class ToyMemoryLM(nn.Module):
    def __init__(self, cfg, condition: str):
        super().__init__()
        ab = cfg.archbench
        self.condition = condition
        self.has_memory = condition != "no_memory"
        self.dim = ab.dim
        self.embed = nn.Embedding(ab.vocab, ab.dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.zeros(1, ab.seq_len + 1, ab.dim))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [ToyTransformerBlock(ab.dim, ab.n_heads) for _ in range(ab.n_layers)])
        if self.has_memory:
            self.memory_layer = min(ab.memory_layer, ab.n_layers - 1)
            self.router = MemoryRouter(ab.dim, top_k=ab.top_k)
            self.fusion = CrossAttentionFusion(ab.dim, num_heads=ab.n_heads)
        else:
            self.memory_layer = -1
        self.head = nn.Linear(ab.dim, ab.vocab)
        self._store_kwargs = dict(capacity=ab.capacity, radius=ab.consolidation_radius,
                                  semantic_bits=ab.semantic_bits)
        self._seed = cfg.seed

    def new_store(self):
        if not self.has_memory:
            return None
        return build_memory_store(self.condition, dim=self.dim, seed=self._seed,
                                  **self._store_kwargs)

    def forward(self, input_ids: torch.Tensor, store=None, write: bool = True) -> torch.Tensor:
        T = input_ids.shape[1]
        h = self.embed(input_ids) + self.pos[:, :T]
        for i, blk in enumerate(self.blocks):
            if self.has_memory and i == self.memory_layer and store is not None:
                mem = store.read_kv().to(h.device)
                sel, w, _ = self.router(h, mem)
                h = self.fusion(h, sel, w)
            h = blk(h)
            if self.has_memory and i == self.memory_layer and store is not None and write:
                # Per-item write: store every token's hidden state (batch x seq),
                # so frequent tokens (repeated keys) produce repeated, consolidable
                # representations -- making corpus redundancy visible in the memory.
                store.write_batch(h.reshape(-1, self.dim))
        return self.head(h)
