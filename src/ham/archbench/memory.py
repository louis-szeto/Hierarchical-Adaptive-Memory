"""Per-item memory-block stores for the stage-F toy architecture experiment.

These ARE the single independent variable. Every condition keeps the SAME raw
recent history -- a FIFO **window** of per-token hidden states. The conditions
differ only in how that window is *encoded* for retrieval and byte-counting:

- ``FlatMemory`` (standard): the window verbatim, float32. No compression.
- ``HamMemory``: the window consolidated into per-key **prototypes**, stored at
  int4 (``ham``) / float32 (``ham_uniform``). Leader clustering merges tokens of
  the same key, so the prototype count = distinct keys in the window -- which is
  exactly what the corpus redundancy lever controls (few dominant keys at high
  redundancy -> few prototypes -> small bytes; many distinct keys at low
  redundancy -> more prototypes). ``ham_no_consolidation`` keeps the window
  verbatim (= standard); ``ham_random_alloc`` clusters randomly (no semantic
  merging -> more prototypes).

``byte_size()`` is byte-honest (int4 prototypes counted at their packed-nibble
size via ``compression.vector_quant``). The int4 quantization error is applied at
consolidation (the prototype value is quantize->dequantize of its member mean),
so downstream quality reflects the compression loss, while ``read_kv`` returns
fast float32 tensors. Stored values are detached, so quantizing never blocks
training. Importing requires torch.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from ..compression.vector_quant import quantize

_FLOAT32_BYTES = 4


def _qd_mean(mean_np: np.ndarray, bits: int) -> np.ndarray:
    """Quantize->dequantize a single prototype mean -> int-bits fidelity (or
    float32 passthrough for bits>=32)."""
    if bits >= 32:
        return mean_np.astype(np.float32)
    q = quantize(mean_np[None, :], f"int{bits}")
    return q.dequantize()[0].astype(np.float32)


def _int4_prototype_bytes(n_prototypes: int, dim: int) -> int:
    """Physical bytes of ``n_prototypes`` int4-quantized D-dim vectors: packed
    nibbles + per-row float32 scale/zero metadata."""
    if n_prototypes == 0:
        return 0
    code_bytes = (n_prototypes * dim + 1) // 2
    meta = n_prototypes * 4 * 2
    return code_bytes + meta


class _WindowStore:
    """Shared FIFO window of recent per-token hidden states (float32, CPU)."""

    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.capacity = capacity
        self.window: list[torch.Tensor] = []

    def reset(self) -> None:
        self.window = []

    def _evict(self) -> None:
        if len(self.window) > self.capacity:
            self.window = self.window[-self.capacity:]

    def write(self, value: torch.Tensor, **_) -> None:
        self.window.append(value.detach().reshape(-1).to("cpu"))
        self._evict()

    def write_batch(self, rows: torch.Tensor, **_) -> None:
        for r in rows:
            self.window.append(r.detach().reshape(-1).to("cpu"))
        self._evict()

    def window_matrix(self) -> torch.Tensor:
        if not self.window:
            return torch.zeros((0, self.dim))
        return torch.stack(self.window, dim=0)


class FlatMemory(_WindowStore):
    """Standard baseline: window verbatim at float32. Never consolidates."""

    kind = "standard_memory"

    def consolidate(self) -> int:
        return 0

    def read_kv(self) -> torch.Tensor:
        return self.window_matrix()

    def occupancy(self) -> dict:
        return {"window": len(self.window), "prototypes": 0}

    def byte_size(self) -> int:
        return len(self.window) * self.dim * _FLOAT32_BYTES


class HamMemory(_WindowStore):
    """HAM memory: the window consolidated into per-key prototypes.

    Modes: ``ham`` (cosine leader-clustering + int4 prototypes), ``uniform``
    (clustering + float32 prototypes), ``no_consolidation`` (window verbatim,
    = standard), ``random_alloc`` (random clustering + int4 -- no semantic merge).
    """

    def __init__(self, dim: int, capacity: int = 512, radius: float = 0.25,
                 semantic_bits: int = 4, mode: str = "ham", seed: int = 0, **_):
        super().__init__(dim, capacity)
        self.radius = radius
        self.bits = semantic_bits
        self.mode = mode
        self._rng = random.Random(f"{seed}:{mode}")
        self.prototypes: list[dict] = []  # {"value": (D,) float32 qd mean, "members": int}

    @property
    def kind(self) -> str:
        return {"ham": "ham_memory", "uniform": "ham_uniform",
                "no_consolidation": "ham_no_consolidation",
                "random_alloc": "ham_random_alloc"}[self.mode]

    def reset(self) -> None:
        super().reset()
        self.prototypes = []

    def _precision_bits(self) -> int:
        return 32 if self.mode == "uniform" else self.bits

    def _nearest(self, v: torch.Tensor) -> dict | None:
        best, best_d = None, 2.0
        for p in self.prototypes:
            d = 1.0 - float(F.cosine_similarity(v.reshape(-1), p["value"].reshape(-1), dim=0))
            if d < best_d:
                best, best_d = p, d
        return best if (best is not None and best_d <= self.radius) else None

    def _new_proto(self, v: torch.Tensor) -> None:
        mean = _qd_mean(v.numpy().astype(np.float32), self._precision_bits())
        self.prototypes.append({"value": torch.from_numpy(mean.copy()), "members": 1})

    def _update_proto(self, p: dict, v: torch.Tensor) -> None:
        k = p["members"]
        mean = (p["value"].numpy() * k + v.numpy()) / (k + 1)
        mean = _qd_mean(mean.astype(np.float32), self._precision_bits())
        p["value"] = torch.from_numpy(mean.copy())
        p["members"] = k + 1

    def consolidate(self) -> int:
        """Re-cluster the current window into prototypes (refreshed each call)."""
        if self.mode == "no_consolidation":
            return 0
        self.prototypes = []
        for v in list(self.window):
            if self.mode == "random_alloc":
                # Random assignment: merge into a random existing prototype (or new).
                if self.prototypes and self._rng.random() < 0.5:
                    self._update_proto(self._rng.choice(self.prototypes), v)
                else:
                    self._new_proto(v)
            else:
                proto = self._nearest(v)
                if proto is None:
                    self._new_proto(v)
                else:
                    self._update_proto(proto, v)
        return len(self.prototypes)

    def read_kv(self) -> torch.Tensor:
        if self.mode == "no_consolidation" or not self.prototypes:
            return self.window_matrix()  # verbatim window (or before first consolidate)
        return torch.stack([p["value"] for p in self.prototypes], dim=0)

    def occupancy(self) -> dict:
        return {"window": len(self.window), "prototypes": len(self.prototypes)}

    def byte_size(self) -> int:
        if self.mode == "no_consolidation":
            return len(self.window) * self.dim * _FLOAT32_BYTES
        n = len(self.prototypes)
        if self.mode == "uniform":
            return n * self.dim * _FLOAT32_BYTES
        return _int4_prototype_bytes(n, self.dim)


def build_memory_store(condition: str, *, dim: int, capacity: int,
                       radius: float, semantic_bits: int, seed: int):
    """Factory: a MemoryStore for the condition, or None for ``no_memory``."""
    if condition == "no_memory":
        return None
    if condition == "standard_memory":
        return FlatMemory(dim=dim, capacity=capacity)
    mode = {"ham_memory": "ham", "ham_uniform": "uniform",
            "ham_no_consolidation": "no_consolidation",
            "ham_random_alloc": "random_alloc"}.get(condition)
    if mode is None:
        raise ValueError(f"unknown archbench condition {condition!r}")
    return HamMemory(dim=dim, capacity=capacity, radius=radius,
                     semantic_bits=semantic_bits, mode=mode, seed=seed)
