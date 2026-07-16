"""LLM backends: deterministic mock (default, CI-safe) and real Hugging Face."""

from __future__ import annotations

from ..config import BackendConfig
from .base import Backend, GenerationResult
from .mock import MockBackend

__all__ = ["Backend", "GenerationResult", "MockBackend", "build_backend"]


def build_backend(cfg: BackendConfig) -> Backend:
    if cfg.kind == "mock":
        return MockBackend(model_id=cfg.model_id, max_new_tokens=cfg.max_new_tokens, seed=cfg.seed)
    if cfg.kind == "hf":
        from .hf import HFBackend  # imported lazily so torch is not required for mock runs

        return HFBackend(cfg)
    raise ValueError(f"unknown backend kind: {cfg.kind!r}")
