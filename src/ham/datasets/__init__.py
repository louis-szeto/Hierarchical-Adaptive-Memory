"""Dataset adapters: synthetic (local, deterministic), LongMemEval, LoCoMo."""

from __future__ import annotations

from ..config import DatasetConfig
from .base import DatasetAdapter, Example, Turn
from .locomo import LoCoMoAdapter
from .longmemeval import LongMemEvalAdapter
from .synthetic import SyntheticAdapter

__all__ = [
    "DatasetAdapter", "Example", "Turn",
    "SyntheticAdapter", "LongMemEvalAdapter", "LoCoMoAdapter", "build_dataset",
]


def build_dataset(cfg: DatasetConfig) -> DatasetAdapter:
    if cfg.name == "synthetic":
        return SyntheticAdapter(
            num_examples=cfg.num_examples, num_sessions=cfg.num_sessions,
            facts_per_session=cfg.facts_per_session,
            distractors_per_session=cfg.distractors_per_session, seed=cfg.seed,
        )
    if cfg.name == "longmemeval":
        return LongMemEvalAdapter(
            path=cfg.path, hf_repo=cfg.hf_repo, hf_file=cfg.hf_file,
            sample_limit=cfg.sample_limit, seed=cfg.seed,
        )
    if cfg.name == "locomo":
        return LoCoMoAdapter(path=cfg.path, sample_limit=cfg.sample_limit, seed=cfg.seed)
    raise ValueError(f"unknown dataset: {cfg.name!r}")
