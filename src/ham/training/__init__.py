"""Stage-C fine-tuning experiment package.

NEW DESIGN (per-leg training with HAM injected into ham leg's training):

Two legs are trained INDEPENDENTLY from the IDENTICAL baseline model:

- ``weights_only``  -> SFT on no-context prompts (``Question -> Answer``)
- ``ham_augmented`` -> SFT on context-augmented prompts (``Context + Question -> Answer``)

Both legs start from the same frozen checkpoint (step 0 invariant enforced), each
has its own optimizer/trajectory, and each is evaluated with its matching prompt
mode. The headline metric is cost-to-target per leg (optimizer steps / training
tokens / wall-clock to reach a common accuracy threshold T) plus the ham/weights
ratio. See ``docs/FINETUNING_PROTOCOL.md``.
"""

from __future__ import annotations

from ..backends.hf import HFBackend
from ..config import FinetuneExperimentConfig
from .corpus import build_corpus, build_qa_pairs
from .hf import HFLegTrainer, _INSTALL_HINT
from .mock import MockLegTrainer
from .protocol import (LEG_TO_PROMPT_MODE, FINETUNE_LEGS, LEGS, CheckpointEval,
                       ExampleResult, LegTrainer, checkpoint_steps)
from .target import cost_ratio, cost_to_target, parity_target

__all__ = [
    "LEG_TO_PROMPT_MODE", "FINETUNE_LEGS", "LEGS", "CheckpointEval", "ExampleResult",
    "LegTrainer", "checkpoint_steps", "cost_to_target", "parity_target", "cost_ratio",
    "build_corpus", "build_qa_pairs", "MockLegTrainer", "HFLegTrainer", "build_leg_trainer",
]


def build_leg_trainer(leg: str, cfg: FinetuneExperimentConfig, backend, embedder,
                       examples, corpus_facts):
    """Build a trainer for ONE leg. Dispatch on ``cfg.finetune.trainer``.

    The mock trainer needs no backend; the hf trainer requires ``backend.kind == 'hf'``.
    """
    if cfg.finetune.trainer == "mock":
        return MockLegTrainer(leg, cfg, examples)
    if cfg.finetune.trainer == "hf":
        if not isinstance(backend, HFBackend):
            raise RuntimeError(
                "finetune.trainer == 'hf' requires backend.kind == 'hf' "
                f"(got {type(backend).__name__}). " + _INSTALL_HINT)
        return HFLegTrainer(leg, backend, embedder, cfg, examples, corpus_facts)
    raise ValueError(f"unknown trainer {cfg.finetune.trainer!r}")
