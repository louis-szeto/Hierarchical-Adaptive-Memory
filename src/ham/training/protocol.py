"""Stage-C fine-tuning experiment: types and the trainer protocol.

NEW DESIGN (per-leg training with HAM injected into ham leg's training):

Two legs are trained INDEPENDENTLY from the IDENTICAL baseline model:

- ``weights_only``  -> SFT on no-context prompts (``Question -> Answer``)
- ``ham_augmented`` -> SFT on context-augmented prompts (``Context + Question -> Answer``)

Both legs start from the same frozen checkpoint (step 0 invariant enforced), each
has its own optimizer/trajectory, and each is evaluated with its matching prompt
mode. The headline metric is cost-to-target per leg (optimizer steps / training
tokens / wall-clock to reach a common accuracy threshold T) plus the ham/weights
ratio.

See ``docs/FINETUNING_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# The two legs (training conditions) of the fine-tuning experiment.
FINETUNE_LEGS = ["weights_only", "ham_augmented"]
LEGS = FINETUNE_LEGS  # short alias used by the trainers

# Each leg maps to a prompt mode for BOTH training and eval.
LEG_TO_PROMPT_MODE = {
    "weights_only": "no_context",      # Question -> Answer
    "ham_augmented": "context_augmented",  # Context + Question -> Answer
}


@dataclass
class ExampleResult:
    """One example's eval outcome under one leg at one checkpoint."""

    example_id: str
    question_type: str
    task_score: float
    exact_match: float
    correct: int  # 0/1 (task_score >= 1.0)
    prompt_tokens: int
    retrieval_recall_at_k: float | None  # None for the weights_only leg


@dataclass
class CheckpointEval:
    """Eval results for ONE leg at one training checkpoint."""

    step: int
    tokens_seen: int
    train_loss: float | None
    leg: str  # "weights_only" or "ham_augmented"
    results: list[ExampleResult] = field(default_factory=list)
    drift_rms: float | None = None  # RMS weight drift from the baseline (forgetting proxy)

    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return float(sum(r.correct for r in self.results)) / len(self.results)


class LegTrainer(Protocol):
    """Trains ONE leg with a given prompt mode.

    - ``MockLegTrainer`` (no torch): a deterministic synthetic learning curve;
      watermarked ``SMOKE TEST`` (never a scientific result).
    - ``HFLegTrainer`` (torch): a real SFT loop that trains from the baseline
      checkpoint and evaluates at checkpoints with the matching prompt mode.

    Returns a list of CheckpointEval for that leg only.
    """

    leg: str

    def run(self) -> list[CheckpointEval]: ...


def checkpoint_steps(max_steps: int, checkpoint_every: int) -> list[int]:
    """Steps at which evaluation occurs: 0 (untrained baseline), every
    ``checkpoint_every`` optimizer steps, and always ``max_steps`` (final)."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    steps = set(range(0, max_steps + 1, checkpoint_every))
    steps.add(0)
    steps.add(max_steps)
    return sorted(steps)
