"""Deterministic mock leg trainer: a synthetic learning curve with NO torch.

Used by tests/CI and the ``finetune_smoke`` config. Its outputs are flagged
``is_smoke=True`` and watermarked ``SMOKE TEST`` -- they are plumbing checks,
never scientific results.

NEW DESIGN (per-leg curves):

- ``weights_only``:   A_w * (1 - exp(-k_w * tokens))         # climbs from 0
- ``ham_augmented``:  R0 + (A_h - R0) * (1 - exp(-k_h * tokens))  # starts at retrieval R0

Both curves can reach the ceiling (1.0) with enough training. The mock parameters
are tuned to ensure non-degenerate behavior with target_accuracy=0.95.
"""

from __future__ import annotations

import math
import random

from ..config import FinetuneExperimentConfig
from ..datasets.base import Example
from .protocol import CheckpointEval, ExampleResult, checkpoint_steps


def _distribute_correct(examples: list[Example], leg: str, frac: float,
                        seed: int) -> set[int]:
    """Deterministically choose which example indices are 'correct' for a leg at
    a given accuracy fraction."""
    n = len(examples)
    if n == 0:
        return set()
    rng = random.Random(f"{seed}:{leg}")
    perm = list(range(n))
    rng.shuffle(perm)
    n_correct = int(round(min(1.0, max(0.0, frac)) * n))
    return set(perm[:n_correct])


def _mock_accuracy(cfg, leg: str, tokens: float) -> float:
    """Synthetic accuracy for one leg at a given token count."""
    ft = cfg.finetune
    if leg == "weights_only":
        # Climbs from 0 to asymptote
        acc = ft.mock_weights_asymptote * (1.0 - math.exp(-ft.mock_weights_rate * tokens))
    else:  # ham_augmented
        # Starts at retrieval baseline, climbs to asymptote
        acc = ft.mock_ham_baseline + (ft.mock_ham_asymptote - ft.mock_ham_baseline) * (
            1.0 - math.exp(-ft.mock_ham_rate * tokens))
    return min(1.0, max(0.0, acc))


class MockLegTrainer:
    """No-op trainer that synthesizes a deterministic per-leg learning curve."""

    def __init__(self, leg: str, cfg: FinetuneExperimentConfig, examples: list[Example]):
        if leg not in ("weights_only", "ham_augmented"):
            raise ValueError(f"leg must be 'weights_only' or 'ham_augmented', got {leg!r}")
        self.leg = leg
        self.cfg = cfg
        self.examples = examples

    def run(self) -> list[CheckpointEval]:
        ft = self.cfg.finetune
        steps = checkpoint_steps(ft.max_steps, ft.checkpoint_every)
        curve: list[CheckpointEval] = []
        for step in steps:
            tokens = step * ft.tokens_per_step
            wall = step * ft.mock_seconds_per_step
            train_loss = max(0.05, 2.0 * math.exp(-1.0e-4 * tokens))

            if step == 0:
                # Step-0 baseline: brand-new model, no-context eval for BOTH legs
                # -> identical 0-accuracy start (mirrors the hf trainer).
                frac = 0.0
            else:
                frac = _mock_accuracy(self.cfg, self.leg, tokens)
            correct_set = _distribute_correct(self.examples, self.leg, frac, self.cfg.seed)

            rows: list[ExampleResult] = []
            for i, ex in enumerate(self.examples):
                ok = i in correct_set
                # Prompt tokens: context-augmented has more tokens
                prompt_tokens = 140 if self.leg == "ham_augmented" else 50
                rows.append(ExampleResult(
                    example_id=ex.example_id, question_type=ex.question_type,
                    task_score=1.0 if ok else 0.0,
                    exact_match=1.0 if ok else 0.0,  # Mock: exact_match = task_score
                    correct=int(ok), prompt_tokens=prompt_tokens,
                    retrieval_recall_at_k=(1.0 if (self.leg == "ham_augmented" and ok) else
                                          (None if self.leg == "weights_only" else 0.0))))
            curve.append(CheckpointEval(
                step=step, tokens_seen=tokens, wall_clock_s=wall,
                train_loss=train_loss, leg=self.leg, results=rows))
        return curve
