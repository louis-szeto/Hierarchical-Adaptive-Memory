"""Pure cost-to-target curve math (no torch, fully unit-testable).

Given a per-leg learning curve (list of CheckpointEval for that leg), find the
training cost to reach a target accuracy. "Cost" is the pair (optimizer steps,
training tokens). Wall-clock is intentionally NOT reported: it is hardware-
dependent and not a universal, reproducible metric.
"""

from __future__ import annotations

from typing import Iterable

from .protocol import CheckpointEval


def _leg_curve(checkpoints: Iterable[CheckpointEval]) -> list[tuple]:
    """Ordered (step, tokens_seen, accuracy) for one leg."""
    out = []
    for ckpt in checkpoints:
        out.append((ckpt.step, ckpt.tokens_seen, ckpt.accuracy()))
    return out


def parity_target(checkpoints: dict, parity_with: str,
                  delta: float) -> float:
    """Target = max accuracy of the ``parity_with`` leg across the curve, minus a
    non-inferiority margin. I.e. 'reach the accuracy full memorization achieves,
    less delta'. Clamped to [0, 1].

    DEPRECATED: kept for compatibility with old configs, but the new design uses
    absolute target_accuracy=0.95 by default.
    """
    curve = _leg_curve(checkpoints.get(parity_with, []))
    if not curve:
        return 0.0
    peak = max(acc for _, _, acc in curve)
    return max(0.0, min(1.0, peak - delta))


def cost_to_target(checkpoints: list[CheckpointEval], leg: str, target: float,
                   interpolate: bool = True) -> dict:
    """First checkpoint where ``leg`` accuracy >= target.

    With ``interpolate`` and a non-trivially-reached target, linearly interpolate
    the token / step cost between the last-below and first-at-or-above checkpoints
    for a smoother estimate. Returns ``reached=False`` honestly if the leg never
    crosses the target.

    NOTE: ``leg`` param is kept for API compatibility but ignored; the curve
    passed in is already for a single leg.
    """
    curve = _leg_curve(checkpoints)
    if not curve or target is None:
        return _cost_result(reached=False)
    # Find first checkpoint at or above target.
    hit_idx = None
    for i, (_s, _t, acc) in enumerate(curve):
        if acc >= target:
            hit_idx = i
            break
    if hit_idx is None:
        return _cost_result(reached=False, final_accuracy=curve[-1][2],
                            max_accuracy=max(a for _, _, a in curve))
    step, tokens, acc = curve[hit_idx]
    if interpolate and hit_idx > 0:
        prev_step, prev_tokens, prev_acc = curve[hit_idx - 1]
        denom = acc - prev_acc
        # Fraction of the way from prev -> hit at which `target` is crossed.
        frac = 0.0 if denom == 0 else (target - prev_acc) / denom
        frac = min(max(frac, 0.0), 1.0)
        tokens = prev_tokens + frac * (tokens - prev_tokens)
        step = prev_step + frac * (step - prev_step)
    return _cost_result(
        reached=True, optimizer_steps=step, training_tokens=tokens,
        accuracy_at_target=acc,
        final_accuracy=curve[-1][2], max_accuracy=max(a for _, _, a in curve))


def _cost_result(reached: bool, optimizer_steps: float | None = None,
                 training_tokens: float | None = None,
                 accuracy_at_target: float | None = None, final_accuracy: float | None = None,
                 max_accuracy: float | None = None) -> dict:
    return {
        "reached": bool(reached),
        "optimizer_steps_to_target": optimizer_steps,
        "training_tokens_to_target": training_tokens,
        "accuracy_at_target": accuracy_at_target,
        "final_accuracy": final_accuracy,
        "max_accuracy": max_accuracy,
    }


def cost_ratio(numer: dict, denom: dict, field_name: str) -> float | None:
    """numer[field] / denom[field], guarded for the 'denominator reached at step 0'
    (cost 0) and 'not reached' cases. Returns None when undefined."""
    if not numer.get("reached") or not denom.get("reached"):
        return None
    d = denom.get(field_name)
    n = numer.get(field_name)
    if d is None or n is None or d == 0:
        return None
    return float(n) / float(d)
