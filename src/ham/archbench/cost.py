"""Pure cost-to-target curve math for the archbench post-hoc analysis (no torch).

Given a per-(condition x redundancy) learning curve of :class:`ArchCheckpoint`,
find the training cost to reach a target quality. "Cost" is the pair
(optimizer steps, training tokens). At-target drift is the L2 weight drift
``sqrt(sum((p - p_init)**2))`` recorded on the reaching checkpoint. Wall-clock
is intentionally NOT reported: it is hardware-dependent and not a universal,
reproducible metric.

The post-hoc compares the **standard flat memory block** against the **HAM
memory block** on the same toy LM (both WITH memory) -- it is NOT the legacy
stage-C SmolLM2/external-retrieval fine-tune. The toy LM is trained from scratch
on a synthetic corpus, so there is no pretrained knowledge to forget; the
diagnostic of interest is the weight-drift overhead HAM's extra
router/fusion/encoding parameters add to reach the same target.

These helpers are pure-math (deterministic, unit-testable) and used by
``runner.run_archbench`` to build the ``finetune_posthoc`` block of
``aggregate.json`` / ``summary.json``.
"""

from __future__ import annotations

from typing import Iterable

from .protocol import ArchCheckpoint

# The two arms of the fine-tuning post-hoc analysis on the toy models.
STANDARD = "standard_memory"
HAM = "ham_memory"


def _curve(checkpoints: Iterable[ArchCheckpoint]) -> list[tuple]:
    """Ordered (step, tokens_seen, quality, drift_rms) for one curve."""
    out = []
    for ckpt in checkpoints:
        out.append((ckpt.step, ckpt.tokens_seen, ckpt.quality, ckpt.drift_rms))
    return out


def parity_target(standard_curve: list[ArchCheckpoint], delta: float) -> float:
    """Target = max quality of the ``standard_memory`` arm across its curve,
    minus a non-inferiority margin. I.e. 'reach the quality standard attains,
    less delta'. Clamped to [0, 1]."""
    if not standard_curve:
        return 0.0
    peak = max(c.quality for c in standard_curve)
    return max(0.0, min(1.0, peak - delta))


def cost_to_target(curve: list[ArchCheckpoint], target: float,
                   interpolate: bool = False) -> dict:
    """First checkpoint where ``quality >= target``.

    By default (``interpolate=False``) the cost is reported at the first
    checkpoint whose quality reaches ``target`` -- i.e. actual executed
    optimizer steps and training tokens, never a fractional interpolation.
    With ``interpolate=True`` the step/token cost is linearly interpolated
    between the last-below and first-at-or-above checkpoints instead. The
    recorded ``drift_rms_at_target`` is always the reaching checkpoint's
    measured drift (interpolation of drift is not meaningful). Returns
    ``reached=False`` honestly if the curve never crosses the target.
    """
    ordered = _curve(curve)
    if not ordered or target is None:
        return _cost_result(reached=False)
    hit_idx = None
    for i, (_s, _t, q, _d) in enumerate(ordered):
        if q >= target:
            hit_idx = i
            break
    if hit_idx is None:
        return _cost_result(
            reached=False, final_quality=ordered[-1][2],
            max_quality=max(q for _, _, q, _ in ordered))
    step, tokens, q, drift = ordered[hit_idx]
    if interpolate and hit_idx > 0:
        prev_step, prev_tokens, prev_q, _ = ordered[hit_idx - 1]
        denom = q - prev_q
        frac = 0.0 if denom == 0 else (target - prev_q) / denom
        frac = min(max(frac, 0.0), 1.0)
        tokens = prev_tokens + frac * (tokens - prev_tokens)
        step = prev_step + frac * (step - prev_step)
    return _cost_result(
        reached=True, optimizer_steps=step, training_tokens=tokens,
        quality_at_target=q, drift_rms_at_target=drift,
        final_quality=ordered[-1][2],
        max_quality=max(qq for _, _, qq, _ in ordered))


def _cost_result(reached: bool, optimizer_steps: float | None = None,
                 training_tokens: float | None = None,
                 quality_at_target: float | None = None,
                 drift_rms_at_target: float | None = None,
                 final_quality: float | None = None,
                 max_quality: float | None = None) -> dict:
    return {
        "reached": bool(reached),
        "optimizer_steps_to_target": optimizer_steps,
        "training_tokens_to_target": training_tokens,
        "quality_at_target": quality_at_target,
        "drift_rms_at_target": drift_rms_at_target,
        "final_quality": final_quality,
        "max_quality": max_quality,
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
