"""Pure cost-to-target curve math (no torch, no backend).

NEW DESIGN: CheckpointEval has `leg` and `results` instead of `arm_results`.
"""

import random

from ham.training.protocol import ExampleResult, CheckpointEval, checkpoint_steps
from ham.training.target import cost_ratio, cost_to_target, parity_target


def _rows(acc: float, n: int, leg: str, seed: int = 0):
    rng = random.Random(f"{seed}:{leg}")
    perm = list(range(n))
    rng.shuffle(perm)
    n_correct = round(acc * n)
    correct_set = set(perm[:n_correct])
    return [ExampleResult(
        example_id=f"e{i}", question_type="single",
        task_score=1.0 if i in correct_set else 0.0,
        exact_match=1.0 if i in correct_set else 0.0,
        correct=int(i in correct_set), prompt_tokens=10,
        retrieval_recall_at_k=None) for i in range(n)]


def _ckpt(step: int, w: float, h: float, n: int = 10):
    """Return TWO checkpoints (one per leg) at the same step."""
    return [
        CheckpointEval(
            step=step, tokens_seen=step * 1000, train_loss=None,
            leg="weights_only", results=_rows(w, n, "weights_only")),
        CheckpointEval(
            step=step, tokens_seen=step * 1000, train_loss=None,
            leg="ham_augmented", results=_rows(h, n, "ham_augmented")),
    ]


def _curve():
    # step:        0     1     2     3     4
    # weights:   0.0   0.2   0.5   0.8   0.9
    # ham:       0.92  0.94  0.96  0.98  1.0
    flat = []
    for ckpt_pair in [_ckpt(0, 0.0, 0.92), _ckpt(1, 0.2, 0.94), _ckpt(2, 0.5, 0.96),
                     _ckpt(3, 0.8, 0.98), _ckpt(4, 0.9, 1.0)]:
        flat.extend(ckpt_pair)
    # Split into per-leg curves
    w_curve = [c for c in flat if c.leg == "weights_only"]
    h_curve = [c for c in flat if c.leg == "ham_augmented"]
    return {"weights_only": w_curve, "ham_augmented": h_curve}


def test_checkpoint_steps():
    assert checkpoint_steps(60, 10) == [0, 10, 20, 30, 40, 50, 60]
    assert checkpoint_steps(5, 10) == [0, 5]   # max < checkpoint_every still hits 0 and max


def test_cost_to_target_interpolates():
    curves = _curve()
    # weights crosses 0.65 between step 2 (0.5) and step 3 (0.8): frac = 0.5.
    c = cost_to_target(curves["weights_only"], "weights_only", 0.65)
    assert c["reached"] is True
    assert abs(c["optimizer_steps_to_target"] - 2.5) < 1e-9
    assert abs(c["training_tokens_to_target"] - 2500) < 1e-9


def test_cost_to_target_exact_checkpoint():
    curves = _curve()
    # weights reaches 0.5 exactly at step 2.
    c = cost_to_target(curves["weights_only"], "weights_only", 0.5)
    assert c["reached"] is True
    assert c["optimizer_steps_to_target"] == 2
    assert c["training_tokens_to_target"] == 2000


def test_cost_to_target_never_reached():
    curves = _curve()
    c = cost_to_target(curves["weights_only"], "weights_only", 0.95)  # max weights acc is 0.9
    assert c["reached"] is False
    assert c["optimizer_steps_to_target"] is None
    assert c["max_accuracy"] == 0.9


def test_parity_target():
    curves = _curve()
    # max weights accuracy (0.9) minus delta 0.03.
    tgt = parity_target(curves, "weights_only", 0.03)
    assert abs(tgt - 0.87) < 1e-9


def test_ham_reaches_target_cheaper():
    curves = _curve()
    target = parity_target(curves, "weights_only", 0.03)  # 0.87
    w = cost_to_target(curves["weights_only"], "weights_only", target)
    h = cost_to_target(curves["ham_augmented"], "ham_augmented", target)
    assert w["reached"] and h["reached"]
    ratio = cost_ratio(h, w, "training_tokens_to_target")
    assert ratio is not None and ratio < 1.0   # HAM reaches parity at lower token cost


def test_cost_ratio_undefined_when_not_reached():
    curves = _curve()
    w = cost_to_target(curves["weights_only"], "weights_only", 0.5)
    h_not = cost_to_target(curves["ham_augmented"], "ham_augmented", 1.5)  # impossible
    assert h_not["reached"] is False
    assert cost_ratio(h_not, w, "training_tokens_to_target") is None
