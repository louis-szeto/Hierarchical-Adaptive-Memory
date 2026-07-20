"""Pure cost-to-target + drift math for the archbench fine-tuning post-hoc
(no torch, no trainer). Mirrors the legacy training/target tests but on
``ArchCheckpoint`` curves.
"""

from ham.archbench.cost import (cost_ratio, cost_to_target, parity_target)
from ham.archbench.protocol import ArchCheckpoint


def _ckpt(step: int, quality: float, drift: float | None = None,
          condition: str = "standard_memory") -> ArchCheckpoint:
    return ArchCheckpoint(
        step=step, tokens_seen=step * 1000, train_loss=None, quality=quality,
        memory_bytes=100, redundancy=0.0, condition=condition, regime="pretrain",
        drift_rms=drift)


def _std_curve():
    # step:        0     1     2     3     4
    # quality:   0.0   0.2   0.5   0.8   0.9
    # drift:     0.0   1.0   2.0   3.0   4.0
    return [_ckpt(0, 0.0, 0.0), _ckpt(1, 0.2, 1.0), _ckpt(2, 0.5, 2.0),
            _ckpt(3, 0.8, 3.0), _ckpt(4, 0.9, 4.0)]


def _ham_curve():
    # HAM reaches the same quality at the SAME steps (the verified toy finding)
    # but with ~1% higher drift at each checkpoint.
    return [_ckpt(0, 0.0, 0.0, "ham_memory"), _ckpt(1, 0.2, 1.01, "ham_memory"),
            _ckpt(2, 0.5, 2.02, "ham_memory"), _ckpt(3, 0.8, 3.03, "ham_memory"),
            _ckpt(4, 0.9, 4.04, "ham_memory")]


def test_parity_target():
    std = _std_curve()
    # max standard quality (0.9) minus delta 0.03.
    assert abs(parity_target(std, 0.03) - 0.87) < 1e-9


def test_parity_target_empty_curve():
    assert parity_target([], 0.03) == 0.0


def test_parity_target_clamped():
    # peak below delta -> clamped to 0.
    assert parity_target([_ckpt(0, 0.0), _ckpt(1, 0.01)], 0.03) == 0.0


def test_cost_to_target_exact_checkpoint():
    curve = _std_curve()
    c = cost_to_target(curve, 0.5)
    assert c["reached"] is True
    assert c["optimizer_steps_to_target"] == 2
    assert c["training_tokens_to_target"] == 2000
    assert c["quality_at_target"] == 0.5
    assert c["drift_rms_at_target"] == 2.0


def test_cost_to_target_interpolate_only_steps_and_tokens():
    curve = _std_curve()
    # weights crosses 0.65 between step 2 (0.5) and step 3 (0.8): frac = 0.5.
    c = cost_to_target(curve, 0.65, interpolate=True)
    assert c["reached"] is True
    assert abs(c["optimizer_steps_to_target"] - 2.5) < 1e-9
    assert abs(c["training_tokens_to_target"] - 2500) < 1e-9
    # Drift is NOT interpolated -- it is the reaching checkpoint's recorded value.
    assert c["drift_rms_at_target"] == 3.0
    assert c["quality_at_target"] == 0.8


def test_cost_to_target_never_reached():
    curve = _std_curve()
    c = cost_to_target(curve, 0.95)  # max quality is 0.9
    assert c["reached"] is False
    assert c["optimizer_steps_to_target"] is None
    assert c["training_tokens_to_target"] is None
    assert c["drift_rms_at_target"] is None
    assert c["max_quality"] == 0.9


def test_cost_to_target_handles_missing_drift():
    # Mock path with drift=None should not break the lookup; drift_at_target
    # comes back None and ratios guard for it.
    curve = [
        ArchCheckpoint(step=0, tokens_seen=0, train_loss=None, quality=0.0,
                       memory_bytes=0, redundancy=0.0,
                       condition="standard_memory", regime="pretrain",
                       drift_rms=None),
        ArchCheckpoint(step=1, tokens_seen=100, train_loss=None, quality=0.6,
                       memory_bytes=0, redundancy=0.0,
                       condition="standard_memory", regime="pretrain",
                       drift_rms=None),
    ]
    c = cost_to_target(curve, 0.5)
    assert c["reached"] is True
    assert c["drift_rms_at_target"] is None


def test_cost_ratio_tokens():
    std = cost_to_target(_std_curve(), 0.5)
    ham = cost_to_target(_ham_curve(), 0.5)
    # Same target -> same tokens (the verified matched-cost finding).
    r = cost_ratio(ham, std, "training_tokens_to_target")
    assert r is not None and abs(r - 1.0) < 1e-9


def test_cost_ratio_undefined_when_not_reached():
    std = cost_to_target(_std_curve(), 0.5)
    ham_not = cost_to_target(_ham_curve(), 1.5)  # impossible target
    assert ham_not["reached"] is False
    assert cost_ratio(ham_not, std, "training_tokens_to_target") is None


def test_cost_ratio_undefined_when_denominator_not_reached():
    # Asymmetric: numerator reaches the target but the denominator does not -> None.
    std_not = cost_to_target(_std_curve(), 1.5)   # impossible target -> not reached
    ham = cost_to_target(_ham_curve(), 0.5)
    assert std_not["reached"] is False and ham["reached"] is True
    assert cost_ratio(ham, std_not, "training_tokens_to_target") is None


def test_cost_ratio_zero_denominator():
    # Reaching at step 0 -> denominator cost 0 -> ratio undefined (guarded).
    std = cost_to_target([_ckpt(0, 0.95, 0.0)], 0.5)
    ham = cost_to_target([_ckpt(0, 0.95, 0.0, "ham_memory")], 0.5)
    assert std["training_tokens_to_target"] == 0
    assert cost_ratio(ham, std, "training_tokens_to_target") is None


def _ft_ckpt(step: int, quality: float, drift: float | None = None,
             condition: str = "standard_memory") -> ArchCheckpoint:
    """A finetune-regime checkpoint for the post-hoc regime-aware tests."""
    return ArchCheckpoint(
        step=step, tokens_seen=step * 1000, train_loss=None, quality=quality,
        memory_bytes=100, redundancy=0.0, condition=condition, regime="finetune",
        drift_rms=drift)


def test_cost_to_target_works_with_finetune_regime_checkpoints():
    # The cost-to-target math is regime-agnostic; it only reads step/tokens/
    # quality/drift from the ArchCheckpoint. A finetune-regime curve (quality
    # starts above 0 because the model is pretrained) works the same way.
    curve = [_ft_ckpt(0, 0.5, 0.0), _ft_ckpt(1, 0.7, 0.5),
             _ft_ckpt(2, 0.85, 1.0), _ft_ckpt(3, 0.9, 1.5)]
    c = cost_to_target(curve, 0.85)
    assert c["reached"] is True
    assert c["optimizer_steps_to_target"] == 2
    assert c["drift_rms_at_target"] == 1.0
    assert c["quality_at_target"] == 0.85


def test_parity_target_with_finetune_curve():
    std = [_ft_ckpt(0, 0.5, 0.0), _ft_ckpt(1, 0.85, 0.5), _ft_ckpt(2, 0.9, 1.0)]
    # target = 0.9 - 0.03 = 0.87 -- only step 2 reaches it.
    target = parity_target(std, 0.03)
    assert abs(target - 0.87) < 1e-9
