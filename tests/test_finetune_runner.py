"""End-to-end mock-trainer finetune run (deterministic, no torch).

NEW DESIGN: per-leg training with HAM injected into ham leg's training.
"""

import json
import os

from ham.config import finetune_from_dict
from ham.training.report import generate
from ham.training.runner import run_finetune


def _cfg(**over):
    base = {
        "name": "ft-test",
        "seed": 0,
        "backend": {"kind": "mock", "max_new_tokens": 12},
        "embedding": {"kind": "hash", "dim": 128},
        "dataset": {"name": "synthetic", "num_examples": 8, "num_sessions": 4,
                    "distractors_per_session": 2},
        "stats": {"bootstrap_resamples": 200, "permutation_resamples": 200},
        "finetune": {"trainer": "mock", "max_steps": 40, "checkpoint_every": 10,
                     "mock_weights_asymptote": 1.0, "mock_weights_rate": 2e-4,
                     "mock_ham_baseline": 0.92, "mock_ham_asymptote": 1.0,
                     "mock_ham_rate": 3e-4, "target_accuracy": 0.95},
    }
    base.update(over)
    return finetune_from_dict(base)


def test_finetune_run_outputs_and_stage_c(tmp_path):
    out = str(tmp_path / "run")
    summary = run_finetune(_cfg(), out)

    # All expected artifacts written.
    for fname in ("manifest.json", "curve.jsonl", "aggregate.json", "aggregate.csv",
                  "stats.json", "summary.json"):
        assert os.path.exists(os.path.join(out, fname)), fname

    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert manifest["target_stage"] == "C_finetuning"
    assert manifest["base_weights_changed"] is False   # mock trainer changes no weights
    assert manifest["is_smoke"] is True
    assert manifest["experiment"] == "stage_c_finetune"
    # Fair-control fingerprint present and is a 64-hex SHA-256.
    assert len(manifest["fair_control"]["fingerprint_sha256"]) == 64

    assert summary["is_smoke"] is True
    assert summary["trainer"] == "mock"
    # NEW: absolute target by default, not parity
    assert summary["target_kind"] == "absolute"
    assert summary["target_accuracy"] == 0.95


def test_curve_has_both_legs_and_examples(tmp_path):
    out = str(tmp_path / "run")
    run_finetune(_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "curve.jsonl"))]
    legs = {r["leg"] for r in rows}
    assert legs == {"weights_only", "ham_augmented"}
    # 5 checkpoints (0,10,20,30,40) x 2 legs x 8 examples.
    assert len(rows) == 5 * 2 * 8
    # weights_only rows carry no retrieval recall; ham rows carry a recall value.
    w = next(r for r in rows if r["leg"] == "weights_only")
    h = next(r for r in rows if r["leg"] == "ham_augmented")
    assert w["retrieval_recall_at_k"] is None
    assert h["retrieval_recall_at_k"] is not None


def test_cost_to_target_and_ratio(tmp_path):
    out = str(tmp_path / "run")
    summary = run_finetune(_cfg(), out)
    agg = summary["aggregate"]
    for leg in ("weights_only", "ham_augmented"):
        assert agg[leg]["reached"] is True
        assert agg[leg]["training_tokens_to_target"] is not None
    # HAM reaches the target at a strictly lower token cost than weights.
    ratio = summary["cost_ratio"]["tokens"]
    assert ratio is not None and 0.0 < ratio < 1.0


def test_report_watermarked_from_smoke(tmp_path):
    out = str(tmp_path / "run")
    run_finetune(_cfg(), out)
    res = generate(out, str(tmp_path / "artifacts"))
    assert res["had_data"] is True
    assert res["is_smoke"] is True
    md = open(os.path.join(res["out_dir"], "table_cost.md")).read()
    assert "SMOKE TEST" in md
    assert "weights_only" in md and "ham_augmented" in md


def test_hf_trainer_requires_hf_backend(tmp_path):
    # A mock backend with trainer=hf must fail loudly, not silently degrade.
    cfg = _cfg(finetune={"trainer": "hf"})
    try:
        run_finetune(cfg, str(tmp_path / "x"))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "hf" in str(e).lower()
