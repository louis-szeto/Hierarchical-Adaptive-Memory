import json
import os

import pytest

from ham.config import StageConfig, from_dict
from ham.runner import run_experiment


def _cfg(**over):
    base = {
        "name": "test",
        "conditions": ["memory_off", "uncompressed_rag", "ham_memory"],
        "backend": {"kind": "mock", "max_new_tokens": 16},
        "embedding": {"kind": "hash", "dim": 128},
        "dataset": {"name": "synthetic", "num_examples": 4, "num_sessions": 3,
                    "distractors_per_session": 2},
        "stats": {"bootstrap_resamples": 100, "permutation_resamples": 100},
    }
    base.update(over)
    return from_dict(base)


def test_stage_defaults_are_stage_E():
    s = StageConfig()
    assert s.target_stage == "E_inference_external_memory"
    assert s.base_weights_changed is False
    assert s.persistent_across_sessions is True
    assert s.integration_mode == "external_context"
    assert s.trainable_router is False


def test_invalid_stage_and_mode_fail_loudly():
    with pytest.raises(ValueError):
        StageConfig(target_stage="Z_not_a_stage")
    with pytest.raises(ValueError):
        StageConfig(integration_mode="telepathy")


def test_manifest_records_stage_and_fair_control(tmp_path):
    cfg = _cfg()
    out = str(tmp_path / "run")
    run_experiment(cfg, out)
    m = json.load(open(os.path.join(out, "manifest.json")))
    for key in ("target_stage", "base_weights_changed", "persistent_across_sessions",
                "integration_mode", "trainable_router", "fair_control"):
        assert key in m, key
    assert m["target_stage"] == "E_inference_external_memory"
    assert "fingerprint_sha256" in m["fair_control"]


def test_per_row_carries_condition_stage_metadata(tmp_path):
    cfg = _cfg()
    out = str(tmp_path / "run")
    run_experiment(cfg, out)
    rows = [json.loads(l) for l in open(os.path.join(out, "per_example.jsonl"))]
    off = next(r for r in rows if r["condition"] == "memory_off")
    ham = next(r for r in rows if r["condition"] == "ham_memory")
    assert off["persistent"] is False and off["adaptive_precision"] is False
    assert ham["persistent"] is True and ham["adaptive_precision"] is True
    assert ham["integration_mode"] == "external_context"
    assert "not a reproduction" in ham["literature_analogue"] or ham["literature_analogue"].startswith("HAM")
