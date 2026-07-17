"""Archbench mock end-to-end: outputs, stage-F manifest, redundancy slope, report.
Deterministic, no torch."""

import json
import os

from ham.archbench import build_trainer
from ham.archbench.report import generate
from ham.archbench.runner import run_archbench
from ham.config import archbench_from_dict


def _cfg(**over):
    base = {"archbench": {
        "trainer": "mock", "task": "recall",
        "redundancy_levels": [0.0, 0.9],
        "conditions": ["no_memory", "standard_memory", "ham_memory"],
        "max_steps": 100, "checkpoint_every": 20}}
    base["archbench"].update(over)
    return archbench_from_dict(base)


def test_archbench_run_outputs_and_stage_f(tmp_path):
    out = str(tmp_path / "run")
    s = run_archbench(_cfg(), out)
    for f in ("manifest.json", "curve.jsonl", "aggregate.json", "aggregate.csv",
              "stats.json", "summary.json"):
        assert os.path.exists(os.path.join(out, f)), f
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["target_stage"] == "F_architecture_level"
    assert m["is_smoke"] is True
    assert m["experiment"] == "stage_f_archbench"
    assert len(m["fair_control"]["fingerprint_sha256"]) == 64
    assert s["n_curves"] > 0


def test_ham_advantage_grows_with_redundancy(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    lo = a["recall|r=0.0|ham_memory"]["bytes_ratio_vs_standard"]
    hi = a["recall|r=0.9|ham_memory"]["bytes_ratio_vs_standard"]
    assert hi < lo                 # advantage grows with redundancy
    assert hi < 1.0                # ham cheaper than standard at high redundancy
    # iso-quality at high redundancy (compression is near-lossless)
    assert abs(a["recall|r=0.9|ham_memory"]["quality_delta_vs_standard"]) < 1e-9


def test_no_consolidation_does_not_compress(tmp_path):
    # The consolidation ablation should not beat standard (isolates the mechanism).
    out = str(tmp_path / "run")
    run_archbench(_cfg(conditions=["no_memory", "standard_memory",
                                   "ham_memory", "ham_no_consolidation"]), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    nc = a["recall|r=0.9|ham_no_consolidation"]["bytes_ratio_vs_standard"]
    assert abs(nc - 1.0) < 1e-9


def test_report_watermarked_and_populated(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    res = generate(out, str(tmp_path / "art"))
    assert res["had_data"] is True and res["is_smoke"] is True
    md = open(os.path.join(res["out_dir"], "table_redundancy.md")).read()
    assert "SMOKE TEST" in md and "ham_memory" in md
    assert os.path.exists(os.path.join(res["out_dir"], "table_quality_bytes.md"))


def test_report_empty_template(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    res = generate(empty, str(tmp_path / "art"))
    assert res["had_data"] is False
    md = open(os.path.join(res["out_dir"], "table_redundancy.md")).read()
    assert "EMPTY TEMPLATE" in md


def test_build_trainer_torch_requires_corpus():
    cfg = _cfg(trainer="torch")
    try:
        build_trainer(cfg, "ham_memory", 0.5, None)  # no corpus
        assert False, "expected ValueError"
    except ValueError as e:
        assert "corpus" in str(e).lower()
