"""KV bench mock end-to-end: outputs, stage-D manifest, redundancy slope, report.
Deterministic, no torch."""

import json
import os

from ham.config import kvbench_from_dict
from ham.kvbench.report import generate
from ham.kvbench.runner import run_kvbench


def _cfg(**over):
    base = {"kvbench": {
        "trainer": "mock", "redundancy_levels": [0.0, 0.9], "n_contexts": 2,
        "conditions": ["full_kv", "ham_kv", "uniform_quant_kv",
                       "random_evict_kv", "ham_no_cluster"]}}
    base["kvbench"].update(over)
    return kvbench_from_dict(base)


def test_kvbench_outputs_stage_d(tmp_path):
    out = str(tmp_path / "run")
    s = run_kvbench(_cfg(), out)
    for f in ("manifest.json", "results.jsonl", "aggregate.json", "aggregate.csv",
              "stats.json", "summary.json"):
        assert os.path.exists(os.path.join(out, f)), f
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["target_stage"] == "D_inference_kv_compression"
    assert m["integration_mode"] == "kv_cache_compression"
    assert m["is_smoke"] is True
    assert m["experiment"] == "stage_d_kvbench"
    assert len(m["fair_control"]["fingerprint_sha256"]) == 64
    assert s["n_results"] > 0


def test_ham_advantage_grows_with_redundancy(tmp_path):
    out = str(tmp_path / "run")
    run_kvbench(_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    lo = a["r=0.0|ham_kv|kr=0.5"]["bytes_ratio_vs_full"]
    hi = a["r=0.9|ham_kv|kr=0.5"]["bytes_ratio_vs_full"]
    assert hi < lo            # advantage grows with redundancy
    assert hi < 1.0           # ham smaller than full at high redundancy


def test_only_ham_scales_with_redundancy(tmp_path):
    # The frequency-agnostic conditions should NOT scale with redundancy.
    out = str(tmp_path / "run")
    run_kvbench(_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    for cond, kr in (("uniform_quant_kv", 1.0), ("random_evict_kv", 0.5),
                     ("ham_no_cluster", 0.5)):
        lo = a[f"r=0.0|{cond}|kr={kr}"]["bytes_ratio_vs_full"]
        hi = a[f"r=0.9|{cond}|kr={kr}"]["bytes_ratio_vs_full"]
        assert abs(hi - lo) < 1e-9


def test_report_populated_and_empty(tmp_path):
    out = str(tmp_path / "run")
    run_kvbench(_cfg(), out)
    res = generate(out, str(tmp_path / "art"))
    assert res["had_data"] is True and res["is_smoke"] is True
    md = open(os.path.join(res["out_dir"], "table_redundancy.md")).read()
    assert "SMOKE TEST" in md and "ham_kv" in md

    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    res2 = generate(empty, str(tmp_path / "art2"))
    assert res2["had_data"] is False
    assert "EMPTY TEMPLATE" in open(os.path.join(res2["out_dir"], "table_redundancy.md")).read()
