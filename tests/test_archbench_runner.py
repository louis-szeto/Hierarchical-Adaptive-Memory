"""Archbench mock end-to-end: outputs, stage-F manifest, redundancy slope, report.
Deterministic, no torch."""

import json
import os

from ham.archbench import build_trainer
from ham.archbench.mock import MockArchTrainer
from ham.archbench.protocol import ArchCheckpoint
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
    lo = a["recall|r=0.0|pretrain|ham_memory"]["bytes_ratio_vs_standard"]
    hi = a["recall|r=0.9|pretrain|ham_memory"]["bytes_ratio_vs_standard"]
    assert hi < lo                 # advantage grows with redundancy
    assert hi < 1.0                # ham cheaper than standard at high redundancy
    # iso-quality at high redundancy (compression is near-lossless)
    assert abs(a["recall|r=0.9|pretrain|ham_memory"]["quality_delta_vs_standard"]) < 1e-9


def test_no_consolidation_does_not_compress(tmp_path):
    # The consolidation ablation should not beat standard (isolates the mechanism).
    out = str(tmp_path / "run")
    run_archbench(_cfg(conditions=["no_memory", "standard_memory",
                                   "ham_memory", "ham_no_consolidation"]), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    nc = a["recall|r=0.9|pretrain|ham_no_consolidation"]["bytes_ratio_vs_standard"]
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


# ---------------------------------------------------------------------------
# drift_rms field (mock + serialized) and the fine-tuning post-hoc analysis
# ---------------------------------------------------------------------------

def test_mock_trainer_emits_drift_rms_curve():
    cfg = _cfg(trainer="mock")
    curve = MockArchTrainer(cfg, "ham_memory", 0.5).run()
    assert len(curve) > 1
    # Every checkpoint carries a non-None drift (mock emits a synthetic curve).
    for c in curve:
        assert isinstance(c, ArchCheckpoint)
        assert c.drift_rms is not None
    # Step 0 has zero drift; drift is monotonically non-decreasing afterwards.
    assert curve[0].step == 0 and curve[0].drift_rms == 0.0
    for prev, cur in zip(curve, curve[1:]):
        assert cur.drift_rms >= prev.drift_rms


def test_mock_drift_rms_ham_higher_than_standard():
    # The synthetic drift scale is ordered: HAM's extra router/fusion params
    # perturb the weights slightly more at any given step.
    cfg = _cfg(trainer="mock")
    std = MockArchTrainer(cfg, "standard_memory", 0.0).run()
    ham = MockArchTrainer(cfg, "ham_memory", 0.0).run()
    for s, h in zip(std, ham):
        if s.step > 0:
            assert h.drift_rms > s.drift_rms


def test_curve_jsonl_records_drift_rms(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "curve.jsonl"))]
    assert rows
    for r in rows:
        assert "drift_rms" in r
        assert r["drift_rms"] is not None


def test_finetune_posthoc_block_in_aggregate_and_summary(tmp_path):
    out = str(tmp_path / "run")
    summary = run_archbench(_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    s = json.load(open(os.path.join(out, "summary.json")))
    # The finetune_posthoc block is present in both aggregate.json and summary.json.
    assert "finetune_posthoc" in a
    assert "finetune_posthoc" in s
    fp = a["finetune_posthoc"]
    assert fp["noninferiority_delta"] == 0.03
    assert fp["primary_task"] == "recall"
    # Cells are keyed by "task|r=redundancy".
    assert "recall|r=0.0" in fp["cells"]
    assert "recall|r=0.9" in fp["cells"]


def test_finetune_posthoc_finite_costs_and_drift(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    fp = a["finetune_posthoc"]
    for key, cell in fp["cells"].items():
        for arm in ("standard", "ham"):
            e = cell[arm]
            # Both arms reach the parity target on the mock (it is constructed so
            # standard/ham are iso-quality at the same step).
            assert e["reached"] is True, f"{key}/{arm} did not reach target"
            assert e["optimizer_steps_to_target"] is not None
            assert e["optimizer_steps_to_target"] > 0
            assert e["training_tokens_to_target"] is not None
            assert e["training_tokens_to_target"] > 0
            assert e["drift_rms_at_target"] is not None
            assert e["drift_rms_at_target"] > 0
        # HAM/standard step + token ratios are finite and equal (iso-step design).
        assert cell["cost_ratio_steps_ham_over_standard"] is not None
        assert cell["cost_ratio_tokens_ham_over_standard"] is not None
        assert abs(cell["cost_ratio_steps_ham_over_standard"]
                   - cell["cost_ratio_tokens_ham_over_standard"]) < 1e-9
        # HAM drift overhead is finite and ordered (HAM > standard).
        assert cell["drift_ratio_ham_over_standard"] is not None
        assert cell["drift_ratio_ham_over_standard"] > 1.0


def test_finetune_posthoc_parity_target_equals_standard_peak_minus_delta(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    fp = a["finetune_posthoc"]
    for key, cell in fp["cells"].items():
        std_peak = cell["standard"]["max_quality"]
        assert abs(cell["target_quality"] - (std_peak - 0.03)) < 1e-9, key


def test_finetune_posthoc_table_rendered(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    res = generate(out, str(tmp_path / "art"))
    md_path = os.path.join(res["out_dir"], "table_finetune_posthoc.md")
    csv_path = os.path.join(res["out_dir"], "table_finetune_posthoc.csv")
    assert os.path.exists(md_path) and os.path.exists(csv_path)
    md = open(md_path).read()
    assert "Fine-tuning post-hoc" in md
    assert "| standard |" in md and "| ham |" in md
    assert "drift ratio" in md


def test_finetune_posthoc_skipped_when_only_one_arm_present(tmp_path):
    # If the run has standard but no ham (or vice versa), no cell is emitted
    # for that (task, redundancy); the report still renders an EMPTY TEMPLATE.
    out = str(tmp_path / "run")
    run_archbench(_cfg(conditions=["no_memory", "standard_memory"]), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    fp = a["finetune_posthoc"]
    assert fp["cells"] == {}   # no standard-vs-ham pair available
    res = generate(out, str(tmp_path / "art"))
    md = open(os.path.join(res["out_dir"], "table_finetune_posthoc.md")).read()
    assert "EMPTY TEMPLATE" in md


# ---------------------------------------------------------------------------
# Fine-tune regime (pretrain -> save state_dicts -> finetune from pretrained)
# ---------------------------------------------------------------------------

def _both_cfg(**over):
    base = {"archbench": {
        "trainer": "mock", "task": "recall", "regime": "both",
        "redundancy_levels": [0.0],
        "conditions": ["standard_memory", "ham_memory"],
        "max_steps": 100, "checkpoint_every": 20}}
    base["archbench"].update(over)
    return archbench_from_dict(base)


def test_finetune_regime_emits_pretrain_and_finetune_curves(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "curve.jsonl"))]
    regimes = {r["regime"] for r in rows}
    assert regimes == {"pretrain", "finetune"}
    # aggregate.json keys now include the regime segment.
    a = json.load(open(os.path.join(out, "aggregate.json")))
    for regime in ("pretrain", "finetune"):
        for cond in ("standard_memory", "ham_memory"):
            key = f"recall|r=0.0|{regime}|{cond}"
            assert key in a, key


def test_finetune_posthoc_uses_finetune_regime_when_present(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    fp = a["finetune_posthoc"]
    assert fp["regime"] == "finetune"
    assert "HEADLINE" in fp["description"]
    # Cells include regime=finetune.
    cell = fp["cells"]["recall|r=0.0"]
    assert cell["regime"] == "finetune"


def test_finetune_posthoc_falls_back_to_pretrain_when_no_finetune(tmp_path):
    # regime=pretrain only -> no finetune curves -> post-hoc uses pretrain.
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)  # default _cfg is regime=pretrain
    a = json.load(open(os.path.join(out, "aggregate.json")))
    fp = a["finetune_posthoc"]
    assert fp["regime"] == "pretrain"


def test_finetune_regime_mock_drift_from_pretrained_init(tmp_path):
    # The mock finetune drift is smaller than the pretrain drift (fine-tuning
    # perturbs weights less than from-scratch training), and step-0 drift is 0
    # (model starts at the loaded pretrained checkpoint, drift is measured from
    # that point, not from random init).
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "curve.jsonl"))]
    fin = [r for r in rows if r["regime"] == "finetune" and r["condition"] == "ham_memory"]
    pre = [r for r in rows if r["regime"] == "pretrain" and r["condition"] == "ham_memory"]
    fin0 = [r for r in fin if r["step"] == 0][0]
    assert fin0["drift_rms"] == 0.0   # drift from the loaded pretrained init
    # At every matching step, finetune drift < pretrain drift (gentler perturbation).
    fin_by_step = {r["step"]: r["drift_rms"] for r in fin}
    pre_by_step = {r["step"]: r["drift_rms"] for r in pre}
    for s, pre_d in pre_by_step.items():
        if s > 0 and s in fin_by_step:
            assert fin_by_step[s] < pre_d, f"step {s}: finetune drift not gentler"


def test_finetune_regime_mock_quality_starts_above_zero(tmp_path):
    # The mock finetune quality starts at a fraction of the ceiling (the
    # pretrained checkpoint already knew something), unlike pretrain which
    # starts at 0.
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "curve.jsonl"))]
    fin0 = [r for r in rows if r["regime"] == "finetune" and r["step"] == 0][0]
    pre0 = [r for r in rows if r["regime"] == "pretrain" and r["step"] == 0][0]
    assert fin0["quality"] > 0.0
    assert pre0["quality"] == 0.0
    assert fin0["quality"] > pre0["quality"]


def test_finetune_regime_drift_ratio_ham_over_standard_above_one(tmp_path):
    # HAM's extra router/fusion params perturb the weights slightly more even
    # in the finetune regime, so the drift ratio is still > 1.0.
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    a = json.load(open(os.path.join(out, "aggregate.json")))
    fp = a["finetune_posthoc"]
    cell = fp["cells"]["recall|r=0.0"]
    assert cell["drift_ratio_ham_over_standard"] is not None
    assert cell["drift_ratio_ham_over_standard"] > 1.0


def test_finetune_regime_posthoc_table_says_headline(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    res = generate(out, str(tmp_path / "art"))
    md = open(os.path.join(res["out_dir"], "table_finetune_posthoc.md")).read()
    assert "HEADLINE" in md
    assert "w_pretrained" in md or "p_pretrained" in md
    assert "catastrophic-forgetting" in md
    assert "SMOKE TEST" in md  # mock -> watermarked


def test_finetune_regime_manifest_records_regime_and_offsets(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_both_cfg(), out)
    m = json.load(open(os.path.join(out, "manifest.json")))
    assert m["regime"] == "both"
    fc = m["fair_control"]
    assert fc["regime"] == "both"
    assert fc["finetune_seed_offset"] == 1001
    assert fc["finetune_n_keys_multiplier"] == 2.0


def test_heldout_corpus_uses_different_seed_and_higher_keys(tmp_path):
    # The held-out corpus is a FRESH association set: different seed and a
    # higher key count, so the model must learn new associations.
    from ham.archbench.runner import _heldout_corpus, _n_items
    from ham.config import archbench_from_dict
    cfg = archbench_from_dict({"archbench": {
        "trainer": "mock", "task": "recall", "regime": "both",
        "redundancy_levels": [0.0],
        "conditions": ["standard_memory", "ham_memory"]}})
    n_pre = _n_items(cfg, "recall")
    heldout = _heldout_corpus(cfg, "recall", 0.0)
    # Held-out n_keys = pretrain n_keys * 2.0 multiplier.
    assert heldout.n_items == n_pre * 2
    # The held-out corpus has the same shape as a recall corpus.
    assert heldout.task == "recall"
    assert heldout.input_ids.shape[0] == cfg.archbench.n_train_streams


def test_curve_jsonl_records_regime(tmp_path):
    out = str(tmp_path / "run")
    run_archbench(_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "curve.jsonl"))]
    assert rows
    for r in rows:
        assert "regime" in r
        assert r["regime"] == "pretrain"   # default _cfg is pretrain

