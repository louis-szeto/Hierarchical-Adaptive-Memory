"""Finetune report: EMPTY TEMPLATE on no data; populated tables + figure from a
real mock run. Deterministic, no torch."""

import json
import os

from ham.config import finetune_from_dict
from ham.training.report import generate
from ham.training.runner import run_finetune


def _cfg(**over):
    base = {
        "name": "ft-report",
        "seed": 0,
        "backend": {"kind": "mock"},
        "embedding": {"kind": "hash", "dim": 128},
        "dataset": {"name": "synthetic", "num_examples": 6, "num_sessions": 4,
                    "distractors_per_session": 2},
        "stats": {"bootstrap_resamples": 200, "permutation_resamples": 200},
        "finetune": {"trainer": "mock", "max_steps": 30, "checkpoint_every": 10},
    }
    base.update(over)
    return finetune_from_dict(base)


def test_report_empty_template_when_no_data(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    res = generate(empty, str(tmp_path / "art"))
    assert res["had_data"] is False
    md = open(os.path.join(res["out_dir"], "table_cost.md")).read()
    assert "EMPTY TEMPLATE" in md
    assert "n.a." in md


def test_report_tables_populated_from_run(tmp_path):
    run_dir = str(tmp_path / "run")
    run_finetune(_cfg(), run_dir)
    res = generate(run_dir, str(tmp_path / "artifacts"))
    assert res["had_data"] is True

    cost = open(os.path.join(res["out_dir"], "table_cost.md")).read()
    assert "Cost-to-target" in cost
    assert "RatioTokens" in cost
    assert os.path.exists(os.path.join(res["out_dir"], "table_cost.csv"))

    curve = open(os.path.join(res["out_dir"], "table_curve.md")).read()
    assert "Accuracy curve" in curve
    assert "weights_only" in curve and "ham_augmented" in curve

    # Either the figure rendered, or matplotlib is absent and the marker exists.
    art_dir = res["out_dir"]
    assert (os.path.exists(os.path.join(art_dir, "fig_accuracy_vs_tokens.png"))
            or os.path.exists(os.path.join(art_dir, "FIGURES_SKIPPED.txt")))

    label = open(os.path.join(res["out_dir"], "RUN_LABEL.txt")).read()
    assert "SMOKE TEST" in label  # mock-trainer run is watermarked


def test_report_reads_target_from_manifest(tmp_path):
    run_dir = str(tmp_path / "run")
    run_finetune(_cfg(), run_dir)
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    generate(run_dir, str(tmp_path / "artifacts"))
    cost = open(os.path.join(tmp_path, "artifacts", "table_cost.md")).read()
    # The target line is present and carries the manifest's target accuracy
    # (rounded to 3 significant figures by the report formatter).
    assert "Target accuracy =" in cost
    assert f"{manifest['target_accuracy']:.3g}" in cost
