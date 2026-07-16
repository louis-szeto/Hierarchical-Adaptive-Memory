import json
import os

from ham.conditions import BASELINE_CONDITIONS, build_condition
from ham.config import CompressionConfig, from_dict
from ham.report import generate
from ham.runner import run_experiment


def _cfg():
    return from_dict({
        "name": "baselines",
        "conditions": list(BASELINE_CONDITIONS),
        "backend": {"kind": "mock", "max_new_tokens": 16},
        "embedding": {"kind": "hash", "dim": 128},
        "dataset": {"name": "synthetic", "num_examples": 5, "num_sessions": 4,
                    "distractors_per_session": 2},
        "stats": {"bootstrap_resamples": 100, "permutation_resamples": 100},
    })


def test_all_baselines_build():
    comp = CompressionConfig()
    for name in BASELINE_CONDITIONS:
        spec = build_condition(name, comp)
        assert spec.name == name


def test_baseline_and_delta_tables_generated(tmp_path):
    out = str(tmp_path / "run")
    run_experiment(_cfg(), out)
    art = str(tmp_path / "art")
    res = generate(out, art)
    assert res["had_data"]

    baselines = open(os.path.join(art, "table_baselines.md")).read()
    assert "Executable baselines" in baselines
    assert "not a reproduction" in baselines
    for name in BASELINE_CONDITIONS:
        assert name in baselines
    assert os.path.exists(os.path.join(art, "table_baselines.csv"))

    deltas = open(os.path.join(art, "table_deltas.md")).read()
    assert "vs_memory_off" in deltas
    assert "vs_uncompressed_rag" in deltas


def test_deltas_present_in_aggregate(tmp_path):
    out = str(tmp_path / "run")
    run_experiment(_cfg(), out)
    agg = json.load(open(os.path.join(out, "aggregate.json")))
    ham = agg["ham_memory"]
    assert "delta_vs_memory_off_task_score" in ham
    assert "delta_vs_uncompressed_rag_physical_serialized_bytes" in ham
    # HAM stores fewer physical bytes than uncompressed RAG (negative delta).
    assert ham["delta_vs_uncompressed_rag_physical_serialized_bytes"] < 0
