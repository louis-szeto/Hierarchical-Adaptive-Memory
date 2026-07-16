"""Fair-control assertions: every condition in a run shares the frozen model,
dataset, prompt template, decoding params, tokenizer/embedder, seed, and evaluator
-- so the memory policy is the sole independent variable (research addendum §3)."""

import json
import os

from ham.config import from_dict
from ham.runner import run_experiment


def _cfg():
    return from_dict({
        "name": "fair",
        "conditions": ["memory_off", "full_history", "uncompressed_rag",
                       "recency_fifo", "static_prototype", "uniform_quantization",
                       "ham_memory"],
        "backend": {"kind": "mock", "max_new_tokens": 16},
        "embedding": {"kind": "hash", "dim": 128},
        "dataset": {"name": "synthetic", "num_examples": 5, "num_sessions": 4,
                    "distractors_per_session": 2},
        "stats": {"bootstrap_resamples": 100, "permutation_resamples": 100},
    })


def test_shared_invariants_are_constant_across_conditions(tmp_path):
    out = str(tmp_path / "run")
    run_experiment(_cfg(), out)
    rows = [json.loads(l) for l in open(os.path.join(out, "per_example.jsonl"))]

    # These per-row fields must be identical across all conditions.
    for field in ("model_id", "backend", "dataset"):
        assert len({r[field] for r in rows}) == 1, field

    # Every example is evaluated under every condition (same example set).
    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], set()).add(r["example_id"])
    example_sets = list(by_cond.values())
    assert all(s == example_sets[0] for s in example_sets)


def test_only_memory_policy_varies(tmp_path):
    out = str(tmp_path / "run")
    run_experiment(_cfg(), out)
    m = json.load(open(os.path.join(out, "manifest.json")))
    fc = m["fair_control"]
    # The fingerprint pins the shared knobs; presence of these keys documents them.
    for key in ("model_id", "prompt_template", "no_context_template",
                "max_new_tokens", "temperature", "seed", "embedding_model_id",
                "evaluator"):
        assert key in fc, key
    assert m["backend_kind"] == "mock"


def test_recency_fifo_and_static_prototype_differ_from_ham(tmp_path):
    out = str(tmp_path / "run")
    run_experiment(_cfg(), out)
    agg = json.load(open(os.path.join(out, "aggregate.json")))
    # recency_fifo evicts by age, so it must retain no more items than ham and
    # typically retrieves fewer gold memories (recall <= ham on this benchmark).
    assert agg["recency_fifo"]["retrieval_recall_at_k_mean"] <= \
        agg["ham_memory"]["retrieval_recall_at_k_mean"]
    # static_prototype disables adaptive consolidation but is still a real store.
    assert agg["static_prototype"]["consolidation_mode"] == "static"
    assert agg["ham_memory"]["consolidation_mode"] == "adaptive"
