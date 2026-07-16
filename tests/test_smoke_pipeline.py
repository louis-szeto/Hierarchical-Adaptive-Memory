import json
import os

from ham.backends import build_backend
from ham.config import from_dict
from ham.report import generate
from ham.runner import run_experiment


def _cfg(**over):
    base = {
        "name": "test",
        "seed": 0,
        "conditions": ["memory_off", "full_history", "uncompressed_retrieval", "ham_memory"],
        "backend": {"kind": "mock", "max_new_tokens": 16},
        "embedding": {"kind": "hash", "dim": 128},
        "dataset": {"name": "synthetic", "num_examples": 6, "num_sessions": 4,
                    "distractors_per_session": 2},
        "stats": {"bootstrap_resamples": 200, "permutation_resamples": 200},
    }
    base.update(over)
    return from_dict(base)


def test_mock_backend_reads_answer_from_context():
    be = build_backend(_cfg().backend)
    prompt = ("Context:\nThe capital of Aurora is Verona.\n\n"
              "Question: What is the capital of Aurora?\nAnswer:")
    res = be.generate(prompt)
    assert "verona" in res.text.lower()
    assert res.prompt_tokens > 0
    # No context => cannot answer.
    res2 = be.generate("Question: What is the capital of Aurora?\nAnswer:")
    assert "verona" not in res2.text.lower()


def test_full_smoke_pipeline(tmp_path):
    cfg = _cfg()
    out = str(tmp_path / "run")
    summary = run_experiment(cfg, out)

    assert summary["is_smoke"] is True
    for fname in ("manifest.json", "per_example.jsonl", "aggregate.json",
                  "aggregate.csv", "stats.json", "summary.json"):
        assert os.path.exists(os.path.join(out, fname)), fname

    agg = summary["aggregate"]
    off = agg["memory_off"]["task_score_mean"]
    ham = agg["ham_memory"]["task_score_mean"]
    # Memory helps (or at least never hurts) vs no-memory on this fact-recall task.
    assert ham >= off
    assert ham > 0.0

    # Byte honesty: HAM physically smaller than uncompressed retrieval.
    assert (agg["ham_memory"]["physical_serialized_bytes_mean"]
            < agg["uncompressed_retrieval"]["physical_serialized_bytes_mean"])
    # Token economy: retrieval uses fewer prompt tokens than full history.
    assert (agg["ham_memory"]["prompt_tokens_mean"]
            < agg["full_history"]["prompt_tokens_mean"])

    # Report generation from real run data; figures watermarked as smoke.
    res = generate(out, str(tmp_path / "artifacts"))
    assert res["had_data"] and res["is_smoke"]
    md = open(os.path.join(res["out_dir"], "table_main.md")).read()
    assert "SMOKE TEST" in md


def test_report_refuses_to_invent_values(tmp_path):
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    res = generate(empty, str(tmp_path / "art"))
    assert res["had_data"] is False
    md = open(os.path.join(res["out_dir"], "table_main.md")).read()
    assert "EMPTY TEMPLATE" in md
    assert "n.a." in md


def test_conditions_share_identical_generation_params():
    # All conditions are built from the same config; the only thing that varies
    # is the memory spec, not the backend/generation settings.
    cfg = _cfg()
    assert cfg.backend.temperature == 0.0
    assert cfg.backend.max_new_tokens == 16
