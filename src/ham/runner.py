"""Experiment runner.

For each example, all conditions share the identical backend, embedder, prompt
template, generation params, seeds, and evaluator. Per-example rows are written
to JSONL; aggregate per-condition rows to CSV/JSON; statistics comparing HAM to
the baselines are computed at the end. Only real run outputs are written -- no
illustrative values anywhere.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

from . import metrics, stats
from .backends import build_backend
from .conditions import build_condition
from .config import ExperimentConfig
from .datasets import build_dataset
from .embeddings import build_embedder
from .instrumentation import CudaMemoryProbe, EnergyMeter, peak_cpu_rss_bytes
from .manifest import build_manifest
from .memory import HAMemory

PROMPT_TEMPLATE = (
    "You are a helpful assistant with access to memory from earlier conversations.\n"
    "Use the context to answer concisely with just the answer.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\nAnswer:"
)

NO_CONTEXT_TEMPLATE = (
    "You are a helpful assistant. Answer concisely with just the answer.\n\n"
    "Question: {question}\nAnswer:"
)


def _build_prompt(context: str, question: str) -> str:
    if context.strip():
        return PROMPT_TEMPLATE.format(context=context, question=question)
    return NO_CONTEXT_TEMPLATE.format(question=question)


def run_experiment(cfg: ExperimentConfig, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    backend = build_backend(cfg.backend)
    embedder = build_embedder(cfg.embedding)
    dataset = build_dataset(cfg.dataset)
    examples = dataset.load()

    # Manifest, including explicit lifecycle-stage fields and a fair-control
    # fingerprint: every condition in a run shares these invariants, so the
    # memory policy is the sole independent variable.
    fair_control = _fair_control_fingerprint(cfg)
    manifest = build_manifest(
        cfg.to_dict(), cfg.config_hash(),
        model_revision=_resolve_model_revision(cfg),
        extra={"backend_kind": cfg.backend.kind, "is_smoke": cfg.is_smoke,
               "n_examples": len(examples), "conditions": cfg.conditions,
               "dataset": cfg.dataset.name,
               "target_stage": cfg.stage.target_stage,
               "base_weights_changed": cfg.stage.base_weights_changed,
               "persistent_across_sessions": cfg.stage.persistent_across_sessions,
               "integration_mode": cfg.stage.integration_mode,
               "trainable_router": cfg.stage.trainable_router,
               "fair_control": fair_control},
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    per_example_path = os.path.join(out_dir, "per_example.jsonl")
    rows: list[dict] = []
    cuda_probe = CudaMemoryProbe(enabled=backend.supports_cuda_metrics())

    store_root = os.path.join(out_dir, "memory_stores")
    os.makedirs(store_root, exist_ok=True)

    with open(per_example_path, "w") as jf:
        for ex in examples:
            for cond_name in cfg.conditions:
                spec = build_condition(cond_name, cfg.compression)
                row = _run_one(cfg, backend, embedder, ex, spec, cuda_probe, store_root)
                rows.append(row)
                jf.write(json.dumps(row) + "\n")

    aggregate = _aggregate(rows, cfg)
    _write_aggregate(out_dir, aggregate)
    stats_out = _compute_stats(rows, cfg)
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)

    summary = {
        "out_dir": out_dir,
        "is_smoke": cfg.is_smoke,
        "n_examples": len(examples),
        "conditions": cfg.conditions,
        "aggregate": aggregate,
        "stats": stats_out,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _resolve_model_revision(cfg: ExperimentConfig) -> str | None:
    """Best-effort exact revision (snapshot commit) of a cached HF model, so the
    real-model PoC records the precise weights it ran. None for the mock backend
    or when the hash can't be resolved (never fabricated)."""
    if cfg.backend.kind != "hf":
        return None
    try:
        import glob
        import os

        from huggingface_hub.constants import HF_HUB_CACHE

        repo = "models--" + cfg.backend.model_id.replace("/", "--")
        snaps = glob.glob(os.path.join(HF_HUB_CACHE, repo, "snapshots", "*"))
        if snaps:
            return os.path.basename(sorted(snaps)[0])
    except Exception:
        return None
    return None


def _fair_control_fingerprint(cfg: ExperimentConfig) -> dict:
    """The invariants shared by every condition in a run. Recorded (and hashed)
    so a reviewer can verify that only the memory policy varied. See
    ``tests/test_fair_controls.py`` for the enforced assertions."""
    import hashlib

    shared = {
        "model_id": cfg.backend.model_id,
        "backend_kind": cfg.backend.kind,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "prompt_template": PROMPT_TEMPLATE,
        "no_context_template": NO_CONTEXT_TEMPLATE,
        "max_new_tokens": cfg.backend.max_new_tokens,
        "temperature": cfg.backend.temperature,
        "top_p": cfg.backend.top_p,
        "embedding_kind": cfg.embedding.kind,
        "embedding_model_id": cfg.embedding.model_id,
        "token_budget": cfg.memory.token_budget,
        "retrieval_k": cfg.memory.retrieval_k,
        "evaluator": "metrics.score_example:task_score=max(exact_match,contains_gold)",
    }
    blob = json.dumps(shared, sort_keys=True, default=str).encode()
    shared["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    return shared


def _run_one(cfg, backend, embedder, ex, spec, cuda_probe, store_root) -> dict:
    mem = HAMemory(cfg.memory, spec, embedder, seed=cfg.seed)
    energy = EnergyMeter(enabled=True)
    energy.start()
    cuda_probe.reset()

    # Ingest full multi-session history in order.
    for sid, turn in ex.all_turns():
        mem.ingest_turn(turn.content, session_id=sid, role=turn.role)

    # Build context + prompt.
    context, cdiag = mem.build_context(ex.question)
    prompt = _build_prompt(context, ex.question)

    gen = backend.generate(prompt)
    score = metrics.score_example(gen.text, ex.answer)
    mem.record_feedback(bool(score["task_score"] >= 1.0))

    # Retrieval quality vs gold memory identity (None when no gold ids exist).
    ret = metrics.retrieval_metrics(
        cdiag.get("retrieved_texts", []), ex.answer,
        getattr(ex, "gold_memory_texts", []), cfg.memory.retrieval_k)

    # Serialize the memory store to disk for real byte accounting.
    store_dir = os.path.join(store_root, f"{ex.example_id}__{spec.name}")
    acc = mem.serialize(store_dir)
    index_size = _dir_size(store_dir)

    # Per-item vector reconstruction error (paper Eq 8): mean over the records
    # actually serialized. None when there is no memory store (memory_off /
    # full_history) or no quantization was applied (kept back-compat / additive
    # diagnostic; not read into bytes/quality).
    q_errs = [r.quantization_error for r in mem.store.retrievable()
              if r.quantization_error is not None]
    mean_quantization_error = float(np.mean(q_errs)) if q_errs else None

    rss, rss_reason = peak_cpu_rss_bytes()
    cuda = cuda_probe.read()
    energy_stats = energy.stop()
    diag = mem.diagnostics()

    row = {
        "example_id": ex.example_id,
        "condition": spec.name,
        "question_type": ex.question_type,
        "dataset": cfg.dataset.name,
        "model_id": cfg.backend.model_id,
        "backend": cfg.backend.kind,
        "is_smoke": cfg.is_smoke,
        # Condition/stage metadata (for the target-stage/method/outcome table).
        "integration_mode": spec.integration_mode,
        "base_weights_changed": spec.base_weights_changed,
        "persistent": spec.persistent,
        "consolidation": spec.consolidation,
        "consolidation_mode": spec.consolidation_mode,
        "adaptive_precision": spec.adaptive_precision,
        "tiering": spec.tiering,
        "allocation": spec.allocation,
        "eviction": spec.eviction,
        "literature_analogue": spec.literature_analogue,
        # Retrieval quality (None when the dataset carries no gold memory ids).
        "retrieval_recall_at_k": ret["retrieval_recall_at_k"],
        "retrieval_mrr": ret["retrieval_mrr"],
        "retrieval_metrics_reason": ret["retrieval_metrics_reason"],
        # Quality
        "task_score": score["task_score"],
        "exact_match": score["exact_match"],
        "f1": score["f1"],
        "contains_gold": score["contains_gold"],
        "prediction": gen.text,
        "gold": ex.answer,
        # Tokens (exact tokenizer via the backend)
        "prompt_tokens": gen.prompt_tokens,
        "input_tokens": gen.prompt_tokens,
        "output_tokens": gen.output_tokens,
        "total_tokens": gen.total_tokens,
        # Bytes (physical vs logical)
        "logical_memory_bytes": acc.logical_bytes,
        "physical_serialized_bytes": acc.physical_bytes,
        "physical_text_bytes": acc.physical_text_bytes,
        "physical_vector_bytes": acc.physical_vector_bytes,
        "physical_meta_bytes": acc.physical_meta_bytes,
        "bytes_per_fact": acc.bytes_per_fact,
        "compression_ratio": acc.compression_ratio,
        "text_codec": acc.text_codec,
        "vector_quant": acc.vector_quant,
        "index_size_bytes": index_size,
        "n_retained_items": acc.n_items,
        # Per-item quantization distortion (paper Eq 8): mean over the records
        # in this example's store that were quantized. Additive diagnostic.
        "mean_quantization_error": mean_quantization_error,
        # Latency / throughput
        "retrieval_latency_s": cdiag["retrieval_latency_s"],
        "context_build_latency_s": cdiag["context_build_latency_s"],
        "prefill_latency_s": gen.prefill_latency_s,
        "decode_latency_s": gen.decode_latency_s,
        "total_latency_s": gen.total_latency_s,
        "tokens_per_second": gen.tokens_per_second,
        "n_retrieved": cdiag["n_retrieved"],
        # Tiers
        "tier_working": diag["tier_working"],
        "tier_episodic": diag["tier_episodic"],
        "tier_semantic": diag["tier_semantic"],
        "n_records": diag["n_records"],
        "n_prototypes": diag["n_prototypes"],
        # Resources
        "peak_cpu_rss_bytes": rss,
        "peak_cpu_rss_reason": rss_reason,
        "latency_source": gen.extra.get("latency", "measured"),
    }
    row.update(cuda)
    row.update(energy_stats)
    return row


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


_MEAN_FIELDS = [
    "task_score", "exact_match", "f1", "contains_gold",
    "prompt_tokens", "input_tokens", "output_tokens", "total_tokens",
    "logical_memory_bytes", "physical_serialized_bytes", "physical_text_bytes",
    "physical_vector_bytes", "bytes_per_fact", "compression_ratio",
    "retrieval_latency_s", "context_build_latency_s", "prefill_latency_s",
    "decode_latency_s", "total_latency_s", "tokens_per_second",
    "index_size_bytes", "n_retained_items", "n_retrieved",
    "tier_working", "tier_episodic", "tier_semantic", "n_prototypes",
    "peak_cpu_rss_bytes", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes",
    "energy_joules", "retrieval_recall_at_k", "retrieval_mrr",
    "mean_quantization_error",
]

# Condition metadata carried through aggregation (constant per condition).
_META_FIELDS = [
    "integration_mode", "base_weights_changed", "persistent", "consolidation",
    "consolidation_mode", "adaptive_precision", "tiering", "allocation",
    "eviction", "literature_analogue",
]


def _aggregate(rows: list[dict], cfg: ExperimentConfig) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    agg = {}
    for cond, crows in by_cond.items():
        entry = {"condition": cond, "n": len(crows), "is_smoke": cfg.is_smoke}
        for field in _META_FIELDS:
            entry[field] = crows[0].get(field)
        for field in _MEAN_FIELDS:
            vals = [r[field] for r in crows if r.get(field) is not None]
            if vals:
                entry[f"{field}_mean"] = float(np.mean(vals))
                entry[f"{field}_std"] = float(np.std(vals))
            else:
                entry[f"{field}_mean"] = None
                entry[f"{field}_std"] = None
        agg[cond] = entry
    _add_deltas(agg)
    return agg


_DELTA_FIELDS = ["task_score", "retrieval_recall_at_k", "retrieval_mrr",
                 "prompt_tokens", "physical_serialized_bytes", "total_latency_s"]


def _add_deltas(agg: dict) -> None:
    """Per-condition deltas vs the two anchor baselines (memory_off, uncompressed_rag).
    Uses uncompressed_rag if present, else its uncompressed_retrieval alias."""
    refs = {"vs_memory_off": "memory_off"}
    if "uncompressed_rag" in agg:
        refs["vs_uncompressed_rag"] = "uncompressed_rag"
    elif "uncompressed_retrieval" in agg:
        refs["vs_uncompressed_rag"] = "uncompressed_retrieval"
    for entry in agg.values():
        for label, ref_name in refs.items():
            ref = agg.get(ref_name)
            if ref is None:
                continue
            for field in _DELTA_FIELDS:
                a = entry.get(f"{field}_mean")
                b = ref.get(f"{field}_mean")
                if a is not None and b is not None:
                    entry[f"delta_{label}_{field}"] = float(a - b)


def _write_aggregate(out_dir: str, aggregate: dict) -> None:
    with open(os.path.join(out_dir, "aggregate.json"), "w") as fh:
        json.dump(aggregate, fh, indent=2)
    if not aggregate:
        return
    fieldnames = sorted({k for entry in aggregate.values() for k in entry})
    fieldnames = ["condition", "n", "is_smoke"] + [f for f in fieldnames
                                                   if f not in ("condition", "n", "is_smoke")]
    with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in aggregate.values():
            writer.writerow(entry)


def _paired(rows, cond, field):
    d = {}
    for r in rows:
        if r["condition"] == cond and r.get(field) is not None:
            d[r["example_id"]] = r[field]
    return d


def _compute_stats(rows: list[dict], cfg: ExperimentConfig) -> dict:
    conds = set(r["condition"] for r in rows)
    scfg = cfg.stats
    out: dict = {"noninferiority_delta": scfg.noninferiority_delta, "comparisons": {}}

    # Baseline preference: uncompressed_rag (canonical), then its alias, else full_history.
    baseline = None
    for cand in ("uncompressed_rag", "uncompressed_retrieval", "full_history"):
        if cand in conds:
            baseline = cand
            break

    target = "ham_memory" if "ham_memory" in conds else None
    if target and baseline:
        c = _paired(rows, target, "task_score")
        b = _paired(rows, baseline, "task_score")
        keys = sorted(set(c) & set(b))
        if keys:
            cs = [c[k] for k in keys]
            bs = [b[k] for k in keys]
            out["comparisons"][f"{target}_vs_{baseline}"] = {
                "metric": "task_score",
                "paired_bootstrap_diff": stats.paired_bootstrap_diff(
                    cs, bs, scfg.bootstrap_resamples, scfg.ci, scfg.seed),
                "paired_permutation": stats.paired_permutation_test(
                    cs, bs, scfg.permutation_resamples, scfg.seed),
                "mcnemar": stats.mcnemar_test(
                    [x >= 1.0 for x in cs], [x >= 1.0 for x in bs]),
                "noninferiority": stats.noninferiority(
                    cs, bs, scfg.noninferiority_delta, scfg.bootstrap_resamples,
                    scfg.ci, scfg.seed),
            }

    # H5: HAM vs random/uniform ablations at equal average rate.
    if target:
        for ablation in ("uniform_quantization", "random_tiering"):
            if ablation in conds:
                c = _paired(rows, target, "task_score")
                a = _paired(rows, ablation, "task_score")
                keys = sorted(set(c) & set(a))
                if keys:
                    cs = [c[k] for k in keys]
                    as_ = [a[k] for k in keys]
                    out["comparisons"][f"{target}_vs_{ablation}"] = {
                        "metric": "task_score",
                        "paired_bootstrap_diff": stats.paired_bootstrap_diff(
                            cs, as_, scfg.bootstrap_resamples, scfg.ci, scfg.seed),
                        "paired_permutation": stats.paired_permutation_test(
                            cs, as_, scfg.permutation_resamples, scfg.seed),
                    }

    # Per-condition task-score CIs.
    out["condition_score_ci"] = {}
    for cond in sorted(conds):
        vals = [r["task_score"] for r in rows if r["condition"] == cond]
        out["condition_score_ci"][cond] = stats.mean_ci_bootstrap(
            vals, scfg.bootstrap_resamples, scfg.ci, scfg.seed)
    return out
