"""Stage-D KV-cache-compression experiment runner.

For each redundancy level x condition, compress a frozen model's KV cache and
record byte-honest size and next-token quality. The headline evidence is the
HAM/full bytes-ratio and quality versus redundancy (the slope proves 'frequency'
is the mechanism). Wall-clock latency is not reported (not universal). See
``docs/KVBENCH_PROTOCOL.md``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from ..manifest import build_manifest
from ..config import KVBenchExperimentConfig
from .protocol import BASELINE, KVResult

HAM = "ham_kv"


def _fair_control_fingerprint(cfg: KVBenchExperimentConfig) -> dict:
    kb = cfg.kvbench
    shared = {
        "trainer": kb.trainer, "model_id": cfg.backend.model_id,
        "conditions": list(kb.conditions), "redundancy_levels": list(kb.redundancy_levels),
        "context_len": kb.context_len, "n_contexts": kb.n_contexts,
        "keep_ratios": list(kb.keep_ratios), "cluster_radius": kb.cluster_radius,
        "kv_bits": kb.kv_bits, "seed": cfg.seed,
        "evaluator": "next-token top-1 agreement vs full_kv + accuracy vs ground truth",
    }
    blob = json.dumps(shared, sort_keys=True, default=str).encode()
    shared["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    return shared


def run_kvbench(cfg: KVBenchExperimentConfig, out_dir: str) -> dict:
    from . import build_trainer
    os.makedirs(out_dir, exist_ok=True)
    trainer = build_trainer(cfg)
    results: list[KVResult] = trainer.run()

    manifest = build_manifest(
        cfg.to_dict(), cfg.config_hash(), extra={
            "backend_kind": cfg.backend.kind, "is_smoke": cfg.is_smoke,
            "experiment": "stage_d_kvbench",
            "target_stage": cfg.stage.target_stage,
            "base_weights_changed": cfg.base_weights_changed,
            "integration_mode": cfg.stage.integration_mode,
            "trainer": cfg.kvbench.trainer,
            "conditions": cfg.kvbench.conditions,
            "redundancy_levels": list(cfg.kvbench.redundancy_levels),
            "n_results": len(results), "fair_control": _fair_control_fingerprint(cfg)})
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    with open(os.path.join(out_dir, "results.jsonl"), "w") as jf:
        for r in results:
            jf.write(json.dumps({
                "condition": r.condition, "redundancy": r.redundancy,
                "keep_ratio": r.keep_ratio, "context_id": r.context_id,
                "kv_bytes": r.kv_bytes, "n_positions": r.n_positions,
                "quality_agreement": r.quality_agreement,
                "quality_accuracy": r.quality_accuracy}) + "\n")

    aggregate = _aggregate(results, cfg)
    _write_aggregate(out_dir, aggregate)
    stats_out = _compute_stats(results, cfg)
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)

    summary = {"out_dir": out_dir, "is_smoke": cfg.is_smoke,
               "experiment": "stage_d_kvbench", "trainer": cfg.kvbench.trainer,
               "n_results": len(results), "aggregate": aggregate, "stats": stats_out}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _aggregate(results: list[KVResult], cfg) -> dict:
    by: dict[tuple, list[KVResult]] = defaultdict(list)
    for r in results:
        by[(r.redundancy, r.condition, r.keep_ratio)].append(r)
    out: dict[str, dict] = {}
    for (r, cond, kr), rows in by.items():
        out[f"r={r}|{cond}|kr={kr}"] = {
            "redundancy": r, "condition": cond, "keep_ratio": kr, "n": len(rows),
            "is_smoke": cfg.is_smoke,
            "kv_bytes_mean": float(np.mean([x.kv_bytes for x in rows])),
            "quality_agreement_mean": float(np.mean([x.quality_agreement for x in rows])),
            "quality_accuracy_mean": float(np.mean([x.quality_accuracy for x in rows])),
            "n_positions_mean": float(np.mean([x.n_positions for x in rows])),
            "bytes_ratio_vs_full": None,
            "quality_delta_vs_full": None,
        }
    for entry in out.values():
        full = out.get(f"r={entry['redundancy']}|{BASELINE}|kr=1.0")
        if full and full["kv_bytes_mean"]:
            entry["bytes_ratio_vs_full"] = entry["kv_bytes_mean"] / full["kv_bytes_mean"]
            entry["quality_delta_vs_full"] = entry["quality_agreement_mean"] - full["quality_agreement_mean"]
    return out


def _write_aggregate(out_dir: str, aggregate: dict) -> None:
    with open(os.path.join(out_dir, "aggregate.json"), "w") as fh:
        json.dump(aggregate, fh, indent=2)
    if not aggregate:
        return
    cols = ["redundancy", "condition", "keep_ratio", "kv_bytes_mean",
            "bytes_ratio_vs_full",
            "quality_agreement_mean", "quality_delta_vs_full", "quality_accuracy_mean"]
    with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in aggregate.values():
            w.writerow({k: e.get(k) for k in cols})


def _compute_stats(results: list[KVResult], cfg) -> dict:
    """Per-redundancy Pareto: at each keep_ratio, ham vs the frequency-agnostic
    conditions (iso-budget quality). The headline is ham's quality at the smallest
    bytes growing with redundancy."""
    by = defaultdict(list)
    for r in results:
        by[(r.redundancy, r.condition, r.keep_ratio)].append(r)
    out = {"noninferiority_delta": 0.03, "pareto": []}
    for r in cfg.kvbench.redundancy_levels:
        for kr in cfg.kvbench.keep_ratios:
            row = {"redundancy": r, "keep_ratio": kr}
            for cond in cfg.kvbench.conditions:
                rows = by.get((r, cond, kr))
                if rows:
                    row[f"{cond}_bytes"] = float(np.mean([x.kv_bytes for x in rows]))
                    row[f"{cond}_agreement"] = float(np.mean([x.quality_agreement for x in rows]))
            out["pareto"].append(row)
    return out
