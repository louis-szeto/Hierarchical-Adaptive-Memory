"""Stage-F architecture memory-block experiment runner.

For each task (recall/lm) x redundancy level x memory condition, train an
identical toy LM (only the memory policy differs) and record a curve of quality,
byte-honest memory size, and inference latency. The headline evidence is the
HAM/standard bytes- and latency-ratios versus redundancy (the slope proves
'frequency' is the mechanism). See ``docs/ARCHBENCH_PROTOCOL.md``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict

import numpy as np

from .. import stats
from ..config import ArchBenchExperimentConfig
from ..manifest import build_manifest
from .protocol import CONDITIONS, ArchCheckpoint
from .task import build_corpus

WEIGHTS = "standard_memory"
HAM = "ham_memory"


def _fair_control_fingerprint(cfg: ArchBenchExperimentConfig) -> dict:
    ab = cfg.archbench
    shared = {
        "trainer": ab.trainer, "task": ab.task, "dim": ab.dim, "n_layers": ab.n_layers,
        "vocab": ab.vocab, "n_heads": ab.n_heads, "top_k": ab.top_k,
        "capacity": ab.capacity, "optimizer": ab.optimizer,
        "learning_rate": ab.learning_rate, "batch_size": ab.batch_size,
        "max_steps": ab.max_steps, "checkpoint_every": ab.checkpoint_every,
        "seq_len": ab.seq_len, "seed": cfg.seed,
        "redundancy_levels": list(ab.redundancy_levels),
        "evaluator": "next-token accuracy over quality-masked positions",
    }
    blob = json.dumps(shared, sort_keys=True, default=str).encode()
    shared["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    return shared


def _tasks(cfg: ArchBenchExperimentConfig) -> list[str]:
    t = cfg.archbench.task
    return ["recall", "lm"] if t == "both" else [t]


def _n_items(cfg: ArchBenchExperimentConfig, task: str) -> int:
    ab = cfg.archbench
    # reasonable defaults: enough keys/motifs to be non-trivial
    return 16 if task == "recall" else 32


def _cost_to_target(curve: list[ArchCheckpoint], target: float) -> dict:
    reached = False
    for ckpt in curve:
        if ckpt.quality >= target:
            return {"reached": True, "step": ckpt.step,
                    "tokens_to_target": ckpt.tokens_seen,
                    "wall_to_target_s": ckpt.wall_clock_s,
                    "quality_at_target": ckpt.quality}
    last = curve[-1] if curve else None
    return {"reached": False, "step": None, "tokens_to_target": None,
            "wall_to_target_s": None, "quality_at_target": last.quality if last else None}


def run_archbench(cfg: ArchBenchExperimentConfig, out_dir: str) -> dict:
    from . import build_trainer
    os.makedirs(out_dir, exist_ok=True)
    ab = cfg.archbench
    all_curves: list[ArchCheckpoint] = []

    for task in _tasks(cfg):
        for r in ab.redundancy_levels:
            train_corpus = build_corpus(
                task, n_streams=ab.n_train_streams, seq_len=ab.seq_len,
                vocab=ab.vocab, n_items=_n_items(cfg, task), redundancy=r,
                seed=cfg.seed)
            for condition in ab.conditions:
                trainer = build_trainer(cfg, condition, r, train_corpus, ab.device)
                for ckpt in trainer.run():
                    ckpt.regime = task  # record which task/corpus this curve is for
                    all_curves.append(ckpt)

    # --- manifest ----------------------------------------------------------
    fair = _fair_control_fingerprint(cfg)
    manifest = build_manifest(
        cfg.to_dict(), cfg.config_hash(), extra={
            "backend_kind": "toy_architecture", "is_smoke": cfg.is_smoke,
            "experiment": "stage_f_archbench",
            "target_stage": cfg.stage.target_stage,
            "base_weights_changed": cfg.base_weights_changed,
            "integration_mode": cfg.stage.integration_mode,
            "trainer": ab.trainer, "task": ab.task,
            "conditions": ab.conditions, "redundancy_levels": list(ab.redundancy_levels),
            "target_quality": ab.target_quality, "n_curves": len(all_curves),
            "fair_control": fair})
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # --- curve.jsonl -------------------------------------------------------
    with open(os.path.join(out_dir, "curve.jsonl"), "w") as jf:
        for c in all_curves:
            jf.write(json.dumps({
                "task": c.regime, "step": c.step, "tokens_seen": c.tokens_seen,
                "wall_clock_s": c.wall_clock_s, "train_loss": c.train_loss,
                "quality": c.quality, "memory_bytes": c.memory_bytes,
                "inference_latency_s": c.inference_latency_s,
                "redundancy": c.redundancy, "condition": c.condition}) + "\n")

    aggregate = _aggregate(all_curves, cfg)
    _write_aggregate(out_dir, aggregate)
    stats_out = _compute_stats(all_curves, cfg)
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)

    summary = {
        "out_dir": out_dir, "is_smoke": cfg.is_smoke,
        "experiment": "stage_f_archbench", "trainer": ab.trainer, "task": ab.task,
        "target_quality": ab.target_quality,
        "n_curves": len(all_curves), "aggregate": aggregate, "stats": stats_out}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _agg_key(c: ArchCheckpoint) -> tuple:
    return (c.regime, c.redundancy, c.condition)


def _aggregate(curves: list[ArchCheckpoint], cfg) -> dict:
    by_key: dict[tuple, list[ArchCheckpoint]] = defaultdict(list)
    for c in curves:
        by_key[_agg_key(c)].append(c)
    out: dict[str, dict] = {}
    target = cfg.archbench.target_quality
    for (task, r, cond), ckpts in by_key.items():
        ckpts = sorted(ckpts, key=lambda c: c.step)
        final = ckpts[-1]
        cost = _cost_to_target(ckpts, target)
        peak_bytes = max(c.memory_bytes for c in ckpts)
        entry = {
            "task": task, "redundancy": r, "condition": cond,
            "n_checkpoints": len(ckpts), "is_smoke": cfg.is_smoke,
            "quality_final": final.quality,
            "quality_max": max(c.quality for c in ckpts),
            "memory_bytes_final": final.memory_bytes,
            "memory_bytes_peak": peak_bytes,
            "latency_final_s": final.inference_latency_s,
            "reached_target": cost["reached"],
            "tokens_to_target": cost["tokens_to_target"],
            "wall_to_target_s": cost["wall_to_target_s"],
            "bytes_ratio_vs_standard": None,
            "latency_ratio_vs_standard": None,
            "quality_delta_vs_standard": None,
        }
        out[f"{task}|r={r}|{cond}"] = entry
    # ratios vs standard_memory at the same (task, redundancy)
    for entry in out.values():
        std = out.get(f"{entry['task']}|r={entry['redundancy']}|{WEIGHTS}")
        if std and std["memory_bytes_peak"]:
            entry["bytes_ratio_vs_standard"] = entry["memory_bytes_peak"] / std["memory_bytes_peak"]
            entry["latency_ratio_vs_standard"] = (
                entry["latency_final_s"] / std["latency_final_s"]
                if std["latency_final_s"] else None)
            entry["quality_delta_vs_standard"] = entry["quality_final"] - std["quality_final"]
    return out


def _write_aggregate(out_dir: str, aggregate: dict) -> None:
    with open(os.path.join(out_dir, "aggregate.json"), "w") as fh:
        json.dump(aggregate, fh, indent=2)
    if not aggregate:
        return
    cols = ["task", "redundancy", "condition", "quality_final", "quality_max",
            "memory_bytes_peak", "latency_final_s", "reached_target",
            "tokens_to_target", "bytes_ratio_vs_standard",
            "latency_ratio_vs_standard", "quality_delta_vs_standard"]
    with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in aggregate.values():
            w.writerow({k: e.get(k) for k in cols})


def _compute_stats(curves: list[ArchCheckpoint], cfg) -> dict:
    """Per-redundancy ham-vs-standard comparison (the headline) + per-condition
    quality spread across redundancy levels."""
    ab = cfg.archbench
    by_key = defaultdict(list)
    for c in curves:
        by_key[_agg_key(c)].append(c)
    out = {"target_quality": ab.target_quality,
           "noninferiority_delta": 0.03, "redundancy_comparisons": []}
    for r in ab.redundancy_levels:
        for task in _tasks(cfg):
            std = sorted(by_key.get((task, r, WEIGHTS), []), key=lambda c: c.step)
            ham = sorted(by_key.get((task, r, HAM), []), key=lambda c: c.step)
            if not std or not ham:
                continue
            out["redundancy_comparisons"].append({
                "task": task, "redundancy": r,
                "standard_quality_final": std[-1].quality,
                "ham_quality_final": ham[-1].quality,
                "standard_bytes_peak": max(c.memory_bytes for c in std),
                "ham_bytes_peak": max(c.memory_bytes for c in ham),
                "standard_latency_final": std[-1].inference_latency_s,
                "ham_latency_final": ham[-1].inference_latency_s,
            })
    return out
