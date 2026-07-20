"""Stage-F architecture memory-block experiment runner.

For each task (recall/lm) x redundancy level x memory condition, train an
identical toy LM (only the memory policy differs) and record a curve of quality
and byte-honest memory size. The headline evidence is the HAM/standard
bytes-ratio versus redundancy (the slope proves 'frequency' is the mechanism).
See ``docs/ARCHBENCH_PROTOCOL.md``.

This module ALSO carries the **fine-tuning post-hoc analysis** on the same toy
models (``finetune_posthoc`` block): comparing the standard flat memory block
(``standard_memory``) against the HAM memory block (``ham_memory``) on cost-to-
target + L2 weight drift (sqrt(sum((p - p_init)**2))). It is NOT the legacy
stage-C SmolLM2/external-retrieval fine-tune -- the toy is trained from
scratch, so there is no pretrained knowledge to forget and no zero-shot
forgetting arm; the diagnostic of interest is the cost-to-target and the
weight-drift overhead HAM's extra router/fusion/encoding parameters add at
matched quality.
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
from .cost import HAM as COST_HAM
from .cost import STANDARD as COST_STD
from .cost import cost_ratio, cost_to_target, parity_target
from .protocol import ArchCheckpoint
from .task import build_corpus

WEIGHTS = "standard_memory"
HAM = "ham_memory"

# Non-inferiority margin for the fine-tuning post-hoc parity target
# (max(standard_quality) - delta). Same value as the other experiments.
_NONINFERIORITY_DELTA = 0.03


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
                    "quality_at_target": ckpt.quality}
    last = curve[-1] if curve else None
    return {"reached": False, "step": None, "tokens_to_target": None,
            "quality_at_target": last.quality if last else None}


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
                    ckpt.task = task  # record which task/corpus this curve is for
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
                "task": c.task, "step": c.step, "tokens_seen": c.tokens_seen,
                "train_loss": c.train_loss,
                "quality": c.quality, "memory_bytes": c.memory_bytes,
                "redundancy": c.redundancy, "condition": c.condition,
                "drift_rms": c.drift_rms}) + "\n")

    aggregate = _aggregate(all_curves, cfg)
    finetune_posthoc = _finetune_posthoc(all_curves, cfg)
    aggregate["finetune_posthoc"] = finetune_posthoc
    _write_aggregate(out_dir, aggregate)
    stats_out = _compute_stats(all_curves, cfg)
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)

    summary = {
        "out_dir": out_dir, "is_smoke": cfg.is_smoke,
        "experiment": "stage_f_archbench", "trainer": ab.trainer, "task": ab.task,
        "target_quality": ab.target_quality,
        "n_curves": len(all_curves), "aggregate": aggregate, "stats": stats_out,
        "finetune_posthoc": finetune_posthoc,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _agg_key(c: ArchCheckpoint) -> tuple:
    return (c.task, c.redundancy, c.condition)


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
            "reached_target": cost["reached"],
            "tokens_to_target": cost["tokens_to_target"],
            "bytes_ratio_vs_standard": None,
            "quality_delta_vs_standard": None,
        }
        out[f"{task}|r={r}|{cond}"] = entry
    # ratios vs standard_memory at the same (task, redundancy)
    for entry in out.values():
        std = out.get(f"{entry['task']}|r={entry['redundancy']}|{WEIGHTS}")
        if std and std["memory_bytes_peak"]:
            entry["bytes_ratio_vs_standard"] = entry["memory_bytes_peak"] / std["memory_bytes_peak"]
            entry["quality_delta_vs_standard"] = entry["quality_final"] - std["quality_final"]
    return out


def _write_aggregate(out_dir: str, aggregate: dict) -> None:
    # aggregate.json carries the per-cell entries PLUS the nested
    # ``finetune_posthoc`` block (it is a separate, clearly-labeled analysis on
    # the same run). aggregate.csv only carries the per-cell rows (one row per
    # task x redundancy x condition), so split it out for the CSV writer.
    finetune_posthoc = aggregate.get("finetune_posthoc")
    with open(os.path.join(out_dir, "aggregate.json"), "w") as fh:
        json.dump(aggregate, fh, indent=2)
    per_cell = {k: v for k, v in aggregate.items()
                if k != "finetune_posthoc" and isinstance(v, dict)
                and "task" in v and "condition" in v}
    if per_cell:
        cols = ["task", "redundancy", "condition", "quality_final", "quality_max",
                "memory_bytes_peak", "reached_target",
                "tokens_to_target", "bytes_ratio_vs_standard",
                "quality_delta_vs_standard"]
        with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for e in per_cell.values():
                w.writerow({k: e.get(k) for k in cols})


def _finetune_posthoc(curves: list[ArchCheckpoint], cfg) -> dict:
    """Fine-tuning post-hoc analysis on the toy models: ``standard_memory`` vs
    ``ham_memory`` cost-to-target + L2 weight drift (sqrt(sum((p - p_init)**2))
    over all params). The toy LM is trained from scratch (no pretrained
    knowledge to forget -> no zero-shot forgetting arm); the diagnostic is the
    weight-drift overhead HAM's extra parameters add to reach the same target.

    Returns a dict keyed by ``(task, redundancy)`` plus a ``noninferiority_delta``
    field. Each cell carries the parity target, each arm's cost-to-target, the
    drift at target, and the HAM/standard ratios. The recall task is the
    primary headline (the toy associative-recall task the design preys on).
    """
    by_key: dict[tuple, list[ArchCheckpoint]] = defaultdict(list)
    for c in curves:
        by_key[_agg_key(c)].append(c)
    out: dict = {
        "noninferiority_delta": _NONINFERIORITY_DELTA,
        "description": (
            "Fine-tuning post-hoc on the stage-F toy models: standard flat "
            "memory block vs HAM memory block. Cost-to-target + L2 weight "
            "drift (sqrt(sum((p - p_init)**2)) over all params) at the parity "
            "target = max(standard_quality) - delta. The toy is trained from "
            "scratch (no forgetting arm)."),
        "primary_task": "recall",
        "cells": {},
    }
    for task in _tasks(cfg):
        for r in cfg.archbench.redundancy_levels:
            std_curve = sorted(by_key.get((task, r, COST_STD), []),
                               key=lambda c: c.step)
            ham_curve = sorted(by_key.get((task, r, COST_HAM), []),
                               key=lambda c: c.step)
            if not std_curve or not ham_curve:
                continue
            target = parity_target(std_curve, _NONINFERIORITY_DELTA)
            std_cost = cost_to_target(std_curve, target, interpolate=False)
            ham_cost = cost_to_target(ham_curve, target, interpolate=False)
            cell = {
                "task": task, "redundancy": r,
                "target_quality": target,
                "standard": {
                    "reached": std_cost["reached"],
                    "quality_at_target": std_cost["quality_at_target"],
                    "optimizer_steps_to_target": std_cost["optimizer_steps_to_target"],
                    "training_tokens_to_target": std_cost["training_tokens_to_target"],
                    "drift_rms_at_target": std_cost["drift_rms_at_target"],
                    "final_quality": std_cost["final_quality"],
                    "max_quality": std_cost["max_quality"],
                },
                "ham": {
                    "reached": ham_cost["reached"],
                    "quality_at_target": ham_cost["quality_at_target"],
                    "optimizer_steps_to_target": ham_cost["optimizer_steps_to_target"],
                    "training_tokens_to_target": ham_cost["training_tokens_to_target"],
                    "drift_rms_at_target": ham_cost["drift_rms_at_target"],
                    "final_quality": ham_cost["final_quality"],
                    "max_quality": ham_cost["max_quality"],
                },
                "cost_ratio_steps_ham_over_standard": cost_ratio(
                    ham_cost, std_cost, "optimizer_steps_to_target"),
                "cost_ratio_tokens_ham_over_standard": cost_ratio(
                    ham_cost, std_cost, "training_tokens_to_target"),
                "drift_ratio_ham_over_standard": (
                    (ham_cost["drift_rms_at_target"] / std_cost["drift_rms_at_target"])
                    if (ham_cost["reached"] and std_cost["reached"]
                        and ham_cost["drift_rms_at_target"] is not None
                        and std_cost["drift_rms_at_target"] not in (None, 0.0))
                    else None),
            }
            out["cells"][f"{task}|r={r}"] = cell
    return out


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
            })
    return out
