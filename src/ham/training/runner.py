"""Stage-C fine-tuning experiment runner.

NEW DESIGN (per-leg training with HAM injected into ham leg's training):

Two legs are trained INDEPENDENTLY from the IDENTICAL baseline model:

- ``weights_only``  -> SFT on no-context prompts (``Question -> Answer``)
- ``ham_augmented`` -> SFT on context-augmented prompts (``Context + Question -> Answer``)

Both legs start from the same frozen checkpoint (step 0 invariant enforced), each
has its own optimizer/trajectory, and each is evaluated with its matching prompt
mode. The headline metric is cost-to-target per leg (optimizer steps / training
tokens / wall-clock to reach a common accuracy threshold T) plus the ham/weights
ratio.

Mirrors ``ham.runner.run_experiment``'s output contract (manifest + jsonl +
aggregate + stats + summary) but for the training curve.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from .. import stats
from ..backends import build_backend
from ..config import FinetuneExperimentConfig
from ..datasets import build_dataset
from ..embeddings import build_embedder
from ..manifest import build_manifest
from ..runner import _resolve_model_revision
from . import eval as eval_mod
from .corpus import build_corpus
from .protocol import FINETUNE_LEGS
from .target import cost_ratio, cost_to_target

WEIGHTS_ONLY = "weights_only"
HAM_AUGMENTED = "ham_augmented"


def _fair_control_fingerprint(cfg: FinetuneExperimentConfig) -> dict:
    """The training invariants shared by both legs (so the eval-time prompt
    mode is provably the sole variable). Recorded with a SHA-256 fingerprint."""
    shared = {
        "model_id": cfg.backend.model_id,
        "backend_kind": cfg.backend.kind,
        "trainer": cfg.finetune.trainer,
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "optimizer": cfg.finetune.optimizer,
        "learning_rate": cfg.finetune.learning_rate,
        "batch_size": cfg.finetune.batch_size,
        "max_steps": cfg.finetune.max_steps,
        "checkpoint_every": cfg.finetune.checkpoint_every,
        "token_budget": cfg.memory.token_budget,
        "retrieval_k": cfg.memory.retrieval_k,
        "embedding_kind": cfg.embedding.kind,
        "evaluator": "metrics.score_example:task_score=max(exact_match,contains_gold)",
    }
    blob = json.dumps(shared, sort_keys=True, default=str).encode()
    shared["fingerprint_sha256"] = hashlib.sha256(blob).hexdigest()
    return shared


def _drift_at_target(curve: list, target: float):
    """RMS weight drift at the first checkpoint whose accuracy reaches ``target``.

    The forgetting proxy of interest: the leg that reaches the target in fewer
    optimizer steps has accumulated less weight movement at that point. Returns
    the checkpoint's recorded ``drift_rms`` (None if the target is never reached
    or drift was not recorded, e.g. the mock trainer)."""
    for c in curve:
        if c.accuracy() >= target:
            return c.drift_rms
    return None


def _resolve_target(cfg: FinetuneExperimentConfig, leg_curves: dict) -> tuple[float, str]:
    ft = cfg.finetune
    if ft.target_accuracy is not None:
        return float(ft.target_accuracy), "absolute"
    # Parity mode: peak accuracy the weights_only leg actually achieves, minus a
    # non-inferiority margin. Computed from the REAL weights curve (not a mock
    # parameter), so the target is always achievable by the weights leg by
    # construction.
    from .target import parity_target
    tgt = parity_target(leg_curves, ft.parity_with, ft.noninferiority_delta)
    return float(tgt), f"parity_with_{ft.parity_with}_minus_delta_{ft.noninferiority_delta}"


def run_finetune(cfg: FinetuneExperimentConfig, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    backend = build_backend(cfg.backend)
    embedder = build_embedder(cfg.embedding)
    examples = build_dataset(cfg.dataset).load()
    corpus_facts = build_corpus(examples)

    # Verify step-0 baseline invariant (both legs identical with no-context)
    baseline_check = eval_mod.verify_step0_baseline(backend, cfg, embedder, corpus_facts, examples)

    # Reuse ONE model for both legs and reset it to the baseline weights between
    # legs (state_dict cached on CPU). The previous design called build_backend()
    # per leg, loading the model 3x without ever freeing it -- that exhausted
    # VRAM on a shared display GPU and hard-crashed it (needed a reboot). One
    # model + one optimizer at a time bounds peak memory.
    initial_state = None
    if cfg.finetune.trainer == "hf":
        from ..backends.hf import HFBackend
        from .hf import _INSTALL_HINT
        if not isinstance(backend, HFBackend):
            raise RuntimeError(
                "finetune.trainer == 'hf' requires backend.kind == 'hf' "
                f"(got {type(backend).__name__}). " + _INSTALL_HINT)
        initial_state = {k: v.detach().cpu().clone()
                         for k, v in backend.model.state_dict().items()}

    leg_curves = {}
    # Zero-shot held-out benchmark: evaluate general-knowledge QA (memory off) on
    # the pretrained baseline and after each leg's final weights, to measure how
    # much each arm has forgotten. Only applicable to the real (hf) trainer.
    from .zeroshot import eval_zeroshot
    zeroshot = {"baseline": None, "legs": {}}
    if cfg.finetune.trainer == "hf":
        zeroshot["baseline"] = eval_zeroshot(backend)["accuracy"]

    for leg in FINETUNE_LEGS:
        if cfg.finetune.trainer == "mock":
            from .mock import MockLegTrainer
            trainer = MockLegTrainer(leg, cfg, examples)
        else:  # hf: reset the shared model to the cached baseline for this leg
            from .hf import HFLegTrainer
            backend.model.load_state_dict(initial_state)
            trainer = HFLegTrainer(leg, backend, embedder, cfg, examples, corpus_facts)
        leg_curves[leg] = trainer.run()
        # Zero-shot on the final weights (diagnostic only; the forgetting metric
        # of record is per-checkpoint weight drift at the target, in the aggregate).
        if cfg.finetune.trainer == "hf":
            zeroshot["legs"][leg] = eval_zeroshot(backend)["accuracy"]
        # Free this leg's optimizer/activations before the next leg.
        del trainer
        if cfg.finetune.trainer == "hf":
            import gc
            gc.collect()
            try:
                if backend._torch.cuda.is_available():
                    backend._torch.cuda.empty_cache()
            except Exception:
                pass

    target, target_kind = _resolve_target(cfg, leg_curves)

    # --- manifest ----------------------------------------------------------
    fair_control = _fair_control_fingerprint(cfg)
    manifest = build_manifest(
        cfg.to_dict(), cfg.config_hash(),
        model_revision=_resolve_model_revision(cfg),
        extra={"backend_kind": cfg.backend.kind, "is_smoke": cfg.is_smoke,
               "experiment": "stage_c_finetune",
               "target_stage": cfg.stage.target_stage,
               "base_weights_changed": cfg.base_weights_changed,
               "integration_mode": cfg.stage.integration_mode,
               "trainable_router": cfg.stage.trainable_router,
               "trainer": cfg.finetune.trainer,
               "legs": FINETUNE_LEGS, "target_accuracy": target,
               "target_kind": target_kind,
               "n_checkpoints_per_leg": len(next(iter(leg_curves.values()))),
               "n_examples": len(examples),
               "n_corpus_facts": len(corpus_facts),
               "step0_baseline_check": baseline_check,
               "zeroshot_forgetting": zeroshot,
               "fair_control": fair_control})
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # --- curve.jsonl (one row per checkpoint x leg x example) --------------
    curve_path = os.path.join(out_dir, "curve.jsonl")
    with open(curve_path, "w") as jf:
        for leg, curve in leg_curves.items():
            for ckpt in curve:
                for r in ckpt.results:
                    jf.write(json.dumps({
                        "step": ckpt.step, "tokens_seen": ckpt.tokens_seen,
                        "train_loss": ckpt.train_loss, "drift_rms": ckpt.drift_rms,
                        "leg": leg, "example_id": r.example_id,
                        "question_type": r.question_type, "task_score": r.task_score,
                        "exact_match": r.exact_match, "correct": r.correct,
                        "prompt_tokens": r.prompt_tokens,
                        "retrieval_recall_at_k": r.retrieval_recall_at_k,
                    }) + "\n")

    aggregate = _aggregate(cfg, leg_curves, target, target_kind)
    _write_aggregate(out_dir, aggregate)
    stats_out = _compute_stats(cfg, leg_curves, target)
    with open(os.path.join(out_dir, "stats.json"), "w") as fh:
        json.dump(stats_out, fh, indent=2)

    summary = {
        "out_dir": out_dir, "is_smoke": cfg.is_smoke,
        "experiment": "stage_c_finetune", "target_stage": cfg.stage.target_stage,
        "base_weights_changed": cfg.base_weights_changed,
        "trainer": cfg.finetune.trainer, "legs": FINETUNE_LEGS,
        "target_accuracy": target, "target_kind": target_kind,
        "n_examples": len(examples),
        "n_checkpoints_per_leg": len(next(iter(leg_curves.values()))),
        "step0_baseline_check": baseline_check,
        "aggregate": aggregate, "stats": stats_out,
        "zeroshot": zeroshot,
        "cost_ratio": _overall_ratio(aggregate),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _aggregate(cfg: FinetuneExperimentConfig, leg_curves: dict, target: float,
               target_kind: str) -> dict:
    weights_cost = cost_to_target(leg_curves[WEIGHTS_ONLY], WEIGHTS_ONLY, target)
    weights_drift = _drift_at_target(leg_curves[WEIGHTS_ONLY], target)
    out: dict[str, dict] = {}
    for leg in FINETUNE_LEGS:
        cost = cost_to_target(leg_curves[leg], leg, target)
        final_prompt = _mean_final_prompt(leg_curves[leg], leg)
        drift_at_target = _drift_at_target(leg_curves[leg], target)
        entry = {
            "leg": leg, "n_examples": len(leg_curves[leg][0].results) if leg_curves[leg] else 0,
            "is_smoke": cfg.is_smoke, "target_accuracy": target, "target_kind": target_kind,
            "reached": cost["reached"],
            "final_accuracy": cost["final_accuracy"], "max_accuracy": cost["max_accuracy"],
            "optimizer_steps_to_target": cost["optimizer_steps_to_target"],
            "training_tokens_to_target": cost["training_tokens_to_target"],
            "drift_rms_at_target": drift_at_target,
            "final_prompt_tokens_mean": final_prompt,
            "cost_ratio_tokens": None, "cost_ratio_steps": None, "cost_ratio_drift": None,
        }
        if leg == HAM_AUGMENTED:
            entry["cost_ratio_tokens"] = cost_ratio(cost, weights_cost, "training_tokens_to_target")
            entry["cost_ratio_steps"] = cost_ratio(cost, weights_cost, "optimizer_steps_to_target")
            if drift_at_target is not None and weights_drift:
                entry["cost_ratio_drift"] = drift_at_target / weights_drift
        out[leg] = entry
    return out


def _mean_final_prompt(curve: list, leg: str) -> float | None:
    if not curve:
        return None
    vals = [r.prompt_tokens for c in curve for r in c.results]
    return float(np.mean(vals)) if vals else None


def _overall_ratio(aggregate: dict) -> dict:
    ham = aggregate.get(HAM_AUGMENTED, {})
    return {
        "tokens": ham.get("cost_ratio_tokens"),
        "steps": ham.get("cost_ratio_steps"),
        "drift": ham.get("cost_ratio_drift"),
        "interpretation": ("ham_augmented reached the target at this fraction of "
                           "weights_only's cost (<1.0 = HAM cheaper)."),
    }


def _write_aggregate(out_dir: str, aggregate: dict) -> None:
    with open(os.path.join(out_dir, "aggregate.json"), "w") as fh:
        json.dump(aggregate, fh, indent=2)
    if not aggregate:
        return
    fieldnames = ["leg", "n_examples", "is_smoke", "reached", "target_accuracy",
                  "target_kind", "final_accuracy", "max_accuracy",
                  "optimizer_steps_to_target", "training_tokens_to_target",
                  "drift_rms_at_target", "final_prompt_tokens_mean",
                  "cost_ratio_tokens", "cost_ratio_steps", "cost_ratio_drift"]
    with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for entry in aggregate.values():
            w.writerow({k: entry.get(k) for k in fieldnames})


def _per_example_correct(leg_curves: dict, leg: str, step: int) -> list[int]:
    for ckpt in leg_curves[leg]:
        if ckpt.step == step:
            return [r.correct for r in ckpt.results]
    return []


def _closest_step(curve: list, target_step: float | None) -> int | None:
    if target_step is None or not curve:
        return None
    return min((c.step for c in curve), key=lambda s: abs(s - target_step))


def _compute_stats(cfg: FinetuneExperimentConfig, leg_curves: dict, target: float) -> dict:
    scfg = cfg.stats
    out: dict = {"target_accuracy": target,
                 "noninferiority_delta": cfg.finetune.noninferiority_delta,
                 "per_checkpoint_ci": {}, "comparisons": {}}

    # Per-checkpoint accuracy CIs per leg (bootstrap over example correctness).
    for leg in FINETUNE_LEGS:
        rows = []
        for ckpt in leg_curves[leg]:
            correct = [r.correct for r in ckpt.results]
            ci = stats.mean_ci_bootstrap(correct, scfg.bootstrap_resamples, scfg.ci, scfg.seed)
            rows.append({"step": ckpt.step, "tokens_seen": ckpt.tokens_seen, **ci})
        out["per_checkpoint_ci"][leg] = rows

    # Paired ham vs weights at (a) the weights target step and (b) the final step.
    weights_cost = cost_to_target(leg_curves[WEIGHTS_ONLY], WEIGHTS_ONLY, target)
    steps_of_interest = []
    wt = _closest_step(leg_curves[WEIGHTS_ONLY], weights_cost["optimizer_steps_to_target"])
    if wt is not None:
        steps_of_interest.append(("at_weights_target", wt))
    if leg_curves[WEIGHTS_ONLY]:
        steps_of_interest.append(("at_final_step", leg_curves[WEIGHTS_ONLY][-1].step))
    for label, step in steps_of_interest:
        c_ham = _per_example_correct(leg_curves, HAM_AUGMENTED, step)
        c_w = _per_example_correct(leg_curves, WEIGHTS_ONLY, step)
        n = min(len(c_ham), len(c_w))
        if n == 0:
            continue
        a = [c_ham[i] for i in range(n)]
        b = [c_w[i] for i in range(n)]
        out["comparisons"][f"{label}_step{step}"] = {
            "metric": "task_correctness", "step": step,
            "paired_bootstrap_diff": stats.paired_bootstrap_diff(
                a, b, scfg.bootstrap_resamples, scfg.ci, scfg.seed),
            "paired_permutation": stats.paired_permutation_test(
                a, b, scfg.permutation_resamples, scfg.seed),
            "mcnemar": stats.mcnemar_test(a, b),
            "noninferiority": stats.noninferiority(
                a, b, cfg.finetune.noninferiority_delta,
                scfg.bootstrap_resamples, scfg.ci, scfg.seed),
        }
    return out
