# Fine-Tuning Experiment Protocol (stage C)

This document pre-registers the design, hypotheses, fairness controls, and
failure criteria for the **stage-C fine-tuning** experiment. It is the
*training-time* analogue of the stage-E (frozen-weight) study in
`EXPERIMENT_PROTOCOL.md`, and is deliberately kept separate from it: fine-tuning
modifies the weights (lifecycle stage C), so it does **not** share the stage-E
"the frozen LLM is the sole independent variable" invariant. Its own invariant is
stated below.

## Research question

How many fine-tuning **tokens / wall-clock seconds / optimizer steps** does it
take to reach a target knowledge accuracy, **with HAM external memory available
at eval** versus **without**? I.e. does external memory let a model reach a given
knowledge-accuracy target with substantially less weight-memorization?

## Design (memory-augments-eval + cost-to-target)

Both arms are fine-tuned **identically** on the same fact corpus. The **only**
difference is the eval-time prompt:

- `weights_only` — eval with `memory_off` (question only). The model must recall
  facts from its weights.
- `ham_augmented` — eval with `ham_memory` (HAM-retrieved context in the prompt).
  The model recalls via memory.

At every checkpoint both arms are evaluated against the **current** weights, so
the `weights_only` curve climbs from ~0 as facts are memorized while
`ham_augmented` starts at retrieval-only accuracy and rises with training. The
headline metric is **cost-to-target**: the first checkpoint at which an arm's
accuracy reaches the target, reported as (optimizer steps, training tokens,
wall-clock), plus the ratio `ham / weights`.

## Target accuracy

Two modes (config: `finetune.target_accuracy`):

- **Absolute** — a fixed value in [0, 1].
- **Parity (default)** — `max(weights_only accuracy) − δ`, i.e. "reach the
  accuracy that full memorization eventually achieves, less a non-inferiority
  margin δ" (default 0.03). This is the defensible default: it asks how much
  cheaper HAM makes reaching the level memorization attains.

If an arm never crosses the target, its `reached` flag is `False` and the cost is
`None` — reported honestly, never fabricated.

## Fairness controls (stage-C invariant)

Both arms share: identical initial weights + revision, identical SFT data, the
same optimizer / learning rate / batch size, the same seed, the same checkpoint
schedule, and the same eval examples. The **eval-time memory policy is the sole
independent variable.** The runner writes a `fair_control` block into the
manifest (training invariants + SHA-256 fingerprint). Because the weights are
modified, `base_weights_changed` is recorded truthfully (`true` only under the
real `hf` trainer; the `mock` trainer changes no weights and is watermarked).

## Falsifiable hypotheses

- **FT1 (cost-to-target reduction).** `ham_augmented` reaches the target in
  fewer training tokens / wall-clock / optimizer steps than `weights_only`;
  ratio < 1. *Falsified if* the ratio ≥ 1 (HAM does not reduce the cost).
- **FT2 (parity achievable).** `ham_augmented` reaches `max(weights_only) − δ`.
  *Falsified if* HAM cannot reach parity.
- **FT3 (no harm).** `ham_augmented` final accuracy ≥ `weights_only` final
  accuracy (memory never hurts at equal training). *Falsified if* HAM finishes
  below weights.

## Statistics

Per-checkpoint accuracy CIs (bootstrap over example correctness); paired
bootstrap / permutation / McNemar of `ham_augmented` vs `weights_only` at (a) the
`weights_only` target step and (b) the final step; a non-inferiority test for FT2
with margin δ. Reuses `ham.stats`.

## Rigor / honesty rules (inherited from stage E)

- **Fabricate nothing.** The finetune report transforms only real run files; with
  no data it emits clearly-labeled EMPTY TEMPLATES and no figure.
- **Mock output is not a result.** The deterministic `mock` trainer produces a
  synthetic curve; its tables/figures are watermarked `SMOKE TEST`. Only the `hf`
  trainer changes weights and produces unwatermarked (but still proof-of-concept,
  not publication) output.
- **Fail loudly.** Unknown config keys, `trainer: hf` without `backend.kind: hf`,
  a missing torch, or an empty corpus raise rather than silently degrade.
- **No superiority claim over production systems.** As with stage E, the
  contribution is the analysis framework and the cost-vs-accuracy
  characterization, not beating external memory systems.

## Novelty boundary

This experiment does **not** claim HAM is a better *training* method than
fine-tuning. It characterizes the **substitution** between weight-memorization
and external retrieval: how much fine-tuning effort external memory can offset
at equal accuracy. The weights-only arm is standard SFT; the HAM arm is the
stage-E memory reused verbatim at eval.

## Commands

```bash
make finetune-smoke   # mock trainer, deterministic, watermarked SMOKE TEST
make finetune-real    # SmolLM2-135M-Instruct SFT (needs pip install -e ".[hf]"; slow on CPU)
# or directly:
ham finetune        --config configs/finetune_smollm.yaml --out results/finetune_smollm
ham finetune-report --run-dir results/finetune_smollm     --out results/finetune_smollm/artifacts
```

Outputs: `manifest.json` (stage C), `curve.jsonl` (per checkpoint × arm ×
example), `aggregate.json/.csv` (cost-to-target + ratio per arm), `stats.json`
(CIs + paired tests), `summary.json`. Report: `table_cost`, `table_curve`,
`table_stats`, `fig_accuracy_vs_tokens`.
