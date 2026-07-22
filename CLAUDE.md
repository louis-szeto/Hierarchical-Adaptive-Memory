# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **research experiment harness** for an academic paper, not an application. It runs a controlled
study of *persistent memory for a frozen LLM* framed as a rate-distortion / information-bottleneck /
MDL problem (the "HAM" system), measuring quality vs. tokens / physical bytes / latency against an
identical frozen-LLM-with-memory-disabled control and a battery of ablations.

Everything is built around one principle that overrides normal instincts: **the frozen LLM is the
sole independent variable that must never change; the memory policy is the only thing that varies
between conditions.** When in doubt, preserve that invariant.

## House rules — read before changing anything that produces numbers

These rules are enforced in code and asserted by tests; violating them silently is the worst
possible outcome. `docs/EXPERIMENT_PROTOCOL.md` is the authoritative source.

- **Fabricate nothing.** Report generators (`src/ham/report.py`) transform *only* real run files. With
  no data they emit clearly-labeled **EMPTY TEMPLATES** and create no figures. Never invent numbers.
- **Mock output is not a result.** Any run with `backend.kind: mock` produces `is_smoke: true` and its
  tables/figures are watermarked `SMOKE TEST`. Never present mock output as a scientific result. A real
  HF-model run is labeled `REAL MODEL, PROOF OF CONCEPT` (still not publication evidence).
- **Byte-honesty.** Headline bytes are **physical serialized bytes** from `os.path.getsize` of the
  store actually written to disk (`src/ham/compression/serialize.py`). Logical/uncompressed float32
  bytes are only an upper-bound sanity check, never the headline number.
- **Fail loudly.** Unknown config keys (`config._coerce`), missing/offline datasets, unavailable
  quantization, and a missing torch for `ham.architecture` all raise with actionable install guidance.
  Never silently fall back to a different condition.
- **Fair-control fingerprint.** The runner writes a `fair_control` block (shared model/backend/dataset/
  seed/prompt/decoding/embedder/evaluator invariants + SHA-256) into `manifest.json`. Every condition
  in a run must share them. `tests/test_fair_controls.py` asserts this.
- **Two artifacts, kept apart.** The runnable PoC (`ham.*`, lifecycle stage E, frozen weights,
  `external_context`) and the optional architecture prototype (`ham.architecture.*`, stage F,
  `hidden_state_fusion`, needs torch, toy-tested only) are deliberately separate. Do not wire
  `ham.architecture` into benchmark runs, and do not claim hidden-state injection into arbitrary HF
  models — it only attaches to self-contained toy blocks.

## Commands

Install is layered by optional extras; **CI uses core + zstd + dev only (no heavy ML deps)**.

```bash
pip install -e .                                 # core: mock backend + synthetic + compression + stats + report (numpy/scipy/pandas/pyyaml)
pip install -e ".[zstd,dev]"                     # CI config — stdlib-beating text codec + pytest
pip install -e ".[hf,faiss,zstd,datasets,plot,instr,dev]"   # full research stack (real model, FAISS, plotting, energy)

make test          # full suite: pytest (unit + smoke-pipeline + architecture + fair-control tests)
make smoke         # tiny mock run -> results/smoke
make smoke-figures # regenerate watermarked tables/figures from results/smoke
python -m pytest tests/test_quantization.py -q                       # one test file
python -m pytest tests/test_serialize.py::test_name -q               # one test
```

CLI (`python -m ham.cli …`, installed as the `ham` script):

```bash
ham run    --config configs/<x>.yaml --out results/<x> [--conditions a,b,c] [--limit N]
ham report --run-dir results/<x> --out results/<x>/paper_artifacts   # alias: ham export
ham info                                      # env + package versions
ham arch-demo [--block transformer|recurrent] [--fusion cross_attention|gated_residual]   # stage-F toy demo, needs torch
```

## How a run works (the pipeline to keep in your head)

`ham run` → `runner.run_experiment(cfg, out_dir)` (`src/ham/runner.py`). One config fully determines
a run. For **every example × every condition** (nested loop), the same code path runs:

1. Build a fresh `HAMemory(cfg.memory, spec, embedder, seed)` — one instance per example.
2. `ingest_turn` the full multi-session history in order → chunk → embed → tier (working/episodic/
   semantic) → importance-score → assign bits → optionally consolidate into prototypes / FIFO-evict.
3. `build_context(question)` → retrieve top-k → budget-bound join (whitespace-token proxy; the
   backend does exact token counting).
4. `backend.generate(prompt)` → `metrics.score_example` (exact-match / contains-gold / F1) +
   `metrics.retrieval_metrics` vs gold memory ids (when the dataset has them).
5. `mem.serialize(store_dir)` → physically write the store grouped by precision, sum real bytes.

Each iteration writes one JSONL row to `per_example.jsonl`. After the loop: `aggregate.json`/`.csv`
(per-condition means + deltas vs `memory_off` and `uncompressed_rag`), `stats.json` (paired bootstrap
/ permutation / McNemar / non-inferiority via `src/ham/stats.py`), `summary.json`, and `manifest.json`.

`ham report` consumes a finished run dir and emits `table_main/baselines/deltas/stats` (.md/.csv) and
figures into a `paper_artifacts/` subdir — reading only the real run files.

## The central abstraction: `ConditionSpec`

`src/ham/conditions.py` is the **single source of truth** for what each condition is. `HAMemory` is
*parameterized* by a `ConditionSpec`, so HAM, every baseline, and every ablation run through the same
code — only the spec knobs differ (`use_memory`, `mode`, `retrieval_method`, `consolidation`,
`eviction`, `use_recency/novelty/reuse`, `allocation`, `tiering`, `vector_quant`, `text_codec`).
- `CONDITION_NAMES` = all valid condition strings; `BASELINE_CONDITIONS` = the headline fair-comparison set.
- `build_condition(name, comp)` returns the frozen spec. Adding a condition means adding it here, to
  `CONDITION_NAMES`, and (if a headline baseline) to `BASELINE_CONDITIONS`.
- Lifecycle/literature metadata on each spec (`integration_mode`, `base_weights_changed`,
  `literature_analogue`, …) flows into the manifest and the baselines table; every entry is an
  *"implemented analogue, not a reproduction"* of any external system.

## Module map

```
src/ham/
  config.py         typed dataclasses from YAML; strict validation (fails loudly on unknown keys)
  cli.py            run / report(export) / info / arch-demo
  runner.py         the example×condition loop; per_example.jsonl, aggregate, stats, manifest, summary
  conditions.py     ConditionSpec registry — the definition of every condition/ablation
  metrics.py        score_example (EM/F1/contains_gold) + retrieval_metrics (recall@k / MRR)
  stats.py          paired bootstrap / permutation / McNemar / non-inferiority
  instrumentation.py  peak RSS, CUDA probe, codecarbon energy
  manifest.py       provenance: versions, config hash, git commit, stage fields, fair_control fingerprint
  report.py         run-dir -> watermarked tables/figures (EMPTY TEMPLATES when no data)
  embeddings.py     hash (deterministic) or sentence-transformers; Matryoshka truncation
  backends/         mock (fully deterministic) + hf (real causal LM, exact tokenizer)
  memory/           ham.py (orchestrator), store.py (tiers/MemoryRecord), importance.py (signals+bits),
                    consolidation.py (leader-clustering prototypes), retrieval.py (cosine/faiss/lexical)
  compression/      text_codec.py (zstd/zlib/raw), vector_quant.py (int8/int4/PQ), serialize.py (ByteAccounting, real bytes)
  datasets/         synthetic (local, carries gold-memory ids), longmemeval (loud errors if missing)
  architecture/     OPTIONAL stage-F torch prototype (TierState/MemoryRouter/fusion/HAMBlock) + toy demo
  archbench/        OPTIONAL stage-F toy-architecture memory-block compression experiment (FlatMemory/HamMemory, mock/torch trainers, own runner + report + fine-tuning post-hoc on standard-vs-HAM memory blocks)
  kvbench/          OPTIONAL stage-D KV-cache compression experiment on a frozen HF model (mock/torch trainers, own runner + report)
configs/            one YAML per experiment (smoke, synthetic, longmemeval, poc_real_smollm, publication_7b, archbench_smoke, archbench_toy, kvbench_smoke, kvbench_smollm)
tests/              mirrors the modules; test_fair_controls.py and test_baselines_report.py guard the integrity rules
docs/               EXPERIMENT_PROTOCOL, REPRODUCIBILITY, METRICS_SCHEMA, STAGE_TAXONOMY, ARCHITECTURE, BASELINE_CROSSWALK
```

## Stage-F architecture memory-block experiment (kept separate)

`ham.archbench.*` is an **optional second experiment** (lifecycle stage F) and the
mechanistic proof of the *compression* thesis: identical toy LMs differ ONLY in
their memory-block policy (`FlatMemory` vs HAM-compressed `HamMemory` with int4
prototypes + frequency-driven consolidation). See `docs/ARCHBENCH_PROTOCOL.md`.

- **Single variable:** the memory policy (`no_memory`/`standard_memory`/`ham_memory`
  + `ham_uniform`/`ham_no_consolidation`/`ham_random_alloc` ablations). Iso-quality
  comparisons on byte-honest memory size + inference latency; the corpus redundancy
  is the lever that should make HAM's advantage scale.
- **Config:** `ArchBenchExperimentConfig` + `ArchBenchConfig` (loaded by
  `load_archbench_config`), separate from the other configs. `is_smoke` is
  `trainer == "mock"`; `base_weights_changed` is `trainer == "torch"`.
- **Trainers:** `MockArchTrainer` (no torch → SMOKE TEST) and `TorchArchTrainer`
  (trains a `ToyMemoryLM` that reuses `ham.architecture.layer`). Stores are
  `FlatMemory`/`HamMemory` (`memory.py`); `byte_size()` is byte-honest via
  `compression.vector_quant`.
- **Entry points:** `ham archbench` / `ham archbench-report`.
- **Per-item memory (all hypotheses hold):** the trainer writes every token's
  hidden state to a FIFO window; `HamMemory` consolidates the window into per-key
  int4 prototypes. Prototype count = distinct keys in the window, so the bytes-ratio
  vs `standard_memory` scales with corpus redundancy (the proof that frequency is
  the mechanism). Ablations (`ham_uniform`/`ham_no_consolidation`/`ham_random_alloc`)
  isolate precision + consolidation. See `docs/ARCHBENCH_PROTOCOL.md`.
- **Fine-tuning post-hoc (built into archbench, NOT a separate experiment):** the
  runner also compares `standard_memory` vs `ham_memory` on cost-to-target +
  L2 weight drift (`sqrt(sum((p - p_init)**2))` over all params) on the same toy
  models. `TorchArchTrainer` snapshots the initial params and records `drift_rms`
  on every `ArchCheckpoint`; `MockArchTrainer` emits a synthetic sqrt(step) drift
  curve. The runner builds a `finetune_posthoc` block in `aggregate.json`/
  `summary.json` (parity target = max(standard_quality) − 0.03; cost = first
  checkpoint at-or-above it; HAM/standard step/token/drift ratios); the report
  emits `table_finetune_posthoc.md`/`.csv`. The toy is trained from scratch, so
  there is no zero-shot forgetting arm — the diagnostic is the drift overhead
  HAM's extra router/fusion/encoding parameters add to reach the same quality.
  Pure-math curve helpers live in `archbench/cost.py`.

## Stage-D KV-cache compression experiment (kept separate)

`ham.kvbench.*` is an **optional third experiment** (lifecycle stage D): compress
a frozen HF model's KV cache and measure byte-honest size + decode latency +
next-token quality. See `docs/KVBENCH_PROTOCOL.md`.

- **Single variable:** the KV-compression policy (`full_kv`/`ham_kv`/
  `uniform_quant_kv`/`h2o_kv`/`random_evict_kv`/`ham_no_cluster`); context
  redundancy is the lever. The model is **frozen** (`base_weights_changed=False`,
  `integration_mode="kv_cache_compression"`).
- **Mechanism:** extract the legacy KV cache (`DynamicCache.to_legacy_cache`),
  derive one position structure from layer-0 K (cluster/evict), apply it uniformly
  across all layers (so the rebuilt `DynamicCache` is consistent), int4 via
  `compression.vector_quant`.
- **Config:** `KVBenchExperimentConfig` + `KVBenchConfig` (`load_kvbench_config`),
  separate. `is_smoke` is `trainer == "mock"`.
- **Entry points:** `ham kvbench` / `ham kvbench-report`.
- **Result:** HAM dominates the quality-vs-bytes Pareto at high redundancy on the
  real model (`ham_kv` = frequency-weighted real-position retention + int4 beats
  random/norm selection at every keep-ratio; `ham_no_cluster` ablation isolates
  frequency). See `docs/KVBENCH_PROTOCOL.md`. An earlier `ham_kv` that
  merged/averaged positions corrupted real-model attention — keeping real
  positions selected by frequency is the version that works.

## Conventions when editing

- Configs are dataclasses in `config.py`; adding a field requires it to flow through `to_dict`/
  `config_hash` and usually into `_MEAN_FIELDS`/`_META_FIELDS` in `runner.py` and the metrics schema doc.
- Keep all seeds explicit and propagated (dataset, embedding, generation, stats). Default decoding is
  greedy (`temperature: 0.0`) for determinism.
- Real HF models are instantiated **only** when a config sets `backend.kind: hf` on an explicit
  `ham run` — never in tests or CI.
- Python ≥ 3.10; the repo uses `from __future__ import annotations` and PEP 604 unions throughout.
