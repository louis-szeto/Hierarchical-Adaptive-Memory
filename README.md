# HAM — Utility-Rate Adaptive Memory for Frozen LLMs

A reproducible experiment harness that treats **persistent memory for a frozen
language model** as a **rate-distortion / information-bottleneck / minimum-description-length**
problem. HAM organizes observations into human-inspired **working / episodic /
semantic** tiers, **consolidates** episodes into prototypes online, allocates
storage **bits per item by an information-utility signal**, and reports
**physically serialized bytes** (not merely estimated) alongside tokens, latency,
and quality — compared against an **identical frozen LLM with memory disabled**,
an uncompressed-retrieval baseline, and a battery of ablations.

> **Scope & honesty.** This repo produces *your own* experimental numbers. It ships
> **no benchmark results**. Anything generated from the deterministic **mock**
> backend is watermarked `SMOKE TEST` and is **not** a scientific result. We make
> **no Shannon-optimality claims** (the Kolmogorov-optimal code length is
> uncomputable). Novelty is the *unified analysis framework*, not beating
> production memory systems — see `docs/EXPERIMENT_PROTOCOL.md`.
>
> Every executable condition is an **implemented analogue** under this harness, **not
> a reproduction** of any external system (MemGPT, Mem0, A-MEM, HippoRAG, AQLM,
> RETRO, Mamba, Titans, Infini-attention, LLM-KICK, or the hardware compression
> controller). External papers' reported numbers never appear in a measured local table.

## Two clearly-separated artifacts (lifecycle stages)

This repository contains **two** things at **different LLM-lifecycle stages**
(`docs/STAGE_TAXONOMY.md`), and they are deliberately kept apart:

| | Runnable PoC | Architecture prototype |
|---|---|---|
| Package | `ham.*` (runner/CLI) | `ham.architecture.*` |
| Lifecycle stage | **E** — inference-time external/persistent memory | **F** — architecture-level HAM layer |
| Base weights | **frozen**, never modified | frozen **or** trainable (router/fusion) |
| Integration | `external_context` (prompt assembly) | `hidden_state_fusion` (cross-attn / gated-residual) |
| Evaluated on benchmarks? | **yes** — the fair, runnable comparison | **no** — unit/toy-tested only |
| Status | publication PoC | skeleton + minimal toy integration |

The architecture prototype does **not** implement generic hidden-state injection
into arbitrary Hugging Face models; it attaches HAM to *self-contained toy blocks*
whose shapes and gradient behavior are verified. Torch is optional and, if missing,
touching a component **fails loudly** with install guidance.

```bash
python -m ham.cli arch-demo --block transformer --fusion cross_attention  # needs [hf]
```

## Why this design

- The frozen model is never modified, so memory is the **sole independent variable**.
- Compression is **actually applied** (real zstd/zlib bytes, real int8/int4 packed
  nibbles, optional FAISS PQ), and byte accounting reads `os.path.getsize` of the
  store that was written to disk.
- Every condition shares identical model, prompts, generation params, examples,
  seeds, embedder, and evaluator.

See the grounding evidence and source URLs in the accompanying evidence report
(Shannon 1948; Tishby–Pereira–Bialek IB 2000; MDL; MemGPT; HippoRAG; A-MEM; Mem0;
LLMLingua/-2; H2O; KIVI; DMC; MiKV; EvicPress; Product Quantization; Matryoshka;
LongMemEval; LoCoMo; RULER; HELMET).

## Install

```bash
# Core: mock backend + synthetic benchmark + compression + stats + report.
# Runs on numpy/scipy/pandas/pyyaml alone.
pip install -e .

# Add the stdlib-beating text codec + test runner (what CI uses):
pip install -e ".[zstd,dev]"

# Full research stack (real HF model, FAISS, datasets, plotting, energy):
pip install -e ".[hf,faiss,zstd,datasets,plot,instr,dev]"
```

Optional extras: `hf`, `faiss`, `zstd`, `quant` (bitsandbytes), `datasets`,
`plot`, `instr` (psutil + codecarbon), `dev`, `all`.

## Quickstart (60-second smoke test)

```bash
make test            # unit + smoke-pipeline + architecture + fair-control tests
make smoke           # tiny mock-backend run -> results/smoke
make smoke-figures   # watermarked tables + figures -> results/smoke/paper_artifacts
```

Or directly:

```bash
python -m ham.cli run    --config configs/smoke.yaml --out results/smoke
python -m ham.cli report --run-dir results/smoke --out results/smoke/paper_artifacts
python -m ham.cli info   # environment + package versions
```

## Code snippet — memory OFF vs HAM memory

```python
import numpy as np
from ham.config import MemoryConfig
from ham.conditions import build_condition
from ham.compression.serialize import ByteAccounting  # noqa: F401 (type ref)
from ham.embeddings import build_embedder, EmbeddingConfig
from ham.memory import HAMemory

embedder = build_embedder(EmbeddingConfig(kind="hash", dim=256))
history = [
    "The capital of Aurora is Verona.",
    "The mascot of Basalt is an otter.",
    "The founding year of Cobalt is 1902.",
]
question = "What is the capital of Aurora?"

# --- Memory OFF: no context is ever built ---
off = HAMemory(MemoryConfig(), build_condition("memory_off", _comp()), embedder)
for t in history:
    off.ingest_turn(t, session_id=0)
ctx_off, _ = off.build_context(question)
assert ctx_off == ""            # the frozen LLM sees only the question

# --- HAM memory: tiered store, consolidation, utility allocation, VQ + codec ---
ham = HAMemory(MemoryConfig(), build_condition("ham_memory", _comp()), embedder)
for t in history:
    ham.ingest_turn(t, session_id=0)
ctx_ham, diag = ham.build_context(question)
print(ctx_ham)                  # retrieved, budget-bounded context
acc = ham.serialize("/tmp/ham_store")   # writes real files, measures real bytes
print("physical bytes:", acc.physical_bytes, "compression ratio:", acc.compression_ratio)
```

```python
def _comp():
    from ham.config import CompressionConfig
    return CompressionConfig(text_codec="auto", vector_quant="int8", allocation="ham")
```

Run the *full* comparison (all conditions, metrics, statistics) with the CLI:

```bash
python -m ham.cli run --config configs/smoke.yaml --out results/smoke \
    --conditions memory_off,ham_memory
```

## Running LongMemEval and LoCoMo

```bash
# LongMemEval (multi-session long-term memory). Cheapest split: oracle.
# 1) Get data: download longmemeval_oracle.json from
#    https://github.com/xiaowu0162/LongMemEval  (or HF mirror
#    xiaowu0162/longmemeval-cleaned) into data/, or install the [datasets] extra.
# 2) Point configs/longmemeval.yaml -> dataset.path, then:
python -m ham.cli run    --config configs/longmemeval.yaml --out results/lme
python -m ham.cli report --run-dir results/lme --out results/lme/paper_artifacts

# LoCoMo (very long-term conversational memory, 10 conversations).
```

If a dataset is gated/offline/missing, the adapter **fails loudly** with
download guidance and **fabricates nothing**.

## Real-model proof of concept (small, CPU-feasible)

A one-command PoC runs a **genuinely downloaded** small instruct model
(`HuggingFaceTB/SmolLM2-135M-Instruct`) on the deterministic synthetic benchmark
over a tiny example/condition set. Because the backend is real (not `mock`), the
outputs are **not** watermarked `SMOKE TEST`; they are labeled **`REAL MODEL,
PROOF OF CONCEPT`** (in `RUN_LABEL.txt`, the report banner, and via the manifest).
This demonstrates the harness on a real tokenizer/model — it is **not** publication
evidence.

```bash
python -m ham.cli run    --config configs/poc_real_smollm.yaml --out results/poc_real_smollm
python -m ham.cli report --run-dir results/poc_real_smollm --out results/poc_real_smollm/artifacts
```

## Stage-F architecture memory-block experiment (toy model)

A mechanistic toy-architecture proof of the **compression** thesis: identical
toy LMs differing ONLY in their memory-block policy (standard `FlatMemory` vs
HAM-compressed `HamMemory` with int4 prototypes + frequency-driven
consolidation), measured at iso-quality on **byte-honest memory size** and
**inference latency**, with a **redundancy lever** and ablations
(`ham_uniform`/`ham_no_consolidation`/`ham_random_alloc`) that isolate the
frequency mechanism. See `docs/ARCHBENCH_PROTOCOL.md`.

The archbench runner also emits a **fine-tuning post-hoc** analysis on the same
toy models: `standard_memory` vs `ham_memory` cost-to-target (optimizer steps /
supervised tokens to reach `max(standard_quality) − δ`) plus the L2
weight-drift (`sqrt(sum((p − p_init)²))`) overhead at that target. The toy is
trained from scratch, so there is no zero-shot forgetting arm — the diagnostic
is the drift overhead HAM's extra router/fusion/encoding parameters add to
reach the same quality.

```bash
make archbench-smoke   # deterministic mock trainer (watermarked SMOKE TEST)
make archbench-toy     # real torch toy model (needs [hf])
ham archbench        --config configs/archbench_toy.yaml --out results/archbench_toy
ham archbench-report --run-dir results/archbench_toy     --out results/archbench_toy/artifacts
```

Headline: HAM/standard bytes- and latency-ratios vs redundancy (the slope proves
frequency is the mechanism), plus the ablation table, plus the standard-vs-HAM
fine-tuning post-hoc (`table_finetune_posthoc.md`). Honest scope: a controlled
toy proof-of-mechanism (small toy LM, synthetic corpora), labeled stage-F PoC.

## Stage-D KV-cache compression experiment (real frozen model)

Compresses a frozen model's KV cache under each policy and measures byte-honest
KV size, decode latency, and next-token quality (agreement vs `full_kv`), across
the redundancy lever; ablations (`uniform_quant_kv`/`h2o_kv`/`random_evict_kv`/
`ham_no_cluster`) isolate precision vs frequency-dedup. See
`docs/KVBENCH_PROTOCOL.md`.

```bash
make kvbench-smoke   # deterministic mock (watermarked SMOKE TEST)
make kvbench-real    # frozen SmolLM2-135M (needs [hf])
ham kvbench        --config configs/kvbench_smollm.yaml --out results/kvbench_smollm
ham kvbench-report --run-dir results/kvbench_smollm     --out results/kvbench_smollm/artifacts
```

Honest scope: HAM dominates the quality-vs-bytes Pareto at high redundancy on the
real model (frequency-weighted position retention + int4), beating random/norm
selection at every keep-ratio; the ablation isolates frequency as the mechanism.
Stage-D PoC on one small frozen model — see `docs/KVBENCH_PROTOCOL.md`.

## Regenerate tables/figures

```bash
python -m ham.cli report --run-dir results/<run> --out results/<run>/paper_artifacts
# alias: `ham export` does the same (paper-artifacts export command).
```

Artifacts produced (only from real run files):
- `table_main.md` / `table_main.csv` — Table T1 (score, tokens, physical bytes,
  bytes/fact, compression ratio, latency, throughput, index size).
- `table_baselines.md` / `.csv` — Table T2, **executable baselines only**, with
  integration mode, base-weights-changed, persistence, consolidation, adaptive
  precision, task score, retrieval recall@k / MRR, prompt tokens, physical bytes,
  compression ratio, latency, peak RSS. Never mixes external reported numbers.
- `table_deltas.md` — Table T3, per-condition deltas vs `memory_off` and
  `uncompressed_rag`.
- `table_stats.md` — paired bootstrap Δ, CIs, permutation p-values, non-inferiority.
- `fig_pareto_quality_bytes.png` — F1 quality-vs-bytes Pareto.
- `fig_task_score.png`, `fig_prompt_tokens.png`, `fig_physical_bytes.png` (F3),
  `fig_latency.png` (F2), `fig_tier_occupancy.png`.

With no data, tables are written as **EMPTY TEMPLATES** and no figures are created.

## Conditions & ablations

Headline **executable baselines** (the fair, runnable comparison; all share one
frozen model / dataset / prompt / decoding / tokenizer / seed / evaluator):
`memory_off`, `full_history`, `uncompressed_rag`, `recency_fifo`,
`static_prototype`, `uniform_quantization`, `ham_memory`.

Additional **ablations**: `random_tiering`, `no_consolidation`, `no_recency`,
`no_novelty`, `no_reuse`, `lexical_retrieval`. (`uncompressed_retrieval` is a
back-compat alias of `uncompressed_rag`.)

The machine-readable mapping of each condition to its purpose, target stage,
closest literature category, and `implemented analogue, not reproduction` label is
in `docs/BASELINE_CROSSWALK.csv` / `.json`. See `docs/EXPERIMENT_PROTOCOL.md`.

## Repository layout

```
src/ham/
  backends/      mock (deterministic) + hf (real causal LM)
  compression/   text_codec (zstd/zlib), vector_quant (int8/int4/PQ), serialize (real bytes)
  memory/        store, importance, consolidation, retrieval, ham orchestrator
  architecture/  OPTIONAL stage-F prototype: layer (router/fusion/tiers/HAMBlock) + toy demo
  archbench/     OPTIONAL stage-F toy-architecture memory-block compression experiment (FlatMemory/HamMemory, mock/torch trainers, own runner + report + fine-tuning post-hoc)
  kvbench/        OPTIONAL stage-D KV-cache compression experiment on a frozen HF model (mock/torch trainers, own runner + report)
  datasets/      synthetic (local, with gold-memory ids), longmemeval adapter
  metrics.py stats.py instrumentation.py manifest.py runner.py report.py cli.py config.py
configs/         smoke, synthetic, longmemeval, publication_7b, poc_real_smollm, archbench_smoke, archbench_toy, kvbench_smoke, kvbench_smollm
docs/            STAGE_TAXONOMY, EXPERIMENT_PROTOCOL, METRICS_SCHEMA, REPRODUCIBILITY,
                 METHODOLOGY_APPENDIX, ARCHITECTURE, BASELINE_CROSSWALK (.csv/.json)
tests/           quantization, serialize, importance/tiering, consolidation,
                 retrieval, metrics/stats, smoke pipeline, architecture,
                 stage metadata, fair controls, retrieval metrics, baselines report
```

## Documentation
- `docs/STAGE_TAXONOMY.md` — LLM lifecycle stages A–F; where the PoC (E) and the
  proposed architecture (F) sit; the fields recorded in every config/manifest.
- `docs/EXPERIMENT_PROTOCOL.md` — hypotheses, controls, failure criteria, novelty boundary.
- `docs/METRICS_SCHEMA.md` — every logged field (incl. retrieval recall@k / MRR).
- `docs/REPRODUCIBILITY.md` — seeds, manifests, dataset access, determinism caveats.
- `docs/ARCHITECTURE.md` — the optional stage-F HAM layer (router/fusion/tiers/modes).
- `docs/ARCHBENCH_PROTOCOL.md` — the optional stage-F toy-architecture memory-block
  compression experiment (hypotheses AB1–AB4, redundancy lever, ablations, current
  limitation + next step). The fine-tuning post-hoc (standard vs HAM memory block
  cost-to-target + drift on the toy models) is part of this experiment.
- `docs/KVBENCH_PROTOCOL.md` — the optional stage-D KV-cache compression experiment
  on a frozen model (hypotheses KV1–KV4, redundancy lever, ablations, honest
  precision-vs-frequency findings).
- `docs/METHODOLOGY_APPENDIX.md` — generic, venue-agnostic method description (no
  repo paths / CLI), suitable for an online appendix.
- `docs/BASELINE_CROSSWALK.csv` / `.json` — condition → purpose / stage / literature.

## License
Apache-2.0. See `LICENSE` and `CITATION.cff`.
