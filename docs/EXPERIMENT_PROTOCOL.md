# Experiment Protocol

This document pre-registers the hypotheses, variables, fairness controls, and
failure criteria for the HAM (Hierarchical Adaptive / Utility-Rate Adaptive)
memory study. It follows Field 6 of the evidence report.

## Design principle

The frozen LLM is **never modified**. Memory is the *sole independent variable*:
every condition shares the identical model + weights, tokenizer, prompt template,
generation parameters, examples, seeds, embedding model, retrieval `k` (where
applicable), evaluator, and hardware. This isolates the memory system as the
cause of any difference and makes "identical frozen LLM with memory disabled" a
fair control.

Contrast with methods that *train* the model (DMC, Gisting, ICAE): those change
the model, so they cannot attribute effects to memory alone.

## Conditions

All conditions are **inference-time, frozen-weight, external-context** (lifecycle
stage E; see `docs/STAGE_TAXONOMY.md`). Each is an **implemented analogue** under
this harness, **not** a reproduction of any external system; the machine-readable
mapping to purpose / stage / literature category is in `docs/BASELINE_CROSSWALK.csv`.

**Headline executable baselines** (fair, runnable comparison):

| Condition | Description |
|---|---|
| `memory_off` | Frozen LLM, question only, no persistent memory (control **A**). |
| `full_history` | Entire concatenated history in-context, stored uncompressed. |
| `uncompressed_rag` | RAG-style exact retrieval over full-text chunks + float32 index (baseline **B**; alias `uncompressed_retrieval`). |
| `recency_fifo` | Recency/FIFO eviction: evict oldest regardless of utility (forgetting analogue). |
| `static_prototype` | Static pre-computed prototypes; no adaptive promotion/consolidation over time. |
| `uniform_quantization` | HAM store but every item gets the same bits (isolates *variable* allocation). |
| `ham_memory` | HAM: working/episodic/semantic tiers + online consolidation + utility-driven bit allocation + vector quantization + text codec (**C**). |

**Additional ablations:**

| Condition | Description |
|---|---|
| `random_tiering` | HAM but tiers assigned at random (isolates *utility-driven* tiering). |
| `no_consolidation` | HAM without episodic→semantic consolidation. |
| `no_recency` | HAM importance without the recency signal. |
| `no_novelty` | HAM importance without the novelty signal. |
| `no_reuse` | HAM importance without the reuse signal. |
| `lexical_retrieval` | HAM with lexical-only (BM25-lite) retrieval instead of dense. |

## Falsifiable hypotheses

- **H1 (quality preservation).** `ham_memory` task score ≥ (`uncompressed_retrieval` − δ)
  for a pre-registered non-inferiority margin δ (default 0.03), and > `memory_off`.
  *Falsified if* C < A, or C < B − δ with statistical significance.
- **H2 (token economy).** C uses fewer prompt/input tokens than B at equal task score.
- **H3 (byte economy).** C's **physical serialized bytes** < B's.
- **H4 (latency).** C's total latency < B's at equal quality.
- **H5 (utility matters).** C > `uniform_quantization` / `random_tiering` at equal
  average rate. This is the strongest novelty test: if utility-driven allocation
  is no better than arbitrary allocation, the hierarchy/utility contribution is
  unsupported and only generic compression matters.

## Variables

- **Independent:** memory condition; compression ratio / bit budget; model size
  {1B, 7–8B}; benchmark {synthetic, LongMemEval, LoCoMo}; retrieved `k`.
- **Dependent:** every metric in `docs/METRICS_SCHEMA.md`.
- **Held constant (fairness controls):** frozen weights, decoding params,
  embedding model (across B and C), retrieval `k`, hardware, seeds, evaluator.
- **Fair-control fingerprint.** The runner writes a `fair_control` block into the
  manifest recording the shared invariants (model id, backend, dataset, seed,
  prompt templates, decoding params, embedder, token budget, retrieval `k`,
  evaluator) plus their SHA-256 fingerprint. `tests/test_fair_controls.py` asserts
  these are identical across conditions and that every example is evaluated under
  every condition, so the memory policy is provably the sole independent variable.

## Statistics

- Per-question **paired** comparisons (shared examples).
- Paired **bootstrap** 95% CIs on the mean difference (≥10,000 resamples for
  publication runs).
- Paired **permutation** test on the mean difference; **McNemar** on binary
  correctness.
- **Non-inferiority** test for H1 with pre-registered δ.
- Report effect sizes and CIs, not just p-values. Multiple-comparison awareness
  (Holm) when comparing across benchmarks.

## Failure criteria (declared before running)

- C's task score below A → memory system is net-harmful.
- H5 fails (C ≈ random/uniform ablation) → hierarchy/utility contribution unsupported.
- Physical bytes not reduced vs B → byte-economy claim withdrawn.
- Latency higher than B at equal quality → latency claim withdrawn.

## Rigor / honesty rules

- **Byte-honesty:** headline **physical serialized bytes** (`os.path.getsize` of the
  actually-written store). Logical (uncompressed float32) bytes are only an
  upper-bound sanity check.
- **No Shannon-optimality claims.** The Kolmogorov-optimal code length is
  uncomputable; we claim only practical compression vs specific codecs.
- **No fabricated results.** Report generators transform only real run files; with
  no data they emit clearly-labeled EMPTY TEMPLATES.
- **SMOKE watermark.** Any figure/table built from mock-backend data is watermarked
  `SMOKE TEST` and is never presented as a scientific result.
- **Fail loudly.** Unknown config keys, missing datasets, and unavailable
  quantization raise errors rather than silently changing the condition.

## Novelty boundary

Each ingredient (tiered memory, biologically-inspired memory, consolidation,
adaptive compression, prompt/vector compression) exists in prior art (MemGPT,
HippoRAG, A-MEM, Mem0, DMC, MiKV, EvicPress, LLMLingua, PQ, Matryoshka — see the
evidence report). The defended contribution is the **unified rate-distortion / IB /
MDL treatment of a biologically-tiered, frozen-model memory with byte-honest
reporting** against an identical no-memory control and random/uniform ablations.
Superiority over production systems is **not** assumed; the claimed advance is the
analysis framework and the token/byte/latency-vs-quality trade-off characterization.
