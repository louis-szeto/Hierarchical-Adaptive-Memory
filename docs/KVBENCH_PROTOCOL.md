# KV-Cache Compression Experiment Protocol (stage D)

Pre-registers the design, hypotheses, and fairness controls for compressing a
**frozen** model's KV cache. Stage D; walled off from stage E/C/F. See
``docs/ARCHBENCH_PROTOCOL.md`` for the toy-architecture companion (stage F).

## Research question

Does frequency-aware compression of a real model's KV cache reduce its byte size
and decode latency at matched quality, with the win scaling with context
redundancy? Is *frequency* the active mechanism?

## Design (single independent variable)

A frozen model (default SmolLM2-135M) prefills a long context; the resulting KV
cache (`past_key_values`, legacy per-layer `(K,V)` of shape
`(batch, num_kv_heads, seq, head_dim)`) is compressed per condition; the
compressed cache is rebuilt (`DynamicCache.from_legacy_cache`) and a short
continuation forward measures size, latency, and quality. The compression policy
is the sole variable; context redundancy is the lever. One position structure
(clustering/eviction, derived from layer-0 K) is applied **uniformly across all
layers** so the rebuilt cache is consistent.

### Conditions
- `full_kv` — no compression (reference).
- `ham_kv` — cluster redundant positions by K similarity, keep ONE representative
  per cluster (dedup), + int4. Treatment.
- `uniform_quant_kv` — int4 all positions, no clustering (isolates precision).
- `h2o_kv` — evict low-norm positions, float32 (eviction baseline).
- `random_evict_kv` — evict random positions, float32 (isolates frequency).
- `ham_no_cluster` — int4 + random position retention (isolates frequency clustering).

### Metrics (byte-honest)
- **KV bytes** — physical bytes (positions × kv-heads × head-dim × precision;
  int4 via packed-nibble sizes from `compression.vector_quant`).
- **decode latency** — continuation forward time with the compressed cache.
- **quality** — next-token top-1 agreement vs `full_kv` (+ accuracy vs ground truth).

### Compression-strength sweep (iso-quality Pareto)
Each position-reducing condition is evaluated at every entry of `keep_ratios`
(the fraction of positions/representatives retained); `full_kv` and
`uniform_quant_kv` keep all positions (one point). This yields a per-condition
**quality-vs-bytes Pareto** curve. `ham_kv` selects the representatives of the
*most-frequent* clusters to fill the budget (frequency-driven selection), so at
high redundancy its frontier should dominate: high quality at the smallest bytes.

### Redundancy lever
Contexts are motifs sampled at a controllable frequency skew (0 = diverse, →1 =
highly repetitive). The proof: HAM's advantage grows with redundancy.

## Hypotheses
- **KV1 (size).** `ham_kv` uses fewer KV bytes than `full_kv` (ratio < 1).
- **KV2 (latency).** `ham_kv` decodes faster (fewer positions).
- **KV3 (mechanism).** Frequency-agnostic conditions (`uniform_quant`,
  `random_evict`, `ham_no_cluster`) do NOT scale with redundancy; `ham_kv` does.
- **KV4 (redundancy slope).** `ham_kv`'s byte ratio drops and its quality rises
  with redundancy (compression is cheap when info is redundant).

## Commands
```bash
make kvbench-smoke   # mock trainer, deterministic, watermarked SMOKE TEST
make kvbench-real    # frozen SmolLM2-135M (needs pip install -e ".[hf]")
ham kvbench        --config configs/kvbench_smollm.yaml --out results/kvbench_smollm
ham kvbench-report --run-dir results/kvbench_smollm     --out results/kvbench_smollm/artifacts
```
Outputs: `manifest.json` (stage D), `results.jsonl`, `aggregate`, `stats`,
`summary`. Report: `table_redundancy` (headline ratios vs redundancy),
`table_quality_bytes` (Pareto), `fig_advantage_vs_redundancy`.

## What the real run shows (honest)
On SmolLM2-135M (256–512-token contexts, keep-ratio sweep), with `ham_kv` using
**frequency-weighted real-position retention** (keep N real positions, filling the
budget from the most-frequent clusters first, + int4):
- **HAM dominates the Pareto at high redundancy.** At r=0.9, `ham_kv` beats
  `random_evict_kv` at *every* keep-ratio on BOTH bytes and quality — e.g. kr=0.25:
  0.039 bytes-ratio / 0.50 agreement vs random's 0.250 / 0.31; kr=0.5: 0.078 / 0.71
  vs 0.500 / 0.64. Smaller bytes AND higher quality.
- **Frequency is the mechanism (ablation holds).** `ham_kv` (frequency selection)
  beats `ham_no_cluster` (random selection at the SAME int4 bytes) at kr=0.25
  (0.50 vs 0.41) and kr=0.5 (0.71 vs 0.64) — the win is from frequency-aware
  selection, not just int4 precision.
- **Redundancy lever holds (KV4).** At low redundancy (r=0) `ham_kv` ≈
  `ham_no_cluster` (no frequency structure to exploit); the advantage opens up as
  redundancy rises.
- **Honest scope:** stage-D PoC on one small frozen model; the defended claim is
  the mechanism + Pareto dominance scaling with redundancy, not superiority over
  production KV-compression (H2O/StreamingLLM/etc.). An earlier `ham_kv` variant
  that *merged/averaged* positions corrupted real-model attention (quality ~0) and
  is documented as the wrong mechanism; keeping real positions selected by
  frequency is the version that works.

## Honesty rules (inherited)
Fabricate nothing; mock output watermarked `SMOKE TEST`; fail loudly (unknown
config, torch missing, backend mismatch); byte-honest accounting; stage-D PoC on
one small frozen model — the defended claim is the mechanism + redundancy slope,
not superiority over production KV-compression (H2O/StreamingLLM/etc.).
