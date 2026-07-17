# Architecture Memory-Block Experiment Protocol (stage F)

This pre-registers the design, hypotheses, and fairness controls for the stage-F
**toy-architecture memory-block compression** experiment. It is the mechanistic
proof of the compression thesis (frequency-aware compression of the model's OWN
memory shrinks it and speeds inference at iso-quality), kept separate from the
stage-E/C frozen-LLM studies.

## Research question

Does frequency-aware compression of the model's memory block reduce its size and
inference latency at matched quality? Is *frequency* (not generic compression)
the active mechanism?

## Design (single independent variable)

Identical toy language models (embedding → `ToyTransformerBlock`s → LM head) are
trained under different **memory-block policies**. The memory policy is the sole
variable; everything else (architecture, data, optimizer, seeds, evaluator) is
identical and fingerprinted (`fair_control`). The memory adapter does
HAM read → fuse → base → write, reusing `MemoryRouter` + `CrossAttentionFusion`
from `ham.architecture.layer`, driving an external `MemoryStore`.

### Conditions
- `no_memory` — plain toy LM (floor).
- `standard_memory` — `FlatMemory`: append-only float32, FIFO (reference).
- `ham_memory` — `HamMemory`: episodic (float32) + semantic (int4 prototypes of
  frequent items via leader-clustering consolidation + `vector_quant`).
- `ham_uniform` — consolidation but float32 prototypes (isolates int4 precision).
- `ham_no_consolidation` — episodic FIFO only (isolates item-reduction).
- `ham_random_alloc` — random item→prototype assignment (isolates frequency clustering).

### Byte-honesty
`MemoryStore.byte_size()` returns real physical bytes: float32 items at `D*4`;
int4 prototypes at their packed-nibble size + per-row scale/zero metadata
(`compression.vector_quant`). This is the headline size metric.

### Redundancy lever
Corpora are generated with a controllable item-frequency skew (`redundancy` in
[0,1]; 0 = uniform, →1 = Zipf). The intended proof: HAM's byte/latency advantage
over `standard_memory` grows with redundancy. **See "Current limitation" below.**

## Hypotheses
- **AB1 (size).** At iso-quality, `ham_memory` uses fewer memory bytes than
  `standard_memory` (bytes_ratio < 1).
- **AB2 (latency).** At iso-quality, `ham_memory` has lower inference latency
  (fewer items to attend over).
- **AB3 (mechanism = frequency).** The advantage is driven by frequency-aware
  consolidation + precision: `ham_no_consolidation` shows no advantage;
  `ham_uniform` < `ham_memory`; `ham_random_alloc` is weaker than `ham_memory`.
- **AB4 (redundancy slope).** HAM's advantage grows with corpus redundancy.

Failure criteria: if `ham_memory` is not iso-quality with `standard_memory`, or
if the ablations match `ham_memory`, the frequency mechanism is unsupported.

## Tasks
- **Associative recall (primary):** streams of (key→value) pairs with skewed key
  frequency; quality = accuracy on value positions.
- **Next-token LM (secondary):** sequences of repeated motifs; quality = overall
  next-token accuracy.

## Commands
```bash
make archbench-smoke   # mock trainer, deterministic, watermarked SMOKE TEST
make archbench-toy     # real torch toy model (needs pip install -e ".[hf]")
ham archbench        --config configs/archbench_toy.yaml --out results/archbench_toy
ham archbench-report --run-dir results/archbench_toy     --out results/archbench_toy/artifacts
```

Outputs: `manifest.json` (stage F), `curve.jsonl`, `aggregate`, `stats`,
`summary`. Report: `table_redundancy` (the headline ratios-vs-redundancy),
`table_quality_bytes` (Pareto), `fig_advantage_vs_redundancy`, `fig_quality_vs_bytes`.

## Per-item memory (design) — all hypotheses hold on the real run

The trainer writes **every token's hidden state** to a FIFO window (per-item),
and `HamMemory` consolidates that window into per-key prototypes. The prototype
count = distinct keys in the window, which is exactly what the redundancy lever
controls. Confirmed on the real torch toy run (SmolLM2-free, dim-48 toy LM):

- **AB1 (size) holds strongly:** e.g. ~20–100× byte compression at iso-quality.
- **AB3 (mechanism) holds:** `ham_no_consolidation` shows no compression (= 1.0);
  `ham_uniform` (float32) > `ham_memory` (int4), isolating precision.
- **AB4 (redundancy slope) holds:** `ham_memory`'s bytes-ratio vs `standard_memory`
  drops as redundancy rises (e.g. 0.021 → 0.020 → 0.010 at r = 0.0/0.5/0.9 on the
  toy). The compression scales with information redundancy — the proof that
  *frequency* is the active mechanism.

Honest scope: this is a controlled toy proof-of-mechanism (small toy LM, synthetic
recall/LM corpora), labeled stage-F PoC. The defended claim is the mechanism +
the redundancy slope, not superiority over production transformers.

## Honesty rules (inherited)
Fabricate nothing; mock output watermarked `SMOKE TEST`; fail loudly (unknown
config keys, `trainer: torch` without torch); byte-honest accounting; labeled
stage-F toy PoC (not a claim about production transformers).
