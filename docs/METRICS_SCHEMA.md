# Metrics Schema

Per-example rows are written to `per_example.jsonl`; per-condition means/stds to
`aggregate.json` / `aggregate.csv`. Fields marked *nullable* are `null` with an
accompanying `*_reason` when the measurement is unavailable — never faked.

## Identity
| Field | Meaning |
|---|---|
| `example_id` | Stable id of the example. |
| `condition` | Condition/ablation name. |
| `question_type` | Dataset-provided category. |
| `dataset`, `model_id`, `backend` | Provenance. |
| `is_smoke` | `true` for mock-backend runs (watermarked, non-scientific). |

## Quality
| Field | Meaning |
|---|---|
| `task_score` | Primary score = max(exact_match, contains_gold) ∈ {0,1}. |
| `exact_match` | Normalized exact match. |
| `f1` | SQuAD-style token F1. |
| `contains_gold` | Normalized gold-substring containment. |
| `prediction`, `gold` | Raw strings (for audit). |

## Tokens (exact tokenizer)
`prompt_tokens`, `input_tokens`, `output_tokens`, `total_tokens` — counted with the
backend's *actual* tokenizer (mock uses a documented regex proxy).

## Bytes (physical vs logical)
| Field | Meaning |
|---|---|
| `logical_memory_bytes` | Uncompressed float32 + raw-text size (upper-bound *estimate*). |
| `physical_serialized_bytes` | **Headline.** Real on-disk bytes actually written. |
| `physical_text_bytes` / `physical_vector_bytes` / `physical_meta_bytes` | Components. |
| `bytes_per_fact` | `physical_serialized_bytes / n_atomic_facts`. |
| `compression_ratio` | `logical_bytes / physical_bytes`. |
| `text_codec` | Codec actually used (`zstd`/`zlib`/`raw`). |
| `vector_quant` | Vector codec (`none`/`int8`/`int4`/`pq`). |
| `index_size_bytes` | Total serialized store directory size. |
| `n_retained_items` | Items physically stored. |
| `mean_quantization_error` | Per-item vector reconstruction error (paper Eq 8): mean-abs `‖x − x̂‖` over the elements of each stored embedding after quantization, averaged over the records in this example's store. `null` when the store has no vectors (e.g. `memory_off`, `full_history`) and `0.0` when vectors are stored without quantization (`vector_quant='none'`). Additive diagnostic; not read into the bytes or quality computation. |

## Latency / throughput
`retrieval_latency_s`, `context_build_latency_s`, `prefill_latency_s`,
`decode_latency_s`, `total_latency_s`, `tokens_per_second`, `n_retrieved`.
`latency_source` = `measured` (HF) or `simulated` (mock, deterministic function of
token counts — never interpret as hardware timing).

## Retrieval quality (nullable + reason)
| Field | Meaning |
|---|---|
| `retrieval_recall_at_k` | 1.0 if any of the top-k retrieved chunks is the gold memory, else 0.0. `null` when the dataset carries no gold-memory identity. |
| `retrieval_mrr` | Reciprocal rank of the first gold hit in the retrieved order; 0.0 on miss; `null` when no gold ids. |
| `retrieval_metrics_reason` | `null` when computed; `"no_gold_memory_ids"` when the dataset provides no gold identity. |

Gold identity is available for the deterministic `synthetic` dataset via
`Example.gold_memory_texts` (the sentence stating the final answer value, or the
update sentence for knowledge-update questions).

## Condition / lifecycle metadata (constant per condition)
| Field | Meaning |
|---|---|
| `integration_mode` | `external_context` (PoC) or `hidden_state_fusion` (architecture). |
| `base_weights_changed` | Whether the base model's weights were modified (always `false` in the PoC). |
| `persistent` | Persistent external store across sessions. |
| `consolidation` / `consolidation_mode` | Whether episodic→semantic consolidation runs; `adaptive` vs `static`. |
| `adaptive_precision` | Utility-driven per-item bit allocation active (HAM), not uniform. |
| `tiering`, `allocation`, `eviction` | Tier-assignment policy, bit-allocation policy, forgetting policy. |
| `literature_analogue` | Disclosure string: implemented analogue under this harness, **not** a reproduction. |

## Tiers
`tier_working`, `tier_episodic`, `tier_semantic`, `n_records`, `n_prototypes`.

## Resources (nullable + reason)
| Field | Meaning |
|---|---|
| `peak_cpu_rss_bytes` (+ `peak_cpu_rss_reason`) | Peak RSS via stdlib `resource`. |
| `peak_cuda_allocated_bytes` / `peak_cuda_reserved_bytes` (+ `cuda_reason`) | Via `torch.cuda` when a CUDA backend is active. |
| `energy_joules` (+ `energy_co2_kg`, `energy_reason`) | Via CodeCarbon when installed; else `null` + reason. |

## Diagnostics available on demand
- Text codec: order-0 Shannon entropy (bits/byte), entropy-floor bytes, achieved
  bits/byte (`ham.compression.text_codec.empirical_code_length_bits`). These are
  *diagnostics*, not optimality claims.
- Vector quantization: max/mean absolute reconstruction error and per-row uniform
  bound (`ham.compression.vector_quant.roundtrip_error`). The per-item mean-abs
  error is also recorded on each `MemoryRecord.quantization_error` at
  serialization time (paper Eq 8) and surfaced per condition as
  `mean_quantization_error` in the aggregate.
