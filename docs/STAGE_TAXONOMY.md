# LLM Lifecycle Stage Taxonomy

Memory and compression can be introduced at very different points in an LLM's
lifecycle. Conflating them makes comparisons unfair. This harness records, for
every run, *where in the lifecycle it sits*, so that a reader can see exactly what
was and was not changed.

## Stages

| Stage | Where | Example techniques (cited, not reproduced here) |
|---|---|---|
| A | Pretraining | training a model from scratch |
| B | Training-time memory | RETRO-style retrieval pretraining |
| C | Fine-tuning | adapters / instruction tuning that edit weights |
| D | Inference-time model / KV compression | AQLM, KV-cache eviction (H2O/KIVI), activation compression, LLM-KICK |
| E | Inference-time external / persistent memory | MemGPT, Mem0, A-MEM, HippoRAG, RAG |
| F | Architecture-level recurrent / stateful memory | Mamba, Titans, Infini-attention |

## Where this repository sits

- **Runnable PoC = stage E** (with a stage-D-*flavored* variable-precision
  serialization aspect: the external store is quantized per item). The base model
  is **frozen**; memory is the sole independent variable. Integration is
  `external_context` — memory is turned into prompt text.
- **Proposed full HAM = stage F** — an architecture-level read/fusion/write layer
  attached to a Transformer or Mamba-style block, frozen **or** jointly trainable.
  In this repo that is a **prototype** (`ham.architecture`), exercised by unit tests
  and a toy integration only; it is **not** evaluated on publication benchmarks and
  does **not** implement generic hidden-state injection into arbitrary HF models.

## Fields recorded in every config and run manifest

| Field | Meaning |
|---|---|
| `target_stage` | One of `A_pretraining`, `B_training_memory`, `C_finetuning`, `D_inference_kv_compression`, `E_inference_external_memory`, `F_architecture_level`. |
| `base_weights_changed` | Whether the base model's parameters were modified. |
| `persistent_across_sessions` | Whether memory persists across sessions. |
| `integration_mode` | `external_context` or `hidden_state_fusion`. |
| `trainable_router` | Whether the memory router/fusion is trainable. |

These are set in the `stage:` block of each YAML config and copied verbatim into
`manifest.json`, alongside a `fair_control` fingerprint (see
`docs/EXPERIMENT_PROTOCOL.md`) that pins the invariants shared by all conditions in
a run.

## Honesty rule

No condition or document in this repository claims to reproduce the hardware
Compression-Aware Memory Controller, LLM-KICK, AQLM, RETRO, Mamba, Titans,
Infini-attention, MemGPT, Mem0, A-MEM, or HippoRAG. Where a condition is
*inspired by* a category, it is labeled **implemented analogue, not reproduction**
(see `docs/BASELINE_CROSSWALK.csv`).
