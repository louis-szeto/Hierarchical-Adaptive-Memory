# Architecture-level HAM layer (stage F prototype)

`ham.architecture` is an **optional** PyTorch prototype of the *proposed* full HAM
architecture (research addendum §6). It is separate from the runnable
`external_context` PoC and is exercised only by unit tests and a minimal toy
integration. **It is not evaluated on publication benchmarks**, and it does **not**
implement generic hidden-state injection into arbitrary Hugging Face models — it
attaches HAM to *self-contained toy blocks* whose shapes and gradient behavior we
can verify.

Torch is an optional dependency (`pip install -e ".[hf]"`). Importing
`ham.architecture` is always safe; touching a torch-backed component when torch is
absent raises a `RuntimeError` with install guidance (never a silent no-op).

## Components (`ham/architecture/layer.py`)

- **`TierState`** — working / episodic / semantic tiers as tensors + a provenance
  and event log. `read_kv()` returns the retrievable `(M, D)` matrix (episodic +
  semantic); `write()` is the post-block hook; `consolidate()` folds episodic items
  into semantic prototypes by leader clustering (running-mean update).
- **`MemoryRouter`** — learned query/key projections score the mean-pooled hidden
  state against memory and select top-k. The top-k selection is non-differentiable
  (cf. Memorizing Transformers' non-differentiable memory); the soft weights provide
  a differentiable path into the router.
- **`CrossAttentionFusion`** — `h' = h + CrossAttn(Q=h, K=V=serialize(m))`, with the
  retrieved keys/values scaled by the router weights (RETRO / Memorizing-Transformers
  mechanism *family* — an implemented analogue, not a reproduction).
- **`GatedResidualFusion`** — `h' = h + g ⊙ (W_m·m)`, `g = σ(W_g[h; m])`. The natural
  interface for a Mamba-style block, which has no attention sub-layer to attach
  cross-attention to.
- **`ToyTransformerBlock` / `ToyRecurrentBlock`** — minimal self-attention+MLP and
  GRU-based stand-ins. **Not** pretrained models and **not** the Mamba architecture;
  shape/gradient stand-ins only.
- **`AsyncConsolidationInterface`** — off-critical-path consolidation: `schedule()`
  marks work, `run_pending()` performs Tier1→Tier2 consolidation between turns.
- **`HAMBlock`** — wraps a base block with read → fuse → base-compute → write.
  - `set_mode("frozen")`: whole forward runs under `no_grad`; the memory read is
    detached (stop-gradient into the base model); no parameter receives gradients.
  - `set_mode("trainable", train_base=False)`: router/fusion parameters are
    trainable; the base block stays frozen unless `train_base=True`.

## Toy integration (`ham/architecture/toy.py`, `ham arch-demo`)

`run_toy_demo()` builds a `HAMBlock` over a toy block and returns a
JSON-serializable evidence dict demonstrating:

1. **shapes** — `(B, T, D)` preserved end to end in both modes;
2. **grad / no-grad** — frozen mode yields a detached output with no parameter
   gradients; trainable mode routes gradients to the router and fusion while the
   frozen base receives none;
3. **memory lifecycle** — writes land in working+episodic; the async interface
   consolidates episodic→semantic between turns; a later read sees the prototypes.

```bash
python -m ham.cli arch-demo --block transformer --fusion cross_attention
python -m ham.cli arch-demo --block recurrent   --fusion gated_residual
```

The same invariants are asserted in `tests/test_architecture.py` (skipped, not
failed, when torch is unavailable).
