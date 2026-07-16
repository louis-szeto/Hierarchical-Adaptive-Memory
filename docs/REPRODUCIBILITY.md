# Reproducibility

## What is pinned
- **Seeds** everywhere: dataset generation, embeddings (hash), bootstrap /
  permutation resampling, and generation (temperature 0 ⇒ greedy by default).
- **Deterministic synthetic data:** regenerated from a seed (RULER-style), no
  download, contamination-free.
- **Run manifest** (`manifest.json`) records: harness version, UTC timestamp,
  config hash, git commit (if a repo), Python/OS/CPU, installed package versions,
  dataset/model revisions, the full resolved config, the lifecycle **stage** fields
  (`target_stage`, `base_weights_changed`, `persistent_across_sessions`,
  `integration_mode`, `trainable_router`), and a **`fair_control`** block (shared
  invariants + SHA-256 fingerprint) proving the memory policy was the only variable.
- **Config hash:** SHA-256 of the sorted config JSON, embedded in the manifest.

## Determinism caveats
- The **mock backend** is fully deterministic across machines; its latencies are a
  deterministic function of token counts and are flagged `simulated`.
- The **HF backend** is deterministic under greedy decoding for a fixed model
  revision + hardware, but exact latencies and (rarely) low-order logits can vary
  across GPU architectures / library versions. Token counts and text outputs are
  stable given the same model revision.
- `sentence-transformers` embeddings depend on the model revision; pin it.

## Exact token accounting
Token counts use the backend's *actual* tokenizer (`apply_chat_template` when the
model provides a chat template). The mock backend uses a documented regex proxy.

## Datasets: gated / offline behavior
- **Synthetic:** always available, generated locally.
- **LongMemEval:** set `dataset.path` to a local `longmemeval_oracle.json` /
  `longmemeval_s.json`, or install the `[datasets]` extra to attempt an HF-hub
  download of `xiaowu0162/longmemeval-cleaned`. If neither works, the adapter
  raises with actionable guidance and **fabricates nothing**.
- **LoCoMo:** clone `snap-research/locomo` and point `dataset.path` at
  `data/locomo10.json`. Missing file ⇒ loud error.

## Large models are never auto-downloaded in tests
Tests and CI use only the mock backend + synthetic data. Real HF models are
instantiated **only** when a config explicitly selects `backend.kind: hf`, which
happens solely on an explicit `ham run`.

## Reproducing the smoke run
```bash
make install           # or: pip install -e ".[zstd,dev]"
make test              # unit + smoke-pipeline tests
make smoke             # tiny end-to-end run (mock backend) -> results/smoke
make smoke-figures     # watermarked tables + figures -> results/smoke/paper_artifacts
```

## Reproducing the real-model proof of concept (small, CPU-feasible)
```bash
pip install -e ".[hf]"
# Downloads/caches HuggingFaceTB/SmolLM2-135M-Instruct on first use; runs a tiny
# synthetic PoC. Outputs are labeled REAL MODEL, PROOF OF CONCEPT (not benchmarks).
python -m ham.cli run    --config configs/poc_real_smollm.yaml --out results/poc_real_smollm
python -m ham.cli report --run-dir results/poc_real_smollm --out results/poc_real_smollm/artifacts
```
The manifest records the exact model id, resolved revision (HF snapshot commit),
Python/OS/CPU, and package versions for that run.

## Reproducing a publication run
```bash
pip install -e ".[hf,faiss,zstd,datasets,plot,instr]"
# LongMemEval oracle (cheap), small model:
python -m ham.cli run --config configs/longmemeval.yaml --out results/lme
python -m ham.cli report --run-dir results/lme --out results/lme/paper_artifacts
```
