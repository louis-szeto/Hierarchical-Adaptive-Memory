"""Condition (and ablation) definitions.

Every condition shares the identical model, prompts, generation params, examples,
seeds, embedder, and evaluator (enforced by the runner). A condition only changes
*how memory is built, tiered, compressed, and retrieved* -- the sole independent
variable. See docs/EXPERIMENT_PROTOCOL.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import CompressionConfig


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    use_memory: bool = True
    mode: str = "retrieval"  # "none" | "full_history" | "retrieval"
    retrieval_method: str = "cosine"  # "cosine" | "faiss" | "lexical"
    consolidation: bool = True
    consolidation_mode: str = "adaptive"  # "adaptive" | "static" (static_prototype)
    eviction: str = "none"  # "none" | "fifo" | "utility" (utility-driven forgetting, Eq 6)
    use_recency: bool = True
    use_novelty: bool = True
    use_reuse: bool = True
    allocation: str = "ham"  # "ham" | "uniform" | "random"
    tiering: str = "ham"  # "ham" | "random"
    vector_quant: str = "int8"  # "none" | "int8" | "int4" | "pq"
    text_codec: str = "auto"  # "raw" | "zlib" | "zstd" | "auto"
    # Lifecycle metadata (research addendum §1). Every runnable PoC baseline is an
    # inference-time, frozen-weight, external-context condition; these fields are
    # surfaced per-condition in manifests and the baselines table so the paper's
    # target-stage/method comparison is explicit and honest.
    integration_mode: str = "external_context"  # "external_context" | "hidden_state_fusion"
    base_weights_changed: bool = False
    persistent: bool = True  # persistent external store across sessions
    description: str = ""
    # Explicit disclosure: implemented analogue under HAM's own harness, not a
    # reproduction of any external system (MemGPT/Mem0/AQLM/RETRO/Mamba/...).
    literature_analogue: str = ""

    @property
    def adaptive_precision(self) -> bool:
        """True iff per-item bit allocation is utility-driven (HAM), not uniform.
        Requires an active compressed store (so no-memory/full-text conditions are
        correctly reported as non-adaptive)."""
        return (self.use_memory and self.mode == "retrieval"
                and self.allocation == "ham" and self.vector_quant in ("int8", "int4"))


def _base_ham(comp: CompressionConfig) -> dict:
    return dict(
        use_memory=True,
        mode="retrieval",
        retrieval_method="cosine",
        consolidation=True,
        eviction="utility",
        use_recency=True,
        use_novelty=True,
        use_reuse=True,
        allocation=comp.allocation,
        tiering="ham",
        vector_quant=comp.vector_quant,
        text_codec=comp.text_codec,
    )


# Registry of the canonical conditions + ablations required by the protocol and
# the research addendum §3 baseline set. Every one is an *implemented behavioral
# analogue* under HAM's own harness -- NOT a reproduction of any external paper.
CONDITION_NAMES = [
    "memory_off",
    "full_history",
    "uncompressed_rag",
    "uncompressed_retrieval",  # alias of uncompressed_rag (back-compat)
    "recency_fifo",
    "static_prototype",
    "ham_memory",
    "uniform_quantization",
    "random_tiering",
    "no_consolidation",
    "no_recency",
    "no_novelty",
    "no_reuse",
    "lexical_retrieval",
]

# Conditions that form the fair, runnable comparison for the paper's baseline
# table (all share the same frozen model / dataset / evaluator). Ablations that
# only perturb one HAM signal are excluded from the headline baseline table.
BASELINE_CONDITIONS = [
    "memory_off",
    "full_history",
    "uncompressed_rag",
    "recency_fifo",
    "static_prototype",
    "uniform_quantization",
    "ham_memory",
]


def build_condition(name: str, comp: CompressionConfig) -> ConditionSpec:
    ham = _base_ham(comp)
    if name == "memory_off":
        return ConditionSpec(name, use_memory=False, mode="none", persistent=False,
                             description="Frozen LLM, question only, no persistent memory (control A).",
                             literature_analogue="no-memory control (implemented)")
    if name in ("full_history", "uncompressed_history"):
        return ConditionSpec(name, mode="full_history", vector_quant="none", text_codec="raw",
                             consolidation=False, persistent=False,
                             description="Entire concatenated history in-context, stored uncompressed.",
                             literature_analogue="full-context control, cf. Mem0 eval (implemented analogue, not reproduction)")
    if name in ("uncompressed_rag", "uncompressed_retrieval", "full_history_retrieval"):
        return ConditionSpec(name, mode="retrieval", retrieval_method="cosine",
                             consolidation=False, allocation="uniform", tiering="ham",
                             vector_quant="none", text_codec="raw",
                             description="RAG-style exact retrieval over full-text chunks + float32 index (baseline B).",
                             literature_analogue="uncompressed RAG, cf. MemGPT/Mem0 external memory (implemented analogue, not reproduction)")
    if name == "recency_fifo":
        return ConditionSpec(name, description="Recency/FIFO eviction: evict oldest regardless of utility (forgetting analogue).",
                             literature_analogue="FIFO forgetting, cf. Mamba selective forget / Mem0 (implemented analogue, not reproduction)",
                             **{**ham, "consolidation": False, "eviction": "fifo",
                                "allocation": "uniform"})
    if name == "static_prototype":
        return ConditionSpec(name, description="Static pre-computed prototypes; no adaptive promotion/consolidation over time.",
                             literature_analogue="static consolidation, cf. A-MEM evolution / HippoRAG index (implemented analogue, not reproduction)",
                             **{**ham, "consolidation_mode": "static"})
    if name == "ham_memory":
        return ConditionSpec(name, description="HAM: tiers + consolidation + utility allocation + VQ + text codec (C).",
                             literature_analogue="HAM (this work); tiered utility-rate adaptive memory", **ham)
    if name == "uniform_quantization":
        return ConditionSpec(name, description="HAM with uniform (non-utility) bit allocation.",
                             literature_analogue="uniform-precision, cf. AQLM improves over uniform (implemented analogue, not reproduction)",
                             **{**ham, "allocation": "uniform"})
    if name == "random_tiering":
        return ConditionSpec(name, description="HAM with random tier assignment.",
                             literature_analogue="random-allocation control (implemented)",
                             **{**ham, "tiering": "random"})
    if name == "no_consolidation":
        return ConditionSpec(name, description="HAM without episodic->semantic consolidation.",
                             literature_analogue="ablation of HAM consolidation",
                             **{**ham, "consolidation": False})
    if name == "no_recency":
        return ConditionSpec(name, description="HAM without the recency signal.",
                             literature_analogue="ablation of HAM recency signal",
                             **{**ham, "use_recency": False})
    if name == "no_novelty":
        return ConditionSpec(name, description="HAM without the novelty signal.",
                             literature_analogue="ablation of HAM novelty signal",
                             **{**ham, "use_novelty": False})
    if name == "no_reuse":
        return ConditionSpec(name, description="HAM without the reuse signal.",
                             literature_analogue="ablation of HAM reuse signal",
                             **{**ham, "use_reuse": False})
    if name == "lexical_retrieval":
        return ConditionSpec(name, description="HAM with lexical-only (BM25-lite) retrieval.",
                             literature_analogue="lexical-retrieval ablation (BM25-lite)",
                             **{**ham, "retrieval_method": "lexical"})
    raise ValueError(f"unknown condition: {name!r} (valid: {CONDITION_NAMES})")
