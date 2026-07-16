"""Typed experiment configuration loaded from YAML.

A single config fully determines a run: model/backend, embeddings, memory
architecture, compression codecs, dataset, conditions, generation params, seeds,
and evaluator. Conditions inherit *all* of these so that the memory mode is the
sole independent variable (see docs/EXPERIMENT_PROTOCOL.md).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml


@dataclass
class BackendConfig:
    kind: str = "mock"  # "mock" | "hf"
    model_id: str = "mock-1"
    device: str = "cpu"  # "cpu" | "cuda" | "auto"
    dtype: str = "float32"  # "float32" | "float16" | "bfloat16"
    quantization: str | None = None  # None | "4bit" | "8bit"
    max_new_tokens: int = 64
    temperature: float = 0.0  # 0.0 => greedy/deterministic
    top_p: float = 1.0
    seed: int = 0
    trust_remote_code: bool = False


@dataclass
class EmbeddingConfig:
    kind: str = "hash"  # "hash" | "sentence-transformers"
    model_id: str = "BAAI/bge-small-en-v1.5"
    dim: int = 256  # hash-embedding dim; ST models override to their native dim
    matryoshka_dim: int | None = None  # truncate embeddings to this many dims if set
    normalize: bool = True
    seed: int = 0


# Lifecycle stages of the LLM development/deployment pipeline (research addendum
# §1). The implemented PoC sits at stage E (inference-time external/persistent
# memory over frozen weights) with a stage-D-flavored variable-precision
# serialization aspect; the *proposed* full HAM layer is stage F (architecture
# level, optionally trainable). These are the only accepted target_stage values.
STAGES = {
    "A_pretraining",
    "B_training_memory",
    "C_finetuning",
    "D_inference_kv_compression",
    "E_inference_external_memory",
    "F_architecture_level",
}
INTEGRATION_MODES = {"external_context", "hidden_state_fusion"}


@dataclass
class StageConfig:
    """Where a run sits in the LLM lifecycle, recorded verbatim in the manifest.

    The runnable publication PoC must keep ``integration_mode='external_context'``
    with frozen base weights; ``hidden_state_fusion`` denotes the architecture
    prototype (unit/toy-tested only, not evaluated on publication benchmarks).
    """

    target_stage: str = "E_inference_external_memory"
    base_weights_changed: bool = False
    persistent_across_sessions: bool = True
    integration_mode: str = "external_context"
    trainable_router: bool = False

    def __post_init__(self) -> None:
        if self.target_stage not in STAGES:
            raise ValueError(
                f"unknown target_stage {self.target_stage!r}; valid: {sorted(STAGES)}"
            )
        if self.integration_mode not in INTEGRATION_MODES:
            raise ValueError(
                f"unknown integration_mode {self.integration_mode!r}; "
                f"valid: {sorted(INTEGRATION_MODES)}"
            )


@dataclass
class CompressionConfig:
    # Textual payload codec for on-disk episodic/semantic text.
    text_codec: str = "auto"  # "auto" | "zstd" | "zlib" | "raw"
    zstd_level: int = 10
    # Vector quantization for stored embeddings.
    vector_quant: str = "int8"  # "none" | "int8" | "int4" | "pq"
    pq_subvectors: int = 8
    pq_bits: int = 8
    # Utility-driven ("ham") vs "uniform" vs "random" bit allocation across items.
    allocation: str = "ham"  # "ham" | "uniform" | "random"


@dataclass
class MemoryConfig:
    working_capacity: int = 6  # recent turns kept verbatim in-context (Cowan/Miller-inspired)
    retrieval_k: int = 5
    token_budget: int = 512  # max tokens the retrieved context may occupy
    chunk_max_chars: int = 400
    # Importance weights: frequency, reuse, recency, novelty, predictive_utility, stability.
    w_frequency: float = 0.15
    w_reuse: float = 0.20
    w_recency: float = 0.20
    w_novelty: float = 0.15
    w_predictive_utility: float = 0.20
    w_stability: float = 0.10
    recency_halflife: float = 10.0  # S in R = exp(-t/S)
    # Deterministic tier thresholds on normalized importance in [0, 1].
    semantic_threshold: float = 0.50
    episodic_threshold: float = 0.30
    # Consolidation (online leader clustering into prototypes).
    consolidation_enabled: bool = True
    consolidation_radius: float = 0.25  # cosine-distance radius for a new prototype
    retrieval: str = "cosine"  # "cosine" | "faiss" | "lexical"
    # Max retrievable items kept when an eviction policy is active (recency_fifo
    # baseline). None => unlimited (no eviction), which is HAM's default.
    retention_capacity: int = 16


@dataclass
class DatasetConfig:
    name: str = "synthetic"  # "synthetic" | "longmemeval" | "locomo"
    # Synthetic knobs.
    num_examples: int = 12
    num_sessions: int = 5
    facts_per_session: int = 4
    distractors_per_session: int = 3
    # Real-dataset knobs.
    path: str | None = None  # local json path (longmemeval/locomo)
    hf_repo: str | None = None
    hf_file: str | None = None
    split: str | None = None
    sample_limit: int | None = 20  # cap examples for a cheap run; None => all
    seed: int = 0


@dataclass
class EvalConfig:
    kind: str = "string"  # "string" (deterministic EM/F1) | "llm_judge"
    judge_model: str | None = None
    judge_backend: str = "hf"


@dataclass
class StatsConfig:
    bootstrap_resamples: int = 10000
    permutation_resamples: int = 10000
    ci: float = 0.95
    noninferiority_delta: float = 0.03  # H1 margin: C must be >= B - delta
    seed: int = 0


@dataclass
class ExperimentConfig:
    name: str = "smoke"
    seed: int = 0
    conditions: list[str] = field(
        default_factory=lambda: ["memory_off", "full_history", "ham_memory"]
    )
    stage: StageConfig = field(default_factory=StageConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    notes: str = ""

    @property
    def is_smoke(self) -> bool:
        """A run is a SMOKE run (and its figures are watermarked) iff it uses the
        deterministic mock backend rather than a real model."""
        return self.backend.kind == "mock"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


_NESTED = {
    "stage": StageConfig,
    "backend": BackendConfig,
    "embedding": EmbeddingConfig,
    "compression": CompressionConfig,
    "memory": MemoryConfig,
    "dataset": DatasetConfig,
    "eval": EvalConfig,
    "stats": StatsConfig,
}


def _coerce(cls, data: dict[str, Any]):
    """Build a dataclass, failing loudly on unknown keys (API uncertainty must
    fail loudly, per the protocol) rather than silently ignoring them."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")
    return cls(**data)


def load_config(path: str) -> ExperimentConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    raw = dict(raw)
    kwargs: dict[str, Any] = {}
    for key, cls in _NESTED.items():
        if key in raw:
            section = raw.pop(key)
            if section is None:
                section = {}
            if not isinstance(section, dict):
                raise ValueError(f"config section '{key}' must be a mapping")
            kwargs[key] = _coerce(cls, section)
    valid_top = {f.name for f in ExperimentConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_top
    if unknown:
        raise ValueError(f"ExperimentConfig: unknown config keys {sorted(unknown)}")
    kwargs.update(raw)
    return ExperimentConfig(**kwargs)
