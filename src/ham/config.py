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
INTEGRATION_MODES = {"external_context", "hidden_state_fusion", "kv_cache_compression"}


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
    # Paper Eq 5 squash kappa for frequency/reuse diminishing-returns
    # (u_freq = 1 - exp(-x / kappa)). Default 3.0 reproduces the previous
    # hardcoded constant exactly.
    squash_kappa: float = 3.0
    # Paper Eq 7 precision threshold rho: importance >= rho earns 8-bit vectors,
    # otherwise 4-bit. Default 0.66 reproduces the previous hardcoded cutoff.
    precision_threshold: float = 0.66


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


# ---------------------------------------------------------------------------
# Stage-F architecture memory-block compression experiment (toy model).
# Separate from the stage-E config; see docs/ARCHBENCH_PROTOCOL.md. The stage-F
# experiment also carries the fine-tuning post-hoc on its toy models (standard
# flat memory block vs HAM memory block) -- it is NOT a separate experiment.
# ---------------------------------------------------------------------------


@dataclass
class ArchBenchConfig:
    """Stage-F toy-architecture experiment: identical toy LMs differing ONLY in
    their memory-block policy (standard FlatMemory vs HAM-compressed), across
    pre-training/fine-tuning, with a redundancy lever that isolates 'frequency'.

    Conditions: no_memory, standard_memory, ham_memory, ham_uniform,
    ham_no_consolidation, ham_random_alloc. ``redundancy_levels`` is the lever
    (0 = uniform/low redundancy, ->1 = highly redundant / Zipf-skewed).
    """

    trainer: str = "mock"  # "mock" | "torch"
    task: str = "recall"   # "recall" | "lm" | "both"
    regime: str = "pretrain"  # "pretrain" | "finetune" | "both"
    conditions: list[str] = field(default_factory=lambda: [
        "no_memory", "standard_memory", "ham_memory",
        "ham_uniform", "ham_no_consolidation", "ham_random_alloc"])
    redundancy_levels: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 0.9])
    # toy model
    dim: int = 64
    n_layers: int = 2
    vocab: int = 256
    n_heads: int = 4
    memory_layer: int = 0      # which block index carries the memory adapter
    top_k: int = 4
    # memory store
    capacity: int = 512        # FIFO cap for FlatMemory / episodic
    consolidation_radius: float = 0.25
    semantic_bits: int = 4     # int4 prototypes
    # training
    optimizer: str = "adamw"
    learning_rate: float = 1.0e-3
    batch_size: int = 16
    max_steps: int = 200
    checkpoint_every: int = 20
    # task corpus
    seq_len: int = 64
    n_train_streams: int = 256
    n_eval_streams: int = 64
    # iso-quality target for the cost-to-target metric
    target_quality: float = 0.9
    device: str = "cpu"        # "cpu" | "cuda" | "auto"
    # --- Fine-tune regime (regime in {"finetune", "both"}) -------------------
    # The held-out association set is a FRESH corpus: different seed and (for
    # recall) a higher key count, so the model must learn new associations on
    # top of the pretrained checkpoint (option (a) of the design). Drift is
    # measured from the loaded pretrained weights, so it is a real
    # catastrophic-forgetting proxy, not training movement from random init.
    # ``finetune_seed_offset`` is added to ``cfg.seed`` for the held-out corpus.
    finetune_seed_offset: int = 1001
    # Multiplier on the pretrain ``n_items`` to set the held-out n_keys (recall).
    # 1.0 = same key count, different permutation; >1.0 = fresh, larger key set.
    finetune_n_keys_multiplier: float = 2.0
    # Optional path to a previous run's ``pretrained_checkpoints/`` dir for
    # cross-invocation fine-tune (regime = "finetune" reading a pretrain dir).
    # When ``regime == "both"`` the runner passes state dicts in memory and
    # ignores this field; when ``regime == "finetune"`` only, the dir is required.
    finetune_init_from_dir: str | None = None
    # mock-trainer synthetic-curve knobs (mock only)
    mock_std_bytes_per_token: float = 256.0
    mock_ham_compress_at_max_redundancy: float = 0.4
    mock_quality_ceiling: float = 0.98

    def __post_init__(self) -> None:
        if self.trainer not in ("mock", "torch"):
            raise ValueError(f"archbench.trainer must be 'mock' or 'torch', got {self.trainer!r}")
        if self.task not in ("recall", "lm", "both"):
            raise ValueError(f"archbench.task must be 'recall'/'lm'/'both', got {self.task!r}")
        if self.regime not in ("pretrain", "finetune", "both"):
            raise ValueError(f"archbench.regime must be 'pretrain'/'finetune'/'both', got {self.regime!r}")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError(f"archbench.optimizer must be 'adamw' or 'sgd', got {self.optimizer!r}")
        if self.max_steps <= 0 or self.checkpoint_every <= 0 or self.batch_size <= 0:
            raise ValueError("archbench max_steps/checkpoint_every/batch_size must be positive")
        if not self.conditions:
            raise ValueError("archbench.conditions must be non-empty")
        if not self.redundancy_levels:
            raise ValueError("archbench.redundancy_levels must be non-empty")
        if not 0.0 <= self.target_quality <= 1.0:
            raise ValueError("archbench.target_quality must be in [0, 1]")


@dataclass
class ArchBenchExperimentConfig:
    """A stage-F toy-architecture run. The toy model's weights ARE trained
    (lifecycle stage F). ``base_weights_changed`` is truthful (trainer == 'torch');
    mock-trainer runs train nothing and are watermarked ``SMOKE TEST``."""

    name: str = "archbench"
    seed: int = 0
    notes: str = ""
    stage: StageConfig = field(default_factory=lambda: StageConfig(
        target_stage="F_architecture_level", base_weights_changed=True,
        persistent_across_sessions=True, integration_mode="hidden_state_fusion",
        trainable_router=True))
    stats: StatsConfig = field(default_factory=StatsConfig)
    archbench: ArchBenchConfig = field(default_factory=ArchBenchConfig)

    @property
    def is_smoke(self) -> bool:
        return self.archbench.trainer == "mock"

    @property
    def base_weights_changed(self) -> bool:
        return self.archbench.trainer == "torch"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


_ARCHBENCH_NESTED = {
    "stage": StageConfig,
    "stats": StatsConfig,
    "archbench": ArchBenchConfig,
}


def archbench_from_dict(raw: dict[str, Any]) -> ArchBenchExperimentConfig:
    raw = dict(raw)
    kwargs: dict[str, Any] = {}
    for key, cls in _ARCHBENCH_NESTED.items():
        if key in raw:
            section = raw.pop(key)
            if section is None:
                section = {}
            if not isinstance(section, dict):
                raise ValueError(f"config section '{key}' must be a mapping")
            kwargs[key] = _coerce(cls, section)
    valid_top = {f.name for f in ArchBenchExperimentConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_top
    if unknown:
        raise ValueError(
            f"ArchBenchExperimentConfig: unknown config keys {sorted(unknown)}")
    kwargs.update(raw)
    return ArchBenchExperimentConfig(**kwargs)


def load_archbench_config(path: str) -> ArchBenchExperimentConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return archbench_from_dict(raw)


# ---------------------------------------------------------------------------
# Stage-D inference KV-cache compression experiment (real frozen model).
# Separate from the stage-E/F configs; see docs/KVBENCH_PROTOCOL.md.
# ---------------------------------------------------------------------------


@dataclass
class KVBenchConfig:
    """Stage-D experiment: compress a frozen model's KV cache and measure
    byte-honest size, decode latency, and next-token quality, across the
    redundancy lever. Conditions: ``full_kv``, ``ham_kv``, ``uniform_quant_kv``,
    ``h2o_kv``, ``random_evict_kv``, ``ham_no_cluster``.

    ``redundancy_levels`` is the lever (0 = diverse/low-redundancy context,
    ->1 = highly repetitive/redundant). ``keep_ratios`` is the sweep of retained-
    position fractions for the iso-quality Pareto; ``cluster_radius`` controls HAM
    clustering (smaller -> more clusters -> less merging).
    """

    trainer: str = "mock"            # "mock" | "torch"
    conditions: list[str] = field(default_factory=lambda: [
        "full_kv", "ham_kv", "uniform_quant_kv",
        "h2o_kv", "random_evict_kv", "ham_no_cluster"])
    redundancy_levels: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.9])
    # Per-condition compression-strength sweep (iso-quality Pareto). Each position-
    # reducing condition is evaluated at every keep fraction; full_kv/uniform_quant
    # ignore it (they keep all positions). ham_kv selects representatives of the
    # MOST-FREQUENT clusters to fill the budget (frequency-driven selection).
    keep_ratios: list[float] = field(default_factory=lambda: [0.5])
    cluster_radius: float = 0.25     # cosine radius for HAM KV clustering
    kv_bits: int = 4                 # int4 for the quantized conditions
    # corpus
    n_contexts: int = 32
    context_len: int = 256
    n_distinct_spans: int = 32
    span_len: int = 8
    decode_len: int = 16              # autoregressive-decode tokens timed for latency
    # iso-quality target (for the cost/quality Pareto)
    target_quality: float = 0.9
    device: str = "cpu"              # "cpu" | "cuda" | "auto"
    # mock-trainer synthetic-curve knobs (mock only)
    mock_full_bytes_per_position: float = 256.0
    mock_ham_compress_at_max_redundancy: float = 0.4

    def __post_init__(self) -> None:
        if self.trainer not in ("mock", "torch"):
            raise ValueError(f"kvbench.trainer must be 'mock' or 'torch', got {self.trainer!r}")
        if not self.conditions:
            raise ValueError("kvbench.conditions must be non-empty")
        if not self.redundancy_levels:
            raise ValueError("kvbench.redundancy_levels must be non-empty")
        if not self.keep_ratios or not all(0.0 < kr <= 1.0 for kr in self.keep_ratios):
            raise ValueError("kvbench.keep_ratios must be non-empty with values in (0, 1]")
        if not 0.0 < self.cluster_radius <= 1.0:
            raise ValueError("kvbench.cluster_radius must be in (0, 1]")
        if self.kv_bits not in (4, 8):
            raise ValueError("kvbench.kv_bits must be 4 or 8")
        if not 0.0 <= self.target_quality <= 1.0:
            raise ValueError("kvbench.target_quality must be in [0, 1]")


@dataclass
class KVBenchExperimentConfig:
    """A stage-D KV-cache-compression run. The model is FROZEN (weights never
    change); only the KV cache is compressed. ``base_weights_changed`` is always
    False; ``is_smoke`` is ``trainer == 'mock'``."""

    name: str = "kvbench"
    seed: int = 0
    notes: str = ""
    stage: StageConfig = field(default_factory=lambda: StageConfig(
        target_stage="D_inference_kv_compression", base_weights_changed=False,
        persistent_across_sessions=False, integration_mode="kv_cache_compression",
        trainable_router=False))
    backend: BackendConfig = field(default_factory=BackendConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    kvbench: KVBenchConfig = field(default_factory=KVBenchConfig)

    @property
    def is_smoke(self) -> bool:
        return self.kvbench.trainer == "mock"

    @property
    def base_weights_changed(self) -> bool:
        return False  # frozen model; only KV compressed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


_KVBENCH_NESTED = {
    "stage": StageConfig,
    "backend": BackendConfig,
    "stats": StatsConfig,
    "kvbench": KVBenchConfig,
}


def kvbench_from_dict(raw: dict[str, Any]) -> KVBenchExperimentConfig:
    raw = dict(raw)
    kwargs: dict[str, Any] = {}
    for key, cls in _KVBENCH_NESTED.items():
        if key in raw:
            section = raw.pop(key)
            if section is None:
                section = {}
            if not isinstance(section, dict):
                raise ValueError(f"config section '{key}' must be a mapping")
            kwargs[key] = _coerce(cls, section)
    valid_top = {f.name for f in KVBenchExperimentConfig.__dataclass_fields__.values()}
    unknown = set(raw) - valid_top
    if unknown:
        raise ValueError(
            f"KVBenchExperimentConfig: unknown config keys {sorted(unknown)}")
    kwargs.update(raw)
    return KVBenchExperimentConfig(**kwargs)


def load_kvbench_config(path: str) -> KVBenchExperimentConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return kvbench_from_dict(raw)
