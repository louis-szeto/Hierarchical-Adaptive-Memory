"""Stage-D KV-cache-compression experiment: types and constants.

A frozen model's KV cache is compressed under different policies; the policy is
the sole independent variable and the context redundancy is the lever. See
``docs/KVBENCH_PROTOCOL.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

# KV-compression policies (conditions). The policy is the sole variable.
CONDITIONS = [
    "full_kv",             # no compression (reference)
    "ham_kv",              # cluster redundant positions -> fewer + int4 (treatment)
    "uniform_quant_kv",    # int4 all positions, no clustering (isolates precision)
    "h2o_kv",              # evict low-norm positions, float32 (eviction baseline)
    "random_evict_kv",     # evict random positions, float32 (isolates frequency)
    "ham_no_cluster",      # int4 + random position retention (isolates frequency clustering)
]
BASELINE = "full_kv"
TREATMENT = "ham_kv"


@dataclass
class KVResult:
    """One (condition x redundancy x keep_ratio x context) measurement."""

    condition: str
    redundancy: float
    keep_ratio: float             # compression-strength sweep point (fraction retained)
    context_id: int
    kv_bytes: int                 # byte-honest compressed KV-cache size
    n_positions: int              # retained positions (after clustering/eviction)
    decode_latency_s: float       # per-token continuation forward time with this cache
    quality_agreement: float      # next-token top-1 agreement vs full_kv (in [0,1])
    quality_accuracy: float       # next-token accuracy vs ground truth (in [0,1])
