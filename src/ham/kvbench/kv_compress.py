"""KV-cache compressor for the stage-D experiment (the single independent variable).

Operates on a frozen model's KV cache in the transformers *legacy* format (a tuple
of per-layer ``(K, V)``, each ``(batch, num_kv_heads, seq, head_dim)``). Each
condition derives ONE position structure -- a clustering (groups) or an eviction
index set -- from a reference (layer-0 K, mean over heads), then applies that
SAME structure to every layer, so the rebuilt ``DynamicCache`` has consistent
per-layer sequence lengths.

- ``full_kv``: verbatim, float32.
- ``ham_kv``: leader-cluster redundant positions into prototypes (fewer) + int4.
- ``uniform_quant_kv``: int4 all positions, no clustering.
- ``h2o_kv``: evict low-norm positions, float32.
- ``random_evict_kv``: evict random positions, float32.
- ``ham_no_cluster``: int4 + random position retention.

``byte_size`` is byte-honest (int4 via packed-nibble sizes from
``compression.vector_quant``). int4 quantization error is applied as
quantize->dequantize so the re-injected (float) cache the model attends over
reflects the compression loss. Importing requires torch.
"""

from __future__ import annotations

import random

import torch

from ..compression.vector_quant import quantize, quantized_nbytes

_FLOAT32 = 4


def extract_legacy_cache(model, input_ids) -> tuple:
    """Prefill ``input_ids`` and return the KV cache in legacy tuple format
    (per-layer ``(K, V)``, each ``(batch, num_kv_heads, seq, head_dim)``).

    Version-robust: uses the legacy ``to_legacy_cache()`` API where available
    (transformers <=4.4x), otherwise reads the current ``DynamicCache.layers``
    API (transformers 5.x) in which ``to_legacy_cache``/``from_legacy_cache``
    were removed and per-layer tensors live on ``layer.keys``/``layer.values``.
    """
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache = out.past_key_values
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple((layer.keys, layer.values) for layer in cache.layers)


def rebuild_cache(comp_legacy):
    """Rebuild a DynamicCache from a compressed legacy tuple for re-injection.

    Version-robust: ``from_legacy_cache`` where available, otherwise the current
    ``DynamicCache(ddp_cache_data=...)`` constructor.
    """
    from transformers.cache_utils import DynamicCache
    comp_legacy = tuple(comp_legacy)
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(comp_legacy)
    return DynamicCache(ddp_cache_data=list(comp_legacy))


def _shapes(legacy):
    batch, n_kv_heads, seq, head_dim = legacy[0][0].shape
    return batch, n_kv_heads, seq, head_dim


def _float_bytes(n_positions, n_kv_heads, head_dim, n_layers):
    return n_layers * n_positions * n_kv_heads * head_dim * _FLOAT32 * 2  # K + V


def _int4_bytes(t: torch.Tensor) -> int:
    b, nh, np_, hd = t.shape
    flat = t.reshape(b * nh * np_, hd).detach().cpu().to(torch.float32).numpy()
    return quantized_nbytes(quantize(flat, "int4"))


def _qd(t: torch.Tensor, bits: int) -> torch.Tensor:
    """quantize->dequantize (lossy for int4) so the forward sees compressed fidelity."""
    b, nh, np_, hd = t.shape
    flat = t.reshape(b * nh * np_, hd).detach().cpu().to(torch.float32).numpy()
    dq = quantize(flat, f"int{bits}").dequantize().reshape(b, nh, np_, hd)
    return torch.from_numpy(dq).to(device=t.device, dtype=t.dtype)


def _cluster_groups(kref: torch.Tensor, radius: float) -> list[list[int]]:
    """Leader-cluster positions (rows of kref) by cosine similarity >= 1-radius."""
    seq = kref.shape[0]
    kn = kref / (kref.norm(dim=1, keepdim=True) + 1e-8)
    groups: list[list[int]] = []
    assigned = [False] * seq
    for i in range(seq):
        if assigned[i]:
            continue
        sim = kn @ kn[i]
        members = [j for j in range(seq) if not assigned[j] and float(sim[j]) >= 1.0 - radius]
        for j in members:
            assigned[j] = True
        groups.append(members)
    return groups


def _merge_layer(K, V, groups):
    Km = torch.stack([K[:, :, g, :].mean(dim=2) for g in groups], dim=2)
    Vm = torch.stack([V[:, :, g, :].mean(dim=2) for g in groups], dim=2)
    return Km, Vm


def _select_layer(K, V, idx):
    return K[:, :, idx, :], V[:, :, idx, :]


def compress_cache(legacy, condition: str, cfg, seed: int, keep_ratio: float = 1.0):
    """Compress a legacy KV cache per ``condition`` at strength ``keep_ratio``
    (fraction of positions to retain; ignored by full_kv/uniform_quant).

    Returns (compressed_legacy, kv_bytes, n_positions). ``compressed_legacy`` is
    float (dequantized for int4 conditions) and ready for ``rebuild_cache``.

    ``ham_kv`` clusters positions, then keeps the representatives of the MOST-
    FREQUENT clusters up to the budget -- frequency-driven selection. At high
    redundancy, few large clusters cover the context, so HAM hits the coverage
    with far fewer representatives (and int4), dominating on bytes at iso-quality.
    """
    rng = random.Random(f"{seed}:{condition}:{keep_ratio}")
    n_layers = len(legacy)
    _batch, n_kv_heads, seq, head_dim = _shapes(legacy)
    radius = cfg.cluster_radius
    n_target = max(1, min(seq, int(round(seq * keep_ratio))))
    bits = cfg.kv_bits
    kref = legacy[0][0][0].mean(dim=0)  # (seq, head_dim): batch 0, mean over kv-heads

    if condition == "full_kv":
        comp = [(K, V) for K, V in legacy]
        return comp, _float_bytes(seq, n_kv_heads, head_dim, n_layers), seq

    if condition == "uniform_quant_kv":
        comp = [(_qd(K, bits), _qd(V, bits)) for K, V in legacy]
        bytes_ = sum(_int4_bytes(c[0]) + _int4_bytes(c[1]) for c in comp)
        return comp, bytes_, seq

    if condition == "ham_kv":
        # Frequency-weighted position retention: keep ``n_target`` REAL positions
        # (not merged reps -- merging corrupts real-model KV attention), filling
        # the budget from the MOST-FREQUENT clusters first. At high redundancy the
        # largest clusters hold the repeated/common content, so prioritising them
        # preserves the prediction at a given position budget -> Pareto win over
        # random/norm selection. int4 on top.
        groups = _cluster_groups(kref, radius)
        kept: list[int] = []
        for g in sorted(groups, key=len, reverse=True):
            kept.extend(g)
            if len(kept) >= n_target:
                break
        idx = sorted(kept[:n_target]) if kept else list(range(n_target))
        comp = [(_qd(Ks, bits), _qd(Vs, bits))
                for Ks, Vs in (_select_layer(K, V, idx) for K, V in legacy)]
        bytes_ = sum(_int4_bytes(c[0]) + _int4_bytes(c[1]) for c in comp)
        return comp, bytes_, len(idx)

    if condition == "ham_no_cluster":
        idx = sorted(rng.sample(range(seq), n_target))
        comp = [(_qd(Ks, bits), _qd(Vs, bits))
                for Ks, Vs in (_select_layer(K, V, idx) for K, V in legacy)]
        bytes_ = sum(_int4_bytes(c[0]) + _int4_bytes(c[1]) for c in comp)
        return comp, bytes_, n_target

    if condition == "h2o_kv":
        scores = kref.norm(dim=1)
        idx = sorted(torch.topk(scores, n_target).indices.tolist())
        comp = [_select_layer(K, V, idx) for K, V in legacy]
        return comp, _float_bytes(n_target, n_kv_heads, head_dim, n_layers), n_target

    if condition == "random_evict_kv":
        idx = sorted(rng.sample(range(seq), n_target))
        comp = [_select_layer(K, V, idx) for K, V in legacy]
        return comp, _float_bytes(n_target, n_kv_heads, head_dim, n_layers), n_target

    raise ValueError(f"unknown kv condition {condition!r}")
