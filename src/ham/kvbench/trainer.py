"""Real torch trainer for the stage-D KV-cache-compression experiment.

For each redundancy level x context: prefill the context once (full KV), then for
each compression condition compress that KV, rebuild the cache, and run a short
continuation forward to measure byte-honest KV size, decode latency, and
next-token quality (agreement vs full_kv + accuracy vs ground truth). Lazy-imports
torch and fails loudly if absent.
"""

from __future__ import annotations

import time

from ..backends import build_backend
from ..config import KVBenchExperimentConfig
from .kv_compress import compress_cache, extract_legacy_cache, rebuild_cache
from .protocol import KVResult
from .task import make_contexts

_INSTALL_HINT = (
    "the torch kvbench trainer requires torch + transformers; install with "
    "`pip install -e \".[hf]\"`."
)


def _resolve_device(name: str) -> str:
    if name == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return name


class TorchKVTrainer:
    def __init__(self, cfg: KVBenchExperimentConfig):
        try:
            import torch  # noqa: F401
        except Exception as exc:  # fail loudly
            raise RuntimeError(_INSTALL_HINT) from exc
        self.cfg = cfg
        self.kb = cfg.kvbench
        self.device = _resolve_device(self.kb.device)

    def run(self) -> list[KVResult]:
        import torch
        backend = build_backend(self.cfg.backend)
        model = backend.model.to(self.device).eval()
        vocab = backend.tokenizer.vocab_size
        results: list[KVResult] = []
        cont_len = max(4, self.kb.span_len)

        for r in self.kb.redundancy_levels:
            inputs, targets = make_contexts(
                n_contexts=self.kb.n_contexts, context_len=self.kb.context_len,
                n_distinct_spans=self.kb.n_distinct_spans, span_len=self.kb.span_len,
                redundancy=r, cont_len=cont_len, vocab=vocab, seed=self.cfg.seed)
            for ci, (ctx, tgt) in enumerate(zip(inputs, targets)):
                ctx_ids = torch.from_numpy(ctx).unsqueeze(0).to(self.device)
                tgt_ids = torch.from_numpy(tgt).unsqueeze(0).to(self.device)
                gt_next = tgt_ids[0, 1:]
                # Prefill once -> full legacy KV.
                legacy_full = extract_legacy_cache(model, ctx_ids)
                # Reference predictions with the full cache.
                full_cache = rebuild_cache([(K, V) for K, V in legacy_full])
                with torch.no_grad():
                    logits_full = model(tgt_ids, past_key_values=full_cache,
                                        use_cache=False).logits
                preds_full = logits_full[0, :-1].argmax(dim=-1)

                for cond in self.kb.conditions:
                    # full_kv / uniform_quant keep all positions (one point at kr=1);
                    # position-reducing conditions sweep keep_ratios (Pareto).
                    kr_list = [1.0] if cond in ("full_kv", "uniform_quant_kv") \
                        else list(self.kb.keep_ratios)
                    for kr in kr_list:
                        comp, kv_bytes, n_pos = compress_cache(
                            legacy_full, cond, self.kb, self.cfg.seed, kr)
                        cache = rebuild_cache(comp)
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            logits = model(tgt_ids, past_key_values=cache,
                                           use_cache=False).logits
                        latency = (time.perf_counter() - t0) / max(1, tgt_ids.shape[1])
                        preds = logits[0, :-1].argmax(dim=-1)
                        agreement = float((preds == preds_full).float().mean().item())
                        accuracy = float((preds == gt_next).float().mean().item())
                        results.append(KVResult(
                            condition=cond, redundancy=r, keep_ratio=kr, context_id=ci,
                            kv_bytes=int(kv_bytes), n_positions=int(n_pos),
                            decode_latency_s=latency, quality_agreement=agreement,
                            quality_accuracy=accuracy))
        return results
