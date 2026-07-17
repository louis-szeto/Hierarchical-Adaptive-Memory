"""Long-context corpus for the stage-D KV experiment, with a controllable
**redundancy lever**.

Contexts are built by concatenating short token motifs drawn from a skewed
frequency distribution: ``redundancy=0`` is uniform (many distinct motifs, low
redundancy -- nothing for HAM to compress); ``redundancy->1`` is strongly Zipf
(few motifs repeated -- highly redundant KV, HAM compresses a lot). The slope of
HAM's advantage vs redundancy is the proof.

Returns ``(input_ids, targets)``: ``input_ids`` (n_contexts, context_len) is the
prefill context whose KV is compressed; ``targets`` (n_contexts, cont_len) is a
short ground-truth continuation for the quality metric. Requires no torch.
"""

from __future__ import annotations

import numpy as np

PAD = 0


def _skewed_indices(n_distinct: int, n_samples: int, redundancy: float,
                    rng: np.random.Generator) -> np.ndarray:
    ranks = np.arange(1, n_distinct + 1, dtype=np.float64)
    alpha = redundancy * 3.0
    weights = ranks ** (-alpha)
    weights /= weights.sum()
    return rng.choice(n_distinct, size=n_samples, p=weights).astype(np.int64)


def make_contexts(*, n_contexts: int, context_len: int, n_distinct_spans: int,
                  span_len: int, redundancy: float, cont_len: int, vocab: int,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build prefill contexts + ground-truth continuations.

    Returns (input_ids (N, context_len) int64, targets (N, cont_len) int64).
    Each row is a stream of motifs sampled at the given redundancy; the first
    ``context_len`` tokens are the context and the next ``cont_len`` are the
    continuation (drawn from the same motif stream so they are predictable).
    """
    rng = np.random.default_rng(seed)
    motifs = rng.integers(1, vocab, size=(max(2, n_distinct_spans), span_len)).astype(np.int64)
    total = context_len + cont_len
    inputs = np.full((n_contexts, context_len), PAD, dtype=np.int64)
    targets = np.full((n_contexts, cont_len), PAD, dtype=np.int64)
    for s in range(n_contexts):
        n_needed = total // span_len + 2
        picks = _skewed_indices(motifs.shape[0], n_needed, redundancy, rng)
        stream = np.concatenate([motifs[p] for p in picks])[:total]
        inputs[s] = np.concatenate([stream, np.full(context_len, PAD)])[:context_len]
        targets[s] = np.concatenate([stream[context_len:], np.full(cont_len, PAD)])[:cont_len]
    return inputs, targets
