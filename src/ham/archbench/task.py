"""Synthetic corpora for the stage-F toy experiment, with a controllable
**redundancy lever**.

Redundancy ``r`` in [0, 1] controls how skewed the item-frequency distribution is:
``r=0`` is uniform (no redundancy -- nothing for HAM to compress); ``r->1`` is
strongly Zipf (a few items dominate -- high redundancy -- HAM compresses a lot).
The slope of HAM's advantage versus ``r`` is the proof that *frequency* is the
mechanism.

Two tasks share the lever:
- **associative recall** (primary): streams of (key -> value) pairs; the model
  must learn the association. Quality = accuracy on value positions.
- **next-token LM** (secondary): sequences of repeated motifs. Quality = overall
  next-token accuracy.

Both return ``(input_ids, targets, quality_mask)`` int64/bool arrays of shape
``(n_streams, seq_len)``; ``targets`` are next-token labels (already shifted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD = 0  # reserved token id


@dataclass
class Corpus:
    input_ids: np.ndarray   # (N, T) int64
    targets: np.ndarray     # (N, T) int64 (next-token)
    quality_mask: np.ndarray  # (N, T) bool: positions counted in the quality metric
    task: str
    redundancy: float
    n_items: int            # number of distinct keys (recall) or motifs (lm)


def _skewed_indices(n_distinct: int, n_samples: int, redundancy: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Sample ``n_samples`` item indices from a Zipf-like distribution whose skew
    is controlled by ``redundancy`` (0 = uniform, 1 = highly skewed)."""
    if n_distinct <= 1:
        return np.zeros(n_samples, dtype=np.int64)
    ranks = np.arange(1, n_distinct + 1, dtype=np.float64)
    alpha = redundancy * 3.0  # map [0,1] -> exponent [0,3]
    weights = ranks ** (-alpha)
    weights /= weights.sum()
    return rng.choice(n_distinct, size=n_samples, p=weights).astype(np.int64)


def make_recall_corpus(*, n_streams: int, seq_len: int, vocab: int,
                       n_keys: int, redundancy: float, seed: int) -> Corpus:
    """Streams of (key, value) pairs with skewed key frequency.

    Keys occupy token ids [1, 1+n_keys); each key has one fixed value in
    [1+n_keys, 1+2*n_keys). A stream emits (key, value) pairs; the model learns
    the association. Quality = accuracy on value positions (target is a value)."""
    rng = np.random.default_rng(seed)
    n_keys = max(2, min(n_keys, (vocab - 1) // 2))
    value_of = np.arange(n_keys)  # permutation index -> offset
    rng.shuffle(value_of)
    val_base = 1 + n_keys
    n_pairs = seq_len // 2
    input_ids = np.full((n_streams, seq_len), PAD, dtype=np.int64)
    targets = np.full((n_streams, seq_len), PAD, dtype=np.int64)
    qmask = np.zeros((n_streams, seq_len), dtype=bool)
    for s in range(n_streams):
        keys = _skewed_indices(n_keys, n_pairs, redundancy, rng)
        stream = np.empty(2 * n_pairs, dtype=np.int64)
        stream[0::2] = 1 + keys
        stream[1::2] = val_base + value_of[keys]
        full = np.concatenate([[PAD], stream])[:seq_len + 1]
        input_ids[s] = full[:seq_len]
        targets[s] = full[1:seq_len + 1]
        # Quality positions: where the TARGET is a value token (preceding input was a key).
        qmask[s] = (targets[s] >= val_base) & (targets[s] < val_base + n_keys)
    return Corpus(input_ids, targets, qmask, "recall", redundancy, n_keys)


def make_lm_corpus(*, n_streams: int, seq_len: int, vocab: int,
                   n_motifs: int, motif_len: int, redundancy: float,
                   seed: int) -> Corpus:
    """Next-token sequences built by concatenating motifs sampled with skewed
    frequency. Quality = overall next-token accuracy."""
    rng = np.random.default_rng(seed)
    n_motifs = max(2, min(n_motifs, 256))
    motifs = rng.integers(1, vocab, size=(n_motifs, motif_len)).astype(np.int64)
    input_ids = np.full((n_streams, seq_len), PAD, dtype=np.int64)
    targets = np.full((n_streams, seq_len), PAD, dtype=np.int64)
    qmask = np.zeros((n_streams, seq_len), dtype=bool)
    for s in range(n_streams):
        n_needed = seq_len // motif_len + 1
        picks = _skewed_indices(n_motifs, n_needed, redundancy, rng)
        stream = np.concatenate([motifs[p] for p in picks])[:seq_len + 1]
        input_ids[s] = np.concatenate([stream, np.full(seq_len + 1 - len(stream), PAD)])[:seq_len]
        targets[s] = np.concatenate([stream[1:], np.full(seq_len + 1 - len(stream), PAD)])[:seq_len]
        qmask[s] = targets[s] != PAD
    return Corpus(input_ids, targets, qmask, "lm", redundancy, n_motifs)


def build_corpus(task: str, *, n_streams: int, seq_len: int, vocab: int,
                 n_items: int, redundancy: float, seed: int) -> Corpus:
    """Dispatch on task. ``n_items`` = n_keys (recall) or n_motifs (lm)."""
    if task == "recall":
        return make_recall_corpus(n_streams=n_streams, seq_len=seq_len, vocab=vocab,
                                   n_keys=n_items, redundancy=redundancy, seed=seed)
    if task == "lm":
        return make_lm_corpus(n_streams=n_streams, seq_len=seq_len, vocab=vocab,
                              n_motifs=n_items, motif_len=4, redundancy=redundancy,
                              seed=seed)
    raise ValueError(f"unknown task {task!r}")


def quality_metric(logits: np.ndarray, targets: np.ndarray,
                   quality_mask: np.ndarray) -> float:
    """Mean accuracy over the quality-masked positions. ``logits`` (N, T, V),
    ``targets`` (N, T), ``quality_mask`` (N, T) bool."""
    preds = logits.argmax(axis=-1)
    if not quality_mask.any():
        return 0.0
    correct = (preds == targets) & quality_mask
    return float(correct.sum()) / int(quality_mask.sum())
