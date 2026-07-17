"""Archbench task/corpus generators: shapes, the redundancy lever, quality metric.
No torch."""

import numpy as np

from ham.archbench.task import (build_corpus, make_lm_corpus, make_recall_corpus,
                                quality_metric)


def test_recall_corpus_shapes():
    c = make_recall_corpus(n_streams=8, seq_len=40, vocab=64, n_keys=8,
                           redundancy=0.5, seed=0)
    assert c.input_ids.shape == (8, 40)
    assert c.targets.shape == (8, 40)
    assert c.quality_mask.shape == (8, 40)
    assert c.quality_mask.any()
    assert c.input_ids.max() < 64 and c.input_ids.min() >= 0


def test_redundancy_lever_lowers_entropy():
    # Higher redundancy -> more concentrated (lower-entropy) item distribution.
    def entropy(c):
        toks = c.input_ids[c.input_ids > 0]
        _, counts = np.unique(toks, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())

    lo = make_recall_corpus(n_streams=64, seq_len=80, vocab=128, n_keys=16,
                            redundancy=0.0, seed=1)
    hi = make_recall_corpus(n_streams=64, seq_len=80, vocab=128, n_keys=16,
                            redundancy=0.9, seed=1)
    assert entropy(hi) < entropy(lo)


def test_lm_corpus_shapes():
    c = make_lm_corpus(n_streams=4, seq_len=32, vocab=64, n_motifs=8,
                       motif_len=4, redundancy=0.5, seed=0)
    assert c.input_ids.shape == (4, 32)
    assert c.targets.shape == (4, 32)
    assert c.quality_mask.any()


def test_build_corpus_dispatch():
    assert build_corpus("recall", n_streams=2, seq_len=16, vocab=32, n_items=4,
                        redundancy=0.0, seed=0).task == "recall"
    assert build_corpus("lm", n_streams=2, seq_len=16, vocab=32, n_items=4,
                        redundancy=0.0, seed=0).task == "lm"


def test_quality_metric_masked_accuracy():
    logits = np.zeros((1, 3, 4))
    logits[0, 0, 1] = 5
    logits[0, 1, 2] = 5
    logits[0, 2, 0] = 5
    targets = np.array([[1, 2, 0]])
    mask = np.array([[True, True, False]])  # only first two positions counted
    assert abs(quality_metric(logits, targets, mask) - 1.0) < 1e-9
    mask2 = np.array([[True, False, False]])  # one position
    assert abs(quality_metric(logits, targets, mask2) - 1.0) < 1e-9
