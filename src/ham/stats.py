"""Statistical comparison utilities: paired bootstrap CIs, paired permutation
tests, McNemar's test, and a non-inferiority check.

All randomized procedures accept a seed for reproducibility.
"""

from __future__ import annotations

import numpy as np


def mean_ci_bootstrap(values, n_resamples: int = 10000, ci: float = 0.95, seed: int = 0):
    x = np.asarray(values, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


def paired_bootstrap_diff(a, b, n_resamples: int = 10000, ci: float = 0.95, seed: int = 0):
    """Bootstrap CI for the paired mean difference (a - b) over shared examples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have identical shape")
    d = a - b
    n = d.shape[0]
    if n == 0:
        return {"mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = d[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


def paired_permutation_test(a, b, n_resamples: int = 10000, seed: int = 0):
    """Two-sided paired permutation test on the mean difference (sign flips)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = d.shape[0]
    obs = abs(d.mean()) if n else 0.0
    if n == 0:
        return {"observed_diff": 0.0, "p_value": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, n))
    perm = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(perm) >= obs - 1e-12) + 1) / (n_resamples + 1)
    return {"observed_diff": float(d.mean()), "p_value": float(p), "n": int(n)}


def mcnemar_test(a_correct, b_correct):
    """Exact McNemar test on paired binary correctness arrays."""
    from scipy.stats import binomtest

    a = np.asarray(a_correct).astype(bool)
    b = np.asarray(b_correct).astype(bool)
    b01 = int(np.sum(~a & b))  # a wrong, b right
    b10 = int(np.sum(a & ~b))  # a right, b wrong
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p_value": 1.0}
    p = binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p)}


def noninferiority(c_scores, b_scores, delta: float, n_resamples: int = 10000,
                   ci: float = 0.95, seed: int = 0):
    """Non-inferiority of C vs B with margin delta: conclude non-inferior if the
    lower CI bound of (C - B) exceeds -delta."""
    res = paired_bootstrap_diff(c_scores, b_scores, n_resamples, ci, seed)
    res["delta"] = delta
    res["non_inferior"] = bool(res["lo"] > -delta) if res["n"] else False
    return res
