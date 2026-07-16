import numpy as np

from ham import metrics, stats


def test_exact_match_and_f1():
    assert metrics.exact_match("Verona", "verona.") == 1.0
    assert metrics.exact_match("Aldgate", "Verona") == 0.0
    assert metrics.f1_score("the red house", "red house") > 0.5
    assert metrics.contains_gold("the answer is verona", "Verona") == 1.0


def test_task_score_rewards_containment():
    s = metrics.score_example("The capital is Verona.", "Verona")
    assert s["task_score"] == 1.0
    assert s["contains_gold"] == 1.0


def test_paired_bootstrap_diff_zero_when_identical():
    a = [1.0, 0.0, 1.0, 1.0]
    res = stats.paired_bootstrap_diff(a, a, n_resamples=500, seed=0)
    assert res["mean_diff"] == 0.0
    assert res["lo"] == 0.0 and res["hi"] == 0.0


def test_paired_bootstrap_detects_positive_shift():
    a = [1.0] * 10
    b = [0.0] * 10
    res = stats.paired_bootstrap_diff(a, b, n_resamples=500, seed=0)
    assert res["mean_diff"] == 1.0
    assert res["lo"] > 0.0


def test_permutation_pvalue_in_unit_interval():
    rng = np.random.default_rng(0)
    a = rng.random(20)
    b = rng.random(20)
    res = stats.paired_permutation_test(a, b, n_resamples=500, seed=0)
    assert 0.0 <= res["p_value"] <= 1.0


def test_mcnemar_and_noninferiority():
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 0, 1, 1]
    m = stats.mcnemar_test(a, b)
    assert 0.0 <= m["p_value"] <= 1.0
    ni = stats.noninferiority([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], delta=0.03,
                              n_resamples=500, seed=0)
    assert ni["non_inferior"] is True
