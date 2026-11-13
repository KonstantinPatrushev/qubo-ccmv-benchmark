"""RQ1 statistics (ANALYSIS_PLAN §2 RQ1, §7.1–§7.3): the building blocks, on synthetic data.

Wilson intervals for shares, the stratified one-sided Wilcoxon signed-rank test of H1a (checked
against a brute-force enumeration of the sign patterns and against scipy), the cluster bootstrap
of the estimates, Holm over the RQ1 family.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.proportion import proportion_confint

from cif.stats import rq1

# --------------------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("k", "n"), [(0, 10), (10, 10), (5, 10), (3, 80), (16, 80), (79, 80), (1, 1)])
@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
def test_wilson_matches_statsmodels(k, n, level):
    lo, hi = rq1.wilson_interval(k, n, level)
    ref_lo, ref_hi = proportion_confint(k, n, alpha=1.0 - level, method="wilson")
    assert lo == pytest.approx(ref_lo, abs=1e-12) and hi == pytest.approx(ref_hi, abs=1e-12)
    assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_known_values():
    lo, hi = rq1.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=5e-5)
    lo, hi = rq1.wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=5e-5)


def test_wilson_of_an_empty_group_is_undefined():
    lo, hi = rq1.wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


@pytest.mark.parametrize(("k", "n"), [(-1, 5), (6, 5), (2, -1)])
def test_wilson_refuses_impossible_counts(k, n):
    with pytest.raises(ValueError):
        rq1.wilson_interval(k, n)


# --------------------------------------------------------------------------------------
# Stratified signed-rank test
# --------------------------------------------------------------------------------------


def _brute_force_p(groups: list[np.ndarray], alternative: str = "greater") -> float:
    """P(T >= t_obs) over all 2^n sign patterns of the non-zero differences, ranks within strata
    (average ranks of |d|, zeros dropped): the definition of the exact conditional test."""
    ranks = []
    t_obs = 0.0
    for d in groups:
        d = d[d != 0.0]
        r = sps.rankdata(np.abs(d), method="average")
        ranks.append(r)
        t_obs += float(r[d > 0].sum())
    flat = np.concatenate(ranks) if ranks else np.empty(0)
    if alternative == "less":
        t_obs = float(flat.sum()) - t_obs
    count = 0
    total = 0
    for signs in itertools.product((0, 1), repeat=flat.size):
        t = float(np.dot(signs, flat))
        total += 1
        count += t >= t_obs - 1e-9
    return count / total


def test_exact_single_stratum_matches_scipy_without_ties():
    rng = np.random.default_rng(1)
    d = rng.normal(0.4, 1.0, size=12)
    res = rq1.stratified_signed_rank(d, np.zeros(12, dtype=int), method="exact")
    ref = sps.wilcoxon(d, alternative="greater", method="exact")
    assert res.p_value == pytest.approx(ref.pvalue, abs=1e-12)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.method == "exact" and res.alternative == "greater"


def test_exact_matches_brute_force_with_ties_and_zeros():
    d = np.array([0.5, 0.5, -0.25, 0.0, 1.0, 0.25, -0.5, 0.0, 0.75, 0.25])
    res = rq1.stratified_signed_rank(d, np.zeros(d.size, dtype=int), method="exact")
    assert res.p_value == pytest.approx(_brute_force_p([d]), abs=1e-12)
    assert res.n_zero == 2 and res.n_nonzero == 8 and res.n_total == 10


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_exact_stratified_matches_brute_force(alternative):
    d1 = np.array([0.3, -0.1, 0.0, 0.7, 0.2, 0.2])
    d2 = np.array([-0.4, 0.9, 0.1, 0.1, 0.5, -0.05])
    d = np.concatenate([d1, d2])
    strata = np.array([16] * d1.size + [24] * d2.size)
    res = rq1.stratified_signed_rank(d, strata, alternative=alternative, method="exact")
    assert res.p_value == pytest.approx(_brute_force_p([d1, d2], alternative), abs=1e-12)
    assert set(res.strata) == {16, 24}
    assert res.strata[16]["n_zero"] == 1 and res.strata[24]["n_zero"] == 0


def test_ranks_are_taken_within_strata():
    """Pooling the strata into one would change the ranks, hence the statistic."""
    d1 = np.array([0.1, 0.2, 0.3, -0.05])
    d2 = np.array([1.0, 2.0, -3.0, 4.0])
    d = np.concatenate([d1, d2])
    strata = np.array([0] * 4 + [1] * 4)
    stratified = rq1.stratified_signed_rank(d, strata, method="exact")
    pooled = rq1.stratified_signed_rank(d, np.zeros(8, dtype=int), method="exact")
    # within strata: T1 = 2 + 3 + 4 -> ranks of (0.05, 0.1, 0.2, 0.3) are 1..4, positives 2+3+4;
    # T2 = ranks of (1, 2, 3, 4) -> positives 1 + 2 + 4
    assert stratified.statistic == pytest.approx((2 + 3 + 4) + (1 + 2 + 4))
    assert stratified.statistic != pooled.statistic


def test_less_is_the_mirror_of_greater():
    rng = np.random.default_rng(3)
    d = rng.normal(0.0, 1.0, size=30)
    strata = rng.integers(0, 2, size=30)
    less = rq1.stratified_signed_rank(d, strata, alternative="less")
    greater_of_negated = rq1.stratified_signed_rank(-d, strata, alternative="greater")
    assert less.p_value == pytest.approx(greater_of_negated.p_value, abs=1e-12)


def test_approximation_agrees_with_scipy_including_the_tie_correction():
    rng = np.random.default_rng(4)
    d = np.round(rng.normal(0.3, 1.0, size=60), 1)  # many ties and some zeros
    res = rq1.stratified_signed_rank(d, np.zeros(60, dtype=int), method="approx")
    ref = sps.wilcoxon(d, alternative="greater", method="approx", zero_method="wilcox",
                       correction=False)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.statistic == pytest.approx(ref.statistic)
    assert res.z is not None and res.variance > 0.0


def test_exact_and_approximate_p_values_are_close_at_the_planned_size():
    rng = np.random.default_rng(5)
    d = np.concatenate([rng.normal(0.2, 1.0, 40), rng.normal(0.2, 1.0, 40)])
    strata = np.repeat([16, 24], 40)
    exact = rq1.stratified_signed_rank(d, strata, method="exact")
    approx = rq1.stratified_signed_rank(d, strata, method="approx")
    assert abs(exact.p_value - approx.p_value) < 0.01


def test_auto_method_is_exact_for_the_planned_size():
    d = np.random.default_rng(6).normal(size=80)
    res = rq1.stratified_signed_rank(d, np.repeat([16, 24], 40))
    assert res.method == "exact"


def test_a_positive_shift_is_detected_and_a_negative_one_is_not():
    rng = np.random.default_rng(7)
    strata = np.repeat([16, 24], 40)
    up = rq1.stratified_signed_rank(rng.normal(0.8, 1.0, 80), strata)
    down = rq1.stratified_signed_rank(rng.normal(-0.8, 1.0, 80), strata)
    assert up.p_value < 0.001 and down.p_value > 0.99


def test_the_null_is_calibrated():
    """Under a symmetric null the exact one-sided test rejects at 5 % about 5 % of the time."""
    rng = np.random.default_rng(8)
    strata = np.repeat([16, 24], 20)
    reject = sum(
        rq1.stratified_signed_rank(rng.standard_t(3, size=40), strata).p_value <= 0.05
        for _ in range(400)
    )
    assert 0.02 <= reject / 400 <= 0.08  # conservative for a discrete statistic, never liberal by much


def test_all_differences_zero_gives_p_one():
    res = rq1.stratified_signed_rank(np.zeros(6), np.zeros(6, dtype=int))
    assert res.p_value == 1.0 and res.n_nonzero == 0 and res.statistic == 0.0


def test_numerical_zeros_are_dropped_with_a_tolerance():
    d = np.array([1e-12, -1e-13, 0.5, 0.25, -0.1, 0.75])
    strict = rq1.stratified_signed_rank(d, np.zeros(6, dtype=int), zero_tol=0.0)
    loose = rq1.stratified_signed_rank(d, np.zeros(6, dtype=int), zero_tol=1e-9)
    assert strict.n_zero == 0 and loose.n_zero == 2 and loose.n_nonzero == 4


def test_the_result_reports_per_stratum_counts_and_medians():
    d = np.array([0.5, -0.5, 0.25, 1.0, 2.0, -1.0, 0.0])
    strata = np.array([16, 16, 16, 24, 24, 24, 24])
    res = rq1.stratified_signed_rank(d, strata)
    assert res.strata[16]["n"] == 3 and res.strata[16]["n_pos"] == 2 and res.strata[16]["n_neg"] == 1
    assert res.strata[24]["n_zero"] == 1 and res.strata[24]["median_d"] == pytest.approx(0.5)
    # |d| = (0.5, 0.5, 0.25): ranks (2.5, 2.5, 1); the positive differences hold 2.5 + 1
    assert res.strata[16]["t"] == pytest.approx(3.5)


@pytest.mark.parametrize(
    "bad",
    [
        {"d": [1.0, 2.0], "strata": [0]},  # length mismatch
        {"d": [1.0, float("nan")], "strata": [0, 0]},
        {"d": [1.0, float("inf")], "strata": [0, 0]},
        {"d": [], "strata": []},
    ],
)
def test_malformed_input_is_refused(bad):
    with pytest.raises(ValueError):
        rq1.stratified_signed_rank(np.array(bad["d"]), np.array(bad["strata"]))


def test_unknown_alternative_or_method_is_refused():
    with pytest.raises(ValueError, match="alternative"):
        rq1.stratified_signed_rank(np.ones(3), np.zeros(3, dtype=int), alternative="two-sided")
    with pytest.raises(ValueError, match="method"):
        rq1.stratified_signed_rank(np.ones(3), np.zeros(3, dtype=int), method="asymptotic")


# --------------------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_and_seed_dependent():
    rng = np.random.default_rng(10)
    v = rng.normal(size=40)
    cl = np.arange(40)
    st = np.repeat([16, 24], 20)
    a = rq1.cluster_bootstrap(v, cl, st, stat="mean", n_boot=999, seed=[7, 1])
    b = rq1.cluster_bootstrap(v, cl, st, stat="mean", n_boot=999, seed=[7, 1])
    c = rq1.cluster_bootstrap(v, cl, st, stat="mean", n_boot=999, seed=[7, 2])
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)
    assert a.estimate == pytest.approx(v.mean())


def test_bootstrap_interval_of_a_mean_is_close_to_the_normal_one():
    rng = np.random.default_rng(11)
    v = rng.normal(2.0, 3.0, size=200)
    res = rq1.cluster_bootstrap(v, np.arange(200), np.zeros(200, dtype=int), stat="mean",
                                n_boot=4999, seed=1)
    half = 1.96 * v.std(ddof=1) / math.sqrt(200)
    assert (res.ci_high - res.ci_low) / 2 == pytest.approx(half, rel=0.12)
    assert res.ci_low < v.mean() < res.ci_high and res.n_clusters == 200 and res.n_rows == 200


def test_resampling_is_stratified():
    """Stratum A holds three zeros, stratum B one 1: with fixed stratum sizes the resampled mean is
    always 0.25; an unstratified bootstrap would give 0, 0.25, 0.5, ... ."""
    v = np.array([0.0, 0.0, 0.0, 1.0])
    st = np.array(["a", "a", "a", "b"])
    res = rq1.cluster_bootstrap(v, np.arange(4), st, stat="mean", n_boot=500, seed=2)
    assert res.ci_low == res.ci_high == pytest.approx(0.25)


def test_rows_of_a_cluster_are_resampled_together():
    """Two identical rows per cluster: resampling clusters is less smooth than resampling rows, so
    the interval is wider than the one that wrongly treats the rows as independent."""
    v = np.repeat(np.array([1.0, 2.0, 3.0, 10.0]), 2)
    cl = np.repeat(np.arange(4), 2)
    st = np.zeros(8, dtype=int)
    res = rq1.cluster_bootstrap(v, cl, st, stat="mean", n_boot=2000, seed=3)
    assert res.n_clusters == 4 and res.n_rows == 8
    unclustered = rq1.cluster_bootstrap(v, np.arange(8), st, stat="mean", n_boot=2000, seed=3)
    assert (res.ci_high - res.ci_low) > (unclustered.ci_high - unclustered.ci_low)


def test_bootstrap_of_a_median():
    rng = np.random.default_rng(12)
    v = rng.exponential(1.0, size=101)
    res = rq1.cluster_bootstrap(v, np.arange(101), np.zeros(101, dtype=int), stat="median",
                                n_boot=1999, seed=4)
    assert res.estimate == pytest.approx(np.median(v))
    assert res.ci_low < np.median(v) < res.ci_high


def test_bootstrap_needs_at_least_two_clusters_to_say_anything():
    res = rq1.cluster_bootstrap(np.array([1.0, 2.0]), np.array([0, 0]), np.array([0, 0]),
                                stat="mean", n_boot=99, seed=5)
    assert res.estimate == pytest.approx(1.5)
    assert math.isnan(res.ci_low) and math.isnan(res.ci_high) and res.n_clusters == 1


def test_bootstrap_refuses_unknown_statistics_and_bad_input():
    with pytest.raises(ValueError, match="stat"):
        rq1.cluster_bootstrap(np.ones(3), np.arange(3), np.zeros(3), stat="mode", n_boot=9, seed=1)
    with pytest.raises(ValueError):
        rq1.cluster_bootstrap(np.ones(3), np.arange(2), np.zeros(3), stat="mean", n_boot=9, seed=1)
    with pytest.raises(ValueError):
        rq1.cluster_bootstrap(np.array([1.0, np.nan]), np.arange(2), np.zeros(2), stat="mean",
                              n_boot=9, seed=1)


def test_a_table_of_columns_shares_the_resamples():
    """Several columns are bootstrapped on the same draws of clusters: a column that is the negative
    of another has the negated, mirrored interval; identical columns have identical ones."""
    rng = np.random.default_rng(13)
    v = rng.normal(1.0, 2.0, size=60)
    out = rq1.cluster_bootstrap_table({"a": v, "neg": -v, "same": v}, np.arange(60), np.repeat([0, 1], 30),
                                      stats=("mean", "median"), n_boot=999, seed=[1, 2])
    a, neg, same = out["a"]["mean"], out["neg"]["mean"], out["same"]["mean"]
    assert neg.ci_low == pytest.approx(-a.ci_high) and neg.ci_high == pytest.approx(-a.ci_low)
    assert (same.ci_low, same.ci_high) == (a.ci_low, a.ci_high)
    assert out["a"]["median"].estimate == pytest.approx(np.median(v))
    assert set(out) == {"a", "neg", "same"} and set(out["a"]) == {"mean", "median"}


def test_the_single_column_bootstrap_is_the_table_of_one_column():
    rng = np.random.default_rng(14)
    v = rng.normal(size=50)
    cl, st = np.arange(50), np.repeat([0, 1], 25)
    single = rq1.cluster_bootstrap(v, cl, st, stat="median", n_boot=499, seed=3)
    table = rq1.cluster_bootstrap_table({"v": v}, cl, st, stats=("median",), n_boot=499, seed=3)["v"]["median"]
    assert single == table


def test_missing_values_are_skipped_per_column():
    v = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    w = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    out = rq1.cluster_bootstrap_table({"v": v, "w": w}, np.arange(6), np.zeros(6, dtype=int),
                                      stats=("mean",), n_boot=199, seed=4)
    assert out["v"]["mean"].estimate == pytest.approx(np.nanmean(v))
    assert out["w"]["mean"].estimate == pytest.approx(w.mean())
    assert out["v"]["mean"].n_rows == 5 and out["w"]["mean"].n_rows == 6
    assert np.isfinite(out["v"]["mean"].ci_low)


def test_an_all_missing_column_is_nan_not_an_error():
    out = rq1.cluster_bootstrap_table({"v": np.full(6, np.nan), "w": np.arange(6.0)}, np.arange(6),
                                      np.zeros(6, dtype=int), stats=("mean", "median"), n_boot=99, seed=5)
    assert math.isnan(out["v"]["mean"].estimate) and math.isnan(out["v"]["median"].ci_low)
    assert out["v"]["mean"].n_rows == 0
    assert out["w"]["mean"].estimate == pytest.approx(2.5)


def test_the_table_refuses_infinite_values_and_unknown_statistics():
    with pytest.raises(ValueError, match=r"finite|infinite"):
        rq1.cluster_bootstrap_table({"v": np.array([1.0, np.inf])}, np.arange(2), np.zeros(2), n_boot=9, seed=1)
    with pytest.raises(ValueError, match="stat"):
        rq1.cluster_bootstrap_table({"v": np.ones(3)}, np.arange(3), np.zeros(3), stats=("mode",), n_boot=9, seed=1)
    with pytest.raises(ValueError):
        rq1.cluster_bootstrap_table({}, np.arange(3), np.zeros(3), n_boot=9, seed=1)


# --------------------------------------------------------------------------------------
# Holm over the RQ1 family
# --------------------------------------------------------------------------------------


def test_holm_over_h1a_and_h1c():
    res = rq1.holm_family({"H1a": 0.02, "H1c": 0.30})
    assert res["complete"] is True
    assert res["adjusted"]["H1a"] == pytest.approx(0.04)
    assert res["adjusted"]["H1c"] == pytest.approx(0.30)
    assert res["reject"] == {"H1a": True, "H1c": False}
    res = rq1.holm_family({"H1a": 0.03, "H1c": 0.04})
    assert res["adjusted"]["H1a"] == pytest.approx(0.06) and res["adjusted"]["H1c"] == pytest.approx(0.06)
    assert res["reject"] == {"H1a": False, "H1c": False}


def test_holm_step_down_enforces_monotonicity():
    res = rq1.holm_family({"H1a": 0.20, "H1c": 0.01})
    assert res["adjusted"]["H1c"] == pytest.approx(0.02)
    assert res["adjusted"]["H1a"] == pytest.approx(0.20)  # max(0.02, 0.20 * 1)


def test_an_incomplete_family_is_not_adjusted_but_bounded():
    res = rq1.holm_family({"H1a": 0.01, "H1c": None})
    assert res["complete"] is False
    assert res["adjusted"] == {"H1a": None, "H1c": None}
    assert res["upper_bound"]["H1a"] == pytest.approx(0.02) and res["upper_bound"]["H1c"] is None
    assert res["reject"] == {"H1a": None, "H1c": None}
    assert "H1c" in res["note"]


def test_the_family_is_exactly_h1a_and_h1c():
    with pytest.raises(ValueError, match="H1a"):
        rq1.holm_family({"H1c": 0.1})
    with pytest.raises(ValueError, match="H1b"):
        rq1.holm_family({"H1a": 0.1, "H1c": 0.1, "H1b": 0.2})
    with pytest.raises(ValueError):
        rq1.holm_family({"H1a": 1.5, "H1c": 0.1})


def test_holm_bound_on_the_incomplete_family_is_an_upper_bound_of_the_adjusted_value():
    for p_a in (0.001, 0.02, 0.2, 0.9):
        for p_c in (0.0005, 0.03, 0.5):
            adjusted = rq1.holm_family({"H1a": p_a, "H1c": p_c})["adjusted"]["H1a"]
            bound = rq1.holm_family({"H1a": p_a, "H1c": None})["upper_bound"]["H1a"]
            assert adjusted <= bound + 1e-12
