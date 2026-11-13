"""RQ2: the pooled estimate of ``log R``, its stratified bootstrap and the five outcomes
(ANALYSIS_PLAN §2 RQ2 "Подтверждающая оценка (N8)", §7.1, §7.2, §10).

The estimator is the mean over cells (equal cell weights) of the cell means of ``log R``; a cell mean is the mean over
the clusters of the cell (an instance; in C5 a port, whose value is the mean over its λ points). Intervals are
percentile intervals of the stratified bootstrap (clusters redrawn within each cell). Golden values are computed by
hand or from an independent loop that follows the documented random stream.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from cif.stats import rq2

LN = math.log(1.5)


# --------------------------------------------------------------------------------------------
# The five outcomes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ratio_interval", "expected"),
    [
        ((0.30, 0.60), "qubo_better"),  # wholly below 1/1.5 = 0.667
        ((1.60, 3.00), "qubo_worse"),  # wholly above 1.5
        ((0.80, 1.20), "equivalent"),
        ((0.70, 0.90), "equivalent"),  # excludes 1, but lies inside the band: equivalent, not "direction only"
        ((1.05, 1.30), "equivalent"),
        ((0.50, 0.90), "direction_only"),  # excludes 1, crosses the lower edge
        ((1.10, 2.00), "direction_only"),  # excludes 1, crosses the upper edge
        ((0.50, 1.20), "undetermined"),  # contains 1, crosses an edge
        ((0.90, 2.00), "undetermined"),
        ((0.40, 2.50), "undetermined"),  # covers the whole band
    ],
)
def test_the_five_outcomes_follow_the_plan(ratio_interval, expected):
    low, high = (math.log(x) for x in ratio_interval)
    assert rq2.outcome(low, high) == expected


def test_the_band_is_closed_and_wholly_outside_is_strict():
    assert rq2.outcome(-LN, LN) == "equivalent"  # the band [1/1.5, 1.5] includes its edges
    assert rq2.outcome(-LN, -LN + 1e-9) == "equivalent"
    assert rq2.outcome(0.0, 0.1) == "equivalent"
    assert rq2.outcome(0.0, 0.0) == "equivalent"  # a point at R = 1
    # touching the lower edge from inside the band is not "wholly below it"; the interval crosses the edge
    assert rq2.outcome(-LN - 1e-9, -LN) == "direction_only"
    assert rq2.outcome(LN, LN + 0.1) == "direction_only"
    assert rq2.outcome(-LN - 1e-9, -LN - 1e-12) == "qubo_better"
    assert rq2.outcome(LN + 1e-12, LN + 1e-9) == "qubo_worse"
    # containing 1 is not "excluding 1"
    assert rq2.outcome(0.0, 1.0) == "undetermined"
    assert rq2.outcome(-1.0, 0.0) == "undetermined"


def test_the_outcome_takes_the_factor_of_the_band_and_refuses_nonsense():
    assert rq2.EQUIVALENCE_FACTOR == 1.5
    assert rq2.outcome(math.log(0.6), math.log(0.9), factor=2.0) == "equivalent"  # inside [0.5, 2]
    assert rq2.outcome(math.log(0.6), math.log(0.9)) == "direction_only"  # 0.6 < 1/1.5
    for low, high in ((0.2, 0.1), (math.nan, 0.1), (0.0, math.inf)):
        with pytest.raises(ValueError, match="interval"):
            rq2.outcome(low, high)
    for factor in (1.0, 0.5, math.nan):
        with pytest.raises(ValueError, match="factor"):
            rq2.outcome(-0.1, 0.1, factor=factor)


def test_the_outcome_identifiers_are_the_five_of_the_plan():
    assert rq2.OUTCOMES == ("qubo_better", "qubo_worse", "equivalent", "direction_only", "undetermined")


# --------------------------------------------------------------------------------------------
# Levels and cluster means
# --------------------------------------------------------------------------------------------


def test_the_bonferroni_level_of_a_cell_in_a_family_of_m():
    assert rq2.FAMILY_LEVEL == 0.99 and rq2.POOLED_LEVEL == 0.95
    assert rq2.family_level(1) == pytest.approx(0.99)
    assert rq2.family_level(4) == pytest.approx(0.9975)  # C1, C2, C3, C5
    assert rq2.family_level(5) == pytest.approx(0.998)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="family"):
            rq2.family_level(bad)


def test_mean_of_means_takes_the_mean_over_clusters_of_the_cluster_means():
    """C5: the mean over the ports of the means over their λ points, not the mean over the points."""
    labels, means = rq2.mean_of_means([0.0, 0.4, 0.6], ["p1", "p1", "p2"])
    assert list(labels) == ["p1", "p2"]
    np.testing.assert_allclose(means, [0.2, 0.6])
    assert float(np.mean(means)) == pytest.approx(0.4) != pytest.approx(1.0 / 3.0)
    # the mean, not the median, of the points of a cluster: [0, 0, 0.9] has the mean 0.3 and the median 0
    _, means = rq2.mean_of_means([0.0, 0.0, 0.9, 0.5], ["p3", "p3", "p3", "p4"])
    np.testing.assert_allclose(means, [0.3, 0.5])
    # the order of the input does not matter; clusters come out sorted
    labels, means = rq2.mean_of_means([0.6, 0.4, 0.0], ["p2", "p1", "p1"])
    assert list(labels) == ["p1", "p2"]
    np.testing.assert_allclose(means, [0.2, 0.6])


def test_mean_of_means_refuses_what_has_no_mean():
    for values, clusters in (([], []), ([1.0, 2.0], ["a"]), ([1.0, math.nan], ["a", "b"])):
        with pytest.raises(ValueError):
            rq2.mean_of_means(values, clusters)


# --------------------------------------------------------------------------------------------
# The pooled estimate
# --------------------------------------------------------------------------------------------


def test_the_pooled_estimate_gives_every_cell_the_same_weight():
    cells = {"A": {"m": [0.1, 0.3]}, "B": {"m": [1.0, 2.0, 3.0, 0.0]}}
    res = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=200, seed=1)
    assert res.cells["m"]["A"].estimate == pytest.approx(0.2)
    assert res.cells["m"]["B"].estimate == pytest.approx(1.5)
    assert res.pooled["m"].estimate == pytest.approx(0.85)  # not (0.1 + 0.3 + 6) / 6 = 1.0667
    assert res.pooled["m"].cells == ("A", "B")


def test_a_cell_outside_the_pool_is_reported_and_does_not_move_the_pooled_estimate():
    cells = {"A": {"m": [0.1, 0.3]}, "B": {"m": [5.0, 7.0]}}
    res = rq2.estimate_rq2(cells, pooled=["A"], n_boot=300, seed=1)
    assert res.pooled["m"].estimate == pytest.approx(0.2) and res.pooled["m"].cells == ("A",)
    b = res.cells["m"]["B"]
    assert b.in_pooled is False and b.estimate == pytest.approx(6.0) and res.cells["m"]["A"].in_pooled is True
    # both cells carry the level of the family of ONE cell; B is outside the family but is shown at the same level
    assert b.interval.level == pytest.approx(0.99) and res.cells["m"]["A"].interval.level == pytest.approx(0.99)


def test_without_a_pooled_cell_there_is_no_pooled_estimate():
    res = rq2.estimate_rq2({"A": {"m": [0.1, 0.3]}}, pooled=[], n_boot=100, seed=1)
    assert res.pooled is None and res.cells["m"]["A"].estimate == pytest.approx(0.2)
    assert res.cells["m"]["A"].interval.level == pytest.approx(0.99)  # a family of one when there is none


def test_two_clusters_make_the_percentile_interval_the_exact_range():
    """The bootstrap mean of [0, 1] is 0, 1/2, 1 with probabilities 1/4, 1/2, 1/4: the 2.5 % and 97.5 % points are
    0 and 1, whatever the seed."""
    for seed in (1, 2, 3):
        res = rq2.estimate_rq2({"A": {"m": [0.0, 1.0]}}, pooled=["A"], n_boot=2000, seed=seed)
        iv = res.pooled["m"].interval
        assert (iv.low, iv.high, iv.level) == (0.0, 1.0, 0.95)


def test_three_clusters_make_the_interval_the_range_and_four_do_not():
    """Each extreme of the bootstrap mean of three clusters has probability 1/27 = 3.7 %, above the 2.5 % tail: the
    interval of a C5 with three ports is the range of its port means. With four clusters it is 1/256: the interval is
    inside the range."""
    three = rq2.estimate_rq2({"A": {"m": [0.0, 1.0, 5.0]}}, pooled=["A"], n_boot=9999, seed=3).pooled["m"].interval
    assert (three.low, three.high) == (0.0, 5.0)
    four = rq2.estimate_rq2({"A": {"m": [0.0, 1.0, 2.0, 5.0]}}, pooled=["A"], n_boot=9999, seed=3).pooled["m"].interval
    assert 0.0 < four.low < four.high < 5.0


def test_the_pooled_interval_is_calibrated_on_simulated_cells_of_unequal_size_mean_and_spread():
    """A check on simulation (fixed seeds, so deterministic): four cells of 20, 20, 30 and 12 clusters with different
    means and spreads; the 95 % interval of the equal-weight mean of the cell means must cover it about 95 % of the
    time (percentile intervals of small samples run a little low: 0.94 at B = 9999 over 1000 repetitions). A bootstrap
    that pooled the clusters of the cells, or weighted them by size, would give a different coverage."""
    rng = np.random.default_rng(12345)
    sizes = {"C1": 20, "C2": 20, "C3": 30, "C5": 12}
    means = {"C1": -0.3, "C2": 0.1, "C3": 0.5, "C5": 0.0}
    sds = {"C1": 0.5, "C2": 1.0, "C3": 2.0, "C5": 0.3}
    truth = float(np.mean(list(means.values())))
    reps, hits = 400, 0
    for r in range(reps):
        cells = {c: {"m": rng.normal(means[c], sds[c], sizes[c])} for c in sizes}
        iv = rq2.estimate_rq2(cells, pooled=list(sizes), n_boot=999, seed=1000 + r).pooled["m"].interval
        hits += iv.low <= truth <= iv.high
    assert 0.89 <= hits / reps <= 0.985


def test_resampling_is_stratified_by_cell():
    """Two cells with no variation inside: the pooled mean of every resample is the same. Resampling the pooled
    instances jointly would move it."""
    cells = {"A": {"m": [10.0, 10.0, 10.0, 10.0]}, "B": {"m": [-10.0, -10.0, -10.0, -10.0]}}
    res = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=500, seed=7)
    p = res.pooled["m"]
    assert p.estimate == 0.0 and (p.interval.low, p.interval.high) == (0.0, 0.0) and p.se == 0.0


def test_the_pooled_resample_is_the_mean_of_the_cell_resamples_under_the_documented_stream():
    """An independent loop: cell ``c`` draws from ``SeedSequence([seed, STREAM_TAG, crc32(c)])`` in chunks of
    ``CHUNK`` resamples of its own size; the pooled value is the mean over the cells of the resampled cell means; the
    interval is ``numpy.quantile`` (linear) of the 95 % level."""
    seed, n_boot = 4321, 2500
    cells = {"A": {"m": [0.5, -0.2, 0.9, 0.1, 0.4, 0.33, -0.61, 0.72, 0.05]},
             "C": {"m": [1.5, 0.7, 1.1, 0.95, 1.27, 0.88, 1.6]}}  # enough clusters for a continuous-looking interval
    res = rq2.estimate_rq2(cells, pooled=["A", "C"], n_boot=n_boot, seed=seed)
    expected = np.zeros(n_boot)
    for name, values in cells.items():
        x = np.asarray(values["m"])
        gen = np.random.default_rng(np.random.SeedSequence([seed, rq2.STREAM_TAG, zlib.crc32(name.encode())]))
        draws = []
        done = 0
        while done < n_boot:
            size = min(rq2.CHUNK, n_boot - done)
            idx = gen.integers(0, x.size, size=(size, x.size))
            draws.extend(float(np.mean([x[j] for j in row])) for row in idx)
            done += size
        expected += np.array(draws) / len(cells)
    lo, hi = np.quantile(expected, [0.025, 0.975])
    p = res.pooled["m"]
    assert p.interval.low == pytest.approx(lo, abs=1e-12) and p.interval.high == pytest.approx(hi, abs=1e-12)
    assert p.se == pytest.approx(float(np.std(expected, ddof=1)), abs=1e-12)
    assert p.estimate == pytest.approx((np.mean(cells["A"]["m"]) + np.mean(cells["C"]["m"])) / 2)


def test_the_stream_of_a_cell_does_not_depend_on_the_other_cells():
    a = {"A": {"m": [0.5, -0.2, 0.9, 0.1, 0.4]}}
    one = rq2.estimate_rq2(a, pooled=["A"], n_boot=1000, seed=5)
    two = rq2.estimate_rq2({**a, "B": {"m": [3.0, 1.0, 2.0]}}, pooled=["A"], n_boot=1000, seed=5)
    assert one.cells["m"]["A"].interval == two.cells["m"]["A"].interval
    assert one.cells["m"]["A"].se == two.cells["m"]["A"].se


def test_one_seed_one_result_and_another_seed_another_resample():
    cells = {"A": {"m": list(np.linspace(-1, 1, 9))}, "B": {"m": list(np.linspace(0, 2, 7))}}
    a = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=800, seed=11)
    b = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=800, seed=11)
    c = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=800, seed=12)
    assert a == b and a.pooled["m"].interval != c.pooled["m"].interval
    assert a.pooled["m"].estimate == c.pooled["m"].estimate


def test_the_metrics_of_a_cell_are_resampled_together():
    """H2 and H2b are estimated on the same instances: one set of redrawn clusters serves every metric, so a linear
    relation between two metrics survives into the intervals."""
    x = [0.3, -0.4, 0.8, 0.1, 0.5, -0.1]
    cells = {"A": {"m1": x, "m2": [2 * v for v in x]}, "B": {"m1": x[:4], "m2": [2 * v for v in x[:4]]}}
    res = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=900, seed=3)
    one, two = res.pooled["m1"], res.pooled["m2"]
    assert two.interval.low == pytest.approx(2 * one.interval.low) and two.interval.high == pytest.approx(
        2 * one.interval.high)
    assert two.se == pytest.approx(2 * one.se)


def test_a_cell_gets_the_bonferroni_interval_of_its_family_and_the_pooled_estimate_the_95_percent_one():
    rng = np.random.default_rng(0)
    cells = {name: {"m": list(rng.normal(0, 1, 12))} for name in ("C1", "C2", "C3", "C5")}
    res = rq2.estimate_rq2(cells, pooled=list(cells), n_boot=3000, seed=9)
    assert res.pooled["m"].interval.level == pytest.approx(0.95)
    for cell in res.cells["m"].values():
        assert cell.interval.level == pytest.approx(0.9975)
    one = rq2.estimate_rq2({"C1": cells["C1"]}, pooled=["C1"], n_boot=3000, seed=9)
    # the same stream and the same resamples: the interval of the family of 4 contains that of the family of 1
    wide, narrow = res.cells["m"]["C1"].interval, one.cells["m"]["C1"].interval
    assert wide.low <= narrow.low and wide.high >= narrow.high and (wide.low, wide.high) != (narrow.low, narrow.high)
    assert res.settings["family_size"] == 4 and res.settings["cell_level"] == pytest.approx(0.9975)


def test_the_outcomes_of_the_pooled_estimate_and_of_the_cells_come_from_their_own_intervals():
    cells = {"A": {"m": [math.log(1.2), math.log(2.0)]}, "B": {"m": [math.log(1.2), math.log(2.0)]}}
    res = rq2.estimate_rq2(cells, pooled=["A", "B"], n_boot=4000, seed=2)
    # the pooled interval is [ln 1.2, ln 2.0] exactly? (a, b) ∈ {1.2, 2.0}² has P(both 1.2) = 1/16 > 2.5 %: yes
    iv = res.pooled["m"].interval
    assert (iv.low, iv.high) == pytest.approx((math.log(1.2), math.log(2.0)), abs=1e-12)
    assert res.pooled["m"].outcome == "direction_only"  # excludes 1, crosses 1.5
    assert res.cells["m"]["A"].outcome == "direction_only"


@pytest.mark.parametrize(
    ("pair", "expected"),
    [
        ((0.30, 0.50), "qubo_better"),
        ((2.00, 3.00), "qubo_worse"),
        ((0.80, 1.20), "equivalent"),
        ((1.20, 2.00), "direction_only"),
        ((0.80, 2.00), "undetermined"),
    ],
)
def test_two_clusters_reproduce_each_outcome_through_the_whole_estimator(pair, expected):
    cells = {"A": {"m": [math.log(pair[0]), math.log(pair[1])]}}
    res = rq2.estimate_rq2(cells, pooled=["A"], n_boot=3000, seed=2)
    assert res.pooled["m"].outcome == expected


# --------------------------------------------------------------------------------------------
# Degenerate input
# --------------------------------------------------------------------------------------------


def test_a_pooled_cell_needs_two_clusters_to_be_resampled_a_descriptive_one_gets_no_interval():
    with pytest.raises(ValueError, match="at least two"):
        rq2.estimate_rq2({"A": {"m": [0.1]}, "B": {"m": [0.2, 0.3]}}, pooled=["A", "B"], n_boot=100, seed=1)
    res = rq2.estimate_rq2({"A": {"m": [0.1, 0.2]}, "B": {"m": [0.3]}}, pooled=["A"], n_boot=100, seed=1)
    b = res.cells["m"]["B"]
    assert b.estimate == pytest.approx(0.3) and b.interval is None and b.outcome is None and b.n_clusters == 1


def test_input_that_cannot_be_estimated_is_refused():
    base = {"A": {"m": [0.1, 0.2]}}
    with pytest.raises(ValueError, match="pooled cell"):
        rq2.estimate_rq2(base, pooled=["Z"], n_boot=100, seed=1)
    with pytest.raises(ValueError, match="finite"):
        rq2.estimate_rq2({"A": {"m": [0.1, math.nan]}}, pooled=["A"], n_boot=100, seed=1)
    with pytest.raises(ValueError, match="metrics"):
        rq2.estimate_rq2({"A": {"m": [0.1, 0.2]}, "B": {"k": [0.1, 0.2]}}, pooled=["A"], n_boot=100, seed=1)
    with pytest.raises(ValueError, match="same number"):
        rq2.estimate_rq2({"A": {"m": [0.1, 0.2], "k": [0.1]}}, pooled=["A"], n_boot=100, seed=1)
    with pytest.raises(ValueError, match="n_boot"):
        rq2.estimate_rq2(base, pooled=["A"], n_boot=0, seed=1)
    with pytest.raises(TypeError, match="seed"):
        rq2.estimate_rq2(base, pooled=["A"], n_boot=100, seed=None)
    with pytest.raises(ValueError, match="no cells"):
        rq2.estimate_rq2({}, pooled=[], n_boot=100, seed=1)


def test_the_prespecified_settings_are_named_constants():
    assert rq2.N_BOOT == 9999 and rq2.SEED == 60000 and rq2.CHUNK == 1000
    assert rq2.EQUIVALENCE_FACTOR == 1.5 and rq2.POOLED_LEVEL == 0.95 and rq2.FAMILY_LEVEL == 0.99


def test_the_result_is_json_ready():
    import json

    res = rq2.estimate_rq2({"A": {"m": [0.1, 0.2, 0.4]}, "B": {"m": [0.3]}}, pooled=["A"], n_boot=200, seed=1)
    text = json.dumps(res.as_dict(), allow_nan=False)
    back = json.loads(text)
    assert back["pooled"]["m"]["outcome"] == res.pooled["m"].outcome
    assert back["cells"]["m"]["B"]["interval"] is None and back["settings"]["n_boot"] == 200
