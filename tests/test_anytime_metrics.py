"""Tests for cif.metrics.anytime: incumbent curves, primal integral, time-to-target, Kaplan–Meier.

The definitions are those of ANALYSIS_PLAN §5 (N5, M8): ``PI(T) = (1/T)∫₀ᵀ γ dt`` with
``γ(t) = min(1, ĝ(f_inc(t)))``, ``γ = 1`` until the common start is ready and not above the start's
gap afterwards; time-to-target censored at the budget; the median over seeds by Kaplan–Meier.
Small step functions are integrated by hand, the Kaplan–Meier median is cross-checked against
``scipy.stats.ecdf``.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

from cif.metrics.anytime import (
    TAU_MAIN,
    TAU_SENSITIVITY,
    TimeToTarget,
    incumbent_curve,
    kaplan_meier,
    km_median,
    km_quantile,
    primal_integral,
    primal_integral_after_start,
    time_to_target,
)

# ---------------------------------------------------------------------------
# 1. incumbent_curve: the step function f_inc(t)
# ---------------------------------------------------------------------------


def test_incumbent_curve_keeps_only_improvements():
    traj = [(1.0, 5.0), (2.0, 6.0), (3.0, 4.0), (3.5, 4.0), (4.0, 3.0)]
    times, values = incumbent_curve(traj)
    np.testing.assert_array_equal(times, [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(values, [5.0, 4.0, 3.0])


def test_incumbent_curve_of_an_empty_trajectory_is_empty():
    times, values = incumbent_curve([])
    assert times.size == 0 and values.size == 0


def test_incumbent_curve_accepts_json_style_lists_and_numpy_scalars():
    times, values = incumbent_curve([[np.float64(1.0), 2.0], [2.0, np.float64(1.0)]])
    np.testing.assert_array_equal(times, [1.0, 2.0])
    np.testing.assert_array_equal(values, [2.0, 1.0])


def test_equal_times_take_the_lowest_value():
    times, values = incumbent_curve([(1.0, 5.0), (1.0, 3.0), (2.0, 4.0)])
    np.testing.assert_array_equal(times, [1.0])
    np.testing.assert_array_equal(values, [3.0])


def test_infinite_entries_are_no_solution_and_ignored():
    times, values = incumbent_curve([(0.5, math.inf), (1.0, 2.0)])
    np.testing.assert_array_equal(times, [1.0])
    np.testing.assert_array_equal(values, [2.0])


def test_the_common_start_is_the_first_incumbent():
    times, values = incumbent_curve([(2.0, 0.5), (4.0, 0.2)], start=(1.0, 0.8))
    np.testing.assert_array_equal(times, [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(values, [0.8, 0.5, 0.2])


def test_an_own_solution_worse_than_the_start_never_raises_the_incumbent():
    times, values = incumbent_curve([(2.0, 0.9), (3.0, 0.4)], start=(1.0, 0.8))
    np.testing.assert_array_equal(times, [1.0, 3.0])
    np.testing.assert_array_equal(values, [0.8, 0.4])


def test_own_solutions_before_the_start_count_from_the_moment_it_is_ready():
    # the incumbent exists only once the start is ready: the earlier find joins it there
    times, values = incumbent_curve([(0.5, 0.3), (2.0, 0.2)], start=(1.0, 0.8))
    np.testing.assert_array_equal(times, [1.0, 2.0])
    np.testing.assert_array_equal(values, [0.3, 0.2])


@pytest.mark.parametrize(
    "traj",
    [
        [(2.0, 1.0), (1.0, 0.5)],  # time goes backwards
        [(-1.0, 1.0)],  # negative time
        [(math.nan, 1.0)],
        [(1.0, math.nan)],
        [(math.inf, 1.0)],
        [(1.0, -math.inf)],
        [(1.0, 2.0, 3.0)],  # not pairs
    ],
)
def test_incumbent_curve_rejects_malformed_trajectories(traj):
    with pytest.raises(ValueError):
        incumbent_curve(traj)


def test_incumbent_curve_rejects_a_malformed_start():
    for start in [(math.nan, 1.0), (-1.0, 1.0), (1.0, math.nan), (1.0, -math.inf)]:
        with pytest.raises(ValueError):
            incumbent_curve([], start=start)


def test_incumbent_curve_does_not_modify_its_input():
    traj = [(1.0, 5.0), (2.0, 6.0)]
    snapshot = list(traj)
    incumbent_curve(traj, start=(0.5, 9.0))
    assert traj == snapshot


# ---------------------------------------------------------------------------
# 2. primal_integral
# ---------------------------------------------------------------------------

# f_ref = 0 and s_ref = 1 make ĝ(f) = f
REF = {"f_ref": 0.0, "s_ref": 1.0}


def test_primal_integral_by_hand_with_a_common_start():
    # γ: [0, 1) → 1 (start not ready), [1, 2) → 0.8, [2, 4) → 0.5, [4, 10] → 0.2
    pi = primal_integral(
        [(2.0, 0.5), (4.0, 0.2)], budget_s=10.0, start=(1.0, 0.8), **REF
    )
    assert pi == pytest.approx((1.0 + 0.8 + 2 * 0.5 + 6 * 0.2) / 10.0, rel=1e-14)


def test_primal_integral_without_a_start_is_one_until_the_first_own_solution():
    # sensitivity variant (§5): γ = 1 until the method's own first solution
    pi = primal_integral([(2.0, 0.5), (4.0, 0.2)], budget_s=10.0, **REF)
    assert pi == pytest.approx((2.0 * 1.0 + 2 * 0.5 + 6 * 0.2) / 10.0, rel=1e-14)


def test_a_common_start_can_only_lower_the_integral_of_the_same_own_trajectory():
    traj = [(2.0, 0.5), (4.0, 0.2)]
    own = primal_integral(traj, budget_s=10.0, **REF)
    for start in [(0.0, 0.9), (1.0, 0.8), (1.9, 0.6)]:
        assert primal_integral(traj, budget_s=10.0, start=start, **REF) <= own + 1e-15


def test_gamma_is_capped_at_one():
    # a start worse than the reference by more than the scale: γ = min(1, ĝ) = 1 on [t₀, 2)
    pi = primal_integral([(2.0, 0.5)], budget_s=4.0, start=(1.0, 3.0), **REF)
    assert pi == pytest.approx((1.0 + 1.0 + 2 * 0.5) / 4.0, rel=1e-14)


def test_gamma_does_not_go_below_zero_for_values_under_the_reference():
    # the reference is meant to be the best known value; a lower one must not give a negative area
    pi = primal_integral([(1.0, -0.3)], budget_s=2.0, start=(0.0, 0.6), **REF)
    assert pi == pytest.approx((1.0 * 0.6 + 1.0 * 0.0) / 2.0, rel=1e-14)


def test_events_after_the_budget_are_ignored():
    traj = [(2.0, 0.5), (4.0, 0.2)]
    pi = primal_integral(traj, budget_s=3.0, start=(1.0, 0.8), **REF)
    assert pi == pytest.approx((1.0 + 0.8 + 0.5) / 3.0, rel=1e-14)
    # an event exactly at the budget has zero width
    assert primal_integral(traj, budget_s=4.0, start=(1.0, 0.8), **REF) == pytest.approx(
        (1.0 + 0.8 + 2 * 0.5) / 4.0, rel=1e-14
    )


def test_no_incumbent_within_the_budget_gives_one():
    assert primal_integral([], budget_s=5.0, **REF) == 1.0
    assert primal_integral([(7.0, 0.0)], budget_s=5.0, start=(6.0, 0.0), **REF) == 1.0


def test_a_start_at_time_zero_has_no_idle_period():
    assert primal_integral([], budget_s=5.0, start=(0.0, 0.25), **REF) == pytest.approx(0.25)


def test_primal_integral_is_the_reference_value_when_the_incumbent_is_optimal_from_the_start():
    assert primal_integral([], budget_s=5.0, start=(0.0, 0.0), **REF) == 0.0


def test_primal_integral_is_invariant_to_the_objective_scale():
    traj = [(2.0, 0.5), (4.0, 0.2)]
    base = primal_integral(traj, budget_s=10.0, start=(1.0, 0.8), **REF)
    c, shift = 37.0, 0.0
    scaled = primal_integral(
        [(t, c * f + shift) for t, f in traj],
        budget_s=10.0,
        start=(1.0, c * 0.8 + shift),
        f_ref=c * 0.0 + shift,
        s_ref=c * 1.0,
    )
    assert scaled == pytest.approx(base, rel=1e-13)


def test_primal_integral_handles_a_nonzero_reference_and_scale():
    # ĝ = (f − f_ref)/s_ref with f_ref = −0.2, s_ref = 0.5
    pi = primal_integral(
        [(1.0, -0.1), (2.0, -0.2)], budget_s=4.0, start=(0.0, 0.1), f_ref=-0.2, s_ref=0.5
    )
    # γ: [0,1) → min(1, 0.3/0.5) = 0.6, [1,2) → 0.2, [2,4] → 0
    assert pi == pytest.approx((0.6 + 0.2 + 0.0) / 4.0, rel=1e-14)


def _riemann_pi(traj, start, budget, n=200_000):
    """Midpoint rule for γ(t) = min(1, max(0, running min of the events so far)), ĝ(f) = f."""
    events = list(traj)
    if start is not None:
        events = [(max(t, start[0]), f) for t, f in events] + [start]
    events.sort()
    times = np.array([t for t, _ in events])
    best = np.minimum.accumulate(np.array([f for _, f in events]))
    grid = (np.arange(n) + 0.5) * (budget / n)
    idx = np.searchsorted(times, grid, side="right") - 1
    gamma = np.ones(n)
    have = idx >= 0
    gamma[have] = np.clip(best[idx[have]], 0.0, 1.0)
    return float(gamma.mean())


def test_primal_integral_matches_a_fine_riemann_sum_on_random_trajectories():
    rng = np.random.default_rng(10001)
    budget = 20.0
    for _ in range(40):
        m = int(rng.integers(0, 8))
        times = np.sort(rng.uniform(0.0, 1.2 * budget, size=m))
        traj = list(zip(times.tolist(), rng.uniform(-0.1, 1.5, size=m).tolist(), strict=True))
        start = (float(rng.uniform(0.0, 5.0)), float(rng.uniform(0.0, 1.5)))
        for use_start in (False, True):
            chosen = start if use_start else None
            if not traj and chosen is None:
                continue
            pi = primal_integral(traj, budget_s=budget, start=chosen, **REF)
            assert pi == pytest.approx(_riemann_pi(traj, chosen, budget), abs=2e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_s": 0.0},
        {"budget_s": -1.0},
        {"budget_s": math.nan},
        {"budget_s": math.inf},
        {"budget_s": 1.0, "s_ref": 0.0},
        {"budget_s": 1.0, "s_ref": -1.0},
        {"budget_s": 1.0, "s_ref": math.nan},
        {"budget_s": 1.0, "f_ref": math.nan},
        {"budget_s": 1.0, "f_ref": math.inf},
    ],
)
def test_primal_integral_rejects_invalid_parameters(kwargs):
    params = {"f_ref": 0.0, "s_ref": 1.0, **kwargs}
    with pytest.raises(ValueError):
        primal_integral([(0.5, 0.1)], **params)


# ---------------------------------------------------------------------------
# 2b. PI⁺: the primal integral after the common start (§12, A2 of 2026-10-11)
# ---------------------------------------------------------------------------


def test_pi_plus_by_hand():
    # γ: [1, 2) → 0.8, [2, 4) → 0.5, [4, 10] → 0.2; nothing before the start is counted
    pi_plus = primal_integral_after_start(
        [(2.0, 0.5), (4.0, 0.2)], budget_s=10.0, start=(1.0, 0.8), **REF
    )
    assert pi_plus == pytest.approx((0.8 + 2 * 0.5 + 6 * 0.2) / 10.0, rel=1e-14)


def test_pi_plus_is_pi_minus_the_common_term_t0_over_t():
    rng = np.random.default_rng(20261011)
    budget = 20.0
    for _ in range(60):
        m = int(rng.integers(0, 8))
        times = np.sort(rng.uniform(0.0, 1.2 * budget, size=m))
        traj = list(zip(times.tolist(), rng.uniform(-0.1, 1.5, size=m).tolist(), strict=True))
        t0 = float(rng.uniform(0.0, 5.0))
        start = (t0, float(rng.uniform(0.0, 1.5)))
        pi = primal_integral(traj, budget_s=budget, start=start, **REF)
        pi_plus = primal_integral_after_start(traj, budget_s=budget, start=start, **REF)
        assert pi_plus == pytest.approx(pi - t0 / budget, abs=1e-13)


def test_pi_plus_is_zero_exactly_when_the_start_is_optimal():
    # the degenerate case of the consult: PI = t0/T for every method, PI⁺ = 0 for every method
    assert primal_integral_after_start([], budget_s=60.0, start=(0.05, 0.0), **REF) == 0.0
    assert primal_integral([], budget_s=60.0, start=(0.05, 0.0), **REF) == pytest.approx(0.05 / 60.0)


def test_pi_plus_is_positive_on_an_instance_where_the_start_leaves_a_gap():
    # a method that closes the gap of 2e-3 ten milliseconds after the start still has PI⁺ > 0
    pi_plus = primal_integral_after_start(
        [(0.19, 0.0)], budget_s=60.0, start=(0.18, 2e-3), **REF
    )
    assert pi_plus == pytest.approx(2e-3 * 0.01 / 60.0, rel=1e-9) and pi_plus > 0.0


def test_pi_plus_tells_apart_methods_that_pi_cannot():
    """The consult's example (A.1, item 3): at N = 400 the start takes t0 = 0.18 s, its gap is 2e-3, and
    one method reaches the optimum 1 s after the start, another 57 s after it. PI over [0, T] cannot go
    below 1/(1 + g·(T − t0)/t0) ≈ 0.60 as a ratio, however different the methods are; PI⁺ gives the
    factor that they differ by."""
    start, budget = (0.18, 2e-3), 60.0
    fast = [(1.18, 0.0)]
    slow = [(57.18, 0.0)]
    ratio_pi = primal_integral(fast, budget_s=budget, start=start, **REF) / primal_integral(
        slow, budget_s=budget, start=start, **REF
    )
    ratio_plus = primal_integral_after_start(
        fast, budget_s=budget, start=start, **REF
    ) / primal_integral_after_start(slow, budget_s=budget, start=start, **REF)
    floor = 1 / (1 + start[1] * (budget - start[0]) / start[0])
    assert floor == pytest.approx(0.6007, abs=1e-4)
    assert ratio_pi >= floor  # a 57-fold difference in the time to the optimum shows as a ratio of 0.62
    assert ratio_pi == pytest.approx(0.619, abs=1e-3)
    assert ratio_plus == pytest.approx(1.0 / 57.0, rel=1e-12)
    assert ratio_plus < floor / 30  # far outside any band around 1


def test_pi_plus_of_a_start_that_failed_counts_from_its_time():
    # the construction failed (f0 = +inf) at 0.5 s: γ = 1 from 0.5 s until the first own solution
    traj = [(2.0, 0.5)]
    pi_plus = primal_integral_after_start(traj, budget_s=10.0, start=(0.5, math.inf), **REF)
    assert pi_plus == pytest.approx((1.5 * 1.0 + 8.0 * 0.5) / 10.0, rel=1e-14)
    assert primal_integral_after_start([], budget_s=10.0, start=(0.5, math.inf), **REF) == pytest.approx(0.95)


def test_pi_plus_counts_own_solutions_timed_before_the_start_from_the_start():
    pi_plus = primal_integral_after_start([(0.2, 0.3)], budget_s=10.0, start=(0.5, 0.9), **REF)
    assert pi_plus == pytest.approx(0.3 * 9.5 / 10.0, rel=1e-14)


def test_pi_plus_is_zero_when_the_start_uses_the_whole_budget():
    assert primal_integral_after_start([], budget_s=1.0, start=(1.0, 0.7), **REF) == 0.0
    assert primal_integral_after_start([], budget_s=1.0, start=(2.5, 0.7), **REF) == 0.0


def test_pi_plus_caps_gamma_and_ignores_events_after_the_budget():
    pi_plus = primal_integral_after_start([(2.0, 0.5)], budget_s=4.0, start=(1.0, 3.0), **REF)
    assert pi_plus == pytest.approx((1.0 * 1.0 + 2 * 0.5) / 4.0, rel=1e-14)
    late = primal_integral_after_start([(2.0, 0.5), (9.0, 0.0)], budget_s=4.0, start=(1.0, 3.0), **REF)
    assert late == pytest.approx(pi_plus, rel=1e-14)


def test_pi_plus_needs_a_start_and_validates_like_the_integral():
    with pytest.raises(TypeError):
        primal_integral_after_start([(1.0, 0.1)], budget_s=5.0, **REF)  # type: ignore[call-arg]
    for kwargs in ({"budget_s": 0.0}, {"budget_s": math.nan}, {"s_ref": 0.0}, {"f_ref": math.inf}):
        params = {"f_ref": 0.0, "s_ref": 1.0, "budget_s": 1.0, "start": (0.0, 0.5), **kwargs}
        with pytest.raises(ValueError):
            primal_integral_after_start([(0.5, 0.1)], **params)
    with pytest.raises(ValueError, match="start"):
        primal_integral_after_start([(0.5, 0.1)], budget_s=1.0, start=(-1.0, 0.5), **REF)


# ---------------------------------------------------------------------------
# 3. time_to_target
# ---------------------------------------------------------------------------


def test_time_to_target_is_the_first_time_the_gap_is_within_tau():
    traj = [(2.0, 0.5), (4.0, 0.2)]
    res = time_to_target(traj, tau=0.3, budget_s=10.0, start=(1.0, 0.8), **REF)
    assert res == TimeToTarget(time_s=4.0, reached=True)
    res = time_to_target(traj, tau=0.5, budget_s=10.0, start=(1.0, 0.8), **REF)
    assert res.time_s == 2.0 and res.reached  # the boundary ĝ = τ counts as reached


def test_a_run_that_does_not_reach_the_target_is_censored_at_the_budget():
    traj = [(2.0, 0.5), (4.0, 0.2)]
    res = time_to_target(traj, tau=0.1, budget_s=10.0, start=(1.0, 0.8), **REF)
    assert res == TimeToTarget(time_s=10.0, reached=False)
    # reached, but only after the budget: still censored
    res = time_to_target(traj, tau=0.3, budget_s=3.0, start=(1.0, 0.8), **REF)
    assert res == TimeToTarget(time_s=3.0, reached=False)
    # reached exactly at the budget counts
    res = time_to_target(traj, tau=0.3, budget_s=4.0, start=(1.0, 0.8), **REF)
    assert res == TimeToTarget(time_s=4.0, reached=True)


def test_the_common_start_can_reach_the_target_itself():
    res = time_to_target([(5.0, 0.0)], tau=0.5, budget_s=10.0, start=(1.5, 0.4), **REF)
    assert res == TimeToTarget(time_s=1.5, reached=True)


def test_without_a_start_only_own_solutions_count():
    res = time_to_target([(5.0, 0.0)], tau=0.5, budget_s=10.0, **REF)
    assert res == TimeToTarget(time_s=5.0, reached=True)
    res = time_to_target([], tau=0.5, budget_s=10.0, **REF)
    assert res == TimeToTarget(time_s=10.0, reached=False)


def test_time_to_target_uses_the_normalised_gap():
    # ĝ = (f − f_ref)/s_ref = (0.1 − (−0.2))/0.5 = 0.6 at t = 1, 0.2 at t = 3
    traj = [(1.0, 0.1), (3.0, -0.1)]
    res = time_to_target(traj, tau=0.3, budget_s=5.0, f_ref=-0.2, s_ref=0.5)
    assert res == TimeToTarget(time_s=3.0, reached=True)


def test_registered_targets():
    assert TAU_MAIN == 1e-3 and TAU_SENSITIVITY == 1e-4  # §5: main and sensitivity


@pytest.mark.parametrize("tau", [-1e-3, math.nan, math.inf])
def test_time_to_target_rejects_an_invalid_tau(tau):
    with pytest.raises(ValueError, match="tau"):
        time_to_target([(1.0, 0.0)], tau=tau, budget_s=5.0, **REF)


# ---------------------------------------------------------------------------
# 4. Kaplan–Meier
# ---------------------------------------------------------------------------


def test_kaplan_meier_curve_by_hand():
    # n = 5; events at 1, 3, 5; censored at 2 and 4
    t, s = kaplan_meier([1.0, 2.0, 3.0, 4.0, 5.0], [True, False, True, False, True])
    np.testing.assert_array_equal(t, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(s, [4 / 5, 4 / 5 * 2 / 3, 0.0], rtol=1e-14)


def test_kaplan_meier_without_censoring_is_the_empirical_survival_function():
    t, s = kaplan_meier([3.0, 1.0, 2.0, 2.0], [True] * 4)
    np.testing.assert_array_equal(t, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(s, [0.75, 0.25, 0.0], rtol=1e-14)


def test_an_event_and_a_censoring_at_the_same_time_keep_the_censored_run_at_risk():
    # at t = 2: one event and one censoring; the censored run is still at risk at 2
    t, s = kaplan_meier([2.0, 2.0, 5.0], [True, False, True])
    np.testing.assert_array_equal(t, [2.0, 5.0])
    np.testing.assert_allclose(s, [2 / 3, 0.0], rtol=1e-14)


def test_median_of_uncensored_data_is_the_lower_median():
    # smallest t with S(t) ≤ 1/2: with n = 10 that is the 5th order statistic
    times = np.arange(1.0, 11.0)
    assert km_median(times, np.ones(10, dtype=bool)) == 5.0
    assert km_median(times[:9], np.ones(9, dtype=bool)) == 5.0  # S(5) = 4/9 < 1/2, S(4) = 5/9


def test_median_is_infinite_when_the_curve_never_reaches_one_half():
    # ten runs, six censored at the budget: only 4 / 10 events, S stays at 0.6
    times = np.array([10.0, 20.0, 30.0, 40.0] + [60.0] * 6)
    observed = np.array([True] * 4 + [False] * 6)
    assert km_median(times, observed) == math.inf
    # five of ten reached the target: S drops to exactly 1/2
    times = np.array([10.0, 20.0, 30.0, 40.0, 50.0] + [60.0] * 5)
    observed = np.array([True] * 5 + [False] * 5)
    assert km_median(times, observed) == 50.0


def test_median_with_censoring_before_the_half_point():
    # n = 7: events at 2, 3, 5, 7; censored at 4, 6, 6 → S(5) = 0.5357, S(7) = 0
    assert km_median([2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0], [1, 1, 0, 1, 0, 0, 1]) == 7.0


def test_all_censored_has_no_median():
    assert km_median([5.0, 5.0, 5.0], [False, False, False]) == math.inf


def test_km_quantiles_are_the_smallest_times_where_the_survival_falls_to_one_minus_q():
    # events at 1, 2, 4 and one run censored at the budget: S = 3/4, 1/2, 1/4
    times, observed = [2.0, 4.0, 60.0, 1.0], [True, True, False, True]
    assert [km_quantile(times, observed, q) for q in (0.25, 0.5, 0.75)] == [1.0, 2.0, 4.0]
    assert km_quantile([5.0, 60.0, 60.0, 60.0], [True, False, False, False], 0.5) == math.inf
    assert km_quantile([5.0, 60.0, 60.0, 60.0], [True, False, False, False], 0.25) == 5.0
    rng = np.random.default_rng(7)
    t, d = rng.exponential(10.0, 50), rng.random(50) < 0.7
    assert km_quantile(t, d, 0.5) == km_median(t, d)
    for q in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="q"):
            km_quantile(t, d, q)
    t, s = kaplan_meier([5.0, 5.0, 5.0], [False, False, False])
    assert t.size == 0 and s.size == 0


def test_median_is_exact_when_the_survival_hits_one_half_in_floating_point():
    # n = 10, ten events at 1..10: S(5) = (9/10)(8/9)(7/8)(6/7)(5/6) = 0.5 up to rounding
    assert km_median(np.arange(1.0, 11.0), np.ones(10, dtype=bool)) == 5.0
    assert km_median(np.arange(1.0, 7.0), np.ones(6, dtype=bool)) == 3.0


def test_median_matches_scipy_ecdf_on_random_censored_samples():
    rng = np.random.default_rng(10002)
    for _ in range(200):
        n = int(rng.integers(3, 25))
        times = np.round(rng.uniform(0.0, 10.0, size=n), 1)  # rounding creates ties
        observed = rng.random(n) < 0.6
        res = stats.ecdf(stats.CensoredData(uncensored=times[observed], right=times[~observed]))
        probs = res.sf.probabilities
        expected = (
            float(res.sf.quantiles[np.argmax(probs <= 0.5 + 1e-12)])
            if np.any(probs <= 0.5 + 1e-12)
            else math.inf
        )
        assert km_median(times, observed) == expected


def test_kaplan_meier_curve_matches_scipy_ecdf_at_the_event_times():
    rng = np.random.default_rng(10003)
    for _ in range(50):
        n = int(rng.integers(3, 25))
        times = np.round(rng.uniform(0.0, 10.0, size=n), 1)
        observed = rng.random(n) < 0.6
        if not observed.any():
            continue
        res = stats.ecdf(stats.CensoredData(uncensored=times[observed], right=times[~observed]))
        t, s = kaplan_meier(times, observed)
        np.testing.assert_array_equal(t, np.unique(times[observed]))
        for tk, sk in zip(t, s, strict=True):
            at = np.flatnonzero(res.sf.quantiles == tk)[0]
            assert sk == pytest.approx(res.sf.probabilities[at], abs=1e-12)


def test_kaplan_meier_accepts_integer_flags_and_rejects_bad_input():
    assert km_median([1.0, 2.0, 3.0], [1, 1, 1]) == 2.0
    with pytest.raises(ValueError):
        km_median([], [])
    with pytest.raises(ValueError):
        km_median([1.0, 2.0], [True])
    with pytest.raises(ValueError):
        km_median([1.0, math.nan], [True, True])
    with pytest.raises(ValueError):
        km_median([-1.0, 2.0], [True, True])


def test_a_larger_censoring_budget_never_makes_the_median_later():
    # sensitivity to the censoring budget (§5): the same ten runs, four budgets. Eight runs reach
    # the target at these times, two never do.
    reached = np.array([4.0, 9.0, 15.0, 20.0, 33.0, 45.0, 52.0, 58.0, math.inf, math.inf])
    medians = []
    for budget in (20.0, 30.0, 40.0, 60.0):
        observed = reached <= budget
        times = np.where(observed, reached, budget)
        medians.append(km_median(times, observed))
    # 4 of 10 by 20 s and by 30 s (S stays at 0.6), the fifth at 33 s
    assert medians == [math.inf, math.inf, 33.0, 33.0]
    assert all(a >= b for a, b in pairwise(medians))
