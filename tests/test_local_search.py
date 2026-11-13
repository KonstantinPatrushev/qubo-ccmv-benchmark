"""Local search on the true objective (ANALYSIS_PLAN §2 RQ2/H2b, §4, §6).

The search logic (first / best improvement, restarts, memo, budget, trajectory) is checked on
synthetic landscapes, where the shared refinement is replaced by a table, so that every expected
path is known; the real refinement is used for the properties that need it (local optimality,
agreement with enumeration, the common start, the relaxed box of the greedy construction). Every
search that is not ended by a clock has a finite number of restarts, so no test can run away.
Instance seeds are calibration ones (1000–1999), solver seeds 10000–19999.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from itertools import combinations, pairwise

import numpy as np
import pytest

from cif.classical.refine import RefineResult, refine_problem, refine_support
from cif.heuristics import local_search as ls
from cif.heuristics.local_search import greedy_construction, local_search
from cif.heuristics.start import CommonStart, common_start
from cif.instances.factor import factor_instance
from cif.metrics.gap import reference_scale
from cif.problem import PortfolioProblem


def _problem(
    n: int = 14, seed: int = 1000, K: int = 4, *, w_min: float = 0.0, w_max: float = 1.0
) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"a{i}" for i in range(n)),
        risk_aversion=5.0,
        w_min=w_min,
        w_max=w_max,
        cardinality=K,
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Landscape:
    """A table in place of the shared refinement: ``value(S)`` for every support ``S``.

    Records every call (the sorted support) and spends ``cost_s`` of fake time on it.
    """

    def __init__(self, value, clock=None, cost_s=0.0):
        self.value = value
        self.clock = clock
        self.cost_s = cost_s
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, mu, sigma, risk_aversion, support, lower=0.0, upper=1.0, budget=1.0):
        assert len(self.calls) < 50_000, "runaway search"
        subset = tuple(sorted(int(i) for i in support))
        self.calls.append(subset)
        if self.clock is not None:
            self.clock.advance(self.cost_s)
        weights = np.zeros(len(mu))
        weights[list(subset)] = budget / len(subset)
        return RefineResult(weights, float(self.value(subset)), "optimal", 0.0)


def _separable(costs):
    costs = np.asarray(costs, dtype=float)
    return lambda subset: float(costs[list(subset)].sum())


def _random_table(salt=0):
    """A reproducible random value for every subset: a landscape with many local optima."""
    return lambda subset: float(np.random.default_rng([salt, *subset]).random())


def _install(monkeypatch, landscape):
    monkeypatch.setattr(ls, "refine_support", landscape)
    return landscape


def _start_at(problem, subset, value, time_s=0.0) -> CommonStart:
    """A common start at ``subset`` with the given objective (for a fake landscape)."""
    weights = np.zeros(problem.n)
    weights[list(subset)] = 1.0 / len(subset)
    return dataclasses.replace(
        common_start(problem),
        weights=weights,
        objective=float(value),
        subset=tuple(sorted(subset)),
        time_s=float(time_s),
    )


# ---------------------------------------------------------------------------
# 1. Search mechanics on synthetic landscapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("improvement", ["first", "best"])
def test_descent_reaches_the_best_subset_of_a_separable_landscape(monkeypatch, improvement):
    n, K = 9, 3
    problem = _problem(n, 1000, K)
    costs = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4])
    _install(monkeypatch, Landscape(_separable(costs)))
    start = _start_at(problem, (5, 7, 1), costs[[5, 7, 1]].sum())  # the three dearest assets
    sol = local_search(
        problem, K, budget_s=1e9, seed=10001, improvement=improvement, num_restarts=1, start=start
    )
    meta = sol.solver_meta
    assert meta["subset"] == [2, 4, 6]  # the K cheapest
    assert sol.objective == pytest.approx(0.1 + 0.2 + 0.3)
    assert meta["stop"] == "restarts" and meta["restarts_done"] == 1
    assert meta["converged_descents"] == 1 and meta["improvement"] == improvement


def test_best_improvement_takes_the_best_swap_of_each_neighbourhood(monkeypatch):
    n, K = 8, 3
    problem = _problem(n, 1000, K)
    costs = np.array([0.50, 0.90, 0.10, 0.70, 0.30, 0.80, 0.20, 0.60])
    _install(monkeypatch, Landscape(_separable(costs)))
    start = _start_at(problem, (1, 5, 3), 0.9 + 0.8 + 0.7)
    sol = local_search(
        problem, K, budget_s=1e9, seed=10002, improvement="best", num_restarts=1, start=start
    )
    values = [f for _, f in sol.solver_meta["trajectory"]]
    # each step trades the dearest held asset for the cheapest one outside: 0.9 → 0.1, 0.8 → 0.2,
    # 0.7 → 0.3, and nothing else is accepted
    assert values == pytest.approx([2.4, 2.4 - 0.8, 2.4 - 0.8 - 0.6, 2.4 - 0.8 - 0.6 - 0.4])
    assert sol.solver_meta["n_swaps"] == 3


def test_first_improvement_replays_from_the_evaluation_log(monkeypatch):
    # the algorithm accepts the first evaluated neighbour that improves: replaying the log with
    # that rule visits only neighbours of the current subset and ends at the returned subset
    n, K = 10, 3
    problem = _problem(n, 1001, K)
    table = _random_table(salt=7)
    landscape = _install(monkeypatch, Landscape(table))
    subset = common_start(problem).subset
    start = _start_at(problem, subset, table(subset))
    sol = local_search(
        problem, K, budget_s=1e9, seed=10003, improvement="first", num_restarts=1, start=start
    )
    current, f_current = start.subset, start.objective
    for call in landscape.calls:
        assert len(set(call) ^ set(current)) == 2, "not a neighbour of the current subset"
        if table(call) < f_current - 1e-12:
            current, f_current = call, table(call)
    assert list(current) == sol.solver_meta["subset"]
    assert sol.solver_meta["converged_descents"] == 1


def test_no_subset_is_refined_twice(monkeypatch):
    n, K = 9, 3
    problem = _problem(n, 1002, K)
    landscape = _install(monkeypatch, Landscape(_random_table(salt=3)))
    local_search(problem, K, budget_s=1e9, seed=10004, num_restarts=5, rcl_size=3)
    assert len(set(landscape.calls)) == len(landscape.calls)


def test_the_search_ends_once_every_subset_has_been_evaluated(monkeypatch):
    n, K = 5, 1  # five single-asset portfolios; one scan of the neighbourhood sees them all
    problem = _problem(n, 1003, K)
    table = _random_table(salt=11)
    _install(monkeypatch, Landscape(table))
    sol = local_search(problem, K, budget_s=1e9, seed=10005, improvement="best")
    meta = sol.solver_meta
    assert meta["stop"] == "exhausted" and meta["n_subsets_evaluated"] == n
    assert sol.objective == pytest.approx(min(table((i,)) for i in range(n)))


def test_restarts_are_counted_and_end_the_search(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1004, K)
    _install(monkeypatch, Landscape(_random_table(salt=5)))
    for restarts in (1, 3):
        sol = local_search(problem, K, budget_s=1e9, seed=10006, num_restarts=restarts)
        meta = sol.solver_meta
        assert meta["restarts_done"] == restarts and meta["stop"] == "restarts"
        assert meta["converged_descents"] == restarts


def test_the_budget_ends_the_search_and_no_refinement_starts_after_it(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1005, K)
    clock = FakeClock()
    landscape = _install(monkeypatch, Landscape(_random_table(salt=2), clock=clock, cost_s=1.0))
    sol = local_search(problem, K, budget_s=10.5, seed=10007, clock=clock)
    assert len(landscape.calls) == 11  # started at 0, 1, ..., 10 s, all before 10.5 s
    meta = sol.solver_meta
    assert meta["stop"] == "budget" and meta["n_evals"] == 11
    assert sol.wall_time_s == pytest.approx(11.0)
    # the greedy construction (12 + 11 + 10 + 9 refinements) does not end within the budget: the search holds no
    # portfolio of its own (audit G2, C-4), and all the fake time went into refinements
    assert meta["first_own_s"] is None and meta["first_own_f"] is None and meta["trajectory"] == []
    assert meta["refine_s"] == pytest.approx(11.0) and meta["overhead_s"] == pytest.approx(0.0)


def test_a_cold_run_has_its_first_own_solution_when_the_greedy_construction_ends(monkeypatch):
    """Audit G2, C-4 (§12 of 2026-10-17): without a common start the first own solution of local search is the first
    portfolio it holds, the end of its greedy construction (the first point of its trajectory), not the refinement of
    the first partial subset of the construction."""
    n, K = 12, 4
    problem = _problem(n, 1005, K)
    clock = FakeClock()
    _install(monkeypatch, Landscape(_random_table(salt=2), clock=clock, cost_s=1.0))
    sol = local_search(problem, K, budget_s=1e9, seed=10007, clock=clock, num_restarts=1)
    meta = sol.solver_meta
    construction = n + (n - 1) + (n - 2) + (n - 3)  # K rounds over the assets not yet taken, 1 s a refinement
    assert meta["trajectory"][0][0] == pytest.approx(construction)
    assert (meta["first_own_s"], meta["first_own_f"]) == tuple(meta["trajectory"][0])


def test_the_value_of_the_first_own_refinement_is_recorded_next_to_its_time(monkeypatch):
    """§12 A4: from a common start, the first own solution is the first K-subset the descent refines, with its value,
    whether or not it beats the start."""
    problem = _problem(12, 1005, 4)
    table = _random_table(salt=2)
    clock = FakeClock()
    landscape = _install(monkeypatch, Landscape(table, clock=clock, cost_s=1.0))
    start = _start_at(problem, (0, 1, 2, 3), -100.0, time_s=0.25)
    sol = local_search(problem, 4, budget_s=6.0, seed=10007, start=start, clock=clock, num_restarts=1)
    assert len(landscape.calls[0]) == 4 and sol.solver_meta["first_own_s"] == pytest.approx(1.25)
    assert sol.solver_meta["first_own_f"] == pytest.approx(table(landscape.calls[0]))
    assert sol.solver_meta["first_own_f"] > -100.0 and sol.objective == -100.0


def test_a_refinement_that_would_start_exactly_at_the_budget_is_not_run(monkeypatch):
    problem = _problem(12, 1031, 4)
    clock = FakeClock()
    landscape = _install(monkeypatch, Landscape(_random_table(salt=37), clock=clock, cost_s=1.0))
    sol = local_search(problem, 4, budget_s=10.0, seed=10051, clock=clock)
    assert len(landscape.calls) == 10 and sol.wall_time_s == pytest.approx(10.0)


def test_the_construction_time_of_the_start_is_spent_before_the_search(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1006, K)
    clock = FakeClock()
    table = _random_table(salt=9)
    landscape = _install(monkeypatch, Landscape(table, clock=clock, cost_s=1.0))
    start = _start_at(problem, (0, 1, 2, 3), table((0, 1, 2, 3)), time_s=7.5)
    sol = local_search(problem, K, budget_s=10.0, seed=10008, start=start, clock=clock)
    assert len(landscape.calls) == 3  # at 7.5, 8.5 and 9.5 s of the common clock
    assert sol.solver_meta["trajectory"][0] == (7.5, start.objective)
    # the start's 7.5 s are not the search's overhead: all 3 s of the loop were refinements
    assert sol.solver_meta["refine_s"] == pytest.approx(3.0)
    assert sol.solver_meta["overhead_s"] == pytest.approx(0.0)


def test_the_result_owns_its_weights_when_the_start_stays_the_best(monkeypatch):
    n, K = 9, 3
    problem = _problem(n, 1030, K)
    costs = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    _install(monkeypatch, Landscape(_separable(costs)))
    start = _start_at(problem, (0, 1, 2), 0.6)  # already the cheapest triple
    sol = local_search(problem, K, budget_s=1e9, seed=10050, num_restarts=1, start=start)
    assert sol.solver_meta["n_swaps"] == 0
    np.testing.assert_array_equal(sol.weights, start.weights)
    assert not np.shares_memory(sol.weights, start.weights)


def test_the_start_is_not_refined_again_and_the_search_begins_next_to_it(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1007, K)
    table = _random_table(salt=13)
    landscape = _install(monkeypatch, Landscape(table))
    start = _start_at(problem, (2, 3, 8, 9), table((2, 3, 8, 9)))
    local_search(problem, K, budget_s=1e9, seed=10009, num_restarts=1, start=start)
    assert start.subset not in landscape.calls
    assert len(set(landscape.calls[0]) ^ set(start.subset)) == 2


def test_the_trajectory_records_every_improvement_of_the_incumbent_only(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1008, K)
    table = _random_table(salt=17)
    clock = FakeClock()
    _install(monkeypatch, Landscape(table, clock=clock, cost_s=0.5))
    sol = local_search(
        problem, K, budget_s=40.0, seed=10010, num_restarts=6, improvement="first", clock=clock
    )
    traj = sol.solver_meta["trajectory"]
    times = [t for t, _ in traj]
    values = [f for _, f in traj]
    assert times == sorted(times) and all(b < a for a, b in pairwise(values))
    assert values[-1] == sol.objective
    held = tuple(sol.solver_meta["subset"])
    assert table(held) == sol.objective
    assert all(0.0 <= t <= sol.wall_time_s for t in times)
    json.dumps(sol.solver_meta)


def test_runs_are_reproducible_for_a_seed_and_depend_on_it(monkeypatch):
    n, K = 14, 4
    problem = _problem(n, 1009, K)
    table = _random_table(salt=19)

    def run(seed):
        landscape = _install(monkeypatch, Landscape(table))
        sol = local_search(problem, K, budget_s=1e9, seed=seed, num_restarts=5, rcl_size=4)
        return landscape.calls, [f for _, f in sol.solver_meta["trajectory"]], sol.objective

    first = run(10011)
    assert first == run(10011)
    assert run(10012)[0] != first[0]  # a different seed explores differently


def test_the_greedy_result_is_offered_to_the_incumbent(monkeypatch):
    n, K = 8, 2
    problem = _problem(n, 1024, K)
    costs = np.array([0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.2, 0.6])
    _install(monkeypatch, Landscape(_separable(costs)))
    sol = local_search(problem, K, budget_s=1e9, seed=10044, num_restarts=1)
    # the deterministic greedy construction already holds the two cheapest assets: no swap is
    # needed, and the incumbent must still be the construction itself
    assert sol.solver_meta["subset"] == [2, 6] and sol.objective == pytest.approx(0.3)
    assert sol.solver_meta["n_swaps"] == 0 and len(sol.solver_meta["trajectory"]) == 1


def test_equal_values_do_not_replace_the_incumbent_or_add_trajectory_points(monkeypatch):
    n, K = 8, 2
    problem = _problem(n, 1025, K)
    costs = np.array([0.1, 0.1, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9])  # {0,1}, {0,2}, {1,2} tie at 0.2
    _install(monkeypatch, Landscape(_separable(costs)))
    sol = local_search(problem, K, budget_s=1e9, seed=10045, num_restarts=8, rcl_size=4)
    values = [f for _, f in sol.solver_meta["trajectory"]]
    assert all(b < a for a, b in pairwise(values))
    assert sol.objective == pytest.approx(0.2)
    assert sol.solver_meta["subset"] == [0, 1]  # greedy ties go to the lower index: found first


@pytest.mark.parametrize("improvement", ["first", "best"])
def test_a_gain_below_the_tolerance_is_not_a_swap(monkeypatch, improvement):
    n, K = 8, 3
    problem = _problem(n, 1026, K)
    home = (0, 1, 2)
    start = _start_at(problem, home, 0.5)
    for gain, expected_swaps in ((1e-12, 0), (1e-6, 1)):
        _install(monkeypatch, Landscape(lambda s, gain=gain: 0.5 if s == home else 0.5 - gain))
        sol = local_search(
            problem, K, budget_s=1e9, seed=10046, num_restarts=1, start=start,
            improvement=improvement,
        )
        # with a gain of 1e-6 the first improving neighbour is taken, and from there nothing is
        # better (all others are equal to it): exactly one swap
        assert sol.solver_meta["n_swaps"] == expected_swaps
        assert sol.solver_meta["converged_descents"] == 1


def test_restarts_draw_different_random_streams(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1027, K)

    def evaluations(restarts):
        landscape = _install(monkeypatch, Landscape(_random_table(salt=29)))
        local_search(problem, K, budget_s=1e9, seed=10047, num_restarts=restarts, rcl_size=4)
        return len(landscape.calls)

    # a restart that repeated an earlier greedy path would evaluate nothing new
    assert evaluations(2) < evaluations(4) < evaluations(8)


def test_with_a_start_the_first_descent_precedes_any_greedy_construction(monkeypatch):
    n, K = 12, 4
    problem = _problem(n, 1028, K)
    table = _random_table(salt=31)
    landscape = _install(monkeypatch, Landscape(table))
    start = _start_at(problem, (0, 1, 2, 3), table((0, 1, 2, 3)))
    local_search(problem, K, budget_s=1e9, seed=10048, num_restarts=3, rcl_size=3, start=start)
    sizes = [len(c) for c in landscape.calls]
    assert sizes[0] == K  # a neighbour of the start
    first_partial = next(i for i, size in enumerate(sizes) if size < K)
    assert first_partial > 0  # the greedy construction of restart 1 comes after the descent


def test_an_infeasible_cardinality_gives_an_infinite_value_not_a_lifted_box():
    problem = _problem(10, 1029, 3, w_max=0.2)  # 3 · 0.2 < 1: no K-subset reaches the budget
    sol = local_search(problem, 3, budget_s=1e9, seed=10049, num_restarts=1)
    assert sol.objective == math.inf and sol.feasible is False
    assert sol.solver_meta["subset"] is None and sol.solver_meta["trajectory"] == []


def test_bookkeeping_counts_fresh_evaluations_and_cache_hits(monkeypatch):
    n, K = 9, 3
    problem = _problem(n, 1010, K)
    landscape = _install(monkeypatch, Landscape(_random_table(salt=23)))
    sol = local_search(problem, K, budget_s=1e9, seed=10013, num_restarts=4)
    meta = sol.solver_meta
    assert meta["n_evals"] == len(landscape.calls)
    assert meta["n_cache_hits"] > 0  # restarts and neighbourhoods overlap
    assert meta["refine_s"] >= 0.0 and meta["overhead_s"] >= -1e-6
    assert meta["first_own_s"] is not None
    assert meta["n_subsets_evaluated"] == len({s for s in landscape.calls if len(s) == K})


# ---------------------------------------------------------------------------
# 2. Greedy construction
# ---------------------------------------------------------------------------


def _greedy_value(problem, subset, K, card_mode="le"):
    """The value the construction ranks by: the refinement, with the box lifted while the partial
    support is too small to reach the budget (and only then)."""
    upper = problem.w_max
    if len(subset) < K and len(subset) * upper < problem.budget - 1e-12:
        upper = problem.budget
    lower = problem.w_min if card_mode == "eq" else 0.0
    return refine_support(
        problem.mu, problem.sigma, problem.risk_aversion, subset, lower=lower, upper=upper,
        budget=problem.budget,
    ).objective


def _expected_greedy(problem, K, card_mode="le"):
    chosen: list[int] = []
    for _ in range(K):
        best = min(
            (j for j in range(problem.n) if j not in chosen),
            key=lambda j: (_greedy_value(problem, tuple(sorted([*chosen, j])), K, card_mode), j),
        )
        chosen.append(best)
    return tuple(sorted(chosen))


def test_greedy_adds_the_asset_with_the_best_refined_value_at_every_step():
    problem = _problem(10, 1011, 4)
    assert greedy_construction(problem, 4) == _expected_greedy(problem, 4)


def test_greedy_is_deterministic_without_a_candidate_list_and_random_with_one():
    problem = _problem(14, 1012, 4)
    assert greedy_construction(problem, 4) == greedy_construction(problem, 4)
    same = [
        greedy_construction(problem, 4, rng=np.random.default_rng(10014), rcl_size=3)
        for _ in range(2)
    ]
    assert same[0] == same[1]
    picks = {
        greedy_construction(problem, 4, rng=np.random.default_rng(s), rcl_size=3)
        for s in range(10015, 10035)
    }
    assert len(picks) > 1  # the randomisation really diversifies the restarts
    assert greedy_construction(problem, 4) in picks or len(picks) > 3


def test_a_candidate_list_needs_an_explicit_generator():
    with pytest.raises(ValueError, match="rng"):
        greedy_construction(_problem(10, 1012, 3), 3, rcl_size=3)


def test_greedy_returns_a_k_subset_with_a_finite_value_under_a_binding_box():
    # u = 0.2: supports of fewer than 5 assets cannot reach the budget, yet the construction works
    problem = _problem(20, 1013, 7, w_max=0.2)
    subset = greedy_construction(problem, 7)
    assert len(subset) == 7
    assert refine_problem(problem, subset).objective < math.inf
    # without the lifted box the first four picks would all be infeasible and go by index
    assert subset == _expected_greedy(problem, 7)
    assert subset != (0, 1, 2, 3, *subset[4:])


def test_greedy_in_eq_mode_ranks_by_the_all_held_refinement():
    problem = _problem(12, 1014, 4, w_min=0.02)
    assert greedy_construction(problem, 4, card_mode="eq") == _expected_greedy(problem, 4, "eq")


def test_greedy_for_a_k_that_cannot_reach_the_budget_has_no_feasible_result():
    problem = _problem(10, 1015, 3, w_max=0.2)  # 3 · 0.2 < 1
    subset = greedy_construction(problem, 3)
    assert len(subset) == 3
    assert refine_problem(problem, subset).objective == math.inf


# ---------------------------------------------------------------------------
# 3. With the real shared refinement
# ---------------------------------------------------------------------------


def _all_swaps(subset, n):
    return [
        tuple(sorted((set(subset) - {i}) | {j}))
        for i in subset
        for j in range(n)
        if j not in subset
    ]


@pytest.mark.parametrize("improvement", ["first", "best"])
def test_the_result_of_a_converged_descent_is_a_local_optimum(improvement):
    problem = _problem(14, 1016, 4)
    start = common_start(problem)
    sol = local_search(
        problem, 4, budget_s=1e9, seed=10036, improvement=improvement, num_restarts=1, start=start
    )
    assert sol.solver_meta["converged_descents"] == 1
    subset = tuple(sol.solver_meta["subset"])
    f = refine_problem(problem, subset).objective
    s = reference_scale(problem, sol.weights)
    assert sol.objective == pytest.approx(f, abs=1e-12)
    for neighbour in _all_swaps(subset, problem.n):
        assert refine_problem(problem, neighbour).objective >= f - 1e-8 * s


def test_restarts_find_the_enumerated_optimum_on_a_small_instance():
    problem = _problem(12, 1017, 4)
    best = min(refine_problem(problem, s).objective for s in combinations(range(12), 4))
    for improvement in ("first", "best"):
        sol = local_search(
            problem, 4, budget_s=120.0, seed=10037, improvement=improvement, num_restarts=40
        )
        assert sol.objective == pytest.approx(best, abs=1e-12)


def test_the_common_start_is_the_first_incumbent_and_never_worsened():
    problem = _problem(40, 1018, 8)
    start = common_start(problem)
    sol = local_search(problem, 8, budget_s=1.0 + start.time_s, seed=10038, start=start)
    traj = sol.solver_meta["trajectory"]
    assert traj[0] == (start.time_s, start.objective)
    assert sol.objective <= start.objective + 1e-14
    assert sol.solver_meta["start_used"] is True
    assert sol.feasible and sol.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.count_nonzero(sol.weights > 1e-9) <= 8
    assert sol.objective == pytest.approx(problem.objective_value(sol.weights), abs=1e-12)


def test_a_wall_clock_run_stops_in_time():
    problem = _problem(60, 1019, 10)
    t0 = time.perf_counter()
    sol = local_search(problem, 10, budget_s=0.5, seed=10039)  # no start: greedy first
    assert time.perf_counter() - t0 < 0.5 + 1.0
    assert sol.solver_meta["stop"] == "budget"
    if math.isfinite(sol.objective):  # the greedy pass may not finish in half a second
        assert sol.feasible


def test_eq_mode_keeps_all_assets_above_the_minimum_purchase():
    problem = _problem(14, 1020, 5, w_min=0.15)  # 5 · 0.15 = 0.75: the bound really binds
    start = common_start(problem, card_mode="eq")
    sol = local_search(
        problem, 5, budget_s=1e9, seed=10040, num_restarts=1, start=start, card_mode="eq"
    )
    held = sol.weights[list(sol.solver_meta["subset"])]
    assert np.all(held >= 0.15 - 1e-9) and np.count_nonzero(sol.weights) == 5
    assert np.any(held < 0.15 + 1e-6)  # at least one weight sits on its bound
    assert sol.objective == pytest.approx(
        refine_problem(problem, sol.solver_meta["subset"], card_mode="eq").objective, abs=1e-12
    )
    assert sol.solver_meta["card_mode"] == "eq"


def test_the_neighbourhood_is_empty_when_k_equals_n():
    problem = _problem(6, 1021, 6)
    sol = local_search(problem, 6, budget_s=1e9, seed=10041, num_restarts=1)
    assert sol.solver_meta["subset"] == list(range(6))
    assert sol.solver_meta["converged_descents"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_s": 0.0},
        {"budget_s": math.nan},
        {"budget_s": math.inf},
        {"seed": -1},
        {"improvement": "steepest"},
        {"num_restarts": 0},
        {"rcl_size": 0},
        {"card_mode": "ge"},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    params = {"budget_s": 1.0, "seed": 10042, **kwargs}
    with pytest.raises(ValueError):
        local_search(_problem(10, 1022, 3), 3, **params)


def test_le_mode_with_a_minimum_purchase_is_rejected():
    with pytest.raises(ValueError, match="w_min"):
        local_search(_problem(10, 1022, 3, w_min=0.05), 3, budget_s=1.0, seed=10042)


def test_cardinality_and_start_are_validated():
    problem = _problem(10, 1023, 3)
    with pytest.raises(ValueError, match="cardinality"):
        local_search(problem, 0, budget_s=1.0, seed=10043)
    with pytest.raises(ValueError, match="cardinality"):
        local_search(problem, 11, budget_s=1.0, seed=10043)
    with pytest.raises(ValueError, match="start"):
        local_search(problem, 3, budget_s=1.0, seed=10043, start=common_start(problem, 4))
