"""One end-to-end clock for every method of RQ2 (ANALYSIS_PLAN §5-§6).

SCIP, outer approximation, the local search and the QUBO loop are given the same common start
(:func:`cif.heuristics.start.common_start`), the same budget ``budget_s`` and, for the exact
solvers, ``clock_offset_s = start.time_s``: the heuristics take the construction time of the start
from the start itself. What is checked here is that their trajectories and end times then live on
one clock, so that :func:`cif.metrics.anytime.primal_integral` takes the trajectories as they are,
and that one budget stops them all. Wall-clock bounds are coarse on purpose (the machine is shared).
Instance seeds are calibration ones (1000-1999), solver seeds 10000-19999.
"""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import pytest

from cif.classical.oa import solve_oa
from cif.classical.scip_native import solve_miqp_native
from cif.heuristics.anytime import anytime_swap_sa
from cif.heuristics.local_search import local_search
from cif.heuristics.start import CommonStart, common_start
from cif.instances.factor import factor_instance
from cif.metrics.anytime import primal_integral
from cif.metrics.gap import reference_scale
from cif.problem import PortfolioProblem

#: Construction time of the common start, chosen large against the solves of the small instances
#: so that a trajectory on the wrong clock cannot pass.
START_TIME_S = 1.5


def _problem(n: int, seed: int) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)), risk_aversion=5.0
    )


def _start(problem: PortfolioProblem, k: int) -> CommonStart:
    return dataclasses.replace(common_start(problem, k), time_s=START_TIME_S)


def _methods(problem: PortfolioProblem, k: int, start: CommonStart, budget_s: float) -> dict:
    return {
        "scip": lambda: solve_miqp_native(
            problem, k, warm_start=start.weights, budget_s=budget_s, clock_offset_s=start.time_s
        ),
        "oa": lambda: solve_oa(
            problem, k, warm_start=list(start.subset), budget_s=budget_s,
            clock_offset_s=start.time_s,
        ),
        "local_search": lambda: local_search(problem, k, budget_s=budget_s, seed=10001, start=start),
        "swap_sa": lambda: anytime_swap_sa(
            problem, k, num_reads_b=20, num_sweeps=20, budget_s=budget_s, seed=10002, start=start
        ),
    }  # fmt: skip


def _trajectory(solution) -> list:
    """The incumbent trajectory of a method: ``trajectory``, OA calls it ``incumbent_trajectory``."""
    meta = solution.solver_meta
    return meta["trajectory"] if "trajectory" in meta else meta["incumbent_trajectory"]


@pytest.fixture(scope="module")
def small_run():
    """N = 24, K = 4: the exact solvers finish well inside the budget, the heuristics use it up."""
    problem, k, budget = _problem(24, 1000), 4, START_TIME_S + 0.6
    start = _start(problem, k)
    runs = {name: run() for name, run in _methods(problem, k, start, budget).items()}
    return problem, start, budget, runs


def test_every_trajectory_and_end_time_is_on_the_clock_that_starts_with_the_start(small_run):
    _, start, _, runs = small_run
    for name, sol in runs.items():
        times = [t for t, _ in _trajectory(sol)]
        assert all(a <= b for a, b in pairwise(times)), name
        # nobody has a solution of their own before the start has been built
        assert not times or min(times) >= start.time_s, name
        assert not times or max(times) <= sol.wall_time_s + 1e-9, name
        assert sol.wall_time_s >= start.time_s, name


def test_the_heuristics_begin_at_the_start_and_the_exact_methods_agree_on_its_clock(small_run):
    _, start, budget, runs = small_run
    for name in ("local_search", "swap_sa"):
        assert _trajectory(runs[name])[0] == pytest.approx(start.entry), name
    for name in ("scip", "oa"):
        meta = runs[name].solver_meta
        assert meta["clock_offset_s"] == start.time_s and meta["budget_s"] == budget, name
        assert meta["warm_start_used"] is True, name
        # build and solve are durations of the call, the end of it is on the common clock
        duration = meta["build_time_s"] + meta["solve_time_s"]
        assert start.time_s + duration <= runs[name].wall_time_s + 1e-3, name
        assert runs[name].wall_time_s <= start.time_s + duration + 1.0, name


def test_the_primal_integral_takes_every_trajectory_as_it_is(small_run):
    problem, start, budget, runs = small_run
    best = min(runs.values(), key=lambda sol: sol.objective)
    f_ref, s_ref = best.objective, reference_scale(problem, best.weights)
    for name, sol in runs.items():
        pi = primal_integral(
            _trajectory(sol), f_ref=f_ref, s_ref=s_ref, budget_s=budget, start=start.entry
        )
        # γ = 1 until the start is built (nobody has an incumbent before), so none is below t₀/T
        assert start.time_s / budget - 1e-12 <= pi <= 1.0, name


def test_one_budget_stops_every_method_near_it():
    """N = 100, K = 7: SCIP needs ~5 s and OA more than 20 s, so the budget binds for both."""
    problem, k = _problem(100, 1000), 7
    budget = START_TIME_S + 0.8
    start = _start(problem, k)
    for name, run in _methods(problem, k, start, budget).items():
        sol = run()
        assert start.time_s <= sol.wall_time_s < budget + 3.0, name
        if name in ("scip", "oa"):
            assert sol.solver_meta["time_limit_hit"] is True, name
            assert sol.solver_meta["solve_limit_s"] < budget - start.time_s, name
        else:
            assert sol.wall_time_s >= budget, name  # the loops run until the budget is spent
