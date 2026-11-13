"""The common start of all methods (ANALYSIS_PLAN §5): top-K rounding of the relaxation, refined.

The relaxation is cross-checked against the CVXPY reference ``solve_continuous_mvo``; the
refinement is the shared :func:`cif.classical.refine.refine_problem`.
"""

from __future__ import annotations

import numpy as np
import pytest

from cif.classical.continuous import solve_continuous_mvo
from cif.classical.refine import refine_problem
from cif.heuristics.start import CommonStart, common_start
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem


def _problem(
    n: int, seed: int, K: int | None = None, *, lam: float = 5.0, w_min: float = 0.0,
    w_max: float = 1.0,
) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"a{i}" for i in range(n)),
        risk_aversion=lam,
        w_min=w_min,
        w_max=w_max,
        cardinality=K,
    )


def test_the_subset_is_the_k_largest_weights_of_the_continuous_relaxation():
    problem = _problem(30, 1000, K=4)
    relaxed = solve_continuous_mvo(problem)  # cardinality is ignored there by design
    expected = tuple(sorted(np.argsort(-relaxed.weights, kind="stable")[:4].tolist()))
    start = common_start(problem)
    assert isinstance(start, CommonStart)
    assert start.subset == expected
    assert len(start.subset) == 4


def test_the_weights_are_the_refinement_of_the_subset():
    problem = _problem(30, 1001, K=5)
    start = common_start(problem)
    ref = refine_problem(problem, start.subset)
    np.testing.assert_allclose(start.weights, ref.weights, atol=1e-12)
    assert start.objective == pytest.approx(ref.objective, abs=1e-14)
    assert start.objective == pytest.approx(problem.objective_value(start.weights), abs=1e-14)
    assert start.status == "optimal"
    assert start.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert set(np.flatnonzero(start.weights > 1e-12)) <= set(start.subset)


def test_the_start_lies_between_the_relaxation_and_equal_weights():
    problem = _problem(40, 1002, K=5)
    f_cont = solve_continuous_mvo(problem).objective
    start = common_start(problem)
    equal = np.zeros(problem.n)
    equal[list(start.subset)] = 1.0 / 5
    assert f_cont <= start.objective + 1e-9
    assert start.objective <= problem.objective_value(equal) + 1e-12


def test_a_cardinality_above_the_relaxation_support_gives_the_continuous_optimum():
    problem = _problem(30, 1003, K=25)  # far above s_cont: the cardinality does not bind
    start = common_start(problem)
    assert start.objective == pytest.approx(solve_continuous_mvo(problem).objective, rel=1e-6)


def test_k_is_the_argument_or_the_problem_cardinality():
    problem = _problem(30, 1004, K=6)
    assert len(common_start(problem).subset) == 6
    assert len(common_start(problem, 3).subset) == 3
    with pytest.raises(ValueError, match="cardinality"):
        common_start(_problem(30, 1004))
    with pytest.raises(ValueError, match="cardinality"):
        common_start(problem, 0)
    with pytest.raises(ValueError, match="cardinality"):
        common_start(problem, 31)


def test_the_construction_is_timed_and_split_into_its_parts():
    problem = _problem(60, 1005, K=6)
    start = common_start(problem)
    assert start.time_s > 0.0 and start.cpu_s >= 0.0
    assert start.relax_s > 0.0 and start.refine_s > 0.0
    # the parts are inside the whole (rounding and bookkeeping take the rest)
    assert start.relax_s + start.refine_s <= start.time_s + 1e-6
    assert start.entry == (start.time_s, start.objective)


def test_the_start_is_deterministic():
    problem = _problem(30, 1006, K=5)
    a, b = common_start(problem), common_start(problem)
    assert a.subset == b.subset
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.objective == b.objective


def test_the_subset_does_not_depend_on_the_scale_of_the_data():
    # OR-Library has Σ ~ 1e-3; the relaxation is solved on the normalised data
    problem = _problem(30, 1007, K=5)
    scaled = PortfolioProblem(
        mu=problem.mu * 1e-3,
        sigma=problem.sigma * 1e-3,
        asset_names=problem.asset_names,
        risk_aversion=problem.risk_aversion,
        cardinality=5,
    )
    a, b = common_start(problem), common_start(scaled)
    assert a.subset == b.subset
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-7)


def test_eq_mode_holds_all_k_assets_above_the_minimum_purchase():
    problem = _problem(30, 1008, K=6, w_min=0.02)
    start = common_start(problem, card_mode="eq")
    held = start.weights[list(start.subset)]
    assert start.status == "optimal"
    assert np.all(held >= 0.02 - 1e-9)
    assert np.count_nonzero(start.weights) == 6
    ref = refine_problem(problem, start.subset, card_mode="eq")
    assert start.objective == pytest.approx(ref.objective, abs=1e-14)


def test_le_mode_rejects_a_minimum_purchase_like_the_refinement_does():
    problem = _problem(30, 1008, K=6, w_min=0.02)
    with pytest.raises(ValueError, match="w_min"):
        common_start(problem)


def test_a_subset_that_cannot_reach_the_budget_is_reported_not_raised():
    problem = _problem(30, 1009, K=5, w_max=0.15)  # 5 · 0.15 < 1: no feasible K-subset
    start = common_start(problem)
    assert start.status == "infeasible"
    assert start.objective == np.inf
    assert np.all(start.weights == 0.0)
    assert len(start.subset) == 5


def test_an_infeasible_relaxation_raises():
    problem = _problem(6, 1010, K=3, w_max=0.1)  # even all 6 assets hold at most 0.6
    with pytest.raises(ValueError, match="infeasible"):
        common_start(problem)
