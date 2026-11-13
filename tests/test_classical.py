"""Sanity tests for the classical solver stack."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cif.classical.brute_force import brute_force_cardinality_continuous, brute_force_discrete
from cif.classical.continuous import minimum_variance_portfolio, solve_continuous_mvo
from cif.classical.milp import _box_ok, solve_miqp_ecos_bb, solve_miqp_scip
from cif.problem import PortfolioProblem


def test_n2_minimum_variance_matches_closed_form():
    """For two uncorrelated assets, the analytic min-var weight is sigma2² / (sigma1² + sigma2²)."""
    sigma1_sq, sigma2_sq = 0.04, 0.09
    problem = PortfolioProblem(
        mu=np.array([0.10, 0.15]),
        sigma=np.array([[sigma1_sq, 0.0], [0.0, sigma2_sq]]),
        asset_names=("A", "B"),
    )
    sol = minimum_variance_portfolio(problem)
    expected_w0 = sigma2_sq / (sigma1_sq + sigma2_sq)
    assert sol.feasible
    assert np.isclose(sol.weights[0], expected_w0, atol=1e-4)
    assert np.isclose(sol.weights.sum(), 1.0, atol=1e-6)


def test_continuous_mvo_obeys_box_constraints():
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.05, 0.20, size=6)
    A = rng.standard_normal((6, 6))
    sigma = A @ A.T * 0.01 + np.eye(6) * 0.01
    problem = PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"A{i}" for i in range(6)),
        w_min=0.0,
        w_max=0.3,
    )
    sol = solve_continuous_mvo(problem)
    assert sol.feasible
    assert np.all(sol.weights >= -1e-6)
    assert np.all(sol.weights <= 0.30 + 1e-6)
    assert np.isclose(sol.weights.sum(), 1.0, atol=1e-6)


def test_brute_force_matches_continuous_for_low_levels(small_problem):
    """At a coarse grid, brute-force discrete should land near the continuous optimum."""
    cont = solve_continuous_mvo(small_problem)
    bf = brute_force_discrete(small_problem, n_levels=20)
    gap = (bf.objective - cont.objective) / abs(cont.objective)
    assert gap < 0.05  # within 5%
    assert np.isclose(bf.weights.sum(), 1.0, atol=1e-6)


def test_brute_force_cardinality_finds_smaller_support():
    rng = np.random.default_rng(2)
    n = 8
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    p = PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"X{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=3,
    )
    bf = brute_force_cardinality_continuous(p, cardinality=3)
    n_held = int((bf.weights > 1e-6).sum())
    assert n_held <= 3
    assert np.isclose(bf.weights.sum(), 1.0, atol=1e-6)


def test_scip_and_ecos_bb_agree_on_optimum(medium_problem):
    """Both MIQP backends should reach the same objective on a small instance."""
    scip = solve_miqp_scip(medium_problem)
    ecos = solve_miqp_ecos_bb(medium_problem)
    assert scip.feasible and ecos.feasible
    assert abs(scip.objective - ecos.objective) < 1e-3
    assert scip.solver_meta["n_nonzero"] <= medium_problem.cardinality
    assert ecos.solver_meta["n_nonzero"] <= medium_problem.cardinality


# ---------------------------------------------------------------------------
# R16: semantics of w_min (minimum purchase in the MIQP, lower bound of all weights in the MVO)
# ---------------------------------------------------------------------------


def _buy_in_problem(cardinality: int | None) -> PortfolioProblem:
    """N = 8, minimum purchase 0.1, cap 0.6: the optimum holds 3 of the 8 assets."""
    n = 8
    rng = np.random.default_rng(103)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"B{i}" for i in range(n)),
        w_min=0.1,
        w_max=0.6,
        risk_aversion=5.0,
        cardinality=cardinality,
    )


def test_miqp_optimum_with_a_buy_in_threshold_is_feasible():
    """w_min bounds only the bought assets, so the zero weights of the others are feasible.

    The box check used to reject them, and the true optimum was reported as infeasible.
    """
    problem = _buy_in_problem(cardinality=3)
    sol = solve_miqp_scip(problem)
    held = sol.weights > 1e-6
    assert 0 < held.sum() <= 3 < problem.n  # some assets stay unbought, with weight exactly 0
    assert np.all(sol.weights[held] >= problem.w_min - 1e-6)
    assert sol.solver_meta["box_ok"]
    assert sol.feasible
    # and it is the optimum: best convex refinement over the supports (one asset cannot reach the
    # budget under w_max = 0.6, so sizes 2 and 3)
    best = min(brute_force_cardinality_continuous(problem, k).objective for k in (2, 3))
    assert sol.objective == pytest.approx(best, rel=1e-5)


def test_box_check_applies_the_buy_in_threshold_to_bought_assets_only():
    w_min, w_max = 0.1, 0.6
    assert _box_ok(np.array([0.0, 0.6, 0.4, 0.0]), w_min, w_max)  # unbought assets sit at 0
    assert _box_ok(np.array([0.0, 0.1, 0.6, 0.3]), w_min, w_max)  # a purchase at the threshold
    assert not _box_ok(np.array([0.0, 0.05, 0.6, 0.35]), w_min, w_max)  # bought below w_min
    assert not _box_ok(np.array([0.0, 0.7, 0.3, 0.0]), w_min, w_max)  # above w_max
    assert not _box_ok(np.array([0.0, 0.0, 0.0, 0.7]), 0.0, w_max)  # the cap holds for w_min = 0
    assert not _box_ok(np.array([-0.2, 0.6, 0.6]), 0.0, w_max)  # a short position is outside
    # solver noise within the 1e-6 slack is not a violation ...
    assert _box_ok(np.array([0.0, 0.1 - 5e-7, 0.6 + 5e-7, 0.3]), w_min, w_max)
    # ... and a residual below the support threshold of n_nonzero / card_ok is not a purchase
    assert _box_ok(np.array([3e-7, 0.6, 0.4 - 3e-7]), w_min, w_max)
    # w_min = 0: only the cap matters
    assert _box_ok(np.array([0.0, 1e-3, 0.999]), 0.0, 1.0)


def test_miqp_without_cardinality_leaves_assets_unbought_under_a_buy_in_threshold():
    problem = _buy_in_problem(cardinality=None)
    sol = solve_miqp_scip(problem)
    held = sol.weights > 1e-6
    assert 0 < held.sum() < problem.n
    assert np.all(sol.weights[held] >= problem.w_min - 1e-6)
    assert sol.solver_meta["box_ok"]
    assert sol.feasible


def test_continuous_mvo_applies_w_min_to_every_weight():
    """Unlike the MIQP, ``solve_continuous_mvo`` has no ``z``: ``w_min`` bounds all weights.

    On a chosen subset that means "every asset of the subset is bought" (the convex refinement of
    the heuristics); on the whole universe it forces all N assets in and is infeasible when
    N·w_min > budget. The behaviour is documented in ``PortfolioProblem`` and pinned here.
    """
    problem = _buy_in_problem(cardinality=None)
    sol = solve_continuous_mvo(problem)
    assert np.all(sol.weights >= problem.w_min - 1e-6)  # all 8 assets are held
    too_dense = dataclasses.replace(problem, w_min=0.2)  # N·w_min = 1.6 > budget
    with pytest.raises(RuntimeError):
        solve_continuous_mvo(too_dense)
    # the same threshold is feasible for the MIQP: it only binds the bought assets
    miqp = solve_miqp_scip(too_dense)
    assert miqp.feasible
    assert 0 < int((miqp.weights > 1e-6).sum()) < too_dense.n


def test_brute_force_discrete_treats_w_min_as_minimum_purchase():
    """R16: in the discrete solvers ``w_min`` binds bought assets only. With K < N and
    ``w_min > 0`` the grid search must find the best portfolio among those whose positive weights
    are all ≥ w_min (it used to require every weight ≥ w_min, so nothing was feasible)."""
    from itertools import product

    rng = np.random.default_rng(5)
    n, levels = 4, 10
    a = rng.standard_normal((n, n))
    p = PortfolioProblem(mu=rng.uniform(0.05, 0.2, n), sigma=a @ a.T / n + 0.05 * np.eye(n),
                         asset_names=tuple("ABCD"), risk_aversion=3.0, cardinality=2,
                         w_min=0.25, w_max=1.0)
    bf = brute_force_discrete(p, n_levels=levels)
    step = 1.0 / levels
    best = min(
        p.objective_value(np.array(c) * step)
        for c in product(range(levels + 1), repeat=n)
        if sum(c) == levels and sum(k > 0 for k in c) <= 2 and all(k == 0 or k * step >= 0.25 - 1e-12 for k in c)
    )
    assert bf.objective == pytest.approx(best)
    held = bf.weights[bf.weights > 1e-9]
    assert len(held) <= 2 and np.all(held >= 0.25 - 1e-9)
