"""Identical decision rule for exact and heuristic solvers (R3, finding 12).

With a turnover penalty ``c·‖w − w_prev‖₁`` the heuristic second stage must optimise the SAME
objective as the native MIQP: the convex refinement of each candidate subset includes the
penalty, and candidates are ranked by the penalised objective. Then enumerating all K-subsets
reproduces the MIQP optimum exactly.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from cif.classical.continuous import solve_continuous_mvo
from cif.classical.scip_native import solve_miqp_native
from cif.heuristics.pipeline import refine_candidates
from cif.heuristics.swap_sa import solve_with_swap_sa
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import solve_with_tabu_selection


def _problem(n=8, seed=3, w_max=0.6, K=3):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.01 + np.eye(n) * 0.02
    return PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"A{i}" for i in range(n)),
                            risk_aversion=3.0, w_max=w_max, cardinality=K)


def _prev(n, support, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(n)
    w[list(support)] = rng.dirichlet(np.ones(len(support)))
    return w


def test_continuous_turnover_penalty_pulls_weights_to_prev():
    p = _problem(n=6, K=None) if False else _problem(n=6)
    prev = np.full(6, 1 / 6)
    free = solve_continuous_mvo(p)
    tied = solve_continuous_mvo(p, prev_weights=prev, turnover_penalty=10.0)
    assert np.abs(tied.weights - prev).sum() < 1e-4
    assert np.abs(free.weights - prev).sum() > 1e-2
    # objective stays the pure mean-variance objective
    assert tied.objective == pytest.approx(p.objective_value(tied.weights))


def test_continuous_without_penalty_is_unchanged():
    p = _problem(n=6)
    a = solve_continuous_mvo(p)
    b = solve_continuous_mvo(p, prev_weights=np.full(6, 1 / 6), turnover_penalty=0.0)
    np.testing.assert_allclose(a.weights, b.weights, atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_enumeration_with_turnover_refinement_matches_native_miqp(seed):
    p = _problem(n=8, seed=seed)
    K = 3
    prev = _prev(p.n, support=(0, 1, 2), seed=seed)
    c = 0.02
    exact = solve_miqp_native(p, K, formulation="perspective", prev_weights=prev,
                              turnover_penalty=c, mip_gap=0.0, time_limit_s=60)
    best, _ = refine_candidates(p, combinations(range(p.n), K), K, m=10**6, prev_weights=prev,
                                turnover_cost=c, turnover_in_refinement=True)
    pen = lambda w: p.objective_value(w) + c * np.abs(w - prev).sum()  # noqa: E731
    # Enumeration over K-subsets covers supports of size ≤ K because refined weights may be 0.
    assert pen(best.weights) == pytest.approx(pen(exact.weights), rel=1e-6, abs=1e-8)


def test_turnover_refinement_never_worse_than_ranking_only():
    p = _problem(n=8, seed=5)
    prev = _prev(p.n, support=(1, 4, 6), seed=5)
    c = 0.05
    pen = lambda w: p.objective_value(w) + c * np.abs(w - prev).sum()  # noqa: E731
    subsets = list(combinations(range(p.n), 3))
    rank_only, _ = refine_candidates(p, subsets, 3, m=10**6, prev_weights=prev, turnover_cost=c)
    refined, _ = refine_candidates(p, subsets, 3, m=10**6, prev_weights=prev, turnover_cost=c,
                                   turnover_in_refinement=True)
    assert pen(refined.weights) <= pen(rank_only.weights) + 1e-9


def test_solvers_pass_turnover_in_refinement_through():
    p = _problem(n=12, seed=7, K=4)
    prev = _prev(p.n, support=(0, 3, 5, 7), seed=7)
    tabu = solve_with_tabu_selection(p, cardinality=4, num_reads=20, tenure=5, seed=1, timeout_ms=None,
                                     num_restarts=2, top_subsets=20, pool_order="energy",
                                     prev_weights=prev, turnover_cost=0.05, turnover_in_refinement=True,
                                     warm_start_subset=(0, 3, 5, 7))
    swap = solve_with_swap_sa(p, 4, num_reads=4, num_sweeps=50, seed=1, m=20, prev_weights=prev,
                              turnover_cost=0.05, turnover_in_refinement=True,
                              warm_start_subset=(0, 3, 5, 7))
    for sol in (tabu, swap):
        assert sol.feasible
        assert "ranking_objective" in sol.solver_meta
