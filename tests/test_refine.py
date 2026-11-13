"""Shared refinement on a support (ANALYSIS_PLAN §4: one implementation for every method).

Checked against the CVXPY reference ``solve_continuous_mvo`` on the subproblem.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cif.classical.continuous import solve_continuous_mvo
from cif.classical.refine import refine_problem, refine_support
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem


def _problem(n: int, seed: int, w_min: float = 0.0, w_max: float = 1.0, lam: float = 5.0) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)),
                            risk_aversion=lam, w_min=w_min, w_max=w_max)


def _reference(problem: PortfolioProblem, support, w_min: float) -> float:
    s = np.asarray(sorted(support))
    sub = PortfolioProblem(mu=problem.mu[s], sigma=problem.sigma[np.ix_(s, s)],
                           asset_names=tuple(problem.asset_names[i] for i in s),
                           risk_aversion=problem.risk_aversion, w_min=w_min, w_max=problem.w_max)
    return solve_continuous_mvo(sub).objective


@pytest.mark.parametrize("w_max", [1.0, 0.3])
def test_matches_cvxpy_reference_with_supports_inside_s(w_max):
    rng = np.random.default_rng(0)
    for seed in range(8):
        problem = _problem(30, 1000 + seed, w_max=w_max)
        support = rng.choice(30, size=8, replace=False)
        res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, support,
                             lower=0.0, upper=w_max)
        assert res.status == "optimal"
        ref = _reference(problem, support, 0.0)
        assert res.objective == pytest.approx(ref, rel=1e-6, abs=1e-9)
        w = res.weights
        assert w.shape == (30,) and np.all(w[np.setdiff1d(np.arange(30), support)] == 0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-9) and np.all(w <= w_max + 1e-9)
        assert res.objective == pytest.approx(problem.objective_value(w), rel=1e-12, abs=1e-14)


def test_all_held_semantics_puts_every_asset_above_the_minimum_buy():
    problem = _problem(20, 1100, w_min=0.01, w_max=1.0)
    support = [0, 3, 5, 7, 11, 13, 17, 19, 2, 4]
    res = refine_problem(problem, support, card_mode="eq")
    assert res.status == "optimal"
    assert np.all(res.weights[support] >= 0.01 - 1e-9)
    assert res.objective == pytest.approx(_reference(problem, support, 0.01), rel=1e-6, abs=1e-9)


def test_le_mode_requires_zero_minimum_buy():
    problem = _problem(10, 1200, w_min=0.05)
    with pytest.raises(ValueError, match="w_min"):
        refine_problem(problem, [0, 1, 2], card_mode="le")


def test_infeasible_bounds_are_reported_not_raised():
    problem = _problem(10, 1300, w_max=0.2)
    res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, [0, 1, 2],
                         lower=0.0, upper=0.2)  # 3 assets × 0.2 < 1
    assert res.status == "infeasible" and res.objective == np.inf
    assert np.all(res.weights == 0.0)


def test_fast_enough_for_enumeration():
    problem = _problem(100, 1400)
    rng = np.random.default_rng(1)
    supports = [rng.choice(100, size=10, replace=False) for _ in range(200)]
    t0 = time.perf_counter()
    for s in supports:
        refine_support(problem.mu, problem.sigma, problem.risk_aversion, s)
    per_call = (time.perf_counter() - t0) / len(supports)
    assert per_call < 0.005  # CVXPY needed 10–13 ms per subset (round-1 review)


# ---------------------------------------------------------------------------
# Duals (return_duals=True): the KKT multipliers behind the OA cuts
# ---------------------------------------------------------------------------

_DUAL_CASES = [(1.0, 0.0), (0.3, 0.0), (0.4, 0.05)]  # (w_max, w_min): caps and buy-ins bind


def _dual_case(w_max: float, w_min: float, seed: int = 0, n: int = 25, k: int = 7):
    problem = _problem(n, 1500 + seed, w_max=w_max, w_min=w_min)
    support = np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))
    res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, support,
                         lower=w_min, upper=w_max, return_duals=True)
    return problem, support, res


def test_duals_are_off_by_default_and_the_flag_leaves_the_solution_untouched():
    problem = _problem(20, 1450, w_max=0.4)
    args = (problem.mu, problem.sigma, problem.risk_aversion, [1, 4, 6, 9, 12, 15])
    plain = refine_support(*args, lower=0.0, upper=0.4)
    with_duals = refine_support(*args, lower=0.0, upper=0.4, return_duals=True)
    assert plain.duals is None
    assert with_duals.duals is not None
    assert np.array_equal(plain.weights, with_duals.weights)
    assert plain.objective == with_duals.objective
    assert plain.status == with_duals.status == "optimal"


@pytest.mark.parametrize(("w_max", "w_min"), _DUAL_CASES)
def test_duals_satisfy_the_kkt_conditions(w_max, w_min):
    """Lagrangian −μᵀw + (λ/2)wᵀΣw + ν(Σw − b) + αᵀ(w − u) + βᵀ(l − w) on the support."""
    saw_cap = saw_floor = False
    for seed in range(6):
        problem, s, res = _dual_case(w_max, w_min, seed)
        d = res.duals
        assert res.status == "optimal" and d is not None
        w, lam = res.weights[s], problem.risk_aversion
        stationarity = (-problem.mu[s] + lam * problem.sigma[np.ix_(s, s)] @ w
                        + d.budget + d.upper[s] - d.lower[s])
        assert np.abs(stationarity).max() < 1e-8
        assert d.upper.min() >= 0.0 and d.lower.min() >= 0.0
        outside = np.setdiff1d(np.arange(problem.n), s)
        assert np.all(d.upper[outside] == 0.0) and np.all(d.lower[outside] == 0.0)
        assert np.abs(d.upper[s] * (w_max - w)).max() < 1e-9  # complementary slackness
        assert np.abs(d.lower[s] * (w - w_min)).max() < 1e-9
        saw_cap |= bool((d.upper > 1e-6).any())
        saw_floor |= bool((d.lower > 1e-6).any())
    assert saw_cap == (w_max < 1.0)  # guard: the bounds under test really are active
    assert saw_floor


def test_duals_are_the_sensitivities_of_the_optimal_value():
    """dV/d(budget) = −ν and dV/d(upper_i) = −α_i, the meaning behind the sign convention."""
    problem, s, res = _dual_case(0.3, 0.0, seed=1)
    d, h = res.duals, 1e-5
    capped = int(np.argmax(d.upper))
    assert d.upper[capped] > 1e-4

    def value(budget: float = 1.0, cap: float = 0.3) -> float:
        upper = np.full(problem.n, 0.3)
        upper[capped] = cap
        r = refine_support(problem.mu, problem.sigma, problem.risk_aversion, s,
                           lower=0.0, upper=upper, budget=budget)
        return r.objective

    assert (value(budget=1 + h) - value(budget=1 - h)) / (2 * h) == pytest.approx(-d.budget, abs=1e-6)
    assert (value(cap=0.3 + h) - value(cap=0.3 - h)) / (2 * h) == pytest.approx(-d.upper[capped], abs=1e-6)


def test_no_duals_when_the_support_is_infeasible():
    problem = _problem(10, 1300, w_max=0.2)
    res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, [0, 1, 2],
                         lower=0.0, upper=0.2, return_duals=True)
    assert res.status == "infeasible" and res.duals is None
