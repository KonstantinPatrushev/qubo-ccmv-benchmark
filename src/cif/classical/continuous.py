"""Continuous mean-variance reference solver (cvxpy SOCP).

This is the *upper bound* of achievable objective for the problem without
cardinality: any discrete solver using the same ``mu, sigma, w_min, w_max``
should land at or worse than the continuous optimum. It also serves as the
warm-start / sanity check for all other experiments.
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from cif.problem import PortfolioProblem, Solution

_DEFAULT_SOLVER_ORDER = ("CLARABEL", "ECOS", "SCS")


def solve_continuous_mvo(
    problem: PortfolioProblem,
    solver: str | None = None,
    verbose: bool = False,
    *,
    prev_weights: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
) -> Solution:
    """Solve the continuous mean-variance problem.

    Formulation (minimisation form, consistent with the rest of the project)::

        minimize   − μᵀw + (λ/2) wᵀΣw  [+ c·‖w − w_prev‖₁]
        subject to sum(w)   = budget
                   w_min   ≤ w ≤ w_max

    The cardinality constraint is **ignored** here by design — this function
    is the convex reference. Discrete cardinality goes through
    :mod:`cif.classical.milp` or the QUBO solvers.

    With ``prev_weights`` and ``turnover_penalty = c > 0`` the L1 trading
    penalty is added (the same term as in
    :func:`cif.classical.scip_native.solve_miqp_native`), so turnover-aware
    decision rules optimise one objective across solvers (R3). The returned
    ``objective`` is always the pure mean-variance objective; the penalised value
    is in ``solver_meta["penalised_objective"]``.
    """
    n = problem.n
    w = cp.Variable(n)

    sigma_psd = cp.psd_wrap(problem.sigma)
    expr = -problem.mu @ w + 0.5 * problem.risk_aversion * cp.quad_form(w, sigma_psd)
    prev = None
    if prev_weights is not None and turnover_penalty > 0.0:
        prev = np.asarray(prev_weights, dtype=float)
        if prev.shape != (n,):
            raise ValueError(f"prev_weights must have shape ({n},), got {prev.shape}")
        expr = expr + float(turnover_penalty) * cp.norm1(w - prev)
    objective = cp.Minimize(expr)
    constraints = [
        cp.sum(w) == problem.budget,
        w >= problem.w_min,
        w <= problem.w_max,
    ]

    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    solver_candidates = (solver,) if solver else _DEFAULT_SOLVER_ORDER
    last_exc: Exception | None = None
    for candidate in solver_candidates:
        try:
            prob.solve(solver=candidate, verbose=verbose)
            if prob.status in {"optimal", "optimal_inaccurate"}:
                break
        except Exception as exc:
            last_exc = exc
            continue
    wall = time.perf_counter() - t0

    if prob.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(
            f"cvxpy continuous MVO failed: status={prob.status}, last_exc={last_exc}"
        )

    weights = np.asarray(w.value, dtype=float)
    weights = np.clip(weights, problem.w_min, problem.w_max)
    objective_value = problem.objective_value(weights)
    meta = {
        "status": prob.status,
        "cvxpy_value": float(prob.value) if prob.value is not None else None,
    }
    if prev is not None:
        meta["turnover_penalty"] = float(turnover_penalty)
        meta["penalised_objective"] = objective_value + float(turnover_penalty) * float(
            np.abs(weights - prev).sum()
        )
    return Solution(
        weights=weights,
        objective=objective_value,
        feasible=True,
        wall_time_s=wall,
        solver=f"cvxpy_continuous/{prob.solver_stats.solver_name if prob.solver_stats else 'unknown'}",
        solver_meta=meta,
    )


def minimum_variance_portfolio(problem: PortfolioProblem) -> Solution:
    """Minimum-variance portfolio — corresponds to ``μ = 0`` in the generic solver."""
    zero_mu_problem = PortfolioProblem(
        mu=np.zeros_like(problem.mu),
        sigma=problem.sigma,
        asset_names=problem.asset_names,
        w_min=problem.w_min,
        w_max=problem.w_max,
        budget=problem.budget,
        risk_aversion=1.0,
    )
    sol = solve_continuous_mvo(zero_mu_problem)
    sol.objective = problem.objective_value(sol.weights)
    sol.solver = "cvxpy_continuous/minvar"
    return sol


def efficient_frontier(
    problem: PortfolioProblem,
    risk_aversions: np.ndarray,
) -> list[Solution]:
    """Trace the efficient frontier by sweeping risk-aversion ``λ``."""
    solutions: list[Solution] = []
    for lam in risk_aversions:
        p = PortfolioProblem(
            mu=problem.mu,
            sigma=problem.sigma,
            asset_names=problem.asset_names,
            w_min=problem.w_min,
            w_max=problem.w_max,
            budget=problem.budget,
            risk_aversion=float(lam),
        )
        solutions.append(solve_continuous_mvo(p))
    return solutions
