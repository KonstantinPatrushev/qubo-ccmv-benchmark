"""Shared refinement: optimal weights on a given support (ANALYSIS_PLAN §4, round 1 N5).

``refine(S)`` solves the convex mean-variance problem restricted to the assets of S::

    minimize   −μ_Sᵀw + (λ/2)·wᵀΣ_SS w
    subject to Σw = budget,  lower_i ≤ w_i ≤ upper_i  (i ∈ S),  w = 0 outside S.

Every method that refines a subset uses this one implementation: the QUBO pipeline, the local
search on the true objective, the subproblems of outer approximation, the RQ1 oracle and the RQ3
gap-injection operators. It calls Clarabel directly (no CVXPY), which keeps a call below a few
milliseconds for the K of the study.

Two semantics of the minimum purchase ``w_min`` (R16), chosen by :func:`refine_problem`:
- ``card_mode="le"`` (``Σz ≤ K``, ``w_min = 0``): lower bound 0, so the support of the result may
  be a strict subset of S. This is consistent with ``f*`` under ``Σz ≤ K``.
- ``card_mode="eq"`` (``Σz = K``, ORlib with the Chang et al. bounds): every asset of S is held with
  at least ``w_min``.
With ``w_min > 0`` and ``Σz ≤ K`` the refinement would itself be combinatorial; that case is
rejected.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp

from cif.problem import PortfolioProblem

_SOLVED = {"Solved", "AlmostSolved"}
_INFEASIBLE = {"PrimalInfeasible", "AlmostPrimalInfeasible"}


@dataclass(frozen=True)
class RefineDuals:
    """KKT multipliers of the refinement, for the Lagrangian

    ``−μᵀw + (λ/2)·wᵀΣw + ν·(Σw − budget) + αᵀ(w − upper) + βᵀ(lower − w)`` with ``α, β ≥ 0``,
    so that on the support ``−μ + λ·Σw + ν + α − β = 0``. Hence ``dV/d budget = −ν`` and
    ``dV/d upper_i = −α_i`` for the optimal value ``V``. ``lower`` (β) and ``upper`` (α) have
    length N and are zero outside the support."""

    budget: float  # ν
    lower: np.ndarray  # β
    upper: np.ndarray  # α


@dataclass(frozen=True)
class RefineResult:
    weights: np.ndarray  # length N, zeros outside the support
    objective: float  # f(w) = −μᵀw + (λ/2)·wᵀΣw; +inf if infeasible or failed
    status: str  # "optimal", "infeasible" or the Clarabel status
    time_s: float
    duals: RefineDuals | None = None  # only with return_duals=True and status "optimal"


def _settings() -> clarabel.DefaultSettings:
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.tol_gap_abs = 1e-10
    s.tol_gap_rel = 1e-10
    s.tol_feas = 1e-10
    return s


def _bounds(value: float | np.ndarray, n: int, s: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return np.full(len(s), float(arr)) if arr.ndim == 0 else arr.reshape(n)[s]


def refine_support(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    support: Sequence[int] | np.ndarray,
    lower: float | np.ndarray = 0.0,
    upper: float | np.ndarray = 1.0,
    budget: float = 1.0,
    return_duals: bool = False,
) -> RefineResult:
    """Optimal weights on ``support``; ``lower``/``upper`` are scalars or length-N vectors.

    With ``return_duals`` the result also carries the multipliers of Clarabel's solution in
    ``RefineResult.duals`` (see :class:`RefineDuals`); weights and objective are unchanged."""
    t0 = time.perf_counter()
    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    s = np.unique(np.asarray(support, dtype=int))
    k = s.size
    lo, up = _bounds(lower, n, s), _bounds(upper, n, s)
    weights = np.zeros(n)
    if k == 0 or lo.sum() > budget + 1e-12 or up.sum() < budget - 1e-12 or np.any(lo > up):
        return RefineResult(weights, np.inf, "infeasible", time.perf_counter() - t0)

    sig = np.asarray(sigma, dtype=float)[np.ix_(s, s)]
    P = sp.csc_matrix(np.triu(risk_aversion * sig))
    q = -mu[s]
    eye = sp.identity(k, format="csc")
    A = sp.vstack([sp.csc_matrix(np.ones((1, k))), -eye, eye], format="csc")
    b = np.concatenate([[budget], -lo, up])
    cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(2 * k)]
    solution = clarabel.DefaultSolver(P, q, A, b, cones, _settings()).solve()
    status = str(solution.status).split(".")[-1]
    if status in _INFEASIBLE:
        return RefineResult(weights, np.inf, "infeasible", time.perf_counter() - t0)
    if status not in _SOLVED:
        return RefineResult(weights, np.inf, status, time.perf_counter() - t0)
    weights[s] = np.clip(np.asarray(solution.x), lo, up)
    objective = float(-mu @ weights + 0.5 * risk_aversion * weights @ np.asarray(sigma) @ weights)
    duals = None
    if return_duals:
        z = np.asarray(solution.z)  # rows of A: budget, −w ≥ −lower, w ≤ upper
        lower_dual, upper_dual = np.zeros(n), np.zeros(n)
        lower_dual[s], upper_dual[s] = z[1 : 1 + k], z[1 + k :]
        duals = RefineDuals(float(z[0]), lower_dual, upper_dual)
    return RefineResult(weights, objective, "optimal", time.perf_counter() - t0, duals)


def refine_problem(
    problem: PortfolioProblem, support: Sequence[int] | np.ndarray, card_mode: str = "le"
) -> RefineResult:
    """:func:`refine_support` with the bounds of ``problem`` and the chosen cardinality semantics."""
    if card_mode not in ("le", "eq"):
        raise ValueError(f"card_mode must be 'le' or 'eq', got {card_mode!r}")
    if card_mode == "le" and problem.w_min > 0:
        raise ValueError(
            "card_mode='le' with w_min > 0 makes the refinement combinatorial; use card_mode='eq' "
            "(every asset of S held) or a problem with w_min = 0"
        )
    lower = problem.w_min if card_mode == "eq" else 0.0
    return refine_support(problem.mu, problem.sigma, problem.risk_aversion, support,
                          lower=lower, upper=problem.w_max, budget=problem.budget)
