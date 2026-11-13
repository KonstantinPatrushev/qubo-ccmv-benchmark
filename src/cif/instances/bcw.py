"""Problem (34) of Bertsimas & Cory-Wright (2022) on the OR-Library instances (validation M11, ANALYSIS_PLAN §12).

Section 5.1 of Bertsimas & Cory-Wright, "A Scalable Algorithm For Sparse Portfolio Selection"
(arXiv:1811.00138v5; BCW 2022 in the plan), solves on the five OR-Library files of Chang et al. (2000)::

    minimize   ½ xᵀΣx + (1/2γ)‖x‖₂² − κ μᵀx
    subject to eᵀx = 1,  x ≥ 0,  ‖x‖₀ ≤ k,                     γ = 100/√n,

with ``κ = 1`` and no other constraint (their Table 3), or with ``κ = 0`` and the minimum return
``μᵀx ≥ r̄`` (Table 4). The threshold is, "as suggested by Cesarone et al. (2009), Zheng et al. (2014)"::

    r_min = μᵀx_min,  x_min = argmin ½ xᵀ(I/γ + Σ)x          on {eᵀx = 1, x ≥ 0},
    r_max = μᵀx_max,  x_max = argmax μᵀx − xᵀx/(2γ)          on {eᵀx = 1, x ≥ 0},
    r̄ = r_min + 0.3 (r_max − r_min).

Here ``Σ`` and ``μ`` are those of the files as they are (:class:`cif.instances.orlib.OrlibInstance`:
per-period moments, ``Σ_ij = corr_ij·std_i·std_j``), and the ridge is folded into the covariance,
``Σ' = Σ + I/γ``, so that (34) is the project's objective ``f(w) = −μᵀw·κ + (λ/2)·wᵀΣ'w`` with ``λ = 1``
(:class:`cif.problem.PortfolioProblem`) and the solvers of :mod:`cif.classical` apply unchanged: their
perspective split of ``Σ'`` takes the ridge into the diagonal. :func:`bcw_objective` evaluates (34) in the form
of the paper (raw ``Σ``, the ridge as ``‖x‖²/2γ``) for the rows of the stage, which report the objective of the
paper and not an internal one.

The check on the data: the second-order-cone lower bound of port2 with ``k = 5`` is 0.009288 in the text of
their Figure 1, and the Boolean relaxation of (34) on these files gives exactly that number
(``tests/test_bcw.py``). It fixes the units of the files, ``γ`` and the ridge; it does not depend on ``r̄``
(the return constraint is slack there), so ``r̄`` itself rests on the rule above and on the closed forms
checked in the tests (``x_max`` is the Euclidean projection of ``γμ`` onto the simplex).

The two QPs for ``r̄`` are solved with cvxpy / CLARABEL at tolerance 1e-12: at the default 1e-8 the weights
of the flat objective are off by up to 1e-4 and ``r̄`` by 1e-5 of itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

from cif.instances.orlib import ORLIB_N, load_orlib
from cif.problem import PortfolioProblem

__all__ = [
    "GAMMA_NUMERATOR",
    "RETURN_SHARE",
    "BcwInstance",
    "ReturnThreshold",
    "bcw_gamma",
    "bcw_instance",
    "bcw_objective",
    "bcw_problem",
    "return_threshold",
]

#: ``γ = GAMMA_NUMERATOR/√n`` (Table 3 and 4 of the paper).
GAMMA_NUMERATOR = 100.0
#: ``r̄ = r_min + RETURN_SHARE·(r_max − r_min)`` (Section 5.1 of the paper).
RETURN_SHARE = 0.3
_QP_TOL = 1e-12  # CLARABEL tolerances of the two QPs of r̄ (module docstring)


def bcw_gamma(n: int) -> float:
    """``γ = 100/√n`` for a universe of ``n`` securities."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return GAMMA_NUMERATOR / math.sqrt(n)


@dataclass(frozen=True)
class ReturnThreshold:
    """``r̄`` of Table 4 and what it is made of: ``r_min`` (return of the minimum-variance portfolio of
    ``Σ'``), ``r_max`` (return of the maximiser of ``μᵀx − xᵀx/(2γ)``) and the ``share`` of the way from the
    first to the second."""

    r_min: float
    r_max: float
    r_bar: float
    share: float


def _simplex_argmin(sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``argmin ½xᵀΣx − μᵀx`` on ``{eᵀx = 1, x ≥ 0}`` (cvxpy / CLARABEL at ``_QP_TOL``)."""
    x = cp.Variable(mu.shape[0])
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(sigma)) - mu @ x),
                         [cp.sum(x) == 1, x >= 0])
    problem.solve(solver="CLARABEL", tol_gap_abs=_QP_TOL, tol_gap_rel=_QP_TOL, tol_feas=_QP_TOL)
    if problem.status != "optimal" or x.value is None:
        raise RuntimeError(f"the QP of the return threshold is {problem.status}")
    return np.asarray(x.value, dtype=float)


def return_threshold(
    mu: np.ndarray, sigma_ridge: np.ndarray, gamma: float, share: float = RETURN_SHARE
) -> ReturnThreshold:
    """``r̄ = r_min + share·(r_max − r_min)`` for the data ``(μ, Σ' = Σ + I/γ)`` (module docstring)."""
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must lie in [0, 1], got {share}")
    mu = np.asarray(mu, dtype=float)
    sigma_ridge = np.asarray(sigma_ridge, dtype=float)
    n = mu.shape[0]
    x_min = _simplex_argmin(sigma_ridge, np.zeros(n))
    x_max = _simplex_argmin(np.eye(n) / gamma, mu)
    r_min, r_max = float(mu @ x_min), float(mu @ x_max)
    return ReturnThreshold(r_min, r_max, r_min + share * (r_max - r_min), float(share))


@dataclass(frozen=True, eq=False)
class BcwInstance:
    """One OR-Library port as the data of (34): ``mu`` and ``sigma`` of the file, ``sigma_ridge = Σ + I/γ`` and
    the threshold of Table 4 (``threshold.r_bar``; it is part of the instance, a function of the data only)."""

    port: int
    n: int
    gamma: float
    mu: np.ndarray
    sigma: np.ndarray
    sigma_ridge: np.ndarray
    threshold: ReturnThreshold

    @property
    def r_bar(self) -> float:
        return self.threshold.r_bar

    @property
    def instance_id(self) -> str:
        return f"orlib/port{self.port}"


def bcw_instance(
    port: int, data_dir: Path | str | None = None, share: float = RETURN_SHARE
) -> BcwInstance:
    """``port<k>.txt`` of ``data_dir`` (default ``data/raw/orlib``) as the data of (34), with the ridge
    folded in and ``r̄`` computed (module docstring). ``ValueError`` for a port outside 1-5."""
    if port not in ORLIB_N:
        raise ValueError(f"port must be in {sorted(ORLIB_N)}, got {port!r}")
    raw = load_orlib(port, data_dir)
    gamma = bcw_gamma(raw.n)
    sigma_ridge = raw.sigma + np.eye(raw.n) / gamma
    return BcwInstance(port, raw.n, gamma, raw.mu, raw.sigma, sigma_ridge,
                       return_threshold(raw.mu, sigma_ridge, gamma, share))


def bcw_problem(inst: BcwInstance, k: int, kappa: float) -> PortfolioProblem:
    """(34) as a :class:`PortfolioProblem`: ``f(w) = −κμᵀw + ½wᵀΣ'w`` on ``{Σw = 1, 0 ≤ w ≤ 1}`` with at most
    ``k`` assets. ``κ = 1`` is Table 3; ``κ = 0`` (``problem.mu`` is zero) is Table 4, whose minimum return
    ``μᵀw ≥ r̄`` is not part of the problem: it goes to the solver as its own row (``linear_ge``)."""
    if not 1 <= k <= inst.n:
        raise ValueError(f"cardinality must lie in [1, {inst.n}], got {k}")
    return PortfolioProblem(
        mu=kappa * inst.mu, sigma=inst.sigma_ridge, asset_names=tuple(f"port{inst.port}_{i + 1}" for i in range(inst.n)),
        w_min=0.0, w_max=1.0, budget=1.0, risk_aversion=1.0, cardinality=int(k),
    )


def bcw_objective(inst: BcwInstance, x: np.ndarray, kappa: float) -> float:
    """The objective of (34) in the form of the paper: ``½xᵀΣx + (1/2γ)‖x‖² − κμᵀx`` with the raw ``Σ``."""
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ inst.sigma @ x + (0.5 / inst.gamma) * (x @ x) - kappa * (inst.mu @ x))
