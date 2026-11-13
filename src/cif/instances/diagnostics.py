"""Does the cardinality constraint bind? Instance diagnostics for R4 / WP-E3.

If the support of the *continuous* long-only optimum (no cardinality) already has at most K
assets, that optimum is feasible for the cardinality-constrained problem, so the discrete
problem is the continuous one (REVIEW R4: bootstrap Σ, ``n_nonzero`` 4-10 at K = 12…125). With
``s_cont`` the support size of the continuous optimum::

    binding = s_cont > K

and the converse holds when Σ ≻ 0: the continuous optimum is unique, so for ``s_cont > K`` it is
infeasible and the constrained optimum is strictly worse (checked by exhaustive search in the
tests). K can therefore be chosen as a fraction of ``s_cont`` (:func:`k_from_scont`).

The relaxation is ``min -μᵀw + (ra/2) wᵀΣw`` over ``Σw = budget, 0 ≤ w ≤ w_max``. A buy-in
threshold ``w_min > 0`` and the cardinality limit are dropped: ``solve_continuous_mvo`` would
impose ``w ≥ w_min`` on *all* assets (infeasible when ``N·w_min > budget``, e.g. port5 with the
Chang default 0.01), whereas the threshold only applies to held assets.

Numerics. The minimiser is invariant under ``(μ, Σ) → (μ, Σ)/s`` but the interior-point
solution is not: the problem is solved with ``s = mean(diag Σ)``. At the raw scale of
OR-Library (Σ ~ 1e-3) the default tolerances act like absolute ones (port1, λ = 1: weight error
1.6e-4 raw, 2.3e-8 normalised) and the support at ``1e-6`` was wrong (15 instead of 10). On the
normalised problem, over the 125 cells of the 25-point grids of
port1-5, the interior-point weight on assets that are exactly zero in an OSQP-polished
reference is ≤ 2.7e-6 (largest weight error 9e-6) and the smallest genuinely positive weight is
1.9e-5; ``support_tol = 1e-5`` reproduces the reference support on all 125 cells (and on 15
legacy dense cells). ``s_cont`` is not a certificate: a weight within a factor ~2 of the
tolerance can flip, which moves ``s_cont`` by 1.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from cif.classical.continuous import solve_continuous_mvo
from cif.problem import PortfolioProblem, Solution

#: Weights above this (in units of the budget) count as held; see the module docstring.
SUPPORT_TOL = 1e-5


def continuous_relaxation(problem: PortfolioProblem) -> Solution:
    """Cardinality-free, threshold-free relaxation of ``problem`` (``w_min = 0``) via
    :func:`solve_continuous_mvo` on the problem normalised by ``mean(diag Σ)``.

    The returned ``objective`` is evaluated on the original ``problem``; ``solver_meta['scale']``
    is the normalisation.
    """
    scale = float(np.mean(np.diag(problem.sigma)))
    if not scale > 0.0:
        raise ValueError("sigma must have a positive average variance")
    relaxed = replace(
        problem,
        mu=problem.mu / scale,
        sigma=problem.sigma / scale,
        w_min=0.0,
        cardinality=None,
    )
    sol = solve_continuous_mvo(relaxed)
    sol.objective = problem.objective_value(sol.weights)
    sol.solver_meta = {**sol.solver_meta, "scale": scale}
    return sol


def _support_size(sol: Solution, problem: PortfolioProblem, support_tol: float) -> int:
    return int((sol.weights > support_tol * problem.budget).sum())


def instance_statistics(mu: np.ndarray, sigma: np.ndarray) -> dict:
    """Cheap descriptive statistics of ``(μ, Σ)`` (no optimisation); JSON-safe.

    ``vol_*`` — ``√diag Σ`` (units of Σ); ``cv_*`` — population std over |mean| (``cv_vol`` of the
    volatilities ``√diag Σ``, ``cv_diag_sigma`` of the variances); ``corr_*`` — over the
    ``N(N - 1)/2`` off-diagonal correlations; ``cond_sigma`` — ``λ_max/λ_min``. Undefined
    quantities (zero mean, singular Σ, a single asset) are ``None``.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.shape[0]
    diag = np.diag(sigma)
    if not np.all(diag > 0.0):
        raise ValueError("sigma must have a positive diagonal")
    vol = np.sqrt(diag)

    eig = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    singular = eig[0] <= n * np.finfo(float).eps * eig[-1]
    mu_mean = float(mu.mean())

    stats: dict = {
        "n": int(n),
        "vol_min": float(vol.min()),
        "vol_median": float(np.median(vol)),
        "vol_max": float(vol.max()),
        "cv_diag_sigma": float(diag.std() / diag.mean()),
        "cv_vol": float(vol.std() / vol.mean()),
        "cv_mu": float(mu.std() / abs(mu_mean)) if mu_mean != 0.0 else None,
        "mu_min": float(mu.min()),
        "mu_mean": mu_mean,
        "mu_max": float(mu.max()),
        "cond_sigma": None if singular else float(eig[-1] / eig[0]),
    }
    if n >= 2:
        corr = sigma / np.outer(vol, vol)
        off = corr[np.triu_indices(n, k=1)]
        stats.update(
            corr_mean=float(off.mean()),
            corr_std=float(off.std()),
            corr_abs_mean=float(np.abs(off).mean()),
        )
    else:
        stats.update(corr_mean=None, corr_std=None, corr_abs_mean=None)
    return stats


def binding_diagnostics(
    problem: PortfolioProblem,
    K: int | None = None,
    support_tol: float = SUPPORT_TOL,
) -> dict:
    """Registry-row diagnostics: ``s_cont``, ``binding`` and the instance statistics.

    ``K`` defaults to ``problem.cardinality``; with neither, ``binding`` and ``K`` are ``None``.
    A weight counts as held when it exceeds ``support_tol · budget``. Keys: ``K``, ``s_cont`` (support of the continuous relaxation, see module docstring),
    ``s_cont_frac = s_cont/N``, ``binding = s_cont > K``, ``f_cont`` (relaxation objective,
    original units; a lower bound for any cardinality-constrained, thresholded variant), the
    keys of :func:`instance_statistics`, and ``cont_solver`` / ``cont_status`` / ``cont_wall_s``
    of the continuous solve (``cont_status = 'optimal_inaccurate'`` means ``s_cont`` is less
    reliable).
    """
    k = problem.cardinality if K is None else K
    if k is not None:
        k = int(k)
        if not 1 <= k <= problem.n:
            raise ValueError(f"K must be in [1, N={problem.n}], got {k}")
    sol = continuous_relaxation(problem)
    s_cont = _support_size(sol, problem, support_tol)
    row = instance_statistics(problem.mu, problem.sigma)
    row.update(
        {
            "K": k,
            "s_cont": s_cont,
            "s_cont_frac": s_cont / problem.n,
            "binding": None if k is None else bool(s_cont > k),
            "f_cont": float(sol.objective),
            "cont_solver": sol.solver,
            "cont_status": sol.solver_meta.get("status"),
            "cont_wall_s": float(sol.wall_time_s),
        }
    )
    return row


def k_from_scont(
    problem: PortfolioProblem,
    fraction: float,
    *,
    s_cont: int | None = None,
    support_tol: float = SUPPORT_TOL,
) -> int:
    """``K = max(1, min(N, ⌊fraction · s_cont⌋))``.

    Floor, so every ``fraction < 1`` binds (``K < s_cont``) unless ``s_cont = 1``;
    ``fraction = 1`` is the boundary ``K = s_cont`` (not binding) and ``fraction > 1`` is slack.
    Pass a precomputed ``s_cont`` to avoid the solve.
    """
    if not (fraction > 0.0 and math.isfinite(fraction)):
        raise ValueError(f"fraction must be a positive finite number, got {fraction!r}")
    if s_cont is None:
        s_cont = _support_size(continuous_relaxation(problem), problem, support_tol)
    k = math.floor(fraction * s_cont + 1e-9)
    return max(1, min(problem.n, k))


def legacy_dense_instance(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The Experiment A generator (``make_synthetic_problem`` in
    ``scripts/experiment_a_scalability.py``): ``μ ~ U(0.05, 0.25)``, ``Σ = 0.005·AAᵀ + 0.01·I``,
    ``A ~ N(0, 1)^{N x N}``. Same draw order, hence bitwise-identical output for the same seed."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.005 + np.eye(n) * 0.01
    return mu, sigma


def describe_legacy_dense(n: int, seed: int) -> dict:
    """:func:`instance_statistics` of the legacy dense generator.

    ``E[Σ_ii] = 0.005 N + 0.01``, so volatility grows as ``√(0.005 N + 0.01)`` (≈ 33 % at
    N = 20, ≈ 100 % at N = 200) and the correlations ``a_i·a_j/(‖a_i‖‖a_j‖)`` are ~N(0, 1/N):
    mean ≈ 0, mean |corr| ≈ 0.8/√N. Neither resembles an equity covariance matrix (REVIEW R4, item 15).
    """
    mu, sigma = legacy_dense_instance(n, seed)
    return {"family": "dense", "seed": int(seed), **instance_statistics(mu, sigma)}
