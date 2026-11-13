"""Outer approximation (multi-tree cutting planes) for the cardinality-constrained MIQP (E2, RQ2).

Exact baseline after Bertsimas & Cory-Wright (2022): the convex function of the binaries is
minimised by a master MILP over ``(z, θ)`` in HiGHS that is refined with subgradient cuts, one
subproblem per master solution. The scheme is multi-tree: HiGHS has no lazy constraints, so the
master is solved afresh after every cut. :func:`support_cut` is the cut, :func:`solve_oa` the loop.

Problem (minimisation; ``l = w_min``, ``u = min(w_max, budget)``, ``b = budget``)::

    minimize   f(w) = −μᵀw + (λ/2)·wᵀΣw
    subject to Σw = b,  l·z ≤ w ≤ u·z,  Σz ≤ K  (``card_mode="le"``) or Σz = K (``"eq"``),
               z ∈ {0, 1}^N.

There is no ridge term. The convexity in ``z`` comes from the perspective split of
:mod:`cif.classical.scip_native`, ``Σ = R + diag(d)`` with ``d ≥ 0`` and ``R ⪰ 0`` (``d`` from
``"mineig"`` or ``"sdp"``). With the perspective ``φ(w_i, z_i) = w_i²/z_i`` for ``z_i > 0``,
``φ(0, 0) = 0`` and ``+∞`` otherwise,

    F(w, z) = −μᵀw + (λ/2)·(wᵀRw + Σ_i d_i·φ(w_i, z_i))

is jointly convex in ``(w, z)`` and equals ``f(w)`` whenever ``w_i = 0`` for ``z_i = 0``. The
constraints ``Σw = b``, ``l·z ≤ w ≤ u·z`` are jointly linear, so

    f(z) = min_w { F(w, z) : Σw = b, l·z ≤ w ≤ u·z }

is convex on ``[0, 1]^N`` (partial minimisation). At a binary ``ẑ`` it is the optimum of the
convex refinement on ``S = supp(ẑ)``, solved by :mod:`cif.classical.refine`. For fractional ``ẑ``
the subproblem is the same QP with ``Σ`` replaced by ``Σ + diag(d·(1/ẑ − 1))`` and the bounds by
``l·ẑ ≤ w ≤ u·ẑ``.

Cut. Let ``(w*, ν, α, β)`` be a KKT point of the subproblem at ``ẑ`` (``ν`` for ``Σw = b``,
``α ≥ 0`` for ``w ≤ u·z``, ``β ≥ 0`` for ``w ≥ l·z``; :class:`cif.classical.refine.RefineDuals`)
and ``L(w, z) = F(w, z) + ν·(1ᵀw − b) + αᵀ(w − u·z) + βᵀ(l·z − w)``. For every feasible ``(w, z)``
``F ≥ L``. If ``(0, g) ∈ ∂L(w*, ẑ)``, i.e. the subgradient has no ``w``-component, then
``L(w, z) ≥ L(w*, ẑ) + gᵀ(z − ẑ)`` for all ``(w, z)``, and ``L(w*, ẑ) = f(ẑ)`` by
complementary slackness. Minimising over ``w`` gives the valid cut

    θ ≥ f(z) ≥ f(ẑ) + gᵀ(z − ẑ)          for all z ∈ [0, 1]^N.

The ``w``-component vanishes by stationarity of the multipliers, coordinate by coordinate:

* ``ẑ_i > 0``. ``φ`` is differentiable there, ``∂φ = (2w_i/z_i, −w_i²/z_i²)``, and stationarity is
  the KKT condition of the QP above. The slope is
  ``g_i = −u·α_i + l·β_i − (λ/2)·d_i·(w*_i/ẑ_i)²``.
* ``ẑ_i = 0`` (then ``w*_i = 0``). ``∂φ(0, 0) = {(s, t): t ≤ −s²/4}`` and stationarity fixes
  ``s_i = 2·(r_i − α_i + β_i)/(λ·d_i)`` with the reduced return of the full problem

      r_i = μ_i − λ·(Σw*)_i − ν,

  the stationarity residual of asset ``i`` under the multiplier ``ν`` of the budget row (for
  ``d_i = 0`` it reads ``α_i − β_i = r_i``). The multipliers ``α_i, β_i ≥ 0`` of the linking
  rows of out-of-support assets do not change ``L(w*, ẑ)`` and are free, so they are chosen to
  maximise the slope: ``g_i = max_{α_i, β_i ≥ 0} −u·α_i + l·β_i − (r_i − α_i + β_i)²/(2λ·d_i)``,

      g_i = min_{s ∈ [l, u]} { (λ/2)·d_i·s² − r_i·s }.

  This is the exact one-sided derivative of ``f`` along ``e_i``: the gain of buying the asset at
  the margin against the own-variance ``d_i·s²`` that the perspective keeps. For ``d_i = 0`` it
  is the linking-row cut ``−u·r_i⁺ + l·r_i⁻`` of plain outer approximation.

Validity needs only ``d ≥ 0``, ``R ⪰ 0`` and the KKT conditions of the subproblem; the stronger
``d`` (``"sdp"``) gives larger ``d_i`` and tighter cuts. The tests check the inequality on all
binary points of small instances, on fractional points against CVXPY, and the slopes against
finite differences.

Master and loop (:func:`solve_oa`)::

    minimize θ  subject to  θ ≥ f(ẑ_t) − δ + g_tᵀ(z − ẑ_t)  (t = 1…T),  θ ≥ f_cont,
                            k_lo ≤ Σz ≤ k_hi,  z ∈ {0, 1}^N,

with ``k_lo = ⌈b/u⌉`` and ``k_hi = min(K, ⌊b/l⌋)`` (``k_lo = k_hi = K`` for ``"eq"``). The
cardinality row keeps every master support feasible for the subproblem (``u·|S| ≥ b`` and
``l·|S| ≤ b``), so no feasibility cuts are needed. ``f_cont`` is the optimum of the relaxation
without cardinality and minimum purchase, a valid lower bound on every ``f(z)``.

One iteration solves the master (the lower bound ``LB`` is HiGHS's dual bound), evaluates ``f`` at
the master's support (the upper bound ``UB`` is the best value so far) and adds the cut at it. The
loop ends when ``UB − LB`` is within the tolerance, when the master returns a support that was
already evaluated (nothing below ``UB`` is left), or at the time limit: ``time_limit_s`` for the
loop alone, ``budget_s`` for the end-to-end budget of the RQ2 protocol, in which the diagonal / SDP,
the relaxation and the master are counted too, on the clock that the QUBO loop and the local search
share (``clock_offset_s`` is the part of it spent on the common start). The incumbent is passed to
HiGHS as a MIP start, so every master solve is cut off at ``UB``.

Root phase (``root_cuts=True``, the default). Cuts at binary points alone leave the master a poor
model of ``f``: the lower bound moves only when the loop happens to visit a support that cheats the
model, and the number of master solves explodes beyond a few dozen assets. So, before the master
becomes binary, its LP relaxation (``z ∈ [0, 1]^N``) is cut at its own optimum, again and again
(Kelley's method). The evaluation point is the LP solution without its dust
(:func:`_evaluation_point`), which keeps the subproblems small; the LP value is a lower bound at
every step and converges from below to the Boolean relaxation of the perspective reformulation,
the SOC lower bound of Bertsimas & Cory-Wright. All root cuts stay in the master. A binary LP
solution is a portfolio and may improve the incumbent.

Numerics. ``μ`` and ``Σ`` are divided by ``mean(diag Σ)`` inside, so the interior-point and HiGHS
tolerances act relative to the variance scale (the units of ORlib and of annualised data alike);
bounds and trajectories are reported in the original units. The split ``d`` is computed from the
original ``Σ`` by the function of :mod:`cif.classical.scip_native`, so the SCIP baseline and this
loop share it (that SDP is posed on ``Σ/λ_max(Σ)`` and does not depend on the units), and scaled
afterwards. Every cut is lowered by
``δ = 1e-9`` (normalised units) to absorb the tolerances of the interior-point duals and of the
master, so a reported dual bound stays a valid lower bound up to that margin; at termination
``UB − LB`` is therefore never below ``δ``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import highspy
import numpy as np

from cif.classical.refine import refine_support
from cif.classical.scip_native import (
    DiagMethod,
    _budget_left,
    _check_budget,
    _diagonal_shift,
    _solve_limit,
)
from cif.problem import PortfolioProblem, Solution

CardMode = Literal["le", "eq"]

_CARD_MODES = ("le", "eq")
_ZERO_Z = 1e-9  # entries of ẑ at or below this count as exact zeros (d/ẑ would blow up)
_CUT_SLACK = 1e-9  # every cut is lowered by this much (normalised units), see the docstring
_MASTER_TOL = 1e-9  # HiGHS feasibility tolerances and absolute MIP gap (normalised units)
_MASTER_OPTIONS: dict[str, float | bool | str] = {
    "mip_rel_gap": 0.0,
    "mip_abs_gap": _MASTER_TOL,
    "mip_feasibility_tolerance": _MASTER_TOL,
    "primal_feasibility_tolerance": _MASTER_TOL,
}
_NONZERO_TOL = 1e-6  # support threshold for n_nonzero and the feasibility checks (as scip_native)
_ROOT_DUST = 1e-6  # LP entries below this are zeros in the root evaluation point
_ROOT_SNAP = 1e-8  # LP entries this close to 1 are 1
_ROOT_TOL = 1e-6  # root phase stops when (best relaxed value − LP bound) ≤ this · |value|
_ROOT_MAX_ITER = 1000


@dataclass(frozen=True)
class Cut:
    """Linear minorant ``f(z) ≥ value + gradient·(z − z_hat)`` of the convex envelope of ``f``."""

    z_hat: np.ndarray  # point of the cut (entries ≤ 1e-9 set to 0)
    value: float  # f(ẑ)
    gradient: np.ndarray  # g, one-sided on the coordinates with ẑ_i = 0
    weights: np.ndarray  # minimiser w*(ẑ) of the subproblem, zeros outside supp(ẑ)
    time_s: float  # wall time of the subproblem and the multipliers

    def __call__(self, z: np.ndarray) -> float:
        """Value of the linear minorant at ``z``."""
        return float(self.value + self.gradient @ (np.asarray(z, dtype=float) - self.z_hat))


def _entry_slope(r: np.ndarray, d: np.ndarray, risk_aversion: float, lower: float, upper: float):
    """``min_{s ∈ [lower, upper]} (λ/2)·d·s² − r·s`` elementwise (convex in s: clipped stationary
    point, or an endpoint where ``d = 0``)."""
    quad = 0.5 * risk_aversion * d
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(quad > 0.0, r / (2.0 * quad), np.where(r > 0.0, upper, lower))
    s = np.clip(s, lower, upper)
    return quad * s * s - r * s


def support_cut(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    d: np.ndarray,
    z_hat: np.ndarray,
    lower: float,
    upper: float,
    budget: float = 1.0,
) -> Cut:
    """Subgradient cut of ``f`` at ``ẑ ∈ [0, 1]^N`` (module docstring).

    ``d ≥ 0`` is the diagonal of the split ``Σ = R + diag(d)``, ``R ⪰ 0``; ``lower``/``upper`` are
    the scalar bounds ``l`` and ``u``. The subproblem is solved by the shared refinement. Raises
    ``ValueError`` for malformed input and ``RuntimeError`` when the subproblem is infeasible
    (``u·Σẑ < b`` or ``l·Σẑ > b``) or the solver fails."""
    t0 = time.perf_counter()
    mu = np.asarray(mu, dtype=float)
    n = mu.shape[0]
    d = np.asarray(d, dtype=float)
    z = np.asarray(z_hat, dtype=float)
    if z.shape != (n,) or d.shape != (n,):
        raise ValueError(f"z_hat and d must have shape ({n},), got {z.shape} and {d.shape}")
    if np.any(d < 0.0):
        raise ValueError("d must be non-negative")
    if np.any(z < -1e-12) or np.any(z > 1.0 + 1e-12):
        raise ValueError("z_hat must lie in [0, 1]^N")
    z = np.where(z <= _ZERO_Z, 0.0, np.minimum(z, 1.0))
    active = np.flatnonzero(z)

    sigma = np.asarray(sigma, dtype=float)
    extra = d[active] * (1.0 / z[active] - 1.0)  # R + diag(d/ẑ) = Σ + diag(d·(1/ẑ − 1))
    sigma_z = sigma
    if np.any(extra != 0.0):
        sigma_z = sigma.copy()
        sigma_z[active, active] += extra
    res = refine_support(mu, sigma_z, risk_aversion, active, lower=lower * z, upper=upper * z,
                         budget=budget, return_duals=True)
    if res.status != "optimal" or res.duals is None:
        raise RuntimeError(f"subproblem at z_hat is {res.status}")

    w, duals = res.weights, res.duals
    gradient = np.zeros(n)
    gradient[active] = (
        -upper * duals.upper[active]
        + lower * duals.lower[active]
        - 0.5 * risk_aversion * d[active] * (w[active] / z[active]) ** 2
    )
    out = np.flatnonzero(z == 0.0)
    if out.size:
        sigma_w = sigma[np.ix_(out, active)] @ w[active]
        reduced_return = mu[out] - risk_aversion * sigma_w - duals.budget
        gradient[out] = _entry_slope(reduced_return, d[out], risk_aversion, lower, upper)
    return Cut(z, res.objective, gradient, w, time.perf_counter() - t0)


def _evaluation_point(z: np.ndarray, upper: float, budget: float) -> np.ndarray:
    """Point of the root cut taken from an LP solution ``z``: entries below ``_ROOT_DUST`` are set
    to 0 (``d/ẑ`` would blow up and the subproblem would be solved on all N assets), entries within
    ``_ROOT_SNAP`` of 1 to 1. Any point of ``[0, 1]^N`` carries a valid cut; if dropping the dust
    leaves ``u·Σẑ < b`` (no feasible subproblem) the dust is lifted to ``_ROOT_DUST`` instead."""
    z = np.minimum(np.asarray(z, dtype=float), 1.0)
    snapped = np.where(z > 1.0 - _ROOT_SNAP, 1.0, z)
    cleaned = np.where(z < _ROOT_DUST, 0.0, snapped)
    if upper * cleaned.sum() < budget * (1.0 + 1e-9):
        cleaned = np.clip(snapped, _ROOT_DUST, 1.0)
    return cleaned


class _CutPool:
    """The cuts ``θ ≥ a_t + G_t·z`` as arrays: the master's objective at a given ``z``."""

    def __init__(self, n: int) -> None:
        self._g = np.empty((64, n))
        self._a = np.empty(64)
        self.size = 0

    def add(self, gradient: np.ndarray, intercept: float) -> None:
        if self.size == self._a.shape[0]:
            self._g = np.vstack([self._g, np.empty_like(self._g)])
            self._a = np.concatenate([self._a, np.empty_like(self._a)])
        self._g[self.size], self._a[self.size] = gradient, intercept
        self.size += 1

    def value(self, z: np.ndarray) -> float:
        """The largest cut at ``z``, i.e. the value of the master there."""
        return float(np.max(self._a[: self.size] + self._g[: self.size] @ z))


def _relative_gap(primal: float, dual: float) -> float:
    """SCIP's gap ``|p − d|/min(|p|, |d|)``: 0 for equal bounds, ``inf`` if one is infinite, zero or
    of opposite sign. The same definition as ``solver_meta["gap"]`` of the SCIP baseline, so that
    :func:`cif.metrics.gap.reference_from_exact` reads both."""
    if primal == dual:
        return 0.0
    if not (math.isfinite(primal) and math.isfinite(dual)) or primal * dual <= 0.0:
        return math.inf
    return abs(primal - dual) / min(abs(primal), abs(dual))


def solve_oa(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    *,
    card_mode: CardMode = "le",
    diag: DiagMethod = "mineig",
    time_limit_s: float | None = None,
    budget_s: float | None = None,
    clock_offset_s: float = 0.0,
    threads: int = 1,
    mip_gap: float = 1e-6,
    abs_gap: float = 0.0,
    warm_start: Sequence[int] | None = None,
    root_cuts: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> Solution:
    """Solve the cardinality-constrained MIQP by multi-tree outer approximation.

    Parameters
    ----------
    cardinality:
        ``K``. If ``None``, ``problem.cardinality`` is used (``None`` there — no limit).
    card_mode:
        ``"le"``: ``Σz ≤ K``; ``"eq"``: ``Σz = K`` (ORlib with the Chang et al. bounds, where every
        held asset carries at least ``w_min``).
    diag:
        Diagonal ``d`` of the perspective split: ``"mineig"`` or ``"sdp"`` (as in
        :func:`cif.classical.scip_native.solve_miqp_native`). Its time is part of ``build_time_s``.
    time_limit_s:
        Limit for the loop (master solves and subproblems); ``None`` — no limit. As for the SCIP
        baseline it does not include the build (the diagonal, the relaxation, the model, the warm
        start). The limit of certification runs; the RQ2 protocol uses ``budget_s``.
    budget_s:
        End-to-end budget in seconds of the common clock (ANALYSIS_PLAN §5-§6, as ``budget_s`` of the
        QUBO loop and the local search and of the SCIP baseline); ``None`` — none. The SDP of
        ``diag="sdp"`` gets what is left at its start as its time limit, and the loop (root phase
        included) gets what is left after the build: HiGHS is given that remainder as its
        ``time_limit`` before every solve, and nothing starts once it is spent (the warm start is
        then the result). With ``time_limit_s`` too, the smaller remaining time binds. A master solve
        or a subproblem may overrun the end by its own granularity and the build cannot be
        interrupted: ``Solution.wall_time_s`` shows where the call ended on the clock.
    clock_offset_s:
        Seconds of the common clock spent before the call, the construction of the common start
        (``CommonStart.time_s``). Charged to ``budget_s``, added to every time of the trajectories,
        ``t_primal_s``, ``t_cert_s`` and ``Solution.wall_time_s``; it changes nothing else.
        ``build_time_s`` and ``solve_time_s`` stay durations of this call.
    threads:
        Must be 1 (ANALYSIS_PLAN §6); kept explicit for the registry.
    mip_gap, abs_gap:
        The loop stops when the SCIP-style relative gap ``|UB − LB|/min(|UB|, |LB|)`` is at most
        ``mip_gap`` or ``UB − LB`` is at most ``abs_gap`` (original units). With both at zero it
        runs until the master returns a support that was already evaluated.
    warm_start:
        Indices of an incumbent support (the common start of ANALYSIS_PLAN §5): it is evaluated, is
        the first incumbent and the first cut. A support that is infeasible for the model (size
        outside ``[k_lo, k_hi]``, or without a feasible subproblem) is ignored;
        ``solver_meta["warm_start_used"]`` says whether it was accepted.
    root_cuts:
        Before the first integer master, run cutting planes on its LP relaxation: solve the LP,
        cut at its solution (fractional points allowed, see :func:`support_cut`), repeat until the
        LP bound is within ``1e-6`` of the best relaxed value. The cuts stay in the master; the LP
        bound converges to the Boolean relaxation of the perspective reformulation (the SOC lower
        bound of Bertsimas & Cory-Wright). ``root_cuts=False`` is the bare loop with cuts at binary
        points only, which needs far more master solves already for a few dozen assets.
    seed:
        HiGHS ``random_seed``.

    Returns
    -------
    Solution
        ``objective`` is ``problem.objective_value(w)`` of the best weights from the shared
        refinement; ``wall_time_s`` is the end of the call on the common clock (``clock_offset_s``
        included). Without an incumbent: ``feasible=False``, ``objective=inf`` and zero weights.
        ``solver_meta`` holds

        * ``status``: ``"optimal"`` (gap closed, or the master has nothing below ``UB`` left),
          ``"timelimit"`` or ``"infeasible"`` (no support satisfies the constraints);
          ``time_limit_hit``;
        * ``primal_bound`` (UB), ``dual_bound`` (LB) and ``gap`` (SCIP's definition);
        * ``iterations`` (integer master solves), ``n_cuts``, ``n_refine`` (subproblems);
        * ``root_cuts``, ``root_iterations`` (LP solves of the root phase), ``root_time_s`` and
          ``root_dual_bound`` (LB after the root phase; ``None`` without it);
        * ``incumbent_trajectory`` and ``bound_trajectory``: lists of ``(t, value)`` with ``t`` the
          wall time on the common clock (build and ``clock_offset_s`` included), one entry per
          master iteration and one at the start (the warm start, if used; the bound after the root
          phase), so without a warm start the incumbent list is one shorter. Binary LP solutions of
          the root phase that improve the incumbent add entries as well. ``t_primal_s``: time of the
          last improvement of the incumbent; ``t_cert_s``: time at which the optimality was
          established (``None`` if never); both on the same clock;
        * ``build_time_s`` (diagonal, relaxation, master, warm start), ``solve_time_s`` and their
          CPU times, durations of this call; ``diag_time_s`` (part of the build), ``solve_limit_s``
          (the limit the loop started with: the smaller of ``time_limit_s`` and the budget's
          remainder after the build; ``None`` if neither is given); ``master_time_s`` (HiGHS),
          ``refine_time_s`` (all subproblems, the warm start included);
        * the inputs that define the run (``time_limit_s``, ``budget_s``, ``clock_offset_s``, ...),
          ``f_cont`` (relaxation without cardinality), the diagonal (``diag_method``,
          ``diag_fallback``, ``d_min``, ``d_max``) and the checks ``budget_ok``, ``box_ok``,
          ``card_ok``, ``n_nonzero``.
    """
    t_start = time.perf_counter()
    cpu_start = time.process_time()
    n = problem.n
    if card_mode not in _CARD_MODES:
        raise ValueError(f"card_mode must be one of {_CARD_MODES}, got {card_mode!r}")
    if threads != 1:
        raise ValueError(f"the study runs one thread: threads must be 1, got {threads}")
    if mip_gap < 0.0 or abs_gap < 0.0:
        raise ValueError(f"mip_gap and abs_gap must be non-negative, got {mip_gap}, {abs_gap}")
    if time_limit_s is not None and time_limit_s < 0.0:
        raise ValueError(f"time_limit_s must be non-negative, got {time_limit_s}")
    _check_budget(budget_s, clock_offset_s)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    card = cardinality if cardinality is not None else problem.cardinality
    if card is not None and card < 1:
        raise ValueError(f"cardinality must be ≥ 1, got {card}")
    start_support: np.ndarray | None = None
    if warm_start is not None:
        raw = np.asarray(warm_start)
        if raw.size and raw.dtype.kind not in "iu":  # a mask or a weight vector is a likely mistake
            raise ValueError(f"warm_start must hold integer asset indices, got dtype {raw.dtype}")
        start_support = np.unique(raw.astype(int))
        if start_support.size and (start_support[0] < 0 or start_support[-1] >= n):
            raise ValueError(f"warm_start indices must lie in [0, {n})")

    risk_aversion = float(problem.risk_aversion)
    lower, budget = float(problem.w_min), float(problem.budget)
    upper = float(min(problem.w_max, budget))
    k = n if card is None else card
    k_lo = max(1, math.ceil(budget / upper - 1e-12))  # u·|S| ≥ b
    k_hi = min(n, k, math.floor(budget / lower + 1e-12)) if lower > 0.0 else min(n, k)
    if card_mode == "eq":
        k_lo, k_hi = max(k_lo, k), min(k_hi, k)

    sigma = 0.5 * (problem.sigma + problem.sigma.T)
    scale = float(np.mean(np.diag(sigma)))
    if not scale > 0.0:
        scale = 1.0
    mu_n, sigma_n = problem.mu / scale, sigma / scale
    # The split is computed from the raw Σ, exactly as in scip_native, so that both use the same d.
    t_diag = time.perf_counter()
    d_raw, diag_used, diag_fallback = _diagonal_shift(
        sigma, diag, time_limit_s=_budget_left(budget_s, clock_offset_s, t_diag - t_start)
    )
    diag_time = time.perf_counter() - t_diag
    d = d_raw / scale

    meta: dict = {
        "formulation": "oa_multitree",
        "diag_method": diag_used,
        "diag_fallback": diag_fallback,
        "d_min": float(d_raw.min()) if d_raw.size else None,  # original units, as scip_native
        "d_max": float(d_raw.max()) if d_raw.size else None,
        "threads": threads,
        "cardinality": card,
        "card_mode": card_mode,
        "time_limit_s": time_limit_s,
        "budget_s": budget_s,
        "clock_offset_s": clock_offset_s,
        "solve_limit_s": None,  # set when the loop starts
        "diag_time_s": diag_time,  # part of build_time_s
        "mip_gap": mip_gap,
        "abs_gap": abs_gap,
        "seed": seed,
        "warm_start_used": False,
        "root_cuts": root_cuts,
        "root_iterations": 0,
        "root_time_s": 0.0,
        "root_dual_bound": None,
        "iterations": 0,
        "n_cuts": 0,
        "n_refine": 0,
        "master_time_s": 0.0,
        "refine_time_s": 0.0,
        "incumbent_trajectory": [],
        "bound_trajectory": [],
        "t_primal_s": None,
        "t_cert_s": None,
        "highs_version": highspy.Highs().version(),
    }

    def elapsed() -> float:  # seconds since the call began
        return time.perf_counter() - t_start

    def now() -> float:  # the same instant on the common clock
        return clock_offset_s + elapsed()

    def finish(status: str, weights: np.ndarray | None, ub: float, lb: float, t_solve: float,
               cpu_solve: float, build: tuple[float, float]) -> Solution:
        has_solution = weights is not None
        w = weights if has_solution else np.zeros(n)
        objective = problem.objective_value(w) if has_solution else math.inf
        primal = objective
        dual = lb * scale if math.isfinite(lb) else lb
        meta.update(
            {
                "status": status,
                "time_limit_hit": status == "timelimit",
                "primal_bound": primal,
                "dual_bound": dual,
                "gap": _relative_gap(primal, dual),
                "build_time_s": build[0],
                "build_cpu_s": build[1],
                "solve_time_s": t_solve,
                "solve_cpu_s": cpu_solve,
            }
        )
        nonzero = w > _NONZERO_TOL
        n_nonzero = int(nonzero.sum())
        feasible = False
        if has_solution:
            budget_ok = bool(abs(w.sum() - budget) < 1e-4)
            box_ok = bool(np.all(w >= -1e-6) and np.all(w <= problem.w_max + 1e-6)
                          and np.all(w[nonzero] >= lower - 1e-6))
            card_ok = n_nonzero <= k and (card_mode == "le" or lower == 0.0 or n_nonzero == k)
            meta.update({"budget_ok": budget_ok, "box_ok": box_ok, "card_ok": card_ok})
            feasible = budget_ok and box_ok and card_ok
        meta["n_nonzero"] = n_nonzero
        return Solution(w, objective, feasible, now(), "oa_highs", meta)

    build = (0.0, 0.0)
    if k_lo > k_hi:  # no support satisfies the cardinality, cap and minimum purchase together
        meta.update({"f_cont": None})
        build = (elapsed(), time.process_time() - cpu_start)
        return finish("infeasible", None, math.inf, math.inf, 0.0, 0.0, build)

    relaxed = refine_support(mu_n, sigma_n, risk_aversion, np.arange(n), lower=0.0, upper=upper,
                             budget=budget)
    if relaxed.status != "optimal":
        raise RuntimeError(f"relaxation without cardinality is {relaxed.status}")
    f_cont = relaxed.objective
    meta["f_cont"] = f_cont * scale

    inf = highspy.kHighsInf
    master = highspy.Highs()
    master.setOptionValue("output_flag", bool(verbose))
    master.setOptionValue("threads", threads)
    master.setOptionValue("random_seed", seed)
    for option, value in _MASTER_OPTIONS.items():
        master.setOptionValue(option, value)
    for _ in range(n):
        master.addVar(0.0, 1.0)  # z: continuous in the root phase, binary afterwards
    master.addVar(f_cont - _CUT_SLACK, inf)  # θ ≥ f_cont
    master.changeColCost(n, 1.0)
    columns = np.arange(n + 1, dtype=np.int32)
    master.addRow(float(k_lo), float(k_hi), n, columns[:n], np.ones(n))

    pool = _CutPool(n)
    visited: set[tuple[int, ...]] = set()

    def add_cut(cut: Cut) -> None:
        intercept = cut.value - _CUT_SLACK - float(cut.gradient @ cut.z_hat)
        master.addRow(intercept, inf, n + 1, columns, np.append(-cut.gradient, 1.0))
        pool.add(cut.gradient, intercept)
        meta["refine_time_s"] += cut.time_s
        meta["n_refine"] += 1
        meta["n_cuts"] = pool.size

    ub, lb = math.inf, f_cont
    best_weights: np.ndarray | None = None
    best_z: np.ndarray | None = None
    incumbent_trajectory: list[tuple[float, float]] = meta["incumbent_trajectory"]
    bound_trajectory: list[tuple[float, float]] = meta["bound_trajectory"]

    def record(improved: bool) -> None:
        t = now()
        if math.isfinite(ub):
            incumbent_trajectory.append((t, ub * scale))
        bound_trajectory.append((t, lb * scale))
        if improved:
            meta["t_primal_s"] = t

    if start_support is not None and k_lo <= start_support.size <= k_hi:
        z0 = np.zeros(n)
        z0[start_support] = 1.0
        try:
            cut = support_cut(mu_n, sigma_n, risk_aversion, d, z0, lower, upper, budget)
        except RuntimeError:
            cut = None  # no feasible subproblem: the start is not usable
        if cut is not None:
            add_cut(cut)
            visited.add(tuple(start_support.tolist()))
            ub, best_weights, best_z = cut.value, cut.weights, z0
            meta["warm_start_used"] = True
            meta["t_primal_s"] = now()
            incumbent_trajectory.append((meta["t_primal_s"], ub * scale))

    build = (elapsed(), time.process_time() - cpu_start)
    solve_start, cpu_solve_start = time.perf_counter(), time.process_time()
    abs_gap_n = abs_gap / scale
    left_at_start = _budget_left(budget_s, clock_offset_s, build[0])
    meta["solve_limit_s"] = _solve_limit(time_limit_s, left_at_start)

    def time_left() -> float | None:
        """Seconds the loop may still run: the smaller of ``time_limit_s`` (counted from the start
        of the loop) and what is left of the end-to-end budget (counted from the call's start)."""
        own = None if time_limit_s is None else time_limit_s - (time.perf_counter() - solve_start)
        left = _budget_left(budget_s, clock_offset_s, elapsed())
        limits = [x for x in (own, left) if x is not None]
        return min(limits) if limits else None

    if root_cuts:
        t_root, best_relaxed = time.perf_counter(), math.inf
        while meta["root_iterations"] < _ROOT_MAX_ITER:
            remaining = time_left()
            if remaining is not None:
                if remaining <= 0.0:
                    break
                master.setOptionValue("time_limit", remaining)
            t_lp = time.perf_counter()
            run_status = master.run()
            meta["master_time_s"] += time.perf_counter() - t_lp
            model_status = master.getModelStatus()
            if run_status == highspy.HighsStatus.kError or model_status not in (
                highspy.HighsModelStatus.kOptimal,
                highspy.HighsModelStatus.kTimeLimit,
            ):
                raise RuntimeError(f"HiGHS root LP: {master.modelStatusToString(model_status)}")
            if model_status == highspy.HighsModelStatus.kTimeLimit:
                break
            values = np.asarray(master.getSolution().col_value)
            theta = float(values[n])
            lb = max(lb, theta)  # LP value: a lower bound, the cuts are valid on all of [0, 1]^N
            meta["root_iterations"] += 1
            z_eval = _evaluation_point(values[:n], upper, budget)
            try:
                cut = support_cut(mu_n, sigma_n, risk_aversion, d, z_eval, lower, upper, budget)
            except RuntimeError:
                break  # the root phase only accelerates: a failing subproblem just ends it
            add_cut(cut)
            best_relaxed = min(best_relaxed, cut.value)
            if np.all((z_eval == 0.0) | (z_eval == 1.0)):  # a binary vertex is a portfolio
                visited.add(tuple(np.flatnonzero(z_eval).tolist()))
                if cut.value < ub:
                    ub, best_weights, best_z = cut.value, cut.weights, z_eval
                    meta["t_primal_s"] = now()
                    incumbent_trajectory.append((meta["t_primal_s"], ub * scale))
            if best_relaxed - theta <= _ROOT_TOL * max(abs(best_relaxed), 1e-12):
                break
        meta["root_time_s"] = time.perf_counter() - t_root
        meta["root_dual_bound"] = lb * scale
    for i in range(n):
        master.changeColIntegrality(i, highspy.HighsVarType.kInteger)
    bound_trajectory.append((now(), lb * scale))

    def gap_closed() -> bool:
        return ub - lb <= abs_gap_n or _relative_gap(ub, lb) <= mip_gap

    status = "timelimit"
    while True:
        if gap_closed():
            status = "optimal"
            meta["t_cert_s"] = now()
            break
        remaining = time_left()
        if remaining is not None and remaining <= 0.0:
            break
        if best_z is not None:  # MIP start: cuts off every master solution above the incumbent
            start = highspy.HighsSolution()
            start.col_value = [*best_z.tolist(), pool.value(best_z)]
            master.setSolution(start)
        if remaining is not None:
            master.setOptionValue("time_limit", remaining)
        t_master = time.perf_counter()
        run_status = master.run()
        meta["master_time_s"] += time.perf_counter() - t_master
        meta["iterations"] += 1
        model_status = master.getModelStatus()
        if run_status == highspy.HighsStatus.kError or model_status not in (
            highspy.HighsModelStatus.kOptimal,
            highspy.HighsModelStatus.kTimeLimit,
        ):
            raise RuntimeError(f"HiGHS master: {master.modelStatusToString(model_status)}")

        dual_bound = float(master.getInfo().mip_dual_bound)
        if math.isfinite(dual_bound):
            lb = max(lb, dual_bound)
        solution = master.getSolution()
        repeated = improved = False
        if solution.value_valid and len(solution.col_value) == n + 1:
            z_new = (np.asarray(solution.col_value[:n]) > 0.5).astype(float)
            repeated = tuple(np.flatnonzero(z_new).tolist()) in visited
            if not repeated:
                cut = support_cut(mu_n, sigma_n, risk_aversion, d, z_new, lower, upper, budget)
                add_cut(cut)
                visited.add(tuple(np.flatnonzero(z_new).tolist()))
                if cut.value < ub:
                    ub, best_weights, best_z = cut.value, cut.weights, z_new
                    improved = True
        record(improved)
        if verbose:
            print(f"[oa] it {meta['iterations']:4d}  t {now():8.3f}  LB {lb * scale:.10g}  "
                  f"UB {ub * scale:.10g}  cuts {pool.size}")
        if model_status == highspy.HighsModelStatus.kTimeLimit:
            if gap_closed():
                status, meta["t_cert_s"] = "optimal", now()
            break
        if repeated:  # the master's optimum is a support already evaluated: nothing below UB left
            status, meta["t_cert_s"] = "optimal", now()
            break
    return finish(status, best_weights, ub, lb, time.perf_counter() - solve_start,
                  time.process_time() - cpu_solve_start, build)
