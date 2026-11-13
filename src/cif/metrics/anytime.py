"""Anytime metrics of RQ2: primal integral, time-to-target, Kaplan–Meier median (ANALYSIS_PLAN §5).

A method that can be stopped at any time leaves an **incumbent trajectory**: a list of ``(t, f)``
pairs, ``f`` the true objective ``−μᵀw + (λ/2)·wᵀΣw`` of the best portfolio known at the end-to-end
time ``t`` (seconds from the beginning of the common clock: model or QUBO construction, solution
and refinement all count). Entries come in chronological order; ``f`` need not decrease, the
incumbent is the running minimum (:func:`incumbent_curve`).

With ``ĝ(f) = (f − f_ref)/s_ref`` (:func:`cif.metrics.gap.relative_gap` with
``scale=s_ref``; the reference is the best value known over all methods) and the budget ``T``:

* **primal integral** ``PI(T) = (1/T)·∫₀ᵀ γ(t) dt``, ``γ(t) = min(1, ĝ(f_inc(t)))``
  (:func:`primal_integral`);
* **time-to-target** (:func:`time_to_target`): the first time with ``ĝ(f_inc(t)) ≤ τ``; a run that
  does not get there by the budget is censored at the budget;
* the **median over seeds** of a censored time is the Kaplan–Meier median
  (:func:`kaplan_meier`, :func:`km_median`).

**Common start (§5, N5).** Top-K rounding of the continuous relaxation with refinement is handed to
every method as a warm start and its construction time is charged to all of them. Pass it as
``start=(t₀, f₀)``: it is the first incumbent, ``γ = 1`` before ``t₀`` and the incumbent never gets
worse than ``f₀`` afterwards, so ``γ ≤ min(1, ĝ(f₀))`` from ``t₀`` on. Solutions of the method
itself timed before ``t₀`` join the incumbent at ``t₀``.

**PI⁺ (§12, A2 of 2026-10-11) is the primary measure of RQ2.** ``PI⁺(T) = (1/T)·∫_{t₀}^{T} γ(t) dt =
PI(T) − t₀/T`` (:func:`primal_integral_after_start`). On ``[0, t₀)`` ``γ = 1`` for every method alike, so
``t₀/T`` is a term common to all methods: with it ``PI(T) ≥ t₀/T`` and a ratio of two PIs is bounded by
``1 + ĝ₀·(T − t₀)/t₀`` (``ĝ₀`` the gap of the start), however different the methods are. On an instance
whose start is optimal ``PI = t₀/T`` for everyone (not 0: the integral counts the construction of the
start) and ``PI⁺ = 0`` for everyone, a 0/0 ratio: such instances are excluded by the eligibility rule
(:mod:`cif.experiments.e2_eligibility`), and on the others ``PI⁺ > 0`` for every method, because
``γ(t₀) = ĝ₀ > 0`` and no method improves at ``t₀`` itself.

**Sensitivity variants (§12, A4).** Without ``start`` the same functions give "``γ = 1`` until the
method's first solution". Fed the trajectory of a warm run, which lists only the improvements over the
common start, this is "``γ = 1`` until the first improvement of the common start"; fed the full
trajectory of a cold run (no common start, the method's own first solution is its first entry) it is
"``γ = 1`` until the method's own first solution".

``γ`` is clipped to ``[0, 1]`` from below too, which matters only when the reference is not the best
value known (a certified optimum can be beaten by numerical noise); with the best known value over
all methods as ``f_ref`` the clip is inactive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

__all__ = [
    "TAU_MAIN",
    "TAU_SENSITIVITY",
    "TimeToTarget",
    "Trajectory",
    "incumbent_curve",
    "kaplan_meier",
    "km_median",
    "km_quantile",
    "primal_integral",
    "primal_integral_after_start",
    "time_to_target",
]

#: A trajectory: ``(t, f)`` pairs, ``t`` non-decreasing; JSON-style lists of lists are accepted.
Trajectory = Sequence[Sequence[float]]

#: Target gap of time-to-target: main and sensitivity value (§5).
TAU_MAIN = 1e-3
TAU_SENSITIVITY = 1e-4

#: S(t) ≤ 1/2 is tested with this slack: a product of ratios may land one ulp above 0.5.
_HALF_TOL = 1e-12


class TimeToTarget(NamedTuple):
    """``time_s``: the time the target was reached, or the budget if it was not (``reached`` is
    ``False`` then: the run is censored there)."""

    time_s: float
    reached: bool


def _pair(entry: Sequence[float], what: str) -> tuple[float, float]:
    """A validated ``(t, f)`` pair: finite ``t ≥ 0``; ``f`` finite or ``+inf`` (no solution)."""
    try:
        t, f = entry
        t, f = float(t), float(f)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be a (time, objective) pair, got {entry!r}") from exc
    if not (math.isfinite(t) and t >= 0.0):
        raise ValueError(f"{what}: the time must be finite and non-negative, got {t}")
    if math.isnan(f) or f == -math.inf:
        raise ValueError(f"{what}: the objective must be finite or +inf (no solution), got {f}")
    return t, f


def incumbent_curve(
    trajectory: Trajectory, *, start: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Breakpoints ``(times, values)`` of the incumbent ``f_inc(t)``, a right-continuous step
    function with strictly decreasing ``values`` (``f_inc = values[k]`` from ``times[k]`` on).

    The running minimum of the trajectory: entries that do not improve on the incumbent, and
    ``+inf`` entries, are dropped; entries at one time keep the lowest value. ``start = (t₀, f₀)``
    is added as an entry, and entries timed before ``t₀`` count from ``t₀``. Before the first
    breakpoint there is no incumbent; an empty trajectory without start gives empty arrays.
    A start with ``f₀ = +inf`` (the construction failed) is the same as no start.

    Raises ``ValueError`` for times that decrease along the trajectory, negative or non-finite
    times, a NaN or ``−inf`` objective and entries that are not pairs.
    """
    events: list[tuple[float, float]] = []
    previous = 0.0
    for entry in trajectory:
        t, f = _pair(entry, "trajectory entry")
        if t < previous:
            raise ValueError(f"trajectory times must not decrease, got {t} after {previous}")
        previous = t
        events.append((t, f))
    if start is not None:
        t0, f0 = _pair(start, "start")
        events = [(max(t, t0), f) for t, f in events]
        events.append((t0, f0))
        events.sort(key=lambda event: event[0])  # stable: equal times keep their order
    times: list[float] = []
    values: list[float] = []
    best = math.inf
    for t, f in events:
        if not f < best:
            continue
        if times and t == times[-1]:
            values[-1] = f
        else:
            times.append(t)
            values.append(f)
        best = f
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _check_reference(f_ref: float, s_ref: float) -> None:
    if not math.isfinite(f_ref):
        raise ValueError(f"f_ref must be finite, got {f_ref}")
    if not (math.isfinite(s_ref) and s_ref > 0.0):
        raise ValueError(f"s_ref must be a finite positive number, got {s_ref}")


def _check_budget(budget_s: float) -> None:
    if not (math.isfinite(budget_s) and budget_s > 0.0):
        raise ValueError(f"budget_s must be a finite positive number, got {budget_s}")


def primal_integral(
    trajectory: Trajectory,
    *,
    f_ref: float,
    s_ref: float,
    budget_s: float,
    start: tuple[float, float] | None = None,
) -> float:
    """``PI(T) = (1/T)·∫₀ᵀ γ(t) dt`` with ``γ = min(1, ĝ(f_inc))`` and ``T = budget_s`` (§5).

    ``ĝ(f) = (f − f_ref)/s_ref``; ``γ = 1`` while there is no incumbent (before the start, or
    before the first own solution when ``start`` is ``None``). The value is in ``[0, 1]``: 0 for an
    optimal incumbent from the very beginning, 1 for no incumbent within the budget. Events after
    the budget are ignored. See the module docstring for ``start``.
    """
    _check_reference(f_ref, s_ref)
    _check_budget(budget_s)
    times, values = incumbent_curve(trajectory, start=start)
    if times.size == 0:
        return 1.0
    gamma = np.clip((values - f_ref) / s_ref, 0.0, 1.0)
    edges = np.minimum(np.append(times, budget_s), budget_s)
    # γ = 1 on [0, edges[0]); γ[k] on [edges[k], edges[k + 1]); the last piece ends at the budget
    area = edges[0] + float(np.dot(gamma, np.diff(edges)))
    return area / budget_s


def primal_integral_after_start(
    trajectory: Trajectory,
    *,
    f_ref: float,
    s_ref: float,
    budget_s: float,
    start: tuple[float, float],
) -> float:
    """``PI⁺(T) = (1/T)·∫_{t₀}^{T} γ(t) dt = PI(T) − t₀/T``, ``t₀ = start[0]`` (§12, A2).

    The primal integral after the common start: the interval ``[0, t₀)``, on which ``γ = 1`` for every
    method, is left out (module docstring). ``start`` is required, as in :func:`primal_integral`
    ``(t₀, f₀)``; a start whose construction failed (``f₀ = +inf``) still fixes ``t₀``, and ``γ = 1``
    runs from ``t₀`` until the first own solution. The value is in ``[0, 1 − t₀/T]``: 0 if the start is
    optimal, and 0 as well when the start has used the whole budget (``t₀ ≥ T``: nothing is left to
    integrate). Events after the budget are ignored.
    """
    _check_reference(f_ref, s_ref)
    _check_budget(budget_s)
    t0, _ = _pair(start, "start")
    times, values = incumbent_curve(trajectory, start=start)
    gamma = np.concatenate(([1.0], np.clip((values - f_ref) / s_ref, 0.0, 1.0)))
    edges = np.concatenate(([0.0], times, [budget_s]))
    edges = np.clip(edges, min(t0, budget_s), budget_s)  # γ pieces left of t₀ get zero width
    return float(np.dot(gamma, np.diff(edges))) / budget_s


def time_to_target(
    trajectory: Trajectory,
    *,
    f_ref: float,
    s_ref: float,
    tau: float,
    budget_s: float,
    start: tuple[float, float] | None = None,
) -> TimeToTarget:
    """First time with ``ĝ(f_inc(t)) ≤ τ`` (``τ = TAU_MAIN`` or ``TAU_SENSITIVITY``), censored.

    ``TimeToTarget(t, True)`` if the target is reached at ``t ≤ budget_s``, otherwise
    ``TimeToTarget(budget_s, False)``: the run is censored at the budget, not dropped. ``start``
    as in :func:`primal_integral` (the start itself may reach the target).
    """
    _check_reference(f_ref, s_ref)
    _check_budget(budget_s)
    if not (math.isfinite(tau) and tau >= 0.0):
        raise ValueError(f"tau must be a finite non-negative number, got {tau}")
    times, values = incumbent_curve(trajectory, start=start)
    hit = np.flatnonzero((values - f_ref) / s_ref <= tau)
    if hit.size and times[hit[0]] <= budget_s:
        return TimeToTarget(float(times[hit[0]]), True)
    return TimeToTarget(float(budget_s), False)


def kaplan_meier(
    times: Sequence[float] | np.ndarray, observed: Sequence[bool] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan–Meier survival curve of right-censored times: ``(event_times, survival)``.

    ``observed[i]`` is ``True`` if run ``i`` reached its target at ``times[i]`` and ``False`` if it
    was censored there (at the budget). ``survival[k]`` is ``S(t)`` right after the ``k``-th
    distinct event time. A run censored at ``t`` is still at risk at ``t``: at tied times the
    events come first. All runs censored gives two empty arrays.
    """
    t = np.asarray(times, dtype=float)
    d = np.asarray(observed, dtype=bool)
    if t.ndim != 1 or t.shape != d.shape:
        raise ValueError(f"times and observed must be 1-D of equal length, got {t.shape}, {d.shape}")
    if t.size == 0:
        raise ValueError("at least one run is needed")
    if not (np.all(np.isfinite(t)) and np.all(t >= 0.0)):
        raise ValueError("times must be finite and non-negative")
    unique, inverse = np.unique(t, return_inverse=True)
    events = np.bincount(inverse, weights=d.astype(float), minlength=unique.size)
    leaving = np.bincount(inverse, minlength=unique.size)
    at_risk = t.size - np.concatenate([[0], np.cumsum(leaving)[:-1]])
    survival = np.cumprod(1.0 - events / at_risk)
    has_event = events > 0
    return unique[has_event], survival[has_event]


def km_median(
    times: Sequence[float] | np.ndarray, observed: Sequence[bool] | np.ndarray
) -> float:
    """Kaplan–Meier median of right-censored times: the smallest ``t`` with ``S(t) ≤ 1/2``.

    ``math.inf`` if the curve never gets to one half (more than half of the runs censored): the
    median is then beyond the budget, and is reported as censored. With no censoring this is the
    lower sample median (the ``⌈n/2⌉``-th order statistic).
    """
    event_times, survival = kaplan_meier(times, observed)
    hit = np.flatnonzero(survival <= 0.5 + _HALF_TOL)
    return float(event_times[hit[0]]) if hit.size else math.inf


def km_quantile(
    times: Sequence[float] | np.ndarray, observed: Sequence[bool] | np.ndarray, q: float
) -> float:
    """Kaplan–Meier ``q``-quantile of right-censored times: the smallest ``t`` with ``S(t) ≤ 1 − q``.

    ``math.inf`` if the curve never gets there (the quantile is beyond the budget). ``q = 1/2`` is
    :func:`km_median`; ``q`` must lie in ``(0, 1)``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie in (0, 1), got {q}")
    event_times, survival = kaplan_meier(times, observed)
    hit = np.flatnonzero(survival <= 1.0 - q + _HALF_TOL)
    return float(event_times[hit[0]]) if hit.size else math.inf
