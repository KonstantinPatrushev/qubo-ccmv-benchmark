"""Optimality gap as registered in the plan, and a typed reference (R9, ANALYSIS_PLAN §5, M10).

The objective is minimised, ``f = −μᵀw + (λ/2)·wᵀΣw``. On realistic data it is small, negative and
passes through zero, so a gap normalised by ``|f_ref|`` flips its sign or blows up exactly where the
study must still resolve differences. The gap of the study (§5) is normalised by the positive scale
of the reference portfolio instead::

    ĝ(f) = (f − f_ref) / s_ref,   s_ref = |μᵀw_ref| + (λ/2)·w_refᵀΣw_ref > 0

(:func:`relative_gap`, :func:`reference_scale`); ``w_ref`` is the portfolio that attains ``f_ref``.
Since ``|f| ≤ s_ref``, the denominator cannot collapse where ``f_ref ≈ 0``. ĝ is the unit of the
primal integral and of the targets τ = 1e-3 and 1e-4 (:mod:`cif.metrics.anytime`) and of the tie
band of :func:`classify_against_reference`. A positive gap means ``f`` is worse than the reference.

A gap speaks about optimality only if the reference is the optimum, hence the reference type:

* ``"certified"`` — the exact solver stopped by closing the gap and its **absolute** primal–dual gap
  is within the registered tolerance on the same scale, ``|primal − dual| ≤ 1e-6·s_ref`` (SCIP
  ``limits/absgap``, ``solve_miqp_native(abs_gap=...)``): the reference *is* the optimum up to
  ``1e-6`` in units of ĝ. SCIP's own relative gap is not the criterion: it divides by the bounds,
  which are small exactly where ``f_ref ≈ 0``, and refuses a certificate the plan grants;
* ``"best_known"`` — the best feasible value found by anybody, with no proof. A gap to it is always
  reported under this name and never as a gap to the optimum.

Three numbers are reported for an exact solver (R9): the gap to the certified optimum where it is
proven, the solver's own primal–dual gap (``solver_meta["abs_gap"]`` and ``["gap"]`` of
:func:`cif.classical.scip_native.solve_miqp_native`) and, explicitly labelled, the gap to the best
known value. Typical use for one instance::

    s_ref = reference_scale(problem, exact.weights)
    f_ref, ref_type = reference_from_exact(exact.solver_meta, scale=s_ref)
    if ref_type == "best_known":  # no proof: the best value of all solvers is the reference
        f_ref = min(f_ref, *heuristic_values)
        s_ref = reference_scale(problem, w_best)  # of the portfolio that attains f_ref
    gap = relative_gap(f, f_ref, scale=s_ref)
    verdict = classify_against_reference(f, f_ref, ref_type, gap_tol, scale=s_ref)

LEGACY. The first definition of the project, ``(f − f_ref)/max(|f_ref|, |f_cont|, 1e-12)``, survives
only as :func:`legacy_relative_gap`, to reproduce numbers computed with it. It is not the registered
gap (audit G-1): where ``f_ref ≈ 0`` it is orders of magnitude larger than ĝ.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from cif.problem import PortfolioProblem

__all__ = [
    "BEST_KNOWN",
    "CERTIFIED",
    "DENOMINATOR_FLOOR",
    "REF_TYPES",
    "RefType",
    "Verdict",
    "classify_against_reference",
    "legacy_relative_gap",
    "reference_from_exact",
    "reference_scale",
    "relative_gap",
]

RefType = Literal["certified", "best_known"]
Verdict = Literal["worse", "tie", "better"]

CERTIFIED: RefType = "certified"
BEST_KNOWN: RefType = "best_known"
REF_TYPES: tuple[RefType, ...] = (CERTIFIED, BEST_KNOWN)

#: Floor of the denominator of the LEGACY gap (:func:`legacy_relative_gap`); not used by ĝ.
DENOMINATOR_FLOOR = 1e-12
#: SCIP statuses that carry a certificate; the absolute gap still has to be within the tolerance.
_CERTIFYING_STATUSES = ("optimal", "gaplimit")


def _as_float(x: Any) -> float:
    """``float(x)``, or NaN when ``x`` is not a number (``None``, a string, ...)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def _checked_scale(scale: float) -> float:
    """``scale`` as a float; ``ValueError`` unless it is finite and positive."""
    value = _as_float(scale)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"scale must be a finite positive number, got {scale!r}")
    return value


def reference_scale(problem: PortfolioProblem, w_ref: np.ndarray) -> float:
    """``s_ref = |μᵀw_ref| + (λ/2)·w_refᵀΣw_ref``, the positive scale of the gap (§5, M10).

    ``w_ref`` is the weight vector of the reference value: the certified optimum, or the best
    known portfolio of all methods. For a budget-one portfolio the scale is positive, ``s_ref ≥
    |f(w_ref)|`` (so the gap is never inflated by a reference near zero), and it scales with the
    objective. Pass it as ``relative_gap(f, f_ref, scale=s_ref)``.

    A degenerate reference (wrong shape, non-finite weights, zero scale) raises ``ValueError``.
    """
    w = np.asarray(w_ref, dtype=float)
    if w.shape != problem.mu.shape:
        raise ValueError(f"w_ref must have shape {problem.mu.shape}, got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("w_ref must be finite")
    scale = abs(float(problem.mu @ w)) + 0.5 * problem.risk_aversion * float(w @ problem.sigma @ w)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"the scale of the reference portfolio must be positive, got {scale}")
    return scale


def relative_gap(f: float, f_ref: float, *, scale: float) -> float:
    """The gap of the plan, ``ĝ = (f − f_ref) / s_ref`` (minimisation; §5, M10).

    ``scale`` is ``s_ref`` of the reference portfolio (:func:`reference_scale`): finite and
    positive, required. Positive: ``f`` is worse than the reference; negative: better (possible
    against a ``best_known`` reference, or by numerical noise against a certified one). Linear in
    the objective and invariant to its units when ``scale`` follows them.

    ``f`` may be ``+inf`` (the solver returned nothing); the gap is then ``+inf``. ``f_ref`` must be
    finite, and nothing may be NaN: an undefined gap is an error, not a value.
    """
    f = float(f)
    f_ref = float(f_ref)
    if not (math.isfinite(f) or f == math.inf):
        raise ValueError(f"f must be a finite number or +inf (no solution), got {f}")
    if not math.isfinite(f_ref):
        raise ValueError(f"f_ref must be finite, got {f_ref}")
    denominator = _checked_scale(scale)
    if f == math.inf:
        return math.inf
    return (f - f_ref) / denominator


def legacy_relative_gap(f: float, f_ref: float, f_cont: float | None = None) -> float:
    """LEGACY: ``(f − f_ref) / max(|f_ref|, |f_cont|, 1e-12)``, the first definition of the project.

    Not the gap registered in the plan (use :func:`relative_gap`): it normalises by the reference
    value, which is tiny where the study needs resolution, and ``f_cont`` (the relaxation value)
    only softens that. Kept to reproduce numbers computed with it; positive means ``f`` is worse.

    ``f`` may be ``+inf``; ``f_ref`` and ``f_cont`` must be finite and nothing may be NaN.
    """
    f = float(f)
    f_ref = float(f_ref)
    cont = 0.0 if f_cont is None else float(f_cont)
    if not (math.isfinite(f) or f == math.inf):
        raise ValueError(f"f must be a finite number or +inf (no solution), got {f}")
    if not math.isfinite(f_ref):
        raise ValueError(f"f_ref must be finite, got {f_ref}")
    if not math.isfinite(cont):
        raise ValueError(f"f_cont must be finite, got {f_cont}")
    if f == math.inf:
        return math.inf
    return (f - f_ref) / max(abs(f_ref), abs(cont), DENOMINATOR_FLOOR)


def reference_from_exact(
    solution_meta: Mapping[str, Any], *, scale: float, tol: float = 1e-6
) -> tuple[float, RefType]:
    """Reference value and its type from the ``solver_meta`` of an exact run (§5).

    Returns ``(f_ref, "certified")`` if the SCIP status is ``"optimal"`` or ``"gaplimit"`` **and**
    the absolute primal–dual gap is within the tolerance on the scale of the reference,
    ``|primal_bound − dual_bound| ≤ tol·scale`` (``tol`` = 1e-6, ``scale`` = ``s_ref`` of the
    incumbent's weights, :func:`reference_scale`: the criterion ``limits/absgap = 1e-6·s_ref``
    enforces); otherwise ``(f_ref, "best_known")``. A time limit or any other status, and a
    missing, NaN or infinite dual bound, never certify, whatever the other fields say.

    ``f_ref`` is ``solution_meta["primal_bound"]``, the incumbent's objective in the model's units:
    the pure mean–variance objective, equal to ``Solution.objective`` up to ~1e-8 relative, and the
    penalised one when a turnover penalty was active (compare like with like). Without an incumbent
    it is ``+inf`` with ``"best_known"``: it composes with ``min`` over the values of other
    solvers, but :func:`relative_gap` refuses it as a reference. A meta without ``primal_bound``
    (e.g. the legacy cvxpy baseline, which reports no bounds) raises ``ValueError``.

    A run stopped by SCIP's relative limit ``mip_gap`` is certified as well when its gap meets the
    criterion: ``|p − d| ≤ mip_gap·min(|p|, |d|)`` and ``s_ref ≥ |f|``, so ``mip_gap ≤ tol`` implies
    it. The bounds hold up to SCIP's feasibility tolerance (``feastol`` in the meta): keep it well
    below ``tol``.
    """
    if not (math.isfinite(tol) and tol >= 0.0):
        raise ValueError(f"tol must be a finite non-negative number, got {tol}")
    band = tol * _checked_scale(scale)
    primal = solution_meta.get("primal_bound")
    f_ref = _as_float(primal)
    if math.isnan(f_ref) or f_ref == -math.inf:
        raise ValueError(
            f"solution_meta['primal_bound'] must be a number or +inf (no incumbent), got {primal!r}"
            " (expected the solver_meta of cif.classical.scip_native.solve_miqp_native)"
        )
    dual = _as_float(solution_meta.get("dual_bound"))
    # a missing, NaN or infinite bound makes the difference NaN or infinite: never within the band
    certified = solution_meta.get("status") in _CERTIFYING_STATUSES and abs(f_ref - dual) <= band
    return f_ref, CERTIFIED if certified else BEST_KNOWN


def classify_against_reference(
    f: float,
    f_ref: float,
    ref_type: RefType,
    gap_tol: float,
    *,
    scale: float,
) -> Verdict:
    """``"worse"``, ``"tie"`` or ``"better"`` than the reference, with a tie band (audit D-1).

    ``f`` and the reference are compared through :func:`relative_gap` with the same ``scale``
    (``s_ref``) as the reported gap: ``"worse"`` if the gap exceeds ``gap_tol``, ``"better"`` if it
    is below ``−gap_tol``, ``"tie"`` otherwise, edges included. A difference inside the band is
    below what the reference can resolve, so it is no win and no loss.

    ``gap_tol`` is the width of the band in units of ĝ: for a certified reference at least its own
    certified gap ``|primal − dual|/s_ref``, floored at the numerical tolerance, e.g. ``1e-6``
    (the registered certificate); never below ~1e-8, where the model's primal bound and the
    re-evaluated objective of one portfolio already differ. For a ``best_known`` reference there is
    no certificate and it is merely a numerical tolerance.

    What a verdict proves depends on ``ref_type``. Against ``"certified"``: ``"worse"`` is worse
    than the optimum, ``"tie"`` is optimal to within the band, and ``"better"`` contradicts the
    certificate (infeasible candidate, a different problem as in D-1, numerics, or a ``gap_tol``
    below the reference's own gap) and must be investigated, not counted as a win. Against
    ``"best_known"``: ``"worse"`` is still worse than the optimum, ``"better"`` is a new best known
    value, and ``"tie"`` means only "as good as the best known", never "optimal".
    """
    if ref_type not in REF_TYPES:
        raise ValueError(f"ref_type must be one of {REF_TYPES}, got {ref_type!r}")
    if not (math.isfinite(gap_tol) and gap_tol >= 0.0):
        raise ValueError(f"gap_tol must be a finite non-negative number, got {gap_tol}")
    gap = relative_gap(f, f_ref, scale=scale)
    if gap > gap_tol:
        return "worse"
    if gap < -gap_tol:
        return "better"
    return "tie"
