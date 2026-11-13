"""Tests for cif.metrics.gap (R9, audit G-1): the gap of the plan and the typed reference.

ANALYSIS_PLAN §5 (M10) registers ``ĝ(f) = (f − f_ref) / s_ref`` with
``s_ref = |μᵀw_ref| + (λ/2)·w_refᵀΣw_ref`` and certifies an optimum by the ABSOLUTE primal–dual gap
``|primal − dual| ≤ 1e-6·s_ref``. The first definition of the project,
``(f − f_ref) / max(|f_ref|, |f_cont|, 1e-12)``, survives as ``legacy_relative_gap`` and is tested
separately. The reference is ``certified`` only when the exact solver proved it, otherwise
``best_known``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cif.classical.scip_native import solve_miqp_native
from cif.instances.factor import factor_instance
from cif.metrics.gap import (
    classify_against_reference,
    legacy_relative_gap,
    reference_from_exact,
    reference_scale,
    relative_gap,
)
from cif.problem import PortfolioProblem

# ---------------------------------------------------------------------------
# 1. The positive scale s_ref (§5, M10)
# ---------------------------------------------------------------------------


def _two_assets() -> PortfolioProblem:
    return PortfolioProblem(
        mu=np.array([0.10, 0.20]),
        sigma=np.diag([0.04, 0.09]),
        asset_names=("A", "B"),
        risk_aversion=5.0,
    )


def test_reference_scale_matches_the_definition():
    problem = _two_assets()
    w = np.array([0.6, 0.4])
    # |0.1·0.6 + 0.2·0.4| + (5/2)·(0.36·0.04 + 0.16·0.09)
    assert reference_scale(problem, w) == pytest.approx(0.14 + 2.5 * 0.0288, rel=1e-14)
    # the return term enters through its absolute value: mu → −mu leaves the scale unchanged
    flipped = PortfolioProblem(
        mu=-problem.mu, sigma=problem.sigma, asset_names=problem.asset_names, risk_aversion=5.0
    )
    assert reference_scale(flipped, w) == pytest.approx(reference_scale(problem, w), rel=1e-14)


def test_reference_scale_is_positive_where_the_objective_vanishes():
    # μᵀw = (λ/2)·wᵀΣw makes f_ref = 0: the legacy denominator collapses to its 1e-12 floor
    problem = _two_assets()
    w = np.array([0.5, 0.5])
    ret, risk = float(problem.mu @ w), 2.5 * float(w @ problem.sigma @ w)
    problem = PortfolioProblem(
        mu=problem.mu * risk / ret,
        sigma=problem.sigma,
        asset_names=problem.asset_names,
        risk_aversion=5.0,
    )
    assert problem.objective_value(w) == pytest.approx(0.0, abs=1e-15)
    scale = reference_scale(problem, w)
    assert scale == pytest.approx(2.0 * risk, rel=1e-12)
    assert legacy_relative_gap(1e-6, 0.0) == pytest.approx(1e6)  # the legacy scale blows up
    assert relative_gap(1e-6, 0.0, scale=scale) == pytest.approx(1e-6 / scale)


@pytest.mark.parametrize("seed", [1000, 1001, 1002])
def test_reference_scale_bounds_the_objective_from_above(seed):
    mu, sigma, _ = factor_instance(30, seed)
    problem = PortfolioProblem(
        mu=mu, sigma=sigma, asset_names=tuple(map(str, range(30))), risk_aversion=5.0
    )
    rng = np.random.default_rng(10000 + seed)
    for _ in range(20):
        w = rng.dirichlet(np.ones(30))
        scale = reference_scale(problem, w)
        assert scale > 0.0
        assert scale >= abs(problem.objective_value(w)) - 1e-15


@pytest.mark.parametrize(
    "w",
    [np.zeros(2), np.array([0.5, np.nan]), np.array([0.5, 0.25, 0.25]), np.array([np.inf, 0.0])],
)
def test_reference_scale_rejects_degenerate_references(w):
    with pytest.raises(ValueError):
        reference_scale(_two_assets(), w)


# ---------------------------------------------------------------------------
# 2. relative_gap: ĝ = (f − f_ref) / s_ref (R9 acceptance, audit G-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f", "f_ref", "scale", "expected"),
    [
        (1.1, 1.0, 1.0, 0.1),  # positive objective, worse than the reference
        (0.9, 1.0, 1.0, -0.1),  # positive objective, better
        # The mean-variance objective is usually negative. (f − f_ref)/f_ref would flip the sign
        # here; with a positive scale "positive gap" always means "f is worse".
        (-0.18, -0.20, 0.5, 0.04),
        (-0.22, -0.20, 0.5, -0.04),
        (-0.2, -0.2, 0.5, 0.0),
    ],
)
def test_relative_gap_is_the_difference_over_the_scale(f, f_ref, scale, expected):
    assert relative_gap(f, f_ref, scale=scale) == pytest.approx(expected, abs=1e-12)


def test_the_plan_example_of_the_audit():
    # f_ref = −1e-4, f = 0, s_ref = 0.1: ĝ = 1e-3, the edge of the target tolerance τ. The legacy
    # normalisation, with the relaxation at −2e-4, gave 0.5: 500 times the registered value.
    assert relative_gap(0.0, -1e-4, scale=0.1) == pytest.approx(1e-3)
    assert legacy_relative_gap(0.0, -1e-4, -2e-4) == pytest.approx(0.5)


def test_relative_gap_is_defined_at_a_zero_reference():
    assert relative_gap(0.0, 0.0, scale=0.2) == 0.0
    # the scale is independent of f_ref, so the gap near a zero reference is moderate, not 1e6
    assert relative_gap(1e-6, 0.0, scale=0.2) == pytest.approx(5e-6)


@pytest.mark.parametrize("c", [1e-3, 7.0, 1e4])
def test_relative_gap_is_invariant_to_the_objective_units(c):
    base = relative_gap(-0.18, -0.20, scale=0.3)
    assert relative_gap(-0.18 * c, -0.20 * c, scale=0.3 * c) == pytest.approx(base, rel=1e-12)


def test_relative_gap_without_a_solution_is_infinite():
    assert relative_gap(math.inf, -0.2, scale=0.3) == math.inf
    assert relative_gap(np.inf, 0.0, scale=0.3) == math.inf


@pytest.mark.parametrize(
    ("f", "f_ref", "scale"),
    [
        (math.nan, -0.2, 0.3),
        (-0.2, math.nan, 0.3),
        (-0.2, -0.2, math.nan),
        (-0.2, math.inf, 0.3),  # a reference must exist: no incumbent is not a reference
        (-0.2, -math.inf, 0.3),
        (-math.inf, -0.2, 0.3),
        (-0.2, -0.2, 0.0),
        (-0.2, -0.2, -0.3),
        (-0.2, -0.2, math.inf),
        (math.inf, -0.2, 0.0),  # a missing solution does not excuse a missing scale
    ],
)
def test_relative_gap_rejects_undefined_inputs(f, f_ref, scale):
    with pytest.raises(ValueError):
        relative_gap(f, f_ref, scale=scale)


def test_the_scale_is_required_and_keyword_only():
    # the registered gap needs s_ref; there is no silent fallback to another normalisation
    with pytest.raises(TypeError):
        relative_gap(-0.2, -0.2)
    with pytest.raises(TypeError):
        relative_gap(-0.2, -0.2, 0.3)


def test_relative_gap_returns_a_python_float_for_numpy_inputs():
    g = relative_gap(np.float64(1.1), np.array(1.0), scale=np.float64(2.0))
    assert type(g) is float
    assert g == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 3. legacy_relative_gap: the first definition, kept by name only (audit G-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f", "f_ref", "expected"),
    [(1.1, 1.0, 0.1), (0.9, 1.0, -0.1), (-0.18, -0.20, 0.1), (-0.22, -0.20, -0.1), (-0.2, -0.2, 0.0)],
)
def test_legacy_gap_sign_and_value(f, f_ref, expected):
    assert legacy_relative_gap(f, f_ref) == pytest.approx(expected, abs=1e-12)


def test_legacy_gap_is_defined_at_a_zero_reference_through_its_floor():
    assert legacy_relative_gap(0.0, 0.0) == 0.0
    g = legacy_relative_gap(1e-6, 0.0)  # |f_ref| = 0: the floor 1e-12, not a division by zero
    assert math.isfinite(g)
    assert g == pytest.approx(1e-6 / 1e-12)
    # the continuous relaxation supplies the scale when the reference is (near) zero
    assert legacy_relative_gap(0.01, 0.0, f_cont=-0.2) == pytest.approx(0.05)


def test_legacy_gap_scale_is_the_largest_of_reference_relaxation_and_floor():
    assert legacy_relative_gap(-0.09, -0.10, f_cont=-0.25) == pytest.approx(0.01 / 0.25)
    assert legacy_relative_gap(-0.09, -0.10, f_cont=-0.05) == pytest.approx(0.01 / 0.10)
    assert legacy_relative_gap(-0.09, -0.10, f_cont=None) == pytest.approx(0.01 / 0.10)
    assert legacy_relative_gap(0.9, 1.0, f_cont=2.0) == legacy_relative_gap(0.9, 1.0, f_cont=-2.0)


@pytest.mark.parametrize("c", [1e-3, 7.0, 1e4])
def test_legacy_gap_is_invariant_to_the_objective_scale(c):
    base = legacy_relative_gap(-0.18, -0.20, f_cont=-0.25)
    assert legacy_relative_gap(-0.18 * c, -0.20 * c, f_cont=-0.25 * c) == pytest.approx(
        base, rel=1e-12
    )


def test_legacy_gap_without_a_solution_is_infinite():
    assert legacy_relative_gap(math.inf, -0.2) == math.inf
    assert legacy_relative_gap(np.inf, 0.0, f_cont=-0.2) == math.inf


@pytest.mark.parametrize(
    ("f", "f_ref", "f_cont"),
    [
        (math.nan, -0.2, None),
        (-0.2, math.nan, None),
        (-0.2, -0.2, math.nan),
        (-0.2, math.inf, None),
        (-0.2, -math.inf, None),
        (-0.2, -0.2, math.inf),
        (-math.inf, -0.2, None),
    ],
)
def test_legacy_gap_rejects_undefined_inputs(f, f_ref, f_cont):
    with pytest.raises(ValueError):
        legacy_relative_gap(f, f_ref, f_cont)


# ---------------------------------------------------------------------------
# 4. reference_from_exact: certified only by the absolute gap |p − d| ≤ tol·s_ref (G-1)
# ---------------------------------------------------------------------------

SCALE = 0.25  # a plausible s_ref; the certificate band is 1e-6 · 0.25 = 2.5e-7


def _meta(status, *, primal=-0.2, dual=-0.2):
    return {"status": status, "primal_bound": primal, "dual_bound": dual}


def test_optimal_and_gaplimit_within_the_absolute_band_are_certified():
    assert reference_from_exact(_meta("optimal"), scale=SCALE) == (-0.2, "certified")
    meta = _meta("gaplimit", dual=-0.2 - 2e-7)  # |p − d| = 2e-7 ≤ 2.5e-7
    assert reference_from_exact(meta, scale=SCALE) == (-0.2, "certified")


def test_the_boundary_of_the_band_belongs_to_certified():
    # exactly representable numbers: |p − d| = 0.5 = tol·scale
    meta = _meta("gaplimit", primal=0.0, dual=-0.5)
    assert reference_from_exact(meta, scale=0.5, tol=1.0) == (0.0, "certified")
    meta = _meta("gaplimit", primal=0.0, dual=-0.5000001)
    assert reference_from_exact(meta, scale=0.5, tol=1.0) == (0.0, "best_known")


def test_the_band_is_tol_times_the_scale_and_both_are_the_callers_choice():
    meta = _meta("gaplimit", dual=-0.2 - 3e-7)  # 3e-7 > 1e-6 · 0.25
    assert reference_from_exact(meta, scale=SCALE)[1] == "best_known"
    assert reference_from_exact(meta, scale=SCALE, tol=2e-6)[1] == "certified"
    assert reference_from_exact(meta, scale=2 * SCALE)[1] == "certified"
    # an "optimal" status does not override a gap that exceeds the band
    assert reference_from_exact(_meta("optimal", dual=-0.2 - 1e-3), scale=SCALE)[1] == "best_known"


def test_the_certificate_is_absolute_not_relative_to_the_bounds():
    # Audit G-1: gaplimit with |primal − dual| = 5e-9 at s_ref = 0.05. The plan certifies
    # (5e-9 ≤ 1e-6·0.05); SCIP's relative gap, 5e-5, does not meet 1e-6 and used to decide.
    meta = {"status": "gaplimit", "primal_bound": -1.0e-4, "dual_bound": -1.0e-4 - 5e-9, "gap": 5e-5}
    assert reference_from_exact(meta, scale=0.05) == (-1.0e-4, "certified")
    # the relative gap of the meta is not consulted, either way
    assert reference_from_exact({**meta, "gap": 0.0}, scale=0.05)[1] == "certified"
    assert reference_from_exact({**meta, "gap": 0.0}, scale=5e-4)[1] == "best_known"  # 5e-9 > 5e-10


@pytest.mark.parametrize(
    "status",
    ["timelimit", "nodelimit", "infeasible", "unbounded", "userinterrupt", "unknown", "", None],
)
def test_other_statuses_are_never_certified(status):
    # even with identical bounds: the status alone must not allow "certified"
    assert reference_from_exact(_meta(status), scale=SCALE)[1] == "best_known"


@pytest.mark.parametrize("dual", [None, math.nan, math.inf, -math.inf])
def test_an_unusable_dual_bound_is_never_certified(dual):
    assert reference_from_exact(_meta("optimal", dual=dual), scale=SCALE)[1] == "best_known"
    assert reference_from_exact(_meta("gaplimit", dual=dual), scale=SCALE)[1] == "best_known"


def test_a_missing_dual_bound_is_never_certified():
    meta = {"status": "optimal", "primal_bound": -0.2}
    assert reference_from_exact(meta, scale=SCALE) == (-0.2, "best_known")


def test_time_limited_run_with_an_incumbent_is_best_known():
    # values of a native run stopped by the time limit (N = 40, K = 6, 0.3 s)
    meta = {"status": "timelimit", "primal_bound": -0.0666, "dual_bound": -0.2159}
    assert reference_from_exact(meta, scale=SCALE) == (-0.0666, "best_known")


def test_bounds_that_disagree_beyond_the_band_are_not_certified_in_either_direction():
    assert reference_from_exact(_meta("optimal", dual=-0.1), scale=SCALE)[1] == "best_known"
    assert reference_from_exact(_meta("optimal", dual=-0.3), scale=SCALE)[1] == "best_known"
    # sign-crossing bounds (SCIP's relative gap is infinite there)
    meta = {"status": "timelimit", "primal_bound": 1e-3, "dual_bound": -0.1}
    assert reference_from_exact(meta, scale=SCALE) == (1e-3, "best_known")


def test_no_incumbent_gives_an_infinite_best_known_reference():
    meta = {
        "status": "timelimit",
        "primal_bound": math.inf,
        "dual_bound": -math.inf,
    }
    f_ref, ref_type = reference_from_exact(meta, scale=SCALE)
    assert f_ref == math.inf
    assert ref_type == "best_known"
    # such a value composes with min() over other solvers, but it cannot define a gap by itself
    with pytest.raises(ValueError):
        relative_gap(-0.1, f_ref, scale=SCALE)


def test_a_certified_reference_must_be_finite():
    # inconsistent meta (a certificate without an incumbent): never "certified"
    meta = _meta("optimal", primal=math.inf, dual=math.inf)
    assert reference_from_exact(meta, scale=SCALE) == (math.inf, "best_known")


@pytest.mark.parametrize("primal", [math.nan, -math.inf, None])
def test_a_nonsensical_primal_bound_is_rejected(primal):
    with pytest.raises(ValueError, match="primal_bound"):
        reference_from_exact(_meta("optimal", primal=primal), scale=SCALE)


def test_a_meta_without_a_primal_bound_is_rejected():
    # e.g. the solver_meta of the legacy cvxpy baseline: no bounds, nothing to certify
    with pytest.raises(ValueError, match="primal_bound"):
        reference_from_exact({"status": "optimal", "cvxpy_value": -0.2}, scale=SCALE)


@pytest.mark.parametrize("tol", [-1e-6, math.nan, math.inf])
def test_tolerance_must_be_a_finite_non_negative_number(tol):
    with pytest.raises(ValueError, match="tol"):
        reference_from_exact(_meta("optimal"), scale=SCALE, tol=tol)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf, None, "big"])
def test_the_scale_must_be_a_finite_positive_number(scale):
    with pytest.raises(ValueError, match="scale"):
        reference_from_exact(_meta("optimal"), scale=scale)


def test_the_scale_is_required_and_the_tolerance_cannot_be_passed_positionally():
    # a positional second argument used to be ``tol``: it must not be read as a scale now
    with pytest.raises(TypeError):
        reference_from_exact(_meta("optimal"))
    with pytest.raises(TypeError):
        reference_from_exact(_meta("optimal"), 1e-6)


def test_the_input_meta_is_not_modified():
    meta = _meta("gaplimit", dual=-0.2 - 1e-7)
    snapshot = dict(meta)
    reference_from_exact(meta, scale=SCALE)
    assert meta == snapshot


# ---------------------------------------------------------------------------
# 5. classify_against_reference: worse / tie / better, on the scale of ĝ (rule of audit D-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f_ref", "scale"),
    [
        (-0.2, 0.25),  # negative objective
        (0.0, 0.25),  # zero reference: the scale does not depend on it
        (3.0, 3.0),  # positive objective
        (-0.2, 0.2),
    ],
)
@pytest.mark.parametrize("ref_type", ["certified", "best_known"])
def test_worse_tie_better_around_the_reference(ref_type, f_ref, scale):
    tol = 1e-6

    def verdict(f):
        return classify_against_reference(f, f_ref, ref_type, tol, scale=scale)

    assert verdict(f_ref) == "tie"
    assert verdict(f_ref + 0.5 * tol * scale) == "tie"
    assert verdict(f_ref - 0.5 * tol * scale) == "tie"
    assert verdict(f_ref + 2.0 * tol * scale) == "worse"
    assert verdict(f_ref - 2.0 * tol * scale) == "better"


def test_the_edge_of_the_tie_band_is_a_tie():
    # exactly representable numbers: ĝ = 0.25 and 0.5 on the scale 1
    assert classify_against_reference(1.25, 1.0, "certified", 0.25, scale=1.0) == "tie"
    assert classify_against_reference(0.75, 1.0, "certified", 0.25, scale=1.0) == "tie"
    assert classify_against_reference(1.5, 1.0, "certified", 0.25, scale=1.0) == "worse"
    assert classify_against_reference(0.5, 1.0, "certified", 0.25, scale=1.0) == "better"


def test_difference_inside_the_certified_gap_is_a_tie_not_a_win():
    # Audit D-1: a heuristic 5e-9 below SCIP's incumbent (certified to 1e-6 of s_ref, here 0.3)
    # was counted as a win, and a difference of 2.8e-5 is a genuine miss.
    pen = -0.2297
    assert classify_against_reference(pen - 5e-9, pen, "certified", 1e-6, scale=0.3) == "tie"
    assert classify_against_reference(pen + 5e-9, pen, "certified", 1e-6, scale=0.3) == "tie"
    assert classify_against_reference(pen + 2.8e-5, pen, "certified", 1e-6, scale=0.3) == "worse"


def test_classification_uses_the_same_scale_as_the_reported_gap():
    # Near-zero reference: on its own value the tie band would collapse to ~1e-18
    f = 1e-8
    assert classify_against_reference(f, 0.0, "certified", 1e-6, scale=0.2) == "tie"
    # tie if and only if |ĝ| ≤ gap_tol
    for ff in (-3e-7, 0.0, 1e-7, 2e-6, -4e-6):
        gap = relative_gap(ff, 0.0, scale=0.2)
        expected = "tie" if abs(gap) <= 1e-6 else ("worse" if gap > 0 else "better")
        assert classify_against_reference(ff, 0.0, "certified", 1e-6, scale=0.2) == expected


def test_a_solver_without_a_solution_is_worse():
    assert classify_against_reference(math.inf, -0.2, "certified", 1e-6, scale=0.3) == "worse"
    assert classify_against_reference(math.inf, -0.2, "best_known", 1e-6, scale=0.3) == "worse"


def test_beating_a_best_known_reference_is_a_legitimate_better():
    # best_known is not an optimum; improving on it is expected (and is the new best known)
    assert classify_against_reference(-0.25, -0.2, "best_known", 1e-6, scale=0.3) == "better"


@pytest.mark.parametrize("ref_type", ["optimal", "Certified", "", None, "best-known"])
def test_unknown_reference_type_is_rejected(ref_type):
    with pytest.raises(ValueError, match="ref_type"):
        classify_against_reference(-0.2, -0.2, ref_type, 1e-6, scale=0.3)


@pytest.mark.parametrize("gap_tol", [-1e-6, math.nan, math.inf])
def test_gap_tolerance_must_be_a_finite_non_negative_number(gap_tol):
    with pytest.raises(ValueError, match="gap_tol"):
        classify_against_reference(-0.2, -0.2, "certified", gap_tol, scale=0.3)


def test_a_nan_objective_cannot_be_classified():
    with pytest.raises(ValueError):
        classify_against_reference(math.nan, -0.2, "certified", 1e-6, scale=0.3)


def test_classification_needs_the_scale_of_the_reference():
    with pytest.raises(TypeError):
        classify_against_reference(-0.2, -0.2, "certified", 1e-6)
    with pytest.raises(ValueError, match="scale"):
        classify_against_reference(-0.2, -0.2, "certified", 1e-6, scale=0.0)


# ---------------------------------------------------------------------------
# 6. With the native SCIP solver: best_known never passes as certified
# ---------------------------------------------------------------------------


def _problem(n: int, seed: int, K: int) -> PortfolioProblem:
    """Synthetic instance, same recipe as ``tests/conftest.py::medium_problem``."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=K,
    )


def test_native_run_to_the_absolute_gap_limit_is_a_certified_reference():
    problem = _problem(12, 703, 3)
    # the plan's recipe: limits/absgap = 1e-6·s_ref (here half of it, leaving margin for the small
    # difference between the scale of a first solution and of the final incumbent)
    s_first = reference_scale(problem, solve_miqp_native(problem, 3).weights)
    sol = solve_miqp_native(problem, 3, mip_gap=0.0, abs_gap=0.5e-6 * s_first)
    assert sol.solver_meta["status"] in ("optimal", "gaplimit")
    s_ref = reference_scale(problem, sol.weights)
    f_ref, ref_type = reference_from_exact(sol.solver_meta, scale=s_ref)
    assert ref_type == "certified"
    # the model's primal bound and the re-evaluated objective of the same portfolio agree
    assert f_ref == pytest.approx(sol.objective, rel=1e-6)
    assert classify_against_reference(sol.objective, f_ref, ref_type, 1e-6, scale=s_ref) == "tie"
    # a clearly worse portfolio (equal weights on the first three assets) is "worse"
    w = np.zeros(problem.n)
    w[:3] = 1.0 / 3.0
    worse = classify_against_reference(problem.objective_value(w), f_ref, ref_type, 1e-6, scale=s_ref)
    assert worse == "worse"


def test_native_run_to_the_relative_gap_limit_is_certified_on_the_absolute_scale_too():
    # mip_gap ≤ tol and s_ref ≥ |f| make the relative stop imply the absolute criterion
    problem = _problem(12, 703, 3)
    sol = solve_miqp_native(problem, 3, mip_gap=1e-7)
    assert sol.solver_meta["status"] in ("optimal", "gaplimit")
    _, ref_type = reference_from_exact(sol.solver_meta, scale=reference_scale(problem, sol.weights))
    assert ref_type == "certified"


def test_native_run_stopped_by_the_time_limit_is_never_certified():
    problem = _problem(30, 11, 4)
    # time limit 0: SCIP stops at once. Without an incumbent the reference does not exist ...
    cold = solve_miqp_native(problem, 4, time_limit_s=0.0)
    assert cold.solver_meta["status"] == "timelimit"
    f_cold, type_cold = reference_from_exact(cold.solver_meta, scale=1.0)
    assert type_cold == "best_known"
    if not math.isfinite(cold.objective):
        assert f_cold == math.inf
    # ... and with a feasible start the incumbent is only the best known value
    start = np.zeros(problem.n)
    start[:4] = 0.25
    warm = solve_miqp_native(problem, 4, time_limit_s=0.0, warm_start=start)
    assert warm.solver_meta["status"] == "timelimit"
    assert warm.solver_meta["warm_start_used"] is True
    s_ref = reference_scale(problem, warm.weights)
    f_ref, ref_type = reference_from_exact(warm.solver_meta, scale=s_ref)
    assert ref_type == "best_known"
    assert f_ref == pytest.approx(warm.objective, rel=1e-6)
    # even a loose tolerance does not turn a missing dual bound into a certificate
    assert reference_from_exact(warm.solver_meta, scale=s_ref, tol=1e6)[1] == "best_known"
