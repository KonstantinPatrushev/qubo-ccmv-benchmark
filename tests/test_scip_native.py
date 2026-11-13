"""Tests for the native PySCIPOpt MIQP baseline (R5; turnover term for R3).

The exact reference is support enumeration: for every support of size ``≤ K``
the convex sub-problem is solved with :func:`solve_continuous_mvo` and the
best objective is kept. This is independent of SCIP and of the MIQP model.
"""

from __future__ import annotations

import gc
import json
import math
import weakref
from itertools import combinations, pairwise

import cvxpy as cp
import numpy as np
import pytest
from pyscipopt import SCIP_PARAMSETTING, Model

import cif.classical.scip_native as scip_native
from cif.classical.continuous import solve_continuous_mvo
from cif.classical.milp import solve_miqp_scip
from cif.classical.refine import refine_problem
from cif.classical.scip_native import _diagonal_shift, solve_miqp_native
from cif.instances.diagnostics import binding_diagnostics
from cif.instances.factor import factor_instance
from cif.instances.orlib import parse_orlib_port
from cif.metrics.gap import reference_scale
from cif.problem import PortfolioProblem
from tests.test_instances import orlib_text

REL_TOL = 1e-6
CERTIFIED = ("optimal", "gaplimit")  # SCIP stops with "gaplimit" once gap ≤ mip_gap


def _make_problem(
    n: int,
    seed: int,
    *,
    risk_aversion: float = 2.0,
    w_min: float = 0.0,
    w_max: float = 1.0,
    K: int | None = None,
) -> PortfolioProblem:
    """Synthetic instance, same recipe as ``tests/conftest.py::medium_problem``."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        w_min=w_min,
        w_max=w_max,
        risk_aversion=risk_aversion,
        cardinality=K,
    )


def _sub_problem(problem: PortfolioProblem, idx: tuple[int, ...]) -> PortfolioProblem:
    sel = np.asarray(idx)
    return PortfolioProblem(
        mu=problem.mu[sel],
        sigma=problem.sigma[np.ix_(sel, sel)],
        asset_names=tuple(problem.asset_names[i] for i in idx),
        w_min=problem.w_min,
        w_max=problem.w_max,
        budget=problem.budget,
        risk_aversion=problem.risk_aversion,
    )


def _support_enumeration(problem: PortfolioProblem, k_max: int) -> dict[int, float]:
    """Best objective over all supports of size ``≤ k`` for every ``k ≤ k_max``.

    With ``w_min > 0`` the sub-problem forces every asset of the subset to hold
    at least ``w_min``, so subsets of every size up to ``k`` are enumerated.
    """
    best_by_size: dict[int, float] = {}
    for k in range(1, k_max + 1):
        best = np.inf
        for idx in combinations(range(problem.n), k):
            try:
                sub = solve_continuous_mvo(_sub_problem(problem, idx))
            except RuntimeError:  # infeasible support under box / min-buy-in
                continue
            full = np.zeros(problem.n)
            full[list(idx)] = sub.weights
            best = min(best, problem.objective_value(full))
        best_by_size[k] = best
    running = np.inf
    out: dict[int, float] = {}
    for k in sorted(best_by_size):
        running = min(running, best_by_size[k])
        out[k] = running
    return out


def _rel_err(value: float, ref: float) -> float:
    return abs(value - ref) / abs(ref)


# ---------------------------------------------------------------------------
# 1. Exact reference by support enumeration (N ≤ 10, K ∈ {2, 3, 4})
# ---------------------------------------------------------------------------

_ENUM_CASES = {
    # λ = 10 diversifies the continuous optimum (support 5-6), so K ≤ 4 binds
    "lam10_n10": {"n": 10, "seed": 101, "risk_aversion": 10.0},
    "lam2_n6": {"n": 6, "seed": 102, "risk_aversion": 2.0},
    "buyin_n8": {"n": 8, "seed": 103, "risk_aversion": 5.0, "w_min": 0.1, "w_max": 0.6},
}
_K_VALUES = (2, 3, 4)


@pytest.fixture(scope="module")
def enumeration_reference():
    cache: dict[str, tuple[PortfolioProblem, dict[int, float]]] = {}

    def get(case: str) -> tuple[PortfolioProblem, dict[int, float]]:
        if case not in cache:
            problem = _make_problem(**_ENUM_CASES[case])
            cache[case] = (problem, _support_enumeration(problem, max(_K_VALUES)))
        return cache[case]

    return get


def test_enumeration_case_is_binding(enumeration_reference):
    """Guard against a vacuous test: on ``lam10_n10`` every tested K binds."""
    problem, ref = enumeration_reference("lam10_n10")
    cont = solve_continuous_mvo(problem)
    assert int((cont.weights > 1e-6).sum()) > max(_K_VALUES)
    for K in _K_VALUES:
        assert ref[K] > cont.objective + 1e-6 * abs(cont.objective)


@pytest.mark.parametrize("case", list(_ENUM_CASES))
@pytest.mark.parametrize(
    ("formulation", "diag"),
    [("bigm", "mineig"), ("perspective", "mineig"), ("perspective", "sdp")],
)
def test_matches_support_enumeration(enumeration_reference, case, formulation, diag):
    problem, ref = enumeration_reference(case)
    for seed, K in enumerate(_K_VALUES):
        sol = solve_miqp_native(problem, K, formulation=formulation, diag=diag, seed=seed)
        meta = sol.solver_meta
        assert sol.feasible
        assert meta["status"] in CERTIFIED
        assert meta["n_nonzero"] <= K
        # small instances often stop inside the root node by the gap limit
        assert meta["root_dual_bound"] is not None
        assert meta["root_dual_bound"] <= meta["dual_bound"] + 1e-9 * abs(meta["dual_bound"])
        assert _rel_err(sol.objective, ref[K]) <= REL_TOL, (
            f"{case} K={K} {formulation}/{diag}: {sol.objective!r} vs reference {ref[K]!r}"
        )
        if formulation == "perspective":
            assert meta["diag_method"] == diag
            assert 0.0 <= meta["d_min"] <= meta["d_max"]


# ---------------------------------------------------------------------------
# 2. big-M vs perspective on N = 25-30, K = 5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "seed"), [(25, 201), (30, 202)])
def test_bigm_and_perspective_agree_and_perspective_is_not_weaker(n, seed):
    problem = _make_problem(n, seed, K=5)
    bigm = solve_miqp_native(problem, 5, formulation="bigm", time_limit_s=60.0)
    persp = solve_miqp_native(problem, 5, formulation="perspective", time_limit_s=60.0)
    for sol in (bigm, persp):
        meta = sol.solver_meta
        assert sol.feasible
        assert meta["status"] in CERTIFIED
        # the dual bound only improves after the root
        assert meta["root_dual_bound"] <= meta["dual_bound"] + 1e-9 * abs(meta["dual_bound"])
    assert _rel_err(persp.objective, bigm.objective) <= REL_TOL
    # Hypothesis of the task: the perspective root bound is at least as tight
    # as the big-M one, or the tree is smaller.
    root_b = bigm.solver_meta["root_dual_bound"]
    root_p = persp.solver_meta["root_dual_bound"]
    nodes_b = bigm.solver_meta["n_nodes"]
    nodes_p = persp.solver_meta["n_nodes"]
    assert root_p >= root_b - 1e-9 or nodes_p < nodes_b, (
        f"root bound persp={root_p} bigm={root_b}; nodes persp={nodes_p} bigm={nodes_b}"
    )


# ---------------------------------------------------------------------------
# 3. Time limit is recognised and the bounds are consistent
# ---------------------------------------------------------------------------


def test_time_limit_is_reported():
    K = 10
    problem = _make_problem(150, 301, K=K)
    sol = solve_miqp_native(problem, K, time_limit_s=0.05)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit"
    assert meta["time_limit_hit"] is True
    assert meta["dual_bound"] <= meta["primal_bound"]
    for key in ("build_time_s", "solve_time_s", "build_cpu_s", "solve_cpu_s"):
        assert meta[key] >= 0.0
    if np.isfinite(sol.objective):
        assert meta["n_nonzero"] <= K
    else:
        assert sol.feasible is False
        assert meta["primal_bound"] == np.inf
        assert meta["gap"] == np.inf


# ---------------------------------------------------------------------------
# 4. Warm start
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("formulation", "with_turnover"),
    [("bigm", False), ("perspective", False), ("perspective", True)],
)
def test_warm_start_with_optimal_solution_is_accepted(formulation, with_turnover):
    K = 4
    problem = _make_problem(15, 401, K=K)
    kwargs: dict = {"formulation": formulation, "mip_gap": 0.0}
    if with_turnover:
        prev = np.zeros(problem.n)
        prev[:K] = 1.0 / K
        kwargs.update(prev_weights=prev, turnover_penalty=0.01)
    cold = solve_miqp_native(problem, K, **kwargs)
    assert cold.solver_meta["warm_start_used"] is False
    warm = solve_miqp_native(problem, K, warm_start=cold.weights, **kwargs)
    assert warm.solver_meta["warm_start_used"] is True
    # mip_gap = 0: SCIP must exhaust the tree, so the status is literally "optimal"
    assert warm.solver_meta["status"] == "optimal"
    assert _rel_err(warm.objective, cold.objective) <= REL_TOL


def test_infeasible_warm_start_is_rejected():
    K = 3
    problem = _make_problem(12, 402, K=K)
    dense = np.full(problem.n, 1.0 / problem.n)  # violates the cardinality limit
    sol = solve_miqp_native(problem, K, warm_start=dense)
    assert sol.solver_meta["warm_start_used"] is False
    assert sol.feasible
    assert sol.solver_meta["status"] in CERTIFIED


# ---------------------------------------------------------------------------
# 5. Turnover term (R3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formulation", ["bigm", "perspective"])
def test_huge_turnover_penalty_pins_previous_weights(formulation):
    K = 4
    problem = _make_problem(12, 501, K=K)
    prev = np.zeros(problem.n)
    prev[[0, 3, 5, 7]] = [0.4, 0.3, 0.2, 0.1]
    sol = solve_miqp_native(
        problem, K, formulation=formulation, prev_weights=prev, turnover_penalty=100.0
    )
    assert sol.feasible
    np.testing.assert_allclose(sol.weights, prev, atol=1e-6)
    assert sol.solver_meta["turnover_l1"] <= 1e-6


def _turnover_enumeration(
    problem: PortfolioProblem, K: int, prev: np.ndarray, penalty: float
) -> float:
    """Best ``f(w) + penalty·‖w - prev‖₁`` over all supports of size ``≤ K``."""
    best = np.inf
    for k in range(1, K + 1):
        for idx in combinations(range(problem.n), k):
            sel = list(idx)
            out = [i for i in range(problem.n) if i not in idx]
            w = cp.Variable(k)
            sigma_s = cp.psd_wrap(problem.sigma[np.ix_(sel, sel)])
            obj = (
                -problem.mu[sel] @ w
                + 0.5 * problem.risk_aversion * cp.quad_form(w, sigma_s)
                + penalty * (cp.sum(cp.abs(w - prev[sel])) + float(np.abs(prev[out]).sum()))
            )
            cons = [cp.sum(w) == problem.budget, w >= problem.w_min, w <= problem.w_max]
            prob = cp.Problem(cp.Minimize(obj), cons)
            prob.solve(solver="CLARABEL")
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                continue
            best = min(best, float(prob.value))
    return best


@pytest.mark.parametrize("formulation", ["bigm", "perspective"])
def test_turnover_objective_matches_support_enumeration(formulation):
    """R3 acceptance: MIQP with the turnover term equals enumeration on N ≤ 10."""
    K, penalty = 3, 0.02
    problem = _make_problem(8, 502, risk_aversion=5.0, K=K)
    prev = np.zeros(problem.n)
    prev[[1, 4, 6]] = [0.5, 0.3, 0.2]
    ref = _turnover_enumeration(problem, K, prev, penalty)
    sol = solve_miqp_native(
        problem, K, formulation=formulation, prev_weights=prev, turnover_penalty=penalty
    )
    assert sol.feasible
    total = sol.objective + penalty * float(np.abs(sol.weights - prev).sum())
    assert _rel_err(total, ref) <= REL_TOL
    assert sol.solver_meta["turnover_l1"] == pytest.approx(float(np.abs(sol.weights - prev).sum()))


# ---------------------------------------------------------------------------
# 6. Agreement with the legacy CVXPY → SCIP baseline
# ---------------------------------------------------------------------------


def test_matches_legacy_cvxpy_scip():
    K = 5
    problem = _make_problem(20, 601, K=K)
    legacy = solve_miqp_scip(problem, cardinality=K)
    native = solve_miqp_native(problem, K)
    assert legacy.solver_meta["status"] == "optimal"
    assert native.solver_meta["status"] in CERTIFIED
    # Legacy runs with SCIP's default feastol 1e-6, so its weights are only
    # accurate to ~1e-6 relative in the objective; compare at 1e-5.
    assert _rel_err(native.objective, legacy.objective) <= 1e-5
    legacy_support = set(np.flatnonzero(legacy.weights > 1e-6).tolist())
    native_support = set(np.flatnonzero(native.weights > 1e-6).tolist())
    assert native_support == legacy_support


# ---------------------------------------------------------------------------
# 7. The support never exceeds K
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formulation", ["bigm", "perspective"])
@pytest.mark.parametrize(
    ("n", "seed", "K", "w_min", "time_limit_s"),
    [(40, 701, 6, 0.0, 0.3), (40, 702, 6, 0.02, 0.3), (12, 703, 3, 0.0, None)],
)
def test_n_nonzero_never_exceeds_cardinality(formulation, n, seed, K, w_min, time_limit_s):
    problem = _make_problem(n, seed, w_min=w_min, K=K)
    sol = solve_miqp_native(problem, K, formulation=formulation, time_limit_s=time_limit_s)
    meta = sol.solver_meta
    if not np.isfinite(sol.objective):  # no incumbent within the limit
        assert sol.feasible is False
        return
    assert meta["n_nonzero"] == int((sol.weights > 1e-6).sum())
    assert meta["n_nonzero"] <= K
    assert sol.feasible


# ---------------------------------------------------------------------------
# Diagonal extraction and edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mineig", "sdp"])
def test_diagonal_shift_keeps_remainder_psd(method):
    problem = _make_problem(20, 801)
    d, used, fallback = _diagonal_shift(problem.sigma, method)
    assert used == method
    assert fallback is None
    assert np.all(d >= 0.0)
    assert np.linalg.eigvalsh(problem.sigma - np.diag(d))[0] >= 0.0


def test_sdp_diagonal_has_larger_trace_than_mineig():
    problem = _make_problem(20, 802)
    d_eig, _, _ = _diagonal_shift(problem.sigma, "mineig")
    d_sdp, _, _ = _diagonal_shift(problem.sigma, "sdp")
    assert d_sdp.sum() >= d_eig.sum()


def test_sdp_falls_back_to_mineig_on_singular_sigma():
    """Rank-deficient Σ: the SDP with margin 1e-8·I is infeasible."""
    n, K = 8, 3
    rng = np.random.default_rng(803)
    B = rng.standard_normal((n, n - 2)) * 0.1
    problem = PortfolioProblem(
        mu=rng.uniform(0.05, 0.25, size=n),
        sigma=B @ B.T,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=2.0,
        cardinality=K,
    )
    sol = solve_miqp_native(problem, K, formulation="perspective", diag="sdp")
    meta = sol.solver_meta
    assert meta["diag_method"] == "mineig"
    assert meta["diag_fallback"]
    assert meta["d_max"] == 0.0
    bigm = solve_miqp_native(problem, K, formulation="bigm")
    assert sol.feasible
    assert abs(sol.objective - bigm.objective) <= REL_TOL * abs(bigm.objective)


class _Panic(BaseException):
    """Stands in for ``pyo3_runtime.PanicException``: a Rust panic is a ``BaseException`` and not
    an ``Exception``, so ``except Exception`` lets it through."""


def _failing_solve(exc: BaseException):
    def solve(self, *args, **kwargs):
        raise exc

    return solve


def test_a_panic_of_the_sdp_solver_falls_back_to_mineig(monkeypatch):
    monkeypatch.setattr(cp.Problem, "solve", _failing_solve(_Panic("Eigval error: Eigen(1)")))
    sigma = _make_problem(12, 804).sigma
    d, used, fallback = _diagonal_shift(sigma, "sdp")
    assert used == "mineig"
    # both SDP solvers were tried; the reason names them and the panic
    assert fallback.count("_Panic: Eigval error: Eigen(1)") == 2
    assert fallback.startswith("CLARABEL:") and "SCS:" in fallback
    np.testing.assert_array_equal(d, _diagonal_shift(sigma, "mineig")[0])


def test_a_panic_of_the_sdp_solver_does_not_kill_the_run_and_is_recorded(monkeypatch):
    K = 3
    problem = _make_problem(10, 805, K=K)
    reference = solve_miqp_native(problem, K, diag="mineig")
    monkeypatch.setattr(cp.Problem, "solve", _failing_solve(_Panic("boom")))
    sol = solve_miqp_native(problem, K, diag="sdp")
    meta = sol.solver_meta
    assert meta["diag_method"] == "mineig" and "_Panic: boom" in meta["diag_fallback"]
    assert sol.feasible
    assert sol.objective == pytest.approx(reference.objective, rel=1e-9)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_during_the_sdp_is_not_swallowed(monkeypatch, interrupt):
    monkeypatch.setattr(cp.Problem, "solve", _failing_solve(interrupt()))
    with pytest.raises(interrupt):
        _diagonal_shift(_make_problem(8, 806).sigma, "sdp")


def test_clarabel_panic_on_a_rank_deficient_sigma_is_a_fallback_not_a_crash():
    """A real one. Σ = BBᵀ with B 8×6 (seed 1080, rank 6): Clarabel 0.11 panics inside its PSD cone
    ("Eigval error"), raised by pyo3 as ``PanicException``. Whatever a later Clarabel does with
    this Σ, the split must come back valid and the solver must stay usable."""
    rng = np.random.default_rng(1080)
    b = rng.standard_normal((8, 6)) * 0.1
    sigma = b @ b.T
    d, used, fallback = _diagonal_shift(sigma, "sdp")
    assert used == "mineig" and fallback
    assert np.all(d == 0.0)  # λ_min(Σ) = 0 to rounding: nothing to split off
    assert _diagonal_shift(_make_problem(10, 807).sigma, "sdp")[1] == "sdp"


# -- units of Σ: the SDP margin is relative to λ_max(Σ) ---------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1e-4, 1e-2, 1e2, 1e4])
def test_sdp_split_is_equivariant_to_the_units_of_sigma(scale):
    """Annual, daily or percent² units: Σ → cΣ must give d → c·d, and the SDP must be accepted in
    all of them. (With the former absolute margin 1e-8 it was rejected for c ≳ 1e2, so the 'sdp'
    variant silently was 'mineig', and for c = 1e-6 the margin took a fifth of Σd.)"""
    sigma = _factor_problem(30, 1000, None).sigma
    d_ref, used_ref, _ = _diagonal_shift(sigma, "sdp")
    assert used_ref == "sdp"
    d, used, fallback = _diagonal_shift(sigma * scale, "sdp")
    assert used == "sdp" and fallback is None
    # the solver's answer for single components is good to ~1e-6; its objective, the sum, to 1e-8
    assert d.sum() == pytest.approx(scale * d_ref.sum(), rel=1e-6)
    np.testing.assert_allclose(d, scale * d_ref, rtol=1e-4, atol=1e-6 * scale * d_ref.max())
    assert np.linalg.eigvalsh(sigma * scale - np.diag(d))[0] > 0.0
    assert d.sum() > _diagonal_shift(sigma * scale, "mineig")[0].sum()  # a real SDP gain


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_perspective_sdp_model_is_valid_and_has_the_same_optimum_in_any_units(scale):
    """Σ → cΣ with λ → λ/c is the same problem in other units: the split scales, the model stays
    convex, and the portfolio and its objective are those of the original units."""
    K = 4
    base = _factor_problem(24, 1003, K)
    units = PortfolioProblem(
        mu=base.mu, sigma=base.sigma * scale, asset_names=base.asset_names,
        risk_aversion=base.risk_aversion / scale, cardinality=K,
    )  # fmt: skip
    ref = solve_miqp_native(base, K, diag="sdp", mip_gap=1e-9)
    sol = solve_miqp_native(units, K, diag="sdp", mip_gap=1e-9)
    meta = sol.solver_meta
    assert ref.solver_meta["diag_method"] == meta["diag_method"] == "sdp"
    assert meta["d_max"] == pytest.approx(scale * ref.solver_meta["d_max"], rel=1e-4)
    assert sol.feasible and meta["status"] in CERTIFIED
    assert sol.objective == pytest.approx(ref.objective, rel=1e-6)
    np.testing.assert_allclose(sol.weights, ref.weights, atol=1e-6)


def test_infeasible_instance_reports_no_solution():
    problem = _make_problem(10, 901, w_max=0.2, K=4)  # 4 · 0.2 < budget
    sol = solve_miqp_native(problem, 4)
    meta = sol.solver_meta
    assert meta["status"] == "infeasible"
    assert sol.feasible is False
    assert sol.objective == np.inf
    assert np.all(sol.weights == 0.0)
    # SCIP's infinity (1e20) is reported as a true infinity, not a finite number;
    # dual bound = +inf is the infeasibility proof (SCIP then reports gap 0)
    assert meta["primal_bound"] == np.inf
    assert meta["dual_bound"] == np.inf


def test_bigm_meta_has_no_diagonal():
    problem = _make_problem(8, 902, K=3)
    meta = solve_miqp_native(problem, 3, formulation="bigm").solver_meta
    assert meta["formulation"] == "bigm"
    assert meta["diag_method"] is None
    assert meta["threads"] == 1
    assert meta["cardinality"] == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threads": 2},
        {"formulation": "socp"},
        {"diag": "trace"},
        {"warm_start": np.zeros(3)},
        {"prev_weights": np.zeros(3), "turnover_penalty": 0.1},
        {"turnover_penalty": -1.0, "prev_weights": np.zeros(8)},
        {"mip_gap": -1e-3},
    ],
)
def test_invalid_arguments_raise(kwargs):
    problem = _make_problem(8, 903, K=3)
    with pytest.raises(ValueError):
        solve_miqp_native(problem, 3, **kwargs)


# ---------------------------------------------------------------------------
# 8. Anytime infrastructure (ANALYSIS_PLAN §4–§6, §11): trajectory, seeds, absgap, Σz = K
# ---------------------------------------------------------------------------


def _factor_problem(
    n: int, seed: int, K: int | None, *, risk_aversion: float = 5.0, w_min: float = 0.0
) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"a{i}" for i in range(n)),
        risk_aversion=risk_aversion,
        w_min=w_min,
        cardinality=K,
    )


class _SpyModel(Model):
    """A ``Model`` that records every ``setParam`` call (the class attribute is reset per test)."""

    calls: list[tuple[str, object]] = []  # noqa: RUF012

    def setParam(self, name, value):
        type(self).calls.append((name, value))
        return super().setParam(name, value)

    def setHeuristics(self, setting):
        type(self).calls.append(("<heuristics>", setting))
        return super().setHeuristics(setting)


@pytest.fixture
def spy_params(monkeypatch):
    _SpyModel.calls = []
    monkeypatch.setattr(scip_native, "Model", _SpyModel)
    return _SpyModel.calls


# -- 8.1 incumbent trajectory ------------------------------------------------


@pytest.fixture(scope="module")
def trajectory_run():
    """A run with several incumbents: N = 36, K = 6 (about a second, 60 nodes)."""
    problem = _factor_problem(36, 1001, 6)
    return problem, solve_miqp_native(problem, 6, time_limit_s=60.0)


def test_trajectory_lists_every_new_best_solution(trajectory_run):
    problem, sol = trajectory_run
    meta = sol.solver_meta
    traj = meta["trajectory"]
    assert len(traj) >= 2
    times = [t for t, _ in traj]
    values = [f for _, f in traj]
    assert all(b >= a for a, b in pairwise(times))
    assert all(b < a for a, b in pairwise(values))  # strict improvements
    # the clock is the end-to-end time of the call: the search starts after the model is built
    assert times[0] >= meta["build_time_s"] - 1e-9
    assert times[-1] <= sol.wall_time_s + 1e-9
    # the last incumbent is the returned portfolio, evaluated on the true objective
    assert values[-1] == pytest.approx(sol.objective, rel=1e-12, abs=1e-15)
    assert values[-1] == pytest.approx(problem.objective_value(sol.weights), rel=1e-12)


def test_trajectory_primal_bounds_are_the_model_values_of_the_same_solutions(trajectory_run):
    _, sol = trajectory_run
    meta = sol.solver_meta
    bounds = meta["trajectory_primal_bound"]
    values = [f for _, f in meta["trajectory"]]
    assert len(bounds) == len(values)
    # SCIP's own objective of the stored solution (the event fires before the primal bound is
    # updated, so ``getPrimalbound`` would lag by one entry): it ends at the final primal bound
    assert bounds[-1] == meta["primal_bound"]
    # the model objective can undercut the true one only by the epigraph slack
    assert np.allclose(bounds, values, rtol=1e-6, atol=1e-9)


def test_trajectory_is_json_serialisable(trajectory_run):
    _, sol = trajectory_run
    meta = sol.solver_meta
    json.dumps({"trajectory": meta["trajectory"], "bounds": meta["trajectory_primal_bound"]})


def test_a_warm_start_is_the_incumbent_not_a_trajectory_entry():
    problem = _factor_problem(36, 1001, 6)
    idx = np.argsort(problem.mu)[::-1][:6]  # a feasible, clearly suboptimal start
    start = np.zeros(problem.n)
    start[idx] = 1.0 / 6.0
    f_start = problem.objective_value(start)
    sol = solve_miqp_native(problem, 6, warm_start=start, time_limit_s=60.0)
    assert sol.solver_meta["warm_start_used"] is True
    traj = sol.solver_meta["trajectory"]
    assert traj and all(f < f_start for _, f in traj)  # only improvements on the start are found


def test_trajectory_is_empty_without_a_solution():
    problem = _make_problem(10, 901, w_max=0.2, K=4)  # 4 · 0.2 < budget: infeasible
    meta = solve_miqp_native(problem, 4).solver_meta
    assert meta["trajectory"] == [] and meta["trajectory_primal_bound"] == []
    cold = solve_miqp_native(_factor_problem(40, 1000, 6), 6, time_limit_s=0.0)
    assert cold.solver_meta["trajectory"] == []


# -- 8.2 seeds: permutation of the model ----------------------------------------


def test_permutation_seed_sets_the_scip_parameters(spy_params):
    problem = _make_problem(8, 902, K=3)
    meta = solve_miqp_native(problem, 3, seed=5, permutation_seed=7).solver_meta
    assert ("randomization/permutevars", True) in spy_params
    assert ("randomization/permutationseed", 7) in spy_params
    assert ("randomization/randomseedshift", 5) in spy_params
    assert meta["permutation_seed"] == 7 and meta["seed"] == 5


def test_without_a_permutation_seed_the_permutation_parameters_are_untouched(spy_params):
    problem = _make_problem(8, 902, K=3)
    meta = solve_miqp_native(problem, 3, seed=5).solver_meta
    names = {name for name, _ in spy_params}
    assert "randomization/permutevars" not in names
    assert "randomization/permutationseed" not in names
    assert meta["permutation_seed"] is None


def test_permutation_seeds_are_reproducible_and_change_the_search():
    problem = _factor_problem(24, 1003, 4)

    def run(permutation_seed):
        sol = solve_miqp_native(
            problem, 4, seed=permutation_seed, permutation_seed=permutation_seed
        )
        return sol.solver_meta["n_nodes"], sol.objective

    first = {s: run(s) for s in (1, 2, 3)}
    again = {s: run(s) for s in (1, 2, 3)}
    assert first == again  # SCIP is deterministic for fixed seeds
    assert len({nodes for nodes, _ in first.values()}) >= 2  # different seeds, different trees
    objectives = [obj for _, obj in first.values()]
    assert max(objectives) - min(objectives) <= 1e-6 * abs(min(objectives))  # same optimum


def test_a_negative_permutation_seed_is_rejected():
    with pytest.raises(ValueError, match="permutation_seed"):
        solve_miqp_native(_make_problem(8, 903, K=3), 3, permutation_seed=-1)


# -- 8.3 absolute gap limit -------------------------------------------------------


def test_absgap_is_passed_to_scip_and_reported(spy_params):
    problem = _make_problem(8, 902, K=3)
    meta = solve_miqp_native(problem, 3, mip_gap=0.0, abs_gap=1e-4).solver_meta
    assert ("limits/absgap", 1e-4) in spy_params
    assert meta["abs_gap_limit"] == 1e-4
    assert 0.0 <= meta["abs_gap"] <= 1e-4 + 1e-12  # |primal − dual| at the stop
    assert meta["abs_gap"] == pytest.approx(abs(meta["primal_bound"] - meta["dual_bound"]))


def test_without_absgap_the_parameter_is_untouched(spy_params):
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3).solver_meta
    assert "limits/absgap" not in {name for name, _ in spy_params}
    assert meta["abs_gap_limit"] is None
    assert math.isfinite(meta["abs_gap"])


def test_the_achieved_absolute_gap_is_infinite_while_a_bound_is_missing():
    infeasible = solve_miqp_native(_make_problem(10, 901, w_max=0.2, K=4), 4).solver_meta
    assert infeasible["abs_gap"] == math.inf
    cold = solve_miqp_native(_factor_problem(40, 1000, 6), 6, time_limit_s=0.0).solver_meta
    assert cold["abs_gap"] == math.inf  # no incumbent within the limit


def test_a_loose_absgap_stops_at_once_with_the_gaplimit_status():
    problem = _factor_problem(36, 1001, 6)
    tight = solve_miqp_native(problem, 6, mip_gap=0.0)
    loose = solve_miqp_native(problem, 6, mip_gap=0.0, abs_gap=1.0)
    assert loose.solver_meta["status"] == "gaplimit"
    assert loose.solver_meta["n_nodes"] < tight.solver_meta["n_nodes"]
    assert loose.solver_meta["abs_gap"] <= 1.0


def test_absgap_of_1e6_s_ref_certifies_the_optimum_on_the_s_ref_scale(enumeration_reference):
    # §5: limits/absgap = 1e-6·s_ref; the optimum is within that of the enumerated one
    problem, ref = enumeration_reference("lam10_n10")
    for K in _K_VALUES:
        s_ref = reference_scale(problem, solve_miqp_native(problem, K).weights)
        sol = solve_miqp_native(problem, K, mip_gap=0.0, abs_gap=1e-6 * s_ref)
        meta = sol.solver_meta
        assert meta["status"] in CERTIFIED
        assert meta["abs_gap"] <= 1e-6 * s_ref * (1 + 1e-9)
        assert sol.objective - ref[K] <= 1e-6 * s_ref + 1e-9
        assert sol.objective >= ref[K] - 1e-9  # nothing below the enumerated optimum


@pytest.mark.parametrize("abs_gap", [-1e-6, math.nan, math.inf])
def test_absgap_must_be_a_finite_non_negative_number(abs_gap):
    with pytest.raises(ValueError, match="abs_gap"):
        solve_miqp_native(_make_problem(8, 903, K=3), 3, abs_gap=abs_gap)


# -- 8.4 Σz = K (ORlib, Chang et al.: every one of the K assets held at least ε) --------


def test_eq_mode_matches_the_enumeration_of_exact_size_k_supports():
    K = 8
    problem = _factor_problem(12, 1002, K, w_min=0.03)
    assert binding_diagnostics(problem, K=K)["s_cont"] < K  # the supports of 'le' have < K assets
    ref = min(
        refine_problem(problem, list(s), card_mode="eq").objective
        for s in combinations(range(problem.n), K)
    )
    for formulation in ("bigm", "perspective"):
        sol = solve_miqp_native(problem, K, formulation=formulation, card_mode="eq")
        meta = sol.solver_meta
        assert sol.feasible and meta["status"] in CERTIFIED
        assert meta["card_mode"] == "eq"
        assert meta["n_nonzero"] == K
        assert np.all(sol.weights[sol.weights > 1e-6] >= problem.w_min - 1e-6)
        assert _rel_err(sol.objective, ref) <= REL_TOL


def test_eq_mode_is_stricter_than_le_when_fewer_than_k_assets_suffice():
    K = 8
    problem = _factor_problem(12, 1002, K, w_min=0.03)
    le = solve_miqp_native(problem, K, card_mode="le")
    eq = solve_miqp_native(problem, K, card_mode="eq")
    assert le.solver_meta["card_mode"] == "le"
    assert le.solver_meta["n_nonzero"] < K
    assert eq.solver_meta["n_nonzero"] == K
    assert eq.objective > le.objective + 1e-6


def test_eq_mode_without_a_minimum_purchase_equals_le():
    # w_min = 0: an asset with z = 1 may weigh 0, so Σz = K costs nothing
    problem = _factor_problem(12, 1002, 5)
    le = solve_miqp_native(problem, 5, card_mode="le")
    eq = solve_miqp_native(problem, 5, card_mode="eq")
    assert eq.feasible and eq.solver_meta["n_nonzero"] <= 5
    assert _rel_err(eq.objective, le.objective) <= REL_TOL


def test_eq_mode_accepts_a_warm_start_with_exactly_k_assets_only():
    K = 8
    problem = _factor_problem(12, 1002, K, w_min=0.03)
    start = np.zeros(problem.n)
    start[:K] = 1.0 / K
    ok = solve_miqp_native(problem, K, card_mode="eq", warm_start=start)
    assert ok.solver_meta["warm_start_used"] is True
    short = np.zeros(problem.n)
    short[: K - 1] = 1.0 / (K - 1)  # K − 1 assets violate Σz = K
    rejected = solve_miqp_native(problem, K, card_mode="eq", warm_start=short)
    assert rejected.solver_meta["warm_start_used"] is False


def test_card_mode_defaults_to_le_and_is_validated():
    problem = _make_problem(8, 903, K=3)
    assert solve_miqp_native(problem, 3).solver_meta["card_mode"] == "le"
    with pytest.raises(ValueError, match="card_mode"):
        solve_miqp_native(problem, 3, card_mode="ge")
    no_card = _make_problem(8, 903)
    with pytest.raises(ValueError, match="cardinality"):
        solve_miqp_native(no_card, None, card_mode="eq")


# -- 8.5 forbidden supersets (RQ3 k-best, ANALYSIS_PLAN §2 (г)) --------------------


def _allowed_objectives(problem, K, forbidden, *, card_mode="le"):
    """Refined objective of every support Z with ``|Z| ≤ K`` (``= K`` for ``"eq"``) that contains
    none of the ``forbidden`` sets; independent of the MIQP."""
    banned = [frozenset(t) for t in forbidden]
    sizes = [K] if card_mode == "eq" else range(1, K + 1)
    out = {}
    for size in sizes:
        for z in combinations(range(problem.n), size):
            if any(t <= set(z) for t in banned):
                continue
            res = refine_problem(problem, z, card_mode=card_mode)
            if res.status == "optimal":
                out[z] = res.objective
    return out


def _support(sol, tol=1e-6):
    return {int(i) for i in np.flatnonzero(sol.weights > tol)}


@pytest.fixture(scope="module")
def cut_case():
    """N = 9, K = 3, binding: the optimum's support, the second portfolio's support."""
    problem = _make_problem(9, 904, K=3, risk_aversion=4.0)
    first = solve_miqp_native(problem, 3, mip_gap=0.0)
    t1 = tuple(sorted(_support(first)))
    second = solve_miqp_native(problem, 3, mip_gap=0.0, forbidden_supersets=[t1])
    t2 = tuple(sorted(_support(second)))
    return problem, t1, t2


@pytest.mark.parametrize("formulation", ["bigm", "perspective"])
def test_a_forbidden_superset_is_excluded_and_nothing_else(cut_case, formulation):
    problem, t1, _ = cut_case
    ref = min(_allowed_objectives(problem, 3, [t1]).values())
    sol = solve_miqp_native(problem, 3, formulation=formulation, mip_gap=0.0,
                            forbidden_supersets=[t1])
    assert sol.feasible and sol.solver_meta["status"] in CERTIFIED
    assert not set(t1) <= _support(sol)
    assert _rel_err(sol.objective, ref) <= REL_TOL
    # the cut costs something: the optimum itself is no longer allowed
    assert sol.objective > solve_miqp_native(problem, 3, mip_gap=0.0).objective + 1e-6


@pytest.mark.parametrize("formulation", ["bigm", "perspective"])
def test_several_forbidden_supersets_are_all_enforced(cut_case, formulation):
    problem, t1, t2 = cut_case
    assert set(t1) != set(t2)
    ref = min(_allowed_objectives(problem, 3, [t1, t2]).values())
    sol = solve_miqp_native(problem, 3, formulation=formulation, mip_gap=0.0,
                            forbidden_supersets=[t1, t2])
    assert not set(t1) <= _support(sol)
    assert not set(t2) <= _support(sol)
    assert _rel_err(sol.objective, ref) <= REL_TOL


def test_a_forbidden_set_of_size_one_removes_the_asset(cut_case):
    problem, t1, _ = cut_case
    banned = t1[0]
    ref = min(_allowed_objectives(problem, 3, [(banned,)]).values())
    sol = solve_miqp_native(problem, 3, mip_gap=0.0, forbidden_supersets=[(banned,)])
    assert sol.weights[banned] == 0.0
    assert _rel_err(sol.objective, ref) <= REL_TOL


def test_a_forbidden_set_the_optimum_does_not_contain_changes_nothing(cut_case):
    problem, t1, _ = cut_case
    outsiders = [i for i in range(problem.n) if i not in t1][:3]  # not a subset of the optimum
    base = solve_miqp_native(problem, 3, mip_gap=0.0)
    cut = solve_miqp_native(problem, 3, mip_gap=0.0, forbidden_supersets=[tuple(outsiders)])
    assert _rel_err(cut.objective, base.objective) <= REL_TOL
    assert _support(cut) == set(t1)


def test_proper_subsets_of_a_forbidden_set_stay_allowed(cut_case):
    # forbidding {a, b, c} must not forbid {a, b}: the cut removes supersets, not subsets
    problem, t1, _ = cut_case
    allowed = _allowed_objectives(problem, 3, [t1])
    assert any(set(z) < set(t1) for z in allowed)  # the guard: proper subsets are allowed ones
    sol = solve_miqp_native(problem, 3, mip_gap=0.0, forbidden_supersets=[t1])
    assert _rel_err(sol.objective, min(allowed.values())) <= REL_TOL


def test_forbidden_supersets_in_eq_mode():
    K = 8
    problem = _factor_problem(12, 1002, K, w_min=0.03)
    banned = [(0, 1, 2), (3, 4)]
    ref = min(_allowed_objectives(problem, K, banned, card_mode="eq").values())
    base = solve_miqp_native(problem, K, card_mode="eq")
    sol = solve_miqp_native(problem, K, card_mode="eq", forbidden_supersets=banned)
    held = _support(sol)
    assert sol.feasible and sol.solver_meta["n_nonzero"] == K
    assert not set(banned[0]) <= held and not set(banned[1]) <= held
    assert _rel_err(sol.objective, ref) <= REL_TOL
    assert sol.objective >= base.objective - 1e-9


def test_the_number_of_forbidden_supersets_is_recorded(cut_case):
    problem, t1, t2 = cut_case
    assert solve_miqp_native(problem, 3).solver_meta["n_forbidden_supersets"] == 0
    meta = solve_miqp_native(problem, 3, forbidden_supersets=[t1, t2]).solver_meta
    assert meta["n_forbidden_supersets"] == 2


def test_an_empty_list_of_forbidden_supersets_changes_nothing(cut_case):
    problem, _, _ = cut_case
    a = solve_miqp_native(problem, 3, mip_gap=0.0)
    b = solve_miqp_native(problem, 3, mip_gap=0.0, forbidden_supersets=[])
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.solver_meta["n_nodes"] == b.solver_meta["n_nodes"]
    assert b.solver_meta["n_forbidden_supersets"] == 0


def test_forbidden_supersets_can_make_the_model_infeasible():
    # N = 3, u = 0.5: at least two assets are needed; with every asset forbidden none is left
    problem = _make_problem(3, 905, w_max=0.5, K=2)
    sol = solve_miqp_native(problem, 2, forbidden_supersets=[(0,), (1,), (2,)])
    assert sol.solver_meta["status"] == "infeasible"
    assert not sol.feasible and sol.objective == math.inf
    assert sol.solver_meta["n_forbidden_supersets"] == 3


def test_a_warm_start_that_contains_a_forbidden_set_is_rejected(cut_case):
    problem, t1, t2 = cut_case
    w1 = refine_problem(problem, t1).weights
    w2 = refine_problem(problem, t2).weights
    rejected = solve_miqp_native(problem, 3, forbidden_supersets=[t1], warm_start=w1)
    assert rejected.solver_meta["warm_start_used"] is False
    accepted = solve_miqp_native(problem, 3, forbidden_supersets=[t1], warm_start=w2)
    assert accepted.solver_meta["warm_start_used"] is True


@pytest.mark.parametrize(
    "bad",
    [[()], [(-1, 2)], [(0, 9)], [(1, 1, 2)], [(0, 1.5)]],
    ids=["empty", "negative", "out-of-range", "duplicate", "non-integer"],
)
def test_invalid_forbidden_supersets_raise(bad):
    with pytest.raises(ValueError, match="forbidden_supersets"):
        solve_miqp_native(_make_problem(9, 904, K=3), 3, forbidden_supersets=bad)


# -- 8.6 end-to-end budget on the common clock (ANALYSIS_PLAN §5-§6) -----------------------------
#
# ``time_limit_s`` is SCIP's ``limits/time`` and counts solving only (certification runs rely on
# it). ``budget_s`` is the RQ2 budget: seconds of the common clock, the diagonal / SDP, the model,
# the warm start and the solve all inside it, and ``clock_offset_s`` of it already spent on the
# construction of the common start.


def _start_weights(problem: PortfolioProblem, k: int) -> np.ndarray:
    idx = np.argsort(problem.mu)[::-1][:k]  # a feasible, clearly suboptimal start
    start = np.zeros(problem.n)
    start[idx] = 1.0 / k
    return start


def test_the_budget_leaves_scip_what_the_build_did_not_use(spy_params):
    problem = _factor_problem(24, 1003, 4)
    meta = solve_miqp_native(problem, 4, budget_s=100.0, clock_offset_s=3.0).solver_meta
    limit = dict(spy_params)["limits/time"]
    assert limit == pytest.approx(100.0 - 3.0 - meta["build_time_s"], abs=1e-9)
    assert meta["solve_limit_s"] == limit
    assert meta["budget_s"] == 100.0 and meta["clock_offset_s"] == 3.0
    assert meta["time_limit_s"] is None and meta["time_limit_hit"] is False


def test_time_limit_still_counts_the_solve_only(spy_params):
    # the meaning certification (E1, the oracle) relies on: the build is not in the limit
    meta = solve_miqp_native(_factor_problem(24, 1003, 4), 4, time_limit_s=60.0).solver_meta
    assert dict(spy_params)["limits/time"] == 60.0
    assert meta["solve_limit_s"] == 60.0 and meta["budget_s"] is None
    assert meta["clock_offset_s"] == 0.0


def test_neither_limit_leaves_the_time_parameter_untouched(spy_params):
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3).solver_meta
    assert "limits/time" not in {name for name, _ in spy_params}
    assert meta["solve_limit_s"] is None


def test_scip_is_told_to_count_wall_clock_time(spy_params):
    # limits/time is on SCIP's clock; the budget is a wall-clock one, whatever the SCIP default is
    solve_miqp_native(_make_problem(8, 902, K=3), 3, budget_s=10.0)
    assert ("timing/clocktype", 2) in spy_params


# SCIP's symmetry detection (presolve) does not look at the clock: on a dense N = 400 model it ran
# from 5.7 s to 23.2 s ("no symmetry present") inside a 10 s budget, and the call ended 6-14 s late;
# with ``misc/usesymmetry = 0`` it ended 0.03 s late. The study uses SCIP's defaults (ANALYSIS_PLAN
# §4), so switching it off is a deviation to be decided and logged, hence an explicit, recorded
# argument and not a new default.


def test_scip_params_are_set_after_the_defaults_and_recorded(spy_params):
    params = {"misc/usesymmetry": 0}
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3, scip_params=params).solver_meta
    assert ("misc/usesymmetry", 0) in spy_params
    assert meta["scip_params"] == {"misc/usesymmetry": 0}
    assert params == {"misc/usesymmetry": 0}  # the caller's dict is not touched


def test_without_scip_params_nothing_beyond_the_defaults_is_set(spy_params):
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3).solver_meta
    assert meta["scip_params"] == {}
    assert "misc/usesymmetry" not in {name for name, _ in spy_params}


@pytest.mark.parametrize(
    "name",
    [
        "limits/time", "limits/gap", "limits/absgap", "numerics/feastol", "lp/threads",
        "timing/clocktype", "randomization/randomseedshift", "randomization/permutevars",
        "randomization/permutationseed",
    ],
)  # fmt: skip
def test_parameters_that_have_their_own_argument_cannot_be_set_through_scip_params(name):
    with pytest.raises(ValueError, match=name):
        solve_miqp_native(_make_problem(8, 902, K=3), 3, scip_params={name: 1})


def test_an_unknown_scip_parameter_raises_with_its_name():
    with pytest.raises(ValueError, match="misc/no_such_parameter"):
        solve_miqp_native(_make_problem(8, 902, K=3), 3, scip_params={"misc/no_such_parameter": 1})


@pytest.mark.parametrize(
    ("time_limit_s", "budget_s", "expected"),
    [(5.0, 100.0, 5.0), (500.0, 100.0, None), (0.0, 100.0, 0.0)],
)
def test_the_smaller_of_the_two_limits_binds(spy_params, time_limit_s, budget_s, expected):
    meta = solve_miqp_native(
        _factor_problem(24, 1003, 4), 4, time_limit_s=time_limit_s, budget_s=budget_s
    ).solver_meta
    limit = dict(spy_params)["limits/time"]
    if expected is None:  # the budget: what is left of it after the build
        assert limit == pytest.approx(budget_s - meta["build_time_s"], abs=1e-9)
        assert limit < time_limit_s
    else:
        assert limit == expected


def test_a_budget_spent_before_the_call_returns_the_start(spy_params):
    problem = _factor_problem(36, 1001, 6)
    start = _start_weights(problem, 6)
    sol = solve_miqp_native(problem, 6, warm_start=start, budget_s=1.0, clock_offset_s=2.0)
    meta = sol.solver_meta
    assert dict(spy_params)["limits/time"] == 0.0 and meta["solve_limit_s"] == 0.0
    assert meta["status"] == "timelimit" and meta["time_limit_hit"] is True
    assert meta["warm_start_used"] is True and meta["trajectory"] == []
    assert sol.feasible and sol.objective == pytest.approx(problem.objective_value(start))
    assert sol.wall_time_s >= 2.0  # the clock is the common one


def test_a_binding_budget_stops_scip_and_the_call_ends_near_it():
    K = 10
    problem = _make_problem(150, 301, K=K)
    budget = 0.5
    sol = solve_miqp_native(problem, K, budget_s=budget)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit" and meta["time_limit_hit"] is True
    assert meta["dual_bound"] <= meta["primal_bound"]
    # SCIP may overrun its limit by a presolve / LP / heuristic step: a coarse bound only
    assert sol.wall_time_s < budget + 5.0
    assert meta["build_time_s"] + meta["solve_time_s"] <= sol.wall_time_s + 1e-3


def test_the_offset_moves_the_trajectory_and_the_wall_time_to_the_common_clock():
    problem = _factor_problem(36, 1001, 6)
    offset = 5.0
    sol = solve_miqp_native(problem, 6, time_limit_s=60.0, clock_offset_s=offset)
    meta = sol.solver_meta
    times = [t for t, _ in meta["trajectory"]]
    assert times and all(a <= b for a, b in pairwise(times))
    assert times[0] >= offset + meta["build_time_s"] - 1e-9  # the search starts after the build
    assert times[-1] <= sol.wall_time_s + 1e-9
    assert offset + meta["build_time_s"] + meta["solve_time_s"] <= sol.wall_time_s + 1e-3
    assert sol.wall_time_s < offset + 60.0


def test_solver_meta_is_json_serialisable_as_it_is():
    """The registry writes rows with ``default=str``: a numpy bool in the metadata (``budget_ok`` was
    one) would be stored as the string "True" and break any filter on the field."""
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3, budget_s=30.0, clock_offset_s=1.0).solver_meta
    back = json.loads(json.dumps(meta))
    assert back["budget_ok"] is True and back["box_ok"] is True and back["card_ok"] is True
    assert back["budget_s"] == 30.0 and back["clock_offset_s"] == 1.0 and back["scip_params"] == {}


def test_the_offset_without_a_budget_changes_neither_the_search_nor_the_result():
    problem = _factor_problem(24, 1003, 4)
    plain = solve_miqp_native(problem, 4)
    shifted = solve_miqp_native(problem, 4, clock_offset_s=7.0)
    assert shifted.objective == plain.objective
    np.testing.assert_array_equal(shifted.weights, plain.weights)
    assert [f for _, f in shifted.solver_meta["trajectory"]] == [
        f for _, f in plain.solver_meta["trajectory"]
    ]
    assert shifted.wall_time_s >= 7.0 > plain.wall_time_s


def test_build_and_diagonal_times_are_recorded():
    problem = _factor_problem(24, 1003, 4)
    sdp = solve_miqp_native(problem, 4, diag="sdp").solver_meta
    mineig = solve_miqp_native(problem, 4, diag="mineig").solver_meta
    bigm = solve_miqp_native(problem, 4, formulation="bigm").solver_meta
    assert 0.0 < sdp["diag_time_s"] <= sdp["build_time_s"]
    assert 0.0 <= mineig["diag_time_s"] <= mineig["build_time_s"]
    assert sdp["diag_time_s"] > mineig["diag_time_s"]
    assert bigm["diag_time_s"] == 0.0  # big-M has no diagonal
    for meta in (sdp, mineig, bigm):
        assert meta["build_time_s"] > 0.0 and meta["solve_time_s"] >= 0.0


class _SolveSpy:
    """Records the keyword arguments of every ``cp.Problem.solve`` and calls through."""

    def __init__(self, monkeypatch):
        self.calls: list[dict] = []
        original = cp.Problem.solve

        def spy(problem, *args, **kwargs):
            self.calls.append(kwargs)
            return original(problem, *args, **kwargs)

        monkeypatch.setattr(cp.Problem, "solve", spy)


def test_the_sdp_gets_the_remaining_budget_as_its_time_limit(monkeypatch):
    spy = _SolveSpy(monkeypatch)
    problem = _factor_problem(24, 1003, 4)
    meta = solve_miqp_native(problem, 4, diag="sdp", budget_s=50.0, clock_offset_s=10.0).solver_meta
    assert meta["diag_method"] == "sdp"
    assert spy.calls[0]["solver"] == "CLARABEL"
    assert 39.0 < spy.calls[0]["time_limit"] <= 40.0  # 50 - 10 minus the little spent so far


def test_the_sdp_is_not_limited_by_the_solve_only_time_limit_or_without_a_budget(monkeypatch):
    spy = _SolveSpy(monkeypatch)
    problem = _factor_problem(24, 1003, 4)
    solve_miqp_native(problem, 4, diag="sdp", time_limit_s=5.0)
    solve_miqp_native(problem, 4, diag="sdp")
    assert [("time_limit" in call) for call in spy.calls] == [False, False]


def test_no_sdp_is_started_when_the_budget_is_already_spent(monkeypatch):
    spy = _SolveSpy(monkeypatch)
    problem = _factor_problem(24, 1003, 4)
    start = _start_weights(problem, 4)
    sol = solve_miqp_native(problem, 4, diag="sdp", warm_start=start, budget_s=1.0, clock_offset_s=2.0)
    meta = sol.solver_meta
    assert spy.calls == []
    assert meta["diag_method"] == "mineig" and "budget" in meta["diag_fallback"]
    assert meta["warm_start_used"] is True and sol.feasible


@pytest.mark.parametrize("solver", scip_native._SDP_SOLVERS)
def test_the_installed_sdp_solvers_accept_the_options_that_are_passed(solver):
    # a renamed option would make cvxpy raise, i.e. every budgeted SDP would silently be "mineig"
    n = 6
    d = cp.Variable(n)
    sdp = cp.Problem(
        cp.Maximize(cp.sum(d)), [np.eye(n) - cp.diag(d) >> 0.1 * np.eye(n), d >= 0]
    )
    options = {scip_native._SDP_TIME_OPTION[solver]: 30.0}
    options.update(scip_native._SDP_THREAD_OPTIONS.get(solver, {}))
    sdp.solve(solver=solver, **options)
    assert sdp.status in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(d.value, 0.9, atol=1e-3)


def test_the_sdp_is_not_attempted_where_clarabel_would_exhaust_the_memory(monkeypatch):
    """Clarabel holds a dense block of (N(N+1)/2)² doubles: peak 1.4 GB at N = 100, 6.5 GB at
    N = 150, a failed 1.6 GB allocation after 8 GB at N = 200, 51 GB for one buffer at N = 400. An
    allocation failure aborts the whole process (SIGABRT), which no ``except`` can catch, so above
    ``_SDP_MAX_N`` the SDP is a recorded fallback and not a crash."""
    attempted: list[int] = []

    def fake_solve(problem, *args, **kwargs):
        attempted.append(problem.variables()[0].size)
        raise RuntimeError("stop here: only whether the SDP was started matters")

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    limit = scip_native._SDP_MAX_N
    inside = _diagonal_shift(np.eye(limit) + 0.1, "sdp")
    assert set(attempted) == {limit} and inside[1] == "mineig"  # attempted (and, here, failed)
    attempted.clear()
    d, used, fallback = _diagonal_shift(np.eye(limit + 1) + 0.1, "sdp")
    assert attempted == []
    assert used == "mineig" and f"N = {limit + 1}" in fallback and "memory" in fallback
    np.testing.assert_array_equal(d, _diagonal_shift(np.eye(limit + 1) + 0.1, "mineig")[0])


def test_the_sdp_runs_clarabel_on_one_thread(monkeypatch):
    """ANALYSIS_PLAN §6: one thread for every solver, and the SDP is inside the timed budget.
    Clarabel's default is automatic and its PSD cone then used two threads (CPU time 1.8-2.0 times
    the wall time at N = 70, whatever OMP_NUM_THREADS says)."""
    spy = _SolveSpy(monkeypatch)
    _diagonal_shift(_factor_problem(12, 1000, None).sigma, "sdp")
    assert spy.calls[0]["solver"] == "CLARABEL" and spy.calls[0]["max_threads"] == 1


def test_an_sdp_stopped_by_its_time_limit_falls_back_to_mineig():
    # N = 60 takes 2-6 s unrestricted; Clarabel checks the clock once per iteration
    sigma = _factor_problem(60, 1000, None).sigma
    d, used, fallback = _diagonal_shift(sigma, "sdp", time_limit_s=0.05)
    assert used == "mineig" and fallback.startswith("CLARABEL: status user_limit")
    assert "SCS" in fallback  # the next solver is tried (or skipped) and reported too
    np.testing.assert_array_equal(d, _diagonal_shift(sigma, "mineig")[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_s": -1.0},
        {"budget_s": math.nan},
        {"budget_s": math.inf},
        {"clock_offset_s": -1.0},
        {"clock_offset_s": math.nan},
        {"clock_offset_s": math.inf},
    ],
)
def test_invalid_budget_arguments_raise(kwargs):
    with pytest.raises(ValueError, match=r"budget_s|clock_offset_s"):
        solve_miqp_native(_make_problem(8, 903, K=3), 3, **kwargs)


# -- memory: a solved model must not outlive the call ------------------------------------------


def test_a_solved_model_is_released_when_the_call_returns(monkeypatch):
    # PySCIPOpt 6.1: Model.catchEvent increfs the model and only dropEvent releases it; the handlers call
    # dropEvent in their EXIT callback, which SCIP runs when the transformed problem is freed. A call that
    # left the transformed problem alone kept its model alive for the rest of the process: 13 MB of SCIP
    # per MIQP at N = 49, 800 MB for the 33 MIQPs of one E3 cell, until the shard's process ended.
    refs = []

    class _Tracked(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            refs.append(weakref.ref(self))

    monkeypatch.setattr(scip_native, "Model", _Tracked)
    solved = solve_miqp_native(_make_problem(8, 902, K=3), 3)
    infeasible = solve_miqp_native(_make_problem(10, 901, w_max=0.2, K=4), 4)  # 4 · 0.2 < budget
    stopped = solve_miqp_native(_factor_problem(40, 1000, 6), 6, time_limit_s=0.0)
    assert solved.feasible and not infeasible.feasible and not stopped.feasible
    gc.collect()
    assert len(refs) == 3 and [r() is None for r in refs] == [True, True, True]


# -- 9. one optional linear row a·w ≥ b (M11: the minimum return of Bertsimas & Cory-Wright, Table 4) -----
#
# ``linear_ge=(a, b)`` adds the row ``a·w ≥ b`` to the model; for the validation of the exact stack
# ``a = μ`` of the data and ``b = r̄`` while the objective carries no return (κ = 0). The exact reference is
# support enumeration with the row, one convex QP per support (cvxpy / CLARABEL): independent of SCIP and
# of the MIQP model.

# (n, seed, K, share of the way from r_min to r_max). ``switch``: the rule of the paper (0.3); the row cuts off
# the free optimum, and the constrained one moves to another support whose own return is above r̄ (the row is
# slack there). ``active``: a higher r̄, the row holds with equality at the optimum.
_ROW_CASES = {"switch": (9, 1301, 3, 0.3), "active": (9, 1303, 4, 0.8)}


def _return_row_problem(n: int, seed: int, K: int, share: float = 0.3):
    """BCW Table 4 in miniature on a synthetic port file: ``min ½wᵀ(Σ + I/γ)w`` (κ = 0, γ = 100/√n) and the
    row ``μᵀw ≥ r̄`` with ``r̄ = r_min + share·(r_max − r_min)`` (``share`` = 0.3 in the paper). Returns
    ``(problem, mu, r_bar)``; ``problem.mu`` is zero: the return is in the row only."""
    inst = parse_orlib_port(orlib_text(n, seed))
    gamma = 100.0 / math.sqrt(n)
    names = tuple(f"a{i}" for i in range(n))
    sigma = inst.sigma + np.eye(n) / gamma
    zero = np.zeros(n)
    x_min = solve_continuous_mvo(PortfolioProblem(mu=zero, sigma=sigma, asset_names=names)).weights
    x_max = solve_continuous_mvo(
        PortfolioProblem(mu=inst.mu, sigma=np.eye(n) / gamma, asset_names=names)
    ).weights
    r_min, r_max = float(inst.mu @ x_min), float(inst.mu @ x_max)
    r_bar = r_min + share * (r_max - r_min)
    return PortfolioProblem(mu=zero, sigma=sigma, asset_names=names, cardinality=K), inst.mu, r_bar


def _support_enumeration_with_row(problem: PortfolioProblem, a: np.ndarray, b: float, k_max: int):
    """Best ``(objective, weights)`` over all supports of size ``≤ k_max`` with the row ``a·w ≥ b``: for each
    support the convex QP with the row."""
    best, best_w = np.inf, None
    for k in range(1, k_max + 1):
        for idx in combinations(range(problem.n), k):
            sel = list(idx)
            w = cp.Variable(k)
            sigma = cp.psd_wrap(problem.sigma[np.ix_(sel, sel)])
            expr = -problem.mu[sel] @ w + 0.5 * problem.risk_aversion * cp.quad_form(w, sigma)
            prob = cp.Problem(
                cp.Minimize(expr),
                [cp.sum(w) == problem.budget, w >= 0, w <= problem.w_max, a[sel] @ w >= b],
            )
            # tight tolerances: the default 1e-8 leaves weights wrong by ~1e-4 where the objective is flat
            prob.solve(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                continue
            full = np.zeros(problem.n)
            full[sel] = np.clip(w.value, 0.0, None)
            value = problem.objective_value(full)
            if value < best:
                best, best_w = value, full
    return best, best_w


@pytest.fixture(scope="module")
def return_row_reference():
    cache: dict[str, tuple] = {}

    def get(case: str):
        if case not in cache:
            n, seed, K, share = _ROW_CASES[case]
            problem, mu, r_bar = _return_row_problem(n, seed, K, share)
            cache[case] = (problem, mu, r_bar, *_support_enumeration_with_row(problem, mu, r_bar, K))
        return cache[case]

    return get


def test_the_return_row_matters_in_the_reference_cases(return_row_reference):
    """Guard against a vacuous test: without the row the optimum earns less than r̄ (the row cuts it off) and
    the row costs something; in the ``active`` case it holds with equality at the constrained optimum."""
    for case in _ROW_CASES:
        problem, mu, r_bar, f_ref, _ = return_row_reference(case)
        free = solve_miqp_native(problem, problem.cardinality)
        assert float(mu @ free.weights) < r_bar - 1e-3 * abs(r_bar)
        assert f_ref > free.objective + 1e-6 * abs(free.objective)
    _, mu, r_bar, _, w_active = return_row_reference("active")
    assert float(mu @ w_active) == pytest.approx(r_bar, rel=1e-5)


@pytest.mark.parametrize("formulation", ["bigm", "perspective"])
@pytest.mark.parametrize("case", sorted(_ROW_CASES))
def test_the_return_row_gives_the_enumerated_optimum(return_row_reference, case, formulation):
    problem, mu, r_bar, f_ref, w_ref = return_row_reference(case)
    sol = solve_miqp_native(problem, problem.cardinality, formulation=formulation,
                            linear_ge=(mu, r_bar), mip_gap=1e-9, time_limit_s=60.0)
    meta = sol.solver_meta
    assert meta["status"] in CERTIFIED and sol.feasible and meta["linear_ok"] is True
    assert _rel_err(sol.objective, f_ref) < REL_TOL
    np.testing.assert_allclose(sol.weights, w_ref, atol=1e-4)
    assert float(mu @ sol.weights) >= r_bar - 1e-8
    assert meta["linear_ge_slack"] == pytest.approx(float(mu @ sol.weights) - r_bar, abs=1e-15)


def test_the_dual_bound_with_the_row_does_not_cut_off_the_enumerated_optimum(return_row_reference):
    problem, mu, r_bar, f_ref, _ = return_row_reference("active")
    meta = solve_miqp_native(problem, 4, linear_ge=(mu, r_bar), mip_gap=1e-4, abs_gap=1e-6).solver_meta
    assert meta["dual_bound"] <= f_ref + 1e-9 and meta["primal_bound"] >= f_ref - 1e-9


@pytest.mark.parametrize("factor", [1e-3, 1e3])
def test_the_row_is_normalised_by_its_largest_coefficient(return_row_reference, factor):
    # the row is divided by max|a| as the risk row is divided by mean(diag Σ): feastol acts relative to its scale
    problem, mu, r_bar, *_ = return_row_reference("active")
    base = solve_miqp_native(problem, 4, linear_ge=(mu, r_bar), mip_gap=1e-9)
    scaled = solve_miqp_native(problem, 4, linear_ge=(factor * mu, factor * r_bar), mip_gap=1e-9)
    assert base.solver_meta["linear_ge_scale"] == float(np.max(np.abs(mu)))
    assert scaled.solver_meta["linear_ge_scale"] == pytest.approx(factor * float(np.max(np.abs(mu))), rel=1e-12)
    assert scaled.solver_meta["linear_ge_rhs"] == pytest.approx(factor * r_bar, rel=1e-12)
    assert scaled.objective == pytest.approx(base.objective, rel=1e-9)
    np.testing.assert_allclose(scaled.weights, base.weights, atol=1e-8)


def test_without_the_row_the_model_has_none_and_the_meta_has_no_trace_of_it(monkeypatch):
    names: list[str] = []

    class _Names(Model):
        def addCons(self, cons, name="", **kwargs):
            names.append(name)
            return super().addCons(cons, name=name, **kwargs)

    monkeypatch.setattr(scip_native, "Model", _Names)
    plain = solve_miqp_native(_make_problem(8, 902, K=3), 3)
    assert "linear_ge" not in names
    assert not [key for key in plain.solver_meta if key.startswith("linear")]
    names.clear()
    problem, mu, r_bar = _return_row_problem(9, 1301, 3)
    with_row = solve_miqp_native(problem, 3, linear_ge=(mu, r_bar))
    assert names.count("linear_ge") == 1
    assert with_row.solver_meta["linear_ge_rhs"] == r_bar
    json.dumps(with_row.solver_meta)  # plain Python numbers only


def test_the_default_model_is_the_same_with_and_without_the_argument(monkeypatch):
    """``linear_ge=None`` is the default: the same constraints in the same order, the same result."""
    seen: list[list[str]] = []

    class _Names(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append([])

        def addCons(self, cons, name="", **kwargs):
            seen[-1].append(name)
            return super().addCons(cons, name=name, **kwargs)

    monkeypatch.setattr(scip_native, "Model", _Names)
    problem = _make_problem(8, 902, K=3)
    default = solve_miqp_native(problem, 3, seed=2, permutation_seed=2)
    explicit = solve_miqp_native(problem, 3, seed=2, permutation_seed=2, linear_ge=None)
    assert seen[0] == seen[1]
    assert default.objective == explicit.objective
    assert default.solver_meta["n_nodes"] == explicit.solver_meta["n_nodes"]


@pytest.mark.parametrize(
    "bad",
    [
        (np.ones(5), 0.1),
        (np.ones(8), math.nan),
        (np.ones(8), math.inf),
        (np.full(8, math.nan), 0.1),
        (np.zeros(8), 0.1),
        np.ones(8),
        (np.ones(8), 0.1, 3),
        "return",
    ],
    ids=["shape", "nan-rhs", "inf-rhs", "nan-coefficient", "zero-row", "not-a-pair", "triple", "string"],
)
def test_an_invalid_linear_row_raises(bad):
    with pytest.raises(ValueError, match="linear_ge"):
        solve_miqp_native(_make_problem(8, 902, K=3), 3, linear_ge=bad)


def test_an_unreachable_return_makes_the_model_infeasible(return_row_reference):
    problem, mu, _, *_ = return_row_reference("switch")
    sol = solve_miqp_native(problem, 3, linear_ge=(mu, 1.5 * float(mu.max())))
    assert sol.solver_meta["status"] == "infeasible" and not sol.feasible
    assert math.isinf(sol.objective) and not sol.weights.any()
    assert sol.solver_meta["linear_ge_rhs"] == 1.5 * float(mu.max())


def test_a_warm_start_is_accepted_only_if_it_satisfies_the_row(return_row_reference):
    problem, mu, r_bar, *_ = return_row_reference("switch")
    rich = np.zeros(problem.n)
    rich[int(np.argmax(mu))] = 1.0  # all in the best asset: feasible and far from optimal
    poor = np.zeros(problem.n)
    poor[int(np.argmin(mu))] = 1.0  # earns the least: below r̄
    assert float(mu @ rich) > r_bar > float(mu @ poor)
    accepted = solve_miqp_native(problem, 3, linear_ge=(mu, r_bar), warm_start=rich)
    rejected = solve_miqp_native(problem, 3, linear_ge=(mu, r_bar), warm_start=poor)
    assert accepted.solver_meta["warm_start_used"] is True
    assert rejected.solver_meta["warm_start_used"] is False
    assert _rel_err(accepted.objective, rejected.objective) < REL_TOL


# Aggressive primal heuristics (the sensitivity of H2 to the exact baseline, ANALYSIS_PLAN §12, 2026-10-19): one meta
# setting of SCIP, an argument of its own and recorded, never a default.


def test_aggressive_primal_heuristics_are_a_setting_of_their_own_recorded_in_the_meta(spy_params):
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3, heuristics="aggressive").solver_meta
    assert ("<heuristics>", SCIP_PARAMSETTING.AGGRESSIVE) in spy_params
    assert meta["heuristics"] == "aggressive"


def test_by_default_the_heuristics_of_scip_are_untouched(spy_params):
    meta = solve_miqp_native(_make_problem(8, 902, K=3), 3).solver_meta
    assert not any(name == "<heuristics>" for name, _ in spy_params)
    assert meta["heuristics"] == "default"


def test_aggressive_heuristics_reach_the_same_certified_optimum():
    problem = _make_problem(10, 903, K=4)
    aggressive = solve_miqp_native(problem, 4, heuristics="aggressive")
    default = solve_miqp_native(problem, 4)
    assert {aggressive.solver_meta["status"], default.solver_meta["status"]} <= {"optimal", "gaplimit"}
    assert aggressive.objective == pytest.approx(default.objective, abs=1e-8)


def test_an_unknown_heuristics_setting_is_an_error():
    with pytest.raises(ValueError, match="heuristics"):
        solve_miqp_native(_make_problem(8, 902, K=3), 3, heuristics="fast")

