"""Tests for the multi-tree outer approximation with a HiGHS master (E2, RQ2).

Exact references independent of the OA code: support enumeration with CVXPY
(:func:`solve_continuous_mvo` on every sub-problem) and the native SCIP model. Instances are
synthetic (``factor_instance``, calibration seeds 1000-1999).
"""

from __future__ import annotations

import itertools
import json
from typing import NamedTuple

import cvxpy as cp
import numpy as np
import pytest

from cif.classical.continuous import solve_continuous_mvo
from cif.classical.oa import _evaluation_point, solve_oa, support_cut
from cif.classical.refine import refine_support
from cif.classical.scip_native import _diagonal_shift, solve_miqp_native
from cif.instances.diagnostics import binding_diagnostics, continuous_relaxation
from cif.instances.factor import factor_instance
from cif.metrics.gap import reference_from_exact, reference_scale
from cif.problem import PortfolioProblem


def _problem(
    n: int, seed: int, *, lam: float = 5.0, w_min: float = 0.0, w_max: float = 1.0
) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)),
        risk_aversion=lam, w_min=w_min, w_max=w_max,
    )


def _f_binary(problem: PortfolioProblem, support) -> float:
    """f(z) at a binary z by the shared refinement (itself tested against CVXPY)."""
    res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, support,
                         lower=problem.w_min, upper=problem.w_max)
    return res.objective


def _f_relaxed(problem: PortfolioProblem, d: np.ndarray, z: np.ndarray) -> float:
    """f(z) of the perspective relaxation by CVXPY: ``min −μᵀw + (λ/2)(wᵀRw + Σ d_i w_i²/z_i)``
    s.t. ``Σw = 1, l·z ≤ w ≤ u·z`` with ``R = Σ − diag(d)``; ``w_i = 0`` where ``z_i = 0``."""
    keep = np.flatnonzero(z > 0.0)
    zk = z[keep]
    quad = problem.sigma[np.ix_(keep, keep)] + np.diag(d[keep] * (1.0 / zk - 1.0))
    w = cp.Variable(keep.size)
    objective = (-problem.mu[keep] @ w
                 + 0.5 * problem.risk_aversion * cp.quad_form(w, cp.psd_wrap(quad)))
    cons = [cp.sum(w) == problem.budget, w >= problem.w_min * zk, w <= problem.w_max * zk]
    prob = cp.Problem(cp.Minimize(objective), cons)
    prob.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    return float(prob.value) if prob.status == "optimal" else np.inf


# ---------------------------------------------------------------------------
# 1. The cut θ ≥ f(ẑ) + gᵀ(z − ẑ)
# ---------------------------------------------------------------------------

_CUT_CASES = {
    "plain": {"seed": 1000, "lam": 5.0, "w_min": 0.0, "w_max": 1.0, "diag": "mineig"},
    "cap": {"seed": 1001, "lam": 5.0, "w_min": 0.0, "w_max": 0.4, "diag": "mineig"},
    "buyin_cap_sdp": {"seed": 1002, "lam": 20.0, "w_min": 0.05, "w_max": 0.5, "diag": "sdp"},
    "cap_sdp": {"seed": 1003, "lam": 5.0, "w_min": 0.0, "w_max": 0.35, "diag": "sdp"},
}
_N_CUT = 9


def _cut_case(case: str):
    spec = dict(_CUT_CASES[case])
    diag = spec.pop("diag")
    problem = _problem(_N_CUT, **spec)
    d, used, _ = _diagonal_shift(problem.sigma, diag)
    assert used == diag
    return problem, d


def _cut_at(problem: PortfolioProblem, d: np.ndarray, z_hat: np.ndarray):
    return support_cut(problem.mu, problem.sigma, problem.risk_aversion, d, z_hat,
                       problem.w_min, problem.w_max, problem.budget)


def _feasible_sizes(problem: PortfolioProblem) -> range:
    k_lo = int(np.ceil(problem.budget / problem.w_max - 1e-12))
    k_hi = problem.n if problem.w_min == 0.0 else int(problem.budget / problem.w_min + 1e-12)
    return range(k_lo, min(problem.n, k_hi) + 1)


@pytest.mark.parametrize("case", list(_CUT_CASES))
def test_cut_is_tight_at_its_point_and_valid_at_every_binary_point(case):
    """f(z) ≥ f(ẑ) + gᵀ(z − ẑ) for all 2⁹ binary z; the cut touches f at ẑ."""
    problem, d = _cut_case(case)
    n, sizes = problem.n, _feasible_sizes(problem)
    supports = [np.array(c) for k in sizes for c in itertools.combinations(range(n), k)]
    f_all = np.array([_f_binary(problem, s) for s in supports])
    z_all = np.zeros((len(supports), n))
    for row, s in zip(z_all, supports, strict=True):
        row[s] = 1.0
    rng = np.random.default_rng(7)
    for j in rng.choice(len(supports), size=4, replace=False):
        cut = _cut_at(problem, d, z_all[j])
        assert cut.value == pytest.approx(f_all[j], rel=1e-8, abs=1e-10)
        assert cut(z_all[j]) == pytest.approx(f_all[j], rel=1e-8, abs=1e-10)
        slack = f_all - np.array([cut(z) for z in z_all])
        assert slack.min() >= -1e-9, f"{case}: cut at {supports[j]} violated by {-slack.min():.2e}"
        assert np.median(slack) > 1e-6  # not vacuous: the cut actually bites away from ẑ


@pytest.mark.parametrize("case", list(_CUT_CASES))
def test_cut_is_valid_on_the_whole_box_not_only_on_binary_points(case):
    """The inequality holds for fractional z (f is then the perspective relaxation, CVXPY),
    for cuts taken at binary points (zeros in ẑ) and at interior fractional points."""
    problem, d = _cut_case(case)
    n, rng = problem.n, np.random.default_rng(11)
    k_min = problem.budget / problem.w_max

    def random_point(zero_some: bool) -> np.ndarray:
        while True:
            z = rng.uniform(0.15, 1.0, size=n)
            if zero_some:
                z[rng.choice(n, size=3, replace=False)] = 0.0
            if problem.w_min * z.sum() <= problem.budget and z.sum() >= k_min * 1.05:
                return z

    cuts = [_cut_at(problem, d, np.isin(np.arange(n), rng.choice(n, size=5, replace=False)).astype(float)),
            _cut_at(problem, d, random_point(zero_some=False)),
            _cut_at(problem, d, random_point(zero_some=True))]
    for cut in cuts:
        assert cut.value == pytest.approx(_f_relaxed(problem, d, cut.z_hat), rel=1e-7, abs=1e-9)
        for _ in range(8):
            z = random_point(zero_some=bool(rng.integers(2)))
            assert _f_relaxed(problem, d, z) >= cut(z) - 1e-7


def test_slope_outside_the_support_is_the_exact_one_sided_derivative():
    """Adding asset i at the margin: (f(ẑ + t·e_i) − f(ẑ))/t → g_i as t ↓ 0 (the strongest valid
    coefficient: the cut is not only valid but tight along every coordinate leaving the support)."""
    problem, d = _cut_case("buyin_cap_sdp")
    n = problem.n
    support = np.array([0, 2, 3, 6])
    z_hat = np.zeros(n)
    z_hat[support] = 1.0
    cut = _cut_at(problem, d, z_hat)
    t = 1e-4
    for i in np.setdiff1d(np.arange(n), support):
        z = z_hat.copy()
        z[i] = t
        forward = (_cut_at(problem, d, z).value - cut.value) / t
        # IPM noise ~1e-10 in f is divided by t: the absolute floor is 1e-6
        assert forward == pytest.approx(cut.gradient[i], abs=5e-6 + 3e-3 * abs(cut.gradient[i]))


def test_gradient_is_the_numerical_gradient_at_an_interior_point():
    """At z ∈ (0, 1)^N the function is differentiable and the cut slope is its gradient."""
    problem, d = _cut_case("cap")
    n = problem.n
    z_hat = np.random.default_rng(3).uniform(0.4, 0.9, size=n)
    cut = _cut_at(problem, d, z_hat)
    h = 1e-4  # IPM noise ~1e-10 in f is divided by 2h
    numeric = np.empty(n)
    for i in range(n):
        up, down = z_hat.copy(), z_hat.copy()
        up[i] += h
        down[i] -= h
        numeric[i] = (_cut_at(problem, d, up).value - _cut_at(problem, d, down).value) / (2 * h)
    np.testing.assert_allclose(cut.gradient, numeric, rtol=1e-3, atol=3e-6)


def test_cut_weights_are_the_minimiser_at_the_point():
    problem, d = _cut_case("cap")
    support = np.array([1, 4, 5, 8])
    z_hat = np.zeros(problem.n)
    z_hat[support] = 1.0
    cut = _cut_at(problem, d, z_hat)
    res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, support,
                         lower=problem.w_min, upper=problem.w_max)
    np.testing.assert_allclose(cut.weights, res.weights, atol=1e-9)
    assert np.all(cut.weights[np.setdiff1d(np.arange(problem.n), support)] == 0.0)


@pytest.mark.parametrize("diag", ["mineig", "sdp"])
def test_cut_is_valid_at_random_binary_points_of_a_larger_instance(diag):
    """N = 40, cuts at random supports, f(z) ≥ cut(z) at 300 random binary z of sizes 4-14 (the
    inequality is exact up to the interior-point accuracy of f, 1e-9)."""
    n = 40
    problem = _problem(n, 1004, w_max=0.4)
    d, used, _ = _diagonal_shift(problem.sigma, diag)
    assert used == diag
    rng = np.random.default_rng(5)

    def random_support() -> np.ndarray:
        return np.sort(rng.choice(n, size=int(rng.integers(4, 15)), replace=False))

    def as_point(support: np.ndarray) -> np.ndarray:
        z = np.zeros(n)
        z[support] = 1.0
        return z

    points = [random_support() for _ in range(300)]
    values = [_f_binary(problem, s) for s in points]
    for _ in range(5):
        cut_support = random_support()
        cut = _cut_at(problem, d, as_point(cut_support))
        assert cut.value == pytest.approx(_f_binary(problem, cut_support), rel=1e-8)
        slack = np.array([f - cut(as_point(s)) for s, f in zip(points, values, strict=True)])
        assert slack.min() >= -1e-9, f"violated by {-slack.min():.2e}"


def test_cut_rejects_a_point_without_a_feasible_subproblem():
    problem, d = _cut_case("cap")  # w_max = 0.4: at least 3 assets are needed
    z_hat = np.zeros(problem.n)
    z_hat[[0, 1]] = 1.0
    with pytest.raises(RuntimeError, match="infeasible"):
        _cut_at(problem, d, z_hat)
    with pytest.raises(ValueError):
        _cut_at(problem, d, np.ones(problem.n - 1))
    with pytest.raises(ValueError):
        _cut_at(problem, d, np.full(problem.n, 1.5))


# ---------------------------------------------------------------------------
# 2. The loop: exact on small instances, both cardinality modes
# ---------------------------------------------------------------------------


def _enumerate_optimum(problem: PortfolioProblem, K: int, mode: str) -> tuple[float, tuple]:
    """Best ``f`` over all supports of size ``K`` (``eq``) or ``≤ K`` (``le``). A held asset carries
    at least ``w_min`` (the minimum purchase), so ``le`` runs through every size."""
    best, arg = np.inf, ()
    for k in [K] if mode == "eq" else range(1, K + 1):
        for support in itertools.combinations(range(problem.n), k):
            value = _f_binary(problem, np.array(support))
            if value < best:
                best, arg = value, support
    return best, arg


_EXACT_CASES = {
    "le_plain": {"n": 10, "seed": 1010, "lam": 5.0, "w_min": 0.0, "w_max": 1.0, "K": 3, "mode": "le"},
    "le_cap": {"n": 10, "seed": 1011, "lam": 5.0, "w_min": 0.0, "w_max": 0.4, "K": 4, "mode": "le"},
    "le_buyin": {"n": 9, "seed": 1012, "lam": 20.0, "w_min": 0.05, "w_max": 0.5, "K": 4, "mode": "le"},
    "eq_buyin": {"n": 10, "seed": 1013, "lam": 5.0, "w_min": 0.03, "w_max": 1.0, "K": 4, "mode": "eq"},
    "eq_cap_buyin": {"n": 10, "seed": 1014, "lam": 10.0, "w_min": 0.05, "w_max": 0.4, "K": 4,
                     "mode": "eq"},
}


class _Reference(NamedTuple):
    problem: PortfolioProblem
    K: int
    mode: str
    f_star: float  # optimum by support enumeration
    support: tuple[int, ...]


@pytest.fixture(scope="module")
def exact_reference():
    cache: dict[str, _Reference] = {}

    def get(case: str) -> _Reference:
        if case not in cache:
            spec = dict(_EXACT_CASES[case])
            K, mode = spec.pop("K"), spec.pop("mode")
            problem = _problem(spec.pop("n"), **spec)
            f_star, support = _enumerate_optimum(problem, K, mode)
            cache[case] = _Reference(problem, K, mode, f_star, support)
        return cache[case]

    return get


@pytest.mark.parametrize("case", list(_EXACT_CASES))
def test_exact_cases_are_binding(exact_reference, case):
    """Guard against a vacuous test: the cardinality limit cuts off the continuous optimum."""
    ref = exact_reference(case)
    problem, K, f_star = ref.problem, ref.K, ref.f_star
    diagnostics = binding_diagnostics(problem, K)
    assert diagnostics["binding"]
    assert f_star > diagnostics["f_cont"] + 1e-4 * abs(diagnostics["f_cont"])


@pytest.mark.parametrize("root_cuts", [False, True])
@pytest.mark.parametrize("diag", ["mineig", "sdp"])
@pytest.mark.parametrize("case", list(_EXACT_CASES))
def test_matches_support_enumeration(exact_reference, case, diag, root_cuts):
    problem, K, mode, f_star, support = exact_reference(case)
    sol = solve_oa(problem, K, card_mode=mode, diag=diag, root_cuts=root_cuts)
    meta = sol.solver_meta
    assert sol.feasible
    assert meta["status"] == "optimal"
    assert sol.objective == pytest.approx(f_star, rel=1e-6)
    assert meta["n_nonzero"] <= K
    if mode == "eq":
        assert meta["n_nonzero"] == K  # every held asset carries at least w_min > 0
    assert meta["diag_method"] == diag
    assert meta["dual_bound"] <= sol.objective + 1e-8 * abs(sol.objective)
    assert meta["primal_bound"] == pytest.approx(sol.objective, rel=1e-9)
    # the independent CVXPY solve of the winning support gives the same value
    idx = list(support)
    sub = PortfolioProblem(
        mu=problem.mu[idx], sigma=problem.sigma[np.ix_(idx, idx)],
        asset_names=tuple(f"s{i}" for i in range(len(idx))), w_min=problem.w_min,
        w_max=problem.w_max, risk_aversion=problem.risk_aversion)
    assert solve_continuous_mvo(sub).objective == pytest.approx(f_star, rel=1e-6)


_SCIP_CASES = [(14, 1020, 4, 1.0), (13, 1021, 5, 0.35), (12, 1022, 3, 0.5)]  # (N, seed, K, w_max)


@pytest.mark.parametrize("root_cuts", [False, True])
@pytest.mark.parametrize(("n", "seed", "K", "w_max"), _SCIP_CASES)
def test_matches_native_scip_model(n, seed, K, w_max, root_cuts):
    problem = _problem(n, seed, w_max=w_max)
    assert binding_diagnostics(problem, K)["binding"]
    scip = solve_miqp_native(problem, K, mip_gap=1e-9)
    oa = solve_oa(problem, K, root_cuts=root_cuts)
    assert scip.solver_meta["status"] in ("optimal", "gaplimit")
    assert oa.solver_meta["status"] == "optimal"
    assert oa.objective == pytest.approx(scip.objective, rel=1e-6)


@pytest.mark.parametrize(("n", "seed", "K", "w_max"), _SCIP_CASES)
def test_matches_native_scip_model_with_exact_cardinality(n, seed, K, w_max):
    problem = _problem(n, seed, w_min=0.02, w_max=w_max)
    scip = solve_miqp_native(problem, K, card_mode="eq", mip_gap=1e-9)
    oa = solve_oa(problem, K, card_mode="eq")
    assert oa.solver_meta["status"] == "optimal"
    assert oa.objective == pytest.approx(scip.objective, rel=1e-6)


@pytest.mark.parametrize("root_cuts", [False, True])
@pytest.mark.parametrize(
    ("n", "seed", "K", "w_min", "w_max"),
    [(14, 1070, 4, 0.02, 1.0), (13, 1071, 5, 0.03, 0.35)],  # exact K; the second with u < 1
)
def test_exact_cardinality_matches_enumeration_up_to_n14(n, seed, K, w_min, w_max, root_cuts):
    """Σz = K at N = 13-14 against enumeration of all C(N, K) supports (the SCIP model above has
    no exact-K mode yet): every held asset carries at least w_min."""
    problem = _problem(n, seed, w_min=w_min, w_max=w_max)
    f_star, support = _enumerate_optimum(problem, K, "eq")
    sol = solve_oa(problem, K, card_mode="eq", root_cuts=root_cuts)
    meta = sol.solver_meta
    assert f_star > binding_diagnostics(problem, K)["f_cont"] + 1e-6  # the constraint costs something
    assert meta["status"] == "optimal" and sol.feasible
    assert sol.objective == pytest.approx(f_star, rel=1e-6)
    assert meta["n_nonzero"] == K
    assert np.all(sol.weights[np.flatnonzero(sol.weights > 1e-6)] >= w_min - 1e-9)
    assert set(np.flatnonzero(sol.weights > 1e-6).tolist()) == set(support) or (
        sol.objective == pytest.approx(f_star, rel=1e-9))  # same support, or an exact tie


def test_cardinality_above_n_is_not_binding():
    problem = _problem(6, 1040)
    # CVXPY on the normalised problem: its default tolerances are absolute (see diagnostics.py)
    reference = continuous_relaxation(problem).objective
    for sol in (solve_oa(problem, 9), solve_oa(problem)):  # K > N and no K at all
        assert sol.solver_meta["status"] == "optimal"
        assert sol.objective == pytest.approx(reference, rel=1e-6)


def test_singular_covariance_has_no_perspective_gain_but_stays_exact():
    """Rank-deficient Σ: d = 0, the cuts reduce to the linking-row cuts and stay valid."""
    n, K = 8, 3
    rng = np.random.default_rng(1080)
    factor = rng.standard_normal((n, n - 2)) * 0.1
    problem = PortfolioProblem(
        mu=rng.uniform(0.05, 0.25, size=n), sigma=factor @ factor.T,
        asset_names=tuple(f"s{i}" for i in range(n)), risk_aversion=2.0)
    f_star, _ = _enumerate_optimum(problem, K, "le")
    for root_cuts in (False, True):  # diag="sdp" is not tried: Clarabel panics on this Σ, see the report
        sol = solve_oa(problem, K, diag="mineig", root_cuts=root_cuts)
        assert sol.solver_meta["d_max"] == 0.0
        assert sol.solver_meta["status"] == "optimal"
        assert sol.objective == pytest.approx(f_star, rel=1e-6)


def test_problem_cardinality_is_the_default():
    problem = _problem(8, 1041)
    explicit = solve_oa(problem, 3)
    problem = PortfolioProblem(mu=problem.mu, sigma=problem.sigma, asset_names=problem.asset_names,
                               risk_aversion=problem.risk_aversion, cardinality=3)
    implicit = solve_oa(problem)
    assert implicit.solver_meta["cardinality"] == 3
    assert implicit.objective == pytest.approx(explicit.objective, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Bounds, trajectories, termination
# ---------------------------------------------------------------------------


def _bad_start(problem: PortfolioProblem, K: int) -> list[int]:
    """The K assets with the lowest mean return: a poor incumbent, so that several iterations run."""
    return sorted(np.argsort(problem.mu)[:K].tolist())


@pytest.mark.parametrize("root_cuts", [False, True])
def test_bounds_are_valid_at_every_iteration(exact_reference, root_cuts):
    ref = exact_reference("le_cap")
    problem, K, f_star = ref.problem, ref.K, ref.f_star
    sol = solve_oa(problem, K, warm_start=_bad_start(problem, K), root_cuts=root_cuts)
    meta = sol.solver_meta
    ub = [f for _, f in meta["incumbent_trajectory"]]
    lb = [b for _, b in meta["bound_trajectory"]]
    tol = 1e-8 * abs(f_star)
    assert len(lb) == meta["iterations"] + 1  # the start (after the root phase) and each iteration
    if root_cuts:
        assert len(ub) >= meta["iterations"] + 1  # + incumbents found by the root LPs
    else:
        assert meta["iterations"] >= 2  # not vacuous
        assert len(ub) == meta["iterations"] + 1
    assert all(b <= f_star + tol for b in lb), "a lower bound exceeds the optimum"
    assert all(f >= f_star - tol for f in ub), "an incumbent beats the optimum"
    assert all(a <= b for a, b in itertools.pairwise(lb))  # LB never decreases
    assert all(a >= b for a, b in itertools.pairwise(ub))  # UB never increases
    assert lb[-1] <= ub[-1]
    assert meta["status"] == "optimal"
    assert meta["gap"] <= 1e-6
    assert meta["dual_bound"] == pytest.approx(lb[-1])
    assert meta["primal_bound"] == pytest.approx(ub[-1], rel=1e-9)


def test_trajectories_are_time_ordered_and_end_at_the_result(exact_reference):
    problem, K = exact_reference("le_plain")[:2]
    start = _bad_start(problem, K)
    sol = solve_oa(problem, K, warm_start=start)
    meta = sol.solver_meta
    traj = meta["incumbent_trajectory"]
    times = [t for t, _ in traj]
    assert times == sorted(times) and times[0] >= 0.0
    bound_times = [t for t, _ in meta["bound_trajectory"]]
    assert bound_times == sorted(bound_times)
    assert traj[0][1] == pytest.approx(_f_binary(problem, np.array(start)))
    assert traj[-1][1] == pytest.approx(sol.objective, rel=1e-9)
    assert times[-1] <= sol.wall_time_s
    improved = [t for (t, f), (_, g) in zip(traj[1:], traj, strict=False) if f < g]
    assert meta["t_primal_s"] == pytest.approx(improved[-1] if improved else times[0])
    assert meta["t_cert_s"] is not None and meta["t_cert_s"] >= meta["t_primal_s"]


def test_zero_tolerances_terminate_when_the_master_repeats_a_point(exact_reference):
    ref = exact_reference("le_plain")
    problem, K, f_star = ref.problem, ref.K, ref.f_star
    sol = solve_oa(problem, K, mip_gap=0.0, abs_gap=0.0)
    assert sol.solver_meta["status"] == "optimal"
    assert sol.objective == pytest.approx(f_star, rel=1e-9)


def test_absolute_gap_stops_the_loop_early(exact_reference):
    ref = exact_reference("le_cap")
    problem, K, f_star = ref.problem, ref.K, ref.f_star
    start = _bad_start(problem, K)
    loose = solve_oa(problem, K, warm_start=start, mip_gap=0.0, abs_gap=10.0 * abs(f_star))
    tight = solve_oa(problem, K, warm_start=start)
    assert loose.solver_meta["iterations"] < tight.solver_meta["iterations"]
    assert loose.solver_meta["status"] == "optimal"  # the requested (loose) gap is met
    assert loose.solver_meta["primal_bound"] - loose.solver_meta["dual_bound"] <= 10.0 * abs(f_star)


def test_time_limit_is_reported():
    K = 10
    problem = _problem(150, 1050)
    sol = solve_oa(problem, K, time_limit_s=0.05)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit"
    assert meta["time_limit_hit"] is True
    assert meta["dual_bound"] <= meta["primal_bound"]
    assert meta["solve_time_s"] < 5.0  # the limit is honoured by the master solves
    if np.isfinite(sol.objective):
        assert sol.feasible and meta["n_nonzero"] <= K
    else:
        assert sol.feasible is False and meta["primal_bound"] == np.inf and meta["gap"] == np.inf


def test_zero_time_limit_returns_the_start():
    problem = _problem(40, 1051)
    K, start = 6, [0, 5, 9, 13, 20, 33]
    sol = solve_oa(problem, K, time_limit_s=0.0, warm_start=start)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit" and meta["iterations"] == 0
    assert meta["warm_start_used"] is True
    assert sol.feasible
    assert sol.objective == pytest.approx(_f_binary(problem, np.array(start)))


def test_optimal_return_is_a_certificate_for_the_gap_module(exact_reference):
    ref = exact_reference("le_plain")
    problem, K, f_star = ref.problem, ref.K, ref.f_star
    sol = solve_oa(problem, K)
    s_ref = reference_scale(problem, sol.weights)
    f_ref, ref_type = reference_from_exact(sol.solver_meta, scale=s_ref)
    assert ref_type == "certified"
    assert f_ref == pytest.approx(f_star, rel=1e-6)
    limited = solve_oa(_problem(150, 1050), 10, time_limit_s=0.05)
    # a time limit never certifies, whatever the scale (the run may have no incumbent to take it from)
    assert reference_from_exact(limited.solver_meta, scale=1.0)[1] == "best_known"


# ---------------------------------------------------------------------------
# 4. Warm start
# ---------------------------------------------------------------------------


def test_warm_start_is_the_first_incumbent_and_is_never_worsened(exact_reference):
    problem, K = exact_reference("le_cap")[:2]
    start = _bad_start(problem, K)
    sol = solve_oa(problem, K, warm_start=start)
    meta = sol.solver_meta
    assert meta["warm_start_used"] is True
    assert meta["incumbent_trajectory"][0][1] == pytest.approx(_f_binary(problem, np.array(start)))
    assert sol.objective <= meta["incumbent_trajectory"][0][1] + 1e-12


def test_optimal_warm_start_needs_no_improvement(exact_reference):
    ref = exact_reference("le_cap")
    problem, K, f_star, support = ref.problem, ref.K, ref.f_star, ref.support
    sol = solve_oa(problem, K, warm_start=list(support))
    meta = sol.solver_meta
    assert meta["status"] == "optimal" and meta["warm_start_used"] is True
    assert sol.objective == pytest.approx(f_star, rel=1e-9)
    assert meta["incumbent_trajectory"][0][1] == pytest.approx(f_star, rel=1e-9)
    assert meta["t_primal_s"] == pytest.approx(meta["incumbent_trajectory"][0][0])


def test_infeasible_warm_start_is_ignored(exact_reference):
    ref = exact_reference("le_cap")
    problem, K, f_star = ref.problem, ref.K, ref.f_star
    too_large = list(range(K + 1))  # violates Σz ≤ K
    too_small = [0, 1]  # u = 0.4: two assets cannot carry the budget
    for start in (too_large, too_small):
        sol = solve_oa(problem, K, warm_start=start)
        assert sol.solver_meta["warm_start_used"] is False
        assert sol.solver_meta["status"] == "optimal"
        assert sol.objective == pytest.approx(f_star, rel=1e-6)


def test_warm_start_in_eq_mode_must_have_exactly_k_assets(exact_reference):
    ref = exact_reference("eq_buyin")
    problem, K, f_star, support = ref.problem, ref.K, ref.f_star, ref.support
    exact = solve_oa(problem, K, card_mode="eq", warm_start=list(support))
    assert exact.solver_meta["warm_start_used"] is True
    smaller = solve_oa(problem, K, card_mode="eq", warm_start=list(support)[:-1])
    assert smaller.solver_meta["warm_start_used"] is False
    assert smaller.objective == pytest.approx(f_star, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Reporting, scale, edge cases
# ---------------------------------------------------------------------------


def test_meta_reports_the_run(exact_reference):
    problem, K = exact_reference("le_buyin")[:2]
    sol = solve_oa(problem, K, seed=3, diag="sdp")
    meta = sol.solver_meta
    assert sol.solver == "oa_highs"
    assert meta["threads"] == 1 and meta["seed"] == 3 and meta["cardinality"] == K
    assert meta["card_mode"] == "le" and meta["mip_gap"] == 1e-6 and meta["abs_gap"] == 0.0
    assert meta["time_limit_s"] is None and meta["time_limit_hit"] is False
    for key in ("build_time_s", "solve_time_s", "build_cpu_s", "solve_cpu_s", "master_time_s",
                "refine_time_s"):
        assert meta[key] >= 0.0
    assert meta["build_time_s"] + meta["solve_time_s"] <= sol.wall_time_s + 1e-3
    assert meta["master_time_s"] + meta["refine_time_s"] <= meta["solve_time_s"] + 1e-3
    assert meta["n_cuts"] == meta["n_refine"] >= max(meta["iterations"] - 1, 0)  # a repeat adds none
    assert 0.0 <= meta["d_min"] <= meta["d_max"]
    assert meta["budget_ok"] and meta["box_ok"] and meta["card_ok"]
    assert meta["f_cont"] <= meta["dual_bound"] + 1e-12


@pytest.mark.parametrize("diag", ["mineig", "sdp"])
def test_diagonal_split_is_the_one_of_scip_native(diag):
    """Same ``d`` as the SCIP baseline computes from the same Σ."""
    problem = _problem(30, 1000)
    d_scip, used_scip, fallback_scip = _diagonal_shift(0.5 * (problem.sigma + problem.sigma.T), diag)
    assert used_scip == diag and fallback_scip is None  # the case is not vacuous
    meta = solve_oa(problem, 4, diag=diag, time_limit_s=0.0).solver_meta
    assert meta["diag_method"] == diag and meta["diag_fallback"] is None
    assert meta["d_min"] == pytest.approx(float(d_scip.min()), rel=1e-12)
    assert meta["d_max"] == pytest.approx(float(d_scip.max()), rel=1e-12)


def test_a_panic_of_the_sdp_solver_is_a_recorded_fallback_to_mineig(monkeypatch):
    """A Rust panic (``pyo3_runtime.PanicException``) is a ``BaseException``, not an ``Exception``."""

    class Panic(BaseException):
        pass

    def failing_solve(self, *args, **kwargs):
        raise Panic("boom")

    problem = _problem(12, 1070)
    reference = solve_oa(problem, 3, diag="mineig")
    monkeypatch.setattr(cp.Problem, "solve", failing_solve)
    sol = solve_oa(problem, 3, diag="sdp")
    meta = sol.solver_meta
    assert meta["diag_method"] == "mineig" and "Panic: boom" in meta["diag_fallback"]
    assert meta["status"] == "optimal"
    assert sol.objective == pytest.approx(reference.objective, rel=1e-9)


def test_scale_of_mu_and_sigma_changes_nothing_but_the_units():
    base = _problem(10, 1060)
    ref = solve_oa(base, 3)
    for scale in (1e-3, 40.0):  # ORlib-like and large units
        scaled = PortfolioProblem(mu=base.mu * scale, sigma=base.sigma * scale,
                                  asset_names=base.asset_names, risk_aversion=base.risk_aversion)
        sol = solve_oa(scaled, 3)
        assert sol.solver_meta["status"] == "optimal"
        np.testing.assert_allclose(sol.weights, ref.weights, atol=1e-7)
        assert sol.objective == pytest.approx(ref.objective * scale, rel=1e-8)
        assert sol.solver_meta["dual_bound"] == pytest.approx(
            ref.solver_meta["dual_bound"] * scale, rel=1e-4)


@pytest.mark.parametrize("scale", [1e-4, 1e4])
def test_sdp_split_and_optimum_are_the_same_in_any_units(scale):
    """The SDP of the split is accepted whatever the units of Σ (its margin is relative to
    λ_max(Σ)), so ``diag="sdp"`` is not silently ``"mineig"`` in daily or percent² units."""
    base = _problem(14, 1064)
    scaled = PortfolioProblem(mu=base.mu * scale, sigma=base.sigma * scale,
                              asset_names=base.asset_names, risk_aversion=base.risk_aversion)
    ref = solve_oa(base, 3, diag="sdp")
    sol = solve_oa(scaled, 3, diag="sdp")
    assert ref.solver_meta["diag_method"] == sol.solver_meta["diag_method"] == "sdp"
    assert sol.solver_meta["d_max"] == pytest.approx(scale * ref.solver_meta["d_max"], rel=1e-4)
    assert sol.solver_meta["status"] == "optimal"
    np.testing.assert_allclose(sol.weights, ref.weights, atol=1e-7)
    assert sol.objective == pytest.approx(ref.objective * scale, rel=1e-8)


def test_infeasible_instance_reports_no_solution():
    problem = _problem(10, 1061, w_max=0.2)  # 4 · 0.2 < budget
    sol = solve_oa(problem, 4)
    meta = sol.solver_meta
    assert meta["status"] == "infeasible"
    assert sol.feasible is False and sol.objective == np.inf
    assert np.all(sol.weights == 0.0)
    assert meta["primal_bound"] == np.inf and meta["dual_bound"] == np.inf


def test_exact_cardinality_with_too_large_minimum_purchase_is_infeasible():
    problem = _problem(10, 1062, w_min=0.3)  # 4 · 0.3 > budget
    sol = solve_oa(problem, 4, card_mode="eq")
    assert sol.solver_meta["status"] == "infeasible" and sol.feasible is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threads": 2},
        {"card_mode": "ge"},
        {"diag": "trace"},
        {"mip_gap": -1e-3},
        {"abs_gap": -1.0},
        {"time_limit_s": -1.0},
        {"budget_s": -1.0},
        {"budget_s": float("nan")},
        {"budget_s": float("inf")},
        {"clock_offset_s": -1.0},
        {"clock_offset_s": float("nan")},
        {"seed": -1},
        {"warm_start": [0, 99]},
        {"warm_start": [-1, 2]},
        {"warm_start": [0.0, 2.0]},  # weights are not indices
        {"warm_start": np.arange(8) < 3},  # a mask is not a list of indices
    ],
)
def test_invalid_arguments_raise(kwargs):
    problem = _problem(8, 1063)
    with pytest.raises(ValueError):
        solve_oa(problem, 3, **kwargs)


def test_cardinality_below_one_raises():
    with pytest.raises(ValueError):
        solve_oa(_problem(8, 1063), 0)


# ---------------------------------------------------------------------------
# 6. Root phase: cuts at the fractional points of the master's LP relaxation
# ---------------------------------------------------------------------------


def _perspective_relaxation(problem: PortfolioProblem, K: int, mode: str) -> float:
    """Boolean relaxation of the perspective reformulation by CVXPY (second-order cones):
    ``min −μᵀw + (λ/2)(wᵀRw + dᵀp)`` s.t. ``w_i² ≤ p_i z_i, Σw = b, l·z ≤ w ≤ u·z, z ∈ [0, 1]^N``
    and the cardinality row. Its value is what the cuts at fractional points converge to."""
    n, u = problem.n, min(problem.w_max, problem.budget)
    d, _, _ = _diagonal_shift(problem.sigma, "mineig")
    w, z, p = cp.Variable(n), cp.Variable(n), cp.Variable(n)
    cons = [cp.sum(w) == problem.budget, w >= problem.w_min * z, w <= u * z, z >= 0, z <= 1,
            cp.sum(z) >= problem.budget / u]
    cons.append(cp.sum(z) == K if mode == "eq" else cp.sum(z) <= K)
    cons += [cp.quad_over_lin(w[i], z[i]) <= p[i] for i in range(n)]
    risk = cp.quad_form(w, cp.psd_wrap(problem.sigma - np.diag(d))) + d @ p
    prob = cp.Problem(cp.Minimize(-problem.mu @ w + 0.5 * problem.risk_aversion * risk), cons)
    prob.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    assert prob.status == "optimal"
    return float(prob.value)


@pytest.mark.parametrize("case", ["le_plain", "le_cap", "eq_buyin"])
def test_root_bound_converges_to_the_perspective_relaxation(exact_reference, case):
    """The cuts at fractional LP points reproduce the SOC relaxation bound, from below."""
    problem, K, mode, f_star, _ = exact_reference(case)
    sol = solve_oa(problem, K, card_mode=mode, root_cuts=True)
    meta = sol.solver_meta
    f_relaxed = _perspective_relaxation(problem, K, mode)
    root = meta["root_dual_bound"]
    assert meta["f_cont"] < f_relaxed - 1e-4 * abs(f_relaxed)  # the relaxation is tighter
    assert root <= f_relaxed + 1e-8 * abs(f_relaxed)  # valid
    assert root >= f_relaxed - 3e-6 * abs(f_relaxed)  # and converged to the root tolerance
    assert root <= f_star + 1e-8 * abs(f_star)
    assert meta["root_iterations"] >= 1 and meta["root_time_s"] > 0.0
    assert meta["status"] == "optimal" and sol.objective == pytest.approx(f_star, rel=1e-6)


def test_root_phase_needs_fewer_master_iterations_than_binary_cuts_alone():
    problem = _problem(20, 1000)
    K = 4
    start = [int(i) for i in np.argsort(-problem.mu)[:K]]
    plain = solve_oa(problem, K, warm_start=start, root_cuts=False)
    rooted = solve_oa(problem, K, warm_start=start, root_cuts=True)
    assert rooted.objective == pytest.approx(plain.objective, rel=1e-9)
    assert rooted.solver_meta["iterations"] < plain.solver_meta["iterations"]
    assert rooted.solver_meta["n_cuts"] > plain.solver_meta["n_cuts"]


def test_root_phase_can_be_switched_off():
    problem = _problem(10, 1010)
    meta = solve_oa(problem, 3, root_cuts=False).solver_meta
    assert meta["root_cuts"] is False
    assert meta["root_iterations"] == 0 and meta["root_dual_bound"] is None
    assert solve_oa(problem, 3).solver_meta["root_cuts"] is True  # on by default


def test_root_phase_honours_the_time_limit():
    sol = solve_oa(_problem(150, 1050), 10, time_limit_s=0.02)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit" and meta["time_limit_hit"] is True
    assert meta["solve_time_s"] < 2.0
    assert meta["dual_bound"] <= meta["primal_bound"]


@pytest.mark.parametrize("case", ["le_cap", "eq_buyin"])
def test_root_phase_incumbents_are_feasible_binary_points(exact_reference, case):
    """No warm start: whatever the root LPs hand over as an incumbent is a real portfolio."""
    problem, K, mode, f_star, _ = exact_reference(case)
    sol = solve_oa(problem, K, card_mode=mode, root_cuts=True)
    meta = sol.solver_meta
    values = [f for _, f in meta["incumbent_trajectory"]]
    assert values and all(f >= f_star - 1e-8 * abs(f_star) for f in values)
    assert sol.feasible and meta["budget_ok"] and meta["box_ok"] and meta["card_ok"]


def test_evaluation_point_drops_dust_and_keeps_the_subproblem_feasible():
    z = np.array([0.7, 1.0 - 1e-10, 0.0, 3e-7, 0.3 + 5e-7])
    cleaned = _evaluation_point(z, upper=1.0, budget=1.0)
    assert cleaned[2] == cleaned[3] == 0.0  # dust and zeros are exact zeros
    assert cleaned[1] == 1.0 and cleaned[0] == pytest.approx(0.7)
    # a near-binary point becomes the binary point
    near_binary = _evaluation_point(np.array([1e-9, 1 - 1e-9, 1.0, 0.0]), 1.0, 1.0)
    assert np.array_equal(near_binary, [0, 1, 1, 0])
    # u = 0.5 needs Σz ≥ 2: dropping the dust would leave 0.5·Σz < b, so the dust is lifted
    tight = _evaluation_point(np.array([1.0, 1.0 - 3e-7, 3e-7]), upper=0.5, budget=1.0)
    assert tight[2] > 0.0 and 0.5 * tight.sum() >= 1.0


# ---------------------------------------------------------------------------
# 6. End-to-end budget on the common clock (ANALYSIS_PLAN §5-§6)
# ---------------------------------------------------------------------------
#
# ``time_limit_s`` counts the loop only (certification runs rely on it). ``budget_s`` is the RQ2
# budget: seconds of the common clock, the diagonal / SDP, the relaxation, the master and the warm
# start all inside it, ``clock_offset_s`` of it already spent on the construction of the common start.


def test_the_budget_leaves_the_loop_what_the_build_did_not_use():
    meta = solve_oa(_problem(12, 1070), 3, budget_s=100.0, clock_offset_s=3.0).solver_meta
    assert meta["solve_limit_s"] == pytest.approx(100.0 - 3.0 - meta["build_time_s"], abs=1e-9)
    assert meta["budget_s"] == 100.0 and meta["clock_offset_s"] == 3.0
    assert meta["time_limit_s"] is None and meta["time_limit_hit"] is False


def test_time_limit_still_counts_the_loop_only():
    # the meaning certification relies on: the build is not in the limit
    meta = solve_oa(_problem(12, 1070), 3, time_limit_s=60.0).solver_meta
    assert meta["solve_limit_s"] == 60.0 and meta["budget_s"] is None
    assert meta["clock_offset_s"] == 0.0
    assert solve_oa(_problem(12, 1070), 3).solver_meta["solve_limit_s"] is None


@pytest.mark.parametrize(
    ("time_limit_s", "budget_s", "expected"),
    [(5.0, 100.0, 5.0), (500.0, 100.0, None), (0.0, 100.0, 0.0)],
)
def test_the_smaller_of_the_two_limits_binds(time_limit_s, budget_s, expected):
    meta = solve_oa(
        _problem(12, 1070), 3, time_limit_s=time_limit_s, budget_s=budget_s
    ).solver_meta
    if expected is None:  # the budget: what is left of it after the build
        assert meta["solve_limit_s"] == pytest.approx(budget_s - meta["build_time_s"], abs=1e-9)
    else:
        assert meta["solve_limit_s"] == expected


@pytest.mark.parametrize("root_cuts", [False, True])
def test_a_budget_smaller_than_the_build_leaves_nothing_to_the_loop(root_cuts):
    """The build (the relaxation, the master, the warm start) takes far more than 100 µs, so with
    an end-to-end budget of that size the loop must not start; with ``time_limit_s`` it would."""
    problem = _problem(40, 1051)
    K, start = 6, [0, 5, 9, 13, 20, 33]
    sol = solve_oa(problem, K, warm_start=start, budget_s=1e-4, root_cuts=root_cuts)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit" and meta["time_limit_hit"] is True
    assert meta["solve_limit_s"] == 0.0
    assert meta["iterations"] == 0 and meta["root_iterations"] == 0
    assert meta["warm_start_used"] is True and sol.feasible
    assert sol.objective == pytest.approx(_f_binary(problem, np.array(start)))


def test_a_budget_spent_before_the_call_returns_the_start():
    problem = _problem(40, 1051)
    K, start = 6, [0, 5, 9, 13, 20, 33]
    sol = solve_oa(problem, K, warm_start=start, budget_s=1.0, clock_offset_s=2.0)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit" and meta["iterations"] == 0
    assert meta["warm_start_used"] is True and sol.feasible
    assert sol.wall_time_s >= 2.0  # the clock is the common one


def test_a_binding_budget_stops_the_loop_and_the_call_ends_near_it():
    K, budget = 10, 0.5
    sol = solve_oa(_problem(150, 1050), K, budget_s=budget)
    meta = sol.solver_meta
    assert meta["status"] == "timelimit" and meta["time_limit_hit"] is True
    assert meta["dual_bound"] <= meta["primal_bound"]
    # a master solve or a subproblem may overrun the limit: a coarse bound only
    assert sol.wall_time_s < budget + 3.0
    assert meta["build_time_s"] + meta["solve_time_s"] <= sol.wall_time_s + 1e-3


def test_the_offset_moves_trajectories_and_stamps_to_the_common_clock(exact_reference):
    problem, K = exact_reference("le_plain")[:2]
    start = _bad_start(problem, K)
    offset = 5.0
    plain = solve_oa(problem, K, warm_start=start)
    sol = solve_oa(problem, K, warm_start=start, clock_offset_s=offset)
    meta = sol.solver_meta
    times = [t for t, _ in meta["incumbent_trajectory"]] + [t for t, _ in meta["bound_trajectory"]]
    assert min(times) >= offset and max(times) <= sol.wall_time_s
    assert meta["t_primal_s"] >= offset and meta["t_cert_s"] >= meta["t_primal_s"]
    assert offset + meta["build_time_s"] + meta["solve_time_s"] <= sol.wall_time_s + 1e-3
    assert sol.wall_time_s < offset + 60.0
    # nothing but the clock changes
    assert sol.objective == plain.objective
    np.testing.assert_array_equal(sol.weights, plain.weights)
    assert [f for _, f in meta["incumbent_trajectory"]] == [
        f for _, f in plain.solver_meta["incumbent_trajectory"]
    ]
    assert sol.wall_time_s >= offset > plain.wall_time_s


def test_solver_meta_is_json_serialisable_as_it_is():
    # the registry writes rows with ``default=str``: a numpy bool or float would become a string
    meta = solve_oa(_problem(12, 1070), 3, budget_s=30.0, clock_offset_s=1.0).solver_meta
    back = json.loads(json.dumps(meta))
    assert back["budget_ok"] is True and back["box_ok"] is True and back["card_ok"] is True
    assert back["budget_s"] == 30.0 and back["clock_offset_s"] == 1.0


def test_build_and_diagonal_times_are_recorded():
    problem = _problem(12, 1070)
    sdp = solve_oa(problem, 3, diag="sdp").solver_meta
    mineig = solve_oa(problem, 3, diag="mineig").solver_meta
    assert 0.0 < sdp["diag_time_s"] <= sdp["build_time_s"]
    assert 0.0 <= mineig["diag_time_s"] <= mineig["build_time_s"]
    assert sdp["diag_time_s"] > mineig["diag_time_s"]


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
    meta = solve_oa(_problem(12, 1070), 3, diag="sdp", budget_s=50.0, clock_offset_s=10.0).solver_meta
    assert meta["diag_method"] == "sdp"
    assert spy.calls[0]["solver"] == "CLARABEL"
    assert 39.0 < spy.calls[0]["time_limit"] <= 40.0  # 50 - 10 minus the little spent so far


def test_the_sdp_is_not_limited_by_the_loop_only_time_limit_or_without_a_budget(monkeypatch):
    spy = _SolveSpy(monkeypatch)
    problem = _problem(12, 1070)
    solve_oa(problem, 3, diag="sdp", time_limit_s=5.0)
    solve_oa(problem, 3, diag="sdp")
    assert [("time_limit" in call) for call in spy.calls] == [False, False]


def test_no_sdp_is_started_when_the_budget_is_already_spent(monkeypatch):
    spy = _SolveSpy(monkeypatch)
    meta = solve_oa(
        _problem(12, 1070), 3, diag="sdp", warm_start=[0, 4, 9], budget_s=1.0, clock_offset_s=2.0
    ).solver_meta
    assert spy.calls == []
    assert meta["diag_method"] == "mineig" and "budget" in meta["diag_fallback"]
    assert meta["warm_start_used"] is True
