"""RQ3 operator (г): the k best supports of a fold (ANALYSIS_PLAN §2 RQ3, §5; consultation 2026-10-08).

The reference is enumeration, independent of SCIP and of the code under test: every K-subset is
refined with the shared refinement, the distinct supports ``{i : w_i > 1e-6}`` of the results are
collected (the best objective per support) and sorted by the objective. The enumerator must return
exactly the first ``R + 1`` of them, each certified. Ties are checked on exact twins (two assets
with identical returns and covariances: swapping them is a symmetry, so the portfolios that hold
exactly one of the twins come in pairs of equal objective).

Two cardinality models are covered. ``card_mode="eq"`` (``Σz = K``, the default) enumerates the
refinements of K-subsets, the object of the plan. ``card_mode="le"`` (``Σz ≤ K``) enumerates the
refinements of all sets of at most K assets, which on small instances contain portfolios that no
K-subset refines to (``test_le_mode_...``).

A SCIP call costs ≥ 0.4 s whatever the size, so the instances are small and R is kept low.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from cif.backtest.injection import BP, gmv_problem
from cif.backtest.kbest import (
    N_RANKS_FF49,
    N_RANKS_US,
    KBestResult,
    _order_with_ties,
    enumerate_kbest,
)
from cif.classical.refine import refine_problem
from cif.classical.scip_native import solve_miqp_native
from cif.instances.factor import factor_instance
from cif.metrics.gap import reference_scale
from cif.problem import PortfolioProblem

SUPPORT_TOL = 1e-6
TIE_REL = 1e-6  # the tolerance of the module: ABS_GAP_REL · s_ref
#: R of the shared enumerations: ranks 2…11 on the binding fold (R up to 10 as the plan's check
#: asks), 2…7 elsewhere to keep the run short
_RANKS = {"binding": 10, "nonbinding": 6, "capped": 6, "gmv": 6}


# ---------------------------------------------------------------------------------------------
# instances and the enumeration oracle
# ---------------------------------------------------------------------------------------------


def _instance(n: int, seed: int, k: int, lam: float, u: float) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)),
                            risk_aversion=lam, w_max=u, cardinality=k)


def _twin_instance(n_base: int, seed: int, k: int, lam: float, u: float,
                   rho: float = 0.5) -> PortfolioProblem:
    """Asset 1 is an exact twin of asset 0: same μ, same covariances with the others, correlation
    ``rho`` between the two. Swapping 0 and 1 maps (μ, Σ) to itself."""
    mu_b, sigma_b, _ = factor_instance(n_base, seed)
    orig = [0, 0, *range(1, n_base)]
    sigma = sigma_b[np.ix_(orig, orig)].copy()
    sigma[0, 1] = sigma[1, 0] = rho * sigma_b[0, 0]
    return PortfolioProblem(mu=mu_b[orig], sigma=sigma,
                            asset_names=tuple(f"t{i}" for i in range(n_base + 1)),
                            risk_aversion=lam, w_max=u, cardinality=k)


def _portfolios(problem: PortfolioProblem, sizes) -> dict[tuple[int, ...], tuple[float, np.ndarray]]:
    """``{support: (F, weights)}`` over the refinements of all sets of the given sizes."""
    out: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for size in sizes:
        for s in itertools.combinations(range(problem.n), size):
            res = refine_problem(problem, s)
            if res.status != "optimal":
                continue
            t = tuple(int(i) for i in np.flatnonzero(res.weights > SUPPORT_TOL))
            if t not in out or res.objective < out[t][0]:
                out[t] = (res.objective, res.weights)
    return out


def _oracle(problem: PortfolioProblem, k: int, mode: str = "eq", tol: float = 0.0):
    """Distinct refined portfolios ``[(support, F, weights)]`` in the order of the plan: by F, and
    portfolios whose F follow each other within ``tol`` (a chain) in lexicographic order of the
    support. ``mode="eq"``: refinements of K-subsets; ``"le"``: of all sets of at most K assets."""
    sizes = (k,) if mode == "eq" else range(1, k + 1)
    items = sorted(((t, f, w) for t, (f, w) in _portfolios(problem, sizes).items()),
                   key=lambda r: (r[1], r[0]))
    groups: list[list] = []
    for item in items:
        if groups and item[1] - groups[-1][-1][1] <= tol:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [item for g in groups for item in sorted(g, key=lambda r: r[0])]


def _clear_cut(weights: np.ndarray) -> bool:
    """No weight sits between the interior-point noise of a zero (≤ 4·10⁻⁷ observed) and a
    clearly positive weight, so the support of the portfolio is not a knife-edge call."""
    return bool(np.all((weights <= 5e-7) | (weights >= 1e-4)))


def _optimum_weights(problem: PortfolioProblem) -> np.ndarray:
    """SCIP's optimum (raw weights, as a caller of the enumerator has them)."""
    return solve_miqp_native(problem, problem.cardinality, mip_gap=0.0).weights


def _support(weights: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(weights > SUPPORT_TOL))


# (n, seed, K, λ, u); the notes say why the instance was chosen
_BINDING = (12, 1003, 4, 5.0, 1.0)  # s_cont = 9 > K = 4
_NONBINDING = (11, 1004, 6, 1.0, 1.0)  # s_cont = 2 < K = 6: optimum on 2 assets
_CAPPED = (10, 1002, 4, 5.0, 0.3)  # u = 0.3, K = 4: four weights in [0.1, 0.3], the cap binds
_DIVERGENT = (8, 1008, 3, 10.0, 1.0)  # "le" and "eq" part at rank 7
_CASES = {
    "binding": _BINDING,
    "nonbinding": _NONBINDING,
    "capped": _CAPPED,
    "gmv": _BINDING,
}


def _case_problem(name: str) -> PortfolioProblem:
    problem = _instance(*_CASES[name])
    return gmv_problem(problem) if name == "gmv" else problem


@pytest.fixture(scope="module")
def cases():
    """Per case: the problem, the oracle (K-subsets), SCIP's optimum and the enumeration."""
    out = {}
    for name in _CASES:
        problem = _case_problem(name)
        oracle = _oracle(problem, problem.cardinality)
        w_star = _optimum_weights(problem)
        out[name] = (problem, oracle, w_star,
                     enumerate_kbest(problem, w_star, n_ranks=_RANKS[name]))
    return out


# ---------------------------------------------------------------------------------------------
# ordering of ties (pure function)
# ---------------------------------------------------------------------------------------------


def test_without_ties_the_alternatives_are_sorted_by_objective():
    out = _order_with_ties([3e-3, 1e-3, 2e-3], [(0, 1), (2, 3), (1, 4)], 0.0, 1e-6)
    assert out.order == (1, 2, 0)
    assert out.tied == (False, False, False)
    assert not out.anchor_tied


def test_tied_alternatives_are_ordered_lexicographically_by_support():
    objectives = [0.5, 0.5 + 1e-9, 0.7]
    supports = [(2, 3), (1, 4), (0, 1)]
    out = _order_with_ties(objectives, supports, 0.0, 1e-6)
    assert out.order == (1, 0, 2)  # (1, 4) < (2, 3) although its objective is the larger
    assert out.tied == (True, True, False)
    assert out.group == (0, 0, 1)


def test_ties_chain_through_neighbours_not_through_the_first_member():
    # 0.8·tol apart each: the ends are 1.6·tol apart but one chain
    objectives = [1.0, 1.0 + 0.8e-6, 1.0 + 1.6e-6]
    out = _order_with_ties(objectives, [(5,), (4,), (3,)], 0.0, 1e-6)
    assert out.order == (2, 1, 0)
    assert out.tied == (True, True, True)
    assert len(set(out.group)) == 1


def test_a_difference_of_exactly_the_tolerance_is_a_tie_and_more_is_not():
    assert _order_with_ties([1.0, 1.25], [(1,), (0,)], 0.0, 0.25).tied == (True, True)
    out = _order_with_ties([1.0, 1.2500001], [(1,), (0,)], 0.0, 0.25)
    assert out.tied == (False, False) and out.order == (0, 1)


def test_the_discovery_order_does_not_matter():
    objectives = [0.9, 0.1, 0.5, 0.1 + 5e-7]
    supports = [(0,), (7,), (3,), (2,)]
    a = _order_with_ties(objectives, supports, 0.0, 1e-6)
    perm = [2, 0, 3, 1]
    b = _order_with_ties([objectives[i] for i in perm], [supports[i] for i in perm], 0.0, 1e-6)
    assert [supports[i] for i in a.order] == [[supports[i] for i in perm][j] for j in b.order]
    assert a.tied[a.order.index(1)] and a.tied[a.order.index(3)]  # the two near 0.1


def test_a_tie_with_the_optimum_is_flagged_but_never_reorders_it():
    out = _order_with_ties([1.0 + 5e-7, 2.0], [(0, 1), (2, 3)], 1.0, 1e-6)
    assert out.anchor_tied
    assert out.tied == (True, False)  # the first alternative is tied with rank 1
    assert out.order == (0, 1)
    far = _order_with_ties([1.0 + 5e-6, 2.0], [(0, 1), (2, 3)], 1.0, 1e-6)
    assert not far.anchor_tied and far.tied == (False, False)


def test_no_alternatives():
    out = _order_with_ties([], [], 1.0, 1e-6)
    assert out.order == () and out.tied == () and not out.anchor_tied


# ---------------------------------------------------------------------------------------------
# exactness against enumeration
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_CASES))
def test_the_ranks_are_the_first_distinct_refined_portfolios_of_the_enumeration(cases, name):
    _, oracle, _, res = cases[name]
    n = res.n_ranks + 1
    assert all(_clear_cut(w) for _, _, w in oracle[: n + 2])  # the guard against knife-edge supports
    assert res.complete and res.all_certified and not res.exhausted
    assert [r.rank for r in res.ranks] == list(range(1, n + 1))
    assert [r.support for r in res.ranks] == [t for t, _, _ in oracle[:n]]
    np.testing.assert_allclose([r.objective for r in res.ranks], [f for _, f, _ in oracle[:n]],
                               rtol=0.0, atol=1e-9)
    assert all(r.certified for r in res.ranks)
    # no repeats, and the first difference to the optimum is a positive dose
    assert len({r.support for r in res.ranks}) == n
    assert res.ranks[1].dose_bp > 1e-3


@pytest.mark.parametrize("name", list(_CASES))
def test_every_rank_is_a_feasible_refinement_of_its_own_support(cases, name):
    problem, _, _, res = cases[name]
    for r in res.ranks:
        w = r.weights
        assert w.shape == (problem.n,)
        assert w.sum() == pytest.approx(problem.budget, abs=1e-8)
        assert np.all(w >= 0.0) and np.all(w <= problem.w_max + 1e-9)
        assert _support(w) == r.support and len(r.support) <= problem.cardinality
        assert r.objective == pytest.approx(problem.objective_value(w), rel=1e-9, abs=1e-12)
        again = refine_problem(problem, r.support)  # refine(T_r) = the portfolio of the rank
        np.testing.assert_allclose(again.weights, w, atol=1e-8)


@pytest.mark.parametrize("name", list(_CASES))
def test_doses_are_the_objective_excess_over_the_optimum_in_basis_points(cases, name):
    _, oracle, _, res = cases[name]
    assert res.f_star == pytest.approx(oracle[0][1], abs=1e-9)
    assert res.ranks[0].dose_bp == 0.0
    for r in res.ranks:
        assert r.dose_bp == pytest.approx((r.objective - res.f_star) * BP, abs=1e-12)
    assert np.all(np.diff(res.doses_bp) > -TIE_REL * res.scale * BP)  # non-decreasing up to ε


def test_the_gmv_dose_is_half_lambda_times_the_variance_increment(cases):
    problem, _, _, res = cases["gmv"]
    assert np.all(problem.mu == 0.0)
    w0 = res.ranks[0].weights
    for r in res.ranks[1:]:
        extra = 0.5 * problem.risk_aversion * (r.weights @ problem.sigma @ r.weights
                                               - w0 @ problem.sigma @ w0)
        assert r.dose_bp == pytest.approx(extra * BP, abs=1e-9)


def test_a_larger_support_ranks_before_every_subset_of_it(cases):
    # the exactness argument: a refined portfolio whose support contains the support of another is
    # strictly better (equal would make them the same portfolio), so it has the lower rank
    for name in _CASES:
        res = cases[name][3]
        rank_of = {r.support: r.rank for r in res.ranks}
        for a, b in itertools.permutations(rank_of, 2):
            if set(b) < set(a):
                assert rank_of[a] < rank_of[b], (name, a, b)


def test_the_binding_cap_is_respected_and_active(cases):
    problem, _, _, res = cases["capped"]
    assert problem.w_max == 0.3
    assert all(r.weights.max() <= 0.3 + 1e-9 for r in res.ranks)
    assert any(np.isclose(r.weights.max(), 0.3, atol=1e-9) for r in res.ranks)  # not a vacuous case
    assert all(len(r.support) >= 4 for r in res.ranks)  # 1/u = 3.33: at least four assets


def test_the_nonbinding_fold_holds_fewer_than_k_assets_in_the_optimum(cases):
    problem, _, _, res = cases["nonbinding"]
    assert len(res.ranks[0].support) < problem.cardinality
    assert any(len(r.support) < problem.cardinality for r in res.ranks)


# ---------------------------------------------------------------------------------------------
# why the cuts are on supersets of the support, not no-good cuts on z
# ---------------------------------------------------------------------------------------------


def test_a_no_good_cut_on_z_returns_the_optimum_again_on_a_nonbinding_fold(cases):
    problem, _, _, res = cases["nonbinding"]
    k = problem.cardinality
    t1 = res.ranks[0].support
    s = len(t1)
    assert s < k
    # copies of the optimum: every K-set that holds the support refines to the same portfolio
    copies = [z for z in itertools.combinations(range(problem.n), k) if set(t1) <= set(z)]
    assert len(copies) == math.comb(problem.n - s, k - s) > 1
    # forbidding one z (a K-set; on Σz ≤ K a superset cut on a K-set is exactly a no-good cut)
    naive = solve_miqp_native(problem, k, mip_gap=0.0, forbidden_supersets=[copies[0]])
    assert set(t1) <= set(_support(naive.weights))
    assert naive.objective == pytest.approx(res.f_star, abs=1e-7)  # the same optimum again
    # the cut on the supersets of the support does not repeat it: rank 2 is a different portfolio
    assert not set(t1) <= set(res.ranks[1].support)
    assert res.ranks[1].dose_bp > 1.0


# ---------------------------------------------------------------------------------------------
# the cardinality model
# ---------------------------------------------------------------------------------------------


def test_le_mode_enumerates_portfolios_that_no_k_subset_refines_to():
    # N = 8, K = 3: at rank 7 the model with Σz ≤ K returns a portfolio on fewer than three assets
    # that every three-asset superset improves (each outside asset would enter); it is not the
    # refinement of any K-subset, so Σz = K (the plan's object) skips it
    problem = _instance(*_DIVERGENT)
    w_star = _optimum_weights(problem)
    eq = enumerate_kbest(problem, w_star, n_ranks=7, card_mode="eq")
    le = enumerate_kbest(problem, w_star, n_ranks=7, card_mode="le")
    assert eq.card_mode == "eq" and le.card_mode == "le"
    k_subset_portfolios = set(_portfolios(problem, (3,)))
    assert [r.support for r in eq.ranks] == [t for t, _, _ in _oracle(problem, 3, "eq")[:8]]
    assert [r.support for r in le.ranks] == [t for t, _, _ in _oracle(problem, 3, "le")[:8]]
    assert [r.support for r in le.ranks[:6]] == [r.support for r in eq.ranks[:6]]
    assert le.ranks[6].support not in k_subset_portfolios and len(le.ranks[6].support) < 3
    assert all(r.support in k_subset_portfolios for r in eq.ranks)
    assert eq.all_certified and le.all_certified


# ---------------------------------------------------------------------------------------------
# degenerate folds
# ---------------------------------------------------------------------------------------------


def test_a_fold_with_fewer_than_r_plus_one_portfolios_is_exhausted_and_degenerate():
    problem = _instance(4, 1003, 3, 5.0, 1.0)  # N = K + 1: four K-subsets in all
    oracle = _oracle(problem, 3)
    assert len(oracle) == 4
    res = enumerate_kbest(problem, _optimum_weights(problem), n_ranks=N_RANKS_FF49)
    assert res.exhausted and not res.complete
    assert res.n_distinct == 4 == len(res.ranks)
    assert res.all_certified  # SCIP proved that nothing else exists
    assert [r.support for r in res.ranks] == [t for t, _, _ in oracle]
    assert res.boundary_tie is None  # rank R + 1 does not exist: nothing to compare
    assert not res.qualifies(optimum_certified=True)


def test_n_equal_to_k_has_a_single_portfolio():
    problem = _instance(5, 1003, 5, 5.0, 1.0)  # Σz = K = N: one selection, and it holds the optimum
    res = enumerate_kbest(problem, _optimum_weights(problem), n_ranks=3)
    assert res.exhausted and res.n_distinct == 1 and res.all_certified
    assert not res.qualifies(optimum_certified=True)


def test_an_exact_count_of_portfolios_is_complete_and_qualifies():
    # R + 1 = 4 portfolios exist and R + 1 are requested: complete, the extra solve proves the end
    problem = _instance(4, 1003, 3, 5.0, 1.0)
    res = enumerate_kbest(problem, _optimum_weights(problem), n_ranks=3)
    assert res.complete and res.exhausted and res.n_distinct == 4
    assert res.boundary_tie is False
    assert res.qualifies(optimum_certified=True)
    assert not res.qualifies(optimum_certified=False)


# ---------------------------------------------------------------------------------------------
# ties and the boundary
# ---------------------------------------------------------------------------------------------

_TWIN = (7, 1011, 3, 5.0, 1.0)  # nine assets with a twin pair; ties after ranks 1, 3, 6, 8, 10


@pytest.fixture(scope="module")
def twin_case():
    """K = 3 on eight assets with a twin pair; the oracle uses the tolerance of the module."""
    problem = _twin_instance(*_TWIN)
    coarse = _oracle(problem, 3, tol=1e-9)  # exact twins agree to 1e-16: only to find s_ref
    tol = TIE_REL * reference_scale(problem, refine_problem(problem, coarse[0][0]).weights)
    return problem, _oracle(problem, 3, tol=tol), tol


def test_twins_make_exact_ties_that_are_flagged_and_ordered_lexicographically(twin_case):
    problem, oracle, tol = twin_case
    r = 8
    assert all(_clear_cut(w) for _, _, w in oracle[: r + 3])
    res = enumerate_kbest(problem, refine_problem(problem, oracle[0][0]).weights, n_ranks=r)
    assert [x.support for x in res.ranks] == [t for t, _, _ in oracle[: r + 1]]
    objectives = [x.objective for x in res.ranks]
    # consecutive ranks within the tolerance are tied and in lexicographic order
    pairs = [(a, b) for a, b in itertools.pairwise(res.ranks) if abs(b.objective - a.objective) <= tol]
    assert len(pairs) == 4  # ranks (1, 2), (3, 4), (6, 7), (8, 9)
    for a, b in pairs:
        assert a.support < b.support and a.tie and b.tie
    # the tie flag is exactly "within the tolerance of a neighbour"
    expected = [
        any(0 <= j < len(objectives) and abs(objectives[i] - objectives[j]) <= tol
            for j in (i - 1, i + 1))
        for i in range(len(objectives))
    ]
    assert [x.tie for x in res.ranks] == expected
    assert res.tie_share == pytest.approx(np.mean(expected[1:]))
    assert 0.0 < res.tie_share < 1.0
    assert res.boundary_tie is False and res.all_certified


def test_a_tie_is_ordered_by_support_not_by_a_tiny_objective_difference():
    # the twins differ by 1e-9 in μ: the pair is still a tie (≈1e-10 ≪ 1e-7) but the portfolio with
    # the better twin has the lower objective, so the order by objective alone is the wrong one
    base = _twin_instance(*_TWIN)
    mu = base.mu.copy()
    mu[1] += 1e-9
    problem = replace(base, mu=mu)
    coarse = _oracle(problem, 3, tol=1e-8)
    tol = TIE_REL * reference_scale(problem, refine_problem(problem, coarse[0][0]).weights)
    oracle = _oracle(problem, 3, tol=tol)
    r = 4
    by_objective = sorted(_portfolios(problem, (3,)).items(), key=lambda kv: kv[1][0])
    assert [t for t, _ in by_objective[: r + 1]] != [t for t, _, _ in oracle[: r + 1]]  # not vacuous
    res = enumerate_kbest(problem, refine_problem(problem, oracle[0][0]).weights, n_ranks=r)
    assert [x.support for x in res.ranks] == [t for t, _, _ in oracle[: r + 1]]
    for a, b in itertools.pairwise(res.ranks):
        if abs(b.objective - a.objective) <= tol:
            assert a.support < b.support and a.tie and b.tie
            assert b.objective != a.objective  # a genuine tiny difference, not an exact tie


def test_the_optimum_is_rank_one_whichever_of_two_tied_optima_is_given(twin_case):
    problem, oracle, tol = twin_case
    t_a, t_b = oracle[0][0], oracle[1][0]
    assert abs(oracle[1][1] - oracle[0][1]) <= tol  # the mirror of the optimum is tied with it
    for first, second in ((t_a, t_b), (t_b, t_a)):
        res = enumerate_kbest(problem, refine_problem(problem, first).weights, n_ranks=2)
        assert res.ranks[0].support == first  # never reordered
        assert res.ranks[1].support == second
        assert res.ranks[1].dose_bp == pytest.approx(0.0, abs=1e-6)
        assert res.ranks[0].tie and res.ranks[1].tie
        assert res.all_certified


def test_a_tie_straddling_the_boundary_is_detected(twin_case):
    problem, oracle, tol = twin_case
    w1 = refine_problem(problem, oracle[0][0]).weights
    f = [x[1] for x in oracle]
    # positions (2, 3) are the pair of ranks 3 and 4: R = 2 keeps rank 3, the extra one is rank 4
    assert f[3] - f[2] <= tol and f[2] - f[1] > 100 * tol
    res = enumerate_kbest(problem, w1, n_ranks=2)
    assert res.complete and len(res.ranks) == 3
    assert res.boundary_tie is True
    assert res.ranks[2].tie
    # the kept rank is the lexicographically smaller of the two tied portfolios
    assert res.ranks[2].support == oracle[2][0] and oracle[2][0] < oracle[3][0]
    # R = 1 keeps ranks 1, 2; rank 3 is far from rank 2: no tie at the boundary
    assert f[2] - f[1] > 100 * tol
    assert enumerate_kbest(problem, w1, n_ranks=1).boundary_tie is False
    # without the extra solve the boundary is not checked
    assert enumerate_kbest(problem, w1, n_ranks=2, boundary_check=False).boundary_tie is None


# ---------------------------------------------------------------------------------------------
# the certificate, the warm start, seeds
# ---------------------------------------------------------------------------------------------


def test_every_rank_solve_is_certified_by_the_gap_it_was_given(cases):
    problem, _, _, res = cases["binding"]
    assert res.abs_gap_limit == pytest.approx(1e-6 * reference_scale(problem, res.ranks[0].weights))
    r = res.n_ranks
    assert [s.solve for s in res.solves] == list(range(1, r + 2))  # R ranks and the extra solve
    for s in res.solves:
        assert s.certified and s.status in ("optimal", "gaplimit")
        assert s.abs_gap <= res.abs_gap_limit * (1 + 1e-9)  # 1e-6·s_ref of the optimum
        assert s.scale > 0.0  # the incumbent's own scale, for information
        assert s.abs_gap_limit == pytest.approx(res.abs_gap_limit)
        assert s.n_cuts == s.solve  # solve k forbids the supersets of the k portfolios found so far
        assert s.solve_time_s >= 0.0 and s.build_time_s >= 0.0 and s.n_nodes >= 0
    assert res.ranks[0].miqp is None
    assert sorted(x.miqp.solve for x in res.ranks[1:]) == list(range(1, r + 1))
    assert res.solve_time_s == pytest.approx(sum(s.solve_time_s for s in res.solves))
    assert res.elapsed_s >= res.solve_time_s


def test_a_rank_closed_within_the_given_gap_is_certified_whatever_the_scale_of_its_incumbent(
    cases, monkeypatch
):
    """SCIP gets ``limits/absgap = 1e-6·s_ref`` of the OPTIMUM (plan §2: "как у оптимума"). A MIQP it
    stops by that gap is certified: judging it by 1e-6 of the incumbent's own, smaller scale would
    refuse a gap that is inside the limit SCIP was given and end the enumeration there (E3: the
    portfolios of the higher ranks earn less, so their scale is smaller and the fold would leave
    F_k, more often where the doses are large)."""
    import cif.backtest.kbest as kbest

    problem, _, w_star, true = cases["binding"]
    scales = [reference_scale(problem, r.weights) for r in true.ranks]
    share = 0.9999
    assert min(scales[1:5]) < share * scales[0]  # the instance has the case: a smaller scale
    real = kbest.solve_miqp_native

    def stops_by_the_gap(*args, **kwargs):
        sol = real(*args, **kwargs)
        meta = dict(sol.solver_meta)
        if meta["status"] != "infeasible":  # SCIP stopped with the gap at 99.99 % of its limit
            meta.update(status="gaplimit", abs_gap=share * kwargs["abs_gap"],
                        dual_bound=meta["primal_bound"] - share * kwargs["abs_gap"])
        sol.solver_meta = meta
        return sol

    monkeypatch.setattr(kbest, "solve_miqp_native", stops_by_the_gap)
    res = enumerate_kbest(problem, w_star, n_ranks=4)
    assert res.complete and res.all_certified and res.n_distinct == 5
    assert all(s.certified and s.status == "gaplimit" for s in res.solves)
    assert [r.support for r in res.ranks] == [r.support for r in true.ranks[:5]]


def test_the_warm_start_is_accepted_at_every_rank_and_does_not_change_the_result(cases):
    problem, _, w_star, res = cases["binding"]
    assert all(s.warm_start_used for s in res.solves)
    cold = enumerate_kbest(problem, w_star, n_ranks=3, warm_start=False)
    assert not any(s.warm_start_used for s in cold.solves) and cold.pool_time_s == 0.0
    assert [r.support for r in cold.ranks] == [r.support for r in res.ranks[:4]]


def test_the_ranks_do_not_depend_on_the_solver_seeds(cases):
    problem, _, w_star, res = cases["binding"]
    for seed, perm in ((1, None), (2, 7)):
        other = enumerate_kbest(problem, w_star, n_ranks=3, seed=seed, permutation_seed=perm)
        assert [r.support for r in other.ranks] == [r.support for r in res.ranks[:4]]
        assert other.all_certified and other.seed == seed and other.permutation_seed == perm


def test_a_repeated_run_is_identical(cases):
    problem, _, w_star, res = cases["capped"]
    again = enumerate_kbest(problem, w_star, n_ranks=3)
    assert [r.support for r in again.ranks] == [r.support for r in res.ranks[:4]]
    np.testing.assert_array_equal([r.objective for r in again.ranks],
                                  [r.objective for r in res.ranks[:4]])


def test_the_big_m_formulation_gives_the_same_ranks(cases):
    problem, _, w_star, res = cases["binding"]
    bigm = enumerate_kbest(problem, w_star, n_ranks=3, formulation="bigm")
    assert [r.support for r in bigm.ranks] == [r.support for r in res.ranks[:4]]
    assert bigm.all_certified


def test_the_cardinality_can_be_passed_instead_of_being_set_on_the_problem(cases):
    problem, _, w_star, res = cases["binding"]
    bare = replace(problem, cardinality=None)
    other = enumerate_kbest(bare, w_star, n_ranks=1, cardinality=4)
    assert [r.support for r in other.ranks] == [r.support for r in res.ranks[:2]]


def test_r_is_a_parameter_and_a_shorter_run_is_a_prefix(cases):
    problem, _, w_star, res = cases["binding"]
    short = enumerate_kbest(problem, w_star, n_ranks=N_RANKS_US)
    assert N_RANKS_US == 1 and N_RANKS_FF49 == 32
    assert short.n_ranks == 1 and len(short.ranks) == 2 and len(short.solves) == 2  # rank 2 + extra
    assert [r.support for r in short.ranks] == [r.support for r in res.ranks[:2]]
    no_extra = enumerate_kbest(problem, w_star, n_ranks=1, boundary_check=False)
    assert len(no_extra.solves) == 1 and no_extra.boundary_tie is None


# ---------------------------------------------------------------------------------------------
# failures
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sixteen():
    mu, sigma, _ = factor_instance(16, 1000)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(map(str, range(16))),
                               risk_aversion=5.0, w_max=0.2, cardinality=6)
    return problem, _optimum_weights(problem)


def test_a_time_limit_without_an_incumbent_ends_the_enumeration_uncertified(sixteen):
    problem, w_star = sixteen
    res = enumerate_kbest(problem, w_star, n_ranks=3, time_limit_s=0.0, warm_start=False)
    assert res.stop_status == "timelimit"
    assert len(res.ranks) == 1 and not res.complete and not res.all_certified
    assert res.n_distinct == 1 and not res.exhausted and res.boundary_tie is None
    assert len(res.solves) == 1 and not res.solves[0].certified
    assert not res.qualifies(optimum_certified=True)


def test_a_time_limit_with_only_the_warm_start_reports_an_uncertified_rank(sixteen):
    problem, w_star = sixteen
    res = enumerate_kbest(problem, w_star, n_ranks=3, time_limit_s=0.0)
    assert res.stop_status == "timelimit" and not res.all_certified and not res.complete
    assert len(res.ranks) == 2 and res.ranks[1].certified is False
    assert res.ranks[1].miqp is not None and not res.ranks[1].miqp.certified
    assert res.ranks[1].miqp.warm_start_used
    assert not res.qualifies(optimum_certified=True)


# ---------------------------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------------------------


def _ok_args():
    problem = _instance(*_BINDING)
    return problem, _optimum_weights(problem)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda p, w: (p, w[:-1], {}), "w_star"),
        (lambda p, w: (p, w * 1.01, {}), "budget"),
        (lambda p, w: (p, np.r_[-0.1, 1.1, np.zeros(w.size - 2)], {}), "box"),
        (lambda p, w: (p, w, {"n_ranks": 0}), "n_ranks"),
        (lambda p, w: (p, w, {"card_mode": "ge"}), "card_mode"),
        (lambda p, w: (p, w, {"abs_gap_rel": -1.0}), "abs_gap_rel"),
        (lambda p, w: (p, w, {"abs_gap_rel": math.nan}), "abs_gap_rel"),
        (lambda p, w: (p, np.r_[np.full(4, 0.25), np.zeros(w.size - 4)], {}), "not optimal"),
        (lambda p, w: (PortfolioProblem(mu=p.mu, sigma=p.sigma, asset_names=p.asset_names,
                                        risk_aversion=p.risk_aversion), w, {}), "cardinality"),
        (lambda p, w: (PortfolioProblem(mu=p.mu, sigma=p.sigma, asset_names=p.asset_names,
                                        risk_aversion=p.risk_aversion, w_min=0.02,
                                        cardinality=4), w, {}), "w_min"),
        (lambda p, w: (p, w, {"cardinality": 3}), "more than K"),
    ],
    ids=["shape", "budget", "box", "n_ranks", "card_mode", "abs_gap_neg", "abs_gap_nan",
         "not-optimal", "no-cardinality", "w_min", "support-larger-than-k"],
)
def test_invalid_arguments_raise(change, match):
    problem, w = _ok_args()
    problem, w, kwargs = change(problem, w)
    with pytest.raises(ValueError, match=match):
        enumerate_kbest(problem, w, **kwargs)


def test_a_better_neighbour_than_the_given_optimum_is_a_broken_certificate(cases):
    problem, _, _, res = cases["binding"]
    second_best = res.ranks[1].weights  # optimal on its own support, but not the optimum
    with pytest.raises(ValueError, match="not optimal"):
        enumerate_kbest(problem, second_best, n_ranks=2)


def test_the_result_type_exposes_the_summary_quantities(cases):
    res = cases["binding"][3]
    assert isinstance(res, KBestResult)
    assert res.n_ranks == 10 and res.n_distinct == 11 and res.complete and res.all_certified
    assert res.supports == tuple(r.support for r in res.ranks)
    assert res.boundary_tie is False
    assert res.qualifies(optimum_certified=True) and not res.qualifies(optimum_certified=False)
    assert 0.0 <= res.tie_share <= 1.0
    assert res.stop_status is None and res.tie_tol == pytest.approx(res.abs_gap_limit)
