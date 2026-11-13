"""RQ1 surrogate oracle (ANALYSIS_PLAN §2 RQ1): enumeration, T_m, E_form / E_pipe / E_samp.

The oracle ranks every K-subset by the equal-weight binary-inclusion surrogate, refines the m best
with the shared :func:`cif.classical.refine.refine_problem` and compares them with the certified
optimum f*. Checked against the BQM energy, brute-force sorting with ``itertools`` and an
exhaustive optimum on tiny instances.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

import cif.heuristics.oracle as oracle_module
from cif.classical.refine import refine_problem
from cif.heuristics.oracle import (
    DEFAULT_MS,
    MAX_SUBSETS,
    SubsetValues,
    SurrogateOracle,
    best_subsets,
    certified_optimum,
    enumerate_subsets,
    exhaustive_optimum,
    objective_scale,
    pool_subsets,
    prefix_minima,
    surrogate_energies,
)
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem
from cif.qubo.builder import build_selection_bqm


def _problem(n: int, seed: int, lam: float = 5.0, w_max: float = 1.0, w_min: float = 0.0):
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)),
                            risk_aversion=lam, w_min=w_min, w_max=w_max)


def _tied_problem(n: int = 6) -> PortfolioProblem:
    """Identical, uncorrelated assets: every K-subset has exactly the same surrogate energy."""
    return PortfolioProblem(mu=np.full(n, 0.07), sigma=0.04 * np.eye(n),
                            asset_names=tuple(f"t{i}" for i in range(n)), risk_aversion=5.0)


# --------------------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", range(1, 10))
def test_enumeration_equals_itertools_in_lexicographic_order(n):
    for k in range(1, n + 1):
        got = enumerate_subsets(n, k)
        expected = np.array(list(combinations(range(n), k)))
        assert got.shape == (math.comb(n, k), k)
        np.testing.assert_array_equal(got, expected)


def test_enumeration_at_the_size_of_the_study_n24_k8():
    got = enumerate_subsets(24, 8)
    assert got.shape == (math.comb(24, 8), 8) == (735_471, 8)
    assert got.min() == 0 and got.max() == 23
    assert np.all(np.diff(got.astype(np.int64), axis=1) > 0)  # sorted tuples of distinct indices
    code = (got.astype(np.int64) * 24 ** np.arange(7, -1, -1)).sum(axis=1)
    assert np.all(np.diff(code) > 0)  # strictly increasing: lexicographic, no duplicates
    np.testing.assert_array_equal(got[0], np.arange(8))
    np.testing.assert_array_equal(got[-1], np.arange(16, 24))


def test_enumeration_refuses_more_than_the_limit_and_bad_sizes():
    assert MAX_SUBSETS == 2_000_000
    with pytest.raises(ValueError, match="limit"):
        enumerate_subsets(40, 10)  # C(40, 10) = 847,660,528
    assert enumerate_subsets(30, 5, max_subsets=math.comb(30, 5)).shape[0] == math.comb(30, 5)
    with pytest.raises(ValueError):
        enumerate_subsets(30, 5, max_subsets=math.comb(30, 5) - 1)
    for bad in (0, -1, 11):
        with pytest.raises(ValueError):
            enumerate_subsets(10, bad)


def test_enumeration_handles_cardinalities_above_half_of_n():
    # C(100, 97) = 161,700 would overflow a naive binomial table
    got = enumerate_subsets(100, 97)
    assert got.shape == (math.comb(100, 3), 97)
    assert got[0].tolist() == list(range(97))
    assert got[-1].tolist() == list(range(3, 100))


# --------------------------------------------------------------------------------------
# Surrogate energy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(14, 5), (24, 8)])
def test_surrogate_energy_is_the_bqm_energy_on_feasible_states(n, k):
    """On |S| = K the penalty vanishes: E(S) = f(x/K), whatever the penalty multiplier."""
    problem = _problem(n, 1000)
    rng = np.random.default_rng(0)
    subsets = np.array([np.sort(rng.choice(n, size=k, replace=False)) for _ in range(25)])
    energy = surrogate_energies(problem.mu, problem.sigma, problem.risk_aversion, subsets)
    for multiplier in (1.0, 100.0):
        bqm = build_selection_bqm(problem, cardinality=k, penalty_multiplier=multiplier)
        for s, e in zip(subsets, energy, strict=True):
            sample = {i: int(i in s) for i in range(n)}
            assert e == pytest.approx(bqm.energy(sample), rel=1e-12, abs=1e-12)
    for s, e in zip(subsets, energy, strict=True):
        x = np.zeros(n)
        x[s] = 1.0 / k
        assert e == pytest.approx(problem.objective_value(x), rel=1e-12, abs=1e-12)


def test_surrogate_energy_rejects_bad_input():
    problem = _problem(10, 1001)
    with pytest.raises(ValueError):
        surrogate_energies(problem.mu, problem.sigma, 5.0, np.array([0, 1, 2]))  # not 2-D
    bad = problem.mu.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        surrogate_energies(bad, problem.sigma, 5.0, np.array([[0, 1, 2]]))


# --------------------------------------------------------------------------------------
# T_m: the m best subsets with deterministic tie-breaking
# --------------------------------------------------------------------------------------


def test_best_subsets_match_a_brute_force_sort():
    problem = _problem(12, 1002)
    k, m = 4, 30
    sur = {s: e for s, e in zip(
        combinations(range(12), k),
        surrogate_energies(problem.mu, problem.sigma, 5.0, np.array(list(combinations(range(12), k)))),
        strict=True)}
    expected = sorted(sur, key=lambda s: (sur[s], s))[:m]
    subsets, energy = best_subsets(problem, k, m)
    assert [tuple(s) for s in subsets.tolist()] == expected
    np.testing.assert_allclose(energy, [sur[s] for s in expected], rtol=0, atol=0)
    assert np.all(np.diff(energy) >= 0)


def test_best_subsets_are_nested_in_m():
    problem = _problem(14, 1003)
    big, _ = best_subsets(problem, 5, 50)
    for m in (1, 5, 20):
        small, _ = best_subsets(problem, 5, m)
        np.testing.assert_array_equal(small, big[:m])


def test_ties_are_broken_by_lexicographic_order_of_the_subset():
    problem = _tied_problem(6)
    subsets, energy = best_subsets(problem, 3, 4)
    assert subsets.tolist() == [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]]
    assert len(set(energy.tolist())) == 1
    # a tie only at the cut: the first of two equal subsets wins the last place
    mu = np.array([0.30, 0.20, 0.10, 0.10, 0.0, 0.0])
    problem2 = PortfolioProblem(mu=mu, sigma=0.04 * np.eye(6), asset_names=tuple("abcdef"),
                                risk_aversion=5.0)
    subsets2, _ = best_subsets(problem2, 2, 2)
    assert subsets2.tolist() == [[0, 1], [0, 2]]  # (0, 2) and (0, 3) tie; the lexicographically first stays


def test_best_subsets_return_everything_if_m_exceeds_the_count():
    problem = _problem(8, 1004)
    subsets, energy = best_subsets(problem, 3, 1000)
    assert subsets.shape == (math.comb(8, 3), 3) and energy.shape == (56,)
    assert len({tuple(s) for s in subsets.tolist()}) == 56


def test_best_subsets_at_the_study_size_n24_k8():
    problem = _problem(24, 1020)
    subsets, energy = best_subsets(problem, 8, 200)
    assert subsets.shape == (200, 8) and np.all(np.diff(energy) >= 0)
    # the best of 735,471 equals a direct minimum over a fresh enumeration
    full = surrogate_energies(problem.mu, problem.sigma, 5.0, enumerate_subsets(24, 8))
    assert energy[0] == full.min()


# --------------------------------------------------------------------------------------
# F(S) through the shared refinement
# --------------------------------------------------------------------------------------


def test_subset_values_equal_the_shared_refinement_and_are_cached():
    problem = _problem(15, 1005)
    values = SubsetValues(problem)
    direct = refine_problem(problem, [1, 4, 7, 9], "le").objective
    assert values((9, 4, 1, 7)) == pytest.approx(direct, rel=0, abs=0)
    assert values([1, 4, 7, 9]) == values((1, 4, 7, 9))
    assert values.n_calls == 3 and values.n_refined == 1
    assert values.time_s > 0.0
    values((0, 1, 2))
    assert values.n_calls == 4 and values.n_refined == 2


def test_subset_values_in_eq_mode_hold_every_asset():
    problem = _problem(12, 1006, w_min=0.05)
    values = SubsetValues(problem, card_mode="eq")
    res = refine_problem(problem, [0, 3, 5], "eq")
    assert values((0, 3, 5)) == res.objective
    with pytest.raises(ValueError, match="w_min"):
        SubsetValues(problem, card_mode="le")((0, 3, 5))


def test_subset_values_report_infeasible_supports_as_inf():
    problem = _problem(10, 1007, w_max=0.2)  # 3 assets cannot carry the budget at 0.2 each
    values = SubsetValues(problem)
    assert values((0, 1, 2)) == math.inf
    assert values.statuses["infeasible"] == 1


# --------------------------------------------------------------------------------------
# Reference f* and the positive scale s*
# --------------------------------------------------------------------------------------


def test_objective_scale_is_positive_and_matches_the_definition():
    problem = _problem(10, 1008)
    w = np.zeros(10)
    w[[2, 5]] = [0.4, 0.6]
    expected = abs(problem.mu @ w) + 0.5 * problem.risk_aversion * w @ problem.sigma @ w
    assert objective_scale(problem, w) == pytest.approx(expected, rel=1e-14)
    # positive even where f changes sign: μ = 0 gives f = (λ/2)wΣw but the scale never passes through 0
    flat = PortfolioProblem(mu=np.zeros(3), sigma=np.eye(3) * 0.04, asset_names=("a", "b", "c"),
                            risk_aversion=5.0)
    assert objective_scale(flat, np.array([1.0, 0.0, 0.0])) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="positive"):
        objective_scale(flat, np.zeros(3))


def test_exhaustive_optimum_equals_the_minimum_over_all_k_subsets():
    problem = _problem(9, 1009)
    ref = exhaustive_optimum(problem, 3)
    brute = min(refine_problem(problem, s, "le").objective for s in combinations(range(9), 3))
    assert ref.f_star == pytest.approx(brute, rel=0, abs=1e-15)
    assert ref.source == "enumeration" and ref.ref_type == "certified"
    assert ref.scale == pytest.approx(objective_scale(problem, ref.weights))
    assert problem.objective_value(ref.weights) == pytest.approx(ref.f_star, abs=1e-12)
    with pytest.raises(ValueError, match="limit"):
        exhaustive_optimum(problem, 3, max_subsets=10)


def test_certified_optimum_from_scip_agrees_with_enumeration():
    problem = _problem(12, 1010)
    ref = certified_optimum(problem, 3, time_limit_s=60.0)
    exact = exhaustive_optimum(problem, 3)
    assert ref.source == "scip" and ref.ref_type == "certified"
    assert ref.f_star == pytest.approx(exact.f_star, rel=0, abs=1e-9)
    # polished with the shared refinement: never above SCIP's own incumbent
    assert ref.f_star <= ref.meta["f_scip"] + 1e-12
    assert ref.meta["dual_bound"] <= ref.f_star + 1e-8
    assert ref.meta["status"] in ("optimal", "gaplimit") and ref.meta["gap"] <= 1e-6
    assert ref.scale > 0.0 and len(ref.meta["support"]) <= 3


def test_certified_optimum_in_eq_mode_agrees_with_enumeration():
    """Σz = K with a binding minimum buy (the ORlib protocol): every asset of the support held."""
    problem = _problem(12, 1011, w_min=0.05, w_max=0.6)
    ref = certified_optimum(problem, 3, card_mode="eq", time_limit_s=60.0)
    exact = exhaustive_optimum(problem, 3, card_mode="eq")
    assert ref.source == "scip" and ref.ref_type == "certified"
    assert ref.f_star == pytest.approx(exact.f_star, rel=0, abs=1e-9)
    support = ref.meta["support"]
    assert len(support) == 3 and np.all(ref.weights[support] >= 0.05 - 1e-9)


def test_certified_optimum_certificate_is_on_the_reference_scale(monkeypatch):
    """ref_type follows reference_from_exact with s_ref of the incumbent; a time limit never certifies."""
    problem = _problem(12, 1010)
    ref = certified_optimum(problem, 3, time_limit_s=60.0)
    gap_abs = ref.meta["primal_bound"] - ref.meta["dual_bound"]
    assert ref.ref_type == "certified" and gap_abs <= 1e-6 * ref.scale

    real = oracle_module.solve_miqp_native

    def stopped_by_the_clock(*args, **kwargs):  # same bounds, but SCIP reports a time limit
        sol = real(*args, **kwargs)
        sol.solver_meta.update(status="timelimit", time_limit_hit=True)
        return sol

    monkeypatch.setattr(oracle_module, "solve_miqp_native", stopped_by_the_clock)
    limited = certified_optimum(problem, 3, time_limit_s=60.0)
    assert limited.ref_type == "best_known" and limited.f_star == pytest.approx(ref.f_star)


# --------------------------------------------------------------------------------------
# Pools and prefix minima
# --------------------------------------------------------------------------------------


def test_prefix_minima_and_counts():
    values = [5.0, 3.0, 4.0, 1.0, 2.0]
    best, count = prefix_minima(values, (1, 2, 4, 10))
    np.testing.assert_array_equal(best, [5.0, 3.0, 1.0, 1.0])
    np.testing.assert_array_equal(count, [1, 2, 4, 5])
    best, count = prefix_minima([], (1, 5))
    assert np.isnan(best).all() and count.tolist() == [0, 0]


def test_pool_subsets_keep_the_given_order_and_drop_invalid_and_repeated():
    pool = [(3, 1, 2), (1, 2), (0, 1, 2), (1, 2, 3), (5, 6, 7), (0, 1, 2), (4, 8, 9), (2, 2, 3)]
    got, stats = pool_subsets(pool, k=3, n=8, limit=10)
    # (1, 2): wrong size; (5, 6, 7): fine; (4, 8, 9): index 8, 9 outside range(8); (2, 2, 3): size 2
    assert got == [(1, 2, 3), (0, 1, 2), (5, 6, 7)]
    assert stats == {"n_input": 8, "n_valid": 5, "n_unique": 3}
    got, stats = pool_subsets(pool, k=3, n=8, limit=2)
    assert got == [(1, 2, 3), (0, 1, 2)]  # truncated to the limit in the given order
    assert stats["n_unique"] == 3  # the counts describe the whole input, not the truncation


# --------------------------------------------------------------------------------------
# The oracle: E_form, E_pipe, E_samp
# --------------------------------------------------------------------------------------


def _oracle(n=9, k=3, seed=1012, ms=(1, 5, 20, 84)):
    problem = _problem(n, seed)
    ref = exhaustive_optimum(problem, k)
    return problem, ref, SurrogateOracle(problem, k, ref, ms=ms)


def test_e_form_is_the_normalised_gap_of_the_best_refined_member_of_t_m():
    problem, ref, oracle = _oracle()
    subsets, _ = best_subsets(problem, 3, 84)
    f = np.array([refine_problem(problem, s, "le").objective for s in subsets.tolist()])
    for m, e in zip(oracle.ms, oracle.e_form, strict=True):
        assert e == pytest.approx((f[:m].min() - ref.f_star) / ref.scale, abs=1e-14)
    assert np.all(oracle.e_form >= -1e-12)  # nobody beats the optimum
    assert np.all(np.diff(oracle.e_form) <= 0)  # nested T_m: non-increasing in m
    assert oracle.e_form[-1] == pytest.approx(0.0, abs=1e-12)  # T_84 is every 3-subset of 9
    assert oracle.n_form.tolist() == [1, 5, 20, 84]


def test_e_form_counts_cap_at_the_number_of_subsets():
    problem = _problem(6, 1013)
    ref = exhaustive_optimum(problem, 2)  # C(6, 2) = 15
    oracle = SurrogateOracle(problem, 2, ref, ms=(1, 5, 50))
    assert oracle.n_form.tolist() == [1, 5, 15]
    assert oracle.e_form[-1] == pytest.approx(0.0, abs=1e-12)


def _misranked():
    """Four uncorrelated assets (A high return and variance, B low variance, C = D medium).

    With K = 2 the equal-weight surrogate ranks (C, D) first and the optimum (A, B) last, because
    A's risk is charged at weight 1/2 while the refinement puts only about 0.23 on it.
    """
    problem = PortfolioProblem(mu=np.array([0.30, 0.05, 0.10, 0.10]),
                               sigma=np.diag([0.25, 0.01, 0.04, 0.04]), asset_names=tuple("ABCD"),
                               risk_aversion=5.0)
    ref = exhaustive_optimum(problem, 2)
    return problem, ref, SurrogateOracle(problem, 2, ref, ms=(1, 4, 6))


def test_the_equal_weight_surrogate_can_rank_the_optimum_last():
    problem, ref, oracle = _misranked()
    subsets, _ = best_subsets(problem, 2, 6)
    assert subsets.tolist() == [[2, 3], [1, 2], [1, 3], [0, 2], [0, 3], [0, 1]]  # ties: lexicographic
    assert ref.f_star == pytest.approx(-0.0596153846, abs=1e-9)
    assert oracle.e_form[0] == pytest.approx((-0.05 - ref.f_star) / ref.scale, abs=1e-12)
    assert oracle.e_form[0] > 0.01  # the ceiling at m = 1 is well above tau = 1e-3
    assert oracle.e_form[1] == pytest.approx((-0.05517241379 - ref.f_star) / ref.scale, abs=1e-9)
    assert oracle.e_form[2] == pytest.approx(0.0, abs=1e-12)


def test_e_pipe_takes_the_first_m_of_the_pool_in_the_given_order():
    problem, ref, oracle = _oracle()
    subsets, _ = best_subsets(problem, 3, 84)
    order = subsets.tolist()
    best_member = tuple(min(order, key=lambda s: refine_problem(problem, s, "le").objective))
    worst_member = tuple(max(order, key=lambda s: refine_problem(problem, s, "le").objective))
    pool = [worst_member, best_member]
    att = oracle.attribute(pool)
    f = lambda s: refine_problem(problem, s, "le").objective  # noqa: E731
    assert att.e_pipe[0] == pytest.approx((f(worst_member) - ref.f_star) / ref.scale, abs=1e-14)
    assert att.e_pipe[1] == pytest.approx((f(best_member) - ref.f_star) / ref.scale, abs=1e-14)
    assert att.e_pipe[1] == pytest.approx(0.0, abs=1e-12)
    # fewer than m valid unique subsets: use all of them and record the count
    assert att.n_pipe.tolist() == [1, 2, 2, 2]
    assert att.e_pipe[3] == att.e_pipe[1]


def test_e_samp_is_the_difference_and_is_negative_when_the_pool_beats_the_ceiling():
    _, _, oracle = _misranked()
    att = oracle.attribute([(0, 1)])  # the sampler found the optimum the surrogate ranks last
    np.testing.assert_allclose(att.e_samp, att.e_pipe - oracle.e_form, rtol=0, atol=1e-15)
    np.testing.assert_allclose(att.e_pipe, 0.0, atol=1e-12)
    assert att.e_samp[0] == pytest.approx(-oracle.e_form[0], abs=1e-12) and att.e_samp[0] < -0.01
    assert att.e_samp[1] < 0.0 and att.e_samp[2] == pytest.approx(0.0, abs=1e-12)
    # a pool that only repeats the surrogate's choice leaves the full ceiling as the gap
    same = oracle.attribute([(2, 3)])
    np.testing.assert_allclose(same.e_pipe, oracle.e_form[0], atol=1e-14)
    assert same.e_samp[0] == pytest.approx(0.0, abs=1e-14) and same.e_samp[2] > 0.0


def test_an_empty_pool_gives_undefined_e_pipe_and_a_zero_count():
    _, _, oracle = _oracle()
    att = oracle.attribute([(0, 1), (7, 7, 7), []])
    assert att.n_pipe.tolist() == [0, 0, 0, 0]
    assert np.isnan(att.e_pipe).all() and np.isnan(att.e_samp).all()
    row = att.row()
    assert row["E_pipe_1"] is None and row["E_samp_1"] is None and row["n_pipe_1"] == 0
    assert row["E_form_1"] == pytest.approx(oracle.e_form[0])


# --------------------------------------------------------------------------------------
# B2 (ANALYSIS_PLAN §12, 2026-10-11): E_samp(m) = E_prec(m) + E_short(m)
# --------------------------------------------------------------------------------------


def test_the_e_form_curve_is_the_prefix_minimum_of_the_refined_t_m():
    """``E_form(n)`` for every n up to m_max: what the oracle already refined, no new computation."""
    problem, ref, oracle = _oracle()
    subsets, _ = best_subsets(problem, 3, 84)
    f = np.array([refine_problem(problem, s, "le").objective for s in subsets.tolist()])
    curve = oracle.e_form_curve
    assert curve.shape == (84,)
    np.testing.assert_allclose(curve, (np.minimum.accumulate(f) - ref.f_star) / ref.scale, rtol=0, atol=1e-14)
    assert np.all(np.diff(curve) <= 0.0)
    for j, m in enumerate(oracle.ms):
        assert curve[m - 1] == oracle.e_form[j]  # the same number, bit for bit


def test_e_prec_compares_the_pool_with_the_oracle_of_the_same_size():
    _, _, oracle = _misranked()  # T_m: (2,3), (1,2), (1,3), (0,2), (0,3), (0,1); ms = (1, 4, 6)
    att = oracle.attribute([(2, 3), (1, 2), (1, 3)])  # the first three of T_m, in order: no sampler loss
    assert att.n_pipe.tolist() == [1, 3, 3]
    curve = oracle.e_form_curve
    np.testing.assert_allclose(att.e_form_pipe, [curve[0], curve[2], curve[2]], rtol=0, atol=0)
    np.testing.assert_allclose(att.e_prec, 0.0, atol=1e-14)  # equal size: the sampler is exact
    np.testing.assert_allclose(att.e_short, [0.0, curve[2] - oracle.e_form[1], curve[2] - oracle.e_form[2]],
                               rtol=0, atol=1e-14)
    assert att.e_short[1] > 0.0 and att.e_short[2] > 0.0  # the price of three subsets instead of four, six


def test_e_prec_is_negative_when_the_pool_beats_the_oracle_of_its_size():
    """The sampler found the optimum the surrogate ranks last: diversity beats accuracy (§2)."""
    _, _, oracle = _misranked()
    att = oracle.attribute([(0, 1)])
    curve = oracle.e_form_curve
    np.testing.assert_allclose(att.e_pipe, 0.0, atol=1e-12)
    np.testing.assert_allclose(att.e_prec, -curve[0], atol=1e-12)  # E_pipe − E_form(1) at every m
    assert att.e_prec[0] < -0.01


@pytest.mark.parametrize("pool", [[(2, 3)], [(0, 1)], [(2, 3), (1, 2), (1, 3)], [(0, 1), (2, 3), (1, 3), (0, 3), (0, 2), (1, 2)],
                                  [(1, 3), (1, 3), (0, 2)]])
def test_the_identity_e_samp_is_e_prec_plus_e_short_and_the_shortfall_is_not_negative(pool):
    _, _, oracle = _misranked()
    att = oracle.attribute(pool)
    np.testing.assert_allclose(att.e_samp, att.e_prec + att.e_short, rtol=0, atol=1e-15)
    assert np.all(att.e_short >= 0.0)  # E_form(n_pipe(m)) >= E_form(m): T_m is nested and n_pipe(m) <= m
    full = att.n_pipe == np.array(oracle.ms)
    assert np.all(att.e_short[full] == 0.0)  # a pool of m subsets pays no price for its size
    assert np.array_equal(att.e_form_pipe[full], oracle.e_form[full])


def test_an_empty_pool_has_no_e_prec_and_no_e_short():
    _, _, oracle = _oracle()
    att = oracle.attribute([(0, 1), (7, 7, 7), []])
    assert np.isnan(att.e_form_pipe).all() and np.isnan(att.e_prec).all() and np.isnan(att.e_short).all()
    row = att.row()
    assert row["E_prec_1"] is None and row["E_short_1"] is None and row["E_form_at_npipe_1"] is None


def test_the_row_carries_what_the_decomposition_needs():
    import json

    problem, _, oracle = _oracle()
    subsets, _ = best_subsets(problem, 3, 4)  # a short pool: 4 subsets for m = 5, 20, 84
    att = oracle.attribute([tuple(x) for x in subsets.tolist()])
    row = att.row()
    for m in oracle.ms:
        for key in ("E_form_at_npipe", "E_prec", "E_short"):
            assert f"{key}_{m}" in row, (key, m)
    json.dumps(row, allow_nan=False)
    for j, m in enumerate(oracle.ms):
        assert row[f"E_samp_{m}"] == pytest.approx(row[f"E_prec_{m}"] + row[f"E_short_{m}"], abs=1e-15)
        assert row[f"E_form_at_npipe_{m}"] == oracle.e_form_curve[int(att.n_pipe[j]) - 1]
    assert row["n_pipe_20"] == 4 and row["E_short_20"] == pytest.approx(row["E_form_at_npipe_20"] - row["E_form_20"])


def test_attribution_row_is_flat_json_friendly_and_complete():
    import json

    problem, _, oracle = _oracle()
    subsets, _ = best_subsets(problem, 3, 30)
    att = oracle.attribute([tuple(s) for s in subsets.tolist()])
    row = att.row()
    for m in oracle.ms:
        for key in ("E_form", "E_pipe", "E_samp", "n_form", "n_pipe"):
            assert f"{key}_{m}" in row
    assert {"tau_top", "tau_pool", "pool_n_input", "pool_n_valid", "pool_n_unique"} <= row.keys()
    json.dumps(row, allow_nan=False)  # no numpy scalars, no NaN, no inf
    # pool = the surrogate's own ranking: nothing is lost by sampling
    assert att.e_samp[0] == pytest.approx(0.0, abs=1e-14)
    assert row["pool_n_unique"] == 30


def test_pool_that_is_the_surrogate_ranking_has_zero_residual_everywhere():
    problem, _, oracle = _oracle(ms=(1, 5, 20))
    subsets, _ = best_subsets(problem, 3, 20)
    att = oracle.attribute([tuple(s) for s in subsets.tolist()])
    np.testing.assert_allclose(att.e_samp, 0.0, atol=1e-14)


def test_oracle_reuses_one_value_cache_between_t_m_and_the_pool():
    problem = _problem(10, 1014)
    ref = exhaustive_optimum(problem, 3)
    values = SubsetValues(problem)
    oracle = SurrogateOracle(problem, 3, ref, ms=(1, 5, 20), values=values)
    refined = values.n_refined
    assert refined == 20  # only T_20 was refined
    subsets, _ = best_subsets(problem, 3, 20)
    oracle.attribute([tuple(s) for s in subsets.tolist()])
    assert values.n_refined == refined  # the pool was in the cache


def test_ms_must_be_positive_and_strictly_increasing():
    problem = _problem(8, 1015)
    ref = exhaustive_optimum(problem, 2)
    for bad in ((), (0, 1), (5, 5), (5, 1)):
        with pytest.raises(ValueError, match="ms"):
            SurrogateOracle(problem, 2, ref, ms=bad)
    assert DEFAULT_MS == (1, 5, 20, 50, 200)


def test_oracle_at_the_study_size_n24_k8_runs_with_scip_reference():
    """One calibration instance at the largest RQ1 cell: certified f*, nested T_m, E >= 0."""
    problem = _problem(24, 1020)
    ref = certified_optimum(problem, 8, time_limit_s=300.0)
    assert ref.ref_type == "certified"
    oracle = SurrogateOracle(problem, 8, ref)
    assert oracle.n_subsets == math.comb(24, 8)
    assert oracle.n_form.tolist() == list(DEFAULT_MS)
    assert np.all(oracle.e_form >= -1e-6)
    assert np.all(np.diff(oracle.e_form) <= 1e-15)
