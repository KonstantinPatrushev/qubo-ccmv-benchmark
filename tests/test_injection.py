"""RQ3 gap injection: operator (б), support swaps (ANALYSIS_PLAN §2 RQ3).

The certified optimum of the test instance is found by enumeration over K-subsets with the shared
refinement, independently of any solver. The Metropolis chain is checked on an additive "gap
oracle" where every neighbour gap is known by hand.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cif.backtest.injection import (
    GapInjector,
    _accepted_directions,
    _bisect_band,
    _complete_start,
    _swap_chain,
    gmv_problem,
    replica_seed,
)
from cif.classical.refine import refine_problem
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem

BP = 1e4


def _instance(n: int, k: int, seed: int, lam: float = 5.0, u: float = 0.4) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)),
                            risk_aversion=lam, w_max=u, cardinality=k)


def _objectives(problem: PortfolioProblem) -> dict[tuple[int, ...], float]:
    """Refined objective of every K-subset (card_mode 'le')."""
    return {s: refine_problem(problem, s).objective
            for s in itertools.combinations(range(problem.n), problem.cardinality)}


def _certified(problem: PortfolioProblem) -> tuple[float, tuple[int, ...], np.ndarray]:
    """Optimum over all K-subsets by enumeration: ``(f*, support, w*)``."""
    objectives = _objectives(problem)
    best = min(objectives, key=objectives.get)
    return objectives[best], best, refine_problem(problem, best).weights


@pytest.fixture(scope="module")
def enumeration():
    problem = _instance(14, 5, 1002)
    return problem, _objectives(problem)


@pytest.fixture(scope="module")
def case(enumeration):
    problem, objectives = enumeration
    best = min(objectives, key=objectives.get)
    return problem, refine_problem(problem, best).weights, objectives[best], best


@pytest.fixture
def injector(case):
    problem, w_star, _, _ = case
    return GapInjector(problem, w_star)


# ---------------------------------------------------------------------------------------------
# start state S0
# ---------------------------------------------------------------------------------------------


def test_start_contains_the_optimum_support_and_has_zero_gap(case, injector):
    problem, w_star, f_star, support = case
    assert injector.support == support
    assert injector.start == support  # binding K: the optimum fills all K places
    assert len(injector.start) == problem.cardinality
    assert injector.f_star == pytest.approx(f_star, abs=1e-12)
    assert injector.gap_bp(injector.start) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(injector.w_star, w_star, atol=1e-8)


def test_complete_start_takes_largest_relaxation_weights_then_the_lowest_index():
    relax = np.array([0.0, 0.3, 0.1, 0.2, 0.4, 0.0, 0.0, 0.0])
    # support {1, 4}; two free places go to the largest relaxation weights outside it: 3, 2
    assert _complete_start((1, 4), relax, 4, tol=1e-6) == (1, 2, 3, 4)
    # weights below the tolerance are ties and are resolved by index, not by solver noise
    noisy = np.array([3e-10, 0.0, 0.0, 2e-9, 0.0, 1e-9, 0.0, 0.0])
    assert _complete_start((1,), noisy, 3, tol=1e-6) == (0, 1, 2)
    assert _complete_start((0, 1, 2), noisy, 3, tol=1e-6) == (0, 1, 2)


def test_nonbinding_optimum_is_completed_to_k_assets_and_has_a_zero_gap_plateau():
    # three good assets, five bad ones, K = 6 > s_cont: the optimum is the continuous one
    mu = np.array([0.30, 0.28, 0.26, 0.05, 0.04, 0.03, 0.02, 0.01])
    sigma = np.diag(np.full(8, 0.04))
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple("abcdefgh"),
                               risk_aversion=5.0, w_max=0.5, cardinality=6)
    _, _, w_star = _certified(problem)
    assert np.count_nonzero(w_star > 1e-6) == 3
    inj = GapInjector(problem, w_star)
    assert inj.support == (0, 1, 2)
    assert set(inj.support) <= set(inj.start) and len(inj.start) == 6
    assert inj.gap_bp(inj.start) == pytest.approx(0.0, abs=1e-9)
    gaps = inj.neighbour_gaps_bp()
    plateau = np.abs(gaps) <= 1e-4
    assert plateau.any() and not plateau.all()  # swapping a zero-weight asset changes nothing
    # "smallest positive" skips the plateau
    assert inj.min_positive_neighbour_gap_bp() == pytest.approx(gaps[~plateau].min())
    assert inj.min_positive_neighbour_gap_bp() > 1e-4


def test_chains_cross_the_zero_gap_plateau_of_a_nonbinding_optimum():
    mu = np.array([0.30, 0.28, 0.26, 0.05, 0.04, 0.03, 0.02, 0.01])
    problem = PortfolioProblem(mu=mu, sigma=np.diag(np.full(8, 0.04)), asset_names=tuple("abcdefgh"),
                               risk_aversion=5.0, w_max=0.5, cardinality=6)
    _, _, w_star = _certified(problem)
    inj = GapInjector(problem, w_star)
    positive = np.sort(inj.neighbour_gaps_bp()[inj.neighbour_gaps_bp() > 1e-4])
    g = float(positive[len(positive) // 2]) / 1.05  # a band that holds one-swap neighbours
    res = inj.swap_level(g, base_seed=3, fold=1, level_key=1, n_replicas=5)
    assert res.reached
    for rep in res.replicas:
        assert g <= rep.gap_bp <= 1.1 * g
        assert set(inj.support) - set(rep.subset)  # a held asset was swapped out: the gap is real


def test_no_outside_asset_means_no_neighbour_and_every_level_is_unreachable():
    mu = np.array([0.10, 0.08, 0.06, 0.05])
    problem = PortfolioProblem(mu=mu, sigma=np.diag([0.04, 0.05, 0.03, 0.06]), asset_names=tuple("abcd"),
                               risk_aversion=5.0, w_max=0.6, cardinality=4)  # K = N
    _, _, w_star = _certified(problem)
    inj = GapInjector(problem, w_star)
    assert inj.neighbour_gaps_bp().size == 0 and inj.min_positive_neighbour_gap_bp() == math.inf
    res = inj.swap_level(10.0, base_seed=1, fold=0, n_replicas=2)
    assert res.status == "unreachable" and res.n_chains == 0


def test_neighbour_gaps_match_direct_refinement(case, injector):
    problem, _, f_star, _ = case
    gaps = injector.neighbour_gaps_bp()
    k, n = problem.cardinality, problem.n
    assert gaps.shape == (k * (n - k),)
    assert gaps.min() >= -1e-6  # a certified optimum has no better neighbour
    expected = sorted(
        (refine_problem(problem, sorted((set(injector.start) - {i}) | {j})).objective - f_star) * BP
        for i in injector.start for j in range(n) if j not in injector.start
    )
    np.testing.assert_allclose(np.sort(gaps), expected, atol=1e-6)


def test_solver_feasibility_noise_in_the_certified_weights_is_accepted(case):
    """SCIP returns weights feasible to its feastol (1e-8): 0.4 + 5e-9, −2e-9, tiny dust."""
    problem, w_star, f_star, _ = case
    noisy = w_star.copy()
    noisy[np.argmax(noisy)] += 5e-9
    noisy[noisy == 0.0] = -2e-9
    inj = GapInjector(problem, noisy)
    assert inj.f_star == pytest.approx(f_star, abs=1e-12)
    assert np.all(inj.w_star >= 0.0)


def test_gap_cache_counts_refinements_not_lookups(injector):
    before = injector.n_evals
    s = tuple(range(5))
    first = injector.gap_bp(s)
    assert injector.n_evals == before + 1
    assert injector.gap_bp(tuple(reversed(s))) == first  # same subset, other order
    assert injector.n_evals == before + 1


# ---------------------------------------------------------------------------------------------
# construction checks
# ---------------------------------------------------------------------------------------------


def test_constructor_rejects_an_uncertified_optimum(case):
    problem, _, _, support = case
    worse = list(support)
    worse[0] = next(i for i in range(problem.n) if i not in support)  # one swap away
    w_bad = refine_problem(problem, worse).weights
    with pytest.raises(ValueError, match="optimal"):
        GapInjector(problem, w_bad)


def test_constructor_rejects_bad_inputs(case):
    problem, w_star, _, _ = case
    with pytest.raises(ValueError, match="cardinality"):
        GapInjector(PortfolioProblem(mu=problem.mu, sigma=problem.sigma,
                                     asset_names=problem.asset_names, risk_aversion=5.0,
                                     w_max=0.4), w_star)
    with pytest.raises(ValueError, match="shape"):
        GapInjector(problem, w_star[:-1])
    with pytest.raises(ValueError, match="budget"):
        GapInjector(problem, w_star * 0.9)
    small_k = PortfolioProblem(mu=problem.mu, sigma=problem.sigma, asset_names=problem.asset_names,
                               risk_aversion=5.0, w_max=0.4, cardinality=4)
    with pytest.raises(ValueError, match="support"):
        GapInjector(small_k, w_star)  # 5 held assets, K = 4
    with pytest.raises(ValueError, match="w_min"):
        GapInjector(PortfolioProblem(mu=problem.mu, sigma=problem.sigma,
                                     asset_names=problem.asset_names, risk_aversion=5.0,
                                     w_max=0.4, w_min=0.01, cardinality=5), w_star)


# ---------------------------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------------------------


def test_first_chain_uses_the_plain_seed_sequence_of_the_plan():
    ss = replica_seed(7, 3, 10, 5)
    assert isinstance(ss, np.random.SeedSequence)
    assert ss.entropy == (7, 3, 10, 5) and ss.spawn_key == ()
    plain = np.random.default_rng(np.random.SeedSequence((7, 3, 10, 5))).random(4)
    np.testing.assert_array_equal(np.random.default_rng(ss).random(4), plain)


def test_every_key_restart_and_operator_has_its_own_stream():
    draws = {
        (b, f, lv, r, op, rs): np.random.default_rng(
            replica_seed(b, f, lv, r, operator=op, restart=rs)).random()
        for b, f, lv, r in itertools.product((7, 8), (3, 4), (10, 25), (5, 6))
        for op, rs in (("swap", 0), ("swap", 1), ("swap", 2), ("perturb", 0))
    }
    assert len(set(draws.values())) == len(draws)
    # restart k is the k-th child spawned from the plan's sequence
    child = np.random.SeedSequence((7, 3, 10, 5)).spawn(2)[1]
    assert np.random.default_rng(replica_seed(7, 3, 10, 5, restart=2)).random() == \
        np.random.default_rng(child).random()
    with pytest.raises(ValueError, match="restart"):
        replica_seed(7, 3, 10, 5, operator="perturb", restart=1)
    with pytest.raises(ValueError, match="restart"):
        replica_seed(7, 3, 10, 5, restart=-1)
    with pytest.raises(ValueError, match="operator"):
        replica_seed(7, 3, 10, 5, operator="swapp")


# ---------------------------------------------------------------------------------------------
# Metropolis chain on an additive oracle: gap(S) = Σ_{i∈S} c_i − Σ_{i∈S0} c_i
# ---------------------------------------------------------------------------------------------


def _additive(c, start):
    c = np.asarray(c, dtype=float)
    base = float(sum(c[i] for i in start))
    return lambda subset: float(sum(c[i] for i in subset)) - base


def _chain(gap_fn, start, n, g, seed, *, max_steps=200, expired=lambda: False):
    return _swap_chain(gap_fn, start, n, g, np.random.default_rng(seed), max_steps=max_steps,
                       band_ratio=1.1, temperature_factor=0.25, expired=expired)


def test_chain_returns_the_state_at_first_passage_into_the_band():
    start = (0, 1, 2)
    gap_fn = _additive([0, 0, 0, 1, 2, 3], start)
    hits = 0
    for seed in range(30):
        out = _chain(gap_fn, start, 6, 2.0, seed)
        if not out.found:
            continue
        hits += 1
        assert 2.0 <= out.gap_bp <= 2.2 and out.gap_bp == gap_fn(out.subset)
        assert len(out.subset) == 3 and out.subset == tuple(sorted(out.subset))
        # first passage: the same chain stopped one step earlier has not entered the band
        earlier = _chain(gap_fn, start, 6, 2.0, seed, max_steps=out.steps - 1)
        assert not earlier.found and earlier.steps == out.steps - 1
        same = _chain(gap_fn, start, 6, 2.0, seed, max_steps=out.steps)
        assert same.found and same.subset == out.subset
    assert hits >= 20  # the band is easy to find: swap in asset 4 (gap 2)


def test_chain_stops_after_max_steps_without_a_band_state():
    start = (0, 1, 2)
    gap_fn = _additive([0, 0, 0, 0, 0, 0], start)  # every state has gap 0: the band is empty
    out = _chain(gap_fn, start, 6, 1.0, 5, max_steps=37)
    assert not out.found and out.steps == 37 and not out.timed_out
    assert out.accepted == 37  # equal energy: always accepted


def test_downhill_moves_are_always_accepted():
    start = (0, 1, 2)
    gap_fn = _additive([0, 0, 0, 0.5, 0.5, 0.5], start)  # every neighbour: gap 0.5, g = 1
    for seed in range(100):
        out = _chain(gap_fn, start, 6, 1.0, seed, max_steps=1)
        assert out.accepted == 1  # E: 1.0 -> 0.5


def test_uphill_moves_are_accepted_with_probability_exp_minus_dE_over_T():
    start = (0, 1, 2)
    # every neighbour has gap 2.5 with g = 1: E goes from 1.0 to 1.4; T = 0.25 -> exp(-1.6)
    gap_fn = _additive([0, 0, 0, 2.5, 2.5, 2.5], start)
    n = 4000
    acc = sum(_chain(gap_fn, start, 6, 1.0, s, max_steps=1).accepted for s in range(n))
    p = math.exp(-1.6)
    assert acc / n == pytest.approx(p, abs=4 * math.sqrt(p * (1 - p) / n))


def test_a_move_into_the_band_is_always_accepted_and_ends_the_chain():
    start = (0, 1, 2)
    gap_fn = _additive([0, 0, 0, 1.05, 1.05, 1.05], start)  # inside [1, 1.1]
    for seed in range(50):
        out = _chain(gap_fn, start, 6, 1.0, seed, max_steps=5)
        assert out.found and out.steps == 1 and out.accepted == 1


def test_infeasible_neighbours_are_never_accepted():
    start = (0, 1, 2)
    gap_fn = lambda s: 0.0 if tuple(s) == start else math.inf  # noqa: E731
    out = _chain(gap_fn, start, 6, 1.0, 3, max_steps=50)
    assert not out.found and out.accepted == 0


def test_chain_checks_the_clock_before_every_evaluation():
    start = (0, 1, 2)
    calls = []
    gap_fn = _additive([0, 0, 0, 0, 0, 0], start)

    def expired():
        calls.append(1)
        return len(calls) > 3

    out = _chain(gap_fn, start, 6, 1.0, 1, expired=expired)
    assert out.timed_out and not out.found and out.steps == 3  # proposals made before the stop


# ---------------------------------------------------------------------------------------------
# swap_level on the real instance
# ---------------------------------------------------------------------------------------------


def test_unreachable_level_skips_the_chains(injector):
    smallest = injector.min_positive_neighbour_gap_bp()
    assert smallest == pytest.approx(1.426, abs=1e-3)  # calibration instance, see _instance
    g = smallest / 1.1 * 0.99  # 1.1·g < the smallest positive neighbour gap
    res = injector.swap_level(g, base_seed=1, fold=0, level_key=1, n_replicas=4)
    assert res.status == "unreachable" and not res.reached
    assert res.replicas == () and res.n_chains == 0
    assert res.min_positive_neighbour_gap_bp == pytest.approx(smallest)


@pytest.mark.parametrize("g", [2.8, 5.0, 10.0, 25.0, 50.0])
def test_reached_level_has_all_replicas_inside_the_band(case, injector, g):
    problem, _, f_star, _ = case
    res = injector.swap_level(g, base_seed=11, fold=2, level_key=int(g), n_replicas=6)
    assert res.status == "reached" and res.reached
    assert len(res.replicas) == res.n_requested == 6
    for rep in res.replicas:
        assert g <= rep.gap_bp <= 1.1 * g
        assert len(rep.subset) == problem.cardinality == len(set(rep.subset))
        assert rep.subset != injector.start
        assert 1 <= rep.steps <= 200 and 0 <= rep.restarts <= 10
        # the replica is the shared refinement of its subset, and its gap is measured against f*
        ref = refine_problem(problem, rep.subset)
        np.testing.assert_allclose(rep.weights, ref.weights, atol=1e-12)
        assert rep.gap_bp == pytest.approx((ref.objective - f_star) * BP, abs=1e-9)
        assert rep.weights.sum() == pytest.approx(1.0) and rep.weights.max() <= problem.w_max + 1e-9
    assert res.n_evals >= 0 and res.elapsed_s >= 0.0


def test_replicas_are_reproducible_from_the_seed_tuple(case):
    problem, w_star, _, _ = case

    def run(base, fold):
        res = GapInjector(problem, w_star).swap_level(10.0, base_seed=base, fold=fold,
                                                      level_key=10, n_replicas=6)
        assert res.reached
        return [(r.subset, r.steps, r.restarts) for r in res.replicas]

    first = run(3, 4)
    assert run(3, 4) == first  # same (base, fold, level): the same replicas
    assert run(3, 5) != first  # another fold
    assert run(4, 4) != first  # another base seed


def test_replicas_come_from_independent_chains(injector):
    res = injector.swap_level(10.0, base_seed=21, fold=0, level_key=10, n_replicas=12)
    assert res.reached
    assert len({r.subset for r in res.replicas}) > 1  # not one chain copied 12 times


def test_distinct_subsets_are_reported_because_a_narrow_band_repeats_one_state(case):
    problem, w_star, _, _ = case
    # [5, 5.5] holds one subset of the 2002 (a one-swap neighbour, gap 5.02): every replica is it
    narrow = GapInjector(problem, w_star).swap_level(5.0, base_seed=1, fold=0, n_replicas=6)
    assert narrow.reached and narrow.n_distinct_subsets == 1
    wide = GapInjector(problem, w_star).swap_level(50.0, base_seed=1, fold=0, n_replicas=12)
    assert wide.reached and 1 < wide.n_distinct_subsets <= 12
    assert wide.n_distinct_subsets == len({r.subset for r in wide.replicas})


def test_levels_follow_the_band_populations_found_by_enumeration(enumeration, case):
    """Ground truth on all 2002 subsets: an empty band is never reached (replicas are real
    states), and the bands that hold several subsets are reached by the chains."""
    problem, objectives = enumeration
    f_star = case[2]
    gaps = np.array([(v - f_star) * BP for v in objectives.values()])
    population = {g: int(((gaps >= g) & (gaps <= 1.1 * g)).sum()) for g in (1, 2, 10, 25, 50, 100)}
    assert population[1] == population[2] == 0 and min(population[g] for g in (10, 25, 50, 100)) >= 4
    for g, count in population.items():
        res = GapInjector(problem, case[1]).swap_level(float(g), base_seed=1, fold=0, level_key=g,
                                                       n_replicas=4)
        assert res.reached == (count > 0), (g, count, res.status)
    # g = 1 is screened out by the neighbours (1.1 < 1.426), g = 2 is searched and not found
    assert GapInjector(problem, case[1]).swap_level(1.0, base_seed=1, fold=0).status == "unreachable"
    assert GapInjector(problem, case[1]).swap_level(2.0, base_seed=1, fold=0).status == "failed"


def test_level_fails_when_a_replica_is_not_found_within_the_restarts(injector):
    # no one-swap neighbour lies in [40, 44] and a chain of one step cannot go further
    res = injector.swap_level(40.0, base_seed=1, fold=0, level_key=40, n_replicas=3,
                              max_steps=1, max_restarts=2)
    assert res.status == "failed" and not res.reached
    assert res.n_chains == 3  # first chain + 2 restarts of replica 0; the level stops there
    assert len(res.replicas) < res.n_requested


def test_restarts_are_counted_per_replica(injector):
    res = injector.swap_level(10.0, base_seed=2, fold=1, level_key=10, n_replicas=8,
                              max_steps=3, max_restarts=10)
    assert res.n_chains == sum(r.restarts + 1 for r in res.replicas) + (0 if res.reached else 1 + 10)


def test_time_limit_makes_the_level_unreached(injector):
    ticks = itertools.count()
    res = injector.swap_level(10.0, base_seed=1, fold=0, level_key=10, n_replicas=30,
                              time_limit_s=5.0, clock=lambda: float(next(ticks)))
    assert res.status == "timeout" and not res.reached
    assert len(res.replicas) < 30


def test_non_integer_level_needs_an_explicit_seed_key(injector):
    with pytest.raises(ValueError, match="level_key"):
        injector.swap_level(2.5, base_seed=1, fold=0, n_replicas=1)
    assert injector.swap_level(5.0, base_seed=1, fold=0, n_replicas=1).reached  # integral: g itself


def test_swap_level_rejects_bad_arguments(injector):
    with pytest.raises(ValueError, match="gap_bp"):
        injector.swap_level(0.0, base_seed=1, fold=0, level_key=0)
    with pytest.raises(ValueError, match="n_replicas"):
        injector.swap_level(5.0, base_seed=1, fold=0, n_replicas=0)


# ---------------------------------------------------------------------------------------------
# GMV arm
# ---------------------------------------------------------------------------------------------


def test_gmv_gap_is_half_lambda_times_the_variance_difference(case):
    problem, _, _, _ = case
    gmv = gmv_problem(problem)
    assert np.all(gmv.mu == 0.0) and gmv.cardinality == problem.cardinality
    assert gmv.w_max == problem.w_max and gmv.risk_aversion == problem.risk_aversion
    _, _, w_star = _certified(gmv)
    inj = GapInjector(gmv, w_star)
    var_star = w_star @ gmv.sigma @ w_star
    res = inj.swap_level(25.0, base_seed=1, fold=0, level_key=25, n_replicas=4)
    assert res.reached  # 18 of the 2002 subsets lie in [25, 27.5] on this arm (the MV arm: 8)
    for rep in res.replicas:
        dvar = rep.weights @ gmv.sigma @ rep.weights - var_star
        assert rep.gap_bp == pytest.approx(0.5 * gmv.risk_aversion * dvar * BP, abs=1e-8)
        assert 25.0 <= rep.gap_bp <= 27.5


# ---------------------------------------------------------------------------------------------
# operator (а): w(t) = (1 − t)·w* + t·v on the optimal support (descriptive)
# ---------------------------------------------------------------------------------------------


def test_directions_are_dirichlet_one_truncated_by_the_box():
    # s = 3, u = 0.5: the box cuts the three corners off the simplex, what is left is the
    # medial triangle (a quarter of the area); the marginal of v1 there has CDF 4·x² on [0, 0.5]
    rng = np.random.default_rng(0)
    taken, raw_draws = [], 0
    for direction, drawn in _accepted_directions(rng, 3, 0.5, max_draws=100_000):
        taken.append(direction)
        raw_draws = drawn
        if len(taken) == 4000:
            break
    v = np.array(taken)
    assert v.max() <= 0.5 and np.allclose(v.sum(axis=1), 1.0) and np.all(v >= 0.0)
    assert np.mean(v[:, 0] <= 0.25) == pytest.approx(0.25, abs=0.02)  # 4·0.25²
    assert np.mean(v[:, 0] <= 0.4) == pytest.approx(0.64, abs=0.02)  # 4·0.4²
    np.testing.assert_allclose(v.mean(axis=0), 1 / 3, atol=0.01)
    assert raw_draws == pytest.approx(4 * 4000, rel=0.05)  # acceptance probability 1/4


def test_direction_stream_stops_at_the_draw_budget():
    rng = np.random.default_rng(1)
    got = list(_accepted_directions(rng, 6, 0.17, max_draws=500))  # 6·0.17 = 1.02: almost a point
    assert len(got) <= 2 and all(d <= 500 for _, d in got)


def test_bisection_finds_a_point_inside_the_band():
    gap = lambda t: 200.0 * t * t  # noqa: E731  (bp; monotone like the real gap)
    for g in (0.5, 5.0, 50.0, 150.0):
        t = _bisect_band(gap, g, 1.1)
        assert 0.0 < t <= 1.0 and g <= gap(t) <= 1.1 * g
    assert _bisect_band(gap, 250.0, 1.1) is None  # even t = 1 gives 200 < 250
    assert _bisect_band(gap, 200.0, 1.1) == 1.0  # the end point itself is in the band


def test_perturbation_is_not_built_when_the_face_is_a_single_point():
    problem = _instance(14, 5, 1002, u=0.2)  # K·u = 1: the only feasible weights are 0.2 each
    _, _, w_star = _certified(problem)
    inj = GapInjector(problem, w_star)
    res = inj.perturb_level(5.0, base_seed=1, fold=0, n_replicas=3)
    assert not res.built and res.status == "not_built" and not res.reached
    assert res.replicas == () and "supp" in res.reason


@pytest.mark.parametrize("g", [1.0, 5.0, 25.0, 100.0])
def test_perturbed_replicas_are_feasible_convex_combinations_in_the_band(case, injector, g):
    problem, _, f_star, support = case
    res = injector.perturb_level(g, base_seed=5, fold=1, level_key=int(g), n_replicas=6)
    assert res.built and res.reached and len(res.replicas) == 6
    off_support = np.setdiff1d(np.arange(problem.n), support)
    for rep in res.replicas:
        w, v = rep.weights, rep.direction
        assert 0.0 < rep.t <= 1.0
        np.testing.assert_allclose(w, (1 - rep.t) * injector.w_star + rep.t * v, atol=1e-12)
        assert w.sum() == pytest.approx(1.0) and w.min() >= 0.0
        assert w.max() <= problem.w_max + 1e-12 and np.count_nonzero(w) <= problem.cardinality
        assert np.all(w[off_support] == 0.0) and np.all(v[off_support] == 0.0)
        assert v.max() <= problem.w_max and v.sum() == pytest.approx(1.0)
        # the gap is the objective excess of the perturbed weights themselves: no refinement
        assert rep.gap_bp == pytest.approx((problem.objective_value(w) - f_star) * BP, abs=1e-9)
        assert g <= rep.gap_bp <= 1.1 * g
        assert rep.n_draws >= 1


def test_perturbation_gap_cannot_reach_a_level_beyond_the_face(case, injector):
    # the largest gap on the face is far below 1e6 bp: no direction reaches it
    res = injector.perturb_level(1e6, base_seed=1, fold=0, level_key=1, n_replicas=2, max_draws=2000)
    assert res.built and res.status == "failed" and not res.reached
    assert res.replicas == ()


def test_perturbation_is_reproducible_and_streams_differ_from_the_swap_operator(case):
    problem, w_star, _, _ = case

    def run(base, fold):
        res = GapInjector(problem, w_star).perturb_level(25.0, base_seed=base, fold=fold,
                                                         level_key=25, n_replicas=4)
        return [(r.t, tuple(r.weights)) for r in res.replicas]

    first = run(3, 4)
    assert run(3, 4) == first and run(3, 5) != first and run(4, 4) != first
    assert len({t for t, _ in first}) == 4  # independent replicas


def test_perturbation_in_the_gmv_arm_uses_the_variance_gap(case):
    problem, _, _, _ = case
    gmv = gmv_problem(problem)
    _, _, w_star = _certified(gmv)
    inj = GapInjector(gmv, w_star)
    res = inj.perturb_level(10.0, base_seed=1, fold=0, level_key=10, n_replicas=3)
    var_star = inj.w_star @ gmv.sigma @ inj.w_star
    for rep in res.replicas:
        dvar = rep.weights @ gmv.sigma @ rep.weights - var_star
        assert rep.gap_bp == pytest.approx(0.5 * gmv.risk_aversion * dvar * BP, abs=1e-8)
