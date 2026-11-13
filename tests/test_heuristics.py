"""Tests for the QUBO-pipeline heuristics (WP-D).

Covers the shared second stage (:func:`cif.heuristics.pipeline.refine_candidates`),
the swap-SA sampler on the cardinality-preserving manifold, the explicit
penalty multiplier of the selection BQM (R8) and the neal wrapper that now
delegates to the shared pipeline.
"""

from __future__ import annotations

import time
from collections import Counter
from itertools import combinations

import cvxpy as cp
import dimod
import numpy as np
import pytest

from cif.classical.continuous import solve_continuous_mvo
from cif.classical.refine import refine_problem
from cif.heuristics import pipeline
from cif.heuristics.pipeline import (
    iter_sampleset_subsets,
    legacy_turnover_cost,
    refine_candidates,
)
from cif.heuristics.swap_sa import (
    SwapSurrogate,
    calibrate_hot_temperature,
    sample_swap_sa,
    solve_with_swap_sa,
)
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem
from cif.quantum import neal_sampler
from cif.quantum.neal_sampler import solve_with_neal_selection, solve_with_tabu_selection
from cif.qubo.builder import build_selection_bqm, selection_penalty_scale


def _make_problem(n: int, K: int, seed: int, risk_aversion: float = 2.0) -> PortfolioProblem:
    """Legacy synthetic generator (identical to scripts/experiment_a_scalability.py)."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=risk_aversion,
        cardinality=K,
    )


def _toy_problem(w_max: float = 1.0) -> PortfolioProblem:
    """8 assets where {0, 1, 2} is the best triple and every weight is interior.

    At risk aversion 10 the refined weights of {0, 1, 2} are about
    (0.46, 0.33, 0.21) and those of the runner-up {0, 1, 3} about
    (0.50, 0.38, 0.12), so the two triples differ in objective and in trades.
    """
    mu = np.array([0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03])
    sigma = np.eye(8) * 0.04 + 0.005
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"T{i}" for i in range(8)),
        risk_aversion=10.0,
        w_max=w_max,
        cardinality=3,
    )


# ---------------------------------------------------------------------------
# build_selection_bqm: explicit penalty multiplier (R8)
# ---------------------------------------------------------------------------


def test_selection_penalty_scale_matches_script_convention():
    problem = _make_problem(20, 5, seed=1)
    expected = float(np.abs(problem.mu).max()) / 5 + problem.risk_aversion * float(
        np.abs(problem.sigma).max()
    ) / 25
    assert selection_penalty_scale(problem, 5) == pytest.approx(expected, rel=1e-14)


def test_selection_bqm_default_penalty_is_100x_scale():
    """R8: the library default must equal the configuration used in all experiments."""
    problem = _make_problem(20, 5, seed=1)
    scale = selection_penalty_scale(problem, 5)
    default = build_selection_bqm(problem, cardinality=5)
    explicit = build_selection_bqm(problem, cardinality=5, cardinality_penalty=100.0 * scale)
    assert default == explicit


def test_selection_bqm_penalty_multiplier_scales_the_penalty():
    problem = _make_problem(20, 5, seed=1)
    scale = selection_penalty_scale(problem, 5)
    for mult in (1.0, 3.0, 30.0):
        via_mult = build_selection_bqm(problem, cardinality=5, penalty_multiplier=mult)
        via_penalty = build_selection_bqm(problem, cardinality=5, cardinality_penalty=mult * scale)
        assert via_mult == via_penalty


def test_selection_bqm_explicit_penalty_has_priority_over_multiplier():
    problem = _make_problem(20, 5, seed=1)
    only_explicit = build_selection_bqm(problem, cardinality=5, cardinality_penalty=0.37)
    both = build_selection_bqm(
        problem, cardinality=5, cardinality_penalty=0.37, penalty_multiplier=7.0
    )
    assert both == only_explicit


def test_selection_bqm_rejects_non_positive_multiplier():
    problem = _make_problem(10, 3, seed=1)
    with pytest.raises(ValueError):
        build_selection_bqm(problem, cardinality=3, penalty_multiplier=0.0)


# ---------------------------------------------------------------------------
# swap-SA surrogate: energy and incremental updates
# ---------------------------------------------------------------------------


def test_surrogate_energy_equals_bqm_energy_on_feasible_states():
    """E(S) is the binary-inclusion BQM energy on |S| = K, whatever the penalty."""
    n, K = 14, 5
    problem = _make_problem(n, K, seed=4)
    bqm = build_selection_bqm(problem, cardinality=K, penalty_multiplier=17.0)
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    rng = np.random.default_rng(0)
    for _ in range(20):
        subset = tuple(sorted(rng.choice(n, size=K, replace=False).tolist()))
        sample = {i: int(i in subset) for i in range(n)}
        assert sur.energy(subset) == pytest.approx(bqm.energy(sample), abs=1e-12)


def test_incremental_delta_matches_recomputation_after_random_swaps():
    """Property: dE and the gain vector stay exact over a long chain of swaps."""
    n, K, reads = 25, 6, 8
    problem = _make_problem(n, K, seed=5)
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    rng = np.random.default_rng(0)
    state = sur.random_state(reads, rng)
    rows = np.arange(reads)
    for _ in range(300):
        pos_in = rng.integers(0, K, size=reads)
        pos_out = rng.integers(0, n - K, size=reads)
        accept = rng.random(reads) < 0.6
        before = np.array([sur.energy(r) for r in state.inside])
        delta = sur.swap_delta(state, pos_in, pos_out)
        leaving = state.inside[rows, pos_in].copy()
        entering = state.outside[rows, pos_out].copy()
        sur.apply_swap(state, pos_in, pos_out, accept, delta)
        after = np.array([sur.energy(r) for r in state.inside])
        np.testing.assert_allclose(after - before, np.where(accept, delta, 0.0), atol=1e-12)
        # the swap really exchanged the two assets for the accepted reads
        assert np.all(state.inside[rows, pos_in][accept] == entering[accept])
        assert np.all(state.outside[rows, pos_out][accept] == leaving[accept])
    recomputed_gain = np.stack([problem.sigma[r].sum(axis=0) for r in state.inside])
    np.testing.assert_allclose(state.gain, recomputed_gain, atol=1e-10)
    np.testing.assert_allclose(state.energy, [sur.energy(r) for r in state.inside], atol=1e-10)
    # inside and outside always partition {0, ..., n-1}
    for r in range(reads):
        union = np.concatenate([state.inside[r], state.outside[r]])
        assert sorted(union.tolist()) == list(range(n))


# ---------------------------------------------------------------------------
# swap-SA sampler
# ---------------------------------------------------------------------------


def test_swap_sa_keeps_cardinality_in_every_read():
    n, K, reads = 30, 7, 25
    problem = _make_problem(n, K, seed=3)
    res = sample_swap_sa(problem, K, num_reads=reads, num_sweeps=15, seed=11)
    assert res.read_subsets.shape == (reads, K)
    for row in res.read_subsets:
        assert len(set(row.tolist())) == K
        assert row.min() >= 0 and row.max() < n
    assert all(len(s) == K for s in res.subsets)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_swap_sa_finds_surrogate_minimum_on_n12(seed):
    n, K = 12, 4
    problem = _make_problem(n, K, seed=100 + seed)
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    energies = {c: sur.energy(c) for c in combinations(range(n), K)}
    best = min(energies, key=energies.get)
    res = sample_swap_sa(problem, K, num_reads=30, num_sweeps=100, seed=seed)
    assert res.subsets[0] == best
    assert res.energies[0] == pytest.approx(energies[best], abs=1e-12)


def test_swap_sa_returns_unique_subsets_by_ascending_energy():
    n, K = 20, 5
    problem = _make_problem(n, K, seed=8)
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    res = sample_swap_sa(problem, K, num_reads=40, num_sweeps=5, seed=2)
    assert len(set(res.subsets)) == len(res.subsets)
    assert np.all(np.diff(res.energies) >= 0)
    for subset, energy in zip(res.subsets, res.energies, strict=True):
        assert subset == tuple(sorted(subset))
        assert energy == pytest.approx(sur.energy(subset), abs=1e-10)


def test_swap_sa_samples_the_boltzmann_distribution_at_constant_temperature():
    """Detailed balance: at fixed T the chain's end states follow exp(-E/T)."""
    n, K, reads = 8, 3, 10000
    problem = _make_problem(n, K, seed=21)
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    subsets = list(combinations(range(n), K))
    energies = np.array([sur.energy(s) for s in subsets])
    temperature = float(energies.std())
    weights = np.exp(-(energies - energies.min()) / temperature)
    expected = weights / weights.sum()
    res = sample_swap_sa(
        problem, K, num_reads=reads, num_sweeps=10, seed=3, t_hot=temperature, t_cold=temperature
    )
    counts = Counter(map(tuple, res.read_subsets.tolist()))
    observed = np.array([counts.get(s, 0) for s in subsets]) / reads
    # total variation distance; pure multinomial noise at 10^4 reads is ~0.026 (99th pct ~0.033)
    assert 0.5 * np.abs(observed - expected).sum() < 0.045


def test_swap_sa_is_reproducible_for_a_seed():
    problem = _make_problem(18, 4, seed=9)
    a = sample_swap_sa(problem, 4, num_reads=10, num_sweeps=10, seed=5)
    b = sample_swap_sa(problem, 4, num_reads=10, num_sweeps=10, seed=5)
    assert a.subsets == b.subsets
    np.testing.assert_array_equal(a.energies, b.energies)
    np.testing.assert_array_equal(a.read_subsets, b.read_subsets)


@pytest.mark.parametrize("K", [1, 9, 10])
def test_swap_sa_handles_extreme_cardinalities(K):
    n = 10
    problem = _make_problem(n, K, seed=12)
    res = sample_swap_sa(problem, K, num_reads=6, num_sweeps=8, seed=1)
    assert all(len(s) == K for s in res.subsets)
    if n == K:
        assert res.subsets == [tuple(range(n))]


def test_swap_sa_automatic_temperatures_and_overrides():
    problem = _make_problem(20, 5, seed=2)
    auto = sample_swap_sa(problem, 5, num_reads=8, num_sweeps=10, seed=1)
    assert auto.t_hot > auto.t_cold > 0.0
    assert 0.0 <= auto.frozen_fraction <= 1.0
    assert 0.0 < auto.accept_rate < 1.0
    manual = sample_swap_sa(
        problem, 5, num_reads=8, num_sweeps=10, seed=1, t_hot=0.5, t_cold=0.01
    )
    assert manual.t_hot == 0.5 and manual.t_cold == 0.01


def test_hot_temperature_gives_target_acceptance_on_fresh_random_swaps():
    """T0 is calibrated on random swaps so that ~80 % of them are accepted."""
    n, K = 40, 10
    problem = _make_problem(n, K, seed=6)
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    t_hot = calibrate_hot_temperature(sur, np.random.default_rng(1), accept_target=0.8)
    assert t_hot > 0.0
    # an independent sample of random swaps from random states
    rng = np.random.default_rng(99)
    state = sur.random_state(256, rng)
    deltas = np.concatenate(
        [
            sur.swap_delta(state, rng.integers(0, K, size=256), rng.integers(0, n - K, size=256))
            for _ in range(8)
        ]
    )
    accept = np.mean(np.minimum(1.0, np.exp(-np.maximum(deltas, 0.0) / t_hot)))
    assert accept == pytest.approx(0.8, abs=0.05)
    # the answer lives on the scale of the mean |dE| of random swaps ...
    scale = float(np.abs(deltas).mean())
    assert 0.2 * scale < t_hot < 20.0 * scale
    # ... also for a target below the share of downhill swaps (never ~0)
    t_low = calibrate_hot_temperature(sur, np.random.default_rng(1), accept_target=0.2)
    assert 0.2 * scale < t_low < 20.0 * scale
    with pytest.raises(ValueError):
        calibrate_hot_temperature(sur, np.random.default_rng(1), accept_target=1.0)


# ---------------------------------------------------------------------------
# refine_candidates: shared second stage
# ---------------------------------------------------------------------------


def _count_refinements(monkeypatch) -> list[int]:
    """Sizes of the supports refined through the shared ``refine_problem`` (audit RF-1)."""
    calls: list[int] = []
    real = pipeline.refine_problem

    def counting(problem, subset, *args, **kwargs):
        calls.append(len(subset))
        return real(problem, subset, *args, **kwargs)

    monkeypatch.setattr(pipeline, "refine_problem", counting)
    return calls


def test_refine_candidates_m_caps_refinements_and_stream_consumption(monkeypatch):
    problem = _toy_problem()
    calls = _count_refinements(monkeypatch)
    triples = list(combinations(range(8), 3))
    stream = [
        triples[0],
        triples[0],  # duplicate
        (0, 1),  # wrong size
        triples[1],
        triples[2],
        triples[3],
        triples[4],
    ]
    consumed: list[int] = []

    def lazy():
        for k, item in enumerate(stream):
            consumed.append(k)
            yield item

    best, stats = refine_candidates(problem, lazy(), K=3, m=3)
    assert best is not None
    assert len(calls) == 3
    assert stats["m"] == 3
    assert stats["n_candidates"] == 3
    assert stats["n_evaluated"] == 3
    assert stats["refine_wall_s"] >= 0.0
    assert consumed == [0, 1, 2, 3, 4]  # stops reading once the pool is full


def test_refine_candidates_uses_arrival_order_not_quality():
    """With m = 1 the first K-subset wins, even if later ones are better."""
    problem = _toy_problem()
    poor, good = (5, 6, 7), (0, 1, 2)
    best, _ = refine_candidates(problem, [poor, good], K=3, m=1)
    assert best.solver_meta["subset"] == list(poor)


def test_refine_candidates_warm_start_joins_the_pool_and_can_win():
    problem = _toy_problem()
    others = [(3, 4, 5), (4, 5, 6), (5, 6, 7), (3, 4, 6), (3, 5, 7)]
    without, stats0 = refine_candidates(problem, others, K=3, m=5)
    assert without.solver_meta["subset"] != [0, 1, 2]
    assert stats0["n_candidates"] == 5
    with_ws, stats1 = refine_candidates(
        problem, others, K=3, m=5, warm_start_subset=[2, 0, 1]
    )
    assert with_ws.solver_meta["subset"] == [0, 1, 2]
    assert stats1["n_candidates"] == 6
    assert with_ws.objective < without.objective


def test_refine_candidates_warm_start_size_rules():
    """Warm starts smaller than K are candidates (current holdings may have < K names, audit A1);
    larger than K, empty, or already pooled are not added."""
    problem = _toy_problem()
    pool = [(0, 1, 2), (3, 4, 5)]
    _, smaller = refine_candidates(problem, pool, K=3, m=5, warm_start_subset=[0, 1])
    assert smaller["n_candidates"] == 3 and smaller["warm_start_in_pool"]
    _, larger = refine_candidates(problem, pool, K=3, m=5, warm_start_subset=[0, 1, 2, 3])
    assert larger["n_candidates"] == 2 and not larger["warm_start_in_pool"]
    _, duplicate = refine_candidates(problem, pool, K=3, m=5, warm_start_subset=[2, 1, 0])
    assert duplicate["n_candidates"] == 2 and duplicate["warm_start_in_pool"]


def test_refine_candidates_turnover_cost_flips_the_choice_at_the_threshold():
    """Ranking is obj + c * sum|w - prev| (full L1, no 1/2 factor)."""
    problem = _toy_problem()
    good, worse = (0, 1, 2), (0, 1, 3)
    sol_good, _ = refine_candidates(problem, [good], K=3, m=1)
    sol_worse, _ = refine_candidates(problem, [worse], K=3, m=1)
    assert sol_good.objective < sol_worse.objective
    prev = sol_worse.weights.copy()
    trade_good = float(np.abs(sol_good.weights - prev).sum())
    assert trade_good > 0.0
    c_star = (sol_worse.objective - sol_good.objective) / trade_good

    def pick(cost: float):
        best, _ = refine_candidates(
            problem, [good, worse], K=3, m=2, prev_weights=prev, turnover_cost=cost
        )
        return best.solver_meta["subset"]

    assert pick(0.0) == list(good)
    assert pick(0.9 * c_star) == list(good)
    assert pick(1.1 * c_star) == list(worse)


def test_refine_candidates_reports_true_objective_not_the_ranking_one():
    problem = _toy_problem()
    sol, _ = refine_candidates(problem, [(0, 1, 2)], K=3, m=1)
    prev = np.full(8, 1.0 / 8)
    ranked, _ = refine_candidates(
        problem, [(0, 1, 2)], K=3, m=1, prev_weights=prev, turnover_cost=0.01
    )
    assert ranked.objective == pytest.approx(sol.objective, abs=1e-12)
    trade = float(np.abs(ranked.weights - prev).sum())
    assert ranked.solver_meta["trade_l1"] == pytest.approx(trade, abs=1e-12)
    assert ranked.solver_meta["ranking_objective"] == pytest.approx(
        sol.objective + 0.01 * trade, abs=1e-12
    )


def test_refine_candidates_weights_respect_problem_bounds():
    problem = _toy_problem(w_max=0.4)
    best, _ = refine_candidates(problem, list(combinations(range(8), 3)), K=3, m=56)
    assert best.feasible
    assert best.weights.min() >= -1e-9
    assert best.weights.max() <= 0.4 + 1e-6
    assert best.weights.sum() == pytest.approx(1.0, abs=1e-6)
    support = set(np.flatnonzero(best.weights > 1e-9).tolist())
    assert support <= set(best.solver_meta["subset"])


def test_refine_candidates_returns_none_without_valid_candidates():
    problem = _toy_problem()
    best, stats = refine_candidates(problem, [(0, 1), (0, 1, 2, 3)], K=3, m=5)
    assert best is None
    assert stats["n_candidates"] == 0 and stats["n_evaluated"] == 0


def test_refine_candidates_without_convex_refinement_uses_equal_weights():
    problem = _toy_problem()
    best, stats = refine_candidates(
        problem, [(0, 1, 2)], K=3, m=1, refine_continuous=False
    )
    np.testing.assert_allclose(best.weights[[0, 1, 2]], 1.0 / 3.0)
    assert stats["n_evaluated"] == 1


def test_legacy_turnover_cost_reproduces_the_old_ranking_formula():
    """Legacy: obj + bps * 0.5 * sum|dw| / 1e4  ==  obj + (bps / 2e4) * sum|dw|."""
    rng = np.random.default_rng(0)
    w, prev = rng.dirichlet(np.ones(12)), rng.dirichlet(np.ones(12))
    bps = 10.0
    old = bps * (0.5 * float(np.abs(w - prev).sum())) / 10000.0
    new = legacy_turnover_cost(bps) * float(np.abs(w - prev).sum())
    assert new == pytest.approx(old, rel=1e-14)


# ---------------------------------------------------------------------------
# refine_candidates on the shared refinement (audit RF-1; ANALYSIS_PLAN §4: one refine for all)
# ---------------------------------------------------------------------------


def _forbid_cvxpy(monkeypatch) -> None:
    def no_cvxpy(self, *args, **kwargs):
        raise AssertionError("CVXPY called")

    monkeypatch.setattr(cp.Problem, "solve", no_cvxpy)


def test_refine_candidates_pure_path_never_touches_cvxpy(monkeypatch):
    problem = _toy_problem()
    _forbid_cvxpy(monkeypatch)
    triples = list(combinations(range(8), 3))
    best, stats = refine_candidates(problem, triples[:10], K=3, m=10)
    assert best is not None and stats["n_evaluated"] == 10
    # a turnover cost that only ranks the candidates leaves the refinement pure
    prev = np.full(8, 1.0 / 8)
    best, stats = refine_candidates(
        problem, triples[:5], K=3, m=5, prev_weights=prev, turnover_cost=0.01
    )
    assert best is not None and stats["n_evaluated"] == 5


def test_refine_candidates_matches_the_cvxpy_reference_on_every_subset():
    problem = _toy_problem(w_max=0.5)
    for subset in combinations(range(8), 3):
        idx = np.asarray(subset)
        sub = PortfolioProblem(
            mu=problem.mu[idx],
            sigma=problem.sigma[np.ix_(idx, idx)],
            asset_names=tuple(problem.asset_names[i] for i in subset),
            risk_aversion=problem.risk_aversion,
            w_max=problem.w_max,
        )
        reference = solve_continuous_mvo(sub)
        best, _ = refine_candidates(problem, [subset], K=3, m=1)
        assert best.objective == pytest.approx(reference.objective, rel=1e-6, abs=1e-9)
        # CVXPY's default tolerance (1e-8) is the limit here: its weights are off by ~4e-5
        np.testing.assert_allclose(best.weights[idx], reference.weights, atol=1e-3)


def test_refine_candidates_with_turnover_in_refinement_keeps_the_cvxpy_path(monkeypatch):
    problem = _toy_problem()
    prev = np.zeros(8)
    prev[[0, 1, 2]] = [0.5, 0.3, 0.2]
    pair = [(0, 1, 2), (0, 1, 3)]
    penalties: list[float] = []
    real = pipeline.solve_continuous_mvo

    def spy(sub_problem, *args, **kwargs):
        penalties.append(kwargs.get("turnover_penalty"))
        return real(sub_problem, *args, **kwargs)

    monkeypatch.setattr(pipeline, "solve_continuous_mvo", spy)
    shared = _count_refinements(monkeypatch)
    # the L1 penalty inside the refinement is the one thing the shared refine does not carry
    refine_candidates(
        problem, pair, K=3, m=2, prev_weights=prev, turnover_cost=0.01, turnover_in_refinement=True
    )
    assert penalties == [0.01, 0.01] and shared == []
    # the same inputs without the flag, or with a zero cost, refine purely
    penalties.clear()
    refine_candidates(problem, pair, K=3, m=2, prev_weights=prev, turnover_cost=0.01)
    refine_candidates(
        problem, pair, K=3, m=2, prev_weights=prev, turnover_cost=0.0, turnover_in_refinement=True
    )
    assert penalties == [] and shared == [3, 3, 3, 3]


def test_refine_candidates_card_mode_follows_the_minimum_purchase_by_default():
    base = _toy_problem()
    problem = PortfolioProblem(
        mu=base.mu,
        sigma=base.sigma,
        asset_names=base.asset_names,
        risk_aversion=base.risk_aversion,
        w_min=0.05,
        w_max=0.6,
        cardinality=3,
    )
    # w_min > 0: every asset of the subset is held at least w_min, as this stage always did
    best, stats = refine_candidates(problem, [(5, 6, 7)], K=3, m=1)
    assert stats["card_mode"] == "eq"
    assert np.all(best.weights[[5, 6, 7]] >= 0.05 - 1e-9)
    assert best.objective == pytest.approx(
        refine_problem(problem, (5, 6, 7), card_mode="eq").objective, abs=1e-14
    )
    # an explicit 'le' with a minimum purchase is the combinatorial case refine_problem rejects,
    # also on the CVXPY path with the penalty inside the refinement
    with pytest.raises(ValueError, match="w_min"):
        refine_candidates(problem, [(5, 6, 7)], K=3, m=1, card_mode="le")
    prev = np.full(8, 1.0 / 8)
    with pytest.raises(ValueError, match="w_min"):
        refine_candidates(
            problem, [(5, 6, 7)], K=3, m=1, card_mode="le",
            prev_weights=prev, turnover_cost=0.01, turnover_in_refinement=True,
        )
    with pytest.raises(ValueError, match="card_mode"):
        refine_candidates(problem, [(5, 6, 7)], K=3, m=1, card_mode="ge", refine_continuous=False)
    # w_min = 0: 'le' by default, and 'eq' is the same refinement
    _, default = refine_candidates(base, [(0, 1, 2)], K=3, m=1)
    eq, stats_eq = refine_candidates(base, [(0, 1, 2)], K=3, m=1, card_mode="eq")
    le, _ = refine_candidates(base, [(0, 1, 2)], K=3, m=1, card_mode="le")
    assert default["card_mode"] == "le" and stats_eq["card_mode"] == "eq"
    assert eq.objective == pytest.approx(le.objective, abs=1e-12)


def test_refine_candidates_without_refinement_ignores_the_turnover_in_refinement_flag(monkeypatch):
    problem = _toy_problem()
    _forbid_cvxpy(monkeypatch)
    prev = np.full(8, 1.0 / 8)
    best, stats = refine_candidates(
        problem, [(0, 1, 2)], K=3, m=1, refine_continuous=False,
        prev_weights=prev, turnover_cost=0.01, turnover_in_refinement=True,
    )
    np.testing.assert_allclose(best.weights[[0, 1, 2]], 1.0 / 3.0)
    assert stats["n_evaluated"] == 1


def test_refine_candidates_skips_a_subset_whose_refinement_fails():
    problem = _toy_problem(w_max=0.3)  # 3 · 0.3 < 1: no 3-subset can reach the budget
    best, stats = refine_candidates(problem, [(0, 1, 2), (3, 4, 5)], K=3, m=2)
    assert best is None
    assert stats["n_candidates"] == 2 and stats["n_evaluated"] == 0


def test_the_pipeline_wrappers_refine_without_cvxpy(monkeypatch):
    problem = _make_problem(20, 5, seed=1)
    _forbid_cvxpy(monkeypatch)
    solutions = [
        solve_with_neal_selection(
            problem, cardinality=5, num_reads=20, num_sweeps=20, seed=10001, top_subsets=5
        ),
        solve_with_swap_sa(problem, 5, num_reads=10, num_sweeps=10, seed=10002, m=5),
        solve_with_tabu_selection(
            problem, cardinality=5, num_reads=10, seed=10003, top_subsets=5,
            timeout_ms=None, num_restarts=1,
        ),
    ]
    for sol in solutions:
        assert sol.feasible and len(sol.solver_meta["subset"]) == 5


# ---------------------------------------------------------------------------
# sample-set decoding helper
# ---------------------------------------------------------------------------


def _fake_sampleset() -> dimod.SampleSet:
    rows = np.array(
        [
            [1, 1, 0, 0, 0],  # energy 3.0
            [0, 1, 1, 0, 0],  # energy 1.0
            [1, 1, 0, 0, 0],  # energy 3.0 (duplicate subset)
            [0, 0, 1, 1, 0],  # energy 2.0
        ]
    )
    return dimod.SampleSet.from_samples(
        rows, energy=[3.0, 1.0, 3.0, 2.0], vartype="BINARY"
    )


def test_iter_sampleset_subsets_read_order_is_the_default_legacy_behaviour():
    ss = _fake_sampleset()
    assert list(iter_sampleset_subsets(ss, 5)) == [(0, 1), (1, 2), (0, 1), (2, 3)]


def test_iter_sampleset_subsets_energy_order():
    ss = _fake_sampleset()
    assert list(iter_sampleset_subsets(ss, 5, order="energy")) == [
        (1, 2),
        (2, 3),
        (0, 1),
        (0, 1),
    ]
    with pytest.raises(ValueError):
        list(iter_sampleset_subsets(ss, 5, order="bogus"))


# ---------------------------------------------------------------------------
# neal wrapper on the shared pipeline
# ---------------------------------------------------------------------------


def test_neal_selection_returns_cardinality_k_subset_with_pipeline_meta():
    problem = _make_problem(20, 5, seed=1)
    sol = solve_with_neal_selection(
        problem, cardinality=5, num_reads=40, num_sweeps=50, seed=3, top_subsets=7
    )
    assert sol.feasible
    assert len(sol.solver_meta["subset"]) == 5
    for key in ("sa_wall_s", "refine_wall_s", "n_candidates", "n_evaluated", "m"):
        assert key in sol.solver_meta
    assert sol.solver_meta["m"] == 7
    assert sol.solver_meta["n_evaluated"] <= 7


def test_neal_selection_legacy_bps_cost_equals_new_convention():
    """turnover_cost_bps = b is the same ranking as turnover_cost = b / 2e4."""
    problem = _make_problem(24, 4, seed=2)
    prev = np.zeros(problem.n)
    prev[[0, 1, 2, 3]] = 0.25
    common = dict(
        cardinality=4, num_reads=30, num_sweeps=30, seed=7, top_subsets=10, prev_weights=prev
    )
    legacy = solve_with_neal_selection(problem, turnover_cost_bps=2000.0, **common)
    new = solve_with_neal_selection(problem, turnover_cost=2000.0 / 20000.0, **common)
    assert legacy.solver_meta["subset"] == new.solver_meta["subset"]
    assert legacy.objective == new.objective


def test_neal_selection_rejects_both_turnover_conventions_at_once():
    problem = _make_problem(12, 3, seed=2)
    with pytest.raises(ValueError):
        solve_with_neal_selection(
            problem,
            cardinality=3,
            num_reads=5,
            num_sweeps=5,
            turnover_cost_bps=10.0,
            turnover_cost=0.001,
        )


def test_neal_selection_warm_start_wins_under_huge_legacy_turnover_cost():
    problem = _make_problem(30, 4, seed=3)
    warm = [0, 1, 2, 3]
    prev = np.zeros(problem.n)
    prev[warm] = 0.25
    sol = solve_with_neal_selection(
        problem,
        cardinality=4,
        num_reads=40,
        num_sweeps=40,
        seed=5,
        top_subsets=20,
        warm_start_subset=warm,
        prev_weights=prev,
        turnover_cost_bps=1e6,
    )
    assert set(sol.solver_meta["subset"]) == set(warm)


def test_neal_selection_forwards_penalty_multiplier_only_when_given(monkeypatch):
    problem = _make_problem(20, 5, seed=1)
    seen: list[dict] = []
    real = neal_sampler.build_selection_bqm

    def spy(prob, cardinality, **kwargs):
        seen.append(kwargs)
        return real(prob, cardinality, **kwargs)

    monkeypatch.setattr(neal_sampler, "build_selection_bqm", spy)
    common = dict(cardinality=5, num_reads=10, num_sweeps=10, seed=1, top_subsets=3)
    solve_with_neal_selection(problem, **common)
    solve_with_neal_selection(problem, penalty_multiplier=7.0, **common)
    assert "penalty_multiplier" not in seen[0]  # builder default (100x) applies
    assert seen[1]["penalty_multiplier"] == 7.0


# ---------------------------------------------------------------------------
# swap-SA through the shared pipeline
# ---------------------------------------------------------------------------


def test_solve_with_swap_sa_pipeline_output():
    problem = _make_problem(24, 6, seed=10)
    sol = solve_with_swap_sa(problem, 6, num_reads=30, num_sweeps=30, seed=4, m=8)
    assert sol.feasible
    assert len(sol.solver_meta["subset"]) == 6
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-6)
    for key in (
        "sample_wall_s",
        "refine_wall_s",
        "n_candidates",
        "n_evaluated",
        "m",
        "num_reads",
        "num_sweeps",
        "t_hot",
        "t_cold",
        "frozen_fraction",
    ):
        assert key in sol.solver_meta
    assert sol.solver_meta["m"] == 8
    assert sol.solver_meta["n_evaluated"] <= 8


def test_solve_with_swap_sa_uses_problem_cardinality_by_default():
    problem = _make_problem(16, 4, seed=11)
    sol = solve_with_swap_sa(problem, num_reads=10, num_sweeps=10, seed=1, m=3)
    assert len(sol.solver_meta["subset"]) == 4


def test_solve_with_swap_sa_requires_a_cardinality():
    base = _make_problem(10, 3, seed=1)
    problem = PortfolioProblem(
        mu=base.mu, sigma=base.sigma, asset_names=base.asset_names, risk_aversion=2.0
    )
    with pytest.raises(ValueError):
        solve_with_swap_sa(problem, num_reads=4, num_sweeps=4, seed=1)


def test_check_feasible_applies_w_min_to_bought_assets_only():
    """R16: ``w_min`` is the minimum purchase; the zero weights of unbought assets are feasible
    (OR-Library uses ε = 0.01), as in ``cif.classical.milp._box_ok``."""
    n = 5
    problem = PortfolioProblem(mu=np.full(n, 0.1), sigma=np.eye(n) * 0.04,
                               asset_names=tuple("ABCDE"), risk_aversion=2.0,
                               cardinality=3, w_min=0.05, w_max=0.6)
    assert pipeline.check_feasible(np.array([0.5, 0.3, 0.2, 0.0, 0.0]), problem)
    assert pipeline.check_feasible(np.array([0.5, 0.3, 0.2, 1e-9, 0.0]), problem)  # solver noise
    assert not pipeline.check_feasible(np.array([0.57, 0.4, 0.03, 0.0, 0.0]), problem)  # 0.03 < w_min
    assert not pipeline.check_feasible(np.array([0.6, 0.45, -0.05, 0.0, 0.0]), problem)  # short
    assert not pipeline.check_feasible(np.array([0.7, 0.3, 0.0, 0.0, 0.0]), problem)  # > w_max
    assert not pipeline.check_feasible(np.array([0.25, 0.25, 0.25, 0.25, 0.0]), problem)  # K


# ---------------------------------------------------------------------------
# swap-SA: time interrupt for the anytime outer loop (ANALYSIS_PLAN §6, N6)
# ---------------------------------------------------------------------------


def _factor_problem(n: int, K: int, seed: int) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"a{i}" for i in range(n)),
        risk_aversion=5.0,
        cardinality=K,
    )


def test_swap_sa_without_an_interrupt_keeps_its_random_stream():
    """Regression against the output of the sampler before the interrupt hook existed."""
    problem = _factor_problem(18, 4, 1000)
    res = sample_swap_sa(problem, 4, num_reads=6, num_sweeps=8, seed=10003)
    assert res.subsets == [(0, 1, 11, 12), (0, 11, 12, 15), (0, 1, 10, 12)]
    assert res.read_subsets.tolist() == [
        [0, 1, 10, 12],
        [0, 1, 11, 12],
        [0, 1, 11, 12],
        [0, 1, 11, 12],
        [0, 1, 11, 12],
        [0, 11, 12, 15],
    ]
    np.testing.assert_allclose(
        res.energies, [-0.04136424947192182, -0.04093941318420023, -0.04054767472232569],
        rtol=1e-12,
    )
    assert res.t_hot == pytest.approx(0.02251488015887588, rel=1e-12)
    assert res.t_cold == pytest.approx(0.0002251488015887588, rel=1e-12)
    assert res.accept_rate == pytest.approx(0.30787037037037035, rel=1e-12)  # 266 / (6·18·8)
    assert res.frozen_fraction == 0.0
    assert res.sweeps_done == 8 and res.interrupted is False


def test_swap_sa_interrupt_function_that_never_fires_changes_nothing():
    problem = _factor_problem(18, 4, 1001)
    a = sample_swap_sa(problem, 4, num_reads=8, num_sweeps=10, seed=10004)
    b = sample_swap_sa(
        problem, 4, num_reads=8, num_sweeps=10, seed=10004, interrupt_function=lambda: False
    )
    assert a.subsets == b.subsets
    np.testing.assert_array_equal(a.read_subsets, b.read_subsets)
    np.testing.assert_array_equal(a.energies, b.energies)
    assert a.accept_rate == b.accept_rate


def test_swap_sa_interrupt_function_is_polled_before_every_sweep():
    problem = _factor_problem(18, 4, 1002)
    polls: list[int] = []

    def interrupt() -> bool:
        polls.append(1)
        return len(polls) > 3  # the first three sweeps run, the fourth is cancelled

    res = sample_swap_sa(
        problem, 4, num_reads=5, num_sweeps=50, seed=10005, interrupt_function=interrupt
    )
    assert res.interrupted is True
    assert res.sweeps_done == 3
    assert len(polls) == 4
    assert res.num_sweeps == 50  # the requested schedule is reported as requested
    # the chains are still hot: the acceptance is averaged over the 3 sweeps run (0.70 here),
    # not over the 50 requested (which would dilute it to 0.04)
    assert 0.3 < res.accept_rate <= 1.0
    # whatever state the chains reached is a valid batch: K-subsets, unique, by energy
    sur = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, 4)
    assert res.read_subsets.shape == (5, 4)
    assert len(set(res.subsets)) == len(res.subsets)
    assert np.all(np.diff(res.energies) >= 0)
    for subset, energy in zip(res.subsets, res.energies, strict=True):
        assert len(subset) == 4
        assert energy == pytest.approx(sur.energy(subset), abs=1e-10)


def test_swap_sa_interrupt_at_once_returns_the_initial_random_subsets():
    problem = _factor_problem(18, 4, 1003)
    res = sample_swap_sa(
        problem, 4, num_reads=6, num_sweeps=20, seed=10006, interrupt_function=lambda: True
    )
    assert res.interrupted is True and res.sweeps_done == 0
    assert res.accept_rate == 0.0
    assert res.read_subsets.shape == (6, 4)
    assert all(len(s) == 4 for s in res.subsets)


def test_swap_sa_time_interrupt_stops_a_long_run_near_the_deadline():
    problem = _factor_problem(40, 6, 1004)
    deadline = time.perf_counter() + 0.3
    t0 = time.perf_counter()
    res = sample_swap_sa(
        problem,
        6,
        num_reads=20,
        num_sweeps=1_000_000,  # hours of work if it were not interrupted
        seed=10007,
        interrupt_function=lambda: time.perf_counter() >= deadline,
    )
    elapsed = time.perf_counter() - t0
    assert res.interrupted is True
    assert 0 < res.sweeps_done < 1_000_000
    assert elapsed < 0.3 + 1.0  # one sweep of 40 proposals per read is far below the slack
    assert res.sample_wall_s == pytest.approx(elapsed, abs=0.2)


def test_swap_sa_with_a_single_subset_has_no_sweeps_to_interrupt():
    problem = _factor_problem(6, 6, 1005)
    res = sample_swap_sa(problem, 6, num_reads=3, num_sweeps=5, seed=10008,
                         interrupt_function=lambda: True)
    assert res.subsets == [tuple(range(6))]
    assert res.sweeps_done == 0 and res.interrupted is False
