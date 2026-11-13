"""Tests for the Tabu-based selection solver."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from dwave.samplers import TabuSampler

from cif.classical.milp import solve_miqp_scip
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import sample_tabu, solve_with_tabu_selection
from cif.qubo.builder import build_selection_bqm


def _make_problem(n: int = 20, K: int = 5, seed: int = 42) -> PortfolioProblem:
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


def test_tabu_returns_feasible_cardinality_K_subset():
    """Tabu must always honour the requested cardinality."""
    problem = _make_problem(n=15, K=4)
    sol = solve_with_tabu_selection(
        problem, cardinality=4, num_reads=50, tenure=10, seed=1
    )
    assert sol.feasible, "Tabu produced an infeasible portfolio"
    assert len(sol.solver_meta["subset"]) == 4
    assert abs(sol.weights.sum() - 1.0) < 1e-6
    # support must coincide with the reported subset
    support = set(int(i) for i in np.flatnonzero(sol.weights > 1e-9))
    assert support == set(sol.solver_meta["subset"])


@pytest.fixture(scope="module")
def scip_reference():
    """30-asset instance where SCIP solves quickly, solved once for the module."""
    problem = _make_problem(n=30, K=8, seed=7)
    scip = solve_miqp_scip(problem, time_limit_s=30.0)
    assert scip.feasible
    return problem, scip


def test_tabu_closes_gap_to_scip_on_small_instance(scip_reference):
    """On a 30-asset instance where SCIP solves quickly, Tabu must reach
    near-optimal (≤ 1% gap)."""
    problem, scip = scip_reference
    tabu = solve_with_tabu_selection(
        problem, cardinality=8, num_reads=200, tenure=20, seed=7
    )
    assert tabu.feasible
    gap = abs(tabu.objective - scip.objective) / max(abs(scip.objective), 1e-9)
    assert gap < 0.01, f"Tabu gap to SCIP = {gap*100:.2f}% (expected <1%)"


def test_tabu_deterministic_mode_closes_gap_to_scip(scip_reference):
    """Same check without a wall-clock cap: termination only by restarts."""
    problem, scip = scip_reference
    tabu = solve_with_tabu_selection(
        problem,
        cardinality=8,
        num_reads=40,
        tenure=20,
        seed=7,
        timeout_ms=None,
        num_restarts=2,
    )
    assert tabu.feasible
    gap = abs(tabu.objective - scip.objective) / max(abs(scip.objective), 1e-9)
    assert gap < 0.01, f"Tabu (deterministic) gap to SCIP = {gap*100:.2f}% (expected <1%)"


def test_tabu_warm_start_keeps_previous_subset_when_no_improvement():
    """With a high turnover cost and a warm-start subset, Tabu should
    re-pick the warm-start if a fresh search doesn't beat it by enough."""
    problem = _make_problem(n=30, K=4, seed=3)
    warm = [0, 1, 2, 3]
    # Build a previous weights vector matching the warm subset (uniform).
    prev = np.zeros(problem.n)
    prev[warm] = 1.0 / 4
    sol = solve_with_tabu_selection(
        problem,
        cardinality=4,
        num_reads=200,
        tenure=20,
        seed=99,
        warm_start_subset=warm,
        prev_weights=prev,
        turnover_cost_bps=1e6,  # massive cost — nothing should beat warm
    )
    assert sol.feasible
    assert set(sol.solver_meta["subset"]) == set(warm)


# ---------------------------------------------------------------------------
# Termination control (WP-D): timeout vs restarts
# ---------------------------------------------------------------------------

_SAMPLER_DEFAULTS = {
    name: param.default
    for name, param in inspect.signature(TabuSampler.sample).parameters.items()
}


def test_tabu_legacy_defaults_reproduce_the_sampler_defaults():
    """timeout_ms=20 / num_restarts=None are exactly what the wrapper used implicitly."""
    params = inspect.signature(solve_with_tabu_selection).parameters
    assert params["timeout_ms"].default == _SAMPLER_DEFAULTS["timeout"] == 20
    assert params["num_restarts"].default is None


def test_tabu_deterministic_mode_gives_identical_samples_and_energies():
    """timeout=None removes the clock: one seed -> bitwise identical sample sets."""
    problem = _make_problem(n=24, K=6, seed=11)
    bqm = build_selection_bqm(problem, cardinality=6)
    kwargs = dict(num_reads=12, tenure=8, seed=21, timeout_ms=None, num_restarts=2)
    a = sample_tabu(bqm, **kwargs)
    b = sample_tabu(bqm, **kwargs)
    assert np.array_equal(a.record.sample, b.record.sample)
    assert np.array_equal(a.record.energy, b.record.energy)
    # every read ran exactly the requested restarts: nothing was cut by a timer
    assert np.all(a.record.num_restarts == 2)


def test_tabu_deterministic_mode_gives_identical_subsets_and_objectives():
    problem = _make_problem(n=24, K=6, seed=11)
    kwargs = dict(
        cardinality=6,
        num_reads=12,
        tenure=8,
        seed=21,
        top_subsets=10,
        timeout_ms=None,
        num_restarts=2,
    )
    a = solve_with_tabu_selection(problem, **kwargs)
    b = solve_with_tabu_selection(problem, **kwargs)
    assert a.solver_meta["subset"] == b.solver_meta["subset"]
    assert a.objective == b.objective
    assert a.solver_meta["sample_best_energy"] == b.solver_meta["sample_best_energy"]


def test_tabu_deterministic_mode_requires_a_restart_cap():
    problem = _make_problem(n=12, K=3)
    with pytest.raises(ValueError, match="num_restarts"):
        solve_with_tabu_selection(
            problem, cardinality=3, num_reads=4, seed=1, timeout_ms=None, num_restarts=None
        )


def test_tabu_solver_meta_in_timeout_mode():
    problem = _make_problem(n=15, K=4)
    sol = solve_with_tabu_selection(
        problem, cardinality=4, num_reads=10, tenure=10, seed=1, top_subsets=6
    )
    meta = sol.solver_meta
    assert meta["termination"] == "timeout"
    assert meta["timeout_ms"] == 20
    assert meta["num_restarts"] == _SAMPLER_DEFAULTS["num_restarts"]
    assert meta["tenure"] == 10
    assert meta["num_reads"] == 10
    assert meta["m"] == 6
    for key in ("sample_wall_s", "refine_wall_s", "restarts_mean"):
        assert meta[key] >= 0.0
    assert "sample_best_energy" in meta
    assert meta["n_evaluated"] <= 6


def test_tabu_solver_meta_in_deterministic_mode():
    problem = _make_problem(n=15, K=4)
    sol = solve_with_tabu_selection(
        problem,
        cardinality=4,
        num_reads=10,
        tenure=10,
        seed=1,
        top_subsets=6,
        timeout_ms=None,
        num_restarts=3,
    )
    meta = sol.solver_meta
    assert meta["termination"] == "restarts"
    assert meta["timeout_ms"] is None
    assert meta["num_restarts"] == 3
    assert meta["restarts_mean"] == 3.0  # no read was truncated by a timer


def test_tabu_tenure_is_capped_below_the_number_of_variables():
    problem = _make_problem(n=8, K=2)
    sol = solve_with_tabu_selection(problem, cardinality=2, num_reads=5, tenure=20, seed=2)
    assert sol.solver_meta["tenure"] == 20  # requested
    assert sol.solver_meta["tenure_effective"] == 7  # n - 1


def test_tabu_selection_penalty_multiplier_and_pool_order_are_accepted():
    problem = _make_problem(n=15, K=4)
    sol = solve_with_tabu_selection(
        problem,
        cardinality=4,
        num_reads=8,
        tenure=10,
        seed=1,
        top_subsets=5,
        timeout_ms=None,
        num_restarts=1,
        penalty_multiplier=30.0,
        pool_order="energy",
    )
    assert len(sol.solver_meta["subset"]) == 4
