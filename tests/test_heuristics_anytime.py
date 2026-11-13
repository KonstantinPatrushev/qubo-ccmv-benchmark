"""The anytime outer loop of the QUBO heuristics (ANALYSIS_PLAN §6, N6; audit RF-1).

The budget logic is checked on a scripted sampler and a fake clock, so no test sleeps: batches cost
what the script says, and the refinement is the real shared one. The real samplers (SA, Tabu,
swap-SA) are exercised on small `factor` instances; the numbers of batches in a wall-clock budget
are machine dependent and are never asserted, only the properties of whatever was run. Instance
seeds are calibration ones (1000–1999), solver seeds 10000–19999.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from itertools import combinations, pairwise

import dimod
import numpy as np
import pytest

from cif.classical.refine import refine_problem
from cif.heuristics import anytime
from cif.heuristics.anytime import (
    ANYTIME_SOLVERS,
    Batch,
    SABatches,
    SwapSABatches,
    TabuBatches,
    anytime_sa,
    anytime_swap_sa,
    anytime_tabu,
    plan_tabu_batch,
    run_anytime,
)
from cif.heuristics.start import CommonStart, common_start
from cif.heuristics.swap_sa import SwapSurrogate
from cif.instances.factor import factor_instance
from cif.metrics.anytime import incumbent_curve
from cif.problem import PortfolioProblem


def _problem(n: int = 24, seed: int = 1000, K: int = 4, *, w_min: float = 0.0) -> PortfolioProblem:
    mu, sigma, _ = factor_instance(n, seed)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"a{i}" for i in range(n)),
        risk_aversion=5.0,
        w_min=w_min,
        cardinality=K,
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class ScriptedSampler:
    """Returns the scripted batch ``k`` (the last one repeats) and spends ``cost_s`` of fake time."""

    label = "scripted"

    def __init__(self, clock, script, *, cost_s=1.0, prepare_s=0.0):
        self.clock = clock
        self.script = script
        self.cost_s = cost_s
        self.prepare_s = prepare_s
        self.prepared = 0
        self.calls: list[tuple[int, float, bool]] = []

    def prepare(self) -> None:
        self.prepared += 1
        self.clock.advance(self.prepare_s)

    def sample(self, seed, *, interrupt, time_left_s) -> Batch:
        # a fake clock that does not advance would loop for ever and fill memory: fail instead
        assert len(self.calls) < 1000, "runaway loop: the fake clock does not advance"
        self.calls.append((seed, time_left_s, interrupt()))
        subsets, energies = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        self.clock.advance(self.cost_s)
        n = len(subsets)
        return Batch(list(subsets), np.asarray(energies, dtype=float), n, n, self.cost_s)


def _spy_refinements(monkeypatch, clock=None, cost_s=0.0) -> list[tuple[int, ...]]:
    """Record the supports refined by the loop; optionally charge ``cost_s`` of fake time each."""
    refined: list[tuple[int, ...]] = []
    real = anytime.refine_problem

    def spy(problem, subset, *args, **kwargs):
        refined.append(tuple(subset))
        if clock is not None:
            clock.advance(cost_s)
        return real(problem, subset, *args, **kwargs)

    monkeypatch.setattr(anytime, "refine_problem", spy)
    return refined


def _ranked_triples(problem):
    """All K = 3 subsets of the first 8 assets, best (lowest refined objective) first."""
    triples = list(combinations(range(8), 3))
    return sorted(triples, key=lambda s: refine_problem(problem, s).objective)


# ---------------------------------------------------------------------------
# 1. The outer loop: budget, pool, incumbent, trajectory (scripted sampler, fake clock)
# ---------------------------------------------------------------------------


def test_the_loop_runs_batches_until_the_budget_is_exhausted():
    problem = _problem(K=3)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0])], cost_s=3.0)
    sol = run_anytime(problem, 3, sampler, budget_s=10.0, seed=10001, clock=clock)
    assert len(sampler.calls) == 4  # batches start at 0, 3, 6 and 9 s
    assert sampler.prepared == 1
    meta = sol.solver_meta
    assert meta["n_batches"] == 4 and meta["stop"] == "budget"
    assert sol.wall_time_s == pytest.approx(12.0)  # the last batch could not be interrupted


def test_a_batch_that_would_start_exactly_at_the_budget_is_not_run():
    problem = _problem(K=3)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0])], cost_s=2.5)
    sol = run_anytime(problem, 3, sampler, budget_s=10.0, seed=10001, clock=clock)
    assert len(sampler.calls) == 4 and sol.wall_time_s == pytest.approx(10.0)


def test_max_batches_stops_the_loop_without_a_clock_limit():
    problem = _problem(K=3)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=1e9, seed=10002, max_batches=1, clock=clock)
    assert len(sampler.calls) == 1
    assert sol.solver_meta["stop"] == "max_batches" and sol.solver_meta["n_batches"] == 1


def test_new_subsets_are_refined_once_in_energy_order_and_at_most_m_b_per_batch(monkeypatch):
    problem = _problem(K=3)
    clock = FakeClock()
    a, b, c, d, e = (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4)
    f, g = (1, 2, 3), (1, 2, 4)
    script = [
        ([a, b, c, d, e], [0.0, 1.0, 2.0, 3.0, 4.0]),  # m_b = 3 refines a, b, c; d and e are cut
        ([b, a, f, c, d, g], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),  # new: f, d, g (d was never refined)
    ]
    refined = _spy_refinements(monkeypatch)
    sampler = ScriptedSampler(clock, script, cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=2.5, seed=10003, m_b=3, clock=clock)
    assert refined == [a, b, c, f, d, g]
    assert len(set(refined)) == len(refined)  # nothing is refined twice
    meta = sol.solver_meta
    assert meta["n_refined"] == 6 and meta["n_batches"] == 3  # the third batch brings nothing new
    assert meta["n_new"] == 6 and meta["pool_size"] == 6  # a batch's cut-off subsets are no pool


def test_the_cumulative_pool_deduplicates_across_batches(monkeypatch):
    problem = _problem(K=3)
    clock = FakeClock()
    refined = _spy_refinements(monkeypatch)
    batch = ([(0, 1, 2), (0, 1, 3)], [0.0, 1.0])
    sampler = ScriptedSampler(clock, [batch], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=5.0, seed=10004, m_b=10, clock=clock)
    assert refined == [(0, 1, 2), (0, 1, 3)]  # five batches, two refinements
    assert sol.solver_meta["n_batches"] == 5
    assert sol.solver_meta["n_unique"] == 10 and sol.solver_meta["n_new"] == 2


def test_the_common_start_is_the_first_incumbent_and_is_in_the_pool(monkeypatch):
    problem = _problem(K=3)
    start = dataclasses.replace(common_start(problem), time_s=0.5)
    assert start.status == "optimal"
    clock = FakeClock()
    refined = _spy_refinements(monkeypatch)
    other = (5, 6, 7)
    sampler = ScriptedSampler(clock, [([start.subset, other], [0.0, 1.0])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=2.0, seed=10005, start=start, clock=clock)
    assert start.subset not in refined and refined == [other]  # the start is not refined again
    traj = sol.solver_meta["trajectory"]
    assert traj[0] == (0.5, start.objective)
    assert sol.objective <= start.objective
    # the clock is the common one: the start's construction time is already spent at the begin,
    # so the second batch starts at 1.5 s and ends at 2.5 s
    assert len(sampler.calls) == 2 and sol.wall_time_s == pytest.approx(2.5)
    assert sol.solver_meta["start_used"] is True


def test_a_start_that_spent_the_whole_budget_leaves_no_time_for_a_batch():
    problem = _problem(K=3)
    start = dataclasses.replace(common_start(problem), time_s=12.0)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(5, 6, 7)], [0.0])])
    sol = run_anytime(problem, 3, sampler, budget_s=10.0, seed=10006, start=start, clock=clock)
    assert sampler.calls == [] and sol.solver_meta["n_batches"] == 0
    np.testing.assert_array_equal(sol.weights, start.weights)
    assert not np.shares_memory(sol.weights, start.weights)  # the result owns its weights
    assert sol.objective == start.objective
    assert sol.solver_meta["trajectory"] == [(12.0, start.objective)]


def test_a_failed_start_is_ignored_but_its_time_is_spent():
    problem = _problem(K=3)
    good = common_start(problem)
    failed = CommonStart(
        weights=np.zeros(problem.n), objective=math.inf, subset=good.subset,
        status="infeasible", time_s=0.3, cpu_s=0.3, relax_s=0.1, refine_s=0.1,
    )
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([good.subset], [0.0])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=3.0, seed=10007, start=failed, clock=clock)
    assert sol.solver_meta["start_used"] is False
    assert sol.objective == pytest.approx(good.objective)  # refined as an ordinary candidate
    assert sol.solver_meta["trajectory"][0][0] == pytest.approx(1.3)  # first entry: batch end


def test_the_trajectory_is_stamped_at_the_end_of_each_improving_batch():
    problem = _problem(K=3)
    ranked = _ranked_triples(problem)
    worst, best = ranked[-1], ranked[0]
    clock = FakeClock()
    script = [
        ([worst], [0.0]),
        ([worst], [0.0]),  # nothing new in the second batch
        ([best], [0.0]),
    ]
    sampler = ScriptedSampler(clock, script, cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=5.0, seed=10008, clock=clock)
    traj = sol.solver_meta["trajectory"]
    assert [t for t, _ in traj] == [1.0, 3.0]  # batch 1 ends at 1 s, batch 3 (2 s .. 3 s) improves
    assert traj[0][1] > traj[1][1]
    assert traj[1][1] == pytest.approx(refine_problem(problem, best).objective)
    assert sol.objective == traj[-1][1]
    assert sol.solver_meta["subset"] == list(best)
    # the curve of the metrics module is the trajectory itself
    times, values = incumbent_curve(traj)
    assert list(times) == [1.0, 3.0] and list(values) == [traj[0][1], traj[1][1]]


def test_the_refinement_of_a_batch_stops_at_the_budget(monkeypatch):
    problem = _problem(K=3)
    clock = FakeClock()
    subsets = _ranked_triples(problem)[:10]
    refined = _spy_refinements(monkeypatch, clock=clock, cost_s=1.0)  # each refinement: 1 s
    sampler = ScriptedSampler(clock, [(subsets, np.arange(10.0))], cost_s=0.0)
    sol = run_anytime(problem, 3, sampler, budget_s=3.5, seed=10009, m_b=10, clock=clock)
    assert len(refined) == 4  # started at 0, 1, 2 and 3 s, all before the budget of 3.5 s
    assert sol.solver_meta["n_refined"] == 4
    # the unrefined six are no pool: they stay eligible
    assert sol.solver_meta["pool_size"] == 4


def test_an_equal_objective_does_not_replace_the_incumbent(monkeypatch):
    problem = _problem(K=3)
    clock = FakeClock()
    real = anytime.refine_problem

    def tied(problem_, subset, *args, **kwargs):  # every subset has the same value
        result = real(problem_, subset, *args, **kwargs)
        return dataclasses.replace(result, objective=-0.123)

    monkeypatch.setattr(anytime, "refine_problem", tied)
    sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0]), ([(0, 1, 3), (0, 2, 3)], [0.0, 1.0])])
    sol = run_anytime(problem, 3, sampler, budget_s=2.5, seed=10010, clock=clock)
    assert sol.solver_meta["subset"] == [0, 1, 2]  # ties keep the earliest
    assert sol.solver_meta["trajectory"] == [(1.0, -0.123)]


def test_the_first_own_solution_is_the_end_of_the_first_batch_that_refined_something():
    problem = _problem(K=3)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([], []), ([(0, 1, 2)], [0.0])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=3.5, seed=10010, clock=clock)
    assert sol.solver_meta["first_own_s"] == pytest.approx(2.0)  # batch 1 was empty
    clock = FakeClock()
    nothing = ScriptedSampler(clock, [([], [])], cost_s=1.0)
    sol = run_anytime(problem, 3, nothing, budget_s=3.5, seed=10010, clock=clock)
    assert sol.solver_meta["first_own_s"] is None


def test_the_value_of_the_first_own_solution_is_the_best_of_the_first_batch_that_refined_something():
    """§12 A4: ``first_own_s`` alone says when the method had a solution of its own, not how good it was."""
    problem = _problem(K=3)
    clock = FakeClock()
    triples = _ranked_triples(problem)
    batch = [triples[5], triples[2], triples[9]]
    sampler = ScriptedSampler(clock, [([], []), (batch, [0.0, 1.0, 2.0]), ([triples[0]], [0.0])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=3.5, seed=10010, clock=clock)
    meta = sol.solver_meta
    expected = min(refine_problem(problem, t).objective for t in batch)
    assert meta["first_own_s"] == pytest.approx(2.0) and meta["first_own_f"] == pytest.approx(expected)
    assert sol.objective < meta["first_own_f"]  # the third batch found better: the first value is not the final
    clock = FakeClock()
    nothing = ScriptedSampler(clock, [([], [])], cost_s=1.0)
    sol = run_anytime(problem, 3, nothing, budget_s=3.5, seed=10010, clock=clock)
    assert sol.solver_meta["first_own_f"] is None


def test_the_first_own_value_is_recorded_when_it_does_not_beat_the_common_start():
    problem = _problem(K=3)
    clock = FakeClock()
    triples = _ranked_triples(problem)
    start = dataclasses.replace(common_start(problem, 3), objective=-10.0, time_s=0.5)  # better than anything
    sampler = ScriptedSampler(clock, [([triples[4]], [0.0])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=3.0, seed=10010, start=start, clock=clock, max_batches=1)
    meta = sol.solver_meta
    assert meta["trajectory"] == [(0.5, -10.0)]  # no improvement over the start ...
    assert meta["first_own_s"] == pytest.approx(1.5)  # ... but the method had a solution of its own
    assert meta["first_own_f"] == pytest.approx(refine_problem(problem, triples[4]).objective)
    assert meta["first_own_f"] > -10.0


def test_the_refinement_time_estimate_follows_the_batches_run(monkeypatch):
    problem = _problem(K=3)
    clock = FakeClock()
    triples = _ranked_triples(problem)
    _spy_refinements(monkeypatch, clock=clock, cost_s=0.01)  # 10 ms per refinement, fake time
    sampler = ScriptedSampler(
        clock, [(triples[:3], np.arange(3.0)), (triples[3:6], np.arange(3.0))], cost_s=1.0
    )
    run_anytime(problem, 3, sampler, budget_s=10.0, seed=10010, m_b=50, max_batches=2, clock=clock)
    # batch 2 starts at 1.03 s; the estimate is 10 ms per refinement, so 50 · 0.01 = 0.5 s are kept
    left = 10.0 - 1.03
    assert sampler.calls[1][1] == pytest.approx(left - 0.5)
    assert sampler.calls[0][1] == pytest.approx(10.0)  # no estimate before the first batch


def test_the_sampler_is_interrupted_before_the_budget_by_the_refinement_reserve():
    problem = _problem(K=3)
    start = dataclasses.replace(common_start(problem), time_s=0.0, refine_s=0.01)
    clock = FakeClock()
    seen: list = []

    class Spy(ScriptedSampler):
        def sample(self, seed, *, interrupt, time_left_s):
            seen.append(interrupt)
            return super().sample(seed, interrupt=interrupt, time_left_s=time_left_s)

    sampler = Spy(clock, [([(5, 6, 7)], [0.0])], cost_s=0.0)
    run_anytime(
        problem, 3, sampler, budget_s=10.0, seed=10010, m_b=50, start=start, max_batches=1,
        clock=clock,
    )
    clock.t = 9.4
    assert seen[0]() is False
    clock.t = 9.6  # past budget − reserve = 9.5, still inside the budget
    assert seen[0]() is True


def test_the_sampler_gets_a_budget_that_leaves_time_to_refine_its_batch():
    problem = _problem(K=3)
    start = dataclasses.replace(common_start(problem), time_s=0.0, refine_s=0.01)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(5, 6, 7)], [0.0])], cost_s=0.0)
    run_anytime(
        problem, 3, sampler, budget_s=10.0, seed=10010, m_b=50, start=start, max_batches=1,
        clock=clock,
    )
    # reserve = m_b · (refinement time estimate) = 0.5 s, below the 20 % cap
    assert sampler.calls[0][1] == pytest.approx(10.0 - 0.5)
    # the cap: a long estimate never takes more than a fifth of the time left
    clock2 = FakeClock()
    sampler2 = ScriptedSampler(clock2, [([(5, 6, 7)], [0.0])], cost_s=0.0)
    slow = dataclasses.replace(start, refine_s=1.0)
    run_anytime(
        problem, 3, sampler2, budget_s=10.0, seed=10010, m_b=50, start=slow, max_batches=1,
        clock=clock2,
    )
    assert sampler2.calls[0][1] == pytest.approx(10.0 - 2.0)


def test_batch_seeds_are_reproducible_distinct_and_depend_on_the_run_seed():
    problem = _problem(K=3)

    def seeds(seed):
        clock = FakeClock()
        sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0])], cost_s=1.0)
        run_anytime(problem, 3, sampler, budget_s=6.0, seed=seed, clock=clock)
        return [s for s, _, _ in sampler.calls]

    first = seeds(10011)
    assert first == seeds(10011)
    assert len(set(first)) == len(first) == 6
    assert set(first).isdisjoint(seeds(10012))
    # dwave-samplers reject a seed of 2**31 or more, although their message says 2**32 − 1
    assert all(0 <= s < 2**31 for s in first)


def test_the_bookkeeping_adds_up():
    problem = _problem(K=3)
    clock = FakeClock()
    batch = ([(0, 1, 2), (0, 1, 3), (0, 2, 3)], [0.0, 1.0, 2.0])
    sampler = ScriptedSampler(clock, [batch], cost_s=2.0, prepare_s=0.5)
    sol = run_anytime(
        problem, 3, sampler, budget_s=7.0, seed=10013, m_b=2, keep_batch_log=True, clock=clock
    )
    meta = sol.solver_meta
    log = meta["batches"]
    assert len(log) == meta["n_batches"] == 4  # batches start at 0.5, 2.5, 4.5 and 6.5 s < 7 s
    assert meta["build_s"] == pytest.approx(0.5)
    assert meta["sample_s"] == pytest.approx(2.0 * meta["n_batches"])
    assert meta["sample_s"] + meta["decode_s"] + meta["refine_s"] + meta["overhead_s"] == (
        pytest.approx(sol.wall_time_s - meta["build_s"])
    )
    assert meta["n_samples"] == sum(entry["n_samples"] for entry in log)
    assert meta["first_own_s"] == pytest.approx(log[0]["t_end"])
    for entry in log:
        assert entry["t_end"] - entry["t_start"] == pytest.approx(2.0)
        assert entry["sample_s"] == 2.0
    json.dumps(meta)  # the registry row must serialise


def test_the_batch_log_is_off_by_default():
    problem = _problem(K=3)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0])])
    sol = run_anytime(problem, 3, sampler, budget_s=2.0, seed=10014, clock=clock)
    assert "batches" not in sol.solver_meta


def test_without_any_candidate_the_result_is_an_infeasible_empty_solution():
    problem = _problem(K=3)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([], [])], cost_s=1.0)
    sol = run_anytime(problem, 3, sampler, budget_s=2.0, seed=10015, clock=clock)
    assert sol.feasible is False and sol.objective == math.inf
    assert np.all(sol.weights == 0.0)
    assert sol.solver_meta["trajectory"] == [] and sol.solver_meta["subset"] is None


def test_eq_mode_uses_the_minimum_purchase_semantics(monkeypatch):
    problem = _problem(K=3, w_min=0.05)
    clock = FakeClock()
    sampler = ScriptedSampler(clock, [([(0, 1, 2)], [0.0])], cost_s=0.1)
    sol = run_anytime(problem, 3, sampler, budget_s=1.0, seed=10016, card_mode="eq", clock=clock)
    ref = refine_problem(problem, (0, 1, 2), card_mode="eq")
    assert sol.objective == pytest.approx(ref.objective, abs=1e-14)
    assert np.all(sol.weights[[0, 1, 2]] >= 0.05 - 1e-9)
    assert sol.solver_meta["card_mode"] == "eq"
    fresh = ScriptedSampler(FakeClock(), [([(0, 1, 2)], [0.0])])
    with pytest.raises(ValueError, match="w_min"):  # 'le' with a minimum purchase is rejected
        run_anytime(problem, 3, fresh, budget_s=1.0, seed=10016, card_mode="le")
    with pytest.raises(ValueError, match="card_mode"):
        run_anytime(problem, 3, fresh, budget_s=1.0, seed=10016, card_mode="ge")
    assert fresh.calls == [] and fresh.prepared == 0  # rejected before any work


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget_s": 0.0},
        {"budget_s": -1.0},
        {"budget_s": math.nan},
        {"budget_s": math.inf},
        {"m_b": 0},
        {"seed": -1},
        {"max_batches": 0},
    ],
)
def test_the_loop_rejects_invalid_parameters(kwargs):
    problem = _problem(K=3)
    params = {"budget_s": 1.0, "seed": 10017, **kwargs}
    with pytest.raises(ValueError):
        run_anytime(problem, 3, ScriptedSampler(FakeClock(), [([(0, 1, 2)], [0.0])]), **params)


def test_the_loop_rejects_a_cardinality_out_of_range_and_a_start_of_the_wrong_size():
    problem = _problem(K=3)
    sampler = ScriptedSampler(FakeClock(), [([(0, 1, 2)], [0.0])])
    with pytest.raises(ValueError, match="cardinality"):
        run_anytime(problem, 0, sampler, budget_s=1.0, seed=10018)
    start = common_start(problem, 4)
    with pytest.raises(ValueError, match="start"):
        run_anytime(problem, 3, sampler, budget_s=1.0, seed=10018, start=start)


# ---------------------------------------------------------------------------
# 2. Samples -> unique feasible subsets in surrogate-energy order
# ---------------------------------------------------------------------------


def test_unique_feasible_subsets_keep_k_sized_samples_by_ascending_energy():
    rows = np.array(
        [
            [1, 1, 0, 0, 0],  # energy 3.0
            [0, 1, 1, 0, 0],  # energy 1.0
            [1, 1, 0, 0, 0],  # energy 3.0 (duplicate)
            [0, 0, 1, 1, 0],  # energy 2.0
            [1, 1, 1, 0, 0],  # size 3: infeasible for K = 2, energy 0.0
            [0, 0, 0, 0, 0],  # size 0
        ]
    )
    sampleset = dimod.SampleSet.from_samples(
        rows, energy=[3.0, 1.0, 3.0, 2.0, 0.0, 9.0], vartype="BINARY"
    )
    subsets, energies, n_feasible = anytime._unique_feasible(sampleset, 5, 2)
    assert subsets == [(1, 2), (2, 3), (0, 1)]
    np.testing.assert_array_equal(energies, [1.0, 2.0, 3.0])
    assert n_feasible == 4  # the duplicate counts as a feasible sample


def test_unique_feasible_handles_an_empty_sample_set():
    sampleset = dimod.SampleSet.from_samples(np.zeros((0, 5), dtype=int), energy=[], vartype="BINARY")
    subsets, energies, n_feasible = anytime._unique_feasible(sampleset, 5, 2)
    assert subsets == [] and energies.size == 0 and n_feasible == 0


# ---------------------------------------------------------------------------
# 3. The Tabu tail: fewer restarts instead of an interrupt
# ---------------------------------------------------------------------------


def test_plan_tabu_batch_keeps_a_batch_that_fits_or_has_no_estimate_yet():
    assert plan_tabu_batch(10, 20, None, 1.0) == (10, 20)
    assert plan_tabu_batch(10, 20, 0.001, 1.0) == (10, 20)  # 210 searches · 1 ms = 0.21 s


def test_plan_tabu_batch_cuts_restarts_first_then_reads():
    # 10 reads · (1 + 20) searches at 10 ms is 2.1 s; 1.0 s holds 100 searches: 10 per read
    assert plan_tabu_batch(10, 20, 0.01, 1.0) == (10, 9)
    # 0.05 s holds 5 searches, fewer than the reads: 5 simple searches
    assert plan_tabu_batch(10, 20, 0.01, 0.05) == (5, 0)
    # not even one search fits
    assert plan_tabu_batch(10, 20, 0.01, 0.005) == (0, 0)
    assert plan_tabu_batch(10, 20, 0.01, 0.0) == (0, 0)


def test_the_planned_batch_never_exceeds_the_time_left():
    rng = np.random.default_rng(10019)
    for _ in range(200):
        reads, restarts = int(rng.integers(1, 30)), int(rng.integers(0, 100))
        per_search, left = float(rng.uniform(1e-4, 0.05)), float(rng.uniform(0.0, 3.0))
        r, k = plan_tabu_batch(reads, restarts, per_search, left)
        assert 0 <= r <= reads and 0 <= k <= restarts
        if reads * (1 + restarts) * per_search <= left:
            assert (r, k) == (reads, restarts)  # a batch that fits is left alone
        else:
            assert r * (1 + k) * per_search <= left + 1e-12


# ---------------------------------------------------------------------------
# 4. Real samplers on a small factor instance
# ---------------------------------------------------------------------------


def _sa(problem, **kw):
    return anytime_sa(problem, 4, num_reads_b=20, num_sweeps=60, **kw)


def _tabu(problem, **kw):
    return anytime_tabu(problem, 4, num_restarts_b=2, num_reads_b=4, tenure=8, **kw)


def _swap(problem, **kw):
    return anytime_swap_sa(problem, 4, num_reads_b=12, num_sweeps=25, **kw)


SOLVERS = {"sa": _sa, "tabu": _tabu, "swap_sa": _swap}


def test_the_registry_names_the_three_anytime_heuristics():
    assert set(ANYTIME_SOLVERS) == {"sa", "tabu", "swap_sa"}
    assert ANYTIME_SOLVERS["sa"] is anytime_sa and ANYTIME_SOLVERS["tabu"] is anytime_tabu
    assert ANYTIME_SOLVERS["swap_sa"] is anytime_swap_sa


@pytest.mark.parametrize("name", list(SOLVERS))
def test_a_fixed_number_of_batches_is_deterministic_for_a_seed(name):
    # R12: no wall-clock criterion inside a batch; max_batches replaces the budget (RQ1 mode)
    problem = _problem(K=4)
    start = common_start(problem)
    runs = [
        SOLVERS[name](problem, budget_s=1e6, seed=10020, start=start, max_batches=3, m_b=20)
        for _ in range(2)
    ]
    a, b = runs
    assert a.solver_meta["n_batches"] == b.solver_meta["n_batches"] == 3
    assert a.solver_meta["subset"] == b.solver_meta["subset"]
    assert a.objective == b.objective
    np.testing.assert_array_equal(a.weights, b.weights)
    assert [f for _, f in a.solver_meta["trajectory"]] == [f for _, f in b.solver_meta["trajectory"]]
    other = SOLVERS[name](problem, budget_s=1e6, seed=10021, start=start, max_batches=3, m_b=20)
    assert other.solver_meta["n_batches"] == 3


@pytest.mark.parametrize("name", list(SOLVERS))
def test_a_run_returns_a_feasible_portfolio_with_a_consistent_trajectory(name):
    problem = _problem(K=4)
    start = common_start(problem)
    sol = SOLVERS[name](problem, budget_s=0.5, seed=10022, start=start, m_b=20)
    meta = sol.solver_meta
    assert sol.feasible
    assert np.count_nonzero(sol.weights > 1e-9) <= 4
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert sol.objective == pytest.approx(problem.objective_value(sol.weights), abs=1e-12)
    assert sol.objective <= start.objective + 1e-12  # the start is in the pool
    traj = meta["trajectory"]
    assert traj[0] == (start.time_s, start.objective)
    times = [t for t, _ in traj]
    values = [f for _, f in traj]
    assert times == sorted(times) and all(b < a for a, b in pairwise(values))
    assert values[-1] == pytest.approx(sol.objective, abs=1e-12)
    assert meta["n_batches"] >= 1 and meta["start_used"] is True
    assert set(meta["subset"]) >= set(np.flatnonzero(sol.weights > 1e-9).tolist())
    json.dumps(meta)
    # end-to-end clock: the budget includes the start; one batch may overshoot a little
    assert sol.wall_time_s < 0.5 + 0.75


@pytest.mark.parametrize("name", list(SOLVERS))
def test_the_incumbent_is_never_worse_than_the_start_and_improves_on_a_bad_one(name):
    problem = _problem(K=4)
    good = common_start(problem)
    # a deliberately poor start: the worst of a sample of 4-subsets, refined
    rng = np.random.default_rng(10023)
    subsets = [tuple(sorted(rng.choice(24, size=4, replace=False).tolist())) for _ in range(30)]
    worst = max(subsets, key=lambda s: refine_problem(problem, s).objective)
    ref = refine_problem(problem, worst)
    poor = dataclasses.replace(good, weights=ref.weights, objective=ref.objective, subset=worst)
    sol = SOLVERS[name](problem, budget_s=0.4, seed=10024, start=poor, m_b=50)
    assert sol.objective < poor.objective  # any sampler beats the worst of 30 random subsets
    assert sol.solver_meta["trajectory"][0] == (poor.time_s, poor.objective)


def test_sa_energies_of_feasible_samples_are_the_surrogate_energies():
    problem = _problem(K=4)
    sampler = SABatches(problem, 4, num_reads_b=30, num_sweeps=40)
    sampler.prepare()
    batch = sampler.sample(10025, interrupt=lambda: False, time_left_s=10.0)
    surrogate = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, 4)
    assert batch.subsets and np.all(np.diff(batch.energies) >= 0)
    for subset, energy in zip(batch.subsets, batch.energies, strict=True):
        assert len(subset) == 4
        assert energy == pytest.approx(surrogate.energy(subset), abs=1e-9)
    assert batch.n_samples == 30 and batch.n_feasible <= 30 and not batch.interrupted


# A batch that no budget of these tests can hold, made of LONG reads: a read of 10 000 sweeps takes
# about 15 ms here, so 50 000 of them take about twelve minutes. Not 500 000 short ones (100 sweeps, as
# these tests had it): the fixed cost of an SA call, result arrays of the size asked for, is not
# interruptible and grows with the reads asked for, 0.5 s for 500 000 reads on this machine and 1.3-1.5 s
# with six busy processes beside it, against 0.07 s (0.2 s loaded) for 50 000. The wall-clock bounds
# below then tested the load of the machine: they failed in 3 of 10 and 5 of 15 runs under six
# busy-loop processes.
LONG_READS, LONG_SWEEPS = 50_000, 10_000


def test_the_sa_batch_is_cut_by_the_interrupt_and_returns_the_reads_done():
    problem = _problem(K=4)
    sampler = SABatches(problem, 4, num_reads_b=LONG_READS, num_sweeps=LONG_SWEEPS)
    sampler.prepare()
    deadline = time.perf_counter() + 0.25
    t0 = time.perf_counter()
    batch = sampler.sample(10026, interrupt=lambda: time.perf_counter() >= deadline, time_left_s=0.25)
    assert batch.interrupted and 0 < batch.n_samples < LONG_READS
    assert time.perf_counter() - t0 < 0.25 + 2.0  # the whole batch would take minutes


def test_the_swap_sa_batch_is_cut_by_the_interrupt():
    problem = _problem(K=4)
    sampler = SwapSABatches(problem, 4, num_reads_b=10, num_sweeps=1_000_000)
    sampler.prepare()
    deadline = time.perf_counter() + 0.25
    t0 = time.perf_counter()
    batch = sampler.sample(10027, interrupt=lambda: time.perf_counter() >= deadline, time_left_s=0.25)
    assert batch.interrupted and 0 < batch.info["sweeps_done"] < 1_000_000
    assert all(len(s) == 4 for s in batch.subsets)
    assert time.perf_counter() - t0 < 0.25 + 1.0


def test_a_run_with_a_batch_longer_than_the_budget_stops_in_time():
    """Every batch asks for far more than the budget allows, so each is cut by the interrupt and the
    run ends soon after the budget. How many batches that makes is timing, not asserted: the
    refinement reserve is estimated from one measurement (the start's), and what is left of it after
    the refinement starts one more, tiny batch (cut at once, with one read: it costs the fixed cost
    of its call, about 1 µs per read asked for: 85 ms for 50 000, a millisecond for the 1000 of the
    tuning space). With the 500 000 short reads of the first version it never happened (27 of 27
    runs had one batch): the fixed cost of the first call alone ran past the budget."""
    problem = _problem(K=4)
    start = common_start(problem)
    budget = 0.4
    t0 = time.perf_counter()
    sol = anytime_sa(
        problem, 4, num_reads_b=LONG_READS, num_sweeps=LONG_SWEEPS, budget_s=budget, seed=10028,
        start=start, keep_batch_log=True,
    )
    assert time.perf_counter() - t0 < budget + 2.0  # the first batch alone would take minutes
    meta = sol.solver_meta
    log = meta["batches"]
    assert meta["stop"] == "budget"
    assert log[0]["interrupted"] is True and 0 < log[0]["n_samples"] < LONG_READS
    assert meta["interrupted_batches"] == meta["n_batches"] == len(log)  # every batch is cut


def test_tabu_runs_deterministic_batches_and_guards_the_first_one_with_a_timer():
    problem = _problem(K=4)
    sampler = TabuBatches(problem, 4, num_restarts_b=2, num_reads_b=3, tenure=8)
    sampler.prepare()
    first = sampler.sample(10029, interrupt=lambda: False, time_left_s=30.0)
    assert first.info["num_reads"] == 3 and first.info["num_restarts"] == 2
    assert first.info["safety_timeout_ms"] == 10_000  # 30 s / 3 reads: no estimate yet
    second = sampler.sample(10030, interrupt=lambda: False, time_left_s=30.0)
    assert second.info["safety_timeout_ms"] is None  # a full batch with an estimate: no timer
    assert second.info["num_restarts"] == 2
    again = TabuBatches(problem, 4, num_restarts_b=2, num_reads_b=3, tenure=8)
    again.prepare()
    repeat = again.sample(10029, interrupt=lambda: False, time_left_s=30.0)
    assert repeat.subsets == first.subsets
    np.testing.assert_array_equal(repeat.energies, first.energies)


def test_a_batch_cut_by_the_safeguard_timer_does_not_set_the_time_per_search():
    problem = _problem(K=4)
    sampler = TabuBatches(problem, 4, num_restarts_b=2000, num_reads_b=3, tenure=8)
    sampler.prepare()
    cut = sampler.sample(10031, interrupt=lambda: False, time_left_s=0.006)  # 2 ms per read
    assert cut.interrupted and cut.info["num_restarts"] == 2000
    assert sampler._per_search_s is None  # a timer-cut batch says nothing about a full one


def test_tabu_shrinks_the_tail_batch_to_the_time_left():
    problem = _problem(K=4)
    sampler = TabuBatches(problem, 4, num_restarts_b=40, num_reads_b=5, tenure=8)
    sampler.prepare()
    full = sampler.sample(10031, interrupt=lambda: False, time_left_s=60.0)
    per_search = full.sample_s / (5 * 41)
    assert full.info["num_restarts"] == 40
    tail = sampler.sample(10032, interrupt=lambda: False, time_left_s=0.4 * 5 * 41 * per_search)
    assert tail.info["num_restarts"] < 40 or tail.info["num_reads"] < 5
    assert tail.info["safety_timeout_ms"] is not None
    assert tail.sample_s < 5 * 41 * per_search  # shorter than a full batch


@pytest.mark.parametrize(
    "factory",
    [
        lambda p: SABatches(p, 4, num_reads_b=0, num_sweeps=10),
        lambda p: SABatches(p, 4, num_reads_b=10, num_sweeps=0),
        lambda p: TabuBatches(p, 4, num_restarts_b=-1, num_reads_b=3, tenure=8),
        lambda p: TabuBatches(p, 4, num_restarts_b=1, num_reads_b=0, tenure=8),
        lambda p: SwapSABatches(p, 4, num_reads_b=0, num_sweeps=10),
        lambda p: SwapSABatches(p, 4, num_reads_b=10, num_sweeps=0),
        lambda p: SABatches(p, 0, num_reads_b=10, num_sweeps=10),
    ],
)
def test_samplers_reject_invalid_parameters(factory):
    with pytest.raises(ValueError):
        factory(_problem(K=4))


def test_the_pool_keeps_the_common_start_from_a_real_sampler(monkeypatch):
    problem = _problem(K=4)
    start = common_start(problem)
    refined = _spy_refinements(monkeypatch)
    anytime_swap_sa(
        problem, 4, num_reads_b=12, num_sweeps=25, budget_s=1e6, seed=10033, start=start,
        max_batches=2, m_b=50,
    )
    assert start.subset not in refined
    assert len(set(refined)) == len(refined)
