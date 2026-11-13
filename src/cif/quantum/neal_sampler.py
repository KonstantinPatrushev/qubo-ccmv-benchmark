"""Wrapper around :class:`dwave.samplers.SimulatedAnnealingSampler`.

The D-Wave ``neal`` sampler runs classical simulated annealing on a BQM, entirely on a CPU (no
API keys, no quotas). In this project it is one of the QUBO heuristics compared against exact
MIQP solvers. It is not quantum-inspired: the legacy report called it so, the study does not
(R11).
"""

from __future__ import annotations

import inspect
import time
from typing import Literal

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler

from cif.heuristics.pipeline import check_feasible as _check_feasible
from cif.heuristics.pipeline import (
    iter_sampleset_subsets,
    legacy_turnover_cost,
    refine_candidates,
)
from cif.problem import PortfolioProblem, Solution
from cif.qubo.builder import (
    Penalties,
    build_bqm,
    build_selection_bqm,
    sample_to_weights,
    selection_bits_to_subset,
)
from cif.qubo.encoding import WeightEncoding

_TABU_SAMPLE_PARAMS = inspect.signature(TabuSampler.sample).parameters
TABU_DEFAULT_TIMEOUT_MS: int = _TABU_SAMPLE_PARAMS["timeout"].default
"""Per-read clock cap of ``TabuSampler`` in dwave-samplers (20 ms)."""
TABU_DEFAULT_NUM_RESTARTS: int = _TABU_SAMPLE_PARAMS["num_restarts"].default
"""Restart cap per read of ``TabuSampler`` (1e6, i.e. practically unlimited)."""


def solve_with_neal(
    problem: PortfolioProblem,
    encoding: WeightEncoding,
    penalties: Penalties,
    num_reads: int = 1000,
    num_sweeps: int = 1000,
    seed: int | None = None,
    beta_range: tuple[float, float] | None = None,
    return_all_solutions: bool = False,
) -> Solution | list[Solution]:
    """Run simulated annealing on a QUBO derived from ``problem``.

    Parameters
    ----------
    num_reads:
        Number of independent anneal runs. Each gives one candidate solution;
        the lowest-energy **feasible** sample is returned.
    num_sweeps:
        Monte Carlo sweeps per run. Default 1000 is a conservative sweet spot
        for N·(L+1) ≤ 1000 variables.
    seed:
        RNG seed for reproducibility.
    beta_range:
        Explicit inverse-temperature schedule bounds. ``None`` lets ``neal``
        choose based on the problem's coefficient magnitudes.
    return_all_solutions:
        If True, return a list of decoded solutions sorted by energy instead
        of a single best one.
    """
    bqm = build_bqm(problem, encoding, penalties)

    sampler = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_range=beta_range,
    )
    wall = time.perf_counter() - t0

    return _sampleset_to_solutions(
        sampleset=sampleset,
        problem=problem,
        encoding=encoding,
        wall_time_s=wall,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        return_all=return_all_solutions,
    )


def _sampleset_to_solutions(
    sampleset: dimod.SampleSet,
    problem: PortfolioProblem,
    encoding: WeightEncoding,
    wall_time_s: float,
    num_reads: int,
    num_sweeps: int,
    return_all: bool,
) -> Solution | list[Solution]:
    solutions: list[Solution] = []
    for record in sampleset.record:
        sample_dict = {
            var: int(record.sample[i])
            for i, var in enumerate(sampleset.variables)
        }
        weights = sample_to_weights(sample_dict, encoding)
        obj = problem.objective_value(weights)
        feasible = _check_feasible(weights, problem)
        solutions.append(
            Solution(
                weights=weights,
                objective=obj,
                feasible=bool(feasible),
                wall_time_s=wall_time_s / max(num_reads, 1),
                solver=f"neal/SA[{num_reads}x{num_sweeps}]",
                solver_meta={
                    "num_reads": num_reads,
                    "num_sweeps": num_sweeps,
                    "qubo_energy": float(record.energy),
                    "num_occurrences": int(record.num_occurrences),
                },
            )
        )

    if not solutions:
        raise RuntimeError("neal returned an empty sample set")

    if return_all:
        solutions.sort(key=lambda s: (not s.feasible, s.objective))
        return solutions

    feasible_solutions = [s for s in solutions if s.feasible]
    best = (
        min(feasible_solutions, key=lambda s: s.objective)
        if feasible_solutions
        else min(solutions, key=lambda s: s.objective)
    )
    best.wall_time_s = wall_time_s
    best.solver_meta = {
        **best.solver_meta,
        "n_samples": len(solutions),
        "n_feasible": len(feasible_solutions),
    }
    return best


def _resolve_turnover_cost(turnover_cost_bps: float, turnover_cost: float) -> float:
    """Cost per unit traded: the new ``turnover_cost`` or the legacy bps argument."""
    if turnover_cost_bps > 0 and turnover_cost > 0:
        raise ValueError("pass either turnover_cost_bps (legacy) or turnover_cost, not both")
    if turnover_cost_bps > 0:
        return legacy_turnover_cost(turnover_cost_bps)
    return float(turnover_cost)


def _penalty_kwargs(penalty_multiplier: float | None) -> dict:
    """Forward the multiplier only when given, so the builder default (R8) applies."""
    return {} if penalty_multiplier is None else {"penalty_multiplier": float(penalty_multiplier)}


def _best_effort_solution(
    sampleset: dimod.SampleSet,
    problem: PortfolioProblem,
    wall_time_s: float,
    solver: str,
    solver_meta: dict,
) -> Solution:
    """Fallback when no size-K subset could be refined: lowest-energy sample, equal weights."""
    first = sampleset.first
    sample_dict = {var: int(first.sample[var]) for var in sampleset.variables}
    subset_any = selection_bits_to_subset(sample_dict, problem.n)
    weights = np.zeros(problem.n, dtype=float)
    if subset_any:
        weights[list(subset_any)] = 1.0 / len(subset_any)
    return Solution(
        weights=weights,
        objective=problem.objective_value(weights),
        feasible=False,
        wall_time_s=wall_time_s,
        solver=solver,
        solver_meta={**solver_meta, "fallback_subset": list(subset_any)},
    )


def solve_with_neal_selection(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    num_reads: int = 500,
    num_sweeps: int = 500,
    seed: int | None = None,
    cardinality_penalty: float | None = None,
    beta_range: tuple[float, float] | None = None,
    top_subsets: int = 5,
    refine_continuous: bool = True,
    warm_start_subset: list[int] | tuple[int, ...] | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_cost_bps: float = 0.0,
    *,
    penalty_multiplier: float | None = None,
    turnover_cost: float = 0.0,
    pool_order: Literal["read", "energy"] = "read",
    turnover_in_refinement: bool = False,
) -> Solution:
    """Two-stage pipeline: ``neal`` picks a subset, then cvxpy refines weights.

    Stage 1: Build the binary-inclusion BQM (only ``N`` variables) and run
    simulated annealing. Collect the first ``top_subsets`` unique subsets of
    exactly ``K`` assets (see ``pool_order``).

    Stage 2: :func:`cif.heuristics.pipeline.refine_candidates` solves the
    continuous mean-variance subproblem on each candidate subset (few
    milliseconds per subset at ``N ≤ 100``) and returns the subset whose
    refinement has the lowest ranking objective.

    This is the usual way of using QUBO heuristics: delegate the hard
    combinatorial selection to the sampler, then do cheap convex
    post-processing. The two-stage timing is what we report.

    Parameters
    ----------
    cardinality_penalty:
        Absolute cardinality penalty ``λ_c``; has priority over
        ``penalty_multiplier``.
    penalty_multiplier:
        ``λ_c`` as a multiple of the objective scale
        (:func:`cif.qubo.builder.selection_penalty_scale`). ``None`` uses the
        builder default (100, R8).
    top_subsets:
        Pool size ``m`` of the refinement stage.
    pool_order:
        ``"read"`` (default, legacy): the pool is the first ``top_subsets``
        unique size-``K`` subsets in the order of the reads. ``neal`` does not
        sort its record by energy, so this is *not* the best ``top_subsets``.
        ``"energy"``: the pool is taken in ascending sampler energy.
    warm_start_subset:
        If provided, this subset is injected into the candidate list
        (alongside SA's findings) for continuous refinement. The comparison
        keeps ``warm_start_subset`` if it produces a better refined
        objective than any of SA's top-K — essential for walk-forward
        stability, where the previous period's subset is usually a strong
        baseline SA may not rediscover.
    prev_weights:
        If provided together with a positive trading cost, candidates are
        ranked by ``obj + cost · ||w - prev_weights||_1``. This makes the
        fold-to-fold comparison honest about transaction-cost drag.
    turnover_cost:
        Cost per unit traded (project convention, see
        :mod:`cif.heuristics.pipeline`).
    turnover_cost_bps:
        LEGACY. Per-rebalance cost in basis points of one-way turnover,
        ranking ``obj + bps · 0.5 · ||w - prev_weights||_1 / 1e4``; equal to
        ``turnover_cost = bps / 2e4``. Mutually exclusive with
        ``turnover_cost``.
    """
    K = cardinality if cardinality is not None else problem.cardinality
    if K is None:
        raise ValueError("cardinality must be provided either on the problem or explicitly")
    K = int(K)
    cost = _resolve_turnover_cost(turnover_cost_bps, turnover_cost)

    bqm = build_selection_bqm(
        problem,
        cardinality=K,
        cardinality_penalty=cardinality_penalty,
        **_penalty_kwargs(penalty_multiplier),
    )

    sampler = SimulatedAnnealingSampler()
    t0 = time.perf_counter()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        beta_range=beta_range,
    )
    sa_wall = time.perf_counter() - t0

    label = (
        f"neal_selection+cvxpy[{num_reads}x{num_sweeps}]"
        if refine_continuous
        else f"neal_selection[{num_reads}x{num_sweeps}]"
    )
    best, stats = refine_candidates(
        problem,
        iter_sampleset_subsets(sampleset, problem.n, order=pool_order),
        K,
        top_subsets,
        prev_weights=prev_weights,
        turnover_cost=cost,
        warm_start_subset=warm_start_subset,
        refine_continuous=refine_continuous,
        solver=label,
        turnover_in_refinement=turnover_in_refinement,
    )
    meta = {
        "sa_wall_s": sa_wall,
        "refine_wall_s": stats["refine_wall_s"],
        "n_candidates": stats["n_candidates"],
        "n_evaluated": stats["n_evaluated"],
        "m": stats["m"],
        "warm_start_in_pool": stats["warm_start_in_pool"],
        "warm_start_selected": stats["warm_start_selected"],
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "sample_best_energy": float(np.min(sampleset.record.energy)),
    }
    wall = sa_wall + stats["refine_wall_s"]
    if best is None:
        # Fallback: no SA sample had exactly K actives. Return best-effort.
        return _best_effort_solution(
            sampleset, problem, wall, f"neal_selection[{num_reads}x{num_sweeps}]", meta
        )

    best.wall_time_s = wall
    best.solver_meta.update(meta)
    return best


def _effective_tabu_tenure(tenure: int, num_variables: int) -> int:
    """Tabu requires 0 ≤ tenure < num_vars; cap to leave at least one free move."""
    return int(min(tenure, max(1, num_variables - 1)))


def sample_tabu(
    bqm: dimod.BinaryQuadraticModel,
    num_reads: int,
    tenure: int = 20,
    seed: int | None = None,
    timeout_ms: int | None = TABU_DEFAULT_TIMEOUT_MS,
    num_restarts: int | None = None,
) -> dimod.SampleSet:
    """Run ``TabuSampler`` with explicit control of how each read terminates.

    Two termination regimes (per read):

    * **timeout** (``timeout_ms`` is an int, the dwave-samplers default is 20):
      the multistart search stops at the wall-clock cap or after
      ``num_restarts`` restarts, whichever comes first. With the default
      restart cap (1e6) the clock is what stops it: the run costs about
      ``num_reads * timeout_ms`` regardless of ``N``, and for a fixed seed the
      result depends on machine speed and load (the sampler documents that
      results are then not reproducible).
    * **restarts** (``timeout_ms=None``): the sampler accepts ``timeout=None``
      (it is mapped internally to a negative value that disables the clock).
      Each read then runs exactly ``num_restarts`` restarts after the first
      simple tabu search and the result is deterministic for a seed.
      ``num_restarts`` must be finite: with the sampler default of 1e6 restarts
      the call would not return in practice.
    """
    if timeout_ms is None and num_restarts is None:
        raise ValueError(
            "timeout_ms=None disables the clock, so a finite num_restarts is required "
            f"(the sampler default is {TABU_DEFAULT_NUM_RESTARTS})"
        )
    kwargs: dict = {"timeout": None if timeout_ms is None else int(timeout_ms)}
    if num_restarts is not None:
        kwargs["num_restarts"] = int(num_restarts)
    return TabuSampler().sample(
        bqm,
        num_reads=num_reads,
        tenure=_effective_tabu_tenure(tenure, len(bqm.variables)),
        seed=seed,
        **kwargs,
    )


def solve_with_tabu_selection(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    num_reads: int = 200,
    tenure: int = 20,
    seed: int | None = None,
    cardinality_penalty: float | None = None,
    top_subsets: int = 50,
    refine_continuous: bool = True,
    warm_start_subset: list[int] | tuple[int, ...] | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_cost_bps: float = 0.0,
    *,
    timeout_ms: int | None = 20,
    num_restarts: int | None = None,
    penalty_multiplier: float | None = None,
    turnover_cost: float = 0.0,
    pool_order: Literal["read", "energy"] = "read",
    turnover_in_refinement: bool = False,
) -> Solution:
    """Two-stage pipeline using TabuSampler instead of simulated annealing.

    Same architecture as :func:`solve_with_neal_selection` (binary-inclusion
    QUBO → pool of K-subset candidates → continuous refinement via cvxpy on
    each, through :func:`cif.heuristics.pipeline.refine_candidates`), but the
    sampler is ``dwave.samplers.TabuSampler``.

    Parameters mirror :func:`solve_with_neal_selection` so the two functions
    are drop-in interchangeable in the experiment runners. Defaults reproduce
    the legacy call (``num_reads=200``, ``tenure=20``, pool 50).

    Parameters
    ----------
    timeout_ms:
        Per-read wall-clock cap in milliseconds (default 20, the sampler's
        own default). ``None`` switches to the deterministic regime.
    num_restarts:
        Restart cap per read. ``None`` keeps the sampler default (1e6). Must
        be a finite int when ``timeout_ms=None``.

    Notes
    -----
    R10: with ``timeout_ms=20`` every read is cut after about 20 ms. The
    near-constant total time (``≈ num_reads · 20 ms``) is a property of that
    cap, not evidence that the search converged, and the results for one seed
    are not reproducible. Use ``timeout_ms=None`` with a ``num_restarts`` for
    reproducible runs and for time comparisons across ``N``.

    ``solver_meta["termination"]`` is the regime that was requested
    (``"timeout"`` or ``"restarts"``); ``solver_meta["restarts_mean"]``, the
    restarts actually completed per read, shows whether the clock was binding
    (a mean below ``num_restarts`` means reads were cut by the timer).
    """
    K = cardinality if cardinality is not None else problem.cardinality
    if K is None:
        raise ValueError("cardinality must be provided either on the problem or explicitly")
    K = int(K)
    cost = _resolve_turnover_cost(turnover_cost_bps, turnover_cost)

    bqm = build_selection_bqm(
        problem,
        cardinality=K,
        cardinality_penalty=cardinality_penalty,
        **_penalty_kwargs(penalty_multiplier),
    )

    t0 = time.perf_counter()
    sampleset = sample_tabu(
        bqm,
        num_reads=num_reads,
        tenure=tenure,
        seed=seed,
        timeout_ms=timeout_ms,
        num_restarts=num_restarts,
    )
    sample_wall = time.perf_counter() - t0

    label = f"tabu_selection[{num_reads}r/tenure={tenure}]"
    if num_restarts is not None and timeout_ms is None:
        label = f"tabu_selection[{num_reads}r/tenure={tenure}/restarts={num_restarts}]"
    if refine_continuous:
        label = label.replace("tabu_selection", "tabu_selection+cvxpy", 1)
    best, stats = refine_candidates(
        problem,
        iter_sampleset_subsets(sampleset, problem.n, order=pool_order),
        K,
        top_subsets,
        prev_weights=prev_weights,
        turnover_cost=cost,
        warm_start_subset=warm_start_subset,
        refine_continuous=refine_continuous,
        solver=label,
        turnover_in_refinement=turnover_in_refinement,
    )
    meta = {
        "sample_wall_s": sample_wall,
        "refine_wall_s": stats["refine_wall_s"],
        "n_candidates": stats["n_candidates"],
        "n_evaluated": stats["n_evaluated"],
        "m": stats["m"],
        "warm_start_in_pool": stats["warm_start_in_pool"],
        "warm_start_selected": stats["warm_start_selected"],
        "num_reads": num_reads,
        "tenure": tenure,
        "tenure_effective": _effective_tabu_tenure(tenure, problem.n),
        "termination": "timeout" if timeout_ms is not None else "restarts",
        "timeout_ms": timeout_ms,
        "num_restarts": TABU_DEFAULT_NUM_RESTARTS if num_restarts is None else int(num_restarts),
        "restarts_mean": float(np.mean(sampleset.record.num_restarts)),
        "sample_best_energy": float(np.min(sampleset.record.energy)),
    }
    wall = sample_wall + stats["refine_wall_s"]
    if best is None:
        return _best_effort_solution(
            sampleset, problem, wall, f"tabu_selection[{num_reads}r/tenure={tenure}]", meta
        )

    best.wall_time_s = wall
    best.solver_meta.update(meta)
    return best
