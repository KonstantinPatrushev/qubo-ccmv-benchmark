"""Anytime outer loop of the QUBO heuristics (ANALYSIS_PLAN §6 "Внешний цикл", N6; §5, N5).

A heuristic is stopped by the clock, not by a fixed number of reads: it repeats **batches** until
the end-to-end budget is exhausted. One batch is

1. a call of the sampler of the binary-inclusion QUBO — SA: ``num_reads_b`` reads; Tabu:
   ``num_reads_b`` reads of ``num_restarts_b`` restarts in the deterministic mode (R12); swap-SA:
   ``num_reads_b`` chains — returning its unique size-K subsets in ascending surrogate energy;
2. the refinement of the subsets of the batch that are **new** to the run, in that order, at most
   ``m_b`` of them, by the refinement shared with every other method
   (:func:`cif.classical.refine.refine_problem`, audit RF-1).

The pool is cumulative and deduplicated: a subset is refined once per run. A subset cut off by
``m_b`` was not refined and is not in the pool, so it is eligible again if a later batch returns
it. The common start (:mod:`cif.heuristics.start`) is in the pool from the beginning.

**Clock.** The trajectory times are seconds on the common end-to-end clock of the budget (§5): the
construction time of the start is already spent at the beginning of the loop, so a start with
``time_s = 0.8`` means the loop starts at 0.8 s and ``budget_s`` is the total, as for SCIP and local
search. Building the QUBO counts. The trajectory (a list of ``(t, f)``, input of
:mod:`cif.metrics.anytime`) has a point at the end of every batch that improved the incumbent
(§5: "после каждого пакета"), the first one being the start.

**Budget (§6).** A batch that does not fit into the remaining budget is cut: SA and swap-SA by
their ``interrupt_function`` (granularity: a read for SA, a sweep for swap-SA), Tabu by fewer
restarts per read (:func:`plan_tabu_batch`) from the measured time per search, with a per-read
timer as a safeguard on a batch that was shortened or has no estimate yet. The sampler is stopped
early enough to leave time to refine its batch: ``m_b`` refinements at the measured mean time per
refinement, at most a fifth of what is left. No refinement starts after the budget. The last batch
may overshoot it by the granularity of the cut.

**Reproducibility.** Batch ``b`` is seeded with a 32-bit number drawn from
``SeedSequence([seed, b])`` (§3). With ``max_batches`` instead of a binding clock (the RQ1 mode: one
batch, no outer loop) a run is exactly reproducible; with the clock it is reproducible batch by
batch, while the number of batches depends on the machine.

**Cost.** ``solver_meta`` reports the time of the sampler (``sample_s``), of decoding its output
into subsets (``decode_s``), of the refinements (``refine_s``) and of the rest of the loop
(``overhead_s``), and ``build_s`` for the QUBO (N5: the cost of the refinement is reported).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from cif.classical.refine import refine_problem
from cif.heuristics.pipeline import check_feasible
from cif.heuristics.start import CommonStart
from cif.heuristics.swap_sa import sample_swap_sa
from cif.problem import PortfolioProblem, Solution
from cif.quantum.neal_sampler import sample_tabu
from cif.qubo.builder import SELECTION_PENALTY_MULTIPLIER, build_selection_bqm

__all__ = [
    "ANYTIME_SOLVERS",
    "Batch",
    "BatchSampler",
    "SABatches",
    "SwapSABatches",
    "TabuBatches",
    "anytime_sa",
    "anytime_swap_sa",
    "anytime_tabu",
    "plan_tabu_batch",
    "run_anytime",
]

#: Longest per-read timer handed to Tabu (milliseconds): the sampler takes a 32-bit integer.
_MAX_TIMEOUT_MS = 2**31 - 1
#: Share of the remaining budget that may be kept back to refine the last batch.
_MAX_RESERVE_SHARE = 0.2


@dataclass
class Batch:
    """What one sampler call produced.

    ``subsets`` / ``energies`` — the unique feasible (size-K) subsets, ascending in surrogate
    energy; ``n_samples`` — reads returned (fewer than requested if the batch was cut),
    ``n_feasible`` — reads with exactly K assets; ``sample_s`` — time in the sampler,
    ``decode_s`` — time to turn its output into the subsets; ``interrupted`` — the batch was cut
    short by the budget; ``info`` — sampler specifics (JSON-safe).
    """

    subsets: list[tuple[int, ...]]
    energies: np.ndarray
    n_samples: int
    n_feasible: int
    sample_s: float
    interrupted: bool = False
    decode_s: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


class BatchSampler(Protocol):
    """A sampler adapter: ``prepare`` builds what is built once (inside the timed region);
    ``sample`` runs one batch, polling ``interrupt`` if it can be cut and given ``time_left_s`` of
    budget for it (the part of the remaining budget not kept for refinement)."""

    label: str

    def prepare(self) -> None: ...

    def sample(
        self, seed: int, *, interrupt: Callable[[], bool], time_left_s: float
    ) -> Batch: ...


def _batch_seed(seed: int, index: int) -> int:
    """Seed of batch ``index``: ``SeedSequence`` entropy from the run seed and the index, in
    ``[0, 2**31)`` — ``dwave-samplers`` reject larger seeds, whatever their error message says."""
    return int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0]) & 0x7FFFFFFF


def _unique_feasible(
    sampleset: dimod.SampleSet, n: int, K: int
) -> tuple[list[tuple[int, ...]], np.ndarray, int]:
    """Unique size-K subsets of a binary-inclusion sample set by ascending energy, with the number
    of feasible reads. For |S| = K the penalty term vanishes, so the sampler energy of a read is the
    surrogate energy of its subset; equal energies keep the lexicographic order of the subsets."""
    record = sampleset.record
    position = {var: k for k, var in enumerate(sampleset.variables)}
    columns = [position[i] for i in range(n)]
    held = np.asarray(record.sample)[:, columns] == 1
    energy = np.asarray(record.energy, dtype=float)
    feasible = np.flatnonzero(held.sum(axis=1) == K)
    if feasible.size == 0:
        return [], np.empty(0), 0
    unique, first = np.unique(held[feasible], axis=0, return_index=True)
    unique_energy = energy[feasible][first]
    order = np.argsort(unique_energy, kind="stable")
    subsets = [tuple(np.flatnonzero(unique[i]).tolist()) for i in order]
    return subsets, unique_energy[order], int(feasible.size)


def plan_tabu_batch(
    reads: int, restarts: int, per_search_s: float | None, time_left_s: float
) -> tuple[int, int]:
    """``(reads, restarts)`` of a Tabu batch that fits ``time_left_s`` (§6: fewer restarts).

    A read is one simple tabu search plus ``restarts`` restarts, ``reads · (1 + restarts)``
    searches in all, each taking about ``per_search_s``. A batch that fits, or has no estimate yet
    (``None``), is left alone. Otherwise the restarts per read are cut first, then the reads (to
    simple searches); ``(0, 0)`` if not even one search fits. The planned batch always fits the
    estimate.
    """
    if time_left_s <= 0.0:
        return 0, 0
    if per_search_s is None or reads * (1 + restarts) * per_search_s <= time_left_s:
        return reads, restarts
    searches = int(time_left_s / per_search_s)
    if searches >= reads:
        return reads, searches // reads - 1
    return searches, 0  # fewer reads than planned; (0, 0) if not even one search fits


class _SelectionQUBO:
    """The binary-inclusion QUBO of SA and Tabu, built once per run inside the timed region."""

    def __init__(self, problem: PortfolioProblem, K: int, penalty_multiplier: float) -> None:
        if not 1 <= int(K) <= problem.n:
            raise ValueError(f"cardinality must be in [1, N={problem.n}], got {K}")
        if not penalty_multiplier > 0.0:
            raise ValueError(f"penalty_multiplier must be > 0, got {penalty_multiplier}")
        self.problem = problem
        self.K = int(K)
        self.penalty_multiplier = float(penalty_multiplier)
        self.bqm: dimod.BinaryQuadraticModel | None = None

    def prepare(self) -> None:
        self.bqm = build_selection_bqm(
            self.problem, self.K, penalty_multiplier=self.penalty_multiplier
        )


class SABatches(_SelectionQUBO):
    """Simulated annealing (``dwave-samplers``) on the selection QUBO: ``num_reads_b`` reads of
    ``num_sweeps`` sweeps per batch, cut by the sampler's ``interrupt_function`` between reads."""

    def __init__(
        self,
        problem: PortfolioProblem,
        K: int,
        *,
        num_reads_b: int,
        num_sweeps: int,
        penalty_multiplier: float = SELECTION_PENALTY_MULTIPLIER,
        beta_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__(problem, K, penalty_multiplier)
        if num_reads_b < 1 or num_sweeps < 1:
            raise ValueError(
                f"num_reads_b and num_sweeps must be >= 1, got {num_reads_b}, {num_sweeps}"
            )
        self.num_reads_b = int(num_reads_b)
        self.num_sweeps = int(num_sweeps)
        self.beta_range = beta_range
        self.label = f"anytime_sa[{self.num_reads_b}x{self.num_sweeps}]"
        self._sampler = SimulatedAnnealingSampler()

    def sample(self, seed: int, *, interrupt: Callable[[], bool], time_left_s: float) -> Batch:
        t0 = time.perf_counter()
        sampleset = self._sampler.sample(
            self.bqm,
            num_reads=self.num_reads_b,
            num_sweeps=self.num_sweeps,
            seed=seed,
            beta_range=self.beta_range,
            interrupt_function=interrupt,
        )
        sample_s = time.perf_counter() - t0
        subsets, energies, n_feasible = _unique_feasible(sampleset, self.problem.n, self.K)
        n_reads = len(sampleset)
        return Batch(
            subsets,
            energies,
            n_reads,
            n_feasible,
            sample_s,
            interrupted=n_reads < self.num_reads_b,
            decode_s=time.perf_counter() - t0 - sample_s,
            info={"num_reads": n_reads, "num_sweeps": self.num_sweeps},
        )


class TabuBatches(_SelectionQUBO):
    """Tabu search (``dwave-samplers``) on the selection QUBO in the deterministic mode (R12):
    ``num_reads_b`` reads of ``num_restarts_b`` restarts each, no clock inside a batch.

    The sampler cannot be interrupted, so the last batch is shortened instead
    (:func:`plan_tabu_batch`) from the time per search measured on full batches. A batch with no estimate yet, or a shortened
    one, also carries a per-read timer of ``time_left / reads`` as a safeguard (``info`` reports
    it); an unguarded batch is exactly reproducible for a seed. ``num_reads_b`` is not fixed by
    the plan (it lists ``num_restarts_b``, ``tenure``, the penalty multiplier and ``m_b`` as tuned).
    """

    def __init__(
        self,
        problem: PortfolioProblem,
        K: int,
        *,
        num_restarts_b: int,
        num_reads_b: int = 10,
        tenure: int = 20,
        penalty_multiplier: float = SELECTION_PENALTY_MULTIPLIER,
    ) -> None:
        super().__init__(problem, K, penalty_multiplier)
        if num_restarts_b < 0 or num_reads_b < 1 or tenure < 0:
            raise ValueError(
                "need num_restarts_b >= 0, num_reads_b >= 1 and tenure >= 0, got "
                f"{num_restarts_b}, {num_reads_b}, {tenure}"
            )
        self.num_restarts_b = int(num_restarts_b)
        self.num_reads_b = int(num_reads_b)
        self.tenure = int(tenure)
        self.label = (
            f"anytime_tabu[{self.num_reads_b}r/{self.num_restarts_b}rs/tenure={self.tenure}]"
        )
        self._per_search_s: float | None = None

    def sample(self, seed: int, *, interrupt: Callable[[], bool], time_left_s: float) -> Batch:
        reads, restarts = plan_tabu_batch(
            self.num_reads_b, self.num_restarts_b, self._per_search_s, time_left_s
        )
        if reads == 0:  # not even one search fits
            info = {"num_reads": 0, "num_restarts": 0, "safety_timeout_ms": None}
            return Batch([], np.empty(0), 0, 0, 0.0, interrupted=True, info=info)
        full = (reads, restarts) == (self.num_reads_b, self.num_restarts_b)
        guarded = not full or self._per_search_s is None
        timeout_ms = (
            min(_MAX_TIMEOUT_MS, max(1, int(1000.0 * time_left_s / reads))) if guarded else None
        )
        t0 = time.perf_counter()
        sampleset = sample_tabu(
            self.bqm,
            num_reads=reads,
            tenure=self.tenure,
            seed=seed,
            timeout_ms=timeout_ms,
            num_restarts=restarts,
        )
        sample_s = time.perf_counter() - t0
        subsets, energies, n_feasible = _unique_feasible(sampleset, self.problem.n, self.K)
        done = np.asarray(sampleset.record.num_restarts)
        cut = bool(np.any(done < restarts))
        if full and not cut:
            self._per_search_s = sample_s / (reads * (1 + restarts))
        return Batch(
            subsets,
            energies,
            len(sampleset),
            n_feasible,
            sample_s,
            interrupted=cut or not full,
            decode_s=time.perf_counter() - t0 - sample_s,
            info={
                "num_reads": reads,
                "num_restarts": restarts,
                "restarts_mean": float(done.mean()),
                "safety_timeout_ms": timeout_ms,
            },
        )


class SwapSABatches:
    """Swap-SA over K-subsets (:func:`cif.heuristics.swap_sa.sample_swap_sa`): ``num_reads_b``
    chains of ``num_sweeps`` sweeps per batch, cut between sweeps by the ``interrupt_function``;
    the hot temperature is calibrated in every batch (``accept_start``)."""

    def __init__(
        self,
        problem: PortfolioProblem,
        K: int,
        *,
        num_reads_b: int,
        num_sweeps: int,
        accept_start: float = 0.8,
        cold_ratio: float = 0.01,
    ) -> None:
        if not 1 <= int(K) <= problem.n:
            raise ValueError(f"cardinality must be in [1, N={problem.n}], got {K}")
        if num_reads_b < 1 or num_sweeps < 1:
            raise ValueError(
                f"num_reads_b and num_sweeps must be >= 1, got {num_reads_b}, {num_sweeps}"
            )
        self.problem = problem
        self.K = int(K)
        self.num_reads_b = int(num_reads_b)
        self.num_sweeps = int(num_sweeps)
        self.accept_start = float(accept_start)
        self.cold_ratio = float(cold_ratio)
        self.label = f"anytime_swap_sa[{self.num_reads_b}x{self.num_sweeps}]"

    def prepare(self) -> None:  # the surrogate is rebuilt per call, in O(N) beyond the data
        return None

    def sample(self, seed: int, *, interrupt: Callable[[], bool], time_left_s: float) -> Batch:
        res = sample_swap_sa(
            self.problem,
            self.K,
            num_reads=self.num_reads_b,
            num_sweeps=self.num_sweeps,
            seed=seed,
            accept_start=self.accept_start,
            cold_ratio=self.cold_ratio,
            interrupt_function=interrupt,
        )
        return Batch(
            res.subsets,
            res.energies,
            res.num_reads,
            res.num_reads,  # every chain holds exactly K assets
            res.sample_wall_s,
            interrupted=res.interrupted,
            info={
                "num_reads": res.num_reads,
                "num_sweeps": res.num_sweeps,
                "sweeps_done": res.sweeps_done,
                "t_hot": res.t_hot,
                "t_cold": res.t_cold,
                "accept_rate": res.accept_rate,
            },
        )


def run_anytime(
    problem: PortfolioProblem,
    K: int,
    sampler: BatchSampler,
    *,
    budget_s: float,
    seed: int,
    m_b: int = 50,
    start: CommonStart | None = None,
    card_mode: str = "le",
    max_batches: int | None = None,
    keep_batch_log: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> Solution:
    """Run ``sampler`` in batches until ``budget_s`` of end-to-end time is exhausted.

    Parameters
    ----------
    budget_s:
        Total budget on the common clock, the construction time of ``start`` included (module
        docstring). A start that already used it leaves no batch.
    m_b:
        At most this many new subsets are refined per batch, in surrogate-energy order.
    start:
        The common start: first incumbent, first trajectory point and member of the pool. A start
        with an infinite objective (failed construction) is not used, but its time is spent.
    card_mode:
        ``"le"`` or ``"eq"``, as in :func:`cif.classical.refine.refine_problem`; ``"le"`` needs
        ``w_min = 0``.
    max_batches:
        Stop after this many batches whatever the clock says (RQ1 uses one batch; tests use a few).
    keep_batch_log:
        Keep one dict per batch in ``solver_meta["batches"]`` (can be long for thousands of
        batches; the totals are always kept).
    clock:
        Wall clock, ``time.perf_counter``; replaceable to test the budget logic without sleeping.

    Returns
    -------
    Solution
        The best portfolio of the pool (``objective = +inf``, zero weights and ``feasible=False``
        if the pool is empty). ``wall_time_s`` is the end of the loop on the common clock.
        ``solver_meta``: ``trajectory`` (list of ``(t, f)``), ``subset``, ``n_batches``, ``stop``
        (``"budget"`` or ``"max_batches"``), the counts ``n_samples``, ``n_feasible``,
        ``n_unique`` (unique subsets returned, summed over batches), ``n_new`` (not yet in the
        pool), ``n_refined``, ``pool_size``, ``interrupted_batches``, the times ``build_s``,
        ``sample_s``, ``decode_s``, ``refine_s``, ``overhead_s``, ``first_own_s`` (end of the first
        batch that refined anything: the delay until the method's own first solution, §5),
        ``first_own_f`` (the best value refined in that batch: the value of that solution, which the
        start may well beat) and the inputs.
    """
    K = int(K)
    m_b = int(m_b)
    if not 1 <= K <= problem.n:
        raise ValueError(f"cardinality must be in [1, N={problem.n}], got {K}")
    if not (math.isfinite(budget_s) and budget_s > 0.0):
        raise ValueError(f"budget_s must be a finite positive number, got {budget_s}")
    if m_b < 1:
        raise ValueError(f"m_b must be >= 1, got {m_b}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if max_batches is not None and max_batches < 1:
        raise ValueError(f"max_batches must be >= 1, got {max_batches}")
    if card_mode not in ("le", "eq"):
        raise ValueError(f"card_mode must be 'le' or 'eq', got {card_mode!r}")
    if card_mode == "le" and problem.w_min > 0.0:
        raise ValueError(
            "card_mode='le' with w_min > 0 makes the refinement combinatorial; use card_mode='eq' "
            "(every asset of the subset held) or a problem with w_min = 0"
        )
    has_start = start is not None and math.isfinite(start.objective)
    if has_start and len(start.subset) != K:
        raise ValueError(f"the common start holds {len(start.subset)} assets, expected K = {K}")

    t0 = clock()
    offset = float(start.time_s) if start is not None else 0.0

    def now() -> float:
        return offset + (clock() - t0)

    seen: set[tuple[int, ...]] = set()
    trajectory: list[tuple[float, float]] = []
    best_weights: np.ndarray | None = None
    best_objective = math.inf
    best_subset: tuple[int, ...] | None = None
    refine_estimate = 0.0
    if has_start:
        seen.add(tuple(start.subset))
        best_weights, best_objective = start.weights, start.objective
        best_subset = tuple(start.subset)
        trajectory.append((float(start.time_s), float(start.objective)))
        refine_estimate = float(start.refine_s)

    sampler.prepare()
    build_s = clock() - t0

    log: list[dict[str, Any]] = []
    totals = {
        "n_samples": 0, "n_feasible": 0, "n_unique": 0, "n_new": 0, "n_refined": 0,
        "interrupted_batches": 0, "sample_s": 0.0, "decode_s": 0.0, "refine_s": 0.0,
        "overhead_s": 0.0,
    }  # fmt: skip
    first_own_s: float | None = None
    first_own_f: float | None = None
    refine_calls = 0
    refine_total_s = 0.0
    n_batches = 0
    while max_batches is None or n_batches < max_batches:
        t_start = now()
        if t_start >= budget_s:
            break
        left = budget_s - t_start
        reserve = min(m_b * refine_estimate, _MAX_RESERVE_SHARE * left)
        stop_at = budget_s - reserve
        batch = sampler.sample(
            _batch_seed(seed, n_batches),
            interrupt=lambda stop_at=stop_at: now() >= stop_at,
            time_left_s=left - reserve,
        )
        new = [subset for subset in batch.subsets if subset not in seen][:m_b]
        refine_s = 0.0
        n_refined = 0
        improved = False
        batch_best = math.inf
        for subset in new:
            if now() >= budget_s:
                break
            r0 = clock()
            refined = refine_problem(problem, subset, card_mode)
            refine_s += clock() - r0
            n_refined += 1
            seen.add(subset)
            batch_best = min(batch_best, refined.objective)
            if refined.objective < best_objective:  # +inf if the refinement failed
                best_weights, best_objective = refined.weights, refined.objective
                best_subset, improved = subset, True
        t_end = now()
        if improved:
            trajectory.append((float(t_end), float(best_objective)))
        if n_refined and first_own_s is None:
            first_own_s = float(t_end)
            first_own_f = float(batch_best) if math.isfinite(batch_best) else None
        refine_calls += n_refined
        refine_total_s += refine_s
        if refine_calls:
            refine_estimate = refine_total_s / refine_calls
        overhead_s = (t_end - t_start) - batch.sample_s - batch.decode_s - refine_s
        for key, value in (
            ("n_samples", batch.n_samples), ("n_feasible", batch.n_feasible),
            ("n_unique", len(batch.subsets)), ("n_new", len(new)), ("n_refined", n_refined),
            ("interrupted_batches", int(batch.interrupted)), ("sample_s", batch.sample_s),
            ("decode_s", batch.decode_s), ("refine_s", refine_s), ("overhead_s", overhead_s),
        ):  # fmt: skip
            totals[key] += value
        if keep_batch_log:
            log.append(
                {
                    "index": n_batches, "t_start": t_start, "t_end": t_end,
                    "n_samples": batch.n_samples, "n_feasible": batch.n_feasible,
                    "n_unique": len(batch.subsets), "n_new": len(new), "n_refined": n_refined,
                    "sample_s": batch.sample_s, "decode_s": batch.decode_s, "refine_s": refine_s,
                    "overhead_s": overhead_s, "improved": improved,
                    "interrupted": batch.interrupted, "info": batch.info,
                }
            )  # fmt: skip
        n_batches += 1

    wall = now()
    if best_weights is None:
        weights = np.zeros(problem.n)
        feasible = False
    else:
        weights = np.array(best_weights, dtype=float)  # a copy: the start's array stays its own
        feasible = bool(check_feasible(weights, problem) and np.count_nonzero(weights > 1e-6) <= K)
    meta: dict[str, Any] = {
        "trajectory": trajectory,
        "subset": list(best_subset) if best_subset is not None else None,
        "n_batches": n_batches,
        "stop": "max_batches" if max_batches is not None and n_batches >= max_batches else "budget",
        "pool_size": len(seen),
        "build_s": build_s,
        "first_own_s": first_own_s,
        "first_own_f": first_own_f,
        "start_used": bool(has_start),
        "budget_s": budget_s,
        "m_b": m_b,
        "seed": seed,
        "card_mode": card_mode,
        "sampler": sampler.label,
        **totals,
    }
    if keep_batch_log:
        meta["batches"] = log
    return Solution(
        weights=weights,
        objective=float(best_objective),
        feasible=feasible,
        wall_time_s=wall,
        solver=sampler.label,
        solver_meta=meta,
    )


def anytime_sa(
    problem: PortfolioProblem,
    K: int,
    *,
    num_reads_b: int,
    num_sweeps: int,
    penalty_multiplier: float = SELECTION_PENALTY_MULTIPLIER,
    beta_range: tuple[float, float] | None = None,
    **run: Any,
) -> Solution:
    """:func:`run_anytime` with simulated annealing (:class:`SABatches`); ``run`` are its
    keyword arguments (``budget_s``, ``seed``, ``m_b``, ``start``, ``card_mode``, ...)."""
    sampler = SABatches(
        problem, K, num_reads_b=num_reads_b, num_sweeps=num_sweeps,
        penalty_multiplier=penalty_multiplier, beta_range=beta_range,
    )  # fmt: skip
    return run_anytime(problem, K, sampler, **run)


def anytime_tabu(
    problem: PortfolioProblem,
    K: int,
    *,
    num_restarts_b: int,
    num_reads_b: int = 10,
    tenure: int = 20,
    penalty_multiplier: float = SELECTION_PENALTY_MULTIPLIER,
    **run: Any,
) -> Solution:
    """:func:`run_anytime` with Tabu search (:class:`TabuBatches`)."""
    sampler = TabuBatches(
        problem, K, num_restarts_b=num_restarts_b, num_reads_b=num_reads_b, tenure=tenure,
        penalty_multiplier=penalty_multiplier,
    )  # fmt: skip
    return run_anytime(problem, K, sampler, **run)


def anytime_swap_sa(
    problem: PortfolioProblem,
    K: int,
    *,
    num_reads_b: int,
    num_sweeps: int,
    accept_start: float = 0.8,
    cold_ratio: float = 0.01,
    **run: Any,
) -> Solution:
    """:func:`run_anytime` with swap-SA (:class:`SwapSABatches`)."""
    sampler = SwapSABatches(
        problem, K, num_reads_b=num_reads_b, num_sweeps=num_sweeps,
        accept_start=accept_start, cold_ratio=cold_ratio,
    )  # fmt: skip
    return run_anytime(problem, K, sampler, **run)


#: The anytime heuristics by the name used in configs (§4, §6).
ANYTIME_SOLVERS: dict[str, Callable[..., Solution]] = {
    "sa": anytime_sa,
    "tabu": anytime_tabu,
    "swap_sa": anytime_swap_sa,
}
