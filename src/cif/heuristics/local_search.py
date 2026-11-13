"""Local search on the true objective: the direct competitor of the QUBO detour (ANALYSIS_PLAN §2
RQ2 / H2b, §4, §6; class of Di Gaspero et al. 2011).

There is no surrogate. A state is a K-subset ``S`` and its value is the true objective after the
shared refinement, ``F(S) = f(refine(S))`` (:func:`cif.classical.refine.refine_support`: one
implementation for every method, audit RF-1). The search is

1. **start**: the common start (:mod:`cif.heuristics.start`) as the first point; later restarts, and
   the first one if there is no start, begin from a **greedy construction** — add the asset whose
   refined value is best, ``K`` times. The restarts use a restricted candidate list (one of the
   ``rcl_size`` best at random) so that they differ; the first construction without a start is
   deterministic;
2. **descent**: one-asset swaps (leave ``i ∈ S``, enter ``j ∉ S``; ``K·(N − K)`` neighbours) with
   **first improvement** (the neighbours in random order, the first one that improves is taken) or
   **best improvement** (the whole neighbourhood, the best swap is taken), until no swap improves
   by more than ``1e-9·s_ref`` of the current portfolio (a local optimum);
3. **restarts** until their number (``num_restarts``, Appendix B: 1–200) or the wall-clock budget is
   exhausted, or until every K-subset has been evaluated.

Every subset is refined at most once per run (a memo of ``F`` and of its weights, ``K`` numbers per
subset). A support smaller than ``K`` whose box cannot reach the budget (``m·w_max < budget``:
fewer than 5 assets for ``u = 0.2``, fewer than 10 for ``u = 0.1``) is ranked with the box lifted to
the budget; the refinement of a K-subset always uses the true box, so an infeasible K is reported
as an infinite value, not hidden.

**Incumbent and trajectory (§5).** The incumbent is the best state the search has *taken*: the
trajectory has a point ``(t, f)`` after each accepted improving swap (§5: "после каждого
улучшающего обмена") and after a greedy construction that beats it. With first improvement the
swap is taken the moment it is evaluated; with best improvement at the end of the scan, so a scan
cut by the budget does not take its best candidate so far. The first point is the start, and the
times are on the common end-to-end clock (the construction time of the start is already spent, as
for the QUBO loop and SCIP). No refinement starts after the budget; the last one may end after it.

**Reproducibility.** Restart ``r`` draws from ``default_rng(SeedSequence([seed, r]))``, so its path
does not depend on the timing of earlier restarts; with a finite ``num_restarts`` and a budget
that does not bind, a run is exactly reproducible.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from cif.classical.refine import refine_support
from cif.heuristics.pipeline import check_feasible
from cif.heuristics.start import CommonStart
from cif.metrics.gap import reference_scale
from cif.problem import PortfolioProblem, Solution

__all__ = ["greedy_construction", "local_search"]

#: A swap is taken if it improves F by more than this share of s_ref of the current portfolio.
_MIN_GAIN = 1e-9
_IMPROVEMENTS = ("first", "best")
_CARD_MODES = ("le", "eq")


class _BudgetExhausted(Exception):
    """The wall-clock budget is spent: no further refinement starts."""


def _check_card_mode(problem: PortfolioProblem, card_mode: str) -> None:
    if card_mode not in _CARD_MODES:
        raise ValueError(f"card_mode must be one of {_CARD_MODES}, got {card_mode!r}")
    if card_mode == "le" and problem.w_min > 0.0:
        raise ValueError(
            "card_mode='le' with w_min > 0 makes the refinement combinatorial; use card_mode='eq' "
            "(every asset of the subset held) or a problem with w_min = 0"
        )


class _Search:
    """State of one run: memo, counters, incumbent and trajectory."""

    def __init__(
        self,
        problem: PortfolioProblem,
        K: int,
        *,
        card_mode: str,
        budget_s: float,
        offset: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.problem = problem
        self.K = K
        self.n = problem.n
        self.lower = problem.w_min if card_mode == "eq" else 0.0
        self.budget_s = budget_s
        self.offset = offset
        self.clock = clock
        self.t0 = clock()
        # support -> (F, weights on the support in the order of the sorted support)
        self.memo: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
        self.n_evals = 0
        self.n_hits = 0
        self.n_swaps = 0
        self.n_k_subsets = 0
        self.refine_s = 0.0
        self.first_own_s: float | None = None
        self.first_own_f: float | None = None
        self.constructing = False  # inside a greedy construction: its subsets are not solutions of its own
        self.trajectory: list[tuple[float, float]] = []
        self.best_f = math.inf
        self.best_w: np.ndarray | None = None
        self.best_subset: tuple[int, ...] | None = None

    def now(self) -> float:
        return self.offset + (self.clock() - self.t0)

    def full_weights(self, subset: tuple[int, ...], on_support: np.ndarray) -> np.ndarray:
        weights = np.zeros(self.n)
        weights[list(subset)] = on_support
        return weights

    def seed_with(self, start: CommonStart) -> None:
        """The common start: first incumbent, first trajectory point, a memo entry."""
        subset = tuple(start.subset)
        self.memo[subset] = (float(start.objective), np.asarray(start.weights)[list(subset)])
        self.n_k_subsets += 1
        self.best_f, self.best_w, self.best_subset = start.objective, start.weights, subset
        self.trajectory.append((float(start.time_s), float(start.objective)))

    def evaluate(self, subset: tuple[int, ...]) -> tuple[float, np.ndarray]:
        """``(F(S), weights on S)``, refined once per run. Raises when the budget is spent."""
        if self.now() >= self.budget_s:
            raise _BudgetExhausted
        known = self.memo.get(subset)
        if known is not None:
            self.n_hits += 1
            return known
        m = len(subset)
        upper = self.problem.w_max
        if m < self.K and m * upper < self.problem.budget - 1e-12:
            upper = self.problem.budget  # too small a support for the box: rank it without
        r0 = self.clock()
        result = refine_support(
            self.problem.mu,
            self.problem.sigma,
            self.problem.risk_aversion,
            subset,
            lower=self.lower,
            upper=upper,
            budget=self.problem.budget,
        )
        self.refine_s += self.clock() - r0
        self.n_evals += 1
        entry = (float(result.objective), np.asarray(result.weights)[list(subset)])
        self.memo[subset] = entry
        if self.first_own_s is None and m == self.K and not self.constructing and math.isfinite(entry[0]):
            # from a common start: the first K-subset the descent refined, whether or not it beats the start (§12 A4)
            self.first_own_s, self.first_own_f = self.now(), entry[0]
        if m == self.K:
            self.n_k_subsets += 1
        return entry

    def consider(self, subset: tuple[int, ...], f: float, weights: np.ndarray) -> None:
        """Offer a state the search has taken to the incumbent; a strict improvement is a point."""
        if f < self.best_f:
            t = float(self.now())
            self.best_f, self.best_w, self.best_subset = f, weights, subset
            self.trajectory.append((t, float(f)))
            if self.first_own_s is None:  # the first portfolio of its own it holds: the end of a construction (C-4)
                self.first_own_s, self.first_own_f = t, float(f)

    def greedy(self, rng: np.random.Generator | None, rcl_size: int) -> tuple[int, ...]:
        """Add, ``K`` times, the asset with the best refined value (one of the ``rcl_size`` best at
        random if ``rcl_size > 1``; ties go to the lower index)."""
        chosen: list[int] = []
        self.constructing = True
        try:
            for _ in range(self.K):
                taken = set(chosen)
                scored = [
                    (self.evaluate(tuple(sorted([*chosen, j])))[0], j)
                    for j in range(self.n)
                    if j not in taken
                ]
                scored.sort()
                pick = 0
                if rcl_size > 1:
                    assert rng is not None
                    pick = int(rng.integers(min(rcl_size, len(scored))))
                chosen.append(scored[pick][1])
        finally:
            self.constructing = False
        return tuple(sorted(chosen))

    def _first_improving(self, subset, f, inside, outside, tol, rng):
        held = set(subset)
        for p in rng.permutation(len(inside) * len(outside)):
            a, b = divmod(int(p), len(outside))
            candidate = tuple(sorted((held - {inside[a]}) | {outside[b]}))
            value, on_support = self.evaluate(candidate)
            if value < f - tol:
                return candidate, value, on_support
        return None

    def _best_improving(self, subset, f, inside, outside, tol):
        held = set(subset)
        best = None
        for i in inside:
            for j in outside:
                candidate = tuple(sorted((held - {i}) | {j}))
                value, on_support = self.evaluate(candidate)
                if value < f - tol and (best is None or value < best[1]):
                    best = (candidate, value, on_support)
        return best

    def descend(
        self,
        subset: tuple[int, ...],
        f: float,
        weights: np.ndarray,
        improvement: str,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], float, np.ndarray, bool]:
        """Swap descent; returns the final ``(subset, F, weights, is_local_optimum)``."""
        while math.isfinite(f):
            held = set(subset)
            inside = list(subset)
            outside = [j for j in range(self.n) if j not in held]
            tol = _MIN_GAIN * reference_scale(self.problem, weights)
            if improvement == "first":
                found = self._first_improving(subset, f, inside, outside, tol, rng)
            else:
                found = self._best_improving(subset, f, inside, outside, tol)
            if found is None:
                break
            subset, f, on_support = found
            weights = self.full_weights(subset, on_support)
            self.n_swaps += 1
            self.consider(subset, f, weights)
        return subset, f, weights, True


def greedy_construction(
    problem: PortfolioProblem,
    K: int,
    *,
    rng: np.random.Generator | None = None,
    rcl_size: int = 1,
    card_mode: str = "le",
) -> tuple[int, ...]:
    """The greedy K-subset of the local search: ``K`` times, add the asset with the best refined
    value of the support so far (the box is lifted while the support cannot reach the budget).

    ``rcl_size = 1`` is deterministic (ties to the lower index); a larger candidate list picks one
    of the best ``rcl_size`` at random and needs an explicit ``rng``. The cost is about ``N·K``
    refinements of supports of up to ``K`` assets.
    """
    K = int(K)
    if not 1 <= K <= problem.n:
        raise ValueError(f"cardinality must be in [1, N={problem.n}], got {K}")
    if rcl_size < 1:
        raise ValueError(f"rcl_size must be >= 1, got {rcl_size}")
    if rcl_size > 1 and rng is None:
        raise ValueError("a restricted candidate list (rcl_size > 1) needs an explicit rng")
    _check_card_mode(problem, card_mode)
    return _Search(problem, K, card_mode=card_mode, budget_s=math.inf).greedy(rng, rcl_size)


def local_search(
    problem: PortfolioProblem,
    K: int,
    *,
    budget_s: float,
    seed: int,
    improvement: str = "first",
    num_restarts: int | None = None,
    rcl_size: int = 3,
    start: CommonStart | None = None,
    card_mode: str = "le",
    clock: Callable[[], float] = time.perf_counter,
) -> Solution:
    """Greedy construction and one-asset swap descent on the true objective, with restarts.

    Parameters
    ----------
    budget_s:
        Total budget on the common end-to-end clock, the construction time of ``start`` included.
    improvement:
        ``"first"`` or ``"best"`` (module docstring).
    num_restarts:
        Number of descents (the first from ``start`` if given): 1–200 in the tuning space. ``None``:
        restart until the budget is spent (or every K-subset has been evaluated).
    rcl_size:
        Restricted candidate list of the greedy construction of restarts after the first (the
        plan does not fix it); a construction without a start for restart 0 is deterministic.
    start:
        The common start, the first incumbent and starting point; with an infinite objective
        (failed construction) it is not used, but its time is spent.
    card_mode:
        ``"le"`` (needs ``w_min = 0``) or ``"eq"`` as in :func:`cif.classical.refine.refine_problem`.
    clock:
        Wall clock; replaceable to test the budget logic without sleeping.

    Returns
    -------
    Solution
        The best K-subset evaluated (``objective = +inf``, zero weights, ``feasible=False`` if none
        was). ``wall_time_s`` is the end of the run on the common clock. ``solver_meta``:
        ``trajectory`` (list of ``(t, f)``), ``subset``, ``stop`` (``"budget"``, ``"restarts"`` or
        ``"exhausted"``), ``restarts_done``, ``converged_descents``, ``n_swaps``, ``n_evals`` (fresh
        refinements), ``n_cache_hits``, ``n_subsets_evaluated`` (K-subsets), ``refine_s``,
        ``overhead_s`` (the rest of the loop), ``first_own_s`` (the method's own first solution, §5:
        without a common start the end of the first greedy construction, the first point of the
        trajectory, not a refinement of a partial subset of it (audit G2, C-4); from a common start
        the first K-subset the descent refined, whether or not it beats the start, §12 A4),
        ``first_own_f`` (its value) and the inputs.
    """
    K = int(K)
    if not 1 <= K <= problem.n:
        raise ValueError(f"cardinality must be in [1, N={problem.n}], got {K}")
    if not (math.isfinite(budget_s) and budget_s > 0.0):
        raise ValueError(f"budget_s must be a finite positive number, got {budget_s}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if improvement not in _IMPROVEMENTS:
        raise ValueError(f"improvement must be one of {_IMPROVEMENTS}, got {improvement!r}")
    if num_restarts is not None and num_restarts < 1:
        raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")
    if rcl_size < 1:
        raise ValueError(f"rcl_size must be >= 1, got {rcl_size}")
    _check_card_mode(problem, card_mode)
    has_start = start is not None and math.isfinite(start.objective)
    if has_start and len(start.subset) != K:
        raise ValueError(f"the common start holds {len(start.subset)} assets, expected K = {K}")

    search = _Search(
        problem,
        K,
        card_mode=card_mode,
        budget_s=budget_s,
        offset=float(start.time_s) if start is not None else 0.0,
        clock=clock,
    )
    if has_start:
        search.seed_with(start)
    space = math.comb(problem.n, K)
    restarts_done = 0
    converged = 0
    stop = "restarts"
    try:
        while num_restarts is None or restarts_done < num_restarts:
            if search.now() >= budget_s:
                raise _BudgetExhausted
            if search.n_k_subsets >= space:
                stop = "exhausted"
                break
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), restarts_done]))
            if restarts_done == 0 and has_start:
                subset = tuple(start.subset)
                f, weights = float(start.objective), np.asarray(start.weights)
            else:
                subset = search.greedy(rng, rcl_size if restarts_done > 0 else 1)
                f, on_support = search.evaluate(subset)
                weights = search.full_weights(subset, on_support)
                search.consider(subset, f, weights)
            _, _, _, is_optimum = search.descend(subset, f, weights, improvement, rng)
            restarts_done += 1
            converged += int(is_optimum)
    except _BudgetExhausted:
        stop = "budget"

    wall = search.now()
    if search.best_w is None:
        weights = np.zeros(problem.n)
        feasible = False
    else:
        weights = np.array(search.best_w, dtype=float)  # a copy: the start's array stays its own
        feasible = bool(check_feasible(weights, problem) and np.count_nonzero(weights > 1e-6) <= K)
    meta: dict[str, Any] = {
        "trajectory": search.trajectory,
        "subset": list(search.best_subset) if search.best_subset is not None else None,
        "stop": stop,
        "improvement": improvement,
        "num_restarts": num_restarts,
        "rcl_size": rcl_size,
        "restarts_done": restarts_done,
        "converged_descents": converged,
        "n_swaps": search.n_swaps,
        "n_evals": search.n_evals,
        "n_cache_hits": search.n_hits,
        "n_subsets_evaluated": search.n_k_subsets,
        "first_own_s": search.first_own_s,
        "first_own_f": search.first_own_f,
        "refine_s": search.refine_s,
        "overhead_s": (wall - search.offset) - search.refine_s,
        "budget_s": budget_s,
        "seed": seed,
        "card_mode": card_mode,
        "start_used": bool(has_start),
    }
    return Solution(
        weights=weights,
        objective=float(search.best_f),
        feasible=feasible,
        wall_time_s=wall,
        solver=f"local_search[{improvement}]",
        solver_meta=meta,
    )
