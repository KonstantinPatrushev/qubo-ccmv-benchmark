"""Shared second stage of the QUBO pipelines: pool of K-subsets -> convex refinement.

Every sampler of the binary-inclusion QUBO (``neal`` SA, Tabu, swap-SA)
produces candidate size-``K`` subsets. The second stage is identical for all
of them and lives here, so that the heuristics differ **only** in the sampler:

1. take the first ``m`` unique size-``K`` subsets in order of arrival
   (plus an optional warm-start subset);
2. solve the continuous mean-variance subproblem on each subset with the
   refinement shared by all methods (:func:`cif.classical.refine.refine_problem`:
   Clarabel directly, bounds of the full problem; ANALYSIS_PLAN §4, audit RF-1).
   Only a turnover penalty *inside* the refinement (``turnover_in_refinement``)
   keeps the CVXPY solver :func:`cif.classical.continuous.solve_continuous_mvo`,
   which carries the L1 term;
3. return the subset with the lowest ranking objective.

Project convention for the trading cost (new, explicit): candidates are
ranked by ``obj + turnover_cost * sum_i |w_i - prev_i|``. ``turnover_cost`` is
the proportional cost of trading one unit of notional (a buy or a sell), so
rebalancing from ``prev`` costs ``turnover_cost * ||w - prev||_1``. The legacy
scripts used ``tc_bps * 0.5 * ||w - prev||_1 / 1e4`` (a cost per unit of
one-way turnover); :func:`legacy_turnover_cost` maps the old argument to the
new one.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator, Sequence

import dimod
import numpy as np

from cif.classical.continuous import solve_continuous_mvo
from cif.classical.refine import refine_problem
from cif.problem import PortfolioProblem, Solution

__all__ = [
    "check_feasible",
    "iter_sampleset_subsets",
    "legacy_turnover_cost",
    "refine_candidates",
]


def check_feasible(weights: np.ndarray, problem: PortfolioProblem, tol: float = 1e-6) -> bool:
    """Budget, box and cardinality feasibility of a weight vector.

    ``w_min`` is the minimum purchase (R16): it binds the bought assets (``|w_i| > 1e-6``, the
    support threshold of the cardinality check) and unbought ones must be zero, as in
    :func:`cif.classical.milp._box_ok`.
    """
    if abs(weights.sum() - problem.budget) > 1e-3:
        return False
    held = np.abs(weights) > 1e-6
    if np.any(weights[held] < problem.w_min - tol) or np.any(weights < -tol):
        return False
    if np.any(weights > problem.w_max + tol):
        return False
    if problem.cardinality is not None:
        n_held = int((weights > 1e-6).sum())
        if n_held > problem.cardinality:
            return False
    return True


def legacy_turnover_cost(turnover_cost_bps: float) -> float:
    """LEGACY: map the old ``turnover_cost_bps`` argument to ``turnover_cost``.

    The scripts ``experiment_a..d`` ranked candidates by
    ``obj + bps * 0.5 * ||w - prev||_1 / 1e4``. In the project convention
    (``turnover_cost * ||w - prev||_1``) this is ``turnover_cost = bps / 2e4``.
    Do not use in new code: state the cost per unit traded directly.
    """
    return float(turnover_cost_bps) / 20000.0


def iter_sampleset_subsets(
    sampleset: dimod.SampleSet,
    n: int,
    order: str = "read",
) -> Iterator[tuple[int, ...]]:
    """Yield the subset (indices with ``x_i = 1``) of every sample, any size.

    Parameters
    ----------
    order:
        ``"read"`` (default, legacy): the order of ``sampleset.record``, i.e.
        the order of the reads. dwave-samplers do **not** sort the record by
        energy, so with a pool of ``m`` this keeps the first ``m`` unique
        subsets, not the ``m`` best. ``"energy"``: ascending sampler energy
        (stable), which makes the pool "the m lowest-energy subsets".
    """
    if order not in ("read", "energy"):
        raise ValueError(f"order must be 'read' or 'energy', got {order!r}")
    position = {var: k for k, var in enumerate(sampleset.variables)}
    columns = [position[i] for i in range(n)]
    samples = np.asarray(sampleset.record.sample)[:, columns]
    rows = np.arange(samples.shape[0])
    if order == "energy":
        rows = np.argsort(np.asarray(sampleset.record.energy), kind="stable")
    for r in rows:
        yield tuple(np.flatnonzero(samples[r] == 1).tolist())


def _normalise_subset(subset: Iterable[int], n: int) -> tuple[int, ...]:
    out = tuple(sorted({int(i) for i in subset}))
    if out and (out[0] < 0 or out[-1] >= n):
        raise ValueError(f"subset {out} has indices outside [0, {n})")
    return out


def refine_candidates(
    problem: PortfolioProblem,
    candidates: Iterable[Sequence[int]],
    K: int,
    m: int,
    *,
    prev_weights: np.ndarray | None = None,
    turnover_cost: float = 0.0,
    warm_start_subset: Sequence[int] | None = None,
    refine_continuous: bool = True,
    solver: str = "selection+cvxpy",
    turnover_in_refinement: bool = False,
    card_mode: str | None = None,
) -> tuple[Solution | None, dict]:
    """Second stage: refine a pool of candidate subsets and return the best.

    Parameters
    ----------
    candidates:
        Subsets in order of arrival (a lazy iterator is consumed only as far
        as needed). Subsets whose size differs from ``K`` and repeated subsets
        are skipped; the first ``m`` unique size-``K`` subsets form the pool.
        The pool is **not** sorted: ordering (e.g. by sampler energy) is the
        caller's responsibility.
    K, m:
        Required subset size and pool size.
    prev_weights, turnover_cost:
        If ``prev_weights`` is given, candidates are ranked by
        ``obj + turnover_cost * sum_i |w_i - prev_i|`` (full L1 trade, not
        half of it; see the module docstring). ``Solution.objective`` is
        always the pure objective. Ties keep the earliest candidate.
    warm_start_subset:
        Added to the pool after the ``m`` sampled subsets unless it is empty,
        larger than ``K`` or already in the pool. Subsets SMALLER than ``K`` are
        accepted: current holdings often have fewer than ``K`` names.
    refine_continuous:
        If ``False`` the subset is weighted equally (``1/K``) instead of the
        convex refinement.
    solver:
        Label stored in ``Solution.solver``.
    turnover_in_refinement:
        If ``True`` (and ``prev_weights`` with ``turnover_cost > 0`` are given),
        the convex refinement of each subset minimises the penalised objective
        ``obj + turnover_cost·‖w − prev‖₁`` itself, not only the ranking. Then
        the second stage optimises exactly the objective of
        :func:`cif.classical.scip_native.solve_miqp_native` with
        ``turnover_penalty``, which makes exact and heuristic solvers follow one
        decision rule (R3). Default ``False`` keeps the legacy behaviour. This
        is the only refinement that still goes through CVXPY.
    card_mode:
        Semantics of ``w_min`` in the shared refinement
        (:func:`cif.classical.refine.refine_problem`, R16): ``"le"`` (``Σz ≤ K``,
        requires ``w_min = 0``; the support may be smaller than the subset) or
        ``"eq"`` (``Σz = K``: every asset of the subset is held at least
        ``w_min``, the OR-Library protocol). ``None`` (default) is ``"eq"`` when
        ``w_min > 0`` and ``"le"`` otherwise: the semantics this stage had before
        it used the shared refinement. Pass it explicitly in experiments.

    Returns
    -------
    best, stats:
        ``best`` is ``None`` when no candidate could be evaluated (empty pool
        or every convex solve failed); callers decide on a fallback.
        ``stats`` has ``n_candidates`` (pool size including the warm start),
        ``n_evaluated`` (successful refinements), ``refine_wall_s``, ``m`` and
        the ``card_mode`` used. ``best.solver_meta`` holds ``subset``; with
        ``prev_weights`` also ``ranking_objective`` and ``trade_l1``.
    """
    K = int(K)
    m = int(m)
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if turnover_cost < 0.0:
        raise ValueError(f"turnover_cost must be >= 0, got {turnover_cost}")
    mode = card_mode if card_mode is not None else ("eq" if problem.w_min > 0.0 else "le")
    if mode not in ("le", "eq"):
        raise ValueError(f"card_mode must be 'le' or 'eq', got {card_mode!r}")
    if refine_continuous and mode == "le" and problem.w_min > 0.0:
        raise ValueError(
            "card_mode='le' with w_min > 0 makes the refinement combinatorial; use card_mode='eq' "
            "(every asset of the subset held) or a problem with w_min = 0"
        )
    n = problem.n
    prev = None
    if prev_weights is not None:
        prev = np.asarray(prev_weights, dtype=float)
        if prev.shape != (n,):
            raise ValueError(f"prev_weights must have shape ({n},), got {prev.shape}")

    seen: set[tuple[int, ...]] = set()
    pool: list[tuple[int, ...]] = []
    for candidate in candidates:
        subset = _normalise_subset(candidate, n)
        if len(subset) != K or subset in seen:
            continue
        seen.add(subset)
        pool.append(subset)
        if len(pool) >= m:
            break

    warm: tuple[int, ...] | None = None
    if warm_start_subset is not None:
        warm = _normalise_subset(warm_start_subset, n)
        # The current holdings may have fewer than K names (refined weights can be 0); with
        # ``w_min = 0`` such a subset is feasible and must stay a candidate (audit A1).
        if 1 <= len(warm) <= K and warm not in seen:
            seen.add(warm)
            pool.append(warm)

    penalty_inside = turnover_in_refinement and prev is not None and turnover_cost > 0.0
    refine_t0 = time.perf_counter()
    best: Solution | None = None
    best_rank = float("inf")
    evaluated = 0
    for subset in pool:
        idx = np.asarray(subset)
        weights = np.zeros(n, dtype=float)
        if refine_continuous and penalty_inside:
            # the L1 trade penalty inside the refinement is the one thing the shared refine does
            # not carry; assets outside the subset are sold to 0: a constant for this subset
            sub_problem = PortfolioProblem(
                mu=problem.mu[idx],
                sigma=problem.sigma[np.ix_(idx, idx)],
                asset_names=tuple(problem.asset_names[i] for i in subset),
                w_min=problem.w_min,
                w_max=problem.w_max,
                budget=problem.budget,
                risk_aversion=problem.risk_aversion,
            )
            try:
                sub_sol = solve_continuous_mvo(
                    sub_problem, prev_weights=prev[idx], turnover_penalty=turnover_cost
                )
            except RuntimeError:
                continue
            weights[idx] = sub_sol.weights
        elif refine_continuous:
            refined = refine_problem(problem, subset, mode)
            if refined.status != "optimal":
                continue
            weights = refined.weights
        else:
            weights[idx] = 1.0 / len(idx)
        obj = problem.objective_value(weights)
        rank = obj
        trade = None
        if prev is not None:
            trade = float(np.sum(np.abs(weights - prev)))
            rank = obj + turnover_cost * trade
        evaluated += 1
        if best is None or rank < best_rank:
            best_rank = rank
            meta: dict = {"subset": list(subset)}
            if trade is not None:
                meta["ranking_objective"] = rank
                meta["trade_l1"] = trade
            best = Solution(
                weights=weights,
                objective=obj,
                feasible=check_feasible(weights, problem),
                wall_time_s=0.0,
                solver=solver,
                solver_meta=meta,
            )
    refine_wall = time.perf_counter() - refine_t0

    warm_in_pool = warm is not None and warm in seen
    stats = {
        "n_candidates": len(pool),
        "n_evaluated": evaluated,
        "refine_wall_s": refine_wall,
        "m": m,
        "card_mode": mode,
        "warm_start_in_pool": bool(warm_in_pool),
        "warm_start_selected": bool(
            warm_in_pool and best is not None
            and tuple(sorted(best.solver_meta.get("subset", []))) == warm
        ),
    }
    return best, stats
