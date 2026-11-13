"""Common start of every method in RQ2: top-K rounding of the relaxation, refined (§5, N5).

All methods are handed the same warm start: the K assets with the largest weights in the
cardinality-free continuous relaxation, with the weights re-optimised on that support by the shared
:func:`cif.classical.refine.refine_problem`. SCIP receives it through ``addSol``, outer
approximation as the initial incumbent, local search as its starting point, the QUBO pipelines in
their pool. The time of building it is charged to all of them: it is the first point of every
incumbent trajectory, ``(time_s, objective)`` (:attr:`CommonStart.entry`), and
:func:`cif.metrics.anytime.primal_integral` takes it as ``start``.

Both solves go through Clarabel directly, without CVXPY (§4). The relaxation is solved on the data
divided by ``mean(diag Σ)``, as in :func:`cif.instances.diagnostics.continuous_relaxation`: its
minimiser does not depend on the scale, the interior-point answer does.

Ties among equal relaxation weights (the zeros of a binding instance) go to the lower asset index,
so the construction is deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from cif.classical.refine import refine_problem, refine_support
from cif.problem import PortfolioProblem

__all__ = ["CommonStart", "common_start"]


@dataclass(frozen=True)
class CommonStart:
    """The shared warm start and what it cost.

    ``subset`` — the K rounded assets (sorted); ``weights`` — their refined weights (zero outside
    the support); ``objective`` — ``f(weights)``, ``+inf`` if the refinement failed (then
    ``status`` is not ``"optimal"``, e.g. ``"infeasible"`` when K assets cannot reach the budget
    under ``w_max``). ``time_s`` / ``cpu_s`` — wall and CPU time of the whole construction,
    ``relax_s`` and ``refine_s`` its two solves.
    """

    weights: np.ndarray
    objective: float
    subset: tuple[int, ...]
    status: str
    time_s: float
    cpu_s: float
    relax_s: float
    refine_s: float

    @property
    def entry(self) -> tuple[float, float]:
        """``(time_s, objective)``: the first trajectory point of every method, and ``start=`` of
        :func:`cif.metrics.anytime.primal_integral`."""
        return (self.time_s, self.objective)


def common_start(
    problem: PortfolioProblem, cardinality: int | None = None, *, card_mode: str = "le"
) -> CommonStart:
    """Top-K rounding of the continuous relaxation, followed by the shared refinement.

    ``cardinality`` defaults to ``problem.cardinality``. ``card_mode`` is that of
    :func:`cif.classical.refine.refine_problem` (``"eq"`` for the OR-Library protocol, where the
    relaxation ignores the minimum purchase and the refinement holds all K assets above it).

    Raises ``ValueError`` for a missing or out-of-range cardinality and for an infeasible
    relaxation (``N·w_max < budget``); a K-subset that cannot reach the budget is reported through
    ``status`` and an infinite ``objective`` instead.
    """
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    K = cardinality if cardinality is not None else problem.cardinality
    if K is None or not 1 <= K <= problem.n:
        raise ValueError(f"cardinality must be in [1, N={problem.n}], got {K}")
    scale = float(np.mean(np.diag(problem.sigma)))
    if not scale > 0.0:
        scale = 1.0
    relaxed = refine_support(
        problem.mu / scale,
        problem.sigma / scale,
        problem.risk_aversion,
        np.arange(problem.n),
        lower=0.0,
        upper=problem.w_max,
        budget=problem.budget,
    )
    if relaxed.status == "infeasible":
        raise ValueError(
            f"the continuous relaxation is infeasible (N·w_max = {problem.n * problem.w_max} "
            f"< budget = {problem.budget})"
        )
    if relaxed.status != "optimal":
        raise RuntimeError(f"the continuous relaxation failed: Clarabel status {relaxed.status}")
    order = np.argsort(-relaxed.weights, kind="stable")
    subset = tuple(sorted(order[:K].tolist()))
    refined = refine_problem(problem, subset, card_mode)
    return CommonStart(
        weights=refined.weights,
        objective=refined.objective,
        subset=subset,
        status=refined.status,
        time_s=time.perf_counter() - t0,
        cpu_s=time.process_time() - cpu0,
        relax_s=relaxed.time_s,
        refine_s=refined.time_s,
    )
