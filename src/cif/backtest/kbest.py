"""RQ3 operator (г): the k best supports of a fold (ANALYSIS_PLAN §2 RQ3, §5; consultation 2026-10-08).

For a fold with a certified optimum ``w*`` the operator lists the *distinct refined portfolios* in
order of the objective ``F = f(refine(·))`` and certifies every one of them with an exact solver.
Rank 1 is the optimum; the portfolio of rank ``r`` is ``refine(T_r)`` with the shared refinement
:mod:`cif.classical.refine` (``l = 0``, ``u = problem.w_max``), its support is
``T_r = {i : w_i > 1e-6}``, and its dose is ``ĝ_r = (F(T_r) − f*)·10⁴`` basis points of certainty
equivalent. The GMV arm is the same machinery on :func:`cif.backtest.injection.gmv_problem`.

Enumeration. Rank ``k + 1`` is the optimum of the MIQP of the optimum itself (``Σz = K``,
``w ≤ u·z``, ``l = 0``, see "Cardinality" below) with, for every support ``T_j`` found so far, the
cut ``Σ_{i∈T_j} z_i ≤ |T_j| − 1`` (:func:`cif.classical.scip_native.solve_miqp_native`,
``forbidden_supersets``). A plain no-good cut on ``z`` would be wrong: with ``l = 0`` the selection
``z_i = 1`` may carry the weight ``w_i = 0``, so the same portfolio comes back under another ``z``;
on a non-binding fold (``|supp(w*)| < K``) there are ``C(N − |supp|, K − |supp|)`` copies of the
optimum.

Exactness. Let ``F(S)`` be the optimum of the convex refinement on a set ``S`` and ``Σ_SS ≻ 0``
(Ledoit–Wolf, factor models), so ``refine(S)`` is unique. (i) ``refine(supp(refine(S))) =
refine(S)``: the refined portfolio is optimal on ``S`` and feasible on its own support, hence the
optimum there. A refined portfolio is therefore determined by its support, and distinct portfolios
have distinct supports. (ii) If ``Z ⊇ T`` then ``F(Z) ≤ F(T)``, with equality only if ``refine(Z) =
refine(T)``. Hence a refined portfolio whose support contains a found ``T_j`` is either ``P_j``
itself or strictly better than ``P_j``, and the latter has a lower rank and is already among the
found ones (the found portfolios are the best ``k``). So every portfolio not found yet has a support
that contains no ``T_j``: it is feasible with the cuts, as the selection ``z = 1_T``. Conversely
an allowed selection ``Z`` (no ``T_j`` inside) refines to a portfolio with ``supp ⊆ Z``, which
contains no ``T_j`` either, so it is not a found one. The optimum of the cut MIQP is therefore the
best portfolio not found yet, and the list is exhaustive, free of repeats and ordered by ``F``.
Ties in ``F`` do not break the argument, which uses strict inequalities only.

Cardinality. The plan names the object "supports of ``refine(S)`` over K-subsets ``S``" (what the
pipeline "pool → refinement" can output). It is enumerated exactly with ``card_mode="eq"``
(``Σz = K``, the default; ``l = 0`` makes every K-set feasible and the optimum is that of
``Σz ≤ K``). With ``card_mode="le"`` (``Σz ≤ K``) the cut MIQP also returns refinements of sets of
fewer than K assets that *no* K-subset refines to (every K-superset improves them, i.e. each of
the outside assets would enter): rank 10 on N = 12, K = 4 and rank 7 on N = 8, K = 3 in the
tests. The two models agree whenever each portfolio of fewer than K assets in the list is optimal
on some K-superset, which needs ``K − |T|`` outside assets that would stay at zero. That is not
guaranteed and fails on tiny instances; on the synthetic N = 49 folds tried
(``scripts/bench_kbest_cost.py --card-mode le``, four fold-arms, R = 32) both models listed the same
33 portfolios in the same order and every rank was a K-subset refinement. Under
``"eq"`` a fold with ``N ≤ K + 1`` has at most ``C(N, K) ≤ K + 1``
portfolios and is degenerate by the count (:func:`cif.backtest.fold_sets.kbest_fold_mask`).

Certificate. Every MIQP runs with SCIP ``limits/absgap = 1e-6·s_ref`` (``s_ref`` of the optimum,
:func:`cif.metrics.gap.reference_scale`; the plan: "как у оптимума") and a time limit per MIQP. A rank
is *certified* iff :func:`cif.metrics.gap.reference_from_exact` says so on that same scale, i.e. SCIP
stopped by the gap and ``|primal − dual| ≤ 1e-6·s_ref``. (Judged by the scale of the incumbent, a gap
inside the limit SCIP was given could be refused: the portfolios of the higher ranks earn less, their
scale is smaller, and the enumeration would end there with the fold leaving ``F_k`` more often where
the doses are large. ``MiqpRecord.scale`` stays the incumbent's, for information.) The dose is
recomputed by the exact refinement and does not depend on the tolerance. Since the portfolio returned
is within ``ε = 1e-6·s_ref`` of the best remaining one, the order is exact only up to ``ε``.

Ties. Portfolios whose ``F`` follow each other within ``ε`` (a chain of consecutive differences)
form a tie group and are ordered lexicographically by the sorted support, so the order does not
depend on the path of the solver. Rank 1 is the optimum supplied (the refinement of its support)
and is never reordered: an alternative tied with it is flagged (its dose is ≈ 0). The tie flag of a rank says that its group
has more than one member (or, for the first group, that it is tied with rank 1); the tie share is
the share of flagged ranks ``2…R+1``. The tolerance equals the SCIP limit because MIQP solutions
within ``ε`` of each other cannot be told apart by the certificate. Near duplicates are ties too:
a portfolio and the same portfolio without an asset ``j`` of weight ``w_j`` differ in ``F`` by at
most ``(λΣ_jj/2)·w_j²``, below ``ε`` for ``w_j`` up to a few 1e-4.

Boundary. The last rank ``R + 1`` is exact only if rank ``R + 2`` is not tied with it. One extra
MIQP (``boundary_check``) finds the next portfolio, the first ``R + 1`` of the sorted window are
kept and ``boundary_tie`` says whether the dropped one belongs to the tie group of the last kept
one; then the identity of rank ``R + 1`` depends on the tie-break (which is lexicographic among the
members found, a larger group is not completed). The fold is not excluded for it: the plan admits
no selection other than certification and degeneracy (``F_k``); the share of such folds is
reported. ``boundary_tie`` is ``None`` when the check did not run or could not finish.

Support threshold. ``W_SUPPORT_TOL = 1e-6`` of the plan. Interior-point weights on true zeros are
as large as 4·10⁻⁷ (observed), so the threshold has a margin of ~2.5. Every support is taken to a
fixed point (:func:`_polish`): the support of a SCIP solution is refined, the weights of the
result above the threshold are its new support, and so on until the support reproduces itself. This
removes the tiny spurious weights that SCIP is allowed to leave within its tolerance.

Warm start. The best one-swap neighbour of the optimum's support that is feasible under the cuts
found so far (scanned once per fold; with ``card_mode="eq"`` only neighbours holding exactly K
assets, since a warm start must satisfy ``Σz = K``: on a non-binding fold, whose neighbours hold
fewer, there is none, and those folds solve fast anyway). On the synthetic N = 49 folds it changed
the node count by −20 % and +9 % on the two folds compared, so it is a convenience, not a lever.

Degenerate folds. When the MIQP becomes infeasible, no further portfolio exists (``exhausted``):
``n_distinct < R + 1`` and the fold leaves ``F_k``. When an MIQP is not certified (time limit,
no incumbent) the enumeration stops, the ranks found are reported in the order of discovery and
the fold leaves ``F_k``; a portfolio that is only an incumbent is reported with ``certified=False``.
Seeds are explicit (``seed`` and ``permutation_seed`` go to SCIP) and recorded.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cif.backtest.fold_sets import kbest_fold_mask
from cif.backtest.injection import BP, CONSISTENCY_TOL_BP, W_SUPPORT_TOL
from cif.classical.refine import RefineResult, refine_problem
from cif.classical.scip_native import CardMode, DiagMethod, Formulation, solve_miqp_native
from cif.metrics.gap import reference_from_exact, reference_scale
from cif.problem import PortfolioProblem

__all__ = [
    "ABS_GAP_REL",
    "N_RANKS_FF49",
    "N_RANKS_US",
    "TIME_LIMIT_S",
    "KBestResult",
    "MiqpRecord",
    "RankedPortfolio",
    "enumerate_kbest",
]

N_RANKS_FF49 = 32  # R for FF49, both arms: ranks 2-33
N_RANKS_US = 1  # R for US-MV: rank 2 only
TIME_LIMIT_S = 600.0  # per MIQP (ANALYSIS_PLAN §2)
ABS_GAP_REL = 1e-6  # limits/absgap = ABS_GAP_REL · s_ref, also the tie tolerance (§5)

_CARD_MODES = ("le", "eq")


@dataclass(frozen=True)
class MiqpRecord:
    """Diagnostics of one MIQP of the enumeration.

    ``solve`` numbers the MIQPs from 1 in the order of discovery: solve ``k`` looks for the
    ``(k + 1)``-th portfolio and carries ``n_cuts = k`` cuts. After the tie rule the ranks are
    permuted, the record keeps its solve number. ``status`` is SCIP's; ``certified`` is true for a
    proof (the gap criterion of the plan, or infeasibility). ``scale`` is ``s_ref`` of the
    incumbent (``None`` without one), ``abs_gap`` the achieved ``|primal − dual|`` and
    ``abs_gap_limit`` SCIP's ``limits/absgap``. The times are SCIP's (build and solve, wall) and
    the process CPU time of the solve.
    """

    solve: int
    status: str
    certified: bool
    primal_bound: float
    dual_bound: float
    abs_gap: float
    abs_gap_limit: float
    scale: float | None
    n_nodes: int
    build_time_s: float
    solve_time_s: float
    solve_cpu_s: float
    warm_start_used: bool
    n_cuts: int


@dataclass(frozen=True)
class RankedPortfolio:
    """One rank. ``weights`` is ``refine(support)`` (full length N), ``objective`` its ``F``,
    ``dose_bp`` the excess over the optimum in basis points of certainty equivalent, ``tie`` the
    tie flag (module docstring) and ``miqp`` the record of the MIQP that found it (``None`` for
    rank 1, which is supplied)."""

    rank: int
    support: tuple[int, ...]
    weights: np.ndarray
    objective: float
    dose_bp: float
    certified: bool
    tie: bool
    miqp: MiqpRecord | None


@dataclass(frozen=True)
class KBestResult:
    """The k best supports of a fold.

    ``ranks[0]`` is the optimum, ``ranks[r - 1]`` the portfolio of rank ``r``; at most ``R + 1``
    entries (the extra MIQP of the boundary check is not a rank). ``solves`` lists every MIQP
    solved, the extra one included. ``f_star`` is ``F`` of the optimum, ``scale`` its ``s_ref``,
    ``abs_gap_limit`` SCIP's absolute gap and ``tie_tol`` the tie tolerance (both in objective
    units). ``exhausted``: an MIQP was proved infeasible, no further portfolio exists.
    ``stop_status``: SCIP status of the MIQP that ended the enumeration without a certificate
    (``None`` otherwise). ``boundary_tie``: see the module docstring. ``pool_time_s`` is the scan
    of the neighbours for the warm start, ``elapsed_s`` the wall time of the whole call.
    """

    ranks: tuple[RankedPortfolio, ...]
    solves: tuple[MiqpRecord, ...]
    n_ranks: int
    card_mode: str
    f_star: float
    scale: float
    abs_gap_limit: float
    tie_tol: float
    exhausted: bool
    boundary_tie: bool | None
    stop_status: str | None
    pool_time_s: float
    elapsed_s: float
    seed: int
    permutation_seed: int | None

    @property
    def n_distinct(self) -> int:
        """Distinct refined portfolios found among ranks ``1…R+1`` (the optimum included)."""
        return len(self.ranks)

    @property
    def complete(self) -> bool:
        """All ``R + 1`` ranks found and certified."""
        return len(self.ranks) == self.n_ranks + 1 and all(r.certified for r in self.ranks)

    @property
    def all_certified(self) -> bool:
        """Every rank found is certified and the list is either complete or proved to end."""
        return all(r.certified for r in self.ranks) and (self.complete or self.exhausted)

    @property
    def supports(self) -> tuple[tuple[int, ...], ...]:
        return tuple(r.support for r in self.ranks)

    @property
    def doses_bp(self) -> np.ndarray:
        return np.array([r.dose_bp for r in self.ranks])

    @property
    def tie_share(self) -> float:
        """Share of the alternatives (ranks ``2…``) that carry the tie flag."""
        flags = [r.tie for r in self.ranks[1:]]
        return float(np.mean(flags)) if flags else 0.0

    @property
    def solve_time_s(self) -> float:
        """SCIP solving time summed over all MIQPs."""
        return float(sum(s.solve_time_s for s in self.solves))

    def qualifies(self, optimum_certified: bool = True) -> bool:
        """The fold rule of ``F_k`` for this arm (:func:`cif.backtest.fold_sets.kbest_fold_mask`)."""
        return bool(
            kbest_fold_mask(optimum_certified, self.all_certified, self.n_distinct, self.n_ranks)
        )


# ---------------------------------------------------------------------------------------------
# refinement to a fixed point, the pool of neighbours
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Portfolio:
    support: tuple[int, ...]
    weights: np.ndarray
    objective: float
    members: frozenset[int]


def _support(weights: np.ndarray, budget: float) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(np.asarray(weights) > W_SUPPORT_TOL * budget))


def _polish(problem: PortfolioProblem, support: Sequence[int]) -> _Portfolio | None:
    """``refine(T)`` with ``T`` a fixed point: the weights above the threshold of the refinement of
    ``T`` are ``T`` again. Each step either stops or drops at least one asset, so the loop ends in
    at most ``|T|`` refinements. ``None`` if a refinement fails (e.g. fewer than ``1/u`` assets)."""
    t = tuple(sorted({int(i) for i in support}))
    for _ in range(len(t) + 1):
        if not t:
            return None
        res: RefineResult = refine_problem(problem, t, card_mode="le")
        if res.status != "optimal":
            return None
        nxt = _support(res.weights, problem.budget)
        if nxt == t:
            return _Portfolio(t, res.weights, float(res.objective), frozenset(t))
        t = nxt
    return None


def _swap_pool(problem: PortfolioProblem, t1: tuple[int, ...]) -> list[_Portfolio]:
    """The refined portfolios of the one-swap neighbours ``(T₁ ∖ {i}) ∪ {j}`` of the optimum's
    support, one per distinct support, best first (ties by support)."""
    held = set(t1)
    outside = [j for j in range(problem.n) if j not in held]
    seen: dict[tuple[int, ...], _Portfolio] = {}
    for i in t1:
        kept = held - {i}
        for j in outside:
            p = _polish(problem, kept | {j})
            if p is not None and p.support not in seen:
                seen[p.support] = p
    return sorted(seen.values(), key=lambda p: (p.objective, p.support))


def _pick_start(
    pool: Sequence[_Portfolio], found: Sequence[frozenset[int]], k: int, card_mode: str
) -> _Portfolio | None:
    """The best pool entry that holds none of the found supports (and, for ``Σz = K``, exactly K
    assets: the selection of a warm start is its support)."""
    for p in pool:
        if card_mode == "eq" and len(p.support) != k:
            continue
        if not any(f <= p.members for f in found):
            return p
    return None


# ---------------------------------------------------------------------------------------------
# the order of the alternatives
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ordering:
    order: tuple[int, ...]  # indices of the alternatives in rank order
    group: tuple[int, ...]  # tie group of the alternative at each position
    tied: tuple[bool, ...]  # tie flag of the alternative at each position
    anchor_tied: bool  # the first alternative is within the tolerance of the optimum


def _order_with_ties(
    objectives: Sequence[float], supports: Sequence[tuple[int, ...]], f_star: float, tol: float
) -> _Ordering:
    """Rank order of the alternatives: by objective, a chain of objectives within ``tol`` of the
    previous one forming a tie group that is ordered lexicographically by support.

    The optimum (``f_star``) is rank 1 and stays so; an alternative within ``tol`` of it is flagged
    (``anchor_tied``, the first group is flagged even when it has a single member).
    """
    idx = sorted(range(len(objectives)), key=lambda i: (objectives[i], supports[i]))
    groups: list[list[int]] = []
    for i in idx:
        if groups and objectives[i] - objectives[groups[-1][-1]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    anchor_tied = bool(groups) and objectives[groups[0][0]] - f_star <= tol
    order: list[int] = []
    group: list[int] = []
    tied: list[bool] = []
    for g, members in enumerate(groups):
        members.sort(key=lambda i: supports[i])
        flag = len(members) > 1 or (g == 0 and anchor_tied)
        for i in members:
            order.append(i)
            group.append(g)
            tied.append(flag)
    return _Ordering(tuple(order), tuple(group), tuple(tied), anchor_tied)


# ---------------------------------------------------------------------------------------------
# the enumeration
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Alternative:
    portfolio: _Portfolio
    record: MiqpRecord


def _record(
    solve: int, meta: dict, certified: bool, abs_gap: float, scale: float | None
) -> MiqpRecord:
    return MiqpRecord(
        solve=solve,
        status=str(meta["status"]),
        certified=certified,
        primal_bound=float(meta["primal_bound"]),
        dual_bound=float(meta["dual_bound"]),
        abs_gap=float(meta["abs_gap"]),
        abs_gap_limit=abs_gap,
        scale=scale,
        n_nodes=int(meta["n_nodes"]),
        build_time_s=float(meta["build_time_s"]),
        solve_time_s=float(meta["solve_time_s"]),
        solve_cpu_s=float(meta["solve_cpu_s"]),
        warm_start_used=bool(meta["warm_start_used"]),
        n_cuts=int(meta["n_forbidden_supersets"]),
    )


def enumerate_kbest(
    problem: PortfolioProblem,
    w_star: np.ndarray,
    *,
    n_ranks: int = N_RANKS_FF49,
    cardinality: int | None = None,
    card_mode: CardMode = "eq",
    time_limit_s: float | None = TIME_LIMIT_S,
    formulation: Formulation = "perspective",
    diag: DiagMethod = "mineig",
    abs_gap_rel: float = ABS_GAP_REL,
    seed: int = 0,
    permutation_seed: int | None = None,
    warm_start: bool = True,
    boundary_check: bool = True,
    consistency_tol_bp: float = CONSISTENCY_TOL_BP,
) -> KBestResult:
    """Ranks ``2…R+1`` of the fold: the best distinct refined portfolios after the optimum.

    Parameters
    ----------
    problem:
        The fold's problem with ``w_min = 0``; ``K`` is ``cardinality`` or ``problem.cardinality``.
        Pass :func:`cif.backtest.injection.gmv_problem` for the GMV arm.
    w_star:
        Weights of the certified optimum (SCIP's or the refinement of its support). The support
        (weights above ``1e-6``) has at most K assets. The optimum is checked: its objective must
        agree with the refinement of its support (``consistency_tol_bp``), no one-swap neighbour of
        the support and no rank may beat it by more than the tolerance, otherwise ``ValueError``
        ("not optimal": the certificate does not hold). The neighbour scan runs with
        ``warm_start``.
    n_ranks:
        ``R``: ranks ``2…R+1`` are enumerated (``N_RANKS_FF49 = 32``, ``N_RANKS_US = 1``).
    card_mode:
        ``"eq"`` (default) or ``"le"`` (module docstring, "Cardinality").
    time_limit_s:
        SCIP time limit per MIQP (``None``: none). An MIQP that does not finish ends the enumeration.
    formulation, diag:
        SCIP model (:func:`cif.classical.scip_native.solve_miqp_native`).
    abs_gap_rel:
        ``limits/absgap = abs_gap_rel·s_ref`` of the optimum; the certificate criterion and the tie
        tolerance use the same relative value. Positive.
    seed, permutation_seed:
        SCIP's ``randomseedshift``, and ``permutevars`` with ``permutationseed`` when not ``None``;
        the same for every MIQP. The ranks do not depend on them except through ties.
    warm_start:
        Hand SCIP the best feasible neighbour of the optimum's support (module docstring).
    boundary_check:
        One extra MIQP to detect a tie at rank ``R + 1``.
    """
    t_start = time.perf_counter()
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be ≥ 1, got {n_ranks}")
    if card_mode not in _CARD_MODES:
        raise ValueError(f"card_mode must be one of {_CARD_MODES}, got {card_mode!r}")
    if not (math.isfinite(abs_gap_rel) and abs_gap_rel > 0.0):
        raise ValueError(f"abs_gap_rel must be a finite positive number, got {abs_gap_rel}")
    k = cardinality if cardinality is not None else problem.cardinality
    if k is None:
        raise ValueError("a cardinality is required (argument or problem.cardinality)")
    if not 1 <= k <= problem.n:
        raise ValueError(f"cardinality must be in [1, N={problem.n}], got {k}")
    if problem.w_min > 0:
        raise ValueError(
            f"w_min = {problem.w_min} > 0: the k best supports use l = 0, where the refinement of "
            f"a set is a convex problem"
        )
    w = np.asarray(w_star, dtype=float)
    if w.shape != (problem.n,):
        raise ValueError(f"w_star must have shape ({problem.n},), got {w.shape}")
    if abs(w.sum() - problem.budget) > 1e-6:
        raise ValueError(f"w_star must sum to the budget {problem.budget}, got {w.sum():.8f}")
    if w.min() < -1e-6 or w.max() > problem.w_max + 1e-6:
        raise ValueError("w_star violates the box 0 ≤ w ≤ w_max")
    support = _support(w, problem.budget)
    if len(support) > k:
        raise ValueError(f"the support of w_star has {len(support)} assets, more than K = {k}")

    # rank 1: the optimum, polished by the shared refinement
    first = _polish(problem, support)
    if first is None:
        raise ValueError("the refinement of the support of w_star failed")
    excess_bp = (problem.objective_value(w) - first.objective) * BP
    if abs(excess_bp) > consistency_tol_bp:
        raise ValueError(
            f"w_star is not optimal on its support: its objective differs from the refinement of "
            f"the support by {excess_bp:.4g} bp"
        )
    t1, f_star = first.support, first.objective
    scale = reference_scale(problem, first.weights)
    abs_gap = abs_gap_rel * scale
    tie_tol = abs_gap

    pool: list[_Portfolio] = []
    pool_time = 0.0
    if warm_start:
        t0 = time.perf_counter()
        pool = _swap_pool(problem, t1)
        pool_time = time.perf_counter() - t0
        if pool and pool[0].objective < f_star - tie_tol:
            raise ValueError(
                f"w_star is not optimal: a one-swap neighbour of its support is better by "
                f"{(f_star - pool[0].objective) * BP:.4g} bp (the certificate does not hold)"
            )

    n_wanted = n_ranks + (1 if boundary_check else 0)
    found: list[frozenset[int]] = [first.members]
    cuts: list[tuple[int, ...]] = [t1]
    alternatives: list[_Alternative] = []
    solves: list[MiqpRecord] = []
    incumbent: _Alternative | None = None  # an uncertified incumbent that ended the enumeration
    exhausted = False
    stop_status: str | None = None
    for number in range(1, n_wanted + 1):
        start = _pick_start(pool, found, k, card_mode) if pool else None
        sol = solve_miqp_native(
            problem, k, formulation=formulation, diag=diag, card_mode=card_mode,
            time_limit_s=time_limit_s, mip_gap=0.0, abs_gap=abs_gap,
            warm_start=None if start is None else start.weights, seed=seed,
            permutation_seed=permutation_seed, forbidden_supersets=cuts,
        )
        meta = sol.solver_meta
        status = str(meta["status"])
        if status == "infeasible":  # every selection holds a found support: nothing is left
            solves.append(_record(number, meta, True, abs_gap, None))
            exhausted = True
            break
        if not math.isfinite(float(meta["primal_bound"])):  # no incumbent (time limit, ...)
            solves.append(_record(number, meta, False, abs_gap, None))
            stop_status = status
            break
        if not sol.feasible:
            raise RuntimeError(f"SCIP returned an infeasible incumbent (status {status})")
        polished = _polish(problem, _support(sol.weights, problem.budget))
        if polished is None:
            raise RuntimeError("the refinement of the support of a SCIP solution failed")
        if any(f <= polished.members for f in found):
            raise RuntimeError(
                f"solve {number} returned a support that holds an earlier one: {polished.support}"
            )
        scale_inc = reference_scale(problem, sol.weights)
        _, ref_type = reference_from_exact(meta, scale=scale, tol=abs_gap_rel)  # the scale of the limit
        record = _record(number, meta, ref_type == "certified", abs_gap, scale_inc)
        solves.append(record)
        if not record.certified:
            incumbent = _Alternative(polished, record)
            stop_status = status
            break
        if polished.objective < f_star - tie_tol:
            raise ValueError(
                f"w_star is not optimal: the portfolio of solve {number} is better by "
                f"{(f_star - polished.objective) * BP:.4g} bp (the certificate does not hold)"
            )
        alternatives.append(_Alternative(polished, record))
        found.append(polished.members)
        cuts.append(polished.support)

    # a failure before rank R + 1 is complete leaves the ranks in the order of discovery
    unsorted = stop_status is not None and len(alternatives) < n_ranks
    if unsorted:
        kept = list(alternatives)
        tied = [False] * len(kept)
        anchor_tied = False
        boundary_tie = None
    else:
        ordering = _order_with_ties(
            [a.portfolio.objective for a in alternatives],
            [a.portfolio.support for a in alternatives], f_star, tie_tol,
        )
        ordered = [alternatives[i] for i in ordering.order]
        kept, tied = ordered[:n_ranks], list(ordering.tied[:n_ranks])
        anchor_tied = ordering.anchor_tied
        if not boundary_check:
            boundary_tie = None
        elif len(ordered) > n_ranks:  # the extra portfolio is the first one dropped
            boundary_tie = ordering.group[n_ranks] == ordering.group[n_ranks - 1]
        elif exhausted and len(kept) == n_ranks:  # the extra MIQP proved that nothing follows
            boundary_tie = False
        else:
            boundary_tie = None

    ranks = [
        RankedPortfolio(1, t1, first.weights, f_star, 0.0, True, anchor_tied, None),
    ]
    for pos, alt in enumerate(kept):
        p = alt.portfolio
        ranks.append(
            RankedPortfolio(pos + 2, p.support, p.weights, p.objective,
                            (p.objective - f_star) * BP, True, tied[pos], alt.record)
        )
    if incumbent is not None and unsorted:  # beyond rank R + 1 an incumbent is only the extra MIQP
        p = incumbent.portfolio
        ranks.append(
            RankedPortfolio(len(ranks) + 1, p.support, p.weights, p.objective,
                            (p.objective - f_star) * BP, False, False, incumbent.record)
        )
    return KBestResult(
        ranks=tuple(ranks), solves=tuple(solves), n_ranks=n_ranks, card_mode=card_mode,
        f_star=f_star, scale=scale, abs_gap_limit=abs_gap, tie_tol=tie_tol,
        exhausted=exhausted, boundary_tie=boundary_tie, stop_status=stop_status,
        pool_time_s=pool_time, elapsed_s=time.perf_counter() - t_start, seed=seed,
        permutation_seed=permutation_seed,
    )
