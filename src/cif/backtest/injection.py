"""RQ3 gap injection: replicas of the certified optimum with a controlled in-sample gap.

ANALYSIS_PLAN §2 RQ3 ("Операторы инъекции зазора"). For a fold with certified optimum ``w*`` and
a level ``g`` (basis points of annual certainty equivalent) :class:`GapInjector` builds ``R``
replicas whose in-sample gap ``ĝ = (F − f*)·10⁴`` lies in the band ``[g, 1.1·g]``; ``F`` is the
shared convex refinement :mod:`cif.classical.refine` (``card_mode="le"``, ``u = problem.w_max``).

Operator (б), support swaps (primary): states are K-subsets, a move exchanges one held asset for
one that is not held, uniformly at random. The energy is the distance to the band::

    E(S) = max(0, g − ĝ(S)) + max(0, ĝ(S) − 1.1·g)

and a move is accepted by the Metropolis rule at ``T = 0.25·g``. A replica is the state at the
FIRST passage into the band; every replica is an independent chain from the start state
``S₀ ⊇ supp(w*)`` (``F(S₀) = f*``). A chain has at most 200 proposals, a replica at most 10
restarts (the first chain plus up to 10 more: 11 chains), a (fold, level) pair at most 60 s; a
level is *reached* on the fold only if all ``R = 30`` replicas are found. Before any chain is run the ``K·(N − K)`` one-swap neighbours of
``S₀`` are scanned: if the smallest positive gap among them exceeds ``1.1·g`` the level is
declared unreachable. (This is the plan's rule, a screening step rather than a proof: it ignores
paths of two or more moves.)

Operator (а), weight perturbation on the optimal support (descriptive, exploratory):
``w(t) = (1 − t)·w* + t·v`` with ``v ~ Dirichlet(1)`` on ``supp(w*)``, redrawn until ``v ≤ u``
(and until the segment can reach the level, ``ĝ(v) ≥ g``); ``t`` by bisection so that
``ĝ(w(t)) ∈ [g, 1.1·g]``. A convex combination of feasible points is feasible, so no projection
is needed. It is not built when ``|supp(w*)|·u ≤ 1 + 1e-9``: the face is then one point. On
average ``v`` is the equal-weight portfolio, so (а) shrinks towards 1/K, a regularisation that
can improve out-of-sample results.

GMV arm: pass :func:`gmv_problem` (``μ = 0``). Then ``F = (λ/2)·wᵀΣw`` and the gap is exactly
``(λ/2)·Δvariance·10⁴``.

Seeds: the first chain of a replica of (б) is ``SeedSequence((base, fold, level, replica))`` as in
the plan, restarts are its spawned children, operator (а) has a stream of its own, see
:func:`replica_seed`. Wall-clock limits make the status of a hard level machine-dependent;
every :class:`LevelResult` records ``n_evals`` and ``elapsed_s`` so that the pilot can see how
close to the limit the levels are.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace

import numpy as np

from cif.backtest.fold_sets import G_STAR_BP, LEVELS_BP
from cif.classical.refine import RefineResult, refine_problem
from cif.instances.diagnostics import SUPPORT_TOL, continuous_relaxation
from cif.problem import PortfolioProblem

__all__ = [
    "BAND_RATIO",
    "BP",
    "G_STAR_BP",
    "LEVELS_BP",
    "MAX_DRAWS",
    "MAX_RESTARTS",
    "MAX_STEPS",
    "N_REPLICAS",
    "TEMPERATURE_FACTOR",
    "TIME_LIMIT_S",
    "GapInjector",
    "LevelResult",
    "PerturbationResult",
    "PerturbedReplica",
    "Replica",
    "gmv_problem",
    "replica_seed",
]

BP = 1e4  # one basis point of certainty equivalent, in objective units
N_REPLICAS = 30  # R
BAND_RATIO = 1.1  # the band is [g, 1.1·g]
TEMPERATURE_FACTOR = 0.25  # T = 0.25·g
MAX_STEPS = 200  # proposals per chain
MAX_RESTARTS = 10  # restarts per replica (after the first chain)
TIME_LIMIT_S = 60.0  # per (fold, level)
MAX_DRAWS = 100_000  # raw Dirichlet draws per replica of operator (а)

#: Support threshold and feasibility tolerance of a certified optimum (as in
#: ``cif.heuristics.pipeline.check_feasible``; SCIP is feasible only to its ``feastol``).
W_SUPPORT_TOL = 1e-6
FEASIBILITY_TOL = 1e-6
#: |ĝ| at most this is zero. Refinements of equivalent supports agree to ≲ 2·10⁻⁸ bp (zero-gap
#: plateau of a non-binding optimum, N = 8); SCIP's objective and the refinement of its own
#: support differ by ≤ 5·10⁻⁵ bp (N = 49, three instances). Real positive gaps can be far smaller
#: than a basis point (a held asset with a tiny weight: 5·10⁻³ bp), but not below this.
ZERO_GAP_BP = 1e-4
#: Largest accepted disagreement between ``F(S₀)`` and the objective of the supplied ``w*``.
CONSISTENCY_TOL_BP = 0.01

_OPERATORS = ("swap", "perturb")
_PERTURB_SPAWN_INDEX = 2**31  # reserved spawn index of operator (а); restarts are ≤ MAX_RESTARTS


def replica_seed(
    base: int, fold: int, level: int, replica: int, *, operator: str = "swap", restart: int = 0
) -> np.random.SeedSequence:
    """``SeedSequence`` of one chain of a replica; ``level`` is an integer label of the level.

    The first chain of operator (б) is exactly the plan's ``SeedSequence((base, fold, level,
    replica))``. Restart ``k ≥ 1`` of the same replica is the ``k``-th child spawned from it
    (``spawn_key = (k − 1,)``), so the streams of one replica are independent. Operator (а) has no
    restarts and takes the child with the reserved index ``2³¹``. (NumPy hashes ``entropy`` and
    ``spawn_key`` as one list, so a marker appended to the entropy would collide with a restart.)
    """
    if operator not in _OPERATORS:
        raise ValueError(f"operator must be one of {_OPERATORS}, got {operator!r}")
    if restart < 0 or (operator == "perturb" and restart != 0):
        raise ValueError(f"restart must be ≥ 0 (and 0 for operator (а)), got {restart}")
    entropy = (int(base), int(fold), int(level), int(replica))
    if operator == "perturb":
        return np.random.SeedSequence(entropy, spawn_key=(_PERTURB_SPAWN_INDEX,))
    return np.random.SeedSequence(entropy, spawn_key=() if restart == 0 else (int(restart) - 1,))


def gmv_problem(problem: PortfolioProblem) -> PortfolioProblem:
    """The GMV arm of ``problem``: ``μ = 0``, so ``F(w) = (λ/2)·wᵀΣw`` (same K, u, λ_f)."""
    return replace(problem, mu=np.zeros(problem.n))


def _complete_start(
    support: Iterable[int], relax_weights: np.ndarray, k: int, tol: float
) -> tuple[int, ...]:
    """``S₀``: the support completed to ``k`` assets by the largest continuous-relaxation weights.

    Relaxation weights not above ``tol`` are ties (interior-point noise) and are resolved by the
    lowest index, so the start does not depend on solver noise.
    """
    held = {int(i) for i in support}
    free = k - len(held)
    if free < 0:
        raise ValueError(f"the support has {len(held)} assets, more than K = {k}")
    relax = np.asarray(relax_weights, dtype=float)
    rest = [i for i in range(relax.shape[0]) if i not in held]
    rest.sort(key=lambda i: (-relax[i] if relax[i] > tol else 0.0, i))
    return tuple(sorted(held | set(rest[:free])))


def _energy(gap: float, g: float, hi: float) -> float:
    """Distance of the gap to the band ``[g, hi]``; ``inf`` for an infeasible subset."""
    return max(0.0, g - gap) + max(0.0, gap - hi)


@dataclass(frozen=True)
class _Chain:
    found: bool
    subset: tuple[int, ...]  # the state at first passage if found, otherwise the last state
    gap_bp: float
    steps: int  # proposals made
    accepted: int
    timed_out: bool


def _swap_chain(
    gap_fn: Callable[[tuple[int, ...]], float],
    start: tuple[int, ...],
    n_assets: int,
    g: float,
    rng: np.random.Generator,
    *,
    max_steps: int,
    band_ratio: float,
    temperature_factor: float,
    expired: Callable[[], bool],
) -> _Chain:
    """One Metropolis chain over K-subsets from ``start`` until the first passage into the band.

    A step proposes to exchange a uniformly drawn held asset for a uniformly drawn non-held one
    (two integer draws), evaluates the gap of the new subset and accepts it when the energy does
    not rise, otherwise with probability ``exp(−ΔE/T)`` (one uniform draw, only then). Rejected
    proposals leave the state unchanged, so the band can only be entered by an accepted move; that
    state is returned. ``expired()`` is polled before every proposal.
    """
    hi = band_ratio * g
    temperature = temperature_factor * g
    inside = list(start)
    held = set(inside)
    outside = [i for i in range(n_assets) if i not in held]
    gap = gap_fn(tuple(sorted(inside)))
    energy = _energy(gap, g, hi)
    steps = accepted = 0
    while steps < max_steps:
        if expired():
            return _Chain(False, tuple(sorted(inside)), gap, steps, accepted, True)
        steps += 1
        a = int(rng.integers(len(inside)))
        b = int(rng.integers(len(outside)))
        candidate = list(inside)
        candidate[a] = outside[b]
        cand_gap = gap_fn(tuple(sorted(candidate)))
        cand_energy = _energy(cand_gap, g, hi)
        if cand_energy <= energy or rng.random() < math.exp(-(cand_energy - energy) / temperature):
            inside[a], outside[b] = outside[b], inside[a]
            gap, energy = cand_gap, cand_energy
            accepted += 1
            if energy == 0.0:
                return _Chain(True, tuple(sorted(inside)), gap, steps, accepted, False)
    return _Chain(False, tuple(sorted(inside)), gap, steps, accepted, False)


_DRAW_BATCH = 4096  # Dirichlet vectors per call; the stream does not depend on it
_BISECT_ITERATIONS = 200
FACE_TOL = 1e-9  # |supp(w*)|·u ≤ budget + FACE_TOL: the face is one point (ANALYSIS_PLAN §11)


def _accepted_directions(
    rng: np.random.Generator, s: int, u: float, *, max_draws: int, batch: int = _DRAW_BATCH
) -> Iterator[tuple[np.ndarray, int]]:
    """Dirichlet(1) vectors on ``s`` assets that satisfy ``v ≤ u`` (rejection), each with the
    number of raw draws used up to and including it; stops after ``max_draws`` raw draws."""
    alpha = np.ones(s)
    drawn = 0
    while drawn < max_draws:
        m = min(batch, max_draws - drawn)
        v = rng.dirichlet(alpha, size=m)
        for i in np.flatnonzero((v <= u).all(axis=1)):
            yield v[i], drawn + int(i) + 1
        drawn += m


def _bisect_band(
    gap_of_t: Callable[[float], float], g: float, band_ratio: float, max_iter: int = _BISECT_ITERATIONS
) -> float | None:
    """``t ∈ (0, 1]`` with ``gap_of_t(t) ∈ [g, band_ratio·g]`` by bisection, or ``None`` when the
    segment cannot reach ``g`` (``gap_of_t(1) < g``) or the search does not end in the band.

    ``gap_of_t(0) = 0 < g ≤ gap_of_t(1)`` and continuity bracket a crossing; a gap that is convex
    along the segment (as the objective is) makes it monotone.
    """
    top = gap_of_t(1.0)
    if top < g:
        return None
    if top <= band_ratio * g:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        t = 0.5 * (lo + hi)
        gap = gap_of_t(t)
        if g <= gap <= band_ratio * g:
            return t
        if gap < g:
            lo = t
        else:
            hi = t
    return None


def _segment_gap(
    objective: Callable[[np.ndarray], float], f_star: float, w0: np.ndarray, v: np.ndarray
) -> Callable[[float], float]:
    """``t ↦ (f((1 − t)·w0 + t·v) − f*)·10⁴`` along the segment."""
    return lambda t: (objective((1.0 - t) * w0 + t * v) - f_star) * BP


@dataclass(frozen=True)
class Replica:
    """One injected portfolio: the refinement of ``subset`` (weights), its in-sample gap, and how
    it was found (proposals of the successful chain, restarts used before it)."""

    subset: tuple[int, ...]
    weights: np.ndarray
    gap_bp: float
    steps: int
    restarts: int


@dataclass(frozen=True)
class LevelResult:
    """Outcome of operator (б) on one (fold, level).

    ``status``: ``"reached"`` (all ``n_requested`` replicas found), ``"unreachable"`` (screening
    by the one-swap neighbours, no chain was run), ``"failed"`` (a replica was not found within
    its restarts; the level stops there) or ``"timeout"`` (time limit). Replicas found before a
    failure are kept for diagnostics, but only a reached level counts. ``n_chains`` counts all
    chains launched, ``n_evals`` the refinements done by the chains (cache hits excluded).
    """

    gap_bp: float
    status: str
    replicas: tuple[Replica, ...]
    n_requested: int
    n_chains: int
    n_evals: int
    elapsed_s: float
    min_positive_neighbour_gap_bp: float

    @property
    def reached(self) -> bool:
        return self.status == "reached"

    @property
    def n_distinct_subsets(self) -> int:
        """Distinct states among the replicas: a narrow band can hold one state only, so all
        replicas coincide (they are independent chains, not independent states)."""
        return len({r.subset for r in self.replicas})


@dataclass(frozen=True)
class PerturbedReplica:
    """A replica of operator (а): ``weights = (1 − t)·w* + t·direction`` (full length N), its gap
    (objective excess of ``weights`` themselves), and the raw Dirichlet draws it took."""

    weights: np.ndarray
    gap_bp: float
    t: float
    direction: np.ndarray
    n_draws: int


@dataclass(frozen=True)
class PerturbationResult:
    """Outcome of operator (а) on one (fold, level).

    ``status``: ``"reached"``, ``"not_built"`` (the face on the support is one point, see the
    module docstring) or ``"failed"`` (no admissible direction reaching ``g`` within the draw
    budget; replicas found before are kept).
    """

    gap_bp: float
    status: str
    replicas: tuple[PerturbedReplica, ...]
    n_requested: int
    reason: str | None = None

    @property
    def reached(self) -> bool:
        return self.status == "reached"

    @property
    def built(self) -> bool:
        return self.status != "not_built"


def _level_key(g: float, level_key: int | None) -> int:
    if level_key is not None:
        return int(level_key)
    if g != int(g):
        raise ValueError(
            f"gap_bp = {g} is not an integer: pass level_key, the integer that enters the SeedSequence"
        )
    return int(g)


class GapInjector:
    """Gap injection for one fold and one arm (MV, or GMV via :func:`gmv_problem`).

    Parameters
    ----------
    problem:
        The fold's problem with ``cardinality = K`` (``Σz ≤ K``) and ``w_min = 0``; ``u`` is
        ``problem.w_max``.
    w_star:
        Weights of the certified optimum. Its support (weights above ``support_tol``) must have
        at most K assets. The constructor checks that ``w*`` is optimal on the completed start
        support (to ``consistency_tol_bp``) and, by scanning the one-swap neighbours, that no
        neighbour is better; otherwise it raises, because every gap is measured against ``f*``.

    Attributes
    ----------
    support, start:
        ``supp(w*)`` and the start state ``S₀`` (sorted tuples); ``f_star = F(S₀)`` and
        ``w_star`` (the refinement of ``S₀``, zero outside the support). ``n_evals`` counts the
        refinements done so far, ``scan_evals`` / ``scan_time_s`` the neighbour scan.

    Gaps are cached per subset, so a subset is refined once for all chains, replicas and levels
    of the fold. The cache does not couple the replicas: it stores a deterministic function.
    """

    def __init__(
        self,
        problem: PortfolioProblem,
        w_star: np.ndarray,
        *,
        support_tol: float = W_SUPPORT_TOL,
        consistency_tol_bp: float = CONSISTENCY_TOL_BP,
        zero_gap_bp: float = ZERO_GAP_BP,
    ) -> None:
        k = problem.cardinality
        if k is None:
            raise ValueError("problem.cardinality (K) is required for gap injection")
        if problem.w_min > 0:
            raise ValueError("gap injection uses card_mode='le', which needs w_min = 0")
        w = np.asarray(w_star, dtype=float)
        if w.shape != (problem.n,):
            raise ValueError(f"w_star must have shape ({problem.n},), got {w.shape}")
        if abs(w.sum() - problem.budget) > FEASIBILITY_TOL:
            raise ValueError(f"w_star must sum to the budget {problem.budget}, got {w.sum():.8f}")
        if w.min() < -FEASIBILITY_TOL or w.max() > problem.w_max + FEASIBILITY_TOL:
            raise ValueError("w_star violates the box 0 ≤ w ≤ w_max")
        self.problem = problem
        self.k = int(k)
        self.zero_gap_bp = float(zero_gap_bp)
        self.support = tuple(int(i) for i in np.flatnonzero(w > support_tol * problem.budget))
        if len(self.support) > self.k:
            raise ValueError(
                f"the support of w_star has {len(self.support)} assets, more than K = {self.k}"
            )
        relax = continuous_relaxation(problem).weights
        self.start = _complete_start(self.support, relax, self.k, SUPPORT_TOL * problem.budget)

        self.n_evals = 0
        self._cache: dict[tuple[int, ...], float] = {}
        ref = self._refine(self.start)
        if ref.status != "optimal":
            raise ValueError(f"the refinement of the start support failed: {ref.status}")
        gap_to_cert = (ref.objective - problem.objective_value(w)) * BP
        if abs(gap_to_cert) > consistency_tol_bp:
            raise ValueError(
                f"w_star is not optimal on its support completed to K assets: the refinement "
                f"differs from its objective by {gap_to_cert:.4g} bp"
            )
        self.f_star = float(ref.objective)
        self._cache[self.start] = 0.0
        self.w_star = np.where(np.isin(np.arange(problem.n), self.support), ref.weights, 0.0)
        self._scan_neighbours(consistency_tol_bp)

    # -- gap of a subset ---------------------------------------------------------------------

    def _refine(self, subset: tuple[int, ...]) -> RefineResult:
        self.n_evals += 1
        return refine_problem(self.problem, subset, card_mode="le")

    def gap_bp(self, subset: Iterable[int]) -> float:
        """``ĝ(S) = (F(S) − f*)·10⁴`` in basis points; ``inf`` if the refinement fails."""
        key = tuple(sorted(int(i) for i in subset))
        gap = self._cache.get(key)
        if gap is None:
            res = self._refine(key)
            gap = (res.objective - self.f_star) * BP if res.status == "optimal" else math.inf
            self._cache[key] = gap
        return gap

    # -- one-swap neighbourhood of S0 --------------------------------------------------------

    def _scan_neighbours(self, tol_bp: float) -> None:
        t0 = time.perf_counter()
        evals0 = self.n_evals
        held = set(self.start)
        outside = [j for j in range(self.problem.n) if j not in held]
        gaps = np.array([self.gap_bp((held - {i}) | {j}) for i in self.start for j in outside])
        self.scan_time_s = time.perf_counter() - t0
        self.scan_evals = self.n_evals - evals0
        self._neighbour_gaps = gaps
        if gaps.size and gaps.min() < -tol_bp:
            raise ValueError(
                f"w_star is not optimal: a one-swap neighbour of its support is better by "
                f"{-gaps.min():.4g} bp (the certificate does not hold)"
            )

    def neighbour_gaps_bp(self) -> np.ndarray:
        """Gaps of all ``K·(N − K)`` one-swap neighbours of ``S₀`` (computed at construction)."""
        return self._neighbour_gaps.copy()

    def min_positive_neighbour_gap_bp(self) -> float:
        """Smallest neighbour gap above the numerical zero (``inf`` if there is none)."""
        positive = self._neighbour_gaps[self._neighbour_gaps > self.zero_gap_bp]
        return float(positive.min()) if positive.size else math.inf

    # -- operator (б) ------------------------------------------------------------------------

    def swap_level(
        self,
        gap_bp: float,
        *,
        base_seed: int,
        fold: int,
        level_key: int | None = None,
        n_replicas: int = N_REPLICAS,
        max_steps: int = MAX_STEPS,
        max_restarts: int = MAX_RESTARTS,
        time_limit_s: float | None = TIME_LIMIT_S,
        band_ratio: float = BAND_RATIO,
        temperature_factor: float = TEMPERATURE_FACTOR,
        clock: Callable[[], float] = time.perf_counter,
    ) -> LevelResult:
        """Operator (б) at level ``gap_bp`` on this fold (see the module docstring).

        ``base_seed``, ``fold`` and ``level_key`` (default: ``gap_bp`` itself, which must then be
        an integer) enter the ``SeedSequence`` of every chain together with the replica index.
        ``time_limit_s`` covers the chains of this call (the neighbour scan is reported in
        ``scan_time_s``); ``None`` disables it. ``clock`` is injectable for tests.
        """
        g = float(gap_bp)
        if not (g > 0.0 and math.isfinite(g)):
            raise ValueError(f"gap_bp must be a positive finite number, got {gap_bp!r}")
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be ≥ 1, got {n_replicas}")
        key = _level_key(g, level_key)
        smallest = self.min_positive_neighbour_gap_bp()
        if smallest > band_ratio * g:
            return LevelResult(g, "unreachable", (), n_replicas, 0, 0, 0.0, smallest)

        evals0 = self.n_evals
        t0 = clock()

        def expired() -> bool:
            return time_limit_s is not None and clock() - t0 >= time_limit_s

        replicas: list[Replica] = []
        n_chains = 0
        status = "reached"
        for replica in range(n_replicas):
            found = None
            for restart in range(max_restarts + 1):
                rng = np.random.default_rng(
                    replica_seed(base_seed, fold, key, replica, operator="swap", restart=restart)
                )
                chain = _swap_chain(
                    self.gap_bp, self.start, self.problem.n, g, rng, max_steps=max_steps,
                    band_ratio=band_ratio, temperature_factor=temperature_factor, expired=expired,
                )
                n_chains += 1
                if chain.found:
                    found = (chain, restart)
                    break
                if chain.timed_out:
                    status = "timeout"
                    break
            if found is None:
                if status != "timeout":
                    status = "failed"
                break
            chain, restart = found
            weights = self._refine(chain.subset).weights
            replicas.append(Replica(chain.subset, weights, chain.gap_bp, chain.steps, restart))
        return LevelResult(
            gap_bp=g,
            status=status,
            replicas=tuple(replicas),
            n_requested=n_replicas,
            n_chains=n_chains,
            n_evals=self.n_evals - evals0,
            elapsed_s=clock() - t0,
            min_positive_neighbour_gap_bp=smallest,
        )

    # -- operator (а) ------------------------------------------------------------------------

    def perturb_level(
        self,
        gap_bp: float,
        *,
        base_seed: int,
        fold: int,
        level_key: int | None = None,
        n_replicas: int = N_REPLICAS,
        max_draws: int = MAX_DRAWS,
        band_ratio: float = BAND_RATIO,
    ) -> PerturbationResult:
        """Operator (а) at level ``gap_bp`` on this fold (descriptive; see the module docstring).

        Per replica: directions ``v`` are drawn until one satisfies ``v ≤ u`` and its segment
        reaches the level (``ĝ(v) ≥ g``), at most ``max_draws`` raw draws; then ``t`` by bisection.
        Seeds as in :meth:`swap_level` (stream ``"perturb"``).
        """
        g = float(gap_bp)
        if not (g > 0.0 and math.isfinite(g)):
            raise ValueError(f"gap_bp must be a positive finite number, got {gap_bp!r}")
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be ≥ 1, got {n_replicas}")
        key = _level_key(g, level_key)
        problem = self.problem
        s = len(self.support)
        if s * problem.w_max <= problem.budget + FACE_TOL:
            return PerturbationResult(
                g, "not_built", (), n_replicas,
                reason=f"|supp(w*)|·u = {s}·{problem.w_max:g} ≤ budget: the face on the support "
                       f"is a single point",
            )
        idx = np.array(self.support)
        mu_s = problem.mu[idx]
        sigma_s = problem.sigma[np.ix_(idx, idx)]
        half_lambda = 0.5 * problem.risk_aversion

        def objective(w: np.ndarray) -> float:
            return float(-mu_s @ w + half_lambda * w @ sigma_s @ w)

        w0 = self.w_star[idx]
        replicas: list[PerturbedReplica] = []
        for replica in range(n_replicas):
            rng = np.random.default_rng(replica_seed(base_seed, fold, key, replica, operator="perturb"))
            found = None
            for v, drawn in _accepted_directions(rng, s, problem.w_max / problem.budget,
                                                 max_draws=max_draws):
                direction = problem.budget * v
                gap_of_t = _segment_gap(objective, self.f_star, w0, direction)
                t = _bisect_band(gap_of_t, g, band_ratio)
                if t is not None:
                    found = (t, direction, gap_of_t(t), drawn)
                    break
            if found is None:
                return PerturbationResult(
                    g, "failed", tuple(replicas), n_replicas,
                    reason=f"replica {replica}: no admissible direction reaches {g:g} bp within "
                           f"{max_draws} draws",
                )
            t, direction, gap, drawn = found
            weights = np.zeros(problem.n)
            weights[idx] = (1.0 - t) * w0 + t * direction
            full_direction = np.zeros(problem.n)
            full_direction[idx] = direction
            replicas.append(PerturbedReplica(weights, gap, t, full_direction, drawn))
        return PerturbationResult(g, "reached", tuple(replicas), n_replicas)
