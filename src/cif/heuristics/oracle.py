"""RQ1 surrogate oracle: the ceiling of the QUBO pipeline and its attribution (ANALYSIS_PLAN §2 RQ1).

The pipeline "QUBO sampler → pool of K-subsets → convex refinement" can only ever be as good as the
surrogate it minimises. The surrogate is the equal-weight binary-inclusion QUBO
(:func:`cif.qubo.builder.build_selection_bqm`). On the feasible manifold ``|S| = K`` the
cardinality penalty vanishes and its energy is the objective part at ``w = x/K``::

    E(S) = −(1/K)·Σ_{i∈S} μ_i + (λ/(2K²))·Σ_{i,j∈S} Σ_ij = f(1_S / K),

so the oracle does not depend on the penalty multiplier: the constraint ``|S| = K`` is imposed by
construction. Everything below is a function of the instance, the shared refinement and a reference
value; no sampler is involved except through the ``pool`` it is given.

Definitions (``F(S) = f(refine(S))`` with :func:`cif.classical.refine.refine_problem`)::

    T_m      the m K-subsets of lowest E(S), by full enumeration (C(N, K) ≤ 2·10⁶)
    E_form(m) = (min_{S∈T_m} F(S) − f*) / s*          ceiling of the surrogate
    E_pipe(m) = (min_{S∈P_m} F(S) − f*) / s*          P_m = first m valid unique subsets of the pool
    E_samp(m) = E_pipe(m) − E_form(m)                 sampler residual; < 0: pool diversity wins
    s*        = |μᵀw*| + (λ/2)·w*ᵀΣw*  > 0           positive normalising scale

``E_pipe = E_form + E_samp`` is an identity (an attribution, not a decomposition into independent
causes). Pools are nested, so every quantity is non-increasing in m.

The residual of the sampler is split further (ANALYSIS_PLAN §12 B2), again as an identity::

    E_samp(m) = E_prec(m) + E_short(m)
    E_prec(m)  = E_pipe(m) − E_form(n_pipe(m))        sampler against the oracle at an EQUAL pool size
    E_short(m) = E_form(n_pipe(m)) − E_form(m)  ≥ 0   the price of a pool smaller than m

with ``n_pipe(m) = min(m, number of unique valid subsets of the pool)`` and ``E_form(n)`` the ceiling
at any pool size ``n ≤ m_max`` (:attr:`SurrogateOracle.e_form_curve`; ``T_{m_max}`` is refined anyway,
so this costs nothing). ``E_prec`` may be negative (the pool beats the oracle of its size: diversity
over accuracy), ``E_short`` is zero when the pool has m subsets.

Determinism. Ties in the surrogate energy are broken by the lexicographic order of the sorted index
tuple, so ``T_m`` is a function of the instance alone, and nested in ``m``. Energies are summed in a
fixed order of subset positions. The pool keeps the order it is given (the order of sampler
energy): the first occurrence of a subset counts, later copies, subsets of a size other than K and
subsets with indices outside ``range(N)`` are dropped.

With ``card_mode="le"`` (``Σz ≤ K``, ``w_min = 0``) the refinement may leave part of S at zero, so
``F(S)`` is the optimum over supports inside S and ``min_{|S|=K} F(S) = f*``. With ``"eq"`` every
asset of S is held (ORlib, ``Σz = K``); there the reference comes from
:func:`exhaustive_optimum` because ``scip_native`` has no ``Σz = K`` mode yet.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from scipy.stats import kendalltau

from cif.classical.refine import refine_problem
from cif.classical.scip_native import solve_miqp_native
from cif.metrics.gap import reference_from_exact, reference_scale
from cif.problem import PortfolioProblem

__all__ = [
    "DEFAULT_MS",
    "MAX_SUBSETS",
    "Attribution",
    "Reference",
    "SubsetValues",
    "SurrogateOracle",
    "best_subsets",
    "certified_optimum",
    "enumerate_subsets",
    "exhaustive_optimum",
    "objective_scale",
    "pool_subsets",
    "prefix_minima",
    "surrogate_energies",
]

#: Largest ``C(N, K)`` the oracle enumerates (ANALYSIS_PLAN §2: "C(N, K) ≤ 2·10⁶").
MAX_SUBSETS = 2_000_000
#: Pool sizes m of the attribution (ANALYSIS_PLAN §2: the quality-versus-time curve).
DEFAULT_MS = (1, 5, 20, 50, 200)

_SATURATION = 1 << 62  # binomials above this are never compared with a rank (rank < C(N, K))
_SUPPORT_TOL = 1e-9  # SCIP reports exact zeros off its support; this only guards rounding


# --------------------------------------------------------------------------------------------
# Enumeration and surrogate energy
# --------------------------------------------------------------------------------------------


def enumerate_subsets(n: int, k: int, *, max_subsets: int = MAX_SUBSETS) -> np.ndarray:
    """All K-subsets of ``range(n)`` as a ``(C(n, k), k)`` array of sorted rows, lexicographic.

    Vectorised unranking: the combinatorial number system gives the subset of a given rank by ``k``
    binary searches over a table of binomials (no Python loop over subsets). It works in colex
    order on the mirrored subsets ``{n − 1 − i}``, which reverses it into lexicographic order.
    Raises ``ValueError`` if ``C(n, k)`` exceeds ``max_subsets`` or ``k`` is not in ``[1, n]``.
    """
    n, k = int(n), int(k)
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    total = math.comb(n, k)
    if total > max_subsets:
        raise ValueError(
            f"C({n}, {k}) = {total} subsets exceed the limit {max_subsets} of the enumeration"
        )
    table = [
        np.array([min(math.comb(c, i), _SATURATION) for c in range(n)], dtype=np.int64)
        for i in range(k + 1)
    ]
    rank = np.arange(total - 1, -1, -1, dtype=np.int64)  # colex rank of the mirrored subset
    out = np.empty((total, k), dtype=np.int16 if n < 2**15 else np.int32)
    for i in range(k, 0, -1):
        c = np.searchsorted(table[i], rank, side="right") - 1  # largest c with C(c, i) ≤ rank
        rank = rank - table[i][c]
        out[:, k - i] = (n - 1) - c
    return out


def surrogate_energies(
    mu: np.ndarray, sigma: np.ndarray, risk_aversion: float, subsets: np.ndarray
) -> np.ndarray:
    """``E(S) = −(1/K)·Σ μ_i + (λ/(2K²))·Σ_{i,j∈S} Σ_ij`` for every row of ``subsets`` (``(M, K)``).

    Equal to the energy of :func:`cif.qubo.builder.build_selection_bqm` on ``|S| = K`` (the penalty
    cancels) and to ``problem.objective_value(1_S / K)``. Gathers are summed in a fixed order of
    positions, so the result is reproducible bit for bit.
    """
    subsets = np.asarray(subsets)
    if subsets.ndim != 2 or subsets.shape[1] < 1:
        raise ValueError(f"subsets must be a 2-D array with K >= 1 columns, got {subsets.shape}")
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
        raise ValueError("mu and sigma must be finite")
    n, k = mu.shape[0], subsets.shape[1]
    cols = np.ascontiguousarray(subsets.T, dtype=np.intp)  # (K, M): contiguous gathers
    flat = sigma.ravel()
    linear = np.zeros(cols.shape[1])
    quad = np.zeros(cols.shape[1])
    for a in range(k):
        linear += mu[cols[a]]
        quad += flat[cols[a] * (n + 1)]  # Σ_aa
        base = cols[a] * n
        for b in range(a + 1, k):
            quad += 2.0 * flat[base + cols[b]]
    return -linear / k + (risk_aversion / (2.0 * k * k)) * quad


def best_subsets(
    problem: PortfolioProblem, k: int, m: int, *, max_subsets: int = MAX_SUBSETS
) -> tuple[np.ndarray, np.ndarray]:
    """``T_m``: the ``m`` K-subsets of lowest surrogate energy, ascending, and their energies.

    Returns ``(subsets, energies)`` with ``min(m, C(N, K))`` rows. Ties are broken by the
    lexicographic order of the sorted index tuple, so ``T_m`` is a prefix of ``T_m'`` for ``m < m'``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    subsets = enumerate_subsets(problem.n, k, max_subsets=max_subsets)
    energy = surrogate_energies(problem.mu, problem.sigma, problem.risk_aversion, subsets)
    m = min(int(m), energy.size)
    if m < energy.size:
        kth = np.partition(energy, m - 1)[m - 1]
        cand = np.flatnonzero(energy <= kth)  # ascending index = lexicographic rank
    else:
        cand = np.arange(energy.size)
    pick = cand[np.argsort(energy[cand], kind="stable")[:m]]  # stable: ties keep lexicographic order
    return subsets[pick].astype(np.intp), energy[pick]


# --------------------------------------------------------------------------------------------
# F(S) through the shared refinement
# --------------------------------------------------------------------------------------------


class SubsetValues:
    """``F(S) = f(refine(S))`` via :func:`cif.classical.refine.refine_problem`, cached by subset.

    One instance serves the oracle's ``T_m`` and every pool of the same problem, so a subset is
    refined once. ``n_calls`` counts requests, ``n_refined`` the refinements actually run (cache
    misses), ``time_s`` their wall time and ``statuses`` the refinement statuses. An infeasible or
    failed refinement gives ``F = +inf``.
    """

    def __init__(self, problem: PortfolioProblem, card_mode: str = "le") -> None:
        self.problem = problem
        self.card_mode = card_mode
        self._cache: dict[tuple[int, ...], float] = {}
        self.n_calls = 0
        self.n_refined = 0
        self.time_s = 0.0
        self.statuses: Counter[str] = Counter()

    def __call__(self, subset: Iterable[int]) -> float:
        key = tuple(sorted({int(i) for i in subset}))
        self.n_calls += 1
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        t0 = time.perf_counter()
        res = refine_problem(self.problem, key, self.card_mode)
        self.time_s += time.perf_counter() - t0
        self.n_refined += 1
        self.statuses[res.status] += 1
        value = float(res.objective)
        self._cache[key] = value
        return value


def objective_scale(problem: PortfolioProblem, weights: np.ndarray) -> float:
    """``s = |μᵀw| + (λ/2)·wᵀΣw``: the positive scale that normalises gaps (ANALYSIS_PLAN §5).

    Since ``|f(w)| ≤ s`` it never passes through zero where ``f`` changes sign. The shared
    :func:`cif.metrics.gap.reference_scale`, under the oracle's name.
    """
    return reference_scale(problem, weights)


# --------------------------------------------------------------------------------------------
# The reference f*
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Reference:
    """The optimum ``f*`` of the true problem, its weights and the scale ``s*`` of the gaps.

    ``ref_type`` is ``"certified"`` (SCIP proved the incumbent within its gap tolerance, or all
    K-subsets were enumerated) or ``"best_known"``. ``source`` is ``"scip"`` or ``"enumeration"``;
    ``meta`` holds the solver status and bounds.
    """

    f_star: float
    scale: float
    ref_type: str
    source: str
    weights: np.ndarray
    meta: dict = field(default_factory=dict)


def exhaustive_optimum(
    problem: PortfolioProblem,
    k: int,
    card_mode: str = "le",
    *,
    values: SubsetValues | None = None,
    max_subsets: int = 20_000,
) -> Reference:
    """``f* = min_{|S|=K} F(S)`` over every K-subset, for tiny instances (tests, spot checks).

    ``card_mode`` is ignored if ``values`` (which carries its own) is given. ``max_subsets`` is
    deliberately small: every subset costs a refinement.
    """
    subsets = enumerate_subsets(problem.n, k, max_subsets=max_subsets)
    values = values if values is not None else SubsetValues(problem, card_mode)
    f = np.array([values(s) for s in subsets.tolist()])
    best = subsets[int(np.argmin(f))].tolist()  # first minimum: lexicographically smallest subset
    res = refine_problem(problem, best, values.card_mode)
    if res.status != "optimal":
        raise RuntimeError(f"no feasible K-subset: refinement of {best} is {res.status}")
    return Reference(
        f_star=float(res.objective),
        scale=objective_scale(problem, res.weights),
        ref_type="certified",
        source="enumeration",
        weights=res.weights,
        meta={"n_subsets": int(f.size), "support": [int(i) for i in best]},
    )


def certified_optimum(
    problem: PortfolioProblem,
    k: int,
    card_mode: str = "le",
    *,
    time_limit_s: float | None = None,
    mip_gap: float = 1e-6,
    formulation: str = "perspective",
    diag: str = "mineig",
    seed: int = 0,
    tol: float = 1e-6,
) -> Reference:
    """``f*`` from :func:`cif.classical.scip_native.solve_miqp_native`, with its certificate.

    ``card_mode`` goes to SCIP and to the polishing refinement: ``"le"`` (``Σz ≤ K``) or ``"eq"``
    (``Σz = K``, every asset of the support held with at least ``w_min``).

    The incumbent's support is polished with the shared refinement, ``f* = min(f_SCIP, F(supp))``:
    SCIP's weights are accurate to its feasibility tolerance, the refinement to 1e-10, and the
    oracle then compares like with like (the optimal subset has ``E = 0`` to rounding). The
    certificate is unaffected: ``dual ≤ f_opt ≤ f* ≤ f_SCIP``. ``ref_type`` follows
    :func:`cif.metrics.gap.reference_from_exact` with ``tol`` on the scale ``s_ref`` of SCIP's
    incumbent; since ``|f| ≤ s``, a relative gap of ``tol`` implies an absolute gap of at most
    ``tol·s_ref`` (the criterion of ANALYSIS_PLAN §5).
    ``meta`` carries the SCIP status, bounds, gap, nodes and times, ``f_scip`` and ``polish_gain``.
    """
    sol = solve_miqp_native(
        problem, k, formulation=formulation, diag=diag, card_mode=card_mode,
        time_limit_s=time_limit_s, mip_gap=mip_gap, seed=seed,
    )
    meta = sol.solver_meta
    if not np.isfinite(sol.objective) or not math.isfinite(float(meta["primal_bound"])):
        raise RuntimeError(f"SCIP returned no incumbent (status {meta.get('status')})")
    _, ref_type = reference_from_exact(meta, scale=reference_scale(problem, sol.weights), tol=tol)
    support = [int(i) for i in np.flatnonzero(sol.weights > _SUPPORT_TOL)]
    polished = refine_problem(problem, support, card_mode)
    if polished.status == "optimal" and polished.objective <= sol.objective:
        f_star, weights = float(polished.objective), polished.weights
    else:
        f_star, weights = float(sol.objective), sol.weights
    return Reference(
        f_star=f_star,
        scale=objective_scale(problem, weights),
        ref_type=ref_type,
        source="scip",
        weights=weights,
        meta={
            "status": meta["status"],
            "gap": meta["gap"],
            "primal_bound": meta["primal_bound"],
            "dual_bound": meta["dual_bound"],
            "f_scip": float(sol.objective),
            "polish_gain": float(sol.objective) - f_star,
            "n_nodes": meta["n_nodes"],
            "build_time_s": meta["build_time_s"],
            "solve_time_s": meta["solve_time_s"],
            "time_limit_hit": meta["time_limit_hit"],
            "formulation": formulation,
            "diag_method": meta["diag_method"],
            "seed": seed,
            "support": support,
        },
    )


# --------------------------------------------------------------------------------------------
# Pools and prefix minima
# --------------------------------------------------------------------------------------------


def pool_subsets(
    pool: Iterable[Iterable[int]], k: int, n: int, limit: int
) -> tuple[list[tuple[int, ...]], dict]:
    """Valid unique K-subsets of a pool in the given order, at most ``limit`` of them.

    An entry is valid if it consists of ``k`` distinct indices in ``range(n)``; the first
    occurrence of a subset counts. ``stats`` describes the *whole* input: ``n_input`` entries,
    ``n_valid`` valid ones, ``n_unique`` distinct valid ones (the number of subsets a pool offers).
    """
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    n_input = n_valid = 0
    for item in pool:
        n_input += 1
        subset = tuple(sorted({int(i) for i in item}))
        if len(subset) != k or subset[0] < 0 or subset[-1] >= n:
            continue
        n_valid += 1
        if subset in seen:
            continue
        seen.add(subset)
        if len(out) < limit:
            out.append(subset)
    return out, {"n_input": n_input, "n_valid": n_valid, "n_unique": len(seen)}


def prefix_minima(values: Sequence[float], ms: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """``min(values[:m])`` and the number of values used, ``min(m, len(values))``, for every m.

    The minimum is NaN where nothing is available (count 0).
    """
    arr = np.asarray(values, dtype=float)
    best = np.full(len(ms), np.nan)
    count = np.zeros(len(ms), dtype=int)
    for j, m in enumerate(ms):
        count[j] = min(int(m), arr.size)
        if count[j]:
            best[j] = arr[: count[j]].min()
    return best, count


def _check_ms(ms: Sequence[int]) -> tuple[int, ...]:
    out = tuple(int(m) for m in ms)
    if not out or out[0] < 1 or any(b <= a for a, b in pairwise(out)):
        raise ValueError(f"ms must be a non-empty, positive, strictly increasing sequence, got {ms}")
    return out


def _json_float(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's τ-b over the finite pairs; NaN with fewer than 3 pairs or a constant input."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.ptp(a[keep]) == 0.0 or np.ptp(b[keep]) == 0.0:
        return math.nan
    return float(kendalltau(a[keep], b[keep]).statistic)


# --------------------------------------------------------------------------------------------
# The attribution
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Attribution:
    """``E_form``, ``E_pipe`` and ``E_samp`` for every m of ``ms``, in units of the scale ``s*``.

    ``n_form`` / ``n_pipe`` are the numbers of subsets actually used, ``min(m, ·)``. ``e_pipe`` is
    NaN where the pool has no valid subset. ``e_form_pipe`` is ``E_form(n_pipe(m))``, the ceiling at
    the size of the pool that was used (NaN for an empty pool). ``tau_top`` is Kendall's τ between surrogate energy and
    ``F`` over ``T_{m_max}``, ``tau_pool`` over the evaluated pool. ``pool`` holds the statistics of
    :func:`pool_subsets`; ``refine_s`` / ``n_refined`` the cost of the refinements the pool needed
    (subsets already in the cache are free).
    """

    ms: tuple[int, ...]
    e_form: np.ndarray
    e_pipe: np.ndarray
    n_form: np.ndarray
    n_pipe: np.ndarray
    tau_top: float
    tau_pool: float
    pool: dict
    e_form_pipe: np.ndarray
    refine_s: float = 0.0
    n_refined: int = 0

    @property
    def e_samp(self) -> np.ndarray:
        return self.e_pipe - self.e_form

    @property
    def e_prec(self) -> np.ndarray:
        """``E_pipe(m) − E_form(n_pipe(m))``: the sampler against the oracle at an equal pool size
        (§12 B2). NaN where the pool has no valid subset."""
        return self.e_pipe - self.e_form_pipe

    @property
    def e_short(self) -> np.ndarray:
        """``E_form(n_pipe(m)) − E_form(m) ≥ 0``: the price of a pool smaller than m (§12 B2); zero
        where the pool has m subsets, NaN where it has none."""
        return self.e_form_pipe - self.e_form

    def row(self) -> dict:
        """Flat, JSON-safe registry fields (``None`` for undefined or infinite values)."""
        row: dict = {}
        e_samp, e_prec, e_short = self.e_samp, self.e_prec, self.e_short
        for j, m in enumerate(self.ms):
            row[f"E_form_{m}"] = _json_float(self.e_form[j])
            row[f"E_pipe_{m}"] = _json_float(self.e_pipe[j])
            row[f"E_samp_{m}"] = _json_float(e_samp[j])
            row[f"E_form_at_npipe_{m}"] = _json_float(self.e_form_pipe[j])
            row[f"E_prec_{m}"] = _json_float(e_prec[j])
            row[f"E_short_{m}"] = _json_float(e_short[j])
            row[f"n_form_{m}"] = int(self.n_form[j])
            row[f"n_pipe_{m}"] = int(self.n_pipe[j])
        row["tau_top"] = _json_float(self.tau_top)
        row["tau_pool"] = _json_float(self.tau_pool)
        row.update({f"pool_{key}": int(value) for key, value in self.pool.items()})
        return row


class SurrogateOracle:
    """The ceiling ``E_form`` of one instance and the attribution of any pool against it.

    Construction enumerates all K-subsets, ranks them by surrogate energy and refines ``T_{m_max}``
    (``m_max = ms[-1]``); :meth:`attribute` then needs only the pool. The oracle is a function of
    the instance, ``K``, ``ms`` and the reference, so one object serves every sampler configuration
    and seed of that instance. Timings: ``enum_s`` (enumeration and ranking), ``refine_s``
    (refinement of ``T_{m_max}``). ``card_mode`` is used only if no ``values`` evaluator is given.
    """

    def __init__(
        self,
        problem: PortfolioProblem,
        k: int,
        reference: Reference,
        *,
        ms: Sequence[int] = DEFAULT_MS,
        card_mode: str = "le",
        values: SubsetValues | None = None,
        max_subsets: int = MAX_SUBSETS,
    ) -> None:
        self.problem = problem
        self.k = int(k)
        self.reference = reference
        self.ms = _check_ms(ms)
        self.values = values if values is not None else SubsetValues(problem, card_mode)
        self.n_subsets = math.comb(problem.n, self.k)
        t0 = time.perf_counter()
        self.top, self.top_energy = best_subsets(problem, self.k, self.ms[-1],
                                                 max_subsets=max_subsets)
        self.enum_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.top_values = np.array([self.values(s) for s in self.top.tolist()])
        self.refine_s = time.perf_counter() - t1
        best, self.n_form = prefix_minima(self.top_values, self.ms)
        self.e_form = (best - reference.f_star) / reference.scale
        #: ``E_form(n)`` for every pool size ``n = 1 .. len(T_{m_max})`` (index ``n − 1``): the ceiling at the
        #: size of any pool a sampler returns (§12 B2); the same numbers as ``e_form`` at the ``ms``.
        self.e_form_curve = (np.minimum.accumulate(self.top_values) - reference.f_star) / reference.scale
        self.tau_top = _kendall(self.top_energy, self.top_values)

    def attribute(self, pool: Iterable[Iterable[int]]) -> Attribution:
        """``E_pipe`` and ``E_samp`` of a pool of subsets ordered by sampler energy."""
        subsets, stats = pool_subsets(pool, self.k, self.problem.n, self.ms[-1])
        refined0, time0 = self.values.n_refined, self.values.time_s
        f_pool = np.array([self.values(s) for s in subsets])
        best, n_pipe = prefix_minima(f_pool, self.ms)
        tau_pool = math.nan
        if subsets:
            energy = surrogate_energies(self.problem.mu, self.problem.sigma,
                                        self.problem.risk_aversion, np.array(subsets))
            tau_pool = _kendall(energy, f_pool)
        return Attribution(
            ms=self.ms,
            e_form=self.e_form,
            e_pipe=(best - self.reference.f_star) / self.reference.scale,
            n_form=self.n_form,
            n_pipe=n_pipe,
            tau_top=self.tau_top,
            tau_pool=tau_pool,
            pool=stats,
            e_form_pipe=self.e_form_at(n_pipe),
            refine_s=self.values.time_s - time0,
            n_refined=self.values.n_refined - refined0,
        )

    def e_form_at(self, counts: Sequence[int]) -> np.ndarray:
        """``E_form(n)`` for each ``n`` of ``counts`` (NaN for ``n = 0``): the ceiling of the surrogate for a
        pool of ``n`` subsets, ``n ≤ m_max`` (§12 B2)."""
        counts = np.asarray(counts, dtype=int)
        if counts.size and (counts.min() < 0 or counts.max() > self.e_form_curve.size):
            raise ValueError(f"pool sizes must lie in [0, {self.e_form_curve.size}], got {counts.tolist()}")
        out = np.full(counts.shape, np.nan)
        positive = counts > 0
        out[positive] = self.e_form_curve[counts[positive] - 1]
        return out
