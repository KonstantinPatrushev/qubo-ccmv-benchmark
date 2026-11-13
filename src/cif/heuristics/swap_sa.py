"""Simulated annealing over K-subsets with cardinality-preserving swap moves.

The sampler works directly on the feasible manifold of the binary-inclusion
QUBO: a state is a size-``K`` subset ``S`` and a move exchanges one held asset
for one that is not held. There is no cardinality penalty, so a swap never has
to climb through an infeasible intermediate state (the barrier a single-flip
sampler meets at ``lambda_c = 100 x scale``).

The energy is the penalty-free surrogate of
:func:`cif.qubo.builder.build_selection_bqm`::

    E(S) = -(1/K) sum_{i in S} mu_i + (lambda / (2 K^2)) sum_{i,j in S} Sigma_ij

For a swap ``i -> j`` (``i`` leaves, ``j`` enters) with the gain vector
``g = Sigma[:, S].sum(axis=1)``::

    dQ = -2 g_i + Sigma_ii + 2 g_j - 2 Sigma_ij + Sigma_jj
    dE = (mu_i - mu_j) / K + (lambda / (2 K^2)) dQ

and ``g += Sigma[:, j] - Sigma[:, i]`` after an accepted move, so one proposal
is O(1) and an accepted move is O(N).

All reads advance in lockstep as numpy arrays (no numba in the environment);
reads are independent chains that share the temperature schedule. A sweep is
``N`` proposals per read.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cif.heuristics.pipeline import refine_candidates
from cif.problem import PortfolioProblem, Solution

__all__ = [
    "SwapSAResult",
    "SwapState",
    "SwapSurrogate",
    "calibrate_hot_temperature",
    "sample_swap_sa",
    "solve_with_swap_sa",
]


@dataclass
class SwapState:
    """Batch of independent chains, one row per read."""

    inside: np.ndarray  # (R, K) assets held, unordered
    outside: np.ndarray  # (R, N-K) assets not held, unordered
    gain: np.ndarray  # (R, N) g = Sigma[:, S].sum(axis=1)
    energy: np.ndarray  # (R,) E(S), updated incrementally


class SwapSurrogate:
    """Penalty-free binary-inclusion energy with O(1) swap deltas.

    ``sigma`` must be symmetric (the gain vector uses ``Sigma[b, :]`` for
    ``Sigma[:, b]``), as every covariance matrix in the project is.
    """

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        risk_aversion: float,
        cardinality: int,
    ) -> None:
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.ascontiguousarray(sigma, dtype=float)
        self.n = int(self.mu.shape[0])
        self.K = int(cardinality)
        if not (1 <= self.K <= self.n):
            raise ValueError(f"cardinality must be in [1, {self.n}], got {self.K}")
        self._a = 1.0 / self.K
        self._b = float(risk_aversion) / (2.0 * self.K * self.K)
        self._diag = np.diag(self.sigma).copy()

    def energy(self, subset) -> float:
        """E(S) recomputed from scratch for one subset."""
        idx = np.asarray(subset, dtype=int)
        return float(
            -self._a * self.mu[idx].sum() + self._b * self.sigma[np.ix_(idx, idx)].sum()
        )

    def init_state(self, subsets: np.ndarray) -> SwapState:
        """State for the given ``(R, K)`` array of subsets (rows of distinct indices)."""
        inside = np.array(subsets, dtype=np.int64, copy=True)
        reads = inside.shape[0]
        held = np.zeros((reads, self.n), dtype=bool)
        np.put_along_axis(held, inside, True, axis=1)
        # np.nonzero is row-major, so every row contributes its N-K entries in turn
        outside = np.nonzero(~held)[1].reshape(reads, self.n - self.K).astype(np.int64)
        gain = self.sigma[inside].sum(axis=1)
        rows = np.arange(reads)[:, None]
        energy = -self._a * self.mu[inside].sum(axis=1) + self._b * gain[rows, inside].sum(axis=1)
        return SwapState(inside=inside, outside=outside, gain=gain, energy=energy)

    def random_state(self, num_reads: int, rng: np.random.Generator) -> SwapState:
        """Uniformly random size-K subsets, one per read."""
        order = rng.permuted(np.tile(np.arange(self.n), (num_reads, 1)), axis=1)
        return self.init_state(order[:, : self.K])

    # -- moves -----------------------------------------------------------

    def _delta(self, gain: np.ndarray, rows: np.ndarray, i: np.ndarray, j: np.ndarray):
        d = self._diag
        dq = d[i] + d[j] - 2.0 * gain[rows, i] + 2.0 * gain[rows, j] - 2.0 * self.sigma[i, j]
        return self._a * (self.mu[i] - self.mu[j]) + self._b * dq

    def swap_delta(self, state: SwapState, pos_in: np.ndarray, pos_out: np.ndarray) -> np.ndarray:
        """dE of swapping ``inside[r, pos_in[r]]`` with ``outside[r, pos_out[r]]``."""
        rows = np.arange(state.inside.shape[0])
        i = state.inside[rows, pos_in]
        j = state.outside[rows, pos_out]
        return self._delta(state.gain, rows, i, j)

    def _commit(
        self,
        state: SwapState,
        idx: np.ndarray,
        pos_in: np.ndarray,
        pos_out: np.ndarray,
        i: np.ndarray,
        j: np.ndarray,
        delta: np.ndarray,
    ) -> None:
        state.inside[idx, pos_in[idx]] = j[idx]
        state.outside[idx, pos_out[idx]] = i[idx]
        state.gain[idx] += self.sigma[j[idx]] - self.sigma[i[idx]]
        state.energy[idx] += delta[idx]

    def apply_swap(
        self,
        state: SwapState,
        pos_in: np.ndarray,
        pos_out: np.ndarray,
        accept: np.ndarray,
        delta: np.ndarray,
    ) -> None:
        """Execute the accepted swaps in place (``delta`` from :meth:`swap_delta`)."""
        idx = np.flatnonzero(accept)
        if idx.size == 0:
            return
        rows = np.arange(state.inside.shape[0])
        i = state.inside[rows, pos_in]
        j = state.outside[rows, pos_out]
        self._commit(state, idx, pos_in, pos_out, i, j, delta)


def calibrate_hot_temperature(
    surrogate: SwapSurrogate,
    rng: np.random.Generator,
    accept_target: float = 0.8,
    num_states: int = 64,
    swaps_per_state: int = 16,
) -> float:
    """Temperature at which random swaps are accepted with probability ``accept_target``.

    Draws ``num_states * swaps_per_state`` random swaps from random subsets and
    solves ``mean(min(1, exp(-dE / T))) = accept_target`` for ``T`` by
    bisection. The scale of the answer is the mean ``|dE|`` of random swaps.
    If ``accept_target`` does not exceed the share of downhill swaps, any ``T``
    meets it and the mean ``|dE|`` itself is returned.
    """
    if not 0.0 < accept_target < 1.0:
        raise ValueError(f"accept_target must be in (0, 1), got {accept_target}")
    n, K = surrogate.n, surrogate.K
    state = surrogate.random_state(num_states, rng)
    deltas = np.concatenate(
        [
            surrogate.swap_delta(
                state,
                rng.integers(0, K, size=num_states),
                rng.integers(0, n - K, size=num_states),
            )
            for _ in range(swaps_per_state)
        ]
    )
    scale = float(np.abs(deltas).mean())
    if scale <= 0.0:
        return 1e-12
    uphill = np.maximum(deltas, 0.0)

    def acceptance(temperature: float) -> float:
        return float(np.mean(np.exp(-uphill / temperature)))

    lo, hi = 1e-9 * scale, 1e3 * scale
    if acceptance(lo) >= accept_target:  # target below the downhill share: any T works
        return scale
    for _ in range(80):
        mid = float(np.sqrt(lo * hi))
        if acceptance(mid) < accept_target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


@dataclass(frozen=True)
class SwapSAResult:
    """Outcome of :func:`sample_swap_sa`.

    ``subsets``/``energies`` are the unique final subsets over all reads in
    ascending surrogate energy. ``read_subsets``/``read_energies`` keep every
    read (rows sorted). ``frozen_fraction`` is the share of reads whose subset
    did not change after the first third of the sweeps; ``accept_rate`` is
    the Metropolis acceptance averaged over the sweeps that were run.

    ``num_sweeps`` is the requested schedule, ``sweeps_done`` the number of sweeps actually run;
    ``interrupted`` is ``True`` if the ``interrupt_function`` cut the run short, and the chains
    then stand at the temperature of the last sweep (not annealed to ``t_cold``; ``frozen_fraction``
    is not meaningful before the first third of the schedule).
    """

    subsets: list[tuple[int, ...]]
    energies: np.ndarray
    read_subsets: np.ndarray
    read_energies: np.ndarray
    t_hot: float
    t_cold: float
    accept_rate: float
    frozen_fraction: float
    sample_wall_s: float
    num_reads: int
    num_sweeps: int
    sweeps_done: int = 0
    interrupted: bool = False


def sample_swap_sa(
    problem: PortfolioProblem,
    K: int,
    *,
    num_reads: int = 100,
    num_sweeps: int = 200,
    seed: int | None = None,
    t_hot: float | None = None,
    t_cold: float | None = None,
    accept_start: float = 0.8,
    cold_ratio: float = 0.01,
    interrupt_function: Callable[[], bool] | None = None,
) -> SwapSAResult:
    """Anneal ``num_reads`` chains over size-``K`` subsets with swap moves.

    Parameters
    ----------
    num_reads:
        Independent chains; each contributes its final subset.
    num_sweeps:
        Number of temperature levels; each is held for one sweep of ``N``
        swap proposals per read.
    t_hot, t_cold:
        Explicit end points of the geometric schedule (same units as the
        energy). By default ``t_hot`` is calibrated on random swaps so that
        ``accept_start`` of them are accepted
        (:func:`calibrate_hot_temperature`) and
        ``t_cold = cold_ratio * t_hot``.
    seed:
        Seeds a ``numpy`` generator; the run is reproducible for a given seed.
    interrupt_function:
        Called without arguments before every sweep, as ``dwave-samplers``' simulated annealing
        calls its function between reads (ANALYSIS_PLAN §6: the budget of the anytime outer loop).
        If it returns ``True`` the run stops at once and returns the chains as they are, with
        ``interrupted=True``. The reads advance in lockstep, so the granularity is one sweep (``N``
        proposals per read). Without it the run and its random stream are unchanged.

    ``SwapSAResult.sample_wall_s`` includes the temperature calibration.
    """
    K = int(K)
    n = problem.n
    reads = int(num_reads)
    sweeps = int(num_sweeps)
    if reads < 1 or sweeps < 1:
        raise ValueError(f"num_reads and num_sweeps must be >= 1, got {reads}, {sweeps}")
    surrogate = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    if n == K:  # the only subset: nothing to swap
        full = np.tile(np.arange(n), (reads, 1))
        energy = surrogate.init_state(full).energy
        return SwapSAResult(
            subsets=[tuple(range(n))],
            energies=energy[:1].copy(),
            read_subsets=full,
            read_energies=energy,
            t_hot=0.0,
            t_cold=0.0,
            accept_rate=0.0,
            frozen_fraction=1.0,
            sample_wall_s=time.perf_counter() - t0,
            num_reads=reads,
            num_sweeps=sweeps,
        )

    hot = calibrate_hot_temperature(surrogate, rng, accept_start) if t_hot is None else float(t_hot)
    cold = hot * cold_ratio if t_cold is None else float(t_cold)
    if not hot >= cold > 0.0:
        raise ValueError(f"need t_hot >= t_cold > 0, got {hot}, {cold}")
    temperatures = np.geomspace(hot, cold, sweeps)

    state = surrogate.random_state(reads, rng)
    rows = np.arange(reads)
    third = max(1, sweeps // 3)
    snapshot = np.sort(state.inside, axis=1)
    accepted = 0
    sweeps_done = 0
    interrupted = False
    for sweep in range(sweeps):
        if interrupt_function is not None and interrupt_function():
            interrupted = True
            break
        inv_t = 1.0 / temperatures[sweep]
        pos_in = rng.integers(0, K, size=(n, reads))
        pos_out = rng.integers(0, n - K, size=(n, reads))
        uniform = rng.random((n, reads))
        for step in range(n):
            p_in, p_out = pos_in[step], pos_out[step]
            i = state.inside[rows, p_in]
            j = state.outside[rows, p_out]
            delta = surrogate._delta(state.gain, rows, i, j)
            accept = (delta <= 0.0) | (uniform[step] < np.exp(-np.maximum(delta, 0.0) * inv_t))
            idx = np.flatnonzero(accept)
            if idx.size:
                accepted += idx.size
                surrogate._commit(state, idx, p_in, p_out, i, j, delta)
        sweeps_done += 1
        if sweep + 1 == third:
            snapshot = np.sort(state.inside, axis=1)

    final = np.sort(state.inside, axis=1)
    energies = surrogate.init_state(final).energy  # exact, no incremental drift
    order = np.argsort(energies, kind="stable")
    seen: set[tuple[int, ...]] = set()
    subsets: list[tuple[int, ...]] = []
    subset_energies: list[float] = []
    for r in order:
        subset = tuple(final[r].tolist())
        if subset not in seen:
            seen.add(subset)
            subsets.append(subset)
            subset_energies.append(float(energies[r]))

    return SwapSAResult(
        subsets=subsets,
        energies=np.asarray(subset_energies),
        read_subsets=final,
        read_energies=energies,
        t_hot=hot,
        t_cold=cold,
        accept_rate=accepted / (reads * n * sweeps_done) if sweeps_done else 0.0,
        frozen_fraction=float(np.mean(np.all(final == snapshot, axis=1))),
        sample_wall_s=time.perf_counter() - t0,
        num_reads=reads,
        num_sweeps=sweeps,
        sweeps_done=sweeps_done,
        interrupted=interrupted,
    )


def solve_with_swap_sa(
    problem: PortfolioProblem,
    K: int | None = None,
    num_reads: int = 100,
    num_sweeps: int = 200,
    seed: int | None = None,
    m: int = 50,
    *,
    refine_continuous: bool = True,
    warm_start_subset: list[int] | tuple[int, ...] | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_cost: float = 0.0,
    t_hot: float | None = None,
    t_cold: float | None = None,
    turnover_in_refinement: bool = False,
) -> Solution:
    """Two-stage pipeline with swap-SA as the sampler.

    Stage 1: :func:`sample_swap_sa` (unique subsets by ascending energy).
    Stage 2: :func:`cif.heuristics.pipeline.refine_candidates` on the ``m``
    lowest-energy subsets, exactly as for ``neal`` and Tabu.
    """
    K = K if K is not None else problem.cardinality
    if K is None:
        raise ValueError("cardinality must be provided either on the problem or explicitly")
    K = int(K)
    res = sample_swap_sa(
        problem,
        K,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
        t_hot=t_hot,
        t_cold=t_cold,
    )
    label = (
        f"swap_sa_selection+cvxpy[{num_reads}x{num_sweeps}]"
        if refine_continuous
        else f"swap_sa_selection[{num_reads}x{num_sweeps}]"
    )
    best, stats = refine_candidates(
        problem,
        res.subsets,
        K,
        m,
        prev_weights=prev_weights,
        turnover_cost=turnover_cost,
        warm_start_subset=warm_start_subset,
        refine_continuous=refine_continuous,
        solver=label,
        turnover_in_refinement=turnover_in_refinement,
    )
    meta = {
        "sample_wall_s": res.sample_wall_s,
        "refine_wall_s": stats["refine_wall_s"],
        "n_candidates": stats["n_candidates"],
        "n_evaluated": stats["n_evaluated"],
        "m": stats["m"],
        "warm_start_in_pool": stats["warm_start_in_pool"],
        "warm_start_selected": stats["warm_start_selected"],
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "t_hot": res.t_hot,
        "t_cold": res.t_cold,
        "accept_rate": res.accept_rate,
        "frozen_fraction": res.frozen_fraction,
        "sample_best_energy": float(res.energies[0]),
    }
    wall = res.sample_wall_s + stats["refine_wall_s"]
    if best is None:  # every convex solve failed: equal weights on the best subset
        subset = res.subsets[0]
        weights = np.zeros(problem.n, dtype=float)
        weights[list(subset)] = 1.0 / K
        return Solution(
            weights=weights,
            objective=problem.objective_value(weights),
            feasible=False,
            wall_time_s=wall,
            solver=label,
            solver_meta={**meta, "fallback_subset": list(subset)},
        )
    best.wall_time_s = wall
    best.solver_meta.update(meta)
    return best
