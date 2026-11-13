"""Block bootstrap for dependent data: index generators, automatic block length, intervals.

References
----------
- Politis, D.N., Romano, J.P. (1994). The stationary bootstrap. JASA 89(428), 1303-1313.
- Politis, D.N., Romano, J.P. (1992). A circular block-resampling procedure for stationary
  data. In: LePage, Billard (eds.), Exploring the Limits of Bootstrap, Wiley, 263-270.
- Politis, D.N., White, H. (2004). Automatic block-length selection for the dependent
  bootstrap. Econometric Reviews 23(1), 53-70 (PW).
- Patton, A., Politis, D.N., White, H. (2009). Correction to "Automatic block-length selection
  for the dependent bootstrap". Econometric Reviews 28(4), 372-375 (PPW).

Every random function takes an explicit seed (``int``) or ``numpy.random.Generator``; there is
no hidden global state. Results are deterministic for a given seed and argument list.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

SeedLike = int | np.random.Generator
BootstrapMethod = Literal["stationary", "circular", "iid"]

# Bootstrap replicates are generated in chunks of this size to bound memory. The value is part of
# the random-number stream: changing it changes results for a given seed.
_CHUNK = 256

__all__ = [
    "BootstrapResult",
    "OptimalBlockLength",
    "bootstrap_ci",
    "circular_block_bootstrap_indices",
    "optimal_block_length",
    "stationary_bootstrap_indices",
]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _as_generator(seed: SeedLike) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, int | np.integer) and not isinstance(seed, bool):
        return np.random.default_rng(int(seed))
    raise TypeError("seed must be an int or a numpy.random.Generator (explicit seeds only)")


def _seed_record(seed: SeedLike) -> int | None:
    return None if isinstance(seed, np.random.Generator) else int(seed)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


# ---------------------------------------------------------------------------------------------
# index generators
# ---------------------------------------------------------------------------------------------


def stationary_bootstrap_indices(
    n: int,
    mean_block: float,
    n_boot: int,
    rng: SeedLike,
    *,
    length: int | None = None,
) -> np.ndarray:
    """Index matrix of the stationary bootstrap (Politis & Romano 1994).

    Each row is one resample of length ``length`` (default ``n``) of the positions ``0..n-1``.
    Block starts are uniform on ``0..n-1``; block lengths are geometric on ``{1, 2, ...}`` with
    mean ``mean_block`` (a new block starts at each position with probability
    ``p = 1/mean_block``); blocks wrap around circularly.

    Returns an integer array of shape ``(n_boot, length)``.
    """
    n = _positive_int(n, "n")
    n_boot = _positive_int(n_boot, "n_boot")
    length = n if length is None else _positive_int(length, "length")
    if not (np.isfinite(mean_block) and mean_block >= 1.0):
        raise ValueError(f"mean_block must be a finite number >= 1, got {mean_block!r}")
    gen = _as_generator(rng)
    total = n_boot * length
    new_block = gen.random(total) < 1.0 / float(mean_block)
    new_block[::length] = True  # every resample starts a block
    start_pos = np.flatnonzero(new_block)
    start_val = gen.integers(0, n, size=start_pos.size)
    block_id = np.cumsum(new_block, dtype=np.int32 if total < 2**31 else np.int64)
    block_id -= 1
    idx = start_val[block_id]
    idx += np.arange(total)
    idx -= start_pos[block_id]
    if length <= n:  # start + offset <= 2n - 2: one wrap at most
        np.subtract(idx, n, out=idx, where=idx >= n)
    else:
        idx %= n
    return idx.reshape(n_boot, length)


def circular_block_bootstrap_indices(n: int, block: int, n_boot: int, rng: SeedLike) -> np.ndarray:
    """Index matrix of the circular block bootstrap (Politis & Romano 1992).

    ``ceil(n / block)`` blocks of fixed length ``block`` with uniform starts on ``0..n-1`` are
    concatenated (wrapping around the end of the sample) and truncated to length ``n``.

    Returns an integer array of shape ``(n_boot, n)``.
    """
    n = _positive_int(n, "n")
    n_boot = _positive_int(n_boot, "n_boot")
    block = _positive_int(block, "block")
    if block > n:
        raise ValueError(f"block ({block}) must not exceed n ({n})")
    gen = _as_generator(rng)
    n_blocks = -(-n // block)
    starts = gen.integers(0, n, size=(n_boot, n_blocks))
    idx = starts[:, :, None] + np.arange(block)
    np.subtract(idx, n, out=idx, where=idx >= n)  # start + offset <= 2n - 2 since block <= n
    return np.ascontiguousarray(idx.reshape(n_boot, n_blocks * block)[:, :n])


def _iid_indices(n: int, n_boot: int, gen: np.random.Generator) -> np.ndarray:
    return gen.integers(0, n, size=(n_boot, n))


# ---------------------------------------------------------------------------------------------
# automatic block length: Politis & White (2004) with the PPW (2009) correction
# ---------------------------------------------------------------------------------------------


class OptimalBlockLength(NamedTuple):
    """Estimated optimal (expected) block lengths: ``stationary`` for the stationary bootstrap,
    ``circular`` for the circular block bootstrap. Floats for 1-D input, arrays for 2-D input."""

    stationary: Any
    circular: Any


def _flat_top(s: np.ndarray) -> np.ndarray:
    """Trapezoidal flat-top lag window: 1 on |s| <= 1/2, 2(1-|s|) on 1/2 < |s| <= 1, else 0."""
    a = np.abs(s)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def _pw_block_lengths(G: float, g0: float, n: int) -> tuple[float, float]:
    """Optimal block lengths given G = sum |k| R(k) and g(0) = sum R(k).

    b_opt = (2 G^2 / D)^(1/3) n^(1/3) (PW 2004, Theorem 3.1, eq. (6); estimator eq. (9) with
    G_hat, g_hat(0)), D_CB = (4/3) g(0)^2 and the corrected D_SB = 2 g(0)^2 (PPW 2009, items
    1, 3-5). Hence b_CB / b_SB = (3/2)^(1/3) exactly.
    """
    if g0 == 0.0:
        return math.inf, math.inf
    n13 = float(n) ** (1.0 / 3.0)
    b_sb = (2.0 * G**2 / (2.0 * g0**2)) ** (1.0 / 3.0) * n13
    b_cb = (2.0 * G**2 / ((4.0 / 3.0) * g0**2)) ** (1.0 / 3.0) * n13
    return b_sb, b_cb


def _single_optimal_block(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    if n < 10:
        raise ValueError("optimal_block_length needs at least 10 observations")
    e = x - x.mean()
    # Tuning constants of the authors' corrected MATLAB code (A. Patton, Dec 2007):
    # c = 2, K_N = max(5, sqrt(log10 n)), m_max = ceil(sqrt n) + K_N, b_max = ceil(min(3 sqrt n, n/3)).
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = math.ceil(math.sqrt(n)) + k_n
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    # Standard full-sample autocovariances R(k) = n^-1 sum (x_t - xbar)(x_{t+k} - xbar), as in PW.
    acov = np.array([e[k:] @ e[: n - k] for k in range(m_max + 1)]) / n
    if acov[0] <= 0.0:
        raise ValueError("optimal_block_length: series has zero variance")
    rho = acov[1:] / acov[0]  # lags 1..m_max
    insignificant = np.abs(rho) < 2.0 * math.sqrt(math.log10(n) / n)
    # m_hat: first lag j starting a run of K_N consecutive insignificant autocorrelations
    # (Politis 2003 rule as implemented in the authors' code); otherwise the largest
    # significant lag.
    runs = np.lib.stride_tricks.sliding_window_view(insignificant, k_n).all(axis=1)
    first_run = int(np.argmax(runs)) + 1
    last_significant = int(np.flatnonzero(~insignificant)[-1]) + 1 if not runs.any() else 0
    m_hat = first_run if runs.any() else last_significant
    M = min(2 * m_hat, m_max)
    k = np.arange(1, M + 1)
    lam = _flat_top(k / M)
    G = 2.0 * float(np.sum(lam * k * acov[1 : M + 1]))
    g0 = float(acov[0] + 2.0 * np.sum(lam * acov[1 : M + 1]))
    b_sb, b_cb = _pw_block_lengths(G, g0, n)
    # Upper cap as in the authors' code; lower floor at 1 (shorter blocks are meaningless).
    return min(max(b_sb, 1.0), b_max), min(max(b_cb, 1.0), b_max)


def optimal_block_length(x: ArrayLike) -> OptimalBlockLength:
    """Politis-White (2004) automatic block length with the PPW (2009) correction.

    For each column of ``x`` (1-D or 2-D, time along axis 0) returns the estimated optimal
    expected block length of the stationary bootstrap and the optimal block length of the
    circular block bootstrap. The flat-top lag window has bandwidth M = 2 m_hat (PW 2004, Sec. 3).
    Values are capped at ``ceil(min(3 sqrt n, n/3))`` and floored at 1. For the circular
    bootstrap PW recommend rounding up to an integer; ``bootstrap_ci`` does that.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError("x must be 1-D or 2-D")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values")
    if arr.ndim == 1:
        b_sb, b_cb = _single_optimal_block(arr)
        return OptimalBlockLength(b_sb, b_cb)
    res = [_single_optimal_block(arr[:, j]) for j in range(arr.shape[1])]
    return OptimalBlockLength(np.array([r[0] for r in res]), np.array([r[1] for r in res]))


# ---------------------------------------------------------------------------------------------
# bootstrap confidence intervals
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap distribution and intervals of a statistic.

    ``ci_percentile`` = (q_{alpha/2}, q_{1-alpha/2}) of the bootstrap replicates;
    ``ci_basic`` = (2 theta_hat - q_{1-alpha/2}, 2 theta_hat - q_{alpha/2}).
    ``block`` is the (mean) block length actually used (``None`` for ``iid``).
    """

    estimate: Any
    ci_percentile: tuple[Any, Any]
    ci_basic: tuple[Any, Any]
    se: Any
    samples: np.ndarray
    method: str
    block: float | int | None
    n_boot: int
    alpha: float
    seed: int | None


def _draw_indices(
    method: str, n: int, block: float | int | None, m: int, gen: np.random.Generator
) -> np.ndarray:
    if method == "stationary":
        return stationary_bootstrap_indices(n, float(block), m, gen)  # type: ignore[arg-type]
    if method == "circular":
        return circular_block_bootstrap_indices(n, int(block), m, gen)  # type: ignore[arg-type]
    return _iid_indices(n, m, gen)


def _resolve_block(
    method: str, block: float | int | None, data: list[np.ndarray], n: int
) -> float | int | None:
    if method == "iid":
        if block not in (None, 1):
            raise ValueError("block must be None for the iid bootstrap")
        return None
    if block is None:
        cols = [
            np.asarray(a, dtype=float).reshape(n, -1)[:, j]
            for a in data
            for j in range(int(np.prod(a.shape[1:], dtype=int)))
        ]
        est = [optimal_block_length(c) for c in cols]
        if method == "stationary":
            return float(max(e.stationary for e in est))
        return int(min(math.ceil(max(e.circular for e in est)), n))
    if method == "stationary":
        if not (np.isfinite(block) and block >= 1.0):
            raise ValueError("mean block length must be >= 1")
        return float(block)
    if int(block) != block or not 1 <= int(block) <= n:
        raise ValueError("circular block length must be an integer in [1, n]")
    return int(block)


def _scalar_or_array(x: np.ndarray) -> Any:
    return float(x) if x.ndim == 0 else x


def bootstrap_ci(
    stat_fn: Callable[..., Any],
    *arrays: ArrayLike,
    method: BootstrapMethod = "stationary",
    block: float | int | None = None,
    n_boot: int = 9999,
    alpha: float = 0.05,
    seed: SeedLike,
    vectorized: bool = False,
) -> BootstrapResult:
    """Bootstrap percentile and basic intervals for ``stat_fn(*arrays)``.

    Parameters
    ----------
    stat_fn
        Statistic. Called as ``stat_fn(*resampled_arrays)``; must return a scalar or a fixed-shape
        array. With ``vectorized=True`` it receives arrays with a leading batch axis
        (``(B, n, ...)``) and must return shape ``(B, ...)``.
    *arrays
        Data with time along axis 0 and a common length ``n``. Paired arrays are resampled with
        the same indices (rows are kept together).
    method
        ``"stationary"`` (Politis & Romano 1994; ``block`` = mean block length), ``"circular"``
        (fixed integer ``block``) or ``"iid"``.
    block
        ``None`` selects the Politis-White (2004/2009) length: the maximum over all columns of all
        arrays (rounded up for the circular bootstrap). Note: for statistics such as the Sharpe
        ratio the dependence of squared returns also matters; pass ``block`` explicitly if needed.
    n_boot, alpha
        Number of replicates and the two-sided level (interval coverage ``1 - alpha``).
    seed
        Mandatory explicit seed (``int``) or ``numpy.random.Generator``.
    """
    if not arrays:
        raise ValueError("at least one data array is required")
    if method not in ("stationary", "circular", "iid"):
        raise ValueError(f"unknown method {method!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n_boot = _positive_int(n_boot, "n_boot")
    data = [np.asarray(a) for a in arrays]
    if any(a.ndim == 0 for a in data):
        raise ValueError("data arrays must have a time axis")
    n = data[0].shape[0]
    if any(a.shape[0] != n for a in data):
        raise ValueError("all arrays must have the same length along axis 0")
    if n < 2:
        raise ValueError("need at least two observations")
    gen = _as_generator(seed)
    block_used = _resolve_block(method, block, data, n)

    if vectorized:
        estimate = np.asarray(stat_fn(*[a[None] for a in data]), dtype=float)[0]
    else:
        estimate = np.asarray(stat_fn(*data), dtype=float)
    samples = np.empty((n_boot, *estimate.shape))
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = _draw_indices(method, n, block_used, m, gen)
        if vectorized:
            out = np.asarray(stat_fn(*[a[idx] for a in data]), dtype=float)
            if out.shape != (m, *estimate.shape):
                raise ValueError(
                    f"vectorized stat_fn returned shape {out.shape}, expected {(m, *estimate.shape)}"
                )
            samples[start : start + m] = out
        else:
            for i in range(m):
                samples[start + i] = stat_fn(*[a[idx[i]] for a in data])
    bad = ~np.isfinite(samples)
    if bad.any():
        raise ValueError(
            f"stat_fn returned non-finite values in {int(bad.reshape(n_boot, -1).any(axis=1).sum())}"
            f" of {n_boot} bootstrap replicates"
        )
    q_lo, q_hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    return BootstrapResult(
        estimate=_scalar_or_array(estimate),
        ci_percentile=(_scalar_or_array(q_lo), _scalar_or_array(q_hi)),
        ci_basic=(_scalar_or_array(2.0 * estimate - q_hi), _scalar_or_array(2.0 * estimate - q_lo)),
        se=_scalar_or_array(samples.std(axis=0, ddof=1)),
        samples=samples,
        method=method,
        block=block_used,
        n_boot=n_boot,
        alpha=alpha,
        seed=_seed_record(seed),
    )
