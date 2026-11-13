"""Multiple testing, equivalence testing and solver-benchmark summaries.

References
----------
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian
  Journal of Statistics 6(2), 65-70.
- Schuirmann, D.J. (1987). A comparison of the two one-sided tests procedure and the power
  approach for assessing the equivalence of average bioavailability. J. Pharmacokinetics and
  Biopharmaceutics 15(6), 657-680. Berger, R.L., Hsu, J.C. (1996). Bioequivalence trials,
  intersection-union tests and equivalence confidence sets. Statistical Science 11(4).
- Dolan, E.D., More, J.J. (2002). Benchmarking optimization software with performance
  profiles. Mathematical Programming 91(2), 201-213.
- Aiex, R.M., Resende, M.G.C., Ribeiro, C.C. (2007). TTT plots: a perl program to create
  time-to-target plots. Optimization Letters 1(4), 355-366.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "TOSTResult",
    "TTTData",
    "holm",
    "performance_profile",
    "performance_ratios",
    "tost_bootstrap",
    "tost_from_ci",
    "ttt_ecdf",
]


def holm(pvals: ArrayLike) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values, returned in the input order and shape.

    p_adj_(i) = max_{j <= i} min(1, (m - j + 1) p_(j)) for the sorted p-values. A hypothesis is
    rejected at family-wise level alpha iff its adjusted p-value is <= alpha.
    """
    p = np.asarray(pvals, dtype=float)
    flat = p.ravel()
    if not np.all(np.isfinite(flat)) or np.any((flat < 0.0) | (flat > 1.0)):
        raise ValueError("p-values must be finite numbers in [0, 1]")
    m = flat.size
    order = np.argsort(flat, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * flat[order]))
    out = np.empty(m)
    out[order] = adjusted
    return out.reshape(p.shape)


@dataclass(frozen=True)
class TOSTResult:
    """Two one-sided tests for equivalence: H1 is margin_low < theta < margin_high.

    ``equivalent`` is True iff the (1 - 2 alpha) interval [ci_low, ci_high] lies strictly inside
    (margin_low, margin_high). For the bootstrap version ``p_lower`` = P*(theta* <= margin_low),
    ``p_upper`` = P*(theta* >= margin_high) and ``pvalue`` = max of the two.
    """

    equivalent: bool
    ci_low: float
    ci_high: float
    margin_low: float
    margin_high: float
    pvalue: float | None = None
    p_lower: float | None = None
    p_upper: float | None = None
    alpha: float | None = None


def _margins(margin: float | tuple[float, float]) -> tuple[float, float]:
    if np.ndim(margin) == 0:
        m = float(margin)  # type: ignore[arg-type]
        if not (np.isfinite(m) and m > 0.0):
            raise ValueError("a symmetric margin must be a positive finite number")
        return -m, m
    lo, hi = (float(v) for v in margin)  # type: ignore[union-attr]
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("margins must be finite with margin_low < margin_high")
    return lo, hi


def tost_from_ci(ci_low: float, ci_high: float, margin: float | tuple[float, float]) -> TOSTResult:
    """TOST decision from a confidence interval (Schuirmann 1987; Berger & Hsu 1996).

    For a level-alpha equivalence test pass the (1 - 2 alpha) two-sided interval (e.g. the 90 %
    interval for alpha = 0.05). ``margin`` is a positive number (region (-m, m)) or a pair
    ``(low, high)``.
    """
    lo, hi = float(ci_low), float(ci_high)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
        raise ValueError("need finite ci_low <= ci_high")
    m_lo, m_hi = _margins(margin)
    return TOSTResult(bool(lo > m_lo and hi < m_hi), lo, hi, m_lo, m_hi)


def tost_bootstrap(
    samples_of_difference: ArrayLike,
    margin: float | tuple[float, float],
    alpha: float = 0.05,
) -> TOSTResult:
    """TOST from bootstrap replicates of a difference (percentile method).

    The decision uses the percentile interval [q_alpha, q_{1-alpha}] (inverted-CDF quantiles) of
    the replicates; the one-sided bootstrap p-values agree with it except at exact ties.
    """
    s = np.asarray(samples_of_difference, dtype=float).ravel()
    if s.size < 2 or not np.all(np.isfinite(s)):
        raise ValueError("need at least two finite bootstrap replicates")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    m_lo, m_hi = _margins(margin)
    lo, hi = np.quantile(s, [alpha, 1.0 - alpha], method="inverted_cdf")
    p_lower = float(np.mean(s <= m_lo))
    p_upper = float(np.mean(s >= m_hi))
    return TOSTResult(
        equivalent=bool(lo > m_lo and hi < m_hi),
        ci_low=float(lo),
        ci_high=float(hi),
        margin_low=m_lo,
        margin_high=m_hi,
        pvalue=max(p_lower, p_upper),
        p_lower=p_lower,
        p_upper=p_upper,
        alpha=float(alpha),
    )


def performance_ratios(times: Mapping[str, ArrayLike]) -> dict[str, np.ndarray]:
    """Performance ratios r_{p,s} = t_{p,s} / min_s t_{p,s} (Dolan & More 2002, Sec. 2).

    ``times`` maps solver name to per-problem costs (same problem order for every solver);
    ``np.inf`` marks a failure and yields r = inf. Problems solved by nobody get r = inf for all
    solvers. NaN and non-positive times are rejected.
    """
    if not times:
        raise ValueError("times is empty")
    names = list(times)
    rows = [np.asarray(times[s], dtype=float).ravel() for s in names]
    if len({r.size for r in rows}) != 1 or rows[0].size == 0:
        raise ValueError("every solver needs a cost for the same, non-empty set of problems")
    t = np.vstack(rows)
    if np.isnan(t).any():
        raise ValueError("NaN in times; encode failures as np.inf")
    if np.any(t <= 0.0):
        raise ValueError("times must be positive")
    best = t.min(axis=0)
    with np.errstate(invalid="ignore"):
        ratios = t / best
    ratios[~np.isfinite(t)] = np.inf
    return {s: ratios[i] for i, s in enumerate(names)}


def performance_profile(times: Mapping[str, ArrayLike], taus: ArrayLike) -> dict[str, np.ndarray]:
    """Performance profiles rho_s(tau) = |{p : r_{p,s} <= tau}| / n_p (Dolan & More 2002).

    Returns, for each solver, the profile evaluated at ``taus``. Failures (``np.inf``) never count.
    """
    tau = np.asarray(taus, dtype=float).ravel()
    if tau.size == 0 or np.isnan(tau).any():
        raise ValueError("taus must be a non-empty array of numbers")
    ratios = performance_ratios(times)
    return {s: (r[None, :] <= tau[:, None]).mean(axis=1) for s, r in ratios.items()}


class TTTData(NamedTuple):
    """Points of a time-to-target plot: sorted successful times and their probabilities."""

    times: np.ndarray
    probabilities: np.ndarray
    n_runs: int
    n_failed: int


def ttt_ecdf(times: ArrayLike) -> TTTData:
    """Empirical distribution for time-to-target plots (Aiex, Resende & Ribeiro 2007).

    The i-th smallest successful time is paired with p_i = (i - 1/2) / n, where n counts all
    runs; failed runs (``np.inf``) are kept in n, so the curve plateaus below one.
    """
    t = np.asarray(times, dtype=float).ravel()
    if t.size == 0 or np.isnan(t).any() or np.any(t < 0.0):
        raise ValueError("times must be a non-empty array of non-negative numbers (inf = failure)")
    ok = np.isfinite(t)
    s = np.sort(t[ok])
    p = (np.arange(1, s.size + 1) - 0.5) / t.size
    return TTTData(s, p, int(t.size), int((~ok).sum()))
