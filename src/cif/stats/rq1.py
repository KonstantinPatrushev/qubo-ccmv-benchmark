"""RQ1 statistics: H1a, the estimates and the Holm family (ANALYSIS_PLAN §2 RQ1, §7.1–§7.2).

The unit of analysis is the instance, or the pair of instances that share a seed and differ in the
level of ``h`` (the volatility dispersion). Building blocks, all on arrays and all deterministic for
an explicit seed:

- :func:`wilson_interval` — Wilson score interval of a share ("доля инстансов с ``E_form(m) > τ``");
- :func:`stratified_signed_rank` — the one-sided Wilcoxon signed-rank test of H1a on the paired
  differences, stratified by N (ranks are taken within a stratum, the statistic is the sum of the
  strata). Zero differences are dropped before ranking (``zero_method="wilcox"`` of scipy, the
  textbook Wilcoxon test); ties get average ranks. The p-value is *exact* given the absolute
  differences: under H0 (symmetry about zero) the signs are exchangeable, so the null distribution
  of the statistic is the distribution of a sum of independent ``rank_i · Bernoulli(1/2)``, computed
  by dynamic programming over the doubled (integer) ranks. It needs no tie correction and no
  normal approximation; ``method="approx"`` gives the normal approximation with the tie correction
  for comparison with scipy, and is used above ``exact_max`` non-zero differences;
- :func:`cluster_bootstrap` — percentile interval of a mean or a median, resampling clusters (the
  pair, whose two levels are dependent) within strata (N);
- :func:`holm_family` — Holm over the RQ1 family {H1a, H1c} (§7.1).

``zero_tol`` of the test is the numerical zero of the differences: ``E_form`` of two instances
whose ceiling reaches the optimum are zero up to the accuracy of the refinement (about 1e-9 of the
scale), and ranking that noise would give it the smallest ranks of the test.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats as sps

from cif.stats.tests_misc import holm

__all__ = [
    "RQ1_FAMILY",
    "BootstrapSummary",
    "SignedRankResult",
    "cluster_bootstrap",
    "cluster_bootstrap_table",
    "holm_family",
    "stratified_signed_rank",
    "wilson_interval",
]

#: The multiplicity family of RQ1 (§7.1): Holm over these two hypotheses.
RQ1_FAMILY = ("H1a", "H1c")


# --------------------------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------------------------


def wilson_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials; ``(nan, nan)`` if ``n = 0``."""
    k, n = int(k), int(n)
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    if n == 0:
        return math.nan, math.nan
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)  # exactly 0 / 1 at the boundary counts,
    high = 1.0 if k == n else min(1.0, centre + half)  # rounding must not move them inside
    return low, high


# --------------------------------------------------------------------------------------------
# Stratified signed-rank test
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedRankResult:
    """Outcome of :func:`stratified_signed_rank`.

    ``statistic`` is the sum over strata of the ranks of the differences in the tested direction
    (positive ones for ``alternative="greater"``). ``expected`` / ``variance`` are its null mean and
    tie-corrected variance (the normal approximation; exact p-values do not use them), ``z`` the
    standardised statistic. ``strata`` maps a stratum label to its counts (``n``, ``n_zero``,
    ``n_nonzero``, ``n_pos``, ``n_neg``), the median of its differences and its part ``t`` of the
    statistic.
    """

    statistic: float
    p_value: float
    method: str
    alternative: str
    n_total: int
    n_nonzero: int
    n_zero: int
    expected: float
    variance: float
    z: float | None
    zero_tol: float
    strata: dict = field(default_factory=dict)


def _exact_upper_tail(doubled_ranks: np.ndarray, observed: int) -> float:
    """``P(Σ w_i B_i ≥ observed)`` for independent fair Bernoulli ``B_i`` and integer weights ``w_i``."""
    total = int(doubled_ranks.sum())
    pmf = np.zeros(total + 1)
    pmf[0] = 1.0
    top = 0  # largest sum reached so far
    for w in doubled_ranks.tolist():
        shifted = np.zeros_like(pmf)
        shifted[w : top + w + 1] = pmf[: top + 1]
        pmf = 0.5 * (pmf + shifted)
        top += w
    if observed > total:
        return 0.0
    return float(min(1.0, max(0.0, pmf[observed:].sum())))


def stratified_signed_rank(
    d: ArrayLike,
    strata: ArrayLike,
    *,
    alternative: str = "greater",
    method: str = "auto",
    zero_tol: float = 0.0,
    exact_max: int = 500,
) -> SignedRankResult:
    """One-sided Wilcoxon signed-rank test of paired differences ``d``, stratified by ``strata``.

    Parameters
    ----------
    d, strata
        Differences (``E_form`` at ``h = 0.45`` minus at ``h = 0.15``, one per pair) and the label of
        the stratum of each (N). Ranks of ``|d|`` are taken within a stratum.
    alternative
        ``"greater"`` (H1a: the differences are shifted above zero) or ``"less"``.
    method
        ``"exact"`` (permutation distribution given the ranks, any ties), ``"approx"`` (normal
        approximation with the tie correction, no continuity correction) or ``"auto"``: exact up to
        ``exact_max`` non-zero differences.
    zero_tol
        ``|d| <= zero_tol`` counts as a zero difference and is dropped before ranking.
    """
    values = np.asarray(d, dtype=float).ravel()
    labels = np.asarray(strata).ravel()
    if values.size == 0 or values.size != labels.size:
        raise ValueError(f"need non-empty d and strata of the same length, got {values.size}, {labels.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("d must be finite")
    if alternative not in ("greater", "less"):
        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative!r}")
    if method not in ("auto", "exact", "approx"):
        raise ValueError(f"method must be 'auto', 'exact' or 'approx', got {method!r}")
    if not zero_tol >= 0.0:
        raise ValueError(f"zero_tol must be >= 0, got {zero_tol}")
    sign = 1.0 if alternative == "greater" else -1.0

    statistic = expected = variance = 0.0
    n_nonzero = 0
    doubled: list[np.ndarray] = []
    per_stratum: dict = {}
    for label in np.unique(labels):
        sub = values[labels == label]
        x = sign * sub
        keep = np.abs(x) > zero_tol
        xn = x[keep]
        n = int(xn.size)
        ranks = sps.rankdata(np.abs(xn), method="average") if n else np.empty(0)
        t = float(ranks[xn > 0.0].sum())
        _, counts = np.unique(np.abs(xn), return_counts=True)
        expected_s = n * (n + 1) / 4.0
        variance_s = n * (n + 1) * (2 * n + 1) / 24.0 - float((counts**3 - counts).sum()) / 48.0
        statistic += t
        expected += expected_s
        variance += variance_s
        n_nonzero += n
        doubled.append(np.rint(2.0 * ranks).astype(np.int64))
        per_stratum[label.item() if hasattr(label, "item") else label] = {
            "n": int(sub.size), "n_zero": int(sub.size - n), "n_nonzero": n,
            "n_pos": int((sub[np.abs(sub) > zero_tol] > 0.0).sum()),
            "n_neg": int((sub[np.abs(sub) > zero_tol] < 0.0).sum()),
            "median_d": float(np.median(sub)), "t": t,
        }

    used = method
    if method == "auto":
        used = "exact" if n_nonzero <= exact_max else "approx"
    z = None
    if n_nonzero == 0:
        p_value = 1.0
    elif used == "exact":
        weights = np.concatenate(doubled)
        p_value = _exact_upper_tail(weights, round(2.0 * statistic))
        z = (statistic - expected) / math.sqrt(variance) if variance > 0.0 else None
    else:
        z = (statistic - expected) / math.sqrt(variance) if variance > 0.0 else 0.0
        p_value = float(sps.norm.sf(z))
    return SignedRankResult(
        statistic=statistic, p_value=p_value, method=used, alternative=alternative,
        n_total=int(values.size), n_nonzero=n_nonzero, n_zero=int(values.size - n_nonzero),
        expected=expected, variance=variance, z=z, zero_tol=float(zero_tol), strata=per_stratum,
    )


# --------------------------------------------------------------------------------------------
# Cluster bootstrap
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapSummary:
    """Percentile interval of a statistic resampled by cluster within strata.

    ``(nan, nan)`` and ``se = nan`` when no stratum has two clusters to resample.
    """

    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_rows: int
    n_clusters: int
    n_boot: int
    level: float
    stat: str


_BOOT_CHUNK = 1000  # resamples per block, to bound memory (part of the stream: do not change)


_STATS = {"mean": np.nanmean, "median": np.nanmedian}


def _nan_stat(function, x: np.ndarray, axis: int) -> np.ndarray:
    """``nanmean`` / ``nanmedian`` that is NaN, without a warning, on an all-missing slice."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(x, axis=axis)


def cluster_bootstrap_table(
    columns: Mapping[str, ArrayLike],
    clusters: ArrayLike,
    strata: ArrayLike,
    *,
    stats: Sequence[str] = ("mean", "median"),
    n_boot: int = 9999,
    level: float = 0.95,
    seed: int | Sequence[int],
) -> dict[str, dict[str, BootstrapSummary]]:
    """Bootstrap the ``mean`` and/or ``median`` of several columns on the **same** resamples.

    Clusters are drawn with replacement within each stratum (as many as the stratum has) and the rows
    of a drawn cluster travel together; every column and statistic is computed on each resample, so
    columns that are linked (``E_pipe = E_form + E_samp``) keep their link in the intervals. NaN in a
    column is a missing value, skipped by that column only (an all-missing column gives NaN);
    infinities are refused. ``seed`` is the entropy of a :class:`numpy.random.SeedSequence` (an int
    or a list of ints, so that every table cell derives its own stream from a base seed and its
    indices, §3). Returns ``{column: {stat: BootstrapSummary}}``.
    """
    if not columns:
        raise ValueError("no columns to bootstrap")
    names = list(columns)
    matrix = np.column_stack([np.asarray(columns[k], dtype=float).ravel() for k in names])
    c = np.asarray(clusters).ravel()
    s = np.asarray(strata).ravel()
    if matrix.shape[0] == 0 or not (matrix.shape[0] == c.size == s.size):
        raise ValueError(f"columns, clusters and strata must be non-empty and equally long, "
                         f"got {matrix.shape[0]}, {c.size}, {s.size}")
    if np.isinf(matrix).any():
        raise ValueError("values must be finite or NaN (missing), not infinite")
    unknown = [x for x in stats if x not in _STATS]
    if unknown or not stats:
        raise ValueError(f"stats must be from {sorted(_STATS)}, got {list(stats)!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    n_rows = [int(np.isfinite(matrix[:, j]).sum()) for j in range(len(names))]
    estimates = {x: _nan_stat(_STATS[x], matrix, 0) for x in stats}
    blocks = []  # per stratum: padded cluster values (n_c, width, columns), NaN beyond a short cluster
    n_clusters = 0
    for label in np.unique(s):
        in_stratum = s == label
        members = [matrix[in_stratum & (c == cl)] for cl in np.unique(c[in_stratum])]
        padded = np.full((len(members), max(m.shape[0] for m in members), len(names)), np.nan)
        for i, m in enumerate(members):
            padded[i, : m.shape[0]] = m
        blocks.append(padded)
        n_clusters += len(members)

    def summaries(samples: dict[str, np.ndarray] | None) -> dict[str, dict[str, BootstrapSummary]]:
        out: dict[str, dict[str, BootstrapSummary]] = {}
        for j, name in enumerate(names):
            out[name] = {}
            for x in stats:
                if samples is None:
                    lo = hi = se = math.nan
                else:
                    col = samples[x][:, j]
                    lo, hi = (float(q) for q in np.quantile(col, [(1.0 - level) / 2.0, (1.0 + level) / 2.0]))
                    se = float(col.std(ddof=1))
                out[name][x] = BootstrapSummary(float(estimates[x][j]), lo, hi, se, n_rows[j],
                                                n_clusters, n_boot, level, x)
        return out

    if max(b.shape[0] for b in blocks) < 2:  # nothing to resample
        return summaries(None)
    gen = np.random.default_rng(np.random.SeedSequence(seed))
    samples = {x: np.empty((n_boot, len(names))) for x in stats}
    done = 0
    while done < n_boot:
        size = min(_BOOT_CHUNK, n_boot - done)
        drawn = [b[gen.integers(0, b.shape[0], size=(size, b.shape[0]))].reshape(size, -1, len(names))
                 for b in blocks]  # (size, n_c · width, columns)
        pooled = np.concatenate(drawn, axis=1)
        for x in stats:
            samples[x][done : done + size] = _nan_stat(_STATS[x], pooled, 1)
        done += size
    return summaries(samples)


def cluster_bootstrap(
    values: ArrayLike,
    clusters: ArrayLike,
    strata: ArrayLike,
    *,
    stat: str = "mean",
    n_boot: int = 9999,
    level: float = 0.95,
    seed: int | Sequence[int],
) -> BootstrapSummary:
    """Bootstrap the ``mean`` or ``median`` of one column of finite ``values``
    (:func:`cluster_bootstrap_table` with a single column)."""
    v = np.asarray(values, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")
    if stat not in _STATS:
        raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
    table = cluster_bootstrap_table({"v": v}, clusters, strata, stats=(stat,), n_boot=n_boot,
                                    level=level, seed=seed)
    return table["v"][stat]


# --------------------------------------------------------------------------------------------
# Holm over the family
# --------------------------------------------------------------------------------------------


def holm_family(p_values: dict[str, float | None], *, alpha: float = 0.05) -> dict:
    """Holm adjustment of the RQ1 family {H1a, H1c} (§7.1).

    ``p_values`` must have exactly the keys of :data:`RQ1_FAMILY`. H1c is tested in the E2 stream
    (the cells C1–C3 of RQ2); if its p-value is not available (``None``) the family cannot be
    adjusted: ``adjusted`` and ``reject`` are ``None`` and ``upper_bound`` gives the Bonferroni
    value ``min(1, 2p)`` of the available member, which is never below its Holm-adjusted p-value.

    Returns ``{"complete", "adjusted", "reject", "upper_bound", "alpha", "note"}``.
    """
    if set(p_values) != set(RQ1_FAMILY):
        missing = sorted(set(RQ1_FAMILY) - set(p_values))
        extra = sorted(set(p_values) - set(RQ1_FAMILY))
        raise ValueError(f"the RQ1 family is {RQ1_FAMILY}: missing {missing}, unexpected {extra}")
    for name, p in p_values.items():
        if p is not None and not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value of {name} must lie in [0, 1], got {p}")
    m = len(RQ1_FAMILY)
    upper = {name: None if p_values[name] is None else min(1.0, m * p_values[name])
             for name in RQ1_FAMILY}
    if any(p_values[name] is None for name in RQ1_FAMILY):
        absent = [name for name in RQ1_FAMILY if p_values[name] is None]
        return {
            "complete": False, "adjusted": dict.fromkeys(RQ1_FAMILY), "reject": dict.fromkeys(RQ1_FAMILY),
            "upper_bound": upper, "alpha": alpha,
            "note": f"no p-value for {', '.join(absent)}: the Holm family is incomplete, "
                    f"only the Bonferroni upper bound is given",
        }
    adjusted = holm([p_values[name] for name in RQ1_FAMILY])
    out = {name: float(a) for name, a in zip(RQ1_FAMILY, adjusted, strict=True)}
    return {
        "complete": True, "adjusted": out, "reject": {name: out[name] <= alpha for name in RQ1_FAMILY},
        "upper_bound": upper, "alpha": alpha, "note": "",
    }
