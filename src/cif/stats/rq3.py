"""RQ3 inference: replica-averaged contrasts, joint block bootstrap, H3a / H3b / H3c.

ANALYSIS_PLAN §2 RQ3 and §7.4. The unit of inference is TIME: replicas add no observations, they
only average out the injection noise, so every statistic is a function of the daily return series
of the optimum and of all replicas, resampled JOINTLY by one set of circular blocks.

Statistic. For a series ``x`` let ``φ(x)`` be its Sharpe ratio (``functional="sharpe"``, annualised
by ``√periods_per_year``) or the log of its variance (``"logvar"``, the GMV arm). A *contrast* is a
linear combination ``θ = Σ_k c_k φ(x_k)``; all RQ3 estimands are contrasts:

* the replica-averaged difference at a level: ``(1/R)·Σ_r φ(x_r) − φ(x*)`` (the difference per
  replica, then the mean);
* the H3b slope and intercept of that difference on ``log g``: OLS weights over the levels (the
  optimum cancels out of the slope).

Inference follows Ledoit & Wolf (2008), Sec. 3.2.2, for a general smooth ``θ``: the standard error
of the sample is the HAC (QS kernel, Andrews bandwidth, AR(1) prewhitening) of the influence
series ``z_t = Σ_k c_k ψ_k(t)``, with ``ψ`` the influence function of ``φ`` (``c/σ − μ(c² − σ²)/
(2σ³)`` for the Sharpe ratio, ``(c² − σ²)/σ²`` for the log variance); each bootstrap sample gets
the block-based ``s*`` of eq. (8); ``d* = |θ* − θ̂|/s*``; the two-sided p-value is eq. (9) and the
symmetric interval of coverage ``1 − α`` is eq. (7). For one replica and the Sharpe ratio this is
the LW test (same resamples, same ``d*``; the HAC of the projected scalar series agreed with
LW's HAC of the four moment conditions to about 1 % in simulations with AR(1) returns, T from 500
to 12 000; the tests allow 5 %). It is dimensionless, so the bootstrap needs no rescaling to
percent. The log-variance functional (the GMV arm) is the same LW (2008) scheme for the difference
of log variances: a studentised circular block bootstrap whose standard error of the sample is the
HAC of the delta-method influence series ``(c² − σ²)/σ²`` of each log variance (for two independent
normal series the standard error is ``√(4/T)``, checked in ``tests/test_stats_rq3_ranks.py``).

Plan v2.4 made the k best supports the primary operator: a *rank path* is a level with one
replica and is analysed in :mod:`cif.stats.rq3_ranks` (H3a on rank 2, the H3b slope on
``log ḡ_r``, H3c residuals) on top of the machinery below. The replicas and ``"calibrate"`` below
stay for the descriptive operator (б).

The block length ``b`` is fixed for the whole panel; ``"calibrate"`` runs LW Algorithm 3.1 once,
on the pair (optimum, mean of all replicas), in percent: the algorithm is defined for a pair of
series (the plan says "on the optimum's series") and LW's HAC inside it is not unit-free. It
calibrates the test of the functional of the arm: the difference of Sharpe ratios for
``"sharpe"`` (:func:`cif.stats.sharpe.calibrate_block_size`, the algorithm of LW) and the
difference of log variances for ``"logvar"``, the GMV arm (:func:`calibrate_logvar_block`: the
same pseudo-series and the same steps, with the arm's own studentised bootstrap test).

Inference is conditional on the replicas drawn: only time is resampled, the choice of the
replicas' subsets is not a source of variance in the interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from cif.stats.bootstrap import (
    _CHUNK,
    SeedLike,
    _as_generator,
    _positive_int,
    _seed_record,
    circular_block_bootstrap_indices,
)
from cif.stats.sharpe import (
    BlockCalibration,
    Kernel,
    _check_block,
    _child_generator,
    _pair,
    _studentize,
    _var1_pseudo_chunks,
    calibrate_block_size,
    hac_covariance,
)
from cif.stats.tests_misc import holm, tost_from_ci

__all__ = [
    "H3A_OUTCOMES",
    "ContrastEstimate",
    "DoseResponse",
    "EquivalenceResult",
    "JointBootstrap",
    "LevelEquivalence",
    "SolverDose",
    "VarianceMargin",
    "calibrate_joint_block",
    "calibrate_logvar_block",
    "dose_response",
    "equivalence_test",
    "h3a_outcome",
    "joint_contrasts",
    "sharpe_margin",
    "solver_dose",
    "variance_margin",
]

Functional = Literal["sharpe", "logvar"]
_FUNCTIONALS = ("sharpe", "logvar")

#: Outcomes of H3a, the plan's table (§2): (IUT-TOST passes, any two-sided test significant).
H3A_OUTCOMES = ("equivalent", "distinguishable_within_sesoi", "different", "inconclusive")

_HAC_DF_ADJUST = 4  # LW (2008): Psi_hat carries T / (T - 4)


# ---------------------------------------------------------------------------------------------
# influence functions
# ---------------------------------------------------------------------------------------------


def _value_and_influence(x: np.ndarray, functional: str, ddof: int) -> tuple[float, np.ndarray]:
    """``φ(x)`` and the influence series ``ψ_t`` of one series (population moments in ``ψ``, the
    sample standard deviation with ``ddof`` in the Sharpe ratio, as in the LW code)."""
    n = x.size
    mu = float(x.mean())
    c = x - mu
    q = c * c
    v = float(q.mean())
    if not v > 0.0:
        raise ValueError("a series has zero variance; the Sharpe ratio and the log variance are undefined")
    if functional == "sharpe":
        sd = math.sqrt(v)
        return mu / math.sqrt(v * n / (n - ddof)), c / sd - mu * (q - v) / (2.0 * sd**3)
    return math.log(v * n / (n - ddof)), (q - v) / v


def _resampled_block_stats(
    xs: np.ndarray, block: int, functional: str, ddof: int
) -> tuple[np.ndarray, np.ndarray]:
    """``φ`` and the LW block sums ``ζ_j = b^{-1/2}·Σ_{block j} ψ_t`` of every resample.

    ``xs`` has shape ``(m, T)``; the influence series is taken at the moments of each resample and
    summed over the ``⌊T/b⌋`` complete blocks (LW 2008, Sec. 3.2.2). Returns ``(m,)`` and
    ``(m, ⌊T/b⌋)``. Block sums of ``c`` and ``c²`` are formed first, so ``ψ`` is never stored.
    """
    m, n = xs.shape
    n_blocks = n // block
    lb = n_blocks * block
    mu = xs.mean(axis=1)
    c = xs - mu[:, None]
    q = c * c
    v = q.mean(axis=1)
    if np.any(v <= 0.0):
        raise ValueError("a bootstrap resample has zero variance")
    sum_c = c[:, :lb].reshape(m, n_blocks, block).sum(axis=2)
    sum_q = q[:, :lb].reshape(m, n_blocks, block).sum(axis=2) - block * v[:, None]
    root_b = math.sqrt(block)
    if functional == "sharpe":
        sd = np.sqrt(v)
        value = mu / np.sqrt(v * n / (n - ddof))
        zeta = (sum_c / sd[:, None] - (mu / (2.0 * sd**3))[:, None] * sum_q) / root_b
    else:
        value = np.log(v * n / (n - ddof))
        zeta = sum_q / (v[:, None] * root_b)
    return value, zeta


def _long_run_se(z: np.ndarray, kernel: Kernel, prewhiten: bool) -> tuple[float, float]:
    """``s(θ̂)`` and the bandwidth: HAC of the scalar influence series ``z`` (mean zero)."""
    if not np.any(z):
        return 0.0, 0.0
    psi, info = hac_covariance(
        z[:, None], kernel=kernel, prewhiten=prewhiten, df_adjust=_HAC_DF_ADJUST
    )
    return math.sqrt(max(float(psi[0, 0]), 0.0) / z.size), float(info["bandwidth"])


# ---------------------------------------------------------------------------------------------
# the joint bootstrap
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JointBootstrap:
    """Joint studentised bootstrap of ``Q`` contrasts on common resamples.

    ``estimate`` and ``se`` (HAC) are ``(Q,)``, ``theta_star`` and ``dstar`` ``(n_boot, Q)``;
    ``estimate``, ``se`` and ``theta_star`` are annualised for the Sharpe functional
    (``scale = √periods_per_year``), ``dstar`` is unit-free. ``bandwidth`` is the HAC bandwidth.
    """

    estimate: np.ndarray
    se: np.ndarray
    theta_star: np.ndarray
    dstar: np.ndarray
    bandwidth: np.ndarray
    scale: float
    functional: str
    block: int
    n_boot: int
    n_obs: int
    seed: int | None

    def pvalue(self, null: float | ArrayLike = 0.0) -> np.ndarray:
        """Two-sided p-values of ``H0: θ = null``, LW eq. (9): ``(#{d* ≥ d} + 1)/(M + 1)``."""
        nulls = np.broadcast_to(np.asarray(null, dtype=float), self.estimate.shape)
        out = np.empty(self.estimate.size)
        for q in range(out.size):
            d = abs(_studentize(float(self.estimate[q] - nulls[q]), float(self.se[q])))
            out[q] = (int(np.count_nonzero(self.dstar[:, q] >= d)) + 1) / (self.n_boot + 1)
        return out

    def zstar(self, alpha: float) -> np.ndarray:
        """The ``⌈(1 − α)(M + 1)⌉``-th smallest ``d*`` per contrast (``inf`` if it does not exist)."""
        _check_alpha(alpha)
        k = math.ceil((1.0 - alpha) * (self.n_boot + 1) - 1e-9)
        if k > self.n_boot:
            return np.full(self.estimate.size, math.inf)
        return np.sort(self.dstar, axis=0)[k - 1]

    def ci(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric studentised interval of coverage ``1 − α``, LW eq. (7); it excludes ``0`` iff
        the two-sided p-value of ``H0: θ = 0`` is at most ``α``."""
        z = self.zstar(alpha)
        half = np.zeros_like(self.estimate)
        positive = self.se > 0.0
        half[positive] = z[positive] * self.se[positive]
        return self.estimate - half, self.estimate + half

    def percentile_ci(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Percentile interval of the bootstrap ``θ*`` (robustness, not used for decisions)."""
        _check_alpha(alpha)
        lo, hi = np.quantile(self.theta_star, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
        return lo, hi


def _check_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(alpha)


def joint_contrasts(
    series: ArrayLike,
    weights: ArrayLike,
    *,
    functional: Functional = "sharpe",
    block: int,
    n_boot: int = 4999,
    seed: SeedLike,
    periods_per_year: float | None = 252,
    ddof: int = 1,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
) -> JointBootstrap:
    """Joint circular block bootstrap of the contrasts ``θ_q = Σ_k W[q, k]·φ(series[:, k])``.

    ``series`` is ``(T, M)`` (all series on the same days), ``weights`` ``(Q, M)``. One set of
    ``⌈T/b⌉`` circular blocks of length ``block`` is drawn per resample and applied to ALL series,
    so the dependence between the optimum, the replicas and the levels is preserved and all
    contrasts are evaluated on the same resamples. ``seed`` is mandatory. Resamples are generated
    in chunks of the same size as in :func:`cif.stats.sharpe.ledoit_wolf_sharpe_test`, so for one
    replica the ``d*`` coincide with the LW test for the same seed.
    """
    if functional not in _FUNCTIONALS:
        raise ValueError(f"functional must be one of {_FUNCTIONALS}, got {functional!r}")
    x = np.asarray(series, dtype=float)
    if x.ndim != 2:
        raise ValueError("series must be a 2-D (T, M) array")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contain non-finite values; every input must be finite")
    n, n_series = x.shape
    if n < 10:
        raise ValueError("need at least 10 observations")
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[1] != n_series:
        raise ValueError(f"weights must be a (Q, {n_series}) matrix, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values")
    block = _check_block(block, n)
    n_boot = _positive_int(n_boot, "n_boot")
    gen = _as_generator(seed)
    n_contrasts = w.shape[0]
    active = np.flatnonzero((w != 0.0).any(axis=0))

    theta = np.zeros(n_contrasts)
    z = np.zeros((n_contrasts, n))
    for k in active:
        value, psi = _value_and_influence(x[:, k], functional, ddof)
        theta += w[:, k] * value
        z += np.outer(w[:, k], psi)
    se = np.empty(n_contrasts)
    bandwidth = np.empty(n_contrasts)
    for q in range(n_contrasts):
        se[q], bandwidth[q] = _long_run_se(z[q], kernel, prewhiten)

    n_blocks = n // block
    theta_star = np.empty((n_boot, n_contrasts))
    dstar = np.empty((n_boot, n_contrasts))
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = circular_block_bootstrap_indices(n, block, m, gen)
        theta_b = np.zeros((m, n_contrasts))
        zeta = np.zeros((n_contrasts, m, n_blocks))
        for k in active:
            value, zeta_k = _resampled_block_stats(x[:, k][idx], block, functional, ddof)
            theta_b += value[:, None] * w[:, k][None, :]
            zeta += w[:, k][:, None, None] * zeta_k[None, :, :]
        if not np.all(np.isfinite(theta_b)):
            raise ValueError("a bootstrap statistic is not finite")
        s_star = np.sqrt((zeta * zeta).mean(axis=2) / n).T  # LW eq. (8), shape (m, Q)
        num = np.abs(theta_b - theta[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(num == 0.0, 0.0, num / s_star)
        theta_star[start : start + m] = theta_b
        dstar[start : start + m] = d

    scale = math.sqrt(periods_per_year) if functional == "sharpe" and periods_per_year else 1.0
    return JointBootstrap(
        estimate=theta * scale,
        se=se * scale,
        theta_star=theta_star * scale,
        dstar=dstar,
        bandwidth=bandwidth,
        scale=scale,
        functional=functional,
        block=block,
        n_boot=n_boot,
        n_obs=n,
        seed=_seed_record(seed),
    )


# ---------------------------------------------------------------------------------------------
# panels of replicas
# ---------------------------------------------------------------------------------------------


def _panel(
    opt: ArrayLike, replicas: Sequence[ArrayLike], levels: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Series matrix ``[opt | replicas of level 1 | …]`` and the per-level contrast rows
    (``−1`` on the optimum, ``1/R_ℓ`` on each replica of the level)."""
    x0 = np.asarray(opt, dtype=float)
    if x0.ndim != 1:
        raise ValueError("opt must be a 1-D return series")
    if len(replicas) != len(levels):
        raise ValueError("levels and replicas must have one entry each (one replica block per level)")
    if not len(levels):
        raise ValueError("need at least one level")
    blocks = []
    for r in replicas:
        a = np.asarray(r, dtype=float)
        a = a.reshape(-1, 1) if a.ndim == 1 else a
        if a.ndim != 2 or a.shape[0] != x0.size:
            raise ValueError("opt and every replica block must have the same length in time")
        blocks.append(a)
    series = np.column_stack([x0, *blocks])
    rows = np.zeros((len(blocks), series.shape[1]))
    start = 1
    for q, a in enumerate(blocks):
        rows[q, 0] = -1.0
        rows[q, start : start + a.shape[1]] = 1.0 / a.shape[1]
        start += a.shape[1]
    return series, rows, blocks


def calibrate_logvar_block(
    r1: ArrayLike,
    r2: ArrayLike,
    *,
    block_grid: Sequence[int] = (1, 2, 4, 6, 8, 10),
    alpha: float = 0.05,
    n_boot: int = 499,
    n_pseudo: int = 1000,
    mean_block_resid: float = 5.0,
    burn_in: int = 50,
    seed: SeedLike,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
    ddof: int = 1,
) -> BlockCalibration:
    """Block size of the studentised bootstrap test of the difference of log variances (the GMV
    arm) by calibration: LW (2008) Algorithm 3.1 with the arm's own statistic.

    Same steps and defaults as :func:`cif.stats.sharpe.calibrate_block_size`. The pseudo-series
    are the same (:func:`cif.stats.sharpe._var1_pseudo_chunks`: a VAR(1) with intercept fitted by
    OLS to the pair, residuals resampled by the stationary bootstrap, ``burn_in`` discarded
    steps). For each candidate block and each pseudo-series the studentised circular block
    bootstrap of ``Δ = log σ²(r1) − log σ²(r2)`` (:func:`joint_contrasts`, functional
    ``"logvar"``: HAC of the delta-method influence series, ``n_boot`` resamples) is tested against
    ``H0: Δ = Δ̂`` of the ORIGINAL pair (LW eq. 9); ``rejection_rates`` is the share of rejections
    at ``alpha`` and ``block`` the candidate whose rate is closest to ``alpha`` (the first on
    ties). The log variance is unit-free, so returns in fractions or in percent give the same
    calibration. The result carries ``functional="logvar"``.
    """
    alpha = _check_alpha(alpha)
    r = _pair(r1, r2, None)
    T = r.shape[0]
    grid = tuple(_check_block(b, T) for b in block_grid)
    if not grid:
        raise ValueError("block_grid is empty")
    n_boot = _positive_int(n_boot, "n_boot")
    n_pseudo = _positive_int(n_pseudo, "n_pseudo")
    burn_in = int(burn_in)
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    gen = _as_generator(seed)
    delta_hat = (
        _value_and_influence(r[:, 0], "logvar", ddof)[0]
        - _value_and_influence(r[:, 1], "logvar", ddof)[0]
    )
    weights = np.array([[1.0, -1.0]])
    rejections = np.zeros(len(grid), dtype=int)
    for pseudo in _var1_pseudo_chunks(r, n_pseudo, mean_block_resid, burn_in, gen):
        for ps in pseudo:
            for j, b in enumerate(grid):
                joint = joint_contrasts(
                    ps, weights, functional="logvar", block=b, n_boot=n_boot, seed=gen,
                    ddof=ddof, kernel=kernel, prewhiten=prewhiten,
                )
                rejections[j] += joint.pvalue(delta_hat)[0] <= alpha
    rates = rejections / n_pseudo
    return BlockCalibration(
        block=grid[int(np.argmin(np.abs(rates - alpha)))],
        block_grid=grid,
        rejection_rates=rates,
        alpha=alpha,
        n_boot=n_boot,
        n_pseudo=n_pseudo,
        mean_block_resid=float(mean_block_resid),
        burn_in=burn_in,
        seed=_seed_record(seed),
        functional="logvar",
    )


def calibrate_joint_block(
    opt: ArrayLike,
    replicas: Sequence[ArrayLike],
    *,
    seed: SeedLike,
    percent: bool = True,
    functional: Functional = "sharpe",
    **calibration_kwargs: Any,
) -> BlockCalibration:
    """LW (2008) Algorithm 3.1 on the pair (optimum, equal-weight mean of all replica series),
    for the test of the functional of the arm.

    ``functional="sharpe"`` (default) calibrates the test of the difference of Sharpe ratios with
    :func:`cif.stats.sharpe.calibrate_block_size`; ``"logvar"`` (the GMV arm) the test of the
    difference of log variances with :func:`calibrate_logvar_block`. The Sharpe calibration uses
    LW's HAC with unit weights over the moment conditions, which is not unit-free, and the plan
    (§7.4) works in percent: with ``percent=True`` (default) the daily returns, given as fractions,
    are multiplied by 100 first, for both functionals. (The bootstrap itself is dimensionless and
    needs no rescaling.) ``calibration_kwargs`` go to the calibration (``block_grid``,
    ``n_pseudo``, ``n_boot``, …); the default grid ``{1, 2, 4, 6, 8, 10}`` is meant for 120
    monthly returns and should be widened for daily data (the plan's grid is
    :data:`cif.stats.rq3_ranks.BLOCK_GRID_DAILY`).
    """
    if functional not in _FUNCTIONALS:
        raise ValueError(f"functional must be one of {_FUNCTIONALS}, got {functional!r}")
    unit = 100.0 if percent else 1.0
    x0 = unit * np.asarray(opt, dtype=float)
    cols = []
    for r in replicas:
        a = np.asarray(r, dtype=float)
        cols.append(a.reshape(-1, 1) if a.ndim == 1 else a)
    partner = unit * np.column_stack(cols).mean(axis=1)
    calibrate = calibrate_logvar_block if functional == "logvar" else calibrate_block_size
    return calibrate(x0, partner, seed=seed, **calibration_kwargs)


def _resolve_block(
    block: int | Literal["calibrate"],
    opt: ArrayLike,
    replicas: Sequence[ArrayLike],
    seed: SeedLike,
    calibration_kwargs: dict[str, Any] | None,
    functional: Functional = "sharpe",
) -> tuple[int, BlockCalibration | None]:
    if isinstance(block, str):
        if block != "calibrate":
            raise ValueError("block must be an integer or 'calibrate'")
        cal = calibrate_joint_block(
            opt, replicas, seed=_child_generator(seed), functional=functional,
            **dict(calibration_kwargs or {}),
        )
        return cal.block, cal
    return block, None


@dataclass(frozen=True)
class ContrastEstimate:
    """One contrast: estimate, HAC standard error, studentised interval (coverage ``1 − α``),
    percentile interval, two-sided p-value of ``H0: θ = 0``."""

    estimate: float
    se: float
    ci: tuple[float, float]
    ci_percentile: tuple[float, float]
    pvalue: float


def _contrast_estimate(joint: JointBootstrap, q: int, alpha: float) -> ContrastEstimate:
    lo, hi = joint.ci(alpha)
    plo, phi = joint.percentile_ci(alpha)
    return ContrastEstimate(
        estimate=float(joint.estimate[q]),
        se=float(joint.se[q]),
        ci=(float(lo[q]), float(hi[q])),
        ci_percentile=(float(plo[q]), float(phi[q])),
        pvalue=float(joint.pvalue()[q]),
    )


# ---------------------------------------------------------------------------------------------
# margins and the H3a table
# ---------------------------------------------------------------------------------------------


def sharpe_margin(sigma_star: float, sesoi: float = 0.0025) -> float:
    """``Δ_SR = SESOI / σ*``: SESOI (25 bp of CE per year) over the annual volatility of the
    optimum's excess return before costs on the pilot period (plan §2, H3a, MV)."""
    if not (math.isfinite(sigma_star) and sigma_star > 0.0):
        raise ValueError(f"sigma_star must be a positive number, got {sigma_star!r}")
    return float(sesoi) / float(sigma_star)


class VarianceMargin(NamedTuple):
    """``δ`` and the margin ``[log(1 − δ), log(1 + δ)]`` of the mean log variance ratio."""

    delta: float
    low: float
    high: float


def variance_margin(lambda_f: float, sigma2_star: float, sesoi: float = 0.0025) -> VarianceMargin:
    """``δ = (2·SESOI/λ_f)/σ²_*`` (the variance increment ``Δσ²`` with ``(λ_f/2)·Δσ² = SESOI``,
    relative to the optimum's annual variance on the pilot period) and its log margin."""
    if not (lambda_f > 0.0 and sigma2_star > 0.0):
        raise ValueError("lambda_f and sigma2_star must be positive")
    delta = (2.0 * sesoi / lambda_f) / sigma2_star
    if not 0.0 < delta < 1.0:
        raise ValueError(
            f"δ = {delta:.4g} must lie in (0, 1): the SESOI is not smaller than the variance, "
            f"log(1 − δ) does not exist"
        )
    return VarianceMargin(delta, math.log1p(-delta), math.log1p(delta))


def h3a_outcome(*, iut_passes: bool, any_significant: bool) -> str:
    """The four outcomes of H3a (plan §2): IUT-TOST passes / fails × at least one two-sided test
    (Holm over levels) significant / none."""
    if iut_passes:
        return "distinguishable_within_sesoi" if any_significant else "equivalent"
    return "different" if any_significant else "inconclusive"


@dataclass(frozen=True)
class LevelEquivalence:
    """H3a at one level: ``ci_tost`` is the ``1 − 2α`` interval of the TOST (90 % for
    ``α = 0.05``), ``ci`` the ``1 − α`` interval for reporting."""

    gap_bp: float
    estimate: float
    se: float
    ci_tost: tuple[float, float]
    ci: tuple[float, float]
    pvalue: float
    pvalue_holm: float
    equivalent: bool
    significant: bool


@dataclass(frozen=True)
class EquivalenceResult:
    """H3a on one arm and panel: per-level tests, the IUT, and the plan's outcome."""

    outcome: str
    iut_passes: bool
    any_significant: bool
    levels: tuple[LevelEquivalence, ...]
    margin: tuple[float, float]
    alpha: float
    functional: str
    block: int
    n_boot: int
    n_obs: int
    seed: int | None
    calibration: BlockCalibration | None


def equivalence_test(
    opt: ArrayLike,
    replicas: Sequence[ArrayLike],
    levels: Sequence[float],
    margin: float | tuple[float, float],
    *,
    functional: Functional = "sharpe",
    block: int | Literal["calibrate"],
    n_boot: int = 4999,
    alpha: float = 0.05,
    seed: SeedLike,
    periods_per_year: float | None = 252,
    ddof: int = 1,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
    calibration_kwargs: dict[str, Any] | None = None,
) -> EquivalenceResult:
    """H3a (plan §2): TOST per level by the ``1 − 2α`` interval, intersection–union over levels,
    and two-sided tests per level with Holm, mapped to the four outcomes.

    ``opt`` is the daily return series of the optimum (excess returns for the Sharpe functional),
    ``replicas[i]`` the ``(T, R)`` daily returns of the replicas at ``levels[i]``, all on the same
    days (the days of ``F_a``; see :func:`cif.backtest.fold_sets.days_in_folds`). ``margin`` is a
    positive number (``Δ_SR``, region ``(−Δ, Δ)``) or ``(low, high)`` (the GMV arm:
    ``(log(1 − δ), log(1 + δ))``, :func:`variance_margin`). The estimate per level is the mean over
    replicas of the difference to the optimum (Sharpe: annual, before costs; log variance: mean log
    ratio), tested jointly on common resamples.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5) for the TOST")
    series, rows, _ = _panel(opt, replicas, levels)
    block_used, calibration = _resolve_block(
        block, opt, replicas, seed, calibration_kwargs, functional
    )
    joint = joint_contrasts(
        series, rows, functional=functional, block=block_used, n_boot=n_boot, seed=seed,
        periods_per_year=periods_per_year, ddof=ddof, kernel=kernel, prewhiten=prewhiten,
    )
    p = joint.pvalue()
    p_holm = holm(p)
    lo_tost, hi_tost = joint.ci(2.0 * alpha)
    lo, hi = joint.ci(alpha)
    tests = []
    for q, g in enumerate(levels):
        tost = tost_from_ci(float(lo_tost[q]), float(hi_tost[q]), margin)
        tests.append(
            LevelEquivalence(
                gap_bp=float(g),
                estimate=float(joint.estimate[q]),
                se=float(joint.se[q]),
                ci_tost=(float(lo_tost[q]), float(hi_tost[q])),
                ci=(float(lo[q]), float(hi[q])),
                pvalue=float(p[q]),
                pvalue_holm=float(p_holm[q]),
                equivalent=tost.equivalent,
                significant=bool(p_holm[q] <= alpha),
            )
        )
    iut = all(t.equivalent for t in tests)
    any_sig = any(t.significant for t in tests)
    return EquivalenceResult(
        outcome=h3a_outcome(iut_passes=iut, any_significant=any_sig),
        iut_passes=iut,
        any_significant=any_sig,
        levels=tuple(tests),
        margin=(tost.margin_low, tost.margin_high),
        alpha=float(alpha),
        functional=functional,
        block=joint.block,
        n_boot=joint.n_boot,
        n_obs=joint.n_obs,
        seed=joint.seed,
        calibration=calibration,
    )


# ---------------------------------------------------------------------------------------------
# H3b: dose-response
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseResponse:
    """H3b: the line ``Δ(g) = a + b·x(g)`` fitted by OLS to the level differences, ``x = log g``
    (``"log"``, primary) or ``g`` (``"linear"``, sensitivity). ``per_level`` are the differences
    themselves (descriptive); ``slope`` and ``intercept`` carry joint-bootstrap inference."""

    levels: tuple[float, ...]
    x_transform: str
    per_level: tuple[ContrastEstimate, ...]
    slope: ContrastEstimate
    intercept: ContrastEstimate
    significant: bool
    alpha: float
    functional: str
    block: int
    n_boot: int
    n_obs: int
    seed: int | None
    calibration: BlockCalibration | None

    def _x(self, g: float) -> float:
        return math.log(g) if self.x_transform == "log" else float(g)

    def predict(self, dose_bp: float) -> float:
        """Fitted ``a + b·x(dose)``, only inside ``[min L, max L]``: outside it the curve is not
        extrapolated and the solver's calibration (H3c) is descriptive."""
        if not min(self.levels) <= dose_bp <= max(self.levels):
            raise ValueError(
                f"dose {dose_bp:g} bp is outside [{min(self.levels):g}, {max(self.levels):g}]: "
                f"no prediction, H3c is descriptive for it"
            )
        return self.intercept.estimate + self.slope.estimate * self._x(dose_bp)


def _ols_weights(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weights of the OLS slope and intercept on the abscissas ``x``: ``b = w_slope·y`` and
    ``a = w_intercept·y`` for any response ``y``; the fitted value at ``x0`` has the weights
    ``w_intercept + x0·w_slope`` (they sum to one)."""
    centred = x - x.mean()
    sxx = float(centred @ centred)
    if sxx == 0.0:
        raise ValueError("the levels (regressor values) must not all be equal")
    w_slope = centred / sxx
    return w_slope, 1.0 / x.size - x.mean() * w_slope


def dose_response(
    opt: ArrayLike,
    replicas: Sequence[ArrayLike],
    levels: Sequence[float],
    *,
    functional: Functional = "sharpe",
    x_transform: Literal["log", "linear"] = "log",
    block: int | Literal["calibrate"],
    n_boot: int = 4999,
    alpha: float = 0.05,
    seed: SeedLike,
    periods_per_year: float | None = 252,
    ddof: int = 1,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
    calibration_kwargs: dict[str, Any] | None = None,
) -> DoseResponse:
    """H3b (plan §2): slope of the replica-averaged difference on ``log g`` over the levels ``L``
    of the balanced panel ``F_b``, with a joint-bootstrap interval and a two-sided test.

    Needs at least three levels; with fewer, H3b is descriptive and the slope is not estimated.
    The slope is a contrast with OLS weights over the level differences, evaluated on the same
    resamples as the differences (the optimum drops out of it).
    """
    if x_transform not in ("log", "linear"):
        raise ValueError(f"x_transform must be 'log' or 'linear', got {x_transform!r}")
    if len(levels) < 3:
        raise ValueError(
            f"the slope needs at least 3 levels, got {len(levels)}: H3b is descriptive"
        )
    g = np.asarray(levels, dtype=float)
    if x_transform == "log" and np.any(g <= 0.0):
        raise ValueError("levels must be positive for the log transform")
    series, rows, _ = _panel(opt, replicas, levels)
    x = np.log(g) if x_transform == "log" else g
    w_slope, w_intercept = _ols_weights(x)
    contrasts = np.vstack([rows, w_slope @ rows, w_intercept @ rows])
    block_used, calibration = _resolve_block(
        block, opt, replicas, seed, calibration_kwargs, functional
    )
    joint = joint_contrasts(
        series, contrasts, functional=functional, block=block_used, n_boot=n_boot, seed=seed,
        periods_per_year=periods_per_year, ddof=ddof, kernel=kernel, prewhiten=prewhiten,
    )
    n_levels = x.size
    slope = _contrast_estimate(joint, n_levels, alpha)
    return DoseResponse(
        levels=tuple(float(v) for v in g),
        x_transform=x_transform,
        per_level=tuple(_contrast_estimate(joint, q, alpha) for q in range(n_levels)),
        slope=slope,
        intercept=_contrast_estimate(joint, n_levels + 1, alpha),
        significant=bool(slope.pvalue <= alpha),
        alpha=float(alpha),
        functional=functional,
        block=joint.block,
        n_boot=joint.n_boot,
        n_obs=joint.n_obs,
        seed=joint.seed,
        calibration=calibration,
    )


# ---------------------------------------------------------------------------------------------
# H3c: the dose of a real solver
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverDose:
    """Dose of a real solver for H3c: its mean realised in-sample gap over ``F_a``. A prediction
    from the H3b curve is made only if ``inside_range``; otherwise ``descriptive`` (``reason``)."""

    dose_bp: float | None
    n_folds: int
    inside_range: bool
    descriptive: bool
    reason: str | None


def solver_dose(
    realised_gap_bp: ArrayLike, fold_mask_a: ArrayLike, levels: Sequence[float]
) -> SolverDose:
    """Mean realised ``ĝ_CE`` of the solver over the folds of ``F_a``; the prediction is allowed
    only for a dose in ``[min L, max L]`` (both ends included)."""
    gaps = np.asarray(realised_gap_bp, dtype=float)
    mask = np.asarray(fold_mask_a, dtype=bool)
    if gaps.ndim != 1 or gaps.shape != mask.shape:
        raise ValueError("realised_gap_bp and fold_mask_a must be 1-D arrays of the same length")
    chosen = gaps[mask]
    if not np.all(np.isfinite(chosen)):
        raise ValueError("the realised gaps on F_a must be finite")
    if chosen.size == 0:
        return SolverDose(None, 0, False, True, "no folds in F_a")
    dose = float(chosen.mean())
    if not len(levels):
        return SolverDose(dose, int(chosen.size), False, True, "L is empty")
    lo, hi = float(min(levels)), float(max(levels))
    if dose < lo:
        return SolverDose(dose, int(chosen.size), False, True,
                          f"dose {dose:.4g} bp is below min L = {lo:g} bp")
    if dose > hi:
        return SolverDose(dose, int(chosen.size), False, True,
                          f"dose {dose:.4g} bp is above max L = {hi:g} bp")
    return SolverDose(dose, int(chosen.size), True, False, None)
