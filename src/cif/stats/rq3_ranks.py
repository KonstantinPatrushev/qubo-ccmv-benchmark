"""RQ3 on rank paths: H3a on rank 2, the H3b slope on ``log ḡ_r``, the block length (plan v2.4).

ANALYSIS_PLAN §2 RQ3 (operator (г), "k лучших носителей"), §7.1, §7.4. For every fold ``f`` of
``F_k`` there is a certified optimum and the portfolios of the ranks ``2..33``; the *rank-r path*
is the walk-forward that holds the rank-``r`` portfolio on every fold: one deterministic daily
return series, no replicas. A rank path is therefore "a level with one replica" of
:mod:`cif.stats.rq3` and everything here is a contrast of the Sharpe ratios (or of the log
variances, the GMV arm) of the optimum and of the rank paths, resampled JOINTLY by one set of
circular blocks (:func:`cif.stats.rq3.joint_contrasts`, the LW (2008) studentised scheme; see the
module docstring there for the standard error). The panel is balanced: the days of the folds
outside ``F_k`` are dropped from every path (:meth:`cif.backtest.fold_sets.RankFoldSet.day_mask`).
"Before costs" is a pass of the engine with ``tc = 0``; the functions take the series they are
given. The descriptive replica operator (б) stays in :mod:`cif.stats.rq3`.

* **H3a** (:func:`rank_equivalence`): rank 2 against the optimum. One TOST on the ``1 − 2α``
  interval (90 % for ``α = 0.05``) with the margin ``Δ_SR`` (MV, annual Sharpe units) or
  ``[log(1 − δ), log(1 + δ)]`` (GMV, log variance ratio), and one two-sided test at ``α``, per arm;
  the four outcomes of the plan (:func:`rank2_outcome`). The primary arm is FF49-MV; the secondary
  FF49-GMV and US-MV are adjusted by Holm over the two (:func:`holm_over_arms`) with the TOST
  p-value ``pvalue_tost`` (the smallest ``α`` at which the interval decision passes).
* **H3b** (:func:`rank_dose_response`): the OLS slope of ``ΔSR(r)`` (path ``r`` minus optimum) on
  ``log ḡ_r`` over the ranks ``{2, 3, 5, 9, 17, 33}``, ``ḡ_r`` the mean dose over ``F_k``
  (:func:`cif.backtest.fold_sets.rank_doses`). The abscissas are in-sample and common to all
  bootstrap draws, so the uncertainty of the slope is the uncertainty in TIME only. The test is
  two-sided and decided by the percentile 95 % interval of the joint bootstrap (B = 4999): the
  slope is significant iff the interval excludes zero. Sensitivity: the regressor ``log r``, and
  all ranks 2–33. GMV (secondary): the same slope of the log variance ratio.
* **H3c** (:func:`solver_residuals`, FF49-MV): for tabu, SA and LS the residual is the observed
  pre-cost Sharpe difference of the solver's path to the optimum minus the prediction
  ``a + b·log(dose)`` of the H3b fit; ``dose`` is the solver's mean realised gap over ``F_k``
  (:func:`rank_solver_dose`). The prediction is made only for a dose in ``[ḡ_2, ḡ_33]``, otherwise
  that solver is descriptive; with fewer than two solvers of a median gap of at least 1 bp, or
  fewer than two with a prediction, H3c is descriptive as a whole. Each residual gets a TOST with
  the margin ``Δ_SR`` and the IUT is the conjunction over the solvers with a prediction. The rank
  of a solver's portfolio per fold is summarised in the bins 1; 2–5; 6–33; > 33
  (:func:`rank_bins`).
* **Block length** (:func:`calibrate_rank_block`): LW Algorithm 3.1 on the pair (optimum, rank 2),
  in percent, separately per market and arm; the daily grid ``{1, 5, 10, 21, 42, 63, 126}``, 1000
  pseudo-series, 499 bootstrap repetitions. The length found is used for all series of the joint
  bootstrap of that market and arm. The algorithm calibrates the test of the arm's own
  statistic (``functional``): for MV (default) the studentised bootstrap test of the difference
  of Sharpe ratios, as in LW; for GMV (``functional="logvar"``) the same algorithm, the same
  pseudo-series and the same steps for the test of the difference of log variances
  (:func:`cif.stats.rq3.calibrate_logvar_block`), the statistic that H3a, H3b and the GMV arm
  actually use. For the descriptive operator (б) the pair is (optimum, mean of the replicas),
  with the same choice per arm (:func:`calibrate_replica_block`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from cif.stats.bootstrap import SeedLike, _as_generator, _positive_int, _seed_record
from cif.stats.rq3 import (
    ContrastEstimate,
    Functional,
    JointBootstrap,
    _check_alpha,
    _contrast_estimate,
    _ols_weights,
    calibrate_joint_block,
    h3a_outcome,
    joint_contrasts,
)
from cif.stats.sharpe import BlockCalibration, Kernel, _check_block
from cif.stats.tests_misc import holm, tost_from_ci

__all__ = [
    "BLOCK_GRID_DAILY",
    "CALIBRATION_N_BOOT",
    "CALIBRATION_N_PSEUDO",
    "H3B_RANKS",
    "MIN_MEDIAN_GAP_BP",
    "RANK_BIN_LABELS",
    "ArmDecision",
    "RankBins",
    "RankDoseResponse",
    "RankEquivalence",
    "RankSolverDose",
    "SolverResidual",
    "SolverResiduals",
    "calibrate_rank_block",
    "calibrate_replica_block",
    "holm_over_arms",
    "rank2_outcome",
    "rank_bins",
    "rank_dose_response",
    "rank_equivalence",
    "rank_solver_dose",
    "solver_residuals",
]

H3B_RANKS = (2, 3, 5, 9, 17, 33)  # log-spaced ranks of the H3b regression
BLOCK_GRID_DAILY = (1, 5, 10, 21, 42, 63, 126)  # daily blocks of the calibration (plan §7.4)
CALIBRATION_N_PSEUDO = 1000  # pseudo-series of Algorithm 3.1
CALIBRATION_N_BOOT = 499  # bootstrap repetitions inside Algorithm 3.1
MIN_MEDIAN_GAP_BP = 1.0  # H3c is descriptive unless two solvers have a median gap of this size
RANK_BIN_LABELS = ("1", "2-5", "6-33", ">33")  # descriptive rank distribution of a solver

Regressor = Literal["log_dose", "log_rank"]
_REGRESSORS = ("log_dose", "log_rank")
_RANGE_RTOL = 1e-9  # the dose range of the curve is closed; slack for rounding of the means


def _series(x: ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be a 1-D return series")
    return a


# ---------------------------------------------------------------------------------------------
# block length
# ---------------------------------------------------------------------------------------------


def calibrate_replica_block(
    opt: ArrayLike,
    replicas: Sequence[ArrayLike],
    *,
    seed: SeedLike,
    percent: bool = True,
    functional: Functional = "sharpe",
    block_grid: Sequence[int] = BLOCK_GRID_DAILY,
    n_pseudo: int = CALIBRATION_N_PSEUDO,
    n_boot: int = CALIBRATION_N_BOOT,
    **calibration_kwargs,
) -> BlockCalibration:
    """LW (2008) Algorithm 3.1 with the plan's daily settings on the pair (optimum, equal-weight
    mean of all replicas): the block length of the descriptive replica operator (б), plan §7.4.

    ``functional`` is the statistic of the arm whose test is calibrated: ``"sharpe"`` (MV, the
    default) or ``"logvar"`` (GMV, the difference of log variances). ``replicas`` are the
    ``(T, R)`` blocks of :func:`cif.stats.rq3.calibrate_joint_block`, which does the work; the
    plan's grid, 1000 pseudo-series and 499 repetitions are the defaults here.
    """
    return calibrate_joint_block(
        opt, replicas, seed=seed, percent=percent, functional=functional,
        block_grid=tuple(block_grid), n_pseudo=n_pseudo, n_boot=n_boot, **calibration_kwargs,
    )


def calibrate_rank_block(
    opt: ArrayLike,
    rank2: ArrayLike,
    *,
    seed: SeedLike,
    percent: bool = True,
    functional: Functional = "sharpe",
    block_grid: Sequence[int] = BLOCK_GRID_DAILY,
    n_pseudo: int = CALIBRATION_N_PSEUDO,
    n_boot: int = CALIBRATION_N_BOOT,
    **calibration_kwargs,
) -> BlockCalibration:
    """LW (2008) Algorithm 3.1 on the pair (optimum, rank 2) with the plan's daily settings, for
    the test of the arm's own statistic.

    ``functional="sharpe"`` (MV, default) calibrates the test of the difference of Sharpe ratios
    (:func:`cif.stats.sharpe.calibrate_block_size`); ``functional="logvar"`` (GMV) the test of the
    difference of log variances (:func:`cif.stats.rq3.calibrate_logvar_block`). The returns, given
    as fractions, are multiplied by 100 first (``percent``): LW's HAC inside the Sharpe
    calibration is not unit-free and the plan works in percent (the log variance does not care).
    Run it once per market and arm and pass ``BlockCalibration.block`` as ``block`` to every test
    of that market and arm (H3a, H3b, H3c). ``seed`` is mandatory (the plan fixes it in the
    config); ``calibration_kwargs`` go to the calibration. At the plan's sizes the algorithm costs
    about ``n_pseudo · len(block_grid)`` bootstrap tests of ``n_boot`` resamples each.
    """
    return calibrate_replica_block(
        _series(opt, "opt"), [_series(rank2, "rank2")], seed=seed, percent=percent,
        functional=functional, block_grid=block_grid, n_pseudo=n_pseudo, n_boot=n_boot,
        **calibration_kwargs,
    )


# ---------------------------------------------------------------------------------------------
# H3a: rank 2 against the optimum
# ---------------------------------------------------------------------------------------------


def rank2_outcome(*, equivalent: bool, significant: bool) -> str:
    """The four outcomes of H3a (plan §2): TOST passes / fails × the two-sided test significant /
    not. The same table as :func:`cif.stats.rq3.h3a_outcome`, with one test of each kind."""
    return h3a_outcome(iut_passes=equivalent, any_significant=significant)


def _tost_interval(joint: JointBootstrap, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """The studentised ``1 − 2α`` interval of the TOST per contrast. Its quantile of ``d*`` exists
    only from ``M ≥ 1/(2α) − 1`` resamples (9 at ``α = 0.05``); with fewer the error says so."""
    lo, hi = joint.ci(2.0 * alpha)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError(
            f"n_boot = {joint.n_boot} is too small for the {100.0 * (1.0 - 2.0 * alpha):g} % "
            f"interval of the TOST at alpha = {alpha:g}: it needs at least "
            f"{math.ceil(1.0 / (2.0 * alpha) - 1.0)} resamples"
        )
    return lo, hi


def _tost_pvalue(
    estimate: float, se: float, dstar: np.ndarray, margin: tuple[float, float]
) -> float:
    """Smallest ``α`` at which the studentised ``1 − 2α`` interval ``est ± z*·se`` lies strictly
    inside the margin, i.e. the TOST p-value that matches the interval decision.

    The interval passes iff the ``⌈(1 − 2α)(M + 1)⌉``-th smallest ``d*`` is below
    ``t = min(est − low, high − est)/se``, i.e. iff ``#{d* ≥ t} + 1 ≤ 2α(M + 1)``.
    """
    low, high = margin
    room = min(estimate - low, high - estimate)
    if room <= 0.0:
        return 1.0  # the point estimate is outside the margin: no α passes
    t = room / se if se > 0.0 else math.inf  # se = 0: the interval is the point estimate
    count = int(np.count_nonzero(dstar >= t))
    return min(1.0, (count + 1) / (2.0 * (dstar.size + 1)))


@dataclass(frozen=True)
class RankEquivalence:
    """H3a on one arm and panel. ``ci_tost`` is the ``1 − 2α`` interval of the TOST (90 % for
    ``α = 0.05``), ``ci`` the ``1 − α`` interval; ``pvalue`` is the two-sided p-value of
    ``H0: θ = 0`` (LW eq. 9) and ``pvalue_tost`` the p-value of the TOST (``equivalent`` iff
    ``pvalue_tost ≤ α``); ``significant`` is ``pvalue ≤ α``. The estimate is rank 2 minus the
    optimum: the annual Sharpe difference (``sharpe``) or the log variance ratio (``logvar``)."""

    outcome: str
    equivalent: bool
    significant: bool
    estimate: float
    se: float
    ci_tost: tuple[float, float]
    ci: tuple[float, float]
    pvalue: float
    pvalue_tost: float
    margin: tuple[float, float]
    alpha: float
    functional: str
    block: int
    n_boot: int
    n_obs: int
    seed: int | None


def rank_equivalence(
    opt: ArrayLike,
    rank2: ArrayLike,
    margin: float | tuple[float, float],
    *,
    functional: Functional = "sharpe",
    block: int,
    n_boot: int = 4999,
    alpha: float = 0.05,
    seed: SeedLike,
    periods_per_year: float | None = 252,
    ddof: int = 1,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
) -> RankEquivalence:
    """H3a (plan §2): the rank-2 path against the optimum, before costs; one TOST on the
    ``1 − 2α`` interval and one two-sided test at ``α``, mapped to the four outcomes.

    ``opt`` and ``rank2`` are the daily returns of the two paths on the days of ``F_k`` (excess
    returns for the Sharpe functional). ``margin`` is a positive number (``Δ_SR``, region
    ``(−Δ, Δ)``) or ``(low, high)`` (GMV: ``(log(1 − δ), log(1 + δ))``,
    :func:`cif.stats.rq3.variance_margin`). The bootstrap is the LW (2008) studentised circular
    block bootstrap of the pair; ``block`` is the length found by :func:`calibrate_rank_block`.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5) for the TOST")
    x0 = _series(opt, "opt")
    x2 = _series(rank2, "rank2")
    if x0.size != x2.size:
        raise ValueError("opt and rank2 must have the same length in time")
    joint = joint_contrasts(
        np.column_stack([x0, x2]), np.array([[-1.0, 1.0]]), functional=functional, block=block,
        n_boot=n_boot, seed=seed, periods_per_year=periods_per_year, ddof=ddof, kernel=kernel,
        prewhiten=prewhiten,
    )
    lo_tost, hi_tost = _tost_interval(joint, alpha)
    lo, hi = joint.ci(alpha)
    tost = tost_from_ci(float(lo_tost[0]), float(hi_tost[0]), margin)
    p = float(joint.pvalue()[0])
    significant = bool(p <= alpha)
    return RankEquivalence(
        outcome=rank2_outcome(equivalent=tost.equivalent, significant=significant),
        equivalent=tost.equivalent,
        significant=significant,
        estimate=float(joint.estimate[0]),
        se=float(joint.se[0]),
        ci_tost=(float(lo_tost[0]), float(hi_tost[0])),
        ci=(float(lo[0]), float(hi[0])),
        pvalue=p,
        pvalue_tost=_tost_pvalue(
            float(joint.estimate[0]), float(joint.se[0]), joint.dstar[:, 0],
            (tost.margin_low, tost.margin_high),
        ),
        margin=(tost.margin_low, tost.margin_high),
        alpha=float(alpha),
        functional=functional,
        block=joint.block,
        n_boot=joint.n_boot,
        n_obs=joint.n_obs,
        seed=joint.seed,
    )


@dataclass(frozen=True)
class ArmDecision:
    """H3a decision of one secondary arm after Holm over the arms."""

    pvalue_tost_holm: float
    pvalue_holm: float
    equivalent: bool
    significant: bool
    outcome: str


def holm_over_arms(
    results: Mapping[str, RankEquivalence], alpha: float = 0.05
) -> dict[str, ArmDecision]:
    """Holm (1979) over the secondary arms of H3a (FF49-GMV and US-MV, plan §7.1).

    The TOST p-values of the arms are adjusted as one family and the two-sided p-values as
    another; an arm is equivalent / significant iff its adjusted p-value is at most ``alpha``.
    The primary arm (FF49-MV) is not in the family.
    """
    if len(results) < 2:
        raise ValueError("Holm over arms needs at least two arms (the plan: two secondary arms)")
    names = list(results)
    p_tost = holm([results[n].pvalue_tost for n in names])
    p_two = holm([results[n].pvalue for n in names])
    out = {}
    for i, name in enumerate(names):
        equivalent = bool(p_tost[i] <= alpha)
        significant = bool(p_two[i] <= alpha)
        out[name] = ArmDecision(
            pvalue_tost_holm=float(p_tost[i]),
            pvalue_holm=float(p_two[i]),
            equivalent=equivalent,
            significant=significant,
            outcome=rank2_outcome(equivalent=equivalent, significant=significant),
        )
    return out


# ---------------------------------------------------------------------------------------------
# H3b: dose-response over the ranks
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankDoseResponse:
    """H3b: the line ``ΔSR(r) = a + b·x_r`` fitted by OLS to the rank differences, ``x_r =
    log ḡ_r`` (``regressor="log_dose"``, primary) or ``log r`` (``"log_rank"``, sensitivity).
    ``x`` are the in-sample abscissas, common to all bootstrap draws. ``per_rank`` are the
    differences themselves (descriptive; empty if not requested). ``slope`` and ``intercept``
    carry the joint-bootstrap inference; ``significant`` is the plan's decision: the percentile
    ``1 − α`` interval of the slope excludes zero (two-sided)."""

    ranks: tuple[int, ...]
    doses_bp: tuple[float, ...]
    regressor: str
    x: tuple[float, ...]
    per_rank: tuple[ContrastEstimate, ...]
    slope: ContrastEstimate
    intercept: ContrastEstimate
    significant: bool
    alpha: float
    functional: str
    block: int
    n_boot: int
    n_obs: int
    seed: int | None

    def predict(self, dose_bp: float) -> float:
        """Fitted ``a + b·log(dose)``, only for the fit on ``log_dose`` and only inside
        ``[min ḡ, max ḡ]`` (``[ḡ_2, ḡ_33]`` for the plan's ranks): outside it the curve is not
        extrapolated and the solver's calibration (H3c) is descriptive."""
        if self.regressor != "log_dose":
            raise ValueError("predict is defined for the fit on log_dose only (the curve of H3c)")
        lo, hi = min(self.doses_bp), max(self.doses_bp)
        if not lo <= dose_bp <= hi:
            raise ValueError(
                f"dose {dose_bp:g} bp is outside [{lo:g}, {hi:g}]: no prediction, "
                f"H3c is descriptive for it"
            )
        return self.intercept.estimate + self.slope.estimate * math.log(dose_bp)


def _check_ranks(ranks: Sequence[int]) -> tuple[int, ...]:
    out = []
    for v in ranks:
        if isinstance(v, bool) or int(v) != v:
            raise ValueError(f"ranks must be distinct integers ≥ 2, got {v!r}")
        out.append(int(v))
    if len(set(out)) != len(out) or any(v < 2 for v in out):
        raise ValueError("ranks must be distinct integers ≥ 2 (rank 1 is the optimum itself)")
    if len(out) < 3:
        raise ValueError(f"the slope needs at least 3 ranks, got {len(out)}: H3b is descriptive")
    return tuple(out)


def _prepare_fit(
    opt: ArrayLike, paths: ArrayLike, ranks: Sequence[int], doses_bp: ArrayLike, regressor: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    """Validated inputs of the regression over the ranks: the optimum, the ``(T, m)`` rank paths,
    the ranks, the mean doses and the in-sample abscissas ``x`` (``log ḡ_r`` or ``log r``)."""
    if regressor not in _REGRESSORS:
        raise ValueError(f"regressor must be one of {_REGRESSORS}, got {regressor!r}")
    x0 = _series(opt, "opt")
    rk = _check_ranks(ranks)
    m = len(rk)
    p = np.asarray(paths, dtype=float)
    if p.shape != (x0.size, m):
        raise ValueError(
            f"paths must be a (T, {m}) matrix, one column per rank, with T = {x0.size} days "
            f"like opt, got shape {p.shape}"
        )
    d = np.asarray(doses_bp, dtype=float)
    if d.shape != (m,):
        raise ValueError(f"doses_bp must hold one mean dose per rank ({m}), got shape {d.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("the mean doses must be finite")
    if regressor == "log_dose" and np.any(d <= 0.0):
        raise ValueError("the mean doses must be positive for the regression on log ḡ_r")
    x = np.log(d) if regressor == "log_dose" else np.log(np.asarray(rk, dtype=float))
    return x0, p, rk, d, x


def rank_dose_response(
    opt: ArrayLike,
    paths: ArrayLike,
    ranks: Sequence[int],
    doses_bp: ArrayLike,
    *,
    functional: Functional = "sharpe",
    regressor: Regressor = "log_dose",
    block: int,
    n_boot: int = 4999,
    alpha: float = 0.05,
    seed: SeedLike,
    per_rank: bool = True,
    periods_per_year: float | None = 252,
    ddof: int = 1,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
) -> RankDoseResponse:
    """H3b (plan §2): the OLS slope of ``ΔSR(r)`` on ``log ḡ_r`` over ``ranks`` of ``F_k``, with a
    percentile ``1 − α`` interval from the joint circular block bootstrap of the optimum and all
    rank paths on common resamples; significant iff the interval excludes zero.

    ``paths`` is ``(T, len(ranks))``, column ``j`` the daily returns of the rank-``ranks[j]``
    path on the days of ``F_k``, ``opt`` the optimum on the same days (excess returns for the
    Sharpe functional, as in :func:`rank_equivalence`). ``doses_bp[j]`` is
    ``ḡ_{ranks[j]}``, the mean in-sample gap over ``F_k``
    (:func:`cif.backtest.fold_sets.rank_doses`), the same for every draw. Sensitivities: pass
    ``regressor="log_rank"`` (``x = log r``) or all ranks 2–33 (``per_rank=False`` keeps only the
    slope and the intercept, which is much cheaper for 32 paths). ``functional="logvar"`` is the
    GMV arm: the slope of the log variance ratio.
    """
    _check_alpha(alpha)
    x0, p, rk, d, x = _prepare_fit(opt, paths, ranks, doses_bp, regressor)
    m = len(rk)
    w_slope, w_intercept = _ols_weights(x)
    diff = np.zeros((m, m + 1))  # ΔSR(r) = path_r − optimum
    diff[:, 0] = -1.0
    diff[np.arange(m), np.arange(m) + 1] = 1.0
    fit = np.vstack([w_slope @ diff, w_intercept @ diff])
    contrasts = np.vstack([diff, fit]) if per_rank else fit
    joint = joint_contrasts(
        np.column_stack([x0, p]), contrasts, functional=functional, block=block, n_boot=n_boot,
        seed=seed, periods_per_year=periods_per_year, ddof=ddof, kernel=kernel,
        prewhiten=prewhiten,
    )
    first = m if per_rank else 0
    slope = _contrast_estimate(joint, first, alpha)
    low, high = slope.ci_percentile
    return RankDoseResponse(
        ranks=rk,
        doses_bp=tuple(float(v) for v in d),
        regressor=regressor,
        x=tuple(float(v) for v in x),
        per_rank=tuple(_contrast_estimate(joint, q, alpha) for q in range(m)) if per_rank else (),
        slope=slope,
        intercept=_contrast_estimate(joint, first + 1, alpha),
        significant=bool(not low <= 0.0 <= high),
        alpha=float(alpha),
        functional=functional,
        block=joint.block,
        n_boot=joint.n_boot,
        n_obs=joint.n_obs,
        seed=joint.seed,
    )


# ---------------------------------------------------------------------------------------------
# H3c: the dose of a real solver and its residual against the H3b curve
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankSolverDose:
    """Dose of a real solver for H3c: its mean realised in-sample gap over ``F_k`` and the median
    gap. ``material`` — the median gap is at least ``min_median_gap_bp``. A prediction from the H3b
    curve is made only if ``inside_range`` (the dose lies in ``[min ḡ, max ḡ]``, ``[ḡ_2, ḡ_33]``
    for the plan's ranks); otherwise the solver is ``descriptive`` (``reason``)."""

    dose_bp: float | None
    median_gap_bp: float | None
    n_folds: int
    material: bool
    inside_range: bool
    descriptive: bool
    reason: str | None


def rank_solver_dose(
    realised_gap_bp: ArrayLike,
    fold_mask: ArrayLike,
    doses_bp: ArrayLike,
    min_median_gap_bp: float = MIN_MEDIAN_GAP_BP,
) -> RankSolverDose:
    """Dose of a solver: the mean of its realised ``ĝ_CE`` over the folds of ``F_k`` (zero where
    it found the optimum), tested against the range of the mean doses ``ḡ_r`` of the curve.

    ``realised_gap_bp[f]`` is the gap of the solver's incumbent on fold ``f`` (it may be missing
    outside ``fold_mask``), ``doses_bp`` the mean doses of the H3b fit. Both ends of the range are
    inside (to a relative ``10⁻⁹`` for rounding: a solver that lands on rank 2 on every fold has
    the dose ``ḡ_2``, formed by another sum).
    """
    gaps = np.asarray(realised_gap_bp, dtype=float)
    mask = np.asarray(fold_mask, dtype=bool)
    if gaps.ndim != 1 or gaps.shape != mask.shape:
        raise ValueError(
            "realised_gap_bp and fold_mask must be 1-D arrays with one entry per fold "
            f"(the same number of folds), got shapes {gaps.shape} and {mask.shape}"
        )
    d = np.asarray(doses_bp, dtype=float)
    if d.ndim != 1 or d.size == 0 or not np.all(np.isfinite(d)):
        raise ValueError("doses_bp must be a non-empty 1-D array of finite mean doses")
    chosen = gaps[mask]
    if not np.all(np.isfinite(chosen)):
        raise ValueError("the realised gaps on the folds of F_k must be finite")
    if chosen.size == 0:
        return RankSolverDose(None, None, 0, False, False, True, "no folds in F_k")
    dose = float(chosen.mean())
    median = float(np.median(chosen))
    lo, hi = float(d.min()), float(d.max())
    reason = None
    if dose < lo * (1.0 - _RANGE_RTOL):
        reason = f"dose {dose:.4g} bp is below the smallest mean dose {lo:g} bp of the curve"
    elif dose > hi * (1.0 + _RANGE_RTOL):
        reason = f"dose {dose:.4g} bp is above the largest mean dose {hi:g} bp of the curve"
    return RankSolverDose(
        dose_bp=dose,
        median_gap_bp=median,
        n_folds=int(chosen.size),
        material=bool(median >= min_median_gap_bp),
        inside_range=reason is None,
        descriptive=reason is not None,
        reason=reason,
    )


@dataclass(frozen=True)
class SolverResidual:
    """H3c for one solver. ``observed`` is the pre-cost difference solver path minus optimum (annual
    Sharpe, ``F_k`` days), ``predicted`` the H3b curve ``a + b·log(dose)`` at the solver's dose,
    ``residual`` their difference with its studentised interval; ``ci_tost`` is the ``1 − 2α``
    interval of the TOST and ``equivalent`` its outcome against the margin. All of them are
    ``None`` for a solver without a prediction (``dose.descriptive``) and in a descriptive H3c."""

    name: str
    dose: RankSolverDose
    observed: ContrastEstimate | None
    predicted: float | None
    residual: ContrastEstimate | None
    ci_tost: tuple[float, float] | None
    equivalent: bool | None


@dataclass(frozen=True)
class SolverResiduals:
    """H3c on FF49-MV. ``iut_passes`` — all tested residuals are equivalent to zero within the
    margin (intersection–union test over the solvers with a prediction); ``None`` if ``descriptive``
    (``reason``): then the calibration is not tested and only the doses are reported."""

    solvers: tuple[SolverResidual, ...]
    iut_passes: bool | None
    descriptive: bool
    reason: str | None
    n_material: int
    margin: tuple[float, float]
    alpha: float
    functional: str
    block: int
    n_boot: int
    n_obs: int
    seed: int | None


def solver_residuals(
    opt: ArrayLike,
    paths: ArrayLike,
    ranks: Sequence[int],
    doses_bp: ArrayLike,
    solver_paths: Mapping[str, ArrayLike],
    solver_gaps_bp: Mapping[str, ArrayLike],
    fold_mask: ArrayLike,
    margin: float | tuple[float, float],
    *,
    functional: Functional = "sharpe",
    block: int,
    n_boot: int = 4999,
    alpha: float = 0.05,
    seed: SeedLike,
    min_median_gap_bp: float = MIN_MEDIAN_GAP_BP,
    periods_per_year: float | None = 252,
    ddof: int = 1,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
) -> SolverResiduals:
    """H3c (plan §2): calibration of real solvers against the H3b curve, before costs.

    The residual of a solver is its observed pre-cost Sharpe difference to the optimum (on the
    days of ``F_k``) minus the prediction ``a + b·log(dose)`` of the H3b fit over ``ranks``
    (``paths``, ``doses_bp`` as in :func:`rank_dose_response`), with ``dose`` the mean of the
    solver's realised gap over ``F_k``. Both are linear in the Sharpe ratios of the paths, so the
    residual ``SR_solver − Σ_r c_r·SR_r`` (the OLS prediction weights ``c_r`` sum to one and the
    optimum drops out) is a contrast of the joint bootstrap of the optimum, the rank paths and the
    solver paths on common resamples; the dose and the abscissas are in-sample and fixed. Each
    residual is tested by TOST on the ``1 − 2α`` interval with the margin ``Δ_SR``; the IUT is the
    conjunction over the solvers with a prediction.

    ``solver_paths[name]`` are the solver's daily (excess) returns on the days of ``F_k``, like
    ``opt`` and ``paths``, and ``solver_gaps_bp[name]`` its realised gap on every fold
    (``fold_mask`` selects ``F_k``; the gaps outside it may be missing). The
    plan's rules make H3c descriptive, without any test: fewer than two solvers have a median gap
    of at least ``min_median_gap_bp`` (1 bp); or fewer than two have a dose inside
    ``[min ḡ, max ḡ]`` (a solver outside it is descriptive for itself in any case).
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5) for the TOST")
    checked = tost_from_ci(0.0, 0.0, margin)  # validates the margin
    m_low, m_high = checked.margin_low, checked.margin_high
    x0, p, rk, d, x = _prepare_fit(opt, paths, ranks, doses_bp, "log_dose")
    # the bootstrap settings are checked up front: a bad one must not hide behind a descriptive H3c
    block = _check_block(block, x0.size)
    n_boot = _positive_int(n_boot, "n_boot")
    _as_generator(seed)
    names = list(solver_paths)
    if not names:
        raise ValueError("need at least one solver")
    if set(names) != set(solver_gaps_bp):
        raise ValueError("solver_paths and solver_gaps_bp must name the same solvers")
    cols = {}
    for name in names:
        a = np.asarray(solver_paths[name], dtype=float)
        if a.shape != x0.shape:
            raise ValueError(
                f"solver path {name!r} must be a 1-D series of the same length as opt "
                f"({x0.size} days), got shape {a.shape}"
            )
        cols[name] = a
    mask = np.asarray(fold_mask, dtype=bool)
    doses = [rank_solver_dose(solver_gaps_bp[n], mask, d, min_median_gap_bp) for n in names]
    n_material = sum(s.material for s in doses)
    lo, hi = float(d.min()), float(d.max())
    testable = [i for i, s in enumerate(doses) if s.inside_range]

    def finish(solvers, iut, reason, n_boot_used, block_used, seed_used):
        return SolverResiduals(
            solvers=tuple(solvers), iut_passes=iut, descriptive=iut is None, reason=reason,
            n_material=n_material, margin=(m_low, m_high), alpha=float(alpha),
            functional=functional, block=block_used, n_boot=n_boot_used, n_obs=x0.size,
            seed=seed_used,
        )

    bare = [
        SolverResidual(n, s, None, None, None, None, None)
        for n, s in zip(names, doses, strict=True)
    ]
    if n_material < 2:
        return finish(
            bare, None,
            f"fewer than two solvers with a median gap of at least {min_median_gap_bp:g} bp "
            f"({n_material} of {len(names)}): H3c is descriptive",
            n_boot, block, _seed_record(seed),
        )
    if len(testable) < 2:
        return finish(
            bare, None,
            f"fewer than two solvers with a dose inside [{lo:g}, {hi:g}] bp ({len(testable)} of "
            f"{len(names)}): no prediction from the H3b curve, H3c is descriptive",
            n_boot, block, _seed_record(seed),
        )

    w_slope, w_intercept = _ols_weights(x)
    k, m = len(testable), len(rk)
    rows = np.zeros((2 * k, 1 + m + k))
    for j, i in enumerate(testable):
        col = 1 + m + j
        weights = w_intercept + math.log(doses[i].dose_bp) * w_slope  # fitted value at the dose
        rows[2 * j, col], rows[2 * j, 0] = 1.0, -1.0  # observed: solver − optimum
        rows[2 * j + 1, col], rows[2 * j + 1, 1 : 1 + m] = 1.0, -weights  # solver − Σ c_r path_r
    joint = joint_contrasts(
        np.column_stack([x0, p, *(cols[names[i]] for i in testable)]), rows,
        functional=functional, block=block, n_boot=n_boot, seed=seed,
        periods_per_year=periods_per_year, ddof=ddof, kernel=kernel, prewhiten=prewhiten,
    )
    lo_tost, hi_tost = _tost_interval(joint, alpha)
    tested: dict[int, SolverResidual] = {}
    for j, i in enumerate(testable):
        observed = _contrast_estimate(joint, 2 * j, alpha)
        residual = _contrast_estimate(joint, 2 * j + 1, alpha)
        ci_tost = (float(lo_tost[2 * j + 1]), float(hi_tost[2 * j + 1]))
        tested[i] = SolverResidual(
            name=names[i],
            dose=doses[i],
            observed=observed,
            predicted=observed.estimate - residual.estimate,
            residual=residual,
            ci_tost=ci_tost,
            equivalent=tost_from_ci(*ci_tost, margin).equivalent,
        )
    solvers = [tested.get(i, bare[i]) for i in range(len(names))]
    iut = all(s.equivalent for s in tested.values())
    return finish(solvers, iut, None, joint.n_boot, joint.block, joint.seed)


# ---------------------------------------------------------------------------------------------
# the rank distribution of a solver (descriptive)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RankBins:
    """Counts and shares of the folds by the rank of a solver's portfolio, in the bins
    :data:`RANK_BIN_LABELS` (rank 1 is the optimum)."""

    labels: tuple[str, ...]
    counts: tuple[int, ...]
    shares: tuple[float, ...]
    n_folds: int


def rank_bins(ranks: ArrayLike, fold_mask: ArrayLike | None = None) -> RankBins:
    """Distribution of the rank of a solver's portfolio over the folds of ``F_k`` in the plan's
    bins: ``1``; ``2–5``; ``6–33``; ``> 33``.

    ``ranks[f]`` is the rank (``≥ 1``, an integer) of the solver's final portfolio among the
    distinct refined portfolios of fold ``f``; a portfolio beyond the enumerated ranks takes any
    value above 33 (e.g. 34). Finding the rank (by the support) is up to the experiment.
    """
    r = np.asarray(ranks)
    if r.ndim != 1:
        raise ValueError("ranks must be a 1-D array with one rank per fold")
    if fold_mask is not None:
        mask = np.asarray(fold_mask, dtype=bool)
        if mask.shape != r.shape:
            raise ValueError(
                f"fold_mask must have one flag per fold ({r.size} folds), got shape {mask.shape}"
            )
        r = r[mask]
    if r.size == 0:
        raise ValueError("no folds to count the ranks of")
    rf = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(rf)) or np.any(rf != np.round(rf)):
        raise ValueError("ranks must be integers")
    ri = rf.astype(int)
    if np.any(ri < 1):
        raise ValueError("ranks start at 1 (the optimum)")
    counts = (
        int(np.count_nonzero(ri == 1)),
        int(np.count_nonzero((ri >= 2) & (ri <= 5))),
        int(np.count_nonzero((ri >= 6) & (ri <= 33))),
        int(np.count_nonzero(ri >= 34)),
    )
    return RankBins(
        labels=RANK_BIN_LABELS,
        counts=counts,
        shares=tuple(c / ri.size for c in counts),
        n_folds=int(ri.size),
    )
