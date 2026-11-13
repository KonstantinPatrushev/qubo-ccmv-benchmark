"""Sharpe-ratio inference: JKM, Ledoit-Wolf (2008) HAC and bootstrap tests, PSR and DSR.

Formulas and where they come from
---------------------------------
Ledoit, O., Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio.
Journal of Empirical Finance 15(5), 850-859 (LW). With v = (mu_1, mu_2, gamma_1, gamma_2),
gamma_i = E r_i^2:

- eqs. (1)-(2): Delta = f(v), f(a, b, c, d) = a / sqrt(c - a^2) - b / sqrt(d - b^2);
- eq. (4): gradient of f; eq. (5): s(Delta_hat) = sqrt(grad' Psi_hat grad / T);
- Sec. 3.1: Psi_hat = T/(T-4) sum_j k(j/S_T) Gamma_hat(j), y_t = (r_1t - mu_1, r_2t - mu_2,
  r_1t^2 - gamma_1, r_2t^2 - gamma_2); p-value 2 Phi(-|Delta_hat| / s(Delta_hat));
- Sec. 3.2.2: studentized circular block bootstrap; Psi*_hat = l^-1 sum_j zeta_j zeta_j',
  zeta_j = b^-1/2 sum of the j-th block of y*_t (centred at bootstrap moments), l = floor(T/b);
  eq. (8): s(Delta*_hat); eq. (7): interval Delta_hat +- z*_{|.|,1-alpha} s(Delta_hat);
  eq. (9): p-value (#{d* >= d} + 1) / (M + 1); Algorithm 3.1: block-size calibration;
- Sec. 4.1: "HAC" = QS kernel with Andrews (1991) automatic bandwidth; "HAC_pw" = prewhitened
  QS kernel (Andrews & Monahan 1992). Remark 4.1: Parzen gives virtually identical results.

Implementation choices that follow the authors' code (Wolf, R/Matlab, version 2026-09) and were
checked against LW Table 3 (see tests/test_stats.py): Delta_hat uses sample standard deviations
(ddof = 1) while the standard error uses population moments (eqs. 4-5); Andrews' AR(1) plug-in
uses unit weights over the four moment conditions; prewhitening is a VAR(1) without intercept
whose singular values are capped at 0.97. The QS kernel is summed over all lags (the 2026 Matlab
code truncates it at S_T; LW Table 3 is reproduced only without truncation).

Deviation from the 2026 authors' code, following the paper instead: the bootstrap sample has
length T (ceil(T/b) blocks, truncated), and Psi*_hat carries no T/(T-4) factor; the 2026 code
uses T_adj = floor(T/b) b observations and multiplies Psi*_hat by T_adj/(T_adj-4).

Caveat (inherited from the method): with unit weights the Andrews bandwidth and the 0.97 cap of
the prewhitening step are not invariant to the units of returns. LW work with returns in percent.
``standardize=True`` makes both unit-free (exploratory option, not in LW).

Other sources: Jobson & Korkie (1981), J. Finance 36(4), 889-908, corrected by Memmel (2003),
Finance Letters 1, 21-23 (delta method with the iid-normal covariance Omega, LW Sec. 2);
Andrews (1991), Econometrica 59(3), 817-858, Sec. 6 (bandwidth constants 2.6614 Parzen, 1.3221 QS,
AR(1) plug-in alpha(2)); Andrews & Monahan (1992), Econometrica 60(4), 953-966 (prewhitening);
Bailey & Lopez de Prado (2012), J. Risk 15(2), eq. (11) (PSR); Bailey & Lopez de Prado (2014),
J. Portfolio Management 40(5), 94-107, eq. (2) and Appendix A.1 eqs. (5)-(6) (DSR).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import fft as sp_fft
from scipy import stats as sps

from cif.stats.bootstrap import (
    _CHUNK,
    SeedLike,
    _as_generator,
    _positive_int,
    _seed_record,
    circular_block_bootstrap_indices,
    stationary_bootstrap_indices,
)

__all__ = [
    "BlockCalibration",
    "DeflatedSharpeResult",
    "SharpeTestResult",
    "calibrate_block_size",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "hac_covariance",
    "jobson_korkie_memmel",
    "ledoit_wolf_sharpe_test",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
]

Kernel = Literal["qs", "parzen"]

# Andrews (1991), Sec. 6: S_T = c (alpha(2) T)^(1/5) for kernels with characteristic exponent 2.
_ANDREWS_CONST: dict[str, float] = {"parzen": 2.6614, "qs": 1.3221}
_PREWHITEN_MAX_SV = 0.97  # cap on singular values of the VAR(1) matrix (authors' code)
_RCOND_MIN = 1e-10  # numerical guards of the authors' 2026 code


# ---------------------------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------------------------


def _excess(r: ArrayLike, rf: ArrayLike | None) -> np.ndarray:
    x = np.asarray(r, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError("returns must be 1-D or 2-D (time along axis 0)")
    if rf is not None:
        f = np.asarray(rf, dtype=float)
        if f.ndim == 1 and x.ndim == 2:
            f = f[:, None]
        if np.broadcast_shapes(x.shape, f.shape) != x.shape:
            raise ValueError("rf must be a scalar or aligned with the returns along axis 0")
        x = x - f
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain non-finite values")
    return x


def _pair(r1: ArrayLike, r2: ArrayLike, rf: ArrayLike | None) -> np.ndarray:
    a = np.asarray(r1, dtype=float)
    b = np.asarray(r2, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("r1 and r2 must be 1-D arrays of equal length")
    r = _excess(np.column_stack([a, b]), rf)
    if r.shape[0] < 10:
        raise ValueError("need at least 10 observations")
    if np.any(r.std(axis=0) == 0.0):
        raise ValueError("a return series has zero variance; the Sharpe ratio is undefined")
    return r


def _check_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(alpha)


def _check_block(block: Any, T: int) -> int:
    b = _positive_int(block, "block")
    if b > T // 2:
        raise ValueError(f"block ({b}) must allow at least two blocks (T = {T})")
    return b


# ---------------------------------------------------------------------------------------------
# moments, gradient and HAC covariance
# ---------------------------------------------------------------------------------------------


def _moments(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Means and population variances along the time axis (axis -2) of (..., T, 2) data."""
    mu = r.mean(axis=-2)
    var = ((r - mu[..., None, :]) ** 2).mean(axis=-2)
    return mu, var


def _sharpe_from_moments(mu: np.ndarray, var: np.ndarray, T: int, ddof: int) -> np.ndarray:
    return mu / np.sqrt(var * T / (T - ddof))


def _gradient(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """LW (2008) eq. (4): grad f at (mu_1, mu_2, gamma_1, gamma_2); gamma - mu^2 = var."""
    gam = var + mu**2
    v15 = var**1.5
    return np.stack(
        [
            gam[..., 0] / v15[..., 0],
            -gam[..., 1] / v15[..., 1],
            -0.5 * mu[..., 0] / v15[..., 0],
            0.5 * mu[..., 1] / v15[..., 1],
        ],
        axis=-1,
    )


def _rcond(a: np.ndarray) -> float:
    c = float(np.linalg.cond(a, 1))
    return 0.0 if not np.isfinite(c) or c == 0.0 else 1.0 / c


def _andrews_alpha2(w: np.ndarray) -> float:
    """Andrews (1991) alpha(2) from univariate AR(1) fits (OLS with intercept), unit weights."""
    T, p = w.shape
    num = den = 0.0
    ones = np.ones(T - 1)
    for a in range(p):
        y = w[:, a]
        X = np.column_stack([ones, y[:-1]])
        xtx = X.T @ X
        if _rcond(xtx) < _RCOND_MIN:
            continue
        beta = np.linalg.solve(xtx, X.T @ y[1:])
        rho = float(beta[1])
        if abs(1.0 - rho) < 1e-6:
            continue
        e = y[1:] - X @ beta
        s2 = float(e @ e) / (T - 1)  # normalisation cancels in the ratio
        num += 4.0 * rho**2 * s2**2 / (1.0 - rho) ** 8
        den += s2**2 / (1.0 - rho) ** 4
    return num / den if den > 0.0 else 0.0


def _kernel_weights(x: np.ndarray, kernel: str) -> np.ndarray:
    if kernel == "parzen":
        a = np.abs(x)
        return np.where(
            a <= 0.5, 1.0 - 6.0 * a**2 + 6.0 * a**3, np.where(a <= 1.0, 2.0 * (1.0 - a) ** 3, 0.0)
        )
    # quadratic spectral, x != 0
    z = 6.0 * np.pi * x / 5.0
    return 25.0 / (12.0 * np.pi**2 * x**2) * (np.sin(z) / z - np.cos(z))


def _autocovariances(w: np.ndarray) -> np.ndarray:
    """Gamma_hat(j) = T^-1 sum_{t>j} w_t w_{t-j}' for j = 0..T-1 (FFT); shape (T, p, p)."""
    T = w.shape[0]
    nfft = sp_fft.next_fast_len(2 * T - 1, real=True)
    F = np.fft.rfft(w, n=nfft, axis=0)
    cross = np.fft.irfft(F[:, :, None] * np.conj(F[:, None, :]), n=nfft, axis=0)[:T]
    return cross / T


def _kernel_psi(w: np.ndarray, kernel: str, df_adjust: int) -> tuple[np.ndarray, float, float]:
    T = w.shape[0]
    alpha2 = _andrews_alpha2(w)
    S = _ANDREWS_CONST[kernel] * (alpha2 * T) ** 0.2 if alpha2 > 0.0 else 0.0
    S = min(S, T - 1.0)
    gam = _autocovariances(w)
    psi = gam[0].copy()
    if S > 0.0:
        weights = _kernel_weights(np.arange(1, T) / S, kernel)
        nz = np.flatnonzero(weights)
        if nz.size:
            acc = np.einsum("j,jab->ab", weights[nz], gam[1:][nz])
            psi += acc + acc.T
    return psi * (T / (T - df_adjust)), float(S), float(alpha2)


def hac_covariance(
    v: ArrayLike,
    *,
    kernel: Kernel = "qs",
    prewhiten: bool = False,
    standardize: bool = False,
    df_adjust: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Kernel (HAC) estimate of the long-run covariance of mean-zero moment conditions.

    ``v`` is a (T, p) array; it is *not* re-centred. Bandwidth: Andrews (1991) AR(1) plug-in with
    unit weights, S_T = 2.6614 (alpha(2) T)^(1/5) for Parzen and 1.3221 (alpha(2) T)^(1/5) for QS,
    capped at T - 1. The estimate is multiplied by T / (T - df_adjust) (LW 2008 use 4).
    ``prewhiten`` applies VAR(1) prewhitening (Andrews & Monahan 1992) with singular values capped
    at 0.97 and recolouring by (I - A)^-1; it falls back to no prewhitening when the regression is
    numerically singular (as the authors' code does). ``standardize`` rescales columns to unit
    variance before bandwidth selection and prewhitening (unit-free; not part of LW).

    Returns ``(psi, info)`` with ``info`` holding ``bandwidth``, ``alpha2``, ``kernel``,
    ``prewhitened`` and ``standardized``.
    """
    w = np.asarray(v, dtype=float)
    if w.ndim != 2:
        raise ValueError("v must be a 2-D (T, p) array")
    if kernel not in _ANDREWS_CONST:
        raise ValueError(f"unknown kernel {kernel!r}; use 'qs' or 'parzen'")
    T, p = w.shape
    if df_adjust >= T - 1 or T < 4:
        raise ValueError("too few observations for the HAC estimator")
    if not np.all(np.isfinite(w)):
        raise ValueError("v contains non-finite values")
    scale = None
    if standardize:
        scale = w.std(axis=0)
        scale[scale == 0.0] = 1.0
        w = w / scale
    prewhitened = False
    if prewhiten:
        X, Y = w[:-1], w[1:]
        xtx = X.T @ X
        if _rcond(xtx) >= _RCOND_MIN:
            A = np.linalg.solve(xtx, X.T @ Y).T  # row j: coefficients of equation j
            U, s, Vt = np.linalg.svd(A)
            A = (U * np.clip(s, -_PREWHITEN_MAX_SV, _PREWHITEN_MAX_SV)) @ Vt
            psi_e, S, alpha2 = _kernel_psi(Y - X @ A.T, kernel, df_adjust)
            D = np.linalg.inv(np.eye(p) - A)
            psi = D @ psi_e @ D.T
            prewhitened = True
    if not prewhitened:
        psi, S, alpha2 = _kernel_psi(w, kernel, df_adjust)
    if scale is not None:
        psi = psi * np.outer(scale, scale)
    info = {
        "bandwidth": S,
        "alpha2": alpha2,
        "kernel": kernel,
        "prewhitened": prewhitened,
        "standardized": standardize,
    }
    return psi, info


def _lw_standard_error(
    r: np.ndarray, kernel: str, prewhiten: bool, standardize: bool
) -> tuple[float, dict[str, Any]]:
    """LW (2008) eq. (5) with the HAC Psi_hat of Sec. 3.1."""
    T = r.shape[0]
    mu, var = _moments(r)
    grad = _gradient(mu, var)
    sq = r**2
    v = np.column_stack([r - mu, sq - sq.mean(axis=0)])
    psi, info = hac_covariance(
        v, kernel=kernel, prewhiten=prewhiten, standardize=standardize, df_adjust=4
    )  # type: ignore[arg-type]
    quad = max(float(grad @ psi @ grad), 0.0)
    return math.sqrt(quad / T), info


# ---------------------------------------------------------------------------------------------
# Sharpe ratio and tests
# ---------------------------------------------------------------------------------------------


def sharpe_ratio(
    r: ArrayLike,
    rf: ArrayLike | None = None,
    periods_per_year: float | None = 252,
    *,
    ddof: int = 1,
) -> Any:
    """Sharpe ratio of excess returns ``r - rf``: mean / std(ddof) * sqrt(periods_per_year).

    ``rf`` is a per-period rate (scalar or aligned with ``r``). ``periods_per_year=None`` gives the
    per-period ratio. 2-D input returns one ratio per column.
    """
    x = _excess(r, rf)
    if x.shape[0] < 2:
        raise ValueError("need at least two observations")
    sd = x.std(axis=0, ddof=ddof)
    if np.any(sd == 0.0):
        raise ValueError("zero variance; the Sharpe ratio is undefined")
    sr = x.mean(axis=0) / sd
    if periods_per_year is not None:
        sr = sr * math.sqrt(periods_per_year)
    return float(sr) if np.ndim(sr) == 0 else sr


@dataclass(frozen=True)
class SharpeTestResult:
    """Result of a test of H0: SR_1 - SR_2 = null.

    ``sharpe1``, ``sharpe2``, ``difference``, ``se``, ``ci_low``, ``ci_high`` are per period, or
    annualised by sqrt(periods_per_year) when that argument was given; ``statistic`` (signed,
    (Delta_hat - null) / se) and ``pvalue`` (two-sided) do not depend on the scaling. The interval
    has coverage ``1 - alpha``.
    """

    sharpe1: float
    sharpe2: float
    difference: float
    se: float
    statistic: float
    pvalue: float
    ci_low: float
    ci_high: float
    alpha: float
    method: str
    n_obs: int
    params: dict[str, Any] = field(default_factory=dict)


def _studentize(num: float, se: float) -> float:
    if num == 0.0:
        return 0.0
    return num / se if se > 0.0 else math.copysign(math.inf, num)


def _finish(
    sr: np.ndarray,
    se: float,
    statistic: float,
    pvalue: float,
    half_width: float,
    alpha: float,
    method: str,
    T: int,
    periods_per_year: float | None,
    params: dict[str, Any],
) -> SharpeTestResult:
    scale = math.sqrt(periods_per_year) if periods_per_year is not None else 1.0
    diff = float(sr[0] - sr[1])
    params = {**params, "periods_per_year": periods_per_year}
    return SharpeTestResult(
        sharpe1=float(sr[0]) * scale,
        sharpe2=float(sr[1]) * scale,
        difference=diff * scale,
        se=se * scale,
        statistic=statistic,
        pvalue=pvalue,
        ci_low=(diff - half_width) * scale,
        ci_high=(diff + half_width) * scale,
        alpha=alpha,
        method=method,
        n_obs=T,
        params=params,
    )


def jobson_korkie_memmel(
    r1: ArrayLike,
    r2: ArrayLike,
    *,
    rf: ArrayLike | None = None,
    alpha: float = 0.05,
    periods_per_year: float | None = None,
) -> SharpeTestResult:
    """Jobson-Korkie (1981) test corrected by Memmel (2003); valid for iid normal returns only.

    z = (s_2 m_1 - s_1 m_2) / sqrt(theta), theta = T^-1 [2 s_1^2 s_2^2 - 2 s_1 s_2 s_12
    + m_1^2 s_2^2 / 2 + m_2^2 s_1^2 / 2 - m_1 m_2 s_12^2 / (s_1 s_2)], i.e. the delta method with
    the iid-normal Omega (LW 2008, Sec. 2). Maximum-likelihood moments (ddof = 0), which
    reproduce LW (2008) Table 3. ``difference`` uses the same moments; ``se`` = sqrt(theta)/(s_1 s_2).
    """
    alpha = _check_alpha(alpha)
    r = _pair(r1, r2, rf)
    T = r.shape[0]
    mu, var = _moments(r)
    c = r - mu
    cov12 = float((c[:, 0] * c[:, 1]).mean())
    s1, s2 = math.sqrt(var[0]), math.sqrt(var[1])
    theta = (
        2.0 * var[0] * var[1]
        - 2.0 * s1 * s2 * cov12
        + 0.5 * mu[0] ** 2 * var[1]
        + 0.5 * mu[1] ** 2 * var[0]
        - mu[0] * mu[1] * cov12**2 / (s1 * s2)
    ) / T
    sr = mu / np.sqrt(var)
    se = math.sqrt(max(float(theta), 0.0)) / (s1 * s2)
    stat = _studentize(float(sr[0] - sr[1]), se)
    pvalue = float(2.0 * sps.norm.sf(abs(stat)))
    half = float(sps.norm.ppf(1.0 - alpha / 2.0)) * se
    return _finish(
        sr, se, stat, pvalue, half, alpha, "jkm", T, periods_per_year, {"moments": "mle"}
    )


def _bootstrap_statistics(
    samples: np.ndarray, diff_hat: float, block: int, ddof: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d*, s(Delta*) and Delta* for bootstrap samples of shape (m, T, 2) (LW 2008, Sec. 3.2.2).

    Psi* = l^-1 sum_j zeta_j zeta_j' over the l = floor(T/b) complete blocks, zeta_j = b^-1/2
    times the block sum of y*_t centred at the bootstrap moments; s(Delta*) by eq. (8).
    d* = |Delta* - Delta_hat| / s(Delta*), with 0/0 := 0.
    """
    s = np.asarray(samples, dtype=float)
    return _bootstrap_statistics_cols(
        np.ascontiguousarray(s[..., 0]), np.ascontiguousarray(s[..., 1]), diff_hat, block, ddof
    )


def _bootstrap_statistics_cols(
    s1: np.ndarray, s2: np.ndarray, diff_hat: float, block: int, ddof: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise core of :func:`_bootstrap_statistics`; ``s1``, ``s2`` have shape (m, T)."""
    m, T = s1.shape
    n_blocks = T // block
    lb = n_blocks * block
    mu1 = s1.mean(axis=1)
    mu2 = s2.mean(axis=1)
    c1 = s1 - mu1[:, None]
    c2 = s2 - mu2[:, None]
    q1 = c1 * c1
    q2 = c2 * c2
    var1 = q1.mean(axis=1)
    var2 = q2.mean(axis=1)
    v1 = var1**1.5
    v2 = var2**1.5
    g1 = (var1 + mu1**2) / v1  # gradient, LW eq. (4)
    g2 = -(var2 + mu2**2) / v2
    g3 = -0.5 * mu1 / v1
    g4 = 0.5 * mu2 / v2
    # grad' y*_t with y*_t = (c1, c2, r1^2 - gamma1, r2^2 - gamma2) and
    # r^2 - gamma = c^2 + 2 mu c - var (exact identity), restricted to the l complete blocks
    proj = (
        (g1 + 2.0 * g3 * mu1)[:, None] * c1[:, :lb]
        + g3[:, None] * (q1[:, :lb] - var1[:, None])
        + (g2 + 2.0 * g4 * mu2)[:, None] * c2[:, :lb]
        + g4[:, None] * (q2[:, :lb] - var2[:, None])
    )
    zeta = proj.reshape(m, n_blocks, block).sum(axis=2) / math.sqrt(block)
    se = np.sqrt((zeta * zeta).mean(axis=1) / T)
    f = T / (T - ddof)
    diff = mu1 / np.sqrt(var1 * f) - mu2 / np.sqrt(var2 * f)
    num = np.abs(diff - diff_hat)
    with np.errstate(divide="ignore", invalid="ignore"):
        dstar = np.where(num == 0.0, 0.0, num / se)
    return dstar, se, diff


def _bootstrap_pvalue(
    r: np.ndarray,
    diff_hat: float,
    se_hat: float,
    block: int,
    n_boot: int,
    gen: np.random.Generator,
    ddof: int,
    null: float,
) -> tuple[float, float, np.ndarray]:
    T = r.shape[0]
    d = abs(_studentize(diff_hat - null, se_hat))
    x1 = np.ascontiguousarray(r[:, 0])
    x2 = np.ascontiguousarray(r[:, 1])
    dstar = np.empty(n_boot)
    for start in range(0, n_boot, _CHUNK):
        m = min(_CHUNK, n_boot - start)
        idx = circular_block_bootstrap_indices(T, block, m, gen)
        dstar[start : start + m] = _bootstrap_statistics_cols(
            x1[idx], x2[idx], diff_hat, block, ddof
        )[0]
    pvalue = (int(np.count_nonzero(dstar >= d)) + 1) / (n_boot + 1)
    return pvalue, d, dstar


def _child_generator(seed: SeedLike) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed.spawn(1)[0]
    return np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(1)[0])


def ledoit_wolf_sharpe_test(
    r1: ArrayLike,
    r2: ArrayLike,
    method: Literal["hac", "bootstrap"] = "hac",
    *,
    rf: ArrayLike | None = None,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
    standardize: bool = False,
    block: int | Literal["calibrate"] | None = None,
    n_boot: int = 4999,
    alpha: float = 0.05,
    seed: SeedLike | None = None,
    ddof: int = 1,
    null_difference: float = 0.0,
    periods_per_year: float | None = None,
    calibration_kwargs: Mapping[str, Any] | None = None,
) -> SharpeTestResult:
    """Ledoit & Wolf (2008) test of H0: SR_1 - SR_2 = ``null_difference``.

    ``r1``, ``r2`` are aligned returns (excess returns, or pass ``rf``). The standard error
    s(Delta_hat) is the HAC estimate of LW Sec. 3.1 with ``kernel`` and Andrews (1991) bandwidth;
    ``prewhiten=True`` gives LW's "HAC_pw" (also the s(Delta_hat) of their bootstrap test,
    Sec. 3.2.2), ``prewhiten=False`` their "HAC". Defaults (QS, prewhitened) follow the paper.

    ``method="hac"``: normal p-value and interval Delta_hat +- z_{1-alpha/2} s.
    ``method="bootstrap"``: studentized circular block bootstrap with block ``block`` (int, or
    ``"calibrate"`` for LW Algorithm 3.1 via :func:`calibrate_block_size`) and ``n_boot``
    replicates; p-value by LW eq. (9) and the symmetric interval of eq. (7) with
    z* = the ceil((1-alpha)(M+1))-th smallest d*, so that p <= alpha iff the interval excludes the
    null. ``seed`` is mandatory for the bootstrap.
    """
    if method not in ("hac", "bootstrap"):
        raise ValueError(f"unknown method {method!r}; use 'hac' or 'bootstrap'")
    alpha = _check_alpha(alpha)
    r = _pair(r1, r2, rf)
    T = r.shape[0]
    mu, var = _moments(r)
    sr = _sharpe_from_moments(mu, var, T, ddof)
    diff = float(sr[0] - sr[1])
    se, info = _lw_standard_error(r, kernel, prewhiten, standardize)
    stat = _studentize(diff - null_difference, se)
    params: dict[str, Any] = {
        "kernel": kernel,
        "prewhiten": prewhiten,
        "prewhitened": info["prewhitened"],
        "standardize": standardize,
        "bandwidth": info["bandwidth"],
        "ddof": ddof,
        "null_difference": null_difference,
    }
    if method == "hac":
        pvalue = float(2.0 * sps.norm.sf(abs(stat)))
        half = float(sps.norm.ppf(1.0 - alpha / 2.0)) * se
        return _finish(sr, se, stat, pvalue, half, alpha, "lw_hac", T, periods_per_year, params)

    if seed is None:
        raise ValueError("the bootstrap test needs an explicit seed")
    if block is None:
        raise ValueError("the bootstrap test needs block=<int> or block='calibrate'")
    n_boot = _positive_int(n_boot, "n_boot")
    gen = _as_generator(seed)
    if isinstance(block, str):
        if block != "calibrate":
            raise ValueError("block must be an integer or 'calibrate'")
        kw = {
            "alpha": alpha,
            "kernel": kernel,
            "prewhiten": prewhiten,
            "standardize": standardize,
            "ddof": ddof,
            **dict(calibration_kwargs or {}),
        }
        cal = calibrate_block_size(r[:, 0], r[:, 1], seed=_child_generator(seed), **kw)
        block_used = cal.block
        params["calibration"] = cal
    else:
        block_used = _check_block(block, T)
    pvalue, _, dstar = _bootstrap_pvalue(
        r, diff, se, block_used, n_boot, gen, ddof, null_difference
    )
    k = math.ceil((1.0 - alpha) * (n_boot + 1) - 1e-9)
    zstar = float(np.sort(dstar)[k - 1]) if k <= n_boot else math.inf
    params.update(block=block_used, n_boot=n_boot, seed=_seed_record(seed), z_star=zstar)
    return _finish(
        sr, se, stat, pvalue, zstar * se, alpha, "lw_bootstrap", T, periods_per_year, params
    )


@dataclass(frozen=True)
class BlockCalibration:
    """Outcome of LW (2008) Algorithm 3.1: empirical rejection rate of the bootstrap test of
    H0: Delta = Delta_hat on pseudo-data, for each candidate block; ``block`` minimises
    |rate - alpha| (first candidate on ties). ``functional`` names the statistic whose test was
    calibrated: ``"sharpe"`` (the difference of Sharpe ratios, here) or ``"logvar"`` (the
    difference of log variances, :func:`cif.stats.rq3.calibrate_logvar_block`)."""

    block: int
    block_grid: tuple[int, ...]
    rejection_rates: np.ndarray
    alpha: float
    n_boot: int
    n_pseudo: int
    mean_block_resid: float
    burn_in: int
    seed: int | None
    functional: str = "sharpe"


def _var1_pseudo_chunks(
    r: np.ndarray, n_pseudo: int, mean_block_resid: float, burn_in: int, gen: np.random.Generator
) -> Iterator[np.ndarray]:
    """Pseudo-series of LW Algorithm 3.1 for the return pairs ``r`` of shape ``(T, 2)``, in chunks
    of shape ``(k, T, 2)``: a VAR(1) with intercept fitted by OLS, residuals resampled by the
    stationary bootstrap (mean block ``mean_block_resid``), started at the first observation with
    ``burn_in`` discarded steps. Shared by the Sharpe and the log-variance calibrations.

    The chunks are produced lazily on purpose: the indices of a chunk are drawn from ``gen`` when
    it is requested, after the caller has used ``gen`` for the bootstrap tests of the previous
    one, so the random stream of a calibration does not depend on how it is organised.
    """
    T = r.shape[0]
    X = np.column_stack([np.ones(T - 1), r[:-1]])
    coef = np.linalg.lstsq(X, r[1:], rcond=None)[0]  # (3, 2): intercepts, lag coefficients
    resid = r[1:] - X @ coef
    length = burn_in + T - 1
    for start in range(0, n_pseudo, _CHUNK):
        k = min(_CHUNK, n_pseudo - start)
        idx = stationary_bootstrap_indices(T - 1, mean_block_resid, k, gen, length=length)
        e = resid[idx]  # (k, length, 2)
        x = np.empty((k, burn_in + T, 2))
        x[:, 0] = r[0]
        for t in range(1, burn_in + T):
            x[:, t] = coef[0] + x[:, t - 1] @ coef[1:] + e[:, t - 1]
        yield x[:, burn_in:]


def calibrate_block_size(
    r1: ArrayLike,
    r2: ArrayLike,
    *,
    block_grid: tuple[int, ...] = (1, 2, 4, 6, 8, 10),
    alpha: float = 0.05,
    n_boot: int = 499,
    n_pseudo: int = 1000,
    mean_block_resid: float = 5.0,
    burn_in: int = 50,
    seed: SeedLike,
    rf: ArrayLike | None = None,
    kernel: Kernel = "qs",
    prewhiten: bool = True,
    standardize: bool = False,
    ddof: int = 1,
) -> BlockCalibration:
    """Block size for the LW bootstrap by calibration (LW 2008, Algorithm 3.1).

    Semi-parametric model as in LW Sec. 6 and the authors' code: a VAR(1) with intercept fitted
    by OLS to the return pairs; pseudo-series are generated with residuals resampled by the
    stationary bootstrap (mean block ``mean_block_resid``), started at the first observation with
    ``burn_in`` discarded steps. For each candidate block the bootstrap test of H0: Delta =
    Delta_hat (original data) is run with ``n_boot`` replicates on ``n_pseudo`` pseudo-series.
    LW use the grid {1, 2, 4, 6, 8, 10} for T = 120 monthly returns and K = 1000-5000; the grid
    should be widened for daily data. Inspect ``rejection_rates`` rather than trusting ``block``
    mechanically (authors' advice).
    """
    alpha = _check_alpha(alpha)
    r = _pair(r1, r2, rf)
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
    mu, var = _moments(r)
    sr = _sharpe_from_moments(mu, var, T, ddof)
    diff_hat = float(sr[0] - sr[1])
    rejections = np.zeros(len(grid), dtype=int)
    for pseudo in _var1_pseudo_chunks(r, n_pseudo, mean_block_resid, burn_in, gen):
        for ps in pseudo:
            mu_i, var_i = _moments(ps)
            sr_i = _sharpe_from_moments(mu_i, var_i, T, ddof)
            diff_i = float(sr_i[0] - sr_i[1])
            se_i, _ = _lw_standard_error(ps, kernel, prewhiten, standardize)
            for j, b in enumerate(grid):
                p, _, _ = _bootstrap_pvalue(ps, diff_i, se_i, b, n_boot, gen, ddof, diff_hat)
                rejections[j] += p <= alpha
    rates = rejections / n_pseudo
    best = grid[int(np.argmin(np.abs(rates - alpha)))]
    return BlockCalibration(
        block=best,
        block_grid=grid,
        rejection_rates=rates,
        alpha=alpha,
        n_boot=n_boot,
        n_pseudo=n_pseudo,
        mean_block_resid=float(mean_block_resid),
        burn_in=burn_in,
        seed=_seed_record(seed),
    )


# ---------------------------------------------------------------------------------------------
# PSR and DSR (Bailey & Lopez de Prado 2012, 2014)
# ---------------------------------------------------------------------------------------------


def _sr_inputs(
    r: ArrayLike | None,
    rf: ArrayLike | None,
    sr: float | None,
    n_obs: int | None,
    skew: float | None,
    kurtosis: float | None,
) -> tuple[float, int, float, float]:
    if r is not None:
        if any(v is not None for v in (sr, n_obs, skew, kurtosis)):
            raise ValueError("pass either returns r or summary statistics, not both")
        x = _excess(r, rf)
        if x.ndim != 1 or x.size < 3:
            raise ValueError("r must be a 1-D series with at least 3 observations")
        sd = x.std(ddof=1)
        if sd == 0.0:
            raise ValueError("zero variance; the Sharpe ratio is undefined")
        return (
            float(x.mean() / sd),
            int(x.size),
            float(sps.skew(x)),
            float(sps.kurtosis(x, fisher=False)),
        )
    if rf is not None:
        raise ValueError("rf is only used together with returns r")
    if any(v is None for v in (sr, n_obs, skew, kurtosis)):
        raise ValueError("without r, sr, n_obs, skew and kurtosis are all required")
    n = int(n_obs)  # type: ignore[arg-type]
    if n < 2:
        raise ValueError("n_obs must be at least 2")
    return float(sr), n, float(skew), float(kurtosis)  # type: ignore[arg-type]


def _psr(sr: float, sr_benchmark: float, n: int, skew: float, kurtosis: float) -> float:
    denom2 = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2
    if not denom2 > 0.0:
        raise ValueError("1 - skew*SR + (kurtosis-1)/4*SR^2 must be positive")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom2)
    return float(sps.norm.cdf(z))


def probabilistic_sharpe_ratio(
    r: ArrayLike | None = None,
    sr_benchmark: float = 0.0,
    *,
    rf: ArrayLike | None = None,
    sr: float | None = None,
    n_obs: int | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Probabilistic Sharpe ratio, Bailey & Lopez de Prado (2012), eq. (11).

    PSR(SR*) = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2)), all quantities per
    period (not annualised); g4 is the kurtosis (3 for the normal). From returns: SR = mean /
    std(ddof=1) of excess returns, g3 and g4 are the moment (biased) estimators.
    """
    sr_, n, g3, g4 = _sr_inputs(r, rf, sr, n_obs, skew, kurtosis)
    return _psr(sr_, float(sr_benchmark), n, g3, g4)


def expected_max_sharpe(sr_variance: float, n_trials: int, sr_mean: float = 0.0) -> float:
    """Expected maximum of ``n_trials`` independent SR estimates, Bailey & Lopez de Prado (2014),
    Appendix A.1, eq. (6): mean + sqrt(V) ((1-g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))),
    g = Euler-Mascheroni constant. ``sr_variance`` is the per-period variance across trials
    (annual variance / periods per year)."""
    if isinstance(n_trials, bool) or int(n_trials) != n_trials or n_trials < 2:
        raise ValueError("n_trials must be an integer >= 2 (the approximation needs N >> 1)")
    if not sr_variance >= 0.0:
        raise ValueError("sr_variance must be non-negative")
    g = float(np.euler_gamma)
    n = float(n_trials)
    z = (1.0 - g) * sps.norm.ppf(1.0 - 1.0 / n) + g * sps.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(sr_mean + math.sqrt(sr_variance) * z)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """DSR = PSR(SR_0) with SR_0 = expected maximum SR under zero skill (per-period units)."""

    dsr: float
    sr0: float
    sr: float
    n_obs: int
    skew: float
    kurtosis: float
    n_trials: int
    sr_variance: float


def deflated_sharpe_ratio(
    r: ArrayLike | None = None,
    *,
    sr_variance: float,
    n_trials: int,
    rf: ArrayLike | None = None,
    sr: float | None = None,
    n_obs: int | None = None,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio, Bailey & Lopez de Prado (2014), eq. (2).

    ``n_trials`` must count every configuration tried (including exploratory ones), and
    ``sr_variance`` is the per-period variance of the SR estimates across those trials.
    """
    sr_, n, g3, g4 = _sr_inputs(r, rf, sr, n_obs, skew, kurtosis)
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    return DeflatedSharpeResult(
        dsr=_psr(sr_, sr0, n, g3, g4),
        sr0=sr0,
        sr=sr_,
        n_obs=n,
        skew=g3,
        kurtosis=g4,
        n_trials=int(n_trials),
        sr_variance=float(sr_variance),
    )
