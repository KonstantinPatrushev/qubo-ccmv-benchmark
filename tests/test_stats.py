"""Tests for ``cif.stats`` (stream E0, WP-C; audit item R7).

Reference numbers below come only from published sources or from independent computations
inside this file. Nothing is "expected" without a source:

* Ledoit & Wolf (2008), J. Empirical Finance 15:850-859, Table 2 (summary statistics) and
  Table 3 (p-values, in percent) for the two data sets shipped with the authors' code
  (``retSharpe.mat`` in ``SharpeMatlab.zip`` from econ.uzh.ch). The file is downloaded at test
  time and cached in pytest's cache directory; it is not redistributed with this repository.
  Set ``CIF_LW2008_MAT=/path/to/retSharpe.mat`` to use a local copy.
* Patton, Politis & White (2009), Econometric Reviews 28:372-375 (correction to Politis &
  White 2004): Table 1 (theoretical optimal block sizes for AR(1)) and Tables 2-3 (Monte Carlo
  mean of b_hat / b_opt).
* Bailey & Lopez de Prado (2012), J. Risk 15(2), Sec. 3: PSR example.
* Bailey & Lopez de Prado (2014), J. Portfolio Management 40(5): DSR numerical example.

Simulation tests (size, power, coverage) use fixed seeds; their acceptance corridors were fixed
before the first run and are justified in the comments next to them.
"""

from __future__ import annotations

import io
import math
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy import stats as sps

from cif.stats import (
    bootstrap_ci,
    calibrate_block_size,
    circular_block_bootstrap_indices,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    hac_covariance,
    holm,
    jobson_korkie_memmel,
    ledoit_wolf_sharpe_test,
    optimal_block_length,
    performance_profile,
    performance_ratios,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    stationary_bootstrap_indices,
    tost_bootstrap,
    tost_from_ci,
    ttt_ecdf,
)
from cif.stats.bootstrap import _pw_block_lengths
from cif.stats.sharpe import _bootstrap_statistics

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _ar1(n: int, phi: float, rng: np.random.Generator, burn: int = 500) -> np.ndarray:
    z = rng.standard_normal(n + burn)
    return signal.lfilter([1.0], [1.0, -phi], z)[burn:]


def _return_pair(
    T: int,
    rng: np.random.Generator,
    *,
    sharpe1: float,
    sharpe2: float,
    phi: float = 0.0,
    rho: float = 0.5,
    burn: int = 200,
) -> np.ndarray:
    """Two return series with correlated Gaussian innovations and a common AR(1) coefficient.

    Both series have the same law up to the mean, so ``sharpe1 == sharpe2`` is the null of equal
    Sharpe ratios. Returns a (T, 2) array.
    """
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    e = rng.standard_normal((T + burn, 2)) @ chol.T
    x = signal.lfilter([1.0], [1.0, -phi], e, axis=0)[burn:]
    sd = 1.0 / math.sqrt(1.0 - phi**2)
    return x + sd * np.array([sharpe1, sharpe2])


def _mean_last_axis(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=-1)


# ---------------------------------------------------------------------------------------------
# bootstrap indices
# ---------------------------------------------------------------------------------------------


def test_stationary_indices_shape_range_and_dtype():
    idx = stationary_bootstrap_indices(50, 4.0, 7, 123)
    assert idx.shape == (7, 50)
    assert np.issubdtype(idx.dtype, np.integer)
    assert idx.min() >= 0 and idx.max() < 50


def test_stationary_indices_mean_block_and_wraparound():
    n, mean_block, n_boot = 200, 8.0, 400
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, np.random.default_rng(1))
    cont = idx[:, 1:] == (idx[:, :-1] + 1) % n
    # A new block starts with probability p = 1/mean_block (Politis & Romano 1994); a fresh
    # start coincides with "previous + 1" with probability 1/n, so P(cont) = (1 - p)(1 + 1/n)...
    # to first order 1 - p. Over 400 x 199 transitions the binomial s.e. is ~0.0012.
    p_hat = 1.0 - cont.mean()
    assert abs(p_hat - 1.0 / mean_block) < 0.01
    # mean length of completed blocks (exclude the truncated last block of each row)
    lengths = []
    for row in cont:
        starts = np.flatnonzero(~row) + 1  # positions (>=1) where a new block starts
        bounds = np.concatenate(([0], starts))
        lengths.extend(np.diff(bounds).tolist())
    assert abs(np.mean(lengths) - mean_block) < 0.4
    # circular wrap-around: some continuation goes n-1 -> 0
    wraps = (idx[:, :-1] == n - 1) & (idx[:, 1:] == 0)
    assert wraps.any()


def test_stationary_indices_mean_block_one_is_iid():
    n = 100
    idx = stationary_bootstrap_indices(n, 1.0, 300, 7)
    cont = idx[:, 1:] == (idx[:, :-1] + 1) % n
    # p = 1: every position is a fresh uniform draw; continuations only by chance (1/n).
    assert cont.mean() < 0.03


def test_stationary_indices_custom_length():
    idx = stationary_bootstrap_indices(30, 5.0, 4, 3, length=95)
    assert idx.shape == (4, 95)
    assert idx.max() < 30


def test_circular_indices_block_structure_and_wrap():
    n, b = 23, 5  # n not a multiple of b: last block is truncated
    idx = circular_block_bootstrap_indices(n, b, 500, 11)
    assert idx.shape == (500, n)
    t = np.arange(n - 1)
    within = (t % b) != (b - 1)
    step_ok = idx[:, 1:] == (idx[:, :-1] + 1) % n
    assert step_ok[:, within].all()
    assert ((idx[:, :-1] == n - 1) & (idx[:, 1:] == 0)).any()
    # block starts are uniform on {0, ..., n-1}
    starts = idx[:, ::b].ravel()
    counts = np.bincount(starts, minlength=n)
    assert sps.chisquare(counts).pvalue > 1e-3


def test_indices_are_deterministic_given_seed():
    a = stationary_bootstrap_indices(40, 3.0, 5, 99)
    b = stationary_bootstrap_indices(40, 3.0, 5, 99)
    c = stationary_bootstrap_indices(40, 3.0, 5, 100)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    d = circular_block_bootstrap_indices(40, 4, 5, np.random.default_rng(5))
    e = circular_block_bootstrap_indices(40, 4, 5, np.random.default_rng(5))
    np.testing.assert_array_equal(d, e)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (stationary_bootstrap_indices, (10, 0.5, 3, 1)),
        (stationary_bootstrap_indices, (0, 2.0, 3, 1)),
        (circular_block_bootstrap_indices, (10, 0, 3, 1)),
        (circular_block_bootstrap_indices, (10, 11, 3, 1)),
    ],
)
def test_indices_reject_invalid_arguments(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


# ---------------------------------------------------------------------------------------------
# optimal block length (Politis & White 2004; Patton, Politis & White 2009)
# ---------------------------------------------------------------------------------------------

# PPW (2009), Table 1: theoretical b_opt for AR(1) with N(0,1) innovations.
# (rho, N): (b_opt_SB, b_opt_CB)
_PPW_TABLE1 = {
    (0.7, 200): (11.47, 13.12),
    (0.7, 800): (18.20, 20.83),
    (0.1, 200): (2.01, 2.31),
    (0.1, 800): (3.20, 3.66),
    (-0.4, 200): (5.66, 6.48),
    (-0.4, 800): (8.99, 10.23),
}
# The CB column must equal (3/2)^(1/3) times the SB column exactly (D_CB = 4/3 g^2 vs
# D_SB = 2 g^2). Two published CB entries are inconsistent with that identity beyond rounding:
# (0.1, 200): 2.01 * 1.1447 = 2.30 (published 2.31); (-0.4, 800): 8.99 * 1.1447 = 10.29
# (published 10.23). They are checked only through the SB column.
_PPW_CB_INCONSISTENT = {(0.1, 200), (-0.4, 800)}


@pytest.mark.parametrize(("key", "expected"), sorted(_PPW_TABLE1.items()))
def test_pw_block_formula_reproduces_ppw2009_table1(key, expected):
    rho, n = key
    g0 = 1.0 / (1.0 - rho) ** 2  # sum_k R(k) for AR(1) with unit innovation variance
    G = 2.0 * rho / ((1.0 - rho) ** 2 * (1.0 - rho**2))  # sum_k |k| R(k)
    b_sb, b_cb = _pw_block_lengths(G, g0, n)
    assert abs(b_sb - expected[0]) <= 0.005 + 1e-12
    assert b_cb / b_sb == pytest.approx(1.5 ** (1.0 / 3.0), rel=1e-12)
    if key not in _PPW_CB_INCONSISTENT:
        assert abs(b_cb - expected[1]) <= 0.005 + 1e-12


def test_pw_estimator_matches_ppw2009_monte_carlo():
    # PPW (2009) Tables 2-3, rho = 0.7, N = 800: mean of b_hat/b_opt is 0.927 (SB) and 0.951 (CB),
    # st.dev. 0.244 in both. The CB entries were evidently computed after rounding b_hat_CB up to
    # an integer, as Politis & White recommend: since b_hat_CB / b_hat_SB = (3/2)^(1/3) exactly,
    # the raw ratios would have identical means; ceil() adds ~0.5/20.83 = 0.024 = 0.951 - 0.927.
    # Tolerance (fixed in advance): 4 Monte Carlo s.e. of the published mean (0.244/sqrt(R))
    # plus 0.02 for implementation details (full-sample vs common-sample autocovariances).
    rng = np.random.default_rng(20090101)
    R, n, rho = 500, 800, 0.7
    b_opt_sb, b_opt_cb = _PPW_TABLE1[(rho, n)]
    r_sb, r_cb = [], []
    for _ in range(R):
        x = _ar1(n, rho, rng)
        est = optimal_block_length(x)
        r_sb.append(est.stationary / b_opt_sb)
        r_cb.append(math.ceil(est.circular) / b_opt_cb)
    tol = 4 * 0.244 / math.sqrt(R) + 0.02
    print(
        f"[stats-report] PPW2009 MC rho=0.7 N=800 R={R}: mean b_SB/b_opt={np.mean(r_sb):.3f} (pub 0.927), mean ceil(b_CB)/b_opt={np.mean(r_cb):.3f} (pub 0.951)"
    )
    assert abs(np.mean(r_sb) - 0.927) <= tol
    assert abs(np.mean(r_cb) - 0.951) <= tol


def test_pw_block_is_larger_for_ar1_than_for_iid():
    rng = np.random.default_rng(42)
    n = 1000
    b_ar = optimal_block_length(_ar1(n, 0.6, rng))
    b_iid = optimal_block_length(rng.standard_normal(n))
    # theoretical SB block for phi = 0.6, n = 1000 is (G/g)^(2/3) n^(1/3) = 15.2; for iid it is ~1
    assert b_ar.stationary > 4.0
    assert b_ar.stationary > 2.0 * b_iid.stationary
    assert b_ar.circular > b_ar.stationary
    assert b_iid.stationary >= 1.0  # floored at one observation


def test_pw_two_dimensional_input_is_columnwise():
    rng = np.random.default_rng(3)
    x = np.column_stack([_ar1(400, 0.5, rng), rng.standard_normal(400)])
    est = optimal_block_length(x)
    for j in range(2):
        single = optimal_block_length(x[:, j])
        assert est.stationary[j] == pytest.approx(single.stationary)
        assert est.circular[j] == pytest.approx(single.circular)


# ---------------------------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_bootstrap_ci_coverage_of_the_mean_iid_normal():
    # 300 replications of n = 200 iid N(1, 1). For the percentile interval of the mean the
    # bootstrap variance is (n-1)/n s^2/n, so the expected coverage is ~0.947. The corridor
    # [0.90, 0.985] is about +-3.5 binomial s.e. (0.013) around it.
    n_rep, n = 300, 200
    rng = np.random.default_rng(2026)
    hits = {"iid": 0, "stationary": 0, "basic_iid": 0}
    for k in range(n_rep):
        x = 1.0 + rng.standard_normal(n)
        res = bootstrap_ci(_mean_last_axis, x, method="iid", n_boot=999, seed=k, vectorized=True)
        hits["iid"] += res.ci_percentile[0] <= 1.0 <= res.ci_percentile[1]
        hits["basic_iid"] += res.ci_basic[0] <= 1.0 <= res.ci_basic[1]
        res = bootstrap_ci(
            _mean_last_axis, x, method="stationary", block=5.0, n_boot=999, seed=k, vectorized=True
        )
        hits["stationary"] += res.ci_percentile[0] <= 1.0 <= res.ci_percentile[1]
    print(
        f"[stats-report] coverage of 95% CI, mean, iid N(1,1), n=200, R={n_rep}: "
        + ", ".join(f"{k}={v / n_rep:.3f}" for k, v in hits.items())
    )
    for name, h in hits.items():
        assert 0.90 <= h / n_rep <= 0.985, (name, h / n_rep)


@pytest.mark.slow
def test_block_bootstrap_beats_iid_bootstrap_on_ar1_data():
    # AR(1), phi = 0.5, n = 1000: the long-run variance of the mean is 3x the marginal one, so
    # the iid bootstrap covers ~ P(|Z| < 1.96/sqrt(3)) = 0.74. With the Politis-White block
    # (b ~ 12) the block bootstraps should cover ~0.92-0.94. Corridors fixed in advance:
    # block methods >= 0.86, iid <= 0.82 (200 replications; binomial s.e. <= 0.031).
    n_rep, n = 200, 1000
    rng = np.random.default_rng(77)
    hits = {"stationary": 0, "circular": 0, "iid": 0}
    for k in range(n_rep):
        x = _ar1(n, 0.5, rng)
        for method in hits:
            res = bootstrap_ci(
                _mean_last_axis, x, method=method, n_boot=499, seed=k, vectorized=True
            )
            hits[method] += res.ci_percentile[0] <= 0.0 <= res.ci_percentile[1]
    print(
        f"[stats-report] coverage of 95% CI, mean, AR(1) phi=0.5, n=1000, R={n_rep}: "
        + ", ".join(f"{k}={v / n_rep:.3f}" for k, v in hits.items())
    )
    assert hits["stationary"] / n_rep >= 0.86
    assert hits["circular"] / n_rep >= 0.86
    assert hits["iid"] / n_rep <= 0.82


def test_bootstrap_ci_resamples_paired_arrays_with_common_indices():
    x = np.random.default_rng(0).standard_normal(150)
    res = bootstrap_ci(lambda a, b: float(np.mean(a - b)), x, x.copy(), n_boot=200, seed=1)
    np.testing.assert_array_equal(res.samples, 0.0)
    assert res.ci_percentile == (0.0, 0.0)


def test_bootstrap_ci_vectorized_and_loop_paths_agree():
    rng = np.random.default_rng(10)
    x = rng.standard_normal(120)
    y = rng.standard_normal(120)

    def stat(a, b):
        return float(np.mean(a) - np.median(b))

    def stat_vec(a, b):
        return a.mean(axis=-1) - np.median(b, axis=-1)

    for method in ("stationary", "circular", "iid"):
        kw = {"block": 4} if method == "circular" else {}
        r1 = bootstrap_ci(stat, x, y, method=method, n_boot=300, seed=5, **kw)
        r2 = bootstrap_ci(stat_vec, x, y, method=method, n_boot=300, seed=5, vectorized=True, **kw)
        np.testing.assert_allclose(r1.samples, r2.samples, rtol=0, atol=1e-12)
        assert r1.estimate == pytest.approx(r2.estimate)


def test_bootstrap_ci_is_deterministic_and_basic_reflects_percentile():
    x = np.random.default_rng(4).standard_normal(80)
    r1 = bootstrap_ci(lambda a: float(np.mean(a)), x, n_boot=500, seed=17, block=3.0)
    r2 = bootstrap_ci(lambda a: float(np.mean(a)), x, n_boot=500, seed=17, block=3.0)
    np.testing.assert_array_equal(r1.samples, r2.samples)
    assert r1.ci_basic[0] == pytest.approx(2 * r1.estimate - r1.ci_percentile[1])
    assert r1.ci_basic[1] == pytest.approx(2 * r1.estimate - r1.ci_percentile[0])
    assert r1.se == pytest.approx(np.std(r1.samples, ddof=1))
    assert r1.n_boot == 500 and r1.method == "stationary" and r1.block == 3.0 and r1.seed == 17


def test_bootstrap_ci_two_dimensional_array_and_automatic_block():
    rng = np.random.default_rng(8)
    xy = np.column_stack([_ar1(300, 0.7, rng), rng.standard_normal(300)])
    res = bootstrap_ci(
        lambda a: float(a[:, 0].mean() - a[:, 1].mean()), xy, method="circular", n_boot=200, seed=2
    )
    expected = math.ceil(optimal_block_length(xy).circular.max())
    assert res.block == expected


def test_bootstrap_ci_vector_valued_statistic():
    x = np.random.default_rng(12).standard_normal(90)

    def stat(a):
        return np.array([a.mean(), np.median(a)])

    res = bootstrap_ci(stat, x, method="iid", n_boot=400, seed=3, alpha=0.1)
    assert res.samples.shape == (400, 2)
    np.testing.assert_allclose(res.estimate, stat(x))
    lo, hi = res.ci_percentile
    np.testing.assert_allclose(lo, np.quantile(res.samples, 0.05, axis=0))
    np.testing.assert_allclose(hi, np.quantile(res.samples, 0.95, axis=0))
    np.testing.assert_allclose(res.ci_basic[0], 2 * res.estimate - hi)


def test_bootstrap_ci_rejects_bad_input():
    x = np.arange(10.0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.mean, x, np.arange(9.0), seed=1)
    with pytest.raises(ValueError):
        bootstrap_ci(np.mean, x, method="moving", seed=1)
    with pytest.raises(ValueError):
        bootstrap_ci(np.mean, x, method="iid", block=5, seed=1)
    with pytest.raises(TypeError):
        bootstrap_ci(np.mean, x)  # seed is mandatory


# ---------------------------------------------------------------------------------------------
# Sharpe ratio, JKM, Ledoit-Wolf
# ---------------------------------------------------------------------------------------------


def test_sharpe_ratio_by_hand():
    r = np.array([0.01, -0.02, 0.03, 0.005, 0.0, 0.012])
    rf = 0.001
    ex = r - rf
    expected = ex.mean() / ex.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(r, rf) == pytest.approx(expected)
    rf_vec = np.linspace(0.0, 0.002, r.size)
    ex2 = r - rf_vec
    assert sharpe_ratio(r, rf_vec, periods_per_year=12) == pytest.approx(
        ex2.mean() / ex2.std(ddof=1) * math.sqrt(12)
    )
    assert sharpe_ratio(r, periods_per_year=None) == pytest.approx(r.mean() / r.std(ddof=1))
    two = sharpe_ratio(np.column_stack([r, 2 * r + 0.001]), periods_per_year=None)
    assert two.shape == (2,)
    assert two[0] == pytest.approx(r.mean() / r.std(ddof=1))


def test_jkm_standard_error_equals_delta_method_with_iid_normal_omega():
    # LW (2008), Sec. 2: Memmel (2003) applies the delta method to u = (mu_i, mu_n, s2_i, s2_n)
    # with the iid-normal covariance Omega. Computed here independently in matrix form.
    r = _return_pair(400, np.random.default_rng(12), sharpe1=0.15, sharpe2=0.05, rho=0.6)
    res = jobson_korkie_memmel(r[:, 0], r[:, 1])
    T = len(r)
    mu = r.mean(axis=0)
    S = np.cov(r.T, ddof=0)
    s1, s2, s12 = S[0, 0], S[1, 1], S[0, 1]
    omega = np.array(
        [
            [s1, s12, 0, 0],
            [s12, s2, 0, 0],
            [0, 0, 2 * s1**2, 2 * s12**2],
            [0, 0, 2 * s12**2, 2 * s2**2],
        ]
    )
    grad = np.array(
        [1 / math.sqrt(s1), -1 / math.sqrt(s2), -mu[0] / (2 * s1**1.5), mu[1] / (2 * s2**1.5)]
    )
    se = math.sqrt(grad @ omega @ grad / T)
    diff = mu[0] / math.sqrt(s1) - mu[1] / math.sqrt(s2)
    assert res.se == pytest.approx(se, rel=1e-10)
    assert res.difference == pytest.approx(diff, rel=1e-10)
    assert res.statistic == pytest.approx(diff / se, rel=1e-10)
    assert res.pvalue == pytest.approx(2 * sps.norm.sf(abs(diff / se)), rel=1e-10)
    assert res.method == "jkm"


def _lw_psi_reference(v: np.ndarray, kernel: str) -> np.ndarray:
    """Literal transcription of the HAC estimator in LW (2008), Sec. 3.1, with Andrews (1991)
    AR(1) plug-in bandwidth (unit weights), written with explicit loops."""
    T, p = v.shape
    num = den = 0.0
    for a in range(p):
        y = v[:, a]
        X = np.column_stack([np.ones(T - 1), y[:-1]])
        beta = np.linalg.solve(X.T @ X, X.T @ y[1:])
        e = y[1:] - X @ beta
        rho, s2 = beta[1], (e @ e) / (T - 1)
        num += 4 * rho**2 * s2**2 / (1 - rho) ** 8
        den += s2**2 / (1 - rho) ** 4
    alpha2 = num / den
    const = 2.6614 if kernel == "parzen" else 1.3221
    S = min(const * (alpha2 * T) ** 0.2, T - 1)

    def k(x: float) -> float:
        x = abs(x)
        if kernel == "parzen":
            if x <= 0.5:
                return 1 - 6 * x**2 + 6 * x**3
            return 2 * (1 - x) ** 3 if x <= 1 else 0.0
        if x == 0:
            return 1.0
        z = 6 * math.pi * x / 5
        return 25 / (12 * math.pi**2 * x**2) * (math.sin(z) / z - math.cos(z))

    psi = np.zeros((p, p))
    for j in range(-(T - 1), T):
        w = k(j / S)
        if w == 0.0:
            continue
        G = np.zeros((p, p))
        if j >= 0:
            for t in range(j, T):
                G += np.outer(v[t], v[t - j])
        else:
            for t in range(-j, T):
                G += np.outer(v[t + j], v[t])
        psi += w * G / T
    return T / (T - 4) * psi


def _lw_se_reference(r: np.ndarray, kernel: str, prewhiten: bool) -> float:
    T = len(r)
    mu = r.mean(axis=0)
    gam = (r**2).mean(axis=0)
    var = gam - mu**2
    grad = np.array(
        [
            gam[0] / var[0] ** 1.5,
            -gam[1] / var[1] ** 1.5,
            -0.5 * mu[0] / var[0] ** 1.5,
            0.5 * mu[1] / var[1] ** 1.5,
        ]
    )
    v = np.column_stack([r - mu, r**2 - gam])
    if not prewhiten:
        psi = _lw_psi_reference(v, kernel)
    else:  # Andrews & Monahan (1992) as in the authors' code: VAR(1), singular values <= 0.97
        X, Y = v[:-1], v[1:]
        A = np.linalg.solve(X.T @ X, X.T @ Y).T
        U, s, Vt = np.linalg.svd(A)
        A = U @ np.diag(np.minimum(s, 0.97)) @ Vt
        psi_e = _lw_psi_reference(Y - X @ A.T, kernel)
        D = np.linalg.inv(np.eye(4) - A)
        psi = D @ psi_e @ D.T
    return math.sqrt(grad @ psi @ grad / T)


@pytest.mark.parametrize("kernel", ["qs", "parzen"])
@pytest.mark.parametrize("prewhiten", [False, True])
def test_lw_hac_standard_error_matches_literal_transcription(kernel, prewhiten):
    r = _return_pair(150, np.random.default_rng(31), sharpe1=0.2, sharpe2=0.1, phi=0.3)
    r = r * np.array([1.5, 0.8])
    res = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], kernel=kernel, prewhiten=prewhiten)
    assert res.se == pytest.approx(_lw_se_reference(r, kernel, prewhiten), rel=1e-9)
    sd = r.std(axis=0, ddof=1)
    diff = r[:, 0].mean() / sd[0] - r[:, 1].mean() / sd[1]
    assert res.difference == pytest.approx(diff, rel=1e-12)  # sample s.d. (ddof=1), as LW code
    assert res.pvalue == pytest.approx(2 * sps.norm.sf(abs(diff) / res.se), rel=1e-9)


def test_hac_covariance_returns_bandwidth_and_matches_reference():
    rng = np.random.default_rng(5)
    v = np.column_stack([_ar1(120, 0.4, rng), _ar1(120, -0.2, rng)])
    v = v - v.mean(axis=0)
    psi, info = hac_covariance(v, kernel="qs", df_adjust=4)
    np.testing.assert_allclose(psi, _lw_psi_reference(v, "qs"), rtol=1e-9, atol=1e-12)
    assert info["bandwidth"] > 0 and info["kernel"] == "qs" and not info["prewhitened"]


def test_lw_identical_series_give_zero_difference_and_unit_pvalue():
    r = _return_pair(300, np.random.default_rng(2), sharpe1=0.1, sharpe2=0.1)[:, 0]
    hac = ledoit_wolf_sharpe_test(r, r.copy())
    assert hac.difference == 0.0
    assert hac.pvalue == pytest.approx(1.0)
    boot = ledoit_wolf_sharpe_test(r, r.copy(), method="bootstrap", block=5, n_boot=199, seed=1)
    assert boot.difference == 0.0
    assert boot.pvalue == 1.0


def test_lw_is_antisymmetric_in_the_two_series():
    r = _return_pair(250, np.random.default_rng(9), sharpe1=0.2, sharpe2=0.05, phi=0.1)
    a = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1])
    b = ledoit_wolf_sharpe_test(r[:, 1], r[:, 0])
    assert b.difference == pytest.approx(-a.difference)
    assert b.se == pytest.approx(a.se, rel=1e-10)
    assert b.pvalue == pytest.approx(a.pvalue, rel=1e-10)
    c = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], method="bootstrap", block=4, n_boot=299, seed=3)
    d = ledoit_wolf_sharpe_test(r[:, 1], r[:, 0], method="bootstrap", block=4, n_boot=299, seed=3)
    assert d.pvalue == pytest.approx(c.pvalue)


def test_lw_standardized_bandwidth_is_unit_free():
    # With unit weights (LW default) the Andrews bandwidth depends on the units of returns;
    # standardize=True removes that dependence. Rescaling both series leaves Delta unchanged.
    r = _return_pair(300, np.random.default_rng(21), sharpe1=0.15, sharpe2=0.0, phi=0.2)
    a = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], standardize=True)
    b = ledoit_wolf_sharpe_test(100 * r[:, 0], 100 * r[:, 1], standardize=True)
    assert b.difference == pytest.approx(a.difference, rel=1e-10)
    assert b.se == pytest.approx(a.se, rel=1e-8)


def test_lw_hac_se_close_to_jkm_on_iid_normal_data():
    # Under iid normality the delta-method variance with the true Psi equals the JKM variance;
    # HAC estimates it nonparametrically. Expected deviation ~ sqrt(S/T) ~ 4% at T = 2000.
    rng = np.random.default_rng(55)
    ratios = []
    for _ in range(10):
        r = _return_pair(2000, rng, sharpe1=0.3, sharpe2=0.2, rho=0.5)
        hac = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1])
        jkm = jobson_korkie_memmel(r[:, 0], r[:, 1])
        ratios.append(hac.se / jkm.se)
    ratios = np.asarray(ratios)
    assert np.all(np.abs(ratios - 1.0) < 0.15), ratios
    print(
        f"[stats-report] SE_HAC/SE_JKM on iid normal, T=2000: median={np.median(ratios):.3f}, range=[{ratios.min():.3f}, {ratios.max():.3f}]"
    )
    assert abs(np.median(ratios) - 1.0) < 0.05


@pytest.mark.slow
def test_lw_hac_size_under_the_null():
    # Two correlated (rho = 0.5) AR(1) (phi = 0.2) Gaussian series with equal Sharpe ratio 0.3
    # per period, T = 500 (cf. LW 2008 "Normal-VAR": HAC_pw 6.1% at T = 120). 400 replications;
    # corridor [2%, 9%] is about +-3 binomial s.e. around 5.5%.
    rng = np.random.default_rng(1234)
    n_rep = 400
    rej_pw = rej_plain = 0
    for _ in range(n_rep):
        r = _return_pair(500, rng, sharpe1=0.3, sharpe2=0.3, phi=0.2)
        rej_pw += ledoit_wolf_sharpe_test(r[:, 0], r[:, 1]).pvalue <= 0.05
        rej_plain += ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], prewhiten=False).pvalue <= 0.05
    print(
        f"[stats-report] size at 5%, T=500, R={n_rep}: HAC_pw={rej_pw / n_rep:.4f}, HAC={rej_plain / n_rep:.4f}"
    )
    assert 0.02 <= rej_pw / n_rep <= 0.09, rej_pw / n_rep
    assert 0.02 <= rej_plain / n_rep <= 0.10, rej_plain / n_rep


@pytest.mark.slow
def test_lw_bootstrap_size_under_the_null():
    # Same DGP, studentized circular block bootstrap with b = 5, M = 499; 200 replications.
    # Corridor [1%, 11%] is about +-3 binomial s.e. (0.016) around 5-6%.
    rng = np.random.default_rng(4321)
    n_rep = 200
    rej = 0
    for k in range(n_rep):
        r = _return_pair(500, rng, sharpe1=0.3, sharpe2=0.3, phi=0.2)
        res = ledoit_wolf_sharpe_test(
            r[:, 0], r[:, 1], method="bootstrap", block=5, n_boot=499, seed=k
        )
        rej += res.pvalue <= 0.05
    print(f"[stats-report] size at 5%, T=500, R={n_rep}: bootstrap (b=5, M=499)={rej / n_rep:.4f}")
    assert 0.01 <= rej / n_rep <= 0.11, rej / n_rep


@pytest.mark.slow
def test_lw_power_against_a_large_difference():
    # Sharpe 0.4 vs 0.1 per period, phi = 0.2, rho = 0.5, T = 500: the delta-method s.e. is
    # ~0.056, so |z| ~ 5.3 and power is essentially one. Thresholds fixed in advance.
    rng = np.random.default_rng(99)
    rej_hac = 0
    for _ in range(100):
        r = _return_pair(500, rng, sharpe1=0.4, sharpe2=0.1, phi=0.2)
        rej_hac += ledoit_wolf_sharpe_test(r[:, 0], r[:, 1]).pvalue <= 0.05
    rej_boot = 0
    for k in range(50):
        r = _return_pair(500, rng, sharpe1=0.4, sharpe2=0.1, phi=0.2)
        res = ledoit_wolf_sharpe_test(
            r[:, 0], r[:, 1], method="bootstrap", block=5, n_boot=499, seed=k
        )
        rej_boot += res.pvalue <= 0.05
    print(
        f"[stats-report] power at 5%, SR 0.4 vs 0.1, T=500: HAC_pw={rej_hac / 100:.3f} (R=100), bootstrap={rej_boot / 50:.3f} (R=50)"
    )
    assert rej_hac / 100 >= 0.95
    assert rej_boot / 50 >= 0.90


def test_lw_bootstrap_pvalue_formula_and_interval_consistency():
    rng = np.random.default_rng(7)
    for k in range(6):
        r = _return_pair(240, rng, sharpe1=0.25, sharpe2=0.05 * k, phi=0.1)
        res = ledoit_wolf_sharpe_test(
            r[:, 0], r[:, 1], method="bootstrap", block=6, n_boot=399, seed=k, alpha=0.05
        )
        # LW (2008), eq. (9): PV = (#{d* >= d} + 1) / (M + 1)
        count = res.pvalue * 400 - 1
        assert count == pytest.approx(round(count), abs=1e-9)
        # inverting the symmetric studentized interval, eq. (7), gives the same decision
        excludes_zero = not (res.ci_low <= 0.0 <= res.ci_high)
        assert excludes_zero == (res.pvalue <= 0.05)
        assert res.params["block"] == 6 and res.params["n_boot"] == 399


def test_lw_bootstrap_statistics_match_literal_definition():
    # Psi* = (1/l) sum_j zeta_j zeta_j', zeta_j = b^{-1/2} sum of the j-th block of y*_t,
    # y*_t centred at bootstrap moments (LW 2008, Sec. 3.2.2); s(Delta*) by eq. (8).
    rng = np.random.default_rng(14)
    T, b = 103, 5  # T not a multiple of b: only the first l*b = 100 observations enter Psi*
    samples = rng.standard_normal((3, T, 2)) + np.array([0.2, 0.1])
    diff_hat = 0.05
    dstar, se_star, diff_star = _bootstrap_statistics(samples, diff_hat, b, ddof=1)
    for m in range(3):
        x = samples[m]
        mu = x.mean(axis=0)
        gam = (x**2).mean(axis=0)
        var = gam - mu**2
        grad = np.array(
            [
                gam[0] / var[0] ** 1.5,
                -gam[1] / var[1] ** 1.5,
                -0.5 * mu[0] / var[0] ** 1.5,
                0.5 * mu[1] / var[1] ** 1.5,
            ]
        )
        y = np.column_stack([x - mu, x**2 - gam])
        n_blocks = T // b
        psi = np.zeros((4, 4))
        for j in range(n_blocks):
            zeta = y[j * b : (j + 1) * b].sum(axis=0) / math.sqrt(b)
            psi += np.outer(zeta, zeta)
        psi /= n_blocks
        se = math.sqrt(grad @ psi @ grad / T)
        sd = x.std(axis=0, ddof=1)
        d = mu[0] / sd[0] - mu[1] / sd[1]
        assert se_star[m] == pytest.approx(se, rel=1e-10)
        assert diff_star[m] == pytest.approx(d, rel=1e-12)
        assert dstar[m] == pytest.approx(abs(d - diff_hat) / se, rel=1e-10)


def test_lw_bootstrap_requires_block_and_seed():
    r = _return_pair(100, np.random.default_rng(0), sharpe1=0.1, sharpe2=0.1)
    with pytest.raises(ValueError):
        ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], method="bootstrap", seed=1)
    with pytest.raises(ValueError):
        ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], method="bootstrap", block=4)
    with pytest.raises(ValueError):
        ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], method="studentized")


def test_lw_annualization_scales_estimates_but_not_the_test():
    r = _return_pair(300, np.random.default_rng(6), sharpe1=0.1, sharpe2=0.02, phi=0.1)
    a = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], alpha=0.10)
    b = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], alpha=0.10, periods_per_year=252)
    f = math.sqrt(252)
    assert b.difference == pytest.approx(f * a.difference)
    assert b.se == pytest.approx(f * a.se)
    assert (b.ci_low, b.ci_high) == pytest.approx((f * a.ci_low, f * a.ci_high))
    assert b.pvalue == pytest.approx(a.pvalue) and b.statistic == pytest.approx(a.statistic)
    z = sps.norm.ppf(0.95)
    assert a.ci_high - a.difference == pytest.approx(z * a.se)


def test_calibrate_block_size_mechanics():
    # LW (2008), Algorithm 3.1: VAR(1) + stationary-bootstrap residuals (mean block 5), burn-in.
    # Only the mechanics are checked here (tiny K and M); see the report for a full-size run.
    r = _return_pair(120, np.random.default_rng(13), sharpe1=0.2, sharpe2=0.1, phi=0.2)
    kw = {"block_grid": (1, 2, 4), "n_pseudo": 12, "n_boot": 59, "seed": 8}
    cal = calibrate_block_size(r[:, 0], r[:, 1], **kw)
    assert cal.block in (1, 2, 4)
    assert cal.rejection_rates.shape == (3,)
    assert np.all((cal.rejection_rates >= 0) & (cal.rejection_rates <= 1))
    # the selected block minimises |rate - alpha| (first one on ties)
    assert cal.block == (1, 2, 4)[int(np.argmin(np.abs(cal.rejection_rates - 0.05)))]
    cal2 = calibrate_block_size(r[:, 0], r[:, 1], **kw)
    np.testing.assert_array_equal(cal.rejection_rates, cal2.rejection_rates)
    res = ledoit_wolf_sharpe_test(
        r[:, 0],
        r[:, 1],
        method="bootstrap",
        block="calibrate",
        n_boot=99,
        seed=8,
        calibration_kwargs={"block_grid": (1, 2, 4), "n_pseudo": 12, "n_boot": 59},
    )
    assert res.params["block"] in (1, 2, 4)
    assert "calibration" in res.params


# ---------------------------------------------------------------------------------------------
# Ledoit & Wolf (2008): published empirical example (Tables 2-3)
# ---------------------------------------------------------------------------------------------

_LW2008_ZIP_URL = (
    "https://www.econ.uzh.ch/dam/jcr:bc6a4f15-7957-4768-bc1b-1d6abc46497c/SharpeMatlab.zip"
)
# Provenance only (file as downloaded on 2026-10-04): sha256 of retSharpe.mat =
# 1a5ca9faf6ac21d26fb33c18ae9392db474b8eb76f39be28ce512ad3826f58ba. Identity of the data is
# verified below against Table 2 of the paper, not against the hash.


@pytest.fixture(scope="module")
def lw2008_data(request):
    from scipy.io import loadmat

    local = os.environ.get("CIF_LW2008_MAT")
    if local:
        path = Path(local)
    else:
        cache = getattr(request.config, "cache", None)
        if cache is not None:
            cache_dir = Path(cache.mkdir("cif_lw2008"))
        else:  # pragma: no cover - cacheprovider disabled
            cache_dir = request.getfixturevalue("tmp_path_factory").mktemp("cif_lw2008")
        path = cache_dir / "retSharpe.mat"
        if not path.exists():
            if os.environ.get("CIF_NETWORK_TESTS") != "1":
                pytest.skip("LW (2008) reference data not cached; set CIF_NETWORK_TESTS=1 to download")
            try:
                with urllib.request.urlopen(_LW2008_ZIP_URL, timeout=30) as resp:
                    payload = resp.read()
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    name = next(
                        n
                        for n in zf.namelist()
                        if n.endswith("retSharpe.mat") and "__MACOSX" not in n
                    )
                    path.write_bytes(zf.read(name))
            except Exception as exc:  # network unavailable or archive moved
                pytest.skip(f"LW (2008) reference data unavailable: {exc!r}")
    d = loadmat(path)
    return d["retAgg"], d["retHedge"]


@pytest.mark.network
def test_lw2008_data_match_table2(lw2008_data):
    agg, hedge = lw2008_data
    assert agg.shape == (120, 2) and hedge.shape == (120, 2)
    # Table 2: mean, st.dev., first-order autocorrelation (3 decimals). The Sharpe column of
    # Table 2 is not used: for Fidelity it is inconsistent with its own mean/sd (0.511/4.760).
    table2 = {
        "Fidelity": (agg[:, 0], 0.511, 4.760, -0.010),
        "Fidelity Aggressive Growth": (agg[:, 1], 0.098, 9.161, 0.090),
        "Coast Enhanced Income": (hedge[:, 1], 0.245, 0.168, 0.152),
        "JMG Capital Partners": (hedge[:, 0], 1.228, 1.211, 0.435),
    }
    for name, (x, mean, sd, rho1) in table2.items():
        assert abs(x.mean() - mean) <= 0.0005, name
        assert abs(x.std(ddof=1) - sd) <= 0.0005, name
        assert abs(np.corrcoef(x[1:], x[:-1])[0, 1] - rho1) <= 0.0006, name


@pytest.mark.network
def test_lw2008_table3_jkm_and_hac(lw2008_data):
    agg, hedge = lw2008_data
    # Table 3, p-values in percent (one decimal): JKM, HAC (QS kernel), HAC_pw (prewhitened QS)
    published = {"mutual": (agg, 3.9, 6.3, 6.7), "hedge": (hedge, 1.0, 14.7, 25.4)}
    for name, (r, p_jkm, p_hac, p_hacpw) in published.items():
        jkm = jobson_korkie_memmel(r[:, 0], r[:, 1])
        hac = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], kernel="qs", prewhiten=False)
        hac_pw = ledoit_wolf_sharpe_test(r[:, 0], r[:, 1], kernel="qs", prewhiten=True)
        print(
            f"[stats-report] LW2008 Table 3 {name}: JKM={100 * jkm.pvalue:.2f}% (pub {p_jkm}), HAC={100 * hac.pvalue:.2f}% (pub {p_hac}), HAC_pw={100 * hac_pw.pvalue:.2f}% (pub {p_hacpw})"
        )
        assert abs(100 * jkm.pvalue - p_jkm) <= 0.05 + 1e-9, (name, 100 * jkm.pvalue)
        assert abs(100 * hac.pvalue - p_hac) <= 0.05 + 1e-9, (name, 100 * hac.pvalue)
        assert abs(100 * hac_pw.pvalue - p_hacpw) <= 0.05 + 1e-9, (name, 100 * hac_pw.pvalue)


@pytest.mark.network
def test_lw2008_table3_bootstrap(lw2008_data):
    agg, hedge = lw2008_data
    # Table 3, Boot-TS: 9.2% (mutual funds, b = 4) and 29.4% (hedge funds, b = 6); the paper used
    # M = 4999 (Sec. 6). Tolerance fixed in advance: 4 Monte Carlo s.e. of the difference between
    # the published p-value (M = 4999) and ours (M = 19999), plus 0.0005 for rounding.
    for r, block, p_pub in ((agg, 4, 0.092), (hedge, 6, 0.294)):
        res = ledoit_wolf_sharpe_test(
            r[:, 0], r[:, 1], method="bootstrap", block=block, n_boot=19999, seed=2008
        )
        tol = 4 * math.sqrt(p_pub * (1 - p_pub) * (1 / 4999 + 1 / 19999)) + 0.0005
        print(
            f"[stats-report] LW2008 Table 3 Boot-TS b={block}: p={100 * res.pvalue:.2f}% (pub {100 * p_pub:.1f}%, tol +-{100 * tol:.2f})"
        )
        assert abs(res.pvalue - p_pub) <= tol, (block, res.pvalue)


# ---------------------------------------------------------------------------------------------
# PSR and DSR (Bailey & Lopez de Prado 2012, 2014)
# ---------------------------------------------------------------------------------------------


def _psr_range(sr: float, n: int, skew: float, kurt: float, rounded_moments: bool):
    """Range of PSR(0) over the rounding box of published 3-decimal inputs (PSR is monotone in
    each input here, so the corners give the extremes)."""
    h = 0.0005
    moments = (
        [(a, b) for a in (skew - h, skew + h) for b in (kurt - h, kurt + h)]
        if rounded_moments
        else [(skew, kurt)]
    )
    vals = [
        probabilistic_sharpe_ratio(sr=s, n_obs=n, skew=a, kurtosis=b)
        for s in (sr - h, sr + h)
        for a, b in moments
    ]
    return min(vals), max(vals)


def test_psr_reproduces_bldp2012_example():
    # BLdP (2012), Sec. 3: monthly track record of 2 years, SR = 0.458 (monthly); published
    # PSR(0) = 0.982 (normal: skew 0, kurtosis 3), 0.913 (skew -2.448, kurtosis 10.164) and
    # 0.953 for 3 years. Inputs and outputs are rounded to 3 decimals, so the published value must
    # lie within the range of PSR over the input rounding box, +-0.0005 output rounding.
    # (At the rounded inputs themselves the 3-year value is 0.95351, i.e. 0.954.)
    cases = [
        (24, 0.0, 3.0, False, 0.982),
        (24, -2.448, 10.164, True, 0.913),
        (36, -2.448, 10.164, True, 0.953),
    ]
    for n, skew, kurt, rounded, published in cases:
        lo, hi = _psr_range(0.458, n, skew, kurt, rounded)
        assert lo - 0.0005 <= published <= hi + 0.0005, (n, lo, hi)
    # the formula at the rounded inputs is within 0.0006 of every published value
    assert (
        abs(probabilistic_sharpe_ratio(sr=0.458, n_obs=24, skew=0.0, kurtosis=3.0) - 0.982) < 6e-4
    )


def test_dsr_reproduces_bldp2014_numerical_example():
    # BLdP (2014), "A numerical example": N = 100 trials, V[SR_n] = 1/2 (annualised), T = 1250
    # daily observations, skew = -3, kurtosis = 10, annualised SR = 2.5, 250 obs per year.
    sr = 2.5 / math.sqrt(250)
    v = 0.5 / 250
    sr0 = expected_max_sharpe(v, 100)
    assert abs(sr0 - 0.1132) <= 0.00005
    common = {"sr": sr, "n_obs": 1250, "sr_variance": v}
    res = deflated_sharpe_ratio(skew=-3.0, kurtosis=10.0, n_trials=100, **common)
    assert abs(res.dsr - 0.9004) <= 0.00005
    assert res.sr0 == pytest.approx(sr0)
    # "after running only N = 46 independent trials ... DSR would have been 0.9505"
    res46 = deflated_sharpe_ratio(skew=-3.0, kurtosis=10.0, n_trials=46, **common)
    assert abs(res46.dsr - 0.9505) <= 0.00005
    # "Normal returns ... DSR = 0.9505 after N = 88 independent trials"
    res88 = deflated_sharpe_ratio(skew=0.0, kurtosis=3.0, n_trials=88, **common)
    assert abs(res88.dsr - 0.9505) <= 0.00005


def test_psr_from_returns_matches_manual_computation():
    rng = np.random.default_rng(3)
    r = 0.001 + 0.01 * (rng.standard_gamma(2.0, 750) - 2.0) / math.sqrt(2.0)  # skewed
    rf = 0.0002
    ex = r - rf
    sr = ex.mean() / ex.std(ddof=1)
    g3 = sps.skew(ex)
    g4 = sps.kurtosis(ex, fisher=False)
    z = (sr - 0.02) * math.sqrt(len(ex) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert probabilistic_sharpe_ratio(r, 0.02, rf=rf) == pytest.approx(sps.norm.cdf(z), rel=1e-12)
    # a benchmark equal to the estimate gives exactly one half
    assert probabilistic_sharpe_ratio(r, sr, rf=rf) == pytest.approx(0.5)
    dsr = deflated_sharpe_ratio(r, rf=rf, sr_variance=1e-4, n_trials=20)
    sr0 = math.sqrt(1e-4) * (
        (1 - np.euler_gamma) * sps.norm.ppf(1 - 1 / 20)
        + np.euler_gamma * sps.norm.ppf(1 - 1 / (20 * math.e))
    )
    z0 = (sr - sr0) * math.sqrt(len(ex) - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    assert dsr.dsr == pytest.approx(sps.norm.cdf(z0), rel=1e-12)
    assert dsr.n_obs == len(ex)


def test_psr_and_dsr_reject_ambiguous_or_invalid_input():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(np.ones(10) * 0.01 + np.arange(10) * 1e-3, sr=0.1)
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(sr=0.1, n_obs=10)  # missing higher moments
    with pytest.raises(ValueError):
        expected_max_sharpe(0.01, 1)


# ---------------------------------------------------------------------------------------------
# Holm, TOST, performance profiles, TTT
# ---------------------------------------------------------------------------------------------


def test_holm_known_example():
    # sorted p: 0.005, 0.01, 0.03, 0.04 -> 4*0.005, 3*0.01, 2*0.03, max(1*0.04, 0.06)
    adj = holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])
    assert holm([0.2]).tolist() == [0.2]


def test_holm_agrees_with_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(1).uniform(0, 0.2, size=17)
    p[3] = p[5]  # a tie
    np.testing.assert_allclose(holm(p), multipletests(p, method="holm")[1], rtol=1e-12)


def test_holm_rejects_invalid_pvalues():
    with pytest.raises(ValueError):
        holm([0.1, np.nan])
    with pytest.raises(ValueError):
        holm([0.1, 1.2])


def test_tost_from_ci():
    assert tost_from_ci(-0.03, 0.04, 0.05).equivalent
    assert not tost_from_ci(-0.03, 0.05, 0.05).equivalent  # boundary is not inside
    assert not tost_from_ci(-0.06, 0.01, 0.05).equivalent
    asym = tost_from_ci(-0.03, 0.04, (-0.02, 0.10))
    assert not asym.equivalent and asym.margin_low == -0.02 and asym.margin_high == 0.10
    with pytest.raises(ValueError):
        tost_from_ci(0.1, -0.1, 0.05)
    with pytest.raises(ValueError):
        tost_from_ci(-0.1, 0.1, -0.05)


def test_tost_bootstrap():
    rng = np.random.default_rng(0)
    near_zero = rng.normal(0.0, 0.01, size=4000)
    res = tost_bootstrap(near_zero, 0.05, alpha=0.05)
    assert res.equivalent
    assert res.pvalue == max(res.p_lower, res.p_upper)
    assert res.p_lower == pytest.approx(np.mean(near_zero <= -0.05))
    lo, hi = np.quantile(near_zero, [0.05, 0.95], method="inverted_cdf")
    assert (res.ci_low, res.ci_high) == (lo, hi)
    shifted = rng.normal(0.045, 0.01, size=4000)
    res2 = tost_bootstrap(shifted, 0.05, alpha=0.05)
    assert not res2.equivalent and res2.pvalue > 0.05


def test_performance_profile_manual_example():
    inf = np.inf
    times = {
        "A": [1.0, 2.0, inf, 4.0, inf],
        "B": [2.0, 1.0, 3.0, inf, inf],  # problem 5 is solved by nobody
    }
    ratios = performance_ratios(times)
    np.testing.assert_array_equal(ratios["A"], [1.0, 2.0, inf, 1.0, inf])
    np.testing.assert_array_equal(ratios["B"], [2.0, 1.0, 1.0, inf, inf])
    prof = performance_profile(times, [1.0, 1.5, 2.0, 10.0])
    np.testing.assert_allclose(prof["A"], [2 / 5, 2 / 5, 3 / 5, 3 / 5])
    np.testing.assert_allclose(prof["B"], [2 / 5, 2 / 5, 3 / 5, 3 / 5])


def test_performance_profile_rejects_bad_input():
    with pytest.raises(ValueError):
        performance_profile({"A": [1.0, 2.0], "B": [1.0]}, [1.0])
    with pytest.raises(ValueError):
        performance_profile({"A": [1.0, np.nan]}, [1.0])
    with pytest.raises(ValueError):
        performance_profile({"A": [0.0, 1.0]}, [1.0])


def test_ttt_ecdf_manual_example():
    ttt = ttt_ecdf([3.0, 1.0, np.inf, 2.0])
    np.testing.assert_array_equal(ttt.times, [1.0, 2.0, 3.0])
    # Aiex, Resende & Ribeiro (2007): p_i = (i - 1/2) / n, n = all runs (failures included)
    np.testing.assert_allclose(ttt.probabilities, [0.125, 0.375, 0.625])
    assert ttt.n_runs == 4 and ttt.n_failed == 1
