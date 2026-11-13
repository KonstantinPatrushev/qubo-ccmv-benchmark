"""Factor-model instances with sector blocks and realistic volatilities / correlations.

The legacy generator ``Σ = 0.005·AAᵀ + 0.01·I`` has volatility ``√(0.005 N + 0.01)`` (33 % at
N = 20, 100 % at N = 200) and correlations ~N(0, 1/N); see
:func:`cif.instances.diagnostics.describe_legacy_dense`. Here::

    Σ = B F Bᵀ + D,   F = diag(f) over F_c common + S sector factors,   D = diag(d) > 0.

Construction (all in "raw" units first, then rescaled per asset, which keeps the factor form with
``B ← diag(c) B, D ← diag(c²) D`` and leaves every correlation unchanged):

1. *Sectors.* ``S = min(n_sectors, N)`` labels; sector sizes ~ Dirichlet(3), every sector
   non-empty; labels shuffled so that the index order carries no block structure.
2. *Loadings.* Factor 0 is the market: ``b_i0 ~ N(1, 0.3)`` clipped to [0.2, 2], variance 1.
   Factors 1…F_c-1 are style tilts: ``b_ik ~ N(0, 0.6)``, variance ``style_strength`` (0.15).
   Sector factor s: loading 1 on its members, variance ``sector_strength`` (0.6).
3. *Idiosyncratic variances* ``d_i = s²·e_i`` with ``e_i`` lognormal (median 1, log-sd 0.35);
   the scale ``s²`` is found by bisection so that the mean pairwise correlation equals
   ``mean_corr`` exactly. With the defaults this gives roughly 0.28 across sectors and 0.45
   within a sector, i.e. visible sector blocks around the requested mean.
4. *Volatilities.* ``v_i = vol_median · exp(vol_dispersion · z_i)``, ``z_i ~ N(0, 1)``, clipped to
   ``vol_range``; rows of ``B`` and ``d`` are rescaled so that ``Σ_ii = v_i²`` exactly.
5. *Expected returns.* ``risk_linked`` (default): ``μ_i = rf + β_i · equity_premium + ε_i``,
   ``ε_i ~ N(0, mu_noise²)``, where ``β_i = N (Σ1)_i / (1ᵀΣ1)`` is the beta against the
   equal-weighted portfolio (mean exactly 1). ``random``: ``μ_i ~ U(mu_range)`` independent of
   risk (the legacy range 0.05-0.25). μ is drawn after Σ, so the same seed gives the same Σ in
   both modes.

Defaults are chosen to meet the WP-E3 targets (annual volatilities 15-60 %, mean pairwise
correlation 0.2-0.4 with sector structure), not fitted to a particular market. The cross-
sectional dispersion of the risk-linked μ is small (sd ≈ 0.03), which keeps the continuous
optimum diversified; raise ``mu_noise`` or use ``mu_mode="random"`` for a more concentrated one.
The binding status is not a property of the family: set K as a fraction of ``s_cont``
(:func:`cif.instances.diagnostics.k_from_scont`).

Seeds are explicit; use calibration 1000-1999 and test 2000-2999 (ANALYSIS_PLAN §3).
"""

from __future__ import annotations

import numpy as np


def _mean_offdiag_corr(cov_sys: np.ndarray, sys_diag: np.ndarray, total_diag: np.ndarray) -> float:
    """Mean off-diagonal correlation of ``cov_sys + diag(total_diag - sys_diag)``."""
    n = cov_sys.shape[0]
    u = 1.0 / np.sqrt(total_diag)
    off = u @ cov_sys @ u - np.sum(sys_diag * u * u)
    return float(off / (n * (n - 1)))


def factor_instance(
    n: int,
    seed: int,
    n_factors: int = 3,
    n_sectors: int = 10,
    *,
    mean_corr: float = 0.30,
    sector_strength: float = 0.6,
    style_strength: float = 0.15,
    vol_median: float = 0.28,
    vol_dispersion: float = 0.30,
    vol_range: tuple[float, float] = (0.15, 0.60),
    mu_mode: str = "risk_linked",
    rf: float = 0.03,
    equity_premium: float = 0.05,
    mu_noise: float = 0.02,
    mu_range: tuple[float, float] = (0.05, 0.25),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Draw ``(μ, Σ, meta)`` for ``n`` assets; annualised units.

    ``meta`` holds the parameters, realised statistics and the decomposition: scalars ``family,
    seed, n_factors, n_sectors, mean_corr_target, mean_corr, vol_min, vol_median, vol_max,
    mu_mode, idio_scale`` and arrays ``sector`` (labels), ``beta``, ``loadings`` (``n x (F_c + S)``,
    already rescaled), ``factor_var`` and ``idio_var`` with
    ``Σ = loadings · diag(factor_var) · loadingsᵀ + diag(idio_var)``. Drop the arrays before
    serialising ``meta`` to JSON.
    """
    lo, hi = vol_range
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if n_factors < 1 or n_sectors < 1:
        raise ValueError("n_factors and n_sectors must be >= 1")
    if not 0.0 < mean_corr < 1.0:
        raise ValueError(f"mean_corr must be in (0, 1), got {mean_corr}")
    if not 0.0 < lo < hi:
        raise ValueError(f"vol_range must satisfy 0 < low < high, got {vol_range}")
    if mu_mode not in ("risk_linked", "random"):
        raise ValueError(f"mu_mode must be 'risk_linked' or 'random', got {mu_mode!r}")
    if min(sector_strength, style_strength, vol_dispersion, mu_noise) < 0.0:
        raise ValueError("strengths, vol_dispersion and mu_noise must be >= 0")

    rng = np.random.default_rng(seed)
    n_sec = min(n_sectors, n)

    # 1. sector labels: every sector non-empty, sizes ~ Dirichlet(3), shuffled
    probs = rng.dirichlet(np.full(n_sec, 3.0))
    sector = np.concatenate([np.arange(n_sec), rng.choice(n_sec, size=n - n_sec, p=probs)])
    sector = rng.permutation(sector)

    # 2. loadings and factor variances (raw units)
    loadings = np.zeros((n, n_factors + n_sec))
    loadings[:, 0] = np.clip(rng.normal(1.0, 0.30, size=n), 0.2, 2.0)
    if n_factors > 1:
        loadings[:, 1:n_factors] = rng.normal(0.0, 0.6, size=(n, n_factors - 1))
    loadings[np.arange(n), n_factors + sector] = 1.0
    factor_var = np.concatenate(
        [[1.0], np.full(n_factors - 1, style_strength), np.full(n_sec, sector_strength)]
    )
    cov_sys = (loadings * factor_var) @ loadings.T
    sys_diag = np.diag(cov_sys).copy()

    # 3. idiosyncratic variances; bisection on log s² for the mean pairwise correlation
    shape = np.exp(rng.normal(0.0, 0.35, size=n))
    lo_s2, hi_s2 = 1e-9 * sys_diag.mean(), 1e9 * sys_diag.mean()
    if _mean_offdiag_corr(cov_sys, sys_diag, sys_diag + lo_s2 * shape) < mean_corr:
        raise ValueError(
            f"mean_corr={mean_corr} is not reachable with n_factors={n_factors}, "
            f"n_sectors={n_sectors}, sector_strength={sector_strength}"
        )
    if _mean_offdiag_corr(cov_sys, sys_diag, sys_diag + hi_s2 * shape) > mean_corr:
        raise ValueError(f"mean_corr={mean_corr} is too small to be calibrated")
    for _ in range(200):
        mid = np.sqrt(lo_s2 * hi_s2)
        if _mean_offdiag_corr(cov_sys, sys_diag, sys_diag + mid * shape) > mean_corr:
            lo_s2 = mid
        else:
            hi_s2 = mid
        if hi_s2 / lo_s2 < 1.0 + 1e-14:
            break
    idio_scale = float(np.sqrt(lo_s2 * hi_s2))
    idio_raw = idio_scale * shape

    # 4. target volatilities; rescale rows of B and D so that Σ_ii = v_i²
    vol = np.clip(vol_median * np.exp(vol_dispersion * rng.standard_normal(n)), lo, hi)
    c = vol / np.sqrt(sys_diag + idio_raw)
    loadings = loadings * c[:, None]
    idio_var = c**2 * idio_raw
    sigma = (loadings * factor_var) @ loadings.T + np.diag(idio_var)
    sigma = 0.5 * (sigma + sigma.T)

    # 5. expected returns (after Σ, so that Σ does not depend on mu_mode)
    ones = np.ones(n)
    beta = n * (sigma @ ones) / (ones @ sigma @ ones)
    if mu_mode == "risk_linked":
        mu = rf + equity_premium * beta + rng.normal(0.0, mu_noise, size=n)
    else:
        mu = rng.uniform(mu_range[0], mu_range[1], size=n)

    sd = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(sd, sd)
    meta = {
        "family": "factor",
        "seed": int(seed),
        "n": int(n),
        "n_factors": int(n_factors),
        "n_sectors": int(n_sec),
        "mean_corr_target": float(mean_corr),
        "mean_corr": float(corr[np.triu_indices(n, k=1)].mean()),
        "vol_min": float(sd.min()),
        "vol_median": float(np.median(sd)),
        "vol_max": float(sd.max()),
        "mu_mode": mu_mode,
        "idio_scale": idio_scale,
        "sector": sector,
        "beta": beta,
        "loadings": loadings,
        "factor_var": factor_var,
        "idio_var": idio_var,
    }
    return mu, sigma, meta
