"""Bootstrap instances on S&P 500 dates, cell C4 (ANALYSIS_PLAN §3 "База boot (N12)").

Seed s draws a month-end formation date uniformly from 2013-2025; the universe is the top ``n`` of
:meth:`cif.data.sp500_pit.SP500PIT.select`; the 756 trading days before the date are resampled with
replacement; μ is the sample mean and Σ the Ledoit-Wolf estimate (annualised as in the walk-forward).

No test touches the real panel: the dataset is a synthetic ``SP500PIT`` of random walks with a common
factor. The λ rule of C4 is tested on synthetic instances and on the pure rule.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chisquare

from cif.data.sp500_pit import SP500PIT
from cif.data.statistics import annualised_mu, annualised_sigma, log_returns
from cif.instances.boot import (
    BOOT_LAMBDAS,
    BOOT_MIN_MEDIAN_SCONT,
    boot_dates,
    boot_instance,
    calibrate_boot_lambda,
    month_ends,
    select_boot_lambda,
)
from cif.instances.diagnostics import binding_diagnostics
from cif.problem import PortfolioProblem

N_TICKERS = 14


def _dataset(seed: int = 0, n_tickers: int = N_TICKERS) -> SP500PIT:
    """Random walks with one common factor, 2010-2025 (weekdays), ticker 0 the most liquid."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", "2025-12-31")
    cols = [f"T{j:02d}" for j in range(n_tickers)]
    common = rng.normal(0.0003, 0.010, size=(len(dates), 1))
    beta = rng.uniform(0.5, 1.5, size=(1, n_tickers))
    rets = common * beta + rng.normal(0.0, 0.012, size=(len(dates), n_tickers))
    prices = pd.DataFrame(50.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)
    volume = pd.DataFrame(np.tile(2.0 ** (n_tickers - np.arange(n_tickers)), (len(dates), 1)),
                          index=dates, columns=cols)
    return SP500PIT(
        membership=pd.DataFrame(True, index=dates, columns=cols),
        prices_adj=prices,
        close_unadj=prices.copy(),
        volume=volume,
        coverage=pd.DataFrame(),
        companies=pd.DataFrame({"ticker": cols, "company_id": cols}),
        failed={},
    )


@pytest.fixture(scope="module")
def data() -> SP500PIT:
    return _dataset()


# --------------------------------------------------------------------------------------
# Formation dates
# --------------------------------------------------------------------------------------


def test_month_ends_cover_2013_to_2025():
    ends = month_ends()
    assert len(ends) == 13 * 12
    assert ends[0] == pd.Timestamp("2013-01-31") and ends[-1] == pd.Timestamp("2025-12-31")
    assert ends.is_month_end.all() and ends.is_monotonic_increasing and ends.is_unique
    assert len(month_ends(2020, 2020)) == 12


def test_formation_dates_are_uniform_over_month_ends_and_reproducible(data):
    ends = month_ends()
    draws = Counter(boot_dates(data, seed)[0] for seed in range(15_600))
    counts = np.array([draws[e] for e in ends])
    assert counts.min() > 0 and set(draws) <= set(ends)
    assert chisquare(counts).pvalue > 1e-3  # 100 draws expected per month
    assert boot_dates(data, 1100) == boot_dates(data, 1100)


def test_formation_date_is_the_last_trading_day_on_or_before_the_month_end(data):
    for seed in (1100, 1101, 1102, 1103):
        month_end, formation = boot_dates(data, seed)
        assert formation <= month_end and formation in data.prices_adj.index
        assert (month_end - formation).days <= 3  # at most a weekend/holiday back
        later = data.prices_adj.index[data.prices_adj.index > formation]
        assert later[0] > month_end  # nothing between the formation date and the month end


# --------------------------------------------------------------------------------------
# The instance
# --------------------------------------------------------------------------------------


def test_instance_has_the_requested_shape_and_a_positive_definite_sigma(data):
    mu, sigma, meta = boot_instance(data, 1100, n=10)
    assert mu.shape == (10,) and sigma.shape == (10, 10)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-15)
    assert np.linalg.eigvalsh(sigma)[0] > 0.0
    assert np.isfinite(mu).all()
    month_end, formation = boot_dates(data, 1100)
    assert meta["family"] == "boot" and meta["seed"] == 1100 and meta["n"] == 10
    assert meta["month_end"] == month_end.date().isoformat()
    assert meta["formation_date"] == formation.date().isoformat()
    assert meta["tickers"] == data.select(formation, n=10, lookback=756)
    assert meta["train_days"] == 756 and meta["n_returns"] == 755


def test_moments_are_the_project_estimators_on_the_resampled_window(data):
    """μ: annualised sample mean; Σ: annualised Ledoit-Wolf; rows drawn with replacement."""
    mu, sigma, meta = boot_instance(data, 1103, n=8)
    window = data.prices_adj.loc[: meta["formation_date"], meta["tickers"]].ffill().iloc[-756:]
    returns = log_returns(window)
    assert len(returns) == 755 and len(meta["rows"]) == 755
    assert meta["rows"].min() >= 0 and meta["rows"].max() < 755
    assert len(set(meta["rows"].tolist())) < 755  # with replacement: repeats (about 63 % distinct)
    assert len(set(meta["rows"].tolist())) > 400
    sample = returns.iloc[meta["rows"]]
    np.testing.assert_allclose(mu, annualised_mu(sample).to_numpy(), rtol=1e-13)
    np.testing.assert_allclose(sigma, annualised_sigma(sample, method="ledoit_wolf").to_numpy(),
                               rtol=1e-12)
    # not the plain window estimate: the bootstrap moves both moments
    assert not np.allclose(mu, annualised_mu(returns).to_numpy(), rtol=1e-3)
    # and not the sample covariance: the shrinkage target (a scaled identity) is mixed in
    assert not np.allclose(sigma, annualised_sigma(sample, method="sample").to_numpy(), rtol=1e-6)


def test_instance_is_reproducible_and_depends_on_the_seed(data):
    a = boot_instance(data, 1100, n=10)
    b = boot_instance(data, 1100, n=10)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2]["rows"], b[2]["rows"])
    c = boot_instance(data, 1101, n=10)
    assert not np.array_equal(a[0], c[0])


def test_two_seeds_with_one_date_still_draw_different_bootstrap_samples(data):
    by_date: dict[pd.Timestamp, list[int]] = {}
    for seed in range(1100, 1400):
        by_date.setdefault(boot_dates(data, seed)[0], []).append(seed)
    s1, s2 = next(v for v in by_date.values() if len(v) >= 2)[:2]
    a, b = boot_instance(data, s1, n=8), boot_instance(data, s2, n=8)
    assert a[2]["formation_date"] == b[2]["formation_date"] and a[2]["tickers"] == b[2]["tickers"]
    assert not np.array_equal(a[2]["rows"], b[2]["rows"])
    assert not np.array_equal(a[0], b[0])


def test_nothing_after_the_formation_date_is_read(data):
    """Overwriting every value dated after the formation date leaves the instance unchanged."""
    seed = 1105
    _, formation = boot_dates(data, seed)
    rng = np.random.default_rng(7)
    after = data.prices_adj.index > formation
    frames = {}
    for name in ("prices_adj", "close_unadj", "volume"):
        frame = getattr(data, name).copy()
        frame.loc[after] = rng.uniform(1.0, 100.0, size=frame.loc[after].shape)
        frames[name] = frame
    other = dataclasses.replace(data, **frames)
    a, b = boot_instance(data, seed, n=9), boot_instance(other, seed, n=9)
    assert a[2]["tickers"] == b[2]["tickers"]
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_a_series_that_starts_inside_the_window_shortens_it_like_the_walk_forward(data):
    """No price before listing: leading rows are dropped for all assets (``log_returns``)."""
    seed = 1100
    _, formation = boot_dates(data, seed)
    pos = data.prices_adj.index.get_loc(formation)
    first_row = pos - 755
    prices = data.prices_adj.copy()
    prices.iloc[: first_row + 30, 1] = np.nan  # T01 is listed 30 rows into the window
    other = dataclasses.replace(data, prices_adj=prices)
    _, _, meta = boot_instance(other, seed, n=8)
    assert "T01" in meta["tickers"]  # 726/756 = 96 % valid: still eligible
    assert meta["n_returns"] == 725 and len(meta["rows"]) == 725


def test_a_gap_inside_the_window_is_forward_filled(data):
    seed = 1100
    _, formation = boot_dates(data, seed)
    pos = data.prices_adj.index.get_loc(formation)
    prices = data.prices_adj.copy()
    prices.iloc[pos - 300 : pos - 295, 2] = np.nan  # five missing days of T02
    other = dataclasses.replace(data, prices_adj=prices)
    _, _, meta = boot_instance(other, seed, n=8)
    assert meta["n_returns"] == 755  # nothing dropped


def test_too_few_qualifying_tickers_is_an_error_not_a_smaller_instance(data):
    with pytest.raises(ValueError, match="qualify"):
        boot_instance(data, 1100, n=N_TICKERS + 1)


def test_meta_without_arrays_is_json_safe(data):
    _, _, meta = boot_instance(data, 1100, n=6)
    json.dumps({k: v for k, v in meta.items() if k != "rows"})


# --------------------------------------------------------------------------------------
# The lambda rule of C4
# --------------------------------------------------------------------------------------


def test_rule_constants_are_the_registered_ones():
    assert BOOT_LAMBDAS == (5.0, 10.0, 20.0) and BOOT_MIN_MEDIAN_SCONT == 20


def test_rule_picks_the_smallest_lambda_whose_median_support_reaches_the_threshold():
    s = {5.0: [10] * 10, 10.0: [20] * 10, 20.0: [40] * 10}
    assert select_boot_lambda(s) == 10.0  # median 20 counts (>=)
    assert select_boot_lambda({5.0: [25] * 10, 10.0: [30] * 10}) == 5.0
    assert select_boot_lambda({5.0: [19] * 10, 10.0: [19] * 10, 20.0: [19] * 10}) is None
    assert select_boot_lambda({20.0: [25] * 10, 5.0: [10] * 10, 10.0: [10] * 10}) == 20.0  # any key order


def test_rule_uses_the_median_not_the_mean():
    outliers = {5.0: [5, 5, 5, 5, 5, 5, 90, 90, 90, 90], 10.0: [19] * 10}  # mean 47, median 5
    assert select_boot_lambda(outliers) is None
    lopsided = {5.0: [30] * 5 + [1] * 5}  # median 15.5
    assert select_boot_lambda(lopsided) is None
    assert select_boot_lambda({5.0: [19] * 5 + [21] * 5}) == 5.0  # median exactly 20


def test_rule_rejects_an_empty_calibration():
    with pytest.raises(ValueError):
        select_boot_lambda({})
    with pytest.raises(ValueError):
        select_boot_lambda({5.0: []})


def _diagonal_draw(spread: float):
    def draw(seed: int):
        rng = np.random.default_rng(seed)
        n = 60
        mu = 0.02 + spread * np.arange(n) / n + rng.normal(0.0, 0.0005, n)
        return mu, 0.04 * np.eye(n)

    return draw


def test_calibration_computes_support_by_lambda_and_applies_the_rule():
    """Diagonal Σ, 60 assets: the support of the long-only optimum grows with λ."""
    result = calibrate_boot_lambda(_diagonal_draw(0.10))
    assert result.seeds == tuple(range(1100, 1110))
    assert result.lambdas == (5.0, 10.0, 20.0) and result.threshold == 20
    medians = result.median_s_cont
    assert medians[5.0] < 20 <= medians[10.0] <= medians[20.0]
    assert result.lam == 10.0
    # every cell is the support of the continuous relaxation of that instance and λ
    mu, sigma = _diagonal_draw(0.10)(1100)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(str(i) for i in range(60)),
                               risk_aversion=10.0)
    assert result.s_cont[10.0][0] == binding_diagnostics(problem)["s_cont"]
    assert all(len(v) == 10 for v in result.s_cont.values())
    json.dumps(result.as_dict(), allow_nan=False)  # destined for the deviation log


def test_calibration_with_a_wide_spread_needs_more_risk_aversion_and_can_fail():
    assert calibrate_boot_lambda(_diagonal_draw(0.05)).lam == 5.0
    assert calibrate_boot_lambda(_diagonal_draw(0.10), min_median_scont=50).lam is None


def test_calibration_draws_each_instance_once():
    calls: list[int] = []

    def draw(seed: int):
        calls.append(seed)
        return _diagonal_draw(0.10)(seed)

    calibrate_boot_lambda(draw, seeds=range(1100, 1103))
    assert calls == [1100, 1101, 1102]
