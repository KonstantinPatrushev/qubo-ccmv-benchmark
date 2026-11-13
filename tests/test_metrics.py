"""Tests for cif.metrics.performance (R2: correct risk-adjusted metrics on daily data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.metrics.performance import (
    annualised_return,
    certainty_equivalent,
    downside_deviation,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarise_returns,
)


def _series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_sharpe_uses_excess_returns_and_sample_std():
    r = _series([0.01, -0.005, 0.002, 0.004, -0.001])
    rf = _series([0.0001] * 5)
    ex = r - rf
    expected = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r, rf) == pytest.approx(expected)
    assert sharpe_ratio(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_downside_deviation_is_over_all_observations():
    r = np.array([0.02, -0.01, 0.0, -0.03])
    # Sortino & Price (1994): sqrt(mean(min(r - target, 0)^2)) over ALL observations
    expected = np.sqrt((0.01**2 + 0.03**2) / 4)
    assert downside_deviation(r) == pytest.approx(expected)


def test_sortino_matches_definition_and_exceeds_sharpe_for_symmetric_noise():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0.0005, 0.01, size=2000))
    dd = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    assert sortino_ratio(r) == pytest.approx(r.mean() / dd * np.sqrt(252))
    # For roughly symmetric returns the downside deviation is ~ std/sqrt(2) -> Sortino > Sharpe
    assert sortino_ratio(r) > sharpe_ratio(r)


def test_max_drawdown_on_known_path():
    equity = _series([1.0, 1.2, 0.9, 1.1, 0.6, 1.3])
    mdd, peak, trough = max_drawdown(equity)
    assert mdd == pytest.approx(0.6 / 1.2 - 1.0)
    assert peak == equity.index[1]
    assert trough == equity.index[4]


def test_annualised_return_uses_calendar_time():
    idx = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
    r = pd.Series([0.0, 0.10, 0.10], index=idx)  # first value is the start marker
    # two years, total growth 1.21 -> CAGR 10 %
    assert annualised_return(r.iloc[1:], start=idx[0]) == pytest.approx(0.10, rel=1e-3)


def test_certainty_equivalent_formula():
    rng = np.random.default_rng(1)
    r = _series(rng.normal(0.0004, 0.012, size=500))
    rf = _series([0.0001] * 500)
    ex = r - rf
    expected = (ex.mean() - 0.5 * 5.0 * ex.var(ddof=1)) * 252
    assert certainty_equivalent(r, rf, gamma=5.0) == pytest.approx(expected)


def test_summarise_returns_keys_and_consistency():
    rng = np.random.default_rng(2)
    r = _series(rng.normal(0.0004, 0.01, size=600))
    s = summarise_returns(r)
    for key in ("cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "ceq", "n_days", "years", "rf_used"):
        assert key in s
    equity = (1.0 + r).cumprod()
    assert s["max_drawdown"] == pytest.approx(max_drawdown(equity)[0])
    assert s["rf_used"] is False
