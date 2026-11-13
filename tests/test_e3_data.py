"""E3 data layer: periods, fold tables, the fold universe and the cardinality rule.

ANALYSIS_PLAN §3 ("Рыночные данные", "Walk-forward"): the training window is 756 days and the test
window 21; the estimation of a fold reads nothing after its formation date; ``K_f = max(⌈1/u⌉ + 1,
⌊s_cont,f/2⌋)``; an FF49 industry enters a fold if its training window has no gaps. Everything runs
on synthetic panels (a factor model with a known covariance).
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from cif.backtest.walkforward import run_walkforward
from cif.data.statistics import annualised_mu, annualised_sigma, log_returns
from cif.experiments import e3_data as d
from tests.e3_fixtures import MARKET, PERIOD


@pytest.fixture(autouse=True)
def _fresh_caches():
    d.clear_caches()
    yield
    d.clear_caches()


def _market(**over) -> d.Market:
    spec = copy.deepcopy(MARKET)
    spec.update(over)
    return d.synthetic_market(spec)


# --------------------------------------------------------------------------------------------
# The cardinality rule
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("u", "floor"), [(0.20, 6), (0.10, 11), (0.5, 3), (0.34, 4), (1.0, 2), (0.25, 5), (0.3, 5)]
)
def test_floor_cardinality_is_ceil_of_one_over_u_plus_one(u, floor):
    assert d.floor_cardinality(u) == floor


@pytest.mark.parametrize(
    ("s_cont", "u", "k", "binding", "lower"),
    [
        (10, 0.20, 6, True, True),     # ⌊10/2⌋ = 5 < 6: the floor acts, 10 > 6 binds
        (12, 0.20, 6, True, True),     # ⌊12/2⌋ = 6 = floor
        (13, 0.20, 6, True, True),
        (14, 0.20, 7, True, False),    # ⌊14/2⌋ = 7 > floor
        (6, 0.20, 6, False, True),     # s_cont = floor: not binding (K = s_cont)
        (4, 0.20, 6, False, True),     # s_cont < floor: K > s_cont
        (22, 0.10, 11, True, True),
        (30, 0.10, 15, True, False),
    ],
)
def test_cardinality_rule_and_binding_follow_the_plan(s_cont, u, k, binding, lower):
    assert d.cardinality_rule(s_cont, u) == k
    assert d.is_binding(s_cont, u) is binding
    assert d.lower_bound_active(s_cont, u) is lower
    # the plan: binds iff s_cont > ⌈1/u⌉ + 1, i.e. iff s_cont > K_f
    assert binding == (s_cont > k)


# --------------------------------------------------------------------------------------------
# Periods and folds
# --------------------------------------------------------------------------------------------


def test_period_grid_starts_the_first_test_window_at_the_period_start():
    idx = pd.bdate_range("2000-01-03", periods=300)
    grid = d.period_grid(idx, "2000-06-01", "2001-02-28", train_days=60, test_days=5)
    first_test = idx[grid.first_test_row]
    assert first_test >= pd.Timestamp("2000-06-01")
    assert idx[grid.first_test_row - 1] < pd.Timestamp("2000-06-01")  # the earliest such day
    assert grid.row_start == grid.first_test_row - 60
    # whole test windows only, the last one ends inside the period
    assert grid.row_end == grid.first_test_row + 5 * grid.n_folds
    assert idx[grid.row_end - 1] <= pd.Timestamp("2001-02-28")
    assert grid.row_end + 5 > np.searchsorted(idx, pd.Timestamp("2001-02-28"), side="right")


def test_period_grid_needs_the_training_history_before_the_start():
    idx = pd.bdate_range("2000-01-03", periods=300)
    with pytest.raises(ValueError, match="history"):
        d.period_grid(idx, "2000-01-05", "2001-02-28", train_days=60, test_days=5)


def test_period_grid_without_a_complete_window_is_an_error():
    idx = pd.bdate_range("2000-01-03", periods=300)
    with pytest.raises(ValueError, match="no complete test window"):
        d.period_grid(idx, "2000-06-01", "2000-06-03", train_days=60, test_days=5)


def test_fold_dates_do_not_look_ahead_and_tile_the_period():
    m = _market()
    table = d.build_fold_table(m, PERIOD)
    folds = table.folds
    assert len(folds) == table.grid.n_folds > 10
    idx = m.prices.index
    for f in folds:
        assert f.train_start <= f.formation_date < f.test_start <= f.test_end
        assert idx.get_loc(f.formation_date) - idx.get_loc(f.train_start) == 59  # 60 prices
        assert idx.get_loc(f.test_end) - idx.get_loc(f.test_start) == 4  # 5 test days
        assert idx.get_loc(f.test_start) - idx.get_loc(f.formation_date) == 1  # next close
    for a, b in pairwise(folds):
        assert idx.get_loc(b.test_start) - idx.get_loc(a.test_end) == 1  # contiguous windows
        assert idx.get_loc(b.formation_date) - idx.get_loc(a.formation_date) == 5  # step = test


def test_fold_problem_is_the_estimate_of_the_training_window():
    m = _market()
    table = d.build_fold_table(m, PERIOD)
    for f in (table.folds[0], table.folds[7], table.folds[-1]):
        train = m.prices.loc[f.train_start : f.formation_date, list(f.assets)]
        returns = log_returns(train)
        np.testing.assert_allclose(f.mu, annualised_mu(returns).to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(f.sigma, annualised_sigma(returns, method="ledoit_wolf").to_numpy(),
                                   rtol=1e-10)
        assert f.train_rows == len(returns) == 59


def test_fold_table_agrees_with_the_native_folds_of_the_engine():
    """Started where the engine's own grid starts, the table holds the engine's problems."""
    m = _market()
    idx = m.prices.index
    period = {"name": "all", "start": str(idx[60].date()), "end": str(idx[-1].date())}
    table = d.build_fold_table(m, period)
    seen = []

    def spy(problem, fold, ctx):
        seen.append((fold, ctx.formation_date, problem))
        return np.full(problem.n, 1.0 / problem.n)

    run_walkforward(m.prices, spy, "spy", "syn", train_days=60, test_days=5, w_max=0.5,
                    select_assets=m.select_assets)
    assert len(seen) == len(table.folds)
    for (fold, date, problem), rec in zip(seen, table.folds, strict=True):
        assert (fold, date) == (rec.fold, rec.formation_date)
        np.testing.assert_array_equal(problem.mu, rec.mu)
        np.testing.assert_array_equal(problem.sigma, rec.sigma)
        assert problem.asset_names == rec.assets


def test_estimation_of_a_fold_ignores_everything_after_its_formation_date():
    m = _market()
    table = d.build_fold_table(m, PERIOD)
    cut = 6
    rec = table.folds[cut]
    row = m.prices.index.get_loc(rec.formation_date)
    tampered = m.prices.copy()
    tampered.iloc[row + 1 :] = tampered.iloc[row + 1 :] * 3.0 + 7.0  # the future changes
    other = replace(m, prices=tampered, select_assets=d.ff49_select_assets(tampered, m.train_days))
    table2 = d.build_fold_table(other, PERIOD)
    for a, b in zip(table.folds[: cut + 1], table2.folds[: cut + 1], strict=True):
        np.testing.assert_array_equal(a.mu, b.mu)
        np.testing.assert_array_equal(a.sigma, b.sigma)
    assert not np.array_equal(table.folds[cut + 3].mu, table2.folds[cut + 3].mu)  # control


def test_fold_record_builds_the_problem_with_the_arm_parameters():
    table = d.build_fold_table(_market(), PERIOD)
    rec = table.folds[3]
    problem = rec.problem(lam=7.0, u=0.5, cardinality=4)
    assert problem.risk_aversion == 7.0 and problem.w_max == 0.5 and problem.cardinality == 4
    assert problem.w_min == 0.0 and problem.asset_names == rec.assets
    np.testing.assert_array_equal(problem.mu, rec.mu)


# --------------------------------------------------------------------------------------------
# The FF49 universe rule: no gaps in the training window
# --------------------------------------------------------------------------------------------


def _gapped_prices(n_days=120, start="2000-01-03"):
    idx = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, 4)), axis=0), index=idx,
        columns=["A", "B", "C", "D"],
    )
    prices.loc[idx[50:53], "B"] = np.nan  # a gap inside some training windows
    prices.loc[idx[:30], "C"] = np.nan  # an industry that starts late
    prices.loc[idx[100:], "D"] = np.nan  # a gap in the future of early folds
    return prices


def test_ff49_selection_keeps_industries_without_gaps_in_the_training_window():
    prices = _gapped_prices()
    select = d.ff49_select_assets(prices, train_days=40)
    idx = prices.index
    assert select(idx[39]) == ["A", "B", "D"]       # C is missing in rows 0-29, inside [0, 39]
    assert select(idx[49]) == ["A", "B", "D"]       # window 10..49: C's gap is still inside
    assert select(idx[52]) == ["A", "D"]            # window 13..52: B has a gap at 50-52
    assert select(idx[69]) == ["A", "C", "D"]       # window 30..69: C is whole, B still gapped
    assert select(idx[92]) == ["A", "B", "C", "D"]  # window 53..92: the gap of B is behind
    assert select(idx[99]) == ["A", "B", "C", "D"]  # D's gap starts at 100: the future
    assert select(idx[110]) == ["A", "B", "C"]      # window 71..110: D has a gap


def test_ff49_selection_is_a_function_of_the_past_only():
    prices = _gapped_prices()
    select = d.ff49_select_assets(prices, train_days=40)
    changed = prices.copy()
    changed.iloc[60:, :] = np.nan  # everything after row 59 disappears
    select2 = d.ff49_select_assets(changed, train_days=40)
    for pos in (39, 45, 52, 59):
        assert select(prices.index[pos]) == select2(prices.index[pos])


def test_selection_is_memoised_and_returns_copies():
    prices = _gapped_prices()
    select = d.ff49_select_assets(prices, train_days=40)
    day = prices.index[60]
    first = select(day)
    assert first == ["A", "D"]  # window 21..60: B (50-52) and C (21-29) have gaps
    first.append("ZZZ")  # the caller may do what it likes with its list
    assert select(day) == ["A", "D"]


def test_gapped_market_folds_use_only_complete_industries():
    m = _market(gap_fraction=0.005, n_assets=12)
    table = d.build_fold_table(m, PERIOD)
    sizes = {len(f.assets) for f in table.folds}
    assert min(sizes) >= 1 and max(sizes) <= 12
    for f in table.folds:
        train = m.prices.loc[f.train_start : f.formation_date, list(f.assets)]
        assert train.notna().all().all()  # every selected industry is complete in the window


# --------------------------------------------------------------------------------------------
# The S&P universe without the irregular-ratio days: per fold (audit G1, D-3)
# --------------------------------------------------------------------------------------------

T_DAYS, S_DAYS = 60, 5  # a short training and test window: the boundaries are easy to read


class _Dataset:
    """What ``_sp500_market`` reads of a ``cif.data.sp500_pit`` dataset: adjusted prices, coverage and the liquidity
    order (``X`` first). No market data."""

    def __init__(self, index, tickers):
        rng = np.random.default_rng(7)
        steps = rng.normal(0, 0.01, (len(index), len(tickers)))
        self.prices_adj = pd.DataFrame(100.0 * np.exp(np.cumsum(steps, 0)), index=index, columns=tickers)
        self.coverage = pd.DataFrame({"coverage": 1.0}, index=index)
        self._order = list(tickers)

    def candidates(self, formation_date, **kwargs):
        return pd.DataFrame({"median_dollar_volume": np.arange(len(self._order), 0, -1)}, index=self._order)

    def select(self, formation_date, n=100, **kwargs):
        return self._order[:n]


def _us_market(tmp_path, monkeypatch, days, variant="no_irregular_ratio"):
    """The ``us`` market of a stub dataset whose ``pseudo_split_days`` are ``days`` = ``[(ticker, row, corrected)]``."""
    import cif.data.sp500_pit as sp500_pit

    index = pd.bdate_range("2010-01-04", periods=400)
    monkeypatch.setattr(sp500_pit, "load_dataset", lambda path: _Dataset(index, ["X", "A", "B", "C"]))
    table = pd.DataFrame([(t, index[r], c) for t, r, c in days], columns=["ticker", "date", "corrected"])
    table.to_parquet(tmp_path / "pseudo_split_days.parquet")
    rf = tmp_path / "rf.parquet"
    pd.DataFrame({"r_daily": 0.0001}, index=index).to_parquet(rf)
    spec = {"name": "us", "source": "sp500_pit", "data_dir": str(tmp_path), "rf_path": str(rf), "variant": variant,
            "n_universe": 3, "liquidity_window": 252, "u": 0.4, "train_days": T_DAYS, "test_days": S_DAYS}
    return d.load_market(spec), index


@pytest.mark.parametrize(
    ("offset", "dropped"),
    [
        (-S_DAYS - 1, False),  # the test window ends the day before the irregular day
        (-S_DAYS, True),  # ... on it: the last day of the test window
        (0, True),  # the formation date is the irregular day
        (T_DAYS - 2, True),  # the day is the first RETURN of the training window (T_DAYS prices, T_DAYS - 1 returns)
        (T_DAYS - 1, False),  # the day is the first price of the window; its return is not in the estimate
    ],
)
def test_a_ticker_leaves_exactly_the_folds_whose_windows_hold_its_irregular_day(tmp_path, monkeypatch, offset, dropped):
    row = 200
    market, index = _us_market(tmp_path, monkeypatch, [("X", row, False)])
    universe = market.select_assets(index[row + offset])  # a fold formed `offset` days after the irregular day
    assert ("X" not in universe) is dropped
    assert len(universe) == 3  # the universe is cut to n_universe AFTER the exclusion: A, B, C move up


def test_a_corrected_day_does_not_take_the_ticker_out(tmp_path, monkeypatch):
    market, index = _us_market(tmp_path, monkeypatch, [("X", 200, True)])  # `adjcorr-2` has fixed this one
    assert market.select_assets(index[200]) == ["X", "A", "B"]


def test_the_irregular_day_of_another_ticker_does_not_matter(tmp_path, monkeypatch):
    market, index = _us_market(tmp_path, monkeypatch, [("B", 200, False)])
    assert market.select_assets(index[200]) == ["X", "A", "C"]
    assert market.select_assets(index[100]) == ["X", "A", "B"]


def test_the_base_variant_ignores_the_irregular_days(tmp_path, monkeypatch):
    market, index = _us_market(tmp_path, monkeypatch, [("X", 200, False)], variant="base")
    assert market.select_assets(index[200]) == ["X", "A", "B"]


# --------------------------------------------------------------------------------------------
# The synthetic market: a factor model with a known covariance
# --------------------------------------------------------------------------------------------


def test_synthetic_market_is_deterministic_and_has_the_requested_shape():
    a, b = _market(), _market()
    pd.testing.assert_frame_equal(a.prices, b.prices)
    assert a.prices.shape == (300, 10)
    assert a.prices.index.is_monotonic_increasing and a.prices.notna().all().all()
    assert a.rf is not None and len(a.rf) == 300 and (a.rf >= 0).all()
    c = _market(seed=1001)
    assert not a.prices.equals(c.prices)


def test_synthetic_returns_follow_the_factor_covariance():
    m = _market(n_days=20_000, n_assets=8)
    r = m.prices.pct_change().dropna().to_numpy()
    true = m.meta["sigma"] / 252
    est = np.cov(r.T)
    err = np.max(np.abs(est - true)) / np.max(np.abs(true))
    assert err < 0.12  # sampling error of 20 000 draws
    # the market factor makes every pair positively correlated: a known structure
    corr = np.corrcoef(r.T)
    assert (corr[np.triu_indices(8, 1)] > 0).mean() > 0.9


def test_market_calendar_for_cells_reads_only_the_dates():
    spec = copy.deepcopy(MARKET)
    idx = d.market_calendar(spec)
    assert idx.equals(_market().prices.index)


# --------------------------------------------------------------------------------------------
# Strategy paths through the engine
# --------------------------------------------------------------------------------------------


def test_run_path_is_the_engine_with_the_markets_settings():
    m = _market()
    table = d.build_fold_table(m, PERIOD)

    def equal(fold, problem):
        return np.full(problem.n, 1.0 / problem.n)

    res = d.run_path(table, equal, tc_bps=0.0, name="eq")
    assert res.summary["n_folds"] == len(table.folds)
    assert len(res.daily_returns) == len(table.folds) * 5
    assert res.daily_returns.index[0] == table.folds[0].test_start
    assert res.daily_returns.index[-1] == table.folds[-1].test_end
    # a manual check of the first fold: equal weights, buy and hold, no costs
    f0 = table.folds[0]
    window = m.prices.loc[f0.test_start : f0.test_end, list(f0.assets)]
    start = m.prices.loc[f0.formation_date, list(f0.assets)]
    value = (window / start).mean(axis=1)  # equal weights bought at the formation close
    manual = value.pct_change()
    manual.iloc[0] = value.iloc[0] - 1.0
    np.testing.assert_allclose(res.daily_returns.iloc[:5].to_numpy(), manual.to_numpy(), rtol=1e-12)


def test_run_path_costs_and_execution_lag_pass_through():
    table = d.build_fold_table(_market(), PERIOD)

    def equal(fold, problem):
        return np.full(problem.n, 1.0 / problem.n)

    free = d.run_path(table, equal, tc_bps=0.0, name="eq")
    paid = d.run_path(table, equal, tc_bps=10.0, name="eq")
    assert (paid.daily_returns <= free.daily_returns + 1e-15).all()
    assert paid.daily_returns.sum() < free.daily_returns.sum()
    lagged = d.run_path(table, equal, tc_bps=0.0, execution_lag=1, name="eq")
    assert lagged.summary["execution_lag"] == 1
    assert not np.allclose(lagged.daily_returns.to_numpy(), free.daily_returns.to_numpy())


def test_run_path_rejects_weights_for_another_universe():
    table = d.build_fold_table(_market(), PERIOD)
    with pytest.raises(ValueError, match="weights"):
        d.run_path(table, lambda fold, problem: np.ones(problem.n + 1), tc_bps=0.0, name="bad")


def test_fold_table_is_cached_per_market_and_period():
    spec = copy.deepcopy(MARKET)
    one = d.get_fold_table(spec, PERIOD)
    two = d.get_fold_table(copy.deepcopy(spec), copy.deepcopy(PERIOD))
    assert one is two
    other = d.get_fold_table({**spec, "seed": 1001}, PERIOD)
    assert other is not one


def test_json_safe_makes_rows_serialisable():
    row = {"a": np.float64(1.5), "b": [np.int64(2), float("nan")], "c": np.array([1.0, np.inf]),
           "d": {"e": pd.Timestamp("2000-03-01"), "f": (1, 2)}, 7: True}
    clean = d.json_safe(row)
    assert clean == {"a": 1.5, "b": [2, None], "c": [1.0, None],
                     "d": {"e": "2000-03-01", "f": [1, 2]}, "7": True}
    json.dumps(clean, allow_nan=False)
