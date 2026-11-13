"""Tests for the corrected walk-forward engine (R1, R2; external findings 1–4, 12).

Hand-computed examples check: daily mark-to-market with weight drift, the first test-day return
(close of formation day → next close) is counted, turnover is measured against DRIFTED weights,
costs are charged on traded notional, risk-free excess returns, market-closure postponement,
and that strategies receive the drifted pre-trade weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.backtest.walkforward import StrategyContext, run_walkforward


def _prices(paths: dict[str, list[float]], start="2021-01-04") -> pd.DataFrame:
    n = len(next(iter(paths.values())))
    return pd.DataFrame(paths, index=pd.bdate_range(start, periods=n))


def _equal(problem, fold):
    return np.full(problem.n, 1.0 / problem.n)


def _fixed(w):
    return lambda problem, fold: np.asarray(w, dtype=float)


def test_buy_and_hold_with_drift_and_first_day_counted():
    # train=3 rows (formation at row 2 close), test=2 rows, single fold
    p = _prices({"A": [10, 10, 10, 11, 12.1], "B": [10, 10, 10, 10, 9]})
    res = run_walkforward(p, _fixed([0.5, 0.5]), "fixed", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0)
    assert res.summary["n_folds"] == 1
    # Day 1: A +10 %, B 0 % -> portfolio +5 %; weights drift to (5.5/10.5, 5/10.5)
    # Day 2: A +10 %, B −10 % -> portfolio = (5.5*1.1 + 5*0.9)/10.5 − 1
    r1 = 0.05
    r2 = (5.5 * 1.1 + 5.0 * 0.9) / 10.5 - 1.0
    np.testing.assert_allclose(res.daily_returns.to_numpy(), [r1, r2])
    assert len(res.daily_returns) == 2  # the first test day is not dropped (finding 2)
    # end of window: A = 5.5·1.1 = 6.05, B = 5·0.9 = 4.5, total 10.55
    np.testing.assert_allclose(res.folds[0].end_weights, [6.05 / 10.55, 4.5 / 10.55])


def test_costs_on_traded_notional_and_turnover_vs_drifted_weights():
    # two folds of length 2; 1/N on two assets that diverge
    p = _prices({"A": [10, 10, 10, 12, 12, 12, 12], "B": [10, 10, 10, 10, 10, 10, 10]})
    tc = 10.0
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=tc)
    f0, f1 = res.folds
    # first fold buys from cash: traded = 1.0, cost = tc·1.0
    assert f0.traded == pytest.approx(1.0)
    assert f0.transaction_cost == pytest.approx(tc / 1e4)
    # end of fold 0: A grew 20 % -> drifted weights (6/11, 5/11); rebalance back to 1/2
    drifted = np.array([6.0 / 11.0, 5.0 / 11.0])
    np.testing.assert_allclose(f1.pre_trade_weights, drifted)
    expected_traded = np.abs(np.array([0.5, 0.5]) - drifted).sum()
    assert f1.traded == pytest.approx(expected_traded)
    assert f1.turnover == pytest.approx(0.5 * expected_traded)
    assert f1.transaction_cost == pytest.approx(tc / 1e4 * expected_traded)
    # 1/N pays for rebalancing in every fold after the first (finding 3)
    assert f1.turnover > 0


def test_cost_is_applied_on_first_day_of_window():
    p = _prices({"A": [10, 10, 10, 10, 10], "B": [10, 10, 10, 10, 10]})
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, transaction_cost_bps=25.0)
    np.testing.assert_allclose(res.daily_returns.to_numpy(), [-25e-4, 0.0])
    res0 = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, transaction_cost_bps=0.0)
    assert float(res0.equity_curve.iloc[-1]) == pytest.approx(1.0)


def test_one_way_cost_convention_halves_costs():
    p = _prices({"A": [10, 10, 10, 12, 12, 12, 12], "B": [10, 10, 10, 10, 10, 10, 10]})
    full = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, transaction_cost_bps=10.0)
    half = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, transaction_cost_bps=10.0,
                           cost_convention="one_way")
    for a, b in zip(full.folds, half.folds, strict=True):
        assert b.transaction_cost == pytest.approx(0.5 * a.transaction_cost)


def test_number_of_daily_returns_equals_folds_times_test_days():
    rng = np.random.default_rng(0)
    n = 300
    p = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 4)), axis=0)) * 100,
                     index=pd.bdate_range("2020-01-01", periods=n), columns=list("ABCD"))
    res = run_walkforward(p, _equal, "eq", "toy", train_days=100, test_days=21)
    assert len(res.daily_returns) == res.summary["n_folds"] * 21
    assert res.daily_returns.index.is_monotonic_increasing
    assert res.daily_returns.index[0] == p.index[100]  # first day after the formation close


def test_risk_free_excess_returns_in_summary():
    rng = np.random.default_rng(1)
    n = 200
    p = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=(n, 3)), axis=0)) * 100,
                     index=pd.bdate_range("2020-01-01", periods=n), columns=list("ABC"))
    rf = pd.Series(0.0002, index=p.index)
    res = run_walkforward(p, _equal, "eq", "toy", train_days=50, test_days=10, rf=rf,
                          transaction_cost_bps=0.0)
    r = res.daily_returns
    ex = r - 0.0002
    assert res.summary["rf_used"] is True
    assert res.summary["sharpe"] == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    dd = np.sqrt(np.mean(np.minimum(ex.to_numpy(), 0) ** 2))
    assert res.summary["sortino"] == pytest.approx(ex.mean() / dd * np.sqrt(252))


def test_strategy_receives_drifted_pre_trade_weights():
    seen: list[StrategyContext] = []

    def strat(problem, fold, ctx: StrategyContext):
        seen.append(ctx)
        return np.array([0.5, 0.5])

    p = _prices({"A": [10, 10, 10, 12, 12, 12, 12], "B": [10, 10, 10, 10, 10, 10, 10]})
    run_walkforward(p, strat, "ctx", "toy", train_days=3, test_days=2, transaction_cost_bps=0.0)
    np.testing.assert_allclose(seen[0].current_weights, [0.0, 0.0])
    np.testing.assert_allclose(seen[1].current_weights, [6 / 11, 5 / 11])
    assert seen[1].formation_date == pd.Timestamp(p.index[4])


def test_closure_idiom_with_defaulted_params_is_called_without_context():
    calls = []

    def make(w):
        def inner(problem, fold, _w=w, _tag="x"):
            calls.append((_w, _tag))
            return np.asarray(_w, dtype=float)
        return inner

    p = _prices({"A": [10, 10, 10, 11, 11], "B": [10, 10, 10, 10, 10]})
    run_walkforward(p, make([0.5, 0.5]), "closure", "toy", train_days=3, test_days=2)
    assert calls == [([0.5, 0.5], "x")]


def test_closed_market_postpones_rebalance():
    p = _prices({"A": [10, 10, 10, 11, 11, 11, 11, 11], "B": [10, 10, 10, 10, 10, 10, 10, 10]})
    tradable = pd.Series(True, index=p.index)
    tradable.iloc[4] = False  # formation date of fold 2 is a closure day
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, tradable=tradable,
                          transaction_cost_bps=10.0)
    f1 = res.folds[1]
    assert f1.formation_date == p.index[4]
    assert f1.execution_date == p.index[5]
    # until execution the fund keeps the drifted fold-1 portfolio, so day p.index[5]'s return is
    # earned on the old weights and the cost is charged after it
    assert res.folds[1].traded > 0


def test_strategy_info_and_fallback_are_recorded():
    def strat(problem, fold):
        return np.full(problem.n, 1.0 / problem.n), {"status": "timelimit", "fallback": fold == 0}

    p = _prices({"A": [10, 10, 10, 11, 11, 11, 11], "B": [10, 10, 10, 10, 10, 10, 10]})
    res = run_walkforward(p, strat, "info", "toy", train_days=3, test_days=2)
    assert res.folds[0].info["status"] == "timelimit"
    assert res.summary["n_fallback"] == 1


def test_weights_must_be_fully_invested():
    p = _prices({"A": [10, 10, 10, 11, 11], "B": [10, 10, 10, 10, 10]})
    with pytest.raises(ValueError, match="sum"):
        run_walkforward(p, _fixed([0.3, 0.3]), "bad", "toy", train_days=3, test_days=2)


def test_legacy_summary_keys_preserved():
    rng = np.random.default_rng(3)
    n = 120
    p = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 3)), axis=0)) * 100,
                     index=pd.bdate_range("2020-01-01", periods=n), columns=list("ABC"))
    res = run_walkforward(p, _equal, "eq", "toy", train_days=40, test_days=20)
    for key in ("n_folds", "years", "final_equity", "total_return", "cagr", "sharpe", "sortino",
                "max_drawdown", "annual_turnover", "annual_costs_bps", "total_turnover",
                "total_costs_frac", "mean_solve_time_s", "tc_bps"):
        assert key in res.summary
    f = res.folds[0]
    for attr in ("fold", "test_start", "test_end", "realised_return", "turnover", "transaction_cost",
                 "solve_time_s", "weights"):
        assert hasattr(f, attr)


def test_per_asset_mask_freezes_untradable_holdings():
    """24.03.2022 on MOEX only 33 stocks traded (audit O-4): untradable holdings keep their drifted
    weight, the rest of the budget goes to tradable assets in proportion to the target."""
    p = _prices({"A": [10, 10, 10, 11, 11, 11, 11], "B": [10, 10, 10, 10, 10, 10, 10],
                 "C": [10, 10, 10, 12, 12, 12, 12]})
    mask = pd.DataFrame(True, index=p.index, columns=p.columns)
    mask.loc[p.index[4], "C"] = False  # C does not trade on the second formation date

    def strat(problem, fold):
        return np.array([1 / 3, 1 / 3, 1 / 3]) if fold == 0 else np.array([0.5, 0.5, 0.0])

    res = run_walkforward(p, strat, "mask", "toy", train_days=3, test_days=2, tradable=mask,
                          transaction_cost_bps=10.0)
    f1 = res.folds[1]
    drifted = np.array([11.0, 10.0, 12.0]) / 33.0
    np.testing.assert_allclose(f1.pre_trade_weights, drifted)
    expected = np.array([0.5 * (1 - drifted[2]), 0.5 * (1 - drifted[2]), drifted[2]])
    np.testing.assert_allclose(f1.weights, expected)
    assert f1.traded == pytest.approx(np.abs(expected - drifted).sum())
    assert f1.info["frozen_assets"] == ["C"]
    assert f1.info["frozen_weight"] == pytest.approx(drifted[2])
    assert f1.execution_date == p.index[4]


def test_per_asset_mask_with_nothing_tradable_postpones_execution():
    p = _prices({"A": [10, 10, 10, 11, 11, 11, 11, 11], "B": [10, 10, 10, 10, 10, 10, 10, 10]})
    mask = pd.DataFrame(True, index=p.index, columns=p.columns)
    mask.loc[p.index[4]] = False  # nothing trades on the formation date of fold 1
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, tradable=mask)
    assert res.folds[1].formation_date == p.index[4]
    assert res.folds[1].execution_date == p.index[5]


def test_execution_lag_trades_k_days_after_the_formation_close():
    """Plan §9 / audit O-9: decided at the formation close, executed at the close of the next day;
    until then the old portfolio (here: cash) is held and the cost hits the day after the trade."""
    p = _prices({"A": [10, 10, 10, 11, 12.1], "B": [10, 10, 10, 10, 10]})
    res = run_walkforward(p, _fixed([0.5, 0.5]), "lag", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=10.0, execution_lag=1)
    f = res.folds[0]
    assert f.formation_date == p.index[2]
    assert f.execution_date == p.index[3]
    assert f.traded == pytest.approx(1.0)
    np.testing.assert_allclose(res.daily_returns.to_numpy(), [0.0, 1.05 * (1 - 1e-3) - 1.0])
    assert res.summary["execution_lag"] == 1


def test_execution_lag_trades_against_weights_drifted_until_execution():
    p = _prices({"A": [10, 10, 10, 12, 12, 12, 12], "B": [10, 10, 10, 10, 10, 10, 10]})
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0, execution_lag=1)
    f1 = res.folds[1]
    assert f1.execution_date == p.index[5]
    # fold 0 bought 1/N at the close of p.index[3] (A = 12), so nothing drifts afterwards
    np.testing.assert_allclose(f1.pre_trade_weights, [0.5, 0.5])
    assert f1.traded == pytest.approx(0.0)


def test_execution_lag_skips_a_closed_day():
    p = _prices({"A": [10, 10, 10, 11, 11, 11], "B": [10, 10, 10, 10, 10, 10]})
    tradable = pd.Series(True, index=p.index)
    tradable.iloc[3] = False  # the lag day is a closure
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=3, tradable=tradable,
                          execution_lag=1)
    assert res.folds[0].execution_date == p.index[4]


def test_execution_lag_must_fit_in_the_test_window():
    p = _prices({"A": [10, 10, 10, 11, 11], "B": [10, 10, 10, 10, 10]})
    with pytest.raises(ValueError, match="execution_lag"):
        run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, execution_lag=2)


# ---------------------------------------------------------------------------------------------
# Per-fold universe (point-in-time panels with gaps and ended series)
# ---------------------------------------------------------------------------------------------


def test_default_mode_still_rejects_missing_prices():
    p = _prices({"A": [10, 10, 10, np.nan, 11], "B": [10, 10, 10, 10, 10]})
    with pytest.raises(ValueError, match="missing values"):
        run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2)


def test_universe_mode_selects_assets_per_fold_and_sells_dropped_holdings():
    p = _prices({"A": [10, 10, 10, 10, 10, 10, 10], "B": [10, 10, 10, 10, 10, 10, 10],
                 "C": [10, 10, 10, 10, 10, 10, 10]})
    chosen = {p.index[2]: ["A", "B"], p.index[4]: ["B", "C"]}
    seen = []

    def strat(problem, fold, ctx):
        seen.append((problem.asset_names, ctx.assets, ctx.current_weights.copy()))
        return np.full(problem.n, 1.0 / problem.n)

    res = run_walkforward(p, strat, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0, select_assets=lambda d: chosen[d])
    f0, f1 = res.folds
    assert f0.assets == ("A", "B") and f1.assets == ("B", "C")
    np.testing.assert_allclose(f0.weights, [0.5, 0.5, 0.0])
    np.testing.assert_allclose(f1.weights, [0.0, 0.5, 0.5])
    assert f1.traded == pytest.approx(1.0)  # sell A 0.5, buy C 0.5
    assert seen[1][0] == ("B", "C") and seen[1][1] == ("B", "C")
    np.testing.assert_allclose(seen[1][2], [0.5, 0.0])  # current weights aligned with the fold
    assert f1.info["outside_weight"] == pytest.approx(0.5)  # A, held but not selected


def test_series_ending_inside_the_window_is_frozen_and_goes_to_cash_at_the_next_formation():
    """Audit W-2. On day 2 the fund cannot know that A's series has ended (it may be a gap), so inside
    the window A is frozen at its last price with zero return, as it was held until the end of fold 0
    (this test used to expect the cash conversion on that day, which needs the future). On the next
    formation date A has no price and is not in the new selection: only then it becomes cash at that
    price, without cost."""
    nan = np.nan
    p = _prices({"A": [10, 10, 10, 11, nan, nan, nan], "B": [10, 10, 10, 10, 10, 12, 12]})
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=10.0,
                          select_assets=lambda d: ["A", "B"] if d == p.index[2] else ["B"])
    f0, f1 = res.folds
    # day 1: A +10 %, B 0 → +5 %; day 2: A has no price → frozen, zero return
    assert res.daily_returns.iloc[0] == pytest.approx(1.05 * (1 - 1e-3) - 1.0)  # bought from cash
    assert res.daily_returns.iloc[1] == pytest.approx(0.0, abs=1e-15)
    assert "ended_assets" not in f0.info
    np.testing.assert_allclose(f0.end_weights, [5.5 / 10.5, 5.0 / 10.5])  # A still held, frozen
    assert f1.info["ended_assets"] == [
        {"asset": "A", "date": p.index[4], "weight": pytest.approx(5.5 / 10.5), "terminal_return": 0.0}
    ]
    np.testing.assert_allclose(f1.pre_trade_weights, [0.0, 5.0 / 10.5])  # A is cash before the trade
    # fold 1 buys B with the cash: traded = 1 − 5/10.5, no sale of A is charged
    assert f1.traded == pytest.approx(1.0 - 5.0 / 10.5)


def test_gap_in_a_selected_series_freezes_it_and_keeps_the_move():
    nan = np.nan
    p = _prices({"A": [10, 10, 10, 10, nan, 12, 12], "B": [10, 10, 10, 10, 10, 10, 10]})
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0, select_assets=lambda d: ["A", "B"])
    f1 = res.folds[1]
    # A has no price on the formation date of fold 1: it cannot be traded, so it is frozen
    assert f1.info["frozen_assets"] == ["A"]
    # the +20 % across the gap is earned on the day trading resumes, not lost
    assert res.daily_returns.loc[p.index[5]] == pytest.approx(0.5 * 0.2)


def test_select_assets_must_return_known_columns():
    p = _prices({"A": [10, 10, 10, 11, 11], "B": [10, 10, 10, 10, 10]})
    with pytest.raises(ValueError, match="unknown"):
        run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                        select_assets=lambda d: ["A", "Z"])
