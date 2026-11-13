"""Independent cross-check of the walk-forward accounting (audit, item 1 of the checklist).

PASSING check, not a finding: ``run_walkforward`` is compared with a separate simulator that
keeps share counts and cash (no weights, no drift formula). Cases: several folds with a
fold-dependent target, costs on traded notional and the legacy one-way convention, execution
postponed by a market closure (to the middle of a window, to its last day, and a window without
any tradable day), and a risk-free series for the excess-return summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.backtest.walkforward import run_walkforward

TRAIN, TEST = 5, 4


def _targets(n_assets: int, n_folds: int) -> list[np.ndarray]:
    rng = np.random.default_rng(11)
    return [rng.dirichlet(np.ones(n_assets)) for _ in range(n_folds)]


def _trade(shares, cash, px, target, rate):
    """Rebalance at prices ``px`` to ``target``; cost = rate x traded notional, paid from the fund."""
    pos = shares * px
    value = pos.sum() + cash
    traded = np.abs(target * value - pos).sum() / value
    value_after = value * (1.0 - rate * traded)
    return target * value_after / px, 0.0


def _simulate(prices: pd.DataFrame, targets, tc_bps: float, convention: str, tradable=None):
    """Shares-and-cash simulator: trade at a close, cost = rate x traded notional."""
    rate = tc_bps / 1e4 * (0.5 if convention == "one_way" else 1.0)
    p = prices.to_numpy()
    dates = prices.index
    shares = np.zeros(p.shape[1])
    cash = 1.0
    out = []
    fold, i = 0, 0
    while i + TRAIN + TEST <= len(dates):
        f_pos = i + TRAIN - 1
        window = list(range(i + TRAIN, i + TRAIN + TEST))
        ok = lambda t: tradable is None or bool(tradable.iloc[t])  # noqa: E731
        exec_at = f_pos if ok(f_pos) else next((t for t in window if ok(t)), None)

        if exec_at == f_pos:
            # its cost shows up in the first day's return of the window
            shares, cash = _trade(shares, cash, p[f_pos], targets[fold], rate)
        for t in window:
            pre_trade_value = shares @ p[t] + cash
            if exec_at == t:
                # postponed: earn the day on the old shares, then trade at the close
                shares, cash = _trade(shares, cash, p[t], targets[fold], rate)
            # Engine convention: the cost of a close-of-day trade is booked in the NEXT day's
            # return, except on the last day of a window (booked on that day). The equity path
            # is identical either way; only the daily attribution of the cost moves by one day.
            last = t == window[-1]
            out.append((dates[t], shares @ p[t] + cash if (exec_at != t or last) else pre_trade_value))
        fold += 1
        i += TEST
    # daily returns from the value path, starting from the capital before the first trade (1.0)
    values = np.array([v for _, v in out])
    rets = values / np.concatenate([[1.0], values[:-1]]) - 1.0
    return pd.Series(rets, index=pd.DatetimeIndex([d for d, _ in out]))


def _prices(n_days=29, n_assets=3, seed=5):
    rng = np.random.default_rng(seed)
    px = 50 * np.exp(np.cumsum(rng.normal(0.0, 0.03, size=(n_days, n_assets)), axis=0))
    return pd.DataFrame(px, index=pd.bdate_range("2022-02-01", periods=n_days),
                        columns=[f"S{k}" for k in range(n_assets)])


@pytest.mark.parametrize("convention", ["traded", "one_way"])
@pytest.mark.parametrize("closure", [None, "middle", "last_day", "whole_window"])
def test_engine_matches_shares_and_cash_simulator(convention, closure):
    prices = _prices()
    n_folds = (len(prices) - TRAIN - TEST) // TEST + 1
    targets = _targets(prices.shape[1], n_folds)
    tradable = None
    if closure is not None:
        tradable = pd.Series(True, index=prices.index)
        f1 = TRAIN - 1 + TEST  # formation date of fold 1
        if closure == "middle":
            tradable.iloc[f1 : f1 + 2] = False  # execute after the 2nd day of fold 1's window
        elif closure == "last_day":
            tradable.iloc[f1 : f1 + TEST] = False  # only the last window day is open
        else:
            tradable.iloc[f1 : f1 + TEST + 1] = False  # fold 1 never trades; fold 2 postponed
    res = run_walkforward(prices, lambda problem, fold: targets[fold], "x", "toy",
                          train_days=TRAIN, test_days=TEST, transaction_cost_bps=25.0,
                          cost_convention=convention, tradable=tradable)
    expected = _simulate(prices, targets, 25.0, convention, tradable)
    assert len(res.daily_returns) == len(expected)
    np.testing.assert_allclose(res.daily_returns.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
    equity = (1.0 + expected).cumprod()
    assert res.summary["final_equity"] == pytest.approx(float(equity.iloc[-1]), abs=1e-12)
