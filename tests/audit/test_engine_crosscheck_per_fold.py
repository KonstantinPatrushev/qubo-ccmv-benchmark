"""Independent cross-check of the per-fold walk-forward additions (audit 2026-10-08).

PASSING check, not a finding: ``run_walkforward`` with ``select_assets``, a per-asset ``tradable``
mask and ``execution_lag`` is compared with a separate simulator that keeps share counts and cash.
The simulator implements the documented rules, not the engine's weight formulas:

* a fold's universe changes, holdings outside it are sold at the trade;
* a missing price freezes an asset (valued at its last price, the move across the gap is earned
  when it resumes); an asset whose series has ended goes to cash at its last price on the first day
  without a price (with the engine's knowledge of the end, see W-2);
* frozen assets keep their holding, the rest of the fund goes to tradable assets in proportion to
  the target (O-4); the cost ``rate × traded notional`` scales the whole fund (engine convention);
* ``execution_lag = 1``: the trade is made at the close of the first open day after formation;
* costs of a close-of-day trade are booked in the next day's return (same day at a window end).

Report: docs/reviews/2026-10-08-audit-e0-additions.md, section "Проверено и корректно".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.backtest.walkforward import run_walkforward

TRAIN, TEST = 5, 4
RATE_BPS = 25.0


def _panel(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = 37
    px = 40 * np.exp(np.cumsum(rng.normal(0.0, 0.03, size=(n_days, 6)), axis=0))
    p = pd.DataFrame(px, index=pd.bdate_range("2023-03-01", periods=n_days),
                     columns=[f"S{k}" for k in range(6)])
    p.iloc[11:14, 0] = np.nan  # S0: a gap that resumes
    p.iloc[11:, 1] = np.nan  # S1: the series ends inside fold 1's window, while held
    p.iloc[:6, 2] = np.nan  # S2: starts late
    return p


def _universe(p: pd.DataFrame):
    n_folds = (len(p) - TRAIN - TEST) // TEST + 1
    fold_of = {p.index[TRAIN - 1 + TEST * f]: f for f in range(n_folds)}

    def select(day):
        f = fold_of[day]
        base = ["S0", "S1", "S3", "S4"] if f < 2 else ["S0", "S2", "S3", "S5"]
        return base if f % 2 == 0 else [*base[1:], "S4" if "S4" not in base else "S5"]

    return select


def _targets(select, p):
    rng = np.random.default_rng(17)
    n_folds = (len(p) - TRAIN - TEST) // TEST + 1
    return {f: rng.dirichlet(np.ones(len(select(p.index[TRAIN - 1 + TEST * f]))))
            for f in range(n_folds)}


def _simulate(p: pd.DataFrame, select, targets, mask: pd.DataFrame, lag: int) -> pd.Series:
    rate = RATE_BPS / 1e4
    cols = list(p.columns)
    price = np.nan_to_num(p.ffill().to_numpy())  # 0 before a series starts (never held)
    has = p.notna().to_numpy() & mask.reindex_like(p).fillna(True).to_numpy(bool)
    valid = p.notna().to_numpy()
    last = np.array([np.flatnonzero(valid[:, j]).max() for j in range(len(cols))])
    shares, cash = np.zeros(len(cols)), 1.0
    out = []

    def to_cash(t):
        nonlocal cash
        for j in np.flatnonzero((shares > 0) & (last < t)):
            cash += shares[j] * price[t, j]
            shares[j] = 0.0

    def trade(t, target):
        nonlocal shares, cash
        value = cash + shares @ price[t]
        held = shares * price[t] / value
        frozen = ~has[t]
        w = np.where(frozen, held, 0.0)
        free = ~frozen
        if target[free].sum() > 0:
            w[free] = target[free] * (1.0 - w.sum()) / target[free].sum()
        cost = rate * np.abs(w - held).sum()
        after = value * (1.0 - cost)
        shares = np.divide(w * after, price[t], out=np.zeros_like(w), where=price[t] > 0)
        cash = (1.0 - w.sum()) * after

    fold, i = 0, 0
    while i + TRAIN + TEST <= len(p):
        f_pos = i + TRAIN - 1
        window = list(range(i + TRAIN, i + TRAIN + TEST))
        assets = select(p.index[f_pos])
        target = np.zeros(len(cols))
        target[[cols.index(a) for a in assets]] = targets[fold]
        open_ = lambda t: bool(has[t].any())  # noqa: E731
        if lag == 0 and open_(f_pos):
            exec_at = f_pos
        else:
            exec_at = next((t for t in window[max(lag - 1, 0):] if open_(t)), None)
        to_cash(f_pos)
        if exec_at == f_pos:
            trade(f_pos, target)
        for t in window:
            value_close = cash + shares @ price[t]
            to_cash(t)
            if exec_at == t:
                trade(t, target)
            booked_here = exec_at != t or t == window[-1]
            out.append((p.index[t], cash + shares @ price[t] if booked_here else value_close))
        fold += 1
        i += TEST
    values = np.array([v for _, v in out])
    return pd.Series(values / np.concatenate([[1.0], values[:-1]]) - 1.0,
                     index=pd.DatetimeIndex([d for d, _ in out]))


@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("with_mask", [False, True])
def test_per_fold_engine_matches_shares_and_cash_simulator(lag, with_mask):
    p = _panel()
    select = _universe(p)
    targets = _targets(select, p)
    mask = pd.DataFrame(True, index=p.index, columns=p.columns)
    if with_mask:
        mask.iloc[TRAIN - 1 + TEST, 3] = False  # S3 cannot trade on fold 1's formation date
        mask.iloc[TRAIN + TEST, 4] = False  # S4 cannot trade on the lag day of fold 1

    def strategy(problem, fold):
        return targets[fold]

    res = run_walkforward(p, strategy, "x", "toy", train_days=TRAIN, test_days=TEST,
                          transaction_cost_bps=RATE_BPS, select_assets=select,
                          tradable=mask if with_mask else None, execution_lag=lag)
    expected = _simulate(p, select, targets, mask, lag)
    assert len(res.daily_returns) == len(expected)
    np.testing.assert_allclose(res.daily_returns.to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12)
    ended = [e["asset"] for f in res.folds for e in f.info.get("ended_assets", [])]
    frozen = [a for f in res.folds for a in f.info.get("frozen_assets", [])]
    assert ended == ["S1"] and "S0" in frozen  # the special paths are exercised
