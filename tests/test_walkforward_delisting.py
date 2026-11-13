"""Per-fold engine after the audit of 2026-10-08: delisting event, execution lag, partial execution.

Findings W-2 … W-5 (docs/reviews/2026-10-08-audit-e0-additions.md); the red tests of the audit are in
``tests/audit/test_walkforward_per_fold.py``. Here the rules are pinned with hand-computed numbers:

* W-2: a series without a price is frozen at its last level (zero return) inside a window; only a
  holding that has no price on a FORMATION date and is not in the new selection becomes cash at its
  last price, without cost (the delisting event). Nothing dated after the formation date is used.
  ``terminal_return`` (default 0; −0.3 / −1.0 for sensitivity runs) is applied at that conversion.
* W-3: ``execution_lag = k`` counts tradable days.
* W-4: ``select_assets`` may not return a ticker twice.
* W-5: partial execution spreads the budget over tradable assets in proportion to the target, no
  weight above ``w_max`` (water-filling), and what does not fit stays in cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.backtest.walkforward import _execute, run_walkforward

nan = np.nan


def _prices(paths: dict[str, list[float]], start="2021-01-04") -> pd.DataFrame:
    n = len(next(iter(paths.values())))
    return pd.DataFrame(paths, index=pd.bdate_range(start, periods=n), dtype=float)


def _equal(problem, fold):
    return np.full(problem.n, 1.0 / problem.n)


# 7 rows, train 3 / test 2: formation rows 2 and 4, windows (3, 4) and (5, 6).
# A is held 0.5 from row 2, gains 10 % on row 3 and has no price from row 4 on:
# at the formation date of fold 1 (row 4) the drifted weights are A = 5.5/10.5, B = 5/10.5.
W_A, W_B = 5.5 / 10.5, 5.0 / 10.5


def _ended_panel() -> pd.DataFrame:
    return _prices({"A": [10, 10, 10, 11, nan, nan, nan], "B": [10, 10, 10, 10, 10, 12, 12]})


def _select_b_after_first(p: pd.DataFrame):
    return lambda d: ["A", "B"] if d == p.index[2] else ["B"]


# --------------------------------------------------------------------------------------------
# W-2: the delisting event
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a_path", "selection", "case"),
    [
        ([10, 10, 10, 11, nan, 12, 12], ["A", "B"], "frozen_and_selected"),
        ([10, 10, 10, 11, 11, 11, 11], ["B"], "priced_and_dropped"),
        ([10, 10, 10, 11, nan, 12, 12], ["B"], "unpriced_and_dropped"),
    ],
)
def test_only_an_unpriced_holding_outside_the_new_selection_becomes_cash(a_path, selection, case):
    p = _prices({"A": a_path, "B": [10] * 7})
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0,
                          select_assets=lambda d: ["A", "B"] if d == p.index[2] else selection)
    f1 = res.folds[1]
    if case == "frozen_and_selected":
        # no price on the formation date, but still wanted: it cannot be traded, it stays
        assert "ended_assets" not in f1.info and f1.info["frozen_assets"] == ["A"]
        assert f1.info["outside_weight"] == 0.0
        np.testing.assert_allclose(f1.pre_trade_weights, [W_A, W_B])
    elif case == "priced_and_dropped":
        # it can be sold, so it is a trade (charged), not a delisting
        assert "ended_assets" not in f1.info
        assert f1.info["outside_weight"] == pytest.approx(W_A)
        assert f1.traded == pytest.approx(2 * W_A)
    else:
        # unpriced and unwanted: cash at the last price, no trade, whatever the series does later
        assert f1.info["ended_assets"] == [
            {"asset": "A", "date": p.index[4], "weight": pytest.approx(W_A), "terminal_return": 0.0}
        ]
        assert f1.info["outside_weight"] == 0.0
        np.testing.assert_allclose(f1.pre_trade_weights, [0.0, W_B])
        assert f1.traded == pytest.approx(1.0 - W_B)  # only B is bought with the cash


def test_inside_a_window_a_missing_series_is_frozen_and_converted_at_the_next_formation_only():
    p = _prices({"A": [10, 10, 10, 11, nan, nan, nan, nan, nan], "B": [10] * 9})
    chosen = {p.index[2]: ["A", "B"], p.index[4]: ["A", "B"], p.index[6]: ["B"]}
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0, select_assets=lambda d: chosen[d])
    f0, f1, f2 = res.folds
    assert "ended_assets" not in f0.info and "ended_assets" not in f1.info
    np.testing.assert_allclose(f0.end_weights, [W_A, W_B])  # frozen, not cash
    assert f1.info["frozen_assets"] == ["A"]
    np.testing.assert_allclose(f1.weights, [W_A, W_B])  # A cannot be traded, B takes the rest
    np.testing.assert_allclose(res.daily_returns.iloc[1:4].to_numpy(), 0.0, atol=1e-15)  # zero return
    assert f2.info["ended_assets"] == [
        {"asset": "A", "date": p.index[6], "weight": pytest.approx(W_A), "terminal_return": 0.0}
    ]
    assert res.summary["n_ended_assets"] == 1


@pytest.mark.parametrize("terminal_return", [0.0, -0.3, -1.0, 0.1])
def test_terminal_return_is_booked_on_the_first_day_after_the_conversion(terminal_return):
    p = _ended_panel()
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                          transaction_cost_bps=0.0, select_assets=_select_b_after_first(p),
                          terminal_return=terminal_return)
    f1 = res.folds[1]
    loss = W_A * terminal_return  # fund return at the conversion
    mult = 1.0 + loss
    assert f1.info["ended_assets"][0]["terminal_return"] == terminal_return
    if terminal_return == 0.0:
        assert "terminal_loss" not in f1.info
    else:
        assert f1.info["terminal_loss"] == pytest.approx(loss)
    # B is now a larger share of a smaller fund; the rest is cash
    pre_b = W_B / mult
    np.testing.assert_allclose(f1.pre_trade_weights, [0.0, pre_b])
    assert f1.traded == pytest.approx(1.0 - pre_b, abs=1e-12)
    # first day of fold 1: B +20 % on the whole fund after the trade, times the conversion value
    assert res.daily_returns.loc[p.index[5]] == pytest.approx(1.2 * mult - 1.0)
    assert res.daily_returns.loc[p.index[6]] == pytest.approx(0.0, abs=1e-15)
    assert f1.realised_return == pytest.approx(1.2 * mult - 1.0)  # a market loss is not a cost
    # the equity curve carries the conversion exactly once
    first_fold = (1.0 + res.daily_returns.iloc[:2]).prod()
    assert (1.0 + res.daily_returns).prod() == pytest.approx(first_fold * 1.2 * mult)
    assert res.summary["n_ended_assets"] == 1


def test_default_terminal_return_keeps_the_value_of_the_conversion():
    p = _ended_panel()
    kwargs = dict(train_days=3, test_days=2, transaction_cost_bps=10.0,
                  select_assets=_select_b_after_first(p))
    default = run_walkforward(p, _equal, "eq", "toy", **kwargs)
    explicit = run_walkforward(p, _equal, "eq", "toy", terminal_return=0.0, **kwargs)
    np.testing.assert_array_equal(default.daily_returns.to_numpy(), explicit.daily_returns.to_numpy())
    assert default.summary["terminal_return"] == 0.0


@pytest.mark.parametrize("bad", [-1.01, -5.0, float("nan"), float("inf")])
def test_terminal_return_must_be_a_loss_of_at_most_everything(bad):
    p = _ended_panel()
    with pytest.raises(ValueError, match="terminal_return"):
        run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                        select_assets=_select_b_after_first(p), terminal_return=bad)


def test_a_terminal_loss_that_wipes_out_the_whole_fund_is_an_error_not_a_zero_division():
    p = _prices({"A": [10, 10, 10, 11, nan, nan, nan], "B": [10] * 7})
    with pytest.raises(ValueError, match="wipes out"):
        run_walkforward(p, lambda problem, fold: np.ones(problem.n), "all-in", "toy",
                        train_days=3, test_days=2, terminal_return=-1.0,
                        select_assets=lambda d: ["A"] if d == p.index[2] else ["B"])


def test_without_a_per_fold_universe_the_terminal_return_has_nothing_to_do():
    p = _prices({"A": [10, 10, 10, 11, 12], "B": [10, 10, 10, 10, 9]})
    a = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2)
    b = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2, terminal_return=-0.3)
    np.testing.assert_array_equal(a.daily_returns.to_numpy(), b.daily_returns.to_numpy())


# --------------------------------------------------------------------------------------------
# W-3: the lag counts tradable days
# --------------------------------------------------------------------------------------------


def test_execution_lag_skips_closures_in_the_middle_of_the_lag():
    p = _prices({"A": [10.0] * 8, "B": [10.0] * 8})
    tradable = pd.Series(True, index=p.index)
    tradable.iloc[4] = False  # formation at row 2; open days after it: rows 3, 5, 6, 7
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=5, tradable=tradable,
                          execution_lag=3)
    assert res.folds[0].execution_date == p.index[6]  # the 3rd open day (row 5 is only the 3rd row)


def test_execution_lag_counts_the_same_without_closures_as_before():
    p = _prices({"A": [10.0] * 8, "B": [10.0] * 8})
    for lag, row in ((1, 3), (2, 4), (3, 5)):
        res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=5, execution_lag=lag)
        assert res.folds[0].execution_date == p.index[row]


def test_execution_lag_without_enough_open_days_makes_no_trade_in_the_fold():
    p = _prices({"A": [10.0] * 8, "B": [10.0] * 8})
    tradable = pd.Series(True, index=p.index)
    tradable.iloc[3:6] = False  # open days after the formation row 2: only rows 6 and 7
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=5, tradable=tradable,
                          execution_lag=3)
    f0 = res.folds[0]
    assert f0.execution_date is None and f0.traded == 0.0
    np.testing.assert_allclose(f0.end_weights, [0.0, 0.0])  # still cash


def test_closed_formation_date_and_next_day_postpone_to_the_first_open_day():
    p = _prices({"A": [10.0] * 7, "B": [10.0] * 7})
    tradable = pd.Series(True, index=p.index)
    tradable.iloc[2:4] = False  # the formation date and the next day
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=3, tradable=tradable)
    assert res.folds[0].execution_date == p.index[4]


# --------------------------------------------------------------------------------------------
# W-4: duplicate tickers
# --------------------------------------------------------------------------------------------


def test_duplicate_tickers_are_rejected_wherever_they_are_and_named():
    p = _prices({"A": [10, 10, 10, 11, 11], "B": [10] * 5, "C": [10] * 5})
    with pytest.raises(ValueError, match=r"duplicate tickers \['A'\]"):
        run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                        select_assets=lambda d: ["A", "B", "A"])


# --------------------------------------------------------------------------------------------
# W-5: partial execution within the cap
# --------------------------------------------------------------------------------------------


def test_execute_without_a_cap_is_the_proportional_rule():
    target = np.array([0.5, 0.3, 0.2])
    current = np.array([0.2, 0.5, 0.3])
    frozen = np.array([False, False, True])
    out = _execute(target, current, frozen)
    np.testing.assert_allclose(out, [0.7 * 0.5 / 0.8, 0.7 * 0.3 / 0.8, 0.3], rtol=0, atol=1e-15)
    np.testing.assert_allclose(_execute(target, current, frozen, 1.0), out, rtol=0, atol=1e-15)


def test_execute_spreads_what_the_cap_refuses_over_the_other_assets():
    target = np.array([0.3, 0.3, 0.3, 0.1])
    frozen = np.array([False, False, False, True])
    out = _execute(target, np.array([0.25, 0.25, 0.4, 0.1]), frozen, 0.3)
    np.testing.assert_allclose(out, [0.3, 0.3, 0.3, 0.1], atol=1e-12)
    # 0.8 for the tradable assets in proportion 6:3:1 would give A 0.48 > 0.4: A is cut to 0.4 and the
    # surplus goes to B and C in proportion 3:1; the frozen weight is not touched
    out = _execute(np.array([0.6, 0.3, 0.1, 0.0]), np.array([0.0, 0.0, 0.0, 0.2]),
                   np.array([False, False, False, True]), 0.4)
    np.testing.assert_allclose(out, [0.4, 0.3, 0.1, 0.2], atol=1e-12)


def test_execute_keeps_what_does_not_fit_in_cash_and_never_touches_frozen_weights():
    frozen = np.array([False, False, True])
    out = _execute(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 0.2]), frozen, 0.3)
    np.testing.assert_allclose(out, [0.3, 0.3, 0.2])
    assert out.sum() == pytest.approx(0.8)  # 0.2 stays in cash
    # a frozen holding above the cap (it drifted there) is left as it is
    out = _execute(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.0, 0.5]), frozen, 0.3)
    np.testing.assert_allclose(out, [0.25, 0.25, 0.5])


def test_execute_rescales_current_holdings_when_the_target_gives_tradables_nothing():
    frozen = np.array([False, False, True])
    out = _execute(np.array([0.0, 0.0, 1.0]), np.array([0.4, 0.4, 0.2]), frozen, 1.0)
    np.testing.assert_allclose(out, [0.4, 0.4, 0.2])
    out = _execute(np.array([0.0, 0.0, 1.0]), np.array([0.4, 0.4, 0.2]), frozen, 0.3)
    np.testing.assert_allclose(out, [0.3, 0.3, 0.2])


def test_execute_with_nothing_frozen_returns_the_target_unchanged():
    target = np.array([0.7, 0.3])
    out = _execute(target, np.array([0.5, 0.5]), np.zeros(2, dtype=bool), 0.4)
    np.testing.assert_array_equal(out, target)
    assert out is not target
    np.testing.assert_array_equal(_execute(target, np.array([0.5, 0.5]), None, 0.4), target)


def test_partial_execution_that_cannot_stay_within_the_cap_leaves_cash_and_says_so():
    p = _prices({"A": [10.0] * 7, "B": [10.0] * 7, "C": [10.0] * 7})
    mask = pd.DataFrame(True, index=p.index, columns=p.columns)
    mask.loc[p.index[4], "B"] = False  # B cannot trade on the second formation date

    def strategy(problem, fold):
        return np.array([0.45, 0.05, 0.5]) if fold == 0 else np.array([0.4, 0.2, 0.4])

    res = run_walkforward(p, strategy, "cap", "toy", train_days=3, test_days=2, tradable=mask,
                          w_max=0.3, transaction_cost_bps=0.0)
    f1 = res.folds[1]
    np.testing.assert_allclose(f1.weights, [0.3, 0.05, 0.3])  # B frozen at 0.05, A and C capped
    assert f1.info["unallocated_weight"] == pytest.approx(0.35)
    assert f1.info["frozen_assets"] == ["B"]
    assert "unallocated_weight" not in res.folds[0].info


def test_cap_does_not_change_executions_it_does_not_bind():
    p = _prices({"A": [10, 10, 10, 11, 11, 11, 11], "B": [10] * 7, "C": [10, 10, 10, 12, 12, 12, 12]})
    mask = pd.DataFrame(True, index=p.index, columns=p.columns)
    mask.loc[p.index[4], "C"] = False

    def strat(problem, fold):
        return np.array([1 / 3, 1 / 3, 1 / 3]) if fold == 0 else np.array([0.5, 0.5, 0.0])

    loose = run_walkforward(p, strat, "mask", "toy", train_days=3, test_days=2, tradable=mask,
                            transaction_cost_bps=10.0)
    tight = run_walkforward(p, strat, "mask", "toy", train_days=3, test_days=2, tradable=mask,
                            transaction_cost_bps=10.0, w_max=0.6)
    np.testing.assert_allclose(tight.folds[1].weights, loose.folds[1].weights, atol=1e-15)
    assert "unallocated_weight" not in tight.folds[1].info
