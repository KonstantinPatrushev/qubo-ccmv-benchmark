"""Audit of the per-fold walk-forward additions (2026-10-08): findings W-2 … W-5.

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

* W-2 (MINOR, look-ahead): ``run_walkforward`` decides that a series has "ended" from
  ``last_pos``, the last valid price over the WHOLE panel. A holding without a price on a formation
  date is turned into cash if the series never resumes and kept (frozen) if it resumes later, so
  the decision context ``ctx`` (current weights, ``outside_weight``, cash) and the executed trade
  depend on prices dated after the formation date.
* W-3 (MINOR): ``execution_lag = k`` counts panel rows, not trading days; with a closed day inside
  the lag the trade happens after fewer than ``k`` trading days (docstring: "the k-th trading day").
* W-4 (MINOR): duplicate tickers from ``select_assets`` are not rejected; part of the target weight
  silently becomes cash.
* W-5 (MINOR): partial execution (O-4) scales the target of tradable assets up and can exceed
  ``w_max``, the cap of the optimisation problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.backtest.walkforward import run_walkforward

nan = np.nan


def _prices(paths: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(next(iter(paths.values())))
    return pd.DataFrame(paths, index=pd.bdate_range(start, periods=n), dtype=float)


def _equal(problem, fold):
    return np.full(problem.n, 1.0 / problem.n)


def _record_context(store: list):
    def strategy(problem, fold, ctx):
        cash = 1.0 - float(ctx.current_weights.sum()) - ctx.outside_weight
        store.append((ctx.assets, ctx.current_weights.copy(), ctx.outside_weight, cash))
        return np.full(problem.n, 1.0 / problem.n)

    return strategy


def test_w2_decision_context_does_not_depend_on_prices_after_the_formation_date():
    """Two panels identical up to and including the formation date of fold 1 (2024-01-05).

    Asset A has no price on that date. In panel ``resumes`` it trades again on 2024-01-08, in
    panel ``ends`` it never does. Nothing dated on or before 2024-01-05 tells the two apart, so the
    strategy must see the same current portfolio and the fund must make the same trade.
    """
    resumes = _prices({"A": [10, 10, 10, 11, nan, 12, 12], "B": [10] * 7})
    ends = _prices({"A": [10, 10, 10, 11, nan, nan, nan], "B": [10] * 7})
    formation_1 = resumes.index[4]
    pd.testing.assert_frame_equal(resumes.loc[:formation_1], ends.loc[:formation_1])

    seen: dict[str, list] = {"resumes": [], "ends": []}
    results = {}
    for tag, panel in (("resumes", resumes), ("ends", ends)):
        results[tag] = run_walkforward(
            panel, _record_context(seen[tag]), tag, "toy", train_days=3, test_days=2,
            transaction_cost_bps=0.0,
            select_assets=lambda d, p=panel: ["A", "B"] if d == p.index[2] else ["B"],
        )
    (_, cur_r, out_r, cash_r), (_, cur_e, out_e, cash_e) = seen["resumes"][1], seen["ends"][1]
    # observed: resumes -> A held outside the universe (0.524), no cash;
    #           ends    -> A already turned into cash (0.524) on 2024-01-05
    assert out_r == pytest.approx(out_e) and cash_r == pytest.approx(cash_e), (
        f"ctx at {formation_1.date()} depends on later prices: outside {out_r:.4f} vs {out_e:.4f}, "
        f"cash {cash_r:.4f} vs {cash_e:.4f}"
    )
    np.testing.assert_allclose(cur_r, cur_e)
    np.testing.assert_allclose(results["resumes"].folds[1].weights, results["ends"].folds[1].weights)


def test_w3_execution_lag_counts_trading_days_not_panel_rows():
    """execution_lag=2 and the first day after formation is closed: the trade must happen at the
    close of the 2nd TRADING day after formation (module docstring), not of the 2nd panel row."""
    p = _prices({"A": [10.0] * 8, "B": [10.0] * 8})
    tradable = pd.Series(True, index=p.index)
    tradable.iloc[3] = False  # the first day after the formation date (row 2) is a closure
    res = run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=4, tradable=tradable,
                          execution_lag=2)
    f0 = res.folds[0]
    trading_days_waited = int(tradable.loc[f0.formation_date:f0.execution_date].iloc[1:].sum())
    assert trading_days_waited == 2, (
        f"executed on {f0.execution_date.date()}, after {trading_days_waited} trading day(s)"
    )


def test_w4_duplicate_tickers_from_select_assets_are_rejected():
    """A duplicated ticker silently drops part of the target: weights sum to 2/3, the rest is cash."""
    p = _prices({"A": [10, 10, 10, 11, 11], "B": [10] * 5, "C": [10] * 5})
    with pytest.raises(ValueError, match="duplicate"):
        run_walkforward(p, _equal, "eq", "toy", train_days=3, test_days=2,
                        transaction_cost_bps=0.0, select_assets=lambda d: ["A", "B", "B"])


def test_w5_partial_execution_respects_the_weight_cap():
    """D cannot trade on the second formation date. The target is feasible (≤ w_max = 0.3); the
    executed portfolio must stay within the cap (observed: A and B at 0.386)."""
    p = _prices({"A": [10.0] * 7, "B": [10.0] * 7, "C": [10.0] * 7, "D": [10.0] * 7})
    mask = pd.DataFrame(True, index=p.index, columns=p.columns)
    mask.loc[p.index[4], "D"] = False

    def strategy(problem, fold):
        return np.array([0.3, 0.3, 0.3, 0.1]) if fold == 0 else np.array([0.3, 0.3, 0.1, 0.3])

    res = run_walkforward(p, strategy, "cap", "toy", train_days=3, test_days=2, tradable=mask,
                          w_max=0.3, transaction_cost_bps=0.0)
    executed = res.folds[1].weights
    assert executed.max() <= 0.3 + 1e-9, f"executed weights {np.round(executed, 4)} exceed w_max"
