"""Audit W-1 (MINOR): the walk-forward engine silently truncates the backtest on missing prices.

Finding W-1 in docs/reviews/2026-10-04-audit-e0.md.

``run_walkforward`` documents "no missing values inside the evaluated range" but does not check
it. ``cif.data.statistics.log_returns`` drops every row with a NaN in ANY column, so a training
window with a delisted (or not yet listed) asset is silently shortened, and once no complete
row is left the loop hits ``if train_returns.empty: break`` and the backtest ENDS without a
warning: the summary then covers fewer folds and a shorter period. Point-in-time panels (E3)
contain such gaps by construction. Expected: a ``ValueError`` naming the offending columns (or a
per-fold universe), never a silent truncation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from cif.backtest.walkforward import run_walkforward


def _equal(problem, fold):
    return np.full(problem.n, 1.0 / problem.n)


def test_missing_prices_raise_instead_of_silently_ending_the_backtest():
    n = 40
    idx = pd.bdate_range("2021-01-04", periods=n)
    rng = np.random.default_rng(0)
    a = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b[20:] = np.nan  # asset B is delisted half-way through the sample
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # pandas pct_change fill_method
        try:
            res = run_walkforward(prices, _equal, "eq", "toy", train_days=10, test_days=5,
                                  transaction_cost_bps=0.0)
        except ValueError:
            return  # acceptable: the precondition is enforced
    # observed: 4 folds and the last test day 2021-02-12, although the panel runs to 2021-02-26
    assert res.summary["n_folds"] == (n - 10 - 5) // 5 + 1, (
        f"backtest silently truncated to {res.summary['n_folds']} folds "
        f"(last day {res.daily_returns.index[-1].date()}, panel ends {idx[-1].date()})"
    )
