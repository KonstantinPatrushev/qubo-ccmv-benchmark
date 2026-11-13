"""Audit M-1 and M-2 (MINOR): risk-free accrual across calendars; drawdown peak label.

Findings M-1 and M-2 in docs/reviews/2026-10-04-audit-e0.md.

M-1. ``cif.metrics.performance.excess_returns`` aligns ``rf`` to the return dates by forward
fill. ``r_daily`` from ``cif.data.riskfree`` is the accrual since the previous trading day of
ITS calendar, so when the two calendars differ the accrual is wrong:

* a day present in ``rf`` but absent from the returns loses its accrual (the legacy MOEX panel
  lacks 24-25.03.2022, so the 27-day closure accrual of RUONIA is dropped);
* a day present in the returns but not in ``rf`` (a holiday or a closure day with a
  forward-filled price) gets the previous day's accrual a second time (2022-01-07, 2022-02-23,
  28.02-04.03.2022 in the legacy MOEX panel).

On the legacy MOEX panel with RUONIA this changes the cumulative rf over the test period by
-0.59 pp (-7 bp a year; excess Sharpe of 1/N -0.105 instead of -0.108). Expected: the rf charged
on a return date is the compounded accrual since the previous return date.

M-2. ``summarise_returns`` measures the drawdown from the initial capital (correct value) but
reports ``mdd_peak`` as the first return date even when the peak is the initial capital at
``start``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.metrics.performance import excess_returns, summarise_returns


def test_rf_accrual_of_a_day_missing_from_the_return_series_is_not_lost():
    rf = pd.Series(0.001, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    r = pd.Series(0.0, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"]))
    charged = -excess_returns(r, rf)
    # 2024-01-05 must carry the accrual of 01-04 and 01-05 (observed: 0.001)
    assert charged.loc["2024-01-05"] == pytest.approx(1.001**2 - 1.0, abs=1e-12)


def test_rf_accrual_is_not_duplicated_on_a_day_missing_from_the_rf_calendar():
    # rf calendar skips the holiday 01-03; the accrual of 01-04 already covers two days
    rf = pd.Series([0.001, 0.002], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    r = pd.Series(0.0, index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    charged = -excess_returns(r, rf)
    total = float(np.prod(1.0 + charged.loc["2024-01-03":].to_numpy()) - 1.0)
    # observed: 0.003 (the 01-02 accrual is forward-filled onto the holiday)
    assert total == pytest.approx(0.002, abs=1e-12)


def test_drawdown_peak_is_the_initial_capital_when_losses_start_on_day_one():
    start = pd.Timestamp("2024-01-01")
    r = pd.Series([-0.10, -0.10, 0.05], index=pd.bdate_range("2024-01-02", periods=3))
    s = summarise_returns(r, start=start)
    assert s["max_drawdown"] == pytest.approx(0.81 - 1.0)  # value is right today
    # observed: mdd_peak = '2024-01-02 00:00:00' (equity 0.9 after the first loss)
    assert pd.Timestamp(s["mdd_peak"]) == start
