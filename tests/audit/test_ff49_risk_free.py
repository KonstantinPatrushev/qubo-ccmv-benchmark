"""Audit F-1 (MINOR, 2026-10-08): the FF49 daily risk-free rate is rounded to 1 bp per day.

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

``data/processed/ff49/rf_daily.parquet`` is the RF column of French's DAILY factors file, which has
two decimals in percent: since 1960 it takes only the values 0.00 … 0.06 %/day, constant within a
month. The implied annual rate therefore jumps between 2.55 % and 5.17 % (0.01 vs 0.02 %/day) with
nothing in between: 2025-06 … 2026-01 at 5.17 %, then 2.55/5.17/2.55 % in 2026-04 … 2026-08;
137 of 800 months since 1960 (17 %) change by more than 2 p.p. from the month before. A month's
rounding error is up to ±0.005 %/day ≈ ±1.3 %/year of excess return.

The first-order effect cancels in the registered RQ3 statistics (differences of Sharpe ratios of
paths with the same rf), but not in levels (Sharpe and CE of the optimum, σ* is unaffected). The
precise daily rate is the monthly one-month T-bill return (French's MONTHLY factors file) spread
over the month's trading days, as French's own note on the daily RF describes.

Data test: skipped without ``data/processed/ff49``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RF = Path(__file__).resolve().parents[2] / "data" / "processed" / "ff49" / "rf_daily.parquet"


@pytest.mark.skipif(not RF.exists(), reason="FF49 data not built (data/processed/ff49)")
def test_f1_daily_risk_free_rate_is_not_rounded_to_one_bp_per_day():
    rf = pd.read_parquet(RF)["rf"].loc["1963-07-01":]  # pilot and confirmatory periods
    in_bp = rf.to_numpy() * 1e4
    on_grid = np.isclose(in_bp, np.round(in_bp), rtol=0.0, atol=1e-9)
    monthly = (1.0 + rf).groupby(rf.index.to_period("M")).first() ** 252 - 1.0
    assert on_grid.mean() < 0.5, (
        f"{on_grid.mean():.0%} of the daily rf values are whole bp/day; implied annual rates "
        f"{sorted(set(np.round(monthly.to_numpy(), 4)))[:8]}"
    )
