"""Audit G1, finding D-3 (MINOR, 2026-10-12): the US sensitivity "без бумаг с нерегулярными коэффициентами"
selects the universe with future information.

Report: docs/reviews/2026-10-12-audit-g1.md.

ANALYSIS_PLAN §3 (US, sensitivity): "исключение бумаг с оставшимися днями нерегулярных коэффициентов Yahoo (102 дня)".
``cif.experiments.e3_data._sp500_market`` with ``variant: no_irregular_ratio`` (``configs/E3/us_sens_irregular.yaml``)
builds ONE set of flagged tickers from every uncorrected day of ``pseudo_split_days.parquet`` (2010-02-11 …
2025-11-03) and drops them from the universe of EVERY fold. A stock whose spin-off happens in 2022 is therefore
missing from the 2013-2021 universes too: the fold's universe depends on a corporate event after its formation date
(the checklist's "отбор вселенной по будущей информации"), and the sensitivity mixes the data error it is meant to
isolate with a selection on future spin-offs.

The test builds a stub dataset (no market data is read): ticker X has an uncorrected irregular day on 2022-01-03.
At the formation date 2015-06-30 neither the training window (756 days) nor the test window contains that day, so X
must stay in the candidate order; at 2022-06-30 the training window contains it and X is out (as now).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import cif.data.sp500_pit as sp500_pit
from cif.experiments import e3_data


class _StubData:
    def __init__(self, index: pd.DatetimeIndex, tickers: list[str]) -> None:
        rng = np.random.default_rng(10123)
        self.prices_adj = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(index), len(tickers))), 0)),
                                       index=index, columns=tickers)
        self.coverage = pd.DataFrame({"coverage": 1.0}, index=index)
        self._order = list(tickers)  # liquidity order: X first

    def candidates(self, formation_date, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"median_dollar_volume": np.arange(len(self._order), 0, -1)}, index=self._order)

    def select(self, formation_date, n=100, **kwargs) -> list[str]:
        return self._order[:n]


def _market(tmp_path, monkeypatch):
    index = pd.bdate_range("2010-01-04", "2023-12-29")
    data = _StubData(index, ["X", "A", "B", "C"])
    monkeypatch.setattr(sp500_pit, "load_dataset", lambda path: data)
    pd.DataFrame({"ticker": ["X"], "date": [pd.Timestamp("2022-01-03")], "ratio": [0.5],
                  "corrected": [False]}).to_parquet(tmp_path / "pseudo_split_days.parquet")
    rf = tmp_path / "rf.parquet"
    pd.DataFrame({"r_daily": 0.0001}, index=index).to_parquet(rf)
    spec = {"name": "us", "source": "sp500_pit", "data_dir": str(tmp_path), "rf_path": str(rf),
            "variant": "no_irregular_ratio", "n_universe": 3, "liquidity_window": 252, "u": 0.4,
            "train_days": 756, "test_days": 21}
    return e3_data.load_market(spec)


def test_d3_a_ticker_is_not_dropped_from_folds_before_its_irregular_day(tmp_path, monkeypatch):
    market = _market(tmp_path, monkeypatch)
    later = market.select_assets(pd.Timestamp("2022-06-30"))
    assert "X" not in later  # the window of this fold holds the uncorrected day: excluded (as intended)
    early = market.select_assets(pd.Timestamp("2015-06-30"))
    assert "X" in early, (
        f"universe on 2015-06-30 is {early}: X is dropped because of an irregular day on 2022-01-03, "
        "6.5 years after the formation date (selection on future information)"
    )
