"""A synthetic mini-panel and a Cbonds-shaped table that exercise every path of the dividend pipeline.

Shared by ``test_moex_dividends_build.py`` and ``test_build_moex_dividends.py``. Prices are random
walks with planted ex-day drops; the sheet rows of ``cbonds_rows()`` are numbered 2..18 and each
documents one path (see the comments there).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cif.data import moex
from cif.data import moex_dividends as md

TS = pd.Timestamp
DAYS = pd.bdate_range("2021-01-04", "2023-12-29", name="date")
N = len(DAYS)


def _pos(date) -> int:
    return int(DAYS.get_loc(TS(date)))


def _walk(seed, drops=None, start=100.0, noise=0.006, beta=1.0):
    r = beta * MARKET_RET + np.random.default_rng(seed).normal(0.0, noise, N)
    for date, extra in (drops or {}).items():
        r[_pos(date)] += extra
    r[0] = 0.0
    return pd.Series(start * np.cumprod(1 + r), index=DAYS)


MARKET_RET = np.random.default_rng(1).normal(0.0, 0.008, N)
MARKET_RET[0] = 0.0
IMOEX = pd.Series(3000.0 * np.cumprod(1 + MARKET_RET), index=DAYS)

SBER_DROPS = {"2021-03-16": -0.05, "2021-06-15": -0.04, "2022-06-13": -0.06, "2022-09-13": -0.05}
SBER = _walk(2, SBER_DROPS)
SBER.loc["2021-08-12"] = np.nan  # a halt: no quote on one day
GMKN = _walk(3, {"2021-12-14": -0.05})              # adjusted basis (after the 1:100 split of 2022-06-01)
TCSG = _walk(4, {"2022-03-15": -0.05}).where(DAYS <= "2022-12-30")
T = _walk(5, {"2023-03-14": -0.05}).where(DAYS >= "2023-01-03")
NEWC = _walk(6).where(DAYS >= "2022-06-01")  # a new listing with no predecessor
SPLIT_DAY = TS("2022-06-01")
GMKN_TRADED = GMKN.where(DAYS >= SPLIT_DAY, GMKN * 100.0)

ADJ = pd.DataFrame({"SBER": SBER, "GMKN": GMKN, "TCSG": TCSG, "T": T, "NEWC": NEWC})
CLOSE = ADJ.assign(GMKN=GMKN_TRADED)
SPLITS = pd.DataFrame({
    "secid": ["GMKN", "T"], "tradedate": [SPLIT_DAY, TS("2026-04-17")], "before": [1, 1], "after": [100, 10],
    "applied": [True, False],
})
UNIVERSE = pd.DataFrame({
    "secid": ["SBER", "GMKN", "TCSG", "T", "NEWC"],
    "index_from": [TS("2020-01-01")] * 2 + [TS("2020-01-01"), TS("2023-02-01"), TS("2022-06-01")],
    "index_till": [TS("2025-12-31")] * 2 + [TS("2022-12-30"), TS("2025-12-31"), TS("2025-12-31")],
})


def make_panel() -> md.Panel:
    cal = moex.build_calendar(DAYS, "2021-01-01", "2023-12-31")
    levels = pd.DataFrame({"IMOEX": IMOEX, "MCFTR": IMOEX * 2.0})
    flow = pd.Series(0.0, index=DAYS, name="dividend_flow")
    mem = pd.DataFrame({"date": [DAYS[0]] * 2, "secid": ["SBER", "GMKN"], "weight": [0.6, 0.4]})
    return md.Panel(close=CLOSE, adj=ADJ, calendar=cal, splits=SPLITS, universe=UNIVERSE, membership=mem,
                    index_levels=levels, index_flow=flow)


def _row(sheet_row, tickers, amount, record, ex, ccy="RUB", ptype="год", period="2020", name="X"):
    return {"name": name, "exchange": "MOEX", "tickers": tickers, "sector": "", "amount": amount, "currency": ccy,
            "yield_pct_current": np.nan, "last_buy_date": TS(ex) - pd.Timedelta(days=1), "ex_date_cbonds": TS(ex),
            "record_date": TS(record), "period": period, "period_type": ptype, "sheet_row": sheet_row}


def _prev(series, date):
    return float(series.iloc[_pos(date) - 1])


def cbonds_rows() -> pd.DataFrame:
    rows = [
        # 2: normal; rule ex = day before R (T+2) = 2021-03-16, the vendor agrees
        _row(2, "SBER", 0.05 * _prev(SBER, "2021-03-16"), "2021-03-17", "2021-03-16"),
        # 3: GMKN, Cbonds amount for today's (1:100) share count; declared = 100 x that
        _row(3, "GMKN, RU_GMKN", 0.05 * _prev(GMKN, "2021-12-14"), "2021-12-15", "2021-12-14"),
        _row(4, "ZZZZ", 1.0, "2021-04-14", "2021-04-13"),
        _row(5, "", 1.0, "2021-04-14", "2021-04-13"),
        _row(6, "SBER", 1.0, "2026-06-17", "2026-06-16"),
        # 7, 8: the same payment twice (generic type first)
        _row(7, "SBER", 0.04 * _prev(SBER, "2021-06-15"), "2021-06-16", "2021-06-15", ptype="регулярные"),
        _row(8, "SBER", 0.04 * _prev(SBER, "2021-06-15"), "2021-06-16", "2021-06-15", ptype="3 месяца (I кв.)"),
        _row(9, "SBER", 1.0, "2021-09-15", "2021-09-14", ccy="USD"),
        # 10, 11: the GDR TCSG and the share T under one ticker cell; the 1:10 split of T (2026) restates both
        _row(10, "T, TCSG", 0.05 * _prev(TCSG, "2022-03-15") / 10, "2022-03-16", "2022-03-15"),
        _row(11, "T, TCSG", 0.05 * _prev(T, "2023-03-14") / 10, "2023-03-15", "2023-03-14"),
        # 12: the vendor's ex-date is a day earlier and the price fell on it
        _row(12, "SBER", 0.06 * _prev(SBER, "2022-06-13"), "2022-06-15", "2022-06-13"),
        # 13: the vendor's ex-date is a day earlier but the price fell on the rule's date
        _row(13, "SBER", 0.05 * _prev(SBER, "2022-09-13"), "2022-09-14", "2022-09-12"),
        # 14: before the panel starts
        _row(14, "SBER", 1.0, "2020-06-17", "2020-06-16"),
        # 15: T before it joined the index (index_from 2023-02-01); a 5 % dividend that the price never paid
        _row(15, "T", 0.05 * _prev(T, "2023-01-18") / 10, "2023-01-19", "2023-01-18"),
        # 16: the vendor's ex-date is a Saturday; the dividend is too small for the price to arbitrate
        _row(16, "SBER", 0.01, "2021-11-15", "2021-11-13"),
        # 17: T on its first day of trading: the day before has no quote of T, but TCSG (its predecessor) had one
        _row(17, "T", 0.5, "2023-01-04", "2023-01-03"),
        # 18: a USD payment whose vendor ex-date is a day earlier than the rule's: no price test without a rate
        _row(18, "SBER", 1.0, "2021-10-13", "2021-10-11", ccy="USD"),
        # 19: the ex-date (rule: the day before the record date) is a day SBER did not trade
        _row(19, "SBER", 0.5, "2021-08-13", "2021-08-12"),
        # 20: TCSG after its last quote (2022-12-30): the GDR no longer trades
        _row(20, "TCSG", 0.5, "2023-02-02", "2023-02-01"),
        # 21: a new listing on its first day of trading: nothing before it, no predecessor
        _row(21, "NEWC", 0.5, "2022-06-02", "2022-06-01"),
    ]
    return pd.DataFrame(rows)
