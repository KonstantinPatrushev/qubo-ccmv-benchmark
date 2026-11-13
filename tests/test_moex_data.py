"""Tests for ``cif.data.moex`` and the opt-in rebasing in ``cif.data.cleaning`` (WP-E1, R6).

Everything runs offline on synthetic fixtures. Network tests run only with ``CIF_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cif.data import moex
from cif.data.cleaning import CleaningReport, clean_prices, mask_closed_days

needs_network = pytest.mark.skipif(
    os.environ.get("CIF_NETWORK_TESTS") != "1",
    reason="network test: set CIF_NETWORK_TESTS=1 to run",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
V2 = REPO_ROOT / "data" / "processed" / "moex_v2"


def _built(directory: Path) -> bool:
    """The dataset is built locally: its provenance (versioned in git since 2026-10-05, so present
    in every clone) and every processed file the provenance records."""
    prov = directory / "provenance.json"
    if not prov.exists():
        return False
    recorded = json.loads(prov.read_text(encoding="utf-8")).get("files", {}).get("processed", {})
    return bool(recorded) and all((directory / name).exists() for name in recorded)


needs_v2 = pytest.mark.skipif(
    not _built(V2),
    reason="data/processed/moex_v2 is not built (scripts/data/build_moex_dataset.py)",
)


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


# --------------------------------------------------------------------------------------
# ISS client: paging, cache, retries
# --------------------------------------------------------------------------------------


class _Resp:
    def __init__(self, payload=None, status=200, text=None):
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)
        self.content = self.text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")


class _Session:
    """Replays prepared responses and records the query parameters of every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, params=None, **kwargs):
        self.calls.append({"url": url, **(params or {})})
        return self.responses.pop(0)


def _page(rows, total, block="history", columns=("A", "B")):
    return {
        block: {"columns": list(columns), "data": rows},
        f"{block}.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, total, 100]]},
    }


def _client(tmp_path, responses, **kw):
    session = _Session(responses)
    client = moex.IssClient(tmp_path, session=session, sleep=lambda s: None, **kw)
    return client, session


def test_table_follows_cursor_until_total(tmp_path):
    p1 = _page([[i, i] for i in range(100)], total=250)
    p2 = _page([[i, i] for i in range(100, 200)], total=250)
    p3 = _page([[i, i] for i in range(200, 250)], total=250)
    client, session = _client(tmp_path, [_Resp(p1), _Resp(p2), _Resp(p3)])
    table = client.table("history/x", "history", {"from": "2020-01-01"}, cache_name="x/a")
    assert len(table.rows) == 250 and table.pages == 3 and not table.from_cache
    assert [c["start"] for c in session.calls] == [0, 100, 200]
    assert table.frame().shape == (250, 2)


def test_table_without_cursor_stops_on_empty_or_repeated_page(tmp_path):
    rows = [["a", 1], ["b", 2]]
    same = {"tickers": {"columns": ["A", "B"], "data": rows}}
    client, session = _client(tmp_path, [_Resp(same), _Resp(same)])
    table = client.table("x", "tickers", cache_name="x/b")
    # the endpoint ignores ``start`` and repeats the page: it must not loop forever
    assert table.rows == rows and len(session.calls) <= 2


def test_second_call_is_served_from_cache(tmp_path):
    page = _page([[1, 2]], total=1)
    client, session = _client(tmp_path, [_Resp(page)])
    first = client.table("history/x", "history", cache_name="x/c")
    second = client.table("history/x", "history", cache_name="x/c")
    assert len(session.calls) == 1
    assert second.from_cache and second.rows == first.rows
    cached = json.loads((tmp_path / "x" / "c.json").read_text())
    assert cached["request"]["path"] == "history/x" and cached["columns"] == ["A", "B"]


def test_refresh_ignores_cache(tmp_path):
    page = _page([[1, 2]], total=1)
    client, session = _client(tmp_path, [_Resp(page), _Resp(page)], refresh=True)
    client.table("history/x", "history", cache_name="x/d")
    client.table("history/x", "history", cache_name="x/d")
    assert len(session.calls) == 2


def test_offline_mode_reads_the_cache_and_refuses_the_network(tmp_path):
    page = _page([[1, 2]], total=1)
    online, _ = _client(tmp_path, [_Resp(page)])
    online.table("history/x", "history", cache_name="x/h")
    session = _Session([])
    offline = moex.IssClient(tmp_path, session=session, offline=True, sleep=lambda s: None)
    assert offline.table("history/x", "history", cache_name="x/h").rows == [[1, 2]]
    with pytest.raises(moex.IssError, match="offline"):
        offline.table("history/x", "history", cache_name="x/missing")
    assert session.calls == []


def test_retry_after_server_error(tmp_path):
    page = _page([[1, 2]], total=1)
    client, session = _client(tmp_path, [_Resp(status=503, text="busy"), _Resp(page)])
    table = client.table("history/x", "history", cache_name="x/e")
    assert table.rows == [[1, 2]] and len(session.calls) == 2


def test_html_instead_of_json_is_an_error(tmp_path):
    # the CCI endpoints answer 200 with an HTML "subscribers only" page
    html = _Resp(text="<html>Информация доступна только подписчикам</html>")
    client, _ = _client(tmp_path, [html] * 3, retries=3)
    with pytest.raises(moex.IssError, match="not JSON"):
        client.table("cci/corp-actions/dividends", "dividends", cache_name="x/f")


def test_missing_block_is_an_error_not_an_empty_table(tmp_path):
    # /securities/SBER/dividends.json silently returns the security description instead
    wrong = {"description": {"columns": ["name"], "data": [["SECID"]]}}
    client, _ = _client(tmp_path, [_Resp(wrong)])
    with pytest.raises(moex.IssError, match="no block 'dividends'"):
        client.table("securities/SBER/dividends", "dividends", cache_name="x/g")


# --------------------------------------------------------------------------------------
# Trading calendar from the index history
# --------------------------------------------------------------------------------------


def _imoex_days_2022() -> pd.DatetimeIndex:
    """IMOEX was calculated on these days around the 2022 closure (28.02-23.03 closed)."""
    days = list(_bdays("2022-02-14", "2022-02-25")) + list(_bdays("2022-03-24", "2022-04-08"))
    days.remove(pd.Timestamp("2022-02-23"))  # Defender of the Fatherland Day
    return pd.DatetimeIndex(days)


def test_calendar_marks_closure_non_tradable():
    cal = moex.build_calendar(_imoex_days_2022(), "2022-02-14", "2022-04-08")
    assert cal.index.is_monotonic_increasing
    assert len(cal) == (pd.Timestamp("2022-04-08") - pd.Timestamp("2022-02-14")).days + 1
    closed = cal.loc["2022-02-28":"2022-03-23"]
    assert not closed["tradable"].any()
    assert cal.loc["2022-02-25", "tradable"] and cal.loc["2022-03-24", "tradable"]
    assert not cal.loc["2022-02-23", "tradable"]  # official holiday
    assert not cal.loc["2022-02-19", "tradable"]  # Saturday


def test_index_record_without_turnover_is_not_a_trading_day():
    # ISS has an IMOEX level for 2022-01-07 with VALUE = 0 (holiday; no share traded)
    hist = pd.DataFrame({
        "TRADEDATE": pd.to_datetime(["2022-01-06", "2022-01-07", "2022-01-10", "2022-01-11"]),
        "CLOSE": [3753.29, 3772.04, 3738.68, np.nan],
        "VALUE": [8.3e10, 0.0, 8.5e10, 8.0e10],
    })
    days = moex.calculated_trading_days(hist)
    assert list(days) == [pd.Timestamp("2022-01-06"), pd.Timestamp("2022-01-10")]
    cal = moex.build_calendar(days, "2022-01-06", "2022-01-10")
    assert not cal.loc["2022-01-07", "tradable"]


def test_calendar_keeps_weekend_session_if_index_was_calculated():
    days = pd.DatetimeIndex(["2020-01-10", "2020-01-11", "2020-01-13"])  # 11th is a Saturday
    cal = moex.build_calendar(days, "2020-01-10", "2020-01-13")
    assert cal.loc["2020-01-11", "tradable"]


def test_closure_spells_find_only_the_long_closure():
    cal = moex.build_calendar(_imoex_days_2022(), "2022-02-14", "2022-04-08")
    spells = moex.closure_spells(cal, min_weekdays=10)
    assert len(spells) == 1
    assert spells.iloc[0]["start"] == pd.Timestamp("2022-02-28")
    assert spells.iloc[0]["end"] == pd.Timestamp("2022-03-23")
    assert spells.iloc[0]["weekdays"] == 18


def test_first_trading_day_of_each_month():
    days = pd.DatetimeIndex(["2024-01-09", "2024-01-10", "2024-02-01", "2024-02-02", "2024-03-04"])
    got = moex.first_trading_days_of_month(days)
    assert list(got) == [pd.Timestamp("2024-01-09"), pd.Timestamp("2024-02-01"),
                         pd.Timestamp("2024-03-04")]


# --------------------------------------------------------------------------------------
# Ex-dividend date: T+2 before 31.07.2023, T+1 after
# --------------------------------------------------------------------------------------


def test_settlement_lag_switches_on_31_july_2023():
    assert moex.settlement_lag("2019-06-13") == 2
    assert moex.settlement_lag("2023-07-28") == 2
    assert moex.settlement_lag("2023-07-31") == 2  # record date still needs T+2 settlement
    assert moex.settlement_lag("2023-08-01") == 1
    assert moex.settlement_lag("2024-07-11") == 1


def test_ex_date_t2_is_the_trading_day_before_record_date():
    days = _bdays("2019-06-03", "2019-06-28")
    # Thursday 2019-06-13: last cum-dividend trade date is Tuesday 06-11, ex-date Wednesday 06-12
    assert moex.ex_dividend_date("2019-06-13", days) == pd.Timestamp("2019-06-12")
    # Monday record date: ex-date is the previous *trading* day, i.e. Friday
    assert moex.ex_dividend_date("2019-06-17", days) == pd.Timestamp("2019-06-14")


def test_ex_date_t2_skips_exchange_holidays():
    days = _bdays("2019-06-03", "2019-06-28").drop(pd.Timestamp("2019-06-12"))  # Russia Day
    assert moex.ex_dividend_date("2019-06-13", days) == pd.Timestamp("2019-06-11")


def test_ex_date_t1_is_the_record_date_itself():
    days = _bdays("2024-07-01", "2024-07-31")
    assert moex.ex_dividend_date("2024-07-11", days) == pd.Timestamp("2024-07-11")


def test_ex_date_around_the_regime_switch():
    days = _bdays("2023-07-17", "2023-08-11")
    # record date 31.07 (Mon) is settled under T+2: trades of Fri 28.07 settle on 01.08
    assert moex.ex_dividend_date("2023-07-31", days) == pd.Timestamp("2023-07-28")
    # record date 01.08 (Tue) under T+1: trades of Mon 31.07 settle on 01.08 -> ex-date 01.08
    assert moex.ex_dividend_date("2023-08-01", days) == pd.Timestamp("2023-08-01")


def test_ex_date_for_non_trading_record_date():
    days = _bdays("2024-07-01", "2024-07-31")
    # Saturday 2024-07-13: the register is fixed as of Friday's close; T+1 -> ex-date Friday
    assert moex.ex_dividend_date("2024-07-13", days) == pd.Timestamp("2024-07-12")
    assert moex.ex_dividend_date("2024-07-13", days, weekend_record_date="next") == pd.Timestamp(
        "2024-07-15"
    )


def test_ex_date_outside_calendar_is_nat():
    days = _bdays("2019-06-03", "2019-06-28")
    assert pd.isna(moex.ex_dividend_date("2019-06-03", days))  # no earlier trading day exists
    assert pd.isna(moex.ex_dividend_date("2020-01-01", days))  # beyond the calendar
    assert pd.isna(moex.ex_dividend_date("2019-05-01", days))  # before the calendar


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


def _splits(*rows):
    return pd.DataFrame(rows, columns=["secid", "tradedate", "before", "after"]).assign(
        tradedate=lambda d: pd.to_datetime(d["tradedate"])
    )


def test_split_adjustment_forward_split_gmkn_like():
    idx = _bdays("2024-04-03", "2024-04-11")
    close = pd.Series([15000.0, 15100.0, 14900.0, 150.0, 151.0, 152.0, 153.0], index=idx)
    adj = moex.adjust_for_splits(close, _splits(("GMKN", "2024-04-08", 1, 100)))
    assert adj.loc["2024-04-05"] == pytest.approx(149.0)
    assert adj.loc["2024-04-08"] == pytest.approx(150.0)  # post-split level is left as is
    assert adj.pct_change().abs().max() < 0.02  # no artefact left


def test_split_adjustment_reverse_split_vtbr_like():
    idx = _bdays("2024-07-10", "2024-07-18")
    close = pd.Series([0.0225, 0.0224, 0.0226, 112.0, 113.0, 111.0, 112.5], index=idx)
    adj = moex.adjust_for_splits(close, _splits(("VTBR", "2024-07-15", 5000, 1)))
    assert adj.loc["2024-07-12"] == pytest.approx(0.0226 * 5000)
    assert adj.pct_change().abs().max() < 0.05


def test_two_splits_compound():
    idx = _bdays("2020-01-01", "2020-01-10")
    close = pd.Series([100.0] * 3 + [50.0] * 3 + [10.0] * 2, index=idx)
    sp = _splits(("X", "2020-01-06", 1, 2), ("X", "2020-01-09", 1, 5))
    adj = moex.adjust_for_splits(close, sp)
    assert (adj == 10.0).all()


def test_splits_outside_the_price_window_are_ignored():
    idx = _bdays("2024-01-01", "2024-01-05")
    close = pd.Series([10.0] * 5, index=idx)
    adj = moex.adjust_for_splits(close, _splits(("X", "2026-04-17", 1, 10)))
    assert (adj == 10.0).all()


# --------------------------------------------------------------------------------------
# Total return: dividends reinvested on the ex-date
# --------------------------------------------------------------------------------------


def test_total_return_reinvests_dividend_on_ex_date_t2():
    days = _bdays("2019-06-10", "2019-06-17")  # Mon..Mon
    close = pd.Series([100.0, 100.0, 100.0, 95.0, 96.0, 96.0], index=days)
    # record date Fri 14th -> ex-date Thu 13th (price falls by the dividend of 5)
    div = pd.DataFrame({"secid": ["X"], "record_date": [pd.Timestamp("2019-06-14")],
                        "amount": [5.0], "currency": ["RUB"]})
    div["ex_date"] = [moex.ex_dividend_date(d, days) for d in div["record_date"]]
    assert div["ex_date"].iloc[0] == pd.Timestamp("2019-06-13")
    tri = moex.total_return_index(close, div[["ex_date", "amount"]])
    assert tri.loc["2019-06-12"] == pytest.approx(100.0)
    assert tri.loc["2019-06-13"] == pytest.approx(100.0)  # (95 + 5) / 100: no drop on ex-date
    assert tri.loc["2019-06-14"] == pytest.approx(100.0 * 96.0 / 95.0)
    price_return = close.iloc[-1] / close.iloc[0] - 1
    assert tri.iloc[-1] / tri.iloc[0] - 1 > price_return  # dividend is not lost


def test_total_return_reinvests_dividend_on_ex_date_t1():
    days = _bdays("2024-07-08", "2024-07-15")  # Mon..Mon
    close = pd.Series([200.0, 200.0, 190.0, 191.0, 192.0, 192.0], index=days)
    # record date Wed 07-10: under T+1 the ex-date is the record date itself
    ex = moex.ex_dividend_date("2024-07-10", days)
    assert ex == pd.Timestamp("2024-07-10")
    tri = moex.total_return_index(close, pd.DataFrame({"ex_date": [ex], "amount": [10.0]}))
    assert tri.loc["2024-07-09"] == pytest.approx(100.0)
    assert tri.loc["2024-07-10"] == pytest.approx(100.0 * (190.0 + 10.0) / 200.0)  # no drop
    assert tri.loc["2024-07-11"] == pytest.approx(100.0 * 191.0 / 190.0)


def test_total_return_without_dividends_equals_price_index():
    days = _bdays("2020-01-01", "2020-01-10")
    close = pd.Series(np.linspace(10, 12, len(days)), index=days)
    tri = moex.total_return_index(close, None)
    assert (tri / tri.iloc[0]).to_numpy() == pytest.approx((close / close.iloc[0]).to_numpy())


def test_dividend_on_a_day_without_trades_is_credited_on_next_trade():
    days = _bdays("2020-01-06", "2020-01-10")
    close = pd.Series([100.0, 100.0, np.nan, 90.0, 91.0], index=days)
    div = pd.DataFrame({"ex_date": [pd.Timestamp("2020-01-08")], "amount": [10.0]})
    tri = moex.total_return_index(close, div)
    assert np.isnan(tri.loc["2020-01-08"])  # no trade, no quote
    assert tri.loc["2020-01-09"] == pytest.approx(100.0)  # (90 + 10) / 100 over the gap


def test_total_return_after_split_keeps_dividend_in_share_basis():
    days = _bdays("2024-04-03", "2024-04-12")
    close = pd.Series([15000.0, 15000.0, 15000.0, 150.0, 150.0, 150.0, 150.0, 150.0], index=days)
    sp = _splits(("GMKN", "2024-04-08", 1, 100))
    # dividend of 1000 RUB per old share with ex-date before the split = 10 RUB per new share
    div = pd.DataFrame({"ex_date": [pd.Timestamp("2024-04-04")], "amount": [1000.0]})
    tri = moex.total_return_index(close, div, splits=sp)
    assert tri.loc["2024-04-04"] == pytest.approx(100.0 * (150.0 + 10.0) / 150.0)
    assert tri.loc["2024-04-08"] == pytest.approx(tri.loc["2024-04-04"])  # split adds no return


def test_index_dividend_flow_is_the_gap_between_total_and_price_return():
    days = _bdays("2024-07-08", "2024-07-12")
    pr = pd.Series([100.0, 101.0, 99.0, 99.0, 100.0], index=days)
    # total return = price return + a 1.5 % dividend credited on the third day
    tr = pr.copy()
    growth = (pr.pct_change().fillna(0.0) + pd.Series([0, 0, 0.015, 0, 0], index=days)) + 1.0
    tr = 100.0 * growth.cumprod()
    flow = moex.index_dividend_flow(pr, tr)
    assert np.isnan(flow.iloc[0])
    assert flow.iloc[2] == pytest.approx(0.015) and flow.drop(flow.index[2]).abs().max() < 1e-12


def test_index_dividend_flow_is_nan_without_a_previous_total_return_level():
    days = _bdays("2022-03-22", "2022-03-25")
    pr = pd.Series([1.0, 1.0, 1.0, 1.0], index=days)
    tr = pd.Series([1.0, 1.0, np.nan, 1.0], index=days)  # no MCFTR on the reopening day
    flow = moex.index_dividend_flow(pr, tr)
    assert flow.isna().tolist() == [True, False, True, True]


def test_assign_ex_dates_and_currency_filter():
    days = _bdays("2024-06-03", "2024-07-31")
    div = pd.DataFrame({
        "secid": ["SBER", "SBER", "POLY", "GAZP"],
        "record_date": pd.to_datetime(["2024-07-11", "2024-07-18", "2024-06-20", "2030-01-01"]),
        "amount": [33.3, 1.0, 0.5, 5.0],
        "currency": ["RUB", "RUB", "USD", "RUB"],
    })
    out, report = moex.prepare_dividends(div, days, tickers={"SBER", "POLY", "GAZP"})
    assert set(out["secid"]) == {"SBER"}
    assert out.loc[out["record_date"] == "2024-07-11", "ex_date"].iloc[0] == pd.Timestamp(
        "2024-07-11"
    )
    assert report["excluded_non_rub"] == 1 and report["excluded_outside_calendar"] == 1
    assert report["n_input"] == 4 and report["n_applied"] == 2


# --------------------------------------------------------------------------------------
# Jumps vs ISS corporate-action records
# --------------------------------------------------------------------------------------


def test_detect_jumps_bridges_gaps_and_reports_them():
    idx = _bdays("2015-01-05", "2015-01-12")
    close = pd.DataFrame({"IRAO": [0.0071, np.nan, np.nan, np.nan, 0.7641, 0.775],
                          "SBER": [60.0, 61.0, 62.0, 61.0, 60.0, 59.0]}, index=idx)
    jumps = moex.detect_jumps(close, threshold=0.6)
    assert list(jumps["ticker"]) == ["IRAO"]
    row = jumps.iloc[0]
    assert row["date"] == pd.Timestamp("2015-01-09") and row["prev_date"] == pd.Timestamp("2015-01-05")
    assert row["gap_days"] == 3 and row["r"] == pytest.approx(0.7641 / 0.0071 - 1)


def test_reconcile_jumps_with_iss_splits():
    idx = _bdays("2024-04-03", "2024-04-11")
    close = pd.DataFrame({"GMKN": [15000, 15100, 14900, 150, 151, 152, 153.0],
                          "AFKS": [10, 10, 10, 10, 21, 21, 21.0]}, index=idx)
    jumps = moex.detect_jumps(close, threshold=0.6)
    out = moex.reconcile_jumps(jumps, _splits(("GMKN", "2024-04-08", 1, 100)))
    by = out.set_index("ticker")
    assert by.loc["GMKN", "status"] == "iss_split"
    assert by.loc["GMKN", "market_return_net_of_split"] == pytest.approx(150 / 14900 * 100 - 1)
    assert by.loc["AFKS", "status"] == "unexplained"


def _member(rows):
    return pd.DataFrame(rows, columns=["tradedate", "issue_size_total", "weight", "ff_factor"]).assign(
        tradedate=lambda d: pd.to_datetime(d["tradedate"]))


def test_share_count_change_irao_like():
    member = _member([
        ("2014-12-17", 10_440_000_000_000, 0.21, 0.18),
        ("2015-01-19", 10_440_000_000_000, 0.18, 0.18),
        ("2015-01-20", 104_400_000_000, 0.20, 0.18),
    ])
    ev = moex.share_count_change(member, pd.Timestamp("2014-12-18"), pd.Timestamp("2015-01-20"))
    assert ev["status"] == "ok" and ev["issue_ratio"] == pytest.approx(0.01)
    assert (ev["date_before"], ev["date_after"]) == ("2014-12-17", "2015-01-20")
    assert moex.as_fraction(ev["issue_size_before"], ev["issue_size_after"]) == (100, 1)


def test_share_count_change_without_index_data():
    empty = _member([])
    assert moex.share_count_change(empty, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))[
        "status"] == "no_index_data"
    one_sided = _member([("2020-01-02", 10, 1.0, 1.0)])
    got = moex.share_count_change(one_sided, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))
    assert got["status"] == "no_index_data_on_one_side"


def test_as_fraction_follows_the_register_notation():
    assert moex.as_fraction(1, 100) == (1, 100)  # GMKN: 1 share -> 100
    assert moex.as_fraction(5000, 1) == (5000, 1)  # VTBR: 5000 shares -> 1
    assert moex.as_fraction(10_440_000_000_000, 104_400_000_000) == (100, 1)  # IRAO
    assert moex.as_fraction(2_000_000, 20_000_000) == (1, 10)
    with pytest.raises(ValueError, match="not a simple split"):
        moex.as_fraction(1_000_000, 1_034_700)  # an issue of new shares, not a split


def test_classify_jump_rules():
    ok = lambda ratio: {"status": "ok", "issue_ratio": ratio}  # noqa: E731
    assert moex.classify_jump(True, {"status": "no_index_data"}, 0.01) == "iss_split"
    # IRAO: shares x0.01, price x107.6 -> net market move +7.6 %
    assert moex.classify_jump(False, ok(0.01), 107.6) == "share_count_change"
    # AFKS: share count unchanged, price x2.06 -> a real move
    assert moex.classify_jump(False, ok(1.0), 2.06) == "market_move"
    # an SPO of +10 % cannot explain a doubling
    assert moex.classify_jump(False, ok(1.1), 2.0) == "market_move_with_share_change"
    assert moex.classify_jump(False, {"status": "no_index_data"}, 2.0) == "unexplained_no_index_data"


# --------------------------------------------------------------------------------------
# Board history -> wide panels, per-ticker mask
# --------------------------------------------------------------------------------------


def test_history_panels_drop_closed_days_and_dedup_boards():
    cal_days = pd.DatetimeIndex(["2024-01-09", "2024-01-10", "2024-01-11"])
    h = pd.DataFrame({
        "SECID": ["A", "A", "A", "A", "B", "B"],
        "BOARDID": ["TQNE", "TQBS", "TQBR", "TQBR", "TQBR", "TQBR"],
        "TRADEDATE": pd.to_datetime(["2024-01-09", "2024-01-09", "2024-01-10", "2024-01-13",
                                     "2024-01-09", "2024-01-11"]),
        "CLOSE": [10.0, 11.0, 12.0, 99.0, 5.0, np.nan],
        "VOLUME": [1, 500, 50, 7, 10, 0],
        "VALUE": [10.0, 5500.0, 600.0, 700.0, 50.0, 0.0],
    })
    panels = moex.history_to_panels(h, cal_days)
    assert panels["close"].index.equals(cal_days)  # 2024-01-13 is not a trading day
    assert panels["close"].loc["2024-01-09", "A"] == 11.0  # the busiest board wins
    assert np.isnan(panels["close"].loc["2024-01-11", "B"])
    assert panels["value"].loc["2024-01-10", "A"] == 600.0
    traded = moex.ticker_traded_mask(panels["close"], panels["volume"])
    assert traded.loc["2024-01-09", "B"] and not traded.loc["2024-01-11", "B"]
    assert not traded.loc["2024-01-11", "A"]  # open market, ticker did not trade


# --------------------------------------------------------------------------------------
# Closure mask and opt-in rebasing in cleaning.py
# --------------------------------------------------------------------------------------


def _panel_with_closure():
    # rows exist for the closure days (all NaN), exactly as in the legacy ISS download
    idx = pd.bdate_range("2022-02-21", "2022-03-30")
    p = pd.DataFrame({"SBER": 100.0, "GAZP": 200.0}, index=idx)
    closed = pd.bdate_range("2022-02-28", "2022-03-23")
    p.loc[closed] = np.nan
    return p, pd.Series(~idx.isin(closed), index=idx)


def test_legacy_ffill_fabricates_trading_days_through_the_closure():
    p, _ = _panel_with_closure()
    filled = p.ffill(limit=5)
    assert filled.loc["2022-02-28":"2022-03-04"].notna().all().all()  # the R6.4 artefact


def test_mask_closed_days_removes_closure_rows_before_ffill():
    p, tradable = _panel_with_closure()
    masked = mask_closed_days(p, tradable)
    assert not masked.index.isin(pd.bdate_range("2022-02-28", "2022-03-23")).any()
    cleaned, _ = clean_prices(masked)
    assert not cleaned.index.isin(pd.bdate_range("2022-02-28", "2022-03-23")).any()
    assert cleaned.index.min() == pd.Timestamp("2022-02-21")
    assert cleaned.loc["2022-03-24"].notna().all()


def test_mask_closed_days_does_not_ffill_across_closure():
    idx = pd.bdate_range("2022-02-24", "2022-04-12")
    p = pd.DataFrame({"X": np.nan}, index=idx)
    p.loc["2022-02-24", "X"] = 100.0
    tradable = pd.Series(True, index=idx)
    tradable.loc["2022-02-28":"2022-03-23"] = False
    legacy = p.ffill(limit=5)
    assert legacy["X"].dropna().index[-1] == pd.Timestamp("2022-03-03")  # fill ends in the closure
    masked = mask_closed_days(p, tradable)
    assert list(masked.index[:3]) == [pd.Timestamp("2022-02-24"), pd.Timestamp("2022-02-25"),
                                      pd.Timestamp("2022-03-24")]
    filled = masked.ffill(limit=5)
    # the limit is spent on open days only: 02-25, 03-24, 03-25, 03-28, 03-29
    assert filled["X"].dropna().index[-1] == pd.Timestamp("2022-03-29")
    assert not filled.index.isin(pd.bdate_range("2022-02-28", "2022-03-23")).any()


def test_mask_closed_days_keeps_open_days_missing_from_the_panel():
    idx = pd.DatetimeIndex(["2024-01-09", "2024-01-11"])
    p = pd.DataFrame({"X": [1.0, 2.0]}, index=idx)
    tradable = pd.Series([True, True, True], index=pd.DatetimeIndex(["2024-01-09", "2024-01-10",
                                                                      "2024-01-11"]))
    masked = mask_closed_days(p, tradable)
    assert list(masked.index) == list(tradable.index)
    assert np.isnan(masked.loc["2024-01-10", "X"])


def _split_artefact_panel():
    idx = pd.bdate_range("2024-07-08", "2024-07-26")
    px = np.full(len(idx), 0.02)
    px[idx >= pd.Timestamp("2024-07-15")] = 100.0
    other = np.linspace(10, 11, len(idx))
    return pd.DataFrame({"VTBR": px, "SBER": other}, index=idx)


def test_clean_prices_default_does_not_rebase():
    prices = _split_artefact_panel()
    cleaned, report = clean_prices(prices)
    jump = cleaned["VTBR"].pct_change().max()
    assert jump > 1000  # the artefact is untouched: adjustment is the caller's job (ISS splits)
    assert report.adjustments == []
    assert "_split_adjustments" not in report.dropped_tickers


def test_clean_prices_explicit_threshold_rebases_as_before():
    prices = _split_artefact_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned, report = clean_prices(prices, max_abs_daily_return=0.60)
    assert cleaned["VTBR"].pct_change().abs().max() < 1e-9
    assert report.dropped_tickers["_split_adjustments"] == "1 price-level rebases"
    assert len(report.adjustments) == 1
    adj = report.adjustments[0]
    assert (adj.ticker, adj.date) == ("VTBR", "2024-07-15")
    assert adj.r == pytest.approx(100.0 / 0.02 - 1)
    assert report.to_dict()["adjustments"] == [
        {"ticker": "VTBR", "date": "2024-07-15", "r": pytest.approx(100.0 / 0.02 - 1)}
    ]


def test_clean_prices_explicit_threshold_warns_about_heuristic():
    with pytest.warns(UserWarning, match="heuristic"):
        clean_prices(_split_artefact_panel(), max_abs_daily_return=0.60)


def test_cleaning_report_to_dict_keeps_legacy_keys():
    d = CleaningReport(n_input_tickers=2, n_output_tickers=2).to_dict()
    for key in ("n_input_tickers", "n_output_tickers", "n_dropped", "dropped_tickers",
                "start_date", "end_date", "n_rows"):
        assert key in d
    assert d["adjustments"] == []


def test_file_manifest_hashes_every_file_except_logs_and_provenance(tmp_path):
    import hashlib

    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_bytes(b"hello")
    (tmp_path / "_logs").mkdir()
    (tmp_path / "_logs" / "run.log").write_text("noise")
    (tmp_path / "provenance.json").write_text("{}")
    manifest = moex.file_manifest(tmp_path)
    assert list(manifest) == ["a/x.json"]
    assert manifest["a/x.json"] == {"sha256": hashlib.sha256(b"hello").hexdigest(), "bytes": 5}


# --------------------------------------------------------------------------------------
# Network smoke tests
# --------------------------------------------------------------------------------------


@needs_network
def test_network_splits_contain_the_known_events(tmp_path):
    client = moex.IssClient(tmp_path)
    sp = moex.fetch_splits(client)
    got = {(r.secid, r.tradedate.date().isoformat()): (r.before, r.after) for r in sp.itertuples()}
    assert got[("GMKN", "2024-04-08")] == (1, 100)
    assert got[("VTBR", "2024-07-15")] == (5000, 1)
    assert got[("PLZL", "2025-03-27")] == (1, 10)
    assert got[("TRNFP", "2024-02-21")] == (1, 100)


@needs_network
def test_network_composition_on_a_date(tmp_path):
    client = moex.IssClient(tmp_path)
    comp = moex.fetch_index_composition(client, "IMOEX", "2024-01-09")
    assert 30 <= len(comp) <= 60
    assert comp["weight"].sum() == pytest.approx(100.0, abs=0.5)
    assert {"SBER", "GAZP", "LKOH"} <= set(comp["ticker"])


# --------------------------------------------------------------------------------------
# Invariants of the built dataset (skipped when data/processed/moex_v2 is absent)
# --------------------------------------------------------------------------------------


@needs_v2
def test_v2_calendar_marks_the_2022_closure_and_the_zero_turnover_holiday():
    cal = pd.read_parquet(V2 / "calendar.parquet")
    assert not cal.loc["2022-02-28":"2022-03-23", "tradable"].any()
    assert cal.loc["2022-02-28":"2022-03-23", "in_closure"].all()
    assert cal.loc["2022-02-25", "tradable"] and cal.loc["2022-03-24", "tradable"]
    assert not cal.loc["2022-01-07", "tradable"]  # IMOEX has a level, no share traded


@needs_v2
def test_v2_panels_live_on_tradable_days_only_and_split_dates_are_clean():
    cal = pd.read_parquet(V2 / "calendar.parquet")
    close = pd.read_parquet(V2 / "prices_close.parquet")
    adj = pd.read_parquet(V2 / "prices_adj.parquet")
    assert cal.loc[close.index, "tradable"].all()
    assert not close.index.isin(pd.bdate_range("2022-02-28", "2022-03-23")).any()
    assert close.index.equals(adj.index) and list(close.columns) == list(adj.columns)
    splits = pd.read_parquet(V2 / "splits.parquet")
    got = {r.secid: (r.before, r.after) for r in splits[splits["applied"]].itertuples()}
    assert got["GMKN"] == (1, 100) and got["VTBR"] == (5000, 1)
    assert got["PLZL"] == (1, 10) and got["TRNFP"] == (1, 100)
    ret = adj.pct_change(fill_method=None)
    for r in splits[splits["applied"]].itertuples():  # no artefact left on the split dates
        assert abs(ret.loc[r.tradedate:, r.secid].dropna().iloc[0]) < 0.25


@needs_v2
def test_v2_membership_weights_sum_to_one_per_snapshot():
    m = pd.read_parquet(V2 / "imoex_membership.parquet")
    sums = m.groupby("date")["weight"].sum()
    assert sums.between(0.998, 1.002).all()
    assert m["date"].nunique() == 156  # 2013-01 .. 2025-12


@needs_v2
def test_v2_provenance_hashes_match_the_files():
    prov = json.loads((V2 / "provenance.json").read_text())
    for rel, meta in prov["files"]["processed"].items():
        assert moex.sha256_file(V2 / rel) == meta["sha256"], rel
    raw = REPO_ROOT / prov["config"]["raw_dir"]
    for rel in list(prov["files"]["raw"])[:: max(1, len(prov["files"]["raw"]) // 25)]:
        assert moex.sha256_file(raw / rel) == prov["files"]["raw"][rel]["sha256"], rel
    assert prov["corporate_actions"] and prov["investigations"]["AFKS_2014-12-18"]["verdict"]
