"""Tests for ``cif.data.cbonds``: the adapter for the Cbonds dividend-calendar export.

Synthetic workbooks pin the schema checks and the type coercion; the block at the end reads the
real exports when they have been imported (``scripts/data/import_cbonds_dividends.py``).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from cif.data import cbonds
from tests.xlsx_builder import write_table

REPO_ROOT = Path(__file__).resolve().parents[1]
CBONDS_DIR = REPO_ROOT / "data" / "raw" / "cbonds_dividends"
CBONDS_RU = CBONDS_DIR / "cbonds_dividend_calendar_moex_ru_20261008.xlsx"
CBONDS_ALL = CBONDS_DIR / "cbonds_dividend_calendar_moex_all_20261008.xlsx"
needs_cbonds = pytest.mark.skipif(
    not (CBONDS_RU.exists() and CBONDS_ALL.exists()),
    reason="Cbonds exports are not imported (scripts/data/import_cbonds_dividends.py)",
)

D = dt.datetime


LAST_BUY, EX, RECORD = D(2019, 6, 10), D(2019, 6, 11), D(2019, 6, 13)


def _record(name="Сбербанк России, акция обыкновенная", ticker="SBER", amount=16.0, ccy="RUB",
            yield_cell="", last_buy=LAST_BUY, ex=EX, record=RECORD,
            period="2018", ptype="год", exchange="Московская Биржа, СПБ Биржа", sector="Банки"):
    return [name, exchange, ticker, sector, amount, ccy, yield_cell, last_buy, ex, record, period, ptype]


def _write(path, records, header=cbonds.CBONDS_HEADER):
    return write_table(path, header, records)


# --------------------------------------------------------------------------------------------
# Ticker cells
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cell, expected",
    [
        ("SBER", ["SBER"]),
        ("T, TCSG", ["T", "TCSG"]),
        ("QIWI, NNDQ", ["QIWI", "NNDQ"]),
        ("AFLT, RU_AFLT", ["AFLT", "RU_AFLT"]),
        ("RUAL, 0486", ["RUAL", "0486"]),
        ("  SVETP,SVET-EP ", ["SVETP", "SVET-EP"]),
        ("A, , B", ["A", "B"]),
        ("", []),
        (None, []),
    ],
)
def test_split_tickers(cell, expected):
    assert cbonds.split_tickers(cell) == expected


def test_explode_tickers_keeps_one_row_per_alias_and_the_sheet_row(tmp_path):
    path = _write(tmp_path / "c.xlsx", [
        _record(ticker="SBER"),
        _record(ticker="T, TCSG", name="Т-Технологии", amount=9.25, ex=D(2024, 11, 25), record=D(2024, 11, 25)),
    ])
    rows = cbonds.read_calendar(path)
    long = cbonds.explode_tickers(rows)
    assert long["alias"].tolist() == ["SBER", "T", "TCSG"]
    assert long["alias_rank"].tolist() == [0, 0, 1]
    assert long["n_aliases"].tolist() == [1, 2, 2]
    assert long["sheet_row"].tolist() == [2, 3, 3]


def test_rows_without_a_ticker_are_kept_with_an_empty_alias(tmp_path):
    # three rows of the export of 2026-10-08 (Nanosoft) have an empty ticker cell
    path = _write(tmp_path / "c.xlsx", [_record(ticker="SBER"), _record(ticker="", name="Нанософт")])
    t = cbonds.read_calendar(path)
    assert t["tickers"].tolist() == ["SBER", ""]
    long = cbonds.explode_tickers(t)
    assert long["alias"].tolist() == ["SBER", ""]
    assert long["n_aliases"].tolist() == [1, 0]
    assert long["sheet_row"].tolist() == [2, 3]


# --------------------------------------------------------------------------------------------
# Reading and schema
# --------------------------------------------------------------------------------------------


def test_read_calendar_renames_columns_and_coerces_types(tmp_path):
    path = _write(tmp_path / "c.xlsx", [
        _record(yield_cell=""),
        _record(ticker="GAZP", amount=7.2, yield_cell=13.3, ex=D(2014, 7, 16), record=D(2014, 7, 17),
                last_buy=D(2014, 7, 15)),
    ])
    t = cbonds.read_calendar(path)
    assert list(t.columns) == [
        "name", "exchange", "tickers", "sector", "amount", "currency", "yield_pct_current",
        "last_buy_date", "ex_date_cbonds", "record_date", "period", "period_type", "sheet_row",
    ]
    assert t["amount"].dtype == float
    assert t["amount"].tolist() == [16.0, 7.2]
    for col in ("last_buy_date", "ex_date_cbonds", "record_date"):
        assert str(t[col].dtype).startswith("datetime64"), col
    assert t["record_date"].tolist() == [pd.Timestamp("2019-06-13"), pd.Timestamp("2014-07-17")]
    assert t["yield_pct_current"].isna().tolist() == [True, False]
    assert t["period"].tolist() == ["2018", "2018"]
    assert t["sheet_row"].tolist() == [2, 3]


def test_read_calendar_normalises_currency_and_strips_text(tmp_path):
    path = _write(tmp_path / "c.xlsx", [_record(ccy=" usd ", ticker=" SBER ", ptype=" год ")])
    t = cbonds.read_calendar(path)
    assert t.loc[0, "currency"] == "USD"
    assert t.loc[0, "tickers"] == "SBER"
    assert t.loc[0, "period_type"] == "год"


def test_a_changed_header_is_refused_with_the_difference(tmp_path):
    header = list(cbonds.CBONDS_HEADER)
    header[4] = "Дивиденд на акцию"
    path = _write(tmp_path / "c.xlsx", [_record()], header=header)
    with pytest.raises(cbonds.CbondsFormatError, match="Дивиденд на акцию"):
        cbonds.read_calendar(path)


def test_a_missing_key_field_is_refused_not_dropped(tmp_path):
    path = _write(tmp_path / "c.xlsx", [_record(), _record(amount=None)])
    with pytest.raises(cbonds.CbondsFormatError, match="amount"):
        cbonds.read_calendar(path)
    path = _write(tmp_path / "d.xlsx", [_record(record=None)])
    with pytest.raises(cbonds.CbondsFormatError, match="record_date"):
        cbonds.read_calendar(path)


def test_a_non_numeric_amount_is_refused(tmp_path):
    path = _write(tmp_path / "c.xlsx", [_record(amount="n/a")])
    with pytest.raises(cbonds.CbondsFormatError, match="amount"):
        cbonds.read_calendar(path)


def test_summary_counts_rows_and_the_ex_date_range(tmp_path):
    path = _write(tmp_path / "c.xlsx", [
        _record(ex=D(2013, 2, 7), record=D(2013, 2, 7)),
        _record(ex=D(2026, 10, 5), record=D(2026, 10, 5)),
    ])
    s = cbonds.summarise(cbonds.read_calendar(path))
    assert s == {"rows": 2, "ex_date_min": "2013-02-07", "ex_date_max": "2026-10-05",
                 "currencies": {"RUB": 2}}


# --------------------------------------------------------------------------------------------
# The real exports
# --------------------------------------------------------------------------------------------


@needs_cbonds
def test_real_export_has_the_documented_shape():
    ru = cbonds.read_calendar(CBONDS_RU)
    allc = cbonds.read_calendar(CBONDS_ALL)
    assert len(ru) == 2149 and len(allc) == 2174
    s = cbonds.summarise(ru)
    assert s["ex_date_min"] == "2013-02-07" and s["ex_date_max"] == "2026-10-05"
    assert s["currencies"] == {"RUB": 2098, "USD": 51}
    assert cbonds.summarise(allc)["currencies"] == {"RUB": 2098, "USD": 51, "JPY": 25}
    assert ru["amount"].gt(0).all()
    assert (ru["record_date"].notna() & ru["ex_date_cbonds"].notna()).all()


@needs_cbonds
def test_real_export_multi_ticker_cells_and_the_qiwi_pairs():
    ru = cbonds.read_calendar(CBONDS_RU)
    long = cbonds.explode_tickers(ru)
    assert len(long) == 2367
    multi = ru[ru["tickers"].str.contains(",")]["tickers"].value_counts().to_dict()
    assert multi["QIWI, NNDQ"] == 40 and multi["T, TCSG"] == 7 and multi["RUAL, 0486"] == 4
    # QIWI: every payment appears twice (type "регулярные" plus a quarter label), same amount
    qiwi = ru[ru["tickers"] == "QIWI, NNDQ"]
    paired = qiwi.groupby(["record_date", "amount"]).size()
    assert (paired[paired > 1] == 2).all()
    assert (paired == 2).sum() == 14 and (paired == 1).sum() == 12  # 28 + 12 = 40 rows


@needs_cbonds
def test_real_export_sber_rows_match_the_known_payments():
    ru = cbonds.read_calendar(CBONDS_RU)
    sber = ru[ru["tickers"] == "SBER"].set_index("record_date")["amount"]
    assert sber[pd.Timestamp("2014-06-17")] == pytest.approx(3.2)
    assert sber[pd.Timestamp("2019-06-13")] == pytest.approx(16.0)
    assert sber[pd.Timestamp("2024-07-11")] == pytest.approx(33.3)
