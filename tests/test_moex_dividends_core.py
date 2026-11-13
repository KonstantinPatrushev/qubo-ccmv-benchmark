"""Core rules of ``cif.data.moex_dividends``: ex-dates, split restatement, mapping to the panel.

Synthetic calendars and panels only; the real data are checked in ``test_moex_dividends_real.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.data import moex
from cif.data import moex_dividends as md

TS = pd.Timestamp


def _calendar(start="2021-12-01", end="2023-09-30", closed=()):
    """Weekdays as trading days, minus the given (start, end) closed spans; returns (days, spells)."""
    days = pd.bdate_range(start, end)
    for a, b in closed:
        days = days[(days < TS(a)) | (days > TS(b))]
    cal = moex.build_calendar(days, start, end)
    return pd.DatetimeIndex(days), moex.closure_spells(cal)


# --------------------------------------------------------------------------------------------
# Ex-date
# --------------------------------------------------------------------------------------------


def test_without_closures_the_rule_equals_the_repository_rule():
    days, spells = _calendar()
    assert spells.empty
    for record in pd.date_range("2021-12-15", "2023-09-15"):
        ours = md.ex_date_from_record(record, days, spells)
        theirs = moex.ex_dividend_date(record, days)
        assert (ours == theirs) or (pd.isna(ours) and pd.isna(theirs)), record


CLOSURE = (("2022-02-28", "2022-03-23"),)


@pytest.mark.parametrize(
    "record, expected",
    [
        ("2022-02-24", "2022-02-23"),  # T+2 before the closure: the trading day before R
        ("2022-02-25", "2022-02-24"),
        ("2022-02-28", "2022-02-25"),  # R on the first closed day: trades of 25.02 settle on 01.03, after R
        ("2022-03-01", "2022-03-24"),  # trades of 25.02 settle on 01.03: still cum, next trading day is 24.03
        ("2022-03-09", "2022-03-24"),  # R inside the closure: ex is the day the market reopens
        ("2022-03-23", "2022-03-24"),
        ("2022-03-24", "2022-03-24"),  # the reopening day itself: nothing traded cum-dividend after 25.02
        ("2022-03-25", "2022-03-24"),
        ("2022-03-28", "2022-03-25"),  # normal T+2 again
    ],
)
def test_record_dates_in_and_around_the_2022_closure(record, expected):
    days, spells = _calendar(closed=CLOSURE)
    assert len(spells) == 1
    assert md.ex_date_from_record(record, days, spells) == TS(expected)


def test_record_date_on_a_weekend_uses_the_previous_session():
    days, spells = _calendar()
    # Sunday 26.12.2021: last session Friday 24.12; T+2 -> last cum trade Wednesday 22.12, ex Thursday 23.12
    assert md.ex_date_from_record("2021-12-26", days, spells) == TS("2021-12-23")


def test_t_plus_one_from_august_2023():
    days, spells = _calendar()
    assert md.ex_date_from_record("2023-08-02", days, spells) == TS("2023-08-02")
    assert md.ex_date_from_record("2023-07-31", days, spells) == TS("2023-07-28")  # still T+2


def test_record_date_outside_the_calendar_is_nat():
    days, spells = _calendar()
    assert pd.isna(md.ex_date_from_record("2021-11-01", days, spells))
    assert pd.isna(md.ex_date_from_record("2024-01-01", days, spells))


def test_align_to_trading_day_next_and_previous():
    days, _ = _calendar()
    sat = TS("2022-01-08")
    assert md.align_to_trading_day(sat, days) == TS("2022-01-10")
    assert md.align_to_trading_day(sat, days, side="previous") == TS("2022-01-07")
    assert md.align_to_trading_day(TS("2022-01-10"), days) == TS("2022-01-10")
    assert pd.isna(md.align_to_trading_day(TS("2024-01-01"), days))
    assert pd.isna(md.align_to_trading_day(TS("2020-01-01"), days, side="previous"))


def test_trading_day_distance_counts_sessions():
    days, _ = _calendar()
    assert md.trading_day_distance(TS("2022-01-12"), TS("2022-01-10"), days) == 2
    assert md.trading_day_distance(TS("2022-01-10"), TS("2022-01-12"), days) == -2
    assert md.trading_day_distance(TS("2022-01-10"), TS("2022-01-10"), days) == 0


# --------------------------------------------------------------------------------------------
# Split restatement
# --------------------------------------------------------------------------------------------

SPLITS = pd.DataFrame({
    "secid": ["GMKN", "VTBR", "T", "X", "X"],
    "tradedate": pd.to_datetime(["2024-04-08", "2024-07-15", "2026-04-17", "2020-01-10", "2022-01-10"]),
    "before": [1, 5000, 1, 1, 1],
    "after": [100, 1, 10, 2, 5],
})


def test_restatement_factor_returns_a_cbonds_amount_to_the_basis_of_the_ex_date():
    # Cbonds shows today's share count: GMKN is 100:1 since 2024, so older amounts are 100 times too small
    assert md.restatement_factor("GMKN", TS("2023-12-26"), SPLITS) == pytest.approx(100.0)
    assert md.restatement_factor("GMKN", TS("2024-04-08"), SPLITS) == 1.0  # on the split day: new basis
    assert md.restatement_factor("GMKN", TS("2024-04-05"), SPLITS) == pytest.approx(100.0)
    # VTBR consolidated 5000:1: older amounts are 5000 times too large
    assert md.restatement_factor("VTBR", TS("2021-07-14"), SPLITS) == pytest.approx(1 / 5000)
    assert md.restatement_factor("VTBR", TS("2025-07-11"), SPLITS) == 1.0


def test_restatement_counts_splits_after_the_end_of_the_panel():
    assert md.restatement_factor("T", TS("2025-05-16"), SPLITS) == pytest.approx(10.0)


def test_restatement_compounds_several_splits():
    assert md.restatement_factor("X", TS("2019-01-01"), SPLITS) == pytest.approx(10.0)
    assert md.restatement_factor("X", TS("2021-01-01"), SPLITS) == pytest.approx(5.0)
    assert md.restatement_factor("X", TS("2023-01-01"), SPLITS) == 1.0


def test_restatement_follows_the_successor_chain():
    # the GDR TCSG became the share T one for one; the 2026 split of T restates the GDR-era amounts too
    assert md.restatement_factor("TCSG", TS("2024-11-25"), SPLITS) == pytest.approx(10.0)


def test_restatement_of_an_unrelated_ticker_is_one():
    assert md.restatement_factor("SBER", TS("2015-01-01"), SPLITS) == 1.0
    assert md.restatement_factor("SBER", TS("2015-01-01"), SPLITS.iloc[0:0]) == 1.0


def test_successor_chain():
    assert md.successor_chain("TCSG") == ["TCSG", "T"]
    assert md.successor_chain("T") == ["T"]
    assert md.successor_chain("EONR") == ["EONR", "UPRO"]
    chain = (md.Succession("A", "B"), md.Succession("B", "C"))
    assert md.successor_chain("A", chain) == ["A", "B", "C"]


# --------------------------------------------------------------------------------------------
# Mapping the vendor tickers to the securities of the panel
# --------------------------------------------------------------------------------------------


def _close():
    idx = pd.bdate_range("2015-01-05", "2017-12-29")
    close = pd.DataFrame(index=idx, columns=["EONR", "UPRO", "SBER", "T", "TCSG"], dtype=float)
    close.loc[idx <= "2016-06-30", "EONR"] = 10.0
    close.loc[idx >= "2016-07-01", "UPRO"] = 20.0
    close["SBER"] = 100.0
    close.loc[idx <= "2016-12-31", "TCSG"] = 5.0
    close.loc[idx >= "2017-01-02", "T"] = 6.0
    return close


@pytest.mark.parametrize(
    "aliases, expected",
    [
        (["SBER"], [("SBER", "alias")]),
        (["UPRO"], [("UPRO", "alias"), ("EONR", "predecessor")]),
        (["T", "TCSG"], [("T", "alias"), ("TCSG", "alias")]),
        (["QIWI", "NNDQ"], []),
        ([""], []),
        ([], []),
    ],
)
def test_candidate_secids(aliases, expected):
    assert md.candidate_secids(aliases, set(_close().columns)) == expected


def test_a_renamed_security_is_assigned_by_the_quote_on_the_ex_date():
    close = _close()
    assert md.assign_secid(["UPRO"], TS("2015-06-10"), close) == ("EONR", "ok")
    assert md.assign_secid(["UPRO"], TS("2017-06-12"), close) == ("UPRO", "ok")


def test_two_aliases_in_the_panel_are_separated_by_which_one_traded():
    close = _close()
    assert md.assign_secid(["T", "TCSG"], TS("2016-12-30"), close) == ("TCSG", "ok")
    assert md.assign_secid(["T", "TCSG"], TS("2017-03-01"), close) == ("T", "ok")


def test_both_aliases_quoted_is_ambiguous_and_takes_the_first():
    close = _close()
    close.loc["2016-12-01":"2016-12-30", "T"] = 6.0
    secid, status = md.assign_secid(["T", "TCSG"], TS("2016-12-15"), close)
    assert (secid, status) == ("T", "ambiguous")


def test_no_quote_on_the_ex_date_keeps_the_first_candidate_and_says_so():
    close = _close()
    assert md.assign_secid(["SBER"], TS("2012-01-10"), close) == ("SBER", "no_quote_on_ex")


def test_unmapped_rows_are_labelled():
    close = _close()
    assert md.assign_secid(["QIWI", "NNDQ"], TS("2016-01-11"), close) == (None, "not_in_universe")
    assert md.assign_secid([""], TS("2016-01-11"), close) == (None, "no_ticker")


# --------------------------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------------------------


def _rows(*rows):
    return pd.DataFrame(rows, columns=["sheet_row", "secid", "record_date", "currency", "amount", "period_type"])


def test_duplicate_payments_collapse_to_the_row_with_the_specific_period_type():
    # both orders occur in the export: label first (rows 10/11), generic type first (rows 1009/1010)
    ev = _rows(
        (10, "QIWI", TS("2016-08-29"), "USD", 0.22, "3 месяца (II кв.)"),
        (11, "QIWI", TS("2016-08-29"), "USD", 0.22, "регулярные"),
        (12, "SBER", TS("2019-06-13"), "RUB", 16.0, "год"),
        (1009, "QIWI", TS("2019-12-02"), "USD", 0.28, "регулярные"),
        (1010, "QIWI", TS("2019-12-02"), "USD", 0.28, "3 месяца (III кв.)"),
    )
    kept, dropped = md.drop_duplicates(ev)
    assert kept["sheet_row"].tolist() == [10, 12, 1010]
    assert kept["n_duplicates"].tolist() == [1, 0, 1]
    assert dropped["sheet_row"].tolist() == [11, 1009]
    assert dropped["duplicate_of"].tolist() == [10, 1010]


def test_the_row_with_the_lower_sheet_number_wins_a_tie():
    ev = _rows(
        (21, "A", TS("2020-01-10"), "RUB", 1.0, "год"),
        (20, "A", TS("2020-01-10"), "RUB", 1.0, "год"),
    )
    kept, dropped = md.drop_duplicates(ev)
    assert kept["sheet_row"].tolist() == [20] and dropped["sheet_row"].tolist() == [21]


def test_payments_with_different_amounts_on_one_record_date_are_both_kept():
    # VTBR, record date 15.07.2021: 0.087 (ordinary) and 6.91 ("Внеплановые") in the restated basis
    ev = _rows(
        (30, "VTBR", TS("2021-07-15"), "RUB", 0.087, "год"),
        (31, "VTBR", TS("2021-07-15"), "RUB", 6.91, "Внеплановые"),
    )
    kept, dropped = md.drop_duplicates(ev)
    assert len(kept) == 2 and dropped.empty


def test_equal_amounts_in_different_currencies_or_dates_are_not_duplicates():
    ev = _rows(
        (40, "A", TS("2020-01-10"), "RUB", 1.0, "год"),
        (41, "A", TS("2020-01-10"), "USD", 1.0, "год"),
        (42, "A", TS("2020-07-10"), "RUB", 1.0, "год"),
    )
    kept, dropped = md.drop_duplicates(ev)
    assert len(kept) == 3 and dropped.empty


def test_amounts_equal_up_to_float_noise_are_duplicates():
    ev = _rows(
        (50, "A", TS("2020-01-10"), "RUB", 0.1 + 0.2, "год"),
        (51, "A", TS("2020-01-10"), "RUB", 0.3, "регулярные"),
    )
    kept, _ = md.drop_duplicates(ev)
    assert len(kept) == 1


# --------------------------------------------------------------------------------------------
# Choosing between the rule and the vendor's ex-date
# --------------------------------------------------------------------------------------------

RULE, VENDOR = TS("2023-05-08"), TS("2023-05-05")


def test_agreeing_dates_need_no_price_evidence():
    assert md.choose_ex_date(RULE, RULE, 0.0, 0.0, 0.05, 0.01) == (RULE, "agree")


def test_without_a_vendor_date_the_rule_stands():
    assert md.choose_ex_date(RULE, pd.NaT, 0.0, 0.0, 0.05, 0.01) == (RULE, "rule_only")


def test_a_small_dividend_cannot_arbitrate_so_the_rule_stands():
    # yield 1.5 sigma: not identifiable
    assert md.choose_ex_date(RULE, VENDOR, 0.0, -0.015, 0.015, 0.01) == (RULE, "rule_uninformative")


def test_a_clear_drop_on_the_vendor_date_overrules_the_rule():
    assert md.choose_ex_date(RULE, VENDOR, 0.002, -0.049, 0.05, 0.01) == (VENDOR, "price_vendor")


def test_a_clear_drop_on_the_rule_date_confirms_it():
    assert md.choose_ex_date(RULE, VENDOR, -0.05, 0.001, 0.05, 0.01) == (RULE, "price_rule")


def test_a_draw_goes_to_the_rule():
    assert md.choose_ex_date(RULE, VENDOR, -0.03, -0.03, 0.05, 0.01) == (RULE, "rule_default")
    # the vendor date must win by a margin of one sigma, not by a hair
    assert md.choose_ex_date(RULE, VENDOR, -0.0300, -0.0305, 0.05, 0.01) == (RULE, "rule_default")


def test_missing_prices_leave_the_rule_in_place():
    assert md.choose_ex_date(RULE, VENDOR, np.nan, -0.05, 0.05, 0.01) == (RULE, "rule_no_price")
    assert md.choose_ex_date(RULE, VENDOR, -0.05, np.nan, 0.05, 0.01) == (RULE, "rule_no_price")
    assert md.choose_ex_date(RULE, VENDOR, -0.05, -0.05, np.nan, 0.01) == (RULE, "rule_no_price")
