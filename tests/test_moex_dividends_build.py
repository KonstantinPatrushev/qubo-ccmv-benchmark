"""End-to-end assembly of the dividend events (``moex_dividends.build_events``) on a synthetic panel.

Every path of the pipeline has a row in the Cbonds-shaped table below: a normal event, a restated
one, an unknown and an empty ticker, a date outside the panel, a duplicated payment, a USD
payment, a renamed security (T/TCSG) and the two outcomes of the choice between the rule's and the
vendor's ex-date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.data import moex_dividends as md
from tests.dividend_fixtures import DAYS, SBER, TS, T, _prev  # noqa: F401
from tests.dividend_fixtures import cbonds_rows as _cbonds
from tests.dividend_fixtures import make_panel as _panel


@pytest.fixture(scope="module")
def built():
    return md.build_events(_cbonds(), _panel())


def _event(built, sheet_row):
    ev = built.events
    out = ev[ev["source_ref"] == f"cbonds:{sheet_row}"]
    assert len(out) == 1, (sheet_row, len(out))
    return out.iloc[0]


def _audit(built, sheet_row):
    return built.audit.set_index("sheet_row").loc[sheet_row]


# --------------------------------------------------------------------------------------------
# Statuses of the vendor rows
# --------------------------------------------------------------------------------------------


def test_every_vendor_row_gets_exactly_one_status(built):
    audit = built.audit
    assert sorted(audit["sheet_row"]) == list(range(2, 22))
    assert audit["sheet_row"].is_unique
    assert audit.set_index("sheet_row")["status"].to_dict() == {
        2: "ok", 3: "ok", 4: "not_in_universe", 5: "no_ticker", 6: "after_panel", 7: "duplicate", 8: "ok",
        9: "ok", 10: "ok", 11: "ok", 12: "ok", 13: "ok", 14: "before_panel", 15: "ok", 16: "ok",
        17: "ok", 18: "ok", 19: "ok", 20: "ok", 21: "ok",
    }
    assert len(built.events) == 15  # rows 2, 3, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21


def test_the_duplicate_row_names_the_row_it_duplicates(built):
    assert _audit(built, 7)["duplicate_of"] == 8
    assert _event(built, 8)["n_duplicates"] == 1
    assert "merged_duplicates" in _event(built, 8)["flags"].split(",")


# --------------------------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------------------------


def test_a_normal_event_keeps_the_vendor_amount_and_passes_the_price_check(built):
    e = _event(built, 2)
    assert e["secid"] == "SBER" and e["currency"] == "RUB" and e["source"] == "cbonds"
    assert e["ex_date"] == TS("2021-03-16") and e["record_date"] == TS("2021-03-17")
    assert e["restatement_factor"] == 1.0 and e["amount"] == pytest.approx(e["cbonds_amount"])
    assert e["ex_basis"] == "agree" and e["drop_status"] == "ok" and bool(e["usable_rub"])
    assert e["dividend_yield"] == pytest.approx(0.05, rel=1e-6)


def test_a_restated_amount_is_returned_to_the_basis_of_the_ex_date(built):
    e = _event(built, 3)
    assert e["secid"] == "GMKN"
    assert e["restatement_factor"] == pytest.approx(100.0)
    assert e["amount"] == pytest.approx(100.0 * e["cbonds_amount"])
    assert e["dividend_yield"] == pytest.approx(0.05, rel=1e-6)
    assert e["drop_status"] == "ok"
    assert "restated" in e["flags"].split(",")


def test_the_renamed_security_is_assigned_by_quote_and_restated_through_the_chain(built):
    old, new = _event(built, 10), _event(built, 11)
    assert (old["secid"], new["secid"]) == ("TCSG", "T")
    assert old["restatement_factor"] == pytest.approx(10.0) and new["restatement_factor"] == pytest.approx(10.0)
    assert old["dividend_yield"] == pytest.approx(0.05, rel=1e-6)
    assert new["dividend_yield"] == pytest.approx(0.05, rel=1e-6)
    assert old["cbonds_tickers"] == "T, TCSG"


def test_a_foreign_currency_event_is_kept_and_marked_unusable_for_the_rub_total(built):
    e = _event(built, 9)
    assert e["currency"] == "USD" and e["drop_status"] == "non_rub"
    assert not bool(e["usable_rub"])
    assert "non_rub" in e["flags"].split(",")


# --------------------------------------------------------------------------------------------
# Choice of the ex-date
# --------------------------------------------------------------------------------------------


def test_the_price_moves_the_ex_date_to_the_vendors_when_it_fell_there(built):
    e = _event(built, 12)
    assert e["ex_rule"] == TS("2022-06-14") and e["ex_vendor"] == TS("2022-06-13")
    assert e["ex_date"] == TS("2022-06-13") and e["ex_basis"] == "price_vendor"
    assert e["drop_status"] == "ok"
    assert "ex_vendor" in e["flags"].split(",")


def test_the_rule_stands_when_the_price_fell_on_its_date(built):
    e = _event(built, 13)
    assert e["ex_rule"] == TS("2022-09-13") and e["ex_vendor"] == TS("2022-09-12")
    assert e["ex_date"] == TS("2022-09-13") and e["ex_basis"] == "price_rule"
    assert "ex_vendor" not in e["flags"].split(",")
    assert "ex_vendor_differs" in e["flags"].split(",")


def test_events_are_sorted_and_ids_are_unique(built):
    ev = built.events
    assert ev["event_id"].is_unique
    assert ev.equals(ev.sort_values(["secid", "ex_date", "record_date", "amount"], kind="stable").reset_index(drop=True))


def test_membership_on_the_ex_date_is_reported(built):
    assert _event(built, 11)["in_index_on_ex"]       # T is in the index from 2023-02-01
    assert _event(built, 10)["in_index_on_ex"]       # TCSG until 2022-12-30
    early = _event(built, 15)
    assert not early["in_index_on_ex"]
    assert early["ex_date"] == TS("2023-01-18")
    flags = early["flags"].split(",")
    assert "not_in_index" in flags and "price_outlier" in flags  # a 5 % dividend with no drop at all
    assert bool(early["usable_rub"])                 # an outlier is flagged, not dropped
    assert built.events["in_index_on_ex"].sum() == len(built.events) - 3  # rows 15, 17 and 20


def test_a_vendor_ex_date_that_is_not_a_trading_day_is_flagged_and_the_rule_stands(built):
    e = _event(built, 16)
    assert e["ex_date"] == TS("2021-11-12") and e["ex_basis"] == "rule_uninformative"
    flags = e["flags"].split(",")
    assert "vendor_ex_not_trading" in flags and "ex_vendor_differs" in flags


def test_an_event_without_a_previous_quote_cannot_be_used_for_the_rub_total(built):
    e = _event(built, 21)  # a new listing: the ex-date is the first day of the series
    assert e["secid"] == "NEWC" and e["ex_date"] == TS("2022-06-01")
    assert e["drop_status"] == "no_quote" and not bool(e["usable_rub"])
    flags = e["flags"].split(",")
    assert "before_first_quote" in flags and "rename_boundary" not in flags


def test_the_first_day_of_a_renamed_series_is_checked_against_the_predecessors_close(built):
    e = _event(built, 17)
    assert e["secid"] == "T" and e["ex_date"] == TS("2023-01-03")
    assert e["prev_secid"] == "TCSG" and e["prev_date"] == TS("2022-12-30")
    assert e["drop_status"] != "no_quote"
    flags = e["flags"].split(",")
    assert "rename_boundary" in flags and "before_first_quote" in flags
    assert not bool(e["usable_rub"])  # a single series cannot credit it: the series starts that day


def test_a_halted_ex_day_is_still_reinvested_on_the_next_quote(built):
    # no quote on the ex-date, quotes before and after: the total return credits the next day
    e = _event(built, 19)
    assert e["ex_date"] == TS("2021-08-12") and e["drop_status"] == "no_quote"
    assert e["secid_status"] == "no_quote_on_ex"
    assert bool(e["usable_rub"])
    flags = e["flags"].split(",")
    assert "before_first_quote" not in flags and "after_last_quote" not in flags


def test_a_payment_after_the_last_quote_of_the_series_is_flagged_and_unusable(built):
    e = _event(built, 20)
    assert e["secid"] == "TCSG" and not bool(e["usable_rub"])
    assert "after_last_quote" in e["flags"].split(",")


def test_a_foreign_currency_event_keeps_the_rule_when_the_vendor_date_differs(built):
    e = _event(built, 18)
    assert e["ex_rule"] == TS("2021-10-12") and e["ex_vendor"] == TS("2021-10-11")
    assert e["ex_date"] == TS("2021-10-12") and e["ex_basis"] == "rule_non_rub"


# --------------------------------------------------------------------------------------------
# Registry events
# --------------------------------------------------------------------------------------------


def _registry(**kw):
    base = {"secid": "SBER", "record_date": TS("2022-12-14"), "ex_date_declared": pd.NaT, "amount": 0.05,
            "currency": "RUB", "period": "2022", "period_type": "год", "source_ref": "registry:SBER-2022"}
    return pd.DataFrame([{**base, **kw}])


def test_registry_events_join_the_table_with_the_same_checks(built):
    out = md.build_events(_cbonds(), _panel(), registry=_registry())
    e = out.events[out.events["source"] == "registry"].iloc[0]
    assert e["secid"] == "SBER" and e["source_ref"] == "registry:SBER-2022"
    assert e["ex_date"] == TS("2022-12-13")  # the rule from the record date
    assert e["restatement_factor"] == 1.0 and np.isnan(e["cbonds_amount"])
    assert len(out.events) == len(built.events) + 1


def test_a_registry_event_that_repeats_a_vendor_payment_is_not_added_twice():
    reg = _registry(record_date=TS("2021-03-17"), amount=0.05 * _prev(SBER, "2021-03-16"))
    out = md.build_events(_cbonds(), _panel(), registry=reg)
    assert (out.events["source"] == "registry").sum() == 0
    assert "registry_overlap" in out.registry_audit["status"].tolist()


def test_a_second_registry_payment_on_a_vendor_record_date_is_added_and_flagged():
    reg = _registry(record_date=TS("2021-03-17"), amount=0.02 * _prev(SBER, "2021-03-16"))
    out = md.build_events(_cbonds(), _panel(), registry=reg)
    e = out.events[out.events["source"] == "registry"]
    assert len(e) == 1
    assert "same_record_date_as_vendor" in e.iloc[0]["flags"].split(",")
    assert out.registry_audit["status"].tolist() == ["ok"]


def test_an_ex_date_the_issuer_states_stands_when_the_price_cannot_decide():
    # the dividend is far too small to show in the price: the issuer's date stands against the rule's
    out = md.build_events(_cbonds(), _panel(), registry=_registry(ex_date_declared=TS("2022-12-12")))
    e = out.events[out.events["source"] == "registry"].iloc[0]
    assert e["ex_date"] == TS("2022-12-12") and e["ex_rule"] == TS("2022-12-13")
    assert e["ex_basis"] == "declared" and "ex_declared" in e["flags"].split(",")


def test_an_ex_date_the_issuer_states_yields_to_a_price_that_clearly_prefers_the_rule():
    # a 5 % dividend of SBER whose rule date carries the planted -5 % drop (2022-06-13), declared a day earlier
    amount = 0.05 * _prev(SBER, "2022-06-13")
    reg = _registry(record_date=TS("2022-06-14"), ex_date_declared=TS("2022-06-10"), amount=amount)
    out = md.build_events(_cbonds(), _panel(), registry=reg)
    e = out.events[out.events["source"] == "registry"].iloc[0]
    assert e["ex_rule"] == TS("2022-06-13") and e["ex_date"] == TS("2022-06-13") and e["ex_basis"] == "price_rule"


def test_an_ex_date_the_issuer_states_stands_for_a_foreign_event_without_a_rate():
    reg = _registry(ex_date_declared=TS("2022-12-12"), currency="USD")
    out = md.build_events(_cbonds(), _panel(), registry=reg)
    e = out.events[out.events["source"] == "registry"].iloc[0]
    assert e["ex_date"] == TS("2022-12-12") and e["ex_basis"] == "declared"


def test_a_vendor_ex_date_never_overrides_the_rule_on_its_own_authority(built):
    # row 16: the vendor's ex-date is a Saturday and the dividend too small to test: the rule stands
    e = _event(built, 16)
    assert e["ex_basis"] == "rule_uninformative" and "ex_declared" not in e["flags"].split(",")


# --------------------------------------------------------------------------------------------
# Exchange rates (an optional table; none ships with the data)
# --------------------------------------------------------------------------------------------


def _fx(end="2021-12-31"):
    idx = pd.bdate_range("2021-01-04", end)
    return pd.DataFrame({"USD": np.linspace(70.0, 80.0, len(idx))}, index=idx)


def test_without_a_rate_table_foreign_events_keep_their_currency_and_no_fx_column_is_filled(built):
    e = _event(built, 9)
    assert e["currency"] == "USD" and e["amount"] == 1.0
    assert np.isnan(e["fx_rate"]) and np.isnan(e["amount_orig"]) and e["currency_orig"] is None
    assert "fx_converted" not in e["flags"] and "fx_missing" not in e["flags"]


def test_a_rate_table_converts_foreign_events_at_the_last_rate_before_the_ex_date():
    fx = _fx()
    out = md.build_events(_cbonds(), _panel(), fx=fx)
    e = out.events[out.events["source_ref"] == "cbonds:9"].iloc[0]  # 1.0 USD, ex-date 2021-09-14
    rate = float(fx.loc[:"2021-09-14", "USD"].iloc[-1])
    assert e["currency"] == "RUB" and e["amount"] == pytest.approx(rate)
    assert e["currency_orig"] == "USD" and e["amount_orig"] == 1.0 and e["fx_rate"] == pytest.approx(rate)
    assert "fx_converted" in e["flags"].split(",")
    assert bool(e["usable_rub"]) and e["drop_status"] != "non_rub"
    native = out.events[out.events["source_ref"] == "cbonds:2"].iloc[0]
    assert native["currency"] == "RUB" and np.isnan(native["fx_rate"]) and native["currency_orig"] is None


def test_an_event_with_no_recent_rate_stays_foreign_and_is_flagged():
    out = md.build_events(_cbonds(), _panel(), fx=_fx(end="2021-06-30"))
    e = out.events[out.events["source_ref"] == "cbonds:9"].iloc[0]
    assert e["currency"] == "USD" and e["amount"] == 1.0 and not bool(e["usable_rub"])
    assert "fx_missing" in e["flags"].split(",")


def test_a_converted_event_records_the_date_of_its_rate_and_others_leave_it_empty():
    out = md.build_events(_cbonds(), _panel(), fx=_fx())
    e = out.events[out.events["source_ref"] == "cbonds:9"].iloc[0]
    assert e["fx_date"] == TS("2021-09-14")  # a rate is set for the ex-date itself
    native = out.events[out.events["source_ref"] == "cbonds:2"].iloc[0]
    assert pd.isna(native["fx_date"])
    plain = md.build_events(_cbonds(), _panel())
    assert plain.events["fx_date"].isna().all()


def test_the_rate_of_the_last_date_before_the_ex_date_is_used_when_none_is_set_for_it():
    fx = _fx().drop(index=TS("2021-09-14"))
    out = md.build_events(_cbonds(), _panel(), fx=fx)
    e = out.events[out.events["source_ref"] == "cbonds:9"].iloc[0]
    assert e["fx_date"] == TS("2021-09-13")
    assert e["fx_rate"] == pytest.approx(float(fx.loc["2021-09-13", "USD"]))


def test_a_rate_set_after_the_ex_date_is_never_used():
    fx = pd.DataFrame({"USD": 75.0}, index=pd.bdate_range("2021-09-15", "2021-12-31"))
    out = md.build_events(_cbonds(), _panel(), fx=fx)
    e = out.events[out.events["source_ref"] == "cbonds:9"].iloc[0]  # ex-date 2021-09-14
    assert e["currency"] == "USD" and "fx_missing" in e["flags"].split(",") and pd.isna(e["fx_date"])


def test_converted_events_enter_the_index_flow_check():
    from cif.data import moex_dividend_checks as ck

    panel = _panel()
    plain = md.build_events(_cbonds(), panel)
    conv = md.build_events(_cbonds(), panel, fx=_fx())
    days = panel.close.index
    a = ck.index_validation(plain.events, panel.close, panel.adj, panel.membership, panel.index_flow, days)
    b = ck.index_validation(conv.events, panel.close, panel.adj, panel.membership, panel.index_flow, days)
    assert b.contrib["secid"].eq("SBER").sum() > a.contrib["secid"].eq("SBER").sum()


# --------------------------------------------------------------------------------------------
# Amount corrections
# --------------------------------------------------------------------------------------------


def _correction(**kw):
    base = {"secid": "SBER", "record_date": TS("2021-03-17"), "cbonds_amount": 0.05 * _prev(SBER, "2021-03-16"),
            "amount": 0.025 * _prev(SBER, "2021-03-16"), "currency": "RUB", "reason": "the export shows the full year",
            "source_ref": "correction:SBER:2021-03-17", "confidence": "medium"}
    return pd.DataFrame([{**base, **kw}])


def test_a_correction_replaces_the_vendor_amount_and_keeps_the_original():
    plain = md.build_events(_cbonds(), _panel())
    fixed = md.build_events(_cbonds(), _panel(), corrections=_correction())
    before = plain.events[plain.events["source_ref"] == "cbonds:2"].iloc[0]
    after = fixed.events[fixed.events["source_ref"] == "cbonds:2"].iloc[0]
    assert after["amount"] == pytest.approx(0.5 * before["amount"])
    assert after["cbonds_amount"] == pytest.approx(before["cbonds_amount"])
    assert "amount_corrected" in after["flags"].split(",")
    assert after["dividend_yield"] == pytest.approx(0.025, rel=1e-6)
    assert fixed.corrections_audit["matched_sheet_rows"].tolist() == [[2]]
    assert "amount_corrected" not in before["flags"].split(",")


def test_a_correction_that_matches_no_vendor_row_is_an_error():
    with pytest.raises(ValueError, match="matches no vendor row"):
        md.build_events(_cbonds(), _panel(), corrections=_correction(cbonds_amount=123.456))
    with pytest.raises(ValueError, match="matches no vendor row"):
        md.build_events(_cbonds(), _panel(), corrections=_correction(record_date=TS("2021-03-18")))


def test_a_correction_in_another_currency_does_not_touch_the_row():
    # row 9 is a 1.0 USD payment of SBER with record date 2021-09-15
    with pytest.raises(ValueError, match="matches no vendor row"):
        md.build_events(_cbonds(), _panel(), corrections=_correction(
            record_date=TS("2021-09-15"), cbonds_amount=1.0, amount=2.0, currency="RUB"))
    out = md.build_events(_cbonds(), _panel(), corrections=_correction(
        record_date=TS("2021-09-15"), cbonds_amount=1.0, amount=2.0, currency="USD"))
    assert out.events.loc[out.events["source_ref"] == "cbonds:9", "amount"].iloc[0] == 2.0


def test_a_correction_applies_to_every_row_of_a_duplicated_payment():
    # rows 7 and 8 are one payment listed twice; a correction keyed on that amount reaches both,
    # so that the duplicate collapse still sees equal amounts
    amount = 0.04 * _prev(SBER, "2021-06-15")
    out = md.build_events(_cbonds(), _panel(), corrections=_correction(
        record_date=TS("2021-06-16"), cbonds_amount=amount, amount=amount / 2))
    assert sorted(out.corrections_audit.iloc[0]["matched_sheet_rows"]) == [7, 8]
    e = out.events[out.events["record_date"] == TS("2021-06-16")]
    assert len(e) == 1 and e.iloc[0]["amount"] == pytest.approx(amount / 2)


# --------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------


def test_coverage_counts_events_per_ticker_and_year(built):
    cov = md.coverage_table(built.events)
    assert cov.loc["SBER", 2021] == 4 and cov.loc["SBER", 2022] == 2
    assert cov.loc["T", 2023] == 3 and cov.loc["TCSG", 2022] == 1 and cov.loc["TCSG", 2023] == 1
    assert cov.loc["GMKN", 2021] == 1
    assert cov.loc["SBER", 2023] == 0                 # years in between are filled with zeros
    assert cov.loc["SBER"].sum() == 6                 # the USD payments are not counted
    with_usd = md.coverage_table(built.events, currency=None)
    assert with_usd.loc["SBER"].sum() == 8
    padded = md.coverage_table(built.events, secids=["SBER", "NOPE"])
    assert padded.loc["NOPE"].sum() == 0
