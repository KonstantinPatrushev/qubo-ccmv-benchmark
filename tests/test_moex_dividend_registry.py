"""Registry of dividends from official disclosures (``cif.data.moex_dividend_registry``).

The first block tests the machinery on small synthetic registries. The second pins the facts of
the registry as shipped: every event must carry a source that was read, and the facts typed into
the module are checked against the price data when ``data/processed/moex_v2`` is built.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace

import pandas as pd
import pytest

from cif.data import moex_dividend_registry as reg

S_ISSUER = reg.Source("https://example.org/ir/dividends", "IR page: the board recommended 5 RUB per share, "
                      "record date 2016-07-15", "issuer", ("amount", "record_date"))
S_NEWS = reg.Source("https://example.org/news/1", "News agency report of the AGM decision", "wire",
                    ("amount", "record_date"))
S_AGG = reg.Source("https://example.org/agg", "Aggregator table", "aggregator", ("amount",))


def _event(**kw):
    base = dict(
        secid="AAAA", issuer="Example plc", period="FY2015", period_type="год", record_date="2016-07-15",
        amount=5.0, currency="RUB", basis="per share", sources=(S_ISSUER,), verified_on="2026-10-08",
    )
    return reg.RegistryEvent(**{**base, **kw})


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def test_a_well_formed_event_passes():
    assert reg.validate([_event()]) == []


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"sources": ()}, "source"),
        ({"amount": 0.0}, "amount"),
        ({"amount": -1.0}, "amount"),
        ({"currency": "rub"}, "currency"),
        ({"currency": "ZZZ"}, "currency"),
        ({"record_date": "15.07.2016"}, "record_date"),
        ({"ex_date_declared": "2016-13-01"}, "ex_date_declared"),
        ({"payment_date": "soon"}, "payment_date"),
        ({"verified_on": ""}, "verified_on"),
        ({"basis": "per cow"}, "basis"),
        ({"confidence": "certain"}, "confidence"),
        ({"period_type": ""}, "period_type"),
    ],
)
def test_malformed_events_are_reported_with_the_field(change, fragment):
    problems = reg.validate([_event(**change)])
    assert problems and any(fragment in p for p in problems), problems


def test_a_source_needs_a_url_a_document_and_a_known_kind():
    for bad in (replace(S_ISSUER, url="not a url"), replace(S_ISSUER, document=""), replace(S_ISSUER, kind="blog")):
        assert reg.validate([_event(sources=(bad,))])


def test_an_event_resting_on_aggregators_alone_is_refused():
    assert reg.validate([_event(sources=(S_AGG,), confidence="low")])
    assert reg.validate([_event(sources=(S_AGG, S_NEWS), confidence="medium")]) == []


def test_high_confidence_needs_an_issuer_or_regulatory_source():
    assert reg.validate([_event(sources=(S_NEWS,), confidence="high")])
    assert reg.validate([_event(sources=(S_ISSUER,), confidence="high")]) == []
    exchange = replace(S_ISSUER, kind="exchange")
    assert reg.validate([_event(sources=(exchange,), confidence="high")]) == []


def test_duplicate_payments_are_reported():
    problems = reg.validate([_event(), _event(sources=(S_NEWS,))])
    assert any("duplicate" in p for p in problems)
    # two payments of one record date with different amounts are legitimate
    assert reg.validate([_event(), _event(amount=6.0)]) == []


def test_the_declared_ex_date_cannot_precede_the_record_date_by_more_than_the_cycle():
    # the ex-date is before the record date; a date after the record date cannot be an ex-date
    assert reg.validate([_event(ex_date_declared="2016-07-13")]) == []
    assert any("ex_date_declared" in p for p in reg.validate([_event(ex_date_declared="2016-07-18")]))


# --------------------------------------------------------------------------------------------
# The frame the loader reads
# --------------------------------------------------------------------------------------------


def test_registry_frame_has_the_loader_columns_and_types():
    frame = reg.registry_frame([_event(ex_date_declared="2016-07-13", payment_date="2016-08-01"),
                                _event(secid="BBBB", record_date="2017-07-14", currency="USD", basis="per GDR")])
    assert list(frame.columns[:8]) == ["secid", "record_date", "ex_date_declared", "amount", "currency",
                                       "period", "period_type", "source_ref"]
    assert frame["record_date"].tolist() == [pd.Timestamp("2016-07-15"), pd.Timestamp("2017-07-14")]
    assert frame.loc[0, "ex_date_declared"] == pd.Timestamp("2016-07-13")
    assert pd.isna(frame.loc[1, "ex_date_declared"])
    assert frame["source_ref"].is_unique
    assert frame.loc[0, "source_ref"].startswith("registry:AAAA:2016-07-15")
    assert frame.loc[0, "confidence"] == "medium" and frame.loc[0, "n_sources"] == 1
    assert frame.loc[1, "basis"] == "per GDR"


def test_registry_frame_refuses_an_invalid_registry():
    with pytest.raises(ValueError, match="amount"):
        reg.registry_frame([_event(amount=0.0)])


def test_registry_frame_of_nothing_is_an_empty_frame_with_columns():
    frame = reg.registry_frame([])
    assert frame.empty and "source_ref" in frame.columns


# --------------------------------------------------------------------------------------------
# No-payment evidence and tickers outside the window
# --------------------------------------------------------------------------------------------


def test_no_payment_evidence_needs_sources_and_a_scope():
    ok = reg.NoPayment("YNDX", "2014-08-16..2024-04-21", "annual reports state no dividends were paid",
                       (S_ISSUER,), "2026-10-08")
    assert reg.validate_no_payment([ok]) == []
    assert reg.validate_no_payment([replace(ok, sources=())])
    assert reg.validate_no_payment([replace(ok, scope="2014")])


# --------------------------------------------------------------------------------------------
# The registry as shipped
# --------------------------------------------------------------------------------------------


def test_the_shipped_registry_is_valid():
    assert reg.validate(reg.REGISTRY_EVENTS) == []
    assert reg.validate_no_payment(reg.NO_PAYMENT) == []


def test_every_shipped_event_names_a_source_read_on_a_known_date():
    for e in reg.REGISTRY_EVENTS:
        assert e.sources, e
        assert dt.date.fromisoformat(e.verified_on) <= dt.date(2026, 12, 31)
        assert all(s.url.startswith("http") for s in e.sources)


def test_the_shipped_registry_covers_only_tickers_that_cbonds_lacks_or_documents_why_not():
    # tickers absent from the Cbonds export (checked on 2026-10-08); the registry is for these
    assert set(e.secid for e in reg.REGISTRY_EVENTS) <= reg.CBONDS_ABSENT | reg.CBONDS_GAP_TICKERS


# --------------------------------------------------------------------------------------------
# The facts typed into the module, pinned so that a typo shows up in review
# --------------------------------------------------------------------------------------------


def _events(secid):
    return [e for e in reg.REGISTRY_EVENTS if e.secid == secid]


def test_polymetal_annual_totals_equal_the_totals_stated_in_the_rns():
    # The RNS state: FY2013 0.09, FY2014 0.41, FY2015 0.51, FY2016 0.42, FY2017 0.44, FY2018 0.48,
    # FY2019 0.82 and FY2020 1.29 US$ per share (read in the declarations of the next final dividends)
    totals = {}
    for e in _events("POLY"):
        year = int(e.period.split()[1][:4]) if e.period.startswith("H1") else int(e.period[2:6])
        totals[year] = round(totals.get(year, 0.0) + e.amount, 4)
    assert totals == {2013: 0.09, 2014: 0.41, 2015: 0.51, 2016: 0.42, 2017: 0.44, 2018: 0.48, 2019: 0.82,
                      2020: 1.29, 2021: 0.45}


def test_polymetal_events_are_the_21_payments_with_the_documented_dates():
    ev = sorted(_events("POLY"), key=lambda e: e.record_date)
    assert len(ev) == 21
    assert all(e.currency == "USD" and e.basis == "per share" for e in ev)
    assert [e.record_date for e in ev][:3] == ["2013-09-06", "2014-05-02", "2014-09-05"]
    assert ev[-1].record_date == "2021-09-10" and ev[-1].amount == 0.45
    # MOEX-specific ex-dates stated by the RNS of 2019-03-13 and 2020-03-04
    declared = {e.record_date: e.ex_date_declared for e in ev if e.ex_date_declared}
    assert declared == {"2019-05-10": "2019-05-08", "2020-05-11": "2020-05-07"}
    # the only event whose record date was not read on a page
    assert [e.record_date for e in ev if e.confidence != "high"] == ["2020-02-14"]
    # no payment after the interim of 2021: the cancelled final of 2021 is not an event
    assert not any(e.amount == 0.52 for e in ev)


def test_polymetal_finals_cite_the_meeting_that_passed_them():
    finals = [e for e in _events("POLY") if e.period_type == "год"]
    assert len(finals) == 8
    for e in finals:
        assert any("annual general meeting" in s.document for s in e.sources), e.period


def test_polyus_gold_international_final_dividend_for_2014():
    (e,) = _events("PGIL")
    assert (e.record_date, e.amount, e.currency, e.payment_date) == ("2015-05-15", 0.0608, "USD", "2015-05-29")
    assert e.confidence == "high" and e.sources[0].kind == "issuer"


def test_uralkali_events_and_the_rejected_interim():
    ev = sorted(_events("URKA"), key=lambda e: e.record_date)
    assert [(e.record_date, e.amount, e.currency) for e in ev] == [("2013-10-29", 2.21, "RUB"),
                                                                    ("2014-06-20", 1.63, "RUB")]
    assert [e.confidence for e in ev] == ["low", "high"]
    # the interim of RUB 2.96 (record date proposed 2015-01-15) was rejected by the meeting: no event
    assert not any(e.amount == 2.96 for e in ev)
    note = next(n for n in reg.NO_PAYMENT if n.secid == "URKA")
    assert "REJECTED" in note.note and note.scope == "2014-06-21..2017-10-22"


def test_megafon_events_and_the_calendar_year_totals_the_company_states():
    ev = sorted(_events("MFON"), key=lambda e: e.record_date)
    assert [(e.record_date, e.amount) for e in ev] == [
        ("2014-07-11", 64.51), ("2015-07-13", 16.13), ("2015-12-22", 64.51), ("2016-07-11", 48.38),
        ("2016-07-11", 8.06), ("2016-12-20", 24.19), ("2017-07-11", 32.25)]
    by_year = {}
    for e in ev:
        by_year[e.record_date[:4]] = round(by_year.get(e.record_date[:4], 0.0) + e.amount, 2)
    assert by_year["2015"] == 80.64   # the company's figure for 2015 (a news site printed 8.64: a typo)
    assert by_year["2016"] == 80.63   # "RUB 50 bn, RUB 80.63" in the March 2017 presentation
    assert [e.confidence for e in ev if e.confidence != "high"] == ["low"]  # the FY2016 final's record date
    assert not any(e.amount == 2.96 for e in ev)


def test_detsky_mir_amounts_times_the_voting_shares_equal_the_totals_in_the_reports():
    rows = reg._DSKY_ROWS
    assert len(rows) == 11
    for period, _, amount, total, *_ in rows:
        assert round(amount * 739_000_000) == total, period
    ev = _events("DSKY")
    assert min(e.record_date for e in ev) == "2017-07-17"
    assert max(e.record_date for e in ev) == "2021-12-26"
    assert all(e.sources[0].kind == "issuer" for e in ev)


def test_x5_amounts_are_per_gdr_in_roubles_and_four_gdrs_make_a_share():
    ev = _events("FIVE")
    assert all(e.basis == "per GDR" and e.currency == "RUB" for e in ev)
    per_share = {e.period: round(e.amount * 4, 2) for e in ev}
    assert per_share["FY2017"] == 318.00           # RUB 318.00 per ordinary share in the annual report
    assert abs(per_share["FY2018"] - 368.23) < 0.02
    assert round(110.49 + 73.64, 2) == 184.13      # the interim and the remainder make the 2020 total per GDR


def test_tcs_group_gdr_era_payments_of_2020_and_2021():
    ev = sorted(_events("TCSG"), key=lambda e: e.record_date)
    assert [(e.record_date, e.amount) for e in ev] == [
        ("2020-03-27", 0.21), ("2020-05-29", 0.14), ("2020-08-21", 0.20), ("2020-11-27", 0.25), ("2021-03-26", 0.24)]
    assert all(e.currency == "USD" and e.basis == "per GDR" for e in ev)
    # Vedomosti, 2022-05-18: TCS Group paid US$0.80 per GDR for 2020
    assert round(sum(e.amount for e in ev if e.record_date.startswith("2020")), 2) == 0.80
    # no issuer page could be read: the dates and amounts rest on the press, and only the second record date is inferred
    assert [e.confidence for e in ev] == ["medium", "low", "medium", "medium", "medium"]
    assert all(e.sources[0].kind in ("wire", "news") for e in ev)


def test_globaltrans_payments_are_in_roubles_per_gdr_with_the_gdr_ex_dates_the_issuer_states():
    ev = sorted(_events("GLTR"), key=lambda e: e.record_date)
    assert [(e.record_date, e.ex_date_declared, e.amount, e.currency, e.basis) for e in ev] == [
        ("2021-04-29", "2021-04-28", 28.0, "RUB", "per GDR"), ("2021-09-07", "2021-09-03", 22.5, "RUB", "per GDR")]
    assert all(e.confidence == "high" and e.sources[0].kind == "issuer" for e in ev)


def test_headhunter_payments_are_in_dollars_per_ordinary_share_and_ads():
    ev = sorted(_events("HHRU"), key=lambda e: e.record_date)
    assert [(e.record_date, e.amount, e.currency, e.basis) for e in ev] == [
        ("2021-06-09", 0.55, "USD", "per ADS"), ("2022-01-24", 0.84, "USD", "per ADS")]
    assert all(e.confidence == "high" and e.sources[0].kind == "regulatory" for e in ev)


def test_fix_price_pays_one_interim_dividend_inside_its_index_window():
    (e,) = _events("FIXP")
    assert (e.record_date, e.ex_date_declared, e.amount, e.currency, e.payment_date) == (
        "2021-09-24", "2021-09-23", 11.5, "RUB", "2021-09-28")
    # the final for 2021 (RUB 6.8 per GDR, record date 2022-05-27) was cancelled: not an event, named in the evidence
    note = next(n for n in reg.NO_PAYMENT if n.secid == "FIXP")
    assert "6.8" in note.statement and not any(x.amount == 6.8 for x in _events("FIXP"))


def test_ros_agro_payments_per_gdr_and_the_totals_the_company_states():
    ev = sorted(_events("AGRO"), key=lambda e: e.record_date)
    assert [(e.record_date, e.amount) for e in ev] == [
        ("2016-05-27", 0.58), ("2017-09-14", 0.23), ("2018-04-13", 0.29), ("2018-09-14", 0.12), ("2019-09-13", 0.19),
        ("2020-04-17", 0.32), ("2020-09-18", 0.19), ("2021-04-16", 1.04), ("2021-09-10", 0.89)]
    assert all(e.currency == "USD" and e.basis == "per GDR" for e in ev)
    by = {e.record_date: e for e in ev}
    # the company's total for 2019 per GDR (the interim and the final): US$0.51
    assert round(by["2019-09-13"].amount + by["2020-04-17"].amount, 2) == 0.51
    # every ex-date the issuer states is the day before the record date
    for e in ev:
        if e.ex_date_declared:
            assert (dt.date.fromisoformat(e.record_date) - dt.date.fromisoformat(e.ex_date_declared)).days == 1, e
    # the amounts of the weaker events: the first (an aggregator's per-GDR figure) and the derived one
    assert [e.record_date for e in ev if e.confidence == "low"] == ["2016-05-27", "2020-09-18"]
    # the final for 2021 (US$1.02 per GDR, record date 2022-04-01) was not paid: no event
    assert not any(e.amount == 1.02 for e in ev)


def test_no_payment_evidence_added_for_issuers_without_dividends_covers_their_index_windows():
    windows = {"RNFT": ("2017-09-22", "2019-12-19"), "UWGN": ("2016-09-16", "2018-12-20"),
               "PHST": ("2013-09-02", "2016-06-15"), "ENPG": ("2021-12-17", "2025-12-30"),
               "VKCO": ("2021-12-14", "2025-12-30"), "YNDX": ("2014-09-16", "2024-03-21")}
    for secid, (start, end) in windows.items():
        scopes = [tuple(n.scope.split("..")) for n in reg.NO_PAYMENT if n.secid == secid]
        assert any(a <= start and b >= end for a, b in scopes), secid


def test_no_payment_evidence_for_mechel_ordinary_shares_covers_the_index_window():
    n = next(n for n in reg.NO_PAYMENT if n.secid == "MTLR")
    assert n.scope.split("..")[0] == "2013-09-02" and n.scope.split("..")[1] >= "2025-06-19"  # MTLR left IMOEX 2025-06-19
    assert any(s.kind == "issuer" for s in n.sources)


# --------------------------------------------------------------------------------------------
# Leads: what is known about payments that are not in the registry
# --------------------------------------------------------------------------------------------


def _lead(**kw):
    base = dict(secid="RSTI", what="Q1 2016 interim dividend, RUB 0.00831813 per ordinary share",
                why_open="the record date was not read on any page", sources=(S_NEWS,), verified_on="2026-10-08")
    return reg.Lead(**{**base, **kw})


def test_a_well_formed_lead_passes_and_malformed_ones_are_reported():
    assert reg.validate_leads([_lead()]) == []
    assert reg.validate_leads([_lead(what="")])
    assert reg.validate_leads([_lead(why_open="  ")])
    assert reg.validate_leads([_lead(sources=())])
    assert reg.validate_leads([_lead(verified_on="yesterday")])
    assert reg.validate_leads([_lead(), _lead()])  # the same lead twice


def test_leads_frame_has_one_row_per_lead_with_the_urls():
    frame = reg.leads_frame([_lead(), _lead(secid="AGRO", what="1H 2016 interim")])
    assert list(frame.columns) == ["secid", "what", "why_open", "n_sources", "urls", "verified_on"]
    assert frame["secid"].tolist() == ["AGRO", "RSTI"] and frame["urls"].iloc[1] == "https://example.org/news/1"
    assert reg.leads_frame([]).empty
    with pytest.raises(ValueError, match="invalid"):
        reg.leads_frame([_lead(what="")])


def test_the_shipped_leads_are_valid_and_belong_to_tickers_the_registry_is_for():
    assert reg.validate_leads(reg.LEADS) == []
    assert {x.secid for x in reg.LEADS} <= reg.CBONDS_ABSENT | reg.CBONDS_GAP_TICKERS
    # Rosseti's ordinary-share payments are documented but left out as small (mean weight 0.2 % of the index)
    assert {"RSTI", "AGRO", "FIVE"} <= {x.secid for x in reg.LEADS}


# --------------------------------------------------------------------------------------------
# Corrections of vendor amounts
# --------------------------------------------------------------------------------------------


def _correction(**kw):
    base = dict(secid="BANE", record_date="2014-06-23", cbonds_amount=410.0, amount=211.0,
                reason="the export's year row is the sum of the interim and the final payment",
                sources=(S_NEWS,), verified_on="2026-10-08")
    return reg.AmountCorrection(**{**base, **kw})


def test_a_well_formed_correction_passes():
    assert reg.validate_corrections([_correction()]) == []


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"sources": ()}, "source"),
        ({"amount": 410.0}, "differs"),
        ({"amount": 0.0}, "amount"),
        ({"cbonds_amount": -1.0}, "cbonds_amount"),
        ({"record_date": "23.06.2014"}, "record_date"),
        ({"reason": ""}, "reason"),
        ({"currency": "XXX"}, "currency"),
        ({"sources": (S_AGG,)}, "aggregator"),
        ({"verified_on": "yesterday"}, "verified_on"),
        ({"confidence": "sure"}, "confidence"),
    ],
)
def test_malformed_corrections_are_reported(change, fragment):
    problems = reg.validate_corrections([_correction(**change)])
    assert problems and any(fragment in p for p in problems), problems


def test_two_corrections_of_one_vendor_row_are_refused():
    assert any("duplicate" in p for p in reg.validate_corrections([_correction(), _correction(amount=200.0)]))


def test_corrections_frame_has_the_columns_the_loader_reads():
    frame = reg.corrections_frame([_correction()])
    assert list(frame.columns[:6]) == ["secid", "record_date", "cbonds_amount", "amount", "currency", "reason"]
    assert frame.loc[0, "record_date"] == pd.Timestamp("2014-06-23")
    assert frame.loc[0, "source_ref"] == "correction:BANE:2014-06-23:410"
    with pytest.raises(ValueError, match="differs"):
        reg.corrections_frame([_correction(amount=410.0)])


def test_the_shipped_corrections_are_valid():
    assert reg.validate_corrections(reg.CORRECTIONS) == []
