"""Verified spin-off days of the S&P 500 panel (adjcorr-2; audit D-2, docs/REVIEW.md R6).

* the mechanism (``pit.apply_verified_events``) on synthetic panels, no network;
* the registry (``cif.data.sp500_events``): the facts read in the filings, pinned so that a typo
  shows up in review;
* the rebuilt dataset in ``data/processed/sp500_pit`` when it exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cif.data import sp500_events as ev
from cif.data import sp500_pit as pit

DAYS, K = 60, 40  # panel length and position of the event day
LATER = 50  # position of a split that follows the event
P0, NEW_PRICE, CHILD, N_CHILD, K_SHARES, RHO = 100.0, 184.0, 40.0, 0.25, 0.5, 0.43
# One old share becomes 0.5 new parent shares (92) and 0.25 child shares (10): the holder had 100
# before the day and 102 after it, +2 %. Yahoo books RHO as a ratio (no share-count ratio): every
# close before the day is divided by it, so its day return is 184 / (100 / 0.43) - 1 = -20.9 %.
TRUTH = (K_SHARES * NEW_PRICE + N_CHILD * CHILD) / P0 - 1.0
IDX = pd.bdate_range("2020-01-02", periods=DAYS, name="date")
DAY = IDX[K]


def _walk(n: int, start: float, seed: int) -> np.ndarray:
    ret = np.random.default_rng(seed).normal(0.0, 0.004, n)
    ret[0] = 0.0
    return start * np.cumprod(1.0 + ret)


@dataclass
class Scenario:
    adj: pd.DataFrame  # Yahoo's Adj Close of the parent and of a bystander
    close: pd.DataFrame  # Yahoo's Close of the same
    splits: pd.DataFrame
    children: pd.DataFrame | None
    event: ev.VerifiedEvent


def _event(**kw) -> ev.VerifiedEvent:
    base = {
        "ticker": "PAR", "name": "Parent Inc.", "date": str(DAY.date()), "yahoo_ratio": RHO,
        "parent_shares_per_share": K_SHARES,
        "distributions": (ev.Distribution("CHD", "Child Inc.", N_CHILD),),
        "record_date": str(IDX[K - 8].date()), "distribution_date": str(IDX[K - 1].date()),
        "sources": (ev.Source("https://example.org/8-k", "Form 8-K, Item 2.01", ("terms",)),),
        "verified_on": "2026-10-08",
        "price_checks": (
            ev.PriceCheck(
                "PAR", str(IDX[K - 1].date()), 99.5, 100.5, "test close", "https://example.org"
            ),
        ),
    }
    return ev.VerifiedEvent(**{**base, **kw})


def _scenario(
    *, parent_later: float = 1.0, child_later: float = 1.0, unlisted: float = 1.0,
    documented: float | None = None, basis: str = "close",
) -> Scenario:
    """Raw closes of a parent that spins a child off on ``DAY``, turned into what Yahoo serves.

    ``parent_later`` and ``child_later`` are splits dated after the day that the data list,
    ``unlisted`` is a split after the end of the data that Yahoo applied to every close but
    ``Stock Splits`` does not list (DD's 1:3 of 2026-06-24 is the real case).
    """
    parent = np.concatenate([_walk(K, 1.0, 1), _walk(DAYS - K, 1.0, 2)])
    parent[:K] *= P0 / parent[K - 1]
    parent[K:] *= NEW_PRICE / parent[K]
    child = np.full(DAYS, np.nan)
    child[K - 5 :] = _walk(DAYS - K + 5, 1.0, 3)
    child *= CHILD / child[K]
    control = _walk(DAYS, 50.0, 4)

    position = np.arange(DAYS)
    steps = np.where(position < K, RHO, 1.0)
    if parent_later != 1.0:
        steps = steps * np.where(position < LATER, parent_later, 1.0)
    close = pd.DataFrame({"PAR": parent / steps / unlisted, "CTRL": control}, index=IDX)
    splits = [("PAR", DAY, RHO)]
    if parent_later != 1.0:
        splits.append(("PAR", IDX[LATER], parent_later))
    if child_later != 1.0:
        splits.append(("CHD", IDX[LATER], child_later))
    frame = pd.DataFrame(splits, columns=["ticker", "date", "ratio"])

    if documented is None:
        distribution = ev.Distribution("CHD", "Child Inc.", N_CHILD)
        children = pd.DataFrame({"CHD": child / np.where(position < LATER, child_later, 1.0)}, index=IDX)
    else:
        distribution = ev.Distribution(
            "CHD", "Child Inc.", N_CHILD, price="documented", documented_close=documented,
            documented_basis=basis, documented_source="https://example.org/8937",
        )
        children = None
    event = _event(
        distributions=(distribution,),
        later_splits_unlisted=(("2030-01-01", unlisted),) if unlisted != 1.0 else (),
    )
    return Scenario(close.copy(), close, frame, children, event)


def _apply(sc: Scenario, **kw) -> pit.VerifiedCorrections:
    return pit.apply_verified_events(
        sc.adj, sc.close, sc.splits, [sc.event], children_close=sc.children, **kw
    )


# --------------------------------------------------------------------------------------------
# The mechanism
# --------------------------------------------------------------------------------------------


def test_the_version_names_the_second_correction_step():
    assert pit.ADJUSTMENT_CORRECTION_VERSION == "adjcorr-2"
    assert pit.ADJUSTMENT_CORRECTION_VERSION_GAPS == "adjcorr-1"


def test_verified_event_replaces_the_day_return_by_the_holders_total_return():
    sc = _scenario()
    out = _apply(sc)
    assert len(out.table) == 1 and out.skipped == {}
    row = out.table.iloc[0]
    assert list(out.table.columns) == pit.CORRECTION_COLUMNS
    assert (row["ticker"], row["date"], row["previous_date"]) == ("PAR", DAY, IDX[K - 1])
    assert row["kind"] == "verified_spin_off" and row["value_source"] == "sec_terms+child_close"
    assert row["split_ratio"] == pytest.approx(RHO) and row["dividend"] == 0.0
    assert row["return_yahoo"] == pytest.approx(NEW_PRICE / (P0 / RHO) - 1.0)
    assert row["return_close"] == pytest.approx(row["return_yahoo"])
    assert row["return_corrected"] == pytest.approx(TRUTH, abs=1e-12)
    assert pytest.approx(0.02) == TRUTH
    assert row["change"] == pytest.approx(TRUTH - row["return_yahoo"])
    assert row["scale_before"] == pytest.approx((1.0 + row["return_yahoo"]) / (1.0 + TRUTH))
    assert row["child_ticker"] == "CHD" and row["child_n"] == pytest.approx(N_CHILD)
    assert bool(row["child_exact"]) and bool(row["plausible"])
    new, old = out.prices_adj, sc.adj
    assert new.loc[DAY, "PAR"] / new.loc[IDX[K - 1], "PAR"] - 1.0 == pytest.approx(TRUTH)
    other = IDX.difference([DAY])
    pd.testing.assert_series_equal(
        new["PAR"].pct_change(fill_method=None).loc[other],
        old["PAR"].pct_change(fill_method=None).loc[other],
    )
    pd.testing.assert_series_equal(new.loc[DAY:, "PAR"], old.loc[DAY:, "PAR"])  # end anchored
    pd.testing.assert_series_equal(new["CTRL"], old["CTRL"])  # a bystander does not move


def test_the_correction_is_exactly_the_scale_before_of_the_day():
    sc = _scenario()
    out = _apply(sc)
    again = pit.apply_return_corrections(sc.adj, out.table)
    pd.testing.assert_frame_equal(again, out.prices_adj, check_exact=False, rtol=1e-12)
    back = pit.undo_return_corrections(out.prices_adj, out.table)
    pd.testing.assert_frame_equal(back, sc.adj, check_exact=False, rtol=1e-12)


def test_a_child_without_a_series_takes_its_documented_price():
    out = _apply(_scenario(documented=CHILD))
    row = out.table.iloc[0]
    assert row["return_corrected"] == pytest.approx(TRUTH, abs=1e-12)
    assert row["value_source"] == "sec_terms+documented_price" and bool(row["child_exact"])
    vwap = _apply(_scenario(documented=CHILD, basis="vwap")).table.iloc[0]
    assert vwap["return_corrected"] == pytest.approx(TRUTH, abs=1e-12)
    assert not bool(vwap["child_exact"])  # a volume weighted average is no close


def test_one_child_from_the_data_and_one_documented_are_both_in_the_total():
    sc = _scenario()
    second = ev.Distribution(
        "DOC", "Documented Inc.", 0.5, price="documented", documented_close=10.0,
        documented_source="https://example.org/10-k",
    )
    event = replace(sc.event, distributions=(*sc.event.distributions, second))
    out = pit.apply_verified_events(
        sc.adj, sc.close, sc.splits, [event], children_close=sc.children
    )
    row = out.table.iloc[0]
    expected = (K_SHARES * NEW_PRICE + N_CHILD * CHILD + 0.5 * 10.0) / P0 - 1.0
    assert row["return_corrected"] == pytest.approx(expected, abs=1e-12)
    assert row["value_source"] == "sec_terms+child_close+documented_price"
    assert row["child_ticker"] == "CHD+DOC" and np.isnan(row["child_n"])


@pytest.mark.parametrize(
    "later",
    [
        {"parent_later": 2.0},
        {"child_later": 3.0},
        {"unlisted": 0.25},
        {"parent_later": 2.0, "child_later": 3.0, "unlisted": 0.25},
    ],
)
def test_later_splits_are_part_of_the_units_not_of_the_value(later):
    row = _apply(_scenario(**later)).table.iloc[0]
    assert row["return_corrected"] == pytest.approx(TRUTH, abs=1e-12)
    expected = later.get("parent_later", 1.0) * later.get("unlisted", 1.0)
    assert row["later_split_factor"] == pytest.approx(expected)


def test_a_split_that_the_data_do_not_list_must_be_declared_and_the_price_check_says_so():
    sc = _scenario(unlisted=0.25)
    undeclared = replace(sc.event, later_splits_unlisted=())
    with pytest.raises(ValueError, match="price check"):
        pit.apply_verified_events(
            sc.adj, sc.close, sc.splits, [undeclared], children_close=sc.children
        )


def test_a_declared_split_that_the_data_list_as_well_is_refused_as_a_double_count():
    sc = _scenario()
    twice = replace(sc.event, later_splits_unlisted=((str(IDX[LATER].date()), 2.0),))
    listed = pd.concat(
        [sc.splits, pd.DataFrame([("PAR", IDX[LATER], 2.0)], columns=["ticker", "date", "ratio"])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="listed"):
        pit.apply_verified_events(sc.adj, sc.close, listed, [twice], children_close=sc.children)


def test_the_registered_yahoo_ratio_must_be_the_one_the_event_was_verified_against():
    sc = _scenario()
    with pytest.raises(ValueError, match="ratio"):
        pit.apply_verified_events(
            sc.adj, sc.close, sc.splits, [replace(sc.event, yahoo_ratio=0.5)],
            children_close=sc.children,
        )
    with pytest.raises(ValueError, match="ratio"):  # Yahoo books nothing on the day any more
        pit.apply_verified_events(
            sc.adj, sc.close, sc.splits.iloc[0:0], [sc.event], children_close=sc.children
        )


def test_a_price_outside_the_documented_range_stops_the_build():
    sc = _scenario()
    check = ev.PriceCheck("CHD", str(DAY.date()), 10.0, 20.0, "test range", "https://example.org")
    with pytest.raises(ValueError, match="price check"):
        pit.apply_verified_events(
            sc.adj, sc.close, sc.splits, [replace(sc.event, price_checks=(check,))],
            children_close=sc.children,
        )
    documented = _scenario(documented=CHILD)
    range_check = ev.PriceCheck(
        "CHD", str(DAY.date()), 50.0, 60.0, "range of the documented close", "https://example.org"
    )
    bad = replace(documented.event, price_checks=(range_check,))
    with pytest.raises(ValueError, match="price check"):  # the documented value is checked too
        pit.apply_verified_events(documented.adj, documented.close, documented.splits, [bad])


def test_a_day_that_the_gap_rule_already_corrected_is_refused():
    with pytest.raises(ValueError, match="already corrected"):
        _apply(_scenario(), corrected={("PAR", DAY)})


def test_a_child_that_has_no_close_on_the_day_is_an_error_not_a_guess():
    sc = _scenario()
    with pytest.raises(ValueError, match="child"):  # no series for a child that needs one
        pit.apply_verified_events(sc.adj, sc.close, sc.splits, [sc.event])
    sc.children.loc[DAY, "CHD"] = np.nan
    with pytest.raises(ValueError, match="child"):
        _apply(sc)


def test_a_parent_without_prices_on_the_day_is_an_error():
    sc = _scenario()
    sc.close.loc[IDX[K - 1], "PAR"] = np.nan
    sc.adj.loc[IDX[K - 1], "PAR"] = np.nan
    with pytest.raises(ValueError, match="price"):
        _apply(sc)


def test_events_outside_the_panel_are_skipped_and_listed_not_applied():
    sc = _scenario()
    elsewhere = replace(sc.event, ticker="NOPE")
    early = replace(sc.event, date=str(IDX[0].date()))
    out = pit.apply_verified_events(
        sc.adj, sc.close, sc.splits, [elsewhere, early, sc.event], children_close=sc.children
    )
    assert len(out.table) == 1 and out.table["ticker"].iloc[0] == "PAR"
    assert set(out.skipped) == {f"NOPE@{DAY.date()}", f"PAR@{IDX[0].date()}"}
    assert all(isinstance(reason, str) and reason for reason in out.skipped.values())
    none = pit.apply_verified_events(sc.adj, sc.close, sc.splits, [])
    pd.testing.assert_frame_equal(none.prices_adj, sc.adj)
    assert list(none.table.columns) == pit.CORRECTION_COLUMNS and none.table.empty


def test_the_value_of_the_correction_reads_nothing_dated_after_the_day():
    sc = _scenario()
    full = _apply(sc).table.iloc[0]["return_corrected"]
    cut = sc.close.loc[:DAY]
    short = pit.apply_verified_events(
        cut.copy(), cut, sc.splits, [sc.event], children_close=sc.children.loc[:DAY]
    ).table.iloc[0]["return_corrected"]
    assert short == pytest.approx(full, abs=1e-12)


def test_the_records_carry_the_terms_the_sources_and_every_number_used():
    (record,) = _apply(_scenario()).records
    json.dumps(record)  # the checks file is JSON
    assert record["ticker"] == "PAR" and record["date"] == str(DAY.date())
    assert record["previous_date"] == str(IDX[K - 1].date())
    terms = record["terms"]
    assert terms["parent_shares_per_share"] == K_SHARES and terms["yahoo_ratio"] == RHO
    assert terms["distributions"][0]["ticker"] == "CHD"
    assert terms["distributions"][0]["shares_per_parent_share"] == N_CHILD
    assert record["sources"][0]["url"] == "https://example.org/8-k"
    prices = record["prices"]
    assert prices["parent_close_previous"] == pytest.approx(P0)
    assert prices["parent_close_event"] == pytest.approx(NEW_PRICE)
    (child,) = prices["children"]
    assert child["ticker"] == "CHD" and child["close"] == pytest.approx(CHILD)
    assert child["source"] == "yahoo_close" and child["shares_per_parent_share"] == N_CHILD
    assert record["returns"]["corrected"] == pytest.approx(TRUTH)
    assert record["returns"]["yahoo_adjusted"] == pytest.approx(NEW_PRICE / (P0 / RHO) - 1.0)
    assert record["price_checks"][0]["ok"] is True and record["verified_on"] == "2026-10-08"


def test_volume_ratio_is_the_day_over_the_median_of_the_previous_sixty_days():
    sc = _scenario()
    volume = pd.DataFrame(1000.0, index=IDX, columns=["PAR", "CTRL"])
    volume.loc[DAY, "PAR"] = 4200.0
    assert _apply(sc, volume=volume).table["volume_ratio"].iloc[0] == pytest.approx(4.2)
    assert np.isnan(_apply(sc).table["volume_ratio"].iloc[0])


def test_gap_corrections_and_verified_events_compose_and_are_undone_together():
    sc = _scenario()
    pos = 20
    paid = 0.5 * sc.close["PAR"].iloc[pos - 1]  # a special dividend of half the price ...
    adj = sc.adj.copy()
    adj.loc[IDX[:pos], "PAR"] *= 1.0 - paid / sc.close["PAR"].iloc[pos - 1]  # ... Yahoo's way
    dividends = pd.DataFrame([("PAR", IDX[pos], paid)], columns=["ticker", "date", "amount"])
    gap = pit.correct_adjustment_artifacts(adj, sc.close, dividends, sc.splits)
    assert list(gap.table["kind"]) == ["cash_dividend"]
    done = {(r.ticker, r.date) for r in gap.table.itertuples()}
    out = pit.apply_verified_events(
        gap.prices_adj, sc.close, sc.splits, [sc.event], children_close=sc.children, corrected=done
    )
    table = pd.concat([gap.table, out.table], ignore_index=True)
    ret = out.prices_adj["PAR"].pct_change(fill_method=None)
    assert ret.loc[DAY] == pytest.approx(TRUTH)
    assert ret.loc[IDX[pos]] == pytest.approx(gap.table["return_corrected"].iloc[0])
    others = IDX.difference([DAY, IDX[pos]])
    pd.testing.assert_series_equal(
        ret.loc[others], adj["PAR"].pct_change(fill_method=None).loc[others]
    )
    back = pit.undo_return_corrections(out.prices_adj, table)
    pd.testing.assert_frame_equal(back, adj, check_exact=False, rtol=1e-12)


def test_the_gap_rule_does_not_see_the_verified_day_and_the_inventory_marks_it_corrected():
    sc = _scenario()
    gap = pit.correct_adjustment_artifacts(
        sc.adj, sc.close, pd.DataFrame(columns=["ticker", "date", "amount"]), sc.splits
    )
    assert gap.table.empty  # no dividend on the day: adjustment_gaps does not list it
    inventory = pit.pseudo_split_inventory(sc.adj, sc.close, sc.splits, corrected={("PAR", DAY)})
    assert list(inventory["ticker"]) == ["PAR"] and bool(inventory["corrected"].iloc[0])


# --------------------------------------------------------------------------------------------
# The registry: the facts as read in the filings
# --------------------------------------------------------------------------------------------

AUDIT_DAYS = {
    ("ITT", "2011-11-01"), ("VTR", "2015-08-18"), ("DD", "2019-06-03"), ("AIV", "2020-12-15"),
    ("LDOS", "2013-09-30"), ("HPQ", "2015-11-02"),
}
BY_TICKER = {e.ticker: e for e in ev.VERIFIED_EVENTS}


def test_the_registry_holds_the_six_days_of_audit_d2_once_each():
    days = [(e.ticker, e.date) for e in ev.VERIFIED_EVENTS]
    assert len(days) == len(set(days)) and set(days) == AUDIT_DAYS


@pytest.mark.parametrize("event", ev.VERIFIED_EVENTS, ids=lambda e: f"{e.ticker}@{e.date}")
def test_every_event_is_documented_by_a_filing_and_guards_its_units(event):
    assert event.sources and all(s.url.startswith("https://") and s.document for s in event.sources)
    assert any(s.kind == "sec" for s in event.sources)
    assert pd.Timestamp(event.record_date) < pd.Timestamp(event.distribution_date)
    assert pd.Timestamp(event.distribution_date) <= pd.Timestamp(event.date)
    assert 0.0 < event.parent_shares_per_share <= 1.0 and event.yahoo_ratio > 0.0
    assert pd.Timestamp(event.verified_on) <= pd.Timestamp("2026-12-31")
    assert event.distributions and all(d.shares_per_share > 0 for d in event.distributions)
    for d in event.distributions:
        if d.price == "documented":
            assert d.documented_close and d.documented_close > 0
            assert d.documented_source.startswith("https://")
            assert d.documented_basis in {"close", "vwap"}
        else:
            assert d.price == "yahoo" and d.documented_close is None
            assert any(c.ticker == d.ticker for c in event.price_checks), "unit chain of the child"
    assert any(c.ticker == event.ticker for c in event.price_checks), "unit chain of the parent"
    for check in event.price_checks:
        assert 0 < check.low <= check.high and check.source.startswith("https://") and check.basis
        pd.Timestamp(check.date)
    assert all(pd.Timestamp(d) > pd.Timestamp(event.date) for d, _ in event.later_splits_unlisted)


def test_terms_as_read_in_the_filings():
    terms = {
        t: (e.parent_shares_per_share, {d.ticker: d.shares_per_share for d in e.distributions})
        for t, e in BY_TICKER.items()
    }
    # ITT 8-K of 2011-10-31 (Items 2.01, 5.03): one Xylem and one Exelis share per ITT share, 1:2
    assert terms["ITT"] == (0.5, {"XYL": 1.0, "XLS": 1.0})
    # CCP information statement: one CCP share for every four Ventas shares
    assert terms["VTR"] == (1.0, {"CCP": 0.25})
    # Form 8937 of 2019-07-01: one Corteva share per three DowDuPont shares, then 1-for-3
    assert terms["DD"] == (pytest.approx(1 / 3), {"CTVA": pytest.approx(1 / 3)})
    # Aimco 8-K of 2020-12-15: one AIR share for each Aimco share, no reverse split
    assert terms["AIV"] == (1.0, {"AIRC": 1.0})
    # Leidos 8-K of 2013-10-01: one SAIC share for every seven shares, then 1-for-4 of Leidos
    assert terms["LDOS"] == (0.25, {"SAIC": pytest.approx(1 / 7)})
    # HP 8-K of 2015-11-05: one HPE share for every HP share
    assert terms["HPQ"] == (1.0, {"HPE": 1.0})


def test_dates_as_read_in_the_filings():
    dates = {t: (e.record_date, e.distribution_date, e.date) for t, e in BY_TICKER.items()}
    assert dates["ITT"] == ("2011-10-17", "2011-10-31", "2011-11-01")
    assert dates["VTR"] == ("2015-08-10", "2015-08-17", "2015-08-18")
    assert dates["DD"] == ("2019-05-24", "2019-06-01", "2019-06-03")
    assert dates["AIV"] == ("2020-12-05", "2020-12-15", "2020-12-15")
    assert dates["LDOS"] == ("2013-09-19", "2013-09-27", "2013-09-30")
    assert dates["HPQ"] == ("2015-10-21", "2015-11-01", "2015-11-02")


def test_children_without_a_yahoo_series_carry_a_documented_price():
    documented = {
        (e.ticker, d.ticker): (d.documented_close, d.documented_basis)
        for e in ev.VERIFIED_EVENTS for d in e.distributions if d.price == "documented"
    }
    assert documented == {
        ("ITT", "XLS"): (11.70, "close"),    # Rochester Business Journal, 2011-11-01
        ("VTR", "CCP"): (35.58, "close"),    # Ventas tax information: NYSE close 2015-08-18
        ("AIV", "AIRC"): (38.0711, "vwap"),  # Aimco Form 8937: volume weighted average 2020-12-15
    }


def test_dd_declares_the_one_for_three_reverse_split_of_2026_that_the_cache_does_not_list():
    assert BY_TICKER["DD"].later_splits_unlisted == (("2026-06-24", pytest.approx(1 / 3)),)
    assert all(not e.later_splits_unlisted for t, e in BY_TICKER.items() if t != "DD")


# --------------------------------------------------------------------------------------------
# The rebuilt dataset (skipped where data/processed is not copied in)
# --------------------------------------------------------------------------------------------

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "sp500_pit"
built = pytest.mark.skipif(
    not (DATA / "adjustment_corrections.parquet").exists(), reason="sp500_pit not built"
)


@built
def test_dataset_has_the_six_days_corrected_to_the_holders_return():
    table = pd.read_parquet(DATA / "adjustment_corrections.parquet")
    rows = table[table["kind"] == "verified_spin_off"]
    assert {(r.ticker, str(r.date.date())) for r in rows.itertuples()} == AUDIT_DAYS
    ret = pd.read_parquet(DATA / "prices_adj.parquet").pct_change(fill_method=None)
    for r in rows.itertuples():
        assert ret.at[r.date, r.ticker] == pytest.approx(r.return_corrected, abs=1e-12)
        assert abs(r.return_corrected) < 0.11 and abs(r.return_yahoo) > 0.10
    prov = json.loads((DATA / "provenance.json").read_text())
    assert prov["prices_adj_version"] == "adjcorr-2"
    assert prov["adjustment_corrections"]["version"] == "adjcorr-2"
    section = prov["adjustment_corrections"]["verified_events"]
    assert section["n_applied"] == 6 and not section["skipped"]


@built
def test_dataset_changes_no_return_outside_the_corrected_days():
    data = pit.load_dataset(DATA)
    new = data.prices_adj.pct_change(fill_method=None)
    old = data.prices_adj_yahoo().pct_change(fill_method=None)
    moved = ((new - old).abs() > 1e-9).stack()
    moved = {(ticker, date) for date, ticker in moved[moved].index}
    expected = {
        (r.ticker, r.date) for r in data.corrections.itertuples() if abs(r.scale_before - 1.0) > 1e-12
    }
    assert moved == expected
    assert len(data.corrections) == 32  # the 26 days of adjcorr-1 and the 6 verified days


@built
def test_dataset_documents_every_verified_day_in_the_checks_file():
    checks = json.loads((DATA / "corporate_action_checks.json").read_text())
    assert {f"{t}@{d}" for t, d in AUDIT_DAYS} <= set(checks)
    assert "BLDR@2015-04-13" in checks  # the earlier check is kept
    table = pd.read_parquet(DATA / "adjustment_corrections.parquet").set_index(["ticker", "date"])
    for t, d in AUDIT_DAYS:
        record = checks[f"{t}@{d}"]
        assert record["sources"] and record["price_checks"] and all(c["ok"] for c in record["price_checks"])
        row = table.loc[(t, pd.Timestamp(d))]
        assert record["returns"]["corrected"] == pytest.approx(row["return_corrected"], abs=1e-12)
        assert record["returns"]["yahoo_adjusted"] == pytest.approx(row["return_yahoo"], abs=1e-12)


@built
def test_dataset_prices_of_the_children_are_the_ones_in_the_checks_file():
    checks = json.loads((DATA / "corporate_action_checks.json").read_text())
    kids = {
        (t, c["ticker"]): c["close"]
        for t, d in AUDIT_DAYS for c in checks[f"{t}@{d}"]["prices"]["children"]
    }
    assert kids[("ITT", "XLS")] == 11.70 and kids[("VTR", "CCP")] == 35.58
    assert kids[("AIV", "AIRC")] == 38.0711
    assert kids[("ITT", "XYL")] == pytest.approx(27.31)
    assert kids[("LDOS", "SAIC")] == pytest.approx(33.75)
    assert kids[("DD", "CTVA")] == pytest.approx(24.81)
    assert kids[("HPQ", "HPE")] == pytest.approx(14.49)
