"""Tests for the point-in-time S&P 500 universe (no network, synthetic fixtures)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from cif.data import sp500_events as ev
from cif.data import sp500_pit as pit

# --------------------------------------------------------------------------------------------
# Synthetic Wikipedia pages (same structure as the live ones: two-row header on the changes table)
# --------------------------------------------------------------------------------------------

CONSTITUENTS_HTML = """
<html><body>
<table class="wikitable sortable" id="constituents">
<thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
<th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr></thead>
<tbody>
<tr><td>AAPL</td><td><a href="/wiki/Apple">Apple Inc.</a></td><td>Information Technology</td>
<td>Hardware</td><td>Cupertino</td><td>1982-11-30</td><td>0000320193</td><td>1977</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td><td>Multi-Sector</td>
<td>Omaha</td><td>2010-02-16</td><td>0001067983</td><td>1839</td></tr>
<tr><td>GOOGL</td><td>Alphabet Inc. (Class A)</td><td>Communication Services</td><td>Interactive</td>
<td>Mountain View</td><td>2006-04-03</td><td>0001652044</td><td>1998</td></tr>
<tr><td>GOOG</td><td>Alphabet Inc. (Class C)</td><td>Communication Services</td><td>Interactive</td>
<td>Mountain View</td><td>2014-04-03</td><td>0001652044</td><td>1998</td></tr>
</tbody></table></body></html>
"""

CHANGES_HTML = """
<html><body>
<table class="wikitable sortable" id="changes"><tbody>
<tr><th data-sort-type="date" rowspan="2">Effective Date</th><th colspan="2">Added</th>
<th colspan="2">Removed</th><th rowspan="2">Reason</th><th rowspan="2">Refs</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>March 24, 2025</td><td>DASH</td><td><a href="/wiki/DoorDash">DoorDash</a></td><td>BWA</td>
<td>BorgWarner</td><td>Market capitalization changes.</td>
<td><sup><a href="#n"><span>[</span>13<span>]</span></a></sup></td></tr>
<tr><td>June 29, 2025</td><td>HONA</td><td>Honeywell Aerospace</td><td></td><td></td>
<td>Honeywell completed the <a href="/wiki/Spin-off">spin-off</a>.</td><td></td></tr>
<tr><td>June 2, 2024</td><td></td><td></td><td>EPAM</td><td>EPAM Systems</td>
<td>Market capitalization changes.</td><td></td></tr>
<tr><td>December 2, 2013</td><td>ALLE |</td><td>Allegion</td><td>JCP |</td><td>JCPenney</td>
<td>Spin-off of Allegion.</td><td></td></tr>
<tr><td>January 10, 2022</td><td>BRK.B</td><td>Berkshire</td><td>BF.B</td><td>Brown-Forman</td>
<td></td><td></td></tr>
</tbody></table></body></html>
"""

# the changes table used to live on the same page as the constituents; the parser must not care
SINGLE_PAGE_HTML = CONSTITUENTS_HTML.replace(
    "</body>", CHANGES_HTML.split("<body>")[1].split("</body>")[0] + "</body>"
)


def _ts(x: str) -> pd.Timestamp:
    return pd.Timestamp(x)


def _constituents(rows: list[tuple]) -> pd.DataFrame:
    """rows: (ticker, security, date_added) -> frame as returned by ``parse_constituents``."""
    df = pd.DataFrame(rows, columns=["ticker", "security", "date_added"])
    df["date_added"] = pd.to_datetime(df["date_added"])
    return df


def _changes(rows: list[tuple]) -> pd.DataFrame:
    """rows: (date, +ticker, +security, -ticker, -security, reason), None for empty cells."""
    df = pd.DataFrame(
        rows,
        columns=[
            "date", "added_ticker", "added_security", "removed_ticker", "removed_security", "reason"
        ],
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def _members(rec: pit.Reconstruction, day: str, start="2010-01-01", end="2025-12-31") -> set[str]:
    matrix = pit.intervals_to_membership(rec.intervals, start, end)
    row = matrix.loc[pd.Timestamp(day)]
    return set(row.index[row.to_numpy()])


# --------------------------------------------------------------------------------------------
# Snapshot and parsing
# --------------------------------------------------------------------------------------------


def test_save_html_snapshot_records_sha256_of_the_saved_bytes(tmp_path):
    html = "<html><script>{\"wgRevisionId\":1376729338}</script>Привет</html>"
    snap = pit.save_html_snapshot(html, tmp_path, label="constituents")
    assert snap.path.exists() and snap.path.parent == tmp_path
    assert snap.sha256 == hashlib.sha256(snap.path.read_bytes()).hexdigest()
    assert snap.n_bytes == snap.path.stat().st_size
    assert snap.revision_id == 1376729338
    assert "constituents" in snap.path.name
    meta = snap.path.with_suffix(".meta.json")
    assert meta.exists()
    assert pit.HtmlSnapshot.from_dict(json.loads(meta.read_text())) == snap


def test_extract_revision_id_absent():
    assert pit.extract_revision_id("<html>no config here</html>") is None


def test_fetch_wikipedia_snapshots_reuses_the_page_when_it_holds_both_tables(tmp_path):
    seen = []

    def fetcher(url):
        seen.append(url)
        return SINGLE_PAGE_HTML

    snaps = pit.fetch_wikipedia_snapshots(tmp_path, fetcher=fetcher)
    assert seen == [pit.WIKI_SP500_URL]
    assert snaps.constituents.path == snaps.changes.path


def test_fetch_wikipedia_snapshots_follows_the_changes_table_to_its_own_page(tmp_path):
    pages = {pit.WIKI_SP500_URL: CONSTITUENTS_HTML, pit.WIKI_CHANGES_URL: CHANGES_HTML}
    seen = []

    def fetcher(url):
        seen.append(url)
        return pages[url]

    snaps = pit.fetch_wikipedia_snapshots(tmp_path, fetcher=fetcher)
    assert seen == [pit.WIKI_SP500_URL, pit.WIKI_CHANGES_URL]
    assert snaps.constituents.path != snaps.changes.path
    assert snaps.changes.url == pit.WIKI_CHANGES_URL
    assert len(pit.parse_changes(snaps.changes.path.read_text(encoding="utf-8"))) == 5


def test_load_wikipedia_snapshots_returns_the_latest_and_checks_the_hash(tmp_path):
    pages = {pit.WIKI_SP500_URL: CONSTITUENTS_HTML, pit.WIKI_CHANGES_URL: CHANGES_HTML}
    pit.fetch_wikipedia_snapshots(tmp_path, fetcher=pages.__getitem__)
    loaded = pit.load_wikipedia_snapshots(tmp_path)
    assert loaded.constituents.url == pit.WIKI_SP500_URL
    assert loaded.changes.url == pit.WIKI_CHANGES_URL
    loaded.changes.path.write_text(CHANGES_HTML + "<!-- edited -->", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        pit.load_wikipedia_snapshots(tmp_path)
    with pytest.raises(FileNotFoundError):
        pit.load_wikipedia_snapshots(tmp_path / "empty")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AAPL", "AAPL"),
        ("brk.b", "BRK-B"),
        ("BF.B", "BF-B"),
        ("ALLE |", "ALLE"),
        (" JCP | ", "JCP"),
        ("MMM[1]", "MMM"),
        ("", None),
        ("—", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_to_yahoo_symbol(raw, expected):
    assert pit.to_yahoo_symbol(raw) == expected


def test_parse_constituents():
    cur = pit.parse_constituents(CONSTITUENTS_HTML)
    assert list(cur["ticker"]) == ["AAPL", "BRK-B", "GOOGL", "GOOG"]
    assert cur.loc[cur["ticker"] == "BRK-B", "date_added"].iloc[0] == _ts("2010-02-16")
    # CIK is a zero-padded 10-digit string even though pandas reads it as an integer
    assert cur.loc[cur["ticker"] == "AAPL", "cik"].iloc[0] == "0000320193"
    assert cur.loc[cur["ticker"] == "GOOG", "security"].iloc[0] == "Alphabet Inc. (Class C)"
    assert set(cur.columns) >= {"ticker", "security", "sector", "cik", "date_added"}


def test_parse_changes_handles_two_row_header_blank_cells_and_pipes():
    ch = pit.parse_changes(CHANGES_HTML)
    assert len(ch) == 5
    first = ch.iloc[0]
    assert first["date"] == _ts("2025-03-24")
    assert (first["added_ticker"], first["added_security"]) == ("DASH", "DoorDash")
    assert (first["removed_ticker"], first["removed_security"]) == ("BWA", "BorgWarner")
    assert first["reason"] == "Market capitalization changes."
    spin = ch.iloc[1]  # added only
    assert spin["added_ticker"] == "HONA" and spin["removed_ticker"] is None
    drop = ch.iloc[2]  # removed only
    assert drop["added_ticker"] is None and drop["removed_ticker"] == "EPAM"
    pipes = ch.iloc[3]
    assert (pipes["added_ticker"], pipes["removed_ticker"]) == ("ALLE", "JCP")
    dots = ch.iloc[4]
    assert (dots["added_ticker"], dots["removed_ticker"]) == ("BRK-B", "BF-B")


def test_parse_changes_works_when_both_tables_share_a_page():
    assert len(pit.parse_changes(SINGLE_PAGE_HTML)) == 5
    assert len(pit.parse_constituents(SINGLE_PAGE_HTML)) == 4
    assert pit.has_changes_table(SINGLE_PAGE_HTML)
    assert not pit.has_changes_table(CONSTITUENTS_HTML)


def test_parse_changes_without_table_raises():
    with pytest.raises(ValueError, match="changes"):
        pit.parse_changes(CONSTITUENTS_HTML)


# --------------------------------------------------------------------------------------------
# Company names and identities
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Alphabet Inc. (Class A)", "Alphabet Inc. (Class C)"),
        ("News Corp (Class A)", "News Corporation Class B"),
        ("Comcast Class K Special", "Comcast Series K"),
        ("Under Armour (Class C)", "Under Armour"),
        ("Willis Towers Watson", "WILLIS TOWERS WATSON"),
    ],
)
def test_normalize_company_name_merges_share_classes(a, b):
    assert pit.normalize_company_name(a) == pit.normalize_company_name(b) != ""


def test_normalize_company_name_keeps_different_companies_apart():
    assert pit.normalize_company_name("Dow Chemical Company") != pit.normalize_company_name("DuPont")
    assert pit.normalize_company_name("") == ""
    assert pit.normalize_company_name(None) == ""


def test_assign_company_ids_by_cik_and_by_name():
    companies = pd.DataFrame(
        {
            "ticker": ["GOOGL", "GOOG", "AAPL", "UA", "UAA", "XYZ1", "XYZ2"],
            "security": [
                "Alphabet Inc. (Class A)", "Alphabet Inc. (Class C)", "Apple Inc.",
                "Under Armour (Class C)", "Under Armour (Class A)", "Foo Corp", "Bar Corp",
            ],
            # the two Alphabet rows share a CIK even though the names are written differently
            "cik": ["0001652044", "0001652044", "0000320193", None, None, "0000000042", "0000000042"],
        }
    )
    ids = pit.assign_company_ids(companies)
    assert ids["GOOGL"] == ids["GOOG"]
    assert ids["UA"] == ids["UAA"]  # no CIK, normalised names agree
    assert ids["XYZ1"] == ids["XYZ2"]  # CIK wins over different names
    assert len({ids["AAPL"], ids["GOOG"], ids["UA"], ids["XYZ1"]}) == 4


# --------------------------------------------------------------------------------------------
# Reconstruction of the membership
# --------------------------------------------------------------------------------------------


def test_reconstruct_basic_replacements_spinoffs_and_drops():
    cur = _constituents(
        [
            ("AAA", "Alpha", "1990-01-01"),
            ("BBB", "Beta", "2020-03-02"),
            ("CCC", "Gamma", "2021-06-15"),
            ("DDD", "Delta", "2022-03-01"),
        ]
    )
    ch = _changes(
        [
            ("2022-03-01", "DDD", "Delta", "EEE", "Epsilon", "Market capitalization changes."),
            ("2021-06-15", "CCC", "Gamma", "FFF", "Phi", "Phi was acquired."),
            ("2021-01-04", None, None, "HHH", "Eta", "Market capitalization changes."),
            ("2020-03-02", "BBB", "Beta", "GGG", "Gee", "Gee was acquired."),
            ("2019-05-05", "HHH", "Eta", None, None, "Spin-off of Eta."),
            ("2018-02-02", None, None, "III", "Iota", "Market capitalization changes."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert len(rec.anomalies) == 0
    expected = {
        "2010-01-01": {"AAA", "EEE", "FFF", "GGG", "III"},
        "2018-02-01": {"AAA", "EEE", "FFF", "GGG", "III"},
        "2018-02-02": {"AAA", "EEE", "FFF", "GGG"},
        "2019-05-05": {"AAA", "EEE", "FFF", "GGG", "HHH"},
        "2020-03-01": {"AAA", "EEE", "FFF", "GGG", "HHH"},
        "2020-03-02": {"AAA", "BBB", "EEE", "FFF", "HHH"},
        "2021-01-03": {"AAA", "BBB", "EEE", "FFF", "HHH"},
        "2021-01-04": {"AAA", "BBB", "EEE", "FFF"},
        "2021-06-14": {"AAA", "BBB", "EEE", "FFF"},
        "2021-06-15": {"AAA", "BBB", "CCC", "EEE"},
        "2022-02-28": {"AAA", "BBB", "CCC", "EEE"},
        "2022-03-01": {"AAA", "BBB", "CCC", "DDD"},
        "2025-12-31": {"AAA", "BBB", "CCC", "DDD"},
    }
    for day, members in expected.items():
        assert _members(rec, day) == members, day


def test_reconstruct_is_independent_of_row_order():
    cur = _constituents([("AAA", "Alpha", "1990-01-01"), ("BBB", "Beta", "2020-03-02")])
    ch = _changes(
        [
            ("2020-03-02", "BBB", "Beta", "GGG", "Gee", "x"),
            ("2015-06-01", "GGG", "Gee", "HHH", "Eta", "x"),
        ]
    )
    a = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    b = pit.reconstruct_membership(cur, ch.iloc[::-1].reset_index(drop=True), start="2010-01-01",
                                   as_of="2026-01-01")
    for day in ["2012-01-01", "2016-01-01", "2021-01-01"]:
        assert _members(a, day) == _members(b, day)
    assert _members(a, "2012-01-01") == {"AAA", "HHH"}
    assert _members(a, "2016-01-01") == {"AAA", "GGG"}


def test_reconstruct_ignores_rows_dated_after_the_snapshot():
    cur = _constituents([("AAA", "Alpha", "1990-01-01")])
    ch = _changes([("2026-03-01", "ZZZ", "Zed", "AAA", "Alpha", "announced, not effective yet")])
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2025-12-31") == {"AAA"}
    assert "future_dated" in set(rec.anomalies["kind"])


def test_reconstruct_rename_stated_in_the_reason_is_merged():
    cur = _constituents([("NEW", "Foo Inc", "2015-01-01"), ("AAA", "Alpha", "1990-01-01")])
    ch = _changes(
        [
            ("2021-01-04", "NEW", "Foo Inc", "OLD", "Foo Inc",
             "Foo Inc changed its ticker symbol from OLD to NEW."),
            ("2015-01-01", "OLD", "Foo Inc", "ZZZ", "Zed", "Zed was acquired."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2014-12-31") == {"AAA", "ZZZ"}
    assert _members(rec, "2015-01-01") == {"AAA", "NEW"}
    assert _members(rec, "2020-12-31") == {"AAA", "NEW"}  # OLD's interval now belongs to NEW
    assert "OLD" not in set(rec.intervals["ticker"])
    renames = rec.renames.set_index("old")
    assert renames.loc["OLD", "new"] == "NEW"
    assert renames.loc["OLD", "rule"] == "reason"


def test_reconstruct_rename_with_identical_names_is_merged_without_keyword():
    cur = _constituents([("BKR", "Baker Hughes", "2017-07-07")])
    ch = _changes([("2017-07-07", "BKR", "Baker Hughes", "BHI", "Baker Hughes", "merged with GE")])
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert set(rec.intervals["ticker"]) == {"BKR"}
    assert _members(rec, "2012-05-05") == {"BKR"}
    assert rec.renames.iloc[0]["rule"] == "name"


def test_reconstruct_does_not_treat_unrelated_rename_keyword_as_rename():
    # the reason talks about the *added* company only; the removed one is a different company
    cur = _constituents([("EVRG", "Evergy", "2018-06-05")])
    ch = _changes(
        [
            ("2018-06-05", "EVRG", "Evergy", "NAVI", "Navient",
             "Westar Energy acquired Great Plains Energy and changed name to Evergy."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert len(rec.renames) == 0
    assert _members(rec, "2018-06-04") == {"NAVI"}
    assert _members(rec, "2018-06-05") == {"EVRG"}


def test_reconstruct_rename_needs_both_sides_in_the_reason():
    # the reason explains why Allergan left; it does not say Allergan became American Airlines
    cur = _constituents([("AAL", "American Airlines Group", "2015-03-23")])
    ch = _changes(
        [
            ("2015-03-23", "AAL", "American Airlines Group", "AGN", "Allergan",
             "Allergan was acquired by Actavis (and changed its name to Allergan)."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert len(rec.renames) == 0 and len(rec.anomalies) == 0
    assert _members(rec, "2015-03-22") == {"AGN"}
    assert _members(rec, "2015-03-23") == {"AAL"}


def test_reconstruct_survivor_rename_resolved_by_date_added():
    # Facebook was added as FB; the current list knows it as META with the same "Date added"
    cur = _constituents(
        [("META", "Meta Platforms", "2013-12-23"), ("TER", "Teradyne", "2020-09-21")]
    )
    ch = _changes(
        [
            ("2020-09-21", "TER", "Teradyne", "KSS", "Kohl's", "Market capitalization changes."),
            ("2013-12-23", "FB", "Facebook", "TER", "Teradyne", "Market capitalization changes."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert "FB" not in set(rec.intervals["ticker"])
    assert _members(rec, "2013-12-22") == {"TER", "KSS"}
    assert _members(rec, "2013-12-23") == {"META", "KSS"}
    assert _members(rec, "2020-09-20") == {"META", "KSS"}
    assert _members(rec, "2020-09-21") == {"META", "TER"}
    ren = rec.renames.set_index("old").loc["FB"]
    assert (ren["new"], ren["rule"]) == ("META", "date_added")
    assert len(rec.anomalies) == 0


def test_reconstruct_date_added_tolerates_a_few_days_of_slack():
    # the current list dates the arrival 3 days after the row (the two tables use different
    # conventions for "effective" days)
    cur = _constituents([("BKNG", "Booking Holdings", "2011-01-06"), ("AAA", "Alpha", "1990-01-01")])
    ch = _changes([("2011-01-03", "PCLN", "Priceline", "YYY", "Why", "x")])
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2011-01-02") == {"AAA", "YYY"}
    assert _members(rec, "2011-01-03") == {"AAA", "BKNG"}
    assert rec.renames.iloc[0]["new"] == "BKNG"


def test_reconstruct_date_added_far_from_the_row_is_not_a_match():
    cur = _constituents([("BKNG", "Booking Holdings", "2011-02-15"), ("AAA", "Alpha", "1990-01-01")])
    ch = _changes([("2011-01-03", "PCLN", "Priceline", "YYY", "Why", "x")])
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert "added_not_active" in set(rec.anomalies["kind"])
    assert len(rec.renames) == 0


def test_reconstruct_swaps_share_class_row_using_date_added():
    # the table says "GOOGL added 2014-04-03", the current list says GOOG was added on that day
    # and GOOGL in 2006: the added ticker of the row is really GOOG
    cur = _constituents(
        [("GOOGL", "Alphabet Inc. (Class A)", "2006-04-03"),
         ("GOOG", "Alphabet Inc. (Class C)", "2014-04-03")]
    )
    ch = _changes([("2014-04-03", "GOOGL", "Google", None, None, "Class C share distribution.")])
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2012-01-01") == {"GOOGL"}
    assert _members(rec, "2014-04-02") == {"GOOGL"}
    assert _members(rec, "2014-04-03") == {"GOOGL", "GOOG"}
    assert rec.renames.iloc[0]["rule"] == "date_added_swap"


def test_reconstruct_resolves_unmatched_add_by_company_name():
    cur = _constituents([("AAA", "Alpha", "1990-01-01"), ("KDP", "Keurig Dr Pepper", "2022-06-21")])
    ch = _changes(
        [
            ("2022-06-21", None, None, "UA", "Under Armour (Class C)", "Market capitalization changes."),
            ("2022-06-21", "KDP", "Keurig Dr Pepper", "UAA", "Under Armour (Class A)", "x"),
            ("2016-04-08", "UA", "Under Armour (Class C)", None, None, "Distribution of class C."),
            ("2014-05-01", "UA", "Under Armour", "BEAM", "Beam Inc.", "Beam was acquired."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2014-04-30") == {"AAA", "BEAM"}
    assert _members(rec, "2014-05-01") == {"AAA", "UAA"}
    assert _members(rec, "2016-04-08") == {"AAA", "UAA", "UA"}
    assert _members(rec, "2022-06-20") == {"AAA", "UAA", "UA"}
    assert _members(rec, "2022-06-21") == {"AAA", "KDP"}
    assert rec.renames.set_index("old").loc["UA", "rule"] == "name"


def test_reconstruct_flags_but_keeps_a_member_whose_date_added_has_no_row():
    # "Date added" of the current list is not the first day in the index for every company
    # (Sempra, Dominion ... carry later dates), so it is reported and never used to cut history
    cur = _constituents([("AAA", "Alpha", "1990-01-01"), ("BRK-B", "Berkshire", "2010-02-16")])
    rec = pit.reconstruct_membership(cur, _changes([]), start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2010-01-02") == {"AAA", "BRK-B"}
    flagged = rec.anomalies[rec.anomalies["kind"] == "date_added_without_row"]
    assert list(flagged["ticker"]) == ["BRK-B"]


def test_reconstruct_same_ticker_added_and_removed_on_one_day_stays_continuous():
    cur = _constituents([("FOXA", "Fox Corporation", "2019-03-19")])
    ch = _changes([("2019-03-19", "FOXA", "Fox Corporation", "FOXA", "21st Century Fox", "successor")])
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert len(rec.intervals) == 1  # the two adjacent intervals are coalesced
    matrix = pit.intervals_to_membership(rec.intervals, "2010-01-01", "2025-12-31")
    assert matrix["FOXA"].all()


def test_reconstruct_reused_ticker_keeps_the_eras_apart():
    cur = _constituents(
        [
            ("Q", "Qnity Electronics", "2025-11-03"),
            ("IQV", "IQVIA", "2017-08-29"),
            ("EW", "Edwards Lifesciences", "2011-03-31"),
        ]
    )
    ch = _changes(
        [
            ("2025-11-03", "Q", "Qnity Electronics", None, None, "Spin-off of Qnity."),
            ("2017-11-15", "IQV", "IQVIA", "Q", "QuintilesIMS",
             "QuintilesIMS changed its name to IQVIA and ticker symbol from Q to IQV."),
            ("2017-08-29", "Q", "QuintilesIMS", "WFM", "Whole Foods Market", "Amazon acquired WFM."),
            ("2011-03-31", "EW", "Edwards Lifesciences", "Q", "Qwest Communications",
             "Qwest was acquired by CenturyLink."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    assert _members(rec, "2011-03-30") == {"Q", "WFM"}  # Qwest
    assert _members(rec, "2011-03-31") == {"EW", "WFM"}
    assert _members(rec, "2017-08-28") == {"EW", "WFM"}
    assert _members(rec, "2017-08-29") == {"EW", "IQV"}  # QuintilesIMS, later IQVIA
    assert _members(rec, "2017-11-20") == {"EW", "IQV"}
    assert _members(rec, "2025-11-02") == {"EW", "IQV"}
    assert _members(rec, "2025-11-03") == {"EW", "IQV", "Q"}  # Qnity
    assert len(rec.anomalies) == 0


def test_reconstruct_reports_unresolved_additions_and_double_removals():
    cur = _constituents([("AAA", "Alpha", "1990-01-01")])
    ch = _changes(
        [
            ("2020-05-12", "AAA", "Alpha", "AGN", "Allergan", "Allergan was acquired."),
            ("2015-03-23", "ZZZ", "Zed", "AGN", "Allergan", "Allergan was acquired."),
        ]
    )
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of="2026-01-01")
    kinds = list(rec.anomalies["kind"])
    assert "added_not_active" in kinds  # ZZZ never was a member as far as the table can tell
    assert "removed_while_active" in kinds  # AGN removed twice without an addition in between
    # AGN stays a member through the gap: the count of slots stays right
    assert "AGN" in _members(rec, "2018-01-01")


def test_intervals_to_membership_inclusive_bounds_and_open_ends():
    intervals = pd.DataFrame(
        {
            "ticker": ["A", "A", "B"],
            "start": [_ts("2020-01-02"), _ts("2020-01-08"), pd.NaT],
            "end": [_ts("2020-01-04"), pd.NaT, _ts("2020-01-03")],
        }
    )
    m = pit.intervals_to_membership(intervals, "2020-01-01", "2020-01-10")
    assert list(m.columns) == ["A", "B"]
    assert m.index[0] == _ts("2020-01-01") and m.index[-1] == _ts("2020-01-10")
    assert m["A"].tolist() == [False, True, True, True, False, False, False, True, True, True]
    assert m["B"].tolist() == [True, True, True, False, False, False, False, False, False, False]
    assert m.dtypes.eq(bool).all()


def test_intervals_to_membership_drops_tickers_outside_the_window():
    intervals = pd.DataFrame(
        {"ticker": ["OLD"], "start": [_ts("2000-01-01")], "end": [_ts("2005-01-01")]}
    )
    m = pit.intervals_to_membership(intervals, "2010-01-01", "2010-01-31")
    assert m.shape[1] == 0


def test_summarize_member_counts_flags_the_days_out_of_band():
    idx = pd.date_range("2020-01-01", periods=10)
    m = pd.DataFrame(True, index=idx, columns=[f"T{i}" for i in range(500)])
    m.loc[idx[3:5], "T0"] = False  # 499 members on two days
    m.loc[idx[7], [f"T{i}" for i in range(20)]] = False  # 480 members on one day
    s = pit.summarize_member_counts(m, low=495, high=510)
    assert (s["min"], s["max"]) == (480, 500)
    assert s["n_days"] == 10 and s["n_out_of_band"] == 1
    assert s["share_in_band"] == pytest.approx(0.9)
    assert s["out_of_band_dates"] == ["2020-01-08"]


# --------------------------------------------------------------------------------------------
# Yahoo download
# --------------------------------------------------------------------------------------------


def _yahoo_frame(tickers, days=5, nan_for=(), start="2020-01-02"):
    """Yahoo-shaped frame; the numbers depend on the first letter: AAA -> close 10, volume 1000."""
    idx = pd.bdate_range(start, periods=days, name="Date")
    cols = {}
    for t in tickers:
        k = ord(t[0]) - ord("A")
        base = 10.0 + k
        for field, val in [
            ("Open", base), ("High", base), ("Low", base), ("Close", base),
            ("Adj Close", base * 0.9), ("Volume", 1000.0 * (k + 1)),
            ("Dividends", 0.0), ("Stock Splits", 0.0),
        ]:
            col = np.full(days, val)
            if t in nan_for:
                col = np.full(days, np.nan)
            cols[(t, field)] = col
    df = pd.DataFrame(cols, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Ticker", "Price"])
    return df


class FakeYahoo:
    """Stands in for ``yf.download``: records the calls, fails on request."""

    def __init__(
        self, missing=(), flaky=(), nan_for=(), splits=None, spurious_in_batch=(), dividends=None
    ):
        self.calls: list[list[str]] = []
        self.missing = set(missing)
        self.flaky = {t: 1 for t in flaky}  # fails once with a transient error, then works
        self.nan_for = set(nan_for)
        self.splits = splits or {}
        self.dividends = dividends or {}
        # listed, but reported "possibly delisted" whenever requested together with others
        self.spurious_in_batch = set(spurious_in_batch)

    def __call__(self, batch, start, end):
        self.calls.append(list(batch))
        errors = {}
        ok = []
        for t in batch:
            if t in self.missing or (t in self.spurious_in_batch and len(batch) > 1):
                errors[t] = "possibly delisted; no timezone found"
            elif self.flaky.get(t, 0) > 0:
                self.flaky[t] -= 1
                errors[t] = "YFRateLimitError('Too Many Requests')"
            else:
                ok.append(t)
        frame = _yahoo_frame(ok, nan_for=self.nan_for) if ok else pd.DataFrame()
        for t, (when, ratio) in self.splits.items():
            if t in ok:
                frame.loc[frame.index[when], (t, "Stock Splits")] = ratio
        for t, (when, amount) in self.dividends.items():
            if t in ok:
                frame.loc[frame.index[when], (t, "Dividends")] = amount
        return frame, errors


def test_download_yahoo_panel_batches_and_collects_failures():
    fake = FakeYahoo(missing={"DEAD"}, nan_for={"EMPTY"})
    panel = pit.download_yahoo_panel(
        ["AAA", "DEAD", "BBB", "EMPTY", "CCC"], "2020-01-01", "2020-02-01",
        batch_size=2, pause=0.0, downloader=fake,
    )
    assert [len(b) for b in fake.calls[:3]] == [2, 2, 1]
    assert list(panel.adj_close.columns) == ["AAA", "BBB", "CCC"]
    assert set(panel.failed) == {"DEAD", "EMPTY"}
    assert "delisted" in panel.failed["DEAD"]
    assert "no data" in panel.failed["EMPTY"]
    # adjusted close, unadjusted close and volume come from different Yahoo fields
    assert panel.adj_close["AAA"].iloc[0] == pytest.approx(9.0)
    assert panel.close["AAA"].iloc[0] == pytest.approx(10.0)
    assert panel.volume["BBB"].iloc[0] == pytest.approx(2000.0)
    assert panel.adj_close.index.tz is None and panel.adj_close.index.name == "date"
    # "possibly delisted" is confirmed by one more request for that ticker alone, not retried more
    assert sum("DEAD" in call for call in fake.calls) == 2 and ["DEAD"] in fake.calls
    assert ["EMPTY"] in fake.calls


def test_download_yahoo_panel_retries_transient_errors_one_by_one():
    fake = FakeYahoo(flaky={"BBB"})
    panel = pit.download_yahoo_panel(
        ["AAA", "BBB"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0, retries=2,
        downloader=fake,
    )
    assert set(panel.adj_close.columns) == {"AAA", "BBB"} and not panel.failed
    assert fake.calls[0] == ["AAA", "BBB"] and fake.calls[1] == ["BBB"]


def test_download_yahoo_panel_gives_up_after_the_retries():
    fake = FakeYahoo(flaky={"BBB"})
    fake.flaky["BBB"] = 99
    panel = pit.download_yahoo_panel(
        ["AAA", "BBB"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0, retries=2,
        downloader=fake,
    )
    assert set(panel.failed) == {"BBB"} and "Rate" in panel.failed["BBB"]
    assert sum("BBB" in call for call in fake.calls) == 3  # first try + 2 retries


def test_download_yahoo_panel_extracts_splits():
    fake = FakeYahoo(splits={"AAA": (2, 4.0)})
    panel = pit.download_yahoo_panel(
        ["AAA", "BBB"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0, downloader=fake,
    )
    assert len(panel.splits) == 1
    row = panel.splits.iloc[0]
    assert row["ticker"] == "AAA" and row["ratio"] == pytest.approx(4.0)
    assert row["date"] == pd.bdate_range("2020-01-02", periods=5)[2]


def test_download_yahoo_panel_extracts_dividends():
    fake = FakeYahoo(dividends={"AAA": (3, 0.25)})
    panel = pit.download_yahoo_panel(
        ["AAA", "BBB"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0, downloader=fake,
    )
    assert list(panel.dividends.columns) == ["ticker", "date", "amount"]
    assert len(panel.dividends) == 1
    row = panel.dividends.iloc[0]
    assert row["ticker"] == "AAA" and row["amount"] == pytest.approx(0.25)
    assert row["date"] == pd.bdate_range("2020-01-02", periods=5)[3]
    empty = pit.download_yahoo_panel(["BBB"], "2020-01-01", "2020-02-01", pause=0.0, downloader=FakeYahoo())
    assert len(empty.dividends) == 0 and list(empty.dividends.columns) == ["ticker", "date", "amount"]


def test_download_yahoo_panel_recovers_a_ticker_that_fails_only_inside_a_batch():
    # Yahoo reports "possibly delisted" for DELL in a batch, yet serves it alone
    fake = FakeYahoo(spurious_in_batch={"BBB"})
    panel = pit.download_yahoo_panel(
        ["AAA", "BBB", "CCC"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0, downloader=fake,
    )
    assert set(panel.adj_close.columns) == {"AAA", "BBB", "CCC"} and not panel.failed


def test_download_yahoo_panel_cache_reconfirms_failures_not_confirmed_before(tmp_path):
    (tmp_path / "AAA.failed.json").write_text(json.dumps({"reason": "possibly delisted"}))  # old
    (tmp_path / "DEAD.failed.json").write_text(
        json.dumps({"reason": "possibly delisted", "confirmed": True})
    )
    fake = FakeYahoo(missing={"DEAD"})
    panel = pit.download_yahoo_panel(
        ["AAA", "DEAD"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0, downloader=fake,
        cache_dir=tmp_path,
    )
    assert fake.calls == [["AAA"]]  # AAA asked again alone; DEAD was confirmed in an earlier run
    assert list(panel.adj_close.columns) == ["AAA"] and set(panel.failed) == {"DEAD"}
    assert not (tmp_path / "AAA.failed.json").exists() and (tmp_path / "AAA.parquet").exists()


def test_download_yahoo_panel_resumes_from_cache(tmp_path):
    first = FakeYahoo(missing={"DEAD"})
    pit.download_yahoo_panel(
        ["AAA", "BBB", "DEAD"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0,
        downloader=first, cache_dir=tmp_path,
    )
    second = FakeYahoo()
    panel = pit.download_yahoo_panel(
        ["AAA", "BBB", "DEAD", "CCC"], "2020-01-01", "2020-02-01", batch_size=5, pause=0.0,
        downloader=second, cache_dir=tmp_path,
    )
    assert second.calls == [["CCC"]]  # AAA, BBB from disk, DEAD known to be permanently gone
    assert set(panel.adj_close.columns) == {"AAA", "BBB", "CCC"}
    assert set(panel.failed) == {"DEAD"}


def test_split_yahoo_frame_accepts_flat_single_ticker_columns():
    flat = _yahoo_frame(["AAA"])["AAA"]  # what old yfinance returned for one ticker
    out = pit.split_yahoo_frame(flat, ["AAA"])
    assert set(out) == {"AAA"} and "Adj Close" in out["AAA"].columns


def test_compare_frames_reports_row_counts_and_largest_relative_difference():
    idx = pd.bdate_range("2020-01-01", periods=6, name="date")
    a = pd.DataFrame({"Adj Close": [10, 11, 12, 13, 14, 15.0], "Volume": [5, 5, 5, 5, 5, 5.0]},
                     index=idx)
    b = a.iloc[1:].copy()  # one row fewer at the start
    b.loc[idx[3], "Adj Close"] = 13.13  # 1% off on one day
    out = pit.compare_frames(a, b)
    assert out["rows_a"] == 6 and out["rows_b"] == 5 and out["rows_common"] == 5
    assert out["Adj Close"] == pytest.approx(0.01)  # |13.13 - 13| / 13, the largest of the days
    assert out["Volume"] == 0.0
    same = pit.compare_frames(a, a)
    assert same["Adj Close"] == 0.0 and same["rows_common"] == 6


# --------------------------------------------------------------------------------------------
# Panels for the selection tests
# --------------------------------------------------------------------------------------------


def _panel(levels: dict[str, float], n_days: int = 400, seed: int = 0):
    """Constant price 100 and volume ``levels[t]`` shares (dollar volume ~ 100 * level)."""
    idx = pd.bdate_range("2018-01-01", periods=n_days, name="date")
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.95, 1.05, size=(n_days, len(levels)))
    prices = pd.DataFrame(100.0, index=idx, columns=list(levels))
    close = prices.copy()
    vol = pd.DataFrame(noise * np.array(list(levels.values())), index=idx, columns=list(levels))
    cal = pd.date_range(idx[0], idx[-1], freq="D")
    membership = pd.DataFrame(True, index=cal, columns=list(levels))
    return prices, close, vol, membership


LEVELS = {"A": 9000.0, "B": 8000.0, "C": 7000.0, "D": 6000.0, "E": 5000.0, "F": 4000.0}


def test_select_universe_ranks_members_by_median_dollar_volume():
    prices, close, vol, mem = _panel(LEVELS)
    fd = prices.index[-1]
    out = pit.select_universe(fd, mem, prices, close, vol, n=3, lookback=300, liquidity_window=100)
    assert out == ["A", "B", "C"]
    assert pit.select_universe(fd, mem, prices, close, vol, n=10, lookback=300,
                               liquidity_window=100) == list("ABCDEF")


def test_select_universe_uses_unadjusted_close_for_the_turnover():
    prices, close, vol, mem = _panel({"A": 5000.0, "B": 5000.0})
    prices["B"] = 1.0  # adjusted prices say nothing about turnover
    close["B"] = 400.0  # B trades at 400 per share, A at 100
    out = pit.select_universe(prices.index[-1], mem, prices, close, vol, n=2, lookback=300,
                              liquidity_window=100)
    assert out == ["B", "A"]


def test_select_universe_only_members_of_the_formation_date():
    prices, close, vol, mem = _panel(LEVELS)
    fd = prices.index[300]
    mem.loc[mem.index <= fd, "A"] = False  # the most liquid name is not (yet) in the index
    mem.loc[mem.index > fd, "A"] = True
    mem.loc[mem.index > fd - pd.Timedelta(days=1), "B"] = False  # B leaves on the formation date
    out = pit.select_universe(fd, mem, prices, close, vol, n=3, lookback=250, liquidity_window=100)
    assert out == ["C", "D", "E"]


def test_select_universe_accepts_a_non_trading_formation_date():
    prices, close, vol, mem = _panel(LEVELS)
    friday = prices.index[204]  # 2018-01-01 is a Monday, so position 204 is a Friday
    assert friday.dayofweek == 4
    saturday, sunday = friday + pd.Timedelta(days=1), friday + pd.Timedelta(days=2)
    mem.loc[saturday:, "A"] = False  # A leaves the index on Saturday
    kwargs = dict(n=2, lookback=150, liquidity_window=50)
    assert pit.select_universe(friday, mem, prices, close, vol, **kwargs) == ["A", "B"]
    # the membership of the calendar day counts; prices come from the last trading day before it
    assert pit.select_universe(saturday, mem, prices, close, vol, **kwargs) == ["B", "C"]
    assert pit.select_universe(sunday, mem, prices, close, vol, **kwargs) == ["B", "C"]


def test_select_universe_requires_complete_history_in_the_training_window():
    prices, close, vol, mem = _panel(LEVELS)
    fd = prices.index[-1]
    # A: listed 20 days into the 300-day window -> 6.7% missing; B: 4% missing; C: 6% missing
    prices.loc[prices.index[-300]:prices.index[-281], "A"] = np.nan  # 20 / 300
    prices.loc[prices.index[-300]:prices.index[-289], "B"] = np.nan  # 12 / 300 = 4%
    prices.loc[prices.index[-300]:prices.index[-283], "C"] = np.nan  # 18 / 300 = 6%
    out = pit.select_universe(fd, mem, prices, close, vol, n=6, lookback=300, liquidity_window=100,
                              min_history_frac=0.95)
    assert out == ["B", "D", "E", "F"]
    relaxed = pit.select_universe(fd, mem, prices, close, vol, n=6, lookback=300,
                                  liquidity_window=100, min_history_frac=0.90)
    assert relaxed == list("ABCDEF")


def test_select_universe_min_history_frac_boundary_is_inclusive_and_nonpositive_is_invalid():
    prices, close, vol, mem = _panel({"A": 9000.0, "B": 8000.0})
    fd = prices.index[-1]
    prices.loc[prices.index[-300]:prices.index[-286], "A"] = np.nan  # exactly 15 / 300 = 5%
    prices.loc[prices.index[-300]:prices.index[-286], "B"] = 0.0  # zeros are invalid too
    prices.loc[prices.index[-285], "B"] = -1.0  # as are negatives
    both = pit.select_universe(fd, mem, prices, close, vol, n=2, lookback=300, liquidity_window=100,
                               min_history_frac=0.95)
    assert both == ["A"]


def test_select_universe_ipo_inside_the_window_is_excluded():
    prices, close, vol, mem = _panel(LEVELS)
    prices.loc[prices.index[:200], "A"] = np.nan  # A starts trading on day 200 of 400
    out = pit.select_universe(prices.index[-1], mem, prices, close, vol, n=3, lookback=250,
                              liquidity_window=100)
    assert "A" not in out and out == ["B", "C", "D"]


def test_select_universe_raises_when_the_panel_is_shorter_than_the_lookback():
    prices, close, vol, mem = _panel(LEVELS, n_days=100)
    with pytest.raises(ValueError, match="lookback"):
        pit.select_universe(prices.index[-1], mem, prices, close, vol, lookback=250)


def test_select_universe_skips_members_without_price_columns():
    prices, close, vol, mem = _panel(LEVELS)
    mem["GHOST"] = True  # member of the index, no data downloaded for it
    out = pit.select_universe(prices.index[-1], mem, prices, close, vol, n=10, lookback=300,
                              liquidity_window=100)
    assert "GHOST" not in out and len(out) == 6


def test_select_universe_has_no_look_ahead():
    prices, close, vol, mem = _panel(LEVELS, n_days=500, seed=3)
    fd = prices.index[350]
    kwargs = dict(n=3, lookback=300, liquidity_window=100)
    base = pit.select_universe(fd, mem, prices, close, vol, **kwargs)
    assert base == ["A", "B", "C"]

    # 1. destroy every observation strictly after the formation date, in all four inputs
    p2, c2, v2, m2 = prices.copy(), close.copy(), vol.copy(), mem.copy()
    future = p2.index > fd
    p2.loc[future] = np.nan
    c2.loc[future] = 1e9
    v2.loc[future, ["E", "F"]] = 1e12  # the two least liquid names become the most liquid
    v2.loc[future, ["A", "B", "C"]] = 0.0
    m2.loc[m2.index > fd] = ~mem.loc[mem.index > fd]  # flips membership of every future day
    assert pit.select_universe(fd, m2, p2, c2, v2, **kwargs) == base

    # 2. the answer equals the one computed on data that physically stops at the formation date
    truncated = pit.select_universe(
        fd, mem.loc[:fd], prices.loc[:fd], close.loc[:fd], vol.loc[:fd], **kwargs
    )
    assert truncated == base

    # 3. control: the test has teeth - changing the *past* does change the selection
    v3 = vol.copy()
    v3.loc[v3.index <= fd, "F"] = 1e9
    assert pit.select_universe(fd, mem, prices, close, v3, **kwargs)[0] == "F"


def test_select_universe_drops_the_less_liquid_share_class_by_cik():
    prices, close, vol, mem = _panel({"GOOGL": 9000.0, "GOOG": 8500.0, "MSFT": 8000.0, "XOM": 7000.0})
    companies = pd.DataFrame(
        {
            "ticker": ["GOOGL", "GOOG", "MSFT", "XOM"],
            "security": ["Alphabet Inc. (Class A)", "Alphabet Inc. (Class C)", "Microsoft", "Exxon"],
            "cik": ["0001652044", "0001652044", "0000789019", "0000034088"],
        }
    )
    out = pit.select_universe(prices.index[-1], mem, prices, close, vol, n=3, lookback=300,
                              liquidity_window=100, companies=companies)
    assert out == ["GOOGL", "MSFT", "XOM"]  # n distinct companies, not n tickers
    # the more liquid class is kept whichever it is
    vol2 = vol.copy()
    vol2["GOOG"] = vol["GOOG"] * 2
    out2 = pit.select_universe(prices.index[-1], mem, prices, close, vol2, n=2, lookback=300,
                               liquidity_window=100, companies=companies)
    assert out2 == ["GOOG", "MSFT"]


def test_select_universe_dedupes_by_normalised_name_when_cik_is_unknown():
    prices, close, vol, mem = _panel({"UA": 9000.0, "UAA": 8000.0, "NKE": 7000.0})
    companies = pd.DataFrame(
        {
            "ticker": ["UA", "UAA", "NKE"],
            "security": ["Under Armour (Class C)", "Under Armour (Class A)", "Nike, Inc."],
            "cik": [None, None, "0000320187"],
        }
    )
    out = pit.select_universe(prices.index[-1], mem, prices, close, vol, n=3, lookback=300,
                              liquidity_window=100, companies=companies)
    assert out == ["UA", "NKE"]


def test_select_universe_without_company_table_keeps_every_ticker():
    prices, close, vol, mem = _panel({"GOOGL": 9000.0, "GOOG": 8500.0})
    out = pit.select_universe(prices.index[-1], mem, prices, close, vol, n=2, lookback=300,
                              liquidity_window=100)
    assert out == ["GOOGL", "GOOG"]


def test_universe_candidates_reports_the_ranking_inputs():
    prices, close, vol, mem = _panel(LEVELS)
    table = pit.universe_candidates(prices.index[-1], mem, prices, close, vol, lookback=300,
                                    liquidity_window=100)
    assert list(table.index) == list("ABCDEF")
    assert {"valid_frac", "median_dollar_volume", "rank"} <= set(table.columns)
    assert table["valid_frac"].eq(1.0).all()
    assert table["median_dollar_volume"].is_monotonic_decreasing
    assert table.loc["A", "median_dollar_volume"] == pytest.approx(100 * 9000, rel=0.06)


# --------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------


def test_compute_coverage_counts_members_with_prices():
    prices, _, _, mem = _panel({"A": 1.0, "B": 1.0, "C": 1.0}, n_days=30)
    mem["D"] = True  # always a member, never downloaded
    mem.loc[:, "C"] = False
    mem.loc[mem.index >= prices.index[10], "C"] = True
    prices.loc[prices.index[:5], "A"] = np.nan
    cov = pit.compute_coverage(mem, prices, lookback=10, min_history_frac=0.95)
    assert list(cov.index) == list(prices.index)
    day0, day7, day12 = prices.index[0], prices.index[7], prices.index[12]
    assert cov.loc[day0, "n_members"] == 3 and cov.loc[day0, "n_with_price"] == 1  # B only
    assert cov.loc[day0, "coverage"] == pytest.approx(1 / 3)
    assert cov.loc[day7, "n_with_price"] == 2  # A is back, C not yet a member
    assert cov.loc[day12, "n_members"] == 4 and cov.loc[day12, "n_with_price"] == 3
    # eligible = members whose last-`lookback` price history is complete; membership is not
    # required for the history itself, so C (priced all along) counts, A (NaNs inside) does not
    assert np.isnan(cov.loc[day7, "n_eligible"])  # fewer than `lookback` rows so far
    assert cov.loc[day12, "n_eligible"] == 2  # B and C; D has no data at all


# --------------------------------------------------------------------------------------------
# Survivorship diagnostics
# --------------------------------------------------------------------------------------------


def test_removal_breakdown_counts_departures_by_reason_and_available_series():
    changes = _changes(
        [
            ("2015-01-01", "N1", "New1", "AAA", "Alpha", "Alpha was acquired by Big Co."),
            ("2016-01-01", "N2", "New2", "BBB", "Beta", "Market capitalization changes."),
            ("2017-01-01", "N3", "New3", "CCC", "Gamma", "Gamma filed for bankruptcy."),
            ("2018-01-01", "NEW", "Foo", "OLD", "Foo", "Foo changed its ticker symbol from OLD to NEW."),
            ("2009-01-01", "N0", "New0", "ZZZ", "Zed", "Zed was acquired."),  # before the window
        ]
    )
    renames = pd.DataFrame(
        [{"date": _ts("2018-01-01"), "old": "OLD", "new": "NEW", "rule": "reason",
          "kind": "ticker_change"}]
    )
    membership = pd.DataFrame(True, index=pd.date_range("2010-01-01", periods=3),
                              columns=["AAA", "BBB", "CCC", "NEW"])
    out = pit.removal_breakdown(changes, renames, membership, downloaded=["BBB", "NEW"],
                                start="2010-01-01", as_of=_ts("2026-01-01"))
    assert out["acquired / merged"] == {"n_removals": 1, "n_with_series": 0, "share_with_series": 0.0}
    assert out["size / eligibility"]["n_with_series"] == 1
    assert out["bankruptcy / failure"]["n_removals"] == 1
    assert sum(cell["n_removals"] for cell in out.values()) == 3  # OLD and ZZZ are not departures


def test_series_vs_membership_flags_series_that_start_late_or_end_early():
    days = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    membership = pd.DataFrame(True, index=days, columns=["OK", "LATE", "EARLY"])
    prices = pd.DataFrame(1.0, index=days, columns=["OK", "LATE", "EARLY"])
    prices.loc[:"2020-06-30", "LATE"] = np.nan  # series appears half a year after membership
    prices.loc["2020-03-01":, "EARLY"] = np.nan  # series ends long before membership does
    out = pit.series_vs_membership(membership, prices, min_gap_days=90)
    flagged = {row["ticker"]: row for row in out["tickers"]}
    assert set(flagged) == {"LATE", "EARLY"} and out["n_tickers"] == 2
    assert flagged["LATE"]["late_days"] == 182 and flagged["LATE"]["early_days"] == 0
    assert flagged["EARLY"]["early_days"] > 300 and flagged["EARLY"]["late_days"] == 0


def test_large_move_inventory_separates_moves_next_to_a_split_from_the_rest():
    idx = pd.bdate_range("2020-01-01", periods=120, name="date")
    prices = pd.DataFrame(10.0, index=idx, columns=["SPLIT", "REAL", "GAP", "QUIET"])
    volume = pd.DataFrame(1000.0, index=idx, columns=prices.columns)
    prices.loc[idx[50]:, "SPLIT"] = 3.0  # -70% on day 50, a split is listed next to it
    prices.loc[idx[60]:, "REAL"] = 18.0  # +80% on day 60, volume spike, no split
    volume.loc[idx[60], "REAL"] = 90_000.0
    prices.loc[idx[70]:idx[79], "GAP"] = np.nan  # halted for ten days, then +100%
    prices.loc[idx[80]:, "GAP"] = 20.0
    splits = pd.DataFrame({"ticker": ["SPLIT"], "date": [idx[51]], "ratio": [10 / 3]})
    inv = pit.large_move_inventory(prices, volume, splits, threshold=0.6)
    assert set(inv["ticker"]) == {"SPLIT", "REAL", "GAP"}
    by = inv.set_index("ticker")
    assert by.loc["SPLIT", "date"] == idx[50] and by.loc["SPLIT", "n_splits_nearby"] == 1
    assert by.loc["SPLIT", "split_ratio_nearby"] == pytest.approx(10 / 3)
    assert by.loc["REAL", "n_splits_nearby"] == 0 and by.loc["REAL", "return"] == pytest.approx(0.8)
    assert by.loc["REAL", "volume_ratio"] == pytest.approx(90.0)
    assert by.loc["GAP", "return"] == pytest.approx(1.0) and by.loc["GAP", "gap_days"] > 10
    assert list(inv["return"].abs()) == sorted(inv["return"].abs(), reverse=True)


def test_large_move_inventory_is_empty_without_large_moves():
    idx = pd.bdate_range("2020-01-01", periods=30, name="date")
    prices = pd.DataFrame(10.0, index=idx, columns=["A"])
    inv = pit.large_move_inventory(prices, prices * 100, pd.DataFrame(columns=["ticker", "date", "ratio"]))
    assert len(inv) == 0 and {"ticker", "date", "return"} <= set(inv.columns)


def _ghost_panel(n_days: int = 300):
    """REAL trades 1e9 $/day; GHOST is a shell (zero volume); LATE is a shell until day 150."""
    idx = pd.bdate_range("2020-01-01", periods=n_days, name="date")
    close = pd.DataFrame(10.0, index=idx, columns=["REAL", "GHOST", "LATE", "OTHER"])
    volume = pd.DataFrame(1e8, index=idx, columns=close.columns)
    volume["GHOST"] = 0.0
    volume.loc[idx[:150], "LATE"] = 0.0
    adj = close * 0.9
    member = pd.DataFrame(True, index=idx, columns=close.columns)
    member.loc[idx[200]:, "OTHER"] = False  # OTHER leaves the index; its later data are not checked
    return adj, close, volume, member


def test_mask_implausible_blanks_member_days_of_a_series_that_cannot_be_a_member():
    adj, close, volume, member = _ghost_panel()
    out = pit.mask_implausible_series(adj, close, volume, member, window=21, min_dollar_volume=1e6)
    idx = adj.index
    # a real member is untouched
    assert out.adj_close["REAL"].notna().all() and out.volume["REAL"].eq(1e8).all()
    # a shell that is a member all the time loses everything and is dropped as a column
    assert "GHOST" not in out.adj_close.columns and "GHOST" in out.dropped
    assert "not the index member" in out.dropped["GHOST"]
    # a shell until day 150: the trailing 21-day median turns real once 11 real days are in it
    late = out.adj_close["LATE"]
    assert late.iloc[:160].isna().all() and late.iloc[160:].notna().all()
    assert out.close_unadj["LATE"].isna().equals(late.isna())
    assert out.volume["LATE"].isna().equals(late.isna())
    # the report says what was removed
    row = out.report.set_index("ticker")
    assert row.loc["LATE", "n_member_days_masked"] == 160
    assert row.loc["GHOST", "n_member_days_masked"] == len(idx)
    assert "REAL" not in row.index
    assert row.loc["LATE", "first_masked"] == idx[0] and row.loc["LATE", "last_masked"] == idx[159]


def test_mask_implausible_leaves_days_outside_membership_alone():
    adj, close, volume, member = _ghost_panel()
    volume.loc[member.index[200]:, "OTHER"] = 0.0  # after it left the index the volume is nil
    out = pit.mask_implausible_series(adj, close, volume, member, window=21, min_dollar_volume=1e6)
    other = out.adj_close["OTHER"]
    assert other.iloc[:200].notna().all()  # member days with real volume stay
    assert other.iloc[200:].notna().all()  # non-member days are not judged, not touched


def test_mask_implausible_drops_series_that_never_meet_a_membership_spell():
    adj, close, volume, member = _ghost_panel()
    adj["REUSED"] = 5.0  # a liquid series of another company, but the index never held REUSED
    close["REUSED"], volume["REUSED"] = 5.0, 1e8
    member["REUSED"] = False
    adj["EARLY"] = 5.0  # member only before its data start
    close["EARLY"], volume["EARLY"] = 5.0, 1e8
    adj.loc[adj.index[:100], "EARLY"] = np.nan
    member["EARLY"] = False
    member.loc[member.index[:50], "EARLY"] = True
    out = pit.mask_implausible_series(adj, close, volume, member, window=21, min_dollar_volume=1e6)
    assert {"REUSED", "EARLY"} <= set(out.dropped) and "REAL" in out.adj_close.columns
    assert "outside the membership" in out.dropped["REUSED"]
    assert "REUSED" not in set(out.report["ticker"])  # nothing was masked there, it was dropped


def test_mask_implausible_uses_only_the_past():
    adj, close, volume, member = _ghost_panel()
    base = pit.mask_implausible_series(adj, close, volume, member, window=21, min_dollar_volume=1e6)
    cut = adj.index[220]
    volume2 = volume.copy()
    volume2.loc[volume2.index > cut, :] = 0.0  # everything after day 220 turns into a ghost
    again = pit.mask_implausible_series(adj, close, volume2, member, window=21, min_dollar_volume=1e6)
    before = adj.index <= cut
    pd.testing.assert_frame_equal(
        base.adj_close.loc[before, ["REAL", "LATE"]], again.adj_close.loc[before, ["REAL", "LATE"]]
    )


def test_mask_implausible_does_not_judge_cells_without_volume():
    adj, close, volume, member = _ghost_panel()
    volume.loc[:, "REAL"] = np.nan  # no volume data at all: cannot be judged, so not removed
    out = pit.mask_implausible_series(adj, close, volume, member, window=21, min_dollar_volume=1e6)
    assert out.adj_close["REAL"].notna().all()


def test_adjustment_gaps_lists_days_where_the_adjusted_return_exceeds_the_close_return():
    idx = pd.bdate_range("2020-01-01", periods=40, name="date")
    close = pd.DataFrame(100.0, index=idx, columns=["DIV", "SPECIAL", "PLAIN"])
    adj = close.copy()
    adj["DIV"] = close["DIV"].where(idx >= idx[10], 99.0)  # +1.0101% on day 10: an ordinary dividend
    adj["SPECIAL"] = close["SPECIAL"].where(idx >= idx[20], 70.0)  # +42.9% on day 20
    gaps = pit.adjustment_gaps(adj, close, threshold=0.10)
    assert list(gaps["ticker"]) == ["SPECIAL"] and gaps["date"].iloc[0] == idx[20]
    assert gaps["gap"].iloc[0] == pytest.approx(100 / 70 - 1)
    assert gaps["return_close"].iloc[0] == 0.0 and gaps["return_adjusted"].iloc[0] > 0.4
    assert len(pit.adjustment_gaps(adj, close, threshold=0.5)) == 0
    assert set(pit.adjustment_gaps(adj, close, threshold=0.005)["ticker"]) == {"DIV", "SPECIAL"}


# --------------------------------------------------------------------------------------------
# Yahoo's adjusted close on distribution days: classification and correction (R6, DHR 2016-07-05)
# --------------------------------------------------------------------------------------------

_DAYS, _EVENT = 60, 40


def _path(k: int, jump: float, start: float = 100.0, seed: int = 0, n: int = _DAYS) -> np.ndarray:
    """Close prices: a quiet random walk whose return on position ``k`` is exactly ``jump``."""
    ret = np.random.default_rng(seed).normal(0.0, 0.004, n)
    ret[0], ret[k] = 0.0, jump
    return start * np.cumprod(1.0 + ret)


def _yahoo_adjusted(close: pd.Series, dividends: dict[int, float]) -> pd.Series:
    """Adj Close as Yahoo builds it: a dividend D scales every earlier price by 1 - D / Close(t-1)."""
    factor = np.ones(len(close))
    for k, amount in dividends.items():
        factor[:k] *= 1.0 - amount / close.iloc[k - 1]
    return close * factor


def _event_panel(paths: dict, dividends: dict, splits: dict | None = None):
    """``(adj, close, dividends_frame, splits_frame)``; dividends/splits: ticker -> {position: value}."""
    idx = pd.bdate_range("2020-01-02", periods=_DAYS, name="date")
    close = pd.DataFrame(paths, index=idx)
    adj = pd.DataFrame(
        {t: _yahoo_adjusted(close[t], dividends.get(t, {})) for t in close}, index=idx
    )
    div = pd.DataFrame(
        [(t, idx[k], a) for t, d in dividends.items() for k, a in d.items()],
        columns=["ticker", "date", "amount"],
    )
    spl = pd.DataFrame(
        [(t, idx[k], r) for t, d in (splits or {}).items() for k, r in d.items()],
        columns=["ticker", "date", "ratio"],
    )
    return adj, close, div, spl


def _correct(paths, dividends, splits=None, **kwargs):
    adj, close, div, spl = _event_panel(paths, dividends, splits)
    return adj, close, pit.correct_adjustment_artifacts(adj, close, div, spl, **kwargs)


@pytest.mark.parametrize(
    "ratio, expected",
    [(2.0, True), (0.5, True), (1.5, True), (0.25, True), (4.0, True), (1.25, True),
     (1.319, False), (1.7412, False), (1.518, False), (9.295, False), (1.0027, False)],
)
def test_is_share_count_ratio_separates_simple_fractions_from_yahoo_value_ratios(ratio, expected):
    assert pit.is_share_count_ratio(ratio) is expected


def test_cash_dividend_day_gets_the_textbook_total_return_and_nothing_else_moves():
    k = _EVENT
    spec = _path(k, jump=-0.55, seed=1)  # price falls to 45% of the previous close ...
    paid = 0.5 * spec[k - 1]  # ... a special dividend of 50% of it was paid: total return -5%
    small = _path(20, 0.0, seed=2)
    adj, _close, out = _correct(
        {"SPEC": spec, "SMALL": small, "CTRL": _path(20, 0.0, seed=3)},
        {"SPEC": {k: paid}, "SMALL": {20: 0.01 * small[19]}},
    )
    idx = adj.index
    row = out.table.iloc[0]
    assert len(out.table) == 1 and (row["ticker"], row["date"]) == ("SPEC", idx[k])
    assert row["kind"] == "cash_dividend" and row["value_source"] == "yahoo_dividend"
    assert row["return_close"] == pytest.approx(-0.55) and row["dividend_over_close"] == pytest.approx(0.5)
    assert row["return_yahoo"] == pytest.approx(-0.10)  # 0.45 / (1 - 0.5): the dividend factor
    assert row["adj_formula_residual"] == pytest.approx(0.0, abs=1e-12)
    assert row["return_corrected"] == pytest.approx(-0.05)
    assert row["change"] == pytest.approx(0.05) and bool(row["plausible"])
    # the corrected panel
    new, old = out.prices_adj, adj
    assert new.loc[idx[k], "SPEC"] / new.loc[idx[k - 1], "SPEC"] - 1 == pytest.approx(-0.05)
    assert row["scale_before"] == pytest.approx(0.90 / 0.95)
    pd.testing.assert_frame_equal(new[["SMALL", "CTRL"]], old[["SMALL", "CTRL"]])  # untouched
    pd.testing.assert_series_equal(new.loc[idx[k]:, "SPEC"], old.loc[idx[k]:, "SPEC"])  # end anchored
    ratio = new.loc[: idx[k - 1], "SPEC"] / old.loc[: idx[k - 1], "SPEC"]
    assert ratio.to_numpy() == pytest.approx(row["scale_before"])  # earlier history rescaled only
    other_days = idx.difference([idx[k]])
    pd.testing.assert_series_equal(
        new["SPEC"].pct_change(fill_method=None).loc[other_days],
        old["SPEC"].pct_change(fill_method=None).loc[other_days],
    )


def test_dividend_with_a_real_split_on_the_same_day_is_per_pre_split_share():
    k = _EVENT
    rev = _path(k, jump=16.52 / 23.68 - 1, start=23.68, seed=4)  # 1:2 reverse split, Close adjusted
    rev = rev * 23.68 / rev[k - 1]
    _adj, _close, out = _correct({"REV": rev}, {"REV": {k: 4.06}}, {"REV": {k: 0.5}})
    row = out.table.iloc[0]
    assert row["kind"] == "dividend_with_real_split" and row["split_ratio"] == 0.5
    assert row["return_yahoo"] == pytest.approx(16.52 / (23.68 - 4.06) - 1)  # Yahoo: -15.8 %
    assert row["return_corrected"] == pytest.approx((16.52 + 4.06 / 0.5) / 23.68 - 1)  # +4.05 %
    assert bool(row["plausible"])


def _spin_off_inputs(later_split: bool):
    """Value 24 handed out per share worth 100 (cum): Yahoo registers the ratio 100/76 AND the dividend 24.

    With ``later_split`` a real 2:1 split follows: every Close is halved, the dividend stays in the
    units of its day (24), so only the ratio x later split factor reconciles the two.
    """
    k = _EVENT
    unit = 0.5 if later_split else 1.0
    path = _path(k, jump=78.0 / 76.0 - 1.0, start=76.0 * unit, seed=5)
    path = path * (76.0 * unit) / path[k - 1]
    splits = {"DUP": {k: 100.0 / 76.0, **({50: 2.0} if later_split else {})}}
    return k, path, splits


@pytest.mark.parametrize("later_split", [False, True])
def test_spin_off_registered_as_ratio_and_dividend_is_counted_once(later_split):
    k, path, splits = _spin_off_inputs(later_split)
    dividend = 24.0
    _adj, close, out = _correct({"DUP": path}, {"DUP": {k: dividend}}, splits)
    row = out.table.iloc[0]
    close_prev = close["DUP"].iloc[k - 1]
    assert row["kind"] == "pseudo_split_duplicate_value"
    assert row["value_source"] == "yahoo_split_ratio"
    # Adj Close credits the dividend on top of the ratio that already took it out of Close
    assert row["return_yahoo"] == pytest.approx(close["DUP"].iloc[k] / (close_prev - dividend) - 1)
    assert row["return_yahoo"] > 0.3  # the artifact (DHR: +61 %)
    s = 100.0 / 76.0
    assert row["return_close"] == pytest.approx(78.0 / 76.0 - 1)
    assert row["return_corrected"] == pytest.approx(row["return_close"] / s)
    assert row["return_corrected"] == pytest.approx(0.02, abs=1e-9)  # (78 + 24) / 100 - 1
    assert row["later_split_factor"] == (2.0 if later_split else 1.0)


def test_spin_off_with_a_separate_cash_dividend_keeps_the_cash():
    k = _EVENT
    s = 100.0 / 58.0  # in-kind value 42 of a price of 100 is in the ratio; 10 cash on top
    path = _path(k, jump=51.0 / 58.0 - 1.0, start=58.0, seed=6)
    path = path * 58.0 / path[k - 1]
    _adj, _close, out = _correct({"MIX": path}, {"MIX": {k: 10.0}}, {"MIX": {k: s}})
    row = out.table.iloc[0]
    assert row["kind"] == "pseudo_split_plus_cash" and row["value_source"].endswith("yahoo_dividend")
    assert row["return_yahoo"] == pytest.approx(51.0 / 48.0 - 1)  # +6.25 %: Yahoo's cash factor
    assert row["return_corrected"] == pytest.approx((51 + 42 + 10) / 100 - 1)  # +3 %


def _child_inputs(parent_prev_child: float, child_event: float):
    k = _EVENT
    parent = _path(k, jump=27.5 / 57.0 - 1.0, start=57.0, seed=7)
    parent = parent * 57.0 / parent[k - 1]
    idx = pd.bdate_range("2020-01-02", periods=_DAYS, name="date")
    child = pd.Series(np.nan, index=idx)
    child.iloc[k - 10 : k] = parent_prev_child  # when-issued quotes before the distribution
    child.iloc[k] = child_event
    child.iloc[k + 1 :] = child_event
    return k, parent, child.to_frame("CHD")


def test_in_kind_distribution_is_revalued_at_the_child_close_when_the_ratio_is_confirmed():
    k, parent, children = _child_inputs(parent_prev_child=30.0, child_event=27.0)
    # D = 15.0 per pre-split share = 0.5 x the child's last when-issued close: a simple ratio
    _adj, _close, plain = _correct({"PAR": parent}, {"PAR": {k: 15.0}}, {"PAR": {k: 0.5}})
    _, _, out = _correct(
        {"PAR": parent}, {"PAR": {k: 15.0}}, {"PAR": {k: 0.5}}, children_close=children
    )
    row, base = out.table.iloc[0], plain.table.iloc[0]
    assert base["return_corrected"] == pytest.approx((27.5 + 30.0) / 57.0 - 1)  # Yahoo's valuation
    assert base["child_ticker"] is None or pd.isna(base["child_ticker"])
    assert row["child_ticker"] == "CHD" and bool(row["child_exact"])
    assert row["child_n"] == pytest.approx(0.5)
    assert row["value_source"] == "child_close"
    assert row["return_corrected"] == pytest.approx((27.5 + 27.0) / 57.0 - 1)  # child at its close


def test_the_amount_as_registered_decides_the_ratio_before_the_later_split_does():
    k, parent, children = _child_inputs(parent_prev_child=30.0, child_event=27.0)
    # a later 1:4 reverse split of the parent: 15.0 * 0.25 = 3.75 = 1/8 of 30.0 fits as well
    _, _, out = _correct(
        {"PAR": parent}, {"PAR": {k: 15.0}}, {"PAR": {k: 0.5, 50: 0.25}}, children_close=children
    )
    row = out.table.iloc[0]
    assert row["later_split_factor"] == 0.25 and row["child_n"] == pytest.approx(0.5)


def test_a_child_whose_price_does_not_give_a_simple_ratio_is_not_used():
    k, parent, children = _child_inputs(parent_prev_child=31.3, child_event=27.0)  # 15 / 31.3 = 0.479
    _, _, out = _correct(
        {"PAR": parent}, {"PAR": {k: 15.0}}, {"PAR": {k: 0.5}}, children_close=children
    )
    row = out.table.iloc[0]
    assert not bool(row["child_exact"]) and row["value_source"] == "yahoo_dividend"
    assert row["return_corrected"] == pytest.approx((27.5 + 30.0) / 57.0 - 1)


def test_a_company_that_starts_on_the_event_day_is_reported_but_not_used():
    k, path, splits = _spin_off_inputs(later_split=False)
    idx = pd.bdate_range("2020-01-02", periods=_DAYS, name="date")
    child = pd.Series(np.nan, index=idx)
    child.iloc[k:] = 48.0  # no quote before the day: the ratio 24 / 48 cannot be confirmed
    _, _, plain = _correct({"DUP": path}, {"DUP": {k: 24.0}}, splits)
    _, _, out = _correct(
        {"DUP": path}, {"DUP": {k: 24.0}}, splits, children_close=child.to_frame("FTV")
    )
    row = out.table.iloc[0]
    assert row["child_ticker"] == "FTV" and not bool(row["child_exact"])
    assert row["child_n"] == pytest.approx(0.5)
    assert row["value_source"] == "yahoo_split_ratio"
    assert row["return_corrected"] == plain.table.iloc[0]["return_corrected"]


def test_volume_ratio_is_the_day_over_the_median_of_the_previous_sixty_days():
    k = _EVENT
    spec = _path(k, jump=-0.55, seed=1)
    adj, close, div, spl = _event_panel({"SPEC": spec}, {"SPEC": {k: 0.5 * spec[k - 1]}})
    volume = pd.DataFrame(1000.0, index=adj.index, columns=["SPEC"])
    volume.iloc[k] = 2900.0
    out = pit.correct_adjustment_artifacts(adj, close, div, spl, volume=volume)
    assert out.table["volume_ratio"].iloc[0] == pytest.approx(2.9)
    assert np.isnan(pit.correct_adjustment_artifacts(adj, close, div, spl).table["volume_ratio"].iloc[0])


def test_a_gap_without_a_registered_action_is_kept_and_listed():
    idx = pd.bdate_range("2020-01-02", periods=_DAYS, name="date")
    close = pd.DataFrame({"X": _path(20, 0.0, seed=8)}, index=idx)
    adj = close.copy()
    adj.loc[idx[:30], "X"] *= 0.7  # Adj Close jumps 43 % on day 30 with no dividend or split
    out = pit.correct_adjustment_artifacts(
        adj, close, pd.DataFrame(columns=["ticker", "date", "amount"]),
        pd.DataFrame(columns=["ticker", "date", "ratio"]),
    )
    row = out.table.iloc[0]
    assert row["kind"] == "unexplained" and row["value_source"] == "none"
    assert row["return_corrected"] == pytest.approx(row["return_yahoo"]) and row["scale_before"] == 1.0
    pd.testing.assert_frame_equal(out.prices_adj, adj)


def test_an_implausibly_large_corrected_return_is_flagged_not_hidden():
    k = _EVENT
    big = _path(k, jump=-0.30, seed=9)  # price -30 %, dividend 15 %: total return -15 %
    _adj, _close, out = _correct({"BIG": big}, {"BIG": {k: 0.15 * big[k - 1]}})
    row = out.table.iloc[0]
    assert len(out.table) == 1
    assert row["return_corrected"] == pytest.approx(-0.15) and not bool(row["plausible"])


def test_corrections_compose_for_two_events_of_one_ticker_and_can_be_undone():
    a, b = 25, _EVENT
    path = _path(a, jump=-0.45, seed=10)
    path[b:] = path[b:] / path[b] * path[b - 1] * 0.50
    divs = {"TWO": {a: 0.40 * path[a - 1], b: 0.45 * path[b - 1]}}
    adj, _close, out = _correct({"TWO": path, "CTRL": _path(20, 0.0, seed=3)}, divs)
    assert list(out.table["date"]) == [adj.index[a], adj.index[b]]
    ret = out.prices_adj["TWO"].pct_change(fill_method=None)
    for r in out.table.itertuples():
        assert ret.loc[r.date] == pytest.approx(r.return_corrected)
    early = out.prices_adj.loc[: adj.index[a - 1], "TWO"] / adj.loc[: adj.index[a - 1], "TWO"]
    assert early.to_numpy() == pytest.approx(out.table["scale_before"].prod())
    back = pit.undo_return_corrections(out.prices_adj, out.table)
    pd.testing.assert_frame_equal(back, adj, check_exact=False, rtol=1e-12)
    again = pit.apply_return_corrections(adj, out.table)
    pd.testing.assert_frame_equal(again, out.prices_adj, check_exact=False, rtol=1e-12)


def test_correction_of_a_day_reads_nothing_dated_after_it():
    k = _EVENT
    spec = _path(k, jump=-0.55, seed=11)
    adj, close, div, spl = _event_panel({"SPEC": spec}, {"SPEC": {k: 0.5 * spec[k - 1]}})
    full = pit.correct_adjustment_artifacts(adj, close, div, spl)
    cut = adj.index[k]
    short = pit.correct_adjustment_artifacts(
        adj.loc[:cut], close.loc[:cut], div[div["date"] <= cut], spl[spl["date"] <= cut]
    )
    assert short.table["return_corrected"].iloc[0] == full.table["return_corrected"].iloc[0]
    # returns before the event are the same numbers before and after the correction
    pd.testing.assert_series_equal(
        full.prices_adj["SPEC"].pct_change(fill_method=None).loc[: adj.index[k - 1]],
        adj["SPEC"].pct_change(fill_method=None).loc[: adj.index[k - 1]],
    )


def test_universe_selection_does_not_depend_on_the_correction():
    prices, close, vol, mem = _panel(LEVELS, n_days=500, seed=3)
    k = 300
    day = prices.index[k]
    close.loc[day:, "C"] = 55.0  # the close falls from 100 to 55 while a dividend of 40 is paid ...
    prices.loc[:prices.index[k - 1], "C"] = 60.0  # ... and Yahoo's Adj Close is 60 before, 55 after
    prices.loc[day:, "C"] = 55.0
    div = pd.DataFrame([("C", day, 40.0)], columns=["ticker", "date", "amount"])
    spl = pd.DataFrame(columns=["ticker", "date", "ratio"])
    fixed = pit.correct_adjustment_artifacts(prices, close, div, spl).prices_adj
    assert not fixed.equals(prices)  # the correction did something (-8.3 % becomes -5 %)
    assert ((fixed > 0) == (prices > 0)).all().all() and fixed.isna().equals(prices.isna())
    kwargs = dict(n=4, lookback=250, liquidity_window=100)
    for date in (prices.index[k - 1], day, prices.index[450]):  # before, on and after the event
        assert pit.select_universe(date, mem, fixed, close, vol, **kwargs) == pit.select_universe(
            date, mem, prices, close, vol, **kwargs
        )


def test_pseudo_split_inventory_lists_value_ratios_with_the_return_they_distort():
    idx = pd.bdate_range("2020-01-02", periods=30, name="date")
    close = pd.DataFrame(10.0, index=idx, columns=["BIG", "REAL", "FLAGGED"])
    close.loc[idx[10]:, "BIG"] = 11.6  # +16 % on a ratio of 9.295: 1.7 % without the ratio
    splits = pd.DataFrame(
        [("BIG", idx[10], 9.295), ("REAL", idx[12], 2.0), ("FLAGGED", idx[14], 1.5180)],
        columns=["ticker", "date", "ratio"],
    )
    inv = pit.pseudo_split_inventory(close, close, splits, corrected={("FLAGGED", idx[14])})
    assert set(inv["ticker"]) == {"BIG", "FLAGGED"}  # 2:1 is a real split
    big = inv.set_index("ticker").loc["BIG"]
    assert big["return_close"] == pytest.approx(0.16)
    assert big["return_if_value_ratio"] == pytest.approx(0.16 / 9.295)
    assert big["distortion"] == pytest.approx(0.16 - 0.16 / 9.295) and not bool(big["corrected"])
    assert bool(inv.set_index("ticker").loc["FLAGGED", "corrected"])


def test_uncorrected_dividend_error_bounds_what_the_threshold_leaves_alone():
    k = 20
    path = _path(k, jump=-0.02, seed=12)
    adj, close, div, spl = _event_panel({"A": path}, {"A": {k: 0.03 * path[k - 1]}})
    err = pit.uncorrected_dividend_error(adj, close, div, spl, flagged=set())
    assert err["n_dividend_days"] == 1
    tr = (path[k] + 0.03 * path[k - 1]) / path[k - 1] - 1
    yahoo = path[k] / (0.97 * path[k - 1]) - 1
    assert err["max_abs_error"] == pytest.approx(abs(yahoo - tr))
    assert pit.uncorrected_dividend_error(adj, close, div, spl, flagged={("A", adj.index[k])})[
        "n_dividend_days"
    ] == 0


# --------------------------------------------------------------------------------------------
# Corporate-action evidence (BLDR-style check)
# --------------------------------------------------------------------------------------------


def _history(n=90, jump_at=60, factor=1.68, vol_mult=1.0, split_ratio=0.0, drift=0.0):
    idx = pd.bdate_range("2015-01-01", periods=n, name="Date")
    close = np.full(n, 6.8)
    close[jump_at:] *= factor
    close = close * (1 + drift * np.arange(n))
    volume = np.full(n, 200_000.0)
    volume[jump_at] *= vol_mult
    splits = np.zeros(n)
    if split_ratio:
        splits[jump_at] = split_ratio
    return pd.DataFrame(
        {"Close": close, "Adj Close": close, "Volume": volume, "Stock Splits": splits,
         "Dividends": 0.0},
        index=idx,
    )


def test_jump_evidence_real_move_has_volume_spike_no_split_and_persists():
    h = _history(vol_mult=120.0)
    ev = pit.jump_evidence(h, h.index[60])
    assert ev["return_unadjusted"] == pytest.approx(0.68)
    assert ev["splits_in_window"] == []
    assert ev["volume_ratio"] == pytest.approx(120.0)
    assert ev["persistence"] == pytest.approx(1.0)
    assert ev["verdict"] == "real market move"


def test_jump_evidence_split_is_recognised():
    # a 3-for-5 reverse split: price x1.667, volume / 1.667 on the day in raw (as-traded) terms
    h = _history(factor=5 / 3, split_ratio=0.6, vol_mult=0.6)
    ev = pit.jump_evidence(h, h.index[60])
    assert ev["splits_in_window"] == [(h.index[60], pytest.approx(0.6))]
    assert ev["verdict"] == "corporate action (split)"


def test_jump_evidence_without_split_or_volume_spike_is_inconclusive():
    h = _history(vol_mult=1.0)  # no volume spike, no split: neither story fits
    ev = pit.jump_evidence(h, h.index[60])
    assert ev["verdict"] == "inconclusive"


# --------------------------------------------------------------------------------------------
# Dataset round-trip
# --------------------------------------------------------------------------------------------


def test_save_and_load_dataset_roundtrip(tmp_path):
    prices, close, vol, mem = _panel({"A": 9000.0, "B": 8000.0}, n_days=30)
    companies = pd.DataFrame(
        {"ticker": ["A", "B"], "security": ["Alpha", "Beta"], "cik": [None, "0000000002"],
         "sector": ["X", "Y"]}
    )
    cov = pit.compute_coverage(mem, prices, lookback=10)
    pit.save_dataset(
        tmp_path, membership=mem, prices_adj=prices, close_unadj=close, volume=vol, coverage=cov,
        companies=companies, failed={"Z": "possibly delisted"},
    )
    for name in ["membership", "prices_adj", "close_unadj", "volume", "coverage", "companies"]:
        assert (tmp_path / f"{name}.parquet").exists()
    assert (tmp_path / "failed_tickers.json").exists()
    data = pit.load_dataset(tmp_path)
    pd.testing.assert_frame_equal(data.prices_adj, prices, check_freq=False)
    pd.testing.assert_frame_equal(data.membership, mem, check_freq=False)
    assert data.failed == {"Z": "possibly delisted"}
    assert "company_id" in data.companies.columns
    out = data.select(prices.index[-1], n=2, lookback=20, liquidity_window=10)
    assert out == ["A", "B"]


def test_loaded_dataset_exposes_the_corrections_and_the_uncorrected_panel(tmp_path):
    k = _EVENT
    spec = _path(k, jump=-0.55, seed=1)
    adj, close, div, spl = _event_panel({"SPEC": spec}, {"SPEC": {k: 0.5 * spec[k - 1]}})
    out = pit.correct_adjustment_artifacts(adj, close, div, spl)
    mem = pd.DataFrame(True, index=pd.date_range(adj.index[0], adj.index[-1]), columns=["SPEC"])
    companies = pd.DataFrame({"ticker": ["SPEC"], "security": ["Spec"], "cik": [None], "sector": ["X"]})
    pit.save_dataset(
        tmp_path, membership=mem, prices_adj=out.prices_adj, close_unadj=close,
        volume=close * 1e5, coverage=pit.compute_coverage(mem, adj, lookback=10),
        companies=companies, failed={}, extra={"adjustment_corrections": out.table},
    )
    data = pit.load_dataset(tmp_path)
    pd.testing.assert_frame_equal(data.prices_adj, out.prices_adj, check_freq=False)
    assert list(data.corrections["ticker"]) == ["SPEC"]
    pd.testing.assert_frame_equal(
        data.prices_adj_yahoo(), adj, check_freq=False, check_exact=False, rtol=1e-12
    )
    plain = tmp_path / "plain"
    pit.save_dataset(
        plain, membership=mem, prices_adj=adj, close_unadj=close, volume=close * 1e5,
        coverage=pit.compute_coverage(mem, adj, lookback=10), companies=companies, failed={},
    )
    old = pit.load_dataset(plain)  # a dataset built before the corrections existed
    assert old.corrections is None
    pd.testing.assert_frame_equal(old.prices_adj_yahoo(), adj, check_freq=False)


# --------------------------------------------------------------------------------------------
# The build script (imported from its path; its logic is exercised on a tiny dataset)
# --------------------------------------------------------------------------------------------


def _load_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "data" / "build_sp500_pit_dataset.py"
    spec = importlib.util.spec_from_file_location("build_sp500_pit_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_dataset(tmp_path):
    prices, close, vol, mem = _panel(LEVELS, n_days=400)
    mem["Z"] = False  # in the table of tickers, never a member
    companies = pd.DataFrame(
        {
            "ticker": [*LEVELS, "Z"],
            "security": [f"Company {t}" for t in [*LEVELS, "Z"]],
            "cik": [f"{k:010d}" for k in range(len(LEVELS) + 1)],
            "sector": ["Tech"] * 3 + ["Energy"] * 4,
            "in_current_list": [True] * len(LEVELS) + [False],
        }
    )
    cov = pit.compute_coverage(mem, prices, lookback=300)
    pit.save_dataset(
        tmp_path, membership=mem, prices_adj=prices, close_unadj=close, volume=vol, coverage=cov,
        companies=companies, failed={},
    )
    return prices


class EventYahoo:
    """Stands in for ``yf.download``: 60 business days per ticker, one special dividend on one."""

    def __init__(self, event_ticker="GOOG", missing=()):
        self.event_ticker, self.missing, self.calls = event_ticker, set(missing), []

    def frame(self, ticker):
        k, event = _EVENT, ticker == self.event_ticker
        idx = pd.bdate_range("2020-01-02", periods=_DAYS, name="Date")
        close = pd.Series(
            _path(k, jump=-0.55 if event else 0.0, start=50.0, seed=sum(map(ord, ticker))), index=idx
        )
        paid = {k: 0.5 * close.iloc[k - 1]} if event else {}
        out = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close,
             "Adj Close": _yahoo_adjusted(close, paid), "Volume": 2e6, "Dividends": 0.0,
             "Stock Splits": 0.0},
            index=idx,
        )
        for pos, amount in paid.items():
            out.iloc[pos, out.columns.get_loc("Dividends")] = amount
        return out

    def __call__(self, batch, start, end):
        self.calls.append(list(batch))
        errors = {t: "possibly delisted; no timezone found" for t in batch if t in self.missing}
        frames = {t: self.frame(t) for t in batch if t not in self.missing}
        return (pd.concat(frames, axis=1) if frames else pd.DataFrame()), errors


_BUILD = ["--start", "2020-01-01", "--end", "2020-03-31", "--lookback", "20", "--pause", "0",
          "--retries", "1", "--verify-sample", "0", "--skip-adj-check"]


def _save_snapshots(raw):
    when = datetime(2020, 3, 31, tzinfo=timezone.utc)
    pit.save_html_snapshot(CONSTITUENTS_HTML, raw, pit.WIKI_SP500_URL, "constituents", when)
    pit.save_html_snapshot(CHANGES_HTML, raw, pit.WIKI_CHANGES_URL, "changes", when)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_script_build_corrects_the_artifact_day_and_hashes_every_cache_file(tmp_path, monkeypatch):
    from cif.data.provenance import verify_provenance

    script = _load_script()
    raw, out = tmp_path / "raw", tmp_path / "out"
    _save_snapshots(raw)
    monkeypatch.setattr(pit, "yahoo_downloader", EventYahoo(missing={"GONE"}))
    args = ["build", "--raw-dir", str(raw), "--out-dir", str(out), *_BUILD]
    assert script.main([*args, "--evidence-tickers", "GONE"]) == 0

    prov = json.loads((out / "provenance.json").read_text())
    cache = raw / "yahoo_2020-01-01_2020-03-31"
    recorded = prov["files"]["raw"]  # same layout as moex_v2: path relative to raw_dir
    assert sorted(recorded) == sorted(f"{cache.name}/{p.name}" for p in cache.iterdir())
    assert f"{cache.name}/GONE.failed.json" in recorded and f"{cache.name}/GOOG.parquet" in recorded
    for rel, entry in recorded.items():
        data = (raw / rel).read_bytes()
        assert entry == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    report = verify_provenance(out / "provenance.json")
    assert report["ok"] and report["n_files"] == len(prov["outputs"]) + 2 + len(recorded)

    table = pd.read_parquet(out / "adjustment_corrections.parquet")
    assert list(table["ticker"]) == ["GOOG"] and table["kind"].iloc[0] == "cash_dividend"
    fixed = pd.read_parquet(out / "prices_adj.parquet")
    assert fixed["GOOG"].pct_change().loc[table["date"].iloc[0]] == pytest.approx(-0.05)
    gaps = pd.read_parquet(out / "adjustment_gaps.parquet")  # the inventory is Yahoo's, unchanged
    assert gaps["return_adjusted"].iloc[0] == pytest.approx(-0.10)
    assert pd.read_parquet(out / "dividends.parquet")["amount"].iloc[0] > 0
    section = prov["adjustment_corrections"]
    assert section["applied"] is True and section["version"] == pit.ADJUSTMENT_CORRECTION_VERSION
    assert section["n_events"] == 1 and section["events"][0]["ticker"] == "GOOG"
    assert prov["yahoo"]["evidence_tickers"]["failed"] == ["GONE"]


def test_script_rebuild_from_the_cache_changes_prices_adj_and_nothing_else(tmp_path, monkeypatch):
    script = _load_script()
    raw = tmp_path / "raw"
    _save_snapshots(raw)
    fake = EventYahoo(missing={"GONE"})
    monkeypatch.setattr(pit, "yahoo_downloader", fake)
    corrected, plain = tmp_path / "corrected", tmp_path / "plain"
    base = ["build", "--raw-dir", str(raw), *_BUILD]
    assert script.main([*base, "--out-dir", str(corrected), "--evidence-tickers", "GONE"]) == 0
    fake.calls.clear()
    assert script.main([*base, "--out-dir", str(plain), "--no-adjustment-corrections"]) == 0
    assert fake.calls == []  # everything came from the cache

    skip = {"prices_adj.parquet", "provenance.json", "adjustment_corrections.parquet"}
    names = {p.name for p in corrected.iterdir() if p.suffix in (".parquet", ".json")} - skip
    assert {"membership.parquet", "close_unadj.parquet", "volume.parquet", "splits.parquet",
            "large_moves.parquet", "adjustment_gaps.parquet", "failed_tickers.json"} <= names
    for name in sorted(names):
        assert _sha(corrected / name) == _sha(plain / name), name
    before = pd.read_parquet(plain / "prices_adj.parquet")
    after = pd.read_parquet(corrected / "prices_adj.parquet")
    pd.testing.assert_frame_equal(before.drop(columns="GOOG"), after.drop(columns="GOOG"))
    assert not before["GOOG"].equals(after["GOOG"])
    assert not (plain / "adjustment_corrections.parquet").exists()
    assert json.loads((plain / "provenance.json").read_text())["adjustment_corrections"]["applied"] is False
    undone = pit.load_dataset(corrected).prices_adj_yahoo()  # Yahoo's panel back from the files
    pd.testing.assert_frame_equal(undone, before, check_freq=False, check_exact=False, rtol=1e-12)


def test_script_rebuild_keeps_the_network_checks_of_the_earlier_build(tmp_path, monkeypatch):
    script = _load_script()
    raw, out = tmp_path / "raw", tmp_path / "out"
    _save_snapshots(raw)
    monkeypatch.setattr(pit, "yahoo_downloader", EventYahoo())
    args = ["build", "--raw-dir", str(raw), "--out-dir", str(out), *_BUILD, "--evidence-tickers"]
    assert script.main(args) == 0
    path = out / "provenance.json"
    prov = json.loads(path.read_text())
    assert prov["yahoo"]["adj_close_vs_auto_adjust"] is None  # skipped and nothing earlier to keep
    # as if the first build had made the two checks that need the network
    prov["yahoo"]["adj_close_vs_auto_adjust"] = {"checked": {"AAPL": {"n_days": 5}}, "max_rel_diff": 1e-6}
    prov["yahoo"]["batch_vs_single_download"] = {
        "seed": 0, "max_rel_diff": 2e-6, "n_row_mismatch": 0, "tickers": {}
    }
    path.write_text(json.dumps(prov))
    for _ in range(2):  # the second rebuild must not replace the date of the check by its own
        assert script.main(args) == 0
        kept = json.loads(path.read_text())["yahoo"]
        assert kept["adj_close_vs_auto_adjust"]["max_rel_diff"] == 1e-6
        assert kept["adj_close_vs_auto_adjust"]["carried_over_from_build"] == prov["built_at"]
        assert kept["batch_vs_single_download"]["carried_over_from_build"] == prov["built_at"]


class SpinOffYahoo(EventYahoo):
    """Yahoo with one spin-off day (business day ``_EVENT``) and no dividend event.

    GOOG closes at 100 before the day; a holder then owns 0.5 new GOOG shares (2 x 92 = 184 each)
    and 0.25 CHILD shares (40 each): 102, +2 %. Yahoo books ``RATIO``, no share-count ratio, and
    divides every earlier close by it.
    """

    RATIO = 0.43
    TRUTH = (0.5 * 184.0 + 0.25 * 40.0) / 100.0 - 1.0

    def __init__(self, **kw):
        super().__init__(event_ticker="NO-EVENT", **kw)

    def frame(self, ticker):
        if ticker not in ("GOOG", "CHILD"):
            return super().frame(ticker)
        idx = pd.bdate_range("2020-01-02", periods=_DAYS, name="Date")
        before = np.arange(_DAYS) < _EVENT
        splits = np.zeros(_DAYS)
        if ticker == "GOOG":
            close = np.where(before, 100.0 / self.RATIO, 184.0)
            splits[_EVENT] = self.RATIO
            volume = np.full(_DAYS, 2e6)
        else:  # when-issued from three days before the distribution
            close = np.where(np.arange(_DAYS) >= _EVENT - 3, 40.0, np.nan)
            volume = np.where(np.isnan(close), np.nan, 2e6)
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close,
             "Volume": volume, "Dividends": 0.0, "Stock Splits": splits},
            index=idx,
        )


def _spin_off_event() -> ev.VerifiedEvent:
    days = pd.bdate_range("2020-01-02", periods=_DAYS)
    day, prev = days[_EVENT], days[_EVENT - 1]
    return ev.VerifiedEvent(
        ticker="GOOG", name="Alphabet (test)", date=str(day.date()),
        yahoo_ratio=SpinOffYahoo.RATIO, parent_shares_per_share=0.5,
        distributions=(ev.Distribution("CHILD", "Child (test)", 0.25),),
        record_date=str((day - pd.Timedelta(days=10)).date()), distribution_date=str(prev.date()),
        sources=(ev.Source("https://example.org/8-k", "Form 8-K (test)", ("terms",)),),
        verified_on="2026-10-08",
        price_checks=(
            ev.PriceCheck("GOOG", str(prev.date()), 99.0, 101.0, "test close", "https://example.org"),
            ev.PriceCheck("CHILD", str(day.date()), 39.0, 41.0, "test close", "https://example.org"),
        ),
    )


def test_script_build_applies_the_verified_events_and_documents_them(tmp_path, monkeypatch):
    from cif.data.provenance import verify_provenance

    script = _load_script()
    raw, out = tmp_path / "raw", tmp_path / "out"
    _save_snapshots(raw)
    out.mkdir()
    earlier = {"BLDR@2015-04-13": {"verdict": "real market move"}}
    (out / "corporate_action_checks.json").write_text(json.dumps(earlier))
    monkeypatch.setattr(pit, "yahoo_downloader", SpinOffYahoo())
    monkeypatch.setattr(ev, "VERIFIED_EVENTS", (_spin_off_event(),))
    day = pd.bdate_range("2020-01-02", periods=_DAYS)[_EVENT]
    assert script.main(["build", "--raw-dir", str(raw), "--out-dir", str(out), *_BUILD]) == 0

    table = pd.read_parquet(out / "adjustment_corrections.parquet")
    assert list(table["kind"]) == ["verified_spin_off"] and table["ticker"].iloc[0] == "GOOG"
    assert table["return_corrected"].iloc[0] == pytest.approx(SpinOffYahoo.TRUTH)
    fixed = pd.read_parquet(out / "prices_adj.parquet")
    assert fixed["GOOG"].pct_change().loc[day] == pytest.approx(SpinOffYahoo.TRUTH)
    gaps = pd.read_parquet(out / "adjustment_gaps.parquet")
    assert gaps.empty  # the inventory is Yahoo's as downloaded; this day is no dividend gap

    prov = json.loads((out / "provenance.json").read_text())
    assert prov["prices_adj_version"] == pit.ADJUSTMENT_CORRECTION_VERSION == "adjcorr-2"
    section = prov["adjustment_corrections"]
    assert section["version"] == "adjcorr-2" and section["n_events"] == 1
    assert section["by_kind"] == {"verified_spin_off": 1}
    verified = section["verified_events"]
    assert verified["applied"] is True and verified["n_applied"] == 1 and verified["skipped"] == {}
    assert [e["ticker"] for e in verified["events"]] == ["GOOG"]
    assert verified["events"][0]["return_corrected"] == pytest.approx(SpinOffYahoo.TRUTH)
    assert "src/cif/data/sp500_events.py" in prov["code_sha256"]
    requested = prov["yahoo"]["evidence_tickers"]["requested"]
    assert "CHILD" in requested and "GOOG" not in requested  # members are not asked for twice

    checks = json.loads((out / "corporate_action_checks.json").read_text())
    assert next(iter(checks)) == "BLDR@2015-04-13" and checks["BLDR@2015-04-13"] == earlier["BLDR@2015-04-13"]
    record = checks[f"GOOG@{day.date()}"]
    assert record["returns"]["corrected"] == pytest.approx(SpinOffYahoo.TRUTH)
    assert record["sources"][0]["url"] == "https://example.org/8-k" and record["price_checks"][0]["ok"]
    pseudo = pd.read_parquet(out / "pseudo_split_days.parquet").set_index("ticker")
    assert bool(pseudo.loc["GOOG", "corrected"])  # corrected now, though not by the gap rule
    assert verify_provenance(out / "provenance.json")["ok"]


def test_script_build_without_verified_events_is_the_gap_correction_only(tmp_path, monkeypatch):
    script = _load_script()
    raw, out = tmp_path / "raw", tmp_path / "out"
    _save_snapshots(raw)
    monkeypatch.setattr(pit, "yahoo_downloader", SpinOffYahoo())
    monkeypatch.setattr(ev, "VERIFIED_EVENTS", (_spin_off_event(),))
    args = ["build", "--raw-dir", str(raw), "--out-dir", str(out), *_BUILD, "--no-verified-events"]
    assert script.main(args) == 0
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["prices_adj_version"] == pit.ADJUSTMENT_CORRECTION_VERSION_GAPS == "adjcorr-1"
    assert prov["adjustment_corrections"]["version"] == "adjcorr-1"
    assert prov["adjustment_corrections"]["verified_events"] == {"applied": False}
    assert pd.read_parquet(out / "adjustment_corrections.parquet").empty
    assert not (out / "corporate_action_checks.json").exists()
    requested = prov["yahoo"]["evidence_tickers"]["requested"]
    assert "CHILD" not in requested  # nothing asks for the children of events that are off


def test_script_build_lists_the_registered_events_that_the_panel_cannot_hold(tmp_path, monkeypatch):
    script = _load_script()
    raw, out = tmp_path / "raw", tmp_path / "out"
    _save_snapshots(raw)
    monkeypatch.setattr(pit, "yahoo_downloader", EventYahoo())
    assert script.main(["build", "--raw-dir", str(raw), "--out-dir", str(out), *_BUILD]) == 0
    prov = json.loads((out / "provenance.json").read_text())
    verified = prov["adjustment_corrections"]["verified_events"]
    assert verified["n_events"] == len(ev.VERIFIED_EVENTS) == 6
    assert verified["n_applied"] == 0 and len(verified["skipped"]) == 6
    assert prov["adjustment_corrections"]["version"] == pit.ADJUSTMENT_CORRECTION_VERSION
    assert not (out / "corporate_action_checks.json").exists()  # nothing was verified here


def test_script_records_paths_inside_the_repository_relative_to_it():
    script = _load_script()
    assert script.repo_relative(script.ROOT / "data" / "raw" / "sp500_pit") == "data/raw/sp500_pit"
    assert script.repo_relative("/somewhere/else/raw") == "/somewhere/else/raw"


def test_script_diagnose_prints_funnel_top_list_checks_and_legacy_overlap(tmp_path, capsys):
    prices = _tiny_dataset(tmp_path)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"tickers": ["A", "B", "ZZ"]}))
    script = _load_script()
    date = str(prices.index[350].date())
    rc = script.main(
        ["diagnose", "--out-dir", str(tmp_path), "--dates", date, "--n", "3", "--check", "A", "D", "Z",
         "--lookback", "300", "--liquidity-window", "100", "--legacy-universe", str(legacy)]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "funnel: index members 6 -> with a downloaded series 6 -> complete history 6" in out
    assert "overlap with the legacy alphabetical 'top-100': 2 of 3" in out
    assert "A    : IN top-3: rank 1" in out
    assert "D    : NOT in top-3: rank 4" in out
    assert "Z    : not an index member on this date" in out


def test_script_survivorship_report_counts_member_days_without_a_price_inside_downloaded_series():
    days = pd.bdate_range("2020-01-01", periods=10, name="date")
    membership = pd.DataFrame(True, index=days, columns=["A", "B"])
    prices = pd.DataFrame(10.0, index=days, columns=["A", "B"])
    prices.loc[days[:4], "B"] = np.nan  # B is downloaded but its series starts on day 5
    companies = pd.DataFrame({"ticker": ["A", "B"], "in_current_list": [True, True]})
    cov = pit.compute_coverage(membership, prices, lookback=5)
    out = _load_script().survivorship_report(membership, cov, {}, companies, ["A", "B"])
    assert out["share_member_days_without_downloaded_series"] == 0.0
    assert out["share_member_days_without_price"] == pytest.approx(4 / 20)


def test_script_survivorship_report_counts_what_the_free_source_cannot_see():
    days = pd.bdate_range("2020-01-01", periods=10, name="date")
    tickers = ["CUR1", "CUR2", "GONE1", "GONE2"]
    membership = pd.DataFrame(True, index=days, columns=tickers)
    prices = pd.DataFrame(10.0, index=days, columns=["CUR1", "CUR2", "GONE2"])
    companies = pd.DataFrame({"ticker": tickers, "in_current_list": [True, True, False, False]})
    cov = pit.compute_coverage(membership, prices, lookback=5)
    out = _load_script().survivorship_report(
        membership, cov, {"GONE1": "possibly delisted; no timezone found"}, companies,
        ["CUR1", "CUR2", "GONE2"],
    )
    assert (out["n_tickers_ever_member"], out["n_downloaded"], out["n_failed"]) == (4, 3, 1)
    assert out["failed_among_current"] == 0 and out["failed_among_removed"] == 1
    assert out["download_rate_current"] == 1.0 and out["download_rate_removed"] == 0.5
    assert out["share_member_days_without_downloaded_series"] == pytest.approx(0.25)
    assert out["share_member_days_without_price"] == pytest.approx(0.25)  # from the coverage frame
    assert out["by_year"][2020]["mean_coverage"] == pytest.approx(0.75)
    assert out["failed_reasons"] == {"possibly delisted": 1}


def test_script_history_is_asked_again_when_yahoo_returns_nothing(monkeypatch):
    script = _load_script()
    monkeypatch.setattr(script.time, "sleep", lambda seconds: None)

    class Stub:
        def __init__(self, answer_on):
            self.calls, self.answer_on = 0, answer_on

        def history(self, **kwargs):
            self.calls += 1
            if self.calls < self.answer_on:
                return pd.DataFrame()
            return pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2020-01-02"]))

    stub = Stub(answer_on=3)
    assert len(script.fetch_history(stub, "2020-01-01", "2020-02-01", attempts=4)) == 1
    assert stub.calls == 3
    with pytest.raises(RuntimeError, match="no history"):
        script.fetch_history(Stub(answer_on=99), "2020-01-01", "2020-02-01", attempts=2)


def test_script_splits_list_accepts_what_yfinance_returns_for_a_stock_without_splits():
    script = _load_script()
    assert script.splits_list(pd.DataFrame(columns=["Stock Splits"])) == []  # newer yfinance
    assert script.splits_list(pd.Series([], dtype=float)) == []
    idx = pd.DatetimeIndex(["2014-06-09", "2020-08-31"], tz="America/New_York")
    series = pd.Series([7.0, 4.0], index=idx)
    assert script.splits_list(series) == [("2014-06-09", 7.0), ("2020-08-31", 4.0)]
    assert script.splits_list(series.to_frame("Stock Splits")) == script.splits_list(series)


# --------------------------------------------------------------------------------------------
# Legacy universe: behaviour unchanged, but it must say what top_n means
# --------------------------------------------------------------------------------------------


def test_legacy_resolve_universe_warns_that_top_n_is_alphabetical_and_changes_nothing(tmp_path):
    import warnings

    from cif.data import universe

    snap = universe.UniverseSnapshot(
        key="sp500", tickers=("MMM", "AOS", "ABT"), source="x", fetched_at="2026-01-01T00:00:00Z",
        notes="legacy",
    )
    universe.save_snapshot(tmp_path, snap)
    with pytest.warns(UserWarning, match="alphabetical by company name"):
        out = universe.resolve_universe("sp500", tmp_path, top_n=2)
    assert out == snap  # the stored snapshot is returned untouched, as before

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no top_n: nothing to warn about
        assert universe.resolve_universe("sp500", tmp_path) == snap


def test_legacy_docstrings_point_to_the_point_in_time_module():
    from cif.data import universe

    for obj in (universe.resolve_universe, universe.fetch_sp500_tickers):
        doc = obj.__doc__ or ""
        assert "sp500_pit" in doc and "alphabetical" in doc


# --------------------------------------------------------------------------------------------
# Network (opt-in): the live pages must still be parseable and give ~500 members
# --------------------------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("CIF_NETWORK_TESTS") != "1", reason="set CIF_NETWORK_TESTS=1")
def test_live_wikipedia_reconstruction_has_about_500_members():
    page = pit.fetch_wikipedia_html()
    cur = pit.parse_constituents(page)
    changes_page = page if pit.has_changes_table(page) else pit.fetch_wikipedia_html(
        pit.WIKI_CHANGES_URL
    )
    ch = pit.parse_changes(changes_page)
    assert 480 <= len(cur) <= 520 and len(ch) > 300
    rec = pit.reconstruct_membership(cur, ch, start="2010-01-01", as_of=pd.Timestamp.now())
    m = pit.intervals_to_membership(rec.intervals, "2010-01-01", "2025-12-31")
    counts = m.sum(axis=1)
    assert counts.between(490, 515).mean() > 0.99
