"""Point-in-time S&P 500 universe: membership on a date, prices, liquidity-ranked top-N.

Replaces the legacy ``resolve_universe("sp500", top_n=100)`` (R6), which takes the first 100 rows
of the Wikipedia table in the order of *company names* (3M ... Charles Schwab), uses today's
membership for the whole 2011-2025 window and keeps both share classes of one company.

Pipeline (see ``scripts/data/build_sp500_pit_dataset.py``):

1. Snapshot the Wikipedia pages (raw HTML + SHA-256): the current constituents and the table of
   index changes (which now lives on "Historical components of the S&P 500").
2. Parse both tables.
3. Undo the changes backwards in time from today's list (:func:`reconstruct_membership`) to get
   the membership on every date.
4. Download adjusted close, unadjusted close and volume from Yahoo Finance for every ticker that
   was a member at some date of the window, and blank the member days on which a series cannot
   be that member (:func:`mask_implausible_series`).
5. Correct the days on which Yahoo's adjusted close is not a total return
   (:func:`correct_adjustment_artifacts`, :func:`apply_verified_events`, "Adjusted close" below).
6. At a formation date, :func:`select_universe` takes the members of that date with a complete
   training history and ranks them by median dollar volume; only data up to the formation date
   is read.

Use::

    data = load_dataset("data/processed/sp500_pit")
    tickers = data.select("2020-01-02", n=100)  # members on that date, one ticker per company
    train = data.prices_adj.loc[:"2020-01-02", tickers].tail(756)

Adjusted close (R6, DHR 2016-07-05). ``prices_adj`` is Yahoo's ``Adj Close``, a total-return
index, except on the days where its return exceeds the split-adjusted close return by more than
10 % (``adjustment_gaps``; 26 days in the 2010-2025 build). There it is not a total return: Yahoo
credits a dividend as ``Close(t) / (Close(t-1) - D)`` instead of ``(Close(t) + D) / Close(t-1)``
(KDP 2018-07-10: +11.5 % against +1.8 % earned), counts a spin-off twice when it registers both a
value ratio and a dividend (DHR 2016-07-05: +61 % against +2.8 %) and ignores that a dividend
booked together with a split is per pre-split share (TMUS, EXPE). Those days get a total return
from the unadjusted close and the amounts Yahoo registered for the day (an in-kind distribution is
valued at the close of the distributed company where the data confirm the ratio); every day, with
its reason and evidence, is a row of ``adjustment_corrections.parquet`` and of ``provenance.json``.

Spin-off days without a dividend (audit D-2, ``adjcorr-2``). Yahoo books a spin-off as a ratio that
is no share-count ratio also on days without a dividend; ``adjustment_gaps`` does not list them, and
the return of ``Close`` and ``Adj Close`` is that of the remainder of the company: ITT 2011-11-01
+24.3 % in the data and +6.2 % for the holder, VTR 2015-08-18 -23.6 % and +0.3 % (Yahoo's ratio
there is the inverse of the value ratio). Six such days with a return beyond 10 % are registered in
:mod:`cif.data.sp500_events` with the terms the companies filed, the sources and the prices that fix
the units; :func:`apply_verified_events` replaces the day return by the holder's total return
``(k * P1 + sum n_i * C_i) / P0 - 1`` from as-traded closes, after the gap rule and by the same
``scale_before``. A distributed company without a Yahoo series (Exelis, Care Capital Properties,
Apartment Income REIT) is priced by the close or average the issuer documented. The units need
care: ``Close`` is divided by every split Yahoo books up to the day of the download, also those
after the end of the cached window that ``Stock Splits`` does not list (DD: 1-for-3 on 2026-06-24),
so the registry declares them and checks one price per series against a number in a filing.

Where the result lives: in ``prices_adj.parquet`` itself (``prices_adj_version`` in
``provenance.json``), not in a side file that the loader applies. A reader of the parquet (a
notebook on the cluster, a script that skips :func:`load_dataset`) then cannot take the false
returns by missing a step, and the version travels with the numbers. Yahoo's panel is not lost:
the cache is hashed (``files.raw``), ``build --no-adjustment-corrections`` writes it and
``SP500PIT.prices_adj_yahoo()`` undoes the correction exactly. Prices *before* a corrected day are
multiplied by a constant (``scale_before``), prices from the day on are Yahoo's, every return
outside the corrected days is as Yahoo has it, and the corrected return of a day reads only data
dated on or before it (the choice of the universe reads prices of ``prices_adj`` only for
validity, so it does not depend on the correction at all).

Known limits, quantified in ``provenance.json`` and ``coverage.parquet``:

* Yahoo has no history for most delisted tickers, so the downloaded panel is itself survivorship
  biased: about one in ten index members that left through an acquisition or merger has a
  series. ``compute_coverage`` gives the share of members with a price on each date.
* Yahoo serves the *current* holder of a symbol with that holder's whole history. For a symbol an
  index member gave up, the series can cover the member's period with an unrelated stock;
  :func:`mask_implausible_series` removes such member days when the stock is nearly untraded. A
  liquid unrelated stock under a reused symbol cannot be told apart.
* The Wikipedia table is a list of "selected changes", not an official feed. Dates follow two
  conventions (the day of the change or the previous trading day), so a change can be off by one
  business day. The reconstructed number of members is validated against ~500 on every date and
  anomalies are returned, never hidden.
* A corrected return can remain implausible (``plausible`` False in the corrections table): the
  amounts Yahoo registered and the price do not fit together and the data do not say which is
  wrong (DD 2019-06-03, a verified day, stays at +10.2 %: the holder's return between the two
  closes). A distribution whose recipient company has no Yahoo series (CSRA, Kraft Foods Group)
  keeps Yahoo's valuation. Yahoo books ratios that are no share-count ratios on days without a
  dividend too; the six days beyond 10 % are verified and corrected (``adjcorr-2``, above). The
  others (``pseudo_split_days.parquet``: 95 with a ratio above one, 7 below, the largest MSI
  2011-01-04 at +8.0 %) are not verified and nothing is changed there. The other dividend days
  keep Yahoo's convention, whose error :func:`uncorrected_dividend_error` bounds.
* A ticker is the key. Renames are resolved only when the pages show them (a reason that states
  the change, identical names on the two sides of a row, or the "Date added" of the current
  list); successors such as DISCA -> WBD or ARNC -> HWM are not linked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import partial
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from cif.data.sp500_events import VerifiedEvent

logger = logging.getLogger(__name__)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
WIKI_USER_AGENT = (
    "Mozilla/5.0 (cif research bot; https://github.com/konstantin/computing_in_finance) "
    "pandas/read_html"
)

# --------------------------------------------------------------------------------------------
# Snapshots of the Wikipedia pages
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HtmlSnapshot:
    """A saved copy of a Wikipedia page together with the facts needed to audit it."""

    path: Path
    url: str
    label: str
    fetched_at: str
    sha256: str
    n_bytes: int
    revision_id: int | None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "url": self.url,
            "label": self.label,
            "fetched_at": self.fetched_at,
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "revision_id": self.revision_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> HtmlSnapshot:
        return cls(
            path=Path(payload["path"]),
            url=payload["url"],
            label=payload["label"],
            fetched_at=payload["fetched_at"],
            sha256=payload["sha256"],
            n_bytes=int(payload["n_bytes"]),
            revision_id=payload.get("revision_id"),
        )

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class WikipediaSnapshots:
    """The two inputs of the reconstruction; ``changes`` may be the same page as ``constituents``."""

    constituents: HtmlSnapshot
    changes: HtmlSnapshot


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_revision_id(html: str) -> int | None:
    """Return MediaWiki's ``wgRevisionId`` embedded in the page, if present.

    Together with the page title it gives a permanent link (``?oldid=<id>``) to the exact text
    the tables were parsed from.
    """
    match = re.search(r'"wgRevisionId"\s*:\s*(\d+)', html)
    return int(match.group(1)) if match else None


def fetch_wikipedia_html(url: str = WIKI_SP500_URL, timeout: float = 30.0) -> str:
    """Download the page text. Wikipedia rejects the default urllib User-Agent."""
    response = requests.get(url, headers={"User-Agent": WIKI_USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.text


def save_html_snapshot(
    html: str,
    raw_dir: Path,
    url: str = WIKI_SP500_URL,
    label: str = "constituents",
    fetched_at: datetime | None = None,
) -> HtmlSnapshot:
    """Write ``wikipedia_<label>_<UTC stamp>.html`` and a ``.meta.json`` with its SHA-256."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fetched_at = fetched_at or datetime.now(timezone.utc)
    stamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    payload = html.encode("utf-8")
    snapshot = HtmlSnapshot(
        path=raw_dir / f"wikipedia_{label}_{stamp}.html",
        url=url,
        label=label,
        fetched_at=fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        sha256=sha256_bytes(payload),
        n_bytes=len(payload),
        revision_id=extract_revision_id(html),
    )
    snapshot.path.write_bytes(payload)
    snapshot.path.with_suffix(".meta.json").write_text(
        json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False)
    )
    return snapshot


def has_changes_table(html: str) -> bool:
    return re.search(r"<table[^>]*\bid=[\"']changes[\"']", html) is not None


def fetch_wikipedia_snapshots(
    raw_dir: Path,
    fetcher: Callable[[str], str] = fetch_wikipedia_html,
) -> WikipediaSnapshots:
    """Fetch and save the constituents page and, if it lacks the changes table, the changes page.

    The changes table moved to "Historical components of the S&P 500"; if it ever returns to the
    main page the snapshot is reused instead of fetching a second copy.
    """
    html = fetcher(WIKI_SP500_URL)
    constituents = save_html_snapshot(html, raw_dir, WIKI_SP500_URL, "constituents")
    if has_changes_table(html):
        return WikipediaSnapshots(constituents, constituents)
    changes_html = fetcher(WIKI_CHANGES_URL)
    changes = save_html_snapshot(changes_html, raw_dir, WIKI_CHANGES_URL, "changes")
    return WikipediaSnapshots(constituents, changes)


def _latest_snapshot(raw_dir: Path, label: str) -> HtmlSnapshot | None:
    metas = sorted(Path(raw_dir).glob(f"wikipedia_{label}_*.meta.json"))
    if not metas:
        return None
    meta = metas[-1]
    snap = HtmlSnapshot.from_dict(json.loads(meta.read_text()))
    # the stored path may be relative to another working directory: take it from the meta file
    html_path = meta.parent / (meta.name.removesuffix(".meta.json") + ".html")
    snap = replace(snap, path=html_path)
    actual = sha256_bytes(html_path.read_bytes())
    if actual != snap.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {html_path}: recorded {snap.sha256}, found {actual}"
        )
    return snap


def load_wikipedia_snapshots(raw_dir: Path) -> WikipediaSnapshots:
    """Latest saved snapshots in ``raw_dir``; every file is checked against its recorded SHA-256."""
    constituents = _latest_snapshot(raw_dir, "constituents")
    if constituents is None:
        raise FileNotFoundError(f"no wikipedia_constituents_*.meta.json in {raw_dir}")
    changes = _latest_snapshot(raw_dir, "changes") or constituents
    return WikipediaSnapshots(constituents, changes)


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------

_EMPTY_MARKS = frozenset({"", "-", "\u2013", "\u2014", "nan", "none"})


def _clean_cell(value: object) -> str | None:
    """Text of a table cell: no footnote marks, no NBSP; ``None`` for empty cells."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    return None if text.lower() in _EMPTY_MARKS else text


def to_yahoo_symbol(raw: object) -> str | None:
    """Wikipedia symbol -> Yahoo symbol: ``BRK.B`` -> ``BRK-B``; ``"ALLE |"`` -> ``ALLE``."""
    text = _clean_cell(raw)
    if text is None:
        return None
    match = re.match(r"[A-Za-z][A-Za-z0-9]*(?:[.\-][A-Za-z0-9]+)?", text)
    return match.group(0).upper().replace(".", "-") if match else None


def _flatten_header(column: object) -> str:
    parts = column if isinstance(column, tuple) else (column,)
    return " ".join(dict.fromkeys(str(p).strip().lower() for p in parts))


def _find_column(columns: Sequence[str], *must: str, any_of: Sequence[str] = ()) -> str | None:
    for col in columns:
        if all(word in col for word in must) and (not any_of or any(w in col for w in any_of)):
            return col
    return None


def parse_constituents(html: str) -> pd.DataFrame:
    """Current constituents: ``ticker, security, sector, sub_industry, cik, date_added``."""
    raw = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    raw.columns = [_flatten_header(c) for c in raw.columns]
    cols = list(raw.columns)
    sym = _find_column(cols, any_of=("symbol", "ticker"))
    sec = _find_column(cols, "security") or _find_column(cols, "company")
    sector = _find_column(cols, "sector")
    sub = _find_column(cols, "sub-industry") or _find_column(cols, "sub")
    cik = _find_column(cols, "cik")
    added = _find_column(cols, "date")
    if sym is None or sec is None:
        raise ValueError(f"unexpected constituents header: {cols}")

    def _cik(value: object) -> str | None:
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return None if pd.isna(number) else f"{int(number):010d}"

    def _date(value: object) -> pd.Timestamp:
        text = _clean_cell(value) or ""
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        return pd.Timestamp(match.group(0)) if match else pd.NaT

    out = pd.DataFrame(
        {
            "ticker": raw[sym].map(to_yahoo_symbol),
            "security": raw[sec].map(_clean_cell),
            "sector": raw[sector].map(_clean_cell) if sector else None,
            "sub_industry": raw[sub].map(_clean_cell) if sub else None,
            "cik": raw[cik].map(_cik) if cik else None,
            "date_added": raw[added].map(_date) if added else pd.NaT,
        }
    )
    out["date_added"] = pd.to_datetime(out["date_added"])
    return out.dropna(subset=["ticker"]).reset_index(drop=True)


def parse_changes(html: str) -> pd.DataFrame:
    """Index changes in table order: ``date, added_ticker/security, removed_ticker/security, reason``.

    Empty cells become ``None`` (a spin-off adds a ticker without removing one, and vice versa).
    Rows whose date cannot be parsed are dropped; their number is in ``df.attrs["n_rows_raw"]``
    minus ``len(df)``.
    """
    if not has_changes_table(html):
        raise ValueError("no table with id='changes' on the page")
    raw = pd.read_html(StringIO(html), attrs={"id": "changes"})[0]
    raw.columns = [_flatten_header(c) for c in raw.columns]
    cols = list(raw.columns)
    date_col = _find_column(cols, "date")
    add_t = _find_column(cols, "added", "ticker")
    add_s = _find_column(cols, "added", "security")
    rem_t = _find_column(cols, "removed", "ticker")
    rem_s = _find_column(cols, "removed", "security")
    reason = _find_column(cols, "reason")
    if None in (date_col, add_t, add_s, rem_t, rem_s):
        raise ValueError(f"unexpected changes header: {cols}")

    def _date(value: object) -> pd.Timestamp:
        text = _clean_cell(value)
        if text is None:
            return pd.NaT
        parsed = pd.to_datetime(text, format="%B %d, %Y", errors="coerce")
        return parsed if not pd.isna(parsed) else pd.to_datetime(text, errors="coerce")

    out = pd.DataFrame(
        {
            "date": raw[date_col].map(_date),
            "added_ticker": raw[add_t].map(to_yahoo_symbol),
            "added_security": raw[add_s].map(_clean_cell),
            "removed_ticker": raw[rem_t].map(to_yahoo_symbol),
            "removed_security": raw[rem_s].map(_clean_cell),
            "reason": raw[reason].map(_clean_cell) if reason else None,
        }
    )
    out["date"] = pd.to_datetime(out["date"])
    parsed = out.dropna(subset=["date"]).reset_index(drop=True)
    for col in parsed.columns.drop("date"):
        parsed[col] = parsed[col].astype(object).where(parsed[col].notna(), None)
    parsed.attrs["n_rows_raw"] = len(raw)
    return parsed


# --------------------------------------------------------------------------------------------
# Company identity
# --------------------------------------------------------------------------------------------

_NAME_NOISE = frozenset(
    {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc",
     "the", "sa", "nv", "ag", "lp", "llc"}
)


def normalize_company_name(name: object) -> str:
    """Name without share-class marks, punctuation and legal suffixes; ``""`` if empty.

    ``"Alphabet Inc. (Class A)"`` and ``"Alphabet Inc. (Class C)"`` both give ``"alphabet"``.
    """
    text = _clean_cell(name)
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(?:class|series)\s+[a-z0-9]+(?:\s+special)?\b", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(tok for tok in text.split() if tok not in _NAME_NOISE)


def assign_company_ids(companies: pd.DataFrame) -> dict[str, str]:
    """Ticker -> company id; share classes of one company get the same id.

    Two tickers belong to one company if they share a CIK or a normalised security name. The
    CIK is known only for current members, so historical classes (e.g. UA/UAA) fall back to the
    name. ``companies`` needs ``ticker`` and ``security``; ``cik`` is optional.
    """
    tickers = list(companies["ticker"])
    parent = {t: t for t in tickers}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_with_key: dict[str, str] = {}
    ciks = companies["cik"] if "cik" in companies else pd.Series([None] * len(companies))
    for ticker, security, cik in zip(tickers, companies["security"], ciks, strict=True):
        keys = []
        if isinstance(cik, str) and cik.strip():
            keys.append(f"cik:{cik.strip()}")
        norm = normalize_company_name(security)
        if norm:
            keys.append(f"name:{norm}")
        for key in keys:
            union(ticker, first_with_key.setdefault(key, ticker))

    cik_of: dict[str, str] = {}
    name_of: dict[str, str] = {}
    for ticker, security, cik in zip(tickers, companies["security"], ciks, strict=True):
        root = find(ticker)
        if isinstance(cik, str) and cik.strip():
            cik_of[root] = min(cik_of.get(root, cik.strip()), cik.strip())
        norm = normalize_company_name(security)
        if norm and root not in name_of:
            name_of[root] = norm
    return {
        t: (f"cik:{cik_of[find(t)]}" if find(t) in cik_of
            else f"name:{name_of[find(t)]}" if find(t) in name_of else f"ticker:{find(t)}")
        for t in tickers
    }


def build_companies(
    constituents: pd.DataFrame, changes: pd.DataFrame, tickers: Iterable[str]
) -> pd.DataFrame:
    """``ticker, security, cik, sector, in_current_list, company_id`` for the given tickers.

    The security name is the current one for today's members and otherwise the name on the most
    recent row of the changes table that mentions the ticker.
    """
    names: dict[str, str] = {}
    for row in changes.sort_values("date", ascending=False, kind="stable").itertuples():
        for ticker, security in (
            (row.removed_ticker, row.removed_security), (row.added_ticker, row.added_security)
        ):
            if ticker and security and ticker not in names:
                names[ticker] = security
    cur = constituents.set_index("ticker")
    rows = []
    for ticker in tickers:
        in_cur = ticker in cur.index
        rows.append(
            {
                "ticker": ticker,
                "security": cur.at[ticker, "security"] if in_cur else names.get(ticker),
                "cik": cur.at[ticker, "cik"] if in_cur else None,
                "sector": cur.at[ticker, "sector"] if in_cur else None,
                "in_current_list": in_cur,
            }
        )
    out = pd.DataFrame(rows, columns=["ticker", "security", "cik", "sector", "in_current_list"])
    out["company_id"] = out["ticker"].map(assign_company_ids(out))
    return out


# --------------------------------------------------------------------------------------------
# Membership on a date: undo the changes backwards from today's list
# --------------------------------------------------------------------------------------------

_ONE_DAY = pd.Timedelta(days=1)
_RENAME_WORDS = re.compile(
    r"renam|chang\w*\s+(?:its\s+|their\s+)?(?:name|ticker)|ticker\s+(?:symbol\s+)?chang"
    r"|ticker\s+symbol|changed\s+to",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Reconstruction:
    """Result of :func:`reconstruct_membership`.

    ``intervals``: ``ticker, start, end`` with inclusive dates; ``end`` is NaT for a member at the
    snapshot. ``anomalies``: ``date, kind, ticker, detail``. ``renames``: ``date, old, new, rule,
    kind`` for every identity decision that was taken.
    """

    intervals: pd.DataFrame
    anomalies: pd.DataFrame
    renames: pd.DataFrame


def _rename_rule(
    added: str | None, added_sec: str | None, removed: str | None, removed_sec: str | None,
    reason: str | None,
) -> str | None:
    """Is a row that adds one ticker and removes another really a ticker change?

    ``"reason"``: the reason states a rename/ticker change *and* names both the removed and the
    added company (by ticker or by name). A reason that explains why the removed company left
    ("Allergan was acquired by Actavis (and changed its name ...)") or that tells the story of
    the added one only ("... changed name to Evergy") does not make them one company.
    ``"name"``: both sides carry the same normalised company name.
    """
    if not added or not removed or added == removed:
        return None
    if reason and _RENAME_WORDS.search(reason):
        text = normalize_company_name(reason)

        def named(ticker: str, security: str | None) -> bool:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", reason):
                return True
            name = normalize_company_name(security)
            return bool(name) and name in text

        if named(removed, removed_sec) and named(added, added_sec):
            return "reason"
    norm_added = normalize_company_name(added_sec)
    if norm_added and norm_added == normalize_company_name(removed_sec):
        return "name"
    return None


def reconstruct_membership(
    constituents: pd.DataFrame,
    changes: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2010-01-01",
    as_of: str | pd.Timestamp,
    date_tolerance_days: int = 7,
) -> Reconstruction:
    """Membership intervals on ``[start, as_of]`` by undoing the changes from today's list.

    ``constituents`` (``ticker, security, date_added``) is the list on ``as_of``; ``changes`` the
    table of additions and removals. A change dated ``d`` makes the added ticker a member from
    ``d`` and the removed one a non-member from ``d``. Going backwards from ``as_of`` the added
    ticker stops being a member at ``d`` and the removed one becomes one with its last day at
    ``d - 1``. Rows dated after ``as_of`` (announced, not yet in the list) and before ``start``
    are not used.

    The two pages disagree in places, so three repairs are applied and recorded in ``renames``
    (identity decisions) or ``anomalies`` (what could not be repaired):

    * a row that adds ``X`` while ``X`` is not a member at that point is matched to a current
      member that no row adds and whose "Date added" lies within ``date_tolerance_days`` of the
      row (``FB`` -> ``META``), else to a member with the same normalised company name
      (``UA`` -> ``UAA``); if neither exists the row is reported as ``added_not_active``;
    * a row that adds ``X`` while the current list dates ``X`` long before and another current
      member carries the row's date as "Date added", the row belongs to that other member
      (``GOOGL`` row -> ``GOOG``);
    * a row that replaces ``A`` by ``B`` and says so (or has the same company on both sides) merges
      the interval of ``A`` that ends the day before into ``B`` (``WLTW`` -> ``WTW``).

    A current member that no row adds but whose "Date added" lies inside the window is kept as a
    member from the window start and reported as ``date_added_without_row``: that column of the
    current list is not the day of first entry for every company, and cutting on it would drop
    long-standing members. On the real pages the member count without such a cut is 500 on every
    day of 2010-2012, i.e. the table is consistent with its own start.
    """
    start = pd.Timestamp(start).normalize()
    as_of = pd.Timestamp(as_of).normalize()
    tol = pd.Timedelta(days=date_tolerance_days)

    cur_added = {
        t: d for t, d in zip(constituents["ticker"], constituents["date_added"], strict=True)
    }
    rows = changes.copy().reset_index(drop=True)
    rows["_order"] = np.arange(len(rows))
    explicit_added = set(rows["added_ticker"].dropna())
    removed_on: dict[str, list[pd.Timestamp]] = {}
    for tick, day in zip(rows["removed_ticker"], rows["date"], strict=True):
        if tick:
            removed_on.setdefault(tick, []).append(day)

    anomalies: list[dict] = []
    renames: list[dict] = []
    pair_renames: list[dict] = []

    def note(date, kind, ticker, detail=""):
        anomalies.append({"date": date, "kind": kind, "ticker": ticker, "detail": detail})

    for row in rows[rows["date"] > as_of].itertuples():
        note(row.date, "future_dated", row.added_ticker or row.removed_ticker,
             f"+{row.added_ticker} -{row.removed_ticker}: after the snapshot, ignored")
    rows = rows[(rows["date"] >= start) & (rows["date"] <= as_of)]
    rows = rows.sort_values(["date", "_order"], ascending=[False, True], kind="stable")

    active: dict[str, pd.Timestamp | None] = {t: None for t in constituents["ticker"]}
    names: dict[str, str] = {
        t: s for t, s in zip(constituents["ticker"], constituents["security"], strict=True) if s
    }
    unclaimed = {t for t in active if t not in explicit_added and pd.notna(cur_added.get(t))}
    spans: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]] = []

    def by_date_added(day: pd.Timestamp) -> str | None:
        near = {
            u: abs((cur_added[u] - day).days) for u in unclaimed
            if u in active and abs(cur_added[u] - day) <= tol
        }
        if not near:
            return None
        best = min(near.values())
        winners = sorted(u for u, gap in near.items() if gap == best)
        return winners[0] if len(winners) == 1 else None

    def by_name(security: str | None) -> str | None:
        norm = normalize_company_name(security)
        if not norm:
            return None
        hits = [u for u in active if normalize_company_name(names.get(u)) == norm]
        return hits[0] if len(hits) == 1 else None

    def close_start(ticker: str, day: pd.Timestamp) -> None:
        spans.append((ticker, day, active.pop(ticker)))
        unclaimed.discard(ticker)

    for day, group in rows.groupby("date", sort=False):
        batch = list(group.itertuples())
        for row in batch:  # undo additions first: a ticker added and removed on one day is continuous
            added = row.added_ticker
            if not added:
                continue
            taken = added
            if added in active:
                dated = cur_added.get(added)
                gone_between = any(dated <= r < day for r in removed_on.get(added, [])) \
                    if pd.notna(dated) else True
                if pd.notna(dated) and dated < day - tol and not gone_between:
                    other = by_date_added(day)
                    if other is not None:
                        renames.append({"date": day, "old": added, "new": other,
                                        "rule": "date_added_swap", "kind": "unmatched_add"})
                        taken = other
            else:
                other = by_date_added(day)
                rule = "date_added"
                if other is None:
                    other, rule = by_name(row.added_security), "name"
                if other is None:
                    note(day, "added_not_active", added, f"{row.added_security}: not a member "
                         "at this point and no current member or name matches")
                    continue
                renames.append({"date": day, "old": added, "new": other, "rule": rule,
                                "kind": "unmatched_add"})
                taken = other
            close_start(taken, day)
            rule = _rename_rule(row.added_ticker, row.added_security, row.removed_ticker,
                                row.removed_security, row.reason)
            if rule:
                pair_renames.append({"date": day, "old": row.removed_ticker, "new": taken,
                                     "rule": rule, "kind": "ticker_change"})
        for row in batch:
            removed = row.removed_ticker
            if not removed:
                continue
            if removed in active:
                note(day, "removed_while_active", removed,
                     "removed twice with no addition in between; kept as a member throughout")
                continue
            active[removed] = day - _ONE_DAY
            if row.removed_security:
                names[removed] = row.removed_security

    for ticker, end in active.items():
        dated = cur_added.get(ticker)
        if end is None and pd.notna(dated) and dated > start:
            # not used to cut the history: the 'Date added' of Sempra, Dominion, T. Rowe ... is
            # later than their real entry, while the member count stays at 500 without the cut
            note(dated, "date_added_without_row", ticker,
                 "current member dated inside the window by the current list, no row adds it; "
                 "kept as a member since the window start")
        spans.append((ticker, None, end))

    intervals = _merge_ticker_changes(spans, pair_renames, start, note)
    renames_df = pd.DataFrame(
        renames + pair_renames, columns=["date", "old", "new", "rule", "kind"]
    ).sort_values("date", ascending=False, kind="stable").reset_index(drop=True)
    anomalies_df = pd.DataFrame(anomalies, columns=["date", "kind", "ticker", "detail"])
    anomalies_df = anomalies_df.sort_values("date", ascending=False, kind="stable")
    return Reconstruction(intervals, anomalies_df.reset_index(drop=True), renames_df)


def _merge_ticker_changes(
    spans: list[tuple[str, pd.Timestamp | None, pd.Timestamp | None]],
    ticker_changes: list[dict],
    start: pd.Timestamp,
    note: Callable[..., None],
) -> pd.DataFrame:
    """Relabel the old ticker's interval to the new ticker and join adjacent intervals."""
    items = [
        {"orig": t, "label": t, "start": start if s is None else max(s, start), "end": e}
        for t, s, e in spans
    ]

    def overlaps(a: dict, b: dict) -> bool:
        a_end = a["end"] if a["end"] is not None else pd.Timestamp.max
        b_end = b["end"] if b["end"] is not None else pd.Timestamp.max
        return a["start"] <= b_end and b["start"] <= a_end

    for event in sorted(ticker_changes, key=lambda e: e["date"], reverse=True):
        day = event["date"]
        new = next((i for i in items if i["orig"] == event["new"] and i["start"] == day), None)
        old = next((i for i in items if i["orig"] == event["old"] and i["label"] == event["old"]
                    and i["end"] is not None and i["end"] == day - _ONE_DAY), None)
        if new is None or old is None:
            note(day, "rename_not_applied", event["old"],
                 f"{event['old']} -> {event['new']}: matching intervals not found")
            continue
        target = new["label"]
        clash = [j for j in items if j is not old and j is not new and j["label"] == target
                 and overlaps(old, j)]
        if clash:
            note(day, "rename_overlap", event["old"],
                 f"{event['old']} -> {target}: {target} is already a member then")
            continue
        old["label"] = target

    items.sort(key=lambda i: (i["label"], i["start"]))
    merged: list[dict] = []
    for item in items:
        last = merged[-1] if merged else None
        if last is not None and last["label"] == item["label"]:
            reach = pd.Timestamp.max if last["end"] is None else last["end"] + _ONE_DAY
            if item["start"] <= reach:  # adjacent or overlapping: one membership spell
                if last["end"] is not None and item["start"] <= last["end"]:
                    note(item["start"], "overlapping_intervals", item["label"],
                         f"{item['label']} is a member twice on {item['start'].date()}; merged")
                if last["end"] is not None and (item["end"] is None or item["end"] > last["end"]):
                    last["end"] = item["end"]
                continue
        merged.append(dict(item))
    out = pd.DataFrame(
        [{"ticker": m["label"], "start": m["start"], "end": m["end"]} for m in merged],
        columns=["ticker", "start", "end"],
    )
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])
    return out


def intervals_to_membership(
    intervals: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """Daily (calendar) boolean matrix ``date x ticker`` on ``[start, end]``.

    Interval bounds are inclusive; a missing ``start`` means "since before the window", a missing
    ``end`` "still a member". Tickers never a member inside the window are dropped.
    """
    index = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D", name="date")
    tickers = sorted(set(intervals["ticker"]))
    position = {t: k for k, t in enumerate(tickers)}
    matrix = np.zeros((len(index), len(tickers)), dtype=bool)
    for row in intervals.itertuples():
        lo = 0 if pd.isna(row.start) else index.searchsorted(row.start, side="left")
        hi = len(index) if pd.isna(row.end) else index.searchsorted(row.end, side="right")
        if lo < hi:
            matrix[lo:hi, position[row.ticker]] = True
    out = pd.DataFrame(matrix, index=index, columns=tickers)
    return out.loc[:, out.any(axis=0)]


def summarize_member_counts(
    membership: pd.DataFrame, low: int = 495, high: int = 510
) -> dict:
    """Distribution of the number of members per day and the days outside ``[low, high]``."""
    counts = membership.sum(axis=1)
    outside = (counts < low) | (counts > high)
    ranges = []
    if outside.any():
        run = (outside != outside.shift(fill_value=False)).cumsum()
        for _, block in counts[outside].groupby(run[outside]):
            ranges.append(
                {"start": str(block.index[0].date()), "end": str(block.index[-1].date()),
                 "n_days": len(block), "min": int(block.min()), "max": int(block.max())}
            )
    by_year = counts.groupby(counts.index.year).agg(["min", "median", "max"])
    return {
        "n_days": len(counts),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "median": float(counts.median()),
        "mean": float(counts.mean()),
        "band": [low, high],
        "share_in_band": float(1.0 - outside.mean()),
        "n_out_of_band": int(outside.sum()),
        "out_of_band_dates": [str(d.date()) for d in counts.index[outside]],
        "out_of_band_ranges": ranges,
        "by_year": {
            int(y): {"min": int(r["min"]), "median": float(r["median"]), "max": int(r["max"])}
            for y, r in by_year.iterrows()
        },
    }


# --------------------------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------------------------

YAHOO_FIELDS = ("Adj Close", "Close", "Volume", "Dividends", "Stock Splits")
_PERMANENT = re.compile(
    r"delisted|no data|no price data|not found|no timezone|symbol may be|invalid", re.IGNORECASE
)

Downloader = Callable[[Sequence[str], str, str], tuple[pd.DataFrame | None, dict[str, str]]]


def yahoo_downloader(
    batch: Sequence[str], start: str, end: str
) -> tuple[pd.DataFrame | None, dict[str, str]]:
    """One ``yf.download`` call; returns the raw frame and Yahoo's per-ticker error messages.

    ``auto_adjust=False`` returns ``Close`` (split-adjusted only) and ``Adj Close`` (splits and
    dividends), the pair needed for dollar volume and total return from a single request.
    """
    import yfinance as yf

    frame = yf.download(
        list(batch), start=str(start), end=str(end), auto_adjust=False, actions=True,
        progress=False, group_by="ticker", threads=True, timeout=30,
    )
    # per-ticker error texts of the last call ("possibly delisted", rate limits, ...)
    raw_errors = getattr(getattr(yf, "shared", None), "_ERRORS", None) or {}
    errors = {str(k).upper(): str(v) for k, v in raw_errors.items()}
    return frame, errors


def split_yahoo_frame(frame: pd.DataFrame | None, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Per-ticker frames from a ``yf.download`` result of either column layout."""
    if frame is None or frame.empty:
        return {}
    if isinstance(frame.columns, pd.MultiIndex):
        for level in (0, 1):
            present = set(frame.columns.get_level_values(level))
            if present & set(tickers):
                return {t: frame.xs(t, axis=1, level=level) for t in tickers if t in present}
        return {}
    return {tickers[0]: frame} if len(tickers) == 1 else {}


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame({"ticker": [], "date": pd.to_datetime([]), "amount": []})


@dataclass(frozen=True)
class YahooPanel:
    """Yahoo's fields as wide frames plus the actions: ``splits`` (``ticker, date, ratio``) and
    ``dividends`` (``ticker, date, amount``), in the units Yahoo reports them."""

    adj_close: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    splits: pd.DataFrame
    failed: dict[str, str]
    dividends: pd.DataFrame = field(default_factory=_empty_dividends)


def tidy_yahoo_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """One ticker's Yahoo frame: the five fields, naive sorted ``date`` index, no empty rows."""
    out = frame.reindex(columns=list(YAHOO_FIELDS)).copy()
    index = pd.to_datetime(out.index)
    out.index = index.tz_localize(None) if index.tz is not None else index
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.dropna(subset=["Adj Close", "Close", "Volume"], how="all")


def _cache_files(cache_dir: Path, ticker: str) -> tuple[Path, Path]:
    return cache_dir / f"{ticker}.parquet", cache_dir / f"{ticker}.failed.json"


def download_yahoo_panel(
    tickers: Iterable[str],
    start: str,
    end: str,
    *,
    batch_size: int = 25,
    pause: float = 1.5,
    retries: int = 2,
    retry_pause: float | None = None,
    downloader: Downloader | None = None,
    cache_dir: Path | None = None,
) -> YahooPanel:
    """Adjusted close, unadjusted close, volume and splits for ``tickers`` from Yahoo Finance.

    Tickers are requested in batches with a pause between requests. Whatever fails is asked for
    again alone, because a batch can report a listed ticker as "possibly delisted" (DELL did):
    a failure that sounds permanent ("possibly delisted", no rows) counts only after one request
    for that ticker alone says so too; anything else (rate limits, timeouts) is retried alone up
    to ``retries`` times. With ``cache_dir`` every result is stored per ticker and a rerun asks
    only for tickers that are neither cached nor confirmed gone, so an interrupted download can
    be resumed (use a directory per ``start``/``end``).
    """
    tickers = list(dict.fromkeys(tickers))
    downloader = downloader or yahoo_downloader
    retry_pause = 3.0 * pause if retry_pause is None else retry_pause
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    failed: dict[str, str] = {}  # confirmed gone
    suspect: dict[str, str] = {}  # sounds permanent, not yet confirmed by a request alone
    transient: dict[str, str] = {}  # rate limits and the like: retry
    pending: list[str] = []
    for ticker in tickers:
        if cache is not None:
            ok_path, bad_path = _cache_files(cache, ticker)
            if ok_path.exists():
                frames[ticker] = pd.read_parquet(ok_path)
                continue
            if bad_path.exists():
                record = json.loads(bad_path.read_text())
                # failures written before they were confirmed one by one are asked for again
                (failed if record.get("confirmed") else suspect)[ticker] = record["reason"]
                continue
        pending.append(ticker)

    def fetch(batch: list[str]) -> None:
        alone = len(batch) == 1
        raw, errors = downloader(batch, start, end)
        per_ticker = split_yahoo_frame(raw, batch)
        for ticker in batch:
            tidy = tidy_yahoo_frame(per_ticker[ticker]) if ticker in per_ticker else None
            if tidy is not None and len(tidy):
                frames[ticker] = tidy
                suspect.pop(ticker, None)
                transient.pop(ticker, None)
                if cache is not None:
                    ok_path, bad_path = _cache_files(cache, ticker)
                    tidy.to_parquet(ok_path)
                    bad_path.unlink(missing_ok=True)
                continue
            reason = errors.get(ticker.upper()) or "no data returned"
            transient.pop(ticker, None)
            if not _PERMANENT.search(reason):
                suspect.pop(ticker, None)
                transient[ticker] = reason
            elif alone:
                suspect.pop(ticker, None)
                failed[ticker] = reason
                if cache is not None:
                    _cache_files(cache, ticker)[1].write_text(
                        json.dumps({"reason": reason, "confirmed": True})
                    )
            else:
                suspect[ticker] = reason

    n_batches = (len(pending) + batch_size - 1) // batch_size
    for k in range(n_batches):
        batch = pending[k * batch_size : (k + 1) * batch_size]
        logger.info("yahoo batch %d/%d (%d tickers)", k + 1, n_batches, len(batch))
        fetch(batch)
        if k + 1 < n_batches:
            time.sleep(pause)
    for attempt in range(max(retries, 1)):
        todo = (list(suspect) if attempt == 0 else []) + (list(transient) if attempt < retries else [])
        if not todo:
            break
        logger.info("yahoo alone, round %d: %d tickers", attempt + 1, len(todo))
        if attempt > 0:
            time.sleep(retry_pause)
        for ticker in dict.fromkeys(todo):
            fetch([ticker])
            time.sleep(pause)
    failed.update(suspect)
    failed.update(transient)

    order = [t for t in tickers if t in frames]
    if order:
        adj = pd.concat({t: frames[t]["Adj Close"] for t in order}, axis=1).sort_index()
        close = pd.concat({t: frames[t]["Close"] for t in order}, axis=1).sort_index()
        volume = pd.concat({t: frames[t]["Volume"] for t in order}, axis=1).sort_index()
    else:
        adj = close = volume = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    for panel in (adj, close, volume):
        panel.index.name = "date"
        panel.columns.name = None
    split_rows, dividend_rows = [], []
    for t in order:
        ratio = frames[t]["Stock Splits"].dropna()
        for day, value in ratio[ratio != 0].items():
            split_rows.append({"ticker": t, "date": day, "ratio": float(value)})
        paid = frames[t]["Dividends"].dropna()
        for day, value in paid[paid != 0].items():
            dividend_rows.append({"ticker": t, "date": day, "amount": float(value)})
    splits = pd.DataFrame(split_rows, columns=["ticker", "date", "ratio"])
    dividends = pd.DataFrame(dividend_rows, columns=["ticker", "date", "amount"])
    dividends["date"] = pd.to_datetime(dividends["date"])
    return YahooPanel(adj, close, volume, splits, dict(sorted(failed.items())), dividends)


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Row counts and, per shared column, the largest ``|a - b| / |a|`` on the common dates.

    Used to check a batch download against a one-ticker download of the same series.
    """
    common = a.index.intersection(b.index)
    out: dict = {"rows_a": len(a), "rows_b": len(b), "rows_common": len(common)}
    mismatches = 0
    for col in a.columns.intersection(b.columns):
        x, y = a.loc[common, col].astype(float), b.loc[common, col].astype(float)
        mismatches += int((x.isna() != y.isna()).sum())
        both = x.notna() & y.notna()
        rel = (x[both] - y[both]).abs() / x[both].abs().replace(0.0, np.nan)
        out[col] = float(rel.max()) if both.any() and rel.notna().any() else 0.0
    out["n_nan_mismatch"] = mismatches
    return out


# --------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------

_EPS = 1e-12


def compute_coverage(
    membership: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    lookback: int = 756,
    min_history_frac: float = 0.95,
) -> pd.DataFrame:
    """Per trading day: members, members with a price, and members with a complete history.

    ``coverage = n_with_price / n_members`` is the share of the index for which data exist; its
    complement is the part of the index the study cannot see (mostly delisted names).
    ``n_eligible`` counts members whose last ``lookback`` days hold at least ``min_history_frac``
    valid prices (NaN until a full window exists).
    """
    index = prices.index
    member = membership.reindex(index=index, fill_value=False).astype(bool)
    valid = (prices.reindex(columns=member.columns) > 0).fillna(False)
    n_members = member.sum(axis=1)
    n_priced = (member & valid).sum(axis=1)
    frac = valid.astype(float).rolling(lookback, min_periods=lookback).mean()
    eligible = (member & (frac >= min_history_frac - _EPS)).sum(axis=1).astype(float)
    eligible[np.arange(len(index)) < lookback - 1] = np.nan
    return pd.DataFrame(
        {
            "n_members": n_members.astype(int),
            "n_with_price": n_priced.astype(int),
            "coverage": (n_priced / n_members.replace(0, np.nan)),
            "n_eligible": eligible,
        },
        index=index,
    )


_REMOVAL_CATEGORIES = (
    ("bankruptcy / failure", r"bankrupt|receivership|fdic|chapter 11|liquidat"),
    ("acquired / merged", r"acqui|merg|buyout|purchase|takeover|taken private|tender|consortium"),
    ("spin-off related", r"spin"),
    ("size / eligibility", r"market cap|public float|float|methodology|eligib|no longer listed"),
)


def removal_breakdown(
    changes: pd.DataFrame,
    renames: pd.DataFrame,
    membership: pd.DataFrame,
    downloaded: Iterable[str],
    start: str | pd.Timestamp,
    as_of: str | pd.Timestamp,
) -> dict[str, dict]:
    """Why index members left inside the window, and for how many of them a series exists.

    Counts the removal rows whose ticker is a column of ``membership``; a ticker that merely
    changed its symbol (``renames`` with ``kind == "ticker_change"``) did not leave. A row is
    classified by the first matching keyword of its reason. The point is the direction of the
    gap: the names a free source no longer has are mostly those that were acquired or failed.
    """
    changed = {(r.date, r.old) for r in renames.itertuples() if r.kind == "ticker_change"}
    rows = changes[
        (changes["date"] >= pd.Timestamp(start)) & (changes["date"] <= pd.Timestamp(as_of))
        & changes["removed_ticker"].notna()
    ]
    have = set(downloaded)
    out: dict[str, dict] = {}
    for row in rows.itertuples():
        if row.removed_ticker not in membership.columns or (row.date, row.removed_ticker) in changed:
            continue
        reason = (row.reason or "").lower()
        label = next(
            (name for name, pattern in _REMOVAL_CATEGORIES if re.search(pattern, reason)), "other"
        )
        cell = out.setdefault(label, {"n_removals": 0, "n_with_series": 0})
        cell["n_removals"] += 1
        cell["n_with_series"] += int(row.removed_ticker in have)
    for cell in out.values():
        cell["share_with_series"] = cell["n_with_series"] / cell["n_removals"]
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n_removals"]))


def series_vs_membership(
    membership: pd.DataFrame, prices: pd.DataFrame, min_gap_days: int = 90
) -> dict:
    """Series whose first (last) price is far after (before) the first (last) day of membership.

    A series that starts long after the membership claims it began is either a stale membership
    (the company was not yet in the index, or not yet listed) or a ticker reused by another
    company; one that ends long before the membership does leaves a hole in a spell of membership.
    """
    rows = []
    for ticker in prices.columns.intersection(membership.columns):
        member_days = membership.index[membership[ticker].to_numpy()]
        valid = prices[ticker].dropna().index
        if len(member_days) == 0 or len(valid) == 0:
            continue
        late = (valid[0] - member_days[0]).days
        early = (member_days[-1] - valid[-1]).days
        if late > min_gap_days or early > min_gap_days:
            rows.append(
                {
                    "ticker": ticker,
                    "first_member": str(member_days[0].date()),
                    "first_price": str(valid[0].date()),
                    "late_days": max(late, 0),
                    "last_member": str(member_days[-1].date()),
                    "last_price": str(valid[-1].date()),
                    "early_days": max(early, 0),
                }
            )
    rows.sort(key=lambda r: -max(r["late_days"], r["early_days"]))
    return {"min_gap_days": min_gap_days, "n_tickers": len(rows), "tickers": rows[:60]}


def large_move_inventory(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    threshold: float = 0.6,
    window_days: int = 5,
    baseline_days: int = 60,
) -> pd.DataFrame:
    """One-day moves larger than ``threshold`` in the adjusted close, with their context.

    For each such day (between two consecutive valid prices of a ticker): the return, the
    calendar gap to the previous valid price, the splits listed by Yahoo within ``window_days``
    calendar days (count, and the ratio of the nearest one) and the volume as a multiple of its
    median over the previous ``baseline_days`` valid days. Nothing is changed in the prices: this
    is the list that a cleaning rule has to be checked against. A move next to a split is an
    adjustment problem; a move with a volume spike and no split is a market move (BLDR).
    Sorted by ``|return|``, largest first.
    """
    columns = ["ticker", "date", "previous_date", "gap_days", "return", "n_splits_nearby",
               "split_ratio_nearby", "volume_ratio"]
    by_ticker = {t: g for t, g in splits.groupby("ticker")} if len(splits) else {}
    window = pd.Timedelta(days=window_days)
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        ret = series.pct_change()
        big = ret[ret.abs() > threshold]
        if big.empty:
            continue
        vol = volume[ticker].reindex(series.index) if ticker in volume else None
        listed = by_ticker.get(ticker)
        for day, value in big.items():
            pos = series.index.get_loc(day)
            near = pd.DataFrame(columns=["date", "ratio"])
            if listed is not None:
                near = listed[(listed["date"] >= day - window) & (listed["date"] <= day + window)]
            ratio = float("nan")
            if len(near):
                ratio = float(near.loc[(near["date"] - day).abs().idxmin(), "ratio"])
            vol_ratio = float("nan")
            if vol is not None:
                base = vol.iloc[max(0, pos - baseline_days) : pos].median()
                if base > 0:
                    vol_ratio = float(vol.iloc[pos] / base)
            rows.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "previous_date": series.index[pos - 1],
                    "gap_days": (day - series.index[pos - 1]).days,
                    "return": float(value),
                    "n_splits_nearby": len(near),
                    "split_ratio_nearby": ratio,
                    "volume_ratio": vol_ratio,
                }
            )
    out = pd.DataFrame(rows, columns=columns)
    return out.reindex(out["return"].abs().sort_values(ascending=False, kind="stable").index
                       ).reset_index(drop=True)


def adjustment_gaps(
    prices_adj: pd.DataFrame, close_unadj: pd.DataFrame, *, threshold: float = 0.10
) -> pd.DataFrame:
    """Days on which the adjusted-close return exceeds the close return by more than ``threshold``.

    ``close_unadj`` is adjusted for splits only, so the difference of the two one-day returns is
    what Yahoo credits as distributions on that day: a percent or two for an ordinary dividend,
    tens of percent for a special dividend or a spin-off. Nothing is changed. The list is for
    checking against corporate events: DHR on 2016-07-05 (the Fortive spin-off) is +61 % in the
    adjusted close while the close itself moved +3.7 %. Sorted by ``|gap|``, largest first.
    """
    ret_adj = prices_adj.pct_change(fill_method=None)
    ret_close = close_unadj.reindex_like(prices_adj).pct_change(fill_method=None)
    gap = (ret_adj - ret_close).stack()
    gap = gap[gap.abs() > threshold]
    gap = gap.reindex(gap.abs().sort_values(ascending=False, kind="stable").index)
    out = gap.rename("gap").reset_index()
    out.columns = ["date", "ticker", "gap"]
    out["return_adjusted"] = [ret_adj.at[d, t] for d, t in zip(out["date"], out["ticker"], strict=True)]
    out["return_close"] = [ret_close.at[d, t] for d, t in zip(out["date"], out["ticker"], strict=True)]
    return out[["ticker", "date", "return_adjusted", "return_close", "gap"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Distribution days: where Yahoo's adjusted close is not a total return (R6, DHR 2016-07-05)
# --------------------------------------------------------------------------------------------

ADJUSTMENT_CORRECTION_VERSION = "adjcorr-2"  # the gap rule below plus the verified events
ADJUSTMENT_CORRECTION_VERSION_GAPS = "adjcorr-1"  # the gap rule alone
PLAUSIBLE_DAY_RETURN = 0.08  # a corrected day return beyond this is flagged (not changed)
_DUPLICATE_TOLERANCE = 0.03  # dividend vs the value the split ratio removed, relative
_CHILD_WINDOW_DAYS = 30  # a distributed company's first (when-issued) quote is at most this early
_CHILD_RATIOS = (*(1.0 / m for m in range(1, 21)), 2.0, 3.0, 4.0, 5.0, 10.0)

CORRECTION_COLUMNS = [
    "ticker", "date", "previous_date", "kind", "value_source", "close_previous", "close_event",
    "dividend", "dividend_over_close", "split_ratio", "later_split_factor", "volume_ratio",
    "return_close", "return_yahoo", "adj_formula_residual", "return_corrected", "change",
    "scale_before", "child_ticker", "child_n", "child_exact", "plausible", "note",
]

_NOTES = {
    "cash_dividend": (
        "Adj Close credits the dividend as Close/(Close_prev - D) - 1, which is the total return "
        "divided by 1 - D/Close_prev; total return = (Close + D)/Close_prev - 1"
    ),
    "dividend_with_real_split": (
        "a share-count split on the same day: Yahoo's amount is per pre-split share, D/ratio per "
        "share held afterwards; total return = (Close + D/ratio)/Close_prev - 1"
    ),
    "pseudo_split_duplicate_value": (
        "the value ratio already took the distribution out of the earlier Close and the dividend "
        "(equal to that value) is credited again in Adj Close; total return = Close return/ratio"
    ),
    "pseudo_split_plus_cash": (
        "the value ratio carries the in-kind distribution and the dividend is separate cash; "
        "total return = (Close return + D/Close_prev)/ratio"
    ),
    "unexplained": (
        "no dividend or split registered on the day explains the gap; Yahoo's value is kept"
    ),
    "verified_spin_off": (
        "the terms are those of the company's filings (cif.data.sp500_events); Yahoo books a ratio "
        "that is no share-count ratio and the day return is that of the remainder of the company; "
        "total return = (k * P1 + sum n_i * C_i) / P0 - 1 from as-traded closes"
    ),
}


def is_share_count_ratio(ratio: float, *, max_denominator: int = 5) -> bool:
    """Does a Yahoo "Stock Splits" ratio change the number of shares (2, 1/2, 3/2, 1/4 ...)?

    Yahoo also books spin-offs and mergers as ratios such as 1.319 (DHR, 2016-07-05) or 9.295
    (AIV, 2020-12-15): the earlier prices are divided by them but nobody's share count changed.
    ``ratio`` or ``1 / ratio`` is a share-count ratio only if it is ``k / q`` with ``q`` at most
    ``max_denominator``. A rule of form, not a proof: on the 285 ratios of the panel it gives 174
    simple ones (0.02 ... 50, 1.2, 1.25, 1.5) and 111 others (0.247 ... 9.295, mostly 1.01 ... 1.2
    with three decimals), among them the spin-offs above and a few exact-looking share changes
    booked with an inexact ratio (GOOG 2014-03-27: 2.002).
    """
    if not ratio > 0:
        return False
    for value in (ratio, 1.0 / ratio):
        for q in range(1, max_denominator + 1):
            scaled = value * q
            if abs(scaled - round(scaled)) < 1e-6 and round(scaled) <= 100:
                return True
    return False


def _later_factor(splits_by: Mapping[str, pd.DataFrame], ticker: str, day: pd.Timestamp) -> float:
    """Product of the ratios of the splits of ``ticker`` dated after ``day``: raw = Close x this."""
    rows = splits_by.get(ticker)
    if rows is None:
        return 1.0
    later = rows.loc[rows["date"] > day, "ratio"]
    return float(later.prod()) if len(later) else 1.0


def _simple_ratio(amount: float, price: float) -> tuple[float, float] | None:
    """The simple ``n`` (1/m, 2, 3 ...) with ``amount == n * price`` up to the rounding of quotes."""
    best = None
    for n in _CHILD_RATIOS:
        err = abs(amount - n * price)
        if err <= 0.0055 + 5e-4 * amount and (best is None or err < best[1]):
            best = (n, err)
    return best


def _confirm_child(
    day: pd.Timestamp,
    amounts: Sequence[float],
    parent: str,
    children_close: pd.DataFrame,
    first_valid: pd.Series,
    splits_by: Mapping[str, pd.DataFrame],
) -> dict | None:
    """A company that began to trade shortly before ``day`` and explains ``amounts`` as ``n`` shares.

    Yahoo values an in-kind distribution at the distributed company's last quote before the
    day (when-issued): EXPE 15.125 = 1/2 x 30.25 of TRIP, XRX 2.98 = 1/5 x 14.90 of Conduent. If
    one of ``amounts`` is such a simple multiple of a candidate's last raw close before ``day``, the
    candidate is confirmed, with ``kappa`` = its raw close on ``day`` over that last close. Only the
    candidates' prices up to ``day`` are read.
    """
    lo = day - pd.Timedelta(days=_CHILD_WINDOW_DAYS)
    best = None
    for ticker in sorted(first_valid.index[(first_valid >= lo) & (first_valid < day)]):
        if ticker == parent:
            continue
        series = children_close[ticker].dropna()
        if day not in series.index:
            continue
        before = series.loc[: day - pd.Timedelta(days=1)]
        prev_day = before.index[-1]
        if (day - prev_day).days > 10:
            continue
        own = splits_by.get(ticker)
        if own is not None and ((own["date"] > prev_day) & (own["date"] <= day)).any():
            continue  # a split between the two quotes would make their ratio meaningless
        prev = float(before.iloc[-1]) * _later_factor(splits_by, ticker, prev_day)
        now = float(series.loc[day]) * _later_factor(splits_by, ticker, day)
        for amount in amounts:
            # the amount as registered comes first: after a later 1:4 split 0.2 would look like 0.05
            hit = _simple_ratio(amount, prev)
            if hit is not None:
                if best is None or hit[1] < best["error"] - 1e-9:
                    best = {"ticker": ticker, "n": hit[0], "error": hit[1], "kappa": now / prev}
                break
    return best


def _starting_child(
    day: pd.Timestamp,
    amount: float,
    parent: str,
    children_close: pd.DataFrame,
    first_valid: pd.Series,
    splits_by: Mapping[str, pd.DataFrame],
) -> tuple[str, float] | None:
    """The one company whose series starts on ``day`` (or <= 3 days before): ``amount`` / its close.

    No quote before the day, so the ratio cannot be confirmed (DHR/FTV: 24.56 / 48.60 = 0.505);
    reported for the record, not used.
    """
    if day not in children_close.index:
        return None
    names = [
        t for t in first_valid.index[(first_valid >= day - pd.Timedelta(days=3)) & (first_valid <= day)]
        if t != parent and pd.notna(children_close.at[day, t])
    ]
    if len(names) != 1:
        return None
    raw = float(children_close.at[day, names[0]]) * _later_factor(splits_by, names[0], day)
    return names[0], amount / raw


def _classify_gap(
    cp: float,
    ct: float,
    amount: float,
    ratio: float,
    later: float,
    child_of: Callable[[Sequence[float]], dict | None],
) -> tuple[str, str, float, dict | None]:
    """``(kind, value_source, total_return, child)`` of one gap day from its registered actions.

    ``cp`` and ``ct`` are the Close before and on the day, ``amount`` the dividend and ``ratio``
    the split ratio Yahoo registered for the day (0 if none), ``later`` the product of the ratios
    of the later splits. ``child_of(amounts)`` looks for the company that makes one of ``amounts``
    a simple multiple of its shares.
    """
    if amount <= 0.0:
        return "unexplained", "none", math.nan, None
    if ratio == 0.0:
        return "cash_dividend", "yahoo_dividend", (ct + amount) / cp - 1.0, None
    amounts = (amount, amount * later)
    if is_share_count_ratio(ratio):
        value, source, child = amount / ratio, "yahoo_dividend", child_of(amounts)
        if child is not None:
            value, source = value * child["kappa"], "child_close"
        return "dividend_with_real_split", source, (ct + value) / cp - 1.0, child
    in_kind = cp * (ratio - 1.0)  # the value the ratio took out of the earlier Close
    if min(abs(amount - x) / x for x in (in_kind, in_kind * later)) <= _DUPLICATE_TOLERANCE:
        source, child = "yahoo_split_ratio", child_of(amounts)
        if child is not None:
            in_kind, source = in_kind * child["kappa"], "child_close"
        return "pseudo_split_duplicate_value", source, (ct + in_kind) / (cp * ratio) - 1.0, child
    r_star = (ct + in_kind + amount) / (cp * ratio) - 1.0
    return "pseudo_split_plus_cash", "yahoo_split_ratio+yahoo_dividend", r_star, None


@dataclass(frozen=True)
class AdjustmentCorrections:
    """``prices_adj`` with the artifact days corrected and ``table``, one row per corrected day."""

    prices_adj: pd.DataFrame
    table: pd.DataFrame


def correct_adjustment_artifacts(
    prices_adj: pd.DataFrame,
    close_unadj: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    volume: pd.DataFrame | None = None,
    children_close: pd.DataFrame | None = None,
    threshold: float = 0.10,
    plausible: float = PLAUSIBLE_DAY_RETURN,
) -> AdjustmentCorrections:
    """Replace Yahoo's day return on the days of :func:`adjustment_gaps` by a total return.

    ``Adj Close`` is a total-return index on ordinary days. On a day where its return exceeds the
    split-adjusted close return by more than ``threshold`` it is not, for reasons the actions
    Yahoo registered for the day tell apart (``kind`` of the table):

    * ``cash_dividend`` (dividend, no split). Yahoo multiplies every earlier price by
      ``1 - D / Close(t-1)``, so its return is ``Close(t) / (Close(t-1) - D) - 1``, the total return
      divided by ``1 - d`` with ``d = D / Close(t-1)``; the holder earned
      ``(Close(t) + D) / Close(t-1) - 1``. KDP 2018-07-10 (a special dividend of 84 % of the
      price): +11.5 % in Adj Close, +1.8 % earned.
    * ``dividend_with_real_split``: a split that changes the share count (1:2) and a dividend on
      the same day (TMUS 2013-05-01, EXPE 2011-12-21). The amount is per pre-split share,
      ``D / ratio`` per share held afterwards; Yahoo's formula above ignores that. Data: EXPE's
      15.125 is exactly 1/2 of the 30.25 of one TRIP share; for TMUS the reading per post-split
      share gives -13 %, per pre-split share +4 %.
    * ``pseudo_split_*``: a ratio that is no share-count ratio (:func:`is_share_count_ratio`).
      Read as a spin-off, Yahoo books ``s = P / (P - V)`` and divides every earlier Close by it:
      Close already leaves out the value ``V = Close(t-1) (s - 1)`` that was handed out and its day
      return is ``s`` times the total return. If the day's dividend equals ``V`` in raw units
      (``pseudo_split_duplicate_value``: DHR 24.56 against 24.75, XRX 2.98 against 2.979) Adj Close
      takes the value out a second time (DHR 2016-07-05: +61 % against +3.6 % in Close and +2.8 %
      earned). If it does not (``pseudo_split_plus_cash``: DXC, 9.083 x 1.156 = 10.50 against the
      29.2 the ratio removed) the dividend is cash on top. Total return ``Close return / s``, plus
      ``D / Close(t-1) / s`` for the cash.
    * ``unexplained``: no registered action fits. Yahoo's value is kept and the day is listed.

    A company that began to trade shortly before the day (``children_close``: adjusted closes of
    candidates, their splits in ``splits``) is the *child* of an in-kind distribution if the
    amount is a simple fraction of its last quote before the day, which is how Yahoo valued it
    (EXPE 15.125 = 1/2 x 30.25 of TRIP). The amount is then revalued at the child's close on
    the day (``child_exact``): the holder can sell the shares only at that price. A company that
    merely starts on the day is reported without being used (DHR / FTV: ``child_n`` 0.505).

    ``scale_before`` is what makes the corrected day return come out: every price dated before the
    day is multiplied by it, prices from the day on are untouched, every other return stays as
    it was (:func:`apply_return_corrections`; :func:`undo_return_corrections` is its inverse).
    The corrected return of a day reads only data dated on or before it. ``plausible`` is False
    where the corrected return is still beyond ``plausible``: the registered amounts and the
    price then do not fit together and nothing in the data says which one is wrong.

    Days below ``threshold`` are left to Yahoo; :func:`uncorrected_dividend_error` bounds that.
    """
    close = close_unadj.reindex_like(prices_adj)
    gaps = adjustment_gaps(prices_adj, close, threshold=threshold).sort_values(
        ["ticker", "date"], kind="stable"
    )
    paid = dividends.groupby(["ticker", "date"])["amount"].sum().to_dict() if len(dividends) else {}
    ratios = splits.groupby(["ticker", "date"])["ratio"].prod().to_dict() if len(splits) else {}
    splits_by = {t: g for t, g in splits.groupby("ticker")} if len(splits) else {}
    first_valid = None
    if children_close is not None and children_close.shape[1]:
        first_valid = pd.to_datetime(children_close.apply(lambda col: col.first_valid_index()))
    index = prices_adj.index

    def child_of(day, ticker, amounts):
        if first_valid is None:
            return None
        return _confirm_child(day, amounts, ticker, children_close, first_valid, splits_by)

    rows = []
    for gap in gaps.itertuples():
        ticker, day = gap.ticker, gap.date
        pos = index.get_loc(day)
        prev_day = index[pos - 1]
        cp, ct = float(close.at[prev_day, ticker]), float(close.at[day, ticker])
        r_close = ct / cp - 1.0
        r_yahoo = float(prices_adj.at[day, ticker] / prices_adj.at[prev_day, ticker] - 1.0)
        amount = float(paid.get((ticker, day), 0.0))
        ratio = float(ratios.get((ticker, day), 0.0))
        later = _later_factor(splits_by, ticker, day)
        residual = r_yahoo - (ct / (cp - amount) - 1.0) if 0.0 < amount < cp else np.nan

        kind, source, r_star, child = _classify_gap(
            cp, ct, amount, ratio, later, partial(child_of, day, ticker)
        )
        if not (np.isfinite(r_star) and r_star > -1.0):  # unexplained, or amounts that cannot be
            kind, source, r_star, child = "unexplained", "none", r_yahoo, None
        loose = None
        if child is None and first_valid is not None and kind.startswith(("pseudo", "dividend_with")):
            loose = _starting_child(day, amount, ticker, children_close, first_valid, splits_by)

        vol_ratio = np.nan
        if volume is not None and ticker in volume.columns:
            series = volume[ticker].reindex(index)
            base = series.iloc[max(0, pos - 60) : pos].median()
            if base > 0:
                vol_ratio = float(series.iloc[pos] / base)
        rows.append(
            {
                "ticker": ticker, "date": day, "previous_date": prev_day, "kind": kind,
                "value_source": source, "close_previous": cp, "close_event": ct,
                "dividend": amount, "dividend_over_close": amount / cp, "split_ratio": ratio,
                "later_split_factor": later, "volume_ratio": vol_ratio, "return_close": r_close,
                "return_yahoo": r_yahoo, "adj_formula_residual": residual,
                "return_corrected": r_star, "change": r_star - r_yahoo,
                "scale_before": 1.0 if kind == "unexplained" else (1.0 + r_yahoo) / (1.0 + r_star),
                "child_ticker": child["ticker"] if child else (loose[0] if loose else None),
                "child_n": child["n"] if child else (loose[1] if loose else np.nan),
                "child_exact": child is not None,
                "plausible": bool(abs(r_star) <= plausible),
                "note": _NOTES[kind],
            }
        )
    table = pd.DataFrame(rows, columns=CORRECTION_COLUMNS)
    for col in ("date", "previous_date"):
        table[col] = pd.to_datetime(table[col])
    for col in ("child_exact", "plausible"):
        table[col] = table[col].astype(bool)
    return AdjustmentCorrections(apply_return_corrections(prices_adj, table), table)


def _rescale(prices_adj: pd.DataFrame, table: pd.DataFrame | None, inverse: bool) -> pd.DataFrame:
    out = prices_adj.copy()
    if table is None or len(table) == 0:
        return out
    for ticker, rows in table.groupby("ticker", sort=True):
        if ticker not in out.columns:
            continue
        scale = np.ones(len(out))
        for row in rows.itertuples():
            scale[out.index < row.date] *= row.scale_before
        values = out[ticker].to_numpy()
        out[ticker] = values / scale if inverse else values * scale
    return out


def apply_return_corrections(prices_adj: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Multiply every price dated before a corrected day by that day's ``scale_before``."""
    return _rescale(prices_adj, table, inverse=False)


def undo_return_corrections(prices_adj: pd.DataFrame, table: pd.DataFrame | None) -> pd.DataFrame:
    """Yahoo's own ``Adj Close`` back from a corrected panel and its corrections table."""
    return _rescale(prices_adj, table, inverse=True)


# --------------------------------------------------------------------------------------------
# Verified spin-off days (adjcorr-2): the holder's total return from the terms in the filings
# --------------------------------------------------------------------------------------------

VERIFIED_KIND = "verified_spin_off"
_UNLISTED_WINDOW = pd.Timedelta(days=5)  # a declared split this close to a listed one is the same


@dataclass(frozen=True)
class VerifiedCorrections:
    """``prices_adj`` with the verified days corrected, and what is needed to document them.

    ``table`` has one row per applied event in the columns of the gap corrections, ``records`` one
    JSON-able dict per applied event (terms, sources, every number used, the price checks) and
    ``skipped`` maps ``"TICKER@DATE"`` to the reason an event does not belong to this panel.
    """

    prices_adj: pd.DataFrame
    table: pd.DataFrame
    records: list[dict]
    skipped: dict[str, str]


def _unlisted_factor(unlisted: Sequence[tuple[str, float]], day: pd.Timestamp) -> float:
    """Product of the ratios declared as unlisted (Yahoo applied them, ``Stock Splits`` does not)
    that are dated after ``day``."""
    factor = 1.0
    for when, ratio in unlisted:
        if pd.Timestamp(when) > day:
            factor *= float(ratio)
    return factor


def _refuse_double_count(
    key: str, listed: pd.DataFrame | None, unlisted: Sequence[tuple[str, float]], who: str
) -> None:
    for when, ratio in unlisted:
        if listed is None or not len(listed):
            continue
        near = listed[(listed["date"] - pd.Timestamp(when)).abs() <= _UNLISTED_WINDOW]
        if ((near["ratio"] / ratio - 1.0).abs() < 1e-6).any():
            raise ValueError(
                f"{key}: the split of {who} on {when} (ratio {ratio:.6g}) is declared as unlisted "
                "but the data list it: it would be counted twice"
            )


def apply_verified_events(
    prices_adj: pd.DataFrame,
    close_unadj: pd.DataFrame,
    splits: pd.DataFrame,
    events: Iterable[VerifiedEvent],
    *,
    children_close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    corrected: Iterable[tuple[str, pd.Timestamp]] = (),
    plausible: float = PLAUSIBLE_DAY_RETURN,
) -> VerifiedCorrections:
    """Replace Yahoo's return on a verified spin-off day by the holder's total return.

    A registered event (:mod:`cif.data.sp500_events`: terms and sources as read in the company's
    filings) says how many shares of the parent (``k``) and of each distributed company (``n_i``)
    one old parent share has become. With the as-traded closes ``P0`` of the parent the day before,
    ``P1`` on the day and ``C_i`` of the distributed companies on the day,

        ``total return = (k * P1 + sum_i n_i * C_i) / P0 - 1``.

    As-traded means ``Close`` times every split that Yahoo divided it by: the ratios of
    ``splits`` dated after the date and the ratios the event declares as unlisted (applied by
    Yahoo to every earlier close but absent from the cached window). The result needs no price
    dated after the day: later splits only fix the units. Every ``PriceCheck`` of the event must
    hold, else the units of a series are not the ones the event was verified in and
    ``ValueError`` stops the build; so does a ratio on the day that is not the one the event was
    verified against, a day the gap rule already corrected (``corrected``), a missing price, and
    an unlisted split that the data list as well.

    A distributed company without a Yahoo series is priced at the close (or the volume weighted
    average) that the event documents. ``children_close`` holds the ``Close`` of the others.
    ``scale_before`` and the table are those of :func:`correct_adjustment_artifacts`: prices
    dated before the day are multiplied by it, every other return stays as it was.

    Events whose ticker or day is not in the panel are not applied and are returned in
    ``skipped`` (a build over another window or with ``--limit-tickers``).
    """
    close = close_unadj.reindex_like(prices_adj)
    index = prices_adj.index
    splits_by = {t: g for t, g in splits.groupby("ticker")} if len(splits) else {}
    booked = splits.groupby(["ticker", "date"])["ratio"].prod().to_dict() if len(splits) else {}
    done = set(corrected)
    rows: list[dict] = []
    records: list[dict] = []
    skipped: dict[str, str] = {}

    def as_traded(frame: pd.DataFrame, ticker: str, day: pd.Timestamp, unlisted=()) -> float:
        value = frame.at[day, ticker] if ticker in frame.columns and day in frame.index else np.nan
        return float(value) * _later_factor(splits_by, ticker, day) * _unlisted_factor(unlisted, day)

    for event in sorted(events, key=lambda e: (e.ticker, e.date)):
        key = f"{event.ticker}@{event.date}"
        ticker, day = event.ticker, pd.Timestamp(event.date)
        if ticker not in prices_adj.columns:
            skipped[key] = "the ticker is not in the panel"
            continue
        if day not in index or index.get_loc(day) == 0:
            skipped[key] = "the day, or the day before it, is not in the panel"
            continue
        prev_day = index[index.get_loc(day) - 1]
        if (ticker, day) in done:
            raise ValueError(f"{key}: the day is already corrected by the gap rule")
        registered = booked.get((ticker, day))
        if registered is None or not math.isclose(registered, event.yahoo_ratio, rel_tol=1e-9):
            raise ValueError(
                f"{key}: the ratio Yahoo registers on the day is {registered}, the event was "
                f"verified against {event.yahoo_ratio}"
            )
        _refuse_double_count(key, splits_by.get(ticker), event.later_splits_unlisted, ticker)
        for d in event.distributions:
            _refuse_double_count(key, splits_by.get(d.ticker), d.later_splits_unlisted, d.ticker)
        cp, ct = float(close.at[prev_day, ticker]), float(close.at[day, ticker])
        adj_prev, adj_day = float(prices_adj.at[prev_day, ticker]), float(prices_adj.at[day, ticker])
        if not all(math.isfinite(x) and x > 0 for x in (cp, ct, adj_prev, adj_day)):
            raise ValueError(f"{key}: the parent has no price on the day or the day before")

        p0 = as_traded(close, ticker, prev_day, event.later_splits_unlisted)
        p1 = as_traded(close, ticker, day, event.later_splits_unlisted)
        children = []
        for d in event.distributions:
            if d.price == "documented":
                value, source = float(d.documented_close), "documented"
            else:
                if children_close is None or d.ticker not in children_close.columns or not (
                    day in children_close.index and pd.notna(children_close.at[day, d.ticker])
                ):
                    raise ValueError(f"{key}: the child {d.ticker} has no close on the day")
                value = as_traded(children_close, d.ticker, day, d.later_splits_unlisted)
                source = "yahoo_close"
            children.append(
                {
                    "ticker": d.ticker, "name": d.name, "shares_per_parent_share": d.shares_per_share,
                    "close": value, "source": source,
                    "basis": "close" if source == "yahoo_close" else d.documented_basis,
                    "url": d.documented_source,
                }
            )

        checks = []
        known = {c["ticker"]: c for c in children}
        for check in event.price_checks:
            when = pd.Timestamp(check.date)
            if check.ticker == ticker:
                value = as_traded(close, ticker, when, event.later_splits_unlisted)
            elif check.ticker in known and known[check.ticker]["source"] == "documented":
                value = known[check.ticker]["close"] if when == day else math.nan
            elif check.ticker in known:
                unlisted = next(
                    d.later_splits_unlisted for d in event.distributions if d.ticker == check.ticker
                )
                value = (
                    as_traded(children_close, check.ticker, when, unlisted)
                    if children_close is not None
                    else math.nan
                )
            else:
                raise ValueError(f"{key}: price check on {check.ticker}, not a company of the event")
            if not math.isfinite(value) or not check.low <= value <= check.high:
                raise ValueError(
                    f"{key}: price check failed: {check.ticker} on {check.date} is {value:.6g}, "
                    f"expected in [{check.low}, {check.high}] ({check.basis}); the units of the "
                    "series are not the ones the event was verified in"
                )
            checks.append(
                {"ticker": check.ticker, "date": check.date, "low": check.low, "high": check.high,
                 "value": round(value, 6), "basis": check.basis, "source": check.source, "ok": True}
            )

        total = event.parent_shares_per_share * p1 + sum(
            c["shares_per_parent_share"] * c["close"] for c in children
        )
        r_star = total / p0 - 1.0
        r_close, r_yahoo = ct / cp - 1.0, adj_day / adj_prev - 1.0
        vol_ratio = np.nan
        if volume is not None and ticker in volume.columns:
            series = volume[ticker].reindex(index)
            pos = index.get_loc(day)
            base = series.iloc[max(0, pos - 60) : pos].median()
            if base > 0:
                vol_ratio = float(series.iloc[pos] / base)
        used = {"child_close" if c["source"] == "yahoo_close" else "documented_price" for c in children}
        rows.append(
            {
                "ticker": ticker, "date": day, "previous_date": prev_day, "kind": VERIFIED_KIND,
                "value_source": "sec_terms+"
                + "+".join(s for s in ("child_close", "documented_price") if s in used),
                "close_previous": cp, "close_event": ct, "dividend": 0.0, "dividend_over_close": 0.0,
                "split_ratio": float(registered),
                "later_split_factor": _later_factor(splits_by, ticker, day)
                * _unlisted_factor(event.later_splits_unlisted, day),
                "volume_ratio": vol_ratio, "return_close": r_close, "return_yahoo": r_yahoo,
                "adj_formula_residual": np.nan, "return_corrected": r_star,
                "change": r_star - r_yahoo, "scale_before": (1.0 + r_yahoo) / (1.0 + r_star),
                "child_ticker": "+".join(c["ticker"] for c in children),
                "child_n": children[0]["shares_per_parent_share"] if len(children) == 1 else np.nan,
                "child_exact": all(c["basis"] == "close" for c in children),
                "plausible": bool(abs(r_star) <= plausible), "note": _NOTES[VERIFIED_KIND],
            }
        )
        records.append(
            {
                "ticker": ticker, "date": event.date, "previous_date": str(prev_day.date()),
                "verdict": "verified spin-off: the day return of prices_adj is the holder's total "
                "return (adjcorr-2)",
                "terms": {
                    "parent": event.name, "kind": event.kind,
                    "parent_shares_per_share": event.parent_shares_per_share,
                    "distributions": [
                        {"ticker": d.ticker, "name": d.name, "shares_per_parent_share": d.shares_per_share}
                        for d in event.distributions
                    ],
                    "record_date": event.record_date, "distribution_date": event.distribution_date,
                    "ex_date": event.date, "yahoo_ratio": event.yahoo_ratio,
                    "later_splits_unlisted": [[w, r] for w, r in event.later_splits_unlisted],
                    "note": event.note,
                },
                "sources": [
                    {"url": s.url, "document": s.document, "supports": list(s.supports), "kind": s.kind}
                    for s in event.sources
                ],
                "prices": {
                    "parent_close_previous": round(p0, 6), "parent_close_event": round(p1, 6),
                    "children": [{**c, "close": round(c["close"], 6)} for c in children],
                    "unit": "as-traded close: Yahoo's Close times the splits dated after the date",
                },
                "returns": {
                    "yahoo_adjusted": r_yahoo, "close_panel": r_close, "corrected": r_star,
                    "change": r_star - r_yahoo,
                },
                "price_checks": checks,
                "verified_on": event.verified_on,
            }
        )
    table = pd.DataFrame(rows, columns=CORRECTION_COLUMNS)
    for col in ("date", "previous_date"):
        table[col] = pd.to_datetime(table[col])
    for col in ("child_exact", "plausible"):
        table[col] = table[col].astype(bool)
    return VerifiedCorrections(apply_return_corrections(prices_adj, table), table, records, skipped)


def pseudo_split_inventory(
    prices_adj: pd.DataFrame,
    close_unadj: pd.DataFrame,
    splits: pd.DataFrame,
    corrected: Iterable[tuple[str, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Days with a ratio that is no share-count ratio, and what they would distort. Nothing changes.

    If the ratio books a spin-off (see :func:`correct_adjustment_artifacts`), Yahoo's Close return
    on the day is ``ratio`` times the total return; without a dividend on the day Adj Close equals
    Close and :func:`adjustment_gaps` does not list it. AIV 2020-12-15: +16.1 % in the data, +1.7 %
    under that reading (the verified holder's return is +6.9 %: the model prices the distributed
    shares at the value Yahoo took out, not at their close on the day). ``return_if_value_ratio``
    is that estimate and ``distortion`` its difference to the Close return. Nothing here is
    verified: the ratio can also be a share change with an inexact ratio (GOOG 2014-03-27), and
    ``value_model_applies`` is False for ratios below one, which can be a real reverse split mixed
    with a spin-off (DD 2019-06-03, MSI 2011-01-04).
    ``corrected`` marks the days that :func:`correct_adjustment_artifacts` or, once verified,
    :func:`apply_verified_events` already handled. Sorted by ``|distortion|``.
    """
    done = set(corrected or ())
    ret_adj = prices_adj.pct_change(fill_method=None)
    ret_close = close_unadj.reindex_like(prices_adj).pct_change(fill_method=None)
    rows = []
    for row in splits.itertuples():
        if is_share_count_ratio(row.ratio) or row.ticker not in ret_close.columns:
            continue
        if row.date not in ret_close.index or pd.isna(ret_close.at[row.date, row.ticker]):
            continue
        r_close = float(ret_close.at[row.date, row.ticker])
        implied = r_close / row.ratio
        rows.append(
            {
                "ticker": row.ticker, "date": row.date, "ratio": float(row.ratio),
                "return_close": r_close, "return_adjusted": float(ret_adj.at[row.date, row.ticker]),
                "return_if_value_ratio": implied, "distortion": r_close - implied,
                "corrected": (row.ticker, row.date) in done, "value_model_applies": row.ratio > 1.0,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=["ticker", "date", "ratio", "return_close", "return_adjusted",
                 "return_if_value_ratio", "distortion", "corrected", "value_model_applies"],
    )
    out["date"] = pd.to_datetime(out["date"])
    order = out["distortion"].abs().sort_values(ascending=False, kind="stable").index
    return out.reindex(order).reset_index(drop=True)


def uncorrected_dividend_error(
    prices_adj: pd.DataFrame,
    close_unadj: pd.DataFrame,
    dividends: pd.DataFrame,
    splits: pd.DataFrame,
    flagged: Iterable[tuple[str, pd.Timestamp]],
) -> dict:
    """How far Yahoo's day return is from ``(Close + D) / Close(t-1) - 1`` on the dividend days
    that :func:`correct_adjustment_artifacts` leaves alone (``flagged`` are the corrected ones).

    Days that also carry a split are left out (the formula does not apply). The bound is what the
    choice of ``threshold`` costs: the error is ``d`` times the day's return, small for a small
    dividend ``d``.
    """
    skip = set(flagged) | set(zip(splits["ticker"], splits["date"], strict=True))
    close = close_unadj.reindex_like(prices_adj)
    ret = prices_adj.pct_change(fill_method=None)
    prev = close.shift(1)
    errors = []
    daily = dividends.groupby(["ticker", "date"], as_index=False)["amount"].sum()
    for row in daily.itertuples():
        if (row.ticker, row.date) in skip or row.ticker not in close.columns:
            continue
        if row.date not in close.index:
            continue
        cp, ct = prev.at[row.date, row.ticker], close.at[row.date, row.ticker]
        ry = ret.at[row.date, row.ticker]
        if not (np.isfinite(cp) and np.isfinite(ct) and np.isfinite(ry)) or cp <= 0:
            continue
        errors.append(abs(ry - ((ct + row.amount) / cp - 1.0)))
    err = np.asarray(errors)
    return {
        "n_dividend_days": int(err.size),
        "max_abs_error": float(err.max()) if err.size else 0.0,
        "share_above_0.001": float((err > 0.001).mean()) if err.size else 0.0,
        "share_above_0.005": float((err > 0.005).mean()) if err.size else 0.0,
    }


@dataclass(frozen=True)
class MaskedPanel:
    """Result of :func:`mask_implausible_series`: the three price frames and what was removed."""

    adj_close: pd.DataFrame
    close_unadj: pd.DataFrame
    volume: pd.DataFrame
    report: pd.DataFrame  # ticker, first_masked, last_masked, n_member_days_masked, median_dollar_volume
    dropped: dict[str, str]


def mask_implausible_series(
    adj_close: pd.DataFrame,
    close_unadj: pd.DataFrame,
    volume: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    window: int = 21,
    min_dollar_volume: float = 1e6,
) -> MaskedPanel:
    """Blank the member days on which a Yahoo series cannot be the index member.

    Yahoo serves the *current* holder of a symbol with that holder's whole history. For a ticker
    that an index member gave up and a shell company took (CPWR, EP, GENZ, SW ...) the series
    therefore covers the period of membership with prices of an unrelated, nearly untraded
    stock. A member of the S&P 500 trades tens of millions of dollars a day, so a member day on
    which the trailing ``window``-day median of ``close_unadj * volume`` is below
    ``min_dollar_volume`` (default 1 million) is not that member and its three values are set to
    NaN. The window is trailing: a day is judged by data up to itself, so no future volume
    enters the mask (the first days of a real listing after a shell stay blank until the median
    turns; a spin-off's when-issued days can cost its first ~10 days). Days with no volume at all
    are not judged. Days outside membership are not judged or touched. A column left with no
    valid member-day price, whether masked or because the series lies entirely outside the
    membership spells of that symbol, is dropped and listed in ``dropped``.
    """
    index, columns = adj_close.index, adj_close.columns
    member = membership.reindex(index=index, columns=columns, fill_value=False).astype(bool)
    dollar = close_unadj.reindex(index=index, columns=columns) * volume.reindex(
        index=index, columns=columns
    )
    typical = dollar.rolling(window, min_periods=1).median()
    bad = member & (typical < min_dollar_volume)

    rows = []
    for ticker in columns[bad.any(axis=0).to_numpy()]:
        days = index[bad[ticker].to_numpy()]
        rows.append(
            {
                "ticker": ticker,
                "first_masked": days[0],
                "last_masked": days[-1],
                "n_member_days_masked": len(days),
                "median_dollar_volume": float(dollar.loc[days, ticker].median()),
            }
        )
    report = pd.DataFrame(
        rows,
        columns=["ticker", "first_masked", "last_masked", "n_member_days_masked",
                 "median_dollar_volume"],
    )
    masked = [frame.reindex(index=index, columns=columns).mask(bad)
              for frame in (adj_close, close_unadj, volume)]
    left = (member & masked[0].notna()).any(axis=0)
    dropped = {}
    for ticker in columns[(~left).to_numpy()]:
        dropped[str(ticker)] = (
            f"series is not the index member (median dollar volume on member days below "
            f"{min_dollar_volume:,.0f})" if bad[ticker].any()
            else "series lies outside the membership spells (the symbol belongs to another company)"
        )
    keep = [t for t in columns if t not in dropped]
    return MaskedPanel(masked[0][keep], masked[1][keep], masked[2][keep], report, dropped)


# --------------------------------------------------------------------------------------------
# The universe on a formation date
# --------------------------------------------------------------------------------------------


def _row_on_or_before(frame: pd.DataFrame, date: pd.Timestamp, what: str) -> pd.Series:
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{what} must be sorted by date")
    upto = frame.loc[:date]
    if upto.empty:
        raise ValueError(f"{what} has no row on or before {date.date()}")
    return upto.iloc[-1]


def universe_candidates(
    formation_date: str | pd.Timestamp,
    membership: pd.DataFrame,
    prices: pd.DataFrame,
    close_unadj: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    lookback: int = 756,
    liquidity_window: int = 252,
    min_history_frac: float = 0.95,
    companies: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """All eligible tickers on ``formation_date`` ranked by median dollar volume (best first).

    Columns: ``valid_frac`` (share of valid prices in the last ``lookback`` trading days),
    ``median_dollar_volume`` (median of unadjusted close x volume over the last
    ``liquidity_window`` days), ``n_dollar_volume_obs``, ``company_id``, ``rank``. Read the
    docstring of :func:`select_universe` for the rules; nothing dated after ``formation_date``
    is read.
    """
    date = pd.Timestamp(formation_date).normalize()
    for frame, name in ((prices, "prices"), (close_unadj, "close_unadj"), (volume, "volume")):
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} must be sorted by date")
    history = prices.loc[:date]
    if len(history) < lookback:
        raise ValueError(
            f"lookback={lookback} trading days needed, only {len(history)} rows up to {date.date()}"
        )
    window = history.iloc[-lookback:]
    liq_index = window.index[-liquidity_window:]

    row = _row_on_or_before(membership, date, "membership")
    members = [t for t in row.index[row.to_numpy(dtype=bool)] if t in window.columns]

    valid_frac = (window[members] > 0).mean()
    dollar = (
        close_unadj.reindex(index=liq_index, columns=members)
        * volume.reindex(index=liq_index, columns=members)
    )
    table = pd.DataFrame(
        {
            "valid_frac": valid_frac,
            "median_dollar_volume": dollar.median(skipna=True),
            "n_dollar_volume_obs": dollar.notna().sum(),
        }
    )
    need_obs = math.ceil(min_history_frac * len(liq_index) - _EPS)
    keep = (
        (table["valid_frac"] >= min_history_frac - _EPS)
        & (table["n_dollar_volume_obs"] >= need_obs)
        & table["median_dollar_volume"].notna()
    )
    table = table[keep].copy()
    table = table.assign(_name=table.index).sort_values(
        ["median_dollar_volume", "_name"], ascending=[False, True], kind="stable"
    ).drop(columns="_name")

    if companies is not None:
        ids = (
            dict(zip(companies["ticker"], companies["company_id"], strict=True))
            if "company_id" in companies else assign_company_ids(companies)
        )
    else:
        ids = {}
    table["company_id"] = [ids.get(t, f"ticker:{t}") for t in table.index]
    if companies is not None:
        table = table[~table["company_id"].duplicated(keep="first")].copy()
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def select_universe(
    formation_date: str | pd.Timestamp,
    membership: pd.DataFrame,
    prices: pd.DataFrame,
    close_unadj: pd.DataFrame,
    volume: pd.DataFrame,
    n: int = 100,
    lookback: int = 756,
    liquidity_window: int = 252,
    min_history_frac: float = 0.95,
    companies: pd.DataFrame | None = None,
) -> list[str]:
    """Top-``n`` index members on ``formation_date`` by median dollar volume, no look-ahead.

    1. Members: the tickers that are members on ``formation_date`` (the last row of
       ``membership`` on or before it).
    2. History: at least ``min_history_frac`` of the last ``lookback`` trading days up to and
       including ``formation_date`` carry a valid (finite, positive) adjusted price.
    3. Liquidity: rank by the median over the last ``liquidity_window`` trading days of
       ``close_unadj * volume`` (unadjusted close, so that dividend adjustments do not tilt it).
    4. Share classes: with ``companies`` (``ticker``, ``security``, optionally ``cik``) one
       ticker per company is kept, the more liquid one, so the result holds ``n`` companies.
       Without it every ticker counts on its own.

    Only rows dated on or before ``formation_date`` are read from any input, so appending or
    rewriting later data cannot change the answer. Fewer than ``n`` tickers are returned if fewer
    qualify. Raises ``ValueError`` if ``prices`` has fewer than ``lookback`` rows up to the date.

    A selected series may end after the formation date (acquisition, delisting); what to assume
    for the rest of the holding period is a decision of the backtest, not of the selection.
    Members for which no series could be downloaded are invisible here; ``compute_coverage``
    reports how many there are.
    """
    table = universe_candidates(
        formation_date, membership, prices, close_unadj, volume,
        lookback=lookback, liquidity_window=liquidity_window,
        min_history_frac=min_history_frac, companies=companies,
    )
    if len(table) < n:
        logger.warning("only %d of the requested %d tickers qualify on %s", len(table), n,
                       pd.Timestamp(formation_date).date())
    return list(table.index[:n])


# --------------------------------------------------------------------------------------------
# Is a large one-day move a corporate action or a market move?
# --------------------------------------------------------------------------------------------


def jump_evidence(
    history: pd.DataFrame,
    event_date: str | pd.Timestamp,
    *,
    window_days: int = 5,
    baseline_days: int = 60,
    after_days: int = 10,
    spike_ratio: float = 3.0,
) -> dict:
    """Evidence on a one-day price jump from a Yahoo history (``auto_adjust=False``, actions).

    ``history`` needs ``Close``, ``Volume`` and optionally ``Adj Close`` and ``Stock Splits``.
    The verdict is rule based and deliberately conservative:

    * ``"corporate action (split)"``: Yahoo lists a split within ``window_days`` calendar days
      whose ratio explains the move (ratio x (1 + return) within 5% of 1);
    * ``"real market move"``: no split, volume on the day at least ``spike_ratio`` times its
      median over the previous ``baseline_days`` days, and the new price level holds
      (median close of the next ``after_days`` days within 0.7-1.4 of the day's close);
    * ``"inconclusive"`` otherwise.
    """
    date = pd.Timestamp(event_date)
    if date not in history.index:
        raise ValueError(f"{date.date()} is not a trading day in the history")
    pos = history.index.get_loc(date)
    if pos == 0:
        raise ValueError("no previous day in the history")
    close = history["Close"]
    ret = float(close.iloc[pos] / close.iloc[pos - 1] - 1.0)
    out: dict = {
        "date": str(date.date()),
        "previous_date": str(history.index[pos - 1].date()),
        "close_previous": float(close.iloc[pos - 1]),
        "close_event": float(close.iloc[pos]),
        "return_unadjusted": ret,
    }
    if "Adj Close" in history:
        adj = history["Adj Close"]
        out["return_adjusted"] = float(adj.iloc[pos] / adj.iloc[pos - 1] - 1.0)

    vol = history["Volume"].astype(float)
    base = vol.iloc[max(0, pos - baseline_days) : pos]
    out["volume_event"] = float(vol.iloc[pos])
    out["volume_previous"] = float(vol.iloc[pos - 1])
    out["volume_baseline_median"] = float(base.median())
    out["volume_ratio"] = float(vol.iloc[pos] / base.median()) if base.median() > 0 else float("nan")

    after = close.iloc[pos + 1 : pos + 1 + after_days]
    out["persistence"] = float(after.median() / close.iloc[pos]) if len(after) else float("nan")

    splits: list[tuple[pd.Timestamp, float]] = []
    if "Stock Splits" in history:
        near = history["Stock Splits"].fillna(0.0)
        near = near[(near.index >= date - pd.Timedelta(days=window_days))
                    & (near.index <= date + pd.Timedelta(days=window_days)) & (near != 0)]
        splits = [(day, float(value)) for day, value in near.items()]
    out["splits_in_window"] = splits

    explains = any(abs(ratio * (1.0 + ret) - 1.0) <= 0.05 for _, ratio in splits)
    held = 0.7 <= out["persistence"] <= 1.4
    if splits and explains:
        out["verdict"] = "corporate action (split)"
    elif not splits and out["volume_ratio"] >= spike_ratio and held:
        out["verdict"] = "real market move"
    else:
        out["verdict"] = "inconclusive"
    return out


# --------------------------------------------------------------------------------------------
# Dataset on disk
# --------------------------------------------------------------------------------------------


@dataclass
class SP500PIT:
    """The processed dataset; :meth:`select` is :func:`select_universe` on it.

    ``prices_adj`` is Yahoo's adjusted close with the days of :func:`correct_adjustment_artifacts`
    and :func:`apply_verified_events` corrected; ``corrections`` lists them (``None`` for a dataset
    built without the correction) and :meth:`prices_adj_yahoo` gives Yahoo's panel back.
    """

    membership: pd.DataFrame
    prices_adj: pd.DataFrame
    close_unadj: pd.DataFrame
    volume: pd.DataFrame
    coverage: pd.DataFrame
    companies: pd.DataFrame
    failed: dict[str, str]
    corrections: pd.DataFrame | None = None

    def prices_adj_yahoo(self) -> pd.DataFrame:
        return undo_return_corrections(self.prices_adj, self.corrections)

    def candidates(self, formation_date: str | pd.Timestamp, **kwargs) -> pd.DataFrame:
        return universe_candidates(
            formation_date, self.membership, self.prices_adj, self.close_unadj, self.volume,
            companies=self.companies, **kwargs,
        )

    def select(self, formation_date: str | pd.Timestamp, n: int = 100, **kwargs) -> list[str]:
        return select_universe(
            formation_date, self.membership, self.prices_adj, self.close_unadj, self.volume,
            n=n, companies=self.companies, **kwargs,
        )


_FRAMES = ("membership", "prices_adj", "close_unadj", "volume", "coverage", "companies")


def save_dataset(
    out_dir: Path,
    *,
    membership: pd.DataFrame,
    prices_adj: pd.DataFrame,
    close_unadj: pd.DataFrame,
    volume: pd.DataFrame,
    coverage: pd.DataFrame,
    companies: pd.DataFrame,
    failed: Mapping[str, str],
    extra: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Write the six parquet files, ``failed_tickers.json`` and optional ``extra`` frames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    companies = companies.copy()
    if "company_id" not in companies:
        companies["company_id"] = companies["ticker"].map(assign_company_ids(companies))
    frames = {
        "membership": membership, "prices_adj": prices_adj, "close_unadj": close_unadj,
        "volume": volume, "coverage": coverage, "companies": companies,
        **(dict(extra) if extra else {}),
    }
    written = {}
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path)
        written[name] = path
    failed_path = out_dir / "failed_tickers.json"
    failed_path.write_text(json.dumps(dict(sorted(failed.items())), indent=2, ensure_ascii=False))
    written["failed_tickers"] = failed_path
    return written


def load_dataset(out_dir: Path) -> SP500PIT:
    out_dir = Path(out_dir)
    frames = {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in _FRAMES}
    failed = json.loads((out_dir / "failed_tickers.json").read_text())
    path = out_dir / "adjustment_corrections.parquet"
    corrections = pd.read_parquet(path) if path.exists() else None
    return SP500PIT(failed=failed, corrections=corrections, **frames)
