"""MOEX ISS data layer for the Russian equity market (WP-E1, R6).

Replaces the legacy :func:`cif.data.providers.fetch_moex_prices` (TQBR ``CLOSE`` of *today's*
IMOEX members, price return, no trading calendar) with building blocks that make the survivorship
bias, the missing dividends and the 2022 closure explicit and testable:

1. :class:`IssClient` -- polite, retrying, **disk-cached** access to ``iss.moex.com``. Every table
   is stored verbatim (``columns`` + ``data``) under ``data/raw/moex_v2`` so the processed files
   can be rebuilt offline and every raw file can be hashed into ``provenance.json``.
2. Fetchers for the endpoints that are actually public (verified 2026-10-04):

   * ``statistics/engines/stock/markets/index/analytics/IMOEX?date=`` -- index composition and
     weights on a date, available from 2001;
   * ``.../analytics/IMOEX/tickers`` -- every ticker that has ever been in the index, with the
     first and last date; ``.../tickers/{ticker}?from=&till=`` -- daily ``issue_size_total``,
     free-float factor and weight of one member (used as evidence for share-count changes);
   * ``history/engines/stock/markets/shares/boardgroups/57/securities/{secid}`` -- daily
     ``CLOSE``/``VOLUME``/``VALUE`` on the T+ board group (TQBR, TQBS, TQNE ...), which keeps a
     ticker's history continuous when MOEX moved it between listing levels;
   * ``statistics/engines/stock/splits`` -- the register of splits and consolidations (starts in
     December 2018, so earlier events need other evidence);
   * ``history/engines/stock/markets/index/securities/{IMOEX,MCFTR,MCFTRR}`` -- index levels;
     the days on which IMOEX was calculated define the trading calendar.

   **Not public:** ``iss/securities/{secid}/dividends.json`` silently returns the security
   description (no ``dividends`` block), and ``iss/cci/corp-actions/dividends`` answers with an
   HTML page "available to subscribers only". :class:`IssClient` raises :class:`IssError` in both
   cases instead of returning an empty table. Dividends therefore enter through
   :func:`prepare_dividends` from a table the caller supplies (see its docstring).
3. Pure functions, tested offline: trading calendar, ex-dividend date, split adjustment,
   total-return index, jump detection and reconciliation with the split register, wide panels and
   the per-ticker "traded" mask.

Conventions
-----------
* Prices are the ``CLOSE`` of the board group, **unadjusted**; ``prices_adj`` corrects splits only;
  the total-return (TR) index also reinvests gross cash dividends on the ex-date. MCFTR (gross)
  is the matching benchmark.
* A day is *tradable* iff IMOEX was calculated from trades on it (positive turnover; see
  :func:`calculated_trading_days`). Panels are indexed by tradable days only, so no operation on
  them can bridge the closure of 28.02-23.03.2022.
* Board group 57 is the main market only from **2013-09-02** (``MAIN_MARKET_START``): before that
  its boards carried 5-15 % of the turnover of SBER, GAZP and LKOH and the main market was the
  EQBR/EQNE boards. The price history starts on that date.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ISS_BASE = "https://iss.moex.com/iss"
USER_AGENT = "cif-research-loader/0.1 (academic research; MOEX ISS public endpoints)"

#: Board group "T+: Akcii i DR" -- TQBR, TQBS, TQNE ... A ticker moves between these boards when
#: MOEX changes its listing level; the group keeps the history continuous.
BOARDGROUP_T_PLUS_SHARES = 57
#: First day on which the T+ board group is the main market for all shares.
MAIN_MARKET_START = pd.Timestamp("2013-09-02")
#: First trade date settled T+1 (equities, Moscow Exchange). Earlier trades settled T+2.
T_PLUS_ONE_START = pd.Timestamp("2023-07-31")

HISTORY_COLUMNS = (
    "BOARDID,TRADEDATE,SECID,NUMTRADES,VALUE,OPEN,LOW,HIGH,LEGALCLOSEPRICE,WAPRICE,CLOSE,VOLUME,"
    "CURRENCYID"
)


class IssError(RuntimeError):
    """The ISS answered, but not with the table that was asked for."""


class _Retryable(Exception):
    pass


# --------------------------------------------------------------------------------------------
# ISS client with a verbatim disk cache
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawTable:
    """One ISS table (all pages concatenated) exactly as served."""

    columns: list[str]
    rows: list[list]
    request: dict
    fetched_at: str
    pages: int
    from_cache: bool = False
    urls: list[str] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


class IssClient:
    """Rate-limited, retrying ISS client; ``table`` caches whole tables on disk.

    Parameters
    ----------
    cache_dir:
        Root of the verbatim cache (``data/raw/moex_v2``); ``None`` disables caching.
    pause:
        Seconds to sleep after every real HTTP request (politeness towards the free API).
    retries, backoff:
        Attempts per request and base of the exponential back-off (seconds ** attempt).
    refresh:
        Ignore cached tables and download again.
    offline:
        Never touch the network: a table missing from the cache raises :class:`IssError`. The
        ``build`` stage runs offline so that the processed files depend on the cache alone.
    sleep:
        Injectable for tests.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        *,
        pause: float = 0.25,
        retries: int = 5,
        backoff: float = 2.0,
        timeout: float = 60.0,
        session: requests.Session | None = None,
        refresh: bool = False,
        offline: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.pause = pause
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.refresh = refresh
        self.offline = offline
        self._sleep = sleep
        self.headers = {"User-Agent": USER_AGENT}
        self.n_requests = 0

    # -- one HTTP request ------------------------------------------------------------------

    def get_json(self, path: str, params: dict | None = None) -> dict:
        url = f"{ISS_BASE}/{path.strip('/')}.json"
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(
                    url, params=params, headers=self.headers, timeout=self.timeout
                )
                self.n_requests += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _Retryable(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                try:
                    payload = resp.json()
                except ValueError as exc:
                    raise _Retryable("response is not JSON (HTML page?)") from exc
                self._sleep(self.pause)
                return payload
            except (requests.RequestException, _Retryable) as exc:
                last = exc
                logger.warning("ISS %s attempt %d/%d failed: %s", url, attempt, self.retries, exc)
                if isinstance(exc, requests.HTTPError):
                    break  # 4xx other than 429: retrying will not help
                if attempt < self.retries:
                    self._sleep(self.backoff**attempt)
        raise IssError(f"{url} params={params}: {last}")

    # -- a whole table, cached -------------------------------------------------------------

    def table(
        self, path: str, block: str, params: dict | None = None, *, cache_name: str | None = None
    ) -> RawTable:
        params = dict(params or {})
        request = {"path": path, "block": block, "params": params}
        cache_file = self._cache_file(request, cache_name)
        if cache_file is not None and cache_file.exists() and not self.refresh:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached["request"] != request:
                raise IssError(f"cache {cache_file} holds another request: {cached['request']}")
            return RawTable(
                columns=cached["columns"], rows=cached["data"], request=request,
                fetched_at=cached["fetched_at"], pages=cached["pages"], from_cache=True,
                urls=cached.get("urls", []),
            )

        if self.offline:
            raise IssError(f"offline mode: {path} {params} is not in the cache ({cache_file})")
        columns: list[str] | None = None
        rows: list[list] = []
        urls: list[str] = []
        start, pages = 0, 0
        while True:
            payload = self.get_json(
                path, {**params, "iss.meta": "off", "iss.only": f"{block},{block}.cursor",
                       "start": start},
            )
            if block not in payload:
                raise IssError(
                    f"{path}: no block {block!r} in the answer (blocks: {sorted(payload)})"
                )
            part = payload[block]
            columns = part["columns"]
            page = part["data"]
            urls.append(f"{ISS_BASE}/{path.strip('/')}.json?start={start}")
            if page and rows and page == rows[-len(page):]:
                break  # endpoint ignores ``start`` and repeats itself
            pages += 1
            rows.extend(page)
            start += len(page)
            total = self._cursor_total(payload.get(f"{block}.cursor"))
            if not page or (total is not None and start >= total):
                break
        assert columns is not None
        table = RawTable(columns=columns, rows=rows, request=request,
                         fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
                         pages=pages, urls=urls)
        if cache_file is not None:
            self._write_cache(cache_file, table)
        return table

    @staticmethod
    def _cursor_total(cursor: dict | None) -> int | None:
        if not cursor or not cursor.get("data"):
            return None
        cols = [c.upper() for c in cursor["columns"]]
        return int(cursor["data"][0][cols.index("TOTAL")]) if "TOTAL" in cols else None

    def _cache_file(self, request: dict, cache_name: str | None) -> Path | None:
        if self.cache_dir is None:
            return None
        if cache_name is None:
            digest = hashlib.sha1(
                json.dumps(request, sort_keys=True).encode(), usedforsecurity=False
            ).hexdigest()[:12]
            cache_name = f"auto/{request['path'].strip('/').replace('/', '_')}_{digest}"
        return self.cache_dir / f"{cache_name}.json"

    @staticmethod
    def _write_cache(path: Path, table: RawTable) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"request": table.request, "fetched_at": table.fetched_at, "pages": table.pages,
                "urls": table.urls, "columns": table.columns, "data": table.rows}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


# --------------------------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------------------------

_ANALYTICS = "statistics/engines/stock/markets/index/analytics"


def fetch_index_history(client: IssClient, index: str, start, end) -> pd.DataFrame:
    """Daily index levels (``CLOSE``, ``VALUE``, ``CAPITALIZATION`` ...) between two dates."""
    table = client.table(
        f"history/engines/stock/markets/index/securities/{index}", "history",
        {"from": str(start), "till": str(end)},
        cache_name=f"index_history/{index}_{start}_{end}",
    )
    df = table.frame()
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    return df.sort_values("TRADEDATE").reset_index(drop=True)


def fetch_index_tickers(client: IssClient, index: str) -> pd.DataFrame:
    """Every ticker that was ever in the index, with first and last date of membership."""
    table = client.table(f"{_ANALYTICS}/{index}/tickers", "tickers", {},
                         cache_name=f"index_tickers/{index}")
    df = table.frame()
    df["from"] = pd.to_datetime(df["from"])
    df["till"] = pd.to_datetime(df["till"])
    return df


def fetch_index_composition(client: IssClient, index: str, date) -> pd.DataFrame:
    """Constituents and weights (percent, as published) on one date; empty if not calculated."""
    date = pd.Timestamp(date).date().isoformat()
    table = client.table(f"{_ANALYTICS}/{index}", "analytics", {"date": date, "limit": 100},
                         cache_name=f"composition/{index}/{date}")
    df = table.frame()
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "secid", "weight", "shortname"])
    return pd.DataFrame({
        "date": pd.to_datetime(df["tradedate"]),
        "ticker": df["ticker"].astype(str).str.strip(),
        "secid": df["secids"].astype(str).str.strip(),
        "weight": df["weight"].astype(float),
        "shortname": df["shortnames"],
    })


def fetch_index_ticker_series(client: IssClient, index: str, ticker: str, start, end) -> pd.DataFrame:
    """Daily per-member analytics (``issue_size_total``, ``ff_factor``, ``weight`` ...)."""
    table = client.table(
        f"{_ANALYTICS}/{index}/tickers/{ticker}", "ticker",
        {"from": str(start), "till": str(end), "limit": 100},
        cache_name=f"index_member/{index}/{ticker}_{start}_{end}",
    )
    df = table.frame()
    if not df.empty:
        df["tradedate"] = pd.to_datetime(df["tradedate"])
        df = df.sort_values("tradedate").reset_index(drop=True)
    return df


def fetch_board_history(
    client: IssClient, secid: str, start, end, boardgroup: int = BOARDGROUP_T_PLUS_SHARES
) -> pd.DataFrame:
    """Daily trading results of one share on a board group (see module docstring)."""
    table = client.table(
        f"history/engines/stock/markets/shares/boardgroups/{boardgroup}/securities/{secid}",
        "history",
        {"from": str(start), "till": str(end), "history.columns": HISTORY_COLUMNS},
        cache_name=f"board_history/bg{boardgroup}/{secid}_{start}_{end}",
    )
    df = table.frame()
    if not df.empty:
        df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    return df


def fetch_splits(client: IssClient) -> pd.DataFrame:
    """ISS register of splits/consolidations: ``before`` old shares become ``after`` new ones."""
    table = client.table("statistics/engines/stock/splits", "splits", {},
                         cache_name="splits/register")
    df = table.frame()
    df["tradedate"] = pd.to_datetime(df["tradedate"])
    return df.sort_values(["tradedate", "secid"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Trading calendar
# --------------------------------------------------------------------------------------------


def calculated_trading_days(index_history: pd.DataFrame) -> pd.DatetimeIndex:
    """Days on which the index was calculated **from trades** (positive turnover).

    ISS keeps a record for 2022-01-07 (Orthodox Christmas) with ``VALUE = 0``: the level is
    carried over from the evening session, every share has ``CLOSE = null`` and ``VOLUME = 0``.
    Taking such a day as a trading day would create a phantom observation with zero returns.
    """
    ok = index_history["CLOSE"].notna() & (index_history["VALUE"].fillna(0.0) > 0)
    return pd.DatetimeIndex(index_history.loc[ok, "TRADEDATE"]).normalize()


def build_calendar(index_dates: Iterable, start, end) -> pd.DataFrame:
    """One row per calendar day; ``tradable`` iff the index was calculated on that day.

    Using the days on which IMOEX was calculated (rather than "Monday to Friday minus a holiday
    list") makes the 28.02-23.03.2022 closure, Saturday sessions and the irregular holidays of
    the exchange come out right without any hand-written list. Pass
    :func:`calculated_trading_days` rather than all dates of the index history.
    """
    idx = pd.date_range(start, end, freq="D", name="date")
    open_days = pd.DatetimeIndex(pd.to_datetime(list(index_dates))).normalize()
    return pd.DataFrame({"tradable": idx.isin(open_days), "is_weekday": idx.weekday < 5}, index=idx)


def closure_spells(calendar: pd.DataFrame, min_weekdays: int = 10) -> pd.DataFrame:
    """Runs of consecutive non-tradable *weekdays*; weekends do not interrupt a run."""
    wk = calendar.loc[calendar["is_weekday"], "tradable"]
    run = (wk != wk.shift()).cumsum()
    rows = []
    for _, grp in wk.groupby(run):
        if not grp.iloc[0] and len(grp) >= min_weekdays:
            rows.append({"start": grp.index[0], "end": grp.index[-1], "weekdays": len(grp),
                         "calendar_days": (grp.index[-1] - grp.index[0]).days + 1})
    return pd.DataFrame(rows, columns=["start", "end", "weekdays", "calendar_days"])


def first_trading_days_of_month(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(trading_days).sort_values()
    first = pd.Series(days, index=days).groupby([days.year, days.month]).min()
    return pd.DatetimeIndex(first.to_numpy())


# --------------------------------------------------------------------------------------------
# Ex-dividend date
# --------------------------------------------------------------------------------------------


def settlement_lag(record_date) -> int:
    """Settlement cycle (trading days) that decides entitlement for a given record date.

    Equities on the Moscow Exchange settled T+2 until the last trade date 28.07.2023 and T+1
    from 31.07.2023. A record date up to and including 31.07.2023 is still fixed by trades that
    settled T+2 (a trade of Friday 28.07 settles on 01.08 and misses the register of 31.07), so
    the first record date under T+1 is 01.08.2023.
    """
    return 1 if pd.Timestamp(record_date) > T_PLUS_ONE_START else 2


def ex_dividend_date(
    record_date, trading_days: pd.DatetimeIndex, *, weekend_record_date: str = "previous"
) -> pd.Timestamp:
    """First trading day on which the share trades without the dividend (``NaT`` if unknown).

    Rule. The right to the dividend belongs to the holder in the shareholder register at the
    close of the record date ``R`` (Russian practice: "date of closing of the register"). A trade
    reaches the register when it settles, so with a settlement cycle of ``L`` trading days the
    last cum-dividend trade date is ``R - L`` and the first ex-date is ``R - L + 1``:

    * T+2 (``L = 2``, until the last trades of 28.07.2023): the **ex-date is the trading day
      before R**;
    * T+1 (``L = 1``, trades from 31.07.2023, i.e. record dates from 01.08.2023): the
      **ex-date is R itself**.

    Sources and assumptions. The T+2 -> T+1 switch date (31.07.2023) is the Moscow Exchange
    announcement given in the task; the register rule is the standard one and no external
    dividend calendar was consulted -- the rule is an *assumption* to be checked against the
    price gaps of the dividend table in use. Days are counted on the **trading** calendar of the
    exchange, not on a settlement calendar; the two differ on rare days (the weeks of the 2022
    closure, when NSD kept settling), and record dates there are not handled specially.
    Example: record date Thu 13.06.2019 with Wed 12.06 a holiday (Russia Day) -> ex-date Tue
    11.06, the trading day before R.

    If ``R`` is not a trading day the register is fixed at the close of the last trading day
    before it (``weekend_record_date="previous"``, the default). ``"next"`` takes the first
    trading day after ``R`` instead; the two readings differ by one trading day and only for the
    rare record dates on weekends and holidays. Record dates outside ``trading_days`` or without
    a preceding trading day give ``NaT``.
    """
    td = pd.DatetimeIndex(trading_days)
    rec = pd.Timestamp(record_date).normalize()
    if len(td) == 0 or rec < td[0] or rec > td[-1]:
        return pd.NaT
    if weekend_record_date == "previous":
        pos = int(td.searchsorted(rec, side="right")) - 1
    elif weekend_record_date == "next":
        pos = int(td.searchsorted(rec, side="left"))
    else:
        raise ValueError(f"weekend_record_date must be 'previous' or 'next', got {weekend_record_date!r}")
    pos_ex = pos - (settlement_lag(td[pos]) - 1)
    return td[pos_ex] if pos_ex >= 0 else pd.NaT


def prepare_dividends(
    dividends: pd.DataFrame, trading_days: pd.DatetimeIndex, tickers: Iterable[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """Validate a dividend table, drop what cannot be used and attach the ex-date.

    ``dividends`` needs the columns ``secid``, ``record_date``, ``amount`` (per share, in the
    share basis at the ex-date, **gross** of withholding tax) and ``currency``. Only RUB is
    kept: a foreign-currency payment would need the exchange rate of the payment date, which is
    not part of this dataset. The returned report counts every exclusion so the loss is visible
    in ``provenance.json``.
    """
    df = dividends.copy()
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["currency"] = df["currency"].astype(str).str.upper().str.strip()
    report: dict = {"n_input": len(df), "currencies": df["currency"].value_counts().to_dict()}
    if tickers is not None:
        keep = df["secid"].isin(set(tickers))
        report["excluded_not_in_universe"] = int((~keep).sum())
        df = df[keep]
    bad = ~(df["amount"].astype(float) > 0)
    report["excluded_invalid_amount"] = int(bad.sum())
    df = df[~bad]
    non_rub = df["currency"] != "RUB"
    report["excluded_non_rub"] = int(non_rub.sum())
    report["non_rub_records"] = df.loc[non_rub, ["secid", "record_date", "amount", "currency"]] \
        .assign(record_date=lambda d: d["record_date"].dt.strftime("%Y-%m-%d")) \
        .to_dict("records")
    df = df[~non_rub].copy()
    df["ex_date"] = [ex_dividend_date(d, trading_days) for d in df["record_date"]]
    df["settlement_lag"] = [settlement_lag(d) for d in df["record_date"]]
    outside = df["ex_date"].isna()
    report["excluded_outside_calendar"] = int(outside.sum())
    df = df[~outside].sort_values(["secid", "ex_date"]).reset_index(drop=True)
    report["n_applied"] = len(df)
    return df, report


# --------------------------------------------------------------------------------------------
# Splits and total return
# --------------------------------------------------------------------------------------------


def split_factor(index: pd.DatetimeIndex, splits: pd.DataFrame | None) -> pd.Series:
    """Multiplier that expresses a price (or per-share amount) in the share basis of the last
    date: ``1 / (shares_after / shares_before)`` of every split strictly after the date.

    A split dated after the last day of ``index`` is ignored (it lies outside the sample).
    """
    index = pd.DatetimeIndex(index)
    factor = pd.Series(1.0, index=index)
    if splits is None or len(splits) == 0 or len(index) == 0:
        return factor
    for s in splits.itertuples():
        when = pd.Timestamp(s.tradedate)
        if when > index[-1]:
            continue
        factor[index < when] /= float(s.after) / float(s.before)
    return factor


def adjust_for_splits(close: pd.Series, splits: pd.DataFrame | None) -> pd.Series:
    """Split-adjusted close in the share basis of the last date (no dividend adjustment)."""
    return close.astype(float) * split_factor(close.index, splits)


def total_return_index(
    close: pd.Series,
    dividends: pd.DataFrame | None,
    splits: pd.DataFrame | None = None,
    *,
    base: float = 100.0,
) -> pd.Series:
    """Total-return index of one share: gross cash dividends reinvested on the ex-date.

    ``r_t = (A_t + D_t) / A_prev - 1`` where ``A`` is the split-adjusted close, ``A_prev`` the
    previous available close and ``D_t`` the dividend whose ex-date is ``t`` (credited on the
    first day with a quote on or after the ex-date if the share did not trade on it). The index
    starts at ``base`` on the first quote and is ``NaN`` on days without a quote. Dividend
    amounts are in the share basis at the ex-date and are converted with :func:`split_factor`.
    """
    adj = adjust_for_splits(close, splits)
    valid = adj.dropna()
    tri = pd.Series(np.nan, index=close.index, name=close.name)
    if valid.empty:
        return tri
    cash = pd.Series(0.0, index=valid.index)
    if dividends is not None and len(dividends):
        last = close.index[-1]
        for ex, amount in zip(dividends["ex_date"], dividends["amount"], strict=True):
            ex = pd.Timestamp(ex)
            pos = int(valid.index.searchsorted(ex))
            if pos == 0 or pos >= len(valid):
                continue  # no previous quote to reinvest at, or ex-date after the last quote
            cash.iloc[pos] += float(amount) * _dividend_basis(ex, last, splits)
    ret = (valid + cash) / valid.shift(1) - 1.0
    ret.iloc[0] = 0.0
    tri.loc[valid.index] = base * (1.0 + ret).cumprod()
    return tri


def index_dividend_flow(price_level: pd.Series, total_return_level: pd.Series) -> pd.Series:
    """Daily dividend return of an index: total-return return minus price return.

    ``MCFTR`` (gross total return) and ``IMOEX`` are published by MOEX for free; on a day when
    constituents go ex-dividend the total-return index does not fall while the price index does,
    so the difference is ``sum_i w_i * d_i / P_i`` over the shares ex on that day. It is the
    only dividend information in the public ISS data: spikes mark ex-dates in aggregate and the
    sum over a year is the index dividend yield, which a per-share dividend table can be checked
    against. Days without both levels on consecutive trading days (2022-03-24, the first day
    after the closure, has no MCFTR) are ``NaN``.
    """
    both = pd.concat([price_level, total_return_level], axis=1, keys=["pr", "tr"])
    prev_ok = both.shift(1).notna().all(axis=1) & both.notna().all(axis=1)
    flow = both["tr"].pct_change(fill_method=None) - both["pr"].pct_change(fill_method=None)
    return flow.where(prev_ok).rename("dividend_flow")


def _dividend_basis(ex: pd.Timestamp, last: pd.Timestamp, splits: pd.DataFrame | None) -> float:
    """Factor converting a per-share amount quoted on ``ex`` into the basis of the last date."""
    factor = 1.0
    if splits is not None:
        for s in splits.itertuples():
            if ex < pd.Timestamp(s.tradedate) <= last:
                factor /= float(s.after) / float(s.before)
    return factor


# --------------------------------------------------------------------------------------------
# Jumps vs the register of corporate actions
# --------------------------------------------------------------------------------------------


def detect_jumps(close: pd.DataFrame, threshold: float = 0.60) -> pd.DataFrame:
    """Close-to-close moves above ``threshold`` between consecutive *available* quotes.

    Unlike the legacy detector, gaps in a ticker's history are bridged and reported
    (``gap_days`` = market days without a quote in between); a legacy detector with a bounded
    ``ffill`` misses a consolidation that follows a trading halt (IRAO, 2015-01-20) while an
    unbounded one flags moves across the 2022 closure as splits.
    """
    rows = []
    for ticker in close.columns:
        s = close[ticker].dropna()
        r = s.pct_change()
        for d in s.index[(r.abs() > threshold).to_numpy()]:
            pos = s.index.get_loc(d)
            prev = s.index[pos - 1]
            gap = int(close.index.get_loc(d) - close.index.get_loc(prev) - 1)
            rows.append({"ticker": ticker, "date": d, "prev_date": prev,
                         "price_prev": float(s.iloc[pos - 1]), "price": float(s.iloc[pos]),
                         "r": float(r.iloc[pos]), "gap_days": gap})
    cols = ["ticker", "date", "prev_date", "price_prev", "price", "r", "gap_days"]
    return pd.DataFrame(rows, columns=cols)


def reconcile_jumps(jumps: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Attach the ISS split record (if any) to every jump.

    ``status`` is ``iss_split`` when the register has an event for the ticker between the two
    quotes, else ``unexplained``. ``market_return_net_of_split`` is the move that remains after
    removing the split ratio, i.e. what a correct adjustment leaves in the returns.
    """
    out = jumps.copy()
    status, ratio, net = [], [], []
    for j in out.itertuples():
        hit = splits[(splits["secid"] == j.ticker) & (splits["tradedate"] > j.prev_date)
                     & (splits["tradedate"] <= j.date)]
        if len(hit):
            rr = float(hit["after"].iloc[0]) / float(hit["before"].iloc[0])
            status.append("iss_split")
            ratio.append(rr)
            net.append(j.price / j.price_prev * rr - 1.0)
        else:
            status.append("unexplained")
            ratio.append(np.nan)
            net.append(np.nan)
    out["status"] = status
    out["split_ratio"] = ratio
    out["market_return_net_of_split"] = net
    return out


def share_count_change(member: pd.DataFrame, prev_date, date) -> dict:
    """Share-count evidence from the per-member index analytics around a price jump.

    ``member`` is :func:`fetch_index_ticker_series`; ``issue_size_total`` is the number of shares
    on the day. Compares the last row up to ``prev_date`` with the first row from ``date``.
    """
    if member.empty:
        return {"status": "no_index_data"}
    before = member[member["tradedate"] <= prev_date] if pd.notna(prev_date) \
        else member[member["tradedate"] < date]
    after = member[member["tradedate"] >= date]
    if before.empty or after.empty:
        return {"status": "no_index_data_on_one_side"}
    b, a = before.iloc[-1], after.iloc[0]
    return {
        "status": "ok",
        "date_before": b["tradedate"].date().isoformat(), "date_after": a["tradedate"].date().isoformat(),
        "issue_size_before": int(b["issue_size_total"]), "issue_size_after": int(a["issue_size_total"]),
        "issue_ratio": float(a["issue_size_total"]) / float(b["issue_size_total"]),
        "weight_before": float(b["weight"]), "weight_after": float(a["weight"]),
        "ff_before": float(b["ff_factor"]), "ff_after": float(a["ff_factor"]),
    }


def as_fraction(shares_before: int, shares_after: int, max_term: int = 10000) -> tuple[int, int]:
    """``(before, after)`` in the notation of the ISS register from two exact share counts.

    ``before`` old shares become ``after`` new ones (GMKN 1 -> 100, VTBR 5000 -> 1). A ratio whose
    reduced terms exceed ``max_term`` (e.g. an issue of 3.47 % new shares) is not a split.
    """
    from math import gcd

    g = gcd(int(shares_before), int(shares_after))
    before, after = int(shares_before) // g, int(shares_after) // g
    if max(before, after) > max_term:
        raise ValueError(f"{shares_before} -> {shares_after} is not a simple split ratio")
    return before, after


def classify_jump(in_register: bool, evidence: dict, price_ratio: float) -> str:
    """Final status of a price jump.

    * ``iss_split`` -- the ISS register has the event;
    * ``share_count_change`` -- no register record, but the share count changed by more than 5 %
      and the price moved against it up to a plausible market move (|net| < 50 %): a
      consolidation/split missing from the register (it starts in December 2018);
    * ``market_move_with_share_change`` -- the share count changed but the price move is far
      larger than the change explains: not adjusted;
    * ``market_move`` -- the share count is unchanged: a real price move, not adjusted;
    * ``unexplained_no_index_data`` -- the share was not in IMOEX around the date, so the
      analytics have no share count: needs manual review, not adjusted.
    """
    if in_register:
        return "iss_split"
    if evidence.get("status") != "ok":
        return "unexplained_no_index_data"
    if abs(evidence["issue_ratio"] - 1.0) <= 0.05:
        return "market_move"
    net = price_ratio * evidence["issue_ratio"] - 1.0
    return "share_count_change" if abs(net) < 0.5 else "market_move_with_share_change"


# --------------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------------


def history_to_panels(history: pd.DataFrame, trading_days: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Wide ``close``/``volume``/``value`` panels on the trading calendar.

    Rows on non-trading days are dropped (a TQBR row with a null ``CLOSE`` on a closure day must
    not become a "day"). If a ticker has several boards on one date, the board with the largest
    ``VALUE`` is used.
    """
    trading_days = pd.DatetimeIndex(trading_days)
    h = history[history["TRADEDATE"].isin(trading_days)].copy()
    h["VALUE"] = h["VALUE"].astype(float).fillna(0.0)
    h = h.sort_values(["SECID", "TRADEDATE", "VALUE"], ascending=[True, True, False])
    h = h.drop_duplicates(["SECID", "TRADEDATE"], keep="first")
    tickers = sorted(h["SECID"].unique())
    panels = {}
    for name, col in (("close", "CLOSE"), ("volume", "VOLUME"), ("value", "VALUE")):
        wide = h.pivot(index="TRADEDATE", columns="SECID", values=col)
        panels[name] = wide.reindex(index=trading_days, columns=tickers).astype(float)
        panels[name].index.name = "date"
    return panels


def ticker_traded_mask(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """True where the ticker had a quote and positive volume on an open market day."""
    return close.notna() & (volume.fillna(0.0) > 0)


# --------------------------------------------------------------------------------------------
# Provenance helpers
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(
    root: Path | str, exclude: tuple[str, ...] = ("provenance.json", "_logs")
) -> dict[str, dict]:
    """``{relative path: {sha256, bytes}}`` for every file below ``root`` (minus ``exclude``)."""
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not any(part in exclude for part in rel.parts):
            out[rel.as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out
