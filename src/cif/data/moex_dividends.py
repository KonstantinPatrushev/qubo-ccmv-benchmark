"""Per-share dividends of the MOEX panel (R6, RQ3 exploratory): tidy events from the Cbonds
calendar and the registry of official disclosures.

The analysis plan (section 3, RU) takes the dividends from the Cbonds calendar, adds the payments
of issuers that the calendar lacks from official disclosures, returns Cbonds' amounts to the share
basis of the payment date, and checks every payment against the price drop on the ex-date and the
sum over the index against the dividend flow ``MCFTR - IMOEX``. This module holds the rules; the
price and index checks are in :mod:`cif.data.moex_dividend_checks`, the registry in
:mod:`cif.data.moex_dividend_registry`, the build in ``scripts/data/build_moex_dividends.py``.

Conventions
-----------
* ``secid`` is the series of the project panel (``data/processed/moex_v2``) that traded on the
  ex-date. Cbonds lists the history of a renamed security under its *current* ticker (E.ON Russia
  under UPRO, TCS Group under T), so the ticker cell is mapped to whichever candidate had a quote
  on the ex-date (:func:`assign_secid`).
* ``amount`` is gross, per share (per receipt for GDR/ADS) **in the share basis of the ex-date**,
  i.e. as the issuer declared it; only RUB events enter the RUB total return. Foreign-currency
  events are converted when the caller supplies a table of exchange rates (``fx`` of
  :func:`build_events`, :func:`convert_to_rub`; the build uses the official Bank of Russia rate of
  ``cif.data.cbr_fx``): ``amount`` and ``currency`` then hold roubles at **the rate of the last date
  on or before the ex-date**, the original amount, currency, rate and the date of the rate staying in
  ``amount_orig`` / ``currency_orig`` / ``fx_rate`` / ``fx_date``. Without a table the event keeps
  its currency and is out of every RUB total.
* A vendor amount that is wrong is replaced only through a documented
  :class:`cif.data.moex_dividend_registry.AmountCorrection` (the exported amount stays in
  ``cbonds_amount``, the event is flagged ``amount_corrected``); a correction that matches no row
  stops the build.
* ``usable_rub`` marks the events a total-return series can credit: RUB, with the ex-date after the
  first quote of the series and not after its last (a halted ex-day is credited on the next quote).
  At the first day of a renamed series (``EONR`` -> ``UPRO``) the price check borrows the
  predecessor's last close (flag ``rename_boundary``), but the event is not creditable within the
  single series.
* The ex-date follows the settlement rule of :func:`cif.data.moex.ex_dividend_date` (T+2 until the
  trades of 28.07.2023, T+1 after), extended over the closure of 28.02-23.03.2022
  (:func:`ex_date_from_record`). Cbonds' own ex-date is a cross-check; where the two disagree and
  the dividend is large enough to show in the price, the price decides
  (:func:`choose_ex_date`); otherwise the rule stands. An ex-date that a registry source *states*
  (an issuer or exchange notice) stands against the rule instead (``ex_basis`` ``declared``, flag
  ``ex_declared``), and yields only to a price that fits the rule's date clearly better.
* Cbonds restates amounts to today's share count. :func:`restatement_factor` undoes that with the
  split register, **including splits after the end of the panel** (the 1:10 split of T on
  2026-04-17), and follows successor chains (the GDR TCSG became the share T).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cif.data import cbonds, moex
from cif.data import moex_dividend_checks as checks
from cif.data.moex_dividend_checks import MARGIN_SIGMA, MIN_YIELD_SIGMA

#: Cbonds period type that repeats a payment already listed under a specific period label.
GENERIC_PERIOD_TYPES = frozenset({"регулярные"})


@dataclass(frozen=True)
class Succession:
    """``predecessor`` (a panel series) continues as ``successor`` one for one."""

    predecessor: str
    successor: str
    note: str = ""


SUCCESSIONS: tuple[Succession, ...] = (
    Succession("EONR", "UPRO", "E.ON Russia renamed Unipro (2016-07); Cbonds lists the whole history "
                               "under UPRO"),
    Succession("EPLN", "SFIN", "renamed at the turn of 2017/2018; Cbonds lists the history under SFIN"),
    Succession("TCSG", "T", "TCS Group GDR exchanged one for one into the shares of T-Technologies "
                            "(2024-11); Cbonds lists the history only from 2024-11 under T"),
)


# --------------------------------------------------------------------------------------------
# Ex-date
# --------------------------------------------------------------------------------------------


def settlement_days(trading_days: pd.DatetimeIndex, closures: pd.DataFrame | None) -> pd.DatetimeIndex:
    """Days on which trades settle: the trading days plus the weekdays inside closure spells.

    During the closure of 28.02-23.03.2022 the exchange did not trade but the depository kept
    settling (the trades of 25.02 settled on 01.03). Elsewhere the exchange and settlement
    calendars are taken to coincide, as in :func:`cif.data.moex.ex_dividend_date`; a day on which
    the exchange traded and the depository did not (the January holidays) cannot be seen from the
    trading calendar, which is why the price check arbitrates against the vendor's date.
    """
    days = pd.DatetimeIndex(trading_days)
    if closures is None or len(closures) == 0:
        return days
    extra = [pd.bdate_range(s, e) for s, e in zip(closures["start"], closures["end"], strict=True)]
    return days.union(pd.DatetimeIndex(np.concatenate([d.values for d in extra])))


def ex_date_from_record(
    record_date, trading_days: pd.DatetimeIndex, closures: pd.DataFrame | None = None
) -> pd.Timestamp:
    """First trading day on which the share trades without the dividend (``NaT`` if unknown).

    A trade reaches the register of the record date ``R`` iff it settles by ``R``; with a cycle
    of ``L`` settlement days the last cum-dividend trading day is the last trading day whose
    settlement day is at most ``L`` settlement days before the last settlement day on or before
    ``R``, and the ex-date is the next trading day. Without closures this is
    :func:`cif.data.moex.ex_dividend_date`; inside the 2022 closure it gives the first trading day
    after it, and a record date up to two days after the reopening gets the reopening day.
    """
    settle = settlement_days(trading_days, closures)
    td = pd.DatetimeIndex(trading_days)
    rec = pd.Timestamp(record_date).normalize()
    if len(settle) == 0 or rec < settle[0] or rec > settle[-1]:
        return pd.NaT
    c = int(settle.searchsorted(rec, side="right")) - 1
    k = c - moex.settlement_lag(settle[c])
    if k < 0:
        return pd.NaT
    last_cum = int(td.searchsorted(settle[k], side="right")) - 1
    ex = last_cum + 1
    return td[ex] if last_cum >= 0 and ex < len(td) else pd.NaT


def align_to_trading_day(date, trading_days: pd.DatetimeIndex, side: str = "next") -> pd.Timestamp:
    """The date itself if it is a trading day, else the next (or previous) one; ``NaT`` past the ends."""
    td = pd.DatetimeIndex(trading_days)
    d = pd.Timestamp(date).normalize()
    if side == "next":
        pos = int(td.searchsorted(d, side="left"))
        return td[pos] if pos < len(td) else pd.NaT
    if side == "previous":
        pos = int(td.searchsorted(d, side="right")) - 1
        return td[pos] if pos >= 0 else pd.NaT
    raise ValueError(f"side must be 'next' or 'previous', got {side!r}")


def trading_day_distance(a, b, trading_days: pd.DatetimeIndex) -> int:
    """Trading sessions from ``b`` to ``a`` (positive when ``a`` is later); dates are aligned forward."""
    td = pd.DatetimeIndex(trading_days)
    return int(td.searchsorted(pd.Timestamp(a))) - int(td.searchsorted(pd.Timestamp(b)))


def choose_ex_date(
    rule: pd.Timestamp, vendor: pd.Timestamp, abn_rule: float, abn_vendor: float, dividend_yield: float,
    sigma: float, *, min_yield_sigma: float = MIN_YIELD_SIGMA, margin_sigma: float = MARGIN_SIGMA,
) -> tuple[pd.Timestamp, str]:
    """Pick the ex-date among the rule's and the vendor's, with the basis of the choice.

    ``abn_*`` is the abnormal return of the share on that date, ``dividend_yield`` the dividend
    over the previous close and ``sigma`` the share's daily standard deviation of abnormal
    returns. The price speaks only when the yield is at least ``min_yield_sigma`` sigmas and one
    date fits ``-yield`` better than the other by ``margin_sigma`` sigmas; otherwise the rule
    (the documented exchange mechanism) stands. Bases: ``agree``, ``rule_only``, ``rule_no_price``,
    ``rule_uninformative``, ``price_rule``, ``price_vendor``, ``rule_default``.
    """
    if pd.isna(vendor):
        return rule, "rule_only"
    if rule == vendor:
        return rule, "agree"
    if any(pd.isna(x) for x in (abn_rule, abn_vendor, dividend_yield, sigma)):
        return rule, "rule_no_price"
    if dividend_yield < min_yield_sigma * sigma:
        return rule, "rule_uninformative"
    err_rule = abs(abn_rule + dividend_yield)
    err_vendor = abs(abn_vendor + dividend_yield)
    if err_vendor + margin_sigma * sigma < err_rule:
        return vendor, "price_vendor"
    if err_rule + margin_sigma * sigma < err_vendor:
        return rule, "price_rule"
    return rule, "rule_default"


# --------------------------------------------------------------------------------------------
# Split restatement
# --------------------------------------------------------------------------------------------


def successor_chain(secid: str, successions: Iterable[Succession] = SUCCESSIONS) -> list[str]:
    """``secid`` followed by the series it continued as: ``TCSG`` -> ``["TCSG", "T"]``."""
    nxt = {s.predecessor: s.successor for s in successions}
    chain = [secid]
    while chain[-1] in nxt and nxt[chain[-1]] not in chain:
        chain.append(nxt[chain[-1]])
    return chain


def restatement_factor(
    secid: str, ex_date, splits: pd.DataFrame | None, successions: Iterable[Succession] = SUCCESSIONS
) -> float:
    """Factor that takes a Cbonds amount back to the share basis of the ex-date.

    Cbonds shows amounts for today's shares. A split ``before -> after`` on a date after the
    ex-date multiplies the share count by ``after / before``, so the declared amount is
    ``cbonds_amount * after / before`` (GMKN 1:100 in 2024: older amounts are 100 times too small;
    VTBR 5000:1: 5000 times too large). All splits count, including those after the end of the
    price panel (``applied == False`` in the register), and the splits of the successors of the
    series (the GDR TCSG became the share T, whose 1:10 split of 2026-04-17 restates the
    GDR-era amounts too). A split on the ex-date itself is already in the new basis.
    """
    if splits is None or len(splits) == 0:
        return 1.0
    ex = pd.Timestamp(ex_date)
    chain = set(successor_chain(secid, successions))
    sel = splits[splits["secid"].isin(chain) & (pd.to_datetime(splits["tradedate"]) > ex)]
    factor = 1.0
    for before, after in zip(sel["before"], sel["after"], strict=True):
        factor *= float(after) / float(before)
    return factor


# --------------------------------------------------------------------------------------------
# Mapping the vendor's tickers to the series of the panel
# --------------------------------------------------------------------------------------------


def candidate_secids(
    aliases: Iterable[str], panel_secids: set[str], successions: Iterable[Succession] = SUCCESSIONS
) -> list[tuple[str, str]]:
    """Panel series a ticker cell can mean: the aliases that are panel series, then predecessors
    of the aliases (``UPRO`` -> ``EONR``). Pairs of ``(secid, "alias" | "predecessor")``."""
    aliases = [a for a in aliases if a]
    out: list[tuple[str, str]] = []
    for alias in aliases:
        if alias in panel_secids and all(c[0] != alias for c in out):
            out.append((alias, "alias"))
    for alias in aliases:
        for s in successions:
            if s.successor == alias and s.predecessor in panel_secids and all(c[0] != s.predecessor for c in out):
                out.append((s.predecessor, "predecessor"))
    return out


def assign_secid(
    aliases: Iterable[str], ex_date, close: pd.DataFrame, successions: Iterable[Succession] = SUCCESSIONS
) -> tuple[str | None, str]:
    """Series of the panel that traded on ``ex_date``, with a status.

    ``ok``: exactly one candidate has a quote on the ex-date. ``ambiguous``: several do (the first
    is taken). ``no_quote_on_ex``: none does (the first candidate is taken; the event cannot be
    checked against prices). ``not_in_universe`` / ``no_ticker``: no candidate at all.
    """
    aliases = list(aliases)
    cands = candidate_secids(aliases, set(close.columns), successions)
    if not cands:
        return None, ("not_in_universe" if any(aliases) else "no_ticker")
    ex = pd.Timestamp(ex_date)
    if ex in close.index:
        row = close.loc[ex]
        quoted = [c for c, _ in cands if pd.notna(row[c])]
    else:
        quoted = []
    if len(quoted) == 1:
        return quoted[0], "ok"
    if len(quoted) > 1:
        return quoted[0], "ambiguous"
    return cands[0][0], "no_quote_on_ex"


# --------------------------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------------------------


def drop_duplicates(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse payments listed twice (QIWI: type "регулярные" plus a quarter label).

    Rows are duplicates when ``secid``, ``record_date``, ``currency`` and the amount (to nine
    decimals) agree; two *different* amounts on one record date are two payments (VTBR 2021)
    and stay. Of a group the row with a specific period type is kept, then the lowest
    ``sheet_row``. Returns ``(kept with n_duplicates, dropped with duplicate_of)``.
    """
    ev = events.copy()
    ev["_amount"] = ev["amount"].astype(float).round(9)
    ev["_generic"] = ev["period_type"].isin(GENERIC_PERIOD_TYPES)
    key = ["secid", "record_date", "currency", "_amount"]
    ranked = ev.sort_values(["_generic", "sheet_row"], kind="stable")
    is_dup = ranked.duplicated(subset=key, keep="first")
    kept, dropped = ranked[~is_dup].copy(), ranked[is_dup].copy()
    if len(dropped):
        owner = kept.set_index(key)["sheet_row"]
        dropped["duplicate_of"] = [owner[tuple(r)] for r in dropped[key].itertuples(index=False)]
    else:
        dropped["duplicate_of"] = pd.Series(dtype="int64")
    counts = dropped.groupby(key).size()
    kept["n_duplicates"] = [int(counts.get(tuple(r), 0)) for r in kept[key].itertuples(index=False)]
    drop_cols = ["_amount", "_generic"]
    kept = kept.drop(columns=drop_cols).sort_values("sheet_row", kind="stable").reset_index(drop=True)
    dropped = dropped.drop(columns=drop_cols).sort_values("sheet_row", kind="stable").reset_index(drop=True)
    return kept, dropped


# --------------------------------------------------------------------------------------------
# The panel the events are checked against
# --------------------------------------------------------------------------------------------


@dataclass
class Panel:
    """The tables of ``data/processed/moex_v2`` that the events are mapped to and checked against."""

    close: pd.DataFrame  # as-traded close, tradable days from the start of the panel
    adj: pd.DataFrame  # close adjusted for the splits that fall inside the panel
    calendar: pd.DataFrame  # every calendar day: tradable, is_weekday
    splits: pd.DataFrame  # secid, tradedate, before, after, applied (incl. splits after the panel)
    universe: pd.DataFrame  # secid, index_from, index_till (days in IMOEX, from the all-time list)
    membership: pd.DataFrame  # monthly composition: date, secid, weight
    index_levels: pd.DataFrame  # IMOEX, MCFTR
    index_flow: pd.Series  # MCFTR return - IMOEX return

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.calendar.index[self.calendar["tradable"]])

    @property
    def closures(self) -> pd.DataFrame:
        return moex.closure_spells(self.calendar)

    @property
    def start(self) -> pd.Timestamp:
        return self.close.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.close.index[-1]

    @property
    def applied_splits(self) -> pd.DataFrame:
        return self.splits[self.splits["applied"].astype(bool)]


def load_panel(directory: Path | str = "data/processed/moex_v2") -> Panel:
    d = Path(directory)
    flow = pd.read_parquet(d / "index_dividend_flow.parquet")["dividend_flow"]
    return Panel(
        close=pd.read_parquet(d / "prices_close.parquet"),
        adj=pd.read_parquet(d / "prices_adj.parquet"),
        calendar=pd.read_parquet(d / "calendar.parquet"),
        splits=pd.read_parquet(d / "splits.parquet"),
        universe=pd.read_parquet(d / "universe.parquet"),
        membership=pd.read_parquet(d / "imoex_membership.parquet"),
        index_levels=pd.read_parquet(d / "index_levels.parquet"),
        index_flow=flow,
    )


@dataclass
class Built:
    """Result of :func:`build_events`."""

    events: pd.DataFrame  # one row per payment kept, with checks and flags
    audit: pd.DataFrame  # one row per vendor row: what became of it
    registry_audit: pd.DataFrame  # the same for the registry rows
    model: pd.DataFrame  # market model (beta, sigma) used for the price checks
    corrections_audit: pd.DataFrame  # vendor rows whose amount was replaced by a documented correction


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------

EVENT_COLUMNS = [
    "event_id", "source", "source_ref", "secid", "ex_date", "record_date", "amount", "currency", "amount_orig",
    "currency_orig", "fx_rate", "fx_date", "period", "period_type", "cbonds_tickers", "cbonds_name", "cbonds_amount",
    "restatement_factor", "ex_rule",
    "ex_vendor", "ex_basis", "secid_status", "n_duplicates", "prev_date", "prev_secid", "prev_close_adj", "ex_return",
    "mkt_return", "beta", "sigma", "abn_return", "amount_adj", "dividend_yield", "z", "capture", "drop_status",
    "in_index_on_ex", "flags", "usable_rub",
]
_PAYMENT_COLUMNS = [
    "source", "source_ref", "secid", "record_date", "ex_rule", "ex_vendor", "ex_date_cbonds", "amount",
    "currency", "period", "period_type", "cbonds_tickers", "cbonds_name", "cbonds_amount",
    "restatement_factor", "n_duplicates", "secid_status", "source_flags",
]


def _align_inside(date, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    """The date's trading day (the next one), but ``NaT`` for a date outside the calendar: a date
    before the first session must not be moved *into* the panel."""
    if pd.isna(date) or date < trading_days[0] or date > trading_days[-1]:
        return pd.NaT
    return align_to_trading_day(date, trading_days, "next")


def _window_status(key_date, raw_date, start: pd.Timestamp, end: pd.Timestamp) -> str | None:
    probe = key_date if pd.notna(key_date) else raw_date
    if pd.isna(probe) or probe < start:
        return "before_panel"
    if probe > end:
        return "after_panel"
    return None


def _vendor_payments(rows: pd.DataFrame, panel: Panel, successions) -> pd.DataFrame:
    """Map every vendor row to a panel series (or say why not), restate and date it."""
    td, closures = panel.trading_days, panel.closures
    pay = rows.copy()
    pay["aliases"] = pay["tickers"].map(cbonds.split_tickers)
    pay["ex_rule"] = [ex_date_from_record(r, td, closures) for r in pay["record_date"]]
    pay["ex_vendor"] = [_align_inside(d, td) for d in pay["ex_date_cbonds"]]
    pay["key_date"] = pay["ex_rule"].where(pay["ex_rule"].notna(), pay["ex_vendor"])
    secids = set(panel.close.columns)
    status, secid, secid_status, factor = [], [], [], []
    for r in pay.itertuples():
        cands = candidate_secids(r.aliases, secids, successions)
        window = _window_status(r.key_date, r.ex_date_cbonds, panel.start, panel.end)
        if not any(r.aliases):
            st, sid, sst = "no_ticker", None, "no_ticker"
        elif not cands:
            st, sid, sst = "not_in_universe", None, "not_in_universe"
        elif window is not None:
            st, sid, sst = window, None, "outside_window"
        else:
            sid, sst = assign_secid(r.aliases, r.key_date, panel.close, successions)
            st = "ok"
        status.append(st)
        secid.append(sid)
        secid_status.append(sst)
        factor.append(restatement_factor(sid, r.key_date, panel.splits, successions) if sid is not None else np.nan)
    pay["status"], pay["secid"], pay["secid_status"], pay["restatement_factor"] = status, secid, secid_status, factor
    pay["cbonds_amount"] = pay["amount"]
    pay["amount"] = pay["cbonds_amount"] * pay["restatement_factor"]
    pay["source_flags"] = ""
    return pay


def _apply_corrections(pay: pd.DataFrame, corrections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace the amount of the vendor rows named by the corrections; return ``(rows, audit)``.

    A row is named by ``secid``, ``record_date``, ``currency`` and the amount as exported. A
    correction that names no row raises :class:`ValueError`: an export that changed would
    otherwise leave a stale correction in place unnoticed.
    """
    pay = pay.copy()
    audit = []
    for c in corrections.itertuples():
        hit = ((pay["status"] == "ok") & (pay["secid"] == c.secid) & (pay["record_date"] == c.record_date)
               & (pay["currency"] == c.currency) & np.isclose(pay["cbonds_amount"], c.cbonds_amount, rtol=1e-9, atol=0.0))
        if not hit.any():
            raise ValueError(f"amount correction {c.source_ref} matches no vendor row")
        pay.loc[hit, "amount"] = c.amount
        pay.loc[hit, "source_flags"] = "amount_corrected"
        audit.append((c.source_ref, c.secid, c.record_date, c.cbonds_amount, c.amount,
                      sorted(int(n) for n in pay.loc[hit, "sheet_row"]), c.reason))
    return pay, pd.DataFrame(audit, columns=["source_ref", "secid", "record_date", "cbonds_amount", "amount",
                                             "matched_sheet_rows", "reason"])


def _arbitrate(
    pay: pd.DataFrame, panel: Panel, model: pd.DataFrame, min_yield_sigma, margin_sigma, predecessors
) -> pd.DataFrame:
    """Choose the ex-date of every payment between the rule's and the vendor's.

    ``ex_basis`` says why: ``agree``, ``rule_only`` (no vendor date), ``rule_non_rub`` (the dividend
    cannot be tested without a rate), ``vendor_only`` (the rule has no date), ``declared`` or one of
    the bases of :func:`choose_ex_date`. The ex-date of a registry event is one that its source
    **states** (an issuer or exchange notice), not a vendor's guess: it stands against the rule
    (``declared``) wherever the price cannot decide, and yields only to a price that fits the
    rule's date clearly better (``price_rule``).
    """
    mkt = panel.index_levels["IMOEX"]
    applied = panel.applied_splits
    ex_dates, bases = [], []
    for r in pay.itertuples():
        rule, vendor = r.ex_rule, r.ex_vendor
        if pd.isna(rule):
            ex_dates.append(vendor)
            bases.append("vendor_only")
            continue
        if pd.isna(vendor) or vendor == rule:
            ex_dates.append(rule)
            bases.append("rule_only" if pd.isna(vendor) else "agree")
            continue
        stated = r.source == "registry"
        if str(r.currency).upper() != "RUB":
            ex_dates.append(vendor if stated else rule)
            bases.append("declared" if stated else "rule_non_rub")
            continue
        sigma = model.at[r.secid, "sigma"] if r.secid in model.index else np.nan
        beta = model.at[r.secid, "beta"] if r.secid in model.index else np.nan
        er = [checks.ex_day_return(panel.adj, mkt, r.secid, d, predecessors=predecessors)
              for d in (rule, vendor, min(rule, vendor))]
        abn = [x["ret"] - beta * x["mkt_ret"] if not np.isnan(x["ret"]) else np.nan for x in er[:2]]
        amount_adj = checks._adjusted_amount(r.secid, min(rule, vendor), r.amount, applied, panel.end)
        y = amount_adj / er[2]["prev_adj"] if er[2]["prev_adj"] > 0 else np.nan
        chosen, basis = choose_ex_date(rule, vendor, abn[0], abn[1], y, sigma, min_yield_sigma=min_yield_sigma,
                                       margin_sigma=margin_sigma)
        if stated and basis in ("rule_no_price", "rule_uninformative", "rule_default"):
            chosen, basis = vendor, "declared"
        ex_dates.append(chosen)
        bases.append(basis)
    out = pay.copy()
    out["ex_date"], out["ex_basis"] = ex_dates, bases
    return out


def convert_to_rub(payments: pd.DataFrame, fx: pd.DataFrame | None, *, max_age_days: int = 7) -> pd.DataFrame:
    """Express foreign-currency payments in RUB with a table of exchange rates.

    ``fx`` has a date index and one column per currency code holding RUB per unit (the central
    bank's official rate is the natural choice, the one the index methodology uses). A payment is
    converted at the last rate on or before its ex-date (the rule's date, or the vendor's if the
    rule has none), if that rate is at most ``max_age_days`` old: a rate set after the ex-date is
    never used. ``amount`` and ``currency`` then hold RUB, and ``amount_orig``, ``currency_orig``,
    ``fx_rate`` and ``fx_date`` (the date the rate is for) the original. A payment without such a
    rate stays as it is and is flagged ``fx_missing``. Without ``fx`` nothing changes and the four
    columns are empty: the data hold no rate table.
    """
    out = payments.copy()
    for col, empty in (("amount_orig", np.nan), ("currency_orig", None), ("fx_rate", np.nan), ("fx_date", pd.NaT)):
        if col not in out.columns:
            out[col] = empty
    out["currency_orig"] = out["currency_orig"].astype(object)
    out["fx_date"] = pd.to_datetime(out["fx_date"])
    if fx is None:
        return out
    flags = out["source_flags"].fillna("").astype(str)
    for i in out.index[out["currency"].astype(str).str.upper() != "RUB"]:
        cur = str(out.at[i, "currency"]).upper()
        date = out.at[i, "ex_rule"] if pd.notna(out.at[i, "ex_rule"]) else out.at[i, "ex_vendor"]
        rate, rate_date, age = np.nan, pd.NaT, np.inf
        if cur in fx.columns and pd.notna(date):
            known = fx[cur].dropna().loc[: pd.Timestamp(date)]
            if len(known):
                rate, rate_date, age = float(known.iloc[-1]), known.index[-1], (pd.Timestamp(date) - known.index[-1]).days
        if np.isfinite(rate) and age <= max_age_days:
            out.at[i, "amount_orig"], out.at[i, "currency_orig"] = out.at[i, "amount"], cur
            out.at[i, "fx_rate"], out.at[i, "fx_date"] = rate, rate_date
            out.at[i, "amount"] = out.at[i, "amount"] * rate
            out.at[i, "currency"] = "RUB"
            flags.at[i] = ",".join(f for f in (flags.at[i], "fx_converted") if f)
        else:
            flags.at[i] = ",".join(f for f in (flags.at[i], "fx_missing") if f)
    out["source_flags"] = flags
    return out


def _flags(row, trading_days: set) -> str:
    f = []
    if row.restatement_factor != 1.0:
        f.append("restated")
    if row.ex_basis in ("price_vendor", "vendor_only"):
        f.append("ex_vendor")
    elif row.ex_basis == "declared":
        f.append("ex_declared")
    elif pd.notna(row.ex_vendor) and row.ex_vendor != row.ex_date:
        f.append("ex_vendor_differs")
    if pd.notna(row.ex_date_cbonds) and row.ex_date_cbonds not in trading_days:
        f.append("vendor_ex_not_trading")
    if row.n_duplicates:
        f.append("merged_duplicates")
    if row.secid_status in ("ambiguous", "no_quote_on_ex"):
        f.append("secid_" + row.secid_status)
    if str(row.currency).upper() != "RUB":
        f.append("non_rub")
    if row.drop_status in ("outlier", "borderline"):
        f.append("price_" + row.drop_status)
    if row.prev_secid is not None and row.prev_secid != row.secid:
        f.append("rename_boundary")
    if row.before_first_quote:
        f.append("before_first_quote")
    if row.after_last_quote:
        f.append("after_last_quote")
    if not row.in_index_on_ex:
        f.append("not_in_index")
    if row.source_flags:
        f.append(row.source_flags)
    return ",".join(f)


def _finalize(
    pay: pd.DataFrame, panel: Panel, model: pd.DataFrame, min_yield_sigma, margin_sigma, predecessors
) -> pd.DataFrame:
    """Dates, price checks, flags and ids for payments that are already mapped to a panel series."""
    pay = _arbitrate(pay, panel, model, min_yield_sigma, margin_sigma, predecessors).reset_index(drop=True)
    checked = checks.check_events(
        pay[["secid", "ex_date", "amount", "currency"]], panel.adj, panel.index_levels["IMOEX"], model,
        splits=panel.splits, last_date=panel.end, min_yield_sigma=min_yield_sigma, predecessors=predecessors)
    out = pd.concat([pay, checked.drop(columns=["secid", "ex_date", "amount", "currency"])], axis=1)
    uni = panel.universe.set_index("secid")
    index_from = uni["index_from"].reindex(out["secid"]).to_numpy()
    index_till = uni["index_till"].reindex(out["secid"]).to_numpy()
    ex = out["ex_date"].to_numpy()
    out["in_index_on_ex"] = (index_from <= ex) & (ex <= index_till)
    first_quote = panel.close.apply(lambda col: col.first_valid_index())
    last_quote = panel.close.apply(lambda col: col.last_valid_index())
    first, last = first_quote.reindex(out["secid"]).to_numpy(), last_quote.reindex(out["secid"]).to_numpy()
    out["before_first_quote"], out["after_last_quote"] = ex <= first, ex > last
    # reinvestable: after the first quote of the series and not after its last (a halted ex-day is credited
    # on the next quote by cif.data.moex.total_return_index)
    out["usable_rub"] = (out["currency"].astype(str).str.upper() == "RUB") & ~out["before_first_quote"] \
        & ~out["after_last_quote"]
    out["source_flags"] = out["source_flags"].fillna("")
    trading_days = set(panel.trading_days)
    out["flags"] = [_flags(r, trading_days) for r in out.itertuples()]
    out["event_id"] = [f"{r.secid}|{r.record_date:%Y-%m-%d}|{r.currency}|{r.amount:.10g}" for r in out.itertuples()]
    return out


def coverage_table(
    events: pd.DataFrame, *, currency: str | None = "RUB", secids: Iterable[str] | None = None
) -> pd.DataFrame:
    """Events per ``secid`` and calendar year of the ex-date (zeros filled in); RUB events only
    by default, ``currency=None`` counts all. ``secids`` adds shares without any event as zero rows."""
    ev = events if currency is None else events[events["currency"] == currency]
    table = ev.groupby(["secid", ev["ex_date"].dt.year]).size().unstack(fill_value=0)
    if len(table):
        table = table.reindex(columns=range(int(table.columns.min()), int(table.columns.max()) + 1), fill_value=0)
    if secids is not None:
        table = table.reindex(sorted(set(table.index) | set(secids)), fill_value=0)
    table.columns.name = "year"
    return table


def build_events(
    rows: pd.DataFrame,
    panel: Panel,
    *,
    registry: pd.DataFrame | None = None,
    corrections: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None,
    successions: Iterable[Succession] = SUCCESSIONS,
    min_yield_sigma: float = MIN_YIELD_SIGMA,
    margin_sigma: float = MARGIN_SIGMA,
) -> Built:
    """Tidy events from the vendor rows (and the registry), checked against the panel.

    ``rows`` is :func:`cif.data.cbonds.read_calendar` output. Every vendor row ends in ``audit``
    with one status: ``ok`` (an event), ``duplicate``, ``no_ticker``, ``not_in_universe``,
    ``before_panel`` or ``after_panel``. ``registry`` has ``secid``, ``record_date``,
    ``ex_date_declared`` (may be ``NaT``), ``amount``, ``currency``, ``period``, ``period_type``
    and ``source_ref``; its rows are dated and checked like the vendor's, with the declared
    ex-date in the place of the vendor's, and a row that repeats a vendor payment is not added.
    ``corrections`` (``secid``, ``record_date``, ``cbonds_amount``, ``amount``, ``currency``,
    ``reason``, ``source_ref``) replace the amount of vendor rows that are known to be wrong; the
    exported amount stays in ``cbonds_amount`` and the event is flagged ``amount_corrected``.
    ``fx`` (see :func:`convert_to_rub`) converts foreign-currency payments; without it they keep
    their currency and stay out of every RUB total.
    """
    successions = tuple(successions)
    model = checks.market_model(panel.adj, panel.index_levels["IMOEX"])
    pay = _vendor_payments(rows, panel, successions)
    corrections_audit = pd.DataFrame(columns=["source_ref", "secid", "record_date", "cbonds_amount", "amount",
                                              "matched_sheet_rows", "reason"])
    if corrections is not None and len(corrections):
        pay, corrections_audit = _apply_corrections(pay, corrections)
    audit = pay[["sheet_row", "tickers", "name", "record_date", "amount", "currency", "status", "secid"]].copy()
    audit["duplicate_of"] = pd.array([pd.NA] * len(audit), dtype="Int64")
    kept, dropped = drop_duplicates(pay[pay["status"] == "ok"].copy())
    if len(dropped):
        idx = audit.set_index("sheet_row")
        idx.loc[dropped["sheet_row"], "status"] = "duplicate"
        idx.loc[dropped["sheet_row"], "duplicate_of"] = dropped["duplicate_of"].to_numpy()
        audit = idx.reset_index()
    kept["source"] = "cbonds"
    kept["source_ref"] = [f"cbonds:{n}" for n in kept["sheet_row"]]
    kept["cbonds_tickers"], kept["cbonds_name"] = kept["tickers"], kept["name"]
    frames = [kept[_PAYMENT_COLUMNS]]
    registry_audit = pd.DataFrame(columns=["source_ref", "secid", "record_date", "status"])
    if registry is not None and len(registry):
        reg, registry_audit = _registry_payments(registry, kept, panel)
        frames.append(reg)
    payments = convert_to_rub(pd.concat(frames, ignore_index=True), fx)
    predecessors = {s.successor: s.predecessor for s in successions}
    events = _finalize(payments, panel, model, min_yield_sigma, margin_sigma, predecessors)
    events = events[EVENT_COLUMNS].sort_values(["secid", "ex_date", "record_date", "amount"], kind="stable")
    return Built(events=events.reset_index(drop=True), audit=audit.sort_values("sheet_row").reset_index(drop=True),
                 registry_audit=registry_audit, model=model, corrections_audit=corrections_audit)


def _registry_payments(registry: pd.DataFrame, vendor_kept: pd.DataFrame, panel: Panel):
    """Registry rows as payments; rows repeating a vendor payment, or outside the panel, are set aside.

    A registry row with the record date of a vendor payment but a different amount is a second
    payment on that date (VTBR 2021 has two); it is added and flagged for review.
    """
    td, closures = panel.trading_days, panel.closures
    reg = registry.copy().reset_index(drop=True)
    reg["ex_rule"] = [ex_date_from_record(r, td, closures) for r in reg["record_date"]]
    reg["ex_vendor"] = [_align_inside(d, td) for d in reg["ex_date_declared"]]
    seen: dict[tuple, list[float]] = {}
    for r in vendor_kept.itertuples():
        seen.setdefault((r.secid, r.record_date, r.currency), []).append(round(float(r.amount), 9))
    status, flags = [], []
    for r in reg.itertuples():
        key_date = r.ex_rule if pd.notna(r.ex_rule) else r.ex_vendor
        window = _window_status(key_date, r.ex_date_declared, panel.start, panel.end)
        known = seen.get((r.secid, r.record_date, r.currency))
        flag = ""
        if r.secid not in panel.close.columns:
            st = "not_in_universe"
        elif window is not None:
            st = window
        elif known is not None and any(abs(k - round(float(r.amount), 9)) < 1e-9 for k in known):
            st = "registry_overlap"
        else:
            st = "ok"
            flag = "same_record_date_as_vendor" if known is not None else ""
        status.append(st)
        flags.append(flag)
    reg["status"], reg["source_flags"] = status, flags
    audit = reg[["source_ref", "secid", "record_date", "status"]].copy()
    ok = reg[reg["status"] == "ok"].copy()
    ok["source"] = "registry"
    ok["cbonds_tickers"], ok["cbonds_name"], ok["cbonds_amount"] = None, None, np.nan
    ok["restatement_factor"], ok["n_duplicates"], ok["secid_status"] = 1.0, 0, "registry"
    ok["ex_date_cbonds"] = pd.NaT
    return ok[_PAYMENT_COLUMNS], audit
