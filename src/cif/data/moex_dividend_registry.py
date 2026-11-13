"""Registry of dividend payments that the Cbonds calendar lacks, from official disclosures.

The Cbonds export of 2026-10-08 has no rows for 29 of the 101 tickers of the universe: delisted
and relocated issuers (the plan, section 3, RU). For each payment of such an issuer while it was in
IMOEX this module holds the amount as declared, the currency and basis, the record date, the
ex-date if a source states it, and **the documents the facts were read in**, one entry per payment,
in the manner of :mod:`cif.data.sp500_events`. The loader
(:func:`cif.data.moex_dividends.build_events`) dates and checks these events like the vendor's.

Rules of the registry
---------------------
* A fact is in the registry only if the page that states it was read (``Source.document`` says what
  it said, in one's own words); ``verified_on`` is the day it was read. A payment that could be
  seen only on aggregator sites is not registered: it stays in the list of leads of the report.
* ``Source.kind``: ``issuer`` (the issuer's own pages, reports and press releases), ``regulatory``
  (RNS, SEC and e-disclosure.ru filings), ``exchange`` (MOEX, NSD, LSE notices), ``wire`` (news
  agencies quoting the issuer), ``news`` (other press), ``aggregator`` (smart-lab, dohod.ru,
  Investing.com and the like). ``confidence`` is ``high`` only with an issuer, regulatory or
  exchange source that states both amount and record date.
* The amount is **per traded instrument as declared** (per share; per receipt for GDR/ADS, with
  ``basis`` saying so) in the currency of the declaration. A foreign-currency event is registered
  in its currency; the build converts it to roubles at the official rate of the ex-date and keeps
  the original amount, currency and rate beside the converted one.
* Each event is checked against the ex-day price drop and the index dividend flow when the dataset
  is built; a registered event that the data contradict is listed in the build report, not hidden.
* ``NO_PAYMENT`` holds the evidence that an issuer paid nothing over a span of the window, so that
  "absent from Cbonds" is not confused with "paid and missing".
* ``LEADS`` holds what is known about payments that are **not** in the registry: a record date that
  no page gives, an issuer whose missing flow is below about 1 % of the index dividend flow of its
  window (left out on purpose, with the evidence in hand), a payment seen on aggregator sites only.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

SOURCE_KINDS = ("issuer", "regulatory", "exchange", "wire", "news", "aggregator")
PRIMARY_KINDS = ("issuer", "regulatory", "exchange")
BASES = ("per share", "per GDR", "per ADS", "per RDR")
CONFIDENCE = ("high", "medium", "low")
CURRENCIES = ("RUB", "USD", "EUR", "GBP", "CNY", "HKD", "JPY")
SUPPORTS = ("amount", "record_date", "ex_date", "payment_date", "currency", "decision", "ratio")

#: Universe tickers with no row at all in the Cbonds export of 2026-10-08 (checked in the plan, section 3).
CBONDS_ABSENT = frozenset({
    "AGRO", "DIXY", "DSKY", "ENPG", "EONR", "EPLN", "FIVE", "FIXP", "GLTR", "HHRU", "LENT", "LNTA", "MAIL",
    "MFON", "MRKH", "MTLR", "PGIL", "PHST", "POGR", "POLY", "RNFT", "RSTI", "RUALR", "UGLD", "URKA", "UWGN",
    "VKCO", "VZRZ", "YNDX",
})
#: Tickers that are in the export but whose history there starts late or has holes shown by the
#: index dividend flow; registry events for them are supplements, each with its own note.
CBONDS_GAP_TICKERS = frozenset({"TCSG"})


@dataclass(frozen=True)
class Source:
    """A document a fact was read in."""

    url: str
    document: str  # what the page is and what it says about the payment, in a line or two
    kind: str  # one of SOURCE_KINDS
    supports: tuple[str, ...] = ()  # which facts it states: amount, record_date, ex_date, ...
    via: str = "direct"  # "direct", or "archive" for a web.archive.org copy


@dataclass(frozen=True)
class RegistryEvent:
    """One dividend payment of an issuer that the vendor calendar lacks."""

    secid: str  # the series of the panel that traded on the ex-date
    issuer: str
    period: str  # the financial period the dividend is for, e.g. "FY2015", "H1 2016"
    period_type: str  # in the vocabulary of the vendor: "год", "6 месяцев (I полугодие)", "Внеплановые" ...
    record_date: str  # date of closing of the register (for receipts: the depositary's record date)
    amount: float  # per traded instrument, as declared, gross
    currency: str
    basis: str  # one of BASES
    sources: tuple[Source, ...]
    verified_on: str
    ex_date_declared: str | None = None  # only if a source states it
    payment_date: str | None = None
    confidence: str = "medium"
    note: str = ""


@dataclass(frozen=True)
class NoPayment:
    """Evidence that an issuer paid no dividend over ``scope`` (``"YYYY-MM-DD..YYYY-MM-DD"``)."""

    secid: str
    scope: str
    statement: str
    sources: tuple[Source, ...]
    verified_on: str
    note: str = ""


@dataclass(frozen=True)
class AmountCorrection:
    """A vendor row whose amount is wrong, with the amount as declared and the document it was read in.

    The row is found by ``(secid, record_date, cbonds_amount)`` (the amount as exported); a
    correction that matches no row is an error, so that a new export cannot silently outdate it.
    """

    secid: str
    record_date: str
    cbonds_amount: float  # the amount in the export that is replaced
    amount: float  # per share as declared, in the basis of the ex-date
    reason: str
    sources: tuple[Source, ...]
    verified_on: str
    currency: str = "RUB"
    confidence: str = "medium"


@dataclass(frozen=True)
class Lead:
    """A payment (or a span) that the registry does not hold, with what is known and why it is open."""

    secid: str
    what: str  # the payment as far as it is known: period, amount, basis, dates
    why_open: str  # why it is not an event: "record date not read", "small: mean weight 0.2 %" ...
    sources: tuple[Source, ...]  # the pages that were read and bear on it
    verified_on: str


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------

_SCOPE = re.compile(r"^\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}$")


def _iso(value: str | None) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None


def _source_problems(label: str, sources: Iterable[Source]) -> list[str]:
    out = []
    sources = tuple(sources)
    if not sources:
        out.append(f"{label}: no source")
    for s in sources:
        if not re.match(r"^https?://\S+$", s.url or ""):
            out.append(f"{label}: source url {s.url!r} is not a URL")
        if not (s.document or "").strip():
            out.append(f"{label}: source {s.url!r} says nothing about what the document states")
        if s.kind not in SOURCE_KINDS:
            out.append(f"{label}: source kind {s.kind!r} is not one of {SOURCE_KINDS}")
        for item in s.supports:
            if item not in SUPPORTS:
                out.append(f"{label}: source supports {item!r}, not one of {SUPPORTS}")
    return out


def validate(events: Iterable[RegistryEvent]) -> list[str]:
    """Problems with a registry (empty list if none)."""
    events = list(events)
    problems: list[str] = []
    seen: dict[tuple, int] = {}
    for e in events:
        label = f"{e.secid} {e.record_date} {e.amount:g} {e.currency}"
        if not (e.amount > 0):
            problems.append(f"{label}: amount must be positive")
        if e.currency not in CURRENCIES:
            problems.append(f"{label}: currency {e.currency!r} is not one of {CURRENCIES}")
        if e.basis not in BASES:
            problems.append(f"{label}: basis {e.basis!r} is not one of {BASES}")
        if e.confidence not in CONFIDENCE:
            problems.append(f"{label}: confidence {e.confidence!r} is not one of {CONFIDENCE}")
        if not (e.period_type or "").strip():
            problems.append(f"{label}: period_type is empty")
        record = _iso(e.record_date)
        if record is None:
            problems.append(f"{label}: record_date {e.record_date!r} is not an ISO date")
        for field in ("ex_date_declared", "payment_date"):
            value = getattr(e, field)
            if value is not None and _iso(value) is None:
                problems.append(f"{label}: {field} {value!r} is not an ISO date")
        ex = _iso(e.ex_date_declared)
        if ex is not None and record is not None and ex > record:
            problems.append(f"{label}: ex_date_declared {e.ex_date_declared} is after the record date")
        if _iso(e.verified_on) is None:
            problems.append(f"{label}: verified_on {e.verified_on!r} is not an ISO date")
        problems += _source_problems(label, e.sources)
        kinds = {s.kind for s in e.sources}
        if e.sources and kinds <= {"aggregator"}:
            problems.append(f"{label}: rests on aggregator sources alone (keep it as a lead, not an event)")
        if e.confidence == "high" and not kinds & set(PRIMARY_KINDS):
            problems.append(f"{label}: confidence high needs an issuer, regulatory or exchange source")
        key = (e.secid, e.record_date, e.currency, round(e.amount, 9))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            problems.append(f"{label}: duplicate of another event (same record date, currency and amount)")
    return problems


def validate_no_payment(items: Iterable[NoPayment]) -> list[str]:
    problems: list[str] = []
    for n in items:
        label = f"{n.secid} {n.scope}"
        if not _SCOPE.match(n.scope or "") or any(_iso(part) is None for part in (n.scope or "..").split("..")):
            problems.append(f"{label}: scope must be 'YYYY-MM-DD..YYYY-MM-DD'")
        if not (n.statement or "").strip():
            problems.append(f"{label}: statement is empty")
        if _iso(n.verified_on) is None:
            problems.append(f"{label}: verified_on {n.verified_on!r} is not an ISO date")
        problems += _source_problems(label, n.sources)
    return problems


def validate_leads(items: Iterable[Lead]) -> list[str]:
    problems: list[str] = []
    seen: set[tuple] = set()
    for x in items:
        label = f"lead {x.secid} {x.what[:40]}"
        if not (x.what or "").strip():
            problems.append(f"{label}: what is empty")
        if not (x.why_open or "").strip():
            problems.append(f"{label}: why_open is empty")
        if _iso(x.verified_on) is None:
            problems.append(f"{label}: verified_on {x.verified_on!r} is not an ISO date")
        problems += _source_problems(label, x.sources)
        key = (x.secid, x.what)
        if key in seen:
            problems.append(f"{label}: duplicate lead")
        seen.add(key)
    return problems


def validate_corrections(items: Iterable[AmountCorrection]) -> list[str]:
    problems: list[str] = []
    seen: set[tuple] = set()
    for c in items:
        label = f"correction {c.secid} {c.record_date} {c.cbonds_amount:g}"
        if not c.cbonds_amount > 0:
            problems.append(f"{label}: cbonds_amount must be positive")
        if not c.amount > 0:
            problems.append(f"{label}: amount must be positive")
        elif c.amount == c.cbonds_amount:
            problems.append(f"{label}: amount differs from cbonds_amount by nothing: not a correction")
        if _iso(c.record_date) is None:
            problems.append(f"{label}: record_date {c.record_date!r} is not an ISO date")
        if c.currency not in CURRENCIES:
            problems.append(f"{label}: currency {c.currency!r} is not one of {CURRENCIES}")
        if not (c.reason or "").strip():
            problems.append(f"{label}: reason is empty")
        if c.confidence not in CONFIDENCE:
            problems.append(f"{label}: confidence {c.confidence!r} is not one of {CONFIDENCE}")
        if _iso(c.verified_on) is None:
            problems.append(f"{label}: verified_on {c.verified_on!r} is not an ISO date")
        problems += _source_problems(label, c.sources)
        if c.sources and {s.kind for s in c.sources} <= {"aggregator"}:
            problems.append(f"{label}: rests on aggregator sources alone")
        key = (c.secid, c.record_date, round(c.cbonds_amount, 9))
        if key in seen:
            problems.append(f"{label}: duplicate correction of the same vendor row")
        seen.add(key)
    return problems


# --------------------------------------------------------------------------------------------
# The registry as shipped
# --------------------------------------------------------------------------------------------

_READ_ON = "2026-10-08"  # the day the pages below were read (by a research assistant; a sample re-read by the engineer)
_INTERIM, _FINAL, _SPECIAL = "6 месяцев (I полугодие)", "год", "Внеплановые"
_ALL_FACTS = ("amount", "record_date", "ex_date", "payment_date", "currency")

# ---- Polymetal International plc (POLY): ordinary shares, declared in US dollars per share -----------
# Every RNS was read in the text mirrored on Investegate. London ex-dividend dates are in the notes: MOEX
# follows its own settlement rule (the RNS of 2019-03-13 and 2020-03-04 state the MOEX dates, which equal
# the rule's). The annual totals the RNS state (FY2013 0.09, FY2014 0.41, FY2015 0.51, FY2016 0.42,
# FY2017 0.44, FY2018 0.48, FY2019 0.82, FY2020 1.29 US$ per share) are the sums of the events below.
_INV = "https://www.investegate.co.uk/announcement/"
_POLY = "polymetal-international--poly"
# (period, period type, wording, amount, record date, London ex-date, payment date, RNS date, RNS path, AGM date, AGM path)
_POLY_ROWS = (
    ("H1 2013", _INTERIM, "interim dividend for the first half of 2013", 0.01, "2013-09-06", "2013-09-04",
     "2013-09-26", "2013-08-28", f"rns/{_POLY}/dividend-declaration-of-us-0-01-per-share/3145849", None, None),
    ("FY2013", _FINAL, "final dividend for 2013", 0.08, "2014-05-02", "2014-04-30", "2014-05-30", "2014-03-31",
     f"rns/{_POLY}/dividend-declaration/3670543", "2014-05-21", f"rns/{_POLY}/result-of-agm/3544978"),
    ("H1 2014", _INTERIM, "interim dividend for the first half of 2014", 0.08, "2014-09-05", "2014-09-03",
     "2014-09-26", "2014-08-27", f"rns/{_POLY}/dividend-declaration/3812778", None, None),
    ("FY2014 special", _SPECIAL, "special dividend for 2014", 0.20, "2014-12-19", "2014-12-18", "2015-01-23",
     "2014-12-04", f"rns/{_POLY}/dividend-declaration/3888545", None, None),
    ("FY2014", _FINAL, "final dividend for 2014", 0.13, "2015-05-01", "2015-04-30", "2015-05-29", "2015-03-30",
     f"rns/{_POLY}/dividend-declaration/4065749", "2015-05-20", f"rns/{_POLY}/result-of-agm/4012041"),
    ("H1 2015", _INTERIM, "interim dividend for the first half of 2015", 0.08, "2015-09-04", "2015-09-03",
     "2015-09-25", "2015-08-25", f"rns/{_POLY}/dividend-declaration-of-us-0-08-per-share/4039756", None, None),
    ("FY2015 special", _SPECIAL, "special dividend for 2015", 0.30, "2015-12-11", "2015-12-10", "2015-12-24",
     "2015-12-03", f"rns/{_POLY}/special-dividend-and-approval-of-the-kyzyl-project/4357316", None, None),
    ("FY2015", _FINAL, "final dividend for 2015", 0.13, "2016-05-06", "2016-05-05", "2016-05-27", "2016-03-29",
     f"rns/{_POLY}/final-dividend-declaration/4382379", "2016-05-17", f"rns/{_POLY}/result-of-agm/4629783"),
    ("H1 2016", _INTERIM, "interim dividend for the first half of 2016", 0.09, "2016-09-02", "2016-09-01",
     "2016-09-23", "2016-08-24", f"rns/{_POLY}/dividend-declaration-of-us-0-09-per-share/4590103", None, None),
    ("FY2016 special", _SPECIAL, "special dividend for 2016", 0.15, "2016-12-16", "2016-12-15", "2016-12-29",
     "2016-12-08", f"rns/{_POLY}/special-dividend-of-us-0-15-per-share/4943373", None, None),
    ("FY2016", _FINAL, "final dividend for 2016", 0.18, "2017-05-05", "2017-05-04", "2017-05-26", "2017-03-15",
     f"rns/{_POLY}/final-dividend-and-approval-of-revised-policy/4573133", "2017-05-16",
     f"rns/{_POLY}/results-of-agm/5139108"),
    ("H1 2017", _INTERIM, "interim dividend for the first half of 2017", 0.14, "2017-09-08", "2017-09-07",
     "2017-09-29", "2017-08-29", f"rns/{_POLY}/dividend-declaration-of-us-0-14-per-share/5044624", None, None),
    ("FY2017", _FINAL, "final dividend for 2017", 0.30, "2018-05-11", "2018-05-10", "2018-05-23", "2018-03-12",
     f"rns/{_POLY}/final-dividend-of-us-0-30-per-share/4981761", "2018-04-25",
     f"rns/{_POLY}/result-of-agm-held-on-25-april-2018/5121597"),
    ("H1 2018", _INTERIM, "interim dividend for the first half of 2018", 0.17, "2018-09-07", "2018-09-06",
     "2018-09-28", "2018-08-21", f"rns/{_POLY}/interim-dividend-declaration-of-us-0-17-per-share/5603952", None, None),
    ("FY2018", _FINAL, "final dividend for 2018", 0.31, "2019-05-10", "2019-05-09", "2019-05-24", "2019-03-11",
     f"eqs/{_POLY}/polymetal-international-plc-final-dividend-dec-/5598595", "2019-04-23",
     f"eqs/{_POLY}/results-of-agm/86262"),
    ("H1 2019", _INTERIM, "interim dividend for the first half of 2019", 0.20, "2019-09-06", "2019-09-05",
     "2019-09-27", "2019-08-27", f"eqs/{_POLY}/interim-dividend-declaration-of-us-0-20-per-share/5877385", None, None),
    ("FY2019", _FINAL, "final dividend for 2019", 0.42, "2020-05-11", "2020-05-07", "2020-05-29", "2020-03-04",
     f"eqs/{_POLY}/polymetal-final-dividend-declaration-of-us-0-/53659", "2020-04-27",
     f"eqs/{_POLY}/polymetal-results-of-agm/5915311"),
    ("H1 2020", _INTERIM, "interim dividend for the first half of 2020", 0.40, "2020-09-04", "2020-09-03",
     "2020-09-25", "2020-08-26", f"eqs/{_POLY}/polymetal-interim-dividend-declaration-of-us-/6331520", None, None),
    ("FY2020", _FINAL, "final dividend for 2020", 0.89, "2021-05-07", "2021-05-06", "2021-05-28", "2021-03-03",
     f"eqs/{_POLY}/polymetal-final-dividend-declaration-of-us-0-/6535818", "2021-04-26",
     f"eqs/{_POLY}/polymetal-results-of-agm/6584189"),
    ("H1 2021", _INTERIM, "interim dividend for the first half of 2021", 0.45, "2021-09-10", "2021-09-09",
     "2021-09-30", "2021-08-26", f"eqs/{_POLY}/polymetal-interim-dividend-declaration-of-us-/6637298", None, None),
)
# MOEX dates that the RNS state for the shares traded in Moscow (the others carry the London timetable only)
_POLY_MOEX_EX = {"2019-05-10": "2019-05-08", "2020-05-11": "2020-05-07"}


def _poly_events() -> tuple[RegistryEvent, ...]:
    out = []
    for period, ptype, wording, amount, rec, ex_lse, pay, rns_date, rns_path, agm_date, agm_path in _POLY_ROWS:
        sources = [Source(
            _INV + rns_path,
            f"Polymetal RNS of {rns_date} (text mirrored on Investegate): the board declares the {wording}, "
            f"US${amount:g} per ordinary share; London ex-dividend date {ex_lse}, record date {rec}, payment "
            f"date {pay}", "regulatory", _ALL_FACTS)]
        if agm_path:
            sources.append(Source(
                _INV + agm_path,
                f"Polymetal RNS of {agm_date}: results of the annual general meeting, all resolutions passed "
                "(the final dividend among them)", "regulatory", ("decision",)))
        if rec == "2019-05-10":
            sources.append(Source(
                _INV + f"eqs/{_POLY}/polymetal-international-plc-final-dividend-cal-/5643495",
                "Polymetal RNS of 2019-03-13: the Moscow Exchange timetable of the 2018 final dividend: ex-dividend "
                "date 2019-05-08, record date 2019-05-10, payment date 2019-05-24; differs from the London dates "
                "because of Russian bank holidays", "regulatory", ("ex_date", "record_date", "payment_date")))
        out.append(RegistryEvent(
            secid="POLY", issuer="Polymetal International plc", period=period, period_type=ptype, record_date=rec,
            amount=amount, currency="USD", basis="per share", sources=tuple(sources), verified_on=_READ_ON,
            ex_date_declared=_POLY_MOEX_EX.get(rec), payment_date=pay, confidence="high",
            note=f"London ex-dividend date {ex_lse}; paid in US dollars unless a sterling (from 2018 also euro) "
                 "election was made",
        ))
    # the special dividend of January 2020 (counted in the 2019 total): the declaration RNS could not be read
    out.append(RegistryEvent(
        secid="POLY", issuer="Polymetal International plc", period="FY2019 special", period_type=_SPECIAL,
        record_date="2020-02-14", amount=0.20, currency="USD", basis="per share", verified_on=_READ_ON,
        payment_date="2020-03-05", confidence="low",
        sources=(
            Source(_INV + f"eqs/{_POLY}/polymetal-exchange-rates-for-special-dividend-/5817607",
                   "Polymetal RNS of 2020-02-19: the sterling and euro conversion rates for the special dividend of "
                   "US$0.20 per ordinary share", "regulatory", ("amount", "currency")),
            Source(_INV + f"eqs/{_POLY}/polymetal-preliminary-results-for-the-year-end-/53612",
                   "Polymetal preliminary results for 2019: the special dividend was announced in January 2020, "
                   "about US$94 m, and is part of the 2019 total of US$385 m (US$0.82 per share)", "regulatory",
                   ("amount", "decision")),
            Source("https://gold.1prime.ru/news/20200131/348560.html",
                   "Prime, 2020-01-31: a special dividend of US$0.2 per share, about US$94 m, payable on 2020-03-05; "
                   "no record date", "wire", ("amount", "payment_date")),
            Source("https://www.dividendmax.com/united-kingdom/london-stock-exchange/mining/polymetal-international-plc/"
                   "dividends", "Dividend table of an aggregator: ex-dividend date 2020-02-13, payment date 2020-03-05",
                   "aggregator", ("ex_date", "payment_date")),
        ),
        note="The declaration RNS of 2020-01-31 could not be fetched (403/404) and the Investegate feed has a gap "
             "from 2019-10-03 to 2020-02-05, so the record date was not read on any page. 2020-02-14 is inferred: "
             "one business day after the ex-dividend date 2020-02-13 of the aggregator (the London pattern of "
             "every other payment) and the date a search summary attributed to the RNS.",
    ))
    return tuple(out)


# ---- PGIL (Polyus Gold International Ltd), 2015: the only dividend of the window -------------------------
_PGIL_FINAL = RegistryEvent(
    secid="PGIL", issuer="Polyus Gold International Limited", period="FY2014", period_type=_FINAL,
    record_date="2015-05-15", amount=0.0608, currency="USD", basis="per share", verified_on=_READ_ON,
    payment_date="2015-05-29", confidence="high",
    sources=(
        Source("https://polyus.com/en/media/press-releases/1338/",
               "Polyus press release of 2015-04-10: the board recommends a final dividend for 2014 of 6.08 US cents "
               "per ordinary share, US$184.5 m in total (30 % of adjusted net income), for holders of record at "
               "the close of business on 2015-05-15, payable on 2015-05-29", "issuer",
               ("amount", "record_date", "payment_date", "currency")),
        Source("https://polyus.com/en/media/press-releases/1346/",
               "Polyus press release of 2015-05-15: results of the annual general meeting; the final dividend of "
               "6.08 US cents per share passed with all votes cast", "issuer", ("decision", "amount")),
        Source("https://polyus.com/en/media/press-releases/1363/",
               "Polyus first-half 2015 results: the final dividend for 2014, US$184 m, was announced and paid in the "
               "second quarter of 2015", "issuer", ("payment_date",)),
        Source("https://interfax.com/newsroom/top-stories/37069/",
               "Interfax, 2015-04-10: 6.08 cents per share, US$184.5 m, general meeting on 15 May; it calls 29 May "
               "the record date, which is the payment date on the issuer's page", "wire", ("amount",)),
    ),
    note="No ex-date is stated by the issuer. The special dividend of US$500 m (16.49 cents) at the end of 2014 "
         "lies before the window of PGIL in IMOEX.",
)

# ---- Uralkali (URKA): ordinary shares, declared in RUB per share (1 GDR = 5 shares by the issuer's figures) ----
_URKA = "https://www.uralkali.com/press_center/press_releases/"
_URKA_EVENTS = (
    RegistryEvent(
        secid="URKA", issuer="PJSC Uralkali", period="2013 interim", period_type=_SPECIAL, record_date="2013-10-29",
        amount=2.21, currency="RUB", basis="per share", verified_on=_READ_ON, confidence="low",
        sources=(
            Source(_URKA + "item231013/",
                   "Uralkali board release of 2013-10-23: recommends an interim dividend of RUB 2.21 per share "
                   "(about US$0.35 per GDR), convenes the extraordinary meeting for 2013-12-18 and sets 2013-10-29 "
                   "as the date of the list for taking part in it", "issuer", ("amount", "decision")),
            Source(_URKA + "item14720/",
                   "Uralkali release of 2013-12-18: the extraordinary meeting approves RUB 2.21 per share, about "
                   "RUB 6.5 bn in total, payable within 60 days", "issuer", ("amount", "decision")),
            Source("https://www.vedomosti.ru/newspaper/articles/2013/10/24/vkratce5",
                   "Vedomosti, 2013-10-24: the board recommended about RUB 6.5 bn of dividends for the first half",
                   "news", ("amount",)),
            Source("https://investfunds.ru/stocks/Uralkali/",
                   "Aggregator table: RUB 2.21, ex-dividend date 2013-10-28, record date 2013-10-29", "aggregator",
                   ("record_date", "ex_date")),
        ),
        note="The issuer's releases name no separate dividend record date: 2013-10-29 is the list date for the "
             "meeting, and the aggregator gives the same date as the record date of the dividend (the 2013 "
             "practice tied the two). The period is not named by the issuer (the press says first half, the "
             "aggregator nine months). Per GDR (5 shares) about RUB 11.05.",
    ),
    RegistryEvent(
        secid="URKA", issuer="PJSC Uralkali", period="FY2013", period_type=_FINAL, record_date="2014-06-20",
        amount=1.63, currency="RUB", basis="per share", verified_on=_READ_ON, confidence="high",
        sources=(
            Source(_URKA + "item16382/",
                   "Uralkali board release of 2014-04-10: recommends a final dividend for 2013 of RUB 1.63 per share "
                   "(RUB 8.15 per GDR) for holders registered on 2014-06-20", "issuer",
                   ("amount", "record_date", "currency")),
            Source(_URKA + "item16707/",
                   "Uralkali release of 2014-06-09: the annual meeting approves RUB 1.63 per share; with the "
                   "interim, RUB 11.3 bn for 2013", "issuer", ("decision", "amount")),
            Source("https://1prime.ru/20140609/786419276.html",
                   "Prime, 2014-06-09: the meeting approved the final dividend of RUB 1.63 per share, about RUB "
                   "4.8 bn", "wire", ("amount",)),
        ),
        note="No ex-date and no payment date are stated by the issuer. Per GDR (5 shares) RUB 8.15.",
    ),
)

# ---- MegaFon (MFON): ordinary shares; the issuer states every amount "per ordinary share (or GDR)" ------
_MF = "https://corp.megafon.ru/ai/document/"
_MF_AR2014 = Source(
    _MF + "6808/file/Megafon_AR2014_RU_Approved.pdf",
    "MegaFon annual report for 2014 (Russian): the meeting of 2014-06-30 approved dividends for 2013 of RUB 40 bn, "
    "RUB 64.51 per ordinary share (or GDR), and fixed 2014-07-11 as the list date", "issuer",
    ("amount", "record_date", "decision"))
_MF_AR2015 = Source(
    _MF + "8890/file/megafon_ar2015_rus.pdf",
    "MegaFon annual report for 2015 (Russian): table of the dividends of 2014-2015 with amounts, record and payment "
    "dates, and the decisions of the 2015 meetings; its footnote takes the payment date to be the day of the London "
    "announcement of full payment", "issuer", ("amount", "record_date", "payment_date", "decision"))
_MF_NSD = Source(
    "https://nfksber.ru/upload/iblock/0ff/klaikro0d4ez1gcyem3wxp86tdsrrasc.pdf",
    "Notice of the National Settlement Depository on the 2016-06-30 meeting, republished by a broker: the draft "
    "resolution on RUB 48.38 (final for 2015) and RUB 8.06 (first quarter of 2016) per share for holders registered on "
    "2016-07-11; the date 2016-05-11 in it is the list for taking part in the meeting", "exchange",
    ("amount", "record_date"))
_MF_AR2016 = Source(
    _MF + "9924/file/1_utverxhdenie_go.pdf",
    "MegaFon annual report for 2016 (the file also holds the 2016 meeting papers): the meeting of 2016-06-30 decided "
    "RUB 29,995,600,000 (RUB 48.38 per share) and RUB 4,997,200,000 (RUB 8.06); the meeting of 2016-12-09 RUB 15 bn "
    "(RUB 24.19)", "issuer", ("amount", "decision"))
MFON_EVENTS = (
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="FY2013", period_type=_FINAL, record_date="2014-07-11",
        amount=64.51, currency="RUB", basis="per share", verified_on=_READ_ON, payment_date="2014-08-06",
        confidence="high", sources=(_MF_AR2014, _MF_AR2015),
        note="Total RUB 40 bn; the payment date is the day of the London announcement of full payment. The Q1-2013 "
             "interim of RUB 10.34 (paid 2013-07-22) lies before the window.",
    ),
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="FY2014", period_type=_FINAL, record_date="2015-07-13",
        amount=16.13, currency="RUB", basis="per share", verified_on=_READ_ON, payment_date="2015-08-17",
        confidence="high",
        sources=(
            _MF_AR2015,
            Source(_MF + "8653/file/MegaFon_Press_Release_Q1_2015_ENG.pdf",
                   "MegaFon first-quarter 2015 release: the board of 2015-04-28 recommends RUB 10 bn for 2014 for the "
                   "meeting of 2015-06-30", "issuer", ("amount", "decision")),
            Source(_MF + "8584/file/MegaFon_Press_Release_Q4_2015_ENG_Final.pdf",
                   "MegaFon fourth-quarter 2015 release: the 2015 annual meeting approved the dividend for 2014 and "
                   "it was distributed in August 2015", "issuer", ("decision", "payment_date")),
        ),
        note="Total RUB 10.00 bn.",
    ),
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="9M 2015", period_type="9 месяцев", record_date="2015-12-22",
        amount=64.51, currency="RUB", basis="per share", verified_on=_READ_ON, confidence="high",
        sources=(
            _MF_AR2015,
            Source("https://corp.megafon.ru/download/~federal/Q3_2015/MegaFon_Press_Release_Q3_2015_RUS_final.pdf",
                   "MegaFon third-quarter 2015 release (Russian): the board of 2015-10-28 recommends RUB 40 bn, "
                   "RUB 64.51 per share, for holders registered on 2015-12-22, subject to the meeting of "
                   "2015-12-11", "issuer", ("amount", "record_date", "decision")),
        ),
        note="An additional dividend for nine months. The annual report's table dates the payment 2016-02-03, the day of "
             "the London announcement of full payment; the fourth-quarter release says it was distributed in "
             "December 2015, so no payment date is registered.",
    ),
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="FY2015", period_type=_FINAL, record_date="2016-07-11",
        amount=48.38, currency="RUB", basis="per share", verified_on=_READ_ON, confidence="high",
        sources=(_MF_NSD, _MF_AR2016), note="Total RUB 29,995,600,000.",
    ),
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="Q1 2016", period_type="3 месяца (I кв.)",
        record_date="2016-07-11", amount=8.06, currency="RUB", basis="per share", verified_on=_READ_ON,
        confidence="high", sources=(_MF_NSD, _MF_AR2016),
        note="Approved at the same meeting as the final for 2015; total RUB 4,997,200,000.",
    ),
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="9M 2016", period_type="9 месяцев", record_date="2016-12-20",
        amount=24.19, currency="RUB", basis="per share", verified_on=_READ_ON, confidence="high",
        sources=(
            Source(_MF + "9664/file/informaciya_dlya_fizlic_nalogoobloxhenie.pdf",
                   "MegaFon note to individual shareholders: if the extraordinary meeting of 2016-12-09 approves the "
                   "interim dividend, the list date is 2016-12-20 and payment is due by 2017-01-31", "issuer",
                   ("record_date", "payment_date")),
            Source(_MF + "9817/file/MegaFon_IR_Presentation_Mar_2017_eng_final.pdf",
                   "MegaFon investor presentation of March 2017: the meeting of 2016-12-09 approved RUB 15 bn, "
                   "RUB 24.19 per share; the total for 2016 is RUB 50 bn (RUB 80.63)", "issuer",
                   ("amount", "decision")),
            _MF_AR2016,
            Source("https://interfax.com/newsroom/top-stories/30309/",
                   "Interfax, 2016-10-27: RUB 24.19 per share, RUB 15 bn, meeting on 9 December; it names 7 November "
                   "as the register date, which is probably the list for the meeting", "wire", ("amount",)),
        ),
        note="The record date rests on the issuer's conditional statement (if approved), approval is in the later "
             "documents. Interfax's 7 November contradicts it and is taken for the meeting's list.",
    ),
    RegistryEvent(
        secid="MFON", issuer="PJSC MegaFon", period="FY2016", period_type=_FINAL, record_date="2017-07-11",
        amount=32.25, currency="RUB", basis="per share", verified_on=_READ_ON, confidence="low",
        sources=(
            Source(_MF + "10619/file/MegaFon_Annual_report_2017_RUS.pdf",
                   "MegaFon annual report for 2017 (Russian): the meeting of 2017-06-30 decided RUB 19,995,000,000 "
                   "(RUB 32.25 per share) for the fourth quarter of 2016; fully paid in August 2017", "issuer",
                   ("amount", "decision")),
            Source("https://smart-lab.ru/q/MFON/dividend/",
                   "Aggregator table: 4Q 2016, RUB 32.25, record date 2017-07-11, last day with the dividend "
                   "2017-07-07", "aggregator", ("record_date",)),
        ),
        note="The record date is on an aggregator only (no issuer page read gives it); the date 2017-06-07 in the "
             "depository's notice is the list for the meeting.",
    ),
)

# ---- Detsky Mir (DSKY): ordinary shares, 739,000,000 votes; voting reports published by the issuer ------
# (period, type, amount, total RUB, meeting kind, meeting date, record date, report URL)
_DM = "http://ir.detmir.ru/wp-content/uploads/"
_DSKY_ROWS = (
    ("FY2016", _FINAL, 3.48, 2_571_720_000, "annual", "2017-06-28", "2017-07-17", _DM + "2019/03/otchet_ob_itogah_golosovaniya_28.06.17.pdf"),
    ("9M 2017", "9 месяцев", 2.97, 2_194_830_000, "extraordinary", "2017-12-04", "2017-12-15", _DM + "2019/05/Отчет-об-итогах-голосования-1.pdf"),
    ("FY2017", _FINAL, 3.88, 2_867_320_000, "annual", "2018-05-18", "2018-05-29", _DM + "2019/03/otchet_ob_itogah_golosovaniya_18.05.18.pdf"),
    ("9M 2018", "9 месяцев", 4.39, 3_244_210_000, "extraordinary", "2018-12-15", "2018-12-26", _DM + "2019/03/otchet_ob_itogah_golosovaniya.pdf"),
    ("FY2018", _FINAL, 4.45, 3_288_550_000, "annual", "2019-05-16", "2019-05-27", "https://ir.detmir.ru/wp-content/uploads/2019/05/Отчет-об-итогах-голосования-2.pdf"),
    ("9M 2019", "9 месяцев", 5.06, 3_739_340_000, "extraordinary", "2019-12-06", "2019-12-17", "https://ir.detmir.ru/wp-content/uploads/2019/12/Отчет-об-итогах-голосования.pdf"),
    ("FY2019", _FINAL, 3.00, 2_217_000_000, "annual", "2020-06-30", "2020-07-11", "https://ir.detmir.ru/wp-content/uploads/2020/07/Отчет-об-итогах-голосования.pdf"),
    ("FY2019 additional", _SPECIAL, 2.50, 1_847_500_000, "extraordinary", "2020-09-18", "2020-09-29", _DM + "2020/09/Отчет-об-итогах-голосованияws.pdf"),
    ("9M 2020", "9 месяцев", 5.08, 3_754_120_000, "extraordinary", "2020-12-17", "2020-12-28", _DM + "2020/12/Отчет-об-итогах-голосования-публ.pdf"),
    ("FY2020", _FINAL, 6.07, 4_485_730_000, "annual", "2021-06-30", "2021-07-11", _DM + "2021/07/отчет-об-итогах-голосования.pdf"),
    ("9M 2021", "9 месяцев", 5.20, 3_842_800_000, "extraordinary", "2021-12-15", "2021-12-26", _DM + "2021/12/Отчет-обитогах-голосования-WS.pdf"),
)


def _dsky_events() -> tuple[RegistryEvent, ...]:
    out = []
    for period, ptype, amount, total, kind, meeting, rec, url in _DSKY_ROWS:
        out.append(RegistryEvent(
            secid="DSKY", issuer="PJSC Detsky Mir", period=period, period_type=ptype, record_date=rec, amount=amount,
            currency="RUB", basis="per share", verified_on=_READ_ON, confidence="high",
            sources=(Source(
                url, f"Report on the voting results of the {kind} general meeting of {meeting}, published on the "
                     f"issuer's investor site: a dividend of RUB {amount:.2f} per share, RUB {total:,} in total "
                     f"(739,000,000 voting shares), for holders on the list of {rec}", "issuer",
                ("amount", "record_date", "decision")),),
            note="Ex-date and payment date are not stated in the report.",
        ))
    return tuple(out)


# ---- X5 Retail Group (FIVE): GDRs, 1 GDR = 0.25 ordinary share; declared in RUB per GDR, paid to GDR holders in USD
_X5 = "https://x5.com/wp-content/uploads/2025/10/"
_X5_ROWS = (
    # period, type, RUB per GDR, record, ex, payment, AGM date, minutes, extra note
    ("FY2017", _FINAL, 79.5, "2018-05-25", "2018-05-24", "2018-06-22", "2018-05-10", "180510_agm_minutes.pdf",
     "RUB 318.00 per ordinary share, RUB 21,590 m in total, the first dividend in the company's history; paid to the "
     "depositary by 2018-06-01 and to GDR holders in USD by 2018-06-22"),
    ("FY2018", _FINAL, 92.06, "2019-05-24", "2019-05-23", None, "2019-05-10", "190510_agm_minutes.pdf",
     "RUB 368.23 per ordinary share, RUB 25,000 m in total; paid in USD within 45 days of the meeting. The adoption "
     "line of this item was not captured in the minutes read; all other items show adoption"),
    ("FY2019", _FINAL, 110.47, "2020-05-29", "2020-05-28", None, "2020-05-12", "200512_agm_minutes.pdf",
     "RUB 30,000 m in total, RUB 10,493 m of it from retained earnings under IFRS 16; paid in USD within 45 days of "
     "the meeting. The adoption line of this item was not captured in the minutes read"),
    ("FY2020", _FINAL, 110.49, "2021-05-28", "2021-05-27", None, "2021-05-12", "210512_agm_minutes.pdf",
     "The remainder of 2020 after the interim of RUB 73.64: RUB 184.13 per GDR for the year, 66.7 % above 2019; paid "
     "in USD within 45 days of the meeting. The adoption line of this item was not captured in the minutes read"),
)


def _x5_events() -> tuple[RegistryEvent, ...]:
    out = []
    for period, ptype, amount, rec, ex, pay, agm, minutes, note in _X5_ROWS:
        out.append(RegistryEvent(
            secid="FIVE", issuer="X5 Retail Group N.V.", period=period, period_type=ptype, record_date=rec,
            amount=amount, currency="RUB", basis="per GDR", verified_on=_READ_ON, ex_date_declared=ex,
            payment_date=pay, confidence="high" if period == "FY2017" else "medium",
            sources=(Source(
                _X5 + minutes, f"Minutes of the annual general meeting of {agm} (issuer's site): dividend of RUB "
                               f"{amount:g} per GDR, ex-dividend date {ex}, record date {rec}", "issuer",
                ("amount", "record_date", "ex_date", "decision")),),
            note=note,
        ))
    return tuple(out)


# ---- TCS Group Holding PLC (TCSG): GDRs, 1 GDR = 1 class A share; declared in US dollars per share or GDR ----
# The issuer's own announcements (RNS of the London Stock Exchange) could not be read, so the five payments of the
# GDR era (2020-21) rest on news agencies and the business press that quote the board's decision: none is "high".
# The Russian entity's payment of 2024-11-25 (RUB 92.5 per share) is in the Cbonds calendar.
_TCS = "TCS Group Holding PLC"
_INTERIM_OTHER = "промежуточные"
TCSG_EVENTS = (
    RegistryEvent(
        secid="TCSG", issuer=_TCS, period="1st 2020 interim (4Q 2019 results)", period_type=_INTERIM_OTHER,
        record_date="2020-03-27", amount=0.21, currency="USD", basis="per GDR", verified_on=_READ_ON,
        payment_date="2020-03-30", confidence="medium",
        sources=(
            Source("https://1prime.ru/finance/20200311/831053683.html",
                   "Prime, 2020-03-11: the board of TCS Group approved the first interim dividend for 2020, US$0.21 per "
                   "share or GDR, about US$42 m; shareholders on the register of 2020-03-27 are entitled (the list is drawn "
                   "up on 2020-03-26); payment to shareholders about 2020-03-30, to GDR holders within five business days",
                   "wire", ("amount", "record_date", "payment_date", "decision")),
            Source("https://www.kommersant.ru/doc/4284113",
                   "Kommersant, 2020-03-11: the same decision, US$0.21 per share or GDR; no record date printed", "news",
                   ("amount",)),
        ),
        note="The issuer's RNS could not be read. GDR holders are paid up to five business days after the shareholders.",
    ),
    RegistryEvent(
        secid="TCSG", issuer=_TCS, period="2nd 2020 interim (1Q 2020 results)", period_type=_INTERIM_OTHER,
        record_date="2020-05-29", amount=0.14, currency="USD", basis="per GDR", verified_on=_READ_ON,
        payment_date="2020-06-01", confidence="low",
        sources=(
            Source("https://www.rbc.ru/quote/news/article/5ebba2619a7947d518e0d798",
                   "RBC, 2020-05-13: US$0.14 per security, US$28 m in total; the last day to buy shares with the dividend is "
                   "2020-05-27, the register closes after it, payment after 1 June", "news", ("amount", "payment_date")),
            Source("https://financialit.net/news/banking/tinkoff-reports-1q-20-results-announces-2nd-2020-interim-dividend",
                   "Reprint of the company's release of 2020-05-13 on a trade site: record date 2020-05-29, ex-dividend date "
                   "2020-05-28, payment indicatively 2020-06-01, GDR holders three to five business days later", "news",
                   ("record_date", "ex_date", "payment_date")),
        ),
        note="No page that was read prints the record date in the board's own words: 2020-05-29 is the date of the reprint "
             "and follows from the last day to buy (2020-05-27) under T+2.",
    ),
    RegistryEvent(
        secid="TCSG", issuer=_TCS, period="3rd 2020 interim (2Q 2020 results)", period_type=_INTERIM_OTHER,
        record_date="2020-08-21", amount=0.20, currency="USD", basis="per GDR", verified_on=_READ_ON,
        payment_date="2020-08-24", confidence="medium",
        sources=(
            Source("https://1prime.ru/finance/20200806/831878124.html",
                   "Prime, 2020-08-06: US$0.20 per share or GDR, about US$39.9 m; shareholders on the register of 2020-08-21 "
                   "are entitled; payment about 2020-08-24, GDR holders about five business days later", "wire",
                   ("amount", "record_date", "payment_date", "decision")),
            Source("https://www.rbc.ru/quote/news/article/5f2bcd049a7947bacd45766a",
                   "RBC, 2020-08-06: the same dividend; the last day to buy is 2020-08-19, payment 2020-08-24", "news",
                   ("amount", "payment_date")),
        ),
        note="The issuer's RNS could not be read.",
    ),
    RegistryEvent(
        secid="TCSG", issuer=_TCS, period="4th 2020 interim (3Q 2020 results)", period_type=_INTERIM_OTHER,
        record_date="2020-11-27", amount=0.25, currency="USD", basis="per GDR", verified_on=_READ_ON, confidence="medium",
        sources=(
            Source("https://www.vedomosti.ru/finance/articles/2020/11/12/846723-gruppa-tinkoff-narastila-pribil-pochti-vdvoe-vishe-prognozov",
                   "Vedomosti, 2020-11-12: the board recommended US$0.25 per security, the register closes on 2020-11-27, "
                   "US$49.8 m in total (about 30 % of net income)", "news", ("amount", "record_date", "decision")),
            Source("https://www.vedomosti.ru/finance/news/2022/05/18/922620-tcs-group-priostanovila-viplati-dividendov",
                   "Vedomosti, 2022-05-18: TCS Group paid US$0.80 per GDR for 2020, the sum of the four interim dividends",
                   "news", ("amount",)),
        ),
        note="The payment date was not found. The issuer's RNS could not be read.",
    ),
    RegistryEvent(
        secid="TCSG", issuer=_TCS, period="1st 2021 interim (4Q 2020 results)", period_type=_INTERIM_OTHER,
        record_date="2021-03-26", amount=0.24, currency="USD", basis="per GDR", verified_on=_READ_ON,
        payment_date="2021-03-29", confidence="medium",
        sources=(
            Source("https://www.interfax.ru/business/755452",
                   "Interfax, 2021-03-11: US$0.24 per share or GDR, about US$48 m; shareholders on the register of "
                   "2021-03-26 are entitled (the list is drawn up on 2021-03-25); payment about 2021-03-29, GDR holders "
                   "about five business days later", "wire", ("amount", "record_date", "payment_date", "decision")),
            Source("https://www.rbc.ru/quote/news/article/6049c5ad9a79478c8d8561e7",
                   "RBC, 2021-03-11: the same dividend; the register is compiled on 2021-03-25, payment about 2021-03-29",
                   "news", ("amount", "payment_date")),
        ),
        note="Interfax gives the register of 2021-03-26 and the list of 2021-03-25, RBC only the 25th: the later date is "
             "registered (an aggregator's ex-dividend date, 2021-03-25, fits it).",
    ),
)

# ---- Globaltrans Investment PLC (GLTR): GDRs, 1 GDR = 1 share; declared in roubles, paid in US dollars -------
_GL = "https://www.globaltrans.com/investors/news/detail/"
GLTR_EVENTS = (
    RegistryEvent(
        secid="GLTR", issuer="Globaltrans Investment PLC", period="FY2020 final and special final", period_type=_FINAL,
        record_date="2021-04-29", amount=28.0, currency="RUB", basis="per GDR", ex_date_declared="2021-04-28",
        verified_on=_READ_ON, confidence="high",
        sources=(Source(
            _GL + "2222", "Globaltrans announcement (issuer's investor site): final and special final dividends for 2020, RUB "
                          "5.0 bn, RUB 28.00 per share/GDR; the GDRs are marked ex-dividend on 2021-04-28; the shareholder "
                          "record date is 2021-04-29, the day of the annual general meeting; paid in US dollars within 30 "
                          "business days of the meeting, to GDR holders about three business days later", "issuer",
            ("amount", "record_date", "ex_date", "decision")),),
        note="With the interim dividends already paid, RUB 74.55 per share/GDR for 2020 (RUB 13.3 bn). The split between the "
             "final and the special final is not given.",
    ),
    RegistryEvent(
        secid="GLTR", issuer="Globaltrans Investment PLC", period="H1 2021 interim and special interim", period_type=_INTERIM,
        record_date="2021-09-07", amount=22.5, currency="RUB", basis="per GDR", ex_date_declared="2021-09-03",
        verified_on=_READ_ON, confidence="high",
        sources=(Source(
            _GL + "2277", "Globaltrans announcement of 2021-08-31 (issuer's investor site): the board approved interim and "
                          "special interim dividends for the first half of 2021, RUB 4.0 bn, RUB 22.50 per share/GDR (about "
                          "US$0.3047 at RUB 73.8506); the GDRs are marked ex-dividend on 2021-09-03 (as set by the London "
                          "Stock Exchange), the shareholder record date is 2021-09-07; paid in US dollars not later than "
                          "2021-09-10", "issuer", ("amount", "record_date", "ex_date", "decision")),),
        note="The ex-date stated for the GDRs (a Friday) precedes the shareholders' record date (the following Tuesday) by "
             "two sessions, one more than the usual cycle; the build tests it against the price.",
    ),
)

# ---- HeadHunter Group PLC (HHRU): ADS on Nasdaq, 1 ADS = 1 ordinary share; declared in US dollars per share ------
_SEC = "https://www.sec.gov/Archives/edgar/data/1721181/"
HHRU_EVENTS = (
    RegistryEvent(
        secid="HHRU", issuer="HeadHunter Group PLC", period="interim 2021 (2020 profit)", period_type=_INTERIM_OTHER,
        record_date="2021-06-09", amount=0.55, currency="USD", basis="per ADS", verified_on=_READ_ON, confidence="high",
        sources=(Source(
            _SEC + "000110465921072697/tm2117688d1_ex99-1.htm",
            "HeadHunter press release of 2021-05-27 (Form 6-K, exhibit 99.1): the board approved an interim dividend of "
            "US$0.55 per share, about 75 % of the 2020 adjusted net income; record date 2021-06-09, payable on or before "
            "2021-07-16", "regulatory", ("amount", "record_date", "payment_date", "currency", "decision")),),
        note="One ADS is one ordinary share (Form 20-F for 2021). The ex-date is not stated; a Russian withholding tax of "
             "15 % applies by default.",
    ),
    RegistryEvent(
        secid="HHRU", issuer="HeadHunter Group PLC", period="interim 2021 (declared 2022-01-13)", period_type=_INTERIM_OTHER,
        record_date="2022-01-24", amount=0.84, currency="USD", basis="per ADS", verified_on=_READ_ON, confidence="high",
        sources=(Source(
            _SEC + "000110465922003815/tm222860d1_ex99-1.htm",
            "HeadHunter press release of 2022-01-13 (Form 6-K, exhibit 99.1): the board approved an interim dividend of "
            "US$0.84 per share; record date 2022-01-24, payable on or before 2022-02-10", "regulatory",
            ("amount", "record_date", "payment_date", "currency", "decision")),),
        note="The results for 2021 (2022-03-04) put the payment at RUB 3,150 m, made in February 2022.",
    ),
)

# ---- Fix Price Group (FIXP): GDRs, 1 GDR = 1 ordinary share; declared in roubles ------------------------------
_FP = "https://ir.fix-price.com/"
FIXP_EVENTS = (
    RegistryEvent(
        secid="FIXP", issuer="Fix Price Group", period="H1 2021 interim", period_type=_INTERIM,
        record_date="2021-09-24", amount=11.5, currency="RUB", basis="per GDR", ex_date_declared="2021-09-23",
        payment_date="2021-09-28", verified_on=_READ_ON, confidence="high",
        sources=(
            Source(_FP + "investors/dividends/",
                   "Fix Price investor relations, dividends table: H1 2021, RUB 11.5 per GDR, RUB 9,788 m in total, record "
                   "date 24.09.2021, ex-dividend date 23.09.2021, payment date 28.09.2021", "issuer", _ALL_FACTS),
            Source(_FP + "upload/iblock/323/Fix%20Price%20announces%20key%20operating%20and%20financial%20results%20for%206M"
                         "%202021_vF%20(ENG)_clean.pdf",
                   "Fix Price press release of 2021-08-12: the board recommended an interim dividend for H1 2021 of RUB 11.5 "
                   "per GDR, 100 % of the H1 IFRS net profit", "issuer", ("amount", "decision")),
        ),
        note="The final dividend for 2021 (RUB 6.8 per GDR, record date 2022-05-27) was declared, postponed and cancelled: "
             "it is not an event (see the no-payment evidence).",
    ),
)

# ---- Ros Agro PLC (AGRO): GDRs, 5 GDRs = 1 ordinary share; declared and paid in US dollars -------------------------
# Board releases on the issuer's investor site; the issuer states the ex-dividend date, the day before the record date.
_RA = "https://www.rusagrogroup.ru/investors/news-events/press-releases/single-view/article/"
# (period, type, US$ per GDR, record date, ex-date, release date, article, meeting, note)
_AGRO_ROWS = (
    ("1H 2017", _INTERIM, 0.23, "2017-09-14", "2017-09-13", "2017-08-14", 715, "extraordinary meeting of 2017-09-14",
     "RUB 13.84 per GDR (RUB 69.20 per share), US$1.15 per share, US$30.9 m in total; payable on or before 2017-09-30"),
    ("FY2017", _FINAL, 0.29, "2018-04-13", "2018-04-12", "2018-03-19", 779, "annual meeting of 2018-04-13",
     "US$1.45 per share, RUB 4,085,839,292.90 in total; payable on or before 2018-04-20"),
    ("1H 2018", _INTERIM, 0.12, "2018-09-14", "2018-09-13", "2018-08-13", 847, "extraordinary meeting of 2018-09-14",
     "US$0.60 per share, RUB 1,026,425,986.25 (US$16.1 m), a payout ratio of 44.3 %; payable on or before 2018-09-28"),
    ("1H 2019", _INTERIM, 0.19, "2019-09-13", "2019-09-12", "2019-08-19", 914, "extraordinary meeting of 2019-09-13",
     "US$0.95 per share, RUB 1,683,841,245.27 (US$25.6 m); payable on or before 2019-10-15"),
    ("FY2019", _FINAL, 0.32, "2020-04-17", "2020-04-16", "2020-03-10", 953, "annual meeting of 2020-04-17",
     "US$1.60 per share; RUB 4,533,124,875.56 for 2019 less the interim, US$43.05 m outstanding; payable on or before "
     "2020-05-29, paid on 2020-04-27 (release 960)"),
    ("FY2020", _FINAL, 1.04, "2021-04-16", "2021-04-15", "2021-03-15", 1026, "annual meeting of 2021-04-16",
     "US$5.20 per share, US$139.9 m outstanding after the interim of US$25.6 m; payable on or before 2021-05-27"),
    ("1H 2021", _INTERIM, 0.89, "2021-09-10", "2021-09-09", "2021-08-16", 1070, "extraordinary meeting of 2021-09-10",
     "US$4.45 per share, US$119.7 m (50 % of the comprehensive income of the half-year); payable on or before 2021-10-21"),
)


def _agro_events() -> tuple[RegistryEvent, ...]:
    out = []
    for period, ptype, amount, rec, ex, released, article, meeting, extra in _AGRO_ROWS:
        out.append(RegistryEvent(
            secid="AGRO", issuer="Ros Agro PLC", period=period, period_type=ptype, record_date=rec, amount=amount,
            currency="USD", basis="per GDR", ex_date_declared=ex, verified_on=_READ_ON, confidence="high",
            payment_date="2020-04-27" if period == "FY2019" else None,
            sources=(Source(
                f"{_RA}{article}/", f"Board release of {released} (issuer's investor site): US${amount:.2f} gross per GDR "
                                    f"(5 GDRs = 1 share), to be approved at the {meeting}; record date {rec}, ex-dividend "
                                    f"date {ex}; {extra}", "issuer", ("amount", "record_date", "ex_date", "decision")),),
            note="Paid in US dollars at the rate of the Central Bank of Russia on the payment day; treasury GDRs are excluded.",
        ))
    # two payments whose figures the issuer's releases do not give per GDR
    out.append(RegistryEvent(
        secid="AGRO", issuer="Ros Agro PLC", period="FY2015", period_type=_FINAL, record_date="2016-05-27", amount=0.58,
        currency="USD", basis="per GDR", ex_date_declared="2016-05-26", verified_on=_READ_ON, confidence="low",
        sources=(
            Source(_RA + "501/", "Notice of the annual general meeting of 2016-05-26 (issuer's investor site, dated "
                                 "2016-04-25): 30 % of 2015 net income, RUB 7,107,100,800 in total; after the interim, RUB "
                                 "5,306,141,964.27 or US$77,535,953.51 to distribute; the amount per share/GDR is fixed on the "
                                 "record date 2016-05-27 (ex-dividend date 2016-05-26)", "issuer",
                   ("record_date", "ex_date", "decision")),
            Source("https://www.interfax.ru/quote/AGRO/Dividends", "Interfax quotes page: 1Q 2016 row, US$0.58 per GDR",
                   "aggregator", ("amount",)),
        ),
        note="The amount per GDR is not in the issuer's notice (it is fixed on the record date); 0.58 is a quotes page's figure.",
    ))
    out.append(RegistryEvent(
        secid="AGRO", issuer="Ros Agro PLC", period="1H 2020", period_type=_INTERIM, record_date="2020-09-18", amount=0.19,
        currency="USD", basis="per GDR", payment_date="2020-09-21", verified_on=_READ_ON, confidence="low",
        sources=(
            Source(_RA + "1026/", "Board release of 2021-03-15 (issuer's investor site): the interim dividends for 2020 were "
                                  "US$25,560,956.88 in total; the final dividend for 2019 of US$43,050,032.64 was US$0.32 per "
                                  "GDR (release 953), so the GDRs entitled number 134.5 m and the interim was US$0.19 per GDR",
                   "issuer", ("amount",)),
            Source("https://investfuture.ru/dividends/ragr", "Aggregator table: record date 2020-09-18", "aggregator",
                   ("record_date",)),
        ),
        note="The release of this interim dividend was not found: the amount per GDR is derived from the total, the record date "
             "is an aggregator's, the payment date (2020-09-21) is that of the annual report of 2020.",
    ))
    return tuple(out)


REGISTRY_EVENTS: tuple[RegistryEvent, ...] = (
    *_poly_events(), _PGIL_FINAL, *_URKA_EVENTS, *MFON_EVENTS, *_dsky_events(), *_x5_events(), *TCSG_EVENTS,
    *GLTR_EVENTS, *HHRU_EVENTS, *FIXP_EVENTS, *_agro_events(),
)

# ---- No-payment evidence added with the second batch of issuers (research assistants' readings of 2026-10-08) ---
_RN = "https://www.russneft.ru/"
_UWGN_AR = ("https://www.uniwagon.com/media/documents/2020_%D0%93%D0%BE%D0%B4%D0%BE%D0%B2%D0%BE%D0%B9_%D0%BE%D1%82%D1%87%D0%B5"
            "%D1%82_%D0%9D%D0%9F%D0%9A_%D0%9E%D0%92%D0%9A.pdf")
_ENP = "https://enplusgroup.com/"
_VK_FILE = Source("https://corp.vkcdn.ru/media/files/istoriya-dividendnyih-vyiplat.pdf",
                  "VK's history of dividend payments (issuer's file): no decision on dividends for 2021-2025", "issuer",
                  ("decision",))
_NO_PAYMENT_2: tuple[NoPayment, ...] = (
    NoPayment(
        secid="RNFT", scope="2017-09-22..2019-12-19",
        statement="RussNeft paid no dividend on its ordinary shares (RNFT): the annual meetings of 2018 and 2019 decided that "
                  "dividends on ordinary shares are not paid, and the issuer's table lists none for 2020-2025. Only the "
                  "cumulative preferred shares, which do not trade as RNFT, were paid (US$0.40803 per share, record dates "
                  "2018-07-12 and 2019-07-11).",
        sources=(
            Source(_RN + "upload/iblock/2de/tcqqieb003bfsjii5dz74tsko82786df.pdf",
                   "RussNeft press release on the annual meeting of 2018-06-22: the dividend on preferred shares is paid, "
                   "dividends on ordinary shares are not", "issuer", ("decision",)),
            Source(_RN + "upload/iblock/836/3u0e7ibgzmslr5wun6r9uqzgiddijshi.docx",
                   "Decisions of the annual meeting of 2019-06-21: dividends on preferred shares, none on ordinary shares",
                   "issuer", ("decision",)),
            Source(_RN + "eng/shareholders/dividends/",
                   "RussNeft dividends page: ordinary-share dividends for 2020-2025 neither declared nor paid", "issuer",
                   ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The lists of material facts of 2018-2019 were read in part only; the issuer's statements above are the evidence.",
    ),
    NoPayment(
        secid="UWGN", scope="2016-09-16..2018-12-20",
        statement="OVK (UWGN) paid no dividend before 2020: the annual report for 2020 says the company paid none for the "
                  "previous years; the press reported the decisions of the meetings of 2017 and 2018 not to pay for 2016 and "
                  "2017.",
        sources=(
            Source(_UWGN_AR, "OVK annual report for 2020, p. 117 (issuer): the company has not paid dividends for the "
                             "previous years", "issuer", ("decision",)),
            Source("https://interfax.com/newsroom/top-stories/27538/",
                   "Interfax, 2017-07-06: shareholders kept the RAS profit of 2016 (RUB 314 m); no dividend", "wire",
                   ("decision",)),
            Source("https://www.vedomosti.ru/finance/articles/2018/07/06/774794-chi-dividendi-bolshe",
                   "Vedomosti, 2018-07-06: holders of ordinary shares receive nothing for 2017", "news", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The resolutions of the meetings of 2017 and 2018 were not read; the annual report's statement covers the window.",
    ),
    NoPayment(
        secid="PHST", scope="2013-09-02..2016-06-15",
        statement="Pharmstandard paid no cash dividend: the annual meetings of 2013 and 2014 decided not to pay, the "
                  "annual reports of 2015 and 2016 list no dividend among the decisions of the meetings of 2015 and 2016, "
                  "and the audited IFRS statements show none paid to the shareholders of the parent in 2014-2016. The "
                  "spin-off of OTCPharm shares (2013-14) was a distribution of shares, not a dividend.",
        sources=(
            Source("https://pharmstd.com/archivedetails_64_2679.html",
                   "Pharmstandard news: the annual meeting of 2013-05-24 decided not to pay dividends for 2012", "issuer",
                   ("decision",)),
            Source("https://pharmstd.com/archivedetails_64_2720.html",
                   "Pharmstandard news: the annual meeting of 2014-06-27 decided not to pay dividends for 2013", "issuer",
                   ("decision",)),
            Source("https://pharmstd.com/pdfs/2015_annual_report_EN.pdf",
                   "Annual report for 2015, p. 74: the decisions of the annual meeting of 2015-05-29 (accounts, profit "
                   "distribution, boards, auditors, charter) include no dividend", "issuer", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The result page of the meeting of 2015-05-29 was not found; the annual reports and the IFRS statements are the "
             "indirect evidence for 2014-2016.",
    ),
    NoPayment(
        secid="ENPG", scope="2021-12-17..2025-12-31",
        statement="EN+ GROUP paid no dividend after 2018: its dividend history lists none for 2018-2025 and the annual "
                  "meetings of 2022, 2023, 2024 and 2025 resolved not to distribute profit.",
        sources=(
            Source(_ENP + "en/investors/shareholders/dividends/",
                   "EN+ GROUP dividends page: payments listed up to 2017; no dividend decision for 2018-2025", "issuer",
                   ("decision",)),
            Source(_ENP + "upload/iblock/bc8/RNS-re-AGM-2022-results-_En_.pdf",
                   "Results of the annual meeting of 2022-06-23: no distribution of profit for 2021", "regulatory",
                   ("decision",)),
            Source(_ENP + "upload/iblock/83a/0s8y3kl0f9hfcmgzt1dxzyk7vszqfiym/AGM-Report-on-voting-results_29-June-2023_signed.pdf",
                   "Report on the voting results of the annual meeting of 2023-06-29: no dividends for 2022", "issuer",
                   ("decision",)),
            Source(_ENP + "upload/iblock/829/2nitg429sudpvofgtbaw5q5gt8ca0rtq/_-_-_-_-_AGM-Report-on-voting-results_26-June-2024.pdf",
                   "Report on the voting results of the annual meeting of 2024-06-26: no dividends for 2023", "issuer",
                   ("decision",)),
            Source(_ENP + "upload/iblock/044/dvyioy4ss8ndbc63tr59h3nhuo8051gt/AGM-Report-on-voting-results_25-June-2025.pdf",
                   "Report on the voting results of the annual meeting of 2025-06-25: no dividends for 2024", "issuer",
                   ("decision",)),
        ),
        verified_on=_READ_ON,
    ),
    NoPayment(
        secid="VKCO", scope="2021-12-14..2025-12-31",
        statement="VK took no decision to pay dividends for 2021-2025.", sources=(_VK_FILE,), verified_on=_READ_ON,
        note="The file's metadata give 2026-08-10 as its creation date.",
    ),
    NoPayment(
        secid="MAIL", scope="2021-01-01..2021-12-13",
        statement="Mail.ru Group (the predecessor of VK) took no decision to pay dividends for 2021; VK's file lists none.",
        sources=(_VK_FILE,), verified_on=_READ_ON,
        note="PARTIAL: the window starts on 2020-09-18 and the span to 2020-12-31 is not covered (the archived version of "
             "the file that would list 2020 could not be opened).",
    ),
    NoPayment(
        secid="YNDX", scope="2014-09-16..2024-03-21",
        statement="Yandex paid no dividend before the first one of the Russian entity (the extraordinary meeting of "
                  "2024-09-09, RUB 80 per share, record date 2024-09-20): Interfax calls it the first in the company's "
                  "history, Kommersant the first since the one for 2010, which precedes the window.",
        sources=(
            Source("https://www.interfax.ru/business/981586",
                   "Interfax, 2024-09-11: the first dividends in the corporate history; inside Yandex N.V. the Russian company "
                   "made no payments to shareholders", "wire", ("decision",)),
            Source("https://www.kommersant.ru/doc/7064129",
                   "Kommersant, 2024-09-11: the first dividend since the one paid for 2010", "news", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="Press only: the dividend section of the Form 20-F could not be opened.",
    ),
    NoPayment(
        secid="POGR", scope="2020-09-18..2021-09-01",
        statement="Petropavlovsk declared no dividend with its results for 2020 (2021-05-17) or for the first half of 2021 "
                  "(2021-09-01).",
        sources=(
            Source("https://petropavlovskplc.com/wp-content/uploads/2021/05/FY-2020-Financials-Results-RNS_vFINAL.pdf",
                   "Results for 2020: no dividend declared or recommended; a dividend policy is to follow the strategy "
                   "due in the third quarter of 2021", "regulatory", ("decision",)),
            Source("https://petropavlovskplc.com/wp-content/uploads/2021/09/H1-2021-Financials-Results_final.pdf",
                   "Results for the first half of 2021 (slides): no dividend", "regulatory", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="PARTIAL: the window runs to 2022-06-29; the results for 2021 and the meetings of 2021-22 were not read.",
    ),
    NoPayment(
        secid="MRKH", scope="2013-09-02..2013-12-30",
        statement="Rosseti (then IDGC Holding) paid nothing on its ordinary shares in the window of MRKH: the press says the "
                  "holding never paid on ordinary shares before 2016 (its only earlier payout, for 2012, was on preferred "
                  "shares).",
        sources=(
            Source("https://www.vedomosti.ru/business/articles/2016/02/29/631719-rosseti-mogut-vpervie-istorii-viplatit-dividendi",
                   "Vedomosti, 2016-02-29: the holding and its predecessor never paid on ordinary shares and paid nothing "
                   "from 2013", "news", ("decision",)),
            Source("https://1prime.ru/20170530/827510979.html",
                   "Prime, 2017-05-30: no dividends were paid for 2013-2015", "wire", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="Press only; the same issuer trades as RSTI from 2014-01-06 (its payments are in the leads).",
    ),
    NoPayment(
        secid="TCSG", scope="2021-03-27..2024-11-21",
        statement="TCS Group paid nothing between the dividend for the fourth quarter of 2020 (record date 2021-03-26) and "
                  "the payment of the Russian entity declared in November 2024: the press names the last declaration as "
                  "March 2021 and the last payout as that for 4Q 2020.",
        sources=(
            Source("https://interfax.com/newsroom/top-stories/105238/",
                   "Interfax, 2024-08-20: the last dividend declaration was in March 2021", "wire", ("decision",)),
            Source("https://www.rbc.ru/quote/news/article/67066e499a794772b0034936",
                   "RBC, 2024-10-09: the last payout was for the fourth quarter of 2020", "news", ("decision",)),
            Source("https://1prime.ru/20240523/tinkoff-848458542.html",
                   "Prime, 2024-05-23: the board of the Russian entity recommended no dividend for 2023", "wire",
                   ("decision",)),
        ),
        verified_on=_READ_ON,
        note="Press only. The payment of 2024-11 (RUB 92.5 per share) is in the Cbonds calendar.",
    ),
    NoPayment(
        secid="GLTR", scope="2021-09-08..2024-10-11",
        statement="Globaltrans paid nothing after the interim of September 2021: it suspended the final dividend for 2021 "
                  "(the annual meeting of 2022-04-27 resolved not to distribute), the results of 2022-2024 say the dividends "
                  "stay suspended, and on 2024-08-30 the board ended the dividend policy.",
        sources=(
            Source(_GL + "2343", "Globaltrans announcement on the annual meeting of 2022-04-27: no final dividend for 2021",
                   "issuer", ("decision",)),
            Source(_GL + "2459", "Globaltrans results for 2023 (2024-04-08): dividends remain suspended", "issuer",
                   ("decision",)),
            Source(_GL + "2476", "Globaltrans results for the first half of 2024 (2024-08-30): the board terminated the "
                                 "dividend policy", "issuer", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="A special interim dividend (announced 2025-04-18) lies after the index window.",
    ),
    NoPayment(
        secid="FIXP", scope="2021-09-25..2023-09-21",
        statement="Fix Price paid nothing after the interim of September 2021: the final dividend for 2021 (RUB 6.8 per GDR, "
                  "RUB 5.8 bn, record date 2022-05-27) was declared, postponed and cancelled when the board suspended the "
                  "dividend policy on 2022-09-16; the next payment has the record date 2024-01-26.",
        sources=(
            Source(_FP + "investors/dividends/",
                   "Fix Price investor relations: the dividends table lists nothing between H1 2021 (paid 2021-09-28) and "
                   "the interim for 2023-24 (record date 2024-01-26); the board suspended the dividend policy on 2022-09-16",
                   "issuer", ("decision",)),
            Source(_FP + "upload/iblock/02e/FIXP_Financial%20results%20for%2012M%202021%20press%20release_ENG.pdf",
                   "Fix Price results for 2021 (2022-02-28): the board recommended a final dividend of RUB 5.8 bn, RUB 6.8 per "
                   "GDR, record date 2022-05-27 (later cancelled)", "issuer", ("amount", "record_date")),
        ),
        verified_on=_READ_ON,
    ),
    NoPayment(
        secid="AGRO", scope="2021-09-11..2024-12-02",
        statement="Ros Agro paid nothing after the interim of September 2021: the final dividend for 2021 (US$1.02 per GDR, "
                  "record date 2022-04-01) was recommended on 2022-02-28 and not paid, and the meetings of 2023 and 2024 "
                  "decided not to distribute profit.",
        sources=(
            Source("https://www.interfax.ru/business/894110",
                   "Interfax, 2023-04-03: the board approved the dividend for 2021 in February 2022 and decided in March not "
                   "to pay; the last dividend was the interim of 2021, US$119.7 m", "wire", ("decision",)),
            Source("https://www.investegate.co.uk/announcement/rns/ros-agro-plc--agro/agm-statement/8125577",
                   "RNS of 2024-04-08: the annual meeting of 2024-04-05 resolved not to distribute profit as dividends",
                   "regulatory", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The recommendation of 2022-02-28 was read on a news page (MarketScreener); the RNS of 2022-03-10 returned 403.",
    ),
)

NO_PAYMENT: tuple[NoPayment, ...] = (
    *_NO_PAYMENT_2,
    NoPayment(
        secid="MTLR", scope="2013-09-02..2025-06-27",
        statement="Mechel paid no dividend on its ordinary shares: the official dividend table lists the last one "
                  "(RUB 8.06 per share for 2011, decided on 2012-06-29) and states that no decision to pay on "
                  "ordinary shares was taken for 2012 to 2024; the annual meetings of 2024 and 2025 and the 20-F "
                  "data for 2017-2021 confirm it. Only the preferred shares (MTLRP) were paid.",
        sources=(
            Source("https://mechel.ru/upload/PDF%20%D0%9C%D0%B5%D1%87%D0%B5%D0%BB/%D0%A2%D0%B0%D0%B1%D0%BB%D0%B8%D1%86%D0%B0"
                   "%20%D0%BF%D0%BE%20%D0%B4%D0%B8%D0%B2%D0%B8%D0%B4%D0%B5%D0%BD%D0%B4%D0%B0%D0%BC%20%D0%BD%D0%B0%20%D1%81%D0"
                   "%B0%D0%B9%D1%82_2025.pdf",
                   "Mechel's dividend table (version after the 2025 meeting): ordinary shares, last dividend for 2011; "
                   "a note says no decision to pay was taken for 2012-2024", "issuer", ("decision",)),
            Source("https://mechel.ru/press/releases/mechel-soobshchaet-o-resheniyakh-godovogo-obshchego-sobraniya-"
                   "aktsionerov2025/",
                   "Mechel release on the annual meeting of 2025-06-27: no dividends on any share category for 2024",
                   "issuer", ("decision",)),
            Source("https://mechel.ru/press/releases/mechel-reports-decisions-of-annual-general-meeting-of-"
                   "shareholders01072024/",
                   "Mechel release on the annual meeting of 2024-06-28: no dividends on ordinary or preferred shares "
                   "for 2023", "issuer", ("decision",)),
            Source("https://www.sec.gov/Archives/edgar/data/1302362/000119312522126835/d237462d20f.htm",
                   "Mechel Form 20-F for 2021, selected financial data: dividends declared per common share nil for "
                   "2017-2021", "regulatory", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="Interim declarations were not checked quarter by quarter; the issuer's table is the evidence.",
    ),
    NoPayment(
        secid="UGLD", scope="2024-12-20..2025-06-30",
        statement="The board recommended and the annual meeting of 2025-06-30 decided not to pay dividends for 2024; "
                  "no earlier interim declaration was found.",
        sources=(
            Source("https://ugold.ru/tpost/0755elnip1-rekomendatsii-soveta-direktorov-k-godovo",
                   "Issuer, 2025-05-30: a majority of the board recommends no dividends for 2024; meeting on "
                   "2025-06-30", "issuer", ("decision",)),
            Source("https://www.interfax.ru/business/1028792",
                   "Interfax, 2025-05-30: the same recommendation; the date 2025-06-06 it names is the list date for "
                   "the meeting", "wire", ("decision",)),
            Source("https://www.vedomosti.ru/business/news/2025/07/03/1121879-aktsioneri-yuzhuralzolota-reshili",
                   "Vedomosti, 2025-07-03: the shareholders decided not to pay dividends for 2024", "news",
                   ("decision",)),
        ),
        verified_on=_READ_ON,
        note="Interim declarations for 2025 were not excluded by an explicit statement (the disclosure feed was not "
             "read): the evidence covers the decisions up to the meeting of 2025-06-30 only.",
    ),
    NoPayment(
        secid="URKA", scope="2014-06-21..2017-10-22",
        statement="Uralkali paid no dividend after the final for 2013: the interim of RUB 2.96 recommended in October "
                  "2014 was not approved by the extraordinary meeting of 2014-12-26; the meetings of 2015 and 2016 "
                  "decided not to pay for 2014 and 2015; the board recommended none for 2016.",
        sources=(
            Source(_URKA + "item20355/",
                   "Uralkali release of 2014-12-26: the extraordinary meeting did not approve the interim dividend "
                   "of RUB 2.96 per share", "issuer", ("decision",)),
            Source(_URKA + "item22791/",
                   "Uralkali release of 2015-06-15: the annual meeting decided not to pay dividends for 2014",
                   "issuer", ("decision",)),
            Source(_URKA + "item33917/",
                   "Uralkali release of 2016-06-17: the annual meeting decided not to pay dividends for 2015",
                   "issuer", ("decision",)),
            Source(_URKA + "item36741/",
                   "Uralkali board release of 2017-04-12: recommends no dividend for 2016 (the meeting's result of "
                   "2017-06-20 was not found)", "issuer", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The interim of RUB 2.96 per share (record date proposed 2015-01-15) was declared by the board and "
             "REJECTED: aggregators list it as paid. The outcome of the 2017 annual meeting was not read.",
    ),
    NoPayment(
        secid="POLY", scope="2021-09-11..2024-04-21",
        statement="Polymetal paid nothing after the interim of September 2021: the final dividend for 2021 "
                  "(US$0.52, declared 2022-03-02) was withdrawn at the annual meeting of 2022-04-25 and cancelled; no "
                  "dividend for the first half of 2022, for 2022, or for the first half of 2023.",
        sources=(
            Source(_INV + f"eqs/{_POLY}/polymetal-results-of-agm/7041211",
                   "Polymetal RNS of 2022-04-25: results of the annual general meeting, the final-dividend "
                   "resolution withdrawn", "regulatory", ("decision",)),
            Source(_INV + f"eqs/{_POLY}/polymetal-half-year-report-for-the-six-month-e-/7419349",
                   "Polymetal half-year report of 2022-09-22: the dividend for 2021 is permanently cancelled and no "
                   "interim for 2022 is proposed", "regulatory", ("decision",)),
            Source(_INV + f"eqs/{_POLY}/polymetal-preliminary-results-for-the-year-end-/7440486",
                   "Polymetal preliminary results of 2023-03-16: the board decided not to propose a dividend for "
                   "2022", "regulatory", ("decision",)),
            Source("https://interfax.com/newsroom/top-stories/95997/",
                   "Interfax, 2023-10-31: the last dividend paid was the interim for the first half of 2021",
                   "wire", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="After the window: the annual meeting of 2024-06-11 approved no dividend for 2023 (Interfax).",
    ),
    NoPayment(
        secid="DIXY", scope="2013-09-02..2018-01-21",
        statement="DIXY Group paid no dividend: the annual meetings of 2014-06-27 and 2015-06-11 decided not to "
                  "declare any for 2013 and 2014, and Interfax wrote in May 2018 that the company had never paid its "
                  "shareholders a dividend (no dividend for 2016 either).",
        sources=(
            Source("https://nfksber.ru/upload/iblock/b3e/hh3fehti2brafi5wtgj2wfdsx4ximgq9.pdf",
                   "Depository notice republished by a broker: the annual meeting of 2014-06-27, the board "
                   "recommended not to pay dividends for 2013", "exchange", ("decision",)),
            Source("https://nfksber.ru/upload/iblock/893/e9afrqvle43pfyv2ic9xdcrso9ncyj9a.pdf",
                   "Depository notice republished by a broker: the annual meeting of 2015-06-11, the board "
                   "recommended not to pay dividends for 2014", "exchange", ("decision",)),
            Source("https://interfax.com/newsroom/top-stories/23801/",
                   "Interfax, 2018-05-21: no dividend for 2016, RUB 132.5 m of RAS profit reinvested, and the company "
                   "has never paid a dividend to its shareholders", "wire", ("decision",)),
            Source("https://www.vedomosti.ru/finance/articles/2016/07/29/651021-kakie-kompanii-viplatyat-investoram-"
                   "samie-visokie-dividendi",
                   "Vedomosti, 2016-07-29: names DIXY among seven companies whose holders received no dividend for "
                   "2015", "news", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The resolutions of the meetings of 2016 and 2017 were not read; the grade B statements cover them.",
    ),
    NoPayment(
        secid="LNTA", scope="2015-08-16..2019-02-22",
        statement="Lenta had never paid a dividend by February 2019; management said dividends might be considered "
                  "from 2020, depending on a change of corporate structure in 2019.",
        sources=(Source("https://interfax.com/newsroom/top-stories/21228/",
                        "Interfax, 2019-02-22: Lenta has never paid dividends; they may be considered from 2020",
                        "wire", ("decision",)),),
        verified_on=_READ_ON,
        note="One wire report and no primary document: the RNS of the issuer returned 403. The span from 2019-02-23 to "
             "the end of the window (2020-04-19, the FY2019 decision) is not covered.",
    ),
    NoPayment(
        secid="MFON", scope="2017-07-12..2018-08-29",
        statement="MegaFon paid no dividend after the final for 2016: the annual meeting of 2018-06-29 decided not to "
                  "distribute profit for 2017 or declare dividends, and the extraordinary meeting of 2018-08-17 "
                  "dealt with loans only.",
        sources=(
            Source("https://nfksber.ru/upload/iblock/2ad/bjngkddhj5155ycdexaebfse00xp6cf4.pdf",
                   "Depository notice republished by a broker: the annual meeting of 2018-06-29 decided not to "
                   "distribute the profit of 2017 and not to declare dividends", "exchange", ("decision",)),
            Source("https://nfksber.ru/upload/iblock/774/us7vodmv3y32ijy2nzs48fu9q8jwswno.pdf",
                   "Depository notice republished by a broker: the extraordinary meeting of 2018-08-17, agenda of "
                   "loan and indemnity agreements", "exchange", ("decision",)),
        ),
        verified_on=_READ_ON,
    ),
    NoPayment(
        secid="DSKY", scope="2021-12-27..2023-04-16",
        statement="Detsky Mir paid nothing after the interim for nine months of 2021: the annual meetings of "
                  "2022-06-30 and 2023-06-30 decided not to distribute profit for 2021 and 2022, and no "
                  "extraordinary meeting in 2022 dealt with dividends.",
        sources=(
            Source("http://ir.detmir.ru/wp-content/uploads/2022/07/Отчет-об-итогах-голосования.pdf",
                   "Report on the voting results of the annual meeting of 2022-06-30 (issuer's site): profit for 2021 "
                   "not distributed, no dividends", "issuer", ("decision",)),
            Source("http://ir.detmir.ru/wp-content/uploads/2023/07/ОТЧЕТ-ОБ-ИТОГАХ-ГОЛОСОВАНИЯ-ПАО-Детский-мир.pdf",
                   "Report on the voting results of the annual meeting of 2023-06-30 (issuer's site): profit for 2022 "
                   "not distributed, no dividends", "issuer", ("decision",)),
        ),
        verified_on=_READ_ON,
        note="The 2023 meeting is after the end of the window; it is the formal answer for 2022.",
    ),
)

LEADS: tuple[Lead, ...] = (
    # ---- Rosseti (RSTI; MRKH before 2014): ordinary shares, RUB. Documented, left out as small (mean weight 0.18 %) ----
    Lead(
        secid="RSTI",
        what="Q1 2016 interim dividend on ordinary shares, RUB 0.00831813 per share (RUB 1.785 bn with the preferred)",
        why_open="the record date was not read on any page (2016-07-19 is a search-engine lead) and neither was the vote of "
                 "the meeting of 2016-06-30; small",
        sources=(
            Source("https://www.rosseti-sz.ru/en/companynews/russiangrids/news78042.html",
                   "Release on the annual meeting of 2016-06-30 (a subsidiary's site): the board proposal of RUB 0.00831813 per "
                   "ordinary share; no record date, the vote to be disclosed by 2016-07-05", "issuer", ("amount",)),
            Source("https://1prime.ru/20170530/827510979.html",
                   "Prime, 2017-05-30: the interim for the first quarter of 2016, RUB 1.785 bn, was paid", "wire",
                   ("decision",)),
        ),
        verified_on=_READ_ON),
    Lead(
        secid="RSTI",
        what="FY2016 final on ordinary shares, RUB 0.00621446 per share, record date 2017-07-19 (RUB 2 bn in all)",
        why_open="small: left out under the 1 % rule; the facts are documented (meeting of 2017-06-30 and the board's per-share "
                 "amounts), no exchange notice was found",
        sources=(
            Source("https://rosseti-sz.ru/en/companynews/russiangrids/news78055.html",
                   "Release of 2017-06-30 (a subsidiary's site): the meeting approved RUB 2 bn; the list of persons entitled "
                   "is drawn up on 2017-07-19", "issuer", ("record_date", "decision")),
            Source("https://1prime.ru/20170530/827510979.html",
                   "Prime, 2017-05-30: the board proposed RUB 0.00621446 per ordinary share and RUB 0.368355281 per "
                   "preferred share; the register closes on 19 July", "wire", ("amount", "record_date")),
        ),
        verified_on=_READ_ON),
    Lead(
        secid="RSTI",
        what="Q1 2018 interim on ordinary shares, RUB 0.011965 per share, record date 2018-07-09",
        why_open="small: left out under the 1 % rule; the notice of the National Settlement Depository was read",
        sources=(Source("https://nfksber.ru/upload/iblock/6a1/vuofnmxj81cv2t973ezfivhrjbk7y9nx.pdf",
                        "NSD notice republished by a broker: dividends for 3 months of 2018, ordinary 0.011965 RUB, record "
                        "date (дата фиксации) 2018-07-09, payment to nominees 2018-07-23", "exchange",
                        ("amount", "record_date", "payment_date")),),
        verified_on=_READ_ON),
    Lead(
        secid="RSTI",
        what="Q1 2019 interim on ordinary shares, RUB 0.02443 per share, record date 2019-07-08",
        why_open="small: left out under the 1 % rule; the notice of the National Settlement Depository was read",
        sources=(Source("https://nfksber.ru/upload/iblock/ab8/n38okug4fllkehe7qea5mji2ftf449lb.pdf",
                        "NSD notice republished by a broker: dividends for 3 months of 2019, ordinary 0.02443 RUB, record date "
                        "2019-07-08, payment to nominees 2019-07-22", "exchange", ("amount", "record_date", "payment_date")),),
        verified_on=_READ_ON),
    Lead(
        secid="RSTI",
        what="FY2019 final on ordinary shares, RUB 0.0885155625 per share, record date 2020-06-15",
        why_open="small: left out under the 1 % rule; the notice of the National Settlement Depository was read",
        sources=(Source("https://nfksber.ru/upload/iblock/540/43d6zsi66uuohoh8ddy5n2i9fxlz4z3i.pdf",
                        "NSD notice republished by a broker: dividends for 12 months of 2019, ordinary 0.0885155625 RUB, "
                        "record date 2020-06-15, payment 2020-06-29 and 2020-07-20", "exchange",
                        ("amount", "record_date", "payment_date")),),
        verified_on=_READ_ON),
    Lead(
        secid="RSTI",
        what="FY2020 final on ordinary shares, RUB 0.02453341692 per share, record date 2021-07-17 (a Saturday)",
        why_open="small: left out under the 1 % rule; the notice of the National Settlement Depository was read",
        sources=(Source("https://nfksber.ru/upload/iblock/233/yqczx11xlc14o8a0ri7jyfgagc32g5gi.pdf",
                        "NSD notice republished by a broker: dividends for 12 months of 2020, ordinary 0.02453341692 RUB, "
                        "record date 2021-07-17, payment 2021-07-30 and 2021-08-20", "exchange",
                        ("amount", "record_date", "payment_date")),),
        verified_on=_READ_ON),
    # ---- Ros Agro (AGRO): payments of the window that no issuer page read gives per GDR --------------------------------
    Lead(
        secid="AGRO",
        what="1H 2016 interim, paid 2016-09-26 (about US$0.23 per GDR on aggregator tables; RUB 2.00 bn in the IFRS of 2016)",
        why_open="the board's release was not found; the record date (2016-09-23) and the amount rest on aggregators; small",
        sources=(Source("https://ar2020.rusagrogroup.ru/en/corporate-governance/relations-shareholders-investors",
                        "Annual report 2020, dividends table: the first-half 2016 payment is dated 2016-09-26", "issuer",
                        ("payment_date",)),),
        verified_on=_READ_ON),
    Lead(
        secid="AGRO",
        what="FY2016 final, RUB 4.62 bn, record date 2017-04-13, paid 2017-04-19 (US$0.55 or 0.58 per GDR on two aggregators)",
        why_open="the board's release was not found; the per-GDR amount is unsettled between aggregators; small",
        sources=(Source("https://ar2020.rusagrogroup.ru/en/corporate-governance/relations-shareholders-investors",
                        "Annual report 2020, dividends table: the second-half 2016 payment is dated 2017-04-19", "issuer",
                        ("payment_date",)),),
        verified_on=_READ_ON),
    Lead(
        secid="AGRO",
        what="FY2018 final, paid 2019-05-16 (RUB 4.48 bn for 2018 less the interim; about US$0.36-0.39 per GDR on aggregators)",
        why_open="the board's release was not found; the record date is in conflict between aggregators (2019-04-26, "
                 "or ex-date 2019-04-30); small",
        sources=(Source("https://ar2020.rusagrogroup.ru/en/corporate-governance/relations-shareholders-investors",
                        "Annual report 2020, dividends table: the second-half 2018 payment is dated 2019-05-16", "issuer",
                        ("payment_date",)),),
        verified_on=_READ_ON),
    # ---- X5 (FIVE): two interim dividends of the GDRs ------------------------------------------------------------------
    Lead(
        secid="FIVE",
        what="interim dividend for 9M 2020, RUB 73.64 per GDR (with the final of RUB 110.49, RUB 184.13 for 2020); and a "
             "second one for 9M 2021, RUB 73.65 (declared 2021-11-25, record date 2021-12-17, ex-date 2021-12-16: a "
             "search-engine summary only)",
        why_open="no page with the record dates was read (the price and the index flow suggest 2020-12-18 for the first); the "
                 "two together carry under 1 % of the index flow of the window of FIVE; the minutes of the meetings of "
                 "2022-24, which would show that nothing followed, were not read",
        sources=(Source("https://x5.com/wp-content/uploads/2025/10/210512_agm_minutes.pdf",
                        "Minutes of the annual meeting of 2021-05-12: the final dividend for 2020 of RUB 110.49 per GDR "
                        "follows the interim of RUB 73.64", "issuer", ("amount",)),),
        verified_on=_READ_ON),
)


CORRECTIONS: tuple[AmountCorrection, ...] = (
    AmountCorrection(
        secid="BANE", record_date="2014-06-23", cbonds_amount=410.0, amount=211.0,
        reason="The export's year row for 2013 is the sum of the interim dividend (199, record date 2013-11-05, "
               "a row of its own) and the final one (211). The final payment alone is what went ex on "
               "2014-06-20: the export shows 211 for the preferred share BANEP on the same record date, the "
               "ex-day price of BANE fell by 2.4 % against a yield of 16.9 % on 410, and the index dividend "
               "flow of the day is 40 % below what 410 would make.",
        sources=(Source(
            "https://energyland.info/analitic-show-122704",
            "Energy trade-press page dated 2014-06-11: the annual shareholder meeting approved dividends for "
            "2013 of RUB 211 per ordinary and per preferred share, RUB 37.48 bn in total, with the record "
            "date 2014-06-23",
            "news", ("amount", "record_date")),),
        verified_on="2026-10-08", confidence="medium",
    ),
)


def registry_frame(events: Iterable[RegistryEvent] | None = None) -> pd.DataFrame:
    """The registry as the table :func:`cif.data.moex_dividends.build_events` reads.

    Raises :class:`ValueError` listing every problem of an invalid registry.
    """
    events = list(REGISTRY_EVENTS if events is None else events)
    problems = validate(events)
    if problems:
        raise ValueError("invalid dividend registry:\n  " + "\n  ".join(problems))
    cols = ["secid", "record_date", "ex_date_declared", "amount", "currency", "period", "period_type",
            "source_ref", "basis", "confidence", "n_sources", "best_source_kind", "issuer", "payment_date",
            "verified_on", "note"]
    rank = {k: i for i, k in enumerate(("issuer", "regulatory", "exchange", "wire", "news", "aggregator"))}
    rows = []
    for e in sorted(events, key=lambda x: (x.secid, x.record_date, x.amount)):
        best = min((s.kind for s in e.sources), key=lambda k: rank[k])
        rows.append((e.secid, pd.Timestamp(e.record_date), pd.Timestamp(e.ex_date_declared) if e.ex_date_declared
                     else pd.NaT, float(e.amount), e.currency, e.period, e.period_type,
                     f"registry:{e.secid}:{e.record_date}:{e.amount:g}:{e.currency}", e.basis, e.confidence,
                     len(e.sources), best, e.issuer, pd.Timestamp(e.payment_date) if e.payment_date else pd.NaT,
                     e.verified_on, e.note))
    frame = pd.DataFrame(rows, columns=cols)
    frame["ex_date_declared"] = pd.to_datetime(frame["ex_date_declared"])
    frame["payment_date"] = pd.to_datetime(frame["payment_date"])
    return frame



def leads_frame(items: Iterable[Lead] | None = None) -> pd.DataFrame:
    """The leads as a table (one row per lead, the urls joined with `` ; ``); raises on an invalid list."""
    items = list(LEADS if items is None else items)
    problems = validate_leads(items)
    if problems:
        raise ValueError("invalid leads:\n  " + "\n  ".join(problems))
    cols = ["secid", "what", "why_open", "n_sources", "urls", "verified_on"]
    rows = [(x.secid, x.what, x.why_open, len(x.sources), " ; ".join(src.url for src in x.sources), x.verified_on)
            for x in sorted(items, key=lambda y: (y.secid, y.what))]
    return pd.DataFrame(rows, columns=cols)


def corrections_frame(items: Iterable[AmountCorrection] | None = None) -> pd.DataFrame:
    """The corrections as the table :func:`cif.data.moex_dividends.build_events` reads."""
    items = list(CORRECTIONS if items is None else items)
    problems = validate_corrections(items)
    if problems:
        raise ValueError("invalid amount corrections:\n  " + "\n  ".join(problems))
    cols = ["secid", "record_date", "cbonds_amount", "amount", "currency", "reason", "source_ref", "confidence",
            "n_sources", "verified_on"]
    rows = [(c.secid, pd.Timestamp(c.record_date), float(c.cbonds_amount), float(c.amount), c.currency, c.reason,
             f"correction:{c.secid}:{c.record_date}:{c.cbonds_amount:g}", c.confidence, len(c.sources),
             c.verified_on) for c in items]
    return pd.DataFrame(rows, columns=cols)
