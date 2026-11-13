"""Spin-off days of the S&P 500 panel that were verified against filings (adjcorr-2).

Yahoo books a spin-off as a ratio that is no share-count ratio, and on the day the return of
``Close`` and of ``Adj Close`` is that of the remainder of the company, not that of the holder
(``cif.data.sp500_pit.pseudo_split_inventory``). Audit D-2 (report
``docs/reviews/2026-10-08-audit-e0-additions.md``) found six such days with a return beyond 10 %
in magnitude that no rule of ``sp500_pit`` handles. Each of them is registered here with the terms
of the transaction as the company filed them, the sources, and the prices that fix the units of the
data; :func:`cif.data.sp500_pit.apply_verified_events` turns a registered event into the holder's
total return on the day,

    ``(k * P1 + sum_i n_i * C_i) / P0 - 1``,

with ``P0`` the as-traded close of the parent before the day, ``P1`` its as-traded close on the
day, ``k`` the number of parent shares one old share has become (``1/2`` after a 1-for-2 reverse
split), ``n_i`` the shares of the distributed company ``i`` per old parent share and ``C_i`` its
as-traded close on the day.

Units. Yahoo's ``Close`` divides every earlier close by the ratios it registers, and also by splits
that fall after the end of the cached window and so are absent from ``Stock Splits`` (DD: a 1-for-3
reverse split on 2026-06-24 multiplies every earlier ``Close`` by three; CTVA: a ratio of 6.665
booked on 2026-10-01, after the cache was downloaded). Such a split is declared in
``later_splits_unlisted``; a :class:`PriceCheck` against a number printed in a filing guards the
unit of every series the correction reads, so a rebuild on data that disagree stops instead of
producing a wrong return.

A distributed company without a Yahoo series (Exelis, Care Capital Properties, Apartment Income
REIT: no longer listed) is priced by ``Distribution.documented_close``, with the basis and the
source (a close, or the volume weighted average price the issuer used for tax purposes).
"""

from __future__ import annotations

from dataclasses import dataclass

_SEC = "https://www.sec.gov/Archives/edgar/data"


@dataclass(frozen=True)
class Source:
    """A document the terms or a price were read in."""

    url: str
    document: str  # what it is and what it says, in a line
    supports: tuple[str, ...]  # "ratio", "dates", "reverse split", "prices", ...
    kind: str = "sec"  # "sec" (EDGAR), "company" (issuer's own page), "news"


@dataclass(frozen=True)
class Distribution:
    """A company handed out to the holders of the parent.

    ``price`` is ``"yahoo"`` (its ``Close`` on the day, from the panel or the evidence tickers) or
    ``"documented"`` (``documented_close``: a close, or a volume weighted average price when
    ``documented_basis`` is ``"vwap"``).
    """

    ticker: str
    name: str
    shares_per_share: float  # shares of this company per share of the parent held before the day
    price: str = "yahoo"
    documented_close: float | None = None
    documented_basis: str = "close"
    documented_source: str = ""
    later_splits_unlisted: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class PriceCheck:
    """The as-traded close of ``ticker`` on ``date`` must lie in ``[low, high]``.

    ``basis`` says what the range is (a printed close with its rounding, the average of the day's
    high and low with a tolerance, the high and low of a month); ``source`` is where it was read.
    """

    ticker: str
    date: str
    low: float
    high: float
    basis: str
    source: str


@dataclass(frozen=True)
class VerifiedEvent:
    """A spin-off (with or without a reverse split of the parent) read in the filings."""

    ticker: str
    name: str
    date: str  # first trading day on which the parent trades without the entitlement (Yahoo's day)
    yahoo_ratio: float  # the ratio Yahoo registers on that day; the event was verified against it
    parent_shares_per_share: float  # k
    distributions: tuple[Distribution, ...]
    record_date: str
    distribution_date: str
    sources: tuple[Source, ...]
    verified_on: str
    later_splits_unlisted: tuple[tuple[str, float], ...] = ()
    price_checks: tuple[PriceCheck, ...] = ()
    kind: str = "spin_off"
    note: str = ""


# --------------------------------------------------------------------------------------------
# The six days of audit D-2
# --------------------------------------------------------------------------------------------

_RBJ = "https://rbj.net/2011/11/01/shares-of-itt-new-spinoffs-climb"
_ITT_8K = f"{_SEC}/0000216228/000095012311095346/y93291e8vk.htm"
_ITT_10K = f"{_SEC}/0000216228/000119312512089618/d261204d10k.htm"
_EXELIS_10K = f"{_SEC}/0001524471/000119312512097004/d270994d10k.htm"

ITT = VerifiedEvent(
    ticker="ITT",
    name="ITT Corporation: separation of Xylem and Exelis, 1-for-2 reverse split",
    date="2011-11-01",
    yahoo_ratio=3.0156815440289506,
    parent_shares_per_share=0.5,
    distributions=(
        Distribution("XYL", "Xylem Inc.", 1.0),
        Distribution(
            "XLS", "Exelis Inc.", 1.0, price="documented", documented_close=11.70,
            documented_basis="close", documented_source=_RBJ,
        ),
    ),
    record_date="2011-10-17",
    distribution_date="2011-10-31",
    sources=(
        Source(
            _ITT_8K,
            "ITT Corp, Form 8-K (Items 2.01, 5.03), report date 2011-10-31: one Exelis and one "
            "Xylem share for each ITT share held on 2011-10-17, distribution at 12:01 a.m. ET on "
            "2011-10-31, one-for-two reverse split effective 7:00 p.m. ET on 2011-10-31",
            ("ratio", "dates", "reverse split"),
        ),
        Source(
            _ITT_10K,
            "ITT Corp, Form 10-K FY2011, Item 5: close of 2011-10-31 $91.20 on the basis adjusted "
            "for the 1-for-2 split ($45.60 as traded); openings of 2011-11-01 (first regular-way "
            "day): ITT $17.02, Exelis $10.33, Xylem $25.60",
            ("prices", "trading start"),
        ),
        Source(
            _EXELIS_10K,
            "Exelis Inc., Form 10-K FY2011, Item 5: regular-way trading from 2011-11-01; high and "
            "low of November 2011 $11.85 and $8.78",
            ("trading start", "price range"),
        ),
        Source(
            _RBJ,
            "Rochester Business Journal, 2011-11-01: closes of the first regular-way day, Exelis "
            "$11.70, Xylem $27.31, ITT $18.80; the only source of the Exelis close, and a news "
            "report: the page itself did not open (HTTP 403), the figures come from two search-"
            "tool excerpts of it. Its Xylem and ITT closes equal Yahoo's, its Exelis close lies "
            "in the range of the Exelis 10-K",
            ("exelis close",),
            "news",
        ),
    ),
    verified_on="2026-10-08",
    price_checks=(
        PriceCheck(
            "ITT", "2011-10-31", 45.595, 45.605,
            "as-traded close: $91.20 of the 10-K on the 1-for-2 adjusted basis, divided by 2",
            _ITT_10K,
        ),
        PriceCheck(
            "ITT", "2011-11-01", 18.795, 18.805, "close of the first day, to the cent", _RBJ
        ),
        PriceCheck("XYL", "2011-11-01", 27.305, 27.315, "close of the first day, to the cent", _RBJ),
        PriceCheck(
            "XLS", "2011-11-01", 8.78, 11.85,
            "the documented close must lie in the high-low range of November 2011", _EXELIS_10K,
        ),
    ),
    note="Yahoo books 3.016: the 1-for-2 reverse split joined to a value ratio of about 6.03 for "
    "the Xylem and Exelis shares handed out. A holder owned half a new ITT share, one Xylem and "
    "one Exelis share after the day; with the Exelis close anywhere in the 10-K's range of "
    "November 2011 ($8.78 to $11.85) the holder's return is between -0.2 % and +6.5 %, never "
    "+24.3 %.",
)

_CCP_8K = f"{_SEC}/0001639947/000110465915061326/a15-16820_28k.htm"
_CCP_INFO = f"{_SEC}/0001639947/000104746915006590/a2225636zex-99_1.htm"
_VENTAS_TAX = (
    "https://s203.q4cdn.com/581756553/files/doc_downloads/Form8937/"
    "Form-8937-Ventas-Inc-2015-Tax-Free-Spin-Off-of-Care-Capital-Properties.pdf"
)

VTR = VerifiedEvent(
    ticker="VTR",
    name="Ventas, Inc.: spin-off of Care Capital Properties",
    date="2015-08-18",
    yahoo_ratio=0.8757,
    parent_shares_per_share=1.0,
    distributions=(
        Distribution(
            "CCP", "Care Capital Properties, Inc.", 0.25, price="documented",
            documented_close=35.58, documented_basis="close", documented_source=_VENTAS_TAX,
        ),
    ),
    record_date="2015-08-10",
    distribution_date="2015-08-17",
    sources=(
        Source(
            _CCP_8K,
            "Care Capital Properties, Form 8-K, report date 2015-08-17: Ventas stockholders of "
            "record at the close of business on 2015-08-10, distribution effective at 11:59 p.m. "
            "ET on 2015-08-17",
            ("dates",),
        ),
        Source(
            _CCP_INFO,
            "Care Capital Properties, information statement (Exhibit 99.1 to the Form 8-K, Item "
            "7.01, accession 0001047469-15-006590): one CCP share for every four Ventas shares; "
            "regular-way trading of CCP on the first trading day after the distribution",
            ("ratio", "dates"),
        ),
        Source(
            _VENTAS_TAX,
            "Ventas, tax information on the Care Capital spin-off (summary of Form 8937): NYSE "
            "closing prices of 2015-08-18, Ventas $59.80 and CCP $35.58, distribution ratio 1:4",
            ("ratio", "prices"),
            "company",
        ),
    ),
    verified_on="2026-10-08",
    price_checks=(
        PriceCheck(
            "VTR", "2015-08-18", 59.795, 59.805, "NYSE close, to the cent (Ventas tax information)",
            _VENTAS_TAX,
        ),
    ),
    note="Yahoo books 0.8757, the inverse of the value ratio (1.142), so every earlier close is "
    "raised by 14 % instead of lowered and the day shows -23.6 %. A holder owned the same Ventas "
    "shares and a quarter of a CCP share more after the day.",
)

_DD_8K_SPLIT = f"{_SEC}/1666700/000166670019000023/dupontreversestocksplit.htm"
_CTVA_8K = f"{_SEC}/1755672/000119312519163314/d753864dex991.htm"
_DD_8937 = (
    "https://s23.q4cdn.com/116192123/files/doc_downloads/2019/07/"
    "Form-8937-Attachment-Combined-Corteva-Spin-and-Rev-Stk-Splt-Final.pdf"
)
_DOW_8937 = (
    "https://s23.q4cdn.com/116192123/files/doc_downloads/2019/04/DWDP-Form-8937-Attachment-Final.pdf"
)
_DD_8K_2026 = f"{_SEC}/0001666700/000166670026000046/dd-20260623.htm"
_INQUIRER = "https://inquirer.com/business/dupont-corteva-dow-breen-shares--20190603.html"

DD = VerifiedEvent(
    ticker="DD",
    name="DowDuPont Inc. (DuPont de Nemours): spin-off of Corteva, 1-for-3 reverse split",
    date="2019-06-03",
    yahoo_ratio=0.4725,
    parent_shares_per_share=1.0 / 3.0,
    distributions=(Distribution("CTVA", "Corteva, Inc.", 1.0 / 3.0),),
    record_date="2019-05-24",
    distribution_date="2019-06-01",
    later_splits_unlisted=(("2026-06-24", 1.0 / 3.0),),
    sources=(
        Source(
            _DD_8K_SPLIT,
            "DowDuPont Inc., Form 8-K, report date 2019-05-23: reverse stock split of one new "
            "share for three, immediately after the Corteva distribution; trading as DuPont de "
            "Nemours under DD from 2019-06-03",
            ("reverse split", "dates"),
        ),
        Source(
            _CTVA_8K,
            "Corteva, Inc., Form 8-K, Exhibit 99.1 (news release of 2019-06-03): one Corteva "
            "share for every three DowDuPont shares, record date 2019-05-24, distribution on "
            "2019-06-01, CTVA begins trading on the NYSE on 2019-06-03",
            ("ratio", "dates"),
        ),
        Source(
            _DD_8937,
            "DuPont de Nemours, Form 8937 attachment (Corteva distribution and 1-for-3 reverse "
            "stock split), signed 2019-07-01: the terms and dates, and the averages of the day's "
            "high and low on 2019-06-04: DuPont $74.94 after the split, Corteva $26.15",
            ("ratio", "dates", "reverse split", "prices"),
            "company",
        ),
        Source(
            _DOW_8937,
            "DowDuPont Inc., Form 8937 attachment (distribution of Dow Inc.), April 2019: average "
            "of the day's high and low on 2019-04-03, DowDuPont $37.19",
            ("prices",),
            "company",
        ),
        Source(
            _DD_8K_2026,
            "DuPont de Nemours, Inc., Form 8-K, report date 2026-06-24: one-for-three reverse "
            "stock split effective at 12:01 a.m. ET on 2026-06-24, after the end of the cached "
            "window: Yahoo's Close of every earlier day is three times the as-traded price and "
            "Stock Splits of the cache does not list it",
            ("later split",),
        ),
        Source(
            _INQUIRER,
            "Philadelphia Inquirer, 2019-06-03: DuPont at $76.10 (an 18 % rise) on the first day "
            "after the reverse split; equal to the as-traded close derived from Yahoo",
            ("dd close",),
            "news",
        ),
    ),
    verified_on="2026-10-08",
    price_checks=(
        PriceCheck(
            "DD", "2019-04-03", 36.07, 38.31,
            "average of the day's high and low $37.19 within 3 % (Form 8937, Dow distribution)",
            _DOW_8937,
        ),
        PriceCheck(
            "DD", "2019-06-03", 76.095, 76.105,
            "close of the first day, to the cent (Philadelphia Inquirer)", _INQUIRER,
        ),
        PriceCheck(
            "DD", "2019-06-04", 72.69, 77.19,
            "average of the day's high and low $74.94 within 3 % (Form 8937, Corteva distribution)",
            _DD_8937,
        ),
        PriceCheck(
            "CTVA", "2019-06-04", 24.06, 28.24,
            "average of the day's high and low $26.15 within 8 % (Form 8937, Corteva distribution)",
            _DD_8937,
        ),
    ),
    note="Yahoo books 0.4725, the 1-for-3 reverse split joined to a value ratio of 1.4175; the "
    "return of the close on the day is +17.8 %. A holder of three DowDuPont shares owned one "
    "DuPont and one Corteva share after the day. The cum-distribution close of DowDuPont "
    "($30.52 as traded) was low against the sum of the parts: +10.2 % is the holder's return "
    "between the two closes, not a data error.",
)

_AIV_8K = f"{_SEC}/922864/000119312520317477/d772672d8k.htm"
_AIR_INFO = f"{_SEC}/1820877/000119312520305781/d17437dex991.htm"
_AIR_PR = (
    "https://investors.aircommunities.com/press-releases/news-details/2020/"
    "Apartment-Income-REIT-Corp.-Completes-Separation-from-Aimco/default.aspx"
)
_AIV_8937 = "https://s29.q4cdn.com/252595921/files/doc_downloads/form8937/8937-AIRC-Distribution.pdf"
_AIV_8K_DIV = f"{_SEC}/922864/000119312520307169/d48150dex991.htm"

AIV = VerifiedEvent(
    ticker="AIV",
    name="Apartment Investment and Management Company (Aimco): separation of AIR Communities",
    date="2020-12-15",
    yahoo_ratio=9.295,
    parent_shares_per_share=1.0,
    distributions=(
        Distribution(
            "AIRC", "Apartment Income REIT Corp.", 1.0, price="documented",
            documented_close=38.0711, documented_basis="vwap", documented_source=_AIV_8937,
        ),
    ),
    record_date="2020-12-05",
    distribution_date="2020-12-15",
    sources=(
        Source(
            _AIV_8K,
            "Aimco, Form 8-K (Item 2.01 among others), filed 2020-12-15: separation completed on "
            "2020-12-15, one AIR Class A share for each Aimco Class A share held on 2020-12-05, "
            "no reverse split of Aimco",
            ("ratio", "dates"),
        ),
        Source(
            _AIR_INFO,
            "Apartment Income REIT Corp., information statement (Exhibit 99.1), 2020-11-30: record "
            "date 2020-12-05, distribution expected on 2020-12-15, one AIR share for each Aimco "
            "share",
            ("ratio", "dates"),
        ),
        Source(
            _AIR_PR,
            "AIR Communities, news release of 2020-12-15: AIR trades 'regular way' on the NYSE "
            "from 2020-12-15 as AIRC",
            ("trading start",),
            "company",
        ),
        Source(
            _AIV_8937,
            "Aimco, Form 8937 (date of action 2020-12-15, signed 2021-01-15): the distribution of "
            "one AIRC share per Aimco share is valued at $38.0711, the volume weighted average "
            "trading price of AIRC on 2020-12-15",
            ("ratio", "prices"),
            "company",
        ),
        Source(
            _AIV_8K_DIV,
            "Aimco, Form 8-K Exhibit 99.1 of 2020-12-01 (special dividend): the 10-day volume "
            "weighted average price ending 2020-11-24 was $30.98, before the 1-for-1.239 reverse "
            "split of 2020-12-01 that Yahoo books as 0.807",
            ("prices",),
        ),
    ),
    verified_on="2026-10-08",
    price_checks=(
        PriceCheck(
            "AIV", "2020-11-24", 29.12, 32.84,
            "10-day volume weighted average price $30.98 within 6 % (before the 2020-12-01 split)",
            _AIV_8K_DIV,
        ),
    ),
    note="Yahoo books 9.295 on a day without a dividend; the Aimco shares closed at $5.04. AIRC "
    "has no Yahoo series (no longer listed): its value is the volume weighted average price of "
    "the day that Aimco itself used, not the close.",
)

_LDOS_8K = f"{_SEC}/0001336920/000119312513387297/d605465d8k.htm"
_SAIC_PR = f"{_SEC}/0001336920/000119312513362164/d595478dex991.htm"
_LDOS_TAX = "https://investors.leidos.com/static-files/871e90e6-9979-4409-be16-62931ed730ac"

LDOS = VerifiedEvent(
    ticker="LDOS",
    name="SAIC, Inc. (Leidos Holdings): spin-off of Science Applications, 1-for-4 reverse split",
    date="2013-09-30",
    yahoo_ratio=0.405,
    parent_shares_per_share=0.25,
    distributions=(Distribution("SAIC", "Science Applications International Corporation", 1.0 / 7.0),),
    record_date="2013-09-19",
    distribution_date="2013-09-27",
    sources=(
        Source(
            _LDOS_8K,
            "Leidos Holdings (formerly SAIC, Inc.), Form 8-K, report date 2013-09-25, filed "
            "2013-10-01 (Items 2.01, 5.03): separation completed at 11:59 p.m. ET on 2013-09-27, "
            "one New SAIC share for every seven shares held on 2013-09-19, one-for-four reverse "
            "stock split",
            ("ratio", "dates", "reverse split"),
        ),
        Source(
            _SAIC_PR,
            "SAIC, Inc., Form 8-K, Exhibit 99.1 (news release of 2013-09-09): 1-for-4 reverse "
            "split of the Leidos shares effective immediately after the distribution, before the "
            "open on 2013-09-30; regular-way trading of LDOS and SAIC from 2013-09-30",
            ("reverse split", "dates"),
        ),
        Source(
            _LDOS_TAX,
            "Leidos investor relations, tax-basis notice for the distribution: when-issued "
            "closing prices of 2013-09-27, Leidos $44.15 (after the split) and New SAIC $35.25; "
            "figures taken from a search summary, the page itself did not open",
            ("prices",),
            "company",
        ),
    ),
    verified_on="2026-10-08",
    price_checks=(
        PriceCheck(
            "SAIC", "2013-09-27", 35.245, 35.255,
            "when-issued close of New SAIC, to the cent (Leidos tax-basis notice)", _LDOS_TAX,
        ),
        PriceCheck(
            "LDOS", "2013-09-27", 15.83, 16.31,
            "cum-distribution close within 1.5 % of the parts: 44.15 / 4 + 35.25 / 7 = 16.07",
            _LDOS_TAX,
        ),
    ),
    note="Yahoo books 0.405 for a reverse split of 1:4 joined to a value ratio; the return of "
    "the close on the day is +15.0 %. A holder of 28 old shares owned 7 Leidos and 4 SAIC shares.",
)

_HP_PR = f"{_SEC}/47217/000119312515334592/d88067dex991.htm"
_HP_8K = f"{_SEC}/47217/000119312515368382/d86365d8k.htm"
_HPE_10K = f"{_SEC}/0001645590/000104746915009279/a2226746z10-k.htm"
_HP_10K = f"{_SEC}/47217/000104746915009251/a2226745z10-k.htm"
_HP_OCC = "https://boxoptions.com/assets/BOXOnnMemo203686.pdf"

HPQ = VerifiedEvent(
    ticker="HPQ",
    name="Hewlett-Packard Company (HP Inc.): spin-off of Hewlett Packard Enterprise",
    date="2015-11-02",
    yahoo_ratio=2.202,
    parent_shares_per_share=1.0,
    distributions=(Distribution("HPE", "Hewlett Packard Enterprise Company", 1.0),),
    record_date="2015-10-21",
    distribution_date="2015-11-01",
    sources=(
        Source(
            _HP_PR,
            "Hewlett-Packard Company, Form 8-K, Exhibit 99.1 (news release of 2015-10-01): one "
            "HPE share for every HP share held on 2015-10-21, distribution on 2015-11-01, "
            "regular-way trading of HPE on 2015-11-02; no split of HP shares",
            ("ratio", "dates"),
        ),
        Source(
            _HP_8K,
            "HP Inc., Form 8-K (Item 2.01), filed 2015-11-05, period of report 2015-10-31: "
            "separation effective at 12:01 a.m. ET on 2015-11-01, one HPE share for every share, "
            "record date 2015-10-21",
            ("ratio", "dates"),
        ),
        Source(
            _HP_OCC,
            "BOX Options Exchange / OCC memo of 2015-10-30: the NYSE has set 2015-11-02 as the "
            "ex-distribution date, record date 2015-10-21, mail date 2015-11-01, ratio 1.0",
            ("ex-date",),
            "exchange",
        ),
        Source(
            _HPE_10K,
            "Hewlett Packard Enterprise, Form 10-K FY2015, Item 5: regular-way trading from "
            "2015-11-02; NYSE prices of 2015-11-02 to 2015-11-30 between $13.12 and $15.17",
            ("trading start", "price range"),
        ),
        Source(
            _HP_10K,
            "HP Inc., Form 10-K FY2015, Item 5: average price paid for the October 2015 "
            "repurchases $27.70",
            ("prices",),
        ),
    ),
    verified_on="2026-10-08",
    price_checks=(
        PriceCheck(
            "HPE", "2015-11-02", 13.12, 15.17,
            "range of the NYSE prices of November 2015 (HPE Form 10-K FY2015)", _HPE_10K,
        ),
        PriceCheck(
            "HPQ", "2015-10-30", 24.93, 30.47,
            "average price of the October 2015 repurchases $27.70 within 10 % (HP Form 10-K)",
            _HP_10K,
        ),
    ),
    note="Yahoo books 2.202 = P / (P - V) with V the when-issued close of HPE ($14.72) on the "
    "last day of the entitlement; the return of the close on the day is +13.0 %.",
)

VERIFIED_EVENTS: tuple[VerifiedEvent, ...] = (ITT, VTR, DD, AIV, LDOS, HPQ)
