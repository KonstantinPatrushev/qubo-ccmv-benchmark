"""Risk-free rates for Sharpe, Sortino and CEQ (WP-E1, external finding 4).

* **US**: FRED ``DTB3`` -- "3-Month Treasury Bill Secondary Market Rate, Discount Basis" (H.15,
  percent per year, one value per Treasury-market business day).
* **RU**: RUONIA, the CBR overnight rate, from the CBR web service ``DailyInfoWebServ``
  (SOAP method ``Ruonia``; verified 2026-10-04). It returns the fixing for the date ``D0`` and the
  date and time of its publication (the next business day, about 14:00 Moscow time). The key rate
  is not needed as a substitute because the service is available.

Basis of the US series
----------------------
``DTB3`` is a *bank discount* rate ``d`` on a 360-day year: a 91-day bill is priced at
``100 * (1 - d * 91 / 360)``. That is not a yield. The bond-equivalent yield (BEY, the 365-day
simple annual yield, comparable to a deposit rate) is
``365 * d / (360 - d * t)`` with ``t`` days to maturity, see :func:`discount_to_bey`: at ``d = 5 %``
BEY is 5.13 %, i.e. about 13 bp per year more. The daily rate prescribed for the study,
``(1 + y/100) ** (days/365) - 1``, is applied to the *published* ``DTB3`` value (``r_daily``); the
same formula on the BEY is returned next to it (``r_daily_bey``) so the effect can be measured. The
difference is a level shift of roughly the T-bill discount, irrelevant for rankings but not for
the absolute Sharpe ratio at 5 % rates.

RUONIA is quoted ACT/365 simple interest, so ``r_daily = y/100 * days / 365``.

No look-ahead
-------------
The accrual for trading day ``d`` runs from the close of the previous trading day ``p`` to the
close of ``d``. It uses the last fixing that was *known at the close of p* (an observation is
known from its publication date, or from its own date for DTB3, which is a market quote), and
the number of calendar days between ``p`` and ``d`` (3 over a weekend, 27 over the 2022 closure).
"""

from __future__ import annotations

import io
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CBR_SOAP_URL = "https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx"
CBR_NS = "http://web.cbr.ru/"
DTB3_MATURITY_DAYS = 91


class RiskFreeError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# Rate conversions
# --------------------------------------------------------------------------------------------


def daily_rate_compound(y_pct, days):
    """``(1 + y/100) ** (days/365) - 1``: the rate earned over ``days`` calendar days."""
    return (1.0 + np.asarray(y_pct, dtype=float) / 100.0) ** (np.asarray(days, dtype=float) / 365.0) - 1.0


def daily_rate_simple(y_pct, days):
    """``y/100 * days/365`` (ACT/365 simple interest, RUONIA convention)."""
    return np.asarray(y_pct, dtype=float) / 100.0 * np.asarray(days, dtype=float) / 365.0


def discount_to_bey(discount_pct, days_to_maturity: int = DTB3_MATURITY_DAYS):
    """Bank-discount rate (percent, 360-day year) -> bond-equivalent yield (percent, 365-day)."""
    d = discount_pct / 100.0
    return 100.0 * 365.0 * d / (360.0 - d * days_to_maturity)


# --------------------------------------------------------------------------------------------
# Parsers and downloads
# --------------------------------------------------------------------------------------------


def parse_fred_csv(text: str) -> pd.Series:
    """FRED ``fredgraph.csv`` -> Series of percent indexed by date; missing markers dropped."""
    df = pd.read_csv(io.StringIO(text), na_values=["."], parse_dates=[0])
    series = df.set_index(df.columns[0])[df.columns[1]].astype(float).dropna()
    series.index.name = "date"
    return series.sort_index()


def _retry(call: Callable[[], requests.Response], what: str, retries: int, sleep) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = call()
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            if attempt < retries:
                sleep(2.0**attempt)
    raise RiskFreeError(f"{what}: {last}")


def fetch_fred_csv(series_id: str = "DTB3", *, session=None, timeout: float = 30.0, retries: int = 4,
                   sleep=time.sleep) -> tuple[pd.Series, bytes]:
    """Download a FRED series; returns ``(series, raw_bytes)``.

    The default ``requests`` user agent is used on purpose: ``fred.stlouisfed.org`` holds the
    connection open without answering for some custom agents.
    """
    session = session or requests
    resp = _retry(lambda: session.get(FRED_CSV_URL, params={"id": series_id}, timeout=timeout),
                  f"FRED {series_id}", retries, sleep)
    series = parse_fred_csv(resp.text)
    series.name = series_id
    return series, resp.content


def parse_ruonia_soap(xml_text: str) -> pd.DataFrame:
    """SOAP answer of ``DailyInfoWebServ.Ruonia`` -> DataFrame indexed by the fixing date.

    Columns: ``y_pct`` (percent per year), ``volume_bn`` (RUB bn of the deals behind the fixing),
    ``published`` (date on which the fixing was published; the Friday fixing appears on Monday)
    and ``published_assumed``. Fixings before about 2013 carry no publication date; for them the
    day after the fixing is assumed, which gives the same as-of result on trading days as the
    real next-business-day publication.
    """
    root = ET.fromstring(xml_text)
    for el in root.iter():
        if el.tag.endswith("}Fault") or el.tag == "Fault":
            msg = " ".join((e.text or "").strip() for e in el.iter() if e.tag.endswith("faultstring"))
            raise RiskFreeError(f"CBR SOAP fault: {msg}")
    rows = []
    for ro in root.iter("ro"):
        field = {child.tag: (child.text or "").strip() for child in ro}
        day = pd.Timestamp(field["D0"][:10])
        stamp = field.get("DateUpdate")
        rows.append({
            "date": day,
            "y_pct": float(field["ruo"]),
            "volume_bn": float(field["vol"]) if field.get("vol") else np.nan,
            "published": pd.Timestamp(stamp[:10]) if stamp else day + pd.Timedelta(days=1),
            "published_assumed": not stamp,
        })
    if not rows:
        raise RiskFreeError("CBR answer contains no <ro> rows")
    df = pd.DataFrame(rows).set_index("date").sort_index()
    return df[~df.index.duplicated(keep="last")]


def fetch_ruonia(start, end, *, session=None, timeout: float = 60.0, retries: int = 4,
                 sleep=time.sleep) -> tuple[pd.DataFrame, bytes]:
    """RUONIA fixings between two dates (SOAP ``Ruonia``); returns ``(frame, raw_xml_bytes)``."""
    body = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/"><soap:Body>'
        f'<Ruonia xmlns="{CBR_NS}"><fromDate>{pd.Timestamp(start).date()}</fromDate>'
        f"<ToDate>{pd.Timestamp(end).date()}</ToDate></Ruonia></soap:Body></soap:Envelope>"
    )
    headers = {"Content-Type": "text/xml; charset=utf-8", "SOAPAction": f'"{CBR_NS}Ruonia"'}
    session = session or requests
    resp = _retry(
        lambda: session.post(CBR_SOAP_URL, data=body.encode("utf-8"), headers=headers, timeout=timeout),
        "CBR Ruonia", retries, sleep,
    )
    return parse_ruonia_soap(resp.text), resp.content


# --------------------------------------------------------------------------------------------
# Alignment to a trading calendar
# --------------------------------------------------------------------------------------------


def daily_rate_on_calendar(
    obs: pd.Series,
    calendar: pd.DatetimeIndex,
    *,
    compounding: str = "compound",
    available_from: pd.Series | None = None,
) -> pd.DataFrame:
    """Risk-free return for every trading day of ``calendar``, without look-ahead.

    Parameters
    ----------
    obs:
        Annual rate in percent indexed by observation date (gaps allowed).
    calendar:
        Trading days of the market (sorted).
    compounding:
        ``"compound"`` (``(1+y)**(days/365)-1``, DTB3) or ``"simple"`` (``y*days/365``, RUONIA).
    available_from:
        Date from which each observation is known (same index as ``obs``); default is the
        observation date itself. For RUONIA pass the publication date.

    Returns
    -------
    DataFrame indexed by ``calendar`` with ``y_pct`` and ``obs_date`` (the fixing used), ``days``
    (calendar days since the previous trading day) and ``r_daily``; the first day is NaN.
    """
    if compounding not in ("compound", "simple"):
        raise ValueError(f"compounding must be 'compound' or 'simple', got {compounding!r}")
    cal = pd.DatetimeIndex(calendar)
    obs = obs.dropna().sort_index()
    known = obs.index.to_series() if available_from is None else available_from.reindex(obs.index)
    known = pd.DatetimeIndex(known.to_numpy())
    if not known.is_monotonic_increasing:
        raise ValueError("publication dates must not decrease with the observation date")

    prev = cal[:-1]
    pos = known.searchsorted(prev, side="right") - 1  # last observation known at the close of p
    ok = pos >= 0
    y = np.full(len(cal), np.nan)
    obs_date = pd.DatetimeIndex([pd.NaT] * len(cal))
    days = np.full(len(cal), np.nan)
    y[1:][ok] = obs.to_numpy()[pos[ok]]
    obs_date_values = obs_date.to_numpy().copy()
    obs_date_values[1:][ok] = obs.index.to_numpy()[pos[ok]]
    days[1:] = (cal[1:] - cal[:-1]).days
    rate = daily_rate_compound if compounding == "compound" else daily_rate_simple
    out = pd.DataFrame(
        {"y_pct": y, "obs_date": pd.DatetimeIndex(obs_date_values), "days": days,
         "r_daily": rate(y, days)},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    return out
