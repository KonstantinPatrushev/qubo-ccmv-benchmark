"""Official exchange rates of the Bank of Russia (cbr.ru, the ``XML_dynamic`` service).

Used to convert dividends paid in a foreign currency (USD for QIWI, RUSAL, Polymetal, …) into
roubles for the Russian market (ANALYSIS_PLAN §3, RU exploratory). The service returns, for one
currency and a date range, the official rate set for each date::

    <ValCurs ID="R01235" DateRange1="01.01.2013" DateRange2="31.12.2025" ...>
      <Record Date="10.01.2013" Id="R01235"><Nominal>1</Nominal><Value>30,2645</Value></Record>
      ...

in windows-1251 with a decimal comma; ``Value`` is the price of ``Nominal`` units. The raw response
is stored verbatim (``raw_name``) with ``<file>.fetched_at.json``; the processed series is the rate
per ONE unit of the currency, in roubles, indexed by the date the rate is set for.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
USD_ID = "R01235"
_NAMES = {USD_ID: "usd_rub"}
_UNITS = {USD_ID: "RUB per 1 USD"}


def dynamic_url(currency_id: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """URL of the ``XML_dynamic`` request for ``currency_id`` over ``[start, end]``."""
    return (f"{BASE_URL}?date_req1={start:%d/%m/%Y}&date_req2={end:%d/%m/%Y}"
            f"&VAL_NM_RQ={currency_id}")


def raw_name(currency_id: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    """File name of the stored raw response."""
    return f"{currency_id}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.xml"


def download_dynamic(
    currency_id: str, start: pd.Timestamp, end: pd.Timestamp, raw_dir: Path | str,
    timeout: float = 60.0,
) -> dict:
    """Download the rates into ``raw_dir`` verbatim; write ``<file>.fetched_at.json``."""
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = dynamic_url(currency_id, start, end)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "cif-research/1.0"})
    response.raise_for_status()
    data = response.content
    parse_dynamic(data, currency_id)  # refuse to store an error page as data
    name = raw_name(currency_id, start, end)
    (raw_dir / name).write_bytes(data)
    meta = {"url": url, "file": name,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (raw_dir / f"{name}.fetched_at.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def parse_dynamic(data: bytes, currency_id: str) -> pd.DataFrame:
    """The ``XML_dynamic`` response as a frame indexed by ``date``: ``rate`` (roubles per one unit)
    and ``nominal``. Rejects other documents, another currency, no records, repeated dates and
    non-positive values."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"not a ValCurs XML document: {exc}") from None
    if root.tag != "ValCurs":
        raise ValueError(f"expected a ValCurs document, got <{root.tag}>")
    if root.get("ID") != currency_id:
        raise ValueError(f"the response is for currency {root.get('ID')!r}, not {currency_id!r}")
    rows = []
    for rec in root.iter("Record"):
        if rec.get("Id") != currency_id:
            raise ValueError(f"a record of currency {rec.get('Id')!r} in a {currency_id} response")
        nominal = int(rec.findtext("Nominal"))
        value = float(rec.findtext("Value").replace(",", "."))
        if nominal <= 0 or value <= 0.0:
            raise ValueError(f"rate on {rec.get('Date')} must be positive, got {value}/{nominal}")
        rows.append((datetime.strptime(rec.get("Date"), "%d.%m.%Y"), value / nominal, nominal))
    if not rows:
        raise ValueError(f"no records for {currency_id} in the response")
    df = pd.DataFrame(rows, columns=["date", "rate", "nominal"]).set_index("date")
    if df.index.has_duplicates:
        raise ValueError(f"duplicate dates in the response: {df.index[df.index.duplicated()][:3].tolist()}")
    return df.sort_index()


def build_dataset(
    raw_dir: Path | str, out_dir: Path | str, currency_id: str, start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Parse the stored response and write ``<name>.parquet`` into ``out_dir``; return a summary."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = parse_dynamic((raw_dir / raw_name(currency_id, start, end)).read_bytes(), currency_id)
    name = _NAMES.get(currency_id, currency_id.lower())
    df.to_parquet(out_dir / f"{name}.parquet")
    return {
        "currency_id": currency_id,
        "series": name,
        "units": _UNITS.get(currency_id, "RUB per 1 unit"),
        "n_days": len(df),
        "first_date": f"{df.index[0]:%Y-%m-%d}",
        "last_date": f"{df.index[-1]:%Y-%m-%d}",
        "nominals": sorted(int(v) for v in df["nominal"].unique()),
    }
