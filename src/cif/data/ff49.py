"""Kenneth French data library: 49 industry portfolios (daily) and the daily risk-free rate (RQ3).

FF49 is the primary market of RQ3 (ANALYSIS_PLAN §3): portfolios of all CRSP firms of an industry,
so there is no survivorship bias in the universe.

Files (French's site, ``ftp/`` directory):
- ``49_Industry_Portfolios_daily_CSV.zip``: two tables, value- and equal-weighted daily returns in
  percent;
- ``F-F_Research_Data_Factors_daily_CSV.zip``: one untitled table ``Mkt-RF, SMB, HML, RF``;
- ``F-F_Research_Data_Factors_CSV.zip`` (optional): the monthly table and annual factors. If it is
  present, the daily risk-free rate is the monthly one-month T-bill return spread over the trading
  days of the month (:func:`daily_rf_from_monthly`): the daily file rounds it to 0.01 %/day, i.e.
  annual rates jump between 2.55 % and 5.17 % (audit F-1).

Both start with a few description lines (the first names the CRSP vintage, e.g. "202608 CRSP
database"; French revises the data every year, so the files are pinned by SHA-256 in the
provenance). Missing returns are coded -99.99 or -999.

:func:`price_index` turns returns into level series for :func:`cif.backtest.walkforward.run_walkforward`:
- before the first valid return the level is missing (the industry has no firms yet);
- a missing return inside the series gives a missing level that day, with the unknown return
  treated as 0 in the level. The engine's ``select_assets`` mode then treats the day as
  untradable and earns nothing on it.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
INDUSTRY_ZIP = "49_Industry_Portfolios_daily_CSV.zip"
FACTORS_ZIP = "F-F_Research_Data_Factors_daily_CSV.zip"
FACTORS_MONTHLY_ZIP = "F-F_Research_Data_Factors_CSV.zip"
VALUE_TITLE = "Average Value Weighted Returns -- Daily"
EQUAL_TITLE = "Average Equal Weighted Returns -- Daily"
MISSING_CODES = (-99.99, -999.0)

_ROW = re.compile(r"^\s*(\d{8}|\d{6}|\d{4})\s*,(.*)$")
_VINTAGE = re.compile(r"using the (\d{6} CRSP database)")


def parse_french_csv(text: str) -> dict[str, pd.DataFrame]:
    """All tables of a French CSV, keyed by their title (``""`` if the header has none above it).

    A table is a header line starting with ``,`` followed by rows ``YYYYMMDD, v1, ...``; its title
    is the non-empty line right above the header. Values stay in the file's units (percent);
    missing codes become NaN. Dates must be strictly increasing.
    """
    lines = text.splitlines()
    tables: dict[str, pd.DataFrame] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(","):
            i += 1
            continue
        title = lines[i - 1].strip() if i > 0 else ""
        columns = [c.strip() for c in line.split(",")[1:]]
        dates, rows = [], []
        i += 1
        while i < len(lines) and (m := _ROW.match(lines[i])):
            values = [float(v) for v in m.group(2).split(",")]
            if len(values) != len(columns):
                raise ValueError(f"row {m.group(1)} of table {title!r}: {len(values)} values, "
                                 f"{len(columns)} columns")
            dates.append(m.group(1))
            rows.append(values)
            i += 1
        frame = pd.DataFrame(rows, index=_index(dates), columns=columns)
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"dates of table {title!r} are not strictly increasing")
        frame = frame.mask(frame.isin(MISSING_CODES))
        frame.index.name = "date"
        tables[title] = frame
    return tables


def _index(dates: list[str]) -> pd.Index:
    """Daily ``YYYYMMDD`` → dates; monthly ``YYYYMM`` and annual ``YYYY`` → periods."""
    if not dates or len(dates[0]) == 8:
        return pd.to_datetime(dates, format="%Y%m%d")
    if len(dates[0]) == 6:
        return pd.PeriodIndex([f"{d[:4]}-{d[4:]}" for d in dates], freq="M")
    return pd.PeriodIndex(dates, freq="Y")


def daily_rf_from_monthly(rf_monthly: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Daily rate that compounds to the month's rate over its trading days in ``calendar``.

    French's daily RF is this quantity rounded to 0.01 %/day (audit F-1); here it is computed from
    the monthly one-month T-bill return (decimals, monthly ``PeriodIndex``) without that rounding.
    """
    months = pd.DatetimeIndex(calendar).to_period("M")
    missing = sorted(set(months) - set(rf_monthly.index))
    if missing:
        raise ValueError(f"no monthly rate for {len(missing)} month(s), e.g. {missing[:3]}")
    n_days = pd.Series(months).value_counts()
    rate = rf_monthly.reindex(months).to_numpy()
    days = n_days.reindex(months).to_numpy()
    return pd.Series((1.0 + rate) ** (1.0 / days) - 1.0, index=pd.DatetimeIndex(calendar), name="rf")


def industry_returns(
    tables: dict[str, pd.DataFrame], weighting: str = "value", expected_columns: int = 49
) -> pd.DataFrame:
    """Daily returns in decimals of the value- or equal-weighted industry portfolios."""
    title = {"value": VALUE_TITLE, "equal": EQUAL_TITLE}[weighting]
    if title not in tables:
        raise ValueError(f"table {title!r} not found; tables: {list(tables)}")
    frame = tables[title]
    if frame.shape[1] != expected_columns:
        raise ValueError(f"expected {expected_columns} industries, got {frame.shape[1]}")
    return frame / 100.0


def risk_free(tables: dict[str, pd.DataFrame]) -> pd.Series:
    """Daily risk-free rate in decimals (the one-month T-bill compounded to a daily rate)."""
    frame = next(iter(tables.values()))
    if "RF" not in frame.columns:
        raise ValueError(f"no RF column; columns: {list(frame.columns)}")
    return (frame["RF"] / 100.0).rename("rf")


def price_index(returns: pd.DataFrame) -> pd.DataFrame:
    """Level series from returns: missing before the first valid return and on missing days."""
    levels = (1.0 + returns.fillna(0.0)).cumprod()
    started = returns.notna().cummax()
    return levels.where(started & returns.notna())


def read_zip_text(path: Path | str) -> str:
    """Text of the single CSV member of a French zip archive."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one CSV member, got {members}")
        return zf.read(members[0]).decode("latin-1")


def vintage(text: str) -> str | None:
    """The CRSP vintage named in the first description line, e.g. ``"202608 CRSP database"``."""
    m = _VINTAGE.search(text[:500])
    return m.group(1) if m else None


def download_zip(name: str, raw_dir: Path | str, timeout: float = 60.0) -> dict:
    """Download ``name`` from French's site into ``raw_dir``; write ``<name>.fetched_at.json``."""
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + name
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    data = response.content
    (raw_dir / name).write_bytes(data)
    meta = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (raw_dir / f"{name}.fetched_at.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def build_dataset(raw_dir: Path | str, out_dir: Path | str, expected_industries: int = 49) -> dict:
    """Parse the two raw zips and write the processed parquet files; return a summary.

    Outputs: ``returns_vw``, ``returns_ew`` (decimal daily returns), ``prices_vw`` (level index for
    the engine), ``rf_daily`` (decimal daily risk-free rate) and ``factors_daily`` (Mkt-RF, SMB,
    HML, RF in decimals).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    industry_text = read_zip_text(raw_dir / INDUSTRY_ZIP)
    factors_text = read_zip_text(raw_dir / FACTORS_ZIP)
    industries = parse_french_csv(industry_text)
    factors = parse_french_csv(factors_text)
    vw = industry_returns(industries, "value", expected_industries)
    ew = industry_returns(industries, "equal", expected_industries)
    if (raw_dir / FACTORS_MONTHLY_ZIP).exists():
        monthly = parse_french_csv(read_zip_text(raw_dir / FACTORS_MONTHLY_ZIP))[""]
        rf = daily_rf_from_monthly(monthly["RF"] / 100.0, vw.index)
        rf_source = ("monthly factors file: one-month T-bill return compounded over the trading "
                     "days of the month")
    else:
        rf = risk_free(factors).reindex(vw.index)
        rf_source = "daily factors file (two decimals in percent: steps of 0.01 %/day)"
    factors_daily = next(iter(factors.values())) / 100.0
    prices = price_index(vw)

    vw.to_parquet(out_dir / "returns_vw.parquet")
    ew.to_parquet(out_dir / "returns_ew.parquet")
    prices.to_parquet(out_dir / "prices_vw.parquet")
    rf.to_frame().to_parquet(out_dir / "rf_daily.parquet")
    factors_daily.to_parquet(out_dir / "factors_daily.parquet")

    first_valid = vw.apply(lambda c: c.first_valid_index())
    after_start = vw.loc["1960-01-01":]
    interior = {c: int(after_start[c].loc[first_valid[c]:].isna().sum())
                for c in vw.columns if first_valid[c] is not None}
    return {
        "vintage": vintage(industry_text),
        "factors_vintage": vintage(factors_text),
        "first_date": str(vw.index[0].date()),
        "last_date": str(vw.index[-1].date()),
        "n_days": len(vw),
        "n_industries": int(vw.shape[1]),
        "first_valid": {c: (str(d.date()) if d is not None else None) for c, d in first_valid.items()},
        "missing_after_first_valid_since_1960": {c: n for c, n in interior.items() if n},
        "rf_annualised_mean": float((1.0 + rf).prod() ** (252.0 / len(rf)) - 1.0) if len(rf) else np.nan,
        "rf_dates_match_returns": bool(risk_free(factors).index.equals(vw.index)),
        "rf_source": rf_source,
    }
