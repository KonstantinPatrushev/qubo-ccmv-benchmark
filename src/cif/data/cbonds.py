"""Adapter for the Cbonds Investfunds dividend-calendar export (``.xlsx``).

The export is a flat list of announced dividends: one row per payment and **current** ticker.
What the layout does and does not carry (checked on the files of 2026-10-08):

* no ISIN and no payment status (a cancelled dividend is not marked), so a row is a claim that
  needs the cross-checks of :mod:`cif.data.moex_dividends`;
* the column "Дивидендная доходность, %" is the yield **at the time of the export**, not the
  yield at the payment; it is read but must not be used as a historical quantity;
* the ticker cell can hold several tickers of one security ("T, TCSG", "QIWI, NNDQ", "AFLT,
  RU_AFLT", "RUAL, 0486"): MOEX tickers, venue aliases and ADR tickers mixed;
* amounts are restated to the **current** share count after splits and consolidations, so older
  rows of GMKN, TRNFP, PLZL, VTBR, T do not carry the number the issuer declared;
* the ex-dividend and last-purchase dates are Cbonds' own and are not always right (weekend and
  holiday record dates, the 2022 closure), so they are a cross-check, not the ex-date.

Only the vendor format is handled here; mapping to the tickers and calendar of the project panel
is in :mod:`cif.data.moex_dividends`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cif.data import xlsx

#: Header of the export, in column order (Russian interface of Cbonds Investfunds).
CBONDS_HEADER: tuple[str, ...] = (
    "Название",
    "Биржа",
    "Тикер",
    "Отрасль",
    "Дивиденд",
    "Валюта",
    "Дивидендная доходность, %",
    "Последняя дата приобретения",
    "Экс-дивидендная дата",
    "Закрытие реестра",
    "Период",
    "Тип периода",
)

#: Header -> column name used in the project.
COLUMNS: dict[str, str] = dict(zip(
    CBONDS_HEADER,
    ("name", "exchange", "tickers", "sector", "amount", "currency", "yield_pct_current",
     "last_buy_date", "ex_date_cbonds", "record_date", "period", "period_type"),
    strict=True,
))

#: Fields without which a row is not a dividend at all; their absence stops the reader. The ticker
#: is not among them: three rows of the export of 2026-10-08 (Nanosoft) have none, and such a row
#: is kept with an empty alias so that the mapping can count it as unmapped instead of losing it.
KEY_FIELDS: tuple[str, ...] = ("amount", "currency", "ex_date_cbonds", "record_date")

_TEXT_COLUMNS = ("name", "exchange", "tickers", "sector", "currency", "period", "period_type")
_DATE_COLUMNS = ("last_buy_date", "ex_date_cbonds", "record_date")
_SEPARATOR = re.compile(r"\s*,\s*")


class CbondsFormatError(ValueError):
    """The file does not have the layout this adapter was written for."""


def _clean_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))  # "Период" typed as a number: 2012.0 -> "2012"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


def read_calendar(path: Path | str) -> pd.DataFrame:
    """Read one export into a typed table with project column names plus ``sheet_row``.

    Raises :class:`CbondsFormatError` when the header differs from :data:`CBONDS_HEADER`, when an
    amount is not a number, or when a key field (:data:`KEY_FIELDS`) is empty in any row: a row
    is never silently dropped.
    """
    raw = xlsx.read_table(path, row_numbers=True)
    header = list(raw.columns[:-1])
    if header != list(CBONDS_HEADER):
        unexpected = [h for h in header if h not in CBONDS_HEADER]
        missing = [h for h in CBONDS_HEADER if h not in header]
        raise CbondsFormatError(
            f"{path}: header differs from the documented Cbonds layout; "
            f"unexpected columns {unexpected}, missing columns {missing}"
        )
    t = raw.rename(columns=COLUMNS)
    for col in _TEXT_COLUMNS:
        t[col] = t[col].map(_clean_text)
    t["currency"] = t["currency"].map(lambda v: v.upper() if isinstance(v, str) else v)
    t["tickers"] = t["tickers"].fillna("")

    amount = pd.to_numeric(t["amount"], errors="coerce")
    bad = t["amount"].notna() & amount.isna()
    if bad.any():
        raise CbondsFormatError(
            f"{path}: column amount has non-numeric values {t.loc[bad, 'amount'].unique().tolist()[:5]}"
        )
    t["amount"] = amount.astype(float)
    t["yield_pct_current"] = pd.to_numeric(t["yield_pct_current"], errors="coerce")
    for col in _DATE_COLUMNS:
        try:
            t[col] = pd.to_datetime(t[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise CbondsFormatError(f"{path}: column {col} holds values that are not dates ({exc})") from exc

    for col in KEY_FIELDS:
        empty = t[col].isna() | (t[col].astype(str).str.strip() == "")
        if empty.any():
            rows = t.loc[empty, "sheet_row"].tolist()[:5]
            raise CbondsFormatError(f"{path}: column {col} is empty in {int(empty.sum())} rows (sheet rows {rows})")
    return t.reset_index(drop=True)


def split_tickers(cell) -> list[str]:
    """Tickers of a ticker cell: ``"T, TCSG"`` -> ``["T", "TCSG"]``; empty cell -> ``[]``."""
    if not isinstance(cell, str):
        return []
    return [part for part in _SEPARATOR.split(cell.strip()) if part]


def explode_tickers(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (payment, ticker alias).

    Adds ``alias`` (one ticker), ``alias_rank`` (its position in the cell) and ``n_aliases``
    (tickers in the cell). The alias rows of one payment share ``sheet_row``; which alias is the
    security of the project panel is decided later, from the quotes on the ex-date. A row with an
    empty ticker cell stays as one row with ``alias == ""`` and ``n_aliases == 0``.
    """
    out = frame.assign(alias=frame["tickers"].map(lambda cell: split_tickers(cell) or [""])).explode("alias")
    out = out.reset_index(drop=True)
    out["alias_rank"] = out.groupby("sheet_row").cumcount()
    out["n_aliases"] = out.groupby("sheet_row")["alias"].transform("size")
    out.loc[out["alias"] == "", "n_aliases"] = 0
    return out


def summarise(frame: pd.DataFrame) -> dict:
    """Row count, ex-date range and currency counts of an export (for the import record)."""
    return {
        "rows": len(frame),
        "ex_date_min": frame["ex_date_cbonds"].min().date().isoformat(),
        "ex_date_max": frame["ex_date_cbonds"].max().date().isoformat(),
        "currencies": {str(k): int(v) for k, v in frame["currency"].value_counts().items()},
    }
