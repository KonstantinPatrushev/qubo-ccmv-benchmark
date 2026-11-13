"""Universe resolution for S&P 500 and MOEX IMOEX.

Two strategies coexist:

1. **Live resolution** (default in `download_data.py`): fetch the current
   constituent list from a public source (Wikipedia for S&P 500, MOEX ISS for
   IMOEX), pick the top-N by some liquidity-adjacent proxy, and persist the
   result to a JSON snapshot under `data/raw/universe_<key>.json` so the rest
   of the pipeline always reads from a frozen list.

2. **Snapshot replay**: if the snapshot already exists, return it without
   touching the network. This is the path used inside the HPC notebook and the
   reproduce target.

The S&P 500 list is current-as-of-today, not point-in-time 2011. This
introduces a documented survivorship bias which we record in `provenance.json`
alongside the universe snapshot.

**Legacy only.** This module is kept so that the legacy tables of the 2026 ISP RAS paper can
be reproduced. Its S&P 500 "top-N" is *not* a capitalisation ranking (see
:func:`resolve_universe`). New work, including the paper's S&P 500 experiments, must take the
universe from :mod:`cif.data.sp500_pit` (index membership on the formation date, complete
history, ranking by median dollar volume, one ticker per company).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MOEX_INDEX_TICKERS_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/index/analytics/{index}.json"
)


@dataclass(frozen=True)
class UniverseSnapshot:
    key: str
    tickers: tuple[str, ...]
    source: str
    fetched_at: str
    notes: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "tickers": list(self.tickers),
            "source": self.source,
            "fetched_at": self.fetched_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> UniverseSnapshot:
        return cls(
            key=payload["key"],
            tickers=tuple(payload["tickers"]),
            source=payload["source"],
            fetched_at=payload["fetched_at"],
            notes=payload.get("notes", ""),
        )


def _snapshot_path(raw_dir: Path, key: str) -> Path:
    return raw_dir / f"universe_{key}.json"


def load_snapshot(raw_dir: Path, key: str) -> UniverseSnapshot | None:
    path = _snapshot_path(raw_dir, key)
    if not path.exists():
        return None
    return UniverseSnapshot.from_dict(json.loads(path.read_text()))


def save_snapshot(raw_dir: Path, snapshot: UniverseSnapshot) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(raw_dir, snapshot.key)
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False))
    return path


def fetch_sp500_tickers() -> list[str]:
    """Return the current S&P 500 constituent tickers from Wikipedia.

    The Wikipedia table is the most accessible source without a paid feed.
    Wikipedia rejects the default urllib User-Agent, so we fetch the HTML
    with ``requests`` first and then hand it to :func:`pandas.read_html`.

    The order is the order of the table, i.e. alphabetical by *company name*
    ("3M", "A. O. Smith", "Abbott", ... "Charles Schwab"), not by market
    capitalisation, and the list is today's membership, not membership on a
    past date. For the point-in-time universe used in the paper see
    :mod:`cif.data.sp500_pit`.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (cif research bot; "
            "https://github.com/konstantin/computing_in_finance) "
            "pandas/read_html"
        )
    }
    response = requests.get(WIKI_SP500_URL, headers=headers, timeout=30)
    response.raise_for_status()
    from io import StringIO

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    raw = df["Symbol"].astype(str).str.strip().tolist()
    cleaned = [t.replace(".", "-") for t in raw]
    return cleaned


def fetch_imoex_tickers() -> list[str]:
    """Return the current IMOEX constituent tickers via MOEX ISS API.

    The analytics endpoint pages at 20 rows by default. We pass an explicit
    ``limit=200`` so the full index composition (~40-50 names) comes back in
    one request.
    """
    url = MOEX_INDEX_TICKERS_URL.format(index="IMOEX")
    response = requests.get(url, params={"start": 0, "limit": 200}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    block = payload.get("analytics", {})
    columns = block.get("columns", [])
    data = block.get("data", [])
    if not columns or not data:
        raise RuntimeError(f"Empty IMOEX response from {url}")
    df = pd.DataFrame(data, columns=columns)
    if "ticker" not in df.columns:
        raise RuntimeError(f"Unexpected IMOEX schema: {columns}")
    if "weight" in df.columns:
        df = df.sort_values("weight", ascending=False)
    return list(dict.fromkeys(df["ticker"].astype(str).str.strip().tolist()))


def resolve_universe(
    key: str,
    raw_dir: Path,
    refresh: bool = False,
    top_n: int | None = None,
) -> UniverseSnapshot:
    """Return a universe snapshot, fetching fresh if requested or absent.

    Parameters
    ----------
    key:
        ``"sp500"`` or ``"moex"``.
    raw_dir:
        Where snapshots live, e.g. ``data/raw``.
    refresh:
        If True, ignore any existing snapshot and fetch a fresh list.
    top_n:
        Optional cap on the number of tickers to keep: the **first N rows in
        the order of the upstream source**. For ``"sp500"`` that source is the
        Wikipedia table, which is sorted alphabetically by company *name*, so
        ``top_n=100`` yields 3M ... Charles Schwab (no MSFT, NVDA, META, JPM),
        **not** the 100 largest companies. For ``"moex"`` the order is by index
        weight. Do not use it to build a capitalisation or liquidity ranking;
        the paper's S&P 500 universe comes from :mod:`cif.data.sp500_pit`.
        The behaviour is unchanged because the legacy tables depend on it; an
        existing snapshot is returned as is (``top_n`` is then not applied).
    """
    if key == "sp500" and top_n is not None:
        warnings.warn(
            f"resolve_universe('sp500', top_n={top_n}) takes the first {top_n} rows of the "
            "Wikipedia table, which is alphabetical by company name, not the largest companies, "
            "and it uses today's membership; use cif.data.sp500_pit for the paper's universe.",
            UserWarning,
            stacklevel=2,
        )
    raw_dir = Path(raw_dir)
    existing = None if refresh else load_snapshot(raw_dir, key)
    if existing is not None:
        return existing

    if key == "sp500":
        tickers = fetch_sp500_tickers()
        source = WIKI_SP500_URL
        notes = (
            "Current S&P 500 constituents (not point-in-time). "
            "Documented survivorship bias for the 2011-2025 backtest window."
        )
    elif key == "moex":
        tickers = fetch_imoex_tickers()
        source = "MOEX ISS analytics endpoint"
        notes = "Current IMOEX constituents via MOEX ISS API."
    else:
        raise ValueError(f"Unknown universe key: {key!r}")

    if top_n is not None:
        tickers = tickers[:top_n]

    snapshot = UniverseSnapshot(
        key=key,
        tickers=tuple(tickers),
        source=source,
        fetched_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        notes=notes,
    )
    save_snapshot(raw_dir, snapshot)
    return snapshot
