#!/usr/bin/env python
"""Build the risk-free series (WP-E1): FRED DTB3 for the US and CBR RUONIA for Russia.

Raw answers are stored verbatim in ``data/raw/riskfree_v2`` (``fred_DTB3.csv``,
``cbr_ruonia_<year>.xml``) and reused unless ``--refresh`` is given; processed files go to
``data/processed/riskfree_v2``:

* ``us_dtb3.parquet``   daily rate on the US trading calendar: ``y_pct`` (DTB3, discount basis),
  ``obs_date``, ``days``, ``r_daily`` and, for comparison, ``bey_pct``/``r_daily_bey``;
* ``ru_ruonia.parquet`` daily rate on the MOEX trading calendar (``data/processed/moex_v2``):
  ``y_pct``, ``obs_date``, ``days``, ``r_daily`` (ACT/365 simple);
* ``us_dtb3_obs.parquet`` / ``ru_ruonia_obs.parquet`` the underlying fixings;
* ``provenance.json`` URLs, download times, SHA-256 of every file, git state, counts.

The US trading calendar is taken from the legacy S&P 500 download (dates on which Yahoo returned
prices), which equals the NYSE calendar; for another panel call
:func:`cif.data.riskfree.daily_rate_on_calendar` with its own dates.

    python scripts/data/build_riskfree.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cif.data import moex, riskfree  # noqa: E402

logger = logging.getLogger("build_riskfree")

CONFIG = {
    "raw_dir": "data/raw/riskfree_v2",
    "processed_dir": "data/processed/riskfree_v2",
    "moex_calendar": "data/processed/moex_v2/calendar.parquet",
    "us_calendar_source": "data/raw/sp500_prices_raw.parquet",
    "fred_series": "DTB3",
    "ruonia_from": "2010-01-01",  # first fixings; the as-of join needs the days before 2013
    "ruonia_to": "2025-12-31",
    "pause": 1.0,
}


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "src", "scripts"))}


def load_fred(raw_dir: Path, refresh: bool) -> tuple[pd.Series, dict]:
    path = raw_dir / f"fred_{CONFIG['fred_series']}.csv"
    if refresh or not path.exists():
        series, raw = riskfree.fetch_fred_csv(CONFIG["fred_series"])
        path.write_bytes(raw)
        fetched = datetime.now(UTC).isoformat(timespec="seconds")
        (raw_dir / f"fred_{CONFIG['fred_series']}.fetched_at").write_text(fetched)
    series = riskfree.parse_fred_csv(path.read_text())
    series.name = CONFIG["fred_series"]
    fetched = (raw_dir / f"fred_{CONFIG['fred_series']}.fetched_at").read_text().strip()
    return series, {"url": f"{riskfree.FRED_CSV_URL}?id={CONFIG['fred_series']}", "fetched_at": fetched}


def load_ruonia(raw_dir: Path, refresh: bool) -> tuple[pd.DataFrame, dict]:
    frames, stamps = [], []
    for year in range(pd.Timestamp(CONFIG["ruonia_from"]).year, pd.Timestamp(CONFIG["ruonia_to"]).year + 1):
        path = raw_dir / f"cbr_ruonia_{year}.xml"
        stamp_path = raw_dir / f"cbr_ruonia_{year}.fetched_at"
        if refresh or not path.exists():
            df, raw = riskfree.fetch_ruonia(f"{year}-01-01", f"{year}-12-31")
            path.write_bytes(raw)
            stamp_path.write_text(datetime.now(UTC).isoformat(timespec="seconds"))
            time.sleep(CONFIG["pause"])
        frames.append(riskfree.parse_ruonia_soap(path.read_text(encoding="utf-8")))
        stamps.append(stamp_path.read_text().strip())
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, {"url": f"{riskfree.CBR_SOAP_URL} (SOAP method Ruonia, one call per year)",
                "fetched_at_first": min(stamps), "fetched_at_last": max(stamps)}


def us_calendar() -> pd.DatetimeIndex:
    path = REPO / CONFIG["us_calendar_source"]
    if not path.exists():
        raise SystemExit(f"{path} is needed as the NYSE calendar (legacy S&P 500 download)")
    return pd.DatetimeIndex(pd.read_parquet(path).index).sort_values().unique()


def summarize(daily: pd.DataFrame, label: str) -> dict:
    ok = daily["r_daily"].dropna()
    span_days = (daily.index[-1] - daily.index[0]).days
    return {
        "label": label, "first_day": daily.index[0].date().isoformat(), "last_day": daily.index[-1].date().isoformat(),
        "n_days": len(daily), "n_missing_r_daily": int(daily["r_daily"].isna().sum()),
        "mean_y_pct": float(daily["y_pct"].mean()), "min_y_pct": float(daily["y_pct"].min()),
        "max_y_pct": float(daily["y_pct"].max()),
        "compounded_annual_return": float((1 + ok).prod() ** (365.25 / span_days) - 1),
        "max_staleness_days": int((daily.index.to_series() - daily["obs_date"]).dt.days.max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="download again instead of reading data/raw")
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    raw_dir, out_dir = REPO / CONFIG["raw_dir"], REPO / CONFIG["processed_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtb3, fred_src = load_fred(raw_dir, args.refresh)
    ruonia, cbr_src = load_ruonia(raw_dir, args.refresh)
    logger.info("DTB3: %d fixings %s..%s", len(dtb3), dtb3.index[0].date(), dtb3.index[-1].date())
    logger.info("RUONIA: %d fixings %s..%s", len(ruonia), ruonia.index[0].date(), ruonia.index[-1].date())

    # ---- US -------------------------------------------------------------------------------
    us_cal = us_calendar()
    us = riskfree.daily_rate_on_calendar(dtb3, us_cal, compounding="compound")
    bey = riskfree.discount_to_bey(dtb3)
    us_bey = riskfree.daily_rate_on_calendar(bey, us_cal, compounding="compound")
    us["bey_pct"] = us_bey["y_pct"]
    us["r_daily_bey"] = us_bey["r_daily"]
    us.to_parquet(out_dir / "us_dtb3.parquet")
    pd.DataFrame({"y_pct": dtb3, "bey_pct": bey}).rename_axis("date").to_parquet(out_dir / "us_dtb3_obs.parquet")

    # ---- RU -------------------------------------------------------------------------------
    cal_path = REPO / CONFIG["moex_calendar"]
    if not cal_path.exists():
        raise SystemExit(f"{cal_path} not found: run build_moex_dataset.py build first")
    cal = pd.read_parquet(cal_path)
    ru_days = cal.index[cal["tradable"]]
    ru = riskfree.daily_rate_on_calendar(ruonia["y_pct"], ru_days, compounding="simple",
                                         available_from=ruonia["published"])
    ru["published"] = ruonia["published"].reindex(ru["obs_date"]).to_numpy()
    ru.to_parquet(out_dir / "ru_ruonia.parquet")
    ruonia.rename_axis("date").to_parquet(out_dir / "ru_ruonia_obs.parquet")

    # ---- cross-checks that go into the provenance ------------------------------------------------
    closure = ru.loc["2022-03-24"] if pd.Timestamp("2022-03-24") in ru.index else None
    def annualized(frame: pd.DataFrame, col: str) -> float:
        years = (frame.index[-1] - frame.index[0]).days / 365.25
        return float((1.0 + frame[col].dropna()).prod() ** (1.0 / years) - 1.0)

    checks = {
        "us_bey_minus_published_pp_mean": float((us["bey_pct"] - us["y_pct"]).mean()),
        "us_annual_return_published_vs_bey_pp": [100 * annualized(us, "r_daily"), 100 * annualized(us, "r_daily_bey")],
        "ru_return_on_first_day_after_2022_closure": None if closure is None else {
            "days": float(closure["days"]), "r_daily": float(closure["r_daily"]), "y_pct": float(closure["y_pct"])},
        "ru_first_day_nan": bool(np.isnan(ru["r_daily"].iloc[0])),
    }
    prov = {
        "dataset": "riskfree_v2", "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(), "config": CONFIG,
        "code_sha256": {p: moex.sha256_file(REPO / p) for p in ("src/cif/data/riskfree.py",
                                                                "scripts/data/build_riskfree.py")},
        "sources": {
            "us": {"series": "FRED DTB3: 3-Month Treasury Bill Secondary Market Rate, Discount Basis (H.15), "
                             "percent per year", **fred_src},
            "ru": {"series": "CBR RUONIA overnight rate, percent per year, ACT/365; 'published' is the "
                             "publication date of each fixing", **cbr_src},
            "ru_key_rate_fallback": "not needed: the RUONIA web service is available",
        },
        "method": {
            "accrual": "trading day d earns the rate known at the close of the previous trading day p "
                       "over the (d - p) calendar days (no look-ahead)",
            "us": "r = (1 + y/100) ** (days/365) - 1 on the published DTB3; r_daily_bey applies it to the "
                  "bond-equivalent yield 365*d/(360 - d*91)",
            "ru": "r = y/100 * days/365; a fixing is known from its publication date (the next business day)",
            "us_calendar": f"dates of {CONFIG['us_calendar_source']} (NYSE days)",
            "ru_calendar": f"tradable days of {CONFIG['moex_calendar']}",
        },
        "summary": {"us": summarize(us, "us_dtb3"), "ru": summarize(ru, "ru_ruonia")},
        "checks": checks,
    }
    prov["files"] = {"raw": moex.file_manifest(raw_dir), "processed": moex.file_manifest(out_dir)}
    (out_dir / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    logger.info("US %s", prov["summary"]["us"])
    logger.info("RU %s", prov["summary"]["ru"])
    logger.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
