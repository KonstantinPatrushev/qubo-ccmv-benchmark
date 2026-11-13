#!/usr/bin/env python
"""Build the MOEX v2 dataset (WP-E1, R6): point-in-time IMOEX universe, split-adjusted and
total-return prices, trading calendar, with a provenance file.

Replaces the legacy ``cif.scripts.download_data --universe moex`` (price return of *today's* index
members; its output ``data/raw/moex_prices_raw.parquet`` and ``data/processed/moex_*.parquet`` is
left untouched because the old tables are reproduced from it).

Stages (all network access is in the ``fetch-*`` stages and is cached verbatim in
``data/raw/moex_v2``; ``build`` is offline and deterministic)::

    python scripts/data/build_moex_dataset.py fetch-core      # index levels, tickers, composition, splits
    python scripts/data/build_moex_dataset.py fetch-prices    # board-group history of every ticker
    python scripts/data/build_moex_dataset.py fetch-evidence  # per-member analytics around jumps
    python scripts/data/build_moex_dataset.py build [--dividends-csv PATH]
    python scripts/data/build_moex_dataset.py all [--dividends-csv PATH]

Parameters live in ``DEFAULT_CONFIG`` below; ``--config file.yaml`` overrides single keys and the
effective configuration is written to ``provenance.json`` (R8).

Dividends: ISS has no public dividend endpoint (see :mod:`cif.data.moex`). ``--dividends-csv``
takes a table with the columns ``secid, record_date, amount, currency[, source]``: ``record_date``
is the date of closing of the register, ``amount`` the gross cash dividend per share in the share
basis at the ex-date, ``currency`` e.g. ``RUB`` (other currencies are excluded and counted).
Without the file ``tr_index.parquet`` and ``dividends.parquet`` are **not** written (a TR index
equal to the price index would be mislabelled) and ``provenance.json`` says so.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cif.data import moex  # noqa: E402
from cif.data.cleaning import clean_prices  # noqa: E402

logger = logging.getLogger("build_moex_dataset")

DEFAULT_CONFIG: dict = {
    "index": "IMOEX",
    "benchmarks": ["MCFTR", "MCFTRR"],
    "composition_start": "2013-01-01",  # membership snapshots start here
    "price_start": moex.MAIN_MARKET_START.date().isoformat(),  # board group 57 is the main market
    "end": "2025-12-31",
    "boardgroup": moex.BOARDGROUP_T_PLUS_SHARES,
    "pause": 0.25,
    "retries": 5,
    "jump_threshold": 0.35,  # scan level for the evidence stage (below the legacy 0.60)
    "evidence_window_days": 15,
    "raw_dir": "data/raw/moex_v2",
    "processed_dir": "data/processed/moex_v2",
    "legacy_universe": "data/raw/universe_moex.json",
    "legacy_raw_prices": "data/raw/moex_prices_raw.parquet",
}


SCHEMA = {
    "prices_close.parquet": "wide; index date (tradable days from price_start), columns secid; unadjusted CLOSE "
                            "of board group 57 in RUB; NaN = no quote that day",
    "prices_adj.parquet": "same shape; CLOSE adjusted for splits/consolidations only, in the share basis of the "
                          "last date (splits.parquet, applied == True)",
    "tr_index.parquet": "same shape; total-return index, base 100 on the first quote of each ticker, gross "
                        "cash dividends reinvested on the ex-date (written only with --dividends-csv)",
    "value_rub.parquet": "same shape; turnover of the ticker on the day in RUB (0 = no trades)",
    "traded_mask.parquet": "same shape, bool; quote present and volume > 0 while the market was open",
    "imoex_membership.parquet": "long; date = first trading day of the month, ticker, secid, weight (fraction), "
                                "weight_pct (percent as published), shortname",
    "calendar.parquet": "index = every calendar day; tradable (IMOEX calculated from trades), is_weekday, "
                        "in_closure (inside a run of >= 10 closed weekdays, i.e. 28.02-23.03.2022)",
    "splits.parquet": "corporate actions: secid, tradedate (first day on the new share basis), before, after "
                      "(before old shares become after new ones), ratio = after / before, source, applied, "
                      "price_prev, price, net_return = price move net of the split",
    "dividends.parquet": "prepared dividends: secid, record_date, amount, currency, ex_date, settlement_lag "
                         "(written only with --dividends-csv)",
    "jumps.parquet": "every close-to-close move above jump_threshold with its final status and the share-count "
                     "evidence (issue_size_before/after, weights)",
    "universe.parquet": "every secid that was in IMOEX in the window or in the legacy universe, with first/last "
                        "quote and number of quotes",
    "index_levels.parquet": "CLOSE of IMOEX, MCFTR (gross total return), MCFTRR (net) by date",
    "index_dividend_flow.parquet": "dividend_flow = MCFTR return - IMOEX return on tradable days: the "
                                   "aggregate gross dividend yield of the shares going ex-dividend that day "
                                   "(the only dividend information in the public ISS data)",
}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg


def make_client(cfg: dict, **kw) -> moex.IssClient:
    return moex.IssClient(REPO / cfg["raw_dir"], pause=cfg["pause"], retries=cfg["retries"], **kw)


# --------------------------------------------------------------------------------------------
# Cached building blocks shared by all stages
# --------------------------------------------------------------------------------------------


def index_levels(client: moex.IssClient, cfg: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for ix in [cfg["index"], *cfg["benchmarks"]]:
        out[ix] = moex.fetch_index_history(client, ix, cfg["composition_start"], cfg["end"])
    return out


def trading_calendar(levels: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    days = moex.calculated_trading_days(levels[cfg["index"]])
    return moex.build_calendar(days, cfg["composition_start"], cfg["end"])


def monthly_composition(client: moex.IssClient, cfg: dict, calendar: pd.DataFrame) -> pd.DataFrame:
    days = calendar.index[calendar["tradable"]]
    frames = []
    for d in moex.first_trading_days_of_month(days):
        frames.append(moex.fetch_index_composition(client, cfg["index"], d))
    return pd.concat(frames, ignore_index=True)


def universe_table(
    composition: pd.DataFrame, all_time: pd.DataFrame, cfg: dict, legacy: list[str]
) -> pd.DataFrame:
    """Every secid that was ever in the index inside the window, plus the legacy tickers."""
    start, end = pd.Timestamp(cfg["composition_start"]), pd.Timestamp(cfg["end"])
    seen = composition.groupby("secid").agg(first_snapshot=("date", "min"), last_snapshot=("date", "max"),
                                            n_snapshots=("date", "nunique"))
    within = all_time[(all_time["till"] >= start) & (all_time["from"] <= end)].set_index("ticker")
    uni = seen.join(within[["from", "till"]].rename(columns={"from": "index_from", "till": "index_till"}),
                    how="outer")
    uni["n_snapshots"] = uni["n_snapshots"].fillna(0).astype(int)
    uni["in_snapshots"] = uni["n_snapshots"] > 0
    uni["in_all_time_list"] = uni["index_from"].notna()
    uni["legacy_universe"] = uni.index.isin(legacy)
    for t in legacy:
        if t not in uni.index:
            uni.loc[t, ["n_snapshots", "in_snapshots", "in_all_time_list", "legacy_universe"]] = [0, False, False, True]
    uni.index.name = "secid"
    return uni.sort_index()


def legacy_tickers(cfg: dict) -> list[str]:
    path = REPO / cfg["legacy_universe"]
    return json.loads(path.read_text())["tickers"] if path.exists() else []


# --------------------------------------------------------------------------------------------
# Fetch stages
# --------------------------------------------------------------------------------------------


def fetch_core(cfg: dict) -> None:
    client = make_client(cfg)
    levels = index_levels(client, cfg)
    for ix, df in levels.items():
        logger.info("index %s: %d days %s..%s", ix, len(df), df["TRADEDATE"].min().date(),
                    df["TRADEDATE"].max().date())
    cal = trading_calendar(levels, cfg)
    all_time = moex.fetch_index_tickers(client, cfg["index"])
    logger.info("all-time %s tickers: %d", cfg["index"], len(all_time))
    comp = monthly_composition(client, cfg, cal)
    logger.info("monthly composition: %d snapshots, %d rows, %d secids", comp["date"].nunique(),
                len(comp), comp["secid"].nunique())
    splits = moex.fetch_splits(client)
    logger.info("split register: %d events", len(splits))
    logger.info("requests made: %d", client.n_requests)


def fetch_prices(cfg: dict, only: list[str] | None = None) -> None:
    client = make_client(cfg)
    levels = index_levels(client, cfg)
    cal = trading_calendar(levels, cfg)
    comp = monthly_composition(client, cfg, cal)
    all_time = moex.fetch_index_tickers(client, cfg["index"])
    uni = universe_table(comp, all_time, cfg, legacy_tickers(cfg))
    tickers = only or list(uni.index)
    logger.info("fetching board-group history for %d tickers", len(tickers))
    for i, secid in enumerate(tickers, 1):
        before = client.n_requests
        df = moex.fetch_board_history(client, secid, cfg["price_start"], cfg["end"], cfg["boardgroup"])
        logger.info("[%d/%d] %s rows=%d first=%s last=%s requests=%d", i, len(tickers), secid, len(df),
                    df["TRADEDATE"].min().date() if len(df) else None,
                    df["TRADEDATE"].max().date() if len(df) else None, client.n_requests - before)
    logger.info("requests made: %d", client.n_requests)


def load_history(client: moex.IssClient, cfg: dict, tickers: list[str]) -> pd.DataFrame:
    frames = [moex.fetch_board_history(client, t, cfg["price_start"], cfg["end"], cfg["boardgroup"])
              for t in tickers]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True)


def member_window(cfg: dict, date: pd.Timestamp, prev_date=None) -> tuple[str, str]:
    """Calendar window of the per-member analytics around an event (must cover ``prev_date``)."""
    w = pd.Timedelta(days=cfg["evidence_window_days"])
    start = date - w
    if prev_date is not None and pd.notna(prev_date):
        start = min(start, prev_date - pd.Timedelta(days=5))
    return start.date().isoformat(), (date + w).date().isoformat()


def evidence_events(cfg: dict, client: moex.IssClient) -> pd.DataFrame:
    """Jumps above the scan threshold plus every register split of a universe ticker."""
    levels = index_levels(client, cfg)
    cal = trading_calendar(levels, cfg)
    comp = monthly_composition(client, cfg, cal)
    uni = universe_table(comp, moex.fetch_index_tickers(client, cfg["index"]), cfg, legacy_tickers(cfg))
    hist = load_history(client, cfg, list(uni.index))
    panels = moex.history_to_panels(hist, cal.index[cal["tradable"]])
    splits = moex.fetch_splits(client)
    jumps = moex.reconcile_jumps(moex.detect_jumps(panels["close"], cfg["jump_threshold"]), splits)
    reg = splits[splits["secid"].isin(uni.index)]
    extra = pd.DataFrame({"ticker": reg["secid"], "date": reg["tradedate"], "prev_date": pd.NaT})
    ev = pd.concat([jumps[["ticker", "date", "prev_date"]], extra], ignore_index=True)
    ev = ev.sort_values(["ticker", "date", "prev_date"]).drop_duplicates(["ticker", "date"])
    return ev.reset_index(drop=True)


def fetch_evidence(cfg: dict) -> None:
    client = make_client(cfg)
    ev = evidence_events(cfg, client)
    logger.info("evidence for %d (ticker, date) events", len(ev))
    for r in ev.itertuples():
        start, end = member_window(cfg, r.date, r.prev_date)
        df = moex.fetch_index_ticker_series(client, cfg["index"], r.ticker, start, end)
        logger.info("%s %s: %d member-days", r.ticker, r.date.date(), len(df))
    logger.info("requests made: %d", client.n_requests)


PROBES = [
    {"name": "securities_dividends", "path": "securities/SBER/dividends.json",
     "note": "documented in the task; expected block 'dividends'", "expect_block": "dividends"},
    {"name": "cci_corp_actions_dividends", "path": "cci/corp-actions/dividends.json",
     "note": "listed in /iss/reference; Corporate Information Services", "expect_block": "dividends"},
]


def probe_endpoints(cfg: dict) -> None:
    """Record what the candidate dividend endpoints actually answer (raw evidence for provenance)."""
    import re

    import requests

    out = []
    for pr in PROBES:
        url = f"{moex.ISS_BASE}/{pr['path']}"
        resp = requests.get(url, params={"iss.meta": "off"}, headers={"User-Agent": moex.USER_AGENT}, timeout=60)
        rec = {**pr, "url": url, "http_status": resp.status_code,
               "content_type": resp.headers.get("Content-Type"),
               "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
        try:
            payload = resp.json()
            rec["blocks"] = sorted(payload)
            rec["has_expected_block"] = pr["expect_block"] in payload
        except ValueError:
            text = re.sub(r"<style.*?</style>", "", resp.content.decode("utf-8", "replace"), flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            rec["not_json"] = True
            rec["text_snippet"] = re.sub(r"\s+", " ", text).strip()[:200]
            rec["has_expected_block"] = False
        out.append(rec)
        logger.info("probe %s: HTTP %s, has_expected_block=%s", pr["name"], resp.status_code, rec["has_expected_block"])
    path = REPO / cfg["raw_dir"] / "endpoint_probes.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------------------------
# Build stage (offline)
# --------------------------------------------------------------------------------------------


def classify_jumps(cfg: dict, client: moex.IssClient, close: pd.DataFrame, splits: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile price jumps with the split register and the share-count evidence.

    Returns ``(jumps, actions)``: every jump with its final ``status`` and the table of corporate
    actions to apply (register events of universe tickers inside the window, plus jumps that
    coincide with a share-count change in the index analytics).
    """
    jumps = moex.reconcile_jumps(moex.detect_jumps(close, cfg["jump_threshold"]), splits)
    extra = {c: [] for c in ("issue_size_before", "issue_size_after", "issue_ratio", "weight_before",
                             "weight_after", "evidence_status")}
    final = []
    derived = []
    for j in jumps.itertuples():
        member = moex.fetch_index_ticker_series(client, cfg["index"], j.ticker,
                                                *member_window(cfg, j.date, j.prev_date))
        ev = moex.share_count_change(member, j.prev_date, j.date)
        extra["evidence_status"].append(ev["status"])
        for k in ("issue_size_before", "issue_size_after", "issue_ratio", "weight_before", "weight_after"):
            extra[k].append(ev.get(k, np.nan))
        status = moex.classify_jump(j.status == "iss_split", ev, j.price / j.price_prev)
        if status == "share_count_change":
            try:
                moex.as_fraction(ev["issue_size_before"], ev["issue_size_after"])
                derived.append((j, ev, j.price / j.price_prev * ev["issue_ratio"] - 1.0))
            except ValueError:
                status = "share_count_change_not_simple"  # reported, not adjusted
        final.append(status)
    jumps = jumps.assign(status=final, **extra)

    start, end = pd.Timestamp(cfg["price_start"]), pd.Timestamp(cfg["end"])
    reg = splits[splits["secid"].isin(close.columns)]
    rows = []
    for r in reg.itertuples():
        applied = start < r.tradedate <= end
        s = close[r.secid].dropna() if r.secid in close else pd.Series(dtype=float)
        prev, cur = s[s.index < r.tradedate], s[s.index >= r.tradedate]
        pp, pc = (prev.iloc[-1] if len(prev) else np.nan), (cur.iloc[0] if len(cur) else np.nan)
        rows.append({"secid": r.secid, "tradedate": r.tradedate, "before": int(r.before), "after": int(r.after),
                     "ratio": r.after / r.before, "source": "iss statistics/engines/stock/splits",
                     "applied": bool(applied), "price_prev": pp, "price": pc,
                     "net_return": pc / pp * (r.after / r.before) - 1.0 if applied else np.nan})
    for j, ev, net in derived:
        before, after = moex.as_fraction(ev["issue_size_before"], ev["issue_size_after"])
        rows.append({"secid": j.ticker, "tradedate": j.date, "before": before, "after": after,
                     "ratio": ev["issue_ratio"], "source": "iss index analytics issue_size_total",
                     "applied": True, "price_prev": j.price_prev, "price": j.price, "net_return": net})
    actions = pd.DataFrame(rows).sort_values(["secid", "tradedate"]).reset_index(drop=True)
    return jumps, actions


def cagr(level: pd.Series) -> float:
    level = level.dropna()
    years = (level.index[-1] - level.index[0]).days / 365.25
    return float((level.iloc[-1] / level.iloc[0]) ** (1.0 / years) - 1.0)


def ew_level(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight, daily-rebalanced buy-and-hold index from simple returns."""
    r = returns.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + r).cumprod()


def pad_variant_clean(prices: pd.DataFrame, th: float = 0.60) -> tuple[pd.DataFrame, list]:
    """``clean_prices`` with the pandas<2.2 ``pct_change()`` default (``fill_method='pad'``)."""
    filled = prices.ffill(limit=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # the deprecated default is the point here
        returns = filled.pct_change(fill_method="pad")
    flagged = returns.abs() > th
    corrected, events = filled.copy(), []
    for t in flagged.columns:
        for d in filled.index[flagged[t].fillna(False)]:
            r = returns.loc[d, t]
            corrected.loc[d:, t] = corrected.loc[d:, t] / (1.0 + r)
            events.append({"ticker": t, "date": d.date().isoformat(), "r": float(r)})
    nan_ratio = corrected.isna().mean()
    surv = corrected.drop(columns=nan_ratio[nan_ratio > 0.10].index)
    start = surv.apply(lambda s: s.first_valid_index()).max()
    return surv.loc[start:].dropna(how="any"), events


def legacy_reconciliation(cfg: dict) -> dict:
    """Why the published paper says 7 corrections while the current detector finds 5."""
    raw_path, proc_path = REPO / cfg["legacy_raw_prices"], REPO / "data/processed/moex_prices.parquet"
    if not (raw_path.exists() and proc_path.exists()):
        return {"status": "legacy files not found"}
    raw, proc = pd.read_parquet(raw_path), pd.read_parquet(proc_path)

    def events(returns, th):
        f = returns.abs() > th
        return [{"ticker": t, "date": d.date().isoformat(), "r": float(returns.loc[d, t])}
                for t in f.columns for d in returns.index[f[t].fillna(False)]]

    current = events(raw.ffill(limit=5).pct_change(fill_method=None), 0.60)
    pad, pad_events = pad_variant_clean(raw)
    same = pad.shape == proc.shape and bool(np.allclose(pad.to_numpy(), proc.to_numpy(), rtol=1e-9, atol=1e-12))
    sweep = {str(th): len(events(raw.ffill(limit=5).pct_change(fill_method=None), th))
             for th in (0.4, 0.45, 0.5, 0.6, 0.7, 0.9)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cur_clean, _ = clean_prices(raw, max_abs_daily_return=0.60)
    irao_leak = float(cur_clean["IRAO"].pct_change().max()) if "IRAO" in cur_clean else None
    return {
        "max_daily_return_of_IRAO_after_current_code_with_threshold_0.60": irao_leak,
        "current_code_events_threshold_0.60": current,
        "events_by_threshold_current_code": sweep,
        "pad_semantics_events_threshold_0.60": pad_events,
        "pad_semantics_reproduces_stored_processed_file": same,
        "stored_processed_shape": list(proc.shape),
        "explanation": (
            "the stored data/processed/moex_prices.parquet is reproduced exactly only when pct_change() "
            "forward-fills without limit (pandas<2.2 default fill_method='pad'); that variant flags 7 "
            "events: the 5 of the current code plus IRAO 2015-01-20 (100:1 consolidation after a "
            "halt, a real corporate action) and VKCO 2022-03-29 (a move across the 2022 closure, a "
            "false positive on a series dropped later). The current code (fill_method=None, ffill limit 5) "
            "no longer sees IRAO because the 12-day halt exceeds the fill limit."),
    }


def afks_investigation(cfg: dict, client: moex.IssClient, close, volume, value, levels, jumps, splits) -> dict:
    t, d = "AFKS", pd.Timestamp("2014-12-18")
    s = close[t].dropna()
    prev_date = s.index[s.index < d][-1]
    post = s[s.index >= d].iloc[:11]
    pre_val = value[t][(value.index < d)].dropna().iloc[-20:]
    imoex = levels[cfg["index"]].set_index("TRADEDATE")["CLOSE"]
    j = jumps[(jumps["ticker"] == t) & (jumps["date"] == d)]
    member = moex.fetch_index_ticker_series(client, cfg["index"], t, *member_window(cfg, d, prev_date))
    ev = moex.share_count_change(member, prev_date, d)
    idx_ret = float(imoex.loc[d] / imoex.shift(1).loc[d] - 1.0)
    out = {
        "ticker": t, "date": d.date().isoformat(), "prev_date": prev_date.date().isoformat(),
        "close_prev": float(s.loc[prev_date]), "close_jump_day": float(s.loc[d]),
        "r": float(s.loc[d] / s.loc[prev_date] - 1.0),
        "closes_next_10_days": [round(float(x), 2) for x in post.iloc[1:].to_numpy()],
        "median_close_next_10_days_over_jump_day_close": float(post.iloc[1:].median() / s.loc[d]),
        "min_close_next_10_days_over_prev_close": float(post.iloc[1:].min() / s.loc[prev_date]),
        "value_rub_jump_day": float(value[t].loc[d]),
        "median_value_rub_prev_20_days": float(pre_val.median()),
        "value_multiple": float(value[t].loc[d] / pre_val.median()),
        "imoex_return_jump_day": idx_ret,
        "in_iss_split_register": bool(((splits["secid"] == t) & (splits["tradedate"].between(prev_date, d))).any()),
        "share_count_evidence": ev,
        "jump_status_in_this_dataset": j["status"].iloc[0] if len(j) else None,
    }
    held = out["median_close_next_10_days_over_jump_day_close"] > 0.75 and out["min_close_next_10_days_over_prev_close"] > 1.0
    unchanged = ev.get("status") == "ok" and abs(ev["issue_ratio"] - 1.0) < 0.005
    out["verdict"] = (
        "real market move, not a corporate action: share count unchanged, no register record, the new "
        "level held and turnover spiked; the legacy |r|>60% rebasing deleted it"
        if held and unchanged and not out["in_iss_split_register"] else "inconclusive: see fields")
    return out


def read_dividends(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path)
    missing = {"secid", "record_date", "amount", "currency"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    return df


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.date().isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT:
        return None
    return obj


def fetched_range(root: Path) -> dict:
    """First and last download time of the cached tables, per sub-directory of the raw cache."""
    stamps: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*.json")):
        if "_logs" in p.parts or p.name == "endpoint_probes.json":
            continue
        try:
            stamp = json.loads(p.read_text(encoding="utf-8")).get("fetched_at")
        except (ValueError, UnicodeDecodeError):
            continue
        if stamp:
            stamps.setdefault(p.relative_to(root).parts[0], []).append(stamp)
    return {k: {"first": min(v), "last": max(v), "n_files": len(v)} for k, v in stamps.items()}


def build(cfg: dict, dividends_csv: str | None = None) -> None:
    client = make_client(cfg, offline=True)
    raw_dir, out_dir = REPO / cfg["raw_dir"], REPO / cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(cfg["price_start"]), pd.Timestamp(cfg["end"])

    levels = index_levels(client, cfg)
    cal = trading_calendar(levels, cfg)
    spells = moex.closure_spells(cal)
    cal["in_closure"] = False
    for sp in spells.itertuples():
        cal.loc[sp.start:sp.end, "in_closure"] = True
    comp = monthly_composition(client, cfg, cal)
    all_time = moex.fetch_index_tickers(client, cfg["index"])
    uni = universe_table(comp, all_time, cfg, legacy_tickers(cfg))
    hist = load_history(client, cfg, list(uni.index))
    tradable = cal.index[cal["tradable"] & (cal.index >= start) & (cal.index <= end)]
    panels = moex.history_to_panels(hist, tradable)
    off_calendar = hist[~hist["TRADEDATE"].isin(cal.index[cal["tradable"]]) & (hist["VOLUME"].fillna(0) > 0)]
    has_data = panels["close"].notna().any()
    close, volume, value = (panels[k].loc[:, has_data] for k in ("close", "volume", "value"))
    traded = moex.ticker_traded_mask(close, volume)
    splits = moex.fetch_splits(client)
    jumps, actions = classify_jumps(cfg, client, close, splits)
    applied = actions[actions["applied"]].rename(columns={"secid": "secid"})
    adj = pd.DataFrame({t: moex.adjust_for_splits(close[t], applied[applied["secid"] == t]) for t in close.columns})

    uni["first_price"] = close.apply(lambda s: s.first_valid_index())
    uni["last_price"] = close.apply(lambda s: s.last_valid_index())
    uni["n_quotes"] = close.notna().sum()
    uni["n_traded_days"] = traded.sum()
    uni["n_quotes"] = uni["n_quotes"].fillna(0).astype(int)
    uni["n_traded_days"] = uni["n_traded_days"].fillna(0).astype(int)

    # --- dividends and total return ------------------------------------------------------
    div_raw = read_dividends(dividends_csv)
    dividends, div_report, tr = None, {"status": "missing"}, None
    if div_raw is not None:
        dividends, div_report = moex.prepare_dividends(div_raw, tradable, tickers=close.columns)
        div_report["status"] = "ok"
        div_report["source_file"] = str(dividends_csv)
        tr = pd.DataFrame({t: moex.total_return_index(
            close[t], dividends[dividends["secid"] == t], applied[applied["secid"] == t]) for t in close.columns})
    else:
        div_report["reason"] = (
            "ISS has no public dividend table: securities/{secid}/dividends.json answers with the security "
            "description and cci/corp-actions/dividends with an HTML 'subscribers only' page "
            "(raw evidence: data/raw/moex_v2/endpoint_probes.json). Supply --dividends-csv.")

    # --- write ---------------------------------------------------------------------------
    for name, frame in (("prices_close", close), ("prices_adj", adj), ("value_rub", value), ("traded_mask", traded)):
        frame.rename_axis("date").to_parquet(out_dir / f"{name}.parquet")
    if tr is not None:
        tr.rename_axis("date").to_parquet(out_dir / "tr_index.parquet")
        dividends.to_parquet(out_dir / "dividends.parquet", index=False)
    membership = comp.assign(weight_pct=comp["weight"], weight=comp["weight"] / 100.0)
    membership[["date", "ticker", "secid", "weight", "weight_pct", "shortname"]].to_parquet(
        out_dir / "imoex_membership.parquet", index=False)
    cal.to_parquet(out_dir / "calendar.parquet")
    actions.to_parquet(out_dir / "splits.parquet", index=False)
    jumps.to_parquet(out_dir / "jumps.parquet", index=False)
    uni.reset_index().to_parquet(out_dir / "universe.parquet", index=False)
    lv = pd.DataFrame({ix: df.set_index("TRADEDATE")["CLOSE"] for ix, df in levels.items()}).rename_axis("date")
    lv.to_parquet(out_dir / "index_levels.parquet")
    open_days = cal.index[cal["tradable"]]
    flow = moex.index_dividend_flow(lv[cfg["index"]].reindex(open_days), lv["MCFTR"].reindex(open_days))
    flow.rename_axis("date").to_frame().to_parquet(out_dir / "index_dividend_flow.parquet")

    # --- diagnostics and investigations -------------------------------------------------------
    win = (pd.Timestamp("2014-06-09"), pd.Timestamp("2025-12-30"))
    lvw = lv.loc[win[0]:win[1]]
    legacy = [t for t in legacy_tickers(cfg) if t in adj.columns]
    adjw = adj.loc[win[0]:win[1], legacy]
    full = adjw.columns[adjw.notna().mean() > 0.995]
    ew_price = ew_level(adjw[full].ffill(limit=5).pct_change(fill_method=None))
    fw = flow.loc[win[0]:win[1]]
    diag = {
        "index_dividend_flow": {
            "definition": "MCFTR return minus IMOEX return on tradable days (aggregate gross dividend yield "
                          "of the shares going ex that day)",
            "sum_by_year_pct": (100 * fw.groupby(fw.index.year).sum()).round(3).to_dict(),
            "days_above_10bp": int((fw > 0.001).sum()),
            "largest_days": {d.date().isoformat(): round(float(v), 5) for d, v in fw.nlargest(8).items()},
        },
        "window": [d.date().isoformat() for d in win],
        "index_cagr": {ix: cagr(lvw[ix]) for ix in lvw.columns},
        "index_dividend_effect_pp_per_year": {
            "MCFTR_minus_IMOEX": 100 * (cagr(lvw["MCFTR"]) - cagr(lvw[cfg["index"]])),
            "MCFTRR_minus_IMOEX": 100 * (cagr(lvw["MCFTRR"]) - cagr(lvw[cfg["index"]]))},
        "equal_weight_legacy_price_cagr": cagr(ew_price),
        "equal_weight_legacy_tickers": list(full),
    }
    if tr is not None:
        trw = tr.loc[win[0]:win[1], full]
        ew_tr = ew_level(trw.ffill(limit=5).pct_change(fill_method=None))
        diag["equal_weight_legacy_total_return_cagr"] = cagr(ew_tr)
        diag["equal_weight_dividend_effect_pp_per_year"] = 100 * (cagr(ew_tr) - cagr(ew_price))

    afks = afks_investigation(cfg, client, close, volume, value, levels, jumps, splits)
    legacy_rec = legacy_reconciliation(cfg)

    # --- provenance ---------------------------------------------------------------------------
    code = {p: moex.sha256_file(REPO / p) for p in ("src/cif/data/moex.py", "src/cif/data/cleaning.py",
                                                    "scripts/data/build_moex_dataset.py")}
    probes = json.loads((raw_dir / "endpoint_probes.json").read_text()) if (raw_dir / "endpoint_probes.json").exists() else []
    prov = {
        "dataset": "moex_v2", "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(), "code_sha256": code, "config": cfg,
        "sources": {
            "iss_base": moex.ISS_BASE,
            "endpoints": [
                {"name": "index composition (monthly, first trading day)",
                 "url": f"{moex.ISS_BASE}/statistics/engines/stock/markets/index/analytics/{cfg['index']}.json?date=YYYY-MM-DD&limit=100"},
                {"name": "all-time index tickers",
                 "url": f"{moex.ISS_BASE}/statistics/engines/stock/markets/index/analytics/{cfg['index']}/tickers.json"},
                {"name": "per-member analytics (share-count evidence)",
                 "url": f"{moex.ISS_BASE}/statistics/engines/stock/markets/index/analytics/{cfg['index']}/tickers/{{ticker}}.json?from=&till="},
                {"name": "board-group history (T+ shares, boardgroup 57)",
                 "url": f"{moex.ISS_BASE}/history/engines/stock/markets/shares/boardgroups/{cfg['boardgroup']}/securities/{{secid}}.json?from=&till="},
                {"name": "index levels (calendar, benchmarks)",
                 "url": f"{moex.ISS_BASE}/history/engines/stock/markets/index/securities/{{IMOEX,MCFTR,MCFTRR}}.json?from=&till="},
                {"name": "split register", "url": f"{moex.ISS_BASE}/statistics/engines/stock/splits.json"},
            ],
            "fetched_at": fetched_range(raw_dir),
            "dividend_endpoint_probes": probes,
        },
        "counts": {
            "calendar_days": len(cal), "tradable_days": int(cal["tradable"].sum()),
            "tradable_days_from_price_start": len(tradable),
            "weekend_sessions": int((cal["tradable"] & ~cal["is_weekday"]).sum()),
            "membership_snapshots": int(comp["date"].nunique()), "membership_rows": len(comp),
            "universe_tickers": len(uni), "tickers_with_prices": int(has_data.sum()),
            "tickers_without_price_history": sorted(uni.index[~uni.index.isin(close.columns)]),
            "ticker_differs_from_secid": comp.loc[comp["ticker"] != comp["secid"], ["ticker", "secid"]]
            .drop_duplicates().to_dict("records"),
            "ever_members_in_snapshots": int(uni["in_snapshots"].sum()),
            "ever_members_in_all_time_list_only": int((uni["in_all_time_list"] & ~uni["in_snapshots"]).sum()),
            "members_on_last_snapshot": int((comp["date"] == comp["date"].max()).sum()),
            "exited_tickers": int((uni["in_snapshots"] & ~uni.index.isin(comp[comp["date"] == comp["date"].max()]["secid"])).sum()),
            "rows_with_volume_on_non_tradable_days_dropped": len(off_calendar),
            "history_rows": len(hist), "history_rows_by_board": hist["BOARDID"].value_counts().to_dict(),
            "history_rows_on_dates_with_two_boards": int(hist.duplicated(["SECID", "TRADEDATE"], keep=False).sum()),
            "splits_in_register_for_universe": len(actions[actions["source"].str.startswith("iss statistics")]),
            "splits_applied": len(applied),
            "jumps_over_scan_threshold": len(jumps),
        },
        "calendar": {"closure_spells_min_10_weekdays": spells.assign(
            start=spells["start"].dt.date.astype(str), end=spells["end"].dt.date.astype(str)).to_dict("records")},
        "corporate_actions": actions.to_dict("records"),
        "price_jumps": jumps.to_dict("records"),
        "dividends": div_report,
        "diagnostics": diag,
        "investigations": {"AFKS_2014-12-18": afks, "legacy_7_vs_5": legacy_rec},
        "schema": SCHEMA,
        "assumptions": {
            "price": "CLOSE on board group 57 (TQBR/TQBS/TQNE), unadjusted; main market from 2013-09-02",
            "tradable": "IMOEX was calculated on the day",
            "ex_date": "T+2: trading day before the record date; T+1 (record dates after 2023-07-31): the record date",
            "membership_weight": "weight = fraction, weight_pct = as published (percent)",
            "ticker_changes": "secids are not stitched (YNDX/YDEX, FIVE/X5, TCSG/T, MAIL/VKCO, HHRU/HEAD are separate columns)",
        },
        "files": {"raw": moex.file_manifest(raw_dir), "processed": moex.file_manifest(out_dir)},
    }
    (out_dir / "provenance.json").write_text(json.dumps(jsonable(prov), ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("built %s: close %s, adj %s, actions %d, jumps %d", out_dir, close.shape, adj.shape, len(actions), len(jumps))


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", "src", "scripts"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch-core", "probe", "fetch-prices", "fetch-evidence", "build", "all"])
    ap.add_argument("--config", help="YAML file overriding DEFAULT_CONFIG keys")
    ap.add_argument("--dividends-csv", help="secid, record_date, amount, currency[, source]")
    ap.add_argument("--only", nargs="*", help="fetch-prices: restrict to these tickers (testing)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    log_dir = REPO / cfg["raw_dir"] / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / f"{args.stage}_{stamp}.log")],
    )
    if args.stage in ("fetch-core", "all"):
        fetch_core(cfg)
    if args.stage in ("probe", "fetch-core", "all"):
        probe_endpoints(cfg)
    if args.stage in ("fetch-prices", "all"):
        fetch_prices(cfg, args.only)
    if args.stage in ("fetch-evidence", "all"):
        fetch_evidence(cfg)
    if args.stage in ("build", "all"):
        build(cfg, args.dividends_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
