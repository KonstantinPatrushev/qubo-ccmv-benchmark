"""Build the point-in-time S&P 500 dataset (membership, prices, volume, provenance).

Subcommands:

``build``       Wikipedia snapshots -> membership on every date -> Yahoo prices -> parquet files
                and ``provenance.json`` in ``--out-dir``.
``diagnose``    Print the liquidity-ranked top-N on given formation dates from a built dataset
                (and why some names are, or are not, in it).
``check-jump``  Evidence on one large one-day move from Yahoo's price history and corporate
                actions (default: BLDR on 2015-04-13, flagged +68 % in the legacy data).

Examples::

    python scripts/data/build_sp500_pit_dataset.py build
    python scripts/data/build_sp500_pit_dataset.py build --skip-adj-check --verify-sample 0  # from the cache
    python scripts/data/build_sp500_pit_dataset.py build --no-adjustment-corrections --out-dir /tmp/yahoo
    python scripts/data/build_sp500_pit_dataset.py build --limit-tickers 30 \\
        --out-dir /tmp/sp500_pit_smoke --raw-dir /tmp/sp500_pit_smoke_raw
    python scripts/data/build_sp500_pit_dataset.py diagnose --dates 2015-01-02 2020-01-02
    python scripts/data/build_sp500_pit_dataset.py check-jump

Every parameter is explicit on the command line and written to ``provenance.json`` together with
the SHA-256 of the Wikipedia snapshots, of every file of the Yahoo cache (``files.raw``, relative to
``raw_dir`` as in moex_v2), of every output and of the code that produced the files
(``python -m cif.data.provenance verify data/processed/sp500_pit/provenance.json`` re-hashes them).
Network is used only for the two Wikipedia pages and for Yahoo Finance: for Yahoo, what the cache
does not hold and, unless ``--skip-adj-check --verify-sample 0`` are given, the two re-download
checks (a rebuild from the cache with these flags is offline and carries the earlier results of the
two checks over, marked with the date of the build that made them).

``prices_adj.parquet`` is Yahoo's adjusted close with the days of ``adjustment_gaps.parquet``
corrected to total returns (``cif.data.sp500_pit.correct_adjustment_artifacts``, one row per day in
``adjustment_corrections.parquet`` and in ``provenance.json``). ``--no-adjustment-corrections``
writes Yahoo's panel as it is; every other output is identical in both modes.

``adjcorr-2`` adds the spin-off days registered in ``cif.data.sp500_events`` (terms and sources as
read in the company's filings; ``cif.data.sp500_pit.apply_verified_events``): their rows join the
corrections table, their evidence goes to ``corporate_action_checks.json`` (the earlier checks stay)
and to ``provenance.json``, and ``pseudo_split_days.parquet`` marks them corrected.
``--no-verified-events`` leaves ``adjcorr-1`` (the gap rule alone). The companies they distribute
that are not index members (``SAIC``; Exelis, Care Capital Properties and Apartment Income REIT,
which have no Yahoo series and leave a ``<ticker>.failed.json`` note in the cache) are asked for
like the evidence tickers and cached with the other series, so a rebuild from the cache is offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cif.data import sp500_events as ev
from cif.data import sp500_pit as pit

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("build_sp500_pit")

DEFAULT_RAW = ROOT / "data" / "raw" / "sp500_pit"
DEFAULT_OUT = ROOT / "data" / "processed" / "sp500_pit"


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=False)
    return result.stdout.strip()


def git_info() -> dict:
    status = _git("status", "--porcelain")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "n_changed_paths": len(status.splitlines()),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _day(series: pd.Series) -> pd.Series:
    """ISO dates of a date column (an empty frame has object dtype, so convert first)."""
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def read_provenance(out_dir: Path) -> dict:
    """The provenance an earlier build left in ``out_dir`` (``{}`` if none), read before it is replaced."""
    path = Path(out_dir) / "provenance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def carried_check(previous: dict, key: str) -> dict | None:
    """A check of the Yahoo cache that needs the network and that this build skipped: the result of
    the earlier build of the same cache, marked with the date of that build."""
    value = (previous.get("yahoo") or {}).get(key)
    if not value:
        return None
    return {**value, "carried_over_from_build": value.get("carried_over_from_build")
            or previous.get("built_at")}


def repo_relative(path: str | Path) -> str:
    """A path as provenance records it: relative to the repository when it lies inside, so that
    ``python -m cif.data.provenance verify`` finds the files on another machine."""
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def raw_file_hashes(raw_dir: Path, cache_dir: Path) -> dict[str, dict]:
    """SHA-256 and size of every file of the Yahoo cache, keyed by the path relative to ``raw_dir``
    (``files.raw`` as in moex_v2): a series (``<ticker>.parquet``) or the note that Yahoo has none
    (``<ticker>.failed.json``)."""
    out = {}
    for path in sorted(Path(cache_dir).iterdir()):
        if path.is_file():
            rel = Path(os.path.relpath(path, raw_dir)).as_posix()
            out[rel] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return out


def correction_version(args: argparse.Namespace) -> str:
    """``adjcorr-2`` with the verified events (default), ``adjcorr-1`` for the gap rule alone."""
    return (
        pit.ADJUSTMENT_CORRECTION_VERSION if args.verified_events
        else pit.ADJUSTMENT_CORRECTION_VERSION_GAPS
    )


def verified_provenance(events: tuple, verified: pit.VerifiedCorrections) -> dict:
    """The ``verified_events`` part of the provenance: what was applied, what was left out and why.

    The terms, sources and prices of every event are in ``corporate_action_checks.json`` (its
    hash is in ``outputs``); this is the summary that sits next to the policy.
    """
    table = verified.table.set_index(["ticker", "date"])
    summary = []
    for record in verified.records:
        row = table.loc[(record["ticker"], pd.Timestamp(record["date"]))]
        summary.append(
            {
                "ticker": record["ticker"], "date": record["date"], "kind": row["kind"],
                "value_source": row["value_source"], "return_close": row["return_close"],
                "return_yahoo": row["return_yahoo"], "return_corrected": row["return_corrected"],
                "change": row["change"], "scale_before": row["scale_before"],
                "plausible": bool(row["plausible"]), "child_exact": bool(row["child_exact"]),
                "children": [
                    {k: c[k] for k in ("ticker", "shares_per_parent_share", "close", "source", "basis")}
                    for c in record["prices"]["children"]
                ],
                "n_sources": len(record["sources"]), "n_price_checks": len(record["price_checks"]),
                "verified_on": record["verified_on"],
            }
        )
    return {
        "applied": True,
        "note": (
            "The spin-off days registered in cif.data.sp500_events: the terms of the transaction "
            "as the company filed them (sources, record and distribution dates in "
            "corporate_action_checks.json) give the holder's total return from the as-traded "
            "closes of the parent and of the distributed companies; a Yahoo series or a documented "
            "price per company; every price check must hold or the build stops. Applied after the "
            "gap rule by the same scale_before mechanism."
        ),
        "module": "src/cif/data/sp500_events.py",
        "n_events": len(events),
        "n_applied": len(verified.records),
        "skipped": dict(verified.skipped),
        "events": summary,
    }


def write_verified_checks(out_dir: Path, records: list[dict]) -> Path:
    """Add the verified events to ``corporate_action_checks.json``; the checks already there stay."""
    path = Path(out_dir) / "corporate_action_checks.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for record in records:
        store[f"{record['ticker']}@{record['date']}"] = _jsonable(
            record | {
                "source": "SEC EDGAR filings and investor-relations pages of the companies (see "
                          "sources); as-traded prices from the Yahoo Finance cache of this build "
                          "and the documented prices (see prices.children)",
            }
        )
    path.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def adjustment_provenance(
    args: argparse.Namespace, corr: pit.AdjustmentCorrections | None, pseudo: pd.DataFrame,
    uncorrected: dict, verified: dict | None = None,
) -> dict:
    """The ``adjustment_corrections`` section: policy, counts, every corrected day with its evidence."""
    if corr is None:
        return {
            "applied": False, "version": None,
            "note": "prices_adj.parquet is Yahoo's adjusted close as downloaded (--no-adjustment-"
                    "corrections); adjustment_gaps.parquet lists the days that would be corrected",
        }
    table = corr.table
    model = pseudo[pseudo["value_model_applies"] & ~pseudo["corrected"]]
    return {
        "applied": True,
        "version": correction_version(args),
        "note": (
            "On the days of adjustment_gaps (the adjusted return exceeds the split-adjusted close "
            "return by more than the threshold) Yahoo's day return is replaced by a total return "
            "from the unadjusted close and the dividends and split ratios Yahoo registered for "
            "the day (cif.data.sp500_pit.correct_adjustment_artifacts); adjcorr-2 adds the "
            "verified spin-off days (verified_events). Every price before such a day is "
            "multiplied by scale_before, prices from the day on are Yahoo's; every other "
            "return is unchanged. adjustment_corrections.parquet lists the days; "
            "SP500PIT.prices_adj_yahoo() undoes them."
        ),
        "verified_events": verified if verified is not None else {"applied": False},
        "threshold": args.adjustment_gap,
        "plausible_day_return": pit.PLAUSIBLE_DAY_RETURN,
        "n_events": len(table),
        "by_kind": {k: int(v) for k, v in table["kind"].value_counts().items()},
        "n_child_confirmed": int(table.loc[table["kind"] != pit.VERIFIED_KIND, "child_exact"].sum()),
        "n_beyond_plausible_return": int((~table["plausible"]).sum()),
        "max_abs_change": float(table["change"].abs().max()) if len(table) else 0.0,
        "uncorrected_dividend_days": {
            "note": "dividend days below the threshold, left to Yahoo: |Yahoo return - (Close + D)/"
                    "Close(t-1) + 1|",
            **uncorrected,
        },
        "pseudo_split_days_not_corrected": {
            "note": "ratios that are no share-count ratios, on days that adjustment_gaps does not list "
                    "because no dividend is registered. If the ratio books a spin-off, Yahoo's Close "
                    "return there is the ratio times the total return (return_if_value_ratio, "
                    "unverified). Listed in pseudo_split_days.parquet; the days of the verified "
                    "events are corrected (marked so), the others are not changed",
            "n_days": len(pseudo),
            "n_with_ratio_above_one_not_corrected": len(model),
            "n_distortion_above_0.03": int((model["distortion"].abs() > 0.03).sum()),
            "largest": model.head(15).assign(date=lambda d: _day(d["date"]))
            .to_dict("records"),
        },
        "events": table.assign(
            date=_day(table["date"]),
            previous_date=_day(table["previous_date"]),
        ).to_dict("records"),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def setup_logging(log_path: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers,
        force=True,
    )


# --------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------


def survivorship_report(
    membership: pd.DataFrame, coverage: pd.DataFrame, failed: dict[str, str],
    companies: pd.DataFrame, downloaded: list[str],
) -> dict:
    """Quantify what the free price source cannot see (delisted index members)."""
    tickers = list(membership.columns)
    current = set(companies.loc[companies["in_current_list"], "ticker"])
    removed = [t for t in tickers if t not in current]
    gone = set(failed)

    idx = coverage.index
    member = membership.reindex(idx, fill_value=False)
    days_per_ticker = member.sum()
    total_days = int(days_per_ticker.sum())
    missing_days = int(days_per_ticker[[t for t in tickers if t not in set(downloaded)]].sum())

    # trading days x members that have a valid price, from the coverage frame: also counts the
    # days before a downloaded series starts (a reused symbol, a late listing)
    share_without_price = 1.0 - float(coverage["n_with_price"].sum() / coverage["n_members"].sum())
    by_year = {}
    for year, block in coverage.groupby(idx.year):
        members = float(block["n_members"].mean())
        priced = float(block["n_with_price"].mean())
        by_year[int(year)] = {
            "mean_members": members,
            "mean_with_price": priced,
            "mean_coverage": priced / members if members else None,
            "min_coverage": float(block["coverage"].min()),
        }
    reasons: dict[str, int] = {}
    for reason in failed.values():
        key = reason.split(";")[0].strip()[:70]
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "n_tickers_ever_member": len(tickers),
        "n_current_members_in_window": len([t for t in tickers if t in current]),
        "n_removed_members_in_window": len(removed),
        "n_downloaded": len(downloaded),
        "n_failed": len(gone),
        "failed_among_current": len([t for t in gone if t in current]),
        "failed_among_removed": len([t for t in gone if t not in current]),
        "download_rate_current": 1 - len([t for t in gone if t in current]) / max(1, len(current & set(tickers))),
        "download_rate_removed": 1 - len([t for t in gone if t not in current]) / max(1, len(removed)),
        "member_days_total": total_days,
        "member_days_without_downloaded_series": missing_days,
        "share_member_days_without_downloaded_series": missing_days / total_days if total_days else None,
        "share_member_days_without_price": share_without_price,
        "failed_reasons": reasons,
        "by_year": by_year,
        "note": (
            "Index weights are not available from free sources, so the share is by count "
            "(equal weight), not by market capitalisation."
        ),
    }


def verify_sample(panel: pit.YahooPanel, start: str, end_exclusive: str, n: int, seed: int) -> dict:
    """Download ``n`` random series again, one ticker per request, and compare with the batch."""
    tickers = sorted(panel.adj_close.columns)
    rng = np.random.default_rng(seed)
    sample = [str(t) for t in rng.choice(tickers, size=min(n, len(tickers)), replace=False)]
    out = {}
    for ticker in sample:
        raw, _ = pit.yahoo_downloader([ticker], start, end_exclusive)
        frame = pit.split_yahoo_frame(raw, [ticker]).get(ticker)
        if frame is None:
            out[ticker] = {"error": "no data on the second download"}
            continue
        again = pit.tidy_yahoo_frame(frame)[["Adj Close", "Close", "Volume"]]
        mine = pd.DataFrame(
            {"Adj Close": panel.adj_close[ticker], "Close": panel.close[ticker],
             "Volume": panel.volume[ticker]}
        ).dropna(how="all")
        out[ticker] = pit.compare_frames(mine, again)
        time.sleep(0.5)
    worst = max((v.get(c, 0.0) for v in out.values() for c in ("Adj Close", "Close", "Volume")),
                default=0.0)
    return {"seed": seed, "tickers": out, "max_rel_diff": worst,
            "n_row_mismatch": sum(v.get("rows_a") != v.get("rows_b") for v in out.values())}


def adj_close_check(panel: pit.YahooPanel, start: str, end_exclusive: str, n: int = 6) -> dict:
    """Is ``Adj Close`` of the raw download the same as ``auto_adjust=True``'s ``Close``?"""
    import yfinance as yf

    wanted = [t for t in ("AAPL", "MSFT", "JPM", "XOM", "BRK-B", "BLDR") if t in panel.adj_close]
    sample = wanted[:n] or list(panel.adj_close.columns[:n])
    if not sample:
        return {"checked": []}
    auto = yf.download(
        sample, start=start, end=end_exclusive, auto_adjust=True, progress=False, threads=True,
        group_by="ticker", timeout=30,
    )
    out = {}
    for t in sample:
        try:
            series = auto[t]["Close"] if isinstance(auto.columns, pd.MultiIndex) else auto["Close"]
        except KeyError:
            continue
        series.index = pd.to_datetime(series.index).tz_localize(None)
        both = pd.concat([series.rename("auto"), panel.adj_close[t].rename("adj")], axis=1).dropna()
        if both.empty:
            continue
        rel = ((both["auto"] - both["adj"]).abs() / both["adj"]).max()
        out[t] = {"n_days": len(both), "max_rel_diff": float(rel)}
    return {"checked": out, "max_rel_diff": max((v["max_rel_diff"] for v in out.values()), default=None)}


def cmd_build(args: argparse.Namespace) -> int:
    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = read_provenance(out_dir)
    setup_logging(out_dir / "build.log")
    logger.info("build start: %s", vars(args))
    t0 = datetime.now(timezone.utc)

    have_snapshots = any(raw_dir.glob("wikipedia_constituents_*.meta.json"))
    if args.refresh_wikipedia or not have_snapshots:
        snaps = pit.fetch_wikipedia_snapshots(raw_dir)
        logger.info("fetched Wikipedia pages")
    else:
        snaps = pit.load_wikipedia_snapshots(raw_dir)
        logger.info("using saved Wikipedia snapshots (SHA-256 verified)")
    cur = pit.parse_constituents(snaps.constituents.read_text())
    changes = pit.parse_changes(snaps.changes.read_text())
    as_of = pd.Timestamp(snaps.constituents.fetched_at[:10])
    logger.info("current constituents: %d, change rows: %d, as_of %s", len(cur), len(changes),
                as_of.date())

    rec = pit.reconstruct_membership(cur, changes, start=args.start, as_of=as_of)
    membership = pit.intervals_to_membership(rec.intervals, args.start, args.end)
    counts = pit.summarize_member_counts(membership, args.min_members, args.max_members)
    logger.info(
        "membership %s: members per day min %d / median %.0f / max %d, in [%d, %d] on %.1f%% of "
        "days", membership.shape, counts["min"], counts["median"], counts["max"], args.min_members,
        args.max_members, 100 * counts["share_in_band"],
    )
    for kind, n in rec.anomalies["kind"].value_counts().items():
        logger.info("anomaly %-24s %d", kind, n)

    tickers = list(membership.columns)
    companies = pit.build_companies(cur, changes, tickers)
    to_download = tickers[: args.limit_tickers] if args.limit_tickers else tickers
    end_exclusive = (pd.Timestamp(args.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_dir = Path(args.yahoo_cache) if args.yahoo_cache else raw_dir / (
        f"yahoo_{args.start}_{args.end}" + (f"_first{args.limit_tickers}" if args.limit_tickers else "")
    )
    logger.info("downloading %d tickers from Yahoo (cache %s)", len(to_download), cache_dir)
    panel = pit.download_yahoo_panel(
        to_download, args.start, end_exclusive, batch_size=args.batch_size, pause=args.pause,
        retries=args.retries, cache_dir=cache_dir,
    )
    window = slice(pd.Timestamp(args.start), pd.Timestamp(args.end))
    masked = pit.mask_implausible_series(
        panel.adj_close.loc[window], panel.close.loc[window], panel.volume.loc[window],
        membership, window=args.ghost_window, min_dollar_volume=args.min_member_dollar_volume,
    )
    prices_adj, close_unadj, volume = masked.adj_close, masked.close_unadj, masked.volume
    failed = {**panel.failed, **masked.dropped}
    splits = panel.splits[panel.splits["ticker"].isin(prices_adj.columns)]
    logger.info(
        "downloaded %d tickers, failed %d; %d series blanked on member days as not the index "
        "member (%d of them dropped)", panel.adj_close.shape[1], len(panel.failed),
        len(masked.report), len(masked.dropped),
    )
    # inventories of Yahoo's panel as downloaded; the correction below leaves them as they are
    moves = pit.large_move_inventory(prices_adj, volume, splits, threshold=args.large_move)
    gaps = pit.adjustment_gaps(prices_adj, close_unadj, threshold=args.adjustment_gap)
    dividends = (
        panel.dividends[panel.dividends["ticker"].isin(prices_adj.columns)]
        .sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    )
    prices_yahoo, evidence, corr, verified = prices_adj, None, None, None
    events = tuple(ev.VERIFIED_EVENTS) if args.verified_events else ()
    evidence_requested: list[str] = []
    if args.adjustment_corrections:
        # members of the panel are in it already; asking for them again would list their splits twice
        wanted = [*args.evidence_tickers, *(d.ticker for e in events for d in e.distributions)]
        evidence_requested = [t for t in dict.fromkeys(wanted) if t not in panel.adj_close.columns]
        evidence = pit.download_yahoo_panel(
            evidence_requested, args.start, end_exclusive, batch_size=args.batch_size,
            pause=args.pause, retries=args.retries, cache_dir=cache_dir,
        )
        children = pd.concat([panel.close.loc[window], evidence.close.loc[window]], axis=1)
        children = children.loc[:, ~children.columns.duplicated()]
        all_splits = [f for f in (panel.splits, evidence.splits) if len(f)]
        splits_all = pd.concat(all_splits, ignore_index=True) if all_splits else panel.splits
        corr = pit.correct_adjustment_artifacts(
            prices_yahoo, close_unadj, dividends, splits_all, volume=volume,
            children_close=children, threshold=args.adjustment_gap,
        )
        prices_adj = corr.prices_adj
        logger.info(
            "adjusted close: %d of %d gap days corrected (%s); evidence tickers: %s downloaded, %s "
            "without data at Yahoo", len(corr.table), len(gaps),
            corr.table["kind"].value_counts().to_dict(), sorted(evidence.adj_close.columns),
            sorted(evidence.failed),
        )
        if events:
            verified = pit.apply_verified_events(
                prices_adj, close_unadj, splits_all, events, children_close=children,
                volume=volume, corrected=set(zip(corr.table["ticker"], corr.table["date"], strict=True)),
            )
            if len(verified.table):
                table = pd.concat([t for t in (corr.table, verified.table) if len(t)], ignore_index=True)
                corr = pit.AdjustmentCorrections(
                    verified.prices_adj,
                    table.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True),
                )
                prices_adj = corr.prices_adj
            logger.info(
                "verified spin-off days: %d of %d events applied (%s); not in this panel: %s",
                len(verified.table), len(events),
                {r["ticker"] + "@" + r["date"]: round(r["returns"]["corrected"], 4)
                 for r in verified.records}, verified.skipped,
            )
    check = (
        adj_close_check(panel, args.start, end_exclusive) if not args.skip_adj_check
        else carried_check(previous, "adj_close_vs_auto_adjust")
    )
    if check:
        logger.info("Adj Close vs auto_adjust=True: %s", check.get("max_rel_diff"))
    rerun = (
        verify_sample(panel, args.start, end_exclusive, args.verify_sample, args.verify_seed)
        if args.verify_sample else carried_check(previous, "batch_vs_single_download")
    )
    if rerun:
        logger.info("batch vs single-ticker re-download of %d series: max rel diff %s, row mismatches %s",
                    len(rerun["tickers"]), rerun["max_rel_diff"], rerun["n_row_mismatch"])

    n_bad = {
        "non_positive_adj_close": int((prices_adj <= 0).sum().sum()),
        "non_positive_close": int((close_unadj <= 0).sum().sum()),
        "negative_volume": int((volume < 0).sum().sum()),
        "adj_close_nan_cells": int(prices_adj.isna().sum().sum()),
    }
    coverage = pit.compute_coverage(
        membership, prices_adj, lookback=args.lookback, min_history_frac=args.min_history_frac,
    )
    surv = survivorship_report(
        membership, coverage, failed, companies, list(prices_adj.columns)
    )
    surv["removals_by_reason"] = pit.removal_breakdown(
        changes, rec.renames, membership, prices_adj.columns, args.start, as_of
    )
    surv["series_vs_membership"] = pit.series_vs_membership(membership, prices_adj)

    no_split = moves["n_splits_nearby"] == 0
    flagged = set(zip(gaps["ticker"], gaps["date"], strict=True))
    verified_days = (
        set(zip(verified.table["ticker"], verified.table["date"], strict=True))
        if verified is not None else set()
    )
    pseudo = pit.pseudo_split_inventory(
        prices_yahoo, close_unadj, splits,
        corrected=flagged | verified_days if corr is not None else (),
    )
    uncorrected = pit.uncorrected_dividend_error(prices_yahoo, close_unadj, dividends, splits, flagged)
    extra = {
        "membership_intervals": rec.intervals,
        "changes": changes,
        "anomalies": rec.anomalies,
        "renames": rec.renames,
        "splits": splits,
        "dividends": dividends,
        "large_moves": moves,
        "adjustment_gaps": gaps,
        "pseudo_split_days": pseudo,
    }
    if corr is not None:
        extra["adjustment_corrections"] = corr.table
    written = pit.save_dataset(
        out_dir, membership=membership, prices_adj=prices_adj, close_unadj=close_unadj,
        volume=volume, coverage=coverage, companies=companies, failed=failed, extra=extra,
    )
    if verified is not None and verified.records:
        written["corporate_action_checks"] = write_verified_checks(out_dir, verified.records)
    import yfinance

    provenance = {
        "built_at": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "point-in-time S&P 500 universe for the paper (R6 / external point 5)",
        "git": git_info(),
        "code_sha256": {
            "src/cif/data/sp500_pit.py": file_sha256(ROOT / "src/cif/data/sp500_pit.py"),
            "src/cif/data/sp500_events.py": file_sha256(ROOT / "src/cif/data/sp500_events.py"),
            "scripts/data/build_sp500_pit_dataset.py": file_sha256(Path(__file__)),
        },
        "command": sys.argv,
        "parameters": {
            k: (repo_relative(v) if k in ("raw_dir", "out_dir", "yahoo_cache") and v else
                str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items() if k != "func"
        },
        "wikipedia": {
            "constituents": snaps.constituents.to_dict() | {"path": snaps.constituents.path.name},
            "changes": snaps.changes.to_dict() | {"path": snaps.changes.path.name},
            "as_of": str(as_of.date()),
            "n_current_constituents": len(cur),
            "n_change_rows": int(changes.attrs.get("n_rows_raw", len(changes))),
            "n_change_rows_parsed": len(changes),
            "n_change_rows_in_window": int(((changes["date"] >= args.start)
                                            & (changes["date"] <= as_of)).sum()),
            "n_change_rows_after_snapshot": int((changes["date"] > as_of).sum()),
            "earliest_change": str(changes["date"].min().date()),
            "latest_change": str(changes["date"].max().date()),
        },
        "reconstruction": {
            "anomalies": {k: int(v) for k, v in rec.anomalies["kind"].value_counts().items()},
            "renames_by_rule": {k: int(v) for k, v in rec.renames["rule"].value_counts().items()},
            "renames": rec.renames.assign(
                date=_day(rec.renames["date"])
            ).to_dict("records"),
            "unresolved": rec.anomalies[rec.anomalies["kind"].isin(
                ["added_not_active", "removed_while_active", "rename_overlap",
                 "rename_not_applied", "overlapping_intervals"])]
            .assign(date=lambda d: _day(d["date"])).to_dict("records"),
            "date_added_without_row": sorted(
                rec.anomalies.loc[rec.anomalies["kind"] == "date_added_without_row", "ticker"]
            ),
            "member_counts": {k: v for k, v in counts.items() if k != "out_of_band_dates"},
        },
        "yahoo": {
            "yfinance_version": yfinance.__version__,
            "start": args.start,
            "end_inclusive": args.end,
            "auto_adjust": False,
            "fields": list(pit.YAHOO_FIELDS),
            "requested": len(to_download),
            "downloaded_series": int(panel.adj_close.shape[1]),
            "downloaded": int(prices_adj.shape[1]),
            "failed": len(failed),
            "failed_at_yahoo": len(panel.failed),
            "adj_close_vs_auto_adjust": check,
            "batch_vs_single_download": rerun,
            "data_quality_counts": n_bad,
            "n_splits": len(splits),
            "n_dividends": len(dividends),
            "cache_dir": repo_relative(cache_dir),
            "evidence_tickers": None if evidence is None else {
                "note": "price evidence only: checks a registered amount against the price of the "
                        "company that may have been handed out; downloaded into the same cache",
                "requested": evidence_requested,
                "downloaded": sorted(evidence.adj_close.columns),
                "failed": sorted(evidence.failed),
            },
        },
        "large_moves": {
            "note": "inventory of one-day moves in Yahoo's adjusted close as downloaded (before "
                    "adjustment_corrections); no price was removed (the legacy |r| > 60% rule is "
                    "not applied to this dataset)",
            "threshold": args.large_move,
            "n_events": len(moves),
            "n_with_split_nearby": int((~no_split).sum()),
            "n_volume_spike_without_split": int((no_split & (moves["volume_ratio"] >= 3)).sum()),
            "n_without_split_or_spike": int((no_split & ~(moves["volume_ratio"] >= 3)).sum()),
            "events": moves.head(60).assign(
                date=lambda d: _day(d["date"]),
                previous_date=lambda d: _day(d["previous_date"]),
            ).to_dict("records"),
        },
        "adjustment_gaps": {
            "note": "inventory of Yahoo's adjusted close as downloaded: days on which its return "
                    "exceeds the split-adjusted close return by more than the threshold (special "
                    "dividends and spin-offs, e.g. DHR 2016-07-05). Corrected in prices_adj.parquet "
                    "and listed in adjustment_corrections.parquet unless --no-adjustment-corrections",
            "threshold": args.adjustment_gap,
            "n_events": len(gaps),
            "events": gaps.head(60).assign(date=lambda d: _day(d["date"]))
            .to_dict("records"),
        },
        "adjustment_corrections": adjustment_provenance(
            args, corr, pseudo, uncorrected,
            verified_provenance(events, verified) if verified is not None else None,
        ),
        "implausible_series": {
            "note": "Yahoo serves the current holder of a symbol with its whole history; on member "
                    "days where the trailing median dollar volume is below the threshold the "
                    "series cannot be the index member and is set to NaN (raw values stay in the "
                    "download cache)",
            "window_days": args.ghost_window,
            "min_dollar_volume": args.min_member_dollar_volume,
            "n_series": len(masked.report),
            "n_member_days_masked": int(masked.report["n_member_days_masked"].sum()),
            "dropped_columns": masked.dropped,
            "series": masked.report.assign(
                first_masked=lambda d: _day(d["first_masked"]),
                last_masked=lambda d: _day(d["last_masked"]),
            ).to_dict("records"),
        },
        "survivorship": surv,
        "selection_defaults": {
            "n": 100, "lookback": args.lookback, "liquidity_window": 252,
            "min_history_frac": args.min_history_frac,
        },
        "libraries": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    outputs = {
        path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name not in ("provenance.json", "build.log")
    }
    assert set(path.name for path in written.values()) <= set(outputs)
    provenance["outputs"] = outputs
    provenance["prices_adj_version"] = correction_version(args) if corr is not None else "yahoo"
    provenance["files"] = {"raw": raw_file_hashes(raw_dir, cache_dir)}
    (out_dir / "provenance.json").write_text(
        json.dumps(_jsonable(provenance), indent=2, ensure_ascii=False)
    )
    logger.info("wrote %s", out_dir / "provenance.json")
    print_build_summary(provenance)
    return 0


def print_build_summary(prov: dict) -> None:
    surv = prov["survivorship"]
    rec = prov["reconstruction"]
    print("\n=== SUMMARY ===")
    print(f"tickers ever members in window : {surv['n_tickers_ever_member']}")
    print(f"downloaded / failed            : {surv['n_downloaded']} / {surv['n_failed']}")
    print(f"  current members failed       : {surv['failed_among_current']} "
          f"(download rate {surv['download_rate_current']:.3f})")
    print(f"  removed members failed       : {surv['failed_among_removed']} of "
          f"{surv['n_removed_members_in_window']} (download rate {surv['download_rate_removed']:.3f})")
    print(f"member-days without a price    : {surv['share_member_days_without_price']:.4f} "
          f"(of them in tickers with no usable series at all: "
          f"{surv['share_member_days_without_downloaded_series']:.4f})")
    print(f"member counts                  : {rec['member_counts']['min']}..{rec['member_counts']['max']}"
          f" (median {rec['member_counts']['median']}), share in band "
          f"{rec['member_counts']['share_in_band']:.4f}")
    print("coverage by year (mean share of members with a price):")
    for year, row in surv["by_year"].items():
        print(f"  {year}: members {row['mean_members']:.1f}, with price {row['mean_with_price']:.1f},"
              f" coverage {row['mean_coverage']:.3f} (min {row['min_coverage']:.3f})")
    print("anomalies:", rec["anomalies"])
    print("renames  :", prov["reconstruction"]["renames_by_rule"])
    print("departures from the index by reason (n, share with a series):")
    for label, cell in surv["removals_by_reason"].items():
        print(f"  {label:<22} {cell['n_removals']:>4}  {cell['share_with_series']:.3f}")
    print(f"days where the adjusted-close return exceeds the close return by more than "
          f"{prov['adjustment_gaps']['threshold']:.0%}: {prov['adjustment_gaps']['n_events']}")
    adj = prov["adjustment_corrections"]
    if adj["applied"]:
        print(f"  corrected to total returns ({adj['version']}): {adj['n_events']} days by kind "
              f"{adj['by_kind']}; child confirmed: {adj['n_child_confirmed']}; corrected return still "
              f"beyond {adj['plausible_day_return']:.0%}: {adj['n_beyond_plausible_return']}; "
              f"value-ratio days not corrected: {adj['pseudo_split_days_not_corrected']['n_with_ratio_above_one_not_corrected']}"
              f" (distortion above 3 pp: {adj['pseudo_split_days_not_corrected']['n_distortion_above_0.03']})")
        verified = adj["verified_events"]
        if verified["applied"]:
            print(f"  verified spin-off days: {verified['n_applied']} of {verified['n_events']} "
                  f"applied; not in this panel: {sorted(verified['skipped'])}")
    else:
        print("  adjusted-close corrections: off (Yahoo's panel as downloaded)")
    ghost = prov["implausible_series"]
    print(f"series blanked as not the index member: {ghost['n_series']} tickers, "
          f"{ghost['n_member_days_masked']} member days; dropped columns: "
          f"{sorted(ghost['dropped_columns'])}")
    lm = prov["large_moves"]
    print(f"one-day moves beyond {lm['threshold']:.0%} in the adjusted close: {lm['n_events']} "
          f"(next to a Yahoo split: {lm['n_with_split_nearby']}, volume spike without split: "
          f"{lm['n_volume_spike_without_split']}, neither: {lm['n_without_split_or_spike']})")
    svm = surv["series_vs_membership"]
    print(f"series starting >{svm['min_gap_days']}d after membership or ending >{svm['min_gap_days']}d "
          f"before it: {svm['n_tickers']} tickers")
    rerun = prov["yahoo"].get("batch_vs_single_download")
    if rerun:
        print(f"batch vs single-ticker re-download: {len(rerun['tickers'])} series, max rel diff "
              f"{rerun['max_rel_diff']:.2e}, row-count mismatches {rerun['n_row_mismatch']}")


# --------------------------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------------------------


def explain_status(data: pit.SP500PIT, ticker: str, date: pd.Timestamp, table: pd.DataFrame,
                   n: int, lookback: int, min_frac: float) -> str:
    row = data.membership.loc[:date].iloc[-1]
    if ticker not in row.index or not bool(row[ticker]):
        return "not an index member on this date"
    if ticker not in data.prices_adj.columns:
        return "member, but no price series downloaded (" + data.failed.get(ticker, "unknown") + ")"
    window = data.prices_adj.loc[:date].iloc[-lookback:]
    frac = float((window[ticker] > 0).mean())
    if frac < min_frac:
        return f"member, valid price share {frac:.3f} < {min_frac}"
    if ticker not in table.index:
        company = data.companies.set_index("ticker")["company_id"]
        twin = table.index[table["company_id"] == company.get(ticker)]
        return f"member, share class dropped in favour of {list(twin)}"
    rank = int(table.loc[ticker, "rank"])
    mdv = table.loc[ticker, "median_dollar_volume"] / 1e6
    return (f"IN top-{n}" if rank <= n else f"NOT in top-{n}") + f": rank {rank}, median $ volume {mdv:,.0f}M"


def cmd_diagnose(args: argparse.Namespace) -> int:
    setup_logging(None)
    data = pit.load_dataset(Path(args.out_dir))
    names = data.companies.set_index("ticker")["security"]
    legacy = None
    legacy_path = Path(args.legacy_universe)
    if legacy_path.exists():
        legacy = list(json.loads(legacy_path.read_text())["tickers"])
    for raw_date in args.dates:
        date = pd.Timestamp(raw_date)
        table = data.candidates(date, lookback=args.lookback, liquidity_window=args.liquidity_window,
                                min_history_frac=args.min_history_frac)
        top = table.head(args.n)
        member_row = data.membership.loc[:date].iloc[-1]
        members = list(member_row.index[member_row.to_numpy()])
        with_series = [t for t in members if t in data.prices_adj.columns]
        window = data.prices_adj.loc[:date].iloc[-args.lookback:]
        history_ok = [t for t in with_series if (window[t] > 0).mean() >= args.min_history_frac - 1e-12]
        print(f"\n===== formation date {date.date()} =====")
        print(f"funnel: index members {len(members)} -> with a downloaded series {len(with_series)} "
              f"-> complete history {len(history_ok)} -> one ticker per company {len(table)} "
              f"-> top-{args.n}")
        invisible = sorted(set(members) - set(with_series))
        print(f"members with no series at all (invisible to the study): {len(invisible)}")
        print("  " + ", ".join(invisible[:60]) + (" ..." if len(invisible) > 60 else ""))
        print(f"top-{args.n} (rank, ticker, security, median $ volume in millions):")
        listing = [f"{int(r['rank']):>3} {t:<6} {str(names.get(t, ''))[:22]:<22} "
                   f"{r['median_dollar_volume'] / 1e6:7.0f}" for t, r in top.iterrows()]
        for k in range(0, len(listing), 3):
            print("  " + " | ".join(listing[k:k + 3]))
        if legacy is not None:
            print(f"overlap with the legacy alphabetical 'top-100': {len(set(top.index) & set(legacy))} "
                  f"of {len(legacy)}")
        print("checks:")
        for ticker in args.check:
            print(f"  {ticker:<5}: {explain_status(data, ticker, date, table, args.n, args.lookback, args.min_history_frac)}")
        sectors = data.companies.set_index("ticker")["sector"].reindex(top.index).fillna("n/a")
        print("sector mix of the top list:", sectors.value_counts().to_dict())
    return 0


# --------------------------------------------------------------------------------------------
# check-jump
# --------------------------------------------------------------------------------------------


def fetch_history(ticker_obj, start: str, end: str, attempts: int = 4) -> pd.DataFrame:
    """``Ticker.history`` with actions; Yahoo now and then answers an empty frame to a valid ticker."""
    for attempt in range(1, attempts + 1):
        frame = ticker_obj.history(start=start, end=end, auto_adjust=False, actions=True)
        if len(frame):
            return frame
        logger.warning("empty history (attempt %d/%d), asking again", attempt, attempts)
        time.sleep(3.0 * attempt)
    raise RuntimeError(f"Yahoo returned no history for {start}..{end} after {attempts} attempts")


def splits_list(splits) -> list[tuple[str, float]]:
    """``Ticker.splits`` as ``[(date, ratio)]``; it is a Series or, without splits, a frame."""
    series = splits["Stock Splits"] if isinstance(splits, pd.DataFrame) else splits
    out = []
    for day, ratio in series[series != 0].items():
        stamp = pd.Timestamp(day)
        out.append((str((stamp.tz_localize(None) if stamp.tzinfo else stamp).date()), float(ratio)))
    return out


def cmd_check_jump(args: argparse.Namespace) -> int:
    import yfinance as yf

    setup_logging(None)
    date = pd.Timestamp(args.date)
    tkr = yf.Ticker(args.ticker)
    hist = fetch_history(
        tkr,
        (date - pd.Timedelta(days=args.pad_before)).strftime("%Y-%m-%d"),
        (date + pd.Timedelta(days=args.pad_after)).strftime("%Y-%m-%d"),
    )
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    evidence = pit.jump_evidence(hist, date)
    evidence["all_splits_in_yahoo_history"] = splits_list(tkr.splits)
    window = hist.loc[date - pd.Timedelta(days=7): date + pd.Timedelta(days=7),
                      ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]]
    evidence["window"] = [
        {"date": str(d.date()), **{k: float(v) for k, v in row.items()}} for d, row in window.iterrows()
    ]
    legacy = {}
    raw_path, clean_path = Path(args.legacy_raw), Path(args.legacy_clean)
    if raw_path.exists() and clean_path.exists():
        raw, clean = pd.read_parquet(raw_path), pd.read_parquet(clean_path)
        if args.ticker in raw.columns and args.ticker in clean.columns:
            r, c = raw[args.ticker], clean[args.ticker]
            prev = r.loc[:date - pd.Timedelta(days=1)].iloc[-1]
            legacy = {
                "raw_prev_close": float(prev), "raw_event_close": float(r.loc[date]),
                "raw_return": float(r.loc[date] / prev - 1),
                "cleaned_prev": float(c.loc[:date - pd.Timedelta(days=1)].iloc[-1]),
                "cleaned_event": float(c.loc[date]),
                "cleaned_return": float(c.loc[date] / c.loc[:date - pd.Timedelta(days=1)].iloc[-1] - 1),
                "raw_last_over_first": float(r.dropna().iloc[-1] / r.dropna().iloc[0]),
                "cleaned_last": float(c.dropna().iloc[-1]),
                "raw_last": float(r.dropna().iloc[-1]),
                "level_removed_by_cleaning": float(r.dropna().iloc[-1] / c.dropna().iloc[-1]),
            }
    evidence["legacy_data"] = legacy
    out = Path(args.out_dir) / "corporate_action_checks.json"
    store = json.loads(out.read_text()) if out.exists() else {}
    store[f"{args.ticker}@{args.date}"] = _jsonable(evidence) | {
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Yahoo Finance via yfinance " + yf.__version__,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(store, indent=2, ensure_ascii=False))
    print(json.dumps(_jsonable({k: v for k, v in evidence.items() if k != "window"}), indent=2))
    print("window (unadjusted OHLC, volume, actions):")
    print(window.to_string())
    print("wrote", out)
    return 0


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the dataset")
    b.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    b.add_argument("--out-dir", default=str(DEFAULT_OUT))
    b.add_argument("--start", default="2010-01-01")
    b.add_argument("--end", default="2025-12-31", help="inclusive")
    b.add_argument("--refresh-wikipedia", action="store_true",
                   help="fetch the pages again instead of using the latest saved snapshot")
    b.add_argument("--batch-size", type=int, default=25)
    b.add_argument("--pause", type=float, default=1.5)
    b.add_argument("--retries", type=int, default=2)
    b.add_argument("--yahoo-cache", default=None)
    b.add_argument("--limit-tickers", type=int, default=0, help="smoke runs: first N tickers only")
    b.add_argument("--lookback", type=int, default=756)
    b.add_argument("--min-history-frac", type=float, default=0.95)
    b.add_argument("--large-move", type=float, default=0.6,
                   help="threshold of the one-day move inventory (nothing is corrected)")
    b.add_argument("--adjustment-gap", type=float, default=0.10,
                   help="threshold of the adjusted-vs-close return inventory; the days above it are "
                        "corrected to total returns")
    b.add_argument("--adjustment-corrections", action=argparse.BooleanOptionalAction, default=True,
                   help="correct those days in prices_adj (default) or write Yahoo's panel as is")
    b.add_argument("--verified-events", action=argparse.BooleanOptionalAction, default=True,
                   help="apply the spin-off days of cif.data.sp500_events after the gap rule "
                        "(adjcorr-2, default) or stop at the gap rule (adjcorr-1)")
    b.add_argument("--evidence-tickers", nargs="*", default=["CNDT", "CSRA", "KRFT"],
                   help="companies whose shares a flagged company may have handed out (Conduent, "
                        "CSRA, Kraft Foods Group); downloaded into the cache and used only to "
                        "check a registered amount against their price, never as a claim. The "
                        "companies that the verified events distribute are added to them")
    b.add_argument("--ghost-window", type=int, default=21,
                   help="trailing window of the plausibility mask (days)")
    b.add_argument("--min-member-dollar-volume", type=float, default=1e6,
                   help="median daily dollar volume below which a series is not an index member")
    b.add_argument("--min-members", type=int, default=495)
    b.add_argument("--max-members", type=int, default=510)
    b.add_argument("--skip-adj-check", action="store_true")
    b.add_argument("--verify-sample", type=int, default=12,
                   help="re-download this many random series one by one and compare (0 = off)")
    b.add_argument("--verify-seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    d = sub.add_parser("diagnose", help="print the top-N on formation dates")
    d.add_argument("--out-dir", default=str(DEFAULT_OUT))
    d.add_argument("--dates", nargs="+", default=["2015-01-02", "2020-01-02", "2025-01-02"])
    d.add_argument("--n", type=int, default=100)
    d.add_argument("--check", nargs="+", default=["MSFT", "AAPL", "NVDA", "JPM"])
    d.add_argument("--lookback", type=int, default=756)
    d.add_argument("--liquidity-window", type=int, default=252)
    d.add_argument("--min-history-frac", type=float, default=0.95)
    d.add_argument("--legacy-universe", default=str(ROOT / "data/raw/universe_sp500.json"))
    d.set_defaults(func=cmd_diagnose)

    j = sub.add_parser("check-jump", help="corporate action or market move?")
    j.add_argument("--ticker", default="BLDR")
    j.add_argument("--date", default="2015-04-13")
    j.add_argument("--pad-before", type=int, default=150)
    j.add_argument("--pad-after", type=int, default=40)
    j.add_argument("--out-dir", default=str(DEFAULT_OUT))
    j.add_argument("--legacy-raw", default=str(ROOT / "data/raw/sp500_prices_raw.parquet"))
    j.add_argument("--legacy-clean", default=str(ROOT / "data/processed/sp500_prices.parquet"))
    j.set_defaults(func=cmd_check_jump)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
