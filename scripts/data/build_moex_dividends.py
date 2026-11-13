#!/usr/bin/env python
"""Build the per-share dividend dataset of the MOEX panel (R6, RQ3 exploratory).

Inputs (both verified against their recorded SHA-256 before anything is read)::

    data/raw/cbonds_dividends/      Cbonds exports, imported by scripts/data/import_cbonds_dividends.py
    data/processed/moex_v2/         prices, calendar, splits, index levels and composition
    data/processed/cbr_fx/          official USD/RUB rate of the Bank of Russia (scripts/data/build_cbr_fx.py)

plus the registry of official disclosures (:mod:`cif.data.moex_dividend_registry`). Output,
``data/processed/moex_dividends/`` (everything but ``provenance.json`` is outside git)::

    events.parquet                one row per dividend payment kept, with the checks and flags
    dividends_rub.parquet         the usable RUB events in the format of ``cif.data.moex``
    tr_index_rub.parquet          total-return index of every share from the RUB events (foreign ones converted)
    cbonds_audit.parquet          what became of every Cbonds row
    registry_audit.parquet        the same for the registry rows
    registry_leads.csv            payments known but not registered (record date unread, left out as small) and why
    corrections_audit.csv         vendor rows whose amount a documented correction replaced
    non_rub_events.csv            events declared in a foreign currency: original amount, rate, rate date, checks
    price_outliers.csv            events whose ex-day drop disagrees with the dividend, with index status
    ex_date_disagreements.csv     events where the vendor's ex-date differs from the rule's
    restatement_verdict.csv       per share: restated or raw amount, by the ex-day drop
    scale_check.csv               per share: median drop / yield (catches splits no register lists)
    coverage_rub.csv              RUB events per share and year
    ticker_status.csv             what is known about each of the 101 tickers (unresolved_small: mean index weight < 1 %)
    index_flow_*.csv|parquet      the dividend flow of the index rebuilt from the events, against MCFTR - IMOEX
    index_missing_leads.csv       for unexplained residuals, the members whose ex-day drop fits best
    provenance.json               inputs, code and output hashes, counts, diagnostics (in git)

Run from a clean working tree (the provenance names the commit)::

    python scripts/data/build_moex_dividends.py [--config file.yaml] [--allow-dirty]

Foreign-currency events (USD: QIWI, RUSAL, Polymetal, Polyus, HeadHunter, the GDRs of the registry) are
converted at the official rate of the last date on or before the ex-date; the original amount, currency, rate
and rate date stay in the event. The total-return table leaves out the events that could not be converted and
the shares whose payments are not established (``ticker_status.csv``): it is a partial total return, named so.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cif.data import cbonds, moex  # noqa: E402
from cif.data import moex_dividend_checks as ck  # noqa: E402
from cif.data import moex_dividend_registry as registry  # noqa: E402
from cif.data import moex_dividends as md  # noqa: E402
from cif.data.provenance import recorded_files, sha256_file, verify_provenance  # noqa: E402

logger = logging.getLogger("build_moex_dividends")

DEFAULT_CONFIG: dict = {
    "raw_dir": "data/raw/cbonds_dividends",
    "calendar_file": "cbonds_dividend_calendar_moex_ru_20261008.xlsx",
    "calendar_all_file": "cbonds_dividend_calendar_moex_all_20261008.xlsx",
    "moex_dir": "data/processed/moex_v2",
    "out_dir": "data/processed/moex_dividends",
    "min_yield_sigma": md.MIN_YIELD_SIGMA,
    "margin_sigma": md.MARGIN_SIGMA,
    # exchange rates per currency code: processed series of cif.data.cbr_fx (RUB per unit, indexed by the date
    # the Bank of Russia set the rate for); each dataset's provenance is verified and recorded. {} = no conversion
    "fx_files": {"USD": "data/processed/cbr_fx/usd_rub.parquet"},
}

MOEX_FILES = ("prices_close", "prices_adj", "calendar", "splits", "universe", "imoex_membership",
              "index_levels", "index_dividend_flow")

CODE_FILES = (
    "src/cif/data/xlsx.py", "src/cif/data/cbonds.py", "src/cif/data/moex.py", "src/cif/data/moex_dividends.py",
    "src/cif/data/moex_dividend_checks.py", "src/cif/data/moex_dividend_registry.py",
    "scripts/data/import_cbonds_dividends.py", "scripts/data/build_moex_dividends.py",
)

#: table name -> file format ("parquet" with the index, or "csv" with or without it)
FORMATS = {
    "events": "parquet", "dividends_rub": "parquet", "cbonds_audit": "parquet", "registry_audit": "parquet",
    "index_flow_daily": "parquet", "tr_index_rub": "parquet-index",
    "non_rub_events": "csv", "registry_leads": "csv", "corrections_audit": "csv", "price_outliers": "csv", "ex_date_disagreements": "csv", "ticker_status": "csv",
    "index_residual_episodes": "csv", "index_missing_leads": "csv", "scale_check": "csv-index",
    "restatement_verdict": "csv-index", "coverage_rub": "csv-index", "index_flow_by_year": "csv-index",
    "index_flow_metrics": "csv-index",
}

SCHEMA = {
    "events": "one row per payment kept: secid (series of the panel on the ex-date), ex_date, record_date, amount "
              "(gross, per share in the basis of the ex-date, as declared), currency, source (cbonds|registry), "
              "restatement_factor (Cbonds amount x factor = amount), ex_rule / ex_vendor / ex_basis (how the ex-date "
              "was chosen), price check (prev_date, ex_return, abn_return, dividend_yield, sigma, z, capture, "
              "drop_status), index_status / index_contribution, in_index_on_ex, flags, usable_rub",
    "dividends_rub": "usable RUB events: secid, record_date, ex_date, amount, currency (+ amount_orig, currency_orig, "
                     "fx_rate, fx_date for converted foreign-currency events; source, ex_basis, checks): the input of "
                     "cif.data.moex.total_return_index",
    "tr_index_rub": "total-return index (base 100 at the first quote) of every share, gross RUB dividends reinvested "
                    "on the ex-date (foreign-currency ones converted at the official rate of the ex-date); EVENTS "
                    "WITHOUT A RATE AND PAYMENTS NOT ESTABLISHED ARE ABSENT, see provenance.tr_index and "
                    "ticker_status.csv",
    "non_rub_events": "events declared in a foreign currency: amount and currency AS DECLARED, fx_rate / fx_date / "
                      "amount_rub (the rouble amount in events.parquet) and fx_status (converted|no_rate|no_table); "
                      "implied_fx_price / implied_fx_flow: RUB per unit of the currency that the ex-day drop and the "
                      "residual of the index flow without these events imply, ratio_*: that over the rate used (a "
                      "check of event, date, amount and rate, not a rate)",
    "ticker_status": "per universe ticker: status (cbonds|registry|no_payment_evidenced|no_payment_partial|"
                     "outside_window|unresolved|unresolved_small), counts of events by source and currency, years of "
                     "IMOEX membership without a RUB event, mean index weight, the leads known (n_leads, lead_summary); "
                     "unresolved_small: no event and no evidence, but a mean index weight under 1 %, so that the missing "
                     "flow is about 1 % of the index dividend flow of the window at most",
    "registry_leads": "payments (or spans) known but not in the registry, with the pages read and why they are open",
    "index_flow_daily": "observed (MCFTR - IMOEX return), rebuilt from the events, residual, by tradable day",
    "index_residual_episodes": "days on which the rebuilt flow misses the observed one by >= 2 bp: shift, "
                               "shift_cluster, missing or extra",
    "index_missing_leads": "for each 'missing' residual, the index members (no event within 3 sessions) ranked by "
                           "the fit of their ex-day drop to the implied yield; leads, not findings",
}


class BuildRefused(RuntimeError):
    """A check failed before anything was written."""


@dataclass
class Assembly:
    built: md.Built
    tables: dict[str, pd.DataFrame]
    summary: dict


# --------------------------------------------------------------------------------------------
# Checks on the inputs
# --------------------------------------------------------------------------------------------


def check_all_countries_file(ru: pd.DataFrame, allc: pd.DataFrame, secids) -> dict:
    """The file without the country filter must be the Russian one plus rows of other issuers.

    Returns the description of the extra rows; raises :class:`BuildRefused` if a Russian row is
    missing from the second file or an extra row belongs to a ticker of the universe.
    """
    key = ["tickers", "amount", "currency", "record_date", "ex_date_cbonds", "period", "period_type"]
    ru_k = ru[key].astype(str).agg("|".join, axis=1)
    all_k = allc[key].astype(str).agg("|".join, axis=1)
    counts = all_k.value_counts()
    need = ru_k.value_counts()
    missing = [k for k, n in need.items() if counts.get(k, 0) < n]
    if missing:
        raise BuildRefused(f"{len(missing)} rows of the Russian export are not in the all-countries export")
    extra_idx = []
    remaining = need.to_dict()
    for i, k in all_k.items():
        if remaining.get(k, 0) > 0:
            remaining[k] -= 1
        else:
            extra_idx.append(i)
    extra = allc.loc[extra_idx]
    secids = set(secids)
    in_universe = [t for t in extra["tickers"] if md.candidate_secids(cbonds.split_tickers(t), secids)]
    if in_universe:
        raise BuildRefused(f"{len(in_universe)} extra rows of the all-countries export are in the universe: "
                           f"{sorted(set(in_universe))}")
    return {"n_extra_rows": len(extra),
            "extra_currencies": {str(k): int(v) for k, v in extra["currency"].value_counts().items()},
            "extra_tickers": sorted(set(extra["tickers"])), "extra_rows_in_universe": 0}


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def _scope(text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = text.split("..")
    return pd.Timestamp(a), pd.Timestamp(b)


#: A ticker with no event and no evidence whose mean index weight is below this is ``unresolved_small``: paying at
#: the index's own yield it would carry that share of the index dividend flow of its window.
SMALL_WEIGHT = 0.01


def ticker_status(
    panel: md.Panel, events: pd.DataFrame, no_payment, leads=(), small_weight: float = SMALL_WEIGHT
) -> pd.DataFrame:
    """What is known about every ticker of the universe (see the module docstring).

    ``mean_weight`` is the average weight of the ticker over the monthly composition snapshots in
    which it appears: how much of the index an unresolved ticker stands for. A ticker that is
    ``unresolved`` with a mean weight below ``small_weight`` is ``unresolved_small``; its ``leads``
    (what is known about its payments) are summarised in ``lead_summary``.
    """
    mean_weight = panel.membership.groupby("secid")["weight"].mean()
    rows = []
    for u in panel.universe.itertuples():
        sid = u.secid
        window = None
        if pd.notna(u.index_from) and pd.notna(u.index_till):
            a, b = max(panel.start, pd.Timestamp(u.index_from)), min(panel.end, pd.Timestamp(u.index_till))
            window = (a, b) if a <= b else None
        mine = events[events["secid"] == sid]
        rub = mine["currency"] == "RUB"
        counts = {f"n_{src}_{kind}": int(((mine["source"] == src) & (rub if kind == "rub" else ~rub)).sum())
                  for src in ("cbonds", "registry") for kind in ("rub", "non_rub")}
        covering, partial = False, False
        for n in no_payment:
            if n.secid != sid or window is None:
                continue
            s, e = _scope(n.scope)
            if s <= window[0] and e >= window[1]:
                covering = True
            elif s <= window[1] and e >= window[0]:
                partial = True
        n_cb = counts["n_cbonds_rub"] + counts["n_cbonds_non_rub"]
        n_rg = counts["n_registry_rub"] + counts["n_registry_non_rub"]
        if window is None:
            status = "outside_window"
        elif n_cb:
            status = "cbonds"
        elif n_rg:
            status = "registry"
        elif covering:
            status = "no_payment_evidenced"
        elif partial:
            status = "no_payment_partial"
        elif float(mean_weight.get(sid, 0.0)) < small_weight:
            status = "unresolved_small"
        else:
            status = "unresolved"
        mine_leads = [x.what for x in leads if x.secid == sid]
        years = ""
        if window is not None:
            have = set(mine.loc[rub, "ex_date"].dt.year)
            years = ",".join(str(y) for y in range(window[0].year, window[1].year + 1) if y not in have)
        rows.append({"secid": sid, "status": status, **counts,
                     "first_ex": mine["ex_date"].min() if len(mine) else pd.NaT,
                     "last_ex": mine["ex_date"].max() if len(mine) else pd.NaT,
                     "index_from": u.index_from, "index_till": u.index_till,
                     "mean_weight": float(mean_weight.get(sid, 0.0)), "years_without_events": years,
                     "n_leads": len(mine_leads), "lead_summary": " | ".join(mine_leads)})
    return pd.DataFrame(rows)


def total_return_table(panel: md.Panel, events: pd.DataFrame) -> pd.DataFrame:
    """Total-return index of every share from the usable RUB events (and only those)."""
    use = events[events["usable_rub"]]
    applied = panel.applied_splits
    cols = {}
    for sid in panel.close.columns:
        div = use.loc[use["secid"] == sid, ["ex_date", "amount"]]
        cols[sid] = moex.total_return_index(panel.close[sid], div, applied[applied["secid"] == sid])
    return pd.DataFrame(cols, index=panel.close.index).rename_axis("date")


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def _flow_summary(iv: ck.IndexValidation) -> dict:
    return {
        "metrics": {str(k): {m: (None if pd.isna(v) else float(v)) for m, v in row.items()}
                    for k, row in iv.metrics.iterrows()},
        "by_year": {str(k): {m: (None if pd.isna(v) else float(v)) for m, v in row.items()}
                    for k, row in iv.by_year.iterrows()},
        "episodes": _counts(iv.episodes["kind"]),
        "event_status": _counts(iv.contrib["index_status"]),
    }


def load_fx(files: Mapping[str, Path | str] | None) -> pd.DataFrame | None:
    """The table of exchange rates: one column of RUB per unit for each currency code.

    Each file is a processed series of :mod:`cif.data.cbr_fx` (``rate``, RUB per one unit, indexed by
    the date the rate is set for). ``None`` for no files. Repeated dates and non-positive rates are
    refused.
    """
    if not files:
        return None
    columns = {}
    for code, path in files.items():
        rate = pd.read_parquet(path)["rate"].astype(float)
        if rate.index.has_duplicates:
            raise BuildRefused(f"{path}: repeated dates in the exchange-rate series")
        if not (rate > 0).all():
            raise BuildRefused(f"{path}: exchange rates must be positive")
        rate.index = pd.DatetimeIndex(rate.index).normalize()
        columns[str(code).upper()] = rate
    return pd.DataFrame(columns).sort_index().sort_index(axis=1)


def fx_inputs(config: dict, *, repo: Path = REPO) -> tuple[pd.DataFrame | None, dict]:
    """The exchange-rate table of the configuration and the record of its inputs for the provenance.

    The ``provenance.json`` beside each rate file must exist, verify against the files it records
    and record the rate file itself; otherwise :class:`BuildRefused`. The record holds, per
    currency, the path and SHA-256 of the file and the path, SHA-256, dataset name, build time and
    code state of that provenance.
    """
    files = config.get("fx_files") or {}
    if not files:
        return None, {}
    record, paths = {}, {}
    for code, rel in files.items():
        path = repo / rel
        prov_path = path.parent / "provenance.json"
        if not prov_path.exists():
            raise BuildRefused(f"{rel}: no provenance.json beside the exchange-rate file ({prov_path})")
        report = verify_provenance(prov_path, repo_root=repo)
        if not report["ok"]:
            raise BuildRefused(f"{rel}: the provenance of the exchange-rate dataset does not verify: "
                               f"{report['mismatch']} {report['missing']}")
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        covered = {Path(f).resolve() for f, _ in recorded_files(prov, prov_path.parent, repo)}
        if path.resolve() not in covered:
            raise BuildRefused(f"{rel}: the file is not covered by {prov_path}")
        record[str(code).upper()] = {
            "path": str(rel), "sha256": sha256_file(path),
            "provenance": {"path": prov_path.relative_to(repo).as_posix(), "sha256": sha256_file(prov_path),
                           "dataset": prov.get("dataset"), "built_at": prov.get("built_at"), "git": prov.get("git")},
        }
        paths[code] = path
    return load_fx(paths), record


def _rate_check(ratio: pd.Series) -> dict:
    """Summary of ``implied / used`` ratios of the exchange rate (1 means the data agree with the rate)."""
    r = ratio.dropna()
    if not len(r):
        return {"n": 0, "median_ratio": None, "p10": None, "p90": None, "share_within_10pct": None}
    return {"n": len(r), "median_ratio": float(r.median()), "p10": float(r.quantile(0.1)),
            "p90": float(r.quantile(0.9)), "share_within_10pct": float(((r - 1.0).abs() <= 0.10).mean())}


RATE_CONVENTION = ("official Bank of Russia rate set for the last date on or before the ex-date (the rule's ex-date, "
                   "the vendor's if the rule has none), at most 7 days old; a rate set after the ex-date is never "
                   "used; amount_orig, currency_orig, fx_rate and fx_date keep the original")


def assemble(
    rows: pd.DataFrame, rows_all: pd.DataFrame, panel: md.Panel, *, registry_events, no_payment, config: dict,
    corrections=(), fx: pd.DataFrame | None = None, registry_leads=(),
) -> Assembly:
    """Run the pipeline and gather every table and the numbers for the provenance."""
    secids = list(panel.close.columns)
    all_check = check_all_countries_file(rows, rows_all, secids)
    reg = registry.registry_frame(registry_events)
    corr = registry.corrections_frame(corrections)
    built = md.build_events(rows, panel, registry=reg, corrections=corr, fx=fx,
                            min_yield_sigma=config["min_yield_sigma"], margin_sigma=config["margin_sigma"])
    ev = built.events
    days, mkt = panel.close.index, panel.index_levels["IMOEX"]

    converted = ev["currency_orig"].notna()
    iv = ck.index_validation(ev, panel.close, panel.adj, panel.membership, panel.index_flow, days)
    # the flow without the converted events: what the index paid in foreign currency is its residual, so the
    # rate it implies can be compared with the rate used
    iv_native = iv if not converted.any() else ck.index_validation(
        ev[~converted], panel.close, panel.adj, panel.membership, panel.index_flow, days)
    iv_cb = ck.index_validation(ev[ev["source"] == "cbonds"], panel.close, panel.adj, panel.membership,
                                panel.index_flow, days)
    contrib = iv.contrib[["secid", "ex_date", "amount", "contribution", "index_status"]].drop_duplicates(
        ["secid", "ex_date", "amount"]).rename(columns={"contribution": "index_contribution"})
    ev = ev.merge(contrib, on=["secid", "ex_date", "amount"], how="left")
    ev["index_status"] = ev["index_status"].fillna("not_in_flow")

    verdict = ck.restatement_verdict(ev, panel.adj, mkt, built.model, splits=panel.splits, last_date=panel.end)
    scale = ck.scale_check(ev)
    placebo = ck.placebo_exceedance(panel.adj, mkt, built.model, exclude=ev[["secid", "ex_date"]])
    leads = ck.missing_event_leads(iv.episodes, iv.weights, ev, panel.adj, mkt, built.model, days)
    status = ticker_status(panel, ev, no_payment, registry_leads)
    tr = total_return_table(panel, ev)

    obs = panel.index_flow.reindex(iv.flow.index)
    obs = obs[obs.index >= pd.Timestamp("2013-09-03")]
    daily = pd.DataFrame({"observed": obs, "rebuilt": iv.flow.reindex(obs.index)})
    daily["residual"] = daily["observed"] - daily["rebuilt"]
    daily = daily.reset_index()

    outliers = ev[ev["drop_status"].isin(["outlier", "borderline"])]
    disagreements = ev[ev["ex_vendor"].notna() & (ev["ex_vendor"] != ev["ex_rule"])]
    # events declared in a foreign currency, in the currency and amount as declared
    non_rub = ev[converted | (ev["currency"] != "RUB")].copy()
    conv = non_rub["currency_orig"].notna()
    non_rub["amount_rub"] = non_rub["amount"].where(conv)
    non_rub["amount_adj"] = (non_rub["amount_adj"] / non_rub["fx_rate"]).where(conv, non_rub["amount_adj"])
    non_rub["amount"] = non_rub["amount_orig"].where(conv, non_rub["amount"])
    non_rub["currency"] = non_rub["currency_orig"].where(conv, non_rub["currency"])
    non_rub["fx_status"] = np.where(conv, "converted", "no_rate" if fx is not None else "no_table")
    non_rub["implied_fx_price"] = ck.implied_fx_from_price(non_rub)
    non_rub["implied_fx_flow"] = ck.implied_fx_from_flow(non_rub, iv_native.weights, panel.close, panel.index_flow,
                                                         iv_native.flow, days)
    non_rub["ratio_price"] = non_rub["implied_fx_price"] / non_rub["fx_rate"]
    non_rub["ratio_flow"] = non_rub["implied_fx_flow"] / non_rub["fx_rate"]
    prev_weight = iv.weights.shift(1)
    non_rub["weight_prev"] = [prev_weight.at[r.ex_date, r.secid] if r.ex_date in prev_weight.index else np.nan
                              for r in non_rub.itertuples()]
    non_rub = non_rub[["secid", "ex_date", "record_date", "amount", "currency", "fx_rate", "fx_date", "amount_rub",
                       "fx_status", "period", "period_type", "source", "source_ref", "weight_prev", "abn_return",
                       "drop_status", "index_status", "implied_fx_price", "implied_fx_flow", "ratio_price",
                       "ratio_flow", "flags"]]
    tables = {
        "events": ev,
        "dividends_rub": ev[ev["usable_rub"]][["secid", "record_date", "ex_date", "amount", "currency", "amount_orig",
                                              "currency_orig", "fx_rate", "fx_date", "source", "source_ref",
                                              "ex_basis", "drop_status", "index_status"]]
        .reset_index(drop=True),
        "cbonds_audit": built.audit,
        "registry_audit": built.registry_audit,
        "registry_leads": registry.leads_frame(registry_leads),
        "corrections_audit": built.corrections_audit,
        "non_rub_events": non_rub.reset_index(drop=True),
        "price_outliers": outliers[["event_id", "secid", "ex_date", "record_date", "amount", "currency",
                                    "restatement_factor", "ex_basis", "dividend_yield", "abn_return", "sigma", "z",
                                    "capture", "drop_status", "index_status", "index_contribution", "flags",
                                    "source_ref"]].reset_index(drop=True),
        "ex_date_disagreements": disagreements[["event_id", "secid", "record_date", "ex_rule", "ex_vendor", "ex_date",
                                                "ex_basis", "dividend_yield", "drop_status", "flags"]]
        .reset_index(drop=True),
        "restatement_verdict": verdict,
        "scale_check": scale,
        "coverage_rub": md.coverage_table(ev, secids=panel.universe["secid"]).rename_axis("secid"),
        "ticker_status": status,
        "index_flow_daily": daily,
        "index_flow_by_year": iv.by_year,
        "index_flow_metrics": iv.metrics.rename_axis("block_sessions"),
        "index_residual_episodes": iv.episodes,
        "index_missing_leads": leads,
        "tr_index_rub": tr,
    }
    not_rub = ev["currency"] != "RUB"
    summary = {
        "all_countries_file": all_check,
        "cbonds_rows": {"total": len(rows), "by_status": _counts(built.audit["status"]),
                        "by_currency": _counts(rows["currency"])},
        "events": {
            "total": len(ev), "by_source": _counts(ev["source"]), "by_currency": _counts(ev["currency"]),
            "usable_rub": int(ev["usable_rub"].sum()), "by_ex_basis": _counts(ev["ex_basis"]),
            "by_drop_status": _counts(ev["drop_status"]), "by_index_status": _counts(ev["index_status"]),
            "by_secid_status": _counts(ev["secid_status"]),
            "flags": _counts(pd.Series([f for fs in ev["flags"] for f in fs.split(",") if f], dtype=object)),
            "tickers_with_events": int(ev["secid"].nunique()),
            "first_ex": str(ev["ex_date"].min().date()) if len(ev) else None,
            "last_ex": str(ev["ex_date"].max().date()) if len(ev) else None,
        },
        "fx": {"provided": fx is not None, "currencies": sorted(fx.columns) if fx is not None else [],
               "converted_events": int(converted.sum()),
               "missing_rate_events": int(ev["flags"].str.contains("fx_missing").sum()),
               "rate_convention": RATE_CONVENTION, "max_age_days": 7,
               # flow_testable: only events large enough to show in the index (not "untestable"/"not_in_flow")
               "rate_check": {
                   "flow": _rate_check(non_rub["ratio_flow"]),
                   "flow_testable": _rate_check(non_rub["ratio_flow"].where(
                       non_rub["index_status"].isin(["confirmed", "inconsistent", "timing_shift"]))),
                   "price": _rate_check(non_rub["ratio_price"])}},
        "corrections": {"n": len(corr), "items": [
            {"source_ref": r.source_ref, "secid": r.secid, "record_date": r.record_date,
             "cbonds_amount": r.cbonds_amount, "amount": r.amount} for r in built.corrections_audit.itertuples()]},
        "registry": {"n_events": len(reg), "n_leads": len(tuple(registry_leads)), "by_secid": _counts(reg["secid"]) if len(reg) else {},
                     "by_status": _counts(built.registry_audit["status"]) if len(built.registry_audit) else {},
                     "no_payment_scopes": [{"secid": n.secid, "scope": n.scope} for n in no_payment]},
        "price_checks": {
            "thresholds": {"min_yield_sigma": config["min_yield_sigma"], "margin_sigma": config["margin_sigma"],
                           "z_borderline": ck.Z_BORDERLINE, "z_outlier": ck.Z_OUTLIER, "scale_band": list(ck.SCALE_BAND)},
            "placebo": placebo,
            "restatement_verdict": {sid: {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v)))
                                          for k, v in row.items()} for sid, row in verdict.iterrows()},
            "scale_check": {"status": _counts(scale["status"]),
                            "suspect": sorted(scale.index[scale["status"] == "scale_suspect"])},
            "outliers": {"n": len(outliers), "by_index_status": _counts(outliers["index_status"]),
                         "by_drop_status": _counts(outliers["drop_status"])},
        },
        "index_flow": {"rub_events": _flow_summary(iv), "native_rub_events": _flow_summary(iv_native),
                       "cbonds_only": _flow_summary(iv_cb)},
        "ticker_status": {"counts": _counts(status["status"]),
                          "unresolved": sorted(status.loc[status["status"] == "unresolved", "secid"]),
                          "unresolved_mean_weight": {
                              r.secid: round(r.mean_weight, 5) for r in status.itertuples()
                              if r.status == "unresolved"},
                          "small_weight": SMALL_WEIGHT,
                          "unresolved_small": {r.secid: round(r.mean_weight, 5) for r in status.itertuples()
                                               if r.status == "unresolved_small"}},
        "non_rub": {"n": int(not_rub.sum()),
                    "by_secid_currency": _counts(ev.loc[not_rub, "secid"] + " " + ev.loc[not_rub, "currency"])},
        "tr_index": {
            "events_used": int(ev["usable_rub"].sum()), "non_rub_events_excluded": int(not_rub.sum()),
            "fx_converted_events_used": int((ev["usable_rub"] & converted).sum()),
            "rub_events_without_previous_quote": int(((~not_rub) & (~ev["usable_rub"])).sum()),
            "tickers_without_events": sorted(set(secids) - set(ev.loc[ev["usable_rub"], "secid"])),
            "note": ("total return from the RUB events and from the foreign-currency events converted at the official "
                     "rate of the ex-date (fx_converted_events_used); events without a rate" if fx is not None else
                     "total return from RUB events only (no exchange-rate table was supplied); events in other "
                     "currencies") + " and payments not established (ticker_status.csv: unresolved, "
                     "no_payment_partial) are absent, so the series understates the total return of the shares "
                     "concerned",
        },
    }
    return Assembly(built=replace(built, events=ev), tables=tables, summary=summary)


# --------------------------------------------------------------------------------------------
# Output and provenance
# --------------------------------------------------------------------------------------------


def write_outputs(assembly: Assembly, out_dir: Path | str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in assembly.tables.items():
        fmt = FORMATS[name]
        if fmt == "parquet":
            frame.to_parquet(out / f"{name}.parquet", index=False)
        elif fmt == "parquet-index":
            frame.to_parquet(out / f"{name}.parquet")
        elif fmt == "csv":
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.6g", encoding="utf-8")
        else:
            frame.to_csv(out / f"{name}.csv", index=True, float_format="%.6g", encoding="utf-8")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.date().isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def provenance_record(
    assembly: Assembly, *, config: dict, git: dict, out_dir: Path, raw_dir: Path, moex_dir: Path, moex_files: list,
    cbonds_import: dict, now: str, fx_record: dict | None = None,
) -> dict:
    code = {p: sha256_file(REPO / p) for p in CODE_FILES}
    return _jsonable({
        "dataset": "moex_dividends", "built_at": now, "git": git, "code_sha256": code,
        "config": {**config, "raw_dir": str(config["raw_dir"])},
        "inputs": {
            "cbonds_import_record": cbonds_import,
            "moex_v2_provenance": {"path": str(Path(config["moex_dir"]) / "provenance.json"),
                                   "sha256": sha256_file(moex_dir / "provenance.json")
                                   if (moex_dir / "provenance.json").exists() else None},
            "moex_v2_files": moex_files,
            "fx": fx_record or {},
            "registry": {"module": "src/cif/data/moex_dividend_registry.py",
                         "n_events": assembly.summary["registry"]["n_events"]},
        },
        "counts": {k: assembly.summary[k]
                   for k in ("cbonds_rows", "events", "corrections", "registry", "non_rub", "ticker_status")},
        "diagnostics": {k: assembly.summary[k]
                        for k in ("all_countries_file", "fx", "price_checks", "index_flow", "tr_index")},
        "schema": SCHEMA,
        "assumptions": {
            "ex_date": "T+2 until record dates up to 31.07.2023, T+1 after; settlement calendar = trading days plus "
                       "the weekdays of the 2022 closure; Cbonds' ex-date arbitrates only when the dividend is "
                       "at least 2 sigma and fits the price better by 1 sigma",
            "amount": "gross, per share in the basis of the ex-date (Cbonds amounts x split restatement factor)",
            "index_flow": "weights from the monthly composition snapshots, drifted with split-adjusted prices; "
                          "RUB-denominated events, foreign-currency ones converted at the official rate (see fx)",
            "fx": RATE_CONVENTION,
            "mcftr_timing": "MCFTR books some dividends one or more sessions off the rule (2013-2014, Dec 2022); "
                            "weekly and monthly shares absorb this, see index_flow",
        },
        "files": {"raw": moex.file_manifest(raw_dir), "processed": moex.file_manifest(out_dir)},
    })


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def git_state(repo: Path = REPO) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return cfg


def build(config: dict, *, git: dict | None = None, allow_dirty: bool = False, now: str | None = None) -> dict:
    git = git or git_state()
    if git["dirty"] and not allow_dirty:
        raise BuildRefused("the working tree is not clean: commit first (the provenance names the commit) or "
                           "pass --allow-dirty")
    raw_dir, moex_dir, out_dir = (REPO / config[k] for k in ("raw_dir", "moex_dir", "out_dir"))
    for directory in (raw_dir, moex_dir):
        report = verify_provenance(directory / "provenance.json", repo_root=REPO)
        if not report["ok"]:
            raise BuildRefused(f"{directory}: provenance does not verify: {report['mismatch']} {report['missing']}")
    rows = cbonds.read_calendar(raw_dir / config["calendar_file"])
    rows_all = cbonds.read_calendar(raw_dir / config["calendar_all_file"])
    panel = md.load_panel(moex_dir)
    fx, fx_record = fx_inputs(config)
    assembly = assemble(rows, rows_all, panel, registry_events=registry.REGISTRY_EVENTS,
                        no_payment=registry.NO_PAYMENT, corrections=registry.CORRECTIONS, config=config, fx=fx,
                        registry_leads=registry.LEADS)
    write_outputs(assembly, out_dir)
    moex_files = [{"path": f"{config['moex_dir']}/{name}.parquet",
                   "sha256": sha256_file(moex_dir / f"{name}.parquet")} for name in MOEX_FILES]
    cbonds_import = json.loads((raw_dir / "provenance.json").read_text(encoding="utf-8"))
    stamp = now or datetime.now(UTC).isoformat(timespec="seconds")
    prov = provenance_record(assembly, config=config, git=git, out_dir=out_dir, raw_dir=raw_dir,
                             moex_dir=moex_dir, moex_files=moex_files, cbonds_import=cbonds_import, now=stamp,
                             fx_record=fx_record)
    (out_dir / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
                                             encoding="utf-8")
    logger.info("built %s: %d events", out_dir, len(assembly.tables["events"]))
    return assembly.summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML file overriding DEFAULT_CONFIG keys")
    ap.add_argument("--allow-dirty", action="store_true", help="build from a dirty working tree")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s")
    try:
        summary = build(load_config(args.config), allow_dirty=args.allow_dirty)
    except BuildRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    ev, flow = summary["events"], summary["index_flow"]["rub_events"]["metrics"]
    print(f"events {ev['total']} (usable RUB {ev['usable_rub']}); by source {ev['by_source']}")
    print(f"index flow explained: ratio of sums {flow['1']['ratio_of_sums']:.3f}; L1 share daily "
          f"{flow['1']['l1_share']:.3f}, weekly {flow['5']['l1_share']:.3f}, monthly {flow['21']['l1_share']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
