"""E3 analysis: H3a, H3b, H3c and their sensitivities on the assembled paths (RQ3).

ANALYSIS_PLAN §2 RQ3, §7.1, §7.4, §10. Input: assemblies (:mod:`cif.experiments.e3_assemble`) of the
markets ``ff49`` (arms MV and GMV), ``us`` (MV) and, as a sensitivity, ``us_irregular``; output: the
numbers of the registry of §7.1 as tables (CSV and one JSON) in the directory of a registered run, not
in ``paper/``. The statistics are :mod:`cif.stats`; what this module does is the hand-off.

* **Series.** The Sharpe functional (MV) gets the daily EXCESS returns of the pass "до издержек"
  (``tc = 0``) on the days of ``F_k``; the log-variance functional (GMV) the returns themselves (the
  variance the arm minimises; rf varies too little to matter). Optimum and rank paths are on the same
  days (:func:`cif.experiments.e3_assemble.fk_series`).
* **Margins** are the values of the pilot (plan §2 H3a, §12): ``Δ_SR = SESOI/σ*`` for MV and US-MV,
  ``δ = (0.005/λ_f)/σ²_*`` for GMV, margin ``[log(1 − δ), log(1 + δ)]`` of the log variance ratio. They
  are in the config; an unset (``null``) margin stops the analysis.
* **Block.** LW Algorithm 3.1 on the pair (optimum, rank 2) of the market and arm, in percent, daily
  grid, 1000 pseudo-series, 499 repetitions (:func:`cif.stats.rq3_ranks.calibrate_rank_block`; sizes
  from the config); its block is used for H3a, H3b and H3c of that market and arm. With
  ``calibration.fixed_block`` the calibration is replaced by that length.
* **H3a** (:func:`cif.stats.rq3_ranks.rank_equivalence`): rank 2 against the optimum, one TOST and one
  two-sided test per arm. FF49-MV is the primary test; FF49-GMV and US-MV are the secondary ones,
  adjusted by Holm over the two (:func:`cif.stats.rq3_ranks.holm_over_arms`).
* **H3b** (:func:`cif.stats.rq3_ranks.rank_dose_response`): the slope of ``ΔSR(r)`` (``Δ`` of the log
  variance for GMV) on ``log ḡ_r`` over ranks 2, 3, 5, 9, 17, 33 of ``F_k``; sensitivities: ``log r``
  and all ranks 2-33. The leverage of the rank-2 point is reported. If some mean dose is not positive
  the logarithm does not exist and H3b is descriptive (the ``log r`` fit remains).
* **H3c** (FF49-MV, tabu / SA / LS; SCIP is excluded): :func:`cif.stats.rq3_ranks.solver_residuals`
  with the dose of the solver, the rank distribution in the bins 1; 2-5; 6-33; > 33 and, separately,
  the NET Sharpe difference and the turnover of the actual path with stationary-bootstrap intervals.
* **Sensitivities** (descriptive): fixed blocks, the margin ``Δ_SR`` of 0.02 and 0.05, the other
  passes of the assembly (execution lag, the terminal return of a delisting), the folds with at least
  ``coverage_min`` of the index members covered by data (US), the HAC test of LW (2008) on percent
  returns, and the net metrics at 0 / 10 / 30 bp.
* **Seeds.** One seed per test and variant: ``seed_for(stats.seed, market, arm, test, variant)``.

Registered stage: ``python -m cif.experiments.e3_analysis --analysis-config configs/E3/analysis.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zlib
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cif.backtest.fold_sets import G_STAR_BP, rank_doses
from cif.experiments import e3_assemble as asmb
from cif.experiments import e3_data as d
from cif.experiments import registry
from cif.stats import (
    bootstrap_ci,
    calibrate_rank_block,
    calibrate_replica_block,
    holm_over_arms,
    joint_contrasts,
    ledoit_wolf_sharpe_test,
    rank_bins,
    rank_dose_response,
    rank_equivalence,
    solver_residuals,
)
from cif.stats.rq3_ranks import H3B_RANKS
from cif.stats.rq3_ranks import RankEquivalence as _RankEquivalence

__all__ = [
    "AnalysisError",
    "analyse_arm",
    "analyse_markets",
    "analyse_operators",
    "describe",
    "main",
    "margin_for",
    "seed_for",
    "stats_config",
    "write_tables",
]

TRADING_DAYS = 252
_MIN_DAYS = 10  # the least number of days the Sharpe tests of cif.stats take
HOLM_FAMILY = ("ff49-GMV", "us-MV")  # the secondary arms of H3a (plan §7.1)
PRIMARY = "ff49-MV"
_STATS_KEYS = ("seed", "alpha", "n_boot", "calibration", "fixed_blocks", "delta_sr_alt", "g_star_bp",
               "h3c_solvers", "coverage_min", "net_passes", "sensitivity_passes", "net_bootstrap",
               "gamma")
_CALIBRATION_KEYS = ("grid", "n_pseudo", "n_boot")


class AnalysisError(RuntimeError):
    """The analysis cannot be run as configured (a pilot value is missing, a margin is not valid)."""


# --------------------------------------------------------------------------------------------
# Configuration, margins, seeds
# --------------------------------------------------------------------------------------------


def stats_config(acfg: dict) -> dict:
    """The ``stats`` section of the analysis config, checked: every key is stated (R8)."""
    sc = acfg.get("stats")
    if not isinstance(sc, dict):
        raise ValueError("the analysis config has no 'stats' section")
    for key in _STATS_KEYS:
        if key not in sc:
            raise ValueError(f"stats: missing '{key}'")
    for key in _CALIBRATION_KEYS:
        if key not in sc["calibration"]:
            raise ValueError(f"stats.calibration: missing '{key}'")
    if "n_boot" not in sc["net_bootstrap"]:
        raise ValueError("stats.net_bootstrap: missing 'n_boot'")
    return sc


def seed_for(base: int, *labels) -> int:
    """A 31-bit seed from the base and the labels of a test (strings are hashed with CRC-32)."""
    ints = [int(x) if isinstance(x, (int, np.integer)) and not isinstance(x, bool)
            else zlib.crc32(str(x).encode("utf-8")) for x in labels]
    seq = np.random.SeedSequence([int(base), *ints])
    return int(seq.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF


def margin_for(arm: str, acfg: dict) -> tuple[float | tuple[float, float], dict]:
    """The margin of H3a for an arm and what it was made of (module docstring)."""
    m = acfg.get("margins") or {}
    sesoi = m.get("sesoi", 0.0025)
    if arm == "MV":
        if m.get("delta_sr") is None:
            raise AnalysisError("margins.delta_sr is not set: Δ_SR = SESOI/σ* is a [пилот] value "
                                "(plan §2 H3a); run the pilot and enter it here and in §12")
        return float(m["delta_sr"]), {"delta_sr": float(m["delta_sr"]), "sesoi": float(sesoi)}
    if m.get("delta_var") is None:
        raise AnalysisError("margins.delta_var is not set: δ = (0.005/λ_f)/σ²_* is a [пилот] value "
                            "(plan §2 H3a); run the pilot and enter it here and in §12")
    delta = float(m["delta_var"])
    if not 0.0 < delta < 1.0:
        raise AnalysisError(f"margins.delta_var = {delta} must lie in (0, 1): log(1 − δ) must exist")
    return (math.log1p(-delta), math.log1p(delta)), {"delta_var": delta, "sesoi": float(sesoi)}


def _functional(arm: str) -> str:
    return "sharpe" if arm == "MV" else "logvar"


# --------------------------------------------------------------------------------------------
# Result dictionaries
# --------------------------------------------------------------------------------------------


def _estimate(c) -> dict:
    return {"estimate": c.estimate, "se": c.se, "ci": list(c.ci), "ci_percentile": list(c.ci_percentile),
            "pvalue": c.pvalue}


def _equivalence(res) -> dict:
    out = asdict(res)
    for key in ("ci_tost", "ci", "margin"):
        out[key] = list(out[key])
    return d.json_safe(out)


def _equivalence_from(h3a: dict) -> _RankEquivalence:
    """A stored H3a result as the dataclass ``holm_over_arms`` reads."""
    fields = dict(h3a)
    for key in ("ci_tost", "ci", "margin"):
        fields[key] = tuple(fields[key])
    return _RankEquivalence(**fields)


def _dose_response(res, descriptive_ranks: bool = False) -> dict:
    return d.json_safe({
        "ranks": list(res.ranks), "doses_bp": list(res.doses_bp), "regressor": res.regressor,
        "x": list(res.x), "per_rank": [_estimate(c) for c in res.per_rank],
        "slope": _estimate(res.slope), "intercept": _estimate(res.intercept),
        "significant": res.significant, "alpha": res.alpha, "functional": res.functional,
        "block": res.block, "n_boot": res.n_boot, "n_obs": res.n_obs, "seed": res.seed,
    })


# --------------------------------------------------------------------------------------------
# The block, H3a, H3b, H3c
# --------------------------------------------------------------------------------------------


def _calibrate(series: asmb.FkSeries, sc: dict, seed: int) -> dict:
    cal = sc["calibration"]
    if cal.get("fixed_block") is not None:
        return {"block": int(cal["fixed_block"]), "source": "fixed", "rejection_rates": None,
                "grid": None, "n_pseudo": None, "n_boot": None, "seed": None}
    res = calibrate_rank_block(
        series.opt, series.ranks[:, 0], seed=seed, functional=series.functional,
        block_grid=[int(b) for b in cal["grid"]], n_pseudo=int(cal["n_pseudo"]),
        n_boot=int(cal["n_boot"]),
    )
    return d.json_safe({
        "block": int(res.block), "source": "calibration",
        "rejection_rates": [float(x) for x in res.rejection_rates],
        "grid": list(res.block_grid), "n_pseudo": res.n_pseudo, "n_boot": res.n_boot,
        "seed": res.seed,
    })


def _h3a(series: asmb.FkSeries, margin, block: int, sc: dict, seed: int) -> dict:
    res = rank_equivalence(
        series.opt, series.ranks[:, 0], margin, functional=series.functional, block=block,
        n_boot=int(sc["n_boot"]), alpha=float(sc["alpha"]), seed=seed,
    )
    return _equivalence(res)


def _mean_doses(asm: asmb.Assembly, arm: str, ranks: list[int], fold_mask: np.ndarray) -> np.ndarray:
    return rank_doses(asm.doses[arm].to_numpy(), list(asm.doses[arm].columns), fold_mask, ranks)


def _leverage_of_first(x: np.ndarray) -> float:
    centred = x - x.mean()
    return float(1.0 / x.size + centred[0] ** 2 / float(centred @ centred))


def _h3b_fit(series: asmb.FkSeries, doses: np.ndarray, regressor: str, block: int, sc: dict,
             seed: int, per_rank: bool) -> dict:
    res = rank_dose_response(
        series.opt, series.ranks, series.rank_ids, doses, functional=series.functional,
        regressor=regressor, block=block, n_boot=int(sc["n_boot"]), alpha=float(sc["alpha"]),
        seed=seed, per_rank=per_rank,
    )
    return _dose_response(res)


def _h3b(asm: asmb.Assembly, arm: str, market: str, series_h3b: asmb.FkSeries,
         series_all: asmb.FkSeries, fold_mask: np.ndarray, block: int, sc: dict, variant: str) -> dict:
    """H3b: the primary fit on ``log ḡ_r`` and the sensitivities (descriptive if a dose is ≤ 0)."""
    ranks = series_h3b.rank_ids
    doses = _mean_doses(asm, arm, ranks, fold_mask)
    base = sc["seed"]

    def seed(regressor: str) -> int:
        return seed_for(base, market, arm, "h3b", regressor if variant == "main" else f"{regressor}:{variant}")

    out: dict = {"ranks": list(ranks), "leverage_rank2": None}
    out["log_rank"] = _h3b_fit(series_h3b, doses if (doses > 0).all() else np.ones(len(ranks)),
                               "log_rank", block, sc, seed("log_rank"), True)
    if not (np.isfinite(doses).all() and (doses > 0).all()):
        out.update(descriptive=True, doses_bp=[float(x) for x in doses],
                   reason="the mean doses of ranks "
                          f"{[r for r, x in zip(ranks, doses, strict=True) if not x > 0]} are not "
                          "positive (ties with the optimum on every fold): log ḡ_r does not exist")
        return d.json_safe(out)
    out["descriptive"] = False
    out["leverage_rank2"] = _leverage_of_first(np.log(doses))
    out["log_dose"] = _h3b_fit(series_h3b, doses, "log_dose", block, sc, seed("log_dose"), True)
    if len(series_all.rank_ids) > len(ranks):
        every = _mean_doses(asm, arm, series_all.rank_ids, fold_mask)
        if (every > 0).all():
            out["all_ranks"] = _h3b_fit(series_all, every, "log_dose", block, sc, seed("all_ranks"),
                                        False)
    return d.json_safe(out)


def _solver_gaps(asm: asmb.Assembly, arm: str, name: str) -> np.ndarray:
    gaps = np.full(len(asm.folds), np.nan)
    sub = asm.solvers[(asm.solvers["arm"] == arm) & (asm.solvers["solver"] == name)]
    gaps[sub["fold"].to_numpy()] = sub["gap_bp"].to_numpy(dtype=float)
    return gaps


def _solver_ranks(asm: asmb.Assembly, arm: str, name: str, fold_mask: np.ndarray) -> tuple[dict, int]:
    ranks = np.full(len(asm.folds), np.nan)
    sub = asm.solvers[(asm.solvers["arm"] == arm) & (asm.solvers["solver"] == name)]
    ranks[sub["fold"].to_numpy()] = sub["rank"].to_numpy(dtype=float)
    chosen = ranks[fold_mask]
    n_unexplained = int(np.isnan(chosen).sum())
    top = int(asm.rank_ids[-1]) + 1  # "beyond the list"
    coded = np.where(np.isnan(ranks), top, ranks)
    bins = rank_bins(coded, fold_mask)
    return {"labels": list(bins.labels), "counts": list(bins.counts), "shares": list(bins.shares),
            "n_folds": bins.n_folds}, n_unexplained


def _h3c(asm: asmb.Assembly, arm: str, market: str, series_h3b: asmb.FkSeries, doses: np.ndarray,
         fold_mask: np.ndarray, block: int, margin, sc: dict, h3b_ok: bool,
         net: dict | None) -> dict:
    names = [s for s in sc["h3c_solvers"] if s in series_h3b.solvers]
    if not names:
        return {"descriptive": True, "iut_passes": None, "reason": "no solver paths", "solvers": []}
    info = []
    for name in names:
        gaps = _solver_gaps(asm, arm, name)
        chosen = gaps[fold_mask]
        bins, n_unexplained = _solver_ranks(asm, arm, name, fold_mask)
        info.append({"name": name, "rank_bins": bins, "n_rank_unexplained": n_unexplained,
                     "gap_mean_bp": float(np.mean(chosen)), "gap_median_bp": float(np.median(chosen)),
                     "net": (net or {}).get(f"solver_{name}")})
    if not h3b_ok:
        return d.json_safe({"descriptive": True, "iut_passes": None,
                            "reason": "H3b is descriptive: there is no curve to calibrate against",
                            "solvers": info})
    gaps = {n: _solver_gaps(asm, arm, n) for n in names}
    res = solver_residuals(
        series_h3b.opt, series_h3b.ranks, series_h3b.rank_ids, doses,
        {n: series_h3b.solvers[n] for n in names}, gaps, fold_mask, margin, functional="sharpe",
        block=block, n_boot=int(sc["n_boot"]), alpha=float(sc["alpha"]),
        seed=seed_for(sc["seed"], market, arm, "h3c", "main"),
    )
    solvers = []
    for s, extra in zip(res.solvers, info, strict=True):
        solvers.append({
            "name": s.name,
            "dose": {"dose_bp": s.dose.dose_bp, "median_gap_bp": s.dose.median_gap_bp,
                     "n_folds": s.dose.n_folds, "material": s.dose.material,
                     "inside_range": s.dose.inside_range, "descriptive": s.dose.descriptive,
                     "reason": s.dose.reason},
            "observed": None if s.observed is None else _estimate(s.observed),
            "predicted": s.predicted,
            "residual": None if s.residual is None else _estimate(s.residual),
            "ci_tost": None if s.ci_tost is None else list(s.ci_tost),
            "equivalent": s.equivalent,
            "rank_bins": extra["rank_bins"], "n_rank_unexplained": extra["n_rank_unexplained"],
            "net": extra["net"],
        })
    return d.json_safe({
        "descriptive": res.descriptive, "iut_passes": res.iut_passes, "reason": res.reason,
        "n_material": res.n_material, "margin": list(res.margin), "alpha": res.alpha,
        "block": res.block, "n_boot": res.n_boot, "n_obs": res.n_obs, "seed": res.seed,
        "solvers": solvers,
    })


def _hac_check(series: asmb.FkSeries) -> dict:
    res = ledoit_wolf_sharpe_test(100.0 * series.ranks[:, 0], 100.0 * series.opt, method="hac",
                                  periods_per_year=TRADING_DAYS)
    return d.json_safe({"difference": res.difference, "se": res.se, "pvalue": res.pvalue,
                        "ci_low": res.ci_low, "ci_high": res.ci_high, "method": res.method,
                        "n_obs": res.n_obs})


# --------------------------------------------------------------------------------------------
# Net-of-cost metrics (secondary, plan §2): estimates with stationary-bootstrap intervals
# --------------------------------------------------------------------------------------------


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * math.sqrt(TRADING_DAYS)


def _ce_rows(x: np.ndarray, gamma: float) -> np.ndarray:
    return (x.mean(axis=1) - 0.5 * gamma * x.var(axis=1, ddof=1)) * TRADING_DAYS


def _paths_for_net(asm: asmb.Assembly, arm: str, frame: pd.DataFrame) -> list[str]:
    names = ["opt"]
    names += [f"rank_{r}" for r in asm.rank_ids if (r == 2 or r in H3B_RANKS)
              and f"{arm}:rank_{r}" in frame.columns]
    names += [c.split(":", 1)[1] for c in frame.columns if c.startswith(f"{arm}:solver_")]
    return names


def _net_metrics(asm: asmb.Assembly, arm: str, market: str, sc: dict) -> dict | None:
    levels = {label: p for label, p in sc["net_passes"].items() if p in asm.paths}
    if not levels:
        return None
    mask = asm.days["in_Fk"].to_numpy()
    fold_in = asm.masks["common"]
    rf = asm.days["rf"].to_numpy()[mask]
    years = mask.sum() / TRADING_DAYS
    gamma = asm.meta["lam"] if sc["gamma"] == "lambda_f" else float(sc["gamma"])
    n_boot = int(sc["net_bootstrap"]["n_boot"])
    table, diffs = [], []
    summary: dict = {}
    for label, pass_name in levels.items():
        frame = asm.paths[pass_name]
        stats = asm.fold_stats[pass_name]
        stats = stats[(stats["arm"] == arm) & stats["fold"].map(lambda f: bool(fold_in[f]))]
        paths = _paths_for_net(asm, arm, frame)
        excess = {p: frame[f"{arm}:{p}"].to_numpy()[mask] - rf for p in paths}
        turnover = {p: stats[stats["path"] == p].sort_values("fold")["turnover"].to_numpy() for p in paths}
        cost = {p: stats[stats["path"] == p]["cost"].sum() for p in paths}
        for p in paths:
            x = excess[p][None, :]
            table.append({
                "cost": label, "pass": pass_name, "path": p, "n_days": int(mask.sum()),
                "sharpe": float(_sharpe_rows(x)[0]), "ce": float(_ce_rows(x, gamma)[0]),
                "annual_turnover": float(turnover[p].sum() / years),
                "annual_cost_bps": float(cost[p] / years * 1e4),
            })
        base = table[-len(paths)]
        for p in paths[1:]:
            row = next(r for r in table[-len(paths):] if r["path"] == p)
            entry = {"cost": label, "pass": pass_name, "path": p,
                     "sharpe_diff": row["sharpe"] - base["sharpe"], "ce_diff": row["ce"] - base["ce"],
                     "turnover_diff": row["annual_turnover"] - base["annual_turnover"],
                     "cost_diff_bps": row["annual_cost_bps"] - base["annual_cost_bps"]}
            if p == "rank_2" or p.startswith("solver_"):
                seed = seed_for(sc["seed"], market, arm, "net", label, p)
                for key, fn in (("sharpe_diff", lambda a, b: _sharpe_rows(b) - _sharpe_rows(a)),
                                ("ce_diff", lambda a, b, g=gamma: _ce_rows(b, g) - _ce_rows(a, g))):
                    ci = bootstrap_ci(fn, excess["opt"], excess[p], method="stationary", n_boot=n_boot,
                                      alpha=float(sc["alpha"]), seed=seed, vectorized=True)
                    entry[f"{key}_lo"], entry[f"{key}_hi"] = (float(ci.ci_percentile[0]),
                                                              float(ci.ci_percentile[1]))
                ci = bootstrap_ci(lambda a, b: b.mean(axis=1) - a.mean(axis=1), turnover["opt"],
                                  turnover[p], method="stationary", n_boot=n_boot,
                                  alpha=float(sc["alpha"]), seed=seed, vectorized=True)
                scale = float(len(turnover["opt"])) / years  # mean per fold to a per-year sum
                entry["turnover_diff_lo"] = float(ci.ci_percentile[0]) * scale
                entry["turnover_diff_hi"] = float(ci.ci_percentile[1]) * scale
                if label == "cost_10" or len(levels) == 1:
                    summary[p] = {"sharpe_difference": {
                        "estimate": entry["sharpe_diff"],
                        "ci_percentile": [entry["sharpe_diff_lo"], entry["sharpe_diff_hi"]]},
                        "turnover_difference": {
                            "estimate": entry["turnover_diff"],
                            "ci_percentile": [entry["turnover_diff_lo"], entry["turnover_diff_hi"]]}}
            diffs.append(entry)
    return {"table": table, "differences": diffs, "summary": summary,
            "bootstrap": {"method": "stationary", "n_boot": n_boot, "block": "politis-white-patton",
                          "alpha": float(sc["alpha"])},
            "years": float(years), "gamma": float(gamma)}


# --------------------------------------------------------------------------------------------
# One arm of one market
# --------------------------------------------------------------------------------------------


def analyse_arm(
    asm: asmb.Assembly, arm: str, market: str, acfg: dict, *, pass_name: str = "gross"
) -> dict:
    """H3a, H3b, H3c, the HAC check, the net metrics and the sensitivities of one arm (module
    docstring). ``market`` is the label of the assembly (``ff49``, ``us``, ``us_irregular``)."""
    sc = stats_config(acfg)
    margin, margin_info = margin_for(arm, acfg)
    functional = _functional(arm)
    key = f"{market}-{arm}"
    series_all = asmb.fk_series(asm, pass_name, arm, functional)
    h3b_ranks = [r for r in H3B_RANKS if r in series_all.rank_ids]
    series_h3b = asmb.fk_series(asm, pass_name, arm, functional, ranks=h3b_ranks)
    mask = asm.masks["common"]
    out: dict = {
        "key": key, "market": market, "arm": arm, "functional": functional, "pass": pass_name,
        "n_days": len(series_all.opt), "n_folds_Fk": int(mask.sum()),
        "years": len(series_all.opt) / TRADING_DAYS, "margin_info": margin_info,
        "lam": asm.meta["lam"], "n_ranks": asm.meta["n_ranks"],
    }
    if 2 not in series_all.rank_ids:
        raise AnalysisError(f"{key}: the pass {pass_name!r} holds no rank-2 path")
    two = asmb.fk_series(asm, pass_name, arm, functional, ranks=[2])
    out["block"] = _calibrate(two, sc, seed_for(sc["seed"], market, arm, "calibration", "main"))
    block = out["block"]["block"]
    out["h3a"] = _h3a(two, margin, block, sc, seed_for(sc["seed"], market, arm, "h3a", "main"))
    if functional == "sharpe":
        out["hac"] = _hac_check(two)

    net = _net_metrics(asm, arm, market, sc) if functional in ("sharpe", "logvar") else None
    if net is not None:
        out["net"] = {k: v for k, v in net.items() if k != "summary"}

    has_h3b = len(h3b_ranks) >= 3
    if has_h3b:
        out["h3b"] = _h3b(asm, arm, market, series_h3b, series_all, mask, block, sc, "main")
        if market == "ff49" and arm == "MV" and series_h3b.solvers:
            doses = _mean_doses(asm, arm, h3b_ranks, mask)
            out["h3c"] = _h3c(asm, arm, market, series_h3b, doses, mask, block, margin, sc,
                              h3b_ok=not out["h3b"].get("descriptive"),
                              net=None if net is None else net["summary"])
    out["sensitivity"] = _sensitivity(asm, arm, market, acfg, sc, margin, block, out)
    return d.json_safe(out)


def _sensitivity(asm, arm, market, acfg, sc, margin, block, main) -> dict:
    functional = _functional(arm)
    mask = asm.masks["common"]
    sens: dict = {}
    two = asmb.fk_series(asm, "gross", arm, functional, ranks=[2])
    h3b_ranks = [r for r in H3B_RANKS if r in asm.rank_ids]
    has_h3b = len(h3b_ranks) >= 3

    def h3b_for(series_h3b, series_all, fold_mask, blk, variant):
        return _h3b(asm, arm, market, series_h3b, series_all, fold_mask, blk, sc, variant)

    full = asmb.fk_series(asm, "gross", arm, functional)
    core = asmb.fk_series(asm, "gross", arm, functional, ranks=h3b_ranks)
    sens["fixed_blocks"] = []
    for b in sc["fixed_blocks"]:
        entry = {"block": int(b), "h3a": _h3a(two, margin, int(b), sc,
                                              seed_for(sc["seed"], market, arm, "h3a", f"block{b}"))}
        if has_h3b:
            entry["h3b"] = h3b_for(core, full, mask, int(b), f"block{b}")
        sens["fixed_blocks"].append(entry)
    if functional == "sharpe":
        sens["delta_sr"] = [
            {"delta_sr": float(delta),
             "h3a": _h3a(two, float(delta), block, sc, seed_for(sc["seed"], market, arm, "h3a", "main"))}
            for delta in sc["delta_sr_alt"]
        ]
    passes = {}
    for label, name in sc["sensitivity_passes"].items():
        if name not in asm.paths:
            passes[label] = {"pass": name, "skipped": "the pass is not in the assembly"}
            continue
        two_p = asmb.fk_series(asm, name, arm, functional, ranks=[2])
        entry = {"pass": name, "h3a": _h3a(two_p, margin, block, sc,
                                           seed_for(sc["seed"], market, arm, "h3a", f"pass:{label}"))}
        ranks_p = [r for r in H3B_RANKS if f"{arm}:rank_{r}" in asm.paths[name].columns]
        if len(ranks_p) >= 3:
            core_p = asmb.fk_series(asm, name, arm, functional, ranks=ranks_p)
            entry["h3b"] = _h3b(asm, arm, market, core_p, core_p, mask, block, sc, f"pass:{label}")
        passes[label] = entry
    if passes:
        sens["passes"] = passes
    if "x_coverage" in asm.folds.columns:
        keep = (asm.folds["x_coverage"].to_numpy(dtype=float) >= float(sc["coverage_min"]))
        kept = keep & mask
        entry = {"coverage_min": float(sc["coverage_min"]), "n_folds": int(kept.sum())}
        if kept.any():
            two_c = asmb.fk_series(asm, "gross", arm, functional, ranks=[2], fold_mask=keep)
            entry["h3a"] = _h3a(two_c, margin, block, sc,
                                seed_for(sc["seed"], market, arm, "h3a", "coverage"))
        else:
            entry["skipped"] = "no fold of F_k is covered well enough"
        sens["coverage"] = entry
    return sens


# --------------------------------------------------------------------------------------------
# Descriptive tables
# --------------------------------------------------------------------------------------------


def describe(asm: asmb.Assembly, acfg: dict) -> dict[str, pd.DataFrame]:
    """The descriptive tables of the plan (§2 "Набор фолдов", "Ничьи"): exclusions, the distribution
    of the dose of rank 2 (all folds of ``F_k`` and split by binding / non-binding), the doses by rank,
    the regime of the folds and the share of ties."""
    sc = stats_config(acfg)
    g_star = float(sc["g_star_bp"] if sc.get("g_star_bp") is not None else G_STAR_BP)
    mask = asm.masks["common"]
    folds = asm.folds
    rows_ex = []
    for key, s in asm.fold_sets.items():
        rows_ex.append({"set": key, "n_folds": s.n_folds, "n_included": s.n_included,
                        "n_excluded": s.n_excluded, "share_excluded": s.share_excluded,
                        "n_excluded_optimum": s.n_excluded_optimum,
                        "n_excluded_degenerate": s.n_excluded_degenerate,
                        "n_excluded_rank": s.n_excluded_rank, "n_missing_cells": asm.meta["n_missing"]})
    binding = folds["binding"].to_numpy(dtype=bool)
    r2_rows, rank_rows, tie_rows = [], [], []
    for arm in asm.doses:
        table = asm.doses[arm]
        g2 = table[2].to_numpy()[mask]
        for stratum, sel in (("all", np.ones(mask.sum(), bool)), ("binding", binding[mask]),
                             ("non_binding", ~binding[mask])):
            g = g2[sel]
            r2_rows.append({
                "arm": arm, "stratum": stratum, "n_folds": int(g.size),
                "mean": float(np.mean(g)) if g.size else None,
                "q1": float(np.quantile(g, 0.25)) if g.size else None,
                "median": float(np.median(g)) if g.size else None,
                "q3": float(np.quantile(g, 0.75)) if g.size else None,
                "share_le_gstar": float(np.mean(g <= g_star)) if g.size else None,
            })
        for rank in table.columns:
            g = table[rank].to_numpy()[mask]
            rank_rows.append({
                "arm": arm, "rank": int(rank), "n_folds": int(g.size), "mean": float(np.mean(g)),
                "q1": float(np.quantile(g, 0.25)), "median": float(np.median(g)),
                "q3": float(np.quantile(g, 0.75)),
            })
        doses = np.column_stack([np.zeros(len(table)), table.to_numpy()])[mask]
        gaps = np.diff(doses, axis=1)
        tol = folds[f"tie_tol_bp_{arm}"].to_numpy(dtype=float)[mask][:, None]
        tied = gaps <= tol
        boundary = folds[f"boundary_tie_{arm}"][mask].dropna().astype(bool)
        tie_rows.append({
            "arm": arm, "n_folds_Fk": int(mask.sum()), "n_pairs": int(tied.size),
            "n_tied_pairs": int(tied.sum()),
            "share_tied_pairs": float(tied.mean()) if tied.size else None,
            "n_boundary_known": int(boundary.size),
            "share_boundary_tie": float(boundary.mean()) if boundary.size else None,
        })
    in_fk = folds[mask]
    regime = [{
        "n_folds": len(folds), "n_folds_Fk": int(mask.sum()),
        "share_binding": float(binding[mask].mean()) if mask.any() else None,
        "share_lower_bound": float(folds["lower_bound_active"].to_numpy(dtype=bool)[mask].mean())
        if mask.any() else None,
        "n_assets_median": float(in_fk["n_assets"].median()) if mask.any() else None,
        "n_assets_min": int(in_fk["n_assets"].min()) if mask.any() else None,
        "K_median": float(in_fk["K"].median()) if mask.any() else None,
        "depth_median": float(in_fk["depth"].median()) if mask.any() else None,
    }]
    return {
        "exclusions": pd.DataFrame(rows_ex), "rank2_doses": pd.DataFrame(r2_rows),
        "doses_by_rank": pd.DataFrame(rank_rows), "regime": pd.DataFrame(regime),
        "ties": pd.DataFrame(tie_rows),
    }


# --------------------------------------------------------------------------------------------
# The descriptive operators (б) and (а): estimates, no tests (plan §2, §7.4)
# --------------------------------------------------------------------------------------------


def _phi(x: np.ndarray, functional: str) -> float:
    """The statistic of the arm: annual Sharpe ratio, or the log of the variance."""
    if functional == "sharpe":
        return float(x.mean() / x.std(ddof=1) * math.sqrt(TRADING_DAYS))
    return float(math.log(x.var(ddof=1)))


def _operator_turnover(ops, op: str, arm: str, path: str, mask: np.ndarray, years: float) -> float | None:
    name = f"{op}:net" if f"{op}:net" in ops.fold_stats else next(
        (k for k in ops.fold_stats if k.startswith(f"{op}:")), None)
    if name is None:
        return None
    stats = ops.fold_stats[name]
    sub = stats[(stats["arm"] == arm) & (stats["path"] == path) & stats["fold"].map(lambda f: bool(mask[f]))]
    return float(sub["turnover"].sum() / years)


def analyse_operators(asm: asmb.Assembly, acfg: dict) -> dict[str, dict]:
    """For each arm and operator of the assembly: the reach (``asm.operators.reach``), and for the
    levels of ``L`` on the days of ``F_b`` the Sharpe difference (log variance ratio, GMV) of every
    replica path to the optimum, their mean with the standard error and the intervals of the joint
    circular block bootstrap (LW 2008; block calibrated on the pair (optimum, mean of the replicas),
    plan §7.4), the realised gaps and the turnover of the replicas against the optimum. Descriptive:
    no test, no margin."""
    sc = stats_config(acfg)
    ops = asm.operators
    if ops is None:
        return {}
    out: dict = {}
    for key, info in ops.sets.items():
        arm, op = key.split(":")
        functional = _functional(arm)
        mask_b = info["mask_b"]
        entry: dict = {
            "arm": arm, "operator": op, "L": list(info["L"]), "L_a": list(info["L_a"]),
            "L_source": info["L_source"], "min_share": info["min_share"], "g_star_bp": info["g_star_bp"],
            "n_ran": info["n_ran"], "n_folds_Fa": int(info["mask_a"].sum()),
            "n_folds_Fb": int(mask_b.sum()), "levels": [], "block": None, "note": None,
            # where L was read: the pilot run, the folds it ran on and the share reached per level
            "pilot_run": info.get("pilot_run"), "pilot_n_ran": info.get("pilot_n_ran"),
            "pilot_share": info.get("pilot_share"), "levels_all": info["levels_all"],
        }
        out[key] = entry
        if not info["L"] or not mask_b.any():
            entry["note"] = ("no level is reached on enough folds (L or F_b is empty): the "
                             "differences are not estimated")
            continue
        day_mask = mask_b[asm.days["fold"].to_numpy()]
        shift = asm.days["rf"].to_numpy()[day_mask] if functional == "sharpe" else 0.0
        n_days = int(day_mask.sum())
        if n_days < _MIN_DAYS:
            entry["note"] = (f"F_b has {n_days} days, fewer than the {_MIN_DAYS} an estimate needs: "
                             "the differences are not estimated")
            continue
        years = n_days / TRADING_DAYS
        frame = ops.paths[f"{op}:gross"]
        opt = frame[f"{arm}:opt"].to_numpy()[day_mask] - shift
        names: dict[float, list[str]] = {}
        replicas: dict[float, np.ndarray] = {}
        for g in info["L"]:
            cols = [c for c in frame.columns if c.startswith(f"{arm}:g{int(g)}:")]
            names[g] = [c.split(":", 1)[1] for c in cols]
            replicas[g] = np.column_stack([frame[c].to_numpy()[day_mask] - shift for c in cols])
        cal = sc["calibration"]
        # a descriptive operator may have few days in F_b: the plan's grid is cut to the blocks that leave
        # two copies in the series (recorded), instead of stopping the analysis after the whole run
        grid = [int(b) for b in cal["grid"]]
        fit = [b for b in grid if b <= n_days // 2]
        cut = [b for b in grid if b > n_days // 2]
        if cal.get("fixed_block") is not None:
            block, source, rates = int(cal["fixed_block"]), "fixed", None
            fit = [block] if block <= n_days // 2 else []
        elif fit:
            res = calibrate_replica_block(
                opt, list(replicas.values()), seed=seed_for(sc["seed"], "ops", arm, op, "calibration"),
                functional=functional, block_grid=fit,
                n_pseudo=int(cal["n_pseudo"]), n_boot=int(cal["n_boot"]),
            )
            block, source, rates = int(res.block), "calibration", [float(x) for x in res.rejection_rates]
        if not fit:
            entry["note"] = (f"F_b has {n_days} days: no block of the grid {grid} leaves two copies in "
                             "the series; the differences are not estimated")
            continue
        entry["block"] = {"block": block, "source": source, "rejection_rates": rates, "grid": fit,
                          "grid_cut": cut}
        series = np.column_stack([opt, *replicas.values()])
        weights = np.zeros((len(info["L"]), series.shape[1]))
        weights[:, 0] = -1.0
        col = 1
        for q, g in enumerate(info["L"]):
            r = replicas[g].shape[1]
            weights[q, col : col + r] = 1.0 / r
            col += r
        joint = joint_contrasts(
            series, weights, functional=functional, block=block, n_boot=int(sc["n_boot"]),
            seed=seed_for(sc["seed"], "ops", arm, op, "bootstrap"),
        )
        lo, hi = joint.ci(float(sc["alpha"]))
        plo, phi = joint.percentile_ci(float(sc["alpha"]))
        phi_opt = _phi(opt, functional)
        gaps = ops.gaps[(ops.gaps["arm"] == arm) & (ops.gaps["operator"] == op)
                        & mask_b[ops.gaps["fold"].to_numpy(dtype=int)]]
        turn_opt = _operator_turnover(ops, op, arm, "opt", mask_b, years)
        for q, g in enumerate(info["L"]):
            per = [_phi(replicas[g][:, j], functional) - phi_opt for j in range(replicas[g].shape[1])]
            sub = gaps[gaps["level"] == g]["gap_bp"].to_numpy(dtype=float)
            turn = [_operator_turnover(ops, op, arm, n, mask_b, years) for n in names[g]]
            turn_rep = None if any(t is None for t in turn) else float(np.mean(turn))
            entry["levels"].append(d.json_safe({
                "level": g, "n_days": int(day_mask.sum()), "n_replicas": len(per), "per_replica": per,
                "mean_difference": float(np.mean(per)), "replica_min": float(np.min(per)),
                "replica_max": float(np.max(per)), "replica_sd": float(np.std(per, ddof=1)) if len(per) > 1 else None,
                "se": float(joint.se[q]), "ci": [float(lo[q]), float(hi[q])],
                "ci_percentile": [float(plo[q]), float(phi[q])],
                "gap_mean_bp": float(sub.mean()) if sub.size else None,
                "gap_median_bp": float(np.median(sub)) if sub.size else None,
                "turnover_replicas": turn_rep, "turnover_optimum": turn_opt,
                "turnover_excess": None if turn_rep is None or turn_opt is None else turn_rep - turn_opt,
            }))
    return out


# --------------------------------------------------------------------------------------------
# The markets together
# --------------------------------------------------------------------------------------------


def _gap_summary(asm: asmb.Assembly) -> pd.DataFrame:
    """Distribution of the realised gaps of the replicas by arm, operator and level: on all folds where
    they were found (``scope`` ``all``) and on the folds of ``F_b`` (``Fb``)."""
    ops = asm.operators
    rows = []
    for (arm, op, level), grp in ops.gaps.groupby(["arm", "operator", "level"]):
        mask_b = ops.sets[f"{arm}:{op}"]["mask_b"]
        for scope, sub in (("all", grp), ("Fb", grp[grp["fold"].map(lambda f, m=mask_b: bool(m[f]))])):
            g = sub["gap_bp"].to_numpy(dtype=float)
            rows.append({
                "arm": arm, "operator": op, "level": level, "scope": scope, "n_replicas": int(g.size),
                "min": float(g.min()) if g.size else None, "q1": float(np.quantile(g, 0.25)) if g.size else None,
                "median": float(np.median(g)) if g.size else None,
                "q3": float(np.quantile(g, 0.75)) if g.size else None, "max": float(g.max()) if g.size else None,
                "mean": float(g.mean()) if g.size else None,
            })
    return pd.DataFrame(rows)


def analyse_markets(assemblies: dict[str, asmb.Assembly], acfg: dict) -> dict:
    """Every arm of every assembly, the Holm family of the secondary arms of H3a and the descriptive
    tables per market. ``assemblies`` is keyed by ``ff49``, ``us``, ``us_irregular``."""
    arms = {}
    for market, asm in assemblies.items():
        for arm in asm.meta["arms"]:
            res = analyse_arm(asm, arm, market, acfg)
            arms[res["key"]] = res
    notes: list[str] = []
    family = {k: arms[k] for k in HOLM_FAMILY if k in arms}
    holm = None
    if len(family) == len(HOLM_FAMILY):
        adjusted = holm_over_arms({k: _equivalence_from(v["h3a"]) for k, v in family.items()},
                                  alpha=float(stats_config(acfg)["alpha"]))
        holm = {k: d.json_safe(asdict(v)) for k, v in adjusted.items()}
    else:
        missing = [k for k in HOLM_FAMILY if k not in arms]
        notes.append(f"holm: the family of the two secondary arms is incomplete (missing: {missing}); "
                     "no adjusted p-values")
    contrast = []
    if PRIMARY in arms and "ff49-GMV" in arms:
        for key in (PRIMARY, "ff49-GMV"):
            r = arms[key]
            contrast.append({
                "key": key, "h3a_outcome": r["h3a"]["outcome"], "h3a_estimate": r["h3a"]["estimate"],
                "h3b_slope": ((r.get("h3b") or {}).get("log_dose") or {}).get("slope", {}).get("estimate"),
                "h3b_significant": ((r.get("h3b") or {}).get("log_dose") or {}).get("significant"),
            })
    operators = {m: analyse_operators(a, acfg) for m, a in assemblies.items() if a.operators is not None}
    operator_tables = {
        m: {"reach": a.operators.reach.to_dict(orient="records"),
            "gaps": _gap_summary(a).to_dict(orient="records")}
        for m, a in assemblies.items() if a.operators is not None
    }
    return {
        "arms": arms, "primary": PRIMARY if PRIMARY in arms else None, "h3a_holm": holm,
        "operators": operators, "operator_tables": operator_tables,
        "arm_contrast": contrast, "notes": notes,
        "describe": {m: {k: v.to_dict(orient="records") for k, v in describe(a, acfg).items()}
                     for m, a in assemblies.items()},
        "meta": {m: {k: a.meta[k] for k in ("lam", "n_ranks", "n_folds", "n_missing", "fold_sets",
                                           "n_days", "n_days_Fk", "source_run_id", "rows_sha256")
                     if k in a.meta} for m, a in assemblies.items()},
    }


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def _h3a_row(key: str, variant: str, pass_name: str, h: dict) -> dict:
    return {
        "key": key, "variant": variant, "pass": pass_name, "estimate": h["estimate"], "se": h["se"],
        "ci_tost_low": h["ci_tost"][0], "ci_tost_high": h["ci_tost"][1], "ci_low": h["ci"][0],
        "ci_high": h["ci"][1], "pvalue": h["pvalue"], "pvalue_tost": h["pvalue_tost"],
        "equivalent": h["equivalent"], "significant": h["significant"], "outcome": h["outcome"],
        "margin_low": h["margin"][0], "margin_high": h["margin"][1], "block": h["block"],
        "n_boot": h["n_boot"], "n_obs": h["n_obs"], "seed": h["seed"],
    }


def _h3b_row(key: str, variant: str, fit_name: str, fit: dict, leverage) -> dict:
    s = fit["slope"]
    return {
        "key": key, "variant": variant, "fit": fit_name, "regressor": fit["regressor"],
        "n_ranks": len(fit["ranks"]), "slope": s["estimate"], "slope_se": s["se"],
        "slope_ci_pct_low": s["ci_percentile"][0], "slope_ci_pct_high": s["ci_percentile"][1],
        "significant": fit["significant"], "intercept": fit["intercept"]["estimate"],
        "leverage_rank2": leverage, "block": fit["block"], "n_boot": fit["n_boot"], "seed": fit["seed"],
        "functional": fit["functional"],
    }


def _tables(results: dict) -> dict[str, pd.DataFrame]:
    h3a, h3b, per_rank, h3c, sens, net, netd = [], [], [], [], [], [], []
    holm = results.get("h3a_holm") or {}
    for key, r in results["arms"].items():
        row = _h3a_row(key, "main", r["pass"], r["h3a"])
        if key in holm:
            row.update({f"{k}_holm": v for k, v in holm[key].items() if k != "outcome"},
                       outcome_holm=holm[key]["outcome"])
        h3a.append(row)
        sens_r = r.get("sensitivity", {})
        for entry in sens_r.get("fixed_blocks", []):
            h3a.append(_h3a_row(key, f"block{entry['block']}", r["pass"], entry["h3a"]))
        for entry in sens_r.get("delta_sr", []):
            h3a.append(_h3a_row(key, f"delta{entry['delta_sr']:g}", r["pass"], entry["h3a"]))
        for label, entry in sens_r.get("passes", {}).items():
            if "h3a" in entry:
                h3a.append(_h3a_row(key, f"pass:{label}", entry["pass"], entry["h3a"]))
        cov = sens_r.get("coverage")
        if cov and "h3a" in cov:
            h3a.append(_h3a_row(key, "coverage", r["pass"], cov["h3a"]))
        b = r.get("h3b")
        if b:
            leverage = b.get("leverage_rank2")
            for fit_name in ("log_dose", "log_rank", "all_ranks"):
                if fit_name in b:
                    h3b.append(_h3b_row(key, "main", fit_name, b[fit_name], leverage))
            if "log_dose" in b:
                for j, rank in enumerate(b["log_dose"]["ranks"]):
                    pr = b["log_dose"]["per_rank"][j]
                    per_rank.append({"key": key, "rank": rank, "dose_bp": b["log_dose"]["doses_bp"][j],
                                     "delta": pr["estimate"], "se": pr["se"],
                                     "ci_low": pr["ci"][0], "ci_high": pr["ci"][1]})
            for entry in sens_r.get("fixed_blocks", []):
                bb = entry.get("h3b")
                if bb and "log_dose" in bb:
                    h3b.append(_h3b_row(key, f"block{entry['block']}", "log_dose", bb["log_dose"],
                                        bb["leverage_rank2"]))
            for label, entry in sens_r.get("passes", {}).items():
                bb = entry.get("h3b")
                if bb and "log_dose" in bb:
                    h3b.append(_h3b_row(key, f"pass:{label}", "log_dose", bb["log_dose"],
                                        bb["leverage_rank2"]))
        c = r.get("h3c")
        if c:
            for s in c["solvers"]:
                dose = s.get("dose") or {}
                h3c.append({
                    "key": key, "solver": s["name"], "descriptive_h3c": c["descriptive"],
                    "iut_passes": c["iut_passes"], "reason": c["reason"],
                    "dose_bp": dose.get("dose_bp"), "median_gap_bp": dose.get("median_gap_bp"),
                    "inside_range": dose.get("inside_range"),
                    "observed": (s.get("observed") or {}).get("estimate"), "predicted": s.get("predicted"),
                    "residual": (s.get("residual") or {}).get("estimate"),
                    "ci_tost_low": (s.get("ci_tost") or [None, None])[0],
                    "ci_tost_high": (s.get("ci_tost") or [None, None])[1],
                    "equivalent": s.get("equivalent"),
                    "rank_1": s["rank_bins"]["shares"][0], "rank_2_5": s["rank_bins"]["shares"][1],
                    "rank_6_33": s["rank_bins"]["shares"][2], "rank_gt33": s["rank_bins"]["shares"][3],
                })
        if r.get("net"):
            net.extend({"key": key, **row} for row in r["net"]["table"])
            netd.extend({"key": key, **row} for row in r["net"]["differences"])
        sens.append({"key": key, "kind": "hac", **(r.get("hac") or {})} if r.get("hac") else {})
    sens = [s for s in sens if s]
    contrast = pd.DataFrame(results.get("arm_contrast") or [])
    d_all = []
    for market, tables in results["describe"].items():
        d_all.append((market, tables))
    out = {
        "h3a": pd.DataFrame(h3a), "h3b": pd.DataFrame(h3b), "h3b_per_rank": pd.DataFrame(per_rank),
        "h3c": pd.DataFrame(h3c), "net_metrics": pd.DataFrame(net),
        "net_differences": pd.DataFrame(netd), "sensitivity": pd.DataFrame(sens),
        "arm_contrast": contrast,
    }
    reach, gap_rows, level_rows, set_rows = [], [], [], []
    for market, tables in (results.get("operator_tables") or {}).items():
        reach += [{"market": market, **r} for r in tables["reach"]]
        gap_rows += [{"market": market, **r} for r in tables["gaps"]]
    for market, entries in (results.get("operators") or {}).items():
        for entry in entries.values():
            set_rows.append({
                "market": market, "arm": entry["arm"], "operator": entry["operator"],
                "L": " ".join(f"{g:g}" for g in entry["L"]), "L_a": " ".join(f"{g:g}" for g in entry["L_a"]),
                "L_source": entry["L_source"], "pilot_run": entry["pilot_run"],
                "pilot_n_ran": entry["pilot_n_ran"],
                "pilot_share": None if entry["pilot_share"] is None else " ".join(
                    f"{g:g}:{x:.3f}" for g, x in zip(entry["levels_all"], entry["pilot_share"], strict=True)),
                "n_ran": entry["n_ran"], "n_folds_Fa": entry["n_folds_Fa"],
                "n_folds_Fb": entry["n_folds_Fb"], "note": entry["note"],
            })
            for lv in entry["levels"]:
                level_rows.append({
                    "market": market, "arm": entry["arm"], "operator": entry["operator"],
                    **{k: v for k, v in lv.items() if k not in ("per_replica", "ci", "ci_percentile")},
                    "ci_low": lv["ci"][0], "ci_high": lv["ci"][1],
                    "ci_pct_low": lv["ci_percentile"][0], "ci_pct_high": lv["ci_percentile"][1],
                })
    out.update(operator_reach=pd.DataFrame(reach), operator_gaps=pd.DataFrame(gap_rows),
               operator_levels=pd.DataFrame(level_rows), operator_sets=pd.DataFrame(set_rows))
    for name in ("rank2_doses", "doses_by_rank", "exclusions", "regime", "ties"):
        frames = []
        for market, tables in d_all:
            frame = pd.DataFrame(tables[name])
            frame.insert(0, "market", market)
            frames.append(frame)
        out[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def write_tables(results: dict, out_dir: Path | str) -> list[Path]:
    """Write ``analysis.json`` and the CSV tables of the analysis to ``out_dir`` (not ``paper/``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    path = out / "analysis.json"
    path.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    files.append(path)
    for name, frame in _tables(results).items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        files.append(path)
    return files


# --------------------------------------------------------------------------------------------
# The registered stage
# --------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else registry.REPO_ROOT / p


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e3_analysis")
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="exploratory analysis on a dirty tree (not for the paper)")
    args = parser.parse_args(None if argv is None else list(argv))

    acfg = yaml.safe_load(args.analysis_config.read_text(encoding="utf-8"))
    paths = {k: _resolve(v) for k, v in (acfg.get("assemblies") or {}).items()}
    if not paths:
        raise SystemExit("the analysis config has no 'assemblies': the assemble runs to analyse")
    stats_config(acfg)
    inputs = {k: p / "meta.json" for k, p in paths.items()}
    record = {
        "analysis_config": str(args.analysis_config), "assemblies": {k: str(v) for k, v in paths.items()},
        "assembly_meta_sha256": {k: _sha256(p) for k, p in inputs.items()},
        "margins": acfg.get("margins"), "stats": acfg["stats"],
    }
    with registry.start_run("E3", "analysis", record, runs_dir=args.runs_dir,
                            seeds=[int(acfg["stats"]["seed"])], inputs=inputs,
                            require_clean=not args.allow_dirty) as run:
        assemblies = {k: asmb.load_assembly(p) for k, p in paths.items()}
        results = analyse_markets(assemblies, acfg)
        files = write_tables(results, run.run_dir / "tables")
        run.manifest["mode"] = "analysis"
        run.manifest.setdefault("artifacts", []).extend(str(f.relative_to(run.run_dir)) for f in files)
        print(f"{run.run_id}: {len(results['arms'])} arms, {len(files)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())

