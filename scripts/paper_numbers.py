"""Result numbers of the manuscript: ``paper/generated/numbers.tex`` from the registered analyses (CLAUDE.md, golden
rule 1; ``paper/README.md``).

Every key is ``\\expandafter\\def\\csname res@<key>\\endcsname{<value>}``, the text writes ``\\res{<key>}``. Above
each key a comment names its source (the analysis directory or run); every block starts with the run ids that source
read. Keys are lower case, ``[a-z0-9.]``; shares are percent without the sign (the text writes ``\\res{key}\\,\\%``),
times are seconds, ratios and estimates have three significant figures, p-values two.

    python scripts/paper_numbers.py           # writes paper/generated/numbers.tex
    python scripts/paper_numbers.py --check   # exit 1 if the file is not what the sources give
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "generated" / "numbers.tex"

E1_H1A = "results/E1/analysis/20261012-160311-E1-attribution-0dc89f-holm"
E2_H1C = "results/E2/analysis/h1c_20261013"
E2_H2 = "results/E2/analysis/h2_20261017"
E2_SECONDARY = "results/E2/analysis/secondary_20261017"
E2_CURVES = "results/E2/analysis/ttt_curves_20261017"
E2_DELAY = "results/E2/analysis/delay_20261017"
E2_M11 = "results/E2/analysis/m11_20261018"  # optional until the validation M11 is analysed
E3_ANALYSIS = "runs/20261012-225458-E3-analysis-142375"
# post hoc descriptions of the review r1 of the manuscript (ANALYSIS_PLAN §12, 2026-10-19)
E2_H2_SA = "results/E2/analysis/h2_sa_20261019"
E2_H2_TABU = "results/E2/analysis/h2_tabu_20261019"
E2_FIRST = "results/E2/analysis/first_quality_20261019"
E2_PILOT = "configs/E2/pilot"
# post hoc sensitivity of H2 to the exact baseline (ANALYSIS_PLAN §12, 2026-10-19; runs bench_exact_scip_aggr, bench_exact_oa)
E2_EXACT_SENS = {"scipaggr": "results/E2/analysis/h2_exact_scip_aggr_20261019",
                 "oa": "results/E2/analysis/h2_exact_oa_20261019",
                 "base": "results/E2/analysis/h2_exact_base_20261019"}  # the representative on the same inputs

ROLES = {"exact_rep": "exact", "qubo_rep": "qubo", "ls": "ls"}
CELLS = ("C1", "C2", "C3", "C4", "C5")
E3_ARMS = {"ff49-MV": "ff.mv", "ff49-GMV": "ff.gmv", "us-MV": "us.mv", "us_irregular-MV": "usirr.mv"}


# --------------------------------------------------------------------------------------------
# Formats
# --------------------------------------------------------------------------------------------


def sci(x: float, digits: int = 2) -> str:
    """``2.9\\times10^{-5}`` in math mode (two significant figures by default)."""
    mantissa, exponent = f"{x:.{digits - 1}e}".split("e")
    return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"


def sig(x: float, digits: int = 3) -> str:
    """``x`` with ``digits`` significant figures, without an exponent between 1e-4 and 1e6 (``sci`` outside)."""
    if x == 0:
        return "0"
    if not 1e-4 <= abs(x) < 1e6:
        return sci(x, digits)
    decimals = max(0, digits - 1 - math.floor(math.log10(abs(x))))
    text = f"{round(x, decimals):.{decimals}f}"
    return rf"\ensuremath{{-}}{text[1:]}" if text.startswith("-") else text


def gap(x: float, digits: int = 2) -> str:
    """A gap that is zero up to the arithmetic of the solvers (below 1e-9 in units of its scale) prints as 0."""
    return "0" if abs(x) < 1e-9 else sig(x, digits)


def median_count(x: float) -> str:
    """A median of whole numbers: the integer when it is one (6, not 6.0), else three significant figures."""
    return str(int(x)) if float(x).is_integer() else sig(x)


def pval(p: float) -> str:
    """A p-value: two significant figures, in scientific notation below 0.001."""
    return sci(p) if p < 1e-3 else sig(p, 2)


def pct(share: float, decimals: int = 0) -> str:
    """A share in percent, without the sign."""
    return f"{100 * share:.{decimals}f}"


def count(n: float) -> str:
    """An integer, with a thin space between the thousands from 10 000 on."""
    n = round(n)
    text = f"{n:,}".replace(",", r"\,")
    return text if abs(n) >= 10_000 else str(n)


def bp(x: float, digits: int = 3) -> str:
    """A gap or a difference in basis points (``x`` as a fraction)."""
    return sig(1e4 * x, digits)


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _csv(path: str) -> list[dict]:
    with (ROOT / path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _runs(inputs: Iterable[Mapping]) -> list[str]:
    return sorted({entry["run_id"] for entry in inputs if isinstance(entry, Mapping) and entry.get("run_id")})


class Numbers:
    """The keys of one generation in the order they are added, each with its source."""

    def __init__(self) -> None:
        self.blocks: list[tuple[str, list[str], list[tuple[str, str]]]] = []

    def block(self, source: str, runs: Iterable[str]) -> Callable[[str, str], None]:
        entries: list[tuple[str, str]] = []
        self.blocks.append((source, sorted(runs), entries))

        def add(key: str, value: str) -> None:
            entries.append((key, value))

        return add

    def keys(self) -> list[str]:
        return [key for _, _, entries in self.blocks for key, _ in entries]

    def text(self) -> str:
        lines = ["% paper/generated/numbers.tex — generated by scripts/paper_numbers.py; do not edit by hand.",
                 "% Every key: \\res{<key>} in the text; the comment above it names its source.", ""]
        for source, runs, entries in self.blocks:
            lines.append(f"% ==== {source}")
            lines += [f"%      run {run}" for run in runs] or ["%      (no run ids recorded in the source)"]
            for key, value in entries:
                lines.append(f"% src: {source}")
                lines.append(rf"\expandafter\def\csname res@{key}\endcsname{{{value}}}")
            lines.append("")
        return "\n".join(lines)


def _interval(add, key: str, estimate: float, low: float, high: float, fmt=sig) -> None:
    add(key, fmt(estimate))
    add(f"{key}.ci.low", fmt(low))
    add(f"{key}.ci.high", fmt(high))


# --------------------------------------------------------------------------------------------
# E1 and H1c
# --------------------------------------------------------------------------------------------


def e1(numbers: Numbers) -> None:
    h1a = _json(f"{E1_H1A}/h1a.json")
    manifest = _json(f"{E1_H1A}/analysis_manifest.json")
    run = manifest.get("run")
    add = numbers.block(f"{E1_H1A}/h1a.json", [run] if isinstance(run, str) else _runs([run] if run else []))
    add("e1.h1a.p", pval(h1a["p_value"]))
    add("e1.h1a.p.holm", pval(h1a["holm"]["adjusted"]["H1a"]))
    add("e1.h1c.p.holm", pval(h1a["holm"]["adjusted"]["H1c"]))
    add("e1.h1a.npairs", count(h1a["n_pairs"]))
    add("e1.h1a.nzero", count(h1a["n_zero"]))
    add("e1.h1a.npos", count(sum(s["n_pos"] for s in h1a["strata"].values())))
    add("e1.h1a.nneg", count(sum(s["n_neg"] for s in h1a["strata"].values())))
    mean = h1a["difference"]["mean"]
    _interval(add, "e1.h1a.diff.mean", mean["estimate"], *mean["ci"])  # in units of the normalised gap ĝ, as τ
    add("e1.h1a.diff.level.pct", pct(h1a["difference"]["level"]))
    add("e1.h1a.m0", count(h1a["m0"]))

    # the estimates of RQ1 by the level h of the dispersion of volatilities (§2: shares with Wilson intervals, means)
    add = numbers.block(E1_H1A, [run] if isinstance(run, str) else _runs([run] if run else []))
    level = {"0.15": "h015", "0.45": "h045"}
    for row in _csv(f"{E1_H1A}/shares_e_form.csv"):
        if row["group"] == "h":
            key = f"e1.eform.m{row['m']}.{level[row['h']]}"
            add(f"{key}.nexceed", count(float(row["n_exceed"])))
            add(f"{key}.ninstances", count(float(row["n_instances"])))
            add(f"{key}.pct", pct(float(row["share"])))
            add(f"{key}.ci.low.pct", pct(float(row["ci_low"])))
            add(f"{key}.ci.high.pct", pct(float(row["ci_high"])))
    for row in _csv(f"{E1_H1A}/e_form.csv"):
        if row["group"] == "h":
            key = f"e1.eform.m{row['m']}.{level[row['h']]}.mean"
            _interval(add, key, float(row["mean"]), float(row["mean_ci_low"]), float(row["mean_ci_high"]))

    h1c = _json(f"{E2_H1C}/h1c.json")
    add = numbers.block(f"{E2_H1C}/h1c.json", [Path(e["source"]).parent.name for e in h1c["inputs"]
                                               if str(e.get("source", "")).startswith("runs/")])
    add("e2.h1c.p", pval(h1c["p_value"]))
    add("e2.h1c.n", count(h1c["n_instances"]))
    add("e2.h1c.tabubetter", count(h1c["direction"]["tabu_better"]))
    add("e2.h1c.sabetter", count(h1c["direction"]["sa_better"]))
    add("e2.h1c.zero", count(h1c["direction"]["zero"]))
    add("e2.h1c.median.diff", sig(h1c["median_normalised_difference"]))  # ĝ_tabu − ĝ_SA


# --------------------------------------------------------------------------------------------
# E2: H2, H2b, A3 and the secondary analyses
# --------------------------------------------------------------------------------------------

_H = {"H2": "h2", "H2b": "h2b", "TTT": "ttt", "TTTb": "tttb"}
_VARIANTS = {"primary": "", "full": ".full", "first_improvement": ".firstimpr", "within_budget": ".withinbudget",
             "own": ".own", "without_C5": ".noc5", "tau_1e-4": ".tausens", "budget_30": ".budget30",
             "budget_10": ".budget10"}


def _pooled_rows(add, rows: Iterable[Mapping]) -> None:
    for row in rows:
        key = f"e2.{_H[row['hypothesis']]}{_VARIANTS[row['variant']]}"
        _interval(add, key, float(row["ratio"]), float(row["ratio_low"]), float(row["ratio_high"]))


def e2(numbers: Numbers) -> None:
    report = _json(f"{E2_H2}/h2.json")
    add = numbers.block(E2_H2, _runs(report["inputs"]))
    # the levels of the intervals, as the report records them (pooled; simultaneous over the cells of the family)
    add("e2.level.pct", pct(report["settings"]["level"]))
    add("e2.celllevel.pct", pct(report["settings"]["cell_level"], 2))
    add("e2.a3.level.pct", "95")  # the field ci95 of the A3 estimates: Wilson intervals at 0.95 (e2_analysis.a3_estimates)
    _pooled_rows(add, _csv(f"{E2_H2}/pooled.csv"))
    for row in _csv(f"{E2_H2}/cells.csv"):
        if row["variant"] == "primary" and row["cell"] in CELLS and row["ratio"]:
            _interval(add, f"e2.{_H[row['hypothesis']]}.{row['cell'].lower()}", float(row["ratio"]),
                      float(row["ratio_low"]), float(row["ratio_high"]))
    add("e2.n.instances", count(report["references"]["n_instances"]))
    for cell in CELLS:
        add(f"e2.n.{cell.lower()}", count(report["cells"][cell]["n_instances"]))
    add("e2.n.uncertified", count(report["references"]["by_type"].get("best_known", 0)))
    c5 = report["cells"]["C5"]
    add("e2.c5.n", count(c5["n_instances"]))
    add("e2.c5.nports", count(c5["n_clusters"]))
    add("e2.c5.uncertified", count(c5["reference"]["by_type"].get("best_known", 0)))
    for cell, part in report["a3"]["cells"].items():
        if cell not in CELLS and cell not in ("D1_100", "D1_200"):
            continue
        c = {"D1_100": "d1.n100", "D1_200": "d1.n200"}.get(cell, cell.lower())
        add(f"e2.a3.{c}.ntrivial", count(part["n_trivial"]))
        add(f"e2.a3.{c}.pct", pct(part["share_trivial"]))
        add(f"e2.a3.{c}.ci.low.pct", pct(part["ci95"][0]))
        add(f"e2.a3.{c}.ci.high.pct", pct(part["ci95"][1]))
        add(f"e2.a3.{c}.nscanned", count(part["n_scanned"]))
    # the factor cells C1–C3 as one range (abstract, introduction)
    factor = [report["a3"]["cells"][c]["share_trivial"] for c in ("C1", "C2", "C3")]
    add("e2.a3.factor.min.pct", pct(min(factor)))
    add("e2.a3.factor.max.pct", pct(max(factor)))

    secondary =_json(f"{E2_SECONDARY}/secondary.json")
    add = numbers.block(E2_SECONDARY, _runs(secondary["inputs"]))
    _pooled_rows(add, _csv(f"{E2_SECONDARY}/ttt_pooled.csv"))
    for row in _csv(f"{E2_SECONDARY}/exact_times.csv"):
        c = row["cell"].lower()
        if row["cell"] not in CELLS:
            continue
        add(f"e2.exact.{c}.tprimal.median", sig(float(row["t_primal_median"]), 2))
        add(f"e2.exact.{c}.cert.pct", pct(float(row["share_in_budget"])))
    for row in _csv(f"{E2_SECONDARY}/profiles.csv"):
        if float(row["ratio"]) == 1.0 and row["cell"] in CELLS:
            measure = "pi" if row["measure"] == "PI_plus" else "ttt"
            add(f"e2.profile.{measure}.{row['cell'].lower()}.{row['role']}.pct", pct(float(row["rho"])))

    curves = _json(f"{E2_CURVES}/ttt_curves.json")
    add = numbers.block(E2_CURVES, _runs(curves["inputs"]))
    for m in curves["methods"]:
        seed = m["instance_id"].rsplit("/", 1)[1]
        tau = "" if m["tau"] == 1e-3 else ".tausens"
        key = f"e2.curves.{seed}.{ROLES[m['role']]}{tau}"
        add(f"{key}.pct", pct(m["share_reached"]))
        if m["median_s"] is not None:
            add(f"{key}.median", sig(m["median_s"], 2))
    for p in curves["pairs"]:
        seed = p["instance_id"].rsplit("/", 1)[1]
        tau = "" if p["tau"] == 1e-3 else ".tausens"
        add(f"e2.curves.{seed}.{ROLES[p['a']]}.{ROLES[p['b']]}{tau}.pfaster", sig(p["p_a_faster"], 2))

    delay = _json(f"{E2_DELAY}/delay.json")
    add = numbers.block(E2_DELAY, _runs(delay["inputs"]))
    for c in delay["cells"]:
        key = f"e2.delay.{c['cell'].lower()}.{ROLES[c['role']]}"
        add(f"{key}.median", sig(c["median_s"], 2))
        add(f"{key}.max", sig(c["max_s"], 2))
    add("e2.delay.ncensored", count(sum(c["n_censored"] for c in delay["cells"])))

    if (ROOT / E2_M11 / "m11.json").exists():
        m11 = _json(f"{E2_M11}/m11.json")
        add = numbers.block(E2_M11, _runs(m11["inputs"]))
        for table, block in m11["tables"].items():
            for view, summary in block["summary"].items():
                key = f"e2.m11.{table.lower()}.{view}"
                if summary["median_ratio"] is not None:
                    add(f"{key}.median", sig(summary["median_ratio"], 2))
                add(f"{key}.nours", count(summary["n_ours_solved"]))
                add(f"{key}.npub", count(summary["n_pub_solved"]))
                add(f"{key}.ncells", count(summary["n_cells"]))
                add(f"{key}.nexcluded", count(summary["n_excluded"]))
        # the instances of a table on which SCIP was faster than the best published method (a ratio below 1, bounds
        # included: an upper bound below 1 is faster still)
        for table, block in m11["tables"].items():
            faster = [c for c in block["cells"] if c["views"]["scip"]["ratio"] is not None and c["views"]["scip"]["ratio"] < 1]
            add(f"e2.m11.{table.lower()}.scip.nfaster", count(len(faster)))
            if faster:
                add(f"e2.m11.{table.lower()}.scip.faster.cells",
                    "; ".join(f"port~{c['port']}, $k = {c['k']}$" for c in faster))


# --------------------------------------------------------------------------------------------
# E3
# --------------------------------------------------------------------------------------------


def e3(numbers: Numbers) -> None:
    analysis = _json(f"{E3_ANALYSIS}/tables/analysis.json")
    runs = [E3_ANALYSIS.split("/")[-1], *(m["source_run_id"] for m in analysis["meta"].values())]
    add = numbers.block(f"{E3_ANALYSIS}/tables/analysis.json", runs)
    primary = analysis["arms"]["ff49-MV"]
    add("e3.h3a.level.pct", pct(1 - 2 * primary["h3a"]["alpha"]))  # TOST at α: the (1 − 2α) interval
    add("e3.h3b.level.pct", pct(1 - primary["h3b"]["log_dose"]["alpha"]))
    for arm, prefix in E3_ARMS.items():
        part = analysis["arms"][arm]
        h3a = part["h3a"]
        _interval(add, f"e3.{prefix}.h3a", h3a["estimate"], *h3a["ci_tost"])
        add(f"e3.{prefix}.h3a.p.tost", pval(h3a["pvalue_tost"]))
        add(f"e3.{prefix}.margin.low", sig(h3a["margin"][0]))
        add(f"e3.{prefix}.margin.high", sig(h3a["margin"][1]))
        holm = analysis.get("h3a_holm", {}).get(arm)
        if holm:
            add(f"e3.{prefix}.h3a.p.tost.holm", pval(holm["pvalue_tost_holm"]))
        h3b = part.get("h3b")
        if h3b and not h3b.get("descriptive"):
            slope = h3b["log_dose"]["slope"]
            # the 95 % percentile interval of the joint block bootstrap is the plan's (§2 H3b), not the normal one
            _interval(add, f"e3.{prefix}.h3b.slope", slope["estimate"], *slope["ci_percentile"])
            add(f"e3.{prefix}.h3b.p", pval(slope["pvalue"]))
            if part["functional"] == "logvar" and h3b["log_dose"]["significant"]:
                # §10 (H3b significant): what a doubling of the in-sample dose costs out of sample, here the change of
                # the out-of-sample variance in percent, exp(b·ln 2) − 1, with the interval of b mapped the same way
                doubling = [100 * (math.exp(b * math.log(2)) - 1) for b in (slope["estimate"], *slope["ci_percentile"])]
                _interval(add, f"e3.{prefix}.h3b.doubling.pct", *doubling, fmt=lambda x: sig(x, 2))
            add(f"e3.{prefix}.leverage", sig(h3b["leverage_rank2"], 2))
        add(f"e3.{prefix}.nfolds", count(part["n_folds_Fk"]))
        add(f"e3.{prefix}.years", sig(part["years"], 3))
    h3c = analysis["arms"]["ff49-MV"].get("h3c")
    if h3c and h3c.get("descriptive"):
        add("e3.ff.mv.h3c.nmaterial", count(h3c["n_material"]))
        add("e3.ff.mv.h3c.nsolvers", count(len(h3c["solvers"])))
    for market, short in (("ff49", "ff"), ("us", "us")):
        (regime,) = analysis["describe"][market]["regime"]
        add(f"e3.{short}.binding.pct", pct(regime["share_binding"]))
        add(f"e3.{short}.k.median", median_count(regime["K_median"]))
        add(f"e3.{short}.nassets.median", median_count(regime["n_assets_median"]))


# --------------------------------------------------------------------------------------------
# Review r1: planned outputs the first draft left out, and the post hoc descriptions of §12, 2026-10-19
# --------------------------------------------------------------------------------------------

#: The reference configurations of RQ1 (``role = reference`` in the E1 tables) and the exploratory hot swap-SA.
E1_SAMPLERS = {"sa/p100/default": "sa", "tabu/p100": "tabu", "swap_sa/c0.01": "swap", "swap_sa/c0.2": "swaphot"}
#: The cells of RQ2 and the dense cells, as they are named in the keys.
CELL_KEYS = {"C1": "c1", "C2": "c2", "C3": "c3", "C4": "c4", "C5": "c5", "D1_100": "d1.n100", "D1_200": "d1.n200"}


def review_r1(numbers: Numbers) -> None:
    # RQ1 (§5: E_samp is a primary metric of RQ1): the samplers of fixed work, their pools and residuals
    manifest = _json(f"{E1_H1A}/analysis_manifest.json")
    run = manifest.get("run")
    add = numbers.block(E1_H1A, [run] if isinstance(run, str) else _runs([run] if run else []))
    for row in _csv(f"{E1_H1A}/e_samp.csv"):
        if row["group"] != "all" or row["config_id"] not in E1_SAMPLERS:
            continue
        name, m = E1_SAMPLERS[row["config_id"]], int(row["m"])
        if m == 1:
            add(f"e1.samp.{name}.m1.mean", gap(float(row["e_samp_mean"])))
            add(f"e1.samp.{name}.m1.within.pct", pct(float(row["share_samp_within_tau"])))
        if m == 200:
            n = int(row["n_instances"])
            add(f"e1.pipe.{name}.m200.nabove", count(round(float(row["share_pipe_above_tau"]) * n)))
            add(f"e1.pipe.{name}.m200.above.pct", pct(float(row["share_pipe_above_tau"])))
            if name == "sa":
                add("e1.pipe.ninstances", count(n))
    for row in _csv(f"{E1_H1A}/sampler_diagnostics.csv"):
        if row["group"] == "all" and row["config_id"] in E1_SAMPLERS:
            name = E1_SAMPLERS[row["config_id"]]
            add(f"e1.pool.{name}.unique", median_count(float(row["pool_n_unique_median"])))
            add(f"e1.samp.{name}.call.s", sig(float(row["solve_s_median"]), 2))
    for row in _csv(f"{E1_H1A}/shares_e_form.csv"):
        if row["group"] == "all" and row["m"] == "50":
            add("e1.eform.m50.all.nexceed", count(float(row["n_exceed"])))

    # A3 as the screening recorded it (§12 A3: ĝ_start in bp CE and t_cert on the trivial instances), the dense cells
    report = _json(f"{E2_H2}/h2.json")
    add = numbers.block(f"{E2_H2}/h2.json", _runs(report["inputs"]))
    for cell, part in report["a3"]["cells"].items():
        c = CELL_KEYS.get(cell)
        if c is None:
            continue
        add(f"e2.a3.{c}.nnontrivial", count(part["n_nontrivial"]))
        add(f"e2.a3.{c}.nundecided", count(part["n_undecided"]))
        add(f"e2.a3.{c}.nnonbinding", count(part["n_nonbinding"]))
        gstart = part.get("g_start_bp")
        if gstart:
            add(f"e2.a3.{c}.gstart.median.bp", gap(gstart["median"]))
            add(f"e2.a3.{c}.gstart.max.bp", sig(gstart["max"], 2))
        tcert = part.get("t_cert_trivial_s")
        if tcert:
            add(f"e2.a3.{c}.tcert.median", sig(tcert["median"], 2))
            add(f"e2.a3.{c}.tcert.max", count(round(tcert["max"])))
    by_cell = {}
    for inst in report["instances"]:
        by_cell.setdefault(inst["cell"], []).append(inst)
    add("e2.n.d1", count(len(by_cell.get("D1_100", [])) + len(by_cell.get("D1_200", []))))
    # τ in bp of the annual certainty equivalent: ĝ = τ is a gap of τ·s_ref, i.e. 1e4·τ·s_ref bp (median over C1-C3)
    scales = [i["s_ref"] for c in ("C1", "C2", "C3") for i in by_cell.get(c, [])]
    add("e2.tau.bp.factor", sig(float(np.median(scales)) * 1e4 * 1e-3, 2))
    add("e2.d1.uncertified", count(sum(i["ref_type"] == "best_known" for c in ("D1_100", "D1_200")
                                       for i in by_cell.get(c, []))))

    # the distribution over the instances (§12, 2026-10-19 (1)): medians, shares below 1, maxima; the time t0
    add = numbers.block(f"{E2_H2}/instances.csv", _runs(report["inputs"]))
    for cell, insts in by_cell.items():
        c = CELL_KEYS.get(cell)
        if c is None:
            continue
        r = np.array([i["r"] for i in insts], dtype=float)
        rb = np.array([i["r_b"] for i in insts], dtype=float)
        add(f"e2.inst.{c}.r.median", sig(float(np.median(r)), 2))
        add(f"e2.inst.{c}.r.below.pct", pct(float(np.mean(r < 1))))
        add(f"e2.inst.{c}.rb.median", sig(float(np.median(rb)), 2))
        add(f"e2.inst.{c}.rb.below.pct", pct(float(np.mean(rb < 1))))
        add(f"e2.inst.{c}.rb.max", median_count(float(np.max(rb))) if np.max(rb) >= 100 else sig(float(np.max(rb)), 2))
        add(f"e2.t0.{c}.median", sig(float(np.median([i["t0_s"] for i in insts])), 2))
    for row in _csv(f"{E2_H2}/cells.csv"):
        if row["variant"] == "primary" and row["cell"] in ("D1_100", "D1_200") and row["ratio"]:
            _interval(add, f"e2.{_H[row['hypothesis']]}.{CELL_KEYS[row['cell']]}", float(row["ratio"]),
                      float(row["ratio_low"]), float(row["ratio_high"]))

    # the runs that never reached the target (secondary, ttt_censoring: the Kaplan-Meier median beyond the budget)
    secondary = _json(f"{E2_SECONDARY}/secondary.json")
    add = numbers.block(f"{E2_SECONDARY}/ttt_censoring.csv", _runs(secondary["inputs"]))
    for row in _csv(f"{E2_SECONDARY}/ttt_censoring.csv"):
        if row["variant"] == "primary" and row["cell"] in CELL_KEYS:
            c = CELL_KEYS[row["cell"]]
            add(f"e2.censor.{c}.n", count(float(row["n_instances"])))
            for role in ("exact", "qubo", "ls"):
                add(f"e2.censor.{c}.{role}", count(float(row[f"censored_{role}"])))

    # post hoc (§12, 2026-10-19 (2)): the penalty samplers of the same runs in the place of the representative
    for name, source in (("sa", E2_H2_SA), ("tabu", E2_H2_TABU)):
        post = _json(f"{source}/h2.json")
        add = numbers.block(f"{source}/h2.json", _runs(post["inputs"]))
        for hypothesis in ("H2", "H2b"):
            part = post["pooled"][hypothesis]
            _interval(add, f"e2.{name}.{_H[hypothesis]}", part["ratio"], *part["ci_ratio"])

    # post hoc (§12, 2026-10-19): the exact class represented by another configuration (when the analyses exist)
    for name, source in E2_EXACT_SENS.items():
        if not (ROOT / source / "h2.json").exists():
            continue
        post = _json(f"{source}/h2.json")
        add = numbers.block(f"{source}/h2.json", _runs(post["inputs"]))
        for hypothesis in ("H2", "H2b"):
            part = post["pooled"][hypothesis]
            _interval(add, f"e2.exactsens.{name}.{_H[hypothesis]}", part["ratio"], *part["ci_ratio"])
        for row in _csv(f"{source}/cells.csv"):
            if row["variant"] == "primary" and row["cell"] in CELLS and row["hypothesis"] == "H2" and row["ratio"]:
                add(f"e2.exactsens.{name}.h2.{row['cell'].lower()}", sig(float(row["ratio"])))
        if name != "base" and (ROOT / E2_EXACT_SENS["base"] / "instances.csv").exists():
            # per cell, the median over the instances of the integral of this configuration over that of the
            # representative on the same instance and references (above 1: it improved its incumbent more slowly)
            base = {r["instance_id"]: r for r in _csv(f"{E2_EXACT_SENS['base']}/instances.csv")}
            by_cell: dict[str, list[float]] = {}
            for r in _csv(f"{source}/instances.csv"):
                if r["cell"] in CELLS:
                    by_cell.setdefault(r["cell"], []).append(float(r["pi_exact"]) / float(base[r["instance_id"]]["pi_exact"]))
            for cell, ratios in by_cell.items():
                add(f"e2.exactsens.{name}.pirel.{cell.lower()}", sig(float(np.median(ratios)), 2))

    # post hoc (§12, 2026-10-19 (3)): the first own portfolio in the cold runs against the common start
    first = _json(f"{E2_FIRST}/first_quality.json")
    add = numbers.block(f"{E2_FIRST}/first_quality.json", _runs(first["inputs"]))
    for c in first["cells"]:
        key = f"e2.first.{CELL_KEYS[c['cell']]}.{ROLES[c['role']]}"
        add(f"{key}.notworse.pct", pct(c["share_not_worse"]))
        if c["median_g_first_over_tau"] is not None:
            add(f"{key}.median.tau", gap(c["median_g_first_over_tau"]))
    add("e2.first.notlater.max.pct", pct(max(c["share_not_later"] for c in first["cells"])))

    # the tuning of the exact class: its whole space against 30 random draws, and OA against SCIP at N = 200
    screen = _json(f"{E2_PILOT}/screen.json")
    add = numbers.block(f"{E2_PILOT}/screen.json", _runs(screen.get("inputs") or []))
    add("e2.tune.exact.n100.nconfigs", count(screen["sizes"]["100"]["groups"]["exact/pi"]["n_configs"]))
    add("e2.tune.exact.n200.nconfigs", count(screen["sizes"]["200"]["groups"]["exact/pi"]["n_configs"]))
    add("e2.tune.heur.nconfigs", count(screen["sizes"]["100"]["groups"]["swap_sa/pi"]["n_configs"]))
    tuned = _json(f"{E2_PILOT}/tuned.json")
    add = numbers.block(f"{E2_PILOT}/tuned.json", _runs(tuned.get("inputs") or []))
    decision = tuned["sizes"]["200"]["decisions"]["exact/pi"]
    for method, short in (("oa-highs", "oa"), ("scip-persp-mineig-sym0", "scip")):
        add(f"e2.tune.n200.{short}.screen", sig(decision["criterion_screen"][method], 2))
        add(f"e2.tune.n200.{short}.long", sig(decision["criterion_long"][method], 2))

    # M11: the absolute times behind the ratios
    if (ROOT / E2_M11).exists():
        m11 = _json(f"{E2_M11}/m11.json")
        add = numbers.block(f"{E2_M11}/cells.csv", _runs(m11["inputs"]))
        rows = _csv(f"{E2_M11}/cells.csv")
        for table in ("T3", "T4"):
            mine = [r for r in rows if r["table"] == table]
            add(f"e2.m11.{table.lower()}.scip.time.median", sig(float(np.median([float(r["scip_time_s"]) for r in mine])), 2))
            pub = [float(r["published_best_s"]) for r in mine]
            add(f"e2.m11.{table.lower()}.pub.time.median", sig(float(np.median(pub)), 2))
            add(f"e2.m11.{table.lower()}.pub.time.max", sig(float(np.max(pub)), 2))
            oa = [float(r["oa_time_s"]) for r in mine if r["oa_time_s"]]
            if oa:
                add(f"e2.m11.{table.lower()}.oa.time.median", sig(float(np.median(oa)), 2))

    # RQ3 (§10: the doses of rank 2 with H3a; §3: the regime K·u ≈ 1; the sensitivities and the net paths)
    analysis = _json(f"{E3_ANALYSIS}/tables/analysis.json")
    runs = [E3_ANALYSIS.split("/")[-1], *(m["source_run_id"] for m in analysis["meta"].values())]
    add = numbers.block(f"{E3_ANALYSIS}/tables/analysis.json", runs)
    for market, short in (("ff49", "ff"), ("us", "us")):
        for dose in analysis["describe"][market]["rank2_doses"]:
            if dose["stratum"] == "all":
                key = f"e3.{short}.{dose['arm'].lower()}.dose2"
                for q in ("median", "q1", "q3"):
                    add(f"{key}.{q}", sig(dose[q], 2))
        (regime,) = analysis["describe"][market]["regime"]
        add(f"e3.{short}.regime.lb.pct", pct(regime["share_lower_bound"]))
    for arm, prefix in E3_ARMS.items():
        part = analysis["arms"][arm]
        add(f"e3.{prefix}.block", count(part["block"]["block"]))
        sens = part.get("sensitivity") or {}
        for entry in sens.get("delta_sr") or ():
            if entry["delta_sr"] == 0.05:
                add(f"e3.{prefix}.sens.dsr005.p.tost", pval(entry["h3a"]["pvalue_tost"]))
        lag = (sens.get("passes") or {}).get("lag1") or {}
        if lag.get("h3a"):
            add(f"e3.{prefix}.sens.lag1.p.tost", pval(lag["h3a"]["pvalue_tost"]))
        if sens.get("coverage"):
            add(f"e3.{prefix}.sens.coverage.nfolds", count(sens["coverage"]["n_folds"]))
            add(f"e3.{prefix}.sens.coverage.p.tost", pval(sens["coverage"]["h3a"]["pvalue_tost"]))
    # the pointwise interval of the path of rank 5, from the table that Figure 4 draws
    for row in _csv(f"{E3_ANALYSIS}/tables/h3b_per_rank.csv"):
        if row["key"] == "ff49-MV" and row["rank"] == "5":
            _interval(add, "e3.ff.mv.rank5", float(row["delta"]), float(row["ci_low"]), float(row["ci_high"]))
            break
    add = numbers.block(f"{E3_ANALYSIS}/tables/net_differences.csv", runs)
    names = {"ff49-MV": "ff.mv", "ff49-GMV": "ff.gmv", "us-MV": "us.mv"}
    for row in _csv(f"{E3_ANALYSIS}/tables/net_differences.csv"):
        if row["path"] != "rank_2" or row["key"] not in names or row["cost"] == "cost_0":
            continue
        cost = row["cost"].split("_")[1]
        key = f"e3.{names[row['key']]}.net{cost}"
        _interval(add, key, float(row["sharpe_diff"]), float(row["sharpe_diff_lo"]), float(row["sharpe_diff_hi"]))
        add(f"{key}.costdiff.bp", sig(float(row["cost_diff_bps"]), 2))
        if cost == "10":  # the turnover of the path does not depend on the cost level
            add(f"e3.{names[row['key']]}.turnover2", sig(float(row["turnover_diff"]), 2))


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def generate() -> Numbers:
    numbers = Numbers()
    e1(numbers)
    e2(numbers)
    e3(numbers)
    review_r1(numbers)
    keys = numbers.keys()
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys: {duplicates}")
    return numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with paper/generated/numbers.tex, write nothing")
    args = parser.parse_args(argv)
    text = generate().text() + "\n"
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is not what the sources give: rerun scripts/paper_numbers.py")
            return 1
        print(f"{OUT.relative_to(ROOT)}: up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)}: {len(generate().keys())} keys")
    return 0


if __name__ == "__main__":
    sys.exit(main())
