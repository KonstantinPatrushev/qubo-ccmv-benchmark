"""TTT curves of the stage ``ttt`` and the delay to the first own solution on the stage ``cold``: descriptive secondary
analyses of RQ2 (ANALYSIS_PLAN §5, §6; the entries of §12 of 2026-10-17, recorded before the rows were read):
``python -m cif.experiments.e2_analysis ttt-curves`` and ``... delay``.

**Both are secondary, descriptive and not confirmatory**: no test, no interval, no pooling over instances; every
output says so (``analysis = secondary``, ``confirmatory = false``, a notice, a column ``analysis`` in every table).

**TTT curves** (§6, Aiex, Resende & Ribeiro 2007). The rows are those of the registered run of the stage ``ttt``: the
three representatives of the classes of RQ2 (roles ``exact_rep``, ``qubo_rep``, ``ls``) with 100 solver seeds each on
three calibration instances of N = 200, from the common start, 60 s. The time of a run is the end-to-end time (the
build of the common start included) of the first moment its incumbent has ``ĝ ≤ τ`` against the reference of the
instance (:func:`cif.experiments.e2_pilot.row_ttt`), the certified optimum of the calibration screening; τ = 1e-3, and
1e-4 as the sensitivity. A run that has not got there by the budget is censored at it and stays in the denominator: the
curve is the empirical distribution ``P(TTT ≤ t)`` over all the runs of a method, with the points
``(t_(i), (i − ½)/n)`` over the runs that got there, so it ends below 1 when some did not. Per method and instance:
the share of runs at the target and the Kaplan-Meier quartiles of the time (``None`` beyond the budget); per pair of
methods, ``P(TTT_A < TTT_B)`` over all pairs of their runs (:func:`p_faster`). An instance whose common start already
meets the target (every time is then ``t₀``) is reported as degenerate.

**Delay to the first own solution** (§5). The rows are those of the registered run of the stage ``cold``: no common
start (in the warm runs the common start is everyone's first solution), one solver seed per method and instance, the
non-trivial test instances of C1-C5. The delay of a run is the time of the first point of its own trajectory: the
first solution of SCIP's event handler, the first master iteration of OA with a refinement, the end of the first
batch of the QUBO pipeline, the first portfolio local search holds. The row field ``first_own_s`` is not read: in the
cold run of 2026-10-13 it is the first refinement of a partial subset for local search (audit G2, C-4). A run without
an own solution within the budget is censored at it. The value of an instance is the Kaplan-Meier median over its
seeds (the delay of the one run of the cold stage); per cell and method the median, quartiles and maximum over the
instances (a censored instance enters at the budget) and the share of censored instances.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cif.experiments import e2_analysis as an
from cif.experiments import e2_eligibility, registry
from cif.metrics import anytime

__all__ = [
    "ANALYSIS",
    "PAIRS",
    "PRESPECIFIED_DELAY",
    "PRESPECIFIED_TTT",
    "ROLES",
    "TAUS",
    "delay_report",
    "first_quality_report",
    "p_faster",
    "run_command",
    "ttt_curves_report",
    "write_delay",
    "write_first_quality",
    "write_ttt_curves",
]

#: The label every table and the JSON carry.
ANALYSIS = "secondary"
#: The budget of the stages ``ttt`` and ``cold`` (§6).
BUDGET_S = an.BUDGET_S
#: The main target and its sensitivity (§5; §12 of 2026-10-17 (2)).
TAUS = (anytime.TAU_MAIN, anytime.TAU_SENSITIVITY)
#: The representatives of the three classes (:data:`cif.experiments.e2_analysis.REPRESENTATIVES`).
ROLES = ("exact_rep", "qubo_rep", "ls")
#: The pairs ``(A, B)`` of ``P(TTT_A < TTT_B)``: QUBO against the exact class and against local search (the
#: orientation of H2 and H2b), and local search against the exact class.
PAIRS = (("qubo_rep", "exact_rep"), ("qubo_rep", "ls"), ("ls", "exact_rep"))
#: The Kaplan-Meier quartiles of the time to the target.
QUARTILES = {"q1_s": 0.25, "median_s": 0.5, "q3_s": 0.75}
NOTICE = "SECONDARY, descriptive: not a confirmatory analysis (ANALYSIS_PLAN §7.1 lists only H2 and H2b for RQ2)"

PRESPECIFIED_TTT = {
    "rows": {"choice": "The registered run of the stage ttt (20261013-121926-E2-ttt-92bd5f): the three representatives, "
                       "100 solver seeds each, three calibration instances of N = 200 (seeds 1014, 1043, 1049), the "
                       "common start, 60 s.",
             "why": "§12 of 2026-10-17, TTT curves (1); §6 (minor 10)."},
    "time": {"choice": "End-to-end time (t0 of the common start included) of the first ĝ <= τ on the trajectory, "
                       "τ = 1e-3 and 1e-4 (sensitivity), against the reference of the instance: the certified optimum "
                       "of the calibration screening (reference_table over the certificates and the rows).",
             "why": "§12 of 2026-10-17, TTT curves (2); §5."},
    "curve": {"choice": "A run that misses the target is censored at the budget and stays in the denominator: the points "
                        "are (t_(i), (i - 1/2)/n) over the runs that got there, n all the runs of the method.",
              "why": "§12 of 2026-10-17, TTT curves (3); Aiex, Resende & Ribeiro (2007)."},
    "summaries": {"choice": "Per method and instance: the share at the target, the Kaplan-Meier quartiles (None beyond "
                            "the budget); per pair, P(TTT_A < TTT_B) over all pairs of runs (a tie counts 1/2, a "
                            "censored run loses to one that got there, two censored runs tie).",
                  "why": "§12 of 2026-10-17, TTT curves (4)."},
    "degenerate": {"choice": "A start that already meets the target makes every time t0: the instance is reported as "
                             "degenerate at that τ (the numbers are kept, flagged).",
                   "why": "§12 of 2026-10-17, TTT curves (5)."},
    "status": {"choice": "Descriptive: no test, no interval, no pooling over the instances.",
               "why": "§12 of 2026-10-17, TTT curves (6)."},
}

PRESPECIFIED_DELAY = {
    "rows": {"choice": "The registered run of the stage cold (20261013-121922-E2-cold-4a6d5e): no common start, one "
                       "solver seed per method and instance.",
             "why": "§12 of 2026-10-17, delay (1): in the warm runs the common start is everyone's first solution."},
    "delay": {"choice": "The time of the first point (finite value) of the method's own trajectory in the cold row; the "
                        "field first_own_s is not read.",
              "why": "§12 of 2026-10-17, delay (2); audit G2, C-4 (the field of local search in the cold run of "
                     "2026-10-13 is the first refinement of a partial subset)."},
    "summaries": {"choice": "The value of an instance is the Kaplan-Meier median over its seeds (the delay of the one run "
                            "of the cold stage); per cell and method the median, quartiles (linear interpolation) and "
                            "maximum over the instances, a censored instance entering at the budget, and the share of "
                            "censored instances.",
                  "why": "§12 of 2026-10-17, delay (3)."},
    "status": {"choice": "Descriptive, no tests.", "why": "§12 of 2026-10-17, delay (4)."},
}


def p_faster(a_times: Sequence[float], a_reached: Sequence[bool], b_times: Sequence[float],
             b_reached: Sequence[bool]) -> float:
    """``P(TTT_A < TTT_B)`` over all pairs of a run of A and a run of B: a pair of runs that both got to the target
    counts 1 if A was faster, ½ on a tie; a run that got there beats a censored one; two censored runs tie."""
    at, ar = np.asarray(a_times, dtype=float)[:, None], np.asarray(a_reached, dtype=bool)[:, None]
    bt, br = np.asarray(b_times, dtype=float)[None, :], np.asarray(b_reached, dtype=bool)[None, :]
    if at.size == 0 or bt.size == 0:
        raise ValueError("both methods need at least one run")
    both = (at < bt) + 0.5 * (at == bt)
    score = np.where(ar & br, both, np.where(ar, 1.0, np.where(br, 0.0, 0.5)))
    return float(score.mean())


def _quantile(times: Sequence[float], reached: Sequence[bool], q: float) -> float | None:
    value = anytime.km_quantile(times, reached, q)
    return None if math.isinf(value) else value


def _tau_key(tau: float) -> str:
    return f"{tau:g}"


def _by_role(group: Sequence[Mapping], where: str, stage: str) -> dict[str, list[Mapping]]:
    out = {}
    for role in ROLES:
        mine = sorted((r for r in group if role in (r.get("roles") or ())), key=lambda r: r.get("solver_seed"))
        if not mine:
            raise ValueError(f"{where}: no rows of the stage {stage!r} for the role {role!r}")
        configs = {str(r.get("config_id")) for r in mine}
        if len(configs) > 1:
            raise ValueError(f"{where}: {len(configs)} configurations for the role {role!r} ({sorted(configs)})")
        seeds = [r.get("solver_seed") for r in mine]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{where}: a solver seed of the role {role!r} appears twice")
        out[role] = mine
    return out


def _same_counts(counts: Mapping[Any, set], what: str) -> None:
    for key, ns in sorted(counts.items(), key=str):
        if len(ns) > 1:
            raise ValueError(f"{what} {key}: the instances have different numbers of runs ({sorted(ns)})")


def ttt_curves_report(rows: Iterable[Mapping], *, references: Iterable[Mapping] = (),
                      budget_s: float = BUDGET_S) -> dict:
    """The TTT curves of the rows of the stage ``ttt`` (module docstring); no I/O."""
    pilot = an._pilot()
    rows = list(rows)
    for row in rows:
        an._check_row(row, "ttt", budget_s)
    used, ignored = an._reference_rows(references)
    refs = pilot.reference_table([*rows, *used])
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[row["instance_id"]].append(row)
    picked = {iid: _by_role(group, iid, "ttt") for iid, group in sorted(groups.items())}
    counts: dict[str, set] = defaultdict(set)
    for runs in picked.values():
        for role, mine in runs.items():
            counts[role].add(len(mine))
    _same_counts(counts, "role")

    instances: dict[str, dict] = {}
    methods: list[dict] = []
    pairs: list[dict] = []
    curves: list[dict] = []
    warnings: list[str] = []
    for iid, runs in picked.items():
        ref = refs[iid]
        group = groups[iid]
        gaps = [pilot.start_gap(r, ref) for r in group]
        degenerate = {_tau_key(tau): all(g <= tau for g in gaps) for tau in TAUS}
        instances[iid] = {
            "reference": {"f_ref": ref.f_ref, "s_ref": ref.s_ref, "ref_type": ref.ref_type, "source": ref.source,
                          "inconsistent": ref.inconsistent},
            "t0_s": float(np.median([float(r["start_time_s"]) for r in group])),
            "start_gap": float(np.median(gaps)), "degenerate": degenerate,
            "n_runs": {role: len(mine) for role, mine in runs.items()}}
        for tau in TAUS:
            flag = degenerate[_tau_key(tau)]
            if flag:
                warnings.append(f"{iid}: the common start meets the target τ = {tau:g} (ĝ = {np.median(gaps):.3g}): "
                                "every time is t0, the instance is degenerate at this target (§12 of 2026-10-17 (5))")
            ttt: dict[str, tuple[list[float], list[bool]]] = {}
            for role, mine in runs.items():
                hits = [pilot.row_ttt(r, ref, tau, budget_s=budget_s) for r in mine]
                times, reached = [h.time_s for h in hits], [h.reached for h in hits]
                ttt[role] = (times, reached)
                observed = sorted(t for t, ok in zip(times, reached, strict=True) if ok)
                n = len(hits)
                curves += [{"analysis": ANALYSIS, "tau": tau, "instance_id": iid, "role": role, "i": i, "time_s": t,
                            "p": (i - 0.5) / n} for i, t in enumerate(observed, 1)]
                methods.append({
                    "analysis": ANALYSIS, "tau": tau, "instance_id": iid, "role": role,
                    "method": (mine[0].get("method") or {}).get("name"), "config_id": mine[0].get("config_id"),
                    "n_runs": n, "n_reached": len(observed), "share_reached": len(observed) / n,
                    **{key: _quantile(times, reached, q) for key, q in QUARTILES.items()}, "degenerate": flag})
            for a, b in PAIRS:
                pairs.append({"analysis": ANALYSIS, "tau": tau, "instance_id": iid, "a": a, "b": b,
                              "p_a_faster": p_faster(*ttt[a], *ttt[b]), "n_pairs": len(ttt[a][0]) * len(ttt[b][0]),
                              "degenerate": flag})
    uncertified = sorted(iid for iid, part in instances.items() if part["reference"]["ref_type"] != "certified")
    if uncertified:
        warnings.append(f"{len(uncertified)} of {len(instances)} instances have no certified reference "
                        f"({', '.join(uncertified)}): the best known value over the rows is their reference")
    inconsistent = sorted(iid for iid, part in instances.items() if part["reference"]["inconsistent"])
    if inconsistent:
        warnings.append(f"a run beat the certified optimum on {', '.join(inconsistent)}: the best known value replaced "
                        "the certificate; investigate")
    if ignored:
        warnings.append(f"{ignored} reference row(s) of other stages (or with an error) were ignored: only the stages "
                        f"{', '.join(an.REFERENCE_STAGES)} carry certificates")
    return {
        "kind": "e2-ttt-curves", "schema": "e2-ttt-curves/1", "plan_version": "v2.4", "analysis": ANALYSIS,
        "confirmatory": False, "notice": NOTICE, "prespecified": PRESPECIFIED_TTT,
        "settings": {"taus": list(TAUS), "budget_s": budget_s, "roles": list(ROLES), "pairs": [list(p) for p in PAIRS],
                     "plotting_position": "(i - 1/2)/n", "quartiles": "Kaplan-Meier"},
        "instances": instances, "methods": methods, "pairs": pairs, "curves": curves,
        "references": {"n_rows_used": len(used), "n_rows_ignored": ignored},
        "warnings": warnings, "inputs": [], "code": None,
    }


def _first_own_time(row: Mapping) -> float | None:
    """The time of the first point with a finite value of the row's own trajectory."""
    for t, f in row.get("trajectory") or ():
        if f is not None and math.isfinite(float(f)):
            return float(t)
    return None


def delay_report(rows: Iterable[Mapping], *, budget_s: float = BUDGET_S) -> dict:
    """The delay to the first own solution on the rows of the stage ``cold`` (module docstring); no I/O."""
    rows = list(rows)
    for row in rows:
        an._check_row(row, "cold", budget_s)
    groups: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["instance_id"])].append(row)
    counts: dict[tuple[str, str], set] = defaultdict(set)
    records: list[dict] = []
    for (cell, iid), group in sorted(groups.items()):
        for role, mine in _by_role(group, f"{cell}/{iid}", "cold").items():
            counts[(cell, role)].add(len(mine))
            firsts = [_first_own_time(r) for r in mine]
            reached = [t is not None and t <= budget_s for t in firsts]
            times = [t if ok else budget_s for t, ok in zip(firsts, reached, strict=True)]
            value = anytime.km_median(times, reached)
            censored = math.isinf(value)
            records.append({"analysis": ANALYSIS, "cell": cell, "instance_id": iid, "role": role,
                            "method": (mine[0].get("method") or {}).get("name"), "config_id": mine[0].get("config_id"),
                            "n_seeds": len(mine), "n_censored_seeds": reached.count(False),
                            "delay_s": budget_s if censored else value, "censored": censored})
    _same_counts(counts, "cell and role")
    present = {r["cell"] for r in records}
    order = [c for c in an.H2_CELLS if c in present] + sorted(present - set(an.H2_CELLS))
    cells: list[dict] = []
    for cell in order:
        for role in ROLES:
            mine = [r for r in records if r["cell"] == cell and r["role"] == role]
            values = np.array([r["delay_s"] for r in mine], dtype=float)
            n_censored = sum(bool(r["censored"]) for r in mine)
            cells.append({"analysis": ANALYSIS, "cell": cell, "role": role, "n_instances": len(mine),
                          "n_censored": n_censored, "share_censored": n_censored / len(mine),
                          "median_s": float(np.median(values)), "q1_s": float(np.percentile(values, 25)),
                          "q3_s": float(np.percentile(values, 75)), "max_s": float(values.max())})
    ls_rows = [r for r in rows if "ls" in (r.get("roles") or ())]
    early = sum(1 for r in ls_rows if r.get("first_own_s") is not None and _first_own_time(r) is not None
                and float(r["first_own_s"]) < _first_own_time(r) - 1e-9)
    warnings = []
    censored = sum(bool(r["censored"]) for r in records)
    if censored:
        warnings.append(f"{censored} of {len(records)} method-instance values have no own solution within the budget: "
                        "censored at it, counted per cell")
    return {
        "kind": "e2-delay", "schema": "e2-delay/1", "plan_version": "v2.4", "analysis": ANALYSIS,
        "confirmatory": False, "notice": NOTICE, "prespecified": PRESPECIFIED_DELAY,
        "settings": {"budget_s": budget_s, "roles": list(ROLES), "quantiles": "linear interpolation over instances"},
        "instances": records, "cells": cells,
        "first_own_field": {"n_ls_rows": len(ls_rows), "n_before_first_portfolio": early,
                            "note": "not read (audit G2, C-4): the delay comes from the trajectories"},
        "warnings": warnings, "inputs": [], "code": None,
    }


# --------------------------------------------------------------------------------------------
# Post hoc: the quality of the first own portfolio (§12, 2026-10-19 (3))
# --------------------------------------------------------------------------------------------

#: The label of the post hoc analysis (review r1 of the manuscript, M2); not a planned output.
POST_HOC = "post_hoc"
POST_HOC_NOTICE = ("POST HOC, descriptive (ANALYSIS_PLAN §12, 2026-10-19 (3), review r1 M2): not a planned analysis "
                   "and not confirmatory")
#: Two values of ĝ closer than this (in units of the scale) count as equal.
FIRST_TOL = 1e-9


def _first_own_point(row: Mapping) -> tuple[float, float] | None:
    """The first point with a finite value of the row's own trajectory, as (time, value)."""
    for t, f in row.get("trajectory") or ():
        if f is not None and math.isfinite(float(f)):
            return float(t), float(f)
    return None


def first_quality_report(rows: Iterable[Mapping], h2: Mapping, *, budget_s: float = BUDGET_S,
                         tau: float = anytime.TAU_MAIN) -> dict:
    """The gap of each representative's first own portfolio in the cold runs against the reference of the instance,
    set against the common start of the same instance (§12, 2026-10-19 (3)); no I/O.

    ``h2`` is the report of ``e2_analysis h2``: its instances give the reference (``f_ref``, ``s_ref``), the gap of
    the common start ``g_start`` and the time to build it ``t0_s``. Per instance and role: ``ĝ_first`` at the first
    finite point of the own trajectory (the median over the seeds; a run without an own portfolio within the budget
    has none and is censored), whether it is not worse than the common start (``ĝ_first ≤ ĝ_start + 1e-9``) and not
    later (``t_first ≤ t0``), and ``dt = t_first − t0``. Per cell and role: the median of ``ĝ_first/τ`` over the
    instances (``None`` when it lies beyond the budget), the two shares (a censored instance is worse and later) and
    the median ``dt``."""
    rows = list(rows)
    for row in rows:
        an._check_row(row, "cold", budget_s)
    known = {str(i["instance_id"]): i for i in h2.get("instances") or ()}
    groups: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        groups[(row["cell"], row["instance_id"])].append(row)
    counts: dict[tuple[str, str], set] = defaultdict(set)
    records: list[dict] = []
    for (cell, iid), group in sorted(groups.items()):
        inst = known.get(str(iid))
        if inst is None or inst.get("cell") != cell:
            raise ValueError(f"{cell}/{iid}: the instance is not in the h2 report (its reference and common start)")
        f_ref, s_ref = float(inst["f_ref"]), float(inst["s_ref"])
        g_start, t0 = float(inst["g_start"]), float(inst["t0_s"])
        for role, mine in _by_role(group, f"{cell}/{iid}", "cold").items():
            counts[(cell, role)].add(len(mine))
            gaps, times = [], []
            for r in mine:
                point = _first_own_point(r)
                if point is None or point[0] > budget_s:
                    gaps.append(math.inf)
                    times.append(math.inf)
                else:
                    gaps.append((point[1] - f_ref) / s_ref)
                    times.append(point[0])
            g_first, t_first = float(np.median(gaps)), float(np.median(times))
            censored = not (math.isfinite(g_first) and math.isfinite(t_first))
            records.append({
                "analysis": POST_HOC, "cell": cell, "instance_id": iid, "role": role,
                "method": (mine[0].get("method") or {}).get("name"), "config_id": mine[0].get("config_id"),
                "n_seeds": len(mine), "censored": censored,
                "g_first": None if censored else g_first, "g_first_over_tau": None if censored else g_first / tau,
                "t_first_s": None if censored else t_first, "g_start": g_start, "t0_s": t0,
                "dt_s": None if censored else t_first - t0,
                "not_worse_than_start": bool(not censored and g_first <= g_start + FIRST_TOL),
                "not_later_than_start": bool(not censored and t_first <= t0)})
    _same_counts(counts, "cell and role")
    present = {r["cell"] for r in records}
    order = [c for c in an.H2_CELLS if c in present] + sorted(present - set(an.H2_CELLS))
    cells: list[dict] = []
    for cell in order:
        for role in ROLES:
            mine = [r for r in records if r["cell"] == cell and r["role"] == role]
            ratios = np.array([math.inf if r["censored"] else r["g_first_over_tau"] for r in mine], dtype=float)
            median = float(np.median(ratios))
            dts = [r["dt_s"] for r in mine if not r["censored"]]
            cells.append({
                "analysis": POST_HOC, "cell": cell, "role": role, "n_instances": len(mine),
                "n_censored": sum(r["censored"] for r in mine),
                "median_g_first_over_tau": median if math.isfinite(median) else None,
                "share_not_worse": sum(r["not_worse_than_start"] for r in mine) / len(mine),
                "share_not_later": sum(r["not_later_than_start"] for r in mine) / len(mine),
                "median_dt_s": float(np.median(dts)) if dts else None})
    warnings = []
    if not h2.get("confirmatory"):
        warnings.append("the h2 report is not the confirmatory one: its references and common starts may differ from "
                        "the registered analysis")
    return {
        "kind": "e2-first-quality", "schema": "e2-first-quality/1", "plan_version": "v2.4", "analysis": POST_HOC,
        "confirmatory": False, "notice": POST_HOC_NOTICE,
        "settings": {"budget_s": budget_s, "tau": tau, "tol": FIRST_TOL, "roles": list(ROLES)},
        "instances": records, "cells": cells, "warnings": warnings, "inputs": [], "code": None,
    }


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

_POINT_COLUMNS = ("analysis", "tau", "instance_id", "role", "i", "time_s", "p")
_METHOD_COLUMNS = ("analysis", "tau", "instance_id", "role", "method", "config_id", "n_runs", "n_reached",
                   "share_reached", "q1_s", "median_s", "q3_s", "degenerate")
_PAIR_COLUMNS = ("analysis", "tau", "instance_id", "a", "b", "p_a_faster", "n_pairs", "degenerate")
_DELAY_INSTANCE_COLUMNS = ("analysis", "cell", "instance_id", "role", "method", "config_id", "n_seeds",
                           "n_censored_seeds", "delay_s", "censored")
_DELAY_CELL_COLUMNS = ("analysis", "cell", "role", "n_instances", "n_censored", "share_censored", "median_s", "q1_s",
                       "q3_s", "max_s")


def _write(report: Mapping, out_dir: Path, name: str, tables: Mapping[str, pd.DataFrame]) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / name]
    written[0].write_text(
        json.dumps(e2_eligibility._json_safe(dict(report)), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    for table, frame in tables.items():
        written.append(out_dir / table)
        frame.to_csv(written[-1], index=False)
    return written


def write_ttt_curves(report: Mapping, out_dir: Path) -> list[Path]:
    """``ttt_curves.json`` and the tables of the points, the methods and the pairs in ``out_dir``."""
    return _write(report, out_dir, "ttt_curves.json", {
        "ttt_curve_points.csv": pd.DataFrame(report["curves"], columns=list(_POINT_COLUMNS)),
        "ttt_curve_methods.csv": pd.DataFrame(report["methods"], columns=list(_METHOD_COLUMNS)),
        "ttt_curve_pairs.csv": pd.DataFrame(report["pairs"], columns=list(_PAIR_COLUMNS))})


def write_delay(report: Mapping, out_dir: Path) -> list[Path]:
    """``delay.json`` and the tables of the instances and the cells in ``out_dir``."""
    return _write(report, out_dir, "delay.json", {
        "delay_instances.csv": pd.DataFrame(report["instances"], columns=list(_DELAY_INSTANCE_COLUMNS)),
        "delay_cells.csv": pd.DataFrame(report["cells"], columns=list(_DELAY_CELL_COLUMNS))})


_FIRST_INSTANCE_COLUMNS = ("analysis", "cell", "instance_id", "role", "method", "config_id", "n_seeds", "censored",
                           "g_first", "g_first_over_tau", "t_first_s", "g_start", "t0_s", "dt_s",
                           "not_worse_than_start", "not_later_than_start")
_FIRST_CELL_COLUMNS = ("analysis", "cell", "role", "n_instances", "n_censored", "median_g_first_over_tau",
                       "share_not_worse", "share_not_later", "median_dt_s")


def write_first_quality(report: Mapping, out_dir: Path) -> list[Path]:
    """``first_quality.json`` and the tables of the instances and the cells in ``out_dir``."""
    return _write(report, out_dir, "first_quality.json", {
        "first_quality_instances.csv": pd.DataFrame(report["instances"], columns=list(_FIRST_INSTANCE_COLUMNS)),
        "first_quality_cells.csv": pd.DataFrame(report["cells"], columns=list(_FIRST_CELL_COLUMNS))})


# --------------------------------------------------------------------------------------------
# Command line (the arguments are defined in cif.experiments.e2_analysis)
# --------------------------------------------------------------------------------------------

#: What the inputs of each command must be: role → (stages of its config, seed roles of its config).
_EXPECTED = {
    "ttt-curves": {"rows": (("ttt",), ("calibration",)), "reference": (an.REFERENCE_STAGES, ("calibration",))},
    "delay": {"rows": (("cold",), ("test",))},
    "first-quality": {"rows": (("cold",), ("test",))},
}


def _input_warnings(inputs: Sequence[Mapping], code: Mapping, expected: Mapping[str, tuple]) -> list[str]:
    """What keeps the inputs from being registered runs of the expected stage and seeds, run and analysed from a
    clean tree."""
    warnings = []
    for entry in inputs:
        if entry.get("run_id") is None:
            warnings.append(f"{entry['source']}: not a registered run directory (no manifest.json)")
            continue
        name = entry["run_id"]
        stages, seed_roles = expected[entry["role"]]
        if entry.get("mode") not in ("full", "ingested"):
            warnings.append(f"{name}: manifest mode = {entry.get('mode')!r} (not a registered run)")
        if entry.get("git_dirty") is not False:
            warnings.append(f"{name}: manifest git_dirty = {entry.get('git_dirty')!r} (not a run from a clean tree)")
        if entry.get("stage") not in stages:
            warnings.append(f"{name}: config stage = {entry.get('stage')!r}, expected one of {stages}")
        if entry.get("seed_role") not in seed_roles:
            warnings.append(f"{name}: config seed_role = {entry.get('seed_role')!r}, expected one of {seed_roles}")
        if entry.get("n_errors") or entry.get("n_missing"):
            warnings.append(f"{name}: the run has {entry.get('n_errors')} cells with errors and "
                            f"{entry.get('n_missing')} missing")
    if code.get("git_dirty") is not False:
        warnings.append(f"the analysis code is from a dirty or unknown tree (git_dirty = {code.get('git_dirty')!r})")
    return warnings


def run_command(args) -> int:
    """``e2_analysis ttt-curves`` and ``e2_analysis delay``: read the registered runs, write the report and the
    tables, print a summary (marked secondary)."""
    pilot = an._pilot()
    rows, errors = pilot.load_rows(args.rows)
    paths = {"rows": args.rows}
    if args.command == "ttt-curves":
        paths["reference"] = args.reference
        report = ttt_curves_report([*rows, *errors], references=pilot.reference_rows(args.reference))
    elif args.command == "first-quality":
        report = first_quality_report([*rows, *errors], json.loads(args.h2.read_text(encoding="utf-8")))
    else:
        report = delay_report([*rows, *errors])
    inputs = [entry for role, given in paths.items() for entry in an._run_inputs(list(given), role)]
    state = registry._git_state()
    code = {"git_sha": state.get("sha") or None, "git_dirty": state.get("dirty"), "branch": state.get("branch") or None}
    problems = _input_warnings(inputs, code, _EXPECTED[args.command])
    if args.command == "first-quality":  # the registered analysis whose references and common starts are used
        inputs.append({**an._provenance(args.h2), "role": "h2"})
        problems += [w for w in report["warnings"] if "confirmatory" in w]
        report["warnings"] = [w for w in report["warnings"] if "confirmatory" not in w]
    versions = {"python": sys.version.split()[0], **{p: metadata.version(p) for p in ("numpy", "pandas", "scipy")}}
    report.update(inputs=inputs, code=code, versions=versions, warnings=[*problems, *report["warnings"]],
                  provenance_clean=not problems)
    if args.command == "ttt-curves":
        write_ttt_curves(report, args.out)
        print(f"e2-ttt-curves: {args.out}: {NOTICE}; {len(report['instances'])} instances")
        for m in report["methods"]:
            if m["tau"] == anytime.TAU_MAIN:
                print(f"  {m['instance_id']} {m['role']}: {100 * m['share_reached']:.0f} % at τ = {m['tau']:g}, "
                      f"median {m['median_s']} s")
    elif args.command == "first-quality":
        write_first_quality(report, args.out)
        print(f"e2-first-quality: {args.out}: {POST_HOC_NOTICE}; {len(report['instances'])} method-instance values")
        for c in report["cells"]:
            print(f"  {c['cell']} {c['role']}: median ĝ_first/τ {c['median_g_first_over_tau']}, not worse than the "
                  f"start {100 * c['share_not_worse']:.0f} %, not later {100 * c['share_not_later']:.0f} %")
    else:
        write_delay(report, args.out)
        print(f"e2-delay: {args.out}: {NOTICE}; {len(report['instances'])} method-instance values")
        for c in report["cells"]:
            print(f"  {c['cell']} {c['role']}: median {c['median_s']:.3g} s (max {c['max_s']:.3g} s), "
                  f"{c['n_censored']} of {c['n_instances']} censored")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0
