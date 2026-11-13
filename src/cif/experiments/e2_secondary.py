"""Secondary analyses of RQ2 from registered rows: time-to-target, ``t_primal`` and ``t_cert`` of the exact class,
performance profiles (ANALYSIS_PLAN §2 RQ2 "Вторично", §5, §7.2, §10; §12 A1-A2 and the entry of 2026-10-12
"Прочтения подтверждающей оценки H2/H2b"): ``python -m cif.experiments.e2_analysis secondary``.

**Everything here is secondary and not confirmatory.** The registry of §7.1 holds H2 and H2b for RQ2 and nothing
else; the estimates below are descriptive, carry no test and no adjustment beyond the Bonferroni level of the cell
intervals of H2, and every output says so (``role = "secondary"``, ``confirmatory = false``, a notice, a column
``analysis`` in every table). They read the same inputs as ``h2`` (rows of the bench stages, the finished
eligibility reports of the test scan, the certificates) through the same code, so the reference of an instance and
its PI⁺ are the numbers of H2. The code is committed before any bench row of a test seed is read; every reading the
plan leaves open is fixed in :data:`PRESPECIFIED` and written into the JSON next to the numbers.

**Time-to-target** (§5). The time of a run is the end-to-end time (the build of the common start included) of the
first moment its incumbent has ``ĝ ≤ τ`` against the reference of the instance (:func:`role_ttt`,
:func:`cif.experiments.e2_pilot.row_ttt`); a run that has not got there by the budget is censored at it. The value of
a representative on an instance is the Kaplan-Meier median over its seeds (:func:`cif.metrics.anytime.km_median`);
the ratios are ``TTT = value(QUBO)/value(exact)`` and ``TTTb = value(QUBO)/value(local search)``, pooled like H2
(:mod:`cif.stats.rq2`: cluster means of the log ratio, equal cell weights, stratified bootstrap, the five outcomes
against the band ``[1/1.5, 1.5]``). The cases the plan leaves open:

* a median beyond the budget (more than half of the seeds censored) is **censored, not dropped**: the instance value
  is the budget and the ratio is marked a bound (``upper`` when the exact / LS method is the censored one, ``lower``
  when QUBO is); such a ratio lies closer to 1 than the truth;
* both medians beyond the budget is ``inf/inf``: the ratio is undefined and the instance **leaves the estimate of that
  pair** (counted, reported by cell), it is not set to 1, which would pull the pooled estimate towards
  "equivalent"; the two pairs leave independently;
* a start that already meets the target (the time of every method is then the build of the start) is what A1
  screens out, but it happens on the instances that the screening lists as flipped by the scale of the optimum
  (``reference_check`` of the eligibility report, "Уточнения A1" (1): the decision stays): such an instance
  **leaves the TTT estimates** and is counted; one that is not listed **stops the analysis**, naming the instance;
* sensitivities: ``τ = 1e-4``, and the **censoring budget** (the same trajectories re-censored at 30 s and 10 s),
  which is what §5 asks for in place of dropping the censored runs.

**``t_primal`` and ``t_cert``** (§2): the rows of the exact representative, separately, per instance (median over the
seeds; Kaplan-Meier median of the certificate time) and per cell (the distribution of those, the share of runs that
certified within the budget). **Performance profiles** (§7.2, Dolan & Moré 2002): per cell over its instances for the
three representatives, on PI⁺ and on the time to the target, as tables of ρ on a fixed grid.

**Output** (:func:`write_secondary`): ``secondary.json`` (the report with ``prespecified``, the inputs with run ids
and SHA-256, the code state) and the tables ``instances.csv`` (one row per instance: reference, PI⁺, ``t_primal`` and
``t_cert``), ``ttt_instances.csv`` (every time to the target and ratio, by variant), ``ttt_cells.csv`` and
``ttt_pooled.csv`` (the estimates), ``ttt_censoring.csv`` (the classes of the instances), ``exact_times.csv`` and
``profiles.csv``; the first column of each is ``analysis = secondary``.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cif.experiments import e2_analysis as an
from cif.experiments import e2_eligibility, registry
from cif.metrics import anytime
from cif.stats import rq2
from cif.stats.rq1 import wilson_interval
from cif.stats.tests_misc import performance_profile

__all__ = [
    "ANALYSIS",
    "BUDGET_S",
    "CENSORING_BUDGETS_S",
    "PAIRS",
    "PRESPECIFIED",
    "PROFILE_RATIOS",
    "SENSITIVITIES",
    "TAU",
    "TAU_SENSITIVITY",
    "exact_instance",
    "profile_block",
    "role_ttt",
    "run_command",
    "secondary_report",
    "ttt_instances",
    "write_secondary",
]

#: The label every table and the JSON carry.
ANALYSIS = "secondary"
#: The budget of the bench stages (§6) and of the main analysis; the censoring budgets below can only be shorter.
BUDGET_S = an.BUDGET_S
#: The main target of the time-to-target (§5; also the target of the eligibility rule A1) and its sensitivity.
TAU = anytime.TAU_MAIN
TAU_SENSITIVITY = anytime.TAU_SENSITIVITY
#: Censoring budgets of the sensitivity (§5, minor 9): the trajectories end at the budget, so only shorter ones exist.
CENSORING_BUDGETS_S = (30.0, 10.0)
#: Grid of the performance profiles: the performance ratio ``r`` of Dolan & Moré (their τ; called ``ratio`` here to
#: keep τ for the target gap). ``ρ(inf)`` is reported next to it.
PROFILE_RATIOS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
#: The order statistics of the descriptive distributions.
QUANTILES = {"min": 0.0, "q1": 0.25, "median": 0.5, "q3": 0.75, "max": 1.0}
#: The secondary counterparts of H2 and H2b: the ratio of QUBO to the method named here.
PAIRS = {"TTT": "exact", "TTTb": "ls"}
#: How a pair of values of one instance is classed: both medians observed, only QUBO's, only the other's, neither.
STATUSES = ("both", "qubo_only", "other_only", "neither")
_METRIC = "log_r"  # the name of the metric inside cif.stats.rq2 (one metric per call: the pairs have their own instances)

#: Sensitivities of the time-to-target: name → (settings, what it is). ``censor_s`` ``None`` is the budget of the rows.
SENSITIVITIES = {
    "tau_1e-4": ({"tau": TAU_SENSITIVITY, "censor_s": None}, "target τ = 1e-4 instead of 1e-3 (§5), budget 60 s"),
    **{f"budget_{b:g}": ({"tau": TAU, "censor_s": b}, f"τ = 1e-3, the trajectories re-censored at {b:g} s (§5, minor 9)")
       for b in CENSORING_BUDGETS_S},
}

#: What the frozen plan leaves open and how the analysis reads it (written into the JSON of every report; the
#: orchestrator records the same in §12 before the bench stages). ``choice`` is what the code does, ``why`` the plan
#: text or the convention of the project it follows.
PRESPECIFIED = {
    "status": {
        "choice": "Secondary and not confirmatory throughout: no test of the registry of §7.1, no multiplicity "
                  "adjustment other than the Bonferroni level of the cell intervals, descriptive intervals; every "
                  "output is flagged (role = secondary, confirmatory = false, a notice, a column `analysis`). The pooled "
                  "TTT estimates get the same five outcomes as H2 and H2b against [1/1.5, 1.5], coded without a "
                  "sentence: §10 has none for them.",
        "why": "§2 RQ2 'Вторично' ('Time-to-target с теми же исходами'); §7.1 lists only H2 and H2b for RQ2; §9 "
               "(only the registry is confirmatory).",
    },
    "inputs": {
        "choice": "The rows, the finished eligibility reports, the certificates and the reference of every instance "
                  "are those of `e2_analysis h2`, through the same code (bench_instances, reference_table): PI+, the "
                  "reference and its type are the same numbers in both analyses. The stage `cold` is not read.",
        "why": "audit G1 RQ2-1 and the entry of §12 of 2026-10-12: one reference per instance, the time to the "
               "target is measured against it.",
    },
    "ttt_clock": {
        "choice": "The time of a run is the end-to-end time (seconds from the start of the common clock, the build of "
                  "the common start t0 included) of the first moment its incumbent has ĝ = (f - f_ref)/s_ref <= τ "
                  "(cif.metrics.anytime.time_to_target on the trajectory of the row with its start); a run that has "
                  "not got there by the budget is censored at it (an event exactly at the budget counts, later ones "
                  "do not). τ = 1e-3 is the main target (the target of the eligibility rule A1), τ = 1e-4 the "
                  "sensitivity. There is no variant without t0: t0 is a step common to all methods and a small part of "
                  "60 s (t0/T is 2e-4 to 3e-3 on the certified calibration instances, configs/E2/README.md), and it is "
                  "reported per instance (t0_s). The target is measured post hoc against the reference: a heuristic "
                  "does not know that it has reached it.",
        "why": "§5 'Время end-to-end до ĝ <= τ, τ = 1e-3 (основное) и 1e-4 (чувствительность)' and 'Время "
               "end-to-end: построение модели или QUBO, решение, доводка'; A2 leaves t0 charged to the budget of "
               "every method; the post-hoc remark is M8 of the review of round 0.",
    },
    "ttt_value": {
        "choice": "The value of a representative on an instance is the Kaplan-Meier median of the times of its runs "
                  "(cif.metrics.anytime.km_median: the smallest time at which the survival curve is <= 1/2; every run "
                  "is censored at the same budget, so it is the ceil(n/2)-th smallest time when that many runs reached "
                  "the target and infinite otherwise: the 2nd of 3 seeds of the exact method, the 5th of 10 of a "
                  "heuristic). An infinite median (more than half of the seeds censored) is neither dropped nor "
                  "extrapolated: the value is the censoring budget, the instance is marked censored for that "
                  "representative and the number of such instances is reported per cell.",
        "why": "§5 'Прогон, не достигший цели, цензурируется на бюджете; медиана по сидам — Каплан–Мейер'; §7.2 "
               "'Сводка по инстансу — медиана по сидам'.",
    },
    "ttt_ratio": {
        "choice": "TTT = value(QUBO)/value(exact) and TTTb = value(QUBO)/value(local search) per instance, natural "
                  "logarithm; below 1 the QUBO pipeline gets to the target sooner. If exactly one of the two values is "
                  "a censored median the ratio is formed with the budget in its place and marked a bound (upper when "
                  "the exact / LS method is the censored one, lower when QUBO is): it is kept and counted. If both are "
                  "censored the ratio is inf/inf and undefined: the instance leaves the estimate of that pair, is "
                  "counted ('neither') and is not set to 1, which would pull the pooled estimate towards "
                  "'equivalent'. The two pairs lose instances independently.",
        "why": "§2 R = PI(QUBO)/PI(exact), R_b = PI(QUBO)/PI(LS) and 'с теми же исходами'; §5 leaves an unreached "
               "median open; as for PI+ = 0 in H2 (a 0/0 ratio is not floored) an undefined ratio is not invented.",
    },
    "ttt_censoring": {
        "choice": "Censoring is reported, not repaired: per variant, cell and pair the numbers of instances with both "
                  "medians observed, only QUBO's, only the other method's, neither; and per representative the "
                  "number of censored instances. A bound lies closer to 1 than the truth, so an 'equivalent' outcome "
                  "of a cell with many bounds is weaker than it reads; the counts are the check.",
        "why": "§5 'Чувствительность — к бюджету цензурирования, а не исключение цензурированных (minor 9)': the "
               "censored are never excluded as runs and the cost of keeping them is shown.",
    },
    "ttt_pooling": {
        "choice": "As H2: cluster means of the log ratio (the instance; in C5 the port, the mean over its λ points), "
                  "the mean of the cell means with equal weights over the cells of summary.cells of the eligibility "
                  "report, the percentile 95 % interval of a stratified bootstrap with B = 9999 and seed 60000, one "
                  "stream [seed, 2, crc32(cell)] per cell (cif.stats.rq2.estimate_rq2), cell intervals at the "
                  "Bonferroni level of the pooled cells (99 % simultaneous), the five outcomes against the closed "
                  "band [1/1.5, 1.5]. The two pairs are estimated separately. A pooled cell with fewer than two "
                  "clusters of a defined ratio in a pair leaves the pool of that pair (the family of the Bonferroni "
                  "levels shrinks with it) instead of stopping the analysis, and the notes say which cell and why; with "
                  "no pooled cell left a pair has no pooled estimate.",
        "why": "§2 and §7.2 (the machinery of H2); a secondary analysis does not stop for a cell that the "
               "censoring has emptied, as the confirmatory one does.",
    },
    "ttt_start": {
        "choice": "An instance whose common start already meets the target (ĝ_start <= τ against the reference of the "
                  "analysis, in a run of any representative) has no time to the target that is worth a ratio: the time "
                  "of every method is the build of the start. If the screening lists it as flipped by the scale of the "
                  "optimum (the reference_check of the eligibility report, class nontrivial) it leaves the TTT "
                  "estimates and the TTT profile at that target (not the other analyses), is counted per cell, and is "
                  "named in the report and in a warning; if it is not listed the analysis stops, naming the "
                  "instance, the representative and the gap. A time that is not positive stops it too.",
        "why": "§12 A1 and 'Уточнения A1' (1): the screening uses the scale of the start and does not revise a "
               "decision when the scale of the optimum would put the instance on the other side of τ, so such "
               "instances exist and stay in H2 and H2b (their PI+ is positive); for the time to the target they are "
               "what A1 screens out. An unlisted one would mean that the screening and the references disagree, a "
               "question for the orchestrator, not for a floor (as PI+ = 0 in H2).",
    },
    "ttt_sensitivity": {
        "choice": "Three sensitivities with the estimator and the resamples of the primary: tau_1e-4 (target 1e-4, "
                  "budget 60 s), budget_30 and budget_10 (target 1e-3, the same trajectories re-censored at 30 s and "
                  "at 10 s; only shorter budgets exist, the rows end at 60 s). A run is censored by the budget, none is "
                  "excluded. A Kaplan-Meier median that lies below the new budget is unchanged by it and one above "
                  "becomes infinite, so what the budgets show is how many medians lie beyond a shorter horizon.",
        "why": "§5 'Чувствительность — к бюджету цензурирования, а не исключение цензурированных (minor 9)'; "
               "dropping the censored favours the methods that often miss the target (round 1 of the review). The "
               "budgets: 30 s is half of the budget of RQ2, 10 s the budget of the tuning criterion and of H1c (§6).",
    },
    "exact_times": {
        "choice": "Rows of the role exact_rep only. t_primal is the field t_primal_s of the row (the end-to-end time of "
                  "the last improvement of the incumbent over the common start, the time of the start if there is "
                  "none); t_cert is t_cert_s (the time at which the solver proved the incumbent optimal to 1e-6·s). A "
                  "run has a certificate within the budget iff certified is true and t_cert_s <= the budget (a "
                  "later one does not count, as events after the budget count nowhere); otherwise it is censored at the "
                  "budget. Per instance: the median of t_primal over the seeds, the Kaplan-Meier median of t_cert "
                  "(null when beyond the budget). Per cell: n, min, q1, median, q3, max of the instance values as "
                  "order statistics (numpy inverted_cdf: the smallest value with at least a share p of the values at or "
                  "below it, a censored value lying beyond all finite ones); the share of instances with an observed "
                  "t_cert and the share of runs certified within the budget, with 95 % Wilson intervals; t_primal of "
                  "the runs split by certified or not (without a certificate t_primal is only the last improvement "
                  "seen, not the time the optimum was found). Descriptive: no test, no pooling over cells.",
        "why": "§2 RQ2 'У точных методов t_primal и t_cert — раздельно'; §5 (M8: search against proof); the rows of "
               "e2_benchmark carry both fields.",
    },
    "profiles": {
        "choice": "Dolan-Moré profiles per cell over all its non-trivial instances (C5: its points, the ports are not "
                  "clusters here) for the three representatives on two costs: PI+ (median over the seeds, positive "
                  "on a non-trivial instance) and the time to the target at τ = 1e-3, budget 60 s (the "
                  "Kaplan-Meier median; infinite when censored, which is a failure of the method on the instance, "
                  "not the budget). The ratio of a cost to the best of the three on the instance; rho(r) is the "
                  "share of instances with a ratio <= r on the grid 1, 1.1, 1.25, 1.5, 2, 3, 4, 8, 16, ..., 1024 and "
                  "rho(inf) the share with a finite cost; an instance solved by no method stays in the denominator "
                  "and is counted; the profile of the time to the target leaves out the instances whose start meets "
                  "the target (counted in n_start_met). Tables only: no plot, no test.",
        "why": "§2 RQ2 'performance profiles'; §7.2 'Performance profiles (Dolan & Moré 2002) ... описательно'.",
    },
    "budget_overrun": {
        "choice": "Rows with budget_ok = false are used as measured (what a run found after the budget does not "
                  "count, in the times and in PI+); their number is reported per cell and representative. There is no "
                  "sensitivity that drops them: for the times it would drop the instances on which the exact method is "
                  "slow.",
        "why": "§5 (time is end-to-end and charged to the method); the reading of H2 for the same rows.",
    },
    "degenerate": {
        "choice": "The analysis stops, naming the instance, on what stops H2 (PI+ = 0 or not finite, rows of an "
                  "instance the screening did not select, a selected instance without rows or without a "
                  "representative, different numbers of seeds across the instances of a cell, several "
                  "configurations for one role in one cell and size, rows with an error, skipped rows, rows of another "
                  "budget), on a start that meets the target and is not listed as flipped, on a time that is not "
                  "positive. No floor is applied.",
        "why": "§12 A1/A2 and the entry of §12 of 2026-10-12 (9).",
    },
}


def _pilot():
    return an._pilot()


def _finite(value: float) -> float | None:
    """``None`` for an infinite or missing number: JSON has no infinity."""
    return None if value is None or not math.isfinite(value) else float(value)


# --------------------------------------------------------------------------------------------
# The time to the target
# --------------------------------------------------------------------------------------------


def role_ttt(rows: Sequence[Mapping], ref, *, tau: float, censor_s: float, where: str = "") -> dict:
    """The time to the target ``τ`` of one representative on one instance: its ``rows`` are the runs (seeds), ``ref``
    the reference of the instance, ``censor_s`` the budget at which a run that has not got there is censored (it
    cannot exceed the budget of the rows: the trajectories end there). Returns the number of seeds, how many reached
    the target, the Kaplan-Meier median in seconds (``None`` when it lies beyond the budget), whether the instance is
    censored for the representative and its ``value_s``: the median, or the budget when censored."""
    pilot = _pilot()
    if not rows:
        raise ValueError(f"{where}: no runs of the representative")
    times: list[float] = []
    reached: list[bool] = []
    for row in rows:
        if censor_s > float(row["budget_s"]):
            raise ValueError(f"{where}: cannot censor at {censor_s} s, the rows end at {row['budget_s']} s")
        result = pilot.row_ttt(row, ref, tau, budget_s=censor_s)
        times.append(result.time_s)
        reached.append(result.reached)
    median = anytime.km_median(times, reached)
    censored = not math.isfinite(median)
    return {"n_seeds": len(rows), "n_reached": int(sum(reached)), "median_s": None if censored else float(median),
            "censored": censored, "value_s": float(censor_s) if censored else float(median)}


def _pair(qubo: Mapping, other: Mapping, *, where: str) -> dict:
    """The ratio of the value of QUBO to that of the other method and how it was formed (module docstring)."""
    if qubo["censored"] and other["censored"]:
        return {"status": "neither", "bound": None, "ratio": None, "log_ratio": None}
    if qubo["censored"]:
        status, bound = "other_only", "lower"
    elif other["censored"]:
        status, bound = "qubo_only", "upper"
    else:
        status, bound = "both", None
    q, o = qubo["value_s"], other["value_s"]
    if not (q > 0.0 and o > 0.0):
        raise ValueError(f"{where}: a time to the target that is not positive (QUBO {q} s, other {o} s) has no "
                         "logarithm")
    return {"status": status, "bound": bound, "ratio": q / o, "log_ratio": math.log(q / o)}


def _start_met(group: Mapping[str, Sequence[Mapping]], ref, tau: float) -> tuple[str, float] | None:
    """The first representative with a run whose common start already meets the target, and the gap of that start."""
    pilot = _pilot()
    for key in an.REPRESENTATIVES:
        for row in group[key]:
            gap = pilot.start_gap(row, ref)
            if gap <= tau:
                return key, gap
    return None


def ttt_instances(
    instances: Sequence[Mapping], groups: Mapping[tuple, Mapping[str, Sequence[Mapping]]], refs: Mapping, *,
    tau: float, censor_s: float, start_met_ok: Iterable[str] = (),
) -> list[dict]:
    """One dict per instance of ``instances`` (the records of :func:`cif.experiments.e2_analysis.bench_instances`):
    the time to the target of each representative (:func:`role_ttt`) and the two ratios of :data:`PAIRS`.
    ``groups[(cell, unit)][role]`` are the rows of the representatives.

    An instance whose common start already meets the target has no ratio (``start_met``, status ``"start_met"``): the
    time of every method is the build of the start. It is allowed only if its id is in ``start_met_ok`` (the
    instances that the screening lists as flipped, module docstring); otherwise it is an error."""
    allowed = set(start_met_ok)
    out = []
    for inst in instances:
        where = f"{inst['cell']}/{an._unit_text(inst['unit'])} ({inst['instance_id']})"
        ref = refs[inst["instance_id"]]
        group = groups[(inst["cell"], inst["unit"])]
        met = _start_met(group, ref, tau)
        if met is not None and inst["instance_id"] not in allowed:
            raise ValueError(
                f"{where}, {met[0]}: the common start already meets the target (ĝ_start = {met[1]:.3g} <= τ = "
                f"{tau:g}), so the time to the target is the time of the start itself. By A1 a selected instance is "
                "not like that unless the screening lists it as flipped by the scale of the optimum (the "
                "reference_check of the eligibility report, 'Уточнения A1' (1)); this one is not listed")
        roles = {key: role_ttt(group[key], ref, tau=tau, censor_s=censor_s, where=f"{where}, {key}")
                 for key in an.REPRESENTATIVES}
        record: dict[str, Any] = {"cell": inst["cell"], "unit": inst["unit"], "cluster": inst["cluster"],
                                  "instance_id": inst["instance_id"], "start_met": met is not None,
                                  "g_start": inst["g_start"], "roles": roles}
        for name, other in PAIRS.items():
            record[name] = (_pair(roles["qubo"], roles[other], where=f"{where}, {name}") if met is None else
                            {"status": "start_met", "bound": None, "ratio": None, "log_ratio": None})
        out.append(record)
    return out


def _estimate_pair(records: Sequence[Mapping], name: str, design_pool: Sequence[str], *, n_boot: int,
                   seed: int) -> tuple[rq2.Rq2Result | None, list[str]]:
    """The estimate of :mod:`cif.stats.rq2` for one pair from the instances with a defined ratio (cluster means:
    instances; ports in C5). A pooled cell with fewer than two clusters leaves the pool of the pair, with a note."""
    usable = [r for r in records if r[name]["log_ratio"] is not None]
    present = {r["cell"] for r in usable}
    order = [c for c in an.H2_CELLS if c in present] + sorted(present - set(an.H2_CELLS))
    cells: dict[str, dict[str, np.ndarray]] = {}
    for cell in order:
        mine = [r for r in usable if r["cell"] == cell]
        _, means = rq2.mean_of_means([r[name]["log_ratio"] for r in mine], [r["cluster"] for r in mine])
        cells[cell] = {_METRIC: means}
    pool: list[str] = []
    notes: list[str] = []
    for cell in design_pool:
        n = len(cells[cell][_METRIC]) if cell in cells else 0
        if n >= 2:
            pool.append(cell)
        else:
            notes.append(f"{name}: {cell} has {n} cluster(s) with a defined ratio, fewer than the two a pooled cell "
                         f"needs: it is not in the pooled estimate of {name}")
    if not cells:
        return None, notes
    return rq2.estimate_rq2(cells, pooled=pool, n_boot=n_boot, seed=seed), notes


def _censoring(mine: Sequence[Mapping]) -> dict:
    """How the instances of a cell were classed (module docstring): those whose start meets the target, and of the
    others the number censored per representative and the classes of the two pairs."""
    rest = [r for r in mine if not r["start_met"]]
    out: dict[str, Any] = {"start_met": len(mine) - len(rest),
                           "censored_instances": {key: sum(1 for r in rest if r["roles"][key]["censored"])
                                                  for key in an.REPRESENTATIVES}}
    for name in PAIRS:
        counts = Counter(r[name]["status"] for r in rest)
        out[name] = {status: counts.get(status, 0) for status in STATUSES}
    return out


def _cell_order(selected: Iterable[str]) -> list[str]:
    cells = list(selected)
    return [c for c in an.H2_CELLS if c in cells] + sorted(c for c in cells if c not in an.H2_CELLS)


def _ttt_variant(
    name: str, definition: str, tau: float, censor_s: float, instances: Sequence[Mapping], groups: Mapping,
    refs: Mapping, selected: Iterable[str], pooled: Sequence[str], start_met_ok: Iterable[str], *, n_boot: int,
    seed: int,
) -> dict:
    """The estimate of the time-to-target ratios for one target and one censoring budget."""
    records = ttt_instances(instances, groups, refs, tau=tau, censor_s=censor_s, start_met_ok=start_met_ok)
    estimates: dict[str, rq2.Rq2Result | None] = {}
    notes: list[str] = []
    for pair in PAIRS:
        estimates[pair], pair_notes = _estimate_pair(records, pair, pooled, n_boot=n_boot, seed=seed)
        notes += pair_notes
    pooled_blocks: dict[str, dict | None] = {}
    for pair, est in estimates.items():
        if est is None or est.pooled is None:
            pooled_blocks[pair] = None
            continue
        result = est.pooled[_METRIC]
        pooled_blocks[pair] = {"cells": list(result.cells), "n_cells": len(result.cells), "weights": "equal",
                               "family_size": est.settings["family_size"], "cell_level": est.settings["cell_level"],
                               **an._hypothesis_block(result)}
    cells: dict[str, dict] = {}
    for cell in _cell_order(selected):
        mine = [r for r in records if r["cell"] == cell]
        block: dict[str, Any] = {"n_instances": len(mine), "in_design_pool": cell in pooled,
                                 "censoring": _censoring(mine)}
        for pair, est in estimates.items():
            result = None if est is None else est.cells[_METRIC].get(cell)
            block[pair] = None if result is None else {
                "n_clusters": result.n_clusters, "in_pooled": result.in_pooled, **an._hypothesis_block(result)}
        cells[cell] = block
    start_met = [{"cell": r["cell"], "unit": r["unit"], "instance_id": r["instance_id"], "g_start": r["g_start"]}
                 for r in records if r["start_met"]]
    return {"name": name, "definition": definition, "tau": tau, "censor_s": censor_s, "pooled": pooled_blocks,
            "cells": cells, "start_met": start_met, "notes": notes, "instances": records}


# --------------------------------------------------------------------------------------------
# t_primal and t_cert of the exact class
# --------------------------------------------------------------------------------------------


def exact_instance(rows: Sequence[Mapping], *, budget_s: float) -> dict:
    """``t_primal`` and ``t_cert`` of the exact representative on one instance, from its runs (seeds): the median of
    ``t_primal_s`` over the runs that have one, the Kaplan-Meier median of the certificate time (``None`` when beyond
    the budget) with the runs that have no certificate within the budget censored at the budget, and the counts of
    certified runs (a row without the fields is a run that did not certify)."""
    primal = [float(r["t_primal_s"]) for r in rows if r.get("t_primal_s") is not None]
    times: list[float] = []
    within: list[bool] = []
    certified = 0
    for r in rows:
        has = r.get("certified") is True
        certified += has
        t = r.get("t_cert_s")
        ok = bool(has and t is not None and float(t) <= budget_s)
        times.append(float(t) if ok else float(budget_s))
        within.append(ok)
    median = anytime.km_median(times, within)
    return {"n_seeds": len(rows), "n_certified": int(certified), "n_certified_in_budget": int(sum(within)),
            "t_primal_median_s": float(np.median(primal)) if primal else None,
            "t_cert_median_s": _finite(median), "t_cert_censored": not math.isfinite(median)}


def _summary(values: Sequence[float]) -> dict:
    """``n`` and the order statistics of ``values`` (:data:`QUANTILES`, numpy ``inverted_cdf``); an infinite one is
    ``None`` (beyond the budget)."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return {"n": 0, **dict.fromkeys(QUANTILES)}
    return {"n": int(x.size), **{name: _finite(float(np.quantile(x, p, method="inverted_cdf")))
                                 for name, p in QUANTILES.items()}}


def _share(k: int, n: int) -> tuple[float | None, list[float | None]]:
    if n == 0:
        return None, [None, None]
    low, high = wilson_interval(k, n)
    return k / n, [low, high]


def _exact_cell(records: Sequence[Mapping], runs: Sequence[Mapping]) -> dict:
    """The distribution over the instances of a cell and over its exact runs (module docstring)."""
    finite_cert = [r["t_cert_median_s"] for r in records if r["t_cert_median_s"] is not None]
    t_cert = _summary([math.inf if r["t_cert_censored"] else r["t_cert_median_s"] for r in records])
    share, ci = _share(len(finite_cert), len(records))
    t_cert.update(n_censored=len(records) - len(finite_cert), share_finite=share, ci95_share_finite=ci)
    n_in_budget = sum(r["n_certified_in_budget"] for r in records)
    share_runs, ci_runs = _share(n_in_budget, len(runs))
    split = {"certified": [float(r["t_primal_s"]) for r in runs if r.get("t_primal_s") is not None
                           and r.get("certified") is True],
             "not_certified": [float(r["t_primal_s"]) for r in runs if r.get("t_primal_s") is not None
                               and r.get("certified") is not True]}
    return {
        "n_instances": len(records), "configs": sorted({str(r.get("config_id")) for r in runs}),
        "t_primal_s": _summary([r["t_primal_median_s"] for r in records if r["t_primal_median_s"] is not None]),
        "t_cert_s": t_cert,
        "runs": {"n_runs": len(runs), "n_certified_in_budget": int(n_in_budget),
                 "n_certified": int(sum(r["n_certified"] for r in records)), "share_in_budget": share_runs,
                 "ci95_share_in_budget": ci_runs, "t_primal_s": {k: _summary(v) for k, v in split.items()}},
    }


# --------------------------------------------------------------------------------------------
# Performance profiles
# --------------------------------------------------------------------------------------------


def profile_block(costs: Mapping[str, Sequence[float]], ratios: Sequence[float] = PROFILE_RATIOS) -> dict:
    """Dolan-Moré profile of ``costs[role]`` (one value per instance, ``inf`` for a failure) on ``ratios``:
    ``rho[role]`` on the grid, ``rho_inf[role]`` the share of instances with a finite cost, and the number of
    instances that no role solved (they stay in the denominator)."""
    matrix = np.vstack([np.asarray(list(v), dtype=float) for v in costs.values()])
    profile = performance_profile(costs, ratios)
    return {"rho": {role: [float(x) for x in values] for role, values in profile.items()},
            "rho_inf": {role: float(np.isfinite(np.asarray(list(v), dtype=float)).mean()) for role, v in costs.items()},
            "n_unsolved_by_all": int((~np.isfinite(matrix)).all(axis=0).sum())}


def _profiles(instances: Sequence[Mapping], records: Sequence[Mapping], selected: Iterable[str]) -> dict:
    """The profiles of each cell: PI⁺ over all its instances, the time to the target over those whose start does not
    meet the target (the others are counted in ``n_start_met``)."""
    cells: dict[str, dict] = {}
    for cell in _cell_order(selected):
        mine = [(i, r) for i, r in zip(instances, records, strict=True) if i["cell"] == cell]
        if not mine:
            continue
        timed = [(i, r) for i, r in mine if not r["start_met"]]
        pi = {key: [i["pi"][key] for i, _ in mine] for key in an.REPRESENTATIVES}
        ttt = {key: [math.inf if r["roles"][key]["censored"] else r["roles"][key]["median_s"] for _, r in timed]
               for key in an.REPRESENTATIVES}
        cells[cell] = {
            "n_instances": len(mine),
            "PI_plus": {"n_instances": len(mine), "n_start_met": 0, **profile_block(pi)},
            "TTT": ({"n_instances": len(timed), "n_start_met": len(mine) - len(timed), **profile_block(ttt)}
                    if timed else {"n_instances": 0, "n_start_met": len(mine), "rho": {}, "rho_inf": {},
                                   "n_unsolved_by_all": 0})}
    return {"ratios": list(PROFILE_RATIOS), "tau": TAU, "cells": cells}


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def _role_groups(rows: Iterable[Mapping]) -> dict[tuple, dict[str, list[Mapping]]]:
    """The rows of each instance by representative: ``{(cell, unit): {"exact": [...], "qubo": [...], "ls": [...]}}``."""
    groups: dict[tuple, dict[str, list[Mapping]]] = defaultdict(lambda: {key: [] for key in an.REPRESENTATIVES})
    for row in rows:
        for key, role in an.REPRESENTATIVES.items():
            if role in (row.get("roles") or ()):
                groups[(row.get("cell"), an._unit_of(row))][key].append(row)
    return groups


def _flipped(reports: Iterable[Mapping]) -> set[str]:
    """The ids of the selected instances that the screening lists as flipped by the scale of the optimum: screened
    non-trivial with the scale of the start, but with ``ĝ_start <= τ`` against the certified optimum (the
    ``reference_check`` of the eligibility report; A1 'Уточнения' (1): the decision stays)."""
    return {e["instance_id"] for report in reports for e in (report.get("reference_check") or {}).get("flipped", [])
            if e.get("class") == "nontrivial"}


def _base_record(inst: Mapping, exact: Mapping) -> dict:
    return {
        "cell": inst["cell"], "unit": inst["unit"], "cluster": inst["cluster"], "instance_id": inst["instance_id"],
        "n_assets": inst["n_assets"], "K": inst["K"], "s_cont": inst["s_cont"], "binding": inst["binding"],
        "ref_type": inst["ref_type"], "ref_source": inst["ref_source"], "ref_inconsistent": inst["ref_inconsistent"],
        "f_ref": inst["f_ref"], "s_ref": inst["s_ref"], "g_start": inst["g_start"], "t0_s": inst["t0_s"],
        "pi_plus": dict(inst["pi"]), "n_seeds": dict(inst["n_seeds"]), "n_over_budget": dict(inst["n_over_budget"]),
        **{k: exact[k] for k in ("t_primal_median_s", "t_cert_median_s", "t_cert_censored", "n_certified")},
    }


def secondary_report(
    rows: Iterable[Mapping], reports: Sequence[Mapping], *, references: Iterable[Mapping] = (),
    budget_s: float = BUDGET_S, n_boot: int = rq2.N_BOOT, seed: int = rq2.SEED,
) -> dict:
    """The secondary analyses (module docstring) from the rows of the bench stages, the finished eligibility reports
    of the test scan (``reports``: the test report and, for D1, the dense one) and the certificates
    (``references``). No I/O: :func:`run_command` adds the inputs and the code state to what this returns."""
    pilot = _pilot()
    rows, reports = list(rows), list(reports)
    selected, _, pooled = an._plan(reports)
    used, ignored = an._reference_rows(references)
    refs = pilot.reference_table([*rows, *used])
    instances = an.bench_instances(rows, reports, refs=refs, budget_s=budget_s)
    groups = _role_groups(rows)
    variant_args = {"instances": instances, "groups": groups, "refs": refs, "selected": selected, "pooled": pooled,
                    "start_met_ok": _flipped(reports), "n_boot": n_boot, "seed": seed}
    main = _ttt_variant("primary", f"τ = 1e-3, budget {budget_s:g} s", TAU, float(budget_s), **variant_args)
    sensitivity: dict[str, dict] = {}
    warnings: list[str] = []
    for name, (spec, definition) in SENSITIVITIES.items():
        censor_s = float(budget_s) if spec["censor_s"] is None else spec["censor_s"]
        if spec["censor_s"] is not None and censor_s >= budget_s:
            warnings.append(f"the sensitivity {name} is not computed: it needs a budget shorter than the {budget_s:g} s "
                            "of the rows")
            continue
        sensitivity[name] = _ttt_variant(name, definition, spec["tau"], censor_s, **variant_args)
    for block in (main, *sensitivity.values()):
        if block["start_met"]:
            ids = ", ".join(e["instance_id"] for e in block["start_met"][:5])
            warnings.append(
                f"{block['name']}: {len(block['start_met'])} instance(s) leave the TTT estimates because the common start "
                f"already meets the target τ = {block['tau']:g} (flipped by the scale of the optimum, listed in the "
                f"reference_check of the eligibility report): {ids}{' …' if len(block['start_met']) > 5 else ''}")
    exact = [exact_instance(groups[(i["cell"], i["unit"])]["exact"], budget_s=budget_s) for i in instances]
    exact_cells = {}
    for cell in _cell_order(selected):
        idx = [k for k, i in enumerate(instances) if i["cell"] == cell]
        exact_cells[cell] = _exact_cell(
            [exact[k] for k in idx], [r for k in idx for r in groups[(instances[k]["cell"], instances[k]["unit"])]["exact"]])
    family_size = max(1, len(pooled))
    prespecified = bool(n_boot == rq2.N_BOOT and seed == rq2.SEED and budget_s == BUDGET_S)
    if not prespecified:
        warnings.append(f"the settings (n_boot = {n_boot}, seed = {seed}, budget_s = {budget_s}) are not the "
                        f"prespecified ones (n_boot = {rq2.N_BOOT}, seed = {rq2.SEED}, budget_s = {BUDGET_S})")
    by_type = Counter(i["ref_type"] for i in instances)
    if by_type.get("best_known"):
        warnings.append(f"{by_type['best_known']} of {len(instances)} instances have no certified reference: the best "
                        "known value over the rows is their reference (§5)")
    flagged = [i["instance_id"] for i in instances if i["ref_inconsistent"]]
    if flagged:
        warnings.append(f"{len(flagged)} instance(s) have a method below the certified optimum (e.g. {flagged[0]}): "
                        "the best known value replaced the certificate; investigate")
    over = sum(sum(i["n_over_budget"].values()) for i in instances)
    if over:
        warnings.append(f"{over} rows overran the budget (budget_ok = false): used as measured, counted in `budget`")
    if ignored:
        warnings.append(f"{ignored} reference row(s) of other stages (or with an error) were ignored: only the stages "
                        f"{', '.join(an.REFERENCE_STAGES)} carry certificates")
    overruns = {cell: {key: sum(i["n_over_budget"][key] for i in instances if i["cell"] == cell)
                       for key in an.REPRESENTATIVES} for cell in dict.fromkeys(i["cell"] for i in instances)}
    descriptive = {cell: reason for report in reports
                   for cell, reason in ((report.get("summary") or {}).get("excluded") or {}).items()}
    return {
        "kind": "e2-secondary", "schema": "e2-secondary/1", "plan_version": "v2.4",
        "role": ANALYSIS, "confirmatory": False,
        "notice": "SECONDARY ANALYSIS, NOT CONFIRMATORY (ANALYSIS_PLAN §2 RQ2 'Вторично', §7.2): descriptive; no test "
                  "of the registry of §7.1; the outcomes are coded with the bands of H2 and carry no sentence of §10",
        "measures": {
            "ttt": "end-to-end time to ĝ <= τ (common start included), censored at the budget; Kaplan-Meier median "
                   "over the solver seeds per representative; ratios QUBO/exact (TTT) and QUBO/local search (TTTb)",
            "exact_times": "t_primal_s and t_cert_s of the rows of the exact representative",
            "profiles": "Dolan-Moré profiles over the instances of a cell on PI+ and on the time to the target"},
        "settings": {
            "budget_s": budget_s, "tau": TAU, "tau_sensitivity": TAU_SENSITIVITY,
            "censoring_budgets_s": list(CENSORING_BUDGETS_S), "n_boot": n_boot, "seed": seed, "level": rq2.POOLED_LEVEL,
            "family_level": rq2.FAMILY_LEVEL, "family_size": family_size, "cell_level": rq2.family_level(family_size),
            "band": [1.0 / rq2.EQUIVALENCE_FACTOR, rq2.EQUIVALENCE_FACTOR], "interval": "percentile",
            "profile_ratios": list(PROFILE_RATIOS), "prespecified": prespecified},
        "prespecified": PRESPECIFIED,
        "ttt": {"primary": main, "sensitivity": sensitivity},
        "exact_times": {"instances": [{"cell": i["cell"], "unit": i["unit"], "cluster": i["cluster"],
                                       "instance_id": i["instance_id"], **e}
                                      for i, e in zip(instances, exact, strict=True)],
                        "cells": exact_cells},
        "profiles": _profiles(instances, main["instances"], selected),
        "descriptive": descriptive,
        "statement": an.NO_POOLED_STATEMENT if not pooled and any(c in an.H2_CELLS for c in selected) else None,
        "references": {"n_instances": len(instances), "by_type": dict(by_type), "n_inconsistent": len(flagged),
                       "n_rows_used": len(used), "n_rows_ignored": ignored, "stages": list(an.REFERENCE_STAGES)},
        "budget": {"n_over_budget": overruns, "n_rows_over_budget": over},
        "instances": [_base_record(i, e) for i, e in zip(instances, exact, strict=True)],
        "warnings": warnings, "provenance_clean": None, "inputs": [], "code": None,
    }


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

_ROLE_FIELDS = ("n_seeds", "n_reached", "median_s", "censored", "value_s")
_PAIR_FIELDS = ("status", "bound", "ratio", "log_ratio")
_INSTANCE_COLUMNS = (
    "analysis", "cell", "unit", "cluster", "instance_id", "n_assets", "K", "s_cont", "binding", "ref_type",
    "ref_source", "ref_inconsistent", "g_start", "t0_s",
    *(f"pi_plus_{k}" for k in an.REPRESENTATIVES), *(f"n_seeds_{k}" for k in an.REPRESENTATIVES),
    *(f"n_over_budget_{k}" for k in an.REPRESENTATIVES),
    "t_primal_median_s", "t_cert_median_s", "t_cert_censored", "n_certified")
_TTT_INSTANCE_COLUMNS = (
    "analysis", "variant", "tau", "censor_s", "cell", "unit", "cluster", "instance_id", "start_met", "g_start",
    *(f"{field}_{k}" for k in an.REPRESENTATIVES for field in _ROLE_FIELDS),
    *(f"{field}_{pair}" for pair in PAIRS for field in _PAIR_FIELDS))
_ESTIMATE_COLUMNS = ("estimate_log", "ratio", "se", "ci_level", "ci_low_log", "ci_high_log", "ratio_low", "ratio_high",
                     "outcome")
_TTT_CELL_COLUMNS = ("analysis", "variant", "cell", "hypothesis", "in_pooled", "n_instances", "n_clusters",
                     *_ESTIMATE_COLUMNS)
_TTT_POOLED_COLUMNS = ("analysis", "variant", "hypothesis", "cells", "n_cells", *_ESTIMATE_COLUMNS)
_CENSORING_COLUMNS = (
    "analysis", "variant", "cell", "n_instances", "start_met", *(f"censored_{k}" for k in an.REPRESENTATIVES),
    *(f"{pair}_{status}" for pair in PAIRS for status in STATUSES))
_EXACT_COLUMNS = (
    "analysis", "cell", "n_instances", "configs",
    *(f"t_primal_{k}" for k in ("n", *QUANTILES)),
    *(f"t_cert_{k}" for k in ("n", "n_censored", *QUANTILES, "share_finite", "ci_low", "ci_high")),
    "n_runs", "n_certified_in_budget", "n_certified", "share_in_budget", "share_ci_low", "share_ci_high",
    *(f"t_primal_{kind}_{k}" for kind in ("certified", "not_certified") for k in ("n", *QUANTILES)))
_PROFILE_COLUMNS = ("analysis", "measure", "cell", "role", "ratio", "rho", "n_instances", "n_start_met",
                    "n_unsolved_by_all")


def _estimate_fields(part: Mapping) -> dict:
    low, high = part["ci_log"] or (None, None)
    return {"estimate_log": part["estimate_log"], "ratio": part["ratio"], "se": part["se"], "ci_level": part["level"],
            "ci_low_log": low, "ci_high_log": high, "ratio_low": None if low is None else math.exp(low),
            "ratio_high": None if high is None else math.exp(high), "outcome": part["outcome"]}


def _variants(report: Mapping) -> list[tuple[str, Mapping]]:
    return [("primary", report["ttt"]["primary"]), *report["ttt"]["sensitivity"].items()]


def _tables(report: Mapping) -> dict[str, pd.DataFrame]:
    instances = [{"analysis": ANALYSIS, **{k: v for k, v in r.items() if k not in ("unit", "pi_plus", "n_seeds",
                                                                                   "n_over_budget")},
                  "unit": an._unit_text(r["unit"]),
                  **{f"pi_plus_{k}": r["pi_plus"].get(k) for k in an.REPRESENTATIVES},
                  **{f"n_seeds_{k}": r["n_seeds"][k] for k in an.REPRESENTATIVES},
                  **{f"n_over_budget_{k}": r["n_over_budget"][k] for k in an.REPRESENTATIVES}}
                 for r in report["instances"]]
    ttt_instances_rows, cells, pooled, censoring = [], [], [], []
    for variant, block in _variants(report):
        for rec in block["instances"]:
            row = {"analysis": ANALYSIS, "variant": variant, "tau": block["tau"], "censor_s": block["censor_s"],
                   "cell": rec["cell"], "unit": an._unit_text(rec["unit"]), "cluster": rec["cluster"],
                   "instance_id": rec["instance_id"], "start_met": rec["start_met"], "g_start": rec["g_start"]}
            row.update({f"{field}_{k}": rec["roles"][k][field] for k in an.REPRESENTATIVES for field in _ROLE_FIELDS})
            row.update({f"{field}_{pair}": rec[pair][field] for pair in PAIRS for field in _PAIR_FIELDS})
            ttt_instances_rows.append(row)
        for pair, part in block["pooled"].items():
            if part:
                pooled.append({"analysis": ANALYSIS, "variant": variant, "hypothesis": pair,
                               "cells": " ".join(part["cells"]), "n_cells": part["n_cells"], **_estimate_fields(part)})
        for cell, cblock in block["cells"].items():
            for pair in PAIRS:
                part = cblock[pair]
                if part:
                    cells.append({"analysis": ANALYSIS, "variant": variant, "cell": cell, "hypothesis": pair,
                                  "in_pooled": part["in_pooled"], "n_instances": cblock["n_instances"],
                                  "n_clusters": part["n_clusters"], **_estimate_fields(part)})
            cen = cblock["censoring"]
            censoring.append({"analysis": ANALYSIS, "variant": variant, "cell": cell, "n_instances": cblock["n_instances"],
                              "start_met": cen["start_met"],
                              **{f"censored_{k}": v for k, v in cen["censored_instances"].items()},
                              **{f"{pair}_{status}": cen[pair][status] for pair in PAIRS for status in STATUSES}})
    exact = []
    for cell, c in report["exact_times"]["cells"].items():
        runs = c["runs"]
        row = {"analysis": ANALYSIS, "cell": cell, "n_instances": c["n_instances"], "configs": " ".join(c["configs"]),
               **{f"t_primal_{k}": v for k, v in c["t_primal_s"].items()},
               **{f"t_cert_{k}": v for k, v in c["t_cert_s"].items() if k != "ci95_share_finite"},
               "t_cert_ci_low": c["t_cert_s"]["ci95_share_finite"][0], "t_cert_ci_high": c["t_cert_s"]["ci95_share_finite"][1],
               "n_runs": runs["n_runs"], "n_certified_in_budget": runs["n_certified_in_budget"],
               "n_certified": runs["n_certified"], "share_in_budget": runs["share_in_budget"],
               "share_ci_low": runs["ci95_share_in_budget"][0], "share_ci_high": runs["ci95_share_in_budget"][1]}
        for kind, summary in runs["t_primal_s"].items():
            row.update({f"t_primal_{kind}_{k}": v for k, v in summary.items()})
        exact.append(row)
    profile_rows = []
    grid = [*report["profiles"]["ratios"], math.inf]
    for cell, c in report["profiles"]["cells"].items():
        for measure in ("PI_plus", "TTT"):
            block = c[measure]
            for role, values in block["rho"].items():
                for ratio, rho in zip(grid, [*values, block["rho_inf"][role]], strict=True):
                    profile_rows.append({"analysis": ANALYSIS, "measure": measure, "cell": cell, "role": role,
                                         "ratio": ratio, "rho": rho, "n_instances": block["n_instances"],
                                         "n_start_met": block["n_start_met"],
                                         "n_unsolved_by_all": block["n_unsolved_by_all"]})
    return {"instances.csv": pd.DataFrame(instances, columns=list(_INSTANCE_COLUMNS)),
            "ttt_instances.csv": pd.DataFrame(ttt_instances_rows, columns=list(_TTT_INSTANCE_COLUMNS)),
            "ttt_cells.csv": pd.DataFrame(cells, columns=list(_TTT_CELL_COLUMNS)),
            "ttt_pooled.csv": pd.DataFrame(pooled, columns=list(_TTT_POOLED_COLUMNS)),
            "ttt_censoring.csv": pd.DataFrame(censoring, columns=list(_CENSORING_COLUMNS)),
            "exact_times.csv": pd.DataFrame(exact, columns=list(_EXACT_COLUMNS)),
            "profiles.csv": pd.DataFrame(profile_rows, columns=list(_PROFILE_COLUMNS))}


def write_secondary(report: Mapping, out_dir: Path) -> list[Path]:
    """``secondary.json`` (the report) and the tables in ``out_dir``; every table starts with the column ``analysis``.
    The JSON is strict (no NaN: missing numbers are ``null``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "secondary.json"]
    written[0].write_text(
        json.dumps(e2_eligibility._json_safe(dict(report)), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    for name, frame in _tables(report).items():
        written.append(out_dir / name)
        frame.to_csv(written[-1], index=False)
    return written


# --------------------------------------------------------------------------------------------
# Command line (the arguments are defined in cif.experiments.e2_analysis)
# --------------------------------------------------------------------------------------------


def run_command(args) -> int:
    """``e2_analysis secondary``: read the registered runs, write the report and the tables, print the pooled
    estimates (marked secondary)."""
    pilot = _pilot()
    files = [args.eligibility, *([args.eligibility_dense] if args.eligibility_dense else [])]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    rows, errors = pilot.load_rows(args.rows)
    report = secondary_report([*rows, *errors], reports, references=pilot.reference_rows(args.reference))
    inputs = [entry for role, paths in (("rows", args.rows), ("reference", args.reference))
              for entry in an._run_inputs(paths, role)]
    inputs += [{**pilot._file_entry(path), "role": "eligibility"} for path in files]
    state = registry._git_state()
    code = {"git_sha": state.get("sha") or None, "git_dirty": state.get("dirty"), "branch": state.get("branch") or None}
    problems = an._provenance_warnings(inputs, code)
    given = {entry["run_id"] for entry in inputs if entry.get("run_id")}
    for path, built in zip(files, reports, strict=True):
        absent = sorted({e["run_id"] for e in built.get("inputs") or [] if isinstance(e, Mapping) and e.get("run_id")}
                        - given)
        if absent:
            problems.append(f"{path.name} was built from the runs {absent}, which are not among the inputs: the "
                            "certificates of their instances are missing from the references")
    versions = {"python": sys.version.split()[0], **{p: metadata.version(p) for p in ("numpy", "pandas", "scipy")}}
    report.update(inputs=inputs, code=code, versions=versions, warnings=[*problems, *report["warnings"]],
                  provenance_clean=bool(not problems and report["settings"]["prespecified"]))
    write_secondary(report, args.out)
    main = report["ttt"]["primary"]
    print(f"e2-secondary: {args.out}: SECONDARY analyses, not confirmatory; {len(report['instances'])} instances in "
          f"{len(main['cells'])} cells")
    if not any(main["pooled"].values()):
        print(f"  no pooled TTT estimate: {report['statement'] or 'no cell of RQ2 in the reports'}")
    for pair, part in main["pooled"].items():
        if part:
            low, high = part["ci_ratio"]
            print(f"  {pair} (tau = {main['tau']:g}, censored at {main['censor_s']:g} s), pooled over "
                  f"{', '.join(part['cells'])} (equal weights): ratio = {part['ratio']:.4g} "
                  f"({100 * part['level']:.0f} % CI {low:.4g} to {high:.4g}), outcome {part['outcome']}")
    for note in main["notes"]:
        print(f"NOTE: {note}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0
