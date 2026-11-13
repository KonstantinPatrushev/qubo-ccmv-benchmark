"""Confirmatory analysis of E2 from registered rows: H1c (ANALYSIS_PLAN §2 RQ1, §7.1; §12 A5 of 2026-10-11).

H1c is the held-out replication of the exploratory finding "tabu beats SA at equal budget": at 10 s end-to-end
and equal tuning (the roles ``h1c_tabu`` and ``h1c_sa``), the incumbent of tabu at the budget is lower than that of
SA. The unit is the **instance**, the instances are the **non-trivial test instances of C1–C3** that the screening
selected (A1: on a trivial instance both pipelines stand at the start or at the optimum, and there is nothing to
compare), the common start is in the pool as in §5.

* the final value of a method on an instance is the median over its seeds of ``f_at_budget`` (10 seeds);
* the difference is ``d = F_tabu − F_SA``, **normalised by the scale** ``s*`` of the instance, i.e. the difference
  of the final gaps ``ĝ`` of §5 (``ĝ_tabu − ĝ_SA = d/s*``): the instances of a stratum differ in scale, and the
  plan speaks of the differences of the final *gap*. ``s*`` is the scale of the reference when ``scales`` gives it
  (``cif.experiments.e2_pilot.reference_table``) and the scale of the common start otherwise; the certificate
  itself is not needed (§2: ``F_tabu − F_SA`` does not depend on ``f*``);
* differences with ``|F_tabu − F_SA| ≤ 10⁻⁸·s*`` are zeros and are **dropped before ranking**
  (``zero_method = wilcox``, as ``zero_tol`` of H1a, :func:`cif.stats.rq1.stratified_signed_rank`);
* the test is the one-sided Wilcoxon signed-rank test (tabu lower), stratified by N, exact up to 500 non-zero
  differences. Its p-value is what the Holm family {H1a, H1c} reads (:func:`cif.stats.rq1.holm_family`,
  ``--h1c-p`` of :mod:`cif.experiments.e1_analysis`).

The report also gives the counts of the three directions, the median normalised difference, the cells that are
descriptive for H2 (fewer than 20 non-trivial instances: H1c still uses all of them, it pools over strata) and the
inputs. Rows of an instance that was not selected stop the analysis: no trivial instance enters quietly.

**H2 and H2b** (ANALYSIS_PLAN §2 RQ2 "Подтверждающая оценка (N8)", §7.1, §7.2, §10; §12 A1–A4; audit G1 RQ2-1) are
the confirmatory estimate of RQ2: ``python -m cif.experiments.e2_analysis h2``. Per non-trivial test instance
``R = PI⁺(QUBO)/PI⁺(exact)`` and ``R_b = PI⁺(QUBO)/PI⁺(LS)`` of the three representatives
(roles ``exact_rep``, ``qubo_rep``, ``ls``; medians over their solver seeds; ``PI⁺`` after the common start, 60 s,
:func:`cif.experiments.e2_pilot.row_pi` against the reference of
:func:`cif.experiments.e2_pilot.reference_table`); the pooled mean of ``log R`` over the cells of
``summary.cells`` of the eligibility report with equal cell weights, C5 as the mean over ports of the means over λ,
its 95 % stratified-bootstrap interval, the cell intervals (Bonferroni, simultaneously 99 %), the five outcomes
(:mod:`cif.stats.rq2`), the estimate of A3 (the share of trivial instances per cell, read from the eligibility
report) and the sensitivities of A4. Everything the plan leaves open is fixed in :data:`PRESPECIFIED` and written
into the JSON next to the numbers; none of it is an option of the command line. A pooled cell with fewer than
:data:`MIN_CLUSTERS` clusters (C5 with its port clusters) is named in the warnings, with the pooled estimate without it
as a post-hoc check (audit G2, C-1; §12 of 2026-10-17): its percentile interval is then little more than the range of
its cluster means.

The **secondary analyses of RQ2** (§2 "Вторично": time-to-target with the same outcomes, ``t_primal`` and ``t_cert`` of
the exact class, performance profiles) are ``python -m cif.experiments.e2_analysis secondary``, implemented in
:mod:`cif.experiments.e2_secondary` on the same inputs and the same reference as ``h2``. They are descriptive and not
confirmatory, and every output is flagged so. So are the **TTT curves** of the stage ``ttt`` and the **delay to the
first own solution** on the stage ``cold`` (§12 of 2026-10-17): ``... e2_analysis ttt-curves`` and ``... delay``,
implemented in :mod:`cif.experiments.e2_curves`.

**The validation M11** (§12, 2026-10-17): ``python -m cif.experiments.e2_analysis m11``, implemented in
:mod:`cif.experiments.e2_m11_analysis`: the rows of the stage ``m11`` (:mod:`cif.experiments.e2_m11`) against the
published times of Bertsimas & Cory-Wright, Tables 3 and 4: the ratio of our median time to the best published one
per instance, the median ratio and the number of instances solved per table, and the verdict of the plan's rule
(median above 10). It tests no hypothesis and reads no test seed.
"""

from __future__ import annotations

import argparse
import hashlib
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

from cif.experiments import e2_eligibility, registry
from cif.stats import rq2
from cif.stats.rq1 import stratified_signed_rank

__all__ = [
    "BUDGET_S",
    "H1C_CELLS",
    "H1C_ZERO_TOL",
    "H2_CELLS",
    "PRESPECIFIED",
    "REPRESENTATIVES",
    "a3_estimates",
    "bench_instances",
    "h1c_differences",
    "h1c_test",
    "h2_report",
    "main",
    "write_h2",
]

#: Differences within this part of the scale are zeros of H1c (§12 A5), as ``zero_tol`` of H1a.
H1C_ZERO_TOL = 1e-8
H1C_CELLS = ("C1", "C2", "C3")
_METHODS = ("tabu", "sa")


def _selected(eligibility: Mapping) -> dict[str, list[int]]:
    return {cell: [int(s) for s in e2_eligibility.selected_units(eligibility, cell)] for cell in H1C_CELLS}


def h1c_differences(
    rows: Iterable[Mapping], eligibility: Mapping, *, scales: Mapping[str, float] | None = None,
) -> list[dict]:
    """One dict per non-trivial test instance of C1–C3: the final values of tabu and SA (medians over their
    seeds), the difference ``F_tabu − F_SA``, its normalisation by the scale and the stratum (N). Rows must be
    those of the stage ``h1c``; every selected instance needs rows of both methods with the same number of seeds,
    and no row may belong to an instance that was not selected."""
    selected = _selected(eligibility)
    wanted = {(cell, seed) for cell, seeds in selected.items() for seed in seeds}
    by_instance: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row.get("stage") != "h1c" or row.get("cell") not in H1C_CELLS or row.get("skipped"):
            continue
        if row.get("error"):
            raise ValueError(f"a row carries an error ({row['error']}): fix and rerun the cell")
        key = (row["cell"], int(row["seed"]))
        if key not in wanted:
            raise ValueError(f"rows of {row.get('instance_id')} in {row['cell']}, which the eligibility report did "
                             "not selected: a trivial or unscreened instance must not enter H1c")
        name = row["method"]["name"]
        if name not in _METHODS:
            raise ValueError(f"H1c compares {_METHODS}, found rows of {name!r}")
        entry = by_instance.setdefault(key, {"instance_id": row["instance_id"], "n": int(row["n_assets"]),
                                             "scale": float(row["start_scale"]), "f": defaultdict(list)})
        if row.get("f_at_budget") is None:
            raise ValueError(f"{row['instance_id']}: no incumbent at the budget in a row of {name}")
        entry["f"][name].append(float(row["f_at_budget"]))
    out = []
    for cell, seed in sorted(wanted):
        entry = by_instance.get((cell, seed))
        if entry is None:
            raise ValueError(f"no rows of the stage h1c for the selected instance {cell}/{seed}")
        f = entry["f"]
        if set(f) != set(_METHODS):
            raise ValueError(f"{entry['instance_id']}: H1c needs the rows of both methods, found {sorted(f)}")
        if len(f["tabu"]) != len(f["sa"]):
            raise ValueError(f"{entry['instance_id']}: tabu has {len(f['tabu'])} seeds and SA {len(f['sa'])}: the "
                             "comparison is at an equal number of seeds")
        scale = float((scales or {}).get(entry["instance_id"], entry["scale"]))
        f_tabu, f_sa = float(np.median(f["tabu"])), float(np.median(f["sa"]))
        out.append({
            "cell": cell, "seed": seed, "instance_id": entry["instance_id"], "stratum": entry["n"],
            "f_tabu": f_tabu, "f_sa": f_sa, "difference": f_tabu - f_sa, "scale": scale,
            "normalised": (f_tabu - f_sa) / scale, "n_seeds": {"tabu": len(f["tabu"]), "sa": len(f["sa"])},
        })
    return out


def h1c_test(
    rows: Iterable[Mapping], eligibility: Mapping, *, scales: Mapping[str, float] | None = None,
    zero_tol: float = H1C_ZERO_TOL,
) -> dict:
    """The H1c test and its report (module docstring)."""
    diffs = h1c_differences(rows, eligibility, scales=scales)
    if not diffs:
        raise ValueError("no non-trivial test instance of C1-C3 to test on")
    d = np.array([x["normalised"] for x in diffs])
    strata = np.array([x["stratum"] for x in diffs])
    res = stratified_signed_rank(d, strata, alternative="less", zero_tol=zero_tol)
    tol = float(zero_tol)
    direction = {"tabu_better": int((d < -tol).sum()), "sa_better": int((d > tol).sum()),
                 "zero": int((np.abs(d) <= tol).sum())}
    cells = eligibility["cells"]
    return {
        "kind": "e2-h1c", "plan_version": "v2.4",
        "alternative": res.alternative, "method": res.method, "statistic": res.statistic,
        "p_value": res.p_value, "z": res.z, "n_instances": res.n_total, "n_nonzero": res.n_nonzero,
        "n_zero": res.n_zero, "zero_tol": res.zero_tol, "direction": direction,
        "strata": {str(k): v for k, v in sorted(res.strata.items())},
        "per_cell": {cell: sum(1 for x in diffs if x["cell"] == cell) for cell in H1C_CELLS},
        "descriptive_cells": [c for c in H1C_CELLS if not cells[c]["enough"]],
        "median_normalised_difference": float(np.median(d)),
        "scale": "reference" if scales else "start",
        "differences": diffs, "inputs": [],
    }


# --------------------------------------------------------------------------------------------
# H2 and H2b: the confirmatory estimate of RQ2
# --------------------------------------------------------------------------------------------

#: The cells of RQ2 (§3). C4 is descriptive by the λ rule of N7, D1 (``D1_100``, ``D1_200``) by design.
H2_CELLS = ("C1", "C2", "C3", "C4", "C5")
#: The budget of the primary analysis of RQ2 (§6); another budget is exploratory (§9).
BUDGET_S = 60.0
#: The representatives of the three classes: the roles of ``tuned.json`` (:mod:`cif.experiments.e2_benchmark`).
REPRESENTATIVES = {"exact": "exact_rep", "qubo": "qubo_rep", "ls": "ls"}
_METRIC = {"H2": "log_r", "H2b": "log_r_b"}
#: Rows of these stages carry certificates (``f_star``); they are the references besides the bench rows.
REFERENCE_STAGES = ("eligibility", "reference")
#: §12 A3: the answer of RQ2 when no cell enters the pooled estimate.
NO_POOLED_STATEMENT = "Нетривиальных инстансов недостаточно для сводной оценки; RQ2 отвечает долей тривиальных"
#: §12 A3: the proviso every H2 / H2b row of §10 gets.
A3_PROVISO = "на инстансах, где общий старт не достигает τ"
#: The wording of §10 for each outcome of H2 (against the exact method) and H2b (against local search).
STATEMENTS = {
    "H2": {
        "qubo_better": "QUBO-конвейер быстрее приближается к оптимуму, чем точный метод",
        "qubo_worse": "Точный метод не уступает и превосходит QUBO-конвейер",
        "equivalent": "Практически эквивалентны (в пределах ×1.5)",
        "direction_only": "Различие есть, но его величина относительно полосы ×1.5 не определена",
        "undetermined": "Данные не позволяют различить методы",
    },
    "H2b": {
        "qubo_better": "QUBO-конвейер быстрее приближается к оптимуму, чем локальный поиск",
        "qubo_worse": "Локальный поиск не уступает и превосходит QUBO-конвейер",
        "equivalent": "Практически эквивалентны (в пределах ×1.5)",
        "direction_only": "Различие есть, но его величина относительно полосы ×1.5 не определена",
        "undetermined": "Данные не позволяют различить методы",
    },
}

#: What the frozen plan leaves open and how the analysis reads it (written into the JSON of every report; the
#: orchestrator records the same in §12 before the bench stages). ``choice`` is what the code does, ``why`` the
#: plan text or the convention of the project it follows.
PRESPECIFIED = {
    "reference": {
        "choice": "The reference of an instance is the best value known over all bench rows of the instance (every "
                  "method, every seed, the common start) and the certificates of the screening and reference runs "
                  "(rows of those stages with f_star); a certificate stays the reference unless some row is below it "
                  "by more than 1e-6·s: then the best known value is the reference, the instance is flagged "
                  "(ref_inconsistent) and counted. An instance without a certificate keeps its best known value.",
        "why": "§5 'опора — лучшее известное значение по всем методам', 'несертифицированные инстансы в RQ2 не "
               "исключаются'; the rule of e2_pilot.reference_table that the tuning and the n rule used.",
    },
    "unit": {
        "choice": "Per instance and representative (exact_rep, qubo_rep, ls): the median over its solver seeds of "
                  "PI+ at 60 s; R = PI+(qubo)/PI+(exact), R_b = PI+(qubo)/PI+(ls); natural logarithms; inside a cell "
                  "every instance (every port in C5) has the weight of one.",
        "why": "§2: 'для стохастических методов берётся медиана по сидам'; §12 A2 (PI+); the computation of "
               "e2_pilot.n_rule_report.",
    },
    "c5": {
        "choice": "In C5 the cluster is the port: the log ratios of the λ points of a port are averaged with equal "
                  "weights, the cell mean is the mean over the ports that have at least one selected point (equal "
                  "port weights); a port without a selected point is absent, not imputed; the bootstrap redraws "
                  "ports with all their selected points.",
        "why": "§2: 'ORlib внутри C5 — среднее по port из средних по λ', 'port в C5'; §12 A1: C5 enters with "
               "points in at least three ports.",
    },
    "pooled_cells": {
        "choice": "The pooled estimate is the mean of the cell means with equal weights over the cells that "
                  "summary.cells of the finished eligibility report lists (C1, C2, C3, C5; C4 only if the λ rule "
                  "admits it, D1 never). A cell with fewer than 20 non-trivial test instances (C5: fewer than 10 "
                  "points or fewer than 3 ports) is descriptive: estimated and shown, not pooled. With no pooled "
                  "cell there is no pooled estimate and the answer is the share of trivial instances.",
        "why": "§2, §3 (N7), §12 A1 and A3.",
    },
    "bootstrap": {
        "choice": "Percentile interval of a stratified bootstrap (clusters redrawn with replacement within each "
                  "cell), B = 9999, seed 60000, one SeedSequence stream [seed, 2, crc32(cell)] per cell; the same "
                  "resamples serve H2 and H2b, the cells and the pooled estimate.",
        "why": "The plan says only 'стратифицированным бутстрапом'; the other interval estimates of the project "
               "(rq1.cluster_bootstrap, the E1 analysis) are percentile intervals with B = 9999 and explicit seeds.",
    },
    "bonferroni": {
        "choice": "The cell intervals are adjusted over the m cells of the pooled estimate: level 1 − 0.01/m "
                  "(99.75 % for C1, C2, C3, C5), percentile, from the same resamples. Descriptive cells (C4, D1, "
                  "cells with too few instances) are shown at the same level and are outside the family.",
        "why": "§2: 'Ячейки — описательно, с одновременными 99 % ДИ (Бонферрони)'; the family is the pooled set, "
               "so it does not depend on which descriptive stages were run.",
    },
    "outcome": {
        "choice": "The 95 % interval of the pooled mean log R, exponentiated, against the closed band [1/1.5, 1.5]: "
                  "better if its upper end is below 1/1.5, worse if its lower end is above 1.5, equivalent if it lies "
                  "inside the band (ends included, also when it excludes 1), direction only if it excludes 1 and is "
                  "neither inside nor wholly outside, undetermined otherwise. The cells get the same mapping from "
                  "their 99 % intervals (descriptive).",
        "why": "§2 'Исходы H2 и H2b', §10; the order of the tests makes the five outcomes exclusive and exhaustive.",
    },
    "budget_overrun": {
        "choice": "Rows with budget_ok = false are used as measured (PI+ counts what the method had delivered by T; "
                  "events after T do not count); their number is reported per cell and role, and the sensitivity "
                  "within_budget drops them (an instance left without a row of a representative is left out of it).",
        "why": "§5: the time is end-to-end and charged to the method; an overrun cannot help a method (what it finds "
               "after T does not count) and dropping such rows would drop the instances on which the exact method "
               "is slow, a bias in its favour; the tuning criterion treated these rows the same way (in the 10 s "
               "tuning on the calibration set 97 of 336 SCIP rows and 3 SA rows overran: the SDP of the perspective "
               "variant, the symmetry detection of SCIP).",
    },
    "degenerate": {
        "choice": "The analysis stops, naming the instance, on: PI+ = 0 or not finite for a representative, rows of "
                  "an instance the screening did not select, a selected instance without rows, a missing "
                  "representative, different numbers of seeds across the instances of a cell, several configurations "
                  "for one role in one cell and size, rows with an error, skipped rows, rows of another budget. No "
                  "floor is applied.",
        "why": "§12 A1/A2: on a non-trivial instance PI+ > 0 for every method; a 0/0 ratio would be settled by the "
               "composition of the instances (e2_pilot.n_rule_report refuses it the same way).",
    },
    "a3": {
        "choice": "The estimate of A3 is read from the finished eligibility reports (per cell: the share of trivial "
                  "instances among those scanned with a Wilson 95 % interval, ĝ_start in bp CE, t_cert on the "
                  "trivial ones) and put into the sentence of §10; it is no part of the pooled estimate.",
        "why": "§12 A3 and 'Уточнения A1' (6).",
    },
    "sensitivity": {
        "choice": "Sensitivities, not confirmatory, with the estimator and the resamples of the primary: full (PI "
                  "over [0, T], before A2), first_improvement (A4 (1)), within_budget, and own (A4 (2)) from the "
                  "cold rows when they are given.",
        "why": "§12 A4.",
    },
}

#: The sensitivities computed from the rows of the bench stages: name → (arguments of bench_instances, what it is).
SENSITIVITIES = {
    "full": ({"variant": "full"}, "PI over [0, T], the definition before A2 (the common term t0/T included)"),
    "first_improvement": ({"variant": "first_improvement"},
                          "γ = 1 until the first improvement of the common start (A4 (1))"),
    "within_budget": ({"drop_over_budget": True}, "the rows with budget_ok = false dropped"),
}
_SENSITIVITY_ROLE = "sensitivity, not the confirmatory outcome"

#: A pooled cell with fewer clusters is named in the warnings of ``h2`` (audit G2, C-1; §12 of 2026-10-17): the
#: stratified bootstrap of a cell of three clusters draws its mean from ten multisets of the three cluster means.
MIN_CLUSTERS = 5
_POST_HOC_ROLE = "post hoc check (audit G2, C-1; §12 of 2026-10-17), not the confirmatory outcome"


def _pilot():
    """:mod:`cif.experiments.e2_pilot` (it pulls in the solvers): imported when a command needs it."""
    from cif.experiments import e2_pilot

    return e2_pilot


def _unit_of(row: Mapping) -> int | tuple[int, int]:
    """The instance of a row: its seed, or ``(port, λ index)`` for an ORlib point."""
    if row.get("port") is not None:
        return int(row["port"]), int(row["lam_index"])
    return int(row["seed"])


def _unit(value: Any) -> int | tuple[int, int]:
    if isinstance(value, list | tuple):
        port, index = value
        return int(port), int(index)
    return int(value)


def _unit_text(unit: int | tuple[int, int]) -> str:
    return f"{unit[0]}:{unit[1]}" if isinstance(unit, tuple) else str(unit)


def _check_report(report: Mapping) -> None:
    if report.get("kind") != e2_eligibility.REPORT_KIND:
        raise ValueError(f"expected an {e2_eligibility.REPORT_KIND!r} report, its kind is {report.get('kind')!r}")
    if report.get("stub"):
        raise ValueError("the eligibility report is a stub (no round of the scan has been evaluated): the analysis "
                         "needs the report of the finished test scan")
    if report.get("role") != "test":
        raise ValueError(f"the analysis needs the report of the test scan, its role is {report.get('role')!r}")


def _plan(reports: Sequence[Mapping]) -> tuple[dict[str, list], dict[str, Mapping], list[str]]:
    """``(selected, states, pooled)`` of finished eligibility reports of the test role: the selected units of every
    cell, the state of every cell, and the cells of the pooled estimate (``summary.cells``), checked against the
    number of instances each has (``enough``)."""
    selected: dict[str, list] = {}
    states: dict[str, Mapping] = {}
    members: set[str] = set()
    for report in reports:
        _check_report(report)
        summary = report.get("summary")
        if not isinstance(summary, Mapping) or "cells" not in summary:
            raise ValueError("the eligibility report has no summary of the pooled cells (a finished report of the "
                             "test scan has one)")
        for cell in report["cells"]:
            if cell in selected:
                raise ValueError(f"the cell {cell!r} is in two eligibility reports")
            selected[cell] = [_unit(u) for u in e2_eligibility.selected_units(report, cell)]
            states[cell] = report["cells"][cell]
        members.update(summary["cells"])
    for cell in sorted(members):
        if cell not in H2_CELLS:
            raise ValueError(f"{cell!r} cannot be in the pooled set: only {', '.join(H2_CELLS)} can (D1 is "
                             "descriptive by design)")
        if cell not in states:
            raise ValueError(f"the pooled cell {cell!r} is in no eligibility report")
        if not states[cell].get("enough"):
            raise ValueError(f"the pooled cell {cell!r} does not have enough non-trivial instances "
                             "(enough = false in the report)")
    for cell in ("C1", "C2", "C3", "C5"):
        if cell in states and states[cell].get("enough") and cell not in members:
            raise ValueError(f"{cell} has enough non-trivial instances but the summary of the report leaves it out "
                             "of the pooled set")
    return selected, states, [c for c in H2_CELLS if c in members]


def _check_row(row: Mapping, stage: str, budget_s: float) -> None:
    where = row.get("instance_id") or row.get("cell_key")
    if row.get("error"):
        raise ValueError(f"a row carries an error ({row['error']}): fix and rerun the cell")
    if row.get("stage") != stage:
        raise ValueError(f"{where}: this analysis reads rows of the stage {stage!r}, found one of {row.get('stage')!r}")
    if row.get("skipped"):
        raise ValueError(f"{where}: a skipped row ({row['skipped']}) for an instance of the analysis")
    if float(row["budget_s"]) != float(budget_s):
        raise ValueError(f"{where}: a row has budget_s = {row['budget_s']} s, the analysis is at {budget_s} s (one "
                         "budget for all rows; another budget is exploratory, §9)")


def _reference_rows(references: Iterable[Mapping]) -> tuple[list[Mapping], int]:
    """The rows of the reference stages without an error, and the number of rows ignored."""
    references = list(references)
    used = [r for r in references if r.get("stage") in REFERENCE_STAGES and not r.get("error")]
    return used, len(references) - len(used)


def bench_instances(
    rows: Iterable[Mapping], reports: Sequence[Mapping], *, references: Iterable[Mapping] = (),
    refs: Mapping | None = None, budget_s: float = BUDGET_S, variant: str = "plus", drop_over_budget: bool = False,
    qubo_role: str = REPRESENTATIVES["qubo"], exact_config: str | None = None,
) -> list[dict]:
    """One dict per selected instance of the cells of ``reports``, in the order of the reports: the medians over the
    seeds of the primal integral of the three representatives (:data:`REPRESENTATIVES`) against the reference, and
    the ratios ``R`` and ``R_b`` with their logarithms.

    ``rows`` are the rows of the stage ``bench`` (the stage ``cold`` for ``variant="own"``): every selected instance
    needs rows of the three roles, no row may belong to an instance that was not selected, all have ``budget_s``
    (``PRESPECIFIED["degenerate"]`` lists what stops the analysis). The reference of an instance is, unless
    ``refs`` (a table of :func:`cif.experiments.e2_pilot.reference_table`) is given, that table over ``rows`` and the
    certificates among ``references``. ``variant`` is a primal-integral variant of
    :data:`cif.experiments.e2_pilot.PI_VARIANTS` (``"plus"``, PI⁺, is the primary measure); ``drop_over_budget`` leaves
    out the rows with ``budget_ok = false`` (an instance without a row of a role is then ``excluded``). ``qubo_role``
    puts another tuned role of the same runs (``sa``, ``tabu``) in the place of the QUBO representative: a post hoc
    description (§12, 2026-10-19 (2)), never the confirmatory estimate. ``exact_config`` takes the rows of that
    exact configuration (by ``config_id``, with or without a role, e.g. from a later run) in the place of the exact
    representative: the sensitivity to the exact baseline of §12, 2026-10-19, never the confirmatory estimate."""
    pilot = _pilot()
    representatives = {**REPRESENTATIVES, "qubo": qubo_role}
    if exact_config is not None:
        representatives["exact"] = f"config:{exact_config}"
    if variant not in pilot.PI_VARIANTS:
        raise ValueError(f"variant must be one of {pilot.PI_VARIANTS}, got {variant!r}")
    stage = "cold" if variant == "own" else "bench"
    selected, _, _ = _plan(reports)
    wanted = {cell: set(units) for cell, units in selected.items()}
    rows = list(rows)
    by_instance: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        _check_row(row, stage, budget_s)
        cell, unit = row.get("cell"), _unit_of(row)
        if cell not in wanted:
            raise ValueError(f"rows of the cell {cell!r} ({row.get('instance_id')}), which no eligibility report "
                             "lists")
        if unit not in wanted[cell]:
            raise ValueError(f"rows of {row.get('instance_id')} in {cell}: the instance is not selected by the "
                             "eligibility report (a trivial or unscreened instance must not enter H2)")
        by_instance[(cell, unit)].append(row)
    if refs is None:
        used, _ = _reference_rows(references)
        refs = pilot.reference_table([*rows, *used])
    # first the structure of the grid of every selected instance, then the numbers: a missing role or seed is reported
    # before a ratio that cannot be formed
    configs: dict[tuple, set] = defaultdict(set)
    seeds: dict[tuple, set] = defaultdict(set)
    grid: list[tuple] = []
    for cell, units in selected.items():
        for unit in units:
            where = f"{cell}/{_unit_text(unit)}"
            group = by_instance.get((cell, unit))
            if not group:
                raise ValueError(f"no rows of the stage {stage!r} for the selected instance {where}")
            ids = {r.get("instance_id") for r in group}
            if len(ids) != 1:
                raise ValueError(f"{where}: rows of several instances ({sorted(map(str, ids))})")
            (iid,) = ids
            if refs.get(iid) is None:
                raise ValueError(f"{where}: no reference for {iid}")
            chosen: dict[str, list[Mapping]] = {}
            for key, role in representatives.items():
                if key == "exact" and exact_config is not None:
                    picked = [r for r in group if r.get("config_id") == exact_config]
                else:
                    picked = [r for r in group if role in (r.get("roles") or ())]
                if not picked:
                    raise ValueError(f"no rows of the role {role!r} on the selected instance {where}")
                for r in picked:
                    configs[(cell, int(r.get("config_size") or 0), role)].add(r.get("config_id"))
                seeds[(cell, role)].add(len(picked))
                chosen[key] = picked
            grid.append((cell, unit, where, iid, group, chosen))
    for (cell, size, role), ids in sorted(configs.items(), key=str):
        if len(ids) > 1:
            raise ValueError(f"cell {cell}, size {size}: {len(ids)} configurations ({sorted(map(str, ids))}) for "
                             f"the role {role!r}: the rows are not those of one tuned report")
    for (cell, role), counts in sorted(seeds.items()):
        if len(counts) > 1:
            raise ValueError(f"cell {cell}: the instances have different numbers of seeds for the role {role!r} "
                             f"({sorted(counts)}): the grid is incomplete or mixed")
    return [_instance_record(cell, unit, where, iid, group, chosen, refs[iid], budget_s=budget_s, variant=variant,
                             drop_over_budget=drop_over_budget)
            for cell, unit, where, iid, group, chosen in grid]


def _instance_record(cell: str, unit, where: str, iid: str, group: Sequence[Mapping],
                     chosen: Mapping[str, Sequence[Mapping]], ref, *, budget_s: float, variant: str,
                     drop_over_budget: bool) -> dict:
    """The numbers of one instance (:func:`bench_instances`): PI of each representative, the ratios, the reference."""
    pilot = _pilot()
    label = "PI+" if variant == "plus" else f"PI[{variant}]"
    n_over = {key: sum(1 for r in picked if r.get("budget_ok") is False) for key, picked in chosen.items()}
    pi: dict[str, float] = {}
    n_kept: dict[str, int] = {}
    excluded = None
    for key, picked in chosen.items():
        kept = [r for r in picked if not (drop_over_budget and r.get("budget_ok") is False)]
        n_kept[key] = len(kept)
        if not kept:
            excluded = excluded or f"no {key} row inside the budget"
            continue
        values = []
        for r in kept:
            try:
                values.append(pilot.row_pi(r, ref, budget_s=budget_s, variant=variant))
            except ValueError as exc:
                raise ValueError(f"{where} ({iid}), {key}: {exc}") from exc
        value = float(np.median(values))
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"{label} = 0 or not finite on {where} ({iid}) for the {key} representative ({value!r}): the common "
                "start is optimal there, so the instance is not non-trivial (A1) and the ratio is 0/0; no floor is "
                "applied")
        pi[key] = value
    complete = excluded is None
    r_ratio = pi["qubo"] / pi["exact"] if complete else None
    rb_ratio = pi["qubo"] / pi["ls"] if complete else None
    used_rows = [r for picked in chosen.values() for r in picked]
    starts = [float(r["start_time_s"]) for r in used_rows if r.get("start_time_s") is not None]
    gaps = [g for g in (pilot.start_gap(r, ref) for r in used_rows) if math.isfinite(g)]
    first = group[0]
    return {
        "cell": cell, "unit": unit, "cluster": f"port{unit[0]}" if isinstance(unit, tuple) else str(unit),
        "instance_id": iid, "n_assets": first.get("n_assets"), "K": first.get("K"), "s_cont": first.get("s_cont"),
        "binding": first.get("binding"), "lam": first.get("lam"), "f_ref": ref.f_ref, "s_ref": ref.s_ref,
        "ref_type": ref.ref_type, "ref_source": ref.source, "ref_inconsistent": ref.inconsistent,
        "g_start": min(gaps) if gaps else None, "t0_s": float(np.median(starts)) if starts else None,
        "config_ids": {key: picked[0].get("config_id") for key, picked in chosen.items()},
        "n_seeds": n_kept, "n_over_budget": n_over, "pi": pi, "r": r_ratio, "r_b": rb_ratio,
        "log_r": math.log(r_ratio) if complete else None, "log_r_b": math.log(rb_ratio) if complete else None,
        "excluded": excluded,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.0f}"


def _a3_sentence(cell: str, est: Mapping) -> str | None:
    """The sentence §12 A3 adds to §10 for a cell, from its estimate; ``None`` if nothing was scanned."""
    share = est.get("share_trivial")
    if share is None:
        return None
    low, high = est["ci95"]
    return (f"На {_pct(share)} % ({_pct(low)}–{_pct(high)} %) инстансов ячейки {cell} округление непрерывной "
            "релаксации уже в пределах τ от сертифицированного оптимума: качество решателя в выборке там не важно; "
            "H2 и H2b относятся к остальным инстансам")


def a3_estimates(reports: Sequence[Mapping]) -> dict:
    """The estimate of §12 A3 per cell of the eligibility reports, as the screening recorded it (the instances
    scanned up to the n-th non-trivial one, undecided ones in the denominator): the share of trivial instances with
    its Wilson interval, the distribution of ``ĝ_start`` in bp CE, ``t_cert`` on the trivial ones, and the sentence
    of §10. ``groups`` are the pooled estimates of grouped cells (D1)."""
    keys = ("n_scanned", "n_trivial", "n_nontrivial", "n_undecided", "n_nonbinding", "share_trivial", "ci95",
            "g_start_bp", "g_start_trivial_upper_bp_max", "t_cert_trivial_s")
    cells: dict[str, dict] = {}
    groups: dict[str, dict] = {}
    for report in reports:
        members = set((report.get("summary") or {}).get("cells") or [])
        for label, state in report["cells"].items():
            est = state["estimate"]
            cells[label] = {"in_pooled": label in members, **{k: est.get(k) for k in keys},
                            "statement": _a3_sentence(label, est)}
        groups.update({name: dict(group) for name, group in (report.get("groups") or {}).items()})
    return {"cells": cells, "groups": groups}


def _hypothesis_block(result: rq2.CellResult | rq2.PooledResult) -> dict:
    iv = result.interval
    return {
        "estimate_log": result.estimate, "ratio": math.exp(result.estimate), "se": result.se,
        "level": iv.level if iv else None, "ci_log": [iv.low, iv.high] if iv else None,
        "ci_ratio": [math.exp(iv.low), math.exp(iv.high)] if iv else None, "outcome": result.outcome,
    }


def _pooled_block(est: rq2.Rq2Result, *, statements: bool) -> dict | None:
    if est.pooled is None:
        return None
    first = next(iter(est.pooled.values()))
    block: dict[str, Any] = {"cells": list(first.cells), "n_cells": len(first.cells), "weights": "equal",
                             "level": first.interval.level}
    for hypothesis, metric in _METRIC.items():
        part = _hypothesis_block(est.pooled[metric])
        if statements:
            part["statement"] = f"{STATEMENTS[hypothesis][part['outcome']]} ({A3_PROVISO})"
        block[hypothesis] = part
    return block


def _estimate(instances: Sequence[Mapping], pooled: Sequence[str], *, n_boot: int, seed: int,
              tolerant: bool = False) -> tuple[rq2.Rq2Result | None, list[str]]:
    """The estimate of :mod:`cif.stats.rq2` from the usable instances (cluster means: instances; ports in C5).
    ``tolerant`` (the sensitivities) lets a pooled cell that has lost its instances or all but one leave the pool,
    saying so in the notes."""
    usable = [i for i in instances if i["excluded"] is None]
    present = [c for c in H2_CELLS if any(i["cell"] == c for i in usable)]
    present += sorted({i["cell"] for i in usable} - set(H2_CELLS))
    cells: dict[str, dict[str, np.ndarray]] = {}
    for cell in present:
        mine = [i for i in usable if i["cell"] == cell]
        clusters = [i["cluster"] for i in mine]
        _, log_r = rq2.mean_of_means([i["log_r"] for i in mine], clusters)
        _, log_rb = rq2.mean_of_means([i["log_r_b"] for i in mine], clusters)
        cells[cell] = {_METRIC["H2"]: log_r, _METRIC["H2b"]: log_rb}
    notes: list[str] = []
    pool = list(pooled)
    if tolerant:
        for cell in pooled:
            n = len(next(iter(cells[cell].values()))) if cell in cells else 0
            if n < 2:
                pool.remove(cell)
                notes.append(f"{cell} has {n} cluster(s) left: it is not in the pool of this sensitivity")
    if not cells:
        return None, notes
    return rq2.estimate_rq2(cells, pooled=pool, n_boot=n_boot, seed=seed), notes


def _cell_blocks(instances: Sequence[Mapping], states: Mapping[str, Mapping], selected: Mapping[str, list],
                 est: rq2.Rq2Result | None) -> dict[str, dict]:
    """The cells of the primary analysis: counts, clusters, seeds, configurations, references, overruns and the
    estimates of H2 and H2b (``None`` for a cell without instances)."""
    order = [c for c in H2_CELLS if c in selected] + sorted(c for c in selected if c not in H2_CELLS)
    out: dict[str, dict] = {}
    for cell in order:
        mine = [i for i in instances if i["cell"] == cell and i["excluded"] is None]
        state = states[cell]
        block: dict[str, Any] = {
            "in_pooled": False,
            "family": state.get("family") or (mine[0]["instance_id"].split("/")[0] if mine else None),
            "n_instances": len(mine), "n_clusters": len({i["cluster"] for i in mine}),
            "cluster": "port" if any(isinstance(i["unit"], tuple) for i in mine) else "instance",
            "seeds": dict(mine[0]["n_seeds"]) if mine else None,
            "configs": {key: sorted({str(i["config_ids"][key]) for i in mine}) for key in REPRESENTATIVES}
            if mine else None,
            "reference": {"by_type": dict(Counter(i["ref_type"] for i in mine)),
                          "n_inconsistent": sum(bool(i["ref_inconsistent"]) for i in mine)},
            "n_over_budget": {key: sum(i["n_over_budget"][key] for i in mine) for key in REPRESENTATIVES},
            "g_start_median": float(np.median([i["g_start"] for i in mine if i["g_start"] is not None]))
            if any(i["g_start"] is not None for i in mine) else None,
            "H2": None, "H2b": None,
        }
        per_port = state.get("per_port")
        if per_port is not None:
            block["points_per_port"] = {str(p): int(n) for p, n in per_port.items()}
            block["ports_without_points"] = sorted(str(p) for p, n in per_port.items() if not n)
        if est is not None and cell in est.cells[_METRIC["H2"]]:
            for hypothesis, metric in _METRIC.items():
                block[hypothesis] = _hypothesis_block(est.cells[metric][cell])
            block["in_pooled"] = est.cells[_METRIC["H2"]][cell].in_pooled
        out[cell] = block
    return out


def _few_clusters(instances: Sequence[Mapping], est: rq2.Rq2Result | None, cells: Mapping[str, Mapping], *,
                  n_boot: int, seed: int) -> tuple[dict[str, dict], list[str]]:
    """The pooled cells with fewer than :data:`MIN_CLUSTERS` clusters, each with the pooled estimate without it (a
    post-hoc check; ``None`` when it is the only pooled cell), and the warnings that name them. The estimator is not
    changed: the bootstrap stream of a cell does not depend on the other cells, so the estimate without a cell is
    that of the remaining cells alone."""
    if est is None or est.pooled is None:
        return {}, []
    pool = list(next(iter(est.pooled.values())).cells)
    out: dict[str, dict] = {}
    warnings: list[str] = []
    for cell in pool:
        n = est.cells[_METRIC["H2"]][cell].n_clusters
        if n >= MIN_CLUSTERS:
            continue
        kind = cells[cell]["cluster"]
        rest = [c for c in pool if c != cell]
        without = _pooled_block(_estimate(instances, rest, n_boot=n_boot, seed=seed)[0], statements=False) \
            if rest else None
        out[cell] = {"n_clusters": n, "cluster": kind, "role": _POST_HOC_ROLE, "without": without}
        text = (f"{cell} enters the pooled estimate with {n} {kind} clusters (fewer than {MIN_CLUSTERS}): its "
                f"percentile bootstrap interval is the range of the {kind} means and understates the variance of the "
                "cell mean (audit G2, C-1)")
        if without is None:
            text += "; it is the only pooled cell, so there is no pooled estimate without it"
        else:
            text += f"; post hoc, without {cell} (pooled over {', '.join(rest)}): " + ", ".join(
                f"{h} {without[h]['ratio']:.3g} (95 % CI {without[h]['ci_ratio'][0]:.3g} to "
                f"{without[h]['ci_ratio'][1]:.3g}), outcome {without[h]['outcome']}" for h in _METRIC)
        warnings.append(text)
    return out, warnings


def _sensitivity(name: str, definition: str, instances: Sequence[Mapping], pooled: Sequence[str], *, n_boot: int,
                 seed: int) -> dict:
    est, notes = _estimate(instances, pooled, n_boot=n_boot, seed=seed, tolerant=True)
    cells: dict[str, dict] = {}
    if est is not None:
        for cell in est.cells[_METRIC["H2"]]:
            mine = est.cells[_METRIC["H2"]][cell]
            cells[cell] = {"in_pooled": mine.in_pooled, "n_clusters": mine.n_clusters,
                           "n_instances": sum(1 for i in instances if i["cell"] == cell and i["excluded"] is None),
                           **{h: _hypothesis_block(est.cells[m][cell]) for h, m in _METRIC.items()}}
    return {"name": name, "role": _SENSITIVITY_ROLE, "definition": definition,
            "n_excluded": dict(Counter(i["cell"] for i in instances if i["excluded"] is not None)),
            "pooled": None if est is None else _pooled_block(est, statements=False), "cells": cells, "notes": notes}


def h2_report(
    rows: Iterable[Mapping], reports: Sequence[Mapping], *, references: Iterable[Mapping] = (),
    cold_rows: Iterable[Mapping] = (), budget_s: float = BUDGET_S, n_boot: int = rq2.N_BOOT, seed: int = rq2.SEED,
    qubo_role: str = REPRESENTATIVES["qubo"], exact_config: str | None = None,
) -> dict:
    """The confirmatory estimate of H2 and H2b and everything around it (module docstring), from the rows of the
    bench stages, the finished eligibility reports of the test scan (``reports``: the test report and, for D1, the
    dense one), the certificates (``references``) and, for the honest variant of A4, the rows of the stage
    ``cold``. ``qubo_role`` other than the representative gives the post hoc description of §12, 2026-10-19 (2): the
    same estimator with another tuned QUBO sampler of the same runs, marked as not prespecified. No I/O: :func:`main`
    adds the inputs and the code state to what this returns."""
    pilot = _pilot()
    rows, reports, cold_rows = list(rows), list(reports), list(cold_rows)
    post_hoc = qubo_role != REPRESENTATIVES["qubo"] or exact_config is not None
    if post_hoc and cold_rows:
        raise ValueError("the cold runs hold the representatives only: the variant own cannot be formed for a post hoc "
                         f"choice (QUBO role {qubo_role!r}, exact configuration {exact_config!r}); leave out the cold "
                         "rows")
    selected, states, pooled = _plan(reports)
    used, ignored = _reference_rows(references)
    refs = pilot.reference_table([*rows, *used])
    instances = bench_instances(rows, reports, refs=refs, budget_s=budget_s, qubo_role=qubo_role,
                                exact_config=exact_config)
    est, _ = _estimate(instances, pooled, n_boot=n_boot, seed=seed)
    family_size = max(1, len(pooled))
    settings_prespecified = bool(n_boot == rq2.N_BOOT and seed == rq2.SEED and budget_s == BUDGET_S)
    prespecified = settings_prespecified and not post_hoc
    warnings: list[str] = []
    if qubo_role != REPRESENTATIVES["qubo"]:
        warnings.append(f"post hoc description (§12, 2026-10-19 (2)): the QUBO class is represented by the tuned role "
                        f"{qubo_role!r} of the same runs, not by the representative {REPRESENTATIVES['qubo']!r}; not the "
                        "confirmatory estimate")
    if exact_config is not None:
        warnings.append(f"post hoc sensitivity (§12, 2026-10-19): the exact class is represented by the configuration "
                        f"{exact_config!r}, not by the representative {REPRESENTATIVES['exact']!r}; not the confirmatory "
                        "estimate")
    if not settings_prespecified:
        warnings.append(f"the settings (n_boot = {n_boot}, seed = {seed}, budget_s = {budget_s}) are not the "
                        f"prespecified ones (n_boot = {rq2.N_BOOT}, seed = {rq2.SEED}, budget_s = {BUDGET_S}): this is "
                        "not the confirmatory estimate")
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
        warnings.append(f"{over} rows overran the budget (budget_ok = false): used as measured, counted in `budget`; "
                        "the sensitivity within_budget drops them")
    if ignored:
        warnings.append(f"{ignored} reference row(s) of other stages (or with an error) were ignored: only the stages "
                        f"{', '.join(REFERENCE_STAGES)} carry certificates")
    cell_blocks = _cell_blocks(instances, states, selected, est)
    few, few_warnings = _few_clusters(instances, est, cell_blocks, n_boot=n_boot, seed=seed)
    warnings += few_warnings
    sensitivity = {
        name: _sensitivity(name, definition, bench_instances(rows, reports, refs=refs, budget_s=budget_s,
                                                             qubo_role=qubo_role, exact_config=exact_config,
                                                             **kwargs),
                           pooled, n_boot=n_boot, seed=seed)
        for name, (kwargs, definition) in SENSITIVITIES.items()}
    if cold_rows:
        cold = bench_instances(cold_rows, [r for r in reports if r["cells"].keys() & set(H2_CELLS)], refs=refs,
                               budget_s=budget_s, variant="own")
        sensitivity["own"] = _sensitivity(
            "own", "γ = 1 until the method's own first solution, from cold runs without the common start (A4 (2))",
            cold, pooled, n_boot=n_boot, seed=seed)
    overruns = {cell: {key: sum(i["n_over_budget"][key] for i in instances if i["cell"] == cell)
                       for key in REPRESENTATIVES} for cell in dict.fromkeys(i["cell"] for i in instances)}
    descriptive = {cell: reason for report in reports
                   for cell, reason in ((report.get("summary") or {}).get("excluded") or {}).items()}
    pooled_block = None if est is None else _pooled_block(est, statements=True)
    return {
        "kind": "e2-h2", "schema": "e2-h2/1", "plan_version": "v2.4",
        "measure": "PI+ = PI - t0/T (ANALYSIS_PLAN §12 A2): the primal integral after the common start, medians over "
                   "the solver seeds of each representative",
        "settings": {
            "budget_s": budget_s, "n_boot": n_boot, "seed": seed, "level": rq2.POOLED_LEVEL,
            "family_level": rq2.FAMILY_LEVEL, "family_size": family_size,
            "cell_level": rq2.family_level(family_size), "band": [1.0 / rq2.EQUIVALENCE_FACTOR, rq2.EQUIVALENCE_FACTOR],
            "interval": "percentile", "prespecified": prespecified,
            **({"qubo_role": qubo_role} if qubo_role != REPRESENTATIVES["qubo"] else {}),
            **({"exact_config": exact_config} if exact_config is not None else {})},  # the confirmatory schema stays
        "prespecified": PRESPECIFIED,
        "pooled": pooled_block,
        "descriptive": descriptive,
        "cells": cell_blocks,
        "few_clusters": few,
        "statement": NO_POOLED_STATEMENT if not pooled and any(c in H2_CELLS for c in selected) else None,
        "a3": a3_estimates(reports),
        "references": {
            "n_instances": len(instances), "by_type": dict(by_type),
            "n_inconsistent": len(flagged), "n_rows_used": len(used), "n_rows_ignored": ignored,
            "stages": list(REFERENCE_STAGES)},
        "budget": {"n_over_budget": overruns, "n_rows_over_budget": over},
        "sensitivity": sensitivity,
        "instances": instances,
        "warnings": warnings, "confirmatory": False, "inputs": [], "code": None,
    }


#: The columns of the tables, fixed so that an empty table still has its header.
_INSTANCE_COLUMNS = (
    "cell", "unit", "cluster", "instance_id", "n_assets", "K", "s_cont", "binding", "ref_type", "ref_source",
    "ref_inconsistent", "g_start", "t0_s", "pi_exact", "pi_qubo", "pi_ls", "r", "r_b", "log_r", "log_r_b",
    "n_seeds_exact", "n_seeds_qubo", "n_seeds_ls", "n_over_budget_exact", "n_over_budget_qubo", "n_over_budget_ls",
    "config_exact", "config_qubo", "config_ls", "excluded")
_ESTIMATE_COLUMNS = ("hypothesis", "estimate_log", "ratio", "se", "ci_level", "ci_low_log", "ci_high_log",
                     "ratio_low", "ratio_high", "outcome")
_CELL_COLUMNS = ("variant", "cell", "in_pooled", "n_instances", "n_clusters", *_ESTIMATE_COLUMNS)
_POOLED_COLUMNS = ("variant", "cells", "n_cells", *_ESTIMATE_COLUMNS)
_A3_COLUMNS = (
    "cell", "in_pooled", "n_scanned", "n_trivial", "n_nontrivial", "n_undecided", "n_nonbinding", "share_trivial",
    "ci_low", "ci_high", "g_start_median_bp", "g_start_q1_bp", "g_start_q3_bp", "g_start_max_bp",
    "g_start_trivial_upper_bp_max", "t_cert_median_s", "t_cert_level", "statement")


def _instance_row(inst: Mapping) -> dict:
    row = {k: inst[k] for k in ("cell", "cluster", "instance_id", "n_assets", "K", "s_cont", "binding", "ref_type",
                                "ref_source", "ref_inconsistent", "g_start", "t0_s", "r", "r_b", "log_r", "log_r_b",
                                "excluded")}
    row["unit"] = _unit_text(inst["unit"])
    for key in REPRESENTATIVES:
        row[f"pi_{key}"] = inst["pi"].get(key)
        row[f"n_seeds_{key}"] = inst["n_seeds"][key]
        row[f"n_over_budget_{key}"] = inst["n_over_budget"][key]
        row[f"config_{key}"] = inst["config_ids"][key]
    return row


def _estimate_rows(variant: str, cell: str | None, block: Mapping, in_pooled: bool | None, n_instances, n_clusters,
                   cells: list[dict], pooled: list[dict]) -> None:
    for hypothesis in _METRIC:
        part = block.get(hypothesis)
        if not part:
            continue
        low, high = part["ci_log"] or (None, None)
        row = {"variant": variant, "hypothesis": hypothesis, "estimate_log": part["estimate_log"],
               "ratio": part["ratio"], "se": part["se"], "ci_level": part["level"], "ci_low_log": low,
               "ci_high_log": high, "ratio_low": None if low is None else math.exp(low),
               "ratio_high": None if high is None else math.exp(high), "outcome": part["outcome"]}
        if cell is None:
            pooled.append({**row, "cells": " ".join(block["cells"]), "n_cells": block["n_cells"]})
        else:
            cells.append({**row, "cell": cell, "in_pooled": in_pooled, "n_instances": n_instances,
                          "n_clusters": n_clusters})


def _tables(report: Mapping) -> dict[str, pd.DataFrame]:
    cells: list[dict] = []
    pooled: list[dict] = []
    variants = [("primary", report["pooled"], report["cells"])] + [
        (name, s["pooled"], s["cells"]) for name, s in report["sensitivity"].items()]
    variants += [(f"without_{cell}", part["without"], {}) for cell, part in (report.get("few_clusters") or {}).items()
                 if part["without"]]
    for variant, pooled_block, cell_blocks in variants:
        if pooled_block:
            _estimate_rows(variant, None, {**pooled_block, "cells": pooled_block["cells"]}, None, None, None,
                           cells, pooled)
        for cell, block in cell_blocks.items():
            _estimate_rows(variant, cell, block, block.get("in_pooled"), block.get("n_instances"),
                           block.get("n_clusters"), cells, pooled)
    a3 = []
    for cell, part in report["a3"]["cells"].items():
        flat = {k: v for k, v in part.items() if k not in ("ci95", "g_start_bp", "t_cert_trivial_s", "statement")}
        low, high = part["ci95"] if part["ci95"] else (None, None)
        g, t = part["g_start_bp"] or {}, part["t_cert_trivial_s"] or {}
        a3.append({"cell": cell, **flat, "ci_low": low, "ci_high": high, "g_start_median_bp": g.get("median"),
                   "g_start_q1_bp": g.get("q1"), "g_start_q3_bp": g.get("q3"), "g_start_max_bp": g.get("max"),
                   "t_cert_median_s": t.get("median"), "t_cert_level": t.get("level"), "statement": part["statement"]})
    return {"instances.csv": pd.DataFrame([_instance_row(i) for i in report["instances"]],
                                          columns=list(_INSTANCE_COLUMNS)),
            "cells.csv": pd.DataFrame(cells, columns=list(_CELL_COLUMNS)),
            "pooled.csv": pd.DataFrame(pooled, columns=list(_POOLED_COLUMNS)),
            "a3.csv": pd.DataFrame(a3, columns=list(_A3_COLUMNS))}


def write_h2(report: Mapping, out_dir: Path) -> list[Path]:
    """``h2.json`` (the report) and the tables ``instances.csv``, ``cells.csv``, ``pooled.csv``, ``a3.csv`` in
    ``out_dir``. The JSON is strict (no NaN: missing numbers are ``null``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "h2.json"]
    written[0].write_text(
        json.dumps(e2_eligibility._json_safe(dict(report)), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    for name, frame in _tables(report).items():
        written.append(out_dir / name)
        frame.to_csv(written[-1], index=False)
    return written


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------

_EXPECTED_STAGES = {"rows": ("bench",), "cold": ("cold",), "reference": REFERENCE_STAGES}


def _run_inputs(paths: Sequence[Path], role: str) -> list[dict]:
    """Where the rows came from (:func:`cif.experiments.e2_pilot._provenance`: source, SHA-256, row count and, for a
    registered run directory, its id, git SHA, config hash and mode) with the seed role and stage of its config and
    the counts of errors and missing cells of its manifest."""
    out = []
    for path, entry in zip(paths, _pilot()._provenance(list(paths)), strict=True):
        entry = {**entry, "role": role}
        if path.is_dir():
            if (path / "manifest.json").is_file():
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                entry.update({k: manifest.get(k) for k in ("n_errors", "n_missing", "bundle_sha")})
            if (path / "config.json").is_file():
                params = json.loads((path / "config.json").read_text(encoding="utf-8")).get("params") or {}
                entry.update(seed_role=params.get("seed_role"), stage=params.get("stage"))
        out.append(entry)
    return out


def _provenance_warnings(inputs: Sequence[Mapping], code: Mapping) -> list[str]:
    """What keeps the analysis from being the confirmatory one: an input that is not a registered run of the test
    seeds from a clean tree, or code from a dirty tree."""
    warnings = []
    for entry in inputs:
        role = entry["role"]
        if role == "eligibility":
            continue
        if entry.get("run_id") is None:
            warnings.append(f"{entry['source']}: not a registered run directory (no manifest.json): its rows cannot "
                            "be traced to a run")
            continue
        name = entry["run_id"]
        if entry.get("mode") not in ("full", "ingested"):
            warnings.append(f"{name}: manifest mode = {entry.get('mode')!r} (rehearsals are not confirmatory)")
        if entry.get("git_dirty") is not False:
            warnings.append(f"{name}: manifest git_dirty = {entry.get('git_dirty')!r} (not a run from a clean tree)")
        if entry.get("seed_role") != "test":
            warnings.append(f"{name}: config seed_role = {entry.get('seed_role')!r} (not the test seeds)")
        if entry.get("stage") not in _EXPECTED_STAGES[role]:
            warnings.append(f"{name}: config stage = {entry.get('stage')!r}, expected one of {_EXPECTED_STAGES[role]}")
        if entry.get("n_errors") or entry.get("n_missing"):
            warnings.append(f"{name}: the run has {entry.get('n_errors')} cells with errors and "
                            f"{entry.get('n_missing')} missing")
    if code.get("git_dirty") is not False:
        warnings.append(f"the analysis code is from a dirty or unknown tree (git_dirty = {code.get('git_dirty')!r})")
    return warnings


def _h2_command(args: argparse.Namespace) -> int:
    pilot = _pilot()
    files = [args.eligibility, *([args.eligibility_dense] if args.eligibility_dense else [])]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    rows, errors = pilot.load_rows(args.rows)
    cold, cold_errors = pilot.load_rows(args.cold)
    report = h2_report([*rows, *errors], reports, references=pilot.reference_rows(args.reference),
                       cold_rows=[*cold, *cold_errors], qubo_role=args.qubo_role, exact_config=args.exact_config)
    inputs = [entry for role, paths in (("rows", args.rows), ("reference", args.reference), ("cold", args.cold))
              for entry in _run_inputs(paths, role)]
    inputs += [{**pilot._file_entry(path), "role": "eligibility"} for path in files]
    state = registry._git_state()
    code = {"git_sha": state.get("sha") or None, "git_dirty": state.get("dirty"), "branch": state.get("branch") or None}
    problems = _provenance_warnings(inputs, code)
    given = {entry["run_id"] for entry in inputs if entry.get("run_id")}
    for path, built in zip(files, reports, strict=True):
        absent = sorted({e["run_id"] for e in built.get("inputs") or [] if isinstance(e, Mapping) and e.get("run_id")}
                        - given)
        if absent:
            problems.append(f"{path.name} was built from the runs {absent}, which are not among the inputs: the "
                            "certificates of their instances are missing from the references")
    versions = {"python": sys.version.split()[0],
                **{p: metadata.version(p) for p in ("numpy", "pandas", "scipy")}}
    report.update(inputs=inputs, code=code, versions=versions, warnings=[*problems, *report["warnings"]],
                  confirmatory=bool(not problems and report["settings"]["prespecified"]))
    write_h2(report, args.out)
    print(f"e2-h2: {args.out}: confirmatory = {report['confirmatory']}, "
          f"{len(report['instances'])} instances in {len(report['cells'])} cells")
    pooled = report["pooled"]
    if pooled is None:
        print(f"  no pooled estimate: {report['statement'] or 'no cell of RQ2 in the reports'}")
    else:
        print(f"  pooled over {', '.join(pooled['cells'])} (equal weights):")
        for hypothesis in _METRIC:
            part = pooled[hypothesis]
            low, high = part["ci_ratio"]
            print(f"  {hypothesis}: R = {part['ratio']:.4g} ({100 * part['level']:.0f} % CI {low:.4g} to {high:.4g}), "
                  f"outcome {part['outcome']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0


def _provenance(path: Path) -> dict:
    return {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e2_analysis", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("h1c", help="the test of H1c on the rows of the stage h1c")
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="results.jsonl files or run directories")
    p.add_argument("--eligibility", type=Path, required=True, help="the finished eligibility report of the test cells")
    p.add_argument("--reference", type=Path, nargs="*", default=[],
                   help="reference runs: their certified optima give the scale s* (default: the scale of the start)")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser(
        "h2", help="the confirmatory estimate of H2 and H2b (pooled mean log R, five outcomes) and the estimate of "
                   "A3 on the rows of the bench stages",
        description="Pooled mean of log R = log(PI+(QUBO)/PI+(exact)) (H2) and of log R_b (QUBO against local "
                    "search, H2b) over the cells of RQ2 with equal weights, 95 % stratified-bootstrap interval, "
                    "simultaneous 99 % (Bonferroni) cell intervals, the five outcomes against the band "
                    "[1/1.5, 1.5], the share of trivial instances (A3) and the sensitivities of A4. The bootstrap "
                    "(B, seed, interval), the budget and the readings of the open points are prespecified in the "
                    "code (PRESPECIFIED in the JSON), not options.")
    p.add_argument("--rows", type=Path, nargs="*", default=[],
                   help="registered runs (or results.jsonl files) of bench_factor, bench_orlib, bench_boot, "
                        "bench_dense; none is needed if no instance was selected")
    p.add_argument("--eligibility", type=Path, required=True,
                   help="the finished eligibility report of the test scan (C1-C5, the pooled set in its summary)")
    p.add_argument("--eligibility-dense", type=Path, default=None,
                   help="the finished eligibility report of D1 (descriptive); its bench rows go to --rows")
    p.add_argument("--reference", type=Path, nargs="*", default=[],
                   help="runs of the test scan (stage elig_test, every round) and of ref_bench: their certified "
                        "optima are the references; rows of other stages are ignored")
    p.add_argument("--cold", type=Path, nargs="*", default=[],
                   help="the run of the stage cold (honest variant of A4); optional")
    p.add_argument("--qubo-role", default=REPRESENTATIVES["qubo"], choices=("qubo_rep", "sa", "tabu", "swap_sa"),
                   help="post hoc (§12, 2026-10-19 (2)): another tuned QUBO role of the same bench runs in the place "
                        "of the representative; the report is then not confirmatory (default: the representative)")
    p.add_argument("--exact-config", default=None,
                   help="post hoc sensitivity (§12, 2026-10-19): the rows of this exact configuration (config_id, e.g. "
                        "of a later run) in the place of the exact representative; the report is then not "
                        "confirmatory")
    p.add_argument("--out", type=Path, required=True, help="results directory: h2.json and the tables")
    p = sub.add_parser(
        "secondary", help="the secondary analyses of RQ2 (not confirmatory): time-to-target, t_primal and t_cert of "
                          "the exact class, performance profiles, on the rows of the bench stages",
        description="Secondary analyses of RQ2, descriptive and not confirmatory (cif.experiments.e2_secondary): the "
                    "pooled ratio of the Kaplan-Meier medians of the time to the target, QUBO against the exact "
                    "method and against local search, with the sensitivities to the target and to the censoring "
                    "budget; t_primal and t_cert of the exact representative by cell; Dolan-More performance "
                    "profiles on PI+ and on the time to the target. The inputs are those of h2; the targets, the "
                    "budgets, the bootstrap and the readings of the open points are prespecified in the code "
                    "(PRESPECIFIED in the JSON), not options. Every output is flagged secondary.")
    p.add_argument("--rows", type=Path, nargs="*", default=[],
                   help="registered runs (or results.jsonl files) of bench_factor, bench_orlib, bench_boot, "
                        "bench_dense; none is needed if no instance was selected")
    p.add_argument("--eligibility", type=Path, required=True,
                   help="the finished eligibility report of the test scan (C1-C5, the pooled set in its summary)")
    p.add_argument("--eligibility-dense", type=Path, default=None,
                   help="the finished eligibility report of D1 (descriptive); its bench rows go to --rows")
    p.add_argument("--reference", type=Path, nargs="*", default=[],
                   help="runs of the test scan (stage elig_test, every round) and of ref_bench: their certified "
                        "optima are the references; rows of other stages are ignored")
    p.add_argument("--out", type=Path, required=True,
                   help="results directory: secondary.json and the tables (every one flagged secondary)")
    p = sub.add_parser(
        "ttt-curves", help="the TTT curves of the stage ttt (secondary, descriptive)",
        description="Time-to-target curves of the three representatives on the calibration instances of the stage ttt "
                    "(cif.experiments.e2_curves): per method and instance the empirical distribution of the time to "
                    "the target with the misses in the denominator, the share at the target and the Kaplan-Meier "
                    "quartiles, per pair P(TTT_A < TTT_B); τ = 1e-3 and 1e-4. Descriptive, flagged secondary.")
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="the registered run of the stage ttt")
    p.add_argument("--reference", type=Path, nargs="*", default=[],
                   help="runs of the calibration screening and reference stages: their certified optima are the "
                        "references; rows of other stages are ignored")
    p.add_argument("--out", type=Path, required=True, help="results directory: ttt_curves.json and the tables")
    p = sub.add_parser(
        "delay", help="the delay to the first own solution on the stage cold (secondary, descriptive)",
        description="The time of the first point of each method's own trajectory in the cold runs, censored at the "
                    "budget, by instance and cell (cif.experiments.e2_curves). Descriptive, flagged secondary.")
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="the registered run of the stage cold")
    p.add_argument("--out", type=Path, required=True, help="results directory: delay.json and the tables")
    p = sub.add_parser(
        "first-quality", help="post hoc (§12, 2026-10-19 (3)): the gap of each method's first own portfolio in the "
                              "cold runs against the common start of the same instance",
        description="Per instance and representative: the gap of the first point of its own trajectory in the cold "
                    "runs against the reference of the registered h2 analysis, and whether it is not worse and not "
                    "later than the common start of the instance; per cell the shares and medians "
                    "(cif.experiments.e2_curves). Post hoc and descriptive.")
    p.add_argument("--rows", type=Path, nargs="+", required=True, help="the registered run of the stage cold")
    p.add_argument("--h2", type=Path, required=True,
                   help="h2.json of the registered analysis of H2 (references, gaps and times of the common start)")
    p.add_argument("--out", type=Path, required=True, help="results directory: first_quality.json and the tables")
    p = sub.add_parser(
        "m11", help="the validation of the exact stack (M11): our times against the published times of Bertsimas & "
                    "Cory-Wright, Tables 3 and 4, on the rows of the stage m11",
        description="Per table and instance (port, k): the ratio of our median time to the best published time "
                    "(times beyond the limit set to the limit and flagged as bounds, cells where both sides failed "
                    "left out and listed); per table the median of the ratios, the number of cells solved within the "
                    "limit by us and by the best published method, and the verdict of the rule of the plan (median "
                    "above 10). The verdict reads SCIP's times (the exact representative of RQ2 at these sizes); OA "
                    "and the better of our methods are in the report too; the readings of the open points are "
                    "prespecified in the code (PRESPECIFIED in the JSON). The hardware is not normalised.")
    p.add_argument("--rows", type=Path, nargs="+", required=True,
                   help="the registered run(s) of the stage m11 (or results.jsonl files)")
    p.add_argument("--published", type=Path, required=True,
                   help="the published times, configs/E2/m11_bcw_published.yaml")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="analyse a grid with missing instances or tables (a rehearsal, a partial run); they are listed")
    p.add_argument("--out", type=Path, required=True, help="results directory: m11.json and the tables")
    args = parser.parse_args(argv)
    if args.command == "h2":
        return _h2_command(args)
    if args.command == "secondary":
        from cif.experiments import e2_secondary

        return e2_secondary.run_command(args)
    if args.command in ("ttt-curves", "delay", "first-quality"):
        from cif.experiments import e2_curves

        return e2_curves.run_command(args)
    if args.command == "m11":
        from cif.experiments import e2_m11_analysis

        return e2_m11_analysis.run_command(args)

    from cif.experiments import e2_pilot

    eligibility = json.loads(args.eligibility.read_text(encoding="utf-8"))
    rows, errors = e2_pilot.load_rows(args.rows)
    scales = None
    if args.reference:
        ref_rows = e2_pilot.reference_rows(args.reference)
        scales = {iid: ref.s_ref for iid, ref in e2_pilot.reference_table([*rows, *ref_rows]).items()}
    result = h1c_test([*rows, *errors], eligibility, scales=scales)
    inputs = [_provenance(p) if p.is_file() else _provenance(p / "results.jsonl") for p in args.rows]
    result["inputs"] = [*inputs, _provenance(args.eligibility)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"e2-h1c: {args.out}: {result['n_instances']} instances ({result['n_zero']} zero), "
          f"p = {result['p_value']:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
