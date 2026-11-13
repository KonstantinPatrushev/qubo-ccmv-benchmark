"""Analysis of the validation M11: our times against the published ones of Bertsimas & Cory-Wright, Tables 3 and 4
(ANALYSIS_PLAN §2 RQ2 "Валидация бейзлайна", §12 entry of 2026-10-17 "Выборы валидации точного стека M11 (до прогона)",
items (5) and (6)): ``python -m cif.experiments.e2_analysis m11``.

Inputs: the registered run(s) of the stage :mod:`cif.experiments.e2_m11` and the published numbers
(``configs/E2/m11_bcw_published.yaml``, :func:`load_published`). Per table and instance ``(port, k)``:

* **our time** of a method is the median over its seeds of the wall time of the call; a run that is not
  ``certified`` (the gap not closed within the limit) counts as the limit, and the median is flagged as a bound
  (``censored``) when at least half of the runs failed (:func:`our_time`);
* **the best published time** is the minimum over the methods of the BCW table of the printed times; a printed
  ``">L"`` is the limit ``L`` and a bound; when every method printed ``">L"`` the best is ``L`` and a bound
  (:attr:`PublishedCell.best`);
* **the ratio** of the cell is ``our time / best published time`` (:func:`cell_ratio`); it is a lower bound when only
  ours is a bound, an upper bound when only the published one is, and a cell where both are bounds has no ratio: it is
  **left out of the summary and listed** (``excluded``);
* **the summary** of a table is the median of the cell ratios (bounds at their values, counted: ``n_lower_bound``,
  ``n_upper_bound``, and ``median_bound`` when the middle of the sample is one) and the number of cells solved within
  the limit by us and by the best published method; **the verdict** applies the rule of the plan, "the median ratio
  is above 10" (:data:`THRESHOLD`; ``statement`` is the wording of the plan).

Everything the plan leaves open is in :data:`PRESPECIFIED` (and in the JSON): chiefly that **the verdict reads SCIP's
times** (view ``scip``: at the sizes of the ports SCIP is the exact representative of RQ2), with OA (``oa``, T3 only)
and the better of our methods (``stack``, the analogue of "the best of the BCW methods") beside it, and that the
verdict of the two tables is "above" if either is (§12, entry of 2026-10-18, before the run).
The hardware is not normalised (their MacBook Pro i9, CPLEX 12.8, Mosek 9 and Julia against the run server, Xeon Silver
4216): the ratios are raw ratios of wall-clock times, and the times printed with two decimals (0.01 s) make the
ratios against the smallest published cells coarse.

**Audits** (:func:`audit_rows`; a non-zero count is a warning in the report): rows that are not feasible, that
overran their limit by more than the stage's tolerance (``budget_ok = false``), whose return row is violated, certified
rows whose recorded gap exceeds the stopping rule (relative 1e-4 and absolute 1e-6, :data:`STOP_REL_GAP`,
:data:`STOP_ABS_GAP`), and **bound conflicts**: a certified dual bound above the incumbent of another certified run of
the same instance (one of the two is wrong; with the rows of SCIP and OA of one instance this is a cross-check of
the solvers and, in T4, of the return row, at the scale of the stage).

**Output** (:func:`write_m11`): ``m11.json`` (the report: ``prespecified``, ``published``, the tables with their cells,
summaries, verdicts and audit, ``inputs`` with run ids and SHA-256, the code state, ``warnings``) and the tables
``cells.csv`` (one row per table and instance), ``summary.csv`` (per table and view) and ``rows.csv`` (the rows read).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from cif.experiments import registry
from cif.instances.orlib import ORLIB_N

__all__ = [
    "PRESPECIFIED",
    "PRIMARY_VIEW",
    "STOP_ABS_GAP",
    "STOP_REL_GAP",
    "THRESHOLD",
    "Best",
    "OurTime",
    "Published",
    "PublishedCell",
    "PublishedTable",
    "Ratio",
    "Time",
    "audit_rows",
    "cell_ratio",
    "load_published",
    "m11_report",
    "our_time",
    "parse_time",
    "run_command",
    "write_m11",
]

#: The rule of the plan (§12, item (6)): the median ratio above this means the conclusions of RQ2 are about *open*
#: exact methods only.
THRESHOLD = 10.0
#: The view whose median ratio is the verdict: SCIP, the exact representative of RQ2 at the sizes of the ports (module
#: docstring; §12, entry of 2026-10-18).
PRIMARY_VIEW = "scip"
#: The better of our methods in a cell (T3: SCIP and OA; T4: SCIP), reported beside the verdict.
STACK_VIEW = "stack"
#: The stopping rule of the stage (CPLEX defaults: relative gap 1e-4, absolute gap 1e-6).
STOP_REL_GAP = 1e-4
STOP_ABS_GAP = 1e-6
_GAP_SLACK = 1e-3  # a recorded gap may exceed the rule by this share (the two definitions of the gap, rounding)
_BOUND_TOL_REL = 1e-6  # a certified dual bound may exceed another run's incumbent by this share (+ 1e-9)
_TABLES = ("T3", "T4")
_METHODS = ("scip", "oa")  # the order of preference on a tie
_PUBLISHED_KIND = "bcw-published"
_GRID = tuple((port, k) for port in sorted(ORLIB_N) for k in (5, 10, 20))

#: What the plan leaves open and how the analysis reads it (written into the JSON of the report). ``choice`` is what
#: the code does, ``why`` the plan text or the convention of the project it follows.
PRESPECIFIED = {
    "our_time": {
        "choice": "The time of a method in a cell is the median over its solver seeds of the wall time of the call "
                  "(build + solve). A run that is not certified within the limit (the gap of the stopping rule not "
                  "closed, or the call ended after the limit) counts as the limit; the median is a bound (censored) when "
                  "at least half of the runs failed.",
        "why": "§12 (2): 'у SCIP — 3 сида перестановок, значение ячейки — медиана'; (5): 'время за лимитом берётся "
               "равным лимиту и помечается границей'.",
    },
    "published": {
        "choice": "The best published time of a cell is the minimum over the methods of the BCW table of the printed "
                  "times (Table 3: Algorithm 3, CPLEX big-M, CPLEX MISOCO; Table 4: and Algorithm 3 with the in-out "
                  "cuts); a printed '>L' counts as L and is a bound, and when every method printed '>L' the best "
                  "published time is L and a bound.",
        "why": "§12 (5): 'лучшее опубликованное время ячейки (минимум по методам таблицы BCW)'.",
    },
    "ratio": {
        "choice": "The ratio of a cell is our time over the best published time. It is a lower bound when our time is "
                  "a bound, an upper bound when the published one is; when both are bounds the cell has no ratio, is "
                  "left out of the summary and is listed (excluded).",
        "why": "§12 (5): 'ячейка, где за лимит вышли обе стороны, из сводки исключается и отчитывается'.",
    },
    "primary_view": {
        "choice": "The verdict reads SCIP's times (view scip): at the sizes of the OR-Library ports (n = 31-225) SCIP "
                  "is the exact representative of RQ2 (tuned.json, sizes 100 and 200; OA is the representative at "
                  "N = 400 only), so the rule of the plan judges the baseline that stood in H2 at these sizes. OA "
                  "(view oa, T3 only) and the better of the two (view stack: the smaller of the medians of our methods, "
                  "the analogue of the best of the BCW methods) are reported beside it and do not enter the verdict.",
        "why": "§12 (2) names the stack (SCIP and OA) and (5) 'наше медианное время' per cell without saying which "
               "method of a stack of two; the entry of §12 of 2026-10-18 (before the run) fixes SCIP, the "
               "conservative reading (exploratory laptop timings of the agent, 18 runs, not registered, were seen "
               "before it).",
    },
    "excluded": {
        "choice": "A cell where both our time and the best published time are bounds is excluded from the summary "
                  "(counted in n_excluded, listed with its port and k); it still counts as not solved by either side.",
        "why": "§12 (5).",
    },
    "summary": {
        "choice": "Per table and view: the median of the cell ratios (bounds at their values; n_lower_bound, "
                  "n_upper_bound and median_bound say how far it rests on bounds) and the number of cells of the 15 "
                  "solved within the limit by us and by the best published method.",
        "why": "§12 (5): 'Сводка — медиана отношений по ячейкам каждой таблицы и число ячеек, решённых в лимите каждой "
               "стороной'.",
    },
    "rule": {
        "choice": "The verdict of a table is above if the median ratio of the primary view is above 10 (10 itself is "
                  "not above). The two tables are judged separately; the overall verdict is above if either table is "
                  "(the conclusions of RQ2 then carry the caveat of the baseline).",
        "why": "§12 (6): 'медианное отношение больше 10 — выводы RQ2 только про открытые точные методы с оговоркой о "
               "бейзлайне; не больше 10 — … в пределах порядка от опубликованного'; the plan does not say how the two "
               "tables combine.",
    },
    "hardware": {
        "choice": "No normalisation to the hardware of the paper: raw ratios of wall-clock times.",
        "why": "§12 (4): 'нормировки по железу нет, это раскрывается'.",
    },
}

_STATEMENTS = {
    True: "Медианное отношение нашего времени к лучшему опубликованному больше 10: выводы RQ2 формулируются только "
          "как сравнение с открытыми точными методами, а исход «QUBO лучше» сопровождается оговоркой о бейзлайне",
    False: "Медианное отношение нашего времени к лучшему опубликованному не больше 10: открытый точный стек на этих "
           "задачах в пределах порядка от опубликованного",
}


# --------------------------------------------------------------------------------------------
# The published numbers
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Time:
    """A time in seconds; ``censored``: it is a bound (a time beyond the limit, set to the limit)."""

    seconds: float
    censored: bool


@dataclass(frozen=True)
class Best(Time):
    """The best published time of a cell and the method that printed it (``None`` when all failed)."""

    method: str | None


@dataclass(frozen=True)
class OurTime(Time):
    """Our time of a cell for one method: the median over ``n_runs`` runs, of which ``n_failed`` did not certify."""

    n_runs: int
    n_failed: int


@dataclass(frozen=True)
class Ratio:
    """``ratio`` of our time to the best published one; ``bound``: ``"lower"``, ``"upper"`` or ``None``;
    ``excluded``: both are bounds, there is no ratio."""

    ratio: float | None
    bound: str | None
    excluded: bool


@dataclass(frozen=True)
class PublishedCell:
    port: int
    n: int
    k: int
    limit_s: float
    methods: tuple[str, ...]
    times: Mapping[str, Time]
    nodes: Mapping[str, int | None] = field(default_factory=dict)
    cuts: Mapping[str, int | None] = field(default_factory=dict)

    @property
    def best(self) -> Best:
        """The smallest time printed in the cell (ties: the first method of the table); all failed: the limit."""
        solved = [(self.times[m].seconds, i, m) for i, m in enumerate(self.methods) if not self.times[m].censored]
        if not solved:
            return Best(self.limit_s, True, None)
        seconds, _, method = min(solved)
        return Best(seconds, False, method)


@dataclass(frozen=True)
class PublishedTable:
    name: str
    kappa: float
    time_limit_s: float
    methods: tuple[str, ...]
    cells: Mapping[tuple[int, int], PublishedCell]


@dataclass(frozen=True)
class Published:
    tables: Mapping[str, PublishedTable]
    source: Mapping[str, Any]
    hardware: Mapping[str, Any]
    setup: tuple[str, ...]
    path: str
    sha256: str


def parse_time(value: Any, limit: float) -> Time:
    """A printed time: a number of seconds within ``[0, limit]``, or the string ``">L"`` with ``L`` the limit of
    the table (not solved within it: a bound at the limit). Anything else is a ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"time {value!r} is not a number of seconds or a string '>L'")
    if isinstance(value, int | float):
        if not (math.isfinite(value) and 0.0 <= value <= limit):
            raise ValueError(f"time {value!r} is not within [0, {limit:g}] s")
        return Time(float(value), False)
    if isinstance(value, str) and value.strip().startswith(">"):
        try:
            bound = float(value.strip()[1:])
        except ValueError:
            raise ValueError(f"time {value!r} is not '>L'") from None
        if bound != limit:
            raise ValueError(f"time {value!r}: the bound must be the limit {limit:g} s of its table")
        return Time(float(limit), True)
    raise ValueError(f"time {value!r} is not a number of seconds or a string '>L'")


def _published_table(name: str, raw: Mapping) -> PublishedTable:
    methods = tuple(raw.get("methods") or ())
    if not methods:
        raise ValueError(f"{name}: no methods")
    limit = float(raw["time_limit_s"])
    cells: dict[tuple[int, int], PublishedCell] = {}
    for item in raw.get("cells") or ():
        port, k = int(item["port"]), int(item["k"])
        if port not in ORLIB_N or int(item["n"]) != ORLIB_N[port]:
            raise ValueError(f"{name}: port {port} has n = {item['n']}, the OR-Library file has {ORLIB_N.get(port)}")
        if (port, k) in cells:
            raise ValueError(f"{name}: cell (port {port}, k {k}) twice")
        raw_times = item.get("time_s") or {}
        missing = [m for m in methods if m not in raw_times]
        if missing:
            raise ValueError(f"{name}: cell (port {port}, k {k}) has no time for {missing[0]}")
        try:
            times = {m: parse_time(raw_times[m], limit) for m in methods}
        except ValueError as exc:
            raise ValueError(f"{name}: cell (port {port}, k {k}): {exc}") from None
        cells[(port, k)] = PublishedCell(port, int(item["n"]), k, limit, methods, times,
                                         dict(item.get("nodes") or {}), dict(item.get("cuts") or {}))
    if sorted(cells) != sorted(_GRID):
        raise ValueError(f"{name}: expected the 15 cells of 5 ports x k in (5, 10, 20), found {len(cells)}")
    return PublishedTable(name, float(raw["kappa"]), limit, methods, cells)


def load_published(path: Path | str) -> Published:
    """The published times of ``configs/E2/m11_bcw_published.yaml`` (header of the file), validated: both tables with
    their 15 cells, every printed time a number within the limit or ``">L"``."""
    path = Path(path)
    data = path.read_bytes()
    raw = yaml.safe_load(data.decode("utf-8"))
    if not isinstance(raw, Mapping) or raw.get("kind") != _PUBLISHED_KIND:
        raise ValueError(f"{path}: not a {_PUBLISHED_KIND!r} file (kind = "
                         f"{raw.get('kind') if isinstance(raw, Mapping) else None!r})")
    tables = raw.get("tables") or {}
    if set(tables) != set(_TABLES):
        raise ValueError(f"{path}: the tables must be {_TABLES}, found {sorted(tables)}")
    return Published({name: _published_table(name, tables[name]) for name in _TABLES}, dict(raw.get("source") or {}),
                     dict(raw.get("hardware") or {}), tuple(raw.get("setup") or ()), str(path),
                     hashlib.sha256(data).hexdigest())


# --------------------------------------------------------------------------------------------
# Our times and the ratios
# --------------------------------------------------------------------------------------------


def our_time(rows: Sequence[Mapping]) -> OurTime:
    """The value of a cell for one method (module docstring): the median of the wall times of its runs, the limit for a
    run that did not certify within it; a bound when at least half of the runs failed."""
    if not rows:
        raise ValueError("no rows: nothing to take the median of")
    limit = float(rows[0]["time_limit_s"])
    times, failed = [], 0
    for row in rows:
        if float(row["time_limit_s"]) != limit:
            raise ValueError("the rows of one cell have different time limits")
        if row.get("certified") is True and float(row["wall_s"]) <= limit:
            times.append(float(row["wall_s"]))
        else:
            times.append(limit)
            failed += 1
    n = len(times)
    return OurTime(float(np.median(times)), failed > (n - 1) / 2, n, failed)


def cell_ratio(ours: Time, best: Time) -> Ratio:
    """``ours / best`` with the bounds of the module docstring."""
    if ours.censored and best.censored:
        return Ratio(None, None, True)
    if not best.seconds > 0.0:
        raise ValueError(f"the best published time must be positive, got {best.seconds}")
    return Ratio(ours.seconds / best.seconds, "lower" if ours.censored else ("upper" if best.censored else None),
                 False)


def _median_bound(values: list[tuple[float, str | None]]) -> str | None:
    """Whether the median of the ratios rests on a bound: the bounds of the middle element(s) of the sample."""
    if not values:
        return None
    ordered = sorted(values, key=lambda v: v[0])
    n = len(ordered)
    middle = [ordered[n // 2]] if n % 2 else [ordered[n // 2 - 1], ordered[n // 2]]
    kinds = {b for _, b in middle}
    if kinds == {None}:
        return None
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _summary(views: Sequence[Mapping | None], best: Sequence[Best]) -> dict:
    """The summary of one view over the 15 cells: ``views[i]`` is the view of cell ``i`` (``None`` if the cell is
    missing)."""
    kept = [(v["ratio"], v["bound"]) for v in views if v is not None and not v["excluded"]]
    return {
        "n_cells": sum(v is not None for v in views),
        "n_ours_solved": sum(1 for v in views if v is not None and not v["censored"]),
        "n_pub_solved": sum(1 for v, b in zip(views, best, strict=True) if v is not None and not b.censored),
        "n_excluded": sum(1 for v in views if v is not None and v["excluded"]),
        "n_ratio_cells": len(kept),
        "n_lower_bound": sum(1 for _, b in kept if b == "lower"),
        "n_upper_bound": sum(1 for _, b in kept if b == "upper"),
        "median_ratio": float(np.median([r for r, _ in kept])) if kept else None,
        "median_bound": _median_bound(kept),
    }


# --------------------------------------------------------------------------------------------
# Audits
# --------------------------------------------------------------------------------------------


def audit_rows(rows: Iterable[Mapping]) -> dict:
    """The counts and conflicts of the module docstring over the rows of the stage."""
    rows = list(rows)
    certified = [r for r in rows if r.get("certified") is True]

    def within_rule(r: Mapping) -> bool:
        rel, ab = r.get("rel_gap"), r.get("abs_gap")
        return bool((rel is not None and rel <= STOP_REL_GAP * (1 + _GAP_SLACK))
                    or (ab is not None and ab <= STOP_ABS_GAP * (1 + _GAP_SLACK)))

    conflicts = []
    cells: dict[tuple, list[Mapping]] = defaultdict(list)
    for r in certified:
        if r.get("primal") is not None and r.get("dual") is not None:
            cells[(r["table"], r["port"], r["k"])].append(r)
    for (table, port, k), group in sorted(cells.items()):
        low = min(r["primal"] for r in group)
        high = max(r["dual"] for r in group)
        if high > low + 1e-9 + _BOUND_TOL_REL * abs(low):
            conflicts.append({"table": table, "port": port, "k": k, "max_dual": high, "min_primal": low,
                              "methods": sorted({r["method"]["name"] for r in group})})
    return {
        "n_rows": len(rows),
        "n_infeasible": sum(1 for r in rows if r.get("feasible") is False),
        "n_over_budget": sum(1 for r in rows if r.get("budget_ok") is False),
        "n_return_violated": sum(1 for r in rows if r.get("return_ok") is False),
        "n_certified_over_tolerance": sum(1 for r in certified if not within_rule(r)),
        "bound_conflicts": conflicts,
    }


def _audit_warnings(audit: Mapping) -> list[str]:
    out = []
    if audit["n_infeasible"]:
        out.append(f"{audit['n_infeasible']} rows are not feasible (feasible = false)")
    if audit["n_over_budget"]:
        out.append(f"{audit['n_over_budget']} rows overran their limit by more than the tolerance of the stage "
                   "(budget_ok = false): their times are as measured")
    if audit["n_return_violated"]:
        out.append(f"{audit['n_return_violated']} rows of T4 violate the return row (return_ok = false)")
    if audit["n_certified_over_tolerance"]:
        out.append(f"{audit['n_certified_over_tolerance']} certified rows have a recorded gap above the stopping rule "
                   f"(relative {STOP_REL_GAP:g} / absolute {STOP_ABS_GAP:g}): exhaustion proofs of OA, or a gap "
                   "that is not the solver's")
    if audit["bound_conflicts"]:
        cells = ", ".join(f"{c['table']} port{c['port']} k{c['k']}" for c in audit["bound_conflicts"][:5])
        out.append(f"{len(audit['bound_conflicts'])} cells have a certified dual bound above the incumbent of another "
                   f"certified run (a wrong bound or a wrong incumbent; e.g. {cells}): investigate before using the "
                   "times")
    return out


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def _check_rows(rows: Sequence[Mapping], published: Published) -> None:
    if not rows:
        raise ValueError("no rows: the analysis reads the rows of a registered run of the stage m11")
    seen: set[tuple] = set()
    for row in rows:
        where = row.get("cell_key") or row.get("table")
        if row.get("error"):
            raise ValueError(f"a row carries an error ({row['error']}): fix and rerun the cell")
        if row.get("stage") != "m11":
            raise ValueError(f"{where}: this analysis reads rows of the stage 'm11', found one of "
                             f"{row.get('stage')!r}")
        if row.get("table") not in _TABLES:
            raise ValueError(f"{where}: unknown table {row.get('table')!r}")
        if row["method"]["name"] not in _METHODS:
            raise ValueError(f"{where}: unknown method {row['method']['name']!r}, expected one of {_METHODS}")
        key = (row["table"], row["port"], row["k"], row["method"]["name"], row["seed"])
        if key in seen:
            raise ValueError(f"the cell {key} is in the rows twice (two runs given for one stage?)")
        seen.add(key)
        cell = published.tables[row["table"]].cells.get((row["port"], row["k"]))
        if cell is None:
            raise ValueError(f"{where}: (port {row['port']}, k {row['k']}) is not an instance of the published table")
        if int(row["n"]) != cell.n:
            raise ValueError(f"{where}: n = {row['n']}, the published instance of port {row['port']} has n = {cell.n}")


def _view(runs: Sequence[Mapping], best: Best, method: str | None = None) -> dict:
    value = our_time(runs)
    ratio = cell_ratio(value, best)
    return {"method": method, "time_s": value.seconds, "censored": value.censored, "n_runs": value.n_runs,
            "n_failed": value.n_failed, "ratio": ratio.ratio, "bound": ratio.bound, "excluded": ratio.excluded}


def _table_block(name: str, table: PublishedTable, rows: Sequence[Mapping], allow_incomplete: bool,
                 warnings: list[str]) -> dict:
    by_cell: dict[tuple[int, int], dict[str, list[Mapping]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_cell[(row["port"], row["k"])][row["method"]["name"]].append(row)
    missing = [{"port": port, "k": k} for port, k in sorted(table.cells) if (port, k) not in by_cell]
    if missing and not allow_incomplete:
        raise ValueError(f"{name}: {len(missing)} of 15 instances are missing from the rows, e.g. {missing[0]}: the stage "
                         "is not complete (--allow-incomplete analyses a partial grid)")
    if missing:
        warnings.append(f"{name}: {len(missing)} of 15 instances missing (a partial grid): "
                        + ", ".join(f"port{m['port']} k{m['k']}" for m in missing))
    method_sets = {tuple(sorted(by_cell[key])) for key in by_cell}
    if len(method_sets) > 1:
        raise ValueError(f"{name}: the instances do not all have the same methods ({sorted(method_sets)})")
    methods = [m for m in _METHODS if m in next(iter(method_sets))]
    for method in methods:
        counts = {len(by_cell[key][method]) for key in by_cell}
        if len(counts) > 1:
            raise ValueError(f"{name}: the instances have different numbers of seeds for {method} ({sorted(counts)}): "
                             "the grid is incomplete or mixed")
    limits = {float(r["time_limit_s"]) for r in rows}
    if limits != {table.time_limit_s}:
        warnings.append(f"{name}: time_limit_s of the rows is {sorted(limits)}, the published table's is "
                        f"{table.time_limit_s:g} s: the bounds are at different limits (a rehearsal?)")
    cells = []
    for key in sorted(by_cell):
        cell = table.cells[key]
        best = cell.best
        views: dict[str, dict | None] = {m: _view(by_cell[key][m], best, m) for m in methods}
        pick = min(methods, key=lambda m: (views[m]["time_s"], _METHODS.index(m)))
        stack = {**views[pick], "method": pick}
        if PRIMARY_VIEW not in views:
            raise ValueError(f"{name}: no rows of {PRIMARY_VIEW!r}, the method of the verdict")
        cells.append({
            "port": key[0], "n": cell.n, "k": key[1],
            "published": {m: {"time_s": t.seconds, "censored": t.censored} for m, t in cell.times.items()},
            "published_best": {"time_s": best.seconds, "censored": best.censored, "method": best.method},
            "views": {**views, STACK_VIEW: stack}})
    best_list = [table.cells[(c["port"], c["k"])].best for c in cells]
    order = [PRIMARY_VIEW, *(m for m in methods if m != PRIMARY_VIEW), STACK_VIEW]
    summary = {view: _summary([c["views"].get(view) for c in cells], best_list) for view in order}
    median = summary[PRIMARY_VIEW]["median_ratio"]
    above = None if median is None else bool(median > THRESHOLD)
    excluded = [{"port": c["port"], "k": c["k"], "reason": "neither side solved the instance within the limit"}
                for c in cells if c["views"][PRIMARY_VIEW]["excluded"]]
    return {
        "time_limit_s": table.time_limit_s, "kappa": table.kappa, "methods": methods,
        "published_methods": list(table.methods), "cells": cells, "missing": missing, "excluded": excluded,
        "summary": summary,
        "verdict": {"view": PRIMARY_VIEW, "median_ratio": median, "threshold": THRESHOLD, "above_threshold": above,
                    "statement": None if above is None else _STATEMENTS[above]},
    }


def m11_report(rows: Iterable[Mapping], published: Published, *, allow_incomplete: bool = False) -> dict:
    """The report of the module docstring from the rows of the stage and the published numbers. No I/O:
    :func:`run_command` adds the inputs and the code state. A table without rows, an instance without rows and different
    numbers of seeds or methods across the instances stop the analysis, unless ``allow_incomplete`` (partial grids and
    rehearsals), which lists the missing parts in the report."""
    rows = list(rows)
    _check_rows(rows, published)
    warnings: list[str] = []
    blocks: dict[str, dict] = {}
    for name in _TABLES:
        mine = [r for r in rows if r["table"] == name]
        if not mine:
            if not allow_incomplete:
                raise ValueError(f"{name}: no rows for the table: the grid of the stage is missing it "
                                 "(--allow-incomplete analyses a partial grid)")
            warnings.append(f"{name}: no rows (the table is not analysed)")
            continue
        blocks[name] = _table_block(name, published.tables[name], mine, allow_incomplete, warnings)
    audit = audit_rows(rows)
    warnings += _audit_warnings(audit)
    above = [name for name, block in blocks.items() if block["verdict"]["above_threshold"]]
    undecided = [name for name, block in blocks.items() if block["verdict"]["above_threshold"] is None]
    overall = True if above else (None if len(undecided) == len(blocks) else False)
    return {
        "kind": "e2-m11", "schema": "e2-m11/1", "plan_version": "v2.4",
        "role": "validation of the exact stack against the published times (not a test of a hypothesis)",
        "settings": {"threshold": THRESHOLD, "stop_rel_gap": STOP_REL_GAP, "stop_abs_gap": STOP_ABS_GAP,
                     "grid": [{"port": p, "k": k} for p, k in _GRID]},
        "primary_view": PRIMARY_VIEW,
        "prespecified": PRESPECIFIED,
        "published": {"source": dict(published.source), "hardware": dict(published.hardware),
                      "setup": list(published.setup), "path": published.path, "sha256": published.sha256},
        "tables": blocks,
        "verdict": {"threshold": THRESHOLD, "above_threshold": overall, "tables_above": above,
                    "tables_undecided": undecided, "statement": None if overall is None else _STATEMENTS[overall]},
        "audit": audit,
        "warnings": warnings, "provenance_clean": None, "inputs": [], "code": None,
    }


# --------------------------------------------------------------------------------------------
# Tables and files
# --------------------------------------------------------------------------------------------

_ROW_COLUMNS = (
    "table", "port", "n", "k", "method", "config_id", "seed", "status", "proved", "certified", "wall_s", "end_s", "cpu_s",
    "time_limit_s", "primal", "dual", "abs_gap", "rel_gap", "rel_gap_cplex", "nodes", "n_cuts", "iterations",
    "feasible", "return_ok", "budget_ok", "overrun_s", "r_bar", "r_realised", "host", "cell_key")
_VIEWS = (*_METHODS, STACK_VIEW)
_CELL_FIELDS = ("time_s", "censored", "n_runs", "n_failed", "ratio", "bound")


def _cell_rows(report: Mapping) -> list[dict]:
    out = []
    for table, block in report["tables"].items():
        for cell in block["cells"]:
            row: dict[str, Any] = {"table": table, "port": cell["port"], "n": cell["n"], "k": cell["k"]}
            for view in _VIEWS:
                v = cell["views"].get(view)
                for key in _CELL_FIELDS:
                    row[f"{view}_{key}"] = None if v is None else v[key]
                if view == STACK_VIEW:
                    row["stack_method"] = None if v is None else v["method"]
            row["excluded"] = cell["views"][PRIMARY_VIEW]["excluded"]
            best = cell["published_best"]
            row.update(published_best_s=best["time_s"], published_best_method=best["method"],
                       published_censored=best["censored"])
            for method in block["published_methods"]:
                t = cell["published"][method]
                row[f"pub_{method}"] = f">{t['time_s']:g}" if t["censored"] else t["time_s"]
            out.append(row)
    return out


def _tables(report: Mapping, rows: Sequence[Mapping]) -> dict[str, pd.DataFrame]:
    cell_columns = ["table", "port", "n", "k"]
    for view in _VIEWS:
        cell_columns += [f"{view}_{key}" for key in _CELL_FIELDS]
        if view == STACK_VIEW:
            cell_columns.append("stack_method")
    cell_columns += ["excluded", "published_best_s", "published_best_method", "published_censored",
                     "pub_algorithm3", "pub_algorithm3_inout", "pub_cplex_bigm", "pub_cplex_misoco"]
    summary = [{"table": table, "view": view, **s}
               for table, block in report["tables"].items() for view, s in block["summary"].items()]
    summary_columns = ["table", "view", "n_cells", "n_ours_solved", "n_pub_solved", "n_excluded", "n_ratio_cells",
                       "n_lower_bound", "n_upper_bound", "median_ratio", "median_bound"]
    flat = []
    for row in rows:
        entry = {key: row.get(key) for key in _ROW_COLUMNS}
        entry["method"] = row["method"]["name"]
        flat.append(entry)
    return {"cells.csv": pd.DataFrame(_cell_rows(report), columns=cell_columns),
            "summary.csv": pd.DataFrame(summary, columns=summary_columns),
            "rows.csv": pd.DataFrame(flat, columns=list(_ROW_COLUMNS))}


def write_m11(report: Mapping, rows: Sequence[Mapping], out_dir: Path) -> list[Path]:
    """``m11.json`` (strict JSON: missing numbers are ``null``) and the tables ``cells.csv``, ``summary.csv``,
    ``rows.csv`` in ``out_dir``."""
    from cif.experiments.e2_eligibility import _json_safe

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [out_dir / "m11.json"]
    written[0].write_text(json.dumps(_json_safe(dict(report)), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")
    for name, frame in _tables(report, rows).items():
        written.append(out_dir / name)
        frame.to_csv(written[-1], index=False)
    return written


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def _provenance_warnings(inputs: Sequence[Mapping], code: Mapping) -> list[str]:
    """What keeps the rows from being a clean, registered full run of the stage, or the code from being a committed
    tree: the validation is not a confirmatory run (its seeds are calibration ones), but its numbers go into the
    paper."""
    warnings = []
    for entry in inputs:
        if entry["role"] != "rows":
            continue
        if entry.get("run_id") is None:
            warnings.append(f"{entry['source']}: not a registered run directory (no manifest.json): its rows cannot be "
                            "traced to a run")
            continue
        name = entry["run_id"]
        if entry.get("mode") not in ("full", "ingested"):
            warnings.append(f"{name}: manifest mode = {entry.get('mode')!r} (rehearsals are not the validation)")
        if entry.get("git_dirty") is not False:
            warnings.append(f"{name}: manifest git_dirty = {entry.get('git_dirty')!r} (not a run from a clean tree)")
        if entry.get("stage") not in (None, "m11"):
            warnings.append(f"{name}: config stage = {entry.get('stage')!r}, expected 'm11'")
        if entry.get("n_errors") or entry.get("n_missing"):
            warnings.append(f"{name}: the run has {entry.get('n_errors')} cells with errors and "
                            f"{entry.get('n_missing')} missing")
    if code.get("git_dirty") is not False:
        warnings.append(f"the analysis code is from a dirty or unknown tree (git_dirty = {code.get('git_dirty')!r})")
    return warnings


def run_command(args) -> int:
    """``e2_analysis m11``: read the registered run(s), compare with the published numbers, write the report and the
    tables, print the summary of each table."""
    from cif.experiments import e2_analysis as an
    from cif.experiments import e2_pilot

    published = load_published(args.published)
    rows, errors = e2_pilot.load_rows(args.rows)
    all_rows = [*rows, *errors]
    report = m11_report(all_rows, published, allow_incomplete=args.allow_incomplete)
    inputs = [*an._run_inputs(args.rows, "rows"),
              {**e2_pilot._file_entry(Path(args.published)), "role": "published"}]
    state = registry._git_state()
    code = {"git_sha": state.get("sha") or None, "git_dirty": state.get("dirty"), "branch": state.get("branch") or None}
    problems = _provenance_warnings(inputs, code)
    versions = {"python": sys.version.split()[0], **{p: metadata.version(p) for p in ("numpy", "pandas", "scipy",
                                                                                      "pyscipopt", "highspy")}}
    report.update(inputs=inputs, code=code, versions=versions, warnings=[*problems, *report["warnings"]],
                  provenance_clean=bool(not problems))
    write_m11(report, all_rows, args.out)
    print(f"e2-m11: {args.out}: {len(all_rows)} rows, provenance clean = {report['provenance_clean']}")
    for name, block in report["tables"].items():
        for view, s in block["summary"].items():
            median = "n/a" if s["median_ratio"] is None else f"{s['median_ratio']:.4g}"
            print(f"  {name} {view}: median ratio {median}"
                  f"{' (rests on a ' + str(s['median_bound']) + ' bound)' if s['median_bound'] else ''}, "
                  f"solved within the limit: ours {s['n_ours_solved']}/{s['n_cells']}, best published "
                  f"{s['n_pub_solved']}/{s['n_cells']}, excluded {s['n_excluded']}")
        above = block["verdict"]["above_threshold"]
        print(f"  {name} verdict ({PRIMARY_VIEW}): "
              + ("undecided (no ratio)" if above is None else f"{'above' if above else 'not above'} {THRESHOLD:g}"))
    overall = report["verdict"]["above_threshold"]
    print("  overall: " + ("undecided" if overall is None else f"{'above' if overall else 'not above'} {THRESHOLD:g}"))
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0

