"""Eligibility of instances for RQ2 and H1c: the screening stage (ANALYSIS_PLAN §12, entries A1 and A3 of 2026-10-11).

**The rule (A1).** Methods are compared only on **non-trivial** instances, where the common start (§5) does
not reach the main target of the time-to-target, ``ĝ_start = (f_start − f*)/s* > τ = 10⁻³``. On the others
(trivial instances) PI⁺ is 0 for every method, a ratio 0/0, and the outcome "practically equivalent" would be
settled by the composition of the instances and not by the methods (consult
``docs/reviews/2026-10-11-q1-pilot-design-consult.md``). The decision is made **once, at the reference stage,
before any method runs on the instance**, and never revised; it is a property of the instance, not of a
method:

* **non-trivial** — a solution better than the start by more than ``τ·s`` has been found;
* **trivial** — a dual bound proves ``ĝ_start ≤ τ``: ``f_start − dual_bound ≤ τ·s`` (a full certificate of §5,
  or SCIP warm-started at the start, stopped as soon as its dual bound is that close);
* **undecided** — neither within the limit (2 h): excluded and counted separately.

``s`` is the scale of the common start: the scale of the optimum, ``s*``, is not known when the decision is
made, which is why the §5 certificate also takes the scale of the start for ``limits/absgap`` (halved there for
safety). Using it on both sides keeps the rule an exact statement about numbers on the row (``f_start``,
``s_start``, the incumbent and the dual bound), so the decision can be recomputed from the evidence
(:func:`verify_rows`). Where the certified ``s*`` of a non-trivial instance would put it on the other side,
:func:`reference_check` reports it; the decision stays.

**Scanning (A1).** The cells C1–C4 are filled by scanning seeds in increasing order until ``n`` non-trivial
instances (test seeds from 2000; calibration: the seeds of the plan, then, for ``factor``, from 1024 separately
for every N and, for ``boot``, from 1110), at most 400 test seeds per cell and 200 calibration seeds per N.
:class:`ScanSpec` is one such cell. The scan is run in **rounds** (each a registered run of the stage
``eligibility``, :mod:`cif.experiments.e2_benchmark`): the report of the rows so far (:func:`build_report`) is
the *state* the next round reads (:func:`progress`, :func:`next_block`), and the report of the last round is the
input of every later stage (:func:`selected_units`). What counts as scanned is a prefix of the order in which
every seed has a decision; the instances used are the first ``n`` non-trivial ones of it, and the protocol of the
scan ends at the ``n``-th of them (a block may have gone past it: those rows are kept but are not part of the
sample, the estimate of A3 included). A cell with fewer than 20 non-trivial test instances (fewer than 8
calibration ones) is **descriptive** and leaves the pooled estimates of H2 and H2b; a calibration size with too
few instances takes its configurations from the nearest size (:func:`nearest_sizes`). C5 (the 125 ORlib points,
a fixed set) enters the pooled estimate only with at least 10 non-trivial binding points in at least 3
portfolios (ports), and D1 is descriptive throughout.

**Reading the rows.** The evidence is taken from the rows of the stage ``eligibility`` and, for a seed that has
none (the 24 calibration instances of the registered reference run, certified before any method ran on
them), from the rows of the stage ``reference``: the same rule applied to a full certificate. A later
reference run of an instance never overrides the decision of its screening.

**The estimate (A3)** is made for every cell from the scanned binding instances (:func:`estimate`): the share of
trivial ones with a 95 % Wilson interval (undecided instances stay in the denominator), the distribution of
``ĝ_start`` in basis points of the certainty equivalent (``(f_start − f)·10⁴``: exact where the run proved its
incumbent optimal, a lower bound otherwise, ``n_exact`` says how many are exact) and the time of the
certificate on the trivial instances with its level (``full`` = 1e-6·s, ``tau``). Instances with a non-binding
cardinality constraint (C5, D1) are counted separately: the question of RQ2 is about binding instances. The
share is of the instances scanned until the ``n``-th non-trivial one, which over-represents the non-trivial
ones slightly (inverse sampling); the plan prescribes the plain share and Wilson's interval.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from cif.metrics.anytime import TAU_MAIN
from cif.stats.rq1 import wilson_interval

__all__ = [
    "C5_MIN_POINTS",
    "C5_MIN_PORTS",
    "CLASSES",
    "MIN_NONTRIVIAL",
    "REPORT_KIND",
    "ScanSpec",
    "Verdict",
    "build_report",
    "cell_state",
    "classify",
    "estimate",
    "evidence_of_row",
    "format_estimate",
    "in_summary",
    "nearest_sizes",
    "next_block",
    "progress",
    "reference_check",
    "row_verdict",
    "scan_order",
    "selected_units",
    "spec_from_group",
    "verify_rows",
]

TAU = TAU_MAIN
REPORT_KIND = "e2-eligibility"
CLASSES = ("trivial", "nontrivial", "undecided")
#: Fewer non-trivial instances than this make a cell descriptive (§12 A1): test cells, calibration sets.
MIN_NONTRIVIAL = {"calibration": 8, "test": 20}
#: C5 enters the pooled estimate with at least this many non-trivial binding points in this many ports.
C5_MIN_POINTS = 10
C5_MIN_PORTS = 3
#: ``dual_bound`` closer than this (in units of the scale) to the incumbent: the incumbent is optimal.
EXACT_TOL = 1e-5
BP = 1e4  # one basis point of the certainty equivalent
_SCAN_KEYS = {"first", "head", "then_from", "cap", "target", "target_from", "min_nontrivial", "block",
              "first_block", "state"}
Unit = int | tuple[int, int]


# --------------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------------


class Verdict(NamedTuple):
    """The class of an instance and the numbers behind it. ``gap_lo = (f_start − f_best)/s_start`` is what a
    solution found shows of ``ĝ_start``; ``gap_hi = (f_start − dual_bound)/s_start`` what a dual bound proves
    it cannot exceed (``inf`` without one); ``threshold = τ·s_start`` is in the units of the objective."""

    label: str
    reason: str
    gap_lo: float
    gap_hi: float
    threshold: float


def _finite(x: Any) -> bool:
    return x is not None and math.isfinite(x)


def classify(
    *, f_start: float, s_start: float, f_best: float | None, dual_bound: float | None, tau: float = TAU,
    status: str | None = None,
) -> Verdict:
    """The class of an instance from what a screening run knows (module docstring).

    ``f_best`` is the best value any run found (``None``/``inf``: none; a value above ``f_start`` is no
    improvement), ``dual_bound`` the best dual bound (``None``/``-inf``: none), ``status`` the solver's, used for
    the reason of an undecided instance only. Non-trivial: ``f_start − f_best > τ·s_start``; trivial:
    ``f_start − dual_bound ≤ τ·s_start``; both is a contradiction (``bounds_cross``) and undecided, as is a start
    that does not exist.
    """
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be a positive number, got {tau}")
    if f_start is None or not math.isfinite(f_start):
        return Verdict("undecided", "no_start", 0.0, math.inf, math.nan)
    if not (math.isfinite(s_start) and s_start > 0.0):
        raise ValueError(f"s_start must be a finite positive number, got {s_start}")
    threshold = tau * s_start
    lo = max(0.0, f_start - float(f_best)) if _finite(f_best) else 0.0
    hi = f_start - float(dual_bound) if _finite(dual_bound) else math.inf
    nontrivial, trivial = lo > threshold, hi <= threshold
    if nontrivial and trivial:
        label, reason = "undecided", "bounds_cross"
    elif nontrivial:
        label, reason = "nontrivial", "improvement"
    elif trivial:
        label, reason = "trivial", "dual_bound"
    else:
        label, reason = "undecided", "time_limit" if status == "timelimit" else "open"
    return Verdict(label, reason, lo / s_start, hi / s_start, threshold)


def evidence_of_row(row: Mapping) -> dict:
    """What the row says about the instance: ``stage``, ``certificate`` (``"tau"`` / ``"full"``: the level the
    screening aimed at; ``"none"`` for a reference run that did not certify), ``f_start``, ``s_start``,
    ``f_best``, ``dual_bound``, ``status``, ``t_cert_s``. A row of the stage ``eligibility`` carries its
    ``evidence``; a row of the stage ``reference`` is read as a full certificate."""
    stage = row.get("stage")
    if stage == "eligibility":
        ev = dict(row["evidence"])
        return {"stage": stage, "certificate": ev["certificate"], "f_start": ev["f_start"],
                "s_start": ev["s_start"], "f_best": ev["f_best"], "dual_bound": ev["dual_bound"],
                "status": ev.get("status", row.get("status")), "t_cert_s": row.get("t_cert_s")}
    if stage == "reference":
        inf = math.inf
        f_final, f_star = row.get("f_final"), row.get("f_star")
        f_best = min(row["start_f"], inf if f_final is None else f_final, inf if f_star is None else f_star)
        dual = (row.get("detail") or {}).get("dual_bound")
        if dual is None and f_star is not None and row.get("certified"):
            dual = f_star - 1e-6 * (row.get("scale_star") or 0.0)
        return {"stage": stage, "certificate": "full" if row.get("certified") else "none",
                "f_start": row["start_f"], "s_start": row["start_scale"], "f_best": f_best,
                "dual_bound": -inf if dual is None else dual, "status": row.get("status"),
                "t_cert_s": row.get("t_cert_s")}
    raise ValueError(f"a row of the stage {stage!r} is no evidence about eligibility")


def row_verdict(row: Mapping, tau: float = TAU) -> Verdict:
    """:func:`classify` applied to :func:`evidence_of_row`."""
    ev = evidence_of_row(row)
    return classify(f_start=ev["f_start"], s_start=ev["s_start"], f_best=ev["f_best"],
                    dual_bound=ev["dual_bound"], tau=tau, status=ev["status"])


def verify_rows(rows: Iterable[Mapping], tau: float = TAU) -> list[str]:
    """Problems of the rows of the stage ``eligibility``: the recorded class (and gaps) must follow from the
    recorded evidence. An empty list is an audit passed."""
    problems = []
    for row in rows:
        if row.get("stage") != "eligibility" or row.get("skipped") or "evidence" not in row:
            continue
        ev = row["evidence"]
        v = classify(f_start=ev["f_start"], s_start=ev["s_start"], f_best=ev["f_best"],
                     dual_bound=ev["dual_bound"], tau=float(ev.get("tau", tau)), status=ev.get("status"))
        where = row.get("instance_id") or row.get("cell_key")
        if v.label != row.get("eligibility"):
            problems.append(f"{where}: recorded class {row.get('eligibility')!r}, the evidence gives {v.label!r}")
        elif not (_close(v.gap_lo, ev.get("gap_lo")) and _close(v.gap_hi, ev.get("gap_hi"))):
            problems.append(f"{where}: recorded gaps ({ev.get('gap_lo')}, {ev.get('gap_hi')}) differ from the "
                            f"evidence ({v.gap_lo}, {v.gap_hi})")
    return problems


def _close(a: float, b: float | None) -> bool:
    if b is None:
        return not math.isfinite(a)
    return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


# --------------------------------------------------------------------------------------------
# The scan
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanSpec:
    """One cell of the scan: ``order`` the units (seeds, or ``(port, lam_index)`` of a fixed set of points) in
    the order they are scanned, at most the cap; ``target`` the number ``n`` of non-trivial instances wanted
    (``None``: a fixed set, every point is screened); ``min_nontrivial`` below which the cell is descriptive;
    ``block`` / ``first_block`` fix the size of a round (``None``: sized from what is known,
    :func:`next_block`)."""

    cell: str
    order: tuple
    target: int | None
    min_nontrivial: int
    family: str = "factor"
    n: int | None = None
    config_size: int | None = None
    block: int | None = None
    first_block: int | None = None
    points: bool = False


def scan_order(
    *, first: int | None = None, head: Sequence[int] = (), then_from: int | None = None, cap: int,
) -> tuple[int, ...]:
    """Seeds in the order of the scan: ``head`` (the seeds of the plan) first, then ``then_from, then_from + 1,
    …`` — or, without a head, ``first, first + 1, …`` — in all ``cap`` of them (the cap on scanned seeds)."""
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}")
    head = [int(s) for s in head]
    if len(head) > cap:
        raise ValueError(f"the {len(head)} seeds of the head exceed the cap of {cap}")
    start = then_from if head else first
    if start is None:
        raise ValueError("a scan order needs 'first' (no head) or a head and 'then_from'")
    order = (*head, *range(int(start), int(start) + cap - len(head)))
    if len(set(order)) != len(order):
        raise ValueError("a scan order must not repeat a seed: the extras overlap the head")
    return tuple(order)


def next_block(spec: ScanSpec, prefix: int, found: int) -> list:
    """The units of the next round given that the first ``prefix`` of the order are screened and ``found`` of
    them are non-trivial; empty if the cell is finished (``target`` reached or the cap spent). A fixed set of
    points is screened in one round. Otherwise the size is ``block`` (``first_block`` in the first round) or,
    unset, enough to expect the missing instances with a margin of a quarter, from the share seen so far
    smoothed by one success and one failure: ``⌈1.25·need·(prefix + 2)/(found + 1)⌉``, within what is left."""
    remaining = len(spec.order) - int(prefix)
    if remaining <= 0:
        return []
    if spec.target is None:
        size = remaining
    else:
        need = spec.target - int(found)
        if need <= 0:
            return []
        if prefix == 0 and spec.first_block:
            size = spec.first_block
        elif spec.block:
            size = spec.block
        else:
            size = -(-(5 * need * (int(prefix) + 2)) // (4 * (int(found) + 1)))
    size = max(1, min(int(size), remaining))
    return list(spec.order[int(prefix): int(prefix) + size])


# --------------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------------


def _unit_of(row: Mapping, points: bool) -> Unit:
    if points:
        return int(row["port"]), int(row["lam_index"])
    return int(row["seed"])


def _unit_json(unit: Unit):
    return list(unit) if isinstance(unit, tuple) else unit


def _decision(row: Mapping, unit: Unit, tau: float) -> dict:
    """The decision a row records, as the report keeps it."""
    base = {"unit": _unit_json(unit), "instance_id": row.get("instance_id"), "stage": row.get("stage")}
    if row.get("skipped") == "not_binding":
        return {**base, "class": "nonbinding", "reason": "not_binding", "certificate": None, "status": "skipped",
                "gap_lo": None, "gap_hi": None, "f_start": None, "s_start": None, "f_best": None,
                "dual_bound": None, "t_cert_s": None}
    ev = evidence_of_row(row)
    v = classify(f_start=ev["f_start"], s_start=ev["s_start"], f_best=ev["f_best"],
                 dual_bound=ev["dual_bound"], tau=tau, status=ev["status"])
    if row.get("stage") == "eligibility" and row.get("eligibility") not in (None, v.label):
        raise ValueError(f"{row.get('instance_id')}: the recorded class {row.get('eligibility')!r} does not follow "
                         f"from its evidence ({v.label!r}); run the audit (verify_rows)")
    return {**base, "class": v.label, "reason": v.reason, "certificate": ev["certificate"],
            "status": ev["status"], "gap_lo": v.gap_lo, "gap_hi": v.gap_hi, "f_start": ev["f_start"],
            "s_start": ev["s_start"], "f_best": ev["f_best"], "dual_bound": ev["dual_bound"],
            "t_cert_s": ev["t_cert_s"]}


def _decisions(spec: ScanSpec, rows: Iterable[Mapping], tau: float) -> tuple[dict, list[dict]]:
    """``(unit → decision, conflicts)`` of the rows of the cell. A decision is never revised: two rows of one
    instance that decide differently (trivial against non-trivial, a valid bound against a solution) are an error,
    whichever came first. A row that leaves the instance undecided, against one that decides it, is outvoted: the
    decision of the stage ``eligibility`` stands over that of a reference run, and the conflict is reported."""
    out: dict[Unit, dict] = {}
    conflicts: list[dict] = []
    for row in rows:
        if row.get("cell") != spec.cell:
            continue
        if row.get("error"):
            raise ValueError(f"cell {spec.cell}: a row carries an error ({row['error']}): fix and rerun the cell")
        if row.get("stage") not in ("eligibility", "reference"):
            continue
        unit = _unit_of(row, spec.points)
        new = _decision(row, unit, tau)
        old = out.get(unit)
        if old is None:
            out[unit] = new
            continue
        if old["class"] != new["class"]:
            if "undecided" not in (old["class"], new["class"]):
                raise ValueError(
                    f"cell {spec.cell}, {old['instance_id']}: two decisions, {old['class']!r} ({old['stage']}) and "
                    f"{new['class']!r} ({new['stage']}): a decision is never revised")
            keep = new if old["stage"] != "eligibility" and new["stage"] == "eligibility" else old
            drop = new if keep is old else old
            out[unit] = keep
            conflicts.append({"unit": _unit_json(unit), "kept": keep["class"], "ignored": drop["class"]})
        elif old["stage"] != "eligibility" and new["stage"] == "eligibility":
            out[unit] = new  # the screening is the record; a reference row only agrees with it
    return out, conflicts


def _quantiles(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "max": float(a.max())}


def estimate(decisions: Sequence[Mapping]) -> dict:
    """The estimate of A3 from the decisions of the scanned instances (module docstring)."""
    binding = [d for d in decisions if d["class"] != "nonbinding"]
    n = len(binding)
    trivial = [d for d in binding if d["class"] == "trivial"]
    k = len(trivial)
    low, high = wilson_interval(k, n, 0.95)
    out: dict[str, Any] = {
        "n_scanned": n, "n_trivial": k, "n_nontrivial": sum(d["class"] == "nontrivial" for d in binding),
        "n_undecided": sum(d["class"] == "undecided" for d in binding),
        "n_nonbinding": len(decisions) - n,
        "share_trivial": k / n if n else None, "ci95": [low, high],
    }
    decided = [d for d in binding if d["class"] != "undecided"]
    if decided:
        values = [d["gap_lo"] * d["s_start"] * BP for d in decided]
        g = _quantiles(values)
        g["n_exact"] = sum(d["gap_hi"] - d["gap_lo"] <= EXACT_TOL for d in decided)
        out["g_start_bp"] = g
    else:
        out["g_start_bp"] = None
    uppers = [d["gap_hi"] * d["s_start"] * BP for d in trivial if math.isfinite(d["gap_hi"])]
    out["g_start_trivial_upper_bp_max"] = max(uppers) if uppers else None
    certs = [d for d in trivial if d["t_cert_s"] is not None]
    if certs:
        levels = {d["certificate"] for d in certs}
        t = _quantiles([d["t_cert_s"] for d in certs])
        t["level"] = levels.pop() if len(levels) == 1 else "mixed"
        out["t_cert_trivial_s"] = t
    else:
        out["t_cert_trivial_s"] = None
    return out


def format_estimate(label: str, est: Mapping) -> str:
    """One line of the table of A3 for a cell or group: the share of trivial instances with its interval, the
    counts, the distribution of ``ĝ_start`` in bp CE and the time of the certificate on the trivial ones."""
    low, high = est["ci95"]
    share = "—" if est["share_trivial"] is None else f"{100 * est['share_trivial']:.0f} %"
    interval = "—" if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)) else (
        f"{100 * low:.0f}–{100 * high:.0f} %")
    line = (f"{label}: trivial {est['n_trivial']} of {est['n_scanned']} ({share}, Wilson {interval}), "
            f"non-trivial {est['n_nontrivial']}, undecided {est['n_undecided']}")
    if est.get("n_nonbinding"):
        line += f", non-binding {est['n_nonbinding']}"
    g = est.get("g_start_bp")
    if g:
        line += (f"; ĝ_start (bp CE) median {g['median']:.2f}, quartiles {g['q1']:.2f}–{g['q3']:.2f}, "
                 f"max {g['max']:.2f} ({g['n_exact']} of {g['n']} exact)")
    t = est.get("t_cert_trivial_s")
    if t:
        line += f"; t_cert on trivial ({t['level']}) median {t['median']:.1f} s, max {t['max']:.1f} s"
    return line


def cell_state(spec: ScanSpec, rows: Iterable[Mapping], *, tau: float = TAU) -> dict:
    """Where the scan of one cell stands, from the rows of the stage ``eligibility`` (and ``reference``): the
    screened prefix of the order, the selection, the counts of the protocol, the estimate of A3 and the next
    round. A JSON-safe dict (module docstring)."""
    decisions, conflicts = _decisions(spec, rows, tau)
    index = {unit: i for i, unit in enumerate(spec.order)}
    prefix = 0
    while prefix < len(spec.order) and spec.order[prefix] in decisions:
        prefix += 1
    beyond = [u for u in decisions if u in index and index[u] >= prefix]
    outside = [u for u in decisions if u not in index]
    in_prefix = [decisions[u] for u in spec.order[:prefix]]
    nontrivial_all = [d for d in in_prefix if d["class"] == "nontrivial"]
    found = len(nontrivial_all)
    if spec.target is None:
        protocol_end = prefix
    elif found >= spec.target:
        protocol_end = next(i for i, d in enumerate(in_prefix) if d["class"] == "nontrivial"
                            and sum(x["class"] == "nontrivial" for x in in_prefix[: i + 1]) == spec.target) + 1
    else:
        protocol_end = prefix
    protocol = in_prefix[:protocol_end]
    selected = [d for d in protocol if d["class"] == "nontrivial"]
    per_port: dict[str, int] = {}
    if spec.points:
        for d in selected:
            per_port[str(d["unit"][0])] = per_port.get(str(d["unit"][0]), 0) + 1
        per_port = {str(p): per_port.get(str(p), 0) for p in sorted({u[0] for u in spec.order})}
    n_ports = sum(v > 0 for v in per_port.values())
    enough = len(selected) >= spec.min_nontrivial and (not spec.points or n_ports >= C5_MIN_PORTS)
    done = prefix >= len(spec.order) or (spec.target is not None and found >= spec.target)
    counts = {c: sum(d["class"] == c for d in protocol) for c in (*CLASSES, "nonbinding")}
    state = {
        "family": spec.family, "n": spec.n, "config_size": spec.config_size, "points": spec.points,
        "target": spec.target, "min_nontrivial": spec.min_nontrivial, "cap": len(spec.order),
        "prefix": prefix, "found": found, "n_screened": len(decisions), "conflicts": conflicts,
        "beyond_gap": [_unit_json(u) for u in beyond if index[u] > prefix],
        "outside_order": [_unit_json(u) for u in outside],
        "protocol": {"n_scanned": counts["trivial"] + counts["nontrivial"] + counts["undecided"],
                     "n_trivial": counts["trivial"], "n_nontrivial": counts["nontrivial"],
                     "n_undecided": counts["undecided"], "n_nonbinding": counts["nonbinding"]},
        "protocol_end": protocol_end,
        "selected": [d["unit"] for d in selected],
        "needs_reference": [d["unit"] for d in selected if d["certificate"] != "full"],
        "enough": enough, "descriptive": done and not enough, "done": done,
        "next_block": [] if done else [_unit_json(u) for u in next_block(spec, prefix, found)],
        "instances": [{**d, "in_protocol": i < protocol_end} for i, d in enumerate(in_prefix)],
        "estimate": estimate(protocol),
    }
    if spec.points:
        state.update(per_port=per_port, n_ports=n_ports)
    return state


# --------------------------------------------------------------------------------------------
# The pooled estimate, borrowed configurations, the report
# --------------------------------------------------------------------------------------------


def in_summary(state: Mapping, cell: str, *, c4_admissible: bool | None = None) -> bool:
    """Whether the cell enters the pooled estimates of H2 and H2b (§12 A1): C1–C3 with enough non-trivial test
    instances, C4 as well if the λ rule of N7 admits it, C5 with enough points in enough ports (``enough`` of
    its state); D1 and the calibration cells never."""
    if cell in ("C1", "C2", "C3", "C5"):
        return bool(state["enough"])
    if cell == "C4":
        return bool(state["enough"]) and c4_admissible is True
    return False


def nearest_sizes(enough: Mapping[int, bool]) -> dict[int, int | None]:
    """For every size without enough non-trivial calibration instances, the nearest size with enough (ties go
    to the smaller); ``None`` if there is none. Sizes with enough are not in the result."""
    donors = sorted(n for n, ok in enough.items() if ok)
    out: dict[int, int | None] = {}
    for n, ok in sorted(enough.items()):
        if not ok:
            out[n] = min(donors, key=lambda d: (abs(d - n), d)) if donors else None
    return out


def _group_of(cell: str) -> str:
    return cell.split("_")[0]


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _today() -> str:
    return datetime.date.today().isoformat()


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f} %"


def build_report(
    rows: Iterable[Mapping], specs: Sequence[ScanSpec], *, role: str, tau: float = TAU,
    c4_admissible: bool | None = None, references: Iterable[Mapping] | None = None,
) -> dict:
    """The eligibility report of the cells ``specs`` from the rows so far (module docstring): per cell the
    state of its scan (:func:`cell_state`), per group of cells (D1) the pooled estimate, for the calibration
    role the sizes that borrow configurations, for the test role the cells of the pooled estimate. ``done``
    when every cell is finished. With ``references`` (rows with a certified ``f_star``) the decisions are
    checked against the scale of the optimum (:func:`reference_check`); instances that would flip are listed,
    never revised. The report of the last round is the input of the later stages."""
    if role not in MIN_NONTRIVIAL:
        raise ValueError(f"role must be one of {sorted(MIN_NONTRIVIAL)}, got {role!r}")
    rows = list(rows)
    cells = {s.cell: cell_state(s, rows, tau=tau) for s in specs}
    warnings: list[str] = []
    for label, st in cells.items():
        if st["beyond_gap"]:
            warnings.append(f"{label}: seeds {st['beyond_gap'][:5]} are screened beyond a gap in the order: they are "
                            "not part of the scan until the seeds before them are")
        for conflict in st["conflicts"]:
            warnings.append(f"{label}: unit {conflict['unit']} has rows that decide differently: kept "
                            f"{conflict['kept']!r}, ignored {conflict['ignored']!r} (a decision is never revised)")
        if st["outside_order"]:
            warnings.append(f"{label}: rows of {len(st['outside_order'])} units that are not in the scan order of "
                            "the config are ignored")
        n_scanned = st["protocol"]["n_scanned"]
        if n_scanned and st["protocol"]["n_undecided"] / n_scanned >= 0.1:
            warnings.append(f"{label}: {st['protocol']['n_undecided']} of {n_scanned} scanned instances are "
                            "undecided within the limit; the share of trivial instances is a lower bound")
    report: dict[str, Any] = {
        "kind": REPORT_KIND, "plan_version": "v2.4", "role": role, "tau": tau,
        "done": all(st["done"] for st in cells.values()), "cells": cells,
    }
    groups: dict[str, list[str]] = {}
    for label in cells:
        groups.setdefault(_group_of(label), []).append(label)
    report["groups"] = {}
    for name, labels in groups.items():
        if len(labels) < 2:
            continue
        parts = [cells[c]["estimate"] for c in labels]
        n = sum(p["n_scanned"] for p in parts)
        k = sum(p["n_trivial"] for p in parts)
        report["groups"][name] = {
            "cells": labels, "n_scanned": n, "n_trivial": k,
            "n_nontrivial": sum(p["n_nontrivial"] for p in parts),
            "n_undecided": sum(p["n_undecided"] for p in parts),
            "share_trivial": k / n if n else None, "ci95": list(wilson_interval(k, n, 0.95)),
        }
    if role == "calibration":
        sized = [s for s in specs if s.family == "factor" and s.config_size is not None]
        final = all(cells[s.cell]["done"] for s in sized)
        # "not enough" is a verdict only when every scan is over: a size that is short midway may still get there
        # (and a donor may still appear), so nothing is borrowed or warned about before
        report["config_from_final"] = final
        borrow = nearest_sizes({s.config_size: cells[s.cell]["enough"] for s in sized}) if final else {}
        report["config_from"] = {str(n): d for n, d in borrow.items()}
        for n, donor in borrow.items():
            if donor is None:
                warnings.append(f"size {n}: no size has enough non-trivial calibration instances to borrow "
                                "configurations from; nothing can be tuned")
        report["summary"] = None
    else:
        members = [s.cell for s in specs if in_summary(cells[s.cell], s.cell, c4_admissible=c4_admissible)]
        excluded = {}
        for s in specs:
            st = cells[s.cell]
            if s.cell in members:
                continue
            if _group_of(s.cell) == "D1":
                excluded[s.cell] = "descriptive by design"
            elif s.cell == "C4" and st["enough"] and c4_admissible is not True:
                excluded[s.cell] = "descriptive: the λ rule of N7 found no admissible λ"
            elif not st["enough"]:
                excluded[s.cell] = (f"descriptive: {len(st['selected'])} non-trivial instances, "
                                    f"fewer than the {s.min_nontrivial} the plan needs")
        statement = None
        if not members and any(s.cell in ("C1", "C2", "C3", "C4", "C5") for s in specs):
            statement = ("Нетривиальных инстансов недостаточно для сводной оценки; RQ2 отвечает долей "
                         "тривиальных")
        report["summary"] = {"cells": members, "excluded": excluded, "statement": statement}
    if references is not None:
        check = reference_check([r for r in rows if r.get("cell") in cells], references, tau=tau)
        report["reference_check"] = check
        for flip in check["flipped"]:
            warnings.append(f"{flip['instance_id']}: screened {flip['class']}, but its certified ĝ_start = "
                            f"{flip['g_start']:.3g} (scale of the optimum) is on the other side of τ: reported, "
                            "not revised")
    report["warnings"] = warnings
    report["inputs"] = []
    report["section12"] = _section12(report, specs)
    return _json_safe(report)


def _section12(report: Mapping, specs: Sequence[ScanSpec]) -> str:
    parts = []
    for s in specs:
        st = report["cells"][s.cell]
        est = st["estimate"]
        part = (f"{s.cell} — {len(st['selected'])} нетривиальных из {est['n_scanned']} просмотренных "
                f"(тривиальных {est['n_trivial']} = {_pct(est['share_trivial'])}, ДИ Уилсона "
                f"{_pct(est['ci95'][0])}–{_pct(est['ci95'][1])}; не определено {est['n_undecided']}")
        if est["n_nonbinding"]:
            part += f"; не связывающих {est['n_nonbinding']}"
        parts.append(part + ")")
    if report["role"] == "calibration":
        tail = "; наборы с менее чем 8 нетривиальными (после просмотра до потолка) описательные, их конфигурации — с ближайшего N"
        borrow = {n: d for n, d in (report.get("config_from") or {}).items()}
        if borrow:
            tail += " (" + ", ".join(f"N = {n} ← {d}" for n, d in borrow.items()) + ")"
        what = "калибровочные наборы"
    else:
        summary = report["summary"]
        tail = "; в сводную оценку H2/H2b входят: " + (", ".join(summary["cells"]) or "ни одна ячейка")
        what = "тестовые ячейки"
    return (f"{_today()} — [пилот E2] отбор нетривиальных инстансов (A1), {what}: " + "; ".join(parts) + tail
            + " — по записи A1 (правило ĝ_start > τ = 10⁻³, решается на этапе опоры) — RQ2, H1c — кто согласовал: —")


# --------------------------------------------------------------------------------------------
# Auditing against the references
# --------------------------------------------------------------------------------------------


def reference_check(screened: Iterable[Mapping], references: Iterable[Mapping], *, tau: float = TAU) -> dict:
    """Instances whose certified ``ĝ_start = (f_start − f*)/s*`` (the scale of the optimum, as in the plan) lies
    on the other side of τ than the screening decided, which used the scale of the start. Reported, never
    revised. ``screened`` are rows of the stage ``eligibility`` with a decision, ``references`` rows with a
    certified ``f_star``."""
    certified = {r["instance_id"]: r for r in references
                 if r.get("f_star") is not None and r.get("ref_type") == "certified" and r.get("scale_star")}
    flipped, checked = [], 0
    for row in screened:
        if row.get("stage") != "eligibility" or row.get("skipped") or row.get("eligibility") not in CLASSES:
            continue
        ref = certified.get(row.get("instance_id"))
        if ref is None or row["eligibility"] == "undecided":
            continue
        checked += 1
        g = (row["evidence"]["f_start"] - ref["f_star"]) / ref["scale_star"]
        side = "nontrivial" if g > tau else "trivial"
        if side != row["eligibility"]:
            flipped.append({"instance_id": row["instance_id"], "class": row["eligibility"], "g_start": g})
    return {"n_checked": checked, "flipped": flipped}


# --------------------------------------------------------------------------------------------
# From configs, and what the later stages read
# --------------------------------------------------------------------------------------------


def spec_from_group(group: Mapping, *, n_rule: Mapping[str, int] | None = None) -> ScanSpec:
    """The :class:`ScanSpec` of a group of ``params.instances``: a group with a ``scan`` section (seeds), with a
    fixed list of ``seeds`` (one round), or an ORlib group (its ports × λ points are a fixed set). ``n_rule`` maps
    cells to the counts of the n rule, for ``target_from``."""
    cell, family = group["cell"], group["family"]
    size = group.get("config_size")
    size = size if isinstance(size, int) and not isinstance(size, bool) else None
    if family == "orlib":
        lam = group.get("lam_indices", 25)
        indices = list(range(lam)) if isinstance(lam, int) else [int(i) for i in lam]
        order = tuple((int(p), i) for p in group["ports"] for i in indices)
        return ScanSpec(cell=cell, order=order, target=None, min_nontrivial=C5_MIN_POINTS, family=family,
                        config_size=size, points=True)
    if group.get("scan") is None and group.get("seeds") is not None:
        # a fixed list: every seed is screened in one round, the non-trivial ones are selected (a static pass)
        return ScanSpec(cell=cell, order=tuple(int(s) for s in group["seeds"]), target=None,
                        min_nontrivial=int(group.get("min_nontrivial", 1)), family=family, n=group.get("n"),
                        config_size=size)
    scan = group.get("scan")
    if not isinstance(scan, Mapping):
        raise ValueError(f"group {cell}: no 'scan' section (or a fixed list of 'seeds')")
    extra = set(scan) - _SCAN_KEYS
    if extra:
        raise ValueError(f"group {cell}: unknown keys in 'scan': {sorted(extra)}")
    if not scan.get("state"):
        raise ValueError(f"group {cell}: 'scan' needs 'state', the label of the eligibility report that holds "
                         "the progress of the scan")
    if scan.get("min_nontrivial") is None:
        raise ValueError(f"group {cell}: 'scan' needs 'min_nontrivial'")
    target = scan.get("target")
    if target is None and scan.get("target_from"):
        if n_rule is None or n_rule.get(cell) is None:
            raise ValueError(f"group {cell}: target_from {scan['target_from']!r} needs the counts of the n_rule "
                             f"report for the cell {cell!r}")
        target = int(n_rule[cell])
    if target is None:
        raise ValueError(f"group {cell}: 'scan' needs a 'target' or a 'target_from'")
    order = scan_order(first=scan.get("first"), head=scan.get("head") or (), then_from=scan.get("then_from"),
                       cap=scan["cap"])
    return ScanSpec(cell=cell, order=order, target=int(target), min_nontrivial=int(scan["min_nontrivial"]),
                    family=family, n=group.get("n"), config_size=size, block=scan.get("block"),
                    first_block=scan.get("first_block"))


def _checked(report: Mapping, cell: str) -> Mapping:
    if report.get("kind") != REPORT_KIND:
        raise ValueError(f"expected an {REPORT_KIND!r} report, its kind is {report.get('kind')!r}")
    if report.get("stub"):
        raise ValueError("the eligibility report is a stub (no round of the scan has been evaluated): run the "
                         "screening and `e2_pilot eligibility`")
    st = (report.get("cells") or {}).get(cell)
    if st is None:
        raise ValueError(f"the eligibility report has no cell {cell!r}")
    return st


def selected_units(report: Mapping, cell: str, *, field: str = "selected", require_enough: bool = False) -> list:
    """The units a later stage runs for ``cell``: ``selected`` (the non-trivial instances) or
    ``needs_reference`` (those without a full certificate). Refuses a stub, a cell that is not in the report and
    a scan that is not finished; a cell without enough instances gives none when ``require_enough`` is set (a
    calibration size that borrows its configurations is not tuned)."""
    st = _checked(report, cell)
    if field not in ("selected", "needs_reference"):
        raise ValueError(f"field must be 'selected' or 'needs_reference', got {field!r}")
    if not st["done"]:
        raise ValueError(f"the scan of {cell} is not finished ({st['prefix']} units screened, "
                         f"{st['found']} non-trivial): run another round")
    if require_enough and not st["enough"]:
        return []
    return list(st[field])


def progress(report: Mapping, cell: str) -> tuple[int, int]:
    """``(prefix, found)`` of a cell in the report of the rounds so far: the units screened from the start of
    the order and the non-trivial ones among them; ``(0, 0)`` for a stub or a cell not scanned yet."""
    if report.get("kind") != REPORT_KIND:
        raise ValueError(f"expected an {REPORT_KIND!r} report, its kind is {report.get('kind')!r}")
    st = (report.get("cells") or {}).get(cell)
    return (0, 0) if st is None else (int(st["prefix"]), int(st["found"]))
