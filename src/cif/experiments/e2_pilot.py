"""Pilot rules of E2: from calibration rows to the values that §12 of the plan records (ANALYSIS_PLAN §2
RQ2, §3, §5, §6 "Настройка").

The runs of :mod:`cif.experiments.e2_benchmark` write rows; this module turns the rows of the
calibration stages into **JSON reports**, one per rule, which the runs of the next stage read as
``inputs`` (hashed into the manifest) and which the orchestrator copies into §12 (``section12`` is a
draft of that line). Nothing here runs a solver. The pilot starts after the public stamp of the plan.

========================  ===========================================================  ===================
command                   rule                                                         report ``kind``
========================  ===========================================================  ===================
``c4-lambda``             smallest λ ∈ {5, 10, 20} with median ``s_cont`` ≥ 20 (N7)     ``e2-c4-lambda``
``eligibility``           rounds of the screening scan: which instances are            ``e2-eligibility``
                          non-trivial (§12 A1), the estimate of A3
``screen``                10 s ranking of ≤ 30 configurations per heuristic; the        ``e2-screen``
                          winner and the runner-up of every group go to the recheck
``tune``                  60 s recheck, 5 % and TTT rules, representatives of classes   ``e2-tuned``
``n-rule``                ``n = clip(⌈(2·1.96·SD/ln 1.5)²⌉, 20, 40)`` per cell           ``e2-n-rule``
``ttt-instances``         the 3 calibration instances of median difficulty (minor 10)   ``e2-ttt-instances``
========================  ===========================================================  ===================

**Measures of a row** (§5). The reference of an instance is the best value known over all rows of the
instance, the common start and the reference run included, with the scale ``s`` of the portfolio that
attains it; a certified optimum from a reference run replaces it (type ``certified``) unless a method is
more than ``1e-6·s`` below it, which is flagged (``inconsistent``). ``ĝ = (f − f_ref)/s_ref``. The
primary measure of a row is **PI⁺**, the primal integral after the common start, ``PI⁺(T) =
PI(T) − t₀/T`` (:func:`row_pi`, §12 A2): the interval ``[0, t₀)`` has ``γ = 1`` for every method alike and
is left out. ``R``, ``R_b``, the tuning criterion at 10 s, the n rule and the choice of the TTT instances
all use it. On an instance where the start is optimal PI⁺ is 0 for every method, a ratio 0/0, and PI is
``t₀/T`` for every method (not 0), so ``R ≡ 1`` up to the noise of the clock: such instances carry no
information about the methods and the eligibility rule (§12 A1, :mod:`cif.experiments.e2_eligibility`)
keeps them out of every comparison, calibration sets included. The named variants of :func:`row_pi` are
the sensitivity analyses: ``"full"`` is PI over ``[0, T]`` (the definition before A2), ``"first_improvement"``
is "``γ = 1`` until the first improvement of the common start" (computed from the same rows) and ``"own"`` is
"``γ = 1`` until the method's own first solution" (cold rows only, A4). The final gap is that of the incumbent
at the budget (:func:`row_gap_at_budget`); the time-to-target is end-to-end, censored at the budget
(:func:`row_ttt`).

**Tuning protocol** (§6; the criterion is PI⁺ at 10 s, for H1c the final gap at 10 s):

1. a configuration's score is the median over instances of the median over seeds;
2. configurations are ordered by the score; those within 5 % of the best of their tie group (anchored at
   the best, not chained) are ordered by the Kaplan–Meier median of their time-to-target (τ = 1e-3, all
   runs pooled, censored at the budget); what the TTT cannot decide stays in score order, then id order;
3. the winner and the runner-up are rerun at 60 s on the calibration set (other solver seeds); if the winner
   is worse than the runner-up by more than 5 % of the runner-up's score (plus 1e-9 absolute, so that zeros
   compare) under the same criterion at 60 s, the runner-up is chosen. For H1c the criterion at 60 s is the
   final gap at 60 s: the plan does not say what a recheck of a 10 s criterion compares;
4. the representative of the QUBO class is the best of the three tuned heuristics by their 10 s score under
   steps 1–2; that of the exact class is the chosen exact variant; that of the direct search is the chosen LS.
   The class choice is not rechecked at 60 s (the plan applies the recheck to the winner of a tuning);
5. the ranking at 10 s uses the references known at the screen (screen rows and reference runs), the
   comparison at 60 s those of all rows, so that the recheck plan is the one the screen produced.

**n per cell.** SD is the larger of the sample standard deviations (ddof = 1) of ``log R`` and ``log R_b``
over the calibration instances of the cell, with ``R = PI⁺(QUBO)/PI⁺(exact)``, ``R_b = PI⁺(QUBO)/PI⁺(LS)`` of
the 60 s runs of the three representatives (median over seeds). There is no floor: a PI⁺ of 0 means that
the start is optimal and the instance is not non-trivial (§12 A1), the ratio would be 0/0, and the report
refuses it with the instances named. On the non-trivial calibration sets PI⁺ is positive for every method.
A cell with fewer than 8 non-trivial calibration instances is descriptive (§12 A1) and has no SD: its n is
the minimum, 20.

**TTT instances.** Among the (non-trivial) calibration instances of the cell, the ``k`` = 3 whose 60 s PI⁺ of
the exact representative is closest to the median of the cell (ties: smaller seed).
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cif.experiments import e2_benchmark, e2_eligibility, registry
from cif.experiments.run import effective_config, load_config, read_rows
from cif.instances.boot import (
    BOOT_CALIBRATION_SEEDS,
    BOOT_LAMBDAS,
    BOOT_MIN_MEDIAN_SCONT,
    calibrate_boot_lambda,
)
from cif.metrics.anytime import (
    TAU_MAIN,
    TimeToTarget,
    km_median,
    primal_integral,
    primal_integral_after_start,
    time_to_target,
)

__all__ = [
    "PI_VARIANTS",
    "ConfigScore",
    "RefValue",
    "c4_lambda_report",
    "config_scores",
    "load_rows",
    "main",
    "n_per_cell",
    "n_rule_report",
    "order_configs",
    "recheck_decision",
    "reference_table",
    "row_gap_at_budget",
    "row_pi",
    "row_ttt",
    "screen_report",
    "ttt_instances_report",
    "tuned_report",
]

TIE_REL = 0.05
ABS_TOL = 1e-9
CERT_TOL = 1e-6
#: Variants of the primal integral of a row (:func:`row_pi`): the primary PI⁺ and the sensitivity analyses.
PI_VARIANTS = ("plus", "full", "first_improvement", "own")
#: Groups of the tuning (the exact class is one group) and the criteria each is ranked by.
GROUP_CRITERIA = {
    "exact": ("pi",), "sa": ("pi", "gap"), "tabu": ("pi", "gap"), "swap_sa": ("pi",), "ls": ("pi",),
}
_GROUP_OF = {"scip": "exact", "oa": "exact", "sa": "sa", "tabu": "tabu", "swap_sa": "swap_sa", "ls": "ls"}
QUBO_GROUPS = ("sa", "tabu", "swap_sa")
#: Calibration cells as the stage configs of ``configs/E2`` label them: report cell → row label, size.
DEFAULT_CELLS = {
    "C1": {"label": "T100", "config_size": 100},
    "C2": {"label": "T200", "config_size": 200},
    "C3": {"label": "T400", "config_size": 400},
    "C4": {"label": "CB200", "config_size": 200},
}
#: A share of calibration instances whose start is within τ of the best known above which the report warns.
DEGENERATE_SHARE = 0.5


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------


def load_rows(paths: Iterable[Path | str]) -> tuple[list[dict], list[dict]]:
    """Rows of result files or registered run directories, merged by ``cell_key``: the last row without
    an error wins (as in :mod:`cif.experiments.ingest`). Returns ``(rows, error rows)`` where the error
    rows are those of cells with no successful row."""
    rows: dict[str, dict] = {}
    errors: dict[str, dict] = {}
    for path in paths:
        p = Path(path)
        for row in read_rows(p / "results.jsonl" if p.is_dir() else p):
            key = row["cell_key"]
            if row.get("error"):
                if key not in rows:
                    errors[key] = row
            else:
                rows[key] = row
                errors.pop(key, None)
    return list(rows.values()), list(errors.values())


def reference_rows(paths: Iterable[Path | str]) -> list[dict]:
    """Rows of reference and screening runs for :func:`reference_table`: within a run the last success of a cell
    wins, as in :func:`load_rows`, but the runs are not merged with each other, so that a certificate of any round
    counts (audit G2, C-2: round 2 of the test screening re-screened the OR-Library points that round 1 had
    certified, and merging the rounds by the last row of a cell lost one certificate). Error rows are left out."""
    out: list[dict] = []
    for path in paths:
        rows, _errors = load_rows([path])
        out.extend(rows)
    return out


# --------------------------------------------------------------------------------------------
# References and the measures of a row
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RefValue:
    """The reference of an instance: value, scale of its portfolio, type, where it came from, the number
    of rows of the instance and whether a method beat a certified optimum."""

    instance_id: str
    f_ref: float
    s_ref: float
    ref_type: str
    source: str
    n_rows: int
    inconsistent: bool


def reference_table(rows: Iterable[Mapping], *, cert_tol: float = CERT_TOL) -> dict[str, RefValue]:
    """The reference of every instance in ``rows`` (module docstring)."""
    candidates: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    certified: dict[str, tuple[float, float]] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        iid = row.get("instance_id")
        if iid is None:
            continue
        if row.get("stage") == "reference" or (row.get("stage") == "eligibility" and row.get("f_star") is not None):
            # a certificate: the reference stage, or a screening run that went on to the optimum (§12 A1)
            f, s = row.get("f_star"), row.get("scale_star")
            if f is not None and s:
                candidates[iid].append((float(f), float(s), "reference"))
                if row.get("ref_type") == "certified" and (iid not in certified or f < certified[iid][0]):
                    certified[iid] = (float(f), float(s))
            continue
        counts[iid] += 1
        if row.get("f_final") is not None and row.get("scale_final"):
            candidates[iid].append((float(row["f_final"]), float(row["scale_final"]), row["config_id"]))
        if row.get("start_f") is not None and row.get("start_scale"):
            candidates[iid].append((float(row["start_f"]), float(row["start_scale"]), "start"))
    out = {}
    for iid, found in candidates.items():
        best_f, best_s, label = min(found, key=lambda c: c[0])
        if iid in certified:
            f_c, s_c = certified[iid]
            if best_f < f_c - cert_tol * s_c:
                out[iid] = RefValue(iid, best_f, best_s, "certified", label, counts[iid], True)
            else:
                out[iid] = RefValue(iid, f_c, s_c, "certified", "reference", counts[iid], False)
        else:
            out[iid] = RefValue(iid, best_f, best_s, "best_known", label, counts[iid], False)
    return out


def _start(row: Mapping) -> tuple[float, float] | None:
    f = row.get("start_f")
    if f is None or not math.isfinite(f):
        return None
    return float(row["start_time_s"]), float(f)


def start_gap(row: Mapping, ref: RefValue) -> float:
    """``ĝ`` of the common start of the row against the reference (``inf`` without a start)."""
    start = _start(row)
    return math.inf if start is None else (start[1] - ref.f_ref) / ref.s_ref


def row_pi(row: Mapping, ref: RefValue, *, budget_s: float | None = None, variant: str = "plus") -> float:
    """Primal integral of the row over ``budget_s`` (default: its own); ``variant`` is one of
    :data:`PI_VARIANTS`.

    * ``"plus"`` (default, the primary measure, §12 A2): PI⁺ = PI − t₀/T, the integral over ``[t₀, T]``
      with the common start as the first incumbent; ``t₀`` is the row's ``start_time_s`` (kept even if the
      construction failed, then ``γ = 1`` runs until the first own solution);
    * ``"full"``: PI over ``[0, T]`` with the common start (the definition before A2; the common term
      ``t₀/T`` included);
    * ``"first_improvement"``: sensitivity, "γ = 1 until the first improvement of the common start" (A4:
      the trajectory of a warm row lists only the improvements over the start);
    * ``"own"``: sensitivity, "γ = 1 until the method's own first solution" (A4): the full trajectory of a
      cold row, which has no common start.

    ``"plus"``, ``"full"`` and ``"first_improvement"`` need a warm row, ``"own"`` a cold one.
    """
    if variant not in PI_VARIANTS:
        raise ValueError(f"variant must be one of {PI_VARIANTS}, got {variant!r}")
    cold = bool(row.get("cold"))
    if cold != (variant == "own"):
        raise ValueError(
            f"variant {variant!r} needs a {'cold' if variant == 'own' else 'warm'} row (a cold row has no common "
            f"start: only 'own' is defined on it), the row is {'cold' if cold else 'warm'}")
    budget = float(row["budget_s"] if budget_s is None else budget_s)
    kwargs = {"f_ref": ref.f_ref, "s_ref": ref.s_ref, "budget_s": budget}
    if variant == "plus":
        t0, f0 = row.get("start_time_s"), row.get("start_f")
        if t0 is None:
            raise ValueError("a warm row without start_time_s cannot give PI+")
        start = (float(t0), float(f0) if f0 is not None and math.isfinite(f0) else math.inf)
        return primal_integral_after_start(row["trajectory"], start=start, **kwargs)
    if variant == "full":
        return primal_integral(row["trajectory"], start=_start(row), **kwargs)
    return primal_integral(row["trajectory"], **kwargs)


def row_gap_at_budget(row: Mapping, ref: RefValue) -> float:
    """``ĝ`` of the incumbent at the budget (``inf`` if there is none by then)."""
    f = row.get("f_at_budget")
    return math.inf if f is None else (float(f) - ref.f_ref) / ref.s_ref


def row_ttt(row: Mapping, ref: RefValue, tau: float, *, budget_s: float | None = None) -> TimeToTarget:
    """End-to-end time of the row to ``ĝ ≤ τ``, censored at the budget."""
    budget = float(row["budget_s"] if budget_s is None else budget_s)
    return time_to_target(row["trajectory"], f_ref=ref.f_ref, s_ref=ref.s_ref, tau=tau, budget_s=budget,
                          start=_start(row))


# --------------------------------------------------------------------------------------------
# Scores and ranking
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigScore:
    """A configuration under a criterion: the score (median over instances of the median over seeds),
    the per-instance values, the Kaplan–Meier median of the pooled TTT (``inf`` if more than half of the
    runs are censored) and the number of censored runs."""

    config_id: str
    name: str
    spec: dict
    n_instances: int
    n_runs: int
    criterion: float
    per_instance: dict
    ttt_median: float
    ttt_censored: int


def config_scores(
    rows: Sequence[Mapping], refs: Mapping[str, RefValue], *, budget_s: float, criterion: str,
    tau: float = TAU_MAIN,
) -> list[ConfigScore]:
    """The score of every configuration (``config_id``) in ``rows``, which must all have ``budget_s``.

    ``criterion`` is ``"pi"`` (primal integral) or ``"gap"`` (final gap at the budget); the time-to-target
    is computed whichever the criterion. Sorted by id."""
    if criterion not in ("pi", "gap"):
        raise ValueError(f"criterion must be 'pi' or 'gap', got {criterion!r}")
    by_config: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        if float(row["budget_s"]) != float(budget_s):
            raise ValueError(f"row of budget {row['budget_s']} s among rows scored at budget_s = {budget_s}")
        by_config[row["config_id"]].append(row)
    scores = []
    for config_id, group in sorted(by_config.items()):
        specs = {json.dumps(r["method"], sort_keys=True) for r in group}
        if len(specs) != 1:
            raise ValueError(f"configuration id {config_id!r} stands for different specs")
        by_instance: dict[str, list[float]] = defaultdict(list)
        times, observed = [], []
        for row in group:
            ref = refs.get(row["instance_id"])
            if ref is None:
                raise ValueError(f"no reference for instance {row['instance_id']!r}")
            by_instance[row["instance_id"]].append(
                row_pi(row, ref, budget_s=budget_s) if criterion == "pi" else row_gap_at_budget(row, ref))
            ttt = row_ttt(row, ref, tau, budget_s=budget_s)
            times.append(ttt.time_s)
            observed.append(ttt.reached)
        per_instance = {iid: float(np.median(v)) for iid, v in sorted(by_instance.items())}
        spec = dict(group[0]["method"])
        scores.append(ConfigScore(
            config_id, spec["name"], spec, len(per_instance), len(group),
            float(np.median(list(per_instance.values()))), per_instance,
            km_median(times, observed), int(len(observed) - sum(observed))))
    return scores


def order_configs(scores: Sequence[ConfigScore], *, tie_rel: float = TIE_REL,
                  abs_tol: float = ABS_TOL) -> list[ConfigScore]:
    """Configurations in the order of the tuning protocol (module docstring, step 2): by score; those
    within ``tie_rel`` of the best of their tie group by the smaller TTT median, then score, then id."""
    pending = sorted(scores, key=lambda s: (s.criterion, s.config_id))
    ordered: list[ConfigScore] = []
    while pending:
        best = pending[0].criterion
        group = [s for s in pending if s.criterion <= best * (1.0 + tie_rel) + abs_tol]
        group.sort(key=lambda s: (s.ttt_median, s.criterion, s.config_id))
        ordered += group
        taken = {s.config_id for s in group}
        pending = [s for s in pending if s.config_id not in taken]
    return ordered


def _order_by_id(order: Sequence[ConfigScore], abs_tol: float) -> bool:
    """The first two of ``order`` tie on the score (to ``abs_tol``) and on the TTT median: only their ids
    put one before the other, so who is the winner is arbitrary."""
    if len(order) < 2:
        return False
    a, b = order[0], order[1]
    same_score = a.criterion == b.criterion or abs(a.criterion - b.criterion) <= abs_tol
    return bool(same_score and a.ttt_median == b.ttt_median)


def recheck_decision(
    order: Sequence[ConfigScore], criterion_long: Mapping[str, float], *, worse_rel: float = TIE_REL,
    abs_tol: float = ABS_TOL,
) -> dict:
    """Winner or runner-up (step 3 of the protocol): the runner-up is chosen if, at the long budget, the
    winner is worse than it by more than ``worse_rel`` of its score (plus ``abs_tol``)."""
    winner = order[0]
    second = order[1] if len(order) > 1 else None
    needed = [winner.config_id, *([second.config_id] if second else [])]
    missing = [c for c in needed if c not in criterion_long]
    if missing:
        raise ValueError(f"no recheck result at the long budget for {missing}")
    chosen, reason = winner.config_id, "winner"
    if second is not None:
        w, s = criterion_long[winner.config_id], criterion_long[second.config_id]
        if w > s * (1.0 + worse_rel) + abs_tol:
            chosen = second.config_id
            reason = (f"runner-up: the winner is worse at the long budget "
                      f"({w:.6g} against {s:.6g}, more than {worse_rel:.0%})")
    return {"winner": winner.config_id, "second": second.config_id if second else None, "chosen": chosen,
            "reason": reason, "order_by_id": _order_by_id(order, abs_tol),
            "criterion_long": {c: float(criterion_long[c]) for c in needed}}


# --------------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------------


def _today() -> str:
    return datetime.date.today().isoformat()


def _inf_none(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _entry(score: ConfigScore) -> dict:
    return {"id": score.config_id, "criterion": score.criterion, "ttt_median": _inf_none(score.ttt_median),
            "ttt_censored": score.ttt_censored, "n_instances": score.n_instances, "n_runs": score.n_runs,
            "spec": score.spec}


def _check_no_errors(rows: Iterable[Mapping]) -> None:
    bad = [r for r in rows if r.get("error")]
    if bad:
        raise ValueError(f"{len(bad)} rows carry an error (fix and rerun the cells), e.g. {bad[0]['error']}")


def _usable(rows: Iterable[Mapping]) -> list[Mapping]:
    """The rows of runs: without the rows of instances a stage skipped (the constraint did not bind)."""
    return [r for r in rows if not r.get("skipped")]


def _size_rows(rows: Sequence[Mapping]) -> dict[int, list[Mapping]]:
    out: dict[int, list[Mapping]] = defaultdict(list)
    for row in rows:
        out[int(row["config_size"])].append(row)
    return dict(sorted(out.items()))


def _group_rows(rows: Sequence[Mapping], group: str) -> list[Mapping]:
    return [r for r in rows if _GROUP_OF[r["method"]["name"]] == group]


def _start_gap_summary(rows: Sequence[Mapping], refs: Mapping[str, RefValue], tau: float) -> dict:
    """How much room the common start leaves: the gap of the start of each instance (the smallest over
    its rows: it is one portfolio), the shares within ``tau`` of the best known and equal to it."""
    best: dict[str, float] = {}
    for row in rows:
        ref = refs.get(row["instance_id"])
        if ref is not None:
            best[row["instance_id"]] = min(best.get(row["instance_id"], math.inf), start_gap(row, ref))
    gaps = np.array(list(best.values()), dtype=float)
    return {
        "n_instances": int(gaps.size), "tau": tau,
        "median": float(np.median(gaps)) if gaps.size else None,
        "share_within_tau": float(np.mean(gaps <= tau)) if gaps.size else None,
        "share_optimal": float(np.mean(gaps <= ABS_TOL)) if gaps.size else None,
    }


def screen_report(
    rows: Sequence[Mapping], *, budget_s: float, tau: float = TAU_MAIN, tie_rel: float = TIE_REL,
    abs_tol: float = ABS_TOL,
) -> dict:
    """The 10 s ranking of the screen, per size, group and criterion, and the plan of the recheck
    (``sizes[N]["recheck"]``: the winner and the runner-up of every group, each once, with the reasons)."""
    _check_no_errors(rows)
    rows = _usable(rows)
    runs = [r for r in rows if r.get("stage") == "screen"]
    if not runs:
        raise ValueError("no rows of the stage 'screen'")
    refs = reference_table(rows)
    sizes: dict[str, dict] = {}
    warnings: list[str] = []
    for size, size_rows in _size_rows(runs).items():
        groups: dict[str, dict] = {}
        plan: dict[str, dict] = {}
        for group, criteria in GROUP_CRITERIA.items():
            rows_g = _group_rows(size_rows, group)
            if not rows_g:
                continue
            for criterion in criteria:
                order = order_configs(config_scores(rows_g, refs, budget_s=budget_s, criterion=criterion, tau=tau),
                                      tie_rel=tie_rel, abs_tol=abs_tol)
                key = f"{group}/{criterion}"
                best = min(s.criterion for s in order)
                n_tied = sum(1 for s in order if s.criterion <= best * (1.0 + tie_rel) + abs_tol)
                by_id = _order_by_id(order, abs_tol)
                groups[key] = {
                    "ranking": [_entry(s) for s in order], "winner": order[0].config_id,
                    "second": order[1].config_id if len(order) > 1 else None, "n_configs": len(order),
                    "n_in_tie_band": n_tied, "order_by_id": by_id,
                }
                if by_id:
                    warnings.append(
                        f"size {size}, {key}: {order[0].config_id} and {order[1].config_id} tie exactly on the "
                        f"score and the TTT median ({n_tied} of {len(order)} configurations lie in the 5 % tie "
                        "band): they are ordered by id, the choice of the winner is arbitrary")
                for place, score in zip(("winner", "second"), order[:2], strict=False):
                    item = plan.setdefault(score.config_id, {"spec": score.spec, "why": []})
                    item["why"].append(f"{key}:{place}")
        summary = _start_gap_summary(size_rows, refs, tau)
        if summary["share_within_tau"] is not None and summary["share_within_tau"] >= DEGENERATE_SHARE:
            warnings.append(
                f"size {size}: the common start is within tau = {tau:g} of the best known on "
                f"{summary['share_within_tau']:.0%} of the calibration instances "
                f"({summary['share_optimal']:.0%} optimal): the primal integral and the time-to-target "
                "hardly separate the configurations (a set selected as non-trivial by the screening, §12 A1, "
                "has none such: was it screened?)")
        sizes[str(size)] = {
            "groups": groups,
            "recheck": [{"spec": item["spec"], "why": sorted(item["why"])} for item in plan.values()],
            "start_gap": summary,
            "n_inconsistent_references": sum(
                1 for r in refs.values() if r.inconsistent and any(
                    row["instance_id"] == r.instance_id for row in size_rows)),
        }
    return {
        "kind": "e2-screen", "plan_version": "v2.4",
        "settings": {"budget_s": budget_s, "tau": tau, "tie_rel": tie_rel, "abs_tol": abs_tol,
                     "criteria": {"pi": "median over instances of the median over seeds of PI+(budget) = PI - t0/T",
                                  "gap": "same for the gap of the incumbent at the budget"}},
        "sizes": sizes, "warnings": warnings, "inputs": [],
        "section12": (f"{_today()} — [пилот E2] ранжирование настройки при {budget_s:g} с по размерам "
                      f"{', '.join(sizes)}; на перепроверку 60 с идут победитель и второе место каждой группы — "
                      "по правилу §6 (N6) — RQ2, H1c — кто согласовал: —"),
    }


GROUP_ROLE = {"exact": "exact_rep", "sa": "sa", "tabu": "tabu", "swap_sa": "swap_sa", "ls": "ls"}


def tuned_report(
    screen_rows: Sequence[Mapping], recheck_rows: Sequence[Mapping], *, budget_s: float, budget_long_s: float,
    tau: float = TAU_MAIN, tie_rel: float = TIE_REL, abs_tol: float = ABS_TOL,
    borrow: Mapping[int, int | None] | None = None,
) -> dict:
    """The tuned configurations: the decisions of the protocol per size, group and criterion, the roles
    (:data:`cif.experiments.e2_benchmark.ROLES`) the benchmark stages read, and the screen they came from.

    ``borrow`` maps a size without enough non-trivial calibration instances to the size whose configurations
    it takes (§12 A1: the nearest N; :func:`cif.experiments.e2_eligibility.nearest_sizes`). Such a size has no
    screening rows; its entry is a copy of the donor's with ``borrowed_from``."""
    plan = screen_report(screen_rows, budget_s=budget_s, tau=tau, tie_rel=tie_rel, abs_tol=abs_tol)
    _check_no_errors(recheck_rows)
    screen_rows, recheck_rows = _usable(screen_rows), _usable(recheck_rows)
    long_rows = [r for r in recheck_rows if r.get("stage") == "recheck"]
    refs_screen = reference_table(screen_rows)
    refs_all = reference_table([*screen_rows, *recheck_rows])
    screen_runs = [r for r in screen_rows if r.get("stage") == "screen"]
    sizes: dict[str, dict] = {}
    for size, size_rows in _size_rows(screen_runs).items():
        long_size = [r for r in long_rows if int(r["config_size"]) == size]
        decisions: dict[str, dict] = {}
        chosen: dict[str, dict] = {}  # "group/criterion" -> ConfigScore of the screen
        for group, criteria in GROUP_CRITERIA.items():
            rows_g = _group_rows(size_rows, group)
            if not rows_g:
                raise ValueError(f"size {size}: no screening rows of the group {group!r}")
            for criterion in criteria:
                key = f"{group}/{criterion}"
                scores = {s.config_id: s for s in config_scores(
                    rows_g, refs_screen, budget_s=budget_s, criterion=criterion, tau=tau)}
                order = order_configs(list(scores.values()), tie_rel=tie_rel, abs_tol=abs_tol)
                planned = [s.config_id for s in order[:2]]
                rows_long = [r for r in _group_rows(long_size, group) if r["config_id"] in planned]
                missing = [c for c in planned if not any(r["config_id"] == c for r in rows_long)]
                if missing:
                    raise ValueError(f"size {size}, {key}: no recheck rows at {budget_long_s:g} s for {missing}")
                long_scores = config_scores(rows_long, refs_all, budget_s=budget_long_s, criterion=criterion,
                                            tau=tau)
                decision = recheck_decision(order, {s.config_id: s.criterion for s in long_scores},
                                            worse_rel=tie_rel, abs_tol=abs_tol)
                decision["criterion_screen"] = {c: scores[c].criterion for c in planned}
                decisions[key] = decision
                chosen[key] = scores[decision["chosen"]]
        roles = {GROUP_ROLE[g]: chosen[f"{g}/pi"].spec for g in GROUP_CRITERIA}
        qubo = order_configs([chosen[f"{g}/pi"] for g in QUBO_GROUPS], tie_rel=tie_rel, abs_tol=abs_tol)[0]
        roles["qubo_rep"] = qubo.spec
        roles["h1c_sa"] = chosen["sa/gap"].spec
        roles["h1c_tabu"] = chosen["tabu/gap"].spec
        sizes[str(size)] = {
            "roles": {role: roles[role] for role in e2_benchmark.ROLES},
            "decisions": decisions,
            "representatives": {"exact": roles["exact_rep"]["id"], "qubo": qubo.config_id, "ls": roles["ls"]["id"]},
            "screen": plan["sizes"][str(size)],
            "inconsistent_references": sum(
                1 for r in refs_all.values() if r.inconsistent and any(
                    row["instance_id"] == r.instance_id for row in size_rows)),
        }
    warnings = list(plan["warnings"])
    for size, donor in (borrow or {}).items():
        if donor is None:
            raise ValueError(f"size {size}: nothing to borrow, no size has enough non-trivial calibration instances")
        if str(size) in sizes:
            raise ValueError(f"size {size} has screening rows but is declared to borrow its configurations")
        if str(donor) not in sizes:
            raise ValueError(f"size {size} borrows from size {donor}, which has no screening rows: the donor must be "
                             "tuned")
        sizes[str(size)] = {**copy.deepcopy(sizes[str(donor)]), "borrowed_from": int(donor)}
        warnings.append(f"size {size} borrows the configurations of size {donor} (too few non-trivial calibration "
                        "instances, §12 A1)")
    sizes = dict(sorted(sizes.items(), key=lambda item: int(item[0])))
    lines = [f"N = {n}: " + ", ".join(f"{role} = {s['roles'][role]['id']}" for role in e2_benchmark.ROLES)
             + (f" (конфигурации N = {s['borrowed_from']})" if "borrowed_from" in s else "")
             for n, s in sizes.items()]
    return {
        "kind": e2_benchmark.TUNED_KIND, "plan_version": "v2.4",
        "settings": {"budget_s": budget_s, "budget_long_s": budget_long_s, "tau": tau, "tie_rel": tie_rel,
                     "abs_tol": abs_tol},
        "sizes": sizes, "warnings": warnings, "inputs": [],
        "section12": (f"{_today()} — [пилот E2] настроенные конфигурации: {'; '.join(lines)} — по правилу §6 "
                      "(N6, A2): критерий PI⁺ за 10 с (H1c: итоговый зазор), перепроверка 60 с, ничья 5 % по медиане "
                      "TTT — RQ2, H1c — кто согласовал: —"),
    }


def n_per_cell(sd: float, *, n_min: int = 20, n_max: int = 40) -> int:
    """``max(20, ⌈(2·1.96·SD/ln 1.5)²⌉)``, at most 40 (§3)."""
    if not (isinstance(sd, int | float) and math.isfinite(sd) and sd >= 0.0):
        raise ValueError(f"SD must be a finite non-negative number, got {sd!r}")
    n = math.ceil((2.0 * 1.96 * sd / math.log(1.5)) ** 2)
    return int(min(n_max, max(n_min, n)))


def _median_pi_by_instance(rows: Sequence[Mapping], refs: Mapping[str, RefValue], budget_s: float) -> dict:
    by_instance: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance_id"]].append(row_pi(row, refs[row["instance_id"]], budget_s=budget_s))
    return {iid: float(np.median(v)) for iid, v in by_instance.items()}


def _representatives(tuned: Mapping, size: int) -> dict[str, str]:
    entry = (tuned.get("sizes") or {}).get(str(size))
    if entry is None:
        raise ValueError(f"the tuned report has no configurations for size {size}")
    roles = entry["roles"]
    return {"exact": roles["exact_rep"]["id"], "qubo": roles["qubo_rep"]["id"], "ls": roles["ls"]["id"]}


def n_rule_report(
    rows: Sequence[Mapping], tuned: Mapping, *, cells: Mapping[str, Mapping] | None = None,
    budget_s: float = 60.0, descriptive: Mapping[str, str] | None = None,
) -> dict:
    """The n-per-cell rule from the 60 s rows of the three representatives on the calibration instances
    of each cell (module docstring), on PI⁺. ``cells`` maps a cell (C1…) to the ``label`` of its
    calibration rows and the ``config_size`` whose representatives are used. ``descriptive`` names the cells
    (with the reason) that have too few non-trivial calibration instances (§12 A1): they have no SD, their n
    is the minimum, 20."""
    _check_no_errors(rows)
    rows = _usable(rows)
    cells = dict(cells or DEFAULT_CELLS)
    refs = reference_table(rows)
    out: dict[str, dict] = {}
    for cell, spec in cells.items():
        reps = _representatives(tuned, spec["config_size"])
        if cell in (descriptive or {}):
            sel = [r for r in rows if r.get("cell") == spec["label"] and r["config_id"] in reps.values()
                   and int(r["config_size"]) == spec["config_size"] and r["budget_s"] == budget_s
                   and r.get("stage") != "reference"]
            out[cell] = {
                "label": spec["label"], "config_size": spec["config_size"], "representatives": reps,
                "n_calibration": len({r["instance_id"] for r in sel}), "sd_log_r": None, "sd_log_r_b": None,
                "n_instances": n_per_cell(0.0), "descriptive": descriptive[cell], "pi": {},
            }
            continue
        pis: dict[str, dict] = {}
        for role, config_id in reps.items():
            sel = [r for r in rows if r.get("cell") == spec["label"] and r["config_id"] == config_id
                   and int(r["config_size"]) == spec["config_size"] and r["budget_s"] == budget_s
                   and r.get("stage") != "reference"]
            pis[role] = _median_pi_by_instance(sel, refs, budget_s)
        common = sorted(set.intersection(*(set(v) for v in pis.values())))
        vanishing = sorted(f"{role}:{i}" for role, per in pis.items() for i in common if not per[i] > 0.0)
        if vanishing:
            raise ValueError(
                f"cell {cell}: PI+ = 0 for {vanishing[:6]}{' …' if len(vanishing) > 6 else ''}: the common start is "
                "optimal there, so the instance is not non-trivial (ANALYSIS_PLAN §12 A1), R is 0/0 and no floor "
                "is applied; screen the calibration set with the eligibility stage")
        if len(common) < 2:
            raise ValueError(
                f"cell {cell}: {len(common)} calibration instances with all three representatives "
                f"(rows labelled {spec['label']!r}, size {spec['config_size']}): at least 2 are needed")
        q = np.array([pis["qubo"][i] for i in common])
        e = np.array([pis["exact"][i] for i in common])
        ls = np.array([pis["ls"][i] for i in common])
        sd_r, sd_rb = float(np.std(np.log(q / e), ddof=1)), float(np.std(np.log(q / ls), ddof=1))
        out[cell] = {
            "label": spec["label"], "config_size": spec["config_size"], "representatives": reps,
            "n_calibration": len(common), "sd_log_r": sd_r, "sd_log_r_b": sd_rb,
            "n_instances": n_per_cell(max(sd_r, sd_rb)),
            "pi": {role: {i: pis[role][i] for i in common} for role in pis},
        }
    n_instances = {cell: v["n_instances"] for cell, v in out.items()}
    return {
        "kind": "e2-n-rule", "plan_version": "v2.4", "formula": "max(20, ceil((2*1.96*SD/ln 1.5)^2)), <= 40",
        "measure": "PI+", "budget_s": budget_s, "n_instances": n_instances, "cells": out, "inputs": [],
        "section12": (f"{_today()} — [пилот E2] число инстансов на ячейку: "
                      f"{', '.join(f'{c} = {n}' for c, n in n_instances.items())} — по правилу §3: "
                      "n = max(20, ⌈(2·1.96·SD/ln 1.5)²⌉), не больше 40; SD — наибольшее из SD log R⁺ и log R⁺_b "
                      "(R по PI⁺, A2) на нетривиальных калибровочных инстансах ячейки; ячейка с менее чем 8 такими "
                      "описательная, n = 20 (A1) — RQ2, H1c (60–120 тестовых инстансов) — кто согласовал: —"),
    }


def ttt_instances_report(
    rows: Sequence[Mapping], tuned: Mapping, *, label: str, config_size: int, k: int = 3,
    budget_s: float = 60.0,
) -> dict:
    """The ``k`` calibration instances of median difficulty (module docstring)."""
    _check_no_errors(rows)
    rows = _usable(rows)
    exact_id = _representatives(tuned, config_size)["exact"]
    refs = reference_table(rows)
    sel = [r for r in rows if r.get("cell") == label and r["config_id"] == exact_id
           and int(r["config_size"]) == config_size and r["budget_s"] == budget_s and r.get("stage") != "reference"]
    seeds: dict[str, int] = {}
    for row in sel:
        seeds[row["instance_id"]] = int(row["seed"])
    pis = _median_pi_by_instance(sel, refs, budget_s)
    if len(pis) < k:
        raise ValueError(f"only {len(pis)} calibration instances with the exact representative {exact_id!r}")
    median = float(np.median(list(pis.values())))
    ranked = sorted(pis, key=lambda i: (abs(pis[i] - median), seeds[i]))[:k]
    chosen = sorted(seeds[i] for i in ranked)
    return {
        "kind": "e2-ttt-instances", "plan_version": "v2.4", "label": label, "config_size": config_size,
        "exact_rep": exact_id, "budget_s": budget_s, "median_pi": median, "instances": chosen,
        "difficulty": sorted(({"seed": seeds[i], "instance_id": i, "pi": pis[i]} for i in pis),
                             key=lambda d: d["seed"]),
        "inputs": [],
        "section12": (f"{_today()} — [пилот E2] инстансы для TTT-кривых (N = {config_size}): сиды "
                      f"{', '.join(map(str, chosen))} — по правилу §6 (minor 10): три калибровочных инстанса с "
                      f"медианной трудностью (PI точного представителя {exact_id} за {budget_s:g} с, медиана "
                      f"{median:.3g}) — TTT-кривые (описательно) — кто согласовал: —"),
    }


def c4_lambda_report(
    draw: Callable[[int], Sequence],
    seeds: Iterable[int] = BOOT_CALIBRATION_SEEDS,
    lambdas: Iterable[float] = BOOT_LAMBDAS,
    min_median_scont: int = BOOT_MIN_MEDIAN_SCONT,
) -> dict:
    """The C4 λ rule (N7) as a report: :func:`cif.instances.boot.calibrate_boot_lambda` on the calibration
    instances ``draw(seed)``. ``lam_used`` is the λ the C4 stage runs at: the chosen one, or, if C4 is
    descriptive (``admissible`` false), the largest candidate: the plan does not say which."""
    seeds, lambdas = tuple(seeds), tuple(lambdas)
    result = calibrate_boot_lambda(draw, seeds, lambdas, min_median_scont)
    admissible = result.lam is not None
    lam_used = result.lam if admissible else max(lambdas)
    medians = ", ".join(f"{lam:g}: {m:g}" for lam, m in result.median_s_cont.items())
    if admissible:
        text = (f"λ = {result.lam:g} для ячейки C4 (u = 1); медианы s_cont по λ — {medians}; порог {min_median_scont}")
    else:
        text = (f"допустимого λ нет (медианы s_cont по λ — {medians}; порог {min_median_scont}): C4 становится "
                f"описательной и в сводную оценку не входит; прогоняется при λ = {lam_used:g}")
    return {
        "kind": "e2-c4-lambda", "plan_version": "v2.4", **result.as_dict(), "admissible": admissible,
        "lam_used": float(lam_used), "inputs": [],
        "section12": (f"{_today()} — [пилот E2] {text} — по правилу §3 (N7): наименьшее λ из {{5, 10, 20}} с "
                      "медианой s_cont на 10 калибровочных инстансах boot (сиды 1100–1109) не меньше 20 — RQ2 (C4) — "
                      "кто согласовал: —"),
    }


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def _provenance(paths: Sequence[Path]) -> list[dict]:
    """Where the rows came from: source, SHA-256 of the result file, number of rows, and for a registered
    run directory its run id, git SHA and config hash. The key is ``source``, not ``path``: a dict with
    ``path`` and ``sha256`` in an input JSON is taken by the runner for a recorded data file and
    re-hashed (:func:`cif.data.provenance.recorded_files`), which a run directory is not."""
    out = []
    for path in paths:
        file = path / "results.jsonl" if path.is_dir() else path
        entry = {"source": str(path), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                 "n_rows": len(read_rows(file))}
        manifest = path / "manifest.json" if path.is_dir() else None
        if manifest is not None and manifest.exists():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            entry.update({k: m.get(k) for k in ("run_id", "git_sha", "git_dirty", "config_hash", "mode")})
        out.append(entry)
    return out


def _rows_of(paths: Sequence[Path]) -> list[dict]:
    rows, errors = load_rows(paths)
    return [*rows, *errors]


def _write(report: dict, out: Path, inputs: list[dict]) -> None:
    report["inputs"] = inputs
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['kind']}: {out}")
    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}")


def _parse_cells(items: Sequence[str] | None) -> dict | None:
    if not items:
        return None
    cells = {}
    for item in items:
        name, _, rest = item.partition("=")
        label, _, size = rest.partition(":")
        cells[name] = {"label": label, "config_size": int(size)}
    return cells


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e2_pilot", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def paths(p, name, required=True):
        p.add_argument(name, type=Path, nargs="+" if required else "*", required=required, default=[])

    p = sub.add_parser("eligibility", help="the screening scan so far: state of the next round, selected "
                                           "instances, estimate of the share of trivial ones (§12 A1, A3)")
    p.add_argument("--config", type=Path, required=True, help="the config of the screening stage (its scan groups)")
    p.add_argument("--smoke", action="store_true", help="read the smoke version of the config")
    paths(p, "--rows", required=False)
    paths(p, "--reference", required=False)
    p.add_argument("--n-rule", type=Path, help="n_rule report: the targets of the scans that take them from it")
    p.add_argument("--c4-lambda", type=Path, help="c4_lambda report: C4 enters the pooled estimate only if admissible")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("screen", help="rank the 10 s screen and plan the recheck")
    paths(p, "--rows")
    paths(p, "--reference", required=False)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("tune", help="apply the 60 s recheck and name the tuned configurations")
    paths(p, "--screen")
    paths(p, "--recheck")
    paths(p, "--reference", required=False)
    p.add_argument("--eligibility", type=Path, help="calibration eligibility report: sizes that borrow configurations")
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--budget-long", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("n-rule", help="number of instances per cell")
    paths(p, "--rows")
    p.add_argument("--tuned", type=Path, required=True)
    p.add_argument("--eligibility", type=Path, help="calibration eligibility report: cells that are descriptive")
    p.add_argument("--cell", action="append", metavar="C1=T100:100",
                   help="cell=label:config_size (default: C1=T100:100 C2=T200:200 C3=T400:400 C4=CB200:200)")
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("ttt-instances", help="the calibration instances of median difficulty")
    paths(p, "--rows")
    p.add_argument("--tuned", type=Path, required=True)
    p.add_argument("--label", default="T200")
    p.add_argument("--config-size", type=int, default=200)
    p.add_argument("--k", type=int, default=3, help="number of instances (the plan: 3)")
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("c4-lambda", help="the λ rule of C4 on the real S&P 500 panel")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--min-median-scont", type=int, default=BOOT_MIN_MEDIAN_SCONT)
    p.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "eligibility":
        return _eligibility_command(args)
    if args.command == "screen":
        report = screen_report(_rows_of([*args.rows, *args.reference]), budget_s=args.budget)
        _write(report, args.out, _provenance([*args.rows, *args.reference]))
    elif args.command == "tune":
        borrow, extra = None, []
        if args.eligibility:
            elig = json.loads(args.eligibility.read_text(encoding="utf-8"))
            if not elig.get("config_from_final", False):
                raise ValueError("the calibration scan is not finished: run another round of elig_calibration "
                                 "before tuning (configs/E2/README.md, step 1)")
            borrow = {int(n): d for n, d in (elig.get("config_from") or {}).items()}
            extra = [_file_entry(args.eligibility)]
        report = tuned_report(_rows_of([*args.screen, *args.reference]), _rows_of(args.recheck),
                              budget_s=args.budget, budget_long_s=args.budget_long, borrow=borrow)
        _write(report, args.out, [*_provenance([*args.screen, *args.recheck, *args.reference]), *extra])
    elif args.command == "n-rule":
        tuned = json.loads(args.tuned.read_text(encoding="utf-8"))
        cells = _parse_cells(args.cell)
        descriptive, extra = None, []
        if args.eligibility:
            elig = json.loads(args.eligibility.read_text(encoding="utf-8"))
            descriptive = _descriptive_cells(elig, cells or DEFAULT_CELLS)
            extra = [_file_entry(args.eligibility)]
        report = n_rule_report(_rows_of(args.rows), tuned, cells=cells, budget_s=args.budget,
                               descriptive=descriptive)
        _write(report, args.out, [*_provenance(args.rows), _file_entry(args.tuned), *extra])
    elif args.command == "ttt-instances":
        tuned = json.loads(args.tuned.read_text(encoding="utf-8"))
        report = ttt_instances_report(_rows_of(args.rows), tuned, label=args.label,
                                      config_size=args.config_size, k=args.k, budget_s=args.budget)
        _write(report, args.out, [*_provenance(args.rows), _file_entry(args.tuned)])
    else:
        from cif.experiments import e2_instances
        from cif.instances.boot import boot_instance

        data = e2_instances.load_panel(str(args.data_dir))
        report = c4_lambda_report(lambda seed: boot_instance(data, seed, n=args.n),
                                  min_median_scont=args.min_median_scont)
        # no registered run stands behind this report (the rule needs no solver): its lineage is the state of the
        # code, as in the manifest of a run. Unknown (outside git) is None, never "clean" (audit G1, L-4)
        state = registry._git_state()
        report.update(git_sha=state["sha"], git_dirty=state["dirty"])
        _write(report, args.out, [_file_entry(args.data_dir / "provenance.json")])
    return 0


def _descriptive_cells(eligibility: Mapping, cells: Mapping[str, Mapping]) -> dict[str, str]:
    """The cells whose calibration set (``label``) has too few non-trivial instances (eligibility report). Only a
    finished scan decides that: a set that is short midway may still reach the minimum."""
    unfinished = [label for label, st in (eligibility.get("cells") or {}).items() if not st.get("done")]
    if unfinished:
        raise ValueError(f"the calibration scan is not finished ({', '.join(unfinished)}): run another round of "
                         "elig_calibration before the tuning rules are applied")
    out = {}
    for cell, spec in cells.items():
        state = (eligibility.get("cells") or {}).get(spec["label"])
        if state is not None and not state["enough"]:
            out[cell] = (f"{len(state['selected'])} non-trivial calibration instances, fewer than the "
                         f"{state['min_nontrivial']} the plan needs (§12 A1)")
    return out


def _eligibility_command(args) -> int:
    """``e2_pilot eligibility``: the report of the rounds of a screening config so far (module docstring of
    :mod:`cif.experiments.e2_eligibility`). The rows are those of the screening runs (``--rows``) and of the
    reference runs (``--reference``: the registered run that certified the plan's calibration instances, and the
    later reference runs, which only serve the check of the scale)."""
    cfg = effective_config(load_config(args.config), smoke=args.smoke)
    params = cfg["params"]
    n_rule = None
    inputs: list[dict] = [_file_entry(args.config)]
    if args.n_rule:
        n_rule = json.loads(args.n_rule.read_text(encoding="utf-8"))["n_instances"]
        inputs.append(_file_entry(args.n_rule))
    c4_admissible = None
    if args.c4_lambda:
        c4_admissible = bool(json.loads(args.c4_lambda.read_text(encoding="utf-8")).get("admissible"))
        inputs.append(_file_entry(args.c4_lambda))
    specs = [e2_eligibility.spec_from_group(g, n_rule=n_rule) for g in params["instances"]]
    tau = float((params.get("eligibility") or {}).get("tau", e2_eligibility.TAU))
    role = params["seed_role"]
    screening, reference = _rows_of(args.rows), _rows_of(args.reference)
    problems = e2_eligibility.verify_rows(screening, tau)
    if problems:
        raise ValueError("the recorded classes do not follow from the recorded evidence:\n  " + "\n  ".join(problems))
    report = e2_eligibility.build_report(
        [*screening, *reference], specs, role=role, tau=tau, c4_admissible=c4_admissible,
        references=[*reference, *screening])
    if any(s.cell == "C4" for s in specs) and c4_admissible is None:
        report["warnings"].append("C4 is in the scan but no c4_lambda report was given: it is left out of the pooled "
                                  "estimate")
    _write(report, args.out, [*_provenance([*args.rows, *args.reference]), *inputs])
    for label, state in report["cells"].items():
        n_sel = len(state["selected"])
        target = "all" if state["target"] is None else state["target"]
        status = "done" if state["done"] else "next round: " + str(len(state["next_block"])) + " units"
        print(f"  {label}: {n_sel} of {target} non-trivial in {state['protocol']['n_scanned']} scanned "
              f"(screened from the start of the order: {state['prefix']} of {state['cap']}), {status}"
              f"{', descriptive' if state['descriptive'] else ''}")
    print(f"scan {'finished' if report['done'] else 'not finished: run another round'}")
    print("estimate of A3 (the instances scanned so far, up to the n-th non-trivial one):")
    for label, state in report["cells"].items():
        print("  " + e2_eligibility.format_estimate(label, state["estimate"]))
    for name, group in report["groups"].items():
        if group["n_scanned"]:
            print(f"  {name} (pooled): trivial {group['n_trivial']} of {group['n_scanned']} "
                  f"({100 * group['share_trivial']:.0f} %, Wilson {100 * group['ci95'][0]:.0f}–"
                  f"{100 * group['ci95'][1]:.0f} %)")
        else:
            print(f"  {name} (pooled): nothing scanned")
    return 0


def _file_entry(path: Path) -> dict:
    return {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    sys.exit(main())
