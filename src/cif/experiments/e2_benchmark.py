"""E2: benchmark of exact methods, QUBO heuristics and local search (RQ2, H1c; ANALYSIS_PLAN §2–§6).

Experiment module of the runner (:mod:`cif.experiments.run`, protocol of :mod:`cif.experiments.toy`).
One module serves every stage of E2; a stage is a list of cells of one kind of run, chosen by
``params.stage`` and by where the methods come from (``params.methods.source``):

===============  =========================================  ===============================================
stage            what                                       methods
===============  =========================================  ===============================================
``eligibility``  is the common start within τ of the        ``reference``: SCIP, long limit, no budget
                 optimum? (§12 A1: the instance is
                 trivial / non-trivial / undecided)
``screen``       tuning at 10 s (§6 N6)                     ``screen``: ≤ 30 random configurations of each
                                                            heuristic and the exact variants, per N
``recheck``      the winner and the runner-up at 60 s       ``recheck``: from the screen report
``reference``    certified optimum (``absgap`` = 1e-6·s)    ``reference``: SCIP, long limit, no budget
``calib_boot``   representatives on the C4 calibration      ``tuned``
``bench``        C1–C5, D1 at 60 s (RQ2)                    ``tuned``: representatives and heuristics
``h1c``          tabu against SA at 10 s (H1c)              ``tuned``: the ``h1c_*`` roles
``ttt``          100 runs on 3 instances (TTT curves)       ``tuned``
``cold``         no common start, one seed per method and   ``tuned``: the representatives
                 instance (§12 A4: "until the method's
                 own first solution")
===============  =========================================  ===============================================

The stage ``eligibility`` comes first: it decides which instances every later stage runs on, from solutions
found and dual bounds proved by SCIP started at the common start, **before any method runs on the instance**
(:mod:`cif.experiments.e2_eligibility`, §12 A1). It runs in rounds (a scan of seeds in increasing order until
``n`` non-trivial instances), and the report of its rows (``e2_pilot eligibility``) is both the state of the next
round and the input ``eligibility_*`` of the stages that follow.

A **cell** is one run: instance (``family`` with ``n`` and ``seed``, or ``port`` and ``lam_index``), method
spec (:mod:`cif.experiments.e2_space`), solver seed, end-to-end ``budget_s`` (the reference has a
``time_limit_s`` instead). The cell carries everything the result depends on except the fixed
parameters of ``params``; nothing computed from data (K, ``binding``, the λ of an ORlib point) is in it,
so the grid is the same on every machine. Cells are ordered instance by instance (builds are cached).

Config (YAML) ``params``::

    stage, seed_role: calibration | test        # instances 1000-1999 / 2000-2999, solvers 10000-19999 / 20000-29999
    budget_s: 10.0                              # null for the reference
    warm_up: true                               # one untimed run of every solver per process
    overrun_tol_rel: 0.05                       # budget_ok: end_s <= budget_s·(1 + tol)
    fixed: {lam, fraction, boot_data_dir, boot_n, orlib_dir, orlib_k, orlib_w_min, orlib_w_max, orlib_m}
    solver: {mip_gap, abs_gap_rel, abs_gap_safety, trajectory_tol}
    solver_seeds: {exact: [...], heuristic: [...]}      # or one list for both classes
    instances: [{cell, family, n, config_size,          # factor / boot / dense
                 seeds: [...] | first_seed + count | first_seed + count_from: n_rule | seeds_from: ttt_instances
                 | scan: {...}                           # stage eligibility: the next round of a scan (below)
                 | eligible: {report, cell, field, require_enough}},   # the instances the scan selected
                {cell, family: orlib, ports: [...], lam_indices: 25 | [...], config_size,
                 eligible: {report, cell}}]              # ORlib: the selected (port, λ) points
    methods: {source: explicit | screen | recheck | tuned | reference, ...}
    eligibility: {tau: 1e-3, certificate: tau | full, margin: 1e-3}     # stage eligibility only

``config_size`` selects the tuned configurations (§3: ORlib with N ≤ 100 uses those of N = 100, the other
ports and C4 those of N = 200); an integer, or a list of rules ``[{max_n: 100, size: 100}, {size: 200}]``.

``scan`` (stage ``eligibility``, seed families): ``{first | head + then_from, cap, target | target_from: n_rule,
min_nontrivial, block, first_block, state: <input label>}``. The cells of a round are the next block of the
order of the scan, computed from the *state* (an ``e2-eligibility`` report of the rounds so far, an ``inputs``
entry, a stub before the first round) and the counts of the plan
(:func:`cif.experiments.e2_eligibility.next_block`); a finished cell has none. ``eligible`` (every later
stage): the seeds, or the ORlib points, that the finished scan selected for ``cell`` (``field``:
``selected``, or ``needs_reference`` for the reference stage; ``require_enough``: nothing if the cell has fewer
than the minimum of the plan, as for a calibration size that borrows its configurations). A report that is a
stub, whose scan is not finished or that lacks the cell stops the stage.

Reports of the pilot (:mod:`cif.experiments.e2_pilot`) enter as ``inputs`` (hashed into the manifest):
``tuned`` (:data:`TUNED_KIND`), ``screen``, ``n_rule``, ``ttt_instances``, ``c4_lambda``. A descriptive C4
(no admissible λ) is run at the largest candidate λ the report names (``lam_used``); the plan does not
say which λ then. A report flagged ``"smoke": true`` (the hand-written stand-ins of ``configs/E2/smoke``)
is accepted in a smoke run only.

The top-level ``seeds``, if given, must be the seeds of the grid. A smoke run (``mode: smoke``) refuses test
seeds: smoke runs use calibration seeds only.

**Run.** The instance is built outside the timed region. Every method starts with the common start (top-K
rounding of the relaxation, refined; :mod:`cif.heuristics.start`) built inside its own run: its time is
``start_time_s`` and counts for every method; the exact solvers get it as ``warm_start`` and
``clock_offset_s``, the heuristics as ``start``. All times are seconds on the one end-to-end clock. One thread
for everything. ``abs_gap`` of the exact methods is ``abs_gap_rel·abs_gap_safety·s(start)``: ``limits/absgap`` =
1e-6·s_ref needs the scale of the optimum, which is not known beforehand; the scale of the start stands in
for it, halved for safety. Whether a run is certified is decided afterwards, on its own incumbent, by
:func:`cif.metrics.gap.reference_from_exact`. A **cold** run (stage ``cold``) gets no start: the clock begins at
the build of the method, the trajectory lists everything the method found from its first solution on, and the
``start_*`` fields are ``None``.

**Screening run** (stage ``eligibility``). SCIP, perspective ``mineig``, warm-started at the common start,
``time_limit_s`` (2 h, §5), the start's time as the clock offset. ``certificate: tau`` stops it as soon as the
question is answered: ``limits/primal`` = ``f_start − τ·s_start·(1 + margin)`` (a solution that much better: the
instance is non-trivial) and ``limits/dual`` = ``f_start − τ·s_start·(1 − margin)`` (a dual bound that close: the
start is within τ of the optimum, trivial); ``s_start`` is the scale of the start and ``margin`` keeps a stop on
SCIP's own rounding from landing on the wrong side of the exact rule. The plan words the dual side as
``limits/absgap = τ·s``: that is the same statement while the incumbent is the start, and the dual limit is the
one that also holds when SCIP has found an improvement smaller than τ·s. ``certificate: full`` is the reference
run of §5 (``limits/absgap`` = 1e-6·0.5·s_start, to the optimum): the class is the same, ``ĝ_start`` is exact and
the row is a reference (``f_star``). On trivial instances of N = 100 and 200 the two cost about the same (the tree
has to be explored nearly to the end before the bound is within τ), and ``full`` saves the reference run of the
non-trivial ones. The row adds ``eligibility`` (the class) and ``evidence`` (the numbers it follows from,
:func:`cif.experiments.e2_eligibility.classify`); ``t_cert_s`` is the time at which the stopping rule of the
trivial side was met (the clock of the run, the construction of the start included).

**Row** (besides the cell): the instance (``instance_id, n_assets, K, card_mode, lam, s_cont, binding,
cond_sigma, cv_vol, cv_mu``, …); the start (``start_time_s, start_cpu_s, start_f, start_scale, start_status,
start_subset``); the result (``f_final``, the returned objective, ``scale_final`` = s of the returned
portfolio, ``support`` and ``weights``, ``feasible``, ``status``); the budget (``end_s``, ``overrun_s``,
``budget_ok``, ``method_wall_s``, ``method_cpu_s``) and the split of the method's time into ``build_s``,
``solve_s`` and ``refine_s`` (:func:`_time_parts`, R10); the **trajectory** (the method's improvements over the
start, ``[t, f]``, strictly decreasing, improvements of more than ``trajectory_tol``·s(start); with the start
``(start_time_s, start_f)`` it gives the incumbent curve for the primal integral and time-to-target, without
it the variant "γ = 1 until the first improvement of the common start"); ``f_at_budget`` (incumbent at ``budget_s``, the start
included); ``first_own_s`` (end of the first batch / refinement the method itself produced, whether it
improved on the start or not; for SCIP and OA the first improvement); ``t_first_improve_s``; ``t_primal_s``
(time of the last improvement of the incumbent, the start's time if none); for the exact methods ``proved``
(the solver says optimal), ``certified`` (and its absolute gap within 1e-6·s) and ``t_cert_s``; ``detail``
(solver diagnostics: batches, nodes, bounds, build / solve / refine times, parameters).
``first_own_f`` is the value of the first own solution next to its time ``first_own_s`` (for the heuristics the
best value refined in the first batch that refined anything, for LS the first K-subset its descent refined from
the start or, in a cold run, the end of its first greedy construction (audit G2, C-4), for SCIP and OA the first
improvement over the start; in a cold run the first point of the trajectory).
The reference stage adds ``f_star`` (polished with the shared refinement, as :func:`cif.heuristics.oracle.
certified_optimum`), ``scale_star`` and ``ref_type``.

**Readings of the plan** (the plan is frozen; where it is open, the most literal reading is taken)::

* the calibration set of N = 100, 200, 400 is the consecutive seeds 1000-1007, 1008-1015, 1016-1023 (8 per N);
* the common start is built by every run (not once per instance) and its measured time is charged to that
  run; the first call of a process is preceded by an untimed warm-up of every solver (``warm_up``);
* SCIP's ``randomseedshift``, ``permutevars`` and ``permutationseed`` all follow the one derived seed; OA
  takes the same seed as HiGHS' ``random_seed`` and the perspective split ``mineig`` (the plan lists OA as
  one variant); **Tabu's ``num_reads_b`` is 10 in every configuration of E2, not tuned** (the plan lists only
  the restarts per batch, :mod:`cif.experiments.e2_space`): a constant of this implementation, to be disclosed
  in Methods. The exploratory configuration behind H1c had 200 reads, so H1c repeats the direction of that
  finding and not its configuration;
* a cell on an instance where the cardinality constraint does not bind (``s_cont <= K``) is skipped when the
  stage says ``skip_nonbinding`` (RQ2 compares binding instances; D1 keeps all): in the screening the 125 ORlib
  points are all cells and the binding ones are found at run time, because that needs a solve and must not decide
  the grid (the later stages run the points the screening selected);
* the certification run is SCIP perspective-``mineig`` with SCIP's defaults and a 7200 s limit (§5: 2 h);
* instances are decided before the methods run (stage ``eligibility``); the cells of the later stages are the
  instances the finished scan selected, not seeds counted from a first seed; every cell of the test stages
  scans from seed 2000 (the instances of different N or families at one seed are not related);
* not done: the validation of the exact stack against the published results of Bertsimas & Cory-Wright and
  Wei & Küçükyavuz (§2: it needs the original return-constrained model, which :mod:`cif.classical.scip_native`
  does not have, and their published numbers). The *confined* primal integral was dropped in §12 (A4): its
  definition was not extracted from the source and it was only a sensitivity analysis; the honest "until the
  method's own first solution" variant comes from the cold stage.
"""

from __future__ import annotations

import json
import math
import time
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cif.classical.oa import solve_oa
from cif.classical.refine import refine_problem
from cif.classical.scip_native import solve_miqp_native
from cif.experiments import e2_eligibility, e2_instances, e2_space
from cif.experiments.run import resolve_inputs
from cif.heuristics.anytime import (
    SABatches,
    SwapSABatches,
    TabuBatches,
    run_anytime,
)
from cif.heuristics.local_search import local_search
from cif.heuristics.start import CommonStart, common_start
from cif.heuristics.swap_sa import SwapSurrogate, calibrate_hot_temperature
from cif.instances.orlib import ORLIB_N
from cif.metrics.gap import reference_from_exact, reference_scale
from cif.problem import PortfolioProblem, Solution

__all__ = [
    "ROLES",
    "SEED_RANGES",
    "STAGES",
    "TUNED_KIND",
    "budget_compliance",
    "cells",
    "clear_caches",
    "derive_seed",
    "run_cell",
]

STAGES = ("eligibility", "screen", "recheck", "reference", "bench", "h1c", "ttt", "calib_boot", "cold")
SOURCES = ("explicit", "screen", "recheck", "tuned", "reference")
#: Roles of a tuned configuration in the ``tuned`` report (``sizes[N]["roles"]``).
ROLES = ("exact_rep", "qubo_rep", "ls", "sa", "tabu", "swap_sa", "h1c_sa", "h1c_tabu")
TUNED_KIND = "e2-tuned"
#: seed ranges of ANALYSIS_PLAN §3 (M2): (instances, solvers) for each role
SEED_RANGES = {
    "calibration": ((1000, 2000), (10_000, 20_000)),
    "test": ((2000, 3000), (20_000, 30_000)),
}
EXACT_METHODS = ("scip", "oa")
_FAMILY_CODE = {"factor": 1, "boot": 2, "orlib": 3, "dense": 4}
_METHOD_CODE = {"sa": 1, "tabu": 2, "swap_sa": 3, "ls": 4, "scip": 5, "oa": 6}
_CALIBRATION_TAG = 0xCA11B  # separates the temperature calibration stream from the sampler stream
#: hot temperature: acceptance of random swaps; cold temperature: ratio to the hot one (swap-SA defaults)
ACCEPT_START = 0.8
COLD_RATIO = 0.01
_DEFAULT_REFERENCE_SPEC = {"name": "scip", "id": "ref-scip-persp-mineig", "formulation": "perspective",
                           "diag": "mineig", "usesymmetry": "default"}
_INSTANCE_KEYS = {"cell", "family", "n", "seeds", "first_seed", "count", "count_from", "seeds_from",
                  "config_size", "ports", "lam_indices", "skip_nonbinding", "scan", "eligible"}
#: stages that run one long SCIP run per instance, with a ``time_limit_s`` and no budget
LONG_STAGES = ("reference", "eligibility")
_ELIGIBILITY_MODES = ("tau", "full")
_DEFAULT_ELIGIBILITY_MARGIN = 1e-3
#: keys of a solver's ``solver_meta`` that are long series, not diagnostics
_SERIES = {"trajectory", "trajectory_primal_bound", "incumbent_trajectory", "bound_trajectory", "subset",
           "batches"}
_DEFAULT_TRAJECTORY_TOL = 1e-9

_warmed_up = False


# --------------------------------------------------------------------------------------------
# Inputs (reports of the pilot)
# --------------------------------------------------------------------------------------------


def _report(cfg: Mapping, label: str, kind: str) -> dict:
    paths = resolve_inputs(dict(cfg))
    if label not in paths:
        raise ValueError(f"this config needs the input {label!r} (a {kind} report): add it to 'inputs'")
    data = json.loads(Path(paths[label]).read_text(encoding="utf-8"))
    if data.get("kind") != kind:
        raise ValueError(f"inputs.{label} must be a {kind} report, its kind is {data.get('kind')!r}")
    if data.get("smoke") and cfg.get("mode") != "smoke":
        raise ValueError(f"inputs.{label} is a hand-written stand-in for smoke rehearsals ('smoke': true): "
                         "a run that is not a smoke run needs the report of the pilot")
    return data


# --------------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------------


def _need(mapping: Mapping, key: str, where: str):
    if mapping.get(key) is None:
        raise ValueError(f"{where}: missing '{key}'")
    return mapping[key]


def _config_size(rule: Any, n_assets: int) -> int:
    if isinstance(rule, int) and not isinstance(rule, bool):
        return rule
    for entry in rule:
        if entry.get("max_n") is None or n_assets <= entry["max_n"]:
            return int(entry["size"])
    raise ValueError(f"config_size {rule!r} has no rule for {n_assets} assets")


def _count(group: Mapping, cfg: Mapping) -> int:
    if group.get("count_from"):
        report = _report(cfg, group["count_from"], "e2-n-rule")
        n = report["n_instances"].get(group["cell"])
        if n is None:
            raise ValueError(f"the n_rule report has no instance count for the cell {group['cell']!r}")
        return int(n)
    return int(_need(group, "count", f"instances ({group['cell']})"))


def _eligible_units(group: Mapping, cfg: Mapping) -> list:
    """The units (seeds, or ``[port, λ index]`` points) the finished scan of the eligibility report selected."""
    spec = group["eligible"]
    if not isinstance(spec, Mapping) or not spec.get("report"):
        raise ValueError(f"instances ({group['cell']}): 'eligible' needs the label of an eligibility report "
                         "(report: <input>, cell: <cell>)")
    report = _report(cfg, spec["report"], e2_eligibility.REPORT_KIND)
    return e2_eligibility.selected_units(
        report, spec.get("cell", group["cell"]), field=spec.get("field", "selected"),
        require_enough=bool(spec.get("require_enough", False)))


def _scan_seeds(group: Mapping, cfg: Mapping) -> list[int]:
    """The seeds of the next round of a scan: the block that follows what the state report has screened."""
    scan = group["scan"]
    n_rule = None
    if isinstance(scan, Mapping) and scan.get("target_from"):
        n_rule = _report(cfg, scan["target_from"], "e2-n-rule")["n_instances"]
    spec = e2_eligibility.spec_from_group(group, n_rule=n_rule)
    state = _report(cfg, scan["state"], e2_eligibility.REPORT_KIND)
    prefix, found = e2_eligibility.progress(state, group["cell"])
    return [int(seed) for seed in e2_eligibility.next_block(spec, prefix, found)]


def _instance_seeds(group: Mapping, cfg: Mapping) -> list[int]:
    where = f"instances ({group['cell']})"
    if group.get("scan") is not None:
        return _scan_seeds(group, cfg)
    if group.get("eligible") is not None:
        return [int(s) for s in _eligible_units(group, cfg)]
    if group.get("seeds") is not None:
        return [int(s) for s in group["seeds"]]
    if group.get("seeds_from"):
        report = _report(cfg, group["seeds_from"], "e2-ttt-instances")
        return [int(s) for s in report["instances"]]
    first = int(_need(group, "first_seed", where))
    return [first + i for i in range(_count(group, cfg))]


def _boot_lambda(cfg: Mapping) -> float:
    return float(_report(cfg, "c4_lambda", "e2-c4-lambda")["lam_used"])


def _expand_instances(cfg: Mapping, p: Mapping) -> list[dict]:
    """The instance coordinates of every group, in order, with their config size."""
    groups = p.get("instances")
    if not groups:
        raise ValueError("params.instances must list at least one group of instances")
    out: list[dict] = []
    for group in groups:
        extra = set(group) - _INSTANCE_KEYS
        if extra:
            raise ValueError(f"instances ({group.get('cell')}): unknown keys {sorted(extra)}")
        cell, family = _need(group, "cell", "instances"), _need(group, "family", "instances")
        if family not in e2_instances.FAMILIES:
            raise ValueError(f"instances ({cell}): unknown family {family!r}")
        rule = group.get("config_size")
        skip = bool(group.get("skip_nonbinding", p.get("skip_nonbinding", False)))
        if family == "orlib":
            lam_indices = group.get("lam_indices", 25)
            indices = list(range(lam_indices)) if isinstance(lam_indices, int) else [int(i) for i in lam_indices]
            points = [(int(port), i) for port in _need(group, "ports", f"instances ({cell})") for i in indices]
            if group.get("eligible") is not None:
                points = [(int(port), int(i)) for port, i in _eligible_units(group, cfg)]
            for port, i in points:
                n = ORLIB_N[port]
                size = _config_size(rule if rule is not None else n, n)
                out.append({"cell": cell, "family": family, "n": n, "port": port, "lam_index": i,
                            "config_size": size, "skip_nonbinding": skip})
            continue
        n = int(_need(group, "n", f"instances ({cell})"))
        size = _config_size(rule if rule is not None else n, n)
        lam = _boot_lambda(cfg) if family == "boot" else None
        for seed in _instance_seeds(group, cfg):
            entry = {"cell": cell, "family": family, "n": n, "seed": seed, "config_size": size,
                     "skip_nonbinding": skip}
            if lam is not None:
                entry["boot_lam"] = lam
            out.append(entry)
    return out


def _methods(cfg: Mapping, p: Mapping, instance: Mapping) -> list[tuple[dict, list[str]]]:
    """``(spec, roles)`` of the methods run on ``instance``, without repeats."""
    m = p.get("methods") or {}
    source = m.get("source")
    if source not in SOURCES:
        raise ValueError(f"params.methods.source must be one of {SOURCES}, got {source!r}")
    size = instance["config_size"]
    specs: list[tuple[dict, list[str]]]
    if source == "explicit":
        specs = [(dict(s), []) for s in _need(m, "specs", "params.methods")]
    elif source == "screen":
        names = list(_need(m, "names", "params.methods"))
        specs = []
        for name in names:
            if name == "exact":
                specs += [(s, []) for s in e2_space.exact_candidates(instance["n"])]
            else:
                specs += [(s, []) for s in e2_space.sample_configs(
                    name, int(_need(m, "n_configs", "params.methods")),
                    int(_need(m, "tuning_seed", "params.methods")), instance["n"])]
    elif source == "recheck":
        screen = _report(cfg, "screen", "e2-screen")
        entry = (screen.get("sizes") or {}).get(str(size))
        if entry is None:
            raise ValueError(f"the screen report has no plan for config size {size}")
        specs = [(dict(item["spec"]), []) for item in entry["recheck"]]
    elif source == "tuned":
        tuned = _report(cfg, "tuned", TUNED_KIND)
        entry = (tuned.get("sizes") or {}).get(str(size))
        if entry is None:
            raise ValueError(f"the tuned report has no configurations for config size {size}")
        specs = []
        for role in _need(m, "roles", "params.methods"):
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
            if role not in entry["roles"]:
                raise ValueError(f"the tuned report has no {role!r} for config size {size}")
            specs.append((dict(entry["roles"][role]), [role]))
    else:  # reference
        specs = [(dict(m.get("spec") or _DEFAULT_REFERENCE_SPEC), [])]
    merged: dict[tuple[str, str], tuple[dict, list[str]]] = {}
    for spec, roles in specs:
        e2_space.check_method(spec)
        key = (spec["name"], spec["id"])
        if key in merged:
            merged[key][1].extend(r for r in roles if r not in merged[key][1])
        else:
            merged[key] = (spec, list(roles))
    return [(spec, sorted(roles)) for spec, roles in merged.values()]


def _eligibility_params(p: Mapping) -> dict:
    """``params.eligibility`` of the screening stage, validated: ``tau`` in (0, 1), ``certificate`` ``tau`` /
    ``full``, ``margin`` in [0, 1)."""
    e = p.get("eligibility")
    if not isinstance(e, Mapping):
        raise ValueError("the stage 'eligibility' needs params.eligibility: {tau, certificate, margin}")
    tau = e.get("tau", e2_eligibility.TAU)
    margin = e.get("margin", _DEFAULT_ELIGIBILITY_MARGIN)
    certificate = e.get("certificate", "tau")
    if not (isinstance(tau, int | float) and not isinstance(tau, bool) and 0.0 < tau < 1.0):
        raise ValueError(f"params.eligibility.tau must lie in (0, 1), got {tau!r}")
    if not (isinstance(margin, int | float) and not isinstance(margin, bool) and 0.0 <= margin < 1.0):
        raise ValueError(f"params.eligibility.margin must lie in [0, 1), got {margin!r}")
    if certificate not in _ELIGIBILITY_MODES:
        raise ValueError(f"params.eligibility.certificate must be one of {_ELIGIBILITY_MODES}, got {certificate!r}")
    return {"tau": float(tau), "margin": float(margin), "certificate": certificate}


def _expand_seeds(value: Any, where: str) -> list[int]:
    """A list of seeds, or ``{first: s, count: n}`` for ``s, s + 1, …, s + n − 1``."""
    if isinstance(value, Mapping):
        first, count = value.get("first"), value.get("count")
        if not (isinstance(first, int) and isinstance(count, int) and count >= 1):
            raise ValueError(f"{where}: a seed range needs an integer 'first' and a 'count' of at least 1, "
                             f"got {dict(value)}")
        return [first + i for i in range(count)]
    return [int(s) for s in value]


def _seeds_of(p: Mapping, spec: Mapping) -> list[int]:
    seeds = p.get("solver_seeds")
    if isinstance(seeds, Mapping) and "first" not in seeds:
        key = "exact" if spec["name"] in EXACT_METHODS else "heuristic"
        seeds = seeds.get(key)
    if seeds is None or (not isinstance(seeds, Mapping) and not len(seeds)):
        raise ValueError(f"params.solver_seeds has no seeds for the method {spec['name']!r}")
    return _expand_seeds(seeds, "params.solver_seeds")


def _check_seeds(cfg: Mapping, role: str, cells_: list[dict]) -> None:
    (lo_i, hi_i), (lo_s, hi_s) = SEED_RANGES[role]
    inst = sorted({c["seed"] for c in cells_ if "seed" in c})
    solv = sorted({c["solver_seed"] for c in cells_})
    bad_i = [s for s in inst if not lo_i <= s < hi_i]
    bad_s = [s for s in solv if not lo_s <= s < hi_s]
    if bad_i or bad_s:
        raise ValueError(
            f"seed_role={role!r} takes instance seeds in [{lo_i}, {hi_i}) and solver seeds in "
            f"[{lo_s}, {hi_s}); out of range: instance seeds {bad_i}, solver seeds {bad_s}"
        )
    declared = cfg.get("seeds")
    if declared is not None and sorted({int(s) for s in declared}) != sorted({*inst, *solv}):
        raise ValueError(
            f"the top-level seeds {sorted(declared)} (recorded in the manifest) must be the seeds of "
            f"the grid {sorted({*inst, *solv})}"
        )


def cells(cfg: dict) -> list[dict]:
    """The grid, instance by instance: instances × methods × solver seeds (module docstring)."""
    p = cfg.get("params")
    if not p:
        raise ValueError("config: missing 'params'")
    stage = p.get("stage")
    if stage not in STAGES:
        raise ValueError(f"params.stage must be one of {STAGES}, got {stage!r}")
    role = p.get("seed_role")
    if role not in SEED_RANGES:
        raise ValueError(f"params.seed_role must be one of {sorted(SEED_RANGES)}, got {role!r}")
    if cfg.get("mode") == "smoke" and role == "test":
        raise ValueError("a smoke run uses calibration seeds only: seed_role 'test' is refused in smoke mode")
    budget = p.get("budget_s")
    reference = stage in LONG_STAGES
    if stage == "eligibility":
        _eligibility_params(p)
        methods = p.get("methods") or {}
        if methods.get("source") != "reference":
            raise ValueError("the stage 'eligibility' runs the reference solver: methods.source must be 'reference'")
        if (methods.get("spec") or _DEFAULT_REFERENCE_SPEC).get("name") != "scip":
            raise ValueError("the stage 'eligibility' is run by SCIP: methods.spec must be a SCIP spec")
    if reference:
        limit = (p.get("methods") or {}).get("time_limit_s")
        if budget is not None or not (isinstance(limit, int | float) and limit > 0):
            raise ValueError(f"the {stage} stage has no budget_s and needs methods.time_limit_s > 0")
    elif not (isinstance(budget, int | float) and not isinstance(budget, bool) and budget > 0):
        raise ValueError(f"params.budget_s must be a positive number, got {budget!r}")
    out = []
    for instance in _expand_instances(cfg, p):
        for spec, roles in _methods(cfg, p, instance):
            for solver_seed in _seeds_of(p, spec):
                cell = {"stage": stage, **instance, "method": spec}
                if roles:
                    cell["roles"] = roles
                cell["solver_seed"] = solver_seed
                cell["budget_s"] = None if reference else float(budget)
                if reference:
                    cell["time_limit_s"] = float(p["methods"]["time_limit_s"])
                out.append(cell)
    _check_seeds(cfg, role, out)
    return out


# --------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------


def derive_seed(cell: Mapping) -> int:
    """The 31-bit seed of the run: ``SeedSequence`` entropy from the solver seed and every other
    coordinate that identifies the stream (instance, size of the configuration, method and its spec)
    but not the budget or the stage, so that a longer run extends a shorter one of the same cell
    (§3: streams from the seed and indices). ``dwave-samplers`` refuse seeds from ``2**31`` on."""
    ident = [cell["seed"], cell["n"]] if "seed" in cell else [cell["port"], cell["lam_index"]]
    spec = cell["method"]
    spec_hash = zlib.crc32(json.dumps(spec, sort_keys=True).encode("utf-8"))
    entropy = [int(cell["solver_seed"]), _FAMILY_CODE[cell["family"]], *map(int, ident),
               int(cell.get("config_size", 0)), _METHOD_CODE[spec["name"]], spec_hash]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF


# --------------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------------


class _SAObjectiveBeta(SABatches):
    """SA whose β range is scaled to the objective, as the swap-SA temperature: hot where random swaps
    of the surrogate are accepted ``ACCEPT_START`` of the time, cold ``COLD_RATIO`` of that. The
    calibration is part of ``prepare``, so inside the timed region of the run."""

    def __init__(self, *args, calibration_seed: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._calibration_seed = calibration_seed

    def prepare(self) -> None:
        super().prepare()
        p = self.problem
        surrogate = SwapSurrogate(p.mu, p.sigma, p.risk_aversion, self.K)
        rng = np.random.default_rng(np.random.SeedSequence([self._calibration_seed, _CALIBRATION_TAG]))
        t_hot = calibrate_hot_temperature(surrogate, rng, ACCEPT_START)
        self.beta_range = (1.0 / t_hot, 1.0 / (COLD_RATIO * t_hot))


def _anytime_sampler(spec: Mapping, problem: PortfolioProblem, k: int, seed: int):
    name = spec["name"]
    if name == "sa":
        kwargs = {"num_reads_b": spec["num_reads_b"], "num_sweeps": spec["num_sweeps"],
                  "penalty_multiplier": spec["penalty"]}
        if spec["beta"] == "objective":
            return _SAObjectiveBeta(problem, k, calibration_seed=seed, **kwargs)
        return SABatches(problem, k, **kwargs)
    if name == "tabu":
        return TabuBatches(problem, k, num_restarts_b=spec["num_restarts_b"],
                           num_reads_b=spec["num_reads_b"], tenure=spec["tenure"],
                           penalty_multiplier=spec["penalty"])
    return SwapSABatches(problem, k, num_reads_b=spec["num_reads_b"], num_sweeps=spec["num_sweeps"])


def _abs_gap(solver: Mapping, problem: PortfolioProblem, start: CommonStart) -> float | None:
    """``limits/absgap`` of the exact methods (module docstring); ``None`` without a usable start."""
    if not math.isfinite(start.objective):
        return None
    scale = reference_scale(problem, start.weights)
    return float(solver["abs_gap_rel"]) * float(solver["abs_gap_safety"]) * scale


def _no_start() -> CommonStart:
    """The "start" of a cold run: none, at time 0. An infinite objective is how the methods spell a start that
    failed (they do not use it and spend its time, which here is none)."""
    return CommonStart(weights=np.zeros(0), objective=math.inf, subset=(), status="none", time_s=0.0, cpu_s=0.0,
                       relax_s=0.0, refine_s=0.0)


def _screening_limits(f_start: float, s_start: float, screening: Mapping) -> dict:
    """SCIP's ``limits/primal`` and ``limits/dual`` of a screening run at the level ``tau`` (module docstring):
    stop at a solution better than the start by τ·s_start·(1 + margin), or at a dual bound within
    τ·s_start·(1 − margin) of it."""
    threshold = screening["tau"] * s_start
    margin = screening["margin"]
    return {"limits/dual": f_start - threshold * (1.0 - margin),
            "limits/primal": f_start - threshold * (1.0 + margin)}


def _run_method(
    spec: Mapping, problem: PortfolioProblem, k: int, card_mode: str, start: CommonStart, cell: Mapping,
    seed: int, solver: Mapping, screening: Mapping | None = None, start_scale: float | None = None,
) -> Solution:
    name = spec["name"]
    budget, limit = cell["budget_s"], cell.get("time_limit_s")
    if name in ("sa", "tabu", "swap_sa"):
        sampler = _anytime_sampler(spec, problem, k, seed)
        sol = run_anytime(problem, k, sampler, budget_s=budget, seed=seed, m_b=spec["m_b"], start=start,
                          card_mode=card_mode)
        sol.solver_meta["beta_mode"] = spec.get("beta")
        if isinstance(sampler, _SAObjectiveBeta) and sampler.beta_range is not None:
            sol.solver_meta["beta_hot"], sol.solver_meta["beta_cold"] = sampler.beta_range
        return sol
    if name == "ls":
        return local_search(problem, k, budget_s=budget, seed=seed, improvement=spec["improvement"],
                            num_restarts=spec["num_restarts"], rcl_size=spec["rcl_size"], start=start,
                            card_mode=card_mode)
    usable = math.isfinite(start.objective)
    abs_gap = _abs_gap(solver, problem, start)
    if name == "scip":
        params = {} if spec["usesymmetry"] == "default" else {"misc/usesymmetry": int(spec["usesymmetry"])}
        if screening is not None and screening["certificate"] == "tau" and usable:
            params.update(_screening_limits(start.objective, start_scale, screening))
            abs_gap = None  # the stop is the limits above, not the gap
        return solve_miqp_native(
            problem, k, formulation=spec["formulation"], diag=spec.get("diag") or "mineig",
            card_mode=card_mode, time_limit_s=limit, budget_s=budget, clock_offset_s=start.time_s,
            mip_gap=float(solver["mip_gap"]), abs_gap=abs_gap, seed=seed, permutation_seed=seed,
            warm_start=start.weights if usable else None, scip_params=params,
            heuristics=spec.get("heuristics", "default"))
    return solve_oa(
        problem, k, card_mode=card_mode, diag=spec["diag"], time_limit_s=limit, budget_s=budget,
        clock_offset_s=start.time_s, mip_gap=float(solver["mip_gap"]),
        abs_gap=abs_gap if abs_gap is not None else 0.0, seed=seed,
        warm_start=list(start.subset) if usable else None)


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------


def _clean(value):
    """Python natives only, ``None`` for NaN and infinities: the row goes to JSON."""
    if isinstance(value, dict):
        return {key: _clean(v) for key, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def budget_compliance(end_s: float, budget_s: float, tol_rel: float) -> tuple[float, bool]:
    """``(overrun_s, ok)``: seconds by which the run ended after its budget (0 if it did not) and whether
    that is within ``tol_rel`` of the budget. Exact solvers can overrun: the build of an SDP and the
    presolve of SCIP do not look at the clock."""
    overrun = max(0.0, float(end_s) - float(budget_s))
    return overrun, overrun <= tol_rel * float(budget_s)


def _improvements(raw, start: CommonStart, tol: float) -> list[list[float]]:
    """The entries of ``raw`` (``(t, f)`` in order of time) that improve the running best, starting from
    the common start, by more than ``tol``: the method's own improvements over the start."""
    best = start.objective if math.isfinite(start.objective) else math.inf
    out: list[list[float]] = []
    for t, f in raw:
        if f < best - tol:
            out.append([float(t), float(f)])
            best = float(f)
    return out


def _incumbent_at(start: CommonStart, own: list[list[float]], t: float) -> float | None:
    best = None
    if math.isfinite(start.objective) and start.time_s <= t:
        best = start.objective
    for time_s, f in own:
        if time_s <= t:
            best = f
    return best


def _detail(meta: Mapping) -> dict:
    detail = {k: v for k, v in meta.items() if k not in _SERIES}
    primal, dual = meta.get("primal_bound"), meta.get("dual_bound")
    if primal is not None and dual is not None and math.isfinite(primal) and math.isfinite(dual):
        detail["abs_gap_achieved"] = abs(float(primal) - float(dual))
    if "abs_gap_limit" not in detail and "abs_gap" in detail and meta.get("formulation") == "oa_multitree":
        detail["abs_gap_limit"] = detail["abs_gap"]  # OA's abs_gap is the input, not the achieved gap
    return detail


def _time_parts(name: str, meta: Mapping) -> tuple[float, float, float]:
    """``(build_s, solve_s, refine_s)`` of a run, R10: building (the QUBO and its temperature, the model of
    SCIP with its diagonal and warm start, the master of OA), the search proper (the sampler and the
    decoding of its output; SCIP's solve; OA's master solves; for LS the bookkeeping of the search outside
    the refinements) and the shared refinement (the QP on a support: the pool of a heuristic, the
    subproblems of OA, the swaps of LS; none inside SCIP)."""
    if name in ("sa", "tabu", "swap_sa"):
        return float(meta["build_s"]), float(meta["sample_s"] + meta["decode_s"]), float(meta["refine_s"])
    if name == "ls":
        return 0.0, max(0.0, float(meta["overhead_s"])), float(meta["refine_s"])
    if name == "scip":
        return float(meta["build_time_s"]), float(meta["solve_time_s"]), 0.0
    return float(meta["build_time_s"]), float(meta["master_time_s"]), float(meta["refine_time_s"])


def _exact_fields(spec: Mapping, sol: Solution, scale: float | None, offset: float) -> dict:
    meta = sol.solver_meta
    status = meta["status"]
    proved = status in ("optimal", "gaplimit")
    certified = False
    if scale is not None and math.isfinite(sol.objective):
        certified = reference_from_exact(meta, scale=scale)[1] == "certified"
    if spec["name"] == "scip":
        t_cert = offset + meta["build_time_s"] + meta["solve_time_s"] if proved else None
    else:
        t_cert = meta.get("t_cert_s")
    return {"proved": bool(proved), "certified": bool(certified), "t_cert_s": t_cert}


def _warm_up(fixed: Mapping | None = None) -> None:
    """Run every solver once, briefly, on a tiny instance: the first call of a process pays for lazy
    loading and allocation, which would otherwise land inside the first timed run."""
    global _warmed_up
    if _warmed_up:
        return
    _warmed_up = True
    from cif.instances.factor import factor_instance

    mu, sigma, _ = factor_instance(16, 1000)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(16)),
                               risk_aversion=5.0, w_min=0.0, w_max=1.0, cardinality=4)
    start = common_start(problem, 4)
    for name in ("sa", "tabu", "swap_sa"):
        run_anytime(problem, 4, _anytime_sampler(
            {"sa": {"name": "sa", "num_reads_b": 5, "num_sweeps": 20, "penalty": 10.0, "beta": "default"},
             "tabu": {"name": "tabu", "num_restarts_b": 1, "num_reads_b": 2, "tenure": 5, "penalty": 10.0},
             "swap_sa": {"name": "swap_sa", "num_reads_b": 5, "num_sweeps": 10}}[name], problem, 4, 1),
            budget_s=0.1, seed=1, m_b=5, start=start, max_batches=1)
    local_search(problem, 4, budget_s=0.1, seed=1, num_restarts=1, start=start)
    for form in ("bigm", "perspective"):
        solve_miqp_native(problem, 4, formulation=form, budget_s=0.5, warm_start=start.weights)
    solve_oa(problem, 4, budget_s=0.5, warm_start=list(start.subset))


def clear_caches() -> None:
    """Forget the cached instances (tests; a process that changes configs)."""
    e2_instances.clear_caches()


def _instance_fields(inst: e2_instances.Instance) -> dict:
    """What every row says about its instance (diagnostics of §3: ``s_cont``, ``binding``, the condition
    number of Σ, the coefficients of variation of the volatilities and of μ)."""
    problem = inst.problem
    return {
        "instance_id": inst.instance_id,
        "n_assets": problem.n,
        "K": inst.k,
        "card_mode": inst.card_mode,
        "lam": problem.risk_aversion,
        "w_min": problem.w_min,
        "w_max": problem.w_max,
        **{key: value for key, value in inst.stats.items() if key != "K"},
        "instance": inst.meta,
    }


def _instance_spec(cell: Mapping) -> dict:
    keys = ("family", "n", "seed", "port", "lam_index")
    return {k: cell[k] for k in keys if k in cell}


def _eligibility_fields(
    screening: Mapping, start: CommonStart, start_scale: float | None, sol: Solution, scale: float | None,
    own: list[list[float]], status: str,
) -> dict:
    """The class of the instance and the evidence it follows from (module docstring,
    :func:`cif.experiments.e2_eligibility.classify`), and the time of the certificate of the trivial side."""
    meta = sol.solver_meta
    tau = screening["tau"]
    f_best = min(start.objective, sol.objective) if math.isfinite(sol.objective) else start.objective
    verdict = e2_eligibility.classify(
        f_start=start.objective, s_start=start_scale if start_scale is not None else math.nan, f_best=f_best,
        dual_bound=meta["dual_bound"], tau=tau, status=status)
    met = status in ("optimal", "gaplimit", "duallimit")  # the stopping rule of the trivial side was met
    t_cert = start.time_s + meta["build_time_s"] + meta["solve_time_s"] if met else None
    t_decide = None
    if verdict.label == "nontrivial":
        t_decide = next((t for t, f in own if start.objective - f > verdict.threshold), None)
    elif verdict.label == "trivial":
        t_decide = t_cert
    scip_params = meta.get("scip_params") or {}
    evidence = {
        "certificate": screening["certificate"], "tau": tau,
        "margin": screening["margin"] if screening["certificate"] == "tau" else None,
        "threshold": verdict.threshold, "f_start": start.objective, "s_start": start_scale, "f_best": f_best,
        "s_best": scale, "dual_bound": meta["dual_bound"], "primal_bound": meta["primal_bound"],
        "gap_lo": verdict.gap_lo, "gap_hi": verdict.gap_hi, "reason": verdict.reason, "status": status,
        "t_decide_s": t_decide, "limit_dual": scip_params.get("limits/dual"),
        "limit_primal": scip_params.get("limits/primal"),
    }
    return {"eligibility": verdict.label, "evidence": evidence, "t_cert_s": t_cert}


def run_cell(cell: dict, cfg: dict) -> dict:
    """The row of one cell (module docstring)."""
    t_cell = time.perf_counter()
    p = cfg["params"]
    if p.get("warm_up", True):
        _warm_up()
    fixed = dict(p.get("fixed") or {})
    if cell.get("boot_lam") is not None:
        fixed["boot_lam"] = cell["boot_lam"]
    solver = {"mip_gap": 1e-6, "abs_gap_rel": 1e-6, "abs_gap_safety": 0.5, "trajectory_tol": _DEFAULT_TRAJECTORY_TOL,
              **(p.get("solver") or {})}
    inst = e2_instances.build_instance(_instance_spec(cell), fixed)
    spec = e2_space.check_method(dict(cell["method"]))
    problem, k, card_mode = inst.problem, inst.k, inst.card_mode
    seed = derive_seed(cell)
    if cell.get("skip_nonbinding") and not inst.binding:
        # RQ2 compares on instances where the cardinality constraint binds (R4): no method is run
        return _clean({**_instance_fields(inst), "config_id": spec["id"], "status": "skipped",
                       "skipped": "not_binding", "cell_s": time.perf_counter() - t_cell})

    stage = cell["stage"]
    cold = stage == "cold"
    screening = _eligibility_params(p) if stage == "eligibility" else None
    start = _no_start() if cold else common_start(problem, k, card_mode=card_mode)
    start_ok = math.isfinite(start.objective)
    start_scale = reference_scale(problem, start.weights) if start_ok else None
    if screening is not None and not start_ok:
        # no start, nothing to compare with: the instance is undecided and no solver is run on it
        verdict = e2_eligibility.classify(f_start=math.inf, s_start=math.nan, f_best=None, dual_bound=None,
                                          tau=screening["tau"], status="no_start")
        return _clean({**_instance_fields(inst), "config_id": spec["id"], "status": "no_start",
                       "eligibility": verdict.label, "start_status": start.status, "start_f": None,
                       "start_time_s": start.time_s,
                       "evidence": {"certificate": screening["certificate"], "tau": screening["tau"],
                                    "f_start": None, "s_start": None, "f_best": None, "dual_bound": None,
                                    "gap_lo": 0.0, "gap_hi": None, "reason": verdict.reason, "status": "no_start"},
                       "t_cert_s": None, "cell_s": time.perf_counter() - t_cell})
    wall0, cpu0 = time.perf_counter(), time.process_time()
    sol = _run_method(spec, problem, k, card_mode, start, cell, seed, solver, screening=screening,
                      start_scale=start_scale)
    method_wall, method_cpu = time.perf_counter() - wall0, time.process_time() - cpu0

    meta = sol.solver_meta
    has_solution = math.isfinite(sol.objective)
    scale = reference_scale(problem, sol.weights) if has_solution and sol.weights.sum() > 0 else None
    support = np.flatnonzero(sol.weights > 1e-6) if has_solution else np.empty(0, dtype=int)
    tol = float(solver["trajectory_tol"]) * (start_scale if start_scale is not None else (scale or 1.0))
    raw = meta["incumbent_trajectory"] if spec["name"] == "oa" else meta["trajectory"]
    own = _improvements(raw, start, tol)
    exact = spec["name"] in EXACT_METHODS
    budget = cell["budget_s"]

    if exact:
        first_own = own[0][0] if own else None
        first_own_f = own[0][1] if own else None
        status = meta["status"]
    else:
        first_own = meta.get("first_own_s")
        first_own_f = meta.get("first_own_f")
        status = meta["stop"]
    overrun, ok = (None, None) if budget is None else budget_compliance(
        sol.wall_time_s, budget, float(p.get("overrun_tol_rel", 0.05)))
    row: dict[str, Any] = {
        **_instance_fields(inst),
        "config_id": spec["id"],
        "threads": 1,
        "seed_used": seed,
        "start_time_s": None if cold else start.time_s,
        "start_cpu_s": None if cold else start.cpu_s,
        "start_relax_s": None if cold else start.relax_s,
        "start_refine_s": None if cold else start.refine_s,
        "start_f": start.objective,  # inf (a failed start, or none in a cold run) is written as null
        "start_scale": start_scale,
        "start_status": None if cold else start.status,
        "start_subset": None if cold else list(start.subset),
        "status": status,
        "f_final": sol.objective,
        "scale_final": scale,
        "feasible": bool(sol.feasible),
        "support": [int(i) for i in support],
        "weights": [float(w) for w in sol.weights[support]],
        "end_s": sol.wall_time_s,
        "overrun_s": overrun,
        "budget_ok": ok,
        "method_wall_s": method_wall,
        "method_cpu_s": method_cpu,
        **dict(zip(("build_s", "solve_s", "refine_s"), _time_parts(spec["name"], meta), strict=True)),
        "trajectory": own,
        "first_own_s": first_own,
        "first_own_f": first_own_f,
        "t_first_improve_s": own[0][0] if own else None,
        "t_primal_s": own[-1][0] if own else (start.time_s if start_ok else None),
        "f_at_budget": _incumbent_at(start, own, budget) if budget is not None else None,
        "t_cert_s": None,
        "detail": _detail(meta),
        "cell_s": None,
    }
    if cold:
        row["cold"] = True
    if exact:
        row.update(_exact_fields(spec, sol, scale, start.time_s))
    if screening is not None:
        row.update(_eligibility_fields(screening, start, start_scale, sol, scale, own, status))
    if stage == "reference" or (screening is not None and screening["certificate"] == "full"):
        row.update(_reference_fields(problem, card_mode, sol, scale, row))
    row["cell_s"] = time.perf_counter() - t_cell
    return _clean(row)


def _reference_fields(problem: PortfolioProblem, card_mode: str, sol: Solution, scale: float | None,
                      row: Mapping) -> dict:
    """``f_star`` and its certificate: the incumbent's support polished by the shared refinement (SCIP's
    weights are accurate to its feasibility tolerance, the refinement to 1e-10), as in
    :func:`cif.heuristics.oracle.certified_optimum`; the type follows
    :func:`cif.metrics.gap.reference_from_exact` on the scale of SCIP's incumbent."""
    if scale is None:
        return {"f_star": None, "scale_star": None, "ref_type": "best_known"}
    _, ref_type = reference_from_exact(sol.solver_meta, scale=scale)
    polished = refine_problem(problem, row["support"], card_mode)
    if polished.status == "optimal" and polished.objective <= sol.objective:
        f_star, weights = float(polished.objective), polished.weights
    else:
        f_star, weights = float(sol.objective), sol.weights
    return {"f_star": f_star, "scale_star": reference_scale(problem, weights), "ref_type": ref_type}
