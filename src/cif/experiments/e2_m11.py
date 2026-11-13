"""E2: validation of the exact stack against Bertsimas & Cory-Wright, Tables 3 and 4 (M11; ANALYSIS_PLAN §2 RQ2
"Валидация бейзлайна", §12 entry "Выборы валидации точного стека M11 (до прогона)", 2026-10-17).

Experiment module of the runner (:mod:`cif.experiments.run`, protocol of :mod:`cif.experiments.toy`), config
``configs/E2/m11.yaml``. The plan asks how the open exact stack of RQ2 compares with the published state of the art
on a standard problem, and rules by the ratio of the times: above 10 (median over the cells of a table) the
conclusions of RQ2 are about *open* exact methods only. The problem is (34) of the paper on the five OR-Library
files (:mod:`cif.instances.bcw`); ``n = 31, 85, 89, 98, 225`` and ``k ∈ {5, 10, 20}`` give 15 instances per table:

* **T3** (their Table 3): ``κ = 1``, no linear constraint, limit 300 s. Our methods: SCIP (``solve_miqp_native``,
  perspective with ``D = λ_min·I``, ``misc/usesymmetry = 0``) and outer approximation with HiGHS (``solve_oa``).
* **T4** (their Table 4): ``κ = 0`` and ``μᵀx ≥ r̄``, ``r̄`` from the rule of the paper, limit 3600 s. Our method:
  SCIP with the one extra row (``linear_ge``); the implementation of OA has no linear rows, so T4 is not run
  with it (the plan says so).

The methods are the exact-class representatives of the pilot: their specs are read from ``configs/E2/pilot/
tuned.json`` (``inputs.tuned``) — ``exact_rep`` of the size 100 (SCIP, ``scip-persp-mineig-sym0``; the same at 200)
and ``exact_rep`` of the size 400 (OA, ``oa-highs``) — and are part of every cell, so that the cell key moves
with them. No common start (no warm start of any kind), one thread, ``limits`` as above.

**Stopping rule.** The relative gap 1e-4 and the absolute gap 1e-6 (the defaults of CPLEX 12.8, which BCW used),
both given to the solver (``mip_gap``, ``abs_gap``); the solver stops at whichever is met first. Both solvers
minimise the objective of (34) as it is — SCIP's objective is ``−κμᵀw + ½·risk_scale·q`` with the risk row
divided by ``risk_scale`` (so ``q`` is rescaled, the objective is not), OA works on ``(μ, Σ)/mean(diag Σ)`` and its
gap is a ratio (scale-free) — and the row reports the gap of the *true* objective: ``primal`` is
:func:`cif.instances.bcw.bcw_objective` of the returned weights (the formula of the paper, raw ``Σ`` and the ridge
as ``‖x‖²/2γ``, not the solver's), ``dual`` the solver's bound. ``rel_gap = |p − d|/min(|p|, |d|)`` is SCIP's
definition, the stopping rule of both solvers (``None``: undefined, a sign change between the bounds, as SCIP's
infinite gap); ``rel_gap_cplex = |p − d|/(1e-10 + |p|)`` is CPLEX's. Since ``min(|p|, |d|) ≤ |p|`` the first is never
smaller than the second: stopping on it is at least as strict as CPLEX's own rule.

**Time.** The wall clock of the call (``wall_s``: build + solve + extraction; the instance, ``r̄`` and the warm-up
are outside), as the timings of the other stages (the call is made with ``budget_s`` = the limit, the end-to-end
budget of the solvers, so SCIP and the master of OA get what is left after the build). ``certified`` is ``proved``
(SCIP ``optimal`` / ``gaplimit``, OA ``optimal``: the gap is closed) *and* ``wall_s`` within the limit. A call that
overruns its limit by more than ``overrun_tol_rel`` has ``budget_ok = false``.

**Seeds.** ``seed_role: calibration``, solver seeds 10000-10002 for SCIP (the median of three permutations of the
model is the value of a cell: ``randomization/permutevars`` and ``permutationseed`` follow the one derived seed, as in
:mod:`cif.experiments.e2_benchmark`) and 10000 for OA (one run: HiGHS is deterministic given its ``random_seed``).
The role is the one of the plan for runs that test nothing: the instances are fixed benchmark files (no instance
seeds), the rows enter no hypothesis test of §7 and no confirmatory estimate, so the test streams (20000-29999,
reserved for the confirmatory stages) stay unspent, and the rows are a validation with a reported consequence (the
ratio against the published times), not a test seed set. ``seed`` of a cell is the solver seed; the stream of the
run is ``derive_seed`` (31 bit, from the seed and every other coordinate of the cell).

**Cell** = ``stage, table, port, n, k, kappa, return_row, method (spec), seed, time_limit_s``; nothing computed from
data (``γ``, ``r̄``, the objective) is in it. **Row** = the cell and ``instance_id, n_assets, gamma, r_bar`` (T4;
``r_min``, ``r_max`` next to it), ``config_id, threads, seed_used``, the result (``status``, ``proved``, ``certified``,
``feasible``, ``support``, ``weights``, ``n_nonzero``, ``r_realised``, ``return_ok``), the bounds (``primal``, ``dual``,
``abs_gap``, ``rel_gap``, ``rel_gap_cplex``), the effort (``nodes`` for SCIP; ``n_cuts``, ``iterations`` for OA),
the times (``wall_s``, ``end_s``, ``cpu_s``, ``build_s``, ``solve_s``, ``refine_s``, ``overrun_s``, ``budget_ok``) and
``detail`` (the solver's diagnostics without the long series). The analysis is :mod:`cif.experiments.e2_m11_analysis`
(``e2_analysis m11``).

Config (YAML) ``params``::

    stage: m11,  seed_role: calibration | test,  warm_up: true,  overrun_tol_rel: 0.05
    orlib_dir: data/raw/orlib                       # relative to the repository root
    ports: [1, 2, 3, 4, 5],  ks: [5, 10, 20]
    solver: {mip_gap: 1e-4, abs_gap: 1e-6}
    tables:  {T3: {kappa: 1.0, time_limit_s: 300.0, methods: [scip, oa]},
              T4: {kappa: 0.0, time_limit_s: 3600.0, methods: [scip], return_row: true}}
    methods: {scip: {size: 100, role: exact_rep}, oa: {size: 400, role: exact_rep}}     # from inputs.tuned
    solver_seeds: {scip: {first: 10000, count: 3}, oa: {first: 10000, count: 1}}        # or lists

The top-level ``seeds``, if given, must be the seeds of the grid (the manifest records them).
"""

from __future__ import annotations

import functools
import json
import math
import time
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cif.classical.oa import solve_oa
from cif.classical.scip_native import solve_miqp_native
from cif.experiments import e2_space, registry
from cif.experiments.e2_benchmark import (
    SEED_RANGES,
    TUNED_KIND,
    _clean,
    _detail,
    _expand_seeds,
    _report,
    _time_parts,
    budget_compliance,
)
from cif.instances import bcw
from cif.instances.orlib import ORLIB_N
from cif.problem import PortfolioProblem, Solution

__all__ = ["METHODS", "STAGE", "TABLES", "cells", "clear_caches", "derive_seed", "run_cell"]

STAGE = "m11"
TABLES = ("T3", "T4")
METHODS = ("scip", "oa")
_TABLE_CODE = {"T3": 3, "T4": 4}
_METHOD_CODE = {"scip": 5, "oa": 6}  # as in e2_benchmark
_STREAM_TAG = 0x4D3131  # "M11": the streams of this stage are not those of any other stage
_TABLE_KEYS = {"kappa", "time_limit_s", "methods", "return_row"}
#: The status of a solver run that closed the gap (SCIP: ``optimal`` or ``gaplimit``; OA: ``optimal``).
_PROVED = ("optimal", "gaplimit")
_CPLEX_GAP_FLOOR = 1e-10  # the 1e-10 of CPLEX's relative MIP gap |bound − incumbent|/(1e-10 + |incumbent|)

_warmed_up = False


# --------------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------------


def _positive(value: Any, where: str) -> float:
    if not (isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value > 0):
        raise ValueError(f"{where} must be a positive number, got {value!r}")
    return float(value)


def _ports(p: Mapping) -> list[int]:
    ports = p.get("ports")
    if not isinstance(ports, list | tuple) or not ports:
        raise ValueError("params.ports must list at least one OR-Library port")
    for port in ports:
        if port not in ORLIB_N:
            raise ValueError(f"params.ports: port {port!r} is not an OR-Library port {sorted(ORLIB_N)}")
    return [int(port) for port in ports]


def _ks(p: Mapping, ports: list[int]) -> list[int]:
    ks = p.get("ks")
    if not isinstance(ks, list | tuple) or not ks:
        raise ValueError("params.ks must list at least one cardinality")
    smallest = min(ORLIB_N[port] for port in ports)
    for k in ks:
        if not (isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= smallest):
            raise ValueError(f"params.ks: k = {k!r} must be an integer in [1, {smallest}] (the smallest port)")
    return [int(k) for k in ks]


def _solver(p: Mapping) -> dict:
    solver = p.get("solver") or {}
    mip_gap = solver.get("mip_gap")
    abs_gap = solver.get("abs_gap")
    if not (isinstance(mip_gap, int | float) and not isinstance(mip_gap, bool) and 0.0 < mip_gap < 1.0):
        raise ValueError(f"params.solver.mip_gap must lie in (0, 1), got {mip_gap!r}")
    if not (isinstance(abs_gap, int | float) and not isinstance(abs_gap, bool) and math.isfinite(abs_gap)
            and abs_gap >= 0.0):
        raise ValueError(f"params.solver.abs_gap must be a non-negative number, got {abs_gap!r}")
    return {"mip_gap": float(mip_gap), "abs_gap": float(abs_gap)}


def _tables(p: Mapping) -> dict[str, dict]:
    tables = p.get("tables")
    if not isinstance(tables, Mapping) or not tables:
        raise ValueError("params.tables must define at least one table")
    out: dict[str, dict] = {}
    for name, spec in tables.items():
        if name not in TABLES:
            raise ValueError(f"params.tables: unknown table {name!r}, expected one of {TABLES}")
        extra = set(spec) - _TABLE_KEYS
        if extra:
            raise ValueError(f"params.tables.{name}: unknown keys {sorted(extra)}")
        kappa = spec.get("kappa")
        if not (isinstance(kappa, int | float) and not isinstance(kappa, bool) and math.isfinite(kappa)):
            raise ValueError(f"params.tables.{name}.kappa must be a number, got {kappa!r}")
        methods = spec.get("methods")
        if not isinstance(methods, list | tuple) or not methods or not set(methods) <= set(METHODS):
            raise ValueError(f"params.tables.{name}.methods must be a non-empty list of methods from {METHODS}, "
                             f"got {methods!r}")
        return_row = bool(spec.get("return_row", False))
        if return_row and "oa" in methods:
            raise ValueError(f"params.tables.{name}: the return row is not available to OA (its implementation has "
                             "no linear rows; the plan runs the table with SCIP only)")
        out[name] = {"kappa": float(kappa), "time_limit_s": _positive(spec.get("time_limit_s"),
                                                                      f"params.tables.{name}.time_limit_s"),
                     "methods": list(dict.fromkeys(methods)), "return_row": return_row}
    return out


def _method_spec(cfg: Mapping, p: Mapping, name: str) -> dict:
    """The spec of ``name`` from the tuned report: ``params.methods.<name>`` = ``{size, role}``."""
    wanted = (p.get("methods") or {}).get(name)
    if not isinstance(wanted, Mapping) or wanted.get("size") is None or not wanted.get("role"):
        raise ValueError(f"params.methods.{name} must give the 'size' and the 'role' of its configuration in the "
                         "tuned report")
    tuned = _report(cfg, "tuned", TUNED_KIND)
    entry = (tuned.get("sizes") or {}).get(str(wanted["size"]))
    if entry is None:
        raise ValueError(f"the tuned report has no configurations for size {wanted['size']}")
    spec = (entry.get("roles") or {}).get(wanted["role"])
    if spec is None:
        raise ValueError(f"the tuned report has no role {wanted['role']!r} for size {wanted['size']}")
    spec = e2_space.check_method(dict(spec))
    if spec["name"] != name:
        raise ValueError(f"params.methods.{name}: the tuned report gives {spec['id']!r} ({spec['name']}) for size "
                         f"{wanted['size']}, role {wanted['role']!r}, not a {name} method")
    return spec


def _seeds_of(p: Mapping, name: str) -> list[int]:
    seeds = (p.get("solver_seeds") or {}).get(name)
    if seeds is None or (not isinstance(seeds, Mapping) and not len(seeds)):
        raise ValueError(f"params.solver_seeds has no seeds for the method {name!r}")
    return _expand_seeds(seeds, f"params.solver_seeds.{name}")


def _check_seeds(cfg: Mapping, role: str, cells_: list[dict]) -> None:
    lo, hi = SEED_RANGES[role][1]
    seeds = sorted({c["seed"] for c in cells_})
    bad = [s for s in seeds if not lo <= s < hi]
    if bad:
        raise ValueError(f"seed_role={role!r} takes solver seeds in [{lo}, {hi}); out of range: seeds {bad}")
    declared = cfg.get("seeds")
    if declared is not None and sorted({int(s) for s in declared}) != seeds:
        raise ValueError(f"the top-level seeds {sorted(declared)} (recorded in the manifest) must be the seeds of "
                         f"the grid {seeds}")


def cells(cfg: dict) -> list[dict]:
    """The grid, instance by instance: tables × ports × k × methods × solver seeds (module docstring)."""
    p = cfg.get("params")
    if not p:
        raise ValueError("config: missing 'params'")
    if p.get("stage") != STAGE:
        raise ValueError(f"params.stage must be {STAGE!r}, got {p.get('stage')!r}")
    role = p.get("seed_role")
    if role not in SEED_RANGES:
        raise ValueError(f"params.seed_role must be one of {sorted(SEED_RANGES)}, got {role!r}")
    if cfg.get("mode") == "smoke" and role == "test":
        raise ValueError("a smoke run uses calibration seeds only: seed_role 'test' is refused in smoke mode")
    ports, tables = _ports(p), _tables(p)
    ks = _ks(p, ports)
    _solver(p)
    used = [m for m in METHODS if any(m in t["methods"] for t in tables.values())]
    specs = {name: _method_spec(cfg, p, name) for name in used}
    seeds = {name: _seeds_of(p, name) for name in used}
    out = []
    for table, t in tables.items():
        for port in ports:
            for k in ks:
                for method in t["methods"]:
                    for seed in seeds[method]:
                        out.append({"stage": STAGE, "table": table, "port": port, "n": ORLIB_N[port], "k": k,
                                    "kappa": t["kappa"], "return_row": t["return_row"], "method": specs[method],
                                    "seed": seed, "time_limit_s": t["time_limit_s"]})
    _check_seeds(cfg, role, out)
    return out


def derive_seed(cell: Mapping) -> int:
    """The 31-bit seed of the run: ``SeedSequence`` entropy from the solver seed and every other coordinate that
    identifies the stream (table, port, k, method and its spec) but not the limit, so that a longer run extends a
    shorter one of the same cell (§3: streams from the seed and indices)."""
    spec = cell["method"]
    spec_hash = zlib.crc32(json.dumps(spec, sort_keys=True).encode("utf-8"))
    entropy = [int(cell["seed"]), _STREAM_TAG, _TABLE_CODE[cell["table"]], int(cell["port"]), int(cell["k"]),
               _METHOD_CODE[spec["name"]], spec_hash]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF


# --------------------------------------------------------------------------------------------
# A run
# --------------------------------------------------------------------------------------------


def _data_dir(p: Mapping) -> str:
    """``params.orlib_dir``: a relative path is the repository's, whatever the working directory of the job."""
    if not p.get("orlib_dir"):
        raise ValueError("params.orlib_dir is required (the directory of port1.txt ... port5.txt)")
    path = Path(str(p["orlib_dir"]))
    return str(path if path.is_absolute() else registry.REPO_ROOT / path)


@functools.lru_cache(maxsize=8)
def _instance(port: int, data_dir: str) -> bcw.BcwInstance:
    """The data of (34) on ``port<k>`` with ``r̄``, built once per process (the cells of one instance follow each
    other)."""
    return bcw.bcw_instance(port, data_dir)


def clear_caches() -> None:
    """Forget the cached instances (tests; a process that changes configs)."""
    _instance.cache_clear()


def _warm_up() -> None:
    """Run both solvers once, briefly, on a tiny instance, with the return row: the first call of a process pays
    for lazy loading and allocation, which would otherwise land inside the first timed run."""
    global _warmed_up
    if _warmed_up:
        return
    _warmed_up = True
    from cif.instances.factor import factor_instance

    mu, sigma, _ = factor_instance(16, 1000)
    problem = PortfolioProblem(mu=np.zeros(16), sigma=sigma, asset_names=tuple(f"a{i}" for i in range(16)),
                               risk_aversion=1.0, cardinality=4)
    solve_miqp_native(problem, 4, budget_s=0.5, scip_params={"misc/usesymmetry": 0},
                      linear_ge=(mu, float(np.median(mu))))
    solve_oa(PortfolioProblem(mu=mu, sigma=sigma, asset_names=problem.asset_names, risk_aversion=1.0,
                              cardinality=4), 4, budget_s=0.5)


def _solve(spec: Mapping, problem: PortfolioProblem, cell: Mapping, seed: int, solver: Mapping,
           linear_ge: tuple[np.ndarray, float] | None) -> Solution:
    """One call of the method of the cell, no warm start, ``budget_s`` = the limit, the stopping rule of the
    config (both gaps)."""
    k, limit = int(cell["k"]), float(cell["time_limit_s"])
    if spec["name"] == "scip":
        params = {} if spec["usesymmetry"] == "default" else {"misc/usesymmetry": int(spec["usesymmetry"])}
        return solve_miqp_native(
            problem, k, formulation=spec["formulation"], diag=spec.get("diag") or "mineig", card_mode="le",
            budget_s=limit, mip_gap=solver["mip_gap"], abs_gap=solver["abs_gap"], seed=seed, permutation_seed=seed,
            scip_params=params, linear_ge=linear_ge)
    if linear_ge is not None:
        raise ValueError("the return row is not available to OA (the implementation has no linear rows)")
    return solve_oa(problem, k, card_mode="le", diag=spec["diag"], budget_s=limit, mip_gap=solver["mip_gap"],
                    abs_gap=solver["abs_gap"], seed=seed)


def _finite(x) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _gaps(primal: float | None, dual: float | None) -> tuple[float | None, float | None, float | None]:
    """``(|p − d|, SCIP's relative gap, CPLEX's relative gap)`` of the bounds of the true objective; ``None`` where
    a bound is missing. SCIP's ``|p − d|/min(|p|, |d|)`` is ``None`` when the bounds differ in sign or one is zero
    (SCIP: infinite); equal bounds give 0."""
    if primal is None or dual is None:
        return None, None, None
    absolute = abs(primal - dual)
    if primal == dual:
        scip = 0.0
    elif primal * dual <= 0.0:
        scip = None
    else:
        scip = absolute / min(abs(primal), abs(dual))
    return absolute, scip, absolute / (_CPLEX_GAP_FLOOR + abs(primal))


def run_cell(cell: dict, cfg: dict) -> dict:
    """The row of one cell (module docstring)."""
    t_cell = time.perf_counter()
    p = cfg["params"]
    if p.get("warm_up", True):
        _warm_up()
    solver = _solver(p)
    inst = _instance(int(cell["port"]), _data_dir(p))
    spec = e2_space.check_method(dict(cell["method"]))
    k, kappa, return_row = int(cell["k"]), float(cell["kappa"]), bool(cell["return_row"])
    problem = bcw.bcw_problem(inst, k, kappa)
    linear_ge = (inst.mu, inst.r_bar) if return_row else None
    seed = derive_seed(cell)
    limit = float(cell["time_limit_s"])

    wall0, cpu0 = time.perf_counter(), time.process_time()
    sol = _solve(spec, problem, cell, seed, solver, linear_ge)
    wall_s, cpu_s = time.perf_counter() - wall0, time.process_time() - cpu0

    meta = sol.solver_meta
    has_solution = math.isfinite(sol.objective)
    x = sol.weights
    support = np.flatnonzero(x > 1e-6) if has_solution else np.empty(0, dtype=int)
    primal = bcw.bcw_objective(inst, x, kappa) if has_solution else None
    dual = _finite(meta.get("dual_bound"))
    absolute, rel_gap, rel_gap_cplex = _gaps(primal, dual)
    status = meta["status"]
    proved = status in _PROVED
    overrun, within = budget_compliance(sol.wall_time_s, limit, float(p.get("overrun_tol_rel", 0.05)))
    r_realised = float(inst.mu @ x) if has_solution else None
    build_s, solve_s, refine_s = _time_parts(spec["name"], meta)
    row: dict[str, Any] = {
        "instance_id": inst.instance_id,
        "n_assets": inst.n,
        "gamma": inst.gamma,
        "r_bar": inst.r_bar if return_row else None,
        "r_min": inst.threshold.r_min if return_row else None,
        "r_max": inst.threshold.r_max if return_row else None,
        "config_id": spec["id"],
        "threads": 1,
        "seed_used": seed,
        "status": status,
        "proved": bool(proved),
        "certified": bool(proved and wall_s <= limit),
        "feasible": bool(sol.feasible),
        "support": [int(i) for i in support],
        "weights": [float(w) for w in x[support]],
        "n_nonzero": int(support.size),
        "r_realised": r_realised,
        "return_ok": bool(meta["linear_ok"]) if return_row and has_solution else None,
        "primal": primal,
        "dual": dual,
        "abs_gap": absolute,
        "rel_gap": rel_gap,
        "rel_gap_cplex": rel_gap_cplex,
        "nodes": int(meta["n_nodes"]) if spec["name"] == "scip" else None,
        "n_cuts": int(meta["n_cuts"]) if spec["name"] == "oa" else None,
        "iterations": int(meta["iterations"]) if spec["name"] == "oa" else None,
        "wall_s": wall_s,
        "end_s": sol.wall_time_s,
        "cpu_s": cpu_s,
        "build_s": build_s,
        "solve_s": solve_s,
        "refine_s": refine_s,
        "overrun_s": overrun,
        "budget_ok": within,
        "detail": _detail(meta),
        "cell_s": time.perf_counter() - t_cell,
    }
    return _clean(row)
