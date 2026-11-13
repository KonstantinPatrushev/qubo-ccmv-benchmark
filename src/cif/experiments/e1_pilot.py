"""E1 pilot: the work of a sampler call and the floor rule of H1a (ANALYSIS_PLAN §2, §6, §12).

Experiment module of the runner (:mod:`cif.experiments.run`) plus a report tool. The pilot runs on
**calibration** instances only (pairs of seeds 1000-1999, solvers 10000-19999, §3) and fixes two
things before the confirmatory run, both written to §12 of the plan:

``floor`` — the rule of the floor for H1a (§2). On the 20 calibration pairs per N (80 instances) the
ceiling ``E_form(m)`` of the surrogate is computed exactly; ``m0 = 50`` if at least 20 % of the
instances have ``E_form(50) > τ`` (``τ = 1e-3``), else ``m0 = 1``. One ``m0`` for the whole design.

``timing`` — the work of one sampler call (§6). The call of the confirmatory run has a fixed work
(reads × sweeps for SA and swap-SA, reads for tabu: §12 B1, each read one plain tabu search with
``num_restarts = 0``); the work is chosen per sampler and N so that
the median time of one call on the calibration instances of that N is about ``target_s`` (1 s). A
cell of this part is one candidate work: ``repeats`` calls on every calibration instance of the N
(R10: at least 3), the call time being the ``solve_s`` that the attribution itself records
(:func:`cif.experiments.e1_attribution._draw_pool`, the same BQM, sampler and seeding). The
statistic is the median over the instances of the median over the repeats. The candidates are a
small explicit grid (a list, or a geometric progression ``start · ratio^k``); the whole grid is
measured and the work whose median is closest to the target **on the log scale** is chosen, a tie
going to the smaller work (:func:`choose_work`): a deterministic search, with the curve of time
against work as by-product. ``status`` says whether the target is bracketed (``ok``: within
``tolerance`` of it; ``edge_low`` / ``edge_high``: even the smallest / largest candidate is off, the
grid must be moved; ``off_target``: inside the grid but not close, the grid is too coarse or the
timing too noisy).

The *reference configuration* that is timed is the one of §12 B3 (``reference`` in the config, which must be
:data:`cif.experiments.e1_attribution.REFERENCE`): the penalty multiplier of the builder (100) and the β range
of ``dwave-samplers`` for SA, the penalty for tabu, the schedule ``T_cold = 0.01·T_hot`` for swap-SA. The
other levels of the nested factors (§7.3) run the same work and their call times are reported by the
attribution.

Rows carry the timing environment (CPU model, governor, load average, running processes, clock) at
the start and the end of a cell; the report aggregates it and warns about a governor that is not
``performance``, a machine busier than half of its physical cores (§6) and hidden threads.

Config (YAML) ``params``::

    seed_role: calibration
    parts: [floor, timing]                      # either, or both
    instances: [{n: 16, pair_seeds: [1000, ...]}, ...]    # as in e1_attribution
    h_levels, lam, w_min, w_max, ms, scip       # as in e1_attribution
    floor: {tau, m_floor, m_fallback, min_share}
    timing:
      target_s: 1.0
      tolerance: 0.2
      solver_seeds: [10000, 10001, 10002]       # the repeats
      warmup_calls: 1
      samplers:
        - name: sa                              # | tabu | swap_sa
          reference: {penalty: 100, beta: default}      # sa: penalty, beta; tabu: penalty; swap_sa: cold_ratio
          grid:
            - {n: 16, scale: num_reads, start: 300, ratio: 1.26, count: 15,
               fixed: {num_sweeps: 1000}}       # or `values: [...]` instead of start/ratio/count
                                                # tabu: scale num_reads, fixed {num_restarts: 0, tenure: max(5, N // 4)}

Command line::

    python -m cif.experiments.e1_pilot report --run runs/<floor> --run runs/<timing> --out pilot_report.json
    python -m cif.experiments.e1_pilot make-config --report pilot_report.json --config configs/E1/attribution.yaml
    python -m cif.experiments.e1_pilot env            # the timing environment as JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

from cif.experiments import e1_attribution as e1
from cif.experiments import registry, run
from cif.experiments.timing import thread_state
from cif.heuristics.oracle import pool_subsets
from cif.stats.rq1 import wilson_interval

__all__ = [
    "PARTS",
    "RunData",
    "apply_pilot_block",
    "build_report",
    "cells",
    "choose_work",
    "expand_grid",
    "floor_rule",
    "load_run",
    "main",
    "pilot_values",
    "render_pilot_block",
    "run_cell",
    "section_12_entry",
    "timing_environment",
]

PARTS = ("floor", "timing")
REPORT_SCHEMA = "e1-pilot-report/1"


# --------------------------------------------------------------------------------------------
# Config checks and the grid of candidate works
# --------------------------------------------------------------------------------------------


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _number(value, where: str, *, low: float = 0.0, high: float | None = None, closed_high=False) -> float:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    ok = ok and value > low and (high is None or value < high or (closed_high and value == high))
    if not ok:
        bound = f"in ({low:g}, {high:g}{']' if closed_high else ')'}" if high is not None else f"> {low:g}"
        raise ValueError(f"{where} must be a number {bound}, got {value!r}")
    return float(value)


def expand_grid(entry: dict) -> list[int]:
    """The candidate values of the scaled parameter: ``values`` as given, or the geometric progression
    ``round(start · ratio^k)``, ``k < count``, with the repeats that rounding creates removed.

    Strictly increasing positive integers, or ``ValueError``.
    """
    if not isinstance(entry.get("scale"), str):
        raise ValueError(f"a grid entry needs 'scale' (the name of the parameter), got {entry!r}")
    geometric = [key for key in ("start", "ratio", "count") if key in entry]
    if ("values" in entry) == bool(geometric):
        raise ValueError(f"grid entry for {entry['scale']!r}: give either 'values' or "
                         f"'start', 'ratio' and 'count'")
    if "values" in entry:
        values = entry["values"]
        if not isinstance(values, list) or not values or not all(_is_count(v) for v in values):
            raise ValueError(f"grid 'values' of {entry['scale']!r} must be a non-empty list of "
                             f"positive integers, got {values!r}")
    else:
        if len(geometric) != 3:
            raise ValueError("a geometric grid needs 'start', 'ratio' and 'count'")
        start = _number(entry["start"], "grid start")
        ratio = _number(entry["ratio"], "grid ratio", low=1.0)
        count = entry["count"]
        if not _is_count(count):
            raise ValueError(f"grid count must be a positive integer, got {count!r}")
        values = []
        for k in range(count):
            v = max(1, math.floor(start * ratio**k + 0.5))
            if not values or v > values[-1]:
                values.append(v)
    if any(b <= a for a, b in pairwise(values)):
        raise ValueError(f"grid of {entry['scale']!r} must be strictly increasing, got {values}")
    return values


def _check_floor(p: dict) -> dict:
    floor = e1._require(p, "floor", "params")
    for key in ("tau", "m_floor", "m_fallback", "min_share"):
        if floor.get(key) is None:
            raise ValueError(f"params.floor: missing {key!r}")
    _number(floor["tau"], "params.floor.tau")
    _number(floor["min_share"], "params.floor.min_share", high=1.0, closed_high=True)
    for key in ("m_floor", "m_fallback"):
        if not _is_count(floor[key]) or floor[key] not in p["ms"]:
            raise ValueError(f"params.floor.{key} must be one of the pool sizes ms {p['ms']}, "
                             f"got {floor[key]!r}")
    return floor


def _check_timing(p: dict, ns: list[int]) -> list[dict]:
    """Validate ``params.timing``; return the timing cells (see :func:`cells`)."""
    timing = e1._require(p, "timing", "params")
    _number(timing.get("target_s"), "params.timing.target_s")
    _number(timing.get("tolerance"), "params.timing.tolerance", high=1.0)
    seeds = timing.get("solver_seeds")
    if not isinstance(seeds, list) or not seeds or not all(_is_count(s) for s in seeds):
        raise ValueError("params.timing.solver_seeds must be a non-empty list of seeds")
    warm = timing.get("warmup_calls")
    if not isinstance(warm, int) or isinstance(warm, bool) or warm < 0:
        raise ValueError(f"params.timing.warmup_calls must be an integer >= 0, got {warm!r}")
    specs = e1._require(timing, "samplers", "params.timing")
    names = [s.get("name") for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"params.timing.samplers lists a sampler twice: {names}")
    out = []
    for spec in specs:
        name = spec.get("name")
        if name not in e1.SAMPLERS:
            raise ValueError(f"params.timing.samplers: unknown sampler {name!r}; expected one of {e1.SAMPLERS}")
        penalty, beta, cold_ratio = _reference(spec)
        batch = e1._BATCH_KEYS[name]
        grids: dict[int, dict] = {}
        for entry in spec.get("grid") or []:
            n = entry.get("n")
            where = f"timing sampler {name!r}, n={n}"
            if n not in ns:
                raise ValueError(f"{where}: n is not in params.instances {ns}")
            if n in grids:
                raise ValueError(f"{where}: two grids for the same n")
            scale = entry.get("scale")
            if scale not in batch:
                raise ValueError(f"{where}: scale {scale!r} is not a batch key of {name} {batch}")
            if name == "tabu" and scale != "num_reads":
                raise ValueError(f"{where}: the scale of tabu must be 'num_reads', the work unit of tabu is "
                                 f"reads (§12 B1), got {scale!r}")
            fixed = entry.get("fixed") or {}
            if scale in fixed:
                raise ValueError(f"{where}: {scale!r} is the scaled parameter, it cannot be fixed too")
            missing = [k for k in batch if k != scale and k not in fixed]
            extra = [k for k in fixed if k not in batch]
            if missing or extra:
                raise ValueError(f"{where}: 'fixed' must hold {[k for k in batch if k != scale]}; "
                                 f"missing {missing}, unexpected {extra}")
            bad = [k for k, v in fixed.items()
                   if not _is_count(v) and not (name == "tabu" and k == "num_restarts")]
            if bad:
                raise ValueError(f"{where}: fixed {bad} must be positive integers")
            if name == "tabu":  # a plain search per read (num_restarts = 0), the tenure of the plan
                e1._check_tabu_regime(fixed, n, where)
            grids[n] = {"scale": scale, "fixed": {k: int(fixed[k]) for k in fixed},
                        "values": expand_grid(entry)}
        for n in ns:
            if n not in grids:
                raise ValueError(f"params.timing: no grid for sampler {name!r} at n={n}")
        for n in ns:
            g = grids[n]
            for value in g["values"]:
                out.append({"kind": "timing", "sampler": name, "n": n, "scaled": g["scale"],
                            **g["fixed"], g["scale"]: value, "penalty": penalty, "beta": beta,
                            "cold_ratio": cold_ratio})
    return out


def _reference(spec: dict) -> tuple[float | None, str | None, float | None]:
    """``(penalty, beta, cold_ratio)`` of the reference configuration of a sampler that is timed.

    It is the reference of ANALYSIS_PLAN §12 B3 (:data:`cif.experiments.e1_attribution.REFERENCE`): the work of
    a call is sized on it, and on nothing else."""
    name = spec["name"]
    ref = spec.get("reference")
    if name == "swap_sa":
        if not isinstance(ref, dict) or ref.get("cold_ratio") is None:
            raise ValueError("timing sampler 'swap_sa': 'reference' must give the cold_ratio of its schedule "
                             "(§12 B3)")
        found = (None, None, _number(ref["cold_ratio"], "timing sampler 'swap_sa' reference cold_ratio",
                                     high=1.0, closed_high=True))
    elif not isinstance(ref, dict) or ref.get("penalty") is None:
        raise ValueError(f"timing sampler {name!r}: 'reference' must give the penalty multiplier")
    elif name == "tabu":
        if ref.get("beta") is not None:
            raise ValueError("timing sampler 'tabu' has no beta range in its reference")
        found = (_number(ref["penalty"], "timing sampler 'tabu' reference penalty"), None, None)
    else:
        penalty = _number(ref["penalty"], "timing sampler 'sa' reference penalty")
        if ref.get("beta") not in e1.BETAS:
            raise ValueError(f"timing sampler 'sa': reference beta must be one of {e1.BETAS}, "
                             f"got {ref.get('beta')!r}")
        found = (penalty, ref["beta"], None)
    wanted = e1.REFERENCE[name]
    if found != (wanted.get("penalty"), wanted.get("beta"), wanted.get("cold_ratio")):
        raise ValueError(f"timing sampler {name!r}: the reference configuration is that of §12 B3 {wanted}, "
                         f"got {ref}")
    return found


def cells(cfg: dict) -> list[dict]:
    """The pilot grid: ``floor`` cells (one per calibration instance), then ``timing`` cells (one per
    candidate work of a sampler and N)."""
    p = e1._require(cfg, "params", "config")
    if p.get("seed_role") != "calibration":
        raise ValueError(f"the pilot runs on calibration instances only: params.seed_role must be "
                         f"'calibration', got {p.get('seed_role')!r}")
    parts = p.get("parts", list(PARTS))
    if not isinstance(parts, list) or not parts or any(x not in PARTS for x in parts):
        raise ValueError(f"params.parts must be a non-empty list from {PARTS}, got {parts!r}")
    e1._check_scalars(p)
    instances = []
    for entry in e1._require(p, "instances", "params"):
        n = int(e1._require(entry, "n", "an instance entry"))
        instances.append((n, [int(s) for s in e1._require(entry, "pair_seeds", f"instances (n={n})")]))
    h_levels = [float(h) for h in e1._require(p, "h_levels", "params")]
    ns = [n for n, _ in instances]
    out: list[dict] = []
    solver_seeds: list[int] = []
    if "floor" in parts:
        _check_floor(p)
        out += [{"kind": "floor", "n": n, "pair_seed": ps, "h": h}
                for n, seeds in instances for ps in seeds for h in h_levels]
    if "timing" in parts:
        out += _check_timing(p, ns)
        solver_seeds = [int(s) for s in p["timing"]["solver_seeds"]]
    e1._check_seeds(cfg, "calibration", [s for _, seeds in instances for s in seeds], solver_seeds)
    return out


# --------------------------------------------------------------------------------------------
# The timing environment (R10)
# --------------------------------------------------------------------------------------------


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cpuinfo_blocks() -> list[dict[str, str]]:
    text = _read("/proc/cpuinfo")
    blocks = []
    for chunk in (text or "").split("\n\n"):
        fields = {}
        for line in chunk.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        if fields:
            blocks.append(fields)
    return blocks


def _dynamic_environment() -> dict:
    """What changes during a run: load average, runnable processes, clocks."""
    try:
        loadavg = [float(x) for x in os.getloadavg()]
    except OSError:
        loadavg = None
    running = None
    raw = _read("/proc/loadavg")
    if raw:
        try:
            running = int(raw.split()[3].split("/")[0])
        except (IndexError, ValueError):
            running = None
    mhz = []
    for block in _cpuinfo_blocks():
        try:
            mhz.append(float(block["cpu MHz"]))
        except (KeyError, ValueError):
            continue
    clock = ({"min": round(min(mhz), 1), "mean": round(float(np.mean(mhz)), 1), "max": round(max(mhz), 1)}
             if mhz else None)
    return {"loadavg": loadavg, "procs_running": running, "cpu_mhz": clock,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def timing_environment() -> dict:
    """The environment of a timing measurement, best effort (``None`` where the system does not say):
    host, CPU model, frequency governor and driver, turbo, logical and physical cores, kernel, thread
    settings, and the dynamic part (:func:`_dynamic_environment`)."""
    blocks = _cpuinfo_blocks()
    cores = {(b["physical id"], b["core id"]) for b in blocks if "physical id" in b and "core id" in b}
    sockets = {b["physical id"] for b in blocks if "physical id" in b}
    try:
        threads = thread_state()
    except Exception:  # best effort: threadpoolctl cannot always inspect the process
        threads = None
    return {
        "host": socket.gethostname(),
        "cpu_model": registry._cpu_model(),
        "governor": registry._cpu_governor(),
        "scaling_driver": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver"),
        "no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "boost": _read("/sys/devices/system/cpu/cpufreq/boost"),
        "cpu_count_logical": os.cpu_count(),
        "cpu_cores_physical": len(cores) or None,
        "sockets": len(sockets) or None,
        "kernel": platform.release(),
        "threads": threads,
        **_dynamic_environment(),
    }


# --------------------------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------------------------


def _median(values) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _run_floor(cell: dict, cfg: dict) -> dict:
    t0 = time.perf_counter()
    p = cfg["params"]
    key = (
        int(cell["n"]), int(cell["pair_seed"]), float(cell["h"]),
        tuple(float(h) for h in p["h_levels"]), float(p["lam"]), float(p["w_min"]),
        float(p["w_max"]), tuple(int(m) for m in p["ms"]), tuple(sorted(p["scip"].items())),
    )
    inst = e1._instance(*key)
    oracle, reference, member = inst.oracle, inst.reference, inst.member
    meta = reference.meta
    row = {
        "status": "ok",
        "K": inst.pair.k, "s_cont": member.s_cont, "s_cont_min_pair": min(m.s_cont for m in inst.pair.members),
        "binding": member.binding, "depth": member.depth, "cv_vol": member.cv_vol,
        "n_subsets": oracle.n_subsets,
        "f_star": reference.f_star, "scale": reference.scale, "ref_type": reference.ref_type,
        "scip_status": meta["status"], "scip_gap": meta["gap"], "scip_solve_s": meta["solve_time_s"],
        "scip_time_limit_hit": meta["time_limit_hit"],
        "tau_top": oracle.tau_top,
        "oracle_s": inst.oracle_s, "oracle_enum_s": oracle.enum_s, "oracle_refine_s": oracle.refine_s,
    }
    for j, m in enumerate(oracle.ms):
        row[f"E_form_{m}"] = float(oracle.e_form[j])
        row[f"n_form_{m}"] = int(oracle.n_form[j])
    row["cell_s"] = time.perf_counter() - t0
    return e1._clean(row)


def _run_timing(cell: dict, cfg: dict) -> dict:
    t0 = time.perf_counter()
    p = cfg["params"]
    timing = p["timing"]
    name, n = cell["sampler"], int(cell["n"])
    spec = {key: int(cell[key]) for key in e1._BATCH_KEYS[name]}
    entry = next(e for e in p["instances"] if int(e["n"]) == n)
    pair_seeds = [int(s) for s in entry["pair_seeds"]]
    h_levels = tuple(float(h) for h in p["h_levels"])
    solver_seeds = [int(s) for s in timing["solver_seeds"]]
    instances = [(ps, h) for ps in pair_seeds for h in h_levels]
    env_start = timing_environment()

    def call(pair_seed: int, h: float, solver_seed: int):
        pair = e1._pair(n, pair_seed, h_levels, float(p["lam"]), float(p["w_min"]), float(p["w_max"]))
        problem, k = pair.member(h).problem, pair.k
        sampler_cell = {"pair_seed": pair_seed, "n": n, "h": h, "sampler": name,
                        "penalty": cell["penalty"], "beta": cell["beta"], "cold_ratio": cell["cold_ratio"],
                        "solver_seed": solver_seed}
        seed, rng = e1._derive(sampler_cell)
        return e1._draw_pool(problem, k, sampler_cell, spec, seed, rng), problem, k

    for _ in range(int(timing["warmup_calls"])):  # untimed: imports, page faults, first-call effects
        call(*instances[0], solver_seeds[0])

    per_instance, ratios, builds, samples, valid, unique, cpu = [], [], [], [], [], [], []
    for pair_seed, h in instances:
        solves = []
        for solver_seed in solver_seeds:
            draw, problem, k = call(pair_seed, h, solver_seed)
            solves.append(draw.solve_s)
            ratios.append(draw.solve_cpu_s / draw.solve_s if draw.solve_s > 0.0 else 1.0)
            cpu.append(draw.solve_cpu_s)
            builds.append(draw.build_s)
            samples.append(draw.n_samples)
            valid.append(draw.n_valid_samples)
            unique.append(pool_subsets(draw.pool, k, problem.n, 0)[1]["n_unique"])
        per_instance.append(_median(solves))
    q25, q75 = np.quantile(per_instance, [0.25, 0.75])
    row = {
        "status": "ok",
        "n_instances": len(instances), "repeats": len(solver_seeds), "pair_seeds": pair_seeds,
        "warmup_calls": int(timing["warmup_calls"]),
        "call_s_median": _median(per_instance), "call_s_q25": float(q25), "call_s_q75": float(q75),
        "call_s_min": float(min(per_instance)), "call_s_max": float(max(per_instance)),
        "call_s_by_instance": per_instance,
        "call_cpu_s_median": _median(cpu), "cpu_over_wall_median": _median(ratios),
        "build_s_median": _median(builds),
        "n_samples_median": _median(samples), "n_valid_median": _median(valid),
        "n_unique_median": _median(unique),
        "env_start": env_start, "env_end": _dynamic_environment(),
        "cell_s": time.perf_counter() - t0,
    }
    return e1._clean(row)


def run_cell(cell: dict, cfg: dict) -> dict:
    """The row of one cell: the ceiling of a calibration instance (``floor``) or the call time of a
    candidate work (``timing``)."""
    if cell["kind"] == "floor":
        return _run_floor(cell, cfg)
    if cell["kind"] == "timing":
        return _run_timing(cell, cfg)
    raise ValueError(f"unknown cell kind {cell['kind']!r}")


# --------------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------------


def _work_of(row: dict) -> dict:
    return {key: int(row[key]) for key in e1._BATCH_KEYS[row["sampler"]]}


def choose_work(rows: list[dict], *, target_s: float, tolerance: float) -> dict:
    """The work whose median call time is closest to ``target_s`` on the log scale (plan §6).

    ``rows`` are the ``timing`` rows of one sampler and one N, one per candidate work; the order of
    the candidates is that of the scaled parameter. A tie goes to the smaller work. ``status``:
    ``ok`` if the chosen median is within ``tolerance`` (relative) of the target; otherwise
    ``edge_low`` (the smallest candidate already takes longer than the target), ``edge_high`` (the
    largest takes less) or ``off_target`` (inside the grid but not close).
    """
    if not rows:
        raise ValueError("no timing rows to choose from")
    if not (math.isfinite(target_s) and target_s > 0.0) or not tolerance > 0.0:
        raise ValueError(f"need target_s > 0 and tolerance > 0, got {target_s}, {tolerance}")
    sampler, n, scale = rows[0]["sampler"], rows[0]["n"], rows[0]["scaled"]
    ordered = sorted(rows, key=lambda r: int(r[scale]))
    works = [_work_of(r) for r in ordered]
    if len({int(r[scale]) for r in ordered}) != len(ordered):
        raise ValueError(f"duplicate candidate works for {sampler} at n={n}")
    medians = [float(r["call_s_median"]) for r in ordered]
    if not all(math.isfinite(t) and t > 0.0 for t in medians):
        raise ValueError(f"call times of {sampler} at n={n} must be positive and finite, got {medians}")
    distance = [abs(math.log(t / target_s)) for t in medians]
    best = min(range(len(ordered)), key=lambda i: (distance[i], i))  # a tie: the smaller work
    t = medians[best]
    if abs(t / target_s - 1.0) <= tolerance:
        status = "ok"
    elif best == 0 and t > target_s:
        status = "edge_low"
    elif best == len(ordered) - 1 and t < target_s:
        status = "edge_high"
    else:
        status = "off_target"
    return {
        "sampler": sampler, "n": int(n), "scaled": scale, "target_s": target_s, "tolerance": tolerance,
        "chosen": works[best], "index": best, "n_candidates": len(ordered),
        "median_call_s": t, "call_s_q25": float(ordered[best]["call_s_q25"]),
        "call_s_q75": float(ordered[best]["call_s_q75"]), "ratio_to_target": t / target_s,
        "status": status,
        "reference": {"penalty": ordered[best].get("penalty"), "beta": ordered[best].get("beta"),
                      "cold_ratio": ordered[best].get("cold_ratio")},
        "candidates": [
            {"work": works[i], "median_call_s": medians[i], "call_s_q25": float(ordered[i]["call_s_q25"]),
             "call_s_q75": float(ordered[i]["call_s_q75"]), "ratio_to_target": medians[i] / target_s}
            for i in range(len(ordered))
        ],
    }


def _share(rows: list[dict], key: str, tau: float) -> dict:
    exceed = sum(1 for r in rows if r[key] > tau)
    lo, hi = wilson_interval(exceed, len(rows))
    return {"n": len(rows), "n_exceed": exceed, "share": exceed / len(rows) if rows else math.nan,
            "wilson_95": [lo, hi]}


def floor_rule(rows: list[dict], *, tau: float, m_floor: int, m_fallback: int, min_share: float) -> dict:
    """The floor rule of H1a (plan §2): ``m0 = m_floor`` if the share of calibration instances with
    ``E_form(m_floor) > tau`` is at least ``min_share``, else ``m_fallback``.

    The share is over all instances of the rows (both N and both levels of h): one ``m0`` for the
    design. The comparison is exact (a fraction, not a float). Shares by N and by h, the share at
    ``m_fallback`` and a Wilson interval are descriptive.
    """
    if not rows:
        raise ValueError("no floor rows")
    keys = (f"E_form_{m_floor}", f"E_form_{m_fallback}")
    for key in keys:
        if any(r.get(key) is None for r in rows):
            raise ValueError(f"floor rows without {key}")
    main = _share(rows, keys[0], tau)
    enough = Fraction(main["n_exceed"], main["n"]) >= Fraction(str(min_share))
    by_n = {str(n): _share([r for r in rows if r["n"] == n], keys[0], tau)
            for n in sorted({r["n"] for r in rows})}
    by_h = {str(h): _share([r for r in rows if r["h"] == h], keys[0], tau)
            for h in sorted({r["h"] for r in rows})}
    fallback = _share(rows, keys[1], tau)
    pool_sizes = sorted(int(k[len("E_form_"):]) for k in rows[0] if re.fullmatch(r"E_form_\d+", k))
    by_m = {str(m): _share(rows, f"E_form_{m}", tau) for m in pool_sizes}
    return {
        "m0": m_floor if enough else m_fallback, "m_floor": m_floor, "m_fallback": m_fallback,
        "tau": tau, "min_share": min_share, "n_instances": main["n"], "n_exceed": main["n_exceed"],
        "share": main["share"], "wilson_95": main["wilson_95"],
        "share_fallback": fallback["share"], "by_n": by_n, "by_h": by_h, "by_m": by_m,
        "rule": (f"m0 = {m_floor} if the share of calibration instances with E_form({m_floor}) > {tau:g} "
                 f"is at least {min_share:g}, else {m_fallback}: {main['n_exceed']} of {main['n']} "
                 f"= {main['share']:.4g}"),
    }


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


@dataclass
class RunData:
    """A registered run as the report reads it: manifest, effective config, result rows."""

    manifest: dict
    config: dict
    rows: list[dict]


def load_run(path: Path | str) -> RunData:
    path = Path(path)
    return RunData(
        manifest=json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        config=json.loads((path / "config.json").read_text(encoding="utf-8")),
        rows=run.read_rows(path / "results.jsonl"),
    )


def _spread(values) -> dict | None:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {"min": min(v), "median": float(np.median(v)), "max": max(v)}


def _environment_summary(timing_rows: list[dict], manifests: list[dict]) -> dict:
    starts = [r["env_start"] for r in timing_rows if r.get("env_start")]
    ends = [r["env_end"] for r in timing_rows if r.get("env_end")]
    snaps = starts + ends

    def uniq(key):
        return sorted({s[key] for s in starts if s.get(key) is not None}, key=str)

    loads = [s["loadavg"][0] for s in snaps if s.get("loadavg")]
    mhz = [s["cpu_mhz"]["mean"] for s in snaps if s.get("cpu_mhz")]
    summary = {
        "hosts": sorted({r["host"] for r in timing_rows if r.get("host")} | set(uniq("host"))),
        "cpu_models": uniq("cpu_model"), "governors": uniq("governor"),
        "scaling_drivers": uniq("scaling_driver"), "no_turbo": uniq("no_turbo"),
        "cpu_count_logical": uniq("cpu_count_logical"), "cpu_cores_physical": uniq("cpu_cores_physical"),
        "kernels": uniq("kernel"),
        "loadavg_1m": _spread(loads), "procs_running": _spread(s.get("procs_running") for s in snaps),
        "cpu_mhz_mean": _spread(mhz),
        "cpu_over_wall_max": max((r["cpu_over_wall_median"] for r in timing_rows
                                  if r.get("cpu_over_wall_median") is not None), default=None),
        "manifest": {
            "cpu": sorted({m.get("cpu") for m in manifests if m.get("cpu")}),
            "cpu_governor": sorted({m.get("cpu_governor") for m in manifests if m.get("cpu_governor")}),
            "host": sorted({m.get("host") for m in manifests if m.get("host")}),
            "modes": sorted({m.get("mode") for m in manifests if m.get("mode")}),
            "note": "an ingested run records the machine that ingested it, not the one that ran; "
                    "the environment of the measurements is in the rows",
        },
    }
    warnings = []
    governors = summary["governors"]
    if not governors:
        warnings.append("the CPU governor is unknown")
    elif governors != ["performance"]:
        warnings.append(f"CPU governor was {governors}, not 'performance': frequency scaling "
                        f"makes the call times load-dependent")
    physical = max(summary["cpu_cores_physical"], default=None)
    basis = "physical"
    if physical is None and summary["cpu_count_logical"]:  # a VM may not report cores: assume 2 threads each
        physical, basis = max(summary["cpu_count_logical"]) // 2, "physical (estimated: half of the logical)"
    if loads and physical and max(loads) > physical / 2 + 1:
        warnings.append(f"load average reached {max(loads):.1f}, above half of the {physical} {basis} "
                        f"cores (plan §6: at most that many single-thread jobs at once)")
    if summary["cpu_over_wall_max"] is not None and summary["cpu_over_wall_max"] > 1.1:
        warnings.append(f"CPU time exceeded wall time by {summary['cpu_over_wall_max'] - 1:.0%}: "
                        f"hidden threads in a sampler call")
    summary["warnings"] = warnings
    return summary


def _check_complete(data: RunData) -> dict[str, dict]:
    """The rows of a run by cell key; every cell of the grid exactly once, none with an error."""
    cfg = data.config
    rid = data.manifest.get("run_id", "?")
    if cfg.get("experiment") != "cif.experiments.e1_pilot":
        raise ValueError(f"run {rid} is not a pilot run (experiment {cfg.get('experiment')!r})")
    expected = {run.cell_key(c): c for c in cells(cfg)}
    errors = [r for r in data.rows if r.get("error")]
    if errors:
        raise ValueError(f"run {rid}: {len(errors)} cells with an error, e.g. {errors[0]['error']}")
    got = {r["cell_key"]: r for r in data.rows}
    missing = [k for k in expected if k not in got]
    foreign = [k for k in got if k not in expected]
    if missing or foreign:
        raise ValueError(f"run {rid}: {len(missing)} cells missing, {len(foreign)} unexpected "
                         f"(wrong config?)")
    return got


def build_report(runs: list[RunData]) -> dict:
    """The pilot report from registered runs of ``floor`` and ``timing`` (one run may hold both).

    ``valid_for_section_12`` is true only if both parts are there, every run is a full (or ingested)
    run from a clean tree, every reference is certified and the target is bracketed for every sampler
    and N; otherwise ``problems`` says why and the values must not be written to §12.
    """
    if not runs:
        raise ValueError("no runs")
    problems: list[str] = []
    by_part: dict[str, tuple[RunData, list[dict]]] = {}
    infos = []
    for data in runs:
        got = _check_complete(data)
        parts = data.config["params"].get("parts", list(PARTS))
        m = data.manifest
        infos.append({
            "run_id": m.get("run_id"), "name": m.get("name"), "parts": parts, "mode": m.get("mode"),
            "git_sha": m.get("git_sha"), "bundle_sha": m.get("bundle_sha"), "git_dirty": m.get("git_dirty"),
            "config_hash": m.get("config_hash"), "started_at": m.get("started_at"),
            "finished_at": m.get("finished_at"), "n_cells": len(got),
            "hosts": sorted({r["host"] for r in got.values() if r.get("host")}),
        })
        if m.get("mode") not in ("full", "ingested"):
            problems.append(f"run {m.get('run_id')} is a {m.get('mode')!r} run, not a pilot (smoke runs "
                            f"are for the machinery only)")
        if m.get("git_dirty") is not False:
            problems.append(f"run {m.get('run_id')} comes from a dirty or unknown code state "
                            f"(git_dirty = {m.get('git_dirty')!r})")
        for part in parts:
            if part in by_part:
                raise ValueError(f"two runs provide the {part} part")
            by_part[part] = (data, [r for r in got.values() if r["kind"] == part])

    if "floor" in by_part and "timing" in by_part:
        keys = ("instances", "h_levels", "lam", "w_min", "w_max")
        a, b = (by_part[part][0].config["params"] for part in ("floor", "timing"))
        differ = [k for k in keys if a.get(k) != b.get(k)]
        if differ:
            problems.append(f"the floor and the timing runs use different calibration instances "
                            f"({', '.join(differ)}): the rule of m0 and the work must come from the same set")

    floor = work = env = None
    plan: dict = {}
    if "floor" in by_part:
        data, rows = by_part["floor"]
        fp = data.config["params"]["floor"]
        floor = floor_rule(rows, tau=fp["tau"], m_floor=fp["m_floor"], m_fallback=fp["m_fallback"],
                           min_share=fp["min_share"])
        floor["n_not_certified"] = sum(1 for r in rows if r.get("ref_type") != "certified")
        floor["share_binding"] = sum(1 for r in rows if r.get("binding")) / len(rows)
        if floor["n_not_certified"]:
            problems.append(f"{floor['n_not_certified']} calibration instances have no certified optimum")
        plan.update(fp)
    else:
        problems.append("no floor part: m0 is not determined")
    if "timing" in by_part:
        data, rows = by_part["timing"]
        tp = data.config["params"]["timing"]
        work = {}
        for entry in data.config["params"]["instances"]:
            n = int(entry["n"])
            work[str(n)] = {}
            for spec in tp["samplers"]:
                name = spec["name"]
                chosen = choose_work([r for r in rows if r["sampler"] == name and r["n"] == n],
                                     target_s=tp["target_s"], tolerance=tp["tolerance"])
                work[str(n)][name] = chosen
                if chosen["status"] != "ok":
                    problems.append(f"work of {name} at n={n}: {chosen['status']} (median "
                                    f"{chosen['median_call_s']:.3g} s against {tp['target_s']:g} s): "
                                    f"extend or refine the grid and rerun the timing part")
        env = _environment_summary(rows, [d.manifest for d in runs])
        plan.update({k: tp[k] for k in ("target_s", "tolerance", "solver_seeds", "warmup_calls")})
    else:
        problems.append("no timing part: the work of a call is not determined")
    base = next(iter(by_part.values()))[0].config["params"]
    plan.update(ms=base["ms"], seed_role=base["seed_role"], instances=base["instances"])
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "valid_for_section_12": not problems,
        "problems": problems,
        "runs": infos,
        "plan": plan,
        "floor": floor,
        "work": work,
        "timing_environment": env,
        "pilot_values": None,
        "section_12": None,
    }
    if report["valid_for_section_12"]:
        report["pilot_values"] = pilot_values(report)
        report["section_12"] = section_12_entry(report)
    return json.loads(json.dumps(report, allow_nan=False))


_SAMPLER_NAMES = {"sa": "SA", "tabu": "tabu", "swap_sa": "swap-SA"}


def _code_shas(report: dict) -> list[str]:
    """The commits of the code that computed the rows: the bundle SHA of an ingested run (``ingest``
    records the local HEAD as ``git_sha``), else the SHA of the tree of the run."""
    return sorted({r.get("bundle_sha") or r["git_sha"] for r in report["runs"]
                   if r.get("bundle_sha") or r.get("git_sha")})


def section_12_entry(report: dict) -> str:
    """The entry for §12 of the plan ("дата — что изменено — почему — затронутые гипотезы — кто
    согласовал") with the pilot values, for a report that is valid for §12. The date and the approver
    are left as ``<дата>`` and ``<кто согласовал>``; no field contains the separator ``" — "``."""
    if not report.get("valid_for_section_12"):
        raise ValueError("no entry for section 12: the report is not valid: "
                         + "; ".join(report.get("problems") or ["unknown problem"]))
    floor, plan, env = report["floor"], report["plan"], report["timing_environment"] or {}
    runs = ", ".join(r["run_id"] for r in report["runs"])
    shas = ", ".join(_code_shas(report))
    per_n = []
    for n, per_sampler in report["work"].items():
        parts = []
        for sampler, choice in per_sampler.items():
            params = ", ".join(f"{k} = {v}" for k, v in choice["chosen"].items())
            parts.append(f"{_SAMPLER_NAMES[sampler]} ({params}), медиана {choice['median_call_s']:.3g} с")
        per_n.append(f"N = {n}: " + "; ".join(parts))
    load = env.get("loadavg_1m") or {}
    where = (f"{', '.join(env.get('cpu_models') or ['?'])}, governor {', '.join(env.get('governors') or ['?'])}, "
             f"хост {', '.join(env.get('hosts') or ['?'])}, физических ядер "
             f"{', '.join(str(c) for c in env.get('cpu_cores_physical') or ['?'])}, load average "
             f"{load.get('min', math.nan):.1f}-{load.get('max', math.nan):.1f}")
    what = (f"значения [пилот] RQ1 (E1) по прогонам {runs} на коммите {shas}: m₀ = {floor['m0']} "
            f"(E_form({floor['m_floor']}) > {floor['tau']:g} у {floor['n_exceed']} из {floor['n_instances']} "
            f"калибровочных инстансов, доля {floor['share']:.3g}, порог {floor['min_share']:g}); работа одного "
            f"вызова сэмплера, медиана времени вызова ≈ {plan['target_s']:g} с на калибровочных парах: "
            + " | ".join(per_n) + f"; машина: {where}")
    why = "правило пола H1a (§2) и бюджет вызова сэмплера в RQ1 (§6), пилот на калибровочных инстансах (§3)"
    return f"<дата> — {what} — {why} — H1a, RQ1 — <кто согласовал>"


def pilot_values(report: dict, *, source: dict | None = None, allow_invalid: bool = False) -> dict:
    """The ``params.pilot`` block of the confirmatory config: ``m0`` and the work per sampler and N.

    Refused unless the report is valid for §12 (``allow_invalid`` is for trying the machinery on a
    smoke report; the block then says so in its ``source``).
    """
    if not report.get("valid_for_section_12") and not allow_invalid:
        raise ValueError("no pilot values: the report is not valid for §12: "
                         + "; ".join(report.get("problems") or ["unknown problem"]))
    if not report.get("floor") or not report.get("work"):
        raise ValueError("no pilot values: the report lacks the floor or the timing part")
    entries = []
    for sampler in next(iter(report["work"].values())):
        for n, per_sampler in report["work"].items():
            entries.append({"sampler": sampler, "n": int(n), **per_sampler[sampler]["chosen"]})
    base = {
        "run_ids": [r["run_id"] for r in report["runs"]],
        "git_sha": _code_shas(report),
        "valid_for_section_12": bool(report.get("valid_for_section_12")),
    }
    return {"source": {**base, **(source or {})}, "m0": int(report["floor"]["m0"]), "work": entries}


# --------------------------------------------------------------------------------------------
# Writing the pilot values into the confirmatory config
# --------------------------------------------------------------------------------------------


class _Dumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):  # indent the items of block sequences
        return super().increase_indent(flow, False)


_OPEN = re.compile(r"^(?P<indent>[ \t]*)# >>> pilot\b.*$", re.MULTILINE)
_CLOSE = re.compile(r"^[ \t]*# <<< pilot\b.*$", re.MULTILINE)


def render_pilot_block(values: dict, indent: str = "") -> str:
    """The YAML of ``pilot: {...}`` (work entries one per line), every line prefixed by ``indent``."""
    text = yaml.dump({"pilot": values}, Dumper=_Dumper, sort_keys=False, default_flow_style=None,
                     width=10_000, allow_unicode=True)
    return "".join(f"{indent}{line}\n" for line in text.splitlines())


def apply_pilot_block(text: str, values: dict) -> str:
    """``text`` with the lines between ``# >>> pilot`` and ``# <<< pilot`` replaced by the block."""
    opens, closes = list(_OPEN.finditer(text)), list(_CLOSE.finditer(text))
    if len(opens) != 1 or len(closes) != 1 or closes[0].start() < opens[0].end():
        raise ValueError("the config must hold exactly one '# >>> pilot' ... '# <<< pilot' block")
    head = text[: opens[0].end()]
    tail = text[closes[0].start():]
    return head + "\n" + render_pilot_block(values, opens[0].group("indent")) + tail


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def _summary(report: dict) -> str:
    lines = [f"pilot report ({'valid for §12' if report['valid_for_section_12'] else 'NOT valid for §12'})"]
    if report["floor"]:
        lines.append("  floor: " + report["floor"]["rule"] + f"  ->  m0 = {report['floor']['m0']}")
    for n, per_sampler in (report["work"] or {}).items():
        for sampler, choice in per_sampler.items():
            lines.append(f"  work n={n:>3} {sampler:8s} {choice['chosen']}  median "
                         f"{choice['median_call_s']:.3f} s  [{choice['status']}]")
    for w in (report["timing_environment"] or {}).get("warnings", []):
        lines.append(f"  warning: {w}")
    for problem in report["problems"]:
        lines.append(f"  PROBLEM: {problem}")
    if report.get("section_12"):
        lines.append("  entry for §12 (fill the date and the approver):")
        lines.append("  " + report["section_12"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e1_pilot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="build the pilot report from registered pilot runs")
    rep.add_argument("--run", type=Path, action="append", required=True,
                     help="runs/<run_id> of the floor and of the timing part (or one run with both)")
    rep.add_argument("--out", type=Path, required=True)
    mk = sub.add_parser("make-config", help="write the pilot values into the confirmatory config")
    mk.add_argument("--report", type=Path, required=True)
    mk.add_argument("--config", type=Path, required=True)
    mk.add_argument("--allow-invalid", action="store_true",
                    help="use a report that is not valid for §12 (machinery tests only)")
    sub.add_parser("env", help="print the timing environment as JSON")
    args = parser.parse_args(argv)

    if args.cmd == "env":
        print(json.dumps(timing_environment(), indent=1))
        return 0
    if args.cmd == "report":
        report = build_report([load_run(p) for p in args.run])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        print(_summary(report))
        print(f"written {args.out}")
        return 0 if report["valid_for_section_12"] else 2
    raw = args.report.read_bytes()
    report = json.loads(raw)
    source = {"report": args.report.name, "report_sha256": hashlib.sha256(raw).hexdigest()}
    values = pilot_values(report, source=source, allow_invalid=args.allow_invalid)
    new_text = apply_pilot_block(args.config.read_text(encoding="utf-8"), values)
    cfg = run.effective_config(yaml.safe_load(new_text))
    n_cells = len(run.grid(cfg))  # the filled config must be a valid grid before it is written
    args.config.write_text(new_text, encoding="utf-8")
    print(f"{args.config}: pilot block written (m0 = {values['m0']}), {n_cells} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
