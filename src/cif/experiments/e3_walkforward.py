"""E3: does the accuracy of the optimisation matter out of sample? The cells (RQ3).

Experiment module of the runner (:mod:`cif.experiments.run`, protocol of :mod:`cif.experiments.toy`):
``cells(cfg)`` is the grid, ``run_cell(cell, cfg)`` one result row. ANALYSIS_PLAN §2 RQ3, §3, §5.

A cell is one (market, period, arm, λ, fold)::

    market, period      the data and the contiguous walk-forward (cif.experiments.e3_data)
    arm                 MV (the objective of §1 with λ) or GMV (min wᵀΣw, the same K_f and u)
    lam                 the risk aversion: ``lambda_f`` (one value), or each of ``lambdas`` (the pilot)
    fold, formation_date  the fold of the period and the close of its last training day

What a cell computes, in order (all on the fold's problem, which is the engine's own estimate,
:mod:`cif.experiments.e3_data`):

1. ``s_cont`` (the support of the continuous optimum of the MV relaxation) and
   ``K_f = max(⌈1/u⌉ + 1, ⌊s_cont/2⌋)``. The GMV arm uses the K_f of the MV relaxation ("с той же
   кардинальностью"); ``s_cont_arm`` is the support of its own relaxation. ``K > N_f`` is clipped
   to ``N_f`` (``K_used``) and ``N_f·u < 1`` leaves the box empty (``infeasible_box``).
2. the certified optimum (``Σz ≤ K``; SCIP to ``limits/absgap`` ≈ ``1e-6·s_ref``, §5) and, if it is
   certified, the k best supports (operator (г), :mod:`cif.backtest.kbest`: ranks 2…R+1, ``Σz = K``,
   superset cuts, 600 s per MIQP). Rank 1 is the refinement of the optimum's support. A rank is a
   dict ``rank, support, w, objective, dose_bp, certified, tie``; ``w`` are the weights on
   ``support`` (indices into ``assets``).
3. the real solvers (в) (``params.solvers``; SCIP 10 s, tabu / SA / LS 1 s end-to-end, one run per
   solver and fold, one common start): the realised gap, and the rank of the final portfolio among
   the listed supports (:func:`locate_rank`; ``rank = len(ranks) + 1`` for "beyond the list").
4. the descriptive operators (б) (``swap``: states K-subsets, 30 replicas per level, at most 66 000
   refinements per level, no wall-clock limit) and (а) (``perturb``), on the arms of
   ``params.operators.arms``.

Everything but (1) and (2) is skipped when the optimum is not certified: every gap is measured
against ``f*``. The status of a row is ``ok`` (certified optimum, all ranks found and certified),
``degenerate`` (certified, but fewer than ``R + 1`` distinct refined portfolios exist: ``N_f ≤ K + 1``),
``kbest_incomplete`` (a MIQP of the enumeration hit its limit), ``kbest_inconsistent`` (the
enumeration contradicts the certificate: recorded, never hidden), ``uncertified_optimum``,
``no_incumbent`` and ``infeasible_box``. Only ``ok`` rows can enter ``F_k`` (§2). The row also holds
the check of the fold (``problem_check``), so that the assembly can see it is the same data and code,
the seeds, and the times (``times``: keys ending in ``_s``; SCIP's and the heuristics' own times are wall
clock of one thread).

Seeds (§3, M2). ``solver_seed`` is in the calibration range 10 000-19 999 (the pilot) or the test
range 20 000-29 999 and goes to SCIP as ``randomization/randomseedshift``; the real solvers get
``SeedSequence([solver_seed, fold, arm, solver])``, the operators ``SeedSequence([solver_seed, arm,
λ])`` as the base of the plan's ``SeedSequence((base, fold, level, replica))``.

Config (YAML) ``params``::

    seed_role: calibration | test         # solver_seed range
    solver_seed: 20000                    # the manifest ``seeds`` must be exactly [solver_seed]
    market: {...}                         # cif.experiments.e3_data.load_market
    period: {name, start, end}            # first test day >= start, whole windows to end
    arms: [MV, GMV]                       # US: [MV]
    lambda_f: 10.0 | lambdas: [5, 10, 20] # exactly one of the two (the λ_f rule is a pilot)
    n_ranks: 32                           # R (FF49: ranks 2-33; US: 1, rank 2)
    optimum: {time_limit_s, mip_gap, formulation, diag}
    kbest: {time_limit_s, card_mode, abs_gap_rel, warm_start, boundary_check}
    solvers: null | {tuned, arms, common_start, scip: {...}, tabu: {...}, sa: {...}, ls: {...}}
    operators: null | {arms, swap: {levels, n_replicas, max_steps, max_restarts} | null,
                       perturb: {levels, n_replicas, max_draws} | null}

A full (not smoke) run refuses ``solvers`` with ``tuned: false``: the tuned configurations for
N = 100 come from E2 (§2 H3c) and the placeholders must not be run by accident.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from cif.backtest.injection import BP, W_SUPPORT_TOL, GapInjector, gmv_problem
from cif.backtest.kbest import KBestResult, _polish, enumerate_kbest
from cif.classical.scip_native import solve_miqp_native
from cif.experiments import e3_data as data
from cif.heuristics.anytime import anytime_sa, anytime_tabu
from cif.heuristics.local_search import local_search
from cif.heuristics.oracle import certified_optimum
from cif.heuristics.start import CommonStart, common_start
from cif.heuristics.swap_sa import SwapSurrogate, calibrate_hot_temperature
from cif.instances.diagnostics import binding_diagnostics
from cif.problem import PortfolioProblem, Solution

__all__ = [
    "ARMS",
    "SEED_RANGES",
    "SOLVERS",
    "calendar_grid",
    "cells",
    "check_params",
    "lambdas_of",
    "locate_rank",
    "operator_seed",
    "run_cell",
    "solver_seed",
]

ARMS = ("MV", "GMV")
SOLVERS = ("scip", "tabu", "sa", "ls")
#: solver seed ranges of ANALYSIS_PLAN §3 (M2): the pilot is a calibration, the rest is the test
SEED_RANGES = {"calibration": (10_000, 20_000), "test": (20_000, 30_000)}

_ARM_CODE = {"MV": 0, "GMV": 1}
_SOLVER_CODE = {"scip": 0, "tabu": 1, "sa": 2, "ls": 3}
_OPERATOR_TAG = 0x0B  # separates the streams of the operators from those of the solvers
_CALIBRATION_TAG = 0xCA11B
_OPTIMUM_KEYS = ("time_limit_s", "mip_gap", "formulation", "diag")
_KBEST_KEYS = ("time_limit_s", "card_mode", "abs_gap_rel", "warm_start", "boundary_check")
_SOLVER_KEYS = {
    "scip": ("formulation", "diag", "budget_s", "mip_gap"),
    "tabu": ("budget_s", "num_restarts_b", "num_reads_b", "tenure", "penalty", "m_b"),
    "sa": ("budget_s", "num_reads_b", "num_sweeps", "penalty", "beta", "m_b"),
    "ls": ("budget_s", "improvement", "num_restarts", "rcl_size"),
}
_SWAP_KEYS = ("levels", "n_replicas", "max_steps", "max_restarts")
_PERTURB_KEYS = ("levels", "n_replicas", "max_draws")


# --------------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------------


def _require(mapping, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping or mapping[key] is None:
        raise ValueError(f"{where}: missing '{key}'")
    return mapping[key]


def _levels(spec: dict, where: str) -> list[float]:
    levels = [float(g) for g in _require(spec, "levels", where)]
    if not levels or any(not (g > 0.0 and g == int(g)) for g in levels):
        raise ValueError(f"{where}.levels must be positive whole numbers of basis points, "
                         f"got {spec['levels']}")  # the integer enters the SeedSequence
    return levels


def _check_solvers(spec: dict) -> None:
    for key in ("tuned", "arms", "common_start"):
        _require(spec, key, "params.solvers")
    bad = [a for a in spec["arms"] if a not in ARMS]
    if bad or not spec["arms"]:
        raise ValueError(f"params.solvers.arms must list arms from {ARMS}, got {spec['arms']}")
    names = [k for k in spec if k in SOLVERS]
    if not names:
        raise ValueError(f"params.solvers: configure at least one of {SOLVERS}")
    unknown = [k for k in spec if k not in SOLVERS and k not in ("tuned", "arms", "common_start")]
    if unknown:
        raise ValueError(f"params.solvers: unknown keys {unknown}")
    for name in names:
        for key in _SOLVER_KEYS[name]:
            if key not in spec[name]:
                raise ValueError(f"params.solvers.{name}: missing '{key}'")
        if not float(spec[name]["budget_s"]) > 0.0:
            raise ValueError(f"params.solvers.{name}.budget_s must be positive")


def _check_operators(spec: dict) -> None:
    arms = _require(spec, "arms", "params.operators")
    if any(a not in ARMS for a in arms) or not arms:
        raise ValueError(f"params.operators.arms must list arms from {ARMS}, got {arms}")
    swap, perturb = spec.get("swap"), spec.get("perturb")
    if swap is None and perturb is None:
        raise ValueError("params.operators: configure 'swap' (б) and / or 'perturb' (а)")
    if swap is not None:
        for key in _SWAP_KEYS:
            _require(swap, key, "params.operators.swap")
        _levels(swap, "params.operators.swap")
    if perturb is not None:
        for key in _PERTURB_KEYS:
            _require(perturb, key, "params.operators.perturb")
        _levels(perturb, "params.operators.perturb")


def lambdas_of(p: dict) -> list[float]:
    one, many = p.get("lambda_f"), p.get("lambdas")
    if one is None and many is None:
        raise ValueError("params.lambda_f is not set: it is a [пилот] value (plan §3, the rule of 70 % "
                         "binding folds); put the value of the pilot report here and in §12 of the plan. "
                         "Until then only a --smoke run is possible")
    if one is not None and many is not None:
        raise ValueError("params: give exactly one of lambda_f (a value) and lambdas (the pilot's "
                         "candidates)")
    values = [float(one)] if one is not None else [float(x) for x in many]
    if not values or any(not (math.isfinite(x) and x > 0.0) for x in values):
        raise ValueError(f"lambda_f / lambdas must be positive numbers, got {one if many is None else many}")
    if len(set(values)) != len(values):
        raise ValueError(f"lambdas must be distinct, got {values}")
    return values


def check_params(cfg: dict) -> dict:
    """Everything the result depends on is stated in the config (R8): fail on a dry run."""
    p = _require(cfg, "params", "config")
    role = p.get("seed_role")
    if role not in SEED_RANGES:
        raise ValueError(f"params.seed_role must be one of {sorted(SEED_RANGES)}, got {role!r}")
    seed = _require(p, "solver_seed", "params")
    lo, hi = SEED_RANGES[role]
    if isinstance(seed, bool) or not isinstance(seed, int) or not lo <= seed < hi:
        raise ValueError(f"solver_seed {seed!r} is outside the {role} range [{lo}, {hi}) of §3")
    declared = cfg.get("seeds")
    if declared is not None and sorted({int(s) for s in declared}) != [seed]:
        raise ValueError(f"the top-level seeds {sorted(declared)} (recorded in the manifest) must "
                         f"be exactly [{seed}], the solver seed")
    for key in ("market", "period"):
        _require(p, key, "params")
    for key in ("name", "start", "end"):
        _require(p["period"], key, "params.period")
    arms = p.get("arms")
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        raise ValueError(f"params.arms must list distinct arms from {ARMS}, got {arms}")
    n_ranks = _require(p, "n_ranks", "params")
    if isinstance(n_ranks, bool) or not isinstance(n_ranks, int) or n_ranks < 1:
        raise ValueError(f"params.n_ranks must be a positive integer, got {n_ranks!r}")
    lambdas_of(p)
    for key, names in (("optimum", _OPTIMUM_KEYS), ("kbest", _KBEST_KEYS)):
        spec = _require(p, key, "params")
        for name in names:
            if name not in spec:
                raise ValueError(f"params.{key}: missing '{name}'")
    if p["kbest"]["card_mode"] not in ("eq", "le"):
        raise ValueError(f"params.kbest.card_mode must be 'eq' or 'le', got {p['kbest']['card_mode']!r}")
    if p.get("solvers") is not None:
        _check_solvers(p["solvers"])
    if p.get("operators") is not None:
        _check_operators(p["operators"])
    return p


def calendar_grid(cfg: dict) -> tuple[pd.DatetimeIndex, data.PeriodGrid]:
    """The trading calendar of the config's market and the test windows of its period. Needs the
    market and the period only, NOT λ: a confirmatory config says ``lambda_f: null`` until the pilot has
    chosen it, and the length of the confirmation is a fact of the calendar (the pilot rescales its
    power to it)."""
    p = _require(cfg, "params", "config")
    for key in ("market", "period"):
        _require(p, key, "params")
    for key in ("name", "start", "end"):
        _require(p["period"], key, "params.period")
    for key in ("train_days", "test_days"):
        _require(p["market"], key, "params.market")
    calendar = data.market_calendar(p["market"])
    grid = data.period_grid(calendar, p["period"]["start"], p["period"]["end"],
                            int(p["market"]["train_days"]), int(p["market"]["test_days"]))
    return calendar, grid


def cells(cfg: dict) -> list[dict]:
    """The grid, ``arm`` then ``lam`` then ``fold``. Reads the trading calendar only.

    The order matters for the run server, which cuts the grid as ``cells[i::S]`` (the runner's
    ``--shard``, the launcher): with the arm as the outer loop every shard holds both arms (the MV arm
    carries the real solvers and the operators, GMV does not) and folds ``S`` apart, whatever the parity
    of ``S``. A cell depends on nothing but its coordinates and the config, so any cut is valid."""
    p = check_params(cfg)
    market, period = p["market"], p["period"]
    calendar, grid = calendar_grid(cfg)
    lambdas = lambdas_of(p)
    return [
        {"market": str(market.get("name", market["source"])), "period": str(period["name"]),
         "arm": arm, "lam": lam, "fold": fold,
         "formation_date": str(calendar[grid.formation_row(fold)].date())}
        for arm in p["arms"]
        for lam in lambdas
        for fold in range(grid.n_folds)
    ]


# --------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------


def _derive(*labels: int) -> int:
    """A 31-bit seed (``dwave`` samplers refuse more) from ``SeedSequence`` of the labels."""
    seq = np.random.SeedSequence([int(x) for x in labels])
    return int(seq.generate_state(1, dtype=np.uint32)[0]) & 0x7FFFFFFF


def solver_seed(base: int, fold: int, arm: str, solver: str) -> int:
    """The seed of a real solver on a fold: ``SeedSequence([base, fold, arm, solver])``."""
    return _derive(base, fold, _ARM_CODE[arm], _SOLVER_CODE[solver])


def operator_seed(base: int, arm: str, lam: float) -> int:
    """The ``base`` of the plan's ``SeedSequence((base, fold, level, replica))`` of (б) and (а)."""
    return _derive(base, _OPERATOR_TAG, _ARM_CODE[arm], round(float(lam) * 1000))


# --------------------------------------------------------------------------------------------
# Rank of a portfolio among the listed supports
# --------------------------------------------------------------------------------------------


def locate_rank(
    problem: PortfolioProblem, weights: np.ndarray, ranks: list[dict], tie_tol: float
) -> dict:
    """The rank of a final portfolio among the supports of the k best (plan, H3c: "ранг решателя по
    носителю"): its support (weights above ``1e-6``) is taken to the fixed point of the refinement,
    as every listed support is (:func:`cif.backtest.kbest._polish`), and looked up.

    Not listed: if its objective ties with a listed one within ``tie_tol`` (a tie group cut at the
    boundary of the list) it takes that rank (``matched`` False, ``tie_matched`` True); if it is worse
    than the last listed rank it is "beyond the list" (``beyond`` True, rank ``len(ranks) + 1``,
    the ``> 33`` bin of the plan); otherwise ``rank`` is ``None`` (unexplained, to be looked at).
    """
    w = np.asarray(weights, dtype=float)
    support = tuple(int(i) for i in np.flatnonzero(w > W_SUPPORT_TOL * problem.budget))
    out = {"rank": None, "matched": False, "beyond": False, "tie_matched": False,
           "support": list(support), "objective": None}
    polished = _polish(problem, support) if support else None
    if polished is None:
        return out
    out.update(support=list(polished.support), objective=float(polished.objective))
    for r in ranks:
        if tuple(r["support"]) == polished.support:
            return {**out, "rank": int(r["rank"]), "matched": True}
    near = [r for r in ranks if abs(float(r["objective"]) - polished.objective) <= tie_tol]
    if near:
        return {**out, "rank": int(near[0]["rank"]), "tie_matched": True}
    if ranks and polished.objective > float(ranks[-1]["objective"]) + tie_tol:
        return {**out, "rank": len(ranks) + 1, "beyond": True}
    return out


# --------------------------------------------------------------------------------------------
# Serialisation of the pieces of a row
# --------------------------------------------------------------------------------------------


def _problem_check(rec: data.FoldRecord) -> dict:
    """Sums that identify the estimate of the fold (compared with a tolerance by the assembly: the
    Ledoit–Wolf fit may differ in the last bits between machines)."""
    return {
        "n": int(rec.mu.size), "mu_sum": float(rec.mu.sum()),
        "sigma_trace": float(np.trace(rec.sigma)), "sigma_fro": float(np.linalg.norm(rec.sigma)),
    }


def _ranks_rows(result: KBestResult) -> list[dict]:
    return [
        {"rank": r.rank, "support": [int(i) for i in r.support],
         "w": [float(r.weights[i]) for i in r.support], "objective": float(r.objective),
         "dose_bp": float(r.dose_bp), "certified": bool(r.certified), "tie": bool(r.tie)}
        for r in result.ranks
    ]


def _solves_rows(result: KBestResult) -> list[dict]:
    """The MIQPs of the enumeration without their clocks (those go to ``times["kbest_solves"]``)."""
    return [
        {"solve": s.solve, "status": s.status, "certified": bool(s.certified),
         "n_nodes": int(s.n_nodes), "warm_start_used": bool(s.warm_start_used),
         "gap_over_limit": (s.abs_gap / s.abs_gap_limit) if s.abs_gap_limit > 0 else None}
        for s in result.solves
    ]


def _solves_clocks(result: KBestResult) -> list[dict]:
    return [{"solve": s.solve, "build_s": s.build_time_s, "solve_s": s.solve_time_s,
             "cpu_s": s.solve_cpu_s} for s in result.solves]


def _positive_part(w: np.ndarray) -> tuple[list[int], list[float]]:
    idx = [int(i) for i in np.flatnonzero(w > 1e-12)]
    return idx, [float(w[i]) for i in idx]


# --------------------------------------------------------------------------------------------
# The real solvers (в)
# --------------------------------------------------------------------------------------------


def _beta_range(problem: PortfolioProblem, k: int, spec: dict, seed: int):
    """SA's inverse-temperature range: the sampler's own (``default``) or scaled to the objective
    (``objective``: hot where random swaps are accepted 80 % of the time, cold 1 % of it, as E1)."""
    if spec["beta"] == "default":
        return None
    if spec["beta"] != "objective":
        raise ValueError(f"solvers.sa.beta must be 'default' or 'objective', got {spec['beta']!r}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, _CALIBRATION_TAG]))
    surrogate = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, k)
    t_hot = calibrate_hot_temperature(surrogate, rng, 0.8)
    return (1.0 / t_hot, 1.0 / (0.01 * t_hot))


def _run_solver(
    name: str, problem: PortfolioProblem, k: int, spec: dict, start: CommonStart | None, seed: int
) -> Solution:
    budget = float(spec["budget_s"])
    extra = {"start": start} if start is not None else {}
    if name == "scip":
        warm = start.weights if start is not None and np.isfinite(start.objective) else None
        offset = float(start.time_s) if start is not None else 0.0
        return solve_miqp_native(
            problem, k, formulation=spec["formulation"], diag=spec["diag"], card_mode="le",
            budget_s=budget, clock_offset_s=offset, warm_start=warm,
            mip_gap=float(spec["mip_gap"]), seed=seed, scip_params=spec.get("scip_params"),
        )
    if name == "tabu":
        return anytime_tabu(
            problem, k, num_restarts_b=int(spec["num_restarts_b"]),
            num_reads_b=int(spec["num_reads_b"]), tenure=int(spec["tenure"]),
            penalty_multiplier=float(spec["penalty"]), budget_s=budget, seed=seed,
            m_b=int(spec["m_b"]), card_mode="le", **extra,
        )
    if name == "sa":
        return anytime_sa(
            problem, k, num_reads_b=int(spec["num_reads_b"]), num_sweeps=int(spec["num_sweeps"]),
            penalty_multiplier=float(spec["penalty"]), beta_range=_beta_range(problem, k, spec, seed),
            budget_s=budget, seed=seed, m_b=int(spec["m_b"]), card_mode="le", **extra,
        )
    restarts = spec["num_restarts"]
    return local_search(
        problem, k, budget_s=budget, seed=seed, improvement=spec["improvement"],
        num_restarts=None if restarts is None else int(restarts), rcl_size=int(spec["rcl_size"]),
        card_mode="le", **extra,
    )


_META_KEYS = ("status", "stop", "n_batches", "pool_size", "restarts_done", "n_evals", "n_nodes",
              "first_own_s", "build_s", "sample_s", "refine_s", "warm_start_used", "gap")


def _solvers_block(
    p: dict, arm: str, fold: int, problem: PortfolioProblem, k: int, f_star: float, ranks: list[dict],
    tie_tol: float, start: CommonStart | None,
) -> dict:
    spec = p["solvers"]
    out: dict = {}
    for name in SOLVERS:
        if name not in spec:
            continue
        seed = solver_seed(p["solver_seed"], fold, arm, name)
        sol = _run_solver(name, problem, k, spec[name], start, seed)
        w = np.asarray(sol.weights, dtype=float)
        idx, vals = _positive_part(w)
        located = locate_rank(problem, w, ranks, tie_tol) if np.isfinite(sol.objective) else {}
        objective = float(sol.objective)
        out[name] = {
            "objective": objective,
            "gap_bp": (objective - f_star) * BP if np.isfinite(objective) else None,
            "feasible": bool(sol.feasible),
            "rank": located.get("rank"), "rank_matched": located.get("matched", False),
            "rank_beyond": located.get("beyond", False),
            "rank_tie_matched": located.get("tie_matched", False),
            "support_fixed_point": located.get("support", []),
            "support": [int(i) for i in np.flatnonzero(w > W_SUPPORT_TOL * problem.budget)],
            "idx": idx, "w": vals, "seed": seed, "budget_s": float(spec[name]["budget_s"]),
            "wall_s": float(sol.wall_time_s),
            "meta": {key: sol.solver_meta[key] for key in _META_KEYS if key in sol.solver_meta},
        }
    return out


def _start_block(start: CommonStart | None) -> dict | None:
    if start is None:
        return None
    return {"objective": float(start.objective), "time_s": float(start.time_s),
            "subset": [int(i) for i in start.subset]}


# --------------------------------------------------------------------------------------------
# The descriptive operators (б) and (а)
# --------------------------------------------------------------------------------------------


def _swap_block(inj: GapInjector, spec: dict, base: int, fold: int) -> tuple[list[dict], list[float]]:
    """The levels of (б) and the seconds each took (for ``times``)."""
    out, clocks = [], []
    for g in _levels(spec, "operators.swap"):
        res = inj.swap_level(
            g, base_seed=base, fold=fold, level_key=int(g), n_replicas=int(spec["n_replicas"]),
            max_steps=int(spec["max_steps"]), max_restarts=int(spec["max_restarts"]),
            time_limit_s=None,  # plan v2.4: the work is bounded by steps and chains, not by the clock
        )
        out.append({
            "gap_target_bp": g, "status": res.status, "n_requested": res.n_requested,
            "n_found": len(res.replicas), "n_chains": res.n_chains, "n_evals": res.n_evals,
            "min_positive_neighbour_gap_bp": res.min_positive_neighbour_gap_bp,
            "replicas": [
                {"support": [int(i) for i in r.subset],
                 "w": [float(r.weights[i]) for i in r.subset], "gap_bp": float(r.gap_bp),
                 "steps": int(r.steps), "restarts": int(r.restarts)}
                for r in res.replicas
            ],
        })
        clocks.append(res.elapsed_s)
    return out, clocks


def _perturb_block(inj: GapInjector, spec: dict, base: int, fold: int) -> list[dict]:
    out = []
    support = [int(i) for i in inj.support]
    for g in _levels(spec, "operators.perturb"):
        res = inj.perturb_level(
            g, base_seed=base, fold=fold, level_key=int(g), n_replicas=int(spec["n_replicas"]),
            max_draws=int(spec["max_draws"]),
        )
        out.append({
            "gap_target_bp": g, "status": res.status, "reason": res.reason,
            "n_requested": res.n_requested, "n_found": len(res.replicas),
            "replicas": [
                {"support": support, "w": [float(r.weights[i]) for i in support],
                 "gap_bp": float(r.gap_bp), "t": float(r.t), "n_draws": int(r.n_draws)}
                for r in res.replicas
            ],
        })
    return out


# --------------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------------


def _empty_row(cell: dict, rec: data.FoldRecord, u: float, variant, seeds: dict) -> dict:
    return {
        "status": None, "variant": variant, "u": u,
        "train_start": str(rec.train_start.date()), "test_start": str(rec.test_start.date()),
        "test_end": str(rec.test_end.date()), "n_assets": len(rec.assets),
        "assets": list(rec.assets), "train_rows": rec.train_rows,
        "problem_check": _problem_check(rec), "extra": dict(rec.extra),
        "floor_K": data.floor_cardinality(u),
        "K": None, "K_used": None, "s_cont": None, "s_cont_arm": None, "binding": None,
        "binding_arm": None, "lower_bound_active": None, "depth": None,
        "cv_vol": None, "cv_mu": None, "cond_sigma": None,
        "optimum_certified": False, "optimum_status": None, "optimum_support": [],
        "optimum_idx": [], "optimum_w": [],
        "f_star": None, "scale": None,
        "kbest_n_ranks": None, "kbest_n_distinct": 0, "kbest_complete": False,
        "kbest_all_certified": False, "kbest_exhausted": False, "kbest_boundary_tie": None,
        "kbest_stop_status": None, "kbest_tie_share": None, "kbest_error": None,
        "kbest_tie_tol": None, "ranks": [], "kbest_solves": [],
        "seed": seeds, "times": {},
    }


def _status(result: KBestResult | None, error: str | None, n_ranks: int) -> str:
    if error is not None:
        return "kbest_inconsistent"
    assert result is not None
    if result.all_certified and result.n_distinct >= n_ranks + 1:
        return "ok"
    if result.all_certified:
        return "degenerate"  # the list ended: fewer than R + 1 distinct refined portfolios exist
    return "kbest_incomplete"


def run_cell(cell: dict, cfg: dict) -> dict:
    """The row of one cell (see the module docstring)."""
    t_cell = time.perf_counter()
    p = cfg["params"]
    spec_solvers = p.get("solvers")
    if (spec_solvers is not None and cell["arm"] in spec_solvers["arms"]
            and not spec_solvers["tuned"] and cfg.get("mode") != "smoke"):
        raise RuntimeError(  # before anything is built or solved
            "params.solvers.tuned is false: the parameters of the real solvers are placeholders; "
            "put the tuned configuration for N = 100 (E2, plan §2 H3c) in the config and set "
            "tuned: true"
        )
    table = data.get_fold_table(p["market"], p["period"])
    rec = table.folds[int(cell["fold"])]
    if str(rec.formation_date.date()) != cell["formation_date"]:
        raise RuntimeError(
            f"the calendar changed since the grid was made: fold {cell['fold']} forms on "
            f"{rec.formation_date.date()}, the cell says {cell['formation_date']}"
        )

    arm, lam, fold = cell["arm"], float(cell["lam"]), int(cell["fold"])
    u = table.market.u
    seeds = {"solver": int(p["solver_seed"]), "operator_base": operator_seed(p["solver_seed"], arm, lam)}
    row = _empty_row(cell, rec, u, table.market.meta.get("variant"), seeds)
    timings = row["times"]
    n_assets = len(rec.assets)

    if n_assets * u < 1.0 - 1e-9:  # the box 0 ≤ w ≤ u cannot reach the budget
        row.update(status="infeasible_box")
        timings["cell_s"] = time.perf_counter() - t_cell
        return data.json_safe(row)

    # (1) the fold: s_cont of the MV relaxation, K_f
    base = rec.problem(lam=lam, u=u)
    diag = binding_diagnostics(base)
    s_cont = int(diag["s_cont"])
    k_rule = data.cardinality_rule(s_cont, u)
    k = min(k_rule, n_assets)
    s_cont_arm = s_cont if arm == "MV" else int(binding_diagnostics(gmv_problem(base))["s_cont"])
    row.update(
        K=k_rule, K_used=k, s_cont=s_cont, s_cont_arm=s_cont_arm,
        binding=data.is_binding(s_cont, u), binding_arm=bool(s_cont_arm > k_rule),
        lower_bound_active=data.lower_bound_active(s_cont, u), depth=s_cont / k,
        cv_vol=diag["cv_vol"], cv_mu=diag["cv_mu"], cond_sigma=diag["cond_sigma"],
    )
    problem_mv = rec.problem(lam=lam, u=u, cardinality=k)
    problem = problem_mv if arm == "MV" else gmv_problem(problem_mv)

    # (2) the certified optimum and the k best supports
    opt, kb = p["optimum"], p["kbest"]
    t0 = time.perf_counter()
    try:
        ref = certified_optimum(
            problem, k, "le", time_limit_s=opt["time_limit_s"], mip_gap=float(opt["mip_gap"]),
            formulation=opt["formulation"], diag=opt["diag"], seed=int(p["solver_seed"]),
        )
    except RuntimeError as exc:
        if "no incumbent" not in str(exc):
            raise
        timings["optimum_s"] = time.perf_counter() - t0
        timings["cell_s"] = time.perf_counter() - t_cell
        row.update(status="no_incumbent", optimum_status=str(exc))
        return data.json_safe(row)
    timings["optimum_s"] = time.perf_counter() - t0
    meta = ref.meta
    certified = ref.ref_type == "certified"
    row["optimum_idx"], row["optimum_w"] = _positive_part(np.asarray(ref.weights, dtype=float))
    row.update(
        optimum_certified=bool(certified), optimum_status=meta["status"], f_star=ref.f_star,
        scale=ref.scale, optimum_gap=meta["gap"], optimum_primal_bound=meta["primal_bound"],
        optimum_dual_bound=meta["dual_bound"], optimum_n_nodes=meta["n_nodes"],
        optimum_build_s=meta["build_time_s"], optimum_solve_s=meta["solve_time_s"],
        optimum_time_limit_hit=meta["time_limit_hit"], optimum_polish_gain=meta["polish_gain"],
        optimum_support=[int(i) for i in np.flatnonzero(ref.weights > W_SUPPORT_TOL * problem.budget)],
    )
    if not certified:
        row.update(status="uncertified_optimum")
        timings["cell_s"] = time.perf_counter() - t_cell
        return data.json_safe(row)

    t0 = time.perf_counter()
    result: KBestResult | None = None
    try:
        result = enumerate_kbest(
            problem, ref.weights, n_ranks=int(p["n_ranks"]), cardinality=k,
            card_mode=kb["card_mode"], time_limit_s=kb["time_limit_s"],
            formulation=opt["formulation"], diag=opt["diag"], abs_gap_rel=float(kb["abs_gap_rel"]),
            seed=int(p["solver_seed"]), permutation_seed=None, warm_start=bool(kb["warm_start"]),
            boundary_check=bool(kb["boundary_check"]),
        )
        error = None
    except ValueError as exc:  # the enumeration contradicts the certificate: recorded, not hidden
        if "not optimal" not in str(exc):
            raise
        error = str(exc)
    timings["kbest_s"] = time.perf_counter() - t0
    row["kbest_error"] = error
    status = _status(result, error, int(p["n_ranks"]))
    row["status"] = status
    if result is None:
        timings["cell_s"] = time.perf_counter() - t_cell
        return data.json_safe(row)
    ranks = _ranks_rows(result)
    row.update(
        ranks=ranks, optimum_support=ranks[0]["support"], kbest_n_ranks=result.n_ranks,
        kbest_n_distinct=result.n_distinct, kbest_complete=bool(result.complete),
        kbest_all_certified=bool(result.all_certified), kbest_exhausted=bool(result.exhausted),
        kbest_boundary_tie=result.boundary_tie, kbest_stop_status=result.stop_status,
        kbest_tie_share=result.tie_share, kbest_tie_tol=result.tie_tol,
        kbest_elapsed_s=result.elapsed_s, kbest_pool_time_s=result.pool_time_s,
        kbest_solves=_solves_rows(result), f_star=result.f_star, scale=result.scale,
    )
    timings["kbest_solves"] = _solves_clocks(result)

    # (3) the real solvers, only where the fold can enter F_k
    if status == "ok" and spec_solvers is not None and arm in spec_solvers["arms"]:
        t0 = time.perf_counter()
        start = common_start(problem, k) if spec_solvers["common_start"] else None
        row["solvers"] = _solvers_block(p, arm, fold, problem, k, result.f_star, ranks,
                                        result.tie_tol, start)
        row["solvers_start"] = _start_block(start)
        timings["solvers_s"] = time.perf_counter() - t0
        row["seed"]["solvers"] = {n: solver_seed(p["solver_seed"], fold, arm, n)
                                  for n in SOLVERS if n in spec_solvers}

    # (4) the descriptive operators
    ops = p.get("operators")
    if status == "ok" and ops is not None and arm in ops["arms"]:
        injector = None
        t0 = time.perf_counter()
        try:
            injector = GapInjector(problem, result.ranks[0].weights)
        except ValueError as exc:
            if "not optimal" not in str(exc):
                raise
            row["operators_error"] = str(exc)
        if injector is not None:
            row["operators_scan"] = {
                "start": [int(i) for i in injector.start], "scan_evals": injector.scan_evals,
                "min_positive_neighbour_gap_bp": injector.min_positive_neighbour_gap_bp(),
            }
            timings["scan_s"] = injector.scan_time_s
            base_seed = seeds["operator_base"]
            if ops.get("swap") is not None:
                row["swap"], timings["swap_levels_s"] = _swap_block(injector, ops["swap"], base_seed, fold)
                timings["swap_s"] = time.perf_counter() - t0
            if ops.get("perturb") is not None:
                t1 = time.perf_counter()
                row["perturb"] = _perturb_block(injector, ops["perturb"], base_seed, fold)
                timings["perturb_s"] = time.perf_counter() - t1
    timings["cell_s"] = time.perf_counter() - t_cell
    return data.json_safe(row)

