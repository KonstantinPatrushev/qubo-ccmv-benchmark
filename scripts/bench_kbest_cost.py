"""EXPLORATORY: cost of the RQ3 operator (г), the k best supports, on synthetic N = 49 folds.

Each ``factor`` instance (calibration seeds 1000-1004, N = 49, u = 0.2 as on FF49, K by the plan's
rule ``max(⌈1/u⌉ + 1, ⌊s_cont/2⌋)``) plays one fold, once per arm (MV, and GMV via
``gmv_problem``). The certified optimum comes from ``certified_optimum`` (native SCIP);
``enumerate_kbest`` then lists ranks 2…R+1 (R = 32 as on FF49) plus the extra MIQP of the boundary
check. Per fold and arm the script records the time of every MIQP (wall, CPU, nodes), the share
certified, the ties, the doses and, for ``--card-mode le``, which ranks no K-subset refines to.

Synthetic data only: no market returns, no out-of-sample quantity, no confirmatory run. This is an
exploration of cost and of the distribution of in-sample doses, not a registry run (no entry in
``runs/registry.jsonl``). It writes a JSONL file (first line: configuration and environment; one
line per fold and arm; last line: the summary), by default to ``runs/explore/``; pass ``--out`` to
write elsewhere. Timings are those of the machine it runs on (record the load).

    python scripts/bench_kbest_cost.py --preset smoke   # one fold, R = 3, about 20 s
    python scripts/bench_kbest_cost.py --preset full --jobs 4   # 5 seeds x 2 arms, R = 32
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cif.experiments.timing import single_thread_env  # noqa: E402 (imports nothing from numpy)

os.environ.update(single_thread_env())  # BLAS and OpenMP read these when numpy is loaded

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402

from cif.backtest.injection import gmv_problem  # noqa: E402
from cif.backtest.kbest import N_RANKS_FF49, enumerate_kbest  # noqa: E402
from cif.classical.refine import refine_support  # noqa: E402
from cif.heuristics.oracle import certified_optimum  # noqa: E402
from cif.instances.diagnostics import binding_diagnostics  # noqa: E402
from cif.instances.factor import factor_instance  # noqa: E402
from cif.problem import PortfolioProblem  # noqa: E402

N_ASSETS = 49
W_MAX = 0.2  # u of FF49
SOLVER_SEED = 10_000  # calibration range of solver seeds (plan §3)
OPTIMUM_TIME_LIMIT_S = 600.0
PRESETS = {
    "smoke": {"seeds": (1000,), "lambdas": (5.0,), "arms": ("MV",), "n_ranks": 3},
    "full": {"seeds": (1000, 1001, 1002, 1003, 1004), "lambdas": (5.0,), "arms": ("MV", "GMV"),
             "n_ranks": N_RANKS_FF49},
}


def instance(seed: int, risk_aversion: float) -> tuple[PortfolioProblem, dict]:
    """The fold: ``factor_instance(49, seed)``, u = 0.2, K by the rule of the plan (§3)."""
    mu, sigma, _ = factor_instance(N_ASSETS, seed)
    names = tuple(f"a{i}" for i in range(N_ASSETS))
    base = PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=risk_aversion,
                            w_max=W_MAX)
    s_cont = binding_diagnostics(base)["s_cont"]
    k = max(math.ceil(1.0 / W_MAX) + 1, s_cont // 2)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=risk_aversion,
                               w_max=W_MAX, cardinality=k)
    return problem, {"s_cont": int(s_cont), "K": int(k), "binding": bool(s_cont > k)}


def reproducible_by_k_subset(problem: PortfolioProblem, support: tuple[int, ...]) -> bool:
    """Is the refined portfolio on ``support`` the refinement of some K-subset? The K − |T| outside
    assets least inclined to enter (largest reduced cost at the portfolio) are added; if the
    refinement of that K-set keeps the support, the answer is yes, otherwise no K-set does (a
    portfolio is optimal on ``T ∪ J`` iff every ``j ∈ J`` has a non-negative reduced cost)."""
    k = problem.cardinality
    if len(support) >= k:
        return True
    res = refine_support(problem.mu, problem.sigma, problem.risk_aversion, support, 0.0,
                         problem.w_max, problem.budget, return_duals=True)
    grad = -problem.mu + problem.risk_aversion * (problem.sigma @ res.weights) + res.duals.budget
    outside = sorted((j for j in range(problem.n) if j not in set(support)), key=lambda j: -grad[j])
    padded = tuple(sorted(set(support) | set(outside[: k - len(support)])))
    again = refine_support(problem.mu, problem.sigma, problem.risk_aversion, padded, 0.0,
                           problem.w_max, problem.budget)
    return tuple(int(i) for i in np.flatnonzero(again.weights > 1e-6)) == tuple(support)


def run_fold(task: dict) -> dict:
    """One fold and arm: optimum, then the enumeration; a JSON-able row."""
    seed, lam, arm = task["seed"], task["lambda"], task["arm"]
    problem, meta = instance(seed, lam)
    if arm == "GMV":
        problem = gmv_problem(problem)
    k = problem.cardinality
    t0 = time.perf_counter()
    ref = certified_optimum(problem, k, time_limit_s=OPTIMUM_TIME_LIMIT_S, seed=SOLVER_SEED)
    t_opt = time.perf_counter() - t0
    row = {"seed": seed, "lambda": lam, "arm": arm, **meta, "optimum_s": t_opt,
           "optimum_ref_type": ref.ref_type, "optimum_nodes": ref.meta["n_nodes"],
           "optimum_solve_s": ref.meta["solve_time_s"], "optimum_support": ref.meta["support"],
           "load_avg_1m": os.getloadavg()[0]}
    if ref.ref_type != "certified":
        return {**row, "skipped": "the optimum is not certified"}
    cpu0 = time.process_time()
    res = enumerate_kbest(
        problem, ref.weights, n_ranks=task["n_ranks"], card_mode=task["card_mode"],
        time_limit_s=task["time_limit_s"], seed=SOLVER_SEED, warm_start=task["warm_start"],
    )
    cpu_total = time.process_time() - cpu0
    closed = [s for s in res.solves if s.status in ("optimal", "gaplimit")]
    row.update({
        "n_ranks": res.n_ranks, "card_mode": res.card_mode, "warm_start": task["warm_start"],
        "n_distinct": res.n_distinct, "complete": res.complete, "all_certified": res.all_certified,
        "exhausted": res.exhausted, "boundary_tie": res.boundary_tie,
        "stop_status": res.stop_status, "tie_share": res.tie_share, "elapsed_s": res.elapsed_s,
        "cpu_total_s": cpu_total, "pool_time_s": res.pool_time_s, "solve_time_s": res.solve_time_s,
        "n_solves": len(res.solves),
        "closed_but_not_certified": sum(1 for s in closed if not s.certified),
        "f_star": res.f_star, "scale": res.scale, "abs_gap_limit": res.abs_gap_limit,
        "doses_bp": [float(d) for d in res.doses_bp],
        "supports": [list(s) for s in res.supports],
        "sizes": [len(s) for s in res.supports],
        "tie": [r.tie for r in res.ranks], "rank_solve": [r.miqp.solve if r.miqp else 0 for r in res.ranks],
        "solves": [
            {"solve": s.solve, "status": s.status, "certified": s.certified,
             "wall_s": s.solve_time_s, "cpu_s": s.solve_cpu_s, "build_s": s.build_time_s,
             "nodes": s.n_nodes, "warm": s.warm_start_used,
             "gap_over_limit": (s.abs_gap / s.abs_gap_limit) if math.isfinite(s.abs_gap) else None,
             "scale_ratio": (s.scale / res.scale) if s.scale else None}
            for s in res.solves
        ],
    })
    if task["card_mode"] == "le":
        row["k_subset_portfolio"] = [reproducible_by_k_subset(problem, r.support) for r in res.ranks]
    return row


def quantiles(values, qs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> list[float]:
    return [float(x) for x in np.quantile(np.asarray(values, dtype=float), qs)]


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if "skipped" not in r]
    solves = [s for r in ok for s in r["solves"]]
    ranked = [s for s in solves if s["solve"] <= (ok[0]["n_ranks"] if ok else 0)]
    out: dict = {"n_folds_arms": len(rows), "n_run": len(ok)}
    if not ok:
        return out
    out["share_certified_solves"] = float(np.mean([s["certified"] for s in solves]))
    out["share_complete_certified_folds"] = float(np.mean([r["all_certified"] and r["complete"] for r in ok]))
    out["closed_but_not_certified"] = int(sum(r["closed_but_not_certified"] for r in ok))
    out["solve_wall_s_quantiles_0_25_50_75_100"] = quantiles([s["wall_s"] for s in solves])
    out["solve_wall_s_p90"] = float(np.quantile([s["wall_s"] for s in solves], 0.9))
    out["solve_cpu_s_median"] = float(np.median([s["cpu_s"] for s in solves]))
    out["nodes_median"] = float(np.median([s["nodes"] for s in solves]))
    out["warm_start_used_share"] = float(np.mean([s["warm"] for s in solves]))
    by_group = {}
    for lo, hi in ((1, 8), (9, 16), (17, 24), (25, 32)):
        g = [s["wall_s"] for s in ranked if lo <= s["solve"] <= hi]
        if g:
            by_group[f"solves {lo}-{hi}"] = {"n": len(g), "median_s": float(np.median(g)),
                                              "max_s": float(np.max(g))}
    out["wall_by_solve_group"] = by_group
    out["elapsed_s_per_fold_arm"] = {"mean": float(np.mean([r["elapsed_s"] for r in ok])),
                                     "median": float(np.median([r["elapsed_s"] for r in ok])),
                                     "max": float(np.max([r["elapsed_s"] for r in ok]))}
    out["solve_time_s_per_fold_arm_mean"] = float(np.mean([r["solve_time_s"] for r in ok]))
    out["cpu_total_s_per_fold_arm_mean"] = float(np.mean([r["cpu_total_s"] for r in ok]))
    out["pool_time_s_mean"] = float(np.mean([r["pool_time_s"] for r in ok]))
    out["tie_share_mean"] = float(np.mean([r["tie_share"] for r in ok]))
    out["folds_with_a_tie"] = int(sum(1 for r in ok if any(r["tie"][1:])))
    out["boundary_ties"] = int(sum(1 for r in ok if r["boundary_tie"] is True))
    out["boundary_checked"] = int(sum(1 for r in ok if r["boundary_tie"] is not None))
    out["n_distinct_min"] = int(min(r["n_distinct"] for r in ok))
    out["rank2_dose_bp_quantiles_0_25_50_75_100"] = quantiles([r["doses_bp"][1] for r in ok])
    out["rank2_dose_bp_share_le_10"] = float(np.mean([r["doses_bp"][1] <= 10.0 for r in ok]))
    for arm in sorted({r["arm"] for r in ok}):
        sub = [r for r in ok if r["arm"] == arm]
        out[f"rank2_dose_bp_{arm}"] = [round(r["doses_bp"][1], 4) for r in sub]
        out[f"total_elapsed_s_{arm}"] = [round(r["elapsed_s"], 1) for r in sub]
    last = [r["doses_bp"][-1] for r in ok]
    out["dose_bp_of_last_rank_quantiles_0_25_50_75_100"] = quantiles(last)
    if any("k_subset_portfolio" in r for r in ok):
        flags = [f for r in ok for f in r.get("k_subset_portfolio", [])]
        out["ranks_not_reproduced_by_a_k_subset"] = int(sum(1 for f in flags if not f))
        out["ranks_checked_for_k_subsets"] = len(flags)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated instance seeds (calibration range 1000-1999)")
    parser.add_argument("--lambdas", default=None, help="comma-separated risk aversions")
    parser.add_argument("--arms", default=None, help="MV, GMV or MV,GMV")
    parser.add_argument("--n-ranks", type=int, default=None)
    parser.add_argument("--card-mode", choices=("eq", "le"), default="eq")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--time-limit-s", type=float, default=600.0)
    parser.add_argument("--jobs", type=int, default=1, help="folds in parallel (one thread each)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = dict(PRESETS[args.preset])
    if args.seeds:
        cfg["seeds"] = tuple(int(s) for s in args.seeds.split(","))
        if not all(1000 <= s <= 1999 for s in cfg["seeds"]):
            parser.error("instance seeds must lie in the calibration range 1000-1999")
    if args.lambdas:
        cfg["lambdas"] = tuple(float(x) for x in args.lambdas.split(","))
    if args.arms:
        cfg["arms"] = tuple(args.arms.split(","))
    if args.n_ranks:
        cfg["n_ranks"] = args.n_ranks
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "runs" / "explore" / f"kbest_cost_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "preset": args.preset, "n_assets": N_ASSETS, "w_max": W_MAX, "solver_seed": SOLVER_SEED,
        "card_mode": args.card_mode, "warm_start": not args.no_warm_start,
        "time_limit_s": args.time_limit_s, "jobs": args.jobs, "python": platform.python_version(),
        "machine": platform.machine(), "node": platform.node(), "cpu_count": os.cpu_count(),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                   "OPENBLAS_NUM_THREADS")},
        "started": datetime.now().isoformat(timespec="seconds"),
        **{k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
    }
    tasks = [
        {"seed": s, "lambda": lam, "arm": arm, "n_ranks": cfg["n_ranks"],
         "card_mode": args.card_mode, "time_limit_s": args.time_limit_s,
         "warm_start": not args.no_warm_start}
        for lam in cfg["lambdas"] for s in cfg["seeds"] for arm in cfg["arms"]
    ]
    rows: list[dict] = []
    with out.open("w") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for row in pool.map(run_fold, tasks):
                rows.append(row)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                tag = f"seed {row['seed']} lam {row['lambda']} {row['arm']} K={row['K']}"
                if "skipped" in row:
                    print(f"{tag}: skipped ({row['skipped']})", flush=True)
                    continue
                print(f"{tag} s_cont={row['s_cont']}: optimum {row['optimum_s']:.1f} s, "
                      f"{row['n_solves']} MIQPs {row['solve_time_s']:.1f} s (total "
                      f"{row['elapsed_s']:.1f} s), certified={row['all_certified']}, "
                      f"ties {row['tie_share']:.2f}, boundary tie {row['boundary_tie']}, "
                      f"dose of rank 2 {row['doses_bp'][1]:.3f} bp", flush=True)
        summary = summarise(rows)
        fh.write(json.dumps({"summary": summary}) + "\n")
    print(json.dumps(summary, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
