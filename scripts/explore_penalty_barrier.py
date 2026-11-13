"""EXPLORATORY: does the cardinality penalty make single-flip SA freeze? (WP-D, items 8-9)

Hypothesis under test. In the binary-inclusion QUBO the penalty
``lambda_c = 100 x scale`` is much larger than the energy differences between
size-K subsets. A single flip that changes ``sum(x)`` pays ``lambda_c``, so the
swap of one asset for another goes through an infeasible intermediate state.
SA with single flips then has no temperature at which it both crosses that
barrier and resolves the objective: hot enough to cross it, the objective is
noise; cold enough to resolve the objective, ``exp(-lambda_c / T)`` freezes the
subset. Tabu search is not Metropolis and takes the two-move path greedily;
swap-SA never leaves the K-manifold.

For every (N, seed) on the legacy synthetic generator, at an equal refinement
pool ``m`` and a comparable *sampling* time budget, the script runs

* (a) SA with single flips (``neal``) for penalty multipliers
  {1, 3, 10, 30, 100} under two schedules: ``default`` (neal's own beta range,
  what the experiments used) and ``matched`` (beta range from the objective
  scale, taken from the swap-SA calibration; a diagnostic that separates a
  schedule that stops too hot from a barrier that cannot be crossed);
* (b) swap-SA over K-subsets (``cif.heuristics.swap_sa``);
* (c) Tabu in the deterministic regime (``timeout_ms=None``, fixed restarts);

and also the legacy pool rule (first ``m`` unique subsets in read order) for
SA at 100x. The reference is the best objective found by anything, including
SCIP under a time limit (``cif.classical.scip_native`` if present, otherwise
the legacy ``solve_miqp_scip``).

Measured per run: share of samples with ``sum(x) = K``; ``frozen_fraction``,
the share of reads whose sample did not change after the first third of the
schedule. neal does not expose trajectories, so the schedule is split in two
calls: the first third is run, its final states are fed to a second call as
``initial_states`` with the remaining betas (the chain is Markov, so this is
the same process as one call); swap-SA records the same snapshot internally.

This is an exploration, not a registry run: it writes
``runs/explore/penalty_barrier_<timestamp>.jsonl`` (first line: the resolved
configuration and the environment). Run the ``smoke`` preset only.

    python scripts/explore_penalty_barrier.py --preset smoke
"""

from __future__ import annotations

import os

# one BLAS thread: the generator is the only dense linear algebra here
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from collections.abc import Callable  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from dwave.samplers import SimulatedAnnealingSampler  # noqa: E402

from cif.classical.milp import solve_miqp_scip  # noqa: E402
from cif.heuristics.pipeline import iter_sampleset_subsets, refine_candidates  # noqa: E402
from cif.heuristics.swap_sa import SwapSurrogate, sample_swap_sa  # noqa: E402
from cif.problem import PortfolioProblem, Solution  # noqa: E402
from cif.quantum.neal_sampler import sample_tabu  # noqa: E402
from cif.qubo.builder import build_selection_bqm, selection_penalty_scale  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Named presets keep the configuration out of CLI defaults (R8); the resolved
# configuration is written to the first line of the output.
PRESETS: dict[str, dict] = {
    "smoke": {
        "N": [100],
        "seeds": [1000, 1001, 1002],
        "k_ratio": 0.25,
        "pool_m": 50,
        "budget_s": 1.0,
        "multipliers": [1.0, 3.0, 10.0, 30.0, 100.0],
        "sa_sweeps": 300,
        "swap_reads": 100,
        "tabu_tenure": 20,
        "tabu_restarts": 2,
        "scip_limit_s": 10.0,
    },
    "calibration": {
        "N": [50, 100, 200],
        "seeds": [1000, 1001, 1002],
        "k_ratio": 0.25,
        "pool_m": 50,
        "budget_s": 5.0,
        "multipliers": [1.0, 3.0, 10.0, 30.0, 100.0],
        "sa_sweeps": 300,
        "swap_reads": 100,
        "tabu_tenure": 20,
        "tabu_restarts": 2,
        "scip_limit_s": 60.0,
    },
}


def _json_default(obj):
    """numpy scalars -> Python scalars for json."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def make_synthetic_problem(n: int, seed: int, K: int, risk_aversion: float = 2.0) -> PortfolioProblem:
    """Legacy generator, identical to scripts/experiment_a_scalability.make_synthetic_problem."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.25, size=n)
    A = rng.standard_normal((n, n))
    sigma = A @ A.T * 0.005 + np.eye(n) * 0.01
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"S{i}" for i in range(n)),
        risk_aversion=risk_aversion,
        cardinality=K,
    )


def load_reference_solver() -> tuple[str, Callable[[PortfolioProblem, int, float], Solution]]:
    """``cif.classical.scip_native.solve_miqp_native`` if importable, else legacy SCIP."""
    try:
        from cif.classical.scip_native import solve_miqp_native

        def native(problem: PortfolioProblem, K: int, limit: float) -> Solution:
            return solve_miqp_native(problem, K, time_limit_s=limit)

        return "scip_native.solve_miqp_native", native
    except Exception as exc:  # module absent or half-written by another stream
        print(f"[reference] scip_native unavailable ({exc!r}); using legacy solve_miqp_scip")

        def legacy(problem: PortfolioProblem, K: int, limit: float) -> Solution:
            return solve_miqp_scip(problem, time_limit_s=limit)

        return "milp.solve_miqp_scip", legacy


def environment() -> dict:
    def git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10, check=False
            )
            return out.stdout.strip()
        except Exception:
            return ""

    versions = {}
    for pkg in ("numpy", "dimod", "dwave-samplers", "pyscipopt", "cvxpy"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "versions": versions,
        "blas_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def to_budget(
    run: Callable[[int], tuple[object, float]],
    units: int,
    budget_s: float,
    lo: int,
    hi: int,
    max_iter: int = 2,
    tol: float = 0.3,
):
    """Run ``run(units)`` (returns result, wall) and rescale ``units`` toward ``budget_s``.

    Wall time per unit is noisy (Tabu reads vary a lot), so the run is repeated
    at the rescaled size until it is within ``tol`` of the budget or
    ``max_iter`` rescalings were spent. The wall actually obtained is what the
    output records.
    """
    result, wall = run(units)
    for _ in range(max_iter):
        if abs(wall - budget_s) <= tol * budget_s:
            break
        units = int(min(hi, max(lo, round(units * budget_s / max(wall, 1e-6)))))
        result, wall = run(units)
    return result, wall, units


def pool_summary(
    problem: PortfolioProblem,
    surrogate: SwapSurrogate,
    subsets,
    K: int,
    m: int,
    order_label: str,
) -> dict:
    """Refine the pool (first m unique K-subsets of ``subsets``) and describe the result."""
    subsets = list(subsets)
    k_subsets = list(dict.fromkeys(s for s in subsets if len(s) == K))
    best, stats = refine_candidates(problem, k_subsets, K, m, solver="explore")
    energies = [surrogate.energy(s) for s in k_subsets]
    return {
        "pool_order": order_label,
        "n_unique_k_subsets": len(k_subsets),
        "best_surrogate_energy": min(energies) if energies else None,
        "n_pool": stats["n_candidates"],
        "refine_wall_s": stats["refine_wall_s"],
        "objective": None if best is None else best.objective,
        "feasible": None if best is None else best.feasible,
        "subset": None if best is None else best.solver_meta["subset"],
    }


def default_beta_range(bqm) -> tuple[float, float]:
    """neal's own beta range for this BQM (reported in ``info`` even for one sweep)."""
    info = SimulatedAnnealingSampler().sample(bqm, num_reads=1, num_sweeps=1, seed=0).info
    b0, b1 = info["beta_range"]
    return float(b0), float(b1)


def run_sa_flips(
    bqm,
    n: int,
    K: int,
    num_reads: int,
    num_sweeps: int,
    seed: int,
    beta_range: tuple[float, float],
):
    """SA with single flips in two chained calls (first third, remaining two thirds)."""
    sampler = SimulatedAnnealingSampler()
    betas = np.geomspace(beta_range[0], beta_range[1], num_sweeps)
    n1 = max(1, num_sweeps // 3)
    t0 = time.perf_counter()
    first = sampler.sample(
        bqm,
        num_reads=num_reads,
        beta_schedule_type="custom",
        beta_schedule=betas[:n1],
        seed=seed,
    )
    final = sampler.sample(
        bqm,
        num_reads=num_reads,
        beta_schedule_type="custom",
        beta_schedule=betas[n1:],
        seed=seed + 1,
        initial_states=(first.record.sample, list(first.variables)),
        initial_states_generator="none",
    )
    wall = time.perf_counter() - t0
    columns = [list(final.variables).index(v) for v in first.variables]
    after = np.asarray(final.record.sample)[:, columns]
    frozen = float(np.mean(np.all(after == np.asarray(first.record.sample), axis=1)))
    p_card = float(np.mean(after.sum(axis=1) == K))
    return final, wall, frozen, p_card


def run_instance(n: int, seed: int, cfg: dict, ref_name: str, ref_solve, out) -> list[dict]:
    K = max(3, round(cfg["k_ratio"] * n))
    m = cfg["pool_m"]
    budget = cfg["budget_s"]
    problem = make_synthetic_problem(n, seed, K)
    surrogate = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, K)
    scale = selection_penalty_scale(problem, K)
    base = {"N": n, "K": K, "seed": seed, "pool_m": m, "budget_s": budget}
    rows: list[dict] = []

    # reference: SCIP under a time limit
    t0 = time.perf_counter()
    ref_row = {
        "type": "reference",
        **base,
        "solver": ref_name,
        "time_limit_s": cfg["scip_limit_s"],
        "objective": None,
        "feasible": False,
    }
    try:
        ref = ref_solve(problem, K, cfg["scip_limit_s"])
        ref_row.update(
            objective=float(ref.objective) if np.isfinite(ref.objective) else None,
            feasible=bool(ref.feasible),
            status=ref.solver_meta.get("status"),
            dual_bound=ref.solver_meta.get("dual_bound"),
            gap=ref.solver_meta.get("gap"),
        )
    except Exception as exc:  # the reference is optional: "best found" still defined
        ref_row["error"] = repr(exc)
    ref_row["wall_s"] = time.perf_counter() - t0

    # (b) swap-SA, reads fixed, sweeps scaled to the budget
    def swap_run(sweeps: int):
        t = time.perf_counter()
        res = sample_swap_sa(
            problem, K, num_reads=cfg["swap_reads"], num_sweeps=sweeps, seed=seed
        )
        return res, time.perf_counter() - t

    res, wall, sweeps = to_budget(swap_run, 40, budget, 10, 5000)
    t_hot, t_cold = res.t_hot, res.t_cold  # objective-scale temperatures of this instance
    row = {
        "type": "run",
        **base,
        "method": "swap_sa",
        "variant": "swap",
        "num_reads": cfg["swap_reads"],
        "num_sweeps": sweeps,
        "sample_wall_s": wall,
        "p_card": 1.0,
        "frozen_fraction": res.frozen_fraction,
        "t_hot": t_hot,
        "t_cold": t_cold,
    }
    row.update(pool_summary(problem, surrogate, res.subsets, K, m, "energy"))
    rows.append(row)

    # (c) Tabu, deterministic regime, penalty multiplier of all experiments (100x)
    bqm100 = build_selection_bqm(problem, K, penalty_multiplier=100.0)

    def tabu_run(reads: int):
        t = time.perf_counter()
        ss = sample_tabu(
            bqm100,
            num_reads=reads,
            tenure=cfg["tabu_tenure"],
            seed=seed,
            timeout_ms=None,
            num_restarts=cfg["tabu_restarts"],
        )
        return ss, time.perf_counter() - t

    ss, wall, reads = to_budget(tabu_run, 20, budget, 3, 5000)
    sizes = np.asarray(ss.record.sample).sum(axis=1)
    row = {
        "type": "run",
        **base,
        "method": "tabu_det",
        "variant": f"restarts={cfg['tabu_restarts']}",
        "num_reads": reads,
        "restarts": cfg["tabu_restarts"],
        "sample_wall_s": wall,
        "p_card": float(np.mean(sizes == K)),
        "frozen_fraction": None,
    }
    row.update(
        pool_summary(
            problem, surrogate, iter_sampleset_subsets(ss, n, order="energy"), K, m, "energy"
        )
    )
    rows.append(row)

    # (a) SA with single flips. The cost of a sweep depends on how many flips are
    # accepted (a frozen, cold run is far cheaper than a hot one), so the number
    # of reads is rescaled for every (multiplier, schedule) to the same budget.
    matched = (1.0 / t_hot, 1.0 / t_cold)
    guess = {"default": 50, "matched": 50}
    for mult in cfg["multipliers"]:
        bqm = build_selection_bqm(problem, K, penalty_multiplier=mult)
        for schedule in ("default", "matched"):
            beta_range = default_beta_range(bqm) if schedule == "default" else matched

            def sa_run(reads: int, bqm=bqm, beta_range=beta_range):
                result = run_sa_flips(bqm, n, K, reads, cfg["sa_sweeps"], seed, beta_range)
                return result, result[1]

            (final, wall, frozen, p_card), _, sa_reads = to_budget(
                sa_run, guess[schedule], budget, 5, 50000, max_iter=1
            )
            guess[schedule] = sa_reads
            row = {
                "type": "run",
                **base,
                "method": "sa_flips",
                "variant": f"mult={mult:g},{schedule}",
                "multiplier": mult,
                "schedule": schedule,
                "num_reads": sa_reads,
                "num_sweeps": cfg["sa_sweeps"],
                "beta_range": list(beta_range),
                "sample_wall_s": wall,
                "p_card": p_card,
                "frozen_fraction": frozen,
                "t_end_over_t_hot": (1.0 / beta_range[1]) / t_hot,
                "barrier_over_t_hot": mult * scale / t_hot,
            }
            row.update(
                pool_summary(
                    problem,
                    surrogate,
                    iter_sampleset_subsets(final, n, order="energy"),
                    K,
                    m,
                    "energy",
                )
            )
            rows.append(row)
            if mult == 100.0 and schedule == "default":
                legacy = {**row, "variant": "mult=100,default,legacy-pool-order"}
                legacy.update(
                    pool_summary(
                        problem,
                        surrogate,
                        iter_sampleset_subsets(final, n, order="read"),
                        K,
                        m,
                        "read",
                    )
                )
                rows.append(legacy)

    # the best objective found by anything is the reference
    found = [r["objective"] for r in rows if r["objective"] is not None]
    if ref_row["objective"] is not None and ref_row["feasible"]:
        found.append(ref_row["objective"])
    f_ref = min(found)
    ref_row["f_ref"] = f_ref
    dual = ref_row.get("dual_bound")
    for r in rows:
        if r["objective"] is None:
            r["gap_abs"] = r["gap_rel"] = r["gap_to_dual_rel"] = None
        else:
            r["gap_abs"] = r["objective"] - f_ref
            r["gap_rel"] = r["gap_abs"] / max(abs(f_ref), 1e-12)
            # R9: distance to the certified bound, when the reference produced one
            r["gap_to_dual_rel"] = (
                None if dual is None else (r["objective"] - dual) / max(abs(dual), 1e-12)
            )
        r["f_ref"] = f_ref
    for r in [ref_row, *rows]:
        out.write(json.dumps(r, default=_json_default) + "\n")
    out.flush()
    return rows


def print_summary(all_rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        groups[(r["N"], r["method"], r["variant"])].append(r)

    def mean(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print(
        "\nmeans over seeds; gaps in % of |f_ref|; f_ref = best objective found by anything"
        " (incl. the reference solver)"
    )
    print(
        f"{'N':>4} {'method':<9} {'variant':<34} {'wall,s':>7} {'P(sum=K)':>9} {'frozen':>7} "
        f"{'best E_surr':>12} {'gap,%':>7} {'gap->dual,%':>12} {'n_uniq':>7} "
        f"{'lam/T0':>7} {'Tend/T0':>8}"
    )
    order = {"swap_sa": 0, "tabu_det": 1, "sa_flips": 2}
    for (n, method, variant), rs in sorted(
        groups.items(), key=lambda kv: (kv[0][0], order[kv[0][1]], kv[0][2])
    ):
        print(
            f"{n:>4} {method:<9} {variant:<34} {mean(rs, 'sample_wall_s'):>7.2f} "
            f"{mean(rs, 'p_card'):>9.2f} {mean(rs, 'frozen_fraction'):>7.2f} "
            f"{mean(rs, 'best_surrogate_energy'):>12.6f} {100 * mean(rs, 'gap_rel'):>7.3f} "
            f"{100 * mean(rs, 'gap_to_dual_rel'):>12.3f} {mean(rs, 'n_unique_k_subsets'):>7.0f} "
            f"{mean(rs, 'barrier_over_t_hot'):>7.3g} {mean(rs, 't_end_over_t_hot'):>8.3g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--N", nargs="+", type=int, help="override preset N")
    parser.add_argument("--seeds", nargs="+", type=int, help="override preset seeds")
    parser.add_argument("--budget-s", type=float, help="sampling budget per method, seconds")
    parser.add_argument("--scip-limit-s", type=float, help="reference solver time limit")
    parser.add_argument("--out-dir", type=Path, default=REPO / "runs" / "explore")
    args = parser.parse_args()

    cfg = dict(PRESETS[args.preset])
    if args.N:
        cfg["N"] = args.N
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.budget_s:
        cfg["budget_s"] = args.budget_s
    if args.scip_limit_s:
        cfg["scip_limit_s"] = args.scip_limit_s

    ref_name, ref_solve = load_reference_solver()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.out_dir / f"penalty_barrier_{stamp}.jsonl"
    all_rows: list[dict] = []
    with path.open("w") as out:
        header = {
            "type": "header",
            "script": "scripts/explore_penalty_barrier.py",
            "status": "exploratory, not a registry run",
            "preset": args.preset,
            "config": cfg,
            "reference_solver": ref_name,
            "environment": environment(),
            "started": datetime.now().isoformat(timespec="seconds"),
        }
        out.write(json.dumps(header, default=_json_default) + "\n")
        for n in cfg["N"]:
            for seed in cfg["seeds"]:
                print(f"[run] N={n} seed={seed}", flush=True)
                all_rows.extend(run_instance(n, seed, cfg, ref_name, ref_solve, out))
    print(f"wrote {path}")
    print_summary(all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
