"""EXPLORATORY: cost of the RQ3 swap operator (б) per (fold, level) on synthetic N = 49 instances.

Each ``factor`` instance (calibration seeds 1000-1004, N = 49, u = 0.2 as on FF49, K by the plan's
rule ``max(⌈1/u⌉ + 1, ⌊s_cont/2⌋)``) plays one fold. The certified optimum comes from the native
SCIP baseline; ``GapInjector`` then runs every candidate level with R = 30 replicas, once with a
FRESH injector (nothing cached but the start state, the stand-alone cost) and once with the
injector SHARED across the levels of the fold (the cost in a production loop). Both arms (MV, and
GMV via ``gmv_problem``) and λ ∈ {5, 20} are run; the injection base seed is a calibration solver
seed (10 000-19 999). Synthetic data only; no market returns, no confirmatory runs.

Per row: status, replicas found, chains, refinements (cache misses), wall time, the smallest
positive one-swap neighbour gap, and the neighbour scan of the fold. This is an exploration, not a
registry run (no ``runs/registry.jsonl`` entry); it writes ``runs/explore/injection_cost_<time>.jsonl``
(first line: configuration and environment).

    python scripts/bench_injection_cost.py --preset smoke   # one instance, about 1 minute
    python scripts/bench_injection_cost.py --preset full    # 5 seeds x 2 lambdas x 2 arms, ~12 minutes
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cif.backtest.fold_sets import LEVELS_BP  # noqa: E402
from cif.backtest.injection import GapInjector, gmv_problem  # noqa: E402
from cif.classical.scip_native import solve_miqp_native  # noqa: E402
from cif.instances.diagnostics import binding_diagnostics  # noqa: E402
from cif.instances.factor import factor_instance  # noqa: E402
from cif.problem import PortfolioProblem  # noqa: E402

N_ASSETS = 49
W_MAX = 0.2  # u of FF49
BASE_SEED = 10_000  # calibration range of solver seeds (plan §3)
PRESETS = {
    "smoke": {"seeds": (1000,), "lambdas": (5.0,), "arms": ("MV",)},
    "full": {"seeds": (1000, 1001, 1002, 1003, 1004), "lambdas": (5.0, 20.0), "arms": ("MV", "GMV")},
}
SCIP_TIME_LIMIT_S = 120.0


def instance(seed: int, risk_aversion: float) -> tuple[PortfolioProblem, dict]:
    mu, sigma, _ = factor_instance(N_ASSETS, seed)
    names = tuple(f"a{i}" for i in range(N_ASSETS))
    base = PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=risk_aversion,
                            w_max=W_MAX)
    s_cont = binding_diagnostics(base)["s_cont"]
    k = max(math.ceil(1.0 / W_MAX) + 1, s_cont // 2)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=risk_aversion,
                               w_max=W_MAX, cardinality=k)
    return problem, {"s_cont": int(s_cont), "K": int(k), "binding": bool(s_cont > k)}


def run(preset: str, out: Path, seeds: tuple[int, ...] | None = None) -> None:
    cfg = {**PRESETS[preset], **({"seeds": seeds} if seeds else {})}
    header = {
        "preset": preset, "n_assets": N_ASSETS, "w_max": W_MAX, "base_seed": BASE_SEED,
        "levels_bp": list(LEVELS_BP), "python": platform.python_version(),
        "machine": platform.machine(), "node": platform.node(), **{k: list(v) for k, v in cfg.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for seed in cfg["seeds"]:
            for lam in cfg["lambdas"]:
                problem, meta = instance(seed, lam)
                for arm in cfg["arms"]:
                    p = problem if arm == "MV" else gmv_problem(problem)
                    sol = solve_miqp_native(p, p.cardinality, time_limit_s=SCIP_TIME_LIMIT_S)
                    status = sol.solver_meta.get("status")
                    if status not in ("optimal", "gaplimit"):
                        print(f"seed {seed} lam {lam} {arm}: SCIP status {status}, skipped")
                        continue
                    t0 = time.perf_counter()
                    shared = GapInjector(p, sol.weights)
                    init_s = time.perf_counter() - t0
                    print(f"seed {seed} lam {lam} {arm} K={meta['K']} s_cont={meta['s_cont']}: "
                          f"init {init_s:.2f} s (scan {shared.scan_evals} refinements, "
                          f"{shared.scan_time_s:.2f} s)", flush=True)
                    for g in LEVELS_BP:
                        for variant in ("fresh", "shared"):
                            inj = GapInjector(p, sol.weights) if variant == "fresh" else shared
                            res = inj.swap_level(g, base_seed=BASE_SEED, fold=seed, level_key=int(g))
                            row = {
                                "seed": seed, "lambda": lam, "arm": arm, **meta, "gap_bp": g,
                                "variant": variant, "status": res.status,
                                "found": len(res.replicas), "n_chains": res.n_chains,
                                "n_evals": res.n_evals, "elapsed_s": res.elapsed_s,
                                "min_positive_neighbour_gap_bp": res.min_positive_neighbour_gap_bp,
                                "n_distinct_subsets": res.n_distinct_subsets,
                                "scan_evals": inj.scan_evals, "scan_s": inj.scan_time_s,
                            }
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                            print(f"   g={g:>5} {variant:>6}: {res.status:<11} "
                                  f"found {len(res.replicas):>2}/30 chains {res.n_chains:>4} "
                                  f"refinements {res.n_evals:>5} {res.elapsed_s:6.2f} s", flush=True)
    print("written", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated instance seeds (calibration range 1000-1999) "
                             "replacing the preset's, to run the grid in pieces")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    if seeds and not all(1000 <= s <= 1999 for s in seeds):
        parser.error("instance seeds must lie in the calibration range 1000-1999")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "runs" / "explore" / f"injection_cost_{stamp}.jsonl"
    run(args.preset, out, seeds)


if __name__ == "__main__":
    main()
