"""Synthetic markets, configs and assemblies for the E3 tests (``cif.experiments.e3_*``).

Everything here is synthetic: a factor-model return panel with a known covariance
(:func:`cif.instances.factor.factor_instance`), tiny sizes so that every MIQP of a test takes a
fraction of a second, and (:func:`synthetic_assembly`) a process with a known dose-response for the
statistics. No market data is read.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

#: A small synthetic market: 10 assets, u = 0.5 (so the lower bound of K is ⌈1/u⌉ + 1 = 3), a
#: 60-day training window and 5-day test windows over 300 business days.
MARKET = {
    "name": "synthetic",
    "source": "synthetic",
    "n_assets": 10,
    "n_days": 300,
    "seed": 1000,
    "start": "2000-01-03",
    "u": 0.5,
    "train_days": 60,
    "test_days": 5,
    "gap_fraction": 0.0,
}
PERIOD = {"name": "confirmatory", "start": "2000-06-01", "end": "2001-02-28"}


def base_config(**params) -> dict:
    """A complete E3 cell config on the synthetic market (``cif.experiments.e3_walkforward``)."""
    cfg = {
        "stream": "E3",
        "name": "synthetic_test",
        "experiment": "cif.experiments.e3_walkforward",
        "seeds": [10000],
        "inputs": {},
        "params": {
            "seed_role": "calibration",
            "solver_seed": 10000,
            "market": copy.deepcopy(MARKET),
            "period": copy.deepcopy(PERIOD),
            "arms": ["MV", "GMV"],
            "lambda_f": 5.0,
            "lambdas": None,
            "n_ranks": 3,
            "optimum": {"time_limit_s": 60.0, "mip_gap": 1e-6, "formulation": "perspective",
                        "diag": "mineig"},
            "kbest": {"time_limit_s": 60.0, "card_mode": "eq", "abs_gap_rel": 1e-6,
                      "warm_start": True, "boundary_check": True},
            "solvers": None,
            "operators": None,
        },
    }
    for key, value in params.items():
        cfg["params"][key] = value
    return cfg


# --------------------------------------------------------------------------------------------
# A synthetic assembly with a known dose-response (for the statistics of ``e3_analysis``)
# --------------------------------------------------------------------------------------------


def synthetic_assembly(
    *,
    n_folds: int = 120,
    test_days: int = 21,
    arms: tuple[str, ...] = ("MV", "GMV"),
    n_ranks: int = 32,
    seed: int = 0,
    drag: float = 5e-5,
    vol_gain: float = 0.05,
    noise: float = 0.001,
    excluded: tuple[int, ...] = (),
    lam: float = 5.0,
    solvers: tuple[str, ...] = ("tabu", "sa", "ls"),
    solver_miss: float = 0.3,
    cost_bp: float = 10.0,
    market: str = "ff49",
):
    """An :class:`cif.experiments.e3_assemble.Assembly` built from a known process, not from cells.

    MV: the optimum earns ``N(5·10⁻⁴, 10⁻²)`` a day above the risk-free rate; the path of rank ``r``
    is the optimum minus a daily drag ``drag·log(ḡ_r)`` plus a little independent noise, so the
    Sharpe difference falls with ``log ḡ_r`` by construction (ḡ_r the mean dose of the rank, doses
    ``≈ 4·(r − 1)^1.1`` bp times a fold factor). GMV: the path of rank ``r`` is the optimum scaled by
    ``√(1 + vol_gain·log ḡ_r)``, so the log variance ratio rises with ``log ḡ_r``. Solver paths (MV)
    hold the optimum, except on a share ``solver_miss`` of the folds where they hold a rank 2-5
    portfolio; their gap is that rank's dose. ``excluded`` folds are outside ``F_k``. The net pass
    charges ``cost_bp`` on the traded notional of a fold (a random turnover) on its first day.
    """
    from cif.backtest.fold_sets import RankFoldSet
    from cif.experiments.e3_assemble import Assembly

    rng = np.random.default_rng(seed)
    n_days = n_folds * test_days
    index = pd.bdate_range("1980-01-01", periods=n_days)
    day_fold = np.repeat(np.arange(n_folds), test_days)
    in_fk = np.ones(n_folds, dtype=bool)
    in_fk[list(excluded)] = False
    rf = np.full(n_days, 5e-5)
    ranks = list(range(2, n_ranks + 2))
    factor = np.exp(rng.normal(0.0, 0.5, n_folds))

    doses = {}
    for arm in arms:
        base = 4.0 * (np.arange(1, n_ranks + 1) ** 1.1)  # rank 2 .. R+1, bp
        doses[arm] = pd.DataFrame(np.outer(factor, base), columns=ranks)
    mean_dose = {arm: doses[arm][in_fk].mean(axis=0).to_numpy() for arm in arms}

    gross: dict[str, np.ndarray] = {}
    mv_opt = rng.normal(5e-4, 1e-2, n_days)
    miss_folds = rng.random(n_folds) < solver_miss
    miss_rank = rng.integers(2, 6, n_folds)
    for arm in arms:
        if arm == "MV":
            gross["MV:opt"] = mv_opt + rf
            for j, r in enumerate(ranks):
                gross[f"MV:rank_{r}"] = (mv_opt - drag * np.log(mean_dose["MV"][j])
                                         + rng.normal(0.0, noise, n_days) + rf)
        else:
            opt = rng.normal(3e-4, 8e-3, n_days)
            gross[f"{arm}:opt"] = opt
            for j, r in enumerate(ranks):
                gross[f"{arm}:rank_{r}"] = (opt * np.sqrt(1.0 + vol_gain * np.log(mean_dose[arm][j]))
                                            + rng.normal(0.0, noise, n_days))
    solver_rows = []
    if "MV" in arms:
        for name in solvers:
            path = gross["MV:opt"].copy()
            for fold in np.flatnonzero(miss_folds):
                sel = day_fold == fold
                path[sel] = gross[f"MV:rank_{miss_rank[fold]}"][sel]
            gross[f"MV:solver_{name}"] = path
            for fold in range(n_folds):
                k = int(miss_rank[fold]) if miss_folds[fold] else 1
                gap = float(doses["MV"].loc[fold, k]) if k > 1 else 0.0
                solver_rows.append({"arm": "MV", "fold": fold, "solver": name, "gap_bp": gap,
                                    "rank": k, "rank_matched": True, "rank_beyond": False,
                                    "objective": -0.1, "wall_s": 1.0, "budget_s": 1.0})
    cols = list(gross)
    paths = {"gross": pd.DataFrame({c: gross[c] for c in cols}, index=index)}
    turnover = rng.uniform(0.05, 0.35, (n_folds, len(cols)))
    net = {c: gross[c].copy() for c in cols}
    first = np.flatnonzero(np.r_[True, day_fold[1:] != day_fold[:-1]])
    stats = []
    for j, c in enumerate(cols):
        arm, path_name = c.split(":")
        for fold in range(n_folds):
            cost = 2.0 * turnover[fold, j] * cost_bp / 1e4
            net[c][first[fold]] = (1.0 + gross[c][first[fold]]) * (1.0 - cost) - 1.0
            stats.append({"arm": arm, "path": path_name, "fold": fold,
                          "turnover": turnover[fold, j], "traded": 2.0 * turnover[fold, j],
                          "cost": cost})
    paths["net"] = pd.DataFrame(net, index=index)
    fold_stats = {"net": pd.DataFrame(stats), "gross": pd.DataFrame(stats).assign(cost=0.0)}

    masks = {arm: in_fk.copy() for arm in arms}
    masks["common"] = in_fk.copy()
    sets ={key: RankFoldSet(m, len(excluded) if key == "common" else 0, 0, 0)
            for key, m in masks.items()}
    binding = rng.random(n_folds) < 0.7
    folds = pd.DataFrame({
        "fold": np.arange(n_folds),
        "formation_date": [str(index[f * test_days].date()) for f in range(n_folds)],
        "test_start": [str(index[f * test_days].date()) for f in range(n_folds)],
        "test_end": [str(index[(f + 1) * test_days - 1].date()) for f in range(n_folds)],
        "n_assets": 49, "K": 10, "K_used": 10, "s_cont": np.where(binding, 22, 9),
        "binding": binding, "lower_bound_active": ~binding, "depth": 2.0, "cv_vol": 0.3,
        "cond_sigma": 50.0, "in_Fk": in_fk,
    })
    for arm in arms:
        folds[f"status_{arm}"] = np.where(in_fk, "ok", "kbest_incomplete")
        folds[f"in_Fk_{arm}"] = in_fk
        folds[f"optimum_certified_{arm}"] = True
        folds[f"tie_share_{arm}"] = 0.05
        folds[f"boundary_tie_{arm}"] = False
        folds[f"tie_tol_bp_{arm}"] = 1e-3
        folds[f"n_distinct_{arm}"] = n_ranks + 1
    if market == "us":
        folds["x_coverage"] = np.linspace(0.7, 0.98, n_folds)
    days = pd.DataFrame({"fold": day_fold, "in_Fk": in_fk[day_fold], "rf": rf}, index=index)
    meta = {
        "market": {**MARKET, "name": market}, "period": dict(PERIOD), "arms": list(arms), "lam": lam,
        "n_ranks": n_ranks, "n_folds": n_folds, "n_missing": 0, "missing": [],
        "passes": {"gross": {"tc_bps": 0.0}, "net": {"tc_bps": cost_bp}},
        "solvers": {"MV": list(solvers) if "MV" in arms else []}, "workers": 1,
        "fold_sets": {k: {"n_included": s.n_included, "n_excluded": s.n_excluded,
                          "n_excluded_optimum": s.n_excluded_optimum,
                          "n_excluded_degenerate": s.n_excluded_degenerate,
                          "n_excluded_rank": s.n_excluded_rank} for k, s in sets.items()},
        "n_days": n_days, "n_days_Fk": int(in_fk[day_fold].sum()), "cell_seconds": {},
    }
    return Assembly(
        meta=meta, folds=folds, days=days, paths=paths, fold_stats=fold_stats, doses=doses,
        solvers=pd.DataFrame(solver_rows, columns=["arm", "fold", "solver", "gap_bp", "rank",
                                                   "rank_matched", "rank_beyond", "objective",
                                                   "wall_s", "budget_s"]),
        masks=masks, fold_sets=sets, rank_ids=ranks,
    )
