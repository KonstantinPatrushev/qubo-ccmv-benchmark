"""E3 assembly: from the rows of the cells to the paths through the engine (RQ3).

ANALYSIS_PLAN §2 RQ3 ("Набор фолдов F_k", "Метрики RQ3"), §7.4. Input: the rows of a registered run of
:mod:`cif.experiments.e3_walkforward` (one per arm and fold). Output (:class:`Assembly`): everything
the statistics of :mod:`cif.experiments.e3_analysis` need, so that they can be rerun without the
engine.

* **Rows are checked** against the fold table that the engine makes of the same data and config
  (:func:`index_rows`): the grid is complete, no row is an error or a duplicate, the formation date,
  the universe and the estimate (sums of ``μ`` and ``Σ``) are those of the table. A row that fails
  stops the assembly; ``allow_incomplete`` turns missing cells into excluded folds and counts them.
* **F_k** (:func:`fold_sets`): per arm, :func:`cif.backtest.fold_sets.rank_fold_set` of the flags of
  the rows (optimum certified, every needed rank certified, at least ``R + 1`` distinct portfolios);
  the set of the FF49 analysis is the common one (``&`` of the arms: "фолд остаётся, только если
  условия выполнены в обоих плечах"), no selection by the size of the gaps. A one-arm market (US-MV)
  has the set of that arm.
* **Paths** are passes of the engine (:func:`cif.experiments.e3_data.run_path`) on the same rows and
  settings: ``opt`` (rank 1), ``rank_r`` and the real solvers' ``solver_<name>`` hold their stored
  weights on every fold of ``F_k``; on a fold outside it EVERY path holds the optimum's incumbent (the
  same portfolio, the days are dropped from the comparison). A pass is a cost (``tc_bps``: 0 is "до
  издержек", 10 is the main one, 30 a sensitivity), an ``execution_lag`` and a ``terminal_return``;
  all else, the risk-free series, the drift and the universe, is common. ``paths: all`` runs every
  path, ``core`` the optimum, rank 2, the ranks of H3b and the solvers.
* **What the statistics get** (:func:`fk_series`): the daily returns of a pass on the days of ``F_k``;
  excess over the risk-free accrual for the Sharpe functional (MV), the returns themselves for the
  log variance (GMV, the variance of the returns the arm minimises).
* **The levels ``L`` of the descriptive operators** (б), (а) are read where the plan reads them: on the
  folds of the PILOT period, per arm ("L — уровни, достигнутые хотя бы на 80 % фолдов пилотного периода").
  :func:`read_pilot_operators` takes the reach from a registered run of
  ``configs/E3/pilot_operators.yaml`` (:class:`PilotOperators`); ``L_a = {g ≤ g*}`` follows, and the fold
  sets ``F_a`` and ``F_b`` are the CONFIRMATORY folds where every level of ``L_a`` / ``L`` is reached.
  ``levels_L`` in the operator's config overrides the pilot (a list for both arms, or a mapping arm →
  list); with neither the assembly refuses. The pilot must be a calibration run at the same λ_f, on the
  same market, with the same operators, and must end before the confirmation begins.

The stage is registered (``python -m cif.experiments.e3_assemble``): the run holds the arrays
(parquet) and a manifest with the run and the SHA-256 of the rows it was made of.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cif.backtest import fold_sets as fs
from cif.backtest.fold_sets import RankFoldSet, enough_distinct, rank_fold_set
from cif.experiments import e3_data as d
from cif.experiments import e3_walkforward as e3
from cif.experiments import registry
from cif.experiments.run import _init_worker, effective_config, load_config, read_rows
from cif.metrics.performance import excess_returns
from cif.stats.rq3_ranks import H3B_RANKS

__all__ = [
    "Assembly",
    "AssemblyError",
    "FkSeries",
    "OperatorAssembly",
    "PilotOperators",
    "RowIndex",
    "assemble",
    "check_lineage",
    "fk_series",
    "fold_sets",
    "index_rows",
    "load_assembly",
    "load_rows",
    "main",
    "read_pilot_operators",
    "save_assembly",
]

FUNCTIONALS = ("sharpe", "logvar")
_CHECK_RTOL = 1e-6  # the Ledoit–Wolf fit may differ in the last bits between machines


class AssemblyError(RuntimeError):
    """The rows cannot be assembled: a cell is missing, failed or belongs to other data."""


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------


def load_rows(source: Path | str) -> list[dict]:
    """The rows of a run directory (or of a ``results.jsonl``)."""
    path = Path(source)
    if path.is_dir():
        path = path / "results.jsonl"
    rows = read_rows(path)
    if not rows:
        raise AssemblyError(f"no rows in {path}")
    return rows


def check_lineage(run_dir: Path | str, config_path: Path | str) -> dict:
    """The manifest of ``run_dir``, after checking that the run was made with ``config_path``: the
    hash of the config (as the runner hashes it) is the one in the manifest, and the run is a full or
    ingested one, not a smoke run."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = effective_config(load_config(config_path))
    digest = registry.config_hash(cfg)
    if manifest.get("config_hash") != digest:
        raise AssemblyError(
            f"the config {config_path} is not the one of run {run_dir.name}: its hash is {digest}, "
            f"the manifest says {manifest.get('config_hash')}"
        )
    if manifest.get("mode") == "smoke":
        raise AssemblyError(f"run {run_dir.name} is a smoke run")
    return manifest


@dataclass(frozen=True)
class RowIndex:
    """The checked rows of one λ by ``(arm, fold)``, the fold table they were checked against, the
    cells that are missing (``allow_incomplete``) and the arm order of the config."""

    cfg: dict
    table: d.FoldTable
    rows: dict[tuple[str, int], dict]
    lam: float
    missing: tuple[tuple[str, int], ...]
    arms: tuple[str, ...]

    @property
    def n_missing(self) -> int:
        return len(self.missing)


def _lambda_of(cfg: dict, lam: float | None) -> float:
    values = e3.lambdas_of(cfg["params"])
    if lam is None:
        if len(values) > 1:
            raise AssemblyError(f"the config has several lam values {values}: say which one (lam=...)")
        return values[0]
    if float(lam) not in values:
        raise AssemblyError(f"lam = {lam} is not in the config's lam values {values}")
    return float(lam)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _CHECK_RTOL * max(1.0, abs(a), abs(b))


def _check_row(row: dict, rec: d.FoldRecord) -> None:
    where = f"{row.get('arm')} fold {row.get('fold')}"
    if row.get("formation_date") != str(rec.formation_date.date()):
        raise AssemblyError(f"{where}: formation date {row.get('formation_date')} is not "
                            f"{rec.formation_date.date()} of the fold table")
    if row.get("assets") != list(rec.assets):
        raise AssemblyError(f"{where}: the universe of the row differs from the fold table's")
    check, mine = row.get("problem_check") or {}, {
        "n": int(rec.mu.size), "mu_sum": float(rec.mu.sum()),
        "sigma_trace": float(np.trace(rec.sigma)), "sigma_fro": float(np.linalg.norm(rec.sigma)),
    }
    if check.get("n") != mine["n"] or not all(
        key in check and _close(float(check[key]), mine[key])
        for key in ("mu_sum", "sigma_trace", "sigma_fro")
    ):
        raise AssemblyError(f"{where}: the estimate of the row differs from the fold table's "
                            f"({check} against {mine}): other data or other code")


def index_rows(
    cfg: dict, rows: Sequence[dict], *, lam: float | None = None, allow_incomplete: bool = False
) -> RowIndex:
    """Check ``rows`` against the config's grid and the fold table and index those of ``lam``."""
    p = e3.check_params(cfg)
    lam = _lambda_of(cfg, lam)
    table = d.get_fold_table(p["market"], p["period"])
    grid = e3.cells(cfg)
    known = {(c["arm"], c["lam"], c["fold"]) for c in grid}
    seen: dict[tuple[str, float, int], dict] = {}
    failed: list[str] = []
    for row in rows:
        key = (row.get("arm"), float(row["lam"]) if row.get("lam") is not None else None,
               row.get("fold"))
        if key not in known:
            raise AssemblyError(f"a row {key} is not in the grid of the config")
        if key in seen:
            raise AssemblyError(f"duplicate rows for the cell {key}")
        if row.get("error"):
            failed.append(f"{key}: {row['error']}")
            continue
        seen[key] = row
    if failed and not allow_incomplete:
        raise AssemblyError(f"{len(failed)} rows are errors, e.g. {failed[:3]}")
    wanted = [(arm, fold) for (arm, lm, fold) in sorted(known, key=lambda k: (k[2], k[0]))
              if lm == lam]
    missing = tuple(k for k in wanted if (k[0], lam, k[1]) not in seen)
    if missing and not allow_incomplete:
        raise AssemblyError(f"{len(missing)} cells are missing, e.g. {missing[:3]}")
    indexed = {}
    for arm, fold in wanted:
        row = seen.get((arm, lam, fold))
        if row is None:
            continue
        _check_row(row, table.folds[fold])
        indexed[(arm, fold)] = row
    return RowIndex(cfg=cfg, table=table, rows=indexed, lam=lam, missing=missing,
                    arms=tuple(p["arms"]))


# --------------------------------------------------------------------------------------------
# F_k
# --------------------------------------------------------------------------------------------


def _flags(index: RowIndex, arm: str, n_ranks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per fold: optimum certified, every needed rank certified, enough distinct portfolios; a missing
    cell fails all three."""
    n = len(index.table.folds)
    opt, ranks, distinct = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for fold in range(n):
        row = index.rows.get((arm, fold))
        if row is None:
            continue
        opt[fold] = bool(row["optimum_certified"])
        ranks[fold] = bool(row["kbest_all_certified"])
        distinct[fold] = bool(enough_distinct(row["kbest_n_distinct"], n_ranks + 1))
    return opt, ranks, distinct


def fold_sets(index: RowIndex, n_ranks: int) -> dict[str, RankFoldSet]:
    """``F_k`` per arm and the common one (key ``"common"``): the folds where the conditions hold in
    every arm (for one arm, that arm's set). See the module docstring."""
    flags = {arm: _flags(index, arm, n_ranks) for arm in index.arms}
    out = {arm: rank_fold_set(*f) for arm, f in flags.items()}
    opt = np.logical_and.reduce([f[0] for f in flags.values()])
    ranks = np.column_stack([f[1] for f in flags.values()])
    distinct = np.logical_and.reduce([f[2] for f in flags.values()])
    out["common"] = rank_fold_set(opt, ranks, distinct)
    return out


# --------------------------------------------------------------------------------------------
# Weights of the paths
# --------------------------------------------------------------------------------------------


def _vec(n: int, idx, w) -> np.ndarray:
    v = np.zeros(n)
    v[list(idx)] = w
    return v


def _optimum_vector(row: dict, n: int) -> np.ndarray | None:
    if row["ranks"]:
        return _vec(n, row["ranks"][0]["support"], row["ranks"][0]["w"])
    if row["optimum_idx"]:
        return _vec(n, row["optimum_idx"], row["optimum_w"])
    return None


def _fallback_vector(row: dict | None, n: int) -> np.ndarray:
    """What every path holds on a fold outside ``F_k``: the optimum's incumbent, else equal weights."""
    w = _optimum_vector(row, n) if row is not None else None
    return w if w is not None else np.full(n, 1.0 / n)


def _path_vector(row: dict, path: str, n: int) -> np.ndarray:
    if path == "opt":
        w = _optimum_vector(row, n)
    elif path.startswith("rank_"):
        rank = int(path.removeprefix("rank_"))
        if len(row["ranks"]) < rank or row["ranks"][rank - 1]["rank"] != rank:
            raise AssemblyError(f"{row['arm']} fold {row['fold']}: no rank {rank} in the row")
        w = _vec(n, row["ranks"][rank - 1]["support"], row["ranks"][rank - 1]["w"])
    elif path.startswith("solver_"):
        block = (row.get("solvers") or {}).get(path.removeprefix("solver_"))
        if block is None:
            raise AssemblyError(f"{row['arm']} fold {row['fold']}: no {path} in the row")
        w = _vec(n, block["idx"], block["w"])
    else:
        raise ValueError(f"unknown path {path!r}")
    if w is None:
        raise AssemblyError(f"{row['arm']} fold {row['fold']}: no weights for {path}")
    return w


def _solver_names(index: RowIndex, arm: str, in_fk: np.ndarray) -> list[str]:
    """The real solvers of an arm: those of the rows of its ``F_k`` folds (the same on every one)."""
    names: list[set[str]] = []
    for fold in np.flatnonzero(in_fk):
        row = index.rows[(arm, int(fold))]
        names.append({n for n in (row.get("solvers") or {})})
    if not names:
        return []
    if any(s != names[0] for s in names):
        raise AssemblyError(f"{arm}: the rows of F_k do not all carry the same solvers")
    return [n for n in e3.SOLVERS if n in names[0]]


def _paths_of(kind: str, rank_ids: list[int], solvers: list[str]) -> list[str]:
    if kind == "all":
        ranks = rank_ids
    elif kind == "core":
        ranks = [r for r in rank_ids if r == 2 or r in H3B_RANKS]
    else:
        raise ValueError(f"passes.*.paths must be 'all' or 'core', got {kind!r}")
    return ["opt", *(f"rank_{r}" for r in ranks), *(f"solver_{s}" for s in solvers)]


# --------------------------------------------------------------------------------------------
# Engine passes
# --------------------------------------------------------------------------------------------


def _run_task(market: dict, period: dict, task: dict) -> dict:
    """One path through the engine (a worker of the pool, or the caller itself)."""
    table = d.get_fold_table(market, period)
    weights = task["weights"]
    res = d.run_path(
        table, lambda fold, problem: weights[fold], tc_bps=task["tc_bps"],
        execution_lag=task["execution_lag"], terminal_return=task["terminal_return"],
        name=task["name"],
    )
    return {
        "key": task["key"], "index": res.daily_returns.index,
        "daily": res.daily_returns.to_numpy(),
        "folds": [(f.fold, f.turnover, f.traded, f.transaction_cost, f.test_start, f.test_end)
                  for f in res.folds],
    }


@dataclass
class OperatorAssembly:
    """The descriptive operators (б) ``swap`` and (а) ``perturb`` of the rows (module docstring).

    ``reach`` — per arm, operator and level: the folds where the operator ran, the statuses and the
    share reached; ``gaps`` — the realised gap of every replica found; ``sets["<arm>:<operator>"]`` —
    ``levels_all``, ``L`` (reached on at least ``min_share`` of the PILOT folds, or given in the config),
    ``L_a = {g ≤ g*}``, the fold masks ``mask_a`` and ``mask_b`` (the CONFIRMATORY folds where every level
    of ``L_a`` / ``L`` is reached; EMPTY when the levels are) and where ``L`` came from (``L_source``;
    for ``pilot`` also ``pilot_run``, ``pilot_n_ran`` and ``pilot_share`` per level); ``paths["<operator>:<pass>"]`` —
    daily returns of the optimum (``"<arm>:opt"``) and of every replica of the levels of ``L``
    (``"<arm>:g<level>:r<replica>"``), which hold the replica on the folds of ``mask_b`` and the
    optimum elsewhere; ``fold_stats`` — turnover and cost of those paths by fold."""

    reach: pd.DataFrame
    gaps: pd.DataFrame
    sets: dict[str, dict]
    paths: dict[str, pd.DataFrame]
    fold_stats: dict[str, pd.DataFrame]


@dataclass
class Assembly:
    """The assembled market (module docstring). ``folds`` has one row per fold, ``days`` one per test
    day (``fold``, ``in_Fk``, ``rf``: the risk-free accrual charged on the day), ``paths[pass]`` the
    daily returns of every path (columns ``"<arm>:<path>"``), ``fold_stats[pass]`` the turnover,
    traded notional and cost of every path and fold, ``doses[arm]`` the dose of ranks ``2…R+1`` (bp)
    by fold (NaN where the row has none), ``solvers`` the realised gap and rank of the real solvers,
    ``masks`` and ``fold_sets`` ``F_k`` per arm and ``"common"``, ``operators`` the descriptive
    operators (``None`` unless asked for)."""

    meta: dict
    folds: pd.DataFrame
    days: pd.DataFrame
    paths: dict[str, pd.DataFrame]
    fold_stats: dict[str, pd.DataFrame]
    doses: dict[str, pd.DataFrame]
    solvers: pd.DataFrame
    masks: dict[str, np.ndarray]
    fold_sets: dict[str, RankFoldSet]
    rank_ids: list[int] = field(default_factory=list)
    operators: OperatorAssembly | None = None


def _folds_frame(index: RowIndex, sets: dict[str, RankFoldSet]) -> pd.DataFrame:
    table, n = index.table, len(index.table.folds)
    records = []
    for fold in range(n):
        rec = table.folds[fold]
        out: dict = {
            "fold": fold, "formation_date": str(rec.formation_date.date()),
            "test_start": str(rec.test_start.date()), "test_end": str(rec.test_end.date()),
            "n_assets": len(rec.assets), **{f"x_{k}": v for k, v in rec.extra.items()},
        }
        first = next((index.rows[(arm, fold)] for arm in index.arms if (arm, fold) in index.rows),
                     None)
        for key in ("K", "K_used", "s_cont", "binding", "lower_bound_active", "depth", "cv_vol",
                    "cond_sigma"):
            out[key] = first[key] if first is not None else None
        for arm in index.arms:
            row = index.rows.get((arm, fold))
            out[f"status_{arm}"] = row["status"] if row is not None else "missing"
            out[f"in_Fk_{arm}"] = bool(sets[arm].mask[fold])
            out[f"optimum_certified_{arm}"] = bool(row["optimum_certified"]) if row else False
            out[f"s_cont_arm_{arm}"] = row["s_cont_arm"] if row else None
            out[f"f_star_{arm}"] = row["f_star"] if row else None
            out[f"n_distinct_{arm}"] = row["kbest_n_distinct"] if row else 0
            out[f"boundary_tie_{arm}"] = row["kbest_boundary_tie"] if row else None
            out[f"tie_share_{arm}"] = row["kbest_tie_share"] if row else None
            tol = row["kbest_tie_tol"] if row else None
            out[f"tie_tol_bp_{arm}"] = None if tol is None else tol * 1e4
        out["in_Fk"] = bool(sets["common"].mask[fold])
        records.append(out)
    return pd.DataFrame(records).set_index("fold", drop=False).rename_axis(None)


def _doses_frame(index: RowIndex, arm: str, rank_ids: list[int]) -> pd.DataFrame:
    n = len(index.table.folds)
    table = np.full((n, len(rank_ids)), np.nan)
    for fold in range(n):
        row = index.rows.get((arm, fold))
        if row is None:
            continue
        for j, rank in enumerate(rank_ids):
            if len(row["ranks"]) >= rank and row["ranks"][rank - 1]["rank"] == rank:
                table[fold, j] = row["ranks"][rank - 1]["dose_bp"]
    return pd.DataFrame(table, columns=rank_ids)


def _solvers_frame(index: RowIndex) -> pd.DataFrame:
    records = []
    for (arm, fold), row in sorted(index.rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        for name, s in (row.get("solvers") or {}).items():
            records.append({
                "arm": arm, "fold": fold, "solver": name, "gap_bp": s["gap_bp"], "rank": s["rank"],
                "rank_matched": s["rank_matched"], "rank_beyond": s["rank_beyond"],
                "objective": s["objective"], "wall_s": s["wall_s"], "budget_s": s["budget_s"],
            })
    cols = ["arm", "fold", "solver", "gap_bp", "rank", "rank_matched", "rank_beyond", "objective",
            "wall_s", "budget_s"]
    return pd.DataFrame(records, columns=cols)


def _cell_cost(index: RowIndex) -> dict:
    """Seconds spent by the cells, by part: the cost of the run the rows came from."""
    totals: dict[str, float] = {}
    for row in index.rows.values():
        for key, value in (row.get("times") or {}).items():
            if isinstance(value, (int, float)):  # the per-MIQP and per-level lists are not summed
                totals[key] = totals.get(key, 0.0) + float(value)
    return totals


_OPERATORS = ("swap", "perturb")
_STATUSES = ("reached", "unreachable", "failed", "timeout", "not_built")


def _reach_matrix(
    index: RowIndex, arm: str, op: str
) -> tuple[list[float], np.ndarray, np.ndarray, dict[int, dict]]:
    """The levels of one arm and operator, ``reached[fold, level]``, the folds the operator ran on (a
    mask) and their rows."""
    n_folds = len(index.table.folds)
    ran = {fold: row for (a, fold), row in index.rows.items() if a == arm and row.get(op)}
    if not ran:
        raise AssemblyError(f"{arm}: no row carries the operator {op!r}: configure it in the cells "
                            "(params.operators)")
    first = next(iter(ran.values()))[op]
    levels = [float(lv["gap_target_bp"]) for lv in first]
    reached = np.zeros((n_folds, len(levels)), dtype=bool)
    ran_mask = np.zeros(n_folds, dtype=bool)
    for fold, row in ran.items():
        if [float(lv["gap_target_bp"]) for lv in row[op]] != levels:
            raise AssemblyError(f"{arm} fold {fold}: the levels of {op!r} differ from the other rows'")
        ran_mask[fold] = True
        reached[fold] = [lv["status"] == "reached" for lv in row[op]]
    return levels, reached, ran_mask, ran


@dataclass(frozen=True)
class PilotOperators:
    """Where ``L`` of the descriptive operators comes from: the reach of (б) and (а) on the folds of the
    PILOT period (plan §2: "L — уровни, достигнутые хотя бы на 80 % фолдов пилотного периода"), read from
    a registered run of ``configs/E3/pilot_operators.yaml`` by :func:`read_pilot_operators`.

    ``reach[(arm, operator)] = (levels, reached)``: ``reached[i, j]`` says whether level ``levels[j]``
    was reached on the ``i``-th pilot fold the operator ran on (a fold with a certified optimum and its
    rank). The rest is what :func:`assemble` checks against the confirmatory run: ``lam``, ``market`` and
    ``operators`` (the ``params`` of the run) must be the same, and the test days of the pilot
    (``test_start`` … ``test_end``) must end before the confirmation begins."""

    run_id: str
    lam: float
    market: dict
    operators: dict
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    reach: dict[tuple[str, str], tuple[tuple[float, ...], np.ndarray]]


def read_pilot_operators(
    cfg: dict, rows: Sequence[dict], *, run_id: str, lam: float | None = None,
    allow_incomplete: bool = False,
) -> PilotOperators:
    """The reach of the operators on the folds of a pilot run (``cfg`` its config, ``rows`` its rows,
    checked against the fold table as for any run). The run must be a calibration run: the pilot period
    is where the plan's pilot values are read, and test seeds are for the confirmation."""
    p = e3.check_params(cfg)
    if p["seed_role"] != "calibration":
        raise AssemblyError(f"the pilot of the operators ({run_id}) must be a calibration run (solver seeds "
                            f"10 000-19 999, plan §3), not {p['seed_role']!r}")
    ops = p.get("operators")
    if ops is None:
        raise AssemblyError(f"the pilot of the operators ({run_id}) has no params.operators")
    index = index_rows(cfg, rows, lam=lam, allow_incomplete=allow_incomplete)
    reach = {}
    for arm in ops["arms"]:
        for op in _OPERATORS:
            if ops.get(op) is None:
                continue
            levels, reached, ran_mask, _ = _reach_matrix(index, arm, op)
            reach[(arm, op)] = (tuple(levels), reached[ran_mask])
    folds = index.table.folds
    return PilotOperators(
        run_id=str(run_id), lam=index.lam, market=copy.deepcopy(p["market"]),
        operators=copy.deepcopy(ops), test_start=folds[0].test_start, test_end=folds[-1].test_end,
        reach=reach,
    )


def _given_levels(spec: dict, arm: str) -> tuple[float, ...] | None:
    """The levels ``L`` that the config gives for ``arm`` (a list for every arm, or a mapping arm → list),
    or ``None``: read them on the pilot folds (``null`` or ``pilot``, the default)."""
    given = spec.get("levels_L")
    if isinstance(given, dict):
        given = given.get(arm)
    if given is None or (isinstance(given, str) and given == "pilot"):
        return None
    if isinstance(given, str):
        raise AssemblyError(f"levels_L must be 'pilot' (the default: read on the pilot folds), a list of "
                            f"levels or a mapping arm -> list, got {given!r}")
    return tuple(sorted(float(x) for x in given))


def _needs_pilot(ops: dict) -> bool:
    """Does any arm and operator of the assembly read its ``L`` on the pilot folds?"""
    return any(_given_levels(ops[op], arm) is None
               for op in _OPERATORS if ops.get(op) is not None for arm in ops["arms"])


def _op_spec(spec: dict | None) -> dict | None:
    if spec is None:
        return None
    return {k: ([float(x) for x in v] if k == "levels" else v) for k, v in spec.items()}


def _check_pilot(pilot: PilotOperators, index: RowIndex, p: dict, ops: dict) -> None:
    """The pilot is the pilot of THIS confirmation: λ_f, market and operators the same, and over before it."""
    where = f"the pilot of the operators ({pilot.run_id})"
    if not _close(pilot.lam, index.lam):
        raise AssemblyError(f"{where} ran at lambda = {pilot.lam:g}, the confirmatory run at "
                            f"{index.lam:g}: L is read at lambda_f")

    def labelled(market: dict) -> dict:
        return {k: v for k, v in market.items() if k != "name"}

    if labelled(pilot.market) != labelled(p["market"]):
        raise AssemblyError(f"{where} ran on another market than the confirmatory run: "
                            f"{labelled(pilot.market)} against {labelled(p['market'])}")
    first = index.table.folds[0].test_start
    if not pilot.test_end < first:
        raise AssemblyError(f"{where} ends on {pilot.test_end.date()}, not before the first confirmatory test "
                            f"day {first.date()}: L is read on the folds of the PILOT period")
    confirmation = p.get("operators") or {}
    for op in _OPERATORS:
        if ops.get(op) is None:
            continue
        mine, theirs = _op_spec(pilot.operators.get(op)), _op_spec(confirmation.get(op))
        if mine != theirs:
            raise AssemblyError(f"{where} ran the operator {op!r} as {mine}, the confirmation as {theirs}: "
                                "'reached' (all replicas found, with these levels and limits) would mean "
                                "something else")


def _operator_sets(
    index: RowIndex, arm: str, op: str, spec: dict, pilot: PilotOperators | None = None
) -> tuple[dict, dict[int, dict]]:
    """``L``, ``L_a`` and the masks of one arm and operator, and the rows the operator ran on. ``L`` is
    given by the config (``levels_L``) or read on the pilot folds (module docstring); the masks are those
    of the confirmatory folds."""
    levels, reached, ran_mask, ran = _reach_matrix(index, arm, op)
    given = _given_levels(spec, arm)
    extra: dict = {}
    if given is not None:
        if not set(given) <= set(levels):
            raise AssemblyError(f"levels_L {given} is not among the levels {levels} of the rows")
        bigl, source = given, "config"
    else:
        if pilot is None:
            raise AssemblyError(
                f"{arm}: the levels L of {op!r} are read on the folds of the PILOT period (plan §2 (б)): "
                "give the pilot run of the operators (sources.pilot_operators of the analysis config) or "
                "levels_L")
        entry = pilot.reach.get((arm, op))
        if entry is None:
            raise AssemblyError(f"the pilot run {pilot.run_id} has no operator {op!r} on the arm {arm}")
        p_levels, p_reached = entry
        if list(p_levels) != levels:
            raise AssemblyError(f"the pilot ran the levels {list(p_levels)} of {op!r} on {arm}, the "
                                f"confirmation {levels}")
        bigl = fs.confirmatory_levels(p_reached, p_levels, float(spec["min_share"]))
        source = "pilot"
        extra = {"pilot_run": pilot.run_id, "pilot_n_ran": int(p_reached.shape[0]),
                 "pilot_share": [float(x) for x in p_reached.mean(axis=0)]}
    ls = fs.level_sets(bigl, float(spec["g_star_bp"]))
    folds = fs.fold_sets(reached, levels, ls)
    return {
        "levels_all": levels, "L": list(ls.levels), "L_a": list(ls.levels_a),
        "g_star_bp": float(spec["g_star_bp"]), "min_share": float(spec["min_share"]),
        "mask_a": folds.mask_a, "mask_b": folds.mask_b, "n_ran": int(ran_mask.sum()),
        "ran": ran_mask, "L_source": source, **extra,
    }, ran


def _operator_tables(arm: str, op: str, ran: dict[int, dict]) -> tuple[list, list]:
    reach, gaps = [], []
    first = next(iter(ran.values()))[op]
    for j, lv0 in enumerate(first):
        g = float(lv0["gap_target_bp"])
        levels = [row[op][j] for row in ran.values()]
        counts = {s: sum(1 for lv in levels if lv["status"] == s) for s in _STATUSES}
        evals = [lv.get("n_evals") for lv in levels if lv.get("n_evals") is not None]
        reach.append({
            "arm": arm, "operator": op, "level": g, "n_ran": len(levels),
            "n_reached": counts["reached"], "share_reached": counts["reached"] / len(levels),
            "n_unreachable": counts["unreachable"], "n_failed": counts["failed"],
            "n_timeout": counts["timeout"], "n_not_built": counts["not_built"],
            "mean_n_evals": float(np.mean(evals)) if evals else None,
            "max_n_evals": int(max(evals)) if evals else None,
        })
        for fold, row in ran.items():
            lv = row[op][j]
            for r, rep in enumerate(lv["replicas"]):
                gaps.append({"arm": arm, "operator": op, "level": g, "fold": fold, "replica": r,
                             "gap_bp": rep["gap_bp"], "level_reached": lv["status"] == "reached"})
    return reach, gaps


def _assemble_operators(
    index: RowIndex, ops: dict, passes: dict[str, dict], workers: int, p: dict, days_index,
    pilot: PilotOperators | None = None,
) -> OperatorAssembly:
    table, n_folds = index.table, len(index.table.folds)
    if pilot is not None:
        _check_pilot(pilot, index, p, ops)
    reach_rows, gap_rows, sets = [], [], {}
    tasks = []
    for arm in ops["arms"]:
        opt = []
        for fold in range(n_folds):
            row, n = index.rows.get((arm, fold)), len(table.folds[fold].assets)
            w = _optimum_vector(row, n) if row is not None else None
            opt.append(w if w is not None else np.full(n, 1.0 / n))
        for op in _OPERATORS:
            spec = ops.get(op)
            if spec is None:
                continue
            info, ran = _operator_sets(index, arm, op, spec, pilot)
            sets[f"{arm}:{op}"] = info
            r, g = _operator_tables(arm, op, ran)
            reach_rows += r
            gap_rows += g
            mask_b = info["mask_b"]
            paths = {"opt": opt}
            n_rep = max((lv["n_requested"] for row in ran.values() for lv in row[op]), default=0)
            for level in info["L"]:
                j = info["levels_all"].index(level)
                for r_i in range(n_rep):
                    vecs = []
                    for fold in range(n_folds):
                        if mask_b[fold]:
                            rep = index.rows[(arm, fold)][op][j]["replicas"][r_i]
                            vecs.append(_vec(len(table.folds[fold].assets), rep["support"], rep["w"]))
                        else:
                            vecs.append(opt[fold])
                    paths[f"g{int(level)}:r{r_i:02d}"] = vecs
            for pass_name in spec["passes"]:
                if pass_name not in passes:
                    raise AssemblyError(f"operators.{op}.passes names {pass_name!r}, which is not "
                                        f"among the passes {sorted(passes)}")
                pspec = passes[pass_name]
                for name, vecs in paths.items():
                    tasks.append({
                        "key": (f"{op}:{pass_name}", arm, name), "name": f"{arm}:{name}",
                        "weights": vecs, "tc_bps": float(pspec["tc_bps"]),
                        "execution_lag": int(pspec.get("execution_lag", 0)),
                        "terminal_return": float(pspec.get("terminal_return", 0.0)),
                    })
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=_init_worker,
                                 initargs=(1,)) as pool:
            futures = [pool.submit(_run_task, p["market"], p["period"], t) for t in tasks]
            results = [f.result() for f in futures]
    else:
        results = [_run_task(p["market"], p["period"], t) for t in tasks]
    paths_out: dict[str, dict[str, np.ndarray]] = {}
    stats: dict[str, list[dict]] = {}
    for res in results:
        if not res["index"].equals(days_index):
            raise AssemblyError(f"the days of {res['key']} differ from the other paths'")
        name, arm, path = res["key"]
        paths_out.setdefault(name, {})[f"{arm}:{path}"] = res["daily"]
        stats.setdefault(name, []).extend(
            {"arm": arm, "path": path, "fold": fold, "turnover": turnover, "traded": traded,
             "cost": cost}
            for fold, turnover, traded, cost, _, _ in res["folds"]
        )
    reach_cols = ["arm", "operator", "level", "n_ran", "n_reached", "share_reached", "n_unreachable",
                  "n_failed", "n_timeout", "n_not_built", "mean_n_evals", "max_n_evals"]
    gap_cols = ["arm", "operator", "level", "fold", "replica", "gap_bp", "level_reached"]
    return OperatorAssembly(
        reach=pd.DataFrame(reach_rows, columns=reach_cols), gaps=pd.DataFrame(gap_rows, columns=gap_cols),
        sets=sets, paths={k: pd.DataFrame(v, index=days_index) for k, v in paths_out.items()},
        fold_stats={k: pd.DataFrame(v) for k, v in stats.items()},
    )


def assemble(
    cfg: dict, rows: Sequence[dict], passes: dict[str, dict], *, lam: float | None = None,
    workers: int = 1, allow_incomplete: bool = False, operators: dict | None = None,
    pilot_operators: PilotOperators | None = None,
) -> Assembly:
    """Index and check the rows, build ``F_k`` and run every pass (module docstring). ``operators``
    (``{"arms": [...], "swap": {...}, "perturb": {...}}``, see :class:`OperatorAssembly`) adds the
    paths of the descriptive operators (б) and (а); their ``L`` is read from ``pilot_operators`` (the
    reach on the pilot folds, :func:`read_pilot_operators`) unless the operator's ``levels_L`` gives it."""
    p = e3.check_params(cfg)
    index = index_rows(cfg, rows, lam=lam, allow_incomplete=allow_incomplete)
    n_ranks = int(p["n_ranks"])
    rank_ids = list(range(2, n_ranks + 2))
    sets = fold_sets(index, n_ranks)
    table, n_folds = index.table, len(index.table.folds)
    masks = {key: s.mask.copy() for key, s in sets.items()}
    used = masks["common"]
    solvers = {arm: _solver_names(index, arm, used) for arm in index.arms}

    weights: dict[tuple[str, str], list[np.ndarray]] = {}

    def vectors(arm: str, path: str) -> list[np.ndarray]:
        if (arm, path) not in weights:
            out = []
            for fold in range(n_folds):
                row, n = index.rows.get((arm, fold)), len(table.folds[fold].assets)
                out.append(_path_vector(row, path, n) if used[fold] else _fallback_vector(row, n))
            weights[(arm, path)] = out
        return weights[(arm, path)]

    tasks = []
    for name, spec in passes.items():
        for arm in index.arms:
            for path in _paths_of(spec.get("paths", "all"), rank_ids, solvers[arm]):
                tasks.append({
                    "key": (name, arm, path), "name": f"{arm}:{path}", "weights": vectors(arm, path),
                    "tc_bps": float(spec["tc_bps"]), "execution_lag": int(spec.get("execution_lag", 0)),
                    "terminal_return": float(spec.get("terminal_return", 0.0)),
                })
    if int(workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=_init_worker,
                                 initargs=(1,)) as pool:
            futures = [pool.submit(_run_task, p["market"], p["period"], t) for t in tasks]
            results = [f.result() for f in futures]
    else:
        results = [_run_task(p["market"], p["period"], t) for t in tasks]
    if not results:
        raise AssemblyError("no passes to run")

    reference = results[0]
    days_index = reference["index"]
    day_fold = np.empty(len(days_index), dtype=int)
    for fold, _, _, _, start, end in reference["folds"]:
        day_fold[(days_index >= start) & (days_index <= end)] = fold
    paths: dict[str, dict[str, np.ndarray]] = {}
    stats: dict[str, list[dict]] = {}
    for res in results:
        if not res["index"].equals(days_index):
            raise AssemblyError(f"the days of {res['key']} differ from the other paths'")
        name, arm, path = res["key"]
        paths.setdefault(name, {})[f"{arm}:{path}"] = res["daily"]
        stats.setdefault(name, []).extend(
            {"arm": arm, "path": path, "fold": fold, "turnover": turnover, "traded": traded,
             "cost": cost}
            for fold, turnover, traded, cost, _, _ in res["folds"]
        )
    # the risk-free accrual charged on each test day, on the whole calendar of the run
    accrual = -excess_returns(pd.Series(0.0, index=days_index), table.market.rf).to_numpy()
    days = pd.DataFrame({"fold": day_fold, "in_Fk": used[day_fold], "rf": accrual}, index=days_index)
    folds = _folds_frame(index, sets)
    meta = {
        "market": p["market"], "period": p["period"], "arms": list(index.arms), "lam": index.lam,
        "n_ranks": n_ranks, "n_folds": n_folds, "n_missing": index.n_missing,
        "missing": [list(m) for m in index.missing],
        "passes": {k: dict(v) for k, v in passes.items()},
        "solvers": solvers, "workers": int(workers),
        "fold_sets": {
            key: {"n_included": s.n_included, "n_excluded": s.n_excluded,
                  "n_excluded_optimum": s.n_excluded_optimum,
                  "n_excluded_degenerate": s.n_excluded_degenerate,
                  "n_excluded_rank": s.n_excluded_rank}
            for key, s in sets.items()
        },
        "n_days": len(days_index), "n_days_Fk": int(used[day_fold].sum()),
        "cell_seconds": _cell_cost(index),
    }
    operator_assembly = (
        _assemble_operators(index, operators, passes, workers, p, days_index, pilot_operators)
        if operators else None
    )
    return Assembly(
        meta=d.json_safe(meta), folds=folds, days=days,
        paths={k: pd.DataFrame(v, index=days_index) for k, v in paths.items()},
        fold_stats={k: pd.DataFrame(v) for k, v in stats.items()},
        doses={arm: _doses_frame(index, arm, rank_ids) for arm in index.arms},
        solvers=_solvers_frame(index), masks=masks, fold_sets=sets, rank_ids=rank_ids,
        operators=operator_assembly,
    )


# --------------------------------------------------------------------------------------------
# What the statistics get
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FkSeries:
    """The daily series of one arm and pass on the days of ``F_k``: ``opt`` ``(T,)``, ``ranks``
    ``(T, len(rank_ids))``, one ``(T,)`` series per real solver, and the dates."""

    opt: np.ndarray
    ranks: np.ndarray
    rank_ids: list[int]
    solvers: dict[str, np.ndarray]
    dates: pd.DatetimeIndex
    functional: str


def fk_series(
    asm: Assembly, pass_name: str, arm: str, functional: str, ranks: Sequence[int] | None = None,
    fold_mask: np.ndarray | None = None,
) -> FkSeries:
    """The balanced panel of ``arm`` in ``pass_name`` on the days of ``F_k`` (module docstring):
    excess returns for ``"sharpe"``, returns for ``"logvar"``. ``ranks`` picks ranks (default all that
    the pass holds). ``fold_mask`` (one flag per fold) restricts ``F_k`` further, for a sensitivity
    (e.g. the folds whose index members are covered by data): the days of the other folds drop out of
    every path alike."""
    if functional not in FUNCTIONALS:
        raise ValueError(f"functional must be one of {FUNCTIONALS}, got {functional!r}")
    frame = asm.paths[pass_name]
    mask = asm.days["in_Fk"].to_numpy()
    if fold_mask is not None:
        flags = np.asarray(fold_mask, dtype=bool)
        if flags.shape != (len(asm.folds),):
            raise ValueError(f"fold_mask needs one flag per fold ({len(asm.folds)}), got {flags.shape}")
        mask = mask & flags[asm.days["fold"].to_numpy()]
    shift = asm.days["rf"].to_numpy()[mask] if functional == "sharpe" else 0.0

    def column(path: str) -> np.ndarray:
        return frame[f"{arm}:{path}"].to_numpy()[mask] - shift

    available = [r for r in asm.rank_ids if f"{arm}:rank_{r}" in frame.columns]
    chosen = available if ranks is None else [r for r in ranks if r in available]
    solvers = [c.split("solver_", 1)[1] for c in frame.columns if c.startswith(f"{arm}:solver_")]
    return FkSeries(
        opt=column("opt"),
        ranks=np.column_stack([column(f"rank_{r}") for r in chosen]) if chosen
        else np.empty((int(mask.sum()), 0)),
        rank_ids=chosen,
        solvers={s: column(f"solver_{s}") for s in solvers},
        dates=asm.days.index[mask],
        functional=functional,
    )


# --------------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------------


def save_assembly(asm: Assembly, out_dir: Path | str) -> Path:
    """Write the assembly to ``out_dir`` (parquet for the arrays, ``meta.json``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {**asm.meta, "rank_ids": asm.rank_ids}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    asm.folds.to_parquet(out / "folds.parquet")
    asm.days.to_parquet(out / "days.parquet")
    asm.solvers.to_parquet(out / "solvers.parquet")
    for name, frame in asm.paths.items():
        frame.to_parquet(out / f"paths_{name}.parquet")
    for name, frame in asm.fold_stats.items():
        frame.to_parquet(out / f"foldstats_{name}.parquet")
    for arm, frame in asm.doses.items():
        frame.rename(columns=str).to_parquet(out / f"doses_{arm}.parquet")
    if asm.operators is not None:
        ops_dir = out / "operators"
        ops_dir.mkdir(exist_ok=True)
        asm.operators.reach.to_parquet(ops_dir / "reach.parquet")
        asm.operators.gaps.to_parquet(ops_dir / "gaps.parquet")
        sets = {
            k: {**{kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)},
                "mask_a": v["mask_a"].astype(int).tolist(), "mask_b": v["mask_b"].astype(int).tolist(),
                "ran": v["ran"].astype(int).tolist()}
            for k, v in asm.operators.sets.items()
        }
        (ops_dir / "sets.json").write_text(json.dumps(sets), encoding="utf-8")
        for name, frame in asm.operators.paths.items():
            frame.to_parquet(ops_dir / f"paths_{name.replace(':', '_')}.parquet")
        for name, frame in asm.operators.fold_stats.items():
            frame.to_parquet(ops_dir / f"foldstats_{name.replace(':', '_')}.parquet")
    return out


def load_assembly(src: Path | str) -> Assembly:
    """The inverse of :func:`save_assembly`."""
    src = Path(src)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    rank_ids = [int(r) for r in meta.pop("rank_ids")]
    folds = pd.read_parquet(src / "folds.parquet")
    arms = meta["arms"]
    masks = {arm: folds[f"in_Fk_{arm}"].to_numpy(dtype=bool) for arm in arms}
    masks["common"] = folds["in_Fk"].to_numpy(dtype=bool)
    sets = {key: RankFoldSet(masks[key], v["n_excluded_optimum"], v["n_excluded_degenerate"],
                             v["n_excluded_rank"]) for key, v in meta["fold_sets"].items()}
    operators = None
    if (src / "operators").is_dir():
        ops_dir = src / "operators"
        raw = json.loads((ops_dir / "sets.json").read_text(encoding="utf-8"))
        sets_ops = {
            k: {**v, "mask_a": np.array(v["mask_a"], dtype=bool), "mask_b": np.array(v["mask_b"], dtype=bool),
                "ran": np.array(v["ran"], dtype=bool)}
            for k, v in raw.items()
        }
        names = sorted(f.stem.removeprefix("paths_") for f in ops_dir.glob("paths_*.parquet"))
        operators = OperatorAssembly(
            reach=pd.read_parquet(ops_dir / "reach.parquet"), gaps=pd.read_parquet(ops_dir / "gaps.parquet"),
            sets=sets_ops,
            paths={n.replace("_", ":", 1): pd.read_parquet(ops_dir / f"paths_{n}.parquet") for n in names},
            fold_stats={n.replace("_", ":", 1): pd.read_parquet(ops_dir / f"foldstats_{n}.parquet")
                        for n in names},
        )
    return Assembly(
        meta=meta, folds=folds, days=pd.read_parquet(src / "days.parquet"),
        paths={f.stem.removeprefix("paths_"): pd.read_parquet(f) for f in sorted(src.glob("paths_*.parquet"))},
        fold_stats={f.stem.removeprefix("foldstats_"): pd.read_parquet(f)
                    for f in sorted(src.glob("foldstats_*.parquet"))},
        doses={arm: pd.read_parquet(src / f"doses_{arm}.parquet").rename(columns=int) for arm in arms},
        solvers=pd.read_parquet(src / "solvers.parquet"), masks=masks, fold_sets=sets,
        rank_ids=rank_ids, operators=operators,
    )


# --------------------------------------------------------------------------------------------
# The registered stage
# --------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else registry.REPO_ROOT / p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e3_assemble")
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--source", required=True, help="a key of sources: in the analysis config")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None, help="overrides assembly.workers")
    parser.add_argument("--lam", type=float, default=None, help="the λ of a pilot-style run")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="exploratory assembly on a dirty tree (not for the paper)")
    args = parser.parse_args(argv)

    acfg = yaml.safe_load(args.analysis_config.read_text(encoding="utf-8"))
    src = acfg["sources"][args.source]
    cfg_path, run_dir = _resolve(src["config"]), _resolve(src["run"])
    manifest = check_lineage(run_dir, cfg_path)
    cfg = effective_config(load_config(cfg_path))
    assembly_cfg = {**acfg["assembly"], **(src.get("assembly") or {})}
    rows_path = run_dir / "results.jsonl"
    allow_incomplete = bool(assembly_cfg.get("allow_incomplete"))
    inputs = {"rows": rows_path}
    pilot, pilot_record = None, None
    ops = assembly_cfg.get("operators")
    if ops and _needs_pilot(ops):  # the plan reads L on the folds of the pilot period: look at that run first
        psrc = acfg["sources"].get("pilot_operators")
        if psrc is None:
            raise AssemblyError(
                "the levels L of the descriptive operators are read on the folds of the PILOT period "
                "(plan §2 (б)): add sources.pilot_operators (the registered run of "
                "configs/E3/pilot_operators.yaml) to the analysis config, or give levels_L")
        pcfg_path, prun = _resolve(psrc["config"]), _resolve(psrc["run"])
        pmanifest = check_lineage(prun, pcfg_path)
        pilot = read_pilot_operators(effective_config(load_config(pcfg_path)), load_rows(prun),
                                     run_id=prun.name, allow_incomplete=allow_incomplete)
        inputs["pilot_operator_rows"] = prun / "results.jsonl"
        pilot_record = {
            "config": str(psrc["config"]), "run_id": prun.name, "config_hash": pmanifest["config_hash"],
            "git_sha": pmanifest.get("git_sha"), "rows_sha256": _sha256(prun / "results.jsonl"),
            "lam": pilot.lam,
        }
    record = {
        "analysis_config": str(args.analysis_config), "source": args.source,
        "source_config": str(src["config"]), "source_run_id": run_dir.name,
        "source_config_hash": manifest["config_hash"], "source_git_sha": manifest.get("git_sha"),
        "rows_sha256": _sha256(rows_path), "assembly": assembly_cfg, "lam": args.lam,
        "pilot_operators": pilot_record,
    }
    workers = int(args.workers if args.workers is not None else assembly_cfg.get("workers", 1))
    with registry.start_run(
        "E3", f"assemble-{args.source}", record, runs_dir=args.runs_dir,
        inputs=inputs, require_clean=not args.allow_dirty,
    ) as run:
        asm = assemble(cfg, load_rows(run_dir), assembly_cfg["passes"], lam=args.lam,
                       workers=workers, allow_incomplete=allow_incomplete, operators=ops,
                       pilot_operators=pilot)
        asm.meta.update(source_run_id=run_dir.name, source=args.source, rows_sha256=record["rows_sha256"],
                        pilot_operators_run_id=None if pilot is None else pilot.run_id)
        save_assembly(asm, run.run_dir)
        run.manifest["mode"] = "assembly"
        print(f"{run.run_id}: {asm.meta['n_folds']} folds, "
              f"{asm.meta['fold_sets']['common']['n_included']} in F_k, "
              f"{len(asm.paths)} passes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
