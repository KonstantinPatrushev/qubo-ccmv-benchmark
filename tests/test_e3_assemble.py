"""E3 assembly (``cif.experiments.e3_assemble``): from the rows of the cells to the paths.

ANALYSIS_PLAN §2 RQ3: ``F_k`` (common to the two FF49 arms), the rank paths and the solver paths run
through the engine, "до издержек" = a pass with ``tc = 0`` and everything else as in the main pass,
the days of the folds outside ``F_k`` dropped from every path. The rows come from a registered run of
the synthetic market (``tests/e3_fixtures.py``); the expected numbers are computed by hand from the
stored weights and the prices.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cif.backtest.fold_sets import rank_fold_set
from cif.experiments import e3_assemble as a
from cif.experiments import e3_data as d
from cif.experiments import e3_walkforward as e3
from cif.experiments import run
from cif.metrics.performance import excess_returns
from tests.e3_fixtures import base_config

PASSES = {
    "gross": {"tc_bps": 0.0, "paths": "all"},
    "net": {"tc_bps": 10.0, "paths": "all"},
    "net30": {"tc_bps": 30.0, "paths": "core"},
    "gross_lag1": {"tc_bps": 0.0, "execution_lag": 1, "paths": "core"},
}
SOLVERS = {
    "tuned": True, "common_start": True, "arms": ["MV"],
    "scip": {"formulation": "perspective", "diag": "mineig", "budget_s": 5.0, "mip_gap": 1e-6,
             "scip_params": None},
    "ls": {"budget_s": 0.2, "improvement": "first", "num_restarts": None, "rcl_size": 3},
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    d.clear_caches()
    yield
    d.clear_caches()


def _config() -> dict:
    cfg = base_config(solvers=copy.deepcopy(SOLVERS))
    cfg["params"]["period"] = {"name": "confirmatory", "start": "2000-06-01", "end": "2000-08-31"}
    return cfg


@pytest.fixture(scope="module")
def registered(tmp_path_factory):
    """A registered run of every cell of the small synthetic market: the config, its path, the run."""
    d.clear_caches()
    tmp = tmp_path_factory.mktemp("e3run")
    cfg = _config()
    path = tmp / "e3.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp / "runs"
    assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    return cfg, path, run_dir


@pytest.fixture(scope="module")
def rows(registered):
    return a.load_rows(registered[2])


@pytest.fixture(scope="module")
def asm(registered, rows):
    d.clear_caches()
    return a.assemble(registered[0], rows, PASSES, workers=1)


# --------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------


def test_rows_load_from_a_run_directory_and_from_the_file(registered):
    cfg, _, run_dir = registered
    by_dir, by_file = a.load_rows(run_dir), a.load_rows(run_dir / "results.jsonl")
    assert by_dir == by_file and len(by_dir) == len(e3.cells(cfg))


def test_the_lineage_of_the_run_is_the_config_hash(registered):
    cfg, path, run_dir = registered
    a.check_lineage(run_dir, path)  # the YAML the run was made with
    other = Path(str(path) + ".other.yaml")
    changed = copy.deepcopy(cfg)
    changed["params"]["n_ranks"] = 5
    other.write_text(yaml.safe_dump(changed))
    with pytest.raises(a.AssemblyError, match="config"):
        a.check_lineage(run_dir, other)


def test_index_rows_checks_every_row_against_the_fold_table(registered, rows):
    cfg = registered[0]
    index = a.index_rows(cfg, rows)
    n_folds = len(index.table.folds)
    assert set(index.rows) == {(arm, f) for arm in ("MV", "GMV") for f in range(n_folds)}
    assert index.lam == 5.0 and index.n_missing == 0

    def broken(i, **change):
        out = copy.deepcopy(rows)
        out[i].update(change)
        return out

    with pytest.raises(a.AssemblyError, match="missing"):
        a.index_rows(cfg, rows[:-1])
    with pytest.raises(a.AssemblyError, match="error"):
        a.index_rows(cfg, broken(3, error="RuntimeError: boom", traceback="..."))
    check = dict(rows[2]["problem_check"], mu_sum=rows[2]["problem_check"]["mu_sum"] + 1.0)
    with pytest.raises(a.AssemblyError, match="estimate"):
        a.index_rows(cfg, broken(2, problem_check=check))
    with pytest.raises(a.AssemblyError, match="universe"):
        a.index_rows(cfg, broken(2, assets=list(reversed(rows[2]["assets"]))))
    with pytest.raises(a.AssemblyError, match="duplicate"):
        a.index_rows(cfg, [*rows, rows[0]])
    with pytest.raises(a.AssemblyError, match="not in the grid"):
        a.index_rows(cfg, [*rows, {**rows[0], "fold": 999, "cell_key": "x"}])


def test_incomplete_rows_can_be_allowed_and_are_counted(registered, rows):
    cfg = registered[0]
    index = a.index_rows(cfg, rows[:-2], allow_incomplete=True)
    assert index.n_missing == 2
    asm = a.assemble(cfg, rows[:-2], {"gross": PASSES["gross"]}, allow_incomplete=True)
    assert asm.meta["n_missing"] == 2
    # the missing cells' folds are out of F_k, as if uncertified
    missing_folds = {r["fold"] for r in rows[-2:]}
    assert not asm.masks["common"][sorted(missing_folds)].any()


# --------------------------------------------------------------------------------------------
# F_k
# --------------------------------------------------------------------------------------------


def test_fold_sets_are_built_from_the_flags_of_each_arm_and_combined(registered, rows):
    cfg = registered[0]
    n_ranks = cfg["params"]["n_ranks"]

    def flags(arm):
        by_fold = {r["fold"]: r for r in rows if r["arm"] == arm}
        n = len(by_fold)
        return (
            np.array([by_fold[f]["optimum_certified"] for f in range(n)]),
            np.array([by_fold[f]["kbest_all_certified"] for f in range(n)]),
            np.array([by_fold[f]["kbest_n_distinct"] >= n_ranks + 1 for f in range(n)]),
        )

    index = a.index_rows(cfg, rows)
    sets = a.fold_sets(index, n_ranks)
    for arm in ("MV", "GMV"):
        opt, ranks, distinct = flags(arm)
        np.testing.assert_array_equal(sets[arm].mask, rank_fold_set(opt, ranks, distinct).mask)
    mo, mr, md = flags("MV")
    go, gr, gd = flags("GMV")
    both = rank_fold_set(mo & go, np.column_stack([mr, gr]), md & gd)
    np.testing.assert_array_equal(sets["common"].mask, both.mask)
    np.testing.assert_array_equal(sets["common"].mask, sets["MV"].mask & sets["GMV"].mask)


def test_one_arm_failing_removes_the_fold_from_both_paths(registered, rows):
    cfg = registered[0]
    fold = 4
    tampered = copy.deepcopy(rows)
    for r in tampered:
        if r["arm"] == "GMV" and r["fold"] == fold:
            r.update(kbest_all_certified=False, status="kbest_incomplete")
    sets = a.fold_sets(a.index_rows(cfg, tampered), cfg["params"]["n_ranks"])
    assert sets["MV"].mask[fold] and not sets["GMV"].mask[fold]
    assert not sets["common"].mask[fold]  # F_k is common to the arms
    assert sets["common"].n_excluded >= 1 and sets["GMV"].n_excluded_rank >= 1


def test_us_style_single_arm_uses_the_fold_set_of_that_arm(registered, rows):
    cfg = copy.deepcopy(registered[0])
    cfg["params"]["arms"] = ["MV"]
    mv_rows = [r for r in rows if r["arm"] == "MV"]
    sets = a.fold_sets(a.index_rows(cfg, mv_rows), cfg["params"]["n_ranks"])
    assert set(sets) == {"MV", "common"}
    np.testing.assert_array_equal(sets["common"].mask, sets["MV"].mask)


# --------------------------------------------------------------------------------------------
# Paths through the engine
# --------------------------------------------------------------------------------------------


def _weights(row_rank: dict, n: int) -> np.ndarray:
    w = np.zeros(n)
    w[row_rank["support"]] = row_rank["w"]
    return w


def _manual_returns(table, fold: int, w: np.ndarray) -> np.ndarray:
    """Buy and hold from the formation close: the daily returns of the five test days."""
    rec = table.folds[fold]
    assets = list(rec.assets)
    prices = table.market.prices
    window = prices.loc[rec.test_start : rec.test_end, assets].to_numpy()
    start = prices.loc[rec.formation_date, assets].to_numpy()
    value = (window / start) @ w
    return np.diff(np.concatenate([[1.0], value])) / np.concatenate([[1.0], value[:-1]])


def test_the_paths_are_the_engine_on_the_stored_weights(asm, rows):
    table = d.get_fold_table(asm.meta["market"], asm.meta["period"])
    by_cell = {(r["arm"], r["fold"]): r for r in rows}
    gross = asm.paths["gross"]
    days = asm.days
    for arm in ("MV", "GMV"):
        for fold in (0, 5, 8):
            if not asm.masks["common"][fold]:
                continue
            row = by_cell[(arm, fold)]
            n = row["n_assets"]
            sel = (days["fold"] == fold).to_numpy()
            for name, rank in (("opt", 1), ("rank_2", 2), ("rank_4", 4)):
                w = _weights(row["ranks"][rank - 1], n)
                np.testing.assert_allclose(gross.loc[sel, f"{arm}:{name}"].to_numpy(),
                                           _manual_returns(table, fold, w), rtol=0, atol=1e-9)
    # a solver path holds the portfolio the solver returned
    row = by_cell[("MV", 5)]
    w = np.zeros(row["n_assets"])
    w[row["solvers"]["ls"]["idx"]] = row["solvers"]["ls"]["w"]
    sel = (days["fold"] == 5).to_numpy()
    np.testing.assert_allclose(gross.loc[sel, "MV:solver_ls"].to_numpy(),
                               _manual_returns(table, 5, w), atol=1e-9)


def test_the_columns_are_the_arms_and_their_paths(asm):
    cols = list(asm.paths["gross"].columns)
    assert "MV:opt" in cols and "GMV:opt" in cols
    assert [c for c in cols if c.startswith("MV:rank_")] == ["MV:rank_2", "MV:rank_3", "MV:rank_4"]
    assert "MV:solver_scip" in cols and "MV:solver_ls" in cols
    assert not [c for c in cols if c.startswith("GMV:solver")]  # solvers ran on the MV arm only
    assert asm.rank_ids == [2, 3, 4]
    core = list(asm.paths["net30"].columns)
    assert "MV:rank_4" not in core and "MV:rank_2" in core and "MV:rank_3" in core
    assert "MV:solver_ls" in core and "MV:opt" in core


def test_days_are_the_test_windows_of_all_folds_and_mark_f_k(asm):
    n_folds = len(asm.folds)
    assert len(asm.days) == n_folds * 5
    assert asm.days.index.is_monotonic_increasing
    for f in range(n_folds):
        assert (asm.days["fold"] == f).sum() == 5
    np.testing.assert_array_equal(asm.days["in_Fk"].to_numpy(),
                                  asm.masks["common"][asm.days["fold"].to_numpy()])
    assert (asm.days["rf"] > 0).all()  # the synthetic market pays 2 % a year
    for df in asm.paths.values():
        assert df.index.equals(asm.days.index) and np.isfinite(df.to_numpy()).all()


def test_folds_outside_f_k_are_held_identically_by_every_path(registered, rows):
    cfg = registered[0]
    fold = 6
    tampered = copy.deepcopy(rows)
    for r in tampered:
        if r["arm"] == "GMV" and r["fold"] == fold:
            r.update(kbest_all_certified=False, status="kbest_incomplete")
    asm = a.assemble(cfg, tampered, {"gross": PASSES["gross"]}, workers=1)
    assert not asm.masks["common"][fold]
    sel = (asm.days["fold"] == fold).to_numpy()
    for arm in ("MV", "GMV"):
        block = asm.paths["gross"].loc[sel, [c for c in asm.paths["gross"] if c.startswith(arm)]]
        assert (block.nunique(axis=1) == 1).all()  # the optimum's incumbent, whatever the path
    assert not asm.days.loc[sel, "in_Fk"].any()


def test_the_pre_cost_pass_has_no_costs_and_the_net_pass_differs_by_them_only(asm):
    gross, net = asm.paths["gross"], asm.paths["net"]
    stats = asm.fold_stats["net"]
    first_day = asm.days.groupby("fold").head(1).index
    for col in ("MV:opt", "MV:rank_2", "GMV:opt"):
        arm, path = col.split(":")
        cost = stats[(stats["arm"] == arm) & (stats["path"] == path)].set_index("fold")["cost"]
        expected = gross[col].copy()
        for fold, day in zip(cost.index, first_day, strict=True):
            expected.loc[day] = (1.0 + gross.loc[day, col]) * (1.0 - cost.loc[fold]) - 1.0
        np.testing.assert_allclose(net[col].to_numpy(), expected.to_numpy(), atol=1e-12)
    zero = asm.fold_stats["gross"]
    assert (zero["cost"] == 0.0).all()
    first = stats[stats["fold"] == 0]
    assert np.allclose(first["traded"], 1.0) and np.allclose(first["cost"], 1e-3)  # bought from cash
    assert (stats["cost"] >= 0).all() and (stats["turnover"] >= 0).all()
    # the same weights: the gross legs of both passes are the same series
    later = stats[stats["fold"] > 0]
    assert (later["traded"] > 0).any()


def test_the_cost_and_lag_sensitivities_use_the_same_weights(asm):
    net30, net = asm.paths["net30"], asm.paths["net"]
    assert (net30["MV:opt"] <= net["MV:opt"] + 1e-15).all()  # 30 bp costs more than 10 bp
    gross, lag = asm.paths["gross"], asm.paths["gross_lag1"]
    assert not np.allclose(lag["MV:opt"].to_numpy(), gross["MV:opt"].to_numpy())
    assert asm.meta["passes"]["gross_lag1"]["execution_lag"] == 1


def test_passes_run_in_parallel_workers_with_the_same_result(registered, rows):
    cfg = registered[0]
    one = a.assemble(cfg, rows, {"gross": PASSES["gross"]}, workers=1)
    two = a.assemble(cfg, rows, {"gross": PASSES["gross"]}, workers=2)
    pd.testing.assert_frame_equal(one.paths["gross"], two.paths["gross"])


# --------------------------------------------------------------------------------------------
# What the statistics get
# --------------------------------------------------------------------------------------------


def test_fk_series_hand_the_balanced_panel_to_the_statistics(asm):
    mask = asm.days["in_Fk"].to_numpy()
    n_days = int(mask.sum())
    assert n_days == int(asm.masks["common"].sum()) * 5 > 0
    mv = a.fk_series(asm, "gross", "MV", "sharpe")
    assert mv.opt.shape == (n_days,) and mv.ranks.shape == (n_days, 3) and mv.rank_ids == [2, 3, 4]
    rf = asm.days.loc[mask, "rf"].to_numpy()
    np.testing.assert_allclose(mv.opt, asm.paths["gross"].loc[mask, "MV:opt"].to_numpy() - rf)
    np.testing.assert_allclose(mv.ranks[:, 0], asm.paths["gross"].loc[mask, "MV:rank_2"].to_numpy() - rf)
    assert set(mv.solvers) == {"scip", "ls"} and mv.solvers["ls"].shape == (n_days,)
    gmv = a.fk_series(asm, "gross", "GMV", "logvar")  # the variance of the returns, not of the excess
    np.testing.assert_allclose(gmv.opt, asm.paths["gross"].loc[mask, "GMV:opt"].to_numpy())
    assert gmv.solvers == {}
    with pytest.raises(ValueError, match="functional"):
        a.fk_series(asm, "gross", "MV", "median")


def test_doses_and_solver_gaps_are_per_fold_tables(asm, rows):
    by_cell = {(r["arm"], r["fold"]): r for r in rows}
    for arm in ("MV", "GMV"):
        table = asm.doses[arm]
        assert list(table.columns) == [2, 3, 4] and len(table) == len(asm.folds)
        for fold in range(len(asm.folds)):
            row = by_cell[(arm, fold)]
            for rank in (2, 3, 4):
                expected = row["ranks"][rank - 1]["dose_bp"] if len(row["ranks"]) >= rank else np.nan
                assert (np.isnan(expected) and np.isnan(table.loc[fold, rank])) or \
                    table.loc[fold, rank] == pytest.approx(expected)
    solvers = asm.solvers
    assert set(solvers["solver"]) == {"scip", "ls"}
    in_fk = asm.masks["common"]
    ok = solvers[solvers["fold"].map(lambda f: bool(in_fk[f]))]
    assert np.isfinite(ok["gap_bp"]).all() and (ok["rank"] >= 1).all()


def test_the_fold_table_reports_the_regime_of_each_fold(asm):
    f = asm.folds
    assert {"formation_date", "test_start", "n_assets", "K", "s_cont", "binding",
            "lower_bound_active", "status_MV", "status_GMV", "in_Fk_MV", "in_Fk_GMV", "in_Fk"} <= set(f.columns)
    assert f["in_Fk"].equals(f["in_Fk_MV"] & f["in_Fk_GMV"])
    assert (f["K"] >= d.floor_cardinality(0.5)).all()


# --------------------------------------------------------------------------------------------
# Storage and the registered stage
# --------------------------------------------------------------------------------------------


def test_an_assembly_round_trips_through_its_directory(asm, tmp_path):
    a.save_assembly(asm, tmp_path / "out")
    again = a.load_assembly(tmp_path / "out")
    assert again.meta == json.loads(json.dumps(asm.meta))
    for key in asm.paths:
        pd.testing.assert_frame_equal(again.paths[key], asm.paths[key])
    for key in asm.masks:
        np.testing.assert_array_equal(again.masks[key], asm.masks[key])
    pd.testing.assert_frame_equal(again.days, asm.days)
    pd.testing.assert_frame_equal(again.folds, asm.folds, check_dtype=False)
    for arm in asm.doses:
        pd.testing.assert_frame_equal(again.doses[arm], asm.doses[arm])
    pd.testing.assert_frame_equal(again.solvers, asm.solvers, check_dtype=False)
    pd.testing.assert_frame_equal(again.fold_stats["net"], asm.fold_stats["net"], check_dtype=False)
    assert again.rank_ids == asm.rank_ids
    assert [(x.mask.sum(), x.n_excluded_rank) for x in again.fold_sets.values()] == \
           [(x.mask.sum(), x.n_excluded_rank) for x in asm.fold_sets.values()]


def test_the_assembly_stage_registers_a_run_with_its_lineage(registered, tmp_path):
    _, path, run_dir = registered
    acfg = {
        "stream": "E3", "name": "assemble_test",
        "sources": {"syn": {"config": str(path), "run": str(run_dir)}},
        "assembly": {"workers": 1, "allow_incomplete": False, "passes": {"gross": PASSES["gross"]}},
    }
    apath = tmp_path / "analysis.yaml"
    apath.write_text(yaml.safe_dump(acfg))
    runs = tmp_path / "runs"
    assert a.main(["--analysis-config", str(apath), "--source", "syn", "--runs-dir", str(runs),
                   "--allow-dirty"]) == 0
    (out,) = [p for p in runs.iterdir() if p.is_dir()]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["stream"] == "E3"
    cfg_json = json.loads((out / "config.json").read_text())
    assert cfg_json["source_run_id"] == run_dir.name and cfg_json["source"] == "syn"
    assert len(cfg_json["rows_sha256"]) == 64
    loaded = a.load_assembly(out)
    assert "gross" in loaded.paths and loaded.meta["source_run_id"] == run_dir.name
    registry_lines = (runs / "registry.jsonl").read_text().splitlines()
    assert json.loads(registry_lines[-1])["run_id"] == out.name


def test_a_pilot_style_config_assembles_one_lambda(registered, tmp_path):
    cfg = copy.deepcopy(registered[0])
    cfg["params"]["lambda_f"] = None
    cfg["params"]["lambdas"] = [5.0, 10.0]
    cfg["params"]["arms"] = ["MV"]
    cfg["params"]["solvers"] = None
    cfg["params"]["period"] = {"name": "confirmatory", "start": "2000-06-01", "end": "2000-07-15"}
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    rows = a.load_rows(run_dir)
    asm = a.assemble(cfg, rows, {"gross": PASSES["gross"]}, lam=10.0)
    assert asm.meta["lam"] == 10.0
    with pytest.raises(a.AssemblyError, match="lam"):
        a.assemble(cfg, rows, {"gross": PASSES["gross"]})  # two lambdas: say which


def test_excess_returns_use_the_rf_of_the_market(asm):
    gross = asm.paths["gross"]["MV:opt"]
    table = d.get_fold_table(asm.meta["market"], asm.meta["period"])
    expected = excess_returns(gross, table.market.rf)
    np.testing.assert_allclose((gross - asm.days["rf"]).to_numpy(), expected.to_numpy(), atol=1e-14)
