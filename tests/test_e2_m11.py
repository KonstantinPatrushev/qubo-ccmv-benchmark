"""The stage of the validation M11 (ANALYSIS_PLAN §12, 2026-10-17): Tables 3 and 4 of Bertsimas & Cory-Wright on
OR-Library port1-5 (``cif.experiments.e2_m11``, ``configs/E2/m11.yaml``).

The grid is checked on the real config and on small ones built here; a cell is run on a synthetic ``port1.txt``
(N = 31 as the real one, so the numbers of the paper are not at stake, only the machinery); the real files are
used where they are cheap (the rehearsal through the launcher, the grid). Independent solves of the same
problems close the loop: the support enumeration with the return row is in ``tests/test_scip_native.py``, a
cvxpy + SCIP model of (34) here.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
import yaml

from cif.experiments import e2_analysis, e2_benchmark, e2_m11, ingest, launch, run
from cif.instances import bcw
from cif.instances.orlib import ORLIB_N
from tests.test_instances import needs_orlib, orlib_text

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "E2" / "m11.yaml"
SCIP_ID, OA_ID = "scip-persp-mineig-sym0", "oa-highs"


def _tuned_spec(size: int) -> dict:
    report = json.loads((ROOT / "configs" / "E2" / "pilot" / "tuned.json").read_text(encoding="utf-8"))
    return report["sizes"][str(size)]["roles"]["exact_rep"]


def _cfg(tmp_path: Path | None = None, **params) -> dict:
    """A small config: port 1, k = 3, both tables, short limits."""
    p = {
        "stage": "m11", "seed_role": "calibration", "warm_up": False,
        "orlib_dir": str(tmp_path) if tmp_path is not None else "data/raw/orlib",
        "ports": [1], "ks": [3], "overrun_tol_rel": 0.05,
        "solver": {"mip_gap": 1e-4, "abs_gap": 1e-6},
        "tables": {"T3": {"kappa": 1.0, "time_limit_s": 30.0, "methods": ["scip", "oa"]},
                   "T4": {"kappa": 0.0, "time_limit_s": 60.0, "methods": ["scip"], "return_row": True}},
        "methods": {"scip": {"size": 100, "role": "exact_rep"}, "oa": {"size": 400, "role": "exact_rep"}},
        "solver_seeds": {"scip": {"first": 10000, "count": 3}, "oa": {"first": 10000, "count": 1}},
    }
    p.update(params)
    return {"stream": "E2", "name": "m11", "experiment": "cif.experiments.e2_m11", "mode": "full",
            "inputs": {"tuned": "configs/E2/pilot/tuned.json"}, "params": p}


@pytest.fixture
def port_dir(tmp_path):
    (tmp_path / "port1.txt").write_text(orlib_text(ORLIB_N[1], seed=5))
    e2_m11.clear_caches()
    yield tmp_path
    e2_m11.clear_caches()


# -- the grid ---------------------------------------------------------------------------------------------


def _full_cells() -> list[dict]:
    return run.grid(run.effective_config(run.load_config(CONFIG)))


def test_the_full_grid_is_15_instances_per_table_with_the_methods_and_seeds_of_the_plan():
    cells = _full_cells()
    assert len(cells) == 15 * (3 + 1) + 15 * 3 == 105
    by_table = {t: [c for c in cells if c["table"] == t] for t in ("T3", "T4")}
    assert len(by_table["T3"]) == 60 and len(by_table["T4"]) == 45
    for group in by_table.values():
        assert {(c["port"], c["k"]) for c in group} == {(p, k) for p in ORLIB_N for k in (5, 10, 20)}
        assert all(c["n"] == ORLIB_N[c["port"]] for c in group)
    assert {c["method"]["name"] for c in by_table["T4"]} == {"scip"}  # OA has no return row (the plan says so)
    assert {c["method"]["name"] for c in by_table["T3"]} == {"scip", "oa"}
    for c in cells:
        assert (c["kappa"], c["return_row"], c["time_limit_s"]) == (
            (1.0, False, 300.0) if c["table"] == "T3" else (0.0, True, 3600.0))
        assert c["stage"] == "m11"
    scip_seeds = {(c["table"], c["port"], c["k"]): sorted(x["seed"] for x in cells
                                                          if x["table"] == c["table"] and x["port"] == c["port"]
                                                          and x["k"] == c["k"] and x["method"]["name"] == "scip")
                  for c in cells if c["method"]["name"] == "scip"}
    assert set(map(tuple, scip_seeds.values())) == {(10000, 10001, 10002)}  # the median of three permutations
    assert {c["seed"] for c in cells if c["method"]["name"] == "oa"} == {10000}  # one run


def test_the_methods_are_the_exact_representatives_of_tuned_json():
    cells = _full_cells()
    scip = {json.dumps(c["method"], sort_keys=True) for c in cells if c["method"]["name"] == "scip"}
    oa = {json.dumps(c["method"], sort_keys=True) for c in cells if c["method"]["name"] == "oa"}
    assert scip == {json.dumps(_tuned_spec(100), sort_keys=True)} == {json.dumps(_tuned_spec(200), sort_keys=True)}
    assert oa == {json.dumps(_tuned_spec(400), sort_keys=True)}
    assert _tuned_spec(100)["id"] == SCIP_ID and _tuned_spec(400)["id"] == OA_ID


def test_cells_are_ordered_instance_by_instance_and_unique():
    cells = _full_cells()
    assert len({run.cell_key(c) for c in cells}) == len(cells)
    firsts = [(c["table"], c["port"], c["k"]) for i, c in enumerate(cells) if i == 0 or
              (c["table"], c["port"], c["k"]) != (cells[i - 1]["table"], cells[i - 1]["port"], cells[i - 1]["k"])]
    assert len(firsts) == len(set(firsts)) == 30  # every (table, port, k) is one block of consecutive cells


def test_the_smoke_grid_is_port1_k5_in_both_tables_with_short_limits_and_calibration_seeds():
    cfg = run.effective_config(run.load_config(CONFIG), smoke=True)
    cells = run.grid(cfg)
    assert cfg["mode"] == "smoke" and cfg["params"]["seed_role"] == "calibration"
    assert {(c["table"], c["port"], c["k"]) for c in cells} == {("T3", 1, 5), ("T4", 1, 5)}
    assert {c["method"]["name"] for c in cells if c["table"] == "T4"} == {"scip"}
    assert max(c["time_limit_s"] for c in cells) <= 30.0
    assert all(10_000 <= c["seed"] < 20_000 for c in cells)


def test_the_solver_seeds_are_calibration_ones_and_the_declared_seeds_match():
    cfg = run.load_config(CONFIG)
    assert cfg["seeds"] == [10000, 10001, 10002]  # the manifest records them
    assert cfg["params"]["seed_role"] == "calibration"
    assert all(10_000 <= c["seed"] < 20_000 for c in _full_cells())


def test_a_smoke_run_refuses_the_test_seed_role():
    cfg = _cfg(seed_role="test", solver_seeds={"scip": [20000], "oa": [20000]})
    cfg["mode"] = "smoke"
    with pytest.raises(ValueError, match="smoke"):
        e2_m11.cells(cfg)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"stage": "bench"}, "stage"),
        ({"seed_role": "validation"}, "seed_role"),
        ({"ports": [6]}, "port"),
        ({"ports": []}, "ports"),
        ({"ks": [32]}, "k"),
        ({"ks": [0]}, "k"),
        ({"solver": {"mip_gap": 0.0, "abs_gap": 1e-6}}, "mip_gap"),
        ({"solver": {"mip_gap": 1e-4, "abs_gap": -1.0}}, "abs_gap"),
        ({"solver_seeds": {"scip": [20000], "oa": [10000]}}, "seed"),
        ({"solver_seeds": {"scip": [10000]}}, "solver_seeds"),
        ({"tables": {"T5": {"kappa": 1.0, "time_limit_s": 1.0, "methods": ["scip"]}}}, "table"),
        ({"tables": {"T3": {"kappa": 1.0, "time_limit_s": 0.0, "methods": ["scip"]}}}, "time_limit_s"),
        ({"tables": {"T3": {"kappa": 1.0, "time_limit_s": 1.0, "methods": ["tabu"]}}}, "method"),
        ({"tables": {"T4": {"kappa": 0.0, "time_limit_s": 1.0, "methods": ["scip", "oa"], "return_row": True}}},
         "return row"),
        ({"tables": {"T3": {"kappa": 1.0, "time_limit_s": 1.0, "methods": ["scip"], "extra": 1}}}, "unknown"),
        ({"methods": {"scip": {"size": 999, "role": "exact_rep"}, "oa": {"size": 400, "role": "exact_rep"}}},
         "size"),
        ({"methods": {"scip": {"size": 400, "role": "exact_rep"}, "oa": {"size": 400, "role": "exact_rep"}}},
         "scip"),
    ],
)
def test_invalid_configs_are_refused_with_a_reason(change, match):
    with pytest.raises(ValueError, match=match):
        e2_m11.cells(_cfg(**change))


def test_the_declared_top_level_seeds_must_be_the_seeds_of_the_grid():
    cfg = _cfg()
    cfg["seeds"] = [10000, 10001, 10002]
    assert e2_m11.cells(cfg)
    cfg["seeds"] = [10000]
    with pytest.raises(ValueError, match="seeds"):
        e2_m11.cells(cfg)


def test_the_tuned_report_of_a_smoke_stand_in_is_refused_in_a_full_run(tmp_path):
    stand_in = ROOT / "configs" / "E2" / "smoke" / "tuned.json"
    cfg = _cfg()
    cfg["inputs"]["tuned"] = str(stand_in)
    with pytest.raises(ValueError, match="smoke"):
        e2_m11.cells(cfg)


# -- the config as the runner and the launcher read it ------------------------------------------------------


def test_the_config_follows_the_runner_contract_and_the_concurrency_cap_of_the_plan():
    cfg = run.load_config(CONFIG)
    assert cfg["stream"] == "E2" and cfg["name"] == "m11" and cfg["experiment"] == "cif.experiments.e2_m11"
    assert "smoke" in cfg, "every stage can be rehearsed"
    assert cfg["threads"] == 1  # explicit, not the default of the runner
    launch_cfg = cfg["launch"]
    assert launch_cfg["timing"] is True  # the times are the result
    assert launch_cfg["jobs"] <= 16  # half of the 32 physical cores of the shared server (plan §12)
    assert launch_cfg["shards"] >= launch_cfg["jobs"]
    assert launch_cfg["shards"] == len(_full_cells())  # one cell per shard: the cells differ by orders of magnitude
    assert set(cfg["inputs"]) == {"tuned", "orlib", "published"}
    for label in ("tuned", "published"):
        assert (ROOT / cfg["inputs"][label]).is_file(), label
    assert cfg["inputs"]["orlib"] == "data/raw/orlib/provenance.json"
    assert cfg["params"]["orlib_dir"] == "data/raw/orlib"


def test_the_stopping_rule_and_the_limits_are_the_ones_of_the_plan():
    p = run.load_config(CONFIG)["params"]
    assert p["solver"] == {"mip_gap": 1e-4, "abs_gap": 1e-6}  # the defaults of CPLEX 12.8
    assert p["tables"]["T3"] == {"kappa": 1.0, "time_limit_s": 300.0, "methods": ["scip", "oa"]}
    assert p["tables"]["T4"] == {"kappa": 0.0, "time_limit_s": 3600.0, "methods": ["scip"], "return_row": True}
    assert p["ports"] == [1, 2, 3, 4, 5] and p["ks"] == [5, 10, 20]


def test_no_cell_has_a_warm_start_or_more_than_one_thread_in_its_call(monkeypatch, port_dir):
    """The call of the solvers: no ``warm_start``, ``threads`` 1 (the defaults of both), ``budget_s`` the limit, the
    stopping rule of the config, SCIP's seeds and its switch of the symmetry detection, the row only in T4."""
    calls = []
    real_scip, real_oa = e2_m11.solve_miqp_native, e2_m11.solve_oa

    def spy(real):
        def wrapper(*args, **kwargs):
            calls.append(kwargs)
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(e2_m11, "solve_miqp_native", spy(real_scip))
    monkeypatch.setattr(e2_m11, "solve_oa", spy(real_oa))
    cfg = _cfg(port_dir)
    for table, method in (("T3", "scip"), ("T3", "oa"), ("T4", "scip")):
        e2_m11.run_cell(_cell(cfg, table, method), cfg)
    assert len(calls) == 3
    for kwargs in calls:
        assert kwargs.get("warm_start") is None and kwargs.get("threads", 1) == 1
        assert kwargs["budget_s"] in (30.0, 60.0) and kwargs["mip_gap"] == 1e-4 and kwargs["abs_gap"] == 1e-6
    assert calls[0]["scip_params"] == {"misc/usesymmetry": 0} and calls[0]["permutation_seed"] == calls[0]["seed"]
    assert calls[0]["linear_ge"] is None and calls[2]["linear_ge"] is not None  # the row only in T4
    assert "linear_ge" not in calls[1]  # OA has no such argument


def test_the_warm_up_runs_both_solvers_once_per_process(monkeypatch):
    monkeypatch.setattr(e2_m11, "_warmed_up", False)
    e2_m11._warm_up()
    assert e2_m11._warmed_up is True
    seen = []
    monkeypatch.setattr(e2_m11, "solve_miqp_native", lambda *a, **k: seen.append("scip"))
    monkeypatch.setattr(e2_m11, "solve_oa", lambda *a, **k: seen.append("oa"))
    e2_m11._warm_up()  # a second call does nothing
    assert seen == []


def test_run_cell_warms_up_when_the_config_says_so(monkeypatch, port_dir):
    calls = []
    monkeypatch.setattr(e2_m11, "_warm_up", lambda: calls.append(1))
    cfg = _cfg(port_dir, warm_up=True, ks=[2])
    e2_m11.run_cell(_cell(cfg, "T3", "scip", k=2), cfg)
    assert calls == [1]
    cfg = _cfg(port_dir, warm_up=False, ks=[2])
    e2_m11.run_cell(_cell(cfg, "T3", "scip", k=2), cfg)
    assert calls == [1]


# -- seeds ------------------------------------------------------------------------------------------------


def test_the_derived_seed_is_a_31_bit_function_of_every_coordinate_of_the_cell():
    cells = e2_m11.cells(_cfg(ports=[1, 2], ks=[5, 10]))
    seeds = [e2_m11.derive_seed(c) for c in cells]
    assert seeds == [e2_m11.derive_seed(copy.deepcopy(c)) for c in cells]  # deterministic
    assert all(0 <= s < 2**31 for s in seeds)
    assert len(set(seeds)) == len(cells)  # no two cells share a stream (tables, ports, k, methods, seeds)


def test_the_derived_seed_does_not_depend_on_the_time_limit():
    cell = e2_m11.cells(_cfg())[0]
    assert e2_m11.derive_seed({**cell, "time_limit_s": 1.0}) == e2_m11.derive_seed(cell)


# -- a cell -----------------------------------------------------------------------------------------------


def _cell(cfg: dict, table: str, method: str, seed: int = 10000, k: int = 3) -> dict:
    return next(c for c in e2_m11.cells(cfg) if c["table"] == table and c["method"]["name"] == method
                and c["seed"] == seed and c["k"] == k)


def test_a_t3_cell_of_scip_reports_the_objective_of_the_paper_and_a_certified_gap(port_dir):
    cfg = _cfg(port_dir)
    row = e2_m11.run_cell(_cell(cfg, "T3", "scip"), cfg)
    inst = bcw.bcw_instance(1, port_dir)
    x = np.zeros(31)
    x[row["support"]] = row["weights"]
    assert row["status"] in ("optimal", "gaplimit") and row["proved"] is True and row["certified"] is True
    assert row["gamma"] == pytest.approx(100.0 / math.sqrt(31)) and row["r_bar"] is None and row["n_assets"] == 31
    assert row["primal"] == pytest.approx(bcw.bcw_objective(inst, x, 1.0), rel=1e-12)  # (34), paper form
    assert row["dual"] <= row["primal"] + 1e-12
    assert row["rel_gap"] <= 1e-4 * (1 + 1e-6) or row["abs_gap"] <= 1e-6 * (1 + 1e-6)
    assert row["nodes"] is not None and row["nodes"] >= 0 and row["config_id"] == SCIP_ID
    assert row["feasible"] is True and 1 <= row["n_nonzero"] <= 3 and sum(row["weights"]) == pytest.approx(1.0)
    assert row["threads"] == 1 and 0 < row["seed_used"] < 2**31 and row["return_ok"] is None


def test_a_t3_cell_of_oa_reports_cuts_instead_of_nodes(port_dir):
    cfg = _cfg(port_dir)
    row = e2_m11.run_cell(_cell(cfg, "T3", "oa"), cfg)
    assert row["status"] == "optimal" and row["certified"] is True and row["config_id"] == OA_ID
    assert row["nodes"] is None and row["n_cuts"] >= 1 and row["iterations"] >= 0
    scip = e2_m11.run_cell(_cell(cfg, "T3", "scip"), cfg)
    assert row["primal"] == pytest.approx(scip["primal"], rel=1e-4)  # the same problem, both inside the gap


def test_a_t4_cell_carries_the_threshold_and_holds_the_return_row(port_dir):
    cfg = _cfg(port_dir)
    row = e2_m11.run_cell(_cell(cfg, "T4", "scip"), cfg)
    inst = bcw.bcw_instance(1, port_dir)
    assert row["r_bar"] == inst.r_bar and (row["r_min"], row["r_max"]) == (inst.threshold.r_min, inst.threshold.r_max)
    assert row["certified"] is True and row["return_ok"] is True and row["feasible"] is True
    assert row["r_realised"] >= inst.r_bar - 1e-6 * float(np.max(np.abs(inst.mu)))
    x = np.zeros(31)
    x[row["support"]] = row["weights"]
    assert row["primal"] == pytest.approx(bcw.bcw_objective(inst, x, 0.0), rel=1e-12)
    assert row["primal"] == pytest.approx(0.5 * x @ inst.sigma_ridge @ x, rel=1e-12)  # κ = 0: no return term
    assert float(inst.mu @ x) == pytest.approx(row["r_realised"], rel=1e-9)


def test_the_row_has_every_field_the_plan_asks_for_and_is_plain_json(port_dir):
    cfg = _cfg(port_dir)
    for table, method in (("T3", "scip"), ("T3", "oa"), ("T4", "scip")):
        cell = _cell(cfg, table, method)
        row = {**cell, **e2_m11.run_cell(cell, cfg)}
        for key in ("table", "port", "n", "k", "gamma", "kappa", "r_bar", "method", "seed", "status", "wall_s",
                    "primal", "dual", "rel_gap", "nodes", "certified", "time_limit_s", "proved", "end_s", "cpu_s",
                    "build_s", "solve_s", "refine_s", "seed_used", "threads", "instance_id", "config_id", "detail"):
            assert key in row, (table, method, key)
        assert row["wall_s"] > 0 and row["wall_s"] >= row["end_s"] - 1e-9 and row["cpu_s"] >= 0
        assert row["instance_id"] == "orlib/port1"
        json.dumps(row, allow_nan=False)  # no NaN or inf


def test_the_wall_time_is_that_of_the_call_and_the_certificate_needs_it_inside_the_limit(port_dir):
    cfg = _cfg(port_dir)
    cell = _cell(cfg, "T3", "scip")
    row = e2_m11.run_cell(cell, cfg)
    assert row["wall_s"] >= row["build_s"] + row["solve_s"] - 1e-6  # the call holds the build and the solve
    assert row["wall_s"] <= cell["time_limit_s"] and row["certified"] is True
    assert row["overrun_s"] == 0.0 and row["budget_ok"] is True


@pytest.mark.parametrize(("table", "method"), [("T3", "scip"), ("T3", "oa"), ("T4", "scip")])
def test_a_limit_that_is_spent_before_the_solve_gives_an_uncertified_row(port_dir, table, method):
    cfg = _cfg(port_dir)
    cfg["params"]["tables"][table]["time_limit_s"] = 0.001
    row = e2_m11.run_cell(_cell(cfg, table, method), cfg)
    assert row["certified"] is False
    assert row["status"] in ("timelimit", "optimal", "gaplimit")  # OA may finish a trivial master in time
    if row["status"] == "timelimit":
        assert row["proved"] is False


def test_a_cell_is_reproducible_in_everything_but_time(port_dir):
    cfg = _cfg(port_dir)
    a = e2_m11.run_cell(_cell(cfg, "T4", "scip", seed=10001), cfg)
    b = e2_m11.run_cell(_cell(cfg, "T4", "scip", seed=10001), cfg)
    for key in ("status", "primal", "dual", "nodes", "support", "weights", "r_bar", "seed_used"):
        assert a[key] == b[key], key


def test_the_instance_is_built_once_per_process(port_dir):
    e2_m11.clear_caches()
    first = e2_m11._instance(1, str(port_dir))
    assert e2_m11._instance(1, str(port_dir)) is first


# -- independent solves -----------------------------------------------------------------------------------


def _cvxpy_scip_optimum(inst, k: int, kappa: float, r_bar: float | None) -> float:
    """(34) as a cvxpy MIQP with big-M linking (``x ≤ z``) and SCIP behind it: another model, another code path
    to SCIP than the native one."""
    n = inst.n
    x = cp.Variable(n, nonneg=True)
    z = cp.Variable(n, boolean=True)
    objective = (0.5 * cp.quad_form(x, cp.psd_wrap(inst.sigma_ridge)) - kappa * (inst.mu @ x))
    constraints = [cp.sum(x) == 1, x <= z, cp.sum(z) <= k]
    if r_bar is not None:
        constraints.append(inst.mu @ x >= r_bar)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="SCIP", scip_params={"limits/gap": 1e-7, "limits/time": 120})
    assert problem.status == "optimal"
    return float(problem.value)


@pytest.mark.parametrize(("table", "kappa", "return_row"), [("T3", 1.0, False), ("T4", 0.0, True)])
@pytest.mark.parametrize("method", ["scip", "oa"])
def test_the_certified_optimum_agrees_with_an_independent_cvxpy_scip_model(port_dir, table, kappa, return_row,
                                                                           method):
    """k = 3 keeps the big-M reference (cvxpy, SCIP) to a few seconds; the real port1 at k = 5 is below."""
    if method == "oa" and return_row:
        pytest.skip("OA has no linear rows: T4 is run with SCIP only")
    cfg = _cfg(port_dir)
    row = e2_m11.run_cell(_cell(cfg, table, method), cfg)
    inst = bcw.bcw_instance(1, port_dir)
    reference = _cvxpy_scip_optimum(inst, 3, kappa, inst.r_bar if return_row else None)
    assert row["certified"] is True
    assert row["primal"] >= reference - 1e-9  # no incumbent below the true optimum
    assert abs(row["primal"] - reference) <= 1e-4 * abs(reference) + 1e-6  # and within the stopping rule of it
    assert row["dual"] <= reference + 1e-9  # the dual bound does not cut the optimum off


@needs_orlib
@pytest.mark.parametrize(("table", "kappa", "return_row", "k"), [("T3", 1.0, False, 5), ("T4", 0.0, True, 3)])
def test_the_certified_optimum_on_the_real_port1_agrees_with_the_independent_model(table, kappa, return_row, k):
    """Port 1 of the OR-Library (the file of the paper), our native SCIP against the big-M model of cvxpy + SCIP."""
    e2_m11.clear_caches()
    cfg = _cfg(None, ks=[k])
    row = e2_m11.run_cell(_cell(cfg, table, "scip", k=k), cfg)
    inst = bcw.bcw_instance(1)
    reference = _cvxpy_scip_optimum(inst, k, kappa, inst.r_bar if return_row else None)
    assert row["certified"] is True
    assert row["primal"] >= reference - 1e-9
    assert abs(row["primal"] - reference) <= 1e-4 * abs(reference) + 1e-6
    assert row["dual"] <= reference + 1e-9


# -- through the runner and the launcher (real files, rehearsal) -------------------------------------------


@needs_orlib
def test_the_launcher_runs_the_rehearsal_of_the_stage_on_the_real_files(tmp_path):
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(CONFIG), "--smoke", "--out-dir", str(out_dir), "--jobs", "2",
                        "--shards", "4", "--cores", "0-3", "--poll", "0.05"]) == 0
    rows = [json.loads(line) for p in sorted(out_dir.glob("*_shard*.jsonl")) for line in
            p.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = run.grid(run.effective_config(run.load_config(CONFIG), smoke=True))
    assert len(rows) == len(expected) and not any(r.get("error") for r in rows)
    assert {r["cell_key"] for r in rows} == {run.cell_key(c) for c in expected}
    assert all(r["threads"] == 1 and r["stage"] == "m11" and r["certified"] for r in rows)
    t4 = [r for r in rows if r["table"] == "T4"]
    assert t4 and all(r["r_bar"] is not None and r["return_ok"] for r in t4)
    port1 = bcw.bcw_instance(1)
    assert all(r["r_bar"] == port1.r_bar for r in t4)
    assert yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["launch"]["timing"] is True
    assert e2_benchmark.SEED_RANGES["calibration"][1] == (10_000, 20_000)


def test_the_stage_goes_through_launcher_ingest_and_analysis_on_a_small_grid(tmp_path, port_dir, monkeypatch):
    """Launcher (shards) → ingest (one registered run: inputs, bundle, conflicts) → ``e2_analysis m11`` on it: the rows
    of the stage are accepted by the registry as they are. T3 of the synthetic port1 at k = 5 only, so it is quick."""
    cfg = _cfg(port_dir, ks=[5], tables={"T3": {"kappa": 1.0, "time_limit_s": 30.0, "methods": ["scip", "oa"]}})
    cfg.pop("mode")
    cfg.update(threads=1, seeds=[10000, 10001, 10002], launch={"timing": True, "jobs": 2, "shards": 4})
    cfg["inputs"] = {"tuned": "configs/E2/pilot/tuned.json", "published": "configs/E2/m11_bcw_published.yaml"}
    config = tmp_path / "m11_small.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out_dir = tmp_path / "out"
    monkeypatch.setenv("CIF_BUNDLE_SHA", "c" * 40)  # the bundle the rows are attributed to (the launcher writes it down)
    assert launch.main(["run", "--config", str(config), "--out-dir", str(out_dir), "--jobs", "2", "--shards", "4",
                        "--cores", "0-3", "--poll", "0.05"]) == 0
    shards = sorted(out_dir.glob("*_shard*.jsonl"))
    assert len(shards) == 4
    runs = tmp_path / "runs"
    assert ingest.main(["--config", str(config), "--shards", *map(str, shards), "--runs-dir", str(runs),
                        "--bundle-sha-file", str(out_dir / "BUNDLE_SHA"),
                        "--logs", str(out_dir / "launch_manifest.json")]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "ingested" and manifest["n_cells"] == 4 and manifest["n_errors"] == 0
    assert manifest["bundle_sha"] == "c" * 40 and manifest["seeds"] == [10000, 10001, 10002]
    assert set(manifest["inputs"]) == {"tuned", "published"}
    assert e2_analysis.main(["m11", "--rows", str(run_dir), "--published", str(ROOT / "configs" / "E2" /
                                                                                  "m11_bcw_published.yaml"),
                             "--out", str(tmp_path / "analysis"), "--allow-incomplete"]) == 0
    report = json.loads((tmp_path / "analysis" / "m11.json").read_text(encoding="utf-8"))
    (rows_input,) = [i for i in report["inputs"] if i["role"] == "rows"]
    assert rows_input["run_id"] == run_dir.name and rows_input["n_rows"] == 4 and rows_input["mode"] == "ingested"
    block = report["tables"]["T3"]
    assert [(c["port"], c["k"]) for c in block["cells"]] == [(1, 5)] and len(block["missing"]) == 14
    view = block["cells"][0]["views"]
    assert view["scip"]["n_runs"] == 3 and view["oa"]["n_runs"] == 1 and view["stack"]["censored"] is False
    assert report["audit"]["bound_conflicts"] == [] and report["audit"]["n_infeasible"] == 0
