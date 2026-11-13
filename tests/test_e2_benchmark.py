"""E2 experiment module for the runner (ANALYSIS_PLAN §2 RQ2/H1c, §3, §5, §6; ``cif.experiments.run``).

A cell is one run of one method (a method spec of :mod:`cif.experiments.e2_space`) on one instance
with one solver seed and one end-to-end budget, or the long certification run of the reference.
Everything runs on tiny instances (N = 12-16) with budgets of a fraction of a second; the instance
seeds are calibration ones (1000-1999) and the solver seeds 10000-19999. Test seeds appear only in
tests of ``cells()`` that must refuse them.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import yaml

from cif.experiments import e2_benchmark as e2
from cif.experiments import e2_instances, ingest, run
from cif.experiments import e2_space as space
from cif.metrics.anytime import incumbent_curve
from tests.test_instances import orlib_text

N = 14  # assets of the tiny factor instances
BUDGET = 0.4

SA = {"name": "sa", "id": "sa-t", "num_reads_b": 20, "num_sweeps": 50, "beta": "default",
      "penalty": 10.0, "m_b": 20}
SA_OBJ = {**SA, "id": "sa-t-obj", "beta": "objective"}
TABU = {"name": "tabu", "id": "tabu-t", "num_restarts_b": 2, "tenure": 5, "num_reads_b": 4,
        "penalty": 10.0, "m_b": 20}
SWAP = {"name": "swap_sa", "id": "swap-t", "num_reads_b": 10, "num_sweeps": 20, "m_b": 20}
LS = {"name": "ls", "id": "ls-t", "num_restarts": 3, "improvement": "first", "rcl_size": 3}
SCIP = {"name": "scip", "id": "scip-persp-mineig-symdef", "formulation": "perspective",
        "diag": "mineig", "usesymmetry": "default"}
SCIP_BIGM = {"name": "scip", "id": "scip-bigm-sym0", "formulation": "bigm", "usesymmetry": 0}
OA = {"name": "oa", "id": "oa-highs", "diag": "mineig"}
HEURISTICS = (SA, SA_OBJ, TABU, SWAP, LS)
EXACT = (SCIP, SCIP_BIGM, OA)


def _fixed(**over) -> dict:
    return {"lam": 5.0, "fraction": 0.5, **over}


def _cfg(**params) -> dict:
    """A tiny config of the ``screen`` stage on two calibration instances."""
    cfg = {
        "stream": "E2",
        "name": "unit",
        "experiment": "cif.experiments.e2_benchmark",
        "inputs": {},
        "params": {
            "stage": "screen",
            "seed_role": "calibration",
            "budget_s": BUDGET,
            "warm_up": False,  # tests do not pay for it on every run (see test_warm_up_*)
            "fixed": _fixed(),
            "solver": {"mip_gap": 1e-6, "abs_gap_rel": 1e-6, "abs_gap_safety": 0.5},
            "overrun_tol_rel": 0.05,
            "instances": [{"cell": "T14", "family": "factor", "n": N, "seeds": [1000, 1001],
                           "config_size": N}],
            "solver_seeds": {"exact": [10000], "heuristic": [10000, 10001]},
            "methods": {"source": "explicit", "specs": [SA, SCIP]},
        },
    }
    cfg["params"].update(params)
    return cfg


@pytest.fixture(autouse=True)
def _fresh_caches():
    e2.clear_caches()
    yield
    e2.clear_caches()


def _cell(cfg: dict, **where) -> dict:
    for cell in e2.cells(cfg):
        if all(cell.get(k) == v or cell["method"].get(k) == v for k, v in where.items()):
            return cell
    raise AssertionError(f"no cell with {where}")


def _row(cell: dict, cfg: dict) -> dict:
    """What the runner stores: the cell's coordinates and the result of ``run_cell``."""
    return {**cell, **e2.run_cell(cell, cfg)}


def _write(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


# --------------------------------------------------------------------------------------
# The grid: explicit specs, sampled configurations, seeds
# --------------------------------------------------------------------------------------


def test_explicit_cells_cover_instances_specs_and_the_seeds_of_their_class():
    cfg = _cfg(methods={"source": "explicit", "specs": [SA, TABU, SCIP, OA]})
    cells = e2.cells(cfg)
    # 2 instances × (heuristics SA, tabu with 2 seeds; exact SCIP, OA with 1 seed)
    assert len(cells) == 2 * (2 * 2 + 2 * 1)
    assert {c["solver_seed"] for c in cells if c["method"]["name"] == "sa"} == {10000, 10001}
    assert {c["solver_seed"] for c in cells if c["method"]["name"] == "scip"} == {10000}
    assert all(c["stage"] == "screen" and c["budget_s"] == BUDGET and c["cell"] == "T14" for c in cells)
    assert all(c["family"] == "factor" and c["n"] == N and c["config_size"] == N for c in cells)
    assert len(run.grid(cfg)) == len(cells)  # unique cell keys, as the runner demands


def test_cells_are_grouped_by_instance_so_that_the_instance_cache_is_reused():
    cells = e2.cells(_cfg(methods={"source": "explicit", "specs": [SA, TABU, SCIP]}))
    seeds = [c["seed"] for c in cells]
    changes = [i for i in range(1, len(seeds)) if seeds[i] != seeds[i - 1]]
    assert seeds[:1] == [1000] and len(changes) == 1  # one block per instance
    assert cells == e2.cells(_cfg(methods={"source": "explicit", "specs": [SA, TABU, SCIP]}))


def test_a_spec_listed_twice_gives_one_set_of_cells():
    cells = e2.cells(_cfg(methods={"source": "explicit", "specs": [SA, SA, SCIP]}))
    assert len(cells) == 2 * (2 + 1)


def test_invalid_specs_are_refused_when_the_grid_is_built():
    bad = {**SA, "beta": "wide"}
    with pytest.raises(ValueError, match="beta"):
        e2.cells(_cfg(methods={"source": "explicit", "specs": [bad]}))


def test_screen_samples_configurations_per_size_with_the_exact_candidates():
    cfg = _cfg(methods={"source": "screen", "names": ["sa", "ls", "exact"], "n_configs": 3,
                        "tuning_seed": 5})
    cells = e2.cells(cfg)
    by_name: dict[str, set[str]] = {}
    for c in cells:
        by_name.setdefault(c["method"]["name"], set()).add(c["method"]["id"])
    assert by_name["sa"] == {"sa-00", "sa-01", "sa-02"} and by_name["ls"] == {"ls-00", "ls-01", "ls-02"}
    assert {s for s in by_name if s in ("scip", "oa")} == {"scip", "oa"}
    assert by_name["scip"] | by_name["oa"] == {c["id"] for c in space.exact_candidates(N)}
    sampled = space.sample_configs("sa", 3, 5, N)
    assert [c["method"] for c in cells if c["method"]["name"] == "sa" and c["seed"] == 1000
            and c["solver_seed"] == 10000] == sampled
    assert len(cells) == 2 * ((3 + 3) * 2 + len(space.exact_candidates(N)) * 1)


def test_screen_sampling_depends_on_the_size_of_the_group():
    cfg = _cfg(instances=[
        {"cell": "A", "family": "factor", "n": 12, "seeds": [1000], "config_size": 12},
        {"cell": "B", "family": "factor", "n": 16, "seeds": [1001], "config_size": 16},
    ], methods={"source": "screen", "names": ["tabu"], "n_configs": 4, "tuning_seed": 1})
    cells = e2.cells(cfg)
    a = [c["method"] for c in cells if c["cell"] == "A" and c["solver_seed"] == 10000]
    b = [c["method"] for c in cells if c["cell"] == "B" and c["solver_seed"] == 10000]
    assert a == space.sample_configs("tabu", 4, 1, 12) and b == space.sample_configs("tabu", 4, 1, 16)


def test_first_seed_and_count_give_consecutive_instance_seeds():
    cfg = _cfg(instances=[{"cell": "T14", "family": "factor", "n": N, "first_seed": 1010, "count": 3,
                           "config_size": N}])
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1010, 1011, 1012]


def test_the_count_may_come_from_the_n_rule_report(tmp_path):
    n_rule = _write(tmp_path, "n_rule.json", {"kind": "e2-n-rule", "n_instances": {"T14": 3, "C2": 25}})
    cfg = _cfg(instances=[{"cell": "T14", "family": "factor", "n": N, "first_seed": 1010,
                           "count_from": "n_rule", "config_size": N}])
    cfg["inputs"] = {"n_rule": n_rule}
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1010, 1011, 1012]
    cfg["inputs"] = {}
    with pytest.raises(ValueError, match="n_rule"):
        e2.cells(cfg)


def test_the_instances_may_come_from_the_ttt_selection(tmp_path):
    ttt = _write(tmp_path, "ttt.json", {"kind": "e2-ttt-instances", "instances": [1003, 1005, 1006]})
    cfg = _cfg(instances=[{"cell": "TTT", "family": "factor", "n": N, "seeds_from": "ttt_instances",
                           "config_size": N}])
    cfg["inputs"] = {"ttt_instances": ttt}
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1003, 1005, 1006]


def test_orlib_cells_name_the_port_and_the_lambda_index_not_the_binding():
    cfg = _cfg(instances=[{"cell": "C5", "family": "orlib", "ports": [1, 2], "lam_indices": 3,
                           "config_size": [{"max_n": 50, "size": 31}, {"size": 85}]}],
               solver_seeds={"exact": [10000], "heuristic": [10000]},
               methods={"source": "explicit", "specs": [LS]})
    cells = e2.cells(cfg)
    assert len(cells) == 2 * 3 * 1
    assert {(c["port"], c["lam_index"], c["n"]) for c in cells} == {
        (p, i, n) for p, n in ((1, 31), (2, 85)) for i in range(3)}
    assert {c["config_size"] for c in cells if c["port"] == 1} == {31}  # N ≤ 100 rule of the plan
    assert {c["config_size"] for c in cells if c["port"] == 2} == {85}
    assert all("seed" not in c for c in cells)


def test_a_boot_cell_carries_the_pilot_lambda(tmp_path):
    c4 = _write(tmp_path, "c4.json", {"kind": "e2-c4-lambda", "lam": 10.0, "lam_used": 10.0,
                                      "admissible": True})
    cfg = _cfg(instances=[{"cell": "C4", "family": "boot", "n": 200, "seeds": [1100], "config_size": 200}],
               methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    cfg["inputs"] = {"c4_lambda": c4}
    cells = e2.cells(cfg)
    assert [c["boot_lam"] for c in cells] == [10.0] and cells[0]["family"] == "boot"
    cfg["inputs"] = {}
    with pytest.raises(ValueError, match="c4_lambda"):
        e2.cells(cfg)


def test_a_descriptive_c4_uses_the_largest_candidate_lambda(tmp_path):
    c4 = _write(tmp_path, "c4.json", {"kind": "e2-c4-lambda", "lam": None, "lam_used": 20.0,
                                      "admissible": False})
    cfg = _cfg(instances=[{"cell": "C4", "family": "boot", "n": 200, "seeds": [1100], "config_size": 200}],
               methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    cfg["inputs"] = {"c4_lambda": c4}
    assert [c["boot_lam"] for c in e2.cells(cfg)] == [20.0]


def test_a_stand_in_report_of_the_smoke_rehearsals_is_refused_outside_a_smoke_run(tmp_path):
    """``configs/E2/smoke/*.json`` are hand-written stand-ins (``"smoke": true``): copied into ``pilot/``
    by mistake they would let a confirmatory stage run on made-up numbers of instances or configurations."""
    n_rule = _write(tmp_path, "n_rule.json", {"kind": "e2-n-rule", "smoke": True, "n_instances": {"T14": 2}})
    cfg = _cfg(instances=[{"cell": "T14", "family": "factor", "n": N, "first_seed": 1010,
                           "count_from": "n_rule", "config_size": N}])
    cfg["inputs"] = {"n_rule": n_rule}
    with pytest.raises(ValueError, match="stand-in"):
        e2.cells(cfg)
    cfg["mode"] = "full"
    with pytest.raises(ValueError, match="stand-in"):
        e2.cells(cfg)
    cfg["mode"] = "smoke"
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1010, 1011]


# --------------------------------------------------------------------------------------
# Seed roles
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("role", "inst_seeds", "solver_seeds"),
    [
        ("calibration", [2000], [10000]),  # a test instance in a calibration config
        ("calibration", [1000], [20000]),  # a test solver seed
        ("calibration", [999], [10000]),
        ("test", [1000], [20000]),
        ("test", [2000], [10000]),
        ("test", [3000], [20000]),
    ],
)
def test_seed_roles_keep_calibration_and_test_ranges_apart(role, inst_seeds, solver_seeds):
    cfg = _cfg(seed_role=role,
               instances=[{"cell": "T14", "family": "factor", "n": N, "seeds": inst_seeds, "config_size": N}],
               solver_seeds={"exact": solver_seeds, "heuristic": solver_seeds})
    with pytest.raises(ValueError, match="seed"):
        e2.cells(cfg)


def test_matching_seed_roles_pass():
    for role, inst, sol in (("calibration", 1999, 19999), ("test", 2000, 20000), ("test", 2999, 29999)):
        cfg = _cfg(seed_role=role,
                   instances=[{"cell": "T14", "family": "factor", "n": N, "seeds": [inst], "config_size": N}],
                   solver_seeds={"exact": [sol], "heuristic": [sol]})
        assert e2.cells(cfg)


def test_a_smoke_run_never_touches_test_seeds():
    cfg = _cfg(seed_role="test",
               instances=[{"cell": "T14", "family": "factor", "n": N, "seeds": [2000], "config_size": N}],
               solver_seeds={"exact": [20000], "heuristic": [20000]})
    assert e2.cells(cfg)  # a full config may use them (the confirmatory runs)
    cfg["mode"] = "smoke"
    with pytest.raises(ValueError, match="smoke"):
        e2.cells(cfg)


def test_declared_seeds_must_be_those_of_the_grid():
    cfg = _cfg(methods={"source": "explicit", "specs": [SA]})
    cfg["seeds"] = [1000, 1001, 10000, 10001]
    assert e2.cells(cfg)
    cfg["seeds"] = [1000, 1001, 10000]
    with pytest.raises(ValueError, match="seeds"):
        e2.cells(cfg)


def test_solver_seeds_may_be_a_first_and_a_count():
    cfg = _cfg(solver_seeds={"exact": {"first": 10100, "count": 3}, "heuristic": {"first": 10200, "count": 2}})
    cells = e2.cells(cfg)
    assert {c["solver_seed"] for c in cells if c["method"]["name"] == "scip"} == {10100, 10101, 10102}
    assert {c["solver_seed"] for c in cells if c["method"]["name"] == "sa"} == {10200, 10201}
    cfg = _cfg(solver_seeds={"first": 10300, "count": 2})  # one range for both classes
    assert {c["solver_seed"] for c in e2.cells(cfg)} == {10300, 10301}
    with pytest.raises(ValueError, match="count"):
        e2.cells(_cfg(solver_seeds={"exact": {"first": 10100, "count": 0}, "heuristic": [10000]}))


def test_stage_and_budget_are_validated():
    with pytest.raises(ValueError, match="stage"):
        e2.cells(_cfg(stage="warp"))
    with pytest.raises(ValueError, match="budget_s"):
        e2.cells(_cfg(budget_s=0))
    with pytest.raises(ValueError, match=r"params\.instances"):
        e2.cells(_cfg(instances=[]))


# --------------------------------------------------------------------------------------
# The grid: recheck, tuned roles, reference
# --------------------------------------------------------------------------------------


def _tuned(tmp_path, sizes=(N,)) -> str:
    roles = {"exact_rep": SCIP, "qubo_rep": SA, "ls": LS, "sa": SA, "tabu": TABU, "swap_sa": SWAP,
             "h1c_sa": SA_OBJ, "h1c_tabu": TABU}
    return _write(tmp_path, "tuned.json", {
        "kind": "e2-tuned", "sizes": {str(n): {"roles": roles} for n in sizes}})


def test_tuned_roles_become_cells_with_their_roles_and_the_seeds_of_their_class(tmp_path):
    cfg = _cfg(stage="bench", methods={"source": "tuned", "roles": ["exact_rep", "qubo_rep", "ls", "sa"]},
               solver_seeds={"exact": [10000, 10001, 10002], "heuristic": [10000, 10001]})
    cfg["inputs"] = {"tuned": _tuned(tmp_path)}
    cells = e2.cells(cfg)
    ids = {c["method"]["id"]: c["roles"] for c in cells}
    assert ids == {"scip-persp-mineig-symdef": ["exact_rep"], "sa-t": ["qubo_rep", "sa"], "ls-t": ["ls"]}
    assert len(cells) == 2 * (3 + 2 + 2)  # 2 instances × (SCIP 3 seeds, SA 2, LS 2)
    assert all(c["stage"] == "bench" for c in cells)


def test_the_h1c_roles_are_separate_roles(tmp_path):
    cfg = _cfg(stage="h1c", methods={"source": "tuned", "roles": ["h1c_sa", "h1c_tabu"]})
    cfg["inputs"] = {"tuned": _tuned(tmp_path)}
    assert {c["method"]["id"] for c in e2.cells(cfg)} == {"sa-t-obj", "tabu-t"}


def test_tuned_configurations_are_looked_up_by_config_size(tmp_path):
    cfg = _cfg(stage="bench", methods={"source": "tuned", "roles": ["ls"]})
    cfg["inputs"] = {"tuned": _tuned(tmp_path, sizes=(100,))}
    with pytest.raises(ValueError, match="config size 14"):
        e2.cells(cfg)
    cfg["params"]["instances"][0]["config_size"] = 100
    assert e2.cells(cfg)


def test_tuned_source_needs_the_tuned_input_and_known_roles(tmp_path):
    cfg = _cfg(stage="bench", methods={"source": "tuned", "roles": ["ls"]})
    with pytest.raises(ValueError, match="tuned"):
        e2.cells(cfg)
    cfg["inputs"] = {"tuned": _tuned(tmp_path)}
    cfg["params"]["methods"]["roles"] = ["champion"]
    with pytest.raises(ValueError, match="role"):
        e2.cells(cfg)


def test_the_recheck_runs_the_configurations_the_screen_report_names(tmp_path):
    report = {"kind": "e2-screen", "sizes": {str(N): {"recheck": [
        {"spec": SA, "why": ["sa/pi"]}, {"spec": SCIP, "why": ["exact/pi"]}, {"spec": SA, "why": ["sa/gap"]}]}}}
    cfg = _cfg(stage="recheck", methods={"source": "recheck"},
               solver_seeds={"exact": [10003, 10004], "heuristic": [10003]})
    cfg["inputs"] = {"screen": _write(tmp_path, "screen.json", report)}
    cells = e2.cells(cfg)
    assert {c["method"]["id"] for c in cells} == {"sa-t", "scip-persp-mineig-symdef"}  # deduplicated
    assert len(cells) == 2 * (1 + 2)
    assert {c["stage"] for c in cells} == {"recheck"}


def test_the_reference_stage_has_one_long_run_per_instance():
    cfg = _cfg(stage="reference", budget_s=None,
               methods={"source": "reference", "time_limit_s": 600.0, "spec": SCIP},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    cells = e2.cells(cfg)
    assert len(cells) == 2 and {c["seed"] for c in cells} == {1000, 1001}
    assert all(c["time_limit_s"] == 600.0 and c["budget_s"] is None and c["method"] == SCIP for c in cells)


def test_the_reference_stage_defaults_to_the_perspective_variant():
    cfg = _cfg(stage="reference", budget_s=None, methods={"source": "reference", "time_limit_s": 600.0})
    method = e2.cells(cfg)[0]["method"]
    assert (method["name"], method["formulation"], method["diag"], method["usesymmetry"]) == (
        "scip", "perspective", "mineig", "default")


# --------------------------------------------------------------------------------------
# Seeds of the solvers
# --------------------------------------------------------------------------------------


def test_derived_seeds_depend_on_every_coordinate_and_fit_the_samplers():
    base = _cell(_cfg(), name="sa", solver_seed=10000, seed=1000)
    seeds = {e2.derive_seed(base)}
    for change in ({"solver_seed": 10001}, {"seed": 1001}, {"n": 16}, {"method": {**SA, "id": "sa-u"}},
                   {"method": {**SA, "num_sweeps": 51}}, {"config_size": 16}):
        seeds.add(e2.derive_seed({**base, **change}))
    assert len(seeds) == 7  # all different
    assert all(0 <= s < 2**31 for s in seeds)  # dwave-samplers refuse larger seeds
    assert e2.derive_seed(base) == e2.derive_seed(dict(base))
    # the budget and the stage do not enter: a longer run extends a shorter one with the same stream
    assert e2.derive_seed({**base, "budget_s": 60.0, "stage": "bench"}) == e2.derive_seed(base)


# --------------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------------


COMMON = ("status", "f_final", "scale_final", "feasible", "support", "weights", "trajectory",
          "start_time_s", "start_f", "end_s", "overrun_s", "budget_ok", "method_wall_s", "method_cpu_s",
          "first_own_s", "first_own_f", "t_first_improve_s", "t_primal_s", "t_cert_s", "f_at_budget", "seed_used",
          "threads", "instance_id", "K", "s_cont", "binding", "cond_sigma", "cv_vol", "cv_mu",
          "card_mode", "lam", "n_assets", "detail")


@pytest.mark.parametrize("spec", [*HEURISTICS, *EXACT], ids=lambda s: s["id"])
def test_every_method_gives_a_coherent_row(spec):
    cfg = _cfg(methods={"source": "explicit", "specs": [spec]})
    cell = _cell(cfg, id=spec["id"], seed=1000, solver_seed=10000)
    row = _row(cell, cfg)
    assert not set(COMMON) - set(row), set(COMMON) - set(row)
    json.dumps(row, allow_nan=False)  # JSON-safe, no NaN or infinity
    assert row["threads"] == 1 and row["n_assets"] == N and row["config_id"] == spec["id"]
    assert row["card_mode"] == "le" and row["lam"] == 5.0 and row["binding"] is True

    # the portfolio is what the objective says, within the cardinality, and feasible
    problem = e2_instances.build_instance({"family": "factor", "n": N, "seed": 1000}, _fixed()).problem
    w = np.zeros(N)
    w[row["support"]] = row["weights"]
    assert row["feasible"] and len(row["support"]) <= row["K"]
    assert abs(problem.objective_value(w) - row["f_final"]) < 1e-9
    assert abs(w.sum() - 1.0) < 1e-6 and (w >= -1e-9).all()
    assert row["scale_final"] == pytest.approx(
        abs(problem.mu @ w) + 0.5 * problem.risk_aversion * (w @ problem.sigma @ w), rel=1e-12)

    # the common start is the first incumbent and its time is charged to the method
    assert row["start_time_s"] > 0 and math.isfinite(row["start_f"])
    assert row["f_final"] <= row["start_f"] + 1e-7  # SCIP's own solution is accurate to its tolerance
    traj = row["trajectory"]
    assert all(t >= row["start_time_s"] - 1e-9 for t, _ in traj), "trajectory is on the common clock"
    assert all(b[0] >= a[0] and b[1] < a[1] for a, b in pairwise(traj))
    assert all(f < row["start_f"] for _, f in traj)  # the method's own improvements over the start only
    if traj:
        assert traj[-1][1] == pytest.approx(row["f_final"], abs=1e-8)
        assert row["t_first_improve_s"] == traj[0][0] and row["t_primal_s"] == traj[-1][0]
    else:
        assert row["t_first_improve_s"] is None and row["t_primal_s"] == row["start_time_s"]
    assert row["end_s"] >= row["start_time_s"]
    # R10: the time of the method split into building, the search proper and the shared refinement
    parts = row["build_s"], row["solve_s"], row["refine_s"]
    assert all(isinstance(x, float) and x >= 0.0 for x in parts)
    assert sum(parts) <= row["method_wall_s"] * 1.02 + 0.05  # parts of the run, not more than the run
    if spec["name"] in ("sa", "tabu", "swap_sa", "ls"):
        assert row["refine_s"] > 0.0  # the pool is refined by the shared QP
    if spec["name"] in ("scip", "oa"):
        assert row["solve_s"] > 0.0
    if spec["name"] == "scip":
        assert row["refine_s"] == 0.0 and row["build_s"] == pytest.approx(row["detail"]["build_time_s"])
    assert row["method_cpu_s"] <= row["method_wall_s"] * 1.05 + 0.05  # one thread: CPU time <= wall time
    # the incumbent at the budget is the one of the curve: never better than the final value, and equal
    # to it when the run ended inside the budget
    assert row["f_at_budget"] >= min(row["f_final"], row["start_f"]) - 1e-8
    if row["end_s"] <= row["budget_s"]:
        assert row["f_at_budget"] == pytest.approx(min(row["f_final"], row["start_f"]), abs=1e-8)


@pytest.mark.parametrize("spec", [SA, SA_OBJ, TABU, SWAP], ids=lambda s: s["id"])
def test_qubo_heuristics_run_batches_until_the_budget_and_report_their_own_first_solution(spec):
    cfg = _cfg(methods={"source": "explicit", "specs": [spec]})
    row = _row(_cell(cfg, id=spec["id"], seed=1000, solver_seed=10000), cfg)
    assert row["budget_s"] == BUDGET and row["end_s"] >= BUDGET * 0.9  # the outer loop uses the budget
    assert row["overrun_s"] < BUDGET and row["budget_ok"] is True
    assert row["first_own_s"] is not None and row["first_own_s"] > row["start_time_s"]
    assert row["first_own_f"] is not None and math.isfinite(row["first_own_f"])  # its value, next to its time
    assert row["f_final"] <= row["first_own_f"] + 1e-12 or row["f_final"] == row["start_f"]
    assert row["detail"]["n_batches"] >= 1 and row["detail"]["n_refined"] >= 1
    assert row["detail"]["pool_size"] >= 2  # the start and at least one subset of its own
    assert row["t_cert_s"] is None and row["status"] in ("budget", "max_batches")
    assert "certified" not in row  # a heuristic proves nothing


def test_local_search_stops_at_its_restarts_and_reports_its_own_first_solution():
    cfg = _cfg(methods={"source": "explicit", "specs": [LS]})
    row = _row(_cell(cfg, id=LS["id"], seed=1000, solver_seed=10000), cfg)
    assert row["status"] in ("restarts", "exhausted", "budget")
    assert row["first_own_s"] is not None and row["first_own_s"] >= row["start_time_s"]
    assert row["detail"]["restarts_done"] >= 1 and row["detail"]["n_evals"] >= 1
    assert row["budget_ok"] is True


def test_the_objective_scaled_beta_range_is_calibrated_inside_the_timed_region():
    cfg = _cfg(methods={"source": "explicit", "specs": [SA_OBJ]})
    row = _row(_cell(cfg, id="sa-t-obj", seed=1000, solver_seed=10000), cfg)
    assert row["detail"]["beta_mode"] == "objective" and row["detail"]["beta_hot"] > 0
    assert row["detail"]["beta_cold"] > row["detail"]["beta_hot"]
    assert row["detail"]["build_s"] > 0  # the QUBO and the temperature calibration are charged


@pytest.mark.parametrize("spec", EXACT, ids=lambda s: s["id"])
def test_exact_methods_certify_a_tiny_instance_and_report_t_primal_and_t_cert(spec):
    cfg = _cfg(budget_s=20.0, methods={"source": "explicit", "specs": [spec]})
    row = _row(_cell(cfg, id=spec["id"], seed=1000, solver_seed=10000), cfg)
    assert row["status"] in ("optimal", "gaplimit")
    assert row["certified"] is True and row["proved"] is True
    assert row["t_cert_s"] is not None and row["start_time_s"] <= row["t_cert_s"] <= row["end_s"] + 1e-6
    assert row["t_primal_s"] <= row["t_cert_s"] + 1e-9  # the incumbent is final before the proof
    assert row["detail"]["abs_gap_achieved"] <= 1e-6 * row["scale_final"] + 1e-12
    assert row["detail"]["abs_gap_limit"] > 0  # limits/absgap = 1e-6·0.5·s(start)
    assert row["end_s"] < 20.0 and row["budget_ok"] is True
    if spec["name"] == "scip":
        assert row["detail"]["warm_start_used"] is True and row["detail"]["permutation_seed"] == row["seed_used"]


def test_a_scip_variant_applies_its_usesymmetry_parameter():
    cfg = _cfg(budget_s=20.0, methods={"source": "explicit", "specs": [SCIP_BIGM]})
    row = _row(_cell(cfg, id=SCIP_BIGM["id"], seed=1000, solver_seed=10000), cfg)
    assert row["detail"]["scip_params"] == {"misc/usesymmetry": 0}
    cfg = _cfg(budget_s=20.0, methods={"source": "explicit", "specs": [SCIP]})
    row = _row(_cell(cfg, id=SCIP["id"], seed=1000, solver_seed=10000), cfg)
    assert row["detail"]["scip_params"] == {}


def test_a_scip_spec_with_aggressive_heuristics_runs_with_them_and_records_them():
    spec = {**SCIP_BIGM, "id": "scip-bigm-sym0-heuraggr", "heuristics": "aggressive"}
    cfg = _cfg(budget_s=20.0, methods={"source": "explicit", "specs": [spec]})
    row = _row(_cell(cfg, id=spec["id"], seed=1000, solver_seed=10000), cfg)
    assert row["detail"]["heuristics"] == "aggressive" and row["method"]["heuristics"] == "aggressive"
    cfg = _cfg(budget_s=20.0, methods={"source": "explicit", "specs": [SCIP_BIGM]})
    row = _row(_cell(cfg, id=SCIP_BIGM["id"], seed=1000, solver_seed=10000), cfg)
    assert row["detail"]["heuristics"] == "default"


def test_exact_runs_are_cut_by_the_end_to_end_budget_not_the_solve_limit():
    cfg = _cfg(budget_s=0.05, methods={"source": "explicit", "specs": [SCIP]})
    row = _row(_cell(cfg, id=SCIP["id"], seed=1000, solver_seed=10000), cfg)
    assert row["detail"]["budget_s"] == 0.05 and row["detail"]["clock_offset_s"] == pytest.approx(
        row["start_time_s"])
    assert row["certified"] in (True, False)


def test_warm_up_runs_every_solver_once_per_process(monkeypatch):
    monkeypatch.setattr(e2, "_warmed_up", False)
    e2._warm_up()
    assert e2._warmed_up is True
    calls = []
    monkeypatch.setattr(e2, "common_start", lambda *a, **k: calls.append(1))
    e2._warm_up()  # a second call does nothing
    assert calls == []


def test_a_cell_warms_up_unless_the_config_says_otherwise(monkeypatch):
    warmed = []
    monkeypatch.setattr(e2, "_warm_up", lambda *a: warmed.append(1))
    cfg = _cfg(methods={"source": "explicit", "specs": [LS]})
    cell = _cell(cfg, id="ls-t", seed=1000, solver_seed=10000)
    e2.run_cell(cell, cfg)
    assert warmed == []  # the test config switches it off
    cfg["params"]["warm_up"] = True
    e2.run_cell(cell, cfg)
    assert warmed == [1]


def test_budget_compliance_flags_an_overrun_beyond_the_tolerance():
    assert e2.budget_compliance(9.0, 10.0, 0.05) == (0.0, True)
    over, ok = e2.budget_compliance(10.4, 10.0, 0.05)
    assert over == pytest.approx(0.4) and ok is True  # within 5 % of the budget
    over, ok = e2.budget_compliance(20.5, 10.0, 0.05)
    assert over == pytest.approx(10.5) and ok is False  # e.g. an SDP that ignores its limit


def test_the_row_records_the_incumbent_at_the_budget_from_the_trajectory():
    cfg = _cfg(methods={"source": "explicit", "specs": [SA]})
    row = _row(_cell(cfg, id="sa-t", seed=1001, solver_seed=10001), cfg)
    times, values = incumbent_curve(row["trajectory"], start=(row["start_time_s"], row["start_f"]))
    at = values[np.searchsorted(times, row["budget_s"], side="right") - 1]
    assert row["f_at_budget"] == pytest.approx(at, abs=1e-12)


def test_the_same_cell_and_seed_start_from_the_same_portfolio():
    cfg = _cfg(methods={"source": "explicit", "specs": [LS]})
    cell = _cell(cfg, id="ls-t", seed=1000, solver_seed=10000)
    a, b = e2.run_cell(cell, cfg), e2.run_cell(cell, cfg)
    assert a["start_f"] == b["start_f"] and a["seed_used"] == b["seed_used"]
    assert a["f_final"] == pytest.approx(b["f_final"], abs=1e-9)  # LS with 3 restarts: reproducible


def test_an_unknown_family_or_a_missing_parameter_is_an_error_of_the_cell():
    cfg = _cfg(methods={"source": "explicit", "specs": [LS]})
    cell = _cell(cfg, id="ls-t", seed=1000, solver_seed=10000)
    with pytest.raises(ValueError, match="family"):
        e2.run_cell({**cell, "family": "fg"}, cfg)
    cfg["params"]["fixed"] = {"fraction": 0.5}
    with pytest.raises(ValueError, match="lam"):
        e2.run_cell(cell, cfg)


# --------------------------------------------------------------------------------------
# ORlib (Σz = K) and dense
# --------------------------------------------------------------------------------------


@pytest.fixture
def orlib_cfg(tmp_path):
    (tmp_path / "port1.txt").write_text(orlib_text(31, seed=3))
    fixed = _fixed(orlib_dir=str(tmp_path), orlib_k=5, orlib_w_min=0.01, orlib_w_max=1.0, orlib_m=25)
    return _cfg(instances=[{"cell": "C5", "family": "orlib", "ports": [1], "lam_indices": [8],
                            "config_size": 31}], fixed=fixed, budget_s=20.0,
                solver_seeds={"exact": [10000], "heuristic": [10000]},
                methods={"source": "explicit", "specs": [LS, SCIP, OA]})


@pytest.mark.parametrize("spec", [LS, SCIP, OA], ids=lambda s: s["id"])
def test_orlib_cells_hold_exactly_k_assets_above_the_minimum_purchase(orlib_cfg, spec):
    cell = _cell(orlib_cfg, id=spec["id"])
    row = _row(cell, orlib_cfg)
    assert row["card_mode"] == "eq" and row["K"] == 5 and row["n_assets"] == 31
    assert row["instance_id"] == "orlib/1/08" and row["feasible"]
    assert len(row["support"]) == 5 and min(row["weights"]) >= 0.01 - 1e-9
    assert row["binding"] == (row["s_cont"] > 5)


def test_lambda_indices_may_be_a_count_or_a_list(orlib_cfg):
    orlib_cfg["params"]["instances"][0]["lam_indices"] = 4
    assert {c["lam_index"] for c in e2.cells(orlib_cfg)} == {0, 1, 2, 3}
    orlib_cfg["params"]["instances"][0]["lam_indices"] = [2, 7]
    assert {c["lam_index"] for c in e2.cells(orlib_cfg)} == {2, 7}


def test_dense_cells_run_with_k_a_quarter_of_n():
    cfg = _cfg(instances=[{"cell": "D1", "family": "dense", "n": 12, "seeds": [1000], "config_size": 12}],
               methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    row = _row(e2.cells(cfg)[0], cfg)
    assert row["K"] == 3 and row["instance_id"] == "dense/12/1000"


# --------------------------------------------------------------------------------------
# Instances where K does not bind
# --------------------------------------------------------------------------------------


def _slack_cfg(**group):
    """Factor instances with K = s_cont (fraction 1): the constraint does not bind."""
    cfg = _cfg(fixed=_fixed(fraction=1.0), methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    cfg["params"]["instances"][0].update(group)
    return cfg


def test_a_cell_marks_the_instances_the_plan_leaves_out():
    cfg = _slack_cfg(skip_nonbinding=True)
    assert all(c["skip_nonbinding"] is True for c in e2.cells(cfg))
    assert all(c["skip_nonbinding"] is False for c in e2.cells(_slack_cfg()))
    cfg["params"]["skip_nonbinding"] = True  # the stage default, overridden by the group
    cfg["params"]["instances"][0]["skip_nonbinding"] = False
    assert all(c["skip_nonbinding"] is False for c in e2.cells(cfg))
    cfg["params"]["instances"][0].pop("skip_nonbinding")
    assert all(c["skip_nonbinding"] is True for c in e2.cells(cfg))


def test_a_nonbinding_instance_is_skipped_without_running_a_method():
    cfg = _slack_cfg(skip_nonbinding=True)
    row = _row(e2.cells(cfg)[0], cfg)
    assert row["skipped"] == "not_binding" and row["status"] == "skipped" and row["binding"] is False
    assert row["K"] == row["s_cont"] and row["instance_id"] == "factor/14/1000"
    assert "f_final" not in row and "trajectory" not in row and "start_f" not in row
    json.dumps(row, allow_nan=False)
    # the same instance without the flag is run (RQ2 leaves it out, D1 does not)
    row = _row(e2.cells(_slack_cfg())[0], _slack_cfg())
    assert "skipped" not in row and row["binding"] is False and math.isfinite(row["f_final"])


def test_a_binding_instance_is_never_skipped():
    cfg = _cfg(methods={"source": "explicit", "specs": [LS]}, solver_seeds={"exact": [10000], "heuristic": [10000]})
    cfg["params"]["instances"][0]["skip_nonbinding"] = True
    row = _row(e2.cells(cfg)[0], cfg)
    assert "skipped" not in row and row["binding"] is True


# --------------------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------------------


def test_the_reference_run_certifies_polishes_and_reports_the_optimum():
    cfg = _cfg(stage="reference", budget_s=None, methods={"source": "reference", "time_limit_s": 60.0})
    row = _row(e2.cells(cfg)[0], cfg)
    assert row["certified"] is True and row["ref_type"] == "certified"
    assert row["f_star"] <= row["f_final"] + 1e-12 and row["scale_star"] > 0
    assert row["detail"]["time_limit_s"] == 60.0 and row["detail"]["budget_s"] is None
    assert row["detail"]["abs_gap_limit"] > 0  # limits/absgap = 1e-6·0.5·s(start)
    json.dumps(row, allow_nan=False)


def test_the_reference_is_never_above_a_heuristic_on_the_same_instance():
    ref_cfg = _cfg(stage="reference", budget_s=None, methods={"source": "reference", "time_limit_s": 60.0})
    ref = _row(e2.cells(ref_cfg)[0], ref_cfg)
    cfg = _cfg(methods={"source": "explicit", "specs": [SA, LS]})
    for spec_id in ("sa-t", "ls-t"):
        heur = _row(_cell(cfg, id=spec_id, seed=1000, solver_seed=10000), cfg)
        assert ref["f_star"] <= heur["f_final"] + 1e-9


# --------------------------------------------------------------------------------------
# Through the runner and the ingest
# --------------------------------------------------------------------------------------


def _yaml(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "e2.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_smoke_run_registers_one_row_per_cell_without_errors(tmp_path):
    cfg = _cfg(methods={"source": "explicit", "specs": [SA, SCIP]})
    cfg["smoke"] = {"params": {"instances": [{"cell": "T14", "family": "factor", "n": N, "seeds": [1000],
                                              "config_size": N}],
                               "solver_seeds": {"exact": [10000], "heuristic": [10000]}}}
    runs = tmp_path / "runs"
    assert run.main(["--config", str(_yaml(tmp_path, cfg)), "--runs-dir", str(runs), "--smoke"]) == 0
    run_dir = next(p for p in runs.iterdir() if p.is_dir())
    rows = _rows(run_dir / "results.jsonl")
    assert len(rows) == 2 and not any(r.get("error") for r in rows)
    assert {r["method"]["name"] for r in rows} == {"sa", "scip"}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "smoke" and manifest["n_errors"] == 0 and manifest["threads"]["pools"] is not None


def test_shards_resume_without_duplicating_rows_and_ingest_registers_the_grid(tmp_path):
    cfg = _cfg(methods={"source": "explicit", "specs": [LS, SCIP]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    path = _yaml(tmp_path, cfg)
    shards = tmp_path / "incoming"
    shards.mkdir()
    for i in range(2):
        assert run.main(["--config", str(path), "--shard", str(i), "--n-shards", "2",
                         "--out", str(shards / f"s{i}.jsonl")]) == 0
    before = [len(_rows(shards / f"s{i}.jsonl")) for i in range(2)]
    for i in range(2):  # a restart finds nothing to do and appends nothing
        assert run.main(["--config", str(path), "--shard", str(i), "--n-shards", "2",
                         "--out", str(shards / f"s{i}.jsonl")]) == 0
    assert [len(_rows(shards / f"s{i}.jsonl")) for i in range(2)] == before
    assert sum(before) == 2 * 2
    runs = tmp_path / "runs"
    assert ingest.main(["--config", str(path), "--shards", str(shards / "s0.jsonl"),
                        str(shards / "s1.jsonl"), "--runs-dir", str(runs), "--allow-missing-meta"]) == 0
