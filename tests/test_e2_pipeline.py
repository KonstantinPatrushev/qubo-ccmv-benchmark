"""The pilot of E2 end to end on tiny real runs: eligibility → reference → screen → report → recheck → tuned
report → benchmark.

Checks that the reports one stage writes are the inputs the next stage reads (the contract between
:mod:`cif.experiments.e2_pilot`, :mod:`cif.experiments.e2_eligibility` and :mod:`cif.experiments.e2_benchmark`),
through the runner and the registry, on 16-asset calibration instances with budgets of a fraction of a second.
The calibration sets are screened first (§12 A1): of the instances 1000-1007 those whose start is optimal are
left out, and everything after runs on the non-trivial ones only. Nothing is asserted about which method is
better: the numbers of such runs mean nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from cif.experiments import e2_benchmark as e2
from cif.experiments import e2_pilot as pilot
from cif.experiments import run

N = 16
CELL = f"T{N}"
SCREEN_BUDGET = 0.15
LONG_BUDGET = 0.3
STATE = "eligibility_calibration"

SPECS = [
    {"name": "sa", "id": "sa-00", "num_reads_b": 10, "num_sweeps": 30, "beta": "default", "penalty": 10.0, "m_b": 20},
    {"name": "sa", "id": "sa-01", "num_reads_b": 10, "num_sweeps": 30, "beta": "objective", "penalty": 3.0, "m_b": 20},
    {"name": "tabu", "id": "tabu-00", "num_restarts_b": 1, "tenure": 5, "num_reads_b": 3, "penalty": 10.0, "m_b": 20},
    {"name": "tabu", "id": "tabu-01", "num_restarts_b": 2, "tenure": 5, "num_reads_b": 3, "penalty": 30.0, "m_b": 20},
    {"name": "swap_sa", "id": "swap_sa-00", "num_reads_b": 5, "num_sweeps": 10, "m_b": 20},
    {"name": "ls", "id": "ls-00", "num_restarts": 2, "improvement": "first", "rcl_size": 3},
    {"name": "scip", "id": "scip-persp-mineig-symdef", "formulation": "perspective", "diag": "mineig",
     "usesymmetry": "default"},
    {"name": "scip", "id": "scip-bigm-sym0", "formulation": "bigm", "usesymmetry": 0},
]


def _eligible(field: str = "selected") -> list[dict]:
    return [{"cell": CELL, "family": "factor", "n": N, "config_size": N,
             "eligible": {"report": STATE, "cell": CELL, "field": field, "require_enough": True}}]


def _config(name: str, stage: str, *, budget, methods: dict, seeds: dict, inputs: dict | None = None,
            instances=None) -> dict:
    return {
        "stream": "E2", "name": name, "experiment": "cif.experiments.e2_benchmark", "inputs": inputs or {},
        "params": {
            "stage": stage, "seed_role": "calibration", "budget_s": budget, "warm_up": False,
            "fixed": {"lam": 5.0, "fraction": 0.5},
            "solver": {"mip_gap": 1e-6, "abs_gap_rel": 1e-6, "abs_gap_safety": 0.5},
            "instances": instances or _eligible(),
            "solver_seeds": seeds, "methods": methods,
        },
    }


def _run(tmp_path: Path, cfg: dict, name: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    return next(p for p in sorted(runs.iterdir()) if p.is_dir() and f"-E2-{name}-" in p.name)


def _rows(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]


def test_the_pilot_chain_hands_its_reports_to_the_next_stage(tmp_path):
    e2.clear_caches()
    # 0. the screening: seeds 1000-1007 of N = 16 until 3 non-trivial instances (the state is a stub before round 1)
    state = tmp_path / "eligibility_calibration.json"
    state.write_text(json.dumps({"kind": "e2-eligibility", "stub": True, "cells": {}}))
    scan = {"head": list(range(1000, 1008)), "then_from": 1024, "cap": 16, "target": 3, "min_nontrivial": 3,
            "first_block": 8, "state": STATE}
    elig_cfg = _config(
        "elig", "eligibility", budget=None, methods={"source": "reference", "time_limit_s": 60.0},
        seeds={"exact": [10000], "heuristic": [10000]}, inputs={STATE: str(state)},
        instances=[{"cell": CELL, "family": "factor", "n": N, "config_size": N, "scan": scan}])
    elig_cfg["params"]["eligibility"] = {"tau": 1e-3, "certificate": "full", "margin": 1e-3}
    elig_run = _run(tmp_path, elig_cfg, "elig")
    screened = _rows(elig_run)
    assert len(screened) == 8 and {r["eligibility"] for r in screened} <= {"trivial", "nontrivial", "undecided"}
    report_json = tmp_path / "eligibility_report.json"
    assert pilot.main(["eligibility", "--config", str(tmp_path / "elig.yaml"), "--rows", str(elig_run),
                       "--out", str(report_json)]) == 0
    elig = json.loads(report_json.read_text())
    cell = elig["cells"][CELL]
    assert elig["kind"] == "e2-eligibility" and elig["done"] is True and elig["role"] == "calibration"
    assert cell["selected"] == [1001, 1005, 1006]  # the first three non-trivial seeds; 1000 and 1002-1004 are not
    assert cell["needs_reference"] == [] and cell["enough"] is True  # screened to the optimum: nothing to certify later
    assert all(r["certified"] is True and r["ref_type"] == "certified" for r in screened if r["eligibility"] == "nontrivial")
    # the scan stops at the third non-trivial instance, 1006: seven seeds, four of them trivial, 1007 is not part of it
    assert cell["protocol"]["n_nontrivial"] == 3 and cell["protocol"]["n_scanned"] == 7
    assert cell["protocol"]["n_trivial"] == 4 and cell["prefix"] == 8
    assert any(i["source"].endswith("elig.yaml") for i in elig["inputs"]) and any(i.get("run_id") for i in elig["inputs"])
    # the report is what a later round of the same scan would read: nothing is left to screen
    elig_cfg["inputs"] = {STATE: str(report_json)}
    assert e2.cells(elig_cfg) == []

    # 1. the 10 s screen (here 0.15 s) on the non-trivial calibration instances only
    inputs = {STATE: str(report_json)}
    screen_run = _run(tmp_path, _config(
        "screen", "screen", budget=SCREEN_BUDGET, methods={"source": "explicit", "specs": SPECS},
        seeds={"exact": [10000, 10001], "heuristic": [10000, 10001]}, inputs=inputs), "screen")
    assert {r["seed"] for r in _rows(screen_run)} == {1001, 1005, 1006}
    screen_json = tmp_path / "screen.json"
    assert pilot.main(["screen", "--rows", str(screen_run), "--reference", str(elig_run),
                       "--budget", str(SCREEN_BUDGET), "--out", str(screen_json)]) == 0  # its rows are references
    screen = json.loads(screen_json.read_text())
    assert screen["inputs"][0]["run_id"] == screen_run.name  # the report names the run it read
    plan_ids = {item["spec"]["id"] for item in screen["sizes"][str(N)]["recheck"]}
    assert plan_ids <= {s["id"] for s in SPECS} and len(plan_ids) >= 5
    # on a non-trivial set the start is not within τ of the optimum: the degenerate-start warning stays quiet
    assert screen["sizes"][str(N)]["start_gap"]["share_within_tau"] == 0.0
    assert not any("start" in w for w in screen["warnings"])

    # 2. the recheck reads the plan: the named configurations, on other solver seeds
    recheck_run = _run(tmp_path, _config(
        "recheck", "recheck", budget=LONG_BUDGET, methods={"source": "recheck"},
        seeds={"exact": [10003, 10004], "heuristic": [10003, 10004]}, inputs={**inputs, "screen": str(screen_json)}),
        "recheck")
    rows = _rows(recheck_run)
    assert {r["config_id"] for r in rows} == plan_ids and len(rows) == len(plan_ids) * 3 * 2

    # 3. the tuned report: every role of the benchmark is filled, no size borrows
    tuned_json = tmp_path / "tuned.json"
    assert pilot.main(["tune", "--screen", str(screen_run), "--recheck", str(recheck_run),
                       "--reference", str(elig_run), "--eligibility", str(report_json),
                       "--budget", str(SCREEN_BUDGET), "--budget-long", str(LONG_BUDGET),
                       "--out", str(tuned_json)]) == 0
    tuned = json.loads(tuned_json.read_text())
    roles = tuned["sizes"][str(N)]["roles"]
    assert set(roles) == set(e2.ROLES) and "borrowed_from" not in tuned["sizes"][str(N)]
    assert roles["exact_rep"]["name"] == "scip" and roles["ls"]["name"] == "ls"
    assert roles["qubo_rep"]["name"] in ("sa", "tabu", "swap_sa")

    # 4. the benchmark and H1c stages read the tuned report (calibration seeds: this is a rehearsal)
    for name, stage, role_list in (("bench", "bench", ["exact_rep", "qubo_rep", "ls", "sa", "tabu", "swap_sa"]),
                                   ("h1c", "h1c", ["h1c_sa", "h1c_tabu"])):
        cfg = _config(name, stage, budget=SCREEN_BUDGET, methods={"source": "tuned", "roles": role_list},
                      seeds={"exact": [10010, 10011, 10012], "heuristic": [10010, 10011]},
                      inputs={**inputs, "tuned": str(tuned_json)})
        cells = e2.cells({**cfg, "mode": "full"})
        assert cells and all(c["method"] in roles.values() for c in cells)
        assert {r for c in cells for r in c.get("roles", [])} == set(role_list)
        assert {c["seed"] for c in cells} == {1001, 1005, 1006}  # the non-trivial instances, nothing else
    bench_run = _run(tmp_path, _config(
        "bench", "bench", budget=SCREEN_BUDGET, methods={"source": "tuned", "roles": ["exact_rep", "qubo_rep", "ls"]},
        seeds={"exact": [10010, 10011, 10012], "heuristic": [10010, 10011]},
        inputs={**inputs, "tuned": str(tuned_json)}), "bench")
    bench_rows = _rows(bench_run)
    assert not any(r.get("error") for r in bench_rows)
    assert {r["config_id"] for r in bench_rows} == {roles[r]["id"] for r in ("exact_rep", "qubo_rep", "ls")}

    # 5. the rules that need the representatives at the long budget: PI⁺ is positive on every non-trivial instance
    n_rule_json = tmp_path / "n_rule.json"
    assert pilot.main(["n-rule", "--rows", str(recheck_run), str(elig_run), "--tuned", str(tuned_json),
                       "--eligibility", str(report_json), "--cell", f"C1={CELL}:{N}", "--budget", str(LONG_BUDGET),
                       "--out", str(n_rule_json)]) == 0
    n_rule = json.loads(n_rule_json.read_text())
    assert 20 <= n_rule["n_instances"]["C1"] <= 40 and n_rule["cells"]["C1"]["n_calibration"] == 3
    assert n_rule["measure"] == "PI+" and "descriptive" not in n_rule["cells"]["C1"]
    ttt_json = tmp_path / "ttt.json"
    assert pilot.main(["ttt-instances", "--rows", str(recheck_run), str(elig_run), "--tuned", str(tuned_json),
                       "--label", CELL, "--config-size", str(N), "--k", "2", "--budget", str(LONG_BUDGET),
                       "--out", str(ttt_json)]) == 0
    assert set(json.loads(ttt_json.read_text())["instances"]) <= {1001, 1005, 1006}

    # 6. a benchmark config reads the n rule and the TTT selection from the reports
    cfg = _config("n", "bench", budget=SCREEN_BUDGET, methods={"source": "tuned", "roles": ["ls"]},
                  seeds={"exact": [10010], "heuristic": [10010]},
                  inputs={"tuned": str(tuned_json), "n_rule": str(n_rule_json), "ttt_instances": str(ttt_json)},
                  instances=[{"cell": "C1", "family": "factor", "n": N, "first_seed": 1010, "count_from": "n_rule",
                              "config_size": N},
                             {"cell": "TTT", "family": "factor", "n": N, "seeds_from": "ttt_instances",
                              "config_size": N}])
    cells = e2.cells(cfg)
    assert len([c for c in cells if c["cell"] == "C1"]) == n_rule["n_instances"]["C1"] * 1
    assert sorted({c["seed"] for c in cells if c["cell"] == "TTT"}) == json.loads(ttt_json.read_text())["instances"]

    # 7. the cold stage on the same instances (calibration seeds here), one seed per method and instance
    cold_run = _run(tmp_path, _config(
        "cold", "cold", budget=SCREEN_BUDGET, methods={"source": "tuned", "roles": ["exact_rep", "qubo_rep", "ls"]},
        seeds={"exact": [10020], "heuristic": [10020]}, inputs={**inputs, "tuned": str(tuned_json)}), "cold")
    cold_rows = _rows(cold_run)
    assert len(cold_rows) == 3 * 3 and all(r["cold"] is True and r["start_f"] is None for r in cold_rows)
    refs = pilot.reference_table([*screened, *cold_rows])
    for row in cold_rows:  # the honest variant is defined on the cold rows, and not zero: nobody starts at the optimum
        assert 0.0 < pilot.row_pi(row, refs[row["instance_id"]], variant="own") <= 1.0


def test_the_tau_level_of_the_screening_leaves_the_reference_runs_to_a_stage_of_their_own(tmp_path):
    """The other level of §12 A1: the screening stops as soon as the question is answered, so the selected instances
    need their certificate: ``needs_reference`` names them and the reference stage runs exactly those."""
    e2.clear_caches()
    state = tmp_path / "eligibility_calibration.json"
    state.write_text(json.dumps({"kind": "e2-eligibility", "stub": True, "cells": {}}))
    elig_cfg = _config(
        "elig", "eligibility", budget=None, methods={"source": "reference", "time_limit_s": 60.0},
        seeds={"exact": [10000], "heuristic": [10000]}, inputs={STATE: str(state)},
        instances=[{"cell": CELL, "family": "factor", "n": N, "config_size": N, "seeds": [1000, 1001, 1005, 1006]}])
    elig_cfg["params"]["eligibility"] = {"tau": 1e-3, "certificate": "tau", "margin": 1e-3}
    elig_run = _run(tmp_path, elig_cfg, "elig")
    screened = _rows(elig_run)
    assert {r["seed"]: r["eligibility"] for r in screened} == {
        1000: "trivial", 1001: "nontrivial", 1005: "nontrivial", 1006: "nontrivial"}
    assert all(r["evidence"]["certificate"] == "tau" and "f_star" not in r for r in screened)
    report_json = tmp_path / "eligibility_report.json"
    assert pilot.main(["eligibility", "--config", str(tmp_path / "elig.yaml"), "--rows", str(elig_run),
                       "--out", str(report_json)]) == 0
    cell = json.loads(report_json.read_text())["cells"][CELL]
    assert cell["selected"] == [1001, 1005, 1006] == cell["needs_reference"]
    ref_run = _run(tmp_path, _config(
        "ref", "reference", budget=None, methods={"source": "reference", "time_limit_s": 60.0},
        seeds={"exact": [10000], "heuristic": [10000]}, inputs={STATE: str(report_json)},
        instances=_eligible("needs_reference")), "ref")
    ref_rows = _rows(ref_run)
    assert sorted(r["seed"] for r in ref_rows) == [1001, 1005, 1006] and all(r["certified"] for r in ref_rows)
    refs = pilot.reference_table([*screened, *ref_rows])
    assert {i: r.ref_type for i, r in refs.items() if i != "factor/16/1000"} == {
        "factor/16/1001": "certified", "factor/16/1005": "certified", "factor/16/1006": "certified"}
    # the exact ĝ_start of the certified instances agrees with the side of τ that the screening chose
    audit = pilot.e2_eligibility.reference_check(screened, ref_rows)
    assert audit["n_checked"] == 3 and audit["flipped"] == []
