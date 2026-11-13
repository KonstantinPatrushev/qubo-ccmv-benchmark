"""The YAML configs of E2 (``configs/E2``): they parse, their grids are the ones the plan describes, the
confirmatory ones use test seeds only in full mode and calibration seeds in smoke mode, and the timing
stages respect the concurrency limit of §6. The reports of the pilot that the configs read are committed
since the pilot (§12, 2026-10-12); the grid tests still stand them in with synthetic ones in a temporary
directory, so that the designs they check do not depend on what the scans selected. Since §12 A1 (2026-10-11) the
instances of every stage after the screening are the non-trivial ones that the scan selected, so the stand-in
eligibility reports carry the selection and the grids are those of the selection.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from cif.experiments import e2_benchmark as e2
from cif.experiments import e2_eligibility as el
from cif.experiments import e2_space
from cif.experiments.run import effective_config, grid, load_config
from cif.instances.orlib import ORLIB_N

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "E2"
REGISTERED = ROOT / "runs" / "20261011-211545-E2-ref_calibration-936867" / "results.jsonl"
#: files of the validation of the exact stack (M11): its own experiment module and its own tests (tests/test_e2_m11.py,
#: tests/test_e2_m11_analysis.py); the second one is the published numbers, data and not a stage
OTHER_MODULES = {"m11", "m11_bcw_published"}
NAMES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem not in OTHER_MODULES)
ELIGIBILITY = {"eligibility_calibration", "eligibility_test", "eligibility_dense"}
REPORTS = {"screen", "tuned", "n_rule", "ttt_instances", "c4_lambda"} | ELIGIBILITY
N_RULE = {"C1": 20, "C2": 21, "C3": 22, "C4": 23}
BOOT_LAM = 10.0
#: the seeds the stand-in scans "selected": every third seed from 2000 / 1000 (so that they are not a first_seed + count)
CAL_SEEDS = {"T100": [1001, 1005, 1006, 1007, 1014, 1016, 1019, 1020],
             "T200": [1009, 1012, 1013, 1014, 1030, 1031, 1034, 1040],
             "T400": [1016, 1017, 1019, 1021, 1023, 1025, 1030, 1031],
             "CB200": [1100, 1102, 1105, 1106, 1111, 1113, 1114, 1120, 1121, 1122]}
TEST_SEEDS = {"C1": [2000 + 3 * i for i in range(20)], "C2": [2001 + 4 * i for i in range(21)],
              "C3": [2002 + 5 * i for i in range(22)], "C4": [2003 + 6 * i for i in range(23)]}
C5_POINTS = [[1, 3], [1, 7], [2, 5], [2, 9], [3, 0], [3, 11], [4, 2], [4, 8], [5, 4], [5, 12], [5, 13]]
D1_SEEDS = {"D1_100": [2000 + 7 * i for i in range(20)], "D1_200": [2002 + 9 * i for i in range(20)]}


def test_all_stage_configs_are_present():
    assert NAMES == ["bench_boot", "bench_dense", "bench_exact_oa", "bench_exact_scip_aggr", "bench_factor",
                     "bench_orlib", "calib_boot", "cold", "elig_calibration", "elig_dense", "elig_test", "h1c",
                     "ref_bench", "ref_calibration", "ttt", "tune_recheck", "tune_screen"]


@pytest.mark.parametrize("name", NAMES)
def test_every_config_follows_the_runner_contract(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg["stream"] == "E2" and cfg["name"] == name
    assert cfg["experiment"] == "cif.experiments.e2_benchmark"
    assert cfg["params"]["stage"] in e2.STAGES
    assert set(cfg.get("inputs") or {}) <= REPORTS | {"sp500_pit", "orlib"}, "unknown input"
    for label, path in (cfg.get("inputs") or {}).items():
        if label in REPORTS:
            assert path == f"configs/E2/pilot/{label}.json", f"{name}: {label}"
    assert "smoke" in cfg, "every stage can be rehearsed"


def test_the_hand_written_stand_in_reports_are_flagged_as_such():
    """The runs refuse a flagged report outside a smoke run (``e2_benchmark._report``): the flag is what
    keeps a stand-in that was copied into ``pilot/`` from feeding a confirmatory stage."""
    paths = sorted((CONFIG_DIR / "smoke").glob("*.json"))
    assert {p.stem for p in paths} == REPORTS
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["smoke"] is True and str(report["kind"]).startswith("e2-"), path.name


def test_a_committed_scan_state_is_a_stub_or_comes_from_registered_runs_and_unfinished_cells_are_refused():
    for label in sorted(ELIGIBILITY):
        report = json.loads((CONFIG_DIR / "pilot" / f"{label}.json").read_text(encoding="utf-8"))
        assert report["kind"] == "e2-eligibility"
        if report.get("stub"):  # before the first round: nothing to run from
            assert report["cells"] == {}
            with pytest.raises(ValueError, match="stub"):
                el.selected_units(report, "C1")
            continue
        # a computed state: every run it was made from is registered here, with the hash it was read with
        runs = [i for i in report["inputs"] if i.get("run_id")]
        assert runs, label
        for inp in runs:
            assert (ROOT / inp["source"]).is_dir() and inp["git_dirty"] is False, inp["source"]
        # a later stage takes no unit of a cell whose scan is not finished
        for cell, st in report["cells"].items():
            if not st["done"]:
                with pytest.raises(ValueError, match="not finished"):
                    el.selected_units(report, cell)
    for name in ("screen.json", "tuned.json", "n_rule.json", "ttt_instances.json"):
        path = CONFIG_DIR / "pilot" / name
        if path.exists():  # pilot values are not made up: they name the registered runs they come from
            report = json.loads(path.read_text(encoding="utf-8"))
            assert any(i.get("run_id") for i in report.get("inputs", [])), name


def _rehearsal(name: str) -> dict:
    return effective_config(load_config(CONFIG_DIR / f"{name}.yaml"), smoke=True)


@pytest.mark.parametrize("name", NAMES)
def test_the_smoke_grid_builds_on_calibration_seeds_only(name):
    cfg = _rehearsal(name)
    assert cfg["mode"] == "smoke" and cfg["params"]["seed_role"] == "calibration"
    cells = grid(cfg)
    assert cells
    assert all(1000 <= c.get("seed", 1000) < 2000 and 10_000 <= c["solver_seed"] < 20_000 for c in cells)
    assert max(c["budget_s"] or 0 for c in cells) <= 1.0  # tiny budgets
    assert max(c.get("time_limit_s") or 0 for c in cells) <= 30.0


@pytest.mark.parametrize("name", NAMES)
def test_no_stage_uses_more_than_half_of_the_32_physical_cores_of_the_shared_server(name):
    """Plan §12 (gpu-07 is shared, no exclusive mode): not more than 16 simultaneous single-thread jobs. The
    certification stages are no timing stages, but the cap is the plan's and not only the timing stages'. The
    screening stages report timings (t_cert of A3) and are held to the cap as timing stages."""
    launch = load_config(CONFIG_DIR / f"{name}.yaml")["launch"]
    assert launch["jobs"] <= 16
    assert launch["shards"] >= launch["jobs"] or name in ("ref_calibration",)
    assert (name in ("ref_calibration", "ref_bench")) == (not launch["timing"])


def test_the_screening_stages_run_at_the_level_the_orchestrator_can_read_and_use_the_plans_limit():
    for name in ("elig_calibration", "elig_test", "elig_dense"):
        p = load_config(CONFIG_DIR / f"{name}.yaml")["params"]
        assert p["eligibility"] == {"tau": 1e-3, "certificate": "full", "margin": 1e-3}  # tau of the plan
        # the 2 h of §5; D1 (elig_dense): 600 s by §12 2026-10-12 (a descriptive stress cell)
        assert p["methods"]["time_limit_s"] == (600.0 if name == "elig_dense" else 7200.0) and p["budget_s"] is None
        assert p["methods"]["spec"]["formulation"] == "perspective" and p["methods"]["spec"]["diag"] == "mineig"
        for group in p["instances"]:
            scan = group.get("scan")
            if scan:
                # caps of A1; D1 scans only its planned seeds 2000-2019 (§12 2026-10-12)
                cap = 20 if name == "elig_dense" else (200 if p["seed_role"] == "calibration" else 400)
                assert scan["cap"] == cap
                assert scan["state"] in ELIGIBILITY and (scan["first"] if "first" in scan else scan["then_from"])


# --------------------------------------------------------------------------------------
# Stand-in reports of the pilot
# --------------------------------------------------------------------------------------


def _cell_state(selected, min_nontrivial):
    return {"done": True, "enough": len(selected) >= min_nontrivial, "prefix": 400, "found": len(selected),
            "min_nontrivial": min_nontrivial, "selected": selected, "needs_reference": list(selected)}


@pytest.fixture
def pilot_reports(tmp_path):
    """Stand-ins for the reports of the pilot, with the sizes of the real stages."""
    def spec(name, i, **kw):
        return {"name": name, "id": f"{name}-{i}", **kw}

    roles = {
        "exact_rep": spec("scip", "x", formulation="perspective", diag="mineig", usesymmetry="default"),
        "qubo_rep": spec("tabu", "q", num_restarts_b=2, tenure=5, num_reads_b=4, penalty=10.0, m_b=20),
        "ls": spec("ls", "l", num_restarts=3, improvement="first", rcl_size=3),
        "sa": spec("sa", "s", num_reads_b=20, num_sweeps=50, beta="default", penalty=10.0, m_b=20),
        "tabu": spec("tabu", "q", num_restarts_b=2, tenure=5, num_reads_b=4, penalty=10.0, m_b=20),
        "swap_sa": spec("swap_sa", "w", num_reads_b=10, num_sweeps=20, m_b=20),
        "h1c_sa": spec("sa", "h", num_reads_b=20, num_sweeps=50, beta="objective", penalty=10.0, m_b=20),
        "h1c_tabu": spec("tabu", "q", num_restarts_b=2, tenure=5, num_reads_b=4, penalty=10.0, m_b=20),
    }
    files = {
        "tuned": {"kind": "e2-tuned", "sizes": {str(n): {"roles": roles} for n in (100, 200, 400)}},
        "n_rule": {"kind": "e2-n-rule", "n_instances": N_RULE},
        "ttt_instances": {"kind": "e2-ttt-instances", "instances": [1014, 1016, 1021]},
        "c4_lambda": {"kind": "e2-c4-lambda", "lam": BOOT_LAM, "lam_used": BOOT_LAM, "admissible": True},
        "screen": {"kind": "e2-screen", "sizes": {str(n): {"recheck": [{"spec": roles["ls"], "why": []},
                                                                       {"spec": roles["exact_rep"], "why": []}]}
                                                  for n in (100, 200, 400)}},
        "eligibility_calibration": {"kind": "e2-eligibility", "role": "calibration", "cells": {
            c: _cell_state(s, 8) for c, s in CAL_SEEDS.items()}},
        "eligibility_test": {"kind": "e2-eligibility", "role": "test", "cells": {
            **{c: _cell_state(s, 20) for c, s in TEST_SEEDS.items()}, "C5": _cell_state(C5_POINTS, 10)}},
        "eligibility_dense": {"kind": "e2-eligibility", "role": "test", "cells": {
            c: _cell_state(s, 20) for c, s in D1_SEEDS.items()}},
    }
    paths = {}
    for label, payload in files.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload))
        paths[label] = str(path)
    return paths


def _full(name: str, reports: dict) -> dict:
    cfg = effective_config(load_config(CONFIG_DIR / f"{name}.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["inputs"] = {k: reports.get(k, v) for k, v in (cfg.get("inputs") or {}).items()}
    return cfg


def _instances(cells: list[dict]) -> set[tuple]:
    return {(c["family"], c["n"], c.get("seed"), c.get("port"), c.get("lam_index")) for c in cells}


# --------------------------------------------------------------------------------------
# The grids
# --------------------------------------------------------------------------------------


def test_the_tuning_sets_are_the_non_trivial_calibration_instances_of_the_scan(pilot_reports):
    cells = grid(_full("tune_screen", pilot_reports))
    by_n = {n: sorted({c["seed"] for c in cells if c["n"] == n}) for n in (100, 200, 400)}
    assert by_n == {100: CAL_SEEDS["T100"], 200: CAL_SEEDS["T200"], 400: CAL_SEEDS["T400"]}
    per_instance = lambda n: 4 * 30 * 3 + len(e2_space.exact_candidates(n)) * 3  # noqa: E731
    assert len(cells) == 8 * (per_instance(100) + per_instance(200) + per_instance(400)) == 9048
    assert {c["budget_s"] for c in cells} == {10.0} and {c["stage"] for c in cells} == {"screen"}
    ids = {c["method"]["id"] for c in cells if c["n"] == 400}
    assert not any("sdp" in i for i in ids)  # SDP only for N <= 150
    assert any("sdp" in c["method"]["id"] for c in cells if c["n"] == 100)
    assert {c["solver_seed"] for c in cells} == {10000, 10001, 10002}


def test_a_calibration_size_without_8_non_trivial_instances_is_not_tuned(pilot_reports):
    path = Path(pilot_reports["eligibility_calibration"])
    report = json.loads(path.read_text())
    report["cells"]["T100"] = _cell_state(CAL_SEEDS["T100"][:5], 8)
    path.write_text(json.dumps(report))
    for name in ("tune_screen", "tune_recheck"):
        cells = grid(_full(name, pilot_reports))
        assert {c["n"] for c in cells} == {200, 400}  # N = 100 borrows the configurations of N = 200


def test_the_reference_stage_runs_the_instances_the_screening_did_not_certify(pilot_reports):
    cells = grid(_full("ref_calibration", pilot_reports))
    assert len(cells) == 24 and {c["n"] for c in cells} == {100, 200, 400}
    assert all(c["time_limit_s"] == 7200.0 and c["stage"] == "reference" for c in cells)
    assert {c["seed"] for c in cells if c["n"] == 200} == set(CAL_SEEDS["T200"])
    path = Path(pilot_reports["eligibility_calibration"])
    report = json.loads(path.read_text())
    for cell in report["cells"].values():
        cell["needs_reference"] = []  # the screening ran at the level `full`: every selected instance is certified
    path.write_text(json.dumps(report))
    assert grid(_full("ref_calibration", pilot_reports)) == []


def test_the_confirmatory_stages_use_test_seeds_and_the_selected_instances(pilot_reports):
    cfg = _full("bench_factor", pilot_reports)
    assert cfg["params"]["seed_role"] == "test"
    cells = e2.cells(cfg)
    seeds = {n: sorted({c["seed"] for c in cells if c["n"] == n}) for n in (100, 200, 400)}
    assert seeds == {100: TEST_SEEDS["C1"], 200: TEST_SEEDS["C2"], 400: TEST_SEEDS["C3"]}  # not first_seed + count
    assert all(2000 <= c["seed"] < 3000 and 20_000 <= c["solver_seed"] < 30_000 for c in cells)
    assert {c["budget_s"] for c in cells} == {60.0} and {c["stage"] for c in cells} == {"bench"}
    assert all(c["skip_nonbinding"] for c in cells) and all(c["config_size"] == c["n"] for c in cells)
    # exact: 3 permutation seeds; each heuristic: 10 seeds
    one = [c for c in cells if c["seed"] == 2000 and c["n"] == 100]
    assert sorted(c["solver_seed"] for c in one if c["method"]["name"] == "scip") == [20000, 20001, 20002]
    for name in ("tabu", "sa", "swap_sa", "ls"):
        assert len([c for c in one if c["method"]["name"] == name]) == 10
    # the QUBO representative is one of the tuned heuristics: its cells are not repeated
    assert len({(c["method"]["id"], c["solver_seed"]) for c in one}) == len(one)


def test_every_test_cell_scans_from_seed_2000_and_still_no_instance_is_run_twice(pilot_reports):
    """Plan §12 A1: test seeds from 2000, at most 400 per cell, for every cell: cells cannot have disjoint seed blocks
    any more. Instances of different cells differ by family or size, and within a cell they are distinct."""
    seen: dict[str, set[tuple]] = {}
    for name in ("bench_factor", "bench_boot", "bench_dense"):
        for c in e2.cells(_full(name, pilot_reports)):
            seen.setdefault(c["cell"], set()).add((c["family"], c["n"], c["seed"]))
    cells = sorted(seen)
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            assert seen[a].isdisjoint(seen[b]), (a, b)
    assert {c: len(v) for c, v in seen.items()} == {"C1": 20, "C2": 21, "C3": 22, "C4": 23, "D1_100": 20, "D1_200": 20}


def test_h1c_runs_on_the_same_test_instances_as_the_factor_benchmark(pilot_reports):
    h1c = e2.cells(_full("h1c", pilot_reports))
    bench = e2.cells(_full("bench_factor", pilot_reports))
    assert _instances(h1c) == _instances(bench)
    assert {c["budget_s"] for c in h1c} == {10.0} and {c["stage"] for c in h1c} == {"h1c"}
    assert {c["method"]["name"] for c in h1c} == {"sa", "tabu"}
    assert {c["solver_seed"] for c in h1c} == set(range(20_000, 20_010))
    assert len(h1c) == len(_instances(bench)) * 2 * 10


def test_orlib_cells_are_the_selected_binding_non_trivial_points_with_the_configurations_of_the_plan(pilot_reports):
    cells = e2.cells(_full("bench_orlib", pilot_reports))
    assert {(c["port"], c["lam_index"]) for c in cells} == {tuple(p) for p in C5_POINTS}
    assert {c["config_size"] for c in cells if c["n"] <= 100} == {100}
    assert {c["config_size"] for c in cells if c["n"] > 100} == {200}
    assert {c["n"] for c in cells} == set(ORLIB_N.values())
    assert all("seed" not in c and c["skip_nonbinding"] for c in cells)


def test_the_boot_cell_runs_at_the_lambda_of_the_pilot(pilot_reports):
    cells = e2.cells(_full("bench_boot", pilot_reports))
    assert {c["boot_lam"] for c in cells} == {BOOT_LAM} and {c["config_size"] for c in cells} == {200}
    assert {c["seed"] for c in cells} == set(TEST_SEEDS["C4"])


def test_the_dense_cells_are_descriptive_and_run_the_selected_instances(pilot_reports):
    cells = e2.cells(_full("bench_dense", pilot_reports))
    assert not any(c["skip_nonbinding"] for c in cells)
    assert {n: sorted({c["seed"] for c in cells if c["n"] == n}) for n in (100, 200)} == {
        100: D1_SEEDS["D1_100"], 200: D1_SEEDS["D1_200"]}
    assert {c["method"]["name"] for c in cells} == {"scip", "tabu", "ls"}  # the three representatives


def test_the_reference_stage_covers_every_selected_test_instance_without_a_certificate(pilot_reports):
    ref = e2.cells(_full("ref_bench", pilot_reports))
    assert {c["stage"] for c in ref} == {"reference"} and {c["time_limit_s"] for c in ref} == {7200.0}
    bench = set()
    for name in ("bench_factor", "bench_boot", "bench_orlib"):
        bench |= _instances(e2.cells(_full(name, pilot_reports)))
    assert _instances(ref) == bench  # D1 is descriptive and not certified here
    assert len(ref) == len(bench)  # one run per instance
    assert {c["cell"] for c in ref} == {"C1", "C2", "C3", "C4", "C5"}


def test_the_cold_stage_runs_the_representatives_once_on_the_selected_instances_of_c1_to_c5(pilot_reports):
    cells = e2.cells(_full("cold", pilot_reports))
    bench = set()
    for name in ("bench_factor", "bench_boot", "bench_orlib"):
        bench |= _instances(e2.cells(_full(name, pilot_reports)))
    assert _instances(cells) == bench and {c["stage"] for c in cells} == {"cold"}
    assert len(cells) == len(bench) * 3  # one seed, three representatives
    assert {c["solver_seed"] for c in cells} == {20_000} and {c["budget_s"] for c in cells} == {60.0}
    assert {r for c in cells for r in c["roles"]} == {"exact_rep", "qubo_rep", "ls"}


def test_the_ttt_stage_runs_100_seeds_of_the_three_representatives_on_the_selected_instances(pilot_reports):
    cells = e2.cells(_full("ttt", pilot_reports))
    assert {c["seed"] for c in cells} == {1014, 1016, 1021} and {c["n"] for c in cells} == {200}
    assert len(cells) == 3 * 3 * 100  # instances × representatives × runs
    assert {c["solver_seed"] for c in cells} == set(range(10_100, 10_200))
    assert {c["budget_s"] for c in cells} == {60.0}


def test_the_recheck_and_the_boot_calibration_stages_use_fresh_calibration_seeds(pilot_reports):
    recheck = e2.cells(_full("tune_recheck", pilot_reports))
    assert {c["solver_seed"] for c in recheck} == {10003, 10004, 10005}  # not those of the screen
    assert {c["budget_s"] for c in recheck} == {60.0}
    boot = e2.cells(_full("calib_boot", pilot_reports))
    assert {c["seed"] for c in boot} == set(CAL_SEEDS["CB200"]) and {c["boot_lam"] for c in boot} == {BOOT_LAM}
    assert {c["solver_seed"] for c in boot} == {10003, 10004, 10005}


# --------------------------------------------------------------------------------------
# The screening: rounds of a scan
# --------------------------------------------------------------------------------------


def _scan_specs(name, **n_rule):
    cfg = effective_config(load_config(CONFIG_DIR / f"{name}.yaml"))
    return [el.spec_from_group(g, n_rule=n_rule or None) for g in cfg["params"]["instances"]]


@pytest.mark.skipif(not REGISTERED.exists(), reason="the registered reference run is absent")
def test_round_one_of_the_calibration_scan_starts_after_the_plan_seeds_the_registered_run_certified(tmp_path):
    """The state is built from the 24 registered full certificates (no method had run on them): the scan resumes at
    seed 1024 for N = 100, 200, 400 (nothing of the plan seeds is screened again) and at the head of boot."""
    rows = [json.loads(line) for line in REGISTERED.read_text().splitlines() if line.strip()]
    specs = _scan_specs("elig_calibration")
    state = el.build_report(rows, specs, role="calibration")
    assert {c: (s["prefix"], s["found"]) for c, s in state["cells"].items()} == {
        "T100": (8, 0), "T200": (8, 1), "T400": (8, 2), "CB200": (0, 0)}
    path = tmp_path / "eligibility_calibration.json"
    path.write_text(json.dumps(state))
    c4 = tmp_path / "c4_lambda.json"
    c4.write_text(json.dumps({"kind": "e2-c4-lambda", "lam": None, "lam_used": 20.0, "admissible": False}))
    cfg = effective_config(load_config(CONFIG_DIR / "elig_calibration.yaml"))
    cfg["inputs"] = {**cfg["inputs"], "eligibility_calibration": str(path), "c4_lambda": str(c4)}
    cells = e2.cells(cfg)
    by_cell = {}
    for c in cells:
        by_cell.setdefault(c["cell"], []).append(c["seed"])
    # blocks sized for the missing instances from the share seen so far: 8 missing at 0/8 → 100 seeds, 7 at 1/8 → 44,
    # 6 at 2/8 → 25, boot: the first block for 10 → 25 seeds (the 10 of the plan, then 1110-1124)
    assert {c: len(v) for c, v in by_cell.items()} == {"T100": 100, "T200": 44, "T400": 25, "CB200": 25}
    assert by_cell["T100"][0] == by_cell["T200"][0] == by_cell["T400"][0] == 1024
    assert by_cell["CB200"][:10] == list(range(1100, 1110)) and by_cell["CB200"][10] == 1110
    assert all(1000 <= s < 2000 for v in by_cell.values() for s in v)
    assert {c["boot_lam"] for c in cells if c["family"] == "boot"} == {20.0}  # the descriptive C4 runs at lam_used
    assert {c["stage"] for c in cells} == {"eligibility"} and {c["time_limit_s"] for c in cells} == {7200.0}
    assert {c["solver_seed"] for c in cells} == {10000}


def test_round_one_of_the_test_scan_has_blocks_sized_for_the_n_rule(pilot_reports, tmp_path):
    cfg = _full("elig_test", pilot_reports)
    stub = tmp_path / "eligibility_test.json"  # the stub of round 0 (the committed file now holds the later rounds)
    stub.write_text(json.dumps({"kind": "e2-eligibility", "stub": True, "role": "test", "cells": {}}), encoding="utf-8")
    cfg["inputs"]["eligibility_test"] = str(stub)
    cells = e2.cells(cfg)
    by_cell = {}
    for c in cells:
        if c["family"] != "orlib":
            by_cell.setdefault(c["cell"], []).append(c["seed"])
    # n = 20, 21, 22, 23 → 1.25·n/0.5 = 50, 53, 55, 58 seeds from 2000
    assert {c: len(v) for c, v in by_cell.items()} == {"C1": 50, "C2": 53, "C3": 55, "C4": 58}
    assert all(v == list(range(2000, 2000 + len(v))) for v in by_cell.values())
    assert all(2000 <= c["seed"] < 3000 and 20_000 <= c["solver_seed"] < 30_000 for c in cells if "seed" in c)
    # C5 is the fixed set of 125 points, screened in one round
    points = {(c["port"], c["lam_index"]) for c in cells if c["family"] == "orlib"}
    assert points == {(p, i) for p in range(1, 6) for i in range(25)}
    assert all(c["skip_nonbinding"] for c in cells)


def test_the_dense_scan_is_a_separate_report_of_the_planned_20_seeds_per_size(pilot_reports, tmp_path):
    cfg = _full("elig_dense", pilot_reports)
    stub = tmp_path / "eligibility_dense.json"  # the stub of round 0 (the committed file now holds the finished scan)
    stub.write_text(json.dumps({"kind": "e2-eligibility", "stub": True, "role": "test", "cells": {}}), encoding="utf-8")
    cfg["inputs"]["eligibility_dense"] = str(stub)
    cells = e2.cells(cfg)
    assert {c["family"] for c in cells} == {"dense"} and {c["n"] for c in cells} == {100, 200}
    assert not any(c["skip_nonbinding"] for c in cells)  # D1 keeps every instance: a non-binding one is trivial
    specs = _scan_specs("elig_dense")
    # §12 2026-10-12: only the planned seeds 2000-2019 per size, no scan beyond them
    assert [(s.cell, s.target, s.min_nontrivial, len(s.order)) for s in specs] == [
        ("D1_100", 20, 20, 20), ("D1_200", 20, 20, 20)]
    assert all(s.order == tuple(range(2000, 2020)) for s in specs)


def test_the_screening_configs_scan_the_orders_of_the_plan():
    specs = {s.cell: s for s in _scan_specs("elig_calibration")}
    assert specs["T100"].order[:9] == (*range(1000, 1008), 1024) and len(specs["T100"].order) == 200
    assert specs["T200"].order[:9] == (*range(1008, 1016), 1024) and specs["T400"].order[:9] == (*range(1016, 1024), 1024)
    assert specs["CB200"].order[:11] == (*range(1100, 1110), 1110) and specs["CB200"].target == 10
    assert {s.target for c, s in specs.items() if c.startswith("T")} == {8}
    test = {s.cell: s for s in _scan_specs("elig_test", C1=20, C2=20, C3=20, C4=20)}
    assert test["C1"].order == tuple(range(2000, 2400)) and test["C5"].points is True and len(test["C5"].order) == 125


#: after the pilot (§12, 2026-10-12) these stages build from the committed reports; ref_calibration builds an
#: empty grid (no calibration instance needs a reference any more); the confirmatory stages wait for the scan
#: of the test seeds, whose committed report is a stub or an unfinished scan until its last round
PILOT_STAGES = ("calib_boot", "elig_test", "ref_calibration", "ttt", "tune_recheck", "tune_screen")
CONFIRMATORY = ("bench_boot", "bench_dense", "bench_factor", "bench_orlib", "cold", "h1c", "ref_bench")
#: the post hoc sensitivity of H2 to the exact baseline (§12, 2026-10-19, review r1, M4): built like the bench stages,
#: from the same scan of the test seeds; not confirmatory
POST_HOC = ("bench_exact_oa", "bench_exact_scip_aggr")


def test_every_stage_after_the_screening_is_in_one_group():
    assert sorted(PILOT_STAGES + CONFIRMATORY + POST_HOC) == [n for n in NAMES
                                                              if n not in ("elig_calibration", "elig_dense")]


@pytest.mark.parametrize("name", PILOT_STAGES)
def test_the_stages_of_the_pilot_build_from_the_committed_reports_and_refuse_without_them(name, tmp_path):
    """The reports of the pilot are committed and the stage builds from them; with any one of its reports
    missing it still fails at once, it does not guess."""
    cfg = effective_config(load_config(CONFIG_DIR / f"{name}.yaml"))
    cells = e2.cells(cfg)
    if name == "ref_calibration":  # needs_reference is empty in every calibration cell: nothing left to run
        assert cells == []
    else:
        assert cells and {c["stage"] for c in cells} == {cfg["params"]["stage"]}
    labels = sorted(set(cfg["inputs"]) & REPORTS)
    assert labels
    for label in labels:
        missing = copy.deepcopy(cfg)
        missing["inputs"][label] = str(tmp_path / f"{label}.json")
        with pytest.raises((ValueError, FileNotFoundError)):
            e2.cells(missing)


@pytest.mark.parametrize("name", CONFIRMATORY + POST_HOC)
def test_the_confirmatory_stages_wait_for_the_scan_of_their_cells(name):
    """A confirmatory stage builds only when the scan of every cell it reads is finished (the committed reports of
    configs/E2/pilot); until then it fails at once, it does not guess."""
    cfg = effective_config(load_config(CONFIG_DIR / f"{name}.yaml"))
    needed = [(g["eligible"]["report"], g["eligible"].get("cell", g["cell"])) for g in cfg["params"]["instances"]
              if isinstance(g.get("eligible"), dict) and g["eligible"].get("report")]
    assert needed
    reports = {label: json.loads((ROOT / cfg["inputs"][label]).read_text(encoding="utf-8")) for label, _ in needed}
    finished = all(not reports[label].get("stub") and reports[label]["cells"].get(cell, {}).get("done")
                   for label, cell in needed)
    if finished:
        cells = e2.cells(cfg)
        # the reference stage is empty once every selected instance has its certificate from the screening
        assert cells or cfg["params"]["stage"] == "reference"
    else:
        with pytest.raises(ValueError, match=r"stub|not finished"):
            e2.cells(cfg)


def test_the_yaml_files_have_no_tabs_and_load_with_a_safe_loader():
    for path in CONFIG_DIR.glob("*.yaml"):
        text = path.read_text()
        assert "\t" not in text
        assert isinstance(yaml.safe_load(text), dict)
