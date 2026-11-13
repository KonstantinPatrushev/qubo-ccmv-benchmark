"""The screening stage and the cold stage of E2 (ANALYSIS_PLAN §12, A1 and A4 of 2026-10-11).

``eligibility`` decides, instance by instance and before any method runs on it, whether the common start is
already within τ of the optimum (trivial), or a solution better than it by more than τ·s exists (non-trivial);
its rows are the evidence and the input of the report (:mod:`cif.experiments.e2_eligibility`) that every later
stage reads. ``cold`` runs the methods without a common start, one seed per method and instance, for the honest
"until the method's own first solution" variant of the primal integral. Everything runs on 16-asset
calibration instances (seeds 1000-1999) with solver seeds 10000-19999; test seeds appear only in tests that
must refuse them. N = 16, K = floor(s_cont/2): seed 1000 is trivial (its start is optimal), seed 1006 non-trivial
(ĝ_start = 1e-2).
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import pytest

from cif.experiments import e2_benchmark as e2
from cif.experiments import e2_eligibility as el
from cif.experiments import e2_pilot as pilot
from cif.metrics.anytime import incumbent_curve
from tests.test_e2_benchmark import LS, OA, SA, SCIP, SWAP, TABU, _cfg, _fixed, _row, _tuned, _write

N = 16
TRIVIAL_SEED, NONTRIVIAL_SEED = 1000, 1006
ELIG = {"tau": 1e-3, "certificate": "tau", "margin": 1e-3}


@pytest.fixture(autouse=True)
def _fresh_caches():
    e2.clear_caches()
    yield
    e2.clear_caches()


def elig_cfg(seeds=(TRIVIAL_SEED, NONTRIVIAL_SEED), **params) -> dict:
    cfg = _cfg(
        stage="eligibility", budget_s=None, eligibility=dict(ELIG),
        instances=[{"cell": "T16", "family": "factor", "n": N, "seeds": list(seeds), "config_size": N}],
        methods={"source": "reference", "time_limit_s": 60.0},
        solver_seeds={"exact": [10000], "heuristic": [10000]},
    )
    cfg["params"].update(params)
    return cfg


def _cell_of(cfg: dict, seed: int) -> dict:
    return next(c for c in e2.cells(cfg) if c["seed"] == seed)


# --------------------------------------------------------------------------------------
# The grid of the screening stage
# --------------------------------------------------------------------------------------


def test_the_screening_stage_is_one_long_run_per_instance_with_the_reference_solver():
    cells = e2.cells(elig_cfg())
    assert [c["seed"] for c in cells] == [TRIVIAL_SEED, NONTRIVIAL_SEED]
    assert all(c["stage"] == "eligibility" and c["budget_s"] is None and c["time_limit_s"] == 60.0 for c in cells)
    method = cells[0]["method"]
    assert (method["name"], method["formulation"], method["diag"]) == ("scip", "perspective", "mineig")
    assert "eligibility" in e2.STAGES and "cold" in e2.STAGES


def test_the_screening_stage_checks_its_parameters():
    with pytest.raises(ValueError, match="certificate"):
        e2.cells(elig_cfg(eligibility={**ELIG, "certificate": "quick"}))
    with pytest.raises(ValueError, match="tau"):
        e2.cells(elig_cfg(eligibility={**ELIG, "tau": 0.0}))
    with pytest.raises(ValueError, match="margin"):
        e2.cells(elig_cfg(eligibility={**ELIG, "margin": 1.0}))
    with pytest.raises(ValueError, match="eligibility"):
        e2.cells(elig_cfg(eligibility=None))
    with pytest.raises(ValueError, match="time_limit_s"):
        e2.cells(elig_cfg(methods={"source": "reference"}))
    with pytest.raises(ValueError, match="reference"):
        e2.cells(elig_cfg(methods={"source": "explicit", "specs": [SA]}))
    with pytest.raises(ValueError, match="SCIP"):
        e2.cells(elig_cfg(methods={"source": "reference", "time_limit_s": 60.0,
                                   "spec": {"name": "oa", "id": "oa", "diag": "mineig"}}))


def _scan_cfg(tmp_path: Path, state: dict | None, **scan) -> dict:
    scan = {"head": [1000, 1001, 1002, 1003], "then_from": 1024, "cap": 12, "target": 2, "min_nontrivial": 2,
            "state": "eligibility_calibration", **scan}
    cfg = elig_cfg()
    cfg["params"]["instances"] = [{"cell": "T16", "family": "factor", "n": N, "config_size": N, "scan": scan}]
    cfg["inputs"] = {"eligibility_calibration": _write(
        tmp_path, "state.json", state or {"kind": "e2-eligibility", "stub": True, "cells": {}})}
    return cfg


def test_the_first_round_of_a_scan_screens_the_first_block_of_the_order(tmp_path):
    cfg = _scan_cfg(tmp_path, None, first_block=4)
    assert [c["seed"] for c in e2.cells(cfg)] == [1000, 1001, 1002, 1003]
    cfg = _scan_cfg(tmp_path, None, target=2)  # sized from the target: 1.25·2/0.5 = 5 seeds
    assert [c["seed"] for c in e2.cells(cfg)] == [1000, 1001, 1002, 1003, 1024]


def test_a_later_round_continues_where_the_report_of_the_rounds_so_far_stopped(tmp_path):
    from tests.test_e2_eligibility import erow, spec

    rows = [erow("T16", 1000 + i, "trivial", n=N) for i in range(4)]
    s = spec("T16", (1000, 1001, 1002, 1003, *range(1024, 1032)), target=2, min_nontrivial=2, n=N, config_size=N)
    state = el.build_report(rows, [s], role="calibration")
    cfg = _scan_cfg(tmp_path, state, cap=12)
    seeds = [c["seed"] for c in e2.cells(cfg)]
    assert seeds[0] == 1024 and seeds == sorted(seeds) and len(seeds) == 8  # the rest of the cap (12 − 4)
    # a finished cell gives no cells at all
    done = el.build_report([*rows[:2], erow("T16", 1002, "nontrivial", n=N), erow("T16", 1003, "nontrivial", n=N)],
                           [s], role="calibration")
    assert e2.cells(_scan_cfg(tmp_path, done, cap=12)) == []


def test_a_scan_takes_its_target_from_the_n_rule_report(tmp_path):
    cfg = _scan_cfg(tmp_path, None, target=None, target_from="n_rule", cap=400, head=[], first=1000)
    with pytest.raises(ValueError, match="n_rule"):
        e2.cells(cfg)
    cfg["inputs"]["n_rule"] = _write(tmp_path, "n_rule.json", {"kind": "e2-n-rule", "n_instances": {"T16": 3}})
    assert len(e2.cells(cfg)) == 8  # 1.25·3/0.5 = 7.5 → 8 seeds


def test_an_eligibility_report_of_another_kind_cannot_be_the_state_of_a_scan(tmp_path):
    cfg = _scan_cfg(tmp_path, {"kind": "e2-n-rule", "n_instances": {}})
    with pytest.raises(ValueError, match="e2-eligibility"):
        e2.cells(cfg)


# --------------------------------------------------------------------------------------
# What the later stages read: the selected instances
# --------------------------------------------------------------------------------------


def _report_file(tmp_path: Path, rows, specs, role="calibration", **kw) -> str:
    return _write(tmp_path, "eligibility.json", el.build_report(rows, specs, role=role, **kw))


def _finished_report(tmp_path: Path) -> str:
    from tests.test_e2_eligibility import erow, spec

    labels = ["trivial", "nontrivial", "trivial", "nontrivial"]
    rows = [erow("T16", 1000 + i, lab, n=N, certificate="full" if i == 1 else "tau") for i, lab in enumerate(labels)]
    return _report_file(tmp_path, rows, [spec("T16", range(1000, 1010), target=2, min_nontrivial=2, n=N,
                                              config_size=N)])


def test_a_stage_runs_the_instances_the_eligibility_report_selected(tmp_path):
    cfg = _cfg(instances=[{"cell": "T16", "family": "factor", "n": N, "config_size": N,
                           "eligible": {"report": "eligibility_calibration", "cell": "T16"}}],
               methods={"source": "explicit", "specs": [LS]}, solver_seeds={"exact": [10000], "heuristic": [10000]})
    cfg["inputs"] = {"eligibility_calibration": _finished_report(tmp_path)}
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1001, 1003]
    # the reference stage runs only those without a full certificate
    cfg["params"]["instances"][0]["eligible"]["field"] = "needs_reference"
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1003]


def test_a_stage_never_runs_on_an_unfinished_scan_a_stub_or_a_cell_the_report_lacks(tmp_path):
    from tests.test_e2_eligibility import erow, spec

    group = {"cell": "T16", "family": "factor", "n": N, "config_size": N,
             "eligible": {"report": "eligibility_calibration", "cell": "T16"}}
    cfg = _cfg(instances=[group], methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    unfinished = _report_file(tmp_path, [erow("T16", 1000, "trivial", n=N)],
                              [spec("T16", range(1000, 1010), target=2, min_nontrivial=2, n=N, config_size=N)])
    cfg["inputs"] = {"eligibility_calibration": unfinished}
    with pytest.raises(ValueError, match="not finished"):
        e2.cells(cfg)
    cfg["inputs"] = {"eligibility_calibration": _write(tmp_path, "stub.json", {"kind": "e2-eligibility",
                                                                                "stub": True, "cells": {}})}
    with pytest.raises(ValueError, match="stub"):
        e2.cells(cfg)
    cfg["inputs"] = {"eligibility_calibration": _finished_report(tmp_path)}
    cfg["params"]["instances"][0]["eligible"]["cell"] = "T99"
    with pytest.raises(ValueError, match="T99"):
        e2.cells(cfg)
    del cfg["inputs"]
    with pytest.raises(ValueError, match="eligibility_calibration"):
        e2.cells(cfg)


def test_a_calibration_size_without_enough_instances_is_not_tuned(tmp_path):
    from tests.test_e2_eligibility import erow, spec

    rows = [erow("T16", 1000, "nontrivial", n=N)] + [erow("T16", 1001 + i, "trivial", n=N) for i in range(9)]
    report = _report_file(tmp_path, rows, [spec("T16", range(1000, 1010), target=8, min_nontrivial=8, n=N,
                                                config_size=N)])
    group = {"cell": "T16", "family": "factor", "n": N, "config_size": N,
             "eligible": {"report": "eligibility_calibration", "cell": "T16", "require_enough": True}}
    cfg = _cfg(instances=[group], methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [10000], "heuristic": [10000]})
    cfg["inputs"] = {"eligibility_calibration": report}
    assert e2.cells(cfg) == []  # descriptive: its configurations come from the nearest N
    group["eligible"]["require_enough"] = False
    assert sorted({c["seed"] for c in e2.cells(cfg)}) == [1000]


def test_the_selected_points_of_the_orlib_cell_are_the_cells_of_a_stage(tmp_path):
    from tests.test_e2_eligibility import _point_rows

    group = {"cell": "C5", "family": "orlib", "ports": [1, 2, 3, 4, 5], "lam_indices": 25,
             "config_size": [{"max_n": 100, "size": 100}, {"size": 200}]}
    spec = el.spec_from_group(group)
    report = el.build_report(_point_rows({1: 4, 2: 3, 3: 3, 4: 1}), [spec], role="test")
    stage = {**group, "eligible": {"report": "eligibility_test", "cell": "C5"}}
    cfg = _cfg(seed_role="test", instances=[stage], methods={"source": "explicit", "specs": [LS]},
               solver_seeds={"exact": [20000], "heuristic": [20000]})
    cfg["inputs"] = {"eligibility_test": _write(tmp_path, "e.json", report)}
    cells = e2.cells(cfg)
    assert len(cells) == 11
    assert {(c["port"], c["lam_index"]) for c in cells} == {(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2),
                                                           (3, 0), (3, 1), (3, 2), (4, 0)}
    assert {c["config_size"] for c in cells if c["n"] <= 100} == {100}  # the rule of the plan for ports 1-4
    assert all(c["skip_nonbinding"] is False for c in cells)


# --------------------------------------------------------------------------------------
# One screening run
# --------------------------------------------------------------------------------------


def _screen(seed: int, **over) -> dict:
    cfg = elig_cfg(**over)
    return _row(_cell_of(cfg, seed), cfg)


def test_a_trivial_instance_is_one_where_the_dual_bound_proves_the_start_within_tau():
    row = _screen(TRIVIAL_SEED)
    assert row["eligibility"] == "trivial"
    ev = row["evidence"]
    assert (ev["certificate"], ev["reason"]) == ("tau", "dual_bound")
    assert ev["f_start"] == row["start_f"] and ev["s_start"] == row["start_scale"]
    assert ev["f_best"] <= ev["f_start"] and ev["gap_hi"] <= 1e-3 + 1e-12 and ev["gap_lo"] <= 1e-3
    assert row["status"] in ("optimal", "gaplimit", "duallimit")
    assert row["t_cert_s"] is not None and row["t_cert_s"] >= row["start_time_s"]
    assert ev["threshold"] == pytest.approx(1e-3 * row["start_scale"])
    assert el.verify_rows([row]) == []
    json.dumps(row, allow_nan=False)


def test_a_non_trivial_instance_is_one_where_a_solution_beats_the_start_by_more_than_tau():
    row = _screen(NONTRIVIAL_SEED)
    assert row["eligibility"] == "nontrivial" and row["evidence"]["reason"] == "improvement"
    ev = row["evidence"]
    assert ev["gap_lo"] > 1e-3 and ev["f_best"] < ev["f_start"] - ev["threshold"]
    assert row["status"] == "primallimit"  # SCIP stops as soon as the question is answered
    assert row["t_cert_s"] is None and ev["t_decide_s"] is not None and ev["t_decide_s"] >= row["start_time_s"]
    assert row["trajectory"] and row["trajectory"][-1][1] == pytest.approx(row["f_final"], abs=1e-8)
    assert el.verify_rows([row]) == []
    assert row["detail"]["scip_params"]["limits/primal"] < ev["f_start"] - ev["threshold"]


def test_scip_is_given_the_stopping_limits_of_the_rule_with_a_margin_on_both_sides():
    row = _screen(TRIVIAL_SEED)
    ev, params = row["evidence"], row["detail"]["scip_params"]
    thr = ev["threshold"]
    assert params["limits/dual"] == pytest.approx(ev["f_start"] - thr * (1 - 1e-3), rel=1e-12)
    assert params["limits/primal"] == pytest.approx(ev["f_start"] - thr * (1 + 1e-3), rel=1e-12)
    assert row["detail"]["abs_gap_limit"] is None  # the gap is not what stops this run


def test_the_start_of_the_screening_is_charged_to_its_clock_as_in_the_reference_stage():
    row = _screen(NONTRIVIAL_SEED)
    assert row["detail"]["clock_offset_s"] == pytest.approx(row["start_time_s"])
    assert all(t >= row["start_time_s"] for t, _ in row["trajectory"])


def test_a_full_screen_certifies_to_1e_6_and_yields_the_reference_optimum():
    """The other level of the plan's wording: "a full certificate of §5". Same class, an exact ĝ_start."""
    row = _screen(NONTRIVIAL_SEED, eligibility={**ELIG, "certificate": "full"})
    assert row["eligibility"] == "nontrivial" and row["evidence"]["certificate"] == "full"
    assert row["certified"] is True and row["ref_type"] == "certified"
    assert row["f_star"] <= row["f_final"] + 1e-12
    ev = row["evidence"]
    assert ev["gap_hi"] - ev["gap_lo"] < 1e-5  # the bounds meet: ĝ_start is known
    assert row["detail"]["abs_gap_limit"] > 0 and "limits/dual" not in row["detail"]["scip_params"]
    # the report takes it as a decision that needs no reference run
    from tests.test_e2_eligibility import spec

    report = el.build_report([row], [spec("T16", [NONTRIVIAL_SEED], target=1, min_nontrivial=1, n=N,
                                          config_size=N)], role="calibration")
    assert report["cells"]["T16"]["needs_reference"] == []


def test_both_levels_of_the_certificate_give_the_same_class():
    for seed in (TRIVIAL_SEED, NONTRIVIAL_SEED):
        tau = _screen(seed)
        full = _screen(seed, eligibility={**ELIG, "certificate": "full"})
        assert tau["eligibility"] == full["eligibility"], seed
        assert tau["start_f"] == full["start_f"]


def test_a_non_binding_instance_is_not_screened_when_the_stage_skips_them():
    cfg = elig_cfg(fixed=_fixed(fraction=1.0))  # K = s_cont
    cfg["params"]["instances"][0]["skip_nonbinding"] = True
    row = _row(e2.cells(cfg)[0], cfg)
    assert row["skipped"] == "not_binding" and "eligibility" not in row
    from tests.test_e2_eligibility import spec

    st = el.cell_state(spec("T16", [1000], target=1, min_nontrivial=1, n=N, config_size=N), [row])
    assert st["protocol"]["n_nonbinding"] == 1 and st["selected"] == []


def test_an_instance_whose_common_start_failed_is_undecided_and_no_solver_is_run_on_it(monkeypatch):
    """No start, nothing to compare with: a recorded undecided row, not a 2 h run of SCIP."""
    import numpy as np

    from cif.heuristics.start import CommonStart

    failed = CommonStart(weights=np.zeros(N), objective=math.inf, subset=(0, 1), status="infeasible", time_s=0.01,
                         cpu_s=0.01, relax_s=0.005, refine_s=0.005)
    called = []
    monkeypatch.setattr(e2, "common_start", lambda *a, **k: failed)
    monkeypatch.setattr(e2, "solve_miqp_native", lambda *a, **k: called.append(1))
    row = _screen(TRIVIAL_SEED)
    assert row["eligibility"] == "undecided" and row["evidence"]["reason"] == "no_start" and called == []
    assert row["status"] == "no_start" and row["start_f"] is None and row["start_time_s"] == 0.01
    assert el.verify_rows([row]) == []
    json.dumps(row, allow_nan=False)
    from tests.test_e2_eligibility import spec

    st = el.cell_state(spec("T16", [TRIVIAL_SEED], target=1, min_nontrivial=1, n=N, config_size=N), [row])
    assert st["protocol"]["n_undecided"] == 1 and st["selected"] == [] and st["done"] is True


def test_the_rows_of_the_screening_feed_the_report_and_the_audit():
    rows = [_screen(s) for s in (TRIVIAL_SEED, NONTRIVIAL_SEED)]
    assert el.verify_rows(rows) == []
    from tests.test_e2_eligibility import spec

    report = el.build_report(rows, [spec("T16", [TRIVIAL_SEED, NONTRIVIAL_SEED], target=1, min_nontrivial=1, n=N,
                                         config_size=N)], role="calibration")
    st = report["cells"]["T16"]
    assert st["selected"] == [NONTRIVIAL_SEED] and st["protocol"]["n_trivial"] == 1
    assert st["needs_reference"] == [NONTRIVIAL_SEED]  # screened at the tau level: a reference run follows
    json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------------------
# The cold stage
# --------------------------------------------------------------------------------------


def cold_cfg(specs, **params) -> dict:
    cfg = _cfg(stage="cold", budget_s=0.4, methods={"source": "explicit", "specs": specs},
               solver_seeds={"exact": [10000], "heuristic": [10000]},
               instances=[{"cell": "T16", "family": "factor", "n": N, "seeds": [NONTRIVIAL_SEED], "config_size": N}])
    cfg["params"].update(params)
    return cfg


def test_the_cold_stage_runs_one_seed_per_method_and_instance_of_the_tuned_roles(tmp_path):
    cfg = _cfg(stage="cold", budget_s=60.0, methods={"source": "tuned", "roles": ["exact_rep", "qubo_rep", "ls"]},
               solver_seeds={"exact": [10000], "heuristic": [10000]},
               instances=[{"cell": "T16", "family": "factor", "n": N, "seeds": [1001, 1006], "config_size": N}])
    cfg["inputs"] = {"tuned": _tuned(tmp_path, sizes=(N,))}
    cells = e2.cells(cfg)
    assert len(cells) == 2 * 3 and {c["stage"] for c in cells} == {"cold"}
    assert {c["budget_s"] for c in cells} == {60.0}


@pytest.mark.parametrize("spec", [SA, TABU, SWAP, LS, SCIP, OA], ids=lambda s: s["id"])
def test_a_cold_run_has_no_common_start_and_its_trajectory_begins_with_its_own_first_solution(spec):
    cfg = cold_cfg([spec], budget_s=5.0 if spec["name"] in ("scip", "oa") else 0.4)
    row = _row(e2.cells(cfg)[0], cfg)
    assert row["cold"] is True
    for key in ("start_time_s", "start_f", "start_scale", "start_subset", "start_cpu_s"):
        assert row[key] is None, key
    assert row["feasible"] and math.isfinite(row["f_final"])
    traj = row["trajectory"]
    assert traj and traj[-1][1] == pytest.approx(row["f_final"], abs=1e-8)
    assert all(b[1] < a[1] for a, b in pairwise(traj))
    # the delay until the first own solution and its value are in the row
    assert row["first_own_s"] is not None and row["first_own_f"] is not None
    assert row["first_own_s"] >= 0.0 and math.isfinite(row["first_own_f"])
    assert row["t_first_improve_s"] == traj[0][0] and row["t_primal_s"] == traj[-1][0]
    if spec["name"] in ("scip", "oa"):
        assert row["first_own_s"] == traj[0][0] and row["first_own_f"] == traj[0][1]
        assert row["detail"].get("warm_start_used") in (None, False)
    elif spec["name"] != "ls":
        assert row["first_own_f"] == pytest.approx(traj[0][1], abs=1e-12)  # the first batch is the incumbent
    json.dumps(row, allow_nan=False)


def test_the_clock_of_a_cold_run_starts_at_zero():
    cfg = cold_cfg([SCIP], budget_s=5.0)
    row = _row(e2.cells(cfg)[0], cfg)
    assert row["detail"]["clock_offset_s"] == 0.0 and row["end_s"] <= 5.0 * 1.05
    assert row["f_at_budget"] == pytest.approx(row["f_final"], abs=1e-8)


def test_the_honest_variant_of_the_primal_integral_is_computed_from_the_cold_row():
    cfg = cold_cfg([SA])
    row = _row(e2.cells(cfg)[0], cfg)
    ref = pilot.RefValue(row["instance_id"], row["f_final"] - 1e-4, row["scale_final"], "best_known", "x", 1, False)
    own = pilot.row_pi(row, ref, variant="own")
    t0 = row["trajectory"][0][0]
    times, _ = incumbent_curve(row["trajectory"])
    assert times[0] == t0 and 0.0 < own <= 1.0
    assert own >= t0 / row["budget_s"] - 1e-12  # γ = 1 until the first own solution
    with pytest.raises(ValueError, match="cold"):
        pilot.row_pi(row, ref)  # PI⁺ is defined on warm rows


def test_warm_rows_record_the_value_of_the_first_own_solution_next_to_its_time():
    cfg = _cfg(methods={"source": "explicit", "specs": [SA, SCIP]})
    for spec in (SA, SCIP):
        row = _row(next(c for c in e2.cells(cfg) if c["method"]["id"] == spec["id"] and c["seed"] == 1001), cfg)
        assert "first_own_f" in row and row.get("cold") is None
        if row["first_own_s"] is not None:
            assert row["first_own_f"] is not None and math.isfinite(row["first_own_f"])
        if spec["name"] == "scip" and row["trajectory"]:
            assert row["first_own_f"] == row["trajectory"][0][1]
