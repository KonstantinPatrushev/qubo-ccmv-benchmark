"""TTT curves of the stage ``ttt`` and the delay to the first own solution on the stage ``cold``: descriptive secondary
analyses of RQ2 (ANALYSIS_PLAN §5, §6; §12 of 2026-10-17, recorded before the rows were read).

The rows are synthetic. A warm run has its common start at ``t0 = 0.5`` s with the gap ``start_gap`` above the
certified optimum ``F_STAR`` and reaches the optimum at the time given (or never); a cold run has no start and its
trajectory begins with its first own solution. Every expected number is known by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from cif.experiments import e2_analysis as an
from cif.experiments import e2_curves as cv
from cif.experiments import registry

F_STAR = -1.0
SCALE = 1.0
T = 60.0
METHOD = {"exact_rep": "scip", "qubo_rep": "swap_sa", "ls": "ls"}
SIZE = {"C1": 100, "C2": 200, "C3": 400, "C4": 200}


# --------------------------------------------------------------------------------------------
# Synthetic rows
# --------------------------------------------------------------------------------------------


def ttt_row(seed, role, solver_seed, hit, *, t0=0.5, start_gap=0.01, traj=None, budget=T, stage="ttt", **extra):
    """A run on ``factor/200/<seed>`` whose incumbent reaches the optimum at ``hit`` (``None``: never)."""
    if traj is None:
        traj = [] if hit is None else [[hit, F_STAR]]
    start_f = F_STAR + start_gap * SCALE
    row = {"cell_key": f"{stage}|TTT|{seed}|{role}|{solver_seed}", "stage": stage, "cell": "TTT", "seed": seed,
           "instance_id": f"factor/200/{seed}", "n": 200, "config_size": 200, "roles": [role],
           "method": {"name": METHOD[role], "id": f"{role}-200"}, "config_id": f"{role}-200",
           "solver_seed": solver_seed, "budget_s": budget, "start_time_s": t0, "start_f": start_f,
           "start_scale": SCALE, "trajectory": [list(p) for p in traj],
           "f_final": traj[-1][1] if traj else start_f, "scale_final": SCALE}
    row.update(extra)
    return row


def instance_rows(seed, times: dict[str, list], **kw) -> list[dict]:
    return [ttt_row(seed, role, 10100 + i, hit, **kw) for role, hits in times.items() for i, hit in enumerate(hits)]


def cert(seed, stage="reference") -> dict:
    """The certificate of the calibration screening for ``factor/200/<seed>``."""
    return {"cell_key": f"{stage}|T200|{seed}", "stage": stage, "instance_id": f"factor/200/{seed}",
            "f_star": F_STAR, "scale_star": SCALE, "ref_type": "certified", "config_id": "ref"}


TIMES = {"exact_rep": [2.0, 4.0, None, 1.0], "qubo_rep": [3.0] * 4, "ls": [1.5] * 4}


def method(out, tau, iid, role) -> dict:
    (hit,) = [m for m in out["methods"] if m["tau"] == tau and m["instance_id"] == iid and m["role"] == role]
    return hit


def pair(out, tau, iid, a, b) -> dict:
    (hit,) = [p for p in out["pairs"] if p["tau"] == tau and p["instance_id"] == iid and (p["a"], p["b"]) == (a, b)]
    return hit


# --------------------------------------------------------------------------------------------
# TTT curves
# --------------------------------------------------------------------------------------------


def test_the_curve_is_the_empirical_distribution_over_all_runs_with_the_misses_in_the_denominator():
    """§12 (3): the points are (t_(i), (i - 1/2)/n) over the runs that got to the target, n counts every run."""
    out = cv.ttt_curves_report(instance_rows(1014, TIMES), references=[cert(1014)])
    points = [(p["time_s"], p["p"]) for p in out["curves"]
              if p["tau"] == 1e-3 and p["instance_id"] == "factor/200/1014" and p["role"] == "exact_rep"]
    assert points == [(1.0, 0.125), (2.0, 0.375), (4.0, 0.625)]  # the curve ends below 1: one run missed
    m = method(out, 1e-3, "factor/200/1014", "exact_rep")
    assert (m["n_runs"], m["n_reached"], m["share_reached"]) == (4, 3, 0.75)
    assert out["analysis"] == "secondary" and out["confirmatory"] is False


def test_the_time_is_end_to_end_and_the_quartiles_are_those_of_kaplan_meier():
    """The time counts from the clock of the run (t0 of the start included); the quartiles are the smallest times
    with S(t) <= 3/4, 1/2, 1/4, and None beyond the budget."""
    times = {**TIMES, "ls": [5.0, None, None, None]}
    out = cv.ttt_curves_report(instance_rows(1014, times), references=[cert(1014)])
    exact = method(out, 1e-3, "factor/200/1014", "exact_rep")
    assert (exact["q1_s"], exact["median_s"], exact["q3_s"]) == (1.0, 2.0, 4.0)
    qubo = method(out, 1e-3, "factor/200/1014", "qubo_rep")
    assert (qubo["q1_s"], qubo["median_s"], qubo["q3_s"]) == (3.0, 3.0, 3.0)
    ls = method(out, 1e-3, "factor/200/1014", "ls")
    assert (ls["q1_s"], ls["median_s"], ls["q3_s"]) == (5.0, None, None) and ls["share_reached"] == 0.25
    assert out["instances"]["factor/200/1014"]["t0_s"] == 0.5


def test_p_faster_counts_a_tie_as_half_a_censored_run_as_the_loser_and_two_censored_runs_as_a_tie():
    assert cv.p_faster([1.0, 60.0], [True, False], [1.0, 2.0], [True, True]) == pytest.approx(0.375)
    assert cv.p_faster([60.0], [False], [60.0], [False]) == pytest.approx(0.5)
    assert cv.p_faster([3.0], [True], [60.0], [False]) == pytest.approx(1.0)


def test_every_pair_of_methods_gets_the_probability_that_the_first_is_faster():
    out = cv.ttt_curves_report(instance_rows(1014, TIMES), references=[cert(1014)])
    iid = "factor/200/1014"
    assert pair(out, 1e-3, iid, "qubo_rep", "exact_rep")["p_a_faster"] == pytest.approx(0.5)
    assert pair(out, 1e-3, iid, "qubo_rep", "ls")["p_a_faster"] == pytest.approx(0.0)
    assert pair(out, 1e-3, iid, "ls", "exact_rep")["p_a_faster"] == pytest.approx(0.75)
    assert pair(out, 1e-3, iid, "ls", "exact_rep")["n_pairs"] == 16


def test_the_sensitivity_target_is_tighter():
    """A run at ĝ = 5e-4 from 2 s and at the optimum from 5 s: 2 s for τ = 1e-3, 5 s for τ = 1e-4."""
    traj = [[2.0, F_STAR + 5e-4], [5.0, F_STAR]]
    rows = [ttt_row(1014, role, 10100 + i, None, traj=traj) for role in METHOD for i in range(2)]
    out = cv.ttt_curves_report(rows, references=[cert(1014)])
    assert method(out, 1e-3, "factor/200/1014", "ls")["median_s"] == 2.0
    assert method(out, 1e-4, "factor/200/1014", "ls")["median_s"] == 5.0
    assert out["settings"]["taus"] == [1e-3, 1e-4]


def test_an_instance_whose_start_meets_the_target_is_degenerate():
    """§12 (5): a start at ĝ = 5e-4 meets τ = 1e-3 (every time is t0) but not τ = 1e-4."""
    rows = instance_rows(1014, TIMES, start_gap=5e-4)
    out = cv.ttt_curves_report(rows, references=[cert(1014)])
    assert out["instances"]["factor/200/1014"]["degenerate"] == {"0.001": True, "0.0001": False}
    m = method(out, 1e-3, "factor/200/1014", "exact_rep")
    assert m["degenerate"] is True and (m["median_s"], m["share_reached"]) == (0.5, 1.0)
    assert method(out, 1e-4, "factor/200/1014", "exact_rep")["degenerate"] is False
    assert any("degenerate" in w for w in out["warnings"])


def test_the_instances_are_reported_one_by_one_without_pooling():
    rows = instance_rows(1014, TIMES) + instance_rows(1043, TIMES)
    out = cv.ttt_curves_report(rows, references=[cert(1014), cert(1043)])
    assert sorted(out["instances"]) == ["factor/200/1014", "factor/200/1043"]
    assert len(out["methods"]) == 2 * 3 * 2  # instances x methods x targets
    assert "pooled" not in out


def test_the_reference_is_the_certificate_and_its_absence_is_a_warning():
    out = cv.ttt_curves_report(instance_rows(1014, TIMES), references=[cert(1014, stage="eligibility")])
    assert out["instances"]["factor/200/1014"]["reference"]["ref_type"] == "certified"
    assert out["warnings"] == []
    bare = cv.ttt_curves_report(instance_rows(1014, TIMES))
    assert bare["instances"]["factor/200/1014"]["reference"]["ref_type"] == "best_known"
    assert any("certified" in w for w in bare["warnings"])


@pytest.mark.parametrize("change, message", [
    ({"stage": "bench"}, "stage"),
    ({"error": "RuntimeError: boom"}, "error"),
    ({"budget_s": 30.0}, "budget"),
])
def test_rows_must_be_ttt_rows_without_errors_of_one_budget(change, message):
    rows = instance_rows(1014, TIMES)
    rows[0] = {**rows[0], **change}
    with pytest.raises(ValueError, match=message):
        cv.ttt_curves_report(rows, references=[cert(1014)])


def test_every_instance_needs_every_method_with_the_same_number_of_runs():
    rows = [r for r in instance_rows(1014, TIMES) if r["roles"] != ["ls"]]
    with pytest.raises(ValueError, match="ls"):
        cv.ttt_curves_report(rows, references=[cert(1014)])
    rows = instance_rows(1014, TIMES) + instance_rows(1043, {**TIMES, "qubo_rep": [3.0] * 3})
    with pytest.raises(ValueError, match="runs"):
        cv.ttt_curves_report(rows, references=[cert(1014), cert(1043)])


# --------------------------------------------------------------------------------------------
# Delay to the first own solution
# --------------------------------------------------------------------------------------------


def cold_row(cell, unit, role, traj, *, budget=T, first_own=None, solver_seed=20000, stage="cold") -> dict:
    iid = f"orlib/{unit[0]}/{unit[1]:02d}" if isinstance(unit, tuple) else f"factor/{SIZE[cell]}/{unit}"
    return {"cell_key": f"{stage}|{cell}|{unit}|{role}|{solver_seed}", "stage": stage, "cold": True, "cell": cell,
            "instance_id": iid, "roles": [role], "config_id": f"{role}-x",
            "method": {"name": METHOD[role], "id": f"{role}-x"}, "solver_seed": solver_seed, "budget_s": budget,
            "start_time_s": None, "start_f": None, "trajectory": [list(p) for p in traj],
            "first_own_s": first_own, "first_own_f": None}


def cold_instance(cell, unit, delays: dict[str, float | None], **kw) -> list[dict]:
    return [cold_row(cell, unit, role, [] if d is None else [[d, -0.5], [d + 1.0, -0.9]], **kw)
            for role, d in delays.items()]


def delay_of(out, iid, role) -> dict:
    (hit,) = [r for r in out["instances"] if r["instance_id"] == iid and r["role"] == role]
    return hit


def cell_of(out, cell, role) -> dict:
    (hit,) = [r for r in out["cells"] if r["cell"] == cell and r["role"] == role]
    return hit


def test_the_delay_is_the_first_point_of_the_own_trajectory_not_the_field_first_own_s():
    """§12 (2): local search holds its first portfolio at the end of its construction (0.6 s); the old field said
    0.001 s (audit G2, C-4) and is not read."""
    rows = [cold_row("C1", 2001, "ls", [[0.6, -0.5]], first_own=0.001),
            cold_row("C1", 2001, "exact_rep", [[0.2, -0.4], [3.0, -0.9]], first_own=0.2),
            cold_row("C1", 2001, "qubo_rep", [[0.3, -0.6]], first_own=0.3)]
    out = cv.delay_report(rows)
    assert delay_of(out, "factor/100/2001", "ls")["delay_s"] == 0.6
    assert delay_of(out, "factor/100/2001", "exact_rep")["delay_s"] == 0.2
    assert out["analysis"] == "secondary" and out["confirmatory"] is False


def test_a_run_without_an_own_solution_in_the_budget_is_censored_at_it_and_counted():
    rows = cold_instance("C1", 2001, {"exact_rep": None, "qubo_rep": 0.3, "ls": 0.6})
    rows += cold_instance("C1", 2002, {"exact_rep": 1.0, "qubo_rep": 0.4, "ls": 0.7})
    out = cv.delay_report(rows)
    miss = delay_of(out, "factor/100/2001", "exact_rep")
    assert (miss["delay_s"], miss["censored"]) == (T, True)
    cell = cell_of(out, "C1", "exact_rep")
    assert (cell["n_instances"], cell["n_censored"], cell["share_censored"]) == (2, 1, 0.5)
    assert cell["max_s"] == T


def test_the_cell_gets_the_median_quartiles_and_maximum_over_its_instances():
    delays = [0.5, 1.0, 2.0, 4.0]
    rows = [r for i, d in enumerate(delays) for r in cold_instance("C2", 2000 + i, {"exact_rep": d, "qubo_rep": d,
                                                                                    "ls": d})]
    out = cv.delay_report(rows)
    cell = cell_of(out, "C2", "ls")
    assert (cell["median_s"], cell["q1_s"], cell["q3_s"], cell["max_s"]) == pytest.approx((1.5, 0.875, 2.5, 4.0))
    assert cell["n_censored"] == 0


def test_the_value_of_an_instance_is_the_median_over_its_seeds():
    """One seed per method in the cold run; with more, the Kaplan-Meier median (the lower median)."""
    rows = [cold_row("C1", 2001, "ls", [[d, -0.5]], solver_seed=20000 + i) for i, d in enumerate((3.0, 1.0, 2.0))]
    rows += [cold_row("C1", 2001, role, [[0.1, -0.5]]) for role in ("exact_rep", "qubo_rep")]
    out = cv.delay_report(rows)
    hit = delay_of(out, "factor/100/2001", "ls")
    assert (hit["delay_s"], hit["n_seeds"]) == (2.0, 3)


def test_c5_points_are_instances_of_the_cell():
    rows = cold_instance("C5", (2, 3), {"exact_rep": 1.0, "qubo_rep": 0.2, "ls": 0.4})
    out = cv.delay_report(rows)
    assert delay_of(out, "orlib/2/03", "qubo_rep")["delay_s"] == 0.2
    assert cell_of(out, "C5", "qubo_rep")["n_instances"] == 1


def test_delay_rows_must_be_cold_rows_with_every_method():
    rows = cold_instance("C1", 2001, {"exact_rep": 1.0, "qubo_rep": 0.2, "ls": 0.4})
    with pytest.raises(ValueError, match="stage"):
        cv.delay_report([{**rows[0], "stage": "bench"}, *rows[1:]])
    with pytest.raises(ValueError, match="qubo_rep"):
        cv.delay_report([r for r in rows if r["roles"] != ["qubo_rep"]])


# --------------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------------


def write_run(root: Path, name: str, rows: list[dict], *, stage: str, seed_role: str, dirty=False) -> Path:
    run = root / name
    run.mkdir()
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"run_id": name, "git_sha": "a" * 40, "git_dirty": dirty, "config_hash": "c0ffee", "mode": "ingested",
                "n_errors": 0, "n_missing": 0}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"stream": "E2", "name": stage, "params": {"stage": stage, "seed_role": seed_role}}
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return run


@pytest.fixture
def clean_tree(monkeypatch):
    state = {"available": True, "sha": "b" * 40, "branch": "stream/E0-engine", "dirty": False, "status": [],
             "untracked_code": []}
    monkeypatch.setattr(registry, "_git_state", lambda: state)


def test_the_ttt_curves_command_reads_registered_runs_and_writes_the_tables(tmp_path, clean_tree, capsys):
    run = write_run(tmp_path, "20261013-121926-E2-ttt-abc123", instance_rows(1014, TIMES), stage="ttt",
                    seed_role="calibration")
    ref = write_run(tmp_path, "20261011-211545-E2-ref_calibration-def456", [cert(1014)], stage="reference",
                    seed_role="calibration")
    out = tmp_path / "ttt_curves"
    assert an.main(["ttt-curves", "--rows", str(run), "--reference", str(ref), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["ttt_curve_methods.csv", "ttt_curve_pairs.csv",
                                                     "ttt_curve_points.csv", "ttt_curves.json"]
    result = json.loads((out / "ttt_curves.json").read_text(encoding="utf-8"))
    assert result["provenance_clean"] is True and result["warnings"] == []
    assert {i["run_id"] for i in result["inputs"] if i.get("run_id")} == {run.name, ref.name}
    assert result["code"]["git_dirty"] is False
    points = pd.read_csv(out / "ttt_curve_points.csv")
    assert set(points["analysis"]) == {"secondary"} and len(points) > 0
    assert "SECONDARY" in capsys.readouterr().out


def test_the_ttt_curves_command_flags_rows_that_are_not_a_registered_ttt_run(tmp_path, clean_tree):
    run = write_run(tmp_path, "ttt", instance_rows(1014, TIMES), stage="ttt", seed_role="calibration", dirty=True)
    ref = write_run(tmp_path, "ref", [cert(1014)], stage="reference", seed_role="calibration")
    out = tmp_path / "ttt_curves"
    an.main(["ttt-curves", "--rows", str(run), "--reference", str(ref), "--out", str(out)])
    result = json.loads((out / "ttt_curves.json").read_text(encoding="utf-8"))
    assert result["provenance_clean"] is False and any("git_dirty" in w for w in result["warnings"])


def test_the_delay_command_reads_the_cold_run_and_writes_the_tables(tmp_path, clean_tree):
    rows = cold_instance("C1", 2001, {"exact_rep": 1.0, "qubo_rep": 0.2, "ls": 0.4})
    run = write_run(tmp_path, "20261013-121922-E2-cold-abc123", rows, stage="cold", seed_role="test")
    out = tmp_path / "delay"
    assert an.main(["delay", "--rows", str(run), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["delay.json", "delay_cells.csv", "delay_instances.csv"]
    result = json.loads((out / "delay.json").read_text(encoding="utf-8"))
    assert result["provenance_clean"] is True and result["inputs"][0]["run_id"] == run.name
    cells = pd.read_csv(out / "delay_cells.csv")
    assert set(cells["analysis"]) == {"secondary"} and set(cells["role"]) == {"exact_rep", "qubo_rep", "ls"}


# --------------------------------------------------------------------------------------------
# Post hoc: the quality of the first own portfolio against the common start (§12, 2026-10-19 (3))
# --------------------------------------------------------------------------------------------


def h2_of(instances, *, confirmatory=True) -> dict:
    """A minimal report of ``e2_analysis h2``: per instance its reference, the gap and the time of the common start."""
    return {"kind": "e2-h2", "confirmatory": confirmatory,
            "instances": [{"cell": c, "instance_id": iid, "f_ref": F_STAR, "s_ref": SCALE, "g_start": g0, "t0_s": t0}
                          for c, iid, g0, t0 in instances]}


def test_the_first_own_portfolio_is_set_against_the_common_start_of_the_same_instance():
    # the common start: ĝ = 3τ at 0.05 s; QUBO first at 0.2 s with 4τ, LS at 0.6 s with 2τ, the exact method at
    # 0.01 s with 10τ
    rows = [cold_row("C1", 2001, "qubo_rep", [[0.2, F_STAR + 0.004], [1.0, F_STAR + 0.001]]),
            cold_row("C1", 2001, "ls", [[0.6, F_STAR + 0.002]]),
            cold_row("C1", 2001, "exact_rep", [[0.01, F_STAR + 0.010]])]
    out = cv.first_quality_report(rows, h2_of([("C1", "factor/100/2001", 0.003, 0.05)]))
    qubo = delay_of(out, "factor/100/2001", "qubo_rep")
    assert qubo["g_first"] == pytest.approx(0.004) and qubo["g_first_over_tau"] == pytest.approx(4.0)
    assert qubo["dt_s"] == pytest.approx(0.15)
    assert (qubo["not_worse_than_start"], qubo["not_later_than_start"]) == (False, False)
    ls = delay_of(out, "factor/100/2001", "ls")
    assert (ls["not_worse_than_start"], ls["not_later_than_start"]) == (True, False)
    exact = delay_of(out, "factor/100/2001", "exact_rep")
    assert (exact["not_worse_than_start"], exact["not_later_than_start"]) == (False, True)
    assert out["analysis"] == "post_hoc" and out["confirmatory"] is False and "2026-10-19" in out["notice"]


def test_the_cell_gets_shares_and_medians_and_a_run_without_a_portfolio_is_worse_and_later():
    h2 = h2_of([("C2", "factor/200/2000", 0.002, 0.05), ("C2", "factor/200/2001", 0.002, 0.05)])
    rows = [cold_row("C2", 2000, "qubo_rep", [[0.1, F_STAR + 0.001]]), cold_row("C2", 2000, "ls", [[0.3, F_STAR + 0.001]]),
            cold_row("C2", 2000, "exact_rep", []),
            cold_row("C2", 2001, "qubo_rep", [[0.02, F_STAR + 0.005]]),
            cold_row("C2", 2001, "ls", [[0.3, F_STAR + 0.001]]),
            cold_row("C2", 2001, "exact_rep", [[5.0, F_STAR + 0.001]])]
    out = cv.first_quality_report(rows, h2)
    qubo = cell_of(out, "C2", "qubo_rep")
    assert (qubo["n_instances"], qubo["n_censored"]) == (2, 0)
    assert (qubo["share_not_worse"], qubo["share_not_later"]) == (0.5, 0.5)
    assert qubo["median_g_first_over_tau"] == pytest.approx(3.0)
    exact = cell_of(out, "C2", "exact_rep")
    assert exact["n_censored"] == 1 and exact["share_not_worse"] == 0.5 and exact["share_not_later"] == 0.0
    assert exact["median_g_first_over_tau"] is None  # the median over the instances is beyond the budget
    missing = delay_of(out, "factor/200/2000", "exact_rep")
    assert missing["censored"] is True and missing["g_first"] is None and missing["not_worse_than_start"] is False


def test_an_instance_outside_the_h2_report_and_rows_of_another_stage_are_errors():
    rows = cold_instance("C1", 2001, {"exact_rep": 1.0, "qubo_rep": 0.2, "ls": 0.4})
    with pytest.raises(ValueError, match="factor/100/2001"):
        cv.first_quality_report(rows, h2_of([("C1", "factor/100/2002", 0.003, 0.05)]))
    with pytest.raises(ValueError, match="stage"):
        cv.first_quality_report([{**rows[0], "stage": "bench"}, *rows[1:]],
                                h2_of([("C1", "factor/100/2001", 0.003, 0.05)]))


def test_the_first_quality_command_records_the_h2_report_among_its_inputs(tmp_path, clean_tree):
    rows = [cold_row("C1", 2001, "qubo_rep", [[0.2, F_STAR + 0.004]]), cold_row("C1", 2001, "ls", [[0.6, F_STAR + 0.002]]),
            cold_row("C1", 2001, "exact_rep", [[0.01, F_STAR + 0.010]])]
    run = write_run(tmp_path, "20261013-121922-E2-cold-abc123", rows, stage="cold", seed_role="test")
    h2_path = tmp_path / "h2.json"
    h2_path.write_text(json.dumps(h2_of([("C1", "factor/100/2001", 0.003, 0.05)])), encoding="utf-8")
    out = tmp_path / "first_quality"
    assert an.main(["first-quality", "--rows", str(run), "--h2", str(h2_path), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["first_quality.json", "first_quality_cells.csv",
                                                     "first_quality_instances.csv"]
    result = json.loads((out / "first_quality.json").read_text(encoding="utf-8"))
    assert result["provenance_clean"] is True and result["warnings"] == []
    assert {i.get("run_id") for i in result["inputs"]} >= {run.name}
    assert any(i["source"] == str(h2_path) and i["sha256"] and i["role"] == "h2" for i in result["inputs"])
    assert set(pd.read_csv(out / "first_quality_cells.csv")["analysis"]) == {"post_hoc"}
    h2_path.write_text(json.dumps(h2_of([("C1", "factor/100/2001", 0.003, 0.05)], confirmatory=False)),
                       encoding="utf-8")
    an.main(["first-quality", "--rows", str(run), "--h2", str(h2_path), "--out", str(tmp_path / "again")])
    again = json.loads((tmp_path / "again" / "first_quality.json").read_text(encoding="utf-8"))
    assert again["provenance_clean"] is False and any("confirmatory" in w for w in again["warnings"])
