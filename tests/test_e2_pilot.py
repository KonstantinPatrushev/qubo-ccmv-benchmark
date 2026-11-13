"""Pilot rules of E2 (ANALYSIS_PLAN §3, §5, §6): references, primal integral and time-to-target of rows,
the tuning protocol (criterion at 10 s, runner-up at 60 s, 5 % and TTT tie rules), the C4 λ rule, the
n-per-cell rule and the TTT-instance selection. All on synthetic rows with hand-computed numbers, plus
one pass of the whole pilot on tiny real runs. Calibration seeds only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cif.experiments import e2_pilot as pilot
from cif.experiments import run
from cif.instances.factor import factor_instance
from cif.metrics.anytime import TAU_MAIN

BIG = 1e9


def _f_at(start, traj, t):
    best = start[1] if start[0] <= t else None
    for time_s, f in traj:
        if time_s <= t:
            best = f
    return best


def make_row(
    instance="i1", config="sa-00", name="sa", *, size=100, seed=10000, budget=10.0, start=(0.1, -0.90),
    traj=(), scale=2.0, stage="screen", cell="T100", spec=None, **extra,
) -> dict:
    """A row as ``e2_benchmark.run_cell`` writes it, reduced to what the pilot reads."""
    traj = [list(p) for p in traj]
    f_final = traj[-1][1] if traj else start[1]
    row = {
        "cell_key": f"{instance}|{size}|{config}|{seed}|{budget}|{stage}",
        "stage": stage, "cell": cell, "instance_id": instance, "config_size": size,
        "method": spec or {"name": name, "id": config}, "config_id": config, "solver_seed": seed,
        "budget_s": budget, "start_time_s": start[0], "start_f": start[1], "start_scale": scale,
        "trajectory": traj, "f_final": f_final, "scale_final": scale, "status": "budget",
        "f_at_budget": _f_at(start, traj, budget),
    }
    row.update(extra)
    return row


def ref_row(instance="i1", f_star=-1.0, scale=2.0, certified=True, size=100) -> dict:
    return {"cell_key": f"ref|{instance}", "stage": "reference", "instance_id": instance,
            "config_size": size, "method": {"name": "scip", "id": "ref"}, "config_id": "ref",
            "solver_seed": 10000, "budget_s": None, "f_star": f_star, "scale_star": scale,
            "ref_type": "certified" if certified else "best_known", "f_final": f_star,
            "scale_final": scale, "start_time_s": 0.1, "start_f": -0.9, "start_scale": 2.0,
            "trajectory": [], "certified": certified}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def test_load_rows_takes_files_and_run_directories_and_keeps_the_last_success(tmp_path):
    a = make_row(seed=10000)
    b = make_row(seed=10001)
    err = {**a, "error": "RuntimeError: boom", "traceback": "..."}
    redo = {**a, "f_final": -0.95}
    run_dir = tmp_path / "20261011-000000-E2-screen-abc123"
    run_dir.mkdir()
    write_jsonl(run_dir / "results.jsonl", [err, b])
    shard = write_jsonl(tmp_path / "s0.jsonl", [a, redo])
    rows, errors = pilot.load_rows([run_dir, shard])
    assert {r["cell_key"]: r["f_final"] for r in rows} == {a["cell_key"]: -0.95, b["cell_key"]: b["f_final"]}
    assert errors == []  # the cell recovered in a later file
    rows, errors = pilot.load_rows([run_dir])
    assert [r["cell_key"] for r in errors] == [a["cell_key"]] and [r["cell_key"] for r in rows] == [b["cell_key"]]


# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------


def test_the_reference_of_an_instance_is_the_best_value_any_method_or_the_start_found():
    rows = [
        make_row(config="sa-00", traj=[(2.0, -0.98)], scale=2.0),
        make_row(config="sa-01", traj=[(1.0, -0.99), (4.0, -1.00)], scale=2.5, seed=10001),
        make_row(instance="i2", config="sa-00", start=(0.1, -0.50), scale=1.0),
    ]
    refs = pilot.reference_table(rows)
    assert refs["i1"].f_ref == -1.00 and refs["i1"].s_ref == 2.5  # the scale of the portfolio that attains it
    assert refs["i1"].ref_type == "best_known" and refs["i1"].n_rows == 2
    assert refs["i2"].f_ref == -0.50 and refs["i2"].s_ref == 1.0  # nothing beat the start


def test_a_certified_optimum_is_the_reference_and_flags_a_method_that_beats_it():
    rows = [make_row(traj=[(2.0, -0.98)]), ref_row(f_star=-1.00, scale=2.1)]
    ref = pilot.reference_table(rows)["i1"]
    assert (ref.f_ref, ref.s_ref, ref.ref_type, ref.inconsistent) == (-1.0, 2.1, "certified", False)
    # an uncertified long run is only another candidate
    rows = [make_row(traj=[(2.0, -0.98)]), ref_row(f_star=-0.99, certified=False)]
    ref = pilot.reference_table(rows)["i1"]
    assert (ref.f_ref, ref.ref_type) == (-0.99, "best_known")
    # a method below a "certified" optimum is a contradiction to investigate, not a win
    rows = [make_row(traj=[(2.0, -1.10)]), ref_row(f_star=-1.00)]
    ref = pilot.reference_table(rows)["i1"]
    assert ref.inconsistent is True and ref.f_ref == -1.10 and ref.ref_type == "certified"


def test_numerical_noise_below_the_certified_optimum_is_not_an_inconsistency():
    rows = [make_row(traj=[(2.0, -1.0 - 1e-9)]), ref_row(f_star=-1.00)]
    ref = pilot.reference_table(rows)["i1"]
    assert ref.inconsistent is False and ref.f_ref == -1.0  # the certificate holds to 1e-6·s


# --------------------------------------------------------------------------------------
# Primal integral, gap, time-to-target of a row
# --------------------------------------------------------------------------------------


def _ref(f=-1.0, s=2.0):
    return pilot.RefValue("i1", f, s, "best_known", "x", 1, False)


def test_primal_integral_of_a_row_is_taken_after_the_common_start():
    """A2: the primary measure of a row is PI⁺ = PI − t₀/T, the integral over [t₀, T] of γ."""
    row = make_row(traj=[(2.0, -0.98), (5.0, -1.0)])
    # γ = 1 on [0, 0.1), 0.05 on [0.1, 2), 0.01 on [2, 5), 0 after
    plus = (1.9 * 0.05 + 3.0 * 0.01) / 10.0
    assert pilot.row_pi(row, _ref()) == pytest.approx(plus, abs=1e-15)
    assert pilot.row_pi(row, _ref(), variant="plus") == pilot.row_pi(row, _ref())
    # the integral of the old definition counts the common construction time of the start as well
    full = (0.1 * 1.0 + 1.9 * 0.05 + 3.0 * 0.01) / 10.0
    assert pilot.row_pi(row, _ref(), variant="full") == pytest.approx(full, abs=1e-15)
    assert full - plus == pytest.approx(0.1 / 10.0, abs=1e-15)


def test_the_sensitivity_variant_of_a_warm_row_is_one_until_the_first_improvement_of_the_start():
    """A4: what ``with_start=False`` computed was this, not "until the method's own first solution"."""
    row = make_row(traj=[(2.0, -0.98), (5.0, -1.0)])
    expected = (2.0 * 1.0 + 3.0 * 0.01) / 10.0
    assert pilot.row_pi(row, _ref(), variant="first_improvement") == pytest.approx(expected, abs=1e-15)
    # on an instance whose start is optimal it is 1 for every method: no information, as the consult says
    optimal = make_row(start=(0.1, -1.0))
    assert pilot.row_pi(optimal, _ref(), variant="first_improvement") == 1.0
    assert pilot.row_pi(optimal, _ref()) == 0.0  # and PI⁺ is 0: the instance cannot be told apart either
    with pytest.raises(TypeError):
        pilot.row_pi(row, _ref(), with_start=False)  # the old, misnamed switch is gone


def _cold_row(traj, **extra):
    """A row of the cold stage: no common start, the trajectory starts at the method's first solution."""
    row = make_row(traj=traj, stage="cold", budget=60.0, **extra)
    row.update(start_time_s=None, start_f=None, start_scale=None, cold=True, f_at_budget=traj[-1][1])
    return row


def test_the_honest_variant_is_computed_from_cold_rows_only():
    row = _cold_row([(1.5, -0.90), (4.0, -0.99), (8.0, -1.0)])
    # γ = 1 until the first own solution at 1.5 s, then 0.05, 0.005 and 0
    expected = (1.5 * 1.0 + 2.5 * 0.05 + 4.0 * 0.005) / 60.0 + 0.0
    assert pilot.row_pi(row, _ref(), variant="own") == pytest.approx(expected, abs=1e-15)
    assert pilot.row_pi(row, _ref(), variant="own", budget_s=10.0) == pytest.approx(
        (1.5 + 2.5 * 0.05 + 4.0 * 0.005) / 10.0, abs=1e-15)
    with pytest.raises(ValueError, match="cold"):
        pilot.row_pi(row, _ref())  # PI⁺ needs the common start the row does not have
    with pytest.raises(ValueError, match="cold"):
        pilot.row_pi(make_row(traj=[(2.0, -0.98)]), _ref(), variant="own")  # a warm row is no cold run
    with pytest.raises(ValueError, match="cold"):
        pilot.row_pi(row, _ref(), variant="first_improvement")


def test_row_pi_refuses_an_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        pilot.row_pi(make_row(), _ref(), variant="confined")


def test_a_row_without_improvements_has_the_integral_of_the_start_after_it():
    row = make_row()
    assert pilot.row_pi(row, _ref()) == pytest.approx(9.9 * 0.05 / 10.0)
    assert pilot.row_pi(row, _ref(), variant="full") == pytest.approx((0.1 + 9.9 * 0.05) / 10.0)


def test_a_row_whose_start_failed_still_counts_from_the_time_the_attempt_took():
    row = make_row(traj=[(2.0, -0.98)])
    row.update(start_f=None)  # the construction failed: no usable start, but 0.1 s were spent on it
    assert pilot.row_pi(row, _ref()) == pytest.approx((1.9 * 1.0 + 8.0 * 0.01) / 10.0, abs=1e-15)


def test_gap_at_the_budget_is_the_normalised_incumbent_at_that_time():
    row = make_row(traj=[(2.0, -0.98), (12.0, -1.0)])  # the last improvement arrives after the budget
    assert pilot.row_gap_at_budget(row, _ref()) == pytest.approx(0.02 / 2.0)
    assert pilot.row_gap_at_budget(make_row(start=(20.0, -0.9)), _ref()) == math.inf  # nothing in time


def test_time_to_target_is_censored_at_the_budget():
    row = make_row(traj=[(2.0, -0.9985), (5.0, -0.9999)])  # ĝ = 7.5e-4 at 2 s, 5e-5 at 5 s
    ttt = pilot.row_ttt(row, _ref(), 1e-3)
    assert (ttt.time_s, ttt.reached) == (2.0, True)
    ttt = pilot.row_ttt(row, _ref(), 1e-4)
    assert (ttt.time_s, ttt.reached) == (5.0, True)
    ttt = pilot.row_ttt(make_row(), _ref(), 1e-3)  # never within 1e-3 of the reference
    assert (ttt.time_s, ttt.reached) == (10.0, False)
    ttt = pilot.row_ttt(make_row(start=(0.1, -0.9995)), _ref(), 1e-3)  # the start is already there
    assert (ttt.time_s, ttt.reached) == (0.1, True)


# --------------------------------------------------------------------------------------
# Scores, ranking and the tuning protocol
# --------------------------------------------------------------------------------------


def _instances_rows(config, pis_by_instance, *, size=100, budget=10.0, name="sa", seeds=(10000, 10001, 10002),
                    stage="screen"):
    """Rows of ``config`` on instances ``i0, i1, …`` whose primal integrals are ``pis_by_instance``:
    the start is already at the reference from ``t_hit`` on, so PI is t_hit/budget·γ + …; here the
    incumbent is a flat gap ``g`` from the start on, PI = (t0 + (budget − t0)·g)/budget."""
    rows = []
    for i, gap in enumerate(pis_by_instance):
        for seed in seeds:
            f = -1.0 + 2.0 * gap  # ĝ = gap with s_ref = 2
            rows.append(make_row(instance=f"i{i}", config=config, name=name, size=size, seed=seed,
                                 budget=budget, start=(0.0, f), stage=stage))
    return rows


def _with_reference(rows, n_instances):
    return rows + [ref_row(instance=f"i{i}", f_star=-1.0, scale=2.0, size=100) for i in range(n_instances)]


def test_config_score_is_the_median_over_instances_of_the_median_over_seeds():
    rows = _instances_rows("sa-00", [0.1, 0.2, 0.5])
    # a bad seed on instance i1 must not move its median
    rows.append(make_row(instance="i1", config="sa-00", seed=10003, start=(0.0, -0.1)))
    refs = pilot.reference_table(_with_reference(rows, 3))
    (score,) = pilot.config_scores(rows, refs, budget_s=10.0, criterion="pi")
    assert score.config_id == "sa-00" and score.n_instances == 3 and score.n_runs == 10
    assert score.criterion == pytest.approx(0.2)  # median of (0.1, 0.2, 0.5)
    assert score.per_instance == pytest.approx({"i0": 0.1, "i1": 0.2, "i2": 0.5})
    (gap_score,) = pilot.config_scores(rows, refs, budget_s=10.0, criterion="gap")
    assert gap_score.criterion == pytest.approx(0.2)  # a flat gap: the final gap equals the integral


def test_scores_refuse_rows_of_another_budget_and_error_free_input_only():
    rows = _instances_rows("sa-00", [0.1]) + _instances_rows("sa-00", [0.1], budget=60.0)
    refs = pilot.reference_table(rows)
    with pytest.raises(ValueError, match="budget"):
        pilot.config_scores(rows, refs, budget_s=10.0, criterion="pi")
    with pytest.raises(ValueError, match="criterion"):
        pilot.config_scores(rows[:3], refs, budget_s=10.0, criterion="rank")


def test_ttt_median_is_the_kaplan_meier_median_over_all_runs():
    rows = []
    for i, hit in enumerate([1.0, 2.0, 3.0, None]):
        traj = [(hit, -0.9999)] if hit is not None else []
        rows.append(make_row(instance=f"i{i}", config="sa-00", start=(0.0, -0.5), traj=traj))
    refs = pilot.reference_table(_with_reference(rows, 4))
    (score,) = pilot.config_scores(rows, refs, budget_s=10.0, criterion="pi")
    assert score.ttt_censored == 1 and score.ttt_median == 2.0  # S drops to ½ at 2 s; one run is censored
    rows = [make_row(instance=f"i{i}", config="sa-00", start=(0.0, -0.5)) for i in range(3)]
    refs = pilot.reference_table(_with_reference(rows, 3))
    (score,) = pilot.config_scores(rows, refs, budget_s=10.0, criterion="pi")
    assert score.ttt_median == math.inf and score.ttt_censored == 3  # the median lies beyond the budget


def _score(config_id, criterion, ttt, name="sa"):
    return pilot.ConfigScore(config_id, name, {"name": name, "id": config_id}, 8, 24, criterion, {}, ttt, 0)


def test_configurations_within_five_percent_of_the_best_are_ordered_by_their_ttt_median():
    scores = [_score("a", 0.100, 5.0), _score("b", 0.103, 2.0), _score("c", 0.104, 9.0), _score("d", 0.200, 1.0)]
    assert [s.config_id for s in pilot.order_configs(scores)] == ["b", "a", "c", "d"]
    # outside the band the criterion decides, whatever the TTT says
    scores = [_score("a", 0.100, 5.0), _score("b", 0.106, 2.0)]
    assert [s.config_id for s in pilot.order_configs(scores)] == ["a", "b"]


def test_ties_the_ttt_cannot_break_fall_back_to_the_criterion_and_the_id():
    scores = [_score("b", 0.1, math.inf), _score("a", 0.1, math.inf), _score("c", 0.101, math.inf)]
    assert [s.config_id for s in pilot.order_configs(scores)] == ["a", "b", "c"]
    zeros = [_score("b", 0.0, 3.0), _score("a", 0.0, 3.0)]
    assert [s.config_id for s in pilot.order_configs(zeros)] == ["a", "b"]  # deterministic


def test_the_tie_band_is_anchored_at_the_best_not_chained():
    scores = [_score("a", 0.100, 9.0), _score("b", 0.104, 5.0), _score("c", 0.108, 1.0)]
    # c is within 5 % of b but not of a: it is not in a's tie group
    assert [s.config_id for s in pilot.order_configs(scores)] == ["b", "a", "c"]


def test_the_winner_at_60_seconds_must_not_be_worse_than_the_runner_up_by_more_than_five_percent():
    order = [_score("w", 0.10, 2.0), _score("s", 0.11, 2.0)]
    keep = pilot.recheck_decision(order, {"w": 0.094, "s": 0.090})
    assert keep["chosen"] == "w" and keep["reason"] == "winner"
    swap = pilot.recheck_decision(order, {"w": 0.100, "s": 0.090})  # 11 % worse
    assert swap["chosen"] == "s" and "runner-up" in swap["reason"]
    assert swap["winner"] == "w" and swap["second"] == "s"
    assert swap["criterion_long"] == {"w": 0.100, "s": 0.090}


def test_a_decision_says_when_the_order_of_the_top_two_was_arbitrary():
    """Configurations that tie on the score and on the TTT median are ordered by id: say so."""
    order = [_score("a", 0.0, 3.0), _score("b", 0.0, 3.0)]
    decision = pilot.recheck_decision(order, {"a": 0.0, "b": 0.0})
    assert decision["order_by_id"] is True
    assert pilot.recheck_decision([_score("a", 0.0, 3.0), _score("b", 0.0, 4.0)], {"a": 0.0, "b": 0.0})[
        "order_by_id"] is False  # the TTT median told them apart
    assert pilot.recheck_decision([_score("a", 0.1, 3.0), _score("b", 0.3, 3.0)], {"a": 0.1, "b": 0.3})[
        "order_by_id"] is False  # the score did
    assert pilot.recheck_decision([_score("a", 0.1, 3.0)], {"a": 0.1})["order_by_id"] is False  # nobody to tell


def test_the_five_percent_rule_has_an_absolute_floor_for_zero_values():
    order = [_score("w", 0.0, 2.0), _score("s", 0.0, 2.0)]
    assert pilot.recheck_decision(order, {"w": 0.0, "s": 0.0})["chosen"] == "w"
    assert pilot.recheck_decision(order, {"w": 1e-4, "s": 0.0})["chosen"] == "s"
    assert pilot.recheck_decision(order, {"w": 1e-12, "s": 0.0})["chosen"] == "w"  # numerical noise


def test_a_lone_configuration_wins_and_a_missing_recheck_is_an_error():
    only = pilot.recheck_decision([_score("w", 0.1, 1.0)], {"w": 0.1})
    assert only["chosen"] == "w" and only["second"] is None
    with pytest.raises(ValueError, match="recheck"):
        pilot.recheck_decision([_score("w", 0.1, 1.0), _score("s", 0.2, 1.0)], {"w": 0.1})


# --------------------------------------------------------------------------------------
# The screen report and the tuned report
# --------------------------------------------------------------------------------------


def _screen_rows(size=100):
    """Three SA, two tabu and one SWAP/LS/SCIP configuration on 3 instances (flat gaps), 3 seeds."""
    gaps = {
        ("sa", "sa-00"): [0.10, 0.10, 0.10], ("sa", "sa-01"): [0.12, 0.12, 0.12],
        ("sa", "sa-02"): [0.30, 0.30, 0.30], ("tabu", "tabu-00"): [0.05, 0.05, 0.05],
        ("tabu", "tabu-01"): [0.20, 0.20, 0.20], ("swap_sa", "swap-00"): [0.07, 0.07, 0.07],
        ("ls", "ls-00"): [0.02, 0.02, 0.02], ("scip", "scip-a"): [0.40, 0.40, 0.40],
        ("oa", "oa-highs"): [0.35, 0.35, 0.35],
    }
    rows = []
    for (name, config), g in gaps.items():
        rows += _instances_rows(config, g, size=size, name=name)
    return rows


def test_the_screen_report_names_winner_and_runner_up_per_group_and_criterion():
    rows = _with_reference(_screen_rows(), 3)
    report = pilot.screen_report(rows, budget_s=10.0)
    assert report["kind"] == "e2-screen"
    size = report["sizes"]["100"]
    groups = size["groups"]
    assert set(groups) == {"exact/pi", "sa/pi", "sa/gap", "tabu/pi", "tabu/gap", "swap_sa/pi", "ls/pi"}
    assert (groups["sa/pi"]["winner"], groups["sa/pi"]["second"]) == ("sa-00", "sa-01")
    assert (groups["exact/pi"]["winner"], groups["exact/pi"]["second"]) == ("oa-highs", "scip-a")
    assert groups["ls/pi"]["second"] is None  # a lone configuration
    # the recheck plan: the union of winners and runners-up, each once, with the reasons
    plan = {item["spec"]["id"]: item["why"] for item in size["recheck"]}
    assert set(plan) == {"sa-00", "sa-01", "tabu-00", "tabu-01", "swap-00", "ls-00", "scip-a", "oa-highs"}
    assert sorted(plan["sa-00"]) == ["sa/gap:winner", "sa/pi:winner"]
    json.dumps(report, allow_nan=False)


def test_the_screen_report_states_how_degenerate_the_common_start_makes_the_criterion():
    rows = _with_reference(_screen_rows(), 3)
    # the start of instance i0 is the reference itself
    rows += [make_row(instance="i0", config="sa-00", seed=10009, start=(0.0, -1.0))]
    report = pilot.screen_report(rows, budget_s=10.0)
    diag = report["sizes"]["100"]["start_gap"]
    assert diag["n_instances"] == 3 and diag["tau"] == TAU_MAIN
    assert diag["share_within_tau"] == pytest.approx(1 / 3)
    assert diag["share_optimal"] == pytest.approx(1 / 3)
    assert report["warnings"] == []  # a third of the instances: no warning
    # when the start is within τ of the best known on most instances, the criterion is degenerate
    easy = [make_row(instance=f"i{i}", config="sa-00", start=(0.0, -1.0 + 1e-5)) for i in range(4)]
    easy += [ref_row(instance=f"i{i}") for i in range(4)]
    report = pilot.screen_report(easy, budget_s=10.0)
    assert report["sizes"]["100"]["start_gap"]["share_within_tau"] == 1.0
    assert any("start" in w and "100" in w for w in report["warnings"])


def test_the_screen_report_warns_when_configurations_cannot_be_told_apart():
    """The degenerate case of a start that is already optimal: every score is 0 and every TTT is the start's
    time, so the 'winner' is whatever the id order puts first. The report must not hide that."""
    rows = []
    for i in range(3):
        for config in ("sa-00", "sa-01", "sa-02"):
            rows.append(make_row(instance=f"i{i}", config=config, name="sa", start=(0.0, -1.0)))
        rows.append(ref_row(instance=f"i{i}", f_star=-1.0))
    report = pilot.screen_report(rows, budget_s=10.0)
    group = report["sizes"]["100"]["groups"]["sa/pi"]
    assert group["order_by_id"] is True and group["n_in_tie_band"] == 3 and group["winner"] == "sa-00"
    assert any("sa/pi" in w and "id" in w for w in report["warnings"])
    # a clear winner raises no such warning
    clear = pilot.screen_report(_with_reference(_screen_rows(), 3), budget_s=10.0)
    assert clear["sizes"]["100"]["groups"]["sa/pi"]["order_by_id"] is False
    assert clear["sizes"]["100"]["groups"]["sa/pi"]["n_in_tie_band"] == 1
    assert not any("tie" in w for w in clear["warnings"])


def test_rows_of_skipped_instances_do_not_enter_the_reports():
    rows = _with_reference(_screen_rows(), 3)
    skipped = {"cell_key": "skip-1", "stage": "screen", "instance_id": "iX", "config_size": 100,
               "method": {"name": "sa", "id": "sa-99"}, "config_id": "sa-99", "status": "skipped",
               "skipped": "not_binding", "binding": False}
    report = pilot.screen_report([*rows, skipped], budget_s=10.0)
    assert "sa-99" not in json.dumps(report) and "iX" not in pilot.reference_table([*rows, skipped])


def test_the_screen_report_refuses_rows_with_errors_and_of_the_wrong_budget():
    rows = _with_reference(_screen_rows(), 3)
    with pytest.raises(ValueError, match="budget"):
        pilot.screen_report(rows, budget_s=60.0)
    bad = {**rows[0], "error": "boom"}
    with pytest.raises(ValueError, match="error"):
        pilot.screen_report([bad, *rows], budget_s=10.0)


def _recheck_rows(ids_gaps, *, size=100, name_of=None):
    rows = []
    for config, gaps in ids_gaps.items():
        name = (name_of or {}).get(config, config.split("-")[0])
        rows += _instances_rows(config, gaps, size=size, name=name, budget=60.0, seeds=(10003, 10004, 10005),
                                stage="recheck")
    return rows


def test_the_tuned_report_picks_per_group_by_the_rules_and_names_the_representatives():
    screen = _with_reference(_screen_rows(), 3)
    plan = pilot.screen_report(screen, budget_s=10.0)
    # at 60 s the SA winner has fallen behind its runner-up: the runner-up is chosen
    recheck = _recheck_rows({
        "sa-00": [0.10] * 3, "sa-01": [0.05] * 3, "tabu-00": [0.04] * 3, "tabu-01": [0.2] * 3,
        "swap-00": [0.06] * 3, "ls-00": [0.01] * 3, "scip-a": [0.3] * 3, "oa-highs": [0.2] * 3,
    }, name_of={"scip-a": "scip", "oa-highs": "oa", "swap-00": "swap_sa"})
    tuned = pilot.tuned_report(screen, recheck, budget_s=10.0, budget_long_s=60.0)
    assert tuned["kind"] == "e2-tuned"
    roles = tuned["sizes"]["100"]["roles"]
    assert roles["sa"]["id"] == "sa-01"  # the runner-up, by the 60 s rule
    assert roles["tabu"]["id"] == "tabu-00" and roles["swap_sa"]["id"] == "swap-00"
    assert roles["ls"]["id"] == "ls-00" and roles["exact_rep"]["id"] == "oa-highs"
    # the QUBO representative is the best of the three tuned heuristics at 10 s: tabu (0.05) < swap (0.07) < sa
    assert roles["qubo_rep"]["id"] == "tabu-00"
    assert roles["h1c_sa"]["id"] == "sa-01" and roles["h1c_tabu"]["id"] == "tabu-00"
    detail = tuned["sizes"]["100"]["decisions"]["sa/pi"]
    assert detail["winner"] == "sa-00" and detail["chosen"] == "sa-01"
    assert tuned["sizes"]["100"]["screen"]["groups"]["sa/pi"]["winner"] == "sa-00"
    assert set(roles) == set(pilot.e2_benchmark.ROLES)
    json.dumps(tuned, allow_nan=False)
    assert plan["sizes"]["100"]["recheck"]  # the same plan the recheck was built from


def test_the_tuned_report_needs_the_recheck_rows_of_every_planned_configuration():
    screen = _with_reference(_screen_rows(), 3)
    recheck = _recheck_rows({"sa-00": [0.1] * 3})
    with pytest.raises(ValueError, match="recheck"):
        pilot.tuned_report(screen, recheck, budget_s=10.0, budget_long_s=60.0)


# --------------------------------------------------------------------------------------
# Eligibility (§12 A1) in the reports of the pilot
# --------------------------------------------------------------------------------------


def test_reference_rows_keep_a_certificate_of_an_earlier_round(tmp_path):
    """Audit G2, C-2: round 2 of the test screening re-screened a point that round 1 had certified, and merging the
    rounds by the last row of a cell lost the certificate. The rows of reference runs are not merged across runs."""
    first = {**ref_row(f_star=-1.0, scale=2.1), "stage": "eligibility", "eligibility": "nontrivial"}
    second = {**first, "ref_type": "best_known", "certified": False, "status": "timelimit"}
    runs = []
    for name, row in (("20261012-203423-E2-elig_test-a", first), ("20261013-032118-E2-elig_test-a", second)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        write_jsonl(run_dir / "results.jsonl", [row])
        runs.append(run_dir)
    # the generic loader keeps the last row of a cell (the semantics the tuning reports rely on) ...
    assert pilot.reference_table(pilot._rows_of(runs))["i1"].ref_type == "best_known"
    # ... the reference loader keeps every run's rows, so a certificate of any round counts
    got = pilot.reference_table(pilot.reference_rows(runs))["i1"]
    assert (got.f_ref, got.s_ref, got.ref_type) == (-1.0, 2.1, "certified")
    # within one run the last success of a cell still wins, and error rows are left out
    redo = {**second, "f_star": -1.2}
    err = {**first, "cell_key": "ref|other", "instance_id": "i2", "error": "RuntimeError: boom"}
    write_jsonl(runs[1] / "results.jsonl", [second, redo, err])
    rows = pilot.reference_rows(runs[1:])
    assert [r["f_star"] for r in rows] == [-1.2]


def test_the_rows_of_a_full_screening_are_references_like_the_rows_of_the_reference_stage():
    ref = {**ref_row(f_star=-1.00, scale=2.1), "stage": "eligibility", "eligibility": "nontrivial"}
    rows = [make_row(traj=[(2.0, -0.98)]), ref]
    got = pilot.reference_table(rows)["i1"]
    assert (got.f_ref, got.s_ref, got.ref_type, got.inconsistent) == (-1.0, 2.1, "certified", False)
    # a screening that stopped at a limit has no f_star and is only another candidate
    stopped = {k: v for k, v in ref.items() if k not in ("f_star", "scale_star", "ref_type")}
    assert pilot.reference_table([make_row(traj=[(2.0, -0.98)]), stopped])["i1"].ref_type == "best_known"


def _tuned_rows_of_size(size, *, scale_gap=1.0):
    """Screen and recheck rows of one size, enough for the tuned report of that size."""
    screen = _with_reference(_screen_rows(size=size), 3)
    recheck = _recheck_rows({
        "sa-00": [0.10] * 3, "sa-01": [0.05] * 3, "tabu-00": [0.04] * 3, "tabu-01": [0.2] * 3,
        "swap-00": [0.06] * 3, "ls-00": [0.01] * 3, "scip-a": [0.3] * 3, "oa-highs": [0.2] * 3,
    }, size=size, name_of={"scip-a": "scip", "oa-highs": "oa", "swap-00": "swap_sa"})
    return screen, recheck


def test_a_size_with_too_few_non_trivial_calibration_instances_borrows_the_configurations_of_another():
    screen, recheck = _tuned_rows_of_size(200)
    tuned = pilot.tuned_report(screen, recheck, budget_s=10.0, budget_long_s=60.0, borrow={100: 200})
    assert set(tuned["sizes"]) == {"100", "200"}
    assert tuned["sizes"]["100"]["borrowed_from"] == 200 and "borrowed_from" not in tuned["sizes"]["200"]
    assert tuned["sizes"]["100"]["roles"] == tuned["sizes"]["200"]["roles"]
    assert set(tuned["sizes"]["100"]["roles"]) == set(pilot.e2_benchmark.ROLES)
    json.dumps(tuned, allow_nan=False)
    assert any("borrow" in w and "100" in w for w in tuned["warnings"])


def test_borrowing_needs_a_donor_that_was_tuned_and_a_size_that_was_not():
    screen, recheck = _tuned_rows_of_size(200)
    with pytest.raises(ValueError, match="nothing to borrow"):
        pilot.tuned_report(screen, recheck, budget_s=10.0, budget_long_s=60.0, borrow={100: None})
    with pytest.raises(ValueError, match="donor"):
        pilot.tuned_report(screen, recheck, budget_s=10.0, budget_long_s=60.0, borrow={100: 400})
    with pytest.raises(ValueError, match="screening rows"):
        pilot.tuned_report(screen, recheck, budget_s=10.0, budget_long_s=60.0, borrow={200: 100})


def test_only_a_finished_calibration_scan_decides_which_cells_are_descriptive():
    unfinished = {"kind": "e2-eligibility", "done": False, "cells": {"T100": {
        "enough": False, "selected": [], "min_nontrivial": 8, "done": False}}}
    cells = {"C1": {"label": "T100", "config_size": 100}}
    with pytest.raises(ValueError, match="not finished"):
        pilot._descriptive_cells(unfinished, cells)
    unfinished["done"] = True
    unfinished["cells"]["T100"]["done"] = True
    assert "fewer than the 8" in pilot._descriptive_cells(unfinished, cells)["C1"]


def test_a_descriptive_cell_gets_the_minimum_n_and_no_sd():
    exact = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    rows = _bench_rows(100, "T100", {"exact": exact, "qubo": [1.3 * x for x in exact], "ls": exact})
    cells = {"C1": {"label": "T100", "config_size": 100}}
    report = pilot.n_rule_report(rows, _tuned_for(100), cells=cells, descriptive={"C1": "fewer than 8"})
    cell = report["cells"]["C1"]
    assert report["n_instances"]["C1"] == 20 and cell["descriptive"] == "fewer than 8"
    assert cell["sd_log_r"] is None and cell["sd_log_r_b"] is None
    # no rows at all is no obstacle for a descriptive cell, and an ordinary cell still needs them
    only = pilot.n_rule_report([], _tuned_for(100), cells=cells, descriptive={"C1": "none"})
    assert only["n_instances"] == {"C1": 20} and only["cells"]["C1"]["n_calibration"] == 0
    with pytest.raises(ValueError, match="calibration instances"):
        pilot.n_rule_report([], _tuned_for(100), cells=cells)
    json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------------------
# The rules: n per cell, TTT instances, C4 lambda
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sd", "n"),
    [(0.0, 20), (0.2, 20), (0.5, 24), (0.4, 20), (0.6, 34), (0.7, 40), (1.0, 40), (3.0, 40)],
)
def test_n_per_cell_follows_the_plan_formula_and_its_clipping(sd, n):
    # n = max(20, ceil((2·1.96·SD / ln 1.5)²)), at most 40
    expected = min(40, max(20, math.ceil((2 * 1.96 * sd / math.log(1.5)) ** 2)))
    assert pilot.n_per_cell(sd) == expected == n


def test_n_per_cell_rejects_a_negative_or_missing_sd():
    for bad in (-0.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="SD"):
            pilot.n_per_cell(bad)


def _bench_rows(size, cell, per_instance, *, stage="recheck", seeds=(10003, 10004, 10005), budget=60.0):
    """Rows of the three representatives (ids exact-r, qubo-r, ls-r) with flat gaps per instance."""
    rows = []
    for config, name, gaps in (("exact-r", "scip", per_instance["exact"]), ("qubo-r", "sa", per_instance["qubo"]),
                               ("ls-r", "ls", per_instance["ls"])):
        for i, gap in enumerate(gaps):
            for seed in seeds:
                rows.append(make_row(instance=f"{cell}/{i}", config=config, name=name, size=size, seed=seed,
                                     budget=budget, start=(0.0, -1.0 + 2.0 * gap), cell=cell, stage=stage))
    for i in range(len(per_instance["exact"])):
        rows.append(ref_row(instance=f"{cell}/{i}", size=size))
    return rows


def _tuned_for(size):
    spec = lambda i, n: {"name": n, "id": i}  # noqa: E731
    return {"kind": "e2-tuned", "sizes": {str(size): {"roles": {
        "exact_rep": spec("exact-r", "scip"), "qubo_rep": spec("qubo-r", "sa"), "ls": spec("ls-r", "ls"),
        "sa": spec("qubo-r", "sa"), "tabu": spec("t", "tabu"), "swap_sa": spec("s", "swap_sa"),
        "h1c_sa": spec("qubo-r", "sa"), "h1c_tabu": spec("t", "tabu")}}}}


def test_the_n_rule_uses_the_larger_sd_of_log_r_and_log_r_b():
    exact = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    qubo = [0.01 * math.exp(x) for x in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.0, 0.3, -0.3)]
    ls = [0.01] * 8
    rows = _bench_rows(100, "T100", {"exact": exact, "qubo": qubo, "ls": ls})
    report = pilot.n_rule_report(rows, _tuned_for(100), cells={"C1": {"label": "T100", "config_size": 100}})
    log_r = np.log(np.array(qubo) / np.array(exact))
    assert report["cells"]["C1"]["sd_log_r"] == pytest.approx(np.std(log_r, ddof=1), rel=1e-6)
    assert report["cells"]["C1"]["sd_log_r_b"] == pytest.approx(np.std(log_r, ddof=1), rel=1e-6)  # LS = exact here
    sd = max(report["cells"]["C1"]["sd_log_r"], report["cells"]["C1"]["sd_log_r_b"])
    assert report["n_instances"]["C1"] == pilot.n_per_cell(sd) and report["cells"]["C1"]["n_calibration"] == 8
    assert report["kind"] == "e2-n-rule"


def test_the_n_rule_uses_pi_plus_and_has_no_floor():
    """A2: R and R_b are ratios of PI⁺. The floor of 1e-6 is gone (it lay below t₀/T and never acted)."""
    assert not hasattr(pilot, "PI_FLOOR")
    exact = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    rows = _bench_rows(100, "T100", {"exact": exact, "qubo": [2 * x for x in exact], "ls": exact})
    for r in rows:  # a start that took 0.2 s: PI⁺ = PI − 0.2/60, a ratio of the first is not one of the second
        if r["stage"] != "reference":
            r["start_time_s"] = 0.2
    report = pilot.n_rule_report(rows, _tuned_for(100), cells={"C1": {"label": "T100", "config_size": 100}})
    cell = report["cells"]["C1"]
    plus = lambda g: g * (60.0 - 0.2) / 60.0  # noqa: E731  (a flat gap g after the start)
    assert cell["pi"]["exact"]["T100/3"] == pytest.approx(plus(0.04), rel=1e-12)
    assert cell["sd_log_r"] == pytest.approx(0.0, abs=1e-9)  # R = 2 on every instance
    assert "pi_floor" not in report and report["measure"] == "PI+"
    # the ratio of PI over [0, T] would have been 2 only if the common term were absent
    full = pilot.row_pi(next(r for r in rows if r["instance_id"] == "T100/3" and r["config_id"] == "qubo-r"),
                        pilot.RefValue("T100/3", -1.0, 2.0, "certified", "reference", 1, False), variant="full")
    assert full != pytest.approx(2 * plus(0.04), rel=1e-6)


def test_the_n_rule_refuses_an_instance_where_pi_plus_vanishes():
    """PI⁺ = 0 means the start is optimal: such an instance is not non-trivial (A1) and must not have got
    into the calibration set; the ratio would be 0/0. No floor hides it."""
    rows = _bench_rows(100, "T100", {"exact": [0.0] + [0.01] * 7, "qubo": [0.0] + [0.02] * 7,
                                     "ls": [0.0] + [0.01] * 7})
    with pytest.raises(ValueError, match=r"PI\+ = 0.*T100/0"):
        pilot.n_rule_report(rows, _tuned_for(100), cells={"C1": {"label": "T100", "config_size": 100}})


def test_the_n_rule_needs_at_least_two_calibration_instances_with_all_three_representatives():
    rows = _bench_rows(100, "T100", {"exact": [0.01], "qubo": [0.02], "ls": [0.01]})
    with pytest.raises(ValueError, match="calibration instances"):
        pilot.n_rule_report(rows, _tuned_for(100), cells={"C1": {"label": "T100", "config_size": 100}})


def test_the_ttt_instances_are_the_three_closest_to_the_median_difficulty_of_the_exact_representative():
    exact = [0.01, 0.02, 0.03, 0.04, 0.05, 0.065, 0.07, 0.08]  # median 0.045
    rows = _bench_rows(200, "T200", {"exact": exact, "qubo": exact, "ls": exact})
    # the instance seeds are not in the instance ids; here instance i ↔ seed 1008 + i
    for r in rows:
        r["seed"] = 1008 + int(r["instance_id"].split("/")[1])
    report = pilot.ttt_instances_report(rows, _tuned_for(200), label="T200", config_size=200)
    assert report["kind"] == "e2-ttt-instances"
    # distances to the median: 0.035 0.025 0.015 0.005 0.005 0.020 0.025 0.035 → instances 2, 3, 4
    assert report["instances"] == [1010, 1011, 1012]
    assert report["median_pi"] == pytest.approx(0.045, abs=1e-9)
    assert len(report["difficulty"]) == 8 and report["exact_rep"] == "exact-r"


def _draw(seed):
    mu, sigma, _ = factor_instance(30, seed)
    return mu, sigma


def test_the_c4_report_wraps_the_rule_for_the_admissible_case():
    report = pilot.c4_lambda_report(_draw, seeds=range(1100, 1104), lambdas=(5.0, 10.0, 20.0),
                                    min_median_scont=1)
    assert report["kind"] == "e2-c4-lambda"
    assert report["lam"] == 5.0 and report["lam_used"] == 5.0 and report["admissible"] is True
    assert report["seeds"] == [1100, 1101, 1102, 1103] and report["threshold"] == 1
    assert set(report["median_s_cont"]) == {"5.0", "10.0", "20.0"}
    assert report["section12"].startswith("20") and "λ" in report["section12"]
    json.dumps(report, allow_nan=False)


def test_a_descriptive_c4_is_run_at_the_largest_candidate_lambda():
    report = pilot.c4_lambda_report(_draw, seeds=range(1100, 1104), lambdas=(5.0, 10.0, 20.0),
                                    min_median_scont=10**6)
    assert report["lam"] is None and report["lam_used"] == 20.0 and report["admissible"] is False
    assert "описат" in report["section12"]  # the log says that C4 becomes descriptive


def test_the_c4_rule_defaults_are_those_of_the_plan():
    import inspect

    sig = inspect.signature(pilot.c4_lambda_report)
    assert sig.parameters["min_median_scont"].default == 20
    assert tuple(sig.parameters["lambdas"].default) == (5.0, 10.0, 20.0)
    assert list(sig.parameters["seeds"].default) == list(range(1100, 1110))


# --------------------------------------------------------------------------------------
# The command line, on files
# --------------------------------------------------------------------------------------


def test_cli_screen_tune_n_rule_and_ttt_write_their_reports(tmp_path):
    screen = write_jsonl(tmp_path / "screen.jsonl", _with_reference(_screen_rows(), 3))
    out = tmp_path / "screen.json"
    assert pilot.main(["screen", "--rows", str(screen), "--budget", "10", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["kind"] == "e2-screen" and report["inputs"][0]["n_rows"] == 81 + 3
    assert report["inputs"][0]["sha256"] and report["section12"].startswith("20")
    recheck = write_jsonl(tmp_path / "recheck.jsonl", _recheck_rows({
        "sa-00": [0.10] * 3, "sa-01": [0.05] * 3, "tabu-00": [0.04] * 3, "tabu-01": [0.2] * 3,
        "swap-00": [0.06] * 3, "ls-00": [0.01] * 3, "scip-a": [0.3] * 3, "oa-highs": [0.2] * 3,
    }, name_of={"scip-a": "scip", "oa-highs": "oa", "swap-00": "swap_sa"}))
    tuned_path = tmp_path / "tuned.json"
    assert pilot.main(["tune", "--screen", str(screen), "--recheck", str(recheck), "--budget", "10",
                       "--budget-long", "60", "--out", str(tuned_path)]) == 0
    tuned = json.loads(tuned_path.read_text())
    assert tuned["kind"] == "e2-tuned" and tuned["sizes"]["100"]["roles"]["qubo_rep"]["id"] == "tabu-00"
    assert [i["source"] for i in tuned["inputs"]] == [str(screen), str(recheck)]
    # a report is an input of a run: the runner must not take its provenance entries for recorded data
    # files (a dict with "path" and "sha256" would be re-hashed, and a run directory cannot be)
    assert run.verify_inputs({"screen": out, "tuned": tuned_path}) == {}


def test_cli_n_rule_and_ttt_instances_read_the_tuned_report(tmp_path):
    exact = [0.01, 0.02, 0.03, 0.04, 0.05, 0.065, 0.07, 0.08]
    qubo = [x * 1.3 for x in exact]
    rows = _bench_rows(200, "T200", {"exact": exact, "qubo": qubo, "ls": exact})
    for r in rows:
        r["seed"] = 1008 + int(r["instance_id"].split("/")[1])
    path = write_jsonl(tmp_path / "recheck.jsonl", rows)
    tuned = tmp_path / "tuned.json"
    tuned.write_text(json.dumps(_tuned_for(200)))
    out = tmp_path / "n_rule.json"
    assert pilot.main(["n-rule", "--rows", str(path), "--tuned", str(tuned), "--cell", "C2=T200:200",
                       "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["n_instances"] == {"C2": 20} and report["cells"]["C2"]["sd_log_r"] == pytest.approx(0.0, abs=1e-9)
    assert {i["source"] for i in report["inputs"]} == {str(path), str(tuned)}
    out = tmp_path / "ttt.json"
    assert pilot.main(["ttt-instances", "--rows", str(path), "--tuned", str(tuned), "--label", "T200",
                       "--config-size", "200", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["instances"] == [1010, 1011, 1012]


def test_cli_c4_lambda_runs_the_rule_on_the_panel(tmp_path, monkeypatch):
    from cif.experiments import e2_instances
    from tests.test_e2_instances import _panel

    panel = _panel()
    monkeypatch.setattr(e2_instances, "load_panel", lambda path: panel)
    (tmp_path / "provenance.json").write_text("{}")
    out = tmp_path / "c4.json"
    assert pilot.main(["c4-lambda", "--data-dir", str(tmp_path), "--n", "10", "--min-median-scont", "1",
                       "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["kind"] == "e2-c4-lambda" and report["lam"] == 5.0 and report["admissible"] is True
    assert report["seeds"] == list(range(1100, 1110)) and report["inputs"][0]["source"].endswith("provenance.json")


def test_the_c4_report_names_the_code_state_it_was_computed_with(tmp_path, monkeypatch):
    """No registered run stands behind the report of the λ rule (audit G1, L-4): the code state is its lineage, as in
    the manifest of a run: the SHA of HEAD and whether the code differed from it. Unknown is never written as clean."""
    from cif.experiments import e2_instances, registry
    from tests.test_e2_instances import _panel

    panel = _panel()
    monkeypatch.setattr(e2_instances, "load_panel", lambda path: panel)
    (tmp_path / "provenance.json").write_text("{}")
    args = ["c4-lambda", "--data-dir", str(tmp_path), "--n", "10", "--min-median-scont", "1"]
    for state in ({"sha": "9f2c1ab", "dirty": False}, {"sha": "9f2c1ab", "dirty": True}, {"sha": "", "dirty": None}):
        monkeypatch.setattr(registry, "_git_state", lambda state=state: state)
        out = tmp_path / f"c4_{state['dirty']}.json"
        assert pilot.main([*args, "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert (report["git_sha"], report["git_dirty"]) == (state["sha"], state["dirty"])


def test_the_eligibility_command_prints_the_table_of_a3_and_the_pooled_group(tmp_path, capsys):
    import yaml

    from tests.test_e2_eligibility import erow

    cfg = {"stream": "E2", "name": "elig_dense", "experiment": "cif.experiments.e2_benchmark", "inputs": {},
           "params": {"stage": "eligibility", "seed_role": "test",
                      "eligibility": {"tau": 1e-3, "certificate": "tau", "margin": 1e-3},
                      "instances": [{"cell": "D1_100", "family": "dense", "n": 100, "config_size": 100,
                                    "seeds": [2000, 2001, 2002]},
                                    {"cell": "D1_200", "family": "dense", "n": 200, "config_size": 200,
                                    "seeds": [2000, 2001]}]}}
    config = tmp_path / "elig_dense.yaml"
    config.write_text(yaml.safe_dump(cfg))
    rows = [erow("D1_100", 2000, "trivial", family="dense"), erow("D1_100", 2001, "nontrivial", family="dense"),
            erow("D1_100", 2002, "undecided", family="dense"), erow("D1_200", 2000, "trivial", family="dense", n=200),
            erow("D1_200", 2001, "trivial", family="dense", n=200)]
    path = write_jsonl(tmp_path / "round1.jsonl", rows)
    out = tmp_path / "eligibility_dense.json"
    assert pilot.main(["eligibility", "--config", str(config), "--rows", str(path), "--out", str(out)]) == 0
    shown = capsys.readouterr().out
    assert "D1_100: trivial 1 of 3" in shown and "D1_200: trivial 2 of 2 (100 %" in shown
    assert "D1 (pooled): trivial 3 of 5 (60 %" in shown and "scan finished" in shown
    report = json.loads(out.read_text())
    assert report["groups"]["D1"]["n_scanned"] == 5 and report["role"] == "test"
    # an unreadable class in the rows stops the command: the audit is part of it
    bad = write_jsonl(tmp_path / "bad.jsonl", [{**rows[0], "eligibility": "nontrivial"}])
    with pytest.raises(ValueError, match="do not follow"):
        pilot.main(["eligibility", "--config", str(config), "--rows", str(bad), "--out", str(out)])


def test_every_report_kind_is_the_one_the_benchmark_reads():
    """The ``kind`` strings are a contract between this module and ``e2_benchmark``'s inputs."""
    from cif.experiments import e2_benchmark

    assert e2_benchmark.TUNED_KIND == "e2-tuned"
    src = Path(e2_benchmark.__file__).read_text()
    for kind in ("e2-screen", "e2-n-rule", "e2-ttt-instances", "e2-c4-lambda"):
        assert f'"{kind}"' in src, kind
