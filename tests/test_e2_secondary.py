"""Secondary analyses of RQ2: time-to-target, ``t_primal`` / ``t_cert`` of the exact class, performance profiles
(ANALYSIS_PLAN §2 RQ2 "Вторично", §5, §7.2, §10; §12 A1-A2, the entry of 2026-10-12 "Прочтения подтверждающей
оценки H2/H2b"). ``python -m cif.experiments.e2_analysis secondary``; the module is ``cif.experiments.e2_secondary``.

The rows here are synthetic and every value below is worked out by hand. The incumbent of a row stands at the gap
0.01 (units of the scale, certificate -1) from the end of the common start to its ``reach`` time; from then on it
stands at 5e-4, which meets the target τ = 1e-3 but not 1e-4; a pair ``(t1, t2)`` goes on to 5e-5 at ``t2``, which
meets 1e-4 as well; ``None`` is a run that never gets there. With one reach time for all the seeds of a role the
Kaplan-Meier median over the seeds is that time. The statistics (bootstrap, outcomes) are tested in
``tests/test_stats_rq2.py``; here it is the wiring: which numbers go into them.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from cif.experiments import e2_analysis as an
from cif.experiments import e2_pilot as pilot
from cif.experiments import e2_secondary as sec
from cif.stats import rq2
from cif.stats.rq1 import wilson_interval
from tests.test_e2_h2 import (
    F_STAR,
    SCALE,
    SEEDS,
    SIZE,
    cert_rows,
    clean_state,
    elig,
    gap_row,
    inst_rows,
    write_json,
    write_run,
)

G_START = 0.01  # gap of the common start
G_1E3, G_1E4 = 5e-4, 5e-5  # gaps that meet τ = 1e-3 only / both targets
ROLE_KEYS = (("exact_rep", "exact"), ("qubo_rep", "qubo"), ("ls", "ls"))
LN2 = math.log(2.0)


# --------------------------------------------------------------------------------------------
# Synthetic rows
# --------------------------------------------------------------------------------------------


def ttt_row(cell, unit, role, solver_seed, reach, *, t0=0.0, g_start=G_START, **kw) -> dict:
    """A row of ``role`` whose incumbent meets τ = 1e-3 at ``reach`` (seconds; a pair ``(t1, t2)`` meets 1e-3 at
    ``t1`` and 1e-4 at ``t2``; ``None`` never); the common start stands at the gap ``g_start``."""
    t1, t2 = reach if isinstance(reach, tuple) else (reach, None)
    traj = []
    if t1 is not None:
        traj.append((t1, F_STAR + G_1E3 * SCALE))
    if t2 is not None:
        traj.append((t2, F_STAR + G_1E4 * SCALE))
    return gap_row(cell, unit, role, solver_seed, g_start, t0=t0, traj=traj, **kw)


def seeds_of(reach, n: int) -> list:
    """One reach time for every seed, or a list with one per seed."""
    reaches = list(reach) if isinstance(reach, list) else [reach] * n
    assert len(reaches) == n
    return reaches


def world(spec: dict[str, list[dict]], *, members=None, cell_kw=None, **elig_kwargs):
    """``spec[cell]`` is a list of instances, each a dict with ``exact``, ``qubo``, ``ls`` (a reach time for all the
    seeds of the role or a list of one per seed) and optionally ``unit`` (default 2000, 2001, …), ``size``, ``t0``,
    ``exact_extra`` (a list of dicts of fields per exact seed); ``cell_kw[cell]`` adds fields to the state of the cell
    in the eligibility report (``enough``). Returns ``(rows, certificates, eligibility report)``."""
    rows, certs, cells = [], [], {}
    for cell, instances in spec.items():
        units = []
        for k, inst in enumerate(instances):
            unit = inst.get("unit", 2000 + k)
            size = inst.get("size") or SIZE.get(cell, 100)
            units.append(unit)
            certs += cert_rows(cell, [unit], size=size)
            for role, key in ROLE_KEYS:
                reaches = seeds_of(inst[key], SEEDS[role])
                extras = inst.get("exact_extra") if role == "exact_rep" else None
                for i, reach in enumerate(reaches):
                    rows.append(ttt_row(cell, unit, role, 20000 + i, reach, size=size, t0=inst.get("t0", 0.0),
                                        g_start=inst.get("g_start", G_START), **(extras[i] if extras else {})))
        cells[cell] = {"selected": units, **(cell_kw or {}).get(cell, {})}
    return rows, certs, elig(cells, members=members, **elig_kwargs)


def standard(exact=20.0, qubo=10.0, ls=5.0, **kw) -> dict:
    return {"exact": exact, "qubo": qubo, "ls": ls, **kw}


def find(records, cell, unit):
    (hit,) = [r for r in records if r["cell"] == cell and r["unit"] == unit]
    return hit


def primary(report) -> dict:
    return report["ttt"]["primary"]


# --------------------------------------------------------------------------------------------
# The time to the target of one role on one instance
# --------------------------------------------------------------------------------------------


def role_rows(reaches, *, t0=2.0, role="exact_rep") -> list[dict]:
    return [ttt_row("C1", 2000, role, 20000 + i, reach, t0=t0) for i, reach in enumerate(reaches)]


REF = pilot.RefValue("factor/100/2000", F_STAR, SCALE, "certified", "reference", 3, False)


def test_the_time_to_target_of_a_row_is_the_end_to_end_time_of_the_first_incumbent_within_tau():
    """Seed 0 starts at 2 s (gap 0.01), improves to 0.004 at 10 s, to 5e-4 at 30 s and to 5e-5 at 50 s: it meets
    1e-3 at 30 s and 1e-4 at 50 s. Seed 1 meets 1e-3 at 25 s and never 1e-4, seed 2 never. At 1e-3 the Kaplan-Meier
    median of (30, 25, censored) is the second event, 30 s; at 1e-4 one run in three has an event, S stays above
    1/2, the median is beyond the budget and is censored at 60 s."""
    rows = [
        gap_row("C1", 2000, "exact_rep", 20000, G_START, t0=2.0, traj=[
            (10.0, F_STAR + 0.004), (30.0, F_STAR + G_1E3), (50.0, F_STAR + G_1E4)]),
        gap_row("C1", 2000, "exact_rep", 20001, G_START, t0=2.0, traj=[(25.0, F_STAR + G_1E3)]),
        gap_row("C1", 2000, "exact_rep", 20002, G_START, t0=2.0),
    ]
    main = sec.role_ttt(rows, REF, tau=1e-3, censor_s=60.0)
    assert (main["n_seeds"], main["n_reached"], main["censored"]) == (3, 2, False)
    assert main["median_s"] == 30.0 and main["value_s"] == 30.0
    strict = sec.role_ttt(rows, REF, tau=1e-4, censor_s=60.0)
    assert (strict["n_reached"], strict["censored"]) == (1, True)
    assert strict["median_s"] is None and strict["value_s"] == 60.0


def test_a_shorter_censoring_budget_recensors_the_same_trajectories():
    """The same three rows at the censoring budget 30 s: the run that meets the target at 30 s still does (the
    budget is inclusive), the median stays 30; at 29 s only the run at 25 s has an event, the median is beyond the
    budget and is censored at 29."""
    rows = role_rows([30.0, 25.0, None])
    at_30 = sec.role_ttt(rows, REF, tau=1e-3, censor_s=30.0)
    assert (at_30["n_reached"], at_30["median_s"], at_30["value_s"], at_30["censored"]) == (2, 30.0, 30.0, False)
    at_29 = sec.role_ttt(rows, REF, tau=1e-3, censor_s=29.0)
    assert (at_29["n_reached"], at_29["median_s"], at_29["value_s"], at_29["censored"]) == (1, None, 29.0, True)


def test_the_instance_value_is_the_lower_median_of_the_reached_times_of_the_seeds():
    """Ten seeds reach at 5, 6, 7, 8, 9, 10 s and four never: S after the fifth event is 5/10, the median is the
    fifth time, 9 s (the lower sample median). With four events the curve stops at 6/10, the median is censored."""
    half = sec.role_ttt(role_rows([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, None, None, None, None]), REF, tau=1e-3,
                        censor_s=60.0)
    assert (half["n_reached"], half["median_s"], half["censored"]) == (6, 9.0, False)
    short = sec.role_ttt(role_rows([5.0, 6.0, 7.0, 8.0] + [None] * 6), REF, tau=1e-3, censor_s=60.0)
    assert (short["n_reached"], short["median_s"], short["value_s"], short["censored"]) == (4, None, 60.0, True)
    none = sec.role_ttt(role_rows([None] * 3), REF, tau=1e-3, censor_s=60.0)
    assert (none["n_reached"], none["censored"], none["value_s"]) == (0, True, 60.0)


def test_a_start_that_already_meets_the_target_stops_the_analysis_naming_the_instance():
    """By A1 a selected instance has a start above τ; a start at 5e-4 (the time to the target would be the time of
    the start itself) means that the screening and the reference disagree. No ratio is formed from it."""
    rows, certs, report = world({"C1": [standard(), standard()]})
    rows = [r for r in rows if not (r["seed"] == 2001 and r["roles"] == ["exact_rep"])]
    rows += [gap_row("C1", 2001, "exact_rep", 20000 + i, 5e-4) for i in range(3)]
    with pytest.raises(ValueError, match=r"factor/100/2001.*exact.*already meets the target"):
        sec.secondary_report(rows, [report], references=certs)
    # the screening lists the instances whose certified gap puts them on the other side of τ (A1, 'Уточнения' (1)); a
    # listed instance is another case (below), and one that is not listed stays an error
    report["reference_check"] = {"n_checked": 2, "flipped": [
        {"instance_id": "factor/100/2000", "class": "nontrivial", "g_start": 5e-4},
        {"instance_id": "factor/100/2001", "class": "trivial", "g_start": 2e-3}]}
    with pytest.raises(ValueError, match=r"factor/100/2001.*exact.*already meets the target.*reference_check"):
        sec.secondary_report(rows, [report], references=certs)


FLIPPED_WORLD = {"C1": [
    standard(exact=(20.0, 40.0), qubo=(10.0, 20.0), ls=(5.0, 10.0)),
    standard(exact=(20.0, 40.0), qubo=(10.0, 20.0), ls=(5.0, 10.0)),
    standard(exact=(5.0, 30.0), qubo=(5.0, 15.0), ls=(5.0, 10.0), g_start=5e-4),  # the start is at 5e-4 ≤ τ
]}


def flipped_world():
    rows, certs, report = world(FLIPPED_WORLD)
    report["reference_check"] = {"n_checked": 3, "flipped": [
        {"instance_id": "factor/100/2002", "class": "nontrivial", "g_start": 5e-4}]}
    return rows, certs, report


def test_an_instance_the_screening_lists_as_flipped_leaves_the_ttt_estimates_and_is_counted():
    """The screening decides with the scale of the start; with the scale of the optimum the start of instance 2002 is at
    5e-4, within τ = 1e-3 (A1 'Уточнения' (1): listed in ``reference_check``, the decision stays). For it the time to the
    target of every method is the build of the start, which says nothing about the methods: it leaves the TTT estimates
    (the ratio would be 1 up to the noise of the clock) and is counted; H2, the PI⁺ profile and the times of the exact
    class keep it. At 1e-4 the start does not meet the target and the instance is in: R = 15/30 = 1/2 like the others."""
    rows, certs, report = flipped_world()
    out = sec.secondary_report(rows, [report], references=certs)
    main = primary(out)
    flipped = find(main["instances"], "C1", 2002)
    assert flipped["start_met"] is True and flipped["TTT"]["status"] == "start_met" and flipped["TTT"]["ratio"] is None
    assert flipped["TTTb"]["status"] == "start_met" and flipped["g_start"] == pytest.approx(5e-4)
    assert find(main["instances"], "C1", 2000)["start_met"] is False
    assert main["start_met"] == [{"cell": "C1", "unit": 2002, "instance_id": "factor/100/2002",
                                  "g_start": pytest.approx(5e-4)}]
    cell = main["cells"]["C1"]
    assert cell["n_instances"] == 3 and cell["TTT"]["n_clusters"] == 2 and cell["censoring"]["start_met"] == 1
    assert cell["censoring"]["TTT"] == {"both": 2, "qubo_only": 0, "other_only": 0, "neither": 0}
    assert cell["censoring"]["censored_instances"] == {"exact": 0, "qubo": 0, "ls": 0}
    assert main["pooled"]["TTT"]["estimate_log"] == pytest.approx(-LN2, abs=1e-12)
    assert any("factor/100/2002" in w and "start" in w for w in out["warnings"])
    strict = out["ttt"]["sensitivity"]["tau_1e-4"]
    assert strict["start_met"] == [] and strict["cells"]["C1"]["TTT"]["n_clusters"] == 3
    assert find(strict["instances"], "C1", 2002)["TTT"]["ratio"] == pytest.approx(0.5)
    assert out["ttt"]["sensitivity"]["budget_30"]["cells"]["C1"]["censoring"]["start_met"] == 1
    # the other analyses keep the instance
    assert len(out["instances"]) == 3 and out["exact_times"]["cells"]["C1"]["n_instances"] == 3
    profiles = out["profiles"]["cells"]["C1"]
    assert profiles["PI_plus"]["n_instances"] == 3 and profiles["TTT"]["n_instances"] == 2
    assert profiles["TTT"]["n_start_met"] == 1 and profiles["PI_plus"]["n_start_met"] == 0


# --------------------------------------------------------------------------------------------
# The ratio of an instance and the cases the plan leaves open
# --------------------------------------------------------------------------------------------


def test_the_ratio_of_an_instance_is_the_ratio_of_the_kaplan_meier_medians_against_the_exact_method_and_ls():
    rows, certs, report = world({"C1": [standard(), standard()]})
    out = sec.secondary_report(rows, [report], references=certs)
    inst = find(primary(out)["instances"], "C1", 2000)
    assert inst["instance_id"] == "factor/100/2000" and inst["cluster"] == "2000"
    assert {k: v["value_s"] for k, v in inst["roles"].items()} == {"exact": 20.0, "qubo": 10.0, "ls": 5.0}
    assert inst["roles"]["qubo"]["n_seeds"] == 10 and inst["roles"]["exact"]["n_seeds"] == 3
    assert inst["TTT"] == {"status": "both", "bound": None, "ratio": pytest.approx(0.5),
                           "log_ratio": pytest.approx(-LN2)}
    assert inst["TTTb"]["ratio"] == pytest.approx(2.0) and inst["TTTb"]["log_ratio"] == pytest.approx(LN2)


CENSORING_WORLD = {"C1": [
    standard(exact=[20.0, None, None]),                      # A: the exact median is beyond the budget
    standard(qubo=[5.0, 6.0, 7.0, 8.0] + [None] * 6),        # B: the QUBO median is beyond the budget
    standard(exact=None, qubo=None),                         # C: neither of the two
]}


def test_a_median_beyond_the_budget_gives_a_ratio_with_the_budget_as_a_bound_not_a_floor():
    """A: the exact median is censored (valued at the budget, 60 s): R = 10/60, an upper bound of the true ratio
    (the exact method is slower than that). B: the QUBO median is censored: R = 60/20 = 3, a lower bound. Against
    local search (5 s) B and C are lower bounds, 60/5 = 12; A is exact, 10/5."""
    rows, certs, report = world(CENSORING_WORLD)
    records = primary(sec.secondary_report(rows, [report], references=certs))["instances"]
    a, b, c = (find(records, "C1", u) for u in (2000, 2001, 2002))
    assert a["TTT"] == {"status": "qubo_only", "bound": "upper", "ratio": pytest.approx(10 / 60),
                        "log_ratio": pytest.approx(math.log(10 / 60))}
    assert b["TTT"] == {"status": "other_only", "bound": "lower", "ratio": pytest.approx(3.0),
                        "log_ratio": pytest.approx(math.log(3.0))}
    assert a["roles"]["exact"]["censored"] is True and a["roles"]["exact"]["value_s"] == 60.0
    assert a["roles"]["exact"]["median_s"] is None
    assert (a["TTTb"]["status"], a["TTTb"]["ratio"]) == ("both", pytest.approx(2.0))
    assert (b["TTTb"]["status"], b["TTTb"]["bound"], b["TTTb"]["ratio"]) == ("other_only", "lower", pytest.approx(12.0))
    assert (c["TTTb"]["status"], c["TTTb"]["bound"], c["TTTb"]["ratio"]) == ("other_only", "lower", pytest.approx(12.0))


def test_an_instance_where_neither_median_is_observed_is_left_out_of_that_ratio_and_counted():
    """C: both medians are beyond the budget, a ratio inf/inf that says nothing about the speed of either: it is
    not floored to 1 (that would pull the pooled estimate towards 'equivalent'); the instance stays in the other pair.
    The cell mean of log R over A and B is (ln 1/6 + ln 3)/2 = -ln 2/2; against LS over A, B, C (ln 2 + 2 ln 12)/3."""
    rows, certs, report = world(CENSORING_WORLD)
    out = sec.secondary_report(rows, [report], references=certs)
    c = find(primary(out)["instances"], "C1", 2002)
    assert c["TTT"] == {"status": "neither", "bound": None, "ratio": None, "log_ratio": None}
    cell = primary(out)["cells"]["C1"]
    assert cell["n_instances"] == 3
    assert cell["censoring"]["TTT"] == {"both": 0, "qubo_only": 1, "other_only": 1, "neither": 1}
    assert cell["censoring"]["TTTb"] == {"both": 1, "qubo_only": 0, "other_only": 2, "neither": 0}
    assert cell["censoring"]["censored_instances"] == {"exact": 2, "qubo": 2, "ls": 0}
    assert cell["TTT"]["n_clusters"] == 2 and cell["TTTb"]["n_clusters"] == 3
    assert cell["TTT"]["estimate_log"] == pytest.approx(-0.5 * LN2, abs=1e-12)
    assert cell["TTTb"]["estimate_log"] == pytest.approx((LN2 + 2 * math.log(12.0)) / 3, abs=1e-12)
    pooled = primary(out)["pooled"]
    assert pooled["TTT"]["cells"] == ["C1"] and pooled["TTT"]["estimate_log"] == pytest.approx(-0.5 * LN2)
    assert pooled["TTTb"]["estimate_log"] == pytest.approx((LN2 + 2 * math.log(12.0)) / 3, abs=1e-12)


# --------------------------------------------------------------------------------------------
# The pooled estimate
# --------------------------------------------------------------------------------------------


def two_cells():
    """C1: three instances with R = 1/2 and R_b = 2; C2: two with R = 1/4 and R_b = 1 (the times to 1e-4 are later, so
    that every variant of the sensitivity has runs that reach its target)."""
    spec = {"C1": [standard(exact=(20.0, 40.0), qubo=(10.0, 20.0), ls=(5.0, 10.0))] * 3,
            "C2": [standard(exact=(40.0, 55.0), qubo=(10.0, 20.0), ls=(10.0, 20.0))] * 2}
    return world(spec)


def test_the_pooled_ttt_ratio_is_the_h2_estimator_applied_to_the_log_ratios_of_the_times():
    """Equal cell weights: (ln 1/2 + ln 1/4)/2 = -1.5 ln 2 against the exact method, (ln 2 + ln 1)/2 = ln 2/2
    against LS. The interval and the standard error are those of ``rq2.estimate_rq2`` on the cluster means with the
    constants of H2; every instance of a cell has the same ratio, so the intervals are points."""
    rows, certs, report = two_cells()
    out = sec.secondary_report(rows, [report], references=certs)
    pooled = primary(out)["pooled"]
    assert pooled["TTT"]["cells"] == ["C1", "C2"] and pooled["TTT"]["weights"] == "equal"
    assert pooled["TTT"]["estimate_log"] == pytest.approx(-1.5 * LN2, abs=1e-12)
    assert pooled["TTT"]["ratio"] == pytest.approx(2.0**-1.5)
    assert pooled["TTTb"]["estimate_log"] == pytest.approx(0.5 * LN2, abs=1e-12)
    direct = rq2.estimate_rq2(
        {"C1": {"log_r": [-LN2] * 3}, "C2": {"log_r": [-2 * LN2] * 2}}, pooled=["C1", "C2"], n_boot=rq2.N_BOOT,
        seed=rq2.SEED)
    assert pooled["TTT"]["ci_log"] == pytest.approx(
        [direct.pooled["log_r"].interval.low, direct.pooled["log_r"].interval.high], abs=1e-12)
    assert pooled["TTT"]["se"] == pytest.approx(direct.pooled["log_r"].se, abs=1e-12)
    assert pooled["TTT"]["level"] == pytest.approx(0.95)
    # the same five outcomes against the band [1/1.5, 1.5] as H2: 0.35 is wholly below 1/1.5, 1.41 inside the band
    assert pooled["TTT"]["outcome"] == "qubo_better" and pooled["TTTb"]["outcome"] == "equivalent"
    for cell in ("C1", "C2"):
        assert primary(out)["cells"][cell]["TTT"]["level"] == pytest.approx(0.995)  # Bonferroni over two cells
    assert primary(out)["cells"]["C2"]["TTT"]["estimate_log"] == pytest.approx(-2 * LN2)


def test_a_pooled_cell_with_fewer_than_two_ratios_leaves_the_pool_of_that_estimate_with_a_note():
    """C2 has two instances where neither median is observed: no cluster is left for the ratio against the exact
    method, so the pooled estimate of that pair is C1 alone, and the note says why; against LS C2 has two."""
    spec = {"C1": [standard()] * 3, "C2": [standard(exact=None, qubo=None)] * 2}
    rows, certs, report = world(spec)
    out = sec.secondary_report(rows, [report], references=certs)
    block = primary(out)
    assert block["pooled"]["TTT"]["cells"] == ["C1"]
    assert block["pooled"]["TTT"]["estimate_log"] == pytest.approx(-LN2)
    assert block["pooled"]["TTTb"]["cells"] == ["C1", "C2"]
    assert block["pooled"]["TTTb"]["estimate_log"] == pytest.approx(0.5 * (LN2 + math.log(12.0)), abs=1e-12)
    assert block["cells"]["C2"]["TTT"] is None and block["cells"]["C2"]["n_instances"] == 2
    assert any("C2" in n and "TTT" in n and "pool" in n for n in block["notes"])


def test_c5_ports_are_the_clusters_of_the_ttt_ratio():
    """Port 1 has two points with R = 1/2 and 1/8, port 2 one with 1/2, port 3 one with 2: the port means are -2 ln 2,
    -ln 2 and ln 2, the cell mean -2/3 ln 2 (the mean over the four points would be -ln 2)."""
    size = {1: 31, 2: 85, 3: 89}
    points = [(1, 0), (1, 1), (2, 0), (3, 0)]
    exact, qubo = [40.0, 40.0, 40.0, 10.0], [20.0, 5.0, 20.0, 20.0]
    instances = [standard(exact=e, qubo=q, unit=p, size=size[p[0]]) for p, e, q in zip(points, exact, qubo, strict=True)]
    rows, certs, _ = world({"C5": instances})
    report = elig({"C5": {"selected": points, "per_port": {1: 2, 2: 1, 3: 1, 4: 0, 5: 0}}})
    out = sec.secondary_report(rows, [report], references=certs)
    cell = primary(out)["cells"]["C5"]
    assert cell["n_instances"] == 4 and cell["TTT"]["n_clusters"] == 3
    assert cell["TTT"]["estimate_log"] == pytest.approx(-2 * LN2 / 3, abs=1e-12)
    assert {r["cluster"] for r in primary(out)["instances"]} == {"port1", "port2", "port3"}
    assert primary(out)["pooled"]["TTT"]["cells"] == ["C5"]
    assert primary(out)["pooled"]["TTT"]["estimate_log"] == pytest.approx(-2 * LN2 / 3, abs=1e-12)


def test_a_descriptive_cell_is_estimated_without_entering_the_pooled_estimate():
    spec = {"C1": [standard()] * 2, "C4": [standard(exact=5.0, qubo=10.0)] * 2}
    rows, certs, report = world(spec, members=["C1"],
                                excluded={"C4": "descriptive: the λ rule of N7 found no admissible λ"})
    out = sec.secondary_report(rows, [report], references=certs)
    assert primary(out)["pooled"]["TTT"]["cells"] == ["C1"]
    assert primary(out)["cells"]["C4"]["TTT"]["estimate_log"] == pytest.approx(LN2)
    assert primary(out)["cells"]["C4"]["TTT"]["in_pooled"] is False
    assert primary(out)["cells"]["C1"]["TTT"]["in_pooled"] is True
    assert out["descriptive"] == {"C4": "descriptive: the λ rule of N7 found no admissible λ"}


def test_when_no_cell_is_pooled_there_is_no_pooled_ttt_estimate_and_the_statement_of_a3_stands():
    statement = an.NO_POOLED_STATEMENT
    rows, certs, report = world({"C1": [standard()] * 2}, members=[], statement=statement,
                                cell_kw={"C1": {"enough": False}})
    out = sec.secondary_report(rows, [report], references=certs)
    assert primary(out)["pooled"] == {"TTT": None, "TTTb": None}
    assert out["statement"] == statement
    assert primary(out)["cells"]["C1"]["TTT"]["estimate_log"] == pytest.approx(-LN2)
    empty = elig({"C1": {"selected": [], "enough": False}}, members=[], statement=statement)
    out = sec.secondary_report([], [empty])
    assert primary(out)["pooled"] == {"TTT": None, "TTTb": None} and out["instances"] == []
    assert primary(out)["cells"]["C1"]["n_instances"] == 0 and out["profiles"]["cells"] == {}


# --------------------------------------------------------------------------------------------
# The sensitivities
# --------------------------------------------------------------------------------------------


def test_the_sensitivity_at_tau_1e_4_uses_the_same_rows_with_the_stricter_target():
    """Exact meets 1e-3 at 30 s and 1e-4 at 50 s, QUBO at 10 and 20 s, LS at 5 and 40 s: at 1e-3 R = 1/3 and
    R_b = 2; at 1e-4 R = 20/50 = 0.4 and R_b = 20/40 = 0.5. The main estimate does not move."""
    inst = standard(exact=(30.0, 50.0), qubo=(10.0, 20.0), ls=(5.0, 40.0))
    rows, certs, report = world({"C1": [inst] * 3})
    out = sec.secondary_report(rows, [report], references=certs)
    main, strict = primary(out), out["ttt"]["sensitivity"]["tau_1e-4"]
    assert main["tau"] == 1e-3 and strict["tau"] == 1e-4 and strict["censor_s"] == 60.0
    assert main["pooled"]["TTT"]["estimate_log"] == pytest.approx(math.log(1 / 3), abs=1e-12)
    assert main["pooled"]["TTTb"]["estimate_log"] == pytest.approx(LN2, abs=1e-12)
    assert strict["pooled"]["TTT"]["estimate_log"] == pytest.approx(math.log(0.4), abs=1e-12)
    assert strict["pooled"]["TTTb"]["estimate_log"] == pytest.approx(math.log(0.5), abs=1e-12)


def test_the_censoring_budget_sensitivities_recensor_the_trajectories_and_never_drop_a_run():
    """Exact reaches the target at 20, 40 and 40 s on every instance, QUBO at 8 s. At the budget 60 the exact median
    is 40 (R = 8/40). At 30 s only one exact run has an event: the median is beyond the budget and is valued at 30
    (R = 8/30, an upper bound). At 10 s no exact run has: valued at 10 (R = 8/10, an upper bound)."""
    inst = standard(exact=[20.0, 40.0, 40.0], qubo=8.0, ls=5.0)
    rows, certs, report = world({"C1": [inst] * 3})
    out = sec.secondary_report(rows, [report], references=certs)
    assert set(out["ttt"]["sensitivity"]) == {"tau_1e-4", "budget_30", "budget_10"}
    assert primary(out)["censor_s"] == 60.0
    assert primary(out)["pooled"]["TTT"]["estimate_log"] == pytest.approx(math.log(8 / 40), abs=1e-12)
    for name, value in (("budget_30", 30.0), ("budget_10", 10.0)):
        block = out["ttt"]["sensitivity"][name]
        assert block["censor_s"] == value and block["tau"] == 1e-3
        assert block["pooled"]["TTT"]["estimate_log"] == pytest.approx(math.log(8 / value), abs=1e-12)
        rec = block["instances"][0]
        assert rec["TTT"]["status"] == "qubo_only" and rec["TTT"]["bound"] == "upper"
        assert rec["roles"]["exact"]["censored"] is True and rec["roles"]["exact"]["value_s"] == value
        assert rec["roles"]["exact"]["n_seeds"] == 3  # the censored runs stay in the sample
    assert out["ttt"]["sensitivity"]["budget_10"]["cells"]["C1"]["censoring"]["censored_instances"]["exact"] == 3


# --------------------------------------------------------------------------------------------
# t_primal and t_cert of the exact class
# --------------------------------------------------------------------------------------------


def exact_extra(*seeds) -> list[dict]:
    """``(t_primal, t_cert or None, certified)`` per seed of the exact representative."""
    return [{"t_primal_s": p, "t_cert_s": c, "certified": cert, "proved": cert} for p, c, cert in seeds]


EXACT_WORLD = {"C1": [
    standard(exact_extra=exact_extra((10.0, 12.0, True), (14.0, 18.0, True), (40.0, None, False))),
    standard(exact_extra=exact_extra((30.0, 30.0, True), (50.0, None, False), (55.0, None, False))),
    standard(exact_extra=exact_extra((5.0, 5.0, True), (6.0, 6.0, True), (7.0, 7.0, True))),
    standard(exact_extra=exact_extra((20.0, 20.0, True), (20.0, 20.0, True), (20.0, 20.0, True))),
]}


def test_t_primal_and_t_cert_of_the_exact_representative_per_instance():
    """Instance 0: t_primal 10, 14, 40 (median 14); the certificates at 12 and 18 s, the third run has none: the
    Kaplan-Meier median is the second event, 18 s. Instance 1: one certificate in three runs: the median is beyond
    the budget (censored), t_primal 30, 50, 55 (median 50). Instances 2 and 3 certify in all runs."""
    rows, certs, report = world(EXACT_WORLD)
    out = sec.secondary_report(rows, [report], references=certs)
    records = out["exact_times"]["instances"]
    expected = {2000: (14.0, 18.0, 2), 2001: (50.0, None, 1), 2002: (6.0, 6.0, 3), 2003: (20.0, 20.0, 3)}
    for unit, (t_primal, t_cert, n_cert) in expected.items():
        rec = find(records, "C1", unit)
        assert rec["t_primal_median_s"] == t_primal and rec["t_cert_median_s"] == t_cert
        assert rec["t_cert_censored"] is (t_cert is None)
        assert rec["n_seeds"] == 3 and rec["n_certified"] == n_cert


def test_the_exact_times_of_a_cell_are_described_by_their_distribution_over_the_instances_and_the_runs():
    """The instance medians of t_primal are 14, 50, 6, 20: as order statistics (the smallest value with at least a
    share p of the values at or below it) min 6, q1 6, median 14, q3 20, max 50. The t_cert medians 18, ∞, 6, 20 give
    6, 6, 18, 20 and a maximum beyond the budget. Nine of the twelve runs certified; the t_primal of the nine is
    5, 6, 7, 10, 14, 20, 20, 20, 30 (median 14), that of the other three 40, 50, 55 (median 50)."""
    rows, certs, report = world(EXACT_WORLD)
    cell = sec.secondary_report(rows, [report], references=certs)["exact_times"]["cells"]["C1"]
    assert cell["n_instances"] == 4 and cell["configs"] == ["exact_rep-100"]
    assert cell["t_primal_s"] == {"n": 4, "min": 6.0, "q1": 6.0, "median": 14.0, "q3": 20.0, "max": 50.0}
    t_cert = cell["t_cert_s"]
    assert (t_cert["n"], t_cert["n_censored"]) == (4, 1)
    assert (t_cert["min"], t_cert["q1"], t_cert["median"], t_cert["q3"], t_cert["max"]) == (6.0, 6.0, 18.0, 20.0, None)
    assert t_cert["share_finite"] == 0.75
    assert t_cert["ci95_share_finite"] == pytest.approx(list(wilson_interval(3, 4)))
    runs = cell["runs"]
    assert (runs["n_runs"], runs["n_certified_in_budget"], runs["n_certified"]) == (12, 9, 9)
    assert runs["share_in_budget"] == 0.75 and runs["ci95_share_in_budget"] == pytest.approx(list(wilson_interval(9, 12)))
    assert runs["t_primal_s"]["certified"] == {"n": 9, "min": 5.0, "q1": 7.0, "median": 14.0, "q3": 20.0, "max": 30.0}
    assert runs["t_primal_s"]["not_certified"]["n"] == 3 and runs["t_primal_s"]["not_certified"]["median"] == 50.0


def test_a_certificate_found_after_the_budget_does_not_count_as_one_within_it():
    """The events after the budget do not count here either (as in PI⁺ and in the time to the target): runs that
    certify at 12, 61 and 62 s have one certificate within 60 s, the median is censored; all three are counted as
    certified runs of the solver."""
    extra = exact_extra((10.0, 12.0, True), (10.0, 61.0, True), (10.0, 62.0, True))
    rows, certs, report = world({"C1": [standard(exact_extra=extra)] * 2})
    out = sec.secondary_report(rows, [report], references=certs)
    rec = out["exact_times"]["instances"][0]
    assert rec["t_cert_censored"] is True and rec["t_cert_median_s"] is None and rec["n_certified"] == 3
    runs = out["exact_times"]["cells"]["C1"]["runs"]
    assert (runs["n_runs"], runs["n_certified_in_budget"], runs["n_certified"]) == (6, 2, 6)


def test_a_run_that_proved_nothing_has_no_certificate_whatever_the_solver_reports():
    extra = exact_extra((10.0, 12.0, True), (14.0, 18.0, False), (20.0, None, False))
    rows, certs, report = world({"C1": [standard(exact_extra=extra)] * 2})
    rec = sec.secondary_report(rows, [report], references=certs)["exact_times"]["instances"][0]
    assert rec["n_certified"] == 1 and rec["t_cert_censored"] is True  # t_cert_s of a run that is not certified is no event


# --------------------------------------------------------------------------------------------
# Performance profiles
# --------------------------------------------------------------------------------------------


def pi_world():
    """Flat gaps that are powers of two (exact in binary), so PI⁺ is the gap itself and the ratios are exact:
    instance 0 (exact, qubo, ls) = (2^-6, 2^-7, 2^-5) gives performance ratios (2, 1, 4); instance 1 (2^-5, 2^-4,
    2^-5) gives (1, 2, 1); instance 2 (2^-4, 2^-4, 2^-6) gives (4, 4, 1)."""
    gaps = [(2**-6, 2**-7, 2**-5), (2**-5, 2**-4, 2**-5), (2**-4, 2**-4, 2**-6)]
    rows = []
    for k, (e, q, ls) in enumerate(gaps):
        rows += inst_rows("C1", 2000 + k, exact=[e] * 3, qubo=[q] * 10, ls=[ls] * 10)
    return rows, cert_rows("C1", [2000, 2001, 2002]), elig({"C1": {"selected": [2000, 2001, 2002]}})


def test_the_performance_profile_of_pi_plus_is_the_dolan_more_profile_of_the_instance_medians():
    """Ratios to the best: exact (2, 1, 4), QUBO (1, 2, 4), LS (4, 1, 1). rho(1) = 1/3, 1/3, 2/3; the first two reach
    2/3 at 2 and all reach 1 at 4. Grid of the plan's reading: 1, 1.1, 1.25, 1.5, 2, 3, 4, …"""
    rows, certs, report = pi_world()
    out = sec.secondary_report(rows, [report], references=certs)
    profiles = out["profiles"]
    assert profiles["ratios"] == list(sec.PROFILE_RATIOS) and sec.PROFILE_RATIOS[:7] == (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0)
    block = profiles["cells"]["C1"]["PI_plus"]
    assert profiles["cells"]["C1"]["n_instances"] == 3 and block["n_unsolved_by_all"] == 0
    index = {r: i for i, r in enumerate(sec.PROFILE_RATIOS)}
    third, two_thirds = 1 / 3, 2 / 3
    for role, expected in {"exact": {1.0: third, 1.5: third, 2.0: two_thirds, 3.0: two_thirds, 4.0: 1.0},
                           "qubo": {1.0: third, 1.5: third, 2.0: two_thirds, 3.0: two_thirds, 4.0: 1.0},
                           "ls": {1.0: two_thirds, 1.5: two_thirds, 2.0: two_thirds, 3.0: two_thirds, 4.0: 1.0}}.items():
        for ratio, rho in expected.items():
            assert block["rho"][role][index[ratio]] == pytest.approx(rho), (role, ratio)
        assert block["rho_inf"][role] == 1.0
    assert block["rho"]["exact"][-1] == 1.0  # beyond the largest ratio everything is solved


def test_the_ttt_profile_counts_a_median_beyond_the_budget_as_a_failure_not_as_the_budget():
    """Times (exact, QUBO, LS): instance 0 (20, 10, 5) ratios (4, 2, 1); 1 (20, ∞, 40) ratios (1, ∞, 2); 2 nobody
    (stays in the denominator); 3 (10, 10, 10) ratios (1, 1, 1). rho_exact(1) = 2/4, rho_exact(4) = 3/4; QUBO (2, ∞, ∞,
    1): rho(1) = 1/4, rho(2) = 2/4, plateau 2/4; LS (1, 2, ∞, 1): rho(1) = 2/4, rho(2) = 3/4."""
    spec = {"C1": [standard(exact=20.0, qubo=10.0, ls=5.0), standard(exact=20.0, qubo=None, ls=40.0),
                   standard(exact=None, qubo=None, ls=None), standard(exact=10.0, qubo=10.0, ls=10.0)]}
    rows, certs, report = world(spec)
    block = sec.secondary_report(rows, [report], references=certs)["profiles"]["cells"]["C1"]["TTT"]
    index = {r: i for i, r in enumerate(sec.PROFILE_RATIOS)}
    assert block["n_unsolved_by_all"] == 1
    assert block["rho"]["exact"][index[1.0]] == 0.5 and block["rho"]["exact"][index[4.0]] == 0.75
    assert block["rho"]["qubo"][index[1.0]] == 0.25 and block["rho"]["qubo"][index[2.0]] == 0.5
    assert block["rho"]["ls"][index[1.0]] == 0.5 and block["rho"]["ls"][index[2.0]] == 0.75
    assert block["rho"]["qubo"][-1] == 0.5  # the plateau: the share of instances where the median is observed
    assert block["rho_inf"] == {"exact": 0.75, "qubo": 0.5, "ls": 0.75}


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def test_the_reference_and_the_pi_plus_of_the_secondary_analyses_are_those_of_h2():
    rows, certs, report = two_cells()
    both = sec.secondary_report(rows, [report], references=certs)
    h2 = an.h2_report(rows, [report], references=certs)
    assert len(both["instances"]) == len(h2["instances"]) == 5
    for mine, theirs in zip(both["instances"], h2["instances"], strict=True):
        assert mine["instance_id"] == theirs["instance_id"] and mine["unit"] == theirs["unit"]
        assert mine["f_ref"] == theirs["f_ref"] and mine["ref_type"] == theirs["ref_type"]
        assert mine["pi_plus"] == theirs["pi"] and mine["g_start"] == theirs["g_start"]
    assert both["references"]["by_type"] == h2["references"]["by_type"] == {"certified": 5}
    assert both["descriptive"] == h2["descriptive"]


def test_the_report_is_flagged_secondary_and_records_every_choice():
    rows, certs, report = two_cells()
    out = sec.secondary_report(rows, [report], references=certs)
    assert out["kind"] == "e2-secondary" and out["role"] == "secondary" and out["confirmatory"] is False
    assert out["notice"].startswith("SECONDARY") and "not confirmatory" in out["notice"].lower()
    assert out["settings"] == {
        "budget_s": 60.0, "tau": 1e-3, "tau_sensitivity": 1e-4, "censoring_budgets_s": [30.0, 10.0],
        "n_boot": 9999, "seed": 60000, "level": 0.95, "family_level": 0.99, "family_size": 2,
        "cell_level": pytest.approx(0.995), "band": [1 / 1.5, 1.5], "interval": "percentile",
        "profile_ratios": list(sec.PROFILE_RATIOS), "prespecified": True}
    for key in ("status", "inputs", "ttt_clock", "ttt_value", "ttt_ratio", "ttt_censoring", "ttt_pooling",
                "ttt_start", "ttt_sensitivity", "exact_times", "profiles", "budget_overrun", "degenerate"):
        assert out["prespecified"][key]["choice"] and out["prespecified"][key]["why"], key
    json.dumps(out, allow_nan=False)  # strict JSON: missing numbers are null
    again = sec.secondary_report(list(reversed(rows)), [report], references=list(reversed(certs)))
    for part in ("ttt", "exact_times", "profiles"):
        assert again[part] == out[part], part


def test_a_non_default_bootstrap_or_budget_is_marked_as_not_prespecified():
    rows, certs, report = two_cells()
    out = sec.secondary_report(rows, [report], references=certs, n_boot=500, seed=1)
    assert out["settings"]["prespecified"] is False and any("prespecified" in w for w in out["warnings"])


def test_rows_over_the_budget_are_used_as_measured_and_counted():
    rows, certs, report = two_cells()
    rows[0]["budget_ok"] = False
    out = sec.secondary_report(rows, [report], references=certs)
    assert out["budget"]["n_rows_over_budget"] == 1 and out["budget"]["n_over_budget"]["C1"]["exact"] == 1
    assert any("overran" in w for w in out["warnings"])


@pytest.mark.parametrize("what", ["unselected", "missing_role", "wrong_budget", "skipped"])
def test_the_structure_of_the_grid_is_checked_as_for_h2(what):
    rows, certs, report = world({"C1": [standard()] * 2})
    if what == "unselected":
        rows += [ttt_row("C1", 2007, role, 20000, 10.0) for role in ("exact_rep", "qubo_rep", "ls")]
        match = "not selected"
    elif what == "missing_role":
        rows = [r for r in rows if r["roles"] != ["ls"]]
        match = "role 'ls'"
    elif what == "wrong_budget":
        rows[0]["budget_s"] = 30.0
        match = "budget_s"
    else:
        rows[0]["skipped"] = "not_binding"
        match = "skipped"
    with pytest.raises(ValueError, match=match):
        sec.secondary_report(rows, [report], references=certs)


# --------------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------------


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(an.registry, "_git_state", lambda: clean_state())


CSV_FILES = ("exact_times.csv", "instances.csv", "profiles.csv", "ttt_cells.csv", "ttt_censoring.csv",
             "ttt_instances.csv", "ttt_pooled.csv")


def run_secondary(tmp_path, rows, certs, report, *, extra=(), name="secondary"):
    bench = write_run(tmp_path, "20261020-100000-E2-bench_factor-abc123", rows)
    screening = write_run(tmp_path, "20261019-100000-E2-elig_test-def456", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / name
    code = an.main(["secondary", "--rows", str(bench), "--eligibility", str(elig_path), "--reference",
                    str(screening), "--out", str(out), *extra])
    return code, out, bench, screening, elig_path


def test_the_command_writes_the_json_and_the_tables_and_flags_every_one_of_them_secondary(tmp_path, clean_tree,
                                                                                         capsys):
    rows, certs, report = two_cells()
    code, out, *_ = run_secondary(tmp_path, rows, certs, report)
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(["secondary.json", *CSV_FILES])
    result = json.loads((out / "secondary.json").read_text(encoding="utf-8"))
    assert result["kind"] == "e2-secondary" and result["role"] == "secondary" and result["confirmatory"] is False
    assert result["provenance_clean"] is True and result["warnings"] == []  # clean, and still not confirmatory
    assert result["ttt"]["primary"]["pooled"]["TTT"]["estimate_log"] == pytest.approx(-1.5 * LN2, abs=1e-12)
    for name in CSV_FILES:
        frame = pd.read_csv(out / name)
        assert len(frame) > 0, name
        assert frame.columns[0] == "analysis" and set(frame["analysis"]) == {"secondary"}, name
    pooled = pd.read_csv(out / "ttt_pooled.csv")
    assert set(pooled["variant"]) == {"primary", "tau_1e-4", "budget_30", "budget_10"}
    assert set(pooled["hypothesis"]) == {"TTT", "TTTb"}
    assert {"estimate_log", "ratio", "ci_low_log", "ci_high_log", "outcome", "cells"} <= set(pooled.columns)
    instances = pd.read_csv(out / "ttt_instances.csv")
    assert len(instances) == 5 * 4 and {"value_s_exact", "status_TTT", "log_ratio_TTTb"} <= set(instances.columns)
    base = pd.read_csv(out / "instances.csv")
    assert len(base) == 5 and {"pi_plus_exact", "t_primal_median_s", "t_cert_median_s", "g_start"} <= set(base.columns)
    profiles = pd.read_csv(out / "profiles.csv")
    assert {"measure", "cell", "role", "ratio", "rho"} <= set(profiles.columns)
    assert set(profiles["measure"]) == {"PI_plus", "TTT"}
    assert set(pd.read_csv(out / "exact_times.csv")["cell"]) == {"C1", "C2"}
    printed = capsys.readouterr().out
    assert "secondary" in printed.lower() and "TTT" in printed and str(out) in printed


def test_the_command_records_the_runs_it_read_with_their_hashes_and_the_state_of_the_code(tmp_path, clean_tree):
    rows, certs, report = two_cells()
    _, out, bench, screening, elig_path = run_secondary(tmp_path, rows, certs, report)
    result = json.loads((out / "secondary.json").read_text(encoding="utf-8"))
    runs = {i["run_id"]: i for i in result["inputs"] if i.get("run_id")}
    assert set(runs) == {bench.name, screening.name}
    assert runs[bench.name]["sha256"] and runs[bench.name]["git_sha"] == "a" * 40
    assert runs[bench.name]["mode"] == "ingested" and runs[bench.name]["seed_role"] == "test"
    assert next(i for i in result["inputs"] if i["source"] == str(elig_path))["sha256"]
    assert result["code"] == {"git_sha": "b" * 40, "git_dirty": False, "branch": "stream/E0-engine"}
    assert set(result["versions"]) == {"python", "numpy", "pandas", "scipy"}


def test_plain_result_files_a_dirty_tree_or_a_rehearsal_are_reported_as_provenance_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(an.registry, "_git_state", lambda: clean_state(dirty=True))
    rows, certs, report = two_cells()
    bench = write_run(tmp_path, "r1", rows, dirty=True, seed_role="calibration", mode="smoke")
    ref = write_run(tmp_path, "r2", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "secondary"
    an.main(["secondary", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(ref),
             "--out", str(out)])
    result = json.loads((out / "secondary.json").read_text(encoding="utf-8"))
    text = " ".join(result["warnings"])
    assert result["provenance_clean"] is False and result["confirmatory"] is False
    assert "git_dirty" in text and "seed_role" in text and "mode" in text


def test_a_screening_round_the_report_was_built_from_must_be_among_the_inputs(tmp_path, clean_tree):
    rows, certs, report = two_cells()
    report["inputs"] = [{"source": "runs/round1", "run_id": "round1", "sha256": "a"},
                        {"source": "runs/round2", "run_id": "round2", "sha256": "b"}]
    bench = write_run(tmp_path, "bench1", rows)
    round1 = write_run(tmp_path, "round1", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "secondary"
    an.main(["secondary", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(round1),
             "--out", str(out)])
    result = json.loads((out / "secondary.json").read_text(encoding="utf-8"))
    assert result["provenance_clean"] is False and any("round2" in w for w in result["warnings"])


def test_the_command_takes_the_dense_report_and_several_run_directories(tmp_path, clean_tree):
    rows, certs, report = two_cells()
    dense = elig({"D1_100": {"selected": [2000, 2001], "enough": False}}, members=[])
    dense_rows, dense_certs, _ = world({"D1_100": [standard(exact=5.0, qubo=10.0)] * 2})
    factor = write_run(tmp_path, "factor", rows)
    dense_run = write_run(tmp_path, "dense", dense_rows)
    ref = write_run(tmp_path, "ref", [*certs, *dense_certs], stage="eligibility")
    test_report = write_json(tmp_path / "eligibility_test.json", report)
    dense_report = write_json(tmp_path / "eligibility_dense.json", dense)
    out = tmp_path / "secondary"
    assert an.main(["secondary", "--rows", str(factor), str(dense_run), "--eligibility", str(test_report),
                    "--eligibility-dense", str(dense_report), "--reference", str(ref), "--out", str(out)]) == 0
    result = json.loads((out / "secondary.json").read_text(encoding="utf-8"))
    d1 = result["ttt"]["primary"]["cells"]["D1_100"]
    assert d1["TTT"]["in_pooled"] is False and d1["TTT"]["estimate_log"] == pytest.approx(LN2)
    assert result["ttt"]["primary"]["pooled"]["TTT"]["cells"] == ["C1", "C2"]
    assert "D1_100" in result["profiles"]["cells"] and "D1_100" in result["exact_times"]["cells"]


def test_the_command_stops_at_the_structure_of_the_grid_before_any_ratio(tmp_path, clean_tree):
    rows, _, report = two_cells()
    bench = write_run(tmp_path, "factor", rows[:-10])
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    with pytest.raises(ValueError, match="role 'ls'"):
        an.main(["secondary", "--rows", str(bench), "--eligibility", str(elig_path),
                 "--out", str(tmp_path / "secondary")])


def test_with_no_selected_instance_the_command_writes_empty_tables_with_their_headers(tmp_path, clean_tree, capsys):
    statement = an.NO_POOLED_STATEMENT
    report = elig({"C1": {"selected": [], "enough": False}}, members=[], statement=statement)
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "secondary"
    assert an.main(["secondary", "--eligibility", str(elig_path), "--out", str(out)]) == 0
    result = json.loads((out / "secondary.json").read_text(encoding="utf-8"))
    assert result["statement"] == statement and result["instances"] == [] and result["confirmatory"] is False
    assert statement in capsys.readouterr().out
    empty = ("instances.csv", "ttt_instances.csv", "ttt_cells.csv", "ttt_pooled.csv", "profiles.csv")
    for name in CSV_FILES:
        frame = pd.read_csv(out / name)
        assert frame.columns[0] == "analysis" and len(frame.columns) > 3, name
        assert (len(frame) == 0) is (name in empty), name  # the cell itself is listed, with no instances
    assert list(pd.read_csv(out / "exact_times.csv")["n_instances"]) == [0]
    assert set(pd.read_csv(out / "ttt_censoring.csv")["n_instances"]) == {0}


def test_the_secondary_command_has_a_help_and_no_knob_for_the_choices(capsys):
    with pytest.raises(SystemExit) as exc:
        an.main(["secondary", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in ("--rows", "--eligibility", "--eligibility-dense", "--reference", "--out"):
        assert option in text
    for knob in ("--seed", "--n-boot", "--level", "--budget", "--tau", "--censor"):
        assert knob not in text  # the targets, the budgets and the bootstrap are prespecified, not options
    assert "secondary" in text.lower() and "confirmatory" in text.lower()


def test_h2_and_h1c_are_still_there():
    with pytest.raises(SystemExit) as exc:
        an.main(["h2", "--help"])
    assert exc.value.code == 0


# --------------------------------------------------------------------------------------------
# The constants of the module
# --------------------------------------------------------------------------------------------


def test_the_prespecified_text_agrees_with_the_constants_it_describes():
    text = " ".join(" ".join(part.values()) for part in sec.PRESPECIFIED.values())
    assert f"B = {rq2.N_BOOT}" in text and f"seed {rq2.SEED}" in text
    assert "1e-3" in text and "1e-4" in text and "[1/1.5, 1.5]" in text
    assert "30 s" in text and "10 s" in text and "inverted_cdf" in text
    assert sec.TAU == 1e-3 and sec.TAU_SENSITIVITY == 1e-4 and sec.CENSORING_BUDGETS_S == (30.0, 10.0)
    assert sec.BUDGET_S == an.BUDGET_S == 60.0
    assert set(sec.SENSITIVITIES) == {"tau_1e-4", "budget_30", "budget_10"}
    assert all(b < sec.BUDGET_S for b in sec.CENSORING_BUDGETS_S)  # the trajectories end at the budget: no longer one
