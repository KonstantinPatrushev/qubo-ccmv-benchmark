"""E1 pilot (ANALYSIS_PLAN §2 H1a "правило пола", §6 "RQ1 бюджет", §12): code and rules.

Two parts, both on CALIBRATION instances only (pairs 1000-1039, solver seeds 10000-19999):

- ``floor``: the ceiling ``E_form(m)`` of every calibration instance; the share of instances with
  ``E_form(50) > τ`` decides ``m0`` (50 if at least 20 %, else 1);
- ``timing``: the median time of one sampler call over the calibration instances of an N, for a
  grid of candidate works (reads for tabu, one plain search per read: ANALYSIS_PLAN §12 B1); the work
  whose median is closest to 1 s (log scale) is the pilot value.

The rules are pure functions and tested without a sampler; the cells are exercised on tiny
instances; the report is assembled from synthetic rows.
"""

from __future__ import annotations

import copy
import json
import math
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import yaml

from cif.experiments import e1_attribution as e1
from cif.experiments import e1_pilot as pilot
from cif.experiments import run

REPO_ROOT = Path(__file__).resolve().parents[1]
MS = [1, 5, 20, 50, 200]
SCIP = {"formulation": "perspective", "diag": "mineig", "mip_gap": 1e-6, "time_limit_s": 120.0, "seed": 0}


def _cfg(parts=("floor", "timing"), **over) -> dict:
    cfg = {
        "stream": "E1",
        "name": "pilot_test",
        "experiment": "cif.experiments.e1_pilot",
        "seeds": [1000, 1001, 1002, 10000, 10001],
        "inputs": {},
        "params": {
            "seed_role": "calibration",
            "parts": list(parts),
            "instances": [{"n": 12, "pair_seeds": [1000, 1001]}, {"n": 14, "pair_seeds": [1002]}],
            "h_levels": [0.15, 0.45],
            "lam": 5.0, "w_min": 0.0, "w_max": 1.0,
            "ms": MS,
            "scip": SCIP,
            "floor": {"tau": 1e-3, "m_floor": 50, "m_fallback": 1, "min_share": 0.2},
            "timing": {
                "target_s": 1.0, "tolerance": 0.2, "solver_seeds": [10000, 10001], "warmup_calls": 1,
                "samplers": [
                    {"name": "sa", "reference": {"penalty": 100, "beta": "default"},
                     "grid": [{"n": 12, "scale": "num_reads", "values": [10, 20, 40],
                               "fixed": {"num_sweeps": 20}},
                              {"n": 14, "scale": "num_reads", "start": 10, "ratio": 2.0, "count": 2,
                               "fixed": {"num_sweeps": 20}}]},
                    {"name": "tabu", "reference": {"penalty": 100},
                     "grid": [{"n": 12, "scale": "num_reads", "values": [3, 6],
                               "fixed": {"num_restarts": 0, "tenure": 5}},
                              {"n": 14, "scale": "num_reads", "values": [3],
                               "fixed": {"num_restarts": 0, "tenure": 5}}]},
                    {"name": "swap_sa", "reference": {"cold_ratio": 0.01},
                     "grid": [{"n": 12, "scale": "num_reads", "values": [5, 10],
                               "fixed": {"num_sweeps": 8}},
                              {"n": 14, "scale": "num_reads", "values": [5],
                               "fixed": {"num_sweeps": 8}}]},
                ],
            },
        },
    }
    cfg["params"].update(over)
    if set(parts) == {"floor"}:  # a floor-only run uses no solver seeds
        cfg["seeds"] = [1000, 1001, 1002]
    return cfg


@pytest.fixture(autouse=True)
def _fresh_caches():
    e1.clear_caches()
    yield
    e1.clear_caches()


# --------------------------------------------------------------------------------------
# The grid of candidate works
# --------------------------------------------------------------------------------------


def test_geometric_grid_is_rounded_and_strictly_increasing():
    grid = pilot.expand_grid({"scale": "num_reads", "start": 100, "ratio": 1.26, "count": 5})
    assert grid == [100, 126, 159, 200, 252]
    small = pilot.expand_grid({"scale": "num_restarts", "start": 1, "ratio": 1.4, "count": 10})
    assert all(isinstance(v, int) for v in small)
    assert all(b > a for a, b in pairwise(small)) and small[0] == 1
    assert len(small) < 10  # rounding merges the first steps


def test_explicit_values_are_taken_as_they_are():
    assert pilot.expand_grid({"scale": "num_reads", "values": [5, 8, 13]}) == [5, 8, 13]


@pytest.mark.parametrize(
    "entry",
    [
        {"scale": "num_reads"},  # neither
        {"scale": "num_reads", "values": [1, 2], "start": 1, "ratio": 2, "count": 3},  # both
        {"scale": "num_reads", "values": [3, 3]},
        {"scale": "num_reads", "values": [4, 2]},
        {"scale": "num_reads", "values": [0, 2]},
        {"scale": "num_reads", "values": []},
        {"scale": "num_reads", "start": 10, "ratio": 1.0, "count": 4},
        {"scale": "num_reads", "start": 10, "ratio": 0.5, "count": 4},
        {"scale": "num_reads", "start": 0, "ratio": 2.0, "count": 4},
        {"scale": "num_reads", "start": 10, "ratio": 2.0, "count": 0},
        {"start": 10, "ratio": 2.0, "count": 3},  # no scale
    ],
)
def test_malformed_grids_are_refused(entry):
    with pytest.raises(ValueError):
        pilot.expand_grid(entry)


# --------------------------------------------------------------------------------------
# The cells
# --------------------------------------------------------------------------------------


def test_cells_cover_the_calibration_instances_and_the_candidate_works():
    cfg = _cfg()
    cells = pilot.cells(cfg)
    floor = [c for c in cells if c["kind"] == "floor"]
    timing = [c for c in cells if c["kind"] == "timing"]
    assert len(floor) == 3 * 2  # pairs × levels
    assert {(c["n"], c["pair_seed"], c["h"]) for c in floor} == {
        (n, ps, h) for n, ps in ((12, 1000), (12, 1001), (14, 1002)) for h in (0.15, 0.45)}
    assert len(timing) == (3 + 2) + (2 + 1) + (2 + 1)  # grid sizes of sa, tabu, swap_sa at n = 12, 14
    assert len(run.grid(cfg)) == len(cells)  # unique cell keys


def test_timing_cells_state_the_full_work_and_the_reference_configuration():
    timing = [c for c in pilot.cells(_cfg()) if c["kind"] == "timing"]
    sa12 = [c for c in timing if c["sampler"] == "sa" and c["n"] == 12]
    assert [c["num_reads"] for c in sa12] == [10, 20, 40]
    assert all(c["num_sweeps"] == 20 and c["scaled"] == "num_reads" for c in sa12)
    assert all("scale" not in c for c in timing)  # `scale` is the s* of a floor row, not the name of a parameter
    assert all(c["penalty"] == 100.0 and c["beta"] == "default" for c in sa12)
    tabu = next(c for c in timing if c["sampler"] == "tabu" and c["n"] == 12)
    assert (tabu["num_reads"], tabu["tenure"], tabu["num_restarts"]) == (3, 5, 0)
    assert tabu["scaled"] == "num_reads"  # B1: the work unit of tabu is reads
    assert tabu["penalty"] == 100.0 and tabu["beta"] is None
    swap = next(c for c in timing if c["sampler"] == "swap_sa")
    assert swap["penalty"] is None and swap["beta"] is None and swap["num_sweeps"] == 8
    assert swap["cold_ratio"] == 0.01  # §12 B3: timed at the reference schedule
    assert all(c["cold_ratio"] is None for c in timing if c["sampler"] != "swap_sa")


def test_parts_select_the_cells():
    assert {c["kind"] for c in pilot.cells(_cfg(parts=("floor",)))} == {"floor"}
    assert {c["kind"] for c in pilot.cells(_cfg(parts=("timing",)))} == {"timing"}
    with pytest.raises(ValueError, match="parts"):
        pilot.cells(_cfg(parts=("floor", "power")))
    with pytest.raises(ValueError, match="parts"):
        pilot.cells(_cfg(parts=()))


def test_the_pilot_runs_on_calibration_seeds_only():
    cfg = _cfg(seed_role="test")
    cfg["seeds"] = [2000, 2001, 2002, 20000, 20001]
    cfg["params"]["instances"] = [{"n": 12, "pair_seeds": [2000, 2001]}, {"n": 14, "pair_seeds": [2002]}]
    cfg["params"]["timing"]["solver_seeds"] = [20000, 20001]
    with pytest.raises(ValueError, match="calibration"):
        pilot.cells(cfg)


@pytest.mark.parametrize(
    ("pair_seeds", "solver_seeds"),
    [([2000], [10000]), ([1000], [20000]), ([999], [10000]), ([1000], [9999])],
)
def test_seeds_outside_the_calibration_ranges_are_refused(pair_seeds, solver_seeds):
    cfg = _cfg()
    cfg["params"]["instances"] = [{"n": 12, "pair_seeds": pair_seeds}]
    cfg["params"]["timing"]["solver_seeds"] = solver_seeds
    for sampler in cfg["params"]["timing"]["samplers"]:
        sampler["grid"] = [g for g in sampler["grid"] if g["n"] == 12]
    cfg["seeds"] = sorted({*pair_seeds, *solver_seeds})
    with pytest.raises(ValueError, match="seed"):
        pilot.cells(cfg)


def test_declared_seeds_must_be_the_seeds_of_the_grid():
    cfg = _cfg()
    cfg["seeds"] = [1000, 10000]
    with pytest.raises(ValueError, match="seeds"):
        pilot.cells(cfg)


def test_a_floor_only_config_declares_no_solver_seeds():
    cfg = _cfg(parts=("floor",))
    assert pilot.cells(cfg)
    cfg["seeds"] = [1000, 1001, 1002, 10000, 10001]
    with pytest.raises(ValueError, match="seeds"):
        pilot.cells(cfg)


def test_floor_parameters_are_checked():
    for bad in ({"tau": 0.0, "m_floor": 50, "m_fallback": 1, "min_share": 0.2},
                {"tau": 1e-3, "m_floor": 7, "m_fallback": 1, "min_share": 0.2},
                {"tau": 1e-3, "m_floor": 50, "m_fallback": 9, "min_share": 0.2},
                {"tau": 1e-3, "m_floor": 50, "m_fallback": 1, "min_share": 0.0},
                {"tau": 1e-3, "m_floor": 50, "m_fallback": 1, "min_share": 1.5},
                {"tau": 1e-3, "m_floor": 50, "m_fallback": 1}):
        with pytest.raises(ValueError, match="floor"):
            pilot.cells(_cfg(floor=bad))


def test_timing_parameters_are_checked():
    for key, bad in (("target_s", 0.0), ("target_s", -1.0), ("tolerance", 0.0), ("tolerance", 1.5),
                     ("solver_seeds", []), ("warmup_calls", -1)):
        cfg = _cfg()
        cfg["params"]["timing"][key] = bad
        with pytest.raises(ValueError, match="timing"):
            pilot.cells(cfg)


def test_every_sampler_and_n_needs_a_grid():
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][0]["grid"].pop()  # sa has no grid at n = 14
    with pytest.raises(ValueError, match=r"sa.*n=14"):
        pilot.cells(cfg)


def test_the_scale_must_be_a_batch_key_of_the_sampler_and_not_fixed():
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][0]["grid"][0]["scale"] = "num_restarts"  # not an SA key
    with pytest.raises(ValueError, match="num_restarts"):
        pilot.cells(cfg)
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][0]["grid"][0]["fixed"]["num_reads"] = 5  # the scale, fixed
    with pytest.raises(ValueError, match="num_reads"):
        pilot.cells(cfg)
    cfg = _cfg()
    del cfg["params"]["timing"]["samplers"][0]["grid"][0]["fixed"]["num_sweeps"]  # a batch key missing
    with pytest.raises(ValueError, match="num_sweeps"):
        pilot.cells(cfg)


def test_the_work_unit_of_tabu_is_reads_each_one_plain_search():
    """§12 B1: the scaled parameter of tabu is ``num_reads``; ``num_restarts`` is fixed at 0."""
    cfg = _cfg()
    tabu = cfg["params"]["timing"]["samplers"][1]
    tabu["grid"][0].update(scale="num_restarts", values=[1, 2],
                           fixed={"num_reads": 3, "tenure": 5})  # the old regime: restarts are scaled
    with pytest.raises(ValueError, match=r"tabu.*n=12.*num_reads.*B1"):
        pilot.cells(cfg)
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][1]["grid"][0]["fixed"]["num_restarts"] = 2
    with pytest.raises(ValueError, match=r"tabu.*n=12.*num_restarts.*B1"):
        pilot.cells(cfg)


@pytest.mark.parametrize("bad", [1, 4, 6, 20])
def test_the_tenure_of_the_tabu_grid_is_the_rule_of_the_plan(bad):
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][1]["grid"][1]["fixed"]["tenure"] = bad  # n = 14: max(5, 14 // 4) = 5
    with pytest.raises(ValueError, match=r"tabu.*n=14.*tenure"):
        pilot.cells(cfg)


def test_zero_restarts_do_not_make_other_fixed_values_zero_too():
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][0]["grid"][0]["fixed"]["num_sweeps"] = 0  # SA: still refused
    with pytest.raises(ValueError, match="num_sweeps"):
        pilot.cells(cfg)


def test_the_reference_configuration_is_checked():
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][0]["reference"] = {"penalty": 100, "beta": "warm"}
    with pytest.raises(ValueError, match="beta"):
        pilot.cells(cfg)
    cfg = _cfg()
    del cfg["params"]["timing"]["samplers"][1]["reference"]  # tabu needs a penalty
    with pytest.raises(ValueError, match="reference"):
        pilot.cells(cfg)
    cfg = _cfg()
    del cfg["params"]["timing"]["samplers"][2]["reference"]  # swap-SA needs its cold ratio (§12 B3)
    with pytest.raises(ValueError, match=r"swap_sa.*cold_ratio"):
        pilot.cells(cfg)


@pytest.mark.parametrize(
    ("index", "reference"),
    [(0, {"penalty": 30, "beta": "default"}), (0, {"penalty": 100, "beta": "objective"}),
     (1, {"penalty": 10}), (2, {"cold_ratio": 0.2}), (2, {"cold_ratio": 0.1})],
)
def test_the_timed_configuration_is_the_reference_of_the_plan(index, reference):
    """§12 B3: the work is sized on the reference, not on any configuration of the grid."""
    cfg = _cfg()
    cfg["params"]["timing"]["samplers"][index]["reference"] = reference
    with pytest.raises(ValueError, match=r"reference.*B3"):
        pilot.cells(cfg)


def test_cells_do_not_mutate_the_config():
    cfg = _cfg()
    before = copy.deepcopy(cfg)
    pilot.cells(cfg)
    assert cfg == before


# --------------------------------------------------------------------------------------
# Choosing the work: the rule of §6
# --------------------------------------------------------------------------------------


def _timing_rows(times, *, sampler="sa", n=16, scale="num_reads", base=100):
    return [
        {"kind": "timing", "sampler": sampler, "n": n, "scaled": scale, scale: base * (i + 1),
         "num_sweeps": 1000, "call_s_median": t, "call_s_q25": 0.9 * t, "call_s_q75": 1.1 * t}
        for i, t in enumerate(times)
    ]


def test_the_closest_median_on_the_log_scale_wins():
    rows = _timing_rows([0.2, 0.5, 0.9, 1.3, 2.0])
    choice = pilot.choose_work(rows, target_s=1.0, tolerance=0.2)
    assert choice["chosen"]["num_reads"] == 300 and choice["status"] == "ok"
    assert choice["median_call_s"] == pytest.approx(0.9)
    assert choice["ratio_to_target"] == pytest.approx(0.9)


def test_the_log_scale_makes_half_and_double_equally_far():
    """|ln 0.5| = |ln 2|: a tie on the log scale goes to the smaller work."""
    choice = pilot.choose_work(_timing_rows([0.5, 2.0]), target_s=1.0, tolerance=0.2)
    assert choice["chosen"]["num_reads"] == 100
    # 0.55 s is closer to 1 s than 1.5 s on the linear scale (0.45 against 0.5), 1.5 s is closer
    # on the log scale (ln 1.5 = 0.41, |ln 0.55| = 0.60): the rule is on the log scale
    choice = pilot.choose_work(_timing_rows([0.55, 1.5]), target_s=1.0, tolerance=0.5)
    assert choice["chosen"]["num_reads"] == 200


def test_candidates_are_ordered_by_work_whatever_the_order_of_the_rows():
    rows = _timing_rows([0.2, 0.5, 0.9, 1.3, 2.0])
    shuffled = [rows[i] for i in (3, 0, 4, 2, 1)]
    a = pilot.choose_work(rows, target_s=1.0, tolerance=0.2)
    b = pilot.choose_work(shuffled, target_s=1.0, tolerance=0.2)
    assert a == b
    assert [c["work"]["num_reads"] for c in a["candidates"]] == [100, 200, 300, 400, 500]


@pytest.mark.parametrize(
    ("times", "status"),
    [
        ([0.9, 1.1, 1.6], "ok"),
        ([0.5, 1.5, 2.5], "off_target"),  # the best is inside the grid but outside the tolerance
        ([1.3, 1.8, 2.5], "edge_low"),  # even the smallest work takes too long
        ([0.2, 0.4, 0.7], "edge_high"),  # even the largest work is too quick
        ([0.5, 0.8, 1.9], "off_target"),  # 0.8 is 20 % under the target, the tolerance is 15 %
    ],
)
def test_status_reports_whether_the_target_is_bracketed(times, status):
    choice = pilot.choose_work(_timing_rows(times), target_s=1.0, tolerance=0.15)
    assert choice["status"] == status


def test_the_tolerance_is_relative_to_the_target():
    ok = pilot.choose_work(_timing_rows([0.5, 0.8, 2.0]), target_s=1.0, tolerance=0.25)
    assert ok["status"] == "ok" and ok["chosen"]["num_reads"] == 200
    tight = pilot.choose_work(_timing_rows([0.5, 0.8, 2.0]), target_s=1.0, tolerance=0.1)
    assert tight["status"] == "off_target"


def test_a_single_candidate_cannot_bracket_the_target():
    assert pilot.choose_work(_timing_rows([1.0]), target_s=1.0, tolerance=0.2)["status"] == "ok"
    assert pilot.choose_work(_timing_rows([3.0]), target_s=1.0, tolerance=0.2)["status"] == "edge_low"
    assert pilot.choose_work(_timing_rows([0.3]), target_s=1.0, tolerance=0.2)["status"] == "edge_high"


def test_the_chosen_work_carries_the_fixed_keys_too():
    rows = _timing_rows([0.9, 1.1], sampler="swap_sa")
    choice = pilot.choose_work(rows, target_s=1.0, tolerance=0.2)
    assert choice["chosen"] == {"num_reads": 200, "num_sweeps": 1000}  # 1.1 s is closer than 0.9 s


def test_choose_work_refuses_empty_input_and_bad_numbers():
    with pytest.raises(ValueError):
        pilot.choose_work([], target_s=1.0, tolerance=0.2)
    with pytest.raises(ValueError, match="duplicate"):
        pilot.choose_work(_timing_rows([1.0, 1.0]) + _timing_rows([1.0])[:1], target_s=1.0, tolerance=0.2)
    with pytest.raises(ValueError):
        pilot.choose_work(_timing_rows([float("nan"), 1.0]), target_s=1.0, tolerance=0.2)
    with pytest.raises(ValueError):
        pilot.choose_work(_timing_rows([1.0]), target_s=0.0, tolerance=0.2)


# --------------------------------------------------------------------------------------
# The floor rule: m0
# --------------------------------------------------------------------------------------


def _floor_rows(n_exceed: int, n_total: int = 80, *, tau=1e-3, m=50):
    """Rows of ``n_total`` instances, the first ``n_exceed`` with ``E_form(m)`` above ``tau``."""
    rows = []
    for i in range(n_total):
        n = 16 if i < n_total // 2 else 24
        rows.append({"kind": "floor", "n": n, "pair_seed": 1000 + i // 2, "h": 0.15 if i % 2 == 0 else 0.45,
                     "E_form_1": 5 * tau, f"E_form_{m}": 3 * tau if i < n_exceed else 0.0,
                     "status": "ok"})
    return rows


@pytest.mark.parametrize(("n_exceed", "m0"), [(0, 1), (15, 1), (16, 50), (17, 50), (80, 50)])
def test_m0_is_50_from_a_share_of_20_percent_inclusive(n_exceed, m0):
    res = pilot.floor_rule(_floor_rows(n_exceed), tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)
    assert res["m0"] == m0
    assert res["n_instances"] == 80 and res["n_exceed"] == n_exceed
    assert res["share"] == pytest.approx(n_exceed / 80)


def test_exceeding_tau_is_strict():
    rows = _floor_rows(80)
    for r in rows:
        r["E_form_50"] = 1e-3  # exactly tau: not above it
    res = pilot.floor_rule(rows, tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)
    assert res["n_exceed"] == 0 and res["m0"] == 1


def test_the_floor_rule_reports_intervals_and_breakdowns():
    rows = _floor_rows(20)
    res = pilot.floor_rule(rows, tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)
    lo, hi = res["wilson_95"]
    assert lo < res["share"] < hi
    assert set(res["by_n"]) == {"16", "24"} and set(res["by_h"]) == {"0.15", "0.45"}
    assert res["by_n"]["16"]["n"] == 40 and res["by_n"]["16"]["n_exceed"] == 20
    assert res["by_n"]["24"]["n_exceed"] == 0
    assert res["by_h"]["0.15"]["n"] == 40
    assert res["share_fallback"] == 1.0  # E_form(1) = 5 tau everywhere in the fixture
    assert res["rule"]


def test_the_floor_rule_reports_the_share_at_every_pool_size_of_the_rows():
    """Descriptive: where the threshold bites, for all m that the rows hold."""
    rows = _floor_rows(20)
    for r in rows:
        r["E_form_5"] = 2e-3 if r["h"] == 0.45 else 5e-4
    res = pilot.floor_rule(rows, tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)
    assert list(res["by_m"]) == ["1", "5", "50"]
    assert res["by_m"]["50"]["n_exceed"] == 20 and res["by_m"]["1"]["share"] == 1.0
    assert res["by_m"]["5"]["n_exceed"] == 40 and res["by_m"]["5"]["n"] == 80


def test_the_floor_rule_pools_the_instances_of_all_n_and_levels():
    """Not per N: one m0 for the whole design (H1a is stratified by N, m0 is not)."""
    rows = _floor_rows(0)
    for r in rows[:16]:  # 16 of 80 instances exceed, all in one N and one level
        r["E_form_50"] = 1e-2
    res = pilot.floor_rule(rows, tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)
    assert res["m0"] == 50


def test_the_floor_rule_refuses_missing_values():
    rows = _floor_rows(10)
    rows[3]["E_form_50"] = None
    with pytest.raises(ValueError, match="E_form_50"):
        pilot.floor_rule(rows, tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)
    with pytest.raises(ValueError):
        pilot.floor_rule([], tau=1e-3, m_floor=50, m_fallback=1, min_share=0.2)


# --------------------------------------------------------------------------------------
# The timing environment
# --------------------------------------------------------------------------------------


def test_timing_environment_has_the_protocol_fields():
    env = pilot.timing_environment()
    for key in ("host", "cpu_model", "governor", "cpu_count_logical", "cpu_cores_physical", "loadavg",
                "procs_running", "cpu_mhz", "threads", "scaling_driver", "no_turbo", "kernel"):
        assert key in env, key
    assert len(env["loadavg"]) == 3 and all(x >= 0.0 for x in env["loadavg"])
    assert env["cpu_count_logical"] == os.cpu_count()
    json.dumps(env, allow_nan=False)


def test_timing_environment_survives_a_machine_without_sysfs_and_loadavg(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("not available")

    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(os, "getloadavg", refuse)
    env = pilot.timing_environment()
    assert env["loadavg"] is None and env["governor"] is None and env["procs_running"] is None
    assert env["cpu_mhz"] is None
    json.dumps(env, allow_nan=False)


# --------------------------------------------------------------------------------------
# Cells on tiny instances
# --------------------------------------------------------------------------------------


def _cell(cfg, **where):
    for cell in pilot.cells(cfg):
        if all(cell.get(k) == v for k, v in where.items()):
            return cell
    raise AssertionError(f"no cell with {where}")


def test_a_floor_cell_reports_the_ceiling_of_the_instance():
    cfg = _cfg()
    cell = _cell(cfg, kind="floor", n=12, pair_seed=1000, h=0.45)
    row = pilot.run_cell(cell, cfg)
    assert row["status"] == "ok" and row["K"] >= 3 and row["scale"] > 0.0
    assert row["ref_type"] == "certified" and row["n_subsets"] == math.comb(12, row["K"])
    form = [row[f"E_form_{m}"] for m in MS]
    assert all(v >= -1e-9 for v in form) and all(b <= a + 1e-15 for a, b in pairwise(form))
    # the same numbers as the E1 oracle of the instance
    key = (12, 1000, 0.45, (0.15, 0.45), 5.0, 0.0, 1.0, tuple(MS), tuple(sorted(SCIP.items())))
    inst = e1._instance(*key)
    for j, m in enumerate(MS):
        assert row[f"E_form_{m}"] == pytest.approx(float(inst.oracle.e_form[j]), abs=0, rel=0)
        assert row[f"n_form_{m}"] == int(inst.oracle.n_form[j])
    json.dumps(row, allow_nan=False)


def test_a_timing_cell_reports_the_median_call_over_the_calibration_instances():
    cfg = _cfg()
    cell = _cell(cfg, kind="timing", sampler="sa", n=12, num_reads=10)
    row = pilot.run_cell(cell, cfg)
    assert row["status"] == "ok"
    assert row["n_instances"] == 4 and row["repeats"] == 2  # 2 pairs × 2 levels, 2 solver seeds
    assert len(row["call_s_by_instance"]) == 4
    assert row["call_s_median"] == pytest.approx(float(np.median(row["call_s_by_instance"])))
    assert row["call_s_q25"] <= row["call_s_median"] <= row["call_s_q75"]
    assert row["call_s_min"] <= row["call_s_median"] <= row["call_s_max"] and row["call_s_min"] > 0.0
    assert row["build_s_median"] >= 0.0 and row["cpu_over_wall_median"] > 0.0
    assert row["n_unique_median"] >= 1
    assert row["env_start"]["loadavg"] is not None and "loadavg" in row["env_end"]
    assert row["warmup_calls"] == 1
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("sampler", ["tabu", "swap_sa"])
def test_every_sampler_can_be_timed(sampler):
    cfg = _cfg()
    cell = _cell(cfg, kind="timing", sampler=sampler, n=12)
    row = {**cell, **pilot.run_cell(cell, cfg)}  # what the runner stores
    assert row["call_s_median"] > 0.0 and row["sampler"] == sampler


def test_a_timing_cell_uses_the_calibration_instances_of_its_n_only():
    cfg = _cfg()
    cell = _cell(cfg, kind="timing", sampler="swap_sa", n=14)
    row = pilot.run_cell(cell, cfg)
    assert row["n_instances"] == 2  # one pair at n = 14
    assert row["pair_seeds"] == [1002]


def test_timing_uses_the_same_code_path_as_the_attribution(monkeypatch):
    """The time of a call is that of ``e1_attribution._draw_pool``: the same BQM, sampler and seed."""
    calls = []
    real = e1._draw_pool

    def spy(problem, k, cell, spec, seed, rng):
        calls.append((cell["sampler"], cell["solver_seed"], cell["penalty"], cell["beta"], seed))
        return real(problem, k, cell, spec, seed, rng)

    monkeypatch.setattr(e1, "_draw_pool", spy)
    cfg = _cfg()
    pilot.run_cell(_cell(cfg, kind="timing", sampler="sa", n=12, num_reads=10), cfg)
    # 1 warm-up + 4 instances × 2 repeats
    assert len(calls) == 1 + 8
    assert {c[:4] for c in calls} == {("sa", 10000, 100.0, "default"), ("sa", 10001, 100.0, "default")}
    seeds = [c[4] for c in calls[1:]]
    assert all(0 <= s < 2**31 for s in seeds) and len(set(seeds)) == len(seeds)


def test_swap_sa_is_timed_at_the_reference_cold_ratio(monkeypatch):
    """§12 B3: the work of a swap-SA call is sized on the schedule ``T_cold = 0.01 · T_hot``; the warmer
    exploratory schedule runs the same work and reports its own time."""
    seen = []
    real = e1._draw_pool

    def spy(problem, k, cell, spec, seed, rng):
        seen.append((cell["sampler"], cell["cold_ratio"]))
        return real(problem, k, cell, spec, seed, rng)

    monkeypatch.setattr(e1, "_draw_pool", spy)
    cfg = _cfg()
    pilot.run_cell(_cell(cfg, kind="timing", sampler="swap_sa", n=12, num_reads=5), cfg)
    assert seen and set(seen) == {("swap_sa", 0.01)}


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def _synthetic_run(cfg, *, mode="full", git_dirty=False, tweak=None):
    """Rows for every cell of the grid with made-up results: floor cells E_form from the table
    below, timing cells a call time linear in the work (1 ms per read or restart)."""
    rows = []
    for i, cell in enumerate(pilot.cells(cfg)):
        row = {"cell_key": run.cell_key(cell), **cell, "host": "server1", "status": "ok"}
        if cell["kind"] == "floor":
            exceed = (i % 2 == 0)  # every second instance has a high ceiling
            row.update({f"E_form_{m}": (5e-3 if exceed else 0.0) for m in MS})
            row.update({"E_form_1": 2e-2, "K": 4, "s_cont": 9, "binding": True, "ref_type": "certified"})
        else:
            work = cell[cell["scaled"]]
            base = {"sa": 1.0e-3, "tabu": 2.0e-3, "swap_sa": 1.5e-3}[cell["sampler"]]
            t = base * work * (1.0 if cell["n"] == 12 else 2.0)
            row.update({"call_s_median": t, "call_s_q25": 0.9 * t, "call_s_q75": 1.1 * t,
                        "call_s_min": 0.8 * t, "call_s_max": 1.2 * t, "n_instances": 4, "repeats": 2,
                        "cpu_over_wall_median": 1.0, "build_s_median": 0.001,
                        "env_start": {"host": "server1", "cpu_model": "TestCPU", "governor": "performance",
                                      "cpu_count_logical": 64, "cpu_cores_physical": 32,
                                      "loadavg": [8.0, 7.0, 6.0], "procs_running": 9,
                                      "scaling_driver": "acpi-cpufreq", "no_turbo": None,
                                      "cpu_mhz": {"min": 2900.0, "mean": 3000.0, "max": 3100.0}},
                        "env_end": {"loadavg": [9.0, 7.5, 6.1], "procs_running": 10,
                                    "cpu_mhz": {"min": 2900.0, "mean": 3000.0, "max": 3100.0}}})
        if tweak:
            tweak(cell, row)
        rows.append(row)
    manifest = {"run_id": "20261101-000000-E1-pilot_test-abc123", "name": "pilot_test", "mode": mode,
                "git_sha": "deadbeef", "git_dirty": git_dirty, "config_hash": "abc123def456",
                "host": "server1", "cpu": "TestCPU", "cpu_governor": "performance",
                "started_at": "2026-11-01T00:00:00+00:00", "finished_at": "2026-11-01T01:00:00+00:00",
                "n_errors": 0, "status": "completed"}
    return pilot.RunData(manifest=manifest, config=run.effective_config(cfg), rows=rows)


def _bracketing_cfg(**kw) -> dict:
    """Grids that bracket 1 s for the synthetic timing of ``_synthetic_run`` (1 ms per read for SA,
    2 ms per read for tabu, 1.5 ms per read for swap-SA, all twice as slow at n = 14)."""
    cfg = _cfg(**kw)
    grids = {"sa": {12: [500, 1000, 2000], 14: [250, 500, 1000]},
             "tabu": {12: [250, 500, 1000], 14: [125, 250, 500]},
             "swap_sa": {12: [333, 667, 1333], 14: [166, 333, 666]}}
    for spec in cfg["params"]["timing"]["samplers"]:
        for g in spec["grid"]:
            for key in ("start", "ratio", "count"):
                g.pop(key, None)
            g["values"] = grids[spec["name"]][g["n"]]
    return cfg


def test_the_report_chooses_the_work_and_m0():
    report = pilot.build_report([_synthetic_run(_bracketing_cfg())])
    assert report["valid_for_section_12"] is True and report["problems"] == []
    work = {n: {s: c["chosen"] for s, c in per.items()} for n, per in report["work"].items()}
    assert work["12"]["sa"] == {"num_reads": 1000, "num_sweeps": 20}  # 1 ms per read: 1 s
    assert work["14"]["sa"] == {"num_reads": 500, "num_sweeps": 20}  # twice as slow at n = 14
    assert work["12"]["tabu"] == {"num_reads": 500, "num_restarts": 0, "tenure": 5}
    assert work["14"]["swap_sa"] == {"num_reads": 333, "num_sweeps": 8}
    assert all(c["status"] == "ok" for per in report["work"].values() for c in per.values())
    assert report["floor"]["m0"] == 50 and report["floor"]["n_instances"] == 6
    assert report["plan"]["target_s"] == 1.0 and report["plan"]["tau"] == 1e-3
    assert report["runs"][0]["run_id"].endswith("abc123") and report["runs"][0]["git_sha"] == "deadbeef"
    json.dumps(report, allow_nan=False)


def test_the_pilot_values_are_the_block_of_the_confirmatory_config():
    report = pilot.build_report([_synthetic_run(_bracketing_cfg())])
    values = report["pilot_values"]
    assert values == pilot.pilot_values(report)
    assert values["m0"] == 50
    assert [(w["sampler"], w["n"]) for w in values["work"]] == [
        ("sa", 12), ("sa", 14), ("tabu", 12), ("tabu", 14), ("swap_sa", 12), ("swap_sa", 14)]
    assert values["work"][0] == {"sampler": "sa", "n": 12, "num_reads": 1000, "num_sweeps": 20}
    assert values["source"]["run_ids"] == [report["runs"][0]["run_id"]]
    assert values["source"]["valid_for_section_12"] is True
    # the block must satisfy the grid of the attribution: same samplers, same N
    block = {"sampler_names": {w["sampler"] for w in values["work"]}, "ns": {w["n"] for w in values["work"]}}
    assert block == {"sampler_names": {"sa", "tabu", "swap_sa"}, "ns": {12, 14}}


def test_a_valid_report_carries_the_entry_for_section_12():
    """The values go to §12 in its format (дата — что изменено — почему — гипотезы — кто согласовал)."""
    report = pilot.build_report([_synthetic_run(_bracketing_cfg())])
    text = report["section_12"]
    assert text == pilot.section_12_entry(report)
    assert text.startswith("<дата> — ") and text.endswith("— <кто согласовал>")
    assert text.count(" — ") == 4  # five fields
    for needle in ("m₀ = 50", "3 из 6", "N = 12", "N = 14", "SA (num_reads = 1000, num_sweeps = 20)",
                   "tabu (num_reads = 500, num_restarts = 0, tenure = 5)",
                   "swap-SA (num_reads = 667, num_sweeps = 8)", "TestCPU", "performance", "server1",
                   "20261101-000000-E1-pilot_test-abc123", "deadbeef", "H1a", "§6"):
        assert needle in text, needle


def test_the_code_of_an_ingested_run_is_its_bundle_sha_not_the_sha_of_the_ingesting_tree():
    """``ingest`` records the local HEAD as ``git_sha``; the code that computed the rows is ``bundle_sha``."""
    data = _synthetic_run(_bracketing_cfg(), mode="ingested")
    data.manifest = {**data.manifest, "bundle_sha": "cafe1234"}
    report = pilot.build_report([data])
    assert report["runs"][0]["bundle_sha"] == "cafe1234"
    assert "cafe1234" in report["section_12"] and "deadbeef" not in report["section_12"]
    assert report["pilot_values"]["source"]["git_sha"] == ["cafe1234"]


def test_an_invalid_report_has_no_entry_for_section_12():
    report = pilot.build_report([_synthetic_run(_cfg())])
    assert report["section_12"] is None
    with pytest.raises(ValueError, match="section 12"):
        pilot.section_12_entry(report)


def test_the_report_records_the_timing_environment():
    report = pilot.build_report([_synthetic_run(_bracketing_cfg())])
    env = report["timing_environment"]
    assert env["hosts"] == ["server1"] and env["cpu_models"] == ["TestCPU"]
    assert env["governors"] == ["performance"] and env["cpu_cores_physical"] == [32]
    assert env["loadavg_1m"]["min"] == 8.0 and env["loadavg_1m"]["max"] == 9.0
    assert env["procs_running"]["max"] == 10
    assert env["manifest"]["cpu"] == ["TestCPU"] and env["manifest"]["cpu_governor"] == ["performance"]
    assert env["warnings"] == []


def test_the_report_warns_about_a_governor_a_busy_machine_and_hidden_threads():
    def tweak(cell, row):
        if cell["kind"] == "timing":
            row["env_start"] = {**row["env_start"], "governor": "powersave", "loadavg": [40.0, 30.0, 20.0],
                                "procs_running": 40}
            row["cpu_over_wall_median"] = 1.8

    report = pilot.build_report([_synthetic_run(_bracketing_cfg(), tweak=tweak)])
    text = " ".join(report["timing_environment"]["warnings"])
    assert "powersave" in text and "load" in text and "threads" in text
    assert report["valid_for_section_12"] is True  # warnings do not invalidate the pilot


def test_a_smoke_or_dirty_run_is_not_a_pilot():
    smoke = pilot.build_report([_synthetic_run(_bracketing_cfg(), mode="smoke")])
    assert smoke["valid_for_section_12"] is False and any("smoke" in p for p in smoke["problems"])
    dirty = pilot.build_report([_synthetic_run(_bracketing_cfg(), git_dirty=True)])
    assert dirty["valid_for_section_12"] is False and any("dirty" in p for p in dirty["problems"])
    unknown = pilot.build_report([_synthetic_run(_bracketing_cfg(), git_dirty=None)])
    assert unknown["valid_for_section_12"] is False
    ingested = pilot.build_report([_synthetic_run(_bracketing_cfg(), mode="ingested")])
    assert ingested["valid_for_section_12"] is True


def test_a_report_needs_every_cell_without_errors():
    data = _synthetic_run(_cfg())
    short = pilot.RunData(data.manifest, data.config, data.rows[:-1])
    with pytest.raises(ValueError, match="missing"):
        pilot.build_report([short])
    broken = copy.deepcopy(data)
    broken.rows[0]["error"] = "RuntimeError: boom"
    with pytest.raises(ValueError, match="error"):
        pilot.build_report([broken])
    foreign = copy.deepcopy(data)
    foreign.rows[0]["cell_key"] = "0" * 12
    with pytest.raises(ValueError, match=r"(foreign|unexpected|missing)"):
        pilot.build_report([foreign])


def test_a_report_from_two_runs_combines_floor_and_timing():
    floor = _synthetic_run(_cfg(parts=("floor",)))
    timing_cfg = _cfg(parts=("timing",))
    timing = _synthetic_run(timing_cfg)
    timing.manifest = {**timing.manifest, "run_id": "20261101-020000-E1-pilot_timing-fff000"}
    report = pilot.build_report([floor, timing])
    assert [r["run_id"][-6:] for r in report["runs"]] == ["abc123", "fff000"]
    assert "m0" in report["floor"] and report["work"]["12"]["tabu"]["status"] in {"ok", "edge_high", "edge_low", "off_target"}


def test_floor_and_timing_runs_must_use_the_same_calibration_instances():
    floor = _synthetic_run(_cfg(parts=("floor",)))
    other = _cfg(parts=("timing",))
    other["params"]["instances"] = [{"n": 12, "pair_seeds": [1000, 1001]}, {"n": 14, "pair_seeds": [1010]}]
    other["seeds"] = [1000, 1001, 1010, 10000, 10001]
    timing = _synthetic_run(other)
    report = pilot.build_report([floor, timing])
    assert report["valid_for_section_12"] is False
    assert any("calibration" in p and "instances" in p for p in report["problems"])
    same = pilot.build_report([_synthetic_run(_cfg(parts=("floor",))), _synthetic_run(_cfg(parts=("timing",)))])
    assert not any("calibration" in p for p in same["problems"])


def test_a_report_without_a_part_says_so():
    only_floor = pilot.build_report([_synthetic_run(_cfg(parts=("floor",)))])
    assert only_floor["work"] is None and only_floor["valid_for_section_12"] is False
    assert any("timing" in p for p in only_floor["problems"])
    only_timing = pilot.build_report([_synthetic_run(_cfg(parts=("timing",)))])
    assert only_timing["floor"] is None and only_timing["valid_for_section_12"] is False
    assert any("floor" in p for p in only_timing["problems"])


def test_a_pilot_that_did_not_bracket_the_target_is_not_valid_for_section_12():
    cfg = _cfg()  # grids are far from 1 s: the synthetic timing is 10-40 ms
    report = pilot.build_report([_synthetic_run(cfg)])
    assert report["valid_for_section_12"] is False
    assert any("edge_high" in p or "extend" in p for p in report["problems"])
    with pytest.raises(ValueError, match="pilot values"):
        pilot.pilot_values(report)


def test_pilot_values_of_an_invalid_report_can_be_forced_for_the_machinery_and_say_so():
    report = pilot.build_report([_synthetic_run(_cfg())])
    values = pilot.pilot_values(report, allow_invalid=True)
    assert values["source"]["valid_for_section_12"] is False and values["m0"] == 50
    assert len(values["work"]) == 6


FLOOR_RUN = REPO_ROOT / "runs" / "20261011-205903-E1-pilot_floor-9da201"


@pytest.mark.skipif(not (FLOOR_RUN / "results.jsonl").is_file(),
                    reason="the raw rows of the registered floor run are not in this checkout (the public snapshot packs the raw rows: python scripts/unpack_runs.py)")
def test_the_report_accepts_the_registered_floor_run_with_a_timing_run_of_the_shipped_config():
    """§12 [пилот] m0: the floor part was run and registered before the timing part was rebuilt (B1, B3); the
    report must take that run with the timing run of the config as it is now, and give back what §12 says.
    The timing rows here are made up (linear in the work): the machinery of the report, not a result."""
    floor = pilot.load_run(FLOOR_RUN)
    cfg = run.load_config(REPO_ROOT / "configs" / "E1" / "pilot_timing.yaml")
    target = {"sa": 600, "tabu": 300, "swap_sa": 800}  # the work at which a made-up call takes 1 s

    def tweak(cell, row):
        if cell["kind"] == "timing":
            t = cell[cell["scaled"]] / target[cell["sampler"]]
            row.update(call_s_median=t, call_s_q25=0.9 * t, call_s_q75=1.1 * t, call_s_min=0.8 * t,
                       call_s_max=1.2 * t)

    timing = _synthetic_run(cfg, tweak=tweak)
    timing.manifest = {**timing.manifest, "run_id": "20261101-020000-E1-pilot_timing-fff000", "name": "pilot_timing"}
    report = pilot.build_report([floor, timing])
    assert report["valid_for_section_12"] is True, report["problems"]
    assert report["runs"][0]["run_id"] == "20261011-205903-E1-pilot_floor-9da201"
    fl = report["floor"]
    assert (fl["m0"], fl["n_exceed"], fl["n_instances"]) == (1, 3, 80) and fl["share"] == pytest.approx(3 / 80)
    lo, hi = fl["wilson_95"]
    assert (round(lo, 3), round(hi, 3)) == (0.013, 0.105)  # the interval of the entry in §12
    assert fl["n_not_certified"] == 0
    for needle in ("m₀ = 1", "3 из 80", "20261011-205903-E1-pilot_floor-9da201"):
        assert needle in report["section_12"], needle
    work = report["work"]
    assert work["16"]["tabu"]["chosen"]["num_restarts"] == 0 and work["24"]["tabu"]["chosen"]["tenure"] == 6
    assert work["16"]["swap_sa"]["reference"] == {"penalty": None, "beta": None, "cold_ratio": 0.01}
    assert work["16"]["sa"]["reference"] == {"penalty": 100.0, "beta": "default", "cold_ratio": None}
    assert all(c["status"] == "ok" for per in work.values() for c in per.values())
    # and the block the report gives fits the confirmatory grid: 27 200 cells, B1's tabu among them
    text = (REPO_ROOT / "configs" / "E1" / "attribution.yaml").read_text(encoding="utf-8")
    filled = run.effective_config(yaml.safe_load(pilot.apply_pilot_block(text, report["pilot_values"])))
    assert filled["params"]["pilot"]["m0"] == 1 and len(run.grid(filled)) == 27_200


# --------------------------------------------------------------------------------------
# Through the runner and the report tool
# --------------------------------------------------------------------------------------


def test_the_pilot_runs_through_the_runner_and_the_report_reads_the_registered_runs(tmp_path, capsys):
    runs = tmp_path / "runs"
    run_dirs = []
    for part, name in (("floor", "pilot_floor_t"), ("timing", "pilot_timing_t")):
        cfg = _cfg(parts=(part,))
        cfg["name"] = name
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
        run_dirs.append(next(p for p in runs.iterdir() if p.is_dir() and name in p.name))
    for d in run_dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        assert manifest["n_errors"] == 0 and manifest["status"] == "completed"
        assert manifest["cpu"] and "cpu_governor" in manifest  # the manifest has the CPU of this host
    out = tmp_path / "report.json"
    code = pilot.main(["report", "--run", str(run_dirs[0]), "--run", str(run_dirs[1]), "--out", str(out)])
    report = json.loads(out.read_text())
    assert code in (0, 2) and (code == 0) == report["valid_for_section_12"]
    assert report["floor"]["m0"] in (1, 50) and report["floor"]["n_instances"] == 6
    assert set(report["work"]) == {"12", "14"} and set(report["work"]["12"]) == {"sa", "tabu", "swap_sa"}
    assert report["timing_environment"]["hosts"] and report["timing_environment"]["cpu_models"]
    assert "written" in capsys.readouterr().out
    # a single timing run, without the floor part, is reported but cannot be valid
    code = pilot.main(["report", "--run", str(run_dirs[1]), "--out", str(tmp_path / "t.json")])
    assert code == 2 and json.loads((tmp_path / "t.json").read_text())["floor"] is None


def test_the_env_command_prints_json(capsys):
    assert pilot.main(["env"]) == 0
    assert json.loads(capsys.readouterr().out)["cpu_count_logical"] == os.cpu_count()
