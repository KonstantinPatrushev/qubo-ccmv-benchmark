"""E1 attribution experiment for the runner (ANALYSIS_PLAN §2 RQ1, §6, §7.3; ``cif.experiments.run``).

A cell is (pair seed, N, h, sampler configuration, solver seed); one sampler batch per cell. Samplers
and their nested factors: SA — penalty × β-range (dwave default / scaled to the objective); tabu —
penalty (reads are its work unit, each one plain search: §12 B1); swap-SA — the cold end of the
schedule, ``T_cold = cold_ratio · T_hot`` (§12 B3: the reference 0.01 and the exploratory level 0.2). A row carries ``E_form(m)``, ``E_pipe(m)``, ``E_samp(m)`` and the
diagnostics. Only calibration seeds are used here (instances 1000-1999, solvers 10000-19999).
"""

from __future__ import annotations

import copy
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import yaml

from cif.experiments import e1_attribution as e1
from cif.experiments import run
from cif.heuristics.oracle import pool_subsets, surrogate_energies
from cif.instances.pairs import factor_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
MS = (1, 5, 20, 50, 200)


def _cfg(**params) -> dict:
    cfg = {
        "stream": "E1",
        "name": "attribution_test",
        "experiment": "cif.experiments.e1_attribution",
        "seeds": [1000, 10000, 10001],
        "inputs": {},
        "params": {
            "seed_role": "calibration",
            "instances": [{"n": 12, "pair_seeds": [1000]}],
            "h_levels": [0.15, 0.45],
            "lam": 5.0,
            "w_min": 0.0,
            "w_max": 1.0,
            "ms": list(MS),
            "solver_seeds": [10000, 10001],
            "scip": {"formulation": "perspective", "diag": "mineig", "mip_gap": 1e-6,
                     "time_limit_s": 120.0, "seed": 0},
            "samplers": [
                {"name": "sa", "penalty": [10, 100], "beta": ["default", "objective"],
                 "num_reads": 30, "num_sweeps": 100},
                {"name": "tabu", "penalty": [100], "num_reads": 6, "num_restarts": 0, "tenure": 5},
                {"name": "swap_sa", "cold_ratio": [0.01, 0.2], "num_reads": 10, "num_sweeps": 30},
            ],
        },
    }
    cfg["params"].update(params)
    return cfg


@pytest.fixture(autouse=True)
def _fresh_caches():
    e1.clear_caches()
    yield
    e1.clear_caches()


def _cell(cfg: dict, **where) -> dict:
    for cell in e1.cells(cfg):
        if all(cell[k] == v for k, v in where.items()):
            return cell
    raise AssertionError(f"no cell with {where}")


def _run(cell: dict, cfg: dict) -> dict:
    """What the runner stores: the cell's coordinates and the result of ``run_cell``."""
    return {**cell, **e1.run_cell(cell, cfg)}


# --------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------


def test_cells_cover_pairs_levels_sampler_configurations_and_solver_seeds():
    cfg = _cfg()
    cells = e1.cells(cfg)
    assert len(cells) == 1 * 2 * (4 + 1 + 2) * 2  # pairs × h × (SA 2×2, tabu 1, swap-SA 2 levels) × seeds
    assert all(set(c) == {"pair_seed", "n", "h", "sampler", "penalty", "beta", "cold_ratio", "solver_seed"}
               for c in cells)
    assert {c["h"] for c in cells} == {0.15, 0.45} and {c["solver_seed"] for c in cells} == {10000, 10001}
    assert len(run.grid(cfg)) == len(cells)  # unique cell keys, as the runner demands


def test_nested_factors_belong_to_their_samplers():
    cells = e1.cells(_cfg())
    sa = {(c["penalty"], c["beta"], c["cold_ratio"]) for c in cells if c["sampler"] == "sa"}
    assert sa == {(10.0, "default", None), (10.0, "objective", None), (100.0, "default", None),
                  (100.0, "objective", None)}
    assert {(c["penalty"], c["beta"], c["cold_ratio"]) for c in cells if c["sampler"] == "tabu"} == {
        (100.0, None, None)}
    assert {(c["penalty"], c["beta"], c["cold_ratio"]) for c in cells if c["sampler"] == "swap_sa"} == {
        (None, None, 0.01), (None, None, 0.2)}


def test_cells_are_grouped_by_instance_so_that_the_oracle_cache_is_reused():
    cfg = _cfg(instances=[{"n": 12, "pair_seeds": [1000, 1001]}, {"n": 16, "pair_seeds": [1000]}])
    cfg["seeds"] = [1000, 1001, 10000, 10001]
    cells = e1.cells(cfg)
    instances = [(c["n"], c["pair_seed"], c["h"]) for c in cells]
    firsts = [t for i, t in enumerate(instances) if i == 0 or instances[i - 1] != t]
    assert len(firsts) == len(set(firsts)) == 3 * 2  # each (n, pair, h) occurs in one block
    assert cells == e1.cells(cfg)  # stable order


def test_pair_seeds_are_given_per_n():
    cfg = _cfg(instances=[{"n": 12, "pair_seeds": [1000, 1001]}, {"n": 16, "pair_seeds": [1020]}],
               samplers=[{"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 4, "num_sweeps": 5}],
               solver_seeds=[10000])
    cfg["seeds"] = [1000, 1001, 1020, 10000]
    cells = e1.cells(cfg)
    assert {(c["n"], c["pair_seed"]) for c in cells} == {(12, 1000), (12, 1001), (16, 1020)}


@pytest.mark.parametrize(
    ("role", "pair_seeds", "solver_seeds"),
    [
        ("calibration", [2000], [10000]),  # a test instance seed in a calibration run
        ("calibration", [1000], [20000]),  # a test solver seed
        ("calibration", [999], [10000]),
        ("test", [1000], [20000]),
        ("test", [2000], [10000]),
        ("test", [3000], [20000]),
    ],
)
def test_seed_roles_keep_calibration_and_test_ranges_apart(role, pair_seeds, solver_seeds):
    cfg = _cfg(seed_role=role, instances=[{"n": 12, "pair_seeds": pair_seeds}],
               solver_seeds=solver_seeds)
    cfg["seeds"] = [*pair_seeds, *solver_seeds]
    with pytest.raises(ValueError, match="seed"):
        e1.cells(cfg)


def test_matching_seed_roles_pass():
    for role, pair_seeds, solver_seeds in (("calibration", [1999], [19999]),
                                           ("test", [2000, 2999], [20000, 29999])):
        cfg = _cfg(seed_role=role, instances=[{"n": 12, "pair_seeds": pair_seeds}],
                   solver_seeds=solver_seeds)
        cfg["seeds"] = sorted({*pair_seeds, *solver_seeds})
        assert e1.cells(cfg)


def test_the_seed_role_must_be_stated():
    cfg = _cfg()
    del cfg["params"]["seed_role"]
    with pytest.raises(ValueError, match="seed_role"):
        e1.cells(cfg)
    with pytest.raises(ValueError, match="seed_role"):
        e1.cells(_cfg(seed_role="exploratory"))


def test_the_seeds_recorded_in_the_manifest_must_be_those_of_the_grid():
    cfg = _cfg()
    cfg["seeds"] = [1000, 10000]  # 10001 is used but not declared
    with pytest.raises(ValueError, match="seeds"):
        e1.cells(cfg)
    del cfg["seeds"]
    assert e1.cells(cfg)  # nothing to contradict


def test_a_malformed_config_is_refused_with_a_message():
    with pytest.raises(ValueError, match="sampler"):
        e1.cells(_cfg(samplers=[{"name": "annealer9000", "num_reads": 1}]))
    with pytest.raises(ValueError, match="beta"):
        e1.cells(_cfg(samplers=[{"name": "sa", "penalty": [10], "beta": ["warm"],
                                 "num_reads": 5, "num_sweeps": 5}]))
    with pytest.raises(ValueError, match="penalty"):
        e1.cells(_cfg(samplers=[{"name": "tabu", "num_reads": 2, "num_restarts": 3}]))
    with pytest.raises(ValueError, match="cold_ratio"):  # swap-SA states the cold end of its schedule (R8)
        e1.cells(_cfg(samplers=[{"name": "swap_sa", "num_reads": 4, "num_sweeps": 5}]))
    with pytest.raises(ValueError, match="cold_ratio"):  # a factor that SA does not have
        e1.cells(_cfg(samplers=[{"name": "sa", "penalty": [10], "beta": ["default"], "cold_ratio": [0.01],
                                 "num_reads": 5, "num_sweeps": 5}]))
    with pytest.raises(ValueError, match="pair_seeds"):
        e1.cells(_cfg(instances=[{"n": 12}]))
    cfg = _cfg()
    del cfg["params"]["h_levels"]
    with pytest.raises(ValueError, match="h_levels"):
        e1.cells(cfg)


@pytest.mark.parametrize("bad", [[], 0.01, [0.0], [-0.1], [1.5], [0.01, 0.01], ["cold"], [True], [None]])
def test_the_cold_ratios_of_swap_sa_are_distinct_ratios_in_the_unit_interval(bad):
    with pytest.raises(ValueError, match="cold_ratio"):
        e1.cells(_cfg(samplers=[{"name": "swap_sa", "cold_ratio": bad, "num_reads": 4, "num_sweeps": 5}]))


@pytest.mark.parametrize("key", ["lam", "w_min", "w_max", "ms", "scip"])
def test_every_scalar_parameter_must_be_stated_in_the_config(key):
    """R8: nothing the result depends on is left to a default in the code; a dry run must fail."""
    cfg = _cfg()
    del cfg["params"][key]
    with pytest.raises(ValueError, match=key):
        e1.cells(cfg)


@pytest.mark.parametrize("missing", ["formulation", "diag", "mip_gap", "time_limit_s", "seed"])
def test_the_scip_settings_must_be_complete(missing):
    cfg = _cfg()
    del cfg["params"]["scip"][missing]
    with pytest.raises(ValueError, match=missing):
        e1.cells(cfg)


@pytest.mark.parametrize("ms", [[], [0, 5], [5, 5], [20, 5], [1.5, 3]])
def test_ms_must_be_positive_integers_in_increasing_order(ms):
    with pytest.raises(ValueError, match="ms"):
        e1.cells(_cfg(ms=ms))


def test_a_sampler_listed_twice_is_refused():
    cfg = _cfg(samplers=[{"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 4, "num_sweeps": 5},
                         {"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 8, "num_sweeps": 5}])
    with pytest.raises(ValueError, match="duplicate sampler"):
        e1.cells(cfg)


# --------------------------------------------------------------------------------------
# Seeds of the samplers
# --------------------------------------------------------------------------------------


def test_sampler_seed_is_a_function_of_every_coordinate_of_the_cell():
    cell = {"pair_seed": 1000, "n": 12, "h": 0.15, "sampler": "sa", "penalty": 10.0,
            "beta": "default", "solver_seed": 10000}
    base = e1.sampler_seed(cell)
    assert base == e1.sampler_seed(dict(cell)) and 0 <= base < 2**31  # dwave refuses larger seeds
    changed = [dict(cell, solver_seed=10001), dict(cell, pair_seed=1001), dict(cell, n=16),
               dict(cell, h=0.45), dict(cell, sampler="tabu", beta=None),
               dict(cell, penalty=100.0), dict(cell, beta="objective")]
    seeds = {e1.sampler_seed(c) for c in changed}
    assert base not in seeds and len(seeds) == len(changed)


def test_the_two_levels_of_the_cold_end_of_swap_sa_do_not_share_a_stream():
    """§3: no two cells share a stream; the cold ratio is a coordinate of a swap-SA cell. SA and tabu cells,
    which have none, keep the seed they had before the coordinate existed."""
    swap = {"pair_seed": 1000, "n": 12, "h": 0.15, "sampler": "swap_sa", "penalty": None, "beta": None,
            "solver_seed": 10000}
    ref, warm = e1.sampler_seed({**swap, "cold_ratio": 0.01}), e1.sampler_seed({**swap, "cold_ratio": 0.2})
    assert ref != warm and 0 <= ref < 2**31 and 0 <= warm < 2**31
    sa = {"pair_seed": 1000, "n": 12, "h": 0.15, "sampler": "sa", "penalty": 10.0, "beta": "default",
          "solver_seed": 10000}
    assert e1.sampler_seed({**sa, "cold_ratio": None}) == e1.sampler_seed(sa)


# --------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------


def _check_row(row: dict, ms=MS) -> None:
    assert row["status"] == "ok" and row["consistent"] is True
    json.dumps(row, allow_nan=False)  # numpy scalars and NaN would fail here
    for m in ms:
        assert row[f"E_samp_{m}"] == pytest.approx(row[f"E_pipe_{m}"] - row[f"E_form_{m}"], abs=1e-12)
        # B2: E_samp = E_prec + E_short; the shortfall is not negative and vanishes for a pool of m subsets
        assert row[f"E_samp_{m}"] == pytest.approx(row[f"E_prec_{m}"] + row[f"E_short_{m}"], abs=1e-12)
        assert row[f"E_prec_{m}"] == pytest.approx(row[f"E_pipe_{m}"] - row[f"E_form_at_npipe_{m}"], abs=1e-12)
        assert row[f"E_short_{m}"] >= 0.0 and row[f"E_form_at_npipe_{m}"] >= row[f"E_form_{m}"]
        if row[f"n_pipe_{m}"] == row[f"n_form_{m}"]:
            assert row[f"E_short_{m}"] == 0.0
        assert row[f"n_form_{m}"] == min(m, row["n_subsets"])
        assert row[f"E_form_{m}"] >= -1e-6 and row[f"E_pipe_{m}"] >= -1e-6
        assert 0 <= row[f"n_pipe_{m}"] <= min(m, row["pool_n_unique"])
    form = [row[f"E_form_{m}"] for m in ms]
    pipe = [row[f"E_pipe_{m}"] for m in ms]
    assert all(b <= a + 1e-15 for a, b in pairwise(form))  # nested T_m
    assert all(b <= a + 1e-15 for a, b in pairwise(pipe))  # nested pools
    assert row["ref_type"] == "certified" and row["scale"] > 0.0
    assert row["pool_n_valid"] <= row["pool_n_input"] and row["pool_n_unique"] <= row["pool_n_valid"]
    assert 0.0 <= row["frac_valid"] <= 1.0
    for key in ("build_s", "solve_s", "refine_s", "oracle_s", "cell_s"):
        assert row[key] >= 0.0
    assert row["solve_s"] > 0.0


@pytest.mark.parametrize(
    "where",
    [
        {"sampler": "sa", "penalty": 10.0, "beta": "default"},
        {"sampler": "sa", "penalty": 100.0, "beta": "objective"},
        {"sampler": "tabu"},
        {"sampler": "swap_sa"},
    ],
)
def test_every_sampler_produces_a_consistent_row(where):
    cfg = _cfg()
    row = _run(_cell(cfg, h=0.45, solver_seed=10000, **where), cfg)
    _check_row(row)
    assert row["sampler"] == where["sampler"] and row["h"] == 0.45


def test_row_carries_the_instance_reference_and_sampler_diagnostics():
    cfg = _cfg()
    cell = _cell(cfg, h=0.15, sampler="sa", penalty=100.0, beta="objective", solver_seed=10000)
    row = _run(cell, cfg)
    for key in ("K", "s_cont", "binding", "depth", "cv_vol", "cv_mu", "cond_sigma", "n_subsets",
                "f_star", "f_scip", "scale", "scip_status", "scip_gap", "scip_dual_bound",
                "scip_primal_bound", "scip_nodes", "scip_solve_s", "oracle_cached",
                "tau_top", "tau_pool", "hit_surrogate_min", "surrogate_gap_best", "penalty_abs",
                "beta_hot", "beta_cold", "n_samples", "n_valid_samples", "config_id", "refine_ms_per_call"):
        assert key in row, key
    assert row["depth"] == pytest.approx(row["s_cont"] / row["K"])
    assert row["n_subsets"] == math.comb(12, row["K"])
    assert row["config_id"] == "sa/p100/objective"
    assert row["penalty_abs"] > 0.0 and 0.0 < row["beta_hot"] < row["beta_cold"]
    assert row["surrogate_gap_best"] is None or row["surrogate_gap_best"] >= 0.0
    assert row["scip_status"] in ("optimal", "gaplimit") and row["scip_gap"] <= 1e-6
    assert row["f_star"] <= row["f_scip"] + 1e-12  # polished with the shared refinement


def test_a_tabu_read_is_one_plain_search_and_the_call_returns_one_state_per_read():
    """§12 B1: ``num_restarts = 0``, the work of the call is its reads; every read returns one state."""
    cfg = _cfg()
    row = _run(_cell(cfg, h=0.45, sampler="tabu", solver_seed=10000), cfg)
    assert row["num_restarts"] == 0 and row["restarts_mean"] == 0.0
    assert row["num_reads"] == row["n_samples"] == 6 and row["tenure"] == 5
    assert row["pool_n_unique"] <= row["num_reads"]  # a pool never has more subsets than the call has reads


def test_swap_sa_anneals_to_the_cold_ratio_of_its_cell():
    """§12 B3: ``T_cold = 0.01 · T_hot`` for the reference, ``0.2 · T_hot`` for the exploratory level;
    the hot temperature is the one of the 80 % acceptance calibration (the sampler's own)."""
    cfg = _cfg()
    ref = _run(_cell(cfg, h=0.45, sampler="swap_sa", cold_ratio=0.01, solver_seed=10000), cfg)
    warm = _run(_cell(cfg, h=0.45, sampler="swap_sa", cold_ratio=0.2, solver_seed=10000), cfg)
    assert ref["t_cold"] == pytest.approx(0.01 * ref["t_hot"], rel=1e-12)
    assert warm["t_cold"] == pytest.approx(0.2 * warm["t_hot"], rel=1e-12)
    assert (ref["config_id"], warm["config_id"]) == ("swap_sa/c0.01", "swap_sa/c0.2")
    assert (ref["cold_ratio"], warm["cold_ratio"]) == (0.01, 0.2)
    _check_row(warm)


def test_a_warmer_end_of_the_schedule_keeps_more_distinct_subsets_in_the_pool():
    """The reason for the exploratory level (§12 B3): the pool of the reference schedule collapses onto the
    minimum of the surrogate, a warmer end of the schedule keeps the diversity. Deterministic for the seed."""
    cfg = _cfg(samplers=[{"name": "swap_sa", "cold_ratio": [0.01, 0.2], "num_reads": 80, "num_sweeps": 40}])
    ref = _run(_cell(cfg, h=0.45, sampler="swap_sa", cold_ratio=0.01, solver_seed=10000), cfg)
    warm = _run(_cell(cfg, h=0.45, sampler="swap_sa", cold_ratio=0.2, solver_seed=10000), cfg)
    assert warm["pool_n_unique"] > ref["pool_n_unique"]


@pytest.mark.parametrize(
    ("config", "role"),
    [
        (("sa", 100.0, "default", None), "reference"),
        (("sa", 100.0, "objective", None), "grid"),
        (("sa", 10.0, "default", None), "grid"),
        (("tabu", 100.0, None, None), "reference"),
        (("tabu", 30.0, None, None), "grid"),
        (("swap_sa", None, None, 0.01), "reference"),
        (("swap_sa", None, None, 0.2), "exploratory"),
        (("swap_sa", None, None, 0.1), "exploratory"),  # whatever is not the reference of swap-SA is not in the plan
    ],
)
def test_the_role_of_a_configuration_follows_section_12_b3(config, role):
    assert e1.config_role(*config) == role


def test_the_references_are_those_of_the_plan():
    from cif.qubo.builder import SELECTION_PENALTY_MULTIPLIER

    assert e1.REFERENCE == {"sa": {"penalty": 100.0, "beta": "default"}, "tabu": {"penalty": 100.0},
                            "swap_sa": {"cold_ratio": 0.01}}
    assert e1.REFERENCE["sa"]["penalty"] == e1.REFERENCE["tabu"]["penalty"] == SELECTION_PENALTY_MULTIPLIER
    assert e1.EXPLORATORY_COLD_RATIO == 0.2


def test_a_confirmatory_grid_holds_the_reference_configuration_of_every_sampler():
    """The contrast of samplers is made at the references: a test-seed grid without one is refused."""
    cfg = _cfg(seed_role="test", instances=[{"n": 12, "pair_seeds": [2000]}], solver_seeds=[20000])
    cfg["seeds"] = [2000, 20000]
    for spec in cfg["params"]["samplers"]:  # the work comes from the pilot block below
        for key in ("num_reads", "num_sweeps", "num_restarts", "tenure"):
            spec.pop(key, None)
    cfg["params"]["pilot"] = {"m0": 1, "work": [
        {"sampler": "sa", "n": 12, "num_reads": 4, "num_sweeps": 5},
        {"sampler": "tabu", "n": 12, "num_reads": 4, "num_restarts": 0, "tenure": 5},
        {"sampler": "swap_sa", "n": 12, "num_reads": 4, "num_sweeps": 5}]}
    assert e1.cells(cfg)  # the base grid holds the reference of every sampler
    cfg["params"]["samplers"][1]["penalty"] = [10]  # tabu without its reference (penalty 100)
    with pytest.raises(ValueError, match=r"reference.*tabu"):
        e1.cells(cfg)
    cfg["params"]["samplers"][1]["penalty"] = [10, 100]
    assert e1.cells(cfg)
    cfg["params"]["samplers"][0]["beta"] = ["objective"]  # SA without the β range of dwave
    with pytest.raises(ValueError, match=r"reference.*sa"):
        e1.cells(cfg)
    cfg["params"]["samplers"][0]["beta"] = ["default", "objective"]
    cfg["params"]["samplers"][2]["cold_ratio"] = [0.2]  # the exploratory level alone is not the reference
    with pytest.raises(ValueError, match=r"reference.*swap_sa"):
        e1.cells(cfg)
    # a calibration (smoke) grid is a rehearsal of the machinery: no such demand
    assert e1.cells(_cfg(samplers=[{"name": "swap_sa", "cold_ratio": [0.2], "num_reads": 4, "num_sweeps": 5}]))


def test_default_beta_range_is_the_one_dwave_chose():
    cfg = _cfg()
    row = _run(_cell(cfg, h=0.15, sampler="sa", penalty=10.0, beta="default",
                            solver_seed=10000), cfg)
    assert 0.0 < row["beta_hot"] < row["beta_cold"]  # reported from the sampler's own info


def test_objective_scaled_beta_range_does_not_depend_on_the_penalty():
    """The penalty levels of one instance and solver seed share the calibrated range exactly, so the
    penalty effect is not confounded with the noise of the calibration."""
    cfg = _cfg()
    low = _run(_cell(cfg, h=0.15, sampler="sa", penalty=10.0, beta="objective",
                            solver_seed=10000), cfg)
    high = _run(_cell(cfg, h=0.15, sampler="sa", penalty=100.0, beta="objective",
                             solver_seed=10000), cfg)
    assert low["beta_hot"] == pytest.approx(high["beta_hot"], rel=0, abs=0)
    assert low["beta_cold"] == pytest.approx(100.0 * low["beta_hot"], rel=1e-12)  # cold = 0.01·hot
    assert high["penalty_abs"] == pytest.approx(10.0 * low["penalty_abs"], rel=1e-12)


def test_the_two_levels_of_a_pair_share_k_and_differ_in_volatility_dispersion():
    cfg = _cfg()
    lo = _run(_cell(cfg, h=0.15, sampler="swap_sa", solver_seed=10000), cfg)
    hi = _run(_cell(cfg, h=0.45, sampler="swap_sa", solver_seed=10000), cfg)
    assert lo["K"] == hi["K"] and lo["n"] == hi["n"] == 12
    assert hi["cv_vol"] > lo["cv_vol"]
    assert lo["s_cont_min_pair"] == hi["s_cont_min_pair"] == min(lo["s_cont"], hi["s_cont"])


@pytest.mark.parametrize(
    "where",
    [
        {"sampler": "sa", "penalty": 100.0, "beta": "default"},
        {"sampler": "sa", "penalty": 10.0, "beta": "objective"},
        {"sampler": "tabu"},
        {"sampler": "swap_sa"},
    ],
)
def test_a_cell_is_reproducible_up_to_timings(where):
    cfg = _cfg()
    cell = _cell(cfg, h=0.45, solver_seed=10001, **where)
    a = _run(cell, cfg)
    e1.clear_caches()
    b = _run(cell, cfg)
    timing = {k for k in a if k.endswith("_s") or k.endswith("_cpu_s") or k == "refine_ms_per_call"}
    timing |= {"oracle_cached", "n_refined"}
    assert {k: v for k, v in a.items() if k not in timing} == {k: v for k, v in b.items() if k not in timing}


def test_the_oracle_is_computed_once_per_instance_and_shared_by_all_cells_of_it():
    cfg = _cfg()
    first = _run(_cell(cfg, h=0.45, sampler="swap_sa", solver_seed=10000), cfg)
    second = _run(_cell(cfg, h=0.45, sampler="tabu", solver_seed=10000), cfg)
    other = _run(_cell(cfg, h=0.15, sampler="tabu", solver_seed=10000), cfg)
    assert first["oracle_cached"] is False and second["oracle_cached"] is True
    assert other["oracle_cached"] is False  # another level is another instance
    for m in MS:
        assert first[f"E_form_{m}"] == second[f"E_form_{m}"]
    assert first["oracle_s"] == second["oracle_s"]  # the cost of the instance, not of the cell
    assert first["f_star"] == second["f_star"]


def test_pool_subsets_already_refined_by_the_oracle_cost_nothing_again():
    cfg = _cfg()
    cells = [_cell(cfg, h=0.45, sampler="swap_sa", solver_seed=s) for s in (10000, 10001)]
    first, second = (_run(c, cfg) for c in cells)
    assert first["n_refined"] >= 0 and second["n_refined"] >= 0
    assert second["refine_ms_per_call"] is None or second["refine_ms_per_call"] > 0.0


@pytest.mark.parametrize("sampler", ["sa", "tabu"])
def test_qubo_pools_are_in_ascending_sampler_energy_and_count_their_valid_samples(sampler):
    """E_pipe takes the FIRST m of the pool, so its order is the order of sampler energy: on the
    valid states the BQM energy is the surrogate energy, which must not decrease along the pool."""
    cfg = _cfg()
    pair = factor_pair(12, 1000)
    problem, k = pair.member(0.45).problem, pair.k
    spec = next(s for s in cfg["params"]["samplers"] if s["name"] == sampler)
    spec = dict(spec, num_reads=200, num_sweeps=20)  # short anneals: many different states
    cell = _cell(cfg, h=0.45, sampler=sampler, solver_seed=10000)
    seed, rng = e1._derive(cell)
    draw = e1._draw_pool(problem, k, cell, spec, seed, rng)
    assert draw.n_samples == len(draw.pool) == 200
    assert draw.n_valid_samples == sum(len(s) == k for s in draw.pool)
    unique, stats = pool_subsets(draw.pool, k, problem.n, limit=10_000)
    assert stats["n_valid"] == draw.n_valid_samples and len(unique) > 3
    energy = surrogate_energies(problem.mu, problem.sigma, problem.risk_aversion, np.array(unique))
    assert all(b >= a - 1e-12 for a, b in pairwise(energy.tolist()))


def test_a_sampler_without_a_valid_subset_gives_a_row_with_undefined_e_pipe(monkeypatch):
    cfg = _cfg()
    cell = _cell(cfg, h=0.45, sampler="sa", penalty=10.0, beta="default", solver_seed=10000)

    def no_valid_subset(problem, k, cell, spec, seed, rng):
        return e1.PoolDraw(pool=[(0,), (1, 2)], n_samples=7, n_valid_samples=0, build_s=0.001,
                           solve_s=0.002, solve_cpu_s=0.002)

    monkeypatch.setattr(e1, "_draw_pool", no_valid_subset)
    row = _run(cell, cfg)
    assert row["status"] == "no_valid_samples"
    assert row["E_pipe_1"] is None and row["E_samp_1"] is None and row["n_pipe_1"] == 0
    assert row["E_prec_1"] is None and row["E_short_1"] is None and row["E_form_at_npipe_1"] is None
    assert row["E_form_1"] is not None and row["frac_valid"] == 0.0
    assert row["hit_surrogate_min"] is None and row["surrogate_gap_best"] is None
    json.dumps(row, allow_nan=False)


def test_a_pool_that_is_the_surrogate_ranking_has_zero_sampler_residual(monkeypatch):
    cfg = _cfg()
    cell = _cell(cfg, h=0.45, sampler="swap_sa", solver_seed=10000)

    def perfect_sampler(problem, k, cell, spec, seed, rng):
        from cif.heuristics.oracle import best_subsets

        top, _ = best_subsets(problem, k, 300)
        return e1.PoolDraw(pool=[tuple(s) for s in top.tolist()], n_samples=300, n_valid_samples=300,
                           build_s=0.0, solve_s=0.001, solve_cpu_s=0.001)

    monkeypatch.setattr(e1, "_draw_pool", perfect_sampler)
    row = _run(cell, cfg)
    for m in MS:
        assert row[f"E_samp_{m}"] == pytest.approx(0.0, abs=1e-14)
        assert row[f"E_prec_{m}"] == pytest.approx(0.0, abs=1e-14) and row[f"E_short_{m}"] == 0.0
    assert row["hit_surrogate_min"] is True and row["surrogate_gap_best"] == 0.0


# --------------------------------------------------------------------------------------
# Through the runner and the shipped smoke config
# --------------------------------------------------------------------------------------


def test_runner_executes_and_registers_the_whole_grid(tmp_path):
    cfg = _cfg(samplers=[{"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 6, "num_sweeps": 20}],
               solver_seeds=[10000])
    cfg["seeds"] = [1000, 10000]
    path = tmp_path / "e1.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and not any(r.get("error") for r in rows)  # h × one sampler × one seed
    assert {r["h"] for r in rows} == {0.15, 0.45} and all(r["status"] == "ok" for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == [1000, 10000] and manifest["n_errors"] == 0


def test_the_shipped_smoke_config_is_calibration_only_and_small():
    path = REPO_ROOT / "configs" / "E1" / "attribution_smoke.yaml"
    cfg = run.load_config(path)
    assert (cfg["stream"], cfg["experiment"]) == ("E1", "cif.experiments.e1_attribution")
    for smoke in (False, True):
        eff = run.effective_config(cfg, smoke=smoke)
        assert eff["params"]["seed_role"] == "calibration"
        cells = run.grid(eff)  # validates seed ranges, manifest seeds and unique keys
        assert cells and all(1000 <= c["pair_seed"] < 2000 and 10000 <= c["solver_seed"] < 20000
                             for c in cells)
        assert {c["h"] for c in cells} == {0.15, 0.45}
    smoke_cells = run.grid(run.effective_config(cfg, smoke=True))
    assert len(smoke_cells) <= 12
    assert {c["sampler"] for c in smoke_cells} == {"sa", "tabu", "swap_sa"}
    assert {c["beta"] for c in smoke_cells if c["sampler"] == "sa"} == {"default", "objective"}
    assert cfg["inputs"] == {}  # synthetic instances: no market data is read


def test_dry_run_of_the_smoke_config_prints_the_grid(capsys):
    path = REPO_ROOT / "configs" / "E1" / "attribution_smoke.yaml"
    assert run.main(["--config", str(path), "--dry-run", "--smoke"]) == 0
    assert "E1/attribution_smoke (smoke)" in capsys.readouterr().out


def test_the_config_file_itself_is_not_mutated_by_cells():
    cfg = _cfg()
    before = copy.deepcopy(cfg)
    e1.cells(cfg)
    assert cfg == before
