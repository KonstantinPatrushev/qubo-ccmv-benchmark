"""E1 attribution: the work of a sampler call per N from the pilot, and the shard-friendly cell order.

ANALYSIS_PLAN §6: the work of one call (reads × sweeps for SA and swap-SA, reads for tabu: §12 B1) is
sized on the pilot per N and is a *parameter* of the confirmatory config, not a guess in the code.
``params.pilot`` holds it; ``cells()`` refuses a config whose pilot values are not filled in.
``params.cell_order: interleaved`` puts the pairs innermost, so that the stride sharding of the
runner (``cells[j::S]``) gives every shard whole pairs and the instance-level oracle (certified
optimum, enumeration) is computed once per shard instead of once per cell.
"""

from __future__ import annotations

import copy

import pytest

from cif.experiments import e1_attribution as e1
from cif.experiments import run

MS = [1, 5, 20, 50, 200]


def _pilot_block(**over) -> dict:
    block = {
        "m0": 50,
        "work": [
            {"sampler": "sa", "n": 12, "num_reads": 30, "num_sweeps": 20},
            {"sampler": "sa", "n": 14, "num_reads": 40, "num_sweeps": 25},
            {"sampler": "tabu", "n": 12, "num_reads": 4, "num_restarts": 0, "tenure": 5},
            {"sampler": "tabu", "n": 14, "num_reads": 5, "num_restarts": 0, "tenure": 5},
            {"sampler": "swap_sa", "n": 12, "num_reads": 8, "num_sweeps": 9},
            {"sampler": "swap_sa", "n": 14, "num_reads": 10, "num_sweeps": 11},
        ],
    }
    block.update(over)
    return block


def _cfg(**params) -> dict:
    cfg = {
        "stream": "E1",
        "name": "attribution_layout_test",
        "experiment": "cif.experiments.e1_attribution",
        "seeds": [1000, 1001, 1002, 1003, 10000, 10001],
        "inputs": {},
        "params": {
            "seed_role": "calibration",
            "instances": [{"n": 12, "pair_seeds": [1000, 1001]}, {"n": 14, "pair_seeds": [1002, 1003]}],
            "h_levels": [0.15, 0.45],
            "lam": 5.0,
            "w_min": 0.0,
            "w_max": 1.0,
            "ms": MS,
            "solver_seeds": [10000, 10001],
            "scip": {"formulation": "perspective", "diag": "mineig", "mip_gap": 1e-6,
                     "time_limit_s": 120.0, "seed": 0},
            "samplers": [
                {"name": "sa", "penalty": [10, 100], "beta": ["default", "objective"]},
                {"name": "tabu", "penalty": [100]},
                {"name": "swap_sa", "cold_ratio": [0.01]},
            ],
            "pilot": _pilot_block(),
        },
    }
    cfg["params"].update(params)
    return cfg


@pytest.fixture(autouse=True)
def _fresh_caches():
    e1.clear_caches()
    yield
    e1.clear_caches()


# --------------------------------------------------------------------------------------
# The pilot block
# --------------------------------------------------------------------------------------


def test_a_config_with_pilot_work_has_its_cells():
    cells = e1.cells(_cfg())
    assert len(cells) == 4 * 2 * (4 + 1 + 1) * 2  # pairs × h × (SA 2×2, tabu 1, swap-SA 1 level) × seeds
    assert len(run.grid(_cfg())) == len(cells)


@pytest.mark.parametrize("field", ["work", "m0"])
def test_an_unfilled_pilot_block_is_refused_with_a_pointer_to_the_pilot(field):
    cfg = _cfg()
    cfg["params"]["pilot"][field] = None
    with pytest.raises(ValueError, match="pilot"):
        e1.cells(cfg)


def test_a_pilot_block_that_is_null_is_refused():
    cfg = _cfg()
    cfg["params"]["pilot"] = {"source": None, "m0": None, "work": None}
    with pytest.raises(ValueError, match=r"pilot.*(not filled|missing|null)"):
        e1.cells(cfg)


def test_a_null_work_value_is_named_in_the_message():
    cfg = _cfg()
    cfg["params"]["pilot"]["work"][1]["num_reads"] = None
    with pytest.raises(ValueError, match=r"sa.*n=14.*num_reads"):
        e1.cells(cfg)


def test_work_missing_for_some_n_is_refused():
    cfg = _cfg()
    del cfg["params"]["pilot"]["work"][3]  # tabu at n = 14
    with pytest.raises(ValueError, match=r"tabu.*n=14"):
        e1.cells(cfg)


def test_a_missing_batch_key_is_named():
    cfg = _cfg()
    del cfg["params"]["pilot"]["work"][2]["num_restarts"]
    with pytest.raises(ValueError, match=r"tabu.*n=12.*num_restarts"):
        e1.cells(cfg)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "50"])
def test_work_values_must_be_positive_integers(bad):
    cfg = _cfg()
    cfg["params"]["pilot"]["work"][0]["num_sweeps"] = bad
    with pytest.raises(ValueError, match="num_sweeps"):
        e1.cells(cfg)


def test_work_of_an_unknown_sampler_or_a_duplicate_entry_is_refused():
    cfg = _cfg()
    cfg["params"]["pilot"]["work"].append({"sampler": "annealer9000", "n": 12, "num_reads": 3})
    with pytest.raises(ValueError, match="annealer9000"):
        e1.cells(cfg)
    cfg = _cfg()
    cfg["params"]["pilot"]["work"].append(dict(cfg["params"]["pilot"]["work"][0]))
    with pytest.raises(ValueError, match=r"duplicate"):
        e1.cells(cfg)


def test_work_for_an_n_outside_the_grid_is_refused():
    cfg = _cfg()
    cfg["params"]["pilot"]["work"].append(
        {"sampler": "swap_sa", "n": 99, "num_reads": 3, "num_sweeps": 3})
    with pytest.raises(ValueError, match="n=99"):
        e1.cells(cfg)


def test_m0_must_be_one_of_the_ms():
    cfg = _cfg()
    cfg["params"]["pilot"]["m0"] = 7
    with pytest.raises(ValueError, match="m0"):
        e1.cells(cfg)


def _test_role_cfg(source) -> dict:
    cfg = _cfg(seed_role="test", solver_seeds=[20000])
    cfg["params"]["instances"] = [{"n": 12, "pair_seeds": [2000, 2001]},
                                  {"n": 14, "pair_seeds": [2002, 2003]}]
    cfg["seeds"] = [2000, 2001, 2002, 2003, 20000]
    cfg["params"]["pilot"]["source"] = source
    return cfg


def test_a_confirmatory_config_refuses_values_from_a_pilot_that_was_not_valid():
    """The block of a smoke or dirty pilot (``valid_for_section_12`` false in its source) must not
    start a run on test seeds; this only builds the grid, nothing is run."""
    with pytest.raises(ValueError, match="not valid for §12"):
        e1.cells(_test_role_cfg({"valid_for_section_12": False, "run_ids": ["smoke"]}))
    assert e1.cells(_test_role_cfg({"valid_for_section_12": True, "run_ids": ["pilot"]}))
    assert e1.cells(_test_role_cfg(None))  # a block written by hand has no source: allowed
    # calibration runs (the smoke config) may use a hand-made block
    cfg = _cfg()
    cfg["params"]["pilot"]["source"] = {"valid_for_section_12": False}
    assert e1.cells(cfg)


def test_the_work_may_not_be_given_twice():
    cfg = _cfg()
    cfg["params"]["samplers"][2] = {"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 4, "num_sweeps": 5}
    with pytest.raises(ValueError, match=r"swap_sa.*(twice|both)"):
        e1.cells(cfg)


def test_a_config_without_a_pilot_block_still_takes_flat_work():
    """The smoke config of the machinery sizes the batches by hand: unchanged."""
    cfg = _cfg()
    del cfg["params"]["pilot"]
    cfg["params"]["samplers"] = [{"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 4, "num_sweeps": 5}]
    assert e1.cells(cfg)
    cfg["params"]["samplers"] = [{"name": "swap_sa", "cold_ratio": [0.01], "num_reads": 4}]
    with pytest.raises(ValueError, match="num_sweeps"):
        e1.cells(cfg)


def test_a_sampler_without_work_is_refused_when_there_is_no_pilot_block():
    cfg = _cfg()
    del cfg["params"]["pilot"]
    with pytest.raises(ValueError, match="num_reads"):
        e1.cells(cfg)


def test_the_cells_do_not_mutate_the_config():
    cfg = _cfg()
    before = copy.deepcopy(cfg)
    e1.cells(cfg)
    assert cfg == before


def test_every_level_of_swap_sa_has_the_work_of_its_n_from_the_pilot():
    """§12 B3: the work is sized on the reference (cold ratio 0.01) and every level of the nested factors,
    the exploratory ratio too, runs the same work."""
    cfg = _cfg()
    cfg["params"]["samplers"][2] = {"name": "swap_sa", "cold_ratio": [0.01, 0.2]}
    swap = [c for c in e1.cells(cfg) if c["sampler"] == "swap_sa"]
    assert {c["cold_ratio"] for c in swap} == {0.01, 0.2}
    for n, work in ((12, {"num_reads": 8, "num_sweeps": 9}), (14, {"num_reads": 10, "num_sweeps": 11})):
        spec = e1._batch_spec(cfg["params"], "swap_sa", n)
        assert {k: spec[k] for k in work} == work


def test_run_cell_hands_the_work_of_its_n_to_the_sampler(monkeypatch):
    cfg = _cfg(solver_seeds=[10000])
    cfg["seeds"] = [1000, 1001, 1002, 1003, 10000]
    seen = []

    def spy(problem, k, cell, spec, seed, rng):
        seen.append((cell["sampler"], cell["n"], {key: spec[key] for key in spec
                                                   if key.startswith("num_")}))
        return e1.PoolDraw(pool=[], n_samples=0, n_valid_samples=0, build_s=0.0, solve_s=0.001,
                           solve_cpu_s=0.001)

    monkeypatch.setattr(e1, "_draw_pool", spy)
    wanted = {("sa", 12), ("sa", 14), ("tabu", 12), ("tabu", 14), ("swap_sa", 12), ("swap_sa", 14)}
    for cell in e1.cells(cfg):
        if cell["h"] == 0.15 and cell["pair_seed"] in (1000, 1002) and (cell["sampler"], cell["n"]) in wanted:
            wanted.discard((cell["sampler"], cell["n"]))
            e1.run_cell(cell, cfg)
    assert not wanted
    got = {(name, n): spec for name, n, spec in seen}
    assert got[("sa", 12)] == {"num_reads": 30, "num_sweeps": 20}
    assert got[("sa", 14)] == {"num_reads": 40, "num_sweeps": 25}
    assert got[("tabu", 12)] == {"num_reads": 4, "num_restarts": 0}
    assert got[("tabu", 14)] == {"num_reads": 5, "num_restarts": 0}
    assert got[("swap_sa", 12)] == {"num_reads": 8, "num_sweeps": 9}
    assert got[("swap_sa", 14)] == {"num_reads": 10, "num_sweeps": 11}


def test_tenure_comes_from_the_pilot_work_as_well():
    """The tenure max(5, N // 4) of B1 is 5 at N = 16 and 6 at N = 24: it travels with the work of its N."""
    params = {"samplers": [{"name": "tabu", "penalty": [100]}], "pilot": {"m0": 1, "work": [
        {"sampler": "tabu", "n": 16, "num_reads": 4, "num_restarts": 0, "tenure": 5},
        {"sampler": "tabu", "n": 24, "num_reads": 5, "num_restarts": 0, "tenure": 6}]}}
    assert e1._batch_spec(params, "tabu", 16)["tenure"] == 5
    assert e1._batch_spec(params, "tabu", 24)["tenure"] == 6


# --------------------------------------------------------------------------------------
# B1: the work unit of tabu is reads (ANALYSIS_PLAN §12, 2026-10-11)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "tenure"), [(12, 5), (16, 5), (20, 5), (24, 6), (28, 7), (100, 25)])
def test_the_tenure_of_tabu_is_max_5_and_a_quarter_of_n(n, tenure):
    assert e1.tabu_tenure(n) == tenure


def test_a_read_of_tabu_is_one_plain_search():
    assert e1.TABU_RESTARTS == 0


@pytest.mark.parametrize("restarts", [1, 3, 40])
def test_tabu_work_with_restarts_is_refused_as_the_old_regime(restarts):
    """B1: ``num_restarts`` of the pilot work is 0, reads are the unit; a pilot of the old regime (10 reads,
    restarts scaled) must not start a confirmatory run."""
    cfg = _cfg()
    cfg["params"]["pilot"]["work"][2]["num_restarts"] = restarts
    with pytest.raises(ValueError, match=r"tabu.*n=12.*num_restarts.*B1"):
        e1.cells(cfg)


@pytest.mark.parametrize("bad", [0, 4, 6, 20])
def test_tabu_work_must_carry_the_tenure_of_the_plan_for_its_n(bad):
    cfg = _cfg()
    cfg["params"]["pilot"]["work"][3]["tenure"] = bad  # n = 14: max(5, 14 // 4) = 5
    with pytest.raises(ValueError, match=r"tabu.*n=14.*tenure"):
        e1.cells(cfg)


def test_tabu_work_with_zero_restarts_and_the_rule_tenure_is_accepted():
    assert e1.cells(_cfg())  # the fixture is in the regime of B1


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "50"])
def test_tabu_reads_must_be_positive_integers_although_the_restarts_are_zero(bad):
    cfg = _cfg()
    cfg["params"]["pilot"]["work"][2]["num_reads"] = bad
    with pytest.raises(ValueError, match="num_reads"):
        e1.cells(cfg)


# --------------------------------------------------------------------------------------
# The cell order
# --------------------------------------------------------------------------------------


def _pair_of(cell: dict) -> tuple[int, int]:
    return cell["n"], cell["pair_seed"]


def test_the_default_order_groups_cells_by_instance():
    cells = e1.cells(_cfg())
    instances = [(c["n"], c["pair_seed"], c["h"]) for c in cells]
    blocks = [t for i, t in enumerate(instances) if i == 0 or instances[i - 1] != t]
    assert len(blocks) == len(set(blocks)) == 8


def test_an_unknown_cell_order_is_refused():
    with pytest.raises(ValueError, match="cell_order"):
        e1.cells(_cfg(cell_order="random"))


def test_interleaved_order_has_the_same_cells_in_another_order():
    base = e1.cells(_cfg())
    inter = e1.cells(_cfg(cell_order="interleaved"))
    key = run.cell_key
    assert sorted(map(key, base)) == sorted(map(key, inter)) and base != inter
    assert len(run.grid(_cfg(cell_order="interleaved"))) == len(base)  # unique keys


@pytest.mark.parametrize("n_shards", [1, 2, 4])
def test_interleaved_shards_hold_whole_pairs(n_shards):
    """``cells[j::S]`` is what ``--shard j --n-shards S`` runs: with S dividing the number of pairs
    every pair, with both levels, every sampler configuration and every seed, falls into one shard."""
    cells = e1.cells(_cfg(cell_order="interleaved"))
    n_pairs = 4
    assert n_pairs % n_shards == 0
    shards = [cells[j::n_shards] for j in range(n_shards)]
    assert sum(map(len, shards)) == len(cells)
    owner = {}
    for j, shard in enumerate(shards):
        for cell in shard:
            assert owner.setdefault(_pair_of(cell), j) == j
    assert len(owner) == n_pairs
    sizes = {len(s) for s in shards}
    assert len(sizes) == 1  # equal work in cells


def test_interleaved_order_spreads_the_pairs_over_the_first_cells():
    """Early cells cover all pairs, so a partial run is a balanced sample of the design."""
    cells = e1.cells(_cfg(cell_order="interleaved"))
    first = cells[:4]
    assert {_pair_of(c) for c in first} == {(12, 1000), (12, 1001), (14, 1002), (14, 1003)}
    assert len({(c["sampler"], c["penalty"], c["beta"], c["solver_seed"], c["h"]) for c in first}) == 1


def test_the_oracle_cache_holds_all_instances_of_a_shard():
    """A shard of 5 pairs touches 10 instances in turn; a cache smaller than that would recompute
    the certified optimum on almost every cell."""
    assert e1._instance.cache_info().maxsize >= 160
