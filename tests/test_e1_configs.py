"""The shipped E1 configs (configs/E1/): sizes against the plan, seed roles, the pilot block.

Only grids are built here (``cells`` / ``--dry-run``): no cell of a test-seed config is ever run.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from cif.experiments import e1_attribution as e1
from cif.experiments import e1_pilot as pilot
from cif.experiments import run

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "E1"


def _effective(name: str, smoke: bool = False) -> dict:
    return run.effective_config(run.load_config(CONFIGS / name), smoke=smoke)


def _fake_report(m0: int = 50, valid: bool = True) -> dict:
    """A report with the shape of ``build_report`` for the N and samplers of the shipped configs."""
    work = {}
    for n in (16, 24):
        work[str(n)] = {
            "sa": {"chosen": {"num_reads": 2000 + n, "num_sweeps": 1000}},
            "tabu": {"chosen": {"num_reads": 400 + n, "num_restarts": 0, "tenure": 5 if n == 16 else 6}},
            "swap_sa": {"chosen": {"num_reads": 3000 + n, "num_sweeps": 200}},
        }
    return {"valid_for_section_12": valid, "problems": [] if valid else ["smoke"],
            "runs": [{"run_id": "20261101-000000-E1-pilot_floor-aaaaaa", "git_sha": "abc123"},
                     {"run_id": "20261101-010000-E1-pilot_timing-bbbbbb", "git_sha": "abc123"}],
            "floor": {"m0": m0}, "work": work}


@pytest.fixture(scope="module")
def attribution_text() -> str:
    return (CONFIGS / "attribution.yaml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def filled_cfg(attribution_text) -> dict:
    values = pilot.pilot_values(_fake_report())
    return run.effective_config(yaml.safe_load(pilot.apply_pilot_block(attribution_text, values)))


@pytest.fixture(scope="module")
def filled_cells(filled_cfg) -> list[dict]:
    return run.grid(filled_cfg)


# --------------------------------------------------------------------------------------
# Pilot, part 1: the floor
# --------------------------------------------------------------------------------------


def test_pilot_floor_has_20_calibration_pairs_per_n():
    cells = run.grid(_effective("pilot_floor.yaml"))
    assert len(cells) == 2 * 20 * 2 and {c["kind"] for c in cells} == {"floor"}
    by_n = {n: sorted({c["pair_seed"] for c in cells if c["n"] == n}) for n in (16, 24)}
    assert by_n[16] == list(range(1000, 1020)) and by_n[24] == list(range(1020, 1040))
    assert {c["h"] for c in cells} == {0.15, 0.45}


def test_pilot_floor_states_the_rule_of_the_plan():
    cfg = _effective("pilot_floor.yaml")
    assert cfg["params"]["floor"] == {"tau": 1e-3, "m_floor": 50, "m_fallback": 1, "min_share": 0.2}
    assert cfg["params"]["seed_role"] == "calibration" and cfg["params"]["parts"] == ["floor"]
    assert cfg["inputs"] == {} and cfg["stream"] == "E1"
    assert cfg["seeds"] == list(range(1000, 1040))


def test_pilot_floor_smoke_is_small():
    cells = run.grid(_effective("pilot_floor.yaml", smoke=True))
    assert len(cells) == 4 and {c["pair_seed"] for c in cells} == {1000, 1020}


# --------------------------------------------------------------------------------------
# Pilot, part 2: the timing grid
# --------------------------------------------------------------------------------------


def test_pilot_timing_measures_a_grid_of_works_per_sampler_and_n():
    cfg = _effective("pilot_timing.yaml")
    cells = run.grid(cfg)
    assert {c["kind"] for c in cells} == {"timing"}
    for sampler in ("sa", "tabu", "swap_sa"):
        for n in (16, 24):
            works = [c[c["scaled"]] for c in cells if c["sampler"] == sampler and c["n"] == n]
            assert len(works) >= 12, (sampler, n)  # a grid, not a handful of points
            assert works == sorted(set(works)) and works[0] >= 1
            assert works[-1] / works[0] >= 15  # wide enough to bracket 1 s on another machine
            ratios = [b / a for a, b in pairwise(works)]
            assert max(ratios) <= 1.6  # fine enough: a chosen median is within ~25 % of the target


def test_pilot_timing_follows_the_protocol():
    cfg = _effective("pilot_timing.yaml")
    t = cfg["params"]["timing"]
    assert t["target_s"] == 1.0 and len(t["solver_seeds"]) >= 3  # R10: at least 3 repeats
    assert all(10_000 <= s < 20_000 for s in t["solver_seeds"])  # calibration solver seeds
    p = cfg["params"]
    assert p["seed_role"] == "calibration" and p["parts"] == ["timing"] and cfg["inputs"] == {}
    pairs = {entry["n"]: entry["pair_seeds"] for entry in p["instances"]}
    assert pairs[16] == list(range(1000, 1020)) and pairs[24] == list(range(1020, 1040))


def test_pilot_timing_scales_the_reads_of_tabu_each_one_plain_search():
    """§12 B1: the work unit of tabu is reads (``num_restarts = 0``), the tenure is max(5, N // 4); the
    grid is sized like the grids of SA and swap-SA, so that one call of about 1 s is bracketed."""
    cfg = _effective("pilot_timing.yaml")
    cells = [c for c in run.grid(cfg) if c["sampler"] == "tabu"]
    assert cells and {c["scaled"] for c in cells} == {"num_reads"}
    assert {c["num_restarts"] for c in cells} == {0}
    assert {(c["n"], c["tenure"]) for c in cells} == {(16, e1.tabu_tenure(16)), (24, e1.tabu_tenure(24))}
    assert (e1.tabu_tenure(16), e1.tabu_tenure(24)) == (5, 6)
    # ~2.4 ms per plain search at N = 16-24 on a quiet laptop (a loaded one: 6-7 ms): 1 s is 150-420 reads
    for n in (16, 24):
        works = [c["num_reads"] for c in cells if c["n"] == n]
        assert works[0] <= 100 and works[-1] >= 1500


def test_pilot_timing_fixes_the_sweeps_of_the_reference_configuration():
    """§12 B3: the work that the pilot sizes is the number of reads; the sweeps are the reference's, SA 1000 (the
    default of ``dwave-samplers``) and swap-SA 200 (the default of ``cif.heuristics.swap_sa``)."""
    from inspect import signature

    from cif.heuristics.swap_sa import sample_swap_sa

    params = signature(sample_swap_sa).parameters
    assert params["num_sweeps"].default == 200
    # the hot temperature by the 80 % acceptance calibration and T_cold = 0.01·T_hot are the defaults of the sampler,
    # and the cell passes only the ratio: the reference is what swap-SA does when nothing is said
    assert params["accept_start"].default == e1.ACCEPT_START == 0.8
    assert params["cold_ratio"].default == e1.COLD_RATIO == e1.REFERENCE["swap_sa"]["cold_ratio"] == 0.01
    cells = run.grid(_effective("pilot_timing.yaml"))
    assert {c["num_sweeps"] for c in cells if c["sampler"] == "sa"} == {1000}
    assert {c["num_sweeps"] for c in cells if c["sampler"] == "swap_sa"} == {200}
    assert {c["scaled"] for c in cells if c["sampler"] in ("sa", "swap_sa")} == {"num_reads"}


def test_pilot_timing_reference_configurations_are_the_defaults():
    p = _effective("pilot_timing.yaml")["params"]["timing"]["samplers"]
    ref = {s["name"]: s.get("reference") for s in p}
    assert ref["sa"] == {"penalty": 100, "beta": "default"}  # SELECTION_PENALTY_MULTIPLIER, dwave β
    assert ref["tabu"] == {"penalty": 100} and ref["swap_sa"] == {"cold_ratio": 0.01}  # §12 B3
    assert ref == e1.REFERENCE  # what the work is sized on is the reference of the plan
    from cif.qubo.builder import SELECTION_PENALTY_MULTIPLIER

    assert ref["sa"]["penalty"] == SELECTION_PENALTY_MULTIPLIER
    cells = [c for c in run.grid(_effective("pilot_timing.yaml")) if c["sampler"] == "swap_sa"]
    assert {c["cold_ratio"] for c in cells} == {0.01}  # the exploratory 0.2 is not timed: same work


def test_pilot_timing_smoke_is_small():
    cells = run.grid(_effective("pilot_timing.yaml", smoke=True))
    assert len(cells) == 12 and {c["n"] for c in cells} == {16, 24}


@pytest.mark.parametrize("name", ["pilot_floor.yaml", "pilot_timing.yaml", "attribution.yaml"])
def test_every_job_is_single_threaded_by_the_config_not_by_a_default(name):
    """Plan §6, R8: one thread per solver is stated in the config (the runner limits BLAS/OpenMP to it)."""
    for smoke in (False, True):
        assert _effective(name, smoke=smoke)["threads"] == 1


def test_the_timing_and_floor_configs_agree_on_the_calibration_instances():
    floor = _effective("pilot_floor.yaml")["params"]
    timing = _effective("pilot_timing.yaml")["params"]
    for key in ("instances", "h_levels", "lam", "w_min", "w_max", "ms", "scip"):
        assert floor[key] == timing[key], key


# --------------------------------------------------------------------------------------
# The confirmatory grid
# --------------------------------------------------------------------------------------


def test_the_attribution_config_holds_the_registered_pilot_and_refuses_without_it():
    # after the pilot (§12, 2026-10-12): the block is the one of the registered report ...
    report = json.loads((CONFIGS.parents[1] / "results" / "E1" / "pilot" / "pilot_report.json").read_text(encoding="utf-8"))
    cfg = _effective("attribution.yaml")
    values = pilot.pilot_values(report)
    assert cfg["params"]["pilot"]["m0"] == values["m0"] == 1
    assert cfg["params"]["pilot"]["work"] == values["work"]
    assert cfg["params"]["pilot"]["source"]["run_ids"] == ["20261011-205903-E1-pilot_floor-9da201",
                                                           "20261012-113427-E1-pilot_timing-e3f5db"]
    assert cfg["params"]["seed_role"] == "test"
    # ... and a config without it still refuses to build the grid
    empty = {**cfg, "params": {**cfg["params"], "pilot": {"source": None, "m0": None, "work": None}}}
    with pytest.raises(ValueError, match="pilot"):
        run.grid(empty)


def test_the_filled_grid_is_the_design_of_the_plan(filled_cells):
    cells = filled_cells
    # pairs × levels × (10 SA + 5 tabu + 2 swap-SA: the reference and the exploratory level, §12 B3) × seeds
    assert len(cells) == 80 * 2 * 17 * 10 == 27_200
    pairs = {(c["n"], c["pair_seed"]) for c in cells}
    assert len(pairs) == 80
    assert sorted(s for n, s in pairs if n == 16) == list(range(2000, 2040))
    assert sorted(s for n, s in pairs if n == 24) == list(range(2040, 2080))
    assert {c["solver_seed"] for c in cells} == set(range(20000, 20010))
    assert {c["h"] for c in cells} == {0.15, 0.45}
    assert all(2000 <= c["pair_seed"] < 3000 and 20000 <= c["solver_seed"] < 30000 for c in cells)


def test_the_nested_grid_of_section_7_3(filled_cells):
    configs = Counter((c["sampler"], c["penalty"], c["beta"], c["cold_ratio"]) for c in filled_cells)
    expected = {("sa", float(p), b, None) for p in (1, 3, 10, 30, 100) for b in ("default", "objective")}
    expected |= {("tabu", float(p), None, None) for p in (1, 3, 10, 30, 100)}
    expected |= {("swap_sa", None, None, 0.01), ("swap_sa", None, None, 0.2)}
    assert set(configs) == expected and len(set(configs.values())) == 1
    assert set(configs.values()) == {80 * 2 * 10}


def test_the_grid_holds_the_references_and_one_exploratory_level(filled_cells):
    """§12 B3: the references of the three samplers are in the grid, and the only level that is not in the
    plan is the swap-SA ratio 0.2; the roles follow from ``config_role``."""
    roles = Counter(e1.config_role(c["sampler"], c["penalty"], c["beta"], c["cold_ratio"]) for c in filled_cells
                    if c["pair_seed"] == 2000 and c["h"] == 0.15 and c["solver_seed"] == 20000)
    assert roles == {"reference": 3, "exploratory": 1, "grid": 13}
    refs = {(c["sampler"], c["penalty"], c["beta"], c["cold_ratio"]) for c in filled_cells
            if e1.config_role(c["sampler"], c["penalty"], c["beta"], c["cold_ratio"]) == "reference"}
    assert refs == {("sa", 100.0, "default", None), ("tabu", 100.0, None, None), ("swap_sa", None, None, 0.01)}
    explore = [c for c in filled_cells if e1.config_role(c["sampler"], c["penalty"], c["beta"], c["cold_ratio"])
               == "exploratory"]
    assert len(explore) == 80 * 2 * 10 == 1_600  # "+1 600 ячеек к 25 600" of the consultation
    assert {c["cold_ratio"] for c in explore} == {e1.EXPLORATORY_COLD_RATIO}


def test_the_filled_config_records_the_pilot_values(filled_cfg):
    pilot_block = filled_cfg["params"]["pilot"]
    assert pilot_block["m0"] == 50 and len(pilot_block["work"]) == 6
    assert pilot_block["source"]["valid_for_section_12"] is True
    assert pilot_block["source"]["run_ids"][0].endswith("pilot_floor-aaaaaa")
    sa16 = next(w for w in pilot_block["work"] if (w["sampler"], w["n"]) == ("sa", 16))
    assert sa16 == {"sampler": "sa", "n": 16, "num_reads": 2016, "num_sweeps": 1000}
    assert filled_cfg["params"]["ms"] == [1, 5, 20, 50, 200] and 50 in filled_cfg["params"]["ms"]


@pytest.mark.parametrize("n_shards", [5, 8, 10, 16, 20, 40, 80])
def test_every_shard_of_the_confirmatory_grid_holds_whole_pairs(filled_cells, n_shards):
    """``--shard j --n-shards S`` runs ``cells[j::S]``: with S dividing the 80 pairs each pair, both
    levels, all configurations and seeds, is in one shard, so its oracle is computed once."""
    owner: dict[tuple[int, int], int] = {}
    sizes = []
    for j in range(n_shards):
        shard = filled_cells[j::n_shards]
        sizes.append(len(shard))
        for c in shard:
            assert owner.setdefault((c["n"], c["pair_seed"]), j) == j
    assert len(owner) == 80 and len(set(sizes)) == 1 and sum(sizes) == len(filled_cells)


def test_a_shard_count_that_does_not_divide_the_pairs_still_covers_every_cell_once(filled_cells):
    keys = [run.cell_key(c) for j in range(7) for c in filled_cells[j::7]]
    assert len(keys) == len(filled_cells) == len(set(keys))


def test_the_smoke_mode_of_the_confirmatory_config_touches_no_test_seed():
    """``--smoke`` on this config must be safe before the pilot: calibration seeds, hand-made work."""
    cfg = _effective("attribution.yaml", smoke=True)
    assert cfg["params"]["seed_role"] == "calibration"
    cells = run.grid(cfg)
    assert len(cells) == 2 * 2 * 5  # pairs × levels × (SA ×2 β, tabu, swap-SA ×2 levels) at one penalty, one seed
    assert {c["cold_ratio"] for c in cells if c["sampler"] == "swap_sa"} == {0.01, 0.2}
    assert all(1000 <= c["pair_seed"] < 2000 and 10_000 <= c["solver_seed"] < 20_000 for c in cells)
    assert cfg["seeds"] == [1000, 1020, 10000]
    assert {c["sampler"] for c in cells} == {"sa", "tabu", "swap_sa"}


def test_dry_runs_of_the_shipped_configs(capsys):
    for name, smoke in (("pilot_floor.yaml", False), ("pilot_timing.yaml", False),
                        ("attribution.yaml", True)):
        argv = ["--config", str(CONFIGS / name), "--dry-run"] + (["--smoke"] if smoke else [])
        assert run.main(argv) == 0
    out = capsys.readouterr().out
    assert "E1/pilot_floor (full): 80 cells" in out and "E1/attribution (smoke)" in out


def test_the_existing_smoke_config_still_works_with_flat_work():
    cells = run.grid(_effective("attribution_smoke.yaml"))
    assert cells and all(c["pair_seed"] < 2000 for c in cells)


# --------------------------------------------------------------------------------------
# The README: the commands in it match the configs
# --------------------------------------------------------------------------------------


def test_the_readme_commands_match_the_configs(filled_cells):
    import re

    text = (CONFIGS / "README.md").read_text(encoding="utf-8")
    # the commands for gpu-07 go through the generic launcher, on the physical cores, pinned, with a result
    # directory outside the repository and the unpacked bundle (the variable OUT is defined there)
    commands = re.findall(r"^scripts/run_server\.sh configs/E1/(\w+)\.yaml (.*)$", text, flags=re.MULTILINE)
    assert [name for name, _ in commands] == ["pilot_timing", "pilot_timing", "attribution"]  # rehearsal, run, run
    for _, rest in commands:
        assert "--cores 0-31 --pin" in rest, rest
        assert re.search(r"--out-dir \$OUT/\w+", rest), rest
    assert sum("--smoke" in rest for _, rest in commands) == 1  # one rehearsal, in a directory of its own
    assert len({re.search(r"--out-dir (\S+)", rest).group(1) for _, rest in commands}) == 3
    (out_line,) = re.findall(r"^OUT=(\S+)", text, flags=re.MULTILINE)
    assert out_line.startswith("$HOME/") and "$PWD" not in out_line and not out_line.startswith((".", "runs"))
    for name, _ in commands:
        assert (CONFIGS / f"{name}.yaml").is_file()
    assert "xargs" not in text  # the launcher replaced the hand-made shard loops
    # sizes in the text are those of the configs
    sizes = {"92 cells": len(run.grid(_effective("pilot_timing.yaml"))),
             "27200 cells": len(filled_cells),
             "20 cells, calibration seeds": len(run.grid(_effective("attribution.yaml", smoke=True)))}
    for phrase, n in sizes.items():
        assert phrase in text and n == int(phrase.split()[0]), phrase
    assert "16 шардов по 5 пар" in text and len(filled_cells) % 16 == 0 and 80 % 16 == 0
    for path in re.findall(r"configs/E1/(\w+\.yaml)", text):
        assert (CONFIGS / path).is_file(), path
    for name in ("pilot_floor.yaml", "pilot_timing.yaml", "attribution.yaml", "analysis.yaml"):
        assert name in text
    assert "20261011-205903-E1-pilot_floor-9da201" in text  # the registered floor run is the first --run of the report


@pytest.mark.parametrize("name", ["pilot_timing.yaml", "attribution.yaml"])
def test_the_launch_blocks_respect_the_machine_of_the_plan(name, filled_cells):
    """§12 (2026-10-11): gpu-07 is shared; at most 16 single-thread jobs (half of the 32 physical cores); the
    timing stages are the pilot's and the confirmatory grid (the call time is reported)."""
    launch = _effective(name)["launch"]
    assert 1 <= launch["jobs"] <= 16 and launch["timing"] is True
    n_cells = len(filled_cells) if name == "attribution.yaml" else len(run.grid(_effective(name)))
    assert 1 <= launch["shards"] <= n_cells
    if name == "pilot_timing.yaml":
        assert launch["shards"] == n_cells  # one timing cell per shard
    if name == "attribution.yaml":
        assert 80 % launch["shards"] == 0  # whole pairs per shard: the oracle of an instance once per shard


FLOOR_RUN = CONFIGS.parents[1] / "runs" / "20261011-205903-E1-pilot_floor-9da201"


@pytest.mark.skipif(not FLOOR_RUN.is_dir(), reason="the registered floor run is not in this checkout")
def test_the_shipped_floor_config_is_the_config_of_the_registered_floor_run():
    """The floor part is registered (§12 [пилот] m0) and is not run again: its config must stay what produced
    the run, to the hash (an added key, even a launcher hint, would break the link)."""
    from cif.experiments import registry

    cfg = _effective("pilot_floor.yaml")
    registered = json.loads((FLOOR_RUN / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((FLOOR_RUN / "manifest.json").read_text(encoding="utf-8"))
    assert cfg == registered and registry.config_hash(cfg) == manifest["config_hash"] == "9da2018d5f2c"


def test_the_launcher_plans_the_stages_of_the_pilot(tmp_path, capsys):
    """The generic launcher of E2 takes E1 configs as they are (no job is started by ``plan``)."""
    from cif.experiments import launch

    for name, cells in (("pilot_timing.yaml", 92), ("pilot_floor.yaml", 80)):  # the floor has no launch block: --jobs
        assert launch.main(["plan", "--config", str(CONFIGS / name), "--cores", "0-3", "--jobs", "2",
                            "--out-dir", str(tmp_path / name)]) == 0
        out = capsys.readouterr().out
        assert f"E1/{name[:-5]}: {cells} cells" in out and "grid " in out


# --------------------------------------------------------------------------------------
# The analysis settings
# --------------------------------------------------------------------------------------


def test_the_analysis_settings_are_valid_and_fit_the_confirmatory_design():
    from cif.experiments import e1_analysis as ana

    settings = ana.load_settings(CONFIGS / "analysis.yaml")
    assert settings["tau"] == 1e-3 and settings["alpha"] == 0.05 and settings["share_ms"] == [1, 50]
    assert settings["bootstrap"]["n_boot"] >= 9999 and isinstance(settings["bootstrap"]["seed"], int)
    assert settings["h1c"]["p_value"] is None  # supplied by the E2 analysis, not guessed here
    cfg = _effective("attribution.yaml")
    spec = {s["name"]: s for s in cfg["params"]["samplers"]}
    reference = settings["models"]["reference"]
    assert set(reference) == set(spec)
    assert reference["sa"]["penalty"] in spec["sa"]["penalty"] and reference["sa"]["beta"] in spec["sa"]["beta"]
    assert reference["tabu"]["penalty"] in spec["tabu"]["penalty"]
    assert reference["swap_sa"]["cold_ratio"] in spec["swap_sa"]["cold_ratio"]
    # §12 B3: the references are the plan's, they are those the work was sized on, and nothing is chosen yet
    assert reference == e1.REFERENCE
    timed = {s["name"]: s.get("reference") for s in _effective("pilot_timing.yaml")["params"]["timing"]["samplers"]}
    assert reference == timed
    assert settings["models"]["chosen_after_grid"] == []
    assert settings["models"]["responses"] == ["E_samp", "E_prec"]  # §12 B2
    assert settings["share_ms"][0] in cfg["params"]["ms"] and settings["share_ms"][1] in cfg["params"]["ms"]


def test_the_pool_size_of_h1a_is_the_value_of_the_pilot_in_the_analysis_settings():
    """§12 [пилот] m0 (2026-10-11): 3 of the 80 calibration instances have E_form(50) > tau, 3.75 % < 20 %, so
    m0 = 1; the analysis settings carry it, and it is one of the pool sizes of the grid."""
    from cif.experiments import e1_analysis as ana

    settings = ana.load_settings(CONFIGS / "analysis.yaml")
    assert settings["h1a"] == {"m0": 1}
    assert settings["h1a"]["m0"] in _effective("attribution.yaml")["params"]["ms"]
    cfg = _effective("attribution.yaml")
    assert ana.resolve_m0(cfg, settings, cfg["params"]["ms"]) == (1, "run config params.pilot.m0")  # filled block
    cfg = {**cfg, "params": {**cfg["params"], "pilot": {"source": None, "m0": None, "work": None}}}
    assert ana.resolve_m0(cfg, settings, cfg["params"]["ms"]) == (1, "analysis settings h1a.m0")  # an empty block
    filled = {**cfg, "params": {**cfg["params"], "pilot": {"m0": 1, "work": []}}}
    assert ana.resolve_m0(filled, settings, cfg["params"]["ms"]) == (1, "run config params.pilot.m0")
    clash = {**cfg, "params": {**cfg["params"], "pilot": {"m0": 50, "work": []}}}
    with pytest.raises(ValueError, match="contradicts"):  # a pilot that decided otherwise must change the settings
        ana.resolve_m0(clash, settings, cfg["params"]["ms"])


# --------------------------------------------------------------------------------------
# Writing the pilot block
# --------------------------------------------------------------------------------------


def test_the_pilot_block_replaces_only_the_lines_between_the_markers(attribution_text):
    values = pilot.pilot_values(_fake_report())
    filled = pilot.apply_pilot_block(attribution_text, values)
    before, after = attribution_text.splitlines(), filled.splitlines()
    first = next(i for i, ln in enumerate(before) if ">>> pilot" in ln)
    last = next(i for i, ln in enumerate(before) if "<<< pilot" in ln)
    assert after[: first + 1] == before[: first + 1]
    tail = len(before) - last
    assert after[-tail:] == before[last:]
    block = after[first + 1 : len(after) - tail]
    assert sum(1 for ln in block if "sampler:" in ln) == 6  # six work lines (and the provenance)


def test_applying_the_block_twice_changes_nothing_more(attribution_text):
    values = pilot.pilot_values(_fake_report())
    once = pilot.apply_pilot_block(attribution_text, values)
    assert pilot.apply_pilot_block(once, values) == once
    other = pilot.apply_pilot_block(once, pilot.pilot_values(_fake_report(m0=1)))
    assert yaml.safe_load(other)["params"]["pilot"]["m0"] == 1


def test_the_rendered_block_is_valid_yaml_with_one_line_per_work_entry():
    values = pilot.pilot_values(_fake_report())
    text = pilot.render_pilot_block(values, "  ")
    assert all(line.startswith("  ") for line in text.splitlines())
    assert sum(1 for line in text.splitlines() if "sampler:" in line) == 6
    assert yaml.safe_load(text)["pilot"] == values


@pytest.mark.parametrize("bad", ["params: {}\n", "# >>> pilot\n# >>> pilot\n# <<< pilot\n",
                                 "# <<< pilot\n# >>> pilot\n"])
def test_the_markers_must_be_there_once_and_in_order(bad):
    with pytest.raises(ValueError, match="pilot"):
        pilot.apply_pilot_block(bad, {"m0": 1})


def test_make_config_writes_and_validates(tmp_path, capsys):
    config = tmp_path / "attribution.yaml"
    config.write_text((CONFIGS / "attribution.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    report = tmp_path / "pilot_report.json"
    report.write_text(json.dumps(_fake_report()), encoding="utf-8")
    assert pilot.main(["make-config", "--report", str(report), "--config", str(config)]) == 0
    assert "27200 cells" in capsys.readouterr().out
    block = yaml.safe_load(config.read_text(encoding="utf-8"))["params"]["pilot"]
    assert block["m0"] == 50 and block["source"]["report"] == "pilot_report.json"
    assert len(block["source"]["report_sha256"]) == 64
    assert len(run.grid(run.effective_config(run.load_config(config)))) == 27_200


def test_make_config_refuses_a_report_that_is_not_valid_unless_told_to(tmp_path):
    config = tmp_path / "attribution.yaml"
    original = (CONFIGS / "attribution.yaml").read_text(encoding="utf-8")
    config.write_text(original, encoding="utf-8")
    report = tmp_path / "bad.json"
    report.write_text(json.dumps(_fake_report(valid=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot values"):
        pilot.main(["make-config", "--report", str(report), "--config", str(config)])
    assert config.read_text(encoding="utf-8") == original  # untouched
    # with --allow-invalid the block is written but says so, and the test-role grid then refuses it
    with pytest.raises(ValueError, match="not valid for §12"):
        pilot.main(["make-config", "--report", str(report), "--config", str(config), "--allow-invalid"])
    assert config.read_text(encoding="utf-8") == original  # validation failed: still untouched


def test_make_config_refuses_a_config_the_values_do_not_fit(tmp_path):
    config = tmp_path / "attribution.yaml"
    text = (CONFIGS / "attribution.yaml").read_text(encoding="utf-8")
    config.write_text(text, encoding="utf-8")
    report = _fake_report()
    del report["work"]["24"]  # the pilot knows nothing about n = 24
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="n=24"):
        pilot.main(["make-config", "--report", str(path), "--config", str(config)])
    assert config.read_text(encoding="utf-8") == text


def test_e1_attribution_still_reads_the_block_it_is_given(filled_cfg):
    """Cells of the filled grid carry no work; ``run_cell`` looks it up by sampler and N."""
    spec = e1._batch_spec(filled_cfg["params"], "tabu", 24)
    assert (spec["num_reads"], spec["num_restarts"], spec["tenure"]) == (424, 0, 6)
