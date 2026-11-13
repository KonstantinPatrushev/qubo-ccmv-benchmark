"""The shipped E3 configs (``configs/E3``): what the plan fixes, the run-server rules, the smoke grids.

The full grids are checked against the REAL trading calendars of the datasets (shape and dates only: the
index of the parquet files; no price, no return and no strategy is read or computed), and skipped where the
data is not at hand. The smoke grids run on synthetic panels.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cif.data.provenance import verify_provenance
from cif.experiments import e3_analysis as an
from cif.experiments import e3_assemble as asmb
from cif.experiments import e3_data as d
from cif.experiments import e3_walkforward as e3
from cif.experiments import ingest, run

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "E3"
NAMES = ("pilot", "pilot_operators", "ff49", "us", "us_sens_irregular")
CONFIRMATORY = ("ff49", "us", "us_sens_irregular")
PILOTS = ("pilot", "pilot_operators")  # calibration seeds, the pilot period of FF49
AT_LAMBDA_F = ("pilot_operators", *CONFIRMATORY)  # wait for the lambda_f that the pilot chooses
DATA = ROOT / "data" / "processed"
#: the folds the plan expects ("около 630", "около 155", "~110") on the calendars of the datasets of 2026-10-08
CELLS = {"pilot": 672, "pilot_operators": 224, "ff49": 1272, "us": 155, "us_sens_irregular": 155}
FOLDS = {"pilot": 112, "pilot_operators": 112, "ff49": 636, "us": 155, "us_sens_irregular": 155}


@pytest.fixture(autouse=True)
def _fresh_caches():
    d.clear_caches()
    yield
    d.clear_caches()


def _raw(name: str) -> dict:
    return run.load_config(CONFIGS / f"{name}.yaml")


def _cfg(name: str, *, smoke: bool = False) -> dict:
    return run.effective_config(_raw(name), smoke=smoke)


def _filled(name: str, lam: float = 10.0) -> dict:
    """The full config with the pilot's lambda_f put in (in memory: the files keep ``null``)."""
    cfg = _cfg(name)
    if cfg["params"].get("lambda_f", 1) is None and not cfg["params"].get("lambdas"):
        cfg["params"]["lambda_f"] = lam
    return cfg


def _has(path: str) -> bool:
    return (DATA / path).exists()


def _skip_without_the_data(name: str) -> None:
    """The runner re-hashes the data files of every input before it starts, a smoke run on the synthetic market too.
    The data sets are not redistributed (the public snapshot carries only their provenance), and without their files
    such a run cannot start: the test skips. A changed file is no reason to skip: the test runs, and fails."""
    for path in run.resolve_inputs(_raw(name)).values():
        if path.suffix == ".json" and path.is_file():
            rep = verify_provenance(path, repo_root=ROOT)
            if rep["missing"] and not rep["mismatch"]:
                pytest.skip(f"the data files of {path.relative_to(ROOT)} are not in this checkout")


# --------------------------------------------------------------------------------------------
# What the plan fixes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_config_is_an_e3_experiment_with_explicit_seeds(name):
    raw = _raw(name)
    assert raw["stream"] == "E3" and raw["experiment"] == "cif.experiments.e3_walkforward"
    p = raw["params"]
    lo, hi = e3.SEED_RANGES[p["seed_role"]]
    assert lo <= p["solver_seed"] < hi and raw["seeds"] == [p["solver_seed"]]
    assert p["seed_role"] == ("calibration" if name in PILOTS else "test")
    assert p["market"]["train_days"] == 756 and p["market"]["test_days"] == 21  # plan §3


def test_the_markets_and_periods_are_those_of_the_plan():
    ff, us = _raw("ff49")["params"], _raw("us")["params"]
    assert ff["market"]["source"] == "ff49" and ff["market"]["u"] == 0.20
    assert ff["period"] == {"name": "confirmatory", "start": "1973-01-01", "end": "2025-12-31"}
    assert _raw("pilot")["params"]["period"] == {"name": "pilot", "start": "1963-07-01", "end": "1972-12-31"}
    assert us["market"]["source"] == "sp500_pit" and us["market"]["u"] == 0.10
    assert us["market"]["n_universe"] == 100 and us["period"]["end"] == "2025-12-31"
    assert _raw("us_sens_irregular")["params"]["market"]["variant"] == "no_irregular_ratio"
    assert us["market"]["variant"] == "base"
    # the variant is the only difference of the two US configs (same seeds, period, lambda_f, K rule)
    a, b = copy.deepcopy(_raw("us")), copy.deepcopy(_raw("us_sens_irregular"))
    for c in (a, b):
        c.pop("name"), c["params"]["market"].pop("variant")
    assert a == b


def test_arms_ranks_and_budgets_follow_the_plan():
    ff, us, pilot = (_raw(n)["params"] for n in ("ff49", "us", "pilot"))
    assert ff["arms"] == ["MV", "GMV"] and ff["n_ranks"] == 32  # ranks 2-33 in both arms
    assert us["arms"] == ["MV"] and us["n_ranks"] == 1  # US: MV, rank 2 only
    assert pilot["arms"] == ["MV", "GMV"] and pilot["lambdas"] == [5.0, 10.0, 20.0] and pilot["n_ranks"] == 1
    for p in (ff, us, pilot):
        assert p["optimum"]["time_limit_s"] == 7200.0  # "считать до закрытия (предел 2 ч)"
        assert p["kbest"]["time_limit_s"] == 600.0 and p["kbest"]["card_mode"] == "eq"
        assert p["kbest"]["abs_gap_rel"] == 1e-6
    assert ff["kbest"]["boundary_check"] and us["kbest"]["boundary_check"]
    s = ff["solvers"]  # (в): SCIP 10 s, tabu / SA / LS 1 s end to end, on the MV arm
    assert s["arms"] == ["MV"] and s["scip"]["budget_s"] == 10.0
    assert all(s[n]["budget_s"] == 1.0 for n in ("tabu", "sa", "ls"))
    assert us["solvers"] is None and us["operators"] is None and pilot["solvers"] is None
    ops = ff["operators"]  # (б) and (а): the levels, R = 30, at most 66 000 refinements per level
    assert ops["swap"]["levels"] == [1, 2, 5, 10, 25, 50, 100] == ops["perturb"]["levels"]
    assert ops["swap"]["n_replicas"] == 30 and ops["swap"]["max_steps"] == 200
    assert ops["swap"]["max_restarts"] == 10  # 11 chains: the first and up to 10 restarts


PILOT_REPORT = ROOT / "runs" / "20261011-232931-E3-pilot-acc916" / "pilot_report.json"


def _pilot_yaml() -> dict:
    """The values the registered pilot report (§12, 2026-10-12) put into the configs."""
    return json.loads(PILOT_REPORT.read_text(encoding="utf-8"))["yaml"]


@pytest.mark.parametrize("name", AT_LAMBDA_F)
def test_the_values_of_the_pilot_are_in_the_config_and_the_grid_refuses_without_them(name):
    # after the pilot (§12, 2026-10-12): lambda_f is the value of the registered pilot report ...
    assert _raw(name)["params"]["lambda_f"] == _pilot_yaml()["lambda_f"]
    # ... and a config without it still refuses to build its grid
    raw = copy.deepcopy(_raw(name))
    raw["params"]["lambda_f"] = None
    with pytest.raises(ValueError, match=r"pilot|пилот"):
        e3.cells(run.effective_config(raw))


def test_the_pilot_of_the_operators_is_the_operators_of_the_confirmation_on_the_pilot_folds_at_lambda_f():
    po, pilot, ff = (_raw(n) for n in ("pilot_operators", "pilot", "ff49"))
    p = po["params"]
    # plan §2 (б): "L — уровни, достигнутые хотя бы на 80 % фолдов пилотного периода" — the pilot folds
    assert p["market"] == pilot["params"]["market"] and p["period"] == pilot["params"]["period"]
    assert p["optimum"] == pilot["params"]["optimum"]
    # L by arm ("L по плечам"): both arms, at the lambda_f the pilot has chosen (one value, filled in by hand)
    assert p["arms"] == ["MV", "GMV"] and p["operators"]["arms"] == ["MV", "GMV"]
    assert p["lambda_f"] == ff["params"]["lambda_f"] == _pilot_yaml()["lambda_f"] and p["lambdas"] is None
    # a cell asks nothing but the operators: rank 2 (the status of a row), no real solver, no tie check
    assert p["n_ranks"] == 1 and p["solvers"] is None and p["kbest"]["boundary_check"] is False
    # "reached" means all replicas found with these levels and limits: the same as in the confirmation
    for op in ("swap", "perturb"):
        assert p["operators"][op] == ff["params"]["operators"][op]
    # calibration seeds, not the pilot's own: the two runs are independent
    assert p["seed_role"] == "calibration" and po["seeds"] != pilot["seeds"]
    assert po["name"] == "ff49_pilot_operators" and po["launch"]["timing"] is False


TUNED_REPORT = ROOT / "configs" / "E2" / "pilot" / "tuned.json"


def _tuned_roles(size: int = 100) -> dict:
    """The roles E2 tuned for instances of ``size`` assets: the committed report of the pilot (§12, 2026-10-12)."""
    report = json.loads(TUNED_REPORT.read_text(encoding="utf-8"))
    assert report["kind"] == "e2-tuned"
    entry = report["sizes"][str(size)]
    assert "borrowed_from" not in entry  # N = 100 was tuned on its own 8 non-trivial calibration instances (§12 A1)
    return entry["roles"]


def test_the_real_solvers_of_ff49_are_the_configurations_e2_tuned_for_n_100():
    # plan §2 H3c: "конфигурации, настроенные для N = 100". Which role feeds which solver of (в):
    roles, solvers = _tuned_roles(100), _raw("ff49")["params"]["solvers"]
    assert solvers["tuned"] is True
    # SCIP <- exact_rep, the representative of the exact class: its variant, and its symmetry switch
    # (E2 turns `usesymmetry: 0` into the SCIP parameter misc/usesymmetry = 0, `default` into no parameter);
    # the budget (10 s end to end) and the gap are E3's own
    exact = roles["exact_rep"]
    assert exact["name"] == "scip"
    assert {k: solvers["scip"][k] for k in ("formulation", "diag")} == {k: exact[k] for k in ("formulation", "diag")}
    expected = None if exact["usesymmetry"] == "default" else {"misc/usesymmetry": int(exact["usesymmetry"])}
    assert solvers["scip"]["scip_params"] == expected
    # tabu, SA, LS <- the roles of the same names (criterion PI+ at 10 s, rechecked at 60 s). Not qubo_rep and
    # swap_sa (the QUBO class of RQ2), not h1c_sa and h1c_tabu (the final gap at 10 s, the comparison of H1c)
    for name in ("tabu", "sa", "ls"):
        tuned = {k: v for k, v in roles[name].items() if k not in ("name", "id")}
        shipped = {k: v for k, v in solvers[name].items() if k != "budget_s"}  # the budget (1 s) is E3's own
        assert roles[name]["name"] == name
        assert shipped == tuned, name


def test_a_full_run_of_ff49_is_refused_without_tuned_solvers_before_any_work():
    cfg = _filled("ff49")
    assert cfg["params"]["solvers"]["tuned"] is True  # the shipped config holds the configurations of E2 ...
    cfg["params"]["solvers"]["tuned"] = False  # ... and the guard of a full run is still there
    cell = {"market": "ff49", "period": "confirmatory", "arm": "MV", "lam": 10.0, "fold": 0,
            "formation_date": "1972-12-29"}
    with pytest.raises(RuntimeError, match="tuned"):  # no data is touched: the check comes first
        e3.run_cell(cell, cfg)


# --------------------------------------------------------------------------------------------
# The run server: launch section, shards, bundle
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_launch_section_asks_for_the_jobs_the_shared_server_allows(name):
    raw = _raw(name)
    launch = raw["launch"]
    assert launch["jobs"] == 16  # plan §12 (2026-10-11): at most half of the 32 physical cores
    # a stage that times a real solver is a timing stage; certification alone is not
    assert launch["timing"] is (raw["params"].get("solvers") is not None)
    assert isinstance(launch["shards"], int) and launch["shards"] >= launch["jobs"]
    smoke = _cfg(name, smoke=True)["launch"]
    assert smoke["jobs"] <= 2 and smoke["shards"] <= 4


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.skipif(not _has("ff49/prices_vw.parquet") or not _has("sp500_pit/prices_adj.parquet"),
                    reason="the datasets are not at hand")
def test_the_full_grids_have_the_planned_size_and_the_shards_the_same_number_of_cells(name):
    cfg = _filled(name)
    cells = e3.cells(cfg)  # reads the trading calendar only
    assert len(cells) == CELLS[name]
    assert len({c["fold"] for c in cells}) == FOLDS[name]
    shards = cfg["launch"]["shards"]
    sizes = {len(cells[i::shards]) for i in range(shards)}
    assert len(sizes) == 1, sizes  # cells[i::S] gives every shard the same number of cells
    if len(cfg["params"]["arms"]) == 2:  # the arm is the outer loop: every shard has both arms
        assert all({c["arm"] for c in cells[i::shards]} == {"MV", "GMV"} for i in range(shards))
    assert len({run.cell_key(c) for c in cells}) == len(cells)


@pytest.mark.skipif(not _has("ff49/prices_vw.parquet"), reason="the FF49 dataset is not at hand")
def test_the_periods_of_ff49_sit_inside_the_dates_of_the_plan():
    for name, (lo, hi) in (("pilot", ("1963-07-01", "1972-12-31")), ("ff49", ("1973-01-01", "2025-12-31"))):
        p = _cfg(name)["params"]
        cal = d.market_calendar(p["market"])
        grid = d.period_grid(cal, p["period"]["start"], p["period"]["end"], 756, 21)
        first, last = cal[grid.first_test_row], cal[grid.row_end - 1]
        assert first >= pd.Timestamp(lo) and last <= pd.Timestamp(hi)
        assert cal[grid.first_test_row - 1] < pd.Timestamp(lo)  # the first day on or after
        assert grid.row_end + 21 > cal.searchsorted(pd.Timestamp(hi), side="right")  # maximal
        assert grid.row_start >= 0  # the training window needs data before the first test day only


@pytest.mark.skipif(not _has("sp500_pit/prices_adj.parquet"), reason="the S&P dataset is not at hand")
def test_the_us_period_starts_on_the_first_day_a_756_day_window_allows():
    p = _cfg("us")["params"]
    cal = d.market_calendar(p["market"])
    with pytest.raises(ValueError, match="history"):  # the plan's "2013-01" starts two days too early
        d.period_grid(cal, "2013-01-01", p["period"]["end"], 756, 21)
    grid = d.period_grid(cal, p["period"]["start"], p["period"]["end"], 756, 21)
    assert grid.row_start == 0 and str(cal[grid.first_test_row].date()) == p["period"]["start"]


@pytest.mark.parametrize("name", NAMES)
def test_the_inputs_are_provenances_of_the_datasets_the_config_reads(name):
    raw = _raw(name)
    for label, path in raw["inputs"].items():
        assert path.startswith("data/processed/") and path.endswith("/provenance.json"), label
    market = raw["params"]["market"]
    assert any(market["data_dir"] in path for path in raw["inputs"].values())
    if market["source"] == "sp500_pit":
        assert market["rf_path"].startswith("data/processed/riskfree_v2/")
        assert any("riskfree_v2" in path for path in raw["inputs"].values())


# --------------------------------------------------------------------------------------------
# Smoke grids: synthetic, small, calibration seeds
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_the_smoke_grid_is_synthetic_small_and_on_calibration_seeds(name):
    eff = _cfg(name, smoke=True)
    p = eff["params"]
    assert eff["mode"] == "smoke" and p["market"]["source"] == "synthetic"
    assert p["seed_role"] == "calibration" and 10_000 <= p["solver_seed"] < 20_000
    cells = run.grid(eff)  # validates the seeds, the lambdas and the unique keys
    assert 6 <= len(cells) <= 30
    assert {c["arm"] for c in cells} == set(p["arms"])
    sizes = {len(cells[i::eff["launch"]["shards"]]) for i in range(eff["launch"]["shards"])}
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("name", ["ff49", "us", "pilot_operators"])
def test_a_smoke_run_of_the_shipped_config_runs_on_the_synthetic_market(name, tmp_path):
    _skip_without_the_data(name)
    cells = len(run.grid(_cfg(name, smoke=True)))
    runs = tmp_path / "runs"
    assert run.main(["--config", str(CONFIGS / f"{name}.yaml"), "--smoke", "--runs-dir", str(runs)]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == cells and not any(r.get("error") for r in rows)
    assert {r["status"] for r in rows} <= {"ok", "degenerate"}
    if name == "ff49":  # the smoke grid exercises the solvers and the operators too
        mv = [r for r in rows if r["arm"] == "MV" and r["status"] == "ok"]
        assert mv and all(set(r["solvers"]) == {"scip", "tabu", "sa", "ls"} for r in mv)
        assert all(r["swap"] and r["perturb"] for r in mv)
    if name == "pilot_operators":  # the operators of both arms, and nothing else is asked of a cell
        ok = [r for r in rows if r["status"] == "ok"]
        assert {r["arm"] for r in ok} == {"MV", "GMV"} and all(r["swap"] and r["perturb"] for r in ok)
        assert not any("solvers" in r for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "smoke" and manifest["status"] == "completed"


def test_the_smoke_pilot_of_the_operators_gives_the_levels_of_both_arms(tmp_path):
    _skip_without_the_data("pilot_operators")
    cfg = _cfg("pilot_operators", smoke=True)
    runs = tmp_path / "runs"
    assert run.main(["--config", str(CONFIGS / "pilot_operators.yaml"), "--smoke", "--runs-dir", str(runs)]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    pilot = asmb.read_pilot_operators(cfg, asmb.load_rows(run_dir), run_id=run_dir.name)
    assert set(pilot.reach) == {("MV", "swap"), ("MV", "perturb"), ("GMV", "swap"), ("GMV", "perturb")}
    levels = cfg["params"]["operators"]["swap"]["levels"]
    assert pilot.reach[("GMV", "swap")][0] == tuple(float(g) for g in levels)
    assert pilot.lam == cfg["params"]["lambda_f"]


def test_the_smoke_pilot_feeds_the_pilot_report(tmp_path):
    _skip_without_the_data("pilot")
    from cif.experiments import e3_pilot as pilot

    cfg = _cfg("pilot", smoke=True)
    runs = tmp_path / "runs"
    assert run.main(["--config", str(CONFIGS / "pilot.yaml"), "--smoke", "--runs-dir", str(runs)]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    acfg = yaml.safe_load((CONFIGS / "analysis.yaml").read_text())
    acfg["stats"]["calibration"] = {"grid": [1, 5], "n_pseudo": 10, "n_boot": 29}
    acfg["stats"]["n_boot"] = 99
    report = pilot.pilot_report(cfg, asmb.load_rows(run_dir), acfg, confirmatory_days=2000)
    assert report["lambda_f"] in (5.0, 20.0) and report["margins"]["delta_sr"] > 0
    assert set(report["yaml"]) == {"lambda_f", "margins"}


def test_the_pilot_reads_the_length_of_the_confirmation_from_a_config_that_has_no_lambda_yet(tmp_path):
    from cif.experiments import e3_pilot as pilot

    # the shipped ff49.yaml as it stands on the day of the pilot (lambda_f: null, solvers not tuned yet),
    # with the synthetic market of its smoke block put in, so that no market data is read
    raw = copy.deepcopy(_raw("ff49"))
    smoke = _cfg("ff49", smoke=True)["params"]
    raw["params"]["market"], raw["params"]["period"] = smoke["market"], smoke["period"]
    raw["params"]["lambda_f"] = None  # the state of the day of the pilot, whatever the shipped value is now
    raw["params"]["solvers"]["tuned"] = False  # likewise: E2 tuned them after the pilot of E3
    path = tmp_path / "ff49_synthetic.yaml"
    path.write_text(yaml.safe_dump(raw))
    _, grid = e3.calendar_grid(run.effective_config(raw))
    assert pilot._confirmatory_days(path) == grid.n_folds * grid.test_days > 0
    with pytest.raises(ValueError, match="lambda_f"):  # the cells, unlike the calendar, do need it
        run.grid(run.effective_config(raw))


# --------------------------------------------------------------------------------------------
# The analysis config
# --------------------------------------------------------------------------------------------


def test_the_analysis_config_is_complete_and_holds_the_pilot_values():
    acfg = yaml.safe_load((CONFIGS / "analysis.yaml").read_text())
    sc = an.stats_config(acfg)
    # the numbers of the plan
    assert sc["n_boot"] == 4999 and sc["alpha"] == 0.05 and sc["fixed_blocks"] == [21, 63]
    assert sc["calibration"] == {"grid": [1, 5, 10, 21, 42, 63, 126], "n_pseudo": 1000, "n_boot": 499}
    assert sc["delta_sr_alt"] == [0.02, 0.05] and sc["h3c_solvers"] == ["tabu", "sa", "ls"]
    assert sc["g_star_bp"] == 10.0 and sc["coverage_min"] == 0.9 and sc["gamma"] == "lambda_f"
    margins = _pilot_yaml()["margins"]
    assert acfg["margins"] == {"sesoi": 0.0025, "delta_sr": margins["delta_sr"],
                               "delta_var": margins["delta_var"]}
    open_cfg = copy.deepcopy(acfg)
    open_cfg["margins"].update(delta_sr=None, delta_var=None)
    for arm in ("MV", "GMV"):
        an.margin_for(arm, acfg)  # the pilot values are in
        with pytest.raises(an.AnalysisError, match=r"pilot|пилот"):
            an.margin_for(arm, open_cfg)
    assert acfg["assembly"]["workers"] == 16  # the cap of the shared server
    passes = acfg["assembly"]["passes"]
    assert passes["gross"]["tc_bps"] == 0.0 and passes["net"]["tc_bps"] == 10.0  # "до издержек", main
    assert passes["net30"]["tc_bps"] == 30.0 and passes["gross_lag1"]["execution_lag"] == 1
    us = acfg["sources"]["us"]["assembly"]["passes"]
    assert us["gross_term30"]["terminal_return"] == -0.3 and us["gross_term100"]["terminal_return"] == -1.0
    assert {a for a in sc["net_passes"].values()} <= set(passes)
    assert set(sc["sensitivity_passes"].values()) <= set(passes) | set(us)


def test_the_analysis_config_names_the_configs_of_the_runs_and_placeholders_or_registered_ids():
    acfg = yaml.safe_load((CONFIGS / "analysis.yaml").read_text())
    assert set(acfg["sources"]) == {"pilot", "pilot_operators", "ff49", "us", "us_irregular"}
    for key, src in acfg["sources"].items():
        assert (ROOT / src["config"]).is_file(), key
        if src["run"].startswith("runs/RUN_ID_"):  # a placeholder that stops the stage
            continue
        # a run that exists was made with the config named here (its hash is in the manifest)
        run_dir = ROOT / src["run"]
        assert (run_dir / "manifest.json").is_file(), key
        assert asmb.check_lineage(run_dir, ROOT / src["config"])["stream"] == "E3", key
    assert acfg["sources"]["pilot_operators"]["config"] == "configs/E3/pilot_operators.yaml"
    assert set(acfg["assemblies"]) == {"ff49", "us", "us_irregular"}
    for key, path in acfg["assemblies"].items():  # a placeholder, or the registered assembly of that source
        if path.startswith("runs/RUN_ID_ASSEMBLE"):
            continue
        manifest = json.loads((ROOT / path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["stream"] == "E3" and manifest["mode"] == "assembly", key
        assert manifest["name"] == f"assemble-{key}" and manifest["status"] == "completed", key
    assert (ROOT / acfg["pilot"]["confirmatory_config"]).is_file() and acfg["pilot"]["min_binding_share"] == 0.7
    ops = acfg["sources"]["ff49"]["assembly"]["operators"]
    assert ops["swap"]["min_share"] == 0.8 and ops["swap"]["g_star_bp"] == 10.0  # L: 80 % of the folds, g*
    # L is read on the folds of the pilot run of the operators (the default); a list would override it
    assert ops["swap"]["levels_L"] == "pilot" == ops["perturb"]["levels_L"]


# --------------------------------------------------------------------------------------------
# The README: every command it shows names a module and a config that exist
# --------------------------------------------------------------------------------------------


def _readme_commands(section: str = "Order of runs") -> list[str]:
    """The commands of the bash blocks of one section of the README (continuations joined, comments out)."""
    text = (CONFIGS / "README.md").read_text(encoding="utf-8")
    start = text.index(f"## {section}")
    end = text.find("\n## ", start + 1)
    blocks = re.findall(r"```bash\n(.*?)```", text[start : end if end > 0 else None], flags=re.S)
    lines = []
    for block in blocks:
        joined = re.sub(r"\\\n\s*", " ", block)
        lines += [ln.strip() for ln in joined.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    return lines


def test_the_readme_commands_name_existing_modules_and_configs():
    import importlib

    commands = _readme_commands()
    assert any("scripts/run_server.sh" in c for c in commands)
    seen_modules = set()
    for command in commands:
        for module in re.findall(r"python(?: -u)? -m (cif\.[\w.]+)", command):
            seen_modules.add(module)
        for path in re.findall(r"\b(configs/E3/[\w.]+\.yaml)", command):
            assert (ROOT / path).is_file(), (command, path)
    assert {"cif.experiments.ingest", "cif.experiments.e3_pilot", "cif.experiments.e3_assemble",
            "cif.experiments.e3_analysis", "cif.experiments.launch"} <= seen_modules
    for module in seen_modules - {"cif.experiments.launch", "cif.data.provenance"}:
        assert hasattr(importlib.import_module(module), "main"), module


def test_the_readme_orders_the_commands_by_stage_and_states_the_server_rules():
    commands = _readme_commands()

    def first(token: str) -> int:
        return next(i for i, c in enumerate(commands) if token in c)

    stages = ["pilot.yaml", "cif.experiments.e3_pilot", "pilot_operators.yaml", "ff49.yaml", "us.yaml",
              "us_sens_irregular.yaml", "cif.experiments.e3_assemble", "cif.experiments.e3_analysis"]
    positions = [first(t) for t in stages]
    assert positions == sorted(positions), dict(zip(stages, positions, strict=True))
    text = (CONFIGS / "README.md").read_text(encoding="utf-8")
    for must in ("--cores 0-31 --pin", "16", "BUNDLE_SHA", "launch.timing", "shards", "data/raw/sp500_pit",
                 "--exclude-cores", "ingest", "gpu-07"):
        assert must in text, must


def test_the_readme_has_its_cost_and_reading_sections_filled():
    text = (CONFIGS / "README.md").read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in text
    for heading in ("## Cost", "## How the plan was read", "## Still open"):
        assert heading in text, heading
    cost = text[text.index("## Cost") : text.index("## How the plan was read")]
    assert "estimate" in cost.lower() and "CPU-h" in cost  # the numbers are marked as estimates


def test_a_smoke_shard_cannot_be_ingested_as_a_full_run(tmp_path):
    _skip_without_the_data("pilot")
    runs = tmp_path / "runs"
    out = tmp_path / "shards"
    path = CONFIGS / "pilot.yaml"
    for i in range(2):
        assert run.main(["--config", str(path), "--smoke", "--shard", str(i), "--n-shards", "2",
                         "--workers", "1", "--out", str(out / f"s{i}.jsonl")]) == 0
    # the ingest compares the config of the sessions with the local one: the smoke flag is part of its hash
    assert ingest.main(["--config", str(path), "--shards", *map(str, sorted(out.glob("s*.jsonl"))),
                        "--runs-dir", str(runs)]) == 1  # a smoke shard is not a full run: refused
