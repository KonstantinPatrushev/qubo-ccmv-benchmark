"""E3 descriptive operators (б) and (а): reach, doses, replica paths and their differences.

ANALYSIS_PLAN §2 RQ3 "(б) Замены носителя — описательно" and "(а) Возмущение весов": the shares of
the folds where a level is reached, the distribution of the realised gaps, the share of failures, the
turnover of the replicas against the optimum and the differences (Sharpe, MV; log variance, GMV) at the
reached levels, per replica and averaged. ``L`` is read on the folds of the PILOT period, per arm
("L — уровни, достигнутые хотя бы на 80 % фолдов пилотного периода"); ``L_a = {g ≤ g*}``; the fold sets
``F_a`` and ``F_b`` are the CONFIRMATORY folds where every level of ``L_a`` / ``L`` is reached
(:mod:`cif.backtest.fold_sets`). No tests: estimates with intervals.

The synthetic market of ``tests/e3_fixtures.py`` has a pilot period (9 folds, the cells of
``PILOT_CONFIG``) before the confirmatory one (17 folds). On it the operators behave differently by arm,
which is what makes the tests discriminate: the swap chains never reach a level on the MV pilot folds
(an empty ``L``), reach 100 and 500 bp on the GMV ones; the perturbation reaches 10 bp on 2 of 9 MV
pilot folds, but on 12 of 17 confirmatory ones (so ``L`` read on the confirmatory folds would differ).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from cif.backtest.fold_sets import confirmatory_levels, fold_sets, level_sets
from cif.experiments import e3_analysis as an
from cif.experiments import e3_assemble as a
from cif.experiments import e3_data as d
from cif.experiments import run
from cif.stats import joint_contrasts
from tests.e3_fixtures import base_config

ARMS = ("MV", "GMV")
OPS = ("swap", "perturb")
PILOT_PERIOD = {"name": "pilot", "start": "2000-03-27", "end": "2000-05-26"}  # 9 folds
CONF_PERIOD = {"name": "confirmatory", "start": "2000-06-01", "end": "2000-09-30"}  # 17 folds
OPERATORS = {
    "arms": ["MV", "GMV"],
    "swap": {"levels": [5, 100, 500], "n_replicas": 3, "max_steps": 60, "max_restarts": 3},
    "perturb": {"levels": [10, 500], "n_replicas": 3, "max_draws": 2000},
}
PASSES = {"gross": {"tc_bps": 0.0, "paths": "core"}, "net": {"tc_bps": 10.0, "paths": "core"}}
OPS_ASSEMBLY = {
    "arms": ["MV", "GMV"],
    # the shares are those of the synthetic pilot folds (module docstring); the plan's 80 % for the swap,
    # 20 % for the perturbation, whose reach on 9 pilot folds is 2/9 and 1/9
    "swap": {"min_share": 0.8, "g_star_bp": 10.0, "levels_L": None, "passes": ["gross", "net"]},
    "perturb": {"min_share": 0.2, "g_star_bp": 10.0, "levels_L": None, "passes": ["gross", "net"]},
}
ACFG = {
    "margins": {"sesoi": 0.0025, "delta_sr": 0.3, "delta_var": 0.15},
    "stats": {
        "seed": 20261201, "alpha": 0.05, "n_boot": 199,
        "calibration": {"grid": [1, 5], "n_pseudo": 10, "n_boot": 29},
        "fixed_blocks": [5], "delta_sr_alt": [0.2], "g_star_bp": 10.0, "h3c_solvers": ["tabu"],
        "coverage_min": 0.9, "net_passes": {"cost_0": "gross", "cost_10": "net"},
        "sensitivity_passes": {}, "net_bootstrap": {"n_boot": 49}, "gamma": "lambda_f",
    },
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    d.clear_caches()
    yield
    d.clear_caches()


def _config(name: str, period: dict) -> dict:
    cfg = base_config(operators=copy.deepcopy(OPERATORS))
    cfg["name"] = name
    cfg["params"]["period"] = dict(period)
    return cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A registered run of the operators on the pilot period and one on the confirmatory period, the
    pilot's reach (:func:`cif.experiments.e3_assemble.pilot_operators`) and the assembly that reads
    ``L`` from it."""
    d.clear_caches()
    tmp = tmp_path_factory.mktemp("e3ops")
    runs = tmp / "runs"
    made = {}
    for key, name, period in (("pilot", "syn_pilot_ops", PILOT_PERIOD), ("conf", "syn_conf_ops", CONF_PERIOD)):
        cfg = _config(name, period)
        path = tmp / f"{key}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
        (run_dir,) = [p for p in runs.glob(f"*-E3-{name}-*") if p.is_dir()]
        made[key] = SimpleNamespace(cfg=cfg, path=path, run_dir=run_dir, rows=a.load_rows(run_dir))
    pilot = a.read_pilot_operators(made["pilot"].cfg, made["pilot"].rows, run_id=made["pilot"].run_dir.name)
    asm = a.assemble(made["conf"].cfg, made["conf"].rows, PASSES, operators=OPS_ASSEMBLY,
                     pilot_operators=pilot)
    return SimpleNamespace(pilot_made=made["pilot"], conf=made["conf"], pilot=pilot, asm=asm, tmp=tmp,
                           runs=runs)


def _weights(rep: dict, n: int) -> np.ndarray:
    w = np.zeros(n)
    w[rep["support"]] = rep["w"]
    return w


def _n_folds(cfg: dict) -> int:
    return len(d.get_fold_table(cfg["params"]["market"], cfg["params"]["period"]).folds)


def _reached(rows: list[dict], arm: str, op: str, n_folds: int):
    """``(levels, reached[fold, level], ran[fold])`` of one arm and operator, from the rows of a run."""
    mine = [r for r in rows if r["arm"] == arm and r.get(op)]
    levels = [float(lv["gap_target_bp"]) for lv in mine[0][op]]
    reached = np.zeros((n_folds, len(levels)), dtype=bool)
    ran = np.zeros(n_folds, dtype=bool)
    for r in mine:
        ran[r["fold"]] = True
        reached[r["fold"]] = [lv["status"] == "reached" for lv in r[op]]
    return levels, reached, ran


# --------------------------------------------------------------------------------------------
# Reach and gaps (from the rows)
# --------------------------------------------------------------------------------------------


def test_the_reach_table_counts_the_levels_of_the_rows(built):
    ops, rows = built.asm.operators, built.conf.rows
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm and "swap" in r]
        for op in OPS:
            sub = ops.reach[(ops.reach["arm"] == arm) & (ops.reach["operator"] == op)].set_index("level")
            levels = [lv["gap_target_bp"] for lv in mine[0][op]]
            assert list(sub.index) == levels
            for j, g in enumerate(levels):
                statuses = [r[op][j]["status"] for r in mine]
                assert sub.loc[g, "n_ran"] == len(mine)
                assert sub.loc[g, "n_reached"] == statuses.count("reached")
                assert sub.loc[g, "share_reached"] == pytest.approx(statuses.count("reached") / len(mine))
                assert sub.loc[g, "n_failed"] == statuses.count("failed")
                assert sub.loc[g, "n_unreachable"] == statuses.count("unreachable")
    assert (ops.gaps["gap_bp"] > 0).all()
    swap_gaps = ops.gaps[ops.gaps["operator"] == "swap"]
    for lv, grp in swap_gaps.groupby("level"):
        assert (grp["gap_bp"] >= lv - 1e-9).all() and (grp["gap_bp"] <= 1.1 * lv + 1e-9).all()


def test_operators_run_on_the_gmv_arm_with_the_gap_of_that_arm(built):
    # the gap of the GMV arm is (λ/2)·Δ(wᵀΣw)·1e4 (plan, minor 5): a replica's gap is measured against the
    # GMV optimum, whose f* is the row's
    rows = [r for r in built.conf.rows if r["arm"] == "GMV" and r.get("swap")]
    assert rows, "the GMV arm runs the operators"
    from cif.backtest.injection import gmv_problem

    table = d.get_fold_table(built.conf.cfg["params"]["market"], built.conf.cfg["params"]["period"])
    row = next(r for r in rows if any(lv["status"] == "reached" for lv in r["swap"]))
    rec = table.folds[row["fold"]]
    problem = gmv_problem(rec.problem(lam=row["lam"], u=table.market.u, cardinality=row["K_used"]))
    lv = next(x for x in row["swap"] if x["status"] == "reached")
    for rep in lv["replicas"]:
        w = _weights(rep, problem.n)
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-8)
        assert math.isclose((problem.objective_value(w) - row["f_star"]) * 1e4, rep["gap_bp"], abs_tol=1e-5)


# --------------------------------------------------------------------------------------------
# L: read on the pilot folds, per arm; the fold sets: on the confirmatory folds
# --------------------------------------------------------------------------------------------


def test_the_fixture_has_an_empty_L_and_one_that_the_confirmatory_folds_would_change(built):
    n_pilot, n_conf = _n_folds(built.pilot_made.cfg), _n_folds(built.conf.cfg)
    empty, changed = [], []
    for arm in ARMS:
        for op in OPS:
            share = OPS_ASSEMBLY[op]["min_share"]
            pl, pr, pran = _reached(built.pilot_made.rows, arm, op, n_pilot)
            cl, cr, cran = _reached(built.conf.rows, arm, op, n_conf)
            from_pilot = confirmatory_levels(pr[pran], pl, share)
            if not from_pilot:
                empty.append((arm, op))
            if from_pilot != confirmatory_levels(cr[cran], cl, share):
                changed.append((arm, op))
    assert empty and changed, "the synthetic outcome of the operators changed: choose other levels"


def test_L_is_read_on_the_pilot_folds_and_the_fold_sets_on_the_confirmatory_ones(built):
    n_pilot, n_conf = _n_folds(built.pilot_made.cfg), _n_folds(built.conf.cfg)
    for arm in ARMS:
        for op in OPS:
            sets = built.asm.operators.sets[f"{arm}:{op}"]
            pl, pr, pran = _reached(built.pilot_made.rows, arm, op, n_pilot)
            expected_l = confirmatory_levels(pr[pran], pl, OPS_ASSEMBLY[op]["min_share"])
            assert tuple(sets["L"]) == expected_l
            cl, cr, _ = _reached(built.conf.rows, arm, op, n_conf)
            ls = level_sets(expected_l, 10.0)
            fs = fold_sets(cr, cl, ls)
            np.testing.assert_array_equal(sets["mask_a"], fs.mask_a)
            np.testing.assert_array_equal(sets["mask_b"], fs.mask_b)
            assert tuple(sets["L_a"]) == ls.levels_a
            assert sets["L_source"] == "pilot" and sets["pilot_run"] == built.pilot.run_id
            assert sets["pilot_n_ran"] == int(pran.sum())
            np.testing.assert_allclose(sets["pilot_share"], pr[pran].mean(axis=0))
    assert built.asm.operators.sets["GMV:swap"]["mask_b"].sum() > 1, "the GMV swap has F_b folds to estimate on"


def test_an_empty_L_leaves_the_fold_sets_empty_not_full(built):
    sets = built.asm.operators.sets["MV:swap"]
    assert tuple(sets["L"]) == () and not sets["mask_a"].any() and not sets["mask_b"].any()
    entry = an.analyse_operators(built.asm, ACFG)["MV:swap"]
    assert entry["levels"] == [] and "no level" in entry["note"]


def test_the_pilot_is_read_for_the_arm_of_the_operator(built):
    # the same operator, two arms, two L
    swap = {arm: built.asm.operators.sets[f"{arm}:swap"]["L"] for arm in ARMS}
    assert swap["MV"] != swap["GMV"]


def test_levels_L_in_the_config_override_the_pilot_per_operator_and_per_arm(built):
    cfg, rows = built.conf.cfg, built.conf.rows
    ops = copy.deepcopy(OPS_ASSEMBLY)
    ops["swap"]["levels_L"] = [500]  # both arms
    ops["perturb"]["levels_L"] = {"GMV": [10]}  # GMV from the config, MV from the pilot
    for op in OPS:
        ops[op]["passes"] = ["gross"]
    asm = a.assemble(cfg, rows, {"gross": PASSES["gross"]}, operators=ops, pilot_operators=built.pilot)
    sets = asm.operators.sets
    assert tuple(sets["MV:swap"]["L"]) == tuple(sets["GMV:swap"]["L"]) == (500.0,)
    assert sets["MV:swap"]["L_source"] == sets["GMV:swap"]["L_source"] == "config"
    assert tuple(sets["GMV:perturb"]["L"]) == (10.0,) and sets["GMV:perturb"]["L_source"] == "config"
    assert sets["MV:perturb"]["L_source"] == "pilot"
    assert tuple(sets["MV:perturb"]["L"]) == tuple(built.asm.operators.sets["MV:perturb"]["L"])
    assert "pilot_run" not in sets["MV:swap"]


def test_with_every_level_given_no_pilot_is_needed(built):
    cfg, rows = built.conf.cfg, built.conf.rows
    ops = copy.deepcopy(OPS_ASSEMBLY)
    for op in OPS:
        ops[op].update(levels_L=[500], passes=["gross"])
    asm = a.assemble(cfg, rows, {"gross": PASSES["gross"]}, operators=ops)
    assert {s["L_source"] for s in asm.operators.sets.values()} == {"config"}


def test_without_a_pilot_and_without_levels_the_assembly_refuses(built):
    ops = copy.deepcopy(OPS_ASSEMBLY)
    for op in OPS:
        ops[op]["passes"] = ["gross"]
    with pytest.raises(a.AssemblyError, match="pilot"):
        a.assemble(built.conf.cfg, built.conf.rows, {"gross": PASSES["gross"]}, operators=ops)
    ops["swap"]["levels_L"] = "confirmatory"  # the folds of the run itself are not an option
    with pytest.raises(a.AssemblyError, match="levels_L"):
        a.assemble(built.conf.cfg, built.conf.rows, {"gross": PASSES["gross"]}, operators=ops,
                   pilot_operators=built.pilot)


def test_levels_L_given_must_be_levels_the_operator_ran(built):
    ops = copy.deepcopy(OPS_ASSEMBLY)
    ops["swap"].update(levels_L=[7], passes=["gross"])
    ops["perturb"].update(levels_L=[10], passes=["gross"])
    with pytest.raises(a.AssemblyError, match="levels_L"):
        a.assemble(built.conf.cfg, built.conf.rows, {"gross": PASSES["gross"]}, operators=ops,
                   pilot_operators=built.pilot)


# --------------------------------------------------------------------------------------------
# The pilot must be a pilot of the same operators: nothing else is accepted as the source of L
# --------------------------------------------------------------------------------------------


def _assemble_with(built, pilot):
    ops = copy.deepcopy(OPS_ASSEMBLY)
    for op in OPS:
        ops[op]["passes"] = ["gross"]
    return a.assemble(built.conf.cfg, built.conf.rows, {"gross": PASSES["gross"]}, operators=ops,
                      pilot_operators=pilot)


def test_the_pilot_describes_its_run(built):
    pilot = built.pilot
    assert pilot.run_id == built.pilot_made.run_dir.name and pilot.lam == 5.0
    assert set(pilot.reach) == {(arm, op) for arm in ARMS for op in OPS}
    levels, reached = pilot.reach[("GMV", "swap")]
    assert levels == (5.0, 100.0, 500.0) and reached.shape == (9, 3) and reached.dtype == bool
    assert pilot.test_end < d.get_fold_table(built.conf.cfg["params"]["market"],
                                             built.conf.cfg["params"]["period"]).folds[0].test_start


def test_a_pilot_at_another_lambda_is_refused(built):
    with pytest.raises(a.AssemblyError, match="lambda"):
        _assemble_with(built, dataclasses.replace(built.pilot, lam=10.0))


def test_a_pilot_on_another_market_is_refused(built):
    market = {**built.pilot.market, "u": 0.4}
    with pytest.raises(a.AssemblyError, match="market"):
        _assemble_with(built, dataclasses.replace(built.pilot, market=market))
    # the name of the market is a label, not a difference
    _assemble_with(built, dataclasses.replace(built.pilot, market={**built.pilot.market, "name": "other"}))


def test_a_pilot_that_does_not_precede_the_confirmation_is_refused(built):
    # the rows of the confirmatory run are not a pilot of it
    own = a.read_pilot_operators(built.conf.cfg, built.conf.rows, run_id=built.conf.run_dir.name)
    with pytest.raises(a.AssemblyError, match="before"):
        _assemble_with(built, own)


def test_a_pilot_with_other_levels_or_another_operator_is_refused(built):
    other = dict(built.pilot.reach)
    levels, reached = other[("GMV", "swap")]
    other[("GMV", "swap")] = (levels[:-1], reached[:, :-1])
    with pytest.raises(a.AssemblyError, match="levels"):
        _assemble_with(built, dataclasses.replace(built.pilot, reach=other))
    spec = copy.deepcopy(built.pilot.operators)
    spec["swap"]["n_replicas"] = 30  # "reached" means all replicas found: another number is another event
    with pytest.raises(a.AssemblyError, match="swap"):
        _assemble_with(built, dataclasses.replace(built.pilot, operators=spec))


def test_a_pilot_without_an_arm_of_the_confirmation_is_refused(built):
    mv_only = {k: v for k, v in built.pilot.reach.items() if k[0] == "MV"}
    with pytest.raises(a.AssemblyError, match="GMV"):
        _assemble_with(built, dataclasses.replace(built.pilot, reach=mv_only))


def test_the_pilot_of_the_operators_must_be_a_calibration_run(built):
    cfg = copy.deepcopy(built.pilot_made.cfg)
    cfg["params"].update(seed_role="test", solver_seed=20000)
    cfg["seeds"] = [20000]
    with pytest.raises(a.AssemblyError, match="calibration"):
        a.read_pilot_operators(cfg, built.pilot_made.rows, run_id="x")
    no_ops = copy.deepcopy(built.pilot_made.cfg)
    no_ops["params"]["operators"] = None
    with pytest.raises(a.AssemblyError, match="operators"):
        a.read_pilot_operators(no_ops, built.pilot_made.rows, run_id="x")


def test_rows_that_are_not_those_of_the_pilot_config_are_refused(built):
    with pytest.raises(a.AssemblyError):
        a.read_pilot_operators(built.pilot_made.cfg, built.conf.rows, run_id="x")


# --------------------------------------------------------------------------------------------
# The assembly stage reads the pilot run it is told about
# --------------------------------------------------------------------------------------------


def _analysis_config(built, tmp_path, *, pilot=True, pilot_config=None):
    sources = {"syn": {"config": str(built.conf.path), "run": str(built.conf.run_dir),
                       "assembly": {"operators": copy.deepcopy(OPS_ASSEMBLY)}}}
    if pilot:
        sources["pilot_operators"] = {"config": str(pilot_config or built.pilot_made.path),
                                      "run": str(built.pilot_made.run_dir)}
    acfg = {"stream": "E3", "name": "assemble_ops_test", "sources": sources,
            "assembly": {"workers": 1, "allow_incomplete": False, "passes": PASSES}}
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(acfg))
    return path


def test_the_assembly_stage_takes_L_from_the_registered_pilot_run_and_records_it(built, tmp_path):
    runs = tmp_path / "runs"
    path = _analysis_config(built, tmp_path)
    assert a.main(["--analysis-config", str(path), "--source", "syn", "--runs-dir", str(runs),
                   "--allow-dirty"]) == 0
    (out,) = [p for p in runs.iterdir() if p.is_dir()]
    loaded = a.load_assembly(out)
    sets = loaded.operators.sets
    assert {s["L_source"] for s in sets.values()} == {"pilot"}
    assert {s["pilot_run"] for s in sets.values()} == {built.pilot_made.run_dir.name}
    for key, s in sets.items():
        assert tuple(s["L"]) == tuple(built.asm.operators.sets[key]["L"])
    record = json.loads((out / "config.json").read_text())
    lineage = record["pilot_operators"]
    assert lineage["run_id"] == built.pilot_made.run_dir.name and len(lineage["rows_sha256"]) == 64
    assert lineage["config_hash"] == json.loads((built.pilot_made.run_dir / "manifest.json").read_text())["config_hash"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert any("pilot" in k for k in manifest["inputs"]), manifest["inputs"]  # the pilot rows are hashed too


def test_the_assembly_stage_without_a_pilot_source_refuses_before_any_run(built, tmp_path):
    runs = tmp_path / "runs"
    path = _analysis_config(built, tmp_path, pilot=False)
    with pytest.raises(a.AssemblyError, match="pilot_operators"):
        a.main(["--analysis-config", str(path), "--source", "syn", "--runs-dir", str(runs), "--allow-dirty"])
    assert not runs.exists() or not any(p.is_dir() for p in runs.iterdir())


def test_the_assembly_stage_checks_the_config_of_the_pilot_run(built, tmp_path):
    path = _analysis_config(built, tmp_path, pilot_config=built.conf.path)  # not the config of that run
    with pytest.raises(a.AssemblyError, match="config"):
        a.main(["--analysis-config", str(path), "--source", "syn", "--runs-dir", str(tmp_path / "runs"),
                "--allow-dirty"])


def test_a_stage_without_operators_needs_no_pilot(built, tmp_path):
    acfg = yaml.safe_load(_analysis_config(built, tmp_path, pilot=False).read_text())
    acfg["sources"]["syn"].pop("assembly")
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump(acfg))
    assert a.main(["--analysis-config", str(path), "--source", "syn", "--runs-dir", str(tmp_path / "r"),
                   "--allow-dirty"]) == 0


# --------------------------------------------------------------------------------------------
# Replica paths
# --------------------------------------------------------------------------------------------


def test_replica_paths_are_the_engine_on_the_stored_replicas(built):
    cfg, rows, asm = built.conf.cfg, built.conf.rows, built.asm
    table = d.get_fold_table(cfg["params"]["market"], cfg["params"]["period"])
    sets = asm.operators.sets["GMV:swap"]
    frame = asm.operators.paths["swap:gross"]
    assert "GMV:opt" in frame.columns
    level = sets["L"][0]
    cols = [c for c in frame.columns if c.startswith(f"GMV:g{int(level)}:")]
    assert len(cols) == 3
    by_cell = {(r["arm"], r["fold"]): r for r in rows}
    fold = int(np.flatnonzero(sets["mask_b"])[0])
    row = by_cell[("GMV", fold)]
    lv = next(x for x in row["swap"] if x["gap_target_bp"] == level)
    rec = table.folds[fold]
    assets = list(rec.assets)
    prices = table.market.prices
    sel = (asm.days["fold"] == fold).to_numpy()
    for j, col in enumerate(cols):
        w = _weights(lv["replicas"][j], row["n_assets"])
        window = prices.loc[rec.test_start : rec.test_end, assets].to_numpy()
        start = prices.loc[rec.formation_date, assets].to_numpy()
        value = (window / start) @ w
        manual = np.diff(np.concatenate([[1.0], value])) / np.concatenate([[1.0], value[:-1]])
        np.testing.assert_allclose(frame.loc[sel, col].to_numpy(), manual, atol=1e-9)
    # off F_b every replica holds the optimum
    off = (~sets["mask_b"][asm.days["fold"].to_numpy()])
    assert off.any()
    for col in cols:
        np.testing.assert_allclose(frame.loc[off, col].to_numpy(), frame.loc[off, "GMV:opt"].to_numpy(),
                                   atol=1e-12)


def test_the_net_pass_of_the_replicas_has_turnover_and_costs(built):
    stats = built.asm.operators.fold_stats["swap:net"]
    assert {"arm", "path", "fold", "turnover", "cost"} <= set(stats.columns)
    assert (stats["cost"] >= 0).all() and set(stats["arm"]) == {"MV", "GMV"}
    first = stats[stats["fold"] == 0]
    assert np.allclose(first["traded"], 1.0)  # bought from cash


def test_operators_round_trip_through_the_assembly_directory(built, tmp_path):
    asm = built.asm
    a.save_assembly(asm, tmp_path / "out")
    again = a.load_assembly(tmp_path / "out")
    ops, ops2 = asm.operators, again.operators
    pd.testing.assert_frame_equal(ops2.reach, ops.reach, check_dtype=False)
    pd.testing.assert_frame_equal(ops2.gaps, ops.gaps, check_dtype=False)
    for key in ops.paths:
        pd.testing.assert_frame_equal(ops2.paths[key], ops.paths[key])
    for key, s in ops.sets.items():
        np.testing.assert_array_equal(ops2.sets[key]["mask_b"], s["mask_b"])
        assert tuple(ops2.sets[key]["L"]) == tuple(s["L"])
        assert ops2.sets[key]["pilot_run"] == s["pilot_run"]
        np.testing.assert_allclose(ops2.sets[key]["pilot_share"], s["pilot_share"])


# --------------------------------------------------------------------------------------------
# The descriptive analysis
# --------------------------------------------------------------------------------------------


def _manual_joint(asm, arm, op, entry, functional):
    """The replica series of the levels of ``L`` on the days of ``F_b`` and the joint contrasts of the
    analysis, rebuilt from the stored paths."""
    sets = asm.operators.sets[f"{arm}:{op}"]
    mask = sets["mask_b"][asm.days["fold"].to_numpy()]
    frame = asm.operators.paths[f"{op}:gross"]
    shift = asm.days["rf"].to_numpy()[mask] if functional == "sharpe" else 0.0
    opt = frame[f"{arm}:opt"].to_numpy()[mask] - shift
    reps = {g: [frame[c].to_numpy()[mask] - shift for c in frame.columns if c.startswith(f"{arm}:g{int(g)}:")]
            for g in sets["L"]}
    series = np.column_stack([opt, *[r for g in sets["L"] for r in reps[g]]])
    weights = np.zeros((len(sets["L"]), series.shape[1]))
    weights[:, 0] = -1.0
    col = 1
    for q, g in enumerate(sets["L"]):
        weights[q, col : col + len(reps[g])] = 1.0 / len(reps[g])
        col += len(reps[g])
    joint = joint_contrasts(series, weights, functional=functional, block=entry["block"]["block"], n_boot=199,
                            seed=an.seed_for(20261201, "ops", arm, op, "bootstrap"))
    return mask, opt, reps, joint


def test_sharpe_differences_are_estimates_per_replica_and_averaged(built):
    asm = built.asm
    entry = an.analyse_operators(asm, ACFG)["MV:perturb"]
    sets = asm.operators.sets["MV:perturb"]
    mask, opt, reps, joint = _manual_joint(asm, "MV", "perturb", entry, "sharpe")

    def sharpe(x):
        return x.mean() / x.std(ddof=1) * math.sqrt(252)

    assert entry["L_source"] == "pilot" and entry["pilot_run"] == built.pilot.run_id
    level = sets["L"][0]
    row = next(x for x in entry["levels"] if x["level"] == level)
    diffs = [sharpe(r) - sharpe(opt) for r in reps[level]]
    np.testing.assert_allclose(row["per_replica"], diffs)
    assert row["mean_difference"] == pytest.approx(np.mean(diffs))
    assert row["n_replicas"] == 3 and row["n_days"] == int(mask.sum())
    # the interval is the joint bootstrap of the contrast "mean over replicas minus optimum"
    assert row["mean_difference"] == pytest.approx(float(joint.estimate[0]), abs=1e-9)
    assert row["ci"][0] < row["ci"][1] and row["se"] == pytest.approx(float(joint.se[0]))
    assert "outcome" not in row and "pvalue_tost" not in row  # descriptive: no tests


def test_gmv_differences_are_log_variance_ratios(built):
    asm = built.asm
    entry = an.analyse_operators(asm, ACFG)["GMV:swap"]
    sets = asm.operators.sets["GMV:swap"]
    _, opt, reps, joint = _manual_joint(asm, "GMV", "swap", entry, "logvar")
    assert [x["level"] for x in entry["levels"]] == list(sets["L"]) and len(sets["L"]) == 2
    for q, g in enumerate(sets["L"]):
        row = entry["levels"][q]
        diffs = [math.log(r.var(ddof=1)) - math.log(opt.var(ddof=1)) for r in reps[g]]
        np.testing.assert_allclose(row["per_replica"], diffs)
        assert row["mean_difference"] == pytest.approx(float(joint.estimate[q]), abs=1e-9)
        assert row["se"] == pytest.approx(float(joint.se[q]))


def test_turnover_of_the_replicas_is_reported_against_the_optimum(built):
    asm = built.asm
    entry = an.analyse_operators(asm, ACFG)["GMV:swap"]
    stats = asm.operators.fold_stats["swap:net"]
    mask = asm.operators.sets["GMV:swap"]["mask_b"]
    years = mask[asm.days["fold"].to_numpy()].sum() / 252
    sub = stats[(stats["arm"] == "GMV") & (stats["path"] == "opt") & stats["fold"].map(lambda f: bool(mask[f]))]
    row = entry["levels"][0]
    assert row["turnover_optimum"] == pytest.approx(sub["turnover"].sum() / years)
    assert row["turnover_replicas"] > 0 and row["turnover_excess"] == pytest.approx(
        row["turnover_replicas"] - row["turnover_optimum"])


def test_the_operator_tables_are_part_of_the_written_tables(built, tmp_path):
    out = an.analyse_markets({"ff49": built.asm}, ACFG)
    assert "ff49" in out["operators"] and "GMV:swap" in out["operators"]["ff49"]
    files = an.write_tables(out, tmp_path / "tables")
    names = {f.name for f in files}
    assert {"operator_reach.csv", "operator_gaps.csv", "operator_levels.csv", "operator_sets.csv"} <= names
    levels = pd.read_csv(tmp_path / "tables" / "operator_levels.csv")
    assert {"market", "arm", "operator", "level", "mean_difference", "ci_low", "ci_high",
            "turnover_excess"} <= set(levels.columns)
    sets = pd.read_csv(tmp_path / "tables" / "operator_sets.csv")
    # where L came from is part of the table: the pilot run and the share reached on its folds per level
    assert {"L_source", "pilot_run", "pilot_n_ran", "pilot_share"} <= set(sets.columns)
    assert set(sets["pilot_run"]) == {built.pilot.run_id} and set(sets["L_source"]) == {"pilot"}


def test_a_block_grid_wider_than_the_days_of_f_b_is_cut_to_the_blocks_that_fit(built):
    # the plan's grid reaches 126 days; the descriptive operators may have few days in F_b, and a block needs
    # two copies in the series: the analysis must not stop after the whole run on that
    asm = built.asm
    sets = asm.operators.sets["GMV:swap"]
    n_days = int(sets["mask_b"][asm.days["fold"].to_numpy()].sum())
    acfg = copy.deepcopy(ACFG)
    acfg["stats"]["calibration"]["grid"] = [1, 5, n_days, 126]
    entry = an.analyse_operators(asm, acfg)["GMV:swap"]
    assert entry["levels"], "the estimates are produced on the blocks that fit"
    fit = [b for b in (1, 5, n_days, 126) if 2 * b <= n_days]
    assert entry["block"]["grid"] == fit and entry["block"]["block"] in fit
    assert entry["block"]["grid_cut"] == [b for b in (1, 5, n_days, 126) if b not in fit]


def test_f_b_with_too_few_days_for_any_estimate_is_reported_not_raised(built):
    asm = built.asm
    sets = asm.operators.sets["GMV:swap"]
    saved = sets["mask_b"].copy()
    try:
        one = np.zeros_like(saved)
        one[int(np.flatnonzero(saved)[0])] = True  # one fold: 5 days, fewer than the 10 the tests need
        sets["mask_b"] = one
        entry = an.analyse_operators(asm, ACFG)["GMV:swap"]
    finally:
        sets["mask_b"] = saved
    assert entry["levels"] == [] and "days" in entry["note"]
