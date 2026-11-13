"""E3 pilot rules as code (``cif.experiments.e3_pilot``): the values the plan leaves to the pilot.

ANALYSIS_PLAN §3 (λ_f: the smallest of {5, 10, 20} at which ``s_cont,f > ⌈1/u⌉ + 1`` on at least 70 %
of the pilot folds, else 20), §2 H3a (``Δ_SR = SESOI/σ*``, ``δ = (0.005/λ_f)/σ²_*``) and §7.4 (the
power pilot: ρ of the rank-2 path and the optimum, the half-width of the 90 % interval rescaled by
``√(T_pilot/T_confirmatory)``). Synthetic markets only.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import yaml

from cif.experiments import e3_assemble as a
from cif.experiments import e3_data as d
from cif.experiments import e3_pilot as pilot
from cif.experiments import run
from cif.stats import rank_equivalence, sharpe_margin, variance_margin
from tests.e3_fixtures import base_config, synthetic_assembly


@pytest.fixture(autouse=True)
def _fresh_caches():
    d.clear_caches()
    yield
    d.clear_caches()


# --------------------------------------------------------------------------------------------
# The rule for λ_f
# --------------------------------------------------------------------------------------------


def test_lambda_is_the_smallest_candidate_that_binds_on_70_percent():
    shares = {5.0: (6, 10), 10.0: (7, 10), 20.0: (10, 10)}
    choice = pilot.choose_lambda(shares, min_share=0.7)
    assert choice.lambda_f == 10.0 and not choice.fallback  # 7/10 is "at least 70 %"
    assert choice.shares == {5.0: 0.6, 10.0: 0.7, 20.0: 1.0}
    assert pilot.choose_lambda({5.0: (7, 10), 10.0: (9, 10)}, 0.7).lambda_f == 5.0
    assert pilot.choose_lambda({5.0: (70, 100), 10.0: (99, 100)}, 0.7).lambda_f == 5.0


def test_without_a_binding_candidate_lambda_is_20_and_the_share_is_reported():
    choice = pilot.choose_lambda({5.0: (1, 10), 10.0: (2, 10), 20.0: (6, 10)}, 0.7)
    assert choice.lambda_f == 20.0 and choice.fallback is True
    assert choice.shares[20.0] == 0.6
    # the fallback is 20 even when 20 is not the last candidate of the config
    assert pilot.choose_lambda({5.0: (1, 10), 20.0: (2, 10), 40.0: (3, 10)}, 0.7).lambda_f == 20.0


def test_candidates_are_tried_in_increasing_order_whatever_the_dict_order():
    shares = {20.0: (10, 10), 5.0: (8, 10), 10.0: (9, 10)}
    assert pilot.choose_lambda(shares, 0.7).lambda_f == 5.0


def test_a_candidate_with_no_folds_cannot_bind():
    with pytest.raises(ValueError, match="no fold"):
        pilot.choose_lambda({5.0: (0, 0), 10.0: (1, 1)}, 0.7)


# --------------------------------------------------------------------------------------------
# σ*, Δ_SR, σ²_*, δ
# --------------------------------------------------------------------------------------------


def test_margins_are_made_of_the_volatility_and_the_variance_of_the_optimum_on_the_pilot():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0, solvers=())
    out = pilot.margins_from(asm, sesoi=0.0025)
    mv = a.fk_series(asm, "gross", "MV", "sharpe")
    gmv = a.fk_series(asm, "gross", "GMV", "logvar")
    sigma = mv.opt.std(ddof=1) * math.sqrt(252)
    assert out["sigma_star"] == pytest.approx(sigma)
    assert out["delta_sr"] == pytest.approx(sharpe_margin(sigma, 0.0025)) == pytest.approx(0.0025 / sigma)
    var = gmv.opt.var(ddof=1) * 252
    assert out["sigma2_star"] == pytest.approx(var)
    expected = variance_margin(5.0, var, 0.0025)
    assert out["delta_var"] == pytest.approx(expected.delta) == pytest.approx((0.005 / 5.0) / var)
    assert out["margin_log_variance"] == pytest.approx([expected.low, expected.high])
    assert out["n_days"] == len(mv.opt)


def test_a_variance_below_the_sesoi_is_reported_not_raised():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0, solvers=(), lam=1000.0)
    out = pilot.margins_from(asm, sesoi=0.0025)  # δ = (0.005/1000)/σ²_* is tiny: fine
    assert 0 < out["delta_var"] < 1
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0, solvers=(), lam=0.001)
    out = pilot.margins_from(asm, sesoi=0.0025)  # δ = 5/σ²_* > 1: log(1 − δ) does not exist
    assert out["delta_var"] is None and "must lie in (0, 1)" in out["delta_var_error"]
    assert out["delta_sr"] > 0


# --------------------------------------------------------------------------------------------
# The power pilot
# --------------------------------------------------------------------------------------------

STATS = {
    "seed": 20261201, "alpha": 0.05, "n_boot": 199,
    "calibration": {"grid": [1, 5, 10], "n_pseudo": 20, "n_boot": 49},
}


def test_power_pilot_rescales_the_halfwidth_to_the_length_of_the_confirmation():
    asm = synthetic_assembly(n_folds=80, drag=1e-4, solver_miss=0.0, solvers=())
    margins = pilot.margins_from(asm, sesoi=0.0025)
    out = pilot.power_pilot(asm, margins, STATS, confirmatory_days=4 * 80 * 21)
    for arm in ("MV", "GMV"):
        res = out[arm]
        series = a.fk_series(asm, "gross", arm, "sharpe" if arm == "MV" else "logvar", ranks=[2])
        assert res["rho"] == pytest.approx(np.corrcoef(series.opt, series.ranks[:, 0])[0, 1])
        assert res["n_days"] == len(series.opt) == 80 * 21
        low, high = res["halfwidth_pilot"], res["halfwidth_confirmatory"]
        assert low > 0 and high == pytest.approx(low * math.sqrt(res["n_days"] / (4 * 80 * 21)))
        assert high == pytest.approx(0.5 * low)  # four times as long: half the width
    mv = out["MV"]
    series = a.fk_series(asm, "gross", "MV", "sharpe", ranks=[2])
    direct = rank_equivalence(series.opt, series.ranks[:, 0], margins["delta_sr"], functional="sharpe",
                              block=mv["block"], n_boot=199, alpha=0.05,
                              seed=pilot.seed_for(STATS["seed"], "pilot", "MV", "power"))
    assert mv["halfwidth_pilot"] == pytest.approx(0.5 * (direct.ci_tost[1] - direct.ci_tost[0]))
    assert mv["expected_inconclusive"] == (mv["halfwidth_confirmatory"] > margins["delta_sr"])


def test_the_rule_flags_an_inconclusive_outcome_when_the_halfwidth_exceeds_the_margin():
    asm = synthetic_assembly(n_folds=40, drag=1e-4, solver_miss=0.0, solvers=(), noise=0.004)
    margins = pilot.margins_from(asm, sesoi=0.0025)
    tiny = dict(margins, delta_sr=1e-4)
    out = pilot.power_pilot(asm, tiny, STATS, confirmatory_days=40 * 21)
    assert out["MV"]["expected_inconclusive"] is True
    huge = dict(margins, delta_sr=50.0)
    assert pilot.power_pilot(asm, huge, STATS, confirmatory_days=40 * 21)["MV"][
        "expected_inconclusive"] is False


# --------------------------------------------------------------------------------------------
# End to end: the cells of a pilot, the report, the stage
# --------------------------------------------------------------------------------------------


def _pilot_config() -> dict:
    cfg = base_config(lambda_f=None, lambdas=[0.5, 5.0, 50.0], n_ranks=1)
    cfg["params"]["period"] = {"name": "pilot", "start": "2000-06-01", "end": "2000-07-15"}
    cfg["params"]["kbest"]["boundary_check"] = False
    return cfg


@pytest.fixture(scope="module")
def pilot_run(tmp_path_factory):
    d.clear_caches()
    tmp = tmp_path_factory.mktemp("e3pilot")
    cfg = _pilot_config()
    path = tmp / "pilot.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp / "runs"
    assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    return cfg, path, run_dir, tmp


def test_the_report_applies_the_rules_to_the_rows_of_the_pilot(pilot_run):
    cfg, _, run_dir, _ = pilot_run
    rows = a.load_rows(run_dir)
    report = pilot.pilot_report(cfg, rows, {"stats": STATS, "margins": {"sesoi": 0.0025}},
                                confirmatory_days=2000)
    rule = report["lambda_rule"]
    u = cfg["params"]["market"]["u"]
    for lam in (0.5, 5.0, 50.0):
        mv = [r for r in rows if r["arm"] == "MV" and r["lam"] == lam]
        n_bind = sum(1 for r in mv if r["s_cont"] > d.floor_cardinality(u))
        assert rule["binding_share"][str(lam)] == pytest.approx(n_bind / len(mv))
        assert rule["n_folds"][str(lam)] == len(mv)
    chosen = pilot.choose_lambda({float(k): (round(v * rule["n_folds"][k]), rule["n_folds"][k])
                                  for k, v in rule["binding_share"].items()}, 0.7)
    assert report["lambda_f"] == chosen.lambda_f == rule["lambda_f"]
    assert report["margins"]["delta_sr"] > 0 and report["power"]["MV"]["rho"] > 0.5
    assert report["n_folds_Fk"] <= report["n_folds"] and report["plan"]["version"] == "v2.4"
    snippet = report["yaml"]
    assert snippet["lambda_f"] == report["lambda_f"]
    assert snippet["margins"]["delta_sr"] == report["margins"]["delta_sr"]
    json.dumps(report, allow_nan=False)


def test_the_report_uses_the_rows_of_the_chosen_lambda_for_the_margins(pilot_run):
    cfg, _, run_dir, _ = pilot_run
    rows = a.load_rows(run_dir)
    report = pilot.pilot_report(cfg, rows, {"stats": STATS, "margins": {"sesoi": 0.0025}},
                                confirmatory_days=2000)
    asm = a.assemble(cfg, rows, {"gross": {"tc_bps": 0.0, "paths": "core"}}, lam=report["lambda_f"])
    sigma = a.fk_series(asm, "gross", "MV", "sharpe").opt.std(ddof=1) * math.sqrt(252)
    assert report["margins"]["sigma_star"] == pytest.approx(sigma)
    assert report["margins"]["lambda_f"] == report["lambda_f"]


@pytest.mark.parametrize("conf_lambda", [None, 5.0])
def test_the_pilot_stage_registers_a_run_and_writes_the_report(pilot_run, tmp_path, conf_lambda):
    # the confirmatory configs have lambda_f: null until the pilot has chosen it (plan §3): the length of
    # the confirmation, all the report needs of them, cannot depend on the value this stage computes
    _, path, run_dir, _ = pilot_run
    conf = base_config(lambda_f=conf_lambda, n_ranks=3)
    conf_path = tmp_path / "confirmatory.yaml"
    conf_path.write_text(yaml.safe_dump(conf))
    acfg = {
        "stream": "E3", "name": "analysis_test",
        "sources": {"pilot": {"config": str(path), "run": str(run_dir)}},
        "margins": {"sesoi": 0.0025, "delta_sr": None, "delta_var": None},
        "stats": STATS,
        "pilot": {"min_binding_share": 0.7, "confirmatory_config": str(conf_path)},
    }
    apath = tmp_path / "analysis.yaml"
    apath.write_text(yaml.safe_dump(acfg))
    runs = tmp_path / "runs"
    assert pilot.main(["--analysis-config", str(apath), "--source", "pilot", "--runs-dir", str(runs),
                       "--allow-dirty"]) == 0
    (out,) = [p for p in runs.iterdir() if p.is_dir()]
    report = json.loads((out / "pilot_report.json").read_text())
    assert report["source_run_id"] == run_dir.name and len(report["rows_sha256"]) == 64
    table = d.get_fold_table(conf["params"]["market"], conf["params"]["period"])
    assert report["power"]["confirmatory_days"] == len(table.folds) * 5
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["stream"] == "E3"
    assert "pilot_report.json" in manifest["artifacts"]


def test_a_pilot_report_needs_a_pilot_style_run(pilot_run):
    cfg = copy.deepcopy(pilot_run[0])
    cfg["params"]["lambdas"] = None
    cfg["params"]["lambda_f"] = 5.0
    with pytest.raises(pilot.PilotError, match="lambdas"):
        pilot.pilot_report(cfg, [], {"stats": STATS, "margins": {}}, confirmatory_days=100)
