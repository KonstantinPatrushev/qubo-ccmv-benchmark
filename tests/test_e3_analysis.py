"""E3 analysis (``cif.experiments.e3_analysis``): H3a, H3b, H3c and their sensitivities.

ANALYSIS_PLAN §2 RQ3, §7.1, §7.4, §10. The statistics themselves are ``cif.stats``; what is tested
here is the hand-off: the right series (excess returns for the Sharpe functional, returns for the log
variance), the right fold set and doses, the margins of the pilot, the block calibrated per market and
arm, one seed per test, Holm over the two secondary arms, and the descriptive tables. The assemblies
come from a process with a known dose-response (``tests/e3_fixtures.py::synthetic_assembly``).
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from cif.experiments import e3_analysis as an
from cif.experiments import e3_assemble as a
from cif.stats import (
    calibrate_rank_block,
    holm_over_arms,
    rank_dose_response,
    rank_equivalence,
    rank_solver_dose,
)
from cif.stats.rq3_ranks import H3B_RANKS
from tests.e3_fixtures import synthetic_assembly

ACFG = {
    "stream": "E3",
    "name": "analysis_test",
    "margins": {"sesoi": 0.0025, "delta_sr": 0.3, "delta_var": 0.15},
    "stats": {
        "seed": 20261201, "alpha": 0.05, "n_boot": 299,
        "calibration": {"grid": [1, 5, 10], "n_pseudo": 20, "n_boot": 49},
        "fixed_blocks": [5, 21],
        "delta_sr_alt": [0.2, 0.5],
        "g_star_bp": 10.0,
        "h3c_solvers": ["tabu", "sa", "ls"],
        "coverage_min": 0.9,
        "net_passes": {"cost_0": "gross", "cost_10": "net"},
        "sensitivity_passes": {},
        "net_bootstrap": {"n_boot": 99},
        "gamma": "lambda_f",
    },
}


def _acfg(**stats) -> dict:
    cfg = copy.deepcopy(ACFG)
    cfg["stats"].update(stats)
    return cfg


@pytest.fixture(scope="module")
def asm():
    return synthetic_assembly(n_folds=100, drag=2.5e-4, vol_gain=0.12, solver_miss=0.8)


# --------------------------------------------------------------------------------------------
# Configuration, margins and seeds
# --------------------------------------------------------------------------------------------


def test_stats_config_requires_every_key():
    an.stats_config(ACFG)  # complete
    for key in ("seed", "n_boot", "calibration", "fixed_blocks", "delta_sr_alt", "h3c_solvers",
                "net_passes", "g_star_bp"):
        broken = copy.deepcopy(ACFG)
        del broken["stats"][key]
        with pytest.raises(ValueError, match=key):
            an.stats_config(broken)
    broken = copy.deepcopy(ACFG)
    del broken["stats"]["calibration"]["n_pseudo"]
    with pytest.raises(ValueError, match="n_pseudo"):
        an.stats_config(broken)


def test_the_margins_are_the_values_of_the_pilot():
    margin, info = an.margin_for("MV", ACFG)
    assert margin == 0.3 and info["delta_sr"] == 0.3
    margin, info = an.margin_for("GMV", ACFG)
    assert margin == pytest.approx((math.log1p(-0.15), math.log1p(0.15)))
    assert info["delta_var"] == 0.15
    for arm, key in (("MV", "delta_sr"), ("GMV", "delta_var")):
        unset = copy.deepcopy(ACFG)
        unset["margins"][key] = None
        with pytest.raises(an.AnalysisError, match="pilot"):
            an.margin_for(arm, unset)
    big = copy.deepcopy(ACFG)
    big["margins"]["delta_var"] = 1.2
    with pytest.raises(an.AnalysisError, match="delta_var"):
        an.margin_for("GMV", big)


def test_seeds_are_stable_and_depend_on_every_label():
    s = an.seed_for(20261201, "ff49", "MV", "h3a", "main")
    assert s == an.seed_for(20261201, "ff49", "MV", "h3a", "main")
    others = {an.seed_for(20261201, "ff49", "MV", "h3a", "block21"),
              an.seed_for(20261201, "ff49", "GMV", "h3a", "main"),
              an.seed_for(20261201, "us", "MV", "h3a", "main"),
              an.seed_for(20261201, "ff49", "MV", "h3b", "main"),
              an.seed_for(20261202, "ff49", "MV", "h3a", "main")}
    assert s not in others and len(others) == 5
    assert 0 <= s < 2**31


# --------------------------------------------------------------------------------------------
# H3a and the block
# --------------------------------------------------------------------------------------------


def test_h3a_is_the_rank_equivalence_of_the_excess_series_with_the_calibrated_block(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    series = a.fk_series(asm, "gross", "MV", "sharpe")
    sc = ACFG["stats"]
    cal = calibrate_rank_block(series.opt, series.ranks[:, 0],
                               seed=an.seed_for(sc["seed"], "ff49", "MV", "calibration", "main"),
                               functional="sharpe", block_grid=sc["calibration"]["grid"],
                               n_pseudo=sc["calibration"]["n_pseudo"],
                               n_boot=sc["calibration"]["n_boot"])
    assert res["block"]["block"] == cal.block and res["block"]["source"] == "calibration"
    assert res["block"]["rejection_rates"] == [float(x) for x in cal.rejection_rates]
    direct = rank_equivalence(series.opt, series.ranks[:, 0], 0.3, functional="sharpe",
                              block=cal.block, n_boot=299, alpha=0.05,
                              seed=an.seed_for(sc["seed"], "ff49", "MV", "h3a", "main"))
    h3a = res["h3a"]
    assert h3a["estimate"] == pytest.approx(direct.estimate, abs=1e-12)
    assert h3a["se"] == pytest.approx(direct.se, abs=1e-12)
    assert h3a["pvalue"] == pytest.approx(direct.pvalue) and h3a["pvalue_tost"] == pytest.approx(direct.pvalue_tost)
    assert h3a["outcome"] == direct.outcome and h3a["margin"] == [-0.3, 0.3]
    assert res["n_days"] == len(series.opt) and res["functional"] == "sharpe"


def test_h3a_of_the_gmv_arm_uses_returns_and_the_log_variance_margin(asm):
    res = an.analyse_arm(asm, "GMV", "ff49", ACFG)
    series = a.fk_series(asm, "gross", "GMV", "logvar")
    margin = (math.log1p(-0.15), math.log1p(0.15))
    direct = rank_equivalence(series.opt, series.ranks[:, 0], margin, functional="logvar",
                              block=res["block"]["block"], n_boot=299, alpha=0.05,
                              seed=an.seed_for(ACFG["stats"]["seed"], "ff49", "GMV", "h3a", "main"))
    assert res["functional"] == "logvar"
    assert res["h3a"]["estimate"] == pytest.approx(direct.estimate, abs=1e-12)
    assert res["h3a"]["margin"] == pytest.approx(list(margin))
    assert res["h3a"]["estimate"] > 0  # the rank-2 path is riskier than the optimum


def test_equivalent_when_rank_two_is_the_optimum_plus_noise_and_different_when_it_is_not():
    near = synthetic_assembly(n_folds=100, drag=0.0, vol_gain=0.0, solver_miss=0.0, seed=3)
    res = an.analyse_arm(near, "MV", "ff49", _acfg())
    assert res["h3a"]["equivalent"] and res["h3a"]["outcome"] in ("equivalent",
                                                                   "distinguishable_within_sesoi")
    far = synthetic_assembly(n_folds=100, drag=1e-3, vol_gain=0.0, solver_miss=0.0, seed=3)
    res = an.analyse_arm(far, "MV", "ff49", _acfg())
    assert not res["h3a"]["equivalent"] and res["h3a"]["significant"]
    assert res["h3a"]["outcome"] == "different" and res["h3a"]["estimate"] < 0


def test_a_fixed_block_replaces_the_calibration():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0)
    cfg = _acfg()
    cfg["stats"]["calibration"] = {**cfg["stats"]["calibration"], "fixed_block": 7}
    res = an.analyse_arm(asm, "MV", "ff49", cfg)
    assert res["block"]["block"] == 7 and res["block"]["source"] == "fixed"
    assert res["h3a"]["block"] == 7


# --------------------------------------------------------------------------------------------
# H3b
# --------------------------------------------------------------------------------------------


def test_h3b_is_the_slope_on_log_mean_dose_over_the_ranks_of_the_plan(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    h3b = res["h3b"]["log_dose"]
    series = a.fk_series(asm, "gross", "MV", "sharpe", ranks=H3B_RANKS)
    assert h3b["ranks"] == list(H3B_RANKS) and series.rank_ids == list(H3B_RANKS)
    doses = asm.doses["MV"].loc[asm.masks["common"], list(H3B_RANKS)].mean(axis=0).to_numpy()
    np.testing.assert_allclose(h3b["doses_bp"], doses)
    direct = rank_dose_response(
        series.opt, series.ranks, series.rank_ids, doses, functional="sharpe", block=res["block"]["block"],
        n_boot=299, alpha=0.05, seed=an.seed_for(ACFG["stats"]["seed"], "ff49", "MV", "h3b", "log_dose"))
    assert h3b["slope"]["estimate"] == pytest.approx(direct.slope.estimate, abs=1e-12)
    assert h3b["slope"]["ci_percentile"] == pytest.approx(list(direct.slope.ci_percentile))
    assert h3b["significant"] is direct.significant
    assert h3b["slope"]["estimate"] < 0 and h3b["significant"]  # the dose-response is built in
    assert len(h3b["per_rank"]) == 6 and h3b["per_rank"][0]["estimate"] > h3b["per_rank"][-1]["estimate"]
    x = np.log(doses)
    leverage = 1 / 6 + (x[0] - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
    assert res["h3b"]["leverage_rank2"] == pytest.approx(leverage)


def test_h3b_sensitivities_use_log_rank_and_all_ranks(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    by_rank = res["h3b"]["log_rank"]
    assert by_rank["regressor"] == "log_rank" and by_rank["slope"]["estimate"] < 0
    every = res["h3b"]["all_ranks"]
    assert every["ranks"] == list(range(2, 34)) and every["per_rank"] == []
    assert every["slope"]["estimate"] < 0


def test_the_gmv_slope_is_that_of_the_log_variance_ratio(asm):
    res = an.analyse_arm(asm, "GMV", "ff49", ACFG)
    assert res["h3b"]["log_dose"]["functional"] == "logvar"
    assert res["h3b"]["log_dose"]["slope"]["estimate"] > 0 and res["h3b"]["log_dose"]["significant"]
    assert "h3c" not in res  # the calibration of the solvers is an MV statement


def test_h3b_is_descriptive_when_a_mean_dose_is_not_positive():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0)
    asm.doses["MV"][3] = 0.0  # ties with the optimum on every fold: log ḡ_3 does not exist
    res = an.analyse_arm(asm, "MV", "ff49", _acfg())
    h3b = res["h3b"]
    assert h3b["descriptive"] is True and "log_dose" not in h3b and "doses" in h3b["reason"]
    assert h3b["log_rank"]["slope"]["estimate"] is not None  # the sensitivity on log r still stands


def test_us_style_assembly_has_h3a_only():
    us = synthetic_assembly(n_folds=60, arms=("MV",), n_ranks=1, solvers=(), market="us")
    res = an.analyse_arm(us, "MV", "us", _acfg())
    assert res["h3a"]["n_obs"] == res["n_days"] and "h3b" not in res and "h3c" not in res


# --------------------------------------------------------------------------------------------
# H3c
# --------------------------------------------------------------------------------------------


def test_h3c_calibrates_the_solvers_against_the_h3b_curve(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    h3c = res["h3c"]
    assert [s["name"] for s in h3c["solvers"]] == ["tabu", "sa", "ls"]
    mask = asm.masks["common"]
    solvers = asm.solvers
    for s in h3c["solvers"]:
        gaps = np.full(len(asm.folds), np.nan)
        sub = solvers[solvers["solver"] == s["name"]]
        gaps[sub["fold"].to_numpy()] = sub["gap_bp"].to_numpy()
        dose = rank_solver_dose(gaps, mask, res["h3b"]["log_dose"]["doses_bp"])
        assert s["dose"]["dose_bp"] == pytest.approx(dose.dose_bp)
        assert s["dose"]["inside_range"] is dose.inside_range
        assert s["dose"]["median_gap_bp"] == pytest.approx(dose.median_gap_bp)
        bins = s["rank_bins"]
        assert bins["labels"] == ["1", "2-5", "6-33", ">33"] and sum(bins["counts"]) == int(mask.sum())
        assert bins["counts"][2] == 0 and bins["counts"][0] > 0  # ranks 1 and 2-5 only
    assert h3c["descriptive"] is False and h3c["iut_passes"] in (True, False)
    for s in h3c["solvers"]:
        assert s["residual"]["estimate"] == pytest.approx(
            s["observed"]["estimate"] - s["predicted"], abs=1e-9)
        assert s["net"]["sharpe_difference"]["ci_percentile"][0] <= s["net"]["sharpe_difference"]["estimate"]


def test_h3c_is_descriptive_when_the_solvers_have_no_material_gap():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0)
    res = an.analyse_arm(asm, "MV", "ff49", _acfg())
    assert res["h3c"]["descriptive"] is True and res["h3c"]["iut_passes"] is None
    assert "fewer than two" in res["h3c"]["reason"]
    assert all(s["dose"]["dose_bp"] == 0.0 for s in res["h3c"]["solvers"])


def test_scip_is_not_among_the_solvers_of_h3c():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.8, solvers=("scip", "tabu", "sa", "ls"))
    res = an.analyse_arm(asm, "MV", "ff49", _acfg())
    assert "scip" not in [s["name"] for s in res["h3c"]["solvers"]]


# --------------------------------------------------------------------------------------------
# Holm over the secondary arms, the whole analysis
# --------------------------------------------------------------------------------------------


def test_holm_adjusts_the_two_secondary_arms_and_not_the_primary():
    ff = synthetic_assembly(n_folds=60, solver_miss=0.0, seed=1)
    us = synthetic_assembly(n_folds=60, arms=("MV",), n_ranks=1, solvers=(), market="us", seed=2)
    out = an.analyse_markets({"ff49": ff, "us": us}, _acfg())
    keys = set(out["arms"])
    assert keys == {"ff49-MV", "ff49-GMV", "us-MV"}
    holm = out["h3a_holm"]
    assert set(holm) == {"ff49-GMV", "us-MV"}
    direct = holm_over_arms({k: rank_equivalence_of(out["arms"][k]) for k in holm})
    for k in holm:
        assert holm[k]["pvalue_holm"] == pytest.approx(direct[k].pvalue_holm)
        assert holm[k]["pvalue_tost_holm"] == pytest.approx(direct[k].pvalue_tost_holm)
        assert holm[k]["outcome"] == direct[k].outcome
        assert holm[k]["pvalue_holm"] >= out["arms"][k]["h3a"]["pvalue"] - 1e-12
    assert out["primary"] == "ff49-MV"


def rank_equivalence_of(arm_result: dict):
    """A ``RankEquivalence``-like view of a stored H3a result (what ``holm_over_arms`` reads)."""
    from types import SimpleNamespace

    h = arm_result["h3a"]
    return SimpleNamespace(pvalue=h["pvalue"], pvalue_tost=h["pvalue_tost"])


def test_the_whole_analysis_runs_without_us_when_only_ff49_is_given():
    out = an.analyse_markets({"ff49": synthetic_assembly(n_folds=60, solver_miss=0.0)}, _acfg())
    assert set(out["arms"]) == {"ff49-MV", "ff49-GMV"}
    assert out["h3a_holm"] is None and "holm" in out["notes"][0]  # one secondary arm: no family


# --------------------------------------------------------------------------------------------
# Sensitivities
# --------------------------------------------------------------------------------------------


def test_fixed_block_and_margin_sensitivities_change_only_what_they_name(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    sens = res["sensitivity"]
    blocks = {b["block"]: b for b in sens["fixed_blocks"]}
    assert set(blocks) == {5, 21}
    for b, entry in blocks.items():
        assert entry["h3a"]["block"] == b and entry["h3b"]["log_dose"]["block"] == b
        assert entry["h3a"]["estimate"] == pytest.approx(res["h3a"]["estimate"])  # the data are the same
    margins = {m["delta_sr"]: m for m in sens["delta_sr"]}
    assert set(margins) == {0.2, 0.5}
    for delta, entry in margins.items():
        assert entry["h3a"]["margin"] == [-delta, delta]
        assert entry["h3a"]["estimate"] == pytest.approx(res["h3a"]["estimate"])
        assert entry["h3a"]["se"] == pytest.approx(res["h3a"]["se"])
    # a wider margin cannot lose an equivalence a narrower one has
    assert margins[0.5]["h3a"]["pvalue_tost"] <= margins[0.2]["h3a"]["pvalue_tost"] + 1e-12


def test_the_gmv_arm_has_no_sharpe_margin_sensitivity(asm):
    sens = an.analyse_arm(asm, "GMV", "ff49", ACFG)["sensitivity"]
    assert "delta_sr" not in sens and {b["block"] for b in sens["fixed_blocks"]} == {5, 21}


def test_passes_such_as_the_execution_lag_are_rerun_on_their_own_paths():
    asm = synthetic_assembly(n_folds=60, solver_miss=0.0)
    asm.paths["gross_lag1"] = asm.paths["gross"].shift(1).fillna(0.0)
    cfg = _acfg(sensitivity_passes={"lag1": "gross_lag1"})
    res = an.analyse_arm(asm, "MV", "ff49", cfg)
    lag = res["sensitivity"]["passes"]["lag1"]
    plain = res["h3a"]
    assert lag["pass"] == "gross_lag1" and lag["h3a"]["estimate"] != pytest.approx(plain["estimate"])
    series = a.fk_series(asm, "gross_lag1", "MV", "sharpe", ranks=[2])
    direct = rank_equivalence(series.opt, series.ranks[:, 0], 0.3, functional="sharpe",
                              block=res["block"]["block"], n_boot=299, alpha=0.05,
                              seed=an.seed_for(ACFG["stats"]["seed"], "ff49", "MV", "h3a", "pass:lag1"))
    assert lag["h3a"]["estimate"] == pytest.approx(direct.estimate)


def test_the_coverage_sensitivity_drops_the_days_of_poorly_covered_folds_from_every_path():
    us = synthetic_assembly(n_folds=60, arms=("MV",), n_ranks=1, solvers=(), market="us")
    res = an.analyse_arm(us, "MV", "us", _acfg())
    cov = res["sensitivity"]["coverage"]
    keep = us.folds["x_coverage"].to_numpy() >= 0.9
    assert cov["n_folds"] == int((keep & us.masks["common"]).sum()) < int(us.masks["common"].sum())
    assert cov["h3a"]["n_obs"] == cov["n_folds"] * 21
    series = a.fk_series(us, "gross", "MV", "sharpe", ranks=[2], fold_mask=keep)
    assert len(series.opt) == cov["h3a"]["n_obs"]


def test_the_hac_check_is_a_robustness_test_on_percent_returns(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    hac = res["hac"]
    series = a.fk_series(asm, "gross", "MV", "sharpe", ranks=[2])
    from cif.stats import ledoit_wolf_sharpe_test

    direct = ledoit_wolf_sharpe_test(100 * series.ranks[:, 0], 100 * series.opt, method="hac",
                                     periods_per_year=252)
    assert hac["difference"] == pytest.approx(direct.difference)
    assert hac["pvalue"] == pytest.approx(direct.pvalue) and hac["method"] == "lw_hac"
    assert hac["difference"] == pytest.approx(res["h3a"]["estimate"], rel=1e-9)


# --------------------------------------------------------------------------------------------
# Net-of-cost metrics
# --------------------------------------------------------------------------------------------


def test_net_metrics_are_computed_on_the_days_of_f_k_by_cost_level(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    table = pd.DataFrame(res["net"]["table"])
    assert set(table["cost"]) == {"cost_0", "cost_10"} and "opt" in set(table["path"])
    mask = asm.days["in_Fk"].to_numpy()
    rf = asm.days["rf"].to_numpy()[mask]
    for cost, name in (("cost_0", "gross"), ("cost_10", "net")):
        r = asm.paths[name]["MV:opt"].to_numpy()[mask] - rf
        row = table[(table["cost"] == cost) & (table["path"] == "opt")].iloc[0]
        assert row["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
        gamma = asm.meta["lam"]
        assert row["ce"] == pytest.approx((r.mean() - 0.5 * gamma * r.var(ddof=1)) * 252)
    stats = asm.fold_stats["net"]
    stats = stats[(stats["arm"] == "MV") & (stats["path"] == "opt")
                  & stats["fold"].map(lambda f: bool(asm.masks["common"][f]))]
    years = mask.sum() / 252
    row = table[(table["cost"] == "cost_10") & (table["path"] == "opt")].iloc[0]
    assert row["annual_turnover"] == pytest.approx(stats["turnover"].sum() / years)
    assert row["annual_cost_bps"] == pytest.approx(stats["cost"].sum() / years * 1e4)
    gross_row = table[(table["cost"] == "cost_0") & (table["path"] == "opt")].iloc[0]
    assert gross_row["annual_cost_bps"] == 0.0


def test_net_differences_come_with_stationary_bootstrap_intervals(asm):
    res = an.analyse_arm(asm, "MV", "ff49", ACFG)
    diffs = pd.DataFrame(res["net"]["differences"])
    assert {"rank_2", "solver_tabu"} <= set(diffs["path"])
    row = diffs[(diffs["path"] == "rank_2") & (diffs["cost"] == "cost_10")].iloc[0]
    table = pd.DataFrame(res["net"]["table"])
    pick = lambda p: table[(table["cost"] == "cost_10") & (table["path"] == p)].iloc[0]  # noqa: E731
    assert row["sharpe_diff"] == pytest.approx(pick("rank_2")["sharpe"] - pick("opt")["sharpe"])
    assert row["sharpe_diff_lo"] < row["sharpe_diff_hi"]
    assert abs(row["sharpe_diff"] - 0.5 * (row["sharpe_diff_lo"] + row["sharpe_diff_hi"])) < \
        row["sharpe_diff_hi"] - row["sharpe_diff_lo"]  # the interval is around the estimate
    assert row["turnover_diff"] == pytest.approx(pick("rank_2")["annual_turnover"] - pick("opt")["annual_turnover"])
    assert np.isfinite(row[["ce_diff_lo", "ce_diff_hi", "turnover_diff_lo", "turnover_diff_hi"]].astype(float)).all()
    assert res["net"]["bootstrap"]["method"] == "stationary"


# --------------------------------------------------------------------------------------------
# Descriptive tables
# --------------------------------------------------------------------------------------------


def test_dose_tables_report_the_distribution_of_rank_two_and_by_rank(asm):
    d = an.describe(asm, ACFG)
    mask = asm.masks["common"]
    for arm in ("MV", "GMV"):
        g2 = asm.doses[arm][2].to_numpy()[mask]
        row = d["rank2_doses"][(d["rank2_doses"]["arm"] == arm) & (d["rank2_doses"]["stratum"] == "all")].iloc[0]
        assert row["n_folds"] == mask.sum()
        assert row["median"] == pytest.approx(np.median(g2))
        assert (row["q1"], row["q3"]) == pytest.approx(tuple(np.quantile(g2, [0.25, 0.75])))
        assert row["share_le_gstar"] == pytest.approx(np.mean(g2 <= 10.0))
        binding = asm.folds["binding"].to_numpy()[mask]
        for stratum, flag in (("binding", True), ("non_binding", False)):
            sub = d["rank2_doses"][(d["rank2_doses"]["arm"] == arm) & (d["rank2_doses"]["stratum"] == stratum)].iloc[0]
            assert sub["n_folds"] == int((binding == flag).sum())
            assert sub["median"] == pytest.approx(np.median(g2[binding == flag]))
    by_rank = d["doses_by_rank"]
    r17 = by_rank[(by_rank["arm"] == "MV") & (by_rank["rank"] == 17)].iloc[0]
    g = asm.doses["MV"][17].to_numpy()[mask]
    assert r17["mean"] == pytest.approx(g.mean()) and r17["median"] == pytest.approx(np.median(g))


def test_exclusions_regime_and_ties_are_reported():
    asm = synthetic_assembly(n_folds=60, excluded=(3, 4, 5), solver_miss=0.0)
    d = an.describe(asm, ACFG)
    ex = d["exclusions"].set_index("set")
    assert ex.loc["common", "n_folds"] == 60 and ex.loc["common", "n_included"] == 57
    assert ex.loc["common", "share_excluded"] == pytest.approx(3 / 60)
    reg = d["regime"].iloc[0]
    assert reg["n_folds_Fk"] == 57
    assert reg["share_binding"] == pytest.approx(asm.folds["binding"].to_numpy()[asm.masks["common"]].mean())
    assert reg["share_lower_bound"] == pytest.approx(
        asm.folds["lower_bound_active"].to_numpy()[asm.masks["common"]].mean())
    ties = d["ties"].set_index("arm")
    assert ties.loc["MV", "n_pairs"] == 57 * 32 and ties.loc["MV", "share_tied_pairs"] == 0.0
    assert ties.loc["MV", "share_boundary_tie"] == 0.0


def test_tied_adjacent_ranks_are_counted_with_the_tolerance_of_the_fold():
    asm = synthetic_assembly(n_folds=40, solver_miss=0.0)
    asm.doses["MV"].loc[:, 5] = asm.doses["MV"].loc[:, 4] + 5e-4  # rank 5 ties with rank 4 (tol 1e-3 bp)
    d = an.describe(asm, ACFG)
    ties = d["ties"].set_index("arm")
    assert ties.loc["MV", "share_tied_pairs"] == pytest.approx(1 / 32)  # one of the 32 pairs 1-2 … 32-33


# --------------------------------------------------------------------------------------------
# Tables and the registered stage
# --------------------------------------------------------------------------------------------


def test_tables_are_written_and_the_summary_is_json(tmp_path, asm):
    out = an.analyse_markets({"ff49": asm}, _acfg())
    files = an.write_tables(out, tmp_path / "tables")
    names = {f.name for f in files}
    assert {"analysis.json", "h3a.csv", "h3b.csv", "h3b_per_rank.csv", "h3c.csv", "rank2_doses.csv",
            "doses_by_rank.csv", "exclusions.csv", "regime.csv", "ties.csv", "net_metrics.csv",
            "net_differences.csv", "sensitivity.csv", "arm_contrast.csv"} <= names
    h3a = pd.read_csv(tmp_path / "tables" / "h3a.csv")
    assert {"key", "variant", "estimate", "se", "ci_tost_low", "ci_tost_high", "pvalue", "pvalue_tost",
            "equivalent", "significant", "outcome", "margin_low", "margin_high", "block"} <= set(h3a.columns)
    assert {"ff49-MV", "ff49-GMV"} == set(h3a["key"]) and "main" in set(h3a["variant"])
    json.loads((tmp_path / "tables" / "analysis.json").read_text())


def test_the_analysis_stage_registers_a_run_with_the_assemblies_as_inputs(tmp_path):
    ff = synthetic_assembly(n_folds=60, solver_miss=0.0)
    a.save_assembly(ff, tmp_path / "asm_ff49")
    cfg = _acfg()
    cfg["assemblies"] = {"ff49": str(tmp_path / "asm_ff49")}
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    assert an.main(["--analysis-config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (out,) = [p for p in runs.iterdir() if p.is_dir()]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["stream"] == "E3"
    assert (out / "tables" / "h3a.csv").exists() and (out / "tables" / "analysis.json").exists()
    config = json.loads((out / "config.json").read_text())
    assert config["assemblies"]["ff49"].endswith("asm_ff49") and len(config["assembly_meta_sha256"]["ff49"]) == 64
    assert manifest["inputs"]["ff49"]["sha256"]
