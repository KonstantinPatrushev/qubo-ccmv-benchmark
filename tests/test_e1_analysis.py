"""E1 analysis stage (ANALYSIS_PLAN §2 RQ1, §7.1 H1a, §7.2, §7.3) on synthetic rows (``tests/e1_rows.py``).

The rows are made up with a known structure; nothing here is a result. The tables are checked
against independent computations on the same rows (pandas / scipy), the tests against the
properties of the generator.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from cif.experiments import e1_analysis as ana
from cif.experiments import e1_attribution as e1
from cif.experiments import run as runner
from cif.stats.rq1 import wilson_interval
from tests.e1_rows import MS, TAU, synthetic_rows

SWAP = "swap_sa/c0.01"  # the config_id of swap-SA at the reference cold end of its schedule (§12 B3)
REFERENCE = {"sa": {"penalty": 100.0, "beta": "default"}, "tabu": {"penalty": 100.0},
             "swap_sa": {"cold_ratio": 0.01}}
SETTINGS = {
    "tau": TAU, "zero_tol": 1e-8, "alpha": 0.05, "share_ms": [1, 50], "level": 0.95,
    "bootstrap": {"n_boot": 199, "seed": 50000}, "h1c": {"p_value": None},
    "models": {"units": ["instance_median", "seed"], "responses": ["E_samp", "E_prec"],
               "reference": REFERENCE, "chosen_after_grid": []},
}


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return synthetic_rows(seed=1)


@pytest.fixture(scope="module")
def frame(rows) -> pd.DataFrame:
    return ana.frame_from_rows(rows, MS)


@pytest.fixture(scope="module")
def inst(frame) -> pd.DataFrame:
    return ana.instance_frame(frame, MS)


@pytest.fixture(scope="module")
def pairs(inst) -> pd.DataFrame:
    return ana.pair_frame(inst, MS, h_lo=0.15, h_hi=0.45)


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------


def test_frame_has_instance_and_pair_ids_and_numeric_nan_for_missing(rows):
    rows = [dict(r) for r in rows[:6]]
    rows[0]["E_samp_1"] = None
    df = ana.frame_from_rows(rows, MS)
    assert df["instance"].iloc[0] == f"{rows[0]['n']}/{rows[0]['pair_seed']}/{rows[0]['h']:g}"
    assert df["pair"].iloc[0] == f"{rows[0]['n']}/{rows[0]['pair_seed']}"
    assert math.isnan(df["E_samp_1"].iloc[0]) and df["E_samp_1"].dtype == float


def test_the_rows_carry_the_split_of_the_residual_and_the_frame_requires_it(rows):
    """§12 B2: E_samp = E_prec + E_short in every cell; the analysis needs the columns, it cannot rebuild
    ``E_form(n_pipe(m))`` without the oracle."""
    for r in rows[:50]:
        for m in MS:
            assert r[f"E_samp_{m}"] == pytest.approx(r[f"E_prec_{m}"] + r[f"E_short_{m}"], abs=1e-15)
            assert r[f"E_short_{m}"] >= 0.0
    bad = [{k: v for k, v in r.items() if k != "E_prec_20"} for r in rows[:6]]
    with pytest.raises(ValueError, match="E_prec_20"):
        ana.frame_from_rows(bad, MS)
    bad = [{k: v for k, v in r.items() if k != "E_form_at_npipe_5"} for r in rows[:6]]
    with pytest.raises(ValueError, match="E_form_at_npipe_5"):
        ana.frame_from_rows(bad, MS)


def test_one_row_per_instance_with_its_ceiling(inst, rows):
    assert len(inst) == 20 * 2
    first = {(r["n"], r["pair_seed"], r["h"]): r for r in rows}
    for r in inst.itertuples():
        src = first[(r.n, r.pair_seed, r.h)]
        for m in MS:
            assert getattr(r, f"E_form_{m}") == src[f"E_form_{m}"]
        assert r.K == 4 and r.pair == f"{r.n}/{r.pair_seed}"


def test_the_oracle_must_be_the_same_in_all_cells_of_an_instance(rows):
    bad = [dict(r) for r in rows]
    bad[7]["E_form_50"] = bad[7]["E_form_50"] + 1e-3  # a shard that computed another oracle
    with pytest.raises(ValueError, match="oracle"):
        ana.instance_frame(ana.frame_from_rows(bad, MS), MS)


def test_a_missing_ceiling_is_refused_not_counted_as_below_tau(rows):
    """A NaN in E_form would count as 'not above tau' in a share and stop the paired test: refuse it."""
    bad = [dict(r) for r in rows]
    for r in bad:
        if (r["n"], r["pair_seed"], r["h"]) == (16, 2000, 0.45):
            r["E_form_50"] = None
    with pytest.raises(ValueError, match=r"E_form_50.*16/2000/0.45"):
        ana.instance_frame(ana.frame_from_rows(bad, MS), MS)


def test_pairs_hold_both_levels_and_their_difference(pairs, inst):
    assert len(pairs) == 20 and set(pairs["n"]) == {16, 24}
    for m in MS:
        d = pairs[f"E_form_{m}_hi"] - pairs[f"E_form_{m}_lo"]
        assert pairs[f"d_{m}"].to_numpy() == pytest.approx(d.to_numpy(), abs=0, rel=0)
    one = pairs.iloc[0]
    lo = inst[(inst["pair_seed"] == one.pair_seed) & (inst["h"] == 0.15)].iloc[0]
    assert one["E_form_50_lo"] == lo["E_form_50"]


def test_a_pair_with_a_missing_level_is_refused(inst):
    with pytest.raises(ValueError, match="level"):
        ana.pair_frame(inst[~((inst["pair_seed"] == 2000) & (inst["h"] == 0.45))], MS, 0.15, 0.45)


def test_missing_ms_columns_are_refused(rows):
    with pytest.raises(ValueError, match="E_pipe_7"):
        ana.frame_from_rows(rows, (*MS, 7))


# --------------------------------------------------------------------------------------
# Integrity checks
# --------------------------------------------------------------------------------------


def test_check_frame_reports_counts(frame):
    diag = ana.check_frame(frame, MS)
    assert diag["n_rows"] == len(frame) and diag["status_counts"] == {"ok": len(frame)}
    assert diag["n_inconsistent"] == 0 and diag["n_not_certified"] == 0 and diag["n_no_valid"] == 0
    assert diag["n_instances"] == 40 and diag["oracle_spread_max"] == 0.0


def test_check_frame_counts_cells_without_a_valid_sample():
    df = ana.frame_from_rows(synthetic_rows(seed=2, no_valid_share=0.1), MS)
    diag = ana.check_frame(df, MS)
    assert diag["n_no_valid"] > 0 and diag["status_counts"]["no_valid_samples"] == diag["n_no_valid"]


def test_check_frame_refuses_an_unknown_status_or_an_error_row(rows):
    bad = [dict(r) for r in rows]
    bad[3]["status"] = "weird"
    with pytest.raises(ValueError, match="status"):
        ana.check_frame(ana.frame_from_rows(bad, MS), MS)
    bad = [dict(r) for r in rows]
    bad[3]["error"] = "RuntimeError: boom"
    with pytest.raises(ValueError, match="error"):
        ana.check_frame(ana.frame_from_rows(bad, MS), MS)


def test_check_frame_flags_inconsistent_and_uncertified_rows(rows):
    bad = [dict(r) for r in rows]
    bad[0]["consistent"] = False
    bad[1]["ref_type"] = "best_known"
    diag = ana.check_frame(ana.frame_from_rows(bad, MS), MS)
    assert diag["n_inconsistent"] == 1 and diag["n_not_certified"] == 1


def test_check_frame_refuses_a_cell_that_occurs_twice(rows):
    """A cell counted twice would weigh double in every median and mean."""
    doubled = [*rows, dict(rows[5])]
    with pytest.raises(ValueError, match=r"twice|duplicate"):
        ana.check_frame(ana.frame_from_rows(doubled, MS), MS)


def test_check_frame_compares_with_the_expected_cells(rows, frame):
    keys = {r["cell_key"] for r in rows}
    assert ana.check_frame(frame, MS, expected_keys=keys)["n_missing"] == 0
    with pytest.raises(ValueError, match="missing"):
        ana.check_frame(frame.iloc[:-5], MS, expected_keys=keys)
    with pytest.raises(ValueError, match="unexpected"):
        ana.check_frame(frame, MS, expected_keys=set(list(keys)[:-3]))


# --------------------------------------------------------------------------------------
# H1a
# --------------------------------------------------------------------------------------


def test_h1a_is_the_stratified_one_sided_signed_rank_test_of_the_pair_differences(pairs):
    res = ana.h1a_result(pairs, 50, SETTINGS)
    d = pairs["d_50"].to_numpy()
    strata = pairs["n"].to_numpy()
    from cif.stats.rq1 import stratified_signed_rank

    ref = stratified_signed_rank(d, strata, alternative="greater", method="exact", zero_tol=1e-8)
    assert res["p_value"] == ref.p_value and res["statistic"] == ref.statistic
    assert res["m0"] == 50 and res["n_pairs"] == 20 and res["alternative"] == "greater"
    assert res["method"] == "exact" and res["n_zero"] == ref.n_zero
    assert set(res["strata"]) == {"16", "24"}
    assert res["share_positive"] == pytest.approx((d > 1e-8).mean())


def test_h1a_detects_the_generated_shift_and_not_its_absence():
    shifted = ana.pair_frame(ana.instance_frame(ana.frame_from_rows(
        synthetic_rows(seed=3, n_pairs={16: 20, 24: 20}, shift_hi=3.0, zero_share=0.0), MS), MS), MS, 0.15, 0.45)
    assert ana.h1a_result(shifted, 1, SETTINGS)["p_value"] < 1e-6
    null = ana.pair_frame(ana.instance_frame(ana.frame_from_rows(
        synthetic_rows(seed=4, n_pairs={16: 20, 24: 20}, shift_hi=1.0, zero_share=0.0), MS), MS), MS, 0.15, 0.45)
    assert ana.h1a_result(null, 1, SETTINGS)["p_value"] > 0.05


def test_h1a_reports_the_size_of_the_difference_with_a_bootstrap_interval(pairs):
    res = ana.h1a_result(pairs, 50, SETTINGS)
    d = pairs["d_50"].to_numpy()
    assert res["difference"]["mean"]["estimate"] == pytest.approx(d.mean())
    assert res["difference"]["median"]["estimate"] == pytest.approx(np.median(d))
    lo, hi = res["difference"]["mean"]["ci"]
    assert lo < d.mean() < hi
    assert res["difference"]["level"] == 0.95 and res["difference"]["n_boot"] == 199


def test_h1a_outcome_follows_the_holm_family(pairs):
    pending = ana.h1a_result(pairs, 50, SETTINGS)
    assert pending["holm"]["complete"] is False
    sure = dict(SETTINGS, h1c={"p_value": 0.9})
    res = ana.h1a_result(pairs, 50, sure)
    assert res["holm"]["complete"] is True and res["holm"]["adjusted"]["H1a"] >= res["p_value"]
    assert res["outcome"] in {"confirmed", "not_confirmed"}
    assert (res["outcome"] == "confirmed") == (res["holm"]["adjusted"]["H1a"] <= 0.05)


@pytest.mark.parametrize(
    ("p", "h1c", "outcome"),
    [
        (0.010, None, "confirmed"),  # 2p = 0.02 <= 0.05: whatever H1c is
        (0.025, None, "confirmed"),  # 2p = 0.05 <= 0.05
        (0.030, None, "pending_h1c"),  # p <= 0.05 < 2p: depends on H1c
        (0.200, None, "not_confirmed"),  # p > 0.05 already: Holm cannot lower it
        (0.030, 0.90, "not_confirmed"),  # complete family: adjusted 0.06
        (0.030, 0.01, "confirmed"),  # H1c first (0.02), then H1a at 0.03 <= 0.05
        (0.010, 0.90, "confirmed"),
    ],
)
def test_the_outcome_of_h1a_from_its_p_value_and_the_family(p, h1c, outcome):
    from cif.stats.rq1 import holm_family

    holm = holm_family({"H1a": p, "H1c": h1c})
    assert ana.h1a_outcome(p, holm, 0.05) == outcome


def test_h1a_needs_the_chosen_m0_column(pairs):
    with pytest.raises(ValueError, match="m0"):
        ana.h1a_result(pairs, 7, SETTINGS)


# --------------------------------------------------------------------------------------
# Shares with Wilson intervals, magnitudes with bootstrap intervals
# --------------------------------------------------------------------------------------


def test_shares_are_counts_over_tau_with_wilson_intervals(inst):
    table = ana.shares_table(inst, MS, SETTINGS)
    assert set(table["m"]) == {1, 50} and set(table["group"]) == {"all", "n", "h", "n_h"}
    for m in (1, 50):
        row = table[(table["m"] == m) & (table["group"] == "all")].iloc[0]
        k = int((inst[f"E_form_{m}"] > TAU).sum())
        assert row["n_instances"] == 40 and row["n_exceed"] == k and row["share"] == pytest.approx(k / 40)
        assert (row["ci_low"], row["ci_high"]) == pytest.approx(wilson_interval(k, 40))
        assert not row["independent"]  # both levels of a pair are in the group: not independent
    h_rows = table[(table["group"] == "h")]
    assert len(h_rows) == 4 and h_rows["independent"].all()
    cell = table[(table["group"] == "n_h") & (table["m"] == 50) & (table["n"] == 16) & (table["h"] == 0.45)].iloc[0]
    sub = inst[(inst["n"] == 16) & (inst["h"] == 0.45)]
    assert cell["n_exceed"] == int((sub["E_form_50"] > TAU).sum()) and cell["n_instances"] == 10


def test_e_form_summaries_have_bootstrap_intervals(inst):
    table = ana.e_form_table(inst, MS, SETTINGS)
    row = table[(table["m"] == 20) & (table["group"] == "h") & (table["h"] == 0.45)].iloc[0]
    sub = inst[inst["h"] == 0.45]["E_form_20"]
    assert row["mean"] == pytest.approx(sub.mean()) and row["median"] == pytest.approx(sub.median())
    assert row["mean_ci_low"] < row["mean"] < row["mean_ci_high"]
    assert row["n_instances"] == 20 and row["n_clusters"] == 20
    both = table[(table["m"] == 20) & (table["group"] == "all")].iloc[0]
    assert both["n_instances"] == 40 and both["n_clusters"] == 20  # clusters are the pairs
    assert set(table["m"]) == set(MS)


def test_the_level_difference_table_is_by_pair(pairs):
    table = ana.difference_table(pairs, MS, SETTINGS)
    row = table[(table["m"] == 50) & (table["group"] == "all")].iloc[0]
    d = pairs["d_50"]
    assert row["n_pairs"] == 20 and row["mean"] == pytest.approx(d.mean())
    assert row["share_positive"] == pytest.approx((d > 1e-8).mean())
    assert row["share_zero"] == pytest.approx((d.abs() <= 1e-8).mean())
    assert row["share_positive"] + row["share_zero"] + row["share_negative"] == pytest.approx(1.0)
    assert set(table["group"]) == {"all", "n"}


def test_the_design_table_reports_binding_and_dispersion_by_level(inst):
    table = ana.design_table(inst)
    row = table[(table["group"] == "h") & (table["h"] == 0.45)].iloc[0]
    sub = inst[inst["h"] == 0.45]
    assert row["n_instances"] == 20 and row["share_binding"] == sub["binding"].mean()
    assert row["depth_median"] == sub["depth"].median() and row["cv_vol_mean"] == pytest.approx(sub["cv_vol"].mean())
    hi, lo = (table[(table["group"] == "h") & (table["h"] == h)].iloc[0] for h in (0.45, 0.15))
    assert hi["cv_vol_mean"] > lo["cv_vol_mean"]
    assert set(table["group"]) == {"all", "h", "n_h"}


def test_the_group_column_n_is_an_integer_column_with_missing_values(inst, frame):
    """``n`` is empty for the groups that do not fix N; it must stay an integer in the CSV (16, not 16.0)."""
    for table in (ana.shares_table(inst, MS, SETTINGS), ana.e_form_table(inst, MS, SETTINGS),
                  ana.design_table(inst), ana.samp_table(frame, MS, SETTINGS),
                  ana.sampler_diagnostics_table(frame), ana.pool_curve_table(frame, MS, SETTINGS)):
        assert str(table["n"].dtype) == "Int64" and table["n"].isna().any()
        assert set(table["n"].dropna()) == {16, 24}
        assert "16.0" not in table.to_csv(index=False)


def test_tables_are_deterministic(inst):
    a = ana.e_form_table(inst, MS, SETTINGS)
    b = ana.e_form_table(inst, MS, SETTINGS)
    pd.testing.assert_frame_equal(a, b)
    other = ana.e_form_table(inst, MS, dict(SETTINGS, bootstrap={"n_boot": 199, "seed": 50001}))
    assert not a["mean_ci_low"].equals(other["mean_ci_low"]) and a["mean"].equals(other["mean"])


def test_the_share_table_counts_agree_with_an_independent_count_by_group(inst):
    table = ana.shares_table(inst, MS, SETTINGS)
    for (n, h), sub in inst.groupby(["n", "h"]):
        row = table[(table["group"] == "n_h") & (table["m"] == 1) & (table["n"] == n) & (table["h"] == h)].iloc[0]
        assert row["n_exceed"] == int((sub["E_form_1"] > TAU).sum())


# --------------------------------------------------------------------------------------
# The sampler residual and the pipeline
# --------------------------------------------------------------------------------------


def _instance_medians(frame: pd.DataFrame, config_id: str, column: str) -> pd.Series:
    return frame[frame["config_id"] == config_id].groupby("instance")[column].median().dropna()


def test_the_sampler_table_summarises_instances_by_the_median_over_seeds(frame):
    table = ana.samp_table(frame, MS, SETTINGS)
    row = table[(table["config_id"] == SWAP) & (table["m"] == 20) & (table["group"] == "all")].iloc[0]
    med = _instance_medians(frame, SWAP, "E_samp_20")
    assert row["n_instances"] == 40 and row["sampler"] == "swap_sa"
    assert row["e_samp_mean"] == pytest.approx(med.mean()) and row["e_samp_median"] == pytest.approx(med.median())
    assert row["e_samp_mean_ci_low"] < row["e_samp_mean"] < row["e_samp_mean_ci_high"]
    pipe = _instance_medians(frame, SWAP, "E_pipe_20")
    assert row["e_pipe_mean"] == pytest.approx(pipe.mean())
    assert row["share_pipe_above_tau"] == pytest.approx((pipe > TAU).mean())
    assert row["share_samp_below_minus_tau"] == pytest.approx((med < -TAU).mean())
    assert row["share_samp_within_tau"] == pytest.approx((med.abs() <= TAU).mean())
    lo, hi = wilson_interval(int((pipe > TAU).sum()), 40)
    assert (row["share_pipe_above_tau_ci_low"], row["share_pipe_above_tau_ci_high"]) == pytest.approx((lo, hi))
    assert row["n_pipe_median"] == pytest.approx(min(20, 120))


def test_the_sampler_table_covers_every_configuration_pool_size_and_group(frame):
    table = ana.samp_table(frame, MS, SETTINGS)
    configs = {"sa/p10/default", "sa/p10/objective", "sa/p100/default", "sa/p100/objective",
               "tabu/p10", "tabu/p100", SWAP}
    assert set(table["config_id"]) == configs and set(table["m"]) == set(MS)
    assert set(table["group"]) == {"all", "n", "h", "n_h"}
    assert len(table) == len(configs) * len(MS) * (1 + 2 + 2 + 4)
    sa = table[table["config_id"] == "sa/p100/objective"].iloc[0]
    assert sa["penalty"] == 100.0 and sa["beta"] == "objective"
    swap = table[table["config_id"] == SWAP].iloc[0]
    assert pd.isna(swap["penalty"]) and pd.isna(swap["beta"])


def test_the_identity_e_pipe_is_e_form_plus_e_samp_holds_for_the_means(frame, inst):
    table = ana.samp_table(frame, MS, SETTINGS)
    row = table[(table["config_id"] == "tabu/p100") & (table["m"] == 5) & (table["group"] == "all")].iloc[0]
    # the median over seeds of a sum is not the sum of medians, but the E_form part is constant over seeds
    form = inst["E_form_5"].mean()
    assert row["e_pipe_mean"] == pytest.approx(form + row["e_samp_mean"], abs=1e-12)


def test_cells_without_a_valid_sample_are_left_out_of_the_instance_medians():
    rows = synthetic_rows(seed=6, no_valid_share=0.3)
    df = ana.frame_from_rows(rows, MS)
    table = ana.samp_table(df, MS, SETTINGS)
    row = table[(table["config_id"] == SWAP) & (table["m"] == 1) & (table["group"] == "all")].iloc[0]
    med = _instance_medians(df, SWAP, "E_samp_1")
    assert row["n_instances"] == len(med) and row["e_samp_mean"] == pytest.approx(med.mean())


def test_the_sampler_diagnostics_report_the_call_time_and_the_valid_share(frame):
    table = ana.sampler_diagnostics_table(frame)
    row = table[(table["config_id"] == "sa/p10/default") & (table["group"] == "all")].iloc[0]
    sub = frame[frame["config_id"] == "sa/p10/default"]
    assert row["n_cells"] == len(sub) and row["solve_s_median"] == pytest.approx(sub["solve_s"].median())
    assert row["solve_s_q25"] <= row["solve_s_median"] <= row["solve_s_q75"]
    assert row["frac_valid_mean"] == pytest.approx(sub["frac_valid"].mean())
    assert row["pool_n_unique_median"] == 120 and row["hit_rate_mean"] == 1.0
    assert row["tau_pool_median"] == pytest.approx(sub["tau_pool"].median())


def test_diagnostics_of_a_column_that_is_missing_everywhere_are_nan_and_silent(rows):
    """Kendall's τ of a pool is undefined when the pool is tiny: the column may be all NaN."""
    import warnings

    df = ana.frame_from_rows([{**r, "tau_pool": None} for r in rows], MS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = ana.sampler_diagnostics_table(df)
    assert table["tau_pool_median"].isna().all() and table["solve_s_median"].notna().all()


def test_the_pool_curve_pairs_the_quality_of_a_pool_with_the_time_of_the_whole_pipeline(frame):
    table = ana.pool_curve_table(frame, MS, SETTINGS)
    row = table[(table["config_id"] == SWAP) & (table["m"] == 50) & (table["group"] == "all")].iloc[0]
    sub = frame[frame["config_id"] == SWAP]
    # time of the pipeline with a pool of m: sampler call + building + m refinements at the cost per
    # refinement that the oracle measured on this instance (oracle_refine_s for n_form_200 of them)
    per_call = sub["oracle_refine_s"] / sub["n_form_200"]
    time = sub["build_s"] + sub["solve_s"] + sub["n_pipe_50"] * per_call
    expect = time.groupby(sub["instance"]).median().median()
    assert row["time_s_median"] == pytest.approx(expect)
    pipe = sub.groupby("instance")["E_pipe_50"].median()
    assert row["e_pipe_mean"] == pytest.approx(pipe.mean()) and row["n_instances"] == 40
    assert row["e_pipe_mean_ci_low"] < row["e_pipe_mean"] < row["e_pipe_mean_ci_high"]
    curve = table[(table["config_id"] == SWAP) & (table["group"] == "all")].sort_values("m")
    assert curve["time_s_median"].is_monotonic_increasing  # a bigger pool costs more refinements
    assert set(table["group"]) == {"all", "n"}


# --------------------------------------------------------------------------------------
# B2: E_samp = E_prec + E_short and the share of pools smaller than m
# --------------------------------------------------------------------------------------

POOLS = {"sa": (120, 120), "tabu": (3, 12), "swap_sa": (6, 30)}  # distinct subsets per call, by sampler


@pytest.fixture(scope="module")
def short_frame() -> pd.DataFrame:
    return ana.frame_from_rows(synthetic_rows(seed=12, pool_unique=POOLS), MS)


@pytest.fixture(scope="module")
def big_frame_short() -> pd.DataFrame:
    return ana.frame_from_rows(synthetic_rows(seed=14, n_pairs={16: 12, 24: 12}, pool_unique=POOLS), MS)


def test_the_decomposition_table_adds_up_in_every_row(short_frame):
    table = ana.decomposition_table(short_frame, MS, SETTINGS)
    assert set(table["group"]) == {"all", "n", "h", "n_h"} and set(table["m"]) == set(MS)
    assert len(table) == 7 * len(MS) * (1 + 2 + 2 + 4)
    gap = table["e_samp_mean"] - table["e_prec_mean"] - table["e_short_mean"]
    assert gap.abs().max() < 1e-12  # the identity, in the unit in which it is linear
    assert (table["identity_gap"] - gap).abs().max() < 1e-15
    assert (table["e_short_mean"] >= -1e-15).all()
    assert not any(c.startswith("e_") and "median" in c for c in table.columns)  # means: what the identity is about
    row = table[(table["config_id"] == "tabu/p100") & (table["m"] == 20) & (table["group"] == "all")].iloc[0]
    sub = short_frame[short_frame["config_id"] == "tabu/p100"]
    means = sub.groupby("instance")[["E_samp_20", "E_prec_20", "E_short_20"]].mean()
    assert row["e_samp_mean"] == pytest.approx(means["E_samp_20"].mean())
    assert row["e_prec_mean"] == pytest.approx(means["E_prec_20"].mean())
    assert row["e_short_mean"] == pytest.approx(means["E_short_20"].mean())
    assert row["e_prec_mean_ci_low"] < row["e_prec_mean"] < row["e_prec_mean_ci_high"]
    assert row["n_instances"] == 40 and row["n_clusters"] == 20


def test_a_pool_that_has_m_subsets_pays_nothing_for_its_size(short_frame):
    table = ana.decomposition_table(short_frame, MS, SETTINGS)
    sa = table[(table["sampler"] == "sa") & (table["m"] <= 50)]
    assert (sa["e_short_mean"] == 0.0).all() and (sa["share_short"] == 0.0).all()  # 120 subsets >= 50
    tabu = table[(table["config_id"] == "tabu/p100") & (table["m"] == 200) & (table["group"] == "all")].iloc[0]
    assert tabu["e_short_mean"] > 0.0 and tabu["share_short"] == pytest.approx(1.0)


def test_the_share_of_cells_with_a_pool_smaller_than_m_by_sampler_n_h_and_m(short_frame):
    table = ana.decomposition_table(short_frame, MS, SETTINGS)

    def by_instance_mean(short, mask):
        return short[mask].groupby(short_frame.loc[mask, "instance"]).mean().mean()

    for m in MS:
        short = (short_frame[f"n_pipe_{m}"] < short_frame[f"n_form_{m}"]).astype(float)
        for config in ("sa/p10/default", "tabu/p100", SWAP):
            mask = short_frame["config_id"] == config
            cond = (table["config_id"] == config) & (table["m"] == m)
            assert table[cond & (table["group"] == "all")].iloc[0]["share_short"] == pytest.approx(
                by_instance_mean(short, mask))
            for n in (16, 24):
                row = table[cond & (table["group"] == "n") & (table["n"] == n)].iloc[0]
                assert row["share_short"] == pytest.approx(by_instance_mean(short, mask & (short_frame["n"] == n)))
            for h in (0.15, 0.45):
                row = table[cond & (table["group"] == "h") & (table["h"] == h)].iloc[0]
                assert row["share_short"] == pytest.approx(by_instance_mean(short, mask & (short_frame["h"] == h)))
    inner = table[(table["config_id"] == SWAP) & (table["m"] == 20) & (table["group"] == "all")].iloc[0]
    assert 0.0 < inner["share_short"] < 1.0  # pools of 6-30 subsets: sometimes short at m = 20
    assert inner["share_short_ci_low"] <= inner["share_short"] <= inner["share_short_ci_high"]


def test_a_cell_without_a_valid_subset_has_an_empty_pool_and_is_short_for_every_m():
    df = ana.frame_from_rows(synthetic_rows(seed=15, no_valid_share=0.2), MS)
    assert (df["n_pipe_1"] == 0).any()
    table = ana.decomposition_table(df, MS, SETTINGS)
    row = table[(table["config_id"] == "sa/p10/default") & (table["m"] == 1) & (table["group"] == "all")].iloc[0]
    sub = df[df["config_id"] == "sa/p10/default"]
    expect = (sub["n_pipe_1"] == 0).astype(float).groupby(sub["instance"]).mean().mean()
    assert row["share_short"] == pytest.approx(expect) and row["share_short"] > 0.0
    assert row["e_samp_mean"] == pytest.approx(row["e_prec_mean"] + row["e_short_mean"], abs=1e-12)


def test_the_decomposition_table_is_deterministic_and_has_integer_n(short_frame):
    a = ana.decomposition_table(short_frame, MS, SETTINGS)
    b = ana.decomposition_table(short_frame, MS, SETTINGS)
    pd.testing.assert_frame_equal(a, b)
    assert str(a["n"].dtype) == "Int64" and set(a["n"].dropna()) == {16, 24}
    assert ".0" not in a[["n"]].to_csv(index=False)  # 16, not 16.0 (the column is empty where a group spans N)


@pytest.mark.parametrize("unit", ["instance_median", "seed"])
def test_long_frames_for_e_prec_hold_the_precision_response(short_frame, unit):
    longs = ana.long_frames(short_frame, MS, SETTINGS, unit=unit, response="E_prec")
    samp = ana.long_frames(short_frame, MS, SETTINGS, unit=unit, response="E_samp")
    swap = longs["swap_sa"]
    src = short_frame[short_frame["sampler"] == "swap_sa"].iloc[3]
    cells = short_frame[(short_frame["sampler"] == "swap_sa") & (short_frame["instance"] == src["instance"])]
    got = swap[(swap["instance"] == src["instance"]) & (swap["log_m"] == math.log(50))]
    if unit == "instance_median":
        assert got["y"].iloc[0] == pytest.approx(math.asinh(cells["E_prec_50"].median() / TAU))
    else:
        assert sorted(got["y"]) == pytest.approx(sorted(np.arcsinh(cells["E_prec_50"] / TAU)))
    assert set(longs) == set(samp) and len(swap) == len(samp["swap_sa"])
    assert not np.allclose(swap["y"].to_numpy(), samp["swap_sa"]["y"].to_numpy())  # pools are short: they differ


def test_an_unknown_response_is_refused(short_frame):
    with pytest.raises(ValueError, match="response"):
        ana.long_frames(short_frame, MS, SETTINGS, unit="seed", response="E_pipe")


def test_with_no_short_pool_the_precision_models_are_the_models_of_the_residual():
    """E_short = 0 everywhere: E_prec = E_samp, so the two sets of models must coincide."""
    rows = synthetic_rows(seed=13, n_pairs={16: 12, 24: 12},
                          pool_unique={"sa": (200, 200), "tabu": (200, 200), "swap_sa": (200, 200)})
    df = ana.frame_from_rows(rows, MS)
    a, _ = ana.model_tables(df, MS, SETTINGS, "instance_median", response="E_samp")
    b, summary = ana.model_tables(df, MS, SETTINGS, "instance_median", response="E_prec")
    pd.testing.assert_frame_equal(a["fixed_effects"].drop(columns="response"), b["fixed_effects"].drop(columns="response"))
    assert set(a["fixed_effects"]["response"]) == {"E_samp"} and set(b["fixed_effects"]["response"]) == {"E_prec"}
    assert summary["sa"]["response"] == "E_prec"
    np.testing.assert_allclose(a["predictions"]["e_samp"], b["predictions"]["e_prec"], rtol=1e-12)
    assert "e_prec" in b["predictions"].columns and "e_prec_a" in b["contrasts"].columns


def test_the_precision_models_differ_from_the_residual_models_when_pools_are_short(big_frame_short):
    a, _ = ana.model_tables(big_frame_short, MS, SETTINGS, "instance_median", response="E_samp")
    b, _ = ana.model_tables(big_frame_short, MS, SETTINGS, "instance_median", response="E_prec")
    ea = a["fixed_effects"].set_index(["sampler", "term"])["estimate"]
    eb = b["fixed_effects"].set_index(["sampler", "term"])["estimate"]
    assert (ea - eb).abs().max() > 1e-3  # the price of a small pool is not in E_prec


# --------------------------------------------------------------------------------------
# Long frames for the models
# --------------------------------------------------------------------------------------


def test_long_frames_by_seed_hold_one_row_per_cell_and_pool_size(frame):
    longs = ana.long_frames(frame, MS, SETTINGS, unit="seed")
    assert set(longs) == {"sa", "tabu", "swap_sa"}
    swap = longs["swap_sa"]
    assert len(swap) == (frame["sampler"] == "swap_sa").sum() * len(MS)
    assert set(swap.columns) >= {"instance", "n", "h", "log_m", "y"}
    src = frame[frame["sampler"] == "swap_sa"].iloc[0]
    got = swap[(swap["instance"] == src["instance"]) & (swap["log_m"] == math.log(20))].iloc[0]
    assert got["y"] in set(np.arcsinh(frame[(frame["sampler"] == "swap_sa") & (frame["instance"] == src["instance"])]
                                      ["E_samp_20"] / TAU))
    assert {"penalty", "beta"} <= set(longs["sa"].columns) and "beta" not in longs["tabu"].columns


def test_long_frames_by_instance_median_hold_one_row_per_instance_configuration_and_pool_size(frame):
    longs = ana.long_frames(frame, MS, SETTINGS, unit="instance_median")
    sa = longs["sa"]
    assert len(sa) == 40 * 4 * len(MS)
    one = sa[(sa["instance"] == "16/2000/0.45") & (sa["penalty"] == 10.0) & (sa["beta"] == "default")
             & (sa["log_m"] == math.log(5))].iloc[0]
    cell = frame[(frame["instance"] == "16/2000/0.45") & (frame["config_id"] == "sa/p10/default")]
    assert one["y"] == pytest.approx(math.asinh(cell["E_samp_5"].median() / TAU))


def test_long_frames_drop_cells_without_a_response():
    df = ana.frame_from_rows(synthetic_rows(seed=7, no_valid_share=0.2), MS)
    longs = ana.long_frames(df, MS, SETTINGS, unit="seed")
    assert longs["swap_sa"]["y"].notna().all()
    assert len(longs["swap_sa"]) == df[df["sampler"] == "swap_sa"][[f"E_samp_{m}" for m in MS]].notna().to_numpy().sum()


def test_an_unknown_model_unit_is_refused(frame):
    with pytest.raises(ValueError, match="unit"):
        ana.long_frames(frame, MS, SETTINGS, unit="cell")


# --------------------------------------------------------------------------------------
# The models and the contrast between samplers
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big_frame() -> pd.DataFrame:
    return ana.frame_from_rows(synthetic_rows(seed=8, n_pairs={16: 30, 24: 30}), MS)


@pytest.mark.parametrize("unit", ["instance_median", "seed"])
def test_the_models_recover_the_generator_and_the_contrast_the_sampler_shifts(big_frame, unit):
    tables, summary = ana.model_tables(big_frame, MS, SETTINGS, unit)
    fe = tables["fixed_effects"]
    sa_h = fe[(fe["sampler"] == "sa") & (fe["term"] == "C(h)[T.0.45]")].iloc[0]
    assert sa_h["estimate"] == pytest.approx(0.40, abs=0.15) and sa_h["ci_low"] < sa_h["estimate"] < sa_h["ci_high"]
    assert set(fe["sampler"]) == {"sa", "tabu", "swap_sa"} and set(tables) >= {"fixed_effects", "predictions", "contrasts"}
    # contrasts at the references: tabu − sa ≈ +0.3 (the generator's shift), swap − sa ≈ −0.05
    c = tables["contrasts"]
    c = c[c["role"] == "reference"]
    tabu_sa = c[(c["a"] == "tabu") & (c["b"] == "sa")]
    assert tabu_sa["estimate"].mean() == pytest.approx(0.30, abs=0.15)
    swap_sa = c[(c["a"] == "swap_sa") & (c["b"] == "sa")]
    assert swap_sa["estimate"].mean() == pytest.approx(-0.05, abs=0.15)
    assert set(c["m"]) == set(MS) and len(c) == 3 * len(MS)
    for sampler in ("sa", "tabu", "swap_sa"):
        s = summary[sampler]
        assert s["converged"] is True and s["n_groups"] == 120 and s["random_var"] > 0.0
        assert s["unit"] == unit and s["formula"].startswith("y ~ C(n) + C(h)")
    assert set(tables["predictions"]["sampler"]) == {"sa", "tabu", "swap_sa"}


def test_the_seed_level_and_instance_level_models_see_the_same_signs(big_frame):
    a, _ = ana.model_tables(big_frame, MS, SETTINGS, "instance_median")
    b, _ = ana.model_tables(big_frame, MS, SETTINGS, "seed")
    ea = a["fixed_effects"].set_index(["sampler", "term"])["estimate"]
    eb = b["fixed_effects"].set_index(["sampler", "term"])["estimate"]
    key = ("sa", "log_m")
    assert np.sign(ea[key]) == np.sign(eb[key]) and abs(ea[key] - eb[key]) < 0.1


def test_a_reference_configuration_outside_the_design_is_refused():
    """The contrast is made at the references (§12 B3): a design without one cannot be analysed."""
    rows = synthetic_rows(seed=16, n_pairs={16: 8, 24: 8}, penalties=(10.0, 30.0))  # no penalty of 100
    df = ana.frame_from_rows(rows, MS)
    with pytest.raises(ValueError, match=r"reference.*(sa|tabu).*penalty"):
        ana.model_tables(df, MS, SETTINGS, "instance_median")


def test_samplers_without_data_are_skipped_with_a_note(big_frame):
    only_sa = big_frame[big_frame["sampler"] == "sa"]
    tables, summary = ana.model_tables(only_sa, MS, SETTINGS, "instance_median")
    assert set(tables["fixed_effects"]["sampler"]) == {"sa"} and len(tables["contrasts"]) == 0
    assert "tabu" in summary["skipped"] and "swap_sa" in summary["skipped"]


# --------------------------------------------------------------------------------------
# B3: references, the nested grid, the exploratory level, configurations chosen after the grid
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warm_frame() -> pd.DataFrame:
    """Rows with the exploratory level of swap-SA (T_cold = 0.2 T_hot) next to the reference."""
    return ana.frame_from_rows(synthetic_rows(seed=17, n_pairs={16: 12, 24: 12}, cold_ratios=(0.01, 0.2)), MS)


def test_the_rows_state_the_cold_end_of_swap_sa_and_the_frame_derives_the_role(warm_frame):
    swap = warm_frame[warm_frame["sampler"] == "swap_sa"]
    assert set(swap["cold_ratio"]) == {0.01, 0.2} and warm_frame.loc[warm_frame["sampler"] != "swap_sa",
                                                                      "cold_ratio"].isna().all()
    roles = warm_frame.drop_duplicates("config_id").set_index("config_id")["role"].to_dict()
    assert roles == {"sa/p10/default": "grid", "sa/p10/objective": "grid", "sa/p100/default": "reference",
                     "sa/p100/objective": "grid", "tabu/p10": "grid", "tabu/p100": "reference",
                     "swap_sa/c0.01": "reference", "swap_sa/c0.2": "exploratory"}


def test_rows_without_the_cold_ratio_are_refused(rows):
    old = [{k: v for k, v in r.items() if k != "cold_ratio"} for r in rows[:12]]
    with pytest.raises(ValueError, match="cold_ratio"):
        ana.frame_from_rows(old, MS)


def test_tables_by_configuration_carry_the_cold_ratio_and_the_role(warm_frame):
    for table in (ana.samp_table(warm_frame, MS, SETTINGS), ana.decomposition_table(warm_frame, MS, SETTINGS),
                  ana.sampler_diagnostics_table(warm_frame), ana.pool_curve_table(warm_frame, MS, SETTINGS)):
        assert {"cold_ratio", "role"} <= set(table.columns)
        swap = table[table["sampler"] == "swap_sa"].drop_duplicates("config_id").set_index("config_id")
        assert set(swap.index) == {"swap_sa/c0.01", "swap_sa/c0.2"}
        assert swap.loc["swap_sa/c0.2", "role"] == "exploratory" and swap.loc["swap_sa/c0.2", "cold_ratio"] == 0.2
        assert swap.loc["swap_sa/c0.01", "role"] == "reference"
        assert table.loc[table["sampler"] != "swap_sa", "cold_ratio"].isna().all()
        ref = table[table["role"] == "reference"]["config_id"].unique()
        assert sorted(ref) == ["sa/p100/default", "swap_sa/c0.01", "tabu/p100"]


def test_the_sampler_table_describes_every_level_of_the_grid_the_exploratory_one_too(warm_frame):
    table = ana.samp_table(warm_frame, MS, SETTINGS)
    assert set(table["config_id"]) == {"sa/p10/default", "sa/p10/objective", "sa/p100/default", "sa/p100/objective",
                                       "tabu/p10", "tabu/p100", "swap_sa/c0.01", "swap_sa/c0.2"}
    warm = table[(table["config_id"] == "swap_sa/c0.2") & (table["m"] == 20) & (table["group"] == "all")].iloc[0]
    med = _instance_medians(warm_frame, "swap_sa/c0.2", "E_samp_20")
    assert warm["n_instances"] == 48 and warm["e_samp_median"] == pytest.approx(med.median())


def test_the_reference_swap_sa_model_does_not_see_the_exploratory_level(warm_frame):
    """§12 B3: the exploratory level is an addition; the model of the plan is fitted to the rows of the
    plan, so it is the model that the grid without the addition gives, to the last bit."""
    plain = warm_frame[warm_frame["cold_ratio"].isna() | (warm_frame["cold_ratio"] == 0.01)]
    with_extra, summary = ana.model_tables(warm_frame, MS, SETTINGS, "instance_median")
    without, _ = ana.model_tables(plain, MS, SETTINGS, "instance_median")
    ref_a = with_extra["fixed_effects"].query("model == 'swap_sa'").drop(columns=["model", "exploratory"])
    ref_b = without["fixed_effects"].query("model == 'swap_sa'").drop(columns=["model", "exploratory"])
    pd.testing.assert_frame_equal(ref_a.reset_index(drop=True), ref_b.reset_index(drop=True))
    assert set(with_extra["fixed_effects"]["model"]) == {"sa", "tabu", "swap_sa", "swap_sa[cold_ratio=0.2]"}
    assert set(without["fixed_effects"]["model"]) == {"sa", "tabu", "swap_sa"}
    extra = with_extra["fixed_effects"].query("model == 'swap_sa[cold_ratio=0.2]'")
    assert (extra["sampler"] == "swap_sa").all() and extra["exploratory"].all()
    assert not with_extra["fixed_effects"].query("model != 'swap_sa[cold_ratio=0.2]'")["exploratory"].any()
    assert summary["swap_sa[cold_ratio=0.2]"]["exploratory_level"] is True
    assert summary["swap_sa"]["exploratory_level"] is False and summary["swap_sa"]["exploratory"] is True
    # the generator's shift of the warmer end (-0.8) shows in the intercept of its own model
    intercept = extra.set_index("term").loc["Intercept", "estimate"]
    base = ref_a.set_index("term").loc["Intercept", "estimate"]
    assert intercept < base - 0.4


def test_predictions_are_made_at_every_level_of_the_nested_grid_with_their_roles(warm_frame):
    tables, _ = ana.model_tables(warm_frame, MS, SETTINGS, "instance_median")
    pred = tables["predictions"]
    cfg = pred.drop_duplicates("config_id").set_index("config_id")
    assert set(cfg.index) == {"sa/p10/default", "sa/p10/objective", "sa/p100/default", "sa/p100/objective",
                              "tabu/p10", "tabu/p100", "swap_sa/c0.01", "swap_sa/c0.2"}
    assert cfg["role"].to_dict() == {"sa/p10/default": "grid", "sa/p10/objective": "grid",
                                     "sa/p100/default": "reference", "sa/p100/objective": "grid",
                                     "tabu/p10": "grid", "tabu/p100": "reference",
                                     "swap_sa/c0.01": "reference", "swap_sa/c0.2": "exploratory"}
    assert len(pred) == 8 * len(MS) and set(pred["m"]) == set(MS)
    assert cfg.loc["swap_sa/c0.2", "model"] == "swap_sa[cold_ratio=0.2]" and cfg.loc["tabu/p10", "model"] == "tabu"
    ref = pred[(pred["config_id"] == "sa/p100/default")].sort_values("m")
    assert (ref["e_samp_ci_low"] < ref["e_samp"]).all() and (ref["e_samp"] < ref["e_samp_ci_high"]).all()


def test_contrasts_are_made_at_the_references_and_described_over_the_grid(warm_frame):
    tables, _ = ana.model_tables(warm_frame, MS, SETTINGS, "instance_median")
    c = tables["contrasts"]
    # pairs of configurations: tabu×sa 2×4, swap_sa×sa 2×4 (reference and exploratory level), swap_sa×tabu 2×2
    assert len(c) == (8 + 8 + 4) * len(MS)
    ref = c[c["role"] == "reference"]
    assert len(ref) == 3 * len(MS)
    assert set(zip(ref["config_a"], ref["config_b"], strict=True)) == {
        ("tabu/p100", "sa/p100/default"), ("swap_sa/c0.01", "sa/p100/default"), ("swap_sa/c0.01", "tabu/p100")}
    exploratory = c[c["role"] == "exploratory"]
    assert set(exploratory["config_a"]) == {"swap_sa/c0.2"} and len(exploratory) == (4 + 2) * len(MS)
    grid = c[c["role"] == "grid"]
    assert "swap_sa/c0.2" not in set(grid["config_a"]) and len(grid) == 11 * len(MS)  # 20 pairs − 3 − 6
    assert set(c["role"]) == {"reference", "grid", "exploratory"}


def test_a_contrast_of_the_grid_is_the_difference_of_the_predictions_of_its_configurations(warm_frame):
    tables, _ = ana.model_tables(warm_frame, MS, SETTINGS, "instance_median")
    pred, c = tables["predictions"], tables["contrasts"]
    row = c[(c["config_a"] == "tabu/p10") & (c["config_b"] == "sa/p100/objective") & (c["m"] == 20)].iloc[0]
    a = pred[(pred["config_id"] == "tabu/p10") & (pred["m"] == 20)].iloc[0]
    b = pred[(pred["config_id"] == "sa/p100/objective") & (pred["m"] == 20)].iloc[0]
    assert row["estimate"] == pytest.approx(a["estimate"] - b["estimate"])
    assert row["se"] == pytest.approx(math.hypot(a["se"], b["se"]))
    assert (row["a"], row["b"], row["role"]) == ("tabu", "sa", "grid")
    assert row["e_samp_a"] == pytest.approx(a["e_samp"]) and row["e_samp_b"] == pytest.approx(b["e_samp"])


def test_a_configuration_chosen_after_the_grid_is_exploratory_everywhere():
    """§12 B3: only the references are the plan's; whatever the analyst picks after seeing the grid is
    labelled so in every table that has a role, and its contrasts are exploratory."""
    df = ana.frame_from_rows(synthetic_rows(seed=18, n_pairs={16: 12, 24: 12}), MS)
    chosen = [{"sampler": "sa", "penalty": 10.0, "beta": "objective"}, {"sampler": "tabu", "penalty": 10.0}]
    settings = {**SETTINGS, "models": {**SETTINGS["models"], "chosen_after_grid": chosen}}
    marked = ana.mark_chosen_after_grid(df, settings)
    roles = marked.drop_duplicates("config_id").set_index("config_id")["role"].to_dict()
    assert roles["sa/p10/objective"] == roles["tabu/p10"] == "exploratory"
    assert roles["sa/p100/default"] == "reference" and roles["tabu/p100"] == "reference"
    assert roles["sa/p10/default"] == "grid"
    assert (df["role"] != "exploratory").all()  # the input frame is not modified
    tables, _ = ana.model_tables(marked, MS, settings, "instance_median")
    c = tables["contrasts"]
    mine = c[(c["config_a"] == "tabu/p10") | (c["config_b"] == "sa/p10/objective") | (c["config_b"] == "tabu/p10")]
    assert not mine.empty and (mine["role"] == "exploratory").all()
    assert (c[c["role"] == "reference"]["config_a"].isin(["tabu/p100", "swap_sa/c0.01"])).all()
    for table in (ana.samp_table(marked, MS, SETTINGS), ana.decomposition_table(marked, MS, SETTINGS)):
        got = table.drop_duplicates("config_id").set_index("config_id")["role"]
        assert got["sa/p10/objective"] == "exploratory" and got["sa/p100/default"] == "reference"


def test_a_reference_cannot_be_demoted_by_a_post_hoc_choice():
    df = ana.frame_from_rows(synthetic_rows(seed=18, n_pairs={16: 4, 24: 4}), MS)
    settings = {**SETTINGS, "models": {**SETTINGS["models"],
                                       "chosen_after_grid": [{"sampler": "tabu", "penalty": 100.0}]}}
    with pytest.raises(ValueError, match=r"reference.*chosen_after_grid"):
        ana.mark_chosen_after_grid(df, settings)


def test_a_chosen_configuration_that_is_not_in_the_rows_is_refused():
    df = ana.frame_from_rows(synthetic_rows(seed=18, n_pairs={16: 4, 24: 4}), MS)
    settings = {**SETTINGS, "models": {**SETTINGS["models"],
                                       "chosen_after_grid": [{"sampler": "sa", "penalty": 7.0, "beta": "default"}]}}
    with pytest.raises(ValueError, match=r"chosen_after_grid.*not in the rows"):
        ana.mark_chosen_after_grid(df, settings)


# --------------------------------------------------------------------------------------
# The analysis settings and the whole stage on a registered run
# --------------------------------------------------------------------------------------

YAML_SETTINGS = {
    "tau": 1.0e-3, "zero_tol": 1.0e-8, "alpha": 0.05, "share_ms": [1, 50], "level": 0.95,
    "bootstrap": {"n_boot": 199, "seed": 50000}, "h1c": {"p_value": None},
    "models": {"units": ["instance_median"], "responses": ["E_samp", "E_prec"],
               "reference": {"sa": {"penalty": 100, "beta": "default"}, "tabu": {"penalty": 100},
                             "swap_sa": {"cold_ratio": 0.01}},
               "chosen_after_grid": []},
}


def _grid_config(m0: int | None = 50, cold_ratios=(0.01,)) -> dict:
    """The confirmatory config whose grid is exactly the cells of ``synthetic_rows`` (20 pairs)."""
    pilot = None if m0 is None else {
        "source": None, "m0": m0,
        "work": [{"sampler": s, "n": n, **w} for n in (16, 24) for s, w in (
            ("sa", {"num_reads": 10, "num_sweeps": 10}),
            ("tabu", {"num_reads": 2, "num_restarts": 0, "tenure": e1.tabu_tenure(n)}),
            ("swap_sa", {"num_reads": 5, "num_sweeps": 5}))]}
    params = {
        "seed_role": "test", "cell_order": "interleaved",
        "instances": [{"n": 16, "pair_seeds": list(range(2000, 2010))},
                      {"n": 24, "pair_seeds": list(range(2010, 2020))}],
        "h_levels": [0.15, 0.45], "lam": 5.0, "w_min": 0.0, "w_max": 1.0, "ms": list(MS),
        "solver_seeds": [20000, 20001, 20002],
        "scip": {"formulation": "perspective", "diag": "mineig", "mip_gap": 1e-6, "time_limit_s": 600.0, "seed": 0},
        "samplers": [{"name": "sa", "penalty": [10.0, 100.0], "beta": ["default", "objective"]},
                     {"name": "tabu", "penalty": [10.0, 100.0]},
                     {"name": "swap_sa", "cold_ratio": list(cold_ratios)}],
    }
    if pilot is not None:
        params["pilot"] = pilot
    else:  # flat work, as the smoke config has it
        params["samplers"] = [{**s, **({"num_reads": 5, "num_sweeps": 5} if s["name"] != "tabu" else
                                       {"num_reads": 2, "num_restarts": 0, "tenure": 5})}
                              for s in params["samplers"]]  # the flat path takes the work as it is
    seeds = sorted({*range(2000, 2020), 20000, 20001, 20002})
    return {"stream": "E1", "name": "attribution", "experiment": "cif.experiments.e1_attribution",
            "seeds": seeds, "inputs": {}, "params": params, "mode": "full"}


def _write_run(root, rows, cfg, *, mode="full", git_dirty=False, run_id="20261201-000000-E1-attribution-abc123"):
    d = root / run_id
    d.mkdir(parents=True)
    manifest = {"run_id": run_id, "stream": "E1", "name": "attribution", "config_hash": "abc123def456",
                "git_sha": "deadbeef", "git_dirty": git_dirty, "mode": mode, "status": "completed",
                "n_errors": 0, "n_rows": len(rows)}
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "config.json").write_text(json.dumps(cfg, sort_keys=True))
    (d / "results.jsonl").write_text("".join(json.dumps({"run_id": run_id, **r}) + "\n" for r in rows))
    return d


def test_the_synthetic_rows_are_the_cells_of_the_grid_config():
    """The fixture of the stage tests is a faithful registered run: same cells, same keys."""
    cells = e1.cells(_grid_config())
    keys = {runner.cell_key(c) for c in cells}
    assert keys == {r["cell_key"] for r in synthetic_rows(seed=1)} and len(cells) == 840
    both = e1.cells(_grid_config(cold_ratios=(0.01, 0.2)))  # with the exploratory level of swap-SA (§12 B3)
    assert {runner.cell_key(c) for c in both} == {
        r["cell_key"] for r in synthetic_rows(seed=1, cold_ratios=(0.01, 0.2))} and len(both) == 960


def test_analysis_settings_are_complete_and_sane(tmp_path):
    assert ana.validate_settings(YAML_SETTINGS) == YAML_SETTINGS
    for key in YAML_SETTINGS:
        broken = {k: v for k, v in YAML_SETTINGS.items() if k != key}
        with pytest.raises(ValueError, match=key):
            ana.validate_settings(broken)
    for key, bad in (("tau", 0.0), ("alpha", 1.5), ("level", 1.0), ("zero_tol", -1.0), ("share_ms", [])):
        with pytest.raises(ValueError, match=key):
            ana.validate_settings({**YAML_SETTINGS, key: bad})
    with pytest.raises(ValueError, match="n_boot"):
        ana.validate_settings({**YAML_SETTINGS, "bootstrap": {"n_boot": 0, "seed": 1}})
    with pytest.raises(ValueError, match="units"):
        ana.validate_settings({**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "units": ["cell"]}})
    for bad in ([], ["E_pipe"], ["E_samp", "E_samp"], "E_prec"):  # §12 B2: the responses are stated, E_samp and E_prec
        with pytest.raises(ValueError, match="responses"):
            ana.validate_settings({**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "responses": bad}})
    for sampler, other in (("sa", {"penalty": 30, "beta": "default"}), ("tabu", {"penalty": 10}),
                           ("swap_sa", {"cold_ratio": 0.2})):  # §12 B3: the references are the plan's
        changed = {**YAML_SETTINGS["models"]["reference"], sampler: other}
        with pytest.raises(ValueError, match=r"reference.*B3.*chosen_after_grid"):
            ana.validate_settings({**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "reference": changed}})
    missing_reference = {k: v for k, v in YAML_SETTINGS["models"].items() if k != "reference"}
    with pytest.raises(ValueError, match="reference"):
        ana.validate_settings({**YAML_SETTINGS, "models": missing_reference})
    for bad in ({"sa": {}}, "sa", [{"sampler": "annealer9000"}], [{"sampler": "sa", "penalty": 10}],
                [{"sampler": "tabu", "beta": "default", "penalty": 10}], [{"penalty": 10}]):
        with pytest.raises(ValueError, match="chosen_after_grid"):
            ana.validate_settings({**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "chosen_after_grid": bad}})
    ok = {"sampler": "sa", "penalty": 30, "beta": "objective"}
    assert ana.validate_settings({**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"],
                                                              "chosen_after_grid": [ok]}})
    for bad_m0 in (0, -1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="m0"):
            ana.validate_settings({**YAML_SETTINGS, "h1a": {"m0": bad_m0}})
    assert ana.validate_settings({**YAML_SETTINGS, "h1a": {"m0": 1}})
    no_responses = {k: v for k, v in YAML_SETTINGS["models"].items() if k != "responses"}
    with pytest.raises(ValueError, match="responses"):
        ana.validate_settings({**YAML_SETTINGS, "models": no_responses})
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(YAML_SETTINGS))
    assert ana.load_settings(path) == YAML_SETTINGS


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    return _write_run(root, synthetic_rows(seed=1), _grid_config())


@pytest.fixture(scope="module")
def stage(run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("analysis")
    result = ana.analyse(run_dir, YAML_SETTINGS, out)
    return out, result


EXPECTED_FILES = {
    "h1a.json", "instances.csv", "pairs.csv", "design.csv", "shares_e_form.csv", "e_form.csv",
    "e_form_difference.csv", "e_samp.csv", "e_decomposition.csv", "sampler_diagnostics.csv", "pool_curve.csv",
    "models_instance_median.json", "fixed_effects_instance_median.csv", "predictions_instance_median.csv",
    "contrasts_instance_median.csv",
    "models_prec_instance_median.json", "fixed_effects_prec_instance_median.csv",
    "predictions_prec_instance_median.csv", "contrasts_prec_instance_median.csv",
    "diagnostics.json", "analysis_manifest.json",
}


def test_the_stage_writes_the_tables_of_the_plan(stage):
    out, result = stage
    assert {p.name for p in out.iterdir()} == EXPECTED_FILES
    assert set(result["files"]) == EXPECTED_FILES


def test_h1a_json_is_the_test_at_the_m0_of_the_runs_pilot_block(stage, pairs):
    out, _ = stage
    h1a = json.loads((out / "h1a.json").read_text())
    expect = ana.h1a_result(ana.pair_frame(ana.instance_frame(ana.frame_from_rows(synthetic_rows(seed=1), MS), MS),
                                           MS, 0.15, 0.45), 50, YAML_SETTINGS)
    assert h1a["m0"] == 50 and h1a["p_value"] == expect["p_value"] and h1a["n_pairs"] == 20
    assert h1a["holm"]["complete"] is False and h1a["outcome"] in {"confirmed", "pending_h1c", "not_confirmed"}


def test_the_manifest_of_the_analysis_records_what_was_analysed(stage):
    out, _ = stage
    man = json.loads((out / "analysis_manifest.json").read_text())
    assert man["run"]["run_id"].endswith("abc123") and man["run"]["seed_role"] == "test"
    assert man["m0"] == {"value": 50, "source": "run config params.pilot.m0"}
    assert man["settings"]["bootstrap"]["seed"] == 50000
    assert man["confirmatory"] is True and man["warnings"] == []
    for name, digest in man["outputs"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert "analysis_manifest.json" not in man["outputs"]
    assert {"numpy", "pandas", "scipy", "statsmodels"} <= set(man["versions"])
    assert set(man["outputs"]) == EXPECTED_FILES - {"analysis_manifest.json"}


def test_every_output_carries_its_status_in_the_plan(stage):
    """§7.1 / §9: only the H1a test is confirmatory; the models and contrasts (§7.3) are exploratory and
    must be labelled so; the rest is estimates and descriptive tables."""
    out, _ = stage
    man = json.loads((out / "analysis_manifest.json").read_text())
    status = man["output_status"]
    assert set(status) == EXPECTED_FILES - {"analysis_manifest.json"}
    assert set(status.values()) <= {"confirmatory", "estimate", "descriptive", "exploratory", "data", "diagnostic"}
    assert status["h1a.json"] == "confirmatory"
    assert {n for n, s in status.items() if s == "confirmatory"} == {"h1a.json"}
    for name in status:
        if name.startswith(("models_", "fixed_effects_", "predictions_", "contrasts_")):
            assert status[name] == "exploratory", name
    assert status["shares_e_form.csv"] == status["e_samp.csv"] == status["e_decomposition.csv"] == "estimate"
    assert status["pool_curve.csv"] == "descriptive" and status["instances.csv"] == "data"
    models = json.loads((out / "models_instance_median.json").read_text())
    assert all(models[s]["exploratory"] is True for s in ("sa", "tabu", "swap_sa"))
    prec = json.loads((out / "models_prec_instance_median.json").read_text())  # §12 B2: the same models
    assert all(prec[s]["exploratory"] is True and prec[s]["response"] == "E_prec" for s in ("sa", "tabu", "swap_sa"))
    assert all(models[s]["response"] == "E_samp" for s in ("sa", "tabu", "swap_sa"))


def test_the_stage_is_deterministic(run_dir, stage, tmp_path):
    out, _ = stage
    ana.analyse(run_dir, YAML_SETTINGS, tmp_path)
    for name in EXPECTED_FILES - {"analysis_manifest.json"}:
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes(), name
    a = json.loads((out / "analysis_manifest.json").read_text())
    b = json.loads((tmp_path / "analysis_manifest.json").read_text())
    a.pop("generated_at"), b.pop("generated_at")
    assert a == b


def test_the_design_and_diagnostics_files_report_what_the_plan_asks(stage):
    out, _ = stage
    design = pd.read_csv(out / "design.csv")
    assert {"share_binding", "depth_median", "cv_vol_mean"} <= set(design.columns)
    diag = json.loads((out / "diagnostics.json").read_text())
    assert diag["frame"]["n_rows"] == 840 and diag["frame"]["n_instances"] == 40
    assert diag["frame"]["n_missing"] == 0 and diag["ms"] == list(MS)


def test_the_h1c_p_value_of_the_analysis_file_and_of_the_command_line(run_dir, tmp_path):
    out = tmp_path / "a"
    ana.analyse(run_dir, {**YAML_SETTINGS, "h1c": {"p_value": 0.2}}, out)
    holm = json.loads((out / "h1a.json").read_text())["holm"]
    assert holm["complete"] is True and holm["adjusted"]["H1c"] == pytest.approx(0.2)
    out2 = tmp_path / "b"
    ana.analyse(run_dir, YAML_SETTINGS, out2, h1c_p=0.4)
    assert json.loads((out2 / "h1a.json").read_text())["holm"]["adjusted"]["H1c"] == pytest.approx(0.4)
    man = json.loads((out2 / "analysis_manifest.json").read_text())
    assert man["settings"]["h1c"]["p_value"] == 0.4


def test_a_run_without_a_pilot_block_needs_m0_from_the_settings(tmp_path):
    d = _write_run(tmp_path / "r", synthetic_rows(seed=1), _grid_config(m0=None))
    with pytest.raises(ValueError, match="m0"):
        ana.analyse(d, YAML_SETTINGS, tmp_path / "o")
    out = tmp_path / "o2"
    ana.analyse(d, {**YAML_SETTINGS, "h1a": {"m0": 1}}, out)
    man = json.loads((out / "analysis_manifest.json").read_text())
    assert man["m0"] == {"value": 1, "source": "analysis settings h1a.m0"}


def test_the_stage_labels_the_exploratory_level_and_records_the_role_of_every_configuration(tmp_path):
    rows = synthetic_rows(seed=1, cold_ratios=(0.01, 0.2))
    d = _write_run(tmp_path / "r", rows, _grid_config(cold_ratios=(0.01, 0.2)))
    out = tmp_path / "o"
    ana.analyse(d, YAML_SETTINGS, out)
    man = json.loads((out / "analysis_manifest.json").read_text())
    assert man["configurations"] == {
        "sa/p10/default": "grid", "sa/p10/objective": "grid", "sa/p100/default": "reference",
        "sa/p100/objective": "grid", "tabu/p10": "grid", "tabu/p100": "reference",
        "swap_sa/c0.01": "reference", "swap_sa/c0.2": "exploratory"}
    assert man["settings"]["models"]["reference"]["swap_sa"] == {"cold_ratio": 0.01}
    for name in ("e_samp.csv", "e_decomposition.csv", "sampler_diagnostics.csv", "pool_curve.csv",
                 "predictions_instance_median.csv", "predictions_prec_instance_median.csv"):
        table = pd.read_csv(out / name)
        assert set(table.loc[table["config_id"] == "swap_sa/c0.2", "role"]) == {"exploratory"}, name
        assert set(table.loc[table["config_id"] == "swap_sa/c0.01", "role"]) == {"reference"}, name
    contrasts = pd.read_csv(out / "contrasts_instance_median.csv")
    assert {"config_a", "config_b", "role", "model_a", "model_b"} <= set(contrasts.columns)
    assert set(contrasts.loc[contrasts["config_a"] == "swap_sa/c0.2", "role"]) == {"exploratory"}
    models = json.loads((out / "models_instance_median.json").read_text())
    assert models["swap_sa[cold_ratio=0.2]"]["exploratory_level"] is True


def test_the_pool_size_of_h1a_may_come_from_the_settings_and_must_agree_with_the_run(tmp_path):
    """``h1a.m0`` of ``configs/E1/analysis.yaml`` (the pilot decided ``m0 = 1``, §12): used when the run's pilot
    block has none, and a contradiction with the block of the run is an error, not a choice."""
    cfg = _grid_config(m0=1)
    assert ana.resolve_m0(cfg, {**YAML_SETTINGS, "h1a": {"m0": 1}}, MS) == (1, "run config params.pilot.m0")
    with pytest.raises(ValueError, match="contradicts"):
        ana.resolve_m0(cfg, {**YAML_SETTINGS, "h1a": {"m0": 50}}, MS)
    none = _grid_config(m0=None)
    assert ana.resolve_m0(none, {**YAML_SETTINGS, "h1a": {"m0": 1}}, MS) == (1, "analysis settings h1a.m0")
    with pytest.raises(ValueError, match="pool sizes"):
        ana.resolve_m0(none, {**YAML_SETTINGS, "h1a": {"m0": 7}}, MS)


def test_an_incomplete_or_foreign_run_is_refused(tmp_path):
    rows = synthetic_rows(seed=1)
    short = _write_run(tmp_path / "s", rows[:-3], _grid_config())
    with pytest.raises(ValueError, match="missing"):
        ana.analyse(short, YAML_SETTINGS, tmp_path / "o1")
    cfg = _grid_config()
    cfg["experiment"] = "cif.experiments.toy"
    foreign = _write_run(tmp_path / "f", rows, cfg)
    with pytest.raises(ValueError, match="e1_attribution"):
        ana.analyse(foreign, YAML_SETTINGS, tmp_path / "o2")


def test_a_run_that_is_not_confirmatory_is_labelled_so(tmp_path):
    smoke = _write_run(tmp_path / "m", synthetic_rows(seed=1), _grid_config(), mode="smoke")
    ana.analyse(smoke, YAML_SETTINGS, tmp_path / "a")
    man = json.loads((tmp_path / "a" / "analysis_manifest.json").read_text())
    assert man["confirmatory"] is False and any("smoke" in w for w in man["warnings"])
    dirty = _write_run(tmp_path / "d", synthetic_rows(seed=1), _grid_config(), git_dirty=True)
    ana.analyse(dirty, YAML_SETTINGS, tmp_path / "b")
    man = json.loads((tmp_path / "b" / "analysis_manifest.json").read_text())
    assert man["confirmatory"] is False and any("dirty" in w for w in man["warnings"])


def test_the_command_line(run_dir, tmp_path, capsys):
    cfg_path = tmp_path / "analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(YAML_SETTINGS))
    out = tmp_path / "out"
    assert ana.main(["--run", str(run_dir), "--config", str(cfg_path), "--out", str(out), "--h1c-p", "0.3"]) == 0
    printed = capsys.readouterr().out
    assert "H1a" in printed and "m0 = 50" in printed and str(out) in printed
    assert (out / "h1a.json").exists()
    assert json.loads((out / "h1a.json").read_text())["holm"]["adjusted"]["H1c"] == pytest.approx(0.3)


def test_both_model_units_can_be_asked_for(run_dir, tmp_path):
    settings = {**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "units": ["instance_median", "seed"]}}
    out = tmp_path / "o"
    ana.analyse(run_dir, settings, out)
    assert (out / "models_seed.json").exists() and (out / "contrasts_instance_median.csv").exists()
    assert (out / "models_prec_seed.json").exists() and (out / "contrasts_prec_instance_median.csv").exists()
    seed_fixed = pd.read_csv(out / "fixed_effects_seed.csv")
    assert set(seed_fixed["unit"]) == {"seed"}


def test_a_single_response_writes_only_its_files(run_dir, tmp_path):
    only = {**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "responses": ["E_samp"]}}
    ana.analyse(run_dir, only, tmp_path / "a")
    names = {p.name for p in (tmp_path / "a").iterdir()}
    assert "models_instance_median.json" in names and not any("_prec_" in n for n in names)
    only = {**YAML_SETTINGS, "models": {**YAML_SETTINGS["models"], "responses": ["E_prec"]}}
    ana.analyse(run_dir, only, tmp_path / "b")
    names = {p.name for p in (tmp_path / "b").iterdir()}
    assert "models_prec_instance_median.json" in names and "models_instance_median.json" not in names


def test_the_stage_reports_the_precision_tables(stage):
    out, _ = stage
    table = pd.read_csv(out / "e_decomposition.csv")
    assert {"e_samp_mean", "e_prec_mean", "e_short_mean", "identity_gap", "share_short", "n_pipe_median"} <= set(table.columns)
    fe = pd.read_csv(out / "fixed_effects_prec_instance_median.csv")
    assert set(fe["response"]) == {"E_prec"} and set(fe["sampler"]) == {"sa", "tabu", "swap_sa"}
    pred = pd.read_csv(out / "predictions_prec_instance_median.csv")
    assert {"e_prec", "e_prec_ci_low", "e_prec_ci_high"} <= set(pred.columns)


def test_the_analysis_reads_the_rows_that_the_attribution_writes(tmp_path):
    """Not only the synthetic rows: a tiny real run (calibration seeds, a few reads) through the runner, then the
    analysis stage with the references, the exploratory level of swap-SA and the split of the residual (§12 B2,
    B3). The numbers mean nothing here; the schema, the roles and the identities do."""
    cfg = {
        "stream": "E1", "name": "attribution_chain_test", "experiment": "cif.experiments.e1_attribution",
        "seeds": [1000, 1001, 1002, 10000, 10001], "inputs": {},
        "params": {
            "seed_role": "calibration",
            "instances": [{"n": 12, "pair_seeds": [1000, 1001, 1002]}],
            "h_levels": [0.15, 0.45], "lam": 5.0, "w_min": 0.0, "w_max": 1.0, "ms": list(MS),
            "solver_seeds": [10000, 10001],
            "scip": {"formulation": "perspective", "diag": "mineig", "mip_gap": 1e-6, "time_limit_s": 120.0, "seed": 0},
            "samplers": [
                {"name": "sa", "penalty": [100], "beta": ["default", "objective"], "num_reads": 40, "num_sweeps": 30},
                {"name": "tabu", "penalty": [100], "num_reads": 12, "num_restarts": 0, "tenure": 5},
                {"name": "swap_sa", "cold_ratio": [0.01, 0.2], "num_reads": 20, "num_sweeps": 20},
            ],
        },
    }
    path = tmp_path / "chain.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    assert runner.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (run_dir,) = [d for d in runs.iterdir() if d.is_dir()]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 3 * 2 * 5 * 2 and not any(r.get("error") for r in rows)
    out = tmp_path / "analysis"
    settings = {**YAML_SETTINGS, "h1a": {"m0": 1}}
    result = ana.analyse(run_dir, settings, out)
    assert result["m0"] == 1 and result["confirmatory"] is False  # calibration seeds, a dirty tree
    man = json.loads((out / "analysis_manifest.json").read_text())
    assert man["configurations"] == {"sa/p100/default": "reference", "sa/p100/objective": "grid",
                                     "tabu/p100": "reference", "swap_sa/c0.01": "reference",
                                     "swap_sa/c0.2": "exploratory"}
    frame = ana.frame_from_rows(rows, MS)
    for m in MS:  # the identity in every real cell
        gap = frame[f"E_samp_{m}"] - frame[f"E_prec_{m}"] - frame[f"E_short_{m}"]
        assert gap.abs().max() < 1e-12 and (frame[f"E_short_{m}"] >= 0.0).all()
    table = pd.read_csv(out / "e_decomposition.csv")
    assert table["identity_gap"].abs().max() < 1e-12 and set(table["role"]) == {"reference", "grid", "exploratory"}
    contrasts = pd.read_csv(out / "contrasts_instance_median.csv")
    ref = contrasts[contrasts["role"] == "reference"]
    assert set(zip(ref["config_a"], ref["config_b"], strict=True)) == {
        ("tabu/p100", "sa/p100/default"), ("swap_sa/c0.01", "sa/p100/default"), ("swap_sa/c0.01", "tabu/p100")}
    assert (out / "models_prec_instance_median.json").exists()
