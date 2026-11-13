"""RQ3 on rank paths (plan v2.4, operator (г)): H3a on rank 2, H3b slope, block length, H3c.

ANALYSIS_PLAN §2 RQ3 and §7.4; consultation ``docs/reviews/2026-10-08-q1-rq3-operator-consult.md``.
Synthetic returns only. An "optimum" series and rank paths that are the optimum plus a known
daily mean shift (and a little independent noise), so the population Sharpe differences are known.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

import cif.stats as stats_pkg
from cif.stats import (
    calibrate_block_size,
    circular_block_bootstrap_indices,
    holm,
    ledoit_wolf_sharpe_test,
    sharpe_ratio,
)
from cif.stats.rq3 import (
    H3A_OUTCOMES,
    calibrate_joint_block,
    calibrate_logvar_block,
    dose_response,
    equivalence_test,
    h3a_outcome,
    joint_contrasts,
    variance_margin,
)
from cif.stats.rq3_ranks import (
    BLOCK_GRID_DAILY,
    CALIBRATION_N_BOOT,
    CALIBRATION_N_PSEUDO,
    H3B_RANKS,
    MIN_MEDIAN_GAP_BP,
    RANK_BIN_LABELS,
    calibrate_rank_block,
    calibrate_replica_block,
    holm_over_arms,
    rank2_outcome,
    rank_bins,
    rank_dose_response,
    rank_equivalence,
    rank_solver_dose,
    solver_residuals,
)
from cif.stats.sharpe import _var1_pseudo_chunks

ROOT252 = math.sqrt(252)
RANKS = H3B_RANKS  # (2, 3, 5, 9, 17, 33)
DOSES = (0.4, 0.9, 2.1, 5.0, 12.0, 30.0)  # ḡ_r in bp of CE: increasing in the rank, as in the plan


def _opt(T: int, rng: np.random.Generator, mu: float = 0.03, phi: float = 0.0) -> np.ndarray:
    z = rng.standard_normal(T + 200)
    x = signal.lfilter([1.0], [1.0, -phi], z)[200:] if phi else z[200:]
    return mu + x


def _rank_path(opt: np.ndarray, rng: np.random.Generator, shift: float, noise: float = 0.05):
    """A rank path: the optimum plus a constant daily shift and small independent noise."""
    return opt + shift + noise * rng.standard_normal(opt.size)


# ---------------------------------------------------------------------------------------------
# GMV: the variance test is the LW2008 scheme for the difference of log variances
# ---------------------------------------------------------------------------------------------


def test_no_stats_docstring_cites_the_unverified_ledoit_wolf_2011():
    """Ledoit & Wolf (2011) is not in refs.bib (``verify_bib`` has not confirmed it yet): the
    variance test is described as the LW2008 scheme and no module of ``cif.stats`` cites 2011."""
    pattern = re.compile(r"(Ledoit|\bLW\b)[^\n]{0,60}\b2011\b|\b2011\b[^\n]{0,30}(Ledoit|\bLW\b)")
    for path in sorted(Path(stats_pkg.__file__).parent.glob("*.py")):
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} cites Ledoit & Wolf (2011)"


def test_log_variance_se_is_the_hac_of_the_delta_method_influence_series():
    # two independent normal series: Var(log v̂) = 2/T each, hence se(difference) = sqrt(4/T)
    T = 6000
    ratios = []
    for seed in range(8):
        series = np.random.default_rng(100 + seed).standard_normal((T, 2))
        res = joint_contrasts(series, np.array([[-1.0, 1.0]]), functional="logvar", block=5,
                              n_boot=19, seed=1)
        ratios.append(res.se[0] / math.sqrt(4.0 / T))
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.03)
    assert min(ratios) > 0.9 and max(ratios) < 1.1


def test_log_variance_difference_of_proportional_series_has_no_sampling_error():
    # y = c·x: the difference of the log variances is 2·log c in every resample, the two
    # influence functions cancel and the standard error is zero
    x = np.random.default_rng(5).standard_normal(2000)
    res = joint_contrasts(np.column_stack([x, 1.1 * x]), np.array([[-1.0, 1.0]]),
                          functional="logvar", block=5, n_boot=99, seed=1)
    assert res.estimate[0] == pytest.approx(2.0 * math.log(1.1), rel=1e-12)
    assert res.se[0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(res.theta_star[:, 0], 2.0 * math.log(1.1), rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# plan constants
# ---------------------------------------------------------------------------------------------


def test_plan_constants_and_defaults():
    assert H3B_RANKS == (2, 3, 5, 9, 17, 33)
    assert BLOCK_GRID_DAILY == (1, 5, 10, 21, 42, 63, 126)
    assert (CALIBRATION_N_PSEUDO, CALIBRATION_N_BOOT) == (1000, 499)
    for fn in (rank_equivalence, rank_dose_response):
        sig = inspect.signature(fn).parameters
        assert sig["n_boot"].default == 4999 and sig["alpha"].default == 0.05  # B = 4999
        assert sig["seed"].default is inspect.Parameter.empty  # seeds are explicit (R8)
        assert sig["periods_per_year"].default == 252
    cal = inspect.signature(calibrate_rank_block).parameters
    assert cal["block_grid"].default == BLOCK_GRID_DAILY and cal["percent"].default is True
    assert cal["n_pseudo"].default == 1000 and cal["n_boot"].default == 499
    assert cal["seed"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------------------------
# H3a: rank 2 against the optimum, one TOST and one two-sided test per arm
# ---------------------------------------------------------------------------------------------


def test_outcome_table_is_the_four_outcomes_of_the_plan():
    assert rank2_outcome(equivalent=True, significant=False) == "equivalent"
    assert rank2_outcome(equivalent=True, significant=True) == "distinguishable_within_sesoi"
    assert rank2_outcome(equivalent=False, significant=True) == "different"
    assert rank2_outcome(equivalent=False, significant=False) == "inconclusive"
    # the same table as the replica version (iut over levels / any significant level)
    for eq, sig in itertools.product((True, False), repeat=2):
        assert rank2_outcome(equivalent=eq, significant=sig) == h3a_outcome(
            iut_passes=eq, any_significant=sig
        )
    assert {rank2_outcome(equivalent=e, significant=s)
            for e, s in itertools.product((True, False), repeat=2)} == set(H3A_OUTCOMES)


def _h3a(shift, *, T=3000, noise=0.05, margin=0.5, seed=21, **kw):
    rng = np.random.default_rng(seed)
    opt = _opt(T, rng)
    rank2 = _rank_path(opt, rng, shift, noise)
    return rank_equivalence(opt, rank2, margin, block=5, n_boot=499, seed=seed, **kw)


def test_equivalent_when_rank_2_is_inside_the_margin_and_not_different():
    res = _h3a(0.0)
    assert res.outcome == "equivalent" and res.equivalent and not res.significant
    assert res.margin == (-0.5, 0.5) and res.alpha == 0.05 and res.functional == "sharpe"
    assert res.block == 5 and res.n_boot == 499 and res.n_obs == 3000 and res.seed == 21


def test_distinguishable_but_within_the_margin():
    # +0.01 per day = +0.16 annual Sharpe: a tiny standard error (rank 2 ≈ optimum), inside ±0.5
    res = _h3a(0.01)
    assert res.outcome == "distinguishable_within_sesoi" and res.equivalent and res.significant
    assert res.estimate == pytest.approx(0.01 * ROOT252, abs=0.03)
    assert res.ci_tost[0] > -0.5 and res.ci_tost[1] < 0.5 and res.pvalue <= 0.05


def test_different_when_rank_2_leaves_the_margin():
    res = _h3a(0.05)  # +0.8 annual Sharpe
    assert res.outcome == "different" and not res.equivalent and res.significant
    assert res.pvalue_tost == 1.0  # the point estimate is outside the margin: no equivalence at any α


def test_inconclusive_when_neither_equivalence_nor_a_difference_can_be_shown():
    rng = np.random.default_rng(22)
    opt = _opt(500, rng)
    independent = 0.03 + rng.standard_normal(500)  # an unrelated path: a wide interval
    res = rank_equivalence(opt, independent, 0.5, block=5, n_boot=499, seed=22)
    assert res.outcome == "inconclusive" and not res.equivalent and not res.significant


def test_identical_paths_are_equivalent_without_sampling_error():
    x = _opt(300, np.random.default_rng(3))
    res = rank_equivalence(x, x.copy(), 0.5, block=5, n_boot=99, seed=1)
    assert res.estimate == 0.0 and res.se == 0.0 and res.outcome == "equivalent"
    assert res.pvalue == 1.0 and res.pvalue_tost == pytest.approx(1.0 / (2 * 100))


def test_tost_uses_the_90_percent_interval_and_the_two_sided_test_the_95_percent_one():
    res = _h3a(0.0)
    assert res.ci_tost[0] > res.ci[0] and res.ci_tost[1] < res.ci[1]  # 90 % inside 95 %
    assert (not res.ci[0] <= 0.0 <= res.ci[1]) == res.significant == (res.pvalue <= 0.05)
    wider = _h3a(0.0, alpha=0.025)  # α = 0.025: the TOST interval is the 95 % one
    assert wider.ci_tost[0] < res.ci_tost[0] and wider.ci_tost[1] > res.ci_tost[1]


@pytest.mark.parametrize("seed", range(6))
def test_tost_pvalue_is_the_smallest_alpha_at_which_the_interval_decision_passes(seed):
    """Holm over the secondary arms needs p-values: ``pvalue_tost`` must agree with the decision
    by the (1 − 2α) interval for every α, on the 1/800 grid of the bootstrap too."""
    rng = np.random.default_rng(100 + seed)
    opt = _opt(400, rng, phi=0.1)
    rank2 = _rank_path(opt, rng, 0.004 * seed, 0.3)
    alphas = (0.01, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.0137, 0.0663, 0.249)
    results = [rank_equivalence(opt, rank2, 0.6, block=5, n_boot=399, alpha=a, seed=seed)
               for a in alphas]
    assert len({r.pvalue_tost for r in results}) == 1  # one bootstrap, one p-value
    for a, r in zip(alphas, results, strict=True):
        assert r.equivalent == (r.pvalue_tost <= a), (a, r.pvalue_tost)
    # the boundary: the p-value itself passes, a hair below does not (the smallest passing α)
    p = results[0].pvalue_tost
    at = rank_equivalence(opt, rank2, 0.6, block=5, n_boot=399, alpha=p, seed=seed)
    below = rank_equivalence(opt, rank2, 0.6, block=5, n_boot=399, alpha=p * (1.0 - 1e-6),
                             seed=seed)
    assert at.equivalent and not below.equivalent


def test_h3a_is_the_ledoit_wolf_2008_bootstrap_of_the_pair():
    rng = np.random.default_rng(8)
    opt = _opt(600, rng, phi=0.1)
    rank2 = opt + 0.015 + 0.3 * rng.standard_normal(600)
    res = rank_equivalence(opt, rank2, 0.5, block=5, n_boot=399, seed=7)
    lw = ledoit_wolf_sharpe_test(rank2, opt, method="bootstrap", block=5, n_boot=399, seed=7,
                                 periods_per_year=252)
    assert res.estimate == pytest.approx(lw.difference, rel=1e-10)
    # the standard error uses a scalar HAC of the influence function: within 5 % of LW's
    assert res.se == pytest.approx(lw.se, rel=0.05)
    assert res.ci == pytest.approx((lw.ci_low, lw.ci_high), rel=0.05, abs=0.02)


def test_gmv_arm_tests_the_log_variance_ratio_against_an_asymmetric_log_margin():
    rng = np.random.default_rng(23)
    opt = 0.01 * _opt(3000, rng, mu=0.0)
    margin = variance_margin(5.0, 0.04)  # δ = 0.025
    e = 0.05

    def rank2(scale):
        # population variance scale² · var(opt): sqrt(1 − e²)·opt + e·(an independent copy)
        return scale * (math.sqrt(1 - e**2) * opt + e * 0.01 * rng.standard_normal(opt.size))

    same = rank_equivalence(opt, rank2(1.0), (margin.low, margin.high), functional="logvar",
                            block=5, n_boot=499, seed=1)
    assert same.outcome == "equivalent" and same.margin == (margin.low, margin.high)
    assert same.estimate == pytest.approx(0.0, abs=1e-2)
    wider = rank_equivalence(opt, rank2(1.10), (margin.low, margin.high), functional="logvar",
                             block=5, n_boot=499, seed=1)
    assert wider.outcome == "different" and wider.functional == "logvar"
    assert wider.estimate == pytest.approx(math.log(1.10**2), abs=0.02)


def test_holm_adjusts_the_two_secondary_arms_separately_for_tost_and_for_the_two_sided_test():
    base = _h3a(0.0)
    a = dataclasses.replace(base, pvalue_tost=0.03, pvalue=0.5, equivalent=True, significant=False)
    b = dataclasses.replace(base, pvalue_tost=0.04, pvalue=0.01, equivalent=True, significant=True)
    out = holm_over_arms({"FF49-GMV": a, "US-MV": b})
    np.testing.assert_allclose(
        [out["FF49-GMV"].pvalue_tost_holm, out["US-MV"].pvalue_tost_holm], holm([0.03, 0.04])
    )
    np.testing.assert_allclose(
        [out["FF49-GMV"].pvalue_holm, out["US-MV"].pvalue_holm], holm([0.5, 0.01])
    )
    # unadjusted both pass the TOST; after Holm 2·0.03 = 0.06 > 0.05 and neither does
    assert not out["FF49-GMV"].equivalent and not out["US-MV"].equivalent
    assert out["US-MV"].significant and not out["FF49-GMV"].significant  # 0.02 vs 0.5
    assert out["FF49-GMV"].outcome == "inconclusive" and out["US-MV"].outcome == "different"
    for d in out.values():
        assert d.outcome == rank2_outcome(equivalent=d.equivalent, significant=d.significant)


def test_holm_over_arms_can_only_make_the_decisions_more_conservative():
    arms = {"FF49-GMV": _h3a(0.0, seed=31), "US-MV": _h3a(0.002, seed=32)}
    out = holm_over_arms(arms)
    for name, raw in arms.items():
        assert out[name].equivalent <= raw.equivalent and out[name].significant <= raw.significant
    with pytest.raises(ValueError, match="two"):
        holm_over_arms({"FF49-GMV": arms["FF49-GMV"]})


def test_tost_needs_enough_resamples_for_its_interval():
    # the (1 − 2α) quantile of d* exists only from M ≥ 1/(2α) − 1 = 9 resamples at α = 0.05; a smoke
    # run with fewer must fail with a clear message, not with "need finite ci_low <= ci_high"
    rng = np.random.default_rng(26)
    opt = _opt(200, rng)
    rank2 = _rank_path(opt, rng, 0.0)
    with pytest.raises(ValueError, match="n_boot"):
        rank_equivalence(opt, rank2, 0.5, block=4, n_boot=5, seed=1)
    ok = rank_equivalence(opt, rank2, 0.5, block=4, n_boot=9, seed=1)  # the smallest admissible
    assert math.isfinite(ok.ci_tost[0]) and math.isfinite(ok.ci_tost[1])
    assert ok.ci == (-math.inf, math.inf)  # the 95 % interval needs M ≥ 19 and is left unbounded
    with pytest.raises(ValueError, match="n_boot"):
        rank_equivalence(opt, rank2, 0.5, block=4, n_boot=15, alpha=0.025, seed=1)  # needs 19


def test_rank_equivalence_checks_its_arguments():
    rng = np.random.default_rng(25)
    opt = _opt(100, rng)
    rank2 = _rank_path(opt, rng, 0.0)
    with pytest.raises(ValueError, match="alpha"):
        rank_equivalence(opt, rank2, 0.5, block=4, n_boot=9, seed=1, alpha=0.6)
    with pytest.raises(ValueError, match="same length"):
        rank_equivalence(opt, rank2[:-1], 0.5, block=4, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="1-D"):
        rank_equivalence(opt, np.column_stack([rank2, rank2]), 0.5, block=4, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="block"):
        rank_equivalence(opt, rank2, 0.5, block=80, n_boot=9, seed=1)
    with pytest.raises(TypeError):
        rank_equivalence(opt, rank2, 0.5, block=4, n_boot=9)  # the seed is mandatory


# ---------------------------------------------------------------------------------------------
# H3b: the slope of ΔSR(r) on log ḡ_r over the ranks, a joint bootstrap of the seven paths
# ---------------------------------------------------------------------------------------------


def _dose_panel(kappa, *, T=3000, noise=0.05, seed=41, doses=DOSES):
    """Daily mean shift of the rank-r path is ``kappa·log ḡ_r`` (per-period units)."""
    rng = np.random.default_rng(seed)
    opt = _opt(T, rng)
    paths = np.column_stack([_rank_path(opt, rng, kappa * math.log(d), noise) for d in doses])
    return opt, paths


def test_slope_is_the_ols_slope_of_the_rank_differences_on_log_mean_dose():
    opt, paths = _dose_panel(0.004)
    res = rank_dose_response(opt, paths, RANKS, DOSES, block=5, n_boot=199, seed=1)
    per_rank = np.array([p.estimate for p in res.per_rank])
    expected = np.array([sharpe_ratio(paths[:, j], periods_per_year=252)
                         - sharpe_ratio(opt, periods_per_year=252) for j in range(len(RANKS))])
    np.testing.assert_allclose(per_rank, expected, rtol=1e-10)
    slope, intercept = np.polyfit(np.log(DOSES), per_rank, 1)
    assert res.slope.estimate == pytest.approx(slope, rel=1e-9)
    assert res.intercept.estimate == pytest.approx(intercept, rel=1e-7, abs=1e-9)
    assert res.ranks == RANKS and res.doses_bp == DOSES and res.regressor == "log_dose"
    assert res.x == pytest.approx(tuple(np.log(DOSES)))
    assert res.functional == "sharpe" and res.block == 5 and res.n_boot == 199 and res.seed == 1


def test_slope_interval_is_the_percentile_one_covering_the_true_slope_and_excluding_zero():
    kappa, noise = 0.004, 0.05
    opt, paths = _dose_panel(kappa, noise=noise)
    res = rank_dose_response(opt, paths, RANKS, DOSES, block=5, n_boot=499, seed=2)
    truth = kappa * ROOT252 / math.sqrt(1 + noise**2)  # per unit of log ḡ, annual Sharpe
    lo, hi = res.slope.ci_percentile
    assert lo <= truth <= hi and lo > 0.0
    assert res.significant  # the 95 % percentile interval excludes zero
    assert res.significant == (not lo <= 0.0 <= hi)


def test_significance_is_decided_by_the_percentile_interval_not_by_the_studentised_p_value():
    # a synthetic panel (found by a search over seeds) where the two rules disagree: the
    # percentile 95 % interval of the slope excludes zero, the studentised p-value is 0.08
    seed, T = 47, 250
    rng = np.random.default_rng(1000 + seed)
    opt = 0.03 + rng.standard_normal(T)
    paths = np.column_stack([opt + 0.0025 * math.log(d) + 0.4 * rng.standard_normal(T)
                             for d in DOSES])
    res = rank_dose_response(opt, paths, RANKS, DOSES, block=5, n_boot=499, seed=seed)
    assert res.slope.ci_percentile[1] < 0.0 and res.significant  # percentile interval excludes 0
    assert res.slope.pvalue > 0.05  # the studentised test of LW eq. (9) would not reject
    assert res.slope.ci[0] < 0.0 < res.slope.ci[1]


def test_no_slope_when_the_rank_paths_do_not_depend_on_the_dose():
    opt, paths = _dose_panel(0.0, noise=0.3, seed=43)
    res = rank_dose_response(opt, paths, RANKS, DOSES, block=5, n_boot=499, seed=3)
    lo, hi = res.slope.ci_percentile
    assert lo < 0.0 < hi and not res.significant


def test_the_regressor_is_fixed_and_shared_by_all_bootstrap_draws():
    """x = log ḡ_r is in-sample and common to all draws: the interval of the slope is the
    percentile interval of the OLS slope of the RESAMPLED differences on the SAME fixed x."""
    T, block, n_boot, seed = 300, 5, 80, 17
    opt, paths = _dose_panel(0.004, T=T, seed=5)
    res = rank_dose_response(opt, paths, RANKS, DOSES, block=block, n_boot=n_boot, seed=seed)
    idx = circular_block_bootstrap_indices(T, block, n_boot, np.random.default_rng(seed))
    series = np.column_stack([opt, paths])
    x = np.log(DOSES)
    slopes = np.empty(n_boot)
    for m in range(n_boot):
        sr = sharpe_ratio(series[idx[m]], periods_per_year=252)  # the same rows for all paths
        slopes[m] = np.polyfit(x, sr[1:] - sr[0], 1)[0]
    np.testing.assert_allclose(res.slope.ci_percentile, np.quantile(slopes, [0.025, 0.975]),
                               rtol=1e-8)
    # other abscissas change the weights of the fit only: the bootstrap of every path is the same
    other = rank_dose_response(opt, paths, RANKS, tuple(reversed(DOSES)), block=block,
                               n_boot=n_boot, seed=seed)
    for a, b in zip(res.per_rank, other.per_rank, strict=True):
        assert a.ci_percentile == b.ci_percentile and a.estimate == b.estimate
    assert other.slope.estimate != res.slope.estimate


def test_log_rank_sensitivity_regresses_on_log_r():
    opt, paths = _dose_panel(0.004)
    res = rank_dose_response(opt, paths, RANKS, DOSES, regressor="log_rank", block=5, n_boot=99,
                             seed=1)
    per_rank = np.array([p.estimate for p in res.per_rank])
    assert res.regressor == "log_rank" and res.x == pytest.approx(tuple(np.log(RANKS)))
    assert res.slope.estimate == pytest.approx(np.polyfit(np.log(RANKS), per_rank, 1)[0], rel=1e-9)
    with pytest.raises(ValueError, match="log_dose"):
        res.predict(1.0)  # the curve of H3c is the one on the doses


def test_all_ranks_2_to_33_sensitivity_fits_the_whole_panel():
    ranks = tuple(range(2, 34))
    doses = tuple(np.geomspace(0.4, 30.0, len(ranks)))
    kappa, noise = 0.004, 0.05
    opt, paths = _dose_panel(kappa, T=2000, noise=noise, seed=45, doses=doses)
    res = rank_dose_response(opt, paths, ranks, doses, block=5, n_boot=199, seed=3, per_rank=False)
    assert len(res.ranks) == 32 and res.per_rank == ()  # only the slope and intercept are kept
    truth = kappa * ROOT252 / math.sqrt(1 + noise**2)
    lo, hi = res.slope.ci_percentile
    assert lo <= truth <= hi and res.significant


def test_prediction_is_the_fitted_line_inside_the_range_of_mean_doses_only():
    opt, paths = _dose_panel(0.004)
    res = rank_dose_response(opt, paths, RANKS, DOSES, block=5, n_boot=99, seed=1)
    for dose in (0.4, 3.0, 30.0):  # both ends of [ḡ_2, ḡ_33] are inside
        assert res.predict(dose) == pytest.approx(
            res.intercept.estimate + res.slope.estimate * math.log(dose), rel=1e-9)
    for dose in (0.39, 30.01, 0.0):
        with pytest.raises(ValueError, match="descriptive"):
            res.predict(dose)


def test_gmv_slope_uses_the_log_variance_ratio():
    rng = np.random.default_rng(44)
    opt = 0.01 * _opt(3000, rng, mu=0.0)
    paths = np.column_stack([(1 + 0.02 * math.log(d)) * opt + 0.0005 * rng.standard_normal(opt.size)
                             for d in DOSES])
    res = rank_dose_response(opt, paths, RANKS, DOSES, functional="logvar", block=5, n_boot=499,
                             seed=1)
    # log ratio = 2·log(1 + 0.02·log ḡ) ≈ 0.04·log ḡ
    assert res.slope.estimate == pytest.approx(0.04, rel=0.15)
    assert res.significant and res.functional == "logvar"


def test_dose_response_checks_its_arguments():
    opt, paths = _dose_panel(0.004, T=200)
    ok = {"block": 5, "n_boot": 9, "seed": 1}
    with pytest.raises(ValueError, match="3 ranks"):
        rank_dose_response(opt, paths[:, :2], RANKS[:2], DOSES[:2], **ok)
    with pytest.raises(ValueError, match="paths"):
        rank_dose_response(opt, paths[:, :5], RANKS, DOSES, **ok)
    with pytest.raises(ValueError, match="paths"):
        rank_dose_response(opt, paths[:-1], RANKS, DOSES, **ok)
    with pytest.raises(ValueError, match="doses"):
        rank_dose_response(opt, paths, RANKS, DOSES[:5], **ok)
    with pytest.raises(ValueError, match="positive"):
        rank_dose_response(opt, paths, RANKS, (0.0, *DOSES[1:]), **ok)
    with pytest.raises(ValueError, match="ranks"):
        rank_dose_response(opt, paths, (1, *RANKS[1:]), DOSES, **ok)  # rank 1 is the optimum
    with pytest.raises(ValueError, match="ranks"):
        rank_dose_response(opt, paths, (2, 2, 5, 9, 17, 33), DOSES, **ok)
    with pytest.raises(ValueError, match="equal"):
        rank_dose_response(opt, paths, RANKS, (3.0,) * 6, **ok)  # no spread on the x axis
    with pytest.raises(ValueError, match="regressor"):
        rank_dose_response(opt, paths, RANKS, DOSES, regressor="dose", **ok)
    with pytest.raises(ValueError, match="alpha"):
        rank_dose_response(opt, paths, RANKS, DOSES, alpha=1.5, **ok)


# ---------------------------------------------------------------------------------------------
# block length: LW Algorithm 3.1 on the pair (optimum, rank 2), in percent
# ---------------------------------------------------------------------------------------------


def _golden_pair(T, seed, phi=0.2, rho=0.5):
    """Two AR(1) series, innovations correlated at ``rho``, returns as fractions (sd about 1 %)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T + 100, 2))
    e1 = z[:, 0]
    e2 = rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1]
    a = signal.lfilter([1.0], [1.0, -phi], e1)[100:]
    b = signal.lfilter([1.0], [1.0, -phi], e2)[100:]
    return 0.01 * (0.03 + a), 0.01 * (0.05 + b)


@pytest.mark.parametrize(("kw", "counts"), [
    ({"block_grid": (1, 3, 6), "n_pseudo": 24, "n_boot": 49, "seed": 5, "alpha": 0.25}, (3, 3, 3)),
    ({"block_grid": (2, 5), "n_pseudo": 20, "n_boot": 39, "seed": 6, "alpha": 0.3,
      "kernel": "parzen", "prewhiten": False, "standardize": True}, (6, 6)),
    ({"block_grid": (1, 4), "n_pseudo": 300, "n_boot": 19, "seed": 7, "alpha": 0.2}, (60, 51)),
    ({"block_grid": (1, 5, 20), "n_pseudo": 60, "n_boot": 29, "seed": 8, "alpha": 0.3},
     (22, 17, 14)),
])
def test_the_sharpe_calibration_keeps_its_golden_rejection_counts(kw, counts):
    """Characterisation of LW Algorithm 3.1 for the Sharpe test, pinned before its pseudo-series
    generator was shared with the log-variance calibration: the random stream, and with it every
    rejection count, must not move (the case with 300 pseudo-series spans two chunks)."""
    a, b = _golden_pair(150, 11)
    cal = calibrate_block_size(100.0 * a, 100.0 * b, **kw)
    np.testing.assert_array_equal(np.rint(cal.rejection_rates * kw["n_pseudo"]).astype(int), counts)
    assert cal.block == kw["block_grid"][int(np.argmin(np.abs(np.array(counts) / kw["n_pseudo"]
                                                            - kw["alpha"])))]


def test_logvar_calibration_follows_algorithm_3_1_on_the_difference_of_log_variances():
    """Algorithm 3.1 for the GMV statistic, written out: the same pseudo-series as for the Sharpe
    test, and for each (pseudo-series, block) the studentised block bootstrap of the difference of
    log variances tested against H0: Δ = Δ̂ of the ORIGINAL pair (p-value of LW eq. 9)."""
    a, b = _golden_pair(150, 11)
    x = np.column_stack([100.0 * a, 100.0 * b])
    grid, n_pseudo, n_boot, alpha = (2, 5, 21), 12, 29, 0.3
    cal = calibrate_logvar_block(x[:, 0], x[:, 1], block_grid=grid, n_pseudo=n_pseudo, n_boot=n_boot,
                                 alpha=alpha, seed=4)
    gen = np.random.default_rng(4)
    delta_hat = math.log(x[:, 0].var(ddof=1)) - math.log(x[:, 1].var(ddof=1))
    rejections = np.zeros(len(grid), dtype=int)
    for pseudo in _var1_pseudo_chunks(x, n_pseudo, 5.0, 50, gen):
        for ps in pseudo:
            for j, blk in enumerate(grid):
                joint = joint_contrasts(ps, np.array([[1.0, -1.0]]), functional="logvar",
                                        block=blk, n_boot=n_boot, seed=gen)
                rejections[j] += joint.pvalue(delta_hat)[0] <= alpha
    # the comparison below is not vacuous: some but not all pseudo-series reject, differently by block
    assert len(set(rejections.tolist())) > 1 and rejections.min() > 0 and rejections.max() < n_pseudo
    np.testing.assert_array_equal(np.rint(cal.rejection_rates * n_pseudo).astype(int), rejections)
    assert cal.functional == "logvar" and cal.block_grid == grid


def test_logvar_calibration_mechanics():
    a, b = _golden_pair(200, 12)
    kw = {"block_grid": (1, 5, 21), "n_pseudo": 30, "n_boot": 39, "alpha": 0.2, "seed": 4}
    cal = calibrate_logvar_block(a, b, **kw)
    assert cal.functional == "logvar" and cal.seed == 4
    assert (cal.alpha, cal.n_boot, cal.n_pseudo) == (0.2, 39, 30)
    assert (cal.mean_block_resid, cal.burn_in) == (5.0, 50)
    assert cal.rejection_rates.shape == (3,) and np.all((cal.rejection_rates >= 0) & (cal.rejection_rates <= 1))
    # the selected block minimises |rate − α| (the first one on ties)
    assert cal.block == (1, 5, 21)[int(np.argmin(np.abs(cal.rejection_rates - 0.2)))]
    again = calibrate_logvar_block(a, b, **kw)
    np.testing.assert_array_equal(cal.rejection_rates, again.rejection_rates)
    # the statistic is unit-free: returns in fractions or in percent give the same calibration
    scaled = calibrate_logvar_block(100.0 * a, 100.0 * b, **kw)
    np.testing.assert_array_equal(cal.rejection_rates, scaled.rejection_rates)
    # the Sharpe calibration of the same pair is another test
    sharpe = calibrate_block_size(a, b, **kw)
    assert sharpe.functional == "sharpe" and not np.array_equal(cal.rejection_rates, sharpe.rejection_rates)


def test_logvar_calibration_checks_its_arguments():
    a, b = _golden_pair(100, 13)
    with pytest.raises(ValueError, match="block"):
        calibrate_logvar_block(a, b, block_grid=(1, 80), n_pseudo=2, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="block_grid"):
        calibrate_logvar_block(a, b, block_grid=(), n_pseudo=2, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="alpha"):
        calibrate_logvar_block(a, b, alpha=1.5, n_pseudo=2, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="n_pseudo"):
        calibrate_logvar_block(a, b, n_pseudo=0, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="equal length"):
        calibrate_logvar_block(a, b[:-1], n_pseudo=2, n_boot=9, seed=1)
    with pytest.raises(TypeError):
        calibrate_logvar_block(a, b, n_pseudo=2, n_boot=9)  # the seed is mandatory


def test_the_functional_picks_the_statistic_of_every_calibration_entry_point():
    rng = np.random.default_rng(32)
    opt = 0.01 * _opt(300, rng, mu=0.0, phi=0.1)
    level_a = 1.03 * opt[:, None] + 0.001 * rng.standard_normal((300, 3))
    level_b = 1.06 * opt[:, None] + 0.001 * rng.standard_normal((300, 2))
    kw = {"block_grid": (1, 5, 21), "n_pseudo": 12, "n_boot": 29, "alpha": 0.2}
    # the rank path: the pair (optimum, rank 2), in percent, whatever the functional
    for functional, reference in (("sharpe", calibrate_block_size), ("logvar", calibrate_logvar_block)):
        got = calibrate_rank_block(opt, level_a[:, 0], functional=functional, seed=5, **kw)
        want = reference(100.0 * opt, 100.0 * level_a[:, 0], seed=5, **kw)
        np.testing.assert_array_equal(got.rejection_rates, want.rejection_rates)
        assert got.functional == functional
        # the replicas (б): the pair (optimum, equal-weight mean of all replicas)
        rep = calibrate_replica_block(opt, [level_a, level_b], functional=functional, seed=5, **kw)
        mean = np.column_stack([level_a, level_b]).mean(axis=1)
        np.testing.assert_array_equal(
            rep.rejection_rates, reference(100.0 * opt, 100.0 * mean, seed=5, **kw).rejection_rates)
        joint = calibrate_joint_block(opt, [level_a, level_b], functional=functional, seed=5, **kw)
        np.testing.assert_array_equal(joint.rejection_rates, rep.rejection_rates)
    # the default is the Sharpe test, as before the functional existed
    assert calibrate_rank_block(opt, level_a[:, 0], seed=5, **kw).functional == "sharpe"
    assert calibrate_replica_block(opt, [level_a], seed=5, **kw).functional == "sharpe"
    for fn in (calibrate_rank_block, calibrate_replica_block):
        assert inspect.signature(fn).parameters["functional"].default == "sharpe"
        with pytest.raises(ValueError, match="functional"):
            fn(opt, level_a[:, 0] if fn is calibrate_rank_block else [level_a], functional="sortino",
               seed=5, **kw)


def test_the_replica_api_calibrates_on_the_functional_of_its_arm():
    """``block="calibrate"`` of the descriptive replica tests uses the statistic of the arm."""
    rng = np.random.default_rng(33)
    opt = 0.01 * _opt(300, rng, mu=0.0, phi=0.1)
    reps = [1.03 * opt[:, None] + 0.001 * rng.standard_normal((300, 2)) for _ in range(3)]
    kw = {"block_grid": (1, 5), "n_pseudo": 8, "n_boot": 29}
    for functional, margin in (("sharpe", 0.5), ("logvar", (-0.3, 0.3))):
        eq = equivalence_test(opt, reps[:1], [10.0], margin, functional=functional,
                              block="calibrate", n_boot=29, seed=3, calibration_kwargs=kw)
        dr = dose_response(opt, reps, [5.0, 10.0, 25.0], functional=functional, block="calibrate",
                           n_boot=29, seed=3, calibration_kwargs=kw)
        assert eq.calibration.functional == functional and dr.calibration.functional == functional


def test_calibration_is_algorithm_3_1_on_the_pair_optimum_rank_2_in_percent():
    rng = np.random.default_rng(1)
    opt = 0.01 * _opt(500, rng, phi=0.1)  # returns as fractions: sd 1 %
    rank2 = opt + 0.0005 + 0.003 * rng.standard_normal(500)
    kw = {"block_grid": (1, 5, 21), "n_pseudo": 10, "n_boot": 39}
    a = calibrate_rank_block(opt, rank2, seed=4, **kw)
    b = calibrate_block_size(100.0 * opt, 100.0 * rank2, seed=4, **kw)
    np.testing.assert_array_equal(a.rejection_rates, b.rejection_rates)
    assert a.block == b.block and a.block_grid == (1, 5, 21) and a.n_pseudo == 10
    # percent=False is the same call on series that are already in percent
    c = calibrate_rank_block(100.0 * opt, 100.0 * rank2, seed=4, percent=False, **kw)
    np.testing.assert_array_equal(a.rejection_rates, c.rejection_rates)


def test_replica_calibration_is_the_same_algorithm_on_the_optimum_and_the_mean_of_replicas():
    """Descriptive operator (б): the pair is (optimum, equal-weight mean of all replicas), with the
    plan's daily settings as defaults; the replica machinery of ``cif.stats.rq3`` keeps working."""
    rng = np.random.default_rng(3)
    opt = 0.01 * _opt(400, rng, phi=0.1)
    level_a = opt[:, None] + 0.0004 + 0.003 * rng.standard_normal((400, 3))
    level_b = opt[:, None] + 0.0009 + 0.003 * rng.standard_normal((400, 2))
    kw = {"block_grid": (1, 5, 21), "n_pseudo": 10, "n_boot": 39}
    a = calibrate_replica_block(opt, [level_a, level_b], seed=4, **kw)
    partner = np.column_stack([level_a, level_b]).mean(axis=1)
    b = calibrate_block_size(100.0 * opt, 100.0 * partner, seed=4, **kw)
    np.testing.assert_array_equal(a.rejection_rates, b.rejection_rates)
    sig = inspect.signature(calibrate_replica_block).parameters
    assert sig["block_grid"].default == BLOCK_GRID_DAILY and sig["percent"].default is True
    assert sig["n_pseudo"].default == 1000 and sig["n_boot"].default == 499
    # one replica block of one column is exactly the rank-path pair
    c = calibrate_replica_block(opt, [level_a[:, :1]], seed=4, **kw)
    d = calibrate_rank_block(opt, level_a[:, 0], seed=4, **kw)
    np.testing.assert_array_equal(c.rejection_rates, d.rejection_rates)


def test_calibration_is_reproducible_and_its_pair_is_the_optimum_and_rank_2():
    rng = np.random.default_rng(2)
    opt = 0.01 * _opt(400, rng, phi=0.1)
    rank2 = opt + 0.0004 + 0.003 * rng.standard_normal(400)
    rank3 = opt + 0.0009 + 0.006 * rng.standard_normal(400)
    kw = {"block_grid": (1, 5, 21), "n_pseudo": 12, "n_boot": 39}
    a = calibrate_rank_block(opt, rank2, seed=9, **kw)
    b = calibrate_rank_block(opt, rank2, seed=9, **kw)
    np.testing.assert_array_equal(a.rejection_rates, b.rejection_rates)
    assert a.seed == 9
    other = calibrate_rank_block(opt, rank3, seed=9, **kw)  # another partner, another pseudo-DGP
    assert not np.array_equal(a.rejection_rates, other.rejection_rates)
    with pytest.raises(ValueError, match="1-D"):
        calibrate_rank_block(opt, np.column_stack([rank2, rank3]), seed=9, **kw)
    with pytest.raises(TypeError):
        calibrate_rank_block(opt, rank2, **kw)  # the seed is mandatory


def test_one_calibrated_block_serves_every_test_of_the_market_and_arm():
    opt, paths = _dose_panel(0.004, T=400, seed=7)
    cal = calibrate_rank_block(0.01 * opt, 0.01 * paths[:, 0], seed=7, block_grid=(1, 5, 10),
                               n_pseudo=8, n_boot=29)
    h3a = rank_equivalence(opt, paths[:, 0], 0.5, block=cal.block, n_boot=99, seed=1)
    h3b = rank_dose_response(opt, paths, RANKS, DOSES, block=cal.block, n_boot=99, seed=1)
    assert h3a.block == h3b.block == cal.block and cal.block in (1, 5, 10)


def _ar_pair(T, rng, phi, rho, mu1, mu2):
    """Two AR(1) series with unit marginal variance and innovations correlated at ``rho``."""
    z = rng.standard_normal((T + 200, 2))
    e1 = z[:, 0]
    e2 = rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1]
    a = signal.lfilter([math.sqrt(1 - phi**2)], [1.0, -phi], e1)[200:]
    b = signal.lfilter([math.sqrt(1 - phi**2)], [1.0, -phi], e2)[200:]
    return mu1 + a, mu2 + b


@pytest.mark.slow
def test_calibrated_block_gives_near_nominal_coverage_on_a_simple_dgp():
    """Algorithm 3.1 followed by the joint bootstrap, end to end: on a persistent (AR(1), φ = 0.3)
    and strongly correlated (ρ = 0.9) pair with a known Sharpe difference the studentised 95 %
    interval at the calibrated block covers the truth. 60 replications with a small calibration
    (20 pseudo-series, 49 repetitions); the corridor is about ±3 binomial s.e. around 95 %."""
    rng = np.random.default_rng(2026)
    truth = (0.05 - 0.03) * ROOT252  # annual Sharpe difference, rank 2 minus optimum
    n_rep, covered = 60, 0
    for k in range(n_rep):
        a, b = _ar_pair(400, rng, 0.3, 0.9, 0.03, 0.05)
        cal = calibrate_rank_block(0.01 * a, 0.01 * b, seed=k, block_grid=(1, 5, 10),
                                   n_pseudo=20, n_boot=49)
        res = rank_equivalence(a, b, 0.5, block=cal.block, n_boot=299, seed=k)
        covered += res.ci[0] <= truth <= res.ci[1]
    print(f"[stats-report] RQ3 rank-2 interval coverage at the calibrated block, T=400, "
          f"R={n_rep}: {covered / n_rep:.4f}")
    assert 0.88 <= covered / n_rep <= 1.0, covered / n_rep


def _garch_pair(T, rng, mu1, mu2, rho, a=0.08, b=0.90, burn=500):
    """GARCH(1,1) 'optimum' with unit unconditional variance and a partner on the same volatility
    path (innovations correlated at ``rho``), so that both have unit variance and the means
    ``mu1`` and ``mu2``: the Sharpe difference is ``mu2 − mu1`` and the variance ratio is one."""
    e = rng.standard_normal(T + burn)
    eta = rng.standard_normal(T + burn)
    s2, s, x = 1.0, np.empty(T + burn), np.empty(T + burn)
    for t in range(T + burn):
        s[t] = math.sqrt(s2)
        x[t] = s[t] * e[t]
        s2 = (1.0 - a - b) + a * x[t] ** 2 + b * s2
    partner = s * (rho * e + math.sqrt(1.0 - rho**2) * eta)
    return mu1 + x[burn:], mu2 + partner[burn:]


@pytest.mark.slow
def test_logvar_calibrated_block_gives_near_nominal_coverage_for_the_gmv_statistic():
    """The GMV arm end to end: Algorithm 3.1 on the difference of log variances picks the block and
    the studentised interval of the log variance ratio of a GARCH(1,1) pair (persistence 0.98, equal
    variances: the truth is 0) uses it. 40 replications with a small calibration (12 pseudo-series,
    39 repetitions); the corridor is about ±3 binomial s.e. around 95 %."""
    rng = np.random.default_rng(2028)
    n_rep, covered, blocks = 40, 0, []
    for k in range(n_rep):
        a, b = _garch_pair(1500, rng, 0.0, 0.0, 0.9)
        cal = calibrate_rank_block(0.01 * a, 0.01 * b, functional="logvar", block_grid=(1, 5, 21),
                                   n_pseudo=12, n_boot=39, seed=k)
        res = rank_equivalence(a, b, (-0.5, 0.5), functional="logvar", block=cal.block, n_boot=199,
                               seed=k)
        covered += res.ci[0] <= 0.0 <= res.ci[1]
        blocks.append(cal.block)
    print(f"[stats-report] RQ3 GMV interval coverage at the log-variance calibrated block, T=1500, "
          f"R={n_rep}: {covered / n_rep:.4f}, blocks {sorted(set(blocks))}")
    assert 0.85 <= covered / n_rep <= 1.0, covered / n_rep


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["sharpe", "logvar"])
def test_rank_2_intervals_cover_the_truth_under_volatility_clustering(functional):
    """GARCH(1,1) with persistence 0.98 (volatility clustering), T = 2000, a pair correlated at 0.9:
    the studentised 95 % interval of the Sharpe difference (known: 0.02 per day) and of the log
    variance ratio (known: 0) at b = 21. 60 replications, corridor about ±3 binomial s.e."""
    rng = np.random.default_rng(777)
    truth = 0.02 * ROOT252 if functional == "sharpe" else 0.0
    margin = 0.5 if functional == "sharpe" else (-0.2, 0.2)
    n_rep, covered = 60, 0
    for k in range(n_rep):
        a, b = _garch_pair(2000, rng, 0.03, 0.05, 0.9)
        res = rank_equivalence(a, b, margin, functional=functional, block=21, n_boot=299, seed=k)
        covered += res.ci[0] <= truth <= res.ci[1]
    print(f"[stats-report] RQ3 rank-2 interval coverage under GARCH, {functional}, T=2000, "
          f"R={n_rep}: {covered / n_rep:.4f}")
    assert 0.86 <= covered / n_rep <= 1.0, covered / n_rep


# ---------------------------------------------------------------------------------------------
# H3c: the dose of a real solver, the range rule, the residuals against the H3b curve
# ---------------------------------------------------------------------------------------------

N_FOLDS = 40


def test_h3c_plan_constants():
    assert MIN_MEDIAN_GAP_BP == 1.0
    assert RANK_BIN_LABELS == ("1", "2-5", "6-33", ">33")
    assert inspect.signature(solver_residuals).parameters["n_boot"].default == 4999


def test_the_dose_of_a_solver_is_its_mean_gap_over_f_k():
    gaps = np.array([2.0, 40.0, 6.0, 8.0, 100.0])
    in_fk = np.array([True, False, True, True, False])  # the gaps outside F_k are not looked at
    d = rank_solver_dose(gaps, in_fk, DOSES)
    assert d.dose_bp == pytest.approx(16.0 / 3.0) and d.median_gap_bp == 6.0 and d.n_folds == 3
    assert d.inside_range and d.material and not d.descriptive and d.reason is None


@pytest.mark.parametrize(("gaps", "word"), [
    ([0.1, 0.2], "below"),  # mean 0.15 < ḡ_2 = 0.4
    ([40.0, 50.0], "above"),  # mean 45 > ḡ_33 = 30
    ([0.0, 0.0], "below"),  # the optimum on every fold: no log of zero, no prediction
])
def test_a_dose_outside_the_range_of_the_curve_is_descriptive(gaps, word):
    d = rank_solver_dose(np.array(gaps), np.array([True, True]), DOSES)
    assert d.descriptive and not d.inside_range and word in d.reason


def test_the_range_is_closed_at_both_ends_up_to_rounding():
    ones = np.ones(5, dtype=bool)
    for dose in (min(DOSES), max(DOSES)):  # a solver that always lands on rank 2 / rank 33
        assert rank_solver_dose(np.full(5, dose), ones, DOSES).inside_range
    # the mean of the same numbers formed another way differs in the last digits: still inside
    assert rank_solver_dose(np.full(5, min(DOSES) * (1.0 - 1e-12)), ones, DOSES).inside_range
    assert not rank_solver_dose(np.full(5, min(DOSES) * (1.0 - 1e-6)), ones, DOSES).inside_range
    assert not rank_solver_dose(np.full(5, max(DOSES) * (1.0 + 1e-6)), ones, DOSES).inside_range


def test_a_solver_is_material_from_a_median_gap_of_one_basis_point():
    ones = np.ones(5, dtype=bool)
    assert rank_solver_dose(np.full(5, 1.0), ones, DOSES).material  # the median is exactly 1 bp
    assert not rank_solver_dose(np.full(5, 0.99), ones, DOSES).material
    # rare big misses: the mean dose is inside the range, the median gap is zero
    rare = np.array([0.0] * 6 + [10.0] * 4)
    d = rank_solver_dose(rare, np.ones(10, dtype=bool), DOSES)
    assert d.dose_bp == pytest.approx(4.0) and d.median_gap_bp == 0.0 and not d.material


def test_rank_solver_dose_checks_its_arguments():
    ones = np.ones(3, dtype=bool)
    with pytest.raises(ValueError, match="finite"):
        rank_solver_dose(np.array([np.nan, 1.0, 2.0]), ones, DOSES)
    with pytest.raises(ValueError, match="folds"):
        rank_solver_dose(np.array([1.0, 2.0]), ones, DOSES)
    none = rank_solver_dose(np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=bool), DOSES)
    assert none.descriptive and none.dose_bp is None and none.n_folds == 0 and not none.material


def _h3c_world(*, T=3000, kappa=0.004, noise=0.05, seed=61, solver_doses=None, extra=None):
    """Rank paths with the known curve ``kappa·log ḡ_r`` and solver paths that follow the same
    curve at their own dose (plus an optional extra daily shift)."""
    doses = solver_doses or {"tabu": 1.5, "SA": 4.0, "LS": 9.0}
    rng = np.random.default_rng(seed)
    opt = _opt(T, rng)
    paths = np.column_stack([_rank_path(opt, rng, kappa * math.log(d), noise) for d in DOSES])
    sol = {n: _rank_path(opt, rng, kappa * math.log(max(d, 0.4)) + (extra or {}).get(n, 0.0), noise)
           for n, d in doses.items()}
    gaps = {n: np.full(N_FOLDS, float(d)) for n, d in doses.items()}
    return opt, paths, sol, gaps


def _residuals(world, *, margin=0.25, n_boot=199, seed=1, fold_mask=None, **kw):
    opt, paths, sol, gaps = world
    mask = np.ones(N_FOLDS, dtype=bool) if fold_mask is None else fold_mask
    return solver_residuals(opt, paths, RANKS, DOSES, sol, gaps, mask, margin, block=5,
                            n_boot=n_boot, seed=seed, **kw)


def test_the_residual_is_the_observed_difference_minus_the_prediction_of_the_h3b_curve():
    world = _h3c_world()
    opt, paths, sol, _ = world
    res = _residuals(world)
    fit = rank_dose_response(opt, paths, RANKS, DOSES, block=5, n_boot=199, seed=1)
    assert not res.descriptive and res.reason is None and res.n_material == 3
    assert [s.name for s in res.solvers] == ["tabu", "SA", "LS"]
    for s in res.solvers:
        observed = (sharpe_ratio(sol[s.name], periods_per_year=252)
                    - sharpe_ratio(opt, periods_per_year=252))
        assert s.observed.estimate == pytest.approx(observed, rel=1e-10)
        assert s.predicted == pytest.approx(fit.predict(s.dose.dose_bp), rel=1e-8, abs=1e-12)
        assert s.residual.estimate == pytest.approx(observed - s.predicted, abs=1e-10)


def test_the_optimum_drops_out_of_the_residual():
    # prediction = Σ_r c_r ΔSR(r) with OLS weights that sum to one, so
    # residual = SR_solver − Σ_r c_r SR_r
    world = _h3c_world(seed=62)
    _, paths, sol, _ = world
    res = _residuals(world)
    x = np.log(DOSES)
    sr = lambda a: sharpe_ratio(a, periods_per_year=252)  # noqa: E731
    for s in res.solvers:
        c = 1.0 / x.size + (math.log(s.dose.dose_bp) - x.mean()) * (x - x.mean()) / (
            (x - x.mean()) @ (x - x.mean()))
        assert c.sum() == pytest.approx(1.0)
        expected = sr(sol[s.name]) - sum(c[j] * sr(paths[:, j]) for j in range(len(RANKS)))
        assert s.residual.estimate == pytest.approx(expected, abs=1e-10)


def test_solvers_on_the_curve_pass_the_tost_and_the_iut_passes():
    res = _residuals(_h3c_world())
    assert res.margin == (-0.25, 0.25) and res.alpha == 0.05 and res.functional == "sharpe"
    for s in res.solvers:
        assert s.equivalent and abs(s.residual.estimate) < 0.25
        assert s.ci_tost[0] > -0.25 and s.ci_tost[1] < 0.25
        assert s.ci_tost[0] > s.residual.ci[0] and s.ci_tost[1] < s.residual.ci[1]  # 90 % in 95 %
    assert res.iut_passes is True


def test_one_solver_off_the_curve_fails_its_tost_and_so_does_the_iut():
    # +0.03 per day = +0.48 annual Sharpe above what its dose predicts: outside the margin
    res = _residuals(_h3c_world(extra={"SA": 0.03}))
    by_name = {s.name: s for s in res.solvers}
    assert by_name["tabu"].equivalent and by_name["LS"].equivalent
    assert by_name["SA"].equivalent is False
    assert by_name["SA"].residual.estimate == pytest.approx(0.03 * ROOT252, abs=0.05)
    assert res.iut_passes is False and not res.descriptive


def test_the_observed_difference_is_the_pairwise_bootstrap_on_the_same_resamples():
    world = _h3c_world()
    opt, _, sol, _ = world
    res = _residuals(world)
    pair = rank_equivalence(opt, sol["tabu"], 0.25, block=5, n_boot=199, seed=1)
    obs = res.solvers[0].observed
    assert obs.estimate == pytest.approx(pair.estimate, rel=1e-12)
    assert obs.se == pytest.approx(pair.se, rel=1e-12)
    assert obs.ci == pytest.approx(pair.ci, rel=1e-9)


def test_the_gaps_outside_f_k_do_not_enter_the_dose():
    opt, paths, sol, gaps = _h3c_world()
    mask = np.ones(N_FOLDS, dtype=bool)
    mask[:10] = False
    gaps = {n: g.copy() for n, g in gaps.items()}
    for g in gaps.values():
        g[:10] = 1e4  # uncertified folds: absurd gaps that must not matter
    res = solver_residuals(opt, paths, RANKS, DOSES, sol, gaps, mask, 0.25, block=5, n_boot=99,
                           seed=1)
    assert [s.dose.dose_bp for s in res.solvers] == pytest.approx([1.5, 4.0, 9.0])
    assert all(s.dose.n_folds == N_FOLDS - 10 for s in res.solvers)


def test_h3c_is_descriptive_with_fewer_than_two_solvers_of_a_median_gap_of_one_bp():
    opt, paths, sol, _ = _h3c_world()
    gaps = {"tabu": np.zeros(N_FOLDS), "SA": np.full(N_FOLDS, 4.0), "LS": np.full(N_FOLDS, 0.5)}
    res = solver_residuals(opt, paths, RANKS, DOSES, sol, gaps, np.ones(N_FOLDS, dtype=bool),
                           0.25, block=5, n_boot=99, seed=1)
    assert res.descriptive and res.n_material == 1 and res.iut_passes is None
    assert "fewer than two" in res.reason and "1 bp" in res.reason
    assert all(s.residual is None and s.observed is None and s.equivalent is None
               for s in res.solvers)
    assert res.solvers[1].dose.dose_bp == pytest.approx(4.0)  # the doses are still reported


def test_a_descriptive_h3c_still_checks_the_bootstrap_settings():
    # the settings are validated up front: a bad block must not hide behind a descriptive outcome
    opt, paths, sol, _ = _h3c_world(T=300)
    zero = {n: np.zeros(N_FOLDS) for n in sol}  # no solver errs: H3c is descriptive, no bootstrap
    ones = np.ones(N_FOLDS, dtype=bool)
    args = (opt, paths, RANKS, DOSES, sol, zero, ones, 0.25)
    assert solver_residuals(*args, block=5, n_boot=99, seed=1).descriptive
    with pytest.raises(ValueError, match="block"):
        solver_residuals(*args, block=1000, n_boot=99, seed=1)
    with pytest.raises(ValueError, match="n_boot"):
        solver_residuals(*args, block=5, n_boot=0, seed=1)
    with pytest.raises(TypeError, match="seed"):
        solver_residuals(*args, block=5, n_boot=99, seed=None)
    # and the TOST interval needs enough resamples here too
    world = _h3c_world(T=300)
    with pytest.raises(ValueError, match="n_boot"):
        _residuals(world, n_boot=5)


def test_a_solver_outside_the_range_is_descriptive_and_the_others_are_still_tested():
    res = _residuals(_h3c_world(solver_doses={"tabu": 1.5, "SA": 4.0, "LS": 120.0}))
    tabu, sa, ls = res.solvers
    assert not res.descriptive and tabu.equivalent and sa.equivalent  # the IUT is over the two
    assert ls.residual is None and ls.equivalent is None and ls.dose.descriptive
    assert "above" in ls.dose.reason
    assert res.iut_passes is True


def test_fewer_than_two_solvers_with_a_prediction_is_descriptive():
    res = _residuals(_h3c_world(solver_doses={"tabu": 1.5, "SA": 100.0, "LS": 120.0}))
    assert res.descriptive and res.iut_passes is None and res.n_material == 3
    assert "prediction" in res.reason
    assert all(s.residual is None for s in res.solvers)


def test_solver_residuals_check_their_arguments():
    opt, paths, sol, gaps = _h3c_world(T=300)
    ones = np.ones(N_FOLDS, dtype=bool)
    ok = {"block": 5, "n_boot": 9, "seed": 1}
    with pytest.raises(ValueError, match="same solvers"):
        solver_residuals(opt, paths, RANKS, DOSES, sol, {"tabu": gaps["tabu"]}, ones, 0.25, **ok)
    with pytest.raises(ValueError, match="solver path"):
        solver_residuals(opt, paths, RANKS, DOSES, {**sol, "SA": sol["SA"][:-1]}, gaps, ones, 0.25,
                         **ok)
    with pytest.raises(ValueError, match="folds"):
        solver_residuals(opt, paths, RANKS, DOSES, sol, gaps, ones[:-1], 0.25, **ok)
    with pytest.raises(ValueError, match="at least one solver"):
        solver_residuals(opt, paths, RANKS, DOSES, {}, {}, ones, 0.25, **ok)
    with pytest.raises(ValueError, match="margin"):
        solver_residuals(opt, paths, RANKS, DOSES, sol, gaps, ones, 0.0, **ok)
    with pytest.raises(ValueError, match="3 ranks"):
        solver_residuals(opt, paths[:, :2], RANKS[:2], DOSES[:2], sol, gaps, ones, 0.25, **ok)
    with pytest.raises(TypeError):
        solver_residuals(opt, paths, RANKS, DOSES, sol, gaps, ones, 0.25, block=5, n_boot=9)


# ---------------------------------------------------------------------------------------------
# the descriptive rank distribution of a solver: bins 1; 2-5; 6-33; > 33
# ---------------------------------------------------------------------------------------------


def test_rank_bins_follow_the_plan():
    ranks = [1, 1, 2, 5, 6, 33, 34, 100]
    b = rank_bins(ranks)
    assert b.labels == ("1", "2-5", "6-33", ">33") and b.counts == (2, 2, 2, 2)
    assert b.shares == pytest.approx((0.25, 0.25, 0.25, 0.25)) and b.n_folds == 8
    # the bin edges: 5 | 6 and 33 | 34
    assert rank_bins([5]).counts == (0, 1, 0, 0) and rank_bins([6]).counts == (0, 0, 1, 0)
    assert rank_bins([33]).counts == (0, 0, 1, 0) and rank_bins([34]).counts == (0, 0, 0, 1)


def test_rank_bins_count_the_folds_of_f_k_only():
    ranks = np.array([1, 7, 1, 40, 3])
    mask = np.array([True, True, False, False, True])
    b = rank_bins(ranks, mask)
    assert b.counts == (1, 1, 1, 0) and b.n_folds == 3 and sum(b.shares) == pytest.approx(1.0)
    assert rank_bins(np.array([1.0, 2.0, 40.0])).counts == (1, 1, 0, 1)  # integral floats are fine


def test_rank_bins_check_their_arguments():
    with pytest.raises(ValueError, match="rank"):
        rank_bins([1, 0, 2])  # ranks start at 1 (the optimum)
    with pytest.raises(ValueError, match="integer"):
        rank_bins([1, 2.5])
    with pytest.raises(ValueError, match="folds"):
        rank_bins([1, 2, 3], np.array([True, False]))
    with pytest.raises(ValueError, match="no folds"):
        rank_bins([1, 2], np.array([False, False]))
    with pytest.raises(ValueError, match="1-D"):
        rank_bins([[1, 2], [3, 4]])
