"""RQ3 inference: replica-averaged contrasts, joint block bootstrap, H3a / H3b / H3c.

ANALYSIS_PLAN §2 RQ3 and §7.4. Synthetic returns only: an "optimum" series and replicas that
are the optimum plus a known mean shift and small noise, so population values are known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from cif.stats import (
    circular_block_bootstrap_indices,
    ledoit_wolf_sharpe_test,
    sharpe_ratio,
)
from cif.stats.rq3 import (
    H3A_OUTCOMES,
    calibrate_joint_block,
    dose_response,
    equivalence_test,
    h3a_outcome,
    joint_contrasts,
    sharpe_margin,
    solver_dose,
    variance_margin,
)

ROOT252 = math.sqrt(252)


def _opt(T: int, rng: np.random.Generator, mu: float = 0.03, phi: float = 0.0) -> np.ndarray:
    z = rng.standard_normal(T + 200)
    x = signal.lfilter([1.0], [1.0, -phi], z)[200:] if phi else z[200:]
    return mu + x


def _replicas(opt: np.ndarray, rng: np.random.Generator, shift: float, n_rep: int,
              noise: float = 0.05) -> np.ndarray:
    """Replicas = optimum + constant daily shift + small independent noise, shape (T, n_rep)."""
    return opt[:, None] + shift + noise * rng.standard_normal((opt.size, n_rep))


def _difference_row(n_rep: int) -> np.ndarray:
    return np.concatenate([[-1.0], np.full(n_rep, 1.0 / n_rep)])[None, :]


# ---------------------------------------------------------------------------------------------
# point estimates
# ---------------------------------------------------------------------------------------------


def test_estimate_is_the_mean_over_replicas_of_the_sharpe_differences_annualised():
    rng = np.random.default_rng(0)
    opt = _opt(400, rng)
    reps = _replicas(opt, rng, 0.01, 3)
    series = np.column_stack([opt, reps])
    res = joint_contrasts(series, _difference_row(3), block=5, n_boot=99, seed=1)
    sr = lambda x: sharpe_ratio(x, periods_per_year=None)  # noqa: E731
    expected = np.mean([sr(reps[:, r]) - sr(opt) for r in range(3)]) * ROOT252
    assert res.estimate[0] == pytest.approx(expected, rel=1e-12)
    assert res.estimate.shape == (1,) and res.n_obs == 400 and res.block == 5
    unscaled = joint_contrasts(series, _difference_row(3), block=5, n_boot=99, seed=1,
                               periods_per_year=None)
    assert unscaled.estimate[0] == pytest.approx(expected / ROOT252, rel=1e-12)


def test_logvar_estimate_is_the_mean_log_variance_ratio():
    rng = np.random.default_rng(1)
    opt = _opt(500, rng)
    reps = np.column_stack([1.03 * opt + 0.05 * rng.standard_normal(500) for _ in range(4)])
    series = np.column_stack([opt, reps])
    res = joint_contrasts(series, _difference_row(4), functional="logvar", block=5, n_boot=99, seed=1)
    expected = np.mean([np.log(reps[:, r].var(ddof=1) / opt.var(ddof=1)) for r in range(4)])
    assert res.estimate[0] == pytest.approx(expected, rel=1e-12)
    assert res.scale == 1.0  # no annualisation of a log ratio


def test_contrast_rows_are_linear_in_the_series_functional():
    rng = np.random.default_rng(2)
    series = rng.standard_normal((300, 4)) + 0.05
    rows = np.array([[1.0, -1.0, 0.0, 0.0], [0.5, 0.5, -0.5, -0.5]])
    res = joint_contrasts(series, rows, block=4, n_boot=99, seed=3)
    sr = np.array([sharpe_ratio(series[:, k], periods_per_year=None) for k in range(4)]) * ROOT252
    np.testing.assert_allclose(res.estimate, rows @ sr, rtol=1e-12)


# ---------------------------------------------------------------------------------------------
# the joint block bootstrap
# ---------------------------------------------------------------------------------------------


def _literal_boot_stats(sample: np.ndarray, w: np.ndarray, block: int, ddof: int = 1):
    """LW (2008) Sec. 3.2.2 for one resample, written out: theta*, s(theta*)."""
    T, M = sample.shape
    theta = 0.0
    proj = np.zeros(T)
    for k in range(M):
        x = sample[:, k]
        mu = x.mean()
        c = x - mu
        v = (c * c).mean()
        sd = math.sqrt(v)
        theta += w[k] * mu / math.sqrt(v * T / (T - ddof))
        proj += w[k] * (c / sd - mu * (c * c - v) / (2 * sd**3))  # influence function of the SR
    n_blocks = T // block
    zeta = proj[: n_blocks * block].reshape(n_blocks, block).sum(axis=1) / math.sqrt(block)
    return theta, math.sqrt(np.mean(zeta**2) / T)


def test_bootstrap_statistics_match_the_literal_definition_with_common_blocks():
    rng = np.random.default_rng(4)
    T, block, n_boot, seed = 203, 5, 12, 9  # T is not a multiple of the block
    opt = _opt(T, rng)
    series = np.column_stack([opt, _replicas(opt, rng, 0.02, 3)])
    w = _difference_row(3)
    res = joint_contrasts(series, w, block=block, n_boot=n_boot, seed=seed, periods_per_year=None)
    idx = circular_block_bootstrap_indices(T, block, n_boot, np.random.default_rng(seed))
    theta_hat = res.estimate[0]
    for m in range(n_boot):
        theta, se = _literal_boot_stats(series[idx[m]], w[0], block)  # the SAME rows for all series
        assert res.theta_star[m, 0] == pytest.approx(theta, rel=1e-10)
        assert res.dstar[m, 0] == pytest.approx(abs(theta - theta_hat) / se, rel=1e-9)


def test_logvar_bootstrap_matches_the_literal_definition():
    rng = np.random.default_rng(5)
    T, block, n_boot, seed = 150, 6, 8, 3
    opt = _opt(T, rng)
    series = np.column_stack([opt, 1.05 * opt + 0.1 * rng.standard_normal(T)])
    w = np.array([[-1.0, 1.0]])
    res = joint_contrasts(series, w, functional="logvar", block=block, n_boot=n_boot, seed=seed)
    idx = circular_block_bootstrap_indices(T, block, n_boot, np.random.default_rng(seed))
    for m in range(n_boot):
        xs = series[idx[m]]
        lv = [np.log(xs[:, k].var(ddof=1)) for k in range(2)]
        # influence function of log(variance): (c² − v) / v
        proj = np.zeros(T)
        for k, sign in ((0, -1.0), (1, 1.0)):
            c = xs[:, k] - xs[:, k].mean()
            v = (c * c).mean()
            proj += sign * (c * c - v) / v
        nb = T // block
        zeta = proj[: nb * block].reshape(nb, block).sum(axis=1) / math.sqrt(block)
        se = math.sqrt(np.mean(zeta**2) / T)
        assert res.theta_star[m, 0] == pytest.approx(lv[1] - lv[0], rel=1e-10)
        assert res.dstar[m, 0] == pytest.approx(abs(lv[1] - lv[0] - res.estimate[0]) / se, rel=1e-9)


def test_contrasts_are_evaluated_on_the_same_resamples():
    rng = np.random.default_rng(6)
    series = rng.standard_normal((260, 4)) * 0.01 + 0.0003
    rows = np.array([[1.0, -1.0, 0.0, 0.0], [0.0, 0.5, 0.5, -1.0]])
    both = joint_contrasts(series, rows, block=4, n_boot=300, seed=11)
    for q in range(2):
        alone = joint_contrasts(series, rows[q : q + 1], block=4, n_boot=300, seed=11)
        np.testing.assert_allclose(both.theta_star[:, q], alone.theta_star[:, 0], rtol=1e-12)
        np.testing.assert_allclose(both.dstar[:, q], alone.dstar[:, 0], rtol=1e-12)
        assert both.se[q] == pytest.approx(alone.se[0], rel=1e-12)


def test_unused_series_do_not_change_the_result():
    rng = np.random.default_rng(7)
    series = rng.standard_normal((200, 3))
    row = np.array([[1.0, -1.0]])
    a = joint_contrasts(series[:, :2], row, block=4, n_boot=99, seed=2)
    b = joint_contrasts(series, np.array([[1.0, -1.0, 0.0]]), block=4, n_boot=99, seed=2)
    np.testing.assert_allclose(a.dstar, b.dstar, rtol=1e-12)


def test_one_replica_reproduces_the_ledoit_wolf_2008_bootstrap():
    rng = np.random.default_rng(8)
    opt = _opt(600, rng, phi=0.1)
    rep = (opt + 0.015 + 0.3 * rng.standard_normal(600))
    lw = ledoit_wolf_sharpe_test(rep, opt, method="bootstrap", block=5, n_boot=399, seed=7,
                                 periods_per_year=252)
    res = joint_contrasts(np.column_stack([opt, rep]), np.array([[-1.0, 1.0]]), block=5,
                          n_boot=399, seed=7)
    # identical resamples and identical bootstrap statistic: the (1 − α) quantile of d* is the same
    assert res.zstar(0.05)[0] == pytest.approx(lw.params["z_star"], rel=1e-9)
    assert res.estimate[0] == pytest.approx(lw.difference, rel=1e-10)
    # the standard error of the sample uses a scalar HAC of the influence function: within 5 %
    assert res.se[0] == pytest.approx(lw.se, rel=0.05)
    lo, hi = res.ci(0.05)
    assert (lo[0], hi[0]) == pytest.approx((lw.ci_low, lw.ci_high), rel=0.05, abs=0.02)


@pytest.mark.parametrize("phi", [0.0, 0.3])
def test_scalar_hac_standard_error_is_close_to_the_lw_pair_hac(phi):
    rng = np.random.default_rng(9)
    ratios = []
    for _ in range(6):
        a = _opt(2000, rng, mu=0.05, phi=phi)
        b = 0.8 * a + 0.6 * _opt(2000, rng, mu=0.03, phi=phi)
        lw = ledoit_wolf_sharpe_test(b, a, periods_per_year=252)
        res = joint_contrasts(np.column_stack([a, b]), np.array([[-1.0, 1.0]]), block=5,
                              n_boot=19, seed=1)
        ratios.append(res.se[0] / lw.se)
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.02) and 0.95 < min(ratios) < max(ratios) < 1.05


def test_pvalue_is_the_lw_formula_and_inverts_the_studentised_interval():
    rng = np.random.default_rng(10)
    for k in range(6):
        opt = _opt(240, rng, phi=0.1)
        reps = _replicas(opt, rng, 0.02 * k, 2, noise=0.4)
        res = joint_contrasts(np.column_stack([opt, reps]), _difference_row(2), block=6,
                              n_boot=399, seed=k)
        count = res.pvalue()[0] * 400 - 1
        assert count == pytest.approx(round(count), abs=1e-9)  # (#{d* ≥ d} + 1) / (M + 1)
        for alpha in (0.05, 0.10):
            lo, hi = res.ci(alpha)
            assert (not lo[0] <= 0.0 <= hi[0]) == (res.pvalue()[0] <= alpha)
        lo90, hi90 = res.ci(0.10)
        lo95, hi95 = res.ci(0.05)
        assert lo95[0] <= lo90[0] <= res.estimate[0] <= hi90[0] <= hi95[0]


def test_percentile_interval_comes_from_the_bootstrap_distribution():
    rng = np.random.default_rng(11)
    opt = _opt(300, rng)
    res = joint_contrasts(np.column_stack([opt, _replicas(opt, rng, 0.02, 2)]), _difference_row(2),
                          block=5, n_boot=499, seed=5)
    lo, hi = res.percentile_ci(0.10)
    np.testing.assert_allclose([lo[0], hi[0]], np.quantile(res.theta_star[:, 0], [0.05, 0.95]))


def test_identical_series_give_a_point_at_zero():
    rng = np.random.default_rng(12)
    x = _opt(200, rng)
    res = joint_contrasts(np.column_stack([x, x]), np.array([[-1.0, 1.0]]), block=4, n_boot=99, seed=1)
    assert res.estimate[0] == 0.0 and res.se[0] == 0.0
    assert res.pvalue()[0] == 1.0
    lo, hi = res.ci(0.10)
    assert lo[0] == hi[0] == 0.0


def test_joint_contrasts_reject_bad_input():
    rng = np.random.default_rng(13)
    s = rng.standard_normal((100, 3))
    w = np.array([[1.0, -1.0, 0.0]])
    with pytest.raises(ValueError, match="functional"):
        joint_contrasts(s, w, functional="sortino", block=4, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="block"):
        joint_contrasts(s, w, block=60, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="weights"):
        joint_contrasts(s, np.array([[1.0, -1.0]]), block=4, n_boot=9, seed=1)
    with pytest.raises(TypeError):
        joint_contrasts(s, w, block=4, n_boot=9)  # the seed is mandatory
    bad = s.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        joint_contrasts(bad, w, block=4, n_boot=9, seed=1)
    const = s.copy()
    const[:, 2] = 0.5
    with pytest.raises(ValueError, match="variance"):
        joint_contrasts(const, np.array([[1.0, 0.0, -1.0]]), block=4, n_boot=9, seed=1)


@pytest.mark.slow
def test_size_under_the_null_of_equal_sharpe_ratios():
    # H0 true: the optimum and three replicas are N(0.03, 1) with correlations 0.89 / 0.8
    rng = np.random.default_rng(31)
    n_rep, rej = 200, 0
    for k in range(n_rep):
        z = rng.standard_normal(500)
        series = np.column_stack(
            [0.03 + z] + [0.03 + math.sqrt(0.8) * z + math.sqrt(0.2) * rng.standard_normal(500)
                          for _ in range(3)]
        )
        res = joint_contrasts(series, _difference_row(3), block=3, n_boot=299, seed=k)
        rej += res.pvalue()[0] <= 0.05
    print(f"[stats-report] RQ3 contrast size at 5%, T=500, R={n_rep}: {rej / n_rep:.4f}")
    assert 0.01 <= rej / n_rep <= 0.11, rej / n_rep


@pytest.mark.slow
def test_studentised_interval_covers_the_true_difference():
    # every replica has mean 0.05 against 0.03 and the same sd: the true annual ΔSR is 0.02·√252
    rng = np.random.default_rng(32)
    truth = 0.02 * ROOT252
    n_rep, covered = 200, 0
    for k in range(n_rep):
        z = rng.standard_normal(500)
        series = np.column_stack(
            [0.03 + z] + [0.05 + math.sqrt(0.8) * z + math.sqrt(0.2) * rng.standard_normal(500)
                          for _ in range(3)]
        )
        res = joint_contrasts(series, _difference_row(3), block=3, n_boot=299, seed=k)
        lo, hi = res.ci(0.05)
        covered += lo[0] <= truth <= hi[0]
    print(f"[stats-report] RQ3 95% interval coverage, T=500, R={n_rep}: {covered / n_rep:.4f}")
    assert 0.90 <= covered / n_rep <= 0.99, covered / n_rep


# ---------------------------------------------------------------------------------------------
# margins
# ---------------------------------------------------------------------------------------------


def test_sharpe_margin_is_the_sesoi_over_the_optimum_volatility():
    assert sharpe_margin(0.10) == pytest.approx(0.025)  # 25 bp of CE per year at 10 % vol
    assert sharpe_margin(0.20, sesoi=0.005) == pytest.approx(0.025)
    with pytest.raises(ValueError):
        sharpe_margin(0.0)


def test_variance_margin_translates_the_sesoi_into_a_relative_variance_band():
    m = variance_margin(5.0, 0.04)  # (λ/2)·Δσ² = 0.0025 → Δσ² = 0.001 → δ = 0.025
    assert m.delta == pytest.approx(0.025)
    assert (m.low, m.high) == pytest.approx((math.log(0.975), math.log(1.025)))
    assert variance_margin(20.0, 0.01).delta == pytest.approx(0.025)
    with pytest.raises(ValueError, match="δ"):
        variance_margin(1.0, 0.004)  # δ = 1.25: log(1 − δ) does not exist


# ---------------------------------------------------------------------------------------------
# the four outcomes of H3a
# ---------------------------------------------------------------------------------------------


def test_outcome_table():
    assert H3A_OUTCOMES == ("equivalent", "distinguishable_within_sesoi", "different", "inconclusive")
    assert h3a_outcome(iut_passes=True, any_significant=False) == "equivalent"
    assert h3a_outcome(iut_passes=True, any_significant=True) == "distinguishable_within_sesoi"
    assert h3a_outcome(iut_passes=False, any_significant=True) == "different"
    assert h3a_outcome(iut_passes=False, any_significant=False) == "inconclusive"


def _equivalence(shifts, *, T=3000, noise=0.05, margin=0.5, seed=21, n_rep=4, **kw):
    rng = np.random.default_rng(seed)
    opt = _opt(T, rng)
    reps = [_replicas(opt, rng, s, n_rep, noise) for s in shifts]
    levels = [5.0, 10.0, 25.0][: len(shifts)]
    return equivalence_test(opt, reps, levels, margin, block=5, n_boot=499, seed=seed, **kw)


def test_equivalent_when_every_level_is_inside_the_margin_and_none_differs():
    res = _equivalence([0.0, 0.0])
    assert res.outcome == "equivalent" and res.iut_passes and not res.any_significant
    assert all(lv.equivalent and not lv.significant for lv in res.levels)
    assert res.margin == (-0.5, 0.5) and res.alpha == 0.05 and res.functional == "sharpe"


def test_distinguishable_but_within_the_margin():
    # +0.01 per day = +0.16 annual Sharpe: tiny standard error (replicas ≈ optimum), inside ±0.5
    res = _equivalence([0.01, 0.01])
    assert res.outcome == "distinguishable_within_sesoi"
    assert res.iut_passes and res.any_significant
    for lv in res.levels:
        assert lv.estimate == pytest.approx(0.01 * ROOT252, abs=0.03)
        assert lv.ci_tost[0] > -0.5 and lv.ci_tost[1] < 0.5 and lv.pvalue_holm <= 0.05


def test_different_when_a_level_leaves_the_margin():
    res = _equivalence([0.0, 0.05])  # the second level: +0.8 annual Sharpe
    assert res.outcome == "different" and not res.iut_passes and res.any_significant
    assert res.levels[0].equivalent and not res.levels[1].equivalent  # IUT: one failure is enough
    assert res.levels[1].significant and not res.levels[0].significant


def test_inconclusive_when_neither_equivalence_nor_a_difference_can_be_shown():
    rng = np.random.default_rng(22)
    opt = _opt(500, rng)
    independent = 0.03 + rng.standard_normal((500, 3))  # unrelated paths: a wide interval
    res = equivalence_test(opt, [independent], [10.0], 0.5, block=5, n_boot=499, seed=22)
    assert res.outcome == "inconclusive" and not res.iut_passes and not res.any_significant


def test_holm_adjusts_the_two_sided_tests_across_levels():
    res = _equivalence([0.0, 0.0, 0.003])
    raw = np.array([lv.pvalue for lv in res.levels])
    adj = np.array([lv.pvalue_holm for lv in res.levels])
    from cif.stats import holm

    np.testing.assert_allclose(adj, holm(raw))
    assert [lv.significant for lv in res.levels] == list(adj <= 0.05)
    assert [lv.gap_bp for lv in res.levels] == [5.0, 10.0, 25.0]


def test_tost_uses_the_one_minus_two_alpha_interval():
    res = _equivalence([0.0], alpha=0.05)
    lv = res.levels[0]
    assert lv.ci_tost[0] > lv.ci[0] and lv.ci_tost[1] < lv.ci[1]  # 90 % inside 95 %
    wider = _equivalence([0.0], alpha=0.025)
    assert wider.levels[0].ci_tost[0] < lv.ci_tost[0]  # α = 0.025: 95 % interval for TOST


def test_gmv_arm_tests_the_log_variance_ratio_against_an_asymmetric_margin():
    rng = np.random.default_rng(23)
    opt = 0.01 * _opt(3000, rng, mu=0.0)
    margin = variance_margin(5.0, 0.04)  # δ = 0.025
    e = 0.05

    def reps(scale):
        # population variance scale² · var(opt): sqrt(1 − e²)·opt + e·(an independent copy)
        return np.column_stack([
            scale * (math.sqrt(1 - e**2) * opt + e * 0.01 * rng.standard_normal(opt.size))
            for _ in range(4)
        ])

    same = equivalence_test(opt, [reps(1.0)], [10.0], (margin.low, margin.high),
                            functional="logvar", block=5, n_boot=499, seed=1)
    assert same.outcome == "equivalent" and same.margin == (margin.low, margin.high)
    assert same.levels[0].estimate == pytest.approx(0.0, abs=5e-3)
    wider = equivalence_test(opt, [reps(1.10)], [10.0], (margin.low, margin.high),
                             functional="logvar", block=5, n_boot=499, seed=1)
    assert wider.outcome == "different"
    assert wider.levels[0].estimate == pytest.approx(math.log(1.10**2), abs=0.01)


def test_equivalence_test_with_a_calibrated_block():
    rng = np.random.default_rng(24)
    opt = _opt(400, rng, phi=0.1)
    reps = _replicas(opt, rng, 0.0, 2, noise=0.3)
    kw = {"block_grid": (1, 2, 4), "n_pseudo": 12, "n_boot": 59}
    res = equivalence_test(opt, [reps], [10.0], 0.5, block="calibrate", n_boot=99, seed=3,
                           calibration_kwargs=kw)
    assert res.block in (1, 2, 4) and res.calibration is not None
    assert res.calibration.block == res.block
    cal = calibrate_joint_block(opt, [reps], seed=np.random.default_rng(1), **kw)
    assert cal.block in (1, 2, 4) and cal.rejection_rates.shape == (3,)


def test_calibration_works_on_returns_in_percent_as_in_the_lw_code():
    """LW's HAC (unit weights, 0.97 cap) is not unit-free and the plan §7.4 uses percent: daily
    returns given as fractions are multiplied by 100 before Algorithm 3.1."""
    rng = np.random.default_rng(26)
    opt = 0.01 * _opt(400, rng, mu=0.03, phi=0.1)  # fractions: sd 1 %
    reps = [opt[:, None] + 0.003 * rng.standard_normal((400, 2))]
    kw = {"block_grid": (1, 2, 4), "n_pseudo": 12, "n_boot": 59}
    a = calibrate_joint_block(opt, reps, seed=5, **kw)  # percent=True by default
    b = calibrate_joint_block(100 * opt, [100 * reps[0]], seed=5, percent=False, **kw)
    np.testing.assert_array_equal(a.rejection_rates, b.rejection_rates)
    assert a.block == b.block


def test_equivalence_test_checks_its_arguments():
    rng = np.random.default_rng(25)
    opt = _opt(100, rng)
    reps = _replicas(opt, rng, 0.0, 2)
    with pytest.raises(ValueError, match="levels"):
        equivalence_test(opt, [reps], [10.0, 25.0], 0.5, block=4, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="same length"):
        equivalence_test(opt, [reps[:-1]], [10.0], 0.5, block=4, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="alpha"):
        equivalence_test(opt, [reps], [10.0], 0.5, block=4, n_boot=9, seed=1, alpha=0.6)
    with pytest.raises(ValueError, match="at least one level"):
        equivalence_test(opt, [], [], 0.5, block=4, n_boot=9, seed=1)
    with pytest.raises(ValueError, match="block"):
        equivalence_test(opt, [reps], [10.0], 0.5, block="auto", n_boot=9, seed=1)


# ---------------------------------------------------------------------------------------------
# H3b: dose-response slope on log g
# ---------------------------------------------------------------------------------------------

_LEVELS = (5.0, 10.0, 25.0, 50.0)


def _dose_panel(kappa, *, T=3000, noise=0.05, n_rep=3, seed=41):
    """Daily mean shift of the replicas at level g is kappa·log g (per-period units)."""
    rng = np.random.default_rng(seed)
    opt = _opt(T, rng)
    reps = [_replicas(opt, rng, kappa * math.log(g), n_rep, noise) for g in _LEVELS]
    return opt, reps


def test_slope_is_the_ols_slope_of_the_level_differences_on_log_g():
    opt, reps = _dose_panel(0.004)
    res = dose_response(opt, reps, _LEVELS, block=5, n_boot=199, seed=1)
    per_level = np.array([lv.estimate for lv in res.per_level])
    slope, intercept = np.polyfit(np.log(_LEVELS), per_level, 1)
    assert res.slope.estimate == pytest.approx(slope, rel=1e-9)
    assert res.intercept.estimate == pytest.approx(intercept, rel=1e-7, abs=1e-9)
    assert res.levels == _LEVELS and res.x_transform == "log"
    for g, lv in zip(_LEVELS, res.per_level, strict=True):
        direct = equivalence_test(opt, [reps[_LEVELS.index(g)]], [g], 1.0, block=5, n_boot=19, seed=1)
        assert lv.estimate == pytest.approx(direct.levels[0].estimate, rel=1e-9)


def test_slope_interval_covers_the_true_slope_and_excludes_zero():
    kappa, noise = 0.004, 0.05
    opt, reps = _dose_panel(kappa, noise=noise)
    res = dose_response(opt, reps, _LEVELS, block=5, n_boot=499, seed=2)
    true_slope = kappa * ROOT252 / math.sqrt(1 + noise**2)  # per unit of log g, annual Sharpe
    lo, hi = res.slope.ci
    assert lo <= true_slope <= hi and lo > 0.0
    assert res.slope.pvalue <= 0.05 and res.significant
    pl, ph = res.slope.ci_percentile
    assert pl < res.slope.estimate < ph


def test_no_slope_when_the_replicas_do_not_depend_on_the_level():
    opt, reps = _dose_panel(0.0, noise=0.3, seed=43)
    res = dose_response(opt, reps, _LEVELS, block=5, n_boot=499, seed=3)
    assert res.slope.pvalue > 0.05 and not res.significant
    assert res.slope.ci[0] < 0.0 < res.slope.ci[1]


def test_linear_sensitivity_regresses_on_g_itself():
    opt, reps = _dose_panel(0.004)
    res = dose_response(opt, reps, _LEVELS, x_transform="linear", block=5, n_boot=99, seed=1)
    per_level = np.array([lv.estimate for lv in res.per_level])
    assert res.x_transform == "linear"
    assert res.slope.estimate == pytest.approx(np.polyfit(_LEVELS, per_level, 1)[0], rel=1e-9)


def test_prediction_is_the_fitted_line_inside_the_range_of_levels_only():
    opt, reps = _dose_panel(0.004)
    res = dose_response(opt, reps, _LEVELS, block=5, n_boot=99, seed=1)
    for dose in (5.0, 12.0, 50.0):
        assert res.predict(dose) == pytest.approx(
            res.intercept.estimate + res.slope.estimate * math.log(dose), rel=1e-9)
    for dose in (4.99, 50.01, 0.5):
        with pytest.raises(ValueError, match="descriptive"):
            res.predict(dose)


def test_slope_needs_three_levels():
    opt, reps = _dose_panel(0.004)
    with pytest.raises(ValueError, match="3 levels"):
        dose_response(opt, reps[:2], _LEVELS[:2], block=5, n_boot=19, seed=1)
    with pytest.raises(ValueError, match="x_transform"):
        dose_response(opt, reps, _LEVELS, x_transform="sqrt", block=5, n_boot=19, seed=1)


def test_gmv_slope_uses_the_mean_log_variance_ratio():
    rng = np.random.default_rng(44)
    opt = 0.01 * _opt(3000, rng, mu=0.0)
    reps = [np.column_stack([(1 + 0.02 * math.log(g)) * opt + 0.0005 * rng.standard_normal(opt.size)
                             for _ in range(3)]) for g in _LEVELS]
    res = dose_response(opt, reps, _LEVELS, functional="logvar", block=5, n_boot=499, seed=1)
    # log ratio = 2·log(1 + 0.02·log g) ≈ 0.04·log g
    assert res.slope.estimate == pytest.approx(0.04, rel=0.15)
    assert res.slope.ci[0] > 0.0 and res.functional == "logvar"


# ---------------------------------------------------------------------------------------------
# H3c: the dose of a real solver
# ---------------------------------------------------------------------------------------------


def test_dose_is_the_mean_realised_gap_over_f_a():
    gaps = np.array([2.0, 40.0, 6.0, 8.0, 100.0])
    in_fa = np.array([True, False, True, True, False])
    d = solver_dose(gaps, in_fa, (5.0, 10.0, 25.0))
    assert d.dose_bp == pytest.approx(16.0 / 3.0) and d.n_folds == 3
    assert d.inside_range and not d.descriptive and d.reason is None


@pytest.mark.parametrize(("gaps", "reason"), [
    ([1.0, 2.0], "below"),    # mean 1.5 < min L = 5
    ([60.0, 80.0], "above"),  # mean 70 > max L = 50
])
def test_dose_outside_the_levels_is_descriptive(gaps, reason):
    d = solver_dose(np.array(gaps), np.array([True, True]), (5.0, 10.0, 25.0, 50.0))
    assert d.descriptive and not d.inside_range and reason in d.reason


def test_dose_range_is_inclusive_and_degenerate_cases_are_descriptive():
    assert not solver_dose(np.array([5.0]), np.array([True]), (5.0, 10.0)).descriptive
    assert not solver_dose(np.array([10.0]), np.array([True]), (5.0, 10.0)).descriptive
    none = solver_dose(np.array([5.0]), np.array([False]), (5.0, 10.0))
    assert none.descriptive and none.dose_bp is None and none.n_folds == 0
    assert solver_dose(np.array([5.0]), np.array([True]), ()).descriptive  # L is empty
    with pytest.raises(ValueError, match="finite"):
        solver_dose(np.array([np.nan, 5.0]), np.array([True, True]), (5.0, 10.0))
    with pytest.raises(ValueError, match="length"):
        solver_dose(np.array([5.0, 6.0]), np.array([True]), (5.0, 10.0))
