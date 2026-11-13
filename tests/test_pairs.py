"""Paired factor instances for RQ1 / H1a (ANALYSIS_PLAN §2 "Ячейки RQ1", §3).

Two instances with a common seed and ``vol_dispersion`` h ∈ {0.15, 0.45}, λ = 5, l = 0, u = 1 and a
cardinality K common to the pair: ``min(8, max(3, ⌊min(s_cont(h = 0.15), s_cont(h = 0.45))/2⌋))``
(the floor 3 is plan v2.4: with K = 2 the oracle's pools cover most of the K-subsets).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from cif.instances.diagnostics import binding_diagnostics, instance_statistics
from cif.instances.factor import factor_instance
from cif.instances.pairs import PAIR_H, PAIR_LAM, factor_pair, pair_cardinality


@pytest.mark.parametrize(
    ("s_conts", "expected"),
    [
        ((30, 40), 8),  # capped at 8 (C(24, 8) ≈ 7·10⁵ is the largest enumeration of RQ1)
        ((17, 40), 8),
        ((16, 100), 8),
        ((15, 100), 7),  # ⌊15/2⌋
        ((10, 12), 5),  # the smaller of the two decides
        ((12, 10), 5),
        ((6, 6), 3),
        ((7, 9), 3),
        ((5, 9), 3),  # ⌊5/2⌋ = 2 is raised to the floor 3 (plan v2.4)
        ((4, 4), 3),
        ((3, 9), 3),
        ((2, 2), 3),
        ((1, 50), 3),
        ((9,), 4),  # any number of levels
        ((20, 16, 30), 8),
    ],
)
def test_pair_cardinality_rule(s_conts, expected):
    assert pair_cardinality(s_conts) == expected


def test_pair_cardinality_needs_at_least_one_level():
    with pytest.raises(ValueError):
        pair_cardinality([])


def test_pair_members_are_the_generator_output_with_the_registered_parameters():
    pair = factor_pair(16, 1000)
    assert pair.n == 16 and pair.seed == 1000 and pair.lam == PAIR_LAM == 5.0
    assert PAIR_H == (0.15, 0.45)
    assert [m.h for m in pair.members] == [0.15, 0.45]
    for member in pair.members:
        mu, sigma, _ = factor_instance(16, 1000, vol_dispersion=member.h)
        p = member.problem
        np.testing.assert_array_equal(p.mu, mu)
        np.testing.assert_array_equal(p.sigma, sigma)
        assert (p.risk_aversion, p.w_min, p.w_max, p.budget) == (5.0, 0.0, 1.0, 1.0)
        assert p.cardinality == pair.k


def test_common_seed_gives_common_correlations_and_ordered_volatilities():
    """Only the volatility dispersion differs: the correlation matrix and the draws are shared."""
    pair = factor_pair(24, 1021)
    lo, hi = pair.members
    vol_lo, vol_hi = np.sqrt(np.diag(lo.problem.sigma)), np.sqrt(np.diag(hi.problem.sigma))
    corr_lo = lo.problem.sigma / np.outer(vol_lo, vol_lo)
    corr_hi = hi.problem.sigma / np.outer(vol_hi, vol_hi)
    np.testing.assert_allclose(corr_lo, corr_hi, atol=1e-12)
    np.testing.assert_array_equal(lo.meta["sector"], hi.meta["sector"])
    assert spearmanr(vol_lo, vol_hi).statistic > 0.95  # the same standardised log-volatility draws
    assert vol_hi.std() > vol_lo.std()


def test_k_is_common_and_follows_the_smaller_continuous_support():
    for n, seed in [(16, 1000), (16, 1001), (24, 1020), (24, 1021)]:
        pair = factor_pair(n, seed)
        s = [binding_diagnostics(m.problem)["s_cont"] for m in pair.members]
        assert [m.s_cont for m in pair.members] == s
        assert pair.k == min(8, max(3, min(s) // 2))
        assert 3 <= pair.k <= 8
        for member in pair.members:
            assert member.problem.cardinality == pair.k
            assert member.binding is (member.s_cont > pair.k)
            assert member.depth == pytest.approx(member.s_cont / pair.k, rel=1e-14)


def test_realised_volatility_cv_is_reported_per_level():
    pair = factor_pair(24, 1022)
    for member in pair.members:
        vol = np.sqrt(np.diag(member.problem.sigma))
        assert member.cv_vol == pytest.approx(vol.std() / vol.mean(), rel=1e-12)
    lo, hi = pair.members
    assert hi.cv_vol > lo.cv_vol  # the realised CV, not just the nominal h, is what H1a is about


def test_rows_are_json_safe_and_carry_the_diagnostics_of_the_plan():
    pair = factor_pair(16, 1003)
    rows = pair.rows()
    assert [r["h"] for r in rows] == [0.15, 0.45]
    for member, row in zip(pair.members, rows, strict=True):
        assert row["family"] == "factor" and row["seed"] == 1003 and row["n"] == 16
        assert row["lam"] == 5.0 and row["K"] == pair.k
        assert row["s_cont"] == member.s_cont and row["binding"] is member.binding
        assert row["depth"] == pytest.approx(member.depth)
        assert row["s_cont_min_pair"] == min(m.s_cont for m in pair.members)
        for key in ("cv_vol", "cv_mu", "cond_sigma", "vol_min", "vol_median", "vol_max",
                    "mean_corr", "f_cont", "cont_status"):
            assert key in row
        assert type(row["binding"]) is bool and type(row["s_cont"]) is int
    assert json.loads(json.dumps(rows, allow_nan=False)) == rows


def test_pairs_are_deterministic_and_seed_dependent():
    a, b, c = factor_pair(16, 1000), factor_pair(16, 1000), factor_pair(16, 1001)
    for ma, mb in zip(a.members, b.members, strict=True):
        np.testing.assert_array_equal(ma.problem.mu, mb.problem.mu)
        np.testing.assert_array_equal(ma.problem.sigma, mb.problem.sigma)
    assert not np.array_equal(a.members[0].problem.sigma, c.members[0].problem.sigma)


def test_member_lookup_by_level_and_custom_levels():
    pair = factor_pair(16, 1004)
    assert pair.member(0.45) is pair.members[1]
    with pytest.raises(KeyError):
        pair.member(0.3)
    three = factor_pair(16, 1004, h_levels=(0.1, 0.3, 0.5))
    assert [m.h for m in three.members] == [0.1, 0.3, 0.5]
    assert three.k == pair_cardinality([m.s_cont for m in three.members])


def test_generator_options_pass_through_but_the_dispersion_is_the_pair_factor():
    pair = factor_pair(16, 1005, mean_corr=0.4)
    assert pair.members[0].meta["mean_corr_target"] == 0.4
    with pytest.raises(TypeError):
        factor_pair(16, 1005, vol_dispersion=0.3)  # h_levels owns it


def test_instance_statistics_report_the_coefficient_of_variation_of_volatilities():
    sigma = np.diag([0.04, 0.09, 0.16])
    stats = instance_statistics(np.array([0.05, 0.10, 0.15]), sigma)
    vol = np.array([0.2, 0.3, 0.4])
    assert stats["cv_vol"] == pytest.approx(vol.std() / vol.mean())
    assert stats["cv_vol"] != pytest.approx(stats["cv_diag_sigma"])  # not the CV of the variances
