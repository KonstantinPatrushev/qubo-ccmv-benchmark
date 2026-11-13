"""RQ3 bookkeeping: confirmatory levels L, L_a, fold sets F_a, F_b (ANALYSIS_PLAN §2 RQ3, N2)."""

from __future__ import annotations

import numpy as np
import pytest

from cif.backtest.fold_sets import (
    G_STAR_BP,
    LEVELS_BP,
    MIN_REACH_SHARE,
    confirmatory_levels,
    days_in_folds,
    fold_sets,
    kbest_fold_mask,
    level_sets,
    reached_on_all,
)

LEVELS = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)


def _matrix(counts, n_folds):
    """reached[f, j] = True for the first counts[j] folds."""
    out = np.zeros((n_folds, len(counts)), dtype=bool)
    for j, c in enumerate(counts):
        out[:c, j] = True
    return out


def test_plan_constants():
    assert LEVELS_BP == LEVELS and G_STAR_BP == 10.0 and MIN_REACH_SHARE == 0.8


def test_confirmatory_levels_are_those_reached_on_at_least_80_percent_of_pilot_folds():
    reached = _matrix([0, 3, 7, 8, 10, 10, 9], 10)
    # 8 of 10 is exactly 80 %: in; 7 of 10: out
    assert confirmatory_levels(reached, LEVELS) == (10.0, 25.0, 50.0, 100.0)
    assert confirmatory_levels(reached, LEVELS, min_share=0.7) == (5.0, 10.0, 25.0, 50.0, 100.0)


@pytest.mark.parametrize(("n_folds", "count"), [(5, 4), (15, 12), (110, 88), (7, 6)])
def test_exact_eighty_percent_is_not_lost_to_floating_point(n_folds, count):
    reached = _matrix([count], n_folds)
    expected = (5.0,) if count / n_folds >= 0.8 - 1e-12 else ()
    assert confirmatory_levels(reached, (5.0,)) == expected
    assert confirmatory_levels(_matrix([count - 1], n_folds), (5.0,)) == ()


def test_confirmatory_levels_are_sorted_and_checked():
    reached = np.array([[True, True], [True, True]])
    assert confirmatory_levels(reached, (50.0, 10.0)) == (10.0, 50.0)
    with pytest.raises(ValueError, match="columns"):
        confirmatory_levels(reached, (10.0, 50.0, 100.0))
    with pytest.raises(ValueError, match="pilot folds"):
        confirmatory_levels(np.zeros((0, 2), dtype=bool), (10.0, 50.0))
    with pytest.raises(ValueError, match="min_share"):
        confirmatory_levels(reached, (10.0, 50.0), min_share=1.5)


def test_level_sets_split_at_g_star_and_flag_the_descriptive_cases():
    s = level_sets((5.0, 10.0, 25.0, 50.0))
    assert s.levels == (5.0, 10.0, 25.0, 50.0) and s.levels_a == (5.0, 10.0)  # g* itself is in
    assert not s.h3a_descriptive and not s.h3b_descriptive and s.g_star == 10.0
    high = level_sets((25.0, 50.0, 100.0))
    assert high.levels_a == () and high.h3a_descriptive and not high.h3b_descriptive
    two = level_sets((10.0, 25.0))
    assert two.levels_a == (10.0,) and not two.h3a_descriptive and two.h3b_descriptive  # |L| < 3
    assert level_sets(()).h3a_descriptive and level_sets(()).h3b_descriptive
    assert level_sets((50.0, 10.0, 25.0)).levels == (10.0, 25.0, 50.0)  # sorted


def test_fold_sets_require_all_levels_of_each_set():
    # folds 0..5; columns are the levels (5, 10, 25, 50)
    reached = np.array([
        [1, 1, 1, 1],  # everything
        [0, 1, 1, 1],  # 5 missed: irrelevant for L = (10, 25, 50)
        [0, 1, 1, 0],  # 50 missed: out of F_b, still in F_a
        [0, 0, 1, 1],  # 10 missed: out of both
        [1, 1, 0, 1],  # 25 missed: out of F_b only
        [0, 1, 1, 1],
    ], dtype=bool)
    sets = level_sets((10.0, 25.0, 50.0))  # L_a = (10,)
    out = fold_sets(reached, (5.0, 10.0, 25.0, 50.0), sets)
    np.testing.assert_array_equal(out.mask_a, [1, 1, 1, 0, 1, 1])
    np.testing.assert_array_equal(out.mask_b, [1, 1, 0, 0, 0, 1])
    assert out.n_folds == 6 and out.share_a == pytest.approx(5 / 6) and out.share_b == pytest.approx(3 / 6)
    assert np.all(out.mask_a[out.mask_b])  # F_b ⊆ F_a because L_a ⊆ L


def test_empty_level_sets_give_empty_fold_sets_instead_of_all_folds():
    reached = np.ones((4, 3), dtype=bool)
    high = level_sets((25.0, 50.0, 100.0))  # L_a empty
    out = fold_sets(reached, (25.0, 50.0, 100.0), high)
    assert not out.mask_a.any() and out.share_a == 0.0  # H3a is descriptive: no F_a
    assert out.mask_b.all()
    none = fold_sets(reached[:, :0], (), level_sets(()))
    assert not none.mask_a.any() and not none.mask_b.any()


def test_levels_are_matched_by_value_and_missing_columns_are_an_error():
    reached = np.ones((3, 2), dtype=bool)
    sets = level_sets((10.0, 25.0, 50.0))
    with pytest.raises(ValueError, match="50"):
        fold_sets(reached, (10.0, 25.0), sets)
    # the order of the columns does not matter
    np.testing.assert_array_equal(
        reached_on_all(np.array([[1, 0], [1, 1]], dtype=bool), (25.0, 10.0), (10.0,)), [False, True]
    )


def test_us_takes_l_a_from_ff49_and_needs_only_those_columns():
    ff49 = level_sets((5.0, 10.0, 25.0, 50.0))
    us_reached = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)  # US ran only L_a = (5, 10)
    mask = reached_on_all(us_reached, (5.0, 10.0), ff49.levels_a)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_empty_levels_in_reached_on_all():
    out = reached_on_all(np.ones((3, 2), dtype=bool), (1.0, 2.0), ())
    assert out.shape == (3,) and not out.any()


def test_days_in_folds_drops_the_test_days_of_excluded_folds():
    day_fold = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    keep = days_in_folds(day_fold, np.array([True, False, True]))
    np.testing.assert_array_equal(keep, [1, 1, 1, 0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="fold"):
        days_in_folds(np.array([0, 3]), np.array([True, False, True]))


def test_fold_set_shapes_are_validated():
    with pytest.raises(ValueError, match="columns"):
        fold_sets(np.ones((3, 2), dtype=bool), (10.0,), level_sets((10.0,)))


# -- F_k: the folds of the k best supports (operator (г), ANALYSIS_PLAN §2) ------------------


def test_a_fold_is_in_f_k_only_if_the_optimum_and_all_ranks_are_certified_and_enough_exist():
    optimum = np.array([1, 1, 1, 1, 0, 1], dtype=bool)
    ranks = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
    n_distinct = np.array([33, 40, 33, 32, 33, 33])
    mask = kbest_fold_mask(optimum, ranks, n_distinct, 32)
    # R = 32 needs 33 distinct portfolios; fold 3 has 32, fold 2 an uncertified rank, fold 4 an
    # uncertified optimum
    np.testing.assert_array_equal(mask, [True, True, False, False, False, True])
    assert mask.dtype == bool


def test_us_needs_rank_two_only():
    mask = kbest_fold_mask([True, True, True], [True, True, False], [1, 2, 5], 1)
    np.testing.assert_array_equal(mask, [False, True, False])  # R = 1: two portfolios at least


def test_the_two_arms_are_combined_by_intersection_and_feed_the_day_mask():
    mv = kbest_fold_mask([1, 1, 1, 1], [1, 1, 1, 1], [33, 33, 33, 20], 32)
    gmv = kbest_fold_mask([1, 1, 1, 1], [1, 0, 1, 1], [33, 33, 33, 33], 32)
    both = mv & gmv
    np.testing.assert_array_equal(both, [True, False, True, False])
    keep = days_in_folds(np.array([0, 0, 1, 2, 3, 3]), both)
    np.testing.assert_array_equal(keep, [1, 1, 0, 1, 0, 0])


def test_the_f_k_rule_accepts_scalars_for_a_single_fold():
    assert bool(kbest_fold_mask(True, True, 33, 32))
    assert not bool(kbest_fold_mask(True, True, 32, 32))


def test_f_k_arguments_are_validated():
    with pytest.raises(ValueError, match="n_ranks"):
        kbest_fold_mask([True], [True], [5], 0)
    with pytest.raises(ValueError, match="shape"):
        kbest_fold_mask([True, True], [True], [5, 5], 1)
