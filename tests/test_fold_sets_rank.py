"""RQ3 bookkeeping for the rank paths (plan v2.4, operator (г)): the fold set ``F_k`` and ``ḡ_r``.

ANALYSIS_PLAN §2 RQ3 ("Набор фолдов F_k"; "Гипотезы", H3b). Synthetic flags and gaps only.
"""

from __future__ import annotations

import numpy as np
import pytest

from cif.backtest.fold_sets import (
    FF49_RANKS,
    US_RANKS,
    enough_distinct,
    rank_doses,
    rank_fold_set,
)


def _flags(rows):
    return np.array(rows, dtype=bool)


def test_plan_ranks():
    assert list(FF49_RANKS) == list(range(2, 34)) and len(FF49_RANKS) == 32  # R = 32
    assert US_RANKS == (2,)


def test_enough_distinct_needs_r_plus_one_portfolios():
    # ranks 2..33 need the optimum and 32 more distinct refined portfolios: 33 in all
    np.testing.assert_array_equal(
        enough_distinct([0, 32, 33, 40], max(FF49_RANKS)), [False, False, True, True]
    )
    np.testing.assert_array_equal(enough_distinct([1, 2], max(US_RANKS)), [False, True])


def test_fold_set_keeps_folds_with_a_certified_optimum_ranks_and_enough_portfolios():
    optimum = _flags([1, 1, 1, 1, 0, 1, 1])
    ranks = _flags([[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    distinct = _flags([1, 1, 1, 0, 1, 1, 1])
    fk = rank_fold_set(optimum, ranks, distinct)
    np.testing.assert_array_equal(fk.mask, [1, 1, 0, 0, 0, 1, 1])
    assert fk.n_folds == 7 and fk.n_included == 4 and fk.n_excluded == 3 and not fk.is_empty
    assert fk.share_included == pytest.approx(4 / 7) and fk.share_excluded == pytest.approx(3 / 7)


def test_every_excluded_fold_is_counted_once_by_the_first_failing_criterion():
    # order of the criteria: optimum not certified > degenerate > a rank not certified
    optimum = _flags([0, 1, 1, 1, 1])
    ranks = _flags([[0], [0], [0], [1], [1]])  # folds 0, 1, 2 miss rank 2
    distinct = _flags([0, 0, 1, 1, 1])  # folds 0, 1 are degenerate
    fk = rank_fold_set(optimum, ranks, distinct)
    np.testing.assert_array_equal(fk.mask, [0, 0, 0, 1, 1])
    assert (fk.n_excluded_optimum, fk.n_excluded_degenerate, fk.n_excluded_rank) == (1, 1, 1)
    assert fk.n_excluded == 3  # the three counts add up to the excluded folds


def test_us_needs_only_rank_two_as_a_vector_or_a_single_column():
    optimum = _flags([1, 1, 1])
    vec = rank_fold_set(optimum, _flags([1, 0, 1]))
    col = rank_fold_set(optimum, _flags([[1], [0], [1]]))
    np.testing.assert_array_equal(vec.mask, [1, 0, 1])
    np.testing.assert_array_equal(vec.mask, col.mask)


def test_one_uncertified_rank_removes_the_fold_from_every_path():
    # the plan: ranks 2..33 are needed on every fold, a missing rank 17 excludes the whole fold
    n = 4
    ranks = np.ones((n, 32), dtype=bool)
    ranks[2, FF49_RANKS.index(17)] = False
    fk = rank_fold_set(np.ones(n, dtype=bool), ranks)
    np.testing.assert_array_equal(fk.mask, [1, 1, 0, 1])


def test_all_folds_enough_by_default_and_an_empty_set_is_flagged():
    fk = rank_fold_set(_flags([1, 1]), _flags([[1], [1]]))
    assert fk.mask.all() and fk.n_excluded == 0
    none = rank_fold_set(_flags([0, 0]), _flags([[1], [1]]))
    assert none.is_empty and none.share_included == 0.0 and none.n_excluded_optimum == 2


def test_excluded_folds_drop_their_days_from_every_path():
    day_fold = np.repeat(np.arange(5), 4)  # five folds of four test days
    fk = rank_fold_set(_flags([1, 0, 1, 1, 1]), _flags([[1], [1], [1], [0], [1]]))
    np.testing.assert_array_equal(fk.mask, [1, 0, 1, 0, 1])  # folds 1 and 3 are out
    keep = fk.day_mask(day_fold)
    assert keep.shape == (20,) and keep.sum() == 12
    opt = np.arange(20.0)
    paths = np.column_stack([opt + 100.0, opt + 200.0, opt + 300.0])  # optimum + three rank paths
    kept_opt, kept_paths = opt[keep], paths[keep]
    # the same days survive in the optimum and in every path: the panel stays balanced
    assert kept_paths.shape == (12, 3) and kept_opt.shape == (12,)
    np.testing.assert_array_equal(kept_paths - kept_opt[:, None], [[100.0, 200.0, 300.0]] * 12)
    np.testing.assert_array_equal(kept_opt[[0, 4, 8]], [0.0, 8.0, 16.0])  # fold 0, 2, 4 starts
    with pytest.raises(ValueError, match="fold"):
        fk.day_mask(np.array([0, 5]))


def test_the_fold_set_ignores_the_gap_structure_so_ties_and_zero_gaps_stay_in():
    # no selection by the structure of the gaps (plan): F_k is a function of the flags alone, and
    # folds with tied or zero doses are averaged like the others
    fk = rank_fold_set(_flags([1, 1, 1]), _flags([[1], [1], [1]]))
    gaps = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 6.0]])  # fold 0: non-binding (all ties at 0)
    np.testing.assert_allclose(rank_doses(gaps, (2, 3), fk.mask), [2.0, 8.0 / 3.0])


def test_rank_doses_are_the_mean_gap_over_f_k_by_rank():
    gaps = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 8.0], [100.0, 200.0, 400.0]])
    mask = np.array([True, True, False])
    np.testing.assert_allclose(rank_doses(gaps, (2, 3, 4), mask), [2.0, 2.5, 6.0])
    # ranks are picked by value, in the requested order
    np.testing.assert_allclose(rank_doses(gaps, (2, 3, 4), mask, ranks=(4, 2)), [6.0, 2.0])


def test_rank_doses_do_not_decrease_with_the_rank_when_each_fold_is_sorted():
    rng = np.random.default_rng(0)
    gaps = np.sort(rng.exponential(5.0, size=(40, 32)), axis=1)
    mask = rng.random(40) < 0.7
    doses = rank_doses(gaps, FF49_RANKS, mask)
    assert np.all(np.diff(doses) >= 0.0) and doses.shape == (32,)


def test_rank_doses_allow_missing_gaps_outside_f_k_only():
    gaps = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])  # fold 1 was not certified
    np.testing.assert_allclose(rank_doses(gaps, (2, 3), np.array([True, False, True])), [2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        rank_doses(gaps, (2, 3), np.array([True, True, True]))


def test_rank_doses_check_their_arguments():
    gaps = np.ones((3, 2))
    mask = np.array([True, True, False])
    with pytest.raises(ValueError, match="rank 7"):
        rank_doses(gaps, (2, 3), mask, ranks=(7,))
    with pytest.raises(ValueError, match="columns"):
        rank_doses(gaps, (2, 3, 4), mask)
    with pytest.raises(ValueError, match="folds"):
        rank_doses(gaps, (2, 3), mask[:2])
    with pytest.raises(ValueError, match="no folds"):
        rank_doses(gaps, (2, 3), np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="rank 1"):
        rank_doses(gaps, (1, 2), mask)  # rank 1 is the optimum: its gap is zero by definition


def test_rank_fold_set_checks_its_arguments():
    ok = _flags([1, 1, 1])
    with pytest.raises(ValueError, match="folds"):
        rank_fold_set(ok, _flags([[1], [1]]))
    with pytest.raises(ValueError, match="folds"):
        rank_fold_set(ok, _flags([[1], [1], [1]]), _flags([1, 1]))
    with pytest.raises(ValueError, match="1-D"):
        rank_fold_set(_flags([[1, 1, 1]]), _flags([[1], [1], [1]]))
