"""RQ3 bookkeeping: confirmatory levels ``L``, ``L_a`` and the fold sets ``F_a``, ``F_b``.

ANALYSIS_PLAN §2 RQ3 ("Уровни и наборы фолдов", N2). The input is the reached / unreached matrix
of operator (б) over (fold, level) (:attr:`cif.backtest.injection.LevelResult.reached`).

* ``L`` — candidate levels reached on at least 80 % of the PILOT folds (:func:`confirmatory_levels`);
  ``L_a = {g ∈ L : g ≤ g*}``, ``g* = 10`` bp (:func:`level_sets`). ``L`` and ``L_a`` are
  determined separately for the MV and GMV arms; US-MV takes ``L_a`` of FF49-MV (no US pilot),
  see :func:`reached_on_all`.
* ``F_a`` — confirmatory folds where every level of ``L_a`` is reached, ``F_b`` — where every
  level of ``L`` is reached (:func:`fold_sets`); ``F_b ⊆ F_a`` since ``L_a ⊆ L``. A fold set whose
  levels are empty is EMPTY, not "all folds": with ``L_a = ∅`` H3a is descriptive
  (:attr:`LevelSets.h3a_descriptive`), with ``|L| < 3`` H3b is (:attr:`LevelSets.h3b_descriptive`).
* On the excluded folds the test-window days drop out of the comparison for the replicas AND the
  optimum alike (:func:`days_in_folds`).

Plan v2.4 makes the k best supports (operator (г), :mod:`cif.backtest.kbest`) the primary
operator; ``L``, ``L_a``, ``F_a``, ``F_b`` above stay only for the descriptive replica operator
(б). For the rank paths the fold set is ``F_k`` (:func:`rank_fold_set`; one arm straight from the
enumerator's output, :func:`kbest_fold_mask`): the folds where the optimum and EVERY rank the
analysis needs are certified and the fold is not degenerate (it has at least ``R + 1`` distinct
refined portfolios, :func:`enough_distinct`). For FF49 ``F_k`` is common to the MV and GMV arms:
combine the per-arm masks with ``&``. It is a function of the per-fold flags alone: there is no
selection by the structure of the gaps. A fold outside ``F_k`` drops out of ALL paths, the optimum
and every rank, so the panel is balanced (:meth:`RankFoldSet.day_mask`). The mean dose of rank
``r`` over ``F_k`` is ``ḡ_r`` (:func:`rank_doses`), the abscissa of the H3b regression.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "FF49_RANKS",
    "G_STAR_BP",
    "LEVELS_BP",
    "MIN_REACH_SHARE",
    "US_RANKS",
    "FoldSets",
    "LevelSets",
    "RankFoldSet",
    "confirmatory_levels",
    "days_in_folds",
    "enough_distinct",
    "fold_sets",
    "kbest_fold_mask",
    "level_sets",
    "rank_doses",
    "rank_fold_set",
    "reached_on_all",
]

LEVELS_BP = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)  # candidate levels g, bp of CE
G_STAR_BP = 10.0  # g*: the largest level of the equivalence test H3a
MIN_REACH_SHARE = 0.8  # a level enters L if reached on at least this share of pilot folds
FF49_RANKS = tuple(range(2, 34))  # ranks 2..33 in both arms of FF49 (R = 32)
US_RANKS = (2,)  # US-MV takes part in H3a only: rank 2

_SHARE_TOL = 1e-12  # 4/5 and 88/110 must count as 80 %
_LEVEL_ATOL = 1e-9  # levels are matched by value


def _matrix(reached, n_levels: int) -> np.ndarray:
    r = np.asarray(reached, dtype=bool)
    if r.ndim != 2 or r.shape[1] != n_levels:
        raise ValueError(
            f"reached must be a (folds × levels) matrix with {n_levels} columns, got shape {r.shape}"
        )
    return r


def confirmatory_levels(
    reached, levels: Sequence[float], min_share: float = MIN_REACH_SHARE
) -> tuple[float, ...]:
    """``L``: levels reached on at least ``min_share`` of the pilot folds, ascending.

    ``reached[f, j]`` says whether level ``levels[j]`` was reached on pilot fold ``f``.
    """
    if not 0.0 < min_share <= 1.0:
        raise ValueError(f"min_share must lie in (0, 1], got {min_share!r}")
    g = np.asarray(levels, dtype=float)
    r = _matrix(reached, g.size)
    if r.shape[0] == 0:
        raise ValueError("no pilot folds")
    keep = r.mean(axis=0) >= min_share - _SHARE_TOL
    return tuple(sorted(float(x) for x in g[keep]))


@dataclass(frozen=True)
class LevelSets:
    """``L`` and ``L_a`` of one arm, ascending, with the plan's rules for the descriptive cases."""

    levels: tuple[float, ...]
    levels_a: tuple[float, ...]
    g_star: float

    @property
    def h3a_descriptive(self) -> bool:
        """``L_a = ∅``: levels up to g* are unreachable by support swaps, H3a is descriptive."""
        return not self.levels_a

    @property
    def h3b_descriptive(self) -> bool:
        """``|L| < 3``: the slope is not estimated, the per-level differences are descriptive."""
        return len(self.levels) < 3


def level_sets(levels: Iterable[float], g_star: float = G_STAR_BP) -> LevelSets:
    """``L`` (as given, sorted) and ``L_a = {g ∈ L : g ≤ g*}``."""
    big = tuple(sorted(float(x) for x in levels))
    return LevelSets(big, tuple(g for g in big if g <= g_star + _LEVEL_ATOL), float(g_star))


def reached_on_all(reached, columns: Sequence[float], levels: Iterable[float]) -> np.ndarray:
    """Folds where every level of ``levels`` is reached; ``reached[:, j]`` belongs to the level
    ``columns[j]`` (matched by value, any order). Empty ``levels`` give an EMPTY mask. A level that
    is not a column is an error: e.g. US-MV runs only ``L_a`` of FF49-MV and asks only for it."""
    cols = np.asarray(columns, dtype=float)
    r = _matrix(reached, cols.size)
    wanted = tuple(levels)
    if not wanted:
        return np.zeros(r.shape[0], dtype=bool)
    picked = []
    for g in wanted:
        hit = np.flatnonzero(np.isclose(cols, g, rtol=0.0, atol=_LEVEL_ATOL))
        if hit.size != 1:
            raise ValueError(f"level {g:g} is not among the columns {tuple(cols.tolist())}")
        picked.append(int(hit[0]))
    return r[:, picked].all(axis=1)


@dataclass(frozen=True)
class FoldSets:
    """Boolean masks over the confirmatory folds: ``mask_a`` is ``F_a``, ``mask_b`` is ``F_b``."""

    mask_a: np.ndarray
    mask_b: np.ndarray

    @property
    def n_folds(self) -> int:
        return int(self.mask_a.size)

    @property
    def share_a(self) -> float:
        """``|F_a| / |F|``."""
        return float(self.mask_a.mean()) if self.n_folds else 0.0

    @property
    def share_b(self) -> float:
        """``|F_b| / |F|``."""
        return float(self.mask_b.mean()) if self.n_folds else 0.0


def fold_sets(reached, columns: Sequence[float], sets: LevelSets) -> FoldSets:
    """``F_a`` and ``F_b`` from the reached matrix of the confirmatory folds (see the module
    docstring); ``columns`` are the levels of its columns and must contain every level of ``L``."""
    return FoldSets(
        mask_a=reached_on_all(reached, columns, sets.levels_a),
        mask_b=reached_on_all(reached, columns, sets.levels),
    )


def kbest_fold_mask(optimum_certified, ranks_certified, n_distinct, n_ranks: int) -> np.ndarray:
    """``F_k`` of one arm: folds where the optimum and all needed ranks are certified and at least
    ``R + 1`` distinct refined portfolios exist (``R = n_ranks``: 32 for FF49, 1 for US-MV).

    Per fold: ``optimum_certified`` — the optimum carries its certificate; ``ranks_certified`` —
    :attr:`cif.backtest.kbest.KBestResult.all_certified`; ``n_distinct`` —
    :attr:`cif.backtest.kbest.KBestResult.n_distinct`. A fold with fewer than ``R + 1`` portfolios
    is degenerate and leaves ``F_k``. With ``Σz = K`` (the default of the enumerator) a fold with
    ``N_f ≤ K + 1`` has at most ``C(N_f, K) ≤ K + 1`` portfolios, so for ``R = 32`` the count alone
    excludes it; for ``R = 1`` (US) it would not, which is moot at ``N_f ≈ 100``. There is no
    selection by the structure of the gaps, and none by a tie at the boundary rank (reported
    separately). Scalars give a 0-d array.
    """
    if n_ranks < 1:
        raise ValueError(f"n_ranks must be ≥ 1, got {n_ranks!r}")
    optimum = np.asarray(optimum_certified, dtype=bool)
    ranks = np.asarray(ranks_certified, dtype=bool)
    count = np.asarray(n_distinct, dtype=int)
    if not optimum.shape == ranks.shape == count.shape:
        raise ValueError(
            f"per-fold inputs must have one shape, got {optimum.shape}, {ranks.shape}, {count.shape}"
        )
    return optimum & ranks & enough_distinct(count, n_ranks + 1)


def days_in_folds(day_fold, fold_mask) -> np.ndarray:
    """Day-level mask from a fold-level mask: ``day_fold[t]`` is the fold of test day ``t``."""
    d = np.asarray(day_fold, dtype=int)
    m = np.asarray(fold_mask, dtype=bool)
    if d.size and (d.min() < 0 or d.max() >= m.size):
        raise ValueError(f"day_fold refers to a fold outside 0..{m.size - 1}")
    return m[d]


# ---------------------------------------------------------------------------------------------
# operator (г), plan v2.4: the fold set F_k and the mean doses ḡ_r
# ---------------------------------------------------------------------------------------------


def enough_distinct(n_distinct, max_rank: int) -> np.ndarray:
    """Folds with enough distinct refined portfolios for ranks up to ``max_rank``.

    Ranks ``2..max_rank`` need the optimum and ``R = max_rank − 1`` more portfolios, that is
    ``R + 1 = max_rank`` in all: 33 for FF49 (ranks 2–33), 2 for US-MV. A fold with fewer is
    degenerate (``N_f ≤ K + 1``) and leaves ``F_k``.
    """
    if isinstance(max_rank, bool) or int(max_rank) != max_rank or max_rank < 2:
        raise ValueError(f"max_rank must be an integer ≥ 2, got {max_rank!r}")
    return np.asarray(n_distinct) >= int(max_rank)


@dataclass(frozen=True)
class RankFoldSet:
    """``F_k``: the boolean ``mask`` over the confirmatory folds and why the others are out.

    An excluded fold is counted once, by the first failing criterion in this order: the optimum
    is not certified (``n_excluded_optimum``), the fold is degenerate (``n_excluded_degenerate``),
    some needed rank is not certified (``n_excluded_rank``); the three add up to ``n_excluded``.
    """

    mask: np.ndarray
    n_excluded_optimum: int
    n_excluded_degenerate: int
    n_excluded_rank: int

    @property
    def n_folds(self) -> int:
        return int(self.mask.size)

    @property
    def n_included(self) -> int:
        return int(self.mask.sum())

    @property
    def n_excluded(self) -> int:
        return self.n_folds - self.n_included

    @property
    def share_included(self) -> float:
        """``|F_k| / |F|``."""
        return self.n_included / self.n_folds if self.n_folds else 0.0

    @property
    def share_excluded(self) -> float:
        """The share of folds that left every path (reported with the results)."""
        return self.n_excluded / self.n_folds if self.n_folds else 0.0

    @property
    def is_empty(self) -> bool:
        """No fold qualifies: the rank-path tests cannot be run."""
        return self.n_included == 0

    def day_mask(self, day_fold) -> np.ndarray:
        """Test days kept in EVERY path (optimum and all ranks): ``day_fold[t]`` is the fold of
        day ``t``; the days of excluded folds drop out of all paths alike."""
        return days_in_folds(day_fold, self.mask)


def rank_fold_set(optimum_certified, ranks_certified, distinct_ok=None) -> RankFoldSet:
    """``F_k`` from per-fold flags (plan v2.4, "Набор фолдов F_k").

    ``optimum_certified[f]`` — the optimum of fold ``f`` is certified; ``ranks_certified[f, j]`` —
    the ``j``-th needed rank is certified (a vector for a single rank, e.g. rank 2 of US-MV; ranks
    2–33 for FF49, in each arm separately); ``distinct_ok[f]`` — the fold has at least ``R + 1``
    distinct refined portfolios (:func:`enough_distinct`), by default every fold has. A fold stays
    only if all of these hold; the gaps themselves play no part in the choice.
    """
    opt = np.asarray(optimum_certified, dtype=bool)
    if opt.ndim != 1:
        raise ValueError("optimum_certified must be a 1-D array with one flag per fold")
    n = opt.size
    ranks = np.asarray(ranks_certified, dtype=bool)
    if ranks.ndim == 1:
        ranks = ranks[:, None]
    if ranks.ndim != 2 or ranks.shape[0] != n or ranks.shape[1] == 0:
        raise ValueError(
            f"ranks_certified must have one row per fold ({n} folds) and at least one rank, "
            f"got shape {ranks.shape}"
        )
    if distinct_ok is None:
        distinct = np.ones(n, dtype=bool)
    else:
        distinct = np.asarray(distinct_ok, dtype=bool)
        if distinct.shape != (n,):
            raise ValueError(
                f"distinct_ok must have one flag per fold ({n} folds), got shape {distinct.shape}"
            )
    ranks_ok = ranks.all(axis=1)
    return RankFoldSet(
        mask=opt & distinct & ranks_ok,
        n_excluded_optimum=int((~opt).sum()),
        n_excluded_degenerate=int((opt & ~distinct).sum()),
        n_excluded_rank=int((opt & distinct & ~ranks_ok).sum()),
    )


def rank_doses(
    gaps_bp, columns: Sequence[int], fold_mask, ranks: Sequence[int] | None = None
) -> np.ndarray:
    """``ḡ_r``: the mean in-sample gap ``ĝ_{f,r}`` (bp of CE) over the folds of ``F_k``, by rank.

    ``gaps_bp[f, j]`` is the dose of rank ``columns[j]`` on fold ``f`` (ranks ``≥ 2``: the
    optimum, rank 1, has the gap zero by definition). Gaps outside ``fold_mask`` are not looked
    at (they are missing on uncertified folds), gaps inside must be finite. ``ranks`` picks and
    orders the result by rank value; by default it is one value per column. By construction
    (each fold's gaps do not decrease in the rank) ``ḡ_r`` does not decrease in ``r``.
    """
    g = np.asarray(gaps_bp, dtype=float)
    cols = [int(c) for c in columns]
    if g.ndim != 2 or g.shape[1] != len(cols):
        raise ValueError(
            f"gaps_bp must be a (folds × ranks) matrix with {len(cols)} columns, got shape {g.shape}"
        )
    low = [c for c in cols if c < 2]
    if low:
        raise ValueError(f"rank {low[0]} is not a column of doses: the optimum (rank 1) has no gap")
    mask = np.asarray(fold_mask, dtype=bool)
    if mask.shape != (g.shape[0],):
        raise ValueError(
            f"fold_mask must have one flag per fold ({g.shape[0]} folds), got shape {mask.shape}"
        )
    chosen = g[mask]
    if chosen.shape[0] == 0:
        raise ValueError("no folds in F_k: the mean dose is undefined")
    if not np.all(np.isfinite(chosen)):
        raise ValueError("the gaps on the folds of F_k must be finite")
    doses = chosen.mean(axis=0)
    if ranks is None:
        return doses
    pick = []
    for r in ranks:
        hit = [j for j, c in enumerate(cols) if c == int(r)]
        if len(hit) != 1:
            raise ValueError(f"rank {r} is not among the columns {tuple(cols)}")
        pick.append(hit[0])
    return doses[pick]
