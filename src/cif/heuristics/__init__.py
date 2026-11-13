"""QUBO-pipeline heuristics: the shared second stage and swap-SA, and the anytime machinery of RQ2.

Imported here: :mod:`~cif.heuristics.pipeline` and :mod:`~cif.heuristics.swap_sa`. The rest is
imported from its module (importing it here would close a cycle with
:mod:`cif.quantum.neal_sampler`): :mod:`~cif.heuristics.start` (the common start of every method),
:mod:`~cif.heuristics.anytime` (the outer loop of SA / Tabu / swap-SA by wall-clock budget) and
:mod:`~cif.heuristics.local_search` (local search on the true objective).
"""

from cif.heuristics.pipeline import (
    check_feasible,
    iter_sampleset_subsets,
    legacy_turnover_cost,
    refine_candidates,
)
from cif.heuristics.swap_sa import (
    SwapSAResult,
    SwapSurrogate,
    calibrate_hot_temperature,
    sample_swap_sa,
    solve_with_swap_sa,
)

__all__ = [
    "SwapSAResult",
    "SwapSurrogate",
    "calibrate_hot_temperature",
    "check_feasible",
    "iter_sampleset_subsets",
    "legacy_turnover_cost",
    "refine_candidates",
    "sample_swap_sa",
    "solve_with_swap_sa",
]
