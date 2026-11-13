"""RQ2 statistics: the pooled estimate of ``log R``, its stratified bootstrap and the five outcomes
(ANALYSIS_PLAN §2 RQ2 "Подтверждающая оценка (N8)", §7.1 rows H2 and H2b, §7.2, §10).

**Estimand.** For an instance ``R = PI⁺(QUBO)/PI⁺(exact)`` (H2) or ``R_b = PI⁺(QUBO)/PI⁺(LS)`` (H2b); ``R < 1``
means the QUBO pipeline is better. The confirmatory answer is the **mean of** ``log R`` **over the cells** C1, C2, C3,
C5 (and C4 if the λ rule admits it) **with equal cell weights**. Inside a cell the mean runs over its *clusters*:
the instances (C1–C4), or, in C5 (ORlib), the ports, each port being the mean over its λ points ("среднее по port из
средних по λ"). This module knows nothing of instances or rows: it takes, per cell, the vector of cluster means of
each metric (``log_r``, ``log_r_b``).

**Interval.** The 95 % interval of the pooled mean is the percentile interval of a **stratified bootstrap**: the
clusters of every cell are redrawn with replacement within the cell, as many as the cell has (instances within cells,
ports in C5), the cell means are recomputed and averaged with equal weights. The cells get their own percentile
intervals from the same resamples at the level of the **Bonferroni** family of the pooled cells (99 % simultaneous:
each of the ``m`` cells at ``1 − 0.01/m``); they are descriptive. The plan fixes none of the bootstrap's details; the
reading is that of the other estimates of the project (:func:`cif.stats.rq1.cluster_bootstrap_table`): percentile
intervals, ``B = 9999`` resamples (:data:`N_BOOT`), an explicit seed (:data:`SEED`), one
``numpy.random.SeedSequence`` stream per cell, ``[seed, STREAM_TAG, crc32(cell)]``, so that the resamples of a cell do
not depend on which other cells are estimated, and the same redrawn clusters serve every metric (H2 and H2b are
estimated on the same instances, and a pooled value is the average of the cell values of one resample).

With very few clusters the percentile interval is coarse. The bootstrap mean of two clusters ``[a, b]`` is ``a``,
``(a+b)/2`` or ``b`` with probabilities 1/4, 1/2, 1/4, so at the levels used here the interval is exactly ``[a, b]``;
for three clusters each extreme has probability 1/27, still above the 2.5 % tail, so the interval is the range of the
three values. C5 enters the pooled estimate with as few as three ports (§12 A1): its interval is then the range of
the port means. That is the interval the plan asks for ("port в C5"); it is reported as it is.

**Outcomes** (§2, §10), for the exponentiated interval ``[lo, hi]`` of ``R`` against the closed band
``[1/1.5, 1.5]`` (:func:`outcome`; on the log scale ``[−ln 1.5, ln 1.5]``):

* ``qubo_better`` — the interval lies wholly below the band (``hi < 1/1.5``);
* ``qubo_worse`` — wholly above it (``lo > 1.5``);
* ``equivalent`` — inside the band (the edges belong to it), whether or not it contains 1;
* ``direction_only`` — it excludes 1 and crosses an edge of the band (the direction is established, the size is not);
* ``undetermined`` — it contains 1 and crosses an edge (or covers the band).

Every random function takes an explicit seed; the result is a deterministic function of the arguments.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "CHUNK",
    "EQUIVALENCE_FACTOR",
    "FAMILY_LEVEL",
    "N_BOOT",
    "OUTCOMES",
    "POOLED_LEVEL",
    "SEED",
    "STREAM_TAG",
    "CellResult",
    "Interval",
    "PooledResult",
    "Rq2Result",
    "cell_bootstrap",
    "estimate_rq2",
    "family_level",
    "mean_of_means",
    "outcome",
]

#: The practical-equivalence band of R is ``[1/1.5, 1.5]`` (§2 RQ2).
EQUIVALENCE_FACTOR = 1.5
#: Level of the interval of the pooled estimate (§2: "95 % ДИ").
POOLED_LEVEL = 0.95
#: Simultaneous level of the cell intervals (§2: "одновременные 99 % ДИ (Бонферрони)").
FAMILY_LEVEL = 0.99
#: Bootstrap resamples: as the other interval estimates of the project (``rq1.cluster_bootstrap``, E1 analysis).
N_BOOT = 9999
#: Bootstrap seed: outside the plan's seed ranges (instances 1000–2999, solvers 10 000–29 999) and E1's 50 000.
SEED = 60000
#: First stream tag of the entropy ``[seed, STREAM_TAG, crc32(cell)]`` (E1 uses its own table tags).
STREAM_TAG = 2
#: Resamples drawn per call of the generator, to bound memory. Part of the random stream: do not change.
CHUNK = 1000
#: The five outcomes of H2 and H2b (§2, §10), in the order of the plan.
OUTCOMES = ("qubo_better", "qubo_worse", "equivalent", "direction_only", "undetermined")


def outcome(low: float, high: float, *, factor: float = EQUIVALENCE_FACTOR) -> str:
    """One of :data:`OUTCOMES` for an interval ``[low, high]`` of the **log** ratio against the band
    ``[−ln factor, ln factor]`` (module docstring). The band is closed, "wholly outside" is strict."""
    low, high = float(low), float(high)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"the interval must be finite with low <= high, got ({low}, {high})")
    if not (math.isfinite(factor) and factor > 1.0):
        raise ValueError(f"the factor of the band must be a finite number above 1, got {factor}")
    edge = math.log(factor)
    if high < -edge:
        return "qubo_better"
    if low > edge:
        return "qubo_worse"
    if low >= -edge and high <= edge:
        return "equivalent"
    if high < 0.0 or low > 0.0:
        return "direction_only"
    return "undetermined"


def family_level(m: int, family: float = FAMILY_LEVEL) -> float:
    """Level of one interval of a Bonferroni family of ``m`` that is ``family`` simultaneously:
    ``1 − (1 − family)/m``."""
    if isinstance(m, bool) or not isinstance(m, int | np.integer) or m < 1:
        raise ValueError(f"the family size must be a positive integer, got {m!r}")
    if not 0.0 < family < 1.0:
        raise ValueError(f"the family level must lie in (0, 1), got {family}")
    return 1.0 - (1.0 - family) / int(m)


def mean_of_means(values: ArrayLike, clusters: ArrayLike) -> tuple[list, np.ndarray]:
    """The mean of ``values`` within each cluster: ``(labels, means)`` with the labels sorted. C5: the λ points of
    a port are averaged first, the ports are the clusters of the cell (their mean is the cell's)."""
    x = np.asarray(values, dtype=float).ravel()
    c = np.asarray(clusters).ravel()
    if x.size == 0 or x.size != c.size:
        raise ValueError(f"need non-empty values and clusters of the same length, got {x.size} and {c.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    labels = sorted(set(c.tolist()))
    return labels, np.array([x[c == label].mean() for label in labels])


def cell_bootstrap(data: ArrayLike, *, n_boot: int, entropy: Sequence[int]) -> np.ndarray:
    """Bootstrap means of the clusters of one cell: ``data`` is ``(n_clusters, n_metrics)``, the result
    ``(n_boot, n_metrics)``. The ``n_clusters`` clusters are redrawn with replacement (``n_clusters`` of them) in
    chunks of :data:`CHUNK` resamples from ``SeedSequence(entropy)``; one set of indices serves every metric."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError(f"data must be (n_clusters, n_metrics), got shape {x.shape}")
    n = x.shape[0]
    gen = np.random.default_rng(np.random.SeedSequence(list(entropy)))
    out = np.empty((n_boot, x.shape[1]))
    done = 0
    while done < n_boot:
        size = min(CHUNK, n_boot - done)
        idx = gen.integers(0, n, size=(size, n))
        out[done : done + size] = x[idx].mean(axis=1)
        done += size
    return out


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """Percentile interval of the mean log ratio at ``level`` (the log scale; ``exp`` gives R)."""

    level: float
    low: float
    high: float


@dataclass(frozen=True)
class CellResult:
    """One metric in one cell: the mean over its clusters, the interval of the family level (``None`` for a cell with
    one cluster, nothing to resample), its outcome (descriptive) and whether the cell is in the pooled estimate."""

    cell: str
    metric: str
    n_clusters: int
    estimate: float
    se: float | None
    interval: Interval | None
    outcome: str | None
    in_pooled: bool


@dataclass(frozen=True)
class PooledResult:
    """One metric pooled over ``cells`` with equal weights: the mean of the cell means, the bootstrap standard
    error, the 95 % interval and its outcome."""

    metric: str
    cells: tuple[str, ...]
    estimate: float
    se: float
    interval: Interval
    outcome: str


@dataclass(frozen=True)
class Rq2Result:
    """``pooled[metric]`` (``None`` without a pooled cell), ``cells[metric][cell]`` and the ``settings`` the numbers
    depend on."""

    pooled: dict[str, PooledResult] | None
    cells: dict[str, dict[str, CellResult]]
    settings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Plain dicts and floats (JSON-ready)."""

        def interval(iv: Interval | None):
            return None if iv is None else {"level": iv.level, "low": iv.low, "high": iv.high}

        pooled = None if self.pooled is None else {
            m: {"metric": r.metric, "cells": list(r.cells), "estimate": r.estimate, "se": r.se,
                "interval": interval(r.interval), "outcome": r.outcome} for m, r in self.pooled.items()}
        cells = {m: {c: {"cell": r.cell, "metric": r.metric, "n_clusters": r.n_clusters, "estimate": r.estimate,
                         "se": r.se, "interval": interval(r.interval), "outcome": r.outcome,
                         "in_pooled": r.in_pooled} for c, r in by_cell.items()} for m, by_cell in self.cells.items()}
        return {"pooled": pooled, "cells": cells, "settings": dict(self.settings)}


# --------------------------------------------------------------------------------------------
# The estimate
# --------------------------------------------------------------------------------------------


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _matrices(cells: Mapping[str, Mapping[str, ArrayLike]]) -> tuple[list[str], dict[str, np.ndarray]]:
    names = list(cells)
    metrics = list(cells[names[0]])
    if not metrics:
        raise ValueError("the cells carry no metrics")
    data = {}
    for name in names:
        if set(cells[name]) != set(metrics):
            raise ValueError(f"every cell needs the same metrics: {name!r} has {sorted(cells[name])}, "
                             f"{names[0]!r} has {sorted(metrics)}")
        columns = [np.asarray(cells[name][m], dtype=float).ravel() for m in metrics]
        if len({c.size for c in columns}) != 1:
            raise ValueError(f"the metrics of cell {name!r} need the same number of clusters")
        if columns[0].size == 0:
            raise ValueError(f"cell {name!r} has no clusters")
        matrix = np.column_stack(columns)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"cluster means of cell {name!r} must be finite")
        data[name] = matrix
    return metrics, data


def estimate_rq2(
    cells: Mapping[str, Mapping[str, ArrayLike]],
    *,
    pooled: Sequence[str],
    seed: int,
    n_boot: int = N_BOOT,
    level: float = POOLED_LEVEL,
    family: float = FAMILY_LEVEL,
    factor: float = EQUIVALENCE_FACTOR,
) -> Rq2Result:
    """The pooled estimate and the cell estimates (module docstring).

    ``cells[cell][metric]`` is the vector of the cluster means of the metric in the cell (all metrics of a cell the same
    length, all cells the same metrics); ``pooled`` names the cells of the pooled estimate (each needs at least two
    clusters). A cell outside ``pooled`` is estimated and described, with the interval of the same level as the cells
    of the family. The family of the Bonferroni intervals is the pooled cells (a family of one if there are none).
    """
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"seed must be an int (explicit seeds only), got {seed!r}")
    n_boot = _positive_int(n_boot, "n_boot")
    if not cells:
        raise ValueError("no cells to estimate")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must lie in (0, 1), got {level}")
    metrics, data = _matrices(cells)
    missing = [c for c in pooled if c not in data]
    if missing:
        raise ValueError(f"pooled cell {missing[0]!r} is not among the cells {sorted(data)}")
    in_pool = [c for c in data if c in set(pooled)]  # the order of ``cells``
    thin = [c for c in in_pool if data[c].shape[0] < 2]
    if thin:
        raise ValueError(f"pooled cell {thin[0]!r} has {data[thin[0]].shape[0]} cluster: at least two are needed to "
                         "resample it")
    family_size = max(1, len(in_pool))
    cell_level = family_level(family_size, family)
    samples = {
        name: cell_bootstrap(matrix, n_boot=n_boot, entropy=[int(seed), STREAM_TAG, zlib.crc32(name.encode("utf-8"))])
        for name, matrix in data.items() if matrix.shape[0] >= 2}

    def interval_of(values: np.ndarray, lvl: float) -> Interval:
        low, high = np.quantile(values, [(1.0 - lvl) / 2.0, (1.0 + lvl) / 2.0])
        return Interval(float(lvl), float(low), float(high))

    by_metric: dict[str, dict[str, CellResult]] = {m: {} for m in metrics}
    for name, matrix in data.items():
        for j, metric in enumerate(metrics):
            iv, se, verdict = None, None, None
            if name in samples:
                iv = interval_of(samples[name][:, j], cell_level)
                se = float(np.std(samples[name][:, j], ddof=1))
                verdict = outcome(iv.low, iv.high, factor=factor)
            by_metric[metric][name] = CellResult(
                name, metric, int(matrix.shape[0]), float(matrix[:, j].mean()), se, iv, verdict, name in in_pool)
    pooled_results = None
    if in_pool:
        pooled_results = {}
        for j, metric in enumerate(metrics):
            draws = np.mean([samples[c][:, j] for c in in_pool], axis=0)
            iv = interval_of(draws, level)
            pooled_results[metric] = PooledResult(
                metric, tuple(in_pool), float(np.mean([by_metric[metric][c].estimate for c in in_pool])),
                float(np.std(draws, ddof=1)), iv, outcome(iv.low, iv.high, factor=factor))
    settings = {"n_boot": n_boot, "seed": int(seed), "level": float(level), "family_level": float(family),
                "family_size": family_size, "cell_level": float(cell_level), "factor": float(factor),
                "interval": "percentile", "chunk": CHUNK, "stream_tag": STREAM_TAG, "pooled_cells": list(in_pool)}
    return Rq2Result(pooled_results, by_metric, settings)
