"""E1: attribution of the gap of the QUBO pipeline to the surrogate and to the sampler (RQ1).

Experiment module of the runner (:mod:`cif.experiments.run`, protocol of :mod:`cif.experiments.toy`):
``cells(cfg)`` is the grid, ``run_cell(cell, cfg)`` one result row. ANALYSIS_PLAN §2 (RQ1), §6, §7.3.

A cell is one sampler batch (about 1 s of sampling, no outer loop) on one instance::

    pair_seed, n, h              the instance: level h of the pair with that seed (cif.instances.pairs)
    sampler, penalty, beta,
    cold_ratio                   the sampler configuration (nested factors, below; ``None`` where the
                                 sampler has none)
    solver_seed                  the seed of the sampler (calibration 10000-19999, test 20000-29999)

Samplers and their factors (§7.3: nested, so modelled separately per sampler)::

    sa       penalty multiplier × β range: ``default`` (dwave's own, which scales with the penalty) or
             ``objective`` (hot T where random swaps of the surrogate are accepted 80 % of the time,
             cold T = 0.01·hot, as in swap-SA; independent of the penalty)
    tabu     penalty multiplier (§12 B1: the work unit is reads, a read is one plain tabu search,
             ``num_restarts = 0``, no clock: the deterministic regime of R12; tenure ``max(5, N // 4)``)
    swap_sa  the cold end of the schedule, ``T_cold = cold_ratio · T_hot`` (it works on the K-manifold, there
             is no penalty): the reference 0.01 and, exploratory (§12 B3), 0.2

The penalty multiplier is that of :func:`cif.qubo.builder.build_selection_bqm` (multiples of
:func:`cif.qubo.builder.selection_penalty_scale`: 1, 3, 10, 30, 100).

**References (§12 B3).** There is no tuning step in RQ1, so the "tuned configuration" of §6 and §7.3 is the
*reference* of every sampler (:data:`REFERENCE`): SA penalty 100 with the β range of ``dwave-samplers``
(1000 sweeps), tabu penalty 100 (reads, B1), swap-SA 200 sweeps with the hot temperature of the 80 %
acceptance calibration and ``T_cold = 0.01·T_hot``; the number of reads of each is sized by the pilot on
the reference. The samplers are contrasted at the references; the other levels of the nested grid are
described. The swap-SA level ``T_cold = 0.2·T_hot`` (:data:`EXPLORATORY_COLD_RATIO`) is exploratory, and so
is any configuration chosen after the grid was seen (:func:`config_role`).

A row holds ``E_form_m``, ``E_pipe_m``, ``E_samp_m`` (units of the scale ``s*``), the counts
``n_form_m`` / ``n_pipe_m`` for every m of ``ms``, the split of the residual ``E_samp_m = E_prec_m +
E_short_m`` (§12 B2; ``E_form_at_npipe_m`` is the ceiling at the size of the pool that was used, from
:attr:`cif.heuristics.oracle.SurrogateOracle.e_form_curve`), and diagnostics: the instance (``K, s_cont, binding,
depth = s_cont/K, cv_vol, cv_mu, cond_sigma``), the reference (``f_star, scale, ref_type`` and SCIP's
status, bounds, gap, nodes, times), the sampler (``build_s, solve_s, solve_cpu_s, n_samples,
n_valid_samples, frac_valid``, β range or temperatures, restarts), the pool (``pool_n_*``,
``hit_surrogate_min`` — did the sampler's best valid subset reach the exact surrogate minimum —,
``surrogate_gap_best``), Kendall's τ and the cost of the refinements (``refine_s, n_refined``).
``consistent`` is False if some E is below −1e-6, i.e. if something beat the reference.

The instance-level part (the pair, the certified optimum f*, the enumeration of all K-subsets and
the refinement of ``T_m``) does not depend on the sampler, so it is computed once per instance and
process and cached; cells are ordered instance by instance to make that effective. ``oracle_s`` is
the cost of that part, reported by every cell of the instance, ``oracle_cached`` says whether this
cell paid it. Subsets that the oracle already refined are not refined again for a pool
(``n_refined`` counts the refinements the cell itself ran).

Config (YAML) ``params``::

    seed_role: calibration | test      # instance seeds 1000-1999 / 2000-2999, solver 10000-19999 / 20000-29999
    instances: [{n: 16, pair_seeds: [1000, 1001]}, ...]
    h_levels: [0.15, 0.45]             # the two levels of the pair
    lam: 5.0, w_min: 0.0, w_max: 1.0
    ms: [1, 5, 20, 50, 200]
    solver_seeds: [10000, 10001]
    scip: {formulation, diag, mip_gap, time_limit_s, seed}
    samplers: [{name: sa, penalty: [...], beta: [default, objective], num_reads, num_sweeps},
               {name: tabu, penalty: [...], num_reads, num_restarts, tenure},
               {name: swap_sa, cold_ratio: [0.01, 0.2], num_reads, num_sweeps}]
    cell_order: instance | interleaved # optional, default instance (below)
    pilot: {m0, work: [{sampler, n, <batch keys>}, ...]}   # optional (below)

and the top-level ``seeds`` (recorded in the manifest) must list exactly the seeds of the grid.

**Work of a call, from the pilot (plan §6).** The size of one sampler call is sized on the pilot per N
(:mod:`cif.experiments.e1_pilot`). The confirmatory config therefore states it in
``params.pilot.work``, one entry per sampler and N with the batch keys of that sampler
(``num_reads``, ``num_sweeps`` or ``num_reads``, ``num_restarts``, ``tenure``), and the sampler specs
list only the factors. For tabu the block is in the regime of §12 B1 (``num_restarts = 0``, the tenure of
:func:`tabu_tenure`). A grid of test seeds holds the reference of every sampler in it (§12 B3). ``m0`` is
the floor rule of H1a found by the pilot (the analysis, :mod:`cif.stats.rq1`, reads it). A pilot block that
is not filled in (null values) is refused by :func:`cells`: nothing runs on guessed values. Without a
``pilot`` block the batch keys sit on the sampler specs and hold for every N (the smoke config).

**Cell order.** ``instance`` (default) lists the cells instance by instance, which keeps the
instance-level oracle cached in a single process. ``interleaved`` puts the pairs innermost (sampler
configuration, solver seed, h, then pair): the runner shards by stride (``cells[j::S]``), so when S
divides the number of pairs every shard gets whole pairs and computes the certified optimum and the
enumeration of an instance once, not once per shard.

The sampler of a cell is seeded from ``SeedSequence([solver_seed, pair_seed, n, h, sampler, penalty,
beta, cold_ratio])`` (the last only where the sampler has one), so that no two cells share a stream (§3);
the calibration of the objective-scaled β range from ``SeedSequence([solver_seed, pair_seed, n, h, tag])``,
so that the penalty levels of one instance and seed share that range exactly.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from cif.experiments.timing import measure
from cif.heuristics.oracle import (
    Reference,
    SurrogateOracle,
    certified_optimum,
    pool_subsets,
    surrogate_energies,
)
from cif.heuristics.pipeline import iter_sampleset_subsets
from cif.heuristics.swap_sa import SwapSurrogate, calibrate_hot_temperature, sample_swap_sa
from cif.instances.pairs import FactorPair, PairMember, factor_pair
from cif.problem import PortfolioProblem
from cif.quantum.neal_sampler import sample_tabu
from cif.qubo.builder import (
    SELECTION_PENALTY_MULTIPLIER,
    build_selection_bqm,
    selection_penalty_scale,
)

__all__ = [
    "EXPLORATORY_COLD_RATIO",
    "REFERENCE",
    "PoolDraw",
    "cells",
    "clear_caches",
    "config_label",
    "config_role",
    "run_cell",
    "sampler_seed",
    "tabu_tenure",
]

SAMPLERS = ("sa", "tabu", "swap_sa")
BETAS = ("default", "objective")
#: seed ranges of ANALYSIS_PLAN §3 (M2): (instances, solvers) for each role
SEED_RANGES = {
    "calibration": ((1000, 2000), (10_000, 20_000)),
    "test": ((2000, 3000), (20_000, 30_000)),
}
#: hot temperature: acceptance of random swaps; cold temperature: ratio to the hot one (swap-SA defaults)
ACCEPT_START = 0.8
COLD_RATIO = 0.01
#: §12 B3 (2026-10-11): the reference configuration of every sampler. RQ1 has no tuning step at N <= 24, so the
#: "tuned configuration" of §6 and §7.3 is this one: the work of a call is sized on it and the samplers are
#: contrasted at it. SA: penalty 100 (the builder's multiplier) and the β range of ``dwave-samplers``
#: (1000 sweeps); tabu: penalty 100 (reads, B1); swap-SA: 200 sweeps, hot temperature by the 80 %
#: acceptance calibration, cold temperature 0.01·hot.
REFERENCE = {
    "sa": {"penalty": SELECTION_PENALTY_MULTIPLIER, "beta": "default"},
    "tabu": {"penalty": SELECTION_PENALTY_MULTIPLIER},
    "swap_sa": {"cold_ratio": COLD_RATIO},
}
#: §12 B3: the swap-SA level ``T_cold = 0.2·T_hot``, an exploratory addition to the nested grid: it shows
#: whether the ceiling of the surrogate is within reach of a real sampler (the schedule of the reference
#: collapses its pool onto the minimum of the surrogate).
EXPLORATORY_COLD_RATIO = 0.2
#: an E below this (in units of s*) means that something beat the reference: the row is flagged
CONSISTENCY_TOL = 1e-6

#: §12 B1 (2026-10-11): the work unit of tabu is reads and a read is one plain tabu search (no restarts,
#: the deterministic regime of R12, the search length of ``dwave-samplers``). A call returns one state per
#: read, so its pool is bounded by the sampler, not by a constant of the implementation.
TABU_RESTARTS = 0

_SAMPLER_CODE = {"sa": 0, "tabu": 1, "swap_sa": 2}
_BETA_CODE = {None: 0, "default": 1, "objective": 2}
_CALIBRATION_TAG = 0xCA11B  # separates the calibration stream from every sampler stream
_BATCH_KEYS = {
    "sa": ("num_reads", "num_sweeps"),
    "tabu": ("num_reads", "num_restarts", "tenure"),
    "swap_sa": ("num_reads", "num_sweeps"),
}
CELL_ORDERS = ("instance", "interleaved")


# --------------------------------------------------------------------------------------------
# The grid
# --------------------------------------------------------------------------------------------


def _require(mapping: dict, key: str, where: str):
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{where}: missing '{key}'")
    return mapping[key]


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def config_label(sampler: str, penalty: float | None, beta: str | None, cold_ratio: float | None) -> str:
    """The id of a configuration: ``sa/p100/default``, ``tabu/p100``, ``swap_sa/c0.01``."""
    parts = [sampler]
    if penalty is not None:
        parts.append(f"p{penalty:g}")
    if beta is not None:
        parts.append(beta)
    if cold_ratio is not None:
        parts.append(f"c{cold_ratio:g}")
    return "/".join(parts)


def config_role(sampler: str, penalty: float | None = None, beta: str | None = None,
                cold_ratio: float | None = None) -> str:
    """``reference`` (:data:`REFERENCE`, §12 B3), ``exploratory`` (a swap-SA level other than the reference:
    the plan names ``T_cold = 0.2·T_hot``, and what is not in the plan is not pre-registered) or ``grid`` (the
    other levels of the nested grid of §7.3, described, not contrasted)."""
    if sampler not in REFERENCE:
        raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}")
    ref = REFERENCE[sampler]
    if sampler == "swap_sa":
        if cold_ratio is None:
            raise ValueError("a swap-SA configuration has a cold_ratio")
        return "reference" if math.isclose(cold_ratio, ref["cold_ratio"]) else "exploratory"
    if penalty is None or not math.isclose(penalty, ref["penalty"]):
        return "grid"
    return "reference" if sampler == "tabu" or beta == ref["beta"] else "grid"


def _sampler_configs(
    spec: dict, *, work_from_pilot: bool = False
) -> list[tuple[str, float | None, str | None, float | None]]:
    """``(sampler, penalty, beta, cold_ratio)`` of every configuration a sampler spec stands for.

    The batch keys sit on the spec, or, with ``work_from_pilot``, in ``params.pilot.work`` (then
    they must not be on the spec: the work is stated once).
    """
    name = spec.get("name")
    if name not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; expected one of {SAMPLERS}")
    where = f"sampler {name!r}"
    factors = {"sa": ("penalty", "beta"), "tabu": ("penalty",), "swap_sa": ("cold_ratio",)}[name]
    if work_from_pilot:
        twice = [key for key in _BATCH_KEYS[name] if spec.get(key) is not None]
        if twice:
            raise ValueError(
                f"{where}: {', '.join(repr(key) for key in twice)} given both on the sampler and in "
                f"params.pilot.work; the work of a call is stated once, in the pilot block"
            )
        required = factors
    else:
        required = (*factors, *_BATCH_KEYS[name])
    missing = [key for key in required if spec.get(key) is None]
    if missing:
        raise ValueError(f"{where}: missing {', '.join(repr(key) for key in missing)}")
    extra = [key for key in ("penalty", "beta", "cold_ratio") if key in spec and key not in factors]
    if extra:
        raise ValueError(f"{where} has no {' or '.join(extra)}: not one of its factors")
    if name == "swap_sa":  # works on the K-manifold: no penalty; the cold end of the schedule varies (§12 B3)
        ratios = spec["cold_ratio"]
        if (not isinstance(ratios, list) or not ratios or len(set(ratios)) != len(ratios)
                or any(isinstance(r, bool) or not isinstance(r, (int, float)) or not 0.0 < r <= 1.0 for r in ratios)):
            raise ValueError(f"{where}: cold_ratio must be a non-empty list of distinct ratios "
                             f"T_cold / T_hot in (0, 1], got {ratios!r}")
        return [(name, None, None, float(r)) for r in ratios]
    penalties = [float(x) for x in spec["penalty"]]
    if not penalties or any(not x > 0.0 for x in penalties):
        raise ValueError(f"{where}: penalty must be a non-empty list of positive multipliers")
    if name == "tabu":
        return [(name, x, None, None) for x in penalties]
    betas = list(spec["beta"])
    if not betas or any(b not in BETAS for b in betas):
        raise ValueError(f"{where}: beta must be a non-empty list of {BETAS}, got {betas}")
    return [(name, x, b, None) for x in penalties for b in betas]


def _check_references(configs: list[tuple[str, float | None, str | None, float | None]]) -> None:
    """A grid of test seeds holds the reference configuration of every sampler in it (§12 B3): the work of a
    call was sized on it and the samplers are contrasted at it."""
    present = set(configs)
    missing = []
    for name in dict.fromkeys(c[0] for c in configs):
        ref = REFERENCE[name]
        if (name, ref.get("penalty"), ref.get("beta"), ref.get("cold_ratio")) not in present:
            missing.append(f"{name} {ref}")
    if missing:
        raise ValueError(f"the reference configuration of {', '.join(missing)} is not in the grid (§12 B3: the "
                         f"work of a call is sized on it and the samplers are contrasted at it)")


def tabu_tenure(n: int) -> int:
    """The tenure of tabu at ``n`` assets: ``max(5, n // 4)`` (§12 B1: the default ``n // 4`` of
    ``dwave-samplers``, not below the lower end of the tuning range of Appendix B)."""
    return max(5, int(n) // 4)


def _check_tabu_regime(batch: dict, n: int, where: str) -> None:
    """The batch keys of tabu are those of §12 B1: ``num_restarts`` is :data:`TABU_RESTARTS` (a read is one
    plain search, the work is the number of reads) and the tenure is :func:`tabu_tenure` of ``n``."""
    restarts, tenure = batch["num_restarts"], batch["tenure"]
    if isinstance(restarts, bool) or restarts != TABU_RESTARTS:
        raise ValueError(f"{where}: num_restarts must be {TABU_RESTARTS} (§12 B1: a read is one plain tabu "
                         f"search, the work is the number of reads), got {restarts!r}")
    if isinstance(tenure, bool) or tenure != tabu_tenure(n):
        raise ValueError(f"{where}: tenure must be max(5, N // 4) = {tabu_tenure(n)} at n={n} (§12 B1), "
                         f"got {tenure!r}")


_SCIP_KEYS = ("formulation", "diag", "mip_gap", "time_limit_s", "seed")


def _check_scalars(p: dict) -> None:
    """Everything the result depends on is stated in the config (R8): fail on a dry run, not per cell."""
    for key in ("lam", "w_min", "w_max"):
        _require(p, key, "params")
    ms = _require(p, "ms", "params")
    if (not isinstance(ms, list) or not ms
            or any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in ms)
            or any(b <= a for a, b in pairwise(ms))):
        raise ValueError(f"params.ms must be a non-empty, strictly increasing list of positive "
                         f"integers, got {ms}")
    scip = _require(p, "scip", "params")
    for key in _SCIP_KEYS:
        _require(scip, key, "params.scip")
    limit = scip["time_limit_s"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not limit > 0:
        raise ValueError(f"params.scip.time_limit_s must be a positive number, got {limit!r}")


def _check_seeds(cfg: dict, role: str, pair_seeds: list[int], solver_seeds: list[int]) -> None:
    (lo_i, hi_i), (lo_s, hi_s) = SEED_RANGES[role]
    bad_i = [s for s in pair_seeds if not lo_i <= s < hi_i]
    bad_s = [s for s in solver_seeds if not lo_s <= s < hi_s]
    if bad_i or bad_s:
        raise ValueError(
            f"seed_role={role!r} takes instance seeds in [{lo_i}, {hi_i}) and solver seeds in "
            f"[{lo_s}, {hi_s}); out of range: instance seeds {bad_i}, solver seeds {bad_s}"
        )
    declared = cfg.get("seeds")
    used = sorted({*pair_seeds, *solver_seeds})
    if declared is not None and sorted({int(s) for s in declared}) != used:
        raise ValueError(
            f"the top-level seeds {sorted(declared)} (recorded in the manifest) must be the seeds "
            f"of the grid {used}"
        )


def _check_pilot(pilot, specs: list[dict], ns: list[int], ms: list[int], role: str) -> None:
    """The pilot block (module docstring) is filled in, complete for every sampler and N, and sound.

    On test seeds the block must not come from a pilot report that was not valid for §12 (a smoke
    or dirty run, a grid that missed the target): ``source.valid_for_section_12`` is false then.
    """
    if not isinstance(pilot, dict):
        raise ValueError(f"params.pilot must be a mapping, got {type(pilot).__name__}")
    where = "params.pilot"
    if pilot.get("work") is None or pilot.get("m0") is None:
        raise ValueError(
            f"{where} is not filled in (m0 and work are null): run the pilot "
            f"(configs/E1/pilot_floor.yaml and pilot_timing.yaml) and write its values into the config "
            f"with `python -m cif.experiments.e1_pilot make-config`; nothing runs on guessed values"
        )
    source = pilot.get("source")
    if role == "test" and isinstance(source, dict) and source.get("valid_for_section_12") is False:
        raise ValueError(
            f"{where}.source: the values come from a pilot report that is not valid for §12 "
            f"({source.get('run_ids') or source.get('note')}); a confirmatory run needs a valid pilot"
        )
    m0 = pilot["m0"]
    if not _is_count(m0) or m0 not in ms:
        raise ValueError(f"{where}.m0 must be one of the pool sizes ms {ms}, got {m0!r}")
    if not isinstance(pilot["work"], list):
        raise ValueError(f"{where}.work must be a list of {{sampler, n, ...}} entries")
    by_name = {s.get("name"): s for s in specs}
    seen: set[tuple[str, int]] = set()
    for entry in pilot["work"]:
        name, n = entry.get("sampler"), entry.get("n")
        if name not in SAMPLERS:
            raise ValueError(f"{where}.work: unknown sampler {name!r}; expected one of {SAMPLERS}")
        if name not in by_name:
            raise ValueError(f"{where}.work: sampler {name!r} is not among params.samplers")
        if n not in ns:
            raise ValueError(f"{where}.work: sampler {name!r}, n={n} is not in params.instances {ns}")
        if (name, n) in seen:
            raise ValueError(f"{where}.work: duplicate entry for sampler {name!r}, n={n}")
        seen.add((name, n))
        label = f"{where}.work: sampler {name!r}, n={n}"
        unexpected = sorted(set(entry) - {"sampler", "n", *_BATCH_KEYS[name]})
        if unexpected:
            raise ValueError(f"{label}: unexpected {', '.join(repr(k) for k in unexpected)} "
                             f"(batch keys of {name} are {_BATCH_KEYS[name]})")
        for key in _BATCH_KEYS[name]:
            if key not in entry:
                raise ValueError(f"{label}: missing {key!r}")
            if entry[key] is None:
                raise ValueError(f"{label}: {key!r} is null (fill in the pilot values)")
            if name == "tabu" and key == "num_restarts":
                continue  # zero is its value (a plain search): checked with the tenure below
            if not _is_count(entry[key]):
                raise ValueError(f"{label}: {key!r} must be a positive integer, got {entry[key]!r}")
        if name == "tabu":
            _check_tabu_regime(entry, n, label)
    for name in by_name:
        for n in ns:
            if (name, n) not in seen:
                raise ValueError(f"{where}.work has no entry for sampler {name!r} at n={n}")


def cells(cfg: dict) -> list[dict]:
    """The grid: pair seeds × h × sampler configurations × solver seeds, in the ``cell_order``."""
    p = _require(cfg, "params", "config")
    role = p.get("seed_role")
    if role not in SEED_RANGES:
        raise ValueError(f"params.seed_role must be one of {sorted(SEED_RANGES)}, got {role!r}")
    order = p.get("cell_order", "instance")
    if order not in CELL_ORDERS:
        raise ValueError(f"params.cell_order must be one of {CELL_ORDERS}, got {order!r}")
    _check_scalars(p)
    h_levels = [float(h) for h in _require(p, "h_levels", "params")]
    solver_seeds = [int(s) for s in _require(p, "solver_seeds", "params")]
    instances = []
    for entry in _require(p, "instances", "params"):
        n = int(_require(entry, "n", "an instance entry"))
        pair_seeds = [int(s) for s in _require(entry, "pair_seeds", f"instances (n={n})")]
        instances.append((n, pair_seeds))
    specs = _require(p, "samplers", "params")
    names = [s.get("name") for s in specs]
    for name in names:
        if names.count(name) > 1:
            raise ValueError(f"duplicate sampler {name!r}: list a sampler once")
    has_pilot = "pilot" in p
    if has_pilot:
        _check_pilot(p["pilot"], specs, [n for n, _ in instances], p["ms"], role)
    configs = [c for spec in specs for c in _sampler_configs(spec, work_from_pilot=has_pilot)]
    if role == "test":
        _check_references(configs)
    _check_seeds(cfg, role, [s for _, seeds in instances for s in seeds], solver_seeds)
    pairs = [(n, pair_seed) for n, pair_seeds in instances for pair_seed in pair_seeds]

    def cell(n, pair_seed, h, sampler, penalty, beta, cold_ratio, solver_seed) -> dict:
        return {"pair_seed": pair_seed, "n": n, "h": h, "sampler": sampler, "penalty": penalty,
                "beta": beta, "cold_ratio": cold_ratio, "solver_seed": solver_seed}

    if order == "interleaved":
        return [
            cell(n, pair_seed, h, sampler, penalty, beta, cold_ratio, solver_seed)
            for sampler, penalty, beta, cold_ratio in configs
            for solver_seed in solver_seeds
            for h in h_levels
            for n, pair_seed in pairs
        ]
    return [
        cell(n, pair_seed, h, sampler, penalty, beta, cold_ratio, solver_seed)
        for n, pair_seed in pairs
        for h in h_levels
        for sampler, penalty, beta, cold_ratio in configs
        for solver_seed in solver_seeds
    ]


# --------------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------------


def _derive(cell: dict) -> tuple[int, np.random.Generator]:
    """``(seed of the sampler, generator for the temperature calibration)`` of a cell.

    The seed has 31 bits: ``dwave`` samplers refuse seeds from ``2**31`` on although their
    message says ``2**32``. It depends on every coordinate of the cell. The calibration stream
    depends on the instance and the solver seed only: the objective-scaled temperature range is a
    property of those, so the penalty levels of one instance and seed share it exactly and the
    penalty effect is not mixed with the noise of the calibration.
    """
    penalty, cold_ratio = cell["penalty"], cell.get("cold_ratio")
    instance = [int(cell["solver_seed"]), int(cell["pair_seed"]), int(cell["n"]),
                round(cell["h"] * 1000)]
    sampler = [_SAMPLER_CODE[cell["sampler"]], 0 if penalty is None else round(penalty * 1000),
               _BETA_CODE[cell["beta"]]]
    if cold_ratio is not None:  # swap-SA only: SA and tabu keep the stream they had without the coordinate
        sampler.append(round(float(cold_ratio) * 1_000_000))
    seed = int(np.random.SeedSequence([*instance, *sampler]).generate_state(1, dtype=np.uint32)[0])
    calibration = np.random.default_rng(np.random.SeedSequence([*instance, _CALIBRATION_TAG]))
    return seed & 0x7FFFFFFF, calibration


def sampler_seed(cell: dict) -> int:
    """The 31-bit seed handed to the sampler of ``cell``: a function of every coordinate of it."""
    return _derive(cell)[0]


# --------------------------------------------------------------------------------------------
# The instance-level part (cached)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Instance:
    pair: FactorPair
    member: PairMember
    reference: Reference
    oracle: SurrogateOracle
    oracle_s: float


# The caches hold every instance of a shard (the confirmatory grid has 80 pairs, 160 instances): a
# shard of the interleaved order touches its instances in turn, and a smaller cache would repeat the
# certified optimum on almost every cell. An instance with its refined subsets is small: about 60 kB
# after the 16 configurations of one solver seed at N = 24 (2000 distinct subsets).
_CACHE_PAIRS = 128
_CACHE_INSTANCES = 256


@functools.lru_cache(maxsize=_CACHE_PAIRS)
def _pair(n, pair_seed, h_levels, lam, w_min, w_max) -> FactorPair:
    return factor_pair(n, pair_seed, h_levels=h_levels, lam=lam, w_min=w_min, w_max=w_max)


@functools.lru_cache(maxsize=_CACHE_INSTANCES)
def _instance(n, pair_seed, h, h_levels, lam, w_min, w_max, ms, scip) -> _Instance:
    t0 = time.perf_counter()
    pair = _pair(n, pair_seed, h_levels, lam, w_min, w_max)
    member = pair.member(h)
    options = dict(scip)  # complete: checked by cells()
    reference = certified_optimum(
        member.problem, pair.k, time_limit_s=float(options["time_limit_s"]),
        mip_gap=float(options["mip_gap"]), formulation=options["formulation"],
        diag=options["diag"], seed=int(options["seed"]),
    )
    oracle = SurrogateOracle(member.problem, pair.k, reference, ms=ms)
    return _Instance(pair, member, reference, oracle, time.perf_counter() - t0)


def clear_caches() -> None:
    """Forget the cached pairs and oracles (tests; a long process that changes configs)."""
    _pair.cache_clear()
    _instance.cache_clear()


# --------------------------------------------------------------------------------------------
# One sampler batch
# --------------------------------------------------------------------------------------------


@dataclass
class PoolDraw:
    """What a sampler batch gives the attribution: the pool and the cost of drawing it.

    ``pool`` lists subsets in the order of sampler energy (any size for the QUBO samplers: the
    oracle drops the invalid ones). ``extra`` holds sampler-specific diagnostics for the row.
    """

    pool: list[tuple[int, ...]]
    n_samples: int
    n_valid_samples: int
    build_s: float
    solve_s: float
    solve_cpu_s: float
    extra: dict = field(default_factory=dict)


def _draw_pool(
    problem: PortfolioProblem, k: int, cell: dict, spec: dict, seed: int, rng: np.random.Generator
) -> PoolDraw:
    """One batch of ``cell["sampler"]`` on ``problem``; ``seed`` and ``rng`` come from the cell."""
    name = cell["sampler"]
    n = problem.n
    if name == "swap_sa":
        res, timing = measure(
            sample_swap_sa, problem, k, num_reads=int(spec["num_reads"]),
            num_sweeps=int(spec["num_sweeps"]), seed=seed, cold_ratio=float(cell["cold_ratio"]),
        )
        return PoolDraw(
            pool=list(res.subsets), n_samples=res.num_reads, n_valid_samples=res.num_reads,
            build_s=0.0, solve_s=timing["wall_s"], solve_cpu_s=timing["cpu_s"],
            extra={"num_reads": res.num_reads, "num_sweeps": res.num_sweeps, "t_hot": res.t_hot,
                   "t_cold": res.t_cold, "accept_rate": res.accept_rate,
                   "frozen_fraction": res.frozen_fraction},
        )

    penalty = float(cell["penalty"])
    t0 = time.perf_counter()
    bqm = build_selection_bqm(problem, k, penalty_multiplier=penalty)
    extra: dict = {"penalty_abs": penalty * selection_penalty_scale(problem, k),
                   "num_reads": int(spec["num_reads"])}
    beta_range = None
    if name == "sa" and cell["beta"] == "objective":
        surrogate = SwapSurrogate(problem.mu, problem.sigma, problem.risk_aversion, k)
        t_hot = calibrate_hot_temperature(surrogate, rng, ACCEPT_START)
        beta_range = (1.0 / t_hot, 1.0 / (COLD_RATIO * t_hot))
    build_s = time.perf_counter() - t0

    if name == "sa":
        sampleset, timing = measure(
            SimulatedAnnealingSampler().sample, bqm, num_reads=int(spec["num_reads"]),
            num_sweeps=int(spec["num_sweeps"]), seed=seed, beta_range=beta_range,
        )
        hot, cold = sampleset.info["beta_range"]  # the sampler's own when none was given
        extra.update(num_sweeps=int(spec["num_sweeps"]), beta_hot=float(hot), beta_cold=float(cold))
    else:
        sampleset, timing = measure(
            sample_tabu, bqm, num_reads=int(spec["num_reads"]), tenure=int(spec["tenure"]),
            seed=seed, timeout_ms=None, num_restarts=int(spec["num_restarts"]),
        )
        extra.update(tenure=int(spec["tenure"]), num_restarts=int(spec["num_restarts"]),
                     restarts_mean=float(np.mean(sampleset.record.num_restarts)))
    sizes = np.asarray(sampleset.record.sample).sum(axis=1)
    return PoolDraw(
        pool=list(iter_sampleset_subsets(sampleset, n, order="energy")),
        n_samples=len(sampleset), n_valid_samples=int((sizes == k).sum()),
        build_s=build_s, solve_s=timing["wall_s"], solve_cpu_s=timing["cpu_s"], extra=extra,
    )


# --------------------------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------------------------


def _clean(value):
    """Python natives only, ``None`` for NaN and infinities: the row goes to JSON."""
    if isinstance(value, dict):
        return {key: _clean(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _batch_spec(p: dict, sampler: str, n: int) -> dict:
    """The sampler spec with the batch keys for ``n``: from ``params.pilot.work`` if there is a
    pilot block, else those on the spec itself (checked by :func:`cells`)."""
    spec = next(s for s in p["samplers"] if s["name"] == sampler)
    if "pilot" not in p:
        return spec
    entry = next(w for w in p["pilot"]["work"] if w["sampler"] == sampler and int(w["n"]) == n)
    return {**spec, **{key: entry[key] for key in _BATCH_KEYS[sampler]}}


def _config_id(cell: dict) -> str:
    return config_label(cell["sampler"], cell["penalty"], cell["beta"], cell.get("cold_ratio"))


def run_cell(cell: dict, cfg: dict) -> dict:
    """The row of one cell (see the module docstring)."""
    t_cell = time.perf_counter()
    p = cfg["params"]
    spec = _batch_spec(p, cell["sampler"], int(cell["n"]))
    key = (
        int(cell["n"]), int(cell["pair_seed"]), float(cell["h"]),
        tuple(float(h) for h in p["h_levels"]), float(p["lam"]), float(p["w_min"]),
        float(p["w_max"]), tuple(int(m) for m in p["ms"]), tuple(sorted(p["scip"].items())),
    )
    hits = _instance.cache_info().hits
    inst = _instance(*key)
    cached = _instance.cache_info().hits > hits
    problem, k = inst.member.problem, inst.pair.k
    oracle, reference = inst.oracle, inst.reference

    seed, rng = _derive(cell)
    draw = _draw_pool(problem, k, cell, spec, seed, rng)
    attribution = oracle.attribute(draw.pool)

    best = pool_subsets(draw.pool, k, problem.n, 1)[0]
    hit = gap = None
    if best:
        energy = float(surrogate_energies(problem.mu, problem.sigma, problem.risk_aversion,
                                          np.array(best))[0])
        hit = energy <= float(oracle.top_energy[0]) + 1e-12
        gap = (energy - float(oracle.top_energy[0])) / reference.scale
    e_values = np.concatenate([attribution.e_form, attribution.e_pipe])
    consistent = bool(np.all(e_values[np.isfinite(e_values)] >= -CONSISTENCY_TOL))
    meta = reference.meta
    stats = inst.member.stats

    row = {
        "status": "ok" if attribution.n_pipe[0] > 0 else "no_valid_samples",
        "config_id": _config_id(cell),
        "consistent": consistent,
        # the instance
        "K": k, "s_cont": inst.member.s_cont, "s_cont_min_pair": min(m.s_cont for m in inst.pair.members),
        "binding": inst.member.binding, "depth": inst.member.depth, "cv_vol": inst.member.cv_vol,
        "cv_mu": stats["cv_mu"], "cond_sigma": stats["cond_sigma"], "mean_corr": inst.member.meta["mean_corr"],
        "n_subsets": oracle.n_subsets,
        # the reference
        "f_star": reference.f_star, "f_scip": meta["f_scip"], "polish_gain": meta["polish_gain"],
        "scale": reference.scale, "ref_type": reference.ref_type, "scip_status": meta["status"],
        "scip_gap": meta["gap"], "scip_primal_bound": meta["primal_bound"],
        "scip_dual_bound": meta["dual_bound"], "scip_nodes": meta["n_nodes"],
        "scip_build_s": meta["build_time_s"], "scip_solve_s": meta["solve_time_s"],
        "scip_time_limit_hit": meta["time_limit_hit"],
        # the oracle (instance level: shared by the cells of the instance)
        "oracle_s": inst.oracle_s, "oracle_enum_s": oracle.enum_s, "oracle_refine_s": oracle.refine_s,
        "oracle_cached": cached,
        # the sampler batch
        "build_s": draw.build_s, "solve_s": draw.solve_s, "solve_cpu_s": draw.solve_cpu_s,
        "n_samples": draw.n_samples, "n_valid_samples": draw.n_valid_samples,
        "frac_valid": draw.n_valid_samples / draw.n_samples if draw.n_samples else 0.0,
        **draw.extra,
        "hit_surrogate_min": hit, "surrogate_gap_best": gap,
        # the attribution and the cost of the refinements the pool needed
        **attribution.row(),
        "refine_s": attribution.refine_s, "n_refined": attribution.n_refined,
        "refine_ms_per_call": (1e3 * attribution.refine_s / attribution.n_refined
                               if attribution.n_refined else None),
        "cell_s": time.perf_counter() - t_cell,
    }
    return _clean(row)
