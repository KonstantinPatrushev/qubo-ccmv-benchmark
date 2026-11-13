"""Tuning spaces of E2: the variants of the exact class and the random search of the heuristics
(ANALYSIS_PLAN §6 "Настройка (N6)", Appendix B).

A **method spec** is a flat JSON-safe dict, the unit that a cell of :mod:`cif.experiments.e2_benchmark`
carries (so that a cell key changes when a configuration does)::

    {"name": "sa", "id": "sa-07", "num_reads_b": 120, "num_sweeps": 1500, "beta": "default",
     "penalty": 10.0, "m_b": 50}
    {"name": "tabu", "id": ..., "num_restarts_b": 20, "tenure": 15, "num_reads_b": 10, "penalty": 10.0, "m_b": 50}
    {"name": "swap_sa", "id": ..., "num_reads_b": 100, "num_sweeps": 300, "m_b": 50}
    {"name": "ls", "id": ..., "num_restarts": 50, "improvement": "first", "rcl_size": 3}
    {"name": "scip", "id": "scip-persp-mineig-sym0", "formulation": "perspective", "diag": "mineig",
     "usesymmetry": 0}
    {"name": "oa", "id": "oa-highs", "diag": "mineig"}

Appendix B, as implemented:

* **SA** — reads per batch ``num_reads_b`` ∈ [10, 1000]; sweeps ∈ [100, 10 000]; β range ``default``
  or ``objective`` (scaled to the objective, as the swap-SA temperature); penalty multiplier ∈
  {1, 3, 10, 30, 100}; refinements per batch ``m_b`` ∈ {20, 50, 200};
* **Tabu** — restarts per batch ``num_restarts_b`` ∈ [1, 100] (the deterministic mode of R12); tenure ∈
  [5, N/2]; penalty and ``m_b`` as for SA. ``num_reads_b`` is not in the space (the plan lists only the
  restarts per batch) and stays at 10, the default of :class:`cif.heuristics.anytime.TabuBatches`;
* **Swap-SA** — ``num_reads_b`` ∈ [10, 500]; chain length ∈ [10·N, 1000·N] proposals, i.e.
  ``num_sweeps`` ∈ [10, 1000] (a sweep is N proposals, :mod:`cif.heuristics.swap_sa`); the temperature
  is calibrated, not tuned; ``m_b`` as for SA;
* **LS** — restarts ∈ [1, 200]; first or best improvement; the neighbourhood is the one-asset swap. The
  restricted candidate list of the later restarts is the module default 3 and recorded;
* **exact** — SCIP big-M, SCIP perspective with ``D = λ_min·I`` (``mineig``), SCIP perspective with ``D``
  from the SDP (only for N ≤ 150: above, the dense PSD block does not fit in memory), each with
  ``misc/usesymmetry`` ∈ {default, 0}, and OA with the HiGHS multi-tree master. All other parameters
  are SCIP's defaults. OA takes the perspective split of ``mineig`` (the default of
  :func:`cif.classical.oa.solve_oa`; the plan lists OA as one variant).

Random search. Up to 30 configurations per heuristic and N, drawn from independent streams
``SeedSequence([tuning_seed, method, N])``, so a longer search extends a shorter one. Integer ranges that
span decades (reads, sweeps, restarts) are drawn log-uniformly, the tenure uniformly, the categorical
factors uniformly (the plan fixes the ranges, not the distributions).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = [
    "HEURISTICS",
    "MAX_CONFIGS",
    "M_B_LEVELS",
    "PENALTY_LEVELS",
    "SDP_MAX_N",
    "check_method",
    "exact_candidates",
    "sample_configs",
]

HEURISTICS = ("sa", "tabu", "swap_sa", "ls")
#: At most this many configurations per heuristic (§6).
MAX_CONFIGS = 30
PENALTY_LEVELS = (1, 3, 10, 30, 100)
M_B_LEVELS = (20, 50, 200)
#: The SDP variant of the exact class exists only up to this many assets (Appendix B).
SDP_MAX_N = 150
_SYMMETRY = ("default", 0)
_STREAM_CODE = {"sa": 1, "tabu": 2, "swap_sa": 3, "ls": 4}


# --------------------------------------------------------------------------------------
# The exact class
# --------------------------------------------------------------------------------------


def exact_candidates(n_assets: int) -> list[dict]:
    """The variants of the exact class that compete for the role of its representative at ``n_assets``:
    SCIP {big-M, perspective-mineig, perspective-SDP (N ≤ 150)} × ``usesymmetry`` {default, 0}, and OA."""
    forms: list[tuple[str, str | None]] = [("bigm", None), ("perspective", "mineig")]
    if n_assets <= SDP_MAX_N:
        forms.append(("perspective", "sdp"))
    out = []
    for formulation, diag in forms:
        for symmetry in _SYMMETRY:
            short = {"bigm": "bigm", "mineig": "persp-mineig", "sdp": "persp-sdp"}[diag or "bigm"]
            spec: dict[str, Any] = {
                "name": "scip",
                "id": f"scip-{short}-sym{'def' if symmetry == 'default' else symmetry}",
                "formulation": formulation,
            }
            if diag is not None:
                spec["diag"] = diag
            spec["usesymmetry"] = symmetry
            out.append(spec)
    out.append({"name": "oa", "id": "oa-highs", "diag": "mineig"})
    return out


# --------------------------------------------------------------------------------------
# Random search
# --------------------------------------------------------------------------------------


def _log_int(rng: np.random.Generator, low: int, high: int) -> int:
    """An integer in ``[low, high]``, uniform on the log scale."""
    x = math.exp(rng.uniform(math.log(low), math.log(high)))
    return int(min(max(round(x), low), high))


def _choice(rng: np.random.Generator, options: tuple):
    return options[int(rng.integers(len(options)))]


def _draw(name: str, rng: np.random.Generator, n_assets: int) -> dict:
    """One configuration; a fixed number of generator calls per method (prefix property)."""
    if name == "sa":
        return {
            "num_reads_b": _log_int(rng, 10, 1000),
            "num_sweeps": _log_int(rng, 100, 10_000),
            "beta": _choice(rng, ("default", "objective")),
            "penalty": float(_choice(rng, PENALTY_LEVELS)),
            "m_b": int(_choice(rng, M_B_LEVELS)),
        }
    if name == "tabu":
        return {
            "num_restarts_b": _log_int(rng, 1, 100),
            "tenure": int(rng.integers(5, max(5, n_assets // 2) + 1)),
            "num_reads_b": 10,
            "penalty": float(_choice(rng, PENALTY_LEVELS)),
            "m_b": int(_choice(rng, M_B_LEVELS)),
        }
    if name == "swap_sa":
        return {
            "num_reads_b": _log_int(rng, 10, 500),
            "num_sweeps": _log_int(rng, 10, 1000),
            "m_b": int(_choice(rng, M_B_LEVELS)),
        }
    return {  # ls
        "num_restarts": _log_int(rng, 1, 200),
        "improvement": _choice(rng, ("first", "best")),
        "rcl_size": 3,
    }


def sample_configs(name: str, n_configs: int, seed: int, n_assets: int) -> list[dict]:
    """``n_configs`` (1 … 30) random configurations of heuristic ``name`` for instances of ``n_assets``
    assets, ids ``<name>-00`` …; reproducible for ``seed`` (the tuning seed of the config), and the
    first ``k`` of a longer search are those of a search of ``k``."""
    if name not in HEURISTICS:
        raise ValueError(f"unknown method {name!r}; expected one of {HEURISTICS}")
    if not 1 <= int(n_configs) <= MAX_CONFIGS:
        raise ValueError(f"n_configs must be at least 1 and at most {MAX_CONFIGS} (§6), got {n_configs}")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _STREAM_CODE[name], int(n_assets)]))
    configs = []
    for i in range(int(n_configs)):
        spec = {"name": name, "id": f"{name}-{i:02d}", **_draw(name, rng, int(n_assets))}
        configs.append(spec)
    return configs


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _need_int(spec: dict, key: str, low: int, high: int | None = None) -> None:
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < low or (
        high is not None and value > high
    ):
        raise ValueError(f"{spec.get('id', spec.get('name'))}: {key} must be an integer in "
                         f"[{low}, {high if high is not None else '∞'}), got {value!r}")


def _need_in(spec: dict, key: str, options: tuple) -> None:
    if spec.get(key) not in options:
        raise ValueError(f"{spec.get('id', spec.get('name'))}: {key} must be one of {options}, "
                         f"got {spec.get(key)!r}")


def check_method(spec: dict) -> dict:
    """``spec`` unchanged if it is a complete, valid method spec; ``ValueError`` naming the first
    problem otherwise. Extra keys are allowed (they are carried into the row for the record)."""
    name = spec.get("name")
    if name is None:
        raise ValueError(f"a method spec needs a name, got {spec}")
    if name not in (*HEURISTICS, "scip", "oa"):
        raise ValueError(f"unknown method {name!r}")
    if not isinstance(spec.get("id"), str) or not spec["id"]:
        raise ValueError(f"{name}: a method spec needs an id (a non-empty string)")
    if name == "sa":
        _need_int(spec, "num_reads_b", 1)
        _need_int(spec, "num_sweeps", 1)
        _need_in(spec, "beta", ("default", "objective"))
    elif name == "tabu":
        _need_int(spec, "num_restarts_b", 0)
        _need_int(spec, "tenure", 0)
        _need_int(spec, "num_reads_b", 1)
    elif name == "swap_sa":
        _need_int(spec, "num_reads_b", 1)
        _need_int(spec, "num_sweeps", 1)
    elif name == "ls":
        _need_int(spec, "num_restarts", 1)
        _need_in(spec, "improvement", ("first", "best"))
        _need_int(spec, "rcl_size", 1)
    elif name == "scip":
        _need_in(spec, "formulation", ("bigm", "perspective"))
        if spec["formulation"] == "perspective":
            _need_in(spec, "diag", ("mineig", "sdp"))
        _need_in(spec, "usesymmetry", _SYMMETRY)
        if "heuristics" in spec:  # SCIP's meta setting of the primal heuristics (§12, 2026-10-19); absent = defaults
            _need_in(spec, "heuristics", ("default", "aggressive"))
    else:  # oa
        _need_in(spec, "diag", ("mineig", "sdp"))
    if name in ("sa", "tabu") and not (
        isinstance(spec.get("penalty"), int | float) and spec["penalty"] > 0
    ):
        raise ValueError(f"{spec['id']}: penalty must be a positive number, got {spec.get('penalty')!r}")
    if name in ("sa", "tabu", "swap_sa"):
        _need_int(spec, "m_b", 1)
    return spec
