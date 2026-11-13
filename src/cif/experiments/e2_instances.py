"""Instances of the E2 cells and their cardinality rules (ANALYSIS_PLAN §3 "Ячейки RQ2").

======  =========  ==========  ====================================================  ===========
cell    family     N           K                                                      card_mode
======  =========  ==========  ====================================================  ===========
C1–C3   factor     100/200/400 ``⌊s_cont/2⌋``                                         le (Σz ≤ K)
C4      boot       200         ``⌊s_cont/2⌋``, λ from the pilot rule (``u = 1``)      le
C5      orlib      31–225      10, ε = 0.01, δ = 1, λ = ``orlib_lambda_grid`` point   eq (Σz = K)
D1      dense      100/200     ``⌈N/4⌉`` (legacy generator, descriptive)              le
======  =========  ==========  ====================================================  ===========

``factor`` and ``dense`` have ``λ = 5``, ``l = 0``, ``u = 1``. :func:`build_instance` takes the instance
part of a cell (``family`` and ``n``/``seed`` or ``port``/``lam_index``) and the fixed parameters of
the config (``lam``, ``fraction``, ``boot_*``, ``orlib_*``) and returns an :class:`Instance`: the problem
with its K, the mode of the cardinality constraint and the diagnostics every row carries (``s_cont``,
``binding``, the condition number of Σ, the coefficients of variation of the volatilities and of μ).

The instance and the K rule depend on the data and on the solution of the continuous relaxation, so
they are **not** part of a cell key: a cell names the instance by its seed or by (port, λ index), and
the K, ``binding`` and λ are results of the build, recorded in the row. Builds are cached per process
(the cells of one instance run one after the other).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from cif.data.sp500_pit import SP500PIT, load_dataset
from cif.experiments import registry
from cif.instances.boot import boot_instance
from cif.instances.diagnostics import (
    binding_diagnostics,
    k_from_scont,
    legacy_dense_instance,
)
from cif.instances.factor import factor_instance
from cif.instances.orlib import (
    ORLIB_N,
    orlib_lambda_grid,
    orlib_problem,
    risk_aversion_from_lambda,
)
from cif.problem import PortfolioProblem

__all__ = ["FAMILIES", "Instance", "build_instance", "clear_caches", "load_panel", "orlib_grid"]

FAMILIES = ("factor", "boot", "orlib", "dense")
#: Diagnostics copied from :func:`cif.instances.diagnostics.binding_diagnostics` into every row.
_STAT_KEYS = ("s_cont", "s_cont_frac", "binding", "f_cont", "cond_sigma", "cv_vol", "cv_mu",
              "corr_mean", "vol_median", "mu_mean", "cont_status")


@dataclass(frozen=True, eq=False)
class Instance:
    """One instance with the K of its cell.

    ``problem`` carries ``cardinality = k``; ``card_mode`` is ``"le"`` (Σz ≤ K, ``l = 0``) or ``"eq"``
    (Σz = K, OR-Library); ``s_cont`` the support of the continuous relaxation and ``binding = s_cont >
    K``; ``stats`` the JSON-safe diagnostics of the row (including ``K``); ``meta`` a JSON-safe
    description of the draw (no arrays).
    """

    problem: PortfolioProblem
    k: int
    card_mode: str
    s_cont: int
    binding: bool
    stats: dict
    meta: dict
    instance_id: str


# --------------------------------------------------------------------------------------
# Data behind the families
# --------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def load_panel(path: str) -> SP500PIT:
    """The point-in-time S&P 500 panel of C4, loaded once per process."""
    return load_dataset(Path(path))


@functools.lru_cache(maxsize=16)
def orlib_grid(port: int, data_dir: str, m: int) -> tuple[float, ...]:
    """The ``corner`` λ grid of ``port<k>`` (``m`` points), ascending."""
    return tuple(float(x) for x in orlib_lambda_grid(port, m, "corner", data_dir=data_dir))


def clear_caches() -> None:
    """Forget the cached builds, panels and grids (tests; a process that changes configs)."""
    _build.cache_clear()
    load_panel.cache_clear()
    orlib_grid.cache_clear()


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------


def _need(fixed: Mapping, key: str, family: str):
    if fixed.get(key) is None:
        raise ValueError(f"family {family!r} needs params.fixed.{key}")
    return fixed[key]


def _data_dir(fixed: Mapping, key: str, family: str) -> str:
    """A data directory of the config: a relative path is the repository's, whatever the working
    directory of the job (the runner resolves ``inputs`` the same way)."""
    path = Path(str(_need(fixed, key, family)))
    return str(path if path.is_absolute() else registry.REPO_ROOT / path)


def _names(n: int) -> tuple[str, ...]:
    return tuple(f"a{i}" for i in range(n))


def _stats(diag: Mapping, k: int, binding: bool) -> dict:
    out = {key: diag.get(key) for key in _STAT_KEYS}
    out.update(K=int(k), binding=bool(binding), s_cont=int(diag["s_cont"]))
    return out


def _finish(problem: PortfolioProblem, diag: Mapping, k: int, card_mode: str, meta: dict,
            instance_id: str) -> Instance:
    s_cont = int(diag["s_cont"])
    binding = bool(s_cont > k)
    return Instance(replace(problem, cardinality=int(k)), int(k), card_mode, s_cont, binding,
                    _stats(diag, k, binding), meta, instance_id)


def _fraction_instance(
    family: str, mu, sigma, lam: float, fraction: float, meta: dict, instance_id: str, n: int
) -> Instance:
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=_names(n), risk_aversion=float(lam),
                               w_min=0.0, w_max=1.0)
    diag = binding_diagnostics(problem)
    k = k_from_scont(problem, float(fraction), s_cont=diag["s_cont"])
    return _finish(problem, diag, k, "le", meta, instance_id)


@functools.lru_cache(maxsize=48)
def _build(key: tuple) -> Instance:
    spec, fixed_items = key
    spec, fixed = dict(spec), dict(fixed_items)
    family = spec["family"]
    if family == "factor":
        n, seed = int(spec["n"]), int(spec["seed"])
        mu, sigma, gen = factor_instance(n, seed)
        lam, fraction = float(_need(fixed, "lam", family)), float(_need(fixed, "fraction", family))
        meta = {"family": family, "seed": seed, "n": n, "lam": lam, "fraction": fraction,
                "mean_corr": gen["mean_corr"], "vol_median": gen["vol_median"]}
        return _fraction_instance(family, mu, sigma, lam, fraction, meta, f"factor/{n}/{seed}", n)
    if family == "dense":
        n, seed = int(spec["n"]), int(spec["seed"])
        mu, sigma = legacy_dense_instance(n, seed)
        lam = float(_need(fixed, "lam", family))
        problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=_names(n), risk_aversion=lam,
                                   w_min=0.0, w_max=1.0)
        return _finish(problem, binding_diagnostics(problem), math.ceil(n / 4), "le",
                       {"family": family, "seed": seed, "n": n, "lam": lam}, f"dense/{n}/{seed}")
    if family == "boot":
        n, seed = int(spec["n"]), int(spec["seed"])
        data_dir = _data_dir(fixed, "boot_data_dir", family)
        lam = float(_need(fixed, "boot_lam", family))
        fraction = float(_need(fixed, "fraction", family))
        mu, sigma, gen = boot_instance(load_panel(data_dir), seed, n=int(fixed.get("boot_n", n)))
        meta = {k: v for k, v in gen.items() if k not in ("tickers", "rows")}
        meta.update(lam=lam, fraction=fraction)
        return _fraction_instance(family, mu, sigma, lam, fraction, meta, f"boot/{n}/{seed}", n)
    if family == "orlib":
        port, index = int(spec["port"]), int(spec["lam_index"])
        data_dir = _data_dir(fixed, "orlib_dir", family)
        k = int(_need(fixed, "orlib_k", family))
        grid = orlib_grid(port, data_dir, int(_need(fixed, "orlib_m", family)))
        if not 0 <= index < len(grid):
            raise ValueError(f"lam_index must lie in [0, {len(grid)}), got {index}")
        lam = grid[index]
        problem = orlib_problem(port, k, lam, float(_need(fixed, "orlib_w_min", family)),
                                float(_need(fixed, "orlib_w_max", family)), data_dir=data_dir)
        meta = {"family": family, "port": port, "lam_index": index, "lam": lam,
                "risk_aversion": risk_aversion_from_lambda(lam), "n": ORLIB_N[port]}
        return _finish(problem, binding_diagnostics(problem, k), k, "eq", meta,
                       f"orlib/{port}/{index:02d}")
    raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")


def build_instance(spec: Mapping, fixed: Mapping) -> Instance:
    """The instance named by ``spec`` (the instance coordinates of a cell) under ``fixed`` (the fixed
    parameters of the config); cached per process. See the module docstring."""
    key = (
        tuple(sorted((k, v) for k, v in spec.items() if k in ("family", "n", "seed", "port", "lam_index"))),
        tuple(sorted((k, v) for k, v in fixed.items() if isinstance(v, str | int | float | bool))),
    )
    if spec.get("family") not in FAMILIES:
        raise ValueError(f"unknown family {spec.get('family')!r}; expected one of {FAMILIES}")
    return _build(key)
