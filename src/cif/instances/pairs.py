"""Paired factor instances for RQ1 / H1a (ANALYSIS_PLAN §2 "Ячейки RQ1", §3).

H1a asks whether the ceiling of the equal-weight surrogate rises with the dispersion of the
volatilities, ``h = vol_dispersion`` of :func:`cif.instances.factor.factor_instance`. The design is
paired: for one seed the instances of the two levels ``h ∈ {0.15, 0.45}`` share the sector labels,
the loadings, the idiosyncratic shapes, the standardised log-volatility draws and the noise of μ,
hence the same correlation matrix. They differ in the volatilities (``vol_i = 0.28·exp(h·z_i)``,
clipped to the generator's range) and, through the beta against the equal-weighted portfolio, in μ.
Everything else is the generator default, with ``λ = 5``, ``l = 0``, ``u = 1``.

The cardinality is common to the pair, so that the comparison of the two levels is not confounded by
K, and is set from the *smaller* continuous support so that both halves tend to bind::

    K = min(8, max(3, ⌊min_h s_cont(h) / 2⌋))

The cap keeps ``C(24, 8) ≈ 7·10⁵``, the largest enumeration of the oracle
(:mod:`cif.heuristics.oracle`), in reach. The floor 3 (plan v2.4) removes the degenerate pairs with
K = 2, where the oracle's pools cover most of the K-subsets (``T_200`` holds 72 % of ``C(24, 2)``)
and ``E_form(m ≥ 50)`` is zero trivially. Pairs are never replaced when a half does not bind
(``s_cont ≤ K``, possible only for ``s_cont ≤ 3``): the share of binding instances, the binding depth
``s_cont / K`` and the realised coefficient of variation of the volatilities are reported for each
level instead. Seeds are explicit: calibration 1000–1999, test 2000–2999 (ANALYSIS_PLAN §3).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from cif.instances.diagnostics import binding_diagnostics
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem

__all__ = ["PAIR_H", "PAIR_LAM", "FactorPair", "PairMember", "factor_pair", "pair_cardinality"]

#: The two levels of the volatility dispersion in the RQ1 cells.
PAIR_H = (0.15, 0.45)
#: Risk aversion of the RQ1 cells.
PAIR_LAM = 5.0


def pair_cardinality(s_conts: Iterable[int], *, k_min: int = 3, k_max: int = 8) -> int:
    """``min(k_max, max(k_min, ⌊min(s_conts) / 2⌋))``: the K a pair (or any group of levels) shares."""
    smallest = min(int(s) for s in s_conts)  # ValueError on an empty input
    return min(int(k_max), max(int(k_min), smallest // 2))


@dataclass(frozen=True, eq=False)
class PairMember:
    """One instance of a pair: the problem (with the pair's K), its diagnostics and the draw.

    ``depth = s_cont / K`` (binding depth), ``cv_vol`` the realised coefficient of variation of the
    volatilities ``√diag Σ``; ``stats`` is the row of
    :func:`cif.instances.diagnostics.binding_diagnostics` and ``meta`` the output of the generator
    (it holds arrays; do not serialise it).
    """

    h: float
    problem: PortfolioProblem
    s_cont: int
    binding: bool
    depth: float
    cv_vol: float
    stats: dict
    meta: dict


@dataclass(frozen=True, eq=False)
class FactorPair:
    """Instances of one seed at the levels ``h``, with the common cardinality ``k``."""

    n: int
    seed: int
    lam: float
    k: int
    members: tuple[PairMember, ...]

    def member(self, h: float) -> PairMember:
        for member in self.members:
            if member.h == h:
                return member
        raise KeyError(f"no member with h={h}; levels are {[m.h for m in self.members]}")

    def rows(self) -> list[dict]:
        """JSON-safe diagnostics, one dict per member in the order of the levels."""
        s_min = min(m.s_cont for m in self.members)
        return [
            {
                **m.stats,
                "family": "factor",
                "seed": self.seed,
                "h": m.h,
                "lam": self.lam,
                "w_min": m.problem.w_min,
                "w_max": m.problem.w_max,
                "K": self.k,
                "s_cont_min_pair": s_min,
                "binding": m.binding,
                "depth": m.depth,
                "mean_corr": m.meta["mean_corr"],
            }
            for m in self.members
        ]


def factor_pair(
    n: int,
    seed: int,
    *,
    h_levels: Sequence[float] = PAIR_H,
    lam: float = PAIR_LAM,
    w_min: float = 0.0,
    w_max: float = 1.0,
    k_min: int = 3,
    k_max: int = 8,
    **generator: object,
) -> FactorPair:
    """The instances of ``seed`` at every level of ``h_levels`` with their common K.

    ``generator`` options are passed to :func:`cif.instances.factor.factor_instance` (the
    dispersion itself belongs to ``h_levels``). ``s_cont`` is the support of the continuous
    relaxation of each level (:func:`cif.instances.diagnostics.binding_diagnostics`); K follows
    :func:`pair_cardinality` and is set on every problem.
    """
    drawn = []
    for h in h_levels:
        mu, sigma, meta = factor_instance(n, seed, vol_dispersion=h, **generator)
        problem = PortfolioProblem(
            mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)),
            risk_aversion=lam, w_min=w_min, w_max=w_max,
        )
        drawn.append((float(h), problem, binding_diagnostics(problem), meta))
    k = pair_cardinality((stats["s_cont"] for _, _, stats, _ in drawn), k_min=k_min, k_max=k_max)
    members = []
    for h, problem, stats, meta in drawn:
        s_cont = int(stats["s_cont"])
        vol = np.sqrt(np.diag(problem.sigma))
        members.append(
            PairMember(
                h=h,
                problem=replace(problem, cardinality=k),
                s_cont=s_cont,
                binding=bool(s_cont > k),
                depth=s_cont / k,
                cv_vol=float(vol.std() / vol.mean()),
                stats={**stats, "K": k, "binding": bool(s_cont > k)},
                meta=meta,
            )
        )
    return FactorPair(n=int(n), seed=int(seed), lam=float(lam), k=k, members=tuple(members))
