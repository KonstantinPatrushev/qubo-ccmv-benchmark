"""Bootstrap instances on S&P 500 dates: cell C4 of RQ2 (ANALYSIS_PLAN §3 "База boot (N12)").

An instance with seed ``s``::

    d  = a month end drawn uniformly from the 156 month ends of 2013-2025 (last trading day on or
         before it is the formation date);
    U  = the top ``n`` (200) index members on d by median dollar volume, no look-ahead
         (:meth:`cif.data.sp500_pit.SP500PIT.select`);
    window = the 756 trading days up to and including d, as in :func:`cif.backtest.walkforward.
         run_walkforward` (forward-filled gaps, leading rows of late-listed series dropped, hence up
         to 755 log returns);
    μ, Σ = annualised sample mean and Ledoit-Wolf covariance (:mod:`cif.data.statistics`) of the
         window with its days resampled with replacement.

The date and the resampling use independent streams spawned from ``SeedSequence(seed)``, so the date
does not depend on the window length. ``u = 1``, ``l = 0``; λ is chosen by the C4 rule (below), K as
a fraction of ``s_cont`` by :func:`cif.instances.diagnostics.k_from_scont`. Seeds: calibration
1000-1999 (C4 rule: 1100-1109), test 2000-2999.

The λ rule. The bootstrap Σ is dense and well conditioned, so the continuous optimum can be
concentrated on a handful of names and the cardinality constraint then does not bind (REVIEW R4).
C4 therefore takes the smallest ``λ ∈ {5, 10, 20}`` whose median ``s_cont`` over the 10 calibration
instances is at least 20 (so that ``K = ⌊s_cont/2⌋ ≥ 10``). If there is none, C4 is descriptive and
stays out of the pooled estimate. :func:`calibrate_boot_lambda` applies the rule to any instance
source; it is deliberately not bound to the real panel.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cif.data.sp500_pit import SP500PIT
from cif.data.statistics import annualised_mu, annualised_sigma, log_returns
from cif.instances.diagnostics import binding_diagnostics
from cif.problem import PortfolioProblem

__all__ = [
    "BOOT_CALIBRATION_SEEDS",
    "BOOT_LAMBDAS",
    "BOOT_MIN_MEDIAN_SCONT",
    "BootLambda",
    "boot_dates",
    "boot_instance",
    "calibrate_boot_lambda",
    "month_ends",
    "select_boot_lambda",
]

FIRST_YEAR, LAST_YEAR = 2013, 2025
TRAIN_DAYS = 756
#: Candidates of the C4 rule, in increasing order.
BOOT_LAMBDAS = (5.0, 10.0, 20.0)
#: Threshold on the median ``s_cont`` of the calibration instances.
BOOT_MIN_MEDIAN_SCONT = 20
#: Calibration instances of C4 (ANALYSIS_PLAN §3).
BOOT_CALIBRATION_SEEDS = tuple(range(1100, 1110))


# --------------------------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------------------------


def month_ends(first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> pd.DatetimeIndex:
    """Calendar month ends from January of ``first_year`` to December of ``last_year``."""
    return pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="ME")


def boot_dates(
    data: SP500PIT, seed: int, *, first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``(month_end, formation_date)`` of the instance ``seed``.

    The month end is drawn uniformly from :func:`month_ends`; the formation date is the last
    trading day of the panel on or before it.
    """
    ends = month_ends(first_year, last_year)
    date_stream, _ = np.random.SeedSequence(int(seed)).spawn(2)
    month_end = ends[int(np.random.default_rng(date_stream).integers(len(ends)))]
    index = data.prices_adj.index
    trading = index[index <= month_end]
    if trading.empty:
        raise ValueError(f"the panel has no trading day on or before {month_end.date()}")
    return month_end, trading[-1]


def boot_instance(
    data: SP500PIT,
    seed: int,
    *,
    n: int = 200,
    train_days: int = TRAIN_DAYS,
    first_year: int = FIRST_YEAR,
    last_year: int = LAST_YEAR,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Draw ``(μ, Σ, meta)`` of the bootstrap instance ``seed`` (annualised units).

    ``meta`` holds ``family, seed, n, month_end, formation_date, train_start, train_end,
    train_days, n_returns`` (ISO dates and scalars), the ``tickers`` in order of liquidity and
    ``rows``, the resampled row positions of the window's returns (an array: drop it before
    serialising). Fewer than ``n`` qualifying tickers is an error rather than a smaller instance.
    """
    month_end, formation = boot_dates(data, seed, first_year=first_year, last_year=last_year)
    tickers = data.select(formation, n=n, lookback=train_days)
    if len(tickers) < n:
        raise ValueError(
            f"only {len(tickers)} of the requested {n} tickers qualify on {formation.date()}"
        )
    window = data.prices_adj.loc[:formation, tickers].ffill().iloc[-train_days:]
    returns = log_returns(window)
    if len(returns) < 2:
        raise ValueError(f"fewer than 2 complete return rows in the window ending {formation.date()}")
    _, boot_stream = np.random.SeedSequence(int(seed)).spawn(2)
    rows = np.random.default_rng(boot_stream).integers(0, len(returns), size=len(returns))
    sample = returns.iloc[rows]
    mu = annualised_mu(sample).to_numpy()
    sigma = annualised_sigma(sample, method="ledoit_wolf").to_numpy()
    meta = {
        "family": "boot",
        "seed": int(seed),
        "n": len(tickers),
        "month_end": month_end.date().isoformat(),
        "formation_date": formation.date().isoformat(),
        "train_start": window.index[0].date().isoformat(),
        "train_end": window.index[-1].date().isoformat(),
        "train_days": int(train_days),
        "n_returns": len(returns),
        "tickers": list(tickers),
        "rows": rows,
    }
    return mu, sigma, meta


# --------------------------------------------------------------------------------------------
# The lambda rule of C4
# --------------------------------------------------------------------------------------------


def select_boot_lambda(
    s_cont_by_lambda: Mapping[float, Sequence[int]],
    min_median: float = BOOT_MIN_MEDIAN_SCONT,
) -> float | None:
    """Smallest λ whose median ``s_cont`` over the calibration instances is ``>= min_median``.

    ``None`` if no λ qualifies (C4 then becomes descriptive). The median, not the mean, decides.
    """
    if not s_cont_by_lambda or any(len(v) == 0 for v in s_cont_by_lambda.values()):
        raise ValueError("every λ needs at least one calibration instance")
    for lam in sorted(s_cont_by_lambda):
        if float(np.median(s_cont_by_lambda[lam])) >= min_median:
            return float(lam)
    return None


@dataclass(frozen=True)
class BootLambda:
    """Outcome of the C4 rule, ready for the deviation log (ANALYSIS_PLAN §12).

    ``lam`` is the chosen λ or ``None``; ``s_cont`` the supports of the continuous relaxation by λ
    (one entry per calibration seed, in the order of ``seeds``), ``median_s_cont`` their medians.
    """

    lam: float | None
    seeds: tuple[int, ...]
    lambdas: tuple[float, ...]
    threshold: int
    s_cont: dict[float, list[int]]
    median_s_cont: dict[float, float]

    def as_dict(self) -> dict:
        return {
            "lam": self.lam,
            "threshold": self.threshold,
            "seeds": list(self.seeds),
            "lambdas": list(self.lambdas),
            "s_cont": {str(lam): v for lam, v in self.s_cont.items()},
            "median_s_cont": {str(lam): v for lam, v in self.median_s_cont.items()},
        }


def calibrate_boot_lambda(
    draw: Callable[[int], Sequence],
    seeds: Iterable[int] = BOOT_CALIBRATION_SEEDS,
    lambdas: Iterable[float] = BOOT_LAMBDAS,
    min_median_scont: float = BOOT_MIN_MEDIAN_SCONT,
) -> BootLambda:
    """Apply the C4 rule: ``draw(seed)`` gives ``(μ, Σ, …)``; each instance is drawn once.

    For every λ the problem has ``l = 0``, ``u = 1`` and no cardinality; ``s_cont`` is the support
    of its continuous relaxation (:func:`cif.instances.diagnostics.binding_diagnostics`). With the
    real panel, ``draw = lambda s: boot_instance(data, s)``; the plan runs it once, on the 10
    calibration seeds, before the first confirmatory run.
    """
    seeds = tuple(int(s) for s in seeds)
    lambdas = tuple(float(lam) for lam in lambdas)
    s_cont: dict[float, list[int]] = {lam: [] for lam in lambdas}
    for seed in seeds:
        mu, sigma = draw(seed)[:2]
        names = tuple(f"a{i}" for i in range(len(mu)))
        for lam in lambdas:
            problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=lam,
                                       w_min=0.0, w_max=1.0)
            s_cont[lam].append(int(binding_diagnostics(problem)["s_cont"]))
    medians = {lam: float(np.median(v)) for lam, v in s_cont.items()}
    return BootLambda(
        lam=select_boot_lambda(s_cont, min_median_scont),
        seeds=seeds,
        lambdas=lambdas,
        threshold=int(min_median_scont),
        s_cont=s_cont,
        median_s_cont=medians,
    )
