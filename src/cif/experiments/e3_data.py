"""E3 data layer: markets, periods, the per-fold estimation table and strategy paths (RQ3).

ANALYSIS_PLAN §3 ("Рыночные данные (E3)", "Walk-forward"). Every stage of E3 (the cells, the
assembly of the paths, the pilot) needs the SAME folds: the same training windows, universes and
estimates ``(μ, Σ)``. They are not recomputed here by a copy of the engine's code: the table of a
period is the engine itself (:func:`cif.backtest.walkforward.run_walkforward`) run once with a
recording strategy, and every later path is a pass of the same engine over the same rows with the
same settings (:func:`run_path`). What a cell optimises is therefore, by construction, what the
engine holds.

Periods. A period is an independent contiguous walk-forward: its first test window starts on the
first trading day on or after ``start`` and its windows are whole ``test_days`` blocks that end on
or before ``end`` (§3: "тестовые окна начинаются не раньше …, кончаются не позже …"). The
training windows may reach back before ``start`` (they read the past only). The engine is run on
the rows ``[row_start, row_end)`` of the panel, so it sees nothing after the last test day.

Markets (``params.market.source``):

* ``ff49`` — Kenneth French's 49 industries, value-weighted (``data/processed/ff49``): price levels
  built from returns, rf from the monthly T-bill return spread over the trading days. An industry
  enters a fold if its training window has no gaps (:func:`ff49_select_assets`, §3).
* ``sp500_pit`` — S&P 500 on the date (``data/processed/sp500_pit``): the top ``n_universe`` members
  by median dollar volume with at least 95 % valid prices in the window
  (:func:`cif.data.sp500_pit.select_universe`), Adj Close with the ``adjcorr-2`` corrections, rf
  DTB3. ``variant: no_irregular_ratio`` drops from a fold the tickers with a remaining day of an irregular
  Yahoo ratio (the 102 days of ``pseudo_split_days`` that are not corrected; §3, sensitivity) among the
  returns of that fold's own training and test windows, and from no other fold: the universe of a fold does
  not depend on events beyond the fold (audit G1, D-3).
* ``synthetic`` — a factor-model return panel with a known covariance
  (:func:`cif.instances.factor.factor_instance`): the tests and the smoke runs. No market data.

The cardinality ``K_f = max(⌈1/u⌉ + 1, ⌊s_cont,f/2⌋)`` is a function of the fold's problem and is
computed by the stage that needs it (:func:`cardinality_rule`).
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cif.backtest.walkforward import BacktestResult, run_walkforward
from cif.data.ff49 import price_index
from cif.experiments import registry
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem

__all__ = [
    "FoldRecord",
    "FoldTable",
    "Market",
    "PeriodGrid",
    "build_fold_table",
    "cardinality_rule",
    "clear_caches",
    "ff49_select_assets",
    "floor_cardinality",
    "get_fold_table",
    "is_binding",
    "json_safe",
    "load_market",
    "lower_bound_active",
    "market_calendar",
    "period_grid",
    "run_path",
    "synthetic_market",
]

SOURCES = ("ff49", "sp500_pit", "synthetic")
TRADING_DAYS = 252


# --------------------------------------------------------------------------------------------
# The cardinality rule (§3)
# --------------------------------------------------------------------------------------------


def floor_cardinality(u: float) -> int:
    """The lower bound of the rule, ``⌈1/u⌉ + 1`` (7 would not fit ``u = 0.2``: it is 6)."""
    if not 0.0 < float(u) <= 1.0:
        raise ValueError(f"u must lie in (0, 1], got {u!r}")
    return math.ceil(1.0 / float(u) - 1e-9) + 1


def cardinality_rule(s_cont: int, u: float) -> int:
    """``K_f = max(⌈1/u⌉ + 1, ⌊s_cont,f/2⌋)`` with ``s_cont,f`` the support of the continuous
    optimum on the training window (§3)."""
    if int(s_cont) < 1:
        raise ValueError(f"s_cont must be at least 1, got {s_cont!r}")
    return max(floor_cardinality(u), int(s_cont) // 2)


def is_binding(s_cont: int, u: float) -> bool:
    """The problem binds iff ``s_cont,f > ⌈1/u⌉ + 1`` (§3); then ``K_f < s_cont,f``."""
    return int(s_cont) > floor_cardinality(u)


def lower_bound_active(s_cont: int, u: float) -> bool:
    """The floor ``K = ⌈1/u⌉ + 1`` acts (``K·u ≈ 1``: the weights are nearly fixed, §3)."""
    return cardinality_rule(s_cont, u) == floor_cardinality(u)


# --------------------------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodGrid:
    """The rows of a period: ``[row_start, row_end)`` of the panel hold the training window of
    fold 0 up to the last test day of the last fold; ``first_test_row`` is the first test day."""

    row_start: int
    first_test_row: int
    row_end: int
    n_folds: int
    train_days: int
    test_days: int

    def formation_row(self, fold: int) -> int:
        """Panel row of the formation date (the close of the last training day) of ``fold``."""
        return self.first_test_row - 1 + fold * self.test_days

    def test_rows(self, fold: int) -> tuple[int, int]:
        """``(first, last)`` panel rows of the test window of ``fold``."""
        first = self.first_test_row + fold * self.test_days
        return first, first + self.test_days - 1


def period_grid(
    index: pd.DatetimeIndex, start, end, train_days: int, test_days: int
) -> PeriodGrid:
    """The folds of the period ``[start, end]`` on the trading calendar ``index`` (module
    docstring): whole test windows from the first trading day ``≥ start`` that end on or before
    ``end``; the training window of the first fold needs ``train_days`` rows before it."""
    idx = pd.DatetimeIndex(index)
    first_test = int(idx.searchsorted(pd.Timestamp(start), side="left"))
    if first_test - train_days < 0:
        raise ValueError(
            f"not enough history: {first_test} rows before {pd.Timestamp(start).date()}, "
            f"the training window needs {train_days}"
        )
    last = int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    n_folds = (last + 1 - first_test) // int(test_days)
    if n_folds < 1:
        raise ValueError(
            f"no complete test window of {test_days} days between {pd.Timestamp(start).date()} "
            f"and {pd.Timestamp(end).date()}"
        )
    return PeriodGrid(
        row_start=first_test - int(train_days),
        first_test_row=first_test,
        row_end=first_test + n_folds * int(test_days),
        n_folds=int(n_folds),
        train_days=int(train_days),
        test_days=int(test_days),
    )


# --------------------------------------------------------------------------------------------
# Markets
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Market:
    """A price panel with everything the engine needs, and what the stages report about it.

    ``prices`` are level series (NaN where a series has no price); ``rf`` the daily risk-free
    accruals on their own calendar; ``select_assets(formation_date)`` the fold's universe (a pure
    function of the past, memoised); ``fold_extra(formation_date)`` optional per-fold facts for the
    rows (S&P: the coverage of the index members by data); ``meta`` JSON-safe facts about the data
    (and, for the synthetic market, the arrays of the generating model).
    """

    name: str
    source: str
    prices: pd.DataFrame
    rf: pd.Series | None
    select_assets: Callable[[pd.Timestamp], list[str]]
    u: float
    train_days: int
    test_days: int
    meta: dict = field(default_factory=dict)
    fold_extra: Callable[[pd.Timestamp], dict] | None = None


def ff49_select_assets(prices: pd.DataFrame, train_days: int) -> Callable[[pd.Timestamp], list[str]]:
    """The FF49 fold universe: the industries with no missing price in the ``train_days`` rows that
    end on the formation date (§3: "Отрасль входит в фолд, если в окне обучения у неё нет
    пропусков"). Only rows up to the formation date are read. The answer is memoised; every call
    returns a fresh list."""
    index = pd.DatetimeIndex(prices.index)
    valid = prices.notna().to_numpy().astype(np.int64)
    cumulative = np.vstack([np.zeros((1, valid.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0)])
    columns = [str(c) for c in prices.columns]
    memo: dict[pd.Timestamp, tuple[str, ...]] = {}

    def select(formation_date) -> list[str]:
        date = pd.Timestamp(formation_date)
        if date not in memo:
            pos = int(index.get_indexer([date])[0])
            if pos < 0:
                raise ValueError(f"{date.date()} is not a trading day of the panel")
            if pos + 1 < train_days:
                raise ValueError(
                    f"{date.date()} has only {pos + 1} rows of history, the window needs {train_days}"
                )
            count = cumulative[pos + 1] - cumulative[pos + 1 - train_days]
            memo[date] = tuple(c for c, k in zip(columns, count, strict=True) if k == train_days)
        return list(memo[date])

    return select


def _memoised(select: Callable[[pd.Timestamp], Sequence[str]]) -> Callable[[pd.Timestamp], list[str]]:
    memo: dict[pd.Timestamp, tuple[str, ...]] = {}

    def wrapped(formation_date) -> list[str]:
        date = pd.Timestamp(formation_date)
        if date not in memo:
            memo[date] = tuple(select(date))
        return list(memo[date])

    return wrapped


def synthetic_market(spec: dict) -> Market:
    """A factor-model market with a known covariance (tests, smoke runs): daily returns
    ``N(μ/252, Σ/252)`` with ``(μ, Σ) = factor_instance(n_assets, seed)``, price levels from the FF49
    convention, a constant risk-free rate; ``gap_fraction`` blanks returns at random (missing days,
    as in the FF49 files) and the FF49 universe rule applies."""
    n, days, seed = int(spec["n_assets"]), int(spec["n_days"]), int(spec["seed"])
    mu, sigma, _ = factor_instance(n, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xE3]))
    chol = np.linalg.cholesky(sigma / TRADING_DAYS)
    daily = mu / TRADING_DAYS + rng.standard_normal((days, n)) @ chol.T
    idx = pd.bdate_range(spec["start"], periods=days)
    returns = pd.DataFrame(daily, index=idx, columns=[f"S{i:02d}" for i in range(n)])
    gap = float(spec.get("gap_fraction", 0.0))
    if gap > 0.0:
        returns = returns.mask(rng.random(returns.shape) < gap)
    prices = price_index(returns)
    rf_annual = float(spec.get("rf_annual", 0.02))
    rf = pd.Series((1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0, index=idx, name="rf")
    train_days = int(spec["train_days"])
    return Market(
        name=str(spec.get("name", "synthetic")),
        source="synthetic",
        prices=prices,
        rf=rf,
        select_assets=ff49_select_assets(prices, train_days),
        u=float(spec["u"]),
        train_days=train_days,
        test_days=int(spec["test_days"]),
        meta={"mu": mu, "sigma": sigma, "seed": seed, "gap_fraction": gap},
    )


def _resolve(path: str | Path, root: Path | None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root if root is not None else registry.REPO_ROOT) / p


def _ff49_market(spec: dict, root: Path | None) -> Market:
    data_dir = _resolve(spec["data_dir"], root)
    prices = pd.read_parquet(data_dir / "prices_vw.parquet").sort_index()
    rf = pd.read_parquet(data_dir / "rf_daily.parquet")["rf"].astype(float).sort_index()
    train_days = int(spec["train_days"])
    return Market(
        name=str(spec.get("name", "ff49")),
        source="ff49",
        prices=prices,
        rf=rf,
        select_assets=ff49_select_assets(prices, train_days),
        u=float(spec["u"]),
        train_days=train_days,
        test_days=int(spec["test_days"]),
        meta={"data_dir": str(spec["data_dir"]), "n_industries": int(prices.shape[1])},
    )


def _sp500_market(spec: dict, root: Path | None) -> Market:
    from cif.data.sp500_pit import load_dataset

    data_dir = _resolve(spec["data_dir"], root)
    data = load_dataset(data_dir)
    train_days, test_days = int(spec["train_days"]), int(spec["test_days"])
    n_universe = int(spec["n_universe"])
    window = {"lookback": train_days, "liquidity_window": int(spec.get("liquidity_window", 252))}
    variant = str(spec.get("variant", "base"))
    if variant not in ("base", "no_irregular_ratio"):
        raise ValueError(f"unknown S&P variant {variant!r}")
    irregular = pd.DataFrame({"ticker": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})
    if variant == "no_irregular_ratio":
        days = pd.read_parquet(data_dir / "pseudo_split_days.parquet")
        days = days.loc[~days["corrected"].astype(bool)]
        irregular = pd.DataFrame({"ticker": days["ticker"].astype(str), "date": pd.to_datetime(days["date"])})
    calendar = pd.DatetimeIndex(data.prices_adj.index).sort_values()

    def irregular_in_fold(formation_date) -> frozenset[str]:
        """The tickers with an uncorrected irregular day among the returns the fold uses: the ``train_days - 1``
        returns of the training window (its ``train_days`` prices end on the formation date) and the
        ``test_days`` returns of the test window. ``pseudo_split_days`` dates the distorted return. A day
        outside both leaves the ticker in the universe of this fold (D-3)."""
        pos = int(calendar.searchsorted(pd.Timestamp(formation_date), side="right")) - 1
        first = calendar[max(pos - train_days + 2, 0)]
        last = calendar[min(pos + test_days, len(calendar) - 1)]
        return frozenset(irregular.loc[irregular["date"].between(first, last), "ticker"])

    def pick(formation_date) -> list[str]:
        flagged = irregular_in_fold(formation_date)
        if not flagged:
            return data.select(formation_date, n=n_universe, **window)
        table = data.candidates(formation_date, **window)
        return [t for t in table.index if t not in flagged][:n_universe]

    rf_path = _resolve(spec["rf_path"], root)
    rf = pd.read_parquet(rf_path)["r_daily"].astype(float).sort_index()
    coverage = data.coverage["coverage"].astype(float)

    def extra(formation_date) -> dict:
        upto = coverage.loc[: pd.Timestamp(formation_date)]
        return {"coverage": float(upto.iloc[-1]) if len(upto) else None}

    return Market(
        name=str(spec.get("name", "us")),
        source="sp500_pit",
        prices=data.prices_adj.sort_index(),
        rf=rf,
        select_assets=_memoised(pick),
        u=float(spec["u"]),
        train_days=train_days,
        test_days=test_days,
        meta={"data_dir": str(spec["data_dir"]), "variant": variant, "n_universe": n_universe,
              "n_flagged_tickers": int(irregular["ticker"].nunique()), "n_irregular_days": len(irregular)},
        fold_extra=extra,
    )


def load_market(spec: dict, root: Path | None = None) -> Market:
    """The market of ``params.market``; relative paths are taken from the repository root."""
    source = spec.get("source")
    if source not in SOURCES:
        raise ValueError(f"market.source must be one of {SOURCES}, got {source!r}")
    for key in ("u", "train_days", "test_days"):
        if key not in spec:
            raise ValueError(f"market: missing '{key}'")
    if source == "synthetic":
        return synthetic_market(spec)
    if source == "ff49":
        return _ff49_market(spec, root)
    return _sp500_market(spec, root)


def _index_of(path: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_parquet(path, columns=[]).index).sort_values()


def market_calendar(spec: dict, root: Path | None = None) -> pd.DatetimeIndex:
    """The trading calendar of the market, without loading its prices (the grid of ``cells``)."""
    source = spec.get("source")
    if source == "synthetic":
        return pd.bdate_range(spec["start"], periods=int(spec["n_days"]))
    if source == "ff49":
        return _index_of(_resolve(spec["data_dir"], root) / "prices_vw.parquet")
    if source == "sp500_pit":
        return _index_of(_resolve(spec["data_dir"], root) / "prices_adj.parquet")
    raise ValueError(f"market.source must be one of {SOURCES}, got {source!r}")


# --------------------------------------------------------------------------------------------
# The fold table: the engine's own estimation, recorded
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldRecord:
    """One fold: its dates, universe and the estimate the engine made on the training window."""

    fold: int
    formation_date: pd.Timestamp
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    assets: tuple[str, ...]
    train_rows: int
    mu: np.ndarray
    sigma: np.ndarray
    extra: dict = field(default_factory=dict)

    def problem(self, *, lam: float, u: float, cardinality: int | None = None) -> PortfolioProblem:
        """The fold's problem (annualised ``μ``, Ledoit–Wolf ``Σ``) with ``w_min = 0``, ``w_max = u``
        and the risk aversion ``lam``; the cardinality is chosen by the caller."""
        return PortfolioProblem(
            mu=self.mu, sigma=self.sigma, asset_names=self.assets, risk_aversion=float(lam),
            w_min=0.0, w_max=float(u), cardinality=cardinality,
        )


@dataclass(frozen=True)
class FoldTable:
    """The folds of a period and the panel rows the engine passes of the period run on."""

    market: Market
    period: dict
    grid: PeriodGrid
    prices: pd.DataFrame
    folds: tuple[FoldRecord, ...]


def build_fold_table(market: Market, period: dict) -> FoldTable:
    """The engine's estimates for every fold of ``period``: one pass of
    :func:`cif.backtest.walkforward.run_walkforward` with a recording strategy (equal weights, no
    costs) on the rows of the period."""
    grid = period_grid(market.prices.index, period["start"], period["end"], market.train_days,
                       market.test_days)
    prices = market.prices.iloc[grid.row_start : grid.row_end]
    seen: dict[int, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}

    def record(problem: PortfolioProblem, fold: int) -> np.ndarray:
        seen[fold] = (problem.mu, problem.sigma, tuple(problem.asset_names))
        return np.full(problem.n, problem.budget / problem.n)

    result = run_walkforward(
        prices, record, "e3-estimation", market.name, train_days=market.train_days,
        test_days=market.test_days, step_days=market.test_days, w_min=0.0, w_max=market.u,
        transaction_cost_bps=0.0, select_assets=market.select_assets,
    )
    if len(result.folds) != grid.n_folds:
        raise RuntimeError(f"the engine ran {len(result.folds)} folds, the grid has {grid.n_folds}")
    folds = []
    for fr in result.folds:
        mu, sigma, assets = seen[fr.fold]
        extra = market.fold_extra(fr.formation_date) if market.fold_extra is not None else {}
        folds.append(
            FoldRecord(
                fold=fr.fold, formation_date=fr.formation_date, train_start=fr.train_start,
                test_start=fr.test_start, test_end=fr.test_end, assets=assets,
                train_rows=int(fr.info["train_rows"]), mu=mu, sigma=sigma, extra=extra,
            )
        )
    return FoldTable(market=market, period=dict(period), grid=grid, prices=prices,
                     folds=tuple(folds))


@functools.lru_cache(maxsize=8)
def _cached_table(market_json: str, period_json: str, root: str | None) -> FoldTable:
    market = load_market(json.loads(market_json), Path(root) if root else None)
    return build_fold_table(market, json.loads(period_json))


def get_fold_table(market_spec: dict, period: dict, root: Path | None = None) -> FoldTable:
    """:func:`build_fold_table` of the configured market and period, built once per process (a
    worker of the runner builds it for its first cell and reuses it)."""
    return _cached_table(
        json.dumps(market_spec, sort_keys=True, default=str),
        json.dumps(period, sort_keys=True, default=str),
        str(root) if root is not None else None,
    )


def clear_caches() -> None:
    """Forget the cached tables (tests; a long process that changes configs)."""
    _cached_table.cache_clear()


def json_safe(value):
    """Python natives only, ``None`` for NaN and the infinities: a row goes to JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------------------------
# Strategy paths
# --------------------------------------------------------------------------------------------


def run_path(
    table: FoldTable,
    weights_for_fold: Callable[[int, PortfolioProblem], np.ndarray],
    *,
    tc_bps: float,
    execution_lag: int = 0,
    terminal_return: float = 0.0,
    name: str = "path",
) -> BacktestResult:
    """One strategy through the period's walk-forward: the engine with the market's universe, the
    risk-free series and the cap ``u``, the cost ``tc_bps`` (10 bp main, 0 before costs, 30
    sensitivity), the execution lag and the terminal return of a delisting (§3).

    ``weights_for_fold(fold, problem)`` returns the target weights of the fold, aligned with
    ``problem.asset_names`` (the universe the table holds for it).
    """
    folds = table.folds

    def strategy(problem: PortfolioProblem, fold: int) -> np.ndarray:
        if tuple(problem.asset_names) != folds[fold].assets:
            raise ValueError(f"fold {fold}: the universe differs from the one of the fold table")
        weights = np.asarray(weights_for_fold(fold, problem), dtype=float)
        if weights.shape != (problem.n,):
            raise ValueError(
                f"weights of fold {fold}: expected shape ({problem.n},), got {weights.shape}"
            )
        return weights

    market = table.market
    return run_walkforward(
        table.prices, strategy, name, market.name, train_days=market.train_days,
        test_days=market.test_days, step_days=market.test_days, w_min=0.0, w_max=market.u,
        transaction_cost_bps=float(tc_bps), rf=market.rf, select_assets=market.select_assets,
        execution_lag=int(execution_lag), terminal_return=float(terminal_return),
    )
