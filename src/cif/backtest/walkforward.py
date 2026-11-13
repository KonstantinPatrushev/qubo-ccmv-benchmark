"""Walk-forward backtest engine for comparing portfolio strategies.

A single ``run_walkforward`` call rolls a ``(train, test)`` window through a price panel,
re-estimates ``(μ, Σ)`` each fold, asks a ``strategy`` for target weights, and marks the fund to
market DAILY over the test window. Mechanics (fixed in E0, see docs/REVIEW.md R1–R2):

* Positions are formed at the close of the last training day (the *formation date*); the first
  test-day return is the move from that close to the next close, so no day is dropped.
* Within the test window the portfolio is buy-and-hold: weights drift with prices.
* At each rebalance the pre-trade weights are the DRIFTED weights, so turnover and costs include
  the cost of returning to target weights (1/N pays for rebalancing too).
* Costs are proportional to traded notional: ``cost = tc_bps/1e4 · Σ|w_target − w_pre|``
  (``cost_convention="traded"``, as in DeMiguel, Garlappi & Uppal 2009). The legacy convention
  charged ``tc_bps`` on one-way turnover ``0.5·Σ|Δw|`` and is available as ``"one_way"``. The
  cost is deducted at the execution close, i.e. it shows up in the first day after the trade.
* If a ``tradable`` calendar is given and the formation date is a non-trading day (e.g. the MOEX
  closure of 28.02–23.03.2022), execution is postponed to the first tradable date in the window;
  until then the drifted old portfolio is held. A per-asset mask (``DataFrame``, date × asset)
  also allows partial execution: assets that do not trade on the execution day keep their drifted
  weight and the remaining budget goes to tradable assets in proportion to the target (24.03.2022
  on MOEX only 33 stocks traded; audit O-4), no weight above ``w_max``: what the cap refuses is
  spread over the other tradable assets (water-filling) and what still does not fit, or has no
  tradable asset to go to, stays in cash (``info["unallocated_weight"]``; audit W-5). A day with
  no tradable asset counts as closed.
* ``execution_lag = k`` (robustness check, audit O-9): the target decided at the formation close
  is executed at the close of the k-th TRADABLE day after it, so closed days do not count towards
  the lag (audit W-3); until then the drifted old portfolio is held. If the window has fewer than
  k tradable days the fold makes no trade. ``k = 0`` (default) trades at the formation close, or on
  the first tradable day if that is closed.
* Performance is computed from the daily net-return series (``cif.metrics.performance``);
  Sharpe, Sortino and CEQ use excess returns when a daily risk-free series ``rf`` is supplied.

Per-fold universe (point-in-time panels)
----------------------------------------
Without ``select_assets`` every column is in every fold and missing prices are an error. With
``select_assets(formation_date) -> tickers`` (which must read nothing dated after the formation
date, e.g. :func:`cif.data.sp500_pit.select_universe`) each fold's problem is built on those
tickers only and the panel may have gaps:

* the training window is forward-filled; leading rows of late-starting series are dropped
  (``info["train_rows"]``);
* ``select_assets`` may not return a ticker twice (``ValueError``; audit W-4);
* an asset without a price on a day cannot be traded that day: a holding is frozen as above, i.e.
  valued at its last price with zero return for as long as the series is missing, and the move
  across a gap is earned on the day trading resumes;
* holdings outside the new selection are sold (``info["outside_weight"]`` before the trade);
* delisting event (audit W-2). A holding that has NO PRICE ON THE FORMATION DATE and is NOT IN THE
  NEW SELECTION is turned into cash at its last price, without cost, before the strategy is called
  (``info["ended_assets"]``: ``asset, date, weight, terminal_return``). Nothing dated after that
  day is used: whether the series resumes later, or is simply never seen again, is unknowable then
  and plays no role. Inside a window nothing is converted, so a series that ends is frozen until
  the next formation date. A holding without a price that the new selection still contains stays
  frozen (it cannot be traded).
* ``terminal_return`` (default 0) is the return of the position at that conversion, the delisting
  return of Shumway (1997): −0.3 and −1.0 are the sensitivity runs of ANALYSIS_PLAN §3. The fund
  loses ``weight × terminal_return`` of its value, booked on the first day after the formation
  date like a cost of a close trade (``info["terminal_loss"]``, part of the gross return: it is a
  market loss, not a cost). A loss that wipes out the whole fund is an error.

A ``strategy`` is any callable ``fn(problem, fold_index)`` or ``fn(problem, fold_index, ctx)``
returning target weights, or ``(weights, info_dict)``. ``ctx`` (:class:`StrategyContext`)
carries the drifted pre-trade weights so turnover-aware rules see the true current portfolio.
Weights must be fully invested (sum to the budget) and are aligned with ``problem.asset_names``.
"""

from __future__ import annotations

import inspect
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from cif.data.statistics import annualised_mu, annualised_sigma, log_returns
from cif.metrics.performance import summarise_returns
from cif.problem import PortfolioProblem

StrategyFn = Callable[..., object]
CostConvention = Literal["traded", "one_way"]


@dataclass(frozen=True)
class StrategyContext:
    """Information available to a strategy at the formation date (no look-ahead)."""

    fold: int
    formation_date: pd.Timestamp
    current_weights: np.ndarray  # drifted pre-trade weights, aligned with problem assets
    transaction_cost_bps: float
    cost_convention: CostConvention
    assets: tuple[str, ...] = ()  # the problem's assets (the fold's universe)
    outside_weight: float = 0.0  # held weight in assets outside the fold's universe (to be sold)


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    formation_date: pd.Timestamp
    execution_date: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    weights: np.ndarray  # target weights after the trade
    pre_trade_weights: np.ndarray
    end_weights: np.ndarray  # drifted weights at the end of the window
    realised_return: float  # gross compounded return over the window (before costs)
    net_return: float  # compounded net return over the window
    realised_std: float
    traded: float  # Σ|Δw|
    turnover: float  # one-way turnover 0.5·Σ|Δw|
    transaction_cost: float  # fraction of fund value paid
    solve_time_s: float
    info: dict = field(default_factory=dict)
    assets: tuple[str, ...] | None = None  # the fold's universe (per-fold mode only)


@dataclass
class BacktestResult:
    strategy_name: str
    universe: str
    folds: list[FoldResult] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    daily_returns: pd.Series | None = None
    summary: dict = field(default_factory=dict)


def _make_problem(
    returns_train: pd.DataFrame,
    cardinality: int | None,
    risk_aversion: float,
    w_min: float,
    w_max: float,
) -> PortfolioProblem:
    mu = annualised_mu(returns_train).values
    sigma = annualised_sigma(returns_train, method="ledoit_wolf").values
    names = tuple(returns_train.columns)
    return PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=names,
        risk_aversion=risk_aversion,
        cardinality=cardinality,
        w_min=w_min,
        w_max=w_max,
    )


def _accepts_context(strategy: StrategyFn) -> bool:
    """True if the strategy takes a third argument for :class:`StrategyContext`.

    Counted are a parameter named ``ctx``, ``*args``, or a third REQUIRED positional parameter;
    defaulted parameters (the ``def f(problem, fold, _x=x)`` closure idiom) are not.
    """
    try:
        params = list(inspect.signature(strategy).parameters.values())
    except (TypeError, ValueError):
        return False
    if any(p.name == "ctx" for p in params) or any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    required = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is inspect.Parameter.empty
    ]
    return len(required) >= 3


def _call_strategy(strategy, accepts_ctx, problem, fold_idx, ctx) -> tuple[np.ndarray, dict]:
    out = strategy(problem, fold_idx, ctx) if accepts_ctx else strategy(problem, fold_idx)
    if isinstance(out, tuple):
        weights, info = out
    else:
        weights, info = out, {}
    return np.asarray(weights, dtype=float), dict(info or {})


_CAP_TOL = 1e-12


def _spread(base: np.ndarray, amount: float, cap: float) -> np.ndarray:
    """``amount`` spread over the entries in proportion to ``base``, none above ``cap`` (water-filling).

    Entries whose proportional share exceeds the cap are set to it and the rest is spread over the
    others in proportion to ``base`` again, until nothing exceeds the cap. What does not fit when
    every entry is at the cap is not allocated (the caller keeps it in cash).
    """
    out = np.zeros_like(base)
    active = base > 0
    left = float(amount)
    while left > _CAP_TOL and active.any():
        share = np.where(active, base, 0.0)
        share = left * share / share.sum()
        over = active & (share > cap + _CAP_TOL)
        if not over.any():
            return out + share
        out[over] = cap
        left -= cap * float(over.sum())
        active &= ~over
    return out


def _execute(
    target: np.ndarray, current: np.ndarray, frozen: np.ndarray | None, cap: float | None = None
) -> np.ndarray:
    """Post-trade weights when the assets flagged ``frozen`` cannot be traded.

    Frozen assets keep their current (drifted) weight; the remaining budget is allocated to the
    tradable assets in proportion to the target, no weight above ``cap`` (``w_max``; audit W-5): see
    :func:`_spread`. If the target puts nothing on tradable assets, their current weights are the
    proportions; if those are zero too, the remainder stays in cash. Without frozen assets the
    target is executed as it is.
    """
    if frozen is None or not frozen.any():
        return target.copy()
    w = np.zeros_like(target)
    w[frozen] = current[frozen]
    remaining = max(float(target.sum() - w.sum()), 0.0)
    free = ~frozen
    base = target[free] if target[free].sum() > 0 else current[free]
    if base.sum() > 0:
        w[free] = _spread(base, remaining, np.inf if cap is None else float(cap))
    return w


def run_walkforward(
    prices: pd.DataFrame,
    strategy: StrategyFn,
    strategy_name: str,
    universe: str,
    train_days: int = 756,  # 3 years of trading days
    test_days: int = 21,  # ~1 month
    step_days: int | None = None,
    cardinality: int | None = None,
    risk_aversion: float = 2.0,
    w_min: float = 0.0,
    w_max: float = 1.0,
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 1.0,
    rf: pd.Series | None = None,
    tradable: pd.Series | pd.DataFrame | None = None,
    cost_convention: CostConvention = "traded",
    execution_lag: int = 0,
    select_assets: Callable[[pd.Timestamp], Sequence[str]] | None = None,
    budget_tol: float = 1e-6,
    progress: bool = False,
    terminal_return: float = 0.0,
) -> BacktestResult:
    """Run one strategy through a rolling walk-forward (see module docstring for mechanics).

    Parameters
    ----------
    prices:
        Wide price panel (total-return adjusted where available), date-indexed, one column per
        ticker; no missing values inside the evaluated range unless ``select_assets`` is given.
    strategy:
        ``fn(problem, fold_index[, ctx]) -> weights | (weights, info)``.
    train_days, test_days, step_days:
        Window sizes in trading days; ``step_days`` defaults to ``test_days`` (contiguous test
        windows).
    transaction_cost_bps:
        Proportional cost in basis points (see ``cost_convention``).
    rf:
        Optional daily risk-free return series (simple returns per trading day, already
        aligned to information available on that day). Used for excess-return metrics.
    tradable:
        Optional boolean ``Series`` (``False`` = market closed that day) or ``DataFrame`` (date ×
        asset; ``False`` = the asset does not trade that day, see the module docstring).
    cost_convention:
        ``"traded"`` (default): cost on ``Σ|Δw|``; ``"one_way"``: legacy, cost on ``0.5·Σ|Δw|``.
    execution_lag:
        Tradable days between the decision (formation close) and the trade, ``0 ≤ k < test_days``.
    select_assets:
        ``fn(formation_date) -> tickers``: the fold's universe, computed without look-ahead and
        without repeated tickers (see "Per-fold universe" in the module docstring).
    terminal_return:
        Return of a position at its conversion to cash in the delisting event of the per-fold
        mode, in ``[-1, ∞)``; 0 keeps the value at the last price (module docstring).
    """
    if step_days is None:
        step_days = test_days
    if cost_convention not in ("traded", "one_way"):
        raise ValueError(f"unknown cost_convention {cost_convention!r}")
    if not 0 <= execution_lag < test_days:
        raise ValueError(f"execution_lag must be in [0, test_days={test_days}), got {execution_lag}")
    if not (math.isfinite(terminal_return) and terminal_return >= -1.0):
        raise ValueError(f"terminal_return must be finite and >= -1, got {terminal_return}")

    prices = prices.sort_index()
    dates = prices.index
    if len(dates) < train_days + test_days:
        raise ValueError(f"Need at least {train_days + test_days} days, got {len(dates)}")
    n_folds_planned = (len(dates) - train_days - test_days) // step_days + 1
    evaluated = prices.iloc[: (n_folds_planned - 1) * step_days + train_days + test_days]
    per_fold = select_assets is not None
    if not per_fold and evaluated.isna().any().any():
        # Never truncate silently (audit W-1): log_returns drops incomplete rows and an empty
        # training window used to end the backtest. Point-in-time panels need a per-fold universe.
        bad = evaluated.columns[evaluated.isna().any()].tolist()
        first = evaluated[bad].isna().idxmax().min()
        raise ValueError(
            f"prices contain missing values in the evaluated range (first on {first.date()}) "
            f"for {len(bad)} column(s): {bad[:10]}{' …' if len(bad) > 10 else ''}"
        )
    # With gaps (per-fold mode) returns come from the forward-filled panel: zero while a series
    # is missing, the whole move on the day it resumes.
    filled = prices.ffill() if per_fold else prices
    asset_returns = filled.pct_change(fill_method=None)
    asset_mask: pd.DataFrame | None = None
    tradable_mask: pd.Series | None = None
    if isinstance(tradable, pd.DataFrame):
        asset_mask = tradable.reindex(index=dates, columns=prices.columns).fillna(True).astype(bool)
        tradable_mask = asset_mask.any(axis=1)
    elif tradable is not None:
        tradable_mask = tradable.reindex(dates).fillna(True).astype(bool)
    col_pos = {c: j for j, c in enumerate(prices.columns)}
    has_price: np.ndarray | None = None
    if per_fold:
        # no price that day = not tradable that day
        fresh = prices.notna()
        asset_mask = fresh if asset_mask is None else asset_mask & fresh
        if tradable_mask is not None and not isinstance(tradable, pd.DataFrame):
            asset_mask = asset_mask & tradable_mask.to_numpy()[:, None]
        tradable_mask = asset_mask.any(axis=1)
        has_price = fresh.to_numpy()

    def frozen_on(day) -> np.ndarray | None:
        return None if asset_mask is None else ~asset_mask.loc[day].to_numpy()
    accepts_ctx = _accepts_context(strategy)
    tc = transaction_cost_bps / 10000.0
    cost_factor = tc if cost_convention == "traded" else 0.5 * tc

    n_assets = prices.shape[1]
    current = np.zeros(n_assets, dtype=float)  # drifted weights of the fund (0 = all cash)
    value = float(initial_capital)
    day_dates: list[pd.Timestamp] = []
    day_returns: list[float] = []
    folds: list[FoldResult] = []

    def delist_at_formation(
        w: np.ndarray, selected: np.ndarray, pos: int, day, info: dict
    ) -> float:
        """The delisting event at the formation row ``pos`` (module docstring; audit W-2).

        Holdings that have no price on that row and are not in the new selection become cash at
        their last price, valued at ``1 + terminal_return``. Reads nothing dated after ``day``.
        ``w`` (fractions of the fund) is rescaled in place to the fund value after the loss;
        returns the multiplier of the fund value (1 if nothing happened).
        """
        assert has_price is not None
        gone = (w > 0) & ~has_price[pos] & ~selected
        if not gone.any():
            return 1.0
        for j in np.flatnonzero(gone):
            info.setdefault("ended_assets", []).append(
                {"asset": str(prices.columns[j]), "date": day, "weight": float(w[j]),
                 "terminal_return": terminal_return}
            )
        loss = float(w[gone].sum()) * terminal_return
        w[gone] = 0.0
        if loss == 0.0:
            return 1.0
        mult = 1.0 + loss
        if mult <= 0.0:
            raise ValueError(
                f"terminal_return={terminal_return} on the delisting of "
                f"{[str(c) for c in prices.columns[gone]]} on {day.date()} wipes out the whole fund"
            )
        w /= mult  # the other holdings are larger shares of a smaller fund
        info["terminal_loss"] = loss
        return mult

    def report_frozen(
        info: dict, frozen, held: np.ndarray, target: np.ndarray, executed: np.ndarray
    ) -> None:
        if frozen is None:
            return
        shown = frozen & ((held > 0) | (target > 0))
        if shown.any():
            info["frozen_assets"] = [str(c) for c in prices.columns[shown]]
            info["frozen_weight"] = float(held[frozen].sum())
        left = float(target.sum() - executed.sum())
        if left > _CAP_TOL:
            info["unallocated_weight"] = left  # what the cap kept out of the tradable assets: cash

    fold_idx = 0
    i = 0
    while i + train_days + test_days <= len(dates):
        formation_pos = i + train_days - 1
        formation_date = dates[formation_pos]
        window = asset_returns.iloc[i + train_days : i + train_days + test_days]
        if per_fold:
            assets = tuple(str(a) for a in select_assets(formation_date))
            unknown = [a for a in assets if a not in col_pos]
            if unknown or not assets:
                raise ValueError(
                    f"select_assets on {formation_date.date()} returned "
                    + (f"unknown tickers {unknown[:5]}" if unknown else "no tickers")
                )
            repeated = sorted(a for a, count in Counter(assets).items() if count > 1)
            if repeated:  # a repeated ticker would silently turn part of the target into cash (W-4)
                raise ValueError(
                    f"select_assets on {formation_date.date()} returned duplicate tickers "
                    f"{repeated[:5]}"
                )
            idx = np.array([col_pos[a] for a in assets])
            train_slice = filled.iloc[i : i + train_days][list(assets)]
        else:
            assets = tuple(str(c) for c in prices.columns)
            idx = np.arange(n_assets)
            train_slice = prices.iloc[i : i + train_days]

        train_returns = log_returns(train_slice)
        if train_returns.empty:
            if per_fold:
                raise ValueError(
                    f"fold {fold_idx}: no complete training rows for the selected assets "
                    f"on {formation_date.date()}"
                )
            break
        problem = _make_problem(train_returns, cardinality, risk_aversion, w_min, w_max)

        info_pre: dict = {}
        fund_mult = 1.0  # value of the fund after a delisting loss, booked on the first window day
        if per_fold:
            selected = np.zeros(n_assets, dtype=bool)
            selected[idx] = True
            fund_mult = delist_at_formation(current, selected, formation_pos, formation_date,
                                            info_pre)
        pre_trade = current.copy()
        outside = float(pre_trade.sum() - pre_trade[idx].sum())
        ctx = StrategyContext(
            fold=fold_idx,
            formation_date=formation_date,
            current_weights=pre_trade[idx].copy(),
            transaction_cost_bps=transaction_cost_bps,
            cost_convention=cost_convention,
            assets=assets,
            outside_weight=outside,
        )
        t0 = time.perf_counter()
        target_sub, info = _call_strategy(strategy, accepts_ctx, problem, fold_idx, ctx)
        solve_time = time.perf_counter() - t0
        if target_sub.shape != (len(idx),):
            raise ValueError(f"strategy returned shape {target_sub.shape}, expected ({len(idx)},)")
        if abs(target_sub.sum() - problem.budget) > budget_tol or np.any(target_sub < -budget_tol):
            raise ValueError(
                f"[{strategy_name}] fold {fold_idx}: weights must be long-only and sum to "
                f"{problem.budget}, got sum={target_sub.sum():.6f}, min={target_sub.min():.3g}"
            )
        target = np.zeros(n_assets, dtype=float)
        target[idx] = target_sub
        info.update(info_pre)
        if per_fold:
            info["outside_weight"] = outside
            info["train_rows"] = len(train_returns)

        # Execution: at the formation close, or at the close of the ``execution_lag``-th TRADABLE
        # day after it (W-3), postponed to the first tradable day if the market is closed.
        exec_pos: int | None = -1  # -1 = formation close; k = after the k-th day of the window
        if execution_lag > 0 or (tradable_mask is not None and not tradable_mask.loc[formation_date]):
            exec_pos = None
            tradable_days = 0
            for k in range(len(window)):
                if tradable_mask is None or tradable_mask.loc[window.index[k]]:
                    tradable_days += 1
                    if tradable_days >= max(execution_lag, 1):
                        exec_pos = k
                        break

        traded = 0.0
        executed: np.ndarray | None = None
        weights = current.copy()
        if exec_pos == -1:
            frozen = frozen_on(formation_date)
            executed = _execute(target, current, frozen, w_max)
            traded = float(np.abs(executed - current).sum())
            report_frozen(info, frozen, current, target, executed)
            weights = executed.copy()
        cost = cost_factor * traded
        execution_date = (
            formation_date if exec_pos == -1 else (window.index[exec_pos] if exec_pos is not None else None)
        )

        gross_growth, net_growth = 1.0, 1.0
        fold_daily: list[float] = []
        pending_cost = cost if exec_pos == -1 else 0.0
        loss_mult = fund_mult  # a delisting loss is booked once, on the first day, like a cost
        for k, (d, r) in enumerate(window.iterrows()):
            rv = np.nan_to_num(r.to_numpy(dtype=float), nan=0.0)
            invested = weights.sum()
            rp = float(weights @ rv)
            growth = (1.0 + rp) * loss_mult
            gross_day = rp if loss_mult == 1.0 else growth - 1.0
            net_mult = growth * (1.0 - pending_cost)
            pending_cost = 0.0
            loss_mult = 1.0
            if invested > 0:
                weights = weights * (1.0 + rv) / (1.0 + rp)
            if exec_pos is not None and exec_pos >= 0 and k == exec_pos:
                # trade at this day's close: costs hit the next day's return
                frozen = frozen_on(d)
                executed = _execute(target, weights, frozen, w_max)
                traded = float(np.abs(executed - weights).sum())
                report_frozen(info, frozen, weights, target, executed)
                cost = cost_factor * traded
                pending_cost = cost
                weights = executed.copy()
            net_day = net_mult - 1.0
            value *= 1.0 + net_day
            gross_growth *= 1.0 + gross_day
            net_growth *= 1.0 + net_day
            day_dates.append(d)
            day_returns.append(net_day)
            fold_daily.append(net_day)
        if pending_cost:
            # trade on the last day of the window: charge it there (no later day in this fold)
            value *= 1.0 - pending_cost
            day_returns[-1] = (1.0 + day_returns[-1]) * (1.0 - pending_cost) - 1.0
            net_growth *= 1.0 - pending_cost

        folds.append(
            FoldResult(
                fold=fold_idx,
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                formation_date=formation_date,
                execution_date=execution_date,
                test_start=window.index[0],
                test_end=window.index[-1],
                weights=executed if executed is not None else pre_trade,  # post-trade holdings
                pre_trade_weights=pre_trade,
                end_weights=weights.copy(),
                realised_return=gross_growth - 1.0,
                net_return=net_growth - 1.0,
                realised_std=float(np.std(fold_daily) * np.sqrt(252)),
                traded=traded,
                turnover=0.5 * traded,
                transaction_cost=cost,
                solve_time_s=solve_time,
                info=info,
                assets=assets if per_fold else None,
            )
        )
        current = weights
        fold_idx += 1
        i += step_days

        if progress and fold_idx % 5 == 0:
            print(
                f"  [{strategy_name}] fold {fold_idx}: eq={value:.3f}, "
                f"turnover={0.5 * traded:.3f}, solve={solve_time * 1000:.0f}ms"
            )

    daily = pd.Series(day_returns, index=pd.DatetimeIndex(day_dates), name=strategy_name)
    equity = initial_capital * (1.0 + daily).cumprod() if len(daily) else None
    summary = _summarise(folds, daily, rf, transaction_cost_bps, cost_convention)
    summary["execution_lag"] = execution_lag
    if per_fold:
        summary["n_ended_assets"] = sum(len(f.info.get("ended_assets", [])) for f in folds)
        summary["terminal_return"] = terminal_return
    return BacktestResult(
        strategy_name=strategy_name,
        universe=universe,
        folds=folds,
        equity_curve=equity,
        daily_returns=daily,
        summary=summary,
    )


def _summarise(
    folds: list[FoldResult],
    daily: pd.Series,
    rf: pd.Series | None,
    tc_bps: float,
    cost_convention: str,
) -> dict:
    if not folds or daily.empty:
        return {}
    start = folds[0].formation_date
    perf = summarise_returns(daily, rf=rf, start=start)
    years = perf["years"]
    total_turnover = sum(f.turnover for f in folds)
    total_traded = sum(f.traded for f in folds)
    total_costs = sum(f.transaction_cost for f in folds)
    return {
        "n_folds": len(folds),
        **perf,
        "annual_turnover": total_turnover / years if years > 0 else total_turnover,
        "annual_traded": total_traded / years if years > 0 else total_traded,
        "annual_costs_bps": total_costs / years * 10000 if years > 0 else total_costs * 10000,
        "total_turnover": total_turnover,
        "total_costs_frac": total_costs,
        "mean_solve_time_s": float(np.mean([f.solve_time_s for f in folds])),
        "n_fallback": int(sum(bool(f.info.get("fallback", False)) for f in folds)),
        "n_postponed": int(sum(f.execution_date != f.formation_date for f in folds)),
        "tc_bps": tc_bps,
        "cost_convention": cost_convention,
    }
