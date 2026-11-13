"""Risk-adjusted performance metrics computed on DAILY return series (R2).

All functions take simple (not log) periodic returns. When a risk-free series is given, Sharpe,
Sortino and CEQ are computed on excess returns ``r − rf`` aligned by date; missing risk-free
values are forward-filled. Conventions:

* Sharpe — mean / sample std (ddof=1) of excess returns, annualised by √periods;
* Sortino — mean excess return / downside deviation, where the downside deviation is
  ``sqrt(mean(min(r − target, 0)²))`` over ALL observations (Sortino & Price, 1994), target = rf;
* maximum drawdown — on the daily equity curve (intra-month drawdowns are visible);
* CEQ — ``mean − (γ/2)·var`` of excess returns, annualised (DeMiguel, Garlappi & Uppal, 2009);
* CAGR — compounded growth over calendar time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def _as_series(r) -> pd.Series:
    return r if isinstance(r, pd.Series) else pd.Series(np.asarray(r, dtype=float))


def excess_returns(r, rf=None) -> pd.Series:
    """``r − rf`` aligned on the index of ``r``; ``rf=None`` returns ``r`` unchanged.

    With date indexes, ``rf`` is a series of per-period accruals on ITS OWN calendar (e.g.
    ``r_daily`` of :mod:`cif.data.riskfree`). The risk-free return charged on a return date is the
    accrual compounded since the previous return date, so days missing from either calendar are
    neither lost nor counted twice (audit M-1). The first return date is charged the accrual of the
    rf periods up to it since the last rf date before the series starts.
    """
    r = _as_series(r).astype(float)
    if rf is None:
        return r
    if np.isscalar(rf):
        return r - float(rf)
    rf = _as_series(rf).astype(float)
    if isinstance(r.index, pd.DatetimeIndex) and isinstance(rf.index, pd.DatetimeIndex):
        rf = rf.sort_index().dropna()
        growth = (1.0 + rf).cumprod()
        level = growth.reindex(growth.index.union(r.index)).ffill().reindex(r.index).fillna(1.0)
        prev = level.shift(1)
        first = r.index[0]
        if first in growth.index:
            before = growth.loc[:first].iloc[:-1]
            # the accrual of the first return date itself (since the previous rf date)
            prev.iloc[0] = before.iloc[-1] if len(before) else 1.0
        else:
            prev.iloc[0] = level.iloc[0]
        return r - (level / prev - 1.0)
    return r - rf.to_numpy()


def downside_deviation(r, target: float = 0.0) -> float:
    """Downside deviation over all observations: ``sqrt(mean(min(r − target, 0)²))``."""
    x = np.asarray(r, dtype=float) - target
    return float(np.sqrt(np.mean(np.minimum(x, 0.0) ** 2))) if x.size else float("nan")


def sharpe_ratio(r, rf=None, periods: int = PERIODS_PER_YEAR) -> float:
    ex = excess_returns(r, rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")


def sortino_ratio(r, rf=None, periods: int = PERIODS_PER_YEAR) -> float:
    ex = excess_returns(r, rf)
    dd = downside_deviation(ex.to_numpy())
    return float(ex.mean() / dd * np.sqrt(periods)) if dd > 0 else float("nan")


def certainty_equivalent(r, rf=None, gamma: float = 5.0, periods: int = PERIODS_PER_YEAR) -> float:
    ex = excess_returns(r, rf)
    return float((ex.mean() - 0.5 * gamma * ex.var(ddof=1)) * periods)


def max_drawdown(equity) -> tuple[float, object, object]:
    """Return ``(mdd, peak_label, trough_label)``; ``mdd ≤ 0`` is relative to the running peak."""
    eq = _as_series(equity).astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def annualised_return(r, start=None, periods: int = PERIODS_PER_YEAR) -> float:
    """CAGR of a simple-return series.

    With a DatetimeIndex, time is measured in calendar years from ``start`` (the date the
    capital was invested; defaults to the day before the first return) to the last date.
    Otherwise ``len(r) / periods`` years are assumed.
    """
    r = _as_series(r).astype(float)
    growth = float(np.prod(1.0 + r.to_numpy()))
    if isinstance(r.index, pd.DatetimeIndex) and len(r):
        t0 = pd.Timestamp(start) if start is not None else r.index[0] - pd.Timedelta(days=1)
        years = (r.index[-1] - t0).days / 365.25
    else:
        years = len(r) / periods
    return growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else float("nan")


def summarise_returns(
    r,
    rf=None,
    start=None,
    gamma: float = 5.0,
    periods: int = PERIODS_PER_YEAR,
) -> dict:
    """Standard summary of a daily net-return series (see module docstring)."""
    r = _as_series(r).astype(float)
    equity = (1.0 + r).cumprod()
    if isinstance(r.index, pd.DatetimeIndex) and len(r):
        t0 = pd.Timestamp(start) if start is not None else r.index[0] - pd.Timedelta(days=1)
        years = (r.index[-1] - t0).days / 365.25
        # the initial capital (1.0 at t0) is a possible peak: drawdowns from day one count, and
        # the peak label is then t0 (audit M-2)
        with_start = pd.concat([pd.Series([1.0], index=[t0]), equity])
    else:
        years = len(r) / periods
        with_start = pd.concat([pd.Series([1.0], index=["start"]), equity.reset_index(drop=True)])
    mdd, peak, trough = max_drawdown(with_start)
    return {
        "n_days": len(r),
        "years": float(years),
        "final_equity": float(equity.iloc[-1]) if len(equity) else 1.0,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "cagr": annualised_return(r, start=start, periods=periods),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(periods)),
        "sharpe": sharpe_ratio(r, rf, periods),
        "sortino": sortino_ratio(r, rf, periods),
        "ceq": certainty_equivalent(r, rf, gamma, periods),
        "max_drawdown": float(mdd),
        "mdd_peak": str(peak),
        "mdd_trough": str(trough),
        "rf_used": rf is not None,
    }
