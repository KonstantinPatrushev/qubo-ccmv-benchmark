"""Light-touch cleaning for daily price panels.

The policy is intentionally simple so it can be reasoned about in the report:

1. Forward-fill gaps up to ``max_ffill_days`` calendar days. Longer gaps remain
   NaN and are handled per-ticker.
2. Drop any ticker whose post-ffill NaN ratio exceeds ``max_nan_ratio``. This
   removes thinly traded names and tickers that started trading well after the
   window opens.
3. Trim the panel to the longest contiguous suffix where every surviving
   ticker has a price (no leading NaN). This guarantees the µ/Σ estimators
   downstream operate on a rectangular matrix.

The function also returns a small report describing what was dropped and why,
so the cleaning pass leaves an audit trail for the reproducibility document.

Two changes from WP-E1 (R6):

* The jump heuristic (``|r| > max_abs_daily_return`` -> rebase the price level) is **opt-in**.
  It cannot tell a split from a real move (AFKS 2014-12-18, +106 % that persisted, was
  "corrected" away) and it misses a consolidation that follows a trading halt (IRAO
  2015-01-20). Corporate actions come from the ISS split register, see
  :func:`cif.data.moex.adjust_for_splits`. The report lists every rebase that was applied.
* :func:`mask_closed_days` removes the days on which the exchange was closed *before* any
  forward fill, so a fill can never bridge the 28.02-23.03.2022 closure.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Adjustment:
    """One price-level rebase applied by the opt-in jump heuristic (``r`` = the flagged return)."""

    ticker: str
    date: str
    r: float


@dataclass
class CleaningReport:
    n_input_tickers: int
    n_output_tickers: int
    dropped_tickers: dict[str, str] = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""
    n_rows: int = 0
    adjustments: list[Adjustment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_input_tickers": self.n_input_tickers,
            "n_output_tickers": self.n_output_tickers,
            "n_dropped": len(self.dropped_tickers),
            "dropped_tickers": self.dropped_tickers,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_rows": self.n_rows,
            "adjustments": [asdict(a) for a in self.adjustments],
        }


def clean_prices(
    prices: pd.DataFrame,
    max_ffill_days: int = 5,
    max_nan_ratio: float = 0.10,
    max_abs_daily_return: float | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a wide price panel and return ``(cleaned, report)``.

    Corporate-action artefacts (**opt-in**, ``max_abs_daily_return=None`` means off): when a
    provider fails to apply a stock-split adjustment (notably VTBR on MOEX ISS which had a
    5000:1 reverse split in 2024), the resulting daily return is many orders of magnitude larger
    than any plausible market move. With a threshold we detect returns with
    ``|r| > max_abs_daily_return`` and rebase the price *level* from that day forward by
    dividing by ``(1 + r)``. The remaining panel then reflects the post-split price scale
    consistently, and the spurious one-day return is eliminated. This is applied *after*
    forward-fill and before the NaN-ratio filter. Every rebase is listed in
    ``report.adjustments``. The heuristic also removes real moves, so prefer adjusting by the
    ISS split register (:func:`cif.data.moex.adjust_for_splits`) and leave it off.
    """
    report = CleaningReport(n_input_tickers=prices.shape[1], n_output_tickers=0)

    if prices.empty:
        return prices, report

    filled = prices.ffill(limit=max_ffill_days)

    if max_abs_daily_return is not None:
        warnings.warn(
            "clean_prices: the |r| > max_abs_daily_return rebasing is an unverified heuristic "
            "(it deletes real moves and misses consolidations after a halt); "
            "adjust by the ISS split register instead.",
            UserWarning,
            stacklevel=2,
        )
        # Detect and neutralise split-like jumps
        returns = filled.pct_change(fill_method=None)
        flagged = returns.abs() > max_abs_daily_return
        adjustments_made = 0
        if flagged.any().any():
            corrected = filled.copy()
            for ticker in flagged.columns:
                flagged_dates = filled.index[flagged[ticker].fillna(False)]
                for bad_date in flagged_dates:
                    r = returns.loc[bad_date, ticker]
                    if pd.isna(r):
                        continue
                    factor = 1.0 + float(r)
                    if factor == 0.0:
                        continue
                    corrected.loc[bad_date:, ticker] = corrected.loc[bad_date:, ticker] / factor
                    adjustments_made += 1
                    report.adjustments.append(
                        Adjustment(str(ticker), str(bad_date.date()), float(r))
                    )
            filled = corrected
        report.dropped_tickers["_split_adjustments"] = f"{adjustments_made} price-level rebases"

    nan_ratio = filled.isna().mean()
    too_sparse = nan_ratio[nan_ratio > max_nan_ratio].index
    for ticker in too_sparse:
        report.dropped_tickers[str(ticker)] = (
            f"nan_ratio={nan_ratio[ticker]:.3f} > {max_nan_ratio}"
        )
    surviving = filled.drop(columns=too_sparse)

    if surviving.empty:
        report.n_output_tickers = 0
        return surviving, report

    first_valid_per_ticker = surviving.apply(lambda s: s.first_valid_index())
    panel_start = first_valid_per_ticker.max()
    if pd.isna(panel_start):
        return surviving.iloc[0:0], report
    rectangular = surviving.loc[panel_start:].dropna(how="any")

    report.n_output_tickers = rectangular.shape[1]
    report.n_rows = rectangular.shape[0]
    if not rectangular.empty:
        report.start_date = str(rectangular.index[0].date())
        report.end_date = str(rectangular.index[-1].date())

    return rectangular, report


def mask_closed_days(
    prices: pd.DataFrame, tradable: pd.Series | pd.DatetimeIndex
) -> pd.DataFrame:
    """Restrict a price panel to the days on which the market was open.

    ``tradable`` is either a boolean Series over calendar days (see
    :func:`cif.data.moex.build_calendar`) or the index of open days. Rows on closed days are
    dropped and open days that are absent from ``prices`` appear as all-NaN rows, limited to the
    date range of the panel. Call it **before** :func:`clean_prices`: forward-filling a panel
    that still has rows for closed days manufactures "trading days" with zero returns, which is
    what the legacy pipeline did on 28.02-04.03.2022, and the fill limit is then spent on days
    that never existed.
    """
    if isinstance(tradable, pd.Series):
        open_days = tradable.index[tradable.to_numpy(dtype=bool)]
    else:
        open_days = pd.DatetimeIndex(tradable)
    if prices.empty:
        return prices
    open_days = open_days[(open_days >= prices.index.min()) & (open_days <= prices.index.max())]
    masked = prices.reindex(open_days.sort_values().unique())
    masked.index.name = prices.index.name
    return masked
