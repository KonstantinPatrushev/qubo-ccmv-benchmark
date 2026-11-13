"""Price and index checks of dividend events (R6; plan section 3, RU).

Two independent kinds of evidence, both from data that are not the dividend table:

1. **The ex-day price drop.** A share that goes ex-dividend loses about the dividend. For every
   event the abnormal return on the ex-date (the share's return less beta times the IMOEX return
   over the same sessions) is compared with ``-yield``, in units of the share's own robust daily
   standard deviation: ``z = (abnormal + yield) / sigma``. Events whose yield is below
   ``MIN_YIELD_SIGMA`` sigmas cannot be tested (``weak``). The drop need not equal the dividend
   (taxes, news on the day), so an ``outlier`` is a reason to look, not proof of an error.
2. **The index dividend flow.** ``MCFTR`` (gross total return) minus ``IMOEX`` (price) return on
   a day is ``sum_i w_i(t-1) * D_i / P_i(t-1)`` over the shares going ex that day, with ``w`` the
   index weights at the previous close. The weights come from the monthly snapshots of the index
   composition, drifted with prices between snapshots (:func:`drift_weights`); the sum is rebuilt
   from the events (:func:`rebuild_index_flow`) and compared with the published flow day by day
   and over blocks of sessions (:func:`flow_metrics`). A residual is a *shift* when an equal
   residual of the opposite sign lies within a few sessions (the index books the dividend on a
   different day than the rule), *missing* when the index paid more than the events explain, and
   *extra* when the events promise more than the index paid (:func:`residual_episodes`).

Only RUB-denominated events enter the rebuilt flow: the index converts foreign-currency dividends at
the central bank rate, so such an event enters once it has been converted (the build supplies the
official rate). The residual of the flow rebuilt *without* them is what the rate check of
:func:`implied_fx_from_flow` divides by the events' weighted yield.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cif.data import moex

#: A dividend is attributed to the vendor's ex-date instead of the rule's only if it is at least
#: this many daily standard deviations of the share's abnormal return (and, in the checks, it is
#: testable against the price at all) ...
MIN_YIELD_SIGMA = 2.0
#: ... and the vendor date fits the price drop better by at least this many standard deviations.
MARGIN_SIGMA = 1.0
#: |z| above which an ex-day price drop is called borderline / an outlier.
Z_BORDERLINE = 3.0
Z_OUTLIER = 4.0
#: A share's median "capture" (drop / yield) outside this band means the scale of its amounts is off.
SCALE_BAND = (1.0 / 3.0, 3.0)
_MAD = 1.4826  # MAD -> standard deviation of a normal sample


# --------------------------------------------------------------------------------------------
# Ex-day price drop
# --------------------------------------------------------------------------------------------


def market_model(
    adj: pd.DataFrame, market_close: pd.Series, *, min_obs: int = 60, clip: float = 4.0
) -> pd.DataFrame:
    """Beta against the market and a robust daily sigma of the residual, per share.

    Uses returns between consecutive quotes only. Returns are winsorised at ``clip`` robust
    standard deviations before the slope is taken, and sigma is the MAD of the residual, so the
    few large ex-day drops and news days do not leak into the baseline. ``NaN`` with fewer than
    ``min_obs`` paired returns.
    """
    rm = market_close.reindex(adj.index).pct_change(fill_method=None)
    ret = adj.pct_change(fill_method=None)
    rows: dict[str, tuple[float, float, int]] = {}
    for secid in adj.columns:
        r = ret[secid]
        ok = r.notna() & rm.notna()
        n = int(ok.sum())
        if n < min_obs:
            rows[secid] = (np.nan, np.nan, n)
            continue
        x, y = rm[ok], r[ok]
        med = y.median()
        scale = _MAD * (y - med).abs().median()
        if not scale > 0 or not x.var(ddof=0) > 0:
            rows[secid] = (np.nan, np.nan, n)
            continue
        yc = y.clip(med - clip * scale, med + clip * scale)
        beta = float(((x - x.mean()) * (yc - yc.mean())).mean() / x.var(ddof=0))
        resid = y - beta * x
        sigma = float(_MAD * (resid - resid.median()).abs().median())
        rows[secid] = (beta, sigma, n)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["beta", "sigma", "n_obs"])


def ex_day_return(
    adj: pd.DataFrame, market_close: pd.Series, secid: str, ex_date, *, max_gap: int = 5,
    predecessors: dict[str, str] | None = None,
) -> dict:
    """Return of ``secid`` on ``ex_date`` against its previous available quote, and the market's
    return over the same sessions. ``gap`` is the number of sessions between the two quotes; a
    gap above ``max_gap`` (or no quote on the day, or no earlier quote) gives ``NaN`` returns.

    ``predecessors`` maps a series to the one it continues (``UPRO`` -> ``EONR``): when the
    series has no earlier quote because the ex-date is its first day, the predecessor's last
    close before the date serves as the previous quote (a renamed security keeps its share
    basis). ``prev_secid`` names the series the previous quote comes from.
    """
    out = {"prev_date": pd.NaT, "prev_adj": np.nan, "ret": np.nan, "mkt_ret": np.nan, "gap": np.nan,
           "prev_secid": None}
    ex = pd.Timestamp(ex_date)
    if secid not in adj.columns or ex not in adj.index or pd.isna(adj.at[ex, secid]):
        return out
    cutoff = ex - pd.Timedelta(days=1)
    prev_secid = secid
    before = adj[secid].loc[:cutoff].dropna()
    if before.empty and predecessors and predecessors.get(secid) in adj.columns:
        prev_secid = predecessors[secid]
        before = adj[prev_secid].loc[:cutoff].dropna()
    if before.empty:
        return out
    prev = before.index[-1]
    out.update(prev_date=prev, prev_adj=float(before.iloc[-1]), prev_secid=prev_secid,
               gap=int(adj.index.get_loc(ex) - adj.index.get_loc(prev)))
    if out["gap"] > max_gap:
        return out
    mkt = market_close.reindex(adj.index)
    out["ret"] = float(adj.at[ex, secid] / before.iloc[-1] - 1.0)
    out["mkt_ret"] = float(mkt.at[ex] / mkt.at[prev] - 1.0)
    return out


def _adjusted_amount(secid, ex, amount, applied, last_date):
    """The amount in the share basis of the adjusted prices (the last date's)."""
    if applied is None or last_date is None or len(applied) == 0:
        return amount
    sub = applied[applied["secid"] == secid]
    if sub.empty:
        return amount
    factor = moex.split_factor(pd.DatetimeIndex([ex, last_date]), sub)
    return amount * float(factor.iloc[0])


def check_events(
    events: pd.DataFrame,
    adj: pd.DataFrame,
    market_close: pd.Series,
    model: pd.DataFrame,
    *,
    splits: pd.DataFrame | None = None,
    last_date=None,
    min_yield_sigma: float = MIN_YIELD_SIGMA,
    z_borderline: float = Z_BORDERLINE,
    z_outlier: float = Z_OUTLIER,
    max_gap: int = 5,
    predecessors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compare every event's ex-day price drop with its dividend yield.

    ``events`` needs ``secid``, ``ex_date``, ``amount`` (gross, per share, in the share basis of
    the ex-date) and ``currency``. ``splits`` (with ``applied``) and ``last_date`` convert the
    amount to the basis of ``adj`` (split-adjusted to the last date); splits that were not
    applied to the prices (after the end of the panel) are left out.

    ``predecessors`` lets the first day of a renamed series borrow the old series' last close
    (:func:`ex_day_return`). Adds ``prev_date``, ``prev_secid``, ``prev_close_adj``, ``ex_return``, ``mkt_return``, ``beta``, ``sigma``,
    ``abn_return``, ``amount_adj``, ``dividend_yield``, ``z``, ``capture`` (``-abn / yield``) and
    ``drop_status``: ``non_rub``, ``no_quote``, ``no_model``, ``weak``, ``ok``, ``borderline``
    (``z_borderline < |z| <= z_outlier``) or ``outlier``.
    """
    out = events.reset_index(drop=True).copy()
    applied = None
    if splits is not None and len(splits):
        applied = splits[splits["applied"].astype(bool)] if "applied" in splits.columns else splits
    cols = ["prev_date", "prev_secid", "prev_close_adj", "ex_return", "mkt_return", "beta", "sigma", "abn_return",
            "amount_adj", "dividend_yield", "z", "capture"]
    rec = {c: [] for c in cols}
    status = []
    for r in out.itertuples(index=False):
        er = ex_day_return(adj, market_close, r.secid, r.ex_date, max_gap=max_gap, predecessors=predecessors)
        beta, sigma = (model.at[r.secid, "beta"], model.at[r.secid, "sigma"]) if r.secid in model.index \
            else (np.nan, np.nan)
        abn = er["ret"] - beta * er["mkt_ret"] if not np.isnan(er["ret"]) else np.nan
        amount_adj = _adjusted_amount(r.secid, r.ex_date, r.amount, applied, last_date)
        y = amount_adj / er["prev_adj"] if er["prev_adj"] > 0 else np.nan
        rub = str(r.currency).upper() == "RUB"
        if not rub:
            z = cap = np.nan
            st = "non_rub"
        else:
            z = (abn + y) / sigma if sigma > 0 else np.nan
            cap = -abn / y if y > 0 else np.nan
            if np.isnan(er["ret"]):
                st = "no_quote"
            elif not sigma > 0 or np.isnan(beta):
                st = "no_model"
            elif y < min_yield_sigma * sigma:
                st = "weak"
            elif abs(z) <= z_borderline:
                st = "ok"
            elif abs(z) <= z_outlier:
                st = "borderline"
            else:
                st = "outlier"
        for c, v in zip(cols, (er["prev_date"], er["prev_secid"], er["prev_adj"], er["ret"], er["mkt_ret"], beta,
                               sigma, abn, amount_adj, y, z, cap), strict=True):
            rec[c].append(v)
        status.append(st)
    for c in cols:
        out[c] = rec[c]
    out["prev_date"] = pd.to_datetime(out["prev_date"])
    out["drop_status"] = status
    return out


def scale_check(
    checked: pd.DataFrame, *, min_events: int = 3, band: tuple[float, float] = SCALE_BAND,
    min_yield_sigma: float = MIN_YIELD_SIGMA,
) -> pd.DataFrame:
    """Is the scale of a share's amounts right? (catches splits that no register lists)

    Per share, over its RUB events in which either the dividend or the price drop is at least
    ``min_yield_sigma`` sigmas, the median of ``capture = drop / yield`` (clipped to 0.01-100)
    should be near 1. Amounts ten times too large give about 0.1, ten times too small about 10.
    An event with a tiny yield and a big drop counts: it is exactly what too-small amounts look
    like. Status ``ok``, ``scale_suspect`` (median outside ``band``) or ``insufficient``.
    """
    t = checked[checked["drop_status"].isin(["weak", "ok", "borderline", "outlier"])].copy()
    informative = (t["dividend_yield"] >= min_yield_sigma * t["sigma"]) | (
        -t["abn_return"] >= min_yield_sigma * t["sigma"])
    t = t[informative]
    t["capture_clipped"] = t["capture"].clip(0.01, 100.0)
    g = t.groupby("secid")
    out = pd.DataFrame({"n_informative": g.size(), "median_capture": g["capture_clipped"].median()})
    out["status"] = np.where(
        out["n_informative"] < min_events, "insufficient",
        np.where(out["median_capture"].between(*band), "ok", "scale_suspect"),
    )
    n_all = checked.groupby("secid").size().rename("n_events")
    return n_all.to_frame().join(out).fillna({"n_informative": 0, "status": "insufficient"}).astype(
        {"n_informative": int})


def restatement_verdict(
    events: pd.DataFrame, adj: pd.DataFrame, market_close: pd.Series, model: pd.DataFrame, *,
    splits: pd.DataFrame | None = None, last_date=None, ratio: float = 0.5,
) -> pd.DataFrame:
    """For the events whose amount was restated, which amount fits the ex-day drop?

    ``events`` needs ``amount`` (restated), ``cbonds_amount`` (as exported) and
    ``restatement_factor``; only rows with a factor other than 1 are examined. Per share the sums
    of ``z**2`` under both hypotheses are compared: ``restated`` if that of the restated amounts
    is below ``ratio`` times the other, ``raw`` in the opposite case, else ``inconclusive``.
    """
    sel = events[events["restatement_factor"] != 1.0]
    cols = ["n_events", "sum_z2_restated", "sum_z2_raw", "median_capture_restated",
            "median_capture_raw", "verdict"]
    if sel.empty:
        return pd.DataFrame(columns=cols)
    kw = {"splits": splits, "last_date": last_date}
    restated = check_events(sel, adj, market_close, model, **kw)
    raw = check_events(sel.assign(amount=sel["cbonds_amount"]), adj, market_close, model, **kw)
    out = pd.DataFrame({
        "n_events": sel.groupby("secid").size(),
        "sum_z2_restated": (restated["z"] ** 2).groupby(restated["secid"]).sum(min_count=1),
        "sum_z2_raw": (raw["z"] ** 2).groupby(raw["secid"]).sum(min_count=1),
        "median_capture_restated": restated.groupby("secid")["capture"].median(),
        "median_capture_raw": raw.groupby("secid")["capture"].median(),
    })
    out["verdict"] = [
        restatement_decision(a, b, c, d, ratio=ratio) for a, b, c, d in zip(
            out["sum_z2_restated"], out["sum_z2_raw"], out["median_capture_restated"], out["median_capture_raw"],
            strict=True)
    ]
    return out[cols]


def restatement_decision(
    z2_restated: float, z2_raw: float, capture_restated: float, capture_raw: float, *, ratio: float = 0.5,
    log_margin: float = 0.5,
) -> str:
    """``restated`` / ``raw`` / ``inconclusive`` from the evidence on one share.

    First the sums of ``z**2``: one hypothesis wins if its sum is below ``ratio`` times the
    other's. When the yields are weak both sums are small and close (T: 2.07 against 2.71), so
    the medians of the capture (drop / yield) decide next: the hypothesis whose median is nearer
    to 1 on the logarithmic scale wins if it is nearer by more than ``log_margin`` decades. A
    median that is not positive gives no verdict.
    """
    if z2_restated < ratio * z2_raw:
        return "restated"
    if z2_raw < ratio * z2_restated:
        return "raw"
    if not (capture_restated > 0 and capture_raw > 0):
        return "inconclusive"
    near_restated, near_raw = abs(np.log10(capture_restated)), abs(np.log10(capture_raw))
    if near_restated + log_margin < near_raw:
        return "restated"
    if near_raw + log_margin < near_restated:
        return "raw"
    return "inconclusive"


def implied_fx_from_price(checked: pd.DataFrame) -> pd.Series:
    """RUB per unit of the dividend currency that makes the ex-day drop equal the dividend.

    ``-abnormal return * previous close / amount`` for events in a foreign currency whose price
    fell on the ex-date; ``NaN`` otherwise. It is a check, not a rate: a value far outside the
    plausible range (a few tens of RUB per USD) says the event, its date or its amount is off, a
    value in the range that moves smoothly over the years says they are right. Small dividends
    give noise.
    """
    drop = -checked["abn_return"]
    ok = (checked["currency"].astype(str).str.upper() != "RUB") & (drop > 0) & (checked["amount_adj"] > 0) \
        & (checked["prev_close_adj"] > 0)
    return (drop * checked["prev_close_adj"] / checked["amount_adj"]).where(ok).rename("implied_fx_price")


def implied_fx_from_flow(
    events: pd.DataFrame, weights: pd.DataFrame, close: pd.DataFrame, observed: pd.Series, rebuilt: pd.Series,
    days: pd.DatetimeIndex,
) -> pd.Series:
    """The exchange rate that the index dividend flow implies for each foreign-currency event.

    The index converts such dividends at the central bank's rate, the rebuilt flow leaves them
    out, so the residual ``observed - rebuilt`` of the ex-date holds ``rate * sum(w * D / P)`` over
    the events of one currency that go ex that day (plus noise from everything else). The rate
    is that residual over the sum, the same for all events of the currency on the day; ``NaN``
    for RUB events, members without weight, or days without a residual.
    """
    days = pd.DatetimeIndex(days)
    position = {d: i for i, d in enumerate(days)}
    w = weights.reindex(days)
    prev_close = close.reindex(days).ffill(limit=5).shift(1)
    resid = (observed.reindex(days) - rebuilt.reindex(days).fillna(0.0))
    parts = []
    for r in events.itertuples():
        i = position.get(pd.Timestamp(r.ex_date))
        cur = str(r.currency).upper()
        value = np.nan
        if i is not None and i > 0 and cur != "RUB" and r.secid in w.columns:
            weight, price = w.iat[i - 1, w.columns.get_loc(r.secid)], prev_close.iat[i, prev_close.columns.get_loc(r.secid)]
            if weight > 0 and price > 0:
                value = weight * r.amount / price
        parts.append((pd.Timestamp(r.ex_date), cur, value))
    frame = pd.DataFrame(parts, columns=["ex_date", "currency", "weighted_yield"], index=events.index)
    total = frame.groupby(["ex_date", "currency"])["weighted_yield"].transform(lambda s: s.sum(min_count=1))
    residual = frame["ex_date"].map(resid)
    rate = (residual / total).where(frame["weighted_yield"].notna() & (total > 0) & (residual > 0))
    return rate.rename("implied_fx_flow")


def placebo_exceedance(
    adj: pd.DataFrame, market_close: pd.Series, model: pd.DataFrame, *, exclude: pd.DataFrame | None = None,
) -> dict:
    """How often a day *without* a dividend looks like an outlier: the false-alarm rate of the
    thresholds. ``exclude`` has ``secid`` and ``ex_date`` of the days to leave out."""
    rm = market_close.reindex(adj.index).pct_change(fill_method=None)
    ret = adj.pct_change(fill_method=None)
    beta = model["beta"].reindex(ret.columns).to_numpy()[None, :]
    sigma = model["sigma"].reindex(ret.columns).to_numpy()[None, :]
    z = (ret - rm.to_numpy()[:, None] * beta) / sigma
    if exclude is not None and len(exclude):
        for secid, ex in zip(exclude["secid"], exclude["ex_date"], strict=True):
            if secid in z.columns and ex in z.index:
                z.at[ex, secid] = np.nan
    vals = z.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    return {"n_days": len(vals),
            "share_abs_above_borderline": float((np.abs(vals) > Z_BORDERLINE).mean()),
            "share_abs_above_outlier": float((np.abs(vals) > Z_OUTLIER).mean()),
            "share_below_minus_outlier": float((vals < -Z_OUTLIER).mean())}


# --------------------------------------------------------------------------------------------
# Index weights and the rebuilt dividend flow
# --------------------------------------------------------------------------------------------


def drift_weights(
    membership: pd.DataFrame, adj: pd.DataFrame, days: pd.DatetimeIndex, *, max_ffill: int = 20
) -> pd.DataFrame:
    """Index weight of every share at the close of every day.

    ``membership`` has the snapshots (``date``, ``secid``, ``weight``); between two snapshots the
    weights of the earlier one drift with the split-adjusted prices and are renormalised. A
    share without a quote (after ``max_ffill`` sessions of carrying the last one) keeps its
    snapshot weight. Days before the first snapshot get zeros, and so do non-members.
    """
    days = pd.DatetimeIndex(days)
    snaps = pd.DatetimeIndex(sorted(membership["date"].unique()))
    w0 = membership.pivot_table(index="date", columns="secid", values="weight", aggfunc="sum")
    prices = adj.ffill(limit=max_ffill)
    out = pd.DataFrame(0.0, index=days, columns=adj.columns)
    for k, s in enumerate(snaps):
        end = snaps[k + 1] if k + 1 < len(snaps) else None
        seg = days[(days >= s) & ((days < end) if end is not None else True)]
        if len(seg) == 0:
            continue
        w = w0.loc[s].dropna()
        members = [m for m in w[w > 0].index if m in prices.columns]
        if not members:
            continue
        base = prices.loc[:s, members]
        base = base.iloc[-1] if len(base) else pd.Series(np.nan, index=members)
        ratio = prices.loc[prices.index.intersection(seg), members].div(base).reindex(seg).fillna(1.0)
        drifted = ratio * w[members]
        out.loc[seg, members] = drifted.div(drifted.sum(axis=1), axis=0).to_numpy()
    return out


def rebuild_index_flow(
    events: pd.DataFrame, close: pd.DataFrame, weights: pd.DataFrame, days: pd.DatetimeIndex
) -> tuple[pd.Series, pd.DataFrame]:
    """``flow[t] = sum_i w_i(t-1) * D_i / P_i(t-1)`` over the RUB events with ex-date ``t``.

    ``D`` is in the share basis of the ex-date and ``P`` the as-traded close of the last session
    with a quote before ``t`` (at most five sessions back). Events without a weight or a price
    are skipped (they are in ``contrib`` only if used). Returns ``(flow over days, contrib)``.
    """
    days = pd.DatetimeIndex(days)
    w = weights.reindex(days)
    prev_close = close.reindex(days).ffill(limit=5).shift(1)
    position = {d: i for i, d in enumerate(days)}
    rows = []
    for r in events[events["currency"].astype(str).str.upper() == "RUB"].itertuples(index=False):
        i = position.get(pd.Timestamp(r.ex_date))
        if i is None or i == 0 or r.secid not in w.columns:
            continue
        weight, price = w.iat[i - 1, w.columns.get_loc(r.secid)], prev_close.iat[i, prev_close.columns.get_loc(r.secid)]
        if not (weight > 0 and price > 0):
            continue
        rows.append((r.secid, pd.Timestamp(r.ex_date), weight, price, r.amount, weight * r.amount / price))
    contrib = pd.DataFrame(rows, columns=["secid", "ex_date", "weight", "prev_close", "amount", "contribution"])
    flow = contrib.groupby("ex_date")["contribution"].sum().reindex(days, fill_value=0.0)
    flow.index.name = "date"
    return flow.rename("rebuilt_flow"), contrib


# --------------------------------------------------------------------------------------------
# Comparing the flows
# --------------------------------------------------------------------------------------------


def _block_l1_share(o: pd.Series, r: pd.Series, k: int) -> float:
    grp = np.arange(len(o)) // k
    ob, rb = o.groupby(grp).sum(), r.groupby(grp).sum()
    denom = ob.abs().sum()
    return float(1.0 - (ob - rb).abs().sum() / denom) if denom > 0 else np.nan


def flow_metrics(observed: pd.Series, rebuilt: pd.Series, *, blocks: tuple[int, ...] = (1, 5, 21)) -> pd.DataFrame:
    """How much of the published flow the rebuilt flow explains, by block size (sessions).

    ``l1_share = 1 - sum|obs - rebuilt| / sum|obs|`` over non-overlapping blocks of ``k``
    sessions: it penalises missing and spurious dividends alike, and blocks absorb a timing
    difference of fewer than ``k`` sessions. ``ratio_of_sums`` is the share of the total flow.
    Days without an observation are left out of both series.
    """
    o = observed.copy()
    r = rebuilt.reindex(o.index).fillna(0.0).where(o.notna(), 0.0)
    o0 = o.fillna(0.0)
    rows = {}
    for k in blocks:
        grp = np.arange(len(o0)) // k
        ob, rb = o0.groupby(grp).sum(), r.groupby(grp).sum()
        corr = float(np.corrcoef(ob, rb)[0, 1]) if len(ob) > 2 and ob.std() > 0 and rb.std() > 0 else np.nan
        rows[k] = {"n_days": int(o.notna().sum()), "sum_obs": float(o0.sum()), "sum_rebuilt": float(r.sum()),
                   "ratio_of_sums": float(r.sum() / o0.sum()) if o0.sum() != 0 else np.nan,
                   "l1_share": _block_l1_share(o0, r, k), "corr": corr}
    return pd.DataFrame.from_dict(rows, orient="index")


def flow_by_year(observed: pd.Series, rebuilt: pd.Series) -> pd.DataFrame:
    """Totals and shares explained per calendar year (daily and 5-session blocks)."""
    o = observed.dropna()
    r = rebuilt.reindex(o.index).fillna(0.0)
    rows = {}
    for year, idx in o.groupby(o.index.year):
        oy, ry = idx, r.loc[idx.index]
        rows[year] = {"obs": float(oy.sum()), "rebuilt": float(ry.sum()),
                      "ratio": float(ry.sum() / oy.sum()) if oy.sum() != 0 else np.nan,
                      "l1_share_daily": _block_l1_share(oy, ry, 1),
                      "l1_share_weekly": _block_l1_share(oy, ry, 5)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "year"
    return out


def residual_episodes(
    observed: pd.Series, rebuilt: pd.Series, *, min_abs: float = 2e-4, window: int = 5, rel_tol: float = 0.35,
) -> pd.DataFrame:
    """Days on which the rebuilt flow misses the published one by at least ``min_abs``, with a name.

    The residual is ``observed - rebuilt``. A positive residual and a negative one of similar size
    (within ``rel_tol`` of the larger) at most ``window`` sessions apart are one ``shift``: the
    index booked the dividend on a different day than the rule (``shift_sessions`` is own session
    less partner session). Of what is left, residuals at most ``window`` sessions apart form a
    cluster; a cluster whose positive and negative parts balance (the smaller is at least
    ``1 - rel_tol`` of the larger) is a ``shift_cluster`` (one dividend booked on several days, or
    several booked together). Otherwise a positive residual is ``missing`` (the index paid more
    than the events explain) and a negative one ``extra`` (the events promise more than it paid).
    ``cluster_id`` groups the rows of one shift, pair or cluster.
    """
    o = observed.dropna()
    res = o - rebuilt.reindex(o.index).fillna(0.0)
    cand = res[res.abs() >= min_abs]
    pos = {d: i for i, d in enumerate(o.index)}
    free = set(cand.index)
    rows = []
    cid = 0
    for d in sorted(cand.index, key=lambda x: (-abs(cand[x]), x)):
        if d not in free:
            continue
        v = cand[d]
        best = None
        for e in sorted(free):
            w = cand[e]
            if e == d or np.sign(w) == np.sign(v) or abs(pos[e] - pos[d]) > window:
                continue
            if abs(abs(w) - abs(v)) > rel_tol * max(abs(w), abs(v)):
                continue
            if best is None or abs(pos[e] - pos[d]) < abs(pos[best] - pos[d]):
                best = e
        if best is not None:
            rows.append((d, "shift", best, pos[d] - pos[best], cid))
            rows.append((best, "shift", d, pos[best] - pos[d], cid))
            free -= {d, best}
            cid += 1
    clusters: list[list] = []
    for d in sorted(free, key=pos.get):
        if clusters and pos[d] - pos[clusters[-1][-1]] <= window:
            clusters[-1].append(d)
        else:
            clusters.append([d])
    for members in clusters:
        vals = cand[members]
        plus, minus = vals[vals > 0].sum(), -vals[vals < 0].sum()
        balanced = plus > 0 and minus > 0 and min(plus, minus) >= (1.0 - rel_tol) * max(plus, minus)
        for d in members:
            kind = "shift_cluster" if balanced else ("missing" if cand[d] > 0 else "extra")
            rows.append((d, kind, pd.NaT, np.nan, cid))
        cid += 1
    out = pd.DataFrame(rows, columns=["date", "kind", "partner_date", "shift_sessions", "cluster_id"])
    out["observed"] = o.reindex(out["date"]).to_numpy()
    out["rebuilt"] = (o - res).reindex(out["date"]).to_numpy()
    out["residual"] = res.reindex(out["date"]).to_numpy()
    out = out.sort_values("date").reset_index(drop=True)
    return out[["date", "observed", "rebuilt", "residual", "kind", "partner_date", "shift_sessions", "cluster_id"]]


def event_index_status(
    contrib: pd.DataFrame, observed: pd.Series, rebuilt: pd.Series, episodes: pd.DataFrame, *,
    min_contribution: float = 3e-4, tol: float = 0.5, noise: float = 0.05,
) -> pd.Series:
    """What the published flow says about each event that entered the rebuilt flow.

    ``untestable``: the event's contribution ``w * yield`` is below ``min_contribution``, too small
    to see. ``timing_shift``: its day is part of a shift in :func:`residual_episodes` (the index
    booked it on another session). ``confirmed``: the day's residual is at most ``tol`` times the
    event's own contribution, or at most ``noise`` times the flow rebuilt for the day (weights are
    approximate, so a small event on a large day cannot be contradicted by a modest gap).
    ``inconsistent``: the residual is larger, so the index does not support the amount (or the
    day) of this event; other events of the day may be the cause.
    """
    obs = observed.dropna()
    res = obs - rebuilt.reindex(obs.index).fillna(0.0)
    shift_days = set(episodes.loc[episodes["kind"].isin(["shift", "shift_cluster"]), "date"])
    out = []
    for r in contrib.itertuples():
        if r.contribution < min_contribution:
            out.append("untestable")
        elif r.ex_date in shift_days:
            out.append("timing_shift")
        elif abs(res.get(r.ex_date, np.nan)) <= max(tol * r.contribution, noise * float(rebuilt.get(r.ex_date, 0.0))):
            out.append("confirmed")
        else:
            out.append("inconsistent")
    return pd.Series(out, index=contrib.index, name="index_status")


def missing_event_leads(
    episodes: pd.DataFrame, weights: pd.DataFrame, events: pd.DataFrame, adj: pd.DataFrame,
    market_close: pd.Series, model: pd.DataFrame, days: pd.DatetimeIndex, *, top: int = 3,
    min_weight: float = 0.002, exclude_sessions: int = 3, max_yield: float = 0.30, kinds: tuple[str, ...] = ("missing",),
) -> pd.DataFrame:
    """For each unexplained positive residual, the index members whose ex-day drop fits it best.

    A residual ``r`` on day ``t`` that one share ``i`` explains needs ``yield_i = r / w_i(t-1)``.
    Members with weight of at least ``min_weight`` and no event within ``exclude_sessions``
    sessions are ranked by ``|z|`` of ``(abnormal return + implied yield) / sigma``; ``plausible``
    means a yield between 0.5 % and ``max_yield`` and ``|z| <= 2``. These are leads, not
    findings: several shares can share the residual.
    """
    days = pd.DatetimeIndex(days)
    position = {d: i for i, d in enumerate(days)}
    near = {}
    for secid, group in events.groupby("secid"):
        near[secid] = [position[d] for d in group["ex_date"] if d in position]
    w = weights.reindex(days)
    rows = []
    for e in episodes[episodes["kind"].isin(kinds)].itertuples():
        i = position.get(e.date)
        if i is None or i == 0:
            continue
        found = []
        for secid, wi in w.iloc[i - 1].items():
            if not wi >= min_weight or secid not in adj.columns or secid not in model.index:
                continue
            if any(abs(j - i) <= exclude_sessions for j in near.get(secid, [])):
                continue
            er = ex_day_return(adj, market_close, secid, e.date)
            beta, sigma = model.at[secid, "beta"], model.at[secid, "sigma"]
            if np.isnan(er["ret"]) or not sigma > 0:
                continue
            implied = e.residual / wi
            abn = er["ret"] - beta * er["mkt_ret"]
            z = (abn + implied) / sigma
            found.append((e.date, e.residual, secid, wi, implied, abn, sigma, z,
                          bool(0.005 <= implied <= max_yield and abs(z) <= 2.0)))
        found.sort(key=lambda r: abs(r[7]))
        rows.extend((*r, rank) for rank, r in enumerate(found[:top], start=1))
    return pd.DataFrame(rows, columns=["date", "residual", "secid", "weight", "implied_yield", "abn_return",
                                       "sigma", "z", "plausible", "rank"])


@dataclass
class IndexValidation:
    """Result of :func:`index_validation`."""

    weights: pd.DataFrame
    flow: pd.Series  # the rebuilt flow
    contrib: pd.DataFrame  # events that entered it, with ``index_status``
    metrics: pd.DataFrame  # by block size
    by_year: pd.DataFrame
    episodes: pd.DataFrame


def index_validation(
    events: pd.DataFrame, close: pd.DataFrame, adj: pd.DataFrame, membership: pd.DataFrame, observed: pd.Series,
    days: pd.DatetimeIndex, *, start="2013-09-03", min_abs: float = 2e-4, window: int = 5,
) -> IndexValidation:
    """Rebuild the index dividend flow from the usable RUB events and compare it with ``observed``."""
    weights = drift_weights(membership, adj, days)
    flow, contrib = rebuild_index_flow(events[events["usable_rub"]], close, weights, days)
    obs = observed.reindex(days)
    obs = obs[obs.index >= pd.Timestamp(start)]
    episodes = residual_episodes(obs, flow, min_abs=min_abs, window=window)
    contrib = contrib.assign(index_status=event_index_status(contrib, obs, flow, episodes))
    return IndexValidation(weights=weights, flow=flow, contrib=contrib, metrics=flow_metrics(obs, flow),
                           by_year=flow_by_year(obs, flow), episodes=episodes)
