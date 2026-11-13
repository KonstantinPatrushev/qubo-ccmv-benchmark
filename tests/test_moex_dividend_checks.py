"""Price and index checks of dividend events (``cif.data.moex_dividend_checks``), on synthetic panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cif.data import moex_dividend_checks as ck

TS = pd.Timestamp
DAYS = pd.bdate_range("2020-01-01", periods=400, name="date")


N_DAYS = len(DAYS)


def _market(seed=1, n=N_DAYS, sd=0.01):
    r = np.random.default_rng(seed).normal(0.0, sd, n)
    r[0] = 0.0
    return pd.Series(1000.0 * np.cumprod(1 + r), index=DAYS)


def _stock(market, beta, noise, seed, start=100.0, drops=None):
    """Price following ``beta`` times the market plus noise; ``drops`` = {position: extra return}."""
    rm = market.pct_change().fillna(0.0).to_numpy()
    eps = np.random.default_rng(seed).normal(0.0, noise, len(market))
    r = beta * rm + eps
    for pos, extra in (drops or {}).items():
        r[pos] += extra
    r[0] = 0.0
    return pd.Series(start * np.cumprod(1 + r), index=DAYS)


# --------------------------------------------------------------------------------------------
# Market model
# --------------------------------------------------------------------------------------------


def test_market_model_recovers_beta_and_sigma_despite_outlier_days():
    mkt = _market()
    px = _stock(mkt, beta=0.8, noise=0.008, seed=2, drops={100: -0.08, 200: -0.06, 300: 0.07})
    m = ck.market_model(pd.DataFrame({"A": px}), mkt)
    assert m.loc["A", "beta"] == pytest.approx(0.8, abs=0.1)
    assert m.loc["A", "sigma"] == pytest.approx(0.008, abs=0.002)
    assert m.loc["A", "n_obs"] > 380


def test_market_model_needs_enough_observations():
    mkt = _market()
    px = _stock(mkt, 1.0, 0.01, 3)
    px.iloc[:350] = np.nan
    m = ck.market_model(pd.DataFrame({"A": px}), mkt, min_obs=60)
    assert np.isnan(m.loc["A", "beta"]) and np.isnan(m.loc["A", "sigma"])


def test_market_model_of_a_constant_price_is_undefined_not_zero():
    mkt = _market()
    flat = pd.Series(50.0, index=DAYS)
    m = ck.market_model(pd.DataFrame({"A": flat}), mkt)
    assert np.isnan(m.loc["A", "sigma"]) and np.isnan(m.loc["A", "beta"])


# --------------------------------------------------------------------------------------------
# Ex-day return
# --------------------------------------------------------------------------------------------


def test_ex_day_return_is_against_the_previous_available_quote():
    mkt = _market()
    px = _stock(mkt, 1.0, 0.0, 4)
    px.iloc[149] = np.nan  # no quote the day before the ex-day
    out = ck.ex_day_return(pd.DataFrame({"A": px}), mkt, "A", DAYS[150])
    assert out["prev_date"] == DAYS[148]
    assert out["gap"] == 2
    assert out["ret"] == pytest.approx(px.iloc[150] / px.iloc[148] - 1)
    assert out["mkt_ret"] == pytest.approx(mkt.iloc[150] / mkt.iloc[148] - 1)


def test_at_a_rename_boundary_the_predecessors_last_close_is_the_previous_quote():
    mkt = _market()
    old = _stock(mkt, 1.0, 0.0, 12, drops={})
    new = old.copy()
    old.iloc[151:] = np.nan  # the old ticker ends on day 150 ...
    new.iloc[:151] = np.nan  # ... and the renamed one starts on day 151
    new.iloc[151] = new.iloc[151] * 0.95  # which is the ex-day: -5 %
    adj = pd.DataFrame({"OLD": old, "NEW": new})
    plain = ck.ex_day_return(adj, mkt, "NEW", DAYS[151])
    assert np.isnan(plain["ret"])
    chained = ck.ex_day_return(adj, mkt, "NEW", DAYS[151], predecessors={"NEW": "OLD"})
    assert chained["prev_date"] == DAYS[150] and chained["prev_secid"] == "OLD"
    assert chained["ret"] == pytest.approx(new.iloc[151] / old.iloc[150] - 1)
    # a series with its own previous quote does not borrow
    own = ck.ex_day_return(adj, mkt, "NEW", DAYS[160], predecessors={"NEW": "OLD"})
    assert own["prev_secid"] == "NEW" and own["gap"] == 1


def test_check_events_uses_the_chain_and_reports_the_previous_series():
    mkt = _market()
    old = _stock(mkt, 1.0, 0.004, 13)
    new = old.copy()
    old.iloc[151:] = np.nan
    new.iloc[:151] = np.nan
    new.iloc[151] = new.iloc[151] * 0.95
    adj = pd.DataFrame({"OLD": old, "NEW": new})
    model = ck.market_model(adj, mkt, min_obs=20)
    ev = pd.DataFrame({"secid": ["NEW"], "ex_date": [DAYS[151]], "amount": [0.05 * old.iloc[150]], "currency": ["RUB"]})
    out = ck.check_events(ev, adj, mkt, model, predecessors={"NEW": "OLD"})
    assert out.loc[0, "prev_secid"] == "OLD" and out.loc[0, "drop_status"] == "ok"
    assert ck.check_events(ev, adj, mkt, model).loc[0, "drop_status"] == "no_quote"


def test_ex_day_return_is_nan_without_a_quote_or_across_a_long_gap():
    mkt = _market()
    px = _stock(mkt, 1.0, 0.0, 5)
    px.iloc[150] = np.nan
    assert np.isnan(ck.ex_day_return(pd.DataFrame({"A": px}), mkt, "A", DAYS[150])["ret"])
    px = _stock(mkt, 1.0, 0.0, 5)
    px.iloc[140:150] = np.nan
    assert np.isnan(ck.ex_day_return(pd.DataFrame({"A": px}), mkt, "A", DAYS[150], max_gap=5)["ret"])
    first = _stock(mkt, 1.0, 0.0, 5)
    first.iloc[:150] = np.nan
    assert np.isnan(ck.ex_day_return(pd.DataFrame({"A": first}), mkt, "A", DAYS[150])["ret"])
    assert np.isnan(ck.ex_day_return(pd.DataFrame({"A": px}), mkt, "A", TS("1999-01-01"))["ret"])


# --------------------------------------------------------------------------------------------
# Event checks
# --------------------------------------------------------------------------------------------


def _panel_with_events():
    mkt = _market()
    drop = {120: -0.05, 180: -0.05, 240: -0.05, 300: -0.001}
    px = _stock(mkt, 1.0, 0.008, 6, drops=drop)
    adj = pd.DataFrame({"A": px})
    model = ck.market_model(adj, mkt)
    ev = pd.DataFrame({
        "secid": ["A"] * 6,
        "ex_date": [DAYS[120], DAYS[180], DAYS[240], DAYS[300], DAYS[120], TS("2019-06-03")],
        "amount": [px.iloc[119] * 0.05, px.iloc[179] * 0.05, px.iloc[239] * 0.5,
                   px.iloc[299] * 0.001, 1.0, 1.0],
        "currency": ["RUB", "RUB", "RUB", "RUB", "USD", "RUB"],
    })
    return adj, mkt, model, ev


def test_check_events_classifies_ok_outlier_weak_non_rub_and_no_quote():
    adj, mkt, model, ev = _panel_with_events()
    out = ck.check_events(ev, adj, mkt, model)
    assert out["drop_status"].tolist() == ["ok", "ok", "outlier", "weak", "non_rub", "no_quote"]
    assert out.loc[0, "dividend_yield"] == pytest.approx(0.05, rel=1e-6)
    assert out.loc[0, "abn_return"] < -0.03
    assert abs(out.loc[0, "z"]) < 3
    assert out.loc[2, "z"] > 4  # a 50 % "dividend" that the price did not pay
    assert out.loc[0, "capture"] == pytest.approx(-out.loc[0, "abn_return"] / 0.05)
    assert np.isnan(out.loc[4, "z"]) and np.isnan(out.loc[5, "z"])


def test_check_events_borderline_between_the_two_thresholds():
    adj, mkt, model, ev = _panel_with_events()
    sigma = model.loc["A", "sigma"]
    one = ev.iloc[[0]].copy()
    abn = ck.check_events(one, adj, mkt, model).loc[0, "abn_return"]
    # choose the amount so that z lands at 3.5
    y = -abn + 3.5 * sigma
    one["amount"] = y * adj["A"].iloc[119]
    out = ck.check_events(one, adj, mkt, model)
    assert out.loc[0, "z"] == pytest.approx(3.5, abs=1e-6)
    assert out.loc[0, "drop_status"] == "borderline"


def test_check_events_converts_the_amount_to_the_adjusted_basis_of_a_later_split():
    mkt = _market()
    split_day = DAYS[200]
    # value path in the basis of the last date; before the split one share traded at ten times that
    adj_series = _stock(mkt, 1.0, 0.0, 7, start=100.0, drops={150: -0.05})
    traded_prev = adj_series.iloc[149] * 10  # as-traded close the day before the ex-day
    ev = pd.DataFrame({"secid": ["S"], "ex_date": [DAYS[150]], "amount": [traded_prev * 0.05],
                       "currency": ["RUB"]})
    splits = pd.DataFrame({"secid": ["S"], "tradedate": [split_day], "before": [1], "after": [10],
                           "applied": [True]})
    adjp = pd.DataFrame({"S": adj_series})
    out = ck.check_events(ev, adjp, mkt, ck.market_model(adjp, mkt), splits=splits, last_date=DAYS[-1])
    assert out.loc[0, "amount_adj"] == pytest.approx(traded_prev * 0.05 / 10)
    assert out.loc[0, "dividend_yield"] == pytest.approx(0.05, rel=1e-6)


def test_unapplied_splits_are_not_applied_to_the_amount():
    mkt = _market()
    close = _stock(mkt, 1.0, 0.0, 8, start=1000.0, drops={150: -0.05})
    ev = pd.DataFrame({"secid": ["S"], "ex_date": [DAYS[150]], "amount": [close.iloc[149] * 0.05],
                       "currency": ["RUB"]})
    splits = pd.DataFrame({"secid": ["S"], "tradedate": [TS("2030-01-01")], "before": [1], "after": [10],
                           "applied": [False]})
    adjp = pd.DataFrame({"S": close})
    out = ck.check_events(ev, adjp, mkt, ck.market_model(adjp, mkt), splits=splits, last_date=DAYS[-1])
    assert out.loc[0, "dividend_yield"] == pytest.approx(0.05, rel=1e-6)


# --------------------------------------------------------------------------------------------
# Scale and restatement
# --------------------------------------------------------------------------------------------


def _many_events(scale):
    mkt = _market(seed=9)
    positions = list(range(40, 380, 40))
    px = _stock(mkt, 1.0, 0.006, 10, drops={p: -0.04 for p in positions})
    adj = pd.DataFrame({"A": px})
    ev = pd.DataFrame({"secid": "A", "ex_date": [DAYS[p] for p in positions],
                       "amount": [px.iloc[p - 1] * 0.04 * scale for p in positions], "currency": "RUB"})
    return adj, mkt, ev


@pytest.mark.parametrize("scale, status", [(1.0, "ok"), (0.9, "ok"), (10.0, "scale_suspect"),
                                           (0.1, "scale_suspect")])
def test_scale_check_flags_a_tenfold_error_in_either_direction(scale, status):
    # amounts ten times too small look "weak" by yield alone; the unexplained drop gives them away
    adj, mkt, ev = _many_events(scale)
    checked = ck.check_events(ev, adj, mkt, ck.market_model(adj, mkt))
    table = ck.scale_check(checked)
    assert table.loc["A", "status"] == status


def test_scale_check_counts_a_large_declared_dividend_that_the_price_never_paid():
    mkt = _market(seed=9)
    positions = list(range(40, 380, 40))
    px = _stock(mkt, 1.0, 0.006, 11)  # no drops at all
    adj = pd.DataFrame({"A": px})
    ev = pd.DataFrame({"secid": "A", "ex_date": [DAYS[p] for p in positions],
                       "amount": [px.iloc[p - 1] * 0.05 for p in positions], "currency": "RUB"})
    checked = ck.check_events(ev, adj, mkt, ck.market_model(adj, mkt))
    out = ck.scale_check(checked)
    assert out.loc["A", "status"] == "scale_suspect"
    assert out.loc["A", "median_capture"] < 0.33


def test_scale_check_needs_enough_informative_events():
    adj, mkt, ev = _many_events(1.0)
    checked = ck.check_events(ev.iloc[:2], adj, mkt, ck.market_model(adj, mkt))
    assert ck.scale_check(checked, min_events=3).loc["A", "status"] == "insufficient"


def test_restatement_verdict_prefers_the_hypothesis_that_fits_the_drop():
    adj, mkt, ev = _many_events(1.0)
    # Cbonds shows the amount 100 times too small; the restated amount is the declared one
    ev["cbonds_amount"] = ev["amount"] / 100
    ev["restatement_factor"] = 100.0
    verdict = ck.restatement_verdict(ev, adj, mkt, ck.market_model(adj, mkt))
    assert verdict.loc["A", "verdict"] == "restated"
    assert verdict.loc["A", "n_events"] == len(ev)
    assert verdict.loc["A", "sum_z2_restated"] < verdict.loc["A", "sum_z2_raw"]
    # Cbonds amount already on the declared basis: the factor must not have been applied
    ev2 = ev.assign(cbonds_amount=ev["amount"], amount=ev["amount"] * 100)
    assert ck.restatement_verdict(ev2, adj, mkt, ck.market_model(adj, mkt)).loc["A", "verdict"] == "raw"


@pytest.mark.parametrize(
    "z_restated, z_raw, cap_restated, cap_raw, expected",
    [
        (4.0, 200.0, 0.8, 80.0, "restated"),      # the sums decide
        (200.0, 4.0, 80.0, 0.8, "raw"),
        (2.07, 2.71, 0.92, 9.19, "restated"),     # T: weak yields, close sums; the captures decide
        (2.71, 2.07, 9.19, 0.92, "raw"),
        (2.07, 2.71, 0.5, 5.0, "inconclusive"),   # |log10| 0.30 against 0.70: short of the 0.5 margin
        (2.0, 2.2, 1.0, 1.5, "inconclusive"),     # nothing separates the hypotheses
        (2.0, 2.2, np.nan, 9.0, "inconclusive"),  # no usable capture
        (2.0, 2.2, -0.4, -4.0, "inconclusive"),   # negative medians (noise): the logarithm is undefined
    ],
)
def test_restatement_decision(z_restated, z_raw, cap_restated, cap_raw, expected):
    assert ck.restatement_decision(z_restated, z_raw, cap_restated, cap_raw) == expected


def test_restatement_verdict_ignores_events_that_were_not_restated():
    adj, mkt, ev = _many_events(1.0)
    ev["cbonds_amount"] = ev["amount"]
    ev["restatement_factor"] = 1.0
    assert ck.restatement_verdict(ev, adj, mkt, ck.market_model(adj, mkt)).empty


def test_placebo_exceedance_counts_false_alarms_outside_the_event_days():
    adj, mkt, model, ev = _panel_with_events()
    plain = ck.placebo_exceedance(adj, mkt, model)
    clean = ck.placebo_exceedance(adj, mkt, model, exclude=ev[ev["currency"] == "RUB"].iloc[:4])
    assert plain["n_days"] > clean["n_days"]
    # the planted 5 % drops are 6 sigma events: leaving them out removes the exceedances
    assert plain["share_below_minus_outlier"] > clean["share_below_minus_outlier"]
    assert clean["share_abs_above_outlier"] < 0.01


# --------------------------------------------------------------------------------------------
# Index weights and the rebuilt dividend flow
# --------------------------------------------------------------------------------------------


def _membership():
    return pd.DataFrame({
        "date": [DAYS[0]] * 2 + [DAYS[100]] * 2,
        "secid": ["A", "B", "A", "B"],
        "weight": [0.6, 0.4, 0.5, 0.5],
    })


def test_weights_drift_with_prices_between_snapshots():
    adj = pd.DataFrame({"A": 100.0, "B": 100.0}, index=DAYS)
    adj.loc[DAYS[10]:, "A"] = 200.0  # A doubles on day 10
    w = ck.drift_weights(_membership(), adj, DAYS)
    assert w.loc[DAYS[5], "A"] == pytest.approx(0.6)
    assert w.loc[DAYS[10], "A"] == pytest.approx(1.2 / 1.6)  # 0.6*2 / (0.6*2 + 0.4)
    assert w.loc[DAYS[10]].sum() == pytest.approx(1.0)
    # the next snapshot resets the weights
    assert w.loc[DAYS[100], "A"] == pytest.approx(0.5)


def test_weights_are_zero_before_the_first_snapshot_and_for_non_members():
    adj = pd.DataFrame({"A": 100.0, "B": 100.0, "C": 100.0}, index=DAYS)
    m = pd.DataFrame({"date": [DAYS[3]] * 2, "secid": ["A", "B"], "weight": [0.6, 0.4]})
    w = ck.drift_weights(m, adj, DAYS)
    assert (w.loc[DAYS[2]] == 0).all()
    assert w.loc[DAYS[3], "A"] == pytest.approx(0.6)
    assert (w["C"] == 0).all()


def test_rebuilt_flow_is_weight_times_yield_on_the_ex_day():
    close = pd.DataFrame({"A": 100.0, "B": 50.0}, index=DAYS)
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=DAYS)
    ev = pd.DataFrame({
        "secid": ["A", "B", "A"], "ex_date": [DAYS[20], DAYS[20], DAYS[30]],
        "amount": [5.0, 1.0, 2.0], "currency": ["RUB", "RUB", "USD"],
    })
    flow, contrib = ck.rebuild_index_flow(ev, close, weights, DAYS)
    assert flow.loc[DAYS[20]] == pytest.approx(0.6 * 0.05 + 0.4 * 0.02)
    assert flow.loc[DAYS[30]] == 0.0  # a USD event cannot enter without a rate
    assert flow.drop(DAYS[20]).abs().sum() == 0.0
    assert len(contrib) == 2


def test_rebuilt_flow_uses_the_weight_at_the_previous_close_not_at_the_ex_day():
    close = pd.DataFrame({"A": 100.0}, index=DAYS)
    weights = pd.DataFrame({"A": 0.10}, index=DAYS)
    weights.loc[DAYS[20]:, "A"] = 0.50  # the weight jumps on the ex-day itself (rebalancing)
    ev = pd.DataFrame({"secid": ["A"], "ex_date": [DAYS[20]], "amount": [5.0], "currency": ["RUB"]})
    flow, contrib = ck.rebuild_index_flow(ev, close, weights, DAYS)
    assert flow.loc[DAYS[20]] == pytest.approx(0.10 * 0.05)
    assert contrib.loc[0, "weight"] == pytest.approx(0.10)


def test_rebuilt_flow_uses_the_last_close_when_the_previous_session_has_no_quote():
    close = pd.DataFrame({"A": 100.0}, index=DAYS)
    close.loc[DAYS[19], "A"] = np.nan
    close.loc[DAYS[18], "A"] = 80.0
    weights = pd.DataFrame({"A": 1.0}, index=DAYS)
    ev = pd.DataFrame({"secid": ["A"], "ex_date": [DAYS[20]], "amount": [4.0], "currency": ["RUB"]})
    flow, _ = ck.rebuild_index_flow(ev, close, weights, DAYS)
    assert flow.loc[DAYS[20]] == pytest.approx(4.0 / 80.0)


# --------------------------------------------------------------------------------------------
# Comparing the flows
# --------------------------------------------------------------------------------------------


def _flows():
    obs = pd.Series(0.0, index=DAYS[:60])
    reb = pd.Series(0.0, index=DAYS[:60])
    obs.iloc[10], reb.iloc[10] = 0.010, 0.010            # exact
    obs.iloc[20], reb.iloc[23] = 0.008, 0.008            # booked three sessions later
    obs.iloc[30] = 0.006                                 # missing from the rebuild
    reb.iloc[40] = 0.004                                 # in the rebuild, not in the index
    return obs, reb


def test_flow_metrics_by_block():
    obs, reb = _flows()
    m = ck.flow_metrics(obs, reb, blocks=(1, 5))
    assert m.loc[1, "sum_obs"] == pytest.approx(0.024)
    assert m.loc[1, "sum_rebuilt"] == pytest.approx(0.022)
    assert m.loc[1, "ratio_of_sums"] == pytest.approx(0.022 / 0.024)
    # daily: only the exact event is explained; L1 error = 0.008*2 + 0.006 + 0.004
    assert m.loc[1, "l1_share"] == pytest.approx(1 - 0.026 / 0.024)
    # blocks of five sessions absorb the shift of three sessions (sessions 20 and 23 share a block)
    assert m.loc[5, "l1_share"] > m.loc[1, "l1_share"]
    assert m.loc[5, "l1_share"] == pytest.approx(1 - 0.010 / 0.024)


def test_flow_metrics_ignore_days_without_an_observation():
    obs, reb = _flows()
    obs.iloc[30] = np.nan
    m = ck.flow_metrics(obs, reb, blocks=(1,))
    assert m.loc[1, "sum_obs"] == pytest.approx(0.018)
    assert m.loc[1, "n_days"] == 59


def test_flow_by_year_has_one_row_per_year():
    idx = pd.bdate_range("2020-06-01", "2021-06-30")
    obs = pd.Series(0.0, index=idx)
    reb = pd.Series(0.0, index=idx)
    obs.loc["2020-07-01"], reb.loc["2020-07-01"] = 0.01, 0.009
    obs.loc["2021-03-01"], reb.loc["2021-03-01"] = 0.02, 0.02
    t = ck.flow_by_year(obs, reb)
    assert t.index.tolist() == [2020, 2021]
    assert t.loc[2020, "ratio"] == pytest.approx(0.9)
    assert t.loc[2021, "ratio"] == pytest.approx(1.0)
    assert t.loc[2021, "l1_share_daily"] == pytest.approx(1.0)


def test_residual_episodes_name_shifts_missing_and_extra_payments():
    obs, reb = _flows()
    ep = ck.residual_episodes(obs, reb, min_abs=1e-3, window=5)
    by_date = ep.set_index("date")
    shift = by_date.loc[DAYS[20]]
    assert shift["kind"] == "shift" and shift["partner_date"] == DAYS[23] and shift["shift_sessions"] == -3
    assert by_date.loc[DAYS[23], "kind"] == "shift"
    assert by_date.loc[DAYS[30], "kind"] == "missing"
    assert by_date.loc[DAYS[40], "kind"] == "extra"
    assert DAYS[10] not in by_date.index  # explained exactly: not an episode


def test_residual_episodes_do_not_pair_events_that_differ_in_size():
    obs = pd.Series(0.0, index=DAYS[:30])
    reb = pd.Series(0.0, index=DAYS[:30])
    obs.iloc[10] = 0.010
    reb.iloc[12] = 0.002
    ep = ck.residual_episodes(obs, reb, min_abs=1e-3, window=5, rel_tol=0.35)
    assert sorted(ep["kind"]) == ["extra", "missing"]
    assert ep["partner_date"].isna().all()


def test_balanced_clusters_of_residuals_are_timing_shifts_even_when_split_over_days():
    # the index booked 0.006 on one session; the rule put it on two other sessions (0.003 + 0.0028)
    obs = pd.Series(0.0, index=DAYS[:40])
    reb = pd.Series(0.0, index=DAYS[:40])
    obs.iloc[10] = 0.006
    reb.iloc[12], reb.iloc[13] = 0.003, 0.0028
    ep = ck.residual_episodes(obs, reb, min_abs=1e-3, window=5, rel_tol=0.35)
    assert ep["kind"].tolist() == ["shift_cluster"] * 3
    assert ep["cluster_id"].nunique() == 1
    # an unbalanced cluster keeps the sign names
    reb.iloc[12], reb.iloc[13] = 0.001, 0.0
    ep = ck.residual_episodes(obs, reb, min_abs=5e-4, window=5, rel_tol=0.35)
    assert set(ep["kind"]) <= {"missing", "extra"}


# --------------------------------------------------------------------------------------------
# Event status against the index and leads for missing payments
# --------------------------------------------------------------------------------------------


def _contrib(*rows):
    return pd.DataFrame(rows, columns=["secid", "ex_date", "weight", "prev_close", "amount", "contribution"])


def test_event_index_status_names_confirmed_shifted_inconsistent_and_untestable():
    obs = pd.Series(0.0, index=DAYS[:60])
    reb = pd.Series(0.0, index=DAYS[:60])
    contrib = _contrib(
        ("A", DAYS[10], 0.5, 100.0, 4.0, 0.020),    # the index paid it: confirmed
        ("B", DAYS[20], 0.5, 100.0, 4.0, 0.020),    # booked three sessions later: timing shift
        ("C", DAYS[30], 0.5, 100.0, 4.0, 0.020),    # the index paid almost nothing: inconsistent
        ("D", DAYS[40], 0.01, 100.0, 1.0, 0.0001),  # too small to see: untestable
    )
    for _, r in contrib.iterrows():
        reb[r["ex_date"]] += r["contribution"]
    obs[DAYS[10]] = 0.019
    obs[DAYS[23]] = 0.020
    obs[DAYS[30]] = 0.002
    obs[DAYS[40]] = 0.0001
    ep = ck.residual_episodes(obs, reb, min_abs=1e-3, window=5)
    status = ck.event_index_status(contrib, obs, reb, ep, min_contribution=3e-4)
    assert status.tolist() == ["confirmed", "timing_shift", "inconsistent", "untestable"]


def test_a_small_event_on_a_large_day_is_not_called_inconsistent_for_a_modest_residual():
    # the day pays 1.77 %; a 3 bp gap (1.7 % of the day) cannot be pinned on the 4 bp event
    obs = pd.Series(0.0, index=DAYS[:30])
    reb = pd.Series(0.0, index=DAYS[:30])
    reb[DAYS[10]] = 0.0177
    obs[DAYS[10]] = 0.0174
    contrib = _contrib(("BIG", DAYS[10], 0.5, 100.0, 17.0, 0.0173), ("SMALL", DAYS[10], 0.01, 100.0, 4.0, 0.0004))
    ep = ck.residual_episodes(obs, reb, min_abs=1e-4, window=3)
    status = ck.event_index_status(contrib, obs, reb, ep, min_contribution=3e-4, tol=0.5, noise=0.05)
    assert status.tolist() == ["confirmed", "confirmed"]
    strict = ck.event_index_status(contrib, obs, reb, ep, min_contribution=3e-4, tol=0.5, noise=0.0)
    assert strict.tolist()[1] == "inconsistent"
    # a gap that is a large part of the day is still a contradiction
    obs[DAYS[10]] = 0.0050
    ep = ck.residual_episodes(obs, reb, min_abs=1e-4, window=3)
    assert ck.event_index_status(contrib, obs, reb, ep, noise=0.05).tolist() == ["inconsistent", "inconsistent"]


def test_missing_event_leads_rank_members_by_the_fit_of_their_ex_day_drop():
    days = DAYS[:300]
    full = _market()
    mkt = full.iloc[:300]
    prices = pd.DataFrame({
        "A": _stock(full, 1.0, 0.006, 21, drops={200: -0.04}).iloc[:300],  # fell 4 % on the day
        "B": _stock(full, 1.0, 0.006, 22).iloc[:300],
        "C": _stock(full, 1.0, 0.006, 23).iloc[:300],
    })
    model = ck.market_model(prices, mkt)
    weights = pd.DataFrame({"A": 0.5, "B": 0.3, "C": 0.2}, index=days)
    # the index paid 2 % on day 200: A's weight 0.5 -> implied yield 4 %, B's 6.7 %, C's 10 %
    ep = pd.DataFrame({"date": [days[200]], "observed": [0.02], "rebuilt": [0.0], "residual": [0.02],
                       "kind": ["missing"], "partner_date": [pd.NaT], "shift_sessions": [np.nan],
                       "cluster_id": [0]})
    no_events = pd.DataFrame({"secid": [], "ex_date": []})
    leads = ck.missing_event_leads(ep, weights, no_events, prices, mkt, model, days, top=3)
    assert leads["secid"].tolist() == ["A", "B", "C"]
    assert leads.loc[0, "implied_yield"] == pytest.approx(0.04)
    assert leads.loc[0, "plausible"] and not leads.loc[2, "plausible"]
    # a member that already has an event within three sessions is not a candidate
    near = pd.DataFrame({"secid": ["A"], "ex_date": [days[198]]})
    again = ck.missing_event_leads(ep, weights, near, prices, mkt, model, days, top=3)
    assert "A" not in again["secid"].tolist()


# --------------------------------------------------------------------------------------------
# Exchange rates implied by the data (for events in a foreign currency)
# --------------------------------------------------------------------------------------------


def test_implied_fx_from_price_is_the_drop_in_rub_per_unit_of_dividend():
    checked = pd.DataFrame({
        "currency": ["USD", "USD", "RUB", "USD"], "abn_return": [-0.04, 0.01, -0.04, np.nan],
        "prev_close_adj": [1000.0, 1000.0, 1000.0, 1000.0], "amount_adj": [0.5, 0.5, 0.5, 0.5],
    })
    out = ck.implied_fx_from_price(checked)
    assert out.iloc[0] == pytest.approx(80.0)   # 4 % of 1000 = 40 RUB for 0.5 USD
    assert np.isnan(out.iloc[1])                # the price rose: no rate can be implied
    assert np.isnan(out.iloc[2])                # RUB events have no rate
    assert np.isnan(out.iloc[3])


def test_implied_fx_from_flow_uses_the_residual_of_the_day_and_pools_the_events_of_one_currency():
    days = DAYS[:60]
    close = pd.DataFrame({"A": 100.0, "B": 50.0}, index=days)
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=days)
    events = pd.DataFrame({
        "secid": ["A", "B", "A"], "ex_date": [days[20], days[20], days[40]], "amount": [0.5, 0.1, 0.5],
        "currency": ["USD", "USD", "USD"],
    })
    rate = 75.0
    obs = pd.Series(0.0, index=days)
    obs[days[20]] = rate * (0.6 * 0.5 / 100.0 + 0.4 * 0.1 / 50.0)      # both events pay on day 20
    reb = pd.Series(0.0, index=days)                                      # nothing in RUB
    out = ck.implied_fx_from_flow(events, weights, close, obs, reb, days)
    assert out.iloc[0] == pytest.approx(rate) and out.iloc[1] == pytest.approx(rate)
    assert np.isnan(out.iloc[2])  # the index booked nothing on day 40: no rate
    # RUB events have no rate, and members without weight give nothing
    other = events.assign(currency=["RUB", "USD", "USD"])
    assert np.isnan(ck.implied_fx_from_flow(other, weights, close, obs, reb, days).iloc[0])
    nobody = ck.implied_fx_from_flow(events, weights.assign(A=0.0, B=0.0), close, obs, reb, days)
    assert nobody.isna().all()


def test_index_validation_wires_weights_flow_status_and_metrics():
    days = DAYS[:120]
    close = pd.DataFrame({"A": 100.0, "B": 50.0}, index=days)
    adj = close.copy()
    membership = pd.DataFrame({"date": [days[0]] * 2, "secid": ["A", "B"], "weight": [0.7, 0.3]})
    events = pd.DataFrame({
        "secid": ["A", "B", "A"], "ex_date": [days[30], days[60], days[90]], "amount": [5.0, 2.0, 1.0],
        "currency": ["RUB", "RUB", "USD"], "usable_rub": [True, True, False],
    })
    truth = pd.Series(0.0, index=days)
    truth[days[30]] = 0.7 * 0.05
    truth[days[60]] = 0.004  # the index paid a third of the 0.012 that the event promises
    out = ck.index_validation(events, close, adj, membership, truth, days, start=days[1])
    assert out.flow[days[30]] == pytest.approx(0.035)
    assert out.flow[days[90]] == 0.0  # the USD event is not in the RUB rebuild
    assert out.contrib["index_status"].tolist() == ["confirmed", "inconsistent"]
    assert out.metrics.loc[1, "sum_obs"] == pytest.approx(0.035 + 0.004)
    assert out.metrics.loc[1, "sum_rebuilt"] == pytest.approx(0.035 + 0.012)
    assert out.episodes["kind"].tolist() == ["extra"]
