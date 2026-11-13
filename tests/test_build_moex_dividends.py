"""Tests for ``scripts/data/build_moex_dividends.py`` on the synthetic mini-panel.

The script's I/O (reading ``data/raw`` and ``data/processed/moex_v2``, verifying their provenance)
is exercised by the real build; here the assembly of the tables, the per-ticker status, the
RUB-only total-return table, the output files and the provenance record.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cif.data import cbonds
from cif.data import moex_dividend_checks as ck
from cif.data import moex_dividend_registry as reg
from cif.data.provenance import sha256_file, verify_provenance
from tests.dividend_fixtures import cbonds_rows, make_panel
from tests.xlsx_builder import write_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "build_moex_dividends.py"
TS = pd.Timestamp

TABLES = {
    "events", "dividends_rub", "cbonds_audit", "registry_audit", "corrections_audit", "non_rub_events", "price_outliers",
    "ex_date_disagreements", "restatement_verdict", "scale_check", "coverage_rub", "ticker_status",
    "index_flow_daily", "index_flow_by_year", "index_flow_metrics", "index_residual_episodes",
    "index_missing_leads", "tr_index_rub", "registry_leads",
}


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("build_moex_dividends", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_moex_dividends"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def assembly(script):
    return script.assemble(cbonds_rows(), cbonds_rows(), make_panel(), registry_events=(), no_payment=(),
                           config=script.DEFAULT_CONFIG, corrections=())


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def test_assemble_returns_the_documented_tables(assembly):
    assert set(assembly.tables) == TABLES
    for name, frame in assembly.tables.items():
        assert isinstance(frame, pd.DataFrame), name


def test_dividends_rub_is_the_usable_rub_events_in_the_total_return_format(assembly):
    d = assembly.tables["dividends_rub"]
    ev = assembly.tables["events"]
    assert list(d.columns[:5]) == ["secid", "record_date", "ex_date", "amount", "currency"]
    assert (d["currency"] == "RUB").all()
    assert len(d) == int(ev["usable_rub"].sum())
    assert not d.duplicated().any()


def test_non_rub_events_are_listed_for_the_exchange_rate_table(assembly):
    n = assembly.tables["non_rub_events"]
    assert set(n["currency"]) == {"USD"}
    assert len(n) == 2 and list(n["secid"]) == ["SBER", "SBER"]
    assert {"ex_date", "record_date", "amount", "currency", "source_ref", "period", "flags"} <= set(n.columns)
    assert {"weight_prev", "abn_return", "implied_fx_price", "implied_fx_flow"} <= set(n.columns)


def test_price_outliers_table_has_the_index_status(assembly):
    o = assembly.tables["price_outliers"]
    assert set(o["drop_status"]) <= {"outlier", "borderline"}
    assert "index_status" in o.columns and "event_id" in o.columns
    assert len(o) >= 1  # row 15 of the fixture: a 5 % dividend the price never paid


def test_ex_date_disagreements_list_every_event_where_the_vendor_date_differs_from_the_rule(assembly):
    t = assembly.tables["ex_date_disagreements"]
    ev = assembly.tables["events"]
    differs = ev[ev["ex_vendor"].notna() & (ev["ex_vendor"] != ev["ex_rule"])]
    assert len(t) == len(differs)
    assert set(t["ex_basis"]) >= {"price_vendor", "price_rule", "rule_uninformative", "rule_non_rub"}


def test_coverage_table_is_ticker_by_year_with_rub_events(assembly):
    c = assembly.tables["coverage_rub"]
    assert c.index.name == "secid"
    assert c.loc["GMKN"].sum() == 1 and c.loc["TCSG"].sum() == 2


def test_the_events_table_carries_the_index_status_and_contribution(assembly):
    ev = assembly.tables["events"]
    assert {"index_status", "index_contribution"} <= set(ev.columns)
    assert ev["index_status"].isin(["confirmed", "timing_shift", "inconsistent", "untestable", "not_in_flow"]).all()


# --------------------------------------------------------------------------------------------
# Status of every ticker of the universe
# --------------------------------------------------------------------------------------------


def _src():
    return (reg.Source("https://example.org/ar", "annual report: no dividends in the period", "issuer"),)


def test_ticker_status_names_what_is_known_about_every_universe_ticker(script):
    panel = make_panel()
    events = pd.DataFrame({
        "secid": ["SBER", "SBER", "T"], "source": ["cbonds", "cbonds", "registry"], "currency": ["RUB", "USD", "RUB"],
        "ex_date": [TS("2021-03-16"), TS("2021-09-14"), TS("2023-03-14")],
    })
    uni = panel.universe.copy()
    extra = pd.DataFrame({
        "secid": ["NOPAY", "OUT", "LOST"],
        "index_from": [TS("2021-01-01"), pd.NaT, TS("2021-01-01")],
        "index_till": [TS("2023-12-31"), pd.NaT, TS("2023-12-31")],
    })
    panel.universe = pd.concat([uni, extra], ignore_index=True)
    nopay = reg.NoPayment("NOPAY", "2021-01-04..2023-12-29", "no dividends", _src(), "2026-10-08")
    partial = reg.NoPayment("LOST", "2021-01-04..2021-12-31", "no dividends in 2021", _src(), "2026-10-08")
    st = script.ticker_status(panel, events, [nopay, partial]).set_index("secid")
    assert st.loc["SBER", "status"] == "cbonds" and st.loc["SBER", "n_cbonds_rub"] == 1
    assert st.loc["SBER", "n_cbonds_non_rub"] == 1
    assert st.loc["T", "status"] == "registry" and st.loc["T", "n_registry_rub"] == 1
    assert st.loc["NOPAY", "status"] == "no_payment_evidenced"
    assert st.loc["OUT", "status"] == "outside_window"
    assert st.loc["LOST", "status"] == "no_payment_partial"
    assert st.loc["GMKN", "status"] == "unresolved"  # no event and no evidence in this toy example


def test_ticker_status_carries_the_mean_index_weight_of_each_ticker(script):
    panel = make_panel()
    events = pd.DataFrame({"secid": [], "source": [], "currency": [], "ex_date": pd.to_datetime([])})
    st = script.ticker_status(panel, events, []).set_index("secid")
    assert st.loc["SBER", "mean_weight"] == pytest.approx(0.6)   # the fixture's membership: SBER 0.6, GMKN 0.4
    assert st.loc["GMKN", "mean_weight"] == pytest.approx(0.4)
    assert st.loc["NEWC", "mean_weight"] == 0.0                   # never in a snapshot


def test_unresolved_tickers_with_a_small_mean_weight_are_marked_small_and_carry_their_leads(script):
    panel = make_panel()
    events = pd.DataFrame({"secid": [], "source": [], "currency": [], "ex_date": pd.to_datetime([])})
    lead = reg.Lead("NEWC", "a dividend of 0.5 per share in 2022", "the record date was not read", _src(), "2026-10-08")
    st = script.ticker_status(panel, events, [], leads=[lead]).set_index("secid")
    assert st.loc["GMKN", "status"] == "unresolved"  # mean weight 0.4: not small
    assert st.loc["NEWC", "status"] == "unresolved_small" and st.loc["NEWC", "mean_weight"] == 0.0
    assert st.loc["NEWC", "n_leads"] == 1 and "0.5 per share" in st.loc["NEWC", "lead_summary"]
    assert st.loc["GMKN", "n_leads"] == 0 and st.loc["GMKN", "lead_summary"] == ""
    # the threshold is a parameter: with a weight limit above 0.4 GMKN is small too
    big = script.ticker_status(panel, events, [], leads=[lead], small_weight=0.5).set_index("secid")
    assert big.loc["GMKN", "status"] == "unresolved_small"


def test_a_ticker_with_events_or_evidence_is_never_marked_small(script):
    panel = make_panel()
    events = pd.DataFrame({"secid": ["NEWC"], "source": ["registry"], "currency": ["RUB"],
                           "ex_date": [TS("2022-06-02")]})
    st = script.ticker_status(panel, events, []).set_index("secid")
    assert st.loc["NEWC", "status"] == "registry"


def test_years_without_events_lists_the_years_of_membership_with_no_rub_event(script):
    panel = make_panel()
    events = pd.DataFrame({"secid": ["SBER"], "source": ["cbonds"], "currency": ["RUB"],
                           "ex_date": [TS("2021-03-16")]})
    st = script.ticker_status(panel, events, []).set_index("secid")
    assert st.loc["SBER", "years_without_events"] == "2022,2023"


# --------------------------------------------------------------------------------------------
# Total return (RUB events only)
# --------------------------------------------------------------------------------------------


def test_the_total_return_table_reinvests_rub_events_and_leaves_the_rest_out(script, assembly):
    panel = make_panel()
    tr = assembly.tables["tr_index_rub"]
    ev = assembly.tables["events"]
    assert list(tr.columns) == list(panel.close.columns)
    s = ev[(ev["secid"] == "SBER") & ev["usable_rub"]]
    price_return = panel.adj["SBER"].iloc[-1] / panel.adj["SBER"].iloc[0]
    tr_return = tr["SBER"].iloc[-1] / tr["SBER"].iloc[0]
    assert tr_return > price_return * 1.1  # the 4+ dividends of 4-6 % each
    s_usd = ev[(ev["secid"] == "SBER") & ~ev["usable_rub"]]
    assert len(s_usd) >= 2
    assert assembly.summary["tr_index"]["non_rub_events_excluded"] == int((ev["currency"] != "RUB").sum())
    assert assembly.summary["tr_index"]["events_used"] == int(ev["usable_rub"].sum())
    assert "RUB events only" in assembly.summary["tr_index"]["note"]
    assert len(s) >= 4


# --------------------------------------------------------------------------------------------
# Output and provenance
# --------------------------------------------------------------------------------------------


def test_summary_is_plain_json(assembly):
    text = json.dumps(assembly.summary, allow_nan=False)
    assert "ex_basis" in text and "drop_status" in text
    s = assembly.summary
    assert s["cbonds_rows"]["total"] == 20  # sheet rows 2..21 of the fixture
    assert s["registry"]["n_events"] == 0
    assert "placebo" in s["price_checks"]
    metrics = s["index_flow"]["rub_events"]["metrics"]
    assert set(metrics) == {"1", "5", "21"}
    assert metrics["1"]["ratio_of_sums"] is None  # the fixture's observed flow is zero: the ratio is undefined
    assert metrics["1"]["sum_rebuilt"] > 0


def test_write_outputs_writes_every_table_and_nothing_else(script, assembly, tmp_path):
    out = tmp_path / "moex_dividends"
    script.write_outputs(assembly, out)
    names = {p.stem for p in out.iterdir()}
    assert names == TABLES
    events = pd.read_parquet(out / "events.parquet")
    assert len(events) == len(assembly.tables["events"])
    assert (out / "non_rub_events.csv").read_text(encoding="utf-8").splitlines()[0].startswith("secid")


def test_provenance_lists_inputs_code_counts_and_hashes_of_the_outputs(script, assembly, tmp_path):
    out = tmp_path / "out"
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.xlsx").write_bytes(b"workbook")
    write_table(raw / "b.xlsx", cbonds.CBONDS_HEADER, [])
    script.write_outputs(assembly, out)
    git = {"sha": "b" * 40, "branch": "x", "dirty": False}
    fx_file = tmp_path / "usd_rub.parquet"
    _cbr_file(fx_file)
    fx_record = {"USD": {"path": "usd_rub.parquet", "sha256": sha256_file(fx_file),
                         "provenance": {"path": "usd_rub.parquet", "sha256": sha256_file(fx_file), "dataset": "cbr_fx"}}}
    prov = script.provenance_record(
        assembly, config={**script.DEFAULT_CONFIG, "raw_dir": str(raw), "out_dir": str(out)}, git=git,
        out_dir=out, raw_dir=raw, moex_dir=tmp_path, moex_files=[], cbonds_import={"imported_at": "x"},
        now="2026-10-08T00:00:00+00:00", fx_record=fx_record)
    assert prov["inputs"]["fx"] == fx_record
    assert prov["dataset"] == "moex_dividends" and prov["git"] == git
    assert set(prov["code_sha256"]) >= {"src/cif/data/moex_dividends.py", "src/cif/data/moex_dividend_checks.py",
                                        "src/cif/data/moex_dividend_registry.py", "src/cif/data/cbonds.py",
                                        "scripts/data/build_moex_dividends.py"}
    assert set(prov["files"]["processed"]) == {p.name for p in out.iterdir() if p.name != "provenance.json"}
    assert set(prov["files"]["raw"]) == {"a.xlsx", "b.xlsx"}
    (out / "provenance.json").write_text(json.dumps(prov, allow_nan=False), encoding="utf-8")
    report = verify_provenance(out / "provenance.json", repo_root=tmp_path)
    assert report["ok"], report


def test_a_dirty_tree_is_refused(script):
    with pytest.raises(script.BuildRefused, match="clean"):
        script.build(script.DEFAULT_CONFIG, git={"sha": "a" * 40, "branch": "b", "dirty": True})


def test_all_countries_file_must_extend_the_russian_one_by_foreign_rows_only(script):
    ru = cbonds_rows()
    extra = ru.iloc[[0]].assign(tickers="IC5, 6240", currency="JPY", sheet_row=999)
    allc = pd.concat([ru, extra], ignore_index=True)
    check = script.check_all_countries_file(ru, allc, ["SBER", "GMKN"])
    assert check == {"n_extra_rows": 1, "extra_currencies": {"JPY": 1}, "extra_tickers": ["IC5, 6240"],
                     "extra_rows_in_universe": 0}
    bad = pd.concat([ru, ru.iloc[[0]].assign(sheet_row=1000, tickers="SBER", amount=123.0)], ignore_index=True)
    with pytest.raises(script.BuildRefused, match="universe"):
        script.check_all_countries_file(ru, bad, ["SBER", "GMKN"])
    missing = ru.iloc[1:].reset_index(drop=True)
    with pytest.raises(script.BuildRefused, match="not in the all-countries"):
        script.check_all_countries_file(ru, missing, ["SBER"])


def _flat_fx(rate=75.0):
    return pd.DataFrame({"USD": rate}, index=pd.bdate_range("2021-01-04", "2022-12-30"))


def _assemble(script, panel=None, fx=None):
    return script.assemble(cbonds_rows(), cbonds_rows(), panel or make_panel(), registry_events=(), no_payment=(),
                           config=script.DEFAULT_CONFIG, fx=fx)


def test_an_exchange_rate_table_moves_foreign_events_into_the_rub_total(script):
    a, plain = _assemble(script, fx=_flat_fx()), _assemble(script)
    fx = a.summary["fx"]
    assert fx["provided"] is True and fx["currencies"] == ["USD"]
    assert fx["converted_events"] == 2 and fx["missing_rate_events"] == 0
    assert a.summary["events"]["usable_rub"] == plain.summary["events"]["usable_rub"] + 2
    assert plain.summary["fx"]["provided"] is False and plain.summary["fx"]["converted_events"] == 0
    assert a.summary["tr_index"]["non_rub_events_excluded"] == 0
    assert plain.summary["tr_index"]["non_rub_events_excluded"] == 2


def test_the_summary_states_the_rate_convention(script):
    fx = _assemble(script, fx=_flat_fx()).summary["fx"]
    assert "ex-date" in fx["rate_convention"] and "Bank of Russia" in fx["rate_convention"]
    assert fx["max_age_days"] == 7


def test_with_a_rate_table_the_foreign_events_are_listed_with_the_original_amount_beside_the_rouble_one(script):
    n = _assemble(script, fx=_flat_fx()).tables["non_rub_events"]
    assert len(n) == 2 and set(n["currency"]) == {"USD"} and set(n["fx_status"]) == {"converted"}
    assert (n["amount"] == 1.0).all() and (n["fx_rate"] == 75.0).all()
    assert n["amount_rub"].tolist() == pytest.approx([75.0, 75.0]) and n["fx_date"].notna().all()


def test_without_a_rate_table_the_foreign_events_say_so(script):
    n = _assemble(script).tables["non_rub_events"]
    assert set(n["fx_status"]) == {"no_table"} and n["fx_rate"].isna().all() and n["amount_rub"].isna().all()


def test_an_event_with_no_recent_rate_is_listed_as_awaiting_one(script):
    fx = pd.DataFrame({"USD": 75.0}, index=pd.bdate_range("2021-01-04", "2021-06-30"))
    a = _assemble(script, fx=fx)
    n = a.tables["non_rub_events"]
    assert set(n["fx_status"]) == {"no_rate"} and a.summary["fx"]["missing_rate_events"] == 2


def test_assemble_writes_the_leads_and_counts_them(script):
    lead = reg.Lead("SBER", "a dividend the registry does not hold", "small", _src(), "2026-10-08")
    a = script.assemble(cbonds_rows(), cbonds_rows(), make_panel(), registry_events=(), no_payment=(),
                        config=script.DEFAULT_CONFIG, registry_leads=[lead])
    t = a.tables["registry_leads"]
    assert t["secid"].tolist() == ["SBER"] and t["what"].iloc[0].startswith("a dividend")
    assert a.summary["registry"]["n_leads"] == 1
    plain = _assemble(script)
    assert plain.tables["registry_leads"].empty and plain.summary["registry"]["n_leads"] == 0


def test_the_summary_lists_unresolved_tickers_apart_from_the_small_ones_with_their_weights(script):
    panel = make_panel()
    extra = pd.DataFrame({"secid": ["ZZZ"], "index_from": [TS("2021-01-01")], "index_till": [TS("2023-12-31")]})
    panel.universe = pd.concat([panel.universe, extra], ignore_index=True)
    s = _assemble(script, panel).summary["ticker_status"]
    assert s["small_weight"] == 0.01 and s["unresolved"] == [] and s["unresolved_mean_weight"] == {}
    assert s["unresolved_small"] == {"ZZZ": 0.0} and s["counts"]["unresolved_small"] == 1


def test_dividends_rub_keeps_the_original_amount_rate_and_date_of_converted_events(script):
    d = _assemble(script, fx=_flat_fx()).tables["dividends_rub"]
    assert {"amount_orig", "currency_orig", "fx_rate", "fx_date"} <= set(d.columns)
    c = d[d["currency_orig"].notna()]
    assert len(c) == 2 and (c["currency_orig"] == "USD").all() and (c["amount_orig"] == 1.0).all()
    assert (c["amount"] == 75.0).all() and (c["currency"] == "RUB").all() and (c["fx_rate"] == 75.0).all()
    native = d[d["currency_orig"].isna()]
    assert native[["amount_orig", "fx_rate", "fx_date"]].isna().all().all()


def test_the_total_return_includes_converted_events_and_says_so(script):
    a, plain = _assemble(script, fx=_flat_fx()), _assemble(script)
    note, plain_note = a.summary["tr_index"]["note"], plain.summary["tr_index"]["note"]
    assert "converted" in note and "RUB events only" not in note
    assert "RUB events only" in plain_note
    tr, tr_plain = a.tables["tr_index_rub"], plain.tables["tr_index_rub"]
    # two 75 RUB dividends on a share that trades near 100: the converted series must end far higher
    assert tr["SBER"].iloc[-1] > 1.5 * tr_plain["SBER"].iloc[-1]


def test_the_rate_implied_by_the_index_flow_is_compared_with_the_rate_used(script):
    panel = make_panel()
    days = panel.close.index
    w = ck.drift_weights(panel.membership, panel.adj, days)
    ex = TS("2021-09-14")  # row 9 of the fixture: SBER, 1.0 USD
    i = days.get_loc(ex)
    prev_close = panel.close["SBER"].ffill(limit=5).shift(1).iloc[i]
    panel.index_flow.iloc[i] = w["SBER"].iloc[i - 1] * (80.0 * 1.0) / prev_close  # the index paid 1 USD at 80 RUB
    a = _assemble(script, panel, fx=_flat_fx(75.0))
    n = a.tables["non_rub_events"]
    row = n[n["ex_date"] == ex].iloc[0]
    assert row["implied_fx_flow"] == pytest.approx(80.0, rel=1e-9)
    assert row["ratio_flow"] == pytest.approx(80.0 / 75.0, rel=1e-9)
    flow = a.summary["fx"]["rate_check"]["flow"]
    assert flow["n"] == 1 and flow["median_ratio"] == pytest.approx(80.0 / 75.0, rel=1e-9)
    assert flow["share_within_10pct"] == 1.0
    assert a.summary["fx"]["rate_check"]["flow_testable"] == flow  # the event is large enough to show in the index
    other = n[n["ex_date"] != ex].iloc[0]
    assert pd.isna(other["implied_fx_flow"]) and pd.isna(other["ratio_flow"])


def test_the_rate_check_of_the_summary_leaves_out_events_too_small_to_show_in_the_index(script):
    panel = make_panel()
    panel.membership["weight"] = [1e-4, 1.0 - 1e-4]  # SBER weighs nothing: its dividend is lost in the noise
    days = panel.close.index
    w = ck.drift_weights(panel.membership, panel.adj, days)
    ex = TS("2021-09-14")
    i = days.get_loc(ex)
    prev_close = panel.close["SBER"].ffill(limit=5).shift(1).iloc[i]
    panel.index_flow.iloc[i] = w["SBER"].iloc[i - 1] * 80.0 / prev_close
    a = _assemble(script, panel, fx=_flat_fx(75.0))
    check = a.summary["fx"]["rate_check"]
    n = a.tables["non_rub_events"]
    assert n.loc[n["ex_date"] == ex, "index_status"].iloc[0] == "untestable"
    assert check["flow"]["n"] == 1 and check["flow_testable"]["n"] == 0 and check["flow_testable"]["median_ratio"] is None


def test_the_price_implied_rate_is_computed_on_the_amount_as_declared(script):
    a = _assemble(script, fx=_flat_fx())
    n = a.tables["non_rub_events"].set_index("source_ref")
    ev = a.tables["events"].set_index("source_ref").loc[n.index]
    expected = -ev["abn_return"] * ev["prev_close_adj"] / ev["amount_orig"]  # RUB per USD that the drop implies
    assert expected.notna().all() and (expected > 0).all()
    assert n["implied_fx_price"].tolist() == pytest.approx(expected.tolist())
    assert n["ratio_price"].tolist() == pytest.approx((expected / 75.0).tolist())


def test_the_flow_check_leaves_converted_events_out_of_the_flow_it_compares_with(script):
    # the check solves for the rate from the residual of the NATIVE rebuilt flow; if the converted event were
    # in it, the residual would be gone and no rate could be implied
    panel = make_panel()
    days = panel.close.index
    w = ck.drift_weights(panel.membership, panel.adj, days)
    ex = TS("2021-09-14")
    i = days.get_loc(ex)
    prev_close = panel.close["SBER"].ffill(limit=5).shift(1).iloc[i]
    panel.index_flow.iloc[i] = w["SBER"].iloc[i - 1] * 75.0 / prev_close
    a = _assemble(script, panel, fx=_flat_fx(75.0))
    # the full rebuilt flow now contains the converted event: its day has no residual left
    daily = a.tables["index_flow_daily"].set_index("date")
    assert daily.loc[ex, "residual"] == pytest.approx(0.0, abs=1e-12)
    n = a.tables["non_rub_events"]
    assert n.loc[n["ex_date"] == ex, "ratio_flow"].iloc[0] == pytest.approx(1.0, rel=1e-9)


def _cbr_file(path, rate=75.0, start="2021-01-04", end="2022-12-30"):
    idx = pd.bdate_range(start, end, name="date")
    pd.DataFrame({"rate": rate, "nominal": 1}, index=idx).to_parquet(path)


def test_load_fx_reads_the_series_of_each_currency(script, tmp_path):
    usd, eur = tmp_path / "usd_rub.parquet", tmp_path / "eur_rub.parquet"
    pd.DataFrame({"rate": [73.9, 74.3], "nominal": [1, 1]},
                 index=pd.DatetimeIndex(["2021-01-04", "2021-01-05"], name="date")).to_parquet(usd)
    pd.DataFrame({"rate": [90.0], "nominal": [1]}, index=pd.DatetimeIndex(["2021-01-05"], name="date")).to_parquet(eur)
    fx = script.load_fx({"usd": usd, "EUR": eur})
    assert list(fx.columns) == ["EUR", "USD"] and fx.index[0] == TS("2021-01-04")
    assert fx["USD"].tolist() == [73.9, 74.3] and fx["EUR"].dropna().tolist() == [90.0]
    assert script.load_fx(None) is None and script.load_fx({}) is None


def test_load_fx_dates_the_rates_at_midnight(script, tmp_path):
    path = tmp_path / "usd_rub.parquet"
    pd.DataFrame({"rate": [73.9]}, index=pd.DatetimeIndex(["2021-01-04 12:00"], name="date")).to_parquet(path)
    assert script.load_fx({"USD": path}).index.tolist() == [TS("2021-01-04")]


def test_load_fx_refuses_repeated_dates_and_non_positive_rates(script, tmp_path):
    dup = tmp_path / "dup.parquet"
    pd.DataFrame({"rate": [74.0, 75.0]}, index=pd.DatetimeIndex(["2021-01-04"] * 2, name="date")).to_parquet(dup)
    with pytest.raises(script.BuildRefused, match="repeated"):
        script.load_fx({"USD": dup})
    zero = tmp_path / "zero.parquet"
    pd.DataFrame({"rate": [74.0, 0.0]}, index=pd.DatetimeIndex(["2021-01-04", "2021-01-05"], name="date")).to_parquet(zero)
    with pytest.raises(script.BuildRefused, match="positive"):
        script.load_fx({"USD": zero})


def _fx_dataset(tmp_path, rate=75.0):
    d = tmp_path / "data" / "processed" / "cbr_fx"
    d.mkdir(parents=True)
    _cbr_file(d / "usd_rub.parquet", rate=rate)
    prov = {"dataset": "cbr_fx", "built_at": "2026-10-08T17:55:36+00:00",
            "git": {"sha": "a" * 40, "branch": "x", "dirty": False},
            "files": {"raw": {}, "processed": {"usd_rub.parquet": {"sha256": sha256_file(d / "usd_rub.parquet")}}}}
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return d


FX_CONFIG = {"fx_files": {"USD": "data/processed/cbr_fx/usd_rub.parquet"}}


def test_fx_inputs_verify_the_dataset_and_record_its_hashes(script, tmp_path):
    d = _fx_dataset(tmp_path)
    fx, record = script.fx_inputs(FX_CONFIG, repo=tmp_path)
    assert list(fx.columns) == ["USD"]
    usd = record["USD"]
    assert usd["path"] == "data/processed/cbr_fx/usd_rub.parquet" and usd["sha256"] == sha256_file(d / "usd_rub.parquet")
    prov = usd["provenance"]
    assert prov["path"] == "data/processed/cbr_fx/provenance.json" and prov["sha256"] == sha256_file(d / "provenance.json")
    assert prov["dataset"] == "cbr_fx" and prov["git"]["sha"] == "a" * 40
    assert script.fx_inputs({"fx_files": None}, repo=tmp_path) == (None, {})
    assert script.fx_inputs({}, repo=tmp_path) == (None, {})


def test_fx_inputs_refuse_a_dataset_that_does_not_verify(script, tmp_path):
    d = _fx_dataset(tmp_path)
    _cbr_file(d / "usd_rub.parquet", rate=80.0)  # the file changed after the provenance was written
    with pytest.raises(script.BuildRefused, match="provenance"):
        script.fx_inputs(FX_CONFIG, repo=tmp_path)


def test_fx_inputs_refuse_a_file_that_the_provenance_does_not_cover(script, tmp_path):
    d = _fx_dataset(tmp_path)
    _cbr_file(d / "other.parquet")
    with pytest.raises(script.BuildRefused, match="not covered"):
        script.fx_inputs({"fx_files": {"USD": "data/processed/cbr_fx/other.parquet"}}, repo=tmp_path)


def test_fx_inputs_refuse_a_missing_dataset(script, tmp_path):
    with pytest.raises(script.BuildRefused, match="provenance"):
        script.fx_inputs(FX_CONFIG, repo=tmp_path)


def test_the_default_configuration_names_the_cbr_dataset(script):
    assert script.DEFAULT_CONFIG["fx_files"] == {"USD": "data/processed/cbr_fx/usd_rub.parquet"}

