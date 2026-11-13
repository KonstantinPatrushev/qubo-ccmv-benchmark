"""The analysis of the validation M11 (``cif.experiments.e2_m11_analysis``, ``e2_analysis m11``): the published
numbers of Bertsimas & Cory-Wright and how the rows of the stage are read against them (ANALYSIS_PLAN §12, entry of
2026-10-17: per cell the ratio of our median time to the best published one, times beyond the limit set to the limit and
flagged, cells that both sides failed left out and listed, the median of the ratios per table and the counts of the
cells solved).

The published file is checked against a second, hand-typed copy of the 105 times (double entry: the YAML was produced
from the text layer of the PDF page, this copy from the figures read off the page by the orchestrator).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from cif.experiments import e2_analysis, e2_m11, e2_m11_analysis, run
from cif.experiments import e2_m11_analysis as an
from cif.instances.orlib import ORLIB_N
from tests.test_e2_m11 import _cell, _cfg
from tests.test_instances import orlib_text

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "configs" / "E2" / "m11_bcw_published.yaml"
CONFIG = ROOT / "configs" / "E2" / "m11.yaml"
PORTS = (1, 2, 3, 4, 5)
KS = (5, 10, 20)

# (port, k) -> times as read off the page: Table 3 = Algorithm 3 | CPLEX big-M | CPLEX MISOCO (limit 300 s)
HAND_T3 = {
    (1, 5): (0.17, 1.98, 0.03), (1, 10): (0.16, 1.11, 0.01), (1, 20): (0.14, 0.01, 0.03),
    (2, 5): (0.01, ">300", 0.11), (2, 10): (0.01, ">300", 0.12), (2, 20): (0.01, ">300", 0.29),
    (3, 5): (0.01, ">300", 0.38), (3, 10): (0.01, ">300", 0.41), (3, 20): (0.02, ">300", 0.11),
    (4, 5): (0.03, ">300", 0.41), (4, 10): (0.02, ">300", 2.74), (4, 20): (0.03, ">300", 0.38),
    (5, 5): (0.15, ">300", 11.17), (5, 10): (0.02, ">300", 3.04), (5, 20): (0.03, ">300", 2.88),
}
# Table 4 = Algorithm 3 | Algorithm 3 + in-out | CPLEX big-M | CPLEX MISOCO (limit 3600 s)
HAND_T4 = {
    (1, 5): (0.22, 0.23, 9.32, 0.83), (1, 10): (0.20, 0.25, 1970, 0.84), (1, 20): (0.16, 0.16, 258.4, 0.05),
    (2, 5): (48.29, 31.47, ">3600", 91.98), (2, 10): (807.3, 891.97, ">3600", 82.44),
    (2, 20): (10.52, 13.0, ">3600", 24.54),
    (3, 5): (175.2, 151.1, ">3600", 213.3), (3, 10): (">3600", ">3600", ">3600", 531.3),
    (3, 20): (119.5, 60.03, ">3600", 21.32),
    (4, 5): (2690, 2475, ">3600", 2779), (4, 10): (">3600", ">3600", ">3600", ">3600"),
    (4, 20): (1638, 2067, ">3600", 148.9),
    (5, 5): (0.85, 0.40, ">3600", 28.3), (5, 10): (0.60, 0.03, ">3600", 3.33), (5, 20): (0.39, 0.08, ">3600", 115.02),
}
T3_METHODS = ("algorithm3", "cplex_bigm", "cplex_misoco")
T4_METHODS = ("algorithm3", "algorithm3_inout", "cplex_bigm", "cplex_misoco")
LIMIT = {"T3": 300.0, "T4": 3600.0}


def _best(table, port, k):
    """The best printed time of a cell by the hand-typed copy (``None``: every method failed)."""
    times = [float(v) for v in (HAND_T3 if table == "T3" else HAND_T4)[(port, k)] if not isinstance(v, str)]
    return min(times) if times else None


@pytest.fixture(scope="module")
def published() -> an.Published:
    return an.load_published(PUBLISHED)


# -- the published numbers --------------------------------------------------------------------------------


def test_the_published_file_equals_the_hand_typed_copy_of_the_105_times(published):
    for table, hand, methods in (("T3", HAND_T3, T3_METHODS), ("T4", HAND_T4, T4_METHODS)):
        t = published.tables[table]
        assert t.methods == methods and t.time_limit_s == LIMIT[table]
        assert sorted(t.cells) == sorted(hand) and len(t.cells) == 15
        for key, printed in hand.items():
            cell = t.cells[key]
            assert cell.n == ORLIB_N[key[0]]
            for method, value in zip(methods, printed, strict=True):
                time = cell.times[method]
                if isinstance(value, str):  # ">L": not solved within the limit L
                    assert time.censored and time.seconds == float(value.removeprefix(">")) == LIMIT[table]
                else:
                    assert not time.censored and time.seconds == float(value), (table, key, method)


def test_the_raw_yaml_has_the_strings_of_the_paper_for_the_failures():
    raw = yaml.safe_load(PUBLISHED.read_text(encoding="utf-8"))
    cell = next(c for c in raw["tables"]["T3"]["cells"] if (c["port"], c["k"]) == (2, 5))
    assert cell["time_s"]["cplex_bigm"] == ">300" and cell["time_s"]["algorithm3"] == 0.01
    assert raw["source"]["arxiv"] == "1811.00138v5"
    assert "2.9 GHz" in raw["hardware"]["machine"] and raw["hardware"]["threads"] == 1
    assert "CPLEX 12.8" in raw["hardware"]["software"]


def test_the_published_nodes_and_cuts_are_as_printed(published):
    t3, t4 = published.tables["T3"], published.tables["T4"]
    assert t3.cells[(1, 5)].nodes == {"algorithm3": 0, "cplex_bigm": 31640, "cplex_misoco": 0}
    assert t3.cells[(5, 5)].cuts == {"algorithm3": 9} and t3.cells[(4, 10)].nodes["cplex_misoco"] == 3
    assert t4.cells[(2, 5)].nodes["algorithm3"] == 73850 and t4.cells[(4, 10)].nodes["cplex_misoco"] == 30190
    assert t4.cells[(1, 5)].cuts == {"algorithm3": 32, "algorithm3_inout": 19}


def test_the_config_and_the_published_file_describe_the_same_grid(published):
    cfg = run.load_config(CONFIG)
    tables = cfg["params"]["tables"]
    for name in ("T3", "T4"):
        assert tables[name]["time_limit_s"] == published.tables[name].time_limit_s
        assert tables[name]["kappa"] == published.tables[name].kappa
        assert {(c["port"], c["k"]) for c in e2_m11.cells(run.effective_config(cfg)) if c["table"] == name} == set(
            published.tables[name].cells)
    assert cfg["inputs"]["published"] == "configs/E2/m11_bcw_published.yaml"
    assert published.sha256 and len(published.sha256) == 64


@pytest.mark.parametrize(
    ("value", "limit", "expected"),
    [(0.17, 300.0, (0.17, False)), (1970, 3600.0, (1970.0, False)), (">300", 300.0, (300.0, True)),
     (">3600", 3600.0, (3600.0, True)), (" > 300", 300.0, (300.0, True))],
)
def test_a_printed_time_is_a_number_or_a_lower_bound_at_the_limit(value, limit, expected):
    time = an.parse_time(value, limit)
    assert (time.seconds, time.censored) == expected


@pytest.mark.parametrize(("value", "limit"), [(">300", 3600.0), ("fast", 300.0), (None, 300.0), (-1.0, 300.0),
                                              (math.nan, 300.0), (301.0, 300.0), (True, 300.0)])
def test_a_misprinted_time_is_refused(value, limit):
    with pytest.raises(ValueError, match="time"):
        an.parse_time(value, limit)


def test_the_best_published_time_is_the_minimum_over_the_methods_and_a_limit_if_all_failed(published):
    best = published.tables["T3"].cells[(1, 20)].best  # 0.14 | 0.01 | 0.03
    assert (best.seconds, best.censored, best.method) == (0.01, False, "cplex_bigm")
    best = published.tables["T4"].cells[(3, 10)].best  # >3600 | >3600 | >3600 | 531.3
    assert (best.seconds, best.censored, best.method) == (531.3, False, "cplex_misoco")
    best = published.tables["T4"].cells[(4, 10)].best  # all four fail
    assert (best.seconds, best.censored, best.method) == (3600.0, True, None)
    best = published.tables["T4"].cells[(5, 10)].best  # 0.60 | 0.03 | >3600 | 3.33
    assert (best.seconds, best.method) == (0.03, "algorithm3_inout")


def test_a_published_file_that_is_not_one_is_refused(tmp_path):
    raw = yaml.safe_load(PUBLISHED.read_text(encoding="utf-8"))
    path = tmp_path / "p.yaml"

    def write(mutated):
        path.write_text(yaml.safe_dump(mutated), encoding="utf-8")
        return path

    bad = json.loads(json.dumps(raw))
    bad["kind"] = "something-else"
    with pytest.raises(ValueError, match="kind"):
        an.load_published(write(bad))
    bad = json.loads(json.dumps(raw))
    del bad["tables"]["T4"]["cells"][0]
    with pytest.raises(ValueError, match="15"):
        an.load_published(write(bad))
    bad = json.loads(json.dumps(raw))
    bad["tables"]["T3"]["cells"][0]["time_s"].pop("cplex_misoco")
    with pytest.raises(ValueError, match="cplex_misoco"):
        an.load_published(write(bad))
    bad = json.loads(json.dumps(raw))
    bad["tables"]["T3"]["cells"][0]["n"] = 30
    with pytest.raises(ValueError, match="n"):
        an.load_published(write(bad))


# -- our rows ---------------------------------------------------------------------------------------------


def _row(table, port, k, method, seed, wall_s, *, certified=True, **over):
    limit = LIMIT[table]
    row = {
        "cell_key": f"{table}/{port}/{k}/{method}/{seed}", "stage": "m11", "table": table, "port": port,
        "n": ORLIB_N[port], "k": k, "kappa": 1.0 if table == "T3" else 0.0, "return_row": table == "T4",
        "method": {"name": method, "id": "scip-persp-mineig-sym0" if method == "scip" else "oa-highs"},
        "seed": seed, "time_limit_s": limit, "status": "gaplimit" if certified else "timelimit",
        "proved": certified, "certified": certified, "wall_s": wall_s if certified else limit + 0.2,
        "end_s": wall_s, "primal": 0.01 * (1 + port), "dual": 0.01 * (1 + port) * (1 - 5e-5), "abs_gap": 5e-7,
        "rel_gap": 5e-5, "rel_gap_cplex": 5e-5, "feasible": True, "budget_ok": True, "overrun_s": 0.0,
        "return_ok": True if table == "T4" else None, "nodes": 7 if method == "scip" else None,
        "n_cuts": 3 if method == "oa" else None, "r_bar": 0.003 if table == "T4" else None,
        "host": "gpu-07",
    }
    row.update(over)
    return row


def _grid(published, scip=None, oa=None, tables=("T3", "T4")):
    """The rows of the full grid: ``scip(table, port, k)`` (a time, or None = not solved) for the three seeds, ``oa``
    for the one run of T3."""
    rows = []
    for table in tables:
        for (port, k) in sorted(published.tables[table].cells):
            for seed, spread in zip((10000, 10001, 10002), (0.9, 1.0, 1.1), strict=True):
                t = (scip or (lambda *_: 1.0))(table, port, k)
                rows.append(_row(table, port, k, "scip", seed, None if t is None else t * spread,
                                 certified=t is not None))
            if table == "T3":
                t = (oa or (lambda *_: 2.0))(table, port, k)
                rows.append(_row(table, port, k, "oa", 10000, t, certified=t is not None))
    return rows


def test_our_value_of_a_cell_is_the_median_over_the_seeds_and_the_limit_for_the_runs_that_failed():
    rows = [_row("T3", 1, 5, "scip", s, t) for s, t in zip((10000, 10001, 10002), (1.0, 3.0, 2.0), strict=True)]
    value = an.our_time(rows)
    assert (value.seconds, value.censored, value.n_runs, value.n_failed) == (2.0, False, 3, 0)
    rows[1] = _row("T3", 1, 5, "scip", 10001, None, certified=False)  # one of three fails: the median is still a time
    value = an.our_time(rows)
    assert (value.seconds, value.censored, value.n_failed) == (2.0, False, 1)
    rows[2] = _row("T3", 1, 5, "scip", 10002, None, certified=False)  # two of three fail: the median is the limit
    value = an.our_time(rows)
    assert (value.seconds, value.censored, value.n_failed) == (300.0, True, 2)


def test_the_median_is_a_bound_when_half_of_the_runs_or_more_failed():
    ok = [_row("T3", 1, 5, "scip", 10000 + i, 1.0 + i) for i in range(2)]
    bad = [_row("T3", 1, 5, "scip", 10002 + i, None, certified=False) for i in range(2)]
    assert an.our_time(ok + bad).censored is True  # 2 of 4
    assert an.our_time(ok + bad[:1]).censored is False  # 1 of 3
    assert an.our_time(ok[:1]).censored is False and an.our_time(bad[:1]).censored is True  # one run


def test_a_run_that_proved_but_ended_after_the_limit_counts_as_a_failure():
    late = _row("T3", 1, 5, "scip", 10000, 300.5, certified=False, proved=True, status="gaplimit")
    value = an.our_time([late])
    assert value.censored and value.seconds == 300.0


def test_the_ratio_of_a_cell_and_its_bounds(published):
    cell = published.tables["T3"].cells[(1, 20)]  # best published: 0.01 s (CPLEX big-M)
    solved = an.cell_ratio(an.Time(0.5, False), cell.best)
    assert (solved.ratio, solved.bound, solved.excluded) == (pytest.approx(50.0), None, False)
    ours_failed = an.cell_ratio(an.Time(300.0, True), cell.best)
    assert (ours_failed.ratio, ours_failed.bound, ours_failed.excluded) == (pytest.approx(30000.0), "lower", False)
    cell = published.tables["T4"].cells[(4, 10)]  # all published fail: 3600 s, a bound
    they_failed = an.cell_ratio(an.Time(100.0, False), cell.best)
    assert (they_failed.ratio, they_failed.bound, they_failed.excluded) == (pytest.approx(100 / 3600), "upper", False)
    both = an.cell_ratio(an.Time(3600.0, True), cell.best)
    assert both.excluded is True and both.ratio is None and both.bound is None


def test_the_report_of_a_full_grid_matches_an_independent_computation(published):
    # ours: 10 x the best published time of the cell, but not beyond 0.9 of the limit (a time above it is a failure);
    # where every published method failed, ours solves in 100 s
    def scip(table, port, k):
        b = _best(table, port, k)
        return min(10.0 * b, 0.9 * LIMIT[table]) if b is not None else 100.0

    rows = _grid(published, scip=scip, oa=lambda table, port, k: 100.0 * (_best(table, port, k) or 1.0))
    report = an.m11_report(rows, published)
    for table in ("T3", "T4"):
        block = report["tables"][table]
        cells = block["cells"]
        assert len(cells) == 15
        expected = []
        for c in cells:
            b = _best(table, c["port"], c["k"])
            expected.append(scip(table, c["port"], c["k"]) / b if b is not None else 100.0 / LIMIT[table])
        got = [c["views"]["stack"]["ratio"] for c in cells]
        np.testing.assert_allclose(got, expected, rtol=1e-12)
        view = block["summary"]["stack"]
        assert view["n_cells"] == 15 and view["n_excluded"] == 0 and view["n_ours_solved"] == 15
        assert view["median_ratio"] == pytest.approx(float(np.median(expected)), rel=1e-12)
        assert view["n_pub_solved"] == sum(_best(table, p, k) is not None for p in PORTS for k in KS)
    assert report["tables"]["T3"]["summary"]["stack"]["median_ratio"] == pytest.approx(10.0)
    assert report["tables"]["T4"]["summary"]["stack"]["n_upper_bound"] == 1  # port 4, k = 10: their bound, our time


def test_the_verdict_reads_scip_and_the_better_of_our_methods_is_given_next_to_it(published):
    """§12, entry of 2026-10-18: at the sizes of the ports SCIP is the exact representative of RQ2, so a fast OA must
    not carry the verdict: SCIP at 50 x the best published time is above 10 even with OA at 1 x."""
    def times(factor):
        return lambda table, port, k: factor * (_best(table, port, k) or 1.0)

    rows = _grid(published, scip=times(50.0), oa=times(1.0), tables=("T3",))
    report = an.m11_report(rows, published, allow_incomplete=True)
    cell = report["tables"]["T3"]["cells"][0]
    assert cell["views"]["stack"]["method"] == "oa"
    assert cell["views"]["stack"]["time_s"] == cell["views"]["oa"]["time_s"] < cell["views"]["scip"]["time_s"]
    assert report["primary_view"] == "scip"
    summary = report["tables"]["T3"]["summary"]
    assert list(summary) == ["scip", "oa", "stack"]
    assert summary["stack"]["median_ratio"] == pytest.approx(1.0)
    verdict = report["tables"]["T3"]["verdict"]
    assert verdict["view"] == "scip" and verdict["median_ratio"] == pytest.approx(50.0)
    assert verdict["above_threshold"] is True and report["verdict"]["above_threshold"] is True


def test_a_cell_both_sides_failed_is_left_out_of_the_summary_and_listed(published):
    # T4 port4 k=10: every published method failed; if ours fails too the cell is excluded
    def scip(table, port, k):
        return None if (table, port, k) == ("T4", 4, 10) else 1.0

    report = an.m11_report(_grid(published, scip=scip), published)
    block = report["tables"]["T4"]
    assert block["excluded"] == [{"port": 4, "k": 10, "reason": "neither side solved the instance within the limit"}]
    view = block["summary"]["stack"]
    assert view["n_cells"] == 15 and view["n_excluded"] == 1 and view["n_ratio_cells"] == 14
    assert view["n_ours_solved"] == 14 and view["n_pub_solved"] == 14
    flagged = next(c for c in block["cells"] if (c["port"], c["k"]) == (4, 10))
    assert flagged["views"]["stack"]["excluded"] is True and flagged["views"]["stack"]["ratio"] is None


def test_the_rule_of_the_plan_is_above_10_and_not_10_itself(published):
    def verdict(factor):
        rows = _grid(published, scip=lambda t, p, k: factor * _best(t, p, k), oa=lambda *_: None, tables=("T3",))
        return an.m11_report(rows, published, allow_incomplete=True)["tables"]["T3"]["verdict"]

    at_ten = verdict(10.0)
    assert at_ten["median_ratio"] == pytest.approx(10.0) and at_ten["above_threshold"] is False  # "больше 10"
    above = verdict(10.5)
    assert above["above_threshold"] is True and "открыт" in above["statement"] and above["threshold"] == 10.0
    below = verdict(0.5)
    assert below["above_threshold"] is False and "порядка" in below["statement"]


def test_the_overall_verdict_is_above_if_any_table_is(published):
    rows = _grid(published, scip=lambda t, p, k: 1000.0 if t == "T4" else 0.001, oa=lambda *_: 0.001)
    report = an.m11_report(rows, published)
    assert report["tables"]["T3"]["verdict"]["above_threshold"] is False
    assert report["tables"]["T4"]["verdict"]["above_threshold"] is True
    assert report["verdict"]["above_threshold"] is True and report["verdict"]["tables_above"] == ["T4"]


def test_the_median_of_the_ratios_says_when_it_rests_on_a_bound(published):
    rows = _grid(published, scip=lambda t, p, k: None if p == 2 else 1.0, tables=("T3",))
    report = an.m11_report(rows, published, allow_incomplete=True)
    view = report["tables"]["T3"]["summary"]["scip"]
    assert view["n_lower_bound"] == 3 and view["n_upper_bound"] == 0 and view["n_ours_solved"] == 12
    assert view["median_bound"] in (None, "lower")


# -- what the analysis refuses ----------------------------------------------------------------------------


def test_rows_with_an_error_stop_the_analysis(published):
    rows = _grid(published)
    rows[3] = {**rows[3], "error": "RuntimeError: boom"}
    with pytest.raises(ValueError, match="error"):
        an.m11_report(rows, published)


def test_rows_of_another_stage_stop_the_analysis(published):
    rows = _grid(published)
    rows[0] = {**rows[0], "stage": "bench"}
    with pytest.raises(ValueError, match="stage"):
        an.m11_report(rows, published)


def test_a_cell_twice_stops_the_analysis(published):
    rows = _grid(published)
    with pytest.raises(ValueError, match=r"twice|duplicate"):
        an.m11_report([*rows, rows[0]], published)


def test_an_incomplete_grid_is_refused_unless_allowed(published):
    rows = [r for r in _grid(published) if not (r["table"] == "T4" and r["port"] == 5)]
    with pytest.raises(ValueError, match="missing"):
        an.m11_report(rows, published)
    report = an.m11_report(rows, published, allow_incomplete=True)
    assert len(report["tables"]["T4"]["cells"]) == 12 and report["tables"]["T4"]["missing"] == [
        {"port": 5, "k": k} for k in KS]
    assert any("missing" in w for w in report["warnings"])


def test_different_numbers_of_seeds_across_the_cells_of_a_method_are_refused(published):
    rows = [r for r in _grid(published) if not (r["table"] == "T3" and r["port"] == 2 and r["k"] == 5
                                               and r["method"]["name"] == "scip" and r["seed"] == 10002)]
    with pytest.raises(ValueError, match="seeds"):
        an.m11_report(rows, published)


def test_a_row_of_another_size_than_the_published_instance_is_refused(published):
    rows = _grid(published)
    rows[0] = {**rows[0], "n": 30}
    with pytest.raises(ValueError, match="n"):
        an.m11_report(rows, published)


def test_a_limit_that_differs_from_the_published_one_is_a_warning_not_an_error(published):
    rows = [{**r, "time_limit_s": 20.0} for r in _grid(published)]
    report = an.m11_report(rows, published)
    assert any("time_limit_s" in w for w in report["warnings"])


def test_no_rows_is_an_error(published):
    with pytest.raises(ValueError, match="rows"):
        an.m11_report([], published)


# -- audits of the rows -----------------------------------------------------------------------------------


def test_the_bounds_of_the_methods_of_a_cell_must_not_contradict_each_other(published):
    rows = _grid(published)
    ok = an.m11_report(rows, published)
    assert ok["audit"]["bound_conflicts"] == [] and ok["audit"]["n_certified_over_tolerance"] == 0
    # a certified dual bound above the incumbent of another certified run of the same cell: one of them is wrong
    bad = [dict(r) for r in rows]
    bad[0]["dual"] = bad[0]["primal"] * 1.5
    report = an.m11_report(bad, published)
    assert report["audit"]["bound_conflicts"] and report["audit"]["bound_conflicts"][0]["table"] == "T3"
    assert any("bound" in w for w in report["warnings"])


def test_a_certified_row_whose_gap_exceeds_the_stopping_rule_is_counted(published):
    rows = _grid(published)
    rows[0] = {**rows[0], "rel_gap": 3e-4, "abs_gap": 5e-6}
    report = an.m11_report(rows, published)
    assert report["audit"]["n_certified_over_tolerance"] == 1
    assert any("stopping rule" in w for w in report["warnings"])


def test_infeasible_rows_overruns_and_return_violations_are_counted(published):
    rows = _grid(published)
    rows[0] = {**rows[0], "feasible": False}
    rows[1] = {**rows[1], "budget_ok": False, "overrun_s": 40.0}
    t4 = next(i for i, r in enumerate(rows) if r["table"] == "T4")
    rows[t4] = {**rows[t4], "return_ok": False}
    audit = an.m11_report(rows, published)["audit"]
    assert (audit["n_infeasible"], audit["n_over_budget"], audit["n_return_violated"]) == (1, 1, 1)


# -- outputs ----------------------------------------------------------------------------------------------


def _register(tmp_path: Path, rows: list[dict], *, name="20261017-120000-E2-m11-abc123", **manifest) -> Path:
    """A registered run directory as the registry writes it (manifest, config, rows)."""
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": name, "git_sha": "a" * 40, "git_dirty": False, "config_hash": "abc123def456", "mode": "ingested",
        "n_errors": 0, "n_missing": 0, "bundle_sha": "b" * 40, **manifest}), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps({
        "stream": "E2", "name": "m11", "params": {"stage": "m11", "seed_role": "calibration"}}), encoding="utf-8")
    return run_dir


def test_the_command_writes_the_report_and_the_tables_and_records_inputs_and_code(tmp_path, published, capsys):
    run_dir = _register(tmp_path, _grid(published))
    out = tmp_path / "out"
    assert e2_analysis.main(["m11", "--rows", str(run_dir), "--published", str(PUBLISHED), "--out", str(out)]) == 0
    report = json.loads((out / "m11.json").read_text(encoding="utf-8"))  # strict JSON: no NaN / Infinity
    assert report["kind"] == "e2-m11" and report["schema"] == "e2-m11/1" and report["primary_view"] == "scip"
    (inp,) = [i for i in report["inputs"] if i["role"] == "rows"]
    assert inp["run_id"] == run_dir.name and len(inp["sha256"]) == 64 and inp["n_rows"] == 105
    assert inp["git_sha"] == "a" * 40 and inp["config_hash"] == "abc123def456"
    (pub,) = [i for i in report["inputs"] if i["role"] == "published"]
    assert pub["sha256"] == published.sha256
    assert set(report["code"]) >= {"git_sha", "git_dirty", "branch"}
    assert report["versions"]["python"] and report["prespecified"]
    assert report["published"]["hardware"]["machine"].startswith("MacBook Pro")
    for name in ("cells.csv", "summary.csv", "rows.csv"):
        assert (out / name).is_file(), name
    with (out / "cells.csv").open(encoding="utf-8") as fh:
        cells = list(csv.DictReader(fh))
    assert len(cells) == 30 and {c["table"] for c in cells} == {"T3", "T4"}
    assert {"port", "k", "n", "stack_time_s", "stack_ratio", "published_best_s", "published_best_method",
            "stack_censored", "published_censored", "stack_bound", "excluded"} <= set(cells[0])
    with (out / "summary.csv").open(encoding="utf-8") as fh:
        summary = list(csv.DictReader(fh))
    assert {(s["table"], s["view"]) for s in summary} == {("T3", "stack"), ("T3", "scip"), ("T3", "oa"),
                                                          ("T4", "stack"), ("T4", "scip")}
    with (out / "rows.csv").open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 105
    printed = capsys.readouterr().out
    assert "e2-m11" in printed and "T3" in printed and "T4" in printed


def test_the_provenance_is_clean_only_for_a_registered_clean_full_run(tmp_path, published):
    clean = an.m11_report(_grid(published), published)
    assert clean["provenance_clean"] is None  # decided by the command, which knows the inputs
    out = tmp_path / "out"
    run_dir = _register(tmp_path, _grid(published))
    e2_analysis.main(["m11", "--rows", str(run_dir), "--published", str(PUBLISHED), "--out", str(out)])
    report = json.loads((out / "m11.json").read_text(encoding="utf-8"))
    assert report["provenance_clean"] is (report["code"]["git_dirty"] is False)
    dirty = _register(tmp_path, _grid(published), name="20261017-130000-E2-m11-ddd000", git_dirty=True, mode="smoke",
                      n_errors=1)
    out2 = tmp_path / "out2"
    e2_analysis.main(["m11", "--rows", str(dirty), "--published", str(PUBLISHED), "--out", str(out2)])
    report = json.loads((out2 / "m11.json").read_text(encoding="utf-8"))
    assert report["provenance_clean"] is False
    text = " ".join(report["warnings"])
    assert "git_dirty" in text and "mode" in text and "errors" in text


def test_a_bare_results_file_is_accepted_but_warned_about(tmp_path, published):
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _grid(published)) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    assert e2_analysis.main(["m11", "--rows", str(path), "--published", str(PUBLISHED), "--out", str(out)]) == 0
    report = json.loads((out / "m11.json").read_text(encoding="utf-8"))
    assert report["provenance_clean"] is False and any("registered" in w for w in report["warnings"])


def test_the_command_refuses_an_incomplete_grid_and_allows_it_on_request(tmp_path, published):
    rows = [r for r in _grid(published) if r["table"] == "T3"]
    run_dir = _register(tmp_path, rows)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="missing"):
        e2_analysis.main(["m11", "--rows", str(run_dir), "--published", str(PUBLISHED), "--out", str(out)])
    assert e2_analysis.main(["m11", "--rows", str(run_dir), "--published", str(PUBLISHED), "--out", str(out),
                             "--allow-incomplete"]) == 0
    report = json.loads((out / "m11.json").read_text(encoding="utf-8"))
    assert set(report["tables"]) == {"T3"} and any("T4" in w for w in report["warnings"])


def test_prespecified_readings_are_in_the_code_and_in_the_json():
    for key in ("our_time", "published", "ratio", "primary_view", "excluded", "summary", "rule", "hardware"):
        assert key in an.PRESPECIFIED and an.PRESPECIFIED[key]["choice"] and an.PRESPECIFIED[key]["why"]
    assert an.THRESHOLD == 10.0 and an.PRIMARY_VIEW == "scip" and e2_m11_analysis.THRESHOLD == 10.0


# -- the rows of the real stage ----------------------------------------------------------------------------


def test_rows_of_the_stage_itself_go_through_the_analysis(tmp_path, published):
    """A cell of each kind run for real on a synthetic port1, then read by the analysis against the published cell
    of port 1 (the numbers mean nothing, the plumbing is the point)."""
    (tmp_path / "port1.txt").write_text(orlib_text(ORLIB_N[1], seed=5))
    e2_m11.clear_caches()
    cfg = _cfg(tmp_path)
    rows = []
    for table, method in (("T3", "scip"), ("T3", "oa"), ("T4", "scip")):
        cell = _cell(cfg, table, method)
        rows.append({"cell_key": run.cell_key(cell), **cell, "host": "x", **e2_m11.run_cell(cell, cfg)})
    ours = an.our_time([r for r in rows if r["table"] == "T3" and r["method"]["name"] == "scip"])
    assert not ours.censored and ours.seconds == rows[0]["wall_s"]
    assert rows[0]["certified"] and rows[2]["certified"] and rows[2]["return_ok"]
    audit = an.audit_rows(rows)
    assert audit["n_infeasible"] == 0 and audit["n_return_violated"] == 0 and audit["n_certified_over_tolerance"] == 0
    e2_m11.clear_caches()
