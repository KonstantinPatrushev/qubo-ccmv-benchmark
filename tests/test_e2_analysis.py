"""H1c: the held-out replication of "tabu beats SA at equal budget" (ANALYSIS_PLAN §2, §7.1; §12 A5).

The instances are the non-trivial test instances of C1-C3 (A1), the statistic the one-sided Wilcoxon signed-rank
test of the difference of the final gaps at 10 s (tabu minus SA, median over seeds), stratified by N. Differences
within ``10⁻⁸·s*`` are zeros and are dropped before ranking, as ``zero_tol`` of H1a: on a trivial instance both
pipelines stand at the start or at the optimum and the noise of the refinement would get the ranks of the test.
The rows are synthetic with hand-computed numbers; the test of the statistic itself is that of
:func:`cif.stats.rq1.stratified_signed_rank`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cif.experiments import e2_analysis as an
from cif.stats.rq1 import stratified_signed_rank

S = 0.14  # scale of the instances in the synthetic rows
BUDGET = 10.0


def report(selected: dict[str, list[int]]) -> dict:
    """A finished eligibility report of the test role with these selected seeds per cell."""
    cells = {c: {"done": True, "enough": True, "prefix": 400, "found": len(seeds), "selected": list(seeds),
                 "needs_reference": list(seeds)} for c, seeds in selected.items()}
    return {"kind": "e2-eligibility", "role": "test", "cells": cells}


def rows_for(cell: str, n: int, seed: int, f_tabu: float, f_sa: float, *, scale: float = S, n_seeds: int = 10,
             noise: float = 0.0) -> list[dict]:
    """The rows of the stage ``h1c`` of one instance: ``n_seeds`` runs of each method whose median incumbent at the
    budget is ``f_tabu`` / ``f_sa`` (a symmetric spread of ``noise`` around it)."""
    rows = []
    spread = np.linspace(-noise, noise, n_seeds)
    for name, f in (("tabu", f_tabu), ("sa", f_sa)):
        for i, d in enumerate(spread):
            rows.append({
                "cell_key": f"{cell}|{seed}|{name}|{i}", "stage": "h1c", "cell": cell, "n_assets": n, "n": n,
                "seed": seed, "instance_id": f"factor/{n}/{seed}", "method": {"name": name, "id": f"{name}-h1c"},
                "config_id": f"{name}-h1c", "solver_seed": 20000 + i, "budget_s": BUDGET,
                "f_at_budget": f + float(d), "start_scale": scale,
                "start_f": f + 1e-3,
            })
    return rows


def test_the_difference_is_the_median_over_seeds_of_tabu_minus_the_median_of_sa():
    rows = rows_for("C1", 100, 2001, -0.0500, -0.0490, noise=1e-3, n_seeds=10)
    diffs = an.h1c_differences(rows, report({"C1": [2001], "C2": [], "C3": []}))
    (d,) = diffs
    assert d["instance_id"] == "factor/100/2001" and d["stratum"] == 100
    assert d["f_tabu"] == pytest.approx(-0.0500, abs=1e-12) and d["f_sa"] == pytest.approx(-0.0490, abs=1e-12)
    assert d["difference"] == pytest.approx(d["f_tabu"] - d["f_sa"], abs=1e-15)
    assert d["difference"] < 0  # tabu is lower, that is better
    assert d["normalised"] == pytest.approx(d["difference"] / S, abs=1e-15)
    assert d["n_seeds"] == {"tabu": 10, "sa": 10}


def test_the_test_is_one_sided_in_favour_of_tabu_and_stratified_by_n():
    selected = {"C1": [2000 + i for i in range(6)], "C2": [2000 + i for i in range(5)], "C3": []}
    rows = []
    rng = np.random.default_rng(7)
    for cell, n in (("C1", 100), ("C2", 200)):
        for seed in selected[cell]:
            gain = float(rng.uniform(1e-4, 1e-3))  # tabu better on every instance
            rows += rows_for(cell, n, seed, -0.05 - gain, -0.05)
    result = an.h1c_test(rows, report(selected))
    assert result["kind"] == "e2-h1c" and result["alternative"] == "less"
    assert result["n_instances"] == 11 and result["n_nonzero"] == 11 and result["n_zero"] == 0
    # 6 + 5 instances all in favour of tabu: the exact one-sided p-value is 2^-6 · 2^-5 per stratum product
    assert result["p_value"] == pytest.approx(2.0 ** -11, rel=1e-12)
    assert set(result["strata"]) == {"100", "200"}
    assert result["strata"]["100"]["n_neg"] == 6 and result["strata"]["200"]["n_pos"] == 0
    assert result["direction"] == {"tabu_better": 11, "sa_better": 0, "zero": 0}
    # the other direction is not significant
    flipped = [{**r, "method": {**r["method"], "name": "sa" if r["method"]["name"] == "tabu" else "tabu"}}
               for r in rows]
    assert an.h1c_test(flipped, report(selected))["p_value"] > 0.99


def test_differences_within_1e_8_of_the_scale_are_zeros_and_are_dropped_before_ranking():
    """§12 A5: |F_tabu − F_SA| ≤ 1e-8·s* is the noise of the refinement, not a result."""
    selected = {"C1": list(range(2000, 2008)), "C2": [], "C3": []}
    rows = []
    real = [2e-4, 4e-4, 1e-3]  # tabu better by this much (units of the objective) on three instances
    for seed, gain in zip(selected["C1"][:3], real, strict=True):
        rows += rows_for("C1", 100, seed, -0.05 - gain, -0.05)
    for seed in selected["C1"][3:]:  # five instances where both pipelines stand at the same portfolio
        rows += rows_for("C1", 100, seed, -0.05 + 0.2e-8 * S, -0.05)  # 0.2e-8·s*: a zero
    result = an.h1c_test(rows, report(selected))
    assert result["n_zero"] == 5 and result["n_nonzero"] == 3 and result["zero_tol"] == 1e-8
    direct = stratified_signed_rank(np.array([-g for g in real]) / S, np.full(3, 100), alternative="less")
    assert result["p_value"] == direct.p_value == pytest.approx(2.0 ** -3, rel=1e-12)
    # kept in the ranking, the five noise differences would have taken the small ranks of the test
    noisy = stratified_signed_rank(
        np.array([-g for g in real] + [0.2e-8] * 5), np.full(8, 100), alternative="less", zero_tol=0.0)
    assert noisy.p_value != pytest.approx(result["p_value"], rel=1e-3)
    assert result["direction"] == {"tabu_better": 3, "sa_better": 0, "zero": 5}


def test_the_zero_tolerance_is_in_units_of_the_scale_of_each_instance():
    selected = {"C1": [2000, 2001], "C2": [], "C3": []}
    tiny = 5e-9  # of the scale 0.14 it is 3.6e-8: not a zero; of the scale 1.0 it is 5e-9: a zero
    rows = rows_for("C1", 100, 2000, -0.05 - tiny, -0.05, scale=0.14) + rows_for(
        "C1", 100, 2001, -0.05 - tiny, -0.05, scale=1.0)
    result = an.h1c_test(rows, report(selected))
    assert result["n_zero"] == 1 and result["n_nonzero"] == 1


def test_the_scale_of_the_reference_replaces_the_scale_of_the_start_when_it_is_given():
    selected = {"C1": [2000], "C2": [], "C3": []}
    rows = rows_for("C1", 100, 2000, -0.05 - 1e-8, -0.05, scale=0.14)  # 7e-8·s of the start: not a zero
    assert an.h1c_test(rows, report(selected))["n_zero"] == 0
    scales = {"factor/100/2000": 10.0}  # the scale of the optimum: 1e-9·s*, a zero
    assert an.h1c_test(rows, report(selected), scales=scales)["n_zero"] == 1


def test_only_the_non_trivial_instances_of_c1_to_c3_enter_and_none_may_be_missing():
    selected = {"C1": [2000, 2001], "C2": [2000], "C3": []}
    rows = (rows_for("C1", 100, 2000, -0.051, -0.050) + rows_for("C1", 100, 2001, -0.052, -0.050)
            + rows_for("C2", 200, 2000, -0.051, -0.050))
    assert an.h1c_test(rows, report(selected))["n_instances"] == 3
    # a row of an instance the screening did not select (a trivial one) is an error, not a quiet exclusion
    with pytest.raises(ValueError, match="not selected"):
        an.h1c_test([*rows, *rows_for("C1", 100, 2002, -0.051, -0.050)], report(selected))
    # an instance without rows is an error too
    with pytest.raises(ValueError, match="no rows"):
        an.h1c_test(rows[:20], report(selected))
    # and so is an instance on which one method is missing
    only_sa = [r for r in rows if not (r["seed"] == 2001 and r["method"]["name"] == "tabu")]
    with pytest.raises(ValueError, match="both"):
        an.h1c_test(only_sa, report(selected))
    # rows of another cell or stage do not belong to H1c
    other = [{**r, "stage": "bench"} for r in rows_for("C1", 100, 2005, -0.1, -0.05)]
    assert an.h1c_test([*rows, *other], report(selected))["n_instances"] == 3


def test_an_unfinished_scan_or_a_cell_without_the_minimum_is_no_basis_for_the_test():
    rows = rows_for("C1", 100, 2000, -0.051, -0.050)
    unfinished = report({"C1": [2000], "C2": [], "C3": []})
    unfinished["cells"]["C1"]["done"] = False
    with pytest.raises(ValueError, match="not finished"):
        an.h1c_test(rows, unfinished)
    thin = report({"C1": [2000], "C2": [], "C3": []})
    thin["cells"]["C1"]["enough"] = False
    result = an.h1c_test(rows, thin)  # H1c uses every non-trivial instance of C1-C3; the minimum is for H2
    assert result["n_instances"] == 1 and result["descriptive_cells"] == ["C1"]


def test_an_equal_number_of_seeds_is_required_of_both_methods_and_reported():
    rows = rows_for("C1", 100, 2000, -0.051, -0.050, n_seeds=10)
    rows = [r for r in rows if not (r["method"]["name"] == "sa" and r["solver_seed"] >= 20007)]
    with pytest.raises(ValueError, match="seeds"):
        an.h1c_test(rows, report({"C1": [2000], "C2": [], "C3": []}))


def test_the_report_is_json_safe_and_gives_the_p_value_the_rq1_family_reads():
    selected = {"C1": [2000, 2001, 2002], "C2": [], "C3": []}
    rows = []
    for i, seed in enumerate(selected["C1"]):
        rows += rows_for("C1", 100, seed, -0.05 - 1e-3 * (i + 1), -0.05)
    result = an.h1c_test(rows, report(selected))
    json.dumps(result, allow_nan=False)
    assert 0.0 < result["p_value"] <= 1.0 and result["method"] in ("exact", "approx")
    assert result["zero_tol"] == an.H1C_ZERO_TOL == 1e-8
    assert result["median_normalised_difference"] == pytest.approx(-2e-3 / S, rel=1e-9)
    from cif.stats.rq1 import holm_family

    holm = holm_family({"H1a": 0.01, "H1c": result["p_value"]}, alpha=0.05)
    assert holm["complete"] is True


def test_the_command_line_reads_registered_rows_and_writes_the_report(tmp_path):
    selected = {"C1": [2000, 2001, 2002], "C2": [], "C3": []}
    rows = []
    for i, seed in enumerate(selected["C1"]):
        rows += rows_for("C1", 100, seed, -0.05 - 1e-3 * (i + 1), -0.05)
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elig = tmp_path / "eligibility_test.json"
    elig.write_text(json.dumps(report(selected)))
    out = tmp_path / "h1c.json"
    assert an.main(["h1c", "--rows", str(path), "--eligibility", str(elig), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["kind"] == "e2-h1c" and result["n_instances"] == 3
    assert {i["source"] for i in result["inputs"]} == {str(path), str(elig)}
    assert result["inputs"][0]["sha256"]
