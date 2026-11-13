"""Eligibility of instances for RQ2 and H1c (ANALYSIS_PLAN §12, entries A1 and A3 of 2026-10-11).

The rule: methods are compared only on instances where the common start does not reach the main target of
the time-to-target, ``ĝ_start > τ = 1e-3``. The decision is made once, at the reference stage, before any
method runs on the instance, from a solution found (non-trivial) or from a dual bound (trivial); what neither
settles within the limit is undecided and counted. The scan of seeds, the caps, the descriptive fallback and
the estimate of the share of trivial instances (A3) are tested on synthetic rows with hand-computed numbers
and on the rows of the registered reference run of the 24 calibration instances
(``20261011-211545-E2-ref_calibration-936867``; calibration seeds only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cif.experiments import e2_eligibility as el
from cif.experiments.e2_eligibility import ScanSpec

REGISTERED = (Path(__file__).resolve().parents[1] / "runs" / "20261011-211545-E2-ref_calibration-936867"
              / "results.jsonl")
needs_registered = pytest.mark.skipif(not REGISTERED.exists(), reason="the registered reference run is absent")

TAU = 1e-3
S = 0.14  # scale of the start in the synthetic rows
F0 = -0.05
THR = TAU * S


# --------------------------------------------------------------------------------------
# Synthetic rows
# --------------------------------------------------------------------------------------


def evidence(label: str, *, certificate: str = "tau", status: str | None = None, s: float = S,
             f_start: float = F0, tau: float = TAU) -> dict:
    """The numbers of a screening run that give ``label`` under the rule."""
    thr = tau * s
    f_best, dual = {
        "trivial": (f_start, f_start - 0.5 * thr),
        "nontrivial": (f_start - 2.0 * thr, f_start - 10.0 * thr),
        "undecided": (f_start - 0.5 * thr, f_start - 10.0 * thr),
    }[label]
    return {"certificate": certificate, "tau": tau, "f_start": f_start, "s_start": s, "f_best": f_best,
            "dual_bound": dual, "status": status or ("timelimit" if label == "undecided" else "duallimit")}


def erow(cell: str, seed: int, label: str, *, n: int = 100, family: str = "factor", binding: bool = True,
         t_cert: float | None = None, certificate: str = "tau", **ev) -> dict:
    """A row of the stage ``eligibility`` as ``run_cell`` writes it, reduced to what the report reads."""
    if not binding:
        return {"cell_key": f"{cell}|{seed}", "stage": "eligibility", "cell": cell, "family": family, "n": n,
                "seed": seed, "instance_id": f"{family}/{n}/{seed}", "binding": False, "status": "skipped",
                "skipped": "not_binding"}
    e = evidence(label, certificate=certificate, **ev)
    verdict = el.classify(f_start=e["f_start"], s_start=e["s_start"], f_best=e["f_best"],
                          dual_bound=e["dual_bound"], tau=e["tau"], status=e["status"])
    assert verdict.label == label, (label, verdict)  # the helper builds what it says it builds
    return {"cell_key": f"{cell}|{seed}", "stage": "eligibility", "cell": cell, "family": family, "n": n,
            "seed": seed, "instance_id": f"{family}/{n}/{seed}", "binding": True, "status": e["status"],
            "eligibility": label, "evidence": {**e, "gap_lo": verdict.gap_lo, "gap_hi": verdict.gap_hi,
                                               "reason": verdict.reason},
            "start_f": e["f_start"], "start_scale": e["s_start"], "t_cert_s": t_cert}


def spec(cell: str = "C1", order=range(2000, 2400), *, target: int | None = 3, min_nontrivial: int = 2, **kw):
    return ScanSpec(cell=cell, order=tuple(order), target=target, min_nontrivial=min_nontrivial,
                    family=kw.pop("family", "factor"), n=kw.pop("n", 100),
                    config_size=kw.pop("config_size", 100), **kw)


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------


def classify(**over):
    base = {"f_start": F0, "s_start": S, "f_best": F0, "dual_bound": -math.inf, "tau": TAU}
    return el.classify(**{**base, **over})


def test_a_dual_bound_within_tau_of_the_start_makes_an_instance_trivial():
    v = classify(dual_bound=F0 - 0.9 * THR)
    assert (v.label, v.reason) == ("trivial", "dual_bound")
    assert v.gap_hi == pytest.approx(0.9 * TAU) and v.gap_lo == 0.0 and v.threshold == pytest.approx(THR)


def test_a_solution_better_by_more_than_tau_makes_it_non_trivial():
    v = classify(f_best=F0 - 1.5 * THR)
    assert (v.label, v.reason) == ("nontrivial", "improvement")
    assert v.gap_lo == pytest.approx(1.5 * TAU)


def test_both_sides_are_decided_at_the_threshold_as_the_plan_words_it():
    """Non-trivial iff ĝ_start > τ (strictly), trivial iff the bound proves ĝ_start ≤ τ."""
    assert classify(f_best=F0 - THR * (1 - 1e-9)).label == "undecided"  # just below τ: no bound either
    assert classify(f_best=F0 - THR * (1 + 1e-9)).label == "nontrivial"
    assert classify(dual_bound=F0 - THR * (1 - 1e-9)).label == "trivial"
    assert classify(dual_bound=F0 - THR * (1 + 1e-9)).label == "undecided"


def test_the_threshold_itself_is_within_tau_and_not_above_it():
    """In exactly representable numbers (scale 1, τ = 1/4): ĝ = τ is trivial if proved and not non-trivial."""
    exact = {"f_start": 0.0, "s_start": 1.0, "tau": 0.25}
    assert el.classify(f_best=-0.25, dual_bound=-math.inf, **exact).label == "undecided"  # not above τ
    assert el.classify(f_best=-0.25, dual_bound=-0.25, **exact).label == "trivial"  # proved within τ
    assert el.classify(f_best=-0.2500001, dual_bound=-math.inf, **exact).label == "nontrivial"


def test_what_neither_a_solution_nor_a_bound_settles_is_undecided_and_says_why():
    open_ = classify(f_best=F0 - 0.5 * THR, dual_bound=F0 - 10 * THR, status="timelimit")
    assert (open_.label, open_.reason) == ("undecided", "time_limit")
    other = classify(f_best=F0 - 0.5 * THR, dual_bound=F0 - 10 * THR, status="userinterrupt")
    assert (other.label, other.reason) == ("undecided", "open")
    assert classify(status="timelimit").gap_hi == math.inf  # no dual bound at all


def test_a_missing_incumbent_does_not_hide_a_proof_about_the_start():
    """The bound is a statement about the start and holds whatever SCIP's own incumbent is."""
    assert classify(f_best=math.inf, dual_bound=F0 - 0.1 * THR).label == "trivial"
    assert classify(f_best=math.inf, dual_bound=-math.inf).label == "undecided"


def test_bounds_that_contradict_each_other_are_undecided_not_a_coin_toss():
    v = classify(f_best=F0 - 2 * THR, dual_bound=F0 - 0.5 * THR)  # the incumbent below a valid bound
    assert (v.label, v.reason) == ("undecided", "bounds_cross")


def test_an_incumbent_worse_than_the_start_is_no_improvement():
    assert classify(f_best=F0 + 5 * THR, dual_bound=F0 - 0.2 * THR).gap_lo == 0.0


def test_a_start_that_failed_leaves_nothing_to_compare_with():
    v = classify(f_start=math.inf)
    assert (v.label, v.reason) == ("undecided", "no_start")


def test_tau_is_a_parameter_and_the_scale_must_be_positive():
    assert classify(dual_bound=F0 - 2 * THR).label == "undecided"
    assert classify(dual_bound=F0 - 2 * THR, tau=3e-3).label == "trivial"
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="s_start"):
            classify(s_start=bad)
    with pytest.raises(ValueError, match="tau"):
        classify(tau=0.0)


# --------------------------------------------------------------------------------------
# The registered reference run: the same rule on full certificates
# --------------------------------------------------------------------------------------


def _registered_rows() -> list[dict]:
    return [json.loads(line) for line in REGISTERED.read_text().splitlines() if line.strip()]


@needs_registered
def test_the_registered_calibration_rows_are_21_trivial_and_3_non_trivial_instances():
    verdicts = {r["seed"]: el.row_verdict(r) for r in _registered_rows()}
    non_trivial = sorted(s for s, v in verdicts.items() if v.label == "nontrivial")
    assert non_trivial == [1014, 1016, 1021]  # ĝ_start = 1.8e-3, 1.9e-3, 2.4e-3
    assert sum(v.label == "trivial" for v in verdicts.values()) == 21
    assert all(v.reason in ("dual_bound", "improvement") for v in verdicts.values())
    # a full certificate leaves no room between the two sides: the bounds meet within 1e-5 of the scale
    assert all(v.gap_hi - v.gap_lo < 1e-5 for v in verdicts.values())
    by_n = {n: sum(el.row_verdict(r).label == "trivial" for r in _registered_rows() if r["n"] == n)
            for n in (100, 200, 400)}
    assert by_n == {100: 8, 200: 7, 400: 6}  # the table of the consult, A.1 item 2


@needs_registered
def test_the_evidence_of_a_reference_row_is_its_full_certificate():
    row = next(r for r in _registered_rows() if r["seed"] == 1014)
    ev = el.evidence_of_row(row)
    assert ev["certificate"] == "full" and ev["stage"] == "reference"
    assert ev["f_start"] == row["start_f"] and ev["s_start"] == row["start_scale"]
    assert ev["f_best"] <= row["f_star"] + 1e-12 and ev["t_cert_s"] == row["t_cert_s"]
    assert ev["dual_bound"] == pytest.approx(row["f_star"], abs=1e-6 * row["scale_star"])


@needs_registered
def test_the_plan_quotes_the_share_of_trivial_instances_and_its_wilson_interval():
    """§12 A1: the start is within τ on 21 of 24 calibration instances (87.5 %, Wilson 69–96 %)."""
    rows = [{**r, "cell": "ALL"} for r in _registered_rows()]
    report = el.build_report(rows, [spec("ALL", range(1000, 1024), target=24, min_nontrivial=8, n=None,
                                         config_size=None)], role="calibration")
    est = report["cells"]["ALL"]["estimate"]
    assert (est["n_scanned"], est["n_trivial"], est["n_nontrivial"], est["n_undecided"]) == (24, 21, 3, 0)
    assert est["share_trivial"] == 0.875
    low, high = est["ci95"]
    assert (round(low, 2), round(high, 2)) == (0.69, 0.96)


@needs_registered
def test_the_a3_table_from_the_registered_rows_has_the_distribution_of_the_gap_in_basis_points():
    rows = _registered_rows()
    report = el.build_report(rows, [spec(c, range(a, a + 8), target=8, min_nontrivial=8, n=n, config_size=n)
                                    for c, n, a in (("T100", 100, 1000), ("T200", 200, 1008),
                                                    ("T400", 400, 1016))], role="calibration")
    table = {c: report["cells"][c]["estimate"] for c in ("T100", "T200", "T400")}
    assert [table[c]["n_trivial"] for c in table] == [8, 7, 6]
    assert [round(table[c]["share_trivial"], 3) for c in table] == [1.0, 0.875, 0.75]
    g = table["T400"]["g_start_bp"]  # (f_start − f*)·1e4 over the 8 instances of N = 400
    by_hand = sorted((r["start_f"] - r["f_star"]) * 1e4 for r in rows if r["n"] == 400)
    assert g["n"] == g["n_exact"] == 8 and g["max"] == pytest.approx(by_hand[-1], rel=1e-9)
    assert g["median"] == pytest.approx((by_hand[3] + by_hand[4]) / 2, rel=1e-9)
    assert g["q1"] == pytest.approx(float(np.percentile(by_hand, 25)), abs=1e-9)
    assert g["q3"] == pytest.approx(float(np.percentile(by_hand, 75)), rel=1e-9)
    # no trivial instance has a gap above τ·s: about 1.4 bp CE at these scales; the non-trivial ones about 3-4 bp
    assert table["T400"]["g_start_trivial_upper_bp_max"] < 1.5
    assert 3.0 < g["max"] < 4.5
    t = table["T400"]["t_cert_trivial_s"]  # the time of the certificate on the trivial instances
    certs = sorted(r["t_cert_s"] for r in rows if r["n"] == 400 and el.row_verdict(r).label == "trivial")
    assert t["n"] == 6 and t["level"] == "full" and t["max"] == pytest.approx(certs[-1])
    assert t["median"] == pytest.approx((certs[2] + certs[3]) / 2)


# --------------------------------------------------------------------------------------
# The order of the scan and the size of a round
# --------------------------------------------------------------------------------------


def test_the_scan_order_of_a_test_cell_starts_at_2000_and_stops_at_the_cap():
    order = el.scan_order(first=2000, cap=400)
    assert order[0] == 2000 and order[-1] == 2399 and len(order) == 400 and list(order) == sorted(order)


def test_the_scan_order_of_a_calibration_cell_takes_the_plan_seeds_first_and_then_the_next_ones():
    order = el.scan_order(head=list(range(1000, 1008)), then_from=1024, cap=200)
    assert order[:9] == (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1024) and len(order) == 200
    assert order[-1] == 1024 + 191
    assert len(set(order)) == 200


def test_a_scan_order_must_not_repeat_a_seed_and_must_fit_the_cap():
    with pytest.raises(ValueError, match="repeat"):
        el.scan_order(head=[1000, 1001, 1024], then_from=1024, cap=10)
    with pytest.raises(ValueError, match="cap"):
        el.scan_order(head=list(range(1000, 1010)), then_from=1024, cap=5)
    with pytest.raises(ValueError, match="cap"):
        el.scan_order(first=2000, cap=0)
    with pytest.raises(ValueError, match="first"):
        el.scan_order(cap=10)


def test_the_first_round_scans_a_block_sized_for_the_target():
    s = spec(target=20)
    assert el.next_block(s, prefix=0, found=0) == list(range(2000, 2050))  # 1.25·20/0.5 seeds
    s = spec(target=20, first_block=12)
    assert el.next_block(s, prefix=0, found=0) == list(range(2000, 2012))
    s = spec(target=20, block=7)
    assert el.next_block(s, prefix=0, found=0) == list(range(2000, 2007))
    assert el.next_block(s, prefix=7, found=1) == list(range(2007, 2014))


def test_a_later_round_is_sized_from_the_share_of_non_trivial_instances_seen_so_far():
    s = spec(target=20)
    # 6 found in 50: the rate (6 + 1)/(50 + 2) = 0.1346 → 1.25·14/0.1346 = 130 seeds
    block = el.next_block(s, prefix=50, found=6)
    assert block == list(range(2050, 2050 + 130))
    # nothing found in 100: the rate is the smoothed 1/102 and the block is cut at the cap
    block = el.next_block(s, prefix=100, found=0)
    assert block[0] == 2100 and block[-1] == 2399


def test_a_finished_cell_has_no_next_block():
    s = spec(target=3)
    assert el.next_block(s, prefix=120, found=3) == []
    assert el.next_block(s, prefix=120, found=5) == []
    assert el.next_block(s, prefix=400, found=1) == []  # the cap is spent
    assert el.next_block(spec(order=range(2000, 2010), target=None), prefix=10, found=2) == []


def test_a_fixed_set_of_points_is_screened_in_one_round():
    pts = [(p, i) for p in range(1, 6) for i in range(25)]
    s = ScanSpec(cell="C5", order=tuple(pts), target=None, min_nontrivial=10, family="orlib", points=True)
    assert el.next_block(s, prefix=0, found=0) == pts
    assert el.next_block(s, prefix=125, found=9) == []


# --------------------------------------------------------------------------------------
# One cell: what was scanned, what is selected, what is counted
# --------------------------------------------------------------------------------------


def _cell_rows(cell="C1", labels=None, start=2000, **kw):
    labels = labels or []
    return [erow(cell, start + i, lab, **kw) for i, lab in enumerate(labels)]


def _state(rows, s=None, **kw):
    s = s or spec(**kw)
    return el.cell_state(s, rows, tau=TAU)


def test_the_selection_is_the_first_n_non_trivial_instances_in_the_order_of_the_seeds():
    labels = ["trivial", "nontrivial", "trivial", "trivial", "nontrivial", "undecided", "nontrivial",
              "nontrivial", "trivial"]
    st = _state(_cell_rows(labels=labels), target=3)
    assert st["selected"] == [2001, 2004, 2006]
    assert st["prefix"] == 9 and st["done"] is True and st["enough"] is True and st["descriptive"] is False
    # the scan stops at the n-th non-trivial instance: what lies after it is not part of the protocol
    assert st["protocol"] == {"n_scanned": 7, "n_trivial": 3, "n_nontrivial": 3, "n_undecided": 1,
                              "n_nonbinding": 0}
    assert st["next_block"] == []


def test_undecided_instances_are_excluded_and_counted():
    labels = ["undecided", "nontrivial", "undecided", "trivial", "nontrivial"]
    st = _state(_cell_rows(labels=labels), target=2)
    assert st["selected"] == [2001, 2004] and st["protocol"]["n_undecided"] == 2
    assert st["protocol"]["n_scanned"] == 5 and st["protocol"]["n_trivial"] == 1


def test_a_cell_is_not_done_until_it_has_n_instances_or_the_cap_is_spent():
    labels = ["trivial", "nontrivial", "trivial"]
    st = _state(_cell_rows(labels=labels), target=3)
    assert st["done"] is False and st["selected"] == [2001] and st["prefix"] == 3
    assert st["next_block"] and st["next_block"][0] == 2003
    st = _state(_cell_rows(labels=labels), target=3, order=range(2000, 2003))
    assert st["done"] is True and st["enough"] is False and st["descriptive"] is True  # 1 < min_nontrivial = 2


def test_rows_beyond_a_gap_in_the_order_do_not_count_as_scanned():
    """Scanning in increasing order means every seed below the last used one was screened."""
    rows = [*_cell_rows(labels=["trivial", "nontrivial", "trivial"]), erow("C1", 2004, "nontrivial")]
    st = _state(rows, target=2)
    assert st["prefix"] == 3 and st["selected"] == [2001] and st["done"] is False
    assert st["n_screened"] == 4 and st["beyond_gap"] == [2004]
    assert st["next_block"][0] == 2003


def test_a_non_binding_instance_uses_up_its_seed_but_is_never_selected():
    rows = [erow("C5", 2000, "trivial", binding=False), erow("C5", 2001, "nontrivial")]
    st = _state(rows, cell="C5", target=1)
    assert st["selected"] == [2001] and st["protocol"]["n_nonbinding"] == 1 and st["protocol"]["n_scanned"] == 1


def test_a_cell_below_the_minimum_of_the_plan_is_descriptive_once_its_scan_is_over():
    labels = ["nontrivial"] + ["trivial"] * 30
    cal = _state(_cell_rows(labels=labels), target=8, min_nontrivial=8, order=range(2000, 2031))
    assert cal["enough"] is False and cal["descriptive"] is True and cal["selected"] == [2000]
    # while seeds are still to be scanned the cell may yet reach the minimum: not descriptive, only short so far
    midway = _state(_cell_rows(labels=labels), target=8, min_nontrivial=8)
    assert midway["done"] is False and midway["enough"] is False and midway["descriptive"] is False
    assert el.MIN_NONTRIVIAL == {"calibration": 8, "test": 20}
    ok = _state(_cell_rows(labels=["nontrivial"] * 8), target=8, min_nontrivial=8)
    assert ok["enough"] is True and ok["descriptive"] is False


def test_a_seed_screened_twice_keeps_one_decision_and_a_contradiction_is_an_error():
    a = erow("C1", 2000, "trivial")
    again = {**a, "cell_key": "other-run"}
    assert _state([a, again], target=1)["protocol"]["n_trivial"] == 1
    clash = {**erow("C1", 2000, "nontrivial"), "cell_key": "other-run"}
    with pytest.raises(ValueError, match="revis"):
        _state([a, clash], target=1)


def test_a_reference_row_of_the_same_instance_never_revises_the_decision():
    """The full certificate of a non-trivial instance, run later, must not change what was decided."""
    screened = erow("C1", 2000, "nontrivial")
    ref = _ref_row(2000, F0 - 2 * THR, F0 - 2 * THR)  # the optimum is 2τ·s below the start: it agrees
    st = _state([screened, ref], target=1)
    assert st["selected"] == [2000] and st["instances"][0]["class"] == "nontrivial"
    assert st["instances"][0]["certificate"] == "tau"


def _ref_row(seed, f_best, dual, *, cell="C1", certified=True):
    return {"stage": "reference", "cell": cell, "seed": seed, "n": 100, "family": "factor",
            "instance_id": f"factor/100/{seed}", "start_f": F0, "start_scale": S, "f_final": f_best, "f_star": f_best,
            "scale_star": S, "certified": certified, "ref_type": "certified" if certified else "best_known",
            "status": "optimal" if certified else "timelimit", "detail": {"dual_bound": dual}, "t_cert_s": 3.0,
            "binding": True}


def test_a_reference_row_that_contradicts_a_decision_is_an_error_not_a_revision():
    """A valid dual bound cannot be contradicted by a later solution: two decided instances that differ mean a
    bug or a corrupted row, and nothing is quietly overridden, whichever of the two rows came first."""
    trivial = erow("C1", 2000, "trivial")
    better = _ref_row(2000, F0 - 5 * THR, F0 - 5 * THR)  # a full certificate that says "non-trivial"
    for rows in ([trivial, better], [better, trivial]):
        with pytest.raises(ValueError, match="never revised"):
            _state(rows, target=1)


def test_a_later_row_that_leaves_the_instance_undecided_does_not_override_the_screening():
    screened = erow("C1", 2000, "nontrivial")
    open_ = _ref_row(2000, F0 - 0.5 * THR, F0 - 10 * THR, certified=False)  # a reference run that timed out
    st = _state([open_, screened], target=1)
    assert st["selected"] == [2000] and st["instances"][0]["stage"] == "eligibility"
    assert st["conflicts"] == [{"unit": 2000, "kept": "nontrivial", "ignored": "undecided"}]
    # and the report says so
    report = el.build_report([open_, screened], [spec(target=1)], role="test")
    assert any("2000" in w and "undecided" in w for w in report["warnings"])
    assert _state([screened], target=1)["conflicts"] == []


def test_instances_without_a_full_certificate_need_a_reference_run():
    rows = [erow("C1", 2000, "nontrivial", certificate="full"), erow("C1", 2001, "nontrivial")]
    st = _state(rows, target=2)
    assert st["selected"] == [2000, 2001] and st["needs_reference"] == [2001]


def test_rows_of_other_cells_and_rows_with_errors_are_not_mixed_in():
    rows = [erow("C2", 2000, "nontrivial", n=200), erow("C1", 2000, "trivial"), erow("C1", 2001, "nontrivial")]
    assert _state(rows, target=1)["selected"] == [2001]
    bad = {**erow("C1", 2000, "trivial"), "error": "boom"}
    with pytest.raises(ValueError, match="error"):
        el.build_report([bad], [spec()], role="test")


# --------------------------------------------------------------------------------------
# Points of a fixed set (C5) and the conditions for the pooled estimate
# --------------------------------------------------------------------------------------


def _points_spec():
    pts = tuple((p, i) for p in range(1, 6) for i in range(25))
    return ScanSpec(cell="C5", order=pts, target=None, min_nontrivial=10, family="orlib", points=True)


def _point_rows(nontrivial_by_port, binding_per_port=15):
    rows = []
    for port in range(1, 6):
        for i in range(25):
            if i >= binding_per_port:
                row = {**erow("C5", 0, "trivial", binding=False, family="orlib", n=30 + port),
                       "port": port, "lam_index": i, "instance_id": f"orlib/{port}/{i:02d}"}
                row.pop("seed")
            else:
                label = "nontrivial" if i < nontrivial_by_port.get(port, 0) else "trivial"
                row = {**erow("C5", 0, label, family="orlib", n=30 + port), "port": port, "lam_index": i,
                       "instance_id": f"orlib/{port}/{i:02d}"}
                row.pop("seed")
            row["cell_key"] = f"C5|{port}|{i}"
            rows.append(row)
    return rows


def test_c5_counts_its_non_trivial_binding_points_per_port():
    st = el.cell_state(_points_spec(), _point_rows({1: 4, 2: 3, 3: 3, 4: 1}), tau=TAU)
    assert st["done"] is True and st["protocol"]["n_nonbinding"] == 50 and st["protocol"]["n_scanned"] == 75
    assert st["protocol"]["n_nontrivial"] == 11 and st["selected"][:2] == [[1, 0], [1, 1]]
    assert st["per_port"] == {"1": 4, "2": 3, "3": 3, "4": 1, "5": 0}
    assert st["n_ports"] == 4 and st["enough"] is True


def test_c5_needs_ten_points_in_three_ports_for_the_pooled_estimate():
    ok = el.cell_state(_points_spec(), _point_rows({1: 4, 2: 3, 3: 3}), tau=TAU)
    assert el.in_summary(ok, "C5") is True
    few = el.cell_state(_points_spec(), _point_rows({1: 4, 2: 3, 3: 2}), tau=TAU)  # 9 points
    assert few["enough"] is False and el.in_summary(few, "C5") is False
    one_port = el.cell_state(_points_spec(), _point_rows({1: 15}), tau=TAU)  # 15 points, 1 port: no cluster
    assert one_port["enough"] is False and one_port["n_ports"] == 1 and el.in_summary(one_port, "C5") is False


def test_the_pooled_estimate_takes_c1_to_c4_when_enough_and_c4_only_if_its_lambda_is_admissible():
    enough = _state(_cell_rows(labels=["nontrivial"] * 3), target=3, min_nontrivial=3)
    thin = _state(_cell_rows("C4", labels=["nontrivial"]), cell="C4", target=3, min_nontrivial=3)
    assert el.in_summary(enough, "C1") is True and el.in_summary(thin, "C4") is False
    c4 = _state(_cell_rows("C4", labels=["nontrivial"] * 3), cell="C4", target=3, min_nontrivial=3)
    assert el.in_summary(c4, "C4", c4_admissible=True) is True
    assert el.in_summary(c4, "C4", c4_admissible=False) is False  # the lambda rule makes C4 descriptive
    assert el.in_summary(c4, "D1_100") is False and el.in_summary(enough, "D1") is False  # D1 is descriptive


def test_without_a_cell_in_the_pooled_estimate_rq2_answers_with_the_share_of_trivial_instances():
    rows = _cell_rows(labels=["trivial"] * 6 + ["nontrivial"])
    report = el.build_report(rows, [spec(target=3, min_nontrivial=3)], role="test")
    assert report["summary"]["cells"] == []
    assert "Нетривиальных инстансов недостаточно" in report["summary"]["statement"]
    assert "долей тривиальных" in report["summary"]["statement"]
    rows = _cell_rows(labels=["nontrivial"] * 3)
    report = el.build_report(rows, [spec(target=3, min_nontrivial=3)], role="test")
    assert report["summary"]["cells"] == ["C1"] and report["summary"]["statement"] is None


# --------------------------------------------------------------------------------------
# Configurations of a calibration cell with too few instances: those of the nearest N
# --------------------------------------------------------------------------------------


def test_a_deficient_calibration_size_borrows_the_configurations_of_the_nearest_size_with_enough():
    sizes = {100: False, 200: True, 400: False}
    assert el.nearest_sizes(sizes) == {100: 200, 400: 200}
    sizes = {100: True, 200: False, 400: True}
    assert el.nearest_sizes(sizes) == {200: 100}  # |200 − 100| < |400 − 200|
    assert el.nearest_sizes({100: True, 200: True, 400: True}) == {}
    assert el.nearest_sizes({100: False, 200: False}) == {100: None, 200: None}  # nothing to borrow from


def test_the_report_names_the_sizes_that_borrow_and_warns_when_none_can_be_tuned():
    ok = [erow("T200", 1000 + i, "nontrivial", n=200) for i in range(2)]
    thin = [erow("T100", 1000, "nontrivial"), *[erow("T100", 1001 + i, "trivial") for i in range(9)]]  # whole order
    specs = [spec("T100", range(1000, 1010), target=2, min_nontrivial=2, config_size=100),
             spec("T200", range(1000, 1010), target=2, min_nontrivial=2, n=200, config_size=200)]
    report = el.build_report(thin + ok, specs, role="calibration")
    assert report["config_from"] == {"100": 200} and report["done"] is True
    report = el.build_report(thin, specs[:1], role="calibration")
    assert report["config_from"] == {"100": None}
    assert any("no size has enough" in w for w in report["warnings"])


def test_the_sizes_are_not_judged_while_a_calibration_scan_is_still_running():
    """Mid-scan "not enough yet" is not a verdict: no borrowing, no warning, until every cell is over."""
    specs = [spec("T100", range(1000, 1200), target=8, min_nontrivial=8, config_size=100),
             spec("T200", range(1000, 1200), target=8, min_nontrivial=8, n=200, config_size=200)]
    rows = [erow("T100", 1000 + i, "trivial") for i in range(8)] + [erow("T200", 1000 + i, "trivial", n=200)
                                                                     for i in range(8)]
    report = el.build_report(rows, specs, role="calibration")
    assert report["done"] is False and report["config_from"] == {} and report["warnings"] == []
    assert report["cells"]["T100"]["descriptive"] is False
    assert report["config_from_final"] is False


# --------------------------------------------------------------------------------------
# The report, the estimate A3 and the access of the later stages
# --------------------------------------------------------------------------------------


def test_the_estimate_counts_undecided_instances_in_the_denominator_and_gives_a_wilson_interval():
    labels = ["trivial"] * 6 + ["nontrivial"] * 2 + ["undecided"] * 2
    est = _state(_cell_rows(labels=labels), target=2)["estimate"]
    # the scan stops at the 2nd non-trivial instance (the 8th seed): 6 trivial, 2 non-trivial, 0 undecided
    assert (est["n_scanned"], est["n_trivial"], est["n_nontrivial"], est["n_undecided"]) == (8, 6, 2, 0)
    assert est["share_trivial"] == 0.75 and est["ci95"][0] < 0.75 < est["ci95"][1]
    est = _state(_cell_rows(labels=labels), target=5)["estimate"]  # never reached: all 10 seeds are the scan
    assert (est["n_scanned"], est["n_undecided"]) == (10, 2) and est["share_trivial"] == 0.6
    from cif.stats.rq1 import wilson_interval

    assert est["ci95"] == list(wilson_interval(6, 10, 0.95))


def test_the_estimate_gives_the_gap_of_the_start_in_basis_points_of_the_ce():
    """ĝ_start·s·10⁴ = (f_start − f_best)·10⁴ for the instances a screening run settled."""
    rows = [erow("C1", 2000 + i, "nontrivial") for i in range(4)]
    for i, r in enumerate(rows):  # improvements of 2, 3, 4, 5 bp (above τ·s = 1.4 bp)
        r["evidence"]["f_best"] = r["evidence"]["f_start"] - (i + 2) * 1e-4
        r["evidence"]["dual_bound"] = r["evidence"]["f_best"] - 1e-9  # proved optimal within 1e-9
        r["evidence"]["status"] = "optimal"
        v = el.classify(f_start=F0, s_start=S, f_best=r["evidence"]["f_best"],
                        dual_bound=r["evidence"]["dual_bound"], tau=TAU, status="optimal")
        r["evidence"].update(gap_lo=v.gap_lo, gap_hi=v.gap_hi, reason=v.reason)
    assert el.verify_rows(rows) == []
    est = _state(rows, target=4)["estimate"]["g_start_bp"]
    assert est["n"] == 4 and est["n_exact"] == 4
    assert (est["median"], est["max"]) == (pytest.approx(3.5), pytest.approx(5.0))
    assert (est["q1"], est["q3"]) == (pytest.approx(2.75), pytest.approx(4.25))


def test_a_bound_that_is_not_tight_makes_the_gap_a_lower_bound_and_the_table_says_so():
    rows = [erow("C1", 2000, "nontrivial"), erow("C1", 2001, "trivial", t_cert=4.0)]
    est = _state(rows, target=1, order=range(2000, 2002))["estimate"]
    assert est["g_start_bp"]["n"] == 1 and est["g_start_bp"]["n_exact"] == 0  # stopped at a primal limit
    est = _state(rows, target=2, order=range(2000, 2002))["estimate"]
    assert est["g_start_trivial_upper_bp_max"] == pytest.approx(0.5 * THR * 1e4)  # f_start − dual = 0.5·τ·s


def test_the_time_of_the_certificate_on_trivial_instances_is_summarised_with_its_level():
    rows = [erow("C1", 2000 + i, "trivial", t_cert=t) for i, t in enumerate([2.0, 4.0, 6.0, 8.0])]
    rows.append(erow("C1", 2004, "trivial", t_cert=100.0, certificate="full"))
    rows.append(erow("C1", 2005, "nontrivial", t_cert=999.0))
    est = _state(rows, target=1)["estimate"]["t_cert_trivial_s"]
    assert est["n"] == 5 and est["level"] == "mixed" and est["max"] == 100.0 and est["median"] == 6.0
    assert _state(rows[:4], target=9)["estimate"]["t_cert_trivial_s"]["level"] == "tau"


def test_the_estimate_of_a_cell_without_trivial_instances_has_no_numbers_to_give():
    est = _state(_cell_rows(labels=["nontrivial"]), target=1)["estimate"]
    assert est["t_cert_trivial_s"] is None and est["share_trivial"] == 0.0 and est["ci95"][0] == 0.0


@needs_registered
def test_the_estimate_of_a_cell_is_printed_as_one_line_of_the_table_of_a3():
    rows = _registered_rows()
    report = el.build_report(rows, [spec("T400", range(1016, 1024), target=8, min_nontrivial=8, n=400,
                                         config_size=400)], role="calibration")
    est = report["cells"]["T400"]["estimate"]
    line = el.format_estimate("T400", est)
    low, high = est["ci95"]
    assert line.startswith("T400: trivial 6 of 8 (75 %, Wilson ") and f"{100 * low:.0f}–{100 * high:.0f} %" in line
    assert "non-trivial 2" in line and "undecided 0" in line
    assert f"max {est['g_start_bp']['max']:.2f}" in line and "(8 of 8 exact)" in line
    assert "t_cert on trivial (full)" in line and f"max {est['t_cert_trivial_s']['max']:.1f} s" in line
    # a cell with nothing decided or no trivial instance says only what there is
    empty = _state(_cell_rows(labels=["undecided", "undecided"]), target=1, order=range(2000, 2002))["estimate"]
    text = el.format_estimate("C1", empty)
    assert "trivial 0 of 2" in text and "undecided 2" in text and "ĝ_start" not in text and "t_cert" not in text


def test_a_cell_nothing_of_which_was_scanned_yet_prints_and_serialises():
    """Round 0 of a boot cell: no rows, so no share and no interval (Wilson of 0 trials is undefined)."""
    state = _state([], target=3)
    est = state["estimate"]
    assert est["n_scanned"] == 0 and est["share_trivial"] is None
    text = el.format_estimate("CB200", est)
    assert text.startswith("CB200: trivial 0 of 0 (—, Wilson —)") and "ĝ_start" not in text
    report = el.build_report([], [spec(target=3)], role="test")  # the JSON round trip turns the NaNs into null
    assert report["cells"]["C1"]["estimate"]["ci95"] == [None, None]
    assert "Wilson —" in el.format_estimate("C1", report["cells"]["C1"]["estimate"])


def test_the_d1_cells_are_also_given_as_one_group():
    rows = _cell_rows("D1_100", ["trivial", "nontrivial"], family="dense") + _cell_rows(
        "D1_200", ["trivial", "trivial", "nontrivial"], family="dense", n=200)
    specs = [spec("D1_100", target=1, min_nontrivial=1, family="dense"),
             spec("D1_200", target=1, min_nontrivial=1, family="dense", n=200)]
    report = el.build_report(rows, specs, role="test")
    group = report["groups"]["D1"]
    assert group["cells"] == ["D1_100", "D1_200"] and group["n_scanned"] == 5 and group["n_trivial"] == 3
    assert group["share_trivial"] == 0.6


def test_the_report_is_json_safe_and_carries_the_evidence_of_every_scanned_instance():
    rows = _cell_rows(labels=["trivial", "nontrivial", "undecided", "nontrivial"], t_cert=3.5)
    report = el.build_report(rows, [spec(target=2)], role="test", c4_admissible=None)
    assert report["kind"] == "e2-eligibility" == el.REPORT_KIND and report["role"] == "test"
    assert report["tau"] == TAU and report["done"] is True
    cell = report["cells"]["C1"]
    first = cell["instances"][0]
    assert {"unit", "instance_id", "class", "reason", "certificate", "status", "gap_lo", "gap_hi",
            "f_start", "s_start", "f_best", "dual_bound", "t_cert_s"} <= set(first)
    assert [i["class"] for i in cell["instances"]] == ["trivial", "nontrivial", "undecided", "nontrivial"]
    json.dumps(report, allow_nan=False)  # infinities become null, as in every row
    assert report["section12"].startswith("20") and "A1" in report["section12"]


def test_a_report_with_a_cell_still_scanning_is_not_done_and_proposes_the_next_block():
    report = el.build_report(_cell_rows(labels=["trivial"] * 5), [spec(target=2)], role="test")
    assert report["done"] is False and report["cells"]["C1"]["next_block"][0] == 2005


def test_a_cell_with_no_rows_yet_starts_from_the_beginning():
    report = el.build_report([], [spec(target=2)], role="test")
    cell = report["cells"]["C1"]
    assert cell["prefix"] == 0 and cell["done"] is False and cell["next_block"][0] == 2000


def test_the_scan_never_runs_past_the_cap():
    s = spec(target=5, order=range(2000, 2012))
    st = _state(_cell_rows(labels=["trivial"] * 12), s=s)
    assert st["done"] is True and st["enough"] is False and st["protocol"]["n_scanned"] == 12


# --------------------------------------------------------------------------------------
# Auditing a report against its rows, and against the references
# --------------------------------------------------------------------------------------


def test_the_recorded_class_must_follow_from_the_recorded_evidence():
    good = erow("C1", 2000, "trivial")
    assert el.verify_rows([good]) == []
    tampered = {**good, "eligibility": "nontrivial"}
    problems = el.verify_rows([tampered])
    assert len(problems) == 1 and "2000" in problems[0] and "nontrivial" in problems[0]
    moved = {**good, "evidence": {**good["evidence"], "dual_bound": good["evidence"]["f_start"] - 9 * THR}}
    assert el.verify_rows([moved])  # evidence and class disagree


def test_the_certificate_of_a_non_trivial_instance_can_be_checked_against_the_decision():
    """The rule uses the scale of the start, the plan's ĝ the scale of the optimum: report the instances
    where the exact value of ĝ_start would put the instance on the other side, never revise them."""
    screened = [erow("C1", 2000, "nontrivial"), erow("C1", 2001, "nontrivial")]
    # certified optima: instance 2000 really is 2τ away, instance 2001 only 0.9τ once s* is used
    refs = [{"stage": "reference", "cell": "C1", "seed": 2000, "instance_id": "factor/100/2000", "start_f": F0,
             "start_scale": S, "f_star": F0 - 2 * THR, "scale_star": S, "ref_type": "certified"},
            {"stage": "reference", "cell": "C1", "seed": 2001, "instance_id": "factor/100/2001", "start_f": F0,
             "start_scale": S, "f_star": F0 - 2 * THR, "scale_star": 2.5 * S, "ref_type": "certified"}]
    check = el.reference_check(screened, refs, tau=TAU)
    assert check["n_checked"] == 2 and [c["instance_id"] for c in check["flipped"]] == ["factor/100/2001"]
    assert check["flipped"][0]["g_start"] == pytest.approx(0.8 * TAU)


# --------------------------------------------------------------------------------------
# Specs from the groups of a config
# --------------------------------------------------------------------------------------


def test_a_scan_group_of_a_config_becomes_a_spec():
    group = {"cell": "C2", "family": "factor", "n": 200, "config_size": 200,
             "scan": {"first": 2000, "cap": 400, "target_from": "n_rule", "min_nontrivial": 20,
                      "state": "eligibility_test"}}
    s = el.spec_from_group(group, n_rule={"C2": 27})
    assert s.cell == "C2" and s.target == 27 and s.min_nontrivial == 20 and s.n == 200
    assert s.order[0] == 2000 and len(s.order) == 400 and s.config_size == 200 and s.family == "factor"
    group["scan"] = {"head": [1000, 1001], "then_from": 1024, "cap": 5, "target": 2, "min_nontrivial": 2,
                     "state": "eligibility_calibration"}
    s = el.spec_from_group(group)
    assert s.order == (1000, 1001, 1024, 1025, 1026) and s.target == 2
    with pytest.raises(ValueError, match="n_rule"):
        el.spec_from_group({**group, "scan": {**group["scan"], "target_from": "n_rule", "target": None}})
    with pytest.raises(ValueError, match="state"):
        el.spec_from_group({**group, "scan": {k: v for k, v in group["scan"].items() if k != "state"}})
    with pytest.raises(ValueError, match="unknown"):
        el.spec_from_group({**group, "scan": {**group["scan"], "blocks": 3}})


def test_a_group_with_a_fixed_list_of_seeds_is_screened_in_one_round():
    """A static pass (and the smoke rehearsals): every listed seed is screened, the non-trivial ones are selected."""
    group = {"cell": "T20", "family": "factor", "n": 20, "config_size": 20, "seeds": [1000, 1001, 1002]}
    s = el.spec_from_group(group)
    assert s.order == (1000, 1001, 1002) and s.target is None and s.min_nontrivial == 1 and s.points is False
    assert el.next_block(s, prefix=0, found=0) == [1000, 1001, 1002] and el.next_block(s, prefix=3, found=1) == []
    assert el.spec_from_group({**group, "min_nontrivial": 2}).min_nontrivial == 2
    st = el.cell_state(s, [erow("T20", 1000, "trivial", n=20), erow("T20", 1001, "nontrivial", n=20),
                           erow("T20", 1002, "nontrivial", n=20)])
    assert st["done"] is True and st["selected"] == [1001, 1002] and st["enough"] is True


def test_an_orlib_group_becomes_the_fixed_set_of_its_points():
    group = {"cell": "C5", "family": "orlib", "ports": [1, 2], "lam_indices": 3, "config_size": 31}
    s = el.spec_from_group(group)
    assert s.points is True and s.target is None and s.min_nontrivial == el.C5_MIN_POINTS == 10
    assert s.order == ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2))


# --------------------------------------------------------------------------------------
# What the later stages read
# --------------------------------------------------------------------------------------


def _finished_report(**kw):
    rows = _cell_rows(labels=["trivial", "nontrivial", "trivial", "nontrivial", "nontrivial"])
    return el.build_report(rows, [spec(target=3, min_nontrivial=3)], role="test", **kw)


def test_later_stages_read_the_selected_instances_of_a_finished_scan():
    report = _finished_report()
    assert el.selected_units(report, "C1") == [2001, 2003, 2004]
    assert el.selected_units(report, "C1", field="needs_reference") == [2001, 2003, 2004]
    assert el.selected_units(report, "C1", require_enough=True) == [2001, 2003, 2004]


def test_a_stage_never_runs_on_a_scan_that_is_not_finished_or_on_a_report_of_another_kind():
    unfinished = el.build_report(_cell_rows(labels=["trivial"] * 3), [spec(target=3)], role="test")
    with pytest.raises(ValueError, match="not finished"):
        el.selected_units(unfinished, "C1")
    with pytest.raises(ValueError, match="no cell"):
        el.selected_units(_finished_report(), "C9")
    with pytest.raises(ValueError, match="kind"):
        el.selected_units({"kind": "e2-n-rule"}, "C1")
    stub = {"kind": "e2-eligibility", "stub": True, "cells": {}}
    with pytest.raises(ValueError, match="stub"):
        el.selected_units(stub, "C1")


def test_a_deficient_calibration_cell_gives_no_tuning_set_when_the_stage_asks_for_enough():
    report = el.build_report(_cell_rows("T100", ["nontrivial"] + ["trivial"] * 9), [
        spec("T100", range(2000, 2010), target=8, min_nontrivial=8)], role="calibration")
    assert el.selected_units(report, "T100") == [2000]  # a descriptive cell still runs what it has
    assert el.selected_units(report, "T100", require_enough=True) == []


def test_progress_of_a_scan_comes_from_the_state_report():
    report = el.build_report(_cell_rows(labels=["trivial", "nontrivial", "trivial"]), [spec(target=5)],
                             role="test")
    assert el.progress(report, "C1") == (3, 1)
    assert el.progress({"kind": "e2-eligibility", "stub": True, "cells": {}}, "C1") == (0, 0)
    assert el.progress(report, "C9") == (0, 0)  # a cell that was never scanned starts from nothing
