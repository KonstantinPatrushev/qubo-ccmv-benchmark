"""Confirmatory audit 2026-10-13, finding C-2 (MINOR): a certificate of the screening is lost when the rounds of the
test screening are merged.

Report: docs/reviews/2026-10-13-audit-confirmatory.md.

The C5 grid (125 OR-Library points) has no ``scan`` in ``configs/E2/elig_test.yaml``, so round 2 of the test screening
(``20261013-032118-E2-elig_test-e009b3``) screened all 125 points again, with the same ``cell_key`` as round 1
(``20261012-203423-E2-elig_test-e009b3``). The classes agree (27 non-trivial in both rounds), but on ``orlib/4/23``
SCIP proved optimality in round 1 (status ``optimal`` at 6 662 s, ``ref_type = certified``) and hit the 2 h limit in
round 2 (``timelimit`` at 7 202 s, ``ref_type = best_known``), with the same ``f_star``.
``cif.experiments.e2_pilot.load_rows`` merges rows by ``cell_key`` and the last row wins, so every analysis that reads
both rounds (``e2_analysis h2``, ``secondary``; ``h2_20261013_complete/instances.csv``) drops the round-1 certificate
and counts ``orlib/4/23`` among the instances "without a certificate": 10 in C5 instead of 9. The value of the
reference is the same, so no ratio changes; the count printed in the paper ("ten of its 27 points lack a
certificate") is off by one, and a re-screen of an already decided instance is a second decision that the A1 rule
("решение ... не пересматривается") does not foresee.

Acceptance: a certified row of any screening round is a certificate of the instance (``reference_table`` over the
rows of both rounds keeps it), and the primary analysis reports ``orlib/4/23`` as certified.

Resolution (2026-10-17, §12): the analyses read the references through ``e2_pilot.reference_rows`` (rows merged within
a run, not across runs); the primary analysis recomputed after the fix is ``h2_20261017`` (the test skips until it
exists).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROUND_1 = ROOT / "runs" / "20261012-203423-E2-elig_test-e009b3"
ROUND_2 = ROOT / "runs" / "20261013-032118-E2-elig_test-e009b3"
H2_COMPLETE = ROOT / "results" / "E2" / "analysis" / "h2_20261017" / "instances.csv"
POINT = "orlib/4/23"


def _need(*paths: Path) -> None:
    for path in paths:
        if not path.exists():
            hint = " (the public snapshot packs the raw rows: python scripts/unpack_runs.py)"
            pytest.skip(f"{path} is absent" + (hint if path.name == "results.jsonl" else ""))


def test_c2_round_1_holds_a_certificate_for_the_point():
    """The fact (passes): round 1 certified the point, round 2 did not; both say non-trivial."""
    _need(ROUND_1 / "results.jsonl", ROUND_2 / "results.jsonl")
    from cif.experiments import e2_pilot

    rows = {}
    for label, path in (("r1", ROUND_1), ("r2", ROUND_2)):
        for row in e2_pilot.read_rows(path / "results.jsonl"):
            if row.get("instance_id") == POINT:
                rows[label] = row
    assert rows["r1"]["ref_type"] == "certified" and rows["r1"]["status"] == "optimal"
    assert rows["r2"]["ref_type"] == "best_known" and rows["r2"]["status"] == "timelimit"
    assert rows["r1"]["eligibility"] == rows["r2"]["eligibility"] == "nontrivial"
    assert rows["r1"]["cell_key"] == rows["r2"]["cell_key"]


def test_c2_merging_the_screening_rounds_keeps_the_certificate():
    _need(ROUND_1 / "results.jsonl", ROUND_2 / "results.jsonl")
    from cif.experiments import e2_analysis, e2_pilot

    used, _ = e2_analysis._reference_rows(e2_pilot.reference_rows([ROUND_1, ROUND_2]))
    ref = e2_pilot.reference_table(used)[POINT]
    assert ref.ref_type == "certified", (
        f"{POINT}: SCIP proved optimality in round 1 of the test screening, but load_rows keeps only the last row of "
        f"a cell_key (round 2, time limit), so the reference is {ref.ref_type!r}"
    )


def test_c2_the_primary_h2_analysis_reports_the_point_as_certified():
    _need(H2_COMPLETE)
    with H2_COMPLETE.open(encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["instance_id"] == POINT)
    assert row["ref_type"] == "certified", (
        f"h2_20261017 reports {POINT} as {row['ref_type']!r}; the paper's 'ten of its 27 points lack a "
        "certificate' inherits the lost round-1 certificate (nine lack one)"
    )
