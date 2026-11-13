"""Audit G1, finding L-4 (MINOR, 2026-10-12): the C4 λ rule, a pilot value of §12, is not traceable to a run.

Report: docs/reviews/2026-10-12-audit-g1.md.

CLAUDE.md, rule 1, and plan §6 ("Без записи в реестре результат не используется"): every number traces to a ``run_id``
of ``runs/registry.jsonl``. The other pilot reports of E2 (``eligibility_calibration``, ``screen``, ``tuned``,
``n_rule``, ``ttt_instances``) list their source runs with ``run_id`` and the SHA-256 of their rows, and the E3 pilot
report is itself a registered run (``20261011-232931-E3-pilot-acc916``). ``configs/E2/pilot/c4_lambda.json`` (the medians
6, 10, 13 of ``s_cont`` and "C4 описательная", §12 2026-10-11) lists only the hash of the S&P provenance: it names no run,
no code SHA and no tree state; §12 names the code (``2cff55c``) in prose only.

Acceptance: the report names a registered run (``run_id`` in the report or in its inputs) or at least the code state
(``git_sha`` with ``git_dirty`` false) it was computed with.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _registered() -> set[str]:
    lines = (ROOT / "runs" / "registry.jsonl").read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["run_id"] for line in lines if line.strip()}


def test_l4_other_pilot_reports_name_registered_runs():
    """Reference (passes): the screen, tuned, n-rule and TTT reports name registered source runs."""
    runs = _registered()
    for name in ("screen", "tuned", "n_rule", "ttt_instances", "eligibility_calibration"):
        report = json.loads((ROOT / "configs" / "E2" / "pilot" / f"{name}.json").read_text(encoding="utf-8"))
        named = {i.get("run_id") for i in report.get("inputs", []) if i.get("run_id")}
        assert named and named <= runs, name


def test_l4_c4_lambda_report_is_traceable_to_a_run_or_a_code_state():
    report = json.loads((ROOT / "configs" / "E2" / "pilot" / "c4_lambda.json").read_text(encoding="utf-8"))
    named = {i.get("run_id") for i in report.get("inputs", []) if i.get("run_id")}
    if report.get("run_id"):
        named.add(report["run_id"])
    traced = bool(named & _registered()) or (bool(report.get("git_sha")) and report.get("git_dirty") is False)
    assert traced, (
        "c4_lambda.json (a [пилот] value of §12) names no registered run and no code state: "
        f"inputs = {report.get('inputs')}"
    )
