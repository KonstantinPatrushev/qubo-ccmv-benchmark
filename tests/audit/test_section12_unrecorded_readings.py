"""Audit G1: protocol choices of the confirmatory stages that the frozen plan leaves open and §12 does not record.

Report: docs/reviews/2026-10-12-audit-g1.md, findings E3-1 (MAJOR), E2-2 (MINOR), E2-3 (MINOR).

The plan is frozen (v2.4); a choice that the plan does not make and that changes a confirmatory or reported number is
recorded in §12 before the first confirmatory run, as was done for the readings of A1 ("Уточнения A1", 2026-10-12).
The three choices below are made in configs and code and disclosed only in code comments, READMEs or pilot reports.

* E3-1. ``configs/E3/ff49.yaml``: ``solvers.common_start: true`` — the real solvers of operator (в) (H3c, a
  confirmatory IUT) get the common start of §5 in their pool / as their warm start. Plan §2 (в) says only "SCIP с
  лимитом 10 с; tabu, SA и LS с бюджетом 1 с end-to-end"; the common start is defined in §5 for the primal integral of
  RQ2. On the E2 calibration SA never improved the start (§12, 2026-10-12, 0 of 2160 runs), so with the start the
  "SA path" of H3c is the path of the rounded relaxation, and its dose is that of the start.
* E2-2. ``configs/E2/pilot/c4_lambda.json``: no admissible λ, ``lam_used = 20``; the descriptive C4 (selection, bench)
  runs at λ = 20 (``e2_benchmark``: "the plan does not say which λ then"). The §12 entry "[пилот] C4" says that C4 is
  descriptive, not at which λ it runs (the draft line of the report did say it).
* E2-3. Tabu in E2 (RQ2 and the confirmatory H1c) reads ``num_reads_b = 10`` per batch in every configuration, a
  constant outside the tuning space of Appendix B (``e2_space``, ``tuned.json``); the plan's batch is "tabu —
  num_restarts_b рестартов" (§6), and §12 B1 reads "рестарты у tabu" as independent searches for RQ1 only. The
  exploratory configuration behind H1c had 200 reads. The design consult (minor 4) asked for a disclosure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "ANALYSIS_PLAN.md"


def _section_12() -> str:
    text = PLAN.read_text(encoding="utf-8")
    start = text.index("## 12. Журнал отклонений")
    end = text.index("## Приложение А", start)
    return text[start:end]


def _entries() -> list[str]:
    """The entries of §12, one string per bullet (continuation lines joined)."""
    entries: list[str] = []
    for line in _section_12().splitlines():
        if line.startswith("- "):
            entries.append(line[2:])
        elif entries and line.startswith("  "):
            entries[-1] += " " + line.strip()
    return entries


def test_e3_1_common_start_of_the_real_solvers_of_h3c_is_recorded():
    cfg = yaml.safe_load((ROOT / "configs" / "E3" / "ff49.yaml").read_text(encoding="utf-8"))
    assert cfg["params"]["solvers"]["common_start"] is True  # the choice under audit
    recorded = [e for e in _entries()
                if re.search(r"общ\w* старт", e) and ("H3c" in e or "(в)" in e)]
    assert recorded, (
        "configs/E3/ff49.yaml gives the real solvers of (в) / H3c the common start of §5, which plan §2 (в) does "
        "not prescribe; no §12 entry records this reading"
    )


def test_e2_2_lambda_of_the_descriptive_c4_is_recorded():
    report = json.loads((ROOT / "configs" / "E2" / "pilot" / "c4_lambda.json").read_text(encoding="utf-8"))
    assert report["admissible"] is False and report["lam_used"] == 20.0  # the choice under audit
    recorded = [e for e in _entries() if "C4" in e and re.search(r"λ\s*=\s*20", e)]
    assert recorded, (
        "the descriptive C4 runs at λ = 20 (c4_lambda.json lam_used); the §12 entry on C4 does not say at which λ"
    )


def test_e2_3_tabu_reads_per_batch_in_e2_are_recorded():
    tuned = json.loads((ROOT / "configs" / "E2" / "pilot" / "tuned.json").read_text(encoding="utf-8"))
    reads = {spec["num_reads_b"] for size in tuned["sizes"].values()
             for role, spec in size["roles"].items() if spec["name"] == "tabu"}
    assert reads == {10}  # the constant under audit
    recorded = [e for e in _entries() if "num_reads_b" in e or re.search(r"10 чтени", e)]
    assert recorded, (
        "tabu in E2 and H1c runs 10 reads per batch in every configuration (outside the tuning space of "
        "Appendix B); no §12 entry records it"
    )
