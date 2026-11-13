"""Audit G1, finding H1c-1 (MAJOR, 2026-10-12): in the frozen tuning space SA does not minimise the surrogate, so the
confirmatory H1c compares tabu with a degenerate SA, and §12 says "гипотезы не затронуты".

Report: docs/reviews/2026-10-12-audit-g1.md.

§12 (2026-10-12, "Раскрытие (§9), пилот E2"): on the 24 non-trivial calibration instances SA did not improve the common
start in any of 2160 runs (all 30 configurations score the gap of the start; the H1c configuration ``sa-00`` was picked by
id). The mechanism is the penalty of Appendix B: the multipliers {1, 3, 10, 30, 100} of ``selection_penalty_scale``
(§6 "Масштаб штрафа") freeze single-flip SA on the K-manifold before it resolves the objective. On the calibration
instance factor/100/1024 (K = 10, start = surrogate minimum −0.04539), 50 reads × 5000 sweeps of ``dwave`` SA reach the
surrogate minimum at multiplier 0.01, −0.04446 at 0.1, and only −0.03140 at 1 (the smallest of the plan); at 1 even
20 000 sweeps and wider β ranges stay above −0.033 (audit probes ``sa_probe2.py``, ``sa_probe3.py`` in the report). The
outcome of H1c ("tabu lower than SA") is then fixed by construction wherever tabu improves the start at all, which a
reviewer will read as a straw-man comparator. The fix is not a change of the frozen design: a §12 entry that names the
mechanism and states that H1c is affected (its interpretation, §10), plus, if wanted, an exploratory SA arm.

This test is the acceptance of that entry: a §12 entry that ties SA, H1c and the penalty (barrier) together.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _section_12_entries() -> list[str]:
    text = (ROOT / "docs" / "ANALYSIS_PLAN.md").read_text(encoding="utf-8")
    start = text.index("## 12. Журнал отклонений")
    section = text[start:text.index("## Приложение А", start)]
    entries: list[str] = []
    for line in section.splitlines():
        if line.startswith("- "):
            entries.append(line[2:])
        elif entries and line.startswith("  "):
            entries[-1] += " " + line.strip()
    return entries


def test_h1c_1_the_sa_arm_is_degenerate_in_the_calibration():
    """The fact (passes): no SA configuration improved the start; every SA score of the gap criterion ties."""
    screen = json.loads((ROOT / "configs" / "E2" / "pilot" / "screen.json").read_text(encoding="utf-8"))
    warnings = " ".join(screen["warnings"])
    for size in ("100", "200", "400"):
        assert f"size {size}, sa/gap: sa-00 and sa-01 tie exactly" in warnings
        assert "30 of 30 configurations lie in the 5 % tie band" in warnings


def test_h1c_1_section_12_records_that_h1c_is_affected_by_the_penalty_barrier_of_sa():
    hits = [e for e in _section_12_entries()
            if "SA" in e and "H1c" in e and re.search(r"штраф|барьер", e)]
    assert hits, (
        "§12 discloses that SA never improved the start but says 'гипотезы не затронуты'; no entry records that "
        "in the frozen penalty space (multipliers >= 1 x selection_penalty_scale) SA does not minimise the surrogate "
        "and that this fixes the H1c contrast"
    )
