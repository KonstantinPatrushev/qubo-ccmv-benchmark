"""Audit G1, finding RQ2-1 (MAJOR, 2026-10-12): the confirmatory estimator of H2 / H2b is not in the code.

Report: docs/reviews/2026-10-12-audit-g1.md.

ANALYSIS_PLAN §2 RQ2 ("Подтверждающая оценка (N8)"), §7.1 (rows H2, H2b), §7.2, §10, §12 A2/A3: the confirmatory
answer of RQ2 is the mean of ``log R`` (and of ``log R_b``) over the cells C1, C2, C3, C5 (C4 only if admissible) with
equal cell weights, ORlib inside C5 as the mean over ports of the means over λ, a 95 % CI by the stratified bootstrap
(instances within cells, ports in C5), the cells with simultaneous 99 % (Bonferroni) intervals, and the five outcomes
against the band ``[1/1.5, 1.5]``; ``R`` is the ratio of PI⁺ (A2), medians over seeds.

At 90dd64f the only confirmatory analysis of E2 is H1c (``python -m cif.experiments.e2_analysis h1c``);
configs/E2/README.md says so: "The statistics (the pooled estimate of ``log R`` with the stratified bootstrap, the five
outcomes, performance profiles, TTT curves) are not here". E1 (H1a) and E3 (H3a-H3c) have their analysis code, tested
on synthetic rows, before their confirmatory runs; the core estimate of the paper (RQ2) would be written after the
test-seed rows of E2 exist, with its open choices (how a SCIP row with PI⁺ = 0 enters, the reference of a test
instance, the port clusters of C5, the mapping to the five outcomes) made with the data in view.

Acceptance: the analysis stage exists before any bench run on test seeds (here: the ``h2`` command of
``cif.experiments.e2_analysis``; another name is fine if this test is adjusted with it).
"""

from __future__ import annotations

import pytest

from cif.experiments import e2_analysis


def test_rq2_1_the_pooled_estimate_of_h2_has_an_analysis_stage():
    with pytest.raises(SystemExit) as exc:
        e2_analysis.main(["h2", "--help"])
    assert exc.value.code == 0, (
        "e2_analysis has no stage for the confirmatory estimate of H2/H2b (mean log R over C1, C2, C3, C5 with "
        "the stratified bootstrap and the five outcomes, plan §2 RQ2, §7.1): only `h1c` exists"
    )
