"""Audit RF-1 (MAJOR, 2026-10-08): the shared refinement is not shared yet.

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

Plan §2/§4 (N5): "Конвейер, LS и подзадачи OA используют одну реализацию refine: прямой вызов
Clarabel без CVXPY". Commit b9f15dd adds ``cif.classical.refine`` ("одна для всех методов"), but
no module calls it: the QUBO second stage ``cif.heuristics.pipeline.refine_candidates`` still
refines every subset with ``cif.classical.continuous.solve_continuous_mvo`` (CVXPY: 5.5 ms per
subset against 0.8 ms for ``refine_support`` at N = 100, K = 10 on the audit machine, checks.txt).
RQ1 attribution (E_pipe vs E_form) and the end-to-end budgets of RQ2/H1c would then compare
methods that refine with different code and at different cost.

The test forbids CVXPY in the second stage and checks that the result equals ``refine_problem``.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cif.classical.refine import refine_problem
from cif.heuristics.pipeline import refine_candidates
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem


def test_rf1_pipeline_second_stage_uses_the_shared_refine(monkeypatch):
    mu, sigma, _ = factor_instance(12, 90001)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(12)),
                               risk_aversion=5.0, w_max=0.4, cardinality=4)
    subsets = [(0, 1, 2, 3), (4, 5, 6, 7), (2, 5, 8, 11)]
    expected = min(refine_problem(problem, s, card_mode="le").objective for s in subsets)

    def no_cvxpy(self, *args, **kwargs):
        raise AssertionError("CVXPY called by the pipeline's second stage")

    monkeypatch.setattr(cp.Problem, "solve", no_cvxpy)
    best, stats = refine_candidates(problem, subsets, K=4, m=3)
    assert best is not None and stats["n_evaluated"] == 3, (
        f"second stage without CVXPY evaluated {stats['n_evaluated']} of 3 subsets"
    )
    assert best.objective == pytest.approx(expected, rel=1e-9, abs=1e-12)
    assert np.isclose(best.weights.sum(), 1.0)
