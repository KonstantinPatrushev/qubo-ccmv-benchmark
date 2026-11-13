"""Confirmatory audit 2026-10-13, finding C-4 (MINOR): in the cold runs the "first own solution" of local search is the
first refinement of a partial greedy subset, not a solution the method holds.

Report: docs/reviews/2026-10-13-audit-confirmatory.md.

§5 asks for "задержка до первого собственного решения" as a secondary measure of RQ2, and §12 A4 (2) defines the honest
variant "γ = 1 until the method's own first solution" on the cold runs (``20261013-121922-E2-cold-4a6d5e``). The cold
PI ("own") is computed from the trajectory and is right: local search holds no portfolio until its greedy construction
(K·N refinements) ends, 0.6 s (N = 100) to 3.7 s (N = 400) after the start. But the row field ``first_own_s`` of local
search is set by ``_Search.evaluate`` on the first refinement of any subset (``src/cif/heuristics/local_search.py``,
``evaluate``: ``if self.first_own_s is None``), i.e. a one-asset subset of the greedy construction, with the box lifted
when it cannot reach the budget; on ORlib (Σz = K = 10, w_min = 0.01) that point is not even feasible. Its median is
0.001 s in every cell, against 0.06-0.57 s for the QUBO pipeline, whose ``first_own_s`` is the end of its first batch of
refined K-subsets. No analysis reports the delay yet (it is missing from ``secondary_20261013``), so no number is wrong
today; computed from these fields, it would show local search ~500 times faster than it is.

Acceptance: in a cold row the first own solution of local search is not earlier than the first point of its
trajectory (the first portfolio it would return if stopped).

Resolution (2026-10-17, §12): the field is fixed in the code for the runs after the fix (the end of the first greedy
construction in a cold run); the rows of ``20261013-121922-E2-cold-4a6d5e`` were recorded before it, so the delay of
§12 is computed from the trajectories and never from this field. The first test documents the recorded rows, the
second checks the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COLD = ROOT / "runs" / "20261013-121922-E2-cold-4a6d5e"


def test_c4_the_recorded_cold_rows_predate_the_fix():
    """The fact (passes): every cold local-search row of the recorded run reports the old field, before the first
    portfolio of its trajectory; this is why the delay is computed from the trajectories."""
    if not (COLD / "results.jsonl").exists():
        pytest.skip(f"{COLD / 'results.jsonl'} is absent (the public snapshot packs the raw rows: python scripts/unpack_runs.py)")
    from cif.experiments import e2_pilot

    rows = [r for r in e2_pilot.read_rows(COLD / "results.jsonl") if r["method"]["name"] == "ls"]
    assert rows, "no local-search rows in the cold run"
    early = [(r["instance_id"], r["first_own_s"], r["trajectory"][0][0]) for r in rows
             if r["trajectory"] and r["first_own_s"] is not None and r["first_own_s"] < r["trajectory"][0][0] - 1e-9]
    assert len(early) == len(rows)


def test_c4_first_own_solution_of_local_search_is_a_solution_it_holds():
    """The fix: a cold local search reports, as its first own solution, the first point of its trajectory."""
    from cif.heuristics.local_search import local_search
    from cif.instances.factor import factor_instance
    from cif.problem import PortfolioProblem

    n, K = 20, 5
    mu, sigma, _ = factor_instance(n, 1000)
    problem = PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(f"a{i}" for i in range(n)), risk_aversion=5.0,
                               w_min=0.0, w_max=1.0, cardinality=K)
    meta = local_search(problem, K, budget_s=60.0, seed=10000, num_restarts=1).solver_meta
    assert meta["trajectory"], "the construction did not end"
    assert (meta["first_own_s"], meta["first_own_f"]) == tuple(meta["trajectory"][0]), (
        f"first own solution at {meta['first_own_s']} s, first portfolio of the trajectory at "
        f"{meta['trajectory'][0][0]} s"
    )
