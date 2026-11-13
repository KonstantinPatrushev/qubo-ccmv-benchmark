"""Confirmatory audit 2026-10-13, finding C-1 (MAJOR): the band outcome of H2b ("QUBO worse") rests on a C5 cell of
three clusters, whose percentile bootstrap interval is the range of three port means.

Report: docs/reviews/2026-10-13-audit-confirmatory.md.

The pooled estimate (plan §2 RQ2, N8; §12 RQ2-1 (3), (5)) averages the cells C1, C2, C3 and C5 with equal weights; in
C5 the cluster is the port and only ports 2, 3 and 4 have non-trivial binding points (5, 10 and 12 points). With three
clusters the stratified bootstrap of C5 draws its mean from 10 multisets of the three port means; it under-states the
variance of the mean by the factor 2/3 and has no tails beyond the extreme port means (``cif.stats.rq2`` says so in its
docstring: the C5 interval "is then the range of the port means"). C5 carries 58 % of the bootstrap variance of the
pooled ``log R_b`` (two thirds with the unbiased variance of a mean of three). The primary analysis ``results/E2/analysis/h2_20261013_complete`` reports
H2b = 2.62 (95 % CI 1.71-4.08) -> ``qubo_worse``. Without C5 (the same estimator, cells C1-C3) the interval is
1.25-2.75 -> ``direction_only``; with an unbiased C5 variance and a Welch t interval (df ~ 4.5) it is 1.31-5.24 ->
``direction_only``. H2 (``qubo_better``) is robust to both. The cell statements of the paper for OR-Library ("clear
advantage", "intervals excluding 1" at "simultaneous 99 %") are the range of the three port means, not a 99.75 %
interval.

Acceptance (no change of the frozen estimator is asked): either the band outcome of H2b does not depend on C5, or the
primary report carries a warning that names the three clusters of C5 and their effect (and the paper qualifies H2b and
the C5 cell statements accordingly).

Resolution (2026-10-17, §12): ``e2_analysis h2`` warns about every pooled cell with fewer than five clusters and gives
the pooled estimate without it (post hoc); the tests read the primary analysis recomputed after the fixes of the G2
conditions, ``h2_20261017`` (they skip until it exists).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
H2_DIR = ROOT / "results" / "E2" / "analysis" / "h2_20261017"


def _instances() -> list[dict]:
    path = H2_DIR / "instances.csv"
    if not path.exists():
        pytest.skip(f"{path} is absent")
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["excluded"] in ("", "None", None)]


def _cells(rows: list[dict], names: tuple[str, ...]) -> dict:
    from cif.stats import rq2

    out = {}
    for cell in names:
        mine = [r for r in rows if r["cell"] == cell]
        clusters = [r["cluster"] for r in mine]
        _, log_r = rq2.mean_of_means([float(r["log_r"]) for r in mine], clusters)
        _, log_rb = rq2.mean_of_means([float(r["log_r_b"]) for r in mine], clusters)
        out[cell] = {"log_r": log_r, "log_r_b": log_rb}
    return out


def test_c1_c5_has_three_clusters_and_dominates_the_variance_of_h2b():
    """The fact (passes): C5 has three port clusters and carries most of the bootstrap variance of pooled log R_b."""
    rows = _instances()
    cells = _cells(rows, ("C1", "C2", "C3", "C5"))
    assert len(cells["C5"]["log_r_b"]) == 3
    var = {c: float(np.var(v["log_r_b"], ddof=0)) / len(v["log_r_b"]) for c, v in cells.items()}  # bootstrap variance
    assert var["C5"] / sum(var.values()) > 0.5


def test_c1_band_outcome_of_h2b_does_not_rest_on_the_three_port_cell_or_is_flagged():
    from cif.stats import rq2

    rows = _instances()
    report = json.loads((H2_DIR / "h2.json").read_text(encoding="utf-8"))
    primary = report["pooled"]["H2b"]["outcome"]
    without_c5 = rq2.estimate_rq2(_cells(rows, ("C1", "C2", "C3")), pooled=["C1", "C2", "C3"], seed=rq2.SEED)
    alt = without_c5.pooled["log_r_b"].outcome
    flagged = any("C5" in w and ("cluster" in w or "port" in w) for w in report.get("warnings", []))
    assert alt == primary or flagged, (
        f"H2b: primary outcome {primary!r} (C1, C2, C3, C5), but {alt!r} without the three-port cell C5 "
        f"(R_b = {math.exp(without_c5.pooled['log_r_b'].estimate):.3f}, 95 % CI "
        f"{math.exp(without_c5.pooled['log_r_b'].interval.low):.3f}-"
        f"{math.exp(without_c5.pooled['log_r_b'].interval.high):.3f}); the report does not warn that the C5 interval "
        "is the range of three port means"
    )
