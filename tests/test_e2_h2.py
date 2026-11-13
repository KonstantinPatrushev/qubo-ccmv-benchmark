"""H2 and H2b: the confirmatory estimate of RQ2 from registered rows (ANALYSIS_PLAN §2 RQ2, §5, §7.1, §7.2, §10; §12
A1-A4, audit G1 finding RQ2-1).

``R = PI⁺(QUBO)/PI⁺(exact)`` and ``R_b = PI⁺(QUBO)/PI⁺(LS)`` per instance (medians over the solver seeds of PI⁺, §12
A2); the pooled estimate is the mean of ``log R`` over the cells C1, C2, C3, C5 (C4 if admissible) with equal cell
weights, C5 as the mean over ports of the means over λ, with a 95 % stratified-bootstrap interval; the cells get
simultaneous 99 % (Bonferroni) intervals; the interval is mapped on the five outcomes against the band [1/1.5, 1.5].

The rows here are synthetic. A row whose incumbent stands at the gap ``g`` (in units of the scale, certificate −1)
from the end of the common start (t₀ = 0) to the budget has ``PI⁺ = g``, so every ratio below is known by hand. The
statistics themselves are tested in ``tests/test_stats_rq2.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from cif.experiments import e2_analysis as an
from cif.experiments import e2_pilot as pilot
from cif.stats import rq2
from cif.stats.rq1 import wilson_interval

T = 60.0
F_STAR = -1.0
SCALE = 1.0
ROLE_METHOD = {"exact_rep": "scip", "qubo_rep": "swap_sa", "ls": "ls", "sa": "sa"}
SEEDS = {"exact_rep": 3, "qubo_rep": 10, "ls": 10}
SIZE = {"C1": 100, "C2": 200, "C3": 400, "C4": 200, "D1_100": 100, "D1_200": 200}


# --------------------------------------------------------------------------------------------
# Synthetic rows
# --------------------------------------------------------------------------------------------


def instance_id(cell: str, unit, size: int) -> str:
    if isinstance(unit, tuple):
        return f"orlib/{unit[0]}/{unit[1]:02d}"
    family = "dense" if cell.startswith("D1") else "boot" if cell == "C4" else "factor"
    return f"{family}/{size}/{unit}"


def gap_row(cell, unit, role, solver_seed, gap, *, t0=0.0, traj=(), budget_ok=True, size=None, config=None,
            budget=T, stage="bench", cold=False, **extra) -> dict:
    """A row of ``role`` on the instance ``unit`` whose common start is ``gap`` above the certificate (so the
    incumbent stands at that gap until the improvements of ``traj``)."""
    size = size or SIZE.get(cell, 100)
    f0 = F_STAR + gap * SCALE
    traj = [list(p) for p in traj]
    config = config or f"{role}-{size}"
    row = {
        "cell_key": f"{stage}|{cell}|{unit}|{role}|{solver_seed}|{config}", "stage": stage, "cell": cell,
        "n": size, "config_size": size, "method": {"name": ROLE_METHOD[role], "id": config}, "config_id": config,
        "roles": [role], "solver_seed": solver_seed, "budget_s": budget,
        "instance_id": instance_id(cell, unit, size), "n_assets": size, "K": 5, "s_cont": 11, "binding": True,
        "start_time_s": None if cold else t0, "start_f": None if cold else f0,
        "start_scale": SCALE, "trajectory": traj, "f_final": traj[-1][1] if traj else f0, "scale_final": SCALE,
        "f_at_budget": traj[-1][1] if traj else f0, "budget_ok": budget_ok,
    }
    if isinstance(unit, tuple):
        row.update(family="orlib", port=unit[0], lam_index=unit[1])
    else:
        row.update(family=instance_id(cell, unit, size).split("/")[0], seed=unit)
    if cold:
        row["cold"] = True
    row.update(extra)
    return row


def inst_rows(cell, unit, *, exact, qubo, ls, **kw) -> list[dict]:
    """The rows of the three representatives of one instance; ``exact``, ``qubo`` and ``ls`` are the gaps of
    their seeds."""
    rows = []
    for role, gaps in (("exact_rep", exact), ("qubo_rep", qubo), ("ls", ls)):
        rows += [gap_row(cell, unit, role, 20000 + i, g, **kw) for i, g in enumerate(gaps)]
    return rows


def ratio_rows(cell, unit, r, r_b, *, base=0.01, **kw) -> list[dict]:
    """An instance with R = ``r`` and R_b = ``r_b`` (flat gaps, all seeds alike)."""
    q = r * base
    return inst_rows(cell, unit, exact=[base] * SEEDS["exact_rep"], qubo=[q] * SEEDS["qubo_rep"],
                     ls=[q / r_b] * SEEDS["ls"], **kw)


def cert_rows(cell, units, *, size=None, stage="eligibility") -> list[dict]:
    """The certificates the screening run leaves: ``f_star`` of a certified optimum."""
    size = size or SIZE.get(cell, 100)
    return [{"cell_key": f"{stage}|{cell}|{u}", "stage": stage, "cell": cell,
             "instance_id": instance_id(cell, u, size), "f_star": F_STAR, "scale_star": SCALE,
             "ref_type": "certified", "f_final": F_STAR, "scale_final": SCALE, "start_f": F_STAR + 0.5,
             "start_scale": SCALE, "config_id": "ref"} for u in units]


def estimate_of(n_trivial, n_nontrivial, n_undecided=0, n_nonbinding=0) -> dict:
    n = n_trivial + n_nontrivial + n_undecided
    low, high = wilson_interval(n_trivial, n, 0.95)
    return {"n_scanned": n, "n_trivial": n_trivial, "n_nontrivial": n_nontrivial, "n_undecided": n_undecided,
            "n_nonbinding": n_nonbinding, "share_trivial": n_trivial / n if n else None, "ci95": [low, high],
            "g_start_bp": {"n": n, "median": 1.5, "q1": 0.5, "q3": 6.0, "max": 40.0, "n_exact": n} if n else None,
            "g_start_trivial_upper_bp_max": 0.9,
            "t_cert_trivial_s": {"n": n_trivial, "median": 3.4, "q1": 2.6, "q3": 5.5, "max": 57.7, "level": "full"}
            if n_trivial else None}


def elig(cells: dict[str, dict], members=None, *, role="test", statement=None, groups=None, excluded=None) -> dict:
    """A finished eligibility report. ``cells`` maps a label to ``selected`` (and optionally ``enough``, ``per_port``,
    ``estimate``); ``members`` are the cells of the pooled estimate (default: the cells with ``enough`` other than
    C4 and the D1 ones)."""
    states = {}
    for label, spec in cells.items():
        selected = [list(u) if isinstance(u, tuple) else u for u in spec["selected"]]
        enough = spec.get("enough", True)
        state = {"done": True, "enough": enough, "prefix": 400, "found": len(selected), "selected": selected,
                 "needs_reference": [], "descriptive": not enough,
                 "estimate": spec.get("estimate") or estimate_of(20, len(selected))}
        if "per_port" in spec:
            state["per_port"] = {str(p): n for p, n in spec["per_port"].items()}
            state["n_ports"] = sum(n > 0 for n in spec["per_port"].values())
        states[label] = state
    if members is None:
        members = [c for c, s in states.items() if s["enough"] and c != "C4" and not c.startswith("D1")]
    return {"kind": "e2-eligibility", "plan_version": "v2.4", "role": role, "tau": 1e-3, "done": True,
            "cells": states, "groups": groups or {},
            "summary": {"cells": list(members), "excluded": excluded or {}, "statement": statement}}


def two_cell_world(r1=(2.0, 2.0, 2.0), r2=(0.25, 0.25)):
    """C1 (N = 100) with instances of the given R and C2 (N = 200) with its own; R_b = 2 on every instance."""
    cells = {"C1": {"selected": list(range(2000, 2000 + len(r1)))},
             "C2": {"selected": list(range(2000, 2000 + len(r2)))}}
    rows, certs = [], []
    for cell, rs in (("C1", r1), ("C2", r2)):
        for unit, r in zip(cells[cell]["selected"], rs, strict=True):
            rows += ratio_rows(cell, unit, r, 2.0)
        certs += cert_rows(cell, cells[cell]["selected"])
    return rows, certs, elig(cells)


def find(instances, cell, unit):
    (hit,) = [i for i in instances if i["cell"] == cell and i["unit"] == unit]
    return hit


# --------------------------------------------------------------------------------------------
# The ratio of an instance
# --------------------------------------------------------------------------------------------


def test_the_ratio_of_an_instance_is_the_ratio_of_the_medians_over_the_seeds_of_pi_plus():
    rows = inst_rows("C1", 2000, exact=[0.010, 0.020, 0.090],
                     qubo=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], ls=[0.011] * 10)
    rows += inst_rows("C1", 2001, exact=[0.01] * 3, qubo=[0.01] * 10, ls=[0.01] * 10)
    report = elig({"C1": {"selected": [2000, 2001]}})
    inst = find(an.bench_instances(rows, [report], references=cert_rows("C1", [2000, 2001])), "C1", 2000)
    # medians: exact 0.020 (3 seeds), qubo (0.05 + 0.06) / 2 = 0.055 (10 seeds), LS 0.011
    assert inst["pi"] == {"exact": pytest.approx(0.020), "qubo": pytest.approx(0.055), "ls": pytest.approx(0.011)}
    assert inst["r"] == pytest.approx(2.75) and inst["r_b"] == pytest.approx(5.0)
    assert inst["log_r"] == pytest.approx(math.log(2.75)) and inst["log_r_b"] == pytest.approx(math.log(5.0))
    assert inst["n_seeds"] == {"exact": 3, "qubo": 10, "ls": 10} and inst["excluded"] is None
    assert inst["instance_id"] == "factor/100/2000" and inst["cluster"] == "2000"
    assert inst["config_ids"] == {"exact": "exact_rep-100", "qubo": "qubo_rep-100", "ls": "ls-100"}


def test_the_measure_is_the_primal_integral_after_the_common_start():
    """§12 A2: PI⁺ = PI − t₀/T. Start at t₀ = 3 s with gap 0.04, gap 0.01 from 10 s, 0 from 30 s:
    PI⁺ = (0.04·7 + 0.01·20 + 0·30) / 60 = 0.008; PI over [0, T] would add t₀/T = 0.05."""
    traj = [(10.0, F_STAR + 0.01), (30.0, F_STAR)]
    rows = [gap_row("C1", 2000, "exact_rep", 20000 + i, 0.02, t0=3.0) for i in range(3)]
    rows += [gap_row("C1", 2000, "qubo_rep", 20000 + i, 0.04, t0=3.0, traj=traj) for i in range(10)]
    rows += [gap_row("C1", 2000, "ls", 20000 + i, 0.04, t0=3.0) for i in range(10)]
    rows += inst_rows("C1", 2001, exact=[0.02] * 3, qubo=[0.04] * 10, ls=[0.04] * 10, t0=3.0)
    certs = cert_rows("C1", [2000, 2001])
    inst = find(an.bench_instances(rows, [elig({"C1": {"selected": [2000, 2001]}})], references=certs), "C1", 2000)
    assert inst["pi"]["qubo"] == pytest.approx(0.008, abs=1e-15)
    assert inst["pi"]["exact"] == pytest.approx(0.02 * 57.0 / 60.0, abs=1e-15)
    ref = pilot.reference_table([*rows, *certs])["factor/100/2000"]
    row = next(r for r in rows if r["seed"] == 2000 and r["roles"] == ["qubo_rep"])
    assert inst["pi"]["qubo"] == pilot.row_pi(row, ref, budget_s=T)  # the row measure of the pilot, not a copy
    assert pilot.row_pi(row, ref, variant="full") == pytest.approx(0.008 + 3.0 / 60.0, abs=1e-15)
    assert inst["t0_s"] == 3.0


def test_the_reference_is_the_certificate_and_a_method_below_it_replaces_it_with_a_flag():
    """§5: the reference is the best value known over all methods. The certificate stays unless a method is below it
    by more than 1e-6·s (then the best known value is the reference and the instance is flagged)."""
    rows = [gap_row("C1", 2000, "exact_rep", 20000 + i, 0.10) for i in range(3)]
    rows += [gap_row("C1", 2000, "qubo_rep", 20000 + i, 0.10, traj=[(20.0, F_STAR - 0.05)]) for i in range(10)]
    rows += [gap_row("C1", 2000, "ls", 20000 + i, 0.10) for i in range(10)]
    rows += inst_rows("C1", 2001, exact=[0.10] * 3, qubo=[0.10] * 10, ls=[0.10] * 10)
    insts = an.bench_instances(rows, [elig({"C1": {"selected": [2000, 2001]}})],
                               references=cert_rows("C1", [2000, 2001]))
    below, held = find(insts, "C1", 2000), find(insts, "C1", 2001)
    assert (below["ref_type"], below["ref_inconsistent"], below["f_ref"]) == ("certified", True, F_STAR - 0.05)
    assert (held["ref_type"], held["ref_inconsistent"], held["f_ref"]) == ("certified", False, F_STAR)
    # against the lower reference the gap of the exact method is 0.15 and the QUBO method reaches it at 20 s
    assert below["pi"]["exact"] == pytest.approx(0.15) and below["pi"]["qubo"] == pytest.approx(0.15 * 20.0 / 60.0)
    assert held["pi"]["exact"] == pytest.approx(0.10)
    assert held["g_start"] == pytest.approx(0.10) and below["g_start"] == pytest.approx(0.15)


def test_without_a_certificate_the_best_known_value_is_the_reference_and_the_instance_stays_in():
    """The plan does not drop uncertified instances (§5). A heuristic that is no representative (role ``sa``) still
    counts as a method that knew a value."""
    rows = []
    for unit in (2000, 2001):
        rows += inst_rows("C1", unit, exact=[0.10] * 3, qubo=[0.04] * 10, ls=[0.10] * 10)
        rows.append(gap_row("C1", unit, "sa", 20000, 0.02))
    insts = an.bench_instances(rows, [elig({"C1": {"selected": [2000, 2001]}})])
    for inst in insts:
        assert inst["ref_type"] == "best_known" and inst["ref_inconsistent"] is False
        assert inst["f_ref"] == pytest.approx(F_STAR + 0.02)
        assert inst["pi"] == {"exact": pytest.approx(0.08), "qubo": pytest.approx(0.02), "ls": pytest.approx(0.08)}
    out = an.h2_report(rows, [elig({"C1": {"selected": [2000, 2001]}})])
    assert out["references"]["by_type"] == {"best_known": 2}
    assert any("certif" in w for w in out["warnings"])


def test_only_certificates_of_the_screening_and_reference_stages_count_as_references():
    rows, certs, report = two_cell_world()
    cold = [gap_row("C1", 2000, "qubo_rep", 20000, 0.05, traj=[(5.0, F_STAR - 0.3)], stage="cold", cold=True)]
    insts = an.bench_instances(rows, [report], references=[*certs, *cold])
    assert find(insts, "C1", 2000)["f_ref"] == F_STAR  # a cold or h1c row passed as a "reference" is not one
    out = an.h2_report(rows, [report], references=[*certs, *cold])
    assert out["references"]["n_rows_used"] == len(certs) and out["references"]["n_rows_ignored"] == 1


# --------------------------------------------------------------------------------------------
# The pooled estimate
# --------------------------------------------------------------------------------------------


def test_every_cell_has_the_same_weight_in_the_pooled_log_r():
    """C1: three instances with R = 2; C2: two with R = 1/4. Pooled mean of log R = (ln 2 + ln 1/4)/2 =
    −0.5·ln 2; weighting the five instances alike would give −0.2·ln 2."""
    rows, certs, report = two_cell_world()
    out = an.h2_report(rows, [report], references=certs)
    pooled = out["pooled"]
    assert pooled["cells"] == ["C1", "C2"] and pooled["weights"] == "equal"
    assert pooled["H2"]["estimate_log"] == pytest.approx(-0.5 * math.log(2.0), abs=1e-12)
    assert pooled["H2"]["ratio"] == pytest.approx(2.0**-0.5)
    assert pooled["H2b"]["estimate_log"] == pytest.approx(math.log(2.0), abs=1e-12)
    assert out["cells"]["C1"]["H2"]["estimate_log"] == pytest.approx(math.log(2.0))
    assert out["cells"]["C2"]["H2"]["estimate_log"] == pytest.approx(math.log(0.25))
    assert out["cells"]["C1"]["n_instances"] == 3 and out["cells"]["C2"]["n_clusters"] == 2
    assert out["cells"]["C1"]["cluster"] == "instance"


def test_c5_is_the_mean_over_ports_of_the_means_over_the_lambda_points():
    """Port 1 has the points λ0 (R = 2) and λ1 (R = 8), port 2 the point λ3 (R = 1/2): the port means are ln 4
    and −ln 2, the cell mean 0.5·ln 2 (the mean over the three points would be ln 2)."""
    points = [(1, 0), (1, 1), (2, 3)]
    rows = (ratio_rows("C5", (1, 0), 2.0, 2.0, size=31) + ratio_rows("C5", (1, 1), 8.0, 2.0, size=31)
            + ratio_rows("C5", (2, 3), 0.5, 2.0, size=85))
    certs = cert_rows("C5", [(1, 0), (1, 1)], size=31) + cert_rows("C5", [(2, 3)], size=85)
    report = elig({"C5": {"selected": points, "per_port": {1: 2, 2: 1, 3: 0}}})
    out = an.h2_report(rows, [report], references=certs)
    c5 = out["cells"]["C5"]
    assert c5["cluster"] == "port" and c5["n_clusters"] == 2 and c5["n_instances"] == 3
    assert c5["H2"]["estimate_log"] == pytest.approx(0.5 * math.log(2.0), abs=1e-12)
    assert out["pooled"]["cells"] == ["C5"]
    assert out["pooled"]["H2"]["estimate_log"] == pytest.approx(0.5 * math.log(2.0), abs=1e-12)
    # the ports of the report that have no selected point are listed, not imputed
    assert c5["points_per_port"] == {"1": 2, "2": 1, "3": 0} and c5["ports_without_points"] == ["3"]
    insts = an.bench_instances(rows, [report], references=certs)
    assert {i["cluster"] for i in insts} == {"port1", "port2"}
    assert find(insts, "C5", (1, 1))["instance_id"] == "orlib/1/01"


def test_a_cell_outside_the_pooled_set_is_reported_without_entering_the_pooled_estimate():
    """C3 has fewer than 20 non-trivial instances (descriptive, §12 A1) and C4 is descriptive by the λ rule."""
    cells = {"C1": {"selected": [2000, 2001]}, "C3": {"selected": [2000, 2001], "enough": False},
             "C4": {"selected": [2000, 2001]}}
    rows, certs = [], []
    for cell, r in (("C1", 2.0), ("C3", 8.0), ("C4", 0.1)):
        for u in (2000, 2001):
            rows += ratio_rows(cell, u, r, 2.0)
        certs += cert_rows(cell, [2000, 2001])
    why = {"C3": "descriptive: 2 non-trivial instances, fewer than the 20 the plan needs",
           "C4": "descriptive: the λ rule of N7 found no admissible λ"}
    out = an.h2_report(rows, [elig(cells, members=["C1"], excluded=why)], references=certs)
    assert out["descriptive"] == why  # the reasons of the eligibility report travel with the cells
    assert out["pooled"]["cells"] == ["C1"]
    assert out["pooled"]["H2"]["estimate_log"] == pytest.approx(math.log(2.0))
    assert out["cells"]["C1"]["in_pooled"] is True
    for cell, r in (("C3", 8.0), ("C4", 0.1)):
        assert out["cells"][cell]["in_pooled"] is False
        assert out["cells"][cell]["H2"]["estimate_log"] == pytest.approx(math.log(r))
    # the Bonferroni family is the pooled cells: a family of one at the 99 % level, descriptive cells at the same level
    assert out["cells"]["C1"]["H2"]["level"] == pytest.approx(0.99) == out["cells"]["C3"]["H2"]["level"]


def test_the_cells_of_the_pooled_estimate_get_the_bonferroni_level_of_their_family():
    cells = {c: {"selected": [2000, 2001]} for c in ("C1", "C2", "C3")}
    cells["C5"] = {"selected": [(1, 0), (2, 0)], "per_port": {1: 1, 2: 1}}
    rows, certs = [], []
    for c in ("C1", "C2", "C3"):
        for u in (2000, 2001):
            rows += ratio_rows(c, u, 1.1, 1.1)
        certs += cert_rows(c, [2000, 2001])
    for u, size in (((1, 0), 31), ((2, 0), 85)):
        rows += ratio_rows("C5", u, 1.1, 1.1, size=size)
        certs += cert_rows("C5", [u], size=size)
    out = an.h2_report(rows, [elig(cells)], references=certs)
    assert out["pooled"]["cells"] == ["C1", "C2", "C3", "C5"] and out["pooled"]["n_cells"] == 4
    assert out["pooled"]["H2"]["level"] == pytest.approx(0.95)
    for cell in ("C1", "C2", "C3", "C5"):
        assert out["cells"][cell]["H2"]["level"] == pytest.approx(0.9975)
    assert out["settings"]["family_size"] == 4 and out["settings"]["cell_level"] == pytest.approx(0.9975)


def test_a_pooled_cell_with_fewer_than_five_clusters_is_flagged_with_the_pooled_estimate_without_it():
    """Audit G2, C-1 (§12 2026-10-17): with three port clusters the percentile interval of C5 is the range of the
    three port means. The report warns about every pooled cell with fewer than five clusters and gives, post hoc, the
    pooled estimate without that cell; the pooled estimate itself is unchanged (the estimator is frozen)."""
    cells = {"C1": {"selected": list(range(2000, 2005))},
             "C5": {"selected": [(1, 0), (2, 0), (3, 0)], "per_port": {1: 1, 2: 1, 3: 1}}}
    rows, certs = [], cert_rows("C1", cells["C1"]["selected"])
    for u in cells["C1"]["selected"]:
        rows += ratio_rows("C1", u, 2.0, 2.0)
    for u, size in (((1, 0), 31), ((2, 0), 85), ((3, 0), 89)):
        rows += ratio_rows("C5", u, 0.5, 4.0, size=size)
        certs += cert_rows("C5", [u], size=size)
    out = an.h2_report(rows, [elig(cells)], references=certs)
    assert out["pooled"]["cells"] == ["C1", "C5"]
    assert out["pooled"]["H2"]["estimate_log"] == pytest.approx(0.0, abs=1e-12)  # (ln 2 + ln 1/2)/2
    (warning,) = [w for w in out["warnings"] if "clusters" in w]
    assert warning.startswith("C5 enters the pooled estimate with 3 port clusters")
    few = out["few_clusters"]
    assert set(few) == {"C5"}
    assert few["C5"]["n_clusters"] == 3 and few["C5"]["cluster"] == "port" and "post hoc" in few["C5"]["role"]
    without = few["C5"]["without"]
    assert without["cells"] == ["C1"]
    assert without["H2"]["estimate_log"] == pytest.approx(math.log(2.0))
    assert without["H2b"]["estimate_log"] == pytest.approx(math.log(2.0))
    pooled = an._tables(out)["pooled.csv"]
    assert set(pooled["variant"]) == {"primary", "full", "first_improvement", "within_budget", "without_C5"}
    assert pooled.loc[(pooled["variant"] == "without_C5") & (pooled["hypothesis"] == "H2b"), "cells"].item() == "C1"


def test_a_pooled_cell_that_is_alone_in_the_pool_is_flagged_without_a_pooled_estimate_without_it():
    report = elig({"C5": {"selected": [(1, 0), (2, 0)], "per_port": {1: 1, 2: 1}}})
    rows = ratio_rows("C5", (1, 0), 2.0, 2.0, size=31) + ratio_rows("C5", (2, 0), 2.0, 2.0, size=85)
    certs = cert_rows("C5", [(1, 0)], size=31) + cert_rows("C5", [(2, 0)], size=85)
    out = an.h2_report(rows, [report], references=certs)
    assert out["few_clusters"]["C5"]["without"] is None
    assert any(w.startswith("C5 enters the pooled estimate with 2 port clusters") and "only pooled cell" in w
               for w in out["warnings"])


def test_when_no_cell_is_pooled_the_answer_of_rq2_is_the_share_of_trivial_instances():
    statement = "Нетривиальных инстансов недостаточно для сводной оценки; RQ2 отвечает долей тривиальных"
    report = elig({"C1": {"selected": [2000, 2001], "enough": False}}, members=[], statement=statement)
    rows = ratio_rows("C1", 2000, 2.0, 2.0) + ratio_rows("C1", 2001, 2.0, 2.0)
    out = an.h2_report(rows, [report], references=cert_rows("C1", [2000, 2001]))
    assert out["pooled"] is None and out["cells"]["C1"]["in_pooled"] is False
    assert out["statement"] == statement
    assert out["a3"]["cells"]["C1"]["share_trivial"] == pytest.approx(20 / 22)
    # with no instance selected at all there are no rows and the report is the share only
    empty = elig({"C1": {"selected": [], "enough": False}}, members=[], statement=statement)
    out = an.h2_report([], [empty])
    assert out["pooled"] is None and out["cells"]["C1"]["n_instances"] == 0 and out["cells"]["C1"]["H2"] is None
    assert out["statement"] == statement and out["instances"] == []


# --------------------------------------------------------------------------------------------
# The five outcomes
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pair", "outcome", "text"),
    [
        ((0.30, 0.50), "qubo_better", "быстрее приближается к оптимуму"),
        ((2.00, 3.00), "qubo_worse", "не уступает и превосходит QUBO"),
        ((0.80, 1.20), "equivalent", "Практически эквивалентны (в пределах ×1.5)"),
        ((1.20, 2.00), "direction_only", "величина относительно полосы ×1.5 не определена"),
        ((0.80, 2.00), "undetermined", "Данные не позволяют различить методы"),
    ],
)
def test_h2_and_h2b_are_mapped_on_the_five_outcomes_independently(pair, outcome, text):
    """Two instances make the percentile interval of the pooled mean the exact range of their log R. H2 gets
    ``pair``; with R_b = 1/R the interval of H2b is its mirror image, which swaps better and worse."""
    rows = []
    for unit, r in zip((2000, 2001), pair, strict=True):
        rows += ratio_rows("C1", unit, r, 1.0 / r)
    out = an.h2_report(rows, [elig({"C1": {"selected": [2000, 2001]}})], references=cert_rows("C1", [2000, 2001]))
    h2, h2b = out["pooled"]["H2"], out["pooled"]["H2b"]
    assert h2["outcome"] == outcome and text in h2["statement"]
    assert "на инстансах, где общий старт не достигает τ" in h2["statement"]
    assert h2["ci_log"] == pytest.approx([math.log(pair[0]), math.log(pair[1])], abs=1e-12)
    assert h2b["outcome"] == {"qubo_better": "qubo_worse", "qubo_worse": "qubo_better"}.get(outcome, outcome)
    assert h2b["ci_log"] == pytest.approx([-math.log(pair[1]), -math.log(pair[0])], abs=1e-12)
    assert out["cells"]["C1"]["H2"]["outcome"] == outcome  # the cell interval is the same range here


def test_h2b_has_its_own_wording_about_local_search():
    rows = ratio_rows("C1", 2000, 2.0, 2.0) + ratio_rows("C1", 2001, 3.0, 2.5)
    out = an.h2_report(rows, [elig({"C1": {"selected": [2000, 2001]}})], references=cert_rows("C1", [2000, 2001]))
    h2, h2b = out["pooled"]["H2"], out["pooled"]["H2b"]
    assert h2b["outcome"] == "qubo_worse"
    assert "Локальный поиск не уступает и превосходит QUBO-конвейер" in h2b["statement"]
    assert "Точный метод не уступает и превосходит QUBO-конвейер" in h2["statement"]


# --------------------------------------------------------------------------------------------
# Budgets, seeds, sensitivities
# --------------------------------------------------------------------------------------------


def _overrun_exact(unit):
    return [gap_row("C1", unit, "exact_rep", 20000, 0.01),
            gap_row("C1", unit, "exact_rep", 20001, 0.30, budget_ok=False),
            gap_row("C1", unit, "exact_rep", 20002, 0.40, budget_ok=False)]


def test_rows_over_the_budget_are_used_as_measured_counted_and_dropped_in_the_sensitivity():
    rows = []
    for unit in (2000, 2001):
        rows += _overrun_exact(unit)
        rows += [gap_row("C1", unit, role, 20000 + i, 0.02) for role in ("qubo_rep", "ls") for i in range(10)]
    certs = cert_rows("C1", [2000, 2001])
    report = elig({"C1": {"selected": [2000, 2001]}})
    primary = find(an.bench_instances(rows, [report], references=certs), "C1", 2000)
    assert primary["pi"]["exact"] == pytest.approx(0.30)  # the median of all three seeds
    assert primary["n_over_budget"] == {"exact": 2, "qubo": 0, "ls": 0}
    within = find(an.bench_instances(rows, [report], references=certs, drop_over_budget=True), "C1", 2000)
    assert within["pi"]["exact"] == pytest.approx(0.01) and within["n_seeds"]["exact"] == 1
    out = an.h2_report(rows, [report], references=certs)
    assert out["budget"]["n_over_budget"]["C1"] == {"exact": 4, "qubo": 0, "ls": 0}
    assert out["cells"]["C1"]["H2"]["estimate_log"] == pytest.approx(math.log(0.02 / 0.30))
    assert out["sensitivity"]["within_budget"]["pooled"]["H2"]["estimate_log"] == pytest.approx(math.log(0.02 / 0.01))


def test_an_instance_without_a_row_inside_the_budget_is_left_out_of_that_sensitivity_and_counted():
    rows = []
    for unit, q in ((2000, 0.02), (2001, 0.04), (2002, 0.08)):
        rows += inst_rows("C1", unit, exact=[0.01] * 3, qubo=[q] * 10, ls=[0.02] * 10)
    for r in rows:
        if r["seed"] == 2002 and r["roles"] == ["exact_rep"]:
            r["budget_ok"] = False
    certs = cert_rows("C1", [2000, 2001, 2002])
    report = elig({"C1": {"selected": [2000, 2001, 2002]}})
    gone = find(an.bench_instances(rows, [report], references=certs, drop_over_budget=True), "C1", 2002)
    assert gone["excluded"] and "exact" in gone["excluded"] and gone["log_r"] is None
    within = an.h2_report(rows, [report], references=certs)["sensitivity"]["within_budget"]
    assert within["n_excluded"] == {"C1": 1} and within["cells"]["C1"]["n_instances"] == 2


def test_the_sensitivities_of_a4_use_the_same_rows_with_the_other_definitions_of_the_integral():
    """QUBO gap 0.04 against exact gap 0.01 with a start of 0.6 s: PI⁺ ratio 4. PI over [0, T] adds t₀/T = 0.01 to
    both. 'First improvement' counts γ = 1 until the first improvement of the start (here none, so until T)."""
    rows = []
    for unit in (2000, 2001):
        rows += inst_rows("C1", unit, exact=[0.01] * 3, qubo=[0.04] * 10, ls=[0.04] * 10, t0=0.6)
    out = an.h2_report(rows, [elig({"C1": {"selected": [2000, 2001]}})], references=cert_rows("C1", [2000, 2001]))
    assert out["pooled"]["H2"]["ratio"] == pytest.approx(4.0)
    full = out["sensitivity"]["full"]["pooled"]["H2"]
    assert full["ratio"] == pytest.approx((0.04 * 59.4 / 60 + 0.01) / (0.01 * 59.4 / 60 + 0.01))
    first = out["sensitivity"]["first_improvement"]["pooled"]["H2"]
    assert first["ratio"] == pytest.approx(1.0)  # nobody improves the start: PI = 1 for all, as the plan warns (A4)
    assert set(out["sensitivity"]) == {"full", "first_improvement", "within_budget"}
    assert "not the confirmatory outcome" in out["sensitivity"]["full"]["role"]


def test_the_honest_variant_of_a4_comes_from_the_cold_rows():
    """Cold rows (no common start) list everything from the method's first solution; PI is taken from time 0:
    γ = 1 to the first solution, then the gap. QUBO: first solution at 5 s with gap 0.04, exact: at 2 s with 0.02."""
    rows, cold = [], []
    for unit in (2000, 2001):
        rows += ratio_rows("C1", unit, 2.0, 2.0)
        for role, t_first, g in (("exact_rep", 2.0, 0.02), ("qubo_rep", 5.0, 0.04), ("ls", 5.0, 0.04)):
            cold.append(gap_row("C1", unit, role, 20000, 0.0, traj=[(t_first, F_STAR + g)], stage="cold", cold=True))
    certs = cert_rows("C1", [2000, 2001])
    report = elig({"C1": {"selected": [2000, 2001]}})
    out = an.h2_report(rows, [report], references=certs, cold_rows=cold)
    own = out["sensitivity"]["own"]["pooled"]["H2"]
    assert own["ratio"] == pytest.approx(((5.0 + 0.04 * 55.0) / T) / ((2.0 + 0.02 * 58.0) / T))
    with pytest.raises(ValueError, match="no rows"):  # an instance without its cold rows stops the analysis
        an.h2_report(rows, [report], references=certs, cold_rows=cold[:-3])
    assert "own" not in an.h2_report(rows, [report], references=certs)["sensitivity"]


# --------------------------------------------------------------------------------------------
# What stops the analysis
# --------------------------------------------------------------------------------------------


def test_rows_of_an_instance_the_screening_did_not_select_are_an_error():
    rows, certs, report = two_cell_world()
    with pytest.raises(ValueError, match="not selected"):
        an.bench_instances([*rows, *ratio_rows("C1", 2099, 2.0, 2.0)], [report], references=certs)
    with pytest.raises(ValueError, match="no eligibility report"):
        an.bench_instances([*rows, *ratio_rows("C4", 2000, 2.0, 2.0)], [report], references=certs)


def test_a_selected_instance_without_rows_or_without_a_representative_is_an_error():
    rows, certs, report = two_cell_world()
    with pytest.raises(ValueError, match=r"no rows.*C1.*2002"):
        an.bench_instances([r for r in rows if not (r["cell"] == "C1" and r["seed"] == 2002)], [report],
                           references=certs)
    no_ls = [r for r in rows if not (r["cell"] == "C2" and r["seed"] == 2001 and r["roles"] == ["ls"])]
    with pytest.raises(ValueError, match=r"role 'ls'.*C2.*2001"):
        an.bench_instances(no_ls, [report], references=certs)


def test_every_instance_of_a_cell_must_have_the_same_number_of_seeds_per_role():
    rows, certs, report = two_cell_world()
    short = [r for r in rows if not (r["cell"] == "C1" and r["seed"] == 2001 and r["roles"] == ["qubo_rep"]
                                     and r["solver_seed"] >= 20007)]
    with pytest.raises(ValueError, match="seeds"):
        an.bench_instances(short, [report], references=certs)


def test_one_configuration_per_role_in_a_cell_and_size():
    rows, certs, report = two_cell_world()
    rows = [dict(r, config_id="other", method={**r["method"], "id": "other"}) if (
        r["cell"] == "C1" and r["seed"] == 2001 and r["roles"] == ["ls"]) else r for r in rows]
    with pytest.raises(ValueError, match="configuration"):
        an.bench_instances(rows, [report], references=certs)


def test_a_vanishing_pi_plus_is_an_error_naming_the_instance_not_a_floored_ratio():
    rows = ratio_rows("C1", 2000, 2.0, 2.0) + ratio_rows("C1", 2001, 2.0, 2.0)
    rows = [dict(r, start_f=F_STAR, f_final=F_STAR, f_at_budget=F_STAR) if (
        r["seed"] == 2001 and r["roles"] == ["qubo_rep"]) else r for r in rows]
    with pytest.raises(ValueError, match=r"PI\+ = 0.*C1/2001.*qubo"):
        an.bench_instances(rows, [elig({"C1": {"selected": [2000, 2001]}})], references=cert_rows("C1", [2000, 2001]))


def test_rows_must_be_bench_rows_of_one_budget_without_errors_or_skips():
    rows, certs, report = two_cell_world()
    with pytest.raises(ValueError, match="budget"):
        an.bench_instances([dict(rows[0], budget_s=30.0), *rows[1:]], [report], references=certs)
    with pytest.raises(ValueError, match="stage"):
        an.bench_instances([dict(rows[0], stage="h1c"), *rows[1:]], [report], references=certs)
    with pytest.raises(ValueError, match="error"):
        an.bench_instances([dict(rows[0], error="RuntimeError: boom"), *rows[1:]], [report], references=certs)
    with pytest.raises(ValueError, match="skipped"):
        an.bench_instances([dict(rows[0], skipped="not_binding"), *rows[1:]], [report], references=certs)
    # a different budget is a different analysis (exploratory, §9), allowed only when it is asked for
    other = [dict(r, budget_s=30.0) for r in rows]
    assert an.bench_instances(other, [report], references=certs, budget_s=30.0)


def test_the_eligibility_reports_must_be_finished_reports_of_the_test_scan():
    rows, certs, report = two_cell_world()
    stub = {"kind": "e2-eligibility", "stub": True, "role": "test", "cells": {}}
    with pytest.raises(ValueError, match="stub"):
        an.h2_report(rows, [stub], references=certs)
    unfinished = json.loads(json.dumps(report))
    unfinished["cells"]["C2"]["done"] = False
    with pytest.raises(ValueError, match="not finished"):
        an.h2_report(rows, [unfinished], references=certs)
    with pytest.raises(ValueError, match="role"):
        an.h2_report(rows, [dict(report, role="calibration")], references=certs)
    with pytest.raises(ValueError, match="kind"):
        an.h2_report(rows, [dict(report, kind="e2-n-rule")], references=certs)


def test_the_pooled_set_must_agree_with_the_cells_that_have_enough_instances():
    rows, certs, report = two_cell_world()
    broken = json.loads(json.dumps(report))
    broken["summary"]["cells"] = ["C1", "C2", "D1_100"]
    with pytest.raises(ValueError, match="D1"):
        an.h2_report(rows, [broken], references=certs)
    thin = json.loads(json.dumps(report))
    thin["cells"]["C2"]["enough"] = False
    with pytest.raises(ValueError, match="enough"):
        an.h2_report(rows, [thin], references=certs)
    left_out = json.loads(json.dumps(report))
    left_out["summary"]["cells"] = ["C1"]
    with pytest.raises(ValueError, match="C2"):
        an.h2_report(rows, [left_out], references=certs)
    no_summary = json.loads(json.dumps(report))
    del no_summary["summary"]
    with pytest.raises(ValueError, match="summary"):
        an.h2_report(rows, [no_summary], references=certs)


# --------------------------------------------------------------------------------------------
# A3 and the report
# --------------------------------------------------------------------------------------------


def test_a3_is_the_share_of_trivial_instances_per_cell_as_the_screening_recorded_it():
    cells = {
        "C1": {"selected": [2000, 2001], "estimate": estimate_of(30, 3)},
        "C2": {"selected": [2000, 2001], "estimate": estimate_of(0, 2)},
        "C5": {"selected": [(1, 0), (2, 0)], "per_port": {1: 1, 2: 1}, "estimate": estimate_of(60, 2, n_nonbinding=63)},
    }
    a3 = an.a3_estimates([elig(cells)])
    c1 = a3["cells"]["C1"]
    assert c1["n_scanned"] == 33 and c1["n_trivial"] == 30 and c1["share_trivial"] == pytest.approx(30 / 33)
    low, high = wilson_interval(30, 33, 0.95)
    assert c1["ci95"] == pytest.approx([low, high]) and c1["in_pooled"] is True
    assert c1["g_start_bp"]["median"] == 1.5 and c1["t_cert_trivial_s"]["level"] == "full"
    assert c1["statement"] == (
        f"На {100 * 30 / 33:.0f} % ({100 * low:.0f}–{100 * high:.0f} %) инстансов ячейки C1 округление непрерывной "
        "релаксации уже в пределах τ от сертифицированного оптимума: качество решателя в выборке там не важно; "
        "H2 и H2b относятся к остальным инстансам")
    assert a3["cells"]["C2"]["share_trivial"] == 0.0 and a3["cells"]["C5"]["n_nonbinding"] == 63


def test_a3_lists_the_pooled_group_of_d1_and_survives_a_cell_with_nothing_scanned():
    dense = elig({"D1_100": {"selected": [2000], "enough": False, "estimate": estimate_of(5, 1)},
                  "D1_200": {"selected": [], "enough": False, "estimate": estimate_of(0, 0)}}, members=[],
                 groups={"D1": {"cells": ["D1_100", "D1_200"], "n_scanned": 6, "n_trivial": 5, "n_nontrivial": 1,
                                "n_undecided": 0, "share_trivial": 5 / 6, "ci95": list(wilson_interval(5, 6, 0.95))}})
    a3 = an.a3_estimates([dense])
    assert a3["groups"]["D1"]["n_trivial"] == 5 and a3["cells"]["D1_100"]["in_pooled"] is False
    assert a3["cells"]["D1_200"]["share_trivial"] is None and a3["cells"]["D1_200"]["statement"] is None


def screening_row(cell, unit, label, *, size=100) -> dict:
    """A row of the stage ``eligibility`` as the screening writes it, reduced to what the eligibility report reads
    (the numbers of ``evidence`` give ``label`` under the rule of §12 A1)."""
    from cif.experiments import e2_eligibility as el

    s, f0, tau = 0.14, -0.05, 1e-3
    thr = tau * s
    f_best, dual = {"trivial": (f0, f0 - 0.5 * thr), "nontrivial": (f0 - 2.0 * thr, f0 - 10.0 * thr)}[label]
    verdict = el.classify(f_start=f0, s_start=s, f_best=f_best, dual_bound=dual, tau=tau, status="duallimit")
    assert verdict.label == label
    ev = {"certificate": "tau", "tau": tau, "f_start": f0, "s_start": s, "f_best": f_best, "dual_bound": dual,
          "status": "duallimit", "gap_lo": verdict.gap_lo, "gap_hi": verdict.gap_hi, "reason": verdict.reason}
    row = {"cell_key": f"{cell}|{unit}", "stage": "eligibility", "cell": cell,
           "instance_id": instance_id(cell, unit, size), "binding": True, "status": "duallimit",
           "eligibility": label, "evidence": ev, "start_f": f0, "start_scale": s,
           "t_cert_s": 1.5 if label == "trivial" else None}
    if isinstance(unit, tuple):
        row.update(family="orlib", port=unit[0], lam_index=unit[1])
    else:
        row.update(family="factor", n=size, seed=unit)
    return row


def test_a_report_built_by_the_screening_code_is_what_the_analysis_reads():
    """The contract between the two modules: the real ``build_report`` output (after a JSON round trip, as the command
    reads it) gives the selected units, the pooled set and the estimate of A3."""
    from cif.experiments import e2_eligibility as el

    labels = ["nontrivial", "trivial", "nontrivial", "trivial", "nontrivial"]  # the scan stops at the 3rd non-trivial
    c1_rows = [screening_row("C1", 2000 + i, lab) for i, lab in enumerate(labels)]
    points = tuple((p, i) for p in (1, 2, 3) for i in range(3))
    c5_rows = [screening_row("C5", (p, i), "nontrivial" if i < 2 else "trivial", size=30 + p) for p, i in points]
    specs = [el.ScanSpec(cell="C1", order=tuple(range(2000, 2400)), target=3, min_nontrivial=2, family="factor",
                         n=100, config_size=100),
             el.ScanSpec(cell="C5", order=points, target=None, min_nontrivial=4, family="orlib", points=True)]
    built = el.build_report([*c1_rows, *c5_rows], specs, role="test")
    report = json.loads(json.dumps(built))
    assert report["summary"]["cells"] == ["C1", "C5"]
    rows = []
    for unit in (2000, 2002, 2004):
        rows += ratio_rows("C1", unit, 2.0, 2.0)
        rows.append(gap_row("C1", unit, "sa", 20000, 0.002))  # a heuristic that knew a better value than any start
    for p, i in points:
        if i < 2:
            rows += ratio_rows("C5", (p, i), 2.0, 2.0, size=30 + p)
            rows.append(gap_row("C5", (p, i), "sa", 20000, 0.002, size=30 + p))
    out = an.h2_report(rows, [report])
    assert out["pooled"]["cells"] == ["C1", "C5"]
    assert out["cells"]["C1"]["n_instances"] == 3 and out["cells"]["C5"]["n_instances"] == 6
    assert out["cells"]["C5"]["n_clusters"] == 3 and out["cells"]["C5"]["ports_without_points"] == []
    assert out["cells"]["C1"]["H2"]["estimate_log"] == pytest.approx(math.log((0.01 * 2 - 0.002) / (0.01 - 0.002)))
    a3 = out["a3"]["cells"]["C1"]
    assert (a3["n_scanned"], a3["n_trivial"], a3["n_nontrivial"]) == (5, 2, 3) and a3["share_trivial"] == 0.4
    assert out["a3"]["cells"]["C5"]["in_pooled"] is True and out["statement"] is None
    # the instances the screening did not select (the trivial ones) have no place in the bench rows
    with pytest.raises(ValueError, match="not selected"):
        an.h2_report([*rows, *ratio_rows("C1", 2001, 2.0, 2.0)], [report])


def test_the_report_records_what_the_paper_needs_to_trace_a_number():
    rows, certs, report = two_cell_world()
    out = an.h2_report(rows, [report], references=certs)
    assert out["kind"] == "e2-h2" and out["plan_version"] == "v2.4" and out["measure"].startswith("PI+")
    assert out["settings"] == {
        "budget_s": 60.0, "n_boot": 9999, "seed": 60000, "level": 0.95, "family_level": 0.99, "family_size": 2,
        "cell_level": pytest.approx(0.995), "band": [1 / 1.5, 1.5], "interval": "percentile", "prespecified": True}
    for key in ("reference", "unit", "c5", "pooled_cells", "bootstrap", "bonferroni", "outcome", "budget_overrun",
                "degenerate", "a3"):
        assert out["prespecified"][key]["choice"] and out["prespecified"][key]["why"]
    json.dumps(out, allow_nan=False)  # strict JSON
    assert len(out["instances"]) == 5 and out["references"]["n_instances"] == 5
    assert out["references"]["by_type"] == {"certified": 5} and out["references"]["n_inconsistent"] == 0
    assert out["budget"]["n_over_budget"]["C1"] == {"exact": 0, "qubo": 0, "ls": 0}
    assert out["cells"]["C1"]["seeds"] == {"exact": 3, "qubo": 10, "ls": 10}
    assert out["cells"]["C1"]["configs"] == {"exact": ["exact_rep-100"], "qubo": ["qubo_rep-100"], "ls": ["ls-100"]}
    # the report does not depend on the order of the rows
    again = an.h2_report(list(reversed(rows)), [report], references=list(reversed(certs)))
    assert again["pooled"] == out["pooled"] and again["cells"] == out["cells"]


def test_a_non_default_bootstrap_is_marked_as_not_prespecified():
    rows, certs, report = two_cell_world()
    out = an.h2_report(rows, [report], references=certs, n_boot=500, seed=1)
    assert out["settings"]["prespecified"] is False and out["settings"]["n_boot"] == 500
    assert any("prespecified" in w for w in out["warnings"])


def test_the_report_is_the_estimate_of_the_stat_module_with_its_constants():
    rows, certs, report = two_cell_world()
    out = an.h2_report(rows, [report], references=certs)
    direct = rq2.estimate_rq2(
        {"C1": {"log_r": [math.log(2.0)] * 3, "log_r_b": [math.log(2.0)] * 3},
         "C2": {"log_r": [math.log(0.25)] * 2, "log_r_b": [math.log(2.0)] * 2}},
        pooled=["C1", "C2"], n_boot=rq2.N_BOOT, seed=rq2.SEED)
    assert out["pooled"]["H2"]["ci_log"] == pytest.approx(
        [direct.pooled["log_r"].interval.low, direct.pooled["log_r"].interval.high], abs=1e-12)
    assert out["pooled"]["H2b"]["se"] == pytest.approx(direct.pooled["log_r_b"].se, abs=1e-12)


# --------------------------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------------------------


def write_run(root: Path, name: str, rows: list[dict], *, mode="ingested", dirty=False, seed_role="test",
              stage="bench") -> Path:
    run = root / name
    run.mkdir()
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {"run_id": name, "git_sha": "a" * 40, "git_dirty": dirty, "config_hash": "c0ffee", "mode": mode,
                "n_errors": 0, "n_missing": 0}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {"stream": "E2", "name": "bench_factor", "params": {"stage": stage, "seed_role": seed_role}}
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return run


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def clean_state(dirty=False):
    return {"available": True, "sha": "b" * 40, "branch": "stream/E0-engine", "dirty": dirty, "status": [],
            "untracked_code": []}


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(an.registry, "_git_state", lambda: clean_state())


def test_the_command_reads_registered_runs_and_writes_the_tables_and_the_json(tmp_path, clean_tree, capsys):
    rows, certs, report = two_cell_world(r1=(2.0,) * 5, r2=(0.25,) * 5)  # five clusters a cell: no C-1 warning
    bench = write_run(tmp_path, "20261020-100000-E2-bench_factor-abc123", rows)
    screening = write_run(tmp_path, "20261019-100000-E2-elig_test-def456", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2"
    code = an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(screening),
                    "--out", str(out)])
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["a3.csv", "cells.csv", "h2.json", "instances.csv", "pooled.csv"]
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["kind"] == "e2-h2" and result["confirmatory"] is True and result["warnings"] == []
    assert result["pooled"]["H2"]["estimate_log"] == pytest.approx(-0.5 * math.log(2.0), abs=1e-12)
    runs = {i["run_id"]: i for i in result["inputs"] if i.get("run_id")}
    assert set(runs) == {bench.name, screening.name}
    assert runs[bench.name]["sha256"] and runs[bench.name]["git_sha"] == "a" * 40
    assert runs[bench.name]["mode"] == "ingested" and runs[bench.name]["seed_role"] == "test"
    assert {i["source"] for i in result["inputs"]} >= {str(bench), str(screening), str(elig_path)}
    assert next(i for i in result["inputs"] if i["source"] == str(elig_path))["sha256"]
    assert result["code"] == {"git_sha": "b" * 40, "git_dirty": False, "branch": "stream/E0-engine"}
    assert set(result["versions"]) == {"python", "numpy", "pandas", "scipy"}
    printed = capsys.readouterr().out
    assert "H2" in printed and "H2b" in printed and str(out) in printed
    cells = pd.read_csv(out / "cells.csv")
    assert {"variant", "cell", "hypothesis", "in_pooled", "estimate_log", "ratio", "ci_low_log", "ci_high_log",
            "ci_level", "outcome"} <= set(cells.columns)
    assert set(cells["variant"]) == {"primary", "full", "first_improvement", "within_budget"}
    instances = pd.read_csv(out / "instances.csv")
    assert len(instances) == 10
    assert {"cell", "unit", "pi_exact", "pi_qubo", "pi_ls", "log_r", "log_r_b"} <= set(instances.columns)
    pooled = pd.read_csv(out / "pooled.csv")
    assert set(pooled["hypothesis"]) == {"H2", "H2b"} and "primary" in set(pooled["variant"])
    assert set(pd.read_csv(out / "a3.csv")["cell"]) == {"C1", "C2"}


def test_plain_result_files_and_a_dirty_tree_make_the_analysis_not_confirmatory(tmp_path, monkeypatch):
    monkeypatch.setattr(an.registry, "_git_state", lambda: clean_state(dirty=True))
    rows, certs, report = two_cell_world()
    bench = tmp_path / "bench.jsonl"
    bench.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(json.dumps(r) + "\n" for r in certs), encoding="utf-8")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2"
    assert an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(ref),
                    "--out", str(out)]) == 0
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["confirmatory"] is False
    text = " ".join(result["warnings"])
    assert "dirty" in text and "registered run" in text
    assert result["code"]["git_dirty"] is True


def test_a_run_from_a_dirty_tree_or_with_calibration_seeds_is_not_confirmatory(tmp_path, clean_tree):
    rows, certs, report = two_cell_world()
    bench = write_run(tmp_path, "r1", rows, dirty=True, seed_role="calibration", mode="smoke")
    ref = write_run(tmp_path, "r2", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2"
    an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(ref), "--out", str(out)])
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    text = " ".join(result["warnings"])
    assert result["confirmatory"] is False
    assert "git_dirty" in text and "seed_role" in text and "mode" in text


def test_a_run_with_missing_cells_or_errors_is_not_confirmatory(tmp_path, clean_tree):
    rows, certs, report = two_cell_world()
    bench = write_run(tmp_path, "bench1", rows)
    manifest = json.loads((bench / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(n_missing=3, n_errors=1)
    write_json(bench / "manifest.json", manifest)
    ref = write_run(tmp_path, "round1", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2"
    an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(ref), "--out", str(out)])
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["confirmatory"] is False
    assert any("1 cells with errors and 3 missing" in w for w in result["warnings"])


def test_a_screening_round_the_report_was_built_from_must_be_among_the_inputs(tmp_path, clean_tree):
    """The certificates of the references come from the rows of the screening rounds. A report built from two rounds
    while the command is given one would leave the instances of the other without a certificate."""
    rows, certs, report = two_cell_world(r1=(2.0,) * 5, r2=(0.25,) * 5)  # five clusters a cell: no C-1 warning
    report["inputs"] = [{"source": "runs/round1", "run_id": "round1", "sha256": "a"},
                        {"source": "runs/round2", "run_id": "round2", "sha256": "b"},
                        {"source": "configs/E2/pilot/n_rule.json", "sha256": "c"}]
    bench = write_run(tmp_path, "bench1", rows)
    round1 = write_run(tmp_path, "round1", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2"
    an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(round1), "--out", str(out)])
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["confirmatory"] is False
    assert any("round2" in w and "eligibility_test.json" in w for w in result["warnings"])
    assert not any("round1" in w for w in result["warnings"])
    round2 = write_run(tmp_path, "round2", [], stage="eligibility")
    out2 = tmp_path / "h2_again"
    an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(round1), str(round2),
             "--out", str(out2)])
    again = json.loads((out2 / "h2.json").read_text(encoding="utf-8"))
    assert again["confirmatory"] is True and again["warnings"] == []


def test_the_command_takes_the_dense_report_the_cold_runs_and_several_run_directories(tmp_path, clean_tree):
    rows, certs, report = two_cell_world()
    dense = elig({"D1_100": {"selected": [2000, 2001], "enough": False, "estimate": estimate_of(5, 2)}}, members=[])
    dense_rows = ratio_rows("D1_100", 2000, 3.0, 1.0) + ratio_rows("D1_100", 2001, 3.0, 1.0)
    cold = [gap_row(cell, unit, role, 20000, 0.0, traj=[(2.0, F_STAR + 0.02)], stage="cold", cold=True)
            for cell, units in (("C1", (2000, 2001, 2002)), ("C2", (2000, 2001))) for unit in units
            for role in ("exact_rep", "qubo_rep", "ls")]
    factor = write_run(tmp_path, "factor", rows)
    dense_run = write_run(tmp_path, "dense", dense_rows)
    cold_run = write_run(tmp_path, "cold", cold, stage="cold")
    ref = write_run(tmp_path, "ref", [*certs, *cert_rows("D1_100", [2000, 2001])], stage="eligibility")
    test_report = write_json(tmp_path / "eligibility_test.json", report)
    dense_report = write_json(tmp_path / "eligibility_dense.json", dense)
    out = tmp_path / "h2"
    assert an.main(["h2", "--rows", str(factor), str(dense_run), "--eligibility", str(test_report),
                    "--eligibility-dense", str(dense_report), "--reference", str(ref), "--cold", str(cold_run),
                    "--out", str(out)]) == 0
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    d1 = result["cells"]["D1_100"]
    assert d1["in_pooled"] is False and d1["H2"]["estimate_log"] == pytest.approx(math.log(3.0))
    assert "D1_100" in result["a3"]["cells"] and "own" in result["sensitivity"]
    assert result["pooled"]["cells"] == ["C1", "C2"] and result["confirmatory"] is True


def test_the_command_stops_at_the_structure_of_the_grid_before_any_ratio(tmp_path, clean_tree):
    """The ls rows of the last instance are missing, and without a reference run the exact method sits on the best
    known value of the first instance (PI+ = 0 there): the missing role is what is reported."""
    rows, _, report = two_cell_world()
    bench = write_run(tmp_path, "factor", rows[:-10])
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    with pytest.raises(ValueError, match="role 'ls'"):
        an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--out", str(tmp_path / "h2")])


def test_the_h2_command_has_a_help_and_no_knob_for_the_statistics(capsys):
    with pytest.raises(SystemExit) as exc:
        an.main(["h2", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for option in ("--rows", "--eligibility", "--eligibility-dense", "--reference", "--cold", "--out"):
        assert option in text
    for knob in ("--seed", "--n-boot", "--level", "--budget"):
        assert knob not in text  # the bootstrap and the budget are prespecified, not options


def test_with_no_selected_instance_the_command_writes_the_share_of_trivial_instances_only(tmp_path, clean_tree,
                                                                                         capsys):
    statement = "Нетривиальных инстансов недостаточно для сводной оценки; RQ2 отвечает долей тривиальных"
    report = elig({"C1": {"selected": [], "enough": False, "estimate": estimate_of(24, 0)}}, members=[],
                  statement=statement)
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2"
    assert an.main(["h2", "--eligibility", str(elig_path), "--out", str(out)]) == 0
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["pooled"] is None and result["statement"] == statement and result["instances"] == []
    assert result["a3"]["cells"]["C1"]["share_trivial"] == 1.0
    assert statement in capsys.readouterr().out
    for name in ("instances.csv", "cells.csv", "pooled.csv"):  # empty tables keep their header
        frame = pd.read_csv(out / name)
        assert len(frame) == 0 and len(frame.columns) > 5
    assert list(pd.read_csv(out / "a3.csv")["cell"]) == ["C1"]


def test_the_prespecified_text_agrees_with_the_constants_it_describes():
    text = " ".join(" ".join(part.values()) for part in an.PRESPECIFIED.values())
    assert f"B = {rq2.N_BOOT}" in text and f"seed {rq2.SEED}" in text and f"[seed, {rq2.STREAM_TAG}, crc32(cell)]" in text
    assert "[1/1.5, 1.5]" in text and "99.75 %" in text and "1e-6" in text
    assert rq2.family_level(4) == pytest.approx(0.9975)


def test_the_constants_of_the_stage_are_those_of_the_plan():
    assert an.BUDGET_S == 60.0 and an.H2_CELLS == ("C1", "C2", "C3", "C4", "C5")
    assert an.REPRESENTATIVES == {"exact": "exact_rep", "qubo": "qubo_rep", "ls": "ls"}
    assert rq2.EQUIVALENCE_FACTOR == 1.5


# --------------------------------------------------------------------------------------------
# Post hoc: another tuned QUBO sampler in the place of the representative (§12, 2026-10-19 (2))
# --------------------------------------------------------------------------------------------


def _world_with_penalty_sa():
    """C1 and C2 where the representative has R = R_b = 2 and the penalty SA of the same runs has R = R_b = 0.5."""
    cells = {"C1": {"selected": [2000, 2001, 2002]}, "C2": {"selected": [2000, 2001]}}
    rows, certs = [], []
    for cell, spec in cells.items():
        for unit in spec["selected"]:
            rows += ratio_rows(cell, unit, 2.0, 2.0)  # exact 0.01, representative 0.02, LS 0.01
            rows += [gap_row(cell, unit, "sa", 20000 + i, 0.005) for i in range(10)]
        certs += cert_rows(cell, spec["selected"])
    return rows, certs, elig(cells)


def test_a_post_hoc_estimate_can_put_another_tuned_qubo_role_in_the_place_of_the_representative():
    rows, certs, report = _world_with_penalty_sa()
    primary = an.h2_report(rows, [report], references=certs)
    post = an.h2_report(rows, [report], references=certs, qubo_role="sa")
    assert primary["pooled"]["H2"]["ratio"] == pytest.approx(2.0)
    assert post["pooled"]["H2"]["ratio"] == pytest.approx(0.5) and post["pooled"]["H2b"]["ratio"] == pytest.approx(0.5)
    assert {i["config_ids"]["qubo"] for i in post["instances"]} == {"sa-100", "sa-200"}
    assert post["settings"]["qubo_role"] == "sa" and "qubo_role" not in primary["settings"]
    assert post["settings"]["prespecified"] is False and primary["settings"]["prespecified"] is True
    assert any("post hoc" in w and "'sa'" in w for w in post["warnings"])
    # the settings themselves are the prespecified ones: no warning says otherwise
    assert not any("are not the prespecified ones" in w for w in post["warnings"])


def test_the_post_hoc_role_is_an_option_of_the_command_and_never_confirmatory(tmp_path, clean_tree):
    rows, certs, report = _world_with_penalty_sa()
    bench = write_run(tmp_path, "20261020-100000-E2-bench_factor-abc123", rows)
    screening = write_run(tmp_path, "20261019-100000-E2-elig_test-def456", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2_sa"
    assert an.main(["h2", "--rows", str(bench), "--eligibility", str(elig_path), "--reference", str(screening),
                    "--qubo-role", "sa", "--out", str(out)]) == 0
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["confirmatory"] is False and result["settings"]["qubo_role"] == "sa"
    assert set(pd.read_csv(out / "instances.csv")["config_qubo"]) == {"sa-100", "sa-200"}


def test_the_post_hoc_role_cannot_be_combined_with_the_cold_rows_of_the_representatives():
    rows, certs, report = _world_with_penalty_sa()
    cold = [gap_row("C1", 2000, "qubo_rep", 20000, 0.02, stage="cold", cold=True)]
    with pytest.raises(ValueError, match="cold"):
        an.h2_report(rows, [report], references=certs, cold_rows=cold, qubo_role="sa")


# --------------------------------------------------------------------------------------------
# Post hoc: another exact configuration in the place of the representative (§12, 2026-10-19, sensitivity to the
# exact baseline)
# --------------------------------------------------------------------------------------------


def _world_with_another_exact_config():
    """C1 and C2 where the representative has R = 2 and an exact configuration of a later run (rows without roles)
    stands at 0.04, so that R = 0.5 against it; R_b = 2 throughout."""
    cells = {"C1": {"selected": [2000, 2001, 2002]}, "C2": {"selected": [2000, 2001]}}
    rows, certs = [], []
    for cell, spec in cells.items():
        for unit in spec["selected"]:
            rows += ratio_rows(cell, unit, 2.0, 2.0)  # exact 0.01, representative 0.02, LS 0.01
            rows += [{**gap_row(cell, unit, "exact_rep", 20000 + i, 0.04, config="scip-aggr"), "roles": []}
                     for i in range(3)]
        certs += cert_rows(cell, spec["selected"])
    return rows, certs, elig(cells)


def test_a_post_hoc_estimate_can_put_another_exact_configuration_in_the_place_of_the_representative():
    rows, certs, report = _world_with_another_exact_config()
    primary = an.h2_report(rows, [report], references=certs)
    post = an.h2_report(rows, [report], references=certs, exact_config="scip-aggr")
    assert primary["pooled"]["H2"]["ratio"] == pytest.approx(2.0)
    assert post["pooled"]["H2"]["ratio"] == pytest.approx(0.5) and post["pooled"]["H2b"]["ratio"] == pytest.approx(2.0)
    assert {i["config_ids"]["exact"] for i in post["instances"]} == {"scip-aggr"}
    assert post["settings"]["exact_config"] == "scip-aggr" and "exact_config" not in primary["settings"]
    assert post["settings"]["prespecified"] is False
    assert any("post hoc" in w and "'scip-aggr'" in w for w in post["warnings"])


def test_an_exact_configuration_missing_on_a_selected_instance_is_an_error():
    rows, certs, report = _world_with_another_exact_config()
    rows = [r for r in rows if not (r["config_id"] == "scip-aggr" and r["cell"] == "C2" and r["seed"] == 2001)]
    with pytest.raises(ValueError, match="scip-aggr"):
        an.h2_report(rows, [report], references=certs, exact_config="scip-aggr")


def test_the_exact_configuration_is_an_option_of_the_command_and_never_confirmatory(tmp_path, clean_tree):
    rows, certs, report = _world_with_another_exact_config()
    bench = write_run(tmp_path, "20261020-100000-E2-bench_factor-abc123", [r for r in rows if r["roles"]])
    later = write_run(tmp_path, "20261021-100000-E2-bench_exact_scip_aggr-abc124", [r for r in rows if not r["roles"]])
    screening = write_run(tmp_path, "20261019-100000-E2-elig_test-def456", certs, stage="eligibility")
    elig_path = write_json(tmp_path / "eligibility_test.json", report)
    out = tmp_path / "h2_exact"
    assert an.main(["h2", "--rows", str(bench), str(later), "--eligibility", str(elig_path), "--reference",
                    str(screening), "--exact-config", "scip-aggr", "--out", str(out)]) == 0
    result = json.loads((out / "h2.json").read_text(encoding="utf-8"))
    assert result["confirmatory"] is False and result["settings"]["exact_config"] == "scip-aggr"
    assert set(pd.read_csv(out / "instances.csv")["config_exact"]) == {"scip-aggr"}
