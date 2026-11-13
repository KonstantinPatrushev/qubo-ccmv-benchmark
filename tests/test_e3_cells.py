"""E3 cells (``cif.experiments.e3_walkforward``): one (market, arm, fold) per cell.

ANALYSIS_PLAN §2 RQ3 and §3: the fold's estimate, ``s_cont`` and ``K_f``, the certified optimum, the
k best supports (operator (г)) with their doses, the real solvers (в), the descriptive operators (б)
and (а). Everything on the tiny synthetic market of ``tests/e3_fixtures.py``; the optimum and the
ranks are checked against an exhaustive enumeration of the K-subsets.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import yaml

from cif.backtest.injection import gmv_problem
from cif.classical.refine import refine_problem
from cif.experiments import e3_data as d
from cif.experiments import e3_walkforward as e3
from cif.experiments import run
from cif.heuristics.oracle import enumerate_subsets
from cif.instances.diagnostics import binding_diagnostics
from tests.e3_fixtures import base_config

BP = 1e4


@pytest.fixture(autouse=True)
def _fresh_caches():
    d.clear_caches()
    yield
    d.clear_caches()


def _cell(cfg: dict, **where) -> dict:
    for cell in e3.cells(cfg):
        if all(cell[k] == v for k, v in where.items()):
            return cell
    raise AssertionError(f"no cell with {where}")


def _row(cfg: dict, **where) -> dict:
    cell = _cell(cfg, **where)
    return {**cell, **e3.run_cell(cell, cfg)}


def _problem(cfg: dict, cell: dict):
    """The fold's MV problem exactly as the cell builds it, with K from the rule."""
    p = cfg["params"]
    table = d.get_fold_table(p["market"], p["period"])
    rec = table.folds[cell["fold"]]
    base = rec.problem(lam=cell["lam"], u=table.market.u)
    s_cont = binding_diagnostics(base)["s_cont"]
    k = d.cardinality_rule(s_cont, table.market.u)
    return rec.problem(lam=cell["lam"], u=table.market.u, cardinality=k), k, s_cont


def _brute_ranks(problem, k):
    """All distinct refined portfolios by exhaustive enumeration: ``[(F, support, weights)]``."""
    seen = {}
    for subset in enumerate_subsets(problem.n, k).tolist():
        t = tuple(subset)
        for _ in range(len(t) + 1):  # the fixed point of refine/support
            res = refine_problem(problem, t, "le")
            nxt = tuple(int(i) for i in np.flatnonzero(res.weights > 1e-6))
            if nxt == t:
                break
            t = nxt
        seen.setdefault(t, (res.objective, t, res.weights))
    return sorted(seen.values(), key=lambda x: (x[0], x[1]))


def _weights(row_rank: dict, n: int) -> np.ndarray:
    w = np.zeros(n)
    w[row_rank["support"]] = row_rank["w"]
    return w


MIDDLE = 12  # a fold in the middle of the synthetic period


# --------------------------------------------------------------------------------------------
# The grid and the configuration
# --------------------------------------------------------------------------------------------


def test_cells_cover_arms_and_folds_of_the_period():
    cfg = base_config()
    cells = e3.cells(cfg)
    table = d.get_fold_table(cfg["params"]["market"], cfg["params"]["period"])
    assert len(cells) == 2 * len(table.folds)
    assert {c["arm"] for c in cells} == {"MV", "GMV"}
    assert {c["fold"] for c in cells} == set(range(len(table.folds)))
    assert all(c["lam"] == 5.0 and c["period"] == "confirmatory" for c in cells)
    assert all(c["formation_date"] == str(table.folds[c["fold"]].formation_date.date())
               for c in cells)
    assert len(run.grid(cfg)) == len(cells)  # unique cell keys
    assert cells == e3.cells(cfg)  # stable order


def test_a_pilot_has_a_cell_per_candidate_lambda():
    cfg = base_config(lambda_f=None, lambdas=[5.0, 10.0, 20.0])
    cells = e3.cells(cfg)
    table = d.get_fold_table(cfg["params"]["market"], cfg["params"]["period"])
    assert len(cells) == 3 * 2 * len(table.folds)
    assert {c["lam"] for c in cells} == {5.0, 10.0, 20.0}


def test_the_test_windows_of_a_config_are_known_without_lambda():
    # the confirmatory configs say lambda_f: null until the pilot has chosen it; the length of the
    # confirmation (the pilot's power pilot rescales to it) is a fact of the calendar, not of lambda
    cfg = base_config(lambda_f=None)
    with pytest.raises(ValueError, match="lambda"):
        e3.cells(cfg)
    calendar, grid = e3.calendar_grid(cfg)
    assert grid.n_folds == len(e3.cells(base_config(lambda_f=5.0))) // 2
    assert str(calendar[grid.formation_row(0)].date()) == e3.cells(base_config())[0]["formation_date"]
    with pytest.raises(ValueError, match="period"):
        e3.calendar_grid({"params": {"market": cfg["params"]["market"]}})


def test_us_style_config_with_one_arm_has_mv_cells_only():
    cells = e3.cells(base_config(arms=["MV"]))
    assert {c["arm"] for c in cells} == {"MV"}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"arms": ["MV", "XX"]}, "arm"),
        ({"arms": []}, "arm"),
        ({"n_ranks": 0}, "n_ranks"),
        ({"lambdas": [5.0]}, "lambda"),  # both lambda_f and lambdas
        ({"lambda_f": None}, "lambda"),  # neither
        ({"lambda_f": -1.0}, "lambda"),
        ({"seed_role": "test"}, "solver_seed"),  # 10000 is a calibration seed
        ({"seed_role": "dev"}, "seed_role"),
        ({"optimum": {"time_limit_s": 60.0}}, "optimum"),
    ],
)
def test_cells_validates_the_configuration(change, message):
    with pytest.raises(ValueError, match=message):
        e3.cells(base_config(**change))


def test_declared_seeds_must_be_the_solver_seed():
    cfg = base_config()
    cfg["seeds"] = [10000, 10001]
    with pytest.raises(ValueError, match="seeds"):
        e3.cells(cfg)


def test_the_period_must_fit_the_market():
    cfg = base_config()
    cfg["params"]["period"] = {"name": "x", "start": "1990-01-01", "end": "1990-12-31"}
    with pytest.raises(ValueError, match="history"):
        e3.cells(cfg)


# --------------------------------------------------------------------------------------------
# The MV row: optimum, K, ranks
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mv_row():
    d.clear_caches()
    cfg = base_config()
    return cfg, _row(cfg, arm="MV", fold=MIDDLE)


def test_the_mv_row_holds_the_fold_the_rule_and_the_certificate(mv_row):
    cfg, row = mv_row
    _, k, s_cont = _problem(cfg, row)
    assert row["status"] == "ok"
    assert row["K"] == k and row["s_cont"] == s_cont and row["K_used"] == k
    assert row["binding"] == (s_cont > d.floor_cardinality(0.5)) == d.is_binding(s_cont, 0.5)
    assert row["lower_bound_active"] == d.lower_bound_active(s_cont, 0.5)
    assert row["n_assets"] == 10 and len(row["assets"]) == 10
    assert row["optimum_certified"] is True and row["optimum_status"] in ("optimal", "gaplimit")
    assert row["kbest_all_certified"] is True and row["kbest_n_distinct"] == 4  # R + 1
    assert row["kbest_exhausted"] is False and row["kbest_error"] is None


def test_optimum_and_ranks_agree_with_an_exhaustive_enumeration(mv_row):
    cfg, row = mv_row
    problem, k, _ = _problem(cfg, row)
    brute = _brute_ranks(problem, k)
    assert math.isclose(row["f_star"], brute[0][0], rel_tol=0, abs_tol=1e-9)
    ranks = row["ranks"]
    assert [r["rank"] for r in ranks] == [1, 2, 3, 4]
    for r, (f, support, _) in zip(ranks, brute, strict=False):
        assert tuple(r["support"]) == support
        assert math.isclose(r["objective"], f, abs_tol=1e-9)
        assert math.isclose(r["dose_bp"], (f - brute[0][0]) * BP, abs_tol=1e-5)
    assert ranks[0]["dose_bp"] == 0.0
    doses = [r["dose_bp"] for r in ranks]
    assert doses == sorted(doses)  # the gap does not decrease with the rank


def test_weights_of_every_rank_are_feasible_and_reproduce_their_objective(mv_row):
    cfg, row = mv_row
    problem, _, _ = _problem(cfg, row)
    for r in row["ranks"]:
        w = _weights(r, problem.n)
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-8) and w.min() >= -1e-9
        assert w.max() <= problem.w_max + 1e-8
        assert math.isclose(problem.objective_value(w), r["objective"], abs_tol=1e-9)
        assert set(r["support"]) == set(np.flatnonzero(w > 1e-6))
    assert row["ranks"][0]["support"] == row["optimum_support"]
    # the incumbent of the optimum is kept whole (the fallback of a fold that leaves F_k)
    w = _weights({"support": row["optimum_idx"], "w": row["optimum_w"]}, problem.n)
    assert math.isclose(w.sum(), 1.0, abs_tol=1e-7)
    assert math.isclose(problem.objective_value(w), row["f_star"], abs_tol=1e-7)


def test_the_row_is_json_safe_and_reproducible_up_to_the_times(mv_row):
    cfg, row = mv_row
    again = _row(cfg, arm="MV", fold=MIDDLE)

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if not (k.endswith("_s") or k == "times")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    json.dumps(row, allow_nan=False)  # numpy-free and NaN-free
    assert strip(row) == strip(again)


def test_the_row_records_what_the_assembly_needs_to_check_the_fold(mv_row):
    cfg, row = mv_row
    p = cfg["params"]
    rec = d.get_fold_table(p["market"], p["period"]).folds[MIDDLE]
    check = row["problem_check"]
    assert check["n"] == 10
    assert math.isclose(check["mu_sum"], float(rec.mu.sum()), rel_tol=1e-12)
    assert math.isclose(check["sigma_trace"], float(np.trace(rec.sigma)), rel_tol=1e-12)
    assert row["train_start"] == str(rec.train_start.date())
    assert row["test_start"] == str(rec.test_start.date()) and row["test_end"] == str(rec.test_end.date())
    assert row["assets"] == list(rec.assets) and row["seed"]["solver"] == 10000


# --------------------------------------------------------------------------------------------
# The GMV row
# --------------------------------------------------------------------------------------------


def test_the_gmv_arm_has_the_cardinality_of_the_mv_relaxation_and_a_variance_objective():
    cfg = base_config()
    mv = _row(cfg, arm="MV", fold=MIDDLE)
    gmv = _row(cfg, arm="GMV", fold=MIDDLE)
    assert gmv["status"] == "ok" and gmv["K"] == mv["K"]  # "с той же кардинальностью"
    assert gmv["s_cont"] == mv["s_cont"]  # K_f is the rule of the MV relaxation
    assert gmv["s_cont_arm"] is not None
    problem, k, _ = _problem(cfg, gmv)
    gp = gmv_problem(problem)
    brute = _brute_ranks(gp, k)
    assert math.isclose(gmv["f_star"], brute[0][0], abs_tol=1e-10)
    for r, (f, support, _) in zip(gmv["ranks"], brute, strict=False):
        assert tuple(r["support"]) == support
        # (λ/2)·Δvariance·10⁴ — the gap of the GMV arm (plan, minor 5)
        w = _weights(r, problem.n)
        w0 = _weights(gmv["ranks"][0], problem.n)
        var_gap = 0.5 * 5.0 * (w @ problem.sigma @ w - w0 @ problem.sigma @ w0) * BP
        assert math.isclose(r["dose_bp"], var_gap, abs_tol=1e-5)
        assert math.isclose(r["dose_bp"], (f - brute[0][0]) * BP, abs_tol=1e-5)


# --------------------------------------------------------------------------------------------
# Degenerate folds and failures of the certificate
# --------------------------------------------------------------------------------------------


def test_k_above_the_number_of_assets_is_degenerate_not_an_error():
    cfg = base_config()
    cfg["params"]["market"].update(n_assets=5, u=0.2, seed=1003)  # floor K = 6 > N = 5
    row = _row(cfg, arm="MV", fold=3)
    assert row["K"] == 6 and row["K_used"] == 5 and row["n_assets"] == 5
    assert row["status"] == "degenerate"
    assert row["kbest_n_distinct"] == 1 and row["kbest_exhausted"] is True
    assert row["optimum_certified"] is True


def test_a_box_that_cannot_reach_the_budget_is_flagged_and_not_solved():
    cfg = base_config()
    cfg["params"]["market"].update(n_assets=5, u=0.1, seed=1003)  # N·u = 0.5 < 1
    row = _row(cfg, arm="MV", fold=3)
    assert row["status"] == "infeasible_box" and row["ranks"] == []
    assert row["optimum_certified"] is False


def test_an_uncertified_optimum_skips_the_enumeration(monkeypatch):
    cfg = base_config()
    real = e3.certified_optimum

    def fake(*args, **kwargs):
        ref = real(*args, **kwargs)
        return type(ref)(ref.f_star, ref.scale, "best_known", ref.source, ref.weights, ref.meta)

    monkeypatch.setattr(e3, "certified_optimum", fake)
    called = []
    monkeypatch.setattr(e3, "enumerate_kbest", lambda *a, **k: called.append(1))
    row = _row(cfg, arm="MV", fold=MIDDLE)
    assert row["status"] == "uncertified_optimum" and row["optimum_certified"] is False
    assert not called and row["kbest_all_certified"] is False
    assert row["kbest_n_distinct"] == 0 and row["optimum_support"]  # the incumbent is kept


def test_an_enumeration_that_contradicts_the_certificate_is_recorded(monkeypatch):
    cfg = base_config()

    def boom(*args, **kwargs):
        raise ValueError("w_star is not optimal: a one-swap neighbour of its support is better")

    monkeypatch.setattr(e3, "enumerate_kbest", boom)
    row = _row(cfg, arm="MV", fold=MIDDLE)
    assert row["status"] == "kbest_inconsistent" and "not optimal" in row["kbest_error"]
    assert row["kbest_all_certified"] is False and row["ranks"] == []


def test_other_value_errors_of_the_enumeration_are_not_swallowed(monkeypatch):
    cfg = base_config()

    def boom(*args, **kwargs):
        raise ValueError("n_ranks must be ≥ 1")

    monkeypatch.setattr(e3, "enumerate_kbest", boom)
    with pytest.raises(ValueError, match="n_ranks"):
        _row(cfg, arm="MV", fold=MIDDLE)


def test_a_stopped_enumeration_is_incomplete(monkeypatch):
    cfg = base_config()
    cfg["params"]["kbest"]["time_limit_s"] = 1e-4  # SCIP stops before proving rank 2
    row = _row(cfg, arm="MV", fold=MIDDLE)
    if row["kbest_all_certified"]:  # a tiny instance may still finish within the limit
        pytest.skip("SCIP proved the ranks within the limit")
    assert row["status"] == "kbest_incomplete" and row["kbest_stop_status"]


# --------------------------------------------------------------------------------------------
# The real solvers (в)
# --------------------------------------------------------------------------------------------

SOLVERS = {
    "tuned": True, "common_start": True, "arms": ["MV"],
    "scip": {"formulation": "perspective", "diag": "mineig", "budget_s": 5.0, "mip_gap": 1e-6,
             "scip_params": None},
    "tabu": {"budget_s": 0.3, "num_restarts_b": 2, "num_reads_b": 2, "tenure": 3, "penalty": 10.0,
             "m_b": 20},
    "sa": {"budget_s": 0.3, "num_reads_b": 20, "num_sweeps": 50, "penalty": 10.0,
           "beta": "default", "m_b": 20},
    "ls": {"budget_s": 0.3, "improvement": "first", "num_restarts": None, "rcl_size": 3},
}


def _with_solvers(**over):
    cfg = base_config(solvers=copy.deepcopy(SOLVERS))
    cfg["params"]["solvers"].update(over)
    return cfg


def test_solvers_report_gap_rank_support_and_weights():
    cfg = _with_solvers()
    row = _row(cfg, arm="MV", fold=MIDDLE)
    assert set(row["solvers"]) == {"scip", "tabu", "sa", "ls"}
    assert row["solvers_start"]["time_s"] >= 0 and len(row["solvers_start"]["subset"]) == row["K"]
    problem, _, _ = _problem(cfg, row)
    f_star = row["f_star"]
    for name, s in row["solvers"].items():
        w = _weights({"support": s["idx"], "w": s["w"]}, problem.n)
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-6) and w.min() >= -1e-6
        assert set(s["support"]) == set(np.flatnonzero(w > 1e-6))  # the plan's support
        assert math.isclose(s["objective"], problem.objective_value(w), abs_tol=1e-7)
        assert math.isclose(s["gap_bp"], (s["objective"] - f_star) * BP, abs_tol=1e-6)
        assert s["gap_bp"] >= -1e-3, name  # nobody beats a certified optimum
        assert s["rank"] is not None and 1 <= s["rank"] <= 5, name
        assert s["rank_matched"] in (True, False)
        if s["rank"] <= 4 and s["rank_matched"]:
            assert tuple(s["support_fixed_point"]) == tuple(row["ranks"][s["rank"] - 1]["support"])
        assert s["budget_s"] > 0 and s["wall_s"] >= 0
    # SCIP has 5 s on ten assets: it proves the optimum, a zero gap, rank 1
    assert row["solvers"]["scip"]["rank"] == 1 and abs(row["solvers"]["scip"]["gap_bp"]) < 1e-3


def test_the_scip_params_of_the_solvers_block_reach_scip():
    # E2's exact representative for N = 100 is SCIP without symmetry detection (configs/E3/ff49.yaml carries it as
    # `scip_params`); the row keeps the status and the bounds of the run only, so the call itself is looked at
    cfg = _with_solvers()
    spec = cfg["params"]["solvers"]["scip"]
    problem, k, _ = _problem(cfg, _cell(cfg, arm="MV", fold=MIDDLE))
    assert e3._run_solver("scip", problem, k, spec, None, 1).solver_meta["scip_params"] == {}
    spec["scip_params"] = {"misc/usesymmetry": 0}
    sol = e3._run_solver("scip", problem, k, spec, None, 1)
    assert sol.solver_meta["scip_params"] == {"misc/usesymmetry": 0} and sol.feasible


def test_untuned_solver_parameters_are_refused_in_a_full_run():
    cfg = _with_solvers(tuned=False)
    cfg["mode"] = "full"
    cell = _cell(cfg, arm="MV", fold=MIDDLE)
    with pytest.raises(RuntimeError, match="tuned"):
        e3.run_cell(cell, cfg)
    cfg["mode"] = "smoke"  # a smoke run may use placeholders
    assert e3.run_cell(cell, cfg)["status"] == "ok"


def test_solvers_run_only_on_the_listed_arms():
    cfg = _with_solvers(arms=["MV"])
    assert "solvers" in _row(cfg, arm="MV", fold=MIDDLE)
    assert "solvers" not in _row(cfg, arm="GMV", fold=MIDDLE)


def test_rank_of_a_portfolio_is_found_by_its_support():
    cfg = base_config(n_ranks=3)
    row = _row(cfg, arm="MV", fold=MIDDLE)
    problem, _, _ = _problem(cfg, row)
    ranks = row["ranks"]
    tol = 1e-8
    for r in ranks:
        found = e3.locate_rank(problem, _weights(r, problem.n), ranks, tol)
        assert found["rank"] == r["rank"] and found["matched"] is True
    # a portfolio worse than the last listed rank is beyond the list
    worse = np.zeros(problem.n)
    brute = _brute_ranks(problem, row["K"])
    f_last = ranks[-1]["objective"]
    candidate = next(w for f, s, w in brute if f > f_last + 1e-6)
    worse = candidate
    found = e3.locate_rank(problem, worse, ranks, tol)
    assert found["rank"] == len(ranks) + 1 and found["matched"] is False and found["beyond"] is True


# --------------------------------------------------------------------------------------------
# The descriptive operators (б) and (а)
# --------------------------------------------------------------------------------------------

OPERATORS = {
    "arms": ["MV"],
    # the noisy sample means of the tiny market make the doses large: 500 bp is reachable on fold 12,
    # 5 bp is not (the best neighbour costs 2.5 bp and the chains fail), as the plan expects of (б)
    "swap": {"levels": [5, 500], "n_replicas": 3, "max_steps": 60, "max_restarts": 3},
    "perturb": {"levels": [10, 500], "n_replicas": 3, "max_draws": 2000},
}


def _with_operators(**over):
    ops = copy.deepcopy(OPERATORS)
    ops.update(over)
    return base_config(operators=ops)


def test_swap_operator_replicas_lie_in_the_band_and_are_reproducible():
    cfg = _with_operators(perturb=None)
    row = _row(cfg, arm="MV", fold=MIDDLE)
    problem, _, _ = _problem(cfg, row)
    swap = row["swap"]
    assert [lv["gap_target_bp"] for lv in swap] == [5.0, 500.0]
    assert [lv["status"] for lv in swap] == ["failed", "reached"]
    for lv in swap:
        assert lv["status"] in ("reached", "unreachable", "failed", "timeout")
        assert lv["n_requested"] == 3 and lv["n_evals"] >= 0
        if lv["status"] == "reached":
            assert len(lv["replicas"]) == 3
        for rep in lv["replicas"]:
            g = lv["gap_target_bp"]
            assert g - 1e-9 <= rep["gap_bp"] <= 1.1 * g + 1e-9
            w = _weights(rep, problem.n)
            assert math.isclose(w.sum(), 1.0, abs_tol=1e-8)
            assert math.isclose((problem.objective_value(w) - row["f_star"]) * BP, rep["gap_bp"],
                                abs_tol=1e-5)
    again = _row(cfg, arm="MV", fold=MIDDLE)
    assert [[r["support"] for r in lv["replicas"]] for lv in swap] == \
           [[r["support"] for r in lv["replicas"]] for lv in again["swap"]]


def test_swap_operator_has_no_wall_clock_limit(monkeypatch):
    seen = {}
    import cif.backtest.injection as injection

    real = injection.GapInjector.swap_level

    def spy(self, gap_bp, **kwargs):
        seen.update(kwargs)
        return real(self, gap_bp, **kwargs)

    monkeypatch.setattr(injection.GapInjector, "swap_level", spy)
    _row(_with_operators(perturb=None, swap={"levels": [500], "n_replicas": 2, "max_steps": 30,
                                             "max_restarts": 1}), arm="MV", fold=MIDDLE)
    assert seen["time_limit_s"] is None  # plan v2.4: work is bounded by steps, not by the clock
    assert seen["max_steps"] == 30 and seen["max_restarts"] == 1


def test_perturbation_operator_replicas_are_feasible_and_in_the_band():
    cfg = _with_operators(swap=None)
    row = _row(cfg, arm="MV", fold=MIDDLE)
    problem, _, _ = _problem(cfg, row)
    levels = row["perturb"]
    assert [lv["gap_target_bp"] for lv in levels] == [10.0, 500.0]
    for lv in levels:
        assert lv["status"] in ("reached", "not_built", "failed")
        for rep in lv["replicas"]:
            w = _weights(rep, problem.n)
            assert math.isclose(w.sum(), 1.0, abs_tol=1e-8)
            assert w.min() >= -1e-9 and w.max() <= problem.w_max + 1e-9
            g = lv["gap_target_bp"]
            assert g - 1e-6 <= (problem.objective_value(w) - row["f_star"]) * BP <= 1.1 * g + 1e-6


def test_operators_run_only_on_the_listed_arms():
    cfg = _with_operators()
    assert "swap" in _row(cfg, arm="MV", fold=MIDDLE)
    gmv = _row(cfg, arm="GMV", fold=MIDDLE)
    assert "swap" not in gmv and "perturb" not in gmv


def test_operators_and_solvers_are_skipped_when_the_optimum_is_not_certified(monkeypatch):
    cfg = base_config(solvers=copy.deepcopy(SOLVERS), operators=copy.deepcopy(OPERATORS))
    real = e3.certified_optimum

    def fake(*args, **kwargs):
        ref = real(*args, **kwargs)
        return type(ref)(ref.f_star, ref.scale, "best_known", ref.source, ref.weights, ref.meta)

    monkeypatch.setattr(e3, "certified_optimum", fake)
    row = _row(cfg, arm="MV", fold=MIDDLE)
    assert row["status"] == "uncertified_optimum"
    assert not {"solvers", "swap", "perturb"} & set(row)  # every gap is measured against f*


def test_operator_seeds_depend_on_the_arm_and_the_fold():
    assert e3.operator_seed(10000, "MV", 5.0) != e3.operator_seed(10000, "GMV", 5.0)
    assert e3.operator_seed(10000, "MV", 5.0) != e3.operator_seed(10001, "MV", 5.0)
    assert e3.operator_seed(10000, "MV", 5.0) == e3.operator_seed(10000, "MV", 5.0)
    assert 0 <= e3.operator_seed(10000, "MV", 5.0) < 2**31


# --------------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------------


def test_runner_executes_a_registered_synthetic_run(tmp_path):
    cfg = base_config(arms=["MV"])
    cfg["params"]["period"] = {"name": "confirmatory", "start": "2000-06-01", "end": "2000-07-15"}
    path = tmp_path / "e3.yaml"
    path.write_text(yaml.safe_dump(cfg))
    runs = tmp_path / "runs"
    assert run.main(["--config", str(path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert rows and not any(r.get("error") for r in rows)
    assert {r["status"] for r in rows} <= {"ok", "degenerate"}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == [10000] and manifest["n_errors"] == 0


def test_the_cell_of_a_changed_calendar_is_refused():
    cfg = base_config()
    cell = dict(_cell(cfg, arm="MV", fold=MIDDLE))
    cell["formation_date"] = "1999-01-01"
    with pytest.raises(RuntimeError, match="calendar"):
        e3.run_cell(cell, cfg)


# --------------------------------------------------------------------------------------------
# The run server: ``cells[i::S]`` shards (``cif.experiments.run --shard``, the launcher)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_shards", [2, 3, 4, 6, 8])
def test_interleaved_shards_mix_the_arms_and_spread_the_folds(n_shards):
    """``cells[i::S]`` must not hand one arm to a whole shard (the MV arm carries the solvers and the
    operators, GMV does not) or neighbouring folds to one shard: with the arm as the outer loop every
    shard of a grid larger than ``S`` holds both arms and folds ``S`` apart."""
    cfg = base_config(lambda_f=None, lambdas=[5.0, 10.0])
    cells = e3.cells(cfg)
    assert len(cells) > 4 * n_shards
    for i in range(n_shards):
        shard = cells[i::n_shards]
        assert {c["arm"] for c in shard} == {"MV", "GMV"}, i
        assert {c["lam"] for c in shard} == {5.0, 10.0}, i
    folds = [c["fold"] for c in cells[0::n_shards] if c["arm"] == "MV" and c["lam"] == 5.0]
    assert len(set(np.diff(folds))) <= 2  # a regular stride, not a block of neighbours


def test_every_cell_is_in_exactly_one_shard_and_the_grid_does_not_depend_on_the_process():
    cfg = base_config()
    cells = e3.cells(cfg)
    for n_shards in (3, 7):
        keys = [run.cell_key(c) for i in range(n_shards) for c in cells[i::n_shards]]
        assert sorted(keys) == sorted(run.cell_key(c) for c in cells)
        assert len(set(keys)) == len(cells)
    d.clear_caches()  # a fresh process builds the same grid
    assert e3.cells(cfg) == cells


def _sharded_config(tmp_path):
    cfg = base_config(operators=copy.deepcopy(OPERATORS))
    cfg["params"]["period"] = {"name": "confirmatory", "start": "2000-06-01", "end": "2000-07-15"}
    path = tmp_path / "e3.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return cfg, path


def test_shards_written_by_the_runner_are_ingested_into_one_registered_run(tmp_path):
    from cif.experiments import ingest

    cfg, path = _sharded_config(tmp_path)
    n_shards = 3
    out = tmp_path / "incoming"
    for i in range(n_shards):
        assert run.main(["--config", str(path), "--shard", str(i), "--n-shards", str(n_shards),
                         "--workers", "1", "--out", str(out / f"shard{i}.jsonl")]) == 0
    shards = sorted(out.glob("shard*.jsonl"))
    runs = tmp_path / "runs"
    assert ingest.main(["--config", str(path), "--shards", *map(str, shards),
                        "--runs-dir", str(runs)]) == 0
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    cells = e3.cells(cfg)
    assert [r["cell_key"] for r in rows] == [run.cell_key(c) for c in cells]  # in the order of the grid
    assert not any(r.get("error") for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "ingested" and manifest["n_cells"] == len(cells)
    # a shard run is the same computation as a local one
    local = tmp_path / "local"
    assert run.main(["--config", str(path), "--runs-dir", str(local), "--allow-dirty"]) == 0
    (local_dir,) = [p for p in local.iterdir() if p.is_dir()]
    local_rows = {r["cell_key"]: r for r in map(json.loads, (local_dir / "results.jsonl").read_text().splitlines())}
    for row in rows:
        assert ingest._result(row) == ingest._result(local_rows[row["cell_key"]])


def test_a_cell_computed_twice_is_a_duplicate_and_not_a_conflict(tmp_path):
    """The certified optimum, the ranks and the operators are deterministic: their repeated row differs
    in the times only, and the times (top-level keys ending in ``_s`` or starting with ``time``: the row's
    ``times``) do not count in the ingest's comparison of two successful rows of one cell. A shard that is run again
    elsewhere must not stop the merge."""
    from cif.experiments import ingest

    cfg, _ = _sharded_config(tmp_path)
    cell = _cell(cfg, arm="MV", fold=MIDDLE - 8)  # the last fold of the shortened period
    one = {"cell_key": run.cell_key(cell), **cell, **e3.run_cell(cell, cfg)}
    two = {"cell_key": run.cell_key(cell), **cell, **e3.run_cell(cell, cfg)}
    assert one["status"] == "ok" and one["swap"]  # the operators ran: their rows are in the comparison
    assert one["times"] != two["times"]
    best, n_duplicates, conflicts = ingest.merge_rows([one, two])
    assert n_duplicates == 1 and conflicts == [] and len(best) == 1
    assert ingest._result(one) == ingest._result(two)
    # no clock is left in the part of the row that the ingest compares, at any depth
    assert list(_clock_keys(json.loads(ingest._result(one)))) == []


def _clock_keys(value, path=""):
    """Paths of the keys that hold a time (``*_s``, ``time*``, ``wall``, ``elapsed``) at any depth."""
    if isinstance(value, dict):
        for key, inner in value.items():
            if key.endswith("_s") or key.startswith(("time", "wall", "elapsed")):
                yield f"{path}/{key}"
            yield from _clock_keys(inner, f"{path}/{key}")
    elif isinstance(value, list):
        for i, inner in enumerate(value):
            yield from _clock_keys(inner, f"{path}[{i}]")
