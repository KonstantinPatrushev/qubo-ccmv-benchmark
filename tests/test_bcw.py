"""Problem (34) of Bertsimas & Cory-Wright (2022) on the OR-Library instances (``cif.instances.bcw``).

The objective is checked against the formula of the paper, written out again here; the threshold ``r̄`` of
Table 4 against two closed forms that share nothing with the QP solver: ``x_max`` is the Euclidean projection
of ``γμ`` onto the simplex (``μᵀx − xᵀx/(2γ) = const − ‖x − γμ‖²/(2γ)``) and ``x_min`` the KKT point
``Σ'⁻¹e / eᵀΣ'⁻¹e`` of the minimum-variance problem when it is positive. The real files are used where the
published numbers can be compared (they are gitignored; those tests are skipped without them).
"""

from __future__ import annotations

import math

import cvxpy as cp
import numpy as np
import pytest

from cif.instances import bcw
from cif.instances.orlib import ORLIB_N, load_orlib
from tests.test_instances import needs_orlib, orlib_text


@pytest.fixture
def port_dir(tmp_path):
    """A synthetic ``port1.txt`` (N = 31 as the real one) in a temporary directory."""
    (tmp_path / "port1.txt").write_text(orlib_text(ORLIB_N[1], seed=5))
    return tmp_path


def _projection_on_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{x ≥ 0, eᵀx = 1}`` (sort-based, exact up to rounding)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / (np.arange(v.size) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


# -- the problem ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", sorted(ORLIB_N.values()))
def test_gamma_is_100_over_the_square_root_of_n(n):
    assert bcw.bcw_gamma(n) == pytest.approx(100.0 / math.sqrt(n), rel=1e-15)


def test_the_ridge_is_folded_into_the_covariance(port_dir):
    inst = bcw.bcw_instance(1, port_dir)
    raw = load_orlib(1, port_dir)
    assert inst.port == 1 and inst.n == 31 and inst.gamma == bcw.bcw_gamma(31)
    np.testing.assert_array_equal(inst.mu, raw.mu)
    np.testing.assert_array_equal(inst.sigma, raw.sigma)
    np.testing.assert_allclose(inst.sigma_ridge, raw.sigma + np.eye(31) / inst.gamma, rtol=0, atol=0)
    assert inst.instance_id == "orlib/port1"


@pytest.mark.parametrize("kappa", [0.0, 1.0])
def test_the_problem_objective_is_the_formula_of_the_paper(port_dir, kappa):
    """(34): ½xᵀΣx + (1/2γ)‖x‖² − κμᵀx, written out here with the raw Σ and no Σ'."""
    inst = bcw.bcw_instance(1, port_dir)
    problem = bcw.bcw_problem(inst, 5, kappa)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.dirichlet(np.ones(31))
        paper = 0.5 * x @ inst.sigma @ x + (0.5 / inst.gamma) * (x @ x) - kappa * (inst.mu @ x)
        assert problem.objective_value(x) == pytest.approx(paper, rel=1e-12, abs=1e-15)
        assert bcw.bcw_objective(inst, x, kappa) == pytest.approx(paper, rel=1e-14, abs=1e-18)


def test_the_problem_has_the_box_budget_and_cardinality_of_the_paper(port_dir):
    inst = bcw.bcw_instance(1, port_dir)
    t3, t4 = bcw.bcw_problem(inst, 10, 1.0), bcw.bcw_problem(inst, 5, 0.0)
    for problem, k in ((t3, 10), (t4, 5)):
        assert (problem.w_min, problem.w_max, problem.budget, problem.risk_aversion) == (0.0, 1.0, 1.0, 1.0)
        assert problem.cardinality == k and problem.n == 31
        np.testing.assert_array_equal(problem.sigma, inst.sigma_ridge)
    np.testing.assert_array_equal(t3.mu, inst.mu)  # κ = 1: the return is in the objective
    assert not t4.mu.any()  # κ = 0: it is not (the constraint carries it)


def test_a_cardinality_outside_the_instance_is_refused(port_dir):
    inst = bcw.bcw_instance(1, port_dir)
    with pytest.raises(ValueError, match="cardinality"):
        bcw.bcw_problem(inst, 0, 1.0)
    with pytest.raises(ValueError, match="cardinality"):
        bcw.bcw_problem(inst, inst.n + 1, 1.0)


def test_an_unknown_port_is_refused(port_dir):
    with pytest.raises(ValueError, match="port"):
        bcw.bcw_instance(6, port_dir)


# -- the threshold r̄ of Table 4 ---------------------------------------------------------------------------


def test_the_threshold_follows_the_rule_of_the_paper_and_the_closed_forms(port_dir):
    inst = bcw.bcw_instance(1, port_dir)
    th = inst.threshold
    y = np.linalg.solve(inst.sigma_ridge, np.ones(inst.n))
    x_min = y / y.sum()
    assert (x_min > 0).all()  # the KKT point is the minimiser only if it is feasible
    x_max = _projection_on_simplex(inst.gamma * inst.mu)
    assert th.r_min == pytest.approx(float(inst.mu @ x_min), rel=1e-8)
    assert th.r_max == pytest.approx(float(inst.mu @ x_max), rel=1e-8)
    assert th.share == 0.3
    assert th.r_bar == pytest.approx(th.r_min + 0.3 * (th.r_max - th.r_min), rel=1e-15, abs=0.0)
    assert th.r_min < th.r_bar < th.r_max
    assert inst.r_bar == th.r_bar


def test_the_share_of_the_threshold_is_an_argument(port_dir):
    low = bcw.bcw_instance(1, port_dir, share=0.0).threshold
    high = bcw.bcw_instance(1, port_dir, share=1.0).threshold
    assert low.r_bar == pytest.approx(low.r_min, rel=1e-15, abs=0.0)
    assert high.r_bar == pytest.approx(high.r_max, rel=1e-15, abs=0.0)
    with pytest.raises(ValueError, match="share"):
        bcw.bcw_instance(1, port_dir, share=1.5)


def test_return_threshold_is_deterministic(port_dir):
    a = bcw.bcw_instance(1, port_dir).threshold
    b = bcw.bcw_instance(1, port_dir).threshold
    assert (a.r_min, a.r_max, a.r_bar) == (b.r_min, b.r_max, b.r_bar)


@needs_orlib
@pytest.mark.parametrize("port", sorted(ORLIB_N))
def test_the_thresholds_of_the_real_ports_match_the_closed_forms(port):
    inst = bcw.bcw_instance(port)
    th = inst.threshold
    y = np.linalg.solve(inst.sigma_ridge, np.ones(inst.n))
    x_min = y / y.sum()
    assert (x_min > 0).all()
    x_max = _projection_on_simplex(inst.gamma * inst.mu)
    assert th.r_min == pytest.approx(float(inst.mu @ x_min), abs=1e-9)
    assert th.r_max == pytest.approx(float(inst.mu @ x_max), abs=1e-9)
    assert inst.n == ORLIB_N[port] and th.r_min < th.r_max


# -- the published number of Figure 1 ---------------------------------------------------------------------


def _soc_bound(inst, k: int, kappa: float, r_bar: float | None) -> float:
    """The Boolean relaxation of the perspective reformulation (35) with ``z ∈ [0, 1]``, ``Σz ≤ k``:
    ``min ½xᵀΣx + (1/2γ)Σθ − κμᵀx`` s.t. ``x_i² ≤ z_iθ_i``, ``eᵀx = 1``, ``x ≥ 0`` [``μᵀx ≥ r̄``]."""
    n = inst.n
    x = cp.Variable(n, nonneg=True)
    theta = cp.Variable(n, nonneg=True)
    z = cp.Variable(n)
    constraints = [cp.sum(x) == 1, z >= 0, z <= 1, cp.sum(z) <= k,
                   cp.SOC(theta + z, cp.vstack([2 * x, theta - z]), axis=0)]
    if r_bar is not None:
        constraints.append(inst.mu @ x >= r_bar)
    objective = (0.5 * cp.quad_form(x, cp.psd_wrap(inst.sigma)) + (0.5 / inst.gamma) * cp.sum(theta)
                 - kappa * (inst.mu @ x))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="CLARABEL")
    assert problem.status == "optimal"
    return float(problem.value)


@needs_orlib
def test_the_soc_bound_of_port2_reproduces_the_number_in_the_text_of_figure_1():
    """BCW, text of Figure 1 (port2, k = 5, minimum return constraint): "the second-order cone lower bound
    is 0.009288 in this instance". It fixes the units of the file, γ = 100/√n and the ridge of the model."""
    inst = bcw.bcw_instance(2)
    assert _soc_bound(inst, 5, 0.0, inst.r_bar) == pytest.approx(0.009288, abs=5e-7)
