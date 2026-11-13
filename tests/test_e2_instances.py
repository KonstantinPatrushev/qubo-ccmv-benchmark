"""Instances of the E2 cells (ANALYSIS_PLAN §3): ``factor`` (C1–C3), ``boot`` (C4), ``orlib`` (C5) and
``dense`` (D1) with their cardinality rules and diagnostics.

Calibration seeds only (instances 1000–1999). No test reads the real S&P panel or the real OR-Library
files: ``boot`` runs on a synthetic ``SP500PIT`` of random walks, ``orlib`` on a synthetic port file.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from cif.data.sp500_pit import SP500PIT
from cif.experiments import e2_instances as inst
from cif.instances.diagnostics import binding_diagnostics, k_from_scont, legacy_dense_instance
from cif.instances.factor import factor_instance
from cif.instances.orlib import ORLIB_N, orlib_lambda_grid
from cif.problem import PortfolioProblem
from tests.test_instances import orlib_text

FIXED = {"lam": 5.0, "fraction": 0.5}


@pytest.fixture(autouse=True)
def _fresh_caches():
    inst.clear_caches()
    yield
    inst.clear_caches()


def _fixed(**over) -> dict:
    return {**FIXED, **over}


# --------------------------------------------------------------------------------------
# factor (C1-C3)
# --------------------------------------------------------------------------------------


def test_factor_instance_follows_the_cell_rules():
    got = inst.build_instance({"family": "factor", "n": 40, "seed": 1000}, _fixed())
    mu, sigma, _ = factor_instance(40, 1000)
    np.testing.assert_array_equal(got.problem.mu, mu)
    np.testing.assert_array_equal(got.problem.sigma, sigma)
    p = got.problem
    assert (p.risk_aversion, p.w_min, p.w_max, p.budget) == (5.0, 0.0, 1.0, 1.0)
    # K = floor(s_cont / 2), the cell rule of C1-C3; binding by construction
    s_cont = binding_diagnostics(PortfolioProblem(mu=mu, sigma=sigma, asset_names=tuple(map(str, range(40))),
                                                  risk_aversion=5.0))["s_cont"]
    assert got.s_cont == s_cont and got.k == p.cardinality == max(1, s_cont // 2)
    assert got.binding is True and got.stats["binding"] is True
    assert got.card_mode == "le" and got.instance_id == "factor/40/1000"


def test_factor_diagnostics_are_recorded_for_every_row():
    got = inst.build_instance({"family": "factor", "n": 30, "seed": 1001}, _fixed())
    for key in ("s_cont", "binding", "cond_sigma", "cv_vol", "cv_mu", "K"):
        assert key in got.stats, key
    assert got.stats["K"] == got.k and got.stats["s_cont"] == got.s_cont
    assert got.meta["family"] == "factor" and got.meta["seed"] == 1001


def test_instances_are_cached_per_process_and_reproducible():
    spec = {"family": "factor", "n": 30, "seed": 1002}
    a = inst.build_instance(spec, _fixed())
    assert inst.build_instance(spec, _fixed()) is a  # cells of one instance share the build
    inst.clear_caches()
    b = inst.build_instance(spec, _fixed())
    assert b is not a and np.array_equal(a.problem.sigma, b.problem.sigma) and a.k == b.k


def test_the_fraction_is_a_parameter_of_the_config():
    full = inst.build_instance({"family": "factor", "n": 30, "seed": 1003}, _fixed(fraction=0.5))
    third = inst.build_instance({"family": "factor", "n": 30, "seed": 1003}, _fixed(fraction=0.25))
    assert third.k == max(1, math.floor(0.25 * full.s_cont + 1e-9)) < full.k


# --------------------------------------------------------------------------------------
# dense (D1)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [20, 30])
def test_dense_instance_is_the_legacy_generator_with_k_a_quarter_of_n(n):
    got = inst.build_instance({"family": "dense", "n": n, "seed": 1004}, _fixed())
    mu, sigma = legacy_dense_instance(n, 1004)
    np.testing.assert_array_equal(got.problem.mu, mu)
    np.testing.assert_array_equal(got.problem.sigma, sigma)
    assert got.k == math.ceil(n / 4) == got.problem.cardinality
    assert (got.problem.risk_aversion, got.problem.w_min, got.problem.w_max) == (5.0, 0.0, 1.0)
    assert got.binding == (got.s_cont > got.k) and got.card_mode == "le"
    assert got.instance_id == f"dense/{n}/1004"


# --------------------------------------------------------------------------------------
# orlib (C5)
# --------------------------------------------------------------------------------------


@pytest.fixture
def orlib_dir(tmp_path):
    """A synthetic ``port1.txt`` (N = 31 as the real one) in a temporary directory."""
    (tmp_path / "port1.txt").write_text(orlib_text(ORLIB_N[1], seed=3))
    return tmp_path


def test_orlib_instance_has_chang_bounds_exactly_k_assets_and_the_corner_lambda(orlib_dir):
    fixed = _fixed(orlib_dir=str(orlib_dir), orlib_k=10, orlib_w_min=0.01, orlib_w_max=1.0, orlib_m=25)
    got = inst.build_instance({"family": "orlib", "port": 1, "lam_index": 6}, fixed)
    p = got.problem
    assert p.n == ORLIB_N[1] and p.cardinality == got.k == 10
    assert (p.w_min, p.w_max) == (0.01, 1.0)
    assert got.card_mode == "eq"  # Σz = K (N11)
    lam = orlib_lambda_grid(1, 25, "corner", data_dir=orlib_dir)[6]
    assert got.meta["lam"] == pytest.approx(lam, rel=1e-12)
    assert p.risk_aversion == pytest.approx(2 * lam / (1 - lam), rel=1e-12)
    assert got.binding == (got.s_cont > 10) and got.stats["K"] == 10
    assert got.instance_id == "orlib/1/06"


def test_relative_data_directories_are_taken_from_the_repository_root(tmp_path, monkeypatch):
    """A job may start in any directory: ``data/raw/orlib`` means the one of the repository."""
    from cif.experiments import registry

    root = tmp_path / "repo"
    (root / "data" / "raw" / "orlib").mkdir(parents=True)
    (root / "data" / "raw" / "orlib" / "port1.txt").write_text(orlib_text(ORLIB_N[1], seed=3))
    monkeypatch.setattr(registry, "REPO_ROOT", root)
    monkeypatch.chdir(tmp_path)  # not the root
    fixed = _fixed(orlib_dir="data/raw/orlib", orlib_k=10, orlib_w_min=0.01, orlib_w_max=1.0, orlib_m=25)
    got = inst.build_instance({"family": "orlib", "port": 1, "lam_index": 3}, fixed)
    assert got.problem.n == ORLIB_N[1]
    absolute = _fixed(orlib_dir=str(root / "data" / "raw" / "orlib"), orlib_k=10, orlib_w_min=0.01,
                      orlib_w_max=1.0, orlib_m=25)
    assert inst.build_instance({"family": "orlib", "port": 1, "lam_index": 3}, absolute).k == got.k


def test_orlib_lambda_index_must_lie_in_the_grid(orlib_dir):
    fixed = _fixed(orlib_dir=str(orlib_dir), orlib_k=10, orlib_w_min=0.01, orlib_w_max=1.0, orlib_m=25)
    with pytest.raises(ValueError, match="lam_index"):
        inst.build_instance({"family": "orlib", "port": 1, "lam_index": 25}, fixed)


def test_orlib_lambda_grid_is_cached_and_increasing(orlib_dir):
    grid = inst.orlib_grid(1, str(orlib_dir), 25)
    assert len(grid) == 25 and all(b > a for a, b in pairwise(grid))
    assert inst.orlib_grid(1, str(orlib_dir), 25) is grid


# --------------------------------------------------------------------------------------
# boot (C4)
# --------------------------------------------------------------------------------------


def _panel(n_tickers: int = 14, seed: int = 0) -> SP500PIT:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", "2025-12-31")
    cols = [f"T{j:02d}" for j in range(n_tickers)]
    common = rng.normal(0.0003, 0.010, size=(len(dates), 1))
    beta = rng.uniform(0.5, 1.5, size=(1, n_tickers))
    rets = common * beta + rng.normal(0.0, 0.012, size=(len(dates), n_tickers))
    prices = pd.DataFrame(50.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=cols)
    volume = pd.DataFrame(np.tile(2.0 ** (n_tickers - np.arange(n_tickers)), (len(dates), 1)),
                          index=dates, columns=cols)
    return SP500PIT(
        membership=pd.DataFrame(True, index=dates, columns=cols),
        prices_adj=prices, close_unadj=prices.copy(), volume=volume, coverage=pd.DataFrame(),
        companies=pd.DataFrame({"ticker": cols, "company_id": cols}), failed={},
    )


def test_boot_instance_uses_the_pilot_lambda_and_the_cell_k_rule(monkeypatch):
    monkeypatch.setattr(inst, "load_panel", lambda path: _panel())
    fixed = _fixed(boot_data_dir="synthetic", boot_n=10, boot_lam=10.0)
    got = inst.build_instance({"family": "boot", "n": 10, "seed": 1100}, fixed)
    assert got.problem.n == 10 and got.problem.risk_aversion == 10.0
    assert (got.problem.w_min, got.problem.w_max) == (0.0, 1.0)  # u = 1, l = 0
    assert got.k == k_from_scont(got.problem, 0.5, s_cont=got.s_cont)
    assert got.meta["family"] == "boot" and got.meta["formation_date"] and "tickers" not in got.meta
    assert got.instance_id == "boot/10/1100"


def test_boot_needs_its_lambda_and_data(monkeypatch):
    monkeypatch.setattr(inst, "load_panel", lambda path: _panel())
    with pytest.raises(ValueError, match="boot_lam"):
        inst.build_instance({"family": "boot", "n": 10, "seed": 1100}, _fixed(boot_data_dir="x", boot_n=10))
    with pytest.raises(ValueError, match="boot_data_dir"):
        inst.build_instance({"family": "boot", "n": 10, "seed": 1100}, _fixed(boot_lam=10.0, boot_n=10))


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError, match="family"):
        inst.build_instance({"family": "fg", "n": 10, "seed": 1000}, _fixed())
