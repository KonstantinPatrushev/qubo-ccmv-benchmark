"""Tests for ``cif.instances``: OR-Library loader, factor generator, binding diagnostics (WP-E3).

Network tests run only with ``CIF_NETWORK_TESTS=1``; tests on downloaded OR-Library files are
skipped when ``data/raw/orlib`` is empty (the directory is gitignored).
"""

from __future__ import annotations
import __future__

import ast
import hashlib
import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from cif.classical.brute_force import brute_force_cardinality_continuous
from cif.classical.continuous import minimum_variance_portfolio, solve_continuous_mvo
from cif.data.bootstrap_synthetic import bootstrap_realistic_instance, load_real_covariance
from cif.instances import orlib
from cif.instances.diagnostics import (
    binding_diagnostics,
    continuous_relaxation,
    describe_legacy_dense,
    instance_statistics,
    k_from_scont,
    legacy_dense_instance,
)
from cif.instances.factor import factor_instance
from cif.problem import PortfolioProblem

REPO_ROOT = Path(__file__).resolve().parents[1]
ORLIB_DIR = REPO_ROOT / "data" / "raw" / "orlib"

needs_network = pytest.mark.skipif(
    os.environ.get("CIF_NETWORK_TESTS") != "1",
    reason="network test: set CIF_NETWORK_TESTS=1 to run",
)
needs_orlib = pytest.mark.skipif(
    not all((ORLIB_DIR / name).exists() for name in orlib.orlib_file_names()),
    reason="OR-Library files are not downloaded (cif.instances.orlib.download_orlib)",
)


# --------------------------------------------------------------------------------------
# Helpers: synthetic text in the OR-Library layout
# --------------------------------------------------------------------------------------

SMALL_PORT = """\
 3
 .010000 .040000
 .020000 .050000
 .030000 .060000
 1 1 1.000000
 1 2 .500000
 1 3 .250000
 2 2 1.000000
 2 3 -.100000
 3 3 1.000000
"""


def orlib_text(n: int, seed: int = 0) -> str:
    """A valid port file of size ``n`` (random PSD correlation, 6 decimals like the originals)."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.0, 0.01, size=n)
    std = rng.uniform(0.03, 0.06, size=n)
    a = rng.standard_normal((n, n + 5))
    cov = a @ a.T
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    lines = [f" {n}"]
    lines += [f" {m:.6f} {s:.6f}" for m, s in zip(mean, std, strict=True)]
    lines += [f" {i + 1} {j + 1} {corr[i, j]:.6f}" for i in range(n) for j in range(i, n)]
    return "\n".join(lines) + "\n"


def frontier_text(m: int = 40) -> str:
    """A port-ef file: ``m`` rows ``mean variance``, descending in return."""
    r = np.linspace(0.0108, 0.0027, m)
    v = 6.4e-4 + 63.0 * (r - 0.0027) ** 2
    return "".join(f" {ri:.10f} {vi:.10f}\n" for ri, vi in zip(r, v, strict=True))


def fake_fetch_factory(files: dict[str, str]):
    """A ``fetch`` stub serving ``files`` (keyed by file name) and counting calls."""
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return files[url.rsplit("/", 1)[-1]].encode("ascii")

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def random_moments(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.002, 0.012, size=n)
    a = rng.standard_normal((n, n + 3))
    sigma = a @ a.T * 4e-4
    return mu, sigma


# --------------------------------------------------------------------------------------
# OR-Library: format parser
# --------------------------------------------------------------------------------------


def test_parse_small_fixture_matches_hand_computation():
    inst = orlib.parse_orlib_port(SMALL_PORT, name="tiny")
    assert inst.n == 3
    assert inst.name == "tiny"
    np.testing.assert_allclose(inst.mu, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(inst.std, [0.04, 0.05, 0.06])
    # 1-based (i, j) -> 0-based, symmetric, unit diagonal; the three off-diagonals differ.
    assert inst.corr[0, 1] == inst.corr[1, 0] == pytest.approx(0.5)
    assert inst.corr[0, 2] == inst.corr[2, 0] == pytest.approx(0.25)
    assert inst.corr[1, 2] == inst.corr[2, 1] == pytest.approx(-0.1)
    np.testing.assert_allclose(np.diag(inst.corr), 1.0)
    expected = np.array(
        [
            [0.0016, 0.5 * 0.04 * 0.05, 0.25 * 0.04 * 0.06],
            [0.5 * 0.04 * 0.05, 0.0025, -0.1 * 0.05 * 0.06],
            [0.25 * 0.04 * 0.06, -0.1 * 0.05 * 0.06, 0.0036],
        ]
    )
    np.testing.assert_allclose(inst.sigma, expected, atol=1e-15)
    np.testing.assert_allclose(inst.sigma, inst.sigma.T, atol=0.0)


def test_parse_tolerates_missing_diagonal_and_either_triangle():
    lines = SMALL_PORT.splitlines()
    head = lines[:4]
    # Diagonal lines dropped; (1,3) given as (3,1); line breaks and spacing are irrelevant.
    body = [" 1 2 .500000", " 3 1 .250000", " 2 3 -.100000"]
    text = "\n".join(head + body) + "\n"
    inst = orlib.parse_orlib_port(text.replace("\n", "  \r\n "))
    ref = orlib.parse_orlib_port(SMALL_PORT)
    np.testing.assert_allclose(inst.sigma, ref.sigma)


def test_parse_accepts_both_orders_when_consistent():
    text = SMALL_PORT + " 2 1 .500000\n"
    inst = orlib.parse_orlib_port(text)
    assert inst.corr[0, 1] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t: t.replace(" 2 3 -.100000\n", ""), "missing"),
        (lambda t: t + " 1 2 .900000\n", "conflict|duplicate"),
        (lambda t: t.replace(" 3 3 1.000000", " 4 3 1.000000"), "index"),
        (lambda t: t.replace(" 3 3 1.000000", " 3 3 1.500000"), "correlation"),
        (lambda t: t.replace(" .020000 .050000", " .020000 -.050000"), "std"),
        (lambda t: t.replace(" 3\n", " 4\n", 1), "truncated|missing|tokens"),
        (lambda t: "<html>Not Found</html>", "integer|parse|asset"),
        (lambda t: "", "empty"),
    ],
)
def test_parse_rejects_malformed_input(mutate, match):
    with pytest.raises(ValueError, match=match):
        orlib.parse_orlib_port(mutate(SMALL_PORT))


def test_parse_rejects_non_psd_correlation():
    text = SMALL_PORT.replace(".500000", ".900000").replace(".250000", ".900000")
    text = text.replace("-.100000", "-.900000")
    with pytest.raises(ValueError, match="PSD"):
        orlib.parse_orlib_port(text)


def test_parse_frontier_returns_mean_and_variance_columns():
    mean, var = orlib.parse_orlib_frontier(frontier_text(25))
    assert mean.shape == var.shape == (25,)
    assert mean[0] > mean[-1]
    assert np.all(var > 0)
    with pytest.raises(ValueError, match=r"two columns|even"):
        orlib.parse_orlib_frontier(" 0.1 0.2 0.3\n")


# --------------------------------------------------------------------------------------
# OR-Library: download + provenance (offline, with an injected fetcher)
# --------------------------------------------------------------------------------------


def test_orlib_urls_follow_the_portinfo_page():
    assert orlib.ORLIB_BASE_URL == "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
    assert orlib.orlib_url("port3.txt") == f"{orlib.ORLIB_BASE_URL}/port3.txt"
    assert orlib.ORLIB_N == {1: 31, 2: 85, 3: 89, 4: 98, 5: 225}


def test_download_writes_files_and_provenance_and_is_idempotent(tmp_path):
    files = {"port1.txt": orlib_text(31, seed=1), "portef1.txt": frontier_text(40)}
    fetch = fake_fetch_factory(files)

    prov = orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256={})
    assert sorted(fetch.calls) == [orlib.orlib_url("port1.txt"), orlib.orlib_url("portef1.txt")]
    for name, text in files.items():
        assert (tmp_path / name).read_text() == text
        entry = prov["files"][name]
        assert entry["url"] == orlib.orlib_url(name)
        assert entry["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert entry["bytes"] == len(text.encode())
        assert entry["downloaded_at"].endswith("Z")

    on_disk = json.loads((tmp_path / "provenance.json").read_text())
    assert on_disk["files"] == prov["files"]
    assert on_disk["info_url"] == orlib.ORLIB_INFO_URL
    assert "Chang" in on_disk["reference"]

    # Second call: verified files are not fetched again.
    fetch.calls.clear()
    orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256={})
    assert fetch.calls == []

    # A corrupted local file is re-downloaded; force=True refetches everything.
    (tmp_path / "port1.txt").write_text("corrupted")
    orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256={})
    assert fetch.calls == [orlib.orlib_url("port1.txt")]
    fetch.calls.clear()
    orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256={}, force=True)
    assert len(fetch.calls) == 2


def test_download_refuses_non_data_response_and_writes_nothing(tmp_path):
    fetch = fake_fetch_factory(
        {"port1.txt": "<html><body>300 Multiple Choices</body></html>", "portef1.txt": ""}
    )
    with pytest.raises(ValueError):
        orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256={})
    assert not (tmp_path / "port1.txt").exists()
    assert not (tmp_path / "provenance.json").exists()


def test_download_checks_expected_size_of_each_port(tmp_path):
    fetch = fake_fetch_factory({"port1.txt": orlib_text(30, seed=2), "portef1.txt": frontier_text()})
    with pytest.raises(ValueError, match="31"):
        orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256={})


def test_download_rejects_unknown_port(tmp_path):
    with pytest.raises(ValueError, match="port"):
        orlib.download_orlib(tmp_path, ports=(6,), fetch=fake_fetch_factory({}), expected_sha256={})


def test_download_checks_pinned_hash(tmp_path):
    files = {"port1.txt": orlib_text(31, seed=1), "portef1.txt": frontier_text(40)}
    fetch = fake_fetch_factory(files)
    pinned = {"port1.txt": "0" * 64}
    with pytest.raises(ValueError, match="SHA-256"):
        orlib.download_orlib(tmp_path, ports=(1,), fetch=fetch, expected_sha256=pinned)
    assert not (tmp_path / "port1.txt").exists()


def test_pinned_hashes_cover_all_ten_files():
    expected = {f"port{k}.txt" for k in range(1, 6)} | {f"portef{k}.txt" for k in range(1, 6)}
    assert set(orlib.ORLIB_SHA256) == expected
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in orlib.ORLIB_SHA256.values())


def test_verify_orlib_flags_modified_and_missing_files(tmp_path):
    files = {"port1.txt": orlib_text(31, seed=1), "portef1.txt": frontier_text(40)}
    orlib.download_orlib(tmp_path, ports=(1,), fetch=fake_fetch_factory(files), expected_sha256={})
    assert orlib.verify_orlib(tmp_path) == {"port1.txt": True, "portef1.txt": True}
    (tmp_path / "portef1.txt").write_text(frontier_text(41))
    assert orlib.verify_orlib(tmp_path) == {"port1.txt": True, "portef1.txt": False}
    (tmp_path / "port1.txt").unlink()
    assert orlib.verify_orlib(tmp_path)["port1.txt"] is False


@needs_network
@pytest.mark.network
def test_download_port1_from_brunel(tmp_path):  # pragma: no cover - network
    prov = orlib.download_orlib(tmp_path, ports=(1,))
    assert prov["files"]["port1.txt"]["sha256"] == orlib.ORLIB_SHA256["port1.txt"]
    assert orlib.load_orlib(1, data_dir=tmp_path).n == 31


# --------------------------------------------------------------------------------------
# Chang et al. (2000) parameterisation  <->  project risk aversion
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lam", "ra"), [(0.0, 0.0), (0.2, 0.5), (0.5, 2.0), (0.8, 8.0), (0.9, 18.0)]
)
def test_risk_aversion_from_lambda_values(lam, ra):
    assert orlib.risk_aversion_from_lambda(lam) == pytest.approx(ra)
    assert orlib.lambda_from_risk_aversion(ra) == pytest.approx(lam)


def test_risk_aversion_mapping_edges_and_validation():
    assert orlib.risk_aversion_from_lambda(1.0) == float("inf")
    assert orlib.lambda_from_risk_aversion(float("inf")) == 1.0
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            orlib.risk_aversion_from_lambda(bad)
    with pytest.raises(ValueError):
        orlib.lambda_from_risk_aversion(-1.0)


@pytest.mark.parametrize("lam", [0.05, 0.3, 0.5, 0.9, 0.99])
def test_chang_objective_is_one_minus_lambda_times_project_objective(lam):
    """min λ wᵀΣw - (1-λ) μᵀw  ==  (1-λ) · [-μᵀw + (ra/2) wᵀΣw],  ra = 2λ/(1-λ)."""
    mu, sigma = random_moments(7, seed=11)
    problem = PortfolioProblem(
        mu=mu,
        sigma=sigma,
        asset_names=tuple(f"A{i}" for i in range(7)),
        risk_aversion=orlib.risk_aversion_from_lambda(lam),
    )
    rng = np.random.default_rng(5)
    for _ in range(20):
        w = rng.dirichlet(np.ones(7))
        chang = orlib.chang_objective(mu, sigma, lam, w)
        assert chang == pytest.approx((1.0 - lam) * problem.objective_value(w), rel=1e-12)


@pytest.mark.parametrize("lam", [0.1, 0.5, 0.9])
def test_chang_and_project_formulations_have_the_same_minimiser(lam):
    n = 8
    mu, sigma = random_moments(n, seed=3)
    w = cp.Variable(n)
    chang = cp.Problem(
        cp.Minimize(lam * cp.quad_form(w, cp.psd_wrap(sigma)) - (1.0 - lam) * mu @ w),
        [cp.sum(w) == 1, w >= 0, w <= 1],
    )
    chang.solve(solver="CLARABEL")
    ours = solve_continuous_mvo(
        PortfolioProblem(
            mu=mu,
            sigma=sigma,
            asset_names=tuple(f"A{i}" for i in range(n)),
            risk_aversion=orlib.risk_aversion_from_lambda(lam),
        )
    )
    np.testing.assert_allclose(ours.weights, w.value, atol=1e-5)
    assert chang.value == pytest.approx((1.0 - lam) * ours.objective, rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------------------
# orlib_problem (built on a synthetic port1.txt in a temp dir)
# --------------------------------------------------------------------------------------


@pytest.fixture
def fake_orlib_dir(tmp_path):
    (tmp_path / "port1.txt").write_text(orlib_text(31, seed=7))
    return tmp_path


def test_orlib_problem_fields_follow_chang_defaults(fake_orlib_dir):
    inst = orlib.load_orlib(1, data_dir=fake_orlib_dir)
    p = orlib.orlib_problem(1, K=10, lam=0.75, data_dir=fake_orlib_dir)
    assert p.n == 31 and inst.n == 31
    assert p.cardinality == 10
    assert (p.w_min, p.w_max, p.budget) == (0.01, 1.0, 1.0)
    assert p.risk_aversion == pytest.approx(6.0)  # 2 * 0.75 / 0.25
    np.testing.assert_allclose(p.mu, inst.mu)
    np.testing.assert_allclose(p.sigma, inst.sigma)
    assert len(p.asset_names) == 31 and len(set(p.asset_names)) == 31
    # Same minimiser as Chang's objective: chang(w) = (1 - lam) * project(w).
    w = np.random.default_rng(0).dirichlet(np.ones(31))
    assert orlib.chang_objective(inst.mu, inst.sigma, 0.75, w) == pytest.approx(
        0.25 * p.objective_value(w), rel=1e-12
    )


def test_orlib_problem_lambda_one_is_minimum_variance(fake_orlib_dir):
    inst = orlib.load_orlib(1, data_dir=fake_orlib_dir)
    p = orlib.orlib_problem(1, K=5, lam=1.0, w_min=0.0, data_dir=fake_orlib_dir)
    w = np.random.default_rng(1).dirichlet(np.ones(31))
    # Objective is exactly the variance (Chang's objective at lambda = 1).
    assert p.objective_value(w) == pytest.approx(float(w @ inst.sigma @ w), rel=1e-12)
    assert p.objective_value(w) == pytest.approx(
        orlib.chang_objective(inst.mu, inst.sigma, 1.0, w), rel=1e-12
    )
    # The λ = 1 problem (μ = 0, ra = 2) has the minimum-variance minimiser. The reference is solved
    # independently on the normalised problem: at the raw scale of the files the optimal variance
    # is ~1e-5 and the solvers' absolute tolerance (1e-8) alone costs 1e-3 in relative terms.
    scale = float(np.mean(np.diag(inst.sigma)))
    w_ref = cp.Variable(inst.n)
    cp.Problem(
        cp.Minimize(cp.quad_form(w_ref, cp.psd_wrap(inst.sigma / scale))),
        [cp.sum(w_ref) == 1, w_ref >= 0, w_ref <= 1],
    ).solve(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    w_ours = continuous_relaxation(p).weights
    assert float(w_ours @ inst.sigma @ w_ours) == pytest.approx(
        float(w_ref.value @ inst.sigma @ w_ref.value), rel=1e-6
    )
    # The generic minimum-variance helper agrees to its (absolute) accuracy.
    w_mv = minimum_variance_portfolio(p).weights
    assert float(w_mv @ inst.sigma @ w_mv) == pytest.approx(
        float(w_ref.value @ inst.sigma @ w_ref.value), abs=1e-7
    )


def test_orlib_problem_lambda_zero_puts_everything_on_the_best_asset(fake_orlib_dir):
    inst = orlib.load_orlib(1, data_dir=fake_orlib_dir)
    p = orlib.orlib_problem(1, K=5, lam=0.0, w_min=0.0, data_dir=fake_orlib_dir)
    assert p.risk_aversion == 0.0
    w = solve_continuous_mvo(p).weights
    assert int(np.argmax(w)) == int(np.argmax(inst.mu))
    assert w.max() == pytest.approx(1.0, abs=1e-4)


def test_orlib_problem_validates_arguments(fake_orlib_dir):
    with pytest.raises(ValueError, match="port"):
        orlib.orlib_problem(6, K=5, lam=0.5, data_dir=fake_orlib_dir)
    with pytest.raises(ValueError, match="lam"):
        orlib.orlib_problem(1, K=5, lam=1.5, data_dir=fake_orlib_dir)
    with pytest.raises(ValueError):
        orlib.orlib_problem(1, K=32, lam=0.5, data_dir=fake_orlib_dir)
    with pytest.raises(FileNotFoundError, match="download_orlib"):
        orlib.orlib_problem(2, K=5, lam=0.5, data_dir=fake_orlib_dir)
    assert orlib.orlib_problem(1, K=None, lam=0.5, data_dir=fake_orlib_dir).cardinality is None


# --------------------------------------------------------------------------------------
# λ grid: anchored at the "corner" where diversification starts
# --------------------------------------------------------------------------------------


def scaled_support(mu: np.ndarray, sigma: np.ndarray, ra: float, tol: float = 1e-5):
    """Long-only continuous optimum on the normalised problem and its support size."""
    scale = float(np.mean(np.diag(sigma)))
    problem = PortfolioProblem(
        mu=mu / scale,
        sigma=sigma / scale,
        asset_names=tuple(f"A{i}" for i in range(len(mu))),
        risk_aversion=ra,
    )
    w = solve_continuous_mvo(problem).weights
    return int((w > tol).sum()), w


def test_corner_risk_aversion_separates_single_asset_from_diversified():
    """Below ra_c the whole budget sits in the best-mean asset; above it a second asset enters."""
    mu, sigma = random_moments(10, seed=21)
    ra_c = orlib.corner_risk_aversion(mu, sigma)
    assert 0.0 < ra_c < float("inf")
    s_below, w_below = scaled_support(mu, sigma, 0.95 * ra_c)
    s_above, _ = scaled_support(mu, sigma, 1.05 * ra_c)
    assert s_below == 1
    assert int(np.argmax(w_below)) == int(np.argmax(mu))
    assert s_above >= 2


def test_corner_risk_aversion_degenerate_cases():
    # The best-mean asset also has the smallest variance contribution: optimal for every ra.
    mu = np.array([0.02, 0.01])
    sigma = np.array([[0.04, 0.04], [0.04, 0.09]])
    assert orlib.corner_risk_aversion(mu, sigma) == float("inf")
    # A tie for the best mean: diversification starts at once (ra_c = 0), no usable grid.
    mu_tie = np.array([0.02, 0.02])
    sigma_diag = np.diag([0.04, 0.09])
    assert orlib.corner_risk_aversion(mu_tie, sigma_diag) == 0.0
    with pytest.raises(ValueError, match="corner"):
        orlib.lambda_grid(mu_tie, sigma_diag)
    with pytest.raises(ValueError, match="corner"):
        orlib.lambda_grid(mu, sigma)


def test_lambda_grid_is_geometric_in_risk_aversion_from_the_corner():
    mu, sigma = random_moments(10, seed=21)
    lam = orlib.lambda_grid(mu, sigma)
    assert lam.shape == (25,)
    assert np.all(np.diff(lam) > 0) and lam[0] > 0.0 and lam[-1] == pytest.approx(0.999)
    ra = np.array([orlib.risk_aversion_from_lambda(x) for x in lam])
    assert ra[0] == pytest.approx(1.1 * orlib.corner_risk_aversion(mu, sigma))
    np.testing.assert_allclose(ra[1:] / ra[:-1], (ra[1] / ra[0]) * np.ones(24), rtol=1e-9)
    # uniform in log-odds of lambda  <=>  geometric in ra
    logit = np.log(lam / (1.0 - lam))
    np.testing.assert_allclose(np.diff(logit), np.diff(logit)[0] * np.ones(24), rtol=1e-9)
    assert orlib.lambda_grid(mu, sigma, m=7).shape == (7,)
    with pytest.raises(ValueError, match="m"):
        orlib.lambda_grid(mu, sigma, m=0)
    for bad in ({"lam_max": 1.0}, {"lam_max": 0.0}, {"margin": 0.0}):
        with pytest.raises(ValueError, match="lam_max"):
            orlib.lambda_grid(mu, sigma, **bad)
    with pytest.raises(ValueError, match="room"):
        orlib.lambda_grid(mu, sigma, lam_max=0.01)  # the corner lies above lam_max


def test_orlib_lambda_grid_rules(fake_orlib_dir):
    corner = orlib.orlib_lambda_grid(1, data_dir=fake_orlib_dir)
    inst = orlib.load_orlib(1, data_dir=fake_orlib_dir)
    np.testing.assert_allclose(corner, orlib.lambda_grid(inst.mu, inst.sigma))
    uniform = orlib.orlib_lambda_grid(1, m=5, rule="uniform", data_dir=fake_orlib_dir)
    np.testing.assert_allclose(uniform, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert orlib.orlib_lambda_grid(1, data_dir=fake_orlib_dir).shape == (25,)
    with pytest.raises(ValueError, match="rule"):
        orlib.orlib_lambda_grid(1, rule="chang", data_dir=fake_orlib_dir)


# --------------------------------------------------------------------------------------
# Downloaded OR-Library files (skipped when data/raw/orlib is empty)
# --------------------------------------------------------------------------------------


@needs_orlib
def test_port1_is_the_31_asset_instance_with_psd_covariance():
    inst = orlib.load_orlib(1)
    assert inst.n == 31
    assert np.linalg.eigvalsh(inst.sigma)[0] > 0.0
    np.testing.assert_allclose(inst.sigma, inst.sigma.T, atol=0.0)
    np.testing.assert_allclose(np.diag(inst.sigma), inst.std**2, rtol=1e-12)


@needs_orlib
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_every_port_has_the_published_size_and_a_psd_covariance(k):
    inst = orlib.load_orlib(k)
    assert inst.n == orlib.ORLIB_N[k]
    assert np.linalg.eigvalsh(inst.corr)[0] > 0.0
    assert np.all(np.abs(inst.corr) <= 1.0 + 1e-12)
    problem = orlib.orlib_problem(k, K=10, lam=0.5)
    assert (problem.n, problem.cardinality, problem.w_min, problem.w_max) == (inst.n, 10, 0.01, 1.0)


@needs_orlib
def test_provenance_records_url_and_hash_of_every_file():
    assert all(orlib.verify_orlib().values())
    prov = json.loads((ORLIB_DIR / "provenance.json").read_text(encoding="utf-8"))
    assert set(prov["files"]) == set(orlib.ORLIB_SHA256)
    for name, digest in orlib.ORLIB_SHA256.items():
        assert prov["files"][name]["sha256"] == digest
        assert prov["files"][name]["url"] == orlib.orlib_url(name)


@needs_orlib
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_portef_has_2000_rows_and_its_second_column_is_variance(k):
    inst = orlib.load_orlib(k)
    mean, var = orlib.load_orlib_frontier(k)
    assert mean.shape == var.shape == (2000,)
    j = int(np.argmax(inst.mu))
    # The top row is the maximum-return corner: all weight on the best asset, variance = std**2.
    assert mean[0] == pytest.approx(inst.mu[j], abs=1e-9)
    assert var[0] == pytest.approx(inst.std[j] ** 2, rel=1e-6)
    assert np.all(np.diff(mean) < 0.0)  # an equally spaced grid in return
    assert np.all(np.diff(var) <= 1e-12)


@needs_orlib
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_continuous_frontier_matches_portef_at_equal_return(k):
    """Our long-only frontier (w in [0,1], no cardinality) against ``portef<k>``.

    Points are matched by *return*: the file is a grid of 2000 equally spaced returns, so its
    variance is a function v_file(r) that we evaluate at the return of each of our λ-points. A
    cubic spline through the file is used: linear interpolation of the piecewise-quadratic frontier
    is off by up to 2e-5 near kinks (h²v''/8), the spline by < 1e-6. Matching by risk would
    need r(v), which is flat near the maximum-return corner.
    """
    inst = orlib.load_orlib(k)
    mean_f, var_f = orlib.load_orlib_frontier(k)
    order = np.argsort(mean_f)
    spline = CubicSpline(mean_f[order], var_f[order])
    for lam in orlib.orlib_lambda_grid(k, m=12):
        problem = orlib.orlib_problem(k, K=None, lam=float(lam), w_min=0.0)
        w = continuous_relaxation(problem).weights
        r, v = float(inst.mu @ w), float(w @ inst.sigma @ w)
        assert mean_f.min() <= r <= mean_f.max()
        assert v == pytest.approx(float(spline(r)), rel=5e-6)


@needs_orlib
@pytest.mark.parametrize("k", [1, 4])
def test_portef_rows_are_points_of_the_minimum_variance_frontier(k):
    """Node-exact check, no interpolation: min wᵀΣw s.t. μᵀw = r_j reproduces the file's variance."""
    inst = orlib.load_orlib(k)
    mean_f, var_f = orlib.load_orlib_frontier(k)
    scale = float(np.mean(np.diag(inst.sigma)))
    for j in (200, 700, 1200, 1800):
        w = cp.Variable(inst.n)
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(inst.sigma / scale))),
            [cp.sum(w) == 1, w >= 0, (inst.mu / scale) @ w == mean_f[j] / scale],
        )
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        assert problem.value * scale == pytest.approx(var_f[j], rel=2e-6)


# --------------------------------------------------------------------------------------
# Instance statistics and binding diagnostics
# --------------------------------------------------------------------------------------


def flat_problem(n: int, mu: np.ndarray, variance: float, **kwargs) -> PortfolioProblem:
    return PortfolioProblem(
        mu=mu,
        sigma=variance * np.eye(n),
        asset_names=tuple(f"A{i}" for i in range(n)),
        risk_aversion=kwargs.pop("risk_aversion", 2.0),
        **kwargs,
    )


def test_instance_statistics_on_a_hand_built_matrix():
    sigma = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, -0.018], [0.0, -0.018, 0.16]])
    mu = np.array([0.05, 0.10, 0.15])
    stats = instance_statistics(mu, sigma)
    rho = np.array([0.012 / (0.2 * 0.3), 0.0, -0.018 / (0.3 * 0.4)])  # (1,2), (1,3), (2,3)
    assert stats["n"] == 3
    assert (stats["vol_min"], stats["vol_median"], stats["vol_max"]) == pytest.approx(
        (0.2, 0.3, 0.4)
    )
    assert stats["corr_mean"] == pytest.approx(rho.mean())
    assert stats["corr_std"] == pytest.approx(rho.std())
    assert stats["corr_abs_mean"] == pytest.approx(np.abs(rho).mean())
    diag = np.diag(sigma)
    assert stats["cv_diag_sigma"] == pytest.approx(diag.std() / diag.mean())
    assert stats["cv_mu"] == pytest.approx(mu.std() / mu.mean())
    assert stats["cond_sigma"] == pytest.approx(np.linalg.cond(sigma))


def test_instance_statistics_undefined_quantities_are_none():
    sigma = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    stats = instance_statistics(np.zeros(2), sigma)
    assert stats["cond_sigma"] is None
    assert stats["cv_mu"] is None
    single = instance_statistics(np.array([0.1]), np.array([[0.04]]))
    assert single["corr_mean"] is None and single["corr_abs_mean"] is None
    json.dumps(stats)
    json.dumps(single)


def test_diagnostics_classify_a_constructed_binding_and_a_constructed_nonbinding_instance():
    """R4 acceptance. Σ = 0.04 I, μ equal: the optimum is 1/N on all N assets (cardinality binds);
    one dominant μ: the whole budget on one asset (cardinality cannot bind)."""
    n = 20
    binding = flat_problem(n, np.full(n, 0.10), 0.04, cardinality=5)
    d = binding_diagnostics(binding)
    assert d["s_cont"] == n and d["K"] == 5 and d["binding"] is True
    assert d["f_cont"] == pytest.approx(-0.10 + 0.5 * 2.0 * 0.04 / n, abs=1e-8)

    dominant = np.full(n, 0.05)
    dominant[0] = 1.0
    nonbinding = flat_problem(n, dominant, 0.04, cardinality=5)
    d = binding_diagnostics(nonbinding)
    assert d["s_cont"] == 1 and d["binding"] is False


def test_s_cont_matches_the_water_filling_solution():
    """Σ = 0.04 I, ra = 2, μ_i = 0.01 i (i = 1..10): w_i = (μ_i - nu)/0.08 on the top four."""
    mu = 0.01 * np.arange(1, 11)
    problem = flat_problem(10, mu, 0.04)
    w = continuous_relaxation(problem).weights
    np.testing.assert_allclose(w[-4:], [0.0625, 0.1875, 0.3125, 0.4375], atol=1e-6)
    assert binding_diagnostics(problem, K=3)["binding"] is True
    assert binding_diagnostics(problem, K=4)["binding"] is False
    assert binding_diagnostics(problem, K=5)["binding"] is False
    assert binding_diagnostics(problem, K=3)["s_cont"] == 4


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_binding_means_the_cardinality_constrained_optimum_is_strictly_worse(seed):
    """Definition check by exhaustive search on N = 8: K = s_cont - 1 binds, K = s_cont does not."""
    rng = np.random.default_rng(seed)
    n = 8
    mu = rng.uniform(0.05, 0.25, size=n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.005 + np.eye(n) * 0.01
    problem = PortfolioProblem(
        mu=mu, sigma=sigma, asset_names=tuple(f"A{i}" for i in range(n)), risk_aversion=2.0
    )
    s = binding_diagnostics(problem)["s_cont"]
    f_cont = solve_continuous_mvo(problem).objective
    at_s = brute_force_cardinality_continuous(problem, cardinality=s)
    assert at_s.objective == pytest.approx(f_cont, abs=1e-7)
    if s > 1:
        below = brute_force_cardinality_continuous(problem, cardinality=s - 1)
        assert below.objective > f_cont + 1e-6
        assert binding_diagnostics(problem, K=s - 1)["binding"] is True
    assert binding_diagnostics(problem, K=s)["binding"] is False


def test_binding_diagnostics_relaxes_buy_in_threshold_and_reads_the_problem_cardinality():
    """w_min > 0 is a buy-in threshold of held assets; solve_continuous_mvo applies it to all."""
    rng = np.random.default_rng(3)
    n = 150
    mu = rng.uniform(0.05, 0.25, size=n)
    a = rng.standard_normal((n, n))
    sigma = a @ a.T * 0.005 + np.eye(n) * 0.01
    names = tuple(f"A{i}" for i in range(n))
    problem = PortfolioProblem(
        mu=mu, sigma=sigma, asset_names=names, w_min=0.01, risk_aversion=2.0, cardinality=10
    )
    with pytest.raises(RuntimeError):  # 150 * 0.01 > 1: infeasible as a global lower bound
        solve_continuous_mvo(problem)
    d = binding_diagnostics(problem)
    assert d["K"] == 10 and d["binding"] is (d["s_cont"] > 10)
    assert binding_diagnostics(problem, K=d["s_cont"] + 1)["binding"] is False
    d_free = binding_diagnostics(
        PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=2.0)
    )
    assert d_free["K"] is None and d_free["binding"] is None
    assert d_free["s_cont"] == d["s_cont"]
    with pytest.raises(ValueError):
        binding_diagnostics(problem, K=n + 1)


def test_support_tolerance_is_in_units_of_the_budget():
    mu = 0.01 * np.arange(1, 11)
    problem = flat_problem(10, mu, 0.04)  # weights 0.0625, 0.1875, 0.3125, 0.4375
    assert binding_diagnostics(problem)["s_cont"] == 4
    assert binding_diagnostics(problem, support_tol=0.1)["s_cont"] == 3
    assert k_from_scont(problem, 1.0, support_tol=0.1) == 3


@pytest.mark.parametrize("factor", [1e-3, 1.0, 1e3])
def test_continuous_relaxation_does_not_depend_on_the_scale_of_the_data(factor):
    """(μ, Σ) -> (aμ, aΣ) leaves the minimiser unchanged; the normalisation makes the numerics
    insensitive to a as well (the objective is evaluated on the original problem)."""
    mu, sigma = random_moments(12, seed=4)
    names = tuple(f"A{i}" for i in range(12))
    base = PortfolioProblem(mu=mu * 40, sigma=sigma * 40, asset_names=names, risk_aversion=2.0)
    scaled = PortfolioProblem(
        mu=mu * 40 * factor, sigma=sigma * 40 * factor, asset_names=names, risk_aversion=2.0
    )
    sol_base, sol = continuous_relaxation(base), continuous_relaxation(scaled)
    # Exact up to rounding: without the normalisation the weights differ by ~1e-5 at a = 1e-3.
    np.testing.assert_allclose(sol.weights, sol_base.weights, atol=1e-9)
    assert sol.objective == pytest.approx(scaled.objective_value(sol.weights))
    assert sol.solver_meta["scale"] == pytest.approx(float(np.mean(np.diag(sigma))) * 40 * factor)
    assert binding_diagnostics(scaled)["s_cont"] == binding_diagnostics(base)["s_cont"]


def test_instance_statistics_rejects_a_nonpositive_diagonal():
    with pytest.raises(ValueError, match="diagonal"):
        instance_statistics(np.zeros(2), np.array([[0.0, 0.0], [0.0, 1.0]]))


def test_binding_diagnostics_row_is_json_serialisable_and_complete():
    problem = flat_problem(12, np.linspace(0.05, 0.2, 12), 0.04, cardinality=3)
    row = binding_diagnostics(problem)
    expected = {
        "n", "K", "s_cont", "s_cont_frac", "binding", "f_cont", "cond_sigma", "cv_diag_sigma",
        "cv_mu", "corr_mean", "corr_std", "corr_abs_mean", "vol_min", "vol_median", "vol_max",
        "cont_solver", "cont_status", "cont_wall_s",
    }  # fmt: skip
    assert expected <= set(row)
    assert json.loads(json.dumps(row)) == row
    assert type(row["binding"]) is bool and type(row["s_cont"]) is int
    assert row["cont_status"] == "optimal" and row["cont_solver"].startswith("cvxpy_continuous/")


def test_k_from_scont_floors_so_that_fractions_below_one_always_bind():
    mu = 0.01 * np.arange(1, 11)
    problem = flat_problem(10, mu, 0.04)  # s_cont = 4
    assert k_from_scont(problem, 0.5) == 2
    assert k_from_scont(problem, 0.74) == 2  # floor(2.96)
    assert k_from_scont(problem, 0.76) == 3
    assert k_from_scont(problem, 1.0) == 4  # K = s_cont: the boundary, not binding
    assert k_from_scont(problem, 1.5) == 6
    assert k_from_scont(problem, 50.0) == 10  # clipped to N
    assert k_from_scont(problem, 0.01) == 1  # never below 1
    assert k_from_scont(problem, 0.5, s_cont=8) == 4  # precomputed s_cont is used as given
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="fraction"):
            k_from_scont(problem, bad)
    for fraction in (0.25, 0.5, 0.75, 0.99):
        k = k_from_scont(problem, fraction)
        assert binding_diagnostics(problem, K=k)["binding"] is (k < 4)


@needs_orlib
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_orlib_grid_yields_both_binding_and_nonbinding_cells(k):
    """The corner-anchored 25-point grid has no single-asset (s_cont = 1) cells and, for K = 5
    and K = 10, at least three cells of each kind on every port."""
    s_cont = []
    for lam in orlib.orlib_lambda_grid(k):
        d = binding_diagnostics(orlib.orlib_problem(k, K=5, lam=float(lam)))
        s_cont.append(d["s_cont"])
    s_cont = np.array(s_cont)
    assert s_cont.min() >= 2
    for big_k in (5, 10):
        assert (s_cont > big_k).sum() >= 3
        assert (s_cont <= big_k).sum() >= 3


@needs_orlib
@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::PendingDeprecationWarning")
def test_s_cont_agrees_with_an_osqp_polished_reference_on_orlib_cells():
    """Calibration of the support tolerance (1e-5 on the normalised problem): the interior-point
    support equals the support of a polished active-set solution on all tested cells."""
    for k in (1, 4):
        inst = orlib.load_orlib(k)
        scale = float(np.mean(np.diag(inst.sigma)))
        for lam in orlib.orlib_lambda_grid(k, m=9):
            ra = orlib.risk_aversion_from_lambda(float(lam))
            w = cp.Variable(inst.n)
            reference = cp.Problem(
                cp.Minimize(
                    -(inst.mu / scale) @ w + 0.5 * ra * cp.quad_form(w, cp.psd_wrap(inst.sigma / scale))
                ),
                [cp.sum(w) == 1, w >= 0, w <= 1],
            )
            reference.solve(solver="OSQP", eps_abs=1e-12, eps_rel=1e-12, polish=True, max_iter=10**6)
            s_ref = int((np.asarray(w.value) > 1e-9).sum())
            d = binding_diagnostics(orlib.orlib_problem(k, K=None, lam=float(lam), w_min=0.0))
            assert d["s_cont"] == s_ref


# --------------------------------------------------------------------------------------
# Legacy dense generator (Experiment A) and the R4 contrast with the bootstrap Σ
# --------------------------------------------------------------------------------------


def load_script_function(path: Path, name: str):
    """Compile one top-level function of a legacy script without importing the script (which
    pulls in the solvers)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    code = compile(
        ast.Module(body=[func], type_ignores=[]),
        str(path),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    namespace = {"np": np, "PortfolioProblem": PortfolioProblem}
    exec(code, namespace)
    return namespace[name]


def test_legacy_dense_instance_reproduces_the_experiment_a_generator():
    path = REPO_ROOT / "scripts" / "experiment_a_scalability.py"
    if not path.exists():
        pytest.skip("scripts/experiment_a_scalability.py not found")
    make_synthetic_problem = load_script_function(path, "make_synthetic_problem")
    for n, seed in ((20, 42), (35, 7)):
        legacy = make_synthetic_problem(n=n, seed=seed, K=5)
        mu, sigma = legacy_dense_instance(n, seed)
        np.testing.assert_array_equal(mu, legacy.mu)
        np.testing.assert_array_equal(sigma, legacy.sigma)


def test_describe_legacy_dense_volatility_grows_with_n_and_correlations_vanish():
    """Σ = 0.005 AAᵀ + 0.01 I: E[Σ_ii] = 0.005 N + 0.01, so volatility ~33 % at N = 20 and
    ~100 % at N = 200, while correlations are ~N(0, 1/N): mean ≈ 0, mean |corr| ≈ 0.8/√N."""
    stats = {n: describe_legacy_dense(n, seed=42) for n in (20, 50, 100, 200)}
    vols = [stats[n]["vol_median"] for n in (20, 50, 100, 200)]
    assert vols == sorted(vols) and len(set(vols)) == 4
    for n, s in stats.items():
        assert s["n"] == n and s["seed"] == 42 and s["family"] == "dense"
        assert s["vol_median"] == pytest.approx(np.sqrt(0.005 * n + 0.01), rel=0.05)
        # mean corr = (‖Σ_i â_i‖² - N)/(N(N - 1)) with ‖Σ_i â_i‖² ≈ χ²_N, so sd ≈ √2/N^1.5.
        assert abs(s["corr_mean"]) < 4.0 * np.sqrt(2.0) / n**1.5
        assert 0.6 / np.sqrt(n) < s["corr_abs_mean"] < 1.0 / np.sqrt(n)
        assert s["mu_min"] >= 0.05 and s["mu_max"] <= 0.25
    assert stats[20]["vol_median"] < 0.40 and stats[200]["vol_median"] > 0.90
    assert stats[200]["corr_abs_mean"] < 0.07


SP500_PRICES = REPO_ROOT / "data" / "processed" / "sp500_prices.parquet"


@pytest.mark.skipif(not SP500_PRICES.exists(), reason="data/processed/sp500_prices.parquet missing")
@pytest.mark.parametrize("n", [50, 100])
def test_r4_bootstrap_sigma_does_not_bind_at_n_over_4_but_legacy_dense_does(n):
    k = max(3, round(0.25 * n))
    mu_real, sigma_real, _ = load_real_covariance(SP500_PRICES)
    names = tuple(f"B{i}" for i in range(n))
    mu_b, sigma_b = bootstrap_realistic_instance(mu_real, sigma_real, n_target=n, seed=42)
    boot = PortfolioProblem(mu=mu_b, sigma=sigma_b, asset_names=names, risk_aversion=2.0, cardinality=k)
    mu_d, sigma_d = legacy_dense_instance(n, 42)
    dense = PortfolioProblem(mu=mu_d, sigma=sigma_d, asset_names=names, risk_aversion=2.0, cardinality=k)
    d_boot, d_dense = binding_diagnostics(boot), binding_diagnostics(dense)
    assert d_boot["binding"] is False and d_boot["s_cont"] < k
    assert d_dense["binding"] is True and d_dense["s_cont"] > k


# --------------------------------------------------------------------------------------
# Factor instances
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "seed"), [(60, 1), (200, 2), (400, 3)])
def test_factor_instance_statistics_are_in_the_declared_ranges(n, seed):
    mu, sigma, meta = factor_instance(n, seed)
    assert mu.shape == (n,) and sigma.shape == (n, n)
    stats = instance_statistics(mu, sigma)
    vols = np.sqrt(np.diag(sigma))
    assert vols.min() >= 0.15 - 1e-12 and vols.max() <= 0.60 + 1e-12
    assert 0.22 <= stats["vol_median"] <= 0.34
    assert 0.20 <= stats["corr_mean"] <= 0.40
    assert stats["corr_mean"] == pytest.approx(0.30, abs=5e-3)  # calibrated, not just in range
    assert meta["mean_corr"] == pytest.approx(stats["corr_mean"], abs=1e-12)
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-15)
    assert np.linalg.eigvalsh(sigma)[0] > 0.0
    assert stats["cond_sigma"] < 1e4
    # Sector blocks: pairs in the same sector are noticeably more correlated.
    sector = meta["sector"]
    corr = sigma / np.outer(vols, vols)
    same = sector[:, None] == sector[None, :]
    off_diag = ~np.eye(n, dtype=bool)
    assert corr[same & off_diag].mean() > corr[~same].mean() + 0.05


def test_factor_instance_returns_the_factor_decomposition():
    n, n_factors, n_sectors = 80, 3, 10
    _, sigma, meta = factor_instance(n, 11, n_factors=n_factors, n_sectors=n_sectors)
    loadings, factor_var, idio_var = meta["loadings"], meta["factor_var"], meta["idio_var"]
    assert loadings.shape == (n, n_factors + n_sectors)
    assert factor_var.shape == (n_factors + n_sectors,) and idio_var.shape == (n,)
    assert np.all(factor_var > 0) and np.all(idio_var > 0)
    np.testing.assert_allclose(
        loadings @ np.diag(factor_var) @ loadings.T + np.diag(idio_var), sigma, rtol=1e-10, atol=1e-14
    )
    assert meta["n_factors"] == n_factors and meta["n_sectors"] == n_sectors
    assert set(np.unique(meta["sector"])) == set(range(n_sectors))  # no empty sector
    for key in ("family", "seed", "mean_corr_target", "vol_median", "mu_mode"):
        assert key in meta
    assert meta["family"] == "factor" and meta["seed"] == 11


def test_factor_instance_is_deterministic_in_the_seed():
    a = factor_instance(50, 5)
    b = factor_instance(50, 5)
    c = factor_instance(50, 6)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.allclose(a[1], c[1])


def test_factor_mu_is_linked_to_risk_by_default_and_random_on_request():
    mu, sigma, meta = factor_instance(300, 5)
    beta = meta["beta"]
    assert beta.mean() == pytest.approx(1.0)  # beta against the equal-weighted portfolio
    assert np.corrcoef(mu, beta)[0, 1] > 0.4
    assert mu.mean() == pytest.approx(0.03 + 0.05, abs=0.01)  # rf + premium * mean(beta)
    mu_r, sigma_r, meta_r = factor_instance(300, 5, mu_mode="random")
    assert mu_r.min() >= 0.05 and mu_r.max() <= 0.25
    assert abs(np.corrcoef(mu_r, meta_r["beta"])[0, 1]) < 0.2
    np.testing.assert_array_equal(sigma, sigma_r)  # same seed: same Σ in both modes
    assert meta_r["mu_mode"] == "random"


def test_factor_instance_validates_arguments_and_handles_few_assets():
    for kwargs in (
        {"n": 1, "seed": 0},
        {"n": 10, "seed": 0, "n_factors": 0},
        {"n": 10, "seed": 0, "n_sectors": 0},
        {"n": 10, "seed": 0, "mean_corr": 0.0},
        {"n": 10, "seed": 0, "mean_corr": 1.0},
        {"n": 10, "seed": 0, "vol_range": (0.5, 0.2)},
        {"n": 10, "seed": 0, "mu_mode": "momentum"},
    ):
        with pytest.raises(ValueError):
            factor_instance(**kwargs)
    _, sigma, meta = factor_instance(6, 0, n_sectors=10)  # more sectors than assets
    assert meta["n_sectors"] == 6 and sigma.shape == (6, 6)
    _, sigma, meta = factor_instance(30, 0, n_factors=1)  # market factor only
    assert meta["loadings"].shape == (30, 1 + 10)
    with pytest.raises(ValueError, match="reachable"):
        factor_instance(30, 0, mean_corr=0.99)
    with pytest.raises(ValueError, match="too small"):
        factor_instance(30, 0, mean_corr=1e-12)


def test_factor_instances_bind_when_k_is_a_fraction_of_s_cont():
    """The point of the family: K = fraction of s_cont gives a controlled binding status."""
    mu, sigma, _ = factor_instance(100, 7)
    names = tuple(f"F{i}" for i in range(100))
    base = PortfolioProblem(mu=mu, sigma=sigma, asset_names=names, risk_aversion=2.0)
    s_cont = binding_diagnostics(base)["s_cont"]
    assert 4 <= s_cont < 100  # at ra = 2 the optimum holds ~10 % of the assets, not a corner
    k_bind = k_from_scont(base, 0.5)
    k_free = k_from_scont(base, 1.5)
    assert binding_diagnostics(base, K=k_bind)["binding"] is True
    assert binding_diagnostics(base, K=k_free)["binding"] is False
