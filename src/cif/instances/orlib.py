"""OR-Library portfolio instances port1-port5 (Beasley 1990; Chang et al. 2000).

The five files are the test problems of Chang, Meade, Beasley and Sharaiha, "Heuristics for
cardinality constrained portfolio optimisation", *Comput. Oper. Res.* 27 (2000) 1271-1302:
Hang Seng, DAX, FTSE, S&P 100 and Nikkei with N = 31, 85, 89, 98, 225 assets. Layout of
``port<k>.txt`` (OR-Library page ``portinfo.html``)::

    N
    mean_1 std_1                      (N lines, one per asset)
    ...
    i j corr_ij                       (1-based; the originals list every pair i ≤ j, with corr_ii = 1)

and of ``portef<k>.txt`` — the *unconstrained* efficient frontier (long-only, no cardinality)
as 2000 rows ``mean_return variance`` (the second column is the **variance**, not the std).

Units are those of the files (per-period, not annualised). Mean-variance minimisers are
invariant to a common rescaling ``(μ, Σ) → (aμ, aΣ)``, so the λ ↔ risk-aversion map below does
not depend on the period.

Parameterisation (Chang et al. 2000, standard CCEF)::

    minimize   λ wᵀΣw - (1 - λ) μᵀw
    subject to Σ w = 1,  Σ z ≤ K,  ε z_i ≤ w_i ≤ δ z_i,  z_i ∈ {0, 1}

With ``PortfolioProblem`` (``-μᵀw + (ra/2) wᵀΣw``) the two objectives differ by the positive
factor ``(1 - λ)``, hence have the same minimisers, for ``ra = 2λ/(1 - λ)``. At ``λ = 1`` the
return term vanishes (minimum variance) and ``ra = ∞``; there ``orlib_problem`` returns
``μ = 0, ra = 2`` so that ``objective_value(w) = wᵀΣw`` exactly.

λ grid (``lambda_grid``, default 25 points). A grid uniform in λ wastes cells: on port1-5 up to
11 of 25 uniform points have the whole budget in the best-mean asset (``s_cont = 1``) and for
K = 10 only 0-5 are binding. We anchor the grid at the "corner" ``ra_c`` — the largest risk
aversion at which that single asset is still optimal (closed form, :func:`corner_risk_aversion`)
— and space the points geometrically in ``ra`` (equivalently uniformly in log-odds of λ) from
``1.1·ra_c`` to ``ra(0.999)``. The upper end is where the support has saturated at the
minimum-variance one. ``rule="uniform"`` gives ``λ_j = j/(m - 1)``, the form used in the
heuristic literature, for comparability.

Differences from Chang et al. to keep in mind:

* Cardinality: the project (``milp.py``) uses ``Σ z ≤ K``. The paper's CCEF may fix exactly K
  assets (``Σ z = K``; not checked against the paper in this repository). The ``≤`` optimum is
  never worse than the ``=`` one and coincides with it whenever it holds K assets.
* ``w_min`` is the buy-in threshold ε of *held* assets (``w_min z ≤ w``) in the MIQP, in
  subset-wise refinement (``neal_sampler``, ``brute_force_cardinality_continuous``). But
  ``solve_continuous_mvo`` imposes ``w ≥ w_min`` on **all** assets: with the Chang default
  ε = 0.01 it is infeasible when N·ε > 1 (port5: 2.25). For the cardinality-free relaxation
  use ``w_min = 0`` (``cif.instances.diagnostics`` does).

Download: ``download_orlib()`` (explicitly commissioned, local only). SHA-256 and URL of every
file go to ``data/raw/orlib/provenance.json`` (R8); the hashes observed on the first download
are also pinned in :data:`ORLIB_SHA256` so that a clean clone can prove it uses the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cif.problem import PortfolioProblem

ORLIB_INFO_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/portinfo.html"
ORLIB_BASE_URL = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
ORLIB_REFERENCE = (
    "Chang, T.-J., Meade, N., Beasley, J.E., Sharaiha, Y.M. (2000) Heuristics for cardinality "
    "constrained portfolio optimisation. Computers & Operations Research 27, 1271-1302"
)

#: Number of assets of port1..port5 (Hang Seng, DAX, FTSE, S&P 100, Nikkei).
ORLIB_N: dict[int, int] = {1: 31, 2: 85, 3: 89, 4: 98, 5: 225}

#: SHA-256 of the files as downloaded on 2026-10-04 (trust on first use; the files are static).
ORLIB_SHA256: dict[str, str] = {
    "port1.txt": "15615853de5990a70d913c71b91fd90aa45856832c94a6f00e79907fddb5b1d2",
    "port2.txt": "6cad94fd24c8561772fd975621e40a6b06bd613dc3e4e145d595b143e9f9742d",
    "port3.txt": "e91527a0befeb789491e627478a703e98f8ce9a79fa9557742d30078c0930838",
    "port4.txt": "063ae8837619a81f0a1c3ab79a8c86a1596c6da8715206d123871b39ff95cea4",
    "port5.txt": "96dababdb62bdf1428b487a21c0ab5a7eaa1bcf8065c0dfdaf3eff2476b8070a",
    "portef1.txt": "dc847d0dcbdf9e295226826df942e702b201db6450c0ec5ea9e28c84b18724a6",
    "portef2.txt": "25a188898555661f95110dbe869599c12b7f3c8592f256e3545b0101dd3189f3",
    "portef3.txt": "ca9ff3a371987d9ba76546533523f369bfe9afd5ba53d92e1dbd18255744cd81",
    "portef4.txt": "715a53e38ac5176b498d270bbdafa3debdff3655dd7bee9c36147ac6bad8ff9c",
    "portef5.txt": "15422dde378a8a683b2743c963a290f3f1fa054a238b74d50b9dcde759d444cc",
}

PROVENANCE_NAME = "provenance.json"


def default_orlib_dir() -> Path:
    """``<repo>/data/raw/orlib`` (the package is installed in editable mode, ``src`` layout)."""
    return Path(__file__).resolve().parents[3] / "data" / "raw" / "orlib"


def orlib_url(name: str) -> str:
    return f"{ORLIB_BASE_URL}/{name}"


def orlib_file_names(ports: Iterable[int] = tuple(ORLIB_N)) -> list[str]:
    """``['port1.txt', 'portef1.txt', 'port2.txt', ...]`` for the requested ports."""
    names: list[str] = []
    for k in ports:
        _check_port(k)
        names += [f"port{k}.txt", f"portef{k}.txt"]
    return names


def _check_port(k: int) -> None:
    if k not in ORLIB_N:
        raise ValueError(f"port must be in {sorted(ORLIB_N)}, got {k!r}")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class OrlibInstance:
    """A parsed port file: per-period moments, correlation and ``Σ_ij = corr_ij · std_i · std_j``."""

    name: str
    mu: np.ndarray
    std: np.ndarray
    corr: np.ndarray
    sigma: np.ndarray

    @property
    def n(self) -> int:
        return self.mu.shape[0]


def parse_orlib_port(text: str, name: str = "port", psd_tol: float = 1e-8) -> OrlibInstance:
    """Parse the text of a ``port<k>`` file.

    The parser is token-based (line breaks and spacing are irrelevant). The pair section may
    list each pair once (either triangle) or twice, with or without the unit diagonal; every
    off-diagonal pair must be present and repeated pairs must agree. The correlation matrix
    must be PSD: ``λ_min ≥ -psd_tol · λ_max``.
    """
    tokens = text.split()
    if not tokens:
        raise ValueError(f"{name}: empty file")
    try:
        n = int(tokens[0])
    except ValueError as exc:
        raise ValueError(f"{name}: first token {tokens[0]!r} is not an integer asset count") from exc
    if n < 1:
        raise ValueError(f"{name}: asset count must be positive, got {n}")
    if len(tokens) < 1 + 2 * n:
        raise ValueError(f"{name}: truncated, expected {2 * n} mean/std tokens after N={n}")
    try:
        values = np.array(tokens[1:], dtype=float)
    except ValueError as exc:
        raise ValueError(f"{name}: non-numeric token in the data section") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: non-finite value in the data section")

    mean_std = values[: 2 * n].reshape(n, 2)
    mu, std = mean_std[:, 0].copy(), mean_std[:, 1].copy()
    if np.any(std <= 0.0):
        raise ValueError(f"{name}: std must be positive for every asset")

    rest = values[2 * n :]
    if rest.size % 3 != 0:
        raise ValueError(
            f"{name}: pair section has {rest.size} tokens, not a multiple of 3 "
            "(truncated or malformed file)"
        )
    pairs = rest.reshape(-1, 3)
    i_raw, j_raw, rho = pairs[:, 0], pairs[:, 1], pairs[:, 2]
    if (
        np.any(i_raw != np.round(i_raw))
        or np.any(j_raw != np.round(j_raw))
        or np.any(i_raw < 1)
        or np.any(j_raw < 1)
        or np.any(i_raw > n)
        or np.any(j_raw > n)
    ):
        raise ValueError(f"{name}: pair index outside 1..{n}")
    if np.any(np.abs(rho) > 1.0 + 1e-9):
        raise ValueError(f"{name}: correlation outside [-1, 1]")
    ii, jj = i_raw.astype(int) - 1, j_raw.astype(int) - 1
    if np.unique(ii * n + jj).size != ii.size:
        raise ValueError(f"{name}: duplicate (i, j) pair in the correlation section")

    corr = np.full((n, n), np.nan)
    corr[ii, jj] = rho
    given = ~np.isnan(corr)
    both = given & given.T
    if np.any(np.abs(corr[both] - corr.T[both]) > 1e-9):
        raise ValueError(f"{name}: conflicting correlations for (i, j) and (j, i)")
    corr = np.where(given, corr, corr.T)
    diag = np.diag_indices(n)
    d = corr[diag]
    if np.any(~np.isnan(d) & (np.abs(d - 1.0) > 1e-6)):
        raise ValueError(f"{name}: diagonal correlation must be 1")
    corr[diag] = 1.0
    missing = np.argwhere(np.triu(np.isnan(corr), k=1))
    if missing.size:
        i, j = missing[0] + 1
        raise ValueError(
            f"{name}: {len(missing)} missing correlation pair(s), first is ({i}, {j})"
        )

    eig = np.linalg.eigvalsh(corr)
    if eig[0] < -psd_tol * eig[-1]:
        raise ValueError(f"{name}: correlation matrix is not PSD (min eigenvalue {eig[0]:.3e})")

    sigma = corr * np.outer(std, std)
    return OrlibInstance(name=name, mu=mu, std=std, corr=corr, sigma=sigma)


def parse_orlib_frontier(text: str, name: str = "portef") -> tuple[np.ndarray, np.ndarray]:
    """Parse ``portef<k>``: rows ``mean_return variance`` → ``(mean, variance)`` in file order."""
    tokens = text.split()
    if not tokens:
        raise ValueError(f"{name}: empty file")
    try:
        values = np.array(tokens, dtype=float)
    except ValueError as exc:
        raise ValueError(f"{name}: non-numeric token") from exc
    if values.size % 2 != 0:
        raise ValueError(f"{name}: expected two columns (mean, variance), got {values.size} values")
    rows = values.reshape(-1, 2)
    if not np.all(np.isfinite(rows)) or np.any(rows[:, 1] < 0.0):
        raise ValueError(f"{name}: non-finite value or negative variance")
    return rows[:, 0].copy(), rows[:, 1].copy()


# --------------------------------------------------------------------------------------
# Download and provenance
# --------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_get(url: str, timeout: float) -> bytes:
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _provenance_header() -> dict:
    return {
        "source": "OR-Library (J. E. Beasley), portfolio optimisation test problems",
        "info_url": ORLIB_INFO_URL,
        "base_url": ORLIB_BASE_URL,
        "reference": ORLIB_REFERENCE,
        "files": {},
    }


def _read_provenance(root: Path) -> dict:
    prov = _provenance_header()
    path = root / PROVENANCE_NAME
    if path.exists():
        prov["files"] = dict(json.loads(path.read_text(encoding="utf-8")).get("files", {}))
    return prov


def _validate_download(name: str, content: bytes, pinned: Mapping[str, str]) -> None:
    digest = _sha256(content)
    if name in pinned and digest != pinned[name]:
        raise ValueError(
            f"{name}: SHA-256 {digest} differs from the pinned {pinned[name]} "
            "(file changed upstream or the download is corrupted)"
        )
    try:
        text = content.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: response is not plain ASCII text") from exc
    if name.startswith("portef"):
        parse_orlib_frontier(text, name=name)
        return
    k = int(name.removeprefix("port").removesuffix(".txt"))
    inst = parse_orlib_port(text, name=name)
    if inst.n != ORLIB_N[k]:
        raise ValueError(f"{name}: expected N={ORLIB_N[k]} assets, the file has N={inst.n}")


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def download_orlib(
    data_dir: Path | str | None = None,
    ports: Iterable[int] = tuple(ORLIB_N),
    *,
    force: bool = False,
    timeout: float = 60.0,
    fetch: Callable[[str], bytes] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
) -> dict:
    """Download ``port<k>.txt`` and ``portef<k>.txt`` and record their provenance.

    Idempotent: a file whose SHA-256 equals the recorded one is not fetched again (``force``
    refetches). All requested files are fetched and validated (pinned SHA-256, plain ASCII, parses
    as a port/frontier file, expected N) *before* anything is written, so a failure leaves the
    directory untouched. Returns the provenance dict (also written to ``provenance.json``).

    ``fetch`` (URL → bytes) replaces HTTP for offline tests; ``expected_sha256`` defaults to
    :data:`ORLIB_SHA256`, ``{}`` disables the pin check.
    """
    root = Path(data_dir) if data_dir is not None else default_orlib_dir()
    pinned = ORLIB_SHA256 if expected_sha256 is None else expected_sha256
    getter = fetch if fetch is not None else (lambda url: _http_get(url, timeout))
    prov = _read_provenance(root)

    fresh: dict[str, bytes] = {}
    for name in orlib_file_names(ports):
        entry = prov["files"].get(name)
        path = root / name
        if (
            not force
            and entry is not None
            and path.exists()
            and _sha256(path.read_bytes()) == entry["sha256"]
        ):
            continue
        content = getter(orlib_url(name))
        _validate_download(name, content, pinned)
        fresh[name] = content

    if fresh:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for name, content in fresh.items():
            _atomic_write(root / name, content)
            prov["files"][name] = {
                "url": orlib_url(name),
                "sha256": _sha256(content),
                "bytes": len(content),
                "downloaded_at": stamp,
            }
        prov["files"] = dict(sorted(prov["files"].items()))
        (root / PROVENANCE_NAME).write_text(
            json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return prov


def verify_orlib(data_dir: Path | str | None = None) -> dict[str, bool]:
    """Re-hash every file listed in ``provenance.json``: ``{name: matches}`` (missing → False)."""
    root = Path(data_dir) if data_dir is not None else default_orlib_dir()
    path = root / PROVENANCE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run cif.instances.orlib.download_orlib()")
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    return {
        name: (root / name).exists() and _sha256((root / name).read_bytes()) == entry["sha256"]
        for name, entry in files.items()
    }


# --------------------------------------------------------------------------------------
# Loading and the problem builder
# --------------------------------------------------------------------------------------


def _read(root: Path, name: str) -> str:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run cif.instances.orlib.download_orlib()")
    return path.read_text(encoding="ascii")


def load_orlib(k: int, data_dir: Path | str | None = None) -> OrlibInstance:
    """Parse ``port<k>.txt`` from ``data_dir`` (default ``data/raw/orlib``); checks N."""
    _check_port(k)
    root = Path(data_dir) if data_dir is not None else default_orlib_dir()
    inst = parse_orlib_port(_read(root, f"port{k}.txt"), name=f"port{k}")
    if inst.n != ORLIB_N[k]:
        raise ValueError(f"port{k}: expected N={ORLIB_N[k]} assets, the file has N={inst.n}")
    return inst


def load_orlib_frontier(
    k: int, data_dir: Path | str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``portef<k>.txt`` → ``(mean_return, variance)`` of the unconstrained frontier (file order)."""
    _check_port(k)
    root = Path(data_dir) if data_dir is not None else default_orlib_dir()
    return parse_orlib_frontier(_read(root, f"portef{k}.txt"), name=f"portef{k}")


def risk_aversion_from_lambda(lam: float) -> float:
    """``ra = 2λ/(1 - λ)``: Chang's ``λ wᵀΣw - (1-λ) μᵀw`` equals ``(1-λ)·(-μᵀw + (ra/2) wᵀΣw)``.

    ``λ = 0`` gives ``ra = 0`` (maximum return), ``λ = 1`` gives ``inf`` (minimum variance).
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam!r}")
    return math.inf if lam == 1.0 else 2.0 * lam / (1.0 - lam)


def lambda_from_risk_aversion(ra: float) -> float:
    """Inverse of :func:`risk_aversion_from_lambda`: ``λ = ra/(2 + ra)``."""
    if not ra >= 0.0:
        raise ValueError(f"risk aversion must be >= 0, got {ra!r}")
    return 1.0 if math.isinf(ra) else ra / (2.0 + ra)


def chang_objective(mu: np.ndarray, sigma: np.ndarray, lam: float, weights: np.ndarray) -> float:
    """Chang et al. objective ``λ wᵀΣw - (1 - λ) μᵀw`` (original scale, true ``μ``)."""
    w = np.asarray(weights, dtype=float)
    return float(lam * (w @ sigma @ w) - (1.0 - lam) * (np.asarray(mu) @ w))


def corner_risk_aversion(mu: np.ndarray, sigma: np.ndarray) -> float:
    """Largest ``ra`` for which the whole budget in the best-mean asset ``j`` is optimal.

    For ``f(w) = -μᵀw + (ra/2) wᵀΣw`` on the simplex, ``w = e_j`` is optimal iff for every ``i``
    ``μ_j - μ_i ≥ ra (Σ_jj - Σ_ij)``. Only ``i`` with ``Σ_jj > Σ_ij`` limit ``ra``, hence
    ``ra_c = min_i (μ_j - μ_i)/(Σ_jj - Σ_ij)`` over those ``i`` (budget 1, ``w_max = 1``). Above
    it a second asset enters. Returns ``inf`` when no ``i`` limits ``ra`` and ``0`` on a tie for
    the best mean.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    j = int(np.argmax(mu))
    d_var = sigma[j, j] - sigma[:, j]
    d_mu = mu[j] - mu
    limiting = d_var > 0.0
    limiting[j] = False
    if not limiting.any():
        return math.inf
    return float(np.min(d_mu[limiting] / d_var[limiting]))


def lambda_grid(
    mu: np.ndarray,
    sigma: np.ndarray,
    m: int = 25,
    lam_max: float = 0.999,
    margin: float = 1.1,
) -> np.ndarray:
    """``m`` values of λ, uniform in log-odds, from just above the corner to ``lam_max``.

    ``ra_j`` is geometric from ``margin · ra_c`` (:func:`corner_risk_aversion`) to
    ``ra(lam_max)``; returned as ``λ_j = ra_j/(2 + ra_j)``. Every point lies above the corner, so
    the exact continuous optimum holds at least two assets.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not (0.0 < lam_max < 1.0 and margin > 0.0):
        raise ValueError(f"need 0 < lam_max < 1 and margin > 0, got {lam_max}, {margin}")
    ra_c = corner_risk_aversion(mu, sigma)
    if not 0.0 < ra_c < math.inf:
        raise ValueError(
            f"lambda_grid: corner risk aversion is {ra_c} (a tie for the best mean or an "
            "asset that is optimal for every ra); there is no informative range"
        )
    ra_lo, ra_hi = margin * ra_c, risk_aversion_from_lambda(lam_max)
    if not ra_lo < ra_hi:
        raise ValueError(f"lambda_grid: corner ra {ra_c:.4g} leaves no room below lam_max={lam_max}")
    ra = np.geomspace(ra_lo, ra_hi, m)
    return ra / (2.0 + ra)


def orlib_lambda_grid(
    k: int,
    m: int = 25,
    rule: str = "corner",
    *,
    data_dir: Path | str | None = None,
) -> np.ndarray:
    """λ grid for ``port<k>``: ``rule="corner"`` (default, see :func:`lambda_grid`) or
    ``rule="uniform"`` (``j/(m - 1)``, endpoints included)."""
    _check_port(k)
    if rule == "uniform":
        return np.linspace(0.0, 1.0, m)
    if rule != "corner":
        raise ValueError(f"rule must be 'corner' or 'uniform', got {rule!r}")
    inst = load_orlib(k, data_dir)
    return lambda_grid(inst.mu, inst.sigma, m)


def orlib_problem(
    k: int,
    K: int | None,
    lam: float,
    w_min: float = 0.01,
    w_max: float = 1.0,
    *,
    data_dir: Path | str | None = None,
) -> PortfolioProblem:
    """Chang et al. (2000) CCEF instance ``port<k>`` as a :class:`PortfolioProblem`.

    ``K`` — at most K held assets (``None`` — no cardinality); ``lam`` — Chang's trade-off λ ∈
    [0, 1]; ``w_min``/``w_max`` — ε and δ of the paper (0.01, 1.0). The risk aversion is
    ``2λ/(1 - λ)``; at ``λ = 1`` the problem has ``μ = 0, ra = 2`` (objective ``wᵀΣw``).
    See the module docstring for the ``≤ K`` and ``w_min`` caveats.
    """
    _check_port(k)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam!r}")
    inst = load_orlib(k, data_dir)
    if lam == 1.0:
        mu, ra = np.zeros(inst.n), 2.0
    else:
        mu, ra = inst.mu, risk_aversion_from_lambda(lam)
    return PortfolioProblem(
        mu=mu,
        sigma=inst.sigma,
        asset_names=tuple(f"port{k}_{i + 1}" for i in range(inst.n)),
        w_min=w_min,
        w_max=w_max,
        budget=1.0,
        risk_aversion=ra,
        cardinality=K,
    )
