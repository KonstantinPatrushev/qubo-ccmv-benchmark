"""Native PySCIPOpt MIQP baseline with a perspective reformulation (R5).

The legacy baseline (:mod:`cif.classical.milp`) reaches SCIP through cvxpy:
a conic rewrite of ``quad_form``, plain ``w ≤ w_max·z`` linking, model
compilation inside the timer, and neither dual bound nor node count in the
output. Here the model is built directly in SCIP and the result carries what
a certified comparison needs (R9, R10): primal and dual bound, gap, nodes,
and build and solve times measured separately.

Model (minimisation; ``l = w_min``, ``u = min(w_max, budget)``)::

    minimize   - μᵀw + (λ/2)·q  [+ τ·Σ t_i]
    subject to Σ w = budget
               l·z_i ≤ w_i ≤ u·z_i,   z_i ∈ {0, 1}
               Σ z ≤ K                          (card_mode="le"; "eq": Σ z = K)
               Σ_{i∈T} z_i ≤ |T| − 1            (for every T in forbidden_supersets)
               risk epigraph on q (below)
               t_i ≥ ±(w_i - w_prev,i)          (only with a turnover penalty)
               a·w ≥ b                          (only with ``linear_ge``: one row, divided by max|a|)

``linear_ge=(a, b)`` is the minimum-return row of Bertsimas & Cory-Wright (2022, Table 4: ``a = μ`` of the
data, ``b = r̄``, the return term out of the objective). It is one optional row and nothing else of the
model changes without it. Like the risk row it is divided by the scale of its coefficients (``max|a|``,
see below), so that SCIP's absolute tolerances act relative to that scale; ``solver_meta`` reports
``linear_ge_rhs``, ``linear_ge_scale``, and with a solution ``linear_ge_slack`` and ``linear_ok``.

``card_mode="eq"`` is the OR-Library protocol of Chang et al. (ANALYSIS_PLAN §3, N11): exactly
K assets are held, each with at least ``l = ε``; the shared refinement
(:func:`cif.classical.refine.refine_problem` with ``card_mode="eq"``) is consistent with it.

``forbidden_supersets`` serves the k best supports of RQ3 (ANALYSIS_PLAN §2 (г), see
:mod:`cif.backtest.kbest`): the cut for ``T`` removes every selection ``z`` that contains ``T`` and
nothing else. It is not a no-good cut on ``z``: with ``l = 0`` the same portfolio comes back under
another ``z`` (``z_i = 1`` at ``w_i = 0`` is allowed), on a non-binding instance as
``C(N − |supp|, K − |supp|)`` copies of the optimum. A cut on supersets of the *support* removes
them all.

SCIP accepts only a linear objective, hence the epigraph variable ``q``:

* ``"bigm"``: ``q ≥ wᵀΣw``;
* ``"perspective"``: split ``Σ = R + diag(d)`` with ``d ≥ 0``, ``R ⪰ 0``;
  then ``q ≥ wᵀRw + Σ d_i p_i`` and ``w_i² ≤ p_i·z_i`` (rotated cone,
  ``p_i ≥ 0``). At ``z_i = 1`` the cone gives ``p_i ≥ w_i²``, at ``z_i = 0``
  it forces ``w_i = 0``. In the continuous relaxation ``p_i ≥ w_i²/z_i`` is the
  perspective of ``w_i²``, which is tighter than big-M (Frangioni & Gentile
  2006; Aktürk, Atamtürk & Gürel 2009; Günlük & Linderoth 2010).

The diagonal ``d`` is either ``"mineig"`` (``0.999·max(λ_min(Σ), 0)`` for
every asset) or ``"sdp"`` (``max Σ d_i`` s.t. ``Σ - diag(d) ⪰ 1e-7·λ_max(Σ)·I``, ``d ≥ 0``, after
Frangioni & Gentile 2007, solved with cvxpy on ``Σ/λ_max(Σ)`` so that the split does not depend on
the units of Σ; see :func:`_sdp_diagonal`). If the SDP fails, ``"mineig"`` is used and
``solver_meta["diag_fallback"]`` says why. A failure includes a Rust panic of Clarabel
(``pyo3_runtime.PanicException``, a ``BaseException``, seen on a rank-deficient Σ): it is caught
like any other error, ``KeyboardInterrupt`` and ``SystemExit`` are not. Above ``_SDP_MAX_N`` = 150
assets the SDP is not started (Clarabel's dense PSD block; a failed allocation aborts the process,
which nothing can catch) and ``"mineig"`` is used with that reason.

Numerics. SCIP's feasibility tolerance is absolute. With the default 1e-6 the
epigraph undercuts ``wᵀΣw`` by up to 1e-6: on annualised synthetic data the
returned weights were up to ~2e-6 (relative) worse than the enumerated
optimum, and the reported primal bound lay *below* the achievable optimum.
Therefore the risk constraint is divided by ``mean(diag Σ)`` (tolerance
relative to the variance scale) and ``numerics/feastol`` defaults to 1e-8. The ``linear_ge`` row is divided by
``max|a|`` for the same reason: with ``a = μ`` of weekly returns (about 1e-3) an absolute 1e-8 would be a
relative 1e-5 of the row.

SCIP solves sequentially; ``threads`` must be 1 and is recorded. The Ipopt
used by SCIP's NLP heuristics is linked into the wheel's libscip without
OpenMP.

Anytime output (ANALYSIS_PLAN §5, N5). ``solver_meta["trajectory"]`` is the list of ``(t, f)`` of
every new best solution SCIP stores: ``t`` seconds of wall clock on the common clock (the model
build counts, and so does ``clock_offset_s``, see below), ``f`` the true objective of the extracted
weights (the same evaluation as the returned ``Solution.objective``, so the last entry is the
returned portfolio). ``trajectory_primal_bound`` holds SCIP's own objective of the same solutions. A
warm start is the first incumbent but not an entry: only solutions found by the search are listed.

End-to-end budget (§5 "Время end-to-end", §6). The RQ2 protocol stops every method by one
wall-clock budget counted from the start of the common clock, as the QUBO loop and the local search
(:mod:`cif.heuristics`) do with their ``budget_s``. Here that is ``budget_s``: the diagonal / SDP,
the model, the warm start and SCIP's solve are inside it, and SCIP's ``limits/time`` is what is left
(wall clock, ``timing/clocktype`` 2) after the build. ``clock_offset_s`` is the part of the clock
that was spent before the call, the construction of the common start (``CommonStart.time_s``): it is
charged to the budget and added to the trajectory times and to ``Solution.wall_time_s``, so that all
methods of an instance share one clock and :func:`cif.metrics.anytime.primal_integral` takes the
trajectories as they are. ``time_limit_s`` is the older, separate limit on the solve alone
(certification runs: E1, the oracle); with both, the smaller remaining time binds. ``solver_meta``
reports ``build_time_s`` (diagonal, model, warm start), ``diag_time_s`` (its part spent in the
diagonal, the SDP being by far the largest), ``solve_time_s`` and ``solve_limit_s`` (the limit
given to SCIP).

SCIP does not look at the clock everywhere. On a dense N = 400 model (factor instance, K = 10) its
symmetry detection in presolve ran 17.5 s ("no symmetry present") and the call ended 6.4 and 13.7 s
after a 10 s budget in two runs; with ``scip_params={"misc/usesymmetry": 0}`` it ended 0.03 s after
(at N = 100 and 200 the runs converged inside the budget, so nothing is known there). The study uses
SCIP's defaults (§4), so the switch is left to the caller, as a recorded deviation
(``solver_meta["scip_params"]``), and not made here.

Seeds (§6). ``seed`` is ``randomization/randomseedshift``; ``permutation_seed`` additionally
permutes the model (``randomization/permutevars`` and ``randomization/permutationseed``). The study
uses at least three seeds with all of them set.

Certification (§5, audit G-1): ``abs_gap = 1e-6·s_ref`` is SCIP's ``limits/absgap``
(:func:`cif.metrics.gap.reference_scale`), and the reference is certified when
``|primal − dual| ≤ 1e-6·s_ref`` (:func:`cif.metrics.gap.reference_from_exact`). SCIP stops at
whichever of ``mip_gap`` and ``abs_gap`` is met first; a stop at ``mip_gap ≤ 1e-6`` meets the
absolute criterion too (``s_ref ≥ |f|``), so the two limits do not conflict.
"""

from __future__ import annotations

import math
import operator
import time
from collections.abc import Mapping, Sequence
from typing import Literal

import cvxpy as cp
import numpy as np
from pyscipopt import SCIP_EVENTTYPE, SCIP_PARAMSETTING, Eventhdlr, Model, quicksum
from pyscipopt.scip import Expr, Term

from cif.problem import PortfolioProblem, Solution

Formulation = Literal["bigm", "perspective"]
DiagMethod = Literal["mineig", "sdp"]
CardMode = Literal["le", "eq"]

_FORMULATIONS = ("bigm", "perspective")
_DIAG_METHODS = ("mineig", "sdp")
_CARD_MODES = ("le", "eq")
_MINEIG_FACTOR = 0.999
_SDP_MARGIN = 1e-7  # relative to λ_max(Σ), see _sdp_diagonal
_SDP_SOLVERS = ("CLARABEL", "SCS")
# the wall-clock limit of each SDP solver, as a keyword of ``cp.Problem.solve``
_SDP_TIME_OPTION = {"CLARABEL": "time_limit", "SCS": "time_limit_secs"}
# One thread for every solver of the study (ANALYSIS_PLAN §6), the SDP being inside the timed budget.
# Clarabel's default is automatic: its PSD cone ran on two threads whatever OMP_NUM_THREADS says.
_SDP_THREAD_OPTIONS = {"CLARABEL": {"max_threads": 1}}
# Clarabel holds a dense block of (N(N+1)/2)² doubles. Measured on factor instances, one thread:
# peak 1.4 GB and 23-30 s at N = 100, 6.5 GB at N = 150 (30 s even with a 5 s limit: the setup is
# not interruptible), a failed 1.6 GB allocation after 8 GB at N = 200 and a failed 51 GB one at
# N = 400. A failed allocation aborts the whole process, which no ``except`` catches, so above this
# size the SDP is a recorded fallback to "mineig" and not a crash.
_SDP_MAX_N = 150
# SCIP parameters that ``solve_miqp_native`` sets from arguments of its own: not for ``scip_params``
_MANAGED_SCIP_PARAMS = frozenset(
    {
        "limits/time", "limits/gap", "limits/absgap", "numerics/feastol", "lp/threads",
        "timing/clocktype", "randomization/randomseedshift", "randomization/permutevars",
        "randomization/permutationseed",
    }
)  # fmt: skip
_D_REL_TOL = 1e-10  # d_i ≤ this · max(diag Σ) is treated as exactly 0
_ZERO_TOL = 1e-9  # weights below this are reported as exact zeros (as in milp.py)
_NONZERO_TOL = 1e-6  # support threshold for n_nonzero and the cardinality check
_LINEAR_TOL = 1e-6  # ``linear_ok``: the row may be missed by this much of its scale (as the box and budget checks)


class _RootBoundRecorder(Eventhdlr):
    """Record the global dual bound each time a root node (depth 0) is solved.

    ``Model.getDualboundRoot`` is unreliable after the solve: once the root
    node is cut off SCIP resets its bound to infinity, even when the tree had
    several nodes. A restart starts a new root; the last completed one wins.
    """

    def __init__(self) -> None:
        self.root_dual_bound: float | None = None

    def eventinit(self) -> None:
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self) -> None:
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexec(self, event) -> None:
        if event.getNode().getDepth() == 0:
            self.root_dual_bound = float(self.model.getDualbound())


def _release(model: Model) -> None:
    """Free the transformed problem once everything has been read from the model.

    PySCIPOpt 6.1: ``Model.catchEvent`` increfs the model and only ``dropEvent`` decrefs it; the handlers
    above call ``dropEvent`` in ``eventexit``, which SCIP runs when the transformed problem is freed.
    A call that left the transformed problem alone kept its model (≈ 13 MB of SCIP at N = 49) alive until
    the process ended: 800 MB for the 33 MIQPs of one E3 cell."""
    model.freeTransform()


def _solution_weights(sol, w: list, z: list) -> np.ndarray:
    """Weights of a SCIP solution: the binaries define the support (``w_i ≤ u·z_i`` leaves at most
    ``u·feastol`` off it), values below ``_ZERO_TOL`` are exact zeros."""
    weights = np.array([sol[v] for v in w], dtype=float)
    selected = np.array([sol[v] for v in z], dtype=float) > 0.5
    return np.where(selected & (weights >= _ZERO_TOL), weights, 0.0)


class _IncumbentRecorder(Eventhdlr):
    """Record ``(t, f)`` each time SCIP stores a new best solution (the anytime trajectory).

    ``t`` is the wall clock since ``t_origin`` (the origin of the common clock: the start of the
    call, moved back by the time already spent on it), ``f`` the true objective of the extracted
    weights. The event fires *before* SCIP updates its primal bound, so
    ``getPrimalbound`` would still show the previous solution: the bound is read from the new best
    solution itself and kept in ``primal_bounds`` (SCIP's objective, in model units).
    """

    def __init__(self, problem: PortfolioProblem, w: list, z: list, t_origin: float) -> None:
        self.problem = problem
        self.w = w
        self.z = z
        self.t_origin = t_origin
        self.trajectory: list[tuple[float, float]] = []
        self.primal_bounds: list[float] = []

    def eventinit(self) -> None:
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self) -> None:
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event) -> None:
        t = time.perf_counter() - self.t_origin
        best = self.model.getBestSol()
        weights = _solution_weights(best, self.w, self.z)
        self.trajectory.append((t, self.problem.objective_value(weights)))
        self.primal_bounds.append(_scip_value(self.model, self.model.getSolObjVal(best)))


def _sdp_diagonal(
    sigma: np.ndarray, d_tol: float, time_limit_s: float | None = None
) -> tuple[np.ndarray | None, str | None]:
    """``max Σ d_i`` s.t. ``Σ - diag(d) ⪰ ρ·s·I``, ``d ≥ 0``, with ``s = λ_max(Σ)`` and
    ``ρ = _SDP_MARGIN``.

    The margin is relative to the spectral norm and the problem is posed on ``Σ/s``, so the
    solver sees the same numbers in annual, daily or percent² units and the split is equivariant:
    ``Σ → cΣ`` gives ``d → c·d``. An absolute margin (it was 1e-8) is not equivariant:

    * the interior-point answer misses the constraint by an amount that grows with the data (up to
      8e-8 of ``mean(diag Σ)`` for N = 20..100 on factor instances), so from ``mean(diag Σ) ≈ 1``
      up the acceptance check below rejected it and the "sdp" variant was silently "mineig"
      (annualised data, ``mean(diag Σ) ≈ 0.1``, sat just under that edge);
    * for ``mean(diag Σ) ≈ 1e-7`` the margin itself took a fifth of Σd.

    Why ``λ_max`` and not ``mean(diag Σ)``, the scale of the rest of the module: on
    ``Σ/mean(diag Σ)`` Clarabel stalled in "optimal_inaccurate" from N = 20, on ``Σ/λ_max`` it
    reports "optimal" up to N = 50 (mixed at N = 100) and misses ρ by at most 2 % up to N = 100,
    against the 50 % at which the acceptance check below rejects it. ρ = 1e-7 costs 1e-6..7e-6 of Σd.

    Returns ``(d, None)`` on success or ``(None, reason)``. A solution is accepted only if
    ``Σ - diag(d)`` passes a numpy eigenvalue check (``λ_min ≥ ρ·s/2``), so an inaccurate solver
    answer never makes the risk constraint nonconvex.

    ``time_limit_s`` (``None``: no limit) is the wall-clock time the SDP may take in all. Each
    solver gets what is left of it as its own limit (``user_limit`` status when it stops there, then
    reported like any other failure) and none is started once the time is gone. The solvers check
    the clock once per iteration, which at N = 100 takes about a second with Clarabel, and cvxpy's
    compilation of the problem is outside their clock: the limit is approximate. Clarabel runs on
    one thread (``max_threads=1``), as every solver of the study. Above ``_SDP_MAX_N`` assets the
    SDP is not started (``(None, reason)``): see the constant.
    """
    t_start = time.perf_counter()
    n = sigma.shape[0]
    if n > _SDP_MAX_N:
        return None, f"N = {n} > {_SDP_MAX_N}: Clarabel's dense PSD block would not fit in memory"
    scale = float(np.linalg.eigvalsh(sigma)[-1])
    if not scale > 0.0:
        scale = 1.0
    d = cp.Variable(n)
    sdp = cp.Problem(
        cp.Maximize(cp.sum(d)),
        [sigma / scale - cp.diag(d) >> _SDP_MARGIN * np.eye(n), d >= 0],
    )
    reasons: list[str] = []
    for solver in _SDP_SOLVERS:
        options: dict[str, float] = dict(_SDP_THREAD_OPTIONS.get(solver, {}))
        if time_limit_s is not None:
            left = time_limit_s - (time.perf_counter() - t_start)
            if left <= 0.0:
                reasons.append(f"{solver}: not started, the time budget is spent")
                continue
            options[_SDP_TIME_OPTION[solver]] = left
        try:
            sdp.solve(solver=solver, **options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            # Solver failures are reported, not raised. Clarabel panics (Rust, e.g. in its PSD cone
            # on a rank-deficient Σ); pyo3 raises that as ``PanicException``, a ``BaseException``.
            reasons.append(f"{solver}: {type(exc).__name__}: {exc}")
            continue
        if sdp.status not in {"optimal", "optimal_inaccurate"} or d.value is None:
            reasons.append(f"{solver}: status {sdp.status}")
            continue
        cand = scale * np.clip(np.asarray(d.value, dtype=float), 0.0, None)
        cand[cand <= d_tol] = 0.0
        eig_min = float(np.linalg.eigvalsh(sigma - np.diag(cand))[0])
        floor = 0.5 * _SDP_MARGIN * scale
        if eig_min < floor:
            reasons.append(f"{solver}: λ_min(Σ - diag(d)) = {eig_min:.2e} < {floor:.0e}")
            continue
        return cand, None
    return None, "; ".join(reasons)


def _diagonal_shift(
    sigma: np.ndarray, method: str, *, time_limit_s: float | None = None
) -> tuple[np.ndarray, str, str | None]:
    """Split ``Σ = R + diag(d)`` with ``d ≥ 0`` and ``R ⪰ 0``.

    Returns ``(d, method_used, fallback_reason)``; the reason is ``None``
    unless ``"sdp"`` was requested and ``"mineig"`` had to be used instead.
    ``time_limit_s`` is the time the SDP may take (:func:`_sdp_diagonal`); ``"mineig"`` is a
    single eigenvalue and ignores it.
    """
    if method not in _DIAG_METHODS:
        raise ValueError(f"diag must be one of {_DIAG_METHODS}, got {method!r}")
    sigma = 0.5 * (sigma + sigma.T)
    n = sigma.shape[0]
    d_tol = _D_REL_TOL * max(float(np.max(np.diag(sigma))), 0.0)
    fallback: str | None = None
    if method == "sdp":
        d_sdp, fallback = _sdp_diagonal(sigma, d_tol, time_limit_s)
        if d_sdp is not None:
            return d_sdp, "sdp", None
    lam_min = float(np.linalg.eigvalsh(sigma)[0])
    d = np.full(n, _MINEIG_FACTOR * max(lam_min, 0.0))
    d[d <= d_tol] = 0.0
    return d, "mineig", fallback


def _quadratic_expr(w: list, matrix: np.ndarray) -> Expr:
    """``wᵀMw`` as a SCIP expression (upper triangle, off-diagonals doubled).

    Built from a term dictionary: about 5x faster than ``quicksum`` at N = 150.
    """
    iu, ju = np.triu_indices(matrix.shape[0])
    coef = np.where(iu == ju, 1.0, 2.0) * matrix[iu, ju]
    keep = coef != 0.0
    return Expr(
        {
            Term(w[i], w[j]): c
            for i, j, c in zip(
                iu[keep].tolist(), ju[keep].tolist(), coef[keep].tolist(), strict=True
            )
        }
    )


def _scip_value(model: Model, x: float) -> float:
    """SCIP reports infinity as ±1e20; return a true ``±inf`` instead."""
    x = float(x)
    return float(np.copysign(np.inf, x)) if model.isInfinity(abs(x)) else x


def _as_vector(x: np.ndarray, n: int, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    if v.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {v.shape}")
    return v


def _supersets_to_forbid(sets: Sequence[Sequence[int]], n: int) -> list[tuple[int, ...]]:
    """Validated ``forbidden_supersets``: sorted tuples of distinct integer indices in ``[0, n)``.

    An empty set is rejected: every selection contains it, so its cut ``0 ≤ −1`` would make the
    model infeasible by a mistake rather than by design.
    """
    out: list[tuple[int, ...]] = []
    for k, raw in enumerate(sets):
        try:
            members = [operator.index(i) for i in raw]
        except TypeError:
            raise ValueError(
                f"forbidden_supersets[{k}] must hold integer asset indices, got {raw!r}"
            ) from None
        if not members:
            raise ValueError(f"forbidden_supersets[{k}] is empty: every selection contains it")
        if min(members) < 0 or max(members) >= n:
            raise ValueError(
                f"forbidden_supersets[{k}] = {tuple(members)} has an index outside [0, {n})"
            )
        if len(set(members)) != len(members):
            raise ValueError(f"forbidden_supersets[{k}] = {tuple(members)} repeats an index")
        out.append(tuple(sorted(members)))
    return out


def _linear_row(linear_ge, n: int) -> tuple[np.ndarray, float, float] | None:
    """Validated ``linear_ge``: ``(a, b, scale)`` with ``scale = max|a|``, the divisor of the row (module
    docstring), or ``None`` for no row. ``a`` must be a finite vector of length ``n`` with a non-zero entry
    (an all-zero row has no scale and is ``0 ≥ b``, a mistake), ``b`` a finite number."""
    if linear_ge is None:
        return None
    try:
        a_raw, b_raw = linear_ge
    except (TypeError, ValueError):
        raise ValueError(f"linear_ge must be a pair (a, b), got {linear_ge!r}") from None
    a = _as_vector(a_raw, n, "linear_ge[0]")
    try:
        b = float(b_raw)
    except (TypeError, ValueError):
        raise ValueError(f"linear_ge[1] must be a number, got {b_raw!r}") from None
    if not np.all(np.isfinite(a)) or not math.isfinite(b):
        raise ValueError("linear_ge must hold finite numbers: the coefficients a and the right-hand side b")
    scale = float(np.max(np.abs(a)))
    if not scale > 0.0:
        raise ValueError("linear_ge[0] is all zeros: the row would read 0 ≥ b")
    return a, b, scale


def _check_budget(budget_s: float | None, clock_offset_s: float) -> None:
    """``ValueError`` unless ``budget_s`` is ``None`` or a finite number ≥ 0 and ``clock_offset_s``
    a finite number ≥ 0. Shared with :func:`cif.classical.oa.solve_oa`."""
    if budget_s is not None and not (math.isfinite(budget_s) and budget_s >= 0.0):
        raise ValueError(f"budget_s must be a finite non-negative number, got {budget_s}")
    if not (math.isfinite(clock_offset_s) and clock_offset_s >= 0.0):
        raise ValueError(
            f"clock_offset_s must be a finite non-negative number, got {clock_offset_s}"
        )


def _budget_left(budget_s: float | None, clock_offset_s: float, elapsed_s: float) -> float | None:
    """Seconds of the end-to-end budget left after ``elapsed_s`` of this call, which began at
    ``clock_offset_s`` on the common clock: negative once the budget is overdrawn, ``None`` without
    one."""
    return None if budget_s is None else budget_s - clock_offset_s - elapsed_s


def _solve_limit(time_limit_s: float | None, left_s: float | None) -> float | None:
    """The limit on the solve phase: the smaller of ``time_limit_s`` (the solve alone) and what is
    left of the end-to-end budget (not below 0); ``None`` if neither is given."""
    candidates = (time_limit_s, None if left_s is None else max(left_s, 0.0))
    limits = [x for x in candidates if x is not None]
    return min(limits) if limits else None


def solve_miqp_native(
    problem: PortfolioProblem,
    cardinality: int | None = None,
    *,
    formulation: Formulation = "perspective",
    diag: DiagMethod = "mineig",
    card_mode: CardMode = "le",
    time_limit_s: float | None = None,
    budget_s: float | None = None,
    clock_offset_s: float = 0.0,
    threads: int = 1,
    mip_gap: float = 1e-6,
    abs_gap: float | None = None,
    feastol: float = 1e-8,
    warm_start: np.ndarray | None = None,
    prev_weights: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
    seed: int = 0,
    permutation_seed: int | None = None,
    forbidden_supersets: Sequence[Sequence[int]] = (),
    linear_ge: tuple[np.ndarray, float] | None = None,
    scip_params: Mapping[str, object] | None = None,
    heuristics: str = "default",
    verbose: bool = False,
) -> Solution:
    """Solve the cardinality-constrained MIQP with a native SCIP model.

    Parameters
    ----------
    cardinality:
        If not ``None``, use this value. Otherwise falls back to
        ``problem.cardinality`` (``None`` — no cardinality constraint).
    formulation:
        ``"perspective"`` (default) or ``"bigm"``; see the module docstring.
    diag:
        Diagonal extraction for ``"perspective"``: ``"mineig"`` or ``"sdp"``.
        Ignored for ``"bigm"``. Its time is part of ``build_time_s``.
    card_mode:
        ``"le"`` (default): ``Σz ≤ K``. ``"eq"``: ``Σz = K``, every selected asset holds at
        least ``w_min`` (OR-Library, ANALYSIS_PLAN §3 N11); requires a cardinality. A warm start
        then needs exactly K assets with positive weight, otherwise SCIP rejects it. With
        ``w_min = 0`` an asset with ``z = 1`` may weigh 0 and the two modes have one optimum.
    time_limit_s:
        SCIP ``limits/time``; ``None`` — no limit. Counts solving only: the build (diagonal / SDP,
        model, warm start) is not in it. The limit of certification runs; the RQ2 protocol uses
        ``budget_s``.
    budget_s:
        End-to-end budget in seconds of the common clock (module docstring); ``None`` — none. The
        SDP of ``diag="sdp"`` gets what is left at its start as its time limit, SCIP gets what is
        left after the build as ``limits/time`` (0 when nothing is: SCIP then returns the warm
        start, if there is one), and with ``time_limit_s`` too the smaller of the two binds. A
        stopped component can overrun by its granularity (an SDP iteration, a SCIP presolve / LP /
        heuristic step), the build cannot be interrupted: ``Solution.wall_time_s`` shows the call's
        end on the clock.
    clock_offset_s:
        Seconds of the common clock spent before the call, the construction of the common start
        (``CommonStart.time_s``). Charged to ``budget_s``, added to the times of ``trajectory`` and
        to ``Solution.wall_time_s``; it changes nothing else. ``build_time_s`` and
        ``solve_time_s`` stay durations of this call.
    threads:
        Must be 1: SCIP solves sequentially. Kept explicit for the registry.
    mip_gap:
        SCIP ``limits/gap``. SCIP then stops with status ``"gaplimit"`` once
        the relative gap is below it; ``"optimal"`` means the tree was
        exhausted. Both are certified to within ``mip_gap``.
    abs_gap:
        SCIP ``limits/absgap`` (``None``: SCIP's default 0, the parameter is not touched). The
        study certifies at ``1e-6·s_ref`` (module docstring); the status is ``"gaplimit"`` again.
        ``solver_meta["abs_gap"]`` is the achieved ``|primal − dual|``.
    feastol:
        SCIP ``numerics/feastol``. Default 1e-8 rather than SCIP's 1e-6, see
        the module docstring.
    warm_start:
        Weight vector handed to SCIP as an initial solution, completed with
        all auxiliary variables (``z``, ``q``, ``p``, ``t``). Its feasibility
        is checked by SCIP; ``solver_meta["warm_start_used"]`` says whether it
        was accepted. The common start of ANALYSIS_PLAN §5 goes here (``CommonStart.weights``),
        with ``clock_offset_s=CommonStart.time_s``.
    prev_weights, turnover_penalty:
        With ``turnover_penalty > 0`` and ``prev_weights`` given, the
        objective gains ``turnover_penalty·Σ|w_i - prev_i|`` (R3). Without
        ``prev_weights`` the start is cash: the term would equal
        ``turnover_penalty·budget`` for every feasible ``w`` and is omitted.
    seed:
        SCIP ``randomization/randomseedshift``.
    permutation_seed:
        If not ``None`` (≥ 0): ``randomization/permutevars = True`` and
        ``randomization/permutationseed = permutation_seed``, which permutes the order of the
        variables and constraints of the model (SCIP's other source of seed-to-seed variation).
        ``None`` (default) leaves both parameters at SCIP's defaults: no permutation.
    forbidden_supersets:
        Sets ``T`` of asset indices (distinct, non-empty) whose supersets are not allowed: the
        model gets ``Σ_{i∈T} z_i ≤ |T| − 1`` for each (module docstring). A selection that holds all
        of ``T`` is cut off, any selection missing one asset of ``T`` stays. Works for both
        formulations and both ``card_mode`` values; a warm start that holds a whole ``T`` is
        rejected by SCIP. ``solver_meta["n_forbidden_supersets"]`` is their number; the default
        ``()`` leaves the model unchanged.
    linear_ge:
        ``(a, b)``: one extra row ``a·w ≥ b`` (module docstring: the minimum-return constraint of
        Bertsimas & Cory-Wright 2022, Table 4, with ``a = μ`` and ``b = r̄``). It enters the model divided by
        ``max|a|``, as the risk row is divided by ``mean(diag Σ)``, so that SCIP's absolute tolerances act
        relative to its scale. ``solver_meta`` then has ``linear_ge_rhs`` (``b``), ``linear_ge_scale``
        (``max|a|``) and, with a solution, ``linear_ge_slack`` (``a·w − b`` of the returned weights) and
        ``linear_ok`` (slack ≥ −1e-6·``max|a|``), which ``Solution.feasible`` includes. A warm start has to
        satisfy the row, or SCIP rejects it. ``None`` (default): no row, the model and ``solver_meta`` are
        those of a call without the argument.
    scip_params:
        Further SCIP parameters (name → value), set after the ones above and before the solve, and
        recorded in ``solver_meta["scip_params"]``. The study uses SCIP's defaults (ANALYSIS_PLAN §4),
        so every entry is a deviation to be decided and logged (§12). Parameters that have an
        argument of their own are rejected, an unknown name raises ``ValueError``. The case for
        it: ``{"misc/usesymmetry": 0}``, because SCIP's symmetry detection in presolve does not look
        at the clock (module docstring).
    heuristics:
        ``"default"`` leaves SCIP's primal heuristics as they are; ``"aggressive"`` applies SCIP's meta
        setting ``setHeuristics(AGGRESSIVE)`` before the parameters above (a sensitivity of the study, not a
        default: ANALYSIS_PLAN §12, 2026-10-19). Recorded in ``solver_meta["heuristics"]``.

    Returns
    -------
    Solution
        ``objective`` is ``problem.objective_value(w)`` (no turnover term);
        ``wall_time_s`` is the end of the call (build + solve + extraction) on the common clock,
        ``clock_offset_s`` included.
        Without an incumbent: ``feasible=False``, ``objective=inf`` and zero
        weights. ``solver_meta`` holds the SCIP status and bounds (SCIP's
        infinity mapped to ``±inf``), ``root_dual_bound`` (global dual bound
        when a root node was last completed, or at the stop if the solve
        ended inside the root), ``abs_gap`` and ``abs_gap_limit``, node count, build and solve
        times (wall and CPU; ``diag_time_s`` is the part of the build spent in the diagonal and
        ``solve_limit_s`` the limit SCIP was given), the anytime ``trajectory`` and
        ``trajectory_primal_bound`` (module docstring) and the inputs that define the run
        (``time_limit_s``, ``budget_s``, ``clock_offset_s``, ...).
    """
    t_start = time.perf_counter()
    cpu_start = time.process_time()
    n = problem.n
    if formulation not in _FORMULATIONS:
        raise ValueError(f"formulation must be one of {_FORMULATIONS}, got {formulation!r}")
    if diag not in _DIAG_METHODS:
        raise ValueError(f"diag must be one of {_DIAG_METHODS}, got {diag!r}")
    if threads != 1:
        raise ValueError(f"SCIP solves sequentially: threads must be 1, got {threads}")
    if card_mode not in _CARD_MODES:
        raise ValueError(f"card_mode must be one of {_CARD_MODES}, got {card_mode!r}")
    if mip_gap < 0.0:
        raise ValueError(f"mip_gap must be non-negative, got {mip_gap}")
    if abs_gap is not None and not (math.isfinite(abs_gap) and abs_gap >= 0.0):
        raise ValueError(f"abs_gap must be a finite non-negative number, got {abs_gap}")
    if turnover_penalty < 0.0:
        raise ValueError(f"turnover_penalty must be non-negative, got {turnover_penalty}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if permutation_seed is not None and permutation_seed < 0:
        raise ValueError(f"permutation_seed must be non-negative, got {permutation_seed}")
    _check_budget(budget_s, clock_offset_s)
    if heuristics not in ("default", "aggressive"):
        raise ValueError(f"heuristics must be 'default' or 'aggressive', got {heuristics!r}")
    extra_params = dict(scip_params) if scip_params is not None else {}
    for name in extra_params:
        if name in _MANAGED_SCIP_PARAMS:
            raise ValueError(f"scip_params cannot set {name!r}: it follows an argument of its own")
    effective_card = cardinality if cardinality is not None else problem.cardinality
    if effective_card is not None and effective_card < 1:
        raise ValueError(f"cardinality must be ≥ 1, got {effective_card}")
    if card_mode == "eq" and effective_card is None:
        raise ValueError("card_mode='eq' needs a cardinality (argument or problem.cardinality)")
    cuts = _supersets_to_forbid(forbidden_supersets, n)
    linear = _linear_row(linear_ge, n)
    ws = _as_vector(warm_start, n, "warm_start") if warm_start is not None else None
    prev = _as_vector(prev_weights, n, "prev_weights") if prev_weights is not None else None
    use_turnover = turnover_penalty > 0.0 and prev is not None

    lower = float(problem.w_min)
    upper = float(min(problem.w_max, problem.budget))
    sigma = 0.5 * (problem.sigma + problem.sigma.T)
    # Divide the risk constraint by the variance scale: SCIP's absolute
    # tolerance on q then acts relative to wᵀΣw.
    risk_scale = float(np.mean(np.diag(sigma)))
    if not risk_scale > 0.0:
        risk_scale = 1.0

    d: np.ndarray | None = None
    diag_used: str | None = None
    diag_fallback: str | None = None
    diag_time = 0.0
    if formulation == "perspective":
        t_diag = time.perf_counter()
        d, diag_used, diag_fallback = _diagonal_shift(
            sigma, diag, time_limit_s=_budget_left(budget_s, clock_offset_s, t_diag - t_start)
        )
        diag_time = time.perf_counter() - t_diag

    model = Model("cif_miqp_native")
    model.hideOutput(not verbose)
    root_recorder = _RootBoundRecorder()
    model.includeEventhdlr(root_recorder, "cif_root_bound", "records the root dual bound")
    w = [model.addVar(name=f"w_{i}", vtype="C", lb=0.0, ub=upper) for i in range(n)]
    z = [model.addVar(name=f"z_{i}", vtype="B") for i in range(n)]
    q = model.addVar(name="q", vtype="C", lb=0.0, ub=None)
    incumbents = _IncumbentRecorder(problem, w, z, t_start - clock_offset_s)
    model.includeEventhdlr(incumbents, "cif_incumbents", "records the anytime trajectory")

    model.addCons(quicksum(w) == problem.budget, name="budget")
    for i in range(n):
        model.addCons(w[i] - upper * z[i] <= 0.0, name=f"link_ub_{i}")
        if lower > 0.0:
            model.addCons(w[i] - lower * z[i] >= 0.0, name=f"link_lb_{i}")
    if effective_card is not None:
        if card_mode == "eq":
            model.addCons(quicksum(z) == effective_card, name="cardinality")
        else:
            model.addCons(quicksum(z) <= effective_card, name="cardinality")
    for k, cut in enumerate(cuts):
        model.addCons(quicksum(z[i] for i in cut) <= len(cut) - 1, name=f"no_superset_{k}")
    if linear is not None:
        a_row, b_row, scale_row = linear
        model.addCons(
            quicksum(float(a_row[i] / scale_row) * w[i] for i in np.flatnonzero(a_row).tolist())
            >= b_row / scale_row,
            name="linear_ge",
        )

    p: dict[int, object] = {}
    if formulation == "bigm":
        model.addCons(_quadratic_expr(w, sigma / risk_scale) - q <= 0.0, name="risk")
    else:
        assert d is not None
        residual = sigma - np.diag(d)
        for i in np.flatnonzero(d > 0.0).tolist():
            # p_i ≤ u² is valid: the smallest feasible p_i is w_i²/z_i ≤ u²·z_i.
            p[i] = model.addVar(name=f"p_{i}", vtype="C", lb=0.0, ub=upper * upper)
            model.addCons(w[i] * w[i] - p[i] * z[i] <= 0.0, name=f"persp_{i}")
        diag_part = quicksum((d[i] / risk_scale) * p[i] for i in p)
        model.addCons(_quadratic_expr(w, residual / risk_scale) + diag_part - q <= 0.0, name="risk")

    t: list = []
    if use_turnover:
        assert prev is not None
        t = [model.addVar(name=f"t_{i}", vtype="C", lb=0.0, ub=None) for i in range(n)]
        for i in range(n):
            model.addCons(t[i] - w[i] >= -prev[i], name=f"turn_pos_{i}")
            model.addCons(t[i] + w[i] >= prev[i], name=f"turn_neg_{i}")

    objective = (
        -quicksum(float(problem.mu[i]) * w[i] for i in range(n))
        + 0.5 * problem.risk_aversion * risk_scale * q
    )
    if use_turnover:
        objective = objective + turnover_penalty * quicksum(t)
    model.setObjective(objective, "minimize")

    if heuristics == "aggressive":  # a meta setting of the primal heuristics only; the parameters below follow it
        model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
    model.setParam("limits/gap", float(mip_gap))
    model.setParam("numerics/feastol", float(feastol))
    model.setParam("randomization/randomseedshift", int(seed))
    model.setParam("lp/threads", int(threads))
    model.setParam("timing/clocktype", 2)  # wall clock: limits/time and solve_time_s follow it
    if abs_gap is not None:
        model.setParam("limits/absgap", float(abs_gap))
    if permutation_seed is not None:
        model.setParam("randomization/permutevars", True)
        model.setParam("randomization/permutationseed", int(permutation_seed))
    for name, value in extra_params.items():
        try:
            model.setParam(name, value)
        except KeyError as exc:
            raise ValueError(f"unknown SCIP parameter {name!r}") from exc

    warm_start_used = False
    if ws is not None:
        active = ws > _ZERO_TOL
        sol = model.createSol()
        for i in range(n):
            model.setSolVal(sol, w[i], float(ws[i]))
            model.setSolVal(sol, z[i], 1.0 if active[i] else 0.0)
        if formulation == "bigm":
            risk = float(ws @ sigma @ ws)
        else:
            assert d is not None
            p_val = np.where(active, ws * ws, 0.0)
            risk = float(ws @ (sigma - np.diag(d)) @ ws) + float(d @ p_val)
            for i, var in p.items():
                model.setSolVal(sol, var, float(p_val[i]))
        model.setSolVal(sol, q, risk / risk_scale)
        if use_turnover:
            assert prev is not None
            for i in range(n):
                model.setSolVal(sol, t[i], float(abs(ws[i] - prev[i])))
        # addSol in the PROBLEM stage stores without checking, hence checkSol first
        if model.checkSol(sol, printreason=verbose, original=True):
            warm_start_used = bool(model.addSol(sol, free=True))
        else:
            model.freeSol(sol)

    build_time = time.perf_counter() - t_start
    # the limit is set after the build, so that the budget's remainder is what SCIP gets
    solve_limit = _solve_limit(time_limit_s, _budget_left(budget_s, clock_offset_s, build_time))
    if solve_limit is not None:
        model.setParam("limits/time", float(solve_limit))
    cpu_solve_start = time.process_time()
    build_cpu = cpu_solve_start - cpu_start
    model.optimize()
    solve_cpu = time.process_time() - cpu_solve_start
    solve_time = float(model.getSolvingTime())

    status = str(model.getStatus())
    n_sols = int(model.getNSols())
    dual_bound = _scip_value(model, model.getDualbound())
    root_dual_bound = root_recorder.root_dual_bound
    if root_dual_bound is None and model.getNNodes() <= 1:
        # The solve stopped inside the root node (gap or time limit), so the
        # root never completed; its bound so far is the final dual bound.
        root_dual_bound = dual_bound
    if root_dual_bound is not None:
        root_dual_bound = _scip_value(model, root_dual_bound)

    primal_bound = _scip_value(model, model.getPrimalbound())
    meta: dict = {
        "status": status,
        "time_limit_hit": status == "timelimit",
        "primal_bound": primal_bound,
        "dual_bound": dual_bound,
        # None only if the solve ended before the root node was processed
        "root_dual_bound": root_dual_bound,
        "gap": _scip_value(model, model.getGap()),
        # |primal − dual|, the quantity limits/absgap bounds; inf while either bound is missing
        "abs_gap": abs(primal_bound - dual_bound)
        if math.isfinite(primal_bound) and math.isfinite(dual_bound)
        else math.inf,
        "abs_gap_limit": abs_gap,
        "trajectory": incumbents.trajectory,
        "trajectory_primal_bound": incumbents.primal_bounds,
        "n_nodes": int(model.getNTotalNodes()),
        "build_time_s": build_time,
        "diag_time_s": diag_time,  # part of the build
        "solve_time_s": solve_time,
        # process CPU time (all threads); ≈ wall time when single-threaded (R10)
        "build_cpu_s": build_cpu,
        "solve_cpu_s": solve_cpu,
        "formulation": formulation,
        "diag_method": diag_used,
        "diag_fallback": diag_fallback,
        "d_min": float(d.min()) if d is not None and d.size else None,
        "d_max": float(d.max()) if d is not None and d.size else None,
        "warm_start_used": warm_start_used,
        "threads": threads,
        "cardinality": effective_card,
        "card_mode": card_mode,
        "time_limit_s": time_limit_s,
        "budget_s": budget_s,
        "clock_offset_s": clock_offset_s,
        "solve_limit_s": solve_limit,  # limits/time as given to SCIP (None: not set)
        "mip_gap": mip_gap,
        "feastol": feastol,
        "seed": seed,
        "permutation_seed": permutation_seed,
        "n_forbidden_supersets": len(cuts),
        "scip_params": extra_params,
        "heuristics": heuristics,
        "turnover_penalty": turnover_penalty if use_turnover else 0.0,
    }
    if linear is not None:
        meta.update({"linear_ge_rhs": linear[1], "linear_ge_scale": linear[2]})

    if n_sols == 0:
        _release(model)
        meta.update({"n_nonzero": 0, "turnover_l1": None})
        return Solution(
            weights=np.zeros(n),
            objective=np.inf,
            feasible=False,
            wall_time_s=clock_offset_s + (time.perf_counter() - t_start),
            solver=f"scip_native/{formulation}",
            solver_meta=meta,
        )

    best = model.getBestSol()
    weights = _solution_weights(best, w, z)
    _release(model)
    obj = problem.objective_value(weights)

    nonzero = weights > _NONZERO_TOL
    n_nonzero = int(nonzero.sum())
    budget_ok = bool(abs(weights.sum() - problem.budget) < 1e-4)  # a plain bool: the registry's JSON
    # Semicontinuous box: each weight is 0 or within [w_min, w_max].
    box_ok = bool(
        np.all(weights >= -1e-6)
        and np.all(weights <= problem.w_max + 1e-6)
        and np.all(weights[nonzero] >= problem.w_min - 1e-6)
    )
    card_ok = effective_card is None or n_nonzero <= effective_card
    if card_mode == "eq" and lower > 0.0:
        card_ok = card_ok and n_nonzero == effective_card  # every one of the K assets is held
    meta.update(
        {
            "n_nonzero": n_nonzero,
            "turnover_l1": float(np.abs(weights - prev).sum()) if prev is not None else None,
            "budget_ok": budget_ok,
            "box_ok": box_ok,
            "card_ok": card_ok,
        }
    )
    linear_ok = True
    if linear is not None:
        slack = float(linear[0] @ weights - linear[1])
        linear_ok = bool(slack >= -_LINEAR_TOL * linear[2])
        meta.update({"linear_ge_slack": slack, "linear_ok": linear_ok})
    return Solution(
        weights=weights,
        objective=obj,
        feasible=budget_ok and box_ok and card_ok and linear_ok,
        wall_time_s=clock_offset_s + (time.perf_counter() - t_start),
        solver=f"scip_native/{formulation}",
        solver_meta=meta,
    )
