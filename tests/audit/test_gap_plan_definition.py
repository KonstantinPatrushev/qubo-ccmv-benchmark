"""Audit G-1 (MAJOR, 2026-10-08): ``cif.metrics.gap`` is not the gap registered in the plan (§5).

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

ANALYSIS_PLAN v2.2 §5 (M10) registers ``ĝ(f) = (f − f_ref) / s_ref`` with
``s_ref = |μᵀw_ref| + (λ/2)·w_refᵀΣw_ref > 0`` and certifies an optimum by SCIP's ABSOLUTE gap
``|primal − dual| ≤ 10⁻⁶·s_ref``; §11 lists "масштаб в cif.metrics.gap" and "absgap у SCIP" as code
needed before the pilot. The module still implements the v1 definition and its docstring calls it
"the gap used in the study, fixed in advance"; REVIEW.md marks R9 🟢 "определение §5".

Example (plan units): f_ref = −1e-4, f = 0, f_cont = −2e-4, s_ref = 0.1.
* plan: ĝ = 1e-4 / 0.1 = 1e-3 (= τ, at the edge of the tolerance);
* code: relative_gap = 1e-4 / max(1e-4, 2e-4) = 0.5, i.e. γ(t) = min(1, ĝ) = 0.5 in the primal
  integral — 500× the registered value. The normalisations disagree exactly where f_ref ≈ 0, the
  case M10 was written for.

Certificate: SCIP stopped at ``gaplimit`` with |primal − dual| = 5e-9 and s_ref = 0.05, so the
plan's absolute criterion (≤ 5e-8) certifies; SCIP's relative gap is 5e-5 > 1e-6 and
``reference_from_exact`` returns ``best_known``.
"""

from __future__ import annotations

import inspect

import pytest

from cif.metrics.gap import reference_from_exact, relative_gap


def _scale_keyword(fn) -> str | None:
    params = inspect.signature(fn).parameters
    return next((name for name in ("s_ref", "scale") if name in params), None)


def test_g1_relative_gap_uses_the_registered_scale():
    kw = _scale_keyword(relative_gap)
    assert kw is not None, (
        "relative_gap has no scale argument; plan §5 registers (f − f_ref)/s_ref, "
        f"the code computes (f − f_ref)/max(|f_ref|, |f_cont|): {relative_gap(0.0, -1e-4, -2e-4)}"
    )
    assert relative_gap(0.0, -1e-4, **{kw: 0.1}) == pytest.approx(1e-3)


def test_g1_certificate_is_the_absolute_gap_of_the_plan():
    meta = {"status": "gaplimit", "primal_bound": -1.0e-4, "dual_bound": -1.0e-4 - 5e-9,
            "gap": 5e-5}
    kw = _scale_keyword(reference_from_exact)
    assert kw is not None, (
        "reference_from_exact certifies by SCIP's relative gap only "
        f"(returns {reference_from_exact(meta)[1]!r} here); plan §5: |primal − dual| ≤ 1e-6·s_ref"
    )
    f_ref, ref_type = reference_from_exact(meta, **{kw: 0.05})
    assert ref_type == "certified" and f_ref == pytest.approx(-1.0e-4)
