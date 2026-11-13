"""Audit D-2 (MINOR, 2026-10-08): large spin-off days outside the "98 days" of R6.

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

``pseudo_split_days.parquet`` lists 111 days with a Yahoo ratio that is no share-count ratio:
3 corrected (adjcorr-1), 98 with ratio > 1 (the "98 дней спин-оффов без дивиденда" of REVIEW R6,
PLAN and the US sensitivity of ANALYSIS_PLAN §3) and 10 with ratio < 1, for which the value model
does not apply (a reverse split mixed with a spin-off). The 10 are in no decision and in no
sensitivity, although four of them move by more than 5 % in a day in ``prices_adj``:

    MSI 2011-01-04 +8.0 %, LDOS 2013-09-30 +15.0 %, DD 2019-06-03 +17.8 %, VTR 2015-08-18 −23.6 %

DD 2019-06-03 (a reverse split mixed with a spin-off, docstring of ``pseudo_split_inventory``) lies
in the TEST window of the fold formed on 2019-05-08, where DD is one of the selected top-100 (audit
script, 155 US folds); the median day return of the panel's stocks was +0.8 % that day
(checks.txt). Expected: every uncorrected day of this list with a day return above 10 % in
magnitude is verified (``corporate_action_checks.json``) or corrected.

Data test: skipped without ``data/processed/sp500_pit``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "processed" / "sp500_pit"


@pytest.mark.skipif(not (DATA / "pseudo_split_days.parquet").exists(), reason="sp500_pit not built")
def test_d2_large_uncorrected_spin_off_days_are_verified_or_corrected():
    days = pd.read_parquet(DATA / "pseudo_split_days.parquet")
    checks = DATA / "corporate_action_checks.json"
    verified = set(json.loads(checks.read_text())) if checks.exists() else set()
    large = days[~days["corrected"] & (days["return_adjusted"].abs() > 0.10)]
    unverified = [
        f"{r.ticker}@{pd.Timestamp(r.date).date()} ({r.return_adjusted:+.1%}, ratio {r.ratio:.4g})"
        for r in large.itertuples()
        if f"{r.ticker}@{pd.Timestamp(r.date).date()}" not in verified
    ]
    assert not unverified, f"uncorrected, unverified day returns beyond ±10 %: {unverified}"
