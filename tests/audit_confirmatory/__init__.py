"""Tests of the audit of the confirmatory stages (docs/reviews/2026-10-13-audit-confirmatory.md).

Every module reproduces one finding on the registered confirmatory runs and is expected to FAIL until the finding
is closed (by a fix in code, a re-run of the analysis, or a recorded qualification, as each module says). The
tests only read ``runs/``, ``results/`` and ``configs/``; they skip when a run directory is absent.
"""
