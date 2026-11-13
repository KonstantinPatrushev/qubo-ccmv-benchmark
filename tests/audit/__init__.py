"""Tests of the methodology audit (docs/reviews/2026-10-04-audit-e0.md).

* ``test_engine_crosscheck.py`` is a PASSING independent check of the walk-forward accounting
  (what was verified and found correct);
* every other module reproduces one audit finding and is expected to FAIL until the finding is
  fixed (red tests; the finding ID is in the module docstring).
"""
