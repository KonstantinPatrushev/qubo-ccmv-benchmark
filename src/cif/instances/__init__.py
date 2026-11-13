"""cif.instances subpackage: standard and realistic benchmark instances (WP-E3, R4).

* :mod:`cif.instances.orlib` — OR-Library port1-port5 (Chang et al. 2000): download with
  provenance, parser, ``orlib_problem`` in the project parameterisation, λ grid.
* :mod:`cif.instances.factor` — factor-model covariance with sector blocks and realistic
  volatilities / correlations.
* :mod:`cif.instances.diagnostics` — does the cardinality constraint bind (``s_cont``, ``binding``),
  instance statistics, description of the legacy dense generator.
* :mod:`cif.instances.pairs` — RQ1 pairs of factor instances that differ only in the volatility
  dispersion, with a common K.
* :mod:`cif.instances.boot` — C4: bootstrap of S&P 500 returns on a drawn month end, and the rule
  that picks its λ.
* :mod:`cif.instances.bcw` — problem (34) of Bertsimas & Cory-Wright on the OR-Library files (ridge
  folded into Σ, the minimum-return threshold ``r̄``): the data of the validation M11.
"""
