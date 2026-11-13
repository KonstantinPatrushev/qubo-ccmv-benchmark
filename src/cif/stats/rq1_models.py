"""Attribution models of RQ1 (ANALYSIS_PLAN §7.3; exploratory: estimates with intervals, no tests).

The response is ``asinh(E_samp(m) / τ)``: a signed transformation, defined also where the sampler
residual is negative (the pool beats the ceiling of the surrogate), linear for ``|E_samp| ≪ τ`` and
logarithmic above. The same models are fitted to ``asinh(E_prec(m) / τ)`` (ANALYSIS_PLAN §12 B2: the
sampler against the oracle of an equal pool size, without the price of a pool smaller than m). The factors are *nested* in the samplers (the penalty exists for SA and tabu,
the β range for SA only), so there is one model per sampler, with the fixed effects of N and h and
a random intercept for the instance (linear mixed model, REML)::

    SA       y ~ C(n) + C(h) + C(penalty) * C(beta) * log_m
    tabu     y ~ C(n) + C(h) + C(penalty) * log_m
    swap-SA  y ~ C(n) + C(h) + log_m

``*`` is the full crossing (main effects and all interactions), ``log_m`` the natural logarithm of
the pool size. Estimates are the coefficients of the treatment coding (reference: N = 16, the lower
h, the lowest penalty multiplier, the dwave β range, m = 1) with Wald intervals; there are no
p-values. The contrast between samplers is made at *given* configurations (the references of
ANALYSIS_PLAN §12 B3, or any pair of levels of the nested grid, for description): the predictions of two
models, averaged with equal weights over the cells of N and h, are subtracted.
The two models are fitted separately to the same instances, so their predictions are positively
correlated and the standard error of the contrast, which adds the variances, is conservative.

Limits of the specification (as in the plan): the random intercept does not model the dependence of
the five pool sizes of one sampler call, nor an instance-by-configuration effect.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix
from scipy import stats as sps

__all__ = [
    "SAMPLER_FACTORS",
    "NestedModel",
    "contrast_of_predictions",
    "fit_nested_model",
    "model_formula",
    "predict_marginal",
    "response",
    "sampler_contrast",
]

#: The factors that exist for each sampler (§7.3: nested, not crossed, hence one model per sampler).
SAMPLER_FACTORS = {"sa": ("penalty", "beta"), "tabu": ("penalty",), "swap_sa": ()}

_RHS = {
    "sa": "C(penalty) * C(beta) * log_m",
    "tabu": "C(penalty) * log_m",
    "swap_sa": "log_m",
}


def _factors(sampler: str) -> tuple[str, ...]:
    if sampler not in SAMPLER_FACTORS:
        raise ValueError(f"unknown sampler {sampler!r}; expected one of {sorted(SAMPLER_FACTORS)}")
    return SAMPLER_FACTORS[sampler]


def model_formula(sampler: str) -> str:
    """The formula of the model of ``sampler`` (see the module docstring)."""
    _factors(sampler)
    return f"y ~ C(n) + C(h) + {_RHS[sampler]}"


def response(e_samp, tau: float) -> np.ndarray:
    """``asinh(E_samp / τ)``."""
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be a positive number, got {tau}")
    return np.arcsinh(np.asarray(e_samp, dtype=float) / tau)


@dataclass(frozen=True, eq=False)
class NestedModel:
    """A fitted model: ``fixed`` (estimate, se and Wald interval per coefficient), the variance of
    the instance intercept (``random_var``), the residual variance, ``icc``, the sizes, the
    convergence flag and the warnings the fit raised. ``levels`` are the levels of the categorical
    factors in the data, ``cov_fixed`` the covariance of the fixed effects."""

    sampler: str
    formula: str
    fixed: pd.DataFrame
    cov_fixed: pd.DataFrame
    random_var: float
    resid_var: float
    icc: float
    n_obs: int
    n_groups: int
    n_missing: int
    converged: bool
    level: float
    levels: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    design_info: object = None


def fit_nested_model(long: pd.DataFrame, sampler: str, *, level: float = 0.95) -> NestedModel:
    """Fit the model of ``sampler`` to a long frame with the columns ``y`` (the response),
    ``instance`` (the random-effect group), ``n``, ``h``, ``log_m`` and the factors of the sampler
    (``penalty``, ``beta``). Rows without a response are dropped and counted (``n_missing``)."""
    factors = _factors(sampler)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    needed = ["y", "instance", "n", "h", "log_m", *factors]
    absent = [c for c in needed if c not in long.columns]
    if absent:
        raise ValueError(f"the long frame lacks the columns {absent}")
    data = long.loc[:, needed].copy()
    n_missing = int(data["y"].isna().sum())
    data = data.dropna(subset=["y"])
    if data[[c for c in needed if c != "y"]].isna().any().any():
        raise ValueError("factors, log_m and instance must not be missing")
    if data["instance"].nunique() < 2:
        raise ValueError("need at least 2 groups (instances) for the random effect")
    levels = {c: sorted(data[c].unique().tolist()) for c in ("n", "h", *factors)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.mixedlm(model_formula(sampler), data, groups=data["instance"]).fit(reml=True)
    k = len(result.fe_params)
    names = list(result.fe_params.index)
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    est = result.fe_params.to_numpy(dtype=float)
    se = result.bse_fe.to_numpy(dtype=float)
    fixed = pd.DataFrame({"estimate": est, "se": se, "ci_low": est - z * se, "ci_high": est + z * se},
                         index=names)
    cov = result.cov_params()
    cov_fixed = pd.DataFrame(np.asarray(cov)[:k, :k], index=names, columns=names)
    random_var = float(np.asarray(result.cov_re)[0, 0])
    resid_var = float(result.scale)
    total = random_var + resid_var
    return NestedModel(
        sampler=sampler, formula=model_formula(sampler), fixed=fixed, cov_fixed=cov_fixed,
        random_var=random_var, resid_var=resid_var, icc=random_var / total if total > 0.0 else math.nan,
        n_obs=len(data), n_groups=int(data["instance"].nunique()), n_missing=n_missing,
        converged=bool(result.converged), level=level, levels=levels,
        warnings=sorted({str(w.message) for w in caught}),
        design_info=result.model.data.design_info,
    )


def _check_level(model: NestedModel, name: str, value) -> None:
    levels = model.levels[name]
    if not any(value == lv or (isinstance(lv, float) and math.isclose(float(value), lv)) for lv in levels):
        raise ValueError(f"{name} = {value!r} is not a level of the design of the {model.sampler} "
                         f"model {levels}")


def predict_marginal(
    model: NestedModel, config: dict, ms: Sequence[int], ns: Sequence[int], hs: Sequence[float],
    tau: float, *, level: float | None = None, name: str = "e_samp",
) -> pd.DataFrame:
    """Predicted response at the sampler configuration ``config`` (``{"penalty": ..., "beta": ...}``
    as far as the sampler has them), for every pool size of ``ms``, averaged with equal weights over
    the cells ``ns × hs``. One row per m: ``estimate`` with ``se`` and the Wald interval on the scale
    of the response, and ``name`` (``e_samp``; ``e_prec`` for the response ``asinh(E_prec / τ)``,
    ANALYSIS_PLAN §12 B2) with its interval ``name_ci_low`` / ``name_ci_high`` on the scale of the
    residual (``τ·sinh`` of the response, monotone, so the interval maps endpoint by endpoint)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    factors = _factors(model.sampler)
    missing = [f for f in factors if f not in config]
    if missing:
        raise ValueError(f"the configuration must give {missing} for the {model.sampler} model")
    extra = [k for k in config if k not in factors]
    if extra:
        raise ValueError(f"the {model.sampler} model has no factor {extra}")
    for f in factors:
        _check_level(model, f, config[f])
    for n in ns:
        _check_level(model, "n", n)
    for h in hs:
        _check_level(model, "h", h)
    level = model.level if level is None else level
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    names = list(model.fixed.index)
    beta = model.fixed["estimate"].to_numpy()
    cov = model.cov_fixed.to_numpy()
    rows = []
    for m in ms:
        grid = pd.DataFrame([{**config, "n": n, "h": h, "log_m": math.log(m)} for n in ns for h in hs])
        design = np.asarray(dmatrix(model.design_info, grid, return_type="dataframe").loc[:, names])
        x = design.mean(axis=0)
        est = float(x @ beta)
        se = float(math.sqrt(max(x @ cov @ x, 0.0)))
        lo, hi = est - z * se, est + z * se
        rows.append({"m": int(m), "estimate": est, "se": se, "ci_low": lo, "ci_high": hi,
                     name: tau * math.sinh(est), f"{name}_ci_low": tau * math.sinh(lo),
                     f"{name}_ci_high": tau * math.sinh(hi)})
    return pd.DataFrame(rows)


def contrast_of_predictions(
    pa: pd.DataFrame, pb: pd.DataFrame, *, names: tuple[str, str] = ("a", "b"), level: float = 0.95,
    value: str = "e_samp",
) -> pd.DataFrame:
    """``a − b`` of two tables of :func:`predict_marginal` (the same pool sizes, the same ``value`` column):
    the estimates subtract, the variances add (module docstring)."""
    z = float(sps.norm.ppf(0.5 + level / 2.0))
    est = pa["estimate"] - pb["estimate"]
    se = np.sqrt(pa["se"] ** 2 + pb["se"] ** 2)
    return pd.DataFrame({
        "m": pa["m"], "a": names[0], "b": names[1], "estimate": est, "se": se,
        "ci_low": est - z * se, "ci_high": est + z * se,
        f"{value}_a": pa[value], f"{value}_b": pb[value],
    })


def sampler_contrast(
    model_a: NestedModel, config_a: dict, model_b: NestedModel, config_b: dict, ms: Sequence[int],
    ns: Sequence[int], hs: Sequence[float], tau: float, *, names: tuple[str, str] = ("a", "b"),
    level: float = 0.95, value: str = "e_samp",
) -> pd.DataFrame:
    """``a − b`` of the marginal predictions at the given configurations, per m (module docstring:
    the variances add). ``value`` names the residual of the response (``e_samp``, ``e_prec``): the
    columns ``value_a`` / ``value_b`` hold its predictions on the scale of the residual."""
    pa = predict_marginal(model_a, config_a, ms, ns, hs, tau, level=level, name=value)
    pb = predict_marginal(model_b, config_b, ms, ns, hs, tau, level=level, name=value)
    return contrast_of_predictions(pa, pb, names=names, level=level, value=value)
