"""The attribution models of ANALYSIS_PLAN §7.3 on synthetic data with known coefficients.

Response ``asinh(E_samp / τ)``; separate models per sampler (the factors are nested); fixed effects
of N and h, a random intercept per instance; estimates with intervals, no tests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from cif.stats import rq1_models as models

TAU = 1e-3
PENALTIES = (1.0, 3.0, 10.0, 30.0, 100.0)
MS = (1, 5, 20, 50, 200)

# true coefficients of the generator below
INTERCEPT, B_N24, B_HHI = 0.30, 0.20, 0.40
B_PEN_HIGH = -0.15  # penalty multiplier of 10 or more (against 1 and 3)
B_OBJ = 0.25  # β range "objective"
B_LOGM = -0.20
B_OBJ_LOGM = 0.10
SD_INSTANCE, SD_NOISE = 0.20, 0.30


def _long(sampler: str, *, seed: int = 1, n_inst: int = 160, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, SD_INSTANCE, n_inst)
    rows = []
    for i in range(n_inst):
        n = 16 if i < n_inst // 2 else 24
        h = 0.15 if i % 2 == 0 else 0.45
        configs = {
            "sa": [(p, b) for p in PENALTIES for b in ("default", "objective")],
            "tabu": [(p, None) for p in PENALTIES],
            "swap_sa": [(None, None)],
        }[sampler]
        for penalty, beta in configs:
            for m in MS:
                lm = math.log(m)
                y = INTERCEPT + shift + B_N24 * (n == 24) + B_HHI * (h == 0.45) + effects[i]
                if penalty is not None and penalty >= 10.0:
                    y += B_PEN_HIGH
                if beta == "objective":
                    y += B_OBJ + B_OBJ_LOGM * lm
                y += B_LOGM * lm + rng.normal(0.0, SD_NOISE)
                rows.append({"instance": f"{n}/{i}/{h}", "n": n, "h": h,
                             "penalty": penalty, "beta": beta, "log_m": lm, "y": y})
    return pd.DataFrame(rows)


def test_the_formulas_follow_the_nesting_of_the_factors():
    assert models.model_formula("sa") == "y ~ C(n) + C(h) + C(penalty) * C(beta) * log_m"
    assert models.model_formula("tabu") == "y ~ C(n) + C(h) + C(penalty) * log_m"
    assert models.model_formula("swap_sa") == "y ~ C(n) + C(h) + log_m"
    with pytest.raises(ValueError, match="sampler"):
        models.model_formula("annealer9000")


def test_asinh_response_is_signed_and_defined_for_negative_residuals():
    e = np.array([-2e-3, 0.0, 1e-3, 5e-2])
    y = models.response(e, TAU)
    assert y[0] == pytest.approx(-math.asinh(2.0)) and y[1] == 0.0 and y[2] == pytest.approx(math.asinh(1.0))
    assert np.all(np.diff(y) > 0)  # monotone: the order of the residuals is kept
    with pytest.raises(ValueError, match="tau"):
        models.response(e, 0.0)


@pytest.fixture(scope="module")
def sa_model():
    return models.fit_nested_model(_long("sa"), "sa")


def test_fixed_effects_of_the_sa_model_recover_the_generator(sa_model):
    fe = sa_model.fixed
    assert set(fe.columns) == {"estimate", "se", "ci_low", "ci_high"}  # estimates and intervals, no p
    assert fe.loc["Intercept", "estimate"] == pytest.approx(INTERCEPT, abs=0.12)
    assert fe.loc["C(n)[T.24]", "estimate"] == pytest.approx(B_N24, abs=0.1)
    assert fe.loc["C(h)[T.0.45]", "estimate"] == pytest.approx(B_HHI, abs=0.1)
    assert fe.loc["C(beta)[T.objective]", "estimate"] == pytest.approx(B_OBJ, abs=0.1)
    assert fe.loc["log_m", "estimate"] == pytest.approx(B_LOGM, abs=0.03)
    assert fe.loc["C(beta)[T.objective]:log_m", "estimate"] == pytest.approx(B_OBJ_LOGM, abs=0.04)
    assert fe.loc["C(penalty)[T.100.0]", "estimate"] == pytest.approx(B_PEN_HIGH, abs=0.05)
    assert fe.loc["C(penalty)[T.3.0]", "estimate"] == pytest.approx(0.0, abs=0.05)
    assert "C(penalty)[T.100.0]:C(beta)[T.objective]:log_m" in fe.index
    assert (fe["ci_low"] < fe["estimate"]).all() and (fe["estimate"] < fe["ci_high"]).all()


def test_the_intervals_cover_most_true_values(sa_model):
    fe = sa_model.fixed
    truth = {"C(n)[T.24]": B_N24, "C(h)[T.0.45]": B_HHI, "log_m": B_LOGM,
             "C(beta)[T.objective]": B_OBJ, "C(beta)[T.objective]:log_m": B_OBJ_LOGM}
    covered = [fe.loc[t, "ci_low"] <= v <= fe.loc[t, "ci_high"] for t, v in truth.items()]
    assert sum(covered) >= 4


def test_the_random_effect_of_the_instance_is_estimated(sa_model):
    assert sa_model.random_var == pytest.approx(SD_INSTANCE**2, rel=0.35)
    assert sa_model.resid_var == pytest.approx(SD_NOISE**2, rel=0.1)
    assert sa_model.icc == pytest.approx(sa_model.random_var / (sa_model.random_var + sa_model.resid_var))
    assert sa_model.n_groups == 160 and sa_model.n_obs == 160 * 10 * 5
    assert sa_model.converged is True and sa_model.formula == models.model_formula("sa")


def test_the_instance_effect_matters_for_the_standard_error_of_between_instance_factors():
    """The effect of N varies between instances only: its standard error must reflect the 160
    instances, not the 8000 rows (a plain regression would be about 4 times too confident)."""
    fit = models.fit_nested_model(_long("sa"), "sa")
    naive_se = SD_NOISE / math.sqrt(8000 / 4)
    assert fit.fixed.loc["C(n)[T.24]", "se"] > 3.0 * naive_se


def test_tabu_and_swap_models_have_only_their_own_factors():
    tabu = models.fit_nested_model(_long("tabu"), "tabu")
    assert not any("beta" in t for t in tabu.fixed.index) and any("penalty" in t for t in tabu.fixed.index)
    swap = models.fit_nested_model(_long("swap_sa"), "swap_sa")
    assert list(swap.fixed.index) == ["Intercept", "C(n)[T.24]", "C(h)[T.0.45]", "log_m"]
    assert swap.fixed.loc["log_m", "estimate"] == pytest.approx(B_LOGM, abs=0.03)


def test_rows_without_a_response_are_dropped_and_counted():
    data = _long("swap_sa")
    data.loc[data.index[:7], "y"] = np.nan
    fit = models.fit_nested_model(data, "swap_sa")
    assert fit.n_obs == len(data) - 7 and fit.n_missing == 7


def test_missing_columns_or_too_little_data_are_refused():
    data = _long("sa")
    with pytest.raises(ValueError, match="beta"):
        models.fit_nested_model(data.drop(columns=["beta"]), "sa")
    with pytest.raises(ValueError, match="instance"):
        models.fit_nested_model(data.drop(columns=["instance"]), "swap_sa")
    with pytest.raises(ValueError, match="sampler"):
        models.fit_nested_model(data, "annealer9000")
    with pytest.raises(ValueError, match="groups"):
        models.fit_nested_model(data[data["instance"] == data["instance"].iloc[0]], "sa")


def test_the_marginal_prediction_is_the_sum_of_the_coefficients_over_the_balanced_design(sa_model):
    config = {"penalty": 100.0, "beta": "objective"}
    pred = models.predict_marginal(sa_model, config, MS, ns=(16, 24), hs=(0.15, 0.45), tau=TAU)
    fe = sa_model.fixed["estimate"]
    for m, row in zip(MS, pred.itertuples(), strict=True):
        lm = math.log(m)
        expect = (fe["Intercept"] + 0.5 * fe["C(n)[T.24]"] + 0.5 * fe["C(h)[T.0.45]"]
                  + fe["C(penalty)[T.100.0]"] + fe["C(beta)[T.objective]"] + fe["log_m"] * lm
                  + fe["C(penalty)[T.100.0]:C(beta)[T.objective]"]
                  + fe["C(penalty)[T.100.0]:log_m"] * lm + fe["C(beta)[T.objective]:log_m"] * lm
                  + fe["C(penalty)[T.100.0]:C(beta)[T.objective]:log_m"] * lm)
        assert row.estimate == pytest.approx(expect, abs=1e-9)
        assert row.ci_low < row.estimate < row.ci_high and row.se > 0.0
        assert row.e_samp == pytest.approx(TAU * math.sinh(row.estimate))  # back to the scale of E_samp
        assert row.e_samp_ci_low < row.e_samp < row.e_samp_ci_high
    assert list(pred["m"]) == list(MS)


def test_a_config_outside_the_design_is_refused(sa_model):
    with pytest.raises(ValueError, match="penalty"):
        models.predict_marginal(sa_model, {"penalty": 7.0, "beta": "default"}, MS, ns=(16, 24),
                                hs=(0.15, 0.45), tau=TAU)
    with pytest.raises(ValueError, match="beta"):
        models.predict_marginal(sa_model, {"penalty": 10.0}, MS, ns=(16, 24), hs=(0.15, 0.45), tau=TAU)


def test_the_contrast_between_samplers_is_the_difference_of_their_predictions():
    sa = models.fit_nested_model(_long("sa", seed=2), "sa")
    swap = models.fit_nested_model(_long("swap_sa", seed=3, shift=0.5), "swap_sa")
    table = models.sampler_contrast(sa, {"penalty": 100.0, "beta": "default"}, swap, {}, MS,
                                    ns=(16, 24), hs=(0.15, 0.45), tau=TAU, names=("sa", "swap_sa"))
    pa = models.predict_marginal(sa, {"penalty": 100.0, "beta": "default"}, MS, (16, 24), (0.15, 0.45), TAU)
    pb = models.predict_marginal(swap, {}, MS, (16, 24), (0.15, 0.45), TAU)
    assert list(table["m"]) == list(MS) and set(table["a"]) == {"sa"} and set(table["b"]) == {"swap_sa"}
    assert table["estimate"].to_numpy() == pytest.approx((pa["estimate"] - pb["estimate"]).to_numpy())
    # independent fits: the variances add (conservative for models fitted to the same instances)
    se = np.sqrt(pa["se"] ** 2 + pb["se"] ** 2)
    assert table["se"].to_numpy() == pytest.approx(se.to_numpy())
    # swap-SA was shifted up by 0.5 in the generator, so sa - swap is about -0.5 + the penalty/β effects
    assert table["estimate"].mean() < -0.2
    half = (table["ci_high"] - table["ci_low"]) / 2
    assert half.to_numpy() == pytest.approx(1.959964 * se.to_numpy(), rel=1e-4)


def test_the_back_transformed_columns_carry_the_name_of_the_response():
    """§12 B2: the same models are fitted to ``asinh(E_prec / τ)``; its columns are named after it."""
    swap = models.fit_nested_model(_long("swap_sa", seed=3), "swap_sa")
    sa = models.fit_nested_model(_long("sa", seed=2), "sa")
    default = models.predict_marginal(swap, {}, MS, (16, 24), (0.15, 0.45), TAU)
    named = models.predict_marginal(swap, {}, MS, (16, 24), (0.15, 0.45), TAU, name="e_prec")
    assert {"e_samp", "e_samp_ci_low", "e_samp_ci_high"} <= set(default.columns)
    assert {"e_prec", "e_prec_ci_low", "e_prec_ci_high"} <= set(named.columns) and "e_samp" not in named.columns
    assert named["e_prec"].to_numpy() == pytest.approx(default["e_samp"].to_numpy())
    table = models.sampler_contrast(sa, {"penalty": 100.0, "beta": "default"}, swap, {}, MS, (16, 24),
                                    (0.15, 0.45), TAU, names=("sa", "swap_sa"), value="e_prec")
    assert {"e_prec_a", "e_prec_b"} <= set(table.columns) and "e_samp_a" not in table.columns
    with pytest.raises(ValueError, match="name"):
        models.predict_marginal(swap, {}, MS, (16, 24), (0.15, 0.45), TAU, name="")


def test_models_are_deterministic():
    a = models.fit_nested_model(_long("swap_sa", seed=4), "swap_sa")
    b = models.fit_nested_model(_long("swap_sa", seed=4), "swap_sa")
    assert a.fixed.equals(b.fixed) and a.random_var == b.random_var
