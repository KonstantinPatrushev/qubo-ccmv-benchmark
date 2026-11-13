"""E1 analysis stage: H1a, the estimates and the attribution models (ANALYSIS_PLAN §2 RQ1, §7.1–§7.3).

Reads a registered run of :mod:`cif.experiments.e1_attribution` (its ``results.jsonl``, the effective
``config.json`` and the manifest) and writes tables to a results directory. Statistics are those of
:mod:`cif.stats.rq1` and :mod:`cif.stats.rq1_models`; this module is the layer that knows the row
schema. Settings (τ, the numerical zero of the differences, bootstrap size and seed, the sampler
configurations at which samplers are contrasted, the p-value of H1c from the E2 stream) are in a YAML
file (``configs/E1/analysis.yaml``), not in defaults of the code (R8).

**The unit.** The instance (``n``, ``pair_seed``, ``h``). The oracle part of a row (``K``, the scale
``s*``, ``E_form(m)``) is the same in all cells of an instance and is checked to be; the sampler part
is summarised per instance by the median over the solver seeds (§7.2). A *pair* is the two levels of
h of one seed: the cluster of the bootstrap and the unit of the paired test.

**H1a** (§7.1): the stratified (by N) one-sided Wilcoxon signed-rank test of the paired differences
``E_form(m0; h = 0.45) − E_form(m0; h = 0.15)``, ``m0`` taken from the pilot block of the run's config;
adjusted by Holm together with H1c (:func:`cif.stats.rq1.holm_family`).

**Estimates** (not tests): shares of instances with ``E_form(m) > τ`` with Wilson intervals; means and
medians of ``E_form(m)``, ``E_pipe(m)``, ``E_samp(m)`` with bootstrap intervals (clusters = pairs,
strata = N); the pool curve (quality against the total time of the pipeline); the design (binding
share, binding depth ``s_cont/K`` and coefficient of variation of the volatilities by h).

**The split of the residual** (§12 B2): ``E_samp(m) = E_prec(m) + E_short(m)``, the sampler against the
oracle at an equal pool size and the price of a pool smaller than m. :func:`decomposition_table` reports
the three with bootstrap intervals and the share of cells with ``n_pipe(m) < m``, by configuration (so by
sampler), pool size, N and h. The unit is the instance mean over the solver seeds: the identity is linear,
and a median of a sum is not the sum of medians (the unit of the models and of ``e_samp.csv`` is the
median, §7.2).

**Models** (§7.3, exploratory): one mixed model per sampler (:mod:`cif.stats.rq1_models`) for the
response ``asinh(E_samp(m)/τ)`` and, by §12 B2, for ``asinh(E_prec(m)/τ)`` (``models.responses``), on the
instance-level medians (``instance_median``, §7.2) and/or on the rows of single seeds (``seed``);
estimates with intervals.

**References, the grid and what is exploratory** (§12 B3). RQ1 has no tuning step, so the "tuned
configuration" of §6 and §7.3 is the reference of each sampler (``models.reference``, which the settings
may not change: :data:`cif.experiments.e1_attribution.REFERENCE`). Every configuration has a ``role``:
``reference``; ``grid`` (the other levels of the nested grid of §7.3); ``exploratory`` (a swap-SA level
other than the reference, i.e. ``T_cold = 0.2·T_hot``, and every configuration the analyst lists in
``models.chosen_after_grid`` after seeing the grid). The samplers are *contrasted* at the references; the
predictions are made, and the contrasts described, over every level of the grid, the role of a contrast
being ``reference`` (both sides), ``exploratory`` (either side) or ``grid``. The swap-SA levels other than
the reference are fitted as models of their own, so that the model of the plan is the model of the rows of
the plan and nothing else.

Command line::

    python -m cif.experiments.e1_analysis --run runs/<run_id> --config configs/E1/analysis.yaml \\
        --out results/E1/analysis/<run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from importlib import metadata
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cif.experiments import e1_attribution as e1
from cif.experiments import run
from cif.stats.rq1 import (
    cluster_bootstrap_table,
    holm_family,
    stratified_signed_rank,
    wilson_interval,
)
from cif.stats.rq1_models import (
    SAMPLER_FACTORS,
    NestedModel,
    contrast_of_predictions,
    fit_nested_model,
    predict_marginal,
)
from cif.stats.rq1_models import response as asinh_response

__all__ = [
    "analyse",
    "check_frame",
    "decomposition_table",
    "design_table",
    "difference_table",
    "e_form_table",
    "frame_from_rows",
    "h1a_outcome",
    "h1a_result",
    "instance_frame",
    "load_run",
    "load_settings",
    "long_frames",
    "main",
    "mark_chosen_after_grid",
    "model_tables",
    "output_status",
    "pair_frame",
    "pool_curve_table",
    "resolve_m0",
    "samp_table",
    "sampler_diagnostics_table",
    "shares_table",
    "validate_settings",
]

STATUSES = ("ok", "no_valid_samples")
#: ``E`` columns of a row for the pool sizes ``ms`` (``E_form_at_npipe`` is ``E_form(n_pipe(m))``, §12 B2)
_E_PREFIXES = ("E_form", "E_pipe", "E_samp", "E_prec", "E_short", "E_form_at_npipe", "n_form", "n_pipe")
#: the oracle part of a row: identical in all cells of an instance
_ORACLE_KEYS = ("K", "scale", "f_star", "s_cont", "binding", "depth", "cv_vol", "n_subsets", "tau_top")
_SEED_TABLE = {"e_form": 1, "difference": 2, "h1a": 3, "e_samp": 4, "pool_curve": 5, "decomposition": 6}
#: responses of the models of §7.3: the residual of the sampler and (§12 B2) its part at an equal pool size
RESPONSES = ("E_samp", "E_prec")
#: file-name tag of the models of a response ("" keeps the names of the primary response)
_RESPONSE_TAG = {"E_samp": "", "E_prec": "prec_"}
#: the coordinates that identify a configuration of each sampler (a model factor, or for swap-SA the level)
_CONFIG_KEYS = {"sa": ("penalty", "beta"), "tabu": ("penalty",), "swap_sa": ("cold_ratio",)}
#: the oracle is deterministic, but shards on different machines may differ in the last bits (SCIP,
#: Clarabel): normalised values may differ by this much between the cells of an instance
ORACLE_TOL = 1e-9


# --------------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------------


def frame_from_rows(rows: Sequence[dict], ms: Sequence[int]) -> pd.DataFrame:
    """The rows as a frame with the ids ``instance`` (``n/pair_seed/h``) and ``pair`` (``n/pair_seed``);
    missing values are NaN. The columns of every pool size of ``ms`` must be there, and ``cold_ratio`` and
    ``config_id`` (§12 B3). ``role`` is the role of the configuration of a row in the plan
    (:func:`cif.experiments.e1_attribution.config_role`)."""
    df = pd.DataFrame(list(rows))
    wanted = [f"{p}_{m}" for p in _E_PREFIXES for m in ms]
    absent = [c for c in (*wanted, "cold_ratio", "config_id") if c not in df.columns]
    if absent:
        raise ValueError(f"the rows lack the columns {absent}")
    for c in wanted:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["cold_ratio"] = pd.to_numeric(df["cold_ratio"], errors="coerce").astype(float)
    h = df["h"].astype(float)
    df["pair"] = df["n"].astype(int).astype(str) + "/" + df["pair_seed"].astype(int).astype(str)
    df["instance"] = df["pair"] + "/" + h.map(lambda x: f"{x:g}")
    df["role"] = _roles(df)
    return df


def _roles(df: pd.DataFrame) -> pd.Series:
    """The plan's role of the configuration of every row (one evaluation per configuration)."""
    roles = {}
    for r in df.drop_duplicates("config_id").itertuples():
        roles[r.config_id] = e1.config_role(
            r.sampler, None if pd.isna(r.penalty) else float(r.penalty),
            r.beta if isinstance(r.beta, str) else None, None if pd.isna(r.cold_ratio) else float(r.cold_ratio))
    return df["config_id"].map(roles)


def mark_chosen_after_grid(df: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """A copy of the frame in which the configurations of ``models.chosen_after_grid`` have the role
    ``exploratory`` (§12 B3: a configuration picked after the nested grid was seen). A reference cannot be
    listed (it is the plan's, not a choice), and a listed configuration must be one of the rows."""
    out = df.copy()
    for entry in settings["models"].get("chosen_after_grid") or []:
        mask = out["sampler"] == entry["sampler"]
        for key in _CONFIG_KEYS[entry["sampler"]]:
            if key == "beta":
                mask &= out[key] == entry[key]
            else:
                mask &= np.isclose(out[key].astype(float), float(entry[key]))
        if not mask.any():
            raise ValueError(f"models.chosen_after_grid: {entry} is not in the rows (a level of the run's grid is meant)")
        if (out.loc[mask, "role"] == "reference").any():
            raise ValueError(f"{entry} is a reference configuration (§12 B3), it cannot be listed in "
                             f"models.chosen_after_grid")
        out.loc[mask, "role"] = "exploratory"
    return out


def check_frame(df: pd.DataFrame, ms: Sequence[int], *, expected_keys: set[str] | None = None) -> dict:
    """Integrity of the rows; blocking problems raise, the rest is counted in the returned dict.

    Blocking: a row with an ``error``; a ``status`` other than ``ok`` / ``no_valid_samples``; cells
    missing from, or not in, ``expected_keys`` (the grid of the config); an oracle that differs
    between the cells of an instance by more than :data:`ORACLE_TOL` (shards that computed different
    references). Counted: rows flagged inconsistent (an ``E`` below −1e-6, something beat the
    reference), references that are not certified, cells without a valid sample.
    """
    if "error" in df.columns and df["error"].notna().any():
        bad = df[df["error"].notna()].iloc[0]
        raise ValueError(f"{int(df['error'].notna().sum())} rows carry an error, e.g. {bad['error']}")
    statuses = df["status"].value_counts().to_dict()
    unknown = sorted(set(statuses) - set(STATUSES))
    if unknown:
        raise ValueError(f"unknown row status {unknown}; expected {STATUSES}")
    if "cell_key" in df.columns and df["cell_key"].duplicated().any():
        raise ValueError(f"{int(df['cell_key'].duplicated().sum())} rows repeat a cell: duplicate cells "
                         f"would weigh twice (the run was not merged by cell_key)")
    n_missing = n_unexpected = 0
    if expected_keys is not None:
        keys = set(df["cell_key"])
        n_missing, n_unexpected = len(expected_keys - keys), len(keys - expected_keys)
        if n_missing:
            raise ValueError(f"{n_missing} cells of the grid are missing from the rows")
        if n_unexpected:
            raise ValueError(f"{n_unexpected} rows are unexpected: not cells of the grid (wrong config?)")
    spread = _oracle_spread(df, ms)
    if spread > ORACLE_TOL:
        raise ValueError(f"the oracle differs between cells of one instance by {spread:.3g}: "
                         f"the shards computed different references")
    certified = df["ref_type"] == "certified" if "ref_type" in df.columns else pd.Series(True, index=df.index)
    return {
        "n_rows": len(df), "n_instances": int(df["instance"].nunique()),
        "status_counts": {k: int(v) for k, v in statuses.items()},
        "n_no_valid": int(statuses.get("no_valid_samples", 0)),
        "n_inconsistent": int((~df["consistent"].astype(bool)).sum()) if "consistent" in df.columns else 0,
        "n_not_certified": int((~certified).sum()),
        "n_missing": n_missing, "oracle_spread_max": spread,
    }


def _oracle_spread(df: pd.DataFrame, ms: Sequence[int]) -> float:
    keys = [c for c in (*_ORACLE_KEYS, *(f"E_form_{m}" for m in ms), *(f"n_form_{m}" for m in ms))
            if c in df.columns]
    numeric = [c for c in keys if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return 0.0
    per_instance = df.groupby("instance")[numeric].agg(["min", "max"])
    worst = 0.0
    for c in numeric:
        diff = (per_instance[(c, "max")].astype(float) - per_instance[(c, "min")].astype(float)).abs()
        worst = max(worst, float(diff.max()))
    return worst


def instance_frame(df: pd.DataFrame, ms: Sequence[int]) -> pd.DataFrame:
    """One row per instance: the oracle part of its rows (``K``, the scale, the diagnostics,
    ``E_form(m)`` and ``n_form(m)``), after checking that all its cells agree on it."""
    spread = _oracle_spread(df, ms)
    if spread > ORACLE_TOL:
        raise ValueError(f"the oracle differs between cells of one instance by {spread:.3g}: "
                         f"the shards computed different references")
    keep = ["instance", "pair", "n", "pair_seed", "h",
            *(c for c in _ORACLE_KEYS if c in df.columns),
            *(c for c in ("s_cont_min_pair", "ref_type", "cv_mu") if c in df.columns),
            *(f"E_form_{m}" for m in ms), *(f"n_form_{m}" for m in ms)]
    inst = df.drop_duplicates("instance")[keep].sort_values(["n", "pair_seed", "h"]).reset_index(drop=True)
    inst["n"], inst["pair_seed"] = inst["n"].astype(int), inst["pair_seed"].astype(int)
    inst["h"] = inst["h"].astype(float)
    for m in ms:  # NaN would count as "not above tau" in a share and stop the paired test
        missing = inst.loc[inst[f"E_form_{m}"].isna(), "instance"]
        if len(missing):
            raise ValueError(f"E_form_{m} is missing for {len(missing)} instances, e.g. {missing.iloc[0]}")
    return inst


def pair_frame(inst: pd.DataFrame, ms: Sequence[int], h_lo: float, h_hi: float) -> pd.DataFrame:
    """One row per pair with ``E_form_{m}_lo`` / ``_hi`` and the difference ``d_{m} = hi − lo``."""
    out = []
    for pair, sub in inst.groupby("pair", sort=False):
        lo, hi = sub[sub["h"] == h_lo], sub[sub["h"] == h_hi]
        if len(lo) != 1 or len(hi) != 1:
            raise ValueError(f"pair {pair} lacks a level: needs one instance at h = {h_lo} and one at h = {h_hi}")
        lo, hi = lo.iloc[0], hi.iloc[0]
        row = {"pair": pair, "n": int(lo["n"]), "pair_seed": int(lo["pair_seed"]), "K": lo["K"]}
        for m in ms:
            row[f"E_form_{m}_lo"], row[f"E_form_{m}_hi"] = float(lo[f"E_form_{m}"]), float(hi[f"E_form_{m}"])
            row[f"d_{m}"] = row[f"E_form_{m}_hi"] - row[f"E_form_{m}_lo"]
        out.append(row)
    return pd.DataFrame(out).sort_values(["n", "pair_seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Groups, seeds
# --------------------------------------------------------------------------------------------


def _groups(frame: pd.DataFrame) -> Iterator[tuple[str, int | None, float | None, bool, pd.DataFrame]]:
    """``(group, n, h, independent, rows)`` for the whole frame, each N, each h and each N × h.

    ``independent`` is true if the group holds at most one instance of every pair (one level of h),
    so that its instances are independent draws.
    """
    yield "all", None, None, False, frame
    for n in sorted(frame["n"].unique()):
        yield "n", int(n), None, False, frame[frame["n"] == n]
    for h in sorted(frame["h"].unique()):
        yield "h", None, float(h), True, frame[frame["h"] == h]
    for n in sorted(frame["n"].unique()):
        for h in sorted(frame["h"].unique()):
            yield "n_h", int(n), float(h), True, frame[(frame["n"] == n) & (frame["h"] == h)]


def _finish(table: pd.DataFrame) -> pd.DataFrame:
    """Tidy a table: ``n`` is an integer column, empty where a group does not fix N (not 16.0)."""
    if "n" in table.columns:
        table["n"] = table["n"].astype("Int64")
    return table


def _seed(settings: dict, *indices: int) -> list[int]:
    return [int(settings["bootstrap"]["seed"]), *(int(i) for i in indices)]


def _boot(settings: dict) -> dict:
    return {"n_boot": int(settings["bootstrap"]["n_boot"]), "level": float(settings["level"])}


# --------------------------------------------------------------------------------------------
# H1a
# --------------------------------------------------------------------------------------------


def h1a_outcome(p_value: float, holm: dict, alpha: float) -> str:
    """``confirmed`` / ``not_confirmed`` / ``pending_h1c`` for H1a (§10, first rows).

    With the family complete: confirmed iff the Holm-adjusted p-value is at most ``alpha``. Without
    the p-value of H1c: the adjusted value lies between ``p`` and the Bonferroni bound ``min(1, 2p)``,
    so it is confirmed if the bound is at most ``alpha``, not confirmed if ``p`` already exceeds it,
    and pending otherwise.
    """
    if holm["complete"]:
        return "confirmed" if holm["adjusted"]["H1a"] <= alpha else "not_confirmed"
    if holm["upper_bound"]["H1a"] <= alpha:
        return "confirmed"
    return "not_confirmed" if p_value > alpha else "pending_h1c"


def h1a_result(pairs: pd.DataFrame, m0: int, settings: dict) -> dict:
    """The test of H1a at the pool size ``m0`` (module docstring) with the size of the effect.

    ``difference`` holds the mean and the median of the pair differences with the percentile
    bootstrap intervals (clusters = pairs, strata = N).
    """
    col = f"d_{m0}"
    if col not in pairs.columns:
        raise ValueError(f"m0 = {m0} is not among the pool sizes of the rows (no column {col})")
    d = pairs[col].to_numpy(dtype=float)
    test = stratified_signed_rank(d, pairs["n"].to_numpy(), alternative="greater", method="exact",
                                  zero_tol=float(settings["zero_tol"]))
    boot = cluster_bootstrap_table({"d": d}, pairs["pair"].to_numpy(), pairs["n"].to_numpy(),
                                   seed=_seed(settings, _SEED_TABLE["h1a"], m0), **_boot(settings))["d"]
    holm = holm_family({"H1a": test.p_value, "H1c": settings["h1c"]["p_value"]}, alpha=float(settings["alpha"]))
    return {
        "hypothesis": "H1a", "m0": int(m0), "tau": float(settings["tau"]), "alternative": "greater",
        "test": "stratified (by N) Wilcoxon signed-rank on the pair differences E_form(m0; hi) - E_form(m0; lo)",
        "method": test.method, "statistic": test.statistic, "p_value": test.p_value,
        "n_pairs": len(pairs), "n_nonzero": test.n_nonzero, "n_zero": test.n_zero,
        "zero_tol": test.zero_tol,
        "strata": {str(k): v for k, v in test.strata.items()},
        "share_positive": float((d > float(settings["zero_tol"])).mean()),
        "difference": {
            "mean": {"estimate": boot["mean"].estimate, "ci": [boot["mean"].ci_low, boot["mean"].ci_high]},
            "median": {"estimate": boot["median"].estimate, "ci": [boot["median"].ci_low, boot["median"].ci_high]},
            "level": float(settings["level"]), "n_boot": int(settings["bootstrap"]["n_boot"]),
            "clusters": "pairs", "strata": "N",
        },
        "holm": holm, "alpha": float(settings["alpha"]),
        "outcome": h1a_outcome(test.p_value, holm, float(settings["alpha"])),
    }


# --------------------------------------------------------------------------------------------
# Estimates
# --------------------------------------------------------------------------------------------


def shares_table(inst: pd.DataFrame, ms: Sequence[int], settings: dict) -> pd.DataFrame:
    """Shares of instances with ``E_form(m) > τ`` for ``m`` in ``settings["share_ms"]``, with Wilson
    intervals, for all instances, each N, each h and each N × h. ``independent`` says whether the
    group holds one level of every pair (otherwise its instances are paired and the interval is
    approximate)."""
    tau, level = float(settings["tau"]), float(settings["level"])
    rows = []
    for m in settings["share_ms"]:
        if f"E_form_{m}" not in inst.columns:
            raise ValueError(f"share_ms has m = {m}, which is not among the pool sizes {list(ms)}")
        for group, n, h, independent, sub in _groups(inst):
            k = int((sub[f"E_form_{m}"] > tau).sum())
            lo, hi = wilson_interval(k, len(sub), level)
            rows.append({"m": int(m), "group": group, "n": n, "h": h, "independent": independent,
                         "n_instances": len(sub), "n_exceed": k, "share": k / len(sub) if len(sub) else math.nan,
                         "ci_low": lo, "ci_high": hi, "tau": tau, "level": level})
    return _finish(pd.DataFrame(rows))


def e_form_table(inst: pd.DataFrame, ms: Sequence[int], settings: dict) -> pd.DataFrame:
    """Mean and median of ``E_form(m)`` with bootstrap intervals (clusters = pairs, strata = N)."""
    rows = []
    for j, m in enumerate(ms):
        for g, (group, n, h, independent, sub) in enumerate(_groups(inst)):
            boot = cluster_bootstrap_table(
                {"v": sub[f"E_form_{m}"].to_numpy()}, sub["pair"].to_numpy(), sub["n"].to_numpy(),
                seed=_seed(settings, _SEED_TABLE["e_form"], j, g), **_boot(settings))["v"]
            rows.append({"m": int(m), "group": group, "n": n, "h": h, "independent": independent,
                         "n_instances": len(sub), "n_clusters": boot["mean"].n_clusters,
                         "mean": boot["mean"].estimate, "mean_ci_low": boot["mean"].ci_low,
                         "mean_ci_high": boot["mean"].ci_high, "median": boot["median"].estimate,
                         "median_ci_low": boot["median"].ci_low, "median_ci_high": boot["median"].ci_high,
                         "level": float(settings["level"]), "n_boot": int(settings["bootstrap"]["n_boot"])})
    return _finish(pd.DataFrame(rows))


def difference_table(pairs: pd.DataFrame, ms: Sequence[int], settings: dict) -> pd.DataFrame:
    """The pair difference ``E_form(m; hi) − E_form(m; lo)`` per m, for all pairs and each N: mean and
    median with bootstrap intervals and the shares of positive, zero and negative differences."""
    zero_tol = float(settings["zero_tol"])
    rows = []
    for j, m in enumerate(ms):
        for g, (group, n, sub) in enumerate([("all", None, pairs)] + [
                ("n", int(k), pairs[pairs["n"] == k]) for k in sorted(pairs["n"].unique())]):
            d = sub[f"d_{m}"].to_numpy(dtype=float)
            boot = cluster_bootstrap_table({"d": d}, sub["pair"].to_numpy(), sub["n"].to_numpy(),
                                           seed=_seed(settings, _SEED_TABLE["difference"], j, g),
                                           **_boot(settings))["d"]
            rows.append({"m": int(m), "group": group, "n": n, "n_pairs": len(sub),
                         "mean": boot["mean"].estimate, "mean_ci_low": boot["mean"].ci_low,
                         "mean_ci_high": boot["mean"].ci_high, "median": boot["median"].estimate,
                         "median_ci_low": boot["median"].ci_low, "median_ci_high": boot["median"].ci_high,
                         "share_positive": float((d > zero_tol).mean()),
                         "share_zero": float((np.abs(d) <= zero_tol).mean()),
                         "share_negative": float((d < -zero_tol).mean()),
                         "level": float(settings["level"]), "n_boot": int(settings["bootstrap"]["n_boot"])})
    return _finish(pd.DataFrame(rows))


def design_table(inst: pd.DataFrame) -> pd.DataFrame:
    """The design as realised (§2 RQ1: "доля связывающих и фактический коэффициент вариации
    волатильностей отчитываются по h"): the share of binding instances (``s_cont > K``), the binding
    depth ``s_cont/K`` and the coefficient of variation of the volatilities, for all instances, each h
    and each N × h; ``K`` is the cardinality of the pair."""
    rows = []
    for group, n, h, _, sub in _groups(inst):
        if group in ("n",):
            continue
        rows.append({"group": group, "n": n, "h": h, "n_instances": len(sub),
                     "share_binding": float(sub["binding"].astype(bool).mean()),
                     "depth_mean": float(sub["depth"].mean()), "depth_median": float(sub["depth"].median()),
                     "depth_min": float(sub["depth"].min()), "depth_max": float(sub["depth"].max()),
                     "cv_vol_mean": float(sub["cv_vol"].mean()), "cv_vol_median": float(sub["cv_vol"].median()),
                     "cv_vol_min": float(sub["cv_vol"].min()), "cv_vol_max": float(sub["cv_vol"].max()),
                     "K_median": float(sub["K"].median()), "K_min": int(sub["K"].min()), "K_max": int(sub["K"].max())})
    return _finish(pd.DataFrame(rows))


# --------------------------------------------------------------------------------------------
# The sampler residual and the pipeline
# --------------------------------------------------------------------------------------------

SAMPLER_ORDER = ("sa", "tabu", "swap_sa")
UNITS = ("instance_median", "seed")
#: contrasts between samplers (a − b): negative means the first has the smaller residual
CONTRAST_PAIRS = (("tabu", "sa"), ("swap_sa", "sa"), ("swap_sa", "tabu"))


def _configurations(df: pd.DataFrame) -> list[dict]:
    """The sampler configurations of the rows with their ``role``, ordered by sampler, penalty, β range
    and cold ratio."""
    columns = ["config_id", "sampler", "penalty", "beta", "cold_ratio", "role"]
    cfg = df.drop_duplicates("config_id")[columns]
    cfg = cfg.assign(_s=cfg["sampler"].map({s: i for i, s in enumerate(SAMPLER_ORDER)}),
                     _p=cfg["penalty"].fillna(math.inf), _b=cfg["beta"].fillna(""),
                     _c=cfg["cold_ratio"].fillna(math.inf))
    cfg = cfg.sort_values(["_s", "_p", "_b", "_c"])
    return cfg[columns].to_dict("records")


def _instance_medians(sub: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The median over the solver seeds of each instance (NaN skipped), with its ``pair``, ``n``, ``h``."""
    per = sub.groupby("instance")[list(columns)].median()
    ids = sub.drop_duplicates("instance").set_index("instance")[["pair", "n", "h"]]
    return per.join(ids).reset_index()


def _share(values: pd.Series, mask: pd.Series, level: float) -> tuple[float, float, float]:
    k, n = int(mask.sum()), len(values)
    lo, hi = wilson_interval(k, n, level)
    return (k / n if n else math.nan), lo, hi


def samp_table(df: pd.DataFrame, ms: Sequence[int], settings: dict) -> pd.DataFrame:
    """``E_samp(m)`` and ``E_pipe(m)`` of every sampler configuration: instance medians over the solver
    seeds (§7.2), then their mean and median with bootstrap intervals (clusters = pairs, strata = N),
    for all instances, each N, each h and each N × h; plus the shares of instances with
    ``E_pipe > τ``, ``E_samp < −τ`` (the pool beats the ceiling) and ``|E_samp| ≤ τ`` with Wilson
    intervals, and the median number of subsets of the pool that was used. Instances without a valid
    sample of a configuration are left out of it."""
    tau, level = float(settings["tau"]), float(settings["level"])
    rows = []
    for c, cfg in enumerate(_configurations(df)):
        sub = df[df["config_id"] == cfg["config_id"]]
        columns = [f"{p}_{m}" for m in ms for p in ("E_samp", "E_pipe", "n_pipe")]
        per = _instance_medians(sub, columns)
        for j, m in enumerate(ms):
            have = per[per[f"E_samp_{m}"].notna()]
            for g, (group, n, h, independent, g_rows) in enumerate(_groups(have)):
                row = {**cfg, "m": int(m), "group": group, "n": n, "h": h, "independent": independent,
                       "n_instances": len(g_rows), "level": level,
                       "n_boot": int(settings["bootstrap"]["n_boot"])}
                if len(g_rows):
                    boot = cluster_bootstrap_table(
                        {"e_samp": g_rows[f"E_samp_{m}"].to_numpy(), "e_pipe": g_rows[f"E_pipe_{m}"].to_numpy()},
                        g_rows["pair"].to_numpy(), g_rows["n"].to_numpy(),
                        seed=_seed(settings, _SEED_TABLE["e_samp"], c, j, g), **_boot(settings))
                    for name, per_stat in boot.items():
                        for stat, b in per_stat.items():
                            row[f"{name}_{stat}"] = b.estimate
                            row[f"{name}_{stat}_ci_low"], row[f"{name}_{stat}_ci_high"] = b.ci_low, b.ci_high
                    row["n_clusters"] = boot["e_samp"]["mean"].n_clusters
                    e_samp, e_pipe = g_rows[f"E_samp_{m}"], g_rows[f"E_pipe_{m}"]
                    for name, values, mask in (("share_pipe_above_tau", e_pipe, e_pipe > tau),
                                               ("share_samp_below_minus_tau", e_samp, e_samp < -tau),
                                               ("share_samp_within_tau", e_samp, e_samp.abs() <= tau)):
                        row[name], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = _share(values, mask, level)
                    row["n_pipe_median"] = float(g_rows[f"n_pipe_{m}"].median())
                rows.append(row)
    return _finish(pd.DataFrame(rows))


def _instance_means(sub: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The mean over the solver seeds of each instance (NaN skipped), with its ``pair``, ``n``, ``h``."""
    per = sub.groupby("instance")[list(columns)].mean()
    ids = sub.drop_duplicates("instance").set_index("instance")[["pair", "n", "h"]]
    return per.join(ids).reset_index()


def decomposition_table(df: pd.DataFrame, ms: Sequence[int], settings: dict) -> pd.DataFrame:
    """``E_samp(m) = E_prec(m) + E_short(m)`` per sampler configuration, pool size and group (§12 B2).

    Instance means over the solver seeds (the identity is linear; the median of a sum is not the sum of
    medians), then the mean over the instances with a bootstrap interval (clusters = pairs, strata = N)
    for the three terms, on the same resamples, so the means add up in every resample; only means, which
    the identity is about (and the bootstrap of a median is several times dearer);
    ``identity_gap`` is ``e_samp_mean − e_prec_mean − e_short_mean`` (rounding only). Also the share of
    cells whose pool has fewer subsets than the oracle's ``T_m`` (``n_pipe(m) < n_form(m)``, which is
    ``n_pipe(m) < m`` in the design, where ``C(N, K) > m_max``): the mean over the instances of the share
    of their seeds, with a cluster bootstrap interval (``share_short``); a cell without a valid subset has
    an empty pool and counts as short. ``n_pipe_median`` is the median over instances of the median
    ``n_pipe(m)``. Groups: all instances, each N, each h, each N × h."""
    level = float(settings["level"])
    parts = ("E_samp", "E_prec", "E_short")
    rows = []
    for c, cfg in enumerate(_configurations(df)):
        cells = df[df["config_id"] == cfg["config_id"]].copy()
        for m in ms:
            cells[f"short_{m}"] = (cells[f"n_pipe_{m}"] < cells[f"n_form_{m}"]).astype(float)
        per = _instance_means(cells, [*(f"{p}_{m}" for m in ms for p in parts), *(f"short_{m}" for m in ms)])
        sizes = cells.groupby("instance")[[f"n_pipe_{m}" for m in ms]].median()
        per = per.join(sizes, on="instance")
        for j, m in enumerate(ms):
            for g, (group, n, h, independent, g_rows) in enumerate(_groups(per)):
                row = {**cfg, "m": int(m), "group": group, "n": n, "h": h, "independent": independent,
                       "n_instances": len(g_rows), "level": level, "n_boot": int(settings["bootstrap"]["n_boot"])}
                boot = cluster_bootstrap_table(
                    {"e_samp": g_rows[f"E_samp_{m}"].to_numpy(), "e_prec": g_rows[f"E_prec_{m}"].to_numpy(),
                     "e_short": g_rows[f"E_short_{m}"].to_numpy(), "share_short": g_rows[f"short_{m}"].to_numpy()},
                    g_rows["pair"].to_numpy(), g_rows["n"].to_numpy(), stats=("mean",),
                    seed=_seed(settings, _SEED_TABLE["decomposition"], c, j, g), **_boot(settings))
                for name, per_stat in boot.items():
                    b = per_stat["mean"]
                    key = name if name == "share_short" else f"{name}_mean"
                    row[key] = b.estimate
                    row[f"{key}_ci_low"], row[f"{key}_ci_high"] = b.ci_low, b.ci_high
                row["n_clusters"] = boot["e_samp"]["mean"].n_clusters
                row["identity_gap"] = row["e_samp_mean"] - row["e_prec_mean"] - row["e_short_mean"]
                row["n_pipe_median"] = float(g_rows[f"n_pipe_{m}"].median())
                rows.append(row)
    return _finish(pd.DataFrame(rows))


def sampler_diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per configuration and group, over the cells: the time of the sampler call (median and
    quartiles of ``solve_s``, the quantity that the pilot sized, plan §6), the share of valid samples,
    Kendall's τ between the surrogate energy and ``F`` over the pool, the rate at which the best valid
    subset reached the exact surrogate minimum and the number of distinct subsets in a pool."""
    def stat(column: str, how: str, g_rows: pd.DataFrame, q: float = 0.5) -> float:
        """NaN (silently) if the column is absent or missing in every row of the group."""
        if column not in g_rows.columns:
            return math.nan
        x = pd.to_numeric(g_rows[column], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return math.nan
        return float(np.mean(x) if how == "mean" else np.quantile(x, q))

    rows = []
    for cfg in _configurations(df):
        sub = df[df["config_id"] == cfg["config_id"]]
        for group, n, h, _, g_rows in _groups(sub):
            row = {**cfg, "group": group, "n": n, "h": h, "n_cells": len(g_rows),
                   "n_no_valid": int((g_rows["status"] == "no_valid_samples").sum()),
                   "solve_s_median": stat("solve_s", "q", g_rows), "solve_s_q25": stat("solve_s", "q", g_rows, 0.25),
                   "solve_s_q75": stat("solve_s", "q", g_rows, 0.75),
                   "build_s_median": stat("build_s", "q", g_rows), "frac_valid_mean": stat("frac_valid", "mean", g_rows),
                   "tau_pool_median": stat("tau_pool", "q", g_rows),
                   "pool_n_unique_median": stat("pool_n_unique", "q", g_rows),
                   "n_samples_median": stat("n_samples", "q", g_rows), "refine_s_median": stat("refine_s", "q", g_rows),
                   "hit_rate_mean": stat("hit_surrogate_min", "mean", g_rows)}
            rows.append(row)
    return _finish(pd.DataFrame(rows))


def pool_curve_table(df: pd.DataFrame, ms: Sequence[int], settings: dict) -> pd.DataFrame:
    """Quality of the pipeline against its total time, per configuration and pool size m (descriptive).

    The time of a pipeline with a pool of m is the building of the QUBO, the sampler call and the
    refinement of the ``n_pipe(m)`` subsets of the pool, each at the cost per refinement that the
    oracle measured on that instance (``oracle_refine_s`` for the refinements of ``T_{m_max}``).
    Instance medians over the seeds; the median and quartiles of the time over instances, and the
    mean and median of ``E_pipe(m)`` with bootstrap intervals; for all instances and each N."""
    m_max = max(ms)
    per_call = df["oracle_refine_s"] / df[f"n_form_{m_max}"]
    base = df["build_s"] + df["solve_s"]
    level = float(settings["level"])
    rows = []
    for c, cfg in enumerate(_configurations(df)):
        mask = df["config_id"] == cfg["config_id"]
        sub = df.loc[mask].copy()
        for m in ms:
            sub[f"time_{m}"] = base[mask] + sub[f"n_pipe_{m}"] * per_call[mask]
        columns = [f"{p}_{m}" for m in ms for p in ("E_pipe", "time")]
        per = _instance_medians(sub, columns)
        for j, m in enumerate(ms):
            have = per[per[f"E_pipe_{m}"].notna()]
            for g, (group, n, _h, _, g_rows) in enumerate(_groups(have)):
                if group not in ("all", "n"):
                    continue
                row = {**cfg, "m": int(m), "group": group, "n": n, "n_instances": len(g_rows), "level": level,
                       "n_boot": int(settings["bootstrap"]["n_boot"])}
                if len(g_rows):
                    t = g_rows[f"time_{m}"]
                    row.update(time_s_median=float(t.median()), time_s_q25=float(t.quantile(0.25)),
                               time_s_q75=float(t.quantile(0.75)))
                    boot = cluster_bootstrap_table(
                        {"e_pipe": g_rows[f"E_pipe_{m}"].to_numpy()}, g_rows["pair"].to_numpy(),
                        g_rows["n"].to_numpy(), seed=_seed(settings, _SEED_TABLE["pool_curve"], c, j, g),
                        **_boot(settings))["e_pipe"]
                    for stat, b in boot.items():
                        row[f"e_pipe_{stat}"] = b.estimate
                        row[f"e_pipe_{stat}_ci_low"], row[f"e_pipe_{stat}_ci_high"] = b.ci_low, b.ci_high
                rows.append(row)
    return _finish(pd.DataFrame(rows))


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def _model_key(sampler: str, cold_ratio: float | None = None) -> str:
    """The key of a model: the sampler; for a swap-SA level other than the reference also its cold ratio."""
    return sampler if cold_ratio is None else f"{sampler}[cold_ratio={cold_ratio:g}]"


def _sampler_of(key: str) -> str:
    return key.split("[", 1)[0]


def long_frames(
    df: pd.DataFrame, ms: Sequence[int], settings: dict, unit: str, response: str = "E_samp"
) -> dict[str, pd.DataFrame]:
    """The long frame of each model for :func:`cif.stats.rq1_models.fit_nested_model`: one row per
    (instance, configuration, pool size) with ``y = asinh(E / τ)`` and ``log_m``, where ``E`` is the
    ``response``: ``E_samp`` or, by §12 B2, ``E_prec``. The keys are the samplers; the levels of swap-SA other
    than the reference (``models.reference``, §12 B3) are separate models, ``swap_sa[cold_ratio=0.2]``, so
    that the model of the plan is fitted to the rows of the plan. ``unit`` is ``instance_median`` (the
    median over the solver seeds of each instance and configuration, §7.2) or ``seed`` (every cell). Rows
    without a response are dropped; their number is in ``frame.attrs["n_missing"]``, the cold ratio of a
    swap-SA model in ``frame.attrs["cold_ratio"]``."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if response not in RESPONSES:
        raise ValueError(f"response must be one of {RESPONSES}, got {response!r}")
    tau = float(settings["tau"])
    cols = [f"{response}_{m}" for m in ms]
    ref_cold = float(settings["models"]["reference"]["swap_sa"]["cold_ratio"])
    out = {}
    for sampler in SAMPLER_ORDER:
        sub = df[df["sampler"] == sampler]
        if sub.empty:
            continue
        parts = [(sampler, None, sub)]
        if sampler == "swap_sa":
            ratio = sub["cold_ratio"].astype(float)
            at_reference = np.isclose(ratio, ref_cold)
            if not at_reference.any():
                raise ValueError(f"the rows of swap_sa have no cell at the reference cold_ratio {ref_cold:g} (§12 B3)")
            parts = [(sampler, ref_cold, sub[at_reference])] + [
                (_model_key(sampler, x), float(x), sub[np.isclose(ratio, x)]) for x in sorted(ratio[~at_reference].unique())]
        for key, cold, part in parts:
            ids = ["instance", "n", "h", *SAMPLER_FACTORS[sampler]]
            per = (part.groupby(ids, sort=False)[cols].median().reset_index() if unit == "instance_median"
                   else part[[*ids, *cols]])
            long = per.melt(id_vars=ids, value_vars=cols, var_name="m_col", value_name="e")
            long["log_m"] = np.log(long.pop("m_col").str.replace(f"{response}_", "", regex=False).astype(int).astype(float))
            long["y"] = asinh_response(long.pop("e"), tau)
            n_missing = int(long["y"].isna().sum())
            long = long.dropna(subset=["y"]).reset_index(drop=True)
            long.attrs["n_missing"] = n_missing
            long.attrs["cold_ratio"] = cold
            out[key] = long
    return out


def _configs_of(model: NestedModel, sampler: str) -> list[dict]:
    """The configurations of the nested grid that a model was fitted to, as the factors it predicts at."""
    if sampler == "sa":
        return [{"penalty": p, "beta": b} for p in model.levels["penalty"] for b in model.levels["beta"]]
    if sampler == "tabu":
        return [{"penalty": p} for p in model.levels["penalty"]]
    return [{}]  # swap-SA: one level per model


def _contrast_role(role_a: str, role_b: str) -> str:
    """``reference`` if both configurations are references, ``exploratory`` if either is, else ``grid``."""
    if "exploratory" in (role_a, role_b):
        return "exploratory"
    return "reference" if role_a == role_b == "reference" else "grid"


def model_tables(
    df: pd.DataFrame, ms: Sequence[int], settings: dict, unit: str, response: str = "E_samp"
) -> tuple[dict, dict]:
    """Fit the models of §7.3 (module docstring) for the ``response`` (``E_samp`` or ``E_prec``) and
    tabulate them.

    Returns ``(tables, summary)``: ``tables`` has ``fixed_effects`` (sampler, model, term, estimate, se,
    interval; ``exploratory`` marks the models of the swap-SA levels other than the reference),
    ``predictions`` (the marginal response and the residual at *every* configuration of the nested grid,
    per m, with its ``config_id`` and ``role``) and ``contrasts`` (:data:`CONTRAST_PAIRS`, per m, for every
    pair of configurations of the two samplers; ``role`` is ``reference`` for the contrast at the references
    of §12 B3, ``exploratory`` if either side is, else ``grid``, descriptive); ``summary`` per model the
    formula, sizes, variance components and convergence, and the samplers ``skipped`` for lack of data.
    The roles are those of the ``role`` column of the frame (:func:`frame_from_rows`,
    :func:`mark_chosen_after_grid`).
    """
    tau, level = float(settings["tau"]), float(settings["level"])
    reference = settings["models"]["reference"]
    longs = long_frames(df, ms, settings, unit, response)
    value = response.lower()  # the column of the back-transformed prediction: e_samp, e_prec
    roles = df.drop_duplicates("config_id").set_index("config_id")["role"].to_dict()
    fitted: dict[str, NestedModel] = {}
    cold_of: dict[str, float | None] = {}
    summary: dict = {"skipped": [s for s in SAMPLER_ORDER if s not in {_sampler_of(k) for k in longs}]}
    fixed_rows = []
    for key, long in longs.items():
        sampler = _sampler_of(key)
        if sampler not in reference:
            raise ValueError(f"models.reference has no configuration for the sampler {sampler!r}")
        model = fit_nested_model(long, sampler, level=level)
        fitted[key], cold_of[key] = model, long.attrs["cold_ratio"]
        extra_level = key != sampler
        for term, r in model.fixed.iterrows():
            fixed_rows.append({"sampler": sampler, "model": key, "term": term, **r.to_dict(), "unit": unit,
                               "response": response, "exploratory": extra_level})
        summary[key] = {
            "exploratory": True,  # §7.3, §9: estimates with intervals, no tests
            "exploratory_level": extra_level,  # a swap-SA level that is not in the plan (§12 B3)
            "response": response, "unit": unit, "formula": model.formula,
            "n_obs": model.n_obs, "n_groups": model.n_groups,
            "n_missing": long.attrs.get("n_missing", 0) + model.n_missing, "converged": model.converged,
            "warnings": model.warnings, "random_var": model.random_var, "resid_var": model.resid_var,
            "icc": model.icc, "level": level, "reference": reference[sampler],
        }
    ns, hs = sorted(df["n"].astype(int).unique().tolist()), sorted(df["h"].astype(float).unique().tolist())
    predictions, pred_of = [], {}
    labels_of: dict[str, list[str]] = {}
    for key, model in fitted.items():
        sampler = _sampler_of(key)
        labels_of[key] = []
        for config in _configs_of(model, sampler):
            label = e1.config_label(sampler, config.get("penalty"), config.get("beta"), cold_of[key])
            role = roles.get(label) or e1.config_role(sampler, config.get("penalty"), config.get("beta"), cold_of[key])
            pred = predict_marginal(model, config, ms, ns, hs, tau, level=level, name=value)
            pred_of[(key, label)] = (pred, role)
            labels_of[key].append(label)
            predictions.append(pred.assign(sampler=sampler, model=key, config_id=label, role=role,
                                           config=str(config), unit=unit, response=response))
        if key == sampler:  # the model of the plan holds the reference of its sampler
            ref = reference[sampler]
            want = e1.config_label(sampler, ref.get("penalty"), ref.get("beta"), ref.get("cold_ratio"))
            if want not in labels_of[key]:
                raise ValueError(f"the reference configuration of {sampler} {ref} is not in the design "
                                 f"(configurations {labels_of[key]}; §12 B3: the samplers are contrasted at it)")
    contrasts = []
    keys_of = {s: [k for k in fitted if _sampler_of(k) == s] for s in SAMPLER_ORDER}
    for a, b in CONTRAST_PAIRS:
        for key_a, key_b in product(keys_of[a], keys_of[b]):
            for label_a, label_b in product(labels_of[key_a], labels_of[key_b]):
                (pa, role_a), (pb, role_b) = pred_of[(key_a, label_a)], pred_of[(key_b, label_b)]
                contrasts.append(
                    contrast_of_predictions(pa, pb, names=(a, b), level=level, value=value)
                    .assign(model_a=key_a, model_b=key_b, config_a=label_a, config_b=label_b,
                            role=_contrast_role(role_a, role_b), unit=unit, response=response))
    tables = {
        "fixed_effects": pd.DataFrame(fixed_rows),
        "predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        "contrasts": pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame(),
    }
    return tables, summary


# --------------------------------------------------------------------------------------------
# Settings, the registered run, the whole stage
# --------------------------------------------------------------------------------------------

_SETTINGS_KEYS = ("tau", "zero_tol", "alpha", "share_ms", "level", "bootstrap", "h1c", "models")

#: The status of each output in the plan (§7.1: one confirmatory test; §2, §5: estimates; §2: the pool
#: curve is descriptive; §7.3 and §9: the attribution models and contrasts are exploratory and are
#: labelled so). The key is the file name or its prefix (up to the first ``_<unit>`` suffix).
OUTPUT_STATUS = {
    "h1a.json": "confirmatory",
    "shares_e_form.csv": "estimate", "e_form.csv": "estimate", "e_form_difference.csv": "estimate",
    "e_samp.csv": "estimate", "e_decomposition.csv": "estimate",
    "pool_curve.csv": "descriptive", "sampler_diagnostics.csv": "descriptive", "design.csv": "descriptive",
    "instances.csv": "data", "pairs.csv": "data",
    "diagnostics.json": "diagnostic",
}
_EXPLORATORY_PREFIXES = ("models_", "fixed_effects_", "predictions_", "contrasts_")


def output_status(name: str) -> str:
    """The status of the output file ``name`` (see :data:`OUTPUT_STATUS`)."""
    if name.startswith(_EXPLORATORY_PREFIXES):
        return "exploratory"
    return OUTPUT_STATUS[name]


def validate_settings(settings: dict) -> dict:
    """The analysis settings (``configs/E1/analysis.yaml``) are complete and sane; returns a copy."""
    if not isinstance(settings, dict):
        raise ValueError("the analysis settings must be a mapping")
    for key in _SETTINGS_KEYS:
        if settings.get(key) is None and key != "h1c":
            raise ValueError(f"analysis settings: missing {key!r}")
    if "h1c" not in settings:
        raise ValueError("analysis settings: missing 'h1c'")

    def number(key, low, high=None, *, closed_low=False):
        v = settings[key]
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        ok = ok and (v >= low if closed_low else v > low) and (high is None or v < high)
        if not ok:
            raise ValueError(f"analysis settings: {key} must be a number in the range "
                             f"{'[' if closed_low else '('}{low}, {high if high is not None else 'inf'}), got {v!r}")

    number("tau", 0.0)
    number("zero_tol", 0.0, closed_low=True)
    number("alpha", 0.0, 1.0)
    number("level", 0.0, 1.0)
    ms = settings["share_ms"]
    if not isinstance(ms, list) or not ms or not all(isinstance(m, int) and not isinstance(m, bool) and m >= 1 for m in ms):
        raise ValueError(f"analysis settings: share_ms must be a non-empty list of pool sizes, got {ms!r}")
    boot = settings["bootstrap"]
    n_boot = boot.get("n_boot") if isinstance(boot, dict) else None
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"analysis settings: bootstrap.n_boot must be a positive integer, got {n_boot!r}")
    if not isinstance(boot.get("seed"), int) or isinstance(boot.get("seed"), bool):
        raise ValueError("analysis settings: bootstrap.seed must be an integer (seeds are explicit)")
    h1c = settings["h1c"]
    p = h1c.get("p_value") if isinstance(h1c, dict) else "bad"
    if p is not None and not (isinstance(p, (int, float)) and not isinstance(p, bool) and 0.0 <= p <= 1.0):
        raise ValueError(f"analysis settings: h1c.p_value must be null or in [0, 1], got {p!r}")
    models = settings["models"]
    units = models.get("units") if isinstance(models, dict) else None
    if not isinstance(units, list) or not units or any(u not in UNITS for u in units):
        raise ValueError(f"analysis settings: models.units must be a non-empty list from {UNITS}, got {units!r}")
    responses = models.get("responses")
    if (not isinstance(responses, list) or not responses or len(set(responses)) != len(responses)
            or any(r not in RESPONSES for r in responses)):
        raise ValueError(f"analysis settings: models.responses must be a non-empty list without repeats from "
                         f"{RESPONSES} (§12 B2), got {responses!r}")
    if models.get("reference") != e1.REFERENCE:
        raise ValueError(f"analysis settings: models.reference must be the reference configurations of the plan "
                         f"(§12 B3) {e1.REFERENCE}, got {models.get('reference')!r}; a configuration chosen after "
                         f"the grid was seen goes to models.chosen_after_grid (exploratory)")
    _validate_chosen(models.get("chosen_after_grid"))
    h1a = settings.get("h1a")
    if h1a is not None:
        m0 = h1a.get("m0") if isinstance(h1a, dict) else None
        if not isinstance(m0, int) or isinstance(m0, bool) or m0 < 1:
            raise ValueError(f"analysis settings: h1a.m0 must be a positive integer (a pool size), got {m0!r}")
    return dict(settings)


def _validate_chosen(chosen) -> None:
    """``models.chosen_after_grid``: a list of ``{sampler, <the coordinates of its configurations>}``."""
    if not isinstance(chosen, list):
        raise ValueError(f"analysis settings: models.chosen_after_grid must be a list (empty if nothing was "
                         f"chosen after the grid), got {chosen!r}")
    for entry in chosen:
        sampler = entry.get("sampler") if isinstance(entry, dict) else None
        if sampler not in _CONFIG_KEYS:
            raise ValueError(f"analysis settings: models.chosen_after_grid: every entry names a 'sampler' from "
                             f"{sorted(_CONFIG_KEYS)}, got {entry!r}")
        keys = _CONFIG_KEYS[sampler]
        if set(entry) != {"sampler", *keys}:
            raise ValueError(f"analysis settings: models.chosen_after_grid: an entry for {sampler} has the keys "
                             f"'sampler' and {list(keys)}, got {sorted(entry)}")
        bad = [k for k in keys if not _coordinate_ok(k, entry[k])]
        if bad:
            raise ValueError(f"analysis settings: models.chosen_after_grid: {bad} of {entry!r} are not valid levels")


def _coordinate_ok(key: str, value) -> bool:
    if key == "beta":
        return value in e1.BETAS
    number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return number and (0.0 < value <= 1.0 if key == "cold_ratio" else value > 0.0)


def load_settings(path: Path | str) -> dict:
    return validate_settings(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def load_run(run_dir: Path | str) -> tuple[dict, dict, list[dict]]:
    """``(manifest, effective config, rows)`` of a registered run directory."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return manifest, cfg, run.read_rows(run_dir / "results.jsonl")


def resolve_m0(cfg: dict, settings: dict, ms: Sequence[int]) -> tuple[int, str]:
    """The pool size of H1a and where it came from: the pilot block of the run's config, else
    ``h1a.m0`` of the analysis settings; if both are given they must agree."""
    block = (cfg["params"].get("pilot") or {}).get("m0")
    given = (settings.get("h1a") or {}).get("m0")
    if block is not None and given is not None and block != given:
        raise ValueError(f"m0 = {given} in the analysis settings contradicts m0 = {block} of the run's pilot block")
    if block is not None:
        m0, source = block, "run config params.pilot.m0"
    elif given is not None:
        m0, source = given, "analysis settings h1a.m0"
    else:
        raise ValueError("m0 is neither in the pilot block of the run's config nor in the analysis "
                         "settings (h1a.m0)")
    if m0 not in ms:
        raise ValueError(f"m0 = {m0} is not among the pool sizes {list(ms)}")
    return int(m0), source


def _json(obj) -> str:
    return json.dumps(e1._clean(obj), indent=1, allow_nan=False) + "\n"


def analyse(run_dir: Path | str, settings: dict, out_dir: Path | str, *, h1c_p: float | None = None) -> dict:
    """The whole stage on one registered run; writes the tables to ``out_dir`` and returns
    ``{"files", "h1a", "m0", "confirmatory", "warnings"}``.

    ``h1c_p`` overrides ``settings["h1c"]["p_value"]`` (the p-value of H1c comes from the E2 stream).
    """
    settings = validate_settings(settings)
    if h1c_p is not None:
        settings = {**settings, "h1c": {"p_value": float(h1c_p)}}
    manifest, cfg, rows = load_run(run_dir)
    if cfg.get("experiment") != "cif.experiments.e1_attribution":
        raise ValueError(f"not a run of cif.experiments.e1_attribution (experiment {cfg.get('experiment')!r})")
    params = cfg["params"]
    ms = [int(m) for m in params["ms"]]
    expected = {run.cell_key(c) for c in e1.cells(cfg)}
    frame = mark_chosen_after_grid(frame_from_rows(rows, ms), settings)
    diag = check_frame(frame, ms, expected_keys=expected)
    m0, m0_source = resolve_m0(cfg, settings, ms)
    levels = sorted(float(h) for h in params["h_levels"])
    inst = instance_frame(frame, ms)
    pairs = pair_frame(inst, ms, levels[0], levels[-1])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def put(name: str, content: str) -> None:
        (out / name).write_text(content, encoding="utf-8")
        written[name] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def put_csv(name: str, table: pd.DataFrame) -> None:
        put(name, table.to_csv(index=False))

    h1a = h1a_result(pairs, m0, settings)
    put("h1a.json", _json(h1a))
    put_csv("instances.csv", inst)
    put_csv("pairs.csv", pairs)
    put_csv("design.csv", design_table(inst))
    put_csv("shares_e_form.csv", shares_table(inst, ms, settings))
    put_csv("e_form.csv", e_form_table(inst, ms, settings))
    put_csv("e_form_difference.csv", difference_table(pairs, ms, settings))
    put_csv("e_samp.csv", samp_table(frame, ms, settings))
    put_csv("e_decomposition.csv", decomposition_table(frame, ms, settings))
    put_csv("sampler_diagnostics.csv", sampler_diagnostics_table(frame))
    put_csv("pool_curve.csv", pool_curve_table(frame, ms, settings))
    for response in settings["models"]["responses"]:
        tag = _RESPONSE_TAG[response]
        for unit in settings["models"]["units"]:
            tables, summary = model_tables(frame, ms, settings, unit, response)
            put(f"models_{tag}{unit}.json", _json(summary))
            for name, table in tables.items():
                put_csv(f"{name}_{tag}{unit}.csv", table)
    put("diagnostics.json", _json({"frame": diag, "ms": ms, "n_boot": settings["bootstrap"]["n_boot"],
                                   "units": settings["models"]["units"],
                                   "responses": settings["models"]["responses"]}))

    role = params.get("seed_role")
    mode, dirty = manifest.get("mode", cfg.get("mode")), manifest.get("git_dirty")
    warnings = []
    if role != "test":
        warnings.append(f"the run used {role!r} seeds: this is not the confirmatory analysis")
    if mode not in ("full", "ingested"):
        warnings.append(f"the run is a {mode!r} run (smoke runs are for the machinery only)")
    if dirty is not False:
        warnings.append(f"the run comes from a dirty or unknown code state (git_dirty = {dirty!r})")
    for key, text in (("n_inconsistent", "rows flagged inconsistent (something beat the reference)"),
                      ("n_not_certified", "rows whose reference is not certified"),
                      ("n_no_valid", "cells without a valid sample (E_pipe undefined)")):
        if diag[key]:
            warnings.append(f"{diag[key]} {text}")
    confirmatory = role == "test" and mode in ("full", "ingested") and dirty is False
    record = {
        "schema": "e1-analysis/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {"run_id": manifest.get("run_id"), "git_sha": manifest.get("git_sha"), "git_dirty": dirty,
                "config_hash": manifest.get("config_hash"), "mode": mode, "seed_role": role,
                "n_rows": diag["n_rows"]},
        "m0": {"value": m0, "source": m0_source},
        "configurations": dict(sorted(frame.drop_duplicates("config_id").set_index("config_id")["role"].items())),
        "settings": settings, "confirmatory": confirmatory, "warnings": warnings,
        "versions": {p: metadata.version(p) for p in ("numpy", "pandas", "scipy", "statsmodels")},
        "outputs": dict(sorted(written.items())),
        "output_status": {name: output_status(name) for name in sorted(written)},
    }
    put("analysis_manifest.json", _json(record))
    written.pop("analysis_manifest.json")  # its own hash is not in the file
    return {"files": sorted([*written, "analysis_manifest.json"]), "h1a": h1a, "m0": m0,
            "confirmatory": confirmatory, "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e1_analysis")
    parser.add_argument("--run", type=Path, required=True, help="runs/<run_id> of the attribution run")
    parser.add_argument("--config", type=Path, required=True, help="analysis settings (configs/E1/analysis.yaml)")
    parser.add_argument("--out", type=Path, required=True, help="results directory, e.g. results/E1/analysis/<run_id>")
    parser.add_argument("--h1c-p", type=float, default=None,
                        help="p-value of H1c from the E2 stream (overrides h1c.p_value of the settings)")
    args = parser.parse_args(argv)
    result = analyse(args.run, load_settings(args.config), args.out, h1c_p=args.h1c_p)
    h1a = result["h1a"]
    holm = h1a["holm"]
    adjusted = holm["adjusted"]["H1a"] if holm["complete"] else f"<= {holm['upper_bound']['H1a']:.4g} (H1c missing)"
    print(f"E1 analysis of {args.run.name}: confirmatory = {result['confirmatory']}")
    print(f"  H1a: m0 = {h1a['m0']}, stratified signed-rank p = {h1a['p_value']:.4g} (n = {h1a['n_pairs']} pairs), "
          f"Holm-adjusted {adjusted}, outcome {h1a['outcome']}")
    for w in result["warnings"]:
        print(f"  warning: {w}")
    print(f"  {len(result['files'])} files written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
