"""E3 pilot: the values the plan leaves to the pilot, as code over the FF49 pilot folds only (RQ3).

ANALYSIS_PLAN §3, §2 H3a, §7.4, §12. The pilot runs on the pilot period of FF49 (test windows in
1963-07-01 … 1972-12-31) and nowhere else; the cells are those of :mod:`cif.experiments.e3_walkforward`
with ``lambdas`` (the candidates {5, 10, 20}), both arms and ``n_ranks: 1`` (rank 2 is all the pilot
needs). From the rows of that run (checked by :func:`cif.experiments.e3_assemble.index_rows`) the report
holds, in this order:

* **λ_f** (§3, "Рычаг связывания"): the smallest candidate at which ``s_cont,f > ⌈1/u⌉ + 1`` on at
  least 70 % of the pilot folds (the MV rows, whatever their certificate); if there is none, 20, and
  the share is reported (:func:`choose_lambda`).
* **σ\\* and Δ_SR** (§2 H3a): the annual volatility of the excess return of the MV optimum before costs
  on the days of the pilot's ``F_k`` at ``λ_f``; ``Δ_SR = 0.0025/σ*``. **σ²_\\* and δ**: the annual
  variance of the returns of the GMV optimum on the same days; ``δ = (0.005/λ_f)/σ²_*`` and the margin
  ``[log(1 − δ), log(1 + δ)]`` of the log variance ratio (:func:`margins_from`). A ``δ`` outside
  ``(0, 1)`` is reported with its reason, not raised: the log margin does not exist and the plan has to
  look at it.
* **The power pilot** (§7.4, :func:`power_pilot`): ρ of the daily returns of the rank-2 path and the
  optimum, the half-width of the 90 % TOST interval of the pilot (LW block calibrated on the pilot
  pair, as in the confirmation) and the same rescaled by ``√(T_pilot/T_confirmatory)``. If the
  rescaled half-width exceeds the margin the outcome "неопределённо" of H3a is expected; this is
  written to §12 but does not change the design.

The stage writes ``pilot_report.json`` into a registered run. The report carries a draft of the line
for the deviation log (§12) and the YAML values for the confirmatory configs and the analysis config;
the plan itself is not touched by code (it is frozen, the orchestrator writes §12).

``python -m cif.experiments.e3_pilot --analysis-config configs/E3/analysis.yaml --source pilot``.
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cif.experiments import e3_analysis as an
from cif.experiments import e3_assemble as asmb
from cif.experiments import e3_data as data
from cif.experiments import e3_walkforward as e3
from cif.experiments import registry
from cif.experiments.run import effective_config, load_config
from cif.stats import calibrate_rank_block, rank_equivalence, sharpe_margin, variance_margin

__all__ = [
    "LambdaChoice",
    "PilotError",
    "choose_lambda",
    "main",
    "margins_from",
    "pilot_report",
    "power_pilot",
    "seed_for",
]

seed_for = an.seed_for
TRADING_DAYS = 252
FALLBACK_LAMBDA = 20.0
MIN_BINDING_SHARE = 0.7
SESOI = 0.0025
_SHARE_TOL = 1e-12  # 7 folds of 10 are "at least 70 %"


class PilotError(RuntimeError):
    """The pilot cannot be reported: the run is not a pilot run or a rule has no answer."""


# --------------------------------------------------------------------------------------------
# λ_f
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LambdaChoice:
    """The λ_f of the rule: ``shares[λ]`` is the share of binding folds of each candidate."""

    lambda_f: float
    fallback: bool
    shares: dict[float, float]


def choose_lambda(
    counts: dict[float, tuple[int, int]], min_share: float = MIN_BINDING_SHARE
) -> LambdaChoice:
    """The smallest candidate with at least ``min_share`` of binding folds; 20 if there is none.

    ``counts[λ] = (binding folds, folds)``. Candidates are tried in increasing order.
    """
    shares = {}
    for lam in sorted(counts):
        binding, n = counts[lam]
        if n <= 0:
            raise ValueError(f"no fold at lam = {lam}: the share of binding folds does not exist")
        shares[float(lam)] = binding / n
    for lam in sorted(counts):
        binding, n = counts[lam]
        if binding >= min_share * n - _SHARE_TOL:
            return LambdaChoice(float(lam), False, shares)
    return LambdaChoice(FALLBACK_LAMBDA, True, shares)


# --------------------------------------------------------------------------------------------
# σ*, Δ_SR, σ²_*, δ
# --------------------------------------------------------------------------------------------


def margins_from(asm: asmb.Assembly, sesoi: float = SESOI) -> dict:
    """The margins of H3a from the optimum paths of the pilot assembly (module docstring)."""
    lam = float(asm.meta["lam"])
    mv = asmb.fk_series(asm, "gross", "MV", "sharpe", ranks=[])
    gmv = asmb.fk_series(asm, "gross", "GMV", "logvar", ranks=[])
    sigma = float(mv.opt.std(ddof=1) * math.sqrt(TRADING_DAYS))
    variance = float(gmv.opt.var(ddof=1) * TRADING_DAYS)
    out: dict = {
        "lambda_f": lam, "sesoi": float(sesoi), "n_days": len(mv.opt), "sigma_star": sigma,
        "delta_sr": sharpe_margin(sigma, sesoi), "sigma2_star": variance,
        "delta_var": None, "margin_log_variance": None, "delta_var_error": None,
    }
    try:
        vm = variance_margin(lam, variance, sesoi)
        out.update(delta_var=vm.delta, margin_log_variance=[vm.low, vm.high])
    except ValueError as exc:
        out["delta_var_error"] = str(exc)
    return out


# --------------------------------------------------------------------------------------------
# The power pilot
# --------------------------------------------------------------------------------------------


def power_pilot(asm: asmb.Assembly, margins: dict, stats: dict, confirmatory_days: int) -> dict:
    """ρ and the rescaled half-width of the 90 % interval of H3a per arm (module docstring).

    ``stats`` is the ``stats`` section of the analysis config (seed, alpha, n_boot, calibration).
    """
    out: dict = {"confirmatory_days": int(confirmatory_days)}
    for arm, functional in (("MV", "sharpe"), ("GMV", "logvar")):
        margin = margins["delta_sr"] if arm == "MV" else margins.get("margin_log_variance")
        series = asmb.fk_series(asm, "gross", arm, functional, ranks=[2])
        rho = float(np.corrcoef(series.opt, series.ranks[:, 0])[0, 1])
        entry: dict = {"rho": rho, "n_days": len(series.opt), "functional": functional}
        if margin is None:
            out[arm] = {**entry, "skipped": "the margin does not exist (see margins.delta_var_error)"}
            continue
        cal = stats["calibration"]
        if cal.get("fixed_block") is not None:
            block, source = int(cal["fixed_block"]), "fixed"
        else:
            block = int(calibrate_rank_block(
                series.opt, series.ranks[:, 0], seed=seed_for(stats["seed"], "pilot", arm, "calibration"),
                functional=functional, block_grid=[int(b) for b in cal["grid"]],
                n_pseudo=int(cal["n_pseudo"]), n_boot=int(cal["n_boot"]),
            ).block)
            source = "calibration"
        res = rank_equivalence(
            series.opt, series.ranks[:, 0], margin if arm == "MV" else tuple(margin),
            functional=functional, block=block, n_boot=int(stats["n_boot"]),
            alpha=float(stats["alpha"]), seed=seed_for(stats["seed"], "pilot", arm, "power"),
        )
        half = 0.5 * (res.ci_tost[1] - res.ci_tost[0])
        rescaled = half * math.sqrt(len(series.opt) / confirmatory_days)
        # the interval must lie inside the margin: the smaller side of an asymmetric margin binds
        limit = float(margin) if arm == "MV" else float(min(-margin[0], margin[1]))
        out[arm] = {
            **entry, "block": block, "block_source": source, "estimate": res.estimate,
            "se": res.se, "ci_tost": list(res.ci_tost), "halfwidth_pilot": half,
            "halfwidth_confirmatory": rescaled, "margin_halfwidth": limit,
            "expected_inconclusive": bool(rescaled > limit), "outcome_pilot": res.outcome,
        }
    return out


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def pilot_report(
    cfg: dict, rows: list[dict], acfg: dict, *, confirmatory_days: int, workers: int = 1
) -> dict:
    """The report of the pilot for §12 (module docstring)."""
    p = e3.check_params(cfg)
    if not p.get("lambdas"):
        raise PilotError("a pilot run has params.lambdas (the candidates of the λ_f rule), not lambda_f")
    if set(p["arms"]) != {"MV", "GMV"}:
        raise PilotError(f"the pilot needs both arms (σ* from MV, σ²_* from GMV), got {p['arms']}")
    stats = acfg["stats"]
    for key in ("seed", "alpha", "n_boot", "calibration"):
        if key not in stats:
            raise ValueError(f"stats: missing '{key}'")
    sesoi = float((acfg.get("margins") or {}).get("sesoi", SESOI))
    min_share = float((acfg.get("pilot") or {}).get("min_binding_share", MIN_BINDING_SHARE))
    u = float(p["market"]["u"])
    floor = data.floor_cardinality(u)
    candidates = e3.lambdas_of(p)
    counts: dict[float, tuple[int, int]] = {}
    totals: dict[float, int] = {}
    for lam in candidates:
        index = asmb.index_rows(cfg, rows, lam=lam)
        mv = [r for (arm, _), r in index.rows.items() if arm == "MV" and r.get("s_cont") is not None]
        counts[lam] = (sum(1 for r in mv if data.is_binding(r["s_cont"], u)), len(mv))
        totals[lam] = sum(1 for (arm, _) in index.rows if arm == "MV")
    choice = choose_lambda(counts, min_share)
    if choice.lambda_f not in candidates:
        raise PilotError(f"the rule chose lambda_f = {choice.lambda_f:g}, which is not among the "
                         f"candidates {candidates}: the pilot has no rows at it")
    asm = asmb.assemble(cfg, rows, {"gross": {"tc_bps": 0.0, "paths": "core"}}, lam=choice.lambda_f,
                        workers=workers)
    margins = margins_from(asm, sesoi)
    power = power_pilot(asm, margins, stats, confirmatory_days)
    in_fk = asm.masks["common"]
    folds = asm.folds
    report = {
        "stage": "E3 pilot", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "plan": {"version": "v2.4", "sections": "§3 (λ_f), §2 H3a (Δ_SR, δ), §7.4 (power pilot), §12"},
        "market": p["market"].get("name", p["market"]["source"]), "period": p["period"],
        "n_folds": asm.meta["n_folds"], "n_folds_Fk": int(in_fk.sum()),
        "fold_sets": asm.meta["fold_sets"],
        "lambda_f": choice.lambda_f,
        "lambda_rule": {
            "candidates": candidates, "min_share": min_share, "u": u, "floor_K": floor,
            "binding_share": {str(k): v for k, v in choice.shares.items()},
            "n_folds": {str(k): n for k, (_, n) in counts.items()},
            "n_binding": {str(k): b for k, (b, _) in counts.items()},
            "n_cells": {str(k): n for k, n in totals.items()},
            "lambda_f": choice.lambda_f, "fallback": choice.fallback,
        },
        "regime": {
            "share_binding": float(folds["binding"].to_numpy(dtype=bool)[in_fk].mean()) if in_fk.any() else None,
            "share_lower_bound": float(folds["lower_bound_active"].to_numpy(dtype=bool)[in_fk].mean())
            if in_fk.any() else None,
            "K_median": float(folds.loc[in_fk, "K"].median()) if in_fk.any() else None,
        },
        "margins": margins, "power": power,
        "yaml": {"lambda_f": choice.lambda_f,
                 "margins": {"delta_sr": margins["delta_sr"], "delta_var": margins["delta_var"]}},
    }
    report["section_12_draft"] = _draft_line(report)
    return data.json_safe(report)


def _draft_line(report: dict) -> str:
    m, pw = report["margins"], report["power"]
    mv = pw.get("MV", {})
    delta_var = m["delta_var"]
    parts = [
        f"{datetime.date.today().isoformat()} — [пилот] E3 (FF49, пилотный период): "
        f"λ_f = {report['lambda_f']:g}"
        + (" (резерв: доли связывающих фолдов меньше 70 %)" if report["lambda_rule"]["fallback"] else ""),
        f"σ* = {m['sigma_star']:.4f}, Δ_SR = {m['delta_sr']:.4f}",
        f"σ²_* = {m['sigma2_star']:.5f}, δ = " + ("не определена" if delta_var is None else f"{delta_var:.4f}"),
    ]
    if "halfwidth_confirmatory" in mv:
        parts.append(f"ρ(ранг 2, оптимум) = {mv['rho']:.4f}, полуширина 90 % ДИ пересчитана на "
                     f"подтверждение = {mv['halfwidth_confirmatory']:.4f} против Δ_SR "
                     f"({'исход «неопределённо» ожидаем' if mv['expected_inconclusive'] else 'не больше Δ_SR'})")
    parts.append("значения заданы правилами плана, не подбирались по исходам — H3a, H3b, H3c — согласовал: "
                 "(указать)")
    return "; ".join(parts[:-1]) + " — " + parts[-1]


# --------------------------------------------------------------------------------------------
# The registered stage
# --------------------------------------------------------------------------------------------


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else registry.REPO_ROOT / p


def _confirmatory_days(path: str | Path) -> int:
    """Days of the confirmatory period of a config: whole test windows from its calendar. The
    confirmatory configs have ``lambda_f: null`` until THIS stage has chosen it, so nothing here may
    ask for λ (:func:`cif.experiments.e3_walkforward.calendar_grid`)."""
    cfg = effective_config(load_config(_resolve(path)))
    _, grid = e3.calendar_grid(cfg)
    return grid.n_folds * grid.test_days


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.e3_pilot")
    parser.add_argument("--analysis-config", type=Path, required=True)
    parser.add_argument("--source", default="pilot", help="the key of the pilot run in sources:")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="exploratory report on a dirty tree (not for §12)")
    args = parser.parse_args(argv)

    acfg = yaml.safe_load(args.analysis_config.read_text(encoding="utf-8"))
    src = acfg["sources"][args.source]
    cfg_path, run_dir = _resolve(src["config"]), _resolve(src["run"])
    manifest = asmb.check_lineage(run_dir, cfg_path)
    cfg = effective_config(load_config(cfg_path))
    pilot_cfg = acfg.get("pilot") or {}
    if "confirmatory_config" not in pilot_cfg:
        raise PilotError("the analysis config needs pilot.confirmatory_config: the config of the "
                         "confirmatory FF49 run, for the length of the confirmation")
    n_conf = _confirmatory_days(pilot_cfg["confirmatory_config"])
    rows_path = run_dir / "results.jsonl"
    record = {
        "analysis_config": str(args.analysis_config), "source": args.source,
        "source_config": str(src["config"]), "source_run_id": run_dir.name,
        "source_config_hash": manifest["config_hash"], "source_git_sha": manifest.get("git_sha"),
        "rows_sha256": asmb._sha256(rows_path), "pilot": pilot_cfg, "stats": acfg["stats"],
        "margins": acfg.get("margins"),
    }
    with registry.start_run("E3", "pilot", record, runs_dir=args.runs_dir,
                            seeds=[int(acfg["stats"]["seed"])], inputs={"rows": rows_path},
                            require_clean=not args.allow_dirty) as run:
        report = pilot_report(cfg, asmb.load_rows(run_dir), acfg, confirmatory_days=n_conf,
                              workers=args.workers)
        report.update(source_run_id=run_dir.name, rows_sha256=record["rows_sha256"],
                      source_config_hash=manifest["config_hash"], source_git_sha=manifest.get("git_sha"))
        run.save_json(report, "pilot_report.json")
        run.manifest["mode"] = "pilot"
        print(f"{run.run_id}: lambda_f = {report['lambda_f']:g}, Δ_SR = {report['margins']['delta_sr']:.4f}, "
              f"δ = {report['margins']['delta_var']}")
        print("§12:", report["section_12_draft"])
        print("YAML:", yaml.safe_dump(report["yaml"], sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
