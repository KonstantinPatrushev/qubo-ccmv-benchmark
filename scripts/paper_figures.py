"""Figures of the manuscript (``paper/generated/figures/*.pdf``), drawn only from the registered analyses that
``scripts/paper_numbers.py`` reads (golden rule 1: no number is typed in). Vector PDF, fonts embedded as TrueType, no
creation date, so that a rerun on the same inputs gives the same bytes.

    python scripts/paper_figures.py

* ``fig_a3_trivial.pdf`` — the share of trivial test instances per cell with Wilson 95 % intervals (``h2.json``, A3);
* ``fig_ttt_curves.pdf`` — time-to-target curves of the three representatives on the calibration instances of the
  stage ``ttt`` (``ttt_curve_points.csv``, ``ttt_curve_methods.csv``; τ = 1e-3);
* ``fig_m11_ratios.pdf`` — the ratio of our time to the best published time per instance of Tables 3 and 4 of
  Bertsimas & Cory-Wright (``m11_20261018/cells.csv``, ``m11.json``);
* ``fig_rq3_dose.pdf`` — RQ3: the out-of-sample difference of the path of rank r against its mean in-sample dose on
  FF49, with the equivalence margins and the fitted H3b line (E3 analysis tables).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_numbers as pn

from cif.stats.rq1 import wilson_interval

ROOT = pn.ROOT
OUT = ROOT / "paper" / "generated" / "figures"
W = 6.3  # inches: the text width of the preprint layout
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 8, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})
METHOD = {"scip": ("exact (SCIP)", "#3f5f8f"), "swap_sa": ("QUBO pipeline (swap-SA)", "#c25a22"),
          "ls": ("local search", "#23865f")}
CELL = {"C1": "C1  factor, N = 100", "C2": "C2  factor, N = 200", "C3": "C3  factor, N = 400",
        "C4": "C4  S&P 500 bootstrap, N = 200", "C5": "C5  OR-Library", "D1_100": "D1  dense, N = 100",
        "D1_200": "D1  dense, N = 200"}


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def fig_a3_trivial(out_dir: Path) -> Path:
    h2 = json.loads((ROOT / pn.E2_H2 / "h2.json").read_text(encoding="utf-8"))
    cells = [c for c in CELL if c in h2["a3"]["cells"]]
    fig, ax = plt.subplots(figsize=(W, 0.32 * len(cells) + 0.6))
    for i, cell in enumerate(cells):
        part = h2["a3"]["cells"][cell]
        # the share among the decided instances: an undetermined one (no decision within the limit of the screening)
        # is shown apart, not counted as non-trivial (review r1, M8); in C1-C5 nothing is undetermined
        decided = part["n_trivial"] + part["n_nontrivial"]
        low, high = wilson_interval(part["n_trivial"], decided, 0.95)
        share = 100 * part["n_trivial"] / decided
        ax.barh(i, share, height=0.6, color="#7f97b8" if cell.startswith("C") else "#c9d3e0", lw=0)
        ax.errorbar(share, i, xerr=[[share - 100 * low], [100 * high - share]], fmt="none", ecolor="black",
                    elinewidth=0.8, capsize=2)
        label = f"{part['n_trivial']} of {decided}"
        if part["n_undecided"]:
            label += f" ({part['n_undecided']} undetermined)"
        ax.text(min(100 * high + 1.5, 101), i, label, va="center", fontsize=7.5)
    ax.set_yticks(range(len(cells)), [CELL[c] for c in cells])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("trivial instances: the common start is already within the target (%)")
    return _save(fig, out_dir, "fig_a3_trivial.pdf")


def fig_ttt_curves(out_dir: Path) -> Path:
    points = pd.read_csv(ROOT / pn.E2_CURVES / "ttt_curve_points.csv")
    methods = pd.read_csv(ROOT / pn.E2_CURVES / "ttt_curve_methods.csv")
    points, methods = points[points.tau == 1e-3], methods[methods.tau == 1e-3]
    instances = sorted(points.instance_id.unique())
    fig, axes = plt.subplots(1, len(instances), figsize=(W, 2.2), sharey=True)
    lo, hi = 0.03, 60.0
    for ax, iid in zip(np.atleast_1d(axes), instances, strict=True):
        for role in ("exact_rep", "qubo_rep", "ls"):
            m = methods[(methods.instance_id == iid) & (methods.role == role)].iloc[0]
            label, color = METHOD[m.method]
            p = points[(points.instance_id == iid) & (points.role == role)].sort_values("i")
            xs = np.concatenate(([lo], p.time_s.to_numpy(), [hi]))
            ys = np.concatenate(([0.0], p.p.to_numpy(), [p.p.to_numpy()[-1] if len(p) else 0.0]))
            ax.step(xs, ys, where="post", color=color, lw=1.4, label=label)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.1, 1, 10, 60], ["0.1", "1", "10", "60"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"instance {iid}", fontsize=8)
        ax.set_xlabel("time from the start of the run (s)")
    np.atleast_1d(axes)[0].set_ylabel("share of the 100 runs at the target")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out_dir, "fig_ttt_curves.pdf")


def fig_m11_ratios(out_dir: Path) -> Path:
    cells = pd.read_csv(ROOT / pn.E2_M11 / "cells.csv")
    report = json.loads((ROOT / pn.E2_M11 / "m11.json").read_text(encoding="utf-8"))
    rows = [("T3", "scip", "BCW-T3, SCIP", "#3f5f8f"), ("T3", "oa", "BCW-T3, OA (HiGHS)", "#6f8fbf"),
            ("T4", "scip", "BCW-T4, SCIP", "#3f5f8f")]
    fig, ax = plt.subplots(figsize=(W, 1.7))
    for i, (table, view, _label, color) in enumerate(rows):
        part = cells[cells.table == table]
        ratios = part[f"{view}_ratio"].to_numpy(dtype=float)
        bounds = part[f"{view}_bound"].astype(str).to_numpy()
        offsets = (part.k.map({5: -0.18, 10: 0.0, 20: 0.18})).to_numpy()
        for r, b, dy in zip(ratios, bounds, offsets, strict=True):
            hollow = b in ("upper", "lower")
            ax.plot(r, i + dy, "o", ms=4, mfc="white" if hollow else color, mec=color, mew=1)
        median = report["tables"][table]["summary"][view]["median_ratio"]
        ax.plot([median, median], [i - 0.32, i + 0.32], color="black", lw=1.6)
    ax.axvline(10, color="#9a5b00", ls="--", lw=1)
    ax.axvline(1, color="0.5", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e5)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("our time / best published time (the rule of the plan: median above 10, dashed)")
    return _save(fig, out_dir, "fig_m11_ratios.pdf")


def fig_rq3_dose(out_dir: Path) -> Path:
    tables = ROOT / pn.E3_ANALYSIS / "tables"
    per = pd.read_csv(tables / "h3b_per_rank.csv")
    h3a = pd.read_csv(tables / "h3a.csv")
    h3b = pd.read_csv(tables / "h3b.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5))
    panels = (("ff49-MV", "FF49, mean–variance: difference of Sharpe ratios"),
              ("ff49-GMV", "FF49, minimum variance: log variance ratio"))
    for ax, (key, title) in zip(axes, panels, strict=True):
        r = per[per.key == key].sort_values("dose_bp")
        m = h3a[(h3a.key == key) & (h3a.variant == "main")].iloc[0]
        fit = h3b[(h3b.key == key) & (h3b.variant == "main") & (h3b.fit == "log_dose")].iloc[0]
        ax.axhspan(m.margin_low, m.margin_high, color="#e3e6ea", lw=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.errorbar(r.dose_bp, r.delta, yerr=[r.delta - r.ci_low, r.ci_high - r.delta], fmt="o", color="#3f5f8f",
                    ms=3.5, capsize=2, lw=0.9)
        xs = np.geomspace(r.dose_bp.min() * 0.9, r.dose_bp.max() * 1.1, 50)
        ax.plot(xs, fit.intercept + fit.slope * np.log(xs), color="#c25a22", lw=1)
        for _, row in r.iterrows():
            ax.annotate(f"r = {int(row['rank'])}", (row.dose_bp, row.ci_high), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7)
        ax.set_xscale("log")
        ticks = [t for t in (5, 10, 20, 40, 70) if r.dose_bp.min() * 0.8 <= t <= r.dose_bp.max() * 1.2]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("mean in-sample dose of rank r (bp of certainty equivalent)")
        ax.set_title(title)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_dose.pdf")


FIGURES = (fig_a3_trivial, fig_ttt_curves, fig_m11_ratios, fig_rq3_dose)


def main(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir) if out_dir is not None else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    written = [figure(out_dir) for figure in FIGURES]
    print(", ".join(p.name for p in written))
    return written


if __name__ == "__main__":
    main()
