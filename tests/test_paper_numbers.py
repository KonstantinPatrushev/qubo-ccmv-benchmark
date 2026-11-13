"""The result numbers of the manuscript (scripts/paper_numbers.py → paper/generated/numbers.tex; golden rule 1):
formats, the structure of the file (a source above every key, unique keys) and spot checks against the registered
analyses."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("paper_numbers", ROOT / "scripts" / "paper_numbers.py")
pn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pn)


def test_formats():
    assert pn.sig(0.164132) == "0.164" and pn.sig(2.621611) == "2.62" and pn.sig(12.2838) == "12.3"
    assert pn.sig(1.0) == "1.00" and pn.sig(0.0) == "0"
    assert pn.sig(-0.00364) == r"\ensuremath{-}0.00364"
    assert pn.sci(2.9491979909423982e-05) == r"\ensuremath{2.9\times10^{-5}}"
    assert pn.pval(0.0012) == "0.0012" and pn.pval(0.0002) == r"\ensuremath{2.0\times10^{-4}}"
    assert pn.pval(5.4e-20) == r"\ensuremath{5.4\times10^{-20}}"
    assert pn.pct(0.7323) == "73" and pn.pct(0.125, 1) == "12.5"
    assert pn.count(13356) == r"13\,356" and pn.count(636) == "636"
    assert pn.bp(0.0022858) == "22.9"
    assert pn.gap(1.1e-13) == "0" and pn.gap(-2e-12) == "0" and pn.gap(0.0972) == "0.097"


@pytest.fixture(scope="module")
def numbers():
    sources = [pn.E1_H1A, pn.E2_H1C, pn.E2_H2, pn.E2_SECONDARY, pn.E2_CURVES, pn.E2_DELAY, pn.E3_ANALYSIS]
    missing = [s for s in sources if not (ROOT / s).exists()]
    if missing:
        pytest.skip(f"analysis outputs absent: {missing}")
    return pn.generate()


def test_every_key_is_unique_well_formed_and_has_its_source_above_it(numbers):
    keys = numbers.keys()
    assert len(keys) == len(set(keys)) and len(keys) > 100
    assert all(re.fullmatch(r"[a-z0-9.]+", k) for k in keys)
    lines = numbers.text().splitlines()
    defs = [i for i, line in enumerate(lines) if line.startswith(r"\expandafter\def")]
    assert len(defs) == len(keys)
    assert all(lines[i - 1].startswith("% src: ") for i in defs)
    assert "run 20261013-102613-E2-bench_factor-623b58" in numbers.text()


def test_spot_checks_against_the_registered_analyses(numbers):
    values = {key: value for _, _, entries in numbers.blocks for key, value in entries}
    assert values["e2.h2"] == "0.164" and values["e2.h2b"] == "2.62"
    assert (values["e2.h2b.noc5"], values["e2.h2b.noc5.ci.low"], values["e2.h2b.noc5.ci.high"]) == ("1.85", "1.26",
                                                                                                     "2.76")
    assert values["e2.n.uncertified"] == "23" and values["e2.c5.uncertified"] == "9" and values["e2.c5.nports"] == "3"
    assert values["e1.h1a.p"] == r"\ensuremath{2.9\times10^{-5}}"
    assert values["e3.ff.gmv.h3a.p.tost.holm"] == r"\ensuremath{2.0\times10^{-4}}"
    assert (values["e3.ff.gmv.h3b.slope"], values["e3.ff.mv.h3b.slope"]) == ("0.0118", r"\ensuremath{4.70\times10^{-5}}")
    # the percentile interval of the plan (§2 H3b), as the analysis decides significance on it
    assert (values["e3.ff.gmv.h3b.slope.ci.low"], values["e3.ff.gmv.h3b.slope.ci.high"]) == ("0.00478", "0.0193")
    assert values["e2.curves.1049.qubo.pct"] == "0"
    if (ROOT / pn.E2_M11).exists():
        assert (values["e2.m11.t3.scip.median"], values["e2.m11.t4.scip.median"], values["e2.m11.t3.oa.median"]) == (
            "262", "14", "3.1")
        assert values["e2.m11.t4.scip.nours"] == "15" and values["e2.m11.t4.scip.npub"] == "14"


def test_the_range_of_trivial_shares_over_the_factor_cells_is_their_minimum_and_maximum(numbers):
    # the abstract and the introduction quote the factor cells C1–C3 as one range
    values = {key: value for _, _, entries in numbers.blocks for key, value in entries}
    shares = [int(values[f"e2.a3.{c}.pct"]) for c in ("c1", "c2", "c3")]
    assert values["e2.a3.factor.min.pct"] == str(min(shares)) == "73"
    assert values["e2.a3.factor.max.pct"] == str(max(shares)) == "80"


def test_the_outputs_added_for_the_review_r1_are_read_from_their_registered_sources(numbers):
    """Planned outputs that the first draft left out and the post hoc descriptions of §12, 2026-10-19 (review r1)."""
    values = {key: value for _, _, entries in numbers.blocks for key, value in entries}
    # RQ1: the pools of the real samplers (distinct subsets per call) and the share of instances above the target
    assert (values["e1.pool.tabu.unique"], values["e1.pool.swap.unique"]) == ("10", "3")
    assert (values["e1.pipe.swap.m200.nabove"], values["e1.pipe.ninstances"], values["e1.eform.m50.all.nexceed"]) == (
        "25", "160", "6")
    # A3: the gap of the common start in bp of CE, the undecided dense instances, the non-binding OR-Library points
    assert values["e2.a3.c2.gstart.max.bp"] == "16" and values["e2.a3.d1.n200.nundecided"] == "17"
    assert values["e2.a3.c5.nnonbinding"] == "55"
    # the distribution over the instances and the runs that never reached the target
    assert values["e2.inst.c2.rb.median"] == "0.47" and values["e2.inst.c2.rb.below.pct"] == "68"
    assert (values["e2.censor.c4.qubo"], values["e2.censor.c4.ls"]) == ("24", "0")
    # post hoc: the penalty samplers in the place of the representative; the first own portfolio
    assert (values["e2.sa.h2"], values["e2.tabu.h2"]) == ("7.74", "3.24")
    assert values["e2.first.c1.qubo.notworse.pct"] == "88" and values["e2.first.notlater.max.pct"] == "0"
    # tuning of the exact class and the times behind the ratios of M11
    assert (values["e2.tune.exact.n100.nconfigs"], values["e2.tune.exact.n200.nconfigs"]) == ("7", "5")
    # RQ3: the dose of rank 2 and the regime of the lower bound of K
    assert values["e3.ff.mv.dose2.median"] == "6.1" and values["e3.ff.regime.lb.pct"] == "96"
    assert values["e3.us.regime.lb.pct"] == "100" and values["e3.ff.mv.block"] == "42"


def test_the_sensitivity_to_the_exact_baseline_is_read_from_its_registered_analyses(numbers):
    """§12, 2026-10-19: the exact class re-run as SCIP with aggressive primal heuristics and as OA in every cell; the
    representative on the same inputs gives back the confirmatory estimate (the references did not move)."""
    values = {key: value for _, _, entries in numbers.blocks for key, value in entries}
    if "e2.exactsens.base.h2" not in values:
        pytest.skip("the analyses of the sensitivity are absent")
    assert (values["e2.exactsens.base.h2"], values["e2.exactsens.base.h2.ci.high"]) == (values["e2.h2"],
                                                                                        values["e2.h2.ci.high"])
    assert (values["e2.exactsens.scipaggr.h2"], values["e2.exactsens.oa.h2"]) == ("0.108", "0.128")
    assert values["e2.exactsens.oa.pirel.c3"] == "1.0"  # in C3 the representative is OA itself


def test_the_committed_file_is_what_the_sources_give(numbers):
    assert pn.OUT.exists(), "run scripts/paper_numbers.py"
    assert pn.OUT.read_text(encoding="utf-8") == numbers.text() + "\n", "rerun scripts/paper_numbers.py"
