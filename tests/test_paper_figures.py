"""Figures and tables of the manuscript (scripts/paper_figures.py, scripts/paper_tables.py): reproducible, and the
numbers of the tables are the strings of the text's keys (scripts/paper_numbers.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    pn = _load("paper_numbers")
    sources = [pn.E2_H2, pn.E2_SECONDARY, pn.E2_CURVES, pn.E2_DELAY, pn.E2_M11, pn.E3_ANALYSIS]
    missing = [s for s in sources if not (ROOT / s).exists()]
    if missing:
        pytest.skip(f"analysis outputs absent: {missing}")
    return pn, _load("paper_figures"), _load("paper_tables")


def test_the_figures_are_pdf_and_the_same_bytes_on_a_rerun(modules, tmp_path):
    _, figures, _ = modules
    first = figures.main(tmp_path / "a")
    second = figures.main(tmp_path / "b")
    assert [p.name for p in first] == ["fig_a3_trivial.pdf", "fig_ttt_curves.pdf", "fig_m11_ratios.pdf",
                                       "fig_rq3_dose.pdf"]
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes().startswith(b"%PDF") and a.read_bytes() == b.read_bytes(), a.name


def test_the_committed_tables_are_what_the_sources_give(modules, tmp_path):
    _, _, tables = modules
    for path in tables.main(tmp_path):
        committed = ROOT / "paper" / "generated" / "tables" / path.name
        assert committed.exists() and committed.read_text(encoding="utf-8") == path.read_text(encoding="utf-8"), (
            f"{path.name}: rerun scripts/paper_tables.py")


def test_the_numbers_of_the_tables_are_the_strings_of_the_keys(modules, tmp_path):
    pn, _, tables = modules
    values = {key: value for _, _, entries in pn.generate().blocks for key, value in entries}
    text = {p.name: p.read_text(encoding="utf-8") for p in tables.main(tmp_path)}

    def interval(key):
        return f"{values[key]} [{values[key + '.ci.low']}, {values[key + '.ci.high']}]"

    for key in ("e2.h2", "e2.h2b", "e2.h2.noc5", "e2.h2b.noc5", "e2.h2b.own", "e2.h2.c2", "e2.h2b.c5"):
        assert interval(key) in text["tab_rq2.tex"], key
    for cell in ("c1", "c3", "c5"):
        assert values[f"e2.exact.{cell}.tprimal.median"] in text["tab_exact.tex"]
        assert values[f"e2.delay.{cell}.ls.median"] in text["tab_exact.tex"]
    assert f"& {values['e2.m11.t3.scip.median']} & {values['e2.m11.t3.oa.median']} &" in text["tab_m11.tex"]
    assert interval("e3.ff.gmv.h3a") in text["tab_rq3.tex"] and interval("e3.ff.mv.h3a") in text["tab_rq3.tex"]
