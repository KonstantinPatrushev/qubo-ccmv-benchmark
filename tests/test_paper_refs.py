"""The manuscript bibliography (scripts/paper_refs.py): every cited key comes from the verified bibliographies of the
literature reviews, and the committed paper/refs.bib is what the citations give (golden rule 2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("paper_refs", ROOT / "scripts" / "paper_refs.py")
refs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(refs)


def test_the_cited_keys_are_found_in_a_verified_bibliography_and_copied_unchanged_but_for_format():
    entries = refs.verified_entries()
    text = refs.build()
    for key in refs.cited_keys([ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))]):
        assert key in entries and f"{{{key}," in text
    assert "&amp;" not in text and "month" not in text


def test_the_committed_bibliography_is_what_the_citations_give():
    assert (ROOT / "paper" / "refs.bib").read_text(encoding="utf-8") == refs.build(), "rerun scripts/paper_refs.py"


def test_cite_commands_with_options_and_comments_are_read(tmp_path):
    section = tmp_path / "x.tex"
    section.write_text("see \\cite{a2001,b2002} and \\citep[p.~3]{c2003} % \\cite{commented2004}\n", encoding="utf-8")
    assert refs.cited_keys([section]) == ["a2001", "b2002", "c2003"]


def test_a_citation_outside_the_verified_bibliographies_stops_the_script(monkeypatch):
    monkeypatch.setattr(refs, "cited_keys", lambda paths: ["nosuchkey2099"])
    with pytest.raises(SystemExit, match="nosuchkey2099"):
        refs.build()


def test_titles_keep_acronyms_and_proper_nouns_under_the_sentence_case_of_the_style():
    # elsarticle-num lowers every title to sentence case: acronyms, CamelCase names and proper nouns need braces
    entry = ("@article{x, title={A computational comparison: SOCP vs. cutting planes in PySCIPOpt for Sparse Markowitz "
             "portfolios with QUBO-QAOA and Cardinality-Constrained Mean-Variance models, part I}, year={2009} }")
    out = refs.normalise(entry)
    for kept in ("{SOCP}", "{PySCIPOpt}", "{Markowitz}", "{QUBO}-{QAOA}", "part {I}"):
        assert kept in out, kept
    for lowered in ("Sparse", "Cardinality-Constrained", "Mean-Variance"):
        assert lowered in out and "{" + lowered not in out, lowered
    assert "title={A computational" in out and "{A}" not in out  # the article "A" is left to the style


def test_braced_parts_of_a_title_are_not_wrapped_twice():
    entry = "@article{x, title={{TTT} plots: a {Perl} program}, year={2006} }"
    assert refs.normalise(entry) == entry


def test_html_italics_of_crossref_titles_become_latex():
    entry = ("@article{x, title={Optimal Versus Naive Diversification: How Inefficient is the 1/\n"
             "                    <i>N</i>\n                  Portfolio Strategy?}, year={2009} }")
    out = refs.normalise(entry)
    assert "<i>" not in out and "How Inefficient is the 1/{$N$} Portfolio Strategy?" in out


def test_no_acronym_in_a_title_of_the_built_bibliography_is_left_to_the_case_change_of_the_style():
    import re
    for title in re.findall(r"\btitle=\{((?:[^{}]|\{[^{}]*\})*)\}", refs.build()):
        top = re.sub(r"\{[^{}]*\}", " ", title)  # what the style can still change
        assert not re.findall(r"[^\W\d_]*[A-Z][^\W\d_]*[A-Z][^\W\d_]*", top), title
        assert "<i>" not in title, title
