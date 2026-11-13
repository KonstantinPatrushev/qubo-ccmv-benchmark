"""tools/lit/verify_bib.py: the normalisation that compares a BibTeX entry with the registry reads LaTeX accents."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_bib", ROOT / "tools" / "lit" / "verify_bib.py")
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)


@pytest.mark.parametrize("bib, registry", [
    (r"Akt{\"u}rk, M. Selim and Atamt{\"u}rk, Alper", "Aktürk"),
    (r"G{\"u}nl{\"u}k, Oktay and Linderoth, Jeff", "Günlük"),
    (r"Rubio-Garc{\'i}a, {\'A}lvaro and Porras, Diego", "Rubio-García"),
    (r"R{\o}nnow, Troels F.", "Rønnow"),
    (r"K{\i}l{\i}{\c{c}}, Mehmet", "Kılıç"),
    (r"{\v{S}}imek, Petr", "Šimek"),
    ("Aktürk, M. Selim", "Aktürk"),
])
def test_the_first_surname_reads_latex_accents_as_the_registry_letters(bib, registry):
    assert vb.first_surname_bib(bib) == vb.norm_title(registry)


def test_titles_keep_their_words_around_latex_commands():
    assert vb.norm_title(r"{TTT} plots: a {Perl} program") == "ttt plots a perl program"
    assert vb.norm_title(r"The \emph{scip} optimization suite") == "the scip optimization suite"


# A DOI printed in an article but not (yet) registered: the publisher's own page may state it, with the title. That is
# reported as WARN, never OK, and only when the page states the same DOI and the title matches (2026-11-04: the DOIs
# of Proceedings of ISP RAS for 2025-2026 do not resolve).

PAGE = ('<meta name="DC.Identifier.DOI" content="10.15514/ISPRAS-2026-38(4)-29"/>'
        '<meta name="citation_author" xml:lang="en" lang="en" content="Sergey Michailovich AVDOSHIN"/>'
        '<meta name="citation_title" xml:lang="en" lang="en" content="Discrete Markowitz Portfolio Optimization: A Study"/>'
        '<meta name="citation_title" xml:lang="ru" lang="ru" content="Дискретная оптимизация портфеля Марковица"/>'
        '<meta name="citation_date" content="2026/10/02"/>'
        '<meta name="citation_doi" content="10.15514/ISPRAS-2026-38(4)-29"/>')
ENTRY = {"ID": "avdoshin2026discrete", "title": "Discrete Markowitz Portfolio Optimization: A Study",
         "author": "Avdoshin, Sergey M. and Patrushev, Konstantin A.", "year": "2026",
         "doi": "10.15514/ISPRAS-2026-38(4)-29", "url": "https://journal.example/article/2233"}


def test_an_unregistered_doi_stated_by_the_publisher_page_with_the_title_is_a_warning(monkeypatch):
    monkeypatch.setattr(vb, "resolve_doi", lambda doi: None)
    monkeypatch.setattr(vb, "http_get", lambda url, **kw: PAGE)
    res = vb.verify_entry(dict(ENTRY))
    assert res["status"] == "WARN" and res["sim"] == 1.0
    assert any("not registered" in n and "journal.example" in n for n in res["notes"])


def test_the_publisher_page_must_state_the_same_doi(monkeypatch):
    monkeypatch.setattr(vb, "resolve_doi", lambda doi: None)
    monkeypatch.setattr(vb, "http_get", lambda url, **kw: PAGE.replace("(4)-29", "(4)-30"))
    assert vb.verify_entry(dict(ENTRY))["status"] == "FAIL"


def test_a_title_the_publisher_page_does_not_state_fails(monkeypatch):
    monkeypatch.setattr(vb, "resolve_doi", lambda doi: None)
    monkeypatch.setattr(vb, "http_get", lambda url, **kw: PAGE)
    assert vb.verify_entry({**ENTRY, "title": "A Different Paper on Something Else"})["status"] == "FAIL"


def test_without_a_url_an_unregistered_doi_still_fails(monkeypatch):
    monkeypatch.setattr(vb, "resolve_doi", lambda doi: None)
    monkeypatch.setattr(vb, "http_get", lambda *a, **kw: pytest.fail("no page to read"))
    entry = {k: v for k, v in ENTRY.items() if k != "url"}
    assert vb.verify_entry(entry)["status"] == "FAIL"


def test_a_registered_doi_is_checked_against_the_registry_not_the_page(monkeypatch):
    registry = {"title": ENTRY["title"], "year": 2026, "surname": "avdoshin", "via": "doi"}
    monkeypatch.setattr(vb, "resolve_doi", lambda doi: registry)
    monkeypatch.setattr(vb, "http_get", lambda *a, **kw: pytest.fail("the page is not read"))
    assert vb.verify_entry(dict(ENTRY))["status"] == "OK"
