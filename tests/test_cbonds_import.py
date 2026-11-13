"""Tests for ``scripts/data/import_cbonds_dividends.py``.

Vendor data enters ``data/raw`` only through a script (CLAUDE.md, rule 3). The script copies the
exports byte for byte, refuses anything whose SHA-256, header or row count differs from what was
inspected, and leaves a README and a provenance record. Everything here runs on synthetic
workbooks in a temporary directory.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cif.data import cbonds
from cif.data.provenance import verify_provenance
from tests.xlsx_builder import write_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "import_cbonds_dividends.py"
D = dt.datetime

# What the task brief fixed for the two files of 2026-10-08.
BRIEF_SHA = {
    "dividend_calendar_export08_10_2026.xlsx":
        "7287279f4104475e92fc268414b14dcc978298f0e8015a48b44ef421fc99624b",
    "dividend_calendar_export08_10_2026 (1).xlsx":
        "08366dc3232081619ad5e8fb5b16ec20b823aa0f57ef5d3e84cd07f2485389a5",
}


@pytest.fixture(scope="module")
def imp():
    spec = importlib.util.spec_from_file_location("import_cbonds_dividends", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["import_cbonds_dividends"] = module
    spec.loader.exec_module(module)
    return module


EX, RECORD, LAST_BUY = D(2019, 6, 11), D(2019, 6, 13), D(2019, 6, 10)


def _row(ticker="SBER", amount=16.0, ccy="RUB", ex=EX, record=RECORD):
    return ["Сбербанк России, акция обыкновенная", "Московская Биржа", ticker, "Банки", amount, ccy, "",
            LAST_BUY, ex, record, "2018", "год"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def exports(tmp_path, imp):
    """Two small Cbonds-shaped exports and the specs that describe them."""
    src = tmp_path / "downloads"
    src.mkdir()
    first = write_table(src / "export_a.xlsx", cbonds.CBONDS_HEADER, [
        _row(), _row("GAZP", 7.2, ex=D(2014, 7, 16), record=D(2014, 7, 17))])
    second = write_table(src / "export_b (1).xlsx", cbonds.CBONDS_HEADER, [
        _row(), _row("GAZP", 7.2, ex=D(2014, 7, 16), record=D(2014, 7, 17)),
        _row("6240", 24.0, "JPY", ex=D(2026, 9, 29), record=D(2026, 9, 30))])
    specs = (
        imp.ExportSpec("export_a.xlsx", "clean_ru.xlsx", _sha(first), 2,
                       {"ex_date_from": "2013-01-28", "ex_date_to": "2026-10-08",
                        "exchange": "Московская Биржа", "country": "Россия"}, "RU only"),
        imp.ExportSpec("export_b (1).xlsx", "clean_all.xlsx", _sha(second), 3,
                       {"ex_date_from": "2013-01-28", "ex_date_to": "2026-10-08",
                        "exchange": "Московская Биржа", "country": "не задана"}, "adds JPY"),
    )
    return src, tmp_path / "raw" / "cbonds", specs


GIT = {"sha": "a" * 40, "branch": "stream/E0-engine", "dirty": False}
NOW = D(2026, 10, 8, 15, 0, 0, tzinfo=dt.UTC)


def _run(imp, exports, **kw):
    src, dest, specs = exports
    return imp.import_exports(specs, src, dest, git=kw.pop("git", GIT), run_by="tester", now=NOW, **kw)


# --------------------------------------------------------------------------------------------
# The constants of the real import
# --------------------------------------------------------------------------------------------


def test_the_real_specs_pin_the_hashes_and_counts_of_the_brief(imp):
    by_name = {s.original_name: s for s in imp.EXPORTS}
    assert {n: s.sha256 for n, s in by_name.items()} == BRIEF_SHA
    assert by_name["dividend_calendar_export08_10_2026.xlsx"].rows == 2149
    assert by_name["dividend_calendar_export08_10_2026 (1).xlsx"].rows == 2174
    assert [s.clean_name for s in imp.EXPORTS] == [
        "cbonds_dividend_calendar_moex_ru_20261008.xlsx",
        "cbonds_dividend_calendar_moex_all_20261008.xlsx",
    ]
    ru, allc = imp.EXPORTS
    assert ru.filters["country"] == "Россия" and "country" in allc.filters
    assert ru.filters["exchange"] == "Московская Биржа"
    assert ru.filters["ex_date_from"] == "2013-01-28" and ru.filters["ex_date_to"] == "2026-10-08"


# --------------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------------


def test_files_are_copied_verbatim_under_clean_names(imp, exports):
    src, dest, _ = exports
    result = _run(imp, exports)
    assert result["status"] == "imported"
    assert (dest / "clean_ru.xlsx").read_bytes() == (src / "export_a.xlsx").read_bytes()
    assert (dest / "clean_all.xlsx").read_bytes() == (src / "export_b (1).xlsx").read_bytes()
    assert not (dest / "export_a.xlsx").exists()


def test_provenance_records_the_facts_of_every_file(imp, exports):
    _, dest, specs = exports
    _run(imp, exports)
    prov = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    assert prov["dataset"] == "cbonds_dividends_raw"
    assert prov["imported_at"] == "2026-10-08T15:00:00+00:00"
    assert prov["importer"]["run_by"] == "tester"
    assert prov["importer"]["git"] == GIT
    assert prov["importer"]["script"] == "scripts/data/import_cbonds_dividends.py"
    assert prov["importer"]["script_sha256"] == _sha(SCRIPT)
    rec = prov["files"]["clean_all.xlsx"]
    assert rec["sha256"] == specs[1].sha256
    assert rec["bytes"] == (dest / "clean_all.xlsx").stat().st_size
    assert rec["original_name"] == "export_b (1).xlsx"
    assert rec["rows"] == 3
    assert rec["ex_date_min"] == "2014-07-16" and rec["ex_date_max"] == "2026-09-29"
    assert rec["currencies"] == {"RUB": 2, "JPY": 1}
    assert rec["filters"]["country"] == "не задана"
    assert rec["source_mtime"].endswith("+00:00")
    assert prov["source"]["vendor"].startswith("Cbonds")


def test_the_provenance_is_readable_by_the_repository_verifier(imp, exports):
    _, dest, _ = exports
    _run(imp, exports)
    report = verify_provenance(dest / "provenance.json")
    assert report["ok"] and report["n_files"] == 2 and report["n_ok"] == 2


def test_readme_names_source_terms_filters_and_quirks(imp, exports):
    _, dest, specs = exports
    _run(imp, exports)
    text = (dest / "README.txt").read_text(encoding="utf-8")
    for needle in ("Cbonds", "Investfunds", "внутреннего использования", "export_b (1).xlsx",
                   "clean_ru.xlsx", specs[0].sha256, specs[1].sha256, "Россия", "2013-01-28",
                   "ISIN", "текущая", "QIWI", "JPY", "сплит", "Тикер"):
        assert needle in text, needle


def test_raw_files_are_made_read_only(imp, exports):
    _, dest, _ = exports
    _run(imp, exports)
    assert not (dest / "clean_ru.xlsx").stat().st_mode & 0o222


def test_a_second_run_with_identical_files_changes_nothing(imp, exports):
    _, dest, _ = exports
    _run(imp, exports)
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in dest.iterdir()}
    result = _run(imp, exports)
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in dest.iterdir()}
    assert result["status"] == "unchanged"
    assert before == after


# --------------------------------------------------------------------------------------------
# Refusals: nothing is written unless everything checks
# --------------------------------------------------------------------------------------------


def test_a_wrong_hash_stops_the_import_before_anything_is_written(imp, exports):
    src, dest, specs = exports
    bad = (specs[0], imp.ExportSpec(**{**specs[1].__dict__, "sha256": "0" * 64}))
    with pytest.raises(imp.ImportRefused, match="SHA-256"):
        imp.import_exports(bad, src, dest, git=GIT, run_by="tester", now=NOW)
    assert not dest.exists() or not any(dest.iterdir())


def test_a_wrong_row_count_stops_the_import(imp, exports):
    src, dest, specs = exports
    bad = (imp.ExportSpec(**{**specs[0].__dict__, "rows": 99}), specs[1])
    with pytest.raises(imp.ImportRefused, match="rows"):
        imp.import_exports(bad, src, dest, git=GIT, run_by="tester", now=NOW)
    assert not dest.exists() or not any(dest.iterdir())


def test_a_changed_header_stops_the_import(imp, tmp_path):
    header = list(cbonds.CBONDS_HEADER)
    header[4] = "Размер"
    src = tmp_path / "d"
    src.mkdir()
    f = write_table(src / "a.xlsx", header, [_row()])
    spec = imp.ExportSpec("a.xlsx", "c.xlsx", _sha(f), 1, {}, "")
    with pytest.raises(imp.ImportRefused, match="Размер"):
        imp.import_exports((spec,), src, tmp_path / "raw", git=GIT, run_by="tester", now=NOW)


def test_a_missing_source_file_is_reported_with_its_name(imp, exports):
    src, dest, specs = exports
    (src / "export_a.xlsx").unlink()
    with pytest.raises(imp.ImportRefused, match=r"export_a\.xlsx"):
        imp.import_exports(specs, src, dest, git=GIT, run_by="tester", now=NOW)


def test_an_existing_different_file_is_never_overwritten(imp, exports):
    _, dest, _ = exports
    dest.mkdir(parents=True)
    (dest / "clean_ru.xlsx").write_bytes(b"something else")
    with pytest.raises(imp.ImportRefused, match="overwrite"):
        _run(imp, exports)
    assert (dest / "clean_ru.xlsx").read_bytes() == b"something else"


def test_a_dirty_tree_is_refused_unless_allowed(imp, exports):
    _, dest, _ = exports
    dirty = {**GIT, "dirty": True}
    with pytest.raises(imp.ImportRefused, match="clean"):
        _run(imp, exports, git=dirty)
    assert not dest.exists()
    result = _run(imp, exports, git=dirty, allow_dirty=True)
    assert result["status"] == "imported"
    prov = json.loads((dest / "provenance.json").read_text(encoding="utf-8"))
    assert prov["importer"]["git"]["dirty"] is True


def test_dry_run_checks_everything_and_writes_nothing(imp, exports):
    _, dest, _ = exports
    result = _run(imp, exports, dry_run=True)
    assert result["status"] == "dry-run"
    assert [f["rows"] for f in result["files"]] == [2, 3]
    assert not dest.exists()


def test_main_runs_with_explicit_arguments(imp, exports, capsys):
    src, dest, specs = exports
    code = imp.main(["--src-dir", str(src), "--dest-dir", str(dest), "--allow-dirty"], specs=specs)
    assert code == 0
    assert (dest / "provenance.json").exists()
    assert "imported" in capsys.readouterr().out
