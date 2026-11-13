"""Tests for ``cif.data.xlsx``: a standard-library reader of ``.xlsx`` workbooks.

The reader exists because ``openpyxl`` is not part of the ``cif`` environment (installing packages
needs the user's permission) and the Cbonds dividend calendar arrives as ``.xlsx``. The synthetic
workbooks below are written byte by byte with :mod:`zipfile`, so the tests do not assume how the
reader builds anything; the last block reads the real Cbonds exports when they have been imported
(``scripts/data/import_cbonds_dividends.py``).
"""

from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import pytest

from cif.data import xlsx
from tests.xlsx_builder import (
    NS,
    RELS,
    WORKBOOK,
    build_xlsx,
    shared_xml,
    sheet_xml,
    styles_xml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CBONDS_DIR = REPO_ROOT / "data" / "raw" / "cbonds_dividends"
CBONDS_RU = CBONDS_DIR / "cbonds_dividend_calendar_moex_ru_20261008.xlsx"
CBONDS_ALL = CBONDS_DIR / "cbonds_dividend_calendar_moex_all_20261008.xlsx"

needs_cbonds = pytest.mark.skipif(
    not (CBONDS_RU.exists() and CBONDS_ALL.exists()),
    reason="Cbonds exports are not imported (scripts/data/import_cbonds_dividends.py)",
)

_styles = styles_xml
_shared = shared_xml
_sheet = sheet_xml


# --------------------------------------------------------------------------------------------
# Cell types
# --------------------------------------------------------------------------------------------


def test_shared_strings_numbers_and_dates(tmp_path):
    shared = _shared(["Name", "Date", "Value", "Alpha"])
    rows = (
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" s="1"><v>41311</v></c>'
        '<c r="C2"><v>0.23000000000000001</v></c></row>'
    )
    path = build_xlsx(tmp_path / "a.xlsx", {"S": _sheet(rows)}, shared=shared, styles=_styles([0, 14]))
    got = xlsx.read_rows(path)
    assert got[0] == ["Name", "Date", "Value"]
    assert got[1][0] == "Alpha"
    assert got[1][1] == dt.datetime(2013, 2, 6)
    assert got[1][2] == pytest.approx(0.23)


def test_integers_stay_integers_and_decimals_floats(tmp_path):
    rows = '<row r="1"><c r="A1"><v>20</v></c><c r="B1"><v>1E-3</v></c><c r="C1"><v>-7</v></c></row>'
    path = build_xlsx(tmp_path / "n.xlsx", {"S": _sheet(rows)})
    (row,) = xlsx.read_rows(path)
    assert row[0] == 20 and isinstance(row[0], int)
    assert row[1] == pytest.approx(0.001) and isinstance(row[1], float)
    assert row[2] == -7


def test_sparse_cells_and_rows_are_padded_with_none(tmp_path):
    rows = (
        '<row r="1"><c r="A1"><v>1</v></c><c r="C1"><v>3</v></c></row>'
        '<row r="2"><c r="B2"><v>5</v></c></row>'
    )
    path = build_xlsx(tmp_path / "s.xlsx", {"S": _sheet(rows)})
    assert xlsx.read_rows(path) == [[1, None, 3], [None, 5, None]]


def test_rich_text_preserved_space_and_phonetic_runs(tmp_path):
    shared = (
        f'<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{NS}">'
        '<si><r><t>Ab</t></r><r><t xml:space="preserve"> c</t></r></si>'
        "<si><t>base</t><rPh sb=\"0\" eb=\"4\"><t>PHONETIC</t></rPh></si>"
        "<si><t/></si></sst>"
    )
    rows = ('<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
            '<c r="C1" t="s"><v>2</v></c></row>')
    path = build_xlsx(tmp_path / "r.xlsx", {"S": _sheet(rows)}, shared=shared)
    assert xlsx.read_rows(path) == [["Ab c", "base", ""]]


def test_inline_string_boolean_error_and_formula_string(tmp_path):
    rows = (
        '<row r="1"><c r="A1" t="inlineStr"><is><t>inline</t></is></c>'
        '<c r="B1" t="b"><v>1</v></c><c r="C1" t="b"><v>0</v></c>'
        '<c r="D1" t="e"><v>#DIV/0!</v></c><c r="E1" t="str"><f>A1</f><v>cached</v></c></row>'
    )
    path = build_xlsx(tmp_path / "t.xlsx", {"S": _sheet(rows)})
    (row,) = xlsx.read_rows(path)
    assert row == ["inline", True, False, "#DIV/0!", "cached"]


def test_cells_without_value_are_none(tmp_path):
    rows = '<row r="1"><c r="A1" s="1"/><c r="B1"><v>2</v></c></row>'
    path = build_xlsx(tmp_path / "e.xlsx", {"S": _sheet(rows)}, styles=_styles([0, 14]))
    assert xlsx.read_rows(path) == [[None, 2]]


# --------------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "serial, expected",
    [
        (61, dt.datetime(1900, 3, 1)),
        (59, dt.datetime(1900, 2, 28)),
        (1, dt.datetime(1900, 1, 1)),
        (43831, dt.datetime(2020, 1, 1)),
        (46300, dt.datetime(2026, 10, 5)),
        (43831.5, dt.datetime(2020, 1, 1, 12, 0, 0)),
        (43831.999999999, dt.datetime(2020, 1, 2, 0, 0, 0)),  # float noise is rounded to the second
    ],
)
def test_serial_to_datetime_1900_system(serial, expected):
    assert xlsx.serial_to_datetime(serial) == expected


def test_serial_60_is_rejected():
    with pytest.raises(xlsx.XlsxError, match="1900-02-29"):
        xlsx.serial_to_datetime(60)


def test_serial_to_datetime_1904_system():
    assert xlsx.serial_to_datetime(0, date1904=True) == dt.datetime(1904, 1, 1)
    assert xlsx.serial_to_datetime(1462, date1904=True) == dt.datetime(1908, 1, 2)


def test_workbook_flag_selects_the_1904_system(tmp_path):
    rows = '<row r="1"><c r="A1" s="1"><v>0</v></c></row>'
    path = build_xlsx(tmp_path / "d.xlsx", {"S": _sheet(rows)}, styles=_styles([0, 14]),
                      workbookpr='<workbookPr date1904="1"/>')
    assert xlsx.read_rows(path) == [[dt.datetime(1904, 1, 1)]]


def test_a_number_in_a_general_cell_is_not_a_date(tmp_path):
    rows = '<row r="1"><c r="A1" s="0"><v>41311</v></c><c r="B1" s="1"><v>41311</v></c></row>'
    path = build_xlsx(tmp_path / "g.xlsx", {"S": _sheet(rows)}, styles=_styles([0, 14]))
    assert xlsx.read_rows(path) == [[41311, dt.datetime(2013, 2, 6)]]


@pytest.mark.parametrize(
    "code, is_date",
    [
        ("General", False), ("0.00", False), ("#,##0.00", False), ("0.00%", False),
        ("dd.mm.yyyy", True), ("yyyy-mm-dd hh:mm:ss", True), ("[$-419]d mmmm yyyy;@", True),
        ('0.0" days"', False), ('"d"0', False), ("[Red]0.00", False), ("h:mm", True),
        ("0.00E+00", False),
    ],
)
def test_custom_format_codes_are_classified(code, is_date):
    assert xlsx.is_date_format_code(code) is is_date


def test_custom_date_format_id_is_detected(tmp_path):
    styles = _styles([0, 164, 165], custom={164: "dd.mm.yyyy", 165: "0.00"})
    rows = ('<row r="1"><c r="A1" s="1"><v>43831</v></c><c r="B1" s="2"><v>43831</v></c></row>')
    path = build_xlsx(tmp_path / "c.xlsx", {"S": _sheet(rows)}, styles=styles)
    assert xlsx.read_rows(path) == [[dt.datetime(2020, 1, 1), 43831]]


# --------------------------------------------------------------------------------------------
# Sheets, tables, row numbers
# --------------------------------------------------------------------------------------------


def test_sheet_selection_by_index_and_name(tmp_path):
    one = _sheet('<row r="1"><c r="A1"><v>1</v></c></row>')
    two = _sheet('<row r="1"><c r="A1"><v>2</v></c></row>')
    path = build_xlsx(tmp_path / "m.xlsx", {"First": one, "Second": two})
    assert xlsx.sheet_names(path) == ["First", "Second"]
    assert xlsx.read_rows(path) == [[1]]
    assert xlsx.read_rows(path, sheet=1) == [[2]]
    assert xlsx.read_rows(path, sheet="Second") == [[2]]
    with pytest.raises(xlsx.XlsxError, match="Third"):
        xlsx.read_rows(path, sheet="Third")
    with pytest.raises(xlsx.XlsxError, match="index 5"):
        xlsx.read_rows(path, sheet=5)


def test_read_table_uses_the_first_row_as_header_and_keeps_sheet_row_numbers(tmp_path):
    shared = _shared(["code", "when", "x", "AAA", "BBB"])
    rows = (
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" s="1"><v>43831</v></c><c r="C2"><v>1.5</v></c></row>'
        '<row r="4"><c r="A4" t="s"><v>4</v></c><c r="B4" s="1"><v>43832</v></c></row>'
    )
    path = build_xlsx(tmp_path / "t.xlsx", {"S": _sheet(rows)}, shared=shared, styles=_styles([0, 14]))
    table = xlsx.read_table(path, row_numbers=True)
    assert list(table.columns) == ["code", "when", "x", "sheet_row"]
    assert table["sheet_row"].tolist() == [2, 4]
    assert str(table["when"].dtype).startswith("datetime64")
    assert table["when"].tolist()[0].isoformat() == "2020-01-01T00:00:00"
    assert table["x"].isna().tolist() == [False, True]


def test_sheet_xml_without_a_relationship_falls_back_to_the_numbered_part(tmp_path):
    rows = '<row r="1"><c r="A1"><v>9</v></c></row>'
    path = tmp_path / "f.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", _sheet(rows))
    assert xlsx.read_rows(path) == [[9]]


def test_strict_namespace_is_read(tmp_path):
    strict = "http://purl.oclc.org/ooxml/spreadsheetml/main"
    sheet = (f'<?xml version="1.0"?><worksheet xmlns="{strict}"><sheetData>'
             '<row r="1"><c r="A1" t="inlineStr"><is><t>ok</t></is></c></row></sheetData></worksheet>')
    path = tmp_path / "strict.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    assert xlsx.read_rows(path) == [["ok"]]


# --------------------------------------------------------------------------------------------
# Defensive behaviour: the files come from outside
# --------------------------------------------------------------------------------------------


def test_not_a_zip_file_is_an_xlsx_error(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"this is not a workbook")
    with pytest.raises(xlsx.XlsxError, match="zip"):
        xlsx.read_rows(p)


def test_entity_declarations_are_refused(tmp_path):
    bomb = ('<?xml version="1.0"?><!DOCTYPE s [<!ENTITY a "aaaa">]>'
            f'<worksheet xmlns="{NS}"><sheetData><row r="1"><c r="A1" t="inlineStr">'
            "<is><t>&a;</t></is></c></row></sheetData></worksheet>")
    path = build_xlsx(tmp_path / "b.xlsx", {"S": bomb})
    with pytest.raises(xlsx.XlsxError, match=r"DOCTYPE|ENTITY"):
        xlsx.read_rows(path)


def test_oversized_members_are_refused(tmp_path):
    rows = "".join(f'<row r="{i}"><c r="A{i}"><v>{i}</v></c></row>' for i in range(1, 400))
    path = build_xlsx(tmp_path / "big.xlsx", {"S": _sheet(rows)})
    with pytest.raises(xlsx.XlsxError, match="larger than"):
        xlsx.read_rows(path, max_member_bytes=1000)


def test_missing_sheet_part_is_an_xlsx_error(tmp_path):
    path = tmp_path / "empty.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", WORKBOOK.format(workbookpr="", sheets='<sheet name="S" sheetId="1" r:id="rId1"/>'))
        z.writestr("xl/_rels/workbook.xml.rels", RELS.format(items=(
            '<Relationship Id="rId1" Target="worksheets/sheet9.xml" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>')))
    with pytest.raises(xlsx.XlsxError, match="sheet9"):
        xlsx.read_rows(path)


# --------------------------------------------------------------------------------------------
# The real Cbonds exports (when imported)
# --------------------------------------------------------------------------------------------

CBONDS_HEADER = [
    "Название", "Биржа", "Тикер", "Отрасль", "Дивиденд", "Валюта", "Дивидендная доходность, %",
    "Последняя дата приобретения", "Экс-дивидендная дата", "Закрытие реестра", "Период", "Тип периода",
]


@needs_cbonds
def test_real_files_have_the_documented_header_and_row_counts():
    ru = xlsx.read_table(CBONDS_RU, row_numbers=True)
    allc = xlsx.read_table(CBONDS_ALL, row_numbers=True)
    assert list(ru.columns[:12]) == CBONDS_HEADER
    assert len(ru) == 2149
    assert len(allc) == 2174
    assert ru["sheet_row"].iloc[0] == 2 and ru["sheet_row"].iloc[-1] == 2150


@needs_cbonds
def test_real_files_first_row_is_the_known_yatek_payment():
    row = xlsx.read_table(CBONDS_RU).iloc[0]
    assert row["Название"] == "ЯТЭК, акция обыкновенная"
    assert row["Тикер"] == "YAKG"
    assert row["Дивиденд"] == pytest.approx(0.23)
    assert row["Валюта"] == "RUB"
    assert row["Последняя дата приобретения"] == dt.datetime(2013, 2, 6)
    assert row["Экс-дивидендная дата"] == dt.datetime(2013, 2, 7)
    assert row["Закрытие реестра"] == dt.datetime(2013, 2, 7)
    assert row["Период"] == "2012" and row["Тип периода"] == "год"


@needs_cbonds
def test_real_files_known_sber_rows_and_date_dtypes():
    t = xlsx.read_table(CBONDS_RU)
    for col in ("Последняя дата приобретения", "Экс-дивидендная дата", "Закрытие реестра"):
        assert str(t[col].dtype).startswith("datetime64"), col
    sber = t[t["Тикер"] == "SBER"].set_index("Закрытие реестра")["Дивиденд"]
    assert sber[dt.datetime(2019, 6, 13)] == pytest.approx(16.0)
    assert sber[dt.datetime(2024, 7, 11)] == pytest.approx(33.3)
    assert t["Экс-дивидендная дата"].min() == dt.datetime(2013, 2, 7)
    assert t["Экс-дивидендная дата"].max() == dt.datetime(2026, 10, 5)


@needs_cbonds
def test_real_files_second_export_adds_exactly_the_japanese_rows():
    ru = xlsx.read_table(CBONDS_RU)
    allc = xlsx.read_table(CBONDS_ALL)
    extra = allc.merge(ru.drop_duplicates(), how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"]
    assert len(extra) == 25
    assert set(extra["Валюта"]) == {"JPY"}
    assert set(extra["Тикер"]) == {"IC5, 6240"}
    assert set(ru["Валюта"]) == {"RUB", "USD"}
