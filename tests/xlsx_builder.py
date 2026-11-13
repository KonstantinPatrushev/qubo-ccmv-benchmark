"""Write small ``.xlsx`` workbooks for tests, byte by byte with :mod:`zipfile`.

Used by ``test_xlsx_reader.py`` (edge cases of the format), ``test_cbonds_import.py`` and the
dividend-loader tests (Cbonds-shaped fixtures). The writer deliberately shares no code with
``cif.data.xlsx``, so a test of the reader does not assume how the reader sees a workbook.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

WORKBOOK = (
    f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS}" xmlns:r="{REL_NS}">'
    "{workbookpr}<sheets>{sheets}</sheets></workbook>"
)
RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{items}'
    "</Relationships>"
)
STYLES = (
    f'<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{NS}">{{numfmts}}'
    '<cellXfs count="{n}">{xfs}</cellXfs></styleSheet>'
)


def styles_xml(xf_ids: list[int], custom: dict[int, str] | None = None) -> str:
    """A ``styles.xml`` whose cell formats ``0..n-1`` use the given ``numFmtId`` values."""
    numfmts = ""
    if custom:
        items = "".join(f'<numFmt numFmtId="{i}" formatCode="{c}"/>' for i, c in custom.items())
        numfmts = f'<numFmts count="{len(custom)}">{items}</numFmts>'
    xfs = "".join(f'<xf numFmtId="{i}" fontId="0" fillId="0" borderId="0"/>' for i in xf_ids)
    return STYLES.format(numfmts=numfmts, n=len(xf_ids), xfs=xfs)


def shared_xml(strings: Sequence[str]) -> str:
    body = "".join(f"<si><t>{escape(s)}</t></si>" for s in strings)
    return (f'<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{NS}" count="{len(strings)}" '
            f'uniqueCount="{len(strings)}">{body}</sst>')


def sheet_xml(rows_xml: str) -> str:
    return (f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{NS}">'
            f"<sheetData>{rows_xml}</sheetData></worksheet>")


def build_xlsx(path: Path, sheets: dict[str, str], *, shared: str | None = None,
               styles: str | None = None, workbookpr: str = "",
               extra: dict[str, bytes] | None = None) -> Path:
    """Write a workbook with the given sheet XMLs (name -> XML), in that order."""
    entries = []
    rels = []
    for i, name in enumerate(sheets, start=1):
        entries.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
        rels.append(f'<Relationship Id="rId{i}" Target="worksheets/sheet{i}.xml" '
                    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", WORKBOOK.format(workbookpr=workbookpr, sheets="".join(entries)))
        z.writestr("xl/_rels/workbook.xml.rels", RELS.format(items="".join(rels)))
        for i, xml in enumerate(sheets.values(), start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", xml)
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", shared)
        if styles is not None:
            z.writestr("xl/styles.xml", styles)
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return path


def _column_letter(j: int) -> str:
    out = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        out = chr(65 + r) + out
    return out


def _excel_serial(value: dt.date | dt.datetime) -> int:
    day = value.date() if isinstance(value, dt.datetime) else value
    return (day - dt.date(1899, 12, 30)).days


def write_table(path: Path, header: Iterable[str], records: Iterable[Sequence[Any]]) -> Path:
    """A one-sheet workbook: ``header`` as the first row, then one row per record.

    ``str`` becomes a shared string, ``datetime``/``date`` a date-formatted serial (``numFmtId``
    14), ``int``/``float`` a number and ``None`` an absent cell.
    """
    strings: list[str] = []
    index: dict[str, int] = {}

    def sid(text: str) -> int:
        if text not in index:
            index[text] = len(strings)
            strings.append(text)
        return index[text]

    def row_xml(number: int, values: Sequence[Any]) -> str:
        cells = []
        for j, v in enumerate(values):
            ref = f"{_column_letter(j)}{number}"
            if v is None:
                continue
            if isinstance(v, str):
                cells.append(f'<c r="{ref}" t="s"><v>{sid(v)}</v></c>')
            elif isinstance(v, (dt.datetime, dt.date)):
                cells.append(f'<c r="{ref}" s="1"><v>{_excel_serial(v)}</v></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{v!r}</v></c>')
        return f'<row r="{number}">{"".join(cells)}</row>'

    rows = [row_xml(1, list(header))]
    rows += [row_xml(i, rec) for i, rec in enumerate(records, start=2)]
    return build_xlsx(path, {"Sheet1": sheet_xml("".join(rows))}, shared=shared_xml(strings),
                      styles=styles_xml([0, 14]))
