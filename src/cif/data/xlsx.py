"""Standard-library reader for ``.xlsx`` workbooks (no ``openpyxl``).

``openpyxl`` is not part of the ``cif`` environment and installing packages needs the user's
permission, while vendor tables (the Cbonds dividend calendar) arrive as ``.xlsx``. A workbook is
a zip archive of XML parts, and what a table needs from it is small: the shared strings, the cell
values of one sheet and the knowledge which number formats are dates. This module reads exactly
that with :mod:`zipfile` and :mod:`xml.etree.ElementTree`.

What is returned
----------------
* text cells (shared, inline and formula strings) -> ``str``;
* numbers -> ``int`` for integer literals, ``float`` otherwise;
* cells whose number format is a date or time format -> :class:`datetime.datetime` (a time of day
  without a date lies on 1899-12-30, a placeholder; the 1904 date system of the workbook flag
  ``date1904`` is honoured);
* booleans -> ``bool``; error cells (``#DIV/0!``) -> the error text; empty cells -> ``None``.

Cell *formulas* are not evaluated: the value cached in the file is returned.

The files come from outside, so the reader refuses what a plain table never needs: XML with a
``DOCTYPE`` or ``ENTITY`` declaration (entity-expansion bombs) and archive members that inflate
beyond ``max_member_bytes``.
"""

from __future__ import annotations

import datetime as dt
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

#: Upper bound on the inflated size of any archive member that is parsed (bytes).
MAX_MEMBER_BYTES = 256 * 1024 * 1024

#: Built-in ``numFmtId`` values that Excel displays as dates or times (ECMA-376 18.8.30, plus the
#: locale-dependent East Asian date formats 27-36 and 50-58).
BUILTIN_DATE_FORMAT_IDS = frozenset({*range(14, 23), *range(27, 37), *range(45, 48), *range(50, 59)})

_INT = re.compile(r"-?\d+")
_QUOTED = re.compile(r'"[^"]*"')
_ESCAPED = re.compile(r"\\.")
_ELAPSED = re.compile(r"\[(?:h+|m+|s+)\]", re.IGNORECASE)
_BRACKETS = re.compile(r"\[[^\]]*\]")
_FILLER = re.compile(r"[_*].")
_DATE_TOKEN = re.compile(r"[dmyhsDMYHS]")
_CELL_REF = re.compile(r"([A-Za-z]+)(\d+)?")


class XlsxError(ValueError):
    """The file is not a workbook this reader can use."""


# --------------------------------------------------------------------------------------------
# Archive and XML helpers
# --------------------------------------------------------------------------------------------


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _open(path: Path | str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise XlsxError(f"{path}: not a zip archive, so not an .xlsx workbook ({exc})") from exc


def _member(zf: zipfile.ZipFile, name: str, max_bytes: int) -> bytes:
    """Inflate one member, never more than ``max_bytes`` (the declared size is not trusted)."""
    try:
        info = zf.getinfo(name)
    except KeyError as exc:
        raise XlsxError(f"{zf.filename}: archive member {name!r} is missing") from exc
    if info.file_size > max_bytes:
        raise XlsxError(f"{name} is larger than {max_bytes} bytes (declared {info.file_size})")
    with zf.open(info) as fh:
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise XlsxError(f"{name} is larger than {max_bytes} bytes once inflated")
    return data


def _parse(data: bytes, name: str) -> ET.Element:
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise XlsxError(f"{name} declares a DOCTYPE or ENTITY; refused (entity-expansion risk)")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise XlsxError(f"{name} is not well-formed XML ({exc})") from exc


def _text_of(item: ET.Element) -> str:
    """Text of a shared-string item or inline string: plain ``t`` and rich-text runs, not the
    phonetic (furigana) runs."""
    parts: list[str] = []
    for child in item:
        kind = _local(child)
        if kind == "t":
            parts.append(child.text or "")
        elif kind == "r":
            parts.extend((t.text or "") for t in child if _local(t) == "t")
    return "".join(parts)


# --------------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------------


def serial_to_datetime(serial: float, *, date1904: bool = False) -> dt.datetime:
    """Excel serial date -> naive :class:`datetime.datetime`, rounded to the second.

    The 1900 system counts the non-existent 1900-02-29 as serial 60: serials 1-59 are shifted by
    one day with respect to the base 1899-12-30 that is right from serial 61 on, and serial 60
    itself is refused.
    """
    if date1904:
        base = dt.datetime(1904, 1, 1)
    elif 1 <= serial < 60:
        base = dt.datetime(1899, 12, 31)
    elif 60 <= serial < 61:
        raise XlsxError(f"date serial {serial} is the non-existent 1900-02-29")
    else:
        base = dt.datetime(1899, 12, 30)
    return base + dt.timedelta(seconds=round(serial * 86400))


def is_date_format_code(code: str) -> bool:
    """Is a custom number-format code a date/time format (``dd.mm.yyyy``, ``h:mm``, ...)?

    Quoted text, escaped characters, ``_x``/``*x`` fillers and bracketed parts (colours, locale
    tags such as ``[$-419]``) are dropped first; elapsed-time brackets (``[h]``) count as time.
    ``e``/``E`` is deliberately not a date token: it is the exponent of scientific notation.
    """
    text = _QUOTED.sub("", code)
    text = _ESCAPED.sub("", text)
    text = _ELAPSED.sub("h", text)
    text = _BRACKETS.sub("", text)
    text = _FILLER.sub("", text)
    return bool(_DATE_TOKEN.search(text))


# --------------------------------------------------------------------------------------------
# Workbook structure
# --------------------------------------------------------------------------------------------


def _workbook(zf: zipfile.ZipFile, max_bytes: int):
    """``([(sheet name, relationship id)], date1904)``; empty list without ``workbook.xml``."""
    if "xl/workbook.xml" not in zf.namelist():
        return [], False
    root = _parse(_member(zf, "xl/workbook.xml", max_bytes), "xl/workbook.xml")
    sheets = []
    for node in root.iter():
        if _local(node) != "sheet":
            continue
        rid = next((v for k, v in node.attrib.items() if k.endswith("}id")), None)
        sheets.append((node.get("name", ""), rid))
    pr = next((n for n in root.iter() if _local(n) == "workbookPr"), None)
    date1904 = pr is not None and pr.get("date1904", "0").lower() in ("1", "true")
    return sheets, date1904


def _relationships(zf: zipfile.ZipFile, max_bytes: int) -> dict[str, str]:
    name = "xl/_rels/workbook.xml.rels"
    if name not in zf.namelist():
        return {}
    root = _parse(_member(zf, name, max_bytes), name)
    return {n.get("Id", ""): n.get("Target", "") for n in root.iter() if _local(n) == "Relationship"}


def sheet_names(path: Path | str) -> list[str]:
    """Names of the sheets in workbook order."""
    with _open(path) as zf:
        return [name for name, _ in _workbook(zf, MAX_MEMBER_BYTES)[0]]


def _sheet_part(zf: zipfile.ZipFile, sheet: int | str, max_bytes: int) -> tuple[str, bool]:
    sheets, date1904 = _workbook(zf, max_bytes)
    if not sheets:
        if isinstance(sheet, str):
            raise XlsxError(f"{zf.filename}: no workbook.xml, so sheet {sheet!r} cannot be found by name")
        return f"xl/worksheets/sheet{sheet + 1}.xml", date1904
    if isinstance(sheet, str):
        matches = [i for i, (name, _) in enumerate(sheets) if name == sheet]
        if not matches:
            raise XlsxError(f"sheet {sheet!r} not found; available: {[n for n, _ in sheets]}")
        index = matches[0]
    else:
        if not 0 <= sheet < len(sheets):
            raise XlsxError(f"sheet index {sheet} out of range (workbook has {len(sheets)} sheets)")
        index = sheet
    rid = sheets[index][1]
    target = _relationships(zf, max_bytes).get(rid or "")
    if not target:
        return f"xl/worksheets/sheet{index + 1}.xml", date1904
    part = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
    return part, date1904


def _shared_strings(zf: zipfile.ZipFile, max_bytes: int) -> list[str]:
    name = "xl/sharedStrings.xml"
    if name not in zf.namelist():
        return []
    root = _parse(_member(zf, name, max_bytes), name)
    return [_text_of(item) for item in root if _local(item) == "si"]


def _date_styles(zf: zipfile.ZipFile, max_bytes: int) -> set[int]:
    """Indices of the cell formats (``cellXfs``) that display dates."""
    name = "xl/styles.xml"
    if name not in zf.namelist():
        return set()
    root = _parse(_member(zf, name, max_bytes), name)
    custom = {
        int(n.get("numFmtId", -1)): n.get("formatCode", "")
        for n in root.iter() if _local(n) == "numFmt"
    }
    out: set[int] = set()
    xfs = next((n for n in root.iter() if _local(n) == "cellXfs"), None)
    if xfs is None:
        return out
    for i, xf in enumerate(c for c in xfs if _local(c) == "xf"):
        fmt = int(xf.get("numFmtId", 0))
        if fmt in BUILTIN_DATE_FORMAT_IDS or (fmt in custom and is_date_format_code(custom[fmt])):
            out.add(i)
    return out


# --------------------------------------------------------------------------------------------
# Cells and rows
# --------------------------------------------------------------------------------------------


def _column_index(ref: str) -> int:
    """``"A1"`` -> 0, ``"AB7"`` -> 27."""
    match = _CELL_REF.fullmatch(ref)
    if match is None:
        raise XlsxError(f"bad cell reference {ref!r}")
    n = 0
    for ch in match.group(1).upper():
        n = n * 26 + ord(ch) - 64
    return n - 1


def _number(text: str) -> int | float:
    return int(text) if _INT.fullmatch(text) else float(text)


def _cell_value(cell: ET.Element, strings: list[str], date_styles: set[int], date1904: bool) -> Any:
    kind = cell.get("t")
    value = next((c for c in cell if _local(c) == "v"), None)
    if kind == "inlineStr":
        inline = next((c for c in cell if _local(c) == "is"), None)
        return _text_of(inline) if inline is not None else None
    text = value.text if value is not None else None
    if text is None:
        return None
    if kind == "s":
        try:
            return strings[int(text)]
        except (ValueError, IndexError) as exc:
            raise XlsxError(f"cell {cell.get('r')}: bad shared-string index {text!r}") from exc
    if kind == "b":
        return text.strip() in ("1", "true")
    if kind in ("e", "str"):
        return text
    if kind == "d":
        return dt.datetime.fromisoformat(text)
    number = _number(text)
    if int(cell.get("s", 0)) in date_styles:
        return serial_to_datetime(float(number), date1904=date1904)
    return number


def iter_numbered_rows(
    path: Path | str, sheet: int | str = 0, *, max_member_bytes: int = MAX_MEMBER_BYTES
):
    """Yield ``(sheet row number, values)``; every row is padded to the width of the sheet.

    Row numbers are the ones Excel shows (1-based); rows absent from the file are skipped, not
    filled in.
    """
    with _open(path) as zf:
        part, date1904 = _sheet_part(zf, sheet, max_member_bytes)
        root = _parse(_member(zf, part, max_member_bytes), part)
        strings = _shared_strings(zf, max_member_bytes)
        date_styles = _date_styles(zf, max_member_bytes)
    rows: list[tuple[int, dict[int, Any]]] = []
    previous = 0
    for row in (n for n in root.iter() if _local(n) == "row"):
        number = int(row.get("r", previous + 1))
        previous = number
        cells: dict[int, Any] = {}
        next_col = 0
        for cell in (c for c in row if _local(c) == "c"):
            ref = cell.get("r")
            col = _column_index(ref) if ref else next_col
            next_col = col + 1
            cells[col] = _cell_value(cell, strings, date_styles, date1904)
        rows.append((number, cells))
    width = max((max(cells) + 1 for _, cells in rows if cells), default=0)
    for number, cells in rows:
        yield number, [cells.get(j) for j in range(width)]


def read_rows(
    path: Path | str, sheet: int | str = 0, *, max_member_bytes: int = MAX_MEMBER_BYTES
) -> list[list[Any]]:
    """All rows of one sheet (by index or name) as lists of Python values."""
    return [values for _, values in iter_numbered_rows(path, sheet, max_member_bytes=max_member_bytes)]


def read_table(
    path: Path | str,
    sheet: int | str = 0,
    *,
    row_numbers: bool = False,
    max_member_bytes: int = MAX_MEMBER_BYTES,
) -> pd.DataFrame:
    """One sheet as a table: the first row is the header, the rest are records.

    With ``row_numbers=True`` the column ``sheet_row`` holds the spreadsheet row number of every
    record, so a record can be traced back to the cell it came from.
    """
    numbered = list(iter_numbered_rows(path, sheet, max_member_bytes=max_member_bytes))
    if not numbered:
        raise XlsxError(f"{path}: sheet {sheet!r} has no rows")
    header = [str(h) if h is not None else f"col{j}" for j, h in enumerate(numbered[0][1])]
    frame = pd.DataFrame([values for _, values in numbered[1:]], columns=header)
    if row_numbers:
        frame["sheet_row"] = [n for n, _ in numbered[1:]]
    return frame
