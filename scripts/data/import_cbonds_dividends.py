#!/usr/bin/env python
"""Import the Cbonds dividend-calendar exports into ``data/raw/cbonds_dividends/``.

Vendor data enters ``data/`` only through scripts (CLAUDE.md, rule 3). This script

1. checks every export in the download directory against the SHA-256 recorded below (the files
   the task inspected), opens it with :mod:`cif.data.cbonds` (header, types, key fields) and
   compares the number of records with the expected one -- *before* anything is written;
2. copies the files byte for byte under clean names and makes them read-only;
3. writes ``README.txt`` (source, filters, terms of use, quirks) and ``provenance.json`` (per
   file: original name, filters, rows, ex-date range, currencies, SHA-256, size, modification
   time of the download; importer: script hash, commit, who ran it).

It refuses to overwrite a different file under the same clean name, and refuses a dirty working
tree unless ``--allow-dirty`` (the provenance must name the code that did the import). A second
run with identical files changes nothing.

Usage::

    python scripts/data/import_cbonds_dividends.py [--src-dir ~/Downloads] [--dest-dir PATH]
                                                   [--dry-run] [--allow-dirty]

The raw directory is outside git (the Cbonds offer allows internal use only); the processed
dataset built from it (``scripts/data/build_moex_dividends.py``) records the hashes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cif.data import cbonds, xlsx  # noqa: E402
from cif.data.provenance import sha256_file  # noqa: E402

DEFAULT_SRC = Path.home() / "Downloads"
DEFAULT_DEST = REPO / "data" / "raw" / "cbonds_dividends"
SCRIPT_REL = "scripts/data/import_cbonds_dividends.py"

SOURCE = {
    "vendor": "Cbonds Investfunds Pro (cbonds.ru), dividend calendar, Excel export",
    "access": "subscription of the HSE library (Sezam)",
    "exported_on": "2026-10-08",
    "exported_by": "project user",
    "terms": "internal use only, no transfer to third parties (Cbonds public offer for legal "
             "entities, cl. 4.3); Cbonds must be named as the source in publications (cl. 4.4); "
             "see docs/reviews/2026-10-07-moex-dividends-sources.md, section 5.4",
}


class ImportRefused(RuntimeError):
    """A check failed; nothing was written."""


@dataclass(frozen=True)
class ExportSpec:
    """One export as inspected by the task: what the file must be, and how to file it."""

    original_name: str
    clean_name: str
    sha256: str
    rows: int  # records, without the header
    filters: dict = field(default_factory=dict)
    contents: str = ""


_COMMON = {"ex_date_from": "2013-01-28", "ex_date_to": "2026-10-08", "exchange": "Московская Биржа"}

EXPORTS: tuple[ExportSpec, ...] = (
    ExportSpec(
        original_name="dividend_calendar_export08_10_2026.xlsx",
        clean_name="cbonds_dividend_calendar_moex_ru_20261008.xlsx",
        sha256="7287279f4104475e92fc268414b14dcc978298f0e8015a48b44ef421fc99624b",
        rows=2149,
        filters={**_COMMON, "country": "Россия"},
        contents="Moscow Exchange securities of Russian issuers; RUB and USD rows.",
    ),
    ExportSpec(
        original_name="dividend_calendar_export08_10_2026 (1).xlsx",
        clean_name="cbonds_dividend_calendar_moex_all_20261008.xlsx",
        sha256="08366dc3232081619ad5e8fb5b16ec20b823aa0f57ef5d3e84cd07f2485389a5",
        rows=2174,
        filters={**_COMMON, "country": "не задана (без фильтра по стране)"},
        contents="All rows of the first file plus 25 rows of Yamashin-Filter (JPY; ticker cell "
                 "'IC5, 6240'; not a security of the project universe).",
    ),
)

README_TEMPLATE = """\
Cbonds Investfunds Pro: календарь дивидендов, выгрузка {exported_on}
{rule}

Источник.  {vendor}; доступ: подписка библиотеки ВШЭ («Сезам»).
Выгрузил.  {exported_by}, {exported_on} (интерфейс Cbonds, раздел «Календарь дивидендов», Excel).
Импорт.    {script}, {imported_at}; коммит {sha} ({tree}); запустил: {run_by}.
           Файлы скопированы побайтно, SHA-256 сверены с записанными заранее. Сырые файлы руками не
           правятся (CLAUDE.md, правило 3): ошибка исправляется повторной выгрузкой и повторным импортом.

Условия использования.
  - Публичная оферта Cbonds для юридических лиц (пп. 4.3 и 4.4): только для внутреннего использования,
    без передачи третьим лицам; в публикациях Cbonds указывается как источник.
  - Поэтому каталог data/raw/ вне git, а в статье публикуются код, хеши и описание запроса, не данные.

Файлы.
{files}

Колонки (12): {columns}.

Что в выгрузке есть и чего нет.
  - Нет ISIN и признака выплаты: отменённый дивиденд не помечен; строка — заявление, а не факт.
  - «Дивидендная доходность, %» — текущая доходность на дату выгрузки, не историческая. Не использовать.
  - Ячейка «Тикер» может содержать несколько тикеров одной бумаги («T, TCSG», «QIWI, NNDQ»,
    «AFLT, RU_AFLT», «RUAL, 0486»).
  - В трёх строках (Нанософт) ячейка «Тикер» пуста; бумага не из вселенной, строки остаются в таблице без тикера.
  - QIWI: каждая выплата записана дважды (тип «регулярные» и метка квартала), суммы равны, экс-даты
    расходятся на 0-3 дня.
  - Суммы пересчитаны на текущее число акций после сплитов и консолидаций (GMKN, TRNFP, PLZL, VTBR, T).
    К базе на дату экс-дивиденда они возвращаются таблицей сплитов и сверкой с падением цены
    (cif.data.moex_dividends).
  - Делистингованных и переехавших эмитентов в выгрузке нет (проверка 2026-10-08: 72 из 101 тикера
    вселенной moex_v2). Их выплаты — в реестре src/cif/data/moex_dividend_registry.py с источником на
    каждую выплату.
  - Экс-даты Cbonds не всегда верны (реестры в выходные и праздники, закрытие 2022 г.): они служат
    сверкой, экс-дата строится по правилу T+2/T+1 с проверкой по цене.

Дальше: scripts/data/build_moex_dividends.py -> data/processed/moex_dividends/.
"""


def git_state(repo: Path = REPO) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def default_user(repo: Path = REPO) -> str:
    name = subprocess.run(["git", "config", "user.name"], cwd=repo, capture_output=True, text=True,
                          check=False).stdout.strip()
    return name or getpass.getuser()


def _inspect(spec: ExportSpec, src_dir: Path) -> dict:
    """Verify one export and return its record; raises :class:`ImportRefused`."""
    path = src_dir / spec.original_name
    if not path.is_file():
        raise ImportRefused(f"{spec.original_name}: not found in {src_dir}")
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise ImportRefused(f"{spec.original_name}: SHA-256 {actual} differs from the recorded {spec.sha256}")
    try:
        summary = cbonds.summarise(cbonds.read_calendar(path))
    except (cbonds.CbondsFormatError, xlsx.XlsxError) as exc:
        raise ImportRefused(f"{spec.original_name}: {exc}") from exc
    if summary["rows"] != spec.rows:
        raise ImportRefused(f"{spec.original_name}: {summary['rows']} rows, the inspected file has {spec.rows}")
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)
    return {
        "name": spec.clean_name, "original_name": spec.original_name, "sha256": actual,
        "bytes": path.stat().st_size, "source_mtime": mtime.isoformat(timespec="seconds"),
        "filters": dict(spec.filters), **summary, "contents": spec.contents,
    }


def render_readme(records: list[dict], *, imported_at: str, git: dict, run_by: str) -> str:
    blocks = []
    for i, rec in enumerate(records, start=1):
        filters = "; ".join(f"{k}={v}" for k, v in rec["filters"].items())
        currencies = ", ".join(f"{k} {v}" for k, v in rec["currencies"].items())
        blocks.append(
            f"  {i}. {rec['name']}\n"
            f"     исходное имя: {rec['original_name']} (изменён {rec['source_mtime']})\n"
            f"     фильтры: {filters}\n"
            f"     строк: {rec['rows']}; экс-даты {rec['ex_date_min']} ... {rec['ex_date_max']}; "
            f"валюты: {currencies}\n"
            f"     SHA-256: {rec['sha256']}\n"
            f"     {rec['contents']}"
        )
    title = f"Cbonds Investfunds Pro: календарь дивидендов, выгрузка {SOURCE['exported_on']}"
    return README_TEMPLATE.format(
        exported_on=SOURCE["exported_on"], rule="=" * len(title), vendor=SOURCE["vendor"],
        exported_by="пользователь проекта", script=SCRIPT_REL, imported_at=imported_at,
        sha=git["sha"][:10], tree="дерево ГРЯЗНОЕ" if git["dirty"] else "дерево чистое", run_by=run_by,
        files="\n".join(blocks), columns=", ".join(cbonds.CBONDS_HEADER),
    )


def _already_imported(records: list[dict], dest_dir: Path) -> bool:
    prov = dest_dir / "provenance.json"
    if not prov.is_file() or not (dest_dir / "README.txt").is_file():
        return False
    try:
        recorded = json.loads(prov.read_text(encoding="utf-8")).get("files", {})
    except ValueError:
        return False
    return all(
        (dest_dir / r["name"]).is_file() and sha256_file(dest_dir / r["name"]) == r["sha256"]
        and recorded.get(r["name"], {}).get("sha256") == r["sha256"]
        for r in records
    )


def import_exports(
    specs: tuple[ExportSpec, ...], src_dir: Path | str, dest_dir: Path | str, *, git: dict,
    run_by: str, now: dt.datetime | None = None, allow_dirty: bool = False, dry_run: bool = False,
) -> dict:
    """Verify, copy and record the exports; see the module docstring.

    Returns ``{"status": "imported" | "unchanged" | "dry-run", "files": [records]}``.
    """
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    if git["dirty"] and not allow_dirty:
        raise ImportRefused("the working tree is not clean: commit first (the provenance names the "
                            "code that imported the data) or pass --allow-dirty")
    records = [_inspect(spec, src_dir) for spec in specs]  # nothing is written until all pass
    if dry_run:
        return {"status": "dry-run", "files": records}
    if _already_imported(records, dest_dir):
        return {"status": "unchanged", "files": records}

    for rec in records:
        target = dest_dir / rec["name"]
        if target.exists() and sha256_file(target) != rec["sha256"]:
            raise ImportRefused(f"{target} exists and differs from the export: refusing to overwrite "
                                "raw data (remove it by hand if it is known to be wrong)")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        target = dest_dir / rec["name"]
        if not target.exists():
            shutil.copyfile(src_dir / rec["original_name"], target)
        if sha256_file(target) != rec["sha256"]:
            raise ImportRefused(f"{target}: the copy differs from the source")
        os.chmod(target, 0o444)

    stamp = (now or dt.datetime.now(dt.UTC)).isoformat(timespec="seconds")
    provenance = {
        "dataset": "cbonds_dividends_raw",
        "imported_at": stamp,
        "importer": {"script": SCRIPT_REL, "script_sha256": sha256_file(Path(__file__)),
                     "git": git, "run_by": run_by},
        "source": SOURCE,
        "files": {rec["name"]: {k: v for k, v in rec.items() if k != "name"} for rec in records},
    }
    (dest_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (dest_dir / "README.txt").write_text(
        render_readme(records, imported_at=stamp, git=git, run_by=run_by), encoding="utf-8")
    return {"status": "imported", "files": records}


def main(argv: list[str] | None = None, specs: tuple[ExportSpec, ...] = EXPORTS) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC, help="where the exports were downloaded")
    ap.add_argument("--dest-dir", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--dry-run", action="store_true", help="verify everything, write nothing")
    ap.add_argument("--allow-dirty", action="store_true", help="import from a dirty working tree")
    args = ap.parse_args(argv)
    try:
        result = import_exports(specs, args.src_dir, args.dest_dir, git=git_state(), run_by=default_user(),
                                allow_dirty=args.allow_dirty, dry_run=args.dry_run)
    except ImportRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(f"{result['status']}: {len(result['files'])} files -> {args.dest_dir}")
    for rec in result["files"]:
        print(f"  {rec['name']}: {rec['rows']} rows, ex-dates {rec['ex_date_min']} .. {rec['ex_date_max']}, "
              f"currencies {rec['currencies']}, sha256 {rec['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
