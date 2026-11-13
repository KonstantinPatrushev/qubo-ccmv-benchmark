"""Re-hash the files a dataset provenance records (R8: recorded SHA-256 must match the files).

Layouts in the repository:

* ``files.raw`` / ``files.processed`` -- ``{relative path: {"sha256": ...}}`` relative to
  ``config.raw_dir`` and to the directory of ``provenance.json`` (moex_v2, riskfree_v2, ff49);
* flat ``files`` -- ``{name: {"sha256": ...}}`` next to ``provenance.json`` (OR-Library);
* ``outputs`` -- the same, relative to the provenance directory (sp500_pit);
* any other ``{"path": ..., "sha256": ...}`` entry: a source file relative to the raw directory
  (sp500_pit Wikipedia snapshots) or, in the legacy provenance of April 2026, to the repository
  root. An absolute path is used as is.

CLI::

    python -m cif.data.provenance verify data/processed/*/provenance.json [--json report.json]

The exit code is 1 when any recorded file is missing or differs, or when no file is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def _path_entries(node, skip: tuple[str, ...]) -> Iterator[dict]:
    """Every dict with both ``path`` and ``sha256``, outside the top-level keys in ``skip``."""
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            yield node
        for key, value in node.items():
            if key not in skip:
                yield from _path_entries(value, ())
    elif isinstance(node, list):
        for value in node:
            yield from _path_entries(value, ())


def recorded_files(prov: dict, prov_dir: Path, repo_root: Path) -> list[tuple[Path, str]]:
    """``(path, recorded sha256)`` for every file the provenance records."""
    settings = prov.get("config") or prov.get("parameters") or {}
    raw_dir = _resolve(settings["raw_dir"], repo_root) if settings.get("raw_dir") else None
    out: list[tuple[Path, str]] = []
    files = prov.get("files") or {}
    if set(files) & {"raw", "processed"}:
        for rel, entry in (files.get("raw") or {}).items():
            out.append((_resolve(rel, raw_dir if raw_dir is not None else repo_root), entry["sha256"]))
        for rel, entry in (files.get("processed") or {}).items():
            out.append((_resolve(rel, prov_dir), entry["sha256"]))
    else:
        # flat layout {name: {"sha256": ...}} next to the provenance (OR-Library, audit P-1)
        for rel, entry in files.items():
            if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                out.append((_resolve(rel, prov_dir), entry["sha256"]))
    for rel, entry in (prov.get("outputs") or {}).items():
        out.append((_resolve(rel, prov_dir), entry["sha256"]))
    for entry in _path_entries(prov, skip=("files", "outputs")):
        bases = [b for b in (raw_dir, repo_root, prov_dir) if b is not None]
        candidates = [_resolve(entry["path"], b) for b in bases]
        path = next((c for c in candidates if c.exists()), candidates[0])
        out.append((path, entry["sha256"]))
    return out


def _display(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def verify_provenance(path: Path | str, repo_root: Path | str | None = None) -> dict:
    """Re-hash every recorded file; report matches, mismatches and missing files."""
    path = Path(path)
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    prov = json.loads(path.read_text(encoding="utf-8"))
    mismatch: list[dict] = []
    missing: list[str] = []
    entries = recorded_files(prov, path.parent, root)
    n_ok = 0
    for file, recorded in entries:
        shown = _display(file, root)
        if not file.exists():
            missing.append(shown)
            continue
        actual = sha256_file(file)
        if actual == recorded:
            n_ok += 1
        else:
            mismatch.append({"file": shown, "recorded": recorded, "actual": actual})
    return {
        "provenance": _display(path, root),
        "dataset": prov.get("dataset") or prov.get("key") or path.parent.name,
        "built_at": prov.get("built_at") or prov.get("downloaded_at"),
        "git": prov.get("git"),
        "n_files": len(entries),
        "n_ok": n_ok,
        "mismatch": mismatch,
        "missing": missing,
        # a provenance from which no file can be read proves nothing (audit P-1)
        "ok": bool(entries) and not mismatch and not missing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.data.provenance")
    sub = parser.add_subparsers(dest="command", required=True)
    ver = sub.add_parser("verify", help="re-hash the files recorded in provenance JSON files")
    ver.add_argument("provenance", nargs="+", type=Path)
    ver.add_argument("--repo-root", type=Path, default=None)
    ver.add_argument("--json", type=Path, default=None, help="write the full reports here")
    args = parser.parse_args(argv)

    reports = [verify_provenance(p, repo_root=args.repo_root) for p in args.provenance]
    for rep in reports:
        git = rep["git"] or {}
        state = f"git {str(git.get('sha', '?'))[:8]} dirty={git.get('dirty')}" if git else "git n/a"
        verdict = "OK" if rep["ok"] else "FAIL"
        print(f"{verdict:4} {rep['provenance']}: {rep['n_ok']}/{rep['n_files']} files match "
              f"(built {rep['built_at']}, {state})")
        if not rep["n_files"]:
            print("     NO RECORDED FILES: the layout is not recognised or the provenance is empty")
        for m in rep["mismatch"]:
            print(f"     MISMATCH {m['file']}: recorded {m['recorded'][:12]}, actual {m['actual'][:12]}")
        for f in rep["missing"]:
            print(f"     MISSING  {f}")
    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0 if all(r["ok"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
