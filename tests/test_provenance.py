"""Re-hashing the files a dataset provenance records (R8: SHA-256 in provenance must match).

Three layouts occur in the repository: ``files.raw`` / ``files.processed`` (moex_v2,
riskfree_v2), ``outputs`` plus ``{path, sha256}`` sources (sp500_pit), and the legacy
``{path, sha256}`` entries relative to the repository root (moex/sp500 provenance of April 2026).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cif.data.provenance import main, verify_provenance


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


def _files_layout(root: Path) -> Path:
    raw, proc = root / "data/raw/ds", root / "data/processed/ds"
    prov = {
        "dataset": "ds",
        "git": {"sha": "abc", "dirty": False},
        "config": {"raw_dir": "data/raw/ds", "processed_dir": "data/processed/ds"},
        "files": {
            "raw": {"sub/a.json": {"sha256": _write(raw / "sub/a.json", b"a"), "bytes": 1}},
            "processed": {"p.parquet": {"sha256": _write(proc / "p.parquet", b"p"), "bytes": 1}},
        },
    }
    (proc / "provenance.json").write_text(json.dumps(prov))
    return proc / "provenance.json"


def test_files_layout_all_match(tmp_path):
    rep = verify_provenance(_files_layout(tmp_path), repo_root=tmp_path)
    assert rep["n_files"] == 2
    assert rep["n_ok"] == 2
    assert rep["mismatch"] == [] and rep["missing"] == []
    assert rep["ok"] is True
    assert rep["git"] == {"sha": "abc", "dirty": False}


def test_changed_file_is_a_mismatch(tmp_path):
    prov = _files_layout(tmp_path)
    (tmp_path / "data/processed/ds/p.parquet").write_bytes(b"changed")
    rep = verify_provenance(prov, repo_root=tmp_path)
    assert rep["ok"] is False
    assert [m["file"] for m in rep["mismatch"]] == ["data/processed/ds/p.parquet"]
    assert rep["mismatch"][0]["recorded"] == _sha(b"p")
    assert rep["mismatch"][0]["actual"] == _sha(b"changed")


def test_deleted_file_is_missing(tmp_path):
    prov = _files_layout(tmp_path)
    (tmp_path / "data/raw/ds/sub/a.json").unlink()
    rep = verify_provenance(prov, repo_root=tmp_path)
    assert rep["ok"] is False
    assert rep["missing"] == ["data/raw/ds/sub/a.json"]


def test_outputs_and_sources_layout_with_absolute_dirs(tmp_path):
    raw, out = tmp_path / "data/raw/sp", tmp_path / "data/processed/sp"
    prov = {
        "git": {"sha": "def", "dirty": True},
        "parameters": {"raw_dir": str(raw), "out_dir": str(out)},
        "wikipedia": {
            "constituents": {"path": "w.html", "sha256": _write(raw / "w.html", b"w"), "url": "u"},
            "changes": {"path": "c.html", "sha256": _write(raw / "c.html", b"c")},
        },
        "outputs": {"m.parquet": {"sha256": _write(out / "m.parquet", b"m"), "bytes": 1}},
    }
    (out / "provenance.json").write_text(json.dumps(prov))
    rep = verify_provenance(out / "provenance.json", repo_root=tmp_path)
    assert rep["n_files"] == 3 and rep["ok"] is True


def test_legacy_layout_paths_relative_to_repo_root(tmp_path):
    prov = {
        "key": "moex",
        "raw_prices": {"path": "data/raw/x_raw.parquet", "sha256": _write(tmp_path / "data/raw/x_raw.parquet", b"r")},
        "cleaned_prices": {"path": "data/processed/x.parquet", "sha256": _sha(b"old")},
    }
    _write(tmp_path / "data/processed/x.parquet", b"new")
    path = tmp_path / "data/processed/x_provenance.json"
    path.write_text(json.dumps(prov))
    rep = verify_provenance(path, repo_root=tmp_path)
    assert rep["n_files"] == 2 and rep["n_ok"] == 1
    assert [m["file"] for m in rep["mismatch"]] == ["data/processed/x.parquet"]
    assert rep["git"] is None


def test_cli_exit_code_and_json(tmp_path, capsys):
    prov = _files_layout(tmp_path)
    assert main(["verify", str(prov), "--repo-root", str(tmp_path)]) == 0
    (tmp_path / "data/processed/ds/p.parquet").write_bytes(b"x")
    out_json = tmp_path / "report.json"
    assert main(["verify", str(prov), "--repo-root", str(tmp_path), "--json", str(out_json)]) == 1
    reports = json.loads(out_json.read_text())
    assert reports[0]["ok"] is False
    assert "MISMATCH" in capsys.readouterr().out
