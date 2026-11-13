"""Audit P-1 (MINOR, 2026-10-08): ``cif.data.provenance verify`` passes vacuously.

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

``recorded_files`` knows ``files.raw`` / ``files.processed``, ``outputs`` and ``{"path", "sha256"}``
entries. The OR-Library provenance (``data/raw/orlib/provenance.json``, written by
``cif.instances.orlib.download_orlib``) records its files as ``files: {name: {"sha256": ...}}``:
none of them is found, and a report with zero recorded files is ``ok``. Observed on the repository::

    OK   data/raw/orlib/provenance.json: 0/0 files match

ORlib is the C5 cell of RQ2 (confirmatory). Expected: the ORlib layout is re-hashed, and a
provenance from which no file can be read is a failure, not a pass.
"""

from __future__ import annotations

import hashlib
import json

from cif.data.provenance import verify_provenance


def test_p1_orlib_layout_with_a_changed_file_fails(tmp_path):
    (tmp_path / "port1.txt").write_text("31\n0.001 0.04\n")
    recorded = hashlib.sha256(b"the bytes that were downloaded").hexdigest()
    (tmp_path / "provenance.json").write_text(json.dumps({
        "source": "OR-Library", "files": {"port1.txt": {"url": "https://x", "sha256": recorded}},
    }))
    rep = verify_provenance(tmp_path / "provenance.json", repo_root=tmp_path)
    assert rep["n_files"] == 1, f"recorded files found: {rep['n_files']}"
    assert rep["ok"] is False


def test_p1_a_provenance_without_any_recorded_file_is_not_ok(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"dataset": "x", "files": {}}))
    rep = verify_provenance(tmp_path / "provenance.json", repo_root=tmp_path)
    assert rep["ok"] is False, f"0/0 files reported as OK: {rep}"
