"""Audit R-1 (MAJOR): the run registry cannot tell which code produced a run.

Finding R-1 in docs/reviews/2026-10-04-audit-e0.md (R8, plan Sec. 6).

``cif.experiments.registry.start_run`` records ``git_sha`` and a boolean ``git_dirty`` only:

* a dirty run stores neither the list of modified files nor the patch, so it cannot be
  reproduced from the registry (4 of the 5 completed E0 runs and the running v5 run are dirty,
  including the replay behind docs/E0_legacy_delta.md);
* untracked code is ignored (``git status --untracked-files=no``): a new, never committed module
  or script that a run imports is registered as a CLEAN run of a commit that does not contain it;
* outside a git checkout (code snapshot on the cluster, a "hybrid" package without
  ``CIF_REPO_ROOT``) the run is registered with ``git_sha = ''`` and ``git_dirty = False``.

Expected after the fix: dirty runs carry the porcelain status and a ``git_diff.patch``;
untracked files under code directories make a run dirty; an unknown git state is never
reported as clean.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import cif.experiments.registry as registry


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=audit@example.org", "-c", "user.name=audit", *args],
        check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "src" / "mod.py").write_text("x = 1\n")
    _git(root, "add", "src/mod.py")
    _git(root, "commit", "-q", "-m", "init")
    monkeypatch.setattr(registry, "REPO_ROOT", root)
    return root


def _manifest(run) -> dict:
    return json.loads((run.run_dir / "manifest.json").read_text())


def test_dirty_run_records_what_was_modified(repo, tmp_path):
    (repo / "src" / "mod.py").write_text("x = 2\n")  # uncommitted change of tracked code
    with registry.start_run("E0", "dirty", {"a": 1}, runs_dir=tmp_path / "runs") as run:
        pass
    m = _manifest(run)
    assert m["git_dirty"] is True
    status = m.get("git_status")
    assert status and any("src/mod.py" in line for line in status), (
        "a dirty run must list the modified files (manifest['git_status'])"
    )
    patch = run.run_dir / "git_diff.patch"
    assert patch.exists() and "x = 2" in patch.read_text(), (
        "a dirty run must store the uncommitted diff (git_diff.patch) to be reproducible"
    )


def test_untracked_code_marks_the_run_dirty(repo, tmp_path):
    (repo / "src" / "new_module.py").write_text("y = 1\n")  # never committed, may be imported
    with registry.start_run("E0", "untracked", {"a": 1}, runs_dir=tmp_path / "runs") as run:
        pass
    assert _manifest(run)["git_dirty"] is True, (
        "untracked code under src/ is not part of git_sha, the run is not clean"
    )


def test_unknown_git_state_is_not_reported_as_clean(tmp_path, monkeypatch):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    monkeypatch.setattr(registry, "REPO_ROOT", plain)
    with registry.start_run("E0", "nogit", {"a": 1}, runs_dir=tmp_path / "runs") as run:
        pass
    m = _manifest(run)
    assert not (m["git_sha"] == "" and m["git_dirty"] is False), (
        "outside a git checkout the registry writes git_sha='' and git_dirty=False (a 'clean' run)"
    )
