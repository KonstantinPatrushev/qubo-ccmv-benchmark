"""Audit of the experiment runner, cluster ingest and run manifest (2026-10-08): L-1 … L-3, TH-1.

Report: docs/reviews/2026-10-08-audit-e0-additions.md.

* L-1 (MAJOR): ``cif.experiments.run`` hashes the input file (a dataset's ``provenance.json``) but
  never checks that the data files match the hashes it records (``cif.data.provenance verify``).
  A run on stale or edited data is registered as ``completed`` under the provenance of other data
  (the R8 failure mode: recorded SHA-256 that do not match the files).
* L-2 (MAJOR): ``cif.experiments.ingest`` merges shard rows without checking that the shard sessions
  ran the local config (``config_hash``) and one code bundle (``bundle_sha``); the manifest then
  names the local config and the first bundle. A shard records its session only at the very end,
  so the rows of a killed (preempted) session have no session record at all.
* L-3 (MINOR): dedup and completeness. Two successful rows of one cell with different results are
  resolved silently by file order; an input-hash mismatch only prints a warning; a run with failed
  cells is ``completed`` in ``runs/registry.jsonl`` with nothing that shows the failures.
* TH-1 (MAJOR): the manifest records the thread environment variables, not the thread pools in
  force. ``python -m cif.experiments.run`` imports numpy (BLAS) before any experiment module, so a
  module that applies ``single_thread_env()`` as ``cif.experiments.timing`` suggests gets a manifest
  that says 1 thread while OpenBLAS runs on every core. Nothing enforces one thread (plan §6).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cif.experiments.registry as registry
from cif.experiments import ingest, run, toy

REPO = Path(__file__).resolve().parents[2]

CONFIG = """\
stream: AUDIT
name: toy
experiment: cif.experiments.toy
seeds: [1, 2, 3]
inputs:
  data: {data}
params:
  n: [10, 20]
"""


def _config(tmp_path: Path, name: str = "toy.yaml", extra: str = "", data: Path | None = None) -> Path:
    if data is None:
        data = tmp_path / "provenance.json"
        if not data.exists():
            data.write_text('{"dataset": "x"}')
    path = tmp_path / name
    path.write_text(CONFIG.format(data=data) + extra)
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _registered(runs: Path) -> list[dict]:
    reg = runs / "registry.jsonl"
    return _rows(reg) if reg.exists() else []


def _shards(cfg_path: Path, out_dir: Path, n_shards: int = 2) -> list[Path]:
    out_dir.mkdir(exist_ok=True)
    paths = []
    for i in range(n_shards):
        out = out_dir / f"s{i}.jsonl"
        assert run.main(["--config", str(cfg_path), "--shard", str(i), "--n-shards", str(n_shards),
                         "--out", str(out)]) == 0
        paths.append(out)
    return paths


# ------------------------------------------------------------------------------------------ L-1


def test_l1_run_refuses_input_data_that_do_not_match_their_provenance(tmp_path):
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    payload = data_dir / "returns.parquet"
    payload.write_bytes(b"version 1 of the data")
    prov = data_dir / "provenance.json"
    prov.write_text(json.dumps({
        "dataset": "toy",
        "files": {"processed": {"returns.parquet": {
            "sha256": hashlib.sha256(b"version 1 of the data").hexdigest()}}},
    }))
    payload.write_bytes(b"version 2, rebuilt or edited after the provenance was written")

    runs = tmp_path / "runs"
    cfg = _config(tmp_path, data=prov)
    try:
        code = run.main(["--config", str(cfg), "--runs-dir", str(runs), "--allow-dirty"])
    except Exception:  # refusing by an exception is fine too
        code = None
    completed = [e for e in _registered(runs) if e["status"] == "completed"]
    assert code != 0 and not completed, (
        "a run on data whose SHA-256 differs from its provenance.json was registered as completed"
    )


# ------------------------------------------------------------------------------------------ L-2


def test_l2_ingest_rejects_shards_computed_under_another_config(tmp_path):
    cluster_cfg = _config(tmp_path, "cluster.yaml", extra="  budget_s: 10\n")
    shards = _shards(cluster_cfg, tmp_path / "incoming")
    local_cfg = _config(tmp_path, "local.yaml", extra="  budget_s: 60\n")  # same grid, other params
    assert registry.config_hash(run.effective_config(run.load_config(cluster_cfg))) != \
        registry.config_hash(run.effective_config(run.load_config(local_cfg)))
    runs = tmp_path / "runs"
    code = ingest.main(["--config", str(local_cfg), "--shards", *map(str, shards),
                        "--runs-dir", str(runs)])
    assert code != 0 and not _registered(runs), (
        "rows computed with budget_s=10 were registered under the config hash of budget_s=60"
    )


def test_l2_ingest_rejects_rows_from_different_code_bundles(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for i, sha in enumerate(("aaaaaaa", "bbbbbbb")):
        monkeypatch.setenv("CIF_BUNDLE_SHA", sha)
        assert run.main(["--config", str(cfg), "--shard", str(i), "--n-shards", "2",
                         "--out", str(incoming / f"s{i}.jsonl")]) == 0
    monkeypatch.delenv("CIF_BUNDLE_SHA")
    (incoming / "BUNDLE_SHA").write_text("aaaaaaa\n")
    runs = tmp_path / "runs"
    code = ingest.main(["--config", str(cfg), "--shards", str(incoming / "s0.jsonl"),
                        str(incoming / "s1.jsonl"), "--bundle-sha-file",
                        str(incoming / "BUNDLE_SHA"), "--runs-dir", str(runs)])
    assert code != 0 and not _registered(runs), (
        "half of the rows came from bundle bbbbbbb, the run was registered as bundle aaaaaaa"
    )


def test_l2_a_killed_shard_session_leaves_a_session_record(tmp_path, monkeypatch):
    """SLURM preemption kills the process: the rows already written must be attributable."""
    cfg = _config(tmp_path)
    out = tmp_path / "s0.jsonl"
    calls = {"n": 0}
    real = toy.run_cell

    def dies_on_second_cell(cell, c):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt  # stands for SIGTERM/SIGKILL: no ``finally`` of the runner
        return real(cell, c)

    monkeypatch.setattr(toy, "run_cell", dies_on_second_cell)
    with pytest.raises(KeyboardInterrupt):
        run.main(["--config", str(cfg), "--shard", "0", "--n-shards", "1", "--out", str(out)])
    assert len(_rows(out)) == 1  # one row reached the shard file
    meta = out.with_suffix(".meta.json")
    sessions = json.loads(meta.read_text())["sessions"] if meta.exists() else []
    assert sessions, "rows were written by a session that left no host/bundle/inputs record"


# ------------------------------------------------------------------------------------------ L-3


def test_l3_ingest_does_not_silently_pick_between_conflicting_rows(tmp_path):
    cfg = _config(tmp_path)
    shards = _shards(cfg, tmp_path / "incoming")
    first = _rows(shards[0])[0]
    other = {**first, "value": first["value"] + 1.0}  # the same cell, another successful result
    with shards[1].open("a") as fh:
        fh.write(json.dumps(other) + "\n")
    runs = tmp_path / "runs"
    code = ingest.main(["--config", str(cfg), "--shards", *map(str, shards), "--runs-dir", str(runs)])
    assert code != 0, "two different successful results for one cell were merged by file order"


def test_l3_ingest_refuses_shards_computed_on_other_inputs(tmp_path):
    cfg = _config(tmp_path)
    shards = _shards(cfg, tmp_path / "incoming")
    (tmp_path / "provenance.json").write_text('{"dataset": "x", "rebuilt": true}')
    runs = tmp_path / "runs"
    code = ingest.main(["--config", str(cfg), "--shards", *map(str, shards), "--runs-dir", str(runs)])
    assert code != 0 and not _registered(runs), "input hash mismatch only printed a warning"


def test_l3_failed_cells_are_visible_in_the_run_registry(tmp_path):
    cfg = _config(tmp_path, extra="  fail_on: [20]\n")
    runs = tmp_path / "runs"
    assert run.main(["--config", str(cfg), "--runs-dir", str(runs), "--allow-dirty"]) == 1
    entry = _registered(runs)[-1]
    assert entry.get("n_errors") == 3 or entry["status"] != "completed", (
        f"registry entry of a run with 3 failed cells: {entry}"
    )


# ------------------------------------------------------------------------------------------ TH-1

_THREADS_PROBE = """
import json, os, sys
sys.path.insert(0, os.path.join(sys.argv[2], "src"))
import cif.experiments.run  # what `python -m cif.experiments.run` imports before the experiment
from cif.experiments.timing import single_thread_env
os.environ.update(single_thread_env())  # an experiment module following cif.experiments.timing
from threadpoolctl import threadpool_info
from cif.experiments import registry
with registry.start_run("AUDIT", "threads", {}, runs_dir=sys.argv[1]) as ctx:
    pass
manifest = json.loads((ctx.run_dir / "manifest.json").read_text())
print(json.dumps({"manifest": manifest["threads"],
                  "pools": {p["internal_api"]: p["num_threads"] for p in threadpool_info()}}))
"""


def test_th1_manifest_threads_are_the_thread_pools_in_force(tmp_path):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one core: a multi-threaded pool cannot be told from a single-threaded one")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", _THREADS_PROBE, str(tmp_path / "runs"), str(REPO)],
                         capture_output=True, text=True, env=env, check=True, cwd=REPO)
    probe = json.loads(out.stdout.strip().splitlines()[-1])
    claims_single = probe["manifest"].get("OPENBLAS_NUM_THREADS") == "1"
    in_force = max(probe["pools"].values(), default=1)
    assert not (claims_single and in_force > 1), (
        f"manifest threads {probe['manifest']} while the pools in force are {probe['pools']}"
    )
