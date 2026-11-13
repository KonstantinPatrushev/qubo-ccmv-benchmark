"""Experiment runner and cluster ingest (R8; contracts in .claude/skills/run-experiment and hpc-job).

All runs use the toy experiment ``cif.experiments.toy`` and temporary run directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cif.experiments.registry as registry
from cif.experiments import ingest, run, toy

CONFIG = """\
stream: E0
name: toy
experiment: cif.experiments.toy
seeds: [1, 2, 3]
inputs:
  data: {data}
smoke:
  seeds: [1]
  params:
    n: [10]
params:
  n: [10, 20]
"""


def _config(tmp_path: Path, **params) -> Path:
    data = tmp_path / "provenance.json"
    data.write_text('{"dataset": "x"}')
    text = CONFIG.format(data=data)
    for key, value in params.items():
        text += f"  {key}: {json.dumps(value)}\n"
    path = tmp_path / "toy.yaml"
    path.write_text(text)
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _only_run(runs_dir: Path) -> Path:
    dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_cell_key_is_stable_and_order_independent():
    assert run.cell_key({"n": 10, "seed": 1}) == run.cell_key({"seed": 1, "n": 10})
    assert run.cell_key({"n": 10, "seed": 1}) != run.cell_key({"n": 10, "seed": 2})


def test_local_run_registers_rows_config_and_log(tmp_path):
    cfg_path = _config(tmp_path)
    runs = tmp_path / "runs"
    assert run.main(["--config", str(cfg_path), "--runs-dir", str(runs), "--allow-dirty"]) == 0
    run_dir = _only_run(runs)
    rows = _rows(run_dir / "results.jsonl")
    assert len(rows) == 6
    assert {(r["n"], r["seed"]) for r in rows} == {(n, s) for n in (10, 20) for s in (1, 2, 3)}
    assert all(r["cell_key"] == run.cell_key({"n": r["n"], "seed": r["seed"]}) for r in rows)
    assert all(r["host"] for r in rows)  # R10: cluster shards come from different nodes
    assert (run_dir / "config.yaml").read_text() == cfg_path.read_text()
    assert (run_dir / "log.txt").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["mode"] == "full" and manifest["n_cells"] == 6 and manifest["n_errors"] == 0
    assert manifest["inputs"]["data"]["sha256"]
    reg = _rows(runs / "registry.jsonl")
    assert reg[-1]["run_id"] == run_dir.name


def test_full_run_requires_a_clean_tree_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "_git_state", lambda: {
        "sha": "x", "branch": "b", "available": True, "dirty": True, "status": [" M src/a.py"],
        "untracked_code": []})
    with pytest.raises(RuntimeError, match="dirty"):
        run.main(["--config", str(_config(tmp_path)), "--runs-dir", str(tmp_path / "runs")])
    # a smoke run is allowed on a dirty tree
    assert run.main(["--config", str(_config(tmp_path)), "--runs-dir", str(tmp_path / "runs"),
                     "--smoke"]) == 0


def test_smoke_applies_overrides_and_is_marked(tmp_path):
    runs = tmp_path / "runs"
    assert run.main(["--config", str(_config(tmp_path)), "--runs-dir", str(runs), "--smoke"]) == 0
    run_dir = _only_run(runs)
    assert len(_rows(run_dir / "results.jsonl")) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "smoke"
    effective = json.loads((run_dir / "config.json").read_text())
    assert "smoke" not in effective and effective["mode"] == "smoke"


def test_dry_run_creates_nothing(tmp_path, capsys):
    runs = tmp_path / "runs"
    assert run.main(["--config", str(_config(tmp_path)), "--runs-dir", str(runs), "--dry-run"]) == 0
    assert not runs.exists()
    assert "6 cells" in capsys.readouterr().out


def test_failed_cell_is_recorded_and_the_exit_code_is_nonzero(tmp_path):
    runs = tmp_path / "runs"
    code = run.main(["--config", str(_config(tmp_path, fail_on=[20])), "--runs-dir", str(runs),
                     "--allow-dirty"])
    assert code == 1
    run_dir = _only_run(runs)
    rows = _rows(run_dir / "results.jsonl")
    errors = [r for r in rows if r.get("error")]
    assert len(errors) == 3 and all("toy failure" in r["error"] for r in errors)
    assert json.loads((run_dir / "manifest.json").read_text())["n_errors"] == 3


def test_shards_partition_the_grid_and_restart_skips_done_cells(tmp_path):
    cfg_path = _config(tmp_path)
    out0, out1 = tmp_path / "s0.jsonl", tmp_path / "s1.jsonl"
    for i, out in enumerate((out0, out1)):
        assert run.main(["--config", str(cfg_path), "--shard", str(i), "--n-shards", "2",
                         "--out", str(out)]) == 0
    k0 = {r["cell_key"] for r in _rows(out0)}
    k1 = {r["cell_key"] for r in _rows(out1)}
    assert k0.isdisjoint(k1) and len(k0 | k1) == 6
    assert "run_id" not in _rows(out0)[0]  # shards are not registered runs
    # restart: nothing to do
    assert run.main(["--config", str(cfg_path), "--shard", "0", "--n-shards", "2",
                     "--out", str(out0)]) == 0
    assert len(_rows(out0)) == len(k0)
    # a lost row is recomputed, the others are not
    kept = _rows(out0)[1:]
    out0.write_text("".join(json.dumps(r) + "\n" for r in kept))
    assert run.main(["--config", str(cfg_path), "--shard", "0", "--n-shards", "2",
                     "--out", str(out0)]) == 0
    assert {r["cell_key"] for r in _rows(out0)} == k0 and len(_rows(out0)) == len(k0)
    meta = json.loads(out0.with_suffix(".meta.json").read_text())
    assert len(meta["sessions"]) == 3
    assert meta["sessions"][0]["inputs"]["data"]["sha256"]


def test_workers_give_the_same_rows(tmp_path):
    cfg_path = _config(tmp_path)
    one, two = tmp_path / "w1.jsonl", tmp_path / "w2.jsonl"
    assert run.main(["--config", str(cfg_path), "--out", str(one)]) == 0
    assert run.main(["--config", str(cfg_path), "--out", str(two), "--workers", "2"]) == 0

    def strip(rows):
        return sorted((r["cell_key"], r["value"]) for r in rows)

    assert strip(_rows(one)) == strip(_rows(two))


def test_ingest_merges_shards_dedupes_and_registers(tmp_path, monkeypatch):
    cfg_path = _config(tmp_path, fail_on=[20])
    shards = tmp_path / "incoming"
    shards.mkdir()
    monkeypatch.setenv("CIF_BUNDLE_SHA", "abc1234")  # the bundle the cluster job ran
    for i in range(2):
        run.main(["--config", str(cfg_path), "--shard", str(i), "--n-shards", "2",
                  "--out", str(shards / f"s{i}.jsonl")])
    # the failed n=20 cells are retried after a fix: append good rows for them to shard 0
    fixed = [{"cell_key": run.cell_key(c), **c, **toy.run_cell(c, {"params": {}})}
             for c in toy.cells({"params": {"n": [20]}, "seeds": [1, 2, 3]})]
    with (shards / "s0.jsonl").open("a") as f:
        f.writelines(json.dumps(r) + "\n" for r in fixed)
    (shards / "BUNDLE_SHA").write_text("abc1234\n")
    (shards / "job-1_0.log").write_text("host=node7 sha=abc1234 job=1 task=0\n")
    runs = tmp_path / "runs"
    code = ingest.main(["--config", str(cfg_path), "--shards", str(shards / "s0.jsonl"),
                        str(shards / "s1.jsonl"), "--bundle-sha-file", str(shards / "BUNDLE_SHA"),
                        "--logs", str(shards / "job-1_0.log"), "--runs-dir", str(runs)])
    assert code == 0
    run_dir = _only_run(runs)
    rows = _rows(run_dir / "results.jsonl")
    assert len(rows) == 6 and not any(r.get("error") for r in rows)
    assert all(r["run_id"] == run_dir.name for r in rows)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["mode"] == "ingested"
    assert manifest["bundle_sha"] == "abc1234"
    assert manifest["n_duplicates"] == 3 and manifest["n_missing"] == 0
    assert manifest["shard_inputs_match"] is True
    assert (run_dir / "ingest" / "job-1_0.log").exists()
    assert (run_dir / "ingest" / "s0.jsonl").exists()


def test_ingest_refuses_an_incomplete_grid(tmp_path, capsys):
    cfg_path = _config(tmp_path)
    out = tmp_path / "s0.jsonl"
    run.main(["--config", str(cfg_path), "--shard", "0", "--n-shards", "2", "--out", str(out)])
    runs = tmp_path / "runs"
    assert ingest.main(["--config", str(cfg_path), "--shards", str(out),
                        "--runs-dir", str(runs)]) == 1
    assert not runs.exists()
    assert "missing" in capsys.readouterr().out
    assert ingest.main(["--config", str(cfg_path), "--shards", str(out), "--runs-dir", str(runs),
                        "--allow-incomplete"]) == 0
    manifest = json.loads((_only_run(runs) / "manifest.json").read_text())
    assert manifest["n_missing"] == 3


def test_shard_sessions_record_the_machine_state(tmp_path):
    """R10: a session says which CPU, governor and load the rows of the shard were timed under."""
    out = tmp_path / "s0.jsonl"
    assert run.main(["--config", str(_config(tmp_path)), "--out", str(out)]) == 0
    session = json.loads(out.with_suffix(".meta.json").read_text())["sessions"][0]
    assert set(session["machine"]) >= {"cpu", "governors", "loadavg", "affinity", "n_cpus"}
    assert len(session["loadavg_end"]) == 3  # written when the session finishes


def test_ingest_summarises_the_machine_state_of_the_sessions(tmp_path):
    cfg_path = _config(tmp_path)
    shards = tmp_path / "incoming"
    shards.mkdir()
    for i in range(2):
        run.main(["--config", str(cfg_path), "--shard", str(i), "--n-shards", "2",
                  "--out", str(shards / f"s{i}.jsonl")])
    # the second shard ran under a different governor and a heavy load, on another CPU
    meta_path = shards / "s1.meta.json"
    meta = json.loads(meta_path.read_text())
    machine = meta["sessions"][0]["machine"]
    machine.update(governors={"schedutil": 64}, loadavg=[9.5, 3.0, 1.0], cpu="Xeon Silver 4216")
    meta["sessions"][0]["loadavg_end"] = [12.25, 5.0, 2.0]
    meta_path.write_text(json.dumps(meta))
    runs = tmp_path / "runs"
    assert ingest.main(["--config", str(cfg_path), "--shards", str(shards / "s0.jsonl"),
                        str(shards / "s1.jsonl"), "--runs-dir", str(runs)]) == 0
    manifest = json.loads((_only_run(runs) / "manifest.json").read_text())
    assert "schedutil" in manifest["cpu_governors"]
    assert "Xeon Silver 4216" in manifest["cpu_models"]
    assert manifest["loadavg_start_max"] >= 9.5 and manifest["loadavg_end_max"] == 12.25
    assert manifest["n_sessions"] == 2
