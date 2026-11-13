"""Tests for the run registry (R8)."""

from __future__ import annotations

import json

import pytest

from cif.experiments.registry import RunContext, config_hash, start_run


def test_run_lifecycle_writes_manifest_results_and_registry(tmp_path):
    cfg = {"experiment": "toy", "seeds": [1, 2], "N": [10]}
    with start_run("E0", "toy", cfg, runs_dir=tmp_path, seeds=[1, 2]) as run:
        assert isinstance(run, RunContext)
        run.log_row({"N": 10, "seed": 1, "objective": -0.1})
        run.log_row({"N": 10, "seed": 2, "objective": -0.2})
    run_dir = tmp_path / run.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["n_rows"] == 2
    assert manifest["config_hash"] == config_hash(cfg)
    assert "git_sha" in manifest and "git_dirty" in manifest
    assert "numpy" in manifest["packages"]
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in rows] == [1, 2]
    assert all(r["run_id"] == run.run_id for r in rows)
    reg = [json.loads(line) for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert reg[-1]["run_id"] == run.run_id and reg[-1]["status"] == "completed"


def test_failed_run_is_registered_as_failed(tmp_path):
    with pytest.raises(RuntimeError), start_run("E0", "boom", {"a": 1}, runs_dir=tmp_path) as run:
        raise RuntimeError("solver crashed")
    manifest = json.loads((tmp_path / run.run_id / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "solver crashed" in manifest["error"]


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_mark_aborted_closes_a_dead_run_and_registers_it(tmp_path):
    import cif.experiments.registry as registry

    run_dir = tmp_path / "20260101-000000-E0-x-abc123"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_dir.name, "status": "running"}))
    (run_dir / "results.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    m = registry.mark_aborted(run_dir, "machine reboot", runs_dir=tmp_path)
    assert m["status"] == "aborted" and m["n_rows"] == 2 and "machine reboot" in m["error"]
    reg = [json.loads(line) for line in (tmp_path / "registry.jsonl").read_text().splitlines()]
    assert reg[-1]["status"] == "aborted"
    with pytest.raises(ValueError, match="not 'running'"):
        registry.mark_aborted(run_dir, "again", runs_dir=tmp_path)


def test_inputs_are_hashed_into_the_manifest(tmp_path):
    """Data lineage: a run records the SHA-256 of its inputs (e.g. dataset provenance files)."""
    import hashlib

    prov = tmp_path / "data" / "provenance.json"
    prov.parent.mkdir()
    prov.write_text('{"dataset": "moex_v2"}')
    with start_run("E0", "lineage", {"a": 1}, runs_dir=tmp_path / "runs",
                   inputs={"moex_v2": prov}) as run:
        pass
    manifest = json.loads((tmp_path / "runs" / run.run_id / "manifest.json").read_text())
    entry = manifest["inputs"]["moex_v2"]
    assert entry["sha256"] == hashlib.sha256(prov.read_bytes()).hexdigest()
    assert entry["bytes"] == prov.stat().st_size
    assert entry["path"].endswith("data/provenance.json")


def test_missing_input_fails_before_the_run_is_created(tmp_path):
    with pytest.raises(FileNotFoundError), start_run(
        "E0", "lineage", {"a": 1}, runs_dir=tmp_path, inputs={"x": tmp_path / "nope.json"}
    ):
        pass
    assert not any(tmp_path.iterdir())


def test_manifest_records_thread_settings(tmp_path, monkeypatch):
    """R10: the thread settings of BLAS/OpenMP go into the manifest next to the CPU model."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with start_run("E0", "threads", {"a": 1}, runs_dir=tmp_path) as run:
        pass
    manifest = json.loads((tmp_path / run.run_id / "manifest.json").read_text())
    assert manifest["threads"]["env"]["OMP_NUM_THREADS"] == "1"
    assert manifest["threads"]["env"]["MKL_NUM_THREADS"] is None
    assert manifest["threads"]["cpu_count"] >= 1
    assert isinstance(manifest["threads"]["pools"], list)  # pools in force (audit TH-1)


def test_manifest_records_the_cpu_frequency_governor(tmp_path):
    """R10: frequency scaling (e.g. ``powersave``) changes wall times; record it (None if unknown)."""
    with start_run("E0", "governor", {"a": 1}, runs_dir=tmp_path) as run:
        pass
    manifest = json.loads((tmp_path / run.run_id / "manifest.json").read_text())
    assert "cpu_governor" in manifest
    assert manifest["cpu_governor"] is None or isinstance(manifest["cpu_governor"], str)


def test_machine_state_reports_cpu_governors_load_and_affinity():
    """R10, plan §6: the manifest says which governor and how much load the timings ran under."""
    import os

    import cif.experiments.registry as registry

    state = registry.machine_state()
    assert state["n_cpus"] == os.cpu_count()
    assert state["cpu"]
    assert isinstance(state["governors"], dict)  # {governor: number of CPUs}, {} if unknown
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in state["governors"].items())
    assert len(state["loadavg"]) == 3 and all(x >= 0.0 for x in state["loadavg"])
    if hasattr(os, "sched_getaffinity"):  # Linux, as on the run server; elsewhere no affinity is recorded
        assert state["affinity"] == sorted(os.sched_getaffinity(0))
    else:
        assert state["affinity"] is None


def test_governors_are_counted_over_all_cpus_of_a_sysfs_tree(tmp_path):
    import cif.experiments.registry as registry

    for cpu, governor in enumerate(["schedutil", "schedutil", "powersave", "performance"]):
        d = tmp_path / f"cpu{cpu}" / "cpufreq"
        d.mkdir(parents=True)
        (d / "scaling_governor").write_text(governor + "\n")
    (tmp_path / "cpu4").mkdir()  # an offline CPU without cpufreq
    (tmp_path / "cpufreq").mkdir()  # not a CPU directory
    assert registry._governors(tmp_path) == {"schedutil": 2, "powersave": 1, "performance": 1}
    assert registry._governors(tmp_path / "missing") == {}


def test_a_registered_run_records_the_machine_state_at_start_and_the_load_at_the_end(tmp_path):
    with start_run("E0", "machine", {"a": 1}, runs_dir=tmp_path) as run:
        pass
    manifest = json.loads((tmp_path / run.run_id / "manifest.json").read_text())
    assert set(manifest["machine"]) >= {"cpu", "n_cpus", "governors", "loadavg", "affinity"}
    assert len(manifest["loadavg_end"]) == 3
