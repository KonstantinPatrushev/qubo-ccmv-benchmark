"""Launcher for a server without a job scheduler (``cif.experiments.launch``; ANALYSIS_PLAN §6).

The launcher runs shards of a config with a fixed number of single-thread jobs, optionally pinned to
physical cores, resumes after an interruption without duplicating rows, keeps timing stages to at most
half of the physical cores and writes the machine state (governor, load average) into a launch manifest.
Real jobs run the toy experiment ``cif.experiments.toy``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from cif.experiments import launch, run

#: the launcher pins the jobs of a timing stage to the cores of the Linux run server: these tests need the CPU affinity
#: (os.sched_getaffinity) and the CPU topology of Linux
LINUX_ONLY = pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only: CPU affinity and topology")

CONFIG = {
    "stream": "E0", "name": "toy", "experiment": "cif.experiments.toy", "seeds": [1, 2, 3], "inputs": {},
    "params": {"n": [10, 20]},
    "launch": {"timing": True, "jobs": 2, "shards": 3},
}


@pytest.fixture(autouse=True)
def _private_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("CIF_LAUNCH_LOCK", str(tmp_path / "launch.lock"))


@pytest.fixture(autouse=True)
def _idle_probe(monkeypatch):
    """The probe of the cores a pinned launch is about to use looks at the real machine for a second: the tests
    that are not about it see an idle machine (``test_..._busy_...`` below put their own probe in)."""
    monkeypatch.setattr(launch, "_probe_busy", lambda *args, **kwargs: {})


def _config(tmp_path: Path, **over) -> Path:
    cfg = json.loads(json.dumps(CONFIG))
    for key, value in over.items():
        if key == "params":
            cfg["params"].update(value)
        elif key == "launch":
            cfg["launch"].update(value)
        else:
            cfg[key] = value
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_rows(out_dir: Path) -> list[dict]:
    return [r for p in sorted(out_dir.glob("*_shard*.jsonl")) for r in _rows(p)]


# --------------------------------------------------------------------------------------
# Cores
# --------------------------------------------------------------------------------------


def test_cores_are_parsed_from_lists_and_ranges():
    assert launch.parse_cores("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert launch.parse_cores("5") == [5] and launch.parse_cores(" 2 - 4 ") == [2, 3, 4]
    for bad in ("", "a", "3-1", "1,,2", "-1"):
        with pytest.raises(ValueError, match="cores"):
            launch.parse_cores(bad)


def _sysfs(tmp_path: Path, layout: dict[int, tuple[int, int]]) -> Path:
    for cpu, (package, core) in layout.items():
        topo = tmp_path / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text(f"{package}\n")
        (topo / "core_id").write_text(f"{core}\n")
    (tmp_path / "cpufreq").mkdir()
    return tmp_path


def test_physical_cores_are_the_lowest_logical_cpu_of_each_core(tmp_path):
    # 2 sockets × 2 cores × 2 threads, numbered as on the target: threads 0-3 | 4-7 first, siblings after
    layout = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (0, 0), 5: (0, 1), 6: (1, 0), 7: (1, 1)}
    topo = launch.detect_topology(_sysfs(tmp_path, layout))
    assert topo == {0: [0, 1], 1: [2, 3]}  # package → physical cores
    assert launch.detect_topology(tmp_path / "missing") == {}


SIBLINGS = {0: (0, 0), 1: (0, 0), 2: (0, 1), 3: (0, 1)}  # two cores, hyperthreads 0/1 and 2/3


def test_hyperthread_siblings_of_the_listed_cpus_are_found(tmp_path):
    root = _sysfs(tmp_path, SIBLINGS)
    assert launch.hyperthread_siblings([0, 1, 2, 3], root) == {1: 0, 3: 2}
    assert launch.hyperthread_siblings([0, 2], root) == {}
    assert launch.hyperthread_siblings([0, 99], root) == {}  # a CPU the system does not know is no sibling
    assert launch.hyperthread_siblings([0, 1], tmp_path / "missing") == {}  # no topology, no claim


def test_a_core_list_with_hyperthreads_is_reported_and_two_pinned_jobs_never_share_a_core(tmp_path, monkeypatch,
                                                                                         capsys):
    monkeypatch.setattr(launch, "_CPU_DIR", _sysfs(tmp_path / "sys", SIBLINGS))
    cfg = _config(tmp_path)
    base = ["plan", "--config", str(cfg), "--cores", "0-3", "--allow-oversubscribe"]
    assert launch.main(base) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "hyperthread" in out and "{1: 0, 3: 2}" in out
    # pinned: jobs on cpus 0 and 2 are on different cores; four jobs would put two on one core
    assert launch.main([*base, "--pin", "--jobs", "2"]) == 0
    assert "cores of the jobs: [0, 2]" in capsys.readouterr().out
    with pytest.raises(launch.LaunchError, match="physical core"):
        launch.main([*base, "--pin", "--jobs", "4"])
    # the warning goes into the manifest of a run as well
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(cfg), "--out-dir", str(out_dir), "--cores", "0-3",
                        "--allow-oversubscribe", "--poll", "0.05"]) == 0
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert any("hyperthread" in w for w in manifest["warnings"])


def test_cpu_counters_are_read_from_proc_stat_and_become_busy_shares(tmp_path):
    stat = tmp_path / "stat"
    stat.write_text("cpu  1 2 3 4 5 6 7 8 0 0\ncpu0 10 0 20 70 0 0 0 0 0 0\ncpu1 5 5 5 80 5 0 0 0 0 0\nintr 1 2\n")
    # busy = all but idle and iowait, over the first eight counters; the aggregate line is not a CPU
    assert launch._cpu_counters(stat) == {0: (30, 100), 1: (15, 100)}
    assert launch._cpu_counters(tmp_path / "missing") == {}
    first = {0: (100, 1000), 1: (0, 1000), 2: (50, 500)}
    second = {0: (190, 1100), 1: (10, 1100), 3: (5, 10)}  # CPU 2 vanished, CPU 3 is new: neither is claimed
    assert launch.busy_shares(first, second) == {0: 0.9, 1: 0.1}
    assert launch.busy_shares(first, first) == {}  # no time passed


def test_a_core_is_busy_if_it_or_a_hyperthread_sibling_is(tmp_path):
    root = _sysfs(tmp_path, SIBLINGS)  # cores {0, 1} and {2, 3}
    shares = {0: 0.05, 1: 0.02, 2: 0.0, 3: 0.6}
    assert launch.busy_cores([0, 2], shares, root) == {2: {3: 0.6}}  # the sibling thread of core 2
    assert launch.busy_cores([0, 2], {0: 0.3}, root) == {0: {0: 0.3}}
    assert launch.busy_cores([0, 2], {}, root) == {}
    assert launch.busy_cores([0, 2], shares, root, threshold=0.7) == {}
    # no topology: only the listed CPUs themselves are looked at
    assert launch.busy_cores([2], {2: 0.5, 3: 0.9}, tmp_path / "missing") == {2: {2: 0.5}}


def test_pinned_jobs_on_busy_cores_draw_a_warning_that_names_the_cores_to_leave_out(tmp_path, monkeypatch, capsys):
    """The server is shared (plan §12): before the jobs are bound to cores, the cores are looked at."""
    monkeypatch.setattr(launch, "_CPU_DIR", _sysfs(tmp_path / "sys", SIBLINGS))
    monkeypatch.setattr(launch, "_probe_busy", lambda *args, **kwargs: {0: 0.02, 2: 0.1, 3: 0.7})
    cfg = _config(tmp_path)
    base = ["plan", "--config", str(cfg), "--cores", "0,2", "--pin", "--allow-oversubscribe"]
    assert launch.main([*base, "--jobs", "2"]) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "busy" in out and "CPU 3 70%" in out and "--exclude-cores 2" in out
    # without the busy core there is nothing to warn about
    assert launch.main([*base, "--jobs", "1", "--exclude-cores", "2"]) == 0
    out = capsys.readouterr().out
    assert "busy" not in out and "cores of the jobs: [0]" in out
    # jobs that are not bound to cores float over the machine: the probe has nothing to say
    monkeypatch.setattr(launch, "_probe_busy", lambda *args, **kwargs: pytest.fail("probed without --pin"))
    assert launch.main(["plan", "--config", str(cfg), "--cores", "0,2", "--allow-oversubscribe"]) == 0


@LINUX_ONLY
def test_the_busy_cores_are_recorded_in_the_manifest_of_a_run(tmp_path, monkeypatch):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two allowed CPUs")
    first, second, sibling = allowed[0], allowed[1], 4096  # a made-up sibling thread of the second core
    monkeypatch.setattr(launch, "_CPU_DIR", _sysfs(tmp_path / "sys", {first: (0, 0), second: (0, 1), sibling: (0, 1)}))
    monkeypatch.setattr(launch, "_probe_busy", lambda *args, **kwargs: {sibling: 0.8})
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(_config(tmp_path)), "--out-dir", str(out_dir), "--jobs", "2", "--pin",
                        "--cores", f"{first},{second}", "--allow-oversubscribe", "--poll", "0.05"]) == 0
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["cores"] == [first, second]
    assert any("busy" in w and f"CPU {sibling} 80%" in w for w in manifest["warnings"])
    assert manifest["excluded_cores"] == []


def test_cores_can_be_left_out_and_half_of_the_machine_still_counts_the_whole_machine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launch, "_CPU_DIR", tmp_path / "no_topology")  # a flat list of cores, whatever this machine is
    cfg = _config(tmp_path, launch={"jobs": 4})
    base = ["plan", "--config", str(cfg), "--cores", "0-7", "--pin"]
    assert launch.main(base) == 0  # 4 jobs are half of the 8 cores
    capsys.readouterr()
    assert launch.main([*base, "--exclude-cores", "0,1"]) == 0  # six cores left, four jobs still allowed
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("cores of the jobs"))
    used = json.loads(line.split(": ", 1)[1])
    assert len(used) == 4 and not {0, 1} & set(used) and "excluded" not in line
    with pytest.raises(launch.LaunchError, match="half"):  # excluding cores does not raise the limit
        launch.main([*base, "--jobs", "5", "--exclude-cores", "0"])
    with pytest.raises(launch.LaunchError, match="cores"):  # two cores left for four jobs
        launch.main([*base, "--exclude-cores", "0-5"])
    with pytest.raises(launch.LaunchError, match="--exclude-cores"):  # not in the pool
        launch.main([*base, "--exclude-cores", "9"])
    with pytest.raises(launch.LaunchError, match="--pin"):  # unpinned jobs go wherever the system puts them
        launch.main(["plan", "--config", str(cfg), "--cores", "0-7", "--exclude-cores", "0"])


def test_jobs_are_spread_over_the_sockets():
    packages = {0: list(range(16)), 1: list(range(16, 32))}
    cores = launch.choose_cores(packages, 16)
    assert len(cores) == len(set(cores)) == 16
    assert sum(c < 16 for c in cores) == sum(c >= 16 for c in cores) == 8
    assert cores[:4] == [0, 16, 1, 17]  # alternates, lowest cores first
    assert launch.choose_cores(packages, 3) == [0, 16, 1]
    assert launch.choose_cores({0: [4, 5, 6]}, 2) == [4, 5]
    with pytest.raises(launch.LaunchError, match="cores"):
        launch.choose_cores({0: [0, 1]}, 3)


def test_a_flat_core_list_is_spread_evenly_without_socket_information():
    assert launch.choose_cores({None: list(range(32))}, 16) == list(range(0, 32, 2))
    assert launch.choose_cores({None: list(range(8))}, 8) == list(range(8))


def test_timing_stages_may_use_half_of_the_physical_cores():
    assert launch.max_timing_jobs(32) == 16 and launch.max_timing_jobs(4) == 2 and launch.max_timing_jobs(1) == 1
    launch.check_jobs(16, timing=True, n_physical=32)
    launch.check_jobs(28, timing=False, n_physical=32)
    with pytest.raises(launch.LaunchError, match="half"):
        launch.check_jobs(17, timing=True, n_physical=32)
    with pytest.raises(launch.LaunchError, match="physical"):
        launch.check_jobs(2, timing=True, n_physical=None)  # unknown: do not guess from logical CPUs
    with pytest.raises(launch.LaunchError, match="cores"):
        launch.check_jobs(33, timing=False, n_physical=32)
    launch.check_jobs(30, timing=True, n_physical=32, allow_oversubscribe=True)


# --------------------------------------------------------------------------------------
# One job
# --------------------------------------------------------------------------------------


def test_a_job_runs_one_shard_with_one_worker_and_one_thread(tmp_path):
    cmd, env = launch.build_command(Path("a.yaml"), 3, 16, Path("out/s003.jsonl"), core=None)
    assert cmd[:3] == [sys.executable, "-u", "-m"] and cmd[3] == "cif.experiments.run"
    assert cmd[cmd.index("--shard") + 1] == "3" and cmd[cmd.index("--n-shards") + 1] == "16"
    assert cmd[cmd.index("--workers") + 1] == "1" and cmd[cmd.index("--out") + 1] == "out/s003.jsonl"
    assert all(env[v] == "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert str(launch.registry.REPO_ROOT / "src") in env["PYTHONPATH"].split(os.pathsep)


def test_a_pinned_job_is_started_through_taskset(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/taskset" if name == "taskset" else None)
    cmd, _ = launch.build_command(Path("a.yaml"), 0, 2, Path("o.jsonl"), core=7)
    assert cmd[:3] == ["/usr/bin/taskset", "-c", "7"] and cmd[3] == sys.executable


def test_without_taskset_the_job_pins_itself(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    cmd, _ = launch.build_command(Path("a.yaml"), 0, 2, Path("o.jsonl"), core=7)
    assert cmd[0] == sys.executable and "--pin-core" not in cmd  # the child is wrapped, see spawn()
    assert launch.pin_function(7) is not None


# --------------------------------------------------------------------------------------
# Running a config
# --------------------------------------------------------------------------------------


def test_run_executes_every_shard_and_writes_the_launch_manifest(tmp_path):
    out_dir = tmp_path / "out"
    code = launch.main(["run", "--config", str(_config(tmp_path)), "--out-dir", str(out_dir),
                        "--cores", "0-3", "--load-interval", "0.2", "--poll", "0.05"])
    assert code == 0
    rows = _all_rows(out_dir)
    assert len(rows) == 6 and len({r["cell_key"] for r in rows}) == 6
    assert sorted(p.name for p in out_dir.glob("*_shard*.jsonl")) == [f"toy_shard{i:04d}.jsonl" for i in range(3)]
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["status"] == "finished" and manifest["n_cells"] == 6 and manifest["n_shards"] == 3
    assert manifest["jobs"] == 2 and manifest["pinned"] is False
    assert manifest["config_hash"] == launch.registry.config_hash(run.effective_config(run.load_config(
        tmp_path / "toy.yaml")))
    assert set(manifest["machine"]) >= {"cpu", "governors", "loadavg", "affinity"}
    assert manifest["loadavg_samples"] and manifest["loadavg_samples"][0]["loadavg"][0] >= 0
    assert manifest["loadavg_start"] and manifest["loadavg_end"] and manifest["loadavg_1m_max"] >= 0
    assert manifest["shards_failed"] == [] and len(manifest["shards"]) == 3
    assert (out_dir / "logs" / "toy_shard0000.log").exists()
    meta = json.loads((out_dir / "toy_shard0000.meta.json").read_text())
    assert meta["sessions"][0]["machine"]["governors"] is not None  # the runner logs it per session too


def test_the_bundle_sha_is_copied_next_to_the_shards(tmp_path, monkeypatch):
    """``ingest --bundle-sha-file <out-dir>/BUNDLE_SHA`` checks the shards against the code that ran them."""
    from cif.experiments import ingest

    monkeypatch.setenv("CIF_BUNDLE_SHA", "abc1234")
    cfg = str(_config(tmp_path))
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", cfg, "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]) == 0
    assert (out_dir / "BUNDLE_SHA").read_text().strip() == "abc1234"
    assert json.loads((out_dir / "launch_manifest.json").read_text())["bundle_sha"] == "abc1234"
    shards = sorted(str(p) for p in out_dir.glob("*_shard*.jsonl"))
    assert ingest.main(["--config", cfg, "--shards", *shards, "--runs-dir", str(tmp_path / "runs"),
                        "--bundle-sha-file", str(out_dir / "BUNDLE_SHA")]) == 0


@LINUX_ONLY
def test_pinned_jobs_run_on_their_core(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2:
        pytest.skip("needs two allowed CPUs")
    pool = f"{allowed[0]},{allowed[1]}"
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(_config(tmp_path)), "--out-dir", str(out_dir),
                        "--jobs", "2", "--pin", "--cores", pool, "--allow-oversubscribe", "--poll", "0.05"]) == 0
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["pinned"] is True and sorted(manifest["cores"]) == allowed[:2]
    seen = set()
    for meta_path in out_dir.glob("*.meta.json"):
        for session in json.loads(meta_path.read_text())["sessions"]:
            assert len(session["machine"]["affinity"]) == 1  # one core per job
            seen.update(session["machine"]["affinity"])
    assert seen <= set(allowed[:2])


def test_a_second_run_does_nothing_and_a_lost_row_is_the_only_one_recomputed(tmp_path):
    cfg = str(_config(tmp_path))
    out_dir = tmp_path / "out"
    args = ["run", "--config", cfg, "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]
    assert launch.main(args) == 0
    before = {p.name: p.read_text() for p in out_dir.glob("*_shard*.jsonl")}
    assert launch.main(args) == 0
    assert {p.name: p.read_text() for p in out_dir.glob("*_shard*.jsonl")} == before  # no duplicate rows
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["shards_skipped"] == [0, 1, 2] and manifest["shards_started"] == []
    # a job killed before it wrote its last row: that cell, and only that one, is computed again
    victim = out_dir / "toy_shard0001.jsonl"
    rows = _rows(victim)
    victim.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
    assert launch.main(args) == 0
    assert len(_rows(victim)) == len(rows) and {r["cell_key"] for r in _rows(victim)} == {r["cell_key"] for r in rows}
    assert len(_all_rows(out_dir)) == 6
    again = json.loads((out_dir / "launch_manifest.json").read_text())
    assert again["shards_started"] == [1] and again["shards_skipped"] == [0, 2]


def test_resuming_with_another_number_of_shards_is_refused(tmp_path):
    cfg = str(_config(tmp_path))
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", cfg, "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]) == 0
    with pytest.raises(launch.LaunchError, match="shards"):
        launch.main(["run", "--config", cfg, "--out-dir", str(out_dir), "--shards", "5", "--cores", "0-3"])
    with pytest.raises(launch.LaunchError, match="config"):
        launch.main(["run", "--config", str(_config(tmp_path, params={"n": [10, 20, 30]})),
                     "--out-dir", str(out_dir), "--cores", "0-3"])


def test_failed_cells_make_the_exit_code_nonzero_and_are_retried_on_resume(tmp_path):
    cfg = _config(tmp_path, params={"fail_on": [20]})
    out_dir = tmp_path / "out"
    args = ["run", "--config", str(cfg), "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]
    assert launch.main(args) == 1
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["status"] == "finished" and manifest["shards_failed"]
    errors = [r for r in _all_rows(out_dir) if r.get("error")]
    assert len(errors) == 3
    assert launch.main(args) == 1  # the errored cells are not "done": they are tried again
    assert len([r for r in _all_rows(out_dir) if r.get("error")]) == 6


def test_timing_stages_refuse_more_than_half_the_cores(tmp_path):
    with pytest.raises(launch.LaunchError, match="half"):
        launch.main(["run", "--config", str(_config(tmp_path)), "--out-dir", str(tmp_path / "o"),
                     "--jobs", "17", "--cores", "0-31"])
    # ... unless the stage is not a timing stage (the certification runs)
    cfg = _config(tmp_path, launch={"timing": False})
    assert launch.main(["plan", "--config", str(cfg), "--jobs", "28", "--cores", "0-31"]) == 0


def test_jobs_must_be_given_by_the_config_or_the_command_line(tmp_path):
    cfg = _config(tmp_path)
    data = yaml.safe_load(cfg.read_text())
    del data["launch"]["jobs"]
    cfg.write_text(yaml.safe_dump(data))
    with pytest.raises(launch.LaunchError, match="jobs"):
        launch.main(["plan", "--config", str(cfg), "--cores", "0-31"])
    assert launch.main(["plan", "--config", str(cfg), "--cores", "0-31", "--jobs", "4"]) == 0


def test_a_rehearsal_runs_the_smoke_grid_and_its_shards_cannot_be_ingested(tmp_path, capsys):
    from cif.experiments import ingest

    cfg = _config(tmp_path)
    data = yaml.safe_load(cfg.read_text())
    data["smoke"] = {"params": {"n": [10]}}  # 1 size × 3 seeds = 3 cells of 6
    cfg.write_text(yaml.safe_dump(data))
    assert launch.main(["plan", "--config", str(cfg), "--smoke", "--cores", "0-31"]) == 0
    assert "3 cells" in capsys.readouterr().out
    cmd, _ = launch.build_command(Path("a.yaml"), 0, 2, Path("o.jsonl"), smoke=True)
    assert "--smoke" in cmd and "--smoke" not in launch.build_command(Path("a.yaml"), 0, 2, Path("o.jsonl"))[0]
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(cfg), "--smoke", "--out-dir", str(out_dir), "--cores", "0-3",
                        "--poll", "0.05"]) == 0
    assert len(_all_rows(out_dir)) == 3
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["smoke"] is True
    shards = sorted(str(p) for p in out_dir.glob("*_shard*.jsonl"))
    assert ingest.main(["--config", str(cfg), "--shards", *shards, "--runs-dir", str(tmp_path / "runs")]) == 1
    assert not (tmp_path / "runs").exists()  # a rehearsal is never registered as a run


def test_a_confirmatory_stage_needs_to_know_which_code_it_ran(tmp_path, monkeypatch):
    """Rows of test seeds must be traceable to a code bundle: no BUNDLE_SHA, no confirmatory run."""
    monkeypatch.delenv("CIF_BUNDLE_SHA", raising=False)
    monkeypatch.setattr(launch.registry, "REPO_ROOT", tmp_path)  # no BUNDLE_SHA file there
    cfg = _config(tmp_path, params={"seed_role": "test"})
    args = ["run", "--config", str(cfg), "--out-dir", str(tmp_path / "out"), "--cores", "0-3", "--poll", "0.05"]
    with pytest.raises(launch.LaunchError, match="BUNDLE_SHA"):
        launch.main(args)
    assert not (tmp_path / "out").exists()  # nothing was started
    (tmp_path / "BUNDLE_SHA").write_text("abc1234\n")
    assert launch.main(args) == 0
    # a calibration stage and a rehearsal need no bundle
    (tmp_path / "BUNDLE_SHA").unlink()
    cal = _config(tmp_path, params={"seed_role": "calibration"})
    assert launch.main(["run", "--config", str(cal), "--out-dir", str(tmp_path / "out2"), "--cores", "0-3",
                        "--poll", "0.05"]) == 0
    assert launch.main(["plan", "--config", str(cfg), "--cores", "0-3"]) == 0  # planning is always allowed


def test_the_grid_hash_identifies_the_cells_not_their_order(tmp_path, capsys):
    """A digest of the cell keys: printed by ``plan`` (compare the laptop's with the server's before a run:
    a platform difference in the instances or in the sampled configurations would show here), recorded in
    the manifest and in the state of the output directory."""
    cells = [{"n": 10, "seed": 1}, {"n": 10, "seed": 2}, {"n": 20, "seed": 1}]
    h = launch.grid_hash(cells)
    assert len(h) == 12 and h == launch.grid_hash(list(reversed(cells)))
    assert h != launch.grid_hash(cells[:2]) and h != launch.grid_hash([*cells[:2], {"n": 20, "seed": 2}])
    cfg = _config(tmp_path)
    assert launch.main(["plan", "--config", str(cfg), "--cores", "0-31"]) == 0
    expected = launch.grid_hash(run.grid(run.effective_config(run.load_config(cfg))))
    assert f"grid {expected}" in capsys.readouterr().out
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(cfg), "--out-dir", str(out_dir), "--cores", "0-3",
                        "--poll", "0.05"]) == 0
    assert json.loads((out_dir / "launch_manifest.json").read_text())["grid_hash"] == expected
    assert json.loads((out_dir / "launch_state.json").read_text())["grid_hash"] == expected


def test_a_directory_cannot_be_resumed_with_another_grid(tmp_path):
    """The grid depends on the input reports (n rule, tuned configurations), which the config hash does not
    cover: if they changed, cells would move between shards and be computed twice."""
    cfg = str(_config(tmp_path))
    out_dir = tmp_path / "out"
    args = ["run", "--config", cfg, "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]
    assert launch.main(args) == 0
    state_path = out_dir / "launch_state.json"
    state = json.loads(state_path.read_text())
    state["grid_hash"] = "deadbeef0000"
    state_path.write_text(json.dumps(state))
    with pytest.raises(launch.LaunchError, match="grid"):
        launch.main(args)


def test_there_are_no_more_shards_than_cells(tmp_path, capsys):
    cfg = _config(tmp_path, launch={"shards": 100})
    assert launch.main(["plan", "--config", str(cfg), "--cores", "0-31"]) == 0
    assert "6 cells, 6 shards" in capsys.readouterr().out
    assert launch.main(["plan", "--config", str(cfg), "--cores", "0-31", "--shards", "50"]) == 0
    assert "6 cells, 6 shards" in capsys.readouterr().out


def test_the_plan_prints_the_shards_and_the_core_of_each_job(tmp_path, capsys):
    assert launch.main(["plan", "--config", str(_config(tmp_path)), "--cores", "0-31", "--pin"]) == 0
    out = capsys.readouterr().out
    assert "6 cells" in out and "3 shards" in out and "2 jobs" in out and "cores" in out


def test_one_launcher_at_a_time_on_a_machine(tmp_path):
    lock = tmp_path / "launch.lock"
    with (
        launch.exclusive_lock(lock),
        pytest.raises(launch.LaunchError, match="another launcher"),
        launch.exclusive_lock(lock),
    ):
        pass
    with launch.exclusive_lock(lock):  # released
        pass


def test_the_lock_outlives_the_launcher_while_one_of_its_jobs_runs(tmp_path):
    """If the launcher is killed, its jobs go on (they have their own session): a second launcher must not
    start the same shards next to them. The jobs inherit the lock."""
    lock = tmp_path / "launch.lock"
    with launch.exclusive_lock(lock) as fd:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"], pass_fds=(fd,))
    try:  # the launcher has left; the job still holds the lock
        with pytest.raises(launch.LaunchError, match="another launcher"), launch.exclusive_lock(lock):
            pass
    finally:
        child.terminate()
        child.wait()
    with launch.exclusive_lock(lock):  # the last holder is gone
        pass


def test_every_job_is_started_with_the_lock_and_a_hangup_stops_the_launcher_like_a_term(tmp_path, monkeypatch):
    import signal

    seen = {}
    real_popen = subprocess.Popen

    def spy(cmd, **kwargs):
        seen.setdefault("pass_fds", []).append(kwargs.get("pass_fds"))
        seen.setdefault("handlers", []).append((signal.getsignal(signal.SIGHUP), signal.getsignal(signal.SIGTERM)))
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spy)
    assert launch.main(["run", "--config", str(_config(tmp_path)), "--out-dir", str(tmp_path / "out"),
                        "--cores", "0-3", "--poll", "0.05"]) == 0
    assert seen["pass_fds"] and all(fds and len(fds) == 1 for fds in seen["pass_fds"])
    assert all(hup is term is launch._raise_interrupt for hup, term in seen["handlers"])
    assert signal.getsignal(signal.SIGHUP) is not launch._raise_interrupt  # restored afterwards


@LINUX_ONLY
def test_status_counts_the_rows_done_in_every_shard(tmp_path, capsys):
    cfg = _config(tmp_path)
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(cfg), "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]) == 0
    capsys.readouterr()
    assert launch.main(["status", "--config", str(cfg), "--out-dir", str(out_dir)]) == 0
    out = capsys.readouterr().out
    assert "6/6 cells done" in out and "3/3 shards complete" in out
    victim = out_dir / "toy_shard0002.jsonl"
    victim.write_text("")
    assert launch.main(["status", "--config", str(cfg), "--out-dir", str(out_dir)]) == 1  # unfinished work
    assert "4/6 cells done" in capsys.readouterr().out


def test_the_ingest_accepts_the_shards_of_the_launcher(tmp_path):
    from cif.experiments import ingest

    cfg = str(_config(tmp_path))
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", cfg, "--out-dir", str(out_dir), "--cores", "0-3", "--poll", "0.05"]) == 0
    shards = sorted(str(p) for p in out_dir.glob("*_shard*.jsonl"))
    runs = tmp_path / "runs"
    assert ingest.main(["--config", cfg, "--shards", *shards, "--runs-dir", str(runs),
                        "--logs", str(out_dir / "launch_manifest.json")]) == 0
    manifest = json.loads(next(runs.glob("*/manifest.json")).read_text())
    assert manifest["n_missing"] == 0 and manifest["n_duplicates"] == 0 and manifest["n_sessions"] == 3
    assert manifest["cpu_governors"] is not None and manifest["loadavg_start_max"] is not None
    run_dir = next(runs.glob("*/manifest.json")).parent
    assert (run_dir / "ingest" / "launch_manifest.json").exists()


def test_the_launcher_runs_a_real_e2_stage_as_a_rehearsal(tmp_path):
    """launcher → runner → ``e2_benchmark`` on the smoke grid of ``configs/E2/bench_factor.yaml``."""
    cfg = launch.registry.REPO_ROOT / "configs" / "E2" / "bench_factor.yaml"
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(cfg), "--smoke", "--out-dir", str(out_dir), "--jobs", "2",
                        "--shards", "4", "--cores", "0-3", "--poll", "0.05"]) == 0
    rows = _all_rows(out_dir)
    expected = run.grid(run.effective_config(run.load_config(cfg), smoke=True))
    assert len(rows) == len(expected) == 18 and not any(r.get("error") for r in rows)
    assert {r["cell_key"] for r in rows} == {run.cell_key(c) for c in expected}
    assert all(r["threads"] == 1 and r["stage"] == "bench" and r["budget_ok"] for r in rows)
    sessions = [s for p in out_dir.glob("*.meta.json") for s in json.loads(p.read_text())["sessions"]]
    assert len(sessions) == 4 and all(s["machine"]["governors"] is not None for s in sessions)
    assert all(s["config_hash"] == sessions[0]["config_hash"] for s in sessions)


def test_the_server_wrapper_starts_the_launcher_with_one_thread_per_job(tmp_path):
    script = launch.registry.REPO_ROOT / "scripts" / "run_server.sh"
    out_dir = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "PYTHONPATH", "CIF_BUNDLE_SHA")}
    env["PYTHON"] = sys.executable
    result = subprocess.run(
        ["bash", str(script), "configs/E2/ref_calibration.yaml", "--smoke", "--cores", "0-3", "--out-dir",
         str(out_dir), "--poll", "0.05"], capture_output=True, text=True, check=False, env=env,
        cwd=tmp_path)  # from another directory: the script finds the repository itself
    assert result.returncode == 0, result.stdout + result.stderr
    assert "host=" in result.stdout and "governor=" in result.stdout and "load=" in result.stdout
    rows = _all_rows(out_dir)
    assert len(rows) == 1 and rows[0]["stage"] == "reference" and rows[0]["certified"] is True
    assert rows[0]["threads"] == 1
    session = json.loads(next(out_dir.glob("*.meta.json")).read_text())["sessions"][0]
    pools = session["threads"]["pools"]
    assert all(p["num_threads"] == 1 for p in pools if p["user_api"] == "blas")
    assert session["threads"]["env"]["OMP_NUM_THREADS"] == "1"


def test_the_server_wrapper_keeps_the_user_site_out_of_the_environment(tmp_path):
    """Plan §12: the run environment is ``environment.lock.yml`` with ``PYTHONNOUSERSITE=1`` -- packages in somebody's
    home directory must not shadow it on a shared server."""
    script = launch.registry.REPO_ROOT / "scripts" / "run_server.sh"
    fake = tmp_path / "fake_python"
    fake.write_text('#!/bin/sh\necho "NOUSERSITE=$PYTHONNOUSERSITE"\n')
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONNOUSERSITE"}
    env["PYTHON"] = str(fake)
    result = subprocess.run(["bash", str(script), "configs/E2/ref_calibration.yaml"], capture_output=True, text=True,
                            check=False, env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOUSERSITE=1" in result.stdout


def test_load_at_the_start_is_reported_as_other_work_on_a_shared_machine(tmp_path, monkeypatch, capsys):
    """There is no exclusive mode on the run server (plan §12): the load is recorded, and said to be someone else's."""
    monkeypatch.setattr(launch.registry, "_loadavg", lambda: [3.5, 2.0, 1.0])
    out_dir = tmp_path / "out"
    assert launch.main(["run", "--config", str(_config(tmp_path)), "--out-dir", str(out_dir), "--cores", "0-3",
                        "--poll", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "load average at the start is 3.50" in out and "shared" in out and "not idle" not in out
    manifest = json.loads((out_dir / "launch_manifest.json").read_text())
    assert manifest["loadavg_start"] == [3.5, 2.0, 1.0] and manifest["loadavg_1m_max"] == 3.5


def test_the_launcher_module_runs_as_a_script(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "cif.experiments.launch", "plan", "--config", str(_config(tmp_path)),
         "--cores", "0-31"], capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(launch.registry.REPO_ROOT / "src")})
    assert result.returncode == 0, result.stderr
    assert "6 cells" in result.stdout
