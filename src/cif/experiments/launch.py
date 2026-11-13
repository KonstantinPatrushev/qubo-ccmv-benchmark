"""Launcher for a server without a job scheduler (ANALYSIS_PLAN §6; contract of ``cif.experiments.run``).

Runs the shards of one config as single-thread jobs, a fixed number at a time, on the run server. The server
is shared (plan §12, 2026-10-11): there is no exclusive mode, so the plan's limits stand in for it -- at most
half of the physical cores in jobs, jobs bound to physical cores, the load and the governor in the manifest --
and the launcher looks at the cores before it binds jobs to them::

    python -m cif.experiments.launch plan   --config configs/E2/bench_factor.yaml --cores 0-31 --pin
    python -m cif.experiments.launch run    --config configs/E2/bench_factor.yaml --cores 0-31 --pin
    python -m cif.experiments.launch status --config configs/E2/bench_factor.yaml

* **Shards and jobs.** The grid of the config is cut as the runner cuts it (``cells[i::S]``) into
  ``S`` shards (``--shards``, default ``launch.shards`` of the config, else ``min(cells, 8·jobs)``); at most
  ``--jobs`` (default ``launch.jobs``) processes ``python -m cif.experiments.run --shard i --n-shards S
  --workers 1 --out <dir>/<name>_shard<i>.jsonl`` run at the same time, so a slow shard does not hold
  the others back. One worker, one BLAS thread (``single_thread_env``) per job.
* **Timing stages** (``launch.timing`` true, the default) run at most half of the physical cores' worth of
  jobs (§6); ``--allow-oversubscribe`` lifts that. The number of physical cores comes from ``--cores`` (the
  list of physical cores, e.g. ``0-31`` on 2 × Xeon Silver 4216 with 64 threads) or from the CPU
  topology of the machine; if it is unknown the launcher refuses, it does not guess from the logical
  CPUs. The launcher does not hold non-timing stages (the certification runs) to that limit, but the configs
  of ``configs/E2`` ask for 16 jobs there as well: §12 caps our simultaneous jobs on the shared server.
* **Pinning** (``--pin``): job slot ``j`` is bound to one physical core by ``taskset -c`` (or
  ``sched_setaffinity`` without ``taskset``). The cores are spread over the sockets: a flat pool is
  spread evenly. Pinning is off by default. A ``--cores`` list that contains hyperthread siblings (as the
  topology of the machine reports them) draws a warning, and two pinned jobs on one physical core are refused.
* **Busy cores.** On a shared machine a job bound to a core somebody else is using gets a fraction of it and its
  timings mean nothing. Before pinned jobs start (``plan`` and ``run``) the launcher samples the CPU counters for
  a second; a core of the jobs whose threads (the core and its hyperthread siblings) were busy at least a quarter
  of that time draws a warning that names the cores to leave out with ``--exclude-cores`` (the list is taken out
  of the pool; "half of the physical cores" still counts the whole of ``--cores``). The warning is recorded in the
  manifest; whether to go on is the operator's call. Later interference shows in the rows
  (``method_cpu_s`` against ``method_wall_s``) and in the load samples.
* **Resume.** The runner skips the cells that have a row without an error in the shard file; the
  launcher starts only the shards that still have cells to do and refuses another ``--shards`` or
  another config in the same ``--out-dir`` (``launch_state.json``): a cell must stay in its shard, or
  a repeated cell would give a second, different timing row and :mod:`cif.experiments.ingest` would
  refuse the merge. A job killed in the middle loses at most the row it was writing.
* **Grid digest.** ``plan`` prints ``grid <digest>`` (a hash of the cell keys, :func:`grid_hash`); the manifest
  and the state of the output directory record it. The grid depends on the input reports and on the platform
  (instances, sampled configurations): compare the digest of the laptop with the server's before a long run, and
  a directory is not resumed with another one.
* **Manifest.** ``<out-dir>/launch_manifest.json``: host, CPU model, frequency governors of all CPUs, load
  average at the start, during the run (a sample every ``--load-interval`` s, with the number of our jobs
  running) and at the end, the cores of the jobs, the shards started, skipped and failed with their exit
  codes, the config hash and the bundle SHA, and the earlier invocations. Pass it to ``ingest --logs``.
  Each shard session also records the machine state (:func:`cif.experiments.registry.machine_state`),
  which ingest summarises in the manifest of the registered run.
* **One launcher at a time** on a machine (``flock`` on ``--lock-file``, default ``$CIF_LAUNCH_LOCK`` or
  ``/tmp/cif_launch.lock``): two stages running side by side would share the cores the plan wants kept free.
  The jobs inherit the lock, so it also holds while the jobs of a launcher that was killed (SIGKILL) are
  still running; wait for them, or kill them, before resuming.

Exit code: 0 if every shard finished without a failed cell, 1 if some did not (their rows are in the
files, the errored cells are tried again on the next run), 130 after SIGINT / SIGTERM / SIGHUP (a dropped
ssh connection stops the jobs like Ctrl-C; run the launcher in `tmux` / `screen` / `nohup` to avoid that).
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from cif.experiments import registry
from cif.experiments.run import cell_key, effective_config, grid, load_config, read_rows
from cif.experiments.timing import single_thread_env

__all__ = [
    "LaunchError",
    "build_command",
    "busy_cores",
    "busy_shares",
    "check_jobs",
    "choose_cores",
    "detect_topology",
    "exclusive_lock",
    "grid_hash",
    "hyperthread_siblings",
    "main",
    "max_timing_jobs",
    "parse_cores",
    "pin_function",
]

#: Jobs per shard-slot: the default number of shards is ``SHARDS_PER_JOB · jobs`` (at most the cells).
SHARDS_PER_JOB = 8
_ABORT_CODES = (2,)  # the runner's exit code for inputs that do not match their provenance


class LaunchError(RuntimeError):
    """The launcher refuses to start (or to resume): the message says why."""


# --------------------------------------------------------------------------------------------
# Cores
# --------------------------------------------------------------------------------------------


def parse_cores(spec: str) -> list[int]:
    """``"0-3,8,10-11"`` → ``[0, 1, 2, 3, 8, 10, 11]``."""
    out: list[int] = []
    try:
        for part in str(spec).split(","):
            lo, sep, hi = part.partition("-")
            first = int(lo.strip())
            last = int(hi.strip()) if sep else first
            if first < 0 or last < first:
                raise ValueError(part)
            out.extend(range(first, last + 1))
    except ValueError:
        raise ValueError(f"cannot read the list of cores {spec!r} (expected e.g. '0-31' or '0,2,4-7')") from None
    return out


#: Where the CPU topology is read from (tests point it at a fake tree).
_CPU_DIR = registry.SYS_CPU_DIR


def _cpu_map(cpu_dir: Path) -> dict[int, tuple[int, int]]:
    """``{logical CPU: (socket, core)}`` for the CPUs whose topology the system reports; ``{}`` if it does not."""
    out: dict[int, tuple[int, int]] = {}
    try:
        entries = [e for e in cpu_dir.iterdir() if e.name.startswith("cpu") and e.name[3:].isdigit()]
    except OSError:
        return {}
    for entry in entries:
        try:
            out[int(entry.name[3:])] = (int((entry / "topology" / "physical_package_id").read_text()),
                                        int((entry / "topology" / "core_id").read_text()))
        except (OSError, ValueError):
            continue
    return out


def detect_topology(cpu_dir: Path | None = None) -> dict[int, list[int]]:
    """``{socket: [physical cores]}`` from the CPU topology (``/sys/devices/system/cpu/cpu*/topology``):
    a physical core is named by its lowest logical CPU (its hyperthread siblings are left out).
    ``{}`` if the system does not say."""
    seen: dict[tuple[int, int], int] = {}
    for cpu, key in sorted(_cpu_map(_CPU_DIR if cpu_dir is None else cpu_dir).items()):
        seen.setdefault(key, cpu)  # sorted: the lowest logical CPU of the core
    out: dict[int, list[int]] = {}
    for (package, _), cpu in seen.items():
        out.setdefault(package, []).append(cpu)
    return {package: sorted(cpus) for package, cpus in sorted(out.items())}


def hyperthread_siblings(cores: Sequence[int], cpu_dir: Path | None = None) -> dict[int, int]:
    """``{cpu: first thread of its core}`` for the listed CPUs that are not the first thread of their
    physical core. A list of "physical cores" that contains such CPUs is wrong, and two jobs pinned to the
    two threads of one core would share it. CPUs the system does not report are not claimed to be siblings."""
    cpus = _cpu_map(_CPU_DIR if cpu_dir is None else cpu_dir)
    first: dict[tuple[int, int], int] = {}
    for cpu, key in sorted(cpus.items()):
        first.setdefault(key, cpu)
    return {c: first[cpus[c]] for c in cores if c in cpus and first[cpus[c]] != c}


def _shared_cores(cores: Sequence[int]) -> list[int]:
    """The CPUs among ``cores`` that sit on a physical core another one of them sits on (``[]`` if the
    topology is unknown)."""
    cpus = _cpu_map(_CPU_DIR)
    by_core: dict[tuple[int, int], list[int]] = {}
    for c in cores:
        if c in cpus:
            by_core.setdefault(cpus[c], []).append(c)
    return sorted(c for group in by_core.values() if len(group) > 1 for c in group)


#: Per-CPU time counters of the system (tests point it at a fake file).
_PROC_STAT = Path("/proc/stat")
#: How long the cores are watched before pinned jobs start, and the share of that time from which a thread is busy.
PROBE_SECONDS = 1.0
BUSY_SHARE = 0.25


def _cpu_counters(stat: Path) -> dict[int, tuple[int, int]]:
    """``{logical CPU: (busy ticks, all ticks)}`` from the per-CPU lines of ``/proc/stat`` (all: the first eight
    counters; busy: all but idle and iowait); ``{}`` if the file cannot be read."""
    try:
        lines = stat.read_text().splitlines()
    except OSError:
        return {}
    out: dict[int, tuple[int, int]] = {}
    for line in lines:
        head, _, rest = line.partition(" ")
        if not (head.startswith("cpu") and head[3:].isdigit()):
            continue  # the aggregate line and the other counters
        try:
            ticks = [int(x) for x in rest.split()[:8]]
        except ValueError:
            continue
        if len(ticks) >= 5:
            total = sum(ticks)
            out[int(head[3:])] = (total - ticks[3] - ticks[4], total)
    return out


def busy_shares(first: Mapping[int, tuple[int, int]], second: Mapping[int, tuple[int, int]]) -> dict[int, float]:
    """``{CPU: share of the time it was busy}`` between two readings of :func:`_cpu_counters`; a CPU that is in
    only one of them, or whose clock did not move, is left out."""
    return {cpu: (second[cpu][0] - busy) / (second[cpu][1] - total)
            for cpu, (busy, total) in first.items() if cpu in second and second[cpu][1] > total}


def _probe_busy(seconds: float = PROBE_SECONDS) -> dict[int, float]:
    """The busy share of every CPU over ``seconds`` (``{}`` if the system does not say)."""
    first = _cpu_counters(_PROC_STAT)
    time.sleep(seconds)
    return busy_shares(first, _cpu_counters(_PROC_STAT))


def busy_cores(cores: Sequence[int], shares: Mapping[int, float], cpu_dir: Path | None = None,
               threshold: float = BUSY_SHARE) -> dict[int, dict[int, float]]:
    """``{core: {CPU: share}}`` for the listed cores that were busy. The threads of a core are the CPU itself and
    its hyperthread siblings (as the topology says; without one, the CPU alone); a thread is busy at ``threshold``
    of the time or more."""
    cpus = _cpu_map(_CPU_DIR if cpu_dir is None else cpu_dir)
    out: dict[int, dict[int, float]] = {}
    for core in cores:
        threads = [c for c, key in sorted(cpus.items()) if key == cpus[core]] if core in cpus else [core]
        busy = {t: shares[t] for t in threads if shares.get(t, 0.0) >= threshold}
        if busy:
            out[core] = busy
    return out


def choose_cores(packages: Mapping[object, Sequence[int]], jobs: int) -> list[int]:
    """The cores of ``jobs`` job slots, spread over the sockets: the lowest core of each socket in turn,
    then the next ones. A single flat pool (key ``None``) is spread evenly over its length."""
    total = sum(len(v) for v in packages.values())
    if jobs > total:
        raise LaunchError(f"{jobs} jobs need {jobs} cores, the pool has {total}")
    if len(packages) == 1:
        (pool,) = packages.values()
        return [pool[i * len(pool) // jobs] for i in range(jobs)] if None in packages else list(pool[:jobs])
    ordered = [list(packages[k]) for k in sorted(packages)]
    out: list[int] = []
    depth = 0
    while len(out) < jobs:
        for pool in ordered:
            if depth < len(pool) and len(out) < jobs:
                out.append(pool[depth])
        depth += 1
    return out


def max_timing_jobs(n_physical: int) -> int:
    """At most this many single-thread jobs may run in a timing stage: half of the physical cores (§6)."""
    return max(1, n_physical // 2)


def check_jobs(jobs: int, *, timing: bool, n_physical: int | None, allow_oversubscribe: bool = False,
               pool_size: int | None = None) -> None:
    """Refuse a number of jobs the plan does not allow (module docstring). ``n_physical`` is the machine's,
    which the half-of-the-cores rule counts; ``pool_size`` the cores the jobs are bound to, if fewer."""
    if jobs < 1:
        raise LaunchError(f"jobs must be at least 1, got {jobs}")
    if allow_oversubscribe:
        return
    if n_physical is None:
        if timing:
            raise LaunchError(
                "the number of physical cores is unknown (no --cores and no CPU topology): a timing stage may use "
                "half of them, and the logical CPUs of a hyperthreaded machine are twice as many; "
                "pass --cores (e.g. 0-31) or --allow-oversubscribe")
        return
    if timing and jobs > max_timing_jobs(n_physical):
        raise LaunchError(
            f"{jobs} jobs exceed half of the {n_physical} physical cores ({max_timing_jobs(n_physical)}): "
            "a timing stage must leave the other half idle (plan §6); use --jobs or --allow-oversubscribe")
    if pool_size is not None and jobs > pool_size:
        raise LaunchError(f"{jobs} jobs on the {pool_size} cores left to choose from: not enough cores")
    if jobs > n_physical:
        raise LaunchError(f"{jobs} jobs on {n_physical} physical cores: not enough cores")


def pin_function(core: int) -> Callable[[], None]:
    """A ``preexec_fn`` that binds the child to ``core`` (used when ``taskset`` is not installed)."""
    def pin() -> None:
        os.sched_setaffinity(0, {core})

    return pin


# --------------------------------------------------------------------------------------------
# A job and the lock
# --------------------------------------------------------------------------------------------


def _bundle_sha() -> str | None:
    if os.environ.get("CIF_BUNDLE_SHA"):
        return os.environ["CIF_BUNDLE_SHA"]
    path = registry.REPO_ROOT / "BUNDLE_SHA"
    return path.read_text().strip() if path.exists() else None


def build_command(
    config: Path, shard: int, n_shards: int, out: Path, core: int | None = None, smoke: bool = False
) -> tuple[list[str], dict[str, str]]:
    """Command line and environment of the job for one shard: one worker, one thread; behind
    ``taskset -c <core>`` if ``core`` is given and ``taskset`` exists. ``smoke`` is a rehearsal on the
    smoke grid of the config."""
    cmd = [sys.executable, "-u", "-m", "cif.experiments.run", "--config", str(config), "--shard", str(shard),
           "--n-shards", str(n_shards), "--workers", "1", "--out", str(out)]
    if smoke:
        cmd.append("--smoke")
    if core is not None and (taskset := shutil.which("taskset")):
        cmd = [taskset, "-c", str(core), *cmd]
    env = {**os.environ, **single_thread_env()}
    src = str(registry.REPO_ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join([src, *filter(None, [env.get("PYTHONPATH")])])
    if (sha := _bundle_sha()):
        env["CIF_BUNDLE_SHA"] = sha
    return cmd, env


@contextlib.contextmanager
def exclusive_lock(path: Path):
    """Hold an exclusive ``flock`` on ``path`` for the duration and yield its descriptor; a second holder
    gets a :class:`LaunchError` at once (one launcher at a time on a machine). The jobs inherit the
    descriptor (``pass_fds``), so the lock outlives a launcher that was killed while its jobs run: the
    jobs are in their own sessions and go on, and a new launcher must not start the same shards beside them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LaunchError(f"another launcher holds {path}: one launcher at a time on this machine") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield fd
    finally:
        os.close(fd)  # the lock is released when the last descriptor of it (ours, the jobs') is closed


# --------------------------------------------------------------------------------------------
# The plan: shards, done cells
# --------------------------------------------------------------------------------------------


def grid_hash(cells: Sequence[dict]) -> str:
    """A short digest of the cell keys of a grid (independent of their order). The grid depends on the
    input reports and on the platform (instances, sampled configurations): compare the digest printed by
    ``plan`` on the laptop with the one on the server before a long run."""
    keys = "\n".join(sorted(cell_key(c) for c in cells))
    return hashlib.sha256(keys.encode("utf-8")).hexdigest()[:12]


def _shard_path(out_dir: Path, name: str, shard: int) -> Path:
    return out_dir / f"{name}_shard{shard:04d}.jsonl"


def _done_keys(path: Path) -> set[str]:
    return {r["cell_key"] for r in read_rows(path) if not r.get("error")}


def shard_status(cells: Sequence[dict], n_shards: int, out_dir: Path, name: str) -> list[dict]:
    """Per shard: its number of cells and how many have a successful row in its file."""
    out = []
    for shard in range(n_shards):
        keys = {cell_key(c) for c in cells[shard::n_shards]}
        out.append({"shard": shard, "n_cells": len(keys),
                    "n_done": len(keys & _done_keys(_shard_path(out_dir, name, shard)))})
    return out


class _Plan:
    """What a ``run`` / ``plan`` / ``status`` invocation works out from the config and the arguments."""

    def __init__(self, args: argparse.Namespace, *, need_jobs: bool = True) -> None:
        self.config = Path(args.config).resolve()
        self.smoke = bool(getattr(args, "smoke", False))
        self.cfg = effective_config(load_config(self.config), smoke=self.smoke)
        self.name = self.cfg["name"]
        self.cells = grid(self.cfg)
        self.grid_hash = grid_hash(self.cells)
        self.config_hash = registry.config_hash(self.cfg)
        launch_cfg = self.cfg.get("launch") or {}
        self.timing = bool(launch_cfg.get("timing", True))
        self.jobs = getattr(args, "jobs", None) or launch_cfg.get("jobs")
        if need_jobs and not self.jobs:
            raise LaunchError("the number of jobs is not set: pass --jobs or put launch.jobs in the config")
        wanted = int(getattr(args, "shards", None) or launch_cfg.get("shards") or SHARDS_PER_JOB * (self.jobs or 1))
        self.n_shards = max(1, min(wanted, len(self.cells)))  # no empty shards
        self.out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else (
            registry.REPO_ROOT / "runs" / "incoming" / f"{self.cfg['stream']}-{self.name}")
        self.pool: dict[object, list[int]] = {}
        self.n_physical: int | None = None
        self.cores: list[int] = []
        self.pin = bool(getattr(args, "pin", False))
        self.warnings: list[str] = []
        cores = getattr(args, "cores", None)
        if cores:
            requested = parse_cores(cores)
            self.pool = {None: requested}
            siblings = hyperthread_siblings(requested)
            if siblings:
                self.warnings.append(
                    f"--cores lists CPUs that the system reports as hyperthread siblings of other cores "
                    f"(CPU: first thread of its core) {siblings}: the physical cores are the first threads; "
                    "two jobs on the two threads of one core would share it")
            detected = detect_topology()
            known = {c for v in detected.values() for c in v}
            if detected and set(self.pool[None]) <= known:
                self.pool = {p: [c for c in v if c in set(self.pool[None])] for p, v in detected.items()}
                self.pool = {p: v for p, v in self.pool.items() if v}
        else:
            self.pool = detect_topology()
        self.n_physical = sum(len(v) for v in self.pool.values()) or None  # of the machine: the half rule counts these
        self.excluded: list[int] = []
        if (leave_out := getattr(args, "exclude_cores", None)):
            left_out = sorted(set(parse_cores(leave_out)))
            if not self.pin:
                raise LaunchError("--exclude-cores applies to jobs bound to cores: add --pin (unpinned jobs go "
                                  "wherever the system puts them)")
            if unknown := sorted(set(left_out) - {c for v in self.pool.values() for c in v}):
                raise LaunchError(f"--exclude-cores lists CPUs that are not among the cores to choose from: {unknown}")
            self.pool = {p: [c for c in v if c not in left_out] for p, v in self.pool.items()}
            self.pool = {p: v for p, v in self.pool.items() if v}
            self.excluded = left_out
        self.n_pool = sum(len(v) for v in self.pool.values())
        if self.jobs:
            check_jobs(int(self.jobs), timing=self.timing, n_physical=self.n_physical,
                       allow_oversubscribe=bool(getattr(args, "allow_oversubscribe", False)),
                       pool_size=self.n_pool if self.excluded else None)
            if self.pin:
                if not self.pool:
                    raise LaunchError("--pin needs the cores to pin to: pass --cores or run on a machine "
                                      "that reports its CPU topology")
                self.cores = choose_cores(self.pool, int(self.jobs))
                if shared := _shared_cores(self.cores):
                    raise LaunchError(
                        f"jobs pinned to CPUs {shared} would share a physical core (hyperthread siblings): "
                        "list the physical cores in --cores (the first thread of each core)")


def _read_state(out_dir: Path) -> dict | None:
    path = out_dir / "launch_state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _check_state(plan: _Plan) -> None:
    state = _read_state(plan.out_dir)
    if state is None:
        return
    if state["config_hash"] != plan.config_hash:
        raise LaunchError(
            f"{plan.out_dir} holds shards of another config (hash {state['config_hash']}, this config "
            f"{plan.config_hash}): use another --out-dir")
    if state.get("grid_hash") not in (None, plan.grid_hash):
        raise LaunchError(
            f"{plan.out_dir} was started with another grid of cells (digest {state['grid_hash']}, now "
            f"{plan.grid_hash}) although the config is the same: an input report changed? Cells would move "
            "between shards and be computed twice; use another --out-dir")
    if state["n_shards"] != plan.n_shards:
        raise LaunchError(
            f"{plan.out_dir} was started with {state['n_shards']} shards, not {plan.n_shards}: a cell must stay "
            "in its shard (resume with the same --shards)")


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------


def _describe(plan: _Plan) -> str:
    cfg = plan.cfg
    lines = [f"{cfg['stream']}/{plan.name}{' (smoke rehearsal)' if plan.smoke else ''}: {len(plan.cells)} cells, "
             f"{plan.n_shards} shards, {plan.jobs} jobs ({'timing' if plan.timing else 'not a timing'} stage); "
             f"output {plan.out_dir}", f"config {plan.config_hash}, grid {plan.grid_hash}"]
    if plan.n_physical:
        lines.append(f"physical cores: {plan.n_physical}; timing stages use at most {max_timing_jobs(plan.n_physical)}")
    if plan.pin:
        lines.append("job slot -> core: " + ", ".join(f"{j}->{c}" for j, c in enumerate(plan.cores)))
        lines.append(f"cores of the jobs: {plan.cores}")
        if plan.excluded:
            lines.append(f"cores left out: {plan.excluded}")
    lines += [f"WARNING: {w}" for w in plan.warnings]
    return "\n".join(lines)


def _preflight(plan: _Plan) -> None:
    """Look at the cores that pinned jobs are about to be bound to (module docstring, "Busy cores"); the
    warning goes to ``plan.warnings``, which ``plan`` prints and ``run`` prints and records."""
    if not (plan.pin and plan.cores):
        return
    busy = busy_cores(plan.cores, _probe_busy())
    if busy:
        seen = "; ".join(f"core {core}: " + ", ".join(f"CPU {t} {share:.0%}" for t, share in threads.items())
                         for core, threads in busy.items())
        plan.warnings.append(
            f"cores the jobs are bound to were busy before the start ({seen}; share of the last {PROBE_SECONDS:g} s). "
            "The server is shared (plan §12): leave them out with "
            f"--exclude-cores {','.join(str(c) for c in sorted(busy))}, or expect disturbed timings")


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = _Plan(args)
    _preflight(plan)
    print(_describe(plan))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    plan = _Plan(args, need_jobs=False)
    state = _read_state(plan.out_dir)
    n_shards = state["n_shards"] if state else plan.n_shards
    status = shard_status(plan.cells, n_shards, plan.out_dir, plan.name)
    done = sum(s["n_done"] for s in status)
    complete = sum(s["n_done"] == s["n_cells"] for s in status)
    for s in status:
        if s["n_done"] < s["n_cells"]:
            print(f"shard {s['shard']}: {s['n_done']}/{s['n_cells']}")
    print(f"{done}/{len(plan.cells)} cells done, {complete}/{n_shards} shards complete ({plan.out_dir})")
    return 0 if done == len(plan.cells) else 1


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def _sample(running: int) -> dict:
    return {"t": time.time(), "loadavg": registry._loadavg(), "jobs_running": running}


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")


def _cmd_run(args: argparse.Namespace) -> int:
    plan = _Plan(args)
    lock = Path(args.lock_file or os.environ.get("CIF_LAUNCH_LOCK") or Path(tempfile.gettempdir()) / "cif_launch.lock")
    with exclusive_lock(lock) as lock_fd:
        return _run_locked(plan, args, lock_fd)


def _check_confirmatory(plan: _Plan) -> None:
    """A stage on test seeds (the confirmatory runs of the plan) must know the code it runs: its rows are
    attributed to ``BUNDLE_SHA`` (env ``CIF_BUNDLE_SHA`` or the file in the repository root)."""
    if plan.cfg.get("params", {}).get("seed_role") == "test" and not plan.smoke and not _bundle_sha():
        raise LaunchError(
            "this stage reads test seeds, so its rows must be attributable to a code bundle: there is no "
            "BUNDLE_SHA (file in the repository root, or CIF_BUNDLE_SHA); build the bundle from the clean "
            "committed tree (.claude/skills/hpc-job)")


def _run_locked(plan: _Plan, args: argparse.Namespace, lock_fd: int) -> int:
    out_dir = plan.out_dir
    _check_confirmatory(plan)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    _check_state(plan)
    state_path = out_dir / "launch_state.json"
    if not state_path.exists():
        state_path.write_text(json.dumps({
            "config_hash": plan.config_hash, "grid_hash": plan.grid_hash, "n_cells": len(plan.cells),
            "n_shards": plan.n_shards, "created_at": registry._now()}, indent=1) + "\n", encoding="utf-8")
    jobs = int(plan.jobs)
    if (sha := _bundle_sha()):  # the file ``ingest --bundle-sha-file`` expects next to the shards
        (out_dir / "BUNDLE_SHA").write_text(sha + "\n", encoding="utf-8")
    status = shard_status(plan.cells, plan.n_shards, out_dir, plan.name)
    pending = [s["shard"] for s in status if s["n_done"] < s["n_cells"]]
    skipped = [s["shard"] for s in status if s["n_done"] == s["n_cells"]]

    manifest_path = out_dir / "launch_manifest.json"
    previous: list[dict] = []
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous = [*old.pop("previous_invocations", []),
                    {k: old.get(k) for k in ("started_at", "finished_at", "status", "jobs", "cores", "shards_started",
                                              "shards_failed", "loadavg_start", "loadavg_end", "loadavg_1m_max")}]
    _preflight(plan)
    started_wall = time.time()
    first = _sample(0)
    manifest: dict = {
        "status": "running", "stream": plan.cfg["stream"], "name": plan.name, "config": str(plan.config),
        "config_hash": plan.config_hash, "grid_hash": plan.grid_hash, "bundle_sha": _bundle_sha(),
        "host": os.uname().nodename,
        "started_at": registry._now(), "finished_at": None, "n_cells": len(plan.cells),
        "n_shards": plan.n_shards, "jobs": jobs, "timing": plan.timing, "pinned": plan.pin, "smoke": plan.smoke,
        "cores": plan.cores, "excluded_cores": plan.excluded, "machine": registry.machine_state(), "argv": sys.argv,
        "warnings": list(plan.warnings), "loadavg_start": first["loadavg"], "loadavg_end": None, "loadavg_1m_max": (first["loadavg"] or [0.0])[0],
        "loadavg_samples": [first], "shards_started": [], "shards_skipped": skipped, "shards_failed": [],
        "shards": [], "previous_invocations": previous,
    }
    if plan.timing and first["loadavg"] and first["loadavg"][0] > 1.0:
        print(f"WARNING: the 1-minute load average at the start is {first['loadavg'][0]:.2f}: other work is running "
              "on this machine (the server is shared, plan §12); the load is sampled into the manifest every "
              f"{float(args.load_interval):g} s")
    _write_manifest(manifest_path, manifest)
    print(_describe(plan))
    print(f"{len(pending)} shards to run, {len(skipped)} complete")

    old_handlers = {sig: signal.signal(sig, _raise_interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)}
    running: dict[int, dict] = {}
    free = list(range(jobs))
    queue = list(pending)
    entries: dict[int, dict] = {s: {"shard": s, "status": "skipped"} for s in skipped}
    next_sample = time.time() + float(args.load_interval)
    interrupted = aborted = False
    try:
        while queue or running:
            while queue and free and not aborted:
                slot = free.pop(0)
                shard = queue.pop(0)
                core = plan.cores[slot] if plan.pin else None
                out = _shard_path(out_dir, plan.name, shard)
                cmd, env = build_command(plan.config, shard, plan.n_shards, out, core, plan.smoke)
                log = open(out_dir / "logs" / f"{plan.name}_shard{shard:04d}.log", "ab")  # noqa: SIM115
                preexec = pin_function(core) if core is not None and "taskset" not in Path(cmd[0]).name else None
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=preexec,
                                        start_new_session=True, pass_fds=(lock_fd,))
                running[shard] = {"proc": proc, "slot": slot, "core": core, "log": log, "t0": time.time()}
                manifest["shards_started"].append(shard)
            if aborted and not running:
                break
            time.sleep(float(args.poll))
            for shard, job in list(running.items()):
                code = job["proc"].poll()
                if code is None:
                    continue
                job["log"].close()
                del running[shard]
                free.append(job["slot"])
                free.sort()
                keys = {cell_key(c) for c in plan.cells[shard::plan.n_shards]}
                done = len(keys & _done_keys(_shard_path(out_dir, plan.name, shard)))
                entries[shard] = {"shard": shard, "status": "finished" if code == 0 else "failed", "exit_code": code,
                                  "core": job["core"], "wall_s": time.time() - job["t0"], "n_cells": len(keys),
                                  "n_done": done}
                if code != 0:
                    manifest["shards_failed"].append(shard)
                if code in _ABORT_CODES:
                    aborted = True
                    print(f"ERROR: shard {shard} exited with code {code} (inputs do not match their provenance); "
                          "no further shards are started")
            if time.time() >= next_sample:
                sample = _sample(len(running))
                manifest["loadavg_samples"].append(sample)
                if sample["loadavg"]:
                    manifest["loadavg_1m_max"] = max(manifest["loadavg_1m_max"], sample["loadavg"][0])
                next_sample = time.time() + float(args.load_interval)
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted: stopping the jobs (the rows written so far are kept; run again to resume)")
        for job in running.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(job["proc"].pid, signal.SIGTERM)
        for job in running.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                job["proc"].wait(timeout=10)
            job["log"].close()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        end = _sample(0)
        manifest.update(
            status="interrupted" if interrupted else "finished", finished_at=registry._now(),
            elapsed_s=time.time() - started_wall, loadavg_end=end["loadavg"],
            shards=[entries[k] for k in sorted(entries)], shards_failed=sorted(manifest["shards_failed"]))
        manifest["loadavg_samples"].append(end)
        if end["loadavg"]:
            manifest["loadavg_1m_max"] = max(manifest["loadavg_1m_max"], end["loadavg"][0])
        _write_manifest(manifest_path, manifest)
    if interrupted:
        return 130
    unfinished = len(plan.cells) - sum(s["n_done"] for s in shard_status(plan.cells, plan.n_shards, out_dir, plan.name))
    print(f"finished: {len(manifest['shards_started'])} shards run, {len(manifest['shards_failed'])} failed, "
          f"{unfinished} cells not done; manifest {manifest_path}")
    return 1 if manifest["shards_failed"] or aborted else 0


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.launch", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in (("run", "run (or resume) the shards"), ("plan", "print what would run"),
                           ("status", "count the cells done per shard")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, default=None,
                       help="shard files, logs and manifest (default runs/incoming/<stream>-<name>)")
        p.add_argument("--smoke", action="store_true",
                       help="a rehearsal on the smoke grid of the config (calibration seeds, tiny budgets); "
                            "its shards cannot be ingested")
        if name != "status":
            p.add_argument("--jobs", type=int, default=None, help="concurrent single-thread jobs (default launch.jobs)")
            p.add_argument("--shards", type=int, default=None, help="number of shards (default launch.shards)")
            p.add_argument("--cores", default=None, metavar="0-31",
                           help="the physical cores of the machine (the pool for --pin)")
            p.add_argument("--pin", action="store_true", help="bind each job slot to one physical core")
            p.add_argument("--exclude-cores", default=None, metavar="3,17",
                           help="with --pin: cores of --cores not to use (busy with other users' work)")
            p.add_argument("--allow-oversubscribe", action="store_true",
                           help="allow more jobs than half of the physical cores in a timing stage")
        else:
            p.add_argument("--shards", type=int, default=None)
        if name == "run":
            p.add_argument("--load-interval", type=float, default=60.0, help="seconds between load samples")
            p.add_argument("--poll", type=float, default=1.0, help="seconds between checks of the jobs")
            p.add_argument("--lock-file", default=None, help="one launcher at a time (default $CIF_LAUNCH_LOCK)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return {"run": _cmd_run, "plan": _cmd_plan, "status": _cmd_status}[args.command](args)


def _cli() -> int:
    try:
        return main()
    except LaunchError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(_cli())
