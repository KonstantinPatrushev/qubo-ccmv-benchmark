"""Run registry: every result that may reach the paper is produced inside a registered run (R8).

``start_run`` creates ``runs/<run_id>/`` with a ``manifest.json`` (git SHA and dirty flag, input
data hashes, config hash, host, CPU and thread settings, package versions, seeds, timings,
status) and a ``results.jsonl`` that rows are appended to. On exit the manifest is finalised and
one line is appended to ``runs/registry.jsonl``. Failed runs are registered as ``failed`` with the
error message.

Usage::

    with start_run("E2", "ttt", config, seeds=seeds) as run:
        run.log_row({...})
        run.save_frame(df, "summary.csv")
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from cif.experiments.timing import thread_state

# CIF_REPO_ROOT lets a code snapshot outside the repo (e.g. a legacy/hybrid package used for
# delta reports) still register runs and record the repository's git state.
REPO_ROOT = Path(os.environ.get("CIF_REPO_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"
TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "cvxpy", "clarabel", "pyscipopt", "highspy", "dwave-samplers",
    "dimod", "qiskit", "qiskit-aer", "scikit-learn", "statsmodels", "yfinance", "apimoex",
)


def config_hash(config: dict) -> str:
    """Stable short hash of a JSON-serialisable config (key order does not matter)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        return ""


#: Paths whose uncommitted state changes what a run computes (audit R-1). Edits elsewhere
#: (docs/, runs/registry.jsonl, ...) are recorded in ``git_status`` but do not make a run dirty.
CODE_PATHS = ("src/", "scripts/", "configs/", "tests/", "tools/", "pyproject.toml", "environment.yml")


def _git_state() -> dict:
    """Provenance of the code: SHA, porcelain status, dirty flag, untracked code files.

    ``dirty`` is ``True`` if tracked code under :data:`CODE_PATHS` differs from HEAD or
    untracked files exist there, and ``None`` (unknown, never "clean") outside a git checkout.
    For code snapshots outside git, ``CIF_CODE_SHA`` may name the commit the snapshot came from.
    """
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return {"available": False, "sha": os.environ.get("CIF_CODE_SHA", ""), "branch": "",
                "dirty": None, "status": [], "untracked_code": []}
    # not _git(): its strip() would eat the leading space of the first porcelain line (" M path")
    raw = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=all"],
                         capture_output=True, text=True, check=False).stdout
    status = [line for line in raw.splitlines() if line.strip()]
    def path(line: str) -> str:
        return line[3:].split(" -> ")[-1].strip('"')
    tracked_code = [ln for ln in status if not ln.startswith("??") and path(ln).startswith(CODE_PATHS)]
    untracked_code = [path(ln) for ln in status if ln.startswith("??") and path(ln).startswith(CODE_PATHS)]
    return {"available": True, "sha": sha, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(tracked_code or untracked_code), "status": status,
            "untracked_code": untracked_code}


def _save_code_state(run_dir: Path, state: dict) -> list[str]:
    """Store the uncommitted diff and copies of untracked code files in the run directory."""
    saved: list[str] = []
    if not state["available"] or not state["dirty"]:
        return saved
    diff = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "HEAD"], capture_output=True,
                          text=True, check=False).stdout
    if diff:
        (run_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
        saved.append("git_diff.patch")
    for rel in state["untracked_code"]:
        src = REPO_ROOT / rel
        if src.is_file():
            # The copy stays local (runs/*/untracked_code/ is gitignored: it may hold unrelated,
            # deliberately uncommitted work); its SHA-256 goes into the committed manifest.
            dst = run_dir / "untracked_code" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = src.read_bytes()
            dst.write_bytes(data)
            state.setdefault("untracked_code_sha256", {})[rel] = hashlib.sha256(data).hexdigest()
            saved.append(f"untracked_code/{rel}")
    return saved


def _packages() -> dict[str, str]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _cpu_governor() -> str | None:
    """CPU frequency governor of core 0 (Linux), e.g. ``powersave``; None if unavailable."""
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return None


SYS_CPU_DIR = Path("/sys/devices/system/cpu")


def _governors(cpu_dir: Path = SYS_CPU_DIR) -> dict[str, int]:
    """Frequency governors of all CPUs with cpufreq, ``{governor: number of CPUs}`` (``{}`` if the
    system exposes none). A shared server may have ``schedutil`` on every CPU: ``cpu_governor``
    alone (core 0) would not show a mixed configuration."""
    out: dict[str, int] = {}
    try:
        for entry in cpu_dir.iterdir():
            if not (entry.name.startswith("cpu") and entry.name[3:].isdigit()):
                continue
            try:
                governor = (entry / "cpufreq" / "scaling_governor").read_text().strip()
            except OSError:
                continue
            out[governor] = out.get(governor, 0) + 1
    except OSError:
        return {}
    return out


def _loadavg() -> list[float] | None:
    try:
        return [float(x) for x in os.getloadavg()]
    except OSError:
        return None


def machine_state() -> dict:
    """The machine the timings are taken on (R10, ANALYSIS_PLAN §6): CPU model, number of logical
    CPUs, the CPUs this process may run on (a pinned job shows one), the frequency governors of all
    CPUs and the 1/5/15-minute load average. Written to the manifest of a run and to every shard
    session, so that a timing can be read against the load it was taken under."""
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "cpu": _cpu_model(),
        "n_cpus": os.cpu_count(),
        "affinity": affinity,
        "governors": _governors(),
        "loadavg": _loadavg(),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_inputs(inputs: dict[str, Path | str]) -> dict[str, dict]:
    """SHA-256 of each input file (data lineage); a missing input is an error before the run."""
    out = {}
    for label, path in inputs.items():
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"run input {label!r} not found: {p}")
        try:
            shown = p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            shown = str(p)
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[label] = {"path": shown, "sha256": h.hexdigest(), "bytes": p.stat().st_size}
    return out


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    manifest: dict
    n_rows: int = 0
    _results: Path = field(init=False)

    def __post_init__(self) -> None:
        self._results = self.run_dir / "results.jsonl"

    def log_row(self, row: dict) -> None:
        with self._results.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"run_id": self.run_id, **row}, default=str) + "\n")
        self.n_rows += 1

    def save_frame(self, df: pd.DataFrame, name: str) -> Path:
        path = self.run_dir / name
        if name.endswith(".parquet"):
            df.to_parquet(path)
        else:
            df.to_csv(path, index=True)
        self.manifest.setdefault("artifacts", []).append(name)
        return path

    def save_json(self, payload: dict, name: str) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        self.manifest.setdefault("artifacts", []).append(name)
        return path


@contextmanager
def start_run(
    stream: str,
    name: str,
    config: dict,
    *,
    runs_dir: Path | str | None = None,
    seeds: list[int] | None = None,
    notes: str = "",
    require_clean: bool = False,
    inputs: dict[str, Path | str] | None = None,
) -> Iterator[RunContext]:
    """Create and register a run; see module docstring.

    ``inputs`` maps a label to an input file, usually a dataset's ``provenance.json`` (whose
    recorded file hashes ``python -m cif.data.provenance verify`` checks); its SHA-256 goes into
    the manifest so every result traces back to the data version it was computed on.
    """
    runs_dir = Path(runs_dir) if runs_dir is not None else DEFAULT_RUNS_DIR
    input_hashes = _hash_inputs(inputs or {})
    chash = config_hash(config)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{stream}-{name}-{chash[:6]}"
    run_dir = runs_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{stamp}-{stream}-{name}-{chash[:6]}-{suffix}"
        run_dir = runs_dir / run_id
        suffix += 1
    if require_clean:
        pre = _git_state()
        if pre["dirty"] is not False:
            raise RuntimeError(
                "require_clean=True but the code state is "
                + ("unknown (no git checkout)" if pre["dirty"] is None else "dirty: " + "; ".join(
                    ln for ln in pre["status"] if ln[3:].startswith(CODE_PATHS)))
            )
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True, default=str))
    state = _git_state()
    code_state_files = _save_code_state(run_dir, state)
    manifest = {
        "run_id": run_id,
        "stream": stream,
        "name": name,
        "config_hash": chash,
        "git_sha": state["sha"],
        "git_branch": state["branch"],
        "git_available": state["available"],
        "git_dirty": state["dirty"],
        "git_status": state["status"],
        "untracked_code": state["untracked_code"],
        "untracked_code_sha256": state.get("untracked_code_sha256", {}),
        "code_state_files": code_state_files,
        "code_parts": os.environ.get("CIF_CODE_PARTS", ""),
        "host": socket.gethostname(),
        "cpu": _cpu_model(),
        "threads": thread_state(),
        "cpu_governor": _cpu_governor(),
        "machine": machine_state(),
        "python": platform.python_version(),
        "packages": _packages(),
        "seeds": seeds or [],
        "inputs": input_hashes,
        "notes": notes,
        "started_at": _now(),
        "status": "running",
    }
    ctx = RunContext(run_id=run_id, run_dir=run_dir, manifest=manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    try:
        yield ctx
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = _now()
        manifest["loadavg_end"] = _loadavg()
        manifest["n_rows"] = ctx.n_rows
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        entry = {k: manifest.get(k) for k in REGISTRY_FIELDS}
        with (runs_dir / "registry.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


REGISTRY_FIELDS = ("run_id", "stream", "name", "status", "config_hash", "git_sha", "git_dirty",
                   "started_at", "finished_at", "n_rows", "mode", "n_errors")


def mark_aborted(run_dir: Path | str, reason: str, runs_dir: Path | str | None = None) -> dict:
    """Close a run whose process died without reaching the ``finally`` of :func:`start_run`
    (machine crash, kill -9): status ``aborted`` with the reason, rows counted from
    ``results.jsonl``, one line appended to the registry. Partial artefacts stay as they are.
    """
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "running":
        raise ValueError(f"{run_dir.name}: status is {manifest.get('status')!r}, not 'running'")
    results = run_dir / "results.jsonl"
    manifest["status"] = "aborted"
    manifest["error"] = f"aborted: {reason}"
    manifest["finished_at"] = _now()
    manifest["n_rows"] = len(results.read_text().splitlines()) if results.exists() else 0
    path.write_text(json.dumps(manifest, indent=1, default=str))
    registry = Path(runs_dir) if runs_dir is not None else run_dir.parent
    with (registry / "registry.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: manifest.get(k) for k in REGISTRY_FIELDS}) + "\n")
    return manifest


if __name__ == "__main__":  # python -m cif.experiments.registry mark-aborted <run_dir> --reason ...
    import argparse

    ap = argparse.ArgumentParser(description="Run registry maintenance")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ab = sub.add_parser("mark-aborted", help="close a run whose process died")
    ab.add_argument("run_dir", type=Path)
    ab.add_argument("--reason", required=True)
    args = ap.parse_args()
    m = mark_aborted(args.run_dir, args.reason)
    print(f"{m['run_id']}: {m['status']} ({m['n_rows']} rows)")
