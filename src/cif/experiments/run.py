"""Experiment runner (R8; contracts in .claude/skills/run-experiment and .claude/skills/hpc-job).

Local, registered run (clean git tree required unless ``--smoke`` or ``--allow-dirty``)::

    python -m cif.experiments.run --config configs/E1/ablation.yaml [--smoke] [--dry-run]

Cluster shard (not registered; restart-safe; merged later by ``cif.experiments.ingest``)::

    python -u -m cif.experiments.run --config ablation.yaml --shard 3 --n-shards 16 \\
        --workers 16 --out results/ablation_shard3.jsonl

Config (YAML)::

    stream: E1
    name: ablation
    experiment: cif.experiments.toy     # module with cells(cfg) and run_cell(cell, cfg)
    seeds: [2000, 2001]
    inputs:                              # hashed into the manifest (data lineage)
      moex_v2: data/processed/moex_v2/provenance.json
    params: {...}                        # experiment-specific
    smoke: {seeds: [2000], params: {...}}   # deep-merged over the config for --smoke

The effective config (``smoke`` removed, ``mode`` = ``full`` / ``smoke``) is what the run hashes
and stores as ``config.json``; the YAML file itself is copied to ``config.yaml``. A result row is
``{"cell_key": ..., **cell, "host": ..., **result}``; a cell whose ``run_cell`` raises gives a row
with ``error`` and makes the exit code 1. In shard mode cells already in ``--out`` without an error are
skipped, so a requeued job continues where it stopped. Each invocation records a session (host,
times, config hash, input hashes, bundle SHA, thread pools) in ``<out>.meta.json`` *before* its
first row, so the rows of a killed job stay attributable. Relative input paths are taken from the
repository root.

Before anything runs, every input that is a provenance with recorded files is re-hashed
(:func:`verify_inputs`); a mismatch stops the run with exit code 2 (audit L-1). BLAS/OpenMP pools
are limited to ``threads`` (config key, default 1; plan §6) in the runner and in every worker, and
the pools in force are written to the manifest (audit TH-1).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import logging
import os
import socket
import sys
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from cif.data.provenance import recorded_files, verify_provenance
from cif.experiments import registry
from cif.experiments.timing import limit_threads, thread_state

log = logging.getLogger("cif.experiments.run")

REQUIRED_KEYS = ("stream", "name", "experiment")


def load_config(path: Path | str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: the config must be a mapping")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def effective_config(cfg: dict, smoke: bool = False) -> dict:
    """The config a run hashes and records: ``smoke`` overrides applied or dropped, ``mode`` set."""
    base = {k: v for k, v in cfg.items() if k != "smoke"}
    out = _deep_merge(base, cfg.get("smoke") or {}) if smoke else base
    out["mode"] = "smoke" if smoke else "full"
    return out


def cell_key(cell: dict) -> str:
    blob = json.dumps(cell, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def grid(cfg: dict) -> list[dict]:
    """All cells of the experiment, checked for unique keys."""
    cells = importlib.import_module(cfg["experiment"]).cells(cfg)
    keys = [cell_key(c) for c in cells]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{cfg['experiment']}.cells() returned duplicate cells")
    return cells


def resolve_inputs(cfg: dict) -> dict[str, Path]:
    out = {}
    for label, path in (cfg.get("inputs") or {}).items():
        p = Path(path)
        out[label] = p if p.is_absolute() else registry.REPO_ROOT / p
    return out


class InputMismatchError(RuntimeError):
    """An input dataset's files do not match the SHA-256 its provenance records (audit L-1)."""


def verify_inputs(inputs: dict[str, Path]) -> dict[str, dict]:
    """Re-hash the data files of every input that is a provenance with recorded files.

    An input is checked if it is a JSON provenance (``cif.data.provenance``) recording at least one
    file; any other input is only hashed into the manifest by the registry. A mismatch or a missing
    file raises :class:`InputMismatchError` before anything is registered.
    """
    out = {}
    for label, path in inputs.items():
        if path.suffix != ".json" or not path.is_file():
            continue
        try:
            prov = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(prov, dict) or not recorded_files(prov, path.parent, registry.REPO_ROOT):
            continue
        rep = verify_provenance(path, repo_root=registry.REPO_ROOT)
        out[label] = {"n_files": rep["n_files"], "n_ok": rep["n_ok"], "ok": rep["ok"]}
        if not rep["ok"]:
            bad = [m["file"] for m in rep["mismatch"]][:5] + rep["missing"][:5]
            raise InputMismatchError(
                f"input {label!r} ({path}) does not match its provenance: "
                f"{len(rep['mismatch'])} changed, {len(rep['missing'])} missing, e.g. {bad}"
            )
    return out


def _init_worker(n_threads: int) -> None:
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=int(n_threads))  # for the lifetime of the worker process


def _run_one(module_name: str, cell: dict, cfg: dict) -> dict:
    row = {"cell_key": cell_key(cell), **cell, "host": socket.gethostname()}
    try:
        result = importlib.import_module(module_name).run_cell(cell, cfg)
        row.update(result)
    except Exception as exc:  # recorded in the row; the run continues with other cells
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=5)
    return row


def run_cells(
    cfg: dict, cells: list[dict], workers: int, emit: Callable[[dict], None], n_threads: int = 1
) -> int:
    """Run ``cells`` (in parallel processes if ``workers > 1``); return the number of errors.

    BLAS/OpenMP pools are limited to ``n_threads`` in this process and in every worker (plan §6:
    one thread per solver; audit TH-1).
    """
    n_errors = 0
    if workers <= 1:
        with limit_threads(n_threads):
            rows: Iterable[dict] = (_run_one(cfg["experiment"], c, cfg) for c in cells)
            for row in rows:
                n_errors += bool(row.get("error"))
                emit(row)
        return n_errors
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(n_threads,)) as pool:
        futures = [pool.submit(_run_one, cfg["experiment"], c, cfg) for c in cells]
        for fut in as_completed(futures):
            row = fut.result()
            n_errors += bool(row.get("error"))
            emit(row)
    return n_errors


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut by a killed job
                log.warning("skipping a truncated line in %s", path)
    return rows


def _bundle_sha() -> str | None:
    if os.environ.get("CIF_BUNDLE_SHA"):
        return os.environ["CIF_BUNDLE_SHA"]
    path = registry.REPO_ROOT / "BUNDLE_SHA"
    return path.read_text().strip() if path.exists() else None


def _run_shard(cfg: dict, cells: list[dict], args) -> int:
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["cell_key"] for r in read_rows(out) if not r.get("error")}
    todo = [c for c in cells if cell_key(c) not in done]
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"sessions": []}
    n_threads = int(cfg.get("threads", 1))
    with limit_threads(n_threads):
        threads = thread_state()
    session = {
        "status": "running",
        "host": socket.gethostname(),
        "started_at": registry._now(),
        "shard": args.shard,
        "n_shards": args.n_shards,
        "workers": args.workers,
        "bundle_sha": _bundle_sha(),
        "config_hash": registry.config_hash(cfg),
        "inputs": registry._hash_inputs(resolve_inputs(cfg)),
        "inputs_verified": args.inputs_verified,
        "packages": registry._packages(),
        "threads": threads,
        "machine": registry.machine_state(),  # CPU, governors, load, affinity (R10)
        "n_cells": len(cells),
        "n_skipped_done": len(cells) - len(todo),
    }
    # the session is recorded before the first row: a killed (preempted) job leaves it as
    # "running", so its rows stay attributable (audit L-2)
    meta["sessions"].append(session)
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    log.info("shard %s/%s: %d cells, %d already done", args.shard, args.n_shards, len(cells),
             len(cells) - len(todo))
    with out.open("a", encoding="utf-8") as fh:
        def emit(row: dict) -> None:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()

        n_errors = run_cells(cfg, todo, args.workers, emit, n_threads=n_threads)
    session.update(status="finished", finished_at=registry._now(), n_run=len(todo),
                   n_errors=n_errors, loadavg_end=registry._loadavg())
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return 1 if n_errors else 0


def _run_local(cfg: dict, cells: list[dict], args) -> int:
    with registry.start_run(
        cfg["stream"], cfg["name"], cfg,
        runs_dir=args.runs_dir,
        seeds=cfg.get("seeds"),
        inputs=resolve_inputs(cfg),
        require_clean=not (args.smoke or args.allow_dirty),
        notes=cfg.get("notes", ""),
    ) as run:
        (run.run_dir / "config.yaml").write_text(args.config.read_text(encoding="utf-8"),
                                                 encoding="utf-8")
        handler = logging.FileHandler(run.run_dir / "log.txt", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            log.info("run %s: %d cells, mode %s", run.run_id, len(cells), cfg["mode"])

            def emit(row: dict) -> None:
                run.log_row(row)

            n_threads = int(cfg.get("threads", 1))
            with limit_threads(n_threads):
                run.manifest["threads_in_force"] = thread_state()["pools"]
            run.manifest["inputs_verified"] = args.inputs_verified
            n_errors = run_cells(cfg, cells, args.workers, emit, n_threads=n_threads)
            run.manifest.update(mode=cfg["mode"], n_cells=len(cells), n_errors=n_errors,
                                workers=args.workers)
            log.info("done: %d rows, %d errors", len(cells), n_errors)
        finally:
            root.removeHandler(handler)
            handler.close()
    return 1 if n_errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="apply the config's smoke overrides")
    parser.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="exploratory run on a dirty tree (not for the paper)")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None,
                        help="shard mode: append rows here instead of registering a run")
    args = parser.parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = effective_config(load_config(args.config), smoke=args.smoke)
    cells = grid(cfg)
    if args.shard is not None:
        if not 0 <= args.shard < args.n_shards:
            parser.error("--shard must be in [0, --n-shards)")
        cells = cells[args.shard::args.n_shards]
    if args.dry_run:
        print(f"{cfg['stream']}/{cfg['name']} ({cfg['mode']}): {len(cells)} cells")
        for cell in cells[:5]:
            print("  ", json.dumps(cell, default=str))
        if len(cells) > 5:
            print(f"   … and {len(cells) - 5} more")
        return 0
    try:
        args.inputs_verified = verify_inputs(resolve_inputs(cfg))
    except InputMismatchError as exc:
        log.error("%s", exc)
        print(f"ERROR: {exc}")
        return 2
    if args.out is not None:
        return _run_shard(cfg, cells, args)
    if args.shard is not None:
        parser.error("--shard needs --out (shards are merged by cif.experiments.ingest)")
    return _run_local(cfg, cells, args)


if __name__ == "__main__":
    sys.exit(main())
