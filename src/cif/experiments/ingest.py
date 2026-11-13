"""Register cluster shards as one run (contract: .claude/skills/hpc-job, "Приём результатов").

    python -m cif.experiments.ingest --config ablation.yaml \\
        --shards runs/incoming/<dir>/*.jsonl --bundle-sha-file runs/incoming/<dir>/BUNDLE_SHA \\
        --logs runs/incoming/<dir>/*.log runs/incoming/<dir>/*.err

Checks before anything is registered (audit L-2, L-3); any failure exits with code 1:
- every shard has a ``<shard>.meta.json`` (unless ``--allow-missing-meta``), and every session in
  it ran the local config (same ``config_hash``, mode ``full``), one code bundle (one ``bundle_sha``,
  equal to ``--bundle-sha-file`` if given) and the local inputs (same input hashes);
- rows are merged by ``cell_key``: per cell the last row without an error wins (the last error row
  if there is none); two successful rows of one cell with different results are a conflict
  (timings, host and traceback do not count); extra identical rows are counted as ``n_duplicates``;
- the grid of the config is complete (unless ``--allow-incomplete``) and has no foreign cells.

The run is registered with mode ``ingested``, the bundle SHA, the hosts, and copies of the shards,
their metadata and the job logs in ``ingest/``. The git state in the manifest is the local one at
ingest time; the code that computed the rows is ``bundle_sha``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from cif.experiments import registry
from cif.experiments.run import (
    cell_key,
    effective_config,
    grid,
    load_config,
    read_rows,
    resolve_inputs,
)

_VOLATILE = {"host", "traceback", "run_id"}


def _result(row: dict) -> str:
    """The part of a row that must agree between two successful rows of one cell."""
    kept = {k: v for k, v in row.items()
            if k not in _VOLATILE and not k.endswith("_s") and not k.startswith("time")}
    return json.dumps(kept, sort_keys=True, default=str)


def merge_rows(rows: list[dict]) -> tuple[dict[str, dict], int, list[str]]:
    """Per ``cell_key`` the last row without an error, else the last row; the number of extra rows;
    and the cells with conflicting successful results."""
    best: dict[str, dict] = {}
    results: dict[str, set[str]] = {}
    for row in rows:
        key = row["cell_key"]
        if not row.get("error"):
            results.setdefault(key, set()).add(_result(row))
        if key not in best or not row.get("error") or best[key].get("error"):
            best[key] = row
    conflicts = sorted(k for k, r in results.items() if len(r) > 1)
    return best, len(rows) - len(best), conflicts


def _hashes(inputs: dict) -> dict[str, str]:
    return {k: v["sha256"] for k, v in (inputs or {}).items()}


def machine_summary(sessions: list[dict]) -> dict:
    """What the manifest of an ingested run says about the machines the shards were timed on (R10):
    the CPU models and governors seen, the highest 1-minute load average at the start and at the end
    of a session (``None`` without data) and the number of sessions."""
    machines = [s["machine"] for s in sessions if s.get("machine")]

    def peak(values: list) -> float | None:
        values = [v[0] for v in values if v]
        return max(values) if values else None

    return {
        "cpu_models": sorted({m["cpu"] for m in machines if m.get("cpu")}),
        "cpu_governors": sorted({g for m in machines for g in (m.get("governors") or {})}),
        "loadavg_start_max": peak([m.get("loadavg") for m in machines]),
        "loadavg_end_max": peak([s.get("loadavg_end") for s in sessions]),
        "n_sessions": len(sessions),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cif.experiments.ingest")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--bundle-sha-file", type=Path, default=None)
    parser.add_argument("--logs", type=Path, nargs="*", default=[])
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-missing-meta", action="store_true",
                        help="accept shards without <shard>.meta.json (sessions not verifiable)")
    args = parser.parse_args(argv)

    def fail(message: str) -> int:
        print(f"ERROR: {message}")
        return 1

    cfg = effective_config(load_config(args.config))
    chash = registry.config_hash(cfg)
    cells = grid(cfg)
    expected = {cell_key(c): c for c in cells}

    # -- sessions: config, code bundle, inputs --------------------------------------------------
    metas, no_meta = [], []
    for path in args.shards:
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            metas.append((meta_path, json.loads(meta_path.read_text(encoding="utf-8"))))
        else:
            no_meta.append(path.name)
    if no_meta and not args.allow_missing_meta:
        return fail(f"no session metadata for {no_meta}; pass --allow-missing-meta to accept")
    sessions = [s for _, m in metas for s in m.get("sessions", [])]
    foreign = sorted({s.get("config_hash") for s in sessions} - {chash})
    if foreign:
        return fail(f"shard sessions ran other configs {foreign} (local config hash {chash})")
    bundles = sorted({s.get("bundle_sha") for s in sessions} - {None})
    file_bundle = args.bundle_sha_file.read_text().strip() if args.bundle_sha_file else None
    if len(bundles) > 1:
        return fail(f"rows come from several code bundles {bundles}")
    if file_bundle and bundles and bundles != [file_bundle]:
        return fail(f"shards ran bundle {bundles[0]}, BUNDLE_SHA says {file_bundle}")
    if file_bundle and sessions and any(s.get("bundle_sha") is None for s in sessions):
        return fail("some shard sessions recorded no bundle SHA while BUNDLE_SHA is given")
    bundle_sha = file_bundle or (bundles[0] if bundles else None)
    inputs = resolve_inputs(cfg)
    local = _hashes(registry._hash_inputs(inputs))
    if any(_hashes(s.get("inputs")) != local for s in sessions):
        return fail("the shards were computed on inputs whose hashes differ from the local ones")

    # -- rows -------------------------------------------------------------------------------------
    rows = [r for path in args.shards for r in read_rows(path)]
    best, n_duplicates, conflicts = merge_rows(rows)
    unexpected = sorted(set(best) - set(expected))
    missing = [k for k in expected if k not in best]
    n_errors = sum(1 for r in best.values() if r.get("error"))
    print(f"{len(rows)} rows from {len(args.shards)} shard(s): {len(best)} cells, "
          f"{n_duplicates} duplicate rows, {len(missing)} missing, {n_errors} with errors")
    if conflicts:
        return fail(f"{len(conflicts)} cells have different successful results, e.g. {conflicts[:3]}")
    if unexpected:
        return fail(f"{len(unexpected)} cells are not in the grid of {args.config} (wrong config?), "
                    f"e.g. {unexpected[:3]}")
    if missing and not args.allow_incomplete:
        return fail(f"{len(missing)} cells missing; rerun the shards or pass --allow-incomplete")

    with registry.start_run(
        cfg["stream"], cfg["name"], cfg, runs_dir=args.runs_dir, seeds=cfg.get("seeds"),
        inputs=inputs, notes=f"ingested from {len(args.shards)} shard(s)",
    ) as run:
        for key in expected:
            if key in best:
                run.log_row(best[key])
        dest = run.run_dir / "ingest"
        dest.mkdir()
        for path in [*args.shards, *(p for p, _ in metas), *args.logs]:
            shutil.copy2(path, dest / path.name)
        if args.bundle_sha_file:
            shutil.copy2(args.bundle_sha_file, dest / "BUNDLE_SHA")
        run.manifest.update(
            mode="ingested",
            bundle_sha=bundle_sha,
            hosts=sorted({s["host"] for s in sessions if s.get("host")}),
            sessions_unfinished=sum(1 for s in sessions if s.get("status") == "running"),
            **machine_summary(sessions),
            n_cells=len(cells),
            n_rows_in_shards=len(rows),
            n_duplicates=n_duplicates,
            n_missing=len(missing),
            n_errors=n_errors,
            shard_inputs_match=True if sessions else None,
            shard_files=[p.name for p in args.shards],
        )
    print(f"registered {run.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
