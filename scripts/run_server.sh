#!/usr/bin/env bash
# Run (or resume) one E2 stage on the run server (no job scheduler): a thin wrapper of
# `python -m cif.experiments.launch run` that fixes the environment of the study (ANALYSIS_PLAN §6 and §12: one
# thread per job, the environment of environment.lock.yml with PYTHONNOUSERSITE=1, on a shared server where the
# limits of the plan stand in for an exclusive mode).
#
#   scripts/run_server.sh configs/E2/tune_screen.yaml --cores 0-31 --pin            # run / resume
#   scripts/run_server.sh configs/E2/tune_screen.yaml --cores 0-31 --pin --smoke    # rehearsal first
#   scripts/run_server.sh configs/E2/tune_screen.yaml --jobs 12 --shards 256        # another concurrency
#
# Everything after the config is passed to the launcher (python -m cif.experiments.launch run --help).
# PYTHON names the interpreter (default: python of the active environment).
# The target (gpu-07: 2 x Xeon Silver 4216, 32 physical cores, 64 threads, governor schedutil that cannot be changed,
# shared with other users): at most 16 jobs in a timing stage, --pin binds each job to one physical core, 0-31 are
# the physical cores; cores that other work keeps busy are named by the launcher and left out with --exclude-cores.
# Resuming is the same command: finished shards are skipped, an interrupted one continues after its last row.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -lt 1 || "$1" == -* ]]; then
    echo "usage: $0 <config.yaml> [launcher options]" >&2
    exit 2
fi
config="$1"
shift

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${CIF_BUNDLE_SHA:-}" && -f BUNDLE_SHA ]]; then
    CIF_BUNDLE_SHA="$(cat BUNDLE_SHA)"
    export CIF_BUNDLE_SHA
fi

# what the timings were taken on, next to the shards (the launcher also records it in its manifest)
echo "host=$(hostname) bundle=${CIF_BUNDLE_SHA:-unknown} governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown) load=$(cut -d' ' -f1-3 /proc/loadavg)"

exec "${PYTHON:-python}" -u -m cif.experiments.launch run --config "$config" "$@"
