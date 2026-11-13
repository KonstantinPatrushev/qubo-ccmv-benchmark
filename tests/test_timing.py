"""Tests for cif.experiments.timing (R10): wall and CPU time, thread environment (manifest)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import cif
from cif.experiments.timing import measure, single_thread_env, thread_env

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SRC = Path(cif.__file__).resolve().parents[1]


def _burn(seconds: float) -> int:
    """Busy single-threaded loop for about ``seconds`` of wall time."""
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        n += 1
    return n


# ---------------------------------------------------------------------------
# 1. measure
# ---------------------------------------------------------------------------


def test_measure_returns_the_result_and_passes_arguments():
    result, _ = measure(lambda a, b=0, *, c=0: (a, b, c), 1, 2, c=3)
    assert result == (1, 2, 3)


def test_measure_reports_wall_and_cpu_seconds():
    _, timing = measure(_burn, 0.02)
    assert set(timing) == {"wall_s", "cpu_s"}
    assert all(isinstance(v, float) for v in timing.values())
    assert timing["wall_s"] >= 0.02 - 1e-3
    assert timing["cpu_s"] > 0.0


def test_measure_uses_perf_counter_for_wall_and_process_time_for_cpu(monkeypatch):
    wall = iter([10.0, 12.5])
    cpu = iter([3.0, 4.0])
    with monkeypatch.context() as m:  # patched only around the call: nothing else reads the clocks
        m.setattr(time, "perf_counter", lambda: next(wall))
        m.setattr(time, "process_time", lambda: next(cpu))
        _, timing = measure(lambda: None)
    assert timing == {"wall_s": 2.5, "cpu_s": 1.0}


def test_cpu_time_is_bounded_by_wall_time_times_the_number_of_cores():
    _, timing = measure(_burn, 0.05)
    assert timing["cpu_s"] <= timing["wall_s"] * (os.cpu_count() or 1) + 1e-3


def test_a_single_thread_does_not_use_more_cpu_than_wall_time():
    # the protocol runs solvers on one thread, where cpu_s ~ wall_s; a generous margin for the
    # different clocks
    _, timing = measure(_burn, 0.05)
    assert timing["cpu_s"] <= timing["wall_s"] * 1.1 + 5e-3


def test_sleeping_costs_wall_time_but_hardly_any_cpu_time():
    _, timing = measure(time.sleep, 0.05)
    assert timing["wall_s"] >= 0.05 - 1e-3
    assert timing["cpu_s"] < 0.5 * timing["wall_s"]


def test_measure_propagates_exceptions():
    def boom():
        raise RuntimeError("solver crashed")

    with pytest.raises(RuntimeError, match="solver crashed"):
        measure(boom)


# ---------------------------------------------------------------------------
# 2. thread_env: what the manifest records
# ---------------------------------------------------------------------------


def test_thread_env_has_the_thread_variables_and_the_cpu_count():
    env = thread_env()
    for name in THREAD_VARS:
        assert name in env
    assert env["cpu_count"] == os.cpu_count()


def test_thread_env_reads_the_environment_and_marks_unset_variables_as_none(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    env = thread_env()
    assert env["OMP_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] is None  # unset: the library default (all cores) is in force


def test_thread_env_is_json_serialisable_for_the_manifest():
    assert json.loads(json.dumps(thread_env())) == thread_env()


# ---------------------------------------------------------------------------
# 3. single_thread_env: set before numpy is imported
# ---------------------------------------------------------------------------


def test_single_thread_env_pins_every_thread_variable_to_one():
    env = single_thread_env()
    assert set(env) == set(THREAD_VARS)
    assert all(value == "1" for value in env.values())  # strings: os.environ needs str
    assert set(env) == set(thread_env()) - {"cpu_count"}


def test_single_thread_env_returns_a_fresh_dict():
    env = single_thread_env()
    env["OMP_NUM_THREADS"] = "8"
    assert single_thread_env()["OMP_NUM_THREADS"] == "1"


def _run_python(code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_importing_the_timing_module_does_not_import_numpy():
    # otherwise a script could not call single_thread_env() before the first numpy import
    out = _run_python(
        "import sys\n"
        "from cif.experiments.timing import measure, single_thread_env, thread_env\n"
        "single_thread_env(); thread_env()\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'numpy'))"
    )
    assert out == "[]"


def test_single_thread_env_applied_before_numpy_gives_single_threaded_blas():
    pytest.importorskip("threadpoolctl")
    out = _run_python(
        "import os\n"
        "from cif.experiments.timing import single_thread_env\n"
        "os.environ.update(single_thread_env())\n"
        "import numpy as np\n"
        "np.ones((200, 200)) @ np.ones((200, 200))\n"
        "from threadpoolctl import threadpool_info\n"
        "print(sorted({lib['num_threads'] for lib in threadpool_info()}))"
    )
    if out == "[]":
        pytest.skip("no BLAS or OpenMP library reports its thread count")
    assert out == "[1]"
