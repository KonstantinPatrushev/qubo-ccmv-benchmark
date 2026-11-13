"""Timing protocol for solver measurements (R10, ANALYSIS_PLAN §6).

The protocol: one machine, one thread for every solver, build / solve / refine times recorded
separately, wall time **and** CPU time, at least 3 repeats per (instance, solver), and the CPU
model and thread settings written to the manifest. This module holds the pieces that need code:

* :func:`measure` — wall time (``perf_counter``) and process CPU time (``process_time``) of a call;
  with one thread ``cpu_s ≈ wall_s``, a larger ratio reveals hidden threads. CPU time covers all
  threads of this process but not child processes;
* :func:`thread_env` — the thread-related environment variables and the core count, for the
  manifest (an unset variable is ``None``: the library default, usually all cores, is in force);
* :func:`single_thread_env` — the same variables set to one thread. BLAS and OpenMP read them when
  the library is loaded, so a script has to apply them **before** the first numpy import; this
  module therefore imports nothing from numpy::

      import os

      from cif.experiments.timing import single_thread_env

      os.environ.update(single_thread_env())  # before any numpy import
      import numpy as np  # noqa: E402
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["THREAD_VARS", "limit_threads", "measure", "single_thread_env", "thread_env", "thread_state"]

T = TypeVar("T")

#: Environment variables that set the thread count of OpenMP, MKL, OpenBLAS and numexpr.
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def measure(fn: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, dict[str, float]]:
    """Call ``fn(*args, **kwargs)``; return ``(result, {"wall_s": ..., "cpu_s": ...})``.

    ``wall_s`` is elapsed time from :func:`time.perf_counter`, ``cpu_s`` the process CPU time
    (user + system, all threads) from :func:`time.process_time`. The wall interval encloses the CPU
    one, so for a single thread ``cpu_s ≤ wall_s``; in general ``cpu_s ≤ wall_s · cores``.
    Exceptions of ``fn`` propagate and no time is returned.
    """
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = fn(*args, **kwargs)
    cpu_s = time.process_time() - cpu_start
    wall_s = time.perf_counter() - wall_start
    return result, {"wall_s": wall_s, "cpu_s": cpu_s}


def thread_env() -> dict[str, str | int | None]:
    """Thread settings for the run manifest: :data:`THREAD_VARS` and ``cpu_count``.

    Values of the variables are strings, or ``None`` if unset; ``cpu_count`` is
    :func:`os.cpu_count` (``None`` if undetermined).
    """
    env: dict[str, str | int | None] = {name: os.environ.get(name) for name in THREAD_VARS}
    env["cpu_count"] = os.cpu_count()
    return env


def single_thread_env() -> dict[str, str]:
    """``{variable: "1"}`` for every name in :data:`THREAD_VARS`; apply with ``os.environ.update``.

    Must run before numpy (BLAS) and other threaded libraries are imported: they read the
    variables at load time and ignore later changes. A fresh dict on every call.
    """
    return dict.fromkeys(THREAD_VARS, "1")


def thread_state() -> dict:
    """Thread settings for the manifest (audit TH-1): ``env`` (:data:`THREAD_VARS`), ``cpu_count``
    and ``pools``, the thread pools actually in force (threadpoolctl).

    The variables act only if set before BLAS is loaded, so ``pools`` is what counts; ``env`` alone
    can claim one thread while OpenBLAS runs on every core.
    """
    from threadpoolctl import threadpool_info

    pools = [
        {k: p.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
        for p in threadpool_info()
    ]
    return {"env": {name: os.environ.get(name) for name in THREAD_VARS},
            "cpu_count": os.cpu_count(), "pools": pools}


def limit_threads(n: int = 1):
    """Context manager limiting every BLAS/OpenMP pool of this process to ``n`` threads (plan §6).

    Works after numpy is loaded, unlike :func:`single_thread_env`.
    """
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=int(n))
