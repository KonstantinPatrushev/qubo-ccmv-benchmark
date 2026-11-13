"""Toy experiment: the reference implementation of the runner protocol (``cif.experiments.run``).

An experiment module provides two functions:

* ``cells(cfg) -> list[dict]`` -- the grid, JSON-serialisable and in a stable order;
* ``run_cell(cell, cfg) -> dict`` -- one result row for one cell.

Here a cell is ``{"n": n, "seed": seed}`` and the result is a deterministic draw. ``params.fail_on``
lists values of ``n`` whose cells raise, to exercise error handling.
"""

from __future__ import annotations

import numpy as np


def cells(cfg: dict) -> list[dict]:
    return [{"n": int(n), "seed": int(s)} for n in cfg["params"]["n"] for s in cfg.get("seeds", [0])]


def run_cell(cell: dict, cfg: dict) -> dict:
    if cell["n"] in cfg["params"].get("fail_on", []):
        raise RuntimeError(f"toy failure at n={cell['n']}")
    rng = np.random.default_rng(cell["seed"])
    return {"value": float(rng.normal() * cell["n"]), "status": "ok"}
