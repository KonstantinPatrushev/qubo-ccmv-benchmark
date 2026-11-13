"""Synthetic result rows of the E1 attribution grid, with the schema of ``e1_attribution.run_cell``.

Not results: made-up numbers with a known structure for testing the analysis stage. The ceiling
``E_form(m) = base · m^-0.4`` of an instance is ``shift_hi`` times larger at the high level of h
(and exactly zero for ``m >= 50`` at the low level in a ``zero_share`` of the instances); the sampler
residual follows a mixed model in the scale of ``asinh(E_samp / τ)`` with known coefficients, the
random intercept of the instance and white noise. The pool of a call has ``pool_unique`` distinct subsets
(per sampler, drawn per cell from its own stream, 120 by default): ``n_pipe(m) = min(m, pool)``, the ceiling
at that size ``E_form(n)`` has the form of ``E_form(m)``, and the residual splits as in §12 B2,
``E_samp = E_prec + E_short``.
"""

from __future__ import annotations

import math

import numpy as np

from cif.experiments import run

MS = (1, 5, 20, 50, 200)
TAU = 1e-3
H_LEVELS = (0.15, 0.45)

# coefficients of y = asinh(E_samp / TAU), treatment coding as in the models
INTERCEPT, B_N24, B_HHI = 0.30, 0.20, 0.40
B_PEN_HIGH, B_OBJ, B_LOGM, B_OBJ_LOGM = -0.15, 0.25, -0.20, 0.10
SAMPLER_SHIFT = {"sa": 0.0, "tabu": 0.3, "swap_sa": -0.2}
COLD_SHIFT = {0.01: 0.0, 0.2: -0.8}  # swap-SA: the warmer end of the schedule leaves a much smaller residual
SD_INSTANCE, SD_NOISE = 0.20, 0.30


def configs(penalties=(10.0, 100.0), cold_ratios=(0.01,)) -> list[tuple[str, float | None, str | None, float | None]]:
    """``(sampler, penalty, beta, cold_ratio)`` of the nested grid (§7.3, §12 B3)."""
    out = [("sa", p, b, None) for p in penalties for b in ("default", "objective")]
    out += [("tabu", p, None, None) for p in penalties]
    out += [("swap_sa", None, None, c) for c in cold_ratios]
    return out


def config_id(sampler: str, penalty: float | None, beta: str | None, cold_ratio: float | None = None) -> str:
    parts = [sampler]
    if penalty is not None:
        parts.append(f"p{penalty:g}")
    if beta is not None:
        parts.append(beta)
    if cold_ratio is not None:
        parts.append(f"c{cold_ratio:g}")
    return "/".join(parts)


def synthetic_rows(
    *,
    n_pairs: dict[int, int] | None = None,
    solver_seeds=(20000, 20001, 20002),
    penalties=(10.0, 100.0),
    shift_hi: float = 3.0,
    zero_share: float = 0.3,
    no_valid_share: float = 0.0,
    seed: int = 0,
    first_pair_seed: int = 2000,
    pool_unique: dict[str, tuple[int, int]] | None = None,
    cold_ratios=(0.01,),
) -> list[dict]:
    """Rows for ``n_pairs`` pairs per N (default 10 at N = 16 and 10 at N = 24), both levels of h.

    ``pool_unique`` maps a sampler to the (low, high) range of the number of distinct subsets in the pool
    of a call; the default is 120 for all (a pool of at least m subsets for every m but 200). ``cold_ratios``
    are the levels of the cold end of the swap-SA schedule (§12 B3: 0.01 is the reference, 0.2 the
    exploratory level).
    """
    n_pairs = n_pairs or {16: 10, 24: 10}
    pool_unique = pool_unique or {}
    rng = np.random.default_rng(seed)
    pool_rng = np.random.default_rng([seed, 7919])  # its own stream: the other draws are unchanged
    rows: list[dict] = []
    pair_seed = first_pair_seed
    for n, count in n_pairs.items():
        for _ in range(count):
            base_lo = math.exp(rng.normal(-6.0, 0.8))
            zero = rng.random() < zero_share
            for level, h in enumerate(H_LEVELS):
                base = base_lo * (shift_hi * math.exp(rng.normal(0.0, 0.3)) if level else 1.0)
                def form_at(n, base=base, zero=zero, level=level):
                    """E_form at a pool of n subsets (and at T_m for n = m)."""
                    return 0.0 if (zero and not level and n >= 50) else base * n**-0.4

                form = {m: form_at(m) for m in MS}
                effect = rng.normal(0.0, SD_INSTANCE)
                scale = 0.1 * math.exp(rng.normal(0.0, 0.1))
                oracle_refine_s = 0.3 + 0.1 * rng.random()
                for sampler, penalty, beta, cold in configs(penalties, cold_ratios):
                    for solver_seed in solver_seeds:
                        cell = {"pair_seed": pair_seed, "n": n, "h": h, "sampler": sampler,
                                "penalty": penalty, "beta": beta, "cold_ratio": cold,
                                "solver_seed": solver_seed}
                        row = {"cell_key": run.cell_key(cell), **cell, "host": "server1"}
                        no_valid = rng.random() < no_valid_share
                        low, high = pool_unique.get(sampler, (120, 120))
                        n_unique = int(pool_rng.integers(low, high + 1))
                        row.update({
                            "status": "no_valid_samples" if no_valid else "ok",
                            "config_id": config_id(sampler, penalty, beta, cold), "consistent": True,
                            "K": 4, "s_cont": 9 + 2 * level, "s_cont_min_pair": 9, "binding": True,
                            "depth": (9 + 2 * level) / 4, "cv_vol": 0.15 + 0.3 * level, "cv_mu": 0.3,
                            "cond_sigma": 50.0, "mean_corr": 0.3, "n_subsets": 1820,
                            "f_star": -0.05, "scale": scale, "ref_type": "certified",
                            "scip_status": "optimal", "scip_gap": 0.0,
                            "oracle_s": 2.0, "oracle_enum_s": 0.1, "oracle_refine_s": oracle_refine_s,
                            "build_s": 0.001, "solve_s": 1.0 * math.exp(rng.normal(0.0, 0.05)),
                            "solve_cpu_s": 1.0, "n_samples": 1000, "n_valid_samples": 990,
                            "frac_valid": 0.99, "tau_top": 0.4, "tau_pool": 0.2 + 0.1 * rng.random(),
                            "hit_surrogate_min": True, "pool_n_unique": n_unique, "refine_s": 0.3,
                        })
                        for m in MS:
                            row[f"n_form_{m}"] = m
                        # the response of the model, then back to E_samp
                        y = INTERCEPT + SAMPLER_SHIFT[sampler] + B_N24 * (n == 24) + B_HHI * level + effect
                        if cold is not None:
                            y += COLD_SHIFT[cold]
                        for m in MS:
                            lm = math.log(m)
                            ym = y + B_LOGM * lm + rng.normal(0.0, SD_NOISE)
                            if penalty is not None and penalty >= 10.0:
                                ym += B_PEN_HIGH
                            if beta == "objective":
                                ym += B_OBJ + B_OBJ_LOGM * lm
                            e_samp = TAU * math.sinh(ym)
                            e_pipe = max(form[m] + e_samp, 0.0)
                            e_samp = e_pipe - form[m]  # keep the identity E_pipe = E_form + E_samp
                            n_pipe = 0 if no_valid else min(m, n_unique)
                            form_np = None if no_valid else form_at(n_pipe)
                            row[f"E_form_{m}"] = form[m]
                            row[f"E_pipe_{m}"] = None if no_valid else e_pipe
                            row[f"E_samp_{m}"] = None if no_valid else e_samp
                            row[f"E_form_at_npipe_{m}"] = form_np
                            row[f"E_prec_{m}"] = None if no_valid else e_pipe - form_np
                            row[f"E_short_{m}"] = None if no_valid else form_np - form[m]
                            row[f"n_pipe_{m}"] = n_pipe
                        rows.append(row)
            pair_seed += 1
    return rows
