# configs/E3 — the E3 experiment (RQ3) on the run server

Everything here implements `docs/ANALYSIS_PLAN.md` v2.4 (frozen) as it stands; where the plan is ambiguous the
module docstrings and the section "How the plan was read" below say how it was read. The code:
`src/cif/experiments/e3_data.py` (markets, periods, the fold table made by the engine itself),
`e3_walkforward.py` (the cells), `e3_pilot.py` (the rules of the pilot), `e3_assemble.py` (`F_k` and the paths
through the engine), `e3_analysis.py` (H3a, H3b, H3c, sensitivities, tables).

## What is here

| file | what | cells | seeds | `launch` (jobs / shards / timing) | needs before it runs |
|---|---|---|---|---|---|
| `pilot.yaml` | FF49 pilot period (test windows 1963-07 … 1972-12): `s_cont`, `K_f`, certified optimum and rank 2, λ ∈ {5, 10, 20}, both arms | 672 = 112 folds × 3 λ × 2 arms | calibration (10000) | 16 / 96 / no | the public stamp of the plan (set 2026-10-11, §12) |
| `pilot_operators.yaml` | the **pilot of the descriptive operators** (б), (а): the same pilot period at `lambda_f`, both arms, rank 2 and the operators with the levels, the 30 replicas and the limits of `ff49.yaml` — the folds on which the plan reads the levels `L` ("L — уровни, достигнутые хотя бы на 80 % фолдов пилотного периода", per arm) | 224 = 112 folds × 2 arms | calibration (10001) | 16 / 56 / no | `lambda_f` from the pilot report (§12) |
| `ff49.yaml` | **FF49 confirmatory** (primary market): test windows 1973-01 … 2025-12, the 32 best supports (ranks 2–33), the real solvers (в) on MV, the descriptive operators (б), (а) on MV | 1272 = 636 folds × 2 arms | test (20000) | 16 / 212 / **yes** | `lambda_f` from the pilot (§12); the tuned solver configs of E2 (`solvers.tuned: true`); `BUNDLE_SHA` |
| `us.yaml` | US-MV, rank 2 (H3a only), S&P 500 on the date, top 100 by dollar volume, test windows 2013-01-04 … 2025-12 | 155 (MV) | test (20000) | 16 / 155 / no | `lambda_f`; `BUNDLE_SHA` |
| `us_sens_irregular.yaml` | the same without the tickers with a remaining day of an irregular Yahoo ratio (102 days, 81 tickers) in the windows of the fold | 155 | test (20000) | 16 / 155 / no | `us.yaml` done; `lambda_f` |
| `analysis.yaml` | the stages after the cells: pilot report, assembly of the paths, analysis | — | — | — | run ids of the registered runs; the pilot margins |

The cell counts are those of the calendars of the datasets of 2026-10-08 (`python -m cif.experiments.run --config … --dry-run`
prints them once `lambda_f` is set; `--smoke --dry-run` works at any time). Every config has a `smoke:` block: a
synthetic factor-model market of 12 assets, a handful of cells, calibration seeds, 0.3 s budgets. **No real market
data is read by a smoke run.**

## The run server (what E3 asks of the launcher)

The server (gpu-07, 2 × Xeon Silver 4216, 32 physical cores, 64 threads, governor `schedutil`) is shared (plan §12,
2026-10-11): there is no exclusive mode, so the plan's limits stand in for it. For E3:

* **At most 16 single-thread jobs, pinned to physical cores:** `scripts/run_server.sh <config> --cores 0-31 --pin`.
  `launch.jobs: 16` is in every config; `--jobs` can lower it. Before it binds the jobs the launcher samples the cores for a
  second and names those that other work keeps busy; then add `--exclude-cores 3,17` (the list it prints). One launcher at a time (`flock` on `/tmp/cif_launch.lock`).
* **`launch.timing`.** Only `ff49.yaml` is a timing stage: its real solvers (в) have wall-clock budgets (SCIP 10 s, tabu / SA /
  LS 1 s end to end), so the incumbent they return depends on the machine and its load, and the launcher holds the stage to
  half of the physical cores (`--allow-oversubscribe` would lift that: do not). The other configs certify (SCIP stops by its
  gap, not by the clock; the operators (б), (а) stop by their step limits, not by the clock) and take no timings, but the
  cap of §12 holds for them as well. Load average and governor go into the manifest of every shard and of the launcher; the
  rows hold `times` (wall seconds of each part) and the solvers' `wall_s`.
* **`launch.shards`** (fixed at the first start of an output directory; resume with the same value): `pilot` 96 × 7 cells,
  `pilot_operators` 56 × 4 cells (2 MV and 2 GMV), `ff49` 212 × 6 cells (3 MV and 3 GMV, folds 212 apart: the grid is ordered arm, λ, fold, so `cells[i::S]` mixes both
  arms and spreads the folds whatever the parity of `S`), `us*` 155 × 1 cell (a cell can take hours at N = 100; a shard
  must not hold a job slot for a day). A cell depends on nothing but its coordinates and the config: any `cells[i::S]`
  cut is valid, and a cell computed twice (a shard repeated elsewhere) gives rows that differ in the clocks only
  (the ingest ignores the `times` of a row), except the real solvers' paths, which are timing-dependent by nature.
* **`BUNDLE_SHA`.** `ff49`, `us`, `us_sens_irregular` read test seeds, so the launcher refuses to start them
  without the code bundle's SHA (`BUNDLE_SHA` in the root of the bundle or `CIF_BUNDLE_SHA`); build the bundle from the clean
  committed tree (recipe below). A smoke rehearsal does not need it.
* **One thread per solver** (SCIP, HiGHS, BLAS; `scripts/run_server.sh` sets `OMP_NUM_THREADS=1` … and
  `PYTHONNOUSERSITE=1`). **Memory:** measured on synthetic panels, two FF49-scale MV cells in one process (32 ranks, the solvers and the operators)
  peak at 0.30 GB resident, the S&P dataset loads to 0.34 GB (0.17 GB of it the interpreter and the libraries): below 1 GB a job,
  16 GB for 16. (The first measurement said 1.9 GB after two cells: `solve_miqp_native` left every SCIP model alive, 13 MB per
  MIQP; fixed in this stream, `tests/test_scip_native.py`.) **Start-up per shard:** the process builds the fold table of the period once,
  which is one pass of the engine (636 folds of N = 49: 7 s on a synthetic panel; 155 folds of N = 100: 3 s plus the choice of the
  universe on 155 dates, which needs the real data and was not measured).
* **Resume = the same command.** Finished shards are skipped, an interrupted one continues after its last row; a cell with
  an `error` row is tried again. Run it in `tmux` / `screen` / `nohup`: a dropped ssh connection stops the jobs.

**The bundle** (the recipe of `configs/E2/README.md`, plus the data of E3; the agents do not use `scp` / `ssh`):

```bash
git status --porcelain            # must be empty: the bundle is the committed tree
SHA=$(git rev-parse --short HEAD); B=e3_bundle_$SHA
mkdir -p $B && git archive HEAD src configs scripts pyproject.toml environment.yml environment.lock.yml | tar -x -C $B
mkdir -p $B/data/raw $B/data/processed
cp -r data/raw/ff49 data/raw/sp500_pit data/raw/riskfree_v2 $B/data/raw/
cp -r data/processed/ff49 data/processed/sp500_pit data/processed/riskfree_v2 $B/data/processed/
echo $SHA > $B/BUNDLE_SHA && tar czf $B.tgz $B
```

The runner re-hashes every input against its provenance before it starts (a missing file stops the job with exit code
2): the provenance of each dataset records its raw files too, so `data/raw/ff49` (the French zips), `data/raw/sp500_pit`
(the Yahoo cache, 68 MB, and the Wikipedia snapshots) and `data/raw/riskfree_v2` go with the processed directories. The
pilot needs `ff49` only (22 MB in all); the US stages need `sp500_pit` (113 MB) and `riskfree_v2` (1.5 MB).

## Order of runs

`<run>` is a registered run directory (`runs/<run_id>`), `<d>` the directory copied back from the server. Cells run on the
server; `ingest`, `e3_pilot`, `e3_assemble` and `e3_analysis` run anywhere the data and the repository are (they are
registered runs of their own; the assembly and the analysis use at most `assembly.workers` = 16 processes).

```bash
# 0. checks (laptop and server): the datasets match their provenance; the rehearsal on the smoke grid
python -m cif.data.provenance verify data/processed/ff49/provenance.json data/processed/sp500_pit/provenance.json data/processed/riskfree_v2/provenance.json
python -m cif.experiments.launch plan --config configs/E3/pilot.yaml --cores 0-31 --pin
scripts/run_server.sh configs/E3/pilot.yaml --cores 0-31 --pin --smoke

# 1. the pilot (FF49 pilot period): cells on the server, register, the report for §12
scripts/run_server.sh configs/E3/pilot.yaml --cores 0-31 --pin
d=runs/incoming/E3-ff49_pilot
python -m cif.experiments.ingest --config configs/E3/pilot.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log
# put the run id into configs/E3/analysis.yaml (sources.pilot.run), then
python -m cif.experiments.e3_pilot --analysis-config configs/E3/analysis.yaml --source pilot
# 2. copy `section_12_draft` of pilot_report.json into §12 of the plan; put `yaml.lambda_f` into pilot_operators.yaml,
#    ff49.yaml, us.yaml and us_sens_irregular.yaml and `yaml.margins` into the `margins:` of analysis.yaml; commit all of it.

# 2b. the pilot of the operators (б), (а) at lambda_f, both arms (224 cells): the plan reads the levels L on its folds
scripts/run_server.sh configs/E3/pilot_operators.yaml --cores 0-31 --pin
d=runs/incoming/E3-ff49_pilot_operators
python -m cif.experiments.ingest --config configs/E3/pilot_operators.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log
# put the run id into configs/E3/analysis.yaml (sources.pilot_operators.run); step 5 reads L from it

# 3. FF49 confirmatory (needs lambda_f, the tuned solvers of E2 with solvers.tuned: true, a clean tree, BUNDLE_SHA)
scripts/run_server.sh configs/E3/ff49.yaml --cores 0-31 --pin
d=runs/incoming/E3-ff49_confirmatory
python -m cif.experiments.ingest --config configs/E3/ff49.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log

# 4. US-MV and its sensitivity (needs lambda_f, BUNDLE_SHA)
scripts/run_server.sh configs/E3/us.yaml --cores 0-31 --pin
d=runs/incoming/E3-us_confirmatory
python -m cif.experiments.ingest --config configs/E3/us.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log
scripts/run_server.sh configs/E3/us_sens_irregular.yaml --cores 0-31 --pin
d=runs/incoming/E3-us_sens_irregular
python -m cif.experiments.ingest --config configs/E3/us_sens_irregular.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log

# 5. the paths through the engine: fill sources.*.run of analysis.yaml with the ingested run ids, then one assembly each
#    (the assembly of ff49 reads the levels L of the operators from sources.pilot_operators, and refuses without it)
python -m cif.experiments.e3_assemble --analysis-config configs/E3/analysis.yaml --source ff49
python -m cif.experiments.e3_assemble --analysis-config configs/E3/analysis.yaml --source us
python -m cif.experiments.e3_assemble --analysis-config configs/E3/analysis.yaml --source us_irregular

# 6. the statistics: fill assemblies.* with the three assembly run ids, then
python -m cif.experiments.e3_analysis --analysis-config configs/E3/analysis.yaml
```

Between the stages: `python -m cif.experiments.launch status --config configs/E3/ff49.yaml` (cells done per shard). The stages
5 and 6 refuse a dirty tree unless `--allow-dirty` (exploratory); the cell stages refuse test seeds without `BUNDLE_SHA`.
Fill the ids and the pilot values in `analysis.yaml` **before** the stage that reads them, commit, and run from the clean tree:
a run that was made from other cells or another config than the one named is refused (the config hash and the SHA-256
of the rows are checked).

What the console says when a stage went through (the rehearsal on the synthetic grid printed exactly these shapes):

| stage | last lines |
|---|---|
| launcher | `finished: 96 shards run, 0 failed, 0 cells not done; manifest …/launch_manifest.json` (`pilot`; 212 for `ff49`, 155 for `us*`) |
| `ingest` | `672 rows from 96 shard(s): 672 cells, 0 duplicate rows, 0 missing, 0 with errors` and `registered <run_id>` (`pilot_operators`: 224 rows from 56 shards; `ff49`: 1272 rows from 212 shards; `us*`: 155 from 155); `missing` other than 0 stops it (exit code 1; `--allow-incomplete` registers anyway), `with errors` other than 0 means cells whose last row is an error: rerun the launcher (it retries them) and ingest again |
| `e3_pilot` | `<run_id>: lambda_f = …, Δ_SR = …, δ = …`, then the line for §12 and the YAML values |
| `e3_assemble` | `<run_id>: 636 folds, <n> in F_k, 4 passes` (`ff49`; `us`: 155 folds) |
| `e3_analysis` | `<run_id>: <n> arms, <m> tables` |

## What each stage writes

| stage | where | what |
|---|---|---|
| cells (launcher) | `runs/incoming/E3-<name>/<name>_shard<i>.jsonl`, `launch_manifest.json`, `BUNDLE_SHA`, `logs/` | one row per cell (below); the manifest has host, CPU, governors, load samples, cores of the jobs |
| `ingest` | `runs/<run_id>/results.jsonl`, `manifest.json` (mode `ingested`) | the registered run of the stage |
| `e3_pilot` | `runs/<run_id>/pilot_report.json` | `lambda_f` and the shares of binding folds, `σ*`, `Δ_SR`, `σ²_*`, `δ`, the power pilot (ρ, the half-width of the pilot and rescaled), `section_12_draft` (a line for §12), `yaml` (values for the configs) |
| `e3_assemble` | `runs/<run_id>/` | `meta.json`, `folds.parquet`, `days.parquet`, `paths_<pass>.parquet` (daily returns of every path), `foldstats_<pass>.parquet` (turnover, cost), `doses_<arm>.parquet`, `solvers.parquet`, `operators/` (reach and gaps of the replicas, the paths of the replicas of the levels of `L`, and `sets.json`: `L`, `L_a`, the masks of `F_a` / `F_b` and where `L` came from — the pilot run and the share reached per level on its folds) |
| `e3_analysis` | `runs/<run_id>/tables/` | `analysis.json`; `h3a.csv`, `h3b.csv`, `h3b_per_rank.csv`, `h3c.csv`; `rank2_doses.csv`, `doses_by_rank.csv`, `exclusions.csv`, `regime.csv`, `ties.csv`; `net_metrics.csv`, `net_differences.csv`; `sensitivity.csv`, `arm_contrast.csv`; `operator_reach.csv`, `operator_gaps.csv`, `operator_levels.csv`, `operator_sets.csv` |

A **cell row** (`e3_walkforward` docstring): the coordinates (`market, period, arm, lam, fold, formation_date`), the fold
(`assets`, dates, `problem_check`: sums of μ and Σ that the assembly compares), `s_cont, K, K_used, binding,
lower_bound_active`, the certificate of the optimum (`optimum_certified, optimum_status, f_star, scale`, SCIP's bounds,
the incumbent `optimum_idx / optimum_w`), the k best supports (`ranks`: `rank, support, w, objective, dose_bp, certified,
tie`; `kbest_n_distinct, kbest_all_certified, kbest_exhausted, kbest_boundary_tie, kbest_tie_share, kbest_error`), the real
solvers (`solvers`: `objective, gap_bp, rank, rank_matched, support, idx, w, wall_s`), the operators (`swap`, `perturb`:
per level `status`, `n_found`, `n_evals`, the replicas' supports, weights and gaps), the seeds, and `times`. `status` is one
of `ok, degenerate, kbest_incomplete, kbest_inconsistent, uncertified_optimum, no_incumbent, infeasible_box`; only `ok`
rows can be in `F_k`.

## Cost (estimates from synthetic timings)

How the numbers were made: the cells and the stages of this stream were run on **synthetic factor-model panels of the plan's
sizes** (FF49: N = 49, u = 0.20, K = 6, 756 / 21 days; US: N = 100, u = 0.10, K = 11; `e3_data.synthetic_market`), eight folds
per arm (six for the US), one thread per job, three jobs at a time, on a laptop (Intel Core i5-10210U) with other work
running. No market data were used. Two things are therefore unknown and these are **estimates**, not forecasts: how hard the
MIQPs of the real covariance matrices are, and how fast a core of the server is. The first measurement of the FF49 cells was
made at a load average of 8–10 and took 1.9 times as long as the repeat at a load average of about 3 (cells 179 s / 211 s
against 94 s / 110 s; the repeat also had the memory fix, so the two causes are not separated); the table gives the lighter
figure and, after a slash, the heavier reading (the measurement at the high load where there is one, twice the lighter figure
otherwise).

| stage | units | per unit (synthetic mean) | CPU-h |
|---|---|---|---|
| `pilot.yaml` cells | 672, rank 2 only | MV 2.6–5.4 s, GMV 2.7–4.6 s (λ = 5 … 20) | 0.4 / 0.7 |
| `pilot_operators.yaml` cells | 224, rank 2 and the operators of both arms | measured 45 s (MV) and 37 s (GMV) at a load average near 10, of which the swap chains 33 s and 27 s; at the lighter load not measured, scaled from the parts of the FF49 cells: about 14 s and 12 s | 0.8 / 2.6 |
| `e3_pilot` | one report | block calibration of 2 arms at 2 352 days, 1000 pseudo-series: 0.10 h + 0.07 h | 0.2 |
| `ff49.yaml` cells | 1272 = 636 × 2 | MV 94 s (optimum 1.2, the 32 ranks and the tie check 78, solvers 4.2, swap 11, perturb 0.2); GMV 110 s (ranks 108) | **36 / 69** |
| `us.yaml`, `us_sens_irregular.yaml` cells | 155 + 155 | 16 s (optimum, rank 2, tie check) | 1.4 / 2.8 |
| `e3_assemble` ff49 | 172 main path passes + up to 840 replica passes of (б), (а) | 7 s per pass of 636 folds (10–13 s under load) | 2 / 3.6 |
| `e3_assemble` us, us_irregular | 12 + 4 passes | 2.5 s | < 0.1 |
| `e3_analysis` | ff49 MV, GMV; us; us_irregular; the operators | calibration with 1000 pseudo-series 0.59 h (MV), 0.48 h (GMV), 0.16 h (each US); the rest 3, 5 and 0.2 min; each operator adds a calibration of up to 0.6 h | 1.6–2.8 |
| **E3 in all** | | | **≈ 44 / 83** |

For scale: §8 of the plan puts E3 at 60–250 CPU-h and the enumeration of the best supports alone, by the reviewer, at 10–40
CPU-h (about 90 s per fold-arm at N = 49); the synthetic cells need 78 s (MV) and 108 s (GMV) per fold-arm at the lighter load,
33 CPU-h for the 636 folds. **Wall time** on 16 jobs is the CPU-h divided by 16 plus the tail of the last shards: the 36 CPU-h of
the FF49 cells are 2.3 h (212 shards of six cells, about 10 minutes each, 20 under the heavier load), the assembly of FF49 about 10 minutes on 16 workers,
`e3_analysis` a few hours (it is serial: the calibrations of the arms could run side by side, but they do not yet).

What bounds a cell is the limits, not these means: 2 h for the optimum and 600 s for each of its 33 MIQPs (7.5 h per FF49 cell,
2.3 h per US cell). A fold that reaches them is `uncertified_optimum` / `kbest_incomplete` and leaves `F_k`; if the real
matrices are harder than the synthetic ones that is where the cost and the exclusions will show first, in the `times` of the first
shards.

Disk: an MV cell row is about 84 KB, a GMV row 18 KB (65 MB of shards for FF49); the assembly of FF49 writes about 0.1 GB with the
operators.

## How the plan was read

Each item is a place where the plan leaves a choice; the code takes the most literal reading.

1. **Periods.** A period is an independent contiguous walk-forward: its first test window starts on the first trading day on
   or after `start`, windows are whole 21-day blocks ending on or before `end`, training windows may reach before `start`
   (they read the past only). On the calendars of 2026-10-08: pilot 112 folds (1963-07-01 … 1972-12-05), confirmatory 636
   (1973-01-02 … 2025-12-19), US 155 (2013-01-04 … 2025-12-11).
2. **US start.** The plan says "2013-01"; the panel begins on 2010-01-04 and a 756-day window ends on 2013-01-03, so the first
   test day is 2013-01-04 (`period.start` in `us.yaml`).
3. **FF49 universe.** An industry is in a fold if its 756-row training window has no missing price (the plan). A price missing
   inside a test window is handled by the engine's per-fold mode: the holding is frozen at its last price (zero return) while the
   series is missing and earns the move across the gap on the day it resumes.
4. **`K_f` and the two arms.** `K_f = max(⌈1/u⌉ + 1, ⌊s_cont,f/2⌋)` is computed from the continuous relaxation of the **MV** problem
   at `λ_f` and used for both arms ("с той же кардинальностью" for GMV); the row also holds `s_cont_arm`, the support of
   the arm's own relaxation. `K_f > N_f` is clipped to `N_f` (`K_used`; such a fold has one portfolio and is `degenerate`); a box
   that cannot reach the budget (`N_f·u < 1`) is `infeasible_box`.
5. **Certificates.** The optimum is `certified_optimum` for `Σz ≤ K` (SCIP `limits/gap = 1e-6`, up to 2 h): a relative gap of
   1e-6 implies the plan's absolute gap `1e-6·s_ref` because `s_ref ≥ |f|`. The ranks use `limits/absgap = 1e-6·s_ref` **of the optimum** ("как у
   оптимума"); `cif.backtest.kbest` used to judge a rank by 1e-6 of the scale of its own, smaller, incumbent and ended the
   enumeration (and dropped the fold from `F_k`, more often where doses are large) on a gap that was inside the limit SCIP had been
   given; it now certifies by the limit SCIP was given (red test first, `tests/test_kbest.py`). Ranks are the refinements
   (the fixed point of re-optimising the weights on the support and dropping zeros) of the K-subsets (`Σz = K` in the
   enumeration), with the extra MIQP for the tie at rank R + 2 (`boundary_check`).
6. **Real solvers (в).** One run per fold and solver, seeds `SeedSequence([solver_seed, fold, arm, solver])`; all four get the common
   start of §5 (SCIP as a warm start, the heuristics in the pool / as the start), its construction time charged to each
   budget; SCIP's budget is end to end (model build included) with the default gap 1e-6; the solvers get the cardinality as `≤ K` (the
   problem of E2) and run only on folds whose row is `ok` (those that can be in `F_k`) and only on the arms of `solvers.arms`
   (MV); the support of a solver's portfolio is taken to the same fixed point and looked up among the ranks (`locate_rank`:
   a listed rank, a tie with one, or "beyond the list" = the plan's bin `> 33`). Their parameters are the configurations E2
   tuned for N = 100 (`configs/E2/pilot/tuned.json`, `sizes["100"].roles`; §12, 2026-10-12): SCIP is `exact_rep`
   (perspective, `D = λ_min·I`, `misc/usesymmetry = 0`), tabu, SA and LS are the roles `tabu`, `sa`, `ls` (criterion PI⁺ at
   10 s, rechecked at 60 s). Not the QUBO representative or swap-SA of RQ2 and not the roles of H1c (the final gap at
   10 s). The budgets, SCIP's gap and the common start are E3's own; `tests/test_e3_configs.py` compares the config with
   the report.
7. **Folds outside `F_k`.** Every path holds the optimum's incumbent there (the same portfolio), so the days drop out of the
   comparison alike; only the cost of the first included fold after such a gap depends on that choice. A fold whose cell is missing or
   failed stops the assembly unless `assembly.allow_incomplete` is set; then it is counted as excluded (`n_missing`).
8. **Series.** The Sharpe functional (MV, US-MV) gets daily excess returns (return − the day's risk-free accrual), the log-variance
   functional (GMV) the returns themselves (the variance the arm minimises); "до издержек" is a pass of the engine with `tc = 0`.
9. **Block.** LW Algorithm 3.1 on (optimum, rank 2) in percent, the plan's grid / 1000 / 499, once per market and arm and used for
   H3a, H3b, H3c of that market and arm; the HAC test (QS, prewhitened) of LW on percent returns is the robustness check.
10. **Net metrics.** Sharpe, CE (`γ = λ_f`) of the excess net returns, annual turnover (one-way, summed over `F_k` folds per year of
    `F_k` days) and annual cost, at 0, 10 and 30 bp; differences to the optimum for rank 2 and the solvers with percentile
    intervals of the stationary bootstrap (Politis–White block, PPW correction).
11. **US sensitivities.** "Исключение бумаг с оставшимися днями нерегулярных коэффициентов" is read as an exclusion per fold
    (audit G1, D-3): a ticker leaves the universe of a fold if one of the 102 uncorrected days of `pseudo_split_days.parquet`
    (81 tickers) lies among the returns that fold uses, the 755 of its training window and the 21 of its test window, and the
    exclusion comes before the top 100 are cut; a day outside the fold's windows leaves the ticker in. The reading of the whole
    dataset took the early universes out on spin-offs years ahead. On the calendar of the dataset 9 to 32 tickers (median 20)
    are flagged per fold, against 81 in every fold under that reading. The plan does not fix the scope (per fold or the whole
    dataset); the look as far as the end of the test window makes this a data-cleaning check, not a tradable rule. The
    coverage check keeps the folds whose coverage on the formation date is at least 90 %; the terminal-return passes
    use the engine's `terminal_return` of −0.3 and −1.0.
12. **Operators (б), (а).** Descriptive only: no time limit (steps and chains bound the work), levels from the plan, 30 replicas.
    `L` is read where the plan reads it, on the folds of the PILOT period and per arm ("L по плечам"): the levels reached on at
    least `min_share` (0.8) of the pilot folds the operator ran on (a fold with a certified optimum and rank 2), from the registered
    run of `pilot_operators.yaml` (`sources.pilot_operators`; the denominator is the folds the operator ran on, not all folds, because
    the operator is defined on a certified optimum). `L_a = {g ≤ g*}`; `F_a` and `F_b` are the CONFIRMATORY folds where all levels of
    `L_a` / `L` are reached (empty if the levels are); estimates with intervals of the joint block bootstrap, no TOST, no margin.
    `levels_L` in the operator's config overrides the pilot (a list for both arms, or arm → list); with neither the assembly stops.
    The assembly checks that the pilot is a calibration run at the same `lambda_f`, on the same market, with the same operator
    levels, replicas and limits, and that it ends before the confirmation begins. A short `F_b` cuts the block grid to the blocks that
    leave two copies in the series (recorded); fewer than 10 days give a note and no estimate.
13. **Pilot.** The share of binding folds for the λ rule counts the MV rows with a defined `s_cont`, whatever their certificate;
    `σ*` and `σ²_*` are taken on the days of the pilot's `F_k` (optimum and rank 2 certified in both arms) at `λ_f`; a `δ` outside
    (0, 1) is reported with its reason, not raised; the half-width is that of the 90 % TOST interval with a block calibrated on
    the pilot pair, compared with `Δ_SR` (MV) or with the smaller side of the log margin (GMV).
14. **Seeds.** Solver seed 10000 (pilot) / 20000 (test) goes to SCIP as `randomization/randomseedshift`; everything else is derived
    from it by `SeedSequence`; the statistics have one seed per test and variant derived from `stats.seed`.


## Still open

* `lambda_f`, `Δ_SR`, `δ` — the pilot (plan §3, §2 H3a); they are `null` in the files on purpose.
* `pilot_operators.yaml` runs the operators on both arms (the plan's `L` is per arm); `ff49.yaml` runs them on MV only, so the
  GMV levels of the pilot are read but not used. To run (б), (а) on GMV in the confirmation too, put GMV into
  `ff49.yaml: params.operators.arms` and into `analysis.yaml: sources.ff49.assembly.operators.arms` (about 4 CPU-h more, 8 under
  the heavier load: the swap chains of 636 more cells, their replica paths, the calibrations of the analysis).
* The Russian market (RU) is exploratory and needs the dividend panel (plan §3): not part of this stream yet.
