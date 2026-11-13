# configs/E2 — the E2 experiment (RQ2, H1c) on the run server

Everything here implements `docs/ANALYSIS_PLAN.md` v2.4 (frozen) **with the entries A1–A5 of §12 (2026-10-11)**; where
the plan is ambiguous the module docstrings say how it was read (`src/cif/experiments/e2_eligibility.py`,
`e2_benchmark.py`, `e2_pilot.py`, `e2_analysis.py`, `e2_space.py`, `e2_instances.py`, `launch.py`). Hand-written stand-ins
for the pilot reports (smoke rehearsals only) are in `smoke/`; the real reports go to `pilot/`.

## What changed with §12 A1–A5

| entry | in the code |
|---|---|
| **A1** — methods are compared only on **non-trivial** instances, `ĝ_start = (f_start − f*)/s* > τ = 10⁻³`, decided at the reference stage, before any method runs on the instance, and never revised | the stage `eligibility` (`elig_*.yaml`), the rule `e2_eligibility.classify`, the scan in rounds, the report `e2-eligibility`; every later stage reads the selected instances (`eligible:` groups) |
| **A2** — the primary measure is `PI⁺ = PI − t₀/T`, `t₀` (the build of the common start) still charged to the budget | `cif.metrics.anytime.primal_integral_after_start`; `e2_pilot.row_pi` (default `"plus"`): R, R_b, the tuning criterion at 10 s, the n rule and the choice of the TTT instances; no `PI_FLOOR` |
| **A3** — per cell (C1–C5, D1): share of trivial instances with a Wilson CI, `ĝ_start` in bp CE, `t_cert` on the trivial ones | `estimate` in every cell of the eligibility report (`e2_pilot eligibility`) |
| **A4** — the sensitivity is named "γ = 1 until the first improvement of the common start" (`row_pi(variant="first_improvement")`); the honest "until the method's own first solution" comes from cold runs; the confined PI is dropped | `cold.yaml` (stage `cold`), `row_pi(variant="own")`; `first_own_f` next to `first_own_s` in the rows |
| **A5** — H1c on the non-trivial test instances of C1–C3; `\|F_tabu − F_SA\| ≤ 10⁻⁸·s*` are zeros, dropped before ranking | `e2_analysis h1c` |

## The stages

| config | what | cells | budget | seeds | timing stage |
|---|---|---|---|---|---|
| `elig_calibration` | screening of the calibration instances: seeds of the plan, then `factor` from 1024 per N and `boot` from 1110, until 8 (boot: 10) non-trivial, at most 200 per cell; **in rounds** | one block per unfinished cell and round | 2 h limit | calibration | yes (descriptive `t_cert`) |
| `ref_calibration` | certified optima of the selected calibration instances that the screening did not certify (only with `certificate: tau`; empty with the default `full`) | `needs_reference` | 2 h limit | calibration | no |
| `tune_screen` | ≤ 30 random configurations per heuristic, the exact variants, 3 seeds, 10 s, on the 8 non-trivial instances per N | 9 048 for 8 per N | 10 s | calibration | yes |
| `tune_recheck` | winner and runner-up of each group, 60 s (fresh seeds 10003-10005) | by the screen | 60 s | calibration | yes |
| `calib_boot` | representatives on the non-trivial calibration instances of C4 (`boot`, N = 200, λ of the C4 rule) | 3 per instance and seed | 60 s | calibration | yes |
| `elig_test` | screening of the test instances: C1–C4 from seed 2000 until n non-trivial (n by the n rule), at most 400 per cell; C5 the 125 ORlib points in one round; **in rounds** | one block per unfinished cell and round | 2 h limit | **test** | yes |
| `elig_dense` | the same for D1 (descriptive); **see the cost warning in the file** | one block per round | 2 h limit | **test** | yes |
| `bench_factor` | C1–C3, exact (3 seeds) and the four heuristics (10 seeds), on the selected instances | by the selection | 60 s | **test** | yes |
| `bench_boot` | C4 (`boot`, N = 200, λ of the C4 rule) | by the selection | 60 s | **test** | yes |
| `bench_orlib` | C5: the selected binding, non-trivial ORlib points | by the selection | 60 s | **test** | yes |
| `bench_dense` | D1 (descriptive; the three representatives) | by the selection | 60 s | **test** | yes |
| `h1c` | tabu against SA, tuned by the final gap, 10 seeds, on the selected C1–C3 instances | by the selection | 10 s | **test** | yes |
| `cold` | the three representatives without a common start, one seed, on the selected instances of C1–C5 | 3 per instance | 60 s | **test** | yes |
| `ttt` | 100 runs of each representative on 3 instances of median difficulty (N = 200, non-trivial) | 900 | 60 s | calibration | yes |
| `ref_bench` | certified optima / best incumbents of the selected test instances without a certificate (only with `certificate: tau`) | `needs_reference` | 2 h limit | **test** | no |
| `m11` | validation of the exact stack against Tables 3 and 4 of Bertsimas & Cory-Wright (own module `e2_m11`, own config format; below) | 105 (T3: 15 × (3 SCIP + 1 OA), T4: 15 × 3 SCIP) | 300 s (T3), 3600 s (T4) | calibration (fixed instances, no test) | yes |

The pilot (steps below) starts only after the public stamp of the plan (plan header, item 4; placed on 2026-10-11 as an
RFC 3161 time stamp, see §12 of the plan). The confirmatory stages (bold) read test seeds; they run only from a clean
tree, after the pilot values are in §12. Every config carries a `smoke:` block (calibration seeds, 20-asset instances,
tiny budgets) for rehearsals; the runner refuses test seeds in a smoke run. The stand-in reports of `smoke/` are flagged
`"smoke": true`, and a run that is not a smoke run refuses them (a stand-in copied into `pilot/` stops the stage).
The reports `pilot/eligibility_*.json` start as **stubs**: they are the state of a scan before its first round, and every
stage that needs the selected instances refuses a stub, an unfinished scan or a cell the report lacks.

## The common start, PI⁺ and the eligibility rule (corrects the earlier text of this file)

The common start (top-K rounding of the relaxation, refined) is handed to every method; the time `t₀` of building it is
charged to every method's budget, and `γ(t) = 1` on `[0, t₀)` for all of them. On the 24 calibration instances of
`20261011-211545-E2-ref_calibration-936867` (all certified) the start is optimal on 12 and within τ on 21
(87.5 %, Wilson 69–96 %). Two things follow, and the earlier text of this file (and the docstring of `e2_pilot`) had both
wrong:

* **The primal integral of such an instance is not 0.** `PI(T)` counts the construction of the start: if the start is
  optimal, `PI = t₀/T` for every method (2·10⁻⁴ to 3·10⁻³ at T = 60 s), and `R = PI(QUBO)/PI(exact)` is the ratio of two
  measurements of `t₀` — about 1 up to the noise of the clock, not 0/0. The ratio of any two PIs is bounded by
  `1 + ĝ_start·(T − t₀)/t₀`: at N = 400 it cannot leave `[1/1.5, 1.5]` while `ĝ_start < 1.5·10⁻³`, whatever the methods do.
  The floor `PI_FLOOR = 10⁻⁶` lay below `t₀/T` and never acted.
* **The rows do not contain what "γ = 1 until the first own solution" needs.** SCIP with `addSol` reports only improvements
  over the start, local search starts from the start, the QUBO pipeline holds it in its pool, and a row keeps only the
  improvements over the start. What `row_pi(with_start=False)` computed was "γ = 1 until the first *improvement* of the
  common start" (on an instance whose start is optimal, 1 for every method); it is now called that
  (`variant="first_improvement"`). The honest variant needs cold runs (`cold.yaml`).

Hence (§12): the primary measure is `PI⁺(T) = (1/T)∫_{t₀}^{T} γ dt = PI(T) − t₀/T`; on an instance where the start is
optimal `PI⁺ = 0` for everyone (0/0), so those instances are left out by the eligibility rule instead of being floored:
`n_rule_report` refuses a PI⁺ of 0 and names the instance. On a non-trivial instance `PI⁺ > 0` for every method
(`γ(t₀) = ĝ_start > τ`). The filter is a property of the instance — a solution better than the start by more than `τ·s`
has been found (non-trivial), or a dual bound proves `f_start − f* ≤ τ·s` (trivial), or neither within 2 h (undecided,
excluded and counted) — and does not look at any method. `s` is the scale of the common start: the scale of the optimum
is unknown at that point (the same stand-in as for the `limits/absgap` of §5); `reference_check` lists the instances
that the exact `s*` of a later certificate would put on the other side of τ and never revises them.

## Prerequisites on the server

* the environment of `environment.lock.yml` (micromamba on the server, Python 3.11; `scripts/run_server.sh` sets
  `PYTHONNOUSERSITE=1`, as the plan's §12 says), one thread per solver;
* a bundle of the committed tree: `git archive HEAD src configs scripts pyproject.toml environment.yml environment.lock.yml`
  plus `BUNDLE_SHA` (the recipe of `.claude/skills/hpc-job`, without the scheduler), and the data the
  stages read, at the same relative paths: `data/raw/orlib/` (ORlib files and `provenance.json`, 0.9 MB) for
  `elig_test`, `bench_orlib`, `cold`, `ref_bench`; `data/processed/sp500_pit/` (45 MB) **and `data/raw/sp500_pit/` (68 MB,
  the Yahoo cache and the snapshots its provenance records)** for `elig_calibration`, `calib_boot`, `elig_test`,
  `bench_boot`, `cold`, `ref_bench`. The runner re-hashes every input against its provenance before it starts (a missing
  file stops the job with exit code 2);
* the server is **shared** (gpu-07: vLLM on the GPU, other users; plan §12 of 2026-10-11): there is no exclusive
  mode, and the plan's limits stand in for it. **At most 16 jobs** (half of the 32 physical cores) run, each bound to a
  physical core (`--cores 0-31 --pin`); the load average and the governor go into the manifests; before pinned jobs start
  the launcher watches the cores for a second and names those that other work keeps busy (leave them out with
  `--exclude-cores`). Interference during a run shows afterwards in the load samples and in every row
  (`method_cpu_s` against `method_wall_s`). The launcher takes a lock (`/tmp/cif_launch.lock`, or `$CIF_LAUNCH_LOCK`):
  one launcher of ours at a time;
* **output directories outside the repository**: the launcher writes its shards, logs and manifest to
  `runs/incoming/E2-<name>` by default, which is inside the working tree and would make it dirty (a confirmatory stage
  needs a clean tree). Always pass `--out-dir` with a directory outside the repository and the bundle, e.g.
  `OUT=$HOME/cif_e2_out`, one directory per stage and per round (a directory is not resumed with another grid).

The bundle (laptop; the user copies it to the server and unpacks it; the agents do not use `scp` / `ssh`):

```bash
git status --porcelain            # must be empty: the bundle is the committed tree
SHA=$(git rev-parse --short HEAD); B=e2_bundle_$SHA
mkdir -p $B && git archive HEAD src configs scripts pyproject.toml environment.yml environment.lock.yml | tar -x -C $B
mkdir -p $B/data/raw $B/data/processed
cp -r data/raw/orlib data/raw/sp500_pit $B/data/raw/ && cp -r data/processed/sp500_pit $B/data/processed/
echo $SHA > $B/BUNDLE_SHA && tar czf $B.tgz $B
# on the server: tar xzf $B.tgz && cd $B && micromamba activate cif   (then the commands below, from the bundle root;
# the environment is created once from environment.lock.yml: micromamba create -f environment.lock.yml)
```

## The launcher

`scripts/run_server.sh <config> [options]` = `python -m cif.experiments.launch run --config <config> [options]`
with one thread per job. The server (gpu-07) has 2 × Xeon Silver 4216 (32 physical cores, 64 threads) and the
governor `schedutil`, which cannot be changed; both go into the manifests.

```bash
OUT=$HOME/cif_e2_out                                                  # outside the repository and the bundle
scripts/run_server.sh configs/E2/elig_calibration.yaml --cores 0-31 --pin --smoke --out-dir $OUT/smoke   # rehearsal
python -m cif.experiments.launch plan --config configs/E2/elig_calibration.yaml --cores 0-31 --pin
scripts/run_server.sh configs/E2/elig_calibration.yaml --cores 0-31 --pin --out-dir $OUT/elig_calibration_r1
python -m cif.experiments.launch status --config configs/E2/elig_calibration.yaml --out-dir $OUT/elig_calibration_r1
```

* `--jobs` (default `launch.jobs` of the config: 16 in every stage, the cap of the plan's §12 for the shared server)
  and `--shards` (`launch.shards`) can be given on the command line; the config itself stays untouched.
  A timing stage is refused above half of the physical cores; `--allow-oversubscribe` lifts that. The launcher does
  not hold the two certification stages (`ref_calibration`, `ref_bench`; no timings are taken) to the limit, so
  `--jobs 24` works for them, but that is a decision against the 16 of §12 and the server's other users.
* `--pin` binds job slot `j` to one physical core of the pool given by `--cores` (spread over the two
  sockets when the topology is readable), by `taskset -c`. `--cores` is the list of the *first* thread of each
  physical core (0-31 on the target); if the machine reports some listed CPUs as hyperthread siblings the launcher
  warns, and it refuses to pin two jobs to the two threads of one core.
* Busy cores (the server is shared). `plan` and `run` with `--pin` sample the CPU counters for a second and warn
  about every core of the jobs whose threads (the core and its hyperthread siblings) were busy a quarter of that
  time or more; the warning names the cores, goes into the manifest, and says what to do:
  `--exclude-cores 3,17` takes those cores out of the pool (the jobs go to other cores; "half of the physical
  cores" still counts all of `--cores`, so 16 jobs stay the limit). Going on in spite of the warning is the
  operator's call; the manifest keeps it.
* Shards (`cells[i::S]`, at most one per cell) are handed to free job slots one by one, so a slow shard does not hold the
  rest back. Results go to `<out-dir>/<name>_shard<i>.jsonl`, logs to `logs/`, and the launcher writes
  `launch_manifest.json` (host, CPU model, governors, load average at the start, every minute and at the end, cores of the
  jobs, shards started / skipped / failed) and `BUNDLE_SHA`.
* **Resume = the same command.** A finished shard is skipped; an interrupted one continues after its last row
  (the runner skips cells with a row without an error; a cell with an error is tried again). The number of
  shards and the config are fixed in `launch_state.json` for the directory: a cell must stay in its shard,
  otherwise it would be computed twice and `ingest` would see two different timings of one cell.
* `plan` prints a digest of the grid (`grid <12 hex>`, also in the manifest). The grid depends on the input reports and
  on the platform (generated instances, sampled configurations): run `plan` on the laptop and on the server and
  compare the digests before a long run; a directory is not resumed with another digest.
* Ctrl-C / SIGTERM / SIGHUP stop the jobs; the rows written so far are kept.

## Order of runs

The reports of the pilot go to `pilot/` (see `pilot/README.md`); `<run>` is a registered run directory. All commands from
the repository root (laptop) unless they say `server`.

```bash
# 0. C4 lambda rule: done, configs/E2/pilot/c4_lambda.json (no admissible lambda: C4 is descriptive, run at lam_used = 20).
#    No registered run stands behind it: the report carries git_sha / git_dirty of the code (audit G1, L-4), so it is
#    made on a clean committed tree:
#    python -m cif.experiments.e2_pilot c4-lambda --data-dir data/processed/sp500_pit --out configs/E2/pilot/c4_lambda.json

# 1. SCREENING OF THE CALIBRATION INSTANCES (§12 A1), in rounds
# 1a. the state after "round 0": the 24 plan instances were certified before any method ran (registered run)
python -m cif.experiments.e2_pilot eligibility --config configs/E2/elig_calibration.yaml \
    --reference runs/20261011-211545-E2-ref_calibration-936867 --c4-lambda configs/E2/pilot/c4_lambda.json \
    --out configs/E2/pilot/eligibility_calibration.json
git add configs/E2/pilot/eligibility_calibration.json && git commit -m "E2: состояние отбора, раунд 0"
# 1b. a round (R = 1, 2, …): bundle (only the state file changes between rounds: copy it into the bundle on the server
#     and compare the `plan` digest on both machines), run, copy $OUT/elig_calibration_r$R back, register, update the state
OUT=$HOME/cif_e2_out; R=1
scripts/run_server.sh configs/E2/elig_calibration.yaml --cores 0-31 --pin --out-dir $OUT/elig_calibration_r$R   # server
d=$OUT/elig_calibration_r$R
python -m cif.experiments.ingest --config configs/E2/elig_calibration.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log      # BEFORE the state file is replaced
python -m cif.experiments.e2_pilot eligibility --config configs/E2/elig_calibration.yaml \
    --rows <elig_calibration run of round 1> [<round 2> …] --reference runs/20261011-211545-E2-ref_calibration-936867 \
    --c4-lambda configs/E2/pilot/c4_lambda.json --out configs/E2/pilot/eligibility_calibration.json
#     it prints per cell "k of n non-trivial in m scanned … done / next round" and "scan finished" at the end; commit the
#     state and the registered runs after every round (the next ingest needs a clean tree); repeat 1b until "scan finished"

# 2. only with `certificate: tau`: references of the selected calibration instances (server), register
scripts/run_server.sh configs/E2/ref_calibration.yaml --cores 0-31 --pin --out-dir $OUT/ref_calibration

# 3. tuning: the 10 s screen on the selected instances (server), register, ranking at 10 s and the plan of the recheck
scripts/run_server.sh configs/E2/tune_screen.yaml --cores 0-31 --pin --out-dir $OUT/tune_screen
#    --reference: every run with certified optima of the calibration instances: the registered 20261011-…-936867, the
#    rounds of elig_calibration (their rows are references with `certificate: full`) and ref_calibration if it was run
python -m cif.experiments.e2_pilot screen --rows <screen run> \
    --reference runs/20261011-211545-E2-ref_calibration-936867 <elig_calibration runs> [<ref_calibration run>] \
    --budget 10 --out configs/E2/pilot/screen.json
# recheck at 60 s (server), register it, then the tuned configurations (a size with < 8 non-trivial instances takes
# those of the nearest N, from the eligibility report)
scripts/run_server.sh configs/E2/tune_recheck.yaml --cores 0-31 --pin --out-dir $OUT/tune_recheck
python -m cif.experiments.e2_pilot tune --screen <screen run> --recheck <recheck run> \
    --reference runs/20261011-211545-E2-ref_calibration-936867 <elig_calibration runs> [<ref_calibration run>] \
    --eligibility configs/E2/pilot/eligibility_calibration.json --budget 10 --budget-long 60 \
    --out configs/E2/pilot/tuned.json

# 4. representatives on the non-trivial calibration instances of C4 (server), register
scripts/run_server.sh configs/E2/calib_boot.yaml --cores 0-31 --pin --out-dir $OUT/calib_boot

# 5. n per cell (PI⁺; a descriptive cell gets n = 20) and the TTT instances
python -m cif.experiments.e2_pilot n-rule --rows <recheck run> <calib_boot run> <the same reference runs as above> \
    --tuned configs/E2/pilot/tuned.json --eligibility configs/E2/pilot/eligibility_calibration.json \
    --cell C1=T100:100 --cell C2=T200:200 --cell C3=T400:400 --cell C4=CB200:200 --out configs/E2/pilot/n_rule.json
python -m cif.experiments.e2_pilot ttt-instances --rows <recheck run> <the same reference runs> \
    --tuned configs/E2/pilot/tuned.json --out configs/E2/pilot/ttt_instances.json

# 6. copy the `section12` lines of the reports (eligibility_calibration, screen, tuned, n_rule, ttt_instances) into §12
#    of the plan, commit configs/E2/pilot/*.json; the plan carries its stamp, the tree is clean

# 7. SCREENING OF THE TEST INSTANCES (confirmatory, server, clean tree), in rounds as in step 1b; the targets come from
#    n_rule.json, C4 enters the pooled estimate only with an admissible lambda
#    (round 1 starts from the committed stub eligibility_test.json; R = 1, 2, …)
scripts/run_server.sh configs/E2/elig_test.yaml --cores 0-31 --pin --out-dir $OUT/elig_test_r$R           # server
d=$OUT/elig_test_r$R
python -m cif.experiments.ingest --config configs/E2/elig_test.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log      # BEFORE the state file is replaced
python -m cif.experiments.e2_pilot eligibility --config configs/E2/elig_test.yaml --n-rule configs/E2/pilot/n_rule.json \
    --c4-lambda configs/E2/pilot/c4_lambda.json --rows <elig_test runs so far> --out configs/E2/pilot/eligibility_test.json
#    D1 only after the decision on its cost (see the warning in elig_dense.yaml): same with elig_dense.yaml and
#    configs/E2/pilot/eligibility_dense.json (no --n-rule, no --c4-lambda)

# 8. the confirmatory stages and the rest (server, clean tree), one launcher at a time, each registered with ingest:
for c in bench_factor bench_boot bench_orlib h1c cold ttt; do     # + bench_dense if D1 was screened; + ref_bench with `tau`
    scripts/run_server.sh configs/E2/$c.yaml --cores 0-31 --pin --out-dir $OUT/$c
done

# 8b. the validation of the exact stack (M11, §12 entry of 2026-10-17): independent of the pilot reports; needs the bundle with
#     data/raw/orlib (and `m11_bcw_published.yaml`, which is in configs/). 105 cells, one per shard; the cells differ by
#     orders of magnitude (T3 in milliseconds to minutes, T4 up to the 3600 s limit), so shards are single cells.
scripts/run_server.sh configs/E2/m11.yaml --cores 0-31 --pin --out-dir $OUT/m11
d=$OUT/m11
python -m cif.experiments.ingest --config configs/E2/m11.yaml --shards $d/*_shard*.jsonl \
    --bundle-sha-file $d/BUNDLE_SHA --logs $d/launch_manifest.json $d/logs/*.log
python -m cif.experiments.e2_analysis m11 --rows <m11 run> --published configs/E2/m11_bcw_published.yaml \
    --out results/E2/analysis/m11          # m11.json, cells.csv, summary.csv, rows.csv

# 9. the analysis: H1c (A5)
python -m cif.experiments.e2_analysis h1c --rows <h1c run> --eligibility configs/E2/pilot/eligibility_test.json \
    --reference <screening runs of C1–C3> --out <where the analysis writes>/h1c.json
#    H2 and H2b, the confirmatory estimate of RQ2 (mean log R over the pooled cells, 95 % stratified bootstrap, the five
#    outcomes, the cells with Bonferroni 99 % intervals, the estimate of A3, the sensitivities of A4). The code must be
#    committed BEFORE the first bench row is read (audit G1, RQ2-1). --reference: every round of elig_test (their rows
#    hold the certificates), of elig_dense if D1 was screened, and ref_bench if it was run; --cold: the run of the
#    stage cold (optional). D1: add its bench run to --rows and --eligibility-dense
python -m cif.experiments.e2_analysis h2 --rows <bench_factor run> <bench_orlib run> [<bench_boot run>] \
    [<bench_dense run>] --eligibility configs/E2/pilot/eligibility_test.json \
    [--eligibility-dense configs/E2/pilot/eligibility_dense.json] --reference <elig_test runs> [<elig_dense runs>] \
    [<ref_bench run>] [--cold <cold run>] --out results/E2/analysis/h2
#    The secondary analyses of RQ2 (descriptive, NOT confirmatory, every output flagged secondary): the pooled ratio of the
#    Kaplan-Meier medians of the time to the target (QUBO against the exact method and against local search, with the
#    sensitivities to the target 1e-4 and to the censoring budget), t_primal and t_cert of the exact class by cell,
#    Dolan-More profiles on PI+ and on the time to the target. The inputs and the reference are those of h2 (no --cold).
#    The code is committed BEFORE the first bench row is read, like h2; the readings of the open points are in
#    PRESPECIFIED of cif.experiments.e2_secondary and in the JSON it writes
python -m cif.experiments.e2_analysis secondary --rows <bench_factor run> <bench_orlib run> [<bench_boot run>] \
    [<bench_dense run>] --eligibility configs/E2/pilot/eligibility_test.json \
    [--eligibility-dense configs/E2/pilot/eligibility_dense.json] --reference <elig_test runs> [<elig_dense runs>] \
    [<ref_bench run>] --out results/E2/analysis/secondary
```

* Run the launcher in `tmux` / `screen` / `nohup`: a dropped ssh connection (SIGHUP) stops the jobs, as Ctrl-C does
  (rows are kept; start the same command again).
* A confirmatory stage (test seeds) refuses to start without `BUNDLE_SHA` (file in the repository root, or
  `CIF_BUNDLE_SHA`): its rows must be attributable to a code bundle.
* If a calibration size has fewer than 8 non-trivial instances after the cap of 200 seeds, it is descriptive and takes
  the configurations of the nearest N (the tuning stages give it no cells); a test cell with fewer than 20 is
  descriptive and leaves the pooled estimates of H2 and H2b; if no cell is in them, §10 says "Нетривиальных инстансов
  недостаточно для сводной оценки; RQ2 отвечает долей тривиальных" (the report's `summary.statement`).
* C4 is descriptive anyway (the λ rule): it is screened and run at `lam_used` and stays out of the pooled estimate.
* `ingest` takes the shards of a stage as one run; `--logs` copies the launch manifest and the job logs into it.

## The validation M11 (Tables 3 and 4 of Bertsimas & Cory-Wright)

`m11.yaml` (module `cif.experiments.e2_m11`, docstring) validates the exact stack on a standard problem, §12 entry of 2026-10-17:
problem (34) of the paper on OR-Library port1–5 (`cif.instances.bcw`), `k ∈ {5, 10, 20}`, **T3** `κ = 1`, no other constraint,
300 s; **T4** `κ = 0` and `μᵀx ≥ r̄` (`r̄ = r_min + 0.3 (r_max − r_min)`, stored in every T4 row), 3600 s. Our stack: SCIP
(`scip-persp-mineig-sym0`, 3 permutation seeds, the median is the value of a cell; with the return row in T4 — the one new
optional row `linear_ge` of `solve_miqp_native`) and OA with HiGHS (`oa-highs`, T3 only: it has no linear rows); specs read from
`tuned.json`, no warm start, one thread. Stop at relative gap 1e-4 **or** absolute gap 1e-6 (the CPLEX defaults of the paper): where
the optimum is small (T3: |f*| ≈ 1e-3) the absolute gap is what stops the solver, at relative gaps up to ~1e-3. The time is the
wall clock of the call (build + solve, `budget_s` = the limit); `certified` = the gap closed within the limit. `seed_role:
calibration`: the instances are fixed files and no hypothesis is tested, so the test streams stay unspent.

The published times are data (`m11_bcw_published.yaml`, an `inputs` entry: hashed into the manifest). The analysis
(`e2_analysis m11`, `cif.experiments.e2_m11_analysis`) gives per instance the ratio of our median time to the best published one
(minimum over the methods of the table), with times beyond the limit set to it and flagged as bounds and the cells that both
sides failed left out and listed; per table the median ratio, the number of cells solved within the limit by us and by the best
published method, and the verdict of the rule of the plan (median above 10). **The verdict reads SCIP's times** (view `scip`:
at the sizes of the ports SCIP is the exact representative of RQ2); OA (T3) and the better of our methods (`stack`) are given
beside it. This reading, which the plan leaves open, is fixed in `PRESPECIFIED` of the analysis and in §12 (entry of
2026-10-18, before the run). The hardware of the paper (MacBook Pro i9, CPLEX 12.8, Mosek 9, Julia) is not
normalised: raw ratios. Their times are printed to 0.01 s, so a ratio against 0.01 s is coarse.

## Cost (estimates, not measurements of the real stages)

Basis: the full certificates of the 24 registered calibration instances (gpu-07), `t_cert` including the start's build:
mean 10.4 s (median 3.5, max 50) at N = 100, 146.6 s (46, 761) at N = 200, 500.3 s (194, 2425) at N = 400; 1.46 CPU-h in
all. On trivial instances the screening at the level `tau` was not cheaper than the full certificate in the few
measurements made for this stage (laptop under load, 5 instances of N = 100: 3.4/3.3, 3.9/4.2, 5.2/4.6 s and two with
identical node counts; one instance of N = 200: 19.6 s against 25.7 s, same 9 nodes; nothing at N = 400): the tree has to
be explored nearly to the end before the bound is within τ. So the default is `certificate: full`, one run per scanned
instance, which also gives `f*` and the exact `ĝ_start`.

| stage | what | CPU-h (**estimate**) | wall-h at 16 jobs |
|---|---|---|---|
| `elig_calibration`, round 1 | from the registered state: 100 × N = 100, 44 × N = 200, 25 × N = 400, 25 boot (priced as N = 200; not measured) | 6.6 (0.3 + 1.8 + 3.5 + 1.0) | 0.4–1.1 (the longest certificate seen took 0.67 h) |
| `elig_calibration`, all rounds | to 8 (boot 10) non-trivial per cell, at most 200 seeds each; with the share of non-trivial 12 % (3 of 24) | 7–20 | 0.5–1.5 |
| `elig_test`, C1–C4 | `n/p` seeds per cell, p = share of non-trivial, n = 20–40, at most 400: p = 25 %, n = 20: 18; p = 25 %, n = 40: 36; p = 10 %, n = 20: 45; p = 10 %, n = 40 (cap 400): 89 | **18–89**, mostly C3 (N = 400: 11–56) | 1–6 |
| `elig_test`, C5 | the ~70 binding of 125 points, one round; not measured (ORlib N = 31–225, K = 10) | ≤ 5 if a point costs like N = 100–200, up to 140 at the 2 h limit | ≤ 9 |
| `elig_dense` (D1) | **unknown and possibly very large**: SCIP decides nothing on dense instances in 30 s; each undecided one costs 2 CPU-h | decision needed (hundreds if most are undecided) | — |
| `ref_calibration`, `ref_bench` | empty with `certificate: full`; with `tau`: the full certificate of the selected instances | as the table of the pilot below | |

Timing stages below, with the selection as input (cells = CPU-seconds / 3600 at the budget; an upper bound: SCIP at N = 100
often proves optimality before the budget, an SDP variant overruns the 10 s budget by 8 s at N = 100, LS with few restarts
stops early). Wall-clock = CPU-hours / 16 jobs. The n rule gives 20–40 instances per cell, i.e. 60–120 for C1–C3.

| stage | cells | CPU-h (upper) | wall-h, 16 jobs |
|---|---|---|---|
| `tune_screen` | 9 048 (8 per N) | 26 | 1.6 |
| `tune_recheck` | 580–1 000 | 10–17 | 0.6–1.1 |
| `calib_boot` | 3 per instance and seed, 10 instances | 1.5 | 0.1 |
| `bench_factor` | 2 580–5 160 | 44–87 | 2.7–5.5 |
| `bench_boot` | 860–1 720 | 15–29 | 0.9–1.8 |
| `bench_orlib` | (selected points) × 43; ≤ 70 points → ≤ 3 010 cells | ≤ 51 | ≤ 3.2 |
| `bench_dense` | 920 if 20 + 20 are found | 16 | 1.0 |
| `h1c` | 1 200–2 400 | 3–7 | 0.2–0.4 |
| `cold` | (selected instances of C1–C5, ~90–230) × 3 | 4.5–11.5 | 0.3–0.7 |
| `ttt` | 900 | 15 | 1.0 |
| `m11` | 105 | ≤ 50 (an upper bound: every cell at its limit, 60 × 300 s + 45 × 3600 s; informal laptop probes of a few T3 and T4 cells took seconds to tens of seconds, `docs/streams/E2.md`) | ≤ 3.1 |

The timing stages together: about 180–270 CPU-h (11–17 h of wall-clock at 16 jobs), plus the screening above. The references
are the large uncertainty (the plan's own estimate for all of E2 is 250–550 CPU-h, pilots 30–60 on top); the consult's
rough figure for the screening, up to 40 CPU-h beyond the plan, is within the range above for p ≥ 25 %.

The statistics are not in the stages. The H1c test and the confirmatory estimate of H2 and H2b (the pooled `log R` with
the stratified bootstrap, the five outcomes, the estimate of A3, the sensitivities of A4: `e2_analysis h2`, the
statistics in `cif.stats.rq2`; every reading the plan leaves open is in `PRESPECIFIED` of `e2_analysis` and in the JSON
it writes) read the registered rows. So do the secondary analyses of RQ2 (`e2_analysis secondary`, module
`cif.experiments.e2_secondary`; descriptive and not confirmatory, every output flagged secondary): the time to the target
(the end-to-end time of the incumbent to ĝ ≤ τ, τ = 10⁻³, sensitivity 10⁻⁴; the Kaplan–Meier median over the seeds per
representative, a median beyond the budget censored at the budget and never dropped; the ratios QUBO/exact and QUBO/LS
pooled like H2; the sensitivity to the censoring budget by re-censoring the trajectories at 30 s and 10 s), `t_primal`
and `t_cert` of the exact class by cell, Dolan–Moré profiles on PI⁺ and on the time to the target. Every reading the plan
leaves open is in its `PRESPECIFIED` and in the JSON. Not written: the TTT curves of the stage `ttt` (Aiex et al., 100
runs on 3 calibration instances; the empirical distribution function `ttt_ecdf` of `cif.stats.tests_misc` is there),
which read other rows. Each row has what the analyses need (see the `e2_benchmark` docstring): the
trajectory of the method's improvements over the common start, the start (`start_time_s`, `start_f`), `f_final`,
`scale_final`, `f_at_budget`, `first_own_s` and `first_own_f`, `t_primal_s`, `t_cert_s`, `certified`, the build / solve
/ refine times, the seeds, the configuration, `card_mode`, the budget compliance, and the instance diagnostics. The
measures of a row against the reference of its instance are in `cif.experiments.e2_pilot` (`reference_table`, `row_pi`
with its variants, `row_gap_at_budget`, `row_ttt`).

## For Methods (disclosures)

* **Tabu reads per batch.** `num_reads_b = 10` for the tabu pipeline in every configuration of E2: it is not in the tuning
  space (the plan lists only the restarts per batch, Appendix B) and is a constant of this implementation
  (`e2_space._draw`, `tests/test_e2_space.py`). The exploratory configuration behind H1c had 200 reads, so the confirmatory
  H1c repeats the direction of that finding, not its configuration, and the row of §10 "H1c подтверждена" must not say
  otherwise.
* **Scale of the rule.** The class of an instance uses the scale of the common start (the scale of the optimum is not known
  when it is decided); the exact `ĝ_start` of the certified instances is in the rows, and `reference_check` reports the
  instances that would flip.
* **The share of trivial instances** is that of the instances scanned until the n-th non-trivial one (the plan's
  definition, Wilson interval); this inverse sampling over-represents the non-trivial instances slightly. Undecided
  instances stay in the denominator.
* **Cold runs** are single-seed; LS has no first *solution* in the cold run other than its trajectory's first point
  (`first_own_f` of LS is the value of its first refinement, a one-asset step of the greedy construction).
