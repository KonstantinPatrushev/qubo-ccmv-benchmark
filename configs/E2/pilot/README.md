# configs/E2/pilot/ — reports of the pilot

The pilot (ANALYSIS_PLAN §3, §6; values are recorded in §12 before the first confirmatory run) writes
its JSON reports here, one per rule. The stage configs read them as `inputs` (their SHA-256 goes into the
manifest of every run). Commit them, after the values are in §12 and before any confirmatory run: a
confirmatory run needs a clean tree.

| file | written by | read by |
|---|---|---|
| `c4_lambda.json` | `e2_pilot c4-lambda --data-dir data/processed/sp500_pit --out configs/E2/pilot/c4_lambda.json` (rule N7; no registered run stands behind it, so it carries `git_sha` and `git_dirty` of the code that computed it: run it on a clean committed tree, audit G1, L-4) | `calib_boot`, `bench_boot`, `ref_bench`, `cold`, `elig_calibration`, `elig_test` |
| `eligibility_calibration.json` | `e2_pilot eligibility --config configs/E2/elig_calibration.yaml` (§12 A1: the state of the calibration scan after each round; the last one holds the non-trivial calibration instances, the sizes that borrow configurations and the estimate of A3) | `elig_calibration` (the state of the next round), `ref_calibration`, `tune_screen`, `tune_recheck`, `calib_boot`; `e2_pilot tune` / `n-rule` (`--eligibility`) |
| `screen.json` | `e2_pilot screen` (10 s ranking on PI⁺, plan of the recheck) | `tune_recheck` |
| `tuned.json` | `e2_pilot tune` (60 s recheck, 5 % and TTT rules, representatives; a size without enough non-trivial instances has the entry of the nearest size with `borrowed_from`) | every `calib_boot`, `bench_*`, `h1c`, `cold`, `ttt` |
| `n_rule.json` | `e2_pilot n-rule` (n per cell from the SD of log R⁺, log R⁺_b; a descriptive cell gets 20) | `elig_test` (the targets of the scans) |
| `ttt_instances.json` | `e2_pilot ttt-instances` (median difficulty by PI⁺, non-trivial calibration instances) | `ttt` |
| `eligibility_test.json` | `e2_pilot eligibility --config configs/E2/elig_test.yaml` (the state of the test scan of C1–C5 after each round; the last one is the selection every test stage runs on, the pooled-estimate membership of the cells and the estimate of A3) | `elig_test`, `bench_factor`, `bench_boot`, `bench_orlib`, `h1c`, `cold`, `ref_bench`; `e2_analysis h1c` |
| `eligibility_dense.json` | `e2_pilot eligibility --config configs/E2/elig_dense.yaml` (the same for D1, descriptive) | `elig_dense`, `bench_dense` |

`eligibility_calibration.json`, `eligibility_test.json` and `eligibility_dense.json` are committed as **stubs**
(`"stub": true`, no cells): the state of a scan before its first round, which a stage that needs the selected instances
refuses. They are overwritten after every round by `e2_pilot eligibility`; the test ones are produced on test seeds and
belong to the confirmatory stages although they live here with the inputs they feed.

`tuned.json` holds, per size N ∈ {100, 200, 400}, the `roles` the stages read (`exact_rep`, `qubo_rep`, `ls`, `sa`,
`tabu`, `swap_sa`, `h1c_sa`, `h1c_tabu`: method specs), the decisions that led to them (winner, runner-up, scores at
10 s and 60 s, the reason) and the screen they came from. E3 (H3c, configurations tuned for N = 100) reads
`sizes["100"].roles.sa / .tabu / .ls`: if N = 100 had fewer than 8 non-trivial calibration instances those are the
configurations of the nearest N (`borrowed_from`), which E3 should know.

Each report has a `section12` line: a draft of the entry for the deviation log (date, what, rule,
hypotheses, who agreed; the last field is for the orchestrator). Hand-written stand-ins for smoke runs
are in `../smoke/`; they are marked `"smoke": true` and are not results.
