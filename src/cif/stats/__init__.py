"""Statistical inference for the pre-registered analysis (docs/ANALYSIS_PLAN.md, Sec. 7).

- :mod:`cif.stats.bootstrap` - stationary and circular block bootstrap, Politis-White block
  length, bootstrap intervals;
- :mod:`cif.stats.sharpe` - Sharpe ratio, Jobson-Korkie-Memmel and Ledoit-Wolf (2008) tests,
  block-size calibration, probabilistic and deflated Sharpe ratios;
- :mod:`cif.stats.tests_misc` - Holm adjustment, TOST, performance profiles, TTT plots;
- :mod:`cif.stats.rq3` - RQ3: replica-averaged Sharpe / log-variance contrasts, joint circular
  block bootstrap, H3a (TOST + IUT + Holm, four outcomes), H3b slope, H3c dose (descriptive
  replica operator (b));
- :mod:`cif.stats.rq3_ranks` - RQ3 on rank paths (plan v2.4, the primary operator (g)): H3a on
  rank 2, the H3b slope on log of the mean dose, block length on (optimum, rank 2), H3c solver
  residuals against the H3b curve, rank bins.

All resampling functions require an explicit seed.
"""

from cif.stats.bootstrap import (
    BootstrapResult,
    OptimalBlockLength,
    bootstrap_ci,
    circular_block_bootstrap_indices,
    optimal_block_length,
    stationary_bootstrap_indices,
)
from cif.stats.rq3 import (
    DoseResponse,
    EquivalenceResult,
    JointBootstrap,
    SolverDose,
    calibrate_logvar_block,
    dose_response,
    equivalence_test,
    h3a_outcome,
    joint_contrasts,
    sharpe_margin,
    solver_dose,
    variance_margin,
)
from cif.stats.rq3_ranks import (
    ArmDecision,
    RankBins,
    RankDoseResponse,
    RankEquivalence,
    RankSolverDose,
    SolverResidual,
    SolverResiduals,
    calibrate_rank_block,
    calibrate_replica_block,
    holm_over_arms,
    rank2_outcome,
    rank_bins,
    rank_dose_response,
    rank_equivalence,
    rank_solver_dose,
    solver_residuals,
)
from cif.stats.sharpe import (
    BlockCalibration,
    DeflatedSharpeResult,
    SharpeTestResult,
    calibrate_block_size,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    hac_covariance,
    jobson_korkie_memmel,
    ledoit_wolf_sharpe_test,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from cif.stats.tests_misc import (
    TOSTResult,
    TTTData,
    holm,
    performance_profile,
    performance_ratios,
    tost_bootstrap,
    tost_from_ci,
    ttt_ecdf,
)

__all__ = [
    "ArmDecision",
    "BlockCalibration",
    "BootstrapResult",
    "DeflatedSharpeResult",
    "DoseResponse",
    "EquivalenceResult",
    "JointBootstrap",
    "OptimalBlockLength",
    "RankBins",
    "RankDoseResponse",
    "RankEquivalence",
    "RankSolverDose",
    "SharpeTestResult",
    "SolverDose",
    "SolverResidual",
    "SolverResiduals",
    "TOSTResult",
    "TTTData",
    "bootstrap_ci",
    "calibrate_block_size",
    "calibrate_logvar_block",
    "calibrate_rank_block",
    "calibrate_replica_block",
    "circular_block_bootstrap_indices",
    "deflated_sharpe_ratio",
    "dose_response",
    "equivalence_test",
    "expected_max_sharpe",
    "h3a_outcome",
    "hac_covariance",
    "holm",
    "holm_over_arms",
    "jobson_korkie_memmel",
    "joint_contrasts",
    "ledoit_wolf_sharpe_test",
    "optimal_block_length",
    "performance_profile",
    "performance_ratios",
    "probabilistic_sharpe_ratio",
    "rank2_outcome",
    "rank_bins",
    "rank_dose_response",
    "rank_equivalence",
    "rank_solver_dose",
    "sharpe_margin",
    "sharpe_ratio",
    "solver_dose",
    "solver_residuals",
    "stationary_bootstrap_indices",
    "tost_bootstrap",
    "tost_from_ci",
    "ttt_ecdf",
    "variance_margin",
]
