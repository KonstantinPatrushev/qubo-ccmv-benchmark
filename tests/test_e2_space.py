"""Tuning spaces of E2 (ANALYSIS_PLAN §6 "Настройка", Appendix B): the exact variants and the random
search of up to 30 configurations per heuristic.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from cif.experiments import e2_space as space

PENALTIES = {1, 3, 10, 30, 100}
M_B = {20, 50, 200}


# --------------------------------------------------------------------------------------
# The exact class
# --------------------------------------------------------------------------------------


def test_exact_candidates_are_the_scip_variants_times_symmetry_and_oa():
    cands = space.exact_candidates(100)
    ids = [c["id"] for c in cands]
    assert len(ids) == len(set(ids)) == 7
    scip = [c for c in cands if c["name"] == "scip"]
    assert Counter((c["formulation"], c.get("diag")) for c in scip) == {
        ("bigm", None): 2, ("perspective", "mineig"): 2, ("perspective", "sdp"): 2}
    # misc/usesymmetry is a tuning parameter of every SCIP variant (Appendix B)
    assert {c["usesymmetry"] for c in scip} == {"default", 0}
    assert [c["name"] for c in cands].count("oa") == 1
    assert next(c for c in cands if c["name"] == "oa")["diag"] == "mineig"


@pytest.mark.parametrize(("n", "with_sdp"), [(31, True), (100, True), (150, True), (151, False), (400, False)])
def test_the_sdp_variant_is_a_candidate_only_up_to_150_assets(n, with_sdp):
    cands = space.exact_candidates(n)
    assert any(c.get("diag") == "sdp" for c in cands) is with_sdp
    assert len(cands) == (7 if with_sdp else 5)


def test_the_sdp_limit_is_that_of_the_scip_model():
    from cif.classical import scip_native

    assert space.SDP_MAX_N == scip_native._SDP_MAX_N == 150


def test_exact_candidates_are_complete_and_json_safe():
    for spec in space.exact_candidates(100):
        assert space.check_method(spec) == spec
        assert json.loads(json.dumps(spec)) == spec


# --------------------------------------------------------------------------------------
# Random search
# --------------------------------------------------------------------------------------


def test_sample_configs_is_reproducible_and_depends_on_every_argument():
    a = space.sample_configs("sa", 30, 7, 100)
    assert a == space.sample_configs("sa", 30, 7, 100)
    assert a != space.sample_configs("sa", 30, 8, 100)  # the tuning seed
    assert a != space.sample_configs("sa", 30, 7, 200)  # tuned separately for every N
    # the first k configurations of a longer search are those of the shorter one
    assert space.sample_configs("sa", 10, 7, 100) == a[:10]
    # the heuristics draw from independent streams
    assert space.sample_configs("tabu", 30, 7, 100)[0] != space.sample_configs("sa", 30, 7, 100)[0]


def test_at_most_thirty_configurations_per_heuristic():
    assert len(space.sample_configs("sa", 30, 0, 100)) == 30
    with pytest.raises(ValueError, match="30"):
        space.sample_configs("sa", 31, 0, 100)
    with pytest.raises(ValueError, match="at least 1"):
        space.sample_configs("sa", 0, 0, 100)


def test_ids_are_stable_and_unique():
    configs = space.sample_configs("tabu", 30, 3, 100)
    assert [c["id"] for c in configs] == [f"tabu-{i:02d}" for i in range(30)]
    assert all(c["name"] == "tabu" for c in configs)


def test_sa_space_follows_appendix_b():
    configs = space.sample_configs("sa", 30, 11, 200)
    for c in configs:
        assert 10 <= c["num_reads_b"] <= 1000 and 100 <= c["num_sweeps"] <= 10_000
        assert c["beta"] in ("default", "objective")
        assert c["penalty"] in PENALTIES and c["m_b"] in M_B
        assert isinstance(c["num_reads_b"], int) and isinstance(c["num_sweeps"], int)
    # over 30 draws the categorical factors take several values
    assert len({c["penalty"] for c in configs}) >= 3 and {c["beta"] for c in configs} == {"default", "objective"}


def test_tabu_space_follows_appendix_b():
    configs = space.sample_configs("tabu", 30, 11, 100)
    for c in configs:
        assert 1 <= c["num_restarts_b"] <= 100
        assert 5 <= c["tenure"] <= 50  # [5, N/2]
        assert c["penalty"] in PENALTIES and c["m_b"] in M_B
        assert c["num_reads_b"] == 10  # not tuned (the plan lists only the restarts per batch)


def test_tabu_tenure_range_follows_n_and_stays_valid_for_small_n():
    assert max(c["tenure"] for c in space.sample_configs("tabu", 30, 1, 400)) <= 200
    assert all(c["tenure"] >= 5 for c in space.sample_configs("tabu", 30, 1, 8))  # N/2 < 5: the floor


def test_swap_sa_space_follows_appendix_b():
    configs = space.sample_configs("swap_sa", 30, 11, 100)
    for c in configs:
        assert 10 <= c["num_reads_b"] <= 500
        assert 10 <= c["num_sweeps"] <= 1000  # chain length 10·N … 1000·N proposals (a sweep is N)
        assert c["m_b"] in M_B
        assert "penalty" not in c  # works on the K-manifold: there is no penalty


def test_ls_space_follows_appendix_b():
    configs = space.sample_configs("ls", 30, 11, 100)
    for c in configs:
        assert 1 <= c["num_restarts"] <= 200
        assert c["improvement"] in ("first", "best")
        assert c["rcl_size"] == 3  # fixed: the plan does not tune the restricted candidate list
    assert {c["improvement"] for c in configs} == {"first", "best"}


def test_integer_ranges_are_sampled_log_uniformly():
    """Ranges that span two decades (reads, sweeps) are drawn on the log scale."""
    configs = space.sample_configs("sa", 30, 5, 100)
    low = sum(c["num_sweeps"] < 1000 for c in configs)  # half of the log range [100, 10000]
    assert 6 <= low <= 24


def test_every_sampled_configuration_passes_the_check():
    for name in space.HEURISTICS:
        for spec in space.sample_configs(name, 30, 2, 100):
            assert space.check_method(spec) == spec
            assert json.loads(json.dumps(spec)) == spec


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        space.sample_configs("qaoa", 5, 0, 100)


# --------------------------------------------------------------------------------------
# check_method
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"name": "sa", "id": "x"}, "num_reads_b"),
        ({"name": "nope", "id": "x"}, "method"),
        ({"id": "x"}, "name"),
        ({"name": "sa", "num_reads_b": 10, "num_sweeps": 100, "beta": "default", "penalty": 10,
          "m_b": 50}, "id"),
        ({"name": "sa", "id": "x", "num_reads_b": 0, "num_sweeps": 100, "beta": "default",
          "penalty": 10, "m_b": 50}, "num_reads_b"),
        ({"name": "sa", "id": "x", "num_reads_b": 10, "num_sweeps": 100, "beta": "wide",
          "penalty": 10, "m_b": 50}, "beta"),
        ({"name": "ls", "id": "x", "num_restarts": 5, "improvement": "sideways", "rcl_size": 3},
         "improvement"),
        ({"name": "scip", "id": "x", "formulation": "bigm", "usesymmetry": 1}, "usesymmetry"),
        ({"name": "scip", "id": "x", "formulation": "perspective", "diag": "magic",
          "usesymmetry": "default"}, "diag"),
        ({"name": "scip", "id": "x", "formulation": "perspective", "diag": "mineig", "usesymmetry": 0,
          "heuristics": "fast"}, "heuristics"),
        ({"name": "tabu", "id": "x", "num_restarts_b": 3, "tenure": -1, "num_reads_b": 10,
          "penalty": 10, "m_b": 50}, "tenure"),
    ],
)
def test_check_method_rejects_incomplete_or_invalid_specs(spec, message):
    with pytest.raises(ValueError, match=message):
        space.check_method(spec)


def test_a_spec_may_carry_extra_keys_for_the_record():
    spec = {**space.sample_configs("ls", 1, 0, 100)[0], "note": "smoke"}
    assert space.check_method(spec) == spec


def test_the_number_of_reads_per_batch_of_tabu_is_fixed_at_ten_in_every_configuration():
    """Not tuned: the plan lists only the restarts per batch. A constant of this implementation, disclosed in
    Methods (configs/E2/README.md); a change of it must be a decision, not an accident."""
    for n in (100, 200, 400):
        assert {c["num_reads_b"] for c in space.sample_configs("tabu", space.MAX_CONFIGS, 20261011, n)} == {10}


def test_a_scip_spec_may_ask_for_aggressive_primal_heuristics():
    # the sensitivity of H2 to the exact baseline (ANALYSIS_PLAN §12, 2026-10-19); absent means SCIP's defaults
    spec = {"name": "scip", "id": "scip-persp-mineig-sym0-heuraggr", "formulation": "perspective", "diag": "mineig",
            "usesymmetry": 0, "heuristics": "aggressive"}
    assert space.check_method(spec) == spec
    assert space.check_method({**spec, "heuristics": "default"})["heuristics"] == "default"

