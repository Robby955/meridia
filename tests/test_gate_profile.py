"""The gate profile: which frozen composites decide, and what the rest still report."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pytest

from meridia.actuarial import perfect_information_allocation
from meridia.verify import (COMPOSITE_GATE_COMPONENTS, DEFAULT_GATE_PROFILE,
                            GATE_PROFILES, evaluate_composite_gates,
                            gate_profile_reported_only, gate_profile_selection,
                            verify_submission)

TAIL_CONTROLS = (
    "development_average_regime", "mean_only_tail", "normal_tail", "padded_tail",
    "predictive_tails", "regime_recombination",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _freeze():
    return _load("freeze_v4_bars_profile",
                 Path(__file__).resolve().parents[1] / "scripts" / "freeze_v4_bars.py")


@lru_cache(maxsize=1)
def _fixtures():
    return _load("composite_fixtures_for_profile",
                 Path(__file__).resolve().parent / "test_freeze_v4_composites.py")


@lru_cache(maxsize=1)
def _actuarial_fixtures():
    return _load("actuarial_fixtures_for_profile",
                 Path(__file__).resolve().parent / "test_actuarial.py")


@lru_cache(maxsize=2)
def _bars(gate_profile: str) -> dict:
    """One complete freeze receipt under the named profile."""
    freeze, fixtures = _freeze(), _fixtures()
    references, replicates, controls = fixtures._evidence(freeze)
    return freeze.calibrate_composite_bars(
        references, replicates, controls, gate_profile=gate_profile,
        **fixtures._calibration_kwargs(freeze, references, controls),
    )


def _wrong_tail_submission(tmp_path):
    """A submission whose gated blocks are right and whose filed q95 is far out.

    The filed q95 sits six ensemble tail widths above the sealed quantile and ES95 is
    carried up with it, so both width errors run past their own frozen bars. The
    allocation is the perfect-information one, so the reserve skill is exact.
    """
    actuarial = _actuarial_fixtures()
    packet, admin, truth, detailed, rate_truth, liability, sealed, total = \
        actuarial._packet(tmp_path)
    source = packet / "participant" / "sources" / "observed.csv"
    source.parent.mkdir(exist_ok=True)
    source.write_text("value\n1\n")
    oracle = perfect_information_allocation(liability, total)
    mean, quantile, shortfall = sealed["mean"], sealed["q"], sealed["es"]
    filed_q95 = mean + 6.0 * (quantile - mean)
    submission = tmp_path / "submission"
    actuarial._oracle_submission(submission, admin, truth, detailed, rate_truth, sealed,
                                 total, oracle, filed_q95)
    actuarial._write(submission / "reserve.csv", {
        "region": [0, 1], "liability_mean": mean, "q95": filed_q95,
        "es95": filed_q95 + (shortfall - quantile), "allocation": oracle,
    })
    return packet, submission


def test_every_profile_is_a_selection_over_the_five_frozen_gates():
    freeze = _freeze()
    assert DEFAULT_GATE_PROFILE == "full"
    assert freeze.DEFAULT_GATE_PROFILE == DEFAULT_GATE_PROFILE
    assert set(GATE_PROFILES) == set(freeze.GATE_PROFILES) \
        == {"full", "standard", "lite"}
    for name in GATE_PROFILES:
        selection = gate_profile_selection(name)
        assert selection == freeze.gate_profile_selection(name)
        assert set(selection) <= set(COMPOSITE_GATE_COMPONENTS)
        for gate, components in selection.items():
            assert components
            assert set(components) <= set(COMPOSITE_GATE_COMPONENTS[gate])
    assert gate_profile_selection("full") == {
        gate: tuple(components)
        for gate, components in COMPOSITE_GATE_COMPONENTS.items()
    }
    lite = gate_profile_selection("lite")
    assert set(lite) == {
        "exposures_and_rates", "release_accuracy", "interval_quality"}
    assert "tail_calibration" not in lite and "reserve_skill" not in lite
    standard = gate_profile_selection("standard")
    assert set(standard) == {
        "exposures_and_rates", "release_accuracy", "interval_quality",
        "tail_calibration"}
    assert "reserve_skill" not in standard
    for gate in ("exposures_and_rates", "release_accuracy", "interval_quality",
                 "tail_calibration"):
        assert standard[gate] == COMPOSITE_GATE_COMPONENTS[gate]
    assert gate_profile_reported_only("standard") == [
        "reserve_skill/skill_loss",
        "reserve_skill/worst_regional_shortfall_probability"]
    assert gate_profile_reported_only("full") == []
    assert gate_profile_reported_only("lite") == [
        "tail_calibration/pooled_exceedance_deviation",
        "tail_calibration/q95_width_relative_error",
        "tail_calibration/es95_width_relative_error",
        "reserve_skill/skill_loss",
        "reserve_skill/worst_regional_shortfall_probability",
    ]
    for name in GATE_PROFILES:
        assert gate_profile_reported_only(name) \
            == freeze.gate_profile_reported_only(name)
    with pytest.raises(ValueError):
        gate_profile_selection("tail_only")
    with pytest.raises(freeze.EvidenceError):
        freeze.gate_profile_selection("tail_only")


def test_a_profile_publishes_no_bar_only_for_a_component_it_reports():
    """The registry that drops a bar can never reach a component a verdict rests on."""
    from meridia.verify import (GATE_PROFILE_UNPUBLISHED_COMPONENTS,
                                gate_profile_unpublished_components)

    freeze = _freeze()
    assert GATE_PROFILE_UNPUBLISHED_COMPONENTS \
        == freeze.GATE_PROFILE_UNPUBLISHED_COMPONENTS
    assert set(GATE_PROFILE_UNPUBLISHED_COMPONENTS) <= set(GATE_PROFILES)
    for name in GATE_PROFILES:
        registered = gate_profile_unpublished_components(name)
        assert registered == freeze.gate_profile_unpublished_components(name)
        assert set(registered) <= set(gate_profile_reported_only(name))
        assert all(isinstance(reason, str) and reason
                   for reason in registered.values())
    assert set(gate_profile_unpublished_components("standard")) == {
        "reserve_skill/skill_loss"}
    assert gate_profile_unpublished_components("full") == {}
    assert gate_profile_unpublished_components("lite") == {}
    freeze.GATE_PROFILE_UNPUBLISHED_COMPONENTS["lite"] = {
        "exposures_and_rates/p95_relative_error": "not readable"}
    try:
        with pytest.raises(freeze.EvidenceError):
            freeze.gate_profile_unpublished_components("lite")
    finally:
        del freeze.GATE_PROFILE_UNPUBLISHED_COMPONENTS["lite"]


def test_lite_passes_a_submission_whose_tails_are_wrong(tmp_path):
    packet, submission = _wrong_tail_submission(tmp_path)
    report = verify_submission(packet, submission, _bars("lite"), gate_profile="lite")
    assert report["hard_pass"] is True
    assert report["reasons"] == []
    assert report["pass"] is True
    assert report["gate_profile"] == "lite"
    tail = report["gate_results"]["tail_calibration"]
    assert tail["gated"] is False and tail["pass"] is None and tail["reasons"] == []
    assert [detail.split()[0] for detail in tail["ungated_failures"]] == [
        "q95_width_relative_error", "es95_width_relative_error"]
    reserve = report["gate_results"]["reserve_skill"]
    assert reserve["gated"] is False and reserve["pass"] is None
    assert reserve["gated_components"] == [] and reserve["reasons"] == []
    # The tail block is still measured on the same submission, and the mean liability the
    # reserve file filed is still recomputed against the sealed ensemble mean.
    assert report["composite_metrics"]["tail_calibration"][
        "q95_width_relative_error"] == pytest.approx(5.0)
    assert report["reserve"]["mean_liability_error"] == pytest.approx(0.0)
    assert [row["sealed"] for row
            in report["elder_reference_evidence"]["liability_mean_by_region"]] \
        == [pytest.approx(row["submitted"]) for row
            in report["elder_reference_evidence"]["liability_mean_by_region"]]


def test_full_fails_the_same_submission(tmp_path):
    packet, submission = _wrong_tail_submission(tmp_path)
    report = verify_submission(packet, submission, _bars("full"))
    assert report["hard_pass"] is True
    assert report["gate_profile"] == "full"
    assert report["pass"] is False
    assert len(report["reasons"]) == 1
    assert report["reasons"][0].startswith("tail_calibration: ")
    assert "q95_width_relative_error" in report["reasons"][0]
    tail = report["gate_results"]["tail_calibration"]
    assert tail["gated"] is True and tail["pass"] is False
    assert tail["ungated_failures"] == []
    for gate in ("exposures_and_rates", "release_accuracy", "interval_quality",
                 "reserve_skill"):
        assert report["gate_results"][gate]["pass"] is True


def test_a_receipt_cannot_decide_under_a_profile_it_did_not_freeze(tmp_path):
    packet, submission = _wrong_tail_submission(tmp_path)
    report = verify_submission(packet, submission, _bars("full"), gate_profile="lite")
    assert report["pass"] is False and report["hard_pass"] is False
    assert any("gate profile" in reason for reason in report["reasons"])


def test_a_receipt_must_carry_the_registered_selection_for_the_profile_it_names():
    from meridia.verify import _bar_schema_errors

    assert _bar_schema_errors(_bars("lite")) == []
    forged = deepcopy(_bars("lite"))
    forged["gate_profile_selection"]["tail_calibration"] = ["pooled_exceedance_deviation"]
    assert any("gate profile selection differs" in error
               for error in _bar_schema_errors(forged))
    renamed = deepcopy(_bars("lite"))
    renamed["gate_profile"] = "tail_only"
    assert any("unregistered gate profile" in error
               for error in _bar_schema_errors(renamed))


def test_the_profile_name_reaches_the_bars_the_report_and_the_verdict(tmp_path):
    freeze = _freeze()
    packet, submission = _wrong_tail_submission(tmp_path)
    for profile, other in (("lite", "full"), ("full", "lite")):
        bars = _bars(profile)
        assert bars["frozen"] is True
        assert bars["gate_profile"] == profile
        assert bars["gate_profile_selection"] == {
            gate: list(components)
            for gate, components in gate_profile_selection(profile).items()
        }
        report_text = freeze.render_freeze_report(bars)
        provenance = freeze.render_provenance(bars)
        assert f"PROFILE: {profile}" in report_text
        assert f"- profile: {profile}" in report_text
        assert f"Gate profile: `{profile}`." in provenance
        assert f"- profile: {profile}" in provenance
        assert f"PROFILE: {other}" not in report_text
        verdict = verify_submission(packet, submission, bars, gate_profile=profile)
        assert verdict["gate_profile"] == profile
    lite_report = freeze.render_freeze_report(_bars("lite"))
    assert "tail_calibration (reported, decides nothing)" in lite_report
    assert "reserve_skill (reported, decides nothing)" in lite_report
    full_report = freeze.render_freeze_report(_bars("full"))
    assert "tail_calibration (decides on pooled_exceedance_deviation, " \
        "q95_width_relative_error, es95_width_relative_error)" in full_report


def test_a_control_that_fails_only_tail_or_reserve_gates_is_a_lite_deletion_candidate():
    freeze = _freeze()
    lite, full = _bars("lite"), _bars("full")
    reported_controls = set(TAIL_CONTROLS) | set(
        freeze.SCIENTIFIC_CONTROLS_BY_GATE["reserve_skill"])
    lite_support = lite["control_support"]["gate_profile"]
    assert lite_support["name"] == "lite"
    assert lite_support["reported_only_gates"] == [
        "tail_calibration", "reserve_skill"]
    assert lite_support["deletion_candidate_controls"] == sorted(reported_controls)
    for record in lite_support["deletion_candidates"]:
        assert record["primary_gate"] in ("tail_calibration", "reserve_skill")
        assert record["primary_gate_decides"] is False
        assert record["failed_gated_worlds"] == []
        assert record["unseparated_worlds"] == list(lite["qualification_worlds"])
    assert set(lite_support["separating_controls"]) == \
        set(freeze.REQUIRED_SCIENTIFIC_CONTROLS) - reported_controls
    # Under lite those wrong tail and wrong reserve methods pass the task, so the
    # report names them.
    for text in (freeze.render_freeze_report(lite), freeze.render_provenance(lite)):
        assert "lite profile deletion candidates" in text
        named = text.split("lite profile deletion candidates")[1]
        assert all(name in named for name in reported_controls)
    # The registered per-primary-gate battery is unchanged, and full names nobody.
    assert full["control_support"]["gate_profile"]["deletion_candidate_controls"] == []
    assert lite["control_support"]["deletion_candidates"] == []
    assert lite["control_support"]["separated_controls_by_gate"] == \
        full["control_support"]["separated_controls_by_gate"]


def test_a_reference_above_a_tail_bar_blocks_full_and_is_recorded_under_lite():
    freeze, fixtures = _freeze(), _fixtures()
    references, replicates, controls = fixtures._evidence(freeze)
    references = deepcopy(references)
    references[0]["report"]["composite_metrics"]["tail_calibration"][
        "q95_width_relative_error"] = 5.0
    fixtures._rebind(freeze, references[0], "reference")
    kwargs = fixtures._calibration_kwargs(freeze, references, controls)
    full = freeze.calibrate_composite_bars(references, replicates, controls, **kwargs)
    lite = freeze.calibrate_composite_bars(references, replicates, controls,
                                           gate_profile="lite", **kwargs)
    assert full["frozen"] is False
    assert any("exceed the p99 bars" in blocker for blocker in full["blockers"])
    assert [row["gate"] for row in full["reference_failures"]] == ["tail_calibration"]
    assert full["ungated_reference_failures"] == []
    assert lite["frozen"] is True and lite["blockers"] == []
    assert lite["reference_failures"] == []
    assert [row["gate"] for row in lite["ungated_reference_failures"]] == \
        ["tail_calibration"]
    text = freeze.render_freeze_report(lite)
    assert "- reference results above a reported bar:" in text
    assert "A/qual-0: tail_calibration q95_width_relative_error" in text


def test_an_ungated_component_never_produces_a_reason():
    metrics = {
        "exposures_and_rates": {"p95_relative_error": 0.1},
        "release_accuracy": {"p95_relative_error": 0.1},
        "interval_quality": {"coverage_deviation": 0.1, "mean_interval_score": 0.1},
        "tail_calibration": {"pooled_exceedance_deviation": 0.9,
                             "q95_width_relative_error": float("nan"),
                             "es95_width_relative_error": 0.1},
        "reserve_skill": {"skill_loss": 0.1,
                          "worst_regional_shortfall_probability": 0.9},
    }
    bars = {"gates": {gate: {"components": {component: {"value": 0.5}
                                            for component in components}}
                      for gate, components in COMPOSITE_GATE_COMPONENTS.items()}}
    lite = evaluate_composite_gates(metrics, bars, True, "lite")
    assert [gate for gate, result in lite.items() if result["gated"]] == [
        "exposures_and_rates", "release_accuracy", "interval_quality"]
    assert all(result["pass"] for result in lite.values() if result["gated"])
    assert lite["reserve_skill"]["pass"] is None
    assert lite["reserve_skill"]["ungated_failures"] == [
        "worst_regional_shortfall_probability 0.9 > 0.5"]
    assert lite["tail_calibration"]["pass"] is None
    assert "non-finite components ['q95_width_relative_error']" in \
        lite["tail_calibration"]["ungated_failures"]
    full = evaluate_composite_gates(metrics, bars, True, "full")
    assert full["tail_calibration"]["pass"] is False
    assert full["tail_calibration"]["reasons"] == [
        "non-finite components ['q95_width_relative_error']"]
    assert full["reserve_skill"]["pass"] is False
    assert full["reserve_skill"]["reasons"] == [
        "worst_regional_shortfall_probability 0.9 > 0.5"]


def test_a_refused_freeze_still_names_the_profile_and_what_it_reports():
    """A fail-closed document says which gates would have decided and which would not."""
    freeze = _freeze()
    for profile, reported in (("full", []),
                              ("lite", ["tail_calibration", "reserve_skill"])):
        refused = freeze.calibrate_composite_bars(
            [], None, [], gate_profile=profile
        )
        assert refused["frozen"] is False and refused["blockers"]
        assert refused["gate_profile"] == profile
        assert refused["reported_only_gates"] == reported
        assert refused["reference_failures"] == []
        assert refused["ungated_reference_failures"] == []
        text = freeze.render_freeze_report(refused)
        assert f"PROFILE: {profile}" in text


def _standard_bars():
    """A complete freeze under standard, on evidence whose shortfall reads one."""
    freeze, fixtures = _freeze(), _fixtures()
    references, replicates, controls = fixtures._evidence(freeze)
    fixtures._saturate_shortfall(freeze, references, replicates, controls)
    return freeze.calibrate_composite_bars(
        references, replicates, controls, gate_profile="standard",
        **fixtures._calibration_kwargs(freeze, references, controls),
    )


def test_standard_reports_the_whole_reserve_block_with_no_bar():
    """Standard decides four blocks and publishes no bar under either reserve component.

    The shortfall probability carries no bar because its calibrated value reaches the top
    of its attainable range. The skill loss carries none because the profile registers
    the reason its finite value is not a ceiling any method should be held to.
    """
    from meridia.verify import gate_profile_unpublished_components

    freeze = _freeze()
    bars = _standard_bars()
    assert bars["gate_profile"] == "standard"
    assert bars["gate_profile_selection"] == {
        gate: list(components)
        for gate, components in gate_profile_selection("standard").items()
    }
    assert bars["reported_only_gates"] == ["reserve_skill"]
    assert bars["reported_only_components"] == [
        "reserve_skill/skill_loss",
        "reserve_skill/worst_regional_shortfall_probability"]
    assert bars["reference_failures"] == []

    reserve = bars["gates"]["reserve_skill"]["components"]
    registered = gate_profile_unpublished_components("standard")
    skill = reserve["skill_loss"]
    assert skill["value"] is None and skill["publishable"] is False
    assert skill["unpublishable"]["calibrated_value"] > 0.0
    assert skill["unpublishable"]["attainable_ceiling"] is None
    assert skill["unpublishable"]["decides_under_profile"] is False
    assert skill["unpublishable"]["reason"] == registered["reserve_skill/skill_loss"]
    assert all(witness["pass"] is None for witness in skill["reference_witnesses"])
    # Nothing is compared against a component with no bar, so its false-fail rate is
    # zero on every line and the block's union reads the same.
    assert set(skill["false_fail_count_by_reference_line"].values()) == {0}
    for line in bars["reference_lines"]:
        assert bars["achieved_false_fail_rates_by_reference_line_and_component"][line][
            "reserve_skill"] == {"skill_loss": 0.0,
                                 "worst_regional_shortfall_probability": 0.0}
        assert bars["achieved_gate_union_false_fail_rates_by_reference_line"][line][
            "reserve_skill"] == 0.0
    shortfall = reserve["worst_regional_shortfall_probability"]
    assert shortfall["value"] is None and shortfall["publishable"] is False
    assert shortfall["unpublishable"]["attainable_ceiling"] == 1.0
    assert "cannot fail" in shortfall["unpublishable"]["reason"]

    for text in (freeze.render_freeze_report(bars), freeze.render_provenance(bars)):
        assert "- profile: standard" in text
        assert ("- reported-only components: reserve_skill/skill_loss, "
                "reserve_skill/worst_regional_shortfall_probability") in text
        assert "reserve_skill (reported, decides nothing)" in text
        assert "value: no bar published" in text

    # The reserve block reports and decides nothing, so its three registered wrong
    # methods separate nothing under this profile and are named rather than blocking.
    assert bars["blockers"] == []
    assert bars["frozen"] is True
    support = bars["control_support"]
    assert support["separating_controls_by_gate"]["reserve_skill"] == []
    assert support["block_roles"]["reserve_skill"]["role"] == "reported"
    assert support["block_roles"]["reserve_skill"]["deletion_candidate_controls"] == \
        list(freeze.SCIENTIFIC_CONTROLS_BY_GATE["reserve_skill"])
    assert support["unseparated_blocks"] == []

    from meridia.verify import _bar_schema_errors
    assert _bar_schema_errors(bars) == []


def test_standard_registers_three_deciding_blocks_as_validity_gates():
    """A deciding block with no separating control stands only where it is registered."""
    from meridia.verify import gate_profile_validity_blocks

    freeze = _freeze()
    assert gate_profile_validity_blocks("standard") == [
        "exposures_and_rates", "release_accuracy", "interval_quality"]
    assert gate_profile_validity_blocks("standard") \
        == freeze.gate_profile_validity_blocks("standard")
    assert gate_profile_validity_blocks("full") == []
    assert gate_profile_validity_blocks("lite") == []
    for name in GATE_PROFILES:
        assert set(gate_profile_validity_blocks(name)) \
            <= set(gate_profile_selection(name))
    freeze.GATE_PROFILE_VALIDITY_BLOCKS["lite"] = ("tail_calibration",)
    try:
        with pytest.raises(freeze.EvidenceError):
            freeze.gate_profile_validity_blocks("lite")
    finally:
        del freeze.GATE_PROFILE_VALIDITY_BLOCKS["lite"]

    bars = _standard_bars()
    support = bars["control_support"]
    assert support["registered_validity_gate_blocks"] == [
        "exposures_and_rates", "release_accuracy", "interval_quality"]
    # On this evidence every registered wrong method still fails its own block on every
    # world, so all four deciding blocks read as discriminating and none leans on the
    # registration.
    assert support["discriminating_blocks"] == [
        "exposures_and_rates", "release_accuracy", "interval_quality",
        "tail_calibration"]
    assert support["validity_gate_blocks"] == []
    assert support["requirement"] == freeze.CONTROL_SEPARATION_REQUIREMENT


def test_standard_scores_four_blocks_and_reports_the_reserve_block(tmp_path):
    packet, submission = _wrong_tail_submission(tmp_path)
    bars = _standard_bars()
    report = verify_submission(packet, submission, bars, gate_profile="standard")

    assert report["gate_profile"] == "standard"
    assert report["reported_only_components"] == [
        "reserve_skill/skill_loss",
        "reserve_skill/worst_regional_shortfall_probability"]
    reserve = report["gate_results"]["reserve_skill"]
    assert reserve["gated"] is False and reserve["pass"] is None
    assert reserve["gated_components"] == [] and reserve["reasons"] == []
    # Neither reserve component carries a bar, so neither can pass nor fail and neither
    # reaches the reasons or the ungated failures.
    assert reserve["ungated_failures"] == []
    for component in ("skill_loss", "worst_regional_shortfall_probability"):
        assert component in report["composite_metrics"]["reserve_skill"]
    # The tail block decides under standard, so this submission fails on it.
    assert report["pass"] is False
    assert [reason.split(":")[0] for reason in report["reasons"]] == \
        ["tail_calibration"]


def test_a_standard_receipt_cannot_decide_under_another_profile(tmp_path):
    packet, submission = _wrong_tail_submission(tmp_path)
    bars = _standard_bars()
    for other in ("full", "lite"):
        report = verify_submission(packet, submission, bars, gate_profile=other)
        assert report["pass"] is False and report["hard_pass"] is False
        assert any("the receipt froze the 'standard' gate profile" in reason
                   for reason in report["reasons"])
    renamed = deepcopy(bars)
    renamed["gate_profile"] = "full"
    renamed["gate_profile_selection"] = {
        gate: list(components)
        for gate, components in gate_profile_selection("full").items()
    }
    from meridia.verify import _bar_schema_errors
    errors = _bar_schema_errors(renamed)
    assert any("must carry a published bar" in error for error in errors)
    assert any("reported-only components differ" in error for error in errors)
