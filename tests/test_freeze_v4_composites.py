"""Synthetic tests for the version-four composite bar freeze."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _freeze():
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_v4_bars.py"
    spec = importlib.util.spec_from_file_location("freeze_v4_bars_composites", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _metrics(offset: float = 0.0) -> dict:
    return {
        "exposures_and_rates": {"p95_relative_error": 0.10 + offset},
        "release_accuracy": {"p95_relative_error": 0.12 + offset},
        "interval_quality": {
            "coverage_deviation": 0.04 + offset,
            "mean_interval_score": 0.15 + offset,
        },
        "tail_calibration": {
            "pooled_exceedance_deviation": 0.03 + offset,
            "q95_width_relative_error": 0.20 + offset,
            "es95_width_relative_error": 0.22 + offset,
        },
        "reserve_skill": {"skill_loss": 0.18 + offset},
    }


def _report(world: str, submission_label: str, offset: float = 0.0,
            *, hard_pass: bool = True) -> dict:
    band_evidence = {
        "0-17": ("scored", 600.0, 12, 700.0),
        "18-44": ("report-only", 600.0, 12, 700.0),
        "45-64": ("report-only", 600.0, 12, 700.0),
        "65-74": ("report-only", 500.0, 10, 421.5),
        "75-84": ("report-only", 500.0, 6, 129.0),
        "85+": ("report-only", 500.0, 0, 1.0),
        "18-64": ("scored", 600.0, 12, 1_400.0),
        "65+": ("scored", 500.0, 12, 578.0),
    }

    def cell_evidence(floor: float, eligible: int, minimum: float) -> list[dict]:
        low_count = 12 - eligible
        low = [minimum] * low_count
        high = [max(floor, minimum) + 100.0] * eligible
        if not low and high:
            high[0] = minimum
        values = low + high
        return [
            {
                "state": index // 2,
                "sex": ("female", "male")[index % 2],
                "exposure_person_years": value,
                "eligible": value >= floor,
            }
            for index, value in enumerate(values)
        ]

    return {
        "hard_pass": hard_pass,
        "composite_metrics": _metrics(offset),
        "rate_metrics": {
            "composite": {
                "cells": [[0, "female", "65+"], [0, "male", "65+"]]
            }
        },
        "eligibility_evidence": {
            "truth_quantity": "retained state-by-sex person-years exposure",
            "bands": {
                band: {
                    "status": status,
                    "floor_person_years": floor,
                    "cell_count": 12,
                    "eligible_count": eligible,
                    "minimum_exposure_person_years": minimum,
                    "cells": cell_evidence(floor, eligible, minimum),
                }
                for band, (status, floor, eligible, minimum) in band_evidence.items()
            },
        },
        "evidence": {
            "schema": "meridia.v4.verifier-evidence.v1",
            "packet_digest_sha256": _digest(f"packet-{world}"),
            "contract_digest_sha256": _digest(f"contract-{world}"),
            "submission_digest_sha256": _digest(submission_label),
            "verifier_digest_sha256": _digest("verifier"),
        },
    }


def _reference(freeze, line: str, world: str, *, offset: float = 0.0) -> dict:
    entry = {
        "reference_line": line,
        "world": world,
        "method_digest_sha256": _digest(f"reference-method-{line}"),
        "runner_digest_sha256": _digest("runner"),
        "deterministic": True,
        "report": _report(world, f"final-submission-{line}-{world}", offset),
    }
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind="reference")
    return entry


def _replicate(freeze, line: str, world: str, replicate: int, offset: float) -> dict:
    entry = {
        "reference_line": line,
        "world": world,
        "replicate_id": str(replicate),
        "method_digest_sha256": _digest(f"reference-method-{line}"),
        "runner_digest_sha256": _digest("runner"),
        "resample_digest_sha256": _digest(
            f"resample-{line}-{world}-{replicate}"
        ),
        "resampling_design": {
            "method": "stratified participant-file bootstrap",
            "version": 1,
        },
        "deterministic": True,
        "report": _report(
            world,
            f"replicate-submission-{line}-{world}-{replicate}",
            offset,
        ),
    }
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind="replicate")
    return entry


def _control(freeze, name: str, world: str, gate: str, component: str, value: float,
             *, hard_pass: bool = True) -> dict:
    report = _report(world, f"control-submission-{name}-{world}", hard_pass=hard_pass)
    report["composite_metrics"][gate][component] = value
    entry = {
        "control": name,
        "world": world,
        "method_digest_sha256": _digest(f"control-method-{name}"),
        "runner_digest_sha256": _digest("runner"),
        "deterministic": True,
        "report": report,
    }
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind="control")
    return entry


def _evidence(freeze, replicates_per_pair: int = 7):
    worlds = [f"qual-{index}" for index in range(6)]
    lines = ["A", "B", "C"]
    references = [
        _reference(freeze, line, world) for line in lines for world in worlds
    ]
    replicates = []
    for line in lines:
        for world in worlds:
            for replicate in range(replicates_per_pair):
                ordinal = replicate + 1
                replicates.append(
                    _replicate(
                        freeze, line, world, replicate, ordinal / 10_000.0
                    )
                )
    control_specs = [
        (name, gate, freeze.GATE_COMPONENTS[gate][0])
        for gate, names in freeze.SCIENTIFIC_CONTROLS_BY_GATE.items()
        for name in names
    ]
    controls = [
        _control(freeze, name, world, gate, component, 0.8)
        for name, gate, component in control_specs
        for world in worlds
    ]
    return references, replicates, controls


def _elder_audit(freeze, references: list[dict]) -> dict:
    by_pair = {(row["reference_line"], row["world"]): row for row in references}
    worlds = []
    for index in range(6):
        world = f"qual-{index}"
        before = by_pair[("A", world)]
        after = by_pair[("C", world)]
        identification = freeze.MORTALITY_IDENTIFICATION_BASE["per_world"][world]
        counts = identification["mortality_spike_years"]
        worlds.append({
            "world": world,
            "before_report_evidence_id": before["evidence_id"],
            "after_report_evidence_id": after["evidence_id"],
            "exposure_65_plus_absolute_error_percent": {
                "definition": freeze.ELDER_EXPOSURE_ERROR_DEFINITION,
                "before": 10.0,
                "after": 5.0,
            },
            "state_65_plus_person_years": [
                {"state": state, "submitted_before": 110.0,
                 "submitted_after": 105.0, "sealed": 100.0}
                for state in range(6)
            ],
            "liability_mean_by_region": [
                {"region": region, "submitted_before": 900.0,
                 "submitted_after": 950.0, "sealed": 1_000.0}
                for region in range(6)
            ],
            "pooled_exceedance_deviation": {
                "definition": freeze.POOLED_EXCEEDANCE_DEFINITION,
                "before": before["report"]["composite_metrics"]
                ["tail_calibration"]["pooled_exceedance_deviation"],
                "after": after["report"]["composite_metrics"]
                ["tail_calibration"]["pooled_exceedance_deviation"],
            },
            "mortality_gap_decomposition": {
                "history_mortality_rate": 0.01,
                "horizon_mortality_rate": 0.009,
                "observed_horizon_to_history_ratio": identification[
                    "horizon_history_ratio"],
                "trend_only_horizon_to_history_ratio": identification[
                    "trend_only_ratio"],
                "residual_observed_to_trend_ratio": identification["residual_ratio"],
                "publication_lag_trend_factor": identification["lag_trend_factor"],
                "trend_active_during_public_experience_window": True,
                "trend_starts_only_after_public_window": False,
                "publication_lag_months": 12,
                "last_exposure_midpoint_to_snapshot_months": 18,
                "continuation_shocks_redrawn_per_member": True,
                "history_mortality_shock_years": list(range(counts["history"])),
                "lag_mortality_shock_years": list(range(counts["lag"])),
                "designated_horizon_mortality_shock_years": list(
                    range(counts["horizon"])),
            },
        })
    return {
        "schema": "meridia.methods.elder_reconstruction_audit.v1",
        "method_digest": {
            "git_commit": "abcdef0",
            "source_sha256": _digest("reference-method-C"),
            "before_line": "A",
            "after_line": "C",
        },
        "shock_redraw": {
            "annual_probability": 0.20,
            "independent_per_member": True,
            "magnitude_source": "participant/contract.json:shock_family",
            "mortality_ranges": [
                {"kind": "mortality_spike", "range": [1.5, 3.0]},
            ],
            "admission_ranges": [
                {"kind": "mortality_spike", "range": [1.4, 2.6]},
            ],
        },
        "eligibility_audit": {
            "scored": {"age_band": "65+", "floor_person_years": 500},
            "report_only": ["65-74", "75-84", "85+"],
            "younger_floors_changed": False,
        },
        "worlds": worlds,
    }


def _regime_audit(freeze) -> dict:
    bindings = [
        {
            "world": f"dev-{index:02d}",
            "regime": "development",
            "participant_digest_sha256": _digest(f"dev-participant-{index}"),
            "packet_manifest_digest_sha256": _digest(f"dev-manifest-{index}"),
        }
        for index in range(12)
    ] + [
        {
            "world": f"qual-{index}",
            "regime": "hidden",
            "participant_digest_sha256": _digest(f"qual-participant-{index}"),
            "packet_manifest_digest_sha256": _digest(f"qual-manifest-{index}"),
        }
        for index in range(6)
    ]
    correlations = {
        "mortality_improvement": 0.50,
        "migration_age_pattern": 0.53,
        "age_reporting_error": 0.65,
        "linkage_urban_gradient": 0.75,
        "administrative_completeness": 0.067,
        "missingness_target_dependence": -0.018,
    }
    axes = {}
    for axis in freeze.REGIME_AXES:
        constrained = axis in freeze.HIDDEN_IN_BAND_AXES
        development = list(freeze.DEVELOPMENT_AXIS_RANGES[axis])
        axes[axis] = {
            "statistic": f"participant statistic for {axis}",
            "expected_sign": freeze.REGIME_EXPECTED_SIGNS[axis],
            "signed_rank_correlation": correlations[axis],
            "within_regime_signed_rank_correlation": {
                "development": 0.5,
                "hidden": 0.5,
            },
            "intensity_range_observed": development,
            "anchor_correlation_qualified": correlations[axis] > 0.4,
            "disposition": (
                "constrained_to_development_range" if constrained
                else "participant_anchor"
            ),
            "development_range": development,
            "hidden_generation_range": (
                development if constrained else list(freeze.PUBLIC_AXIS_RANGES[axis])
            ),
            "hidden_out_of_band_allowed": not constrained,
        }
    audit = {
        "schema": freeze.REGIME_IDENTIFIABILITY_SCHEMA,
        "anchor_correlation_threshold": 0.4,
        "world_count": 18,
        "world_bindings": bindings,
        "measurement_rows_digest_sha256": _digest("identifiability-measurements"),
        "generator_source_digest_sha256": _digest("identifiability-sources"),
        "generator_policy": {
            "outside_axis_count": 2,
            "eligible_for_outside_development_band": list(
                freeze.HIDDEN_EXTRAPOLATION_AXES
            ),
            "held_inside_development_band": list(freeze.HIDDEN_IN_BAND_AXES),
        },
        "axes": axes,
    }
    audit["digest_sha256"] = freeze._canonical_digest(audit)
    return audit


def _calibrate(freeze, references, replicates, controls):
    return freeze.calibrate_composite_bars(
        references,
        replicates,
        controls,
        elder_reconstruction_audit=_elder_audit(freeze, references),
        regime_identifiability_audit=_regime_audit(freeze),
    )


def _rebind(freeze, entry: dict, kind: str) -> None:
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind=kind)


def test_p99_is_the_exact_ceiling_order_statistic_without_interpolation():
    freeze = _freeze()
    values = list(range(100))
    assert freeze.empirical_p99(values) == 98
    assert freeze.empirical_order_statistic([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    with pytest.raises(freeze.EvidenceError, match="finite"):
        freeze.empirical_p99([0.1, float("nan")])


def test_complete_freeze_has_only_five_composites_and_auditable_bars():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)

    from meridia.verify import _bar_schema_errors
    assert _bar_schema_errors(bars) == []

    assert bars["schema"] == "meridia.v4.composite-bars.v1"
    assert bars["frozen"] is True
    assert tuple(bars["gates"]) == tuple(freeze.GATE_COMPONENTS)
    assert bars["target_false_fail_rate"] == 0.01
    assert bars["target_marginal_product"] == pytest.approx(0.99 ** 15)
    assert bars["target_marginal_product"] == pytest.approx(0.8600583546412883)
    assert bars["reference_lines"] == ["A", "B", "C"]
    assert bars["qualification_worlds"] == [f"qual-{index}" for index in range(6)]
    assert bars["reference_report_count"] == 18
    assert bars["replicate_report_count"] == 126
    assert bars["replicates_per_reference_line_and_world"] == 7
    assert bars["control_report_count"] == 132
    assert bars["evidence_provenance"]["schema"] \
        == "meridia.v4.freeze-provenance.v1"
    assert len(bars["evidence_provenance"]["digest_sha256"]) == 64
    assert bars["control_support"]["complete_gate_count"] == 5
    assert bars["control_support"]["full_separation"] is True
    assert bars["control_support"]["deletion_candidates"] == []
    assert bars["control_support"]["required_control_count"] == 22
    assert bars["control_support"]["required_report_count"] == 132
    matrix = bars["control_support"]["matrix"]
    assert matrix["predictive_tails"]["hard_structure_pass"] is True
    assert matrix["predictive_tails"]["gates"]["tail_calibration"]["failed_worlds"] \
        == [f"qual-{index}" for index in range(6)]
    assert matrix["predictive_tails"]["gates"]["tail_calibration"] \
        ["passed_worlds"] == []

    record = bars["gates"]["exposures_and_rates"]["components"]["p95_relative_error"]
    raw = [row["report"]["composite_metrics"]["exposures_and_rates"]
           ["p95_relative_error"] for row in replicates]
    assert record["value"] == freeze.empirical_p99(raw)
    assert record["direction"] == "ceiling"
    assert record["range"] == [0.0, None]
    assert record["quantile"] == 0.99
    assert record["target_false_fail_rate"] == 0.01
    assert record["sample_count"] == len(replicates)
    assert record["worlds"] == [f"qual-{index}" for index in range(6)]
    assert record["witnesses"] == ["A", "B", "C"]
    assert record["supporting_controls"] == [
        "deterministic_linkage", "ignore_health_selection", "informative_selection"
    ]
    assert record["eligible_cells"]["distinct"] == [
        [0, "female", "65+"], [0, "male", "65+"]
    ]
    assert all(witness["pass"] for witness in record["reference_witnesses"])
    assert len(record["replicate_evidence_ids"]) == len(replicates)
    assert len(record["replicate_evidence_digest_sha256"]) == 64
    assert "not empirical pass probabilities" in bars["caveats"][0]
    assert "Only six qualification worlds" in bars["caveats"][1]

    report = freeze.render_freeze_report(bars)
    provenance = freeze.render_provenance(bars)
    for document in (report, provenance):
        for gate, components in freeze.GATE_COMPONENTS.items():
            assert gate in document
            assert (
                f"gate-union leave-one-world-out false-fail rate: "
                f"{bars['achieved_false_fail_rates'][gate]:.6%}"
            ) in document
            for component in components:
                component_record = bars["gates"][gate]["components"][component]
                assert component in document
                assert f"value: {component_record['value']:.12g}" in document
                assert (
                    "attainable range: "
                    + json.dumps(component_record["range"], separators=(",", ":"))
                ) in document
                assert "worlds: qual-0, qual-1, qual-2, qual-3, qual-4, qual-5" \
                    in document
                assert "witnesses: A, B, C" in document
        assert "eligible cells by qualification world" in document
        assert '[0,"female","65+"]' in document
        assert '[0,"male","65+"]' in document
        assert "q95 is order statistic ceil(0.95 * M) of the M continuations" \
            in document
        assert "ES95 is the mean of all continuations tied at or above q95" \
            in document
        assert "Mortality identification evidence for the tail gate" in document
        assert "Elder cohort-component qualification" in document
        assert "Regime-axis identifiability" in document
        assert (
            "administrative_completeness: signed rank correlation 0.067; "
            "disposition constrained_to_development_range"
        ) in document
        assert (
            "missingness_target_dependence: signed rank correlation -0.018; "
            "disposition constrained_to_development_range"
        ) in document
        assert "Control separation" in document
        assert "normal_tail: failed worlds qual-0, qual-1, qual-2, qual-3, qual-4, qual-5" \
            in document
        assert "absolute 65+ exposure error 10.0% before, 5.0% after" in document
        assert "pooled exceedance deviation 0.03 before, 0.03 after" in document
        assert "region 0 liability mean: 900.0 before, 950.0 after, 1000.0 sealed" \
            in document
        assert (
            "target marginal product over five gates and three graded worlds: "
            f"{bars['target_marginal_product']:.6f}"
        ) in document
        assert (
            "achieved conditional marginal-rate product: "
            f"{bars['achieved_marginal_rate_product']:.6f}"
        ) in document


def test_final_reports_are_not_used_as_missing_replicates():
    freeze = _freeze()
    references, _, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, None, controls)
    assert bars["frozen"] is False
    assert bars["gates"] == {}
    assert any("replicate evidence missing" in blocker for blocker in bars["blockers"])
    assert any("cannot be bootstrapped" in blocker for blocker in bars["blockers"])


def test_tail_freeze_requires_the_bound_elder_reconstruction_audit():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    missing = freeze.calibrate_composite_bars(references, replicates, controls)
    assert missing["frozen"] is False
    assert any("elder_reconstruction_audit" in blocker for blocker in missing["blockers"])

    audit = _elder_audit(freeze, references)
    audit["shock_redraw"]["independent_per_member"] = False
    wrong_shocks = freeze.calibrate_composite_bars(
        references,
        replicates,
        controls,
        elder_reconstruction_audit=audit,
    )
    assert wrong_shocks["frozen"] is False
    assert any("shock redraw" in blocker for blocker in wrong_shocks["blockers"])


def test_freeze_requires_unidentified_axes_to_stay_inside_the_development_range():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    audit = _regime_audit(freeze)
    audit["axes"]["administrative_completeness"]["hidden_out_of_band_allowed"] = True
    audit["axes"]["administrative_completeness"]["disposition"] = "participant_anchor"
    audit["axes"]["administrative_completeness"]["hidden_generation_range"] = \
        list(freeze.PUBLIC_AXIS_RANGES["administrative_completeness"])
    audit["digest_sha256"] = freeze._canonical_digest({
        key: value for key, value in audit.items() if key != "digest_sha256"
    })

    bars = freeze.calibrate_composite_bars(
        references,
        replicates,
        controls,
        elder_reconstruction_audit=_elder_audit(freeze, references),
        regime_identifiability_audit=audit,
    )

    assert bars["frozen"] is False
    assert any(
        "administrative_completeness: unanchored axis is not held in range" in blocker
        for blocker in bars["blockers"]
    )


def test_nonfinite_or_unequally_weighted_replicates_fail_closed():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    broken = deepcopy(replicates)
    broken[0]["report"]["composite_metrics"]["reserve_skill"]["skill_loss"] = math.inf
    bars = _calibrate(freeze, references, broken, controls)
    assert bars["frozen"] is False
    assert any("finite JSON" in blocker for blocker in bars["blockers"])

    unbalanced = replicates[:-1]
    bars = _calibrate(freeze, references, unbalanced, controls)
    assert bars["frozen"] is False
    assert any("equal" in blocker for blocker in bars["blockers"])


def test_too_few_replicates_cannot_claim_an_empirical_one_percent_tail():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze, replicates_per_pair=6)
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "each leave-one-world-out training fold needs at least 100" in blocker
        and "found 90" in blocker
        for blocker in bars["blockers"]
    )


def test_a_mismatched_evidence_identifier_fails_closed():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    replicates[0]["evidence_id"] = "0" * 64
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "evidence_id does not match its replay binding" in blocker
        for blocker in bars["blockers"]
    )


def test_duplicate_resample_digest_within_a_line_world_pair_fails_closed():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    replicates[1]["resample_digest_sha256"] = \
        replicates[0]["resample_digest_sha256"]
    replicates[1]["evidence_id"] = freeze.evidence_id_for(
        replicates[1], kind="replicate"
    )
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "duplicate resample digest within ('A', 'qual-0')" in blocker
        for blocker in bars["blockers"]
    )


def test_final_and_replicate_method_digest_mismatch_fails_closed():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    replicates[0]["method_digest_sha256"] = _digest("tampered-method")
    replicates[0]["evidence_id"] = freeze.evidence_id_for(
        replicates[0], kind="replicate"
    )
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "A: final and replicate method digests differ" in blocker
        for blocker in bars["blockers"]
    )


def test_packet_digest_disagreement_within_a_world_fails_closed():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    references[0]["report"]["evidence"]["packet_digest_sha256"] = \
        _digest("tampered-packet")
    references[0]["evidence_id"] = freeze.evidence_id_for(
        references[0], kind="reference"
    )
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "qual-0: evidence disagrees on packet_digest_sha256" in blocker
        for blocker in bars["blockers"]
    )


def test_eligible_cells_must_be_present_and_reference_independent():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    references[0]["report"]["rate_metrics"]["composite"]["cells"] = [
        [9, "female", "65+"]
    ]
    _rebind(freeze, references[0], "reference")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("differs across reference lines" in blocker for blocker in bars["blockers"])


def test_a_control_must_pass_hard_structure_and_fail_its_registered_gate():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    controls[0]["report"]["hard_pass"] = False
    _rebind(freeze, controls[0], "control")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert "deterministic_linkage" not in \
        bars["gates"]["exposures_and_rates"]["supporting_controls"]
    candidate = next(
        row for row in bars["control_support"]["deletion_candidates"]
        if row["gate"] == "exposures_and_rates"
    )
    failure = next(
        row for row in candidate["nonseparating_controls"]
        if row["control"] == "deterministic_linkage"
    )
    assert failure["hard_invalid_worlds"] == ["qual-0"]

    references, replicates, controls = _evidence(freeze)
    controls[:6] = [
        _control(
            freeze,
            "unregistered_shortcut",
            f"qual-{index}",
            "exposures_and_rates",
            "p95_relative_error",
            0.9,
        )
        for index in range(6)
    ]
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert "unregistered_shortcut" in bars["control_support"]["unexpected_controls"]
    assert "deterministic_linkage" not in \
        bars["gates"]["exposures_and_rates"]["supporting_controls"]


def test_a_control_must_cover_each_qualification_world_exactly_once():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    controls.pop(0)
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert "deterministic_linkage" not in \
        bars["gates"]["exposures_and_rates"]["supporting_controls"]
    missing = bars["control_support"]["matrix"]["deterministic_linkage"]
    assert missing["missing_worlds"] == ["qual-0"]

    references, replicates, controls = _evidence(freeze)
    controls[0] = deepcopy(controls[1])
    controls[0]["report"]["evidence"]["submission_digest_sha256"] = \
        _digest("duplicate-world-control-submission")
    controls[0]["evidence_id"] = freeze.evidence_id_for(
        controls[0], kind="control"
    )
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    duplicate = bars["control_support"]["matrix"]["deterministic_linkage"]
    assert duplicate["missing_worlds"] == ["qual-0"]
    assert duplicate["duplicate_worlds"] == ["qual-1"]


def test_one_control_pass_on_qual_two_makes_the_gate_a_deletion_candidate():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    row = next(
        entry for entry in controls
        if entry["control"] == "normal_tail" and entry["world"] == "qual-2"
    )
    row["report"]["composite_metrics"]["tail_calibration"] \
        ["pooled_exceedance_deviation"] = 0.01
    _rebind(freeze, row, "control")

    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is False
    assert bars["control_support"]["full_separation"] is False
    candidate = next(
        item for item in bars["control_support"]["deletion_candidates"]
        if item["gate"] == "tail_calibration"
    )
    normal = next(
        item for item in candidate["nonseparating_controls"]
        if item["control"] == "normal_tail"
    )
    assert normal["passed_worlds"] == ["qual-2"]
    assert normal["failed_worlds"] == [
        "qual-0", "qual-1", "qual-3", "qual-4", "qual-5"
    ]
    comparison = normal["per_world"]["qual-2"]["components"] \
        ["pooled_exceedance_deviation"]
    assert comparison["value"] == 0.01
    assert comparison["ceiling"] == bars["gates"]["tail_calibration"] \
        ["components"]["pooled_exceedance_deviation"]["value"]
    assert comparison["exceeds"] is False


def test_every_final_reference_must_clear_the_p99_bars():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    references[0]["report"]["composite_metrics"]["tail_calibration"] \
        ["es95_width_relative_error"] = 0.9
    _rebind(freeze, references[0], "reference")
    expected_evidence_id = references[0]["evidence_id"]
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert bars["reference_failures"] == [{
        "reference_line": "A",
        "world": "qual-0",
        "gate": "tail_calibration",
        "components": ["es95_width_relative_error"],
        "evidence_id": expected_evidence_id,
    }]


def test_leave_one_world_out_rate_can_stop_an_in_sample_p99_freeze():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    for row in replicates:
        if row["world"] == "qual-5":
            row["report"]["composite_metrics"]["release_accuracy"] \
                ["p95_relative_error"] += 0.2
            _rebind(freeze, row, "replicate")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert bars["achieved_false_fail_rates"]["release_accuracy"] > 0.01
    assert any("leave-one-world-out" in blocker for blocker in bars["blockers"])


def test_cli_writes_an_incomplete_bar_set_when_only_packet_paths_are_given(tmp_path):
    freeze = _freeze()
    out = tmp_path / "bars"
    exit_code = freeze.main([
        "--dev", "development-0",
        "--qualification", "qual-0",
        "--out", str(out),
    ])
    assert exit_code == 1
    bars = __import__("json").loads((out / "bars.json").read_text())
    assert bars["schema"] == "meridia.v4.composite-bars.v1"
    assert bars["frozen"] is False
    assert any("replicate evidence missing" in blocker for blocker in bars["blockers"])
    assert "NOT FROZEN" in (out / "freeze_report.txt").read_text()
