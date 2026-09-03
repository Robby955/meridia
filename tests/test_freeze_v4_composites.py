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
        "reserve_q95_feasibility": {
            "q95_sum": 60.0,
            "allocation_sum": 100.0,
            "reserve_total": 100.0,
            "total_minus_q95_sum": 40.0,
            "all_regions_at_or_above_q95": True,
            "allocation_sums_to_total": True,
            "feasible": True,
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
        "measurement_contract_digest_sha256": _digest("measurement-contract"),
        "run_receipt_digest_sha256": _digest(f"reference-receipt-{line}-{world}"),
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
        "measurement_contract_digest_sha256": _digest("measurement-contract"),
        "run_receipt_digest_sha256": _digest(
            f"replicate-receipt-{line}-{world}-{replicate}"
        ),
        "resample_digest_sha256": _digest(f"resample-{world}-{replicate}"),
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
        "measurement_contract_digest_sha256": _digest("measurement-contract"),
        "run_receipt_digest_sha256": _digest(f"control-receipt-{name}-{world}"),
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


def _diagnostic(freeze, name: str, world: str) -> dict:
    entry = {
        "diagnostic": name,
        "world": world,
        "method_digest_sha256": _digest(f"diagnostic-method-{name}"),
        "runner_digest_sha256": _digest("runner"),
        "measurement_contract_digest_sha256": _digest("measurement-contract"),
        "run_receipt_digest_sha256": _digest(f"diagnostic-receipt-{name}-{world}"),
        "deterministic": True,
        "report": _report(world, f"diagnostic-submission-{name}-{world}"),
    }
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind="diagnostic")
    return entry


def _diagnostics(freeze) -> list[dict]:
    return [
        _diagnostic(freeze, name, world)
        for name in freeze.DEVELOPMENT_DIAGNOSTICS
        for world in freeze.DEVELOPMENT_WORLDS
    ]


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


def _signed(freeze, payload: dict) -> dict:
    payload.pop("digest_sha256", None)
    payload["digest_sha256"] = freeze._canonical_digest(payload)
    return payload


def _reserve_audits(freeze, references: list[dict], controls: list[dict]):
    measurement_contract = _digest("measurement-contract")
    calibration = _signed(freeze, {
        "schema": freeze.RESERVE_CALIBRATION_SCHEMA,
        "measurement_contract_digest_sha256": measurement_contract,
        "candidate": True,
        "accepted": True,
        "blockers": [],
        "rate_per_person_year": 1.0,
        "rate_grid": 1.0,
        "tail_slack_share": 0.25,
        "target_rule": "sum(q95) + tail_slack_share * sum(ES95 - q95)",
        "reference_lines": list(freeze.REFERENCE_LINES),
        "qualification_worlds": list(freeze.QUALIFICATION_WORLDS),
        "evidence": [
            {
                "reference_line": row["reference_line"],
                "world": row["world"],
                "evidence_id": row["evidence_id"],
                "submitted_q95_sum": 60.0,
                "submitted_es95_sum": 80.0,
                "candidate_reserve_total": 100.0,
                "candidate_margin": 35.0,
            }
            for row in references
        ],
    })
    red_team = _signed(freeze, {
        "schema": freeze.RESERVE_RED_TEAM_SCHEMA,
        "measurement_contract_digest_sha256": measurement_contract,
        "independent_unit": "world",
        "world_counts": {"development": 12, "qualification": 6, "total": 18},
        "reserve_total_public_rule_verified": True,
        "primary_measure": (
            "qualification incremental regional R2 over development region means"
        ),
        "public_quantities": {
            "development": [
                {
                    "world": world,
                    "latest_year_total_exposure": 1000.0,
                    "reserve_total": 100.0,
                }
                for world in freeze.DEVELOPMENT_WORLDS
            ],
            "qualification": [
                {
                    "world": world,
                    "latest_year_total_exposure": 1000.0,
                    "reserve_total": 100.0,
                }
                for world in freeze.QUALIFICATION_WORLDS
            ],
        },
        "qualification_incremental_regional_r2_over_region_means": {
            "q95": 0.10,
            "es95": 0.20,
            "headline_max": 0.20,
        },
    })

    def qualification_row(row: dict, identity: str, skill_pass: bool) -> dict:
        receipt = row["report"]["reserve_q95_feasibility"]
        return {
            identity: row[identity],
            "world": row["world"],
            "evidence_id": row["evidence_id"],
            "q95_feasible": True,
            "reserve_skill_pass": skill_pass,
            **{
                key: receipt[key]
                for key in (
                    "q95_sum", "allocation_sum", "reserve_total",
                    "total_minus_q95_sum",
                )
            },
        }

    qualification = _signed(freeze, {
        "schema": freeze.RESERVE_QUALIFICATION_SCHEMA,
        "measurement_contract_digest_sha256": measurement_contract,
        "reference_lines": list(freeze.REFERENCE_LINES),
        "qualification_worlds": list(freeze.QUALIFICATION_WORLDS),
        "calibration_audit_digest_sha256": calibration["digest_sha256"],
        "red_team_audit_digest_sha256": red_team["digest_sha256"],
        "reference_results": [
            qualification_row(row, "reference_line", True) for row in references
        ],
        "proportional_reserve_results": [
            qualification_row(row, "control", False)
            for row in controls
            if row["control"] == "proportional_reserve"
        ],
    })
    return qualification, calibration, red_team


def _calibrate(freeze, references, replicates, controls):
    return freeze.calibrate_composite_bars(
        references,
        replicates,
        controls,
        **_calibration_kwargs(freeze, references, controls),
    )


def _calibration_kwargs(freeze, references, controls) -> dict:
    qualification, calibration, red_team = _reserve_audits(
        freeze, references, controls
    )
    return {
        "development_diagnostic_reports": _diagnostics(freeze),
        "elder_reconstruction_audit": _elder_audit(freeze, references),
        "regime_identifiability_audit": _regime_audit(freeze),
        "reserve_qualification_audit": qualification,
        "reserve_calibration_audit": calibration,
        "reserve_red_team_audit": red_team,
    }


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
        assert "Authenticated evidence design" in document
        assert "reference results at false-fail rate" in document
        assert "A/qual-0: pass; evidence" in document
        assert "deterministic_linkage [primary]: failed worlds" in document
        assert "q95 feasibility margin" in document
        assert "unique run receipts: 300" in document
        assert "normal_tail [primary]: failed worlds qual-0, qual-1, qual-2, qual-3, qual-4, qual-5" \
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


def test_freezer_uses_the_fixed_world_counts_and_control_registry():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    wrong_world_count = freeze.calibrate_composite_bars(
        references, replicates, controls, graded_world_count=4, **kwargs
    )
    assert wrong_world_count["frozen"] is False
    assert any("three graded worlds" in blocker
               for blocker in wrong_world_count["blockers"])

    registry = {
        gate: tuple(names)
        for gate, names in freeze.SCIENTIFIC_CONTROLS_BY_GATE.items()
    }
    registry["exposures_and_rates"] = tuple(
        reversed(registry["exposures_and_rates"])
    )
    wrong_registry = freeze.calibrate_composite_bars(
        references, replicates, controls, control_registry=registry, **kwargs
    )
    assert wrong_registry["frozen"] is False
    assert any("fixed V4 battery" in blocker for blocker in wrong_registry["blockers"])


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
    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs.pop("elder_reconstruction_audit")
    missing = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert missing["frozen"] is False
    assert any("elder_reconstruction_audit" in blocker for blocker in missing["blockers"])

    audit = _elder_audit(freeze, references)
    audit["shock_redraw"]["independent_per_member"] = False
    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["elder_reconstruction_audit"] = audit
    wrong_shocks = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
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

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["regime_identifiability_audit"] = audit
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
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
    assert any("exactly 126" in blocker for blocker in bars["blockers"])


def test_too_few_replicates_cannot_claim_an_empirical_one_percent_tail():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze, replicates_per_pair=6)
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "exactly 126" in blocker
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
    assert any(
        "unexpected ['unregistered_shortcut']" in blocker
        for blocker in bars["blockers"]
    )


def test_a_control_must_cover_each_qualification_world_exactly_once():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    controls.pop(0)
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("exactly 132" in blocker for blocker in bars["blockers"])

    references, replicates, controls = _evidence(freeze)
    controls[0] = deepcopy(controls[1])
    controls[0]["report"]["evidence"]["submission_digest_sha256"] = \
        _digest("duplicate-world-control-submission")
    controls[0]["evidence_id"] = freeze.evidence_id_for(
        controls[0], kind="control"
    )
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("run receipt was reused" in blocker for blocker in bars["blockers"])


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


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("measurement_contract_digest_sha256", "measurement_contract_digest_sha256"),
        ("run_receipt_digest_sha256", "run_receipt_digest_sha256"),
    ],
)
def test_every_report_must_bind_the_measurement_contract_and_run_receipt(field, message):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    del references[0][field]
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(message in blocker for blocker in bars["blockers"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runner_digest_sha256", _digest("other-runner"), "more than one runner"),
        (
            "measurement_contract_digest_sha256",
            _digest("other-measurement-contract"),
            "more than one measurement_contract",
        ),
        (
            "resampling_design",
            {"method": "different bootstrap", "version": 1},
            "more than one resampling design",
        ),
    ],
)
def test_all_reports_share_one_runner_measurement_contract_and_resampling_design(
    field, value, message
):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    replicates[0][field] = value
    _rebind(freeze, replicates[0], "replicate")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(message in blocker for blocker in bars["blockers"])


def test_reference_lines_are_exactly_a_b_and_c():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    for row in [*references, *replicates]:
        if row["reference_line"] == "C":
            row["reference_line"] = "D"
            row["method_digest_sha256"] = _digest("reference-method-D")
            _rebind(
                freeze,
                row,
                "replicate" if "replicate_id" in row else "reference",
            )
    bars = freeze.calibrate_composite_bars(references, replicates, controls)
    assert bars["frozen"] is False
    assert any("exactly A, B, and C" in blocker for blocker in bars["blockers"])


def test_reference_resamples_are_paired_across_lines():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    row = next(
        row for row in replicates
        if row["reference_line"] == "B"
        and row["world"] == "qual-0"
        and row["replicate_id"] == "0"
    )
    row["resample_digest_sha256"] = _digest("unpaired-resample")
    _rebind(freeze, row, "replicate")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "A, B, and C use different resample digests" in blocker
        for blocker in bars["blockers"]
    )


def test_run_receipts_cannot_be_reused_or_control_methods_relabelled():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    controls[1]["run_receipt_digest_sha256"] = controls[0][
        "run_receipt_digest_sha256"
    ]
    _rebind(freeze, controls[1], "control")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("run receipt was reused" in blocker for blocker in bars["blockers"])

    references, replicates, controls = _evidence(freeze)
    register_digest = next(
        row["method_digest_sha256"]
        for row in controls if row["control"] == "register_only"
    )
    for row in controls:
        if row["control"] == "survey_only":
            row["method_digest_sha256"] = register_digest
            _rebind(freeze, row, "control")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "method digest reused under different" in blocker for blocker in bars["blockers"]
    )


def test_control_method_digest_must_be_stable_across_worlds():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    row = next(
        row for row in controls
        if row["control"] == "normal_tail" and row["world"] == "qual-2"
    )
    row["method_digest_sha256"] = _digest("changed-normal-tail-method")
    _rebind(freeze, row, "control")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "normal_tail: control method digest is not stable" in blocker
        for blocker in bars["blockers"]
    )


def test_development_diagnostics_are_required_and_do_not_count_as_controls():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["control_report_count"] == 132
    assert bars["development_diagnostic_report_count"] == 24
    assert bars["development_diagnostics"]["counts_as_qualification_control"] is False
    assert set(bars["control_support"]["matrix"]) \
        == set(freeze.REQUIRED_SCIENTIFIC_CONTROLS)

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["development_diagnostic_reports"] = kwargs[
        "development_diagnostic_reports"
    ][:-1]
    incomplete = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert incomplete["frozen"] is False
    assert any("exactly 24" in blocker for blocker in incomplete["blockers"])


def test_reserve_audits_and_control_q95_feasibility_are_authenticated():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["reserve_calibration_audit"]["evidence"][0]["submitted_q95_sum"] = 61.0
    kwargs["reserve_calibration_audit"]["digest_sha256"] = freeze._canonical_digest({
        key: value
        for key, value in kwargs["reserve_calibration_audit"].items()
        if key != "digest_sha256"
    })
    mismatched = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert mismatched["frozen"] is False
    assert any(
        "calibration q95 sum differs from the verifier" in blocker
        for blocker in mismatched["blockers"]
    )

    references, replicates, controls = _evidence(freeze)
    controls[0]["report"]["reserve_q95_feasibility"]["q95_sum"] = 101.0
    controls[0]["report"]["reserve_q95_feasibility"][
        "total_minus_q95_sum"
    ] = -1.0
    _rebind(freeze, controls[0], "control")
    invalid = _calibrate(freeze, references, replicates, controls)
    assert invalid["frozen"] is False
    assert any("reserve q95 feasibility" in blocker for blocker in invalid["blockers"])


def test_bar_schema_rejects_relabelled_methods_and_duplicate_run_receipts():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    relabelled = deepcopy(bars)
    rows = relabelled["evidence_provenance"]["control_reports"]
    register_digest = next(
        row["method_digest_sha256"] for row in rows
        if row["control"] == "register_only"
    )
    for row in rows:
        if row["control"] == "survey_only":
            row["method_digest_sha256"] = register_digest
            unsigned = {key: value for key, value in row.items() if key != "evidence_id"}
            row["evidence_id"] = freeze._canonical_digest(unsigned)
    provenance = relabelled["evidence_provenance"]
    provenance["digest_sha256"] = freeze._canonical_digest({
        key: value for key, value in provenance.items() if key != "digest_sha256"
    })
    assert any(
        "relabeled" in error for error in _bar_schema_errors(relabelled)
    )

    duplicated = deepcopy(bars)
    rows = duplicated["evidence_provenance"]["control_reports"]
    rows[1]["run_receipt_digest_sha256"] = rows[0]["run_receipt_digest_sha256"]
    unsigned = {key: value for key, value in rows[1].items() if key != "evidence_id"}
    rows[1]["evidence_id"] = freeze._canonical_digest(unsigned)
    provenance = duplicated["evidence_provenance"]
    provenance["digest_sha256"] = freeze._canonical_digest({
        key: value for key, value in provenance.items() if key != "digest_sha256"
    })
    assert any(
        "reuses a run receipt" in error for error in _bar_schema_errors(duplicated)
    )


def test_bar_schema_binds_gate_witnesses_to_the_registered_reports():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    fake_reference = deepcopy(bars)
    component = fake_reference["gates"]["release_accuracy"]["components"][
        "p95_relative_error"
    ]
    component["reference_witnesses"][0]["evidence_id"] = _digest(
        "unregistered-reference"
    )
    assert any(
        "final witness receipt" in error for error in _bar_schema_errors(fake_reference)
    )

    fake_replicate = deepcopy(bars)
    component = fake_replicate["gates"]["release_accuracy"]["components"][
        "p95_relative_error"
    ]
    component["replicate_evidence_ids"][0] = _digest("unregistered-replicate")
    component["replicate_evidence_ids"].sort()
    component["replicate_evidence_digest_sha256"] = hashlib.sha256(
        "\n".join(component["replicate_evidence_ids"]).encode("utf-8")
    ).hexdigest()
    assert any(
        "replicate evidence receipt" in error
        for error in _bar_schema_errors(fake_replicate)
    )


def test_bar_schema_rejects_re_signed_reserve_values_not_bound_to_run_evidence():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    calibration = changed["reserve_audits"]["calibration"]
    qualification = changed["reserve_audits"]["qualification"]
    calibration_row = calibration["evidence"][0]
    qualification_row = next(
        row for row in qualification["reference_results"]
        if row["reference_line"] == calibration_row["reference_line"]
        and row["world"] == calibration_row["world"]
    )
    calibration_row.update({
        "submitted_q95_sum": 90.0,
        "submitted_es95_sum": 95.0,
        "candidate_reserve_total": 100.0,
        "candidate_margin": 8.75,
    })
    qualification_row["q95_sum"] = 90.0
    qualification_row["total_minus_q95_sum"] = 10.0
    _signed(freeze, calibration)
    qualification["calibration_audit_digest_sha256"] = calibration["digest_sha256"]
    _signed(freeze, qualification)
    assert any(
        "reserve qualification, calibration, or red-team audit is invalid" in error
        for error in _bar_schema_errors(changed)
    )

    changed = deepcopy(bars)
    red_team = changed["reserve_audits"]["red_team"]
    qualification = changed["reserve_audits"]["qualification"]
    del red_team["public_quantities"]["development"][0][
        "latest_year_total_exposure"
    ]
    _signed(freeze, red_team)
    qualification["red_team_audit_digest_sha256"] = red_team["digest_sha256"]
    _signed(freeze, qualification)
    assert any(
        "reserve qualification, calibration, or red-team audit is invalid" in error
        for error in _bar_schema_errors(changed)
    )


def test_bar_schema_binds_control_q95_feasibility_to_its_report():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    receipt = changed["control_support"]["matrix"]["deterministic_linkage"][
        "gates"
    ]["exposures_and_rates"]["per_world"]["qual-0"][
        "reserve_q95_feasibility"
    ]
    receipt["q95_sum"] = 50.0
    receipt["total_minus_q95_sum"] = 50.0
    assert any(
        "separation receipt is incomplete" in error
        for error in _bar_schema_errors(changed)
    )

    changed = deepcopy(bars)
    changed["control_support"]["matrix"]["deterministic_linkage"]["gates"][
        "tail_calibration"
    ] = {}
    assert any(
        "tail_calibration/deterministic_linkage" in error
        for error in _bar_schema_errors(changed)
    )


def test_bar_schema_rejects_wrong_pairs_and_malformed_resample_digests_without_crashing():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    wrong_pair = deepcopy(bars)
    provenance = wrong_pair["evidence_provenance"]
    row = provenance["reference_reports"][0]
    row["world"] = "qual-1"
    row["evidence_id"] = freeze._canonical_digest({
        key: value for key, value in row.items() if key != "evidence_id"
    })
    _signed(freeze, provenance)
    assert any(
        "replay-bound evidence provenance" in error
        for error in _bar_schema_errors(wrong_pair)
    )

    malformed = deepcopy(bars)
    provenance = malformed["evidence_provenance"]
    row = provenance["replicate_reports"][0]
    row["resample_digest_sha256"] = "not-a-sha256"
    row["evidence_id"] = freeze._canonical_digest({
        key: value for key, value in row.items() if key != "evidence_id"
    })
    _signed(freeze, provenance)
    assert any(
        "replay-bound evidence provenance" in error
        for error in _bar_schema_errors(malformed)
    )

    bad_digest = deepcopy(bars)
    bad_digest["evidence_provenance"]["digest_sha256"] = "0" * 64
    errors = _bar_schema_errors(bad_digest)
    assert "freeze receipt lacks a valid replay-bound evidence provenance" in errors


def test_malformed_red_team_rows_fail_closed_instead_of_raising():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    red_team = kwargs["reserve_red_team_audit"]
    red_team["public_quantities"]["development"].append("bad-row")
    _signed(freeze, red_team)

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("reserve red-team development worlds differ" in blocker
               for blocker in bars["blockers"])

    for audit_name, block_name in (
        ("reserve_calibration_audit", "evidence"),
        ("reserve_qualification_audit", "reference_results"),
    ):
        kwargs = _calibration_kwargs(freeze, references, controls)
        audit = kwargs[audit_name]
        audit[block_name][0]["reference_line"] = []
        _signed(freeze, audit)
        bars = freeze.calibrate_composite_bars(
            references, replicates, controls, **kwargs
        )
        assert bars["frozen"] is False
        assert any(
            "reference_line" in blocker
            or "pairs differ" in blocker
            or "identities differ" in blocker
            for blocker in bars["blockers"]
        )

    kwargs = _calibration_kwargs(freeze, references, controls)
    qualification = kwargs["reserve_qualification_audit"]
    qualification["reference_results"][0]["extra"] = "not registered"
    _signed(freeze, qualification)
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("result fields differ" in blocker for blocker in bars["blockers"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    calibration = kwargs["reserve_calibration_audit"]
    calibration["rate_grid"] = "1.0"
    _signed(freeze, calibration)
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("rate_grid must be numeric" in blocker for blocker in bars["blockers"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    calibration = kwargs["reserve_calibration_audit"]
    calibration["evidence"][0]["reference_line"] = " A "
    _signed(freeze, calibration)
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("identities are not canonical" in blocker for blocker in bars["blockers"])


def test_bar_schema_handles_malformed_gate_and_audit_shapes_without_raising():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    for malformed_bars in ([], "bad", 1):
        assert _bar_schema_errors(malformed_bars) == [
            "composite bars must be a JSON object"
        ]

    for malformed_gates in ([], {**bars["gates"], "reserve_skill": "bad"}):
        changed = deepcopy(bars)
        changed["gates"] = malformed_gates
        assert _bar_schema_errors(changed)

    changed = deepcopy(bars)
    changed["gates"]["release_accuracy"]["components"][
        "p95_relative_error"
    ]["replicate_evidence_ids"][0] = []
    assert _bar_schema_errors(changed)

    for audit_name, block_name in (
        ("calibration", "evidence"),
        ("qualification", "reference_results"),
    ):
        changed = deepcopy(bars)
        audit = changed["reserve_audits"][audit_name]
        audit[block_name][0]["reference_line"] = []
        _signed(freeze, audit)
        if audit_name == "calibration":
            qualification = changed["reserve_audits"]["qualification"]
            qualification["calibration_audit_digest_sha256"] = audit["digest_sha256"]
            _signed(freeze, qualification)
        assert _bar_schema_errors(changed)
