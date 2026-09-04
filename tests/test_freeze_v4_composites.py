"""Synthetic tests for the version-four composite bar freeze."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from copy import deepcopy
from functools import lru_cache
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
            "coverage_deviation": 0.12 + offset,
            "mean_interval_score": 0.15 + offset,
        },
        "tail_calibration": {
            "pooled_exceedance_deviation": 0.04 + offset,
            "q95_width_relative_error": 0.20 + offset,
            "es95_width_relative_error": 0.22 + offset,
        },
        "reserve_skill": {
            "skill_loss": 0.18 + offset,
            "worst_regional_shortfall_probability": 0.16 + offset,
        },
    }


@lru_cache(maxsize=1)
def _shock_source_digest() -> str:
    from meridia.packet import continuation_source_law_digest

    return continuation_source_law_digest()


def _shock_runtime() -> dict:
    schedules = [
        {
            "member": 0,
            "future_shocks": [{
                "year": 10,
                "kind": "mortality_spike",
                "mortality_multiplier": 2.0,
                "admission_multiplier": 2.0,
            }],
        },
        {"member": 1, "future_shocks": []},
        {
            "member": 2,
            "future_shocks": [{
                "year": 11,
                "kind": "migration_wave",
                "leave_home_multiplier": 2.0,
            }],
        },
        {"member": 3, "future_shocks": []},
    ]
    return {
        "schema": "meridia.v4.continuation-shock-redraw.v1",
        "continuation_source_law_sha256": _shock_source_digest(),
        "member_count": 4,
        "redrawn_member_count": 4,
        "first_future_year": 10,
        "future_year_count": 5,
        "future_year_opportunity_count": 20,
        "member_schedules": schedules,
        "ordered_member_schedule_digest_sha256": hashlib.sha256(json.dumps(
            schedules, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "distinct_future_schedule_count": 3,
        "future_shock_year_count": 2,
        "future_mortality_spike_year_count": 1,
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

    packet_digest = _digest(f"packet-{world}")
    submission_digest = _digest(submission_label)
    liability_digest = _digest(f"liability-{world}")
    shock_runtime = _shock_runtime()
    shock_file_digest = hashlib.sha256((json.dumps(
        shock_runtime, indent=1, sort_keys=True, allow_nan=False,
    ) + "\n").encode()).hexdigest()
    after = submission_label.startswith("final-submission-C-")
    report = {
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
        "reserve_tail_evidence": {
            "schema": "meridia.v4.reserve-tail-evidence.v1",
            "valid": True,
            "q95_sum": 60.0,
            "es95_sum": 80.0,
            "reserve_submission_sha256": _digest(
                f"reserve-file-{submission_label}"
            ),
        },
        "reserve_rule_evidence": {
            "valid": True,
            "selected_year": 8,
            "exposure_person_years": 100.0,
            "rate_per_person_year": 1.0,
            "rounding_unit": 10.0,
            "reserve_total": 100.0,
            "experience_sha256": _digest(f"experience-{world}"),
        },
        "evidence": {
            "schema": "meridia.v4.verifier-evidence.v1",
            "packet_digest_sha256": packet_digest,
            "contract_digest_sha256": _digest(f"contract-{world}"),
            "submission_digest_sha256": submission_digest,
            "submission_file_sha256": {
                "reserve.csv": _digest(f"reserve-file-{submission_label}"),
            },
            "packet_file_sha256": {
                "participant/contract.json": _digest(f"contract-{world}"),
                "participant/experience_history.csv": _digest(
                    f"experience-{world}"
                ),
                "retained/continuation_liabilities.npz": _digest(
                    f"liability-{world}"
                ),
                "retained/continuation_shock_redraw.json": shock_file_digest,
            },
            "verifier_digest_sha256": _digest("verifier"),
        },
    }
    report["continuation_shock_redraw_evidence"] = {
        "schema": "meridia.v4.continuation-shock-redraw-report.v1",
        "runtime_evidence_file_sha256": shock_file_digest,
        "liability_archive_sha256": liability_digest,
        "runtime_evidence": shock_runtime,
    }
    report["elder_reference_evidence"] = {
        "schema": "meridia.v4.elder-reference-evidence.v1",
        "valid": True,
        "packet_digest_sha256": packet_digest,
        "submission_digest_sha256": submission_digest,
        "state_65_plus_person_years": [
            {
                "state": state,
                "submitted_person_years": 105.0 if after else 110.0,
                "sealed_person_years": 100.0,
            }
            for state in range(6)
        ],
        "liability_mean_by_region": [
            {
                "region": region,
                "submitted": 950.0 if after else 900.0,
                "sealed": 1_000.0,
            }
            for region in range(6)
        ],
    }
    return report


def _reference(freeze, line: str, world: str, *, offset: float = 0.0) -> dict:
    entry = {
        "reference_line": line,
        "world": world,
        "method_digest_sha256": _digest(f"reference-method-{line}"),
        "runner_digest_sha256": _digest("runner"),
        "measurement_contract_digest_sha256": _digest("measurement-contract"),
        "measurement_params": dict(freeze.REGISTERED_MEASUREMENT_PARAMS),
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
        "measurement_params": dict(freeze.REGISTERED_MEASUREMENT_PARAMS),
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
        "measurement_params": dict(freeze.REGISTERED_MEASUREMENT_PARAMS),
        "run_receipt_digest_sha256": _digest(f"control-receipt-{name}-{world}"),
        "deterministic": True,
        "report": report,
    }
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind="control")
    return entry


def _evidence(freeze, replicates_per_pair: int = 17):
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
        "measurement_params": dict(freeze.REGISTERED_MEASUREMENT_PARAMS),
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
        improvement = -0.02 + 0.015 * index
        history_rate = 0.01 + 0.0001 * index
        horizon_rate = 0.009 + 0.0002 * index
        observed_ratio = horizon_rate / history_rate
        trend_ratio = (1.0 - improvement) ** 5.0
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
                "hidden_mortality_improvement": improvement,
                "history_mortality_rate": history_rate,
                "horizon_mortality_rate": horizon_rate,
                "observed_horizon_to_history_ratio": observed_ratio,
                "trend_only_horizon_to_history_ratio": trend_ratio,
                "residual_observed_to_trend_ratio": observed_ratio / trend_ratio,
                "publication_lag_trend_factor": 1.0 - improvement,
                "trend_active_during_public_experience_window": True,
                "trend_starts_only_after_public_window": False,
                "trend_application": "all event months relative to the snapshot tick",
                "publication_lag_months": 12,
                "last_exposure_midpoint_to_snapshot_months": 18,
                "last_exposure_midpoint_to_snapshot_trend_factor": (
                    (1.0 - improvement) ** 1.5
                ),
                "continuation_shocks_redrawn_per_member": True,
                "history_mortality_shock_years": [4] if index % 3 == 0 else [],
                "lag_mortality_shock_years": [9] if index == 1 else [],
                "designated_horizon_mortality_shock_years": (
                    [10 + index] if index % 2 else []
                ),
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
        raw_hidden = (
            list(development)
            if constrained
            else list(freeze.PUBLIC_AXIS_RANGES[axis])
        )
        realized_development = list(
            freeze.REALIZED_MECHANISM_ENVELOPES[axis]["development"]
        )
        realized_hidden = (
            list(realized_development)
            if constrained
            else list(freeze.REALIZED_MECHANISM_ENVELOPES[axis]["public"])
        )
        axes[axis] = {
            "statistic": f"participant statistic for {axis}",
            "expected_sign": freeze.REGIME_EXPECTED_SIGNS[axis],
            "signed_rank_correlation": correlations[axis],
            "within_regime_signed_rank_correlation": {
                "development": 0.5,
                "hidden": 0.5,
            },
            "correlation_target": "realized_mechanism",
            "realized_mechanism_definition": (
                freeze.REALIZED_MECHANISM_DEFINITIONS[axis]
            ),
            "axis_intensity_range_observed": {
                "pooled": [
                    min(development[0], raw_hidden[0]),
                    max(development[1], raw_hidden[1]),
                ],
                "development": development,
                "hidden": raw_hidden,
            },
            "realized_mechanism_range_observed": {
                "pooled": [
                    min(realized_development[0], realized_hidden[0]),
                    max(realized_development[1], realized_hidden[1]),
                ],
                "development": realized_development,
                "hidden": realized_hidden,
            },
            "registered_realized_mechanism_envelopes": {
                family: list(bounds)
                for family, bounds in
                freeze.REALIZED_MECHANISM_ENVELOPES[axis].items()
            },
            "anchor_correlation_qualified": correlations[axis] > 0.4,
            "hidden_regime_correlation_qualified": True,
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
    binding = min(correlations, key=lambda axis: correlations[axis])
    audit = {
        "schema": freeze.REGIME_IDENTIFIABILITY_SCHEMA,
        "anchor_correlation_threshold": 0.4,
        "binding_axis": {
            "axis": binding,
            "signed_rank_correlation": correlations[binding],
        },
        "hidden_regime_correlation_shortfalls": [],
        "world_count": 18,
        "world_bindings": bindings,
        "measurement_rows_digest_sha256": _digest("identifiability-measurements"),
        "generator_source_digest_sha256": freeze._canonical_digest([
            {
                "path": relative,
                "sha256": hashlib.sha256(
                    (Path(freeze.__file__).resolve().parents[1] / relative).read_bytes()
                ).hexdigest(),
            }
            for relative in freeze.IDENTIFIABILITY_SOURCE_FILES
        ]),
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


def _mortality_audit(freeze, references: list[dict], regime: dict) -> dict:
    elder = _elder_audit(freeze, references)
    decompositions = {
        row["world"]: row["mortality_gap_decomposition"] for row in elder["worlds"]
    }
    bindings = {row["world"]: row for row in regime["world_bindings"]}
    reference_by_world = {}
    for reference in references:
        reference_by_world.setdefault(reference["world"], []).append(reference)
    worlds = []
    for world in freeze.QUALIFICATION_WORLDS:
        rows = reference_by_world[world]
        packet_inputs = rows[0]["report"]["evidence"]["packet_file_sha256"]
        worlds.append({
            "world": world,
            "packet_manifest_digest_sha256": bindings[world][
                "packet_manifest_digest_sha256"
            ],
            "packet_input_sha256": {
                name: packet_inputs[name] for name in freeze.RED_TEAM_INPUT_FILES
            },
            "reference_evidence_ids": {
                row["reference_line"]: row["evidence_id"] for row in rows
            },
            "shock_redraw_evidence": deepcopy(rows[0]["report"][
                "continuation_shock_redraw_evidence"
            ]),
            "decomposition": decompositions[world],
        })
    lag_effects = [
        100.0 * (row["publication_lag_trend_factor"] - 1.0)
        for row in decompositions.values()
    ]
    source = Path(__file__).resolve().parents[1] / "meridia/methods/phase_three.py"
    audit = {
        "schema": freeze.MORTALITY_IDENTIFICATION_AUDIT_SCHEMA,
        "supports_gate": "tail_calibration",
        "measurement_source": {
            "file": "meridia/methods/phase_three.py",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "function": "mortality_gap_decomposition",
        },
        "qualification_worlds": list(freeze.QUALIFICATION_WORLDS),
        "summary": {
            "trend_active_during_public_experience_window": True,
            "trend_starts_only_after_publication": False,
            "publication_lag_months": [12],
            "publication_lag_trend_effect_percent_range": [
                min(lag_effects), max(lag_effects)
            ],
            "shock_annual_probability": 0.20,
            "continuation_shocks_redrawn_per_member": True,
        },
        "worlds": worlds,
    }
    audit["digest_sha256"] = freeze._canonical_digest(audit)
    return audit


def _digest_bound(freeze, payload: dict) -> dict:
    payload.pop("digest_sha256", None)
    payload["digest_sha256"] = freeze._canonical_digest(payload)
    return payload


def _reserve_audits(freeze, references: list[dict], controls: list[dict],
                    diagnostics: list[dict]):
    calibration = {
        "schema": freeze.RESERVE_CALIBRATION_SCHEMA,
        "candidate": True,
        "accepted": False,
        "blockers": list(freeze.RESERVE_CALIBRATION_PENDING_BLOCKERS),
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
                "exposure_person_years": 100.0,
                "rounding_unit": 10.0,
                "submitted_q95_sum": 60.0,
                "submitted_es95_sum": 80.0,
                "target_reserve_before_rounding": 65.0,
                "required_rate": 0.65,
                "experience_sha256": row["report"]["reserve_rule_evidence"]
                ["experience_sha256"],
                "reserve_submission_sha256": row["report"]["evidence"]
                ["submission_file_sha256"]["reserve.csv"],
                "candidate_reserve_total": 100.0,
                "candidate_margin": 35.0,
            }
            for row in references
        ],
    }
    def packet_inputs(entries: list[dict], worlds: tuple[str, ...]) -> list[dict]:
        return [
            {
                "world": world,
                "file_sha256": {
                    name: next(
                        row["report"]["evidence"]["packet_file_sha256"]
                        for row in entries if row["world"] == world
                    )[name]
                    for name in freeze.RED_TEAM_INPUT_FILES
                },
            }
            for world in worlds
        ]

    def models() -> list[dict]:
        return [
            {
                "region": region,
                "intercept": 1.0,
                "reserve_total_coefficient": 0.5,
            }
            for region in range(6)
        ]

    red_team = {
        "schema": freeze.RESERVE_RED_TEAM_SCHEMA,
        "measurement_source": {
            "file": "scripts/red_team_reserve_total.py",
            "sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1]
                 / "scripts/red_team_reserve_total.py").read_bytes()
            ).hexdigest(),
        },
        "input_bindings": {
            "development": packet_inputs(diagnostics, freeze.DEVELOPMENT_WORLDS),
            "qualification": packet_inputs(references, freeze.QUALIFICATION_WORLDS),
        },
        "independent_unit": "world",
        "world_counts": {"development": 12, "qualification": 6, "total": 18},
        "regions_per_world": 6,
        "files_read_per_world": [
            "participant/contract.json",
            "participant/experience_history.csv",
            "retained/continuation_liabilities.npz:liability",
        ],
        "reserve_total_public_rule_verified": True,
        "tail_definition": {
            "level": 0.95,
            "quantile_rank": "ceil(level * members), one-indexed",
            "expected_shortfall": (
                "mean of all members at or above the quantile, ties included"
            ),
        },
        "primary_measure": (
            "qualification incremental regional R2 over development region means"
        ),
        "public_quantities": {
            "development": [
                {
                    "world": world,
                    "latest_year_total_exposure": 100.0,
                    "reserve_total": 100.0,
                }
                for world in freeze.DEVELOPMENT_WORLDS
            ],
            "qualification": [
                {
                    "world": world,
                    "latest_year_total_exposure": 100.0,
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
        "development_regional_models": {
            "q95": models(),
            "es95": models(),
        },
        "qualification_predictive_regional_r2": {
            "q95": 0.10,
            "es95": 0.20,
            "per_region": {
                "q95": [0.10] * 6,
                "es95": [0.20] * 6,
            },
        },
        "descriptive_pooled_regional_r2": {
            "q95": 0.10,
            "es95": 0.20,
            "headline_max": 0.20,
            "models": {"q95": models(), "es95": models()},
        },
        "world_aggregate_tail_r2": {
            "qualification_predictive": {
                "q95": 0.10, "es95": 0.20, "headline_max": 0.20,
            },
            "descriptive_pooled": {
                "q95": 0.10, "es95": 0.20, "headline_max": 0.20,
            },
        },
        "interpretation": "Synthetic reserve-total red-team fixture.",
    }

    return None, calibration, red_team


def _calibrate(freeze, references, replicates, controls):
    return freeze.calibrate_composite_bars(
        references,
        replicates,
        controls,
        **_calibration_kwargs(freeze, references, controls),
    )


def _calibration_kwargs(freeze, references, controls) -> dict:
    diagnostics = _diagnostics(freeze)
    qualification, calibration, red_team = _reserve_audits(
        freeze, references, controls, diagnostics
    )
    regime = _regime_audit(freeze)
    return {
        "development_diagnostic_reports": diagnostics,
        "elder_reconstruction_audit": _elder_audit(freeze, references),
        "mortality_identification_audit": _mortality_audit(
            freeze, references, regime
        ),
        "regime_identifiability_audit": regime,
        "reserve_qualification_audit": qualification,
        "reserve_calibration_audit": calibration,
        "reserve_red_team_audit": red_team,
    }


def _rebind(freeze, entry: dict, kind: str) -> None:
    entry["evidence_id"] = freeze.evidence_id_for(entry, kind=kind)


def _make_replicate_packets_distinct_from_base(freeze, replicates: list[dict]) -> None:
    """Give each paired outer resample its own authenticated participant bytes."""
    by_pair = {}
    for entry in replicates:
        by_pair.setdefault((entry["world"], entry["replicate_id"]), []).append(entry)
    for (world, replicate_id), rows in by_pair.items():
        packet_digest = _digest(f"resampled-packet-{world}-{replicate_id}")
        experience_digest = _digest(f"resampled-experience-{world}-{replicate_id}")
        for entry in rows:
            report = entry["report"]
            report["evidence"]["packet_digest_sha256"] = packet_digest
            report["evidence"]["packet_file_sha256"][
                "participant/experience_history.csv"
            ] = experience_digest
            report["reserve_rule_evidence"]["experience_sha256"] = experience_digest
            _rebind(freeze, entry, "replicate")


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
    assert bars["replicate_report_count"] == 306
    assert bars["replicates_per_reference_line_and_world"] == 17
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
    assert record["sample_count_per_reference_line"] == 102
    assert record["order_statistic_rank_per_reference_line"] == 101
    assert record["calibration_method"] == "derived-from-joint-gate-max-severity"
    assert record["worlds"] == [f"qual-{index}" for index in range(6)]
    assert record["witnesses"] == ["A", "B", "C"]
    assert record["supporting_controls"] == [
        "deterministic_linkage", "ignore_health_selection", "informative_selection",
        "version_three_recipe",
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
            assert "joint per-reference-line false-fail rates:" in document
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
            "hidden regime 0.5, qualified True; "
            "disposition constrained_to_development_range"
        ) in document
        assert (
            "missingness_target_dependence: signed rank correlation -0.018; "
            "hidden regime 0.5, qualified True; "
            "disposition constrained_to_development_range"
        ) in document
        assert (
            "- binding axis on the pooled rule: missingness_target_dependence at -0.018"
        ) in document
        assert (
            "- anchored axes below the threshold within the six hidden worlds: none"
        ) in document
        assert "Control separation" in document
        assert "Authenticated evidence design" in document
        assert "reference results at per-line joint false-fail rates" in document
        assert "A/qual-0: pass; evidence" in document
        assert "deterministic_linkage [primary]: failed worlds" in document
        assert "q95 diagnostic margin" in document
        assert "unique run receipts: 480" in document
        assert "normal_tail [primary]: failed worlds qual-0, qual-1, qual-2, qual-3, qual-4, qual-5" \
            in document
        assert "absolute 65+ exposure error 10.0% before, 5.0% after" in document
        assert "pooled exceedance deviation 0.04 before, 0.04 after" in document
        assert "region 0 liability mean: 900.0 before, 950.0 after, 1000.0 sealed" \
            in document
        assert (
            "target marginal product over five gates and three graded worlds: "
            f"{bars['target_marginal_product']:.6f}"
        ) in document
        assert (
            "conservative achieved conditional marginal-rate product: "
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


def test_rehashed_elder_numbers_cannot_replace_authenticated_reference_values():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    forged_audit = kwargs["elder_reconstruction_audit"]
    row = forged_audit["worlds"][0]
    for state in row["state_65_plus_person_years"]:
        state["submitted_after"] = 120.0
    row["exposure_65_plus_absolute_error_percent"]["after"] = 20.0
    _digest_bound(freeze, forged_audit)

    rejected = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert rejected["frozen"] is False
    assert any("elder exposure values differ from authenticated reports" in blocker
               for blocker in rejected["blockers"])

    bars = _calibrate(freeze, references, replicates, controls)
    embedded = bars["elder_reconstruction_audit"]
    embedded_row = embedded["worlds"][0]
    embedded_row["liability_mean_by_region"][0]["submitted_after"] = 951.0
    _digest_bound(freeze, embedded)
    from meridia.verify import _bar_schema_errors
    assert "elder reconstruction qualification audit is invalid" in \
        _bar_schema_errors(bars)


def test_mortality_identification_is_measured_dynamically_and_cross_bound():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is True
    mortality = bars["mortality_identification_evidence"]
    assert mortality["schema"] == freeze.MORTALITY_IDENTIFICATION_AUDIT_SCHEMA
    assert mortality["worlds"][0]["decomposition"][
        "hidden_mortality_improvement"
    ] == pytest.approx(-0.02)
    assert "per_world" not in mortality

    forged = deepcopy(bars)
    forged_mortality = forged["mortality_identification_evidence"]
    forged_mortality["worlds"][0]["packet_input_sha256"][
        "participant/experience_history.csv"
    ] = _digest("other-experience")
    _digest_bound(freeze, forged_mortality)

    from meridia.verify import _bar_schema_errors
    assert "mortality identification evidence is invalid" in _bar_schema_errors(forged)

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs.pop("mortality_identification_audit")
    missing = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert missing["frozen"] is False
    assert any(freeze.MORTALITY_IDENTIFICATION_AUDIT_SCHEMA in blocker
               for blocker in missing["blockers"])


def test_rehashed_shock_claim_cannot_replace_measured_member_schedules():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    mortality = kwargs["mortality_identification_audit"]
    shock = mortality["worlds"][0]["shock_redraw_evidence"]
    runtime = shock["runtime_evidence"]
    for member in runtime["member_schedules"]:
        member["future_shocks"] = []
    runtime["ordered_member_schedule_digest_sha256"] = freeze._canonical_digest(
        runtime["member_schedules"]
    )
    runtime["distinct_future_schedule_count"] = 1
    runtime["future_shock_year_count"] = 0
    runtime["future_mortality_spike_year_count"] = 0
    shock["runtime_evidence_file_sha256"] = hashlib.sha256((json.dumps(
        runtime, indent=1, sort_keys=True, allow_nan=False,
    ) + "\n").encode()).hexdigest()
    _digest_bound(freeze, mortality)

    rejected = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert rejected["frozen"] is False
    assert any("shock redraw evidence" in blocker
               for blocker in rejected["blockers"])

    bars = _calibrate(freeze, references, replicates, controls)
    mortality = bars["mortality_identification_evidence"]
    shock = mortality["worlds"][0]["shock_redraw_evidence"]
    runtime = shock["runtime_evidence"]
    runtime["member_schedules"][0]["future_shocks"] = []
    runtime["ordered_member_schedule_digest_sha256"] = freeze._canonical_digest(
        runtime["member_schedules"]
    )
    runtime["distinct_future_schedule_count"] = 2
    runtime["future_shock_year_count"] = 1
    runtime["future_mortality_spike_year_count"] = 0
    shock["runtime_evidence_file_sha256"] = hashlib.sha256((json.dumps(
        runtime, indent=1, sort_keys=True, allow_nan=False,
    ) + "\n").encode()).hexdigest()
    _digest_bound(freeze, mortality)
    from meridia.verify import _bar_schema_errors
    assert "mortality identification evidence is invalid" in _bar_schema_errors(bars)


def test_reference_lines_must_share_the_same_measured_shock_runtime():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    report = references[0]["report"]
    shock = report["continuation_shock_redraw_evidence"]
    runtime = shock["runtime_evidence"]
    runtime["member_schedules"][2]["future_shocks"][0]["year"] = 12
    runtime["ordered_member_schedule_digest_sha256"] = freeze._canonical_digest(
        runtime["member_schedules"]
    )
    file_digest = hashlib.sha256((json.dumps(
        runtime, indent=1, sort_keys=True, allow_nan=False,
    ) + "\n").encode()).hexdigest()
    shock["runtime_evidence_file_sha256"] = file_digest
    report["evidence"]["packet_file_sha256"][
        "retained/continuation_shock_redraw.json"
    ] = file_digest
    _rebind(freeze, references[0], "reference")

    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("continuation_shock_redraw_evidence_digest_sha256" in blocker
               for blocker in bars["blockers"])


def test_freeze_rejects_stale_or_unbound_mortality_identification():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["mortality_identification_audit"]["worlds"][0][
        "packet_manifest_digest_sha256"
    ] = _digest("stale-packet-manifest")
    _digest_bound(freeze, kwargs["mortality_identification_audit"])
    unbound = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert unbound["frozen"] is False
    assert any("packet manifest is not cross-bound" in blocker
               for blocker in unbound["blockers"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["mortality_identification_audit"]["worlds"][0]["decomposition"][
        "observed_horizon_to_history_ratio"
    ] += 0.25
    _digest_bound(freeze, kwargs["mortality_identification_audit"])
    stale = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert stale["frozen"] is False
    assert any("elder mortality decomposition differs" in blocker
               for blocker in stale["blockers"])


def test_verifier_rejects_malformed_mortality_years_without_raising():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    bars["mortality_identification_evidence"]["worlds"][0]["decomposition"][
        "history_mortality_shock_years"
    ] = [{}]
    _digest_bound(freeze, bars["mortality_identification_evidence"])

    from meridia.verify import _bar_schema_errors
    assert "mortality identification evidence is invalid" in _bar_schema_errors(bars)


def _rebind_regime_audit(freeze, audit):
    audit["digest_sha256"] = freeze._canonical_digest({
        key: value for key, value in audit.items() if key != "digest_sha256"
    })
    return audit


def test_a_hidden_regime_reading_below_the_threshold_is_reported_and_does_not_refuse():
    """Six worlds is too few for a within-regime rank correlation to decide a freeze.

    The registered gate is the pooled eighteen-world correlation. The hidden block is
    measured on the same number and carried into the freeze report beside the axis it
    belongs to, so a reader sees where an anchor is weakest without a threshold being
    applied to six points.
    """
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    audit = _regime_audit(freeze)
    audit["axes"]["mortality_improvement"][
        "within_regime_signed_rank_correlation"]["hidden"] = -0.086
    audit["axes"]["mortality_improvement"][
        "hidden_regime_correlation_qualified"] = False
    audit["hidden_regime_correlation_shortfalls"] = ["mortality_improvement"]
    _rebind_regime_audit(freeze, audit)

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["regime_identifiability_audit"] = audit
    bars = freeze.calibrate_composite_bars(references, replicates, controls, **kwargs)

    assert not any("hidden-regime identifiability" in blocker
                   for blocker in bars["blockers"])
    assert bars["regime_identifiability_audit"][
        "hidden_regime_correlation_shortfalls"] == ["mortality_improvement"]

    report = freeze.render_freeze_report(bars)
    assert "mortality_improvement -0.086" in report
    assert "reported and do not decide" in report


def test_a_hidden_regime_shortfall_cannot_be_hidden_by_dropping_it_from_the_list():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    audit = _regime_audit(freeze)
    audit["axes"]["migration_age_pattern"][
        "within_regime_signed_rank_correlation"]["hidden"] = 0.257
    audit["axes"]["migration_age_pattern"][
        "hidden_regime_correlation_qualified"] = True
    _rebind_regime_audit(freeze, audit)

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["regime_identifiability_audit"] = audit
    bars = freeze.calibrate_composite_bars(references, replicates, controls, **kwargs)

    assert bars["frozen"] is False
    assert any("hidden-regime correlation qualification does not match" in blocker
               for blocker in bars["blockers"])


def test_the_receipt_has_to_name_the_axis_that_binds_the_pooled_rule():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    audit = _regime_audit(freeze)
    assert audit["binding_axis"] == {
        "axis": "missingness_target_dependence", "signed_rank_correlation": -0.018}
    audit["binding_axis"] = {
        "axis": "migration_age_pattern", "signed_rank_correlation": 0.53}
    _rebind_regime_audit(freeze, audit)

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["regime_identifiability_audit"] = audit
    bars = freeze.calibrate_composite_bars(references, replicates, controls, **kwargs)

    assert bars["frozen"] is False
    assert any("does not name the axis that binds the pooled rule" in blocker
               for blocker in bars["blockers"])


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


def test_identifiability_v2_separates_raw_policy_from_realized_interaction():
    freeze = _freeze()
    audit = _regime_audit(freeze)
    target = audit["axes"]["missingness_target_dependence"]
    target["axis_intensity_range_observed"] = {
        "development": [1.016875136531417, 1.016875136531417],
        "hidden": [0.25, 1.25],
        "pooled": [0.25, 1.25],
    }
    target["realized_mechanism_range_observed"] = {
        "development": [1.3322169380099145, 1.3322169380099145],
        "hidden": [0.386, 1.1947],
        "pooled": [0.386, 1.3322169380099145],
    }
    _digest_bound(freeze, audit)

    validated = freeze._validate_regime_identifiability_audit(audit)
    assert validated["axes"]["missingness_target_dependence"][
        "axis_intensity_range_observed"
    ]["development"] == [1.016875136531417, 1.016875136531417]


def test_identifiability_v2_rejects_raw_hidden_axis_outside_policy_band():
    freeze = _freeze()
    audit = _regime_audit(freeze)
    raw = audit["axes"]["missingness_target_dependence"][
        "axis_intensity_range_observed"
    ]
    raw["hidden"] = [1.31, 1.31]
    raw["pooled"] = [raw["development"][0], 1.31]
    _digest_bound(freeze, audit)

    with pytest.raises(freeze.EvidenceError, match="hidden raw axis intensity"):
        freeze._validate_regime_identifiability_audit(audit)


def test_identifiability_v2_rejects_changed_interaction_registration():
    freeze = _freeze()
    audit = _regime_audit(freeze)
    audit["axes"]["linkage_urban_gradient"][
        "registered_realized_mechanism_envelopes"
    ]["public"][1] += 0.01
    _digest_bound(freeze, audit)

    with pytest.raises(freeze.EvidenceError, match="differ from registration"):
        freeze._validate_regime_identifiability_audit(audit)


def test_verifier_rejects_identifiability_v2_raw_policy_tamper():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    raw = bars["regime_identifiability_audit"]["axes"][
        "missingness_target_dependence"
    ]["axis_intensity_range_observed"]
    raw["hidden"] = [1.31, 1.31]
    raw["pooled"] = [raw["development"][0], 1.31]
    _digest_bound(freeze, bars["regime_identifiability_audit"])

    from meridia.verify import _bar_schema_errors
    assert "regime identifiability and hidden-axis constraint evidence is invalid" \
        in _bar_schema_errors(bars)


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
    assert any("exactly 306" in blocker for blocker in bars["blockers"])


def test_too_few_replicates_cannot_claim_an_empirical_one_percent_tail():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze, replicates_per_pair=6)
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "exactly 306" in blocker
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
    references[0]["report"]["elder_reference_evidence"][
        "packet_digest_sha256"
    ] = _digest("tampered-packet")
    references[0]["evidence_id"] = freeze.evidence_id_for(
        references[0], kind="reference"
    )
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any(
        "qual-0: evidence disagrees on packet_digest_sha256" in blocker
        for blocker in bars["blockers"]
    )


def test_reserve_rule_experience_digest_must_match_packet_input_before_freeze():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    references[0]["report"]["reserve_rule_evidence"]["experience_sha256"] = \
        _digest("different-experience-input")

    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is False
    assert any(
        "reserve rule experience digest differs from the packet input digest" in blocker
        for blocker in bars["blockers"]
    )


@pytest.mark.parametrize("kind", ["reference", "replicate", "control", "diagnostic"])
def test_public_reserve_rule_must_agree_across_every_report_for_a_world(kind):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    collections = {
        "reference": references,
        "replicate": replicates,
        "control": controls,
        "diagnostic": kwargs["development_diagnostic_reports"],
    }
    target = collections[kind][0]
    target["report"]["reserve_rule_evidence"]["selected_year"] = 9
    _rebind(freeze, target, kind)

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert any(
        "evidence disagrees on reserve_rule_evidence" in blocker
        for blocker in bars["blockers"]
    )


def test_packet_input_map_must_agree_across_every_report_for_a_world():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    target = controls[0]
    changed_digest = _digest("different-experience-input")
    target["report"]["evidence"]["packet_file_sha256"][
        "participant/experience_history.csv"
    ] = changed_digest
    target["report"]["reserve_rule_evidence"][
        "experience_sha256"
    ] = changed_digest
    _rebind(freeze, target, "control")

    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is False
    assert any(
        "evidence disagrees on packet_input_sha256" in blocker
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


def test_no_component_carries_the_severity_of_a_component_on_another_scale():
    """Each component's bar is its own reference scale, not the gate's worst.

    With one shared normalizer the max-severity of a gate is whichever component has the
    largest raw numbers, and every other component of that gate is published at the same
    number. On the tail block that put the exceedance bar at its attainable ceiling.
    """
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    for gate, components in freeze.GATE_COMPONENTS.items():
        record = bars["gates"][gate]
        ceiling = record["severity_ceiling"]
        for component in components:
            normalizer = freeze.GATE_COMPONENT_NORMALIZERS[gate][component]
            assert record["components"][component]["normalizer"] == normalizer
            assert record["components"][component]["value"] \
                == pytest.approx(ceiling * normalizer)
        if len(components) > 1:
            values = {record["components"][c]["value"] for c in components}
            assert len(values) == len(components)


def test_a_bar_at_its_attainable_ceiling_stops_the_freeze():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    _, high = freeze.COMPONENT_RANGES[
        ("tail_calibration", "pooled_exceedance_deviation")]
    normalizer = freeze.GATE_COMPONENT_NORMALIZERS[
        "tail_calibration"]["pooled_exceedance_deviation"]
    for row in replicates:
        row["report"]["composite_metrics"]["tail_calibration"][
            "pooled_exceedance_deviation"] = high
        _rebind(freeze, row, "replicate")
    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is False
    assert any(
        "tail_calibration/pooled_exceedance_deviation" in blocker
        and "reaches its attainable ceiling" in blocker
        for blocker in bars["blockers"]
    )
    assert high / normalizer >= 1.0


def test_every_final_reference_must_clear_the_p99_bars():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    references[0]["report"]["composite_metrics"]["tail_calibration"] \
        ["es95_width_relative_error"] = 5.0
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


def test_joint_gate_p99_controls_the_component_union_per_reference_line():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    # One severity, reached on two different components of the same gate by two
    # different replicates. Each component's own p99 stays below it, and the published
    # bar is that severity carried back through the component's registered normalizer.
    normalizers = freeze.GATE_COMPONENT_NORMALIZERS["tail_calibration"]
    severity = 1.4
    line_a = [row for row in replicates if row["reference_line"] == "A"]
    line_a[0]["report"]["composite_metrics"]["tail_calibration"] \
        ["q95_width_relative_error"] = severity * normalizers["q95_width_relative_error"]
    line_a[1]["report"]["composite_metrics"]["tail_calibration"] \
        ["es95_width_relative_error"] = \
        severity * normalizers["es95_width_relative_error"]
    _rebind(freeze, line_a[0], "replicate")
    _rebind(freeze, line_a[1], "replicate")
    bars = _calibrate(freeze, references, replicates, controls)
    gate = bars["gates"]["tail_calibration"]
    assert bars["frozen"] is True
    assert gate["sample_count_per_reference_line"] == 102
    assert gate["order_statistic_rank_per_reference_line"] == 101
    assert gate["reference_line_calibration"]["A"]["severity_p99"] == severity
    assert gate["components"]["q95_width_relative_error"][
        "empirical_p99_by_reference_line"
    ]["A"] < severity * normalizers["q95_width_relative_error"]
    assert gate["components"]["es95_width_relative_error"][
        "empirical_p99_by_reference_line"
    ]["A"] < severity * normalizers["es95_width_relative_error"]
    for component, normalizer in normalizers.items():
        assert gate["components"][component]["value"] \
            == pytest.approx(severity * normalizer)


def test_one_percent_claim_is_reported_independently_for_each_reference_line():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    line_a = next(row for row in replicates if row["reference_line"] == "A")
    line_a["report"]["composite_metrics"]["release_accuracy"][
        "p95_relative_error"
    ] = 0.7
    _rebind(freeze, line_a, "replicate")

    bars = _calibrate(freeze, references, replicates, controls)

    expected = 1.0 / 102.0
    assert bars["frozen"] is True
    assert bars["achieved_false_fail_rates_by_reference_line"]["A"][
        "release_accuracy"
    ] == pytest.approx(expected)
    assert bars["achieved_false_fail_rates_by_reference_line"]["B"][
        "release_accuracy"
    ] == 0.0
    assert bars["achieved_false_fail_rates"]["release_accuracy"] \
        == pytest.approx(expected)
    assert bars["achieved_marginal_rate_product"] == pytest.approx(
        min(bars["achieved_marginal_rate_product_by_reference_line"].values())
    )


def test_verifier_rejects_joint_calibration_or_per_line_rate_tampering():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    changed["gates"]["interval_quality"]["severity_ceiling"] += 0.01
    assert any(
        "interval_quality" in error and "calibration" in error
        for error in _bar_schema_errors(changed)
    )

    changed = deepcopy(bars)
    changed["achieved_false_fail_rates_by_reference_line"]["A"][
        "tail_calibration"
    ] = 0.01
    assert any(
        "tail_calibration" in error and "calibration" in error
        for error in _bar_schema_errors(changed)
    )


def test_freeze_binding_requires_registered_final_measurement_parameters():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    references[0]["measurement_params"]["simulation_paths"] = 128
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("100/400/2048/12" in blocker for blocker in bars["blockers"])

    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    changed["evidence_provenance"]["reference_reports"][0][
        "measurement_params"
    ]["simulation_paths"] = 128
    assert any("evidence provenance" in error for error in _bar_schema_errors(changed))


def test_cli_writes_an_incomplete_bar_set_when_only_packet_paths_are_given(tmp_path):
    freeze = _freeze()
    out = tmp_path / "bars"
    out.mkdir()
    (out / "reserve_calibration_accepted.json").write_text("stale")
    (out / "reserve_qualification_audit.json").write_text("stale")
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
    assert not (out / "reserve_calibration_accepted.json").exists()
    assert not (out / "reserve_qualification_audit.json").exists()


def test_cli_writes_promoted_reserve_audits_as_standalone_receipts(tmp_path):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    manifest = {
        "schema": freeze.EVIDENCE_SCHEMA,
        "reference_reports": references,
        "replicate_reports": replicates,
        "control_reports": controls,
        "development_diagnostic_reports": kwargs[
            "development_diagnostic_reports"
        ],
        "elder_reconstruction_audit": kwargs["elder_reconstruction_audit"],
        "mortality_identification_audit": kwargs[
            "mortality_identification_audit"
        ],
        "regime_identifiability_audit": kwargs[
            "regime_identifiability_audit"
        ],
        "reserve_calibration_audit": kwargs["reserve_calibration_audit"],
        "reserve_red_team_audit": kwargs["reserve_red_team_audit"],
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(manifest))
    out = tmp_path / "bars"

    assert freeze.main(["--evidence", str(evidence), "--out", str(out)]) == 0

    bars = json.loads((out / "bars.json").read_text())
    accepted = json.loads((out / "reserve_calibration_accepted.json").read_text())
    qualification = json.loads((out / "reserve_qualification_audit.json").read_text())
    assert accepted == bars["reserve_audits"]["calibration"]
    assert qualification == bars["reserve_audits"]["qualification"]


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
    mismatched = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert mismatched["frozen"] is False
    assert any(
        "calibration q95 sum differs from the verifier" in blocker
        for blocker in mismatched["blockers"]
    )

    references, replicates, controls = _evidence(freeze)
    controls[0]["report"]["reserve_q95_feasibility"][
        "all_regions_at_or_above_q95"
    ] = False
    controls[0]["report"]["reserve_q95_feasibility"]["feasible"] = False
    unbound = _calibrate(freeze, references, replicates, controls)
    assert unbound["frozen"] is False
    assert any("evidence_id does not match" in blocker for blocker in unbound["blockers"])

    _rebind(freeze, controls[0], "control")
    diagnostic_only = _calibrate(freeze, references, replicates, controls)
    assert diagnostic_only["frozen"] is True


def test_valid_reserve_candidate_is_promoted_and_qualification_is_generated():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    candidate_digest = freeze._canonical_digest(
        kwargs["reserve_calibration_audit"]
    )

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is True
    calibration = bars["reserve_audits"]["calibration"]
    qualification = bars["reserve_audits"]["qualification"]
    assert calibration["accepted"] is True
    assert calibration["blockers"] == []
    assert calibration["candidate_source_digest_sha256"] == candidate_digest
    assert calibration["measurement_contract_digest_sha256"] \
        == _digest("measurement-contract")
    assert calibration["digest_sha256"] == freeze._canonical_digest({
        key: value for key, value in calibration.items() if key != "digest_sha256"
    })
    assert len(qualification["reference_results"]) == 18
    assert all(row["reserve_skill_pass"] is True
               for row in qualification["reference_results"])
    assert len(qualification["proportional_reserve_results"]) == 6
    assert all(row["reserve_skill_pass"] is False
               for row in qualification["proportional_reserve_results"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["reserve_qualification_audit"] = deepcopy(qualification)
    verified = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert verified["frozen"] is True
    assert verified["reserve_audits"]["qualification"] == qualification


@pytest.mark.parametrize(
    ("field", "value"),
    [("rate_grid", 0.5), ("tail_slack_share", 0.0)],
)
def test_reserve_candidate_cannot_change_registered_calibration_constants(field, value):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["reserve_calibration_audit"][field] = value

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert any("RATE_GRID=1.0" in blocker for blocker in bars["blockers"])


def test_reserve_candidate_es95_must_match_authenticated_tail_evidence():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    row = kwargs["reserve_calibration_audit"]["evidence"][0]
    row.update({
        "submitted_es95_sum": 100.0,
        "target_reserve_before_rounding": 70.0,
        "required_rate": 0.7,
        "candidate_margin": 30.0,
    })

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert any("candidate is infeasible" in blocker for blocker in bars["blockers"])


def test_reserve_candidate_requires_one_rounding_unit_across_final_reports():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    for entries, kind in (
        (references, "reference"),
        (replicates, "replicate"),
        (controls, "control"),
    ):
        for entry in entries:
            if entry["world"] == "qual-0":
                entry["report"]["reserve_rule_evidence"]["rounding_unit"] = 20.0
                _rebind(freeze, entry, kind)
    kwargs = _calibration_kwargs(freeze, references, controls)
    for row in kwargs["reserve_calibration_audit"]["evidence"]:
        if row["world"] == "qual-0":
            row["rounding_unit"] = 20.0

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert any("one reserve rounding unit" in blocker for blocker in bars["blockers"])


@pytest.mark.parametrize("mutation", ["r2", "input", "source", "quantity", "extra"])
def test_reserve_red_team_measurement_rejects_exploit_mutations(mutation):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    audit = kwargs["reserve_red_team_audit"]
    if mutation == "r2":
        primary = audit[
            "qualification_incremental_regional_r2_over_region_means"
        ]
        primary["q95"] = 1.01
        primary["headline_max"] = 1.01
    elif mutation == "input":
        audit["input_bindings"]["qualification"][0]["file_sha256"][
            "participant/contract.json"
        ] = _digest("unbound-packet-input")
    elif mutation == "source":
        audit["measurement_source"]["sha256"] = _digest("other-source")
    elif mutation == "quantity":
        audit["public_quantities"]["qualification"][0][
            "latest_year_total_exposure"
        ] = 101.0
    else:
        audit["unregistered"] = True

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    expected = {
        "r2": "no greater than one",
        "input": "packet inputs differ",
        "source": "measurement source differs",
        "quantity": "public quantities differ from verifier evidence",
        "extra": "measurement fields differ",
    }[mutation]
    assert any(expected in blocker for blocker in bars["blockers"])


def test_late_freeze_blocker_prevents_reserve_audit_promotion():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    passing = next(
        row for row in controls
        if row["control"] == "deterministic_linkage" and row["world"] == "qual-0"
    )
    passing["report"]["composite_metrics"]["exposures_and_rates"][
        "p95_relative_error"
    ] = 0.1
    _rebind(freeze, passing, "control")
    kwargs = _calibration_kwargs(freeze, references, controls)

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert "reserve_audits" not in bars
    assert any("deletion candidates" in blocker for blocker in bars["blockers"])


def test_preaccepted_reserve_candidate_cannot_bypass_freeze_promotion():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    candidate = kwargs["reserve_calibration_audit"]
    candidate["accepted"] = True
    candidate["blockers"] = []
    candidate["measurement_contract_digest_sha256"] = _digest(
        "measurement-contract"
    )
    _digest_bound(freeze, candidate)

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert any("canonical unaccepted candidate" in blocker
               for blocker in bars["blockers"])


@pytest.mark.parametrize("mutation", ["reserve_total", "reserve_file_digest"])
def test_reserve_candidate_values_must_match_authenticated_final_reports(mutation):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    row = kwargs["reserve_calibration_audit"]["evidence"][0]
    if mutation == "reserve_total":
        row["candidate_reserve_total"] = 110.0
        row["candidate_margin"] = 45.0
    else:
        row["reserve_submission_sha256"] = _digest("different-reserve-file")

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )

    assert bars["frozen"] is False
    assert any("candidate is infeasible" in blocker for blocker in bars["blockers"])


def test_reserve_candidate_stays_blocked_until_each_registered_check_passes():
    freeze = _freeze()

    references, replicates, controls = _evidence(freeze)
    failed_reference = references[0]
    failed_reference["report"]["composite_metrics"]["reserve_skill"] \
        ["skill_loss"] = 0.9
    _rebind(freeze, failed_reference, "reference")
    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["reserve_calibration_audit"]["evidence"][0]["evidence_id"] \
        = failed_reference["evidence_id"]
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("final references fail reserve_skill" in blocker
               for blocker in bars["blockers"])

    references, replicates, controls = _evidence(freeze)
    passing_control = next(
        row for row in controls
        if row["control"] == "proportional_reserve" and row["world"] == "qual-0"
    )
    passing_control["report"]["composite_metrics"]["reserve_skill"] \
        ["skill_loss"] = 0.1
    _rebind(freeze, passing_control, "control")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is False
    assert any("proportional_reserve passes reserve_skill" in blocker
               for blocker in bars["blockers"])

    references, replicates, controls = _evidence(freeze)
    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["reserve_red_team_audit"] = None
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any(freeze.RESERVE_RED_TEAM_SCHEMA in blocker
               for blocker in bars["blockers"])

    references, replicates, controls = _evidence(freeze)
    below_q95 = references[0]
    below_q95["report"]["reserve_q95_feasibility"][
        "all_regions_at_or_above_q95"
    ] = False
    below_q95["report"]["reserve_q95_feasibility"]["feasible"] = False
    _rebind(freeze, below_q95, "reference")
    bars = _calibrate(freeze, references, replicates, controls)
    assert bars["frozen"] is True
    result = next(
        row for row in bars["reserve_audits"]["qualification"]["reference_results"]
        if row["reference_line"] == "A" and row["world"] == "qual-0"
    )
    assert result["q95_feasible"] is False


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


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("packet_input", "freeze evidence packet input binding differs"),
        ("reserve_rule", "freeze evidence reserve rule binding differs"),
    ],
)
def test_bar_schema_rejects_split_world_packet_and_reserve_rule_evidence(
    mutation, expected_error
):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    provenance = changed["evidence_provenance"]
    row = provenance["control_reports"][0]
    if mutation == "packet_input":
        changed_digest = _digest("different-experience-input")
        row["packet_input_sha256"][
            "participant/experience_history.csv"
        ] = changed_digest
        row["reserve_rule_evidence"]["experience_sha256"] = changed_digest
    else:
        row["reserve_rule_evidence"]["selected_year"] = 9
    row["reserve_rule_evidence_digest_sha256"] = freeze._canonical_digest(
        row["reserve_rule_evidence"]
    )
    row["evidence_id"] = freeze._canonical_digest({
        key: value for key, value in row.items() if key != "evidence_id"
    })
    _digest_bound(freeze, provenance)

    assert any(
        expected_error in error for error in _bar_schema_errors(changed)
    )


def test_paired_outer_resamples_may_change_participant_bytes_by_replicate():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    _make_replicate_packets_distinct_from_base(freeze, replicates)

    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is True
    from meridia.verify import _bar_schema_errors
    assert _bar_schema_errors(bars) == []


def test_paired_outer_resample_requires_one_packet_binding_across_lines():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    _make_replicate_packets_distinct_from_base(freeze, replicates)
    row = next(
        entry for entry in replicates
        if entry["reference_line"] == "B"
        and entry["world"] == "qual-0"
        and entry["replicate_id"] == "0"
    )
    row["report"]["evidence"]["packet_digest_sha256"] = _digest(
        "wrong-paired-packet"
    )
    _rebind(freeze, row, "replicate")

    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is False
    assert any(
        "qual-0/0: evidence disagrees on packet_digest_sha256" in blocker
        for blocker in bars["blockers"]
    )


def test_paired_outer_resample_cannot_change_fixed_public_quantities():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    _make_replicate_packets_distinct_from_base(freeze, replicates)
    for row in replicates:
        if row["world"] == "qual-0" and row["replicate_id"] == "0":
            rule = row["report"]["reserve_rule_evidence"]
            rule["rate_per_person_year"] = 2.0
            rule["reserve_total"] = 200.0
            _rebind(freeze, row, "replicate")

    bars = _calibrate(freeze, references, replicates, controls)

    assert bars["frozen"] is False
    assert any(
        "qual-0/0: resample changes reserve rule rate_per_person_year" in blocker
        for blocker in bars["blockers"]
    )


def test_bar_schema_rejects_reserve_rule_digest_not_bound_to_packet_input():
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    provenance = changed["evidence_provenance"]
    row = provenance["reference_reports"][0]
    row["reserve_rule_evidence"]["experience_sha256"] = _digest(
        "different-experience-input"
    )
    row["reserve_rule_evidence_digest_sha256"] = freeze._canonical_digest(
        row["reserve_rule_evidence"]
    )
    row["evidence_id"] = freeze._canonical_digest({
        key: value for key, value in row.items() if key != "evidence_id"
    })
    _digest_bound(freeze, provenance)

    assert "freeze receipt lacks a valid replay-bound evidence provenance" \
        in _bar_schema_errors(changed)


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


def test_bar_schema_rejects_rehashed_reserve_values_not_bound_to_run_evidence():
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
    _digest_bound(freeze, calibration)
    qualification["calibration_audit_digest_sha256"] = calibration["digest_sha256"]
    _digest_bound(freeze, qualification)
    assert any(
        "reserve qualification, calibration, or red-team audit is invalid" in error
        for error in _bar_schema_errors(changed)
    )


@pytest.mark.parametrize("mutation", ["r2", "input", "source", "quantity", "extra"])
def test_bar_schema_rejects_rehashed_red_team_exploit_mutations(mutation):
    freeze = _freeze()
    references, replicates, controls = _evidence(freeze)
    bars = _calibrate(freeze, references, replicates, controls)
    from meridia.verify import _bar_schema_errors

    changed = deepcopy(bars)
    red_team = changed["reserve_audits"]["red_team"]
    qualification = changed["reserve_audits"]["qualification"]
    if mutation == "r2":
        primary = red_team[
            "qualification_incremental_regional_r2_over_region_means"
        ]
        primary["q95"] = 1.01
        primary["headline_max"] = 1.01
    elif mutation == "input":
        red_team["input_bindings"]["qualification"][0]["file_sha256"][
            "participant/contract.json"
        ] = _digest("unbound-packet-input")
    elif mutation == "source":
        red_team["measurement_source"]["sha256"] = _digest("other-source")
    elif mutation == "quantity":
        red_team["public_quantities"]["qualification"][0][
            "latest_year_total_exposure"
        ] = 101.0
    else:
        red_team["unregistered"] = True
    _digest_bound(freeze, red_team)
    qualification["red_team_audit_digest_sha256"] = red_team["digest_sha256"]
    _digest_bound(freeze, qualification)

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
    _digest_bound(freeze, red_team)
    qualification["red_team_audit_digest_sha256"] = red_team["digest_sha256"]
    _digest_bound(freeze, qualification)
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
    _digest_bound(freeze, provenance)
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
    _digest_bound(freeze, provenance)
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

    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("reserve red-team development worlds differ" in blocker
               for blocker in bars["blockers"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    kwargs["reserve_calibration_audit"]["evidence"][0]["reference_line"] = []
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("reference_line" in blocker for blocker in bars["blockers"])

    accepted = _calibrate(freeze, references, replicates, controls)
    kwargs = _calibration_kwargs(freeze, references, controls)
    qualification = deepcopy(accepted["reserve_audits"]["qualification"])
    qualification["reference_results"][0]["extra"] = "not registered"
    _digest_bound(freeze, qualification)
    kwargs["reserve_qualification_audit"] = qualification
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("differs from the deterministic audit" in blocker
               for blocker in bars["blockers"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    calibration = kwargs["reserve_calibration_audit"]
    calibration["rate_grid"] = "1.0"
    bars = freeze.calibrate_composite_bars(
        references, replicates, controls, **kwargs
    )
    assert bars["frozen"] is False
    assert any("rate_grid must be numeric" in blocker for blocker in bars["blockers"])

    kwargs = _calibration_kwargs(freeze, references, controls)
    calibration = kwargs["reserve_calibration_audit"]
    calibration["evidence"][0]["reference_line"] = " A "
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
        _digest_bound(freeze, audit)
        if audit_name == "calibration":
            qualification = changed["reserve_audits"]["qualification"]
            qualification["calibration_audit_digest_sha256"] = audit["digest_sha256"]
            _digest_bound(freeze, qualification)
        assert _bar_schema_errors(changed)
