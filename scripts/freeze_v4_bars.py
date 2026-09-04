"""Freeze the five version-four composite gates from qualification evidence.

A completed freeze needs three separate kinds of evidence: one deterministic final
witness report for each reference line and qualification world, independently identified
deterministic replicate reports for every line-world pair, and reports from registered
scientific controls that pass the deterministic hard checks.

The reserve-rate input is the unaccepted candidate emitted by
``calibrate_reserve_rate.py``.  This freezer is the only component that can promote it:
promotion happens after the final references, proportional-reserve controls, calibrated
reserve-skill bar, authenticated q95 diagnostics, and red-team audit have all been verified.

Only replicate reports set the exact empirical p99 gate-severity ceilings. Calibration is
independent by reference line and joint across a gate's components. Final reports are never
bootstrapped or resampled as fake replication. Final witnesses must then clear the frozen
bars. The command line consumes JSON evidence and writes ``bars.json``, the two
standalone promoted reserve-audit receipts, ``freeze_report.txt``, and
``PROVENANCE.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any


SCHEMA = "meridia.v4.composite-bars.v1"
EVIDENCE_SCHEMA = "meridia.v4.composite-freeze-evidence.v1"
VERIFIER_EVIDENCE_SCHEMA = "meridia.v4.verifier-evidence.v1"
EVIDENCE_BINDING_SCHEMA = "meridia.v4.freeze-evidence-binding.v3"
PROVENANCE_SCHEMA = "meridia.v4.freeze-provenance.v1"
DEVELOPMENT_DIAGNOSTIC_SCHEMA = "meridia.v4.development-diagnostics.v1"
ELDER_AUDIT_SCHEMA = "meridia.methods.elder_reconstruction_audit.v1"
REGIME_IDENTIFIABILITY_SCHEMA = "meridia.v4.regime-identifiability-audit.v3"
MORTALITY_IDENTIFICATION_AUDIT_SCHEMA = (
    "meridia.v4.mortality-identification-audit.v1"
)
RESERVE_QUALIFICATION_SCHEMA = "meridia.v4.reserve-qualification-audit.v1"
RESERVE_CALIBRATION_SCHEMA = "meridia.reserve-rate-calibration.v2"
RESERVE_RED_TEAM_SCHEMA = "meridia.reserve-total-red-team.v1"
RESERVE_TAIL_EVIDENCE_SCHEMA = "meridia.v4.reserve-tail-evidence.v1"
ELDER_REFERENCE_EVIDENCE_SCHEMA = "meridia.v4.elder-reference-evidence.v1"
SHOCK_REDRAW_REPORT_SCHEMA = "meridia.v4.continuation-shock-redraw-report.v1"
RED_TEAM_INPUT_FILES = (
    "participant/contract.json",
    "participant/experience_history.csv",
    "retained/continuation_liabilities.npz",
)
RESERVE_CALIBRATION_PENDING_BLOCKERS = (
    "rerun every reference at the candidate rate and clear reserve skill",
    "show the proportional reserve control failing at the candidate rate",
    "record the held-out reserve-total red-team measurement",
)
RESERVE_CALIBRATION_CANDIDATE_KEYS = {
    "schema",
    "candidate",
    "accepted",
    "blockers",
    "rate_per_person_year",
    "rate_grid",
    "tail_slack_share",
    "target_rule",
    "reference_lines",
    "qualification_worlds",
    "evidence",
}
RESERVE_CALIBRATION_EVIDENCE_KEYS = {
    "reference_line",
    "world",
    "evidence_id",
    "exposure_person_years",
    "rounding_unit",
    "submitted_q95_sum",
    "submitted_es95_sum",
    "target_reserve_before_rounding",
    "required_rate",
    "experience_sha256",
    "reserve_submission_sha256",
    "candidate_reserve_total",
    "candidate_margin",
}
RESERVE_RED_TEAM_MEASUREMENT_KEYS = {
    "schema",
    "measurement_source",
    "input_bindings",
    "independent_unit",
    "world_counts",
    "regions_per_world",
    "files_read_per_world",
    "reserve_total_public_rule_verified",
    "tail_definition",
    "public_quantities",
    "development_regional_models",
    "qualification_predictive_regional_r2",
    "qualification_incremental_regional_r2_over_region_means",
    "primary_measure",
    "descriptive_pooled_regional_r2",
    "world_aggregate_tail_r2",
    "interpretation",
}
QUANTILE = 0.99
TARGET_FALSE_FAIL_RATE = 0.01
EXPECTED_QUALIFICATION_WORLDS = 6
GRADED_WORLD_COUNT = 3
MIN_P99_SAMPLE_COUNT = 100
REGISTERED_MEASUREMENT_PARAMS = {
    "bootstrap_replicates": 100,
    "bayesian_sweeps": 400,
    "simulation_paths": 2048,
    "linkage_bootstraps": 12,
}
ANCHOR_CORRELATION_THRESHOLD = 0.4
REFERENCE_LINES = ("A", "B", "C")
QUALIFICATION_WORLDS = tuple(
    f"qual-{index}" for index in range(EXPECTED_QUALIFICATION_WORLDS)
)
REPLICATES_PER_LINE_WORLD = 17
REFERENCE_REPORT_COUNT = len(REFERENCE_LINES) * len(QUALIFICATION_WORLDS)
REPLICATE_REPORT_COUNT = (
    REFERENCE_REPORT_COUNT * REPLICATES_PER_LINE_WORLD
)
DEVELOPMENT_WORLDS = tuple(f"dev-{index:02d}" for index in range(12))
DEVELOPMENT_DIAGNOSTICS = (
    "design_reconstruction_oracle_tail",
    "true_population_normal_tail",
)
DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT = (
    len(DEVELOPMENT_WORLDS) * len(DEVELOPMENT_DIAGNOSTICS)
)

REGIME_AXES = (
    "mortality_improvement",
    "migration_age_pattern",
    "age_reporting_error",
    "linkage_urban_gradient",
    "administrative_completeness",
    "missingness_target_dependence",
)
REGIME_EXPECTED_SIGNS = {
    "mortality_improvement": 1,
    "migration_age_pattern": 1,
    "age_reporting_error": 1,
    "linkage_urban_gradient": -1,
    "administrative_completeness": 1,
    "missingness_target_dependence": 1,
}
HIDDEN_IN_BAND_AXES = (
    "administrative_completeness",
    "missingness_target_dependence",
)
HIDDEN_EXTRAPOLATION_AXES = tuple(
    axis for axis in REGIME_AXES if axis not in HIDDEN_IN_BAND_AXES
)
DEVELOPMENT_AXIS_RANGES: dict[str, tuple[float, float]] = {
    "mortality_improvement": (-0.010, 0.048),
    "migration_age_pattern": (0.25, 1.55),
    "age_reporting_error": (0.70, 2.05),
    "linkage_urban_gradient": (0.30, 1.55),
    "administrative_completeness": (0.30, 1.70),
    "missingness_target_dependence": (0.20, 1.30),
}
PUBLIC_AXIS_RANGES: dict[str, tuple[float, float]] = {
    "mortality_improvement": (-0.030, 0.075),
    "migration_age_pattern": (0.00, 2.40),
    "age_reporting_error": (0.35, 3.40),
    "linkage_urban_gradient": (0.00, 2.60),
    "administrative_completeness": (0.00, 2.80),
    "missingness_target_dependence": (0.00, 2.20),
}
REALIZED_MECHANISM_ENVELOPES: dict[
    str, dict[str, tuple[float, float]]
] = {
    "mortality_improvement": {
        "development": (-0.010, 0.048),
        "public": (-0.030, 0.075),
    },
    "migration_age_pattern": {
        "development": (0.25, 1.55),
        "public": (0.00, 2.40),
    },
    "age_reporting_error": {
        "development": (0.596, 2.4248571428571424),
        "public": (0.298, 4.021714285714285),
    },
    "linkage_urban_gradient": {
        "development": (0.13125, 2.189375),
        "public": (0.0, 5.33),
    },
    "administrative_completeness": {
        "development": (0.30, 1.70),
        "public": (0.00, 2.80),
    },
    "missingness_target_dependence": {
        "development": (0.074, 2.119),
        "public": (0.0, 5.764),
    },
}
REALIZED_MECHANISM_DEFINITIONS = {
    axis: "axis_intensity" for axis in REGIME_AXES
}
REALIZED_MECHANISM_DEFINITIONS.update({
    "age_reporting_error": (
        "age_reporting_error * age_error_mortality_scale"
    ),
    "linkage_urban_gradient": (
        "linkage_urban_gradient * (1 + linkage_gradient_by_migration * "
        "(migration_age_pattern - 1))"
    ),
    "missingness_target_dependence": (
        "missingness_target_dependence * "
        "(1 + health_inclusion_completeness_by_target * "
        "(administrative_completeness - 1))"
    ),
})
IDENTIFIABILITY_SOURCE_FILES = (
    "scripts/identifiability_v4.py",
    "meridia/character.py",
    "meridia/events.py",
    "meridia/mechanisms.py",
    "meridia/packet.py",
    "meridia/sources.py",
    "scripts/build_sealed_reconstruction_packet.py",
    "scripts/build_v4_worlds.py",
)

GATE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "exposures_and_rates": ("p95_relative_error",),
    "release_accuracy": ("p95_relative_error",),
    "interval_quality": ("coverage_deviation", "mean_interval_score"),
    "tail_calibration": (
        "pooled_exceedance_deviation",
        "q95_width_relative_error",
        "es95_width_relative_error",
    ),
    "reserve_skill": ("skill_loss", "worst_regional_shortfall_probability"),
}

# A gate profile selects which of the five calibrated composites decide a verdict. It is
# a subset of GATE_COMPONENTS: no profile adds a gate, adds a component, or moves a bar.
# Every profile calibrates all five gates on the same replicate design and reports all
# five; the profile only says which ones decide. "lite" reports the tail block and
# decides on the population, exposure and rate, projection, and reserve-skill blocks.
DEFAULT_GATE_PROFILE = "full"
GATE_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "full": {gate: tuple(components) for gate, components in GATE_COMPONENTS.items()},
    "lite": {
        "exposures_and_rates": ("p95_relative_error",),
        "release_accuracy": ("p95_relative_error",),
        "interval_quality": ("coverage_deviation", "mean_interval_score"),
        "reserve_skill": ("skill_loss",),
    },
}

# Every component is a dimensionless loss with zero as its ideal value, but the components
# inside one gate are not on one scale. Pooled exceedance deviation lives in a band 0.95
# wide while the q95 and ES95 width errors on the same block run past ten, so a max-severity
# taken with equal weights is the width error alone and the exceedance component is carried
# to whatever ceiling the width error sets. Clipped at its attainable range that made the
# published exceedance bar exactly 0.95, a bar no submission can exceed and no tail control
# can fail. Interval coverage deviation against the mean interval score, and the worst
# regional shortfall probability against skill loss, sit the same way.
#
# Each component therefore carries the scale of its own reference distribution: the median
# of that component over the eighteen final reference reports on the six qualification
# worlds, three fixed-seed lines each. A severity of one is a typical reference report on
# that component, so the max over components is a comparison of like with like.
#
# These are registration-time constants, read once and written here. They are not
# recomputed from the sample a freeze calibrates on, which would let that sample tune the
# relative weights it is then judged by. A single-component gate is invariant to its
# normalizer, since the value divides out of the order statistic and multiplies back into
# the bar, so those two stay at one and their published bar is the order statistic itself.
GATE_COMPONENT_NORMALIZERS: dict[str, dict[str, float]] = {
    "exposures_and_rates": {"p95_relative_error": 1.0},
    "release_accuracy": {"p95_relative_error": 1.0},
    "interval_quality": {
        "coverage_deviation": 0.52,
        "mean_interval_score": 1.4523,
    },
    "tail_calibration": {
        "pooled_exceedance_deviation": 0.05,
        "q95_width_relative_error": 3.7149,
        "es95_width_relative_error": 4.3735,
    },
    "reserve_skill": {
        "skill_loss": 1.1793,
        "worst_regional_shortfall_probability": 0.3364,
    },
}

# Criterion ranges, not tunable attainability caps. An unbounded endpoint is null in JSON.
COMPONENT_RANGES: dict[tuple[str, str], tuple[float, float | None]] = {
    ("exposures_and_rates", "p95_relative_error"): (0.0, None),
    ("release_accuracy", "p95_relative_error"): (0.0, None),
    ("interval_quality", "coverage_deviation"): (0.0, 1.0),
    ("interval_quality", "mean_interval_score"): (0.0, None),
    ("tail_calibration", "pooled_exceedance_deviation"): (0.0, 0.95),
    ("tail_calibration", "q95_width_relative_error"): (0.0, None),
    ("tail_calibration", "es95_width_relative_error"): (0.0, None),
    ("reserve_skill", "skill_loss"): (0.0, None),
    ("reserve_skill", "worst_regional_shortfall_probability"): (0.0, 1.0),
}

# A name here denotes an omitted scientific layer or a deliberately wrong scientific
# method. An arbitrary submission cannot support a gate merely by calling itself a control.
# Every registered control must be run once on every qualification world and must fail
# every one of those runs at its single primary gate.  The registry includes the complete
# runnable V4 qualification battery.  Operator-only oracle decompositions remain
# diagnostics: using sealed truth makes them unsuitable as wrong participant methods.
SCIENTIFIC_CONTROLS_BY_GATE: dict[str, tuple[str, ...]] = {
    "exposures_and_rates": (
        "deterministic_linkage",
        "ignore_health_selection",
        "informative_selection",
        "version_three_recipe",
    ),
    "release_accuracy": (
        "register_only",
        "survey_only",
        "no_dedup",
        "static_projection",
        "benchmark_only",
        "exact_key_union",
        "experience_history_only",
    ),
    "interval_quality": (
        "inflated_intervals",
        "reconstruction_uncertainty",
    ),
    "tail_calibration": (
        "development_average_regime",
        "mean_only_tail",
        "normal_tail",
        "padded_tail",
        "regime_recombination",
        "predictive_tails",
    ),
    "reserve_skill": (
        "uniform_allocation",
        "reserve_allocation",
        "proportional_reserve",
    ),
}

REQUIRED_SCIENTIFIC_CONTROLS = tuple(sorted({
    control
    for controls in SCIENTIFIC_CONTROLS_BY_GATE.values()
    for control in controls
}))
CONTROL_REPORT_COUNT = len(REQUIRED_SCIENTIFIC_CONTROLS) * len(QUALIFICATION_WORLDS)

CORRELATION_CAVEAT = (
    "The marginal products assume independent gate and world failures. They are "
    "arithmetic summaries, not empirical pass probabilities; failures can be correlated."
)
FINITE_WORLD_CAVEAT = (
    "Only six qualification worlds support this freeze. Replicate false-fail rates are "
    "conditional on those worlds and do not establish a one-percent rate on new worlds."
)
TAIL_DEFINITION_LINES = (
    "q95 is order statistic ceil(0.95 * M) of the M continuations.",
    "ES95 is the mean of all continuations tied at or above q95.",
)
ELIGIBILITY_BANDS = (
    "0-17", "18-44", "45-64", "65-74", "75-84", "85+", "18-64", "65+"
)
ELDER_EXPOSURE_ERROR_DEFINITION = (
    "100 * sum_state abs(submitted_state_65plus_person_years - "
    "sealed_state_65plus_person_years) / sum_state sealed_state_65plus_person_years"
)
POOLED_EXCEEDANCE_DEFINITION = (
    "mean_region abs(sealed Pr(L > submitted_q95) - 0.05)"
)
EXPECTED_MORTALITY_SHOCK_RANGES = [
    {"kind": "mortality_spike", "range": [1.5, 3.0]},
]
EXPECTED_ADMISSION_SHOCK_RANGES = [
    {"kind": "mortality_spike", "range": [1.4, 2.6]},
]


class EvidenceError(ValueError):
    """Evidence is absent, ambiguous, duplicated, or non-finite."""


def gate_profile_selection(name: Any) -> dict[str, tuple[str, ...]]:
    """Return the gates and components one profile decides on.

    The selection is validated against the registered gate list on every call, so a
    profile can never name a gate or a component the freeze does not calibrate.
    """

    if not isinstance(name, str) or name not in GATE_PROFILES:
        raise EvidenceError(f"unknown gate profile {name!r}")
    selection: dict[str, tuple[str, ...]] = {}
    for gate, components in GATE_PROFILES[name].items():
        if gate not in GATE_COMPONENTS or not components \
                or not set(components) <= set(GATE_COMPONENTS[gate]):
            raise EvidenceError(
                f"gate profile {name!r} is not a selection over the registered gates"
            )
        selection[gate] = tuple(components)
    return selection


def _canonical_digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence binding must be finite JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, label: str) -> str:
    text = _identifier(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise EvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _report_binding(report: Mapping[str, Any]) -> dict[str, str]:
    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping) \
            or evidence.get("schema") != VERIFIER_EVIDENCE_SCHEMA:
        raise EvidenceError(
            f"verifier report must carry {VERIFIER_EVIDENCE_SCHEMA} evidence"
        )
    return {
        name: _sha256(evidence.get(name), name)
        for name in (
            "packet_digest_sha256",
            "contract_digest_sha256",
            "submission_digest_sha256",
            "verifier_digest_sha256",
        )
    }


def _packet_input_digests(report: Mapping[str, Any]) -> dict[str, str]:
    evidence = report.get("evidence")
    files = evidence.get("packet_file_sha256") \
        if isinstance(evidence, Mapping) else None
    if not isinstance(files, Mapping):
        raise EvidenceError("verifier evidence has no packet file digest map")
    return {
        name: _sha256(files.get(name), f"{name} packet input digest")
        for name in RED_TEAM_INPUT_FILES
    }


def _elder_reference_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    receipt = report.get("elder_reference_evidence")
    expected_keys = {
        "schema", "valid", "packet_digest_sha256", "submission_digest_sha256",
        "state_65_plus_person_years", "liability_mean_by_region",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys \
            or receipt.get("schema") != ELDER_REFERENCE_EVIDENCE_SCHEMA \
            or receipt.get("valid") is not True:
        raise EvidenceError("final reference report lacks elder reconstruction evidence")
    binding = _report_binding(report)
    if _sha256(receipt.get("packet_digest_sha256"), "elder packet digest") \
            != binding["packet_digest_sha256"] \
            or _sha256(
                receipt.get("submission_digest_sha256"), "elder submission digest"
            ) != binding["submission_digest_sha256"]:
        raise EvidenceError("elder reconstruction evidence has different report bytes")
    for field, identity, submitted, sealed in (
        ("state_65_plus_person_years", "state", "submitted_person_years",
         "sealed_person_years"),
        ("liability_mean_by_region", "region", "submitted", "sealed"),
    ):
        rows = receipt.get(field)
        if not isinstance(rows, list) or len(rows) != 6 \
                or {row.get(identity) for row in rows if isinstance(row, Mapping)} \
                != set(range(6)):
            raise EvidenceError(f"elder reconstruction {field} rows are incomplete")
        for row in rows:
            if not isinstance(row, Mapping) \
                    or set(row) != {identity, submitted, sealed}:
                raise EvidenceError(f"elder reconstruction {field} row fields differ")
            _audit_number(row.get(submitted), f"elder {field} submitted")
            _audit_number(row.get(sealed), f"elder {field} sealed")
    return json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))


def _shock_redraw_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    receipt = report.get("continuation_shock_redraw_evidence")
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema", "runtime_evidence_file_sha256", "liability_archive_sha256",
        "runtime_evidence",
    } or receipt.get("schema") != SHOCK_REDRAW_REPORT_SCHEMA:
        raise EvidenceError("verifier report lacks continuation shock redraw evidence")
    evidence = report.get("evidence")
    files = evidence.get("packet_file_sha256") \
        if isinstance(evidence, Mapping) else None
    if not isinstance(files, Mapping) \
            or _sha256(
                receipt.get("runtime_evidence_file_sha256"),
                "continuation shock runtime file digest",
            ) != _sha256(
                files.get("retained/continuation_shock_redraw.json"),
                "retained continuation shock runtime file digest",
            ) \
            or _sha256(
                receipt.get("liability_archive_sha256"),
                "continuation shock liability archive digest",
            ) != _sha256(
                files.get("retained/continuation_liabilities.npz"),
                "retained continuation liability digest",
            ):
        raise EvidenceError("continuation shock evidence is bound to different files")
    try:
        from meridia.packet import _validate_shock_redraw_evidence

        runtime = _validate_shock_redraw_evidence(receipt.get("runtime_evidence"))
    except (ImportError, TypeError, ValueError) as exc:
        raise EvidenceError("continuation shock runtime measurement is invalid") from exc
    runtime_file_digest = hashlib.sha256((
        json.dumps(runtime, indent=1, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")).hexdigest()
    if runtime_file_digest != receipt.get("runtime_evidence_file_sha256"):
        raise EvidenceError(
            "continuation shock runtime measurement differs from its retained file"
        )
    if runtime["redrawn_member_count"] != runtime["member_count"] \
            or runtime["distinct_future_schedule_count"] <= 1 \
            or not 0 < runtime["future_shock_year_count"] \
            < runtime["future_year_opportunity_count"] \
            or runtime["future_mortality_spike_year_count"] <= 0:
        raise EvidenceError("continuation shock runtime shows no independent redraws")
    return json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))


def evidence_binding(entry: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    """Return the exact replay binding whose digest is the evidence identifier."""

    if kind not in {"reference", "replicate", "control", "diagnostic"}:
        raise EvidenceError(f"unknown evidence kind {kind!r}")
    report = _report(entry)
    metadata = _metadata(entry)
    q95_feasibility = report.get("reserve_q95_feasibility")
    if not isinstance(q95_feasibility, Mapping):
        raise EvidenceError("verifier report has no reserve_q95_feasibility object")
    tail_evidence = _reserve_tail_evidence(report)
    reserve_rule_evidence = _reserve_rule_evidence(report)
    shock_redraw_evidence = _shock_redraw_evidence(report)
    packet_input_sha256 = _packet_input_digests(report)
    if reserve_rule_evidence["experience_sha256"] \
            != packet_input_sha256["participant/experience_history.csv"]:
        raise EvidenceError(
            "reserve rule experience digest differs from the packet input digest"
        )
    try:
        measurement_params = json.loads(json.dumps(
            _first(metadata, "measurement_params"),
            sort_keys=True,
            allow_nan=False,
        ))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("measurement_params must be finite JSON") from exc
    binding: dict[str, Any] = {
        "schema": EVIDENCE_BINDING_SCHEMA,
        "kind": kind,
        "world": _identifier(_first(metadata, "world", "world_id"), "world"),
        "method_digest_sha256": _sha256(
            _first(metadata, "method_digest_sha256"), "method_digest_sha256"
        ),
        "runner_digest_sha256": _sha256(
            _first(metadata, "runner_digest_sha256"), "runner_digest_sha256"
        ),
        "measurement_contract_digest_sha256": _sha256(
            _first(metadata, "measurement_contract_digest_sha256"),
            "measurement_contract_digest_sha256",
        ),
        "measurement_params": measurement_params,
        "run_receipt_digest_sha256": _sha256(
            _first(metadata, "run_receipt_digest_sha256"),
            "run_receipt_digest_sha256",
        ),
        **_report_binding(report),
        "verifier_report_digest_sha256": _canonical_digest(report),
        "reserve_q95_feasibility_digest_sha256": _canonical_digest(
            q95_feasibility
        ),
        "reserve_tail_evidence": tail_evidence,
        "reserve_tail_evidence_digest_sha256": _canonical_digest(tail_evidence),
        "reserve_rule_evidence": reserve_rule_evidence,
        "reserve_rule_evidence_digest_sha256": _canonical_digest(
            reserve_rule_evidence
        ),
        "continuation_shock_redraw_evidence_digest_sha256": _canonical_digest(
            shock_redraw_evidence
        ),
        "continuation_shock_redraw_file_sha256": shock_redraw_evidence[
            "runtime_evidence_file_sha256"
        ],
        "continuation_source_law_sha256": shock_redraw_evidence[
            "runtime_evidence"
        ]["continuation_source_law_sha256"],
        "packet_input_sha256": packet_input_sha256,
    }
    if binding["measurement_params"] != REGISTERED_MEASUREMENT_PARAMS:
        raise EvidenceError(
            "freeze evidence measurement parameters differ from the registered "
            "100/400/2048/12 design"
        )
    if kind == "control":
        binding["control"] = _identifier(
            _first(metadata, "control", "control_name"), "control"
        )
    elif kind == "diagnostic":
        binding["diagnostic"] = _identifier(
            _first(metadata, "diagnostic", "diagnostic_name"), "diagnostic"
        )
    else:
        binding["reference_line"] = _identifier(
            _first(metadata, "reference_line", "witness", "line"),
            "reference_line",
        )
        if kind == "reference":
            elder_evidence = _elder_reference_evidence(report)
            binding["elder_reference_evidence"] = elder_evidence
            binding["elder_reference_evidence_digest_sha256"] = _canonical_digest(
                elder_evidence
            )
    if kind == "replicate":
        binding["replicate_id"] = _identifier(
            _first(metadata, "replicate_id", "replicate"), "replicate_id"
        )
        binding["resample_digest_sha256"] = _sha256(
            _first(metadata, "resample_digest_sha256"),
            "resample_digest_sha256",
        )
        design = _first(metadata, "resampling_design")
        if not isinstance(design, Mapping) or not design:
            raise EvidenceError("resampling_design must be a nonempty object")
        # A canonical round trip strips custom mapping behavior and rejects NaN.
        try:
            binding["resampling_design"] = json.loads(json.dumps(
                design, sort_keys=True, allow_nan=False
            ))
        except (TypeError, ValueError) as exc:
            raise EvidenceError("resampling_design must be finite JSON") from exc
    return binding


def evidence_id_for(entry: Mapping[str, Any], *, kind: str) -> str:
    """Compute the registered evidence identifier for a report wrapper."""

    return _canonical_digest(evidence_binding(entry, kind=kind))


def empirical_order_statistic(values: Sequence[float], quantile: float = QUANTILE) -> float:
    """Return order statistic ``ceil(quantile * N)`` with no interpolation."""

    if isinstance(values, (str, bytes)) or not values:
        raise EvidenceError("an empirical quantile needs at least one value")
    if not math.isfinite(float(quantile)) or not 0.0 < float(quantile) <= 1.0:
        raise EvidenceError("quantile must be finite and in (0, 1]")
    clean: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise EvidenceError("boolean values are not metric observations")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"metric observation is not numeric: {value!r}") from exc
        if not math.isfinite(number):
            raise EvidenceError("metric observations must all be finite")
        clean.append(number)
    rank = math.ceil(float(quantile) * len(clean))
    return sorted(clean)[rank - 1]


def empirical_p99(values: Sequence[float]) -> float:
    """Return the registered freeze statistic."""

    return empirical_order_statistic(values, QUANTILE)


def _metadata(entry: Mapping[str, Any]) -> dict[str, Any]:
    report = entry.get("report")
    nested = report.get("evidence") if isinstance(report, Mapping) else None
    outer = entry.get("evidence")
    merged: dict[str, Any] = {}
    if isinstance(nested, Mapping):
        merged.update(nested)
    if isinstance(outer, Mapping):
        merged.update(outer)
    merged.update({key: value for key, value in entry.items()
                   if key not in ("report", "evidence")})
    return merged


def _report(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    report = entry.get("report", entry)
    if not isinstance(report, Mapping):
        raise EvidenceError("each evidence entry must contain a verifier report object")
    return report


def _first(metadata: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in metadata:
            return metadata[name]
    return None


def _identifier(value: Any, label: str) -> str:
    if isinstance(value, bool) or value is None:
        raise EvidenceError(f"{label} is missing")
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, int):
        text = str(value)
    else:
        raise EvidenceError(f"{label} must be a string or integer identifier")
    if not text:
        raise EvidenceError(f"{label} is missing")
    return text


def _hard_pass(report: Mapping[str, Any]) -> bool:
    """Read an explicit deterministic hard-check result; absence is failure."""

    for key in ("hard_pass", "hard_checks_passed", "hard_structure_pass"):
        if key in report:
            return report[key] is True
    checks = report.get("hard_checks")
    if isinstance(checks, bool):
        return checks
    if isinstance(checks, Mapping) and checks:
        return all(value is True for value in checks.values())
    return False


def _number(value: Any, gate: str, component: str) -> float:
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, bool):
        raise EvidenceError(f"{gate}/{component} is boolean, not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{gate}/{component} is missing or non-numeric") from exc
    if not math.isfinite(number):
        raise EvidenceError(f"{gate}/{component} is non-finite")
    low, high = COMPONENT_RANGES[(gate, component)]
    if number < low or (high is not None and number > high):
        raise EvidenceError(
            f"{gate}/{component}={number} is outside its range [{low}, {high}]"
        )
    return number


def extract_composite_metrics(report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Validate and extract the exact five-gate metric surface from one report."""

    metrics = report.get("composite_metrics")
    if not isinstance(metrics, Mapping):
        raise EvidenceError("verifier report has no composite_metrics object")
    if set(metrics) != set(GATE_COMPONENTS):
        missing = sorted(set(GATE_COMPONENTS) - set(metrics))
        unexpected = sorted(set(metrics) - set(GATE_COMPONENTS))
        raise EvidenceError(
            f"composite gate names differ from the freeze schema; missing {missing}, "
            f"unexpected {unexpected}"
        )
    extracted: dict[str, dict[str, float]] = {}
    for gate, components in GATE_COMPONENTS.items():
        block = metrics.get(gate)
        if not isinstance(block, Mapping):
            raise EvidenceError(f"{gate} metrics must be an object")
        extracted[gate] = {
            component: _number(block.get(component), gate, component)
            for component in components
        }
    return extracted


def _evidence_identity(entry: Mapping[str, Any], *, replicate: bool,
                       control: bool = False, diagnostic: bool = False) -> dict[str, Any]:
    metadata = _metadata(entry)
    if control and diagnostic:
        raise EvidenceError("an evidence report cannot be both a control and a diagnostic")
    kind = "control" if control else (
        "diagnostic" if diagnostic else ("replicate" if replicate else "reference")
    )
    binding = evidence_binding(entry, kind=kind)
    evidence_id = _identifier(
        _first(metadata, "evidence_id", "evidence_digest", "report_id"),
        "evidence_id",
    )
    expected_id = _canonical_digest(binding)
    if evidence_id != expected_id:
        raise EvidenceError(
            f"{evidence_id}: evidence_id does not match its replay binding; "
            f"expected {expected_id}"
        )
    identity = {
        "world": binding["world"],
        "evidence_id": evidence_id,
        "binding": binding,
    }
    if metadata.get("deterministic") is not True:
        raise EvidenceError(
            f"{identity['evidence_id']}: deterministic must be explicitly true"
        )
    if control:
        identity["control"] = binding["control"]
    elif diagnostic:
        identity["diagnostic"] = binding["diagnostic"]
    else:
        identity["reference_line"] = binding["reference_line"]
    if replicate:
        identity["replicate_id"] = binding["replicate_id"]
    return identity


def _reserve_q95_feasibility(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit participant-facing feasibility receipt in a report."""

    receipt = report.get("reserve_q95_feasibility")
    if not isinstance(receipt, Mapping):
        raise EvidenceError("verifier report has no reserve_q95_feasibility object")
    fields = {
        name: _audit_number(receipt.get(name), f"reserve q95 feasibility {name}")
        for name in ("q95_sum", "allocation_sum", "reserve_total")
    }
    fields["total_minus_q95_sum"] = _audit_number(
        receipt.get("total_minus_q95_sum"),
        "reserve q95 feasibility total_minus_q95_sum",
        low=float("-inf"),
    )
    expected_keys = set(fields) | {
        "all_regions_at_or_above_q95",
        "allocation_sums_to_total",
        "feasible",
    }
    if set(receipt) != expected_keys:
        raise EvidenceError("reserve_q95_feasibility fields differ from the contract")
    flags = {
        name: receipt.get(name)
        for name in (
            "all_regions_at_or_above_q95", "allocation_sums_to_total", "feasible"
        )
    }
    if any(not isinstance(value, bool) for value in flags.values()):
        raise EvidenceError("reserve q95 diagnostic flags must be boolean")
    tolerance = 1e-10 * max(1.0, fields["reserve_total"])
    sums_to_total = abs(
        fields["allocation_sum"] - fields["reserve_total"]
    ) <= tolerance
    if flags["allocation_sums_to_total"] is not sums_to_total:
        raise EvidenceError("reserve q95 allocation-sum diagnostic is inconsistent")
    if abs(
        fields["total_minus_q95_sum"]
        - (fields["reserve_total"] - fields["q95_sum"])
    ) > tolerance:
        raise EvidenceError("reserve q95 diagnostic margin is inconsistent")
    if flags["feasible"] is not (
        flags["all_regions_at_or_above_q95"] and sums_to_total
    ):
        raise EvidenceError("reserve q95 floor diagnostic is internally inconsistent")
    return json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))


def _reserve_rule_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public reserve-total recomputation recorded by the verifier."""

    receipt = report.get("reserve_rule_evidence")
    expected_keys = {
        "valid", "selected_year", "exposure_person_years",
        "rate_per_person_year", "rounding_unit", "reserve_total",
        "experience_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys \
            or receipt.get("valid") is not True:
        raise EvidenceError("verifier report has no valid reserve_rule_evidence object")
    selected_year = receipt.get("selected_year")
    if isinstance(selected_year, bool) or not isinstance(selected_year, int):
        raise EvidenceError("reserve rule selected_year must be an integer")
    exposure = _audit_number(
        receipt.get("exposure_person_years"),
        "reserve rule exposure_person_years",
        low=1e-300,
    )
    rate = _audit_number(
        receipt.get("rate_per_person_year"),
        "reserve rule rate_per_person_year",
        low=1e-300,
    )
    unit = _audit_number(
        receipt.get("rounding_unit"), "reserve rule rounding_unit", low=1e-300
    )
    total = _audit_number(receipt.get("reserve_total"), "reserve rule reserve_total")
    if not math.isclose(
        total, _public_reserve_total(exposure, rate, unit),
        rel_tol=1e-12, abs_tol=1e-9,
    ):
        raise EvidenceError("reserve rule total does not recompute")
    _sha256(receipt.get("experience_sha256"), "reserve rule experience_sha256")
    return json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))


def _reserve_tail_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    receipt = report.get("reserve_tail_evidence")
    expected_keys = {
        "schema", "valid", "q95_sum", "es95_sum",
        "reserve_submission_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys \
            or receipt.get("schema") != RESERVE_TAIL_EVIDENCE_SCHEMA \
            or receipt.get("valid") is not True:
        raise EvidenceError("verifier report has no valid reserve_tail_evidence object")
    q95_sum = _audit_number(
        receipt.get("q95_sum"), "reserve tail q95_sum"
    )
    es95_sum = _audit_number(
        receipt.get("es95_sum"), "reserve tail es95_sum"
    )
    if es95_sum < q95_sum:
        raise EvidenceError("reserve tail ES95 sum is below its q95 sum")
    reserve_digest = _sha256(
        receipt.get("reserve_submission_sha256"),
        "reserve tail reserve_submission_sha256",
    )
    evidence = report.get("evidence")
    files = evidence.get("submission_file_sha256") \
        if isinstance(evidence, Mapping) else None
    if not isinstance(files, Mapping):
        raise EvidenceError("verifier evidence has no submission file digest map")
    if reserve_digest != _sha256(
        files.get("reserve.csv"), "reserve.csv submission digest"
    ):
        raise EvidenceError("reserve tail evidence is bound to different reserve.csv bytes")
    feasibility = _reserve_q95_feasibility(report)
    if not math.isclose(
        q95_sum, feasibility["q95_sum"], rel_tol=1e-12, abs_tol=1e-9
    ):
        raise EvidenceError("reserve tail q95 sum differs from the q95 diagnostic")
    return json.loads(json.dumps(receipt, sort_keys=True, allow_nan=False))


def _normalized_reference(entry: Mapping[str, Any], *, replicate: bool) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise EvidenceError("reference evidence entries must be objects")
    report = _report(entry)
    identity = _evidence_identity(entry, replicate=replicate)
    return identity | {
        "hard_pass": _hard_pass(report),
        "metrics": extract_composite_metrics(report),
        "reserve_q95_feasibility": _reserve_q95_feasibility(report),
        "report": report,
    }


def _normalized_control(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise EvidenceError("control evidence entries must be objects")
    report = _report(entry)
    identity = _evidence_identity(entry, replicate=False, control=True)
    return identity | {
        "hard_pass": _hard_pass(report),
        "metrics": extract_composite_metrics(report),
        "reserve_q95_feasibility": _reserve_q95_feasibility(report),
        "report": report,
    }


def _normalized_diagnostic(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise EvidenceError("development diagnostic evidence entries must be objects")
    report = _report(entry)
    identity = _evidence_identity(
        entry, replicate=False, diagnostic=True
    )
    return identity | {
        "hard_pass": _hard_pass(report),
        "metrics": extract_composite_metrics(report),
        "reserve_q95_feasibility": _reserve_q95_feasibility(report),
        "report": report,
    }


def _normalize_entries(entries: Iterable[Mapping[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    normalizer = {
        "reference": lambda entry: _normalized_reference(entry, replicate=False),
        "replicate": lambda entry: _normalized_reference(entry, replicate=True),
        "control": _normalized_control,
        "diagnostic": _normalized_diagnostic,
    }[kind]
    normalized = [normalizer(entry) for entry in entries]
    evidence_ids = [entry["evidence_id"] for entry in normalized]
    counts: defaultdict[str, int] = defaultdict(int)
    for evidence_id in evidence_ids:
        counts[evidence_id] += 1
    duplicates = sorted(item for item, count in counts.items() if count > 1)
    if duplicates:
        raise EvidenceError(f"duplicate {kind} evidence_id values: {duplicates}")
    return normalized


def _check_binding_consistency(references: Sequence[Mapping[str, Any]],
                               replicates: Sequence[Mapping[str, Any]],
                               controls: Sequence[Mapping[str, Any]],
                               diagnostics: Sequence[Mapping[str, Any]],
                               lines: Sequence[str], worlds: Sequence[str]) -> None:
    all_entries = [*references, *replicates, *controls, *diagnostics]
    if any(entry["world"] not in worlds for entry in controls):
        raise EvidenceError("control evidence names an unregistered qualification world")
    observed_controls = {entry["control"] for entry in controls}
    if observed_controls != set(REQUIRED_SCIENTIFIC_CONTROLS):
        raise EvidenceError(
            "qualification control labels differ from the registered battery; "
            f"missing {sorted(set(REQUIRED_SCIENTIFIC_CONTROLS) - observed_controls)}, "
            f"unexpected {sorted(observed_controls - set(REQUIRED_SCIENTIFIC_CONTROLS))}"
        )
    if any(entry["world"] not in DEVELOPMENT_WORLDS for entry in diagnostics):
        raise EvidenceError("diagnostic evidence names an unregistered development world")
    for field in (
        "verifier_digest_sha256",
        "runner_digest_sha256",
        "measurement_contract_digest_sha256",
    ):
        values = {entry["binding"][field] for entry in all_entries}
        if len(values) != 1:
            raise EvidenceError(f"evidence was produced by more than one {field[:-7]}")
    run_receipts = [
        entry["binding"]["run_receipt_digest_sha256"] for entry in all_entries
    ]
    if len(run_receipts) != len(set(run_receipts)):
        raise EvidenceError("a run receipt was reused or relabeled as another report")
    evidence_ids = [entry["evidence_id"] for entry in all_entries]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceError("an evidence identifier was reused across report classes")
    def require_same_binding(
        rows: Sequence[Mapping[str, Any]], label: str
    ) -> None:
        for field in ("packet_digest_sha256", "contract_digest_sha256"):
            values = {entry["binding"][field] for entry in rows}
            if len(values) != 1:
                raise EvidenceError(f"{label}: evidence disagrees on {field}")
        for field in (
            "continuation_shock_redraw_evidence_digest_sha256",
            "continuation_shock_redraw_file_sha256",
            "continuation_source_law_sha256",
        ):
            values = {entry["binding"][field] for entry in rows}
            if len(values) != 1:
                raise EvidenceError(f"{label}: evidence disagrees on {field}")
        for field in ("packet_input_sha256", "reserve_rule_evidence"):
            values = {
                _canonical_digest(entry["binding"][field]) for entry in rows
            }
            if len(values) != 1:
                raise EvidenceError(f"{label}: evidence disagrees on {field}")

    base_by_world: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in (*references, *controls):
        base_by_world[entry["world"]].append(entry)
    for world in worlds:
        require_same_binding(base_by_world[world], world)
    for world in DEVELOPMENT_WORLDS:
        require_same_binding(
            [entry for entry in diagnostics if entry["world"] == world], world
        )
    method_by_identity: dict[str, str] = {}
    for line in lines:
        values = {
            entry["binding"]["method_digest_sha256"]
            for entry in (*references, *replicates)
            if entry["reference_line"] == line
        }
        if len(values) != 1:
            raise EvidenceError(f"{line}: final and replicate method digests differ")
        method_by_identity[f"reference:{line}"] = next(iter(values))
    for name in REQUIRED_SCIENTIFIC_CONTROLS:
        values = {
            entry["binding"]["method_digest_sha256"]
            for entry in controls
            if entry["control"] == name
        }
        if len(values) != 1:
            raise EvidenceError(f"{name}: control method digest is not stable")
        method_by_identity[f"control:{name}"] = next(iter(values))
    for name in DEVELOPMENT_DIAGNOSTICS:
        values = {
            entry["binding"]["method_digest_sha256"]
            for entry in diagnostics
            if entry["diagnostic"] == name
        }
        if len(values) != 1:
            raise EvidenceError(f"{name}: diagnostic method digest is not stable")
        method_by_identity[f"diagnostic:{name}"] = next(iter(values))
    digest_owners: defaultdict[str, list[str]] = defaultdict(list)
    for identity, digest in method_by_identity.items():
        digest_owners[digest].append(identity)
    relabeled = {
        digest: sorted(owners)
        for digest, owners in digest_owners.items()
        if len(owners) > 1
    }
    if relabeled:
        raise EvidenceError(
            "method digest reused under different line, control, or diagnostic labels: "
            f"{relabeled}"
        )
    by_pair: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    designs: set[str] = set()
    for entry in replicates:
        by_pair[(entry["reference_line"], entry["world"])].append(entry)
        designs.add(_canonical_digest(entry["binding"]["resampling_design"]))
    if len(designs) != 1:
        raise EvidenceError("replicate reports use more than one resampling design")
    for pair, rows in by_pair.items():
        digests = [row["binding"]["resample_digest_sha256"] for row in rows]
        if len(digests) != len(set(digests)):
            raise EvidenceError(f"duplicate resample digest within {pair}")
    resample_owners: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for world in worlds:
        by_replicate: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for entry in replicates:
            if entry["world"] == world:
                by_replicate[entry["replicate_id"]].append(entry)
        if len(by_replicate) != REPLICATES_PER_LINE_WORLD:
            raise EvidenceError(
                f"{world}: expected {REPLICATES_PER_LINE_WORLD} paired resamples, "
                f"found {len(by_replicate)}"
            )
        for replicate_id, rows in by_replicate.items():
            observed_lines = sorted(row["reference_line"] for row in rows)
            if observed_lines != list(REFERENCE_LINES) or len(rows) != len(REFERENCE_LINES):
                raise EvidenceError(
                    f"{world}/{replicate_id}: resample is not paired across A, B, and C"
                )
            digests = {row["binding"]["resample_digest_sha256"] for row in rows}
            if len(digests) != 1:
                raise EvidenceError(
                    f"{world}/{replicate_id}: A, B, and C use different resample digests"
                )
            require_same_binding(rows, f"{world}/{replicate_id}")
            base = base_by_world[world][0]["binding"]
            for field in ("contract_digest_sha256",):
                if rows[0]["binding"][field] != base[field]:
                    raise EvidenceError(
                        f"{world}/{replicate_id}: resample changes {field}"
                    )
            for field in (
                "continuation_shock_redraw_evidence_digest_sha256",
                "continuation_shock_redraw_file_sha256",
                "continuation_source_law_sha256",
            ):
                if rows[0]["binding"][field] != base[field]:
                    raise EvidenceError(
                        f"{world}/{replicate_id}: resample changes {field}"
                    )
            packet_inputs = rows[0]["binding"]["packet_input_sha256"]
            base_inputs = base["packet_input_sha256"]
            for name in (
                "participant/contract.json",
                "retained/continuation_liabilities.npz",
            ):
                if packet_inputs[name] != base_inputs[name]:
                    raise EvidenceError(
                        f"{world}/{replicate_id}: resample changes fixed input {name}"
                    )
            reserve_rule = rows[0]["binding"]["reserve_rule_evidence"]
            base_rule = base["reserve_rule_evidence"]
            for field in (
                "valid",
                "selected_year",
                "exposure_person_years",
                "rate_per_person_year",
                "rounding_unit",
                "reserve_total",
            ):
                if reserve_rule[field] != base_rule[field]:
                    raise EvidenceError(
                        f"{world}/{replicate_id}: resample changes reserve rule {field}"
                    )
            resample_owners[next(iter(digests))].append((world, replicate_id))
    duplicated_resamples = {
        digest: owners for digest, owners in resample_owners.items() if len(owners) > 1
    }
    if duplicated_resamples:
        raise EvidenceError(
            "a paired resample digest was reused across world or replicate identifiers: "
            f"{duplicated_resamples}"
        )


def _check_development_diagnostic_design(
    diagnostics: Sequence[Mapping[str, Any]],
) -> None:
    if len(diagnostics) != DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT:
        raise EvidenceError(
            "development diagnostics must contain exactly "
            f"{DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT} reports"
        )
    expected = {
        (name, world)
        for name in DEVELOPMENT_DIAGNOSTICS
        for world in DEVELOPMENT_WORLDS
    }
    groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in diagnostics:
        groups[(entry["diagnostic"], entry["world"])].append(entry)
    if set(groups) != expected or any(len(rows) != 1 for rows in groups.values()):
        raise EvidenceError(
            "development diagnostics need each registered diagnostic on dev-00 through "
            "dev-11 exactly once"
        )
    if any(not entry["hard_pass"] for entry in diagnostics):
        failed = sorted(
            f"{entry['diagnostic']}/{entry['world']}"
            for entry in diagnostics
            if not entry["hard_pass"]
        )
        raise EvidenceError(f"development diagnostics failed hard checks: {failed}")


def _freeze_provenance(references: Sequence[Mapping[str, Any]],
                       replicates: Sequence[Mapping[str, Any]],
                       controls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [dict(row["binding"], evidence_id=row["evidence_id"]) for row in rows],
            key=lambda row: (
                row["kind"], row.get("reference_line", row.get("control", "")),
                row["world"], row.get("replicate_id", ""), row["evidence_id"],
            ),
        )

    record = {
        "schema": PROVENANCE_SCHEMA,
        "reference_reports": records(references),
        "replicate_reports": records(replicates),
        "control_reports": records(controls),
    }
    record["digest_sha256"] = _canonical_digest(record)
    return record


def _development_diagnostic_block(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports = sorted(
        [dict(row["binding"], evidence_id=row["evidence_id"]) for row in diagnostics],
        key=lambda row: (row["diagnostic"], row["world"], row["evidence_id"]),
    )
    block = {
        "schema": DEVELOPMENT_DIAGNOSTIC_SCHEMA,
        "registered_diagnostics": list(DEVELOPMENT_DIAGNOSTICS),
        "development_worlds": list(DEVELOPMENT_WORLDS),
        "report_count": len(reports),
        "counts_as_qualification_control": False,
        "reports": reports,
    }
    block["digest_sha256"] = _canonical_digest(block)
    return block


def _audit_number(value: Any, label: str, *, low: float = 0.0,
                  high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < low or (high is not None and number > high):
        raise EvidenceError(f"{label} is outside its registered range")
    return number


def _digest_bound_audit(
    audit: Mapping[str, Any] | None,
    *,
    schema: str,
    label: str,
    measurement_contract_digest: str,
) -> dict[str, Any]:
    if not isinstance(audit, Mapping) or audit.get("schema") != schema:
        raise EvidenceError(f"a complete {schema} report is required")
    try:
        normalized = json.loads(json.dumps(audit, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{label} must be finite JSON") from exc
    raw_recorded_digest = normalized.pop("digest_sha256", None)
    recorded_digest = _sha256(raw_recorded_digest, f"{label} digest_sha256")
    if raw_recorded_digest != recorded_digest:
        raise EvidenceError(f"{label} digest_sha256 is not canonical lowercase")
    if recorded_digest != _canonical_digest(normalized):
        raise EvidenceError(f"{label} digest differs from its content")
    raw_contract = normalized.get("measurement_contract_digest_sha256")
    observed_contract = _sha256(
        raw_contract,
        f"{label} measurement_contract_digest_sha256",
    )
    if raw_contract != observed_contract:
        raise EvidenceError(
            f"{label} measurement_contract_digest_sha256 is not canonical lowercase"
        )
    if observed_contract != measurement_contract_digest:
        raise EvidenceError(f"{label} is bound to a different measurement contract")
    normalized["digest_sha256"] = recorded_digest
    return normalized


def _public_reserve_total(exposure: float, rate: float, unit: float) -> float:
    raw_units = Decimal(str(exposure)) * Decimal(str(rate)) / Decimal(str(unit))
    return float(raw_units.to_integral_value(rounding=ROUND_CEILING) * Decimal(str(unit)))


def _validate_reserve_calibration_candidate(
    audit: Mapping[str, Any] | None,
    references: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(audit, Mapping) \
            or audit.get("schema") != RESERVE_CALIBRATION_SCHEMA:
        raise EvidenceError(
            f"an unaccepted {RESERVE_CALIBRATION_SCHEMA} candidate is required"
        )
    try:
        normalized = json.loads(json.dumps(audit, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("reserve calibration candidate must be finite JSON") from exc
    if set(normalized) != RESERVE_CALIBRATION_CANDIDATE_KEYS \
            or normalized.get("candidate") is not True \
            or normalized.get("accepted") is not False \
            or normalized.get("blockers") \
            != list(RESERVE_CALIBRATION_PENDING_BLOCKERS) \
            or normalized.get("reference_lines") != list(REFERENCE_LINES) \
            or normalized.get("qualification_worlds") != list(QUALIFICATION_WORLDS) \
            or normalized.get("target_rule") \
            != "sum(q95) + tail_slack_share * sum(ES95 - q95)":
        raise EvidenceError(
            "reserve calibration input must be the canonical unaccepted candidate"
        )
    rate = _audit_number(
        normalized.get("rate_per_person_year"),
        "reserve calibration rate_per_person_year",
        low=1e-300,
    )
    grid = _audit_number(
        normalized.get("rate_grid"), "reserve calibration rate_grid", low=1e-300
    )
    slack = _audit_number(
        normalized.get("tail_slack_share"),
        "reserve calibration tail_slack_share",
        low=0.0,
        high=1.0,
    )
    if grid != 1.0 or slack != 0.25:
        raise EvidenceError(
            "reserve calibration must use RATE_GRID=1.0 and TAIL_SLACK_SHARE=0.25"
        )
    rows = normalized.get("evidence")
    if not isinstance(rows, list) or len(rows) != REFERENCE_REPORT_COUNT:
        raise EvidenceError("reserve calibration audit needs all 18 reference reports")
    expected = {
        (row["reference_line"], row["world"]): row for row in references
    }
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    required_rates: list[float] = []
    rounding_units: set[float] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != RESERVE_CALIBRATION_EVIDENCE_KEYS:
            raise EvidenceError("reserve calibration candidate evidence fields differ")
        line = _identifier(
            row.get("reference_line"), "reserve calibration reference_line"
        )
        world = _identifier(row.get("world"), "reserve calibration world")
        if row.get("reference_line") != line or row.get("world") != world:
            raise EvidenceError("reserve calibration identities are not canonical")
        key = (line, world)
        if key not in expected or key in observed:
            raise EvidenceError("reserve calibration evidence pairs differ")
        reference = expected[key]
        if row.get("evidence_id") != reference["evidence_id"]:
            raise EvidenceError("reserve calibration evidence id differs")
        experience_digest = _sha256(
            row.get("experience_sha256"),
            "reserve calibration experience_sha256",
        )
        reserve_digest = _sha256(
            row.get("reserve_submission_sha256"),
            "reserve calibration reserve_submission_sha256",
        )
        exposure = _audit_number(
            row.get("exposure_person_years"),
            "reserve calibration exposure_person_years",
            low=1e-300,
        )
        rounding_unit = _audit_number(
            row.get("rounding_unit"),
            "reserve calibration rounding_unit",
            low=1e-300,
        )
        q95_sum = _audit_number(
            row.get("submitted_q95_sum"), "reserve calibration submitted_q95_sum"
        )
        if not math.isclose(
            q95_sum,
            reference["reserve_q95_feasibility"]["q95_sum"],
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise EvidenceError("reserve calibration q95 sum differs from the verifier")
        margin = _audit_number(
            row.get("candidate_margin"), "reserve calibration candidate_margin"
        )
        es95_sum = _audit_number(
            row.get("submitted_es95_sum"), "reserve calibration submitted_es95_sum"
        )
        candidate_total = _audit_number(
            row.get("candidate_reserve_total"),
            "reserve calibration candidate_reserve_total",
        )
        target = q95_sum + slack * (es95_sum - q95_sum)
        recorded_target = _audit_number(
            row.get("target_reserve_before_rounding"),
            "reserve calibration target_reserve_before_rounding",
        )
        required_rate = _audit_number(
            row.get("required_rate"), "reserve calibration required_rate"
        )
        reference_total = reference["reserve_q95_feasibility"]["reserve_total"]
        reserve_rule = _reserve_rule_evidence(reference["report"])
        tail_evidence = _reserve_tail_evidence(reference["report"])
        if es95_sum < q95_sum \
                or candidate_total + 1e-9 < q95_sum \
                or margin < 0.0 \
                or experience_digest != reserve_rule["experience_sha256"] \
                or reserve_digest != tail_evidence["reserve_submission_sha256"] \
                or not math.isclose(
                    q95_sum, tail_evidence["q95_sum"],
                    rel_tol=1e-12, abs_tol=1e-9,
                ) \
                or not math.isclose(
                    es95_sum, tail_evidence["es95_sum"],
                    rel_tol=1e-12, abs_tol=1e-9,
                ) \
                or not math.isclose(
                    exposure, reserve_rule["exposure_person_years"],
                    rel_tol=1e-12, abs_tol=1e-9,
                ) \
                or not math.isclose(
                    rate, reserve_rule["rate_per_person_year"],
                    rel_tol=1e-12, abs_tol=1e-12,
                ) \
                or not math.isclose(
                    rounding_unit, reserve_rule["rounding_unit"],
                    rel_tol=1e-12, abs_tol=1e-12,
                ) \
                or not math.isclose(
                    recorded_target, target, rel_tol=1e-12, abs_tol=1e-9
                ) \
                or not math.isclose(
                    required_rate, target / exposure, rel_tol=1e-12, abs_tol=1e-12
                ) \
                or not math.isclose(
                    candidate_total,
                    _public_reserve_total(exposure, rate, rounding_unit),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                ) \
                or not math.isclose(
                    candidate_total, reference_total, rel_tol=1e-12, abs_tol=1e-9
                ) \
                or not math.isclose(
                    candidate_total, reserve_rule["reserve_total"],
                    rel_tol=1e-12, abs_tol=1e-9,
                ) \
                or not math.isclose(
                    margin, candidate_total - target, rel_tol=1e-12, abs_tol=1e-9
                ):
            raise EvidenceError("reserve calibration candidate is infeasible")
        observed[key] = row
        required_rates.append(required_rate)
        rounding_units.add(rounding_unit)
    if set(observed) != set(expected):
        raise EvidenceError("reserve calibration evidence pairs are incomplete")
    if len(rounding_units) != 1:
        raise EvidenceError(
            "qualification reference reports do not share one reserve rounding unit"
        )
    expected_rate = math.ceil(max(required_rates) / grid) * grid
    if not math.isclose(rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError(
            "reserve calibration rate is not the smallest registered feasible rate"
        )

    reserve_ceilings = {
        component: gates["reserve_skill"]["components"][component]["value"]
        for component in GATE_COMPONENTS["reserve_skill"]
    }
    failed_references = [
        f"{row['reference_line']}/{row['world']}"
        for row in references
        if any(
            row["metrics"]["reserve_skill"][component] > reserve_ceilings[component]
            for component in GATE_COMPONENTS["reserve_skill"]
        )
    ]
    if failed_references:
        raise EvidenceError(
            "reserve calibration remains blocked because final references fail "
            "reserve_skill: " + ", ".join(failed_references)
        )
    proportional = sorted(
        (row for row in controls if row["control"] == "proportional_reserve"),
        key=lambda row: row["world"],
    )
    if len(proportional) != len(QUALIFICATION_WORLDS) \
            or [row["world"] for row in proportional] != list(QUALIFICATION_WORLDS):
        raise EvidenceError(
            "reserve calibration needs one proportional_reserve report per world"
        )
    hard_invalid_proportional = [
        row["world"] for row in proportional if row["hard_pass"] is not True
    ]
    if hard_invalid_proportional:
        raise EvidenceError(
            "reserve calibration remains blocked because proportional_reserve is "
            "hard-invalid: " + ", ".join(hard_invalid_proportional)
        )
    passed_proportional = [
        row["world"] for row in proportional
        if all(
            row["metrics"]["reserve_skill"][component] <= reserve_ceilings[component]
            for component in GATE_COMPONENTS["reserve_skill"]
        )
    ]
    if passed_proportional:
        raise EvidenceError(
            "reserve calibration remains blocked because proportional_reserve passes "
            "reserve_skill: " + ", ".join(passed_proportional)
        )

    return normalized


def _promote_reserve_calibration_candidate(
    candidate: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    measurement_contract_digest: str,
    red_team: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_digest = _canonical_digest(candidate)
    ceilings = {
        component: gates["reserve_skill"]["components"][component]["value"]
        for component in GATE_COMPONENTS["reserve_skill"]
    }
    proportional = sorted(
        (row for row in controls if row["control"] == "proportional_reserve"),
        key=lambda row: row["world"],
    )
    promoted = dict(candidate)
    promoted.update({
        "accepted": True,
        "blockers": [],
        "candidate_source_digest_sha256": candidate_digest,
        "measurement_contract_digest_sha256": measurement_contract_digest,
        "acceptance_evidence": {
            "reserve_skill_component_ceilings": ceilings,
            "reference_evidence_ids": sorted(
                row["evidence_id"] for row in references
            ),
            "proportional_reserve_evidence_ids": sorted(
                row["evidence_id"] for row in proportional
            ),
            "red_team_audit_digest_sha256": red_team["digest_sha256"],
        },
    })
    promoted["digest_sha256"] = _canonical_digest(promoted)
    return promoted


def _red_team_r2(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) > 1.0:
        raise EvidenceError(f"{label} must be finite and no greater than one")
    return float(value)


def _validate_red_team_headline(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"q95", "es95", "headline_max"}:
        raise EvidenceError(f"{label} fields differ")
    q95 = _red_team_r2(value.get("q95"), f"{label} q95")
    es95 = _red_team_r2(value.get("es95"), f"{label} ES95")
    headline = _red_team_r2(value.get("headline_max"), f"{label} headline")
    if not math.isclose(headline, max(q95, es95), rel_tol=1e-12, abs_tol=1e-15):
        raise EvidenceError(f"{label} headline does not recompute")


def _validate_red_team_models(value: Any, regions: int, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"q95", "es95"}:
        raise EvidenceError(f"{label} model groups differ")
    for outcome in ("q95", "es95"):
        rows = value.get(outcome)
        if not isinstance(rows, list) or len(rows) != regions:
            raise EvidenceError(f"{label} {outcome} model count differs")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {
                "region", "intercept", "reserve_total_coefficient"
            } or row.get("region") != index:
                raise EvidenceError(f"{label} {outcome} model fields differ")
            for field in ("intercept", "reserve_total_coefficient"):
                number = row.get(field)
                if isinstance(number, bool) or not isinstance(number, (int, float)) \
                        or not math.isfinite(float(number)):
                    raise EvidenceError(f"{label} {outcome} model is non-finite")


def _expected_red_team_inputs(
    entries: Sequence[Mapping[str, Any]], worlds: Sequence[str], label: str
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for world in worlds:
        candidates = {
            _canonical_digest(row["binding"]["packet_input_sha256"]):
            row["binding"]["packet_input_sha256"]
            for row in entries if row["world"] == world
        }
        if len(candidates) != 1:
            raise EvidenceError(
                f"{label} {world} reports do not bind one common packet input"
            )
        result.append({
            "world": world,
            "file_sha256": next(iter(candidates.values())),
        })
    return result


def _expected_public_reserve_quantities(
    entries: Sequence[Mapping[str, Any]], worlds: Sequence[str], label: str
) -> list[dict[str, Any]]:
    """Recover one verifier-computed public reserve rule per world."""

    result: list[dict[str, Any]] = []
    for world in worlds:
        candidates = {
            _canonical_digest(receipt): receipt
            for row in entries if row["world"] == world
            for receipt in [_reserve_rule_evidence(row["report"])]
        }
        if len(candidates) != 1:
            raise EvidenceError(
                f"{label} {world} reports do not bind one public reserve rule"
            )
        receipt = next(iter(candidates.values()))
        result.append({
            "world": world,
            "latest_year_total_exposure": receipt["exposure_person_years"],
            "reserve_total": receipt["reserve_total"],
        })
    return result


def _validate_reserve_red_team_measurement(
    audit: Mapping[str, Any] | None,
    references: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(audit, Mapping) or audit.get("schema") != RESERVE_RED_TEAM_SCHEMA:
        raise EvidenceError(f"a complete {RESERVE_RED_TEAM_SCHEMA} measurement is required")
    try:
        normalized = json.loads(json.dumps(audit, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("reserve red-team measurement must be finite JSON") from exc
    if set(normalized) != RESERVE_RED_TEAM_MEASUREMENT_KEYS:
        raise EvidenceError("reserve red-team measurement fields differ")
    if normalized.get("independent_unit") != "world" \
            or normalized.get("world_counts") \
            != {"development": 12, "qualification": 6, "total": 18} \
            or normalized.get("reserve_total_public_rule_verified") is not True \
            or normalized.get("primary_measure") \
            != "qualification incremental regional R2 over development region means":
        raise EvidenceError("reserve red-team design differs from the registered measurement")
    regions = normalized.get("regions_per_world")
    if isinstance(regions, bool) or not isinstance(regions, int) or regions <= 0:
        raise EvidenceError("reserve red-team regions_per_world must be positive")
    if normalized.get("files_read_per_world") != [
        "participant/contract.json",
        "participant/experience_history.csv",
        "retained/continuation_liabilities.npz:liability",
    ]:
        raise EvidenceError("reserve red-team file list differs")
    if normalized.get("tail_definition") != {
        "level": 0.95,
        "quantile_rank": "ceil(level * members), one-indexed",
        "expected_shortfall": (
            "mean of all members at or above the quantile, ties included"
        ),
    }:
        raise EvidenceError("reserve red-team tail definition differs")
    if not isinstance(normalized.get("interpretation"), str) \
            or not normalized["interpretation"].strip():
        raise EvidenceError("reserve red-team interpretation is missing")

    source = normalized.get("measurement_source")
    source_path = Path(__file__).resolve().parents[1] / "scripts/red_team_reserve_total.py"
    expected_source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if not isinstance(source, Mapping) or set(source) != {"file", "sha256"} \
            or source.get("file") != "scripts/red_team_reserve_total.py" \
            or source.get("sha256") != expected_source_digest:
        raise EvidenceError("reserve red-team measurement source differs")

    bindings = normalized.get("input_bindings")
    expected_bindings = {
        "development": _expected_red_team_inputs(
            diagnostics, DEVELOPMENT_WORLDS, "development"
        ),
        "qualification": _expected_red_team_inputs(
            references, QUALIFICATION_WORLDS, "qualification"
        ),
    }
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_bindings) \
            or bindings != expected_bindings:
        raise EvidenceError(
            "reserve red-team packet inputs differ from authenticated verifier evidence"
        )

    quantities = normalized.get("public_quantities")
    if not isinstance(quantities, Mapping) or set(quantities) != {
        "development", "qualification"
    }:
        raise EvidenceError("reserve red-team public quantities are missing")
    expected_names = {
        "development": list(DEVELOPMENT_WORLDS),
        "qualification": list(QUALIFICATION_WORLDS),
    }
    expected_quantities = {
        "development": _expected_public_reserve_quantities(
            diagnostics, DEVELOPMENT_WORLDS, "development"
        ),
        "qualification": _expected_public_reserve_quantities(
            references, QUALIFICATION_WORLDS, "qualification"
        ),
    }
    for regime, names in expected_names.items():
        rows = quantities.get(regime)
        if not isinstance(rows, list) \
                or len(rows) != len(names) \
                or not all(isinstance(row, Mapping) for row in rows) \
                or [row.get("world") for row in rows] != names:
            raise EvidenceError(f"reserve red-team {regime} worlds differ")
        for row in rows:
            if set(row) != {
                "world", "latest_year_total_exposure", "reserve_total"
            }:
                raise EvidenceError(f"reserve red-team {regime} quantity fields differ")
            _audit_number(
                row.get("latest_year_total_exposure"),
                f"reserve red-team {regime} exposure",
            )
            _audit_number(
                row.get("reserve_total"), f"reserve red-team {regime} total"
            )
        for observed, expected in zip(rows, expected_quantities[regime], strict=True):
            if not math.isclose(
                float(observed["latest_year_total_exposure"]),
                float(expected["latest_year_total_exposure"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ) or not math.isclose(
                float(observed["reserve_total"]),
                float(expected["reserve_total"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise EvidenceError(
                    f"reserve red-team {regime} public quantities differ from "
                    "verifier evidence"
                )
    _validate_red_team_models(
        normalized.get("development_regional_models"),
        regions,
        "reserve red-team development",
    )
    predictive = normalized.get("qualification_predictive_regional_r2")
    if not isinstance(predictive, Mapping) or set(predictive) != {
        "q95", "es95", "per_region"
    }:
        raise EvidenceError("reserve red-team predictive R2 fields differ")
    _red_team_r2(predictive.get("q95"), "reserve red-team predictive q95")
    _red_team_r2(predictive.get("es95"), "reserve red-team predictive ES95")
    per_region = predictive.get("per_region")
    if not isinstance(per_region, Mapping) or set(per_region) != {"q95", "es95"}:
        raise EvidenceError("reserve red-team per-region R2 fields differ")
    for outcome in ("q95", "es95"):
        values = per_region.get(outcome)
        if not isinstance(values, list) or len(values) != regions:
            raise EvidenceError("reserve red-team per-region R2 count differs")
        for value in values:
            if value is not None:
                _red_team_r2(value, "reserve red-team per-region R2")
    _validate_red_team_headline(
        normalized.get("qualification_incremental_regional_r2_over_region_means"),
        "reserve red-team primary measurement",
    )
    descriptive = normalized.get("descriptive_pooled_regional_r2")
    if not isinstance(descriptive, Mapping) or set(descriptive) != {
        "q95", "es95", "headline_max", "models"
    }:
        raise EvidenceError("reserve red-team descriptive R2 fields differ")
    _validate_red_team_headline(
        {key: descriptive.get(key) for key in ("q95", "es95", "headline_max")},
        "reserve red-team descriptive measurement",
    )
    _validate_red_team_models(
        descriptive.get("models"), regions, "reserve red-team pooled"
    )
    aggregate = normalized.get("world_aggregate_tail_r2")
    if not isinstance(aggregate, Mapping) or set(aggregate) != {
        "qualification_predictive", "descriptive_pooled"
    }:
        raise EvidenceError("reserve red-team aggregate R2 fields differ")
    for key in ("qualification_predictive", "descriptive_pooled"):
        _validate_red_team_headline(
            aggregate.get(key), f"reserve red-team aggregate {key}"
        )
    return normalized


def _bind_reserve_red_team_measurement(
    measurement: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    measurement_contract_digest: str,
) -> dict[str, Any]:
    bound = dict(measurement)
    bound["measurement_contract_digest_sha256"] = measurement_contract_digest
    bound["evidence_cross_binding"] = {
        "qualification_reference_evidence_ids": sorted(
            row["evidence_id"] for row in references
        ),
        "development_diagnostic_evidence_ids": sorted(
            row["evidence_id"] for row in diagnostics
        ),
    }
    bound["digest_sha256"] = _canonical_digest(bound)
    return bound


def _build_reserve_qualification_audit(
    audit: Mapping[str, Any] | None,
    references: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    measurement_contract_digest: str,
    calibration: Mapping[str, Any],
    red_team: Mapping[str, Any],
) -> dict[str, Any]:
    ceilings = {
        component: gates["reserve_skill"]["components"][component]["value"]
        for component in GATE_COMPONENTS["reserve_skill"]
    }

    def result_row(source: Mapping[str, Any], identity_field: str) -> dict[str, Any]:
        receipt = source["reserve_q95_feasibility"]
        return {
            identity_field: source[identity_field],
            "world": source["world"],
            "evidence_id": source["evidence_id"],
            "q95_feasible": receipt["feasible"],
            "reserve_skill_pass": all(
                source["metrics"]["reserve_skill"][component] <= ceilings[component]
                for component in GATE_COMPONENTS["reserve_skill"]
            ),
            **{
                field: receipt[field]
                for field in (
                    "q95_sum", "allocation_sum", "reserve_total",
                    "total_minus_q95_sum",
                )
            },
        }

    proportional = sorted(
        (row for row in controls if row["control"] == "proportional_reserve"),
        key=lambda row: row["world"],
    )
    generated = {
        "schema": RESERVE_QUALIFICATION_SCHEMA,
        "measurement_contract_digest_sha256": measurement_contract_digest,
        "reference_lines": list(REFERENCE_LINES),
        "qualification_worlds": list(QUALIFICATION_WORLDS),
        "calibration_audit_digest_sha256": calibration["digest_sha256"],
        "red_team_audit_digest_sha256": red_team["digest_sha256"],
        "reference_results": [
            result_row(row, "reference_line")
            for row in sorted(
                references, key=lambda row: (row["reference_line"], row["world"])
            )
        ],
        "proportional_reserve_results": [
            result_row(row, "control") for row in proportional
        ],
    }
    generated["digest_sha256"] = _canonical_digest(generated)
    if audit is None:
        return generated
    supplied = _digest_bound_audit(
        audit,
        schema=RESERVE_QUALIFICATION_SCHEMA,
        label="reserve qualification audit",
        measurement_contract_digest=measurement_contract_digest,
    )
    if supplied != generated:
        raise EvidenceError(
            "supplied reserve qualification audit differs from the deterministic audit"
        )
    return supplied


def _valid_shock_ranges(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(record, Mapping) and isinstance(record.get("kind"), str)
        and bool(record["kind"])
        and isinstance(record.get("range"), list) and len(record["range"]) == 2
        and all(not isinstance(item, bool) and isinstance(item, (int, float))
                and math.isfinite(float(item)) for item in record["range"])
        and 0.0 < float(record["range"][0]) <= float(record["range"][1])
        for record in value
    )


def _validate_mortality_identification_audit(
    audit: Mapping[str, Any] | None,
    references: Sequence[Mapping[str, Any]],
    regime_audit: Mapping[str, Any],
    worlds: Sequence[str],
) -> dict[str, Any]:
    """Validate measured P4 mortality evidence without frozen v9 world values."""

    if not isinstance(audit, Mapping) \
            or audit.get("schema") != MORTALITY_IDENTIFICATION_AUDIT_SCHEMA:
        raise EvidenceError(
            f"a complete {MORTALITY_IDENTIFICATION_AUDIT_SCHEMA} report is required"
        )
    try:
        normalized = json.loads(json.dumps(audit, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("mortality identification audit must be finite JSON") from exc
    recorded_digest = normalized.pop("digest_sha256", None)
    if not isinstance(recorded_digest, str) \
            or recorded_digest != _canonical_digest(normalized):
        raise EvidenceError("mortality identification audit digest differs from its content")
    if normalized.get("supports_gate") != "tail_calibration" \
            or normalized.get("qualification_worlds") != list(worlds):
        raise EvidenceError("mortality identification audit design differs from the freeze")
    source = normalized.get("measurement_source")
    source_path = Path(__file__).resolve().parents[1] / "meridia/methods/phase_three.py"
    try:
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvidenceError("mortality identification measurement source is unavailable") \
            from exc
    if not isinstance(source, Mapping) or set(source) != {"file", "sha256", "function"} \
            or source.get("file") != "meridia/methods/phase_three.py" \
            or source.get("function") != "mortality_gap_decomposition" \
            or source.get("sha256") != source_digest:
        raise EvidenceError("mortality identification measurement source differs")

    regime_bindings = {
        row.get("world"): row
        for row in regime_audit.get("world_bindings", [])
        if isinstance(row, Mapping)
    }
    reference_by_world: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for reference in references:
        reference_by_world[reference["world"]].append(reference)
    rows = normalized.get("worlds")
    if not isinstance(rows, list) or len(rows) != len(worlds) \
            or [row.get("world") for row in rows if isinstance(row, Mapping)] != list(worlds):
        raise EvidenceError("mortality identification audit needs qual-0 through qual-5")
    decomposition_fields = {
        "hidden_mortality_improvement",
        "trend_active_during_public_experience_window",
        "trend_starts_only_after_public_window",
        "trend_application",
        "history_mortality_rate",
        "horizon_mortality_rate",
        "observed_horizon_to_history_ratio",
        "trend_only_horizon_to_history_ratio",
        "residual_observed_to_trend_ratio",
        "publication_lag_months",
        "last_exposure_midpoint_to_snapshot_months",
        "publication_lag_trend_factor",
        "last_exposure_midpoint_to_snapshot_trend_factor",
        "history_mortality_shock_years",
        "lag_mortality_shock_years",
        "designated_horizon_mortality_shock_years",
        "continuation_shocks_redrawn_per_member",
    }
    decompositions = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "world", "packet_manifest_digest_sha256", "packet_input_sha256",
            "reference_evidence_ids", "shock_redraw_evidence", "decomposition",
        }:
            raise EvidenceError("mortality identification world binding is invalid")
        world = row["world"]
        regime_binding = regime_bindings.get(world)
        if not isinstance(regime_binding, Mapping) \
                or row.get("packet_manifest_digest_sha256") \
                != regime_binding.get("packet_manifest_digest_sha256"):
            raise EvidenceError(f"{world}: mortality audit packet manifest is not cross-bound")
        inputs = row.get("packet_input_sha256")
        if not isinstance(inputs, Mapping) or set(inputs) != set(RED_TEAM_INPUT_FILES) \
                or any(not isinstance(value, str) or len(value) != 64
                       or any(character not in "0123456789abcdef" for character in value)
                       for value in inputs.values()):
            raise EvidenceError(f"{world}: mortality audit packet inputs are invalid")
        references_for_world = reference_by_world[world]
        expected_ids = {
            reference["reference_line"]: reference["evidence_id"]
            for reference in references_for_world
        }
        if row.get("reference_evidence_ids") != expected_ids:
            raise EvidenceError(f"{world}: mortality audit reference IDs differ")
        if any(reference["binding"]["packet_input_sha256"] != inputs
               for reference in references_for_world):
            raise EvidenceError(f"{world}: mortality audit packet inputs are not cross-bound")
        shock = row.get("shock_redraw_evidence")
        try:
            from meridia.packet import _validate_shock_redraw_evidence

            runtime = _validate_shock_redraw_evidence(
                shock.get("runtime_evidence") if isinstance(shock, Mapping) else None
            )
        except (ImportError, TypeError, ValueError) as exc:
            raise EvidenceError(f"{world}: mortality shock redraw evidence is invalid") \
                from exc
        if not isinstance(shock, Mapping) or set(shock) != {
            "schema", "runtime_evidence_file_sha256", "liability_archive_sha256",
            "runtime_evidence",
        } or shock.get("schema") != SHOCK_REDRAW_REPORT_SCHEMA \
                or shock.get("liability_archive_sha256") \
                != inputs["retained/continuation_liabilities.npz"] \
                or any(
                    reference["binding"][
                        "continuation_shock_redraw_evidence_digest_sha256"
                    ] != _canonical_digest(shock)
                    or reference["binding"]["continuation_shock_redraw_file_sha256"]
                    != shock.get("runtime_evidence_file_sha256")
                    or reference["binding"]["continuation_source_law_sha256"]
                    != runtime["continuation_source_law_sha256"]
                    for reference in references_for_world
                ) \
                or runtime["redrawn_member_count"] != runtime["member_count"] \
                or runtime["distinct_future_schedule_count"] <= 1 \
                or not 0 < runtime["future_shock_year_count"] \
                < runtime["future_year_opportunity_count"] \
                or runtime["future_mortality_spike_year_count"] <= 0:
            raise EvidenceError(
                f"{world}: mortality shock redraw evidence is not cross-bound"
            )
        decomposition = row.get("decomposition")
        if not isinstance(decomposition, Mapping) or set(decomposition) != decomposition_fields:
            raise EvidenceError(f"{world}: mortality decomposition fields differ")
        _audit_number(
            decomposition.get("hidden_mortality_improvement"),
            f"{world}: hidden_mortality_improvement",
            low=-1.0,
            high=1.0,
        )
        for field in (
            "history_mortality_rate",
            "horizon_mortality_rate", "observed_horizon_to_history_ratio",
            "trend_only_horizon_to_history_ratio",
            "residual_observed_to_trend_ratio", "publication_lag_trend_factor",
            "last_exposure_midpoint_to_snapshot_trend_factor",
        ):
            _audit_number(decomposition.get(field), f"{world}: {field}")
        if decomposition.get("trend_active_during_public_experience_window") is not True \
                or decomposition.get("trend_starts_only_after_public_window") is not False \
                or decomposition.get("trend_application") \
                != "all event months relative to the snapshot tick" \
                or decomposition.get("publication_lag_months") != 12 \
                or decomposition.get("last_exposure_midpoint_to_snapshot_months") != 18 \
                or decomposition.get("continuation_shocks_redrawn_per_member") is not True:
            raise EvidenceError(f"{world}: mortality timing evidence differs")
        for field in (
            "history_mortality_shock_years", "lag_mortality_shock_years",
            "designated_horizon_mortality_shock_years",
        ):
            years = decomposition.get(field)
            if not isinstance(years, list) \
                    or any(isinstance(year, bool) or not isinstance(year, int)
                           or year < 0 for year in years) \
                    or len(years) != len(set(years)):
                raise EvidenceError(f"{world}: mortality shock years are invalid")
        decompositions[world] = decomposition

    summary = normalized.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != {
        "trend_active_during_public_experience_window",
        "trend_starts_only_after_publication",
        "publication_lag_months",
        "publication_lag_trend_effect_percent_range",
        "shock_annual_probability",
        "continuation_shocks_redrawn_per_member",
    }:
        raise EvidenceError("mortality identification summary fields differ")
    lag_effects = [
        100.0 * (float(row["publication_lag_trend_factor"]) - 1.0)
        for row in decompositions.values()
    ]
    observed_range = summary.get("publication_lag_trend_effect_percent_range")
    if not isinstance(observed_range, list) or len(observed_range) != 2:
        raise EvidenceError("mortality identification lag-effect range is invalid")
    observed_range_values = [
        _audit_number(value, "mortality identification lag-effect range", low=-1e9)
        for value in observed_range
    ]
    if summary.get("trend_active_during_public_experience_window") is not True \
            or summary.get("trend_starts_only_after_publication") is not False \
            or summary.get("publication_lag_months") != [12] \
            or not all(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                       for actual, expected in zip(
                           observed_range_values,
                           [min(lag_effects), max(lag_effects)],
                           strict=True,
                       )) \
            or summary.get("shock_annual_probability") != 0.20 \
            or summary.get("continuation_shocks_redrawn_per_member") is not True:
        raise EvidenceError("mortality identification summary does not recompute")
    normalized["digest_sha256"] = recorded_digest
    return normalized


def _validate_elder_audit(audit: Mapping[str, Any] | None,
                          references: Sequence[Mapping[str, Any]],
                          lines: Sequence[str], worlds: Sequence[str],
                          mortality_audit: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(audit, Mapping) or audit.get("schema") != ELDER_AUDIT_SCHEMA:
        raise EvidenceError(f"a complete {ELDER_AUDIT_SCHEMA} report is required")
    try:
        normalized = json.loads(json.dumps(audit, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("elder reconstruction audit must be finite JSON") from exc
    recorded_digest = normalized.pop("digest_sha256", None)
    if recorded_digest is not None and (
        not isinstance(recorded_digest, str)
        or recorded_digest != _canonical_digest(normalized)
    ):
        raise EvidenceError("elder reconstruction audit digest differs from its content")
    method = normalized.get("method_digest")
    if not isinstance(method, Mapping):
        raise EvidenceError("elder audit method digest is missing")
    before_line = _identifier(method.get("before_line"), "elder audit before_line")
    after_line = _identifier(method.get("after_line"), "elder audit after_line")
    source_digest = _sha256(method.get("source_sha256"), "elder audit source_sha256")
    commit = _identifier(method.get("git_commit"), "elder audit git_commit").lower()
    if len(commit) < 7 or len(commit) > 40 \
            or any(character not in "0123456789abcdef" for character in commit):
        raise EvidenceError("elder audit git_commit must be a hexadecimal commit id")
    if before_line not in lines or after_line not in lines or before_line == after_line:
        raise EvidenceError("elder audit lines are absent from the reference design")
    if before_line != "A" or after_line != "C":
        raise EvidenceError("elder audit must compare reference line A with elder line C")
    after_digests = {
        row["binding"]["method_digest_sha256"] for row in references
        if row["reference_line"] == after_line
    }
    if after_digests != {source_digest}:
        raise EvidenceError("elder audit source digest differs from the third reference line")

    shock = normalized.get("shock_redraw")
    if not isinstance(shock, Mapping) or shock.get("annual_probability") != 0.20 \
            or shock.get("independent_per_member") is not True \
            or shock.get("magnitude_source") != "participant/contract.json:shock_family" \
            or not _valid_shock_ranges(shock.get("mortality_ranges")) \
            or not _valid_shock_ranges(shock.get("admission_ranges")) \
            or shock.get("mortality_ranges") != EXPECTED_MORTALITY_SHOCK_RANGES \
            or shock.get("admission_ranges") != EXPECTED_ADMISSION_SHOCK_RANGES:
        raise EvidenceError("elder audit shock redraw does not match the public family")
    eligibility = normalized.get("eligibility_audit")
    scored = eligibility.get("scored") if isinstance(eligibility, Mapping) else None
    if not isinstance(scored, Mapping) or scored.get("age_band") != "65+" \
            or scored.get("floor_person_years") != 500 \
            or eligibility.get("report_only") != ["65-74", "75-84", "85+"] \
            or eligibility.get("younger_floors_changed") is not False:
        raise EvidenceError("elder audit eligibility rule differs from the frozen decision")

    rows = normalized.get("worlds")
    if not isinstance(rows, list) or len(rows) != len(worlds) \
            or [row.get("world") for row in rows if isinstance(row, Mapping)] != list(worlds):
        raise EvidenceError("elder audit must contain qual-0 through qual-5 in order")
    reference_by_key = {
        (row["reference_line"], row["world"]): row for row in references
    }
    mortality_by_world = {
        row["world"]: row["decomposition"] for row in mortality_audit["worlds"]
    }
    before_errors: list[float] = []
    after_errors: list[float] = []
    for row in rows:
        world = row["world"]
        before_reference = reference_by_key[(before_line, world)]
        after_reference = reference_by_key[(after_line, world)]
        if row.get("before_report_evidence_id") != before_reference["evidence_id"] \
                or row.get("after_report_evidence_id") != after_reference["evidence_id"]:
            raise EvidenceError(f"{world}: elder audit is not bound to its verifier reports")
        exposure = row.get("exposure_65_plus_absolute_error_percent")
        if not isinstance(exposure, Mapping) \
                or exposure.get("definition") != ELDER_EXPOSURE_ERROR_DEFINITION:
            raise EvidenceError(f"{world}: elder exposure error definition differs")
        before_error = _audit_number(exposure.get("before"), f"{world}: before exposure error")
        after_error = _audit_number(exposure.get("after"), f"{world}: after exposure error")
        before_elder = before_reference["binding"].get("elder_reference_evidence")
        after_elder = after_reference["binding"].get("elder_reference_evidence")
        if not isinstance(before_elder, Mapping) or not isinstance(after_elder, Mapping):
            raise EvidenceError(f"{world}: elder audit lacks authenticated reference detail")
        before_states = {
            item["state"]: item
            for item in before_elder["state_65_plus_person_years"]
        }
        after_states = {
            item["state"]: item
            for item in after_elder["state_65_plus_person_years"]
        }
        state_rows = row.get("state_65_plus_person_years")
        if not isinstance(state_rows, list) or len(state_rows) != 6 \
                or {item.get("state") for item in state_rows
                    if isinstance(item, Mapping)} != set(range(6)):
            raise EvidenceError(f"{world}: state elder exposure rows are incomplete")
        denominator = 0.0
        before_numerator = 0.0
        after_numerator = 0.0
        for item in state_rows:
            state = item.get("state")
            sealed = _audit_number(item.get("sealed"), f"{world}: sealed exposure")
            submitted_before = _audit_number(
                item.get("submitted_before"), f"{world}: before exposure")
            submitted_after = _audit_number(
                item.get("submitted_after"), f"{world}: after exposure")
            expected_before = before_states[state]
            expected_after = after_states[state]
            if not math.isclose(
                sealed, expected_before["sealed_person_years"],
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                sealed, expected_after["sealed_person_years"],
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                submitted_before, expected_before["submitted_person_years"],
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                submitted_after, expected_after["submitted_person_years"],
                rel_tol=1e-12, abs_tol=1e-9,
            ):
                raise EvidenceError(
                    f"{world}: elder exposure values differ from authenticated reports"
                )
            denominator += sealed
            before_numerator += abs(submitted_before - sealed)
            after_numerator += abs(submitted_after - sealed)
        if denominator <= 0.0 \
                or not math.isclose(before_error, 100.0 * before_numerator / denominator,
                                    rel_tol=1e-9, abs_tol=1e-9) \
                or not math.isclose(after_error, 100.0 * after_numerator / denominator,
                                    rel_tol=1e-9, abs_tol=1e-9):
            raise EvidenceError(f"{world}: elder exposure error does not recompute")
        liability = row.get("liability_mean_by_region")
        if not isinstance(liability, list) or len(liability) != 6 \
                or {item.get("region") for item in liability
                    if isinstance(item, Mapping)} != set(range(6)):
            raise EvidenceError(f"{world}: liability mean rows are incomplete")
        before_liability = {
            item["region"]: item
            for item in before_elder["liability_mean_by_region"]
        }
        after_liability = {
            item["region"]: item
            for item in after_elder["liability_mean_by_region"]
        }
        for item in liability:
            for field in ("submitted_before", "submitted_after", "sealed"):
                _audit_number(item.get(field), f"{world}: liability {field}")
            region = item.get("region")
            expected_before = before_liability[region]
            expected_after = after_liability[region]
            if not math.isclose(
                item["sealed"], expected_before["sealed"],
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                item["sealed"], expected_after["sealed"],
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                item["submitted_before"], expected_before["submitted"],
                rel_tol=1e-12, abs_tol=1e-9,
            ) or not math.isclose(
                item["submitted_after"], expected_after["submitted"],
                rel_tol=1e-12, abs_tol=1e-9,
            ):
                raise EvidenceError(
                    f"{world}: liability values differ from authenticated reports"
                )
        exceedance = row.get("pooled_exceedance_deviation")
        if not isinstance(exceedance, Mapping) \
                or exceedance.get("definition") != POOLED_EXCEEDANCE_DEFINITION:
            raise EvidenceError(f"{world}: pooled exceedance definition differs")
        before_exceedance = _audit_number(
            exceedance.get("before"), f"{world}: before pooled exceedance", high=0.95)
        after_exceedance = _audit_number(
            exceedance.get("after"), f"{world}: after pooled exceedance", high=0.95)
        if not math.isclose(
            before_exceedance,
            before_reference["metrics"]["tail_calibration"]["pooled_exceedance_deviation"],
            rel_tol=1e-12, abs_tol=1e-12,
        ) or not math.isclose(
            after_exceedance,
            after_reference["metrics"]["tail_calibration"]["pooled_exceedance_deviation"],
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise EvidenceError(f"{world}: pooled exceedance differs from the verifier")
        decomposition = row.get("mortality_gap_decomposition")
        if not isinstance(decomposition, Mapping):
            raise EvidenceError(f"{world}: mortality decomposition is missing")
        if _canonical_digest(decomposition) \
                != _canonical_digest(mortality_by_world.get(world)):
            raise EvidenceError(
                f"{world}: elder mortality decomposition differs from measured P4 evidence"
            )
        before_errors.append(before_error)
        after_errors.append(after_error)
    if statistics.median(after_errors) >= 10.0:
        raise EvidenceError("third-line median elder exposure error is not single digit")
    if statistics.median(after_errors) >= statistics.median(before_errors):
        raise EvidenceError("third line does not improve median elder exposure error")
    normalized["digest_sha256"] = _canonical_digest(normalized)
    return normalized


def _validate_regime_identifiability_audit(
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the observable-anchor or in-band disposition for every regime axis."""

    def correlation(value: Any, label: str) -> float:
        if isinstance(value, bool):
            raise EvidenceError(f"{label} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"{label} must be numeric") from exc
        if not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise EvidenceError(f"{label} must be in [-1, 1]")
        return number

    def observed_ranges(value: Any, label: str) -> dict[str, list[float]]:
        if not isinstance(value, Mapping) \
                or set(value) != {"pooled", "development", "hidden"}:
            raise EvidenceError(
                f"{label} must contain pooled, development, and hidden ranges"
            )
        ranges: dict[str, list[float]] = {}
        for family in ("pooled", "development", "hidden"):
            pair = value[family]
            if not isinstance(pair, list) or len(pair) != 2:
                raise EvidenceError(f"{label} {family} range must contain two values")
            low = _audit_number(
                pair[0], f"{label} {family} minimum", low=-math.inf
            )
            high = _audit_number(
                pair[1], f"{label} {family} maximum", low=-math.inf
            )
            if low > high:
                raise EvidenceError(f"{label} {family} range is reversed")
            ranges[family] = [low, high]
        union = [
            min(ranges["development"][0], ranges["hidden"][0]),
            max(ranges["development"][1], ranges["hidden"][1]),
        ]
        if ranges["pooled"] != union:
            raise EvidenceError(
                f"{label} pooled range is not the union of its two regimes"
            )
        return ranges

    def require_inside(
        observed: list[float],
        envelope: tuple[float, float],
        label: str,
    ) -> None:
        tolerance = 1e-12
        if not (
            envelope[0] - tolerance <= observed[0] <= observed[1]
            <= envelope[1] + tolerance
        ):
            raise EvidenceError(
                f"{label} {observed} lies outside registered envelope "
                f"{list(envelope)}"
            )

    if not isinstance(audit, Mapping) \
            or audit.get("schema") != REGIME_IDENTIFIABILITY_SCHEMA:
        raise EvidenceError(f"a complete {REGIME_IDENTIFIABILITY_SCHEMA} report is required")
    try:
        normalized = json.loads(json.dumps(audit, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceError("regime identifiability audit must be finite JSON") from exc
    if set(normalized) != {
        "schema", "anchor_correlation_threshold", "binding_axis",
        "hidden_regime_correlation_shortfalls", "world_count", "world_bindings",
        "measurement_rows_digest_sha256", "generator_source_digest_sha256",
        "generator_policy", "axes", "digest_sha256",
    }:
        raise EvidenceError("regime identifiability audit fields differ from schema v3")
    recorded_digest = normalized.pop("digest_sha256", None)
    if not isinstance(recorded_digest, str) \
            or recorded_digest != _canonical_digest(normalized):
        raise EvidenceError("regime identifiability audit digest differs from its content")
    if normalized.get("anchor_correlation_threshold") != ANCHOR_CORRELATION_THRESHOLD \
            or normalized.get("world_count") != 18 \
            or not _sha256(normalized.get("measurement_rows_digest_sha256"),
                           "measurement_rows_digest_sha256") \
            or not _sha256(normalized.get("generator_source_digest_sha256"),
                           "generator_source_digest_sha256"):
        raise EvidenceError("regime identifiability audit design differs from the freeze")
    source_root = Path(__file__).resolve().parents[1]
    try:
        expected_source_digest = _canonical_digest([
            {
                "path": relative,
                "sha256": hashlib.sha256((source_root / relative).read_bytes()).hexdigest(),
            }
            for relative in IDENTIFIABILITY_SOURCE_FILES
        ])
    except OSError as exc:
        raise EvidenceError("identifiability generator source is unavailable") from exc
    if normalized.get("generator_source_digest_sha256") != expected_source_digest:
        raise EvidenceError("identifiability generator source digest differs")
    bindings = normalized.get("world_bindings")
    expected_worlds = {
        **{f"dev-{index:02d}": "development" for index in range(12)},
        **{f"qual-{index}": "hidden" for index in range(6)},
    }
    if not isinstance(bindings, list) or len(bindings) != len(expected_worlds):
        raise EvidenceError("regime identifiability audit needs twelve development and six qualification worlds")
    seen: dict[str, str] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != {
            "world", "regime", "participant_digest_sha256",
            "packet_manifest_digest_sha256",
        }:
            raise EvidenceError("regime identifiability world bindings must be objects")
        world = _identifier(binding.get("world"), "identifiability world")
        regime = _identifier(binding.get("regime"), f"{world} regime")
        _sha256(binding.get("participant_digest_sha256"), f"{world} participant digest")
        _sha256(binding.get("packet_manifest_digest_sha256"), f"{world} manifest digest")
        if world in seen:
            raise EvidenceError(f"{world}: duplicate identifiability world binding")
        seen[world] = regime
    if seen != expected_worlds:
        raise EvidenceError("regime identifiability world names or regimes differ")
    policy = normalized.get("generator_policy")
    expected_policy = {
        "outside_axis_count": 2,
        "eligible_for_outside_development_band": list(HIDDEN_EXTRAPOLATION_AXES),
        "held_inside_development_band": list(HIDDEN_IN_BAND_AXES),
    }
    if policy != expected_policy:
        raise EvidenceError("hidden-axis generator policy differs from the registered constraint")
    axes = normalized.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != set(REGIME_AXES):
        raise EvidenceError("regime identifiability audit must contain all six axes")
    for axis in REGIME_AXES:
        record = axes[axis]
        if not isinstance(record, Mapping) or set(record) != {
            "statistic", "expected_sign", "signed_rank_correlation",
            "within_regime_signed_rank_correlation", "correlation_target",
            "realized_mechanism_definition", "axis_intensity_range_observed",
            "realized_mechanism_range_observed",
            "registered_realized_mechanism_envelopes",
            "anchor_correlation_qualified", "hidden_regime_correlation_qualified",
            "disposition", "development_range",
            "hidden_generation_range", "hidden_out_of_band_allowed",
        }:
            raise EvidenceError(f"{axis}: identifiability record must be an object")
        signed = correlation(
            record.get("signed_rank_correlation"),
            f"{axis}: signed rank correlation",
        )
        qualified = signed > ANCHOR_CORRELATION_THRESHOLD
        within = record.get("within_regime_signed_rank_correlation")
        _identifier(record.get("statistic"), f"{axis}: statistic")
        if record.get("expected_sign") != REGIME_EXPECTED_SIGNS[axis] \
                or not isinstance(within, Mapping) or set(within) != {"development", "hidden"} \
                or record.get("anchor_correlation_qualified") is not qualified \
                or record.get("correlation_target") != "realized_mechanism" \
                or record.get("realized_mechanism_definition") \
                != REALIZED_MECHANISM_DEFINITIONS[axis] \
                or record.get("development_range") != list(DEVELOPMENT_AXIS_RANGES[axis]):
            raise EvidenceError(f"{axis}: identifiability measurement is invalid")
        for regime, value in within.items():
            correlation(value, f"{axis}: {regime} signed rank correlation")
        hidden_qualified = float(within["hidden"]) > ANCHOR_CORRELATION_THRESHOLD
        if record.get("hidden_regime_correlation_qualified") is not hidden_qualified:
            raise EvidenceError(
                f"{axis}: hidden-regime correlation qualification does not match its value"
            )
        axis_intensity_ranges = observed_ranges(
            record.get("axis_intensity_range_observed"),
            f"{axis}: raw axis intensity",
        )
        realized_mechanism_ranges = observed_ranges(
            record.get("realized_mechanism_range_observed"),
            f"{axis}: realized mechanism",
        )
        expected_realized_envelopes = {
            family: list(bounds)
            for family, bounds in REALIZED_MECHANISM_ENVELOPES[axis].items()
        }
        if record.get("registered_realized_mechanism_envelopes") \
                != expected_realized_envelopes:
            raise EvidenceError(
                f"{axis}: realized mechanism envelopes differ from registration"
            )
        require_inside(
            axis_intensity_ranges["development"],
            DEVELOPMENT_AXIS_RANGES[axis],
            f"{axis}: development raw axis intensity",
        )
        require_inside(
            axis_intensity_ranges["pooled"],
            PUBLIC_AXIS_RANGES[axis],
            f"{axis}: pooled raw axis intensity",
        )
        require_inside(
            realized_mechanism_ranges["development"],
            REALIZED_MECHANISM_ENVELOPES[axis]["development"],
            f"{axis}: development realized mechanism",
        )
        require_inside(
            realized_mechanism_ranges["pooled"],
            REALIZED_MECHANISM_ENVELOPES[axis]["public"],
            f"{axis}: pooled realized mechanism",
        )
        if axis in HIDDEN_IN_BAND_AXES:
            low, high = DEVELOPMENT_AXIS_RANGES[axis]
            if record.get("disposition") != "constrained_to_development_range" \
                    or record.get("hidden_out_of_band_allowed") is not False \
                    or record.get("hidden_generation_range") != [low, high]:
                raise EvidenceError(f"{axis}: unanchored axis is not held in range")
            require_inside(
                axis_intensity_ranges["hidden"],
                DEVELOPMENT_AXIS_RANGES[axis],
                f"{axis}: hidden raw axis intensity",
            )
            require_inside(
                realized_mechanism_ranges["hidden"],
                REALIZED_MECHANISM_ENVELOPES[axis]["development"],
                f"{axis}: hidden realized mechanism",
            )
        else:
            if not qualified \
                    or record.get("disposition") != "participant_anchor" \
                    or record.get("hidden_out_of_band_allowed") is not True \
                    or record.get("hidden_generation_range") \
                    != list(PUBLIC_AXIS_RANGES[axis]):
                raise EvidenceError(f"{axis}: extrapolated axis lacks a 0.4 participant trace")
            require_inside(
                axis_intensity_ranges["hidden"],
                PUBLIC_AXIS_RANGES[axis],
                f"{axis}: hidden raw axis intensity",
            )
            require_inside(
                realized_mechanism_ranges["hidden"],
                REALIZED_MECHANISM_ENVELOPES[axis]["public"],
                f"{axis}: hidden realized mechanism",
            )

    # The pooled correlation runs over twelve development worlds and six hidden ones, so
    # an axis can clear it on the development block alone while carrying less trace in the
    # six worlds a submission is scored on. The registered gate stays on the pooled
    # eighteen-world reading, and the receipt has to name the axis that binds it rather
    # than leaving a reader to find it.
    binding = min(
        REGIME_AXES,
        key=lambda name: float(axes[name]["signed_rank_correlation"]),
    )
    recorded_binding = normalized.get("binding_axis")
    if not isinstance(recorded_binding, Mapping) \
            or set(recorded_binding) != {"axis", "signed_rank_correlation"} \
            or recorded_binding.get("axis") != binding \
            or float(recorded_binding.get("signed_rank_correlation")) \
            != float(axes[binding]["signed_rank_correlation"]):
        raise EvidenceError(
            "regime identifiability audit does not name the axis that binds the pooled rule"
        )
    shortfalls = sorted(
        axis for axis in HIDDEN_EXTRAPOLATION_AXES
        if axes[axis]["hidden_regime_correlation_qualified"] is not True
    )
    if normalized.get("hidden_regime_correlation_shortfalls") != shortfalls:
        raise EvidenceError(
            "regime identifiability audit does not record its hidden-regime shortfalls"
        )
    # The within-hidden reading is reported and does not refuse. Six worlds is too few
    # points for a rank correlation to carry a threshold: one world's rank changes the
    # value by more than the margin the threshold asks for, so a refusal there would be a
    # statement about six draws rather than about the anchor. The list has to be exact,
    # and every axis it names is written into the freeze report beside its value and the
    # reason it does not decide.
    normalized["digest_sha256"] = recorded_digest
    return normalized


def _eligible_cells(report: Mapping[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    metrics = report.get("composite_metrics")
    if isinstance(metrics, Mapping):
        block = metrics.get("exposures_and_rates")
        if isinstance(block, Mapping):
            cells = block.get("eligible_cells")
            if isinstance(cells, list):
                candidates.extend(cells)
            component = block.get("p95_relative_error")
            if isinstance(component, Mapping) and isinstance(component.get("eligible_cells"), list):
                candidates.extend(component["eligible_cells"])
    rates = report.get("rate_metrics")
    if isinstance(rates, Mapping) and isinstance(rates.get("composite"), Mapping):
        cells = rates["composite"].get("eligible_cells", rates["composite"].get("cells"))
        if isinstance(cells, list):
            candidates.extend(cells)
    return candidates


def _unique_json_values(values: Iterable[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for value in values:
        try:
            key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise EvidenceError("eligible-cell records must be JSON serializable") from exc
        unique[key] = value
    return [unique[key] for key in sorted(unique)]


def _eligibility_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    evidence = report.get("eligibility_evidence")
    bands = evidence.get("bands") if isinstance(evidence, Mapping) else None
    expected = {"0-17", "18-44", "45-64", "65-74", "75-84", "85+",
                "18-64", "65+"}
    if not isinstance(evidence, Mapping) \
            or evidence.get("truth_quantity") \
            != "retained state-by-sex person-years exposure" \
            or not isinstance(bands, Mapping) or set(bands) != expected:
        raise EvidenceError("state-by-sex eligibility evidence is incomplete")
    for band, record in bands.items():
        if not isinstance(record, Mapping):
            raise EvidenceError(f"eligibility evidence for {band} is not an object")
        floor = record.get("floor_person_years")
        count = record.get("cell_count")
        eligible = record.get("eligible_count")
        minimum = record.get("minimum_exposure_person_years")
        cells = record.get("cells")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) \
                or not math.isfinite(float(floor)) or float(floor) <= 0.0 \
                or isinstance(count, bool) or not isinstance(count, int) or count <= 0 \
                or isinstance(eligible, bool) or not isinstance(eligible, int) \
                or not 0 <= eligible <= count \
                or isinstance(minimum, bool) or not isinstance(minimum, (int, float)) \
                or not math.isfinite(float(minimum)) or float(minimum) < 0.0 \
                or not isinstance(cells, list) or len(cells) != count:
            raise EvidenceError(f"eligibility evidence for {band} is invalid")
        expected_pairs: set[tuple[int, str]] = set()
        values: list[float] = []
        observed_eligible = 0
        for cell in cells:
            if not isinstance(cell, Mapping):
                raise EvidenceError(f"eligibility cell for {band} is not an object")
            state = cell.get("state")
            sex = cell.get("sex")
            exposure = cell.get("exposure_person_years")
            decision = cell.get("eligible")
            if isinstance(state, bool) or not isinstance(state, int) or state < 0 \
                    or not isinstance(sex, str) or not sex \
                    or isinstance(exposure, bool) \
                    or not isinstance(exposure, (int, float)) \
                    or not math.isfinite(float(exposure)) or float(exposure) < 0.0 \
                    or not isinstance(decision, bool) \
                    or decision != (float(exposure) >= float(floor)):
                raise EvidenceError(f"eligibility cell for {band} is invalid")
            pair = (state, sex)
            if pair in expected_pairs:
                raise EvidenceError(f"eligibility cell for {band} is duplicated")
            expected_pairs.add(pair)
            values.append(float(exposure))
            observed_eligible += int(decision)
        if observed_eligible != eligible \
                or not math.isclose(min(values), float(minimum), rel_tol=0.0, abs_tol=0.0):
            raise EvidenceError(f"eligibility summary for {band} differs from its cells")
    if bands["65+"].get("status") != "scored" \
            or float(bands["65+"].get("floor_person_years")) != 500.0 \
            or bands["65+"].get("eligible_count") != bands["65+"].get("cell_count"):
        raise EvidenceError("the broad 65+ eligibility rule does not include every cell")
    for band in ("65-74", "75-84", "85+"):
        if bands[band].get("status") != "report-only":
            raise EvidenceError(f"{band} must remain report-only")
    return json.loads(json.dumps(evidence, sort_keys=True, allow_nan=False))


def _validated_eligible_cells(
    references: Sequence[Mapping[str, Any]], lines: Sequence[str], worlds: Sequence[str]
) -> tuple[dict[str, list[Any]], dict[str, dict[str, Any]]]:
    """Require truth-defined eligible-cell provenance to agree across reference lines."""

    by_pair: dict[tuple[str, str], list[Any]] = {}
    audit_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in references:
        cells = _unique_json_values(_eligible_cells(entry["report"]))
        if not cells:
            raise EvidenceError(
                f"eligible-cell evidence missing for "
                f"{entry['reference_line']}/{entry['world']}"
            )
        by_pair[(entry["reference_line"], entry["world"])] = cells
        audit_by_pair[(entry["reference_line"], entry["world"])] = \
            _eligibility_evidence(entry["report"])
    by_world: dict[str, list[Any]] = {}
    audit_by_world: dict[str, dict[str, Any]] = {}
    for world in worlds:
        expected = by_pair[(lines[0], world)]
        disagree = [line for line in lines[1:]
                    if by_pair[(line, world)] != expected]
        if disagree:
            raise EvidenceError(
                f"eligible-cell evidence for {world} differs across reference lines"
            )
        by_world[world] = expected
        expected_audit = audit_by_pair[(lines[0], world)]
        audit_disagree = [
            line for line in lines[1:]
            if audit_by_pair[(line, world)] != expected_audit
        ]
        if audit_disagree:
            raise EvidenceError(
                f"eligibility audit for {world} differs across reference lines"
            )
        audit_by_world[world] = expected_audit
    return by_world, audit_by_world


def _check_reference_design(references: list[dict[str, Any]],
                            replicates: list[dict[str, Any]],
                            expected_world_count: int) -> tuple[list[str], list[str], int]:
    if not references:
        raise EvidenceError("final reference reports are missing")
    if not replicates:
        raise EvidenceError(
            "replicate evidence missing; final verifier reports cannot be bootstrapped "
            "or resampled as replacement evidence"
        )
    reused = sorted(
        {entry["evidence_id"] for entry in references}
        & {entry["evidence_id"] for entry in replicates}
    )
    if reused:
        raise EvidenceError(
            f"final reports were reused as replicate evidence: {reused}"
        )
    lines = sorted({entry["reference_line"] for entry in references})
    worlds = sorted({entry["world"] for entry in references})
    if lines != list(REFERENCE_LINES):
        raise EvidenceError(
            "reference lines must be exactly A, B, and C"
        )
    if expected_world_count != EXPECTED_QUALIFICATION_WORLDS:
        raise EvidenceError(
            f"the V4 freeze requires exactly {EXPECTED_QUALIFICATION_WORLDS} "
            "qualification worlds"
        )
    if worlds != list(QUALIFICATION_WORLDS):
        raise EvidenceError(
            "qualification worlds must be exactly qual-0 through qual-5"
        )
    if len(references) != REFERENCE_REPORT_COUNT:
        raise EvidenceError(
            f"final reference evidence must contain exactly {REFERENCE_REPORT_COUNT} reports"
        )
    if len(replicates) != REPLICATE_REPORT_COUNT:
        raise EvidenceError(
            f"replicate evidence must contain exactly {REPLICATE_REPORT_COUNT} reports"
        )
    expected = {(line, world) for line in lines for world in worlds}
    final_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in references:
        final_groups[(entry["reference_line"], entry["world"])].append(entry)
    if set(final_groups) != expected or any(len(rows) != 1 for rows in final_groups.values()):
        raise EvidenceError(
            "final reference evidence must contain exactly one report for every "
            "reference-line and qualification-world pair"
        )
    replicate_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in replicates:
        replicate_groups[(entry["reference_line"], entry["world"])].append(entry)
    if set(replicate_groups) != expected:
        missing = sorted(expected - set(replicate_groups))
        unexpected = sorted(set(replicate_groups) - expected)
        raise EvidenceError(
            f"replicate groups do not match final reference groups; missing {missing}, "
            f"unexpected {unexpected}"
        )
    counts = {pair: len(rows) for pair, rows in replicate_groups.items()}
    if len(set(counts.values())) != 1:
        raise EvidenceError(
            "replicate counts must be equal for every reference-line and world pair "
            "so each pair has equal weight"
        )
    per_pair = next(iter(counts.values()))
    if per_pair != REPLICATES_PER_LINE_WORLD:
        raise EvidenceError(
            f"each reference-line and world pair needs exactly "
            f"{REPLICATES_PER_LINE_WORLD} paired deterministic replicates"
        )
    per_line = per_pair * len(worlds)
    if per_line < MIN_P99_SAMPLE_COUNT:
        raise EvidenceError(
            f"each reference line needs at least {MIN_P99_SAMPLE_COUNT} independent "
            f"replicate reports to resolve an empirical one-percent tail; found "
            f"{per_line}"
        )
    for pair, rows in replicate_groups.items():
        ids = [row["replicate_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise EvidenceError(f"duplicate replicate_id within {pair}")
    if any(not entry["hard_pass"] for entry in references):
        failed = sorted(
            f"{entry['reference_line']}/{entry['world']}"
            for entry in references if not entry["hard_pass"]
        )
        raise EvidenceError(f"final reference reports failed hard checks: {failed}")
    if any(not entry["hard_pass"] for entry in replicates):
        failed = sorted(
            f"{entry['reference_line']}/{entry['world']}/{entry['replicate_id']}"
            for entry in replicates if not entry["hard_pass"]
        )
        raise EvidenceError(f"replicate reports failed hard checks: {failed[:10]}")
    return lines, worlds, per_pair


def _calibrate_components(references: list[dict[str, Any]],
                          replicates: list[dict[str, Any]],
                          lines: list[str], worlds: list[str],
                          eligible_cells_by_world: Mapping[str, list[Any]],
                          eligibility_audit_by_world: Mapping[str, dict[str, Any]]) \
        -> dict[str, Any]:
    """Calibrate each composite jointly, independently within each reference line.

    Each metric is a dimensionless loss and has a frozen normalizer.  A replicate's gate
    severity is the maximum normalized component loss.  The empirical p99 is taken over
    the 102 independent reports for each line, then the largest of the three line p99s is
    the common gate ceiling.  Thus no line is hidden by pooling and component multiplicity
    cannot inflate the registered one-percent gate-level false-fail rate.
    """

    gates: dict[str, Any] = {}
    for gate, components in GATE_COMPONENTS.items():
        normalizers = GATE_COMPONENT_NORMALIZERS[gate]
        line_rows: dict[str, list[dict[str, Any]]] = {
            line: sorted(
                (row for row in replicates if row["reference_line"] == line),
                key=lambda row: (row["world"], row["replicate_id"]),
            )
            for line in lines
        }
        sample_count_per_line = len(worlds) * REPLICATES_PER_LINE_WORLD
        if any(len(rows) != sample_count_per_line for rows in line_rows.values()):
            raise EvidenceError(
                f"{gate}: per-line replicate counts differ from the registered design"
            )

        severity_rows: dict[str, list[dict[str, Any]]] = {}
        line_p99: dict[str, float] = {}
        for line, rows in line_rows.items():
            observed: list[dict[str, Any]] = []
            for row in rows:
                component_values = {
                    component: row["metrics"][gate][component]
                    for component in components
                }
                component_severities = {
                    component: component_values[component] / normalizers[component]
                    for component in components
                }
                observed.append({
                    "reference_line": line,
                    "world": row["world"],
                    "replicate_id": row["replicate_id"],
                    "evidence_id": row["evidence_id"],
                    "component_values": component_values,
                    "component_severities": component_severities,
                    "severity": max(component_severities.values()),
                })
            severity_rows[line] = observed
            line_p99[line] = empirical_p99([row["severity"] for row in observed])

        severity_ceiling = max(line_p99.values())
        rank = math.ceil(QUANTILE * sample_count_per_line)
        line_calibration: dict[str, Any] = {}
        for line in lines:
            observed = severity_rows[line]
            failed = sum(row["severity"] > severity_ceiling for row in observed)
            evidence_ids = sorted(row["evidence_id"] for row in observed)
            line_calibration[line] = {
                "severity_p99": line_p99[line],
                "sample_count": sample_count_per_line,
                "order_statistic_rank": rank,
                "false_fail_count": failed,
                "false_fail_rate": failed / sample_count_per_line,
                "quantile_witnesses": [
                    row for row in observed if row["severity"] == line_p99[line]
                ],
                "replicate_evidence_ids": evidence_ids,
                "replicate_evidence_digest_sha256": hashlib.sha256(
                    "\n".join(evidence_ids).encode("utf-8")
                ).hexdigest(),
            }

        component_records: dict[str, Any] = {}
        for component in components:
            normalizer = normalizers[component]
            _, attainable_high = COMPONENT_RANGES[(gate, component)]
            value = severity_ceiling * normalizer
            if attainable_high is not None:
                # A bar at the top of the component's attainable range is not a bar. No
                # submission can exceed it and no control can fail on it, so the component
                # decides nothing and the gate is weaker than its receipt reads. The
                # freeze refuses rather than publishing one.
                if value >= attainable_high or math.isclose(
                    value, attainable_high, rel_tol=1e-9, abs_tol=1e-12
                ):
                    raise EvidenceError(
                        f"{gate}/{component}: the calibrated bar "
                        f"{value:.12g} reaches its attainable ceiling "
                        f"{attainable_high:.12g}, so the component cannot fail"
                    )
                value = min(value, attainable_high)
            reference_witnesses = [
                {
                    "reference_line": entry["reference_line"],
                    "world": entry["world"],
                    "evidence_id": entry["evidence_id"],
                    "value": entry["metrics"][gate][component],
                    "pass": entry["metrics"][gate][component] <= value,
                }
                for entry in sorted(
                    references, key=lambda row: (row["reference_line"], row["world"])
                )
            ]
            evidence_ids = sorted(entry["evidence_id"] for entry in replicates)
            evidence_digest = hashlib.sha256(
                "\n".join(evidence_ids).encode("utf-8")
            ).hexdigest()
            record: dict[str, Any] = {
                "value": value,
                "direction": "ceiling",
                "range": list(COMPONENT_RANGES[(gate, component)]),
                "normalizer": normalizer,
                "calibration_method": "derived-from-joint-gate-max-severity",
                "quantile": QUANTILE,
                "target_false_fail_rate": TARGET_FALSE_FAIL_RATE,
                "sample_count": len(replicates),
                "sample_count_per_reference_line": sample_count_per_line,
                "order_statistic_rank_per_reference_line": rank,
                "worlds": worlds,
                "witnesses": lines,
                "reference_witnesses": reference_witnesses,
                "empirical_p99_by_reference_line": {
                    line: empirical_p99([
                        row["component_values"][component]
                        for row in severity_rows[line]
                    ])
                    for line in lines
                },
                "observed_range_by_reference_line": {
                    line: [
                        min(row["component_values"][component]
                            for row in severity_rows[line]),
                        max(row["component_values"][component]
                            for row in severity_rows[line]),
                    ]
                    for line in lines
                },
                "component_quantile_witnesses_by_reference_line": {
                    line: [
                        {
                            "reference_line": row["reference_line"],
                            "world": row["world"],
                            "replicate_id": row["replicate_id"],
                            "evidence_id": row["evidence_id"],
                            "value": row["component_values"][component],
                        }
                        for row in severity_rows[line]
                        if row["component_values"][component]
                        == empirical_p99([
                            item["component_values"][component]
                            for item in severity_rows[line]
                        ])
                    ]
                    for line in lines
                },
                "component_exceedance_rate_at_joint_ceiling_by_reference_line": {
                    line: sum(
                        row["component_values"][component] > value
                        for row in severity_rows[line]
                    ) / sample_count_per_line
                    for line in lines
                },
                "supporting_controls": [],
                "component_exceedance_controls": [],
                "replicate_evidence_ids": evidence_ids,
                "replicate_evidence_digest_sha256": evidence_digest,
            }
            if gate == "exposures_and_rates":
                counts: dict[str, int] = {}
                cells: list[Any] = []
                for entry in references:
                    found = _eligible_cells(entry["report"])
                    counts[f"{entry['world']}/{entry['reference_line']}"] = len(found)
                    cells.extend(found)
                record["eligible_cells"] = {
                    "distinct": _unique_json_values(cells),
                    "counts_by_world_and_witness": dict(sorted(counts.items())),
                    "by_world": dict(eligible_cells_by_world),
                    "band_audit_by_world": dict(eligibility_audit_by_world),
                }
            component_records[component] = record
        gates[gate] = {
            "calibration_method": "per-reference-line-joint-max-severity",
            "normalizers": dict(normalizers),
            "severity_ceiling": severity_ceiling,
            "quantile": QUANTILE,
            "target_false_fail_rate": TARGET_FALSE_FAIL_RATE,
            "sample_count_per_reference_line": sample_count_per_line,
            "order_statistic_rank_per_reference_line": rank,
            "ceiling_witness_lines": [
                line for line in lines if line_p99[line] == severity_ceiling
            ],
            "reference_line_calibration": line_calibration,
            "components": component_records,
            "supporting_controls": [],
        }
    return gates


def _validated_control_registry(
    registry: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    """Return the single primary gate for every registered control."""

    if set(registry) != set(GATE_COMPONENTS):
        raise EvidenceError("the scientific-control registry must name the five gates")
    primary_gate: dict[str, str] = {}
    for gate in GATE_COMPONENTS:
        names = registry[gate]
        if isinstance(names, (str, bytes)) or not names:
            raise EvidenceError(f"{gate}: registered control list must be nonempty")
        if len(names) != len(set(names)):
            raise EvidenceError(f"{gate}: registered control names must be unique")
        for raw_name in names:
            name = _identifier(raw_name, f"{gate} control")
            if name in primary_gate:
                raise EvidenceError(
                    f"{name}: a control may have only one registered primary gate"
                )
            primary_gate[name] = gate
    return primary_gate


def _control_matrix(
    controls: list[dict[str, Any]],
    gates: Mapping[str, Any],
    registry: Mapping[str, Sequence[str]],
    worlds: Sequence[str],
) -> dict[str, Any]:
    """Record exact per-world comparisons for the full qualification battery."""

    primary_gate = _validated_control_registry(registry)
    by_name: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in controls:
        by_name[entry["control"]].append(entry)
    matrix: dict[str, Any] = {}
    for name in sorted(set(primary_gate) | set(by_name)):
        rows = sorted(
            by_name.get(name, []), key=lambda row: (row["world"], row["evidence_id"])
        )
        reported_worlds = [row["world"] for row in rows]
        world_counts = {world: reported_worlds.count(world) for world in worlds}
        complete = len(rows) == len(worlds) and all(
            world_counts[world] == 1 for world in worlds
        )
        missing_worlds = [world for world in worlds if world_counts[world] == 0]
        duplicate_worlds = [world for world in worlds if world_counts[world] > 1]
        unexpected_worlds = sorted(set(reported_worlds) - set(worlds))
        gate_rows: dict[str, Any] = {}
        for gate, components in GATE_COMPONENTS.items():
            per_world: dict[str, Any] = {}
            for row in rows:
                if row["world"] not in worlds or row["world"] in per_world:
                    continue
                comparisons = {
                    component: {
                        "value": row["metrics"][gate][component],
                        "ceiling": gates[gate]["components"][component]["value"],
                        "exceeds": row["metrics"][gate][component]
                        > gates[gate]["components"][component]["value"],
                    }
                    for component in components
                }
                hard_pass = row["hard_pass"] is True
                failed = hard_pass and any(
                    comparison["exceeds"] for comparison in comparisons.values()
                )
                per_world[row["world"]] = {
                    "hard_structure_pass": hard_pass,
                    "outcome": "fail" if failed else (
                        "pass" if hard_pass else "hard-invalid"
                    ),
                    "failed": failed,
                    "components": comparisons,
                    "evidence_id": row["evidence_id"],
                    "reserve_q95_feasibility": row["reserve_q95_feasibility"],
                }
            failed_worlds = [
                world for world in worlds
                if per_world.get(world, {}).get("failed") is True
            ]
            hard_invalid_worlds = [
                world for world in worlds
                if per_world.get(world, {}).get("hard_structure_pass") is False
            ]
            passed_worlds = [
                world for world in worlds
                if per_world.get(world, {}).get("outcome") == "pass"
            ]
            gate_rows[gate] = {
                "scientifically_registered": primary_gate.get(name) == gate,
                "separates_all_worlds": complete
                and not hard_invalid_worlds
                and failed_worlds == list(worlds),
                "failed_worlds": failed_worlds,
                "passed_worlds": passed_worlds,
                "hard_invalid_worlds": hard_invalid_worlds,
                "per_world": per_world,
            }
        matrix[name] = {
            "registered": name in primary_gate,
            "primary_gate": primary_gate.get(name),
            "coverage_complete": complete,
            "worlds": reported_worlds,
            "missing_worlds": missing_worlds,
            "duplicate_worlds": duplicate_worlds,
            "unexpected_worlds": unexpected_worlds,
            "hard_structure_pass": complete and all(row["hard_pass"] for row in rows),
            "evidence_ids": [row["evidence_id"] for row in rows],
            "gates": gate_rows,
        }
    return matrix


def _profile_separation(
    matrix: Mapping[str, Any],
    primary_gate: Mapping[str, str],
    worlds: Sequence[str],
    profile: str,
    selection: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Record which controls fail a gated block on every qualification world.

    A control whose registered primary gate decides under the profile is judged on that
    gate, exactly as the registered battery is judged. A control whose primary gate does
    not decide has to fail some other gated gate on every world to separate anything, so
    a control that only fails the tail block separates nothing under a profile that does
    not gate the tail. Those controls are named as deletion candidates for the profile:
    under it they no longer test the surface a participant is scored on.
    """

    def fails_gated(record: Mapping[str, Any], gate: str,
                    components: Sequence[str], world: str) -> bool:
        row = record["gates"][gate]["per_world"].get(world)
        return isinstance(row, Mapping) \
            and row.get("hard_structure_pass") is True \
            and any(row["components"][component]["exceeds"]
                    for component in components)

    separating: list[str] = []
    candidates: list[dict[str, Any]] = []
    for name in sorted(primary_gate):
        gate = primary_gate[name]
        record = matrix[name]
        on_primary = gate in selection
        if on_primary:
            gated_failed = [world for world in worlds
                            if fails_gated(record, gate, selection[gate], world)]
        else:
            gated_failed = [
                world for world in worlds
                if any(fails_gated(record, other, components, world)
                       for other, components in selection.items())
            ]
        if record["coverage_complete"] and gated_failed == list(worlds):
            separating.append(name)
            continue
        candidates.append({
            "control": name,
            "primary_gate": gate,
            "primary_gate_decides": on_primary,
            "judged_on": gate if on_primary else "any gated gate",
            "failed_gated_worlds": gated_failed,
            "unseparated_worlds": [
                world for world in worlds if world not in gated_failed
            ],
            "missing_worlds": record["missing_worlds"],
        })
    return {
        "name": profile,
        "gated_components": {gate: list(components)
                             for gate, components in selection.items()},
        "reported_only_gates": [gate for gate in GATE_COMPONENTS
                                if gate not in selection],
        "requirement": (
            "every registered control fails a gated composite gate on every "
            "qualification world"
        ),
        "separating_controls": separating,
        "deletion_candidates": candidates,
        "deletion_candidate_controls": [record["control"] for record in candidates],
    }


def _attach_control_separation(
    gates: dict[str, Any],
    controls: list[dict[str, Any]],
    registry: Mapping[str, Sequence[str]],
    worlds: Sequence[str],
    profile: str = DEFAULT_GATE_PROFILE,
) -> dict[str, Any]:
    """Attach the all-controls-by-all-worlds separation requirement."""

    primary_gate = _validated_control_registry(registry)
    matrix = _control_matrix(controls, gates, registry, worlds)
    registered_by_gate = {
        gate: list(registry[gate]) for gate in GATE_COMPONENTS
    }
    separated_by_gate: dict[str, list[str]] = {}
    deletion_candidates: list[dict[str, Any]] = []
    for gate, names in registered_by_gate.items():
        separated = [
            name for name in names
            if matrix[name]["gates"][gate]["separates_all_worlds"] is True
        ]
        separated_by_gate[gate] = separated
        nonseparating = []
        for name in names:
            result = matrix[name]["gates"][gate]
            if result["separates_all_worlds"]:
                continue
            nonseparating.append({
                "control": name,
                "failed_worlds": result["failed_worlds"],
                "passed_worlds": result["passed_worlds"],
                "hard_invalid_worlds": result["hard_invalid_worlds"],
                "missing_worlds": matrix[name]["missing_worlds"],
                "duplicate_worlds": matrix[name]["duplicate_worlds"],
                "unexpected_worlds": matrix[name]["unexpected_worlds"],
                "per_world": result["per_world"],
            })
        if nonseparating:
            deletion_candidates.append({
                "gate": gate,
                "reason": (
                    "at least one registered wrong method does not fail this gate "
                    "on every qualification world"
                ),
                "registered_controls": names,
                "nonseparating_controls": nonseparating,
            })
        gates[gate]["supporting_controls"] = separated
        for component in GATE_COMPONENTS[gate]:
            record = gates[gate]["components"][component]
            record["supporting_controls"] = separated
            record["component_exceedance_controls"] = [
                name for name in separated
                if all(
                    matrix[name]["gates"][gate]["per_world"][world]
                    ["components"][component]["exceeds"]
                    for world in worlds
                )
            ]
    unexpected_controls = sorted(
        name for name, record in matrix.items() if record["registered"] is False
    )
    return {
        "gate_profile": _profile_separation(
            matrix, primary_gate, worlds, profile, gate_profile_selection(profile)
        ),
        "requirement": (
            "every registered control hard-passes structure and fails its primary "
            "composite gate on every qualification world"
        ),
        "registered_controls": sorted(matrix_name for matrix_name in matrix
                                       if matrix[matrix_name]["registered"]),
        "registered_controls_by_gate": registered_by_gate,
        "separated_controls_by_gate": separated_by_gate,
        "required_control_count": len(primary_gate),
        "required_report_count": len(primary_gate) * len(worlds),
        "complete_gate_count": sum(
            separated_by_gate[gate] == registered_by_gate[gate]
            for gate in GATE_COMPONENTS
        ),
        "full_separation": not deletion_candidates and not unexpected_controls,
        "unexpected_controls": unexpected_controls,
        "deletion_candidates": deletion_candidates,
        "matrix": matrix,
    }


def _empty_result(blockers: Sequence[str], *, expected_world_count: int,
                  graded_world_count: int,
                  gate_profile: str = DEFAULT_GATE_PROFILE) -> dict[str, Any]:
    target_product = (1.0 - TARGET_FALSE_FAIL_RATE) ** (
        len(GATE_COMPONENTS) * graded_world_count
    )
    try:
        selection = gate_profile_selection(gate_profile)
    except EvidenceError:
        selection = {}
    return {
        "schema": SCHEMA,
        "frozen": False,
        "gate_profile": gate_profile,
        "gate_profile_selection": {gate: list(components)
                                   for gate, components in selection.items()},
        "gates": {},
        "blockers": list(blockers),
        "target_false_fail_rate": TARGET_FALSE_FAIL_RATE,
        "quantile": QUANTILE,
        "qualification_world_count": expected_world_count,
        "graded_world_count": graded_world_count,
        "target_marginal_product": target_product,
        "achieved_marginal_rate_product": None,
        "achieved_marginal_rate_product_by_reference_line": {},
        "achieved_false_fail_rates": {},
        "achieved_false_fail_rates_by_reference_line": {},
        "mortality_identification_evidence": None,
        "caveats": [CORRELATION_CAVEAT, FINITE_WORLD_CAVEAT],
    }


def calibrate_composite_bars(
    reference_reports: Sequence[Mapping[str, Any]],
    replicate_reports: Sequence[Mapping[str, Any]] | None,
    control_reports: Sequence[Mapping[str, Any]],
    *,
    development_diagnostic_reports: Sequence[Mapping[str, Any]] = (),
    elder_reconstruction_audit: Mapping[str, Any] | None = None,
    mortality_identification_audit: Mapping[str, Any] | None = None,
    regime_identifiability_audit: Mapping[str, Any] | None = None,
    reserve_qualification_audit: Mapping[str, Any] | None = None,
    reserve_calibration_audit: Mapping[str, Any] | None = None,
    reserve_red_team_audit: Mapping[str, Any] | None = None,
    expected_qualification_worlds: int = EXPECTED_QUALIFICATION_WORLDS,
    graded_world_count: int = GRADED_WORLD_COUNT,
    control_registry: Mapping[str, Sequence[str]] = SCIENTIFIC_CONTROLS_BY_GATE,
    gate_profile: str = DEFAULT_GATE_PROFILE,
) -> dict[str, Any]:
    """Build a complete or explicitly blocked composite bar document.

    ``reference_reports`` are final line-by-world witnesses. ``replicate_reports`` are
    mandatory, uniquely identified observations used for the order statistics. The elder
    reconstruction audit binds the third participant-only line to its before-and-after
    level and tail measurements.

    ``gate_profile`` names which calibrated gates decide. Every profile calibrates all
    five on the same per-line replicate design, so the receipt a verifier reads is the
    same document under either profile; the profile only moves gates between deciding and
    reported. A reference exceedance on a reported gate is recorded rather than blocking,
    and a control that separates no gated gate is named as a deletion candidate for the
    profile.
    """

    try:
        profile_selection = gate_profile_selection(gate_profile)
    except EvidenceError as exc:
        return _empty_result(
            [str(exc)],
            expected_world_count=expected_qualification_worlds,
            graded_world_count=graded_world_count,
            gate_profile=gate_profile,
        )
    if isinstance(expected_qualification_worlds, bool) \
            or not isinstance(expected_qualification_worlds, int) \
            or isinstance(graded_world_count, bool) \
            or not isinstance(graded_world_count, int) \
            or expected_qualification_worlds <= 0 \
            or graded_world_count <= 0:
        return _empty_result(
            ["world counts must be positive"],
            expected_world_count=EXPECTED_QUALIFICATION_WORLDS,
            graded_world_count=GRADED_WORLD_COUNT,
            gate_profile=gate_profile,
        )
    if expected_qualification_worlds != EXPECTED_QUALIFICATION_WORLDS \
            or graded_world_count != GRADED_WORLD_COUNT:
        return _empty_result(
            [
                "V4 requires exactly six qualification worlds and three graded worlds"
            ],
            expected_world_count=expected_qualification_worlds,
            graded_world_count=graded_world_count,
            gate_profile=gate_profile,
        )
    try:
        _validated_control_registry(control_registry)
        if any(
            tuple(control_registry[gate]) != SCIENTIFIC_CONTROLS_BY_GATE[gate]
            for gate in GATE_COMPONENTS
        ):
            raise EvidenceError(
                "the scientific-control registry differs from the fixed V4 battery"
            )
        references = _normalize_entries(reference_reports, kind="reference")
        if replicate_reports is None:
            raise EvidenceError(
                "replicate evidence missing; final verifier reports cannot be bootstrapped "
                "or resampled as replacement evidence"
            )
        replicates = _normalize_entries(replicate_reports, kind="replicate")
        controls = _normalize_entries(control_reports, kind="control")
        diagnostics = _normalize_entries(
            development_diagnostic_reports, kind="diagnostic"
        )
        lines, worlds, replicates_per_pair = _check_reference_design(
            references, replicates, expected_qualification_worlds
        )
        if len(controls) != CONTROL_REPORT_COUNT:
            raise EvidenceError(
                f"qualification controls must contain exactly {CONTROL_REPORT_COUNT} reports"
            )
        _check_development_diagnostic_design(diagnostics)
        _check_binding_consistency(
            references, replicates, controls, diagnostics, lines, worlds
        )
        evidence_provenance = _freeze_provenance(references, replicates, controls)
        development_diagnostics = _development_diagnostic_block(diagnostics)
        measurement_contract_digest = references[0]["binding"][
            "measurement_contract_digest_sha256"
        ]
        regime_audit = _validate_regime_identifiability_audit(
            regime_identifiability_audit
        )
        mortality_audit = _validate_mortality_identification_audit(
            mortality_identification_audit, references, regime_audit, worlds
        )
        elder_audit = _validate_elder_audit(
            elder_reconstruction_audit, references, lines, worlds, mortality_audit
        )
        eligible_cells_by_world, eligibility_audit_by_world = _validated_eligible_cells(
            references, lines, worlds
        )
        gates = _calibrate_components(
            references, replicates, lines, worlds, eligible_cells_by_world,
            eligibility_audit_by_world,
        )
        reserve_red_team_measurement = _validate_reserve_red_team_measurement(
            reserve_red_team_audit, references, diagnostics
        )
        reserve_calibration_candidate = _validate_reserve_calibration_candidate(
            reserve_calibration_audit,
            references,
            controls,
            gates,
        )
    except EvidenceError as exc:
        return _empty_result(
            [str(exc)],
            expected_world_count=expected_qualification_worlds,
            graded_world_count=graded_world_count,
            gate_profile=gate_profile,
        )

    blockers: list[str] = []
    reference_failures: list[dict[str, Any]] = []
    ungated_reference_failures: list[dict[str, Any]] = []
    for entry in references:
        for gate, components in GATE_COMPONENTS.items():
            gated = profile_selection.get(gate, ())
            failures = [
                component for component in components
                if entry["metrics"][gate][component]
                > gates[gate]["components"][component]["value"]
            ]
            if not failures:
                continue
            record = {
                "reference_line": entry["reference_line"],
                "world": entry["world"],
                "gate": gate,
                "components": failures,
                "evidence_id": entry["evidence_id"],
            }
            # A reference exceedance on a gate the profile does not gate is recorded, not
            # blocking. That is the whole content of a reduced profile, so the receipt
            # has to carry the exceedance where a reader cannot miss it.
            if [component for component in failures if component in gated]:
                reference_failures.append(record)
            else:
                ungated_reference_failures.append(record)
    if reference_failures:
        blockers.append(
            f"{len(reference_failures)} final reference gate results exceed the p99 bars"
        )

    control_support = _attach_control_separation(
        gates, controls, control_registry, worlds, gate_profile
    )
    if control_support["unexpected_controls"]:
        blockers.append(
            "unregistered control reports were supplied: "
            + ", ".join(control_support["unexpected_controls"])
        )
    if control_support["deletion_candidates"]:
        blockers.append(
            "control separation is incomplete; deletion candidates: "
            + ", ".join(
                record["gate"] for record in control_support["deletion_candidates"]
            )
        )

    gate_rates_by_line = {
        line: {
            gate: gates[gate]["reference_line_calibration"][line][
                "false_fail_rate"
            ]
            for gate in GATE_COMPONENTS
        }
        for line in lines
    }
    gate_rates = {
        gate: max(gate_rates_by_line[line][gate] for line in lines)
        for gate in GATE_COMPONENTS
    }
    unattainable = [
        f"{line}/{gate}"
        for line in lines
        for gate, rate in gate_rates_by_line[line].items()
        if rate > TARGET_FALSE_FAIL_RATE + 1e-15
    ]
    if unattainable:
        blockers.append(
            "per-reference-line joint gate false-fail rate exceeds one percent for: "
            + ", ".join(unattainable)
        )
    target_product = (1.0 - TARGET_FALSE_FAIL_RATE) ** (
        len(GATE_COMPONENTS) * graded_world_count
    )
    achieved_products_by_line = {
        line: math.prod(
            (1.0 - gate_rates_by_line[line][gate]) ** graded_world_count
            for gate in GATE_COMPONENTS
        )
        for line in lines
    }
    achieved_product = min(achieved_products_by_line.values())
    reserve_audits: dict[str, Any] | None = None
    if not blockers:
        try:
            reserve_red_team = _bind_reserve_red_team_measurement(
                reserve_red_team_measurement,
                references,
                diagnostics,
                measurement_contract_digest,
            )
            reserve_calibration = _promote_reserve_calibration_candidate(
                reserve_calibration_candidate,
                references,
                controls,
                gates,
                measurement_contract_digest,
                reserve_red_team,
            )
            reserve_qualification = _build_reserve_qualification_audit(
                reserve_qualification_audit,
                references,
                controls,
                gates,
                measurement_contract_digest,
                reserve_calibration,
                reserve_red_team,
            )
            reserve_audits = {
                "qualification": reserve_qualification,
                "calibration": reserve_calibration,
                "red_team": reserve_red_team,
            }
        except EvidenceError as exc:
            blockers.append(str(exc))

    result = {
        "schema": SCHEMA,
        "frozen": not blockers,
        "gate_profile": gate_profile,
        "gate_profile_selection": {gate: list(components)
                                   for gate, components in profile_selection.items()},
        "reported_only_gates": [gate for gate in GATE_COMPONENTS
                                if gate not in profile_selection],
        "gates": gates,
        "blockers": blockers,
        "target_false_fail_rate": TARGET_FALSE_FAIL_RATE,
        "quantile": QUANTILE,
        "reference_lines": lines,
        "qualification_worlds": worlds,
        "qualification_world_count": len(worlds),
        "graded_world_count": graded_world_count,
        "replicates_per_reference_line_and_world": replicates_per_pair,
        "paired_resamples_per_world": replicates_per_pair,
        "paired_resample_count": replicates_per_pair * len(worlds),
        "equal_weighting": "equal replicate count for every reference-line and world pair",
        "reference_report_count": len(references),
        "replicate_report_count": len(replicates),
        "control_report_count": len(controls),
        "development_diagnostic_report_count": len(diagnostics),
        "run_receipt_count": len(references) + len(replicates) + len(controls)
        + len(diagnostics),
        "runner_digest_sha256": references[0]["binding"]["runner_digest_sha256"],
        "measurement_contract_digest_sha256": measurement_contract_digest,
        "evidence_provenance": evidence_provenance,
        "development_diagnostics": development_diagnostics,
        "elder_reconstruction_audit": elder_audit,
        "regime_identifiability_audit": regime_audit,
        "reference_failures": reference_failures,
        "ungated_reference_failures": ungated_reference_failures,
        "control_support": control_support,
        "achieved_false_fail_rates": gate_rates,
        "achieved_false_fail_rates_by_reference_line": gate_rates_by_line,
        "achieved_false_fail_rate_method": (
            "per-reference-line joint max-severity empirical p99"
        ),
        "target_marginal_product": target_product,
        "achieved_marginal_rate_product": achieved_product,
        "achieved_marginal_rate_product_by_reference_line": (
            achieved_products_by_line
        ),
        "mortality_identification_evidence": mortality_audit,
        "caveats": [CORRELATION_CAVEAT, FINITE_WORLD_CAVEAT],
    }
    if reserve_audits is not None and not blockers:
        result["reserve_audits"] = reserve_audits
    return result


freeze_composite_bars = calibrate_composite_bars


def _load_entries(paths: Sequence[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping) and "reports" in payload:
            payload = payload["reports"]
        if isinstance(payload, Mapping):
            entries.append(dict(payload))
        elif isinstance(payload, list) and all(isinstance(row, Mapping) for row in payload):
            entries.extend(dict(row) for row in payload)
        else:
            raise EvidenceError(f"{path}: expected an evidence object or list of objects")
    return entries


def _append_eligibility_audit(lines: list[str], eligible: Mapping[str, Any],
                              indent: str) -> None:
    audit = eligible.get("band_audit_by_world")
    if not isinstance(audit, Mapping):
        return
    lines.append(f"{indent}state-by-sex cell counts per band and world:")
    for world, world_record in audit.items():
        bands = world_record.get("bands", {}) if isinstance(world_record, Mapping) else {}
        for band in ELIGIBILITY_BANDS:
            record = bands.get(band, {}) if isinstance(bands, Mapping) else {}
            lines.append(
                f"{indent}  {world} {band}: {record.get('eligible_count', 'missing')} of "
                f"{record.get('cell_count', 'missing')} at floor "
                f"{record.get('floor_person_years', 'missing')}; "
                f"{record.get('status', 'missing')}; minimum exposure "
                f"{record.get('minimum_exposure_person_years', 'missing')}"
            )
            cells = record.get("cells")
            if isinstance(cells, list):
                for cell in cells:
                    lines.append(
                        f"{indent}    state {cell.get('state', 'missing')} "
                        f"{cell.get('sex', 'missing')}: "
                        f"{cell.get('exposure_person_years', 'missing')} person-years; "
                        f"eligible {cell.get('eligible', 'missing')}"
                    )


def _append_mortality_identification(lines: list[str], bars: Mapping[str, Any]) -> None:
    evidence = bars.get("mortality_identification_evidence")
    if not isinstance(evidence, Mapping):
        return
    summary = evidence.get("summary", {})
    lag_range = summary.get("publication_lag_trend_effect_percent_range")
    if not isinstance(lag_range, list) or len(lag_range) != 2:
        lag_range = ["missing", "missing"]
    source = evidence.get("measurement_source", {})
    lines.extend([
        "## Mortality identification evidence for the tail gate",
        "",
        "- mortality improvement is active throughout the public experience window: "
        + str(bool(summary.get(
            "trend_active_during_public_experience_window"
        ))).lower(),
        "- mortality improvement starts only after publication: "
        + str(bool(summary.get("trend_starts_only_after_publication"))).lower(),
        f"- the observed publication lags {summary.get('publication_lag_months')} "
        f"month(s) have trend effects from {lag_range[0]}% to {lag_range[1]}%",
        f"- the public shock process has annual probability "
        f"{summary.get('shock_annual_probability')}",
        "- every continuation redraws the public shock process independently: "
        + str(bool(summary.get(
            "continuation_shocks_redrawn_per_member"
        ))).lower(),
        f"- measurement source: {source.get('file')}::{source.get('function')} "
        f"(`{source.get('sha256')}`)",
        f"- audit digest: `{evidence.get('digest_sha256')}`",
    ])
    for row in evidence.get("worlds", []):
        if not isinstance(row, Mapping):
            continue
        record = row.get("decomposition", {})
        lines.append(
            f"- {row.get('world')}: horizon/history "
            f"{record.get('observed_horizon_to_history_ratio')}, trend-only "
            f"{record.get('trend_only_horizon_to_history_ratio')}, residual "
            f"{record.get('residual_observed_to_trend_ratio')}, lag factor "
            f"{record.get('publication_lag_trend_factor')}, designated horizon "
            f"mortality shock years "
            f"{record.get('designated_horizon_mortality_shock_years')}"
        )
    lines.append("")


def _append_elder_audit(lines: list[str], bars: Mapping[str, Any]) -> None:
    audit = bars.get("elder_reconstruction_audit")
    if not isinstance(audit, Mapping):
        return
    method = audit.get("method_digest", {})
    shock = audit.get("shock_redraw", {})
    lines.extend([
        "## Elder cohort-component qualification",
        "",
        f"- before line: {method.get('before_line')}",
        f"- after line: {method.get('after_line')}",
        f"- method source digest: `{method.get('source_sha256')}`",
        f"- method commit: `{method.get('git_commit')}`",
        f"- shock redraw: annual probability {shock.get('annual_probability')}, "
        f"independent per member {str(bool(shock.get('independent_per_member'))).lower()}",
        f"- shock magnitude source: {shock.get('magnitude_source')}",
    ])
    for record in audit.get("worlds", []):
        exposure = record.get("exposure_65_plus_absolute_error_percent", {})
        exceedance = record.get("pooled_exceedance_deviation", {})
        lines.append(
            f"- {record.get('world')}: absolute 65+ exposure error "
            f"{exposure.get('before')}% before, {exposure.get('after')}% after; "
            f"pooled exceedance deviation {exceedance.get('before')} before, "
            f"{exceedance.get('after')} after"
        )
        for region in record.get("liability_mean_by_region", []):
            lines.append(
                f"  - region {region.get('region')} liability mean: "
                f"{region.get('submitted_before')} before, "
                f"{region.get('submitted_after')} after, {region.get('sealed')} sealed"
            )
    lines.extend([
        f"- audit digest: `{audit.get('digest_sha256')}`",
        "",
    ])


def _profile_accounting_line(bars: Mapping[str, Any]) -> str:
    """Say how many of the calibrated gates the reported rates actually decide."""

    selection = bars.get("gate_profile_selection")
    deciding = len(selection) if isinstance(selection, Mapping) else len(GATE_COMPONENTS)
    return (
        f"- the rates below cover all {len(GATE_COMPONENTS)} calibrated gates; "
        f"{deciding} of them decide under the "
        f"{bars.get('gate_profile', DEFAULT_GATE_PROFILE)} profile"
    )


def _append_gate_profile(lines: list[str], bars: Mapping[str, Any]) -> None:
    """State which gates decide, which are reported only, and what that costs."""

    profile = bars.get("gate_profile", DEFAULT_GATE_PROFILE)
    selection = bars.get("gate_profile_selection")
    selection = selection if isinstance(selection, Mapping) else {}
    reported_only = [gate for gate in GATE_COMPONENTS if gate not in selection]
    lines.extend([
        "## Gate profile",
        "",
        f"- profile: {profile}",
        "- gates that decide: "
        + (", ".join(
            f"{gate} ({', '.join(selection.get(gate, []))})"
            for gate in GATE_COMPONENTS if gate in selection
        ) or "none"),
        "- measured and reported, deciding nothing: "
        + (", ".join(
            gate + (
                " (" + ", ".join(
                    component for component in GATE_COMPONENTS[gate]
                    if component not in selection.get(gate, [])
                ) + ")"
                if gate in selection else ""
            )
            for gate in GATE_COMPONENTS
            if gate in reported_only
            or set(selection.get(gate, [])) != set(GATE_COMPONENTS[gate])
        ) or "none"),
    ])
    ungated = bars.get("ungated_reference_failures")
    lines.append("- reference results above a reported bar:")
    if isinstance(ungated, list) and ungated:
        for record in ungated:
            if not isinstance(record, Mapping):
                continue
            lines.append(
                f"  - {record.get('reference_line')}/{record.get('world')}: "
                f"{record.get('gate')} "
                f"{', '.join(record.get('components', []))}; evidence "
                f"`{record.get('evidence_id')}`"
            )
    else:
        lines.append("  - none")
    support = bars.get("control_support")
    profile_support = support.get("gate_profile") \
        if isinstance(support, Mapping) else None
    if isinstance(profile_support, Mapping):
        candidates = profile_support.get("deletion_candidate_controls")
        lines.append(
            f"- {profile} profile deletion candidates, controls that fail no gate this "
            "profile decides on, on every qualification world: "
            + (", ".join(candidates) if isinstance(candidates, list) and candidates
               else "none")
        )
    lines.append("")


def _append_control_separation(lines: list[str], bars: Mapping[str, Any]) -> None:
    support = bars.get("control_support")
    if not isinstance(support, Mapping):
        return
    registry = support.get("registered_controls_by_gate")
    matrix = support.get("matrix")
    if not isinstance(registry, Mapping) or not isinstance(matrix, Mapping):
        return
    lines.extend([
        "## Control separation",
        "",
        "A gate is retained only when every control registered to that gate is a",
        "hard-valid submission and fails the gate on every qualification world.",
        "Every control is reported against every gate. The primary marker identifies",
        "the registered deletion test; the component values and frozen ceilings are",
        "the deletion-test numbers.",
        "",
    ])
    for gate in GATE_COMPONENTS:
        lines.append(f"- {gate}")
        registered_names = registry.get(gate, [])
        registered_set = set(registered_names) \
            if isinstance(registered_names, list) else set()
        for name in sorted(matrix):
            record = matrix.get(name, {})
            result = record.get("gates", {}).get(gate, {}) \
                if isinstance(record, Mapping) else {}
            lines.append(
                f"  - {name}{' [primary]' if name in registered_set else ''}: "
                "failed worlds "
                f"{', '.join(result.get('failed_worlds', [])) or 'none'}; "
                f"passed worlds {', '.join(result.get('passed_worlds', [])) or 'none'}; "
                "hard-invalid worlds "
                f"{', '.join(result.get('hard_invalid_worlds', [])) or 'none'}; "
                "missing worlds "
                f"{', '.join(record.get('missing_worlds', [])) or 'none'}"
            )
            per_world = result.get("per_world", {})
            if not isinstance(per_world, Mapping):
                continue
            for world in bars.get("qualification_worlds", []):
                comparison = per_world.get(world)
                if not isinstance(comparison, Mapping):
                    continue
                values = []
                components = comparison.get("components", {})
                if isinstance(components, Mapping):
                    for component in GATE_COMPONENTS[gate]:
                        item = components.get(component, {})
                        if isinstance(item, Mapping):
                            values.append(
                                f"{component}={item.get('value')} vs "
                                f"{item.get('ceiling')}"
                            )
                lines.append(
                    f"    - {world}: {comparison.get('outcome', 'missing')}; "
                    + "; ".join(values)
                )
                feasibility = comparison.get("reserve_q95_feasibility")
                if isinstance(feasibility, Mapping):
                    lines.append(
                        f"      q95 sum {feasibility.get('q95_sum')}; reserve total "
                        f"{feasibility.get('reserve_total')}; q95 diagnostic margin "
                        f"{feasibility.get('total_minus_q95_sum')}; feasible "
                        f"{str(bool(feasibility.get('feasible'))).lower()}"
                    )
    candidates = support.get("deletion_candidates", [])
    lines.extend(["", "Deletion candidates:"])
    if isinstance(candidates, list) and candidates:
        for candidate in candidates:
            controls = candidate.get("nonseparating_controls", []) \
                if isinstance(candidate, Mapping) else []
            names = [
                item.get("control", "missing") for item in controls
                if isinstance(item, Mapping)
            ]
            lines.append(
                f"- {candidate.get('gate', 'missing')}: "
                + ", ".join(names)
            )
    else:
        lines.append("- none")
    lines.append("")


def _hidden_regime_readings(audit: Mapping[str, Any]) -> str:
    """Each anchored axis under the threshold within the hidden block, with its value."""
    axes = audit.get("axes")
    named = audit.get("hidden_regime_correlation_shortfalls") or []
    parts = []
    for axis in named:
        record = axes.get(axis) if isinstance(axes, Mapping) else None
        within = record.get("within_regime_signed_rank_correlation") \
            if isinstance(record, Mapping) else None
        value = within.get("hidden") if isinstance(within, Mapping) else None
        parts.append(f"{axis} {value:+.3f}" if isinstance(value, (int, float))
                     else str(axis))
    return ", ".join(parts)


def _append_regime_identifiability(lines: list[str], bars: Mapping[str, Any]) -> None:
    audit = bars.get("regime_identifiability_audit")
    if not isinstance(audit, Mapping):
        return
    policy = audit.get("generator_policy", {})
    lines.extend([
        "## Regime-axis identifiability",
        "",
        f"- participant-anchor threshold: signed rank correlation greater than "
        f"{audit.get('anchor_correlation_threshold')} pooled over the eighteen worlds, "
        f"which is the reading that decides",
        f"- binding axis on the pooled rule: "
        f"{(audit.get('binding_axis') or {}).get('axis')} at "
        f"{(audit.get('binding_axis') or {}).get('signed_rank_correlation')}",
        "- anchored axes below the threshold within the six hidden worlds: "
        + (_hidden_regime_readings(audit) or "none"),
        "- those readings are reported and do not decide. A rank correlation over six "
        "worlds moves by more than the margin the threshold asks for when a single world "
        "changes rank, so an anchor is judged on the pooled eighteen-world reading and "
        "the hidden block is carried here for a reader to weigh.",
        "- axes eligible to leave the development band: "
        + ", ".join(policy.get("eligible_for_outside_development_band", [])),
        "- axes held inside the development band: "
        + ", ".join(policy.get("held_inside_development_band", [])),
        f"- measurement rows digest: `{audit.get('measurement_rows_digest_sha256')}`",
        f"- generator source digest: `{audit.get('generator_source_digest_sha256')}`",
    ])
    axes = audit.get("axes", {})
    if isinstance(axes, Mapping):
        for axis in REGIME_AXES:
            record = axes.get(axis, {})
            if not isinstance(record, Mapping):
                continue
            lines.append(
                f"- {axis}: signed rank correlation "
                f"{record.get('signed_rank_correlation')}; hidden regime "
                f"{(record.get('within_regime_signed_rank_correlation') or {}).get('hidden')}"
                f", qualified {record.get('hidden_regime_correlation_qualified')}"
                f"; disposition "
                f"{record.get('disposition')}; development range "
                f"{record.get('development_range')}; hidden generation range "
                f"{record.get('hidden_generation_range')}; raw axis observed "
                f"{record.get('axis_intensity_range_observed')}; realized mechanism "
                f"observed {record.get('realized_mechanism_range_observed')}"
            )
    lines.extend([
        f"- audit digest: `{audit.get('digest_sha256')}`",
        "",
    ])


def _append_authenticated_evidence(lines: list[str], bars: Mapping[str, Any]) -> None:
    diagnostics = bars.get("development_diagnostics")
    reserve = bars.get("reserve_audits")
    lines.extend([
        "## Authenticated evidence design",
        "",
        f"- measurement contract digest: `"
        f"{bars.get('measurement_contract_digest_sha256', 'missing')}`",
        f"- common runner digest: `{bars.get('runner_digest_sha256', 'missing')}`",
        f"- final reference reports: {bars.get('reference_report_count', 0)}",
        f"- paired replicate reports: {bars.get('replicate_report_count', 0)}; "
        f"{bars.get('paired_resamples_per_world', 0)} resamples per world",
        f"- qualification control reports: {bars.get('control_report_count', 0)}",
        f"- development diagnostic reports: "
        f"{bars.get('development_diagnostic_report_count', 0)}; these do not count as "
        "qualification controls",
        f"- unique run receipts: {bars.get('run_receipt_count', 0)}",
    ])
    if isinstance(diagnostics, Mapping):
        lines.append(
            f"- development diagnostic digest: `"
            f"{diagnostics.get('digest_sha256', 'missing')}`"
        )
    if isinstance(reserve, Mapping):
        for name in ("qualification", "calibration", "red_team"):
            audit = reserve.get(name)
            if isinstance(audit, Mapping):
                lines.append(
                    f"- reserve {name.replace('_', '-')} audit digest: `"
                    f"{audit.get('digest_sha256', 'missing')}`"
                )
    lines.append("")


def _append_reference_gate_results(
    lines: list[str],
    bars: Mapping[str, Any],
    gate: str,
    gate_record: Mapping[str, Any],
) -> None:
    components = gate_record.get("components")
    if not isinstance(components, Mapping):
        return
    by_component: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for component in GATE_COMPONENTS[gate]:
        record = components.get(component)
        witnesses = record.get("reference_witnesses") \
            if isinstance(record, Mapping) else None
        if not isinstance(witnesses, list):
            return
        by_component[component] = {
            (row.get("reference_line"), row.get("world")): row
            for row in witnesses
            if isinstance(row, Mapping)
        }
    rates = bars.get("achieved_false_fail_rates_by_reference_line", {})
    rendered_rates = ", ".join(
        f"{line} {float(rates.get(line, {}).get(gate)):.6%}"
        for line in bars.get("reference_lines", [])
        if isinstance(rates.get(line, {}).get(gate), (int, float))
        and not isinstance(rates.get(line, {}).get(gate), bool)
    )
    lines.append(
        "  reference results at per-line joint false-fail rates "
        f"{rendered_rates or 'missing'}:"
    )
    for line in bars.get("reference_lines", []):
        for world in bars.get("qualification_worlds", []):
            pair = (line, world)
            rows = [by_component[component].get(pair) for component in GATE_COMPONENTS[gate]]
            passed = bool(rows) and all(
                isinstance(row, Mapping) and row.get("pass") is True for row in rows
            )
            evidence_ids = sorted({
                str(row.get("evidence_id"))
                for row in rows
                if isinstance(row, Mapping)
            })
            evidence = evidence_ids[0] if len(evidence_ids) == 1 else "missing"
            lines.append(
                f"    - {line}/{world}: {'pass' if passed else 'fail'}; "
                f"evidence `{evidence}`"
            )


def render_freeze_report(bars: Mapping[str, Any]) -> str:
    lines = ["# Version-four composite bar freeze report", ""]
    lines.append("RESULT: " + ("FROZEN" if bars.get("frozen") else "NOT FROZEN"))
    lines.append(
        "PROFILE: " + str(bars.get("gate_profile", DEFAULT_GATE_PROFILE))
    )
    lines.append("")
    blockers = bars.get("blockers", [])
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
        lines.append("")
    _append_gate_profile(lines, bars)
    gates = bars.get("gates", {})
    if gates:
        lines.extend(["## Composite gates", ""])
        selection = bars.get("gate_profile_selection")
        selection = selection if isinstance(selection, Mapping) else {}
        for gate, components in GATE_COMPONENTS.items():
            record = gates[gate]
            gated = list(selection.get(gate, []))
            decision = "decides on " + ", ".join(gated) if gated \
                else "reported, decides nothing"
            lines.append(f"- {gate} ({decision})")
            lines.append(
                "  joint per-reference-line false-fail rates: "
                + ", ".join(
                    f"{line} "
                    f"{float(bars['achieved_false_fail_rates_by_reference_line'][line][gate]):.6%}"
                    for line in bars["reference_lines"]
                )
            )
            lines.append(
                f"  joint max-severity p99 ceiling: {record['severity_ceiling']:.12g}; "
                f"rank {record['order_statistic_rank_per_reference_line']} of "
                f"{record['sample_count_per_reference_line']} per line"
            )
            lines.append(
                "  supporting controls: "
                f"{', '.join(record.get('supporting_controls', [])) or 'none'}"
            )
            for component in components:
                bar = record["components"][component]
                lines.extend([
                    f"  - {component}",
                    f"    value: {bar['value']:.12g}",
                    "    attainable range: "
                    + json.dumps(bar["range"], separators=(",", ":")),
                    f"    fixed normalizer: {bar['normalizer']:.12g}",
                    f"    worlds: {', '.join(bar['worlds'])}",
                    f"    witnesses: {', '.join(bar['witnesses'])}",
                    "    diagnostic component p99 by line: "
                    + ", ".join(
                        f"{line} {float(value):.12g}"
                        for line, value in bar[
                            "empirical_p99_by_reference_line"
                        ].items()
                    ),
                ])
                eligible = bar.get("eligible_cells")
                if isinstance(eligible, Mapping):
                    lines.append("    eligible cells by qualification world:")
                    for world, cells in eligible.get("by_world", {}).items():
                        lines.append(
                            f"      {world}: "
                            + json.dumps(cells, separators=(",", ":"))
                        )
                    _append_eligibility_audit(lines, eligible, "    ")
            _append_reference_gate_results(lines, bars, gate, record)
        lines.append("")
    _append_control_separation(lines, bars)
    _append_authenticated_evidence(lines, bars)
    lines.extend(["## Empirical tail definition", "", *TAIL_DEFINITION_LINES, ""])
    _append_mortality_identification(lines, bars)
    _append_elder_audit(lines, bars)
    _append_regime_identifiability(lines, bars)
    lines.extend(["## False-fail accounting", ""])
    lines.append(_profile_accounting_line(bars))
    lines.append(f"- target per gate: {float(bars['target_false_fail_rate']):.2%}")
    lines.append(
        "- target marginal product over five gates and three graded worlds: "
        f"{float(bars['target_marginal_product']):.6f}"
    )
    achieved = bars.get("achieved_marginal_rate_product")
    lines.append(
        "- conservative achieved conditional marginal-rate product: "
        + ("unavailable" if achieved is None else f"{float(achieved):.6f}")
    )
    for line, value in bars.get(
        "achieved_marginal_rate_product_by_reference_line", {}
    ).items():
        lines.append(f"- {line} achieved conditional marginal-rate product: {value:.6f}")
    for caveat in bars.get("caveats", []):
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def render_provenance(bars: Mapping[str, Any]) -> str:
    lines = [
        "# Provenance of the version-four composite bars",
        "",
        "Each gate ceiling is the maximum of the three reference-line empirical p99",
        "order statistics of per-row max severity under fixed component normalizers.",
        "Each line contributes 102 independent deterministic replicate reports; lines",
        "are never pooled for the one-percent claim. Component records preserve their",
        "own empirical diagnostics but do not make marginal false-fail claims.",
        "Final witness reports are checked against the bars but are not resampled.",
        "Scientific controls must pass the deterministic hard checks before a failure",
        "can support a gate.",
        "",
        f"Schema: `{bars.get('schema')}`.",
        f"Frozen: `{json.dumps(bool(bars.get('frozen')))}`.",
        f"Gate profile: `{bars.get('gate_profile', DEFAULT_GATE_PROFILE)}`.",
        "",
    ]
    if bars.get("reference_lines"):
        lines.append("Reference lines: " + ", ".join(bars["reference_lines"]) + ".")
    if bars.get("qualification_worlds"):
        lines.append(
            "Qualification worlds: " + ", ".join(bars["qualification_worlds"]) + "."
        )
    provenance = bars.get("evidence_provenance")
    if isinstance(provenance, Mapping):
        lines.append(
            "Evidence provenance digest: `"
            + str(provenance.get("digest_sha256", "missing")) + "`."
        )
    gates = bars.get("gates", {})
    if gates:
        lines.extend(["", "## Per-bar provenance", ""])
        selection = bars.get("gate_profile_selection")
        selection = selection if isinstance(selection, Mapping) else {}
        for gate, components in GATE_COMPONENTS.items():
            record = gates[gate]
            gated = list(selection.get(gate, []))
            decision = "decides on " + ", ".join(gated) if gated \
                else "reported, decides nothing"
            lines.append(f"- {gate} ({decision})")
            lines.append(
                "  joint per-reference-line false-fail rates: "
                + ", ".join(
                    f"{line} "
                    f"{float(bars['achieved_false_fail_rates_by_reference_line'][line][gate]):.6%}"
                    for line in bars["reference_lines"]
                )
            )
            lines.append(
                f"  joint max-severity p99 ceiling: {record['severity_ceiling']:.12g}; "
                f"rank {record['order_statistic_rank_per_reference_line']} of "
                f"{record['sample_count_per_reference_line']} per line"
            )
            lines.append(
                "  supporting controls: "
                f"{', '.join(record.get('supporting_controls', [])) or 'none'}"
            )
            for component in components:
                bar = record["components"][component]
                lines.extend([
                    f"  - {component}",
                    f"    value: {bar['value']:.12g}",
                    "    attainable range: "
                    + json.dumps(bar["range"], separators=(",", ":")),
                    f"    fixed normalizer: {bar['normalizer']:.12g}",
                    f"    worlds: {', '.join(bar['worlds'])}",
                    f"    witnesses: {', '.join(bar['witnesses'])}",
                    "    diagnostic component p99 by line: "
                    + ", ".join(
                        f"{line} {float(value):.12g}"
                        for line, value in bar[
                            "empirical_p99_by_reference_line"
                        ].items()
                    ),
                    "    replicate evidence digest: `"
                    f"{bar['replicate_evidence_digest_sha256']}`",
                ])
                eligible = bar.get("eligible_cells")
                if isinstance(eligible, Mapping):
                    lines.append("    eligible cells by qualification world:")
                    for world, cells in eligible.get("by_world", {}).items():
                        lines.append(
                            f"      {world}: "
                            + json.dumps(cells, separators=(",", ":"))
                        )
                    _append_eligibility_audit(lines, eligible, "    ")
            _append_reference_gate_results(lines, bars, gate, record)
    lines.append("")
    _append_gate_profile(lines, bars)
    _append_control_separation(lines, bars)
    _append_authenticated_evidence(lines, bars)
    lines.extend(["", "## Empirical tail definition", "", *TAIL_DEFINITION_LINES, ""])
    _append_mortality_identification(lines, bars)
    _append_elder_audit(lines, bars)
    _append_regime_identifiability(lines, bars)
    lines.extend(["## False-fail accounting", ""])
    lines.append(_profile_accounting_line(bars))
    lines.append(
        "- target marginal product over five gates and three graded worlds: "
        f"{float(bars['target_marginal_product']):.6f}"
    )
    achieved = bars.get("achieved_marginal_rate_product")
    lines.append(
        "- conservative achieved conditional marginal-rate product: "
        + ("unavailable" if achieved is None else f"{float(achieved):.6f}")
    )
    for line, value in bars.get(
        "achieved_marginal_rate_product_by_reference_line", {}
    ).items():
        lines.append(f"- {line} achieved conditional marginal-rate product: {value:.6f}")
    lines.extend(["", CORRELATION_CAVEAT, "", FINITE_WORLD_CAVEAT, ""])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", action="append", default=[],
                        help="JSON manifest containing all three report lists")
    parser.add_argument("--reference-report", action="append", default=[])
    parser.add_argument("--replicate-report", action="append", default=[])
    parser.add_argument("--control-report", action="append", default=[])
    parser.add_argument("--development-diagnostic-report", action="append", default=[])
    parser.add_argument("--elder-audit",
                        help="machine-readable elder reconstruction qualification report")
    parser.add_argument(
        "--mortality-identification-audit",
        help="measured, packet-bound P4 mortality identification report",
    )
    parser.add_argument(
        "--regime-identifiability-audit",
        help="machine-readable participant-trace and hidden-axis policy report",
    )
    parser.add_argument(
        "--reserve-qualification-audit",
        help="optional digest-bound audit; it must equal the audit generated from reports",
    )
    parser.add_argument(
        "--reserve-calibration-audit",
        help="unaccepted candidate emitted by calibrate_reserve_rate.py",
    )
    parser.add_argument(
        "--reserve-red-team-audit",
        help="raw held-out reserve-total red-team measurement",
    )
    parser.add_argument(
        "--gate-profile", choices=sorted(GATE_PROFILES), default=DEFAULT_GATE_PROFILE,
        help="which calibrated composites decide; the rest are reported only",
    )
    parser.add_argument("--qualification-world-count", type=int,
                        default=EXPECTED_QUALIFICATION_WORLDS)
    parser.add_argument("--graded-world-count", type=int, default=GRADED_WORLD_COUNT)
    # Accepted only to turn an old invocation into a recorded fail-closed result.
    parser.add_argument("--dev", nargs="*", default=[])
    parser.add_argument("--qualification", nargs="*", default=[])
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--sweeps", type=int)
    parser.add_argument("--controls")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    references: list[dict[str, Any]] = []
    replicates: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    elder_audit: dict[str, Any] | None = None
    mortality_audit: dict[str, Any] | None = None
    regime_audit: dict[str, Any] | None = None
    reserve_qualification_audit: dict[str, Any] | None = None
    reserve_calibration_audit: dict[str, Any] | None = None
    reserve_red_team_audit: dict[str, Any] | None = None
    try:
        for path in args.evidence:
            payload = json.loads(Path(path).read_text())
            if not isinstance(payload, Mapping):
                raise EvidenceError(f"{path}: evidence manifest must be an object")
            if payload.get("schema") not in (None, EVIDENCE_SCHEMA):
                raise EvidenceError(f"{path}: unsupported evidence schema")
            embedded_audit = payload.get("elder_reconstruction_audit")
            if embedded_audit is not None:
                if elder_audit is not None or not isinstance(embedded_audit, Mapping):
                    raise EvidenceError(
                        "exactly one elder_reconstruction_audit object may be supplied"
                    )
                elder_audit = dict(embedded_audit)
            embedded_mortality_audit = payload.get("mortality_identification_audit")
            if embedded_mortality_audit is not None:
                if mortality_audit is not None \
                        or not isinstance(embedded_mortality_audit, Mapping):
                    raise EvidenceError(
                        "exactly one mortality_identification_audit object may be supplied"
                    )
                mortality_audit = dict(embedded_mortality_audit)
            embedded_regime_audit = payload.get("regime_identifiability_audit")
            if embedded_regime_audit is not None:
                if regime_audit is not None \
                        or not isinstance(embedded_regime_audit, Mapping):
                    raise EvidenceError(
                        "exactly one regime_identifiability_audit object may be supplied"
                    )
                regime_audit = dict(embedded_regime_audit)
            for key, current in (
                ("reserve_qualification_audit", reserve_qualification_audit),
                ("reserve_calibration_audit", reserve_calibration_audit),
                ("reserve_red_team_audit", reserve_red_team_audit),
            ):
                embedded = payload.get(key)
                if embedded is not None:
                    if current is not None or not isinstance(embedded, Mapping):
                        raise EvidenceError(
                            f"exactly one {key} object may be supplied"
                        )
                    if key == "reserve_qualification_audit":
                        reserve_qualification_audit = dict(embedded)
                    elif key == "reserve_calibration_audit":
                        reserve_calibration_audit = dict(embedded)
                    else:
                        reserve_red_team_audit = dict(embedded)
            for key, target in (
                ("reference_reports", references),
                ("replicate_reports", replicates),
                ("control_reports", controls),
                ("development_diagnostic_reports", diagnostics),
            ):
                rows = payload.get(key, [])
                if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
                    raise EvidenceError(f"{path}: {key} must be a list of objects")
                target.extend(dict(row) for row in rows)
        references.extend(_load_entries(args.reference_report))
        replicates.extend(_load_entries(args.replicate_report))
        controls.extend(_load_entries(args.control_report))
        diagnostics.extend(_load_entries(args.development_diagnostic_report))
        if args.elder_audit:
            if elder_audit is not None:
                raise EvidenceError("elder audit was supplied more than once")
            payload = json.loads(Path(args.elder_audit).read_text())
            if not isinstance(payload, Mapping):
                raise EvidenceError("elder audit must be a JSON object")
            elder_audit = dict(payload)
        if args.regime_identifiability_audit:
            if regime_audit is not None:
                raise EvidenceError("regime identifiability audit was supplied more than once")
            payload = json.loads(Path(args.regime_identifiability_audit).read_text())
            if not isinstance(payload, Mapping):
                raise EvidenceError("regime identifiability audit must be a JSON object")
            regime_audit = dict(payload)
        if args.mortality_identification_audit:
            if mortality_audit is not None:
                raise EvidenceError(
                    "mortality identification audit was supplied more than once"
                )
            payload = json.loads(Path(args.mortality_identification_audit).read_text())
            if not isinstance(payload, Mapping):
                raise EvidenceError("mortality identification audit must be a JSON object")
            mortality_audit = dict(payload)
        for argument, current, label in (
            (args.reserve_qualification_audit, reserve_qualification_audit,
             "reserve qualification audit"),
            (args.reserve_calibration_audit, reserve_calibration_audit,
             "reserve calibration audit"),
            (args.reserve_red_team_audit, reserve_red_team_audit,
             "reserve red-team audit"),
        ):
            if not argument:
                continue
            if current is not None:
                raise EvidenceError(f"{label} was supplied more than once")
            payload = json.loads(Path(argument).read_text())
            if not isinstance(payload, Mapping):
                raise EvidenceError(f"{label} must be a JSON object")
            if label == "reserve qualification audit":
                reserve_qualification_audit = dict(payload)
            elif label == "reserve calibration audit":
                reserve_calibration_audit = dict(payload)
            else:
                reserve_red_team_audit = dict(payload)
        bars = calibrate_composite_bars(
            references,
            replicates if replicates else None,
            controls,
            development_diagnostic_reports=diagnostics,
            elder_reconstruction_audit=elder_audit,
            mortality_identification_audit=mortality_audit,
            regime_identifiability_audit=regime_audit,
            reserve_qualification_audit=reserve_qualification_audit,
            reserve_calibration_audit=reserve_calibration_audit,
            reserve_red_team_audit=reserve_red_team_audit,
            expected_qualification_worlds=args.qualification_world_count,
            graded_world_count=args.graded_world_count,
            gate_profile=args.gate_profile,
        )
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        bars = _empty_result(
            [str(exc)],
            expected_world_count=args.qualification_world_count,
            graded_world_count=args.graded_world_count,
            gate_profile=args.gate_profile,
        )

    if (args.dev or args.qualification) and not replicates:
        note = (
            "replicate evidence missing; packet paths cannot substitute for deterministic "
            "replicate-level verifier reports"
        )
        if note not in bars["blockers"]:
            bars["blockers"].append(note)
        bars["frozen"] = False

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bars.json").write_text(json.dumps(bars, indent=2, sort_keys=True) + "\n")
    reserve_audits = bars.get("reserve_audits")
    standalone_audits = {
        "reserve_calibration_accepted.json": "calibration",
        "reserve_qualification_audit.json": "qualification",
    }
    for filename, key in standalone_audits.items():
        path = out / filename
        audit = reserve_audits.get(key) \
            if bars.get("frozen") is True and isinstance(reserve_audits, Mapping) \
            else None
        if isinstance(audit, Mapping):
            path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        elif path.exists():
            path.unlink()
    (out / "freeze_report.txt").write_text(render_freeze_report(bars))
    (out / "PROVENANCE.md").write_text(render_provenance(bars))
    print(render_freeze_report(bars), end="")
    return 0 if bars["frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
