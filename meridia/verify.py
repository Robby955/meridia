"""Verifier: score a submission directory against a packet's retained truth.

A version-four submission is three flat files in one directory:

- ``release.csv``: the revised release plus exposure and rate rows;
- ``projection.csv``: the horizon release;
- ``reserve.csv``: regional mean, q95, ES95, and allocation.

The verifier reads the submitted numbers and the retained truth only. It never inspects
the method. Bars are optional; with none it reports metrics and the hard checks (schema,
additivity, rate consistency, and reserve feasibility).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .actuarial import (ACTUARIAL_AGE_BAND_LABELS, BROAD_AGE_BAND_LABELS,
                        EXPOSURE_ESTIMAND, RATE_ESTIMANDS, RATE_EXTRA_COLUMNS,
                        RESERVE_COLUMNS, V4_SUBMISSION_COLUMNS,
                        ActuarialThresholds, ContinuationEnsemble, ObligationContract,
                        check_rate_additivity, eligibility_floor,
                        parse_rate_rows, parse_reserve_rows,
                        reserve_total, score_rates, score_reserve)
from .projection import score_allocation
from .release import AGE_BAND_LABELS, SEX_LABELS
from .scoring import (check_additivity, disclosure_audit, evaluate_gates, score_release,
                      validate_release)


def _read_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path)


def load_truth(path: Path) -> dict:
    frame = _read_csv(path)
    return {(str(r.estimand), str(r.level), int(r.unit)): float(r.value) for r in frame.itertuples()}


def load_rows(path: Path) -> list[dict]:
    frame = _read_csv(path)
    rows = []
    for r in frame.itertuples():
        rows.append({"estimand": str(r.estimand), "level": str(r.level), "unit": r.unit,
                     "estimate": float(r.estimate), "lower": float(r.lower),
                     "upper": float(r.upper)})
    return rows


def admin_from_packet(packet_dir: Path) -> dict:
    geography = _read_csv(Path(packet_dir) / "participant" / "geography.csv")
    county_state = geography["state"].to_numpy(dtype=np.int64)
    return {"n_counties": int(len(county_state)), "n_states": int(county_state.max()) + 1,
            "county_state": county_state}


def load_detailed(path: Path, n_counties: int) -> np.ndarray:
    frame = _read_csv(path)
    cube = np.full((n_counties, len(AGE_BAND_LABELS), len(SEX_LABELS)), np.nan)
    band_index = {label: i for i, label in enumerate(AGE_BAND_LABELS)}
    sex_index = {label: i for i, label in enumerate(SEX_LABELS)}
    for r in frame.itertuples():
        value = r.count
        cube[int(r.county), band_index[str(r.age_band)], sex_index[str(r.sex)]] = \
            np.nan if (value is None or (isinstance(value, float) and math.isnan(value))) else float(value)
    return cube


CORE_SUBMISSION_FILES = ("release.csv", "projection.csv", "allocation.csv")
ACTUARIAL_SUBMISSION_FILES = tuple(V4_SUBMISSION_COLUMNS)
ACTUARIAL_PACKET_SCHEMA = "meridia.packet.v4"
PARTICIPANT_CSV_FILES = (
    "experience_history.csv",
    "geography.csv",
    "survey_preliminary.csv",
    "survey_revised.csv",
    "sources/benchmark_preliminary.csv",
    "sources/benchmark_revised.csv",
    "sources/business_preliminary.csv",
    "sources/business_revised.csv",
    "sources/health_preliminary.csv",
    "sources/health_revised.csv",
    "sources/income_preliminary.csv",
    "sources/income_revised.csv",
    "sources/population_preliminary.csv",
    "sources/population_revised.csv",
)
SUBMISSION_FILES = ("release.csv", "detailed.csv", "projection.csv", "allocation.csv")
OPTIONAL_FILES = ("totals.csv",)
TOTAL_KINDS = {"county_age": ("county", "age_band"), "county_sex": ("county", "sex"),
               "county": ("county",), "age_sex": ("age_band", "sex")}

COMPOSITE_BAR_SCHEMA = "meridia.v4.composite-bars.v1"
VERIFIER_EVIDENCE_SCHEMA = "meridia.v4.verifier-evidence.v1"
RESERVE_TAIL_EVIDENCE_SCHEMA = "meridia.v4.reserve-tail-evidence.v1"
ELDER_REFERENCE_EVIDENCE_SCHEMA = "meridia.v4.elder-reference-evidence.v1"
SHOCK_REDRAW_REPORT_SCHEMA = "meridia.v4.continuation-shock-redraw-report.v1"
FREEZE_PROVENANCE_SCHEMA = "meridia.v4.freeze-provenance.v1"
FREEZE_EVIDENCE_BINDING_SCHEMA = "meridia.v4.freeze-evidence-binding.v3"
QUALIFICATION_WORLD_NAMES = tuple(f"qual-{index}" for index in range(6))
REFERENCE_LINES = ("A", "B", "C")
REPLICATES_PER_LINE_WORLD = 17
REFERENCE_REPORT_COUNT = len(REFERENCE_LINES) * len(QUALIFICATION_WORLD_NAMES)
REPLICATE_REPORT_COUNT = REFERENCE_REPORT_COUNT * REPLICATES_PER_LINE_WORLD
REGISTERED_MEASUREMENT_PARAMS = {
    "bootstrap_replicates": 100,
    "bayesian_sweeps": 400,
    "simulation_paths": 2048,
    "linkage_bootstraps": 12,
}
DEVELOPMENT_WORLD_NAMES = tuple(f"dev-{index:02d}" for index in range(12))
DEVELOPMENT_DIAGNOSTICS = (
    "design_reconstruction_oracle_tail",
    "true_population_normal_tail",
)
DEVELOPMENT_DIAGNOSTIC_SCHEMA = "meridia.v4.development-diagnostics.v1"
DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT = (
    len(DEVELOPMENT_WORLD_NAMES) * len(DEVELOPMENT_DIAGNOSTICS)
)
RESERVE_QUALIFICATION_SCHEMA = "meridia.v4.reserve-qualification-audit.v1"
RESERVE_CALIBRATION_SCHEMA = "meridia.reserve-rate-calibration.v2"
RESERVE_RED_TEAM_SCHEMA = "meridia.reserve-total-red-team.v1"
COMPOSITE_GATE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "exposures_and_rates": ("p95_relative_error",),
    "release_accuracy": ("p95_relative_error",),
    "interval_quality": ("coverage_deviation", "mean_interval_score"),
    "tail_calibration": ("pooled_exceedance_deviation",
                         "q95_width_relative_error", "es95_width_relative_error"),
    "reserve_skill": ("skill_loss", "worst_regional_shortfall_probability"),
}
GATE_COMPONENT_NORMALIZERS: dict[str, dict[str, float]] = {
    gate: {component: 1.0 for component in components}
    for gate, components in COMPOSITE_GATE_COMPONENTS.items()
}
# A gate profile selects which of the five frozen composites decide a verdict. It never
# adds a gate, never adds a component, and never moves a frozen ceiling: every profile is
# a subset of COMPOSITE_GATE_COMPONENTS. Whatever a profile leaves out is still measured
# and still reported; it only stops deciding. "full" is the default and decides on all
# five. "lite" drops the tail block: exceedance deviations, the q95 and ES95
# width-relative errors, and the regional shortfall probabilities are reported, and the
# population, exposure and rate, projection, and reserve-skill blocks decide.
DEFAULT_GATE_PROFILE = "full"
GATE_PROFILES: dict[str, dict[str, tuple[str, ...]]] = {
    "full": {gate: tuple(components)
             for gate, components in COMPOSITE_GATE_COMPONENTS.items()},
    "lite": {
        "exposures_and_rates": ("p95_relative_error",),
        "release_accuracy": ("p95_relative_error",),
        "interval_quality": ("coverage_deviation", "mean_interval_score"),
        "reserve_skill": ("skill_loss",),
    },
}


def gate_profile_selection(name: str) -> dict[str, tuple[str, ...]]:
    """Return the gates and components one profile decides on.

    An unknown name raises rather than falling back to a default selection.
    """
    if not isinstance(name, str) or name not in GATE_PROFILES:
        raise ValueError(f"unknown gate profile {name!r}")
    return {gate: tuple(components)
            for gate, components in GATE_PROFILES[name].items()}


SCIENTIFIC_CONTROLS_BY_GATE: dict[str, tuple[str, ...]] = {
    "exposures_and_rates": (
        "deterministic_linkage", "ignore_health_selection", "informative_selection",
    ),
    "release_accuracy": (
        "register_only", "survey_only", "no_dedup", "static_projection",
        "benchmark_only", "exact_key_union", "version_three_recipe",
        "experience_history_only",
    ),
    "interval_quality": ("inflated_intervals", "reconstruction_uncertainty"),
    "tail_calibration": (
        "development_average_regime", "mean_only_tail", "normal_tail",
        "padded_tail", "regime_recombination", "predictive_tails",
    ),
    "reserve_skill": (
        "uniform_allocation", "reserve_allocation", "proportional_reserve",
    ),
}
REQUIRED_SCIENTIFIC_CONTROLS = tuple(sorted({
    control
    for controls in SCIENTIFIC_CONTROLS_BY_GATE.values()
    for control in controls
}))
REGIME_AXES = (
    "mortality_improvement", "migration_age_pattern", "age_reporting_error",
    "linkage_urban_gradient", "administrative_completeness",
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
    "administrative_completeness", "missingness_target_dependence",
)
HIDDEN_EXTRAPOLATION_AXES = tuple(
    axis for axis in REGIME_AXES if axis not in HIDDEN_IN_BAND_AXES
)
DEVELOPMENT_AXIS_RANGES = {
    "mortality_improvement": [-0.010, 0.048],
    "migration_age_pattern": [0.25, 1.55],
    "age_reporting_error": [0.70, 2.05],
    "linkage_urban_gradient": [0.30, 1.55],
    "administrative_completeness": [0.30, 1.70],
    "missingness_target_dependence": [0.20, 1.30],
}
PUBLIC_AXIS_RANGES = {
    "mortality_improvement": [-0.030, 0.075],
    "migration_age_pattern": [0.00, 2.40],
    "age_reporting_error": [0.35, 3.40],
    "linkage_urban_gradient": [0.00, 2.60],
    "administrative_completeness": [0.00, 2.80],
    "missingness_target_dependence": [0.00, 2.20],
}
REALIZED_MECHANISM_ENVELOPES = {
    "mortality_improvement": {
        "development": [-0.010, 0.048],
        "public": [-0.030, 0.075],
    },
    "migration_age_pattern": {
        "development": [0.25, 1.55],
        "public": [0.00, 2.40],
    },
    "age_reporting_error": {
        "development": [0.596, 2.4248571428571424],
        "public": [0.298, 4.021714285714285],
    },
    "linkage_urban_gradient": {
        "development": [0.13125, 2.189375],
        "public": [0.0, 5.33],
    },
    "administrative_completeness": {
        "development": [0.30, 1.70],
        "public": [0.00, 2.80],
    },
    "missingness_target_dependence": {
        "development": [0.074, 2.119],
        "public": [0.0, 5.764],
    },
}
REALIZED_MECHANISM_DEFINITIONS = {
    axis: "axis_intensity" for axis in REGIME_AXES
}
REALIZED_MECHANISM_DEFINITIONS.update({
    "age_reporting_error": "age_reporting_error * age_error_mortality_scale",
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
COMPOSITE_COMPONENT_RANGES: dict[tuple[str, str], tuple[float, float | None]] = {
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
EXPERIENCE_HISTORY_COLUMNS = (
    "year", "age_band", "sex", "state", "exposure", "deaths",
    "qualifying_events", "net_migration",
)


def _digest_named_files(root: Path, names: tuple[str, ...]) -> str:
    """Hash file names and bytes in a stable order, failing on links or non-files."""

    digest = hashlib.sha256()
    for name in sorted(names):
        path = Path(root) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"cannot bind missing or non-regular evidence file {name}")
        encoded = name.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _regular_tree_names(root: Path, relative_root: str) -> tuple[str, ...]:
    """List every regular file under one packet subtree without following links."""

    subtree = Path(root) / relative_root
    if subtree.is_symlink() or not subtree.is_dir():
        raise ValueError(f"cannot bind missing or linked evidence directory {relative_root}")
    names: list[str] = []
    for path in sorted(subtree.rglob("*")):
        relative = path.relative_to(Path(root)).as_posix()
        if path.is_symlink():
            raise ValueError(f"cannot bind symbolic-link evidence entry {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"cannot bind non-regular evidence entry {relative}")
        names.append(relative)
    if not names:
        raise ValueError(f"cannot bind empty evidence directory {relative_root}")
    return tuple(names)


def _v4_evidence(packet_dir: Path, submission_dir: Path) -> dict[str, object]:
    """Bind a report to all participant inputs, exact scorer inputs, and scorer source."""

    packet_files = _regular_tree_names(packet_dir, "participant") + (
        "retained/continuation_liabilities.npz",
        "retained/continuation_shock_redraw.json",
        "retained/rate_truth_horizon.csv",
        "retained/truth_horizon.csv",
        "retained/truth_revised.csv",
    )
    source_root = Path(__file__).resolve().parent
    source_files = (
        "actuarial.py",
        "projection.py",
        "release.py",
        "scoring.py",
        "verify.py",
    )
    return {
        "schema": VERIFIER_EVIDENCE_SCHEMA,
        "packet_digest_sha256": _digest_named_files(packet_dir, packet_files),
        "contract_digest_sha256": hashlib.sha256(
            (Path(packet_dir) / "participant" / "contract.json").read_bytes()
        ).hexdigest(),
        "submission_digest_sha256": _digest_named_files(
            submission_dir, tuple(V4_SUBMISSION_COLUMNS)
        ),
        "packet_file_sha256": {
            name: hashlib.sha256((Path(packet_dir) / name).read_bytes()).hexdigest()
            for name in sorted(packet_files)
        },
        "submission_file_sha256": {
            name: hashlib.sha256((Path(submission_dir) / name).read_bytes()).hexdigest()
            for name in sorted(V4_SUBMISSION_COLUMNS)
        },
        "verifier_digest_sha256": _digest_named_files(source_root, source_files),
    }


def _public_reserve_rule_evidence(packet_dir: Path, contract: dict) -> tuple[dict, list[str]]:
    """Recompute the reserve total from the exact public experience file."""

    errors: list[str] = []
    evidence: dict = {"valid": False}
    reserve = contract.get("reserve")
    history = contract.get("experience_history")
    rule = reserve.get("total_rule") if isinstance(reserve, dict) else None
    if not isinstance(reserve, dict) or not isinstance(history, dict) \
            or not isinstance(rule, dict):
        return evidence, ["public reserve total rule is missing"]
    if history.get("file") != "experience_history.csv" \
            or tuple(history.get("columns", ())) != EXPERIENCE_HISTORY_COLUMNS:
        errors.append("experience-history contract differs from the public rule")
    required_rule = {
        "file": "experience_history.csv",
        "year": "maximum published year",
        "year_column": "year",
        "exposure_column": "exposure",
        "aggregation": "sum exposure over every row in the selected year",
        "rounding": "up",
    }
    if any(rule.get(key) != value for key, value in required_rule.items()):
        errors.append("public reserve total rule fields differ")
    path = Path(packet_dir) / "participant" / "experience_history.csv"
    if path.is_symlink() or not path.is_file():
        errors.append("public reserve experience file is missing or linked")
        return evidence, errors
    try:
        frame = _read_csv(path)
        if tuple(str(column) for column in frame.columns) != EXPERIENCE_HISTORY_COLUMNS:
            errors.append("public reserve experience columns differ")
        years = frame["year"].to_numpy(dtype=np.float64)
        exposures = frame["exposure"].to_numpy(dtype=np.float64)
        if not len(years) or not np.isfinite(years).all() \
                or not np.equal(years, np.floor(years)).all() \
                or not np.isfinite(exposures).all() or (exposures < 0.0).any():
            errors.append("public reserve experience values are invalid")
            return evidence, errors
        latest = int(years.max())
        exposure = float(exposures[years == latest].sum())
        rate = float(rule.get("rate_per_person_year"))
        unit = float(rule.get("rounding_unit"))
        total = float(reserve.get("total"))
        recomputed = reserve_total(exposure, rate, unit)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append(f"public reserve total cannot be recomputed ({type(exc).__name__})")
        return evidence, errors
    if rule.get("selected_year") != latest:
        errors.append("public reserve selected year differs from the experience file")
    recorded_exposure = rule.get("exposure_person_years")
    if isinstance(recorded_exposure, bool) \
            or not isinstance(recorded_exposure, (int, float)) \
            or not math.isclose(float(recorded_exposure), exposure,
                                rel_tol=1e-12, abs_tol=1e-9):
        errors.append("public reserve exposure differs from the experience file")
    if not math.isclose(total, recomputed, rel_tol=1e-12, abs_tol=1e-9):
        errors.append("public reserve total differs from its published rule")
    evidence = {
        "valid": not errors,
        "selected_year": latest,
        "exposure_person_years": exposure,
        "rate_per_person_year": rate,
        "rounding_unit": unit,
        "reserve_total": total,
        "experience_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return evidence, errors


def _reserve_q95_feasibility_evidence(
    parsed: dict | None,
    scored: dict | None,
    reserve_total_value: float,
    tolerance: float,
) -> dict[str, object]:
    """Record the submitted q95 floor as an authenticated diagnostic only."""

    if parsed is None or scored is None:
        return {"valid": False}
    q95 = np.asarray(parsed.get("q95"), dtype=np.float64)
    allocation = np.asarray(parsed.get("allocation"), dtype=np.float64)
    if q95.shape != allocation.shape or not q95.size \
            or not np.isfinite(q95).all() or not np.isfinite(allocation).all() \
            or not math.isfinite(float(reserve_total_value)):
        return {"valid": False}
    total = float(reserve_total_value)
    q95_sum = float(q95.sum())
    allocation_sum = float(allocation.sum())
    all_above = bool(
        np.all(allocation + tolerance * np.maximum(np.abs(q95), 1.0) >= q95)
    )
    sums_to_total = abs(allocation_sum - total) <= tolerance * max(1.0, abs(total))
    return {
        "q95_sum": q95_sum,
        "allocation_sum": allocation_sum,
        "reserve_total": total,
        "total_minus_q95_sum": total - q95_sum,
        "all_regions_at_or_above_q95": all_above,
        "allocation_sums_to_total": sums_to_total,
        "feasible": bool(scored.get("feasible") is True and all_above and sums_to_total),
    }


def _reserve_tail_evidence(
    parsed: dict | None, reserve_submission_sha256: object
) -> dict[str, object]:
    """Bind submitted q95 and ES95 totals to the exact reserve.csv bytes."""

    invalid = {"schema": RESERVE_TAIL_EVIDENCE_SCHEMA, "valid": False}
    if parsed is None or not isinstance(reserve_submission_sha256, str) \
            or len(reserve_submission_sha256) != 64 \
            or any(character not in "0123456789abcdef"
                   for character in reserve_submission_sha256):
        return invalid
    q95 = np.asarray(parsed.get("q95"), dtype=np.float64)
    es95 = np.asarray(parsed.get("es95"), dtype=np.float64)
    if q95.shape != es95.shape or not q95.size \
            or not np.isfinite(q95).all() or not np.isfinite(es95).all() \
            or (q95 < 0.0).any() or (es95 < q95).any():
        return invalid
    return {
        "schema": RESERVE_TAIL_EVIDENCE_SCHEMA,
        "valid": True,
        "q95_sum": float(q95.sum()),
        "es95_sum": float(es95.sum()),
        "reserve_submission_sha256": reserve_submission_sha256,
    }


def load_totals(path: Path, n_counties: int) -> dict[str, np.ndarray]:
    """Published totals of the detailed table: kind, county, age_band, sex, count.

    Every published total enters the disclosure audit as a linear constraint. A total
    published anywhere else is not a total the verifier missed: any file outside the
    declared set fails the submission outright.
    """
    frame = _read_csv(path)
    band_index = {label: i for i, label in enumerate(AGE_BAND_LABELS)}
    sex_index = {label: i for i, label in enumerate(SEX_LABELS)}
    shapes = {"county_age": (n_counties, len(AGE_BAND_LABELS)), "county_sex": (n_counties, len(SEX_LABELS)),
              "county": (n_counties,), "age_sex": (len(AGE_BAND_LABELS), len(SEX_LABELS))}
    totals = {kind: np.full(shape, np.nan) for kind, shape in shapes.items()}
    for r in frame.itertuples():
        kind = str(r.kind)
        if kind not in TOTAL_KINDS:
            raise ValueError(f"unknown total kind {kind!r}")
        idx = []
        for axis in TOTAL_KINDS[kind]:
            v = getattr(r, axis)
            idx.append(int(v) if axis == "county" else (band_index[str(v)] if axis == "age_band" else sex_index[str(v)]))
        value = r.count
        totals[kind][tuple(idx)] = np.nan if (value is None or (isinstance(value, float) and math.isnan(value))) else float(value)
    return {kind: t for kind, t in totals.items() if np.isfinite(t).any()}


def verify_submission(packet_dir: Path, submission_dir: Path, bars: dict | None = None,
                      alpha: float = 0.10, *, score_disclosure: bool = True,
                      allow_unfrozen: bool = False,
                      gate_profile: str = DEFAULT_GATE_PROFILE) -> dict:
    """Score one submission against one packet, dispatching on the packet's schema.

    ``allow_unfrozen`` is retained only for call compatibility and has no effect. A
    version-four verdict always requires a complete frozen receipt. Freeze measurements
    obtain ungated component metrics by omitting ``bars`` and reading ``hard_pass``.
    ``gate_profile`` selects which composites decide a version-four verdict; the version-
    three surface below has no composite gates and ignores it.
    """
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    contract = json.loads((packet_dir / "participant" / "contract.json").read_text())
    if str(contract.get("schema", "")).startswith(ACTUARIAL_PACKET_SCHEMA):
        del allow_unfrozen
        if bars is not None and bars.get("frozen") is not True:
            return _failed_v4_report(
                "bars: this bar set does not record a completed freeze",
                gate_profile=gate_profile)
        return verify_actuarial_submission(packet_dir, submission_dir, bars, alpha,
                                           gate_profile=gate_profile)
    admin = admin_from_packet(packet_dir)
    # Fail closed on the selected output contract. The reusable Meridia verifier keeps
    # the detailed-table audit; the benchmark surface scores the other three files.
    required_files = SUBMISSION_FILES if score_disclosure else CORE_SUBMISSION_FILES
    optional_files = OPTIONAL_FILES if score_disclosure else ()
    present = sorted(p.name for p in submission_dir.iterdir() if p.is_file())
    unexpected = [n for n in present if n not in required_files + optional_files]
    missing = [n for n in required_files if n not in present]
    if unexpected or missing:
        report = {"pass": False,
                  "reasons": [f"file set: unexpected {unexpected}, missing {missing}"],
                  "schema_errors": [], "additivity_errors": [], "metrics": {},
                  "projection_metrics": {}, "allocation": {"feasible": False}}
        if score_disclosure:
            report["disclosure"] = {"pass": False, "n_protected": 0,
                                    "n_suppressed": 0, "published_protected": [],
                                    "recoverable": []}
        return report
    retained = packet_dir / "retained"
    truth_now = load_truth(retained / "truth_revised.csv")
    truth_future = load_truth(retained / "truth_horizon.csv")

    release = load_rows(submission_dir / "release.csv")
    schema_errors = validate_release(release, admin)
    additivity_errors = check_additivity(release, admin) if not schema_errors else []
    metrics = score_release(release, truth_now, admin, alpha)

    disclosure = None
    if score_disclosure:
        detailed_truth = load_detailed(retained / "detailed_revised.csv",
                                       admin["n_counties"]).astype(np.int64)
        published = load_detailed(submission_dir / "detailed.csv", admin["n_counties"])
        marginals = load_totals(submission_dir / "totals.csv", admin["n_counties"]) \
            if (submission_dir / "totals.csv").exists() else None
        disclosure = disclosure_audit(published, detailed_truth,
                                      int(contract["disclosure_threshold"]), marginals)

    projection = load_rows(submission_dir / "projection.csv")
    projection_schema = validate_release(projection, admin)
    projection_metrics = score_release(projection, truth_future, admin, alpha)

    allocation_frame = _read_csv(submission_dir / "allocation.csv").sort_values("county")
    allocation = np.zeros(admin["n_counties"])
    allocation[allocation_frame["county"].to_numpy(dtype=np.int64)] = \
        allocation_frame["allocation"].to_numpy(dtype=np.float64)
    demand = np.asarray([truth_future[(contract["allocation"]["demand"], "county", c)]
                         for c in range(admin["n_counties"])])
    allocation_score = score_allocation(allocation, demand,
                                         float(contract["allocation"]["budget"]))

    gates = evaluate_gates(schema_errors, additivity_errors, metrics, disclosure, bars)
    projection_gates = evaluate_gates(projection_schema, [], projection_metrics, None,
                                      (bars or {}).get("projection"))
    reasons = list(gates["reasons"]) + [f"projection {r}" for r in projection_gates["reasons"]]
    if not allocation_score["feasible"]:
        reasons.append("allocation: infeasible (negative, non-finite, or over budget)")
    ceiling = (bars or {}).get("allocation_regret_ceiling")
    if ceiling is not None and allocation_score["feasible"] and allocation_score["regret"] > ceiling:
        reasons.append(f"allocation: regret {allocation_score['regret']:.4f} > {ceiling}")
    report = {
        "pass": not reasons, "reasons": reasons,
        "schema_errors": schema_errors, "additivity_errors": additivity_errors,
        "metrics": metrics, "projection_metrics": projection_metrics,
        "allocation": allocation_score,
    }
    if disclosure is not None:
        report["disclosure"] = {k: v for k, v in disclosure.items()
                                if k in ("pass", "n_protected", "n_suppressed",
                                         "published_protected", "recoverable",
                                         "utility", "n_scored", "detailed_error",
                                         "detailed_error_p95", "detailed_worst_error")}
    return report




# --------------------------------------------------------------- version-four surface

def load_release_blocks(path: Path) -> tuple[list[dict], list[dict]]:
    """Split a version-four release or projection table into its two blocks.

    Rows whose ``age_band`` is filled belong to the exposure and rate block; the rest are
    the eight version-three estimands, which leave both extra columns blank.
    """
    frame = _read_csv(path)
    for column in RATE_EXTRA_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    core: list[dict] = []
    rates: list[dict] = []
    for r in frame.itertuples():
        band = "" if (r.age_band is None or (isinstance(r.age_band, float)
                                             and math.isnan(r.age_band))) else str(r.age_band)
        sex = "" if (r.sex is None or (isinstance(r.sex, float)
                                       and math.isnan(r.sex))) else str(r.sex)
        row = {"estimand": str(r.estimand), "level": str(r.level), "unit": r.unit,
               "estimate": float(r.estimate), "lower": float(r.lower),
               "upper": float(r.upper)}
        if band or str(r.estimand) in RATE_ESTIMANDS:
            rates.append(row | {"sex": sex, "age_band": band})
        else:
            core.append(row | {"sex": sex, "age_band": band})
    return core, rates


def load_rate_truth(path: Path) -> dict:
    """Retained exposure and rate truth: estimand, level, unit, sex, age_band, value."""
    frame = _read_csv(path)
    return {(str(r.estimand), str(r.level), int(r.unit), str(r.sex), str(r.age_band)):
            float(r.value) for r in frame.itertuples()}


def build_eligibility_evidence(rate_truth: dict,
                               thresholds: ActuarialThresholds) -> dict:
    """Record every state-by-sex exposure behind each eligibility decision."""

    rows: dict[str, dict] = {}
    for band in (*ACTUARIAL_AGE_BAND_LABELS, *BROAD_AGE_BAND_LABELS):
        floor = eligibility_floor(thresholds, band)
        cells = [
            {
                "state": int(key[2]),
                "sex": str(key[3]),
                "exposure_person_years": float(value),
                "eligible": bool(float(value) >= floor),
            }
            for key, value in rate_truth.items()
            if key[0] == EXPOSURE_ESTIMAND and key[1] == "state" and key[4] == band
            and math.isfinite(float(value))
        ]
        cells.sort(key=lambda cell: (cell["state"], cell["sex"]))
        values = [float(cell["exposure_person_years"]) for cell in cells]
        rows[band] = {
            "status": "scored" if band in BROAD_AGE_BAND_LABELS else "report-only",
            "floor_person_years": floor,
            "cell_count": len(values),
            "eligible_count": sum(value >= floor for value in values),
            "minimum_exposure_person_years": min(values) if values else None,
            "cells": cells,
        }
    return {
        "truth_quantity": "retained state-by-sex person-years exposure",
        "bands": rows,
    }


def load_continuation_ensemble(path: Path) -> ContinuationEnsemble:
    """Retained regional liabilities on every committed continuation.

    New V4 archives carry only ``liability`` of shape (members, regions): every row is an
    independently simulated predictive member, while horizon point truth is stored
    separately.  A legacy ``realized_member`` scalar is accepted for old development
    fixtures, but scoring never reads a designated row. Nothing about the ensemble reaches
    the participant side.
    """
    with np.load(Path(path)) as archive:
        realized = int(archive["realized_member"]) if "realized_member" in archive else 0
        return ContinuationEnsemble(np.asarray(archive["liability"], dtype=np.float64),
                                    realized)


def _continuation_shock_redraw_report(
    packet_dir: Path, contract: dict, evidence: dict
) -> dict:
    """Validate measured member schedules and bind them to the scored liability archive."""

    from .packet import _validate_shock_redraw_evidence

    path = Path(packet_dir) / "retained" / "continuation_shock_redraw.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("continuation shock redraw evidence is missing or linked")
    try:
        raw_payload = path.read_bytes()
        payload = json.loads(raw_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("continuation shock redraw evidence is invalid JSON") from exc
    canonical_payload = (
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if raw_payload != canonical_payload:
        raise ValueError("continuation shock redraw evidence is not canonical JSON")
    expected_members = contract.get("reserve", {}).get("members")
    if isinstance(expected_members, bool) or not isinstance(expected_members, int):
        raise ValueError("public contract has no continuation member count")
    payload = _validate_shock_redraw_evidence(
        payload, expected_members=expected_members
    )
    if expected_members >= 100 and (
        payload["distinct_future_schedule_count"] <= 1
        or not 0 < payload["future_shock_year_count"]
        < payload["future_year_opportunity_count"]
        or payload["future_mortality_spike_year_count"] <= 0
    ):
        raise ValueError("continuation shock redraw measurement shows no member variation")
    packet_files = evidence.get("packet_file_sha256")
    if not isinstance(packet_files, dict):
        raise ValueError("verifier packet file evidence is missing")
    runtime_digest = packet_files.get("retained/continuation_shock_redraw.json")
    liability_digest = packet_files.get("retained/continuation_liabilities.npz")
    if not all(isinstance(value, str) and len(value) == 64 for value in (
        runtime_digest, liability_digest
    )) or runtime_digest != hashlib.sha256(raw_payload).hexdigest():
        raise ValueError("continuation shock redraw byte bindings are missing")
    return {
        "schema": SHOCK_REDRAW_REPORT_SCHEMA,
        "runtime_evidence_file_sha256": runtime_digest,
        "liability_archive_sha256": liability_digest,
        "runtime_evidence": payload,
    }


def load_reserve_rows(path: Path) -> list[dict]:
    frame = _read_csv(path)
    missing = [c for c in RESERVE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"reserve.csv is missing {missing}")
    return [{"region": r.region, "liability_mean": float(r.liability_mean),
             "q95": float(r.q95), "es95": float(r.es95),
             "allocation": float(r.allocation)} for r in frame.itertuples()]


def _elder_reference_evidence(
    parsed_rates: dict,
    rate_truth: dict,
    parsed_reserve: dict,
    ensemble: ContinuationEnsemble,
    n_states: int,
    evidence: dict,
) -> dict:
    """Recompute elder quantities from the exact scored packet and submission."""

    elder_bands = ACTUARIAL_AGE_BAND_LABELS[-3:]
    states = []
    for state in range(n_states):
        submitted = 0.0
        sealed = 0.0
        for sex in SEX_LABELS:
            for band in elder_bands:
                key = (EXPOSURE_ESTIMAND, "state", state, sex, band)
                submitted_row = parsed_rates.get(key)
                sealed_value = rate_truth.get(key)
                if submitted_row is None or sealed_value is None:
                    raise ValueError("elder reference evidence lacks a state exposure cell")
                submitted += float(submitted_row[0])
                sealed += float(sealed_value)
        states.append({
            "state": state,
            "submitted_person_years": submitted,
            "sealed_person_years": sealed,
        })

    submitted_means = np.asarray(parsed_reserve["liability_mean"], dtype=np.float64)
    sealed_means = np.asarray(ensemble.liability, dtype=np.float64).mean(axis=0)
    if submitted_means.shape != sealed_means.shape \
            or not np.isfinite(submitted_means).all() \
            or not np.isfinite(sealed_means).all():
        raise ValueError("elder reference evidence has invalid regional liability means")
    packet_digest = evidence.get("packet_digest_sha256")
    submission_digest = evidence.get("submission_digest_sha256")
    if not isinstance(packet_digest, str) or not isinstance(submission_digest, str):
        raise ValueError("elder reference evidence lacks verifier byte bindings")
    return {
        "schema": ELDER_REFERENCE_EVIDENCE_SCHEMA,
        "valid": True,
        "packet_digest_sha256": packet_digest,
        "submission_digest_sha256": submission_digest,
        "state_65_plus_person_years": states,
        "liability_mean_by_region": [
            {
                "region": region,
                "submitted": float(submitted_means[region]),
                "sealed": float(sealed_means[region]),
            }
            for region in range(len(sealed_means))
        ],
    }


def _v4_file_errors(submission_dir: Path) -> list[str]:
    """Require the exact three regular files, rejecting directories and symlinks."""
    if not submission_dir.is_dir():
        return ["submission path is not a directory"]
    entries = {entry.name: entry for entry in submission_dir.iterdir()}
    expected = set(ACTUARIAL_SUBMISSION_FILES)
    errors: list[str] = []
    unexpected = sorted(set(entries) - expected)
    missing = sorted(expected - set(entries))
    if unexpected or missing:
        errors.append(f"unexpected {unexpected}, missing {missing}")
    for name in sorted(expected & set(entries)):
        entry = entries[name]
        if entry.is_symlink() or not entry.is_file():
            errors.append(f"{name} is not a regular non-symlink file")
    return errors


def _v4_header_errors(submission_dir: Path) -> list[str]:
    errors: list[str] = []
    for name, expected in V4_SUBMISSION_COLUMNS.items():
        try:
            columns = tuple(str(column) for column in _read_csv(
                submission_dir / name).columns)
        except Exception as exc:
            errors.append(f"{name}: cannot read CSV ({type(exc).__name__})")
            continue
        if columns != expected:
            errors.append(f"{name}: columns {list(columns)} differ from {list(expected)}")
    return errors


def _contract_submission_errors(
    packet_dir: Path,
    contract: dict,
    thresholds: ActuarialThresholds,
) -> list[str]:
    block = contract.get("submission")
    if not isinstance(block, dict):
        return ["contract has no submission schema"]
    files = block.get("files")
    expected = {name: list(columns) for name, columns in V4_SUBMISSION_COLUMNS.items()}
    errors = []
    if files != expected:
        errors.append("contract submission files or ordered columns differ from the verifier")
    if block.get("additional_entries") != "forbidden":
        errors.append("contract does not forbid additional submission entries")
    reserve = contract.get("reserve")
    allocation_rule = reserve.get("allocation_rule") \
        if isinstance(reserve, dict) else None
    expected_allocation_rule = {
        "finite": True,
        "minimum": 0.0,
        "sum": "reserve.total",
        "tolerance": thresholds.feasibility_tolerance,
    }
    if allocation_rule != expected_allocation_rule:
        errors.append("contract reserve allocation rule differs from the verifier")
    benchmark = contract.get("benchmark")
    if not isinstance(benchmark, dict) \
            or benchmark.get("file") != "sources/benchmark_revised.csv":
        errors.append("contract benchmark path differs from the participant packet")

    schemas = contract.get("participant_csv_schemas")
    if not isinstance(schemas, dict) or set(schemas) != set(PARTICIPANT_CSV_FILES):
        errors.append("contract participant CSV schema inventory differs")
        return errors
    participant = Path(packet_dir) / "participant"
    for relative in PARTICIPANT_CSV_FILES:
        declared = schemas.get(relative)
        path = participant / relative
        if not isinstance(declared, list) or not declared \
                or any(not isinstance(column, str) or not column for column in declared):
            errors.append(f"contract participant CSV columns are invalid for {relative}")
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"participant CSV is missing or linked: {relative}")
            continue
        try:
            with path.open(newline="") as handle:
                observed = next(csv.reader(handle))
        except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
            errors.append(
                f"participant CSV header cannot be read for {relative} "
                f"({type(exc).__name__})"
            )
            continue
        if observed != declared:
            errors.append(f"participant CSV header differs from contract for {relative}")
    return errors


def _release_accuracy(metrics: dict, projection_metrics: dict) -> float:
    values = [float(item["p95_error"])
              for block in (metrics, projection_metrics)
              for key, item in block.items() if key.endswith("/all")]
    return float(max(values)) if values and all(math.isfinite(v) for v in values) \
        else float("nan")


def _interval_quality(metrics: dict, projection_metrics: dict,
                      rate_metrics: dict, alpha: float) -> tuple[float, float]:
    groups: list[tuple[int, float, float]] = []
    for block in (metrics, projection_metrics):
        groups.extend((int(item["n_units"]), float(item["coverage"]),
                       float(item["mean_interval_score"]))
                      for key, item in block.items() if key.endswith("/all"))
    rates = rate_metrics.get("composite", {})
    if int(rates.get("n_cells", 0)):
        groups.append((int(rates["n_cells"]), float(rates["coverage"]),
                       float(rates["mean_interval_score"])))
    if not groups or any(n <= 0 or not math.isfinite(coverage)
                         or not math.isfinite(score)
                         for n, coverage, score in groups):
        return float("nan"), float("nan")
    total = sum(n for n, _, _ in groups)
    # Opposite subgroup errors must not cancel into nominal pooled coverage. Keeping the
    # worst deviation as one component preserves one pass event while making every
    # estimand-wide interval block accountable.
    coverage_deviation = max(abs(value - (1.0 - float(alpha)))
                             for _, value, _ in groups)
    score = sum(n * value for n, _, value in groups) / total
    return float(coverage_deviation), float(score)


def build_composite_metrics(metrics: dict, projection_metrics: dict, rate_metrics: dict,
                            reserve: dict | None, alpha: float) -> dict:
    """Five named stochastic measurements; hard validity checks live outside them."""
    rate = rate_metrics.get("composite", {})
    coverage_deviation, interval = _interval_quality(
        metrics, projection_metrics, rate_metrics, alpha)
    reserve = reserve or {}
    calibration = reserve.get("calibration", {})
    skill = float(reserve.get("skill", float("nan")))
    shortfall = np.asarray(
        reserve.get("regional_shortfall_probability", []), dtype=np.float64
    )
    worst_shortfall = float(np.max(shortfall)) \
        if shortfall.size and np.isfinite(shortfall).all() else float("nan")
    return {
        "exposures_and_rates": {
            "p95_relative_error": float(rate.get("p95_relative_error", float("nan"))),
        },
        "release_accuracy": {"p95_relative_error": _release_accuracy(
            metrics, projection_metrics)},
        "interval_quality": {"coverage_deviation": coverage_deviation,
                             "mean_interval_score": interval},
        "tail_calibration": {
            "pooled_exceedance_deviation": float(calibration.get("pooled", float("nan"))),
            "q95_width_relative_error": float(
                reserve.get("mean_q95_width_error", float("nan"))),
            "es95_width_relative_error": float(
                reserve.get("mean_es95_width_error", float("nan"))),
        },
        "reserve_skill": {
            "skill_loss": max(0.0, 1.0 - skill) if math.isfinite(skill)
            else float("nan"),
            "worst_regional_shortfall_probability": worst_shortfall,
        },
    }


def _elder_audit_values_match(
    audit_row: object, before_binding: object, after_binding: object
) -> bool:
    """Check audit values against the verifier detail carried by A and C bindings."""

    if not isinstance(audit_row, dict) or not isinstance(before_binding, dict) \
            or not isinstance(after_binding, dict):
        return False
    before = before_binding.get("elder_reference_evidence")
    after = after_binding.get("elder_reference_evidence")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    audit_states = audit_row.get("state_65_plus_person_years")
    before_states = before.get("state_65_plus_person_years")
    after_states = after.get("state_65_plus_person_years")
    if not all(isinstance(rows, list) and len(rows) == 6 for rows in (
        audit_states, before_states, after_states
    )):
        return False
    try:
        audit_by_state = {item["state"]: item for item in audit_states}
        before_by_state = {item["state"]: item for item in before_states}
        after_by_state = {item["state"]: item for item in after_states}
    except (KeyError, TypeError):
        return False
    if any(set(rows) != set(range(6)) for rows in (
        audit_by_state, before_by_state, after_by_state
    )):
        return False
    denominator = before_numerator = after_numerator = 0.0
    for state in range(6):
        observed = audit_by_state[state]
        expected_before = before_by_state[state]
        expected_after = after_by_state[state]
        if not isinstance(observed, dict) or set(observed) != {
            "state", "submitted_before", "submitted_after", "sealed"
        }:
            return False
        values = (
            observed.get("submitted_before"), observed.get("submitted_after"),
            observed.get("sealed"), expected_before.get("submitted_person_years"),
            expected_after.get("submitted_person_years"),
            expected_before.get("sealed_person_years"),
            expected_after.get("sealed_person_years"),
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) < 0.0
               for value in values):
            return False
        submitted_before, submitted_after, sealed = map(float, values[:3])
        if not all(math.isclose(actual, float(expected), rel_tol=1e-12, abs_tol=1e-9)
                   for actual, expected in (
                       (submitted_before, values[3]),
                       (submitted_after, values[4]),
                       (sealed, values[5]),
                       (sealed, values[6]),
                   )):
            return False
        denominator += sealed
        before_numerator += abs(submitted_before - sealed)
        after_numerator += abs(submitted_after - sealed)
    exposure = audit_row.get("exposure_65_plus_absolute_error_percent")
    if denominator <= 0.0 or not isinstance(exposure, dict):
        return False
    expected_errors = (
        100.0 * before_numerator / denominator,
        100.0 * after_numerator / denominator,
    )
    if any(
        isinstance(exposure.get(name), bool)
        or not isinstance(exposure.get(name), (int, float))
        or not math.isclose(
            float(exposure[name]), expected, rel_tol=1e-12, abs_tol=1e-9
        )
        for name, expected in zip(("before", "after"), expected_errors, strict=True)
    ):
        return False

    audit_regions = audit_row.get("liability_mean_by_region")
    before_regions = before.get("liability_mean_by_region")
    after_regions = after.get("liability_mean_by_region")
    if not all(isinstance(rows, list) and len(rows) == 6 for rows in (
        audit_regions, before_regions, after_regions
    )):
        return False
    try:
        audit_by_region = {item["region"]: item for item in audit_regions}
        before_by_region = {item["region"]: item for item in before_regions}
        after_by_region = {item["region"]: item for item in after_regions}
    except (KeyError, TypeError):
        return False
    if any(set(rows) != set(range(6)) for rows in (
        audit_by_region, before_by_region, after_by_region
    )):
        return False
    for region in range(6):
        observed = audit_by_region[region]
        expected_before = before_by_region[region]
        expected_after = after_by_region[region]
        if not isinstance(observed, dict) or set(observed) != {
            "region", "submitted_before", "submitted_after", "sealed"
        }:
            return False
        values = (
            observed.get("submitted_before"), observed.get("submitted_after"),
            observed.get("sealed"), expected_before.get("submitted"),
            expected_after.get("submitted"), expected_before.get("sealed"),
            expected_after.get("sealed"),
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) or float(value) < 0.0
               for value in values):
            return False
        if not all(math.isclose(float(actual), float(expected), rel_tol=1e-12,
                                abs_tol=1e-9)
                   for actual, expected in (
                       (values[0], values[3]), (values[1], values[4]),
                       (values[2], values[5]), (values[2], values[6]),
                   )):
            return False
    return True


def _bar_schema_errors(bars: dict | None) -> list[str]:
    if bars is None:
        return []
    if not isinstance(bars, dict):
        return ["composite bars must be a JSON object"]

    def is_sha256(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 \
            and all(character in "0123456789abcdef" for character in value)

    def canonical_digest(value: object) -> str | None:
        try:
            encoded = json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(encoded).hexdigest()

    def finite_number(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float)) \
            and math.isfinite(float(value))

    def observed_ranges(value: object) -> dict[str, list[float]] | None:
        if not isinstance(value, dict) \
                or set(value) != {"pooled", "development", "hidden"}:
            return None
        ranges = {}
        for family in ("pooled", "development", "hidden"):
            pair = value.get(family)
            if not isinstance(pair, list) or len(pair) != 2 \
                    or not all(finite_number(item) for item in pair) \
                    or float(pair[0]) > float(pair[1]):
                return None
            ranges[family] = [float(pair[0]), float(pair[1])]
        union = [
            min(ranges["development"][0], ranges["hidden"][0]),
            max(ranges["development"][1], ranges["hidden"][1]),
        ]
        return ranges if ranges["pooled"] == union else None

    def range_inside(observed: list[float], envelope: list[float]) -> bool:
        tolerance = 1e-12
        return (
            float(envelope[0]) - tolerance <= observed[0] <= observed[1]
            <= float(envelope[1]) + tolerance
        )

    q95_feasibility_fields = {
        "q95_sum", "allocation_sum", "reserve_total", "total_minus_q95_sum",
        "all_regions_at_or_above_q95", "allocation_sums_to_total", "feasible",
    }

    def valid_q95_feasibility(value: object) -> bool:
        if not isinstance(value, dict) or set(value) != q95_feasibility_fields \
                or not all(finite_number(value.get(field)) for field in (
                    "q95_sum", "allocation_sum", "reserve_total",
                    "total_minus_q95_sum",
                )) \
                or any(float(value[field]) < 0.0 for field in (
                    "q95_sum", "allocation_sum", "reserve_total",
                )) \
                or any(not isinstance(value.get(field), bool) for field in (
                    "all_regions_at_or_above_q95", "allocation_sums_to_total",
                    "feasible",
                )):
            return False
        total = float(value["reserve_total"])
        tolerance = 1e-10 * max(1.0, abs(total))
        sums_to_total = abs(float(value["allocation_sum"]) - total) <= tolerance
        return value["allocation_sums_to_total"] is sums_to_total \
            and value["feasible"] is (
                value["all_regions_at_or_above_q95"] and sums_to_total
            ) \
            and abs(
                float(value["total_minus_q95_sum"])
                - (total - float(value["q95_sum"]))
            ) <= tolerance

    errors: list[str] = []
    if bars.get("schema") != COMPOSITE_BAR_SCHEMA:
        errors.append(f"schema must be {COMPOSITE_BAR_SCHEMA}")
    blockers = bars.get("blockers")
    if blockers != []:
        errors.append("freeze receipt must contain an empty blockers list")
    if bars.get("frozen") is not True:
        errors.append("freeze receipt must say frozen true")
    if bars.get("quantile") != 0.99:
        errors.append("freeze quantile must be 0.99")
    if bars.get("target_false_fail_rate") != 0.01:
        errors.append("target false-fail rate must be 0.01")
    # A receipt has to name a registered profile and carry that profile's own selection.
    # A receipt with no profile field predates profiles and is read as the full one.
    try:
        registered_selection = {
            gate: list(components) for gate, components
            in gate_profile_selection(bars.get("gate_profile",
                                               DEFAULT_GATE_PROFILE)).items()
        }
    except ValueError:
        errors.append("freeze receipt names an unregistered gate profile")
    else:
        if bars.get("gate_profile_selection", registered_selection) \
                != registered_selection:
            errors.append("gate profile selection differs from the registered profile")
    worlds = bars.get("qualification_worlds")
    if worlds != list(QUALIFICATION_WORLD_NAMES):
        errors.append("freeze receipt must name qual-0 through qual-5 in order")
        worlds = []
    lines = bars.get("reference_lines")
    if lines != list(REFERENCE_LINES):
        errors.append("freeze receipt must name exactly reference lines A, B, and C")
        lines = []
    if bars.get("qualification_world_count") != 6:
        errors.append("qualification world count must be six")
    if bars.get("graded_world_count") != 3:
        errors.append("graded world count must be three")
    report_count = bars.get("replicate_report_count")
    per_pair = bars.get("replicates_per_reference_line_and_world")
    if isinstance(report_count, bool) or not isinstance(report_count, int) \
            or isinstance(per_pair, bool) or not isinstance(per_pair, int) \
            or report_count != REPLICATE_REPORT_COUNT \
            or per_pair != REPLICATES_PER_LINE_WORLD:
        errors.append("freeze receipt has invalid replicate counts")
    elif worlds and lines:
        if report_count != per_pair * len(worlds) * len(lines):
            errors.append("replicate counts do not match the balanced design")
        if per_pair * len(worlds) < 100:
            errors.append("each reference-line p99 needs at least 100 reports")
        if bars.get("reference_report_count") != len(lines) * len(worlds):
            errors.append("final reference count does not match the balanced design")
    if bars.get("reference_report_count") != REFERENCE_REPORT_COUNT:
        errors.append("the exact eighteen final reference reports are required")
    if bars.get("paired_resamples_per_world") != REPLICATES_PER_LINE_WORLD \
            or bars.get("paired_resample_count") \
            != REPLICATES_PER_LINE_WORLD * len(QUALIFICATION_WORLD_NAMES):
        errors.append("the exact seventeen paired resamples per world are required")
    control_count = bars.get("control_report_count")
    if isinstance(control_count, bool) or not isinstance(control_count, int) \
            or control_count != len(REQUIRED_SCIENTIFIC_CONTROLS) * 6:
        errors.append("the complete twenty-two-control six-world battery is required")
    if bars.get("development_diagnostic_report_count") \
            != DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT:
        errors.append("the separate twenty-four-report development diagnostic block is required")
    expected_run_receipts = (
        REFERENCE_REPORT_COUNT
        + REPLICATE_REPORT_COUNT
        + len(REQUIRED_SCIENTIFIC_CONTROLS) * len(QUALIFICATION_WORLD_NAMES)
        + DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT
    )
    if bars.get("run_receipt_count") != expected_run_receipts:
        errors.append("run receipt count differs from the complete evidence design")
    if not is_sha256(bars.get("runner_digest_sha256")) \
            or not is_sha256(bars.get("measurement_contract_digest_sha256")):
        errors.append("freeze receipt lacks its common runner or measurement contract digest")
    rates_by_line = bars.get("achieved_false_fail_rates_by_reference_line")
    rates_by_line_ok = isinstance(rates_by_line, dict) \
        and set(rates_by_line) == set(REFERENCE_LINES)
    if rates_by_line_ok:
        for line in REFERENCE_LINES:
            line_rates = rates_by_line[line]
            if not isinstance(line_rates, dict) \
                    or set(line_rates) != set(COMPOSITE_GATE_COMPONENTS) \
                    or any(
                        not finite_number(value)
                        or not 0.0 <= float(value) <= 0.01
                        for value in line_rates.values()
                    ):
                rates_by_line_ok = False
                break
    if not rates_by_line_ok:
        errors.append("per-reference-line false-fail rates differ from the design")
        rates_by_line = {}
    rates = bars.get("achieved_false_fail_rates")
    if not isinstance(rates, dict) or set(rates) != set(COMPOSITE_GATE_COMPONENTS):
        errors.append("conservative false-fail rates differ from the five gates")
        rates = {}
    elif rates_by_line:
        for gate, value in rates.items():
            expected_rate = max(
                float(rates_by_line[line][gate]) for line in REFERENCE_LINES
            )
            if not finite_number(value) \
                    or not 0.0 <= float(value) <= 0.01 \
                    or not math.isclose(
                        float(value), expected_rate, rel_tol=1e-12, abs_tol=1e-15
                    ):
                errors.append(f"{gate}: conservative false-fail rate is inconsistent")
    if bars.get("achieved_false_fail_rate_method") \
            != "per-reference-line joint max-severity empirical p99":
        errors.append("false-fail calibration method differs from the registered design")
    target_product = 0.99 ** (len(COMPOSITE_GATE_COMPONENTS) * 3)
    if not finite_number(bars.get("target_marginal_product")) \
            or not math.isclose(float(bars["target_marginal_product"]), target_product,
                                rel_tol=1e-12, abs_tol=1e-15):
        errors.append("target marginal product differs from the registered design")
    products_by_line = bars.get("achieved_marginal_rate_product_by_reference_line")
    products_ok = isinstance(products_by_line, dict) \
        and set(products_by_line) == set(REFERENCE_LINES) and bool(rates_by_line)
    if products_ok:
        for line in REFERENCE_LINES:
            expected_line_product = math.prod(
                (1.0 - float(rates_by_line[line][gate])) ** 3
                for gate in COMPOSITE_GATE_COMPONENTS
            )
            if not finite_number(products_by_line[line]) or not math.isclose(
                float(products_by_line[line]), expected_line_product,
                rel_tol=1e-12, abs_tol=1e-15,
            ):
                products_ok = False
                break
    if not products_ok:
        errors.append("per-reference-line marginal products are inconsistent")
    achieved_product = min(float(value) for value in products_by_line.values()) \
        if products_ok else None
    if achieved_product is None \
            or not finite_number(bars.get("achieved_marginal_rate_product")) \
            or not math.isclose(float(bars["achieved_marginal_rate_product"]),
                                achieved_product, rel_tol=1e-12, abs_tol=1e-15):
        errors.append("conservative marginal product does not match the line products")
    if bars.get("reference_failures") != []:
        errors.append("freeze receipt contains final reference failures")

    binding_base_keys = {
        "schema", "kind", "world", "method_digest_sha256",
        "runner_digest_sha256", "measurement_contract_digest_sha256",
        "measurement_params",
        "run_receipt_digest_sha256", "packet_digest_sha256",
        "contract_digest_sha256", "submission_digest_sha256",
        "verifier_digest_sha256", "verifier_report_digest_sha256",
        "reserve_q95_feasibility_digest_sha256",
        "reserve_tail_evidence", "reserve_tail_evidence_digest_sha256",
        "reserve_rule_evidence", "reserve_rule_evidence_digest_sha256",
        "continuation_shock_redraw_evidence_digest_sha256",
        "continuation_shock_redraw_file_sha256",
        "continuation_source_law_sha256",
        "packet_input_sha256", "evidence_id",
    }

    def valid_binding(row: object, expected_kind: str) -> bool:
        if not isinstance(row, dict) or row.get("kind") != expected_kind \
                or row.get("schema") != FREEZE_EVIDENCE_BINDING_SCHEMA:
            return False
        if row.get("measurement_params") != REGISTERED_MEASUREMENT_PARAMS:
            return False
        expected_keys = set(binding_base_keys)
        identity_field: str
        if expected_kind == "reference":
            expected_keys.update({
                "reference_line", "elder_reference_evidence",
                "elder_reference_evidence_digest_sha256",
            })
            identity_field = "reference_line"
        elif expected_kind == "replicate":
            expected_keys.update({
                "reference_line", "replicate_id", "resample_digest_sha256",
                "resampling_design",
            })
            identity_field = "reference_line"
        elif expected_kind == "control":
            expected_keys.add("control")
            identity_field = "control"
        elif expected_kind == "diagnostic":
            expected_keys.add("diagnostic")
            identity_field = "diagnostic"
        else:
            return False
        digest_fields = (
            "packet_digest_sha256", "contract_digest_sha256",
            "submission_digest_sha256", "verifier_digest_sha256",
            "method_digest_sha256", "runner_digest_sha256",
            "measurement_contract_digest_sha256", "run_receipt_digest_sha256",
            "verifier_report_digest_sha256",
            "reserve_q95_feasibility_digest_sha256",
            "reserve_tail_evidence_digest_sha256",
            "reserve_rule_evidence_digest_sha256",
            "continuation_shock_redraw_evidence_digest_sha256",
            "continuation_shock_redraw_file_sha256",
            "continuation_source_law_sha256",
        )
        evidence_id = row.get("evidence_id")
        unsigned = dict(row)
        unsigned.pop("evidence_id", None)
        tail = row.get("reserve_tail_evidence")
        tail_ok = isinstance(tail, dict) \
            and set(tail) == {
                "schema", "valid", "q95_sum", "es95_sum",
                "reserve_submission_sha256",
            } \
            and tail.get("schema") == RESERVE_TAIL_EVIDENCE_SCHEMA \
            and tail.get("valid") is True \
            and finite_number(tail.get("q95_sum")) \
            and finite_number(tail.get("es95_sum")) \
            and 0.0 <= float(tail["q95_sum"]) <= float(tail["es95_sum"]) \
            and is_sha256(tail.get("reserve_submission_sha256")) \
            and canonical_digest(tail) \
            == row.get("reserve_tail_evidence_digest_sha256")
        packet_inputs = row.get("packet_input_sha256")
        packet_inputs_ok = isinstance(packet_inputs, dict) \
            and set(packet_inputs) == {
                "participant/contract.json",
                "participant/experience_history.csv",
                "retained/continuation_liabilities.npz",
            } \
            and all(is_sha256(value) for value in packet_inputs.values())
        reserve_rule = row.get("reserve_rule_evidence")
        reserve_rule_ok = isinstance(reserve_rule, dict) \
            and set(reserve_rule) == {
                "valid", "selected_year", "exposure_person_years",
                "rate_per_person_year", "rounding_unit", "reserve_total",
                "experience_sha256",
            } \
            and reserve_rule.get("valid") is True \
            and not isinstance(reserve_rule.get("selected_year"), bool) \
            and isinstance(reserve_rule.get("selected_year"), int) \
            and all(
                finite_number(reserve_rule.get(field))
                and float(reserve_rule[field]) > 0.0
                for field in (
                    "exposure_person_years", "rate_per_person_year",
                    "rounding_unit", "reserve_total",
                )
            ) \
            and math.isclose(
                float(reserve_rule.get("reserve_total", float("nan"))),
                reserve_total(
                    float(reserve_rule.get("exposure_person_years", float("nan"))),
                    float(reserve_rule.get("rate_per_person_year", float("nan"))),
                    float(reserve_rule.get("rounding_unit", float("nan"))),
                ),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ) \
            and reserve_rule.get("experience_sha256") == (
                packet_inputs.get("participant/experience_history.csv")
                if isinstance(packet_inputs, dict) else None
            ) \
            and canonical_digest(reserve_rule) \
            == row.get("reserve_rule_evidence_digest_sha256")
        elder = row.get("elder_reference_evidence")
        elder_ok = expected_kind != "reference"
        if expected_kind == "reference" and isinstance(elder, dict):
            state_rows = elder.get("state_65_plus_person_years")
            liability_rows = elder.get("liability_mean_by_region")
            elder_ok = set(elder) == {
                "schema", "valid", "packet_digest_sha256",
                "submission_digest_sha256", "state_65_plus_person_years",
                "liability_mean_by_region",
            } \
                and elder.get("schema") == ELDER_REFERENCE_EVIDENCE_SCHEMA \
                and elder.get("valid") is True \
                and elder.get("packet_digest_sha256") \
                == row.get("packet_digest_sha256") \
                and elder.get("submission_digest_sha256") \
                == row.get("submission_digest_sha256") \
                and isinstance(state_rows, list) and len(state_rows) == 6 \
                and {item.get("state") for item in state_rows
                     if isinstance(item, dict)} == set(range(6)) \
                and all(
                    isinstance(item, dict)
                    and set(item) == {
                        "state", "submitted_person_years", "sealed_person_years"
                    }
                    and finite_number(item.get("submitted_person_years"))
                    and float(item["submitted_person_years"]) >= 0.0
                    and finite_number(item.get("sealed_person_years"))
                    and float(item["sealed_person_years"]) >= 0.0
                    for item in state_rows
                ) \
                and isinstance(liability_rows, list) and len(liability_rows) == 6 \
                and {item.get("region") for item in liability_rows
                     if isinstance(item, dict)} == set(range(6)) \
                and all(
                    isinstance(item, dict)
                    and set(item) == {"region", "submitted", "sealed"}
                    and finite_number(item.get("submitted"))
                    and float(item["submitted"]) >= 0.0
                    and finite_number(item.get("sealed"))
                    and float(item["sealed"]) >= 0.0
                    for item in liability_rows
                ) \
                and canonical_digest(elder) \
                == row.get("elder_reference_evidence_digest_sha256")
        return set(row) == expected_keys \
            and all(is_sha256(row.get(field)) for field in digest_fields) \
            and tail_ok and packet_inputs_ok and reserve_rule_ok and elder_ok \
            and (
                expected_kind != "replicate"
                or is_sha256(row.get("resample_digest_sha256"))
            ) \
            and is_sha256(evidence_id) \
            and isinstance(row.get("world"), str) and bool(row["world"]) \
            and isinstance(row.get(identity_field), str) and bool(row[identity_field]) \
            and (
                expected_kind != "replicate"
                or (
                    isinstance(row.get("replicate_id"), str)
                    and bool(row["replicate_id"])
                    and isinstance(row.get("resampling_design"), dict)
                    and bool(row["resampling_design"])
                )
            ) \
            and canonical_digest(unsigned) == evidence_id

    provenance = bars.get("evidence_provenance")
    provenance_ok = isinstance(provenance, dict) \
        and provenance.get("schema") == FREEZE_PROVENANCE_SCHEMA \
        and is_sha256(provenance.get("digest_sha256"))
    if provenance_ok:
        unsigned = dict(provenance)
        recorded_digest = unsigned.pop("digest_sha256")
        provenance_ok = canonical_digest(unsigned) == recorded_digest
    expected_provenance_counts = {
        "reference_reports": bars.get("reference_report_count"),
        "replicate_reports": report_count,
        "control_reports": control_count,
    }
    if provenance_ok:
        for name, expected_count in expected_provenance_counts.items():
            rows = provenance.get(name)
            expected_kind = name.removesuffix("_reports").removesuffix("s")
            if not isinstance(rows, list) or len(rows) != expected_count:
                provenance_ok = False
                break
            for row in rows:
                if not valid_binding(row, expected_kind):
                    provenance_ok = False
                    break
            if not provenance_ok:
                break
    if provenance_ok:
        reference_pairs = [
            (row["reference_line"], row["world"])
            for row in provenance["reference_reports"]
        ]
        expected_reference_pairs = {
            (line, world)
            for line in REFERENCE_LINES
            for world in QUALIFICATION_WORLD_NAMES
        }
        control_pairs = [
            (row["control"], row["world"])
            for row in provenance["control_reports"]
        ]
        expected_control_pairs = {
            (control, world)
            for control in REQUIRED_SCIENTIFIC_CONTROLS
            for world in QUALIFICATION_WORLD_NAMES
        }
        provenance_ok = len(reference_pairs) == len(set(reference_pairs)) \
            and set(reference_pairs) == expected_reference_pairs \
            and len(control_pairs) == len(set(control_pairs)) \
            and set(control_pairs) == expected_control_pairs
    if not provenance_ok:
        errors.append("freeze receipt lacks a valid replay-bound evidence provenance")

    diagnostic_block = bars.get("development_diagnostics")
    diagnostics_ok = isinstance(diagnostic_block, dict) \
        and diagnostic_block.get("schema") == DEVELOPMENT_DIAGNOSTIC_SCHEMA \
        and diagnostic_block.get("registered_diagnostics") \
        == list(DEVELOPMENT_DIAGNOSTICS) \
        and diagnostic_block.get("development_worlds") == list(DEVELOPMENT_WORLD_NAMES) \
        and diagnostic_block.get("report_count") == DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT \
        and diagnostic_block.get("counts_as_qualification_control") is False \
        and is_sha256(diagnostic_block.get("digest_sha256"))
    diagnostic_rows = diagnostic_block.get("reports") \
        if isinstance(diagnostic_block, dict) else None
    if diagnostics_ok:
        unsigned = dict(diagnostic_block)
        recorded_digest = unsigned.pop("digest_sha256")
        diagnostics_ok = canonical_digest(unsigned) == recorded_digest \
            and isinstance(diagnostic_rows, list) \
            and len(diagnostic_rows) == DEVELOPMENT_DIAGNOSTIC_REPORT_COUNT \
            and all(valid_binding(row, "diagnostic") for row in diagnostic_rows)
    if diagnostics_ok:
        observed_pairs = [
            (row["diagnostic"], row["world"]) for row in diagnostic_rows
        ]
        expected_pairs = {
            (name, world)
            for name in DEVELOPMENT_DIAGNOSTICS
            for world in DEVELOPMENT_WORLD_NAMES
        }
        diagnostics_ok = len(observed_pairs) == len(set(observed_pairs)) \
            and set(observed_pairs) == expected_pairs
    if not diagnostics_ok:
        errors.append("development diagnostics are missing or counted as controls")

    all_bound_rows: list[dict] = []
    if provenance_ok:
        for name in expected_provenance_counts:
            all_bound_rows.extend(provenance[name])
    if diagnostics_ok:
        all_bound_rows.extend(diagnostic_rows)
    if provenance_ok and diagnostics_ok:
        receipt_digests = [row["run_receipt_digest_sha256"] for row in all_bound_rows]
        evidence_ids = [row["evidence_id"] for row in all_bound_rows]
        common_runner = {row["runner_digest_sha256"] for row in all_bound_rows}
        common_contract = {
            row["measurement_contract_digest_sha256"] for row in all_bound_rows
        }
        common_verifier = {
            row["verifier_digest_sha256"] for row in all_bound_rows
        }
        if len(receipt_digests) != len(set(receipt_digests)) \
                or len(evidence_ids) != len(set(evidence_ids)):
            errors.append("freeze evidence reuses a run receipt or evidence identifier")
        if common_runner != {bars.get("runner_digest_sha256")} \
                or common_contract != {bars.get("measurement_contract_digest_sha256")} \
                or len(common_verifier) != 1:
            errors.append(
                "freeze evidence does not share one verifier, runner, and measurement contract"
            )
        def same_binding(rows: list[dict], label: str) -> None:
            if len({row["packet_digest_sha256"] for row in rows}) != 1 \
                    or len({row["contract_digest_sha256"] for row in rows}) != 1:
                errors.append(f"{label}: freeze evidence packet or contract binding differs")
            if len({
                canonical_digest(row["packet_input_sha256"]) for row in rows
            }) != 1:
                errors.append(f"{label}: freeze evidence packet input binding differs")
            if len({
                canonical_digest(row["reserve_rule_evidence"]) for row in rows
            }) != 1:
                errors.append(f"{label}: freeze evidence reserve rule binding differs")
            for field in (
                "continuation_shock_redraw_evidence_digest_sha256",
                "continuation_shock_redraw_file_sha256",
                "continuation_source_law_sha256",
            ):
                if len({row[field] for row in rows}) != 1:
                    errors.append(f"{label}: freeze evidence {field} differs")

        base_by_world: dict[str, list[dict]] = {}
        for row in (
            provenance["reference_reports"] + provenance["control_reports"]
        ):
            base_by_world.setdefault(row["world"], []).append(row)
        for world in QUALIFICATION_WORLD_NAMES:
            same_binding(base_by_world.get(world, []), world)
        diagnostics_by_world: dict[str, list[dict]] = {}
        for row in diagnostic_rows:
            diagnostics_by_world.setdefault(row["world"], []).append(row)
        for world in DEVELOPMENT_WORLD_NAMES:
            same_binding(diagnostics_by_world.get(world, []), world)

        identity_digests: dict[str, set[str]] = {}
        for row in all_bound_rows:
            if row["kind"] in {"reference", "replicate"}:
                identity = f"reference:{row['reference_line']}"
            elif row["kind"] == "control":
                identity = f"control:{row['control']}"
            else:
                identity = f"diagnostic:{row['diagnostic']}"
            identity_digests.setdefault(identity, set()).add(row["method_digest_sha256"])
        if any(len(values) != 1 for values in identity_digests.values()):
            errors.append("a line, control, or diagnostic changes method digest across runs")
        stable_digests = [next(iter(values)) for values in identity_digests.values()]
        expected_identity_count = (
            len(REFERENCE_LINES)
            + len(REQUIRED_SCIENTIFIC_CONTROLS)
            + len(DEVELOPMENT_DIAGNOSTICS)
        )
        if len(identity_digests) != expected_identity_count:
            errors.append("freeze evidence omits a registered method identity")
        elif len(stable_digests) != len(set(stable_digests)):
            errors.append("one method digest is relabeled as multiple evidence methods")

        replicate_rows = provenance["replicate_reports"]
        resampling_design_digests = {
            canonical_digest(row["resampling_design"])
            for row in replicate_rows
        }
        if None in resampling_design_digests or len(resampling_design_digests) != 1:
            errors.append("reference replicates do not share one resampling design")
        paired_digests: dict[str, tuple[str, str]] = {}
        for world in QUALIFICATION_WORLD_NAMES:
            pairs: dict[str, list[dict]] = {}
            for row in replicate_rows:
                if row["world"] == world:
                    pairs.setdefault(row["replicate_id"], []).append(row)
            if len(pairs) != REPLICATES_PER_LINE_WORLD:
                errors.append(f"{world}: paired resample count differs")
                continue
            for replicate_id, rows in pairs.items():
                if sorted(row["reference_line"] for row in rows) != list(REFERENCE_LINES) \
                        or len({row["resample_digest_sha256"] for row in rows}) != 1:
                    errors.append(f"{world}/{replicate_id}: resample is not paired across lines")
                    continue
                label = f"{world}/{replicate_id}"
                same_binding(rows, label)
                base = base_by_world[world][0]
                if rows[0]["contract_digest_sha256"] \
                        != base["contract_digest_sha256"]:
                    errors.append(f"{label}: resample changes the public contract")
                for field in (
                    "continuation_shock_redraw_evidence_digest_sha256",
                    "continuation_shock_redraw_file_sha256",
                    "continuation_source_law_sha256",
                ):
                    if rows[0][field] != base[field]:
                        errors.append(f"{label}: resample changes {field}")
                for name in (
                    "participant/contract.json",
                    "retained/continuation_liabilities.npz",
                ):
                    if rows[0]["packet_input_sha256"][name] \
                            != base["packet_input_sha256"][name]:
                        errors.append(f"{label}: resample changes fixed input {name}")
                for field in (
                    "valid",
                    "selected_year",
                    "exposure_person_years",
                    "rate_per_person_year",
                    "rounding_unit",
                    "reserve_total",
                ):
                    if rows[0]["reserve_rule_evidence"][field] \
                            != base["reserve_rule_evidence"][field]:
                        errors.append(
                            f"{label}: resample changes reserve rule {field}"
                        )
                digest = rows[0]["resample_digest_sha256"]
                owner = (world, replicate_id)
                if digest in paired_digests and paired_digests[digest] != owner:
                    errors.append("a paired resample digest is reused across identifiers")
                paired_digests[digest] = owner

    def valid_digest_bound_audit(value: object, schema: str) -> bool:
        if not isinstance(value, dict) or value.get("schema") != schema \
                or not is_sha256(value.get("digest_sha256")) \
                or value.get("measurement_contract_digest_sha256") \
                != bars.get("measurement_contract_digest_sha256"):
            return False
        unsigned = dict(value)
        recorded = unsigned.pop("digest_sha256")
        return canonical_digest(unsigned) == recorded

    reserve_audits = bars.get("reserve_audits")
    reserve_audits_ok = isinstance(reserve_audits, dict) \
        and set(reserve_audits) == {"qualification", "calibration", "red_team"}
    qualification_audit = reserve_audits.get("qualification") \
        if isinstance(reserve_audits, dict) else None
    calibration_audit = reserve_audits.get("calibration") \
        if isinstance(reserve_audits, dict) else None
    red_team_audit = reserve_audits.get("red_team") \
        if isinstance(reserve_audits, dict) else None
    reserve_audits_ok = reserve_audits_ok \
        and valid_digest_bound_audit(qualification_audit, RESERVE_QUALIFICATION_SCHEMA) \
        and valid_digest_bound_audit(calibration_audit, RESERVE_CALIBRATION_SCHEMA) \
        and valid_digest_bound_audit(red_team_audit, RESERVE_RED_TEAM_SCHEMA)
    if reserve_audits_ok:
        reserve_audits_ok = qualification_audit.get("calibration_audit_digest_sha256") \
            == calibration_audit.get("digest_sha256") \
            and qualification_audit.get("red_team_audit_digest_sha256") \
            == red_team_audit.get("digest_sha256") \
            and qualification_audit.get("reference_lines") == list(REFERENCE_LINES) \
            and qualification_audit.get("qualification_worlds") \
            == list(QUALIFICATION_WORLD_NAMES) \
            and calibration_audit.get("candidate") is True \
            and calibration_audit.get("accepted") is True \
            and calibration_audit.get("blockers") == [] \
            and calibration_audit.get("reference_lines") == list(REFERENCE_LINES) \
            and calibration_audit.get("qualification_worlds") \
            == list(QUALIFICATION_WORLD_NAMES) \
            and calibration_audit.get("target_rule") \
            == "sum(q95) + tail_slack_share * sum(ES95 - q95)" \
            and finite_number(calibration_audit.get("rate_per_person_year")) \
            and float(calibration_audit["rate_per_person_year"]) > 0.0 \
            and finite_number(calibration_audit.get("rate_grid")) \
            and float(calibration_audit["rate_grid"]) == 1.0 \
            and finite_number(calibration_audit.get("tail_slack_share")) \
            and float(calibration_audit["tail_slack_share"]) == 0.25 \
            and red_team_audit.get("independent_unit") == "world" \
            and red_team_audit.get("world_counts") \
            == {"development": 12, "qualification": 6, "total": 18} \
            and red_team_audit.get("reserve_total_public_rule_verified") is True \
            and red_team_audit.get("primary_measure") \
            == "qualification incremental regional R2 over development region means"
    if reserve_audits_ok and provenance_ok:
        reference_rows = provenance["reference_reports"]
        control_rows = provenance["control_reports"]
        reference_ids = {row["evidence_id"] for row in reference_rows}
        proportional_ids = {
            row["evidence_id"] for row in control_rows
            if row.get("control") == "proportional_reserve"
        }
        qualification_references = qualification_audit.get("reference_results")
        qualification_proportional = qualification_audit.get(
            "proportional_reserve_results"
        )
        calibration_rows = calibration_audit.get("evidence")
        reserve_audits_ok = isinstance(qualification_references, list) \
            and len(qualification_references) == REFERENCE_REPORT_COUNT \
            and isinstance(qualification_proportional, list) \
            and len(qualification_proportional) == len(QUALIFICATION_WORLD_NAMES) \
            and isinstance(calibration_rows, list) \
            and len(calibration_rows) == REFERENCE_REPORT_COUNT \
            and {row.get("evidence_id") for row in qualification_references
                 if isinstance(row, dict)} == reference_ids \
            and {row.get("evidence_id") for row in calibration_rows
                 if isinstance(row, dict)} == reference_ids \
            and {row.get("evidence_id") for row in qualification_proportional
                 if isinstance(row, dict)} == proportional_ids
        if reserve_audits_ok:
            result_keys = {
                "reference_line", "world", "evidence_id", "q95_feasible",
                "reserve_skill_pass", "q95_sum", "allocation_sum", "reserve_total",
                "total_minus_q95_sum",
            }
            control_result_keys = (
                result_keys - {"reference_line"}
            ) | {"control"}
            reserve_audits_ok = all(
                isinstance(row, dict) and set(row) == result_keys
                and row.get("reference_line") in REFERENCE_LINES
                and row.get("world") in QUALIFICATION_WORLD_NAMES
                and isinstance(row.get("q95_feasible"), bool)
                and row.get("reserve_skill_pass") is True
                and all(finite_number(row.get(field)) for field in (
                    "q95_sum", "allocation_sum", "reserve_total",
                    "total_minus_q95_sum",
                ))
                for row in qualification_references
            ) and all(
                isinstance(row, dict) and set(row) == control_result_keys
                and row.get("control") == "proportional_reserve"
                and row.get("world") in QUALIFICATION_WORLD_NAMES
                and isinstance(row.get("q95_feasible"), bool)
                and row.get("reserve_skill_pass") is False
                and all(finite_number(row.get(field)) for field in (
                    "q95_sum", "allocation_sum", "reserve_total",
                    "total_minus_q95_sum",
                ))
                for row in qualification_proportional
            ) and all(
                isinstance(row, dict)
                and row.get("reference_line") in REFERENCE_LINES
                and row.get("world") in QUALIFICATION_WORLD_NAMES
                and is_sha256(row.get("evidence_id"))
                for row in calibration_rows
            )
    if reserve_audits_ok and provenance_ok:
        reference_bindings = {
            (row["reference_line"], row["world"]): row
            for row in provenance["reference_reports"]
        }
        proportional_bindings = {
            (row["control"], row["world"]): row
            for row in provenance["control_reports"]
            if row["control"] == "proportional_reserve"
        }
        qualification_by_pair = {
            (row["reference_line"], row["world"]): row
            for row in qualification_references
        }
        proportional_by_pair = {
            (row["control"], row["world"]): row
            for row in qualification_proportional
        }
        calibration_by_pair = {
            (row.get("reference_line"), row.get("world")): row
            for row in calibration_rows
        }
        reserve_audits_ok = len(qualification_by_pair) == len(qualification_references) \
            and set(qualification_by_pair) == set(reference_bindings) \
            and len(proportional_by_pair) == len(qualification_proportional) \
            and set(proportional_by_pair) == set(proportional_bindings) \
            and len(calibration_by_pair) == len(calibration_rows) \
            and set(calibration_by_pair) == set(reference_bindings)
        if reserve_audits_ok:
            slack = float(calibration_audit["tail_slack_share"])
            rate = float(calibration_audit["rate_per_person_year"])
            grid = float(calibration_audit["rate_grid"])
            required_rates: list[float] = []
            rounding_units: set[float] = set()
            for pair, result in qualification_by_pair.items():
                binding = reference_bindings[pair]
                calibration = calibration_by_pair[pair]
                tail = binding.get("reserve_tail_evidence")
                feasibility = {
                    "q95_sum": result["q95_sum"],
                    "allocation_sum": result["allocation_sum"],
                    "reserve_total": result["reserve_total"],
                    "total_minus_q95_sum": result["total_minus_q95_sum"],
                    "all_regions_at_or_above_q95": result["q95_feasible"],
                    "allocation_sums_to_total": True,
                    "feasible": result["q95_feasible"],
                }
                q95 = calibration.get("submitted_q95_sum")
                es95 = calibration.get("submitted_es95_sum")
                candidate_total = calibration.get("candidate_reserve_total")
                candidate_margin = calibration.get("candidate_margin")
                exposure = calibration.get("exposure_person_years")
                rounding_unit = calibration.get("rounding_unit")
                recorded_target = calibration.get("target_reserve_before_rounding")
                required_rate = calibration.get("required_rate")
                numeric = all(finite_number(value) for value in (
                    q95, es95, candidate_total, candidate_margin, exposure,
                    rounding_unit, recorded_target, required_rate,
                ))
                if not numeric:
                    reserve_audits_ok = False
                    break
                target = float(q95) + slack * (float(es95) - float(q95))
                if result["evidence_id"] != binding["evidence_id"] \
                        or calibration.get("evidence_id") != binding["evidence_id"] \
                        or not isinstance(tail, dict) \
                        or not valid_q95_feasibility(feasibility) \
                        or canonical_digest(feasibility) != binding[
                            "reserve_q95_feasibility_digest_sha256"
                        ] \
                        or not math.isclose(
                            float(result["q95_sum"]), float(q95),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or not math.isclose(
                            float(q95), float(tail["q95_sum"]),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or not math.isclose(
                            float(es95), float(tail["es95_sum"]),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or calibration.get("reserve_submission_sha256") \
                        != tail["reserve_submission_sha256"] \
                        or calibration.get("experience_sha256") \
                        != binding["packet_input_sha256"][
                            "participant/experience_history.csv"
                        ] \
                        or not math.isclose(
                            float(result["reserve_total"]), float(candidate_total),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or float(exposure) <= 0.0 \
                        or float(rounding_unit) <= 0.0 \
                        or float(es95) < float(q95) \
                        or float(candidate_total) < float(q95) \
                        or float(candidate_margin) < 0.0 \
                        or not math.isclose(
                            float(recorded_target), target,
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or not math.isclose(
                            float(required_rate), target / float(exposure),
                            rel_tol=1e-12, abs_tol=1e-12,
                        ) \
                        or not math.isclose(
                            float(candidate_total),
                            reserve_total(float(exposure), rate, float(rounding_unit)),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or not math.isclose(
                            float(candidate_margin), float(candidate_total) - target,
                            rel_tol=1e-12, abs_tol=1e-9,
                        ):
                    reserve_audits_ok = False
                    break
                required_rates.append(float(required_rate))
                rounding_units.add(float(rounding_unit))
            if reserve_audits_ok:
                expected_rate = math.ceil(max(required_rates) / grid) * grid
                reserve_audits_ok = len(rounding_units) == 1 and math.isclose(
                    rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12
                )
        if reserve_audits_ok:
            for pair, result in proportional_by_pair.items():
                binding = proportional_bindings[pair]
                feasibility = {
                    "q95_sum": result["q95_sum"],
                    "allocation_sum": result["allocation_sum"],
                    "reserve_total": result["reserve_total"],
                    "total_minus_q95_sum": result["total_minus_q95_sum"],
                    "all_regions_at_or_above_q95": result["q95_feasible"],
                    "allocation_sums_to_total": True,
                    "feasible": result["q95_feasible"],
                }
                if result["evidence_id"] != binding["evidence_id"] \
                        or not valid_q95_feasibility(feasibility) \
                        or canonical_digest(feasibility) != binding[
                            "reserve_q95_feasibility_digest_sha256"
                        ]:
                    reserve_audits_ok = False
                    break
    if reserve_audits_ok and provenance_ok and diagnostics_ok:
        measurement_keys = {
            "schema", "measurement_source", "input_bindings", "independent_unit",
            "world_counts", "regions_per_world", "files_read_per_world",
            "reserve_total_public_rule_verified", "tail_definition",
            "public_quantities", "development_regional_models",
            "qualification_predictive_regional_r2",
            "qualification_incremental_regional_r2_over_region_means",
            "primary_measure", "descriptive_pooled_regional_r2",
            "world_aggregate_tail_r2", "interpretation",
        }
        bound_measurement_keys = measurement_keys | {
            "measurement_contract_digest_sha256", "evidence_cross_binding",
            "digest_sha256",
        }

        def valid_r2(value: object) -> bool:
            return finite_number(value) and float(value) <= 1.0

        def valid_headline(value: object) -> bool:
            if not isinstance(value, dict) \
                    or set(value) != {"q95", "es95", "headline_max"} \
                    or not all(valid_r2(value.get(field)) for field in (
                        "q95", "es95", "headline_max"
                    )):
                return False
            return math.isclose(
                float(value["headline_max"]),
                max(float(value["q95"]), float(value["es95"])),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )

        def valid_models(value: object, regions: int) -> bool:
            if not isinstance(value, dict) or set(value) != {"q95", "es95"}:
                return False
            for outcome in ("q95", "es95"):
                rows = value.get(outcome)
                if not isinstance(rows, list) or len(rows) != regions:
                    return False
                for index, row in enumerate(rows):
                    if not isinstance(row, dict) or set(row) != {
                        "region", "intercept", "reserve_total_coefficient"
                    } or row.get("region") != index \
                            or not finite_number(row.get("intercept")) \
                            or not finite_number(row.get("reserve_total_coefficient")):
                        return False
            return True

        def expected_packet_inputs(
            rows: list[dict], expected_worlds: tuple[str, ...]
        ) -> list[dict] | None:
            result: list[dict] = []
            for world in expected_worlds:
                candidates = {
                    canonical_digest(row.get("packet_input_sha256")):
                    row.get("packet_input_sha256")
                    for row in rows if row.get("world") == world
                }
                if None in candidates or len(candidates) != 1:
                    return None
                result.append({
                    "world": world,
                    "file_sha256": next(iter(candidates.values())),
                })
            return result

        regions = red_team_audit.get("regions_per_world")
        regions_ok = not isinstance(regions, bool) \
            and isinstance(regions, int) and regions > 0
        try:
            source_digest = hashlib.sha256(
                (Path(__file__).resolve().parents[1]
                 / "scripts/red_team_reserve_total.py").read_bytes()
            ).hexdigest()
        except OSError:
            source_digest = None
        source = red_team_audit.get("measurement_source")
        source_ok = isinstance(source, dict) \
            and set(source) == {"file", "sha256"} \
            and source.get("file") == "scripts/red_team_reserve_total.py" \
            and source.get("sha256") == source_digest

        reference_rows = provenance["reference_reports"]
        diagnostic_rows = diagnostic_block["reports"]
        expected_bindings = {
            "development": expected_packet_inputs(
                diagnostic_rows, DEVELOPMENT_WORLD_NAMES
            ),
            "qualification": expected_packet_inputs(
                reference_rows, QUALIFICATION_WORLD_NAMES
            ),
        }
        input_bindings = red_team_audit.get("input_bindings")
        input_bindings_ok = None not in expected_bindings.values() \
            and isinstance(input_bindings, dict) \
            and set(input_bindings) == set(expected_bindings) \
            and input_bindings == expected_bindings

        def expected_public_quantities(
            rows: list[dict], expected_worlds: tuple[str, ...]
        ) -> list[dict] | None:
            result: list[dict] = []
            for world in expected_worlds:
                candidates = {
                    canonical_digest(row.get("reserve_rule_evidence")):
                    row.get("reserve_rule_evidence")
                    for row in rows if row.get("world") == world
                }
                if len(candidates) != 1:
                    return None
                receipt = next(iter(candidates.values()))
                if not isinstance(receipt, dict):
                    return None
                result.append({
                    "world": world,
                    "latest_year_total_exposure": receipt.get(
                        "exposure_person_years"
                    ),
                    "reserve_total": receipt.get("reserve_total"),
                })
            return result

        cross_binding = red_team_audit.get("evidence_cross_binding")
        cross_binding_ok = isinstance(cross_binding, dict) \
            and set(cross_binding) == {
                "qualification_reference_evidence_ids",
                "development_diagnostic_evidence_ids",
            } \
            and cross_binding.get("qualification_reference_evidence_ids") == sorted(
                row["evidence_id"] for row in reference_rows
            ) \
            and cross_binding.get("development_diagnostic_evidence_ids") == sorted(
                row["evidence_id"] for row in diagnostic_rows
            )

        quantities = red_team_audit.get("public_quantities")
        development_quantities = quantities.get("development") \
            if isinstance(quantities, dict) else None
        qualification_quantities = quantities.get("qualification") \
            if isinstance(quantities, dict) else None
        quantities_ok = isinstance(quantities, dict) \
            and set(quantities) == {"development", "qualification"} \
            and isinstance(development_quantities, list) \
            and len(development_quantities) == len(DEVELOPMENT_WORLD_NAMES) \
            and isinstance(qualification_quantities, list) \
            and len(qualification_quantities) == len(QUALIFICATION_WORLD_NAMES) \
            and [row.get("world") for row in development_quantities
                 if isinstance(row, dict)] == list(DEVELOPMENT_WORLD_NAMES) \
            and [row.get("world") for row in qualification_quantities
                 if isinstance(row, dict)] == list(QUALIFICATION_WORLD_NAMES) \
            and all(
                isinstance(row, dict)
                and set(row) == {
                    "world", "latest_year_total_exposure", "reserve_total"
                }
                and finite_number(row.get("latest_year_total_exposure"))
                and float(row["latest_year_total_exposure"]) >= 0.0
                and finite_number(row.get("reserve_total"))
                and float(row["reserve_total"]) >= 0.0
                for row in [*(development_quantities or []),
                            *(qualification_quantities or [])]
            )
        expected_quantities = {
            "development": expected_public_quantities(
                diagnostic_rows, DEVELOPMENT_WORLD_NAMES
            ),
            "qualification": expected_public_quantities(
                reference_rows, QUALIFICATION_WORLD_NAMES
            ),
        }
        if quantities_ok:
            quantities_ok = all(
                expected_quantities[regime] is not None
                and all(
                    observed["world"] == expected["world"]
                    and math.isclose(
                        float(observed["latest_year_total_exposure"]),
                        float(expected["latest_year_total_exposure"]),
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                    and math.isclose(
                        float(observed["reserve_total"]),
                        float(expected["reserve_total"]),
                        rel_tol=1e-12,
                        abs_tol=1e-9,
                    )
                    for observed, expected in zip(
                        quantities[regime], expected_quantities[regime], strict=True
                    )
                )
                for regime in ("development", "qualification")
            )

        predictive = red_team_audit.get("qualification_predictive_regional_r2")
        predictive_per_region = predictive.get("per_region") \
            if isinstance(predictive, dict) else None
        predictive_ok = isinstance(predictive, dict) \
            and set(predictive) == {"q95", "es95", "per_region"} \
            and valid_r2(predictive.get("q95")) \
            and valid_r2(predictive.get("es95")) \
            and isinstance(predictive_per_region, dict) \
            and set(predictive_per_region) == {"q95", "es95"} \
            and all(
                isinstance(predictive_per_region.get(outcome), list)
                and len(predictive_per_region[outcome]) == regions
                and all(value is None or valid_r2(value)
                        for value in predictive_per_region[outcome])
                for outcome in ("q95", "es95")
            ) if regions_ok else False

        descriptive = red_team_audit.get("descriptive_pooled_regional_r2")
        descriptive_ok = isinstance(descriptive, dict) \
            and set(descriptive) == {"q95", "es95", "headline_max", "models"} \
            and valid_headline({
                field: descriptive.get(field)
                for field in ("q95", "es95", "headline_max")
            }) \
            and valid_models(descriptive.get("models"), regions) \
            if regions_ok else False

        aggregate = red_team_audit.get("world_aggregate_tail_r2")
        aggregate_ok = isinstance(aggregate, dict) \
            and set(aggregate) == {"qualification_predictive", "descriptive_pooled"} \
            and valid_headline(aggregate.get("qualification_predictive")) \
            and valid_headline(aggregate.get("descriptive_pooled"))

        reserve_audits_ok = set(red_team_audit) == bound_measurement_keys \
            and source_ok and input_bindings_ok and cross_binding_ok \
            and regions_ok and quantities_ok and predictive_ok \
            and valid_models(
                red_team_audit.get("development_regional_models"), regions
            ) \
            and valid_headline(
                red_team_audit.get(
                    "qualification_incremental_regional_r2_over_region_means"
                )
            ) \
            and descriptive_ok and aggregate_ok \
            and red_team_audit.get("files_read_per_world") == [
                "participant/contract.json",
                "participant/experience_history.csv",
                "retained/continuation_liabilities.npz:liability",
            ] \
            and red_team_audit.get("tail_definition") == {
                "level": 0.95,
                "quantile_rank": "ceil(level * members), one-indexed",
                "expected_shortfall": (
                    "mean of all members at or above the quantile, ties included"
                ),
            } \
            and isinstance(red_team_audit.get("interpretation"), str) \
            and bool(red_team_audit["interpretation"].strip())
    if not reserve_audits_ok:
        errors.append("reserve qualification, calibration, or red-team audit is invalid")

    identification = bars.get("mortality_identification_evidence")
    identification_ok = isinstance(identification, dict) \
        and identification.get("schema") \
        == "meridia.v4.mortality-identification-audit.v1" \
        and identification.get("supports_gate") == "tail_calibration" \
        and is_sha256(identification.get("digest_sha256")) \
        and identification.get("qualification_worlds") \
        == list(QUALIFICATION_WORLD_NAMES)
    identification_rows_by_world: dict[str, dict] = {}
    mortality_decomposition_fields = {
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
    mortality_nonnegative_fields = (
        "history_mortality_rate",
        "horizon_mortality_rate",
        "observed_horizon_to_history_ratio",
        "trend_only_horizon_to_history_ratio",
        "residual_observed_to_trend_ratio",
        "publication_lag_trend_factor",
        "last_exposure_midpoint_to_snapshot_trend_factor",
    )
    mortality_year_fields = (
        "history_mortality_shock_years",
        "lag_mortality_shock_years",
        "designated_horizon_mortality_shock_years",
    )
    if identification_ok:
        unsigned = dict(identification)
        recorded_digest = unsigned.pop("digest_sha256")
        source = unsigned.get("measurement_source")
        summary = unsigned.get("summary")
        rows = unsigned.get("worlds")
        source_path = Path(__file__).resolve().parent / "methods/phase_three.py"
        identification_ok = canonical_digest(unsigned) == recorded_digest \
            and isinstance(source, dict) \
            and source == {
                "file": "meridia/methods/phase_three.py",
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "function": "mortality_gap_decomposition",
            } \
            and isinstance(summary, dict) \
            and set(summary) == {
                "trend_active_during_public_experience_window",
                "trend_starts_only_after_publication",
                "publication_lag_months",
                "publication_lag_trend_effect_percent_range",
                "shock_annual_probability",
                "continuation_shocks_redrawn_per_member",
            } \
            and summary.get("trend_active_during_public_experience_window") is True \
            and summary.get("trend_starts_only_after_publication") is False \
            and summary.get("publication_lag_months") == [12] \
            and summary.get("shock_annual_probability") == 0.20 \
            and summary.get("continuation_shocks_redrawn_per_member") is True \
            and isinstance(rows, list) and len(rows) == len(QUALIFICATION_WORLD_NAMES)
        if identification_ok:
            for row in rows:
                if not isinstance(row, dict) or set(row) != {
                    "world", "packet_manifest_digest_sha256", "packet_input_sha256",
                    "reference_evidence_ids", "shock_redraw_evidence", "decomposition",
                } or row.get("world") in identification_rows_by_world \
                        or not is_sha256(row.get("packet_manifest_digest_sha256")):
                    identification_ok = False
                    break
                inputs = row.get("packet_input_sha256")
                shock = row.get("shock_redraw_evidence")
                decomposition = row.get("decomposition")
                if not isinstance(inputs, dict) or set(inputs) != {
                    "participant/contract.json",
                    "participant/experience_history.csv",
                    "retained/continuation_liabilities.npz",
                } or not all(is_sha256(value) for value in inputs.values()) \
                        or not isinstance(row.get("reference_evidence_ids"), dict) \
                        or set(row["reference_evidence_ids"]) != set(REFERENCE_LINES) \
                        or not all(is_sha256(value)
                                   for value in row["reference_evidence_ids"].values()) \
                        or not isinstance(shock, dict) \
                        or set(shock) != {
                            "schema", "runtime_evidence_file_sha256",
                            "liability_archive_sha256", "runtime_evidence",
                        } \
                        or shock.get("schema") != SHOCK_REDRAW_REPORT_SCHEMA \
                        or not is_sha256(shock.get("runtime_evidence_file_sha256")) \
                        or shock.get("liability_archive_sha256") \
                        != inputs.get("retained/continuation_liabilities.npz") \
                        or not isinstance(decomposition, dict) \
                        or set(decomposition) != mortality_decomposition_fields \
                        or not finite_number(decomposition.get(
                            "hidden_mortality_improvement"
                        )) \
                        or not -1.0 <= float(decomposition[
                            "hidden_mortality_improvement"
                        ]) <= 1.0 \
                        or not all(
                            finite_number(decomposition.get(field))
                            and float(decomposition[field]) >= 0.0
                            for field in mortality_nonnegative_fields
                        ) \
                        or decomposition.get(
                            "trend_active_during_public_experience_window"
                        ) is not True \
                        or decomposition.get("trend_starts_only_after_public_window") \
                        is not False \
                        or decomposition.get("trend_application") \
                        != "all event months relative to the snapshot tick" \
                        or decomposition.get("publication_lag_months") != 12 \
                        or decomposition.get(
                            "last_exposure_midpoint_to_snapshot_months"
                        ) != 18 \
                        or decomposition.get("continuation_shocks_redrawn_per_member") \
                        is not True \
                        or any(
                            not isinstance(decomposition.get(field), list)
                            or any(
                                isinstance(year, bool)
                                or not isinstance(year, int)
                                or year < 0
                                for year in decomposition[field]
                            )
                            or len(decomposition[field])
                            != len(set(decomposition[field]))
                            for field in mortality_year_fields
                        ):
                    identification_ok = False
                    break
                try:
                    from .packet import _validate_shock_redraw_evidence

                    shock_runtime = _validate_shock_redraw_evidence(
                        shock["runtime_evidence"]
                    )
                except (ImportError, TypeError, ValueError):
                    identification_ok = False
                    break
                shock_runtime_file_digest = hashlib.sha256((
                    json.dumps(
                        shock_runtime, indent=1, sort_keys=True, allow_nan=False
                    ) + "\n"
                ).encode("utf-8")).hexdigest()
                if shock.get("runtime_evidence_file_sha256") \
                        != shock_runtime_file_digest \
                        or shock_runtime["redrawn_member_count"] \
                        != shock_runtime["member_count"] \
                        or shock_runtime["distinct_future_schedule_count"] <= 1 \
                        or not 0 < shock_runtime["future_shock_year_count"] \
                        < shock_runtime["future_year_opportunity_count"] \
                        or shock_runtime["future_mortality_spike_year_count"] <= 0 \
                        or decomposition["continuation_shocks_redrawn_per_member"] \
                        is not True:
                    identification_ok = False
                    break
                identification_rows_by_world[row["world"]] = row
            identification_ok = identification_ok \
                and list(identification_rows_by_world) == list(QUALIFICATION_WORLD_NAMES)
        if identification_ok:
            lag_effects = [
                100.0 * (
                    float(row["decomposition"]["publication_lag_trend_factor"])
                    - 1.0
                )
                for row in identification_rows_by_world.values()
            ]
            observed_range = summary.get(
                "publication_lag_trend_effect_percent_range"
            )
            identification_ok = isinstance(observed_range, list) \
                and len(observed_range) == 2 \
                and all(finite_number(value) for value in observed_range) \
                and math.isclose(
                    float(observed_range[0]), min(lag_effects),
                    rel_tol=1e-12, abs_tol=1e-12,
                ) \
                and math.isclose(
                    float(observed_range[1]), max(lag_effects),
                    rel_tol=1e-12, abs_tol=1e-12,
                )

    regime_audit = bars.get("regime_identifiability_audit")
    regime_ok = isinstance(regime_audit, dict) \
        and set(regime_audit) == {
            "schema", "anchor_correlation_threshold", "world_count",
            "world_bindings", "measurement_rows_digest_sha256",
            "generator_source_digest_sha256", "generator_policy", "axes",
            "digest_sha256",
        } \
        and regime_audit.get("schema") == "meridia.v4.regime-identifiability-audit.v2" \
        and is_sha256(regime_audit.get("digest_sha256"))
    if regime_ok:
        unsigned = dict(regime_audit)
        recorded_digest = unsigned.pop("digest_sha256")
        policy = unsigned.get("generator_policy")
        axes = unsigned.get("axes")
        bindings = unsigned.get("world_bindings")
        expected_world_regimes = {
            **{f"dev-{index:02d}": "development" for index in range(12)},
            **{f"qual-{index}": "hidden" for index in range(6)},
        }
        expected_policy = {
            "outside_axis_count": 2,
            "eligible_for_outside_development_band": list(HIDDEN_EXTRAPOLATION_AXES),
            "held_inside_development_band": list(HIDDEN_IN_BAND_AXES),
        }
        regime_ok = canonical_digest(unsigned) == recorded_digest \
            and unsigned.get("anchor_correlation_threshold") == 0.4 \
            and unsigned.get("world_count") == 18 \
            and is_sha256(unsigned.get("measurement_rows_digest_sha256")) \
            and is_sha256(unsigned.get("generator_source_digest_sha256")) \
            and policy == expected_policy \
            and isinstance(axes, dict) and set(axes) == set(REGIME_AXES) \
            and isinstance(bindings, list) and len(bindings) == 18
        if regime_ok:
            observed_world_regimes = {}
            for binding in bindings:
                if not isinstance(binding, dict) \
                        or set(binding) != {
                            "world", "regime", "participant_digest_sha256",
                            "packet_manifest_digest_sha256",
                        } \
                        or not isinstance(binding.get("world"), str) \
                        or not isinstance(binding.get("regime"), str) \
                        or not is_sha256(binding.get("participant_digest_sha256")) \
                        or not is_sha256(binding.get("packet_manifest_digest_sha256")) \
                        or binding["world"] in observed_world_regimes:
                    regime_ok = False
                    break
                observed_world_regimes[binding["world"]] = binding["regime"]
            regime_ok = regime_ok and observed_world_regimes == expected_world_regimes
        if regime_ok:
            axis_fields = {
                "statistic", "expected_sign", "signed_rank_correlation",
                "within_regime_signed_rank_correlation", "correlation_target",
                "realized_mechanism_definition", "axis_intensity_range_observed",
                "realized_mechanism_range_observed",
                "registered_realized_mechanism_envelopes",
                "anchor_correlation_qualified", "disposition", "development_range",
                "hidden_generation_range", "hidden_out_of_band_allowed",
            }
            for axis in REGIME_AXES:
                record = axes[axis]
                signed = record.get("signed_rank_correlation") \
                    if isinstance(record, dict) else None
                within = record.get("within_regime_signed_rank_correlation") \
                    if isinstance(record, dict) else None
                raw_ranges = observed_ranges(
                    record.get("axis_intensity_range_observed")
                    if isinstance(record, dict) else None
                )
                realized_ranges = observed_ranges(
                    record.get("realized_mechanism_range_observed")
                    if isinstance(record, dict) else None
                )
                qualified = finite_number(signed) and float(signed) > 0.4
                base_ok = isinstance(record, dict) \
                    and set(record) == axis_fields \
                    and isinstance(record.get("statistic"), str) \
                    and bool(record["statistic"]) \
                    and record.get("expected_sign") == REGIME_EXPECTED_SIGNS[axis] \
                    and finite_number(signed) and -1.0 <= float(signed) <= 1.0 \
                    and isinstance(within, dict) \
                    and set(within) == {"development", "hidden"} \
                    and all(finite_number(value) and -1.0 <= float(value) <= 1.0
                            for value in within.values()) \
                    and record.get("correlation_target") == "realized_mechanism" \
                    and record.get("realized_mechanism_definition") \
                    == REALIZED_MECHANISM_DEFINITIONS[axis] \
                    and raw_ranges is not None and realized_ranges is not None \
                    and record.get("registered_realized_mechanism_envelopes") \
                    == REALIZED_MECHANISM_ENVELOPES[axis] \
                    and record.get("anchor_correlation_qualified") is qualified \
                    and record.get("development_range") == DEVELOPMENT_AXIS_RANGES[axis] \
                    and range_inside(
                        raw_ranges["development"], DEVELOPMENT_AXIS_RANGES[axis]
                    ) \
                    and range_inside(
                        raw_ranges["pooled"], PUBLIC_AXIS_RANGES[axis]
                    ) \
                    and range_inside(
                        realized_ranges["development"],
                        REALIZED_MECHANISM_ENVELOPES[axis]["development"],
                    ) \
                    and range_inside(
                        realized_ranges["pooled"],
                        REALIZED_MECHANISM_ENVELOPES[axis]["public"],
                    )
                if axis in HIDDEN_IN_BAND_AXES:
                    low, high = DEVELOPMENT_AXIS_RANGES[axis]
                    base_ok = base_ok \
                        and record.get("disposition") == "constrained_to_development_range" \
                        and record.get("hidden_out_of_band_allowed") is False \
                        and record.get("hidden_generation_range") == [low, high] \
                        and range_inside(
                            raw_ranges["hidden"], DEVELOPMENT_AXIS_RANGES[axis]
                        ) \
                        and range_inside(
                            realized_ranges["hidden"],
                            REALIZED_MECHANISM_ENVELOPES[axis]["development"],
                        )
                else:
                    base_ok = base_ok and qualified \
                        and record.get("disposition") == "participant_anchor" \
                        and record.get("hidden_out_of_band_allowed") is True \
                        and record.get("hidden_generation_range") == PUBLIC_AXIS_RANGES[axis] \
                        and range_inside(
                            raw_ranges["hidden"], PUBLIC_AXIS_RANGES[axis]
                        ) \
                        and range_inside(
                            realized_ranges["hidden"],
                            REALIZED_MECHANISM_ENVELOPES[axis]["public"],
                        )
                if not base_ok:
                    regime_ok = False
                    break
    if not regime_ok:
        errors.append("regime identifiability and hidden-axis constraint evidence is invalid")

    if identification_ok:
        if not provenance_ok or not regime_ok:
            identification_ok = False
        else:
            reference_rows_by_world: dict[str, list[dict]] = {}
            for reference in provenance["reference_reports"]:
                reference_rows_by_world.setdefault(reference["world"], []).append(reference)
            regime_bindings_by_world = {
                binding["world"]: binding
                for binding in regime_audit["world_bindings"]
            }
            for world in QUALIFICATION_WORLD_NAMES:
                row = identification_rows_by_world.get(world)
                references_for_world = reference_rows_by_world.get(world, [])
                expected_ids = {
                    reference["reference_line"]: reference["evidence_id"]
                    for reference in references_for_world
                }
                expected_inputs = {
                    canonical_digest(reference["packet_input_sha256"]):
                    reference["packet_input_sha256"]
                    for reference in references_for_world
                }
                shock = row.get("shock_redraw_evidence") if row else None
                if row is None or len(references_for_world) != len(REFERENCE_LINES) \
                        or len(expected_inputs) != 1 \
                        or row.get("packet_input_sha256") \
                        != next(iter(expected_inputs.values())) \
                        or row.get("reference_evidence_ids") != expected_ids \
                        or row.get("packet_manifest_digest_sha256") \
                        != regime_bindings_by_world[world][
                            "packet_manifest_digest_sha256"
                        ] \
                        or not isinstance(shock, dict) \
                        or any(
                            reference.get(
                                "continuation_shock_redraw_evidence_digest_sha256"
                            ) != canonical_digest(shock)
                            or reference.get("continuation_shock_redraw_file_sha256")
                            != shock.get("runtime_evidence_file_sha256")
                            or reference.get("continuation_source_law_sha256")
                            != shock.get("runtime_evidence", {}).get(
                                "continuation_source_law_sha256"
                            )
                            for reference in references_for_world
                        ):
                    identification_ok = False
                    break
    if not identification_ok:
        errors.append("mortality identification evidence is invalid")

    elder_audit = bars.get("elder_reconstruction_audit")
    elder_ok = isinstance(elder_audit, dict) \
        and elder_audit.get("schema") == "meridia.methods.elder_reconstruction_audit.v1" \
        and is_sha256(elder_audit.get("digest_sha256"))
    if elder_ok:
        unsigned = dict(elder_audit)
        recorded_digest = unsigned.pop("digest_sha256")
        method = unsigned.get("method_digest", {})
        shock = unsigned.get("shock_redraw", {})
        eligibility = unsigned.get("eligibility_audit", {})
        scored = eligibility.get("scored", {}) if isinstance(eligibility, dict) else {}
        audit_worlds = unsigned.get("worlds")
        elder_ok = canonical_digest(unsigned) == recorded_digest \
            and isinstance(method, dict) \
            and is_sha256(method.get("source_sha256")) \
            and method.get("before_line") == "A" \
            and method.get("after_line") == "C" \
            and isinstance(shock, dict) \
            and shock.get("annual_probability") == 0.20 \
            and shock.get("independent_per_member") is True \
            and shock.get("magnitude_source") == "participant/contract.json:shock_family" \
            and shock.get("mortality_ranges") == [
                {"kind": "mortality_spike", "range": [1.5, 3.0]}] \
            and shock.get("admission_ranges") == [
                {"kind": "mortality_spike", "range": [1.4, 2.6]}] \
            and scored == {"age_band": "65+", "floor_person_years": 500} \
            and eligibility.get("report_only") == ["65-74", "75-84", "85+"] \
            and eligibility.get("younger_floors_changed") is False \
            and isinstance(audit_worlds, list) \
            and [row.get("world") for row in audit_worlds
                 if isinstance(row, dict)] == list(QUALIFICATION_WORLD_NAMES)
        if elder_ok:
            reference_binding_by_pair = {
                (reference["reference_line"], reference["world"]): reference
                for reference in provenance.get("reference_reports", [])
            } if provenance_ok else {}
            before_values = []
            after_values = []
            for row in audit_worlds:
                exposure = row.get("exposure_65_plus_absolute_error_percent", {})
                world = row.get("world")
                before_binding = reference_binding_by_pair.get(("A", world))
                after_binding = reference_binding_by_pair.get(("C", world))
                mortality_row = identification_rows_by_world.get(world)
                decomposition = row.get("mortality_gap_decomposition")
                if not isinstance(exposure, dict) \
                        or exposure.get("definition") != (
                            "100 * sum_state abs(submitted_state_65plus_person_years - "
                            "sealed_state_65plus_person_years) / sum_state "
                            "sealed_state_65plus_person_years"
                        ) \
                        or not finite_number(exposure.get("before")) \
                        or not finite_number(exposure.get("after")) \
                        or float(exposure["before"]) < 0.0 \
                        or float(exposure["after"]) < 0.0 \
                        or not identification_ok \
                        or before_binding is None or after_binding is None \
                        or row.get("before_report_evidence_id") \
                        != before_binding["evidence_id"] \
                        or row.get("after_report_evidence_id") \
                        != after_binding["evidence_id"] \
                        or not _elder_audit_values_match(
                            row, before_binding, after_binding
                        ) \
                        or not isinstance(decomposition, dict) \
                        or mortality_row is None \
                        or canonical_digest(decomposition) \
                        != canonical_digest(mortality_row["decomposition"]):
                    elder_ok = False
                    break
                before_values.append(float(exposure["before"]))
                after_values.append(float(exposure["after"]))
            if elder_ok:
                before_values.sort()
                after_values.sort()
                before_median = 0.5 * (before_values[2] + before_values[3])
                after_median = 0.5 * (after_values[2] + after_values[3])
                elder_ok = after_median < 10.0 and after_median < before_median
        if elder_ok and provenance_ok:
            after_line = method["after_line"]
            method_digests = {
                row.get("method_digest_sha256")
                for row in provenance.get("reference_reports", [])
                if row.get("reference_line") == after_line
            }
            elder_ok = method_digests == {method["source_sha256"]}
    if not elder_ok:
        errors.append("elder reconstruction qualification audit is invalid")

    control_binding_index = {
        (row["control"], row["world"]): row
        for row in provenance.get("control_reports", [])
    } if provenance_ok else {}
    support = bars.get("control_support")
    registered = support.get("registered_controls_by_gate") \
        if isinstance(support, dict) else None
    separated = support.get("separated_controls_by_gate") \
        if isinstance(support, dict) else None
    matrix = support.get("matrix") if isinstance(support, dict) else None
    expected_registry = {
        gate: list(names) for gate, names in SCIENTIFIC_CONTROLS_BY_GATE.items()
    }
    expected_controls = set(REQUIRED_SCIENTIFIC_CONTROLS)
    control_surface_ok = provenance_ok \
        and isinstance(support, dict) \
        and support.get("requirement") == (
            "every registered control hard-passes structure and fails its primary "
            "composite gate on every qualification world"
        ) \
        and support.get("registered_controls") == sorted(expected_controls) \
        and support.get("registered_controls_by_gate") == expected_registry \
        and support.get("separated_controls_by_gate") == expected_registry \
        and support.get("required_control_count") == len(expected_controls) \
        and support.get("required_report_count") == len(expected_controls) * 6 \
        and support.get("complete_gate_count") == len(COMPOSITE_GATE_COMPONENTS) \
        and support.get("full_separation") is True \
        and support.get("unexpected_controls") == [] \
        and support.get("deletion_candidates") == [] \
        and isinstance(registered, dict) \
        and isinstance(separated, dict) \
        and isinstance(matrix, dict) \
        and set(matrix) == expected_controls
    if not control_surface_ok:
        errors.append("freeze receipt lacks all-control all-world gate separation")
    else:
        raw_gate_bars = bars.get("gates")
        gate_bars = raw_gate_bars if isinstance(raw_gate_bars, dict) else {}
        primary_by_control = {
            control: gate
            for gate, controls in expected_registry.items()
            for control in controls
        }
        for control in sorted(expected_controls):
            record = matrix[control]
            primary_gate = primary_by_control[control]
            base_ok = isinstance(record, dict) \
                and record.get("registered") is True \
                and record.get("primary_gate") == primary_gate \
                and record.get("coverage_complete") is True \
                and record.get("worlds") == worlds \
                and record.get("missing_worlds") == [] \
                and record.get("duplicate_worlds") == [] \
                and record.get("unexpected_worlds") == [] \
                and record.get("hard_structure_pass") is True \
                and record.get("evidence_ids") == [
                    control_binding_index[(control, world)]["evidence_id"]
                    for world in worlds
                ] \
                and isinstance(record.get("gates"), dict) \
                and set(record["gates"]) == set(COMPOSITE_GATE_COMPONENTS)
            for gate, expected_components in COMPOSITE_GATE_COMPONENTS.items():
                result = record["gates"].get(gate) if base_ok else None
                result_ok = isinstance(result, dict) \
                    and result.get("scientifically_registered") \
                    is (gate == primary_gate) \
                    and result.get("hard_invalid_worlds") == [] \
                    and isinstance(result.get("per_world"), dict) \
                    and set(result["per_world"]) == set(worlds)
                failed_worlds: list[str] = []
                passed_worlds: list[str] = []
                if result_ok:
                    for world in worlds:
                        row = result["per_world"][world]
                        binding = control_binding_index[(control, world)]
                        comparisons = row.get("components") \
                            if isinstance(row, dict) else None
                        feasibility = row.get("reserve_q95_feasibility") \
                            if isinstance(row, dict) else None
                        feasibility_ok = valid_q95_feasibility(feasibility) \
                            and canonical_digest(feasibility) == binding[
                                "reserve_q95_feasibility_digest_sha256"
                            ]
                        if not isinstance(row, dict) \
                                or row.get("hard_structure_pass") is not True \
                                or row.get("evidence_id") != binding["evidence_id"] \
                                or not feasibility_ok \
                                or not isinstance(comparisons, dict) \
                                or set(comparisons) != set(expected_components):
                            result_ok = False
                            break
                        has_exceedance = False
                        for component, comparison in comparisons.items():
                            gate_bar = gate_bars.get(gate)
                            component_bars = gate_bar.get("components") \
                                if isinstance(gate_bar, dict) else None
                            bar = component_bars.get(component, {}) \
                                if isinstance(component_bars, dict) else {}
                            value = comparison.get("value") \
                                if isinstance(comparison, dict) else None
                            ceiling = comparison.get("ceiling") \
                                if isinstance(comparison, dict) else None
                            exceeds = comparison.get("exceeds") \
                                if isinstance(comparison, dict) else None
                            if not finite_number(value) or not finite_number(ceiling) \
                                    or ceiling != bar.get("value") \
                                    or not isinstance(exceeds, bool) \
                                    or exceeds != (float(value) > float(ceiling)):
                                result_ok = False
                                break
                            has_exceedance = has_exceedance or exceeds
                        expected_outcome = "fail" if has_exceedance else "pass"
                        if not result_ok \
                                or row.get("failed") is not has_exceedance \
                                or row.get("outcome") != expected_outcome:
                            result_ok = False
                            break
                        (failed_worlds if has_exceedance else passed_worlds).append(world)
                separates = result_ok and failed_worlds == worlds
                result_ok = result_ok \
                    and result.get("failed_worlds") == failed_worlds \
                    and result.get("passed_worlds") == passed_worlds \
                    and result.get("separates_all_worlds") is separates \
                    and (gate != primary_gate or separates)
                if not result_ok:
                    errors.append(f"{gate}/{control}: separation receipt is incomplete")

    gates = bars.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(COMPOSITE_GATE_COMPONENTS):
        errors.append("gate names differ from the five frozen composite gates")
        return errors
    reference_binding_index = {
        (row["reference_line"], row["world"]): row
        for row in provenance.get("reference_reports", [])
    } if provenance_ok else {}
    replicate_binding_index = {
        (row["reference_line"], row["world"], row["replicate_id"]): row
        for row in provenance.get("replicate_reports", [])
    } if provenance_ok else {}
    expected_replicate_evidence_ids = sorted(
        row["evidence_id"] for row in replicate_binding_index.values()
    )
    expected_line_replicate_evidence_ids = {
        line: sorted(
            row["evidence_id"]
            for (observed_line, _, _), row in replicate_binding_index.items()
            if observed_line == line
        )
        for line in REFERENCE_LINES
    }
    for gate, expected in COMPOSITE_GATE_COMPONENTS.items():
        gate_record = gates[gate]
        components = gate_record.get("components") \
            if isinstance(gate_record, dict) else None
        if not isinstance(components, dict) or set(components) != set(expected):
            errors.append(f"{gate}: component names differ from the verifier")
            continue
        normalizers = gate_record.get("normalizers")
        severity_ceiling = gate_record.get("severity_ceiling")
        line_calibration = gate_record.get("reference_line_calibration")
        expected_per_line = REPLICATES_PER_LINE_WORLD * len(QUALIFICATION_WORLD_NAMES)
        expected_rank = math.ceil(0.99 * expected_per_line)
        gate_ok = gate_record.get("calibration_method") \
            == "per-reference-line-joint-max-severity" \
            and normalizers == GATE_COMPONENT_NORMALIZERS[gate] \
            and finite_number(severity_ceiling) \
            and float(severity_ceiling) >= 0.0 \
            and gate_record.get("quantile") == 0.99 \
            and gate_record.get("target_false_fail_rate") == 0.01 \
            and gate_record.get("sample_count_per_reference_line") == expected_per_line \
            and gate_record.get("order_statistic_rank_per_reference_line") \
            == expected_rank \
            and isinstance(line_calibration, dict) \
            and set(line_calibration) == set(REFERENCE_LINES)
        observed_ceiling_lines: list[str] = []
        if gate_ok:
            for line in REFERENCE_LINES:
                line_record = line_calibration[line]
                expected_keys = {
                    "severity_p99", "sample_count", "order_statistic_rank",
                    "false_fail_count", "false_fail_rate", "quantile_witnesses",
                    "replicate_evidence_ids", "replicate_evidence_digest_sha256",
                }
                count = line_record.get("false_fail_count") \
                    if isinstance(line_record, dict) else None
                line_rate = line_record.get("false_fail_rate") \
                    if isinstance(line_record, dict) else None
                p99 = line_record.get("severity_p99") \
                    if isinstance(line_record, dict) else None
                evidence_ids = line_record.get("replicate_evidence_ids") \
                    if isinstance(line_record, dict) else None
                digest = line_record.get("replicate_evidence_digest_sha256") \
                    if isinstance(line_record, dict) else None
                witnesses = line_record.get("quantile_witnesses") \
                    if isinstance(line_record, dict) else None
                line_ok = isinstance(line_record, dict) \
                    and set(line_record) == expected_keys \
                    and finite_number(p99) \
                    and 0.0 <= float(p99) <= float(severity_ceiling) \
                    and line_record.get("sample_count") == expected_per_line \
                    and line_record.get("order_statistic_rank") == expected_rank \
                    and isinstance(count, int) and not isinstance(count, bool) \
                    and 0 <= count <= 1 \
                    and finite_number(line_rate) \
                    and math.isclose(
                        float(line_rate), count / expected_per_line,
                        rel_tol=1e-12, abs_tol=1e-15,
                    ) \
                    and bool(rates_by_line) \
                    and math.isclose(
                        float(line_rate), float(rates_by_line[line][gate]),
                        rel_tol=1e-12, abs_tol=1e-15,
                    ) \
                    and evidence_ids == expected_line_replicate_evidence_ids[line] \
                    and is_sha256(digest) \
                    and hashlib.sha256(
                        "\n".join(evidence_ids).encode("utf-8")
                    ).hexdigest() == digest \
                    and isinstance(witnesses, list) and bool(witnesses)
                if line_ok:
                    witness_identities: set[tuple[str, str, str]] = set()
                    for witness in witnesses:
                        component_values = witness.get("component_values") \
                            if isinstance(witness, dict) else None
                        component_severities = witness.get("component_severities") \
                            if isinstance(witness, dict) else None
                        identity = (
                            witness.get("reference_line"), witness.get("world"),
                            witness.get("replicate_id"),
                        ) if isinstance(witness, dict) else (None, None, None)
                        witness_ok = isinstance(witness, dict) \
                            and set(witness) == {
                                "reference_line", "world", "replicate_id", "evidence_id",
                                "component_values", "component_severities", "severity",
                            } \
                            and identity[0] == line \
                            and identity not in witness_identities \
                            and witness.get("evidence_id") \
                            == replicate_binding_index.get(identity, {}).get("evidence_id") \
                            and isinstance(component_values, dict) \
                            and set(component_values) == set(expected) \
                            and isinstance(component_severities, dict) \
                            and set(component_severities) == set(expected) \
                            and all(
                                finite_number(component_values[component])
                                and finite_number(component_severities[component])
                                and math.isclose(
                                    float(component_severities[component]),
                                    float(component_values[component])
                                    / GATE_COMPONENT_NORMALIZERS[gate][component],
                                    rel_tol=1e-12, abs_tol=1e-15,
                                )
                                for component in expected
                            ) \
                            and finite_number(witness.get("severity")) \
                            and math.isclose(
                                float(witness["severity"]),
                                max(float(value) for value in component_severities.values()),
                                rel_tol=1e-12, abs_tol=1e-15,
                            ) \
                            and math.isclose(
                                float(witness["severity"]), float(p99),
                                rel_tol=1e-12, abs_tol=1e-15,
                            )
                        if not witness_ok:
                            line_ok = False
                            break
                        witness_identities.add(identity)
                if not line_ok:
                    gate_ok = False
                    break
                if math.isclose(
                    float(p99), float(severity_ceiling), rel_tol=0.0, abs_tol=0.0
                ):
                    observed_ceiling_lines.append(line)
        if gate_ok:
            gate_ok = gate_record.get("ceiling_witness_lines") \
                == observed_ceiling_lines and bool(observed_ceiling_lines)
        if not gate_ok:
            errors.append(f"{gate}: joint per-reference-line calibration is invalid")
        for component in expected:
            record = components[component]
            if not isinstance(record, dict) or record.get("direction") != "ceiling":
                errors.append(f"{gate}/{component}: direction must be ceiling")
                continue
            value = record.get("value")
            if not finite_number(value) or float(value) < 0.0:
                errors.append(f"{gate}/{component}: value must be finite and non-negative")
                continue
            expected_range = list(COMPOSITE_COMPONENT_RANGES[(gate, component)])
            if record.get("range") != expected_range:
                errors.append(f"{gate}/{component}: attainable range differs")
            high = expected_range[1]
            if high is not None and float(value) > float(high):
                errors.append(f"{gate}/{component}: value exceeds its attainable range")
            normalizer = GATE_COMPONENT_NORMALIZERS[gate][component]
            expected_value = float(severity_ceiling) * normalizer \
                if finite_number(severity_ceiling) else float("nan")
            if high is not None:
                expected_value = min(expected_value, float(high))
            if record.get("normalizer") != normalizer \
                    or record.get("calibration_method") \
                    != "derived-from-joint-gate-max-severity" \
                    or not math.isclose(
                        float(value), expected_value, rel_tol=1e-12, abs_tol=1e-15
                    ) \
                    or record.get("quantile") != 0.99 \
                    or record.get("target_false_fail_rate") != 0.01:
                errors.append(f"{gate}/{component}: joint calibration metadata differs")
            sample_count = record.get("sample_count")
            if isinstance(sample_count, bool) or not isinstance(sample_count, int) \
                    or sample_count != report_count \
                    or record.get("sample_count_per_reference_line") != expected_per_line \
                    or record.get("order_statistic_rank_per_reference_line") \
                    != expected_rank:
                errors.append(f"{gate}/{component}: per-line sample receipt differs")
            if worlds and record.get("worlds") != worlds:
                errors.append(f"{gate}/{component}: qualification worlds differ")
            if lines and record.get("witnesses") != lines:
                errors.append(f"{gate}/{component}: reference witnesses differ")
            if not isinstance(record.get("supporting_controls"), list) \
                    or not record["supporting_controls"] \
                    or (isinstance(separated, dict)
                        and record["supporting_controls"] != separated[gate]):
                errors.append(f"{gate}/{component}: supporting controls are missing")
            witnesses = record.get("reference_witnesses")
            expected_pairs = {(line, world) for line in lines for world in worlds}
            witness_identities_valid = isinstance(witnesses, list) and all(
                isinstance(witness, dict)
                and isinstance(witness.get("reference_line"), str)
                and isinstance(witness.get("world"), str)
                for witness in witnesses
            )
            witness_pairs = {
                (witness.get("reference_line"), witness.get("world"))
                for witness in witnesses
            } if witness_identities_valid else set()
            if not witness_identities_valid or witness_pairs != expected_pairs \
                    or len(witnesses) != len(expected_pairs) \
                    or any(not isinstance(witness, dict)
                           or witness.get("pass") is not True
                           or not finite_number(witness.get("value"))
                           or float(witness["value"]) > float(value)
                           or witness.get("evidence_id") != reference_binding_index.get((
                               witness.get("reference_line"), witness.get("world")
                           ), {}).get("evidence_id")
                           for witness in witnesses):
                errors.append(f"{gate}/{component}: final witness receipt is incomplete")
            evidence_ids = record.get("replicate_evidence_ids")
            digest = record.get("replicate_evidence_digest_sha256")
            if not isinstance(evidence_ids, list) or len(evidence_ids) != report_count \
                    or any(not is_sha256(item) for item in evidence_ids) \
                    or len(set(evidence_ids)) != len(evidence_ids) \
                    or evidence_ids != expected_replicate_evidence_ids \
                    or not is_sha256(digest) \
                    or hashlib.sha256("\n".join(sorted(evidence_ids)).encode("utf-8")) \
                    .hexdigest() != digest:
                errors.append(f"{gate}/{component}: replicate evidence receipt is invalid")
            component_p99 = record.get("empirical_p99_by_reference_line")
            observed_ranges = record.get("observed_range_by_reference_line")
            component_witnesses = record.get(
                "component_quantile_witnesses_by_reference_line"
            )
            component_rates = record.get(
                "component_exceedance_rate_at_joint_ceiling_by_reference_line"
            )
            diagnostics_ok = bool(rates_by_line) and all(
                isinstance(item, dict) and set(item) == set(REFERENCE_LINES)
                for item in (
                    component_p99, observed_ranges, component_witnesses,
                    component_rates,
                )
            )
            if diagnostics_ok:
                low_bound, high_bound = COMPOSITE_COMPONENT_RANGES[(gate, component)]
                for line in REFERENCE_LINES:
                    p99 = component_p99[line]
                    observed = observed_ranges[line]
                    witnesses = component_witnesses[line]
                    component_rate = component_rates[line]
                    line_ok = finite_number(p99) \
                        and float(p99) >= low_bound \
                        and (high_bound is None or float(p99) <= high_bound) \
                        and float(p99) <= float(value) \
                        and isinstance(observed, list) and len(observed) == 2 \
                        and all(finite_number(item) for item in observed) \
                        and float(observed[0]) <= float(p99) <= float(observed[1]) \
                        and float(observed[0]) >= low_bound \
                        and (high_bound is None or float(observed[1]) <= high_bound) \
                        and finite_number(component_rate) \
                        and 0.0 <= float(component_rate) \
                        <= float(rates_by_line[line][gate]) \
                        and isinstance(witnesses, list) and bool(witnesses)
                    identities: set[tuple[str, str, str]] = set()
                    if line_ok:
                        for witness in witnesses:
                            identity = (
                                witness.get("reference_line"), witness.get("world"),
                                witness.get("replicate_id"),
                            ) if isinstance(witness, dict) else (None, None, None)
                            witness_ok = isinstance(witness, dict) \
                                and set(witness) == {
                                    "reference_line", "world", "replicate_id",
                                    "evidence_id", "value",
                                } \
                                and identity[0] == line \
                                and identity not in identities \
                                and witness.get("evidence_id") \
                                == replicate_binding_index.get(identity, {}).get(
                                    "evidence_id"
                                ) \
                                and finite_number(witness.get("value")) \
                                and math.isclose(
                                    float(witness["value"]), float(p99),
                                    rel_tol=0.0, abs_tol=0.0,
                                )
                            if not witness_ok:
                                line_ok = False
                                break
                            identities.add(identity)
                    if not line_ok:
                        diagnostics_ok = False
                        break
            if not diagnostics_ok:
                errors.append(f"{gate}/{component}: component evidence is invalid")
            if gate == "exposures_and_rates":
                eligible = record.get("eligible_cells")
                by_world = eligible.get("by_world") if isinstance(eligible, dict) else None
                audits = eligible.get("band_audit_by_world") \
                    if isinstance(eligible, dict) else None
                if not isinstance(by_world, dict) or set(by_world) != set(worlds) \
                        or any(not isinstance(cells, list) or not cells
                               for cells in by_world.values()):
                    errors.append(f"{gate}/{component}: eligible-cell provenance is incomplete")
                expected_bands = {
                    "0-17", "18-44", "45-64", "65-74", "75-84", "85+",
                    "18-64", "65+",
                }
                audit_ok = isinstance(audits, dict) and set(audits) == set(worlds)
                if audit_ok:
                    expected_state_sex = {
                        (state, sex) for state in range(6) for sex in SEX_LABELS
                    }
                    for world in worlds:
                        world_record = audits[world]
                        bands = world_record.get("bands") \
                            if isinstance(world_record, dict) else None
                        if not isinstance(bands, dict) or set(bands) != expected_bands \
                                or bands["65+"].get("status") != "scored" \
                                or bands["65+"].get("floor_person_years") != 500.0 \
                                or bands["65+"].get("eligible_count") \
                                != bands["65+"].get("cell_count") \
                                or any(bands[band].get("status") != "report-only"
                                       for band in ("65-74", "75-84", "85+")):
                            audit_ok = False
                            break
                        for band in expected_bands:
                            band_record = bands[band]
                            floor = band_record.get("floor_person_years")
                            cells = band_record.get("cells")
                            if not finite_number(floor) or float(floor) <= 0.0 \
                                    or band_record.get("cell_count") != 12 \
                                    or not isinstance(cells, list) or len(cells) != 12:
                                audit_ok = False
                                break
                            pairs: set[tuple[int, str]] = set()
                            values: list[float] = []
                            eligible_count = 0
                            for cell in cells:
                                state = cell.get("state") if isinstance(cell, dict) else None
                                sex = cell.get("sex") if isinstance(cell, dict) else None
                                exposure = cell.get("exposure_person_years") \
                                    if isinstance(cell, dict) else None
                                decision = cell.get("eligible") \
                                    if isinstance(cell, dict) else None
                                if isinstance(state, bool) or not isinstance(state, int) \
                                        or not isinstance(sex, str) \
                                        or not finite_number(exposure) \
                                        or float(exposure) < 0.0 \
                                        or not isinstance(decision, bool) \
                                        or decision != (float(exposure) >= float(floor)):
                                    audit_ok = False
                                    break
                                pairs.add((state, sex))
                                values.append(float(exposure))
                                eligible_count += int(decision)
                            if not audit_ok:
                                break
                            if pairs != expected_state_sex \
                                    or band_record.get("eligible_count") != eligible_count \
                                    or not math.isclose(
                                        float(band_record.get(
                                            "minimum_exposure_person_years", float("nan"))),
                                        min(values), rel_tol=0.0, abs_tol=0.0,
                                    ):
                                audit_ok = False
                                break
                        if not audit_ok:
                            break
                    if audit_ok:
                        audit_ok = all(
                            sum(audits[world]["bands"][band]["cell_count"]
                                for world in worlds) == 72
                            for band in ("65-74", "75-84", "85+", "65+")
                        )
                if not audit_ok:
                    errors.append(
                        f"{gate}/{component}: per-band qualification counts are incomplete"
                    )
    return errors


UNDEFINED_COMPONENT_REASONS: dict[tuple[str, str], str] = {
    ("reserve_skill", "skill_loss"): (
        "the reserve skill denominator J(A_B) - J(A*) is not positive at this published "
        "total, so no allocation is separable from any other and the skill score does "
        "not exist"
    ),
}


def component_value(value: object) -> float:
    """Read one composite component as a float, without ever raising.

    An in-memory report carries an undefined component as a non-finite float and a
    receipt read back from disk carries the same thing as null. Both mean the component
    does not exist, and a gate has to reach the same verdict either way.
    """
    if value is None or isinstance(value, bool):
        return float("nan")
    if isinstance(value, Mapping):
        value = value.get("value")
        if value is None or isinstance(value, bool):
            return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _undefined_reason(gate: str, components: Sequence[str]) -> str:
    """Name the undefined components, and say why when the cause is registered."""
    reason = f"non-finite components {list(components)}"
    notes = [UNDEFINED_COMPONENT_REASONS[(gate, component)]
             for component in components
             if (gate, component) in UNDEFINED_COMPONENT_REASONS]
    return "; ".join([reason, *notes])


def evaluate_composite_gates(composite_metrics: dict, bars: dict | None,
                             hard_pass: bool,
                             gate_profile: str = DEFAULT_GATE_PROFILE) -> dict[str, dict]:
    """Compare every composite against its frozen bar, deciding only on the profile.

    All five gates are compared and reported. ``gated`` says which of them decide the
    verdict under the named profile. A component the profile leaves out can never produce
    a reason: its exceedance is recorded in ``ungated_failures`` instead. A gate with no
    gated component decides nothing, and its ``pass`` is null rather than a verdict.
    """
    selection = gate_profile_selection(gate_profile)
    results: dict[str, dict] = {}
    for gate, components in COMPOSITE_GATE_COMPONENTS.items():
        gated = selection.get(gate, ())
        marker = {"gated": bool(gated), "gated_components": list(gated)}
        values = composite_metrics.get(gate, {})
        if not hard_pass:
            results[gate] = {"pass": False, "evaluated": False,
                             "reasons": ["hard checks failed"],
                             "ungated_failures": [], **marker}
            continue
        readings = {component: component_value(values.get(component))
                    for component in components}
        nonfinite = [component for component in components
                     if not math.isfinite(readings[component])]
        gated_nonfinite = [component for component in nonfinite if component in gated]
        reported_nonfinite = [component for component in nonfinite
                              if component not in gated]
        ungated_failures = [_undefined_reason(gate, reported_nonfinite)] \
            if reported_nonfinite else []
        if gated_nonfinite:
            results[gate] = {"pass": False, "evaluated": True,
                             "reasons": [_undefined_reason(gate, gated_nonfinite)],
                             "ungated_failures": ungated_failures, **marker}
            continue
        if bars is None:
            results[gate] = {"pass": False, "evaluated": False,
                             "reasons": ["frozen bars not supplied"],
                             "ungated_failures": ungated_failures, **marker}
            continue
        failures = []
        frozen = bars["gates"][gate]["components"]
        for component in components:
            if component in nonfinite:
                continue
            value = readings[component]
            ceiling = float(frozen[component]["value"])
            if value > ceiling:
                detail = f"{component} {value:.6g} > {ceiling:.6g}"
                (failures if component in gated else ungated_failures).append(detail)
        results[gate] = {"pass": (not failures) if gated else None, "evaluated": True,
                         "reasons": failures, "ungated_failures": ungated_failures,
                         **marker}
    return results


def _failed_v4_report(reason: str, *, schema_errors: list[str] | None = None,
                      gate_profile: str = DEFAULT_GATE_PROFILE) -> dict:
    empty = {gate: {} for gate in COMPOSITE_GATE_COMPONENTS}
    return {"pass": False, "hard_pass": False, "reasons": [reason],
            "gate_profile": gate_profile,
            "schema_errors": list(schema_errors or []), "additivity_errors": [],
            "rate_errors": [], "reserve_errors": [], "metrics": {},
            "projection_metrics": {}, "rate_metrics": {},
            "composite_metrics": empty, "gate_results": evaluate_composite_gates(
                empty, None, False, gate_profile), "reserve": {"feasible": False},
            "reserve_q95_feasibility": {"valid": False},
            "reserve_tail_evidence": {
                "schema": RESERVE_TAIL_EVIDENCE_SCHEMA,
                "valid": False,
            },
            "reserve_rule_evidence": {"valid": False}, "reserve_rule_errors": []}


def verify_actuarial_submission(packet_dir: Path, submission_dir: Path,
                                bars: dict | None = None, alpha: float = 0.10,
                                thresholds: ActuarialThresholds | None = None,
                                gate_profile: str = DEFAULT_GATE_PROFILE) -> dict:
    """Score the exact three-file version-four surface.

    The release and projection tables carry the eight version-three estimands and the
    exposure and rate block. The reserve file replaces the point allocation: its
    feasibility requires finite nonnegative allocations that sum to the published total;
    its value reads the retained continuation ensemble, never one realized path. The
    submitted q95 allocation-floor comparison remains an authenticated diagnostic and is
    not a hard check. Schema, additivity, and feasibility are deterministic hard checks.
    The stochastic verdict has exactly five composite pass events.

    ``gate_profile`` names which of those five decide. The default profile decides on all
    of them. Every profile measures and reports all five, and the report names the profile
    the verdict came from. An unknown profile name raises rather than scoring.
    """
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    gate_profile_selection(gate_profile)
    try:
        contract_file = json.loads(
            (packet_dir / "participant" / "contract.json").read_text())
        reserve_contract = contract_file["reserve"]
        obligation = ObligationContract.from_public(reserve_contract["obligation"])
        admin = admin_from_packet(packet_dir)
    except Exception as exc:
        return _failed_v4_report(
            f"packet: cannot read public contract ({type(exc).__name__})",
            gate_profile=gate_profile)
    thresholds = thresholds or ActuarialThresholds()
    reserve_rule_evidence, reserve_rule_errors = _public_reserve_rule_evidence(
        packet_dir, contract_file)

    file_errors = _v4_file_errors(submission_dir)
    if file_errors:
        return _failed_v4_report(f"file set: {'; '.join(file_errors)}",
                                 gate_profile=gate_profile)
    header_errors = _v4_header_errors(submission_dir)
    contract_errors = _contract_submission_errors(packet_dir, contract_file, thresholds)
    if header_errors or contract_errors:
        errors = contract_errors + header_errors
        return _failed_v4_report(f"schema: {len(errors)} violation(s)",
                                 schema_errors=errors, gate_profile=gate_profile)

    try:
        evidence = _v4_evidence(packet_dir, submission_dir)
    except (OSError, ValueError) as exc:
        return _failed_v4_report(
            "evidence: cannot bind verifier inputs",
            schema_errors=[str(exc)],
            gate_profile=gate_profile,
        )

    retained = packet_dir / "retained"
    try:
        truth_now = load_truth(retained / "truth_revised.csv")
        truth_future = load_truth(retained / "truth_horizon.csv")
        rate_truth = load_rate_truth(retained / "rate_truth_horizon.csv")
        ensemble = load_continuation_ensemble(retained / "continuation_liabilities.npz")
        release_core, release_rates = load_release_blocks(submission_dir / "release.csv")
        projection_core, projection_rates = load_release_blocks(
            submission_dir / "projection.csv")
        reserve_rows = load_reserve_rows(submission_dir / "reserve.csv")
        shock_redraw_evidence = _continuation_shock_redraw_report(
            packet_dir, contract_file, evidence
        )
    except Exception as exc:
        return _failed_v4_report(f"schema: cannot parse a required file ({type(exc).__name__})",
                                 schema_errors=[str(exc)], gate_profile=gate_profile)

    schema_errors = validate_release(release_core, admin,
                                     extra_columns=RATE_EXTRA_COLUMNS,
                                     skip_estimands=RATE_ESTIMANDS)
    schema_errors.extend(
        f"release core row {index}: sex and age_band must be empty"
        for index, row in enumerate(release_core)
        if row.get("sex") or row.get("age_band")
    )
    additivity_errors = check_additivity(release_core, admin) if not schema_errors else []
    metrics = score_release(release_core, truth_now, admin, alpha)

    parsed_rates, rate_errors = parse_rate_rows(release_rates, admin)
    rate_errors = rate_errors + check_rate_additivity(parsed_rates, admin)
    rate_metrics = score_rates(parsed_rates, rate_truth, thresholds, alpha)

    if projection_rates:
        schema_errors.append("projection.csv contains exposure or rate rows")
    projection_schema = validate_release(projection_core, admin,
                                         extra_columns=RATE_EXTRA_COLUMNS,
                                         skip_estimands=RATE_ESTIMANDS)
    projection_additivity = check_additivity(projection_core, admin) \
        if not projection_schema else []
    projection_metrics = score_release(projection_core, truth_future, admin, alpha)

    parsed_reserve, reserve_errors = parse_reserve_rows(reserve_rows, ensemble.n_regions)
    elder_reference_evidence = None
    if not rate_errors and not reserve_errors:
        try:
            elder_reference_evidence = _elder_reference_evidence(
                parsed_rates,
                rate_truth,
                parsed_reserve,
                ensemble,
                int(admin["n_states"]),
                evidence,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rate_errors.append(
                "elder reference evidence cannot be recomputed "
                f"({type(exc).__name__})"
            )
    reserve = None
    if not reserve_errors:
        reserve = score_reserve(parsed_reserve["allocation"], parsed_reserve["q95"],
                                parsed_reserve["es95"], parsed_reserve["liability_mean"],
                                ensemble.liability, float(reserve_contract["total"]),
                                thresholds,
                                weights=np.asarray(reserve_contract.get("weights"),
                                                   dtype=np.float64)
                                if reserve_contract.get("weights") else None,
                                scale=np.asarray(reserve_contract["scale"], dtype=np.float64)
                                if reserve_contract.get("scale") else None,
                                baseline_share=np.asarray(
                                    reserve_contract["baseline_share"], dtype=np.float64)
                                if reserve_contract.get("baseline_share") else None)
    reserve_q95_feasibility = _reserve_q95_feasibility_evidence(
        parsed_reserve if not reserve_errors else None,
        reserve,
        float(reserve_contract["total"]),
        thresholds.feasibility_tolerance,
    )
    reserve_tail_evidence = _reserve_tail_evidence(
        parsed_reserve if not reserve_errors else None,
        evidence.get("submission_file_sha256", {}).get("reserve.csv"),
    )

    all_schema_errors = schema_errors + projection_schema
    all_additivity_errors = additivity_errors + projection_additivity
    bar_errors = _bar_schema_errors(bars)
    hard_reasons: list[str] = []
    if all_schema_errors:
        hard_reasons.append(f"schema: {len(all_schema_errors)} violation(s)")
    if all_additivity_errors:
        hard_reasons.append(f"additivity: {len(all_additivity_errors)} violation(s)")
    if rate_errors:
        hard_reasons.append(f"rate schema: {len(rate_errors)} violation(s)")
    if reserve_errors:
        hard_reasons.append(f"reserve schema: {len(reserve_errors)} violation(s)")
    if reserve is None:
        hard_reasons.append("reserve: no valid regional reserve was scored")
    elif not reserve["feasible"]:
        hard_reasons.append("reserve: infeasible (" + "; ".join(
            reserve["feasibility_reasons"]) + ")")
    if reserve_rule_errors:
        hard_reasons.append(
            f"public reserve rule: {len(reserve_rule_errors)} violation(s)")
    if bar_errors:
        hard_reasons.append(f"bars: {len(bar_errors)} schema violation(s)")
    # A receipt frozen under one profile cannot decide under another: the reader of a
    # verdict must see the same profile name on the bars and on the report.
    bar_profile = bars.get("gate_profile", DEFAULT_GATE_PROFILE) \
        if isinstance(bars, dict) else None
    if bars is not None and bar_profile != gate_profile:
        hard_reasons.append(
            f"bars: the receipt froze the {bar_profile!r} gate profile, "
            f"not {gate_profile!r}")

    composite_metrics = build_composite_metrics(
        metrics, projection_metrics, rate_metrics, reserve, alpha)
    hard_pass = not hard_reasons
    gate_results = evaluate_composite_gates(
        composite_metrics, bars if not bar_errors else None, hard_pass, gate_profile)
    gate_reasons = [f"{gate}: " + "; ".join(result["reasons"])
                    for gate, result in gate_results.items()
                    if result["evaluated"] and result["gated"] and not result["pass"]]
    reasons = hard_reasons + gate_reasons
    if bars is None:
        reasons.append("bars: no frozen composite bar receipt was supplied")
    report = {
        "pass": not reasons, "hard_pass": hard_pass, "reasons": reasons,
        "gate_profile": gate_profile,
        "schema_errors": all_schema_errors, "additivity_errors": all_additivity_errors,
        "rate_errors": rate_errors, "reserve_errors": reserve_errors,
        "metrics": metrics, "projection_metrics": projection_metrics,
        "rate_metrics": rate_metrics, "composite_metrics": composite_metrics,
        "gate_results": gate_results, "bar_schema_errors": bar_errors,
        "reserve": reserve if reserve is not None else {"feasible": False},
        "reserve_q95_feasibility": reserve_q95_feasibility,
        "reserve_tail_evidence": reserve_tail_evidence,
        "reserve_rule_evidence": reserve_rule_evidence,
        "reserve_rule_errors": reserve_rule_errors,
        "obligation": obligation.as_public(),
        "evidence": evidence,
        "elder_reference_evidence": elder_reference_evidence,
        "continuation_shock_redraw_evidence": shock_redraw_evidence,
        "eligibility_evidence": build_eligibility_evidence(rate_truth, thresholds),
    }
    return report


def verify_release_projection_allocation(packet_dir: Path, submission_dir: Path,
                                         bars: dict | None = None,
                                         alpha: float = 0.10) -> dict:
    """Verify the exact three-file population-reconstruction task surface."""
    return verify_submission(packet_dir, submission_dir, bars, alpha,
                             score_disclosure=False)


def summary_table(report: dict) -> str:
    """Plain-text summary of worst errors and coverage per estimand and level."""
    lines = []
    for block in ("metrics", "projection_metrics"):
        lines.append(block)
        for key, m in sorted(report[block].items()):
            lines.append(f"  {key:40s} worst {m['worst_error']:.4f}  mean {m['mean_error']:.4f}"
                         f"  coverage {m['coverage']:.2f}  iscore {m['mean_interval_score']:.4f}")
    decision = report.get("reserve", report.get("allocation", {}))
    lines.append(
        f"reserve feasible={decision.get('feasible', False)} "
        f"loss={decision.get('loss', float('nan')):.4f} "
        f"skill={component_value(decision.get('skill')):.4f}"
    )
    if "disclosure" in report:
        lines.append(f"disclosure pass={report['disclosure']['pass']} "
                     f"protected={report['disclosure']['n_protected']} "
                     f"suppressed={report['disclosure']['n_suppressed']}")
    if report.get("gate_profile"):
        lines.append(f"gate profile {report['gate_profile']}")
    lines.append("PASS" if report["pass"] else "FAIL: " + "; ".join(report["reasons"]))
    return "\n".join(lines)
