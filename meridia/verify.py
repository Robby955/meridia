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

import hashlib
import json
import math
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
SUBMISSION_FILES = ("release.csv", "detailed.csv", "projection.csv", "allocation.csv")
OPTIONAL_FILES = ("totals.csv",)
TOTAL_KINDS = {"county_age": ("county", "age_band"), "county_sex": ("county", "sex"),
               "county": ("county",), "age_sex": ("age_band", "sex")}

COMPOSITE_BAR_SCHEMA = "meridia.v4.composite-bars.v1"
VERIFIER_EVIDENCE_SCHEMA = "meridia.v4.verifier-evidence.v1"
FREEZE_PROVENANCE_SCHEMA = "meridia.v4.freeze-provenance.v1"
QUALIFICATION_WORLD_NAMES = tuple(f"qual-{index}" for index in range(6))
REFERENCE_LINES = ("A", "B", "C")
REPLICATES_PER_LINE_WORLD = 7
REFERENCE_REPORT_COUNT = len(REFERENCE_LINES) * len(QUALIFICATION_WORLD_NAMES)
REPLICATE_REPORT_COUNT = REFERENCE_REPORT_COUNT * REPLICATES_PER_LINE_WORLD
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
RESERVE_CALIBRATION_SCHEMA = "meridia.reserve-rate-calibration.v1"
RESERVE_RED_TEAM_SCHEMA = "meridia.reserve-total-red-team.v1"
COMPOSITE_GATE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "exposures_and_rates": ("p95_relative_error",),
    "release_accuracy": ("p95_relative_error",),
    "interval_quality": ("coverage_deviation", "mean_interval_score"),
    "tail_calibration": ("pooled_exceedance_deviation",
                         "q95_width_relative_error", "es95_width_relative_error"),
    "reserve_skill": ("skill_loss",),
}
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
COMPOSITE_COMPONENT_RANGES: dict[tuple[str, str], tuple[float, float | None]] = {
    ("exposures_and_rates", "p95_relative_error"): (0.0, None),
    ("release_accuracy", "p95_relative_error"): (0.0, None),
    ("interval_quality", "coverage_deviation"): (0.0, 1.0),
    ("interval_quality", "mean_interval_score"): (0.0, None),
    ("tail_calibration", "pooled_exceedance_deviation"): (0.0, 0.95),
    ("tail_calibration", "q95_width_relative_error"): (0.0, None),
    ("tail_calibration", "es95_width_relative_error"): (0.0, None),
    ("reserve_skill", "skill_loss"): (0.0, None),
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
    """Record the submitted q95 floor check independently of the skill metric."""

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
                      allow_unfrozen: bool = False) -> dict:
    """Score one submission against one packet, dispatching on the packet's schema.

    ``allow_unfrozen`` is retained only for call compatibility and has no effect. A
    version-four verdict always requires a complete frozen receipt. Freeze measurements
    obtain ungated component metrics by omitting ``bars`` and reading ``hard_pass``.
    """
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    contract = json.loads((packet_dir / "participant" / "contract.json").read_text())
    if str(contract.get("schema", "")).startswith(ACTUARIAL_PACKET_SCHEMA):
        del allow_unfrozen
        if bars is not None and bars.get("frozen") is not True:
            return _failed_v4_report(
                "bars: this bar set does not record a completed freeze")
        return verify_actuarial_submission(packet_dir, submission_dir, bars, alpha)
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

    The archive carries ``liability`` of shape (members, regions) and the index of the one
    designated realized future. Nothing about the ensemble reaches the participant side.
    """
    with np.load(Path(path)) as archive:
        realized = int(archive["realized_member"]) if "realized_member" in archive else 0
        return ContinuationEnsemble(np.asarray(archive["liability"], dtype=np.float64),
                                    realized)


def load_reserve_rows(path: Path) -> list[dict]:
    frame = _read_csv(path)
    missing = [c for c in RESERVE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"reserve.csv is missing {missing}")
    return [{"region": r.region, "liability_mean": float(r.liability_mean),
             "q95": float(r.q95), "es95": float(r.es95),
             "allocation": float(r.allocation)} for r in frame.itertuples()]


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


def _contract_submission_errors(contract: dict) -> list[str]:
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
        "reserve_skill": {"skill_loss": max(0.0, 1.0 - skill) if math.isfinite(skill)
                          else float("nan")},
    }


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
                    "total_minus_q95_sum",
                )) \
                or value.get("all_regions_at_or_above_q95") is not True \
                or value.get("allocation_sums_to_total") is not True \
                or value.get("feasible") is not True:
            return False
        total = float(value["reserve_total"])
        tolerance = 1e-10 * max(1.0, abs(total))
        return abs(float(value["allocation_sum"]) - total) <= tolerance \
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
        if per_pair * len(lines) * (len(worlds) - 1) < 100:
            errors.append("leave-one-world-out p99 training sets need at least 100 reports")
        if bars.get("reference_report_count") != len(lines) * len(worlds):
            errors.append("final reference count does not match the balanced design")
    if bars.get("reference_report_count") != REFERENCE_REPORT_COUNT:
        errors.append("the exact eighteen final reference reports are required")
    if bars.get("paired_resamples_per_world") != REPLICATES_PER_LINE_WORLD \
            or bars.get("paired_resample_count") \
            != REPLICATES_PER_LINE_WORLD * len(QUALIFICATION_WORLD_NAMES):
        errors.append("the exact seven paired resamples per world are required")
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
    rates = bars.get("achieved_false_fail_rates")
    if not isinstance(rates, dict) or set(rates) != set(COMPOSITE_GATE_COMPONENTS):
        errors.append("achieved false-fail rates differ from the five gates")
        rates = {}
    else:
        for gate, value in rates.items():
            if not finite_number(value) or not 0.0 <= float(value) <= 0.01:
                errors.append(f"{gate}: achieved false-fail rate exceeds the target")
    target_product = 0.99 ** (len(COMPOSITE_GATE_COMPONENTS) * 3)
    if not finite_number(bars.get("target_marginal_product")) \
            or not math.isclose(float(bars["target_marginal_product"]), target_product,
                                rel_tol=1e-12, abs_tol=1e-15):
        errors.append("target marginal product differs from the registered design")
    achieved_product = math.prod(
        (1.0 - float(rates[gate])) ** 3 for gate in COMPOSITE_GATE_COMPONENTS
    ) if rates else None
    if achieved_product is None \
            or not finite_number(bars.get("achieved_marginal_rate_product")) \
            or not math.isclose(float(bars["achieved_marginal_rate_product"]),
                                achieved_product, rel_tol=1e-12, abs_tol=1e-15):
        errors.append("achieved marginal product does not match the gate rates")
    if bars.get("reference_failures") != []:
        errors.append("freeze receipt contains final reference failures")

    binding_base_keys = {
        "schema", "kind", "world", "method_digest_sha256",
        "runner_digest_sha256", "measurement_contract_digest_sha256",
        "run_receipt_digest_sha256", "packet_digest_sha256",
        "contract_digest_sha256", "submission_digest_sha256",
        "verifier_digest_sha256", "verifier_report_digest_sha256",
        "reserve_q95_feasibility_digest_sha256", "evidence_id",
    }

    def valid_binding(row: object, expected_kind: str) -> bool:
        if not isinstance(row, dict) or row.get("kind") != expected_kind \
                or row.get("schema") != "meridia.v4.freeze-evidence-binding.v1":
            return False
        expected_keys = set(binding_base_keys)
        identity_field: str
        if expected_kind == "reference":
            expected_keys.add("reference_line")
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
        )
        evidence_id = row.get("evidence_id")
        unsigned = dict(row)
        unsigned.pop("evidence_id", None)
        return set(row) == expected_keys \
            and all(is_sha256(row.get(field)) for field in digest_fields) \
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
        by_world: dict[str, list[dict]] = {}
        for row in all_bound_rows:
            by_world.setdefault(row["world"], []).append(row)
        for world, rows in by_world.items():
            if len({row["packet_digest_sha256"] for row in rows}) != 1 \
                    or len({row["contract_digest_sha256"] for row in rows}) != 1:
                errors.append(f"{world}: freeze evidence packet or contract binding differs")

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
                digest = rows[0]["resample_digest_sha256"]
                owner = (world, replicate_id)
                if digest in paired_digests and paired_digests[digest] != owner:
                    errors.append("a paired resample digest is reused across identifiers")
                paired_digests[digest] = owner

    def valid_signed_audit(value: object, schema: str) -> bool:
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
        and valid_signed_audit(qualification_audit, RESERVE_QUALIFICATION_SCHEMA) \
        and valid_signed_audit(calibration_audit, RESERVE_CALIBRATION_SCHEMA) \
        and valid_signed_audit(red_team_audit, RESERVE_RED_TEAM_SCHEMA)
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
            and float(calibration_audit["rate_grid"]) > 0.0 \
            and finite_number(calibration_audit.get("tail_slack_share")) \
            and 0.0 <= float(calibration_audit["tail_slack_share"]) <= 1.0 \
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
                and row.get("q95_feasible") is True
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
                and row.get("q95_feasible") is True
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
            for pair, result in qualification_by_pair.items():
                binding = reference_bindings[pair]
                calibration = calibration_by_pair[pair]
                feasibility = {
                    "q95_sum": result["q95_sum"],
                    "allocation_sum": result["allocation_sum"],
                    "reserve_total": result["reserve_total"],
                    "total_minus_q95_sum": result["total_minus_q95_sum"],
                    "all_regions_at_or_above_q95": True,
                    "allocation_sums_to_total": True,
                    "feasible": True,
                }
                q95 = calibration.get("submitted_q95_sum")
                es95 = calibration.get("submitted_es95_sum")
                candidate_total = calibration.get("candidate_reserve_total")
                candidate_margin = calibration.get("candidate_margin")
                numeric = all(finite_number(value) for value in (
                    q95, es95, candidate_total, candidate_margin
                ))
                if not numeric:
                    reserve_audits_ok = False
                    break
                target = float(q95) + slack * (float(es95) - float(q95))
                if result["evidence_id"] != binding["evidence_id"] \
                        or calibration.get("evidence_id") != binding["evidence_id"] \
                        or not valid_q95_feasibility(feasibility) \
                        or canonical_digest(feasibility) != binding[
                            "reserve_q95_feasibility_digest_sha256"
                        ] \
                        or not math.isclose(
                            float(result["q95_sum"]), float(q95),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or not math.isclose(
                            float(result["reserve_total"]), float(candidate_total),
                            rel_tol=1e-12, abs_tol=1e-9,
                        ) \
                        or float(es95) < float(q95) \
                        or float(candidate_total) < float(q95) \
                        or float(candidate_margin) < 0.0 \
                        or not math.isclose(
                            float(candidate_margin), float(candidate_total) - target,
                            rel_tol=1e-12, abs_tol=1e-9,
                        ):
                    reserve_audits_ok = False
                    break
        if reserve_audits_ok:
            for pair, result in proportional_by_pair.items():
                binding = proportional_bindings[pair]
                feasibility = {
                    "q95_sum": result["q95_sum"],
                    "allocation_sum": result["allocation_sum"],
                    "reserve_total": result["reserve_total"],
                    "total_minus_q95_sum": result["total_minus_q95_sum"],
                    "all_regions_at_or_above_q95": True,
                    "allocation_sums_to_total": True,
                    "feasible": True,
                }
                if result["evidence_id"] != binding["evidence_id"] \
                        or not valid_q95_feasibility(feasibility) \
                        or canonical_digest(feasibility) != binding[
                            "reserve_q95_feasibility_digest_sha256"
                        ]:
                    reserve_audits_ok = False
                    break
    if reserve_audits_ok:
        quantities = red_team_audit.get("public_quantities")
        primary = red_team_audit.get(
            "qualification_incremental_regional_r2_over_region_means"
        )
        development_quantities = quantities.get("development") \
            if isinstance(quantities, dict) else None
        qualification_quantities = quantities.get("qualification") \
            if isinstance(quantities, dict) else None
        reserve_audits_ok = isinstance(quantities, dict) \
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
                and finite_number(row.get("latest_year_total_exposure"))
                and float(row["latest_year_total_exposure"]) >= 0.0
                and finite_number(row.get("reserve_total"))
                and float(row["reserve_total"]) >= 0.0
                for row in [*development_quantities, *qualification_quantities]
            ) \
            and isinstance(primary, dict) \
            and all(finite_number(primary.get(field))
                    for field in ("q95", "es95", "headline_max")) \
            and math.isclose(
                float(primary["headline_max"]),
                max(float(primary["q95"]), float(primary["es95"])),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
    if not reserve_audits_ok:
        errors.append("reserve qualification, calibration, or red-team audit is invalid")

    identification = bars.get("mortality_identification_evidence")
    identification_ok = isinstance(identification, dict) \
        and identification.get("schema") == "meridia.v4.mortality-identification.v1" \
        and identification.get("supports_gate") == "tail_calibration" \
        and is_sha256(identification.get("generator_source_digest_sha256")) \
        and is_sha256(identification.get("diagnostic_digest_sha256"))
    if identification_ok:
        unsigned = dict(identification)
        recorded_digest = unsigned.pop("diagnostic_digest_sha256")
        trend = unsigned.get("trend", {})
        lag = unsigned.get("publication_lag", {})
        shock = unsigned.get("shock_process", {})
        identification_ok = canonical_digest(unsigned) == recorded_digest \
            and trend.get("active_during_public_experience_window") is True \
            and trend.get("starts_only_after_publication") is False \
            and lag.get("months") == 12 \
            and lag.get("trend_effect_percent_range") == [-7.24, 1.6] \
            and shock.get("annual_probability") == 0.20 \
            and shock.get("expected_mortality_spike_years_per_five_year_horizon") == 0.333 \
            and shock.get("redrawn_independently_in_every_continuation") is True \
            and set(unsigned.get("per_world", {})) == set(QUALIFICATION_WORLD_NAMES)
    if not identification_ok:
        errors.append("mortality identification evidence for the tail gate is invalid")

    regime_audit = bars.get("regime_identifiability_audit")
    regime_ok = isinstance(regime_audit, dict) \
        and regime_audit.get("schema") == "meridia.v4.regime-identifiability-audit.v1" \
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
            for axis in REGIME_AXES:
                record = axes[axis]
                signed = record.get("signed_rank_correlation") \
                    if isinstance(record, dict) else None
                within = record.get("within_regime_signed_rank_correlation") \
                    if isinstance(record, dict) else None
                observed = record.get("intensity_range_observed") \
                    if isinstance(record, dict) else None
                qualified = finite_number(signed) and float(signed) > 0.4
                base_ok = isinstance(record, dict) \
                    and isinstance(record.get("statistic"), str) \
                    and bool(record["statistic"]) \
                    and record.get("expected_sign") == REGIME_EXPECTED_SIGNS[axis] \
                    and finite_number(signed) and -1.0 <= float(signed) <= 1.0 \
                    and isinstance(within, dict) \
                    and set(within) == {"development", "hidden"} \
                    and all(finite_number(value) and -1.0 <= float(value) <= 1.0
                            for value in within.values()) \
                    and isinstance(observed, list) and len(observed) == 2 \
                    and all(finite_number(value) for value in observed) \
                    and float(observed[0]) <= float(observed[1]) \
                    and record.get("anchor_correlation_qualified") is qualified \
                    and record.get("development_range") == DEVELOPMENT_AXIS_RANGES[axis]
                if axis in HIDDEN_IN_BAND_AXES:
                    low, high = DEVELOPMENT_AXIS_RANGES[axis]
                    base_ok = base_ok \
                        and record.get("disposition") == "constrained_to_development_range" \
                        and record.get("hidden_out_of_band_allowed") is False \
                        and record.get("hidden_generation_range") == [low, high] \
                        and low <= float(observed[0]) <= float(observed[1]) <= high
                else:
                    base_ok = base_ok and qualified \
                        and record.get("disposition") == "participant_anchor" \
                        and record.get("hidden_out_of_band_allowed") is True \
                        and record.get("hidden_generation_range") == PUBLIC_AXIS_RANGES[axis]
                if not base_ok:
                    regime_ok = False
                    break
    if not regime_ok:
        errors.append("regime identifiability and hidden-axis constraint evidence is invalid")

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
            and method.get("before_line") in lines \
            and method.get("after_line") in lines \
            and method.get("before_line") != method.get("after_line") \
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
            after_values = []
            for row in audit_worlds:
                exposure = row.get("exposure_65_plus_absolute_error_percent", {})
                if not isinstance(exposure, dict) or not finite_number(exposure.get("after")) \
                        or not is_sha256(row.get("before_report_evidence_id")) \
                        or not is_sha256(row.get("after_report_evidence_id")):
                    elder_ok = False
                    break
                after_values.append(float(exposure["after"]))
            if elder_ok:
                after_values.sort()
                elder_ok = 0.5 * (after_values[2] + after_values[3]) < 10.0
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

    gate_detail = bars.get("leave_one_world_out_gate_results")
    if not isinstance(gate_detail, dict) \
            or set(gate_detail) != set(COMPOSITE_GATE_COMPONENTS):
        errors.append("gate-union leave-one-world-out results are incomplete")
        gate_detail = {}
    elif worlds and lines and isinstance(per_pair, int):
        held_out_size = per_pair * len(lines)
        training_size = per_pair * len(lines) * (len(worlds) - 1)
        for gate, rate in rates.items():
            records = gate_detail.get(gate)
            if not isinstance(records, dict) or set(records) != set(worlds):
                errors.append(f"{gate}: held-out gate results are incomplete")
                continue
            failure_count = 0
            valid = True
            for world in worlds:
                row = records[world]
                count = row.get("false_fail_count") if isinstance(row, dict) else None
                observed_rate = row.get("false_fail_rate") if isinstance(row, dict) else None
                if not isinstance(row, dict) \
                        or row.get("training_sample_count") != training_size \
                        or row.get("test_sample_count") != held_out_size \
                        or isinstance(count, bool) or not isinstance(count, int) \
                        or not 0 <= count <= held_out_size \
                        or not finite_number(observed_rate) \
                        or not math.isclose(float(observed_rate), count / held_out_size,
                                            rel_tol=1e-12, abs_tol=1e-15):
                    valid = False
                    break
                failure_count += count
            if not valid or not math.isclose(
                    float(rate), failure_count / report_count,
                    rel_tol=1e-12, abs_tol=1e-15):
                errors.append(f"{gate}: gate-union false-fail receipt is inconsistent")

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
    for gate, expected in COMPOSITE_GATE_COMPONENTS.items():
        components = gates[gate].get("components") if isinstance(gates[gate], dict) else None
        if not isinstance(components, dict) or set(components) != set(expected):
            errors.append(f"{gate}: component names differ from the verifier")
            continue
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
            if record.get("quantile") != 0.99 \
                    or record.get("target_false_fail_rate") != 0.01:
                errors.append(f"{gate}/{component}: freeze target metadata differs")
            sample_count = record.get("sample_count")
            rank = record.get("order_statistic_rank")
            if isinstance(sample_count, bool) or not isinstance(sample_count, int) \
                    or sample_count != report_count \
                    or rank != math.ceil(0.99 * sample_count):
                errors.append(f"{gate}/{component}: order-statistic receipt differs")
            if worlds and record.get("worlds") != worlds:
                errors.append(f"{gate}/{component}: qualification worlds differ")
            if lines and record.get("witnesses") != lines:
                errors.append(f"{gate}/{component}: reference witnesses differ")
            component_rate = record.get("achieved_false_fail_rate")
            if not finite_number(component_rate) \
                    or not 0.0 <= float(component_rate) <= 0.01:
                errors.append(f"{gate}/{component}: achieved false-fail rate exceeds target")
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
            quantile_witnesses = record.get("quantile_witnesses")
            quantile_identities_valid = isinstance(quantile_witnesses, list) and all(
                isinstance(witness, dict)
                and isinstance(witness.get("reference_line"), str)
                and isinstance(witness.get("world"), str)
                and isinstance(witness.get("replicate_id"), str)
                for witness in quantile_witnesses
            )
            if not isinstance(quantile_witnesses, list) or not quantile_witnesses \
                    or not quantile_identities_valid \
                    or len({
                        (
                            witness.get("reference_line"),
                            witness.get("world"),
                            witness.get("replicate_id"),
                        )
                        for witness in quantile_witnesses
                        if isinstance(witness, dict)
                    }) != len(quantile_witnesses) \
                    or any(not isinstance(witness, dict)
                           or witness.get("evidence_id") \
                           != replicate_binding_index.get((
                               witness.get("reference_line"),
                               witness.get("world"),
                               witness.get("replicate_id"),
                           ), {}).get("evidence_id")
                           or not finite_number(witness.get("value"))
                           or not math.isclose(float(witness["value"]), float(value),
                                               rel_tol=0.0, abs_tol=0.0)
                           for witness in quantile_witnesses):
                errors.append(f"{gate}/{component}: p99 witness receipt is invalid")
            component_detail = record.get("leave_one_world_out")
            if not isinstance(component_detail, dict) \
                    or set(component_detail) != set(worlds):
                errors.append(f"{gate}/{component}: held-out component results are incomplete")
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


def evaluate_composite_gates(composite_metrics: dict, bars: dict | None,
                             hard_pass: bool) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for gate, components in COMPOSITE_GATE_COMPONENTS.items():
        values = composite_metrics.get(gate, {})
        if not hard_pass:
            results[gate] = {"pass": False, "evaluated": False,
                             "reasons": ["hard checks failed"]}
            continue
        nonfinite = [component for component in components
                     if not math.isfinite(float(values.get(component, float("nan"))))]
        if nonfinite:
            results[gate] = {"pass": False, "evaluated": True,
                             "reasons": [f"non-finite components {nonfinite}"]}
            continue
        if bars is None:
            results[gate] = {"pass": False, "evaluated": False,
                             "reasons": ["frozen bars not supplied"]}
            continue
        failures = []
        frozen = bars["gates"][gate]["components"]
        for component in components:
            value = float(values[component])
            ceiling = float(frozen[component]["value"])
            if value > ceiling:
                failures.append(f"{component} {value:.6g} > {ceiling:.6g}")
        results[gate] = {"pass": not failures, "evaluated": True,
                         "reasons": failures}
    return results


def _failed_v4_report(reason: str, *, schema_errors: list[str] | None = None) -> dict:
    empty = {gate: {} for gate in COMPOSITE_GATE_COMPONENTS}
    return {"pass": False, "hard_pass": False, "reasons": [reason],
            "schema_errors": list(schema_errors or []), "additivity_errors": [],
            "rate_errors": [], "reserve_errors": [], "metrics": {},
            "projection_metrics": {}, "rate_metrics": {},
            "composite_metrics": empty, "gate_results": evaluate_composite_gates(
                empty, None, False), "reserve": {"feasible": False},
            "reserve_q95_feasibility": {"valid": False},
            "reserve_rule_evidence": {"valid": False}, "reserve_rule_errors": []}


def verify_actuarial_submission(packet_dir: Path, submission_dir: Path,
                                bars: dict | None = None, alpha: float = 0.10,
                                thresholds: ActuarialThresholds | None = None) -> dict:
    """Score the exact three-file version-four surface.

    The release and projection tables carry the eight version-three estimands and the
    exposure and rate block. The reserve file replaces the point allocation: its
    feasibility reads the submission's own quantiles and the published total, and its
    value reads the retained continuation ensemble, never one realized path. Schema,
    additivity, and feasibility are deterministic hard checks. The stochastic verdict has
    exactly five composite pass events.
    """
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    try:
        contract_file = json.loads(
            (packet_dir / "participant" / "contract.json").read_text())
        reserve_contract = contract_file["reserve"]
        obligation = ObligationContract.from_public(reserve_contract["obligation"])
        admin = admin_from_packet(packet_dir)
    except Exception as exc:
        return _failed_v4_report(f"packet: cannot read public contract ({type(exc).__name__})")
    thresholds = thresholds or ActuarialThresholds()
    reserve_rule_evidence, reserve_rule_errors = _public_reserve_rule_evidence(
        packet_dir, contract_file)

    file_errors = _v4_file_errors(submission_dir)
    if file_errors:
        return _failed_v4_report(f"file set: {'; '.join(file_errors)}")
    header_errors = _v4_header_errors(submission_dir)
    contract_errors = _contract_submission_errors(contract_file)
    if header_errors or contract_errors:
        errors = contract_errors + header_errors
        return _failed_v4_report(f"schema: {len(errors)} violation(s)",
                                 schema_errors=errors)

    try:
        evidence = _v4_evidence(packet_dir, submission_dir)
    except (OSError, ValueError) as exc:
        return _failed_v4_report(
            "evidence: cannot bind verifier inputs",
            schema_errors=[str(exc)],
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
    except Exception as exc:
        return _failed_v4_report(f"schema: cannot parse a required file ({type(exc).__name__})",
                                 schema_errors=[str(exc)])

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

    composite_metrics = build_composite_metrics(
        metrics, projection_metrics, rate_metrics, reserve, alpha)
    hard_pass = not hard_reasons
    gate_results = evaluate_composite_gates(
        composite_metrics, bars if not bar_errors else None, hard_pass)
    gate_reasons = [f"{gate}: " + "; ".join(result["reasons"])
                    for gate, result in gate_results.items()
                    if result["evaluated"] and not result["pass"]]
    reasons = hard_reasons + gate_reasons
    if bars is None:
        reasons.append("bars: no frozen composite bar receipt was supplied")
    report = {
        "pass": not reasons, "hard_pass": hard_pass, "reasons": reasons,
        "schema_errors": all_schema_errors, "additivity_errors": all_additivity_errors,
        "rate_errors": rate_errors, "reserve_errors": reserve_errors,
        "metrics": metrics, "projection_metrics": projection_metrics,
        "rate_metrics": rate_metrics, "composite_metrics": composite_metrics,
        "gate_results": gate_results, "bar_schema_errors": bar_errors,
        "reserve": reserve if reserve is not None else {"feasible": False},
        "reserve_q95_feasibility": reserve_q95_feasibility,
        "reserve_rule_evidence": reserve_rule_evidence,
        "reserve_rule_errors": reserve_rule_errors,
        "obligation": obligation.as_public(),
        "evidence": evidence,
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
        f"skill={decision.get('skill', float('nan')):.4f}"
    )
    if "disclosure" in report:
        lines.append(f"disclosure pass={report['disclosure']['pass']} "
                     f"protected={report['disclosure']['n_protected']} "
                     f"suppressed={report['disclosure']['n_suppressed']}")
    lines.append("PASS" if report["pass"] else "FAIL: " + "; ".join(report["reasons"]))
    return "\n".join(lines)
