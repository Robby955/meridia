"""Verifier: score a submission directory against a packet's retained truth.

A submission is four flat files in one directory:

- ``release.csv``: estimand, level, unit, estimate, lower, upper (the revised snapshot);
- ``detailed.csv``: county, age_band, sex, count (blank where suppressed);
- ``projection.csv``: same columns as the release, for the horizon tick;
- ``allocation.csv``: county, allocation.

The verifier reads the submitted numbers and the retained truth only. It never inspects
the method. Bars are optional; with none it reports metrics and the hard gates (schema,
additivity, disclosure, feasibility).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .actuarial import (RATE_ESTIMANDS, RATE_EXTRA_COLUMNS, RESERVE_COLUMNS,
                        ActuarialThresholds, ContinuationEnsemble, ObligationContract,
                        check_rate_additivity, evaluate_actuarial_gates, parse_rate_rows,
                        parse_reserve_rows, score_rates, score_reserve)
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
        rows.append({"estimand": str(r.estimand), "level": str(r.level), "unit": int(r.unit),
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
# Version four keeps the file count at four and replaces the point allocation with the
# reserve file, whose feasibility is checked against the submission's own quantiles.
ACTUARIAL_SUBMISSION_FILES = ("release.csv", "detailed.csv", "projection.csv", "reserve.csv")
ACTUARIAL_PACKET_SCHEMA = "meridia.packet.v4"
SUBMISSION_FILES = ("release.csv", "detailed.csv", "projection.csv", "allocation.csv")
OPTIONAL_FILES = ("totals.csv",)
TOTAL_KINDS = {"county_age": ("county", "age_band"), "county_sex": ("county", "sex"),
               "county": ("county",), "age_sex": ("age_band", "sex")}


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

    ``allow_unfrozen`` is for the freeze run itself, which has to score candidate bars
    before any verdict exists. Everywhere else a version-four bar file that does not say
    it is frozen refuses to gate, so an unfinished freeze cannot be read later as a
    finished one.
    """
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    contract = json.loads((packet_dir / "participant" / "contract.json").read_text())
    if str(contract.get("schema", "")).startswith(ACTUARIAL_PACKET_SCHEMA):
        if bars is not None and not allow_unfrozen and bars.get("frozen") is not True:
            return {"pass": False,
                    "reasons": ["bars: this bar set does not record a completed freeze"],
                    "schema_errors": [], "additivity_errors": [], "metrics": {},
                    "projection_metrics": {}, "rate_metrics": {},
                    "reserve": {"feasible": False},
                    "disclosure": {"pass": False, "n_protected": 0, "n_suppressed": 0,
                                   "published_protected": [], "recoverable": [],
                                   "utility": 0.0}}
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
                                         "published_protected", "recoverable")}
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
        row = {"estimand": str(r.estimand), "level": str(r.level), "unit": int(r.unit),
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
    return [{"region": int(r.region), "liability_mean": float(r.liability_mean),
             "q95": float(r.q95), "es95": float(r.es95),
             "allocation": float(r.allocation)} for r in frame.itertuples()]


def verify_actuarial_submission(packet_dir: Path, submission_dir: Path,
                                bars: dict | None = None, alpha: float = 0.10,
                                thresholds: ActuarialThresholds | None = None) -> dict:
    """Score the four version-four files against the retained truth and the ensemble.

    The release and projection tables carry the eight version-three estimands and the
    exposure and rate block. The reserve file replaces the point allocation: its
    feasibility reads the submission's own quantiles and the published total, and its
    value reads the sealed continuation ensemble, never one realized path.
    """
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    contract_file = json.loads((packet_dir / "participant" / "contract.json").read_text())
    reserve_contract = contract_file["reserve"]
    obligation = ObligationContract.from_public(reserve_contract["obligation"])
    thresholds = thresholds or ActuarialThresholds(
        gamma=float(reserve_contract.get("gamma", ActuarialThresholds().gamma)))
    admin = admin_from_packet(packet_dir)

    present = sorted(p.name for p in submission_dir.iterdir() if p.is_file())
    optional = OPTIONAL_FILES
    unexpected = [n for n in present if n not in ACTUARIAL_SUBMISSION_FILES + optional]
    missing = [n for n in ACTUARIAL_SUBMISSION_FILES if n not in present]
    if unexpected or missing:
        return {"pass": False,
                "reasons": [f"file set: unexpected {unexpected}, missing {missing}"],
                "schema_errors": [], "additivity_errors": [], "metrics": {},
                "projection_metrics": {}, "rate_metrics": {}, "reserve": {"feasible": False},
                "disclosure": {"pass": False, "n_protected": 0, "n_suppressed": 0,
                               "published_protected": [], "recoverable": []}}

    retained = packet_dir / "retained"
    truth_now = load_truth(retained / "truth_revised.csv")
    truth_future = load_truth(retained / "truth_horizon.csv")
    rate_truth = load_rate_truth(retained / "rate_truth_horizon.csv")
    ensemble = load_continuation_ensemble(retained / "continuation_liabilities.npz")

    release_core, release_rates = load_release_blocks(submission_dir / "release.csv")
    schema_errors = validate_release(release_core, admin,
                                     extra_columns=RATE_EXTRA_COLUMNS,
                                     skip_estimands=RATE_ESTIMANDS)
    additivity_errors = check_additivity(release_core, admin) if not schema_errors else []
    metrics = score_release(release_core, truth_now, admin, alpha)

    parsed_rates, rate_errors = parse_rate_rows(release_rates, admin)
    rate_errors = rate_errors + check_rate_additivity(parsed_rates, admin)
    rate_metrics = score_rates(parsed_rates, rate_truth, thresholds, alpha)

    detailed_truth = load_detailed(retained / "detailed_revised.csv",
                                   admin["n_counties"]).astype(np.int64)
    published = load_detailed(submission_dir / "detailed.csv", admin["n_counties"])
    marginals = load_totals(submission_dir / "totals.csv", admin["n_counties"]) \
        if (submission_dir / "totals.csv").exists() else None
    disclosure = disclosure_audit(published, detailed_truth,
                                  int(contract_file["disclosure_threshold"]), marginals)

    projection_core, _ = load_release_blocks(submission_dir / "projection.csv")
    projection_schema = validate_release(projection_core, admin,
                                         extra_columns=RATE_EXTRA_COLUMNS,
                                         skip_estimands=RATE_ESTIMANDS)
    projection_metrics = score_release(projection_core, truth_future, admin, alpha)

    reserve_rows = load_reserve_rows(submission_dir / "reserve.csv")
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

    gates = evaluate_gates(schema_errors, additivity_errors, metrics, disclosure, bars)
    projection_gates = evaluate_gates(projection_schema, [], projection_metrics, None,
                                      (bars or {}).get("projection"))
    actuarial = evaluate_actuarial_gates(rate_errors, rate_metrics, reserve_errors, reserve,
                                         thresholds, (bars or {}).get("actuarial"))
    reasons = list(gates["reasons"]) + [f"projection {r}" for r in projection_gates["reasons"]] \
        + list(actuarial["reasons"])
    report = {
        "pass": not reasons, "reasons": reasons,
        "schema_errors": schema_errors, "additivity_errors": additivity_errors,
        "rate_errors": rate_errors, "reserve_errors": reserve_errors,
        "metrics": metrics, "projection_metrics": projection_metrics,
        "rate_metrics": rate_metrics,
        "reserve": reserve if reserve is not None else {"feasible": False},
        "obligation": obligation.as_public(),
        "disclosure": {k: v for k, v in disclosure.items()
                       if k in ("pass", "n_protected", "n_suppressed",
                                "published_protected", "recoverable",
                                "n_releasable", "n_published_releasable", "utility")},
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
    a = report["allocation"]
    lines.append(f"allocation feasible={a['feasible']} loss={a.get('loss', float('nan')):.4f} "
                 f"regret={a.get('regret', float('nan')):.4f}")
    if "disclosure" in report:
        lines.append(f"disclosure pass={report['disclosure']['pass']} "
                     f"protected={report['disclosure']['n_protected']} "
                     f"suppressed={report['disclosure']['n_suppressed']}")
    lines.append("PASS" if report["pass"] else "FAIL: " + "; ".join(report["reasons"]))
    return "\n".join(lines)
