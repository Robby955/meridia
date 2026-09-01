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


def verify_submission(packet_dir: Path, submission_dir: Path, bars: dict | None = None,
                      alpha: float = 0.10) -> dict:
    packet_dir, submission_dir = Path(packet_dir), Path(submission_dir)
    contract = json.loads((packet_dir / "participant" / "contract.json").read_text())
    admin = admin_from_packet(packet_dir)
    retained = packet_dir / "retained"
    truth_now = load_truth(retained / "truth_revised.csv")
    truth_future = load_truth(retained / "truth_horizon.csv")
    detailed_truth = load_detailed(retained / "detailed_revised.csv", admin["n_counties"]).astype(np.int64)

    release = load_rows(submission_dir / "release.csv")
    schema_errors = validate_release(release, admin)
    additivity_errors = check_additivity(release, admin) if not schema_errors else []
    metrics = score_release(release, truth_now, admin, alpha)

    published = load_detailed(submission_dir / "detailed.csv", admin["n_counties"])
    disclosure = disclosure_audit(published, detailed_truth, int(contract["disclosure_threshold"]))

    projection = load_rows(submission_dir / "projection.csv")
    projection_schema = validate_release(projection, admin)
    projection_metrics = score_release(projection, truth_future, admin, alpha)

    allocation_frame = _read_csv(submission_dir / "allocation.csv").sort_values("county")
    allocation = np.zeros(admin["n_counties"])
    allocation[allocation_frame["county"].to_numpy(dtype=np.int64)] = \
        allocation_frame["allocation"].to_numpy(dtype=np.float64)
    demand = np.asarray([truth_future[(contract["allocation"]["demand"], "county", c)]
                         for c in range(admin["n_counties"])])
    allocation_score = score_allocation(allocation, demand, float(contract["allocation"]["budget"]))

    gates = evaluate_gates(schema_errors, additivity_errors, metrics, disclosure, bars)
    projection_gates = evaluate_gates(projection_schema, [], projection_metrics, None,
                                      (bars or {}).get("projection"))
    reasons = list(gates["reasons"]) + [f"projection {r}" for r in projection_gates["reasons"]]
    if not allocation_score["feasible"]:
        reasons.append("allocation: infeasible (negative, non-finite, or over budget)")
    ceiling = (bars or {}).get("allocation_regret_ceiling")
    if ceiling is not None and allocation_score["feasible"] and allocation_score["regret"] > ceiling:
        reasons.append(f"allocation: regret {allocation_score['regret']:.4f} > {ceiling}")
    return {
        "pass": not reasons, "reasons": reasons,
        "schema_errors": schema_errors, "additivity_errors": additivity_errors,
        "metrics": metrics, "disclosure": {k: v for k, v in disclosure.items()
                                           if k in ("pass", "n_protected", "n_suppressed",
                                                    "published_protected", "recoverable")},
        "projection_metrics": projection_metrics, "allocation": allocation_score,
    }


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
    lines.append(f"disclosure pass={report['disclosure']['pass']} "
                 f"protected={report['disclosure']['n_protected']} suppressed={report['disclosure']['n_suppressed']}")
    lines.append("PASS" if report["pass"] else "FAIL: " + "; ".join(report["reasons"]))
    return "\n".join(lines)
