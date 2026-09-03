"""Select the public reserve rate from qualification reference submissions.

The input manifest names one participant packet and one reference submission for every
reference-line and qualification-world pair. Only participant experience and submitted
q95 and ES95 values are read. The selected rate is the smallest point on the registered
rate grid whose public reserve total covers every submitted q95 plus one quarter of the
submitted q95-to-ES95 spread.

This script returns a candidate, not a completed freeze. The references must be rerun at
that candidate rate, the reserve-skill gate must pass, the proportional control must fail,
and the held-out reserve-total red-team measurement must be recorded before the rate is
accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.actuarial import RESERVE_COLUMNS, reserve_total


SCHEMA = "meridia.reserve-rate-calibration.v1"
MANIFEST_SCHEMA = "meridia.reserve-rate-evidence.v1"
QUALIFICATION_WORLD_NAMES = tuple(f"qual-{index}" for index in range(6))
TAIL_SLACK_SHARE = 0.25
RATE_GRID = 1.0
PENDING_BLOCKERS = (
    "rerun every reference at the candidate rate and clear reserve skill",
    "show the proportional reserve control failing at the candidate rate",
    "record the held-out reserve-total red-team measurement",
)
EXPERIENCE_COLUMNS = (
    "year", "age_band", "sex", "state", "exposure", "deaths",
    "qualifying_events", "net_migration",
)
FORBIDDEN_PATH_FRAGMENTS = ("graded", "sealed", "hidden")


class CalibrationError(ValueError):
    """Evidence is missing, unsafe, duplicated, or numerically invalid."""


def _safe_path(path: Path) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve(strict=False)
    for part in (*candidate.absolute().parts, *resolved.parts):
        lowered = part.casefold()
        if any(fragment in lowered for fragment in FORBIDDEN_PATH_FRAGMENTS):
            raise CalibrationError("an evidence path contains a forbidden component")
    return resolved


def _regular_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink():
        raise CalibrationError(f"{root.name}: {relative} must not be a symbolic link")
    path = _safe_path(path)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CalibrationError(f"{root.name}: {relative} leaves its declared root") from exc
    if not path.is_file():
        raise CalibrationError(f"{root.name}: missing {relative}")
    return path


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise CalibrationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise CalibrationError(f"{label} must be positive and finite" if positive
                               else f"{label} must be finite")
    return result


def _public_exposure(packet: Path) -> tuple[float, float, int, str]:
    contract_path = _regular_file(packet, "participant/contract.json")
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"{packet.name}: contract.json is invalid") from exc
    if not isinstance(contract, dict) or contract.get("schema") != "meridia.packet.v4":
        raise CalibrationError(f"{packet.name}: packet schema is not version four")
    n_states_value = contract.get("n_states")
    if isinstance(n_states_value, bool) or not isinstance(n_states_value, int) \
            or n_states_value <= 0:
        raise CalibrationError(f"{packet.name}: n_states must be a positive integer")
    reserve = contract.get("reserve")
    if not isinstance(reserve, dict):
        raise CalibrationError(f"{packet.name}: reserve contract is missing")
    rounding_unit = _number(
        reserve.get("rounding_unit"), f"{packet.name}: reserve rounding unit", positive=True)
    experience = contract.get("experience_history")
    if not isinstance(experience, dict) \
            or experience.get("file") != "experience_history.csv" \
            or tuple(experience.get("columns", ())) != EXPERIENCE_COLUMNS:
        raise CalibrationError(f"{packet.name}: experience contract differs")
    path = _regular_file(packet, "participant/experience_history.csv")
    rows: list[tuple[int, float]] = []
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPERIENCE_COLUMNS:
                raise CalibrationError(f"{packet.name}: experience columns differ")
            for row in reader:
                year_value = _number(row["year"], f"{packet.name}: experience year")
                if not year_value.is_integer():
                    raise CalibrationError(f"{packet.name}: experience year is not an integer")
                exposure = _number(
                    row["exposure"], f"{packet.name}: experience exposure")
                if exposure < 0.0:
                    raise CalibrationError(f"{packet.name}: experience exposure is negative")
                rows.append((int(year_value), exposure))
    except CalibrationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CalibrationError(f"{packet.name}: experience file cannot be read") from exc
    if not rows:
        raise CalibrationError(f"{packet.name}: experience file is empty")
    latest = max(year for year, _ in rows)
    exposure = float(sum(value for year, value in rows if year == latest))
    if exposure <= 0.0 or not math.isfinite(exposure):
        raise CalibrationError(f"{packet.name}: latest exposure is not positive and finite")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return exposure, rounding_unit, n_states_value, digest


def _submitted_tail(submission: Path, n_states: int) -> tuple[float, float, str]:
    path = _regular_file(submission, "reserve.csv")
    q_sum = 0.0
    es_sum = 0.0
    seen: set[int] = set()
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RESERVE_COLUMNS:
                raise CalibrationError(f"{submission.name}: reserve columns differ")
            for row in reader:
                region_value = _number(row["region"], f"{submission.name}: region")
                if not region_value.is_integer() or int(region_value) < 0 \
                        or int(region_value) in seen:
                    raise CalibrationError(f"{submission.name}: regions are invalid")
                seen.add(int(region_value))
                q95 = _number(row["q95"], f"{submission.name}: q95")
                es95 = _number(row["es95"], f"{submission.name}: ES95")
                if q95 < 0.0 or es95 < q95:
                    raise CalibrationError(f"{submission.name}: submitted tails are invalid")
                q_sum += q95
                es_sum += es95
    except CalibrationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CalibrationError(f"{submission.name}: reserve file cannot be read") from exc
    expected_regions = set(range(n_states))
    if seen != expected_regions:
        raise CalibrationError(
            f"{submission.name}: regions differ from 0 through {n_states - 1}"
        )
    return q_sum, es_sum, hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(entries: Sequence[Mapping[str, Any]], *,
              expected_worlds: Sequence[str] = QUALIFICATION_WORLD_NAMES,
              tail_slack_share: float = TAIL_SLACK_SHARE,
              rate_grid: float = RATE_GRID) -> dict[str, Any]:
    """Return a fail-closed candidate-rate record from legal reference outputs."""
    try:
        slack = _number(tail_slack_share, "tail slack share")
        grid = _number(rate_grid, "rate grid", positive=True)
        if not 0.0 <= slack <= 1.0:
            raise CalibrationError("tail slack share must lie in [0, 1]")
        normalized: list[dict[str, Any]] = []
        pairs: defaultdict[tuple[str, str], int] = defaultdict(int)
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("deterministic") is not True:
                raise CalibrationError("every evidence entry must be deterministic")
            line = str(entry.get("reference_line", "")).strip()
            world = str(entry.get("world", "")).strip()
            evidence_id = str(entry.get("evidence_id", "")).strip()
            if not line or not world or not evidence_id:
                raise CalibrationError("reference line, world, and evidence id are required")
            packet = _safe_path(Path(str(entry.get("packet_dir", ""))))
            submission = _safe_path(Path(str(entry.get("submission_dir", ""))))
            if packet.name != world:
                raise CalibrationError(f"{evidence_id}: packet name differs from world")
            exposure, rounding_unit, n_states, experience_digest = _public_exposure(packet)
            q_sum, es_sum, reserve_digest = _submitted_tail(submission, n_states)
            target = q_sum + slack * (es_sum - q_sum)
            required_rate = target / exposure
            normalized.append({
                "reference_line": line,
                "world": world,
                "evidence_id": evidence_id,
                "exposure_person_years": exposure,
                "rounding_unit": rounding_unit,
                "submitted_q95_sum": q_sum,
                "submitted_es95_sum": es_sum,
                "target_reserve_before_rounding": target,
                "required_rate": required_rate,
                "experience_sha256": experience_digest,
                "reserve_submission_sha256": reserve_digest,
            })
            pairs[(line, world)] += 1
        lines = sorted({row["reference_line"] for row in normalized})
        worlds = sorted({row["world"] for row in normalized})
        expected = {(line, world) for line in lines for world in expected_worlds}
        if len(lines) < 3:
            raise CalibrationError("at least three reference lines are required")
        if worlds != sorted(expected_worlds):
            raise CalibrationError("qualification worlds differ from the registered set")
        if set(pairs) != expected or any(count != 1 for count in pairs.values()):
            raise CalibrationError("exactly one entry is required for every line-world pair")
        units = {row["rounding_unit"] for row in normalized}
        if len(units) != 1:
            raise CalibrationError("qualification packets use different reserve rounding units")
        rate = math.ceil(max(row["required_rate"] for row in normalized) / grid) * grid
        for row in normalized:
            total = reserve_total(row["exposure_person_years"], rate, row["rounding_unit"])
            row["candidate_reserve_total"] = total
            row["candidate_margin"] = total - row["target_reserve_before_rounding"]
            if row["candidate_margin"] < -1e-8:
                raise CalibrationError(f"{row['evidence_id']}: candidate total misses its target")
    except (CalibrationError, OSError) as exc:
        return {"schema": SCHEMA, "candidate": False, "blockers": [str(exc)]}
    return {
        "schema": SCHEMA,
        "candidate": True,
        "accepted": False,
        "blockers": list(PENDING_BLOCKERS),
        "rate_per_person_year": rate,
        "rate_grid": grid,
        "tail_slack_share": slack,
        "target_rule": "sum(q95) + tail_slack_share * sum(ES95 - q95)",
        "reference_lines": lines,
        "qualification_worlds": list(expected_worlds),
        "evidence": sorted(normalized, key=lambda row: (
            row["reference_line"], row["world"])),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.evidence.read_text())
        if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA \
                or not isinstance(payload.get("entries"), list):
            raise CalibrationError("evidence manifest schema or entries differ")
        result = calibrate(payload["entries"])
    except (OSError, UnicodeError, json.JSONDecodeError, CalibrationError) as exc:
        result = {"schema": SCHEMA, "candidate": False, "blockers": [str(exc)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("candidate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
