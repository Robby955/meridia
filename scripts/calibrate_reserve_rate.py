"""Select the public reserve rate from qualification reference submissions.

The input manifest names one participant packet and one reference submission for every
reference-line and qualification-world pair. Each reference contributes one candidate
rate, the sum of its submitted regional mean liabilities divided by the packet's public
exposure, rounded up to the registered rate grid. The published rate is the largest of
those candidates at which the reserve decision is still identified on every qualification
world.

A world is identified at a rate when the expected uncovered obligation under the published
proportional baseline exceeds the same quantity under a perfect-information allocation of
the same total by at least a registered share of that world's mean total liability. That
difference is the denominator of the reserve skill score. Where it collapses to zero the
skill score is undefined and no allocation can be told from any other, so a rate that
leaves any qualification world unidentified is refused here rather than after the freeze
halts on a missing statistic.

The identification half reads the qualification packet's retained continuation ensemble.
That is freeze-side evidence and is never available to a participant. The published rate
itself is a number in the contract, and the published total remains reproducible from the
participant's own experience file and that rate.

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

import numpy as np

from meridia.actuarial import (RESERVE_COLUMNS, ensemble_truth, expected_uncovered,
                               perfect_information_allocation,
                               proportional_baseline_allocation, reserve_total)


SCHEMA = "meridia.reserve-rate-calibration.v2"
MANIFEST_SCHEMA = "meridia.reserve-rate-evidence.v1"
QUALIFICATION_WORLD_NAMES = tuple(f"qual-{index}" for index in range(6))
RATE_GRID = 1.0
# The skill denominator a rate has to leave on every qualification world, as a share of
# that world's sealed mean total liability. One percent of the liability being reserved
# against is the smallest gap that still separates allocations on this world set: the
# denominator falls to a few thousand currency units within a few hundred rate points of
# where it vanishes, and a gap that small is a rounding artefact rather than a decision.
IDENTIFICATION_MARGIN_SHARE = 0.01
TARGET_RULE = "sum(submitted regional liability_mean) / public exposure"
IDENTIFICATION_RULE = (
    "largest candidate rate whose baseline-minus-oracle expected uncovered obligation is "
    "at least identification_margin_share of the sealed mean total liability on every "
    "qualification world"
)
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


def _submitted_reserve(submission: Path, n_states: int) -> tuple[float, float, float, str]:
    """Return the submitted mean, q95 and ES95 sums, and the file digest.

    The mean sum is what the rate targets. The two tail sums are read and ordered so that
    an unordered or negative reserve row is refused here as well, and they are carried in
    the record because the earlier rate rule read them and a reader comparing the two
    rules needs both quantities on one page.
    """
    path = _regular_file(submission, "reserve.csv")
    mean_sum = 0.0
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
                mean = _number(row["liability_mean"], f"{submission.name}: liability mean")
                q95 = _number(row["q95"], f"{submission.name}: q95")
                es95 = _number(row["es95"], f"{submission.name}: ES95")
                if mean < 0.0 or q95 < mean or es95 < q95:
                    raise CalibrationError(f"{submission.name}: submitted tails are invalid")
                mean_sum += mean
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
    if mean_sum <= 0.0:
        raise CalibrationError(f"{submission.name}: submitted mean liability is not positive")
    return mean_sum, q_sum, es_sum, hashlib.sha256(path.read_bytes()).hexdigest()


def _identification_inputs(packet: Path) -> dict[str, Any]:
    """Read the world quantities the identification margin is measured against.

    Every one of them is freeze-side. ``baseline_share`` and ``weights`` are published in
    the contract, and the continuation ensemble is the retained truth the reserve gate
    already scores against.
    """
    contract_path = _regular_file(packet, "participant/contract.json")
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"{packet.name}: contract.json is invalid") from exc
    reserve = contract.get("reserve") if isinstance(contract, dict) else None
    if not isinstance(reserve, dict):
        raise CalibrationError(f"{packet.name}: reserve contract is missing")
    path = _regular_file(packet, "retained/continuation_liabilities.npz")
    try:
        with np.load(path) as archive:
            liability = np.asarray(archive["liability"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as exc:
        raise CalibrationError(
            f"{packet.name}: retained continuation liabilities cannot be read") from exc
    if liability.ndim != 2 or liability.size == 0 or not np.isfinite(liability).all():
        raise CalibrationError(f"{packet.name}: continuation liabilities are invalid")
    truth = ensemble_truth(liability)
    raw_share = reserve.get("baseline_share")
    share = np.asarray(raw_share, dtype=np.float64) if raw_share else truth["q"]
    if share.shape != (liability.shape[1],) or not np.isfinite(share).all() \
            or float(share.sum()) <= 0.0:
        raise CalibrationError(f"{packet.name}: published baseline share is invalid")
    raw_weights = reserve.get("weights")
    weights = np.asarray(raw_weights, dtype=np.float64) if raw_weights else None
    if weights is not None and (weights.shape != (liability.shape[1],)
                                or not np.isfinite(weights).all()):
        raise CalibrationError(f"{packet.name}: published shortfall weights are invalid")
    sealed_mean_total = float(np.asarray(truth["mean"], dtype=np.float64).sum())
    if not math.isfinite(sealed_mean_total) or sealed_mean_total <= 0.0:
        raise CalibrationError(f"{packet.name}: sealed mean total liability is not positive")
    return {
        "liability": liability,
        "share": share,
        "weights": weights,
        "sealed_mean_total_liability": sealed_mean_total,
    }


def _identification(world: Mapping[str, Any], exposure: float, rounding_unit: float,
                    rate: float) -> dict[str, Any]:
    """Measure the reserve skill denominator on one world at one rate."""
    total = reserve_total(exposure, rate, rounding_unit)
    liability = world["liability"]
    weights = world["weights"]
    baseline = proportional_baseline_allocation(world["share"], total)
    oracle = perfect_information_allocation(liability, total, weights)
    j_baseline = float(expected_uncovered(baseline, liability, weights))
    j_oracle = float(expected_uncovered(oracle, liability, weights))
    denominator = j_baseline - j_oracle
    sealed = world["sealed_mean_total_liability"]
    return {
        "reserve_total": float(total),
        "j_baseline": j_baseline,
        "j_oracle": j_oracle,
        "skill_denominator": denominator,
        "margin_share": denominator / sealed,
        "sealed_mean_total_liability": sealed,
    }


def calibrate(entries: Sequence[Mapping[str, Any]], *,
              expected_worlds: Sequence[str] = QUALIFICATION_WORLD_NAMES,
              margin_share: float = IDENTIFICATION_MARGIN_SHARE,
              rate_grid: float = RATE_GRID) -> dict[str, Any]:
    """Return a fail-closed candidate-rate record from legal reference outputs."""
    try:
        margin = _number(margin_share, "identification margin share")
        grid = _number(rate_grid, "rate grid", positive=True)
        if not 0.0 < margin <= 1.0:
            raise CalibrationError("identification margin share must lie in (0, 1]")
        normalized: list[dict[str, Any]] = []
        pairs: defaultdict[tuple[str, str], int] = defaultdict(int)
        packets: dict[str, Path] = {}
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
            if packets.setdefault(world, packet) != packet:
                raise CalibrationError(f"{world}: entries name different packets")
            exposure, rounding_unit, n_states, experience_digest = _public_exposure(packet)
            mean_sum, q_sum, es_sum, reserve_digest = _submitted_reserve(submission, n_states)
            required_rate = mean_sum / exposure
            normalized.append({
                "reference_line": line,
                "world": world,
                "evidence_id": evidence_id,
                "exposure_person_years": exposure,
                "rounding_unit": rounding_unit,
                "submitted_liability_mean_sum": mean_sum,
                "submitted_q95_sum": q_sum,
                "submitted_es95_sum": es_sum,
                "target_reserve_before_rounding": mean_sum,
                "required_rate": required_rate,
                "candidate_rate": math.ceil(required_rate / grid) * grid,
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
        geometry = {world: _identification_inputs(packets[world]) for world in worlds}
        exposures = {row["world"]: row["exposure_person_years"] for row in normalized}
        rounding = normalized[0]["rounding_unit"]
        candidates: list[dict[str, Any]] = []
        for value in sorted({row["candidate_rate"] for row in normalized}, reverse=True):
            per_world = {
                world: _identification(geometry[world], exposures[world], rounding, value)
                for world in worlds
            }
            worst = min(reading["margin_share"] for reading in per_world.values())
            candidates.append({
                "rate_per_person_year": value,
                "identified": bool(worst >= margin),
                "worst_margin_share": worst,
                "worst_world": min(
                    per_world, key=lambda name: per_world[name]["margin_share"]),
                "worlds": per_world,
            })
        identified = [row for row in candidates if row["identified"]]
        if not identified:
            best = max(candidates, key=lambda row: row["worst_margin_share"])
            raise CalibrationError(
                "no candidate rate keeps the reserve decision identified on every "
                f"qualification world; the best is {best['rate_per_person_year']:.12g} at "
                f"a worst margin of {best['worst_margin_share']:.6f} on "
                f"{best['worst_world']} against a required {margin:.6f}"
            )
        chosen = identified[0]
        rate = float(chosen["rate_per_person_year"])
        for row in normalized:
            total = reserve_total(row["exposure_person_years"], rate, row["rounding_unit"])
            row["candidate_reserve_total"] = total
            row["candidate_margin"] = total - row["target_reserve_before_rounding"]
    except (CalibrationError, OSError) as exc:
        return {"schema": SCHEMA, "candidate": False, "blockers": [str(exc)]}
    return {
        "schema": SCHEMA,
        "candidate": True,
        "accepted": False,
        "blockers": list(PENDING_BLOCKERS),
        "rate_per_person_year": rate,
        "rate_grid": grid,
        "identification_margin_share": margin,
        "target_rule": TARGET_RULE,
        "identification_rule": IDENTIFICATION_RULE,
        "binding_reference": min(
            (row for row in normalized if row["candidate_rate"] == rate),
            key=lambda row: (row["reference_line"], row["world"]),
        )["evidence_id"],
        "identification": {
            "chosen": {
                "rate_per_person_year": rate,
                "worst_margin_share": chosen["worst_margin_share"],
                "worst_world": chosen["worst_world"],
                "worlds": {
                    world: {key: value for key, value in reading.items()}
                    for world, reading in chosen["worlds"].items()
                },
            },
            "candidates": [
                {
                    "rate_per_person_year": row["rate_per_person_year"],
                    "identified": row["identified"],
                    "worst_margin_share": row["worst_margin_share"],
                    "worst_world": row["worst_world"],
                    "margin_share": {
                        world: reading["margin_share"]
                        for world, reading in row["worlds"].items()
                    },
                    "skill_denominator": {
                        world: reading["skill_denominator"]
                        for world, reading in row["worlds"].items()
                    },
                }
                for row in candidates
            ],
        },
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
