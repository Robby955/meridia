"""Fail-closed authorization for constructing V4 graded packets.

Qualification packets are built before any bars exist. Graded packets are different:
their seed material may be opened only after the complete composite-bar receipt is frozen
and its independently supplied reserve-rate audit matches the rate compiled into the
graded packet parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .verify import (
    QUALIFICATION_WORLD_NAMES,
    REFERENCE_LINES,
    RESERVE_CALIBRATION_SCHEMA,
    _bar_schema_errors,
)


class GradedReadinessError(ValueError):
    """A graded build lacks complete, matching freeze evidence."""


@dataclass(frozen=True)
class GradedReadiness:
    """Opaque result of validating the two receipts needed by a graded build."""

    bars_sha256: str
    reserve_calibration_sha256: str
    reserve_rate_per_person_year: float
    graded_world_count: int


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise GradedReadinessError(f"{label} must not be a symbolic link")
    if not source.is_file():
        raise GradedReadinessError(f"{label} is missing")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GradedReadinessError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise GradedReadinessError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GradedReadinessError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise GradedReadinessError(f"{label} must be positive and finite")
    return result


def validate_graded_readiness_payloads(
    bars: Mapping[str, Any],
    reserve_calibration: Mapping[str, Any],
    *,
    expected_rate_per_person_year: float,
    bars_sha256: str = "",
    reserve_calibration_sha256: str = "",
) -> GradedReadiness:
    """Validate already-loaded receipts without reading any graded seed material."""
    bars_object = dict(bars)
    errors = _bar_schema_errors(bars_object)
    if errors:
        raise GradedReadinessError(
            "composite bars are not a complete frozen receipt: " + "; ".join(errors)
        )

    audits = bars_object.get("reserve_audits")
    embedded = audits.get("calibration") if isinstance(audits, dict) else None
    supplied = dict(reserve_calibration)
    if embedded != supplied:
        raise GradedReadinessError(
            "reserve calibration receipt differs from the audit frozen into the bars"
        )
    if supplied.get("schema") != RESERVE_CALIBRATION_SCHEMA \
            or supplied.get("candidate") is not True \
            or supplied.get("accepted") is not True \
            or supplied.get("blockers") != []:
        raise GradedReadinessError("reserve calibration receipt is not accepted")
    if supplied.get("reference_lines") != list(REFERENCE_LINES) \
            or supplied.get("qualification_worlds") \
            != list(QUALIFICATION_WORLD_NAMES):
        raise GradedReadinessError(
            "reserve calibration receipt is not bound to the registered qualification set"
        )
    if supplied.get("measurement_contract_digest_sha256") \
            != bars_object.get("measurement_contract_digest_sha256"):
        raise GradedReadinessError(
            "reserve calibration and bars use different measurement contracts"
        )

    expected_rate = _rate(
        expected_rate_per_person_year, "compiled graded reserve rate"
    )
    observed_rate = _rate(
        supplied.get("rate_per_person_year"), "accepted reserve rate"
    )
    if observed_rate != expected_rate:
        raise GradedReadinessError(
            "accepted reserve rate does not match the graded packet parameters"
        )
    graded_world_count = bars_object.get("graded_world_count")
    if isinstance(graded_world_count, bool) \
            or not isinstance(graded_world_count, int) \
            or graded_world_count < 1:
        raise GradedReadinessError("frozen bars have an invalid graded world count")
    return GradedReadiness(
        bars_sha256=bars_sha256,
        reserve_calibration_sha256=reserve_calibration_sha256,
        reserve_rate_per_person_year=observed_rate,
        graded_world_count=graded_world_count,
    )


def validate_graded_readiness(
    bars_path: Path,
    reserve_calibration_path: Path,
    *,
    expected_rate_per_person_year: float,
) -> GradedReadiness:
    """Read and validate freeze receipts before a caller opens graded seed material."""
    bars, bars_digest = _read_json_object(bars_path, "composite bars receipt")
    audit, audit_digest = _read_json_object(
        reserve_calibration_path, "reserve calibration receipt"
    )
    return validate_graded_readiness_payloads(
        bars,
        audit,
        expected_rate_per_person_year=expected_rate_per_person_year,
        bars_sha256=bars_digest,
        reserve_calibration_sha256=audit_digest,
    )
