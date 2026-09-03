"""A graded seed remains unopened until both freeze receipts are valid."""

from __future__ import annotations

import json

import pytest

from meridia import graded_readiness as readiness


def _payloads() -> tuple[dict, dict]:
    audit = {
        "schema": readiness.RESERVE_CALIBRATION_SCHEMA,
        "candidate": True,
        "accepted": True,
        "blockers": [],
        "measurement_contract_digest_sha256": "a" * 64,
        "rate_per_person_year": 4_600.0,
        "reference_lines": list(readiness.REFERENCE_LINES),
        "qualification_worlds": list(readiness.QUALIFICATION_WORLD_NAMES),
    }
    bars = {
        "frozen": True,
        "graded_world_count": 3,
        "measurement_contract_digest_sha256": "a" * 64,
        "reserve_audits": {"calibration": audit},
    }
    return bars, audit


def test_matching_frozen_receipts_authorize_the_compiled_rate(monkeypatch):
    monkeypatch.setattr(readiness, "_bar_schema_errors", lambda bars: [])
    bars, audit = _payloads()
    result = readiness.validate_graded_readiness_payloads(
        bars,
        audit,
        expected_rate_per_person_year=4_600.0,
        bars_sha256="b" * 64,
        reserve_calibration_sha256="c" * 64,
    )
    assert result.reserve_rate_per_person_year == 4_600.0
    assert result.graded_world_count == 3
    assert result.bars_sha256 == "b" * 64
    assert result.reserve_calibration_sha256 == "c" * 64


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda bars, audit: audit.update(accepted=False), "not accepted"),
        (lambda bars, audit: audit.update(blockers=["open"]), "not accepted"),
        (
            lambda bars, audit: audit.update(reference_lines=["A", "B"]),
            "registered qualification set",
        ),
        (
            lambda bars, audit: audit.update(
                measurement_contract_digest_sha256="d" * 64
            ),
            "different measurement contracts",
        ),
        (
            lambda bars, audit: audit.update(rate_per_person_year=4_601.0),
            "does not match",
        ),
    ],
)
def test_invalid_reserve_receipts_fail_closed(monkeypatch, mutation, match):
    monkeypatch.setattr(readiness, "_bar_schema_errors", lambda bars: [])
    bars, audit = _payloads()
    mutation(bars, audit)
    bars["reserve_audits"]["calibration"] = audit
    with pytest.raises(readiness.GradedReadinessError, match=match):
        readiness.validate_graded_readiness_payloads(
            bars,
            audit,
            expected_rate_per_person_year=4_600.0,
        )


def test_a_separate_receipt_must_equal_the_one_frozen_into_the_bars(monkeypatch):
    monkeypatch.setattr(readiness, "_bar_schema_errors", lambda bars: [])
    bars, audit = _payloads()
    changed = dict(audit, rate_per_person_year=4_599.0)
    with pytest.raises(readiness.GradedReadinessError, match="differs"):
        readiness.validate_graded_readiness_payloads(
            bars,
            changed,
            expected_rate_per_person_year=4_600.0,
        )


def test_bar_schema_errors_are_not_bypassed(monkeypatch):
    monkeypatch.setattr(
        readiness, "_bar_schema_errors", lambda bars: ["freeze receipt must say frozen true"]
    )
    bars, audit = _payloads()
    with pytest.raises(readiness.GradedReadinessError, match="frozen true"):
        readiness.validate_graded_readiness_payloads(
            bars,
            audit,
            expected_rate_per_person_year=4_600.0,
        )


def test_receipt_files_must_be_regular_json_objects(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "_bar_schema_errors", lambda bars: [])
    bars, audit = _payloads()
    bars_path = tmp_path / "bars.json"
    audit_path = tmp_path / "reserve.json"
    bars_path.write_text(json.dumps(bars))
    audit_path.write_text(json.dumps(audit))
    result = readiness.validate_graded_readiness(
        bars_path,
        audit_path,
        expected_rate_per_person_year=4_600.0,
    )
    assert len(result.bars_sha256) == 64
    assert len(result.reserve_calibration_sha256) == 64

    link = tmp_path / "bars-link.json"
    link.symlink_to(bars_path)
    with pytest.raises(readiness.GradedReadinessError, match="symbolic link"):
        readiness.validate_graded_readiness(
            link,
            audit_path,
            expected_rate_per_person_year=4_600.0,
        )
