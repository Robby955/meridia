import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from meridia.methods import phase_three


def _packet(path: Path, development: bool) -> Path:
    (path / "participant").mkdir(parents=True)
    (path / "retained").mkdir()
    contract = path / "participant" / "contract.json"
    contract.write_text("{}\n")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": phase_three.PACKET_MANIFEST_SCHEMA,
                "development": development,
                "packet_class": "development" if development else "qualification",
                "participant": {
                    "contract.json": {
                        "bytes": contract.stat().st_size,
                        "sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
                    }
                },
                "retained": {},
            }
        )
        + "\n"
    )
    return path


def _summary(*, hard_check_pass: bool, failed=()) -> dict:
    return {
        "hard_check_pass": hard_check_pass,
        "gate_evaluation_complete": hard_check_pass,
        "failed_composites": list(failed),
        "reserve": {
            "J": 1.0,
            "skill": 0.5,
            "mean_quantile_score": 2.0,
            "mean_shortfall_error": 3.0,
        },
    }


def _rebind_manifest_file(packet: Path, side: str, name: str) -> None:
    path = packet / side / name
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[side][name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")


def test_packet_validation_checks_the_resolved_path_before_reading(tmp_path):
    target = tmp_path / "graded-hidden" / "world"
    _packet(target, True)
    alias = tmp_path / "safe-world"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="refuse graded"):
        phase_three._validate_packet_group([alias], 1, True)


def test_packet_validation_returns_resolved_paths_and_requires_exact_class(tmp_path):
    target = _packet(tmp_path / "development-world", True)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    assert phase_three._validate_packet_group([alias], 1, True) == [target.resolve()]

    (target / "manifest.json").write_text(
        json.dumps(
            {
                "schema": phase_three.PACKET_MANIFEST_SCHEMA,
                "development": 1,
                "packet_class": "development",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="not a development packet"):
        phase_three._validate_packet_group([target], 1, True)


def test_packet_class_cannot_be_changed_by_renaming_a_graded_packet(tmp_path):
    packet = _packet(tmp_path / "qualification" / "qual-0", False)
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["packet_class"] = "graded"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="not a qualification packet"):
        phase_three._validate_packet_group([packet], 1, False)


def test_full_packet_sets_require_canonical_parent_and_names(tmp_path):
    qualification = [
        _packet(tmp_path / "not-qualification" / f"qual-{index}", False)
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="canonical qual-0..qual-5"):
        phase_three._validate_packet_group(qualification, 6, False)


def test_full_packet_sets_reject_mixed_build_roots(tmp_path):
    qualification = [
        _packet(
            tmp_path
            / ("build-a" if index < 3 else "build-b")
            / "qualification"
            / f"qual-{index}",
            False,
        )
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="share one resolved parent"):
        phase_three._validate_packet_group(qualification, 6, False)

    development = [tmp_path / "build-a" / "development" / f"dev-{index:02d}" for index in range(12)]
    qualification = [tmp_path / "build-b" / "qualification" / f"qual-{index}" for index in range(6)]
    with pytest.raises(ValueError, match="share one worlds root"):
        phase_three._validate_shared_worlds_root(development, qualification)


def test_nonempty_unbound_measurement_output_is_refused(tmp_path):
    out = tmp_path / "measurement"
    out.mkdir()
    (out / "orphan.json").write_text("{}\n")

    with pytest.raises(ValueError, match="nonempty measurement output"):
        phase_three._bind_measurement_output(out, {"run": 1})


def test_lone_interrupted_measurement_contract_is_recovered(tmp_path):
    out = tmp_path / "measurement"
    out.mkdir()
    temporary = out / ".measurement_contract.json.tmp"
    temporary.write_text("{torn")

    phase_three._bind_measurement_output(out, {"run": 1})

    assert json.loads((out / "measurement_contract.json").read_text()) == {"run": 1}
    assert not temporary.exists()


def test_interrupted_measurement_contract_with_companion_is_refused(tmp_path):
    out = tmp_path / "measurement"
    out.mkdir()
    (out / ".measurement_contract.json.tmp").write_text("{torn")
    (out / "orphan.json").write_text("{}\n")

    with pytest.raises(ValueError, match="nonempty measurement output"):
        phase_three._bind_measurement_output(out, {"run": 1})


def test_measurement_contract_rejects_a_symlinked_contract_file(tmp_path):
    packet = _packet(tmp_path / "packet", True)
    contract = packet / "participant" / "contract.json"
    external = tmp_path / "contract.json"
    contract.rename(external)
    contract.symlink_to(external)

    with pytest.raises(ValueError, match="inventory contains symlinks"):
        phase_three._measurement_contract(
            [packet], [], tmp_path / "bars.json", phase_three.MeasurementParams()
        )


def test_measurement_output_rejects_linked_contract_and_output_roots(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="directory may not be a symlink"):
        phase_three._bind_measurement_output(linked_root, {"run": 1})

    out = tmp_path / "measurement"
    out.mkdir()
    external = tmp_path / "external-contract.json"
    external.write_text('{"run": 1}\n')
    (out / "measurement_contract.json").symlink_to(external)
    with pytest.raises(ValueError, match="may not use symlinked paths"):
        phase_three._bind_measurement_output(out, {"run": 1})


def test_calibration_artifact_is_hash_bound_across_restart(tmp_path):
    out = tmp_path / "measurement"
    out.mkdir()
    phase_three._bind_measurement_output(out, {"run": 1})
    calls = []

    def generate(path):
        calls.append(path)
        path.write_text('{"coefficient": 1.25}\n')

    artifact = phase_three._ensure_calibration_artifact(out, "A", generate)
    assert json.loads(artifact.read_text()) == {"coefficient": 1.25}
    assert len(calls) == 1
    assert phase_three._ensure_calibration_artifact(out, "A", generate) == artifact
    assert len(calls) == 1

    artifact.write_text('{"coefficient": 9.0}\n')
    with pytest.raises(ValueError, match="changed after it was bound"):
        phase_three._ensure_calibration_artifact(out, "A", generate)


def test_unreceipted_or_partial_calibration_is_refused(tmp_path):
    out = tmp_path / "measurement"
    out.mkdir()
    phase_three._bind_measurement_output(out, {"run": 1})
    (out / "calibration_A.json").write_text('{"coefficient": 1.0}\n')
    with pytest.raises(ValueError, match="without a bound receipt"):
        phase_three._ensure_calibration_artifact(out, "A", lambda _: None)

    (out / "calibration_A.json").unlink()
    (out / "calibration_receipts.json").write_text("{")
    with pytest.raises(ValueError, match="incomplete or invalid"):
        phase_three._ensure_calibration_artifact(out, "A", lambda _: None)


def test_measurement_contract_hashes_every_meridia_python_source(tmp_path):
    development = _packet(tmp_path / "development", True)
    qualification = _packet(tmp_path / "qualification", False)
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')

    contract = phase_three._measurement_contract(
        [development],
        [qualification],
        bars,
        phase_three.MeasurementParams(),
    )
    repo_root = Path(phase_three.__file__).resolve().parents[2]
    expected = {
        str(path.relative_to(repo_root))
        for path in (repo_root / "meridia").rglob("*.py")
    }
    assert set(contract["source_sha256"]) == expected
    assert contract["packet_file_sha256"][str(development.resolve())] == {
        "participant/contract.json": hashlib.sha256(b"{}\n").hexdigest()
    }

    (development / "participant" / "contract.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        phase_three._measurement_contract(
            [development],
            [qualification],
            bars,
            phase_three.MeasurementParams(),
        )


def test_method_run_rejects_packet_mutation_after_contract_binding(tmp_path):
    packet = _packet(tmp_path / "packet", False)
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    contract = phase_three._measurement_contract(
        [], [packet], bars, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)

    def runner(stage):
        stage.mkdir(parents=True)
        for name in phase_three.SUBMISSION_FILES:
            (stage / name).write_text(f"{name}\n")
        participant_contract = packet / "participant" / "contract.json"
        participant_contract.write_text('{"changed": true}\n')
        _rebind_manifest_file(packet, "participant", "contract.json")

    with pytest.raises(ValueError, match="changed after measurement binding"):
        phase_three._run_once(
            packet,
            output_root / "qualification" / "qual-0" / "A",
            runner,
            {},
            True,
            output_root,
            contract_sha256,
            {"method": "A"},
        )


def test_calibration_rejects_packet_mutation_during_generation(tmp_path):
    packet = _packet(tmp_path / "packet", True)
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    contract = phase_three._measurement_contract(
        [packet], [], bars, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)

    def generate(path):
        path.write_text("{}\n")
        participant_contract = packet / "participant" / "contract.json"
        participant_contract.write_text('{"changed": true}\n')
        _rebind_manifest_file(packet, "participant", "contract.json")

    with pytest.raises(ValueError, match="changed after measurement binding"):
        phase_three._ensure_bound_calibration_artifact(
            output_root,
            "A",
            generate,
            contract_sha256,
            [packet],
        )


def test_bound_bars_reject_changed_bytes(tmp_path):
    packet = _packet(tmp_path / "packet", False)
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    contract = phase_three._measurement_contract(
        [], [packet], bars, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)
    bars.write_text('{"frozen": false}\n')

    with pytest.raises(ValueError, match="bars changed after measurement binding"):
        phase_three._load_bound_bars(output_root, contract_sha256, bars)


def test_measurement_mode_requires_either_raw_or_frozen_bars(tmp_path):
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')
    assert phase_three._validate_measurement_mode(None, True) is None
    assert phase_three._validate_measurement_mode(bars, False) == bars
    with pytest.raises(ValueError, match="cannot be combined"):
        phase_three._validate_measurement_mode(bars, True)
    with pytest.raises(ValueError, match="requires --bars"):
        phase_three._validate_measurement_mode(None, False)


def test_gate_metrics_are_json_safe():
    gates = {
        name: {"pass": True, "evaluated": True, "reasons": []}
        for name in phase_three.COMPOSITE_FAMILIES
    }
    summary = phase_three.summarize_report(
        {
            "pass": True,
            "hard_pass": True,
            "reasons": [],
            "metrics": {"error": np.asarray([0.1, 0.2])},
            "rate_metrics": {"eligible": np.int64(12)},
            "composite_metrics": {"release_accuracy": {"error": np.float64(0.2)}},
            "gate_results": gates,
            "eligibility_evidence": {"cells": np.asarray([500.0])},
            "evidence": {"schema": "v4", "digest": "a" * 64},
            "reserve": {"feasible": True, "skill": np.float64(0.5)},
        }
    )
    encoded = json.dumps(summary)
    assert '"error": [0.1, 0.2]' in encoded
    assert '"eligible": 12' in encoded
    assert summary["gate_evaluation_complete"] is True
    assert summary["failed_composites"] == []
    assert summary["eligibility_evidence"] == {"cells": [500.0]}
    assert len(summary["verifier_evidence_id"]) == 64


def test_raw_pre_freeze_calls_the_verifier_without_bars_and_retains_fields(
    monkeypatch, tmp_path
):
    packet = tmp_path / "packet"
    packet.mkdir()
    submission = tmp_path / "submission"
    submission.mkdir()
    observed = []
    raw_gates = {
        name: {
            "pass": False,
            "evaluated": False,
            "reasons": ["frozen bars not supplied"],
        }
        for name in phase_three.COMPOSITE_FAMILIES
    }
    evidence = {"schema": "meridia.v4.verifier-evidence.v1", "digest": "b" * 64}

    def verify(packet_dir, submission_dir, bars):
        observed.append((packet_dir, submission_dir, bars))
        return {
            "pass": False,
            "hard_pass": True,
            "reasons": ["bars: no frozen composite bar receipt was supplied"],
            "metrics": {"persons": {"p95": 0.1}},
            "projection_metrics": {"persons": {"p95": 0.2}},
            "rate_metrics": {"mortality": {"p95": 0.3}},
            "composite_metrics": {
                "release_accuracy": {"p95_relative_error": 0.1}
            },
            "gate_results": raw_gates,
            "eligibility_evidence": {"scored_cells": 72},
            "reserve_rule_evidence": {"valid": True},
            "reserve_rule_errors": [],
            "reserve": {"feasible": True, "skill": 0.4},
            "evidence": evidence,
        }

    monkeypatch.setattr(phase_three, "verify_submission", verify)
    monkeypatch.setattr(phase_three, "regional_liability_means", lambda *args: [])
    monkeypatch.setattr(
        phase_three, "elder_state_exposure_survival", lambda *args: {"states": []}
    )
    monkeypatch.setattr(
        phase_three,
        "sealed_exceedance_audit",
        lambda *args: {"pooled_exceedance_deviation": 0.2},
    )

    scored = phase_three._score(packet, submission, None, True)
    assert observed == [(packet, submission, None)]
    assert scored["measurement_mode"] == phase_three.RAW_PRE_FREEZE_MODE
    assert scored["hard_pass"] is True
    assert scored["composite_metrics"] == {
        "release_accuracy": {"p95_relative_error": 0.1}
    }
    assert scored["gate_results"] == raw_gates
    assert scored["gate_evaluation_complete"] is False
    assert scored["composite_pass"] is None
    assert scored["failed_composites"] is None
    assert scored["eligibility_evidence"] == {"scored_cells": 72}
    assert scored["reserve"] == {"feasible": True, "skill": 0.4}
    assert scored["evidence"] == evidence
    with pytest.raises(ValueError, match="must not receive bars"):
        phase_three._score(packet, submission, {}, True)


def test_packet_inventory_rejects_a_symlinked_side_directory(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    external = tmp_path / "graded-external"
    external.mkdir()
    (external / "contract.json").write_text("{}\n")
    (packet / "participant").symlink_to(external, target_is_directory=True)
    (packet / "retained").mkdir()
    (packet / "manifest.json").write_text(
        json.dumps(
            {
                "schema": phase_three.PACKET_MANIFEST_SCHEMA,
                "development": False,
                "packet_class": "qualification",
                "participant": {
                    "contract.json": {
                        "bytes": 3,
                        "sha256": hashlib.sha256(b"{}\n").hexdigest(),
                    }
                },
                "retained": {},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="may not be a symlink"):
        phase_three._verified_packet_files(packet)


def test_packet_inventory_rejects_an_omitted_nested_directory_symlink(tmp_path):
    packet = _packet(tmp_path / "packet", True)
    external = tmp_path / "external-sources"
    external.mkdir()
    (external / "survey.csv").write_text("value\n1\n")
    (packet / "participant" / "sources").symlink_to(
        external, target_is_directory=True
    )

    with pytest.raises(ValueError, match="inventory contains symlinks"):
        phase_three._verified_packet_files(packet)


def test_phase_three_requires_the_exact_v4_three_file_surface(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    for name in phase_three.SUBMISSION_FILES:
        (submission / name).write_text(f"{name}\n")

    assert phase_three.SUBMISSION_FILES == tuple(phase_three.V4_SUBMISSION_COLUMNS)
    assert set(phase_three._submission_hashes(submission)) == {
        "release.csv",
        "projection.csv",
        "reserve.csv",
    }
    (submission / "totals.csv").write_text("kind,count\ncounty,1\n")
    with pytest.raises(ValueError, match=r"unexpected \['totals.csv'\]"):
        phase_three._submission_hashes(submission)
    (submission / "totals.csv").unlink()
    (submission / "detailed.csv").write_text("county,count\n0,1\n")
    with pytest.raises(ValueError, match=r"unexpected \['detailed.csv'\]"):
        phase_three._submission_hashes(submission)


def test_method_run_restart_requires_a_matching_receipt(monkeypatch, tmp_path):
    packet = _packet(tmp_path / "packet", False)
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')
    contract = phase_three._measurement_contract(
        [], [packet], bars, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)
    submission = output_root / "qualification" / "qual-0" / "A"
    calls = []

    def runner(stage):
        calls.append(stage)
        stage.mkdir(parents=True)
        for name in phase_three.SUBMISSION_FILES:
            (stage / name).write_text(f"{name}\n")

    monkeypatch.setattr(
        phase_three,
        "_score",
        lambda packet, submission, bars, allow_unfrozen: {"scored": True},
    )
    arguments = (
        packet,
        submission,
        runner,
        {},
        True,
        output_root,
        contract_sha256,
        {"method": "A", "bootstrap_replicates": 100},
    )
    assert phase_three._run_once(*arguments) == {"scored": True}
    assert phase_three._run_once(*arguments) == {"scored": True}
    assert len(calls) == 1

    (submission / "release.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="receipt or output"):
        phase_three._run_once(*arguments)


@pytest.mark.parametrize("restart", [False, True])
def test_method_run_rechecks_receipt_after_scoring(monkeypatch, tmp_path, restart):
    packet = _packet(tmp_path / "packet", False)
    bars_path = tmp_path / "bars.json"
    bars_path.write_text('{"frozen": true}\n')
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    contract = phase_three._measurement_contract(
        [], [packet], bars_path, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)
    submission = output_root / "qualification" / "qual-0" / "A"

    def runner(stage):
        stage.mkdir(parents=True)
        for name in phase_three.SUBMISSION_FILES:
            (stage / name).write_text(f"{name}\n")

    arguments = (
        packet,
        submission,
        runner,
        {},
        True,
        output_root,
        contract_sha256,
        {"method": "A"},
    )
    if restart:
        monkeypatch.setattr(phase_three, "_score", lambda *args: {"scored": True})
        assert phase_three._run_once(*arguments) == {"scored": True}

    def mutating_score(packet, submission, bars, allow_unfrozen):
        (submission / "release.csv").write_text("changed during score\n")
        return {"scored": True}

    monkeypatch.setattr(phase_three, "_score", mutating_score)
    with pytest.raises(ValueError, match="receipt or output"):
        phase_three._run_once(*arguments)


def test_method_run_rechecks_bound_calibration_after_runner(monkeypatch, tmp_path):
    packet = _packet(tmp_path / "packet", False)
    bars_path = tmp_path / "bars.json"
    bars_path.write_text('{"frozen": true}\n')
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    contract = phase_three._measurement_contract(
        [], [packet], bars_path, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)
    calibration = phase_three._ensure_bound_calibration_artifact(
        output_root,
        "A",
        lambda path: path.write_text("{}\n"),
        contract_sha256,
        [packet],
    )
    calibration_sha256 = phase_three._bound_calibration_sha256(
        output_root, contract_sha256, "A", calibration
    )

    def runner(stage):
        stage.mkdir(parents=True)
        for name in phase_three.SUBMISSION_FILES:
            (stage / name).write_text(f"{name}\n")
        calibration.write_text('{"changed": true}\n')

    monkeypatch.setattr(phase_three, "_score", lambda *args: {"scored": True})
    with pytest.raises(ValueError, match="bound method input changed"):
        phase_three._run_once(
            packet,
            output_root / "qualification" / "qual-0" / "A",
            runner,
            {},
            True,
            output_root,
            contract_sha256,
            {"method": "A", "calibration_sha256": calibration_sha256},
            {calibration: calibration_sha256},
        )


def test_method_run_refuses_an_unreceipted_linked_submission(monkeypatch, tmp_path):
    packet = _packet(tmp_path / "packet", False)
    output_root = tmp_path / "measurement"
    output_root.mkdir()
    bars = tmp_path / "bars.json"
    bars.write_text('{"frozen": true}\n')
    contract = phase_three._measurement_contract(
        [], [packet], bars, phase_three.MeasurementParams()
    )
    phase_three._bind_measurement_output(output_root, contract)
    contract_sha256 = phase_three._measurement_contract_sha256(output_root)
    target = output_root / "copied-A"
    target.mkdir()
    for name in phase_three.SUBMISSION_FILES:
        (target / name).write_text(f"{name}\n")
    submission = output_root / "qualification" / "qual-0" / "third"
    submission.parent.mkdir(parents=True)
    submission.symlink_to(target, target_is_directory=True)
    called = False

    def runner(stage):
        nonlocal called
        called = True

    monkeypatch.setattr(phase_three, "_score", lambda *args: {})
    with pytest.raises(ValueError, match="may not use symlinked paths"):
        phase_three._run_once(
            packet,
            submission,
            runner,
            {},
            True,
            output_root,
            contract_sha256,
            {"method": "third"},
        )
    assert called is False


def test_hard_invalid_score_skips_truth_audits(monkeypatch, tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "manifest.json").write_text("{}\n")
    submission = tmp_path / "submission"
    submission.mkdir()
    monkeypatch.setattr(
        phase_three,
        "verify_submission",
        lambda *args, **kwargs: {
            "pass": False,
            "hard_pass": False,
            "reasons": ["file set: unexpected [], missing ['reserve.csv']"],
            "composite_metrics": {},
            "gate_results": {},
            "reserve": {"feasible": False},
        },
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("truth audit ran for a hard-invalid report")

    monkeypatch.setattr(phase_three, "regional_liability_means", should_not_run)
    monkeypatch.setattr(phase_three, "elder_state_exposure_survival", should_not_run)
    scored = phase_three._score(packet, submission, None, True)
    assert scored["hard_check_pass"] is False
    assert scored["truth_audit_status"] == "unavailable_due_to_hard_check_failure"
    assert scored["regional_liability_means"] is None
    assert scored["state_65_plus"] is None


def test_verifier_parse_exception_becomes_a_bound_hard_failure(monkeypatch, tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "manifest.json").write_text("{}\n")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "broken.csv").write_text("bad\n")
    monkeypatch.setattr(
        phase_three,
        "verify_submission",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing column")),
    )
    scored = phase_three._score(packet, submission, None, True)
    assert scored["hard_check_pass"] is False
    assert scored["hard_check_failures"] == [
        "schema: verifier raised while parsing the submission: ValueError: missing column"
    ]
    assert scored["evidence"] == {}
    assert scored["verifier_evidence_id"] is None


def test_method_and_deletion_comparisons_are_indeterminate_when_hard_invalid():
    valid = _summary(hard_check_pass=True)
    invalid = _summary(hard_check_pass=False)

    comparison = phase_three._comparison(valid, invalid, valid)
    assert comparison == {
        "valid": False,
        "status": "indeterminate_due_to_hard_check_failure",
        "hard_invalid": ["A"],
        "composites": {},
    }
    change = phase_three._reserve_change(valid, invalid)
    assert change["valid"] is False
    assert change["changed"] is None
    assert change["hard_invalid"] == ["deletion"]


def test_valid_method_comparison_uses_only_the_five_retained_composites():
    third = _summary(hard_check_pass=True, failed=("reserve_skill",))
    reference_a = _summary(hard_check_pass=True, failed=("reserve_skill",))
    reference_b = _summary(hard_check_pass=True)

    comparison = phase_three._comparison(third, reference_a, reference_b)
    assert comparison["valid"] is True
    assert tuple(comparison["composites"]) == phase_three.COMPOSITE_FAMILIES
    assert comparison["composites"]["reserve_skill"]["relative"] == "matches_A"


def test_elder_audit_writes_exact_six_world_schema(monkeypatch, tmp_path):
    packets = []
    report = {"qualification": {}}
    family = {
        "annual_rate": 0.20,
        "kinds": {
            "mortality_spike": {
                "mortality_multiplier": [1.5, 3.0],
                "admission_multiplier": [1.4, 2.6],
            }
        },
    }
    for index in range(6):
        packet = _packet(tmp_path / "qualification" / f"qual-{index}", False)
        (packet / "participant" / "contract.json").write_text(
            json.dumps({"shock_family": family}) + "\n"
        )
        packets.append(packet)

        def line(evidence_id, estimate):
            return {
                "hard_check_pass": True,
                "verifier_evidence_id": evidence_id,
                "state_65_plus": {
                    "states": [
                        {
                            "state": 0,
                            "estimated_person_years": estimate,
                            "sealed_person_years": 100.0,
                        }
                    ]
                },
                "regional_liability_means": [
                    {"region": 0, "submitted_mean": estimate, "sealed_mean": 100.0}
                ],
                "sealed_exceedance": {"pooled_exceedance_deviation": 0.1},
            }

        report["qualification"][packet.name] = {
            "methods": {
                "A": line(f"before-{index}", 90.0),
                "third": line(f"after-{index}", 95.0),
            }
        }
    monkeypatch.setattr(
        phase_three,
        "_method_source_digest",
        lambda: ("a" * 64, "b" * 40),
    )
    monkeypatch.setattr(
        phase_three,
        "mortality_gap_decomposition",
        lambda packet: {
            "world": Path(packet).name,
            "continuation_shocks_redrawn_per_member": True,
        },
    )
    monkeypatch.setattr(
        phase_three,
        "elder_eligibility_audit",
        lambda packet: {"world": Path(packet).name},
    )
    result = phase_three.write_elder_reconstruction_audit(report, packets, tmp_path)
    payload = result["payload"]
    assert payload["schema"] == "meridia.methods.elder_reconstruction_audit.v1"
    assert len(payload["worlds"]) == 6
    assert payload["shock_redraw"] == {
        "annual_probability": 0.20,
        "independent_per_member": True,
        "magnitude_source": "participant/contract.json:shock_family",
        "mortality_ranges": [
            {"kind": "mortality_spike", "range": [1.5, 3.0]}
        ],
        "admission_ranges": [
            {"kind": "mortality_spike", "range": [1.4, 2.6]}
        ],
    }
    assert payload["eligibility_audit"]["younger_floors_changed"] is False
    assert Path(result["json_path"]).is_file()
    assert Path(result["text_path"]).is_file()
