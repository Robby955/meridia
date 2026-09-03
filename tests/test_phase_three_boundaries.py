import hashlib
import json
from pathlib import Path

import pytest

from meridia.methods import phase_three


def _packet(path: Path, development: bool) -> Path:
    (path / "participant").mkdir(parents=True)
    contract = path / "participant" / "contract.json"
    contract.write_text("{}\n")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "development": development,
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
        "failed_composites": list(failed),
        "reserve": {
            "J": 1.0,
            "skill": 0.5,
            "mean_quantile_score": 2.0,
            "mean_shortfall_error": 3.0,
        },
    }


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

    (target / "manifest.json").write_text('{"development": 1}\n')
    with pytest.raises(ValueError, match="not a development packet"):
        phase_three._validate_packet_group([target], 1, True)


def test_full_packet_sets_require_canonical_parent_and_names(tmp_path):
    qualification = [
        _packet(tmp_path / "not-qualification" / f"qual-{index}", False)
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="canonical qual-0..qual-5"):
        phase_three._validate_packet_group(qualification, 6, False)


def test_nonempty_unbound_measurement_output_is_refused(tmp_path):
    out = tmp_path / "measurement"
    out.mkdir()
    (out / "orphan.json").write_text("{}\n")

    with pytest.raises(ValueError, match="nonempty measurement output"):
        phase_three._bind_measurement_output(out, {"run": 1})


def test_measurement_contract_rejects_a_symlinked_contract_file(tmp_path):
    packet = _packet(tmp_path / "packet", True)
    (packet / "retained").mkdir()
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
    (packet / "retained").mkdir()
    external = tmp_path / "external-sources"
    external.mkdir()
    (external / "survey.csv").write_text("value\n1\n")
    (packet / "participant" / "sources").symlink_to(
        external, target_is_directory=True
    )

    with pytest.raises(ValueError, match="inventory contains symlinks"):
        phase_three._verified_packet_files(packet)


def test_final_evidence_binds_optional_totals_file(tmp_path):
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "manifest.json").write_text("{}\n")
    submission = tmp_path / "submission"
    submission.mkdir()
    for name in phase_three.SUBMISSION_FILES:
        (submission / name).write_text(f"{name}\n")

    without_totals = phase_three._final_evidence_wrapper(packet, submission, {})
    (submission / "totals.csv").write_text("kind,count\ncounty,1\n")
    with_totals = phase_three._final_evidence_wrapper(packet, submission, {})
    assert without_totals["evidence_id"] != with_totals["evidence_id"]


def test_method_run_restart_requires_a_matching_receipt(monkeypatch, tmp_path):
    packet = _packet(tmp_path / "packet", False)
    output_root = tmp_path / "measurement"
    output_root.mkdir()
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
        "a" * 64,
        {"method": "A", "bootstrap_replicates": 100},
    )
    assert phase_three._run_once(*arguments) == {"scored": True}
    assert phase_three._run_once(*arguments) == {"scored": True}
    assert len(calls) == 1

    (submission / "release.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="receipt or output"):
        phase_three._run_once(*arguments)


def test_method_run_refuses_an_unreceipted_linked_submission(monkeypatch, tmp_path):
    packet = _packet(tmp_path / "packet", False)
    output_root = tmp_path / "measurement"
    output_root.mkdir()
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
            "a" * 64,
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
            "reasons": ["file set: unexpected [], missing ['reserve.csv']"],
            "reserve": {"feasible": False},
        },
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("truth audit ran for a hard-invalid report")

    monkeypatch.setattr(phase_three, "regional_liability_means", should_not_run)
    monkeypatch.setattr(phase_three, "elder_state_exposure_survival", should_not_run)
    scored = phase_three._score(packet, submission, {}, True)
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
    scored = phase_three._score(packet, submission, {}, True)
    assert scored["hard_check_pass"] is False
    assert scored["hard_check_failures"] == [
        "schema: verifier raised while parsing the submission: ValueError: missing column"
    ]
    assert len(scored["evidence"]["evidence_id"]) == 64


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
                "evidence": {"evidence_id": evidence_id},
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
        lambda packet: {"world": Path(packet).name},
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
