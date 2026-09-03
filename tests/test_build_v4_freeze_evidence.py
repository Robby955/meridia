"""Focused tests for the restartable P4 freeze-evidence runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _shock_report(packet_inputs: dict[str, str]) -> dict:
    from meridia.packet import continuation_source_law_digest

    schedules = [
        {"member": 0, "future_shocks": []},
        {
            "member": 1,
            "future_shocks": [{
                "year": 10,
                "kind": "mortality_spike",
                "mortality_multiplier": 2.0,
                "admission_multiplier": 2.0,
            }],
        },
        {"member": 2, "future_shocks": []},
        {
            "member": 3,
            "future_shocks": [{
                "year": 11,
                "kind": "migration_wave",
                "leave_home_multiplier": 2.0,
            }],
        },
    ]
    runtime = {
        "schema": "meridia.v4.continuation-shock-redraw.v1",
        "continuation_source_law_sha256": continuation_source_law_digest(),
        "member_count": 4,
        "redrawn_member_count": 4,
        "first_future_year": 10,
        "future_year_count": 5,
        "future_year_opportunity_count": 20,
        "member_schedules": schedules,
        "ordered_member_schedule_digest_sha256": hashlib.sha256(json.dumps(
            schedules, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        "distinct_future_schedule_count": 3,
        "future_shock_year_count": 2,
        "future_mortality_spike_year_count": 1,
    }
    runtime_file_digest = hashlib.sha256((json.dumps(
        runtime, indent=1, sort_keys=True, allow_nan=False,
    ) + "\n").encode()).hexdigest()
    packet_inputs["retained/continuation_shock_redraw.json"] = runtime_file_digest
    return {
        "schema": "meridia.v4.continuation-shock-redraw-report.v1",
        "runtime_evidence_file_sha256": runtime_file_digest,
        "liability_archive_sha256": packet_inputs[
            "retained/continuation_liabilities.npz"
        ],
        "runtime_evidence": runtime,
    }


def _elder_report(line: str, world: str) -> dict:
    packet_digest = _digest(f"packet-{world}")
    submission_digest = _digest(f"submission-{line}-{world}")
    packet_files = {
        "participant/contract.json": _digest(f"contract-{world}"),
        "participant/experience_history.csv": _digest(f"experience-{world}"),
        "retained/continuation_liabilities.npz": _digest(f"liability-{world}"),
        "retained/continuation_shock_redraw.json": _digest(f"shock-{world}"),
    }
    return {
        "evidence": {
            "schema": "meridia.v4.verifier-evidence.v1",
            "packet_digest_sha256": packet_digest,
            "contract_digest_sha256": _digest(f"contract-{world}"),
            "submission_digest_sha256": submission_digest,
            "verifier_digest_sha256": _digest("verifier"),
            "packet_file_sha256": packet_files,
        },
        "continuation_shock_redraw_evidence": _shock_report(packet_files),
        "elder_reference_evidence": {
            "schema": "meridia.v4.elder-reference-evidence.v1",
            "valid": True,
            "packet_digest_sha256": packet_digest,
            "submission_digest_sha256": submission_digest,
            "state_65_plus_person_years": [
                {
                    "state": state,
                    "submitted_person_years": 105.0 if line == "C" else 110.0,
                    "sealed_person_years": 100.0,
                }
                for state in range(6)
            ],
            "liability_mean_by_region": [
                {
                    "region": region,
                    "submitted": 950.0 if line == "C" else 900.0,
                    "sealed": 1_000.0,
                }
                for region in range(6)
            ],
        },
    }


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/build_v4_freeze_evidence.py"
    spec = importlib.util.spec_from_file_location("build_v4_freeze_evidence_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _phase_three_run(runner, monkeypatch, tmp_path, method):
    phase = runner.phase_three
    output_root = tmp_path / "phase-three"
    output_root.mkdir()
    packet = tmp_path / "qual-0"
    packet.mkdir()
    submission = output_root / "qualification" / "qual-0" / "A"
    monkeypatch.setattr(phase, "_verify_bound_packet", lambda *args: None)
    monkeypatch.setattr(phase, "_verify_bound_inputs", lambda *args: None)
    monkeypatch.setattr(phase, "_bound_manifest_sha256", lambda *args: _digest("packet"))
    monkeypatch.setattr(phase, "_score", lambda *args: {"scored": True})
    arguments = (
        packet,
        submission,
        method,
        {},
        True,
        output_root,
        _digest("contract"),
        {"method": "A", "method_seed": runner.resampling.REFERENCE_METHOD_SEEDS["A"]},
    )
    return phase, submission, arguments


def _replicate_run(runner, tmp_path):
    out = tmp_path / "evidence"
    out.mkdir()
    base_packet = tmp_path / "qual-0"
    (base_packet / "retained").mkdir(parents=True)
    (base_packet / "retained" / "truth.csv").write_text("value\n1\n")
    (base_packet / "manifest.json").write_text("manifest\n")
    resample_packet = out / "outer_resamples" / "qual-0" / "outer-000"
    participant = resample_packet / "participant"
    participant.mkdir(parents=True)
    (participant / "input.csv").write_text("value\n1\n")
    resampling_manifest = resample_packet.parent / "manifest.json"
    resampling_manifest.write_text("resampling\n")
    calibration_a = out / "calibration_A.json"
    calibration_b = out / "calibration_B.json"
    calibration_a.write_text("{}\n")
    calibration_b.write_text("{}\n")
    inventory = runner._tree_inventory(participant)
    participant_digest = runner._canonical_digest(
        {
            name: {
                "bytes": (participant / name).stat().st_size,
                "sha256": digest,
            }
            for name, digest in inventory.items()
        }
    )
    submission = out / "replicate_submissions" / "qual-0" / "outer-000" / "A"
    arguments = {
        "line": "A",
        "base_packet": base_packet,
        "resample_packet": resample_packet,
        "resample_row": {
            "replicate_id": "outer-000",
            "participant_digest_sha256": participant_digest,
        },
        "resampling_manifest_path": resampling_manifest,
        "submission": submission,
        "calibration_a": calibration_a,
        "calibration_b": calibration_b,
        "params": runner.phase_three.MeasurementParams(),
        "run_spec": {
            "method": "A",
            "method_seed": runner.resampling.REFERENCE_METHOD_SEEDS["A"],
        },
        "method_digest": _digest("method-A"),
        "out": out,
        "contract_digest": _digest("contract"),
    }
    return submission, arguments


def _write_submission(runner, stage):
    stage.mkdir(parents=True, exist_ok=True)
    for name in runner.phase_three.SUBMISSION_FILES:
        (stage / name).write_text(f"{name}\n")


def test_paths_fail_closed_and_restart_content_is_immutable(monkeypatch, tmp_path):
    runner = _runner()
    with pytest.raises(runner.EvidenceBuildError, match="forbidden path"):
        runner._safe_path(tmp_path / "graded" / "evidence", "output", must_exist=False)

    out = runner._output_root(tmp_path / "evidence")
    path = out / "receipt.json"
    runner._write_json_once(out, path, {"value": 1}, "receipt")
    runner._write_json_once(out, path, {"value": 1}, "receipt")
    with pytest.raises(runner.EvidenceBuildError, match="differs"):
        runner._write_json_once(out, path, {"value": 2}, "receipt")

    packet = tmp_path / "source" / "qual-0"
    packet.mkdir(parents=True)
    overlapping = packet / "evidence"
    monkeypatch.setattr(
        runner,
        "_packet_roots",
        lambda development_root, qualification_root: ([], [packet]),
    )
    with pytest.raises(runner.EvidenceBuildError, match="must not overlap"):
        runner.build_evidence(tmp_path / "dev", tmp_path / "qual", overlapping)
    assert not overlapping.exists()


def test_lone_interrupted_evidence_contract_is_recovered(monkeypatch, tmp_path):
    runner = _runner()
    out = runner._output_root(tmp_path / "evidence")
    temporary = out / ".evidence_measurement_contract.json.tmp"
    temporary.write_text("{torn")
    monkeypatch.setattr(
        runner,
        "_contract_payload",
        lambda development, qualification, params: {"schema": "test"},
    )

    payload, digest = runner._bind_contract(
        out, [], [], runner.phase_three.MeasurementParams()
    )

    final = out / "evidence_measurement_contract.json"
    assert payload == {"schema": "test"}
    assert digest == runner._sha256(final)
    assert json.loads(final.read_text()) == payload
    assert not temporary.exists()


def test_interrupted_evidence_contract_with_companion_is_refused(
    monkeypatch, tmp_path
):
    runner = _runner()
    out = runner._output_root(tmp_path / "evidence")
    (out / ".evidence_measurement_contract.json.tmp").write_text("{torn")
    (out / "orphan.json").write_text("{}\n")
    monkeypatch.setattr(
        runner,
        "_contract_payload",
        lambda development, qualification, params: {"schema": "test"},
    )

    with pytest.raises(runner.EvidenceBuildError, match="nonempty evidence output"):
        runner._bind_contract(out, [], [], runner.phase_three.MeasurementParams())


def test_final_evidence_rejects_reduced_measurement_parameters(tmp_path):
    runner = _runner()
    cheap = runner.phase_three.MeasurementParams(
        bootstrap_replicates=10,
        bayesian_sweeps=40,
        simulation_paths=128,
        linkage_bootstraps=4,
    )

    with pytest.raises(runner.EvidenceBuildError, match="registered measurement"):
        runner.build_evidence(
            tmp_path / "dev", tmp_path / "qual", tmp_path / "evidence", cheap
        )

    assert not (tmp_path / "evidence").exists()


def test_phase_three_restart_discards_an_interrupted_stage(monkeypatch, tmp_path):
    runner = _runner()
    attempts = 0

    def method(stage):
        nonlocal attempts
        attempts += 1
        stage.mkdir(parents=True)
        (stage / "release.csv").write_text("partial\n")
        if attempts == 1:
            raise RuntimeError("injected method crash")
        for name in runner.phase_three.SUBMISSION_FILES:
            (stage / name).write_text(f"{name}\n")

    phase, submission, arguments = _phase_three_run(
        runner, monkeypatch, tmp_path, method
    )
    with pytest.raises(RuntimeError, match="injected method crash"):
        phase._run_once(*arguments)
    stage = submission.parent / ".A.phase-three-tmp"
    assert stage.is_dir()
    assert (submission.parent / ".A.run_intent.json").is_file()

    assert phase._run_once(*arguments) == {"scored": True}
    assert attempts == 2
    assert submission.is_dir()
    assert not stage.exists()
    assert not (submission.parent / ".A.run_intent.json").exists()


def test_phase_three_restart_receipts_a_published_submission(monkeypatch, tmp_path):
    runner = _runner()
    calls = 0

    def method(stage):
        nonlocal calls
        calls += 1
        _write_submission(runner, stage)

    phase, submission, arguments = _phase_three_run(
        runner, monkeypatch, tmp_path, method
    )
    write_json = phase._write_json_atomic
    crashed = False

    def crash_before_receipt(root, path, payload, label):
        nonlocal crashed
        if label == "method run receipt" and not crashed:
            crashed = True
            path.with_name(f".{path.name}.tmp").write_text("{")
            raise RuntimeError("injected receipt crash")
        return write_json(root, path, payload, label)

    monkeypatch.setattr(phase, "_write_json_atomic", crash_before_receipt)
    with pytest.raises(RuntimeError, match="injected receipt crash"):
        phase._run_once(*arguments)
    assert submission.is_dir()
    assert not (submission.parent / ".A.run_receipt.json").exists()
    assert (submission.parent / "..A.run_receipt.json.tmp").is_file()
    assert (submission.parent / ".A.run_intent.json").is_file()

    monkeypatch.setattr(phase, "_write_json_atomic", write_json)
    assert phase._run_once(*arguments) == {"scored": True}
    assert calls == 1
    assert (submission.parent / ".A.run_receipt.json").is_file()
    assert not (submission.parent / ".A.run_intent.json").exists()


def test_replicate_restart_discards_an_interrupted_stage(monkeypatch, tmp_path):
    runner = _runner()
    submission, arguments = _replicate_run(runner, tmp_path)
    attempts = 0

    def method(line, packet, stage, calibration_a, calibration_b, params):
        nonlocal attempts
        del line, packet, calibration_a, calibration_b, params
        attempts += 1
        (stage / "release.csv").write_text("partial\n")
        if attempts == 1:
            raise RuntimeError("injected replicate crash")
        _write_submission(runner, stage)

    monkeypatch.setattr(runner, "_run_reference_line", method)
    remove_tree = runner.shutil.rmtree
    monkeypatch.setattr(runner.shutil, "rmtree", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="injected replicate crash"):
        runner._replicate_submission_once(**arguments)
    stage = submission.parent / ".A.replicate-tmp"
    assert stage.is_dir()
    assert (submission.parent / ".A.run_intent.json").is_file()

    monkeypatch.setattr(runner.shutil, "rmtree", remove_tree)
    runner._replicate_submission_once(**arguments)
    assert attempts == 2
    assert submission.is_dir()
    assert not stage.exists()
    assert not (submission.parent / ".A.run_intent.json").exists()


def test_replicate_restart_receipts_a_published_submission(monkeypatch, tmp_path):
    runner = _runner()
    submission, arguments = _replicate_run(runner, tmp_path)
    calls = 0

    def method(line, packet, stage, calibration_a, calibration_b, params):
        nonlocal calls
        del line, packet, calibration_a, calibration_b, params
        calls += 1
        _write_submission(runner, stage)

    monkeypatch.setattr(runner, "_run_reference_line", method)
    write_json = runner._write_json_once
    crashed = False

    def crash_before_receipt(root, path, payload, label):
        nonlocal crashed
        if label == "replicate run receipt" and not crashed:
            crashed = True
            path.with_name(f".{path.name}.tmp").write_text("{")
            raise RuntimeError("injected replicate receipt crash")
        return write_json(root, path, payload, label)

    monkeypatch.setattr(runner, "_write_json_once", crash_before_receipt)
    with pytest.raises(RuntimeError, match="injected replicate receipt crash"):
        runner._replicate_submission_once(**arguments)
    assert submission.is_dir()
    assert not (submission.parent / ".A.run_receipt.json").exists()
    assert (submission.parent / "..A.run_receipt.json.tmp").is_file()
    assert (submission.parent / ".A.run_intent.json").is_file()

    monkeypatch.setattr(runner, "_write_json_once", write_json)
    receipt_path = runner._replicate_submission_once(**arguments)
    receipt = json.loads(receipt_path.read_text())
    assert calls == 1
    assert receipt["method_seed"] == runner.resampling.REFERENCE_METHOD_SEEDS["A"]
    assert receipt["run_spec"] == arguments["run_spec"]
    assert not (submission.parent / ".A.run_intent.json").exists()


def test_reference_runner_preserves_registered_method_seeds(monkeypatch, tmp_path):
    runner = _runner()
    observed = {}
    monkeypatch.setattr(
        runner.A, "run", lambda packet, out, params: observed.setdefault("A", params)
    )
    monkeypatch.setattr(
        runner.B, "run", lambda packet, out, params: observed.setdefault("B", params)
    )
    monkeypatch.setattr(
        runner.C, "run", lambda packet, out, params: observed.setdefault("C", params)
    )
    params = runner.phase_three.MeasurementParams(
        bootstrap_replicates=11,
        bayesian_sweeps=20,
        simulation_paths=32,
        linkage_bootstraps=7,
    )
    for line in runner.REFERENCE_LINES:
        runner._run_reference_line(
            line,
            tmp_path / "packet",
            tmp_path / line,
            tmp_path / "calibration-a.json",
            tmp_path / "calibration-b.json",
            params,
        )

    assert {line: observed[line].seed for line in runner.REFERENCE_LINES} \
        == runner.resampling.REFERENCE_METHOD_SEEDS
    assert observed["A"].bootstrap_replicates == 11
    assert observed["B"].sweeps == 20
    assert observed["B"].burn_in == 5
    assert observed["C"].linkage_bootstraps == 7


def test_elder_audit_is_normalized_to_c_and_wrapper_ids():
    runner = _runner()
    references = [
        {
            "reference_line": line,
            "world": f"qual-{world}",
            "method_digest_sha256": _digest(f"method-{line}"),
            "evidence_id": _digest(f"evidence-{line}-{world}"),
            "report": _elder_report(line, f"qual-{world}"),
        }
        for line in runner.REFERENCE_LINES
        for world in range(6)
    ]
    raw = {
        "schema": runner.freeze_v4_bars.ELDER_AUDIT_SCHEMA,
        "method_digest": {
            "git_commit": "abcdef0",
            "source_sha256": _digest("old-source"),
            "before_line": "A",
            "after_line": "third_cohort_component",
        },
        "shock_redraw": {"independent_per_member": False},
        "worlds": [{"world": f"qual-{world}"} for world in range(6)],
    }

    audit = runner._normalized_elder_audit(raw, references)

    assert audit["method_digest"]["before_line"] == "A"
    assert audit["method_digest"]["after_line"] == "C"
    assert audit["method_digest"]["source_sha256"] == _digest("method-C")
    assert audit["shock_redraw"]["independent_per_member"] is True
    for row in audit["worlds"]:
        world = row["world"]
        assert row["before_report_evidence_id"] == _digest(f"evidence-A-{world[-1]}")
        assert row["after_report_evidence_id"] == _digest(f"evidence-C-{world[-1]}")
    unsigned = dict(audit)
    recorded = unsigned.pop("digest_sha256")
    assert recorded == runner._canonical_digest(unsigned)


def test_mortality_audit_uses_measured_packet_values(monkeypatch, tmp_path):
    runner = _runner()
    packet = tmp_path / "qual-0"
    (packet / "participant").mkdir(parents=True)
    (packet / "manifest.json").write_text("manifest\n")
    (packet / "participant/contract.json").write_text(json.dumps({
        "shock_family": {"annual_rate": 0.20}
    }))
    decomposition = {
        "trend_active_during_public_experience_window": True,
        "trend_starts_only_after_public_window": False,
        "publication_lag_months": 12,
        "publication_lag_trend_factor": 0.97,
        "continuation_shocks_redrawn_per_member": True,
    }
    monkeypatch.setattr(
        runner.phase_three,
        "mortality_gap_decomposition",
        lambda current: dict(decomposition),
    )
    red_team_inputs = {
        name: _digest(name) for name in runner.freeze_v4_bars.RED_TEAM_INPUT_FILES
    }
    packet_inputs = dict(red_team_inputs)
    packet_inputs["retained/continuation_shock_redraw.json"] = _digest(
        "shock-runtime"
    )
    shock_report = _shock_report(packet_inputs)
    references = [
        {
            "reference_line": line,
            "world": "qual-0",
            "evidence_id": _digest(f"evidence-{line}"),
            "report": {
                "evidence": {"packet_file_sha256": packet_inputs},
                "continuation_shock_redraw_evidence": shock_report,
            },
        }
        for line in runner.REFERENCE_LINES
    ]

    audit = runner._mortality_identification_audit([packet], references)

    row = audit["worlds"][0]
    assert row["packet_manifest_digest_sha256"] == runner._sha256(
        packet / "manifest.json"
    )
    assert row["packet_input_sha256"] == red_team_inputs
    assert row["decomposition"] == decomposition
    assert row["reference_evidence_ids"] == {
        line: _digest(f"evidence-{line}") for line in runner.REFERENCE_LINES
    }
    assert audit["summary"]["publication_lag_trend_effect_percent_range"] \
        == pytest.approx([-3.0, -3.0])


def test_identifiability_audit_is_write_once_on_restart(monkeypatch, tmp_path):
    runner = _runner()
    out = runner._output_root(tmp_path / "evidence")
    revision = {"value": 1}

    def fake_run(command, **kwargs):
        receipt = Path(command[command.index("--receipt") + 1])
        receipt.write_text(json.dumps({
            "schema": runner.freeze_v4_bars.REGIME_IDENTIFIABILITY_SCHEMA,
            "revision": revision["value"],
        }))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    first, path = runner._identifiability_audit([], out)
    second, same_path = runner._identifiability_audit([], out)
    assert first == second
    assert path == same_path

    revision["value"] = 2
    with pytest.raises(runner.EvidenceBuildError, match="differs"):
        runner._identifiability_audit([], out)


def test_manifest_design_requires_exact_counts_and_paired_resamples():
    runner = _runner()
    manifest = {
        "reference_reports": [
            {
                "reference_line": line,
                "world": f"qual-{world}",
                "evidence_id": _digest(f"reference-{line}-{world}"),
            }
            for line in runner.REFERENCE_LINES
            for world in range(6)
        ],
        "replicate_reports": [
            {
                "reference_line": line,
                "world": f"qual-{world}",
                "replicate_id": f"outer-{replicate:03d}",
                "evidence_id": _digest(f"replicate-{line}-{world}-{replicate}"),
            }
            for world in range(6)
            for replicate in range(runner.REPLICATES_PER_WORLD)
            for line in runner.REFERENCE_LINES
        ],
        "control_reports": [
            {
                "control": name,
                "world": f"qual-{world}",
                "evidence_id": _digest(f"control-{name}-{world}"),
            }
            for name in runner.controls.QUALIFICATION_CONTROLS
            for world in range(6)
        ],
        "development_diagnostic_reports": [
            {
                "diagnostic": name,
                "world": f"dev-{world:02d}",
                "evidence_id": _digest(f"diagnostic-{name}-{world}"),
            }
            for name in runner.controls.DECOMPOSITION_CONTROLS
            for world in range(12)
        ],
    }
    runner._validate_manifest_design(manifest)

    manifest["replicate_reports"][0]["reference_line"] = "B"
    with pytest.raises(runner.EvidenceBuildError, match="not paired"):
        runner._validate_manifest_design(manifest)


def test_candidate_mismatch_stops_before_full_battery(monkeypatch, tmp_path):
    runner = _runner()
    development = [tmp_path / f"dev-{index:02d}" for index in range(12)]
    qualification = [tmp_path / f"qual-{index}" for index in range(6)]
    out = tmp_path / "evidence"
    monkeypatch.setattr(
        runner, "_packet_roots", lambda development_root, qualification_root: (
            development, qualification
        )
    )
    monkeypatch.setattr(
        runner,
        "_bind_contract",
        lambda output, dev, qual, params: (
            {"runner_digest_sha256": _digest("runner"), "source_sha256": {}},
            _digest("contract"),
        ),
    )
    monkeypatch.setattr(runner, "_run_reference_preflight", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_contract", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "_collect_reference_entries",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("full reference wrappers must not run")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_preflight_calibration_inputs",
        lambda *args: {"schema": "test", "entries": []},
    )
    compiled = float(runner.PacketParams().reserve_rate_per_person_year)
    monkeypatch.setattr(
        runner.calibrate_reserve_rate,
        "calibrate",
        lambda entries: {"candidate": True, "rate_per_person_year": compiled + 1.0},
    )
    called = {"full": False}

    def full_battery(*args, **kwargs):
        called["full"] = True
        raise AssertionError("full battery must not run")

    monkeypatch.setattr(runner.phase_three, "measure", full_battery)
    with pytest.raises(runner.EvidenceBuildError, match="differs from compiled"):
        runner.build_evidence(tmp_path / "dev", tmp_path / "qual", out)

    assert called["full"] is False
    candidate = json.loads(
        (out / "reserve_rate_preflight_candidate.json").read_text()
    )
    assert candidate["rate_per_person_year"] == compiled + 1.0


def test_references_only_mode_never_starts_full_battery(monkeypatch, tmp_path):
    runner = _runner()
    development = [tmp_path / f"dev-{index:02d}" for index in range(12)]
    qualification = [tmp_path / f"qual-{index}" for index in range(6)]
    references = [
        {
            "reference_line": line,
            "world": f"qual-{world}",
            "evidence_id": _digest(f"reference-{line}-{world}"),
            "report": {"reserve_rule_evidence": {
                "valid": True,
                "rate_per_person_year": float(
                    runner.PacketParams().reserve_rate_per_person_year
                ),
            }},
        }
        for line in runner.REFERENCE_LINES
        for world in range(6)
    ]
    monkeypatch.setattr(
        runner, "_packet_roots", lambda development_root, qualification_root: (
            development, qualification
        )
    )
    monkeypatch.setattr(
        runner,
        "_bind_contract",
        lambda output, dev, qual, params: (
            {"runner_digest_sha256": _digest("runner"), "source_sha256": {}},
            _digest("contract"),
        ),
    )
    monkeypatch.setattr(runner, "_run_reference_preflight", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_contract", lambda *args: None)
    monkeypatch.setattr(
        runner, "_collect_reference_entries", lambda *args: (references, {})
    )
    monkeypatch.setattr(
        runner,
        "_preflight_calibration_inputs",
        lambda *args: {"schema": "test", "entries": []},
    )
    monkeypatch.setattr(
        runner,
        "_calibration_inputs",
        lambda *args: {"schema": "test", "entries": []},
    )
    compiled = float(runner.PacketParams().reserve_rate_per_person_year)
    monkeypatch.setattr(
        runner.calibrate_reserve_rate,
        "calibrate",
        lambda entries: {"candidate": True, "rate_per_person_year": compiled},
    )
    monkeypatch.setattr(
        runner, "_mortality_identification_audit", lambda *args: {"schema": "test"}
    )
    identifiability_path = tmp_path / "evidence/regime_identifiability_audit.json"
    monkeypatch.setattr(
        runner,
        "_identifiability_audit",
        lambda *args: ({"schema": "test"}, identifiability_path),
    )
    monkeypatch.setattr(
        runner, "_validate_reference_preflight_audits", lambda *args: None
    )
    monkeypatch.setattr(
        runner.phase_three,
        "measure",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full battery must not run")
        ),
    )

    result = runner.build_evidence(
        tmp_path / "dev", tmp_path / "qual", tmp_path / "evidence",
        references_only=True,
    )

    assert result["reference_report_count"] == 18
    assert result["full_battery_authorized"] is True
