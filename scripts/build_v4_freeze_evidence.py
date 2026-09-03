"""Build the complete replay-bound V4 qualification evidence manifest.

The runner consumes one canonical ``worlds-p4`` development/qualification pair.
It reuses the phase-three methods for the 18 final references, 132 registered
qualification controls, and 24 development diagnostics, then runs the same three
fixed-seed reference lines on seventeen paired participant-data resamples per
qualification world.  Every submission, verifier report, audit, and input is bound
to a restart-safe measurement contract before one freeze evidence manifest is
published.

This runner never opens a graded path and never changes packet bytes.  If the reserve
rate selected from the final references differs from the public rate already frozen
into the input packets, it writes the candidate audit and stops: rebuilding packets
at that rate is a generator operation and requires a second evidence pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import bayesian as B
from meridia.methods import controls
from meridia.methods import design_based as A
from meridia.methods import phase_three
from meridia.methods import resampling
from meridia.methods import third_reference as C
from meridia.packet import PacketParams
from meridia.verify import verify_submission

import calibrate_reserve_rate
import freeze_v4_bars
import red_team_reserve_total


PIPELINE_SCHEMA = "meridia.v4.freeze-evidence-run.v1"
REPORT_RECEIPT_SCHEMA = "meridia.v4.freeze-evidence-report-receipt.v1"
REPLICATE_RUN_RECEIPT_SCHEMA = "meridia.v4.outer-reference-run-receipt.v1"
REPLICATE_RUN_INTENT_SCHEMA = "meridia.v4.outer-reference-run-intent.v1"
REFERENCE_LINES = ("A", "B", "C")
REPLICATES_PER_WORLD = 17
OUTER_RESAMPLE_SEED = 20260906
FORBIDDEN_PATH_FRAGMENTS = ("graded", "sealed", "hidden")
EXPECTED_COUNTS = {
    "reference_reports": 18,
    "replicate_reports": 306,
    "control_reports": 132,
    "development_diagnostic_reports": 24,
}
MORTALITY_AUDIT_SCHEMA = "meridia.v4.mortality-identification-audit.v1"


class EvidenceBuildError(ValueError):
    """The requested evidence run is unsafe, incomplete, or not replayable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceBuildError("evidence must be finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _json_copy(value: object) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise EvidenceBuildError("evidence must be finite JSON") from exc


def _safe_path(path: Path, label: str, *, must_exist: bool = True) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve(strict=False)
    for part in (*candidate.absolute().parts, *resolved.parts):
        lowered = part.casefold()
        if any(fragment in lowered for fragment in FORBIDDEN_PATH_FRAGMENTS):
            raise EvidenceBuildError(f"{label} contains a forbidden path component")
    if candidate.is_symlink() or resolved.is_symlink():
        raise EvidenceBuildError(f"{label} may not be a symbolic link")
    if must_exist and not resolved.exists():
        raise EvidenceBuildError(f"{label} does not exist")
    return resolved


def _output_root(path: Path) -> Path:
    out = _safe_path(path, "evidence output", must_exist=False)
    if out.exists() and not out.is_dir():
        raise EvidenceBuildError("evidence output must be a directory")
    out.mkdir(parents=True, exist_ok=True)
    if out.is_symlink():
        raise EvidenceBuildError("evidence output may not be a symbolic link")
    return out


def _under(root: Path, path: Path, label: str) -> Path:
    root = Path(root).resolve()
    candidate = Path(path).absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise EvidenceBuildError(f"{label} escapes the evidence output") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvidenceBuildError(f"{label} may not use symbolic links")
    return candidate


def _write_json_once(root: Path, path: Path, payload: Mapping[str, Any], label: str) -> Path:
    path = _under(root, path, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_text() != encoded:
            raise EvidenceBuildError(f"existing {label} differs from this evidence run")
        return path
    temporary = _under(root, path.with_name(f".{path.name}.tmp"), label)
    if temporary.exists() or temporary.is_symlink():
        if temporary.is_symlink() or not temporary.is_file():
            raise EvidenceBuildError(f"partial {label} is not a regular file")
        # No final path exists, so downstream publication has not committed this JSON.
        # A torn temporary is safe to discard and regenerate from the bound payload.
        temporary.unlink()
    temporary.write_text(encoded)
    temporary.replace(path)
    return path


def _read_matching_json(
    path: Path, expected: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceBuildError(f"{label} must be a regular file")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBuildError(f"{label} is invalid") from exc
    if not isinstance(payload, dict) or payload != expected:
        raise EvidenceBuildError(f"{label} differs on restart")
    return payload


def _clear_replicate_intent(path: Path) -> None:
    """Separate final step so crash injection can exercise committed receipts."""
    path.unlink()


def _packet_roots(
    development_root: Path, qualification_root: Path
) -> tuple[list[Path], list[Path]]:
    development_root = _safe_path(development_root, "development root")
    qualification_root = _safe_path(qualification_root, "qualification root")
    development = [
        development_root / f"dev-{index:02d}" for index in range(12)
    ]
    qualification = [
        qualification_root / f"qual-{index}" for index in range(6)
    ]
    try:
        development = phase_three._validate_packet_group(development, 12, True)
        qualification = phase_three._validate_packet_group(qualification, 6, False)
        phase_three._validate_shared_worlds_root(development, qualification)
    except ValueError as exc:
        raise EvidenceBuildError(str(exc)) from exc
    return development, qualification


def _source_hashes() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[1]
    sources = list((repo / "meridia").rglob("*.py"))
    sources.extend(
        repo / "scripts" / name
        for name in (
            "build_v4_freeze_evidence.py",
            "calibrate_reserve_rate.py",
            "freeze_v4_bars.py",
            "identifiability_v4.py",
            "red_team_reserve_total.py",
        )
    )
    result = {}
    for path in sorted(set(sources)):
        if path.is_symlink() or not path.is_file():
            raise EvidenceBuildError(f"evidence source is missing or linked: {path.name}")
        result[path.relative_to(repo).as_posix()] = _sha256(path)
    return result


def _contract_payload(
    development: Sequence[Path],
    qualification: Sequence[Path],
    params: phase_three.MeasurementParams,
) -> dict[str, Any]:
    source_sha256 = _source_hashes()
    packet_files = {}
    packet_manifests = {}
    for packet in (*development, *qualification):
        try:
            packet_files[str(packet)] = phase_three._verified_packet_files(packet)
        except ValueError as exc:
            raise EvidenceBuildError(str(exc)) from exc
        packet_manifests[str(packet)] = _sha256(packet / "manifest.json")
    runner_digest = _canonical_digest(
        {"schema": PIPELINE_SCHEMA, "source_sha256": source_sha256}
    )
    return {
        "schema": PIPELINE_SCHEMA,
        "development_packets": [str(path) for path in development],
        "qualification_packets": [str(path) for path in qualification],
        "packet_file_sha256": packet_files,
        "packet_manifest_sha256": packet_manifests,
        "params": {
            "bootstrap_replicates": params.bootstrap_replicates,
            "bayesian_sweeps": params.bayesian_sweeps,
            "simulation_paths": params.simulation_paths,
            "linkage_bootstraps": params.linkage_bootstraps,
        },
        "outer_resampling": {
            "schema": resampling.OUTER_RESAMPLE_SCHEMA,
            "outer_seed": OUTER_RESAMPLE_SEED,
            "replicates_per_world": REPLICATES_PER_WORLD,
            "reference_lines": list(REFERENCE_LINES),
            "method_seeds": dict(resampling.REFERENCE_METHOD_SEEDS),
            "method_seeds_fixed_across_outer_resamples": True,
        },
        "registered_controls": list(controls.QUALIFICATION_CONTROLS),
        "development_diagnostics": list(controls.DECOMPOSITION_CONTROLS),
        "expected_report_counts": dict(EXPECTED_COUNTS),
        "source_sha256": source_sha256,
        "runner_digest_sha256": runner_digest,
    }


def _bind_contract(
    out: Path,
    development: Sequence[Path],
    qualification: Sequence[Path],
    params: phase_three.MeasurementParams,
) -> tuple[dict[str, Any], str]:
    payload = _contract_payload(development, qualification, params)
    path = out / "evidence_measurement_contract.json"
    temporary = _under(
        out,
        path.with_name(f".{path.name}.tmp"),
        "evidence measurement contract",
    )
    if path.exists() and (temporary.exists() or temporary.is_symlink()):
        raise EvidenceBuildError(
            "partial evidence measurement contract accompanies final contract"
        )
    if not path.exists():
        entries = list(out.iterdir())
        if entries and (
            entries != [temporary]
            or temporary.is_symlink()
            or not temporary.is_file()
        ):
            raise EvidenceBuildError(
                "nonempty evidence output has no evidence_measurement_contract.json"
            )
    _write_json_once(out, path, payload, "evidence measurement contract")
    return payload, _sha256(path)


def _verify_contract(
    out: Path,
    expected: Mapping[str, Any],
    development: Sequence[Path],
    qualification: Sequence[Path],
    params: phase_three.MeasurementParams,
) -> None:
    current = _contract_payload(development, qualification, params)
    if current != expected:
        raise EvidenceBuildError("evidence inputs or source changed during the run")
    path = out / "evidence_measurement_contract.json"
    if path.is_symlink() or not path.is_file():
        raise EvidenceBuildError("evidence measurement contract is missing or linked")
    try:
        recorded = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBuildError("evidence measurement contract is invalid") from exc
    if recorded != expected:
        raise EvidenceBuildError("evidence measurement contract changed during the run")


def _phase_receipt(phase_out: Path, submission: Path) -> tuple[dict[str, Any], Path]:
    path = phase_three._run_receipt_path(phase_out, submission)
    if path.is_symlink() or not path.is_file():
        raise EvidenceBuildError(f"phase-three run receipt is missing for {submission}")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBuildError("phase-three run receipt is invalid") from exc
    if not isinstance(receipt, dict) or receipt.get("schema") != phase_three.RUN_RECEIPT_SCHEMA:
        raise EvidenceBuildError("phase-three run receipt schema differs")
    if Path(str(receipt.get("submission"))).resolve() != submission.resolve():
        raise EvidenceBuildError("phase-three run receipt names a different submission")
    try:
        output_sha256 = phase_three._submission_hashes(submission)
    except ValueError as exc:
        raise EvidenceBuildError(str(exc)) from exc
    if receipt.get("output_sha256") != output_sha256:
        raise EvidenceBuildError("phase-three run receipt output digest differs")
    run_spec = receipt.get("run_spec")
    if not isinstance(run_spec, dict):
        raise EvidenceBuildError("phase-three run receipt has no run specification")
    return receipt, path


def _method_digest(
    identity: str, run_spec: Mapping[str, Any], contract: Mapping[str, Any]
) -> str:
    return _canonical_digest(
        {
            "identity": identity,
            "run_spec": run_spec,
            "implementation_source_sha256": contract["source_sha256"],
        }
    )


def _report_once(
    *,
    kind: str,
    identity_field: str,
    identity: str,
    world: str,
    packet: Path,
    submission: Path,
    method_receipt_path: Path,
    run_spec: Mapping[str, Any],
    out: Path,
    contract: Mapping[str, Any],
    contract_digest: str,
    replicate_id: str | None = None,
    resample_digest: str | None = None,
    resampling_design: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        method_receipt = json.loads(method_receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBuildError("method run receipt is invalid") from exc
    expected_output_sha256 = method_receipt.get("output_sha256") \
        if isinstance(method_receipt, Mapping) else None
    method_receipt_digest = _sha256(method_receipt_path)
    try:
        report = phase_three._json_safe(verify_submission(packet, submission, None))
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceBuildError(
            f"{world}/{identity}: verifier could not produce freeze evidence"
        ) from exc
    if not isinstance(report, dict):
        raise EvidenceBuildError("verifier report is not an object")
    report_submission_sha256 = report.get("evidence", {}).get(
        "submission_file_sha256"
    )
    try:
        current_output_sha256 = phase_three._submission_hashes(submission)
    except ValueError as exc:
        raise EvidenceBuildError(str(exc)) from exc
    if not isinstance(expected_output_sha256, Mapping) \
            or dict(expected_output_sha256) != current_output_sha256 \
            or report_submission_sha256 != current_output_sha256 \
            or _sha256(method_receipt_path) != method_receipt_digest:
        raise EvidenceBuildError(
            f"{world}/{identity}: submission or method receipt changed during scoring"
        )
    relative = Path(kind) / identity / world
    if replicate_id is not None:
        relative /= replicate_id
    report_path = out / "reports" / relative.with_suffix(".json")
    receipt_path = out / "report_receipts" / relative.with_suffix(".json")
    _write_json_once(out, report_path, report, "verifier report")
    method_digest = _method_digest(identity, run_spec, contract)
    receipt: dict[str, Any] = {
        "schema": REPORT_RECEIPT_SCHEMA,
        "kind": kind,
        identity_field: identity,
        "world": world,
        "evidence_measurement_contract_sha256": contract_digest,
        "runner_digest_sha256": contract["runner_digest_sha256"],
        "measurement_params": _json_copy(contract["params"]),
        "method_digest_sha256": method_digest,
        "method_run_receipt_sha256": method_receipt_digest,
        "method_run_spec": _json_copy(run_spec),
        "verifier_report_sha256": _sha256(report_path),
        "packet_digest_sha256": report.get("evidence", {}).get(
            "packet_digest_sha256"
        ),
        "submission_digest_sha256": report.get("evidence", {}).get(
            "submission_digest_sha256"
        ),
    }
    if replicate_id is not None:
        receipt.update(
            {
                "replicate_id": replicate_id,
                "resample_digest_sha256": resample_digest,
                "resampling_design": _json_copy(resampling_design),
            }
        )
    _write_json_once(out, receipt_path, receipt, "verifier report receipt")
    entry: dict[str, Any] = {
        identity_field: identity,
        "world": world,
        "method_digest_sha256": method_digest,
        "runner_digest_sha256": contract["runner_digest_sha256"],
        "measurement_contract_digest_sha256": contract_digest,
        "measurement_params": _json_copy(contract["params"]),
        "run_receipt_digest_sha256": _sha256(receipt_path),
        "deterministic": True,
        "report": report,
    }
    if replicate_id is not None:
        entry.update(
            {
                "replicate_id": replicate_id,
                "resample_digest_sha256": resample_digest,
                "resampling_design": _json_copy(resampling_design),
            }
        )
    entry["evidence_id"] = freeze_v4_bars.evidence_id_for(entry, kind=kind)
    return entry


def _base_submission_path(phase_out: Path, world: str, kind: str, identity: str) -> Path:
    if kind == "reference":
        name = "third" if identity == "C" else identity
        return phase_out / "qualification" / world / name
    if kind == "diagnostic":
        return phase_out / "development" / world / identity
    collection = "controls" if identity in controls.ALL_CONTROLS else "deletions"
    return phase_out / "qualification" / world / collection / identity


def _collect_base_entries(
    development: Sequence[Path],
    qualification: Sequence[Path],
    phase_out: Path,
    out: Path,
    contract: Mapping[str, Any],
    contract_digest: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "reference_reports": [],
        "control_reports": [],
        "development_diagnostic_reports": [],
    }
    specs: dict[str, dict[str, Any]] = {}

    def add(kind: str, identity: str, packet: Path) -> None:
        identity_field = {
            "reference": "reference_line",
            "control": "control",
            "diagnostic": "diagnostic",
        }[kind]
        submission = _base_submission_path(phase_out, packet.name, kind, identity)
        receipt, receipt_path = _phase_receipt(phase_out, submission)
        run_spec = receipt["run_spec"]
        spec_key = f"{kind}:{identity}"
        previous = specs.setdefault(spec_key, _json_copy(run_spec))
        if previous != run_spec:
            raise EvidenceBuildError(f"{spec_key} changes run specification across worlds")
        entry = _report_once(
            kind=kind,
            identity_field=identity_field,
            identity=identity,
            world=packet.name,
            packet=packet,
            submission=submission,
            method_receipt_path=receipt_path,
            run_spec=run_spec,
            out=out,
            contract=contract,
            contract_digest=contract_digest,
        )
        groups[f"{kind}_reports"].append(entry)

    for line in REFERENCE_LINES:
        for packet in qualification:
            add("reference", line, packet)
    for name in controls.QUALIFICATION_CONTROLS:
        for packet in qualification:
            add("control", name, packet)
    for name in controls.DECOMPOSITION_CONTROLS:
        for packet in development:
            add("diagnostic", name, packet)
    return groups, specs


def _reference_run_specs(
    params: phase_three.MeasurementParams,
    calibration_a_sha256: str,
    calibration_b_sha256: str,
) -> dict[str, dict[str, Any]]:
    return {
        "A": {
            "method": "A",
            "bootstrap_replicates": params.bootstrap_replicates,
            "method_seed": resampling.REFERENCE_METHOD_SEEDS["A"],
            "simulation_paths": params.simulation_paths,
            "simulation_seed": phase_three.SHARED_SIMULATION_SEED,
            "actuarial_layer_seed": phase_three.SHARED_ACTUARIAL_LAYER_SEED,
            "calibration_sha256": calibration_a_sha256,
        },
        "B": {
            "method": "B",
            "sweeps": params.bayesian_sweeps,
            "burn_in": params.bayesian_sweeps // 4,
            "method_seed": resampling.REFERENCE_METHOD_SEEDS["B"],
            "simulation_paths": params.simulation_paths,
            "simulation_seed": phase_three.SHARED_SIMULATION_SEED,
            "actuarial_layer_seed": phase_three.SHARED_ACTUARIAL_LAYER_SEED,
            "calibration_sha256": calibration_b_sha256,
        },
        "C": {
            "method": "third_cohort_component",
            "bootstrap_replicates": params.bootstrap_replicates,
            "linkage_bootstraps": params.linkage_bootstraps,
            "method_seed": resampling.REFERENCE_METHOD_SEEDS["C"],
            "simulation_paths": params.simulation_paths,
            "calibration_sha256": calibration_a_sha256,
        },
    }


def _run_reference_preflight(
    development: Sequence[Path],
    qualification: Sequence[Path],
    phase_out: Path,
    params: phase_three.MeasurementParams,
) -> None:
    """Run only the 18 references, reusing the exact full phase-three receipts."""

    phase_contract = phase_three._measurement_contract(
        list(development), list(qualification), None, params, raw_pre_freeze=True
    )
    phase_three._bind_measurement_output(phase_out, phase_contract)
    phase_contract_digest = phase_three._measurement_contract_sha256(phase_out)
    phase_three._verify_bound_packet_group(
        phase_out, phase_contract_digest, [*development, *qualification]
    )
    calibration_a = phase_three._ensure_bound_calibration_artifact(
        phase_out,
        "A",
        lambda path: A.calibrate(list(development), path),
        phase_contract_digest,
        list(development),
    )
    calibration_b = phase_three._ensure_bound_calibration_artifact(
        phase_out,
        "B",
        lambda path: B.calibrate(list(development), path),
        phase_contract_digest,
        list(development),
    )
    calibration_a_sha256 = phase_three._bound_calibration_sha256(
        phase_out, phase_contract_digest, "A", calibration_a
    )
    calibration_b_sha256 = phase_three._bound_calibration_sha256(
        phase_out, phase_contract_digest, "B", calibration_b
    )
    specs = _reference_run_specs(
        params, calibration_a_sha256, calibration_b_sha256
    )
    for packet in qualification:
        for line in REFERENCE_LINES:
            submission = _base_submission_path(
                phase_out, packet.name, "reference", line
            )
            calibration = calibration_b if line == "B" else calibration_a
            phase_three._run_once(
                packet,
                submission,
                lambda stage, current=line, source=packet: _run_reference_line(
                    current,
                    source,
                    stage,
                    calibration_a,
                    calibration_b,
                    params,
                ),
                None,
                True,
                phase_out,
                phase_contract_digest,
                specs[line],
                {
                    calibration: (
                        calibration_b_sha256 if line == "B" else calibration_a_sha256
                    )
                },
            )
    phase_three._verify_bound_packet_group(
        phase_out, phase_contract_digest, [*development, *qualification]
    )


def _collect_reference_entries(
    qualification: Sequence[Path],
    phase_out: Path,
    out: Path,
    contract: Mapping[str, Any],
    contract_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entries = []
    specs = {}
    for line in REFERENCE_LINES:
        for packet in qualification:
            submission = _base_submission_path(
                phase_out, packet.name, "reference", line
            )
            receipt, receipt_path = _phase_receipt(phase_out, submission)
            run_spec = receipt["run_spec"]
            previous = specs.setdefault(f"reference:{line}", _json_copy(run_spec))
            if previous != run_spec:
                raise EvidenceBuildError(
                    f"reference:{line} changes run specification across worlds"
                )
            entries.append(
                _report_once(
                    kind="reference",
                    identity_field="reference_line",
                    identity=line,
                    world=packet.name,
                    packet=packet,
                    submission=submission,
                    method_receipt_path=receipt_path,
                    run_spec=run_spec,
                    out=out,
                    contract=contract,
                    contract_digest=contract_digest,
                )
            )
    return entries, specs


def _tree_inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise EvidenceBuildError(f"input tree is missing or linked: {root.name}")
    rows = list(root.rglob("*"))
    linked = sorted(str(path.relative_to(root)) for path in rows if path.is_symlink())
    if linked:
        raise EvidenceBuildError(f"input tree contains linked paths: {linked}")
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(rows)
        if path.is_file()
    }


def _materialize_retained_view(base_packet: Path, resample_packet: Path) -> dict[str, str]:
    source = base_packet / "retained"
    target = resample_packet / "retained"
    expected = _tree_inventory(source)
    if target.exists():
        if _tree_inventory(target) != expected:
            raise EvidenceBuildError("resample retained view differs from the source packet")
        return expected
    staging = resample_packet / ".retained-view-tmp"
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not staging.is_dir():
            raise EvidenceBuildError("partial resample retained view is not a directory")
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for name in expected:
            source_file = source / name
            target_file = staging / name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source_file, target_file)
            except OSError:
                shutil.copy2(source_file, target_file)
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if _tree_inventory(target) != expected:
        raise EvidenceBuildError("resample retained view failed byte verification")
    return expected


def _resampling_design() -> dict[str, Any]:
    return {
        "schema": resampling.OUTER_RESAMPLE_SCHEMA,
        "outer_seed": OUTER_RESAMPLE_SEED,
        "replicates_per_world": REPLICATES_PER_WORLD,
        "reference_lines": list(REFERENCE_LINES),
        "method_seeds": dict(resampling.REFERENCE_METHOD_SEEDS),
        "method_seeds_fixed_across_outer_resamples": True,
        "survey_files": list(resampling.SURVEY_FILES),
        "experience_count_columns": list(resampling.EXPERIENCE_COUNT_COLUMNS),
    }


def _run_reference_line(
    line: str,
    packet: Path,
    submission: Path,
    calibration_a: Path,
    calibration_b: Path,
    params: phase_three.MeasurementParams,
) -> None:
    if line == "A":
        A.run(
            packet,
            submission,
            A.MethodParams(
                bootstrap_replicates=params.bootstrap_replicates,
                seed=resampling.REFERENCE_METHOD_SEEDS["A"],
                calibration_path=str(calibration_a),
                actuarial="on",
                actuarial_params=phase_three._shared_reference_layer(
                    params.simulation_paths
                ),
            ),
        )
    elif line == "B":
        B.run(
            packet,
            submission,
            B.MethodParams(
                sweeps=params.bayesian_sweeps,
                burn_in=params.bayesian_sweeps // 4,
                seed=resampling.REFERENCE_METHOD_SEEDS["B"],
                calibration_path=str(calibration_b),
                actuarial="on",
                actuarial_params=phase_three._shared_reference_layer(
                    params.simulation_paths
                ),
            ),
        )
    elif line == "C":
        C.run(
            packet,
            submission,
            C.ThirdReferenceParams(
                bootstrap_replicates=params.bootstrap_replicates,
                linkage_bootstraps=params.linkage_bootstraps,
                simulation_paths=params.simulation_paths,
                seed=resampling.REFERENCE_METHOD_SEEDS["C"],
                calibration_path=str(calibration_a),
            ),
        )
    else:
        raise EvidenceBuildError(f"unknown reference line {line}")


def _replicate_submission_once(
    *,
    line: str,
    base_packet: Path,
    resample_packet: Path,
    resample_row: Mapping[str, Any],
    resampling_manifest_path: Path,
    submission: Path,
    calibration_a: Path,
    calibration_b: Path,
    params: phase_three.MeasurementParams,
    run_spec: Mapping[str, Any],
    method_digest: str,
    out: Path,
    contract_digest: str,
) -> Path:
    receipt_path = submission.parent / f".{submission.name}.run_receipt.json"
    intent_path = submission.parent / f".{submission.name}.run_intent.json"
    stage = submission.parent / f".{submission.name}.replicate-tmp"
    participant_inventory = _tree_inventory(resample_packet / "participant")
    if _canonical_digest(
        {
            name: {
                "bytes": (resample_packet / "participant" / name).stat().st_size,
                "sha256": digest,
            }
            for name, digest in participant_inventory.items()
        }
    ) != resample_row.get("participant_digest_sha256"):
        raise EvidenceBuildError("resample participant digest differs from its manifest")
    retained_inventory = _materialize_retained_view(base_packet, resample_packet)
    base_manifest_sha256 = _sha256(base_packet / "manifest.json")
    resampling_manifest_sha256 = _sha256(resampling_manifest_path)
    calibration_sha256 = {
        "A": _sha256(calibration_a),
        "B": _sha256(calibration_b),
    }

    publication_binding: dict[str, Any] = {
        "evidence_measurement_contract_sha256": contract_digest,
        "reference_line": line,
        "world": base_packet.name,
        "replicate_id": resample_row["replicate_id"],
        "method_seed": resampling.REFERENCE_METHOD_SEEDS[line],
        "method_digest_sha256": method_digest,
        "run_spec": _json_copy(run_spec),
        "base_packet_manifest_sha256": base_manifest_sha256,
        "resampling_manifest_sha256": resampling_manifest_sha256,
        "resample_participant_digest_sha256": resample_row[
            "participant_digest_sha256"
        ],
        "retained_file_sha256": retained_inventory,
        "calibration_sha256": calibration_sha256,
        "submission": str(submission.resolve()),
    }
    expected_intent = {
        "schema": REPLICATE_RUN_INTENT_SCHEMA,
        **publication_binding,
    }

    def expected_receipt() -> dict[str, Any]:
        return {
            "schema": REPLICATE_RUN_RECEIPT_SCHEMA,
            **publication_binding,
            "output_sha256": phase_three._submission_hashes(submission),
        }

    submission.parent.mkdir(parents=True, exist_ok=True)
    def present(path: Path) -> bool:
        return path.exists() or path.is_symlink()

    intent_temporary = intent_path.with_name(f".{intent_path.name}.tmp")
    receipt_temporary = receipt_path.with_name(f".{receipt_path.name}.tmp")

    if present(receipt_path):
        if submission.is_symlink() or not submission.is_dir():
            raise EvidenceBuildError("replicate submission is not a real directory")
        if present(stage) or present(receipt_temporary):
            raise EvidenceBuildError(
                "completed replicate run has ambiguous partial artifacts"
            )
        _read_matching_json(receipt_path, expected_receipt(), "replicate run receipt")
        if present(intent_path):
            _read_matching_json(intent_path, expected_intent, "replicate run intent")
            _clear_replicate_intent(intent_path)
        return receipt_path

    if not present(intent_path):
        if present(intent_temporary) and any(
            present(path)
            for path in (stage, submission, receipt_path, receipt_temporary)
        ):
            raise EvidenceBuildError(
                "partial replicate run intent has ambiguous companion artifacts"
            )
        if present(submission):
            raise EvidenceBuildError(
                "replicate submission exists without its run intent or receipt"
            )
        if present(stage) or present(receipt_temporary):
            raise EvidenceBuildError("unbound partial replicate run is present")
        _write_json_once(out, intent_path, expected_intent, "replicate run intent")
    _read_matching_json(intent_path, expected_intent, "replicate run intent")

    if present(submission):
        if present(stage):
            raise EvidenceBuildError(
                "replicate run has both staged and published submissions"
            )
        _write_json_once(
            out, receipt_path, expected_receipt(), "replicate run receipt"
        )
        _clear_replicate_intent(intent_path)
        return receipt_path

    if present(receipt_temporary):
        raise EvidenceBuildError(
            "replicate receipt was staged without a published submission"
        )
    if present(stage):
        if stage.is_symlink() or not stage.is_dir():
            raise EvidenceBuildError("partial replicate submission is not a directory")
        shutil.rmtree(stage)
    stage.mkdir(mode=0o700)
    try:
        _run_reference_line(
            line, resample_packet, stage, calibration_a, calibration_b, params
        )
        phase_three._submission_hashes(stage)
        if _tree_inventory(resample_packet / "participant") != participant_inventory \
                or _tree_inventory(resample_packet / "retained") != retained_inventory \
                or _sha256(base_packet / "manifest.json") != base_manifest_sha256 \
                or _sha256(resampling_manifest_path) != resampling_manifest_sha256 \
                or {"A": _sha256(calibration_a), "B": _sha256(calibration_b)} \
                != calibration_sha256:
            raise EvidenceBuildError("resample input changed during the method run")
        stage.replace(submission)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    _write_json_once(out, receipt_path, expected_receipt(), "replicate run receipt")
    _read_matching_json(receipt_path, expected_receipt(), "replicate run receipt")
    _clear_replicate_intent(intent_path)
    return receipt_path


def _collect_replicate_entries(
    qualification: Sequence[Path],
    phase_out: Path,
    out: Path,
    contract: Mapping[str, Any],
    contract_digest: str,
    params: phase_three.MeasurementParams,
    specs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    calibration_a = phase_out / "calibration_A.json"
    calibration_b = phase_out / "calibration_B.json"
    if any(path.is_symlink() or not path.is_file() for path in (calibration_a, calibration_b)):
        raise EvidenceBuildError("phase-three calibration artifacts are missing")
    design = _resampling_design()
    entries = []
    for base_packet in qualification:
        resample_root = out / "outer_resamples" / base_packet.name
        manifest = resampling.materialize_paired_outer_resamples(
            base_packet,
            resample_root,
            replicates=REPLICATES_PER_WORLD,
            seed=OUTER_RESAMPLE_SEED,
            method_seeds=resampling.REFERENCE_METHOD_SEEDS,
        )
        if manifest.get("method_seeds") != resampling.REFERENCE_METHOD_SEEDS:
            raise EvidenceBuildError("outer resample method seeds differ from the register")
        manifest_path = resample_root / "manifest.json"
        for row in manifest["resamples"]:
            replicate_id = str(row["replicate_id"])
            resample_packet = Path(str(row["packet"])).resolve()
            _materialize_retained_view(base_packet, resample_packet)
            for line in REFERENCE_LINES:
                spec = specs[f"reference:{line}"]
                method_digest = _method_digest(line, spec, contract)
                submission = (
                    out
                    / "replicate_submissions"
                    / base_packet.name
                    / replicate_id
                    / line
                )
                method_receipt = _replicate_submission_once(
                    line=line,
                    base_packet=base_packet,
                    resample_packet=resample_packet,
                    resample_row=row,
                    resampling_manifest_path=manifest_path,
                    submission=submission,
                    calibration_a=calibration_a,
                    calibration_b=calibration_b,
                    params=params,
                    run_spec=spec,
                    method_digest=method_digest,
                    out=out,
                    contract_digest=contract_digest,
                )
                entries.append(
                    _report_once(
                        kind="replicate",
                        identity_field="reference_line",
                        identity=line,
                        world=base_packet.name,
                        packet=resample_packet,
                        submission=submission,
                        method_receipt_path=method_receipt,
                        run_spec=spec,
                        out=out,
                        contract=contract,
                        contract_digest=contract_digest,
                        replicate_id=replicate_id,
                        resample_digest=str(row["participant_digest_sha256"]),
                        resampling_design=design,
                    )
                )
        resampling.verify_paired_outer_resamples(resample_root)
    return entries


def _calibration_inputs(
    references: Sequence[Mapping[str, Any]],
    qualification: Sequence[Path],
    phase_out: Path,
) -> dict[str, Any]:
    packets = {packet.name: packet for packet in qualification}
    rows = []
    for entry in references:
        line = str(entry["reference_line"])
        world = str(entry["world"])
        rows.append(
            {
                "reference_line": line,
                "world": world,
                "evidence_id": entry["evidence_id"],
                "deterministic": True,
                "packet_dir": str(packets[world]),
                "submission_dir": str(
                    _base_submission_path(phase_out, world, "reference", line)
                ),
            }
        )
    return {
        "schema": calibrate_reserve_rate.MANIFEST_SCHEMA,
        "entries": sorted(rows, key=lambda row: (row["reference_line"], row["world"])),
    }


def _preflight_calibration_inputs(
    qualification: Sequence[Path], phase_out: Path
) -> dict[str, Any]:
    """Bind rate-only preflight rows to the phase-three method receipts.

    This path deliberately does not require reserve feasibility: if the compiled total
    is too small for a reference's submitted q95 values, the candidate rate must still
    be recoverable so the generator can rebuild the packets.  A candidate that matches
    the compiled rate is regenerated below with the full verifier wrapper IDs before it
    can enter the freeze manifest.
    """

    rows = []
    for line in REFERENCE_LINES:
        for packet in qualification:
            submission = _base_submission_path(
                phase_out, packet.name, "reference", line
            )
            receipt, receipt_path = _phase_receipt(phase_out, submission)
            evidence_id = _canonical_digest({
                "schema": "meridia.v4.reserve-rate-preflight-evidence.v1",
                "reference_line": line,
                "world": packet.name,
                "packet_manifest_sha256": _sha256(packet / "manifest.json"),
                "method_run_receipt_sha256": _sha256(receipt_path),
                "method_output_sha256": receipt["output_sha256"],
            })
            rows.append({
                "reference_line": line,
                "world": packet.name,
                "evidence_id": evidence_id,
                "deterministic": True,
                "packet_dir": str(packet),
                "submission_dir": str(submission),
            })
    return {
        "schema": calibrate_reserve_rate.MANIFEST_SCHEMA,
        "entries": rows,
    }


def _require_packets_at_candidate_rate(
    candidate: Mapping[str, Any], references: Sequence[Mapping[str, Any]]
) -> None:
    if candidate.get("candidate") is not True:
        raise EvidenceBuildError("reserve calibration did not produce a candidate")
    raw_rate = candidate.get("rate_per_person_year")
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)) \
            or not math.isfinite(float(raw_rate)) or float(raw_rate) <= 0.0:
        raise EvidenceBuildError("reserve calibration candidate rate is invalid")
    rate = float(raw_rate)
    compiled_rate = float(PacketParams().reserve_rate_per_person_year)
    if rate != compiled_rate:
        raise EvidenceBuildError(
            f"reserve candidate {rate:.12g} differs from compiled PacketParams "
            f"{compiled_rate:.12g}; rebuild policy before running the full battery"
        )
    for entry in references:
        rule = entry["report"].get("reserve_rule_evidence")
        rule_rate = rule.get("rate_per_person_year") \
            if isinstance(rule, Mapping) else None
        if not isinstance(rule, Mapping) or rule.get("valid") is not True \
                or isinstance(rule_rate, bool) \
                or not isinstance(rule_rate, (int, float)) \
                or not math.isfinite(float(rule_rate)) \
                or float(rule_rate) != rate:
            raise EvidenceBuildError(
                "reserve candidate differs from the input packet rate; rebuild P4 packets "
                "at the recorded candidate rate and rerun this evidence pipeline"
            )


def _normalized_elder_audit(
    raw_audit: Mapping[str, Any],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    audit = _json_copy(raw_audit)
    if not isinstance(audit, dict) \
            or audit.get("schema") != freeze_v4_bars.ELDER_AUDIT_SCHEMA:
        raise EvidenceBuildError("phase-three elder reconstruction audit is missing")
    by_pair = {
        (entry["reference_line"], entry["world"]): entry for entry in references
    }
    c_digests = {
        entry["method_digest_sha256"]
        for entry in references
        if entry["reference_line"] == "C"
    }
    if len(c_digests) != 1:
        raise EvidenceBuildError("reference line C does not have one method digest")
    method = audit.get("method_digest")
    if not isinstance(method, dict):
        raise EvidenceBuildError("elder audit method digest is missing")
    method["before_line"] = "A"
    method["after_line"] = "C"
    method["source_sha256"] = next(iter(c_digests))
    rows = audit.get("worlds")
    if not isinstance(rows, list) or len(rows) != 6:
        raise EvidenceBuildError("elder audit does not contain six qualification worlds")
    shock_measurements = []
    for row in rows:
        if not isinstance(row, dict):
            raise EvidenceBuildError("elder audit contains an invalid world row")
        world = str(row.get("world", ""))
        try:
            before_reference = by_pair[("A", world)]
            after_reference = by_pair[("C", world)]
        except KeyError as exc:
            raise EvidenceBuildError(f"{world}: elder audit reference is missing") from exc
        if row.get("available") is False:
            raise EvidenceBuildError(f"{world}: elder audit reference detail is unavailable")
        try:
            before = freeze_v4_bars._elder_reference_evidence(
                before_reference["report"]
            )
            after = freeze_v4_bars._elder_reference_evidence(
                after_reference["report"]
            )
            before_shock = freeze_v4_bars._shock_redraw_evidence(
                before_reference["report"]
            )
            after_shock = freeze_v4_bars._shock_redraw_evidence(
                after_reference["report"]
            )
        except freeze_v4_bars.EvidenceError as exc:
            raise EvidenceBuildError(
                f"{world}: elder audit reference detail is invalid"
            ) from exc
        if _canonical_digest(before_shock) != _canonical_digest(after_shock):
            raise EvidenceBuildError(
                f"{world}: elder audit reference shock measurements disagree"
            )
        shock_measurements.append(before_shock["runtime_evidence"])
        before_states = {
            item["state"]: item for item in before["state_65_plus_person_years"]
        }
        after_states = {
            item["state"]: item for item in after["state_65_plus_person_years"]
        }
        state_rows = []
        denominator = before_numerator = after_numerator = 0.0
        for state in range(6):
            before_state = before_states[state]
            after_state = after_states[state]
            sealed = float(before_state["sealed_person_years"])
            if not math.isclose(
                sealed,
                float(after_state["sealed_person_years"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise EvidenceBuildError(
                    f"{world}: reference lines disagree on sealed elder exposure"
                )
            submitted_before = float(before_state["submitted_person_years"])
            submitted_after = float(after_state["submitted_person_years"])
            state_rows.append({
                "state": state,
                "submitted_before": submitted_before,
                "submitted_after": submitted_after,
                "sealed": sealed,
            })
            denominator += sealed
            before_numerator += abs(submitted_before - sealed)
            after_numerator += abs(submitted_after - sealed)
        if denominator <= 0.0:
            raise EvidenceBuildError(f"{world}: sealed elder exposure is not positive")
        before_regions = {
            item["region"]: item for item in before["liability_mean_by_region"]
        }
        after_regions = {
            item["region"]: item for item in after["liability_mean_by_region"]
        }
        liability_rows = []
        for region in range(6):
            before_region = before_regions[region]
            after_region = after_regions[region]
            sealed = float(before_region["sealed"])
            if not math.isclose(
                sealed,
                float(after_region["sealed"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise EvidenceBuildError(
                    f"{world}: reference lines disagree on sealed regional liability"
                )
            liability_rows.append({
                "region": region,
                "submitted_before": float(before_region["submitted"]),
                "submitted_after": float(after_region["submitted"]),
                "sealed": sealed,
            })
        row["before_report_evidence_id"] = before_reference["evidence_id"]
        row["after_report_evidence_id"] = after_reference["evidence_id"]
        row["exposure_65_plus_absolute_error_percent"] = {
            "definition": freeze_v4_bars.ELDER_EXPOSURE_ERROR_DEFINITION,
            "before": 100.0 * before_numerator / denominator,
            "after": 100.0 * after_numerator / denominator,
        }
        row["state_65_plus_person_years"] = state_rows
        row["liability_mean_by_region"] = liability_rows
    shock = audit.get("shock_redraw")
    if not isinstance(shock, dict):
        raise EvidenceBuildError("elder audit shock redraw description is missing")
    shock["independent_per_member"] = all(
        runtime["redrawn_member_count"] == runtime["member_count"]
        and runtime["distinct_future_schedule_count"] > 1
        and runtime["future_shock_year_count"] > 0
        for runtime in shock_measurements
    )
    audit.pop("digest_sha256", None)
    audit["digest_sha256"] = _canonical_digest(audit)
    return audit


def _mortality_identification_audit(
    qualification: Sequence[Path],
    references: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure the P4 mortality gap and bind it to packets and reference reports."""

    by_world: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in references:
        by_world[str(entry["world"])].append(entry)
    worlds = []
    annual_rates = set()
    for packet in qualification:
        rows = by_world[packet.name]
        if sorted(str(row["reference_line"]) for row in rows) != list(REFERENCE_LINES):
            raise EvidenceBuildError(
                f"{packet.name}: mortality audit lacks reference lines A/B/C"
            )
        packet_inputs = []
        for row in rows:
            evidence = row["report"].get("evidence")
            files = evidence.get("packet_file_sha256") \
                if isinstance(evidence, Mapping) else None
            if not isinstance(files, Mapping):
                raise EvidenceBuildError("reference report lacks packet input digests")
            packet_inputs.append(
                {
                    name: files[name]
                    for name in freeze_v4_bars.RED_TEAM_INPUT_FILES
                }
            )
        if len({_canonical_digest(value) for value in packet_inputs}) != 1:
            raise EvidenceBuildError(
                f"{packet.name}: references disagree on mortality-audit packet inputs"
            )
        try:
            shock_rows = [
                freeze_v4_bars._shock_redraw_evidence(row["report"])
                for row in rows
            ]
        except freeze_v4_bars.EvidenceError as exc:
            raise EvidenceBuildError(
                f"{packet.name}: references lack measured shock redraw evidence"
            ) from exc
        if len({_canonical_digest(value) for value in shock_rows}) != 1:
            raise EvidenceBuildError(
                f"{packet.name}: references disagree on shock redraw evidence"
            )
        contract = json.loads((packet / "participant" / "contract.json").read_text())
        annual_rates.add(float(contract["shock_family"]["annual_rate"]))
        decomposition = phase_three.mortality_gap_decomposition(packet)
        runtime = shock_rows[0]["runtime_evidence"]
        decomposition["continuation_shocks_redrawn_per_member"] = bool(
            runtime["redrawn_member_count"] == runtime["member_count"]
            and runtime["distinct_future_schedule_count"] > 1
            and runtime["future_shock_year_count"] > 0
        )
        worlds.append(
            {
                "world": packet.name,
                "packet_manifest_digest_sha256": _sha256(packet / "manifest.json"),
                "packet_input_sha256": packet_inputs[0],
                "reference_evidence_ids": {
                    str(row["reference_line"]): row["evidence_id"] for row in rows
                },
                "shock_redraw_evidence": shock_rows[0],
                "decomposition": decomposition,
            }
        )
    if len(annual_rates) != 1:
        raise EvidenceBuildError("qualification shock annual rates differ")
    decompositions = [row["decomposition"] for row in worlds]
    lag_effects = [
        100.0 * (float(row["publication_lag_trend_factor"]) - 1.0)
        for row in decompositions
    ]
    audit = {
        "schema": MORTALITY_AUDIT_SCHEMA,
        "supports_gate": "tail_calibration",
        "measurement_source": {
            "file": "meridia/methods/phase_three.py",
            "sha256": _sha256(Path(phase_three.__file__).resolve()),
            "function": "mortality_gap_decomposition",
        },
        "qualification_worlds": [packet.name for packet in qualification],
        "summary": {
            "trend_active_during_public_experience_window": all(
                row["trend_active_during_public_experience_window"] is True
                for row in decompositions
            ),
            "trend_starts_only_after_publication": any(
                row["trend_starts_only_after_public_window"] is True
                for row in decompositions
            ),
            "publication_lag_months": sorted(
                {int(row["publication_lag_months"]) for row in decompositions}
            ),
            "publication_lag_trend_effect_percent_range": [
                min(lag_effects), max(lag_effects)
            ],
            "shock_annual_probability": annual_rates.pop(),
            "continuation_shocks_redrawn_per_member": all(
                row["continuation_shocks_redrawn_per_member"] is True
                for row in decompositions
            ),
        },
        "worlds": worlds,
    }
    audit["digest_sha256"] = _canonical_digest(audit)
    return audit


def _identifiability_audit(
    packets: Sequence[Path], out: Path
) -> tuple[dict[str, Any], Path]:
    path = _under(out, out / "regime_identifiability_audit.json", "identifiability audit")
    temporary = _under(
        out,
        out / ".regime_identifiability_audit.json.tmp",
        "identifiability audit",
    )
    if temporary.exists() or temporary.is_symlink():
        raise EvidenceBuildError("partial regime identifiability audit is present")
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "identifiability_v4.py"),
        "--packets",
        *(str(packet) for packet in packets),
        "--receipt",
        str(temporary),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise EvidenceBuildError(f"regime identifiability audit failed{suffix}")
    if temporary.is_symlink() or not temporary.is_file():
        raise EvidenceBuildError("regime identifiability audit was not written")
    try:
        audit = json.loads(temporary.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBuildError("regime identifiability audit is invalid") from exc
    if not isinstance(audit, dict) \
            or audit.get("schema") != freeze_v4_bars.REGIME_IDENTIFIABILITY_SCHEMA:
        raise EvidenceBuildError("regime identifiability audit schema differs")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() \
                or path.read_bytes() != temporary.read_bytes():
            raise EvidenceBuildError(
                "existing regime identifiability audit differs from this evidence run"
            )
        temporary.unlink()
    else:
        temporary.replace(path)
    return audit, path


def _validate_reference_preflight_audits(
    references: Sequence[Mapping[str, Any]],
    mortality_audit: Mapping[str, Any],
    identifiability_audit: Mapping[str, Any],
) -> None:
    try:
        normalized_references = freeze_v4_bars._normalize_entries(
            references, kind="reference"
        )
        regime = freeze_v4_bars._validate_regime_identifiability_audit(
            identifiability_audit
        )
        freeze_v4_bars._validate_mortality_identification_audit(
            mortality_audit,
            normalized_references,
            regime,
            freeze_v4_bars.QUALIFICATION_WORLDS,
        )
    except freeze_v4_bars.EvidenceError as exc:
        raise EvidenceBuildError(f"reference preflight audit failed: {exc}") from exc


def _validate_manifest_design(manifest: Mapping[str, Any]) -> None:
    for name, count in EXPECTED_COUNTS.items():
        rows = manifest.get(name)
        if not isinstance(rows, list) or len(rows) != count:
            raise EvidenceBuildError(f"{name} must contain exactly {count} reports")
    references = manifest["reference_reports"]
    expected_references = {
        (line, f"qual-{index}") for line in REFERENCE_LINES for index in range(6)
    }
    if {(row["reference_line"], row["world"]) for row in references} \
            != expected_references:
        raise EvidenceBuildError("final reference design is incomplete or duplicated")
    replicates = manifest["replicate_reports"]
    groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for row in replicates:
        groups[(row["world"], row["replicate_id"])].append(row["reference_line"])
    if len(groups) != 6 * REPLICATES_PER_WORLD or any(
        sorted(lines) != list(REFERENCE_LINES) for lines in groups.values()
    ):
        raise EvidenceBuildError("outer reference reports are not paired A/B/C")
    expected_controls = {
        (name, f"qual-{index}")
        for name in controls.QUALIFICATION_CONTROLS
        for index in range(6)
    }
    if {(row["control"], row["world"]) for row in manifest["control_reports"]} \
            != expected_controls:
        raise EvidenceBuildError("qualification control design is incomplete or duplicated")
    expected_diagnostics = {
        (name, f"dev-{index:02d}")
        for name in controls.DECOMPOSITION_CONTROLS
        for index in range(12)
    }
    if {
        (row["diagnostic"], row["world"])
        for row in manifest["development_diagnostic_reports"]
    } != expected_diagnostics:
        raise EvidenceBuildError("development diagnostic design is incomplete or duplicated")
    evidence_ids = [
        row["evidence_id"]
        for name in EXPECTED_COUNTS
        for row in manifest[name]
    ]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise EvidenceBuildError("evidence identifiers are reused across reports")


def build_evidence(
    development_root: Path,
    qualification_root: Path,
    out_dir: Path,
    params: phase_three.MeasurementParams = phase_three.MeasurementParams(),
    *,
    references_only: bool = False,
) -> dict[str, Any]:
    """Run or resume the complete P4 qualification evidence design."""

    registered_params = phase_three.MeasurementParams()
    if params != registered_params:
        raise EvidenceBuildError(
            "final freeze evidence requires the registered measurement parameters "
            "(bootstrap=100, sweeps=400, simulation_paths=2048, "
            "linkage_bootstraps=12)"
        )

    development, qualification = _packet_roots(development_root, qualification_root)
    requested_out = _safe_path(out_dir, "evidence output", must_exist=False)
    for packet in (*development, *qualification):
        if requested_out == packet or requested_out in packet.parents \
                or packet in requested_out.parents:
            raise EvidenceBuildError("evidence output must not overlap a source packet")
    out = _output_root(requested_out)
    contract, contract_digest = _bind_contract(out, development, qualification, params)
    phase_out = out / "phase_three"
    try:
        _run_reference_preflight(
            development, qualification, phase_out, params
        )
    except ValueError as exc:
        raise EvidenceBuildError(str(exc)) from exc
    _verify_contract(out, contract, development, qualification, params)
    rate_preflight_input = _preflight_calibration_inputs(qualification, phase_out)
    rate_preflight_input_path = _write_json_once(
        out,
        out / "reserve_rate_preflight_inputs.json",
        rate_preflight_input,
        "reserve rate preflight input",
    )
    rate_preflight_candidate = calibrate_reserve_rate.calibrate(
        rate_preflight_input["entries"]
    )
    rate_preflight_candidate_path = _write_json_once(
        out,
        out / "reserve_rate_preflight_candidate.json",
        rate_preflight_candidate,
        "reserve rate preflight candidate",
    )
    _require_packets_at_candidate_rate(rate_preflight_candidate, ())

    reference_reports, reference_specs = _collect_reference_entries(
        qualification, phase_out, out, contract, contract_digest
    )
    calibration_input = _calibration_inputs(
        reference_reports, qualification, phase_out
    )
    calibration_input_path = _write_json_once(
        out,
        out / "reserve_calibration_inputs.json",
        calibration_input,
        "reserve calibration input",
    )
    calibration_candidate = calibrate_reserve_rate.calibrate(
        calibration_input["entries"]
    )
    candidate_path = _write_json_once(
        out,
        out / "reserve_rate_candidate.json",
        calibration_candidate,
        "reserve calibration candidate",
    )
    _require_packets_at_candidate_rate(
        calibration_candidate, reference_reports
    )
    if float(calibration_candidate["rate_per_person_year"]) \
            != float(rate_preflight_candidate["rate_per_person_year"]):
        raise EvidenceBuildError(
            "authenticated reference reports changed the reserve-rate preflight"
        )
    mortality_audit = _mortality_identification_audit(
        qualification, reference_reports
    )
    mortality_path = _write_json_once(
        out,
        out / "mortality_identification_audit.json",
        mortality_audit,
        "mortality identification audit",
    )
    identifiability, identifiability_path = _identifiability_audit(
        [*development, *qualification], out
    )
    _validate_reference_preflight_audits(
        reference_reports, mortality_audit, identifiability
    )
    _verify_contract(out, contract, development, qualification, params)
    preflight = {
        "schema": "meridia.v4.reserve-rate-reference-preflight.v1",
        "compiled_packet_rate_per_person_year": float(
            PacketParams().reserve_rate_per_person_year
        ),
        "reference_report_count": len(reference_reports),
        "reference_evidence_ids": [
            row["evidence_id"] for row in reference_reports
        ],
        "reserve_rate_preflight_candidate": rate_preflight_candidate,
        "reserve_calibration_candidate": calibration_candidate,
        "mortality_identification_audit": mortality_audit,
        "regime_identifiability_audit": identifiability,
        "full_battery_authorized": True,
    }
    _write_json_once(
        out, out / "reference_preflight.json", preflight, "reference preflight"
    )
    if references_only:
        return preflight

    try:
        phase_result = phase_three.measure(
            list(development),
            list(qualification),
            phase_out,
            None,
            params,
            raw_pre_freeze=True,
        )
    except ValueError as exc:
        raise EvidenceBuildError(str(exc)) from exc
    _verify_contract(out, contract, development, qualification, params)
    base, specs = _collect_base_entries(
        development, qualification, phase_out, out, contract, contract_digest
    )
    if base["reference_reports"] != reference_reports:
        raise EvidenceBuildError("final references changed after the preflight")
    for key, spec in reference_specs.items():
        if specs.get(key) != spec:
            raise EvidenceBuildError(f"{key} changed after the preflight")

    replicate_reports = _collect_replicate_entries(
        qualification,
        phase_out,
        out,
        contract,
        contract_digest,
        params,
        specs,
    )
    _verify_contract(out, contract, development, qualification, params)

    raw_elder_path = Path(str(phase_result["elder_reconstruction_audit"]["json_path"]))
    if raw_elder_path.is_symlink() or not raw_elder_path.is_file():
        raise EvidenceBuildError("phase-three elder audit artifact is missing")
    raw_elder = json.loads(raw_elder_path.read_text())
    elder_audit = _normalized_elder_audit(raw_elder, base["reference_reports"])
    elder_path = _write_json_once(
        out, out / "elder_reconstruction_audit.json", elder_audit, "elder audit"
    )

    red_team = red_team_reserve_total.run_measurement(
        development[0].parent, qualification[0].parent
    )
    red_team_path = _write_json_once(
        out, out / "reserve_red_team_audit.json", red_team, "reserve red-team audit"
    )
    _verify_contract(out, contract, development, qualification, params)

    manifest = {
        "schema": freeze_v4_bars.EVIDENCE_SCHEMA,
        "reference_reports": base["reference_reports"],
        "replicate_reports": replicate_reports,
        "control_reports": base["control_reports"],
        "development_diagnostic_reports": base[
            "development_diagnostic_reports"
        ],
        "elder_reconstruction_audit": elder_audit,
        "mortality_identification_audit": mortality_audit,
        "regime_identifiability_audit": identifiability,
        "reserve_calibration_audit": calibration_candidate,
        "reserve_red_team_audit": red_team,
        "evidence_run": {
            "schema": PIPELINE_SCHEMA,
            "measurement_contract_path": str(
                (out / "evidence_measurement_contract.json").resolve()
            ),
            "measurement_contract_digest_sha256": contract_digest,
            "runner_digest_sha256": contract["runner_digest_sha256"],
            "fixed_method_seeds": dict(resampling.REFERENCE_METHOD_SEEDS),
            "artifacts": {
                "reserve_rate_preflight_inputs": {
                    "path": str(rate_preflight_input_path.resolve()),
                    "sha256": _sha256(rate_preflight_input_path),
                },
                "reserve_rate_preflight_candidate": {
                    "path": str(rate_preflight_candidate_path.resolve()),
                    "sha256": _sha256(rate_preflight_candidate_path),
                },
                "reserve_calibration_inputs": {
                    "path": str(calibration_input_path.resolve()),
                    "sha256": _sha256(calibration_input_path),
                },
                "reserve_calibration_candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": _sha256(candidate_path),
                },
                "elder_reconstruction_audit": {
                    "path": str(elder_path.resolve()),
                    "sha256": _sha256(elder_path),
                },
                "mortality_identification_audit": {
                    "path": str(mortality_path.resolve()),
                    "sha256": _sha256(mortality_path),
                },
                "reserve_red_team_audit": {
                    "path": str(red_team_path.resolve()),
                    "sha256": _sha256(red_team_path),
                },
                "regime_identifiability_audit": {
                    "path": str(identifiability_path.resolve()),
                    "sha256": _sha256(identifiability_path),
                },
            },
            "audit_inputs": {
                "development_root": str(development[0].parent),
                "qualification_root": str(qualification[0].parent),
                "qualification_reference_submissions": [
                    {
                        "reference_line": row["reference_line"],
                        "world": row["world"],
                        "evidence_id": row["evidence_id"],
                        "packet_dir": next(
                            item["packet_dir"]
                            for item in calibration_input["entries"]
                            if item["reference_line"] == row["reference_line"]
                            and item["world"] == row["world"]
                        ),
                        "submission_dir": next(
                            item["submission_dir"]
                            for item in calibration_input["entries"]
                            if item["reference_line"] == row["reference_line"]
                            and item["world"] == row["world"]
                        ),
                    }
                    for row in base["reference_reports"]
                ],
            },
        },
    }
    _validate_manifest_design(manifest)
    _write_json_once(
        out,
        out / "freeze_evidence_manifest.json",
        manifest,
        "freeze evidence manifest",
    )
    _verify_contract(out, contract, development, qualification, params)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=400)
    parser.add_argument("--simulation-paths", type=int, default=2048)
    parser.add_argument("--linkage-bootstraps", type=int, default=12)
    parser.add_argument(
        "--references-only",
        action="store_true",
        help="stop after the 18-reference reserve-rate preflight",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_evidence(
            args.development_root,
            args.qualification_root,
            args.out,
            phase_three.MeasurementParams(
                bootstrap_replicates=args.bootstrap,
                bayesian_sweeps=args.sweeps,
                simulation_paths=args.simulation_paths,
                linkage_bootstraps=args.linkage_bootstraps,
            ),
            references_only=args.references_only,
        )
    except (EvidenceBuildError, red_team_reserve_total.MeasurementError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result = (
        {
            "preflight": str((Path(args.out) / "reference_preflight.json").resolve()),
            "reference_report_count": manifest["reference_report_count"],
            "full_battery_authorized": manifest["full_battery_authorized"],
        }
        if args.references_only
        else {
            "manifest": str((Path(args.out) / "freeze_evidence_manifest.json").resolve()),
            "report_counts": {name: len(manifest[name]) for name in EXPECTED_COUNTS},
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
