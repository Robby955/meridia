"""Phase-three measurements for controls, decomposition, and the third line.

This module never calibrates a submitted tail from the public reserve total. It keeps
the current four-file writer for compatibility, but disclosure and detailed-table
reasons are excluded from the retained five-composite report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..verify import verify_submission
from . import actuarial_reference as AR
from . import bayesian as B
from . import controls
from . import design_based as A
from . import third_reference as C
from .common import load_packet


COMPOSITE_FAMILIES = (
    "exposures_and_rates",
    "release_accuracy",
    "interval_quality",
    "tail_calibration",
    "reserve_skill",
)
RATE_NAMES = ("person_years_exposure", "mortality_rate", "qualifying_event_rate")
CURRENTLY_IGNORED_PREFIXES = (
    "disclosure:",
    "disclosure utility:",
    "detailed accuracy:",
)
HARD_CHECK_PREFIXES = (
    "file set:",
    "schema:",
    "additivity:",
    "projection schema:",
    "projection additivity:",
    "rate schema:",
    "reserve schema:",
    "reserve: infeasible",
)
SUBMISSION_FILES = ("release.csv", "projection.csv", "detailed.csv", "reserve.csv")
OPTIONAL_SUBMISSION_FILES = ("totals.csv",)
CALIBRATION_RECEIPT_SCHEMA = "meridia-phase-three-calibration-receipts-v1"
RUN_RECEIPT_SCHEMA = "meridia-phase-three-method-run-receipt-v1"


@dataclass(frozen=True)
class MeasurementParams:
    bootstrap_replicates: int = 100
    bayesian_sweeps: int = 400
    simulation_paths: int = 2048
    linkage_bootstraps: int = 12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_packet_manifest(packet: Path) -> tuple[dict, Path]:
    """Read a regular in-packet manifest only after rejecting linked paths."""
    packet = Path(packet).resolve()
    manifest_path = packet / "manifest.json"
    if manifest_path.is_symlink() or manifest_path.resolve().parent != packet:
        raise ValueError(f"packet manifest may not be linked: {packet}")
    if not manifest_path.is_file():
        raise ValueError(f"packet has no manifest: {packet}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"packet has an invalid manifest: {packet}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"packet manifest is not a JSON object: {packet}")
    return manifest, manifest_path


def _prepare_output_dir(out_dir: Path) -> Path:
    """Create one real output directory and reject a linked output root."""
    out_dir = Path(out_dir)
    if out_dir.is_symlink():
        raise ValueError("measurement output directory may not be a symlink")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir() or out_dir.is_symlink():
        raise ValueError("measurement output must be a real directory")
    return out_dir.resolve()


def _assert_output_location(root: Path, path: Path, label: str) -> Path:
    """Require an output path to remain under root with no linked component."""
    root = Path(root).resolve()
    path = Path(path).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the measurement output") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} may not use symlinked paths")
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the measurement output")
    return path


def _verified_packet_files(packet: Path) -> dict[str, str]:
    """Verify every manifest claim and return hashes of the packet bytes."""
    packet = Path(packet).resolve()
    manifest, _ = _read_packet_manifest(packet)
    inventory = {}
    for side in ("participant", "retained"):
        entries = manifest.get(side)
        if not isinstance(entries, dict):
            raise ValueError(f"packet manifest has no {side} file inventory: {packet}")
        side_path = packet / side
        if side_path.is_symlink():
            raise ValueError(f"packet {side} directory may not be a symlink")
        root = side_path.resolve()
        if root.parent != packet:
            raise ValueError(f"packet {side} directory escapes the packet root")
        candidates = list(root.rglob("*"))
        linked = [str(path.relative_to(root)) for path in candidates if path.is_symlink()]
        if linked:
            raise ValueError(
                f"packet {side} inventory contains symlinks: {sorted(linked)}"
            )
        actual = {
            str(path.relative_to(root))
            for path in candidates
            if path.is_file()
        }
        if actual != set(entries):
            missing = sorted(set(entries) - actual)
            unexpected = sorted(actual - set(entries))
            raise ValueError(
                f"packet {side} inventory mismatch; missing {missing}, unexpected {unexpected}"
            )
        for name, claim in entries.items():
            path = root / name
            if path.is_symlink() or root not in path.resolve().parents:
                raise ValueError(f"packet manifest path escapes {side}: {name}")
            if not isinstance(claim, dict):
                raise ValueError(f"packet manifest entry is invalid: {side}/{name}")
            size = path.stat().st_size
            digest = _sha256(path)
            if claim.get("bytes") != size or claim.get("sha256") != digest:
                raise ValueError(f"packet manifest hash mismatch: {side}/{name}")
            inventory[f"{side}/{name}"] = digest
    return inventory


def _measurement_contract(
    development_packets: list[Path],
    qualification_packets: list[Path],
    bars_path: Path,
    params: MeasurementParams,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    sources = sorted((repo_root / "meridia").rglob("*.py"))
    packets = [Path(path).resolve() for path in development_packets + qualification_packets]
    packet_files = {}
    manifest_sha256 = {}
    contract_sha256 = {}
    for packet in packets:
        inventory = _verified_packet_files(packet)
        if "participant/contract.json" not in inventory:
            raise ValueError(f"packet manifest does not bind participant/contract.json: {packet}")
        _, manifest_path = _read_packet_manifest(packet)
        key = str(packet)
        packet_files[key] = inventory
        manifest_sha256[key] = _sha256(manifest_path)
        contract_sha256[key] = inventory["participant/contract.json"]
    return {
        "schema": "meridia-phase-three-measurement-v1",
        "development_packets": [str(path.resolve()) for path in development_packets],
        "qualification_packets": [
            str(path.resolve()) for path in qualification_packets
        ],
        "packet_contract_sha256": contract_sha256,
        "packet_manifest_sha256": manifest_sha256,
        "packet_file_sha256": packet_files,
        "bars_sha256": _sha256(bars_path),
        "params": asdict(params),
        "composites": list(COMPOSITE_FAMILIES),
        "source_sha256": {
            str(path.relative_to(repo_root)): _sha256(path) for path in sources
        },
        "reserve_total_used_for_tail_calibration": False,
    }


def _bind_measurement_output(out_dir: Path, contract: dict) -> None:
    out_dir = _prepare_output_dir(out_dir)
    path = _assert_output_location(
        out_dir, out_dir / "measurement_contract.json", "measurement contract"
    )
    temporary = _assert_output_location(
        out_dir, out_dir / ".measurement_contract.json.tmp", "measurement contract"
    )
    encoded = json.dumps(contract, indent=2, sort_keys=True) + "\n"
    if not path.exists() and any(out_dir.iterdir()):
        raise ValueError(
            "nonempty measurement output has no measurement_contract.json"
        )
    if path.exists():
        if not path.is_file():
            raise ValueError("measurement_contract.json is not a regular file")
        if path.read_text() != encoded:
            raise ValueError(
                "measurement output already belongs to a different code or run"
            )
        return
    if temporary.exists():
        raise ValueError("partial measurement contract is present")
    temporary.write_text(encoded)
    temporary.replace(path)


def _measurement_contract_sha256(out_dir: Path) -> str:
    out_dir = _prepare_output_dir(out_dir)
    contract_path = _assert_output_location(
        out_dir, out_dir / "measurement_contract.json", "measurement contract"
    )
    if not contract_path.is_file():
        raise ValueError("measurement output has no bound measurement contract")
    return _sha256(contract_path)


def _load_calibration_receipts(out_dir: Path, contract_sha256: str) -> dict:
    out_dir = _prepare_output_dir(out_dir)
    path = _assert_output_location(
        out_dir, out_dir / "calibration_receipts.json", "calibration receipt"
    )
    temporary = _assert_output_location(
        out_dir, out_dir / ".calibration_receipts.json.tmp", "calibration receipt"
    )
    if temporary.exists():
        raise ValueError("partial calibration receipt is present")
    if not path.exists():
        return {
            "schema": CALIBRATION_RECEIPT_SCHEMA,
            "measurement_contract_sha256": contract_sha256,
            "artifacts": {},
        }
    if not path.is_file():
        raise ValueError("calibration_receipts.json is not a regular file")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("calibration receipt is incomplete or invalid") from exc
    if not isinstance(receipt, dict):
        raise ValueError("calibration receipt is incomplete or invalid")
    if receipt.get("schema") != CALIBRATION_RECEIPT_SCHEMA:
        raise ValueError("calibration receipt has the wrong schema")
    if receipt.get("measurement_contract_sha256") != contract_sha256:
        raise ValueError("calibration receipt belongs to a different measurement run")
    if not isinstance(receipt.get("artifacts"), dict):
        raise ValueError("calibration receipt has no artifact map")
    return receipt


def _write_calibration_receipts(out_dir: Path, receipt: dict) -> None:
    out_dir = _prepare_output_dir(out_dir)
    path = _assert_output_location(
        out_dir, out_dir / "calibration_receipts.json", "calibration receipt"
    )
    temporary = _assert_output_location(
        out_dir, out_dir / ".calibration_receipts.json.tmp", "calibration receipt"
    )
    if temporary.exists():
        raise ValueError("partial calibration receipt is present")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_calibration_json(path: Path, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"calibration {label} artifact is missing")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"calibration {label} artifact is incomplete or invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"calibration {label} artifact is not a JSON object")


def _ensure_calibration_artifact(
    out_dir: Path,
    label: str,
    generator: Callable[[Path], object],
) -> Path:
    """Create once, then hash-bind one deterministic calibration across restarts."""
    out_dir = _prepare_output_dir(out_dir)
    artifact = _assert_output_location(
        out_dir, out_dir / f"calibration_{label}.json", f"calibration {label} artifact"
    )
    contract_sha256 = _measurement_contract_sha256(out_dir)
    receipt = _load_calibration_receipts(out_dir, contract_sha256)
    recorded = receipt["artifacts"].get(label)
    if artifact.exists():
        if not artifact.is_file():
            raise ValueError(f"calibration {label} artifact is not a regular file")
        if not isinstance(recorded, dict):
            raise ValueError(
                f"calibration {label} artifact exists without a bound receipt"
            )
        expected = {
            "file": artifact.name,
            "sha256": _sha256(artifact),
        }
        if recorded != expected:
            raise ValueError(f"calibration {label} artifact changed after it was bound")
        _validate_calibration_json(artifact, label)
        return artifact
    if recorded is not None:
        raise ValueError(f"calibration {label} receipt exists but its artifact is missing")

    temporary = _assert_output_location(
        out_dir,
        artifact.with_name(f".{artifact.name}.tmp"),
        f"calibration {label} artifact",
    )
    if temporary.exists():
        raise ValueError(f"partial calibration {label} artifact is present")
    generator(temporary)
    _validate_calibration_json(temporary, label)
    temporary.replace(artifact)
    receipt["artifacts"][label] = {
        "file": artifact.name,
        "sha256": _sha256(artifact),
    }
    _write_calibration_receipts(out_dir, receipt)
    return artifact


def reason_composite(reason: str) -> str | None:
    """Map one current verifier reason to one retained composite family."""
    text = str(reason).strip().lower()
    if text.startswith(CURRENTLY_IGNORED_PREFIXES + HARD_CHECK_PREFIXES):
        return None
    if text.startswith(("exposure:", "rate:")):
        return "exposures_and_rates"
    if text.startswith("coverage:") and any(name in text for name in RATE_NAMES):
        return "exposures_and_rates"
    if text.startswith(("tail:",)):
        return "tail_calibration"
    if text.startswith("reserve:"):
        return "reserve_skill"
    if text.startswith(
        (
            "coverage:",
            "interval score:",
            "projection coverage:",
            "projection interval score:",
        )
    ):
        return "interval_quality"
    if text.startswith(
        (
            "accuracy:",
            "projection accuracy:",
        )
    ):
        return "release_accuracy"
    if text.startswith("bars:"):
        raise ValueError(
            f"cannot classify a report that did not use frozen bars: {reason}"
        )
    raise ValueError(f"unmapped verifier reason: {reason}")


def failed_composites(report: dict) -> tuple[list[str], list[str]]:
    """Return retained failures and current detailed-stage reasons separately."""
    failed, ignored = set(), []
    for reason in report.get("reasons", []):
        text = str(reason).strip().lower()
        if text.startswith(HARD_CHECK_PREFIXES):
            continue
        composite = reason_composite(reason)
        if composite is None:
            ignored.append(str(reason))
        else:
            failed.add(composite)
    return [name for name in COMPOSITE_FAMILIES if name in failed], ignored


def hard_check_failures(report: dict) -> list[str]:
    """Return structural reasons that are not scientific composite evidence."""
    return [
        str(reason)
        for reason in report.get("reasons", [])
        if str(reason).strip().lower().startswith(HARD_CHECK_PREFIXES)
    ]


def _finite(value, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def summarize_report(report: dict) -> dict:
    """Small JSON-safe gate and reserve summary used by every measurement row."""
    failed, ignored = failed_composites(report)
    hard = hard_check_failures(report)
    reserve = report.get("reserve") or {}
    return {
        "composite_pass": not failed,
        "failed_composites": failed,
        "hard_check_pass": not hard,
        "hard_check_failures": hard,
        "current_verifier_pass": bool(report.get("pass", False)),
        "ignored_current_stage_reasons": ignored,
        "raw_reasons": [str(reason) for reason in report.get("reasons", [])],
        "gate_metrics": {
            "release": report.get("metrics") or {},
            "projection": report.get("projection_metrics") or {},
            "rates": report.get("rate_metrics") or {},
            "reserve": reserve,
        },
        "reserve": {
            "feasible": bool(reserve.get("feasible", False)),
            "J": _finite(reserve.get("J")),
            "J_baseline": _finite(reserve.get("J_baseline")),
            "J_oracle": _finite(reserve.get("J_oracle")),
            "skill": _finite(reserve.get("skill")),
            "mean_quantile_score": _finite(reserve.get("mean_quantile_score")),
            "mean_shortfall_error": _finite(reserve.get("mean_shortfall_error")),
        },
    }


def regional_liability_means(packet_dir: Path, submission_dir: Path) -> list[dict]:
    """Compare each filed liability mean with the sealed ensemble mean by region."""
    import pandas as pd

    submitted = pd.read_csv(Path(submission_dir) / "reserve.csv").sort_values("region")
    with np.load(
        Path(packet_dir) / "retained" / "continuation_liabilities.npz"
    ) as archive:
        liability = np.asarray(archive["liability"], dtype=np.float64)
    sealed_summary = AR.tail_summary(liability)
    sealed = sealed_summary["mean"]
    tail_width = np.maximum(sealed_summary["q"] - sealed, 1e-12)
    filed = submitted["liability_mean"].to_numpy(dtype=np.float64)
    regions = submitted["region"].to_numpy(dtype=np.int64)
    if len(filed) != len(sealed) or not np.array_equal(regions, np.arange(len(sealed))):
        raise ValueError("reserve rows do not match the retained regions")
    out = []
    for region, (estimate, truth) in enumerate(zip(filed, sealed)):
        out.append(
            {
                "region": region,
                "submitted_mean": float(estimate),
                "sealed_mean": float(truth),
                "ratio": float(estimate / max(truth, 1e-12)),
                "relative_error": float((estimate - truth) / max(abs(truth), 1e-12)),
                "level_error_in_tail_widths": float(
                    (estimate - truth) / tail_width[region]
                ),
                "within_one_tail_width": bool(
                    abs(estimate - truth) <= tail_width[region]
                ),
            }
        )
    return out


def sealed_exceedance_audit(packet_dir: Path, submission_dir: Path) -> dict:
    """Evaluate a submitted q95 against the retained continuation ensemble."""
    import pandas as pd

    submitted = pd.read_csv(Path(submission_dir) / "reserve.csv").sort_values("region")
    regions = submitted["region"].to_numpy(dtype=np.int64)
    q95 = submitted["q95"].to_numpy(dtype=np.float64)
    with np.load(
        Path(packet_dir) / "retained" / "continuation_liabilities.npz"
    ) as archive:
        liability = np.asarray(archive["liability"], dtype=np.float64)
    if liability.ndim != 2 or len(q95) != liability.shape[1]:
        raise ValueError("submitted q95 does not match retained regions")
    if not np.array_equal(regions, np.arange(liability.shape[1])):
        raise ValueError("reserve rows do not contain each retained region once")
    exceedance = (liability > q95[None, :]).mean(axis=0)
    deviation = np.abs(exceedance - 0.05)
    return {
        "pooled_exceedance_deviation": float(deviation.mean()),
        "regions": [
            {
                "region": int(region),
                "submitted_q95": float(q95[region]),
                "sealed_exceedance_probability": float(exceedance[region]),
                "absolute_deviation_from_0_05": float(deviation[region]),
            }
            for region in range(liability.shape[1])
        ],
    }


def _require_elder_cells(
    frame, n_states: int, elder_bands: set[str], label: str
) -> None:
    expected = {
        (state, str(sex), str(band))
        for state in range(n_states)
        for sex in AR.SEX_LABELS
        for band in elder_bands
    }
    observed = {
        (int(row.unit), str(row.sex), str(row.age_band)) for row in frame.itertuples()
    }
    if observed != expected or len(frame) != len(expected):
        missing = len(expected - observed)
        extra = len(observed - expected) + max(len(frame) - len(observed), 0)
        raise ValueError(
            f"{label} needs every state 65-plus band and sex cell; "
            f"missing {missing}, extra or duplicate {extra}"
        )


def elder_state_exposure_survival(
    packet_dir: Path, submission_dir: Path, bars: dict | None = None
) -> dict:
    """Quantify state 65-plus exposure and implied survival against retained truth."""
    import pandas as pd

    release = pd.read_csv(Path(submission_dir) / "release.csv")
    truth = pd.read_csv(Path(packet_dir) / "retained" / "rate_truth_horizon.csv")
    elder_bands = {"65-74", "75-84", "85+"}
    keys = ["level", "unit", "sex", "age_band"]
    estimate = release[
        (release["level"] == "state")
        & (release["age_band"].isin(elder_bands))
        & (release["estimand"].isin(("person_years_exposure", "mortality_rate")))
    ]
    retained = truth[
        (truth["level"] == "state")
        & (truth["age_band"].isin(elder_bands))
        & (truth["estimand"].isin(("person_years_exposure", "mortality_rate")))
    ]
    exposure_est = estimate[estimate["estimand"] == "person_years_exposure"]
    exposure_true = retained[retained["estimand"] == "person_years_exposure"]
    mortality_est = estimate[estimate["estimand"] == "mortality_rate"]
    mortality_true = retained[retained["estimand"] == "mortality_rate"]
    contract = json.loads(
        (Path(packet_dir) / "participant" / "contract.json").read_text()
    )
    n_states = int(contract["n_states"])
    for frame, label in (
        (exposure_est, "submitted exposure"),
        (mortality_est, "submitted mortality"),
        (exposure_true, "retained exposure"),
        (mortality_true, "retained mortality"),
    ):
        _require_elder_cells(frame, n_states, elder_bands, label)
    est = exposure_est[keys + ["estimate"]].rename(columns={"estimate": "exposure_est"})
    est = est.merge(
        mortality_est[keys + ["estimate"]].rename(
            columns={"estimate": "mortality_est"}
        ),
        on=keys,
        validate="one_to_one",
    )
    actual = exposure_true[keys + ["value"]].rename(columns={"value": "exposure_true"})
    actual = actual.merge(
        mortality_true[keys + ["value"]].rename(columns={"value": "mortality_true"}),
        on=keys,
        validate="one_to_one",
    )
    cells = est.merge(actual, on=keys, validate="one_to_one")
    years = (
        float(contract["ticks"]["horizon"]) - float(contract["ticks"]["revised"])
    ) / 12.0
    actuarial_bars = (bars or {}).get("actuarial", {})
    exposure_ceiling = _finite(actuarial_bars.get("exposure_error_ceiling"))
    mortality_ceiling = _finite(actuarial_bars.get("mortality_error_ceiling"))
    rows = []
    for state, block in cells.groupby("unit", sort=True):
        exposure_estimate = float(block["exposure_est"].sum())
        exposure_truth = float(block["exposure_true"].sum())
        mortality_estimate = float(
            np.average(
                block["mortality_est"], weights=np.maximum(block["exposure_est"], 1e-12)
            )
        )
        mortality_truth = float(
            np.average(
                block["mortality_true"],
                weights=np.maximum(block["exposure_true"], 1e-12),
            )
        )
        survival_estimate = float(np.exp(-years * mortality_estimate))
        survival_truth = float(np.exp(-years * mortality_truth))
        rows.append(
            {
                "state": int(state),
                "estimated_person_years": exposure_estimate,
                "sealed_person_years": exposure_truth,
                "exposure_relative_error": float(
                    (exposure_estimate - exposure_truth) / max(exposure_truth, 1e-12)
                ),
                "estimated_mortality_rate": mortality_estimate,
                "sealed_mortality_rate": mortality_truth,
                "mortality_rate_ratio": float(
                    mortality_estimate / max(mortality_truth, 1e-12)
                ),
                "estimated_survival": survival_estimate,
                "sealed_survival": survival_truth,
                "survival_difference": survival_estimate - survival_truth,
                "exposure_within_aggregate_proxy_ceiling": None
                if exposure_ceiling is None
                else bool(
                    abs(exposure_estimate / max(exposure_truth, 1e-12) - 1.0)
                    <= exposure_ceiling
                ),
                "mortality_within_aggregate_proxy_ceiling": None
                if mortality_ceiling is None
                else bool(
                    abs(mortality_estimate / max(mortality_truth, 1e-12) - 1.0)
                    <= mortality_ceiling
                ),
            }
        )
    return {
        "thresholds": {
            "aggregate_exposure_relative_error_ceiling_copied_from_bar": exposure_ceiling,
            "aggregate_mortality_relative_error_ceiling_copied_from_bar": mortality_ceiling,
            "criterion": "absolute aggregate relative error, not the verifier cell percentile",
        },
        "states": rows,
        "exposure_within_aggregate_proxy_ceiling": None
        if exposure_ceiling is None
        else all(row["exposure_within_aggregate_proxy_ceiling"] for row in rows),
        "mortality_within_aggregate_proxy_ceiling": None
        if mortality_ceiling is None
        else all(row["mortality_within_aggregate_proxy_ceiling"] for row in rows),
    }


def participant_elder_identifiability(
    packet_dir: Path, bars: dict | None = None
) -> dict:
    """Measure what the public elder inputs identify before fitting a reference.

    The last experience exposure is annualized over the horizon and compared with the
    retained horizon exposure. Its state share is also compared separately, since a
    common stale level cancels from that composition. Mortality is the five-year pooled
    public experience rate. The register comparison is one record per current identifier.
    None of these simple readings uses a submitted reference estimate.
    """
    import pandas as pd

    packet_dir = Path(packet_dir)
    data = load_packet(packet_dir)
    contract = data["contract"]
    experience = AR.load_experience(packet_dir, contract)
    if experience is None:
        raise AR.MissingActuarialInputs(
            "participant elder audit needs the published experience file"
        )
    n_states = int(np.asarray(data["county_state"]).max()) + 1
    arrays = AR.experience_arrays(experience, n_states)
    elder = np.asarray([low >= 65 for low, _ in AR.ACTUARIAL_AGE_BANDS], dtype=bool)
    last_exposure = np.asarray(arrays["exposure"][-1])[:, elder, :].sum(axis=(1, 2))
    historical_exposure = np.asarray(arrays["exposure"][:, :, elder, :]).sum(
        axis=(0, 2, 3)
    )
    historical_deaths = np.asarray(arrays["deaths"][:, :, elder, :]).sum(axis=(0, 2, 3))
    historical_mortality = historical_deaths / np.maximum(historical_exposure, 1e-12)

    truth = pd.read_csv(packet_dir / "retained" / "rate_truth_horizon.csv")
    elder_labels = {
        label
        for label, (low, _) in zip(AR.ACTUARIAL_AGE_BAND_LABELS, AR.ACTUARIAL_AGE_BANDS)
        if low >= 65
    }
    exposure_truth = truth[
        (truth["level"] == "state")
        & (truth["estimand"] == "person_years_exposure")
        & (truth["age_band"].isin(elder_labels))
    ][["unit", "sex", "age_band", "value"]].rename(columns={"value": "exposure"})
    mortality_truth = truth[
        (truth["level"] == "state")
        & (truth["estimand"] == "mortality_rate")
        & (truth["age_band"].isin(elder_labels))
    ][["unit", "sex", "age_band", "value"]].rename(columns={"value": "mortality"})
    _require_elder_cells(exposure_truth, n_states, elder_labels, "retained exposure")
    _require_elder_cells(mortality_truth, n_states, elder_labels, "retained mortality")
    retained = exposure_truth.merge(
        mortality_truth,
        on=["unit", "sex", "age_band"],
        validate="one_to_one",
    )
    retained["deaths"] = retained["exposure"] * retained["mortality"]
    retained = retained.groupby("unit", sort=True).agg(
        exposure=("exposure", "sum"), deaths=("deaths", "sum")
    )
    retained["mortality"] = retained["deaths"] / np.maximum(retained["exposure"], 1e-12)
    if not np.array_equal(retained.index.to_numpy(dtype=np.int64), np.arange(n_states)):
        raise ValueError("retained elder audit does not contain every state")

    tick = int(contract["ticks"]["revised"])
    years = (float(contract["ticks"]["horizon"]) - tick) / 12.0
    register = data["population"].drop_duplicates("person_id").copy()
    register = register[
        (register["county"] >= 0) & (register["county"] < len(data["county_state"]))
    ]
    register["age"] = (tick - register["birth_tick"].to_numpy(dtype=np.int64)) // 12
    state = np.asarray(data["county_state"], dtype=np.int64)[
        register["county"].to_numpy(dtype=np.int64)
    ]
    register_elder = np.bincount(
        state,
        weights=(register["age"].to_numpy(dtype=np.int64) >= 65),
        minlength=n_states,
    )

    actuarial_bars = (bars or {}).get("actuarial", {})
    exposure_ceiling = _finite(actuarial_bars.get("exposure_error_ceiling"))
    mortality_ceiling = _finite(actuarial_bars.get("mortality_error_ceiling"))
    retained_exposure = retained["exposure"].to_numpy(dtype=np.float64)
    retained_mortality = retained["mortality"].to_numpy(dtype=np.float64)
    public_exposure = years * last_exposure
    register_exposure = years * register_elder
    public_share = public_exposure / max(public_exposure.sum(), 1e-12)
    retained_share = retained_exposure / max(retained_exposure.sum(), 1e-12)
    rows = []
    for state_index in range(n_states):
        exposure_error = (
            public_exposure[state_index] / max(retained_exposure[state_index], 1e-12)
            - 1.0
        )
        mortality_error = (
            historical_mortality[state_index]
            / max(retained_mortality[state_index], 1e-12)
            - 1.0
        )
        public_survival = float(np.exp(-years * historical_mortality[state_index]))
        retained_survival = float(np.exp(-years * retained_mortality[state_index]))
        rows.append(
            {
                "state": state_index,
                "public_experience_person_years": float(public_exposure[state_index]),
                "sealed_person_years": float(retained_exposure[state_index]),
                "public_experience_level_relative_error": float(exposure_error),
                "public_experience_state_share_error": float(
                    public_share[state_index] - retained_share[state_index]
                ),
                "register_level_relative_error": float(
                    register_exposure[state_index]
                    / max(retained_exposure[state_index], 1e-12)
                    - 1.0
                ),
                "public_experience_mortality_rate": float(
                    historical_mortality[state_index]
                ),
                "sealed_mortality_rate": float(retained_mortality[state_index]),
                "mortality_relative_error": float(mortality_error),
                "public_experience_survival": public_survival,
                "sealed_survival": retained_survival,
                "survival_difference": public_survival - retained_survival,
                "exposure_within_aggregate_proxy_ceiling": None
                if exposure_ceiling is None
                else bool(abs(exposure_error) <= exposure_ceiling),
                "mortality_within_aggregate_proxy_ceiling": None
                if mortality_ceiling is None
                else bool(abs(mortality_error) <= mortality_ceiling),
            }
        )
    return {
        "thresholds": {
            "aggregate_exposure_relative_error_ceiling_copied_from_bar": exposure_ceiling,
            "aggregate_mortality_relative_error_ceiling_copied_from_bar": mortality_ceiling,
            "criterion": "absolute aggregate relative error, not the verifier cell percentile",
        },
        "states": rows,
        "exposure_within_aggregate_proxy_ceiling": None
        if exposure_ceiling is None
        else all(row["exposure_within_aggregate_proxy_ceiling"] for row in rows),
        "mortality_within_aggregate_proxy_ceiling": None
        if mortality_ceiling is None
        else all(row["mortality_within_aggregate_proxy_ceiling"] for row in rows),
    }


def _elder_error_summary(rows: list[dict], exposure_key: str) -> dict:
    exposure = np.abs(np.asarray([row[exposure_key] for row in rows], dtype=np.float64))
    survival = np.abs(
        np.asarray([row["survival_difference"] for row in rows], dtype=np.float64)
    )
    return {
        "median_absolute_exposure_relative_error": float(np.median(exposure)),
        "maximum_absolute_exposure_relative_error": float(exposure.max()),
        "median_absolute_survival_difference": float(np.median(survival)),
        "maximum_absolute_survival_difference": float(survival.max()),
    }


def third_elder_comparison(participant: dict, third: dict) -> dict:
    """Compare the third line with the unfitted public-experience reading."""
    public = _elder_error_summary(
        participant["states"], "public_experience_level_relative_error"
    )
    fitted = _elder_error_summary(third["states"], "exposure_relative_error")
    comparison = {"public_experience": public, "third_line": fitted}
    for statistic in public:
        denominator = max(public[statistic], 1e-12)
        comparison[f"{statistic}_relative_reduction"] = float(
            (public[statistic] - fitted[statistic]) / denominator
        )
    return comparison


def _submitted_reference_summary(report: dict) -> dict:
    elder = report["state_65_plus"]["states"]
    exposure = np.abs(
        np.asarray([row["exposure_relative_error"] for row in elder], dtype=np.float64)
    )
    mortality = np.abs(
        np.asarray(
            [row["mortality_rate_ratio"] - 1.0 for row in elder], dtype=np.float64
        )
    )
    survival = np.abs(
        np.asarray([row["survival_difference"] for row in elder], dtype=np.float64)
    )
    regional = report["regional_liability_means"]
    level = np.abs(
        np.asarray([row["relative_error"] for row in regional], dtype=np.float64)
    )
    level_width = np.abs(
        np.asarray(
            [row["level_error_in_tail_widths"] for row in regional], dtype=np.float64
        )
    )
    reserve = report["reserve"]
    return {
        "elder_exposure_median_absolute_relative_error": float(np.median(exposure)),
        "elder_exposure_maximum_absolute_relative_error": float(exposure.max()),
        "elder_mortality_median_absolute_relative_error": float(np.median(mortality)),
        "elder_mortality_maximum_absolute_relative_error": float(mortality.max()),
        "elder_survival_median_absolute_difference": float(np.median(survival)),
        "elder_survival_maximum_absolute_difference": float(survival.max()),
        "liability_mean_median_absolute_relative_error": float(np.median(level)),
        "liability_mean_maximum_absolute_relative_error": float(level.max()),
        "liability_level_maximum_absolute_tail_widths": float(level_width.max()),
        "tail_mean_quantile_score": reserve["mean_quantile_score"],
        "tail_mean_shortfall_error": reserve["mean_shortfall_error"],
        "reserve_skill": reserve["skill"],
        "reserve_J": reserve["J"],
    }


def third_reference_deltas(third: dict, reference_a: dict, reference_b: dict) -> dict:
    """Raw elder, liability-level, tail, and reserve deltas against both lines."""
    rows = {
        "A": _submitted_reference_summary(reference_a),
        "B": _submitted_reference_summary(reference_b),
        "third": _submitted_reference_summary(third),
    }
    out = {"raw": rows, "delta_third_minus_A": {}, "delta_third_minus_B": {}}
    for reference in ("A", "B"):
        target = out[f"delta_third_minus_{reference}"]
        for key, third_value in rows["third"].items():
            reference_value = rows[reference][key]
            target[key] = (
                None
                if third_value is None or reference_value is None
                else float(third_value - reference_value)
            )
    return out


def mortality_gap_decomposition(packet_dir: Path) -> dict:
    """Separate observed elder mortality change into trend, shocks, and lag timing."""
    import pandas as pd

    packet = Path(packet_dir)
    contract = json.loads((packet / "participant" / "contract.json").read_text())
    experience = AR.load_experience(packet, contract)
    if experience is None:
        raise AR.MissingActuarialInputs(
            "mortality decomposition needs the published experience file"
        )
    data = load_packet(packet)
    n_states = int(np.asarray(data["county_state"]).max()) + 1
    arrays = AR.experience_arrays(experience, n_states)
    elder = np.asarray([low >= 65 for low, _ in AR.ACTUARIAL_AGE_BANDS], dtype=bool)
    history_exposure = float(np.asarray(arrays["exposure"][:, :, elder, :]).sum())
    history_deaths = float(np.asarray(arrays["deaths"][:, :, elder, :]).sum())
    history_rate = history_deaths / max(history_exposure, 1e-12)

    truth = pd.read_csv(packet / "retained" / "rate_truth_horizon.csv")
    elder_labels = {
        label
        for label, (low, _) in zip(
            AR.ACTUARIAL_AGE_BAND_LABELS, AR.ACTUARIAL_AGE_BANDS
        )
        if low >= 65
    }
    truth = truth[
        (truth["level"] == "state")
        & (truth["age_band"].isin(elder_labels))
        & (truth["estimand"].isin(("person_years_exposure", "mortality_rate")))
    ]
    exposure = truth[truth["estimand"] == "person_years_exposure"]
    mortality = truth[truth["estimand"] == "mortality_rate"]
    keys = ["level", "unit", "sex", "age_band"]
    joined = exposure[keys + ["value"]].rename(
        columns={"value": "exposure"}
    ).merge(
        mortality[keys + ["value"]].rename(columns={"value": "mortality"}),
        on=keys,
        validate="one_to_one",
    )
    horizon_exposure = float(joined["exposure"].sum())
    horizon_rate = float(
        (joined["exposure"] * joined["mortality"]).sum()
        / max(horizon_exposure, 1e-12)
    )

    retained = json.loads((packet / "retained" / "world.json").read_text())
    improvement = float(
        retained["mechanisms"]["coefficients"]["mortality_improvement"]
    )
    history = contract["experience_history"]
    first_tick = int(history["first_year_starts_at_tick"])
    last_tick = int(history["last_year_ends_at_tick"])
    revised_tick = int(contract["ticks"]["revised"])
    horizon_tick = int(contract["ticks"]["horizon"])
    history_midpoint = 0.5 * (first_tick + last_tick)
    horizon_midpoint = 0.5 * (revised_tick + horizon_tick)
    trend_ratio = float(
        np.power(
            max(1.0 - improvement, 1e-12),
            (horizon_midpoint - history_midpoint) / 12.0,
        )
    )
    observed_ratio = horizon_rate / max(history_rate, 1e-12)
    last_exposure_midpoint = last_tick - 6.0
    publication_lag_factor = float(
        np.power(
            max(1.0 - improvement, 1e-12),
            float(history["publication_lag_months"]) / 12.0,
        )
    )
    midpoint_to_snapshot_factor = float(
        np.power(
            max(1.0 - improvement, 1e-12),
            (revised_tick - last_exposure_midpoint) / 12.0,
        )
    )

    shocks = retained.get("shocks", [])

    def mortality_shock_years(start_tick: int, stop_tick: int) -> list[int]:
        return sorted(
            {
                int(shock["year"])
                for shock in shocks
                if "mortality_multiplier" in shock
                and start_tick <= 12 * int(shock["year"]) < stop_tick
            }
        )

    return {
        "hidden_mortality_improvement": improvement,
        "trend_active_during_public_experience_window": True,
        "trend_starts_only_after_public_window": False,
        "trend_application": "all event months relative to the snapshot tick",
        "history_mortality_rate": history_rate,
        "horizon_mortality_rate": horizon_rate,
        "observed_horizon_to_history_ratio": observed_ratio,
        "trend_only_horizon_to_history_ratio": trend_ratio,
        "residual_observed_to_trend_ratio": observed_ratio / max(trend_ratio, 1e-12),
        "publication_lag_months": int(history["publication_lag_months"]),
        "last_exposure_midpoint_to_snapshot_months": int(
            revised_tick - last_exposure_midpoint
        ),
        "publication_lag_trend_factor": publication_lag_factor,
        "last_exposure_midpoint_to_snapshot_trend_factor": (
            midpoint_to_snapshot_factor
        ),
        "history_mortality_shock_years": mortality_shock_years(
            first_tick, last_tick
        ),
        "lag_mortality_shock_years": mortality_shock_years(last_tick, revised_tick),
        "designated_horizon_mortality_shock_years": mortality_shock_years(
            revised_tick, horizon_tick
        ),
        "continuation_shocks_redrawn_per_member": True,
    }


def elder_eligibility_audit(packet_dir: Path, floor: float = 500.0) -> dict:
    """Report the proposed broad elder rate cell floor and the finer diagnostics."""
    import pandas as pd

    packet = Path(packet_dir)
    contract = json.loads((packet / "participant" / "contract.json").read_text())
    n_states = int(contract["n_states"])
    truth = pd.read_csv(packet / "retained" / "rate_truth_horizon.csv")
    elder_labels = ("65-74", "75-84", "85+")
    cells = truth[
        (truth["level"] == "state")
        & (truth["estimand"] == "person_years_exposure")
        & (truth["age_band"].isin(elder_labels))
    ]
    _require_elder_cells(
        cells, n_states, set(elder_labels), "retained eligibility exposure"
    )
    broad = cells.groupby(["unit", "sex"], sort=True)["value"].sum()
    return {
        "scored": {
            "age_band": "65+",
            "floor_person_years": float(floor),
            "n_cells": int(len(broad)),
            "n_eligible_cells": int((broad >= floor).sum()),
            "all_cells_eligible": bool((broad >= floor).all()),
            "minimum_sealed_person_years": float(broad.min()),
        },
        "report_only": [
            {
                "age_band": label,
                "n_cells": int(len(cells[cells["age_band"] == label])),
                "minimum_sealed_person_years": float(
                    cells[cells["age_band"] == label]["value"].min()
                ),
            }
            for label in elder_labels
        ],
    }


def _method_source_digest() -> tuple[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    files = sorted((repo_root / "meridia").rglob("*.py"))
    payload = {
        str(path.relative_to(repo_root)): _sha256(path) for path in sorted(files)
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    source_sha256 = hashlib.sha256(encoded).hexdigest()
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source_sha256, git_commit


def _exposure_l1_percent(audit: dict) -> float:
    estimated = np.asarray(
        [row["estimated_person_years"] for row in audit["states"]],
        dtype=np.float64,
    )
    sealed = np.asarray(
        [row["sealed_person_years"] for row in audit["states"]],
        dtype=np.float64,
    )
    return float(100.0 * np.abs(estimated - sealed).sum() / max(sealed.sum(), 1e-12))


def _audit_world(packet: Path, before: dict, after: dict) -> dict:
    unavailable = [
        name
        for name, row in (("before", before), ("after", after))
        if not row["hard_check_pass"]
    ]
    base = {
        "world": packet.name,
        "before_report_evidence_id": before["evidence"]["evidence_id"],
        "after_report_evidence_id": after["evidence"]["evidence_id"],
        "mortality_gap_decomposition": mortality_gap_decomposition(packet),
        "eligibility": elder_eligibility_audit(packet),
    }
    if unavailable:
        return base | {
            "available": False,
            "unavailable_reason": "hard-invalid scored report: " + ", ".join(unavailable),
            "exposure_65_plus_absolute_error_percent": {"before": None, "after": None},
            "state_65_plus_person_years": None,
            "liability_mean_by_region": None,
            "pooled_exceedance_deviation": {"before": None, "after": None},
        }

    before_elder = before["state_65_plus"]
    after_elder = after["state_65_plus"]
    before_states = {row["state"]: row for row in before_elder["states"]}
    after_states = {row["state"]: row for row in after_elder["states"]}
    if set(before_states) != set(after_states):
        raise ValueError("elder audit lines do not contain the same states")
    before_regions = {
        row["region"]: row for row in before["regional_liability_means"]
    }
    after_regions = {
        row["region"]: row for row in after["regional_liability_means"]
    }
    if set(before_regions) != set(after_regions):
        raise ValueError("liability audit lines do not contain the same regions")
    return base | {
        "available": True,
        "unavailable_reason": None,
        "exposure_65_plus_absolute_error_percent": {
            "definition": (
                "100 * sum_state abs(submitted_state_65plus_person_years - "
                "sealed_state_65plus_person_years) / sum_state "
                "sealed_state_65plus_person_years"
            ),
            "before": _exposure_l1_percent(before_elder),
            "after": _exposure_l1_percent(after_elder),
        },
        "state_65_plus_person_years": [
            {
                "state": int(state),
                "submitted_before": float(before_states[state]["estimated_person_years"]),
                "submitted_after": float(after_states[state]["estimated_person_years"]),
                "sealed": float(before_states[state]["sealed_person_years"]),
            }
            for state in sorted(before_states)
        ],
        "liability_mean_by_region": [
            {
                "region": int(region),
                "submitted_before": float(before_regions[region]["submitted_mean"]),
                "submitted_after": float(after_regions[region]["submitted_mean"]),
                "sealed": float(before_regions[region]["sealed_mean"]),
            }
            for region in sorted(before_regions)
        ],
        "pooled_exceedance_deviation": {
            "definition": (
                "mean_region abs(sealed Pr(L > submitted_q95) - 0.05)"
            ),
            "before": float(
                before["sealed_exceedance"]["pooled_exceedance_deviation"]
            ),
            "after": float(
                after["sealed_exceedance"]["pooled_exceedance_deviation"]
            ),
        },
    }


def write_elder_reconstruction_audit(
    report: dict, qualification_packets: list[Path], out_dir: Path
) -> dict:
    """Write the six-world cohort-component audit as JSON and plain text."""
    source_sha256, git_commit = _method_source_digest()
    if len(source_sha256) != 64 or source_sha256.lower() != source_sha256:
        raise ValueError("source digest is not lowercase sha256")
    worlds = []
    shock_ranges = None
    for packet in qualification_packets:
        rows = report["qualification"][packet.name]["methods"]
        worlds.append(_audit_world(packet, rows["A"], rows["third"]))
        contract = json.loads((packet / "participant" / "contract.json").read_text())
        family = contract["shock_family"]
        current = {
            "annual_probability": float(family["annual_rate"]),
            "independent_per_member": True,
            "magnitude_source": "participant/contract.json:shock_family",
            "mortality_ranges": [
                {"kind": str(kind), "range": list(fields["mortality_multiplier"])}
                for kind, fields in family["kinds"].items()
                if "mortality_multiplier" in fields
            ],
            "admission_ranges": [
                {"kind": str(kind), "range": list(fields["admission_multiplier"])}
                for kind, fields in family["kinds"].items()
                if "admission_multiplier" in fields
            ],
        }
        if current["annual_probability"] != 0.20:
            raise ValueError("qualification shock family annual probability is not 0.20")
        if shock_ranges is None:
            shock_ranges = current
        elif shock_ranges != current:
            raise ValueError("qualification shock families do not match")
    if len(worlds) != 6:
        raise ValueError("elder reconstruction audit requires exactly six worlds")
    payload = {
        "schema": "meridia.methods.elder_reconstruction_audit.v1",
        "method_digest": {
            "git_commit": git_commit,
            "source_sha256": source_sha256,
            "before_line": "A",
            "after_line": "third_cohort_component",
        },
        "shock_redraw": shock_ranges,
        "eligibility_audit": {
            "scored": {"age_band": "65+", "floor_person_years": 500.0},
            "report_only": ["65-74", "75-84", "85+"],
            "younger_floors_changed": False,
        },
        "worlds": worlds,
    }
    out_dir = _prepare_output_dir(out_dir)
    json_path = out_dir / "elder_reconstruction_audit.json"
    text_path = out_dir / "elder_reconstruction_audit.txt"
    _write_json_atomic(out_dir, json_path, payload, "elder reconstruction audit")
    lines = [
        "Meridia elder reconstruction audit v1",
        f"Git commit: {git_commit}",
        f"Source sha256: {source_sha256}",
        "Broad 65-plus is scored at 500 person-years. Finer elder bands are report-only.",
    ]
    for world in worlds:
        lines.append("")
        lines.append(world["world"])
        if not world["available"]:
            lines.append(f"Unavailable: {world['unavailable_reason']}")
            continue
        exposure_error = world["exposure_65_plus_absolute_error_percent"]
        tail = world["pooled_exceedance_deviation"]
        lines.append(
            "65-plus exposure weighted absolute error percent, before/after: "
            f"{exposure_error['before']:.6f} / {exposure_error['after']:.6f}"
        )
        lines.append(
            "Pooled exceedance deviation, before/after: "
            f"{tail['before']:.6f} / {tail['after']:.6f}"
        )
        for row in world["liability_mean_by_region"]:
            lines.append(
                f"Region {row['region']} liability mean, before/after/sealed: "
                f"{row['submitted_before']:.6f} / {row['submitted_after']:.6f} / "
                f"{row['sealed']:.6f}"
            )
    _write_text_atomic(
        out_dir,
        text_path,
        "\n".join(lines) + "\n",
        "elder reconstruction audit text",
    )
    return {
        "json_path": str(json_path.resolve()),
        "text_path": str(text_path.resolve()),
        "payload": payload,
    }


def _submission_hashes(submission: Path) -> dict[str, str]:
    """Hash every flat submission file without following linked entries."""
    submission = Path(submission)
    if submission.is_symlink() or not submission.is_dir():
        raise ValueError("method submission must be a real directory")
    entries = list(submission.iterdir())
    linked = sorted(path.name for path in entries if path.is_symlink())
    if linked:
        raise ValueError(f"method submission contains symlinks: {linked}")
    present = sorted(path.name for path in entries if path.is_file())
    nested = sorted(path.name for path in entries if not path.is_file())
    if nested:
        raise ValueError(f"method submission contains nested entries: {nested}")
    return {name: _sha256(submission / name) for name in present}


def _run_receipt_path(output_root: Path, submission: Path) -> Path:
    return _assert_output_location(
        output_root,
        Path(submission).parent / f".{Path(submission).name}.run_receipt.json",
        "method run receipt",
    )


def _write_json_atomic(
    output_root: Path, path: Path, payload: dict, label: str
) -> None:
    path = _assert_output_location(output_root, path, label)
    temporary = _assert_output_location(
        output_root, path.with_name(f".{path.name}.tmp"), label
    )
    if temporary.exists():
        raise ValueError(f"partial {label} is present")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_text_atomic(
    output_root: Path, path: Path, content: str, label: str
) -> None:
    path = _assert_output_location(output_root, path, label)
    temporary = _assert_output_location(
        output_root, path.with_name(f".{path.name}.tmp"), label
    )
    if temporary.exists():
        raise ValueError(f"partial {label} is present")
    temporary.write_text(content)
    temporary.replace(path)


def _expected_run_receipt(
    output_root: Path,
    packet: Path,
    submission: Path,
    measurement_contract_sha256: str,
    run_spec: dict,
) -> dict:
    _, manifest_path = _read_packet_manifest(packet)
    normalized_spec = json.loads(json.dumps(run_spec, sort_keys=True))
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "measurement_contract_sha256": measurement_contract_sha256,
        "packet": str(Path(packet).resolve()),
        "packet_manifest_sha256": _sha256(manifest_path),
        "submission": str(Path(submission).resolve()),
        "run_spec": normalized_spec,
        "output_sha256": _submission_hashes(submission),
    }


def _validate_run_receipt(
    output_root: Path,
    packet: Path,
    submission: Path,
    measurement_contract_sha256: str,
    run_spec: dict,
) -> bool:
    receipt_path = _run_receipt_path(output_root, submission)
    if not receipt_path.exists():
        return False
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("method run receipt must be a regular file")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("method run receipt is incomplete or invalid") from error
    expected = _expected_run_receipt(
        output_root,
        packet,
        submission,
        measurement_contract_sha256,
        run_spec,
    )
    if receipt != expected:
        raise ValueError("method run receipt or output does not match this run")
    return True


def _final_evidence_wrapper(packet: Path, submission: Path, bars: dict) -> dict:
    submission = Path(submission)
    if submission.is_symlink() or not submission.is_dir():
        raise ValueError("evidence submission must be a real directory")
    entries = list(submission.iterdir())
    linked = sorted(path.name for path in entries if path.is_symlink())
    if linked:
        raise ValueError(f"evidence submission contains symlinks: {linked}")
    present = sorted(path.name for path in entries if path.is_file())
    nested = sorted(path.name for path in entries if not path.is_file())
    accepted = set(SUBMISSION_FILES + OPTIONAL_SUBMISSION_FILES)
    missing = sorted(set(SUBMISSION_FILES) - set(present))
    unexpected = sorted((set(present) - accepted) | set(nested))
    submission_sha256 = {name: _sha256(submission / name) for name in present}
    _, manifest_path = _read_packet_manifest(packet)
    payload = {
        "packet": str(Path(packet).resolve()),
        "packet_manifest_sha256": _sha256(manifest_path),
        "submission": str(submission.resolve()),
        "submission_sha256": submission_sha256,
        "missing_required_submission_files": missing,
        "unexpected_submission_files": unexpected,
        "bars": bars,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    evidence_id = hashlib.sha256(encoded).hexdigest()
    return {
        "report_class": "final_control_measurement",
        "evidence_id": evidence_id,
        "replicate_id": f"final-{evidence_id[:20]}",
        "eligible_for_freeze_calibration": False,
    }


def _score(packet: Path, submission: Path, bars: dict, allow_unfrozen: bool) -> dict:
    try:
        report = verify_submission(
            packet, submission, bars, allow_unfrozen=allow_unfrozen
        )
    except (OSError, ValueError, KeyError) as error:
        report = {
            "pass": False,
            "reasons": [
                "schema: verifier raised while parsing the submission: "
                f"{type(error).__name__}: {error}"
            ],
            "reserve": {"feasible": False},
        }
    summary = summarize_report(report)
    evidence = _final_evidence_wrapper(packet, submission, bars)
    if not summary["hard_check_pass"]:
        return summary | {
            "truth_audit_status": "unavailable_due_to_hard_check_failure",
            "regional_liability_means": None,
            "regional_level_within_one_tail_width": None,
            "state_65_plus": None,
            "sealed_exceedance": None,
            "evidence": evidence,
        }
    regional = regional_liability_means(packet, submission)
    return summary | {
        "truth_audit_status": "available",
        "regional_liability_means": regional,
        "regional_level_within_one_tail_width": all(
            row["within_one_tail_width"] for row in regional
        ),
        "state_65_plus": elder_state_exposure_survival(packet, submission, bars),
        "sealed_exceedance": sealed_exceedance_audit(packet, submission),
        "evidence": _final_evidence_wrapper(packet, submission, bars),
    }


def _run_once(
    packet: Path,
    submission: Path,
    runner: Callable[[Path], object],
    bars: dict,
    allow_unfrozen: bool,
    output_root: Path,
    measurement_contract_sha256: str,
    run_spec: dict,
) -> dict:
    output_root = _prepare_output_dir(output_root)
    submission = _assert_output_location(
        output_root, submission, "method submission"
    )
    submission.parent.mkdir(parents=True, exist_ok=True)
    _assert_output_location(output_root, submission.parent, "method submission parent")
    stage = _assert_output_location(
        output_root,
        submission.parent / f".{submission.name}.phase-three-tmp",
        "method submission staging directory",
    )
    if stage.exists():
        raise ValueError(f"partial method submission is present: {stage}")
    if _validate_run_receipt(
        output_root,
        packet,
        submission,
        measurement_contract_sha256,
        run_spec,
    ):
        return _score(packet, submission, bars, allow_unfrozen)
    if submission.exists():
        raise ValueError(f"unreceipted method submission is present: {submission}")
    runner(stage)
    _submission_hashes(stage)
    stage.replace(submission)
    receipt = _expected_run_receipt(
        output_root,
        packet,
        submission,
        measurement_contract_sha256,
        run_spec,
    )
    _write_json_atomic(
        output_root,
        _run_receipt_path(output_root, submission),
        receipt,
        "method run receipt",
    )
    return _score(packet, submission, bars, allow_unfrozen)


def _comparison(third: dict, reference_a: dict, reference_b: dict) -> dict:
    hard_invalid = [
        label
        for label, report in (
            ("third", third),
            ("A", reference_a),
            ("B", reference_b),
        )
        if not report["hard_check_pass"]
    ]
    if hard_invalid:
        return {
            "valid": False,
            "status": "indeterminate_due_to_hard_check_failure",
            "hard_invalid": hard_invalid,
            "composites": {},
        }

    out = {}
    for family in COMPOSITE_FAMILIES:
        third_pass = family not in third["failed_composites"]
        a_pass = family not in reference_a["failed_composites"]
        b_pass = family not in reference_b["failed_composites"]
        if third_pass == a_pass == b_pass:
            relative = "matches_both"
        elif third_pass and not a_pass and not b_pass:
            relative = "better_than_both"
        elif not third_pass and a_pass and b_pass:
            relative = "worse_than_both"
        elif third_pass == a_pass:
            relative = "matches_A"
        else:
            relative = "matches_B"
        out[family] = {
            "third_pass": third_pass,
            "A_pass": a_pass,
            "B_pass": b_pass,
            "relative": relative,
        }
    return {
        "valid": True,
        "status": "available",
        "hard_invalid": [],
        "composites": out,
    }


def _reserve_change(reference: dict, deletion: dict) -> dict:
    hard_invalid = [
        label
        for label, report in (("A", reference), ("deletion", deletion))
        if not report["hard_check_pass"]
    ]
    if hard_invalid:
        return {
            "valid": False,
            "status": "indeterminate_due_to_hard_check_failure",
            "hard_invalid": hard_invalid,
            "failed_composites_changed": None,
            "J_delta": None,
            "skill_delta": None,
            "mean_quantile_score_delta": None,
            "mean_shortfall_error_delta": None,
            "changed": None,
        }
    ref = reference["reserve"]
    changed = reference["failed_composites"] != deletion["failed_composites"]
    out = {
        "valid": True,
        "status": "available",
        "hard_invalid": [],
        "failed_composites_changed": changed,
    }
    for key in ("J", "skill", "mean_quantile_score", "mean_shortfall_error"):
        before, after = ref.get(key), deletion["reserve"].get(key)
        delta = None if before is None or after is None else float(after - before)
        out[f"{key}_delta"] = delta
        if delta is not None and abs(delta) > 1e-9 * max(1.0, abs(before)):
            changed = True
    out["changed"] = changed
    return out


def _validate_packet_group(
    packets: list[Path], expected_count: int, development: bool
) -> list[Path]:
    label = "development" if development else "qualification"
    if len(packets) != expected_count:
        raise ValueError(
            f"phase three requires exactly {expected_count} {label} worlds"
        )
    resolved = [Path(path).resolve() for path in packets]
    if any(
        part.lower().startswith("graded")
        for packet in resolved
        for part in packet.parts
    ):
        raise ValueError("phase-three measurements refuse graded packet paths")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"phase-three {label} packet paths must be distinct")
    names = [path.name for path in resolved]
    if len(set(names)) != len(names):
        raise ValueError(f"phase-three {label} world names must be distinct")
    parents = {path.parent for path in resolved}
    if len(parents) != 1:
        raise ValueError(
            f"phase-three {label} packets must share one resolved parent"
        )
    if expected_count == 12 and development:
        expected_names = {f"dev-{index:02d}" for index in range(12)}
        if set(names) != expected_names or any(
            path.parent.name != "development" for path in resolved
        ):
            raise ValueError(
                "phase-three development packets must be the canonical dev-00..dev-11 set"
            )
    if expected_count == 6 and not development:
        expected_names = {f"qual-{index}" for index in range(6)}
        if set(names) != expected_names or any(
            path.parent.name != "qualification" for path in resolved
        ):
            raise ValueError(
                "phase-three qualification packets must be the canonical qual-0..qual-5 set"
            )
    for packet in resolved:
        manifest, _ = _read_packet_manifest(packet)
        if manifest.get("development") is not development:
            raise ValueError(f"{packet} is not a {label} packet")
    return resolved


def _validate_shared_worlds_root(
    development_packets: list[Path], qualification_packets: list[Path]
) -> None:
    development_parent = development_packets[0].parent
    qualification_parent = qualification_packets[0].parent
    if development_parent.parent != qualification_parent.parent:
        raise ValueError(
            "phase-three development and qualification packets must share one worlds root"
        )


def measure_elder_reconstruction(
    development_packets: list[Path],
    qualification_packets: list[Path],
    out_dir: Path,
    bars_path: Path,
    params: MeasurementParams = MeasurementParams(),
    allow_unfrozen: bool = False,
) -> dict:
    """Run only A and the cohort-component third line for the elder audit."""
    development_packets = _validate_packet_group(development_packets, 12, True)
    qualification_packets = _validate_packet_group(qualification_packets, 6, False)
    _validate_shared_worlds_root(development_packets, qualification_packets)
    out_dir = _prepare_output_dir(out_dir)
    contract = _measurement_contract(
        development_packets, qualification_packets, Path(bars_path), params
    )
    contract["measurement_scope"] = "elder_reconstruction_before_after"
    _bind_measurement_output(out_dir, contract)
    contract_sha256 = _measurement_contract_sha256(out_dir)
    bars = json.loads(Path(bars_path).read_text())
    if bars.get("frozen") is not True and not allow_unfrozen:
        raise ValueError("elder measurement requires frozen bars or --allow-unfrozen")
    calibration_a = _ensure_calibration_artifact(
        out_dir,
        "A",
        lambda path: A.calibrate(development_packets, path),
    )
    shared_layer = AR.LayerParams(
        simulation=AR.SimulationParams(n_paths=params.simulation_paths),
        calibrate_tail_to_total=False,
    )
    report = {
        "report_class": "elder_reconstruction_before_after",
        "qualification": {},
        "reserve_total_used_for_tail_calibration": False,
    }
    report_path = out_dir / "elder_reconstruction_measurements.json"
    for packet in qualification_packets:
        root = out_dir / "qualification" / packet.name
        before = _run_once(
            packet,
            root / "A",
            lambda o, p=packet: A.run(
                p,
                o,
                A.MethodParams(
                    bootstrap_replicates=params.bootstrap_replicates,
                    calibration_path=str(calibration_a),
                    actuarial="on",
                    actuarial_params=shared_layer,
                ),
            ),
            bars,
            allow_unfrozen,
            out_dir,
            contract_sha256,
            {
                "method": "A",
                "bootstrap_replicates": params.bootstrap_replicates,
                "simulation_paths": params.simulation_paths,
                "calibration_sha256": _sha256(calibration_a),
                "calibrate_tail_to_total": False,
            },
        )
        after = _run_once(
            packet,
            root / "third",
            lambda o, p=packet: C.run(
                p,
                o,
                C.ThirdReferenceParams(
                    bootstrap_replicates=params.bootstrap_replicates,
                    linkage_bootstraps=params.linkage_bootstraps,
                    simulation_paths=params.simulation_paths,
                    calibration_path=str(calibration_a),
                ),
            ),
            bars,
            allow_unfrozen,
            out_dir,
            contract_sha256,
            {
                "method": "third_cohort_component",
                "bootstrap_replicates": params.bootstrap_replicates,
                "linkage_bootstraps": params.linkage_bootstraps,
                "simulation_paths": params.simulation_paths,
                "calibration_sha256": _sha256(calibration_a),
                "calibrate_tail_to_total": False,
            },
        )
        report["qualification"][packet.name] = {
            "methods": {"A": before, "third": after}
        }
        _write_json_atomic(
            out_dir, report_path, report, "elder reconstruction measurements"
        )
    audit = write_elder_reconstruction_audit(report, qualification_packets, out_dir)
    report["elder_reconstruction_audit"] = {
        "json_path": audit["json_path"],
        "text_path": audit["text_path"],
        "source_sha256": audit["payload"]["method_digest"]["source_sha256"],
        "git_commit": audit["payload"]["method_digest"]["git_commit"],
    }
    _write_json_atomic(
        out_dir, report_path, report, "elder reconstruction measurements"
    )
    return report


def measure(
    development_packets: list[Path],
    qualification_packets: list[Path],
    out_dir: Path,
    bars_path: Path,
    params: MeasurementParams = MeasurementParams(),
    allow_unfrozen: bool = False,
) -> dict:
    """Run and record every phase-three comparison in a restartable output tree."""
    development_packets = _validate_packet_group(development_packets, 12, True)
    qualification_packets = _validate_packet_group(qualification_packets, 6, False)
    _validate_shared_worlds_root(development_packets, qualification_packets)
    out_dir = _prepare_output_dir(out_dir)
    _bind_measurement_output(
        out_dir,
        _measurement_contract(
            development_packets, qualification_packets, Path(bars_path), params
        ),
    )
    contract_sha256 = _measurement_contract_sha256(out_dir)
    bars = json.loads(Path(bars_path).read_text())
    if bars.get("frozen") is not True and not allow_unfrozen:
        raise ValueError("phase-three measurements require a completed frozen bar set")
    calibration_a = _ensure_calibration_artifact(
        out_dir,
        "A",
        lambda path: A.calibrate(development_packets, path),
    )

    shared_layer = AR.LayerParams(
        simulation=AR.SimulationParams(n_paths=params.simulation_paths),
        calibrate_tail_to_total=False,
    )
    report = {
        "composites": list(COMPOSITE_FAMILIES),
        "reserve_total_used_for_tail_calibration": False,
        "qualification": {},
        "development_decomposition": {},
        "third_line_position": {},
        "participant_elder_identifiability": {},
        "third_elder_comparison": {},
        "third_reference_deltas": {},
        "control_failures": {},
        "control_hard_failures": {},
        "control_targets": dict(controls.CONTROL_TARGET_COMPOSITES),
        "qualification_control_names": list(controls.QUALIFICATION_CONTROLS),
        "control_target_results": {},
        "gate_control_deletion_candidates": [],
        "gate_retention": {},
        "deletion_candidates": [],
        "deletion_indeterminate": {},
    }

    for packet in development_packets:
        world = packet.name
        root = out_dir / "development" / world
        rows = {}
        for name in controls.DECOMPOSITION_CONTROLS:
            rows[name] = _run_once(
                packet,
                root / name,
                lambda o, n=name, p=packet: controls.run_decomposition(
                    n,
                    p,
                    o,
                    calibration_path=str(calibration_a),
                    bootstrap_replicates=params.bootstrap_replicates,
                    simulation_paths=params.simulation_paths,
                ),
                bars,
                allow_unfrozen,
                out_dir,
                contract_sha256,
                {
                    "method": f"decomposition/{name}",
                    "bootstrap_replicates": params.bootstrap_replicates,
                    "simulation_paths": params.simulation_paths,
                    "calibration_sha256": _sha256(calibration_a),
                    "calibrate_tail_to_total": False,
                },
            )
        report["development_decomposition"][world] = rows
        _write_json_atomic(
            out_dir,
            out_dir / "phase_three_measurements.json",
            report,
            "phase three measurements",
        )

    calibration_b = _ensure_calibration_artifact(
        out_dir,
        "B",
        lambda path: B.calibrate(development_packets, path),
    )

    for packet in qualification_packets:
        world = packet.name
        root = out_dir / "qualification" / world
        participant_elder = participant_elder_identifiability(packet, bars)
        report["participant_elder_identifiability"][world] = participant_elder
        methods = {}
        methods["A"] = _run_once(
            packet,
            root / "A",
            lambda o, p=packet: A.run(
                p,
                o,
                A.MethodParams(
                    bootstrap_replicates=params.bootstrap_replicates,
                    calibration_path=str(calibration_a),
                    actuarial="on",
                    actuarial_params=shared_layer,
                ),
            ),
            bars,
            allow_unfrozen,
            out_dir,
            contract_sha256,
            {
                "method": "A",
                "bootstrap_replicates": params.bootstrap_replicates,
                "simulation_paths": params.simulation_paths,
                "calibration_sha256": _sha256(calibration_a),
                "calibrate_tail_to_total": False,
            },
        )
        reconstruction = _run_once(
            packet,
            root / "deletions" / "reconstruction_uncertainty",
            lambda o, p=packet: (
                controls.run_deletion(
                    "reconstruction_uncertainty",
                    p,
                    o,
                    calibration_path=str(calibration_a),
                    bootstrap_replicates=params.bootstrap_replicates,
                    simulation_paths=params.simulation_paths,
                )
            ),
            bars,
            allow_unfrozen,
            out_dir,
            contract_sha256,
            {
                "method": "deletion/reconstruction_uncertainty",
                "bootstrap_replicates": params.bootstrap_replicates,
                "simulation_paths": params.simulation_paths,
                "calibration_sha256": _sha256(calibration_a),
                "calibrate_tail_to_total": False,
            },
        )
        reconstruction["change_from_A"] = _reserve_change(methods["A"], reconstruction)
        methods["B"] = _run_once(
            packet,
            root / "B",
            lambda o, p=packet: B.run(
                p,
                o,
                B.MethodParams(
                    sweeps=params.bayesian_sweeps,
                    burn_in=params.bayesian_sweeps // 4,
                    calibration_path=str(calibration_b),
                    actuarial="on",
                    actuarial_params=shared_layer,
                ),
            ),
            bars,
            allow_unfrozen,
            out_dir,
            contract_sha256,
            {
                "method": "B",
                "sweeps": params.bayesian_sweeps,
                "burn_in": params.bayesian_sweeps // 4,
                "simulation_paths": params.simulation_paths,
                "calibration_sha256": _sha256(calibration_b),
                "calibrate_tail_to_total": False,
            },
        )
        methods["third"] = _run_once(
            packet,
            root / "third",
            lambda o, p=packet: C.run(
                p,
                o,
                C.ThirdReferenceParams(
                    bootstrap_replicates=params.bootstrap_replicates,
                    linkage_bootstraps=params.linkage_bootstraps,
                    simulation_paths=params.simulation_paths,
                    calibration_path=str(calibration_a),
                ),
            ),
            bars,
            allow_unfrozen,
            out_dir,
            contract_sha256,
            {
                "method": "third_cohort_component",
                "bootstrap_replicates": params.bootstrap_replicates,
                "linkage_bootstraps": params.linkage_bootstraps,
                "simulation_paths": params.simulation_paths,
                "calibration_sha256": _sha256(calibration_a),
                "calibrate_tail_to_total": False,
            },
        )
        control_rows = {}
        for name in controls.ALL_CONTROLS:
            control_rows[name] = _run_once(
                packet,
                root / "controls" / name,
                lambda o, n=name, p=packet: controls.run(
                    n,
                    p,
                    o,
                    calibration_path=str(calibration_a),
                    simulation_paths=params.simulation_paths,
                ),
                bars,
                allow_unfrozen,
                out_dir,
                contract_sha256,
                {
                    "method": f"control/{name}",
                    "simulation_paths": params.simulation_paths,
                    "calibration_sha256": _sha256(calibration_a),
                    "calibrate_tail_to_total": False,
                    "target_composite": controls.CONTROL_TARGET_COMPOSITES[name],
                },
            )
        deletions = {"reconstruction_uncertainty": reconstruction}
        for name in controls.DELETION_CONTROLS:
            if name == "reconstruction_uncertainty":
                continue
            deletion = _run_once(
                packet,
                root / "deletions" / name,
                lambda o, n=name, p=packet: (
                    controls.run_deletion(
                        n,
                        p,
                        o,
                        calibration_path=str(calibration_a),
                        bootstrap_replicates=params.bootstrap_replicates,
                        simulation_paths=params.simulation_paths,
                    )
                ),
                bars,
                allow_unfrozen,
                out_dir,
                contract_sha256,
                {
                    "method": f"deletion/{name}",
                    "bootstrap_replicates": params.bootstrap_replicates,
                    "simulation_paths": params.simulation_paths,
                    "calibration_sha256": _sha256(calibration_a),
                    "calibrate_tail_to_total": False,
                },
            )
            deletion["change_from_A"] = _reserve_change(methods["A"], deletion)
            deletions[name] = deletion
        report["qualification"][world] = {
            "methods": methods,
            "controls": control_rows,
            "deletions": deletions,
        }
        report["third_line_position"][world] = _comparison(
            methods["third"],
            methods["A"],
            methods["B"],
        )
        hard_invalid_references = report["third_line_position"][world]["hard_invalid"]
        if hard_invalid_references:
            unavailable = {
                "valid": False,
                "status": "indeterminate_due_to_hard_check_failure",
                "hard_invalid": hard_invalid_references,
            }
            report["third_elder_comparison"][world] = unavailable
            report["third_reference_deltas"][world] = dict(unavailable)
        else:
            report["third_elder_comparison"][world] = {
                "valid": True,
                "status": "available",
                "hard_invalid": [],
                **third_elder_comparison(
                    participant_elder, methods["third"]["state_65_plus"]
                ),
            }
            report["third_reference_deltas"][world] = {
                "valid": True,
                "status": "available",
                "hard_invalid": [],
                **third_reference_deltas(methods["third"], methods["A"], methods["B"]),
            }
        report["control_failures"][world] = {
            **{
                name: row["failed_composites"]
                for name, row in control_rows.items()
            },
            **{name: row["failed_composites"] for name, row in deletions.items()},
        }
        report["control_hard_failures"][world] = {
            **{
                name: row["hard_check_failures"]
                for name, row in control_rows.items()
            },
            **{name: row["hard_check_failures"] for name, row in deletions.items()},
        }
        _write_json_atomic(
            out_dir,
            out_dir / "phase_three_measurements.json",
            report,
            "phase three measurements",
        )

    for name in controls.DELETION_CONTROLS:
        hard_invalid = []
        changes = []
        for packet in qualification_packets:
            rows = report["qualification"][packet.name]
            reference = rows["methods"]["A"]
            deletion = rows["deletions"][name]
            if not reference["hard_check_pass"]:
                hard_invalid.append(f"{packet.name}/A")
            if not deletion["hard_check_pass"]:
                hard_invalid.append(f"{packet.name}/{name}")
            if reference["hard_check_pass"] and deletion["hard_check_pass"]:
                changes.append(bool(deletion["change_from_A"]["changed"]))
        if hard_invalid:
            report["deletion_indeterminate"][name] = hard_invalid
            continue
        if not any(changes):
            report["deletion_candidates"].append(name)
    report["version_three_scientific_passed_worlds"] = [
        packet.name
        for packet in qualification_packets
        if report["qualification"][packet.name]["controls"]["version_three_recipe"][
            "composite_pass"
        ]
    ]
    report["version_three_valid_passed_worlds"] = [
        packet.name
        for packet in qualification_packets
        if report["qualification"][packet.name]["controls"]["version_three_recipe"][
            "composite_pass"
        ]
        and report["qualification"][packet.name]["controls"]["version_three_recipe"][
            "hard_check_pass"
        ]
    ]
    report["invalid_control_reports"] = [
        f"{world}/{control}"
        for world, rows in report["control_hard_failures"].items()
        for control, failures in rows.items()
        if failures
    ]
    for name in controls.QUALIFICATION_CONTROLS:
        target = controls.CONTROL_TARGET_COMPOSITES[name]
        worlds = {}
        hard_invalid = []
        passed_target = []
        failed_target = []
        for packet in qualification_packets:
            world = packet.name
            collection = (
                "controls" if name in controls.ALL_CONTROLS else "deletions"
            )
            row = report["qualification"][world][collection][name]
            target_failed = target in row["failed_composites"]
            if not row["hard_check_pass"]:
                hard_invalid.append(world)
            elif target_failed:
                failed_target.append(world)
            else:
                passed_target.append(world)
            worlds[world] = {
                "hard_check_pass": row["hard_check_pass"],
                "hard_check_failures": row["hard_check_failures"],
                "target_failed": target_failed,
                "failed_composites": row["failed_composites"],
                "other_failed_composites": [
                    family
                    for family in row["failed_composites"]
                    if family != target
                ],
                "exact_gate_metrics": row["gate_metrics"],
                "evidence_id": row["evidence"]["evidence_id"],
            }
        status = (
            "indeterminate_due_to_hard_check_failure"
            if hard_invalid
            else "deletion_candidate"
            if passed_target
            else "retain"
        )
        result = {
            "target_composite": target,
            "status": status,
            "retained": status == "retain",
            "hard_invalid_worlds": hard_invalid,
            "target_failed_worlds": failed_target,
            "target_passed_worlds": passed_target,
            "worlds": worlds,
        }
        report["control_target_results"][name] = result
        if status == "deletion_candidate":
            report["gate_control_deletion_candidates"].append(
                {
                    "control": name,
                    "target_composite": target,
                    "target_passed_worlds": passed_target,
                    "exact_gate_metrics_by_world": {
                        world: worlds[world]["exact_gate_metrics"]
                        for world in passed_target
                    },
                }
            )

    for family in COMPOSITE_FAMILIES:
        registered = [
            name
            for name in controls.QUALIFICATION_CONTROLS
            if controls.CONTROL_TARGET_COMPOSITES[name] == family
        ]
        retained = [
            name
            for name in registered
            if report["control_target_results"][name]["status"] == "retain"
        ]
        candidates = [
            name
            for name in registered
            if report["control_target_results"][name]["status"]
            == "deletion_candidate"
        ]
        indeterminate = [
            name
            for name in registered
            if report["control_target_results"][name]["status"]
            == "indeterminate_due_to_hard_check_failure"
        ]
        report["gate_retention"][family] = {
            "status": "retain"
            if len(retained) == len(registered)
            else "blocked_by_control_evidence",
            "retain": len(retained) == len(registered),
            "registered_controls": registered,
            "retained_controls": retained,
            "deletion_candidates": candidates,
            "hard_invalid_controls": indeterminate,
        }
    elder_audit = write_elder_reconstruction_audit(
        report, qualification_packets, out_dir
    )
    report["elder_reconstruction_audit"] = {
        "schema": elder_audit["payload"]["schema"],
        "json_path": elder_audit["json_path"],
        "text_path": elder_audit["text_path"],
        "source_sha256": elder_audit["payload"]["method_digest"]["source_sha256"],
        "git_commit": elder_audit["payload"]["method_digest"]["git_commit"],
    }
    report_path = out_dir / "phase_three_measurements.json"
    _write_json_atomic(
        out_dir, report_path, report, "phase three measurements"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--qualification", nargs="+", required=True)
    parser.add_argument("--bars", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=400)
    parser.add_argument("--simulation-paths", type=int, default=2048)
    parser.add_argument("--linkage-bootstraps", type=int, default=12)
    parser.add_argument("--allow-unfrozen", action="store_true")
    parser.add_argument("--elder-only", action="store_true")
    args = parser.parse_args(argv)
    runner = measure_elder_reconstruction if args.elder_only else measure
    result = runner(
        [Path(path) for path in args.dev],
        [Path(path) for path in args.qualification],
        Path(args.out),
        Path(args.bars),
        MeasurementParams(
            args.bootstrap, args.sweeps, args.simulation_paths, args.linkage_bootstraps
        ),
        allow_unfrozen=args.allow_unfrozen,
    )
    if args.elder_only:
        summary = {
            "qualification_worlds": len(result["qualification"]),
            **result["elder_reconstruction_audit"],
        }
    else:
        summary = {
            "qualification_worlds": len(result["qualification"]),
            "development_worlds": len(result["development_decomposition"]),
            "version_three_scientific_passed_worlds": result[
                "version_three_scientific_passed_worlds"
            ],
            "invalid_control_reports": result["invalid_control_reports"],
            "deletion_candidates": result["deletion_candidates"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
