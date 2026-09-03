"""Paired outer resamples of the participant data for reference-line calibration.

Every reference line receives the same materialized participant resample. Survey
sampling variation is represented by Rao-Wu rescaled PSU multipliers within strata.
Deaths and qualifying events in the public experience table receive independent
Poisson parametric draws conditional on their observed counts. Exposures, migration,
registers, and public contract quantities stay fixed. Method seeds also stay fixed, so
outer variation is participant-data variation rather than a mixture of input and
implementation Monte Carlo noise.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np


OUTER_RESAMPLE_SCHEMA = "meridia.methods.paired_outer_resamples.v1"
REFERENCE_LINES = ("A", "B", "C")
REFERENCE_METHOD_SEEDS = {
    "A": 20260901,
    "B": 20260902,
    "C": 20260905,
}
ORACLE_DIAGNOSTICS = (
    "design_reconstruction_oracle_tail",
    "true_population_normal_tail",
)
SURVEY_FILES = ("survey_preliminary.csv", "survey_revised.csv")
EXPERIENCE_COUNT_COLUMNS = ("deaths", "qualifying_events")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[str, dict[str, int | str]]:
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"participant tree must be a real directory: {root}")
    entries = list(root.rglob("*"))
    linked = sorted(str(path.relative_to(root)) for path in entries if path.is_symlink())
    if linked:
        raise ValueError(f"participant tree contains linked paths: {linked}")
    files = [path for path in entries if path.is_file()]
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    }


def _inventory_digest(inventory: Mapping[str, object]) -> str:
    encoded = json.dumps(
        inventory, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _component_seed(root_seed: int, replicate: int, component: str) -> int:
    encoded = f"meridia-outer-resample-v1:{root_seed}:{replicate}:{component}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


def rao_wu_resample(frame, rng: np.random.Generator):
    """Apply a Rao-Wu rescaled bootstrap to survey design weights.

    For a stratum with ``m`` sampled PSUs, ``m - 1`` PSUs are sampled with
    replacement. A PSU drawn ``k`` times receives multiplier ``m*k/(m-1)``.
    This has expectation one for every original PSU. Singleton strata are retained
    with multiplier one because their between-PSU variance is not identified.
    """
    required = {"stratum", "psu", "design_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"survey is missing resampling columns {missing}")
    out = frame.copy()
    if out.empty:
        raise ValueError("survey has no rows to resample")
    weight = out["design_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("survey design weights must be finite and positive")
    multiplier = np.zeros(len(out), dtype=np.float64)
    strata = sorted(out["stratum"].drop_duplicates().tolist())
    all_psu_factors = []
    evidence = {
        "design": "Rao-Wu rescaled PSU bootstrap within stratum",
        "strata": len(strata),
        "sampled_psus": 0,
        "singleton_strata": 0,
        "zero_multiplier_psus": 0,
    }
    for stratum in strata:
        mask = out["stratum"].to_numpy() == stratum
        psus = np.sort(out.loc[mask, "psu"].drop_duplicates().to_numpy())
        m = len(psus)
        if m == 0:
            continue
        evidence["sampled_psus"] += m
        if m == 1:
            counts = np.ones(1, dtype=np.int64)
            factors = np.ones(1, dtype=np.float64)
            evidence["singleton_strata"] += 1
        else:
            selected = rng.integers(0, m, size=m - 1)
            counts = np.bincount(selected, minlength=m)
            factors = counts.astype(np.float64) * m / (m - 1)
        evidence["zero_multiplier_psus"] += int((counts == 0).sum())
        all_psu_factors.extend(factors.tolist())
        factor_by_psu = dict(zip(psus.tolist(), factors.tolist()))
        multiplier[mask] = out.loc[mask, "psu"].map(factor_by_psu).to_numpy(
            dtype=np.float64
        )
    if not np.isfinite(multiplier).all() or (multiplier < 0).any():
        raise ValueError("Rao-Wu multipliers are invalid")
    out["design_weight"] = weight * multiplier
    evidence["positive_weight_rows"] = int((out["design_weight"] > 0).sum())
    evidence["zero_weight_rows"] = int((out["design_weight"] == 0).sum())
    evidence["mean_psu_multiplier"] = float(np.mean(all_psu_factors))
    return out, evidence


def parametric_experience_resample(frame, rng: np.random.Generator):
    """Redraw public death and qualifying-event counts from plug-in Poisson laws."""
    missing = sorted(set(EXPERIENCE_COUNT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"experience file is missing count columns {missing}")
    out = frame.copy()
    evidence: dict[str, object] = {
        "design": "independent plug-in Poisson counts conditional on public experience",
        "fixed_columns": [
            name for name in frame.columns if name not in EXPERIENCE_COUNT_COLUMNS
        ],
        "counts": {},
    }
    for column in EXPERIENCE_COUNT_COLUMNS:
        observed = frame[column].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(observed).all()
            or (observed < 0).any()
            or not np.allclose(observed, np.rint(observed))
        ):
            raise ValueError(f"experience {column} must be nonnegative integer counts")
        redrawn = rng.poisson(observed).astype(np.int64)
        out[column] = redrawn
        evidence["counts"][column] = {
            "observed_total": int(np.rint(observed).sum()),
            "resampled_total": int(redrawn.sum()),
            "changed_cells": int((redrawn != observed).sum()),
        }
    return out, evidence


def _copy_participant_tree(source: Path, target: Path) -> dict[str, int]:
    """Materialize a read-only input view, hard-linking unchanged regular files."""
    inventory = _inventory(source)
    target.mkdir(parents=True)
    linked, copied = 0, 0
    for name in inventory:
        source_file = source / name
        target_file = target / name
        target_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_file, target_file)
            linked += 1
        except OSError:
            shutil.copy2(source_file, target_file)
            copied += 1
    return {"hardlinked_files": linked, "copied_files": copied}


def _write_frame_replacing_link(frame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.resample-tmp")
    if temporary.exists():
        raise ValueError(f"partial resampled file is present: {temporary}")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def _experience_file(participant: Path) -> str:
    contract_path = participant / "contract.json"
    try:
        contract = json.loads(contract_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("participant contract is missing or invalid") from error
    name = (contract.get("experience_history") or {}).get(
        "file", "experience_history.csv"
    )
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("participant contract names an invalid experience file")
    return name


def _normalized_method_seeds(
    method_seeds: Mapping[str, int] | None,
) -> dict[str, int]:
    seeds = dict(REFERENCE_METHOD_SEEDS if method_seeds is None else method_seeds)
    if set(seeds) != set(REFERENCE_LINES):
        raise ValueError("method seeds must name exactly A, B, and C")
    normalized = {}
    for name in REFERENCE_LINES:
        value = seeds[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"method seed for {name} is not an integer")
        normalized[name] = int(value)
    return normalized


def materialize_paired_outer_resamples(
    packet_dir: Path,
    out_dir: Path,
    *,
    replicates: int,
    seed: int = 20260906,
    method_seeds: Mapping[str, int] | None = None,
) -> dict:
    """Write restartable participant resamples shared by reference lines A, B, and C."""
    import pandas as pd

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("outer resampling seed must be an integer")
    seeds = _normalized_method_seeds(method_seeds)
    packet = Path(packet_dir).resolve()
    participant = packet / "participant"
    source_inventory = _inventory(participant)
    source_digest = _inventory_digest(source_inventory)
    experience_name = _experience_file(participant)
    required = set(SURVEY_FILES + (experience_name, "contract.json"))
    missing = sorted(required - set(source_inventory))
    if missing:
        raise ValueError(f"participant packet is missing resampling inputs {missing}")

    out = Path(out_dir).absolute()
    if out == packet or packet in out.parents or out in packet.parents:
        raise ValueError("outer resamples must not overlap the source packet")
    if out.exists():
        manifest = verify_paired_outer_resamples(out)
        expected = {
            "source_participant_digest_sha256": source_digest,
            "replicates": replicates,
            "outer_seed": seed,
            "method_seeds": seeds,
        }
        actual = {
            key: manifest.get(key)
            for key in expected
        }
        if actual != expected:
            raise ValueError("existing outer resamples belong to a different plan")
        return manifest

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.resampling-", dir=out.parent)
    )
    try:
        manifest: dict[str, object] = {
            "schema": OUTER_RESAMPLE_SCHEMA,
            "source_packet": str(packet),
            "source_participant_digest_sha256": source_digest,
            "source_participant_file_sha256": source_inventory,
            "replicates": replicates,
            "outer_seed": seed,
            "reference_lines": list(REFERENCE_LINES),
            "method_seeds": seeds,
            "method_seeds_fixed_across_outer_resamples": True,
            "oracle_diagnostics": {
                "included": False,
                "names": list(ORACLE_DIAGNOSTICS),
                "reason": "development-only oracle diagnostics are not reference lines",
            },
            "resamples": [],
        }
        for replicate in range(replicates):
            replicate_id = f"outer-{replicate:03d}"
            replicate_root = staging / replicate_id
            target = replicate_root / "participant"
            storage = _copy_participant_tree(participant, target)
            surveys = {}
            for survey_name in SURVEY_FILES:
                path = target / survey_name
                rng = np.random.default_rng(
                    _component_seed(seed, replicate, f"survey:{survey_name}")
                )
                resampled, evidence = rao_wu_resample(pd.read_csv(path), rng)
                _write_frame_replacing_link(resampled, path)
                surveys[survey_name] = evidence
            experience_path = target / experience_name
            experience_rng = np.random.default_rng(
                _component_seed(seed, replicate, f"experience:{experience_name}")
            )
            experience, experience_evidence = parametric_experience_resample(
                pd.read_csv(experience_path), experience_rng
            )
            _write_frame_replacing_link(experience, experience_path)
            inventory = _inventory(target)
            manifest["resamples"].append(
                {
                    "replicate_id": replicate_id,
                    "packet": str((out / replicate_id).absolute()),
                    "participant_digest_sha256": _inventory_digest(inventory),
                    "participant_file_sha256": inventory,
                    "storage": storage,
                    "survey_resampling": surveys,
                    "experience_resampling": experience_evidence,
                }
            )
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.replace(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_paired_outer_resamples(out)


def verify_paired_outer_resamples(out_dir: Path) -> dict:
    """Verify every participant byte against the materialized resample manifest."""
    requested = Path(out_dir)
    if requested.is_symlink():
        raise ValueError("outer resampling root may not be a symlink")
    out = requested.resolve()
    manifest_path = out / "manifest.json"
    if out.is_symlink() or manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("outer resampling manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("outer resampling manifest is invalid") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != OUTER_RESAMPLE_SCHEMA:
        raise ValueError("outer resampling manifest has the wrong schema")
    if manifest.get("reference_lines") != list(REFERENCE_LINES):
        raise ValueError("outer resampling manifest does not name reference lines A/B/C")
    _normalized_method_seeds(manifest.get("method_seeds"))
    if manifest.get("method_seeds_fixed_across_outer_resamples") is not True:
        raise ValueError("outer resampling method seeds are not recorded as fixed")
    if (manifest.get("oracle_diagnostics") or {}).get("included") is not False:
        raise ValueError("oracle diagnostics may not be mixed into outer reference runs")
    rows = manifest.get("resamples")
    if not isinstance(rows, list) or len(rows) != manifest.get("replicates"):
        raise ValueError("outer resampling manifest has an incomplete replicate list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("outer resampling manifest has an invalid replicate")
        replicate_id = row.get("replicate_id")
        if not isinstance(replicate_id, str) or replicate_id in seen:
            raise ValueError("outer resampling replicate IDs are invalid")
        seen.add(replicate_id)
        expected_path = out / replicate_id
        if Path(str(row.get("packet"))).absolute() != expected_path.absolute():
            raise ValueError(f"outer resampling packet path changed: {replicate_id}")
        inventory = _inventory(expected_path / "participant")
        if inventory != row.get("participant_file_sha256"):
            raise ValueError(f"outer resampling bytes changed: {replicate_id}")
        if _inventory_digest(inventory) != row.get("participant_digest_sha256"):
            raise ValueError(f"outer resampling digest changed: {replicate_id}")
    return manifest


def paired_reference_inputs(out_dir: Path) -> list[dict]:
    """Return digest-verified packet paths and fixed seeds for paired A/B/C runs."""
    manifest = verify_paired_outer_resamples(out_dir)
    seeds = dict(manifest["method_seeds"])
    return [
        {
            "replicate_id": row["replicate_id"],
            "packet": Path(row["packet"]),
            "participant_digest_sha256": row["participant_digest_sha256"],
            "method_seeds": dict(seeds),
        }
        for row in manifest["resamples"]
    ]
