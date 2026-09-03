"""Run the registered one-look confirmation on one keyed hidden packet.

This script never derives new bars. It runs the two frozen strong witnesses and every
control in the battery against the exact three-file task surface, writes a seed-free
receipt, and fails if a strong witness misses or a control passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.methods import bayesian, controls, design_based
from meridia.sealing import DEFAULT_KEY_PATH, sealed_seed, verify_sealed_world
from meridia.verify import verify_release_projection_allocation


# Digests of the bar set this script is allowed to confirm against: the version-three
# set in `bars/national-v7/`, which `scripts/freeze_bars.py` reports as frozen on the
# nine qualification worlds (`RESULT: bars frozen; every control fails a named gate`).
# Any other input is refused. The refusal is the point: a confirmation spends the sealed
# world's only look, and a bar set that a control clears is not worth spending it on.
# See `bars/national-v7/PROVENANCE.md`.
EXPECTED_SHA256 = {
    "bars": "f53983161245bb1eca4cb36a6a35325aba3c58262d9d46ef2cd97047d4f6aae9",
    "calibration_a": "61cf3367661bee14155934be70e1afddf0ae4cf04c57494cd92537f6e2c0b107",
    "calibration_b": "5fb058218393b113a2d2d16d7939aa0e69ca2a8f1041dac8954a9ab813dc6790",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _verify_packet_manifest(packet: Path) -> str:
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("development") is not False:
        raise RuntimeError("confirmation packet is not hidden")
    for side in ("participant", "retained"):
        for relative, expected in manifest[side].items():
            path = packet / side / relative
            if not path.is_file() or _sha256(path) != expected["sha256"]:
                raise RuntimeError(f"packet manifest mismatch: {side}/{relative}")
    return _sha256(manifest_path)


def _three_files(out: Path) -> None:
    for legacy in ("detailed.csv", "totals.csv"):
        path = out / legacy
        if path.exists():
            path.unlink()


def _blind_packet(packet: Path, out: Path) -> Path:
    blind = out / "blind-packet"
    blind.mkdir()
    shutil.copytree(packet / "participant", blind / "participant", copy_function=os.link)
    return blind


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--calibration-a", type=Path, required=True)
    parser.add_argument("--calibration-b", type=Path, required=True)
    parser.add_argument("--seal-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"confirmation directory already exists: {args.out}")
    observed_hashes = {
        "bars": _sha256(args.bars),
        "calibration_a": _sha256(args.calibration_a),
        "calibration_b": _sha256(args.calibration_b),
    }
    if observed_hashes != EXPECTED_SHA256:
        raise RuntimeError(f"frozen input digest mismatch: {observed_hashes}")
    if not verify_sealed_world(args.index, args.seal_manifest, args.key):
        raise RuntimeError("registered world digest replay failed")
    expected_seed = sealed_seed(args.key.read_bytes(), args.index)
    retained_world = json.loads((args.packet / "retained" / "world.json").read_text())
    if int(retained_world["seed"]) != expected_seed:
        raise RuntimeError("packet is not the registered keyed world")
    packet_manifest_sha256 = _verify_packet_manifest(args.packet)

    args.out.mkdir(parents=True)
    blind = _blind_packet(args.packet, args.out)
    bars = json.loads(args.bars.read_text())
    reports = {"strong": {}, "controls": {}}

    strong_a = args.out / "strong_A"
    design_based.run(
        blind,
        strong_a,
        design_based.MethodParams(
            bootstrap_replicates=100,
            calibration_path=str(args.calibration_a),
        ),
    )
    _three_files(strong_a)
    reports["strong"]["A"] = verify_release_projection_allocation(
        args.packet, strong_a, bars
    )

    strong_b = args.out / "strong_B"
    bayesian.run(
        blind,
        strong_b,
        bayesian.MethodParams(
            sweeps=400,
            burn_in=100,
            calibration_path=str(args.calibration_b),
        ),
    )
    _three_files(strong_b)
    reports["strong"]["B"] = verify_release_projection_allocation(
        args.packet, strong_b, bars
    )

    for name in controls.CONTROLS:
        control_out = args.out / f"control_{name}"
        controls.run(name, blind, control_out, calibration_path=str(args.calibration_a))
        _three_files(control_out)
        reports["controls"][name] = verify_release_projection_allocation(
            args.packet, control_out, bars
        )

    strong_pass = all(report["pass"] for report in reports["strong"].values())
    controls_fail = all(not report["pass"] for report in reports["controls"].values())
    receipt = {
        "schema": "meridia.reconstruction.confirmation.v2",
        "seal_manifest_sha256": _sha256(args.seal_manifest),
        "sealed_world_index": args.index,
        "packet_manifest_sha256": packet_manifest_sha256,
        "frozen_input_sha256": observed_hashes,
        "task_surface": ["release.csv", "projection.csv", "allocation.csv"],
        "strong_methods_pass": strong_pass,
        "all_controls_fail": controls_fail,
        "reports": reports,
    }
    (args.out / "confirmation.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True, default=_json_default) + "\n"
    )

    for name, report in reports["strong"].items():
        print(f"strong {name}: {'PASS' if report['pass'] else 'FAIL'}")
    for name, report in reports["controls"].items():
        status = "FAILS_AS_REQUIRED" if not report["pass"] else "UNEXPECTED_PASS"
        families = sorted({reason.split(":", 1)[0] for reason in report["reasons"]})
        print(f"control {name}: {status} gates={','.join(families)}")
    print("CONFIRMATION_PASS" if strong_pass and controls_fail else "CONFIRMATION_STOP")
    return 0 if strong_pass and controls_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
