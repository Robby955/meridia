"""Build a hidden reconstruction packet from a registered keyed Meridia world.

The derived seed is never accepted on the command line or written to stdout. It exists
only in the retained packet metadata, which does not enter the participant image.

The build is refused unless the world carries the hidden mechanism design: no development
design cell, two identifiable intensities outside the development band but inside the
public plausibility envelope, and every axis without a participant-file trace held inside
the development band.

The world is ``packet.GRADING_WORLD``, the same size the development and qualification
worlds are built at, so a bar frozen on those is read on the same object here. ``--workers``
divides the continuation ensemble between processes and changes nothing in the packet.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.graded_readiness import validate_graded_readiness
from meridia.packet import (GRADING_WORLD, PacketParams, build_packet,
                            validate_packet_directory)
from meridia.sealing import (DEFAULT_KEY_PATH, V4PublicationAuthorization,
                             verify_and_derive_v4_seed)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--reserve-calibration-audit", type=Path, required=True)
    parser.add_argument("--workers", type=_positive_int, default=1,
                        help="processes the continuation ensemble is divided between; "
                             "it changes nothing in the packet")
    args = parser.parse_args(argv)

    # This call must remain ahead of every read of the seal manifest or key. A failed
    # freeze therefore cannot even derive, log, or partially materialize a graded seed.
    readiness = validate_graded_readiness(
        args.bars,
        args.reserve_calibration_audit,
        expected_rate_per_person_year=GRADING_WORLD.reserve_rate_per_person_year,
    )

    params = PacketParams(**{**GRADING_WORLD.__dict__, "regime": "hidden"})
    authorization = verify_and_derive_v4_seed(
        args.index,
        args.seal_manifest,
        args.key,
        params=params,
        readiness=readiness,
    )
    seed = authorization.seed
    publication_authority = V4PublicationAuthorization(
        before=authorization,
        index=args.index,
        seal_manifest_path=args.seal_manifest,
        key_path=args.key,
        bars_path=args.bars,
        reserve_calibration_path=args.reserve_calibration_audit,
    )
    if args.out.exists():
        packet_manifest = validate_packet_directory(
            args.out,
            expected_packet_class="graded",
            expected_params=params,
            expected_seed=seed,
        )
        publication_authority.confirm(seed=seed, params=params)
    else:
        build_packet(
            seed,
            args.out,
            params,
            development=False,
            packet_class="graded",
            workers=args.workers,
            graded_authorization=publication_authority,
        )
        packet_manifest = validate_packet_directory(
            args.out,
            expected_packet_class="graded",
            expected_params=params,
            expected_seed=seed,
        )

    packet_manifest_path = args.out / "manifest.json"
    digest = hashlib.sha256(packet_manifest_path.read_bytes()).hexdigest()
    if packet_manifest.get("development") is not False:
        raise RuntimeError("hidden packet was marked as a development packet")
    if packet_manifest.get("packet_class") != "graded":
        raise RuntimeError("hidden packet was not marked as a graded packet")
    print(f"SEALED_PACKET_BUILT manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
