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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.mechanisms import (DEVELOPMENT_BAND, HIDDEN_EXTRAPOLATION_AXES,
                                HIDDEN_IN_BAND_AXES, HIDDEN_LEVEL_PATTERNS,
                                N_HIDDEN_OUTSIDE_AXES, PUBLIC_ENVELOPE)
from meridia.packet import GRADING_WORLD, PacketParams, build_packet
from meridia.sealing import (DEFAULT_KEY_PATH, sealed_seed,
                             verify_sealed_world)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    parser.add_argument("--workers", type=int, default=1,
                        help="processes the continuation ensemble is divided between; "
                             "it changes nothing in the packet")
    args = parser.parse_args()

    manifest = json.loads(args.seal_manifest.read_text())
    registered = {int(world["index"]) for world in manifest["worlds"]}
    if args.index not in registered:
        raise ValueError(f"world index {args.index} is not registered")
    if not verify_sealed_world(args.index, args.seal_manifest, args.key):
        raise RuntimeError("registered world digest replay failed")

    master = args.key.read_bytes()
    seed = sealed_seed(master, args.index)
    params = PacketParams(**{**GRADING_WORLD.__dict__, "regime": "hidden"})
    packet_manifest = build_packet(seed, args.out, params, development=False,
                                   workers=args.workers)
    retained_world = json.loads((args.out / "retained" / "world.json").read_text())
    if int(retained_world["seed"]) != seed:
        raise RuntimeError("packet seed does not match registered sealed world")
    if retained_world.get("regime") != "hidden":
        raise RuntimeError("hidden packet was not built under the hidden source regime")
    design = retained_world["mechanisms"]["design"]
    if design["regime"] != "hidden" or design["cell"] != -1:
        raise RuntimeError("hidden packet carries a development mechanism design")
    if len(design["outside"]) != N_HIDDEN_OUTSIDE_AXES:
        raise RuntimeError("hidden packet does not move the declared number of intensities")
    if not set(design["outside"]) <= set(HIDDEN_EXTRAPOLATION_AXES):
        raise RuntimeError("hidden packet extrapolates an axis without a public trace")
    if tuple(int(v) for v in design["levels"]) not in HIDDEN_LEVEL_PATTERNS:
        raise RuntimeError("hidden packet takes a level pattern the development design spends")
    for axis in design["outside"]:
        value = float(design["intensity"][axis])
        low, high = DEVELOPMENT_BAND[axis]
        envelope_low, envelope_high = PUBLIC_ENVELOPE[axis]
        if low <= value <= high:
            raise RuntimeError(f"hidden intensity {axis} stayed inside the development band")
        if not envelope_low <= value <= envelope_high:
            raise RuntimeError(f"hidden intensity {axis} left the public envelope")
    for axis in HIDDEN_IN_BAND_AXES:
        value = float(design["intensity"][axis])
        low, high = DEVELOPMENT_BAND[axis]
        if axis in design["outside"] or not low <= value <= high:
            raise RuntimeError(
                f"hidden intensity {axis} left its required development range"
            )

    packet_manifest_path = args.out / "manifest.json"
    digest = hashlib.sha256(packet_manifest_path.read_bytes()).hexdigest()
    if packet_manifest.get("development") is not False:
        raise RuntimeError("hidden packet was marked as a development packet")
    print(f"SEALED_PACKET_BUILT manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
