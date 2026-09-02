"""Build a hidden reconstruction packet from a registered keyed Meridia world.

The derived seed is never accepted on the command line or written to stdout. It exists
only in the retained packet metadata, which does not enter the participant image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.packet import build_packet
from meridia.sealing import (DEFAULT_KEY_PATH, sealed_seed,
                             verify_sealed_world)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", type=Path, default=DEFAULT_KEY_PATH)
    args = parser.parse_args()

    manifest = json.loads(args.seal_manifest.read_text())
    registered = {int(world["index"]) for world in manifest["worlds"]}
    if args.index not in registered:
        raise ValueError(f"world index {args.index} is not registered")
    if not verify_sealed_world(args.index, args.seal_manifest, args.key):
        raise RuntimeError("registered world digest replay failed")

    master = args.key.read_bytes()
    seed = sealed_seed(master, args.index)
    packet_manifest = build_packet(seed, args.out, development=False)
    retained_world = json.loads((args.out / "retained" / "world.json").read_text())
    if int(retained_world["seed"]) != seed:
        raise RuntimeError("packet seed does not match registered sealed world")

    packet_manifest_path = args.out / "manifest.json"
    digest = hashlib.sha256(packet_manifest_path.read_bytes()).hexdigest()
    if packet_manifest.get("development") is not False:
        raise RuntimeError("hidden packet was marked as a development packet")
    print(f"SEALED_PACKET_BUILT manifest_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
