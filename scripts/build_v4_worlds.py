"""Build the committed version-four world set: development, qualification, and graded.

Three families, one script, so the seeds and the world size are written once and never
drift between them.

- ``development``: twelve worlds, one per row of the committed twelve-run design, under
  the development source regime. These are the worlds a method may tune on.
- ``qualification``: six worlds under the hidden source regime, minted before any graded
  world. Thresholds are frozen on these and on nothing else.
- ``graded``: three independent worlds under the hidden source regime, minted after the
  thresholds are frozen and never read back into them.

Every packet is a deterministic function of its seed and the shared parameters. The
``--workers`` flag divides the continuation ensemble between processes and changes
nothing in the output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from meridia.mechanisms import DEVELOPMENT_DESIGN
from meridia.packet import GRADING_WORLD, PacketParams, build_packet

WORLD = GRADING_WORLD

DEVELOPMENT_SEEDS = tuple(1101 + i for i in range(len(DEVELOPMENT_DESIGN)))
QUALIFICATION_SEEDS = (2101, 2102, 2103, 2104, 2105, 2106)
GRADED_SEEDS = (3101, 3102, 3103)


def world_plan() -> dict[str, list[dict]]:
    """Every world the version-four surface commits to, with its parameters."""
    development = [{"name": f"dev-{cell:02d}", "seed": seed,
                    "params": PacketParams(**{**WORLD.__dict__, "design_cell": cell}),
                    "development": True}
                   for cell, seed in enumerate(DEVELOPMENT_SEEDS)]
    qualification = [{"name": f"qual-{i}", "seed": seed,
                      "params": PacketParams(**{**WORLD.__dict__, "regime": "hidden"}),
                      "development": False}
                     for i, seed in enumerate(QUALIFICATION_SEEDS)]
    graded = [{"name": f"graded-{i}", "seed": seed,
               "params": PacketParams(**{**WORLD.__dict__, "regime": "hidden"}),
               "development": False}
              for i, seed in enumerate(GRADED_SEEDS)]
    return {"development": development, "qualification": qualification, "graded": graded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--family", default="all",
                        choices=("all", "development", "qualification", "graded"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    plan = world_plan()
    families = list(plan) if args.family == "all" else [args.family]
    args.out.mkdir(parents=True, exist_ok=True)
    for family in families:
        for entry in plan[family]:
            directory = args.out / family / entry["name"]
            if directory.exists():
                print(f"{family}/{entry['name']}: already built", flush=True)
                continue
            start = time.time()
            manifest = build_packet(entry["seed"], directory, entry["params"],
                                    development=entry["development"],
                                    workers=args.workers)
            digest = json.loads((directory / "manifest.json").read_text())
            n_files = len(digest["participant"]) + len(digest["retained"])
            print(f"{family}/{entry['name']}: seed {entry['seed']} files {n_files} "
                  f"{time.time() - start:.0f}s", flush=True)
            assert manifest["development"] == entry["development"]


if __name__ == "__main__":
    main()
