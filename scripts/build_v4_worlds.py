"""Build the committed version-four world set: development, qualification, and graded.

Three families, one script, so the world size is written once and never drifts between
them.

- ``development``: twelve worlds, one per row of the committed twelve-run design, under
  the development source regime. These are the worlds a method may tune on, and their
  seeds are committed here.
- ``qualification``: six worlds under the hidden source regime, minted before any graded
  world. Thresholds are frozen on these and on nothing else.
- ``graded``: independent worlds under the hidden source regime, minted after the
  thresholds are frozen and never read back into them. Their seeds are read from a sealed
  file outside the repository and are never printed, written into a packet, or committed.
  A world's whole configuration follows from its seed, so a graded seed in the tree is
  the graded configuration in the tree.

Every packet is a deterministic function of its seed and the shared parameters. The
``--workers`` flag divides the continuation ensemble between processes and changes
nothing in the output.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from meridia.mechanisms import DEVELOPMENT_DESIGN
from meridia.packet import GRADING_WORLD, PacketParams, build_packet

WORLD = GRADING_WORLD

DEVELOPMENT_SEEDS = tuple(1101 + i for i in range(len(DEVELOPMENT_DESIGN)))
QUALIFICATION_SEEDS = (2101, 2102, 2103, 2104, 2105, 2106)

# The sealed file holding the graded seeds, one JSON list of integers. It lives outside
# the repository so that a clone carries the surface without carrying the worlds it is
# graded on. Override with MERIDIA_GRADED_SEED_FILE.
GRADED_SEED_FILE = Path(
    os.environ.get("MERIDIA_GRADED_SEED_FILE",
                   Path.home() / ".config" / "meridia" / "v4_graded_seeds.json")
).expanduser()


def graded_seeds(path: Path | None = None) -> tuple[int, ...]:
    """The graded seeds, read from the sealed file. Never logged and never committed."""
    source = Path(path) if path is not None else GRADED_SEED_FILE
    if not source.is_file():
        raise FileNotFoundError(
            f"the graded seed file is missing at {source}. Write a JSON list of "
            "integers there, or point MERIDIA_GRADED_SEED_FILE at it. It stays outside "
            "the repository."
        )
    values = json.loads(source.read_text())
    if not isinstance(values, list) or not values or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise ValueError(f"{source} must hold a non-empty JSON list of integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{source} repeats a seed")
    return tuple(int(v) for v in values)


def family_plan(family: str, graded_seed_file: Path | None = None) -> list[dict]:
    """The worlds of one family, with the parameters each is built under."""
    if family == "development":
        return [{"name": f"dev-{cell:02d}", "seed": seed,
                 "params": PacketParams(**{**WORLD.__dict__, "design_cell": cell}),
                 "development": True, "public_seed": True}
                for cell, seed in enumerate(DEVELOPMENT_SEEDS)]
    if family == "qualification":
        return [{"name": f"qual-{i}", "seed": seed,
                 "params": PacketParams(**{**WORLD.__dict__, "regime": "hidden"}),
                 "development": False, "public_seed": False}
                for i, seed in enumerate(QUALIFICATION_SEEDS)]
    if family == "graded":
        return [{"name": f"graded-{i}", "seed": seed,
                 "params": PacketParams(**{**WORLD.__dict__, "regime": "hidden"}),
                 "development": False, "public_seed": False}
                for i, seed in enumerate(graded_seeds(graded_seed_file))]
    raise ValueError(f"unknown world family {family!r}")


FAMILIES = ("development", "qualification", "graded")


def progress_line(family: str, entry: dict, n_files: int, seconds: float) -> str:
    """One line of build log. A seed reaches it only for the worlds a method may tune on."""
    named = f"seed {entry['seed']} " if entry["public_seed"] else ""
    return f"{family}/{entry['name']}: {named}files {n_files} {seconds:.0f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--family", default="all", choices=("all",) + FAMILIES)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    families = list(FAMILIES) if args.family == "all" else [args.family]
    args.out.mkdir(parents=True, exist_ok=True)
    for family in families:
        for entry in family_plan(family):
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
            print(progress_line(family, entry, n_files, time.time() - start), flush=True)
            assert manifest["development"] == entry["development"]


if __name__ == "__main__":
    main()
