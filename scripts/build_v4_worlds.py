"""Build the committed version-four world set: development, qualification, and graded.

Three families, one script, so the world size is written once and never drifts between
them.

- ``development``: twelve worlds, one per row of the committed twelve-run design, under
  the development source regime. These are the worlds a method may tune on, and their
  seeds are committed here.
- ``qualification``: six worlds under the hidden source regime, minted before any graded
  world. Thresholds are frozen on these and on nothing else.
- ``identifiability``: the twelve development worlds and the eighteen hidden worlds the
  identifiability measurement is read against, built with a short continuation ensemble.
  Their seeds are committed, nothing is graded on them, and no bar is frozen on them.
- ``graded``: independent worlds under the hidden source regime, minted after the
  thresholds are frozen and never read back into them. Their seeds are read from a sealed
  file outside the repository and are never printed, written into a packet, or committed.
  A world's whole configuration follows from its seed, so a graded seed in the tree is
  the graded configuration in the tree.

Every packet is a deterministic function of its seed and the shared parameters. Two
flags divide the work and neither changes the output. ``--workers`` divides one world's
continuation ensemble between processes; ``--world-workers`` builds whole worlds at once.
The second is the one that scales: the ensemble step is what a packet costs, it holds no
lock and shares nothing with the baseline step, and a world's ledger is a single
process's work whatever the ensemble is doing. Use one or the other, since together they
oversubscribe the machine.

``--cache`` points at a directory of continuation ensembles keyed on the digest of the
baseline ledger that produced them. A rebuild that changes only what a verifier or a bar
reads takes the futures off the shelf instead of paying for them again.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from meridia.mechanisms import DEVELOPMENT_DESIGN
from meridia.packet import GRADING_WORLD, PacketParams, build_packet

WORLD = GRADING_WORLD

DEVELOPMENT_SEEDS = tuple(1101 + i for i in range(len(DEVELOPMENT_DESIGN)))
QUALIFICATION_SEEDS = (2101, 2102, 2103, 2104, 2105, 2106)

# The eighteen hidden-regime worlds the identifiability measurement is read against, and
# the size of the continuation ensemble that measurement builds them at. Nothing is
# graded on these worlds and no bar is frozen on them, so their seeds are committed here:
# the six pooled correlations in the decisions record are a claim about a definite set of
# worlds, and a set that lives only in one run is a claim a reader cannot check. The
# values sit clear of the development band at 1101, the qualification band at 2101 and
# the three burned values 3101 to 3103. A graded seed is a sixty-three bit digest of the
# master secret, so no small committed value can name a graded world.
IDENTIFIABILITY_HIDDEN_SEEDS = tuple(4101 + i for i in range(18))
# The ensemble enters none of the six statistics and is the whole cost of a packet, so the
# measurement builds the committed world with a short one.
IDENTIFIABILITY_MEMBERS = 8

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
    if family == "identifiability":
        members = {"ensemble_members": IDENTIFIABILITY_MEMBERS}
        development = [
            {"name": f"ident-dev-{cell:02d}", "seed": seed,
             "params": PacketParams(**{**WORLD.__dict__, "design_cell": cell, **members}),
             "development": True, "public_seed": True}
            for cell, seed in enumerate(DEVELOPMENT_SEEDS)]
        hidden = [
            {"name": f"ident-hidden-{i:02d}", "seed": seed,
             "params": PacketParams(**{**WORLD.__dict__, "regime": "hidden", **members}),
             "development": False, "public_seed": False}
            for i, seed in enumerate(IDENTIFIABILITY_HIDDEN_SEEDS)]
        return development + hidden
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
# Built on request and never part of ``--family all``: the measurement set is not a world
# a method is handed or graded on, and rebuilding it costs the same as rebuilding the
# committed set.
BUILDABLE = FAMILIES + ("identifiability",)


def progress_line(family: str, entry: dict, n_files: int, seconds: float) -> str:
    """One line of build log. A seed reaches it only for the worlds a method may tune on."""
    named = f"seed {entry['seed']} " if entry["public_seed"] else ""
    return f"{family}/{entry['name']}: {named}files {n_files} {seconds:.0f}s"


def build_one(job: dict) -> str:
    """Build one world and return its log line. Runs in this process or a worker."""
    family, entry = job["family"], job["entry"]
    directory = Path(job["out"]) / family / entry["name"]
    if directory.exists():
        return f"{family}/{entry['name']}: already built"
    start = time.time()
    manifest = build_packet(entry["seed"], directory, entry["params"],
                            development=entry["development"],
                            workers=job["workers"],
                            cache_dir=Path(job["cache"]) if job["cache"] else None)
    if manifest["development"] != entry["development"]:
        raise RuntimeError(f"{family}/{entry['name']} was written on the wrong side")
    digest = json.loads((directory / "manifest.json").read_text())
    n_files = len(digest["participant"]) + len(digest["retained"])
    return progress_line(family, entry, n_files, time.time() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--family", default="all", choices=("all",) + BUILDABLE)
    parser.add_argument("--workers", type=int, default=1,
                        help="processes inside one world's continuation ensemble")
    parser.add_argument("--world-workers", type=int, default=1,
                        help="worlds built at once, each on its own process")
    parser.add_argument("--cache", type=Path, default=None,
                        help="directory of continuation ensembles, keyed on the "
                             "baseline ledger digest")
    args = parser.parse_args()
    families = list(FAMILIES) if args.family == "all" else [args.family]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [{"family": family, "entry": entry, "out": str(args.out),
             "workers": args.workers,
             "cache": str(args.cache) if args.cache else None}
            for family in families for entry in family_plan(family)]
    if args.world_workers > 1:
        with ProcessPoolExecutor(max_workers=args.world_workers) as pool:
            for line in pool.map(build_one, jobs):
                print(line, flush=True)
        return
    for job in jobs:
        print(build_one(job), flush=True)


if __name__ == "__main__":
    main()
