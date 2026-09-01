"""Sealing protocol: evaluation worlds nobody has seen, provably unchanged.

A sealed world's seed is derived from a master secret that never enters the repository:
seed_i = SHA-256(master_secret || index). The committed manifest records, per index,
only the digests of the generated layers. Nothing else is retained: generation runs
headless, arrays are hashed and discarded, and no render, summary, or statistic beyond
the digests exists anywhere. Anyone holding the manifest can later confirm that a world
used for grading is byte-identical to the one sealed on registration day; nobody without
the master secret can regenerate it, and nobody with it has looked.

The no-inspection rule is enforced by shape: `generate_and_digest` returns digests only.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .character import draw_world_character
from .microdata import build_microdata
from .population import build_population, draw_national_total
from .terrain import generate_elevation

DEFAULT_KEY_PATH = Path.home() / ".meridia" / "sealed_master.key"
GRID = (288, 384)


def create_master_key(path: Path = DEFAULT_KEY_PATH) -> Path:
    """Create the master secret once; refuses to overwrite an existing key."""
    if path.exists():
        raise FileExistsError(f"master key already exists at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(secrets.token_bytes(32))
    path.chmod(0o600)
    return path


def sealed_seed(master: bytes, index: int) -> int:
    digest = hashlib.sha256(master + index.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def generate_and_digest(seed: int) -> dict:
    """Generate a full world and return layer digests only; arrays are discarded."""
    height, width = GRID
    character = draw_world_character(seed)
    world = generate_elevation(seed, height, width)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    total = draw_national_total(seed, int(world["land"].sum()))
    people = build_population(world, accumulation, total, 24,
                              params=character["population"], seed=seed)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], seed, params=character["microdata"])
    digests = {
        "elevation": _digest(world["elevation"]),
        "flow_direction": _digest(direction),
        "population_grid": _digest(people["population"]),
        "person_table": hashlib.sha256(
            b"".join(np.ascontiguousarray(v).tobytes()
                     for v in micro["person"].values())).hexdigest(),
        "household_cells": _digest(micro["household_cell"]),
    }
    return digests


def seal_worlds(n_worlds: int, manifest_path: Path,
                key_path: Path = DEFAULT_KEY_PATH) -> dict:
    """Register and seal n worlds; writes the public manifest, returns it."""
    master = key_path.read_bytes()
    worlds = []
    for index in range(n_worlds):
        seed = sealed_seed(master, index)
        digests = generate_and_digest(seed)
        commitment = hashlib.sha256(
            master + index.to_bytes(8, "big") + b"commit").hexdigest()
        worlds.append({"index": index, "commitment": commitment, "digests": digests})
    manifest = {"schema": "meridia.sealed.v1", "grid": list(GRID),
                "n_worlds": n_worlds, "worlds": worlds}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify_sealed_world(index: int, manifest_path: Path,
                        key_path: Path = DEFAULT_KEY_PATH) -> bool:
    """Regenerate world `index` from the secret and check every digest matches."""
    master = key_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    entry = next(w for w in manifest["worlds"] if w["index"] == index)
    digests = generate_and_digest(sealed_seed(master, index))
    return digests == entry["digests"]
