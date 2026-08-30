"""Determinism and conservation tests for the first two Meridia layers."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions, outflow_to_outlets
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 96, 128


def _world():
    terrain = generate_elevation(SEED, H, W)
    outlets = ~terrain["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(terrain["elevation"], terrain["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    return terrain, outlets, filled, direction, accumulation


def test_determinism_byte_identical():
    a = generate_elevation(SEED, H, W)["elevation"]
    b = generate_elevation(SEED, H, W)["elevation"]
    assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()


def test_different_seeds_differ():
    a = generate_elevation(SEED, H, W)["elevation"]
    b = generate_elevation(SEED + 1, H, W)["elevation"]
    assert not np.array_equal(a, b)


def test_fill_never_lowers_and_land_drains():
    terrain, outlets, filled, direction, _ = _world()
    assert np.all(filled >= terrain["elevation"] - 1e-12)
    assert np.all(direction[~outlets] >= 0)


def test_no_cycles_all_paths_reach_outlet():
    _, outlets, _, direction, _ = _world()
    from meridia.hydrology import NEIGHBORS
    height, width = direction.shape
    for r0 in range(0, height, 7):
        for c0 in range(0, width, 11):
            if outlets[r0, c0]:
                continue
            r, c, steps = r0, c0, 0
            while direction[r, c] >= 0 and steps <= height * width:
                k = direction[r, c]
                r, c = r + NEIGHBORS[k][0], c + NEIGHBORS[k][1]
                steps += 1
                if outlets[r, c]:
                    break
            assert outlets[r, c], f"path from ({r0},{c0}) did not reach an outlet"


def test_conservation_exact():
    _, outlets, _, direction, accumulation = _world()
    interior_land = int((~outlets).sum())
    assert outflow_to_outlets(direction, accumulation, outlets) == interior_land


def test_rivers_exist():
    _, outlets, _, _, accumulation = _world()
    assert int(accumulation.max()) > 50  # at least one river draining a real basin
