"""Microdata: exact population-grid consistency, household coherence, and determinism."""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 250_000


def _microdata():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 8)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], SEED)
    return people, micro


def test_persons_match_the_population_grid_exactly():
    people, micro = _microdata()
    assert micro["n_persons"] == TOTAL
    counts = np.bincount(micro["person"]["cell"], minlength=H * W)
    assert np.array_equal(counts, people["population"].flatten())


def test_households_partition_persons():
    _, micro = _microdata()
    hh = micro["person"]["household"]
    assert hh.min() == 0 and hh.max() == micro["n_households"] - 1
    sizes = np.bincount(hh)
    assert sizes.min() >= 1 and sizes.sum() == micro["n_persons"]
    # every household lives in one cell, the one its members share
    cell_by_person = micro["person"]["cell"]
    first_cell = np.zeros(micro["n_households"], dtype=np.int64)
    first_cell[hh[::-1]] = cell_by_person[::-1]
    assert np.array_equal(first_cell[hh], cell_by_person)
    assert np.array_equal(first_cell, micro["household_cell"])


def test_household_heads_are_adults():
    _, micro = _microdata()
    role = micro["person"]["role"]
    age = micro["person"]["age"]
    assert (age[role == 0] >= 20).all()
    assert (age[role == 2] < 18).all()
    heads_per_household = np.bincount(micro["person"]["household"][role == 0])
    assert (heads_per_household == 1).all()


def test_children_have_no_income_and_low_education():
    _, micro = _microdata()
    young = micro["person"]["age"] < 16
    assert float(micro["person"]["income"][young].sum()) == 0.0
    assert int(micro["person"]["education"][young].max(initial=0)) == 0


def test_urban_gradient_in_education_and_income():
    _, micro = _microdata()
    urb = micro["urbanity"].flatten()[micro["person"]["cell"]]
    adult = micro["person"]["age"] >= 25
    high_urb = adult & (urb > np.quantile(urb[adult], 0.8))
    low_urb = adult & (urb < np.quantile(urb[adult], 0.2))
    assert micro["person"]["education"][high_urb].mean() > micro["person"]["education"][low_urb].mean()
    assert micro["person"]["income"][high_urb].mean() > micro["person"]["income"][low_urb].mean()


def test_microdata_deterministic():
    digests = []
    for _ in range(2):
        _, micro = _microdata()
        blob = b"".join(np.ascontiguousarray(v).tobytes()
                        for v in micro["person"].values())
        digests.append(hashlib.sha256(blob).hexdigest())
    assert digests[0] == digests[1]
