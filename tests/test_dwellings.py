"""Persistent identities and exact dwelling-stock accounting."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.dwellings import DwellingParams, build_dwellings
from meridia.dwellings import vacant_stock_target, validate_dwelling_conservation
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.identities import ENTITY_NAMESPACE, build_initial_identity_map
from meridia.identities import entity_namespace
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 64, 80
TOTAL = 50_000


def _start(seed: int = SEED):
    world = generate_elevation(seed, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 6)
    micro = build_microdata(
        people["population"], people["habitability"], people["settlements"], seed
    )
    identities = build_initial_identity_map(micro, seed)
    return micro, identities


def _stock_digest(stock: dict) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(stock["truth_world_id"], dtype=np.uint64).tobytes())
    digest.update(np.asarray(stock["snapshot_tick"], dtype=np.int64).tobytes())
    for name, values in stock["dwelling"].items():
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def test_initial_truth_identities_are_persistent_and_namespace_separated():
    micro, identities = _start()
    repeated = build_initial_identity_map(micro, SEED)
    person_id = identities["identity"]["truth_person_id"]
    household_id = identities["identity"]["truth_household_id"]

    assert identities["truth_world_id"] == repeated["truth_world_id"]
    assert np.array_equal(person_id, repeated["identity"]["truth_person_id"])
    assert np.array_equal(household_id, repeated["identity"]["truth_household_id"])
    assert (entity_namespace(person_id) == ENTITY_NAMESPACE["person"]).all()
    assert (entity_namespace(household_id) == ENTITY_NAMESPACE["household"]).all()
    assert len(np.unique(person_id)) == micro["n_persons"]
    assert len(np.unique(household_id)) == micro["n_households"]
    assert np.intersect1d(person_id, household_id).size == 0
    assert (
        build_initial_identity_map(micro, SEED + 1)["truth_world_id"]
        != (identities["truth_world_id"])
    )


def test_dwelling_stock_conserves_households_vacancies_and_persons_exactly():
    micro, identities = _start()
    stock = build_dwellings(micro, SEED, identities)
    table = stock["dwelling"]
    occupied = table["is_occupied"]

    assert stock["n_occupied"] == micro["n_households"]
    assert stock["n_vacant"] == vacant_stock_target(micro["n_households"], 0.08)
    assert stock["n_dwellings"] == stock["n_occupied"] + stock["n_vacant"]
    assert int(occupied.sum()) == micro["n_households"]
    assert int(table["resident_count"].sum()) == micro["n_persons"]

    household_sizes = np.bincount(
        micro["person"]["household"], minlength=micro["n_households"]
    )
    assert np.array_equal(table["resident_count"][occupied], household_sizes)
    assert np.array_equal(table["cell"][occupied], micro["household_cell"])
    assert np.array_equal(
        table["truth_household_id"][occupied],
        identities["identity"]["truth_household_id"],
    )
    assert (table["truth_household_id"][~occupied] == 0).all()
    assert (table["resident_count"][~occupied] == 0).all()
    validate_dwelling_conservation(stock, micro, identities)


def test_dwelling_schema_and_vacant_stock_are_materially_populated():
    micro, identities = _start()
    params = DwellingParams(vacancy_rate=0.11)
    stock = build_dwellings(micro, SEED, identities, params)
    table = stock["dwelling"]
    vacant = ~table["is_occupied"]

    assert stock["n_vacant"] == vacant_stock_target(micro["n_households"], 0.11)
    assert vacant.any()
    assert np.isin(table["cell"][vacant], micro["household_cell"]).all()
    assert set(np.unique(table["dwelling_type"])) == {0, 1, 2, 3}
    assert set(np.unique(table["tenure"])) == {0, 1, 2, 3, 4}
    assert (table["bedrooms"] >= 1).all()
    assert (table["floor_area_m2"] > 0).all()
    assert (table["assessed_value"] > 0).all()
    assert (table["monthly_rent"][vacant] == 0).all()


def test_dwelling_stock_is_byte_deterministic():
    digests = []
    for _ in range(2):
        micro, identities = _start()
        stock = build_dwellings(micro, SEED, identities)
        digests.append(_stock_digest(stock))
    assert digests[0] == digests[1]


def test_conservation_validator_rejects_a_changed_resident_count():
    micro, identities = _start()
    stock = build_dwellings(micro, SEED, identities)
    changed = {**stock, "dwelling": {**stock["dwelling"]}}
    changed["dwelling"]["resident_count"] = stock["dwelling"]["resident_count"].copy()
    changed["dwelling"]["resident_count"][0] += 1

    with pytest.raises(ValueError, match="resident counts"):
        validate_dwelling_conservation(changed, micro, identities)


def test_dwelling_builder_rejects_a_seed_from_another_truth_world():
    micro, identities = _start()
    with pytest.raises(ValueError, match="seed does not match"):
        build_dwellings(micro, SEED + 1, identities)
