"""Hospital capacity, staffing, accessibility, and occupied-bed accounting."""

import hashlib
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.businesses import INDUSTRIES, build_businesses
from meridia.character import draw_world_character
from meridia.hospitals import build_hospitals, hospital_params_from_character
from meridia.hospitals import validate_hospital_conservation
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.identities import ENTITY_NAMESPACE, build_initial_identity_map
from meridia.identities import entity_namespace
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 60, 72
TOTAL = 40_000
HEALTH_INDUSTRY = INDUSTRIES.index("health")


@lru_cache(maxsize=2)
def _start(seed: int = SEED):
    character = draw_world_character(seed)
    world = generate_elevation(seed, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(
        world,
        accumulation,
        TOTAL,
        6,
        params=character["population"],
        seed=seed,
    )
    micro = build_microdata(
        people["population"],
        people["habitability"],
        people["settlements"],
        seed,
        params=character["microdata"],
    )
    identities = build_initial_identity_map(micro, seed)
    businesses = build_businesses(micro, seed, identities)
    return micro, identities, businesses


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in state["hospital_params"].items():
        digest.update(name.encode("utf-8"))
        digest.update(repr(value).encode("ascii"))
    for table_name in ("hospital", "staffing", "encounter"):
        digest.update(table_name.encode("utf-8"))
        for name, values in state[table_name].items():
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def test_hospitals_use_accessible_health_establishments_and_conserve_catchments():
    micro, identities, businesses = _start()
    state = build_hospitals(micro, SEED, identities, businesses)
    hospital = state["hospital"]
    establishment = businesses["establishment"]
    establishment_position = np.searchsorted(
        establishment["truth_establishment_id"],
        hospital["truth_establishment_id"],
    )
    population = np.bincount(micro["person"]["cell"], minlength=micro["urbanity"].size)

    assert (
        entity_namespace(hospital["truth_hospital_id"]) == ENTITY_NAMESPACE["hospital"]
    ).all()
    assert (establishment["industry"][establishment_position] == HEALTH_INDUSTRY).all()
    assert np.array_equal(
        hospital["cell"], establishment["cell"][establishment_position]
    )
    assert (population[hospital["cell"]] > 0).all()
    assert int(hospital["catchment_population"].sum()) == micro["n_persons"]
    assert int(hospital["bed_count"].sum()) == state["total_beds"]


def test_staffing_reconciles_exactly_to_health_sector_jobs():
    micro, identities, businesses = _start()
    state = build_hospitals(micro, SEED, identities, businesses)
    hospital = state["hospital"]
    staffing = state["staffing"]
    establishment = businesses["establishment"]
    job = businesses["job"]

    job_position = np.searchsorted(job["truth_job_id"], staffing["truth_job_id"])
    job_establishment_position = np.searchsorted(
        establishment["truth_establishment_id"],
        job["truth_establishment_id"][job_position],
    )
    staffing_hospital_position = np.searchsorted(
        hospital["truth_hospital_id"], staffing["truth_hospital_id"]
    )
    hospital_establishment_position = np.searchsorted(
        establishment["truth_establishment_id"],
        hospital["truth_establishment_id"],
    )
    expected_count = np.bincount(
        staffing_hospital_position, minlength=state["n_hospitals"]
    ).astype(np.int32)

    assert len(np.unique(staffing["truth_job_id"])) == state["n_staffing"]
    assert (
        establishment["industry"][job_establishment_position] == HEALTH_INDUSTRY
    ).all()
    assert np.array_equal(
        job_establishment_position,
        hospital_establishment_position[staffing_hospital_position],
    )
    assert np.array_equal(hospital["staffed_position_count"], expected_count)


def test_open_encounters_reconcile_to_unique_people_and_beds():
    micro, identities, businesses = _start()
    state = build_hospitals(micro, SEED, identities, businesses)
    hospital = state["hospital"]
    encounter = state["encounter"]
    is_open = encounter["is_open"]
    hospital_position = np.searchsorted(
        hospital["truth_hospital_id"], encounter["truth_hospital_id"]
    )

    assert np.isin(
        encounter["truth_person_id"], identities["identity"]["truth_person_id"]
    ).all()
    assert np.isin(encounter["truth_hospital_id"], hospital["truth_hospital_id"]).all()
    assert (
        entity_namespace(encounter["truth_encounter_id"])
        == ENTITY_NAMESPACE["encounter"]
    ).all()
    assert int(is_open.sum()) == state["n_open_encounters"]
    assert int(hospital["occupied_bed_count"].sum()) == state["n_open_encounters"]
    assert len(np.unique(encounter["truth_person_id"][is_open])) == int(is_open.sum())
    occupied_pairs = np.column_stack(
        (hospital_position[is_open], encounter["bed_number"][is_open])
    )
    assert len(np.unique(occupied_pairs, axis=0)) == int(is_open.sum())
    validate_hospital_conservation(state, micro, identities, businesses, SEED)


def test_hospital_bed_character_dial_is_load_bearing():
    micro, identities, businesses = _start()
    base = hospital_params_from_character(draw_world_character(SEED)["business"])
    low = build_hospitals(
        micro, SEED, identities, businesses, replace(base, beds_per_1000=1.8)
    )
    high = build_hospitals(
        micro, SEED, identities, businesses, replace(base, beds_per_1000=6.5)
    )

    assert low["total_beds"] == round(micro["n_persons"] * 1.8 / 1000.0)
    assert high["total_beds"] == round(micro["n_persons"] * 6.5 / 1000.0)
    assert high["total_beds"] > low["total_beds"]
    assert high["n_hospitals"] > low["n_hospitals"]


def test_hospital_state_is_byte_deterministic():
    micro, identities, businesses = _start()
    first = build_hospitals(micro, SEED, identities, businesses)
    second = build_hospitals(micro, SEED, identities, businesses)
    assert _state_digest(first) == _state_digest(second)


def test_occupied_bed_tamper_is_rejected():
    micro, identities, businesses = _start()
    state = build_hospitals(micro, SEED, identities, businesses)
    changed = {**state, "hospital": {**state["hospital"]}}
    changed["hospital"]["occupied_bed_count"] = state["hospital"][
        "occupied_bed_count"
    ].copy()
    changed["hospital"]["occupied_bed_count"][0] += 1

    with pytest.raises(ValueError, match="occupied beds"):
        validate_hospital_conservation(changed, micro, identities, businesses, SEED)


def test_staffing_tamper_is_rejected():
    micro, identities, businesses = _start()
    state = build_hospitals(micro, SEED, identities, businesses)
    establishment = businesses["establishment"]
    job = businesses["job"]
    job_establishment_position = np.searchsorted(
        establishment["truth_establishment_id"], job["truth_establishment_id"]
    )
    non_health_job = np.flatnonzero(
        establishment["industry"][job_establishment_position] != HEALTH_INDUSTRY
    )[0]
    changed = {**state, "staffing": {**state["staffing"]}}
    changed["staffing"]["truth_job_id"] = state["staffing"]["truth_job_id"].copy()
    changed["staffing"]["truth_job_id"][0] = job["truth_job_id"][non_health_job]

    with pytest.raises(ValueError, match="staff job"):
        validate_hospital_conservation(changed, micro, identities, businesses, SEED)


def test_hospital_builder_rejects_a_seed_from_another_truth_world():
    micro, identities, businesses = _start()
    with pytest.raises(ValueError, match="seed does not match"):
        build_hospitals(micro, SEED + 1, identities, businesses)
