"""Persistent monthly histories, reporting lag, and exact replay."""

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.businesses import build_businesses
from meridia.character import draw_world_character
from meridia.dwellings import build_dwellings
from meridia.events import EVENT_TYPES, build_event_history, events_visible_at
from meridia.events import replay_event_history, validate_event_history
from meridia.hospitals import HospitalParams, build_hospitals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.identities import ENTITY_NAMESPACE, build_initial_identity_map
from meridia.identities import entity_namespace
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 60, 72
TOTAL = 40_000


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
    dwellings = build_dwellings(micro, seed, identities)
    businesses = build_businesses(micro, seed, identities)
    hospitals = build_hospitals(micro, seed, identities, businesses)
    return micro, identities, dwellings, businesses, hospitals


@lru_cache(maxsize=8)
def _history(months: int = 18):
    micro, identities, dwellings, businesses, hospitals = _start()
    return build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=months,
    )


def _history_digest(history: dict) -> str:
    digest = hashlib.sha256()
    for table_name in sorted(history["initial_state"]):
        for name, values in history["initial_state"][table_name].items():
            digest.update(table_name.encode())
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    for name, values in history["event"].items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(values).tobytes())
    for table_name in sorted(history["terminal_state"]):
        for name, values in history["terminal_state"][table_name].items():
            digest.update(table_name.encode())
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def test_history_contains_every_required_change_family_with_persistent_ids():
    history = _history()
    event = history["event"]

    for event_type in EVENT_TYPES.values():
        assert (event["event_type"] == event_type).any()
    assert (
        entity_namespace(event["truth_event_id"]) == ENTITY_NAMESPACE["event"]
    ).all()
    assert np.array_equal(
        event["truth_event_id"],
        np.sort(event["truth_event_id"], kind="stable"),
    )
    assert (event["recorded_tick"] >= event["tick"]).all()
    assert (event["recorded_tick"] > event["tick"]).any()
    assert not any(name.startswith("observed_") for name in event)


def test_replay_conserves_population_housing_jobs_and_beds_exactly():
    micro, identities, dwellings, businesses, hospitals = _start()
    history = _history()
    replayed = replay_event_history(history)

    for table_name in replayed:
        for name in replayed[table_name]:
            assert np.array_equal(
                replayed[table_name][name],
                history["terminal_state"][table_name][name],
            )
    births = int((history["event"]["event_type"] == EVENT_TYPES["person_birth"]).sum())
    deaths = int((history["event"]["event_type"] == EVENT_TYPES["person_death"]).sum())
    assert int(replayed["person"]["is_alive"].sum()) == TOTAL + births - deaths
    assert int(replayed["dwelling"]["resident_count"].sum()) == int(
        replayed["person"]["is_alive"].sum()
    )
    active_jobs = replayed["job"]["is_active"]
    assert len(np.unique(replayed["job"]["truth_person_id"][active_jobs])) == int(
        active_jobs.sum()
    )
    open_encounters = replayed["encounter"]["is_open"]
    assert len(
        np.unique(replayed["encounter"]["truth_person_id"][open_encounters])
    ) == int(open_encounters.sum())
    validate_event_history(
        history,
        micro,
        identities,
        dwellings,
        businesses,
        hospitals,
        SEED,
    )


def test_longer_generation_only_appends_and_vintages_hide_late_reports():
    six_months = _history(6)
    twelve_months = _history(12)
    prefix_length = six_months["n_events"]

    for name in six_months["event"]:
        assert np.array_equal(
            six_months["event"][name],
            twelve_months["event"][name][:prefix_length],
        )
    six_from_long_history = replay_event_history(twelve_months, through_tick=6)
    for table_name in six_from_long_history:
        for name in six_from_long_history[table_name]:
            assert np.array_equal(
                six_from_long_history[table_name][name],
                six_months["terminal_state"][table_name][name],
            )
    preliminary = events_visible_at(twelve_months, 6)
    effective_by_six = int((twelve_months["event"]["tick"] <= 6).sum())
    assert len(preliminary["truth_event_id"]) < effective_by_six
    assert (preliminary["recorded_tick"] <= 6).all()


def test_history_is_byte_deterministic():
    micro, identities, dwellings, businesses, hospitals = _start()
    first = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=6,
    )
    second = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=6,
    )
    assert _history_digest(first) == _history_digest(second)


def test_capacity_filtered_admissions_do_not_gap_encounter_identities():
    micro, identities, dwellings, businesses, _ = _start()
    scarce_hospitals = build_hospitals(
        micro,
        SEED,
        identities,
        businesses,
        HospitalParams(beds_per_1000=0.10),
    )
    history = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        scarce_hospitals,
        months=1,
    )
    encounter_ids = history["terminal_state"]["encounter"]["truth_encounter_id"]
    sequence = encounter_ids & np.uint64((1 << 56) - 1)

    assert np.array_equal(sequence, np.arange(1, len(sequence) + 1))


def test_terminal_state_tamper_is_rejected():
    micro, identities, dwellings, businesses, hospitals = _start()
    history = _history(6)
    changed = {
        **history,
        "terminal_state": {
            **history["terminal_state"],
            "dwelling": {**history["terminal_state"]["dwelling"]},
        },
    }
    changed["terminal_state"]["dwelling"]["resident_count"] = history["terminal_state"][
        "dwelling"
    ]["resident_count"].copy()
    changed["terminal_state"]["dwelling"]["resident_count"][0] += 1

    with pytest.raises(ValueError, match="terminal event state differs from replay"):
        validate_event_history(
            changed,
            micro,
            identities,
            dwellings,
            businesses,
            hospitals,
            SEED,
        )


def test_move_history_tamper_is_rejected():
    micro, identities, dwellings, businesses, hospitals = _start()
    history = _history(6)
    changed = {**history, "event": {**history["event"]}}
    changed["event"]["truth_prior_dwelling_id"] = history["event"][
        "truth_prior_dwelling_id"
    ].copy()
    move_position = np.flatnonzero(
        changed["event"]["event_type"] == EVENT_TYPES["household_moved"]
    )[0]
    changed["event"]["truth_prior_dwelling_id"][move_position] = changed["event"][
        "truth_dwelling_id"
    ][move_position]

    with pytest.raises(ValueError, match="wrong prior dwelling"):
        validate_event_history(
            changed,
            micro,
            identities,
            dwellings,
            businesses,
            hospitals,
            SEED,
        )


def test_shock_schedule_changes_demographic_event_counts():
    micro, identities, dwellings, businesses, hospitals = _start()
    baseline = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=12,
        shocks=[],
    )
    shocked = build_event_history(
        micro,
        SEED,
        identities,
        dwellings,
        businesses,
        hospitals,
        months=12,
        shocks=[
            {
                "year": 0,
                "kind": "compound_intervention",
                "mortality_multiplier": 4.0,
                "fertility_multiplier": 0.25,
                "leave_home_multiplier": 3.0,
            }
        ],
    )

    baseline_deaths = int(
        (baseline["event"]["event_type"] == EVENT_TYPES["person_death"]).sum()
    )
    shocked_deaths = int(
        (shocked["event"]["event_type"] == EVENT_TYPES["person_death"]).sum()
    )
    baseline_births = int(
        (baseline["event"]["event_type"] == EVENT_TYPES["person_birth"]).sum()
    )
    shocked_births = int(
        (shocked["event"]["event_type"] == EVENT_TYPES["person_birth"]).sum()
    )
    assert shocked_deaths > baseline_deaths * 2
    assert shocked_births < baseline_births * 0.5


def test_event_builder_rejects_a_seed_from_another_truth_world():
    micro, identities, dwellings, businesses, hospitals = _start()
    with pytest.raises(ValueError, match="seed does not match"):
        build_event_history(
            micro,
            SEED + 1,
            identities,
            dwellings,
            businesses,
            hospitals,
            months=2,
        )
