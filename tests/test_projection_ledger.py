"""Stage ten on the ledger: the replayed future is the same world the sources came from,
and at the snapshot it reproduces the microdata truth exactly."""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.admin import build_admin
from meridia.businesses import build_businesses
from meridia.character import draw_world_character
from meridia.demography import draw_world_shocks
from meridia.dwellings import build_dwellings
from meridia.events import EVENT_TYPES, build_event_history
from meridia.hospitals import build_hospitals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.identities import build_initial_identity_map
from meridia.microdata import build_microdata
from meridia.population import build_population, resource_outposts
from meridia.projection import (demand_from_truth, person_table_from_state,
                                project_truth_from_history, score_allocation)
from meridia.release import compute_truth, required_rows
from meridia.scoring import rows_from_values, score_release, validate_release
from meridia.terrain import generate_elevation

SEED = 4242
H, W = 72, 96
TOTAL = 40_000
MONTHS = 24


@lru_cache(maxsize=1)
def _world():
    character = draw_world_character(SEED)
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 6,
                              params=character["population"], seed=SEED)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], SEED, params=character["microdata"])
    admin = build_admin(world["land"], people["settlements"],
                        resource_outposts(world, SEED), n_states=2)
    identities = build_initial_identity_map(micro, SEED)
    dwellings = build_dwellings(micro, SEED, identities)
    businesses = build_businesses(micro, SEED, identities)
    hospitals = build_hospitals(micro, SEED, identities, businesses)
    history = build_event_history(micro, SEED, identities, dwellings, businesses, hospitals,
                                  months=MONTHS, shocks=draw_world_shocks(SEED, 3))
    return micro, admin, history


def test_snapshot_replay_reproduces_microdata_truth_exactly():
    micro, admin, history = _world()
    at_start = project_truth_from_history(history, admin, int(history["snapshot_tick"]))
    direct = compute_truth(micro["person"], micro["household_cell"], admin)
    assert set(at_start["truth"]) == required_rows(admin)
    for key, value in direct.items():
        other = at_start["truth"][key]
        if np.isnan(value):
            assert np.isnan(other)
        else:
            assert other == value, key


def test_future_persons_equal_initial_plus_births_minus_deaths():
    micro, admin, history = _world()
    future = project_truth_from_history(history, admin)
    event = history["event"]
    births = int((event["event_type"] == EVENT_TYPES["person_birth"]).sum())
    deaths = int((event["event_type"] == EVENT_TYPES["person_death"]).sum())
    assert future["n_persons"] == micro["n_persons"] + births - deaths
    assert future["truth"][("persons", "nation", 0)] == future["n_persons"]
    assert future["truth"][("households", "nation", 0)] == future["n_households"]
    counties = sum(future["truth"][("persons", "county", c)] for c in range(admin["n_counties"]))
    assert counties == future["n_persons"]


def test_intermediate_tick_is_a_prefix_of_the_future():
    micro, admin, history = _world()
    mid_tick = int(history["snapshot_tick"]) + MONTHS // 2
    mid = project_truth_from_history(history, admin, mid_tick)
    end = project_truth_from_history(history, admin)
    assert mid["tick"] < end["tick"]
    assert mid["truth"] != end["truth"]
    assert mid["n_persons"] > 0 and end["n_persons"] > 0


def test_ledger_future_is_scored_like_a_release_and_priced():
    micro, admin, history = _world()
    now = compute_truth(micro["person"], micro["household_cell"], admin)
    future = project_truth_from_history(history, admin)
    naive = rows_from_values(now, lambda e, v: 0.02 * max(abs(v), 1.0))
    assert validate_release(naive, admin) == []
    metrics = score_release(naive, future["truth"], admin)
    assert metrics["persons/nation"]["worst_error"] > 0.0
    demand = demand_from_truth(future["truth"], admin)
    budget = 0.8 * demand.sum()
    proportional = score_allocation(demand * budget / demand.sum(), demand, budget)
    assert proportional["feasible"] and abs(proportional["regret"]) < 1e-12


def test_person_table_excludes_the_dead_and_inactive_households():
    _, _, history = _world()
    from meridia.events import replay_event_history
    state = replay_event_history(history, int(history["terminal_tick"]))
    person, household_cell = person_table_from_state(state, int(history["terminal_tick"]))
    assert len(person["age"]) == int(state["person"]["is_alive"].sum())
    assert len(household_cell) == int(state["household"]["is_active"].sum())
    assert person["household"].max() < len(household_cell)
    assert (person["age"] >= 0).all()
