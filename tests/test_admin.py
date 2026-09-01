"""Administrative geography: exact partition, seats in their own counties, determinism."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.admin import build_admin, county_totals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population, resource_outposts
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 250_000
SETTLEMENTS = 8


def _setup():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, SETTLEMENTS, seed=SEED)
    outposts = resource_outposts(world, SEED)
    admin = build_admin(world["land"], people["settlements"], outposts, n_states=3)
    return world, people, outposts, admin


def test_every_land_cell_has_one_county_and_state():
    world, _, _, admin = _setup()
    land = world["land"]
    assert (admin["county"][land] >= 0).all() and (admin["county"][~land] == -1).all()
    assert (admin["state"][land] >= 0).all() and (admin["state"][~land] == -1).all()
    assert admin["county"].max() == admin["n_counties"] - 1
    assert admin["state"].max() == admin["n_states"] - 1


def test_seats_sit_in_their_own_county_and_capitals_in_their_own_state():
    _, _, _, admin = _setup()
    county_flat = admin["county"].flatten()
    for k, seat in enumerate(admin["county_seat"]):
        assert county_flat[seat] == k
    state_flat = admin["state"].flatten()
    for s, capital in enumerate(admin["state_capital"]):
        assert state_flat[capital] == s


def test_population_conserved_through_the_hierarchy():
    _, people, _, admin = _setup()
    by_county = county_totals(people["population"], admin["county"].flatten(), admin["n_counties"])
    assert by_county.sum() == TOTAL
    by_state = np.zeros(admin["n_states"], dtype=np.int64)
    np.add.at(by_state, admin["county_state"], by_county)
    assert by_state.sum() == TOTAL
    state_direct = np.bincount(admin["state"].flatten()[admin["state"].flatten() >= 0],
                               weights=people["population"].flatten()[admin["state"].flatten() >= 0],
                               minlength=admin["n_states"]).astype(np.int64)
    assert (state_direct == by_state).all()


def test_outposts_become_thin_counties():
    _, people, outposts, admin = _setup()
    assert len(outposts) > 0
    by_county = county_totals(people["population"], admin["county"].flatten(), admin["n_counties"])
    outpost_pop = by_county[admin["county_is_outpost"]]
    settlement_pop = by_county[~admin["county_is_outpost"]]
    assert outpost_pop.max() < settlement_pop.max()
    assert np.median(outpost_pop) < np.median(settlement_pop)


def test_nearest_seat_rule_and_rank_tiebreak():
    land = np.ones((5, 9), dtype=bool)
    admin = build_admin(land, [(2, 2), (2, 6)], n_states=1)
    # Column 4 is equidistant (Chebyshev 2) from both seats: the higher-ranked seat wins.
    assert (admin["county"][:, :5] == 0).all() and (admin["county"][:, 5:] == 1).all()
    assert admin["n_states"] == 1 and (admin["state"] == 0).all()


def test_outpost_on_a_settlement_cell_is_folded_into_that_county():
    land = np.ones((5, 9), dtype=bool)
    admin = build_admin(land, [(2, 2), (2, 6)], outposts=[(2, 2), (0, 8)], n_states=1)
    assert admin["n_counties"] == 3
    assert list(admin["county_is_outpost"]) == [False, False, True]


def test_seat_off_land_rejected():
    land = np.ones((4, 4), dtype=bool)
    land[0, 0] = False
    with pytest.raises(ValueError):
        build_admin(land, [(0, 0)])


def test_admin_deterministic():
    _, _, _, a = _setup()
    _, _, _, b = _setup()
    assert (a["county"] == b["county"]).all() and (a["state"] == b["state"]).all()
