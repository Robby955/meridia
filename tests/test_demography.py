"""Demography: exact accounting, plausible rates, emergent life table, determinism."""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.demography import (SHOCK_LOADING_BAND, draw_shock_loadings,
                                period_life_expectancy, regional_multiplier, run_years,
                                step_year)
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 250_000


def _start():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 8)
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], SEED)
    return micro


def test_population_accounting_exact():
    micro = _start()
    person, hh_cell, registers = run_years(
        micro["person"], micro["household_cell"], micro["urbanity"].flatten(), SEED, 5)
    n = TOTAL
    for reg in registers:
        assert reg["population_start"] == n
        n = n + reg["births"] - reg["deaths"]
        assert reg["population_end"] == n
    assert len(person["age"]) == n


def test_rates_plausible():
    micro = _start()
    _, _, registers = run_years(
        micro["person"], micro["household_cell"], micro["urbanity"].flatten(), SEED, 5)
    for reg in registers:
        crude_death = reg["deaths"] / reg["population_start"] * 1000
        crude_birth = reg["births"] / reg["population_start"] * 1000
        assert 5 < crude_death < 20
        assert 5 < crude_birth < 25


def test_implied_life_expectancy_realistic():
    e0 = period_life_expectancy()
    assert 70 < e0 < 90


def test_old_die_more_than_young():
    micro = _start()
    _, _, registers = run_years(
        micro["person"], micro["household_cell"], micro["urbanity"].flatten(), SEED, 3)
    ages = np.concatenate([r["death_ages"] for r in registers])
    assert np.median(ages) > 60


def test_movers_form_new_households():
    micro = _start()
    person, hh_cell, registers = run_years(
        micro["person"], micro["household_cell"], micro["urbanity"].flatten(), SEED, 5)
    total_moves = sum(r["moves"] for r in registers)
    assert total_moves > 0
    assert len(hh_cell) == micro["n_households"] + total_moves
    # household cells stay consistent with member cells
    assert np.array_equal(hh_cell[person["household"]], person["cell"])


def test_step_deterministic():
    micro = _start()
    digests = []
    for _ in range(2):
        person, _, _ = step_year(micro["person"], micro["household_cell"],
                                 micro["urbanity"].flatten(), SEED, 0)
        blob = b"".join(np.ascontiguousarray(v).tobytes() for v in person.values())
        digests.append(hashlib.sha256(blob).hexdigest())
    assert digests[0] == digests[1]


def test_shock_dial_creates_excess_deaths_and_is_recorded():
    from meridia.demography import draw_world_shocks
    micro = _start()
    shocks = [{"year": 3, "kind": "mortality_spike", "mortality_multiplier": 2.5}]
    _, _, registers = run_years(micro["person"], micro["household_cell"],
                                micro["urbanity"].flatten(), SEED, 5, shocks=shocks)
    assert registers[3]["shocked"] and not registers[2]["shocked"]
    assert registers[3]["deaths"] > 1.6 * registers[2]["deaths"]
    for s in (7, 9, 11):
        a, b = draw_world_shocks(s, 30), draw_world_shocks(s, 30)
        assert a == b
        for shock in a:
            assert 2 <= shock["year"] < 30


def test_regional_shock_loadings_are_a_per_world_draw_from_the_published_band():
    """Each region's share of the shared shock family, drawn once per world."""
    low, high = SHOCK_LOADING_BAND
    first = draw_shock_loadings(11, 6)
    second = draw_shock_loadings(12, 6)
    assert first.shape == (6,)
    assert (first >= low).all() and (first <= high).all()
    assert not np.allclose(first, second)
    assert np.array_equal(first, draw_shock_loadings(11, 6))
    assert float(first.std()) > 0.0
    try:
        draw_shock_loadings(11, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("a world with no region should be refused")


def test_a_loading_scales_the_departure_from_one_and_nothing_else():
    """A loading of one takes the whole multiplier and a loading of zero takes none."""
    loading = np.asarray([0.0, 0.5, 1.0, 2.0])
    assert regional_multiplier(1.0, loading).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert regional_multiplier(3.0, loading).tolist() == [1.0, 2.0, 3.0, 5.0]
    # A multiplier under one thins less where the loading is low, which is the same rule
    # read in the other direction.
    thinned = regional_multiplier(0.5, loading)
    assert thinned.tolist() == [1.0, 0.75, 0.5, 0.0]
