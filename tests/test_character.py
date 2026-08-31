"""World character: draws differ, stay in declared ranges, and shape real societies."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.character import CHARACTER_RANGES, draw_world_character, gini
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

H, W = 96, 128
TOTAL = 120_000


def _nation(seed: int):
    character = draw_world_character(seed)
    world = generate_elevation(seed, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 8, params=character["population"])
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], seed, params=character["microdata"])
    return character, micro


def test_draws_within_declared_ranges_and_deterministic():
    for seed in (1, 5, 99):
        a = draw_world_character(seed)["draw"]
        b = draw_world_character(seed)["draw"]
        assert a == b
        for name, value in a.items():
            lo, hi = CHARACTER_RANGES[name]
            assert lo <= value <= hi


def test_worlds_differ_in_drawn_character():
    draws = [tuple(draw_world_character(s)["draw"].values()) for s in range(8)]
    assert len(set(draws)) == 8


def test_inequality_follows_the_drawn_dial():
    seeds = list(range(10))
    sigmas = [draw_world_character(s)["draw"]["income_sigma"] for s in seeds]
    low_seed = seeds[int(np.argmin(sigmas))]
    high_seed = seeds[int(np.argmax(sigmas))]
    _, micro_low = _nation(low_seed)
    _, micro_high = _nation(high_seed)
    g_low = gini(micro_low["person"]["income"])
    g_high = gini(micro_high["person"]["income"])
    assert g_high > g_low + 0.03


def test_gini_known_values():
    assert gini(np.array([1.0, 1.0, 1.0, 1.0])) == 0.0
    highly_unequal = gini(np.array([0.001] * 99 + [1000.0]))
    assert highly_unequal > 0.9
