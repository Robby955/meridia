"""World character: draws differ, stay in declared ranges, and shape real societies."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.character import CHARACTER_RANGES, draw_world_character, gini
from meridia.demography import DemographyParams, mortality_probability
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


def test_the_mortality_shape_and_the_move_rule_are_drawn_per_world():
    """Four demography fields were world constants no character draw replaced.

    The age slope is the sharpest of them: it is the first quantity a mortality model
    fits, and the five-year experience file identifies it well, so one value in every
    world handed a method the hardest parameter of the mortality model for free. The
    hazard floor, the first-year excess and the share of movers heading to a city are
    the same argument one step down.
    """
    fields = ("gompertz_b", "makeham", "infant_extra", "move_city_prob")
    default = DemographyParams()
    drawn = {name: [] for name in fields}
    for seed in range(12):
        demography = draw_world_character(seed)["demography"]
        for name in fields:
            value = float(getattr(demography, name))
            low, high = CHARACTER_RANGES[name]
            assert low <= value <= high, (seed, name, value)
            drawn[name].append(value)
    for name in fields:
        assert len(set(drawn[name])) == 12, name
        # Not the dataclass default in every world, which is what a constant looks like.
        assert sum(v == float(getattr(default, name)) for v in drawn[name]) == 0, name

    # The drawn slope reaches the hazard the ledger reads, not just the record.
    age = np.arange(0, 96)
    steep = max(range(12), key=lambda s: draw_world_character(s)["demography"].gompertz_b)
    flat = min(range(12), key=lambda s: draw_world_character(s)["demography"].gompertz_b)
    q_steep = mortality_probability(age, draw_world_character(steep)["demography"])
    q_flat = mortality_probability(age, draw_world_character(flat)["demography"])
    assert q_steep[90] / q_steep[60] > q_flat[90] / q_flat[60]
