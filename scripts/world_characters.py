"""Print the character sheet of several worlds: same laws, different societies."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.character import draw_world_character, gini
from meridia.demography import period_life_expectancy
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population, draw_national_total
from meridia.terrain import generate_elevation

H, W = 144, 192
SEEDS = (11, 23, 47, 89, 20260831)

print(f"{'seed':>10} {'people':>10} {'gini':>6} {'e0':>6} {'fertility':>10} {'primacy':>8}")
for seed in SEEDS:
    character = draw_world_character(seed)
    world = generate_elevation(seed, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    total = draw_national_total(seed, int(world["land"].sum()))
    people = build_population(world, accumulation, total, 12, params=character["population"])
    micro = build_microdata(people["population"], people["habitability"],
                            people["settlements"], seed, params=character["microdata"])
    g = gini(micro["person"]["income"])
    e0 = period_life_expectancy(character["demography"])
    print(f"{seed:>10} {total:>10,} {g:>6.3f} {e0:>6.1f} "
          f"{character['draw']['fertility_rate']:>10.3f} "
          f"{character['draw']['zipf_exponent']:>8.2f}")
