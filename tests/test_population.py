"""Population layer: conservation, placement, and determinism."""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population, grid_distance, habitability, seed_settlements
from meridia.terrain import generate_elevation

SEED = 777
H, W = 96, 128
TOTAL = 250_000


def _world():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    return world, accumulation


def test_population_conserved_exactly():
    world, accumulation = _world()
    people = build_population(world, accumulation, TOTAL, 8)
    assert int(people["population"].sum()) == TOTAL


def test_no_population_off_land():
    world, accumulation = _world()
    people = build_population(world, accumulation, TOTAL, 8)
    assert int(people["population"][~world["land"]].sum()) == 0
    assert float(people["habitability"][~world["land"]].max(initial=0.0)) == 0.0


def test_settlements_on_land_and_spaced():
    world, accumulation = _world()
    hab = habitability(world, accumulation)
    sites = seed_settlements(hab, 8)
    assert len(sites) == 8
    for r, c in sites:
        assert world["land"][r, c]
    for i, (r1, c1) in enumerate(sites):
        for r2, c2 in sites[i + 1:]:
            assert max(abs(r1 - r2), abs(c1 - c2)) > 14 // 2


def test_grid_distance_zero_at_sources():
    sources = np.zeros((10, 10), dtype=bool)
    sources[3, 4] = True
    d = grid_distance(sources)
    assert d[3, 4] == 0.0
    assert d[0, 0] == 4.0  # Chebyshev metric: max(3, 4)


def test_population_deterministic():
    digests = []
    for _ in range(2):
        world, accumulation = _world()
        people = build_population(world, accumulation, TOTAL, 8)
        digests.append(hashlib.sha256(people["population"].tobytes()).hexdigest())
    assert digests[0] == digests[1]


def test_capital_is_densest_neighborhood():
    world, accumulation = _world()
    people = build_population(world, accumulation, TOTAL, 8)
    grid = people["population"]
    r, c = people["settlements"][0]
    capital = grid[max(0, r - 4):r + 5, max(0, c - 4):c + 5].sum()
    rng = np.random.default_rng(0)
    land_idx = np.argwhere(world["land"])
    samples = []
    for i in rng.choice(len(land_idx), size=50, replace=False):
        rr, cc = land_idx[i]
        samples.append(grid[max(0, rr - 4):rr + 5, max(0, cc - 4):cc + 5].sum())
    assert capital >= max(samples)


def test_drawn_totals_vary_and_conserve():
    from meridia.population import draw_national_total
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    totals = [draw_national_total(s, int(world["land"].sum())) for s in (1, 2, 3, 4)]
    assert len(set(totals)) == 4
    people = build_population(world, accumulation, None, 8, seed=SEED)
    assert int(people["population"].sum()) == people["total"]
    assert people["total"] == draw_national_total(SEED, int(world["land"].sum()))


def test_outposts_put_people_in_rough_country():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, TOTAL, 8, seed=SEED)
    land = world["land"]
    rel = np.where(land, world["elevation"] - world["sea_level"], -np.inf)
    highland = land & (rel >= np.quantile(rel[land], 0.75))
    highland_pop = int(people["population"][highland].sum())
    assert highland_pop > 0.005 * TOTAL  # rough country holds real, small settlements
    assert int(people["population"].sum()) == TOTAL  # conservation untouched
