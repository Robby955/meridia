"""Render the first nation inhabited: population glow over the verified terrain."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population
from meridia.render import render_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 288, 384
TOTAL = 2_400_000
SETTLEMENTS = 24

world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)

people = build_population(world, accumulation, TOTAL, SETTLEMENTS, seed=SEED)
grid = people["population"]
sites = people["settlements"]

out = Path(__file__).resolve().parents[1] / "renders"
out.mkdir(exist_ok=True)
path = out / f"meridia-nation-{SEED}-population.png"
render_population(world, direction, accumulation, grid, sites, str(path), river_threshold=60,
                  title=f"Meridia, first nation (seed {SEED}): {TOTAL:,} people, "
                        f"{len(sites)} settlements, population conserved exactly")
print(path)
capital = grid[max(0, sites[0][0] - 6):sites[0][0] + 7, max(0, sites[0][1] - 6):sites[0][1] + 7].sum()
print(f"total population {int(grid.sum()):,} (declared {TOTAL:,})")
print(f"settlements {len(sites)}; capital neighborhood holds {int(capital):,}")
print(f"populated cells {int((grid > 0).sum())} of {int(world['land'].sum())} land cells; "
      f"max cell {int(grid.max()):,}")
