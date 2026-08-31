"""Render the first look at Meridia: one seeded nation, terrain plus rivers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.render import render_map
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 288, 384

world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)
out = Path(__file__).resolve().parents[1] / "renders"
out.mkdir(exist_ok=True)
path = out / f"meridia-nation-{SEED}.png"
render_map(world, direction, accumulation, str(path), river_threshold=60,
           title=f"Meridia, first nation (seed {SEED}): terrain, coast, and rivers from flow accumulation")
print(path)
print(f"land cells {int(world['land'].sum())} of {H*W}; max river accumulation {int(accumulation.max())}")
