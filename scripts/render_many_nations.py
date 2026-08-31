"""Six nations from six seeds: the engine's replication claim, in one picture."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population, draw_national_total
from meridia.render import hillshade
from meridia.terrain import generate_elevation

H, W = 192, 256
SETTLEMENTS = 16
SEEDS = (11, 23, 47, 89, 131, 20260831)


def nation_rgb(seed: int) -> tuple:
    world = generate_elevation(seed, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    accumulation = flow_accumulation(direction, outlets)
    people = build_population(world, accumulation, None, SETTLEMENTS, seed=seed)

    elevation = world["elevation"]
    land = world["land"]
    shade = hillshade(elevation)
    rgb = np.zeros((H, W, 3))
    depth = np.where(~land, world["sea_level"] - elevation, 0.0)
    dmax = max(depth.max(), 1e-9)
    rgb[~land] = np.stack([0.10 - 0.05 * (depth[~land] / dmax),
                           0.16 - 0.08 * (depth[~land] / dmax),
                           0.28 - 0.10 * (depth[~land] / dmax)], axis=1)
    rel = np.where(land, elevation - world["sea_level"], 0.0)
    t = np.clip(rel / max(rel.max(), 1e-9), 0, 1)
    dark, pale = np.array([0.13, 0.16, 0.14]), np.array([0.30, 0.30, 0.26])
    rgb[land] = ((dark + (pale - dark) * t[..., None]) * (0.6 + 0.4 * shade[..., None]))[land]
    rivers = (accumulation >= 40) & land
    rgb[rivers] = [0.10, 0.22, 0.38]
    glow = np.log1p(people["population"].astype(np.float64))
    glow /= max(glow.max(), 1e-9)
    lights = (np.clip(glow - 0.42, 0.0, 1.0) / 0.58) ** 1.6
    warm = np.array([1.00, 0.82, 0.40])
    return np.clip(rgb * (1.0 - 0.85 * lights[..., None]) + warm * lights[..., None], 0, 1), people["total"]


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), dpi=170)
for ax, seed in zip(axes.flat, SEEDS):
    rgb, total = nation_rgb(seed)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"seed {seed}: {total:,} people", fontsize=9, color="#555555", loc="left")
fig.suptitle("Six nations from six seeds: geography, cities, and population size all drawn from the seed",
             fontsize=12)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "renders" / "meridia-six-nations.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
print(f"six worlds in {time.time()-t0:.1f}s")
