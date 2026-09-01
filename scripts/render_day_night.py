"""Animate one day over the first nation: sunlight crosses the terrain, then the
cities light up. Every frame is drawn from the same truth arrays the tests verify."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population
from meridia.render import hillshade
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 288, 384
TOTAL = 2_400_000
SETTLEMENTS = 24
FRAMES = 48
RIVER_THRESHOLD = 60

world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)
people = build_population(world, accumulation, TOTAL, SETTLEMENTS, seed=SEED)

elevation = world["elevation"]
land = world["land"]
rivers = (accumulation >= RIVER_THRESHOLD) & land
depth = np.where(~land, world["sea_level"] - elevation, 0.0)
dmax = max(depth.max(), 1e-9)
rel = np.where(land, elevation - world["sea_level"], 0.0)
t_land = np.clip(rel / max(rel.max(), 1e-9), 0, 1)

glow = np.log1p(people["population"].astype(np.float64))
glow /= max(glow.max(), 1e-9)
lights = np.clip(glow - 0.42, 0.0, 1.0) / 0.58
lights = lights ** 1.6
warm = np.array([1.00, 0.82, 0.40])


def day_rgb(shade):
    rgb = np.zeros((H, W, 3))
    rgb[~land] = np.stack([0.25 - 0.15 * (depth[~land] / dmax),
                           0.45 - 0.25 * (depth[~land] / dmax),
                           0.70 - 0.20 * (depth[~land] / dmax)], axis=1)
    low, mid, high = (np.array([0.35, 0.55, 0.30]), np.array([0.55, 0.48, 0.32]),
                      np.array([0.92, 0.92, 0.94]))
    col = np.where(t_land[..., None] < 0.5,
                   low + (mid - low) * (t_land[..., None] / 0.5),
                   mid + (high - mid) * ((t_land[..., None] - 0.5) / 0.5))
    rgb[land] = (col * (0.55 + 0.45 * shade[..., None]))[land]
    rgb[rivers] = [0.16, 0.35, 0.65]
    return rgb


def night_rgb():
    rgb = np.zeros((H, W, 3))
    rgb[~land] = np.stack([0.10 - 0.05 * (depth[~land] / dmax),
                           0.16 - 0.08 * (depth[~land] / dmax),
                           0.28 - 0.10 * (depth[~land] / dmax)], axis=1)
    dark, pale = np.array([0.13, 0.16, 0.14]), np.array([0.30, 0.30, 0.26])
    shade = hillshade(elevation)
    rgb[land] = ((dark + (pale - dark) * t_land[..., None]) * (0.6 + 0.4 * shade[..., None]))[land]
    rgb[rivers] = [0.10, 0.22, 0.38]
    return np.clip(rgb * (1.0 - 0.85 * lights[..., None]) + warm * lights[..., None], 0, 1)


night = night_rgb()
frames = []
for f in range(FRAMES):
    hour = 24.0 * f / FRAMES
    # daylight weight: full day near noon, full night near midnight, smooth dawn/dusk
    daylight = np.clip(0.5 + 0.5 * -np.cos(2 * np.pi * hour / 24.0), 0.0, 1.0) ** 1.5
    # sun sweeps east to west through the day; azimuth only matters while it is up
    azimuth = 90.0 + 180.0 * (hour / 24.0)
    day = day_rgb(hillshade(elevation, azimuth_deg=azimuth,
                            altitude_deg=20.0 + 40.0 * daylight))
    frame = daylight * day + (1.0 - daylight) * night
    frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

from PIL import Image

out = Path(__file__).resolve().parents[1] / "renders" / f"meridia-nation-{SEED}-day.gif"
images = [Image.fromarray(fr) for fr in frames]
images[0].save(out, save_all=True, append_images=images[1:], duration=120, loop=0)
print(out)
print(f"{FRAMES} frames, one simulated day: dawn light from the east, dusk from the west, city lights at night")
