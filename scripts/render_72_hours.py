"""72 Hours in Meridia: weather state over the first nation, drawn frame by frame.

Clouds are the simulated moisture field, rain is the simulated precipitation, rivers
widen where routed discharge is high, and the day-night cycle lights the terrain. Every
pixel traces to stored world state.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population
from meridia.render import hillshade
from meridia.terrain import generate_elevation
from meridia.weather import river_discharge, simulate_weather

SEED = 20260831
H, W = 288, 384
HOURS = 72
BURN_IN = 24

world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)
people = build_population(world, accumulation, None, 24, seed=SEED)

weather_all = simulate_weather(world, HOURS + BURN_IN, SEED)
weather = {k: v[BURN_IN:] for k, v in weather_all.items()}
discharge = river_discharge(direction, outlets, weather["precip"], window=6)

elevation = world["elevation"]
land = world["land"]
depth = np.where(~land, world["sea_level"] - elevation, 0.0)
dmax = max(depth.max(), 1e-9)
rel = np.where(land, elevation - world["sea_level"], 0.0)
t_land = np.clip(rel / max(rel.max(), 1e-9), 0, 1)

glow = np.log1p(people["population"].astype(np.float64))
glow /= max(glow.max(), 1e-9)
lights = (np.clip(glow - 0.42, 0.0, 1.0) / 0.58) ** 1.6
warm = np.array([1.00, 0.82, 0.40])

base_flow = accumulation.astype(np.float64)
base_flow /= max(base_flow.max(), 1e-9)
q_hi = np.quantile(discharge[discharge > 0], 0.995)


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
    return rgb


def night_rgb():
    rgb = np.zeros((H, W, 3))
    rgb[~land] = np.stack([0.10 - 0.05 * (depth[~land] / dmax),
                           0.16 - 0.08 * (depth[~land] / dmax),
                           0.28 - 0.10 * (depth[~land] / dmax)], axis=1)
    dark, pale = np.array([0.13, 0.16, 0.14]), np.array([0.30, 0.30, 0.26])
    shade = hillshade(elevation)
    rgb[land] = ((dark + (pale - dark) * t_land[..., None]) * (0.6 + 0.4 * shade[..., None]))[land]
    return np.clip(rgb * (1.0 - 0.85 * lights[..., None]) + warm * lights[..., None], 0, 1)


night = night_rgb()
frames = []
for t in range(HOURS):
    hour = t % 24
    daylight = np.clip(0.5 + 0.5 * -np.cos(2 * np.pi * hour / 24.0), 0.0, 1.0) ** 1.5
    azimuth = 90.0 + 180.0 * (hour / 24.0)
    day = day_rgb(hillshade(elevation, azimuth_deg=azimuth, altitude_deg=20.0 + 40.0 * daylight))
    frame = daylight * day + (1.0 - daylight) * night

    # rivers widen with routed discharge
    flow = discharge[t] / max(q_hi, 1e-9)
    river = (accumulation >= 160) & land
    swollen = river & (np.clip(flow, 0, 1) > 0.35)
    frame[river] = daylight * np.array([0.16, 0.35, 0.65]) + (1 - daylight) * np.array([0.10, 0.22, 0.38])
    frame[swollen] = daylight * np.array([0.20, 0.45, 0.80]) + (1 - daylight) * np.array([0.14, 0.30, 0.55])

    # rain darkens the ground beneath it
    rain = np.clip(weather["precip"][t] / 0.06, 0, 1)
    frame *= (1.0 - 0.25 * rain[..., None])
    # clouds are the moisture field
    cloud = np.clip((weather["moisture"][t] - 0.45) / 0.55, 0, 1) ** 1.5
    cloud_color = (0.92 - 0.55 * (1.0 - daylight))
    frame = frame * (1.0 - 0.55 * cloud[..., None]) + cloud_color * 0.55 * cloud[..., None]
    frames.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))

from PIL import Image, ImageDraw

images = []
for t, fr in enumerate(frames):
    im = Image.fromarray(fr)
    draw = ImageDraw.Draw(im)
    draw.rectangle([8, 8, 150, 28], fill=(10, 12, 16))
    draw.text((14, 12), f"day {t // 24 + 1}  {t % 24:02d}:00", fill=(240, 235, 220))
    images.append(im.quantize(colors=128, dither=Image.Dither.NONE))
out = Path(__file__).resolve().parents[1] / "renders" / "meridia-72-hours.gif"
images[0].save(out, save_all=True, append_images=images[1:], duration=140, loop=0)
print(out)
land_rain = weather["precip"][:, land].mean(axis=1)
print(f"{HOURS} hourly frames; wettest hour {int(np.argmax(land_rain))} "
      f"({land_rain.max():.4f} mean precip); rivers swell after it")
