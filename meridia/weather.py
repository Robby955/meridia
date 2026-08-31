"""Weather layer v0: wind, moisture, orographic rain, and rivers that respond.

Weather is simulated state, not decoration. A prevailing wind advects a moisture field
across the map; the sea recharges it by evaporation; where wind pushes moist air up a
slope, it rains, and rain removes the moisture that fell. Precipitation is then routed
down the same D8 drainage tree the hydrology layer verified, so rivers rise after rain
with the delay of a trailing window. Every frame of any weather animation is a picture
of this state.

Conservation check: routing a uniform weight of one through the drainage tree reproduces
the hydrology layer's flow accumulation exactly.

Deterministic in (seed, world, hours).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .hydrology import NEIGHBORS


@dataclass(frozen=True)
class WeatherParams:
    wind_speed: float = 3.0          # cells per hour
    wind_drift: float = 0.08         # radians per hour of slow direction drift
    gust_sigma: float = 0.15         # per-hour random component of direction
    evaporation: float = 0.16        # sea recharge rate toward saturation
    base_rain: float = 0.015         # drizzle everywhere in proportion to moisture
    orographic_gain: float = 9.0     # rain multiplier for wind-driven uplift
    rain_cap: float = 0.5            # at most this fraction of moisture falls per hour
    diffusion: float = 0.8           # gaussian smoothing of the moisture field


def simulate_weather(world: dict, hours: int, seed: int,
                     params: WeatherParams = WeatherParams()) -> dict:
    """Hourly moisture and precipitation fields over the nation."""
    elevation = world["elevation"]
    land = world["land"]
    height, width = elevation.shape
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4EA1]))
    gy, gx = np.gradient(elevation)

    moisture = np.where(~land, 0.9, 0.35).astype(np.float64)
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    moisture_frames = np.empty((hours, height, width), dtype=np.float32)
    precip_frames = np.empty((hours, height, width), dtype=np.float32)

    for t in range(hours):
        angle += params.wind_drift + float(rng.normal(0.0, params.gust_sigma))
        wy, wx = np.sin(angle) * params.wind_speed, np.cos(angle) * params.wind_speed
        moisture = ndimage.shift(moisture, (wy, wx), order=1, mode="nearest")
        moisture = ndimage.gaussian_filter(moisture, params.diffusion)
        moisture[~land] += params.evaporation * (1.0 - moisture[~land])
        uplift = np.maximum(0.0, wy * gy + wx * gx)
        rain_fraction = np.minimum(params.rain_cap,
                                   params.base_rain + params.orographic_gain * uplift)
        precip = moisture * rain_fraction
        moisture = np.clip(moisture - precip, 0.0, 1.2)
        moisture_frames[t] = moisture
        precip_frames[t] = precip
    return {"moisture": moisture_frames, "precip": precip_frames}


def weighted_accumulation(direction: np.ndarray, outlet_mask: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
    """Route arbitrary nonnegative weights down the D8 tree (Kahn order).

    With unit weights this equals the hydrology layer's flow accumulation exactly.
    """
    height, width = direction.shape
    accumulation = weights.astype(np.float64).copy()
    accumulation[outlet_mask] = 0.0
    indegree = np.zeros((height, width), dtype=np.int32)
    for r in range(height):
        for c in range(width):
            k = direction[r, c]
            if k >= 0:
                nr, nc = r + NEIGHBORS[k][0], c + NEIGHBORS[k][1]
                indegree[nr, nc] += 1
    stack = [(r, c) for r in range(height) for c in range(width)
             if indegree[r, c] == 0 and not outlet_mask[r, c]]
    while stack:
        r, c = stack.pop()
        k = direction[r, c]
        if k < 0:
            continue
        nr, nc = r + NEIGHBORS[k][0], c + NEIGHBORS[k][1]
        if not outlet_mask[nr, nc]:
            accumulation[nr, nc] += accumulation[r, c]
        indegree[nr, nc] -= 1
        if indegree[nr, nc] == 0 and not outlet_mask[nr, nc]:
            stack.append((nr, nc))
    return accumulation


def river_discharge(direction: np.ndarray, outlet_mask: np.ndarray,
                    precip_frames: np.ndarray, window: int = 6) -> np.ndarray:
    """Hourly discharge: trailing-window precipitation routed down the drainage tree."""
    hours = precip_frames.shape[0]
    discharge = np.empty_like(precip_frames, dtype=np.float64)
    for t in range(hours):
        lo = max(0, t - window + 1)
        recent = precip_frames[lo:t + 1].mean(axis=0).astype(np.float64)
        discharge[t] = weighted_accumulation(direction, outlet_mask, recent)
    return discharge
