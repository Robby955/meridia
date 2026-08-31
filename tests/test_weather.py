"""Weather: conservation of the routing, orographic signal, bounds, determinism."""

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.terrain import generate_elevation
from meridia.weather import river_discharge, simulate_weather, weighted_accumulation

SEED = 777
H, W = 96, 128
HOURS = 30


def _world():
    world = generate_elevation(SEED, H, W)
    outlets = ~world["land"]
    outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
    filled = fill_depressions(world["elevation"], world["sea_level"])
    direction = flow_directions(filled, outlets)
    return world, outlets, direction


def test_unit_weights_reproduce_flow_accumulation():
    _, outlets, direction = _world()
    counts = flow_accumulation(direction, outlets)
    weighted = weighted_accumulation(direction, outlets, np.ones((H, W)))
    interior = ~outlets
    assert np.allclose(weighted[interior], counts[interior].astype(np.float64))


def test_moisture_and_precip_bounded():
    world, _, _ = _world()
    wx = simulate_weather(world, HOURS, SEED)
    assert float(wx["moisture"].min()) >= 0.0
    assert float(wx["moisture"].max()) <= 1.2
    assert float(wx["precip"].min()) >= 0.0


def test_orographic_rain_prefers_uplift():
    world, _, _ = _world()
    wx = simulate_weather(world, HOURS, SEED)
    mean_precip = wx["precip"].mean(axis=0)
    gy, gx = np.gradient(world["elevation"])
    slope = np.hypot(gx, gy)
    land = world["land"]
    steep = land & (slope > np.quantile(slope[land], 0.8))
    flat = land & (slope < np.quantile(slope[land], 0.2))
    assert mean_precip[steep].mean() > mean_precip[flat].mean()


def test_rivers_respond_to_rain_with_lag():
    world, outlets, direction = _world()
    wx = simulate_weather(world, HOURS, SEED)
    discharge = river_discharge(direction, outlets, wx["precip"], window=6)
    land = world["land"]
    rain = wx["precip"][:, land].mean(axis=1)
    flow = discharge[:, land].mean(axis=1)
    # trailing-window routing: current flow correlates more with past rain than future
    past = np.corrcoef(rain[:-3], flow[3:])[0, 1]
    future = np.corrcoef(rain[3:], flow[:-3])[0, 1]
    assert past > future


def test_weather_deterministic():
    world, _, _ = _world()
    digests = []
    for _ in range(2):
        wx = simulate_weather(world, HOURS, SEED)
        digests.append(hashlib.sha256(wx["precip"].tobytes()).hexdigest())
    assert digests[0] == digests[1]
