"""Render layer: map figures straight from the truth arrays.

Every rendering is generated from engine outputs only, so a picture is also a check: rivers
follow accumulation, coasts follow sea level, shading follows the elevation the tests
verified.
"""

from __future__ import annotations

import numpy as np


def hillshade(elevation: np.ndarray, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    gy, gx = np.gradient(elevation)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy) * 40.0)
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0.0, 1.0)


def render_map(world: dict, direction: np.ndarray, accumulation: np.ndarray, path: str, river_threshold: int = 40, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elevation = world["elevation"]
    land = world["land"]
    shade = hillshade(elevation)
    height, width = elevation.shape

    rgb = np.zeros((height, width, 3))
    # sea: depth-graded blues
    depth = np.where(~land, world["sea_level"] - elevation, 0.0)
    dmax = max(depth.max(), 1e-9)
    rgb[~land] = np.stack([0.25 - 0.15 * (depth[~land] / dmax),
                           0.45 - 0.25 * (depth[~land] / dmax),
                           0.70 - 0.20 * (depth[~land] / dmax)], axis=1)
    # land: elevation-graded greens to browns to white, modulated by hillshade
    rel = np.where(land, elevation - world["sea_level"], 0.0)
    rmax = max(rel.max(), 1e-9)
    t = np.clip(rel / rmax, 0, 1)
    low = np.array([0.35, 0.55, 0.30])
    mid = np.array([0.55, 0.48, 0.32])
    high = np.array([0.92, 0.92, 0.94])
    landcol = (np.where(t[..., None] < 0.5,
                        low + (mid - low) * (t[..., None] / 0.5),
                        mid + (high - mid) * ((t[..., None] - 0.5) / 0.5)))
    rgb[land] = (landcol * (0.55 + 0.45 * shade[..., None]))[land]
    # rivers: width by accumulation
    rivers = (accumulation >= river_threshold) & land
    big = (accumulation >= 4 * river_threshold) & land
    rgb[rivers] = [0.16, 0.35, 0.65]
    rgb[big] = [0.10, 0.28, 0.58]

    fig, ax = plt.subplots(figsize=(width / 24, height / 24), dpi=220)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_population(world: dict, direction: np.ndarray, accumulation: np.ndarray,
                      population: np.ndarray, settlements: list, path: str,
                      river_threshold: int = 40, title: str = "") -> None:
    """Terrain base with population as warm glow and settlements as ranked markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elevation = world["elevation"]
    land = world["land"]
    shade = hillshade(elevation)
    height, width = elevation.shape

    rgb = np.zeros((height, width, 3))
    depth = np.where(~land, world["sea_level"] - elevation, 0.0)
    dmax = max(depth.max(), 1e-9)
    rgb[~land] = np.stack([0.10 - 0.05 * (depth[~land] / dmax),
                           0.16 - 0.08 * (depth[~land] / dmax),
                           0.28 - 0.10 * (depth[~land] / dmax)], axis=1)
    rel = np.where(land, elevation - world["sea_level"], 0.0)
    t = np.clip(rel / max(rel.max(), 1e-9), 0, 1)
    dark = np.array([0.13, 0.16, 0.14])
    pale = np.array([0.30, 0.30, 0.26])
    rgb[land] = ((dark + (pale - dark) * t[..., None]) * (0.6 + 0.4 * shade[..., None]))[land]
    rivers = (accumulation >= river_threshold) & land
    rgb[rivers] = [0.10, 0.22, 0.38]

    glow = np.log1p(population.astype(np.float64))
    gmax = max(glow.max(), 1e-9)
    # Floor the mapping so sparse rural mass stays dark land and the urban
    # hierarchy carries the light; the allocation itself is untouched.
    intensity = np.clip(glow / gmax - 0.42, 0.0, 1.0) / 0.58
    intensity = intensity ** 1.6
    warm = np.array([1.00, 0.82, 0.40])
    rgb = np.clip(rgb * (1.0 - 0.85 * intensity[..., None]) + warm * intensity[..., None], 0, 1)

    fig, ax = plt.subplots(figsize=(width / 24, height / 24), dpi=220)
    ax.imshow(rgb, interpolation="nearest")
    for rank, (r, c) in enumerate(settlements, start=1):
        size = max(26.0 / rank ** 0.5, 5.0)
        ax.scatter([c], [r], s=size, facecolors="none", edgecolors="#ffd27a",
                   linewidths=0.7, zorder=3)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, loc="left", color="#333333")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
