"""Population layer v0: habitability, settlements, and an exact-count population grid.

Habitability is computed from the layers the engine already verified: elevation above sea
level (lowlands preferred), local slope (flat preferred), distance to fresh water (river
cells from flow accumulation), and distance to the coast. Settlements are seeded greedily
on habitability with a spacing constraint and given rank-size (Zipf) weights. People are
spread as habitability-weighted mass around settlements and allocated to cells as exact
integers by the largest-remainder method, so the grid sums to the declared total exactly:
population is conserved the way runoff is.

Everything is float64/int64, single-threaded, and fully determined by its inputs; the same
inputs yield byte-identical arrays on the same platform.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass(frozen=True)
class PopulationParams:
    river_threshold: int = 60        # accumulation at or above this counts as fresh water
    elevation_scale: float = 0.6     # e-folding of the elevation penalty (relative units)
    slope_scale: float = 25.0        # slope penalty steepness
    water_scale: float = 12.0        # e-folding distance (cells) of the fresh-water bonus
    coast_scale: float = 30.0        # e-folding distance (cells) of the coast bonus
    water_weight: float = 0.45       # mix of the fresh-water bonus
    coast_weight: float = 0.25       # mix of the coast bonus
    settlement_spacing: int = 14     # minimum Chebyshev distance between settlements
    settlement_reach: float = 18.0   # e-folding distance (cells) of a settlement's pull
    zipf_exponent: float = 1.0       # rank-size law for settlement weights
    background_share: float = 0.12   # mass spread by habitability alone (rural floor)


def grid_distance(sources: np.ndarray) -> np.ndarray:
    """Chebyshev-metric BFS distance (in cells) from the nearest True cell; inf if none."""
    height, width = sources.shape
    distance = np.full((height, width), np.inf)
    queue: deque[tuple[int, int]] = deque()
    for r, c in zip(*np.nonzero(sources)):
        distance[r, c] = 0.0
        queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        d = distance[r, c] + 1.0
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and distance[nr, nc] > d:
                distance[nr, nc] = d
                queue.append((nr, nc))
    return distance


def habitability(world: dict, accumulation: np.ndarray,
                 params: PopulationParams = PopulationParams()) -> np.ndarray:
    """Habitability in [0, 1]; exactly zero off land."""
    elevation = world["elevation"]
    land = world["land"]
    rel = np.where(land, elevation - world["sea_level"], 0.0)
    rmax = max(rel.max(), 1e-12)
    elev_term = np.exp(-(rel / rmax) / params.elevation_scale)
    gy, gx = np.gradient(elevation)
    slope = np.hypot(gx, gy)
    slope_term = np.exp(-params.slope_scale * slope)
    rivers = (accumulation >= params.river_threshold) & land
    water_term = np.exp(-grid_distance(rivers) / params.water_scale) if rivers.any() else np.zeros_like(elevation)
    coast_term = np.exp(-grid_distance(~land) / params.coast_scale)
    base_weight = 1.0 - params.water_weight - params.coast_weight
    score = elev_term * slope_term * (base_weight
                                      + params.water_weight * water_term
                                      + params.coast_weight * coast_term)
    score = np.where(land, score, 0.0)
    smax = score.max()
    return score / smax if smax > 0 else score


def seed_settlements(habitability_grid: np.ndarray, n_settlements: int,
                     params: PopulationParams = PopulationParams()) -> list[tuple[int, int]]:
    """Greedy argmax with a spacing constraint; deterministic (ties break by flat index)."""
    working = habitability_grid.copy()
    height, width = working.shape
    sites: list[tuple[int, int]] = []
    s = params.settlement_spacing
    for _ in range(n_settlements):
        flat = int(np.argmax(working))
        if working.flat[flat] <= 0.0:
            break
        r, c = divmod(flat, width)
        sites.append((r, c))
        r0, r1 = max(0, r - s), min(height, r + s + 1)
        c0, c1 = max(0, c - s), min(width, c + s + 1)
        working[r0:r1, c0:c1] = 0.0
    return sites


def population_grid(habitability_grid: np.ndarray, sites: list[tuple[int, int]], total: int,
                    params: PopulationParams = PopulationParams()) -> np.ndarray:
    """Integer population per cell, summing to ``total`` exactly (largest remainder)."""
    height, width = habitability_grid.shape
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    pull = np.zeros((height, width))
    for rank, (r, c) in enumerate(sites, start=1):
        distance = np.maximum(np.abs(rows - r), np.abs(cols - c))
        pull += (rank ** -params.zipf_exponent) * np.exp(-distance / params.settlement_reach)
    urban = habitability_grid * pull
    rural = habitability_grid.astype(np.float64)
    urban_mass, rural_mass = urban.sum(), rural.sum()
    if rural_mass <= 0:
        raise ValueError("no habitable mass to allocate population onto")
    if urban_mass > 0:
        density = ((1.0 - params.background_share) * urban / urban_mass
                   + params.background_share * rural / rural_mass)
    else:
        density = rural / rural_mass
    mass = density.sum()
    shares = density.flatten() * (total / mass)
    floors = np.floor(shares).astype(np.int64)
    remainder = int(total - floors.sum())
    if remainder > 0:
        order = np.argsort(-(shares - floors), kind="stable")
        floors[order[:remainder]] += 1
    return floors.reshape(height, width)


def build_population(world: dict, accumulation: np.ndarray, total: int, n_settlements: int,
                     params: PopulationParams = PopulationParams()) -> dict:
    """One call from verified layers to a conserved population grid."""
    hab = habitability(world, accumulation, params)
    sites = seed_settlements(hab, n_settlements, params)
    grid = population_grid(hab, sites, total, params)
    return {"habitability": hab, "settlements": sites, "population": grid}
