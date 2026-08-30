"""Hydrology layer: depression filling, flow directions, and flow accumulation.

Priority-flood depression filling (Barnes et al. 2014, the standard exact algorithm) turns
the raw elevation into a hydrologically conditioned surface where every land cell drains to
the sea. D8 flow directions point each cell to its steepest-descent neighbor on the filled
surface; flow accumulation counts, for every cell, the number of cells draining through it
(each cell contributes one unit of runoff), computed in topological order. Conservation
holds exactly: the accumulation delivered into outlets (sea and border) equals the interior land-cell count.
"""

from __future__ import annotations

import heapq

import numpy as np

NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def fill_depressions(elevation: np.ndarray, sea_level: float) -> np.ndarray:
    """Priority-flood fill: minimal raising so every land cell drains to the sea."""
    height, width = elevation.shape
    filled = elevation.copy()
    visited = np.zeros((height, width), dtype=bool)
    heap: list[tuple[float, int, int, int]] = []
    counter = 0
    sea = elevation <= sea_level
    edge = np.zeros_like(visited)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    for r, c in zip(*np.nonzero(sea | edge)):
        visited[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), counter, int(r), int(c)))
        counter += 1
    epsilon = 1e-9  # strictly increasing fill so no flats survive and every path descends
    while heap:
        level, _, r, c = heapq.heappop(heap)
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                visited[nr, nc] = True
                filled[nr, nc] = max(float(filled[nr, nc]), level + epsilon)
                heapq.heappush(heap, (float(filled[nr, nc]), counter, nr, nc))
                counter += 1
    return filled


def flow_directions(filled: np.ndarray, outlet_mask: np.ndarray) -> np.ndarray:
    """D8 steepest descent on the filled surface; -1 marks outlet cells (sea and border).

    Ties break by fixed neighbor order, so directions are deterministic. On the filled
    surface every land cell has a neighbor at or below its level along a drainage path;
    flats drain because fill order induces a consistent gradient via tie-breaking on
    strictly-lower-or-equal neighbors already connected to the sea.
    """
    height, width = filled.shape
    direction = np.full((height, width), -1, dtype=np.int8)
    for r in range(height):
        for c in range(width):
            if outlet_mask[r, c]:
                continue
            best_drop, best_k = -np.inf, -1
            for k, (dr, dc) in enumerate(NEIGHBORS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    drop = filled[r, c] - filled[nr, nc]
                    if drop > best_drop:
                        best_drop, best_k = drop, k
            direction[r, c] = best_k
    return direction


def flow_accumulation(direction: np.ndarray, outlet_mask: np.ndarray) -> np.ndarray:
    """Cells draining through each cell (inclusive), by Kahn topological order."""
    height, width = direction.shape
    accumulation = np.ones((height, width), dtype=np.int64)
    accumulation[outlet_mask] = 0
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


def outflow_to_outlets(direction: np.ndarray, accumulation: np.ndarray, outlet_mask: np.ndarray) -> int:
    """Total units delivered into outlet cells; equals the interior land-cell count exactly."""
    height, width = direction.shape
    total = 0
    for r in range(height):
        for c in range(width):
            k = direction[r, c]
            if k < 0:
                continue
            nr, nc = r + NEIGHBORS[k][0], c + NEIGHBORS[k][1]
            if outlet_mask[nr, nc]:
                total += int(accumulation[r, c])
    return total
