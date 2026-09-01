"""Administrative geography v0: nation, states, counties.

A statistical release is published by geography, so the world needs a partition of its
land into named areas that estimates can be scored on. The hierarchy is nation, state,
county (settlement is a point, not an area). Counties are catchments: every land cell
belongs to the nearest county seat under the Chebyshev metric, ties broken by seat rank.
Seats are the settlements in rank order followed by the resource outposts, so outposts
become small counties in rough country: the thin domains a release must still cover.
States group counties: the largest settlements are state capitals, and a county belongs
to the state whose capital is nearest to its seat, ties again by rank.

The partition is exact. Every land cell has exactly one county and every county exactly
one state; population summed by county, by state, and nationally is the same integer.
Deterministic in its inputs; no random draw is made.
"""

from __future__ import annotations

import numpy as np


def _chebyshev_labels(height: int, width: int, seats: list[tuple[int, int]]) -> np.ndarray:
    """Label of the nearest seat for every cell; equal distances go to the lower index."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    best_distance = np.full((height, width), np.iinfo(np.int64).max, dtype=np.int64)
    label = np.full((height, width), -1, dtype=np.int64)
    for index, (r, c) in enumerate(seats):
        distance = np.maximum(np.abs(rows - r), np.abs(cols - c))
        closer = distance < best_distance      # strict: earlier seats keep ties
        best_distance = np.where(closer, distance, best_distance)
        label = np.where(closer, index, label)
    return label


def build_admin(land: np.ndarray, settlements: list[tuple[int, int]],
                outposts: list[tuple[int, int]] | None = None,
                n_states: int = 6) -> dict:
    """Partition the land into counties and states.

    ``settlements`` are in rank order (largest first); ``outposts`` are the resource
    sites, which become their own small counties. ``n_states`` capitals are the top-ranked
    settlements (capped at the settlement count).
    """
    height, width = land.shape
    seats = [tuple(int(v) for v in s) for s in settlements]
    seats += [tuple(int(v) for v in o) for o in (outposts or [])]
    if not seats:
        raise ValueError("at least one seat is required")
    for r, c in seats:
        if not land[r, c]:
            raise ValueError(f"seat {(r, c)} is not on land")
    if len(set(seats)) != len(seats):
        raise ValueError("seats must be distinct cells")

    county = _chebyshev_labels(height, width, seats)
    county = np.where(land, county, -1)

    n_states = max(1, min(n_states, len(settlements)))
    capitals = seats[:n_states]
    seat_rows = np.asarray([r for r, _ in seats], dtype=np.int64)
    seat_cols = np.asarray([c for _, c in seats], dtype=np.int64)
    county_state = np.empty(len(seats), dtype=np.int64)
    for k, (r, c) in enumerate(seats):
        distances = [max(abs(r - cr), abs(c - cc)) for cr, cc in capitals]
        county_state[k] = int(np.argmin(distances))   # argmin takes the first minimum
    state = np.where(county >= 0, county_state[np.maximum(county, 0)], -1)

    return {
        "county": county,
        "state": state,
        "n_counties": len(seats),
        "n_states": n_states,
        "county_seat": seat_rows * width + seat_cols,
        "county_state": county_state,
        "county_is_outpost": np.arange(len(seats)) >= len(settlements),
        "state_capital": np.asarray([r * width + c for r, c in capitals], dtype=np.int64),
    }


def county_totals(values: np.ndarray, county_flat: np.ndarray, n_counties: int) -> np.ndarray:
    """Exact integer totals of a per-cell grid by county (off-land cells are excluded)."""
    flat = np.asarray(values).flatten().astype(np.int64)
    on_land = county_flat >= 0
    totals = np.zeros(n_counties, dtype=np.int64)
    np.add.at(totals, county_flat[on_land], flat[on_land])
    return totals
