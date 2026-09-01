"""Draw the first nation's states and counties from stored state, and print the thin
counties a release must still cover."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.admin import build_admin, county_totals
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.population import build_population, resource_outposts
from meridia.render import hillshade
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 288, 384
TOTAL = 2_400_000
SETTLEMENTS = 24
STATES = 6
OUT = Path(__file__).resolve().parents[1] / "renders" / f"meridia-nation-{SEED}-admin.png"

t0 = time.time()
world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)
people = build_population(world, accumulation, TOTAL, SETTLEMENTS, seed=SEED)
outposts = resource_outposts(world, SEED)
admin = build_admin(world["land"], people["settlements"], outposts, n_states=STATES)
by_county = county_totals(people["population"], admin["county"].flatten(), admin["n_counties"])
by_state = np.zeros(admin["n_states"], dtype=np.int64)
np.add.at(by_state, admin["county_state"], by_county)
print(f"built in {time.time() - t0:.1f}s: {admin['n_states']} states, {admin['n_counties']} counties")
print("state populations:", ", ".join(f"{p:,}" for p in by_state))
order = np.argsort(by_county)
print("five smallest counties:", ", ".join(
    f"{by_county[c]:,}{' (outpost)' if admin['county_is_outpost'][c] else ''}" for c in order[:5]))
print("five largest counties:", ", ".join(f"{by_county[c]:,}" for c in order[-5:][::-1]))
assert by_county.sum() == TOTAL and by_state.sum() == TOTAL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

land = world["land"]
shade = hillshade(world["elevation"])
palette = np.array([[0.85, 0.55, 0.45], [0.50, 0.70, 0.50], [0.55, 0.60, 0.85],
                    [0.90, 0.80, 0.45], [0.75, 0.55, 0.80], [0.50, 0.80, 0.80]])
rgb = np.full((H, W, 3), [0.30, 0.45, 0.65])
state = admin["state"]
colors = palette[state % len(palette)] * (0.55 + 0.45 * shade[..., None])
rgb[land] = colors[land]
county = admin["county"]
edge = np.zeros((H, W), dtype=bool)
edge[:, 1:] |= (county[:, 1:] != county[:, :-1]) & land[:, 1:] & land[:, :-1]
edge[1:, :] |= (county[1:, :] != county[:-1, :]) & land[1:, :] & land[:-1, :]
rgb[edge] = [0.15, 0.15, 0.15]

fig, ax = plt.subplots(figsize=(W / 24, H / 24), dpi=220)
ax.imshow(rgb, interpolation="nearest")
for k, flat in enumerate(admin["county_seat"]):
    r, c = divmod(int(flat), W)
    if admin["county_is_outpost"][k]:
        ax.plot(c, r, marker="^", color="black", markersize=2.5, linewidth=0)
    else:
        ax.plot(c, r, marker="o", color="black", markersize=2.0, linewidth=0)
for flat in admin["state_capital"]:
    r, c = divmod(int(flat), W)
    ax.plot(c, r, marker="*", color="white", markeredgecolor="black", markersize=7, linewidth=0)
ax.set_axis_off()
ax.set_title(f"Seed {SEED}: {admin['n_states']} states (colour), {admin['n_counties']} counties "
             f"(lines); stars are state capitals, triangles outpost seats", fontsize=8, loc="left")
fig.tight_layout(pad=0.2)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
