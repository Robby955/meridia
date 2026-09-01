"""Build the first nation's full microdata and render its demographic profile."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 288, 384
TOTAL = 2_400_000
SETTLEMENTS = 24

t0 = time.time()
world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)
people = build_population(world, accumulation, TOTAL, SETTLEMENTS, seed=SEED)
t1 = time.time()
micro = build_microdata(people["population"], people["habitability"], people["settlements"], SEED)
t2 = time.time()

person = micro["person"]
print(f"world+census {t1-t0:.1f}s, microdata {t2-t1:.1f}s")
print(f"persons {micro['n_persons']:,}; households {micro['n_households']:,}; "
      f"mean size {micro['n_persons']/micro['n_households']:.2f}")
adults = person["age"] >= 16
print(f"median age {int(np.median(person['age']))}; "
      f"children {(person['age'] < 18).mean():.1%}; elders 65+ {(person['age'] >= 65).mean():.1%}")
print(f"adult median income {np.median(person['income'][adults]):,.0f}; "
      f"education shares {np.bincount(person['education'], minlength=4) / len(person['education'])}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=180)
ages = np.arange(0, 96, 5)
male = np.histogram(person["age"][person["sex"] == 0], bins=ages)[0]
female = np.histogram(person["age"][person["sex"] == 1], bins=ages)[0]
centers = ages[:-1] + 2.5
axes[0].barh(centers, -male, height=4, color="#3b6ea5", label="male")
axes[0].barh(centers, female, height=4, color="#c98a3d", label="female")
axes[0].set_title("Age pyramid")
axes[0].set_ylabel("age")
axes[0].legend(fontsize=7)
axes[0].set_xticks([])

urb = micro["urbanity"].flatten()[person["cell"]]
bins = np.quantile(urb[adults], np.linspace(0, 1, 9))
mids, med = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sel = adults & (urb >= lo) & (urb <= hi)
    if sel.sum() > 100:
        mids.append((lo + hi) / 2)
        med.append(np.median(person["income"][sel]))
axes[1].plot(mids, med, marker="o", color="#3b6ea5")
axes[1].set_title("Median adult income by urbanity")
axes[1].set_xlabel("urbanity (settlement pull)")

sizes = np.bincount(person["household"])
axes[2].hist(sizes, bins=np.arange(1, 9) - 0.5, color="#6a8f5f", rwidth=0.85)
axes[2].set_title("Household sizes")
axes[2].set_xlabel("persons per household")

fig.suptitle(f"Meridia, first nation (seed {SEED}): {micro['n_persons']:,} persons, "
             f"{micro['n_households']:,} households", fontsize=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "renders" / f"meridia-nation-{SEED}-demographics.png"
fig.savefig(out, bbox_inches="tight")
print(out)
