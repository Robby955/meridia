"""Run the first nation forward thirty years and chart how the country changes."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meridia.demography import period_life_expectancy, run_years
from meridia.hydrology import fill_depressions, flow_accumulation, flow_directions
from meridia.microdata import build_microdata
from meridia.population import build_population
from meridia.terrain import generate_elevation

SEED = 20260831
H, W = 288, 384
TOTAL = 2_400_000
YEARS = 30

t0 = time.time()
world = generate_elevation(SEED, H, W)
outlets = ~world["land"]
outlets[0, :] = outlets[-1, :] = outlets[:, 0] = outlets[:, -1] = True
filled = fill_depressions(world["elevation"], world["sea_level"])
direction = flow_directions(filled, outlets)
accumulation = flow_accumulation(direction, outlets)
people = build_population(world, accumulation, TOTAL, 24, seed=SEED)
micro = build_microdata(people["population"], people["habitability"], people["settlements"], SEED)
start_ages = micro["person"]["age"].copy()

person, hh_cell, registers = run_years(
    micro["person"], micro["household_cell"], micro["urbanity"].flatten(), SEED, YEARS)
t1 = time.time()

pop_path = [registers[0]["population_start"]] + [r["population_end"] for r in registers]
death_ages = np.concatenate([r["death_ages"] for r in registers])
print(f"{YEARS} years in {t1-t0:.1f}s")
print(f"population {pop_path[0]:,} -> {pop_path[-1]:,}")
print(f"births {sum(r['births'] for r in registers):,}; deaths {sum(r['deaths'] for r in registers):,}; "
      f"moves {sum(r['moves'] for r in registers):,}")
print(f"median age {int(np.median(start_ages))} -> {int(np.median(person['age']))}")
print(f"implied period e0 {period_life_expectancy():.1f}; median simulated age at death {int(np.median(death_ages))}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=180)
axes[0].plot(range(YEARS + 1), pop_path, color="#3b6ea5")
axes[0].set_title(f"Population over {YEARS} years")
axes[0].set_xlabel("year")
axes[0].ticklabel_format(style="plain", axis="y")

bins = np.arange(0, 101, 5)
axes[1].hist(start_ages, bins=bins, alpha=0.55, color="#3b6ea5", label="year 0", density=True)
axes[1].hist(person["age"], bins=bins, alpha=0.55, color="#c98a3d", label=f"year {YEARS}", density=True)
axes[1].set_title("Age distribution, then and now")
axes[1].set_xlabel("age")
axes[1].legend(fontsize=8)

axes[2].hist(death_ages, bins=np.arange(0, 106, 5), color="#6a8f5f", rwidth=0.9)
axes[2].set_title("Age at death (30-year register)")
axes[2].set_xlabel("age")

fig.suptitle(f"Meridia, first nation: thirty simulated years (seed {SEED})", fontsize=10)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "renders" / f"meridia-nation-{SEED}-thirty-years.png"
fig.savefig(out, bbox_inches="tight")
print(out)
