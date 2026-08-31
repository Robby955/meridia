# Meridia

A synthetic world with exactly known truth, built for verifiable statistics.

Meridia generates nations — terrain, rivers, settlements, people, and the survey
products a statistical office would publish about them — from a single seed. Every
quantity in the world is retained: the true population of every cell, the true income of
every person, the true inclusion probability of every sampled household. Any estimate
produced from the world's survey products can therefore be checked against truth
exactly. The intended use is evaluation: statistical methodology, imputation and editing
systems, and research agents can be scored on whether their answers and their stated
uncertainty hold against a sealed ground truth they never see.

![The first nation: population as light](renders/meridia-nation-20260831-population.png)

*The first nation (seed 20260831): 2,400,000 people rendered as light over the terrain.
Cities glow along coasts and river valleys; the mountain interior stays dark. The
picture is drawn from the same arrays the tests verify, so a rendering is also a check.*

## Layers

**Terrain** — seeded spectral elevation with ridge chains and a continental gradient
toward a coast. Deterministic: the same seed yields byte-identical arrays.

**Hydrology** — priority-flood depression filling (Barnes et al. 2014), D8 flow
directions, and flow accumulation in topological order. Conservation is exact and
tested: the runoff delivered to outlets equals the interior land-cell count, unit for
unit.

**Census** — a habitability surface derived from the verified layers (elevation, slope,
distance to fresh water and coast), settlements seeded with rank-size weights, and an
integer population grid allocated by largest remainder. The grid sums to the declared
national total exactly; population is conserved the way runoff is.

**Microdata** — persons and households consistent with the census grid to the person.
Ages follow household roles with correlated spouse ages; education and income carry
urban gradients. The first nation holds 2,400,000 persons in 989,062 households.

![Demographic profile of the first nation](renders/meridia-nation-20260831-demographics.png)

**Survey instrument** — two-stage stratified sampling over the real geography with
recorded inclusion probabilities, unit nonresponse selective on income and urbanity,
item nonresponse with a not-at-random dial, and measurement error (income misreporting,
age heaping). The participant-facing file carries reported values only; the truth bundle
retains everything needed to verify any estimate.

## Design principles

- **Exact truth.** Totals are integers, and conservation laws are tested, not assumed.
- **Determinism.** Every layer is a pure function of its seed and inputs, seeded end to
  end; versions freeze with manifests and byte-identity regeneration; renders come from
  the truth arrays.
- **Own engine.** Python with NumPy and SciPy only; no third-party simulators; synthetic
  data and public methodology only.
- **Symmetric information.** Evaluation built on Meridia gives participants a
  development world with truth included; sealed evaluation worlds are generated at
  registered seeds and never inspected.
- **Real mechanisms.** Missingness, measurement error, and coverage error are planted
  with explicit, retained mechanisms, so methods that model the mechanism genuinely win.

## Roadmap

Climate and weather; demography over time (births, deaths, migration, emergent life
tables); administrative registers with linkage truth; epidemic and transport dynamics;
additional nations, including sealed ones.

## Running

```bash
python -m pytest tests/            # 26 tests: conservation, mechanisms, determinism
python scripts/render_first_nation.py
python scripts/render_first_nation_population.py
python scripts/build_first_nation_microdata.py
```

Requires Python 3.11+, NumPy, pandas, matplotlib.
