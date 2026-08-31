# Meridia

Meridia generates synthetic countries for testing statistical methods and AI research
agents. One seed produces terrain, rivers, settlements, and a population of individual
people with households, ages, and incomes; the national population size is drawn from
the seed as well, so nations differ in scale the way countries do. Surveys are then
drawn from that population with the defects real collection has: unit and item
nonresponse, measurement error, rounding. Because the complete population is retained,
the error of any estimate computed from a survey is known exactly, and stated
uncertainty can be checked against true coverage. No judge models, no real data.

The intended use is a full statistical production chain: the world is observed
imperfectly by a survey; the observations are edited and imputed; estimates are
published with uncertainty; retained truth decides whether the release held. Each stage
of that chain is a benchmark task; the world keeps them mutually consistent.

![Seventy-two hours over the first nation](renders/meridia-72-hours.gif)

*Three simulated days. Weather is state, not decoration: a wind field advects moisture,
mountains wring rain out of it, the rain routes down the verified drainage tree and the
rivers swell afterward, and the cities light up at night. Every frame is drawn from
stored world state.*

![The first nation at night](renders/meridia-nation-20260831-population.png)

*The first nation, seed 20260831. Its 2,400,000 people shown as light: cities sit on
coasts and rivers, the mountain interior is empty.*

## What exists so far

- **Terrain.** Seeded elevation with mountain chains and a coast.
- **Rivers.** Water routing where every unit of runoff is accounted for, exactly,
  with a test proving it.
- **Census.** People are placed where land is livable (low, flat, near water), with
  bigger and smaller cities. The map's cell counts add up to the national total exactly,
  to the person.
- **People.** 2,400,000 persons in 989,062 households. Ages fit household roles, spouses
  have similar ages, and cities are richer and more educated than the countryside.
- **Surveys.** A sample drawn the way real surveys are drawn (by region, then area, then
  household), with recorded selection probabilities. On top of the clean sample:
  richer households respond less, income questions go unanswered more when income is
  high, reported incomes are a bit wrong, and ages get rounded to fives. The survey file
  shows only what respondents reported; the truth stays on file for grading.

- **Weather.** A wind field advects moisture; the sea recharges it; slopes facing the
  wind get the rain (tested); routed precipitation makes river discharge rise with a
  lag after storms (tested). This is the layer that will drive storm-related data
  collection failures in survey tasks.
- **Years passing.** The country ages: everyone gets older, deaths follow an
  age-specific mortality curve, babies join their mother's household, and young adults
  leave home, mostly for the cities. Every event lands in a vital-events register, and
  next year's population equals this year's plus births minus deaths, exactly. Thirty
  years of 2.4 million people simulate in about ten seconds.

![Who lives there](renders/meridia-nation-20260831-demographics.png)

![Thirty years pass](renders/meridia-nation-20260831-thirty-years.png)

![Six nations from six seeds](renders/meridia-six-nations.png)

Worlds differ socially as well as physically. Each seed draws the society's parameters
from declared ranges: how unequal income is, how wealth concentrates in cities, how
young or old the population runs, how dominant the largest city is. Across a handful of
seeds that yields nations from 280 thousand to 1.6 million people, Gini coefficients
from 0.40 to 0.55, and life expectancies from 69 to 78. The ranges are public; a sealed
evaluation world's specific draw is not, so a method must estimate the society it is in
from the data.

## A research instrument, not just a testbed

Because a nation costs seconds, the unit of replication stops being a sample and becomes
the whole world. That allows studies real data can never support:

- **Sampling distributions of entire pipelines.** Run an estimator, or an AI research
  agent, across two hundred seeded nations and you get the true distribution of its
  whole workflow: editing, imputation, weighting, inference, end to end, scored against
  exact truth in every world.
- **Shift and shock experiments.** Worlds can branch: the same nation with and without a
  break year (a mortality spike, a migration wave). Robustness under distribution shift
  becomes a controlled experiment where the counterfactual actually exists.
- **Measured power for study designs.** Whether a proposed experiment can detect what it
  claims gets answered by generating worlds under both hypotheses and counting, before
  any real compute is spent.
- **Verifiable training data.** Agent runs on world tasks carry rewards computed against
  exact truth, with no judge models and no label noise.

Evaluation stays clean through sealing: development worlds are open instruments; worlds
used for grading are generated at registered seeds and never looked at, by anyone.

## Rules the code follows

- Same seed, same world, byte for byte.
- Plain Python with NumPy. No game engines, no third-party simulators, no real data.
- Totals are exact integers, and the conservation checks are tests, not intentions.
- The flaws in the survey data are planted through explicit mechanisms that are kept on
  file, so a method that models the mechanism genuinely wins.
- Worlds used for sealed evaluation are generated at registered seeds and never looked
  at, by anyone.

## Run it

```bash
python -m pytest tests/                        # 32 tests
python scripts/render_first_nation.py          # terrain and rivers map
python scripts/render_first_nation_population.py   # population map
python scripts/build_first_nation_microdata.py     # people, households, demographics
python scripts/render_72_hours.py              # three days of weather over the nation
python scripts/run_thirty_years.py             # thirty years of the country
python scripts/render_many_nations.py          # six worlds in about two seconds
python scripts/world_characters.py             # the character sheet of five societies
```

Python 3.11+, NumPy, pandas, matplotlib, Pillow.

## Planned

The full build order is in [docs/ROADMAP.md](docs/ROADMAP.md).

Administrative registers with known linkage truth (the
vital-events register is the start). Epidemics and commuting. More nations, including
sealed ones.
