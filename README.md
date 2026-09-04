# Meridia

Meridia generates synthetic countries for testing statistical methods. One seed produces
terrain, rivers, settlements, and a population of individual people with households,
ages, incomes, employers, and health histories. The national population size is drawn
from the seed as well, so nations differ in scale the way countries do. Surveys and
administrative registers are then drawn from that population, and they carry the flaws
real collection produces. Households skip the survey, questions go unanswered, incomes
are misreported, ages get rounded, identifiers split and merge across sources. Because
the complete population is retained, the error of any estimate computed from those
sources is known exactly, and stated uncertainty can be checked against true coverage.
No judge models, no real data.

The intended use is population science on a world whose truth is known. The world is
observed imperfectly through surveys and archives, a method reconciles and imputes those
sources, estimates and projections are published with uncertainty, and retained truth
decides whether they held. Each stage of that chain is a scored task, and the world keeps
the stages mutually consistent.

![Seventy-two hours over the first nation](renders/meridia-72-hours.gif)

*Three simulated days. Weather is state, not decoration: a wind field advects moisture,
mountains wring rain out of it, the rain routes down the verified drainage tree and the
rivers swell afterward, and the cities light up at night. Every frame is drawn from
stored world state.*

## Install

Python 3.11 or newer, with NumPy, pandas, SciPy, Matplotlib, and Pillow. NumPy and pandas
carry the generator and the verifier, SciPy the weather layer, Matplotlib the figures, and
Pillow the animated renders. Tests need pytest.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib pillow pytest
```

Then confirm the tree is sound:

```bash
python -m pytest tests/
```

That collects 657 tests. They are the specification: every conservation identity, every
determinism claim, and every gate stated in this README is asserted by one of them. The
count is what `python -m pytest tests/ --collect-only -q` reports on this commit.

## Build a world

`scripts/build_v4_worlds.py` builds the committed world set. Worlds come in three
families, and the script writes the world size once so it cannot drift between them.

```bash
python scripts/build_v4_worlds.py --out worlds --family development --world-workers 4
```

That writes twelve development worlds under `worlds/development/dev-00` through
`dev-11`, one per row of the committed design in `meridia/mechanisms.py`. Development
worlds are the open instruments. They ship with their truth under `participant/truth/`
and a method may tune on them freely. Their seeds are committed in the script.

The other two families are `qualification` and `graded`. Qualification worlds are the six
worlds the bars are calibrated on, and graded worlds are the sealed set. Neither family
carries its seeds in the tree. Both read them from a JSON file outside the repository, at
the path `MERIDIA_QUALIFICATION_SEED_FILE` names or at
`~/.config/meridia/v4_qualification_seeds.json`. Every refusal on that path names the file
and the fault and carries no seed value.

Each world directory holds a `participant/` tree and a truth tree beside it. The
participant tree is everything a method is allowed to read: the observed sources, the
historical experience file, the geography map, and `contract.json`, which publishes the
schema, the reference ticks, the public parameter ranges, and every threshold that will
be scored. The truth tree is the generator's exact population, and the packet builder
refuses to write any participant file that carries a truth column.

`--world-workers` builds whole worlds at once. `--workers` divides a single world's
continuation ensemble between processes instead. Use one or the other. `--cache` points
at a directory of continuation ensembles so a rebuild that changes nothing upstream of
the ledger takes the futures off the shelf.

Every world is a deterministic function of its seed and the shared parameters. The same
seed gives the same world, byte for byte, and a test proves it.

## Run the verifier

The verifier is `meridia/verify.py`. It reads a packet directory, a submission
directory, and a bar set, and it returns a report. It reads nothing else. There is no
method inspection, no expected intermediate, and no required library on the submission
side.

```python
from pathlib import Path
import json

from meridia.verify import verify_submission, summary_table

bars = json.loads(Path("bars/national-v10/bars.json").read_text())
report = verify_submission(
    Path("worlds/development/dev-00"),
    Path("submission"),
    bars=bars,
)
print(summary_table(report))
```

`verify_submission` dispatches on the packet's schema. A version-four packet goes to
`verify_actuarial_submission`, which scores the three-file surface directly. A
version-four submission directory contains exactly `release.csv`, `projection.csv`, and
`reserve.csv`, with the exact headers given in `docs/SUBMISSION_FORMAT.md`. Extra
entries, subdirectories, and symbolic links fail the file check before any number is
read.

Schema, additivity through the geographic hierarchy, and reserve feasibility are
deterministic hard checks. The stochastic verdict is then five composite pass events:
`exposures_and_rates`, `release_accuracy`, `interval_quality`, `tail_calibration`, and
`reserve_skill`. All five are computed and reported on every run. Which of them decide
the verdict is set by the gate profile, `full` by default, and a component the profile
leaves out records its exceedance as an ungated failure rather than a reason. Passing a
version-four submission also requires a bar set that records a completed freeze;
`verify_submission` returns a failed report on any set that does not.

## Where the bars are

Frozen thresholds live in `bars/`. Each directory is one freeze. Every set carries
`bars.json` with the thresholds themselves, `PROVENANCE.md` naming the worlds and
witnesses behind them, and a freeze report saying what the freeze did and what it did
not do. The calibrated sets add `calibration_A.json` and `calibration_B.json` for the two
reference lines.

`bars/national-v7` is the version-three set and the last one that gated a submission. It
was frozen on nine qualification worlds under the hidden source regime, with every
registered control failing a named gate on all of them.

`bars/national-v8`, `bars/national-v9`, and `bars/national-v10` are the version-four
sets, and `national-v10` is the newest. All three record `"frozen": false` in
`bars.json`, so none of them can decide a version-four verdict. Read them as the record
of an open calibration, not as thresholds a submission must clear. `national-v10` names
its one remaining blocker in `bars.json`: replicate evidence is missing, and final
verifier reports cannot be bootstrapped or resampled as a replacement for it. It also
carries its own caveats in the same file. The marginal products in it assume independent
gate and world failures, which makes them arithmetic summaries rather than empirical pass
probabilities, and its false-fail rates are conditional on six qualification worlds and do
not establish a one-percent rate on new worlds. `bars/national-v9/freeze_report.txt` names
each bar that is written at its declared attainability cap rather than at a value a
witness reached.

A bar is set from the worse of two methodologically different reference lines, times a
margin declared before any world is scored, and never from the better line. A method that
is merely different from both has room by construction. The two lines are a design-based
line in `meridia/methods/design_based.py` and a Bayesian line in
`meridia/methods/bayesian.py`. They keep their own reconstruction of the county age cube
and their own uncertainty, and they pass those through one shared actuarial layer.
`docs/INDEPENDENCE.md` states plainly how much independence that leaves.

## What is in the repository

- **Terrain and rivers.** Seeded elevation with mountain chains and a coast, and water
  routing where every unit of runoff is accounted for exactly.
- **Population.** People placed where land is livable, with cities of realistic rank
  sizes. Cell counts add to the national total to the person. Ages fit household roles,
  spouses have similar ages, and cities are richer and more educated than the
  countryside.
- **Weather.** A wind field advects moisture, the sea recharges it, slopes facing the
  wind get the rain, and routed precipitation makes river discharge rise with a lag after
  storms. All four are tested.
- **Years passing.** Everyone ages, deaths follow an age-specific mortality law, babies
  join their mother's household, and young adults leave home. Every event lands in a
  vital-events register, and next year's population equals this year's plus births minus
  deaths, exactly.
- **Administrative geography.** States and counties as an exact partition around
  settlements and outposts. Every person sits in one county and one state, and counts add
  up through the hierarchy to the person.
- **Dwellings, businesses, and hospitals.** Housing stock, enterprises and
  establishments with payroll conserved to the cent, and facilities with patient
  encounters.
- **Observed sources.** Population, income, business, and health registers over the
  retained truth, with duplicates, stale records, split and merged identifiers, reporting
  delay, and no exact cross-source person key. Names are two synthetic tokens with a
  variant spelling for every family name.
- **Surveys.** A sample drawn the way real surveys are drawn, by region, then area, then
  household, with recorded selection probabilities. Richer households respond less,
  income questions go unanswered more when income is high, reported incomes are wrong,
  and ages round to fives.
- **The release contract and its scorer.** What a published estimate table must contain
  and how it is judged. See `docs/RELEASE_CONTRACT_V0.md`.
- **Reference methods.** Two full reconstruction and projection lines from the public
  literature, plus a registered battery of wrong methods that must fail.

![The first nation at night](renders/meridia-nation-20260831-population.png)

*The first nation, seed 20260831. Its 2,400,000 people shown as light: cities sit on
coasts and rivers, the mountain interior is empty.*

![States and counties](renders/meridia-nation-20260831-admin.png)

![Who lives there](renders/meridia-nation-20260831-demographics.png)

![Thirty years pass](renders/meridia-nation-20260831-thirty-years.png)

![Six nations from six seeds](renders/meridia-six-nations.png)

Worlds differ socially as well as physically. Each seed draws its society's parameters
from the declared ranges in `meridia/character.py`: how unequal income is, how wealth
concentrates in cities, how young or old the population runs, how dominant the largest
city is. Across a handful of seeds that yields nations from 280 thousand to 1.6 million
people, Gini coefficients from 0.40 to 0.55, and life expectancies from 69 to 78. The
ranges are public. A sealed world's specific draw is not, so a method must estimate the
society it is in from the data it is given.

## Figures

Each script under `scripts/` draws one figure from stored world state.
`render_first_nation.py` draws terrain and rivers, `render_first_nation_population.py`
the population, `render_admin.py` the states and counties,
`build_first_nation_microdata.py` the people and households, `render_72_hours.py` three
days of weather, `run_thirty_years.py` thirty years of the country,
`render_many_nations.py` six worlds, `render_gate_matrix.py` the gate matrix of a bar
set, and `world_characters.py` the character sheet of five societies.

## What a cheap world allows

Because a nation costs seconds, the unit of replication stops being a sample and becomes
the whole world. Studies that real data cannot support become routine.

- Run an estimator, or an automated research pipeline, across two hundred seeded nations
  and you have the sampling distribution of the whole workflow, from editing through
  inference, scored against exact truth in every world.
- Worlds branch. The same nation with and without a break year, a mortality spike or a
  migration wave, turns behaviour under a shift into a controlled experiment, because the
  counterfactual world is on disk.
- Whether a proposed experiment can detect what it claims is answered by generating
  worlds under both hypotheses and counting, before any real compute is spent.
- Automated runs on world tasks carry rewards computed against exact truth, with no judge
  models and no label noise.

## Sealing, and what stays closed

Evaluation stays clean by shape rather than by promise. A sealed world's seed is derived
from a master secret that never enters this repository. The committed manifests in
`seals/` record, per world index, only the digests of the generated layers and a
commitment over them. Generation runs headless, arrays are hashed and discarded, and the
function that generates a sealed world returns digests and nothing else.

Anyone holding a manifest can later confirm that a world used for grading is
byte-identical to the one sealed on registration day. Nobody without the master secret
can regenerate it, and nobody with it has looked. Graded seeds are read from a file
outside the repository, are never printed or logged, and never reach a participant file
or a build log. A world's whole configuration follows from its seed, which is why the
seed is the thing that is kept.

## Rules the code follows

- Same seed, same world, byte for byte.
- Plain Python with NumPy, pandas, and SciPy. No game engines, no third-party simulators,
  no real data.
- Totals are exact integers, and the conservation checks are tests, not intentions.
- The flaws in the observed data are planted through explicit mechanisms that are kept on
  file, so a method that models the mechanism is rewarded for it.
- Truth identifiers never appear in any participant-facing file, and observed identifiers
  are never derivable from truth identifiers.
- The parameter ranges of a world's character are public. A sealed world's draw is not.
- Worlds used for sealed evaluation are generated at registered seeds and never looked
  at, by anyone.

## Documents

- `docs/RELEASE_CONTRACT_V0.md`: what a published estimate table must contain and how it
  is scored.
- `docs/SUBMISSION_FORMAT.md`: the exact files, headers, and column rules a submission
  must satisfy.
- `docs/IDENTITY_AND_SCHEMA_V0.md`: the identity model and the schema of every observed
  source.
- `docs/INDEPENDENCE.md`: provenance, the public source of every method at its point of
  use, and how independent the two reference lines actually are.
- `docs/REVIEWER_GATES.md`: what a reviewer should check, and where each check is read
  from.
- `docs/V4_DECISIONS.md`: every design decision on the version-four surface, with the
  measurement behind it and the open items named.
- `docs/ROADMAP.md`: build order for what is not yet here.

`docs/FRONTIER_TRIALS.md`, `docs/PRE_SUBMISSION_AUDIT.md`, `docs/RUBRIC_MAP.md`, and
`docs/V4_OBLIGATIONS.md` are working records of the calibration rather than reader-facing
specifications. They are kept in the tree because the freeze cites them.

## Independence

Meridia is independent research by Robert Sneiderman. Every world is synthetic and
generated by the code in this repository from a seed. No real microdata, no
administrative records, and no survey responses were used. The methods are taken from the
public scientific literature and cited at the point of use in `docs/INDEPENDENCE.md`.

## License

Apache License 2.0. See `LICENSE`.
