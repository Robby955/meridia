# Meridia

Meridia generates synthetic countries for testing statistical methods. One seed produces
terrain, rivers, settlements, and a population of individual people with households, ages,
incomes, employers, and health histories. Surveys and registers drawn from that population
carry the flaws real collection produces, from nonresponse and heaped ages to identifiers
that split and merge across sources. Because the complete population is retained, the
error of any estimate is known exactly, and stated uncertainty can be checked against true
coverage. No judge models, no real data.

![Seventy-two hours over the first nation](renders/meridia-72-hours.gif)

*Three simulated days: moisture advects, mountains wring rain out of it, and rivers swell
after it routes down the verified drainage tree.*

## Install

Python 3.11 or newer with NumPy, pandas, SciPy, Matplotlib, Pillow, and pytest.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy matplotlib pillow pytest
```

Confirm the tree is sound.

```bash
python -m pytest tests/
```

That collects 679 tests, the count `python -m pytest tests/ --collect-only -q` reports
here. They are the specification: every conservation identity, determinism claim, and
threshold stated here is asserted by one of them.

## Build a world

```bash
python scripts/build_v4_worlds.py --out worlds --family development --world-workers 4
```

That writes twelve development worlds under `worlds/development/dev-00` through `dev-11`,
one per row of the committed design in `meridia/mechanisms.py`. Development worlds are the
open instruments: they ship with their truth, a method may tune on them freely, and their
seeds are in the script. The other families are `qualification`, the six worlds the bars
are calibrated on, and `graded`, the sealed set.

Each world holds a `participant/` tree and a truth tree beside it. The participant tree is
all a method may read, including `contract.json`, which publishes the schema, the public
parameter ranges, and every threshold scored. No participant file carries a truth column,
and every world is a deterministic function of its seed.

## Run the verifier

`meridia/verify.py` reads a packet directory, a submission directory, and a bar set, and
returns a report. It inspects no method and requires no library of a submission.

```python
from pathlib import Path
import json

from meridia.verify import verify_submission, summary_table

bars = json.loads(Path("bars/national-v14-standard/bars.json").read_text())
report = verify_submission(
    Path("worlds/development/dev-00"),
    Path("submission"),
    bars=bars,
    gate_profile="standard",
)
print(summary_table(report))
```

A submission directory holds exactly three regular files: `release.csv` for the
population, exposure, and rate release, `projection.csv` for the horizon values, and
`reserve.csv` for the regional liability distribution and allocation. Headers are
fixed in `docs/SUBMISSION_FORMAT.md`; any extra entry, subdirectory, or symbolic link
fails the file check before a number is read.

Schema, additivity through the geographic hierarchy, and reserve feasibility are
deterministic hard checks. The stochastic verdict is five composite pass events:
`exposures_and_rates`, `release_accuracy`, `interval_quality`, `tail_calibration`, and
`reserve_skill`. All five are computed and reported on every run, and the gate profile
names which decide. A receipt frozen under one profile decides under no other.

## Where the bars are

Frozen thresholds live in `bars/`, one directory per freeze. `bars/national-v14-standard`
is the shipping set and the only version-four set recording `"frozen": true`. It carries
`bars.json`, `PROVENANCE.md` naming the worlds and witnesses behind every bar, and
`freeze_report.txt`. Under the `standard` profile four blocks decide, being exposures and
rates, release accuracy, interval quality, and tail calibration, on seven components in
all. The reserve block is measured and reported whole and decides nothing: at the compiled
rate its shortfall probability sits at the top of its attainable range, and its skill loss
is no ceiling to hold a method to.

Each bar is the ninety-ninth percentile of the worst of three reference lines over its 102
replicates, never the best line, at a target false-fail rate of one percent per component.
The lines are design-based, Bayesian, and a third with its own elder exposure, linkage,
and mortality choices, all under `meridia/methods/`; `docs/INDEPENDENCE.md` states how much
independence that leaves. `bars.json` carries the freeze's own caveats, including that six
qualification worlds do not establish a one percent rate on new worlds. Unfrozen `full`
and `lite` sets read the same evidence under the other profiles.

## Sealing

No seed of a graded world is in this repository. Seeds are read from a file outside the
tree and derived inside the sealed builder through a keyed digest of a master secret that
is never committed, so no seed reaches a tracked file, a participant file, or a build log.
The manifests in `seals/` record, per world index, only the digests of the generated
layers and a commitment over them; generation runs headless and returns digests and
nothing else. Anyone holding a manifest can confirm a graded world is byte-identical to
the one sealed on registration day; nobody without the secret can regenerate it.

![The first nation at night](renders/meridia-nation-20260831-population.png)

*The first nation, seed 20260831, its 2,400,000 people shown as light. Each seed draws its
society from the public ranges in `meridia/character.py`; a sealed world's draw is not
public.*

## Documents

- `docs/SUBMISSION_FORMAT.md`: files, headers, column rules.
- `docs/RELEASE_CONTRACT_V0.md`: what a release must contain.
- `docs/IDENTITY_AND_SCHEMA_V0.md`: identities and schemas.
- `docs/INDEPENDENCE.md`: every method's public source, at its point of use.
- `docs/REVIEWER_GATES.md`: what a reviewer checks, and where.
- `docs/V4_DECISIONS.md`: every version-four decision and its measurement.
- `docs/ROADMAP.md`: what is not here yet.

## Independence

Meridia is independent research by Robert Sneiderman. Every world is synthetic, generated
here from a seed. No real microdata, administrative records, or survey responses were
used.

## License

Apache License 2.0. See `LICENSE`.
