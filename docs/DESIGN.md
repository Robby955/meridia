# Meridia design

A seed builds a country. The whole population is retained, imperfect sources are drawn
over it, and a submitted release is scored against the truth underneath. The question this
design answers is not how to generate data. It is how a threshold gets set without the
author choosing it, and how much a threshold set that way can actually tell apart.

Every number below names the file it is read from.

## 1. The world

One seed fixes terrain, settlements, persons, households, employers, and health histories,
and every layer is a deterministic function of it, asserted in
`tests/test_build_v4_worlds.py`. A world publishes a `participant/` tree carrying
`contract.json` with the schema, the public parameter ranges, and every scored threshold,
and retains a truth tree beside it that no participant file touches;
`tests/test_packet.py::test_participant_side_carries_no_truth_columns` asserts the second
half.

There are three families. Twelve development worlds ship with their truth and carry their
seeds in the builder, at `DEVELOPMENT_DESIGN` in `meridia/mechanisms.py` and
`DEVELOPMENT_SEEDS` in `scripts/build_v4_worlds.py`, one world per row of a twelve-run
Plackett-Burman layout. Six qualification worlds, `qual-0` through `qual-5` in
`qualification_worlds` in `bars/national-v14-standard/bars.json`, read their seeds from a
file outside the repository. Three graded worlds, `graded_world_count` in the same file,
have their seeds derived inside a sealed builder from a master secret that is never
committed. The continuation ensemble is 2,048 members, the committed size recorded at
`meridia/packet.py`, and each member is a predictive future rather than a draw around a
point.

## 2. What is scored

A submission is three files with fixed headers: `release.csv`, `projection.csv`, and
`reserve.csv`, specified in `docs/SUBMISSION_FORMAT.md` and enumerated in
`V4_SUBMISSION_COLUMNS` in `meridia/actuarial.py`. A fourth entry, a subdirectory, or a
symbolic link fails the file check before a number is read.

Above the file check sit the deterministic hard checks. Schema and additivity through
nation, state, and county are recomputed, the values are checked against `contract.json`,
and the reserve total is checked for feasibility. Above those
sit five composite blocks, the keys of `gates` in `bars/national-v14-standard/bars.json`:
`exposures_and_rates`, `release_accuracy`, `interval_quality`, `tail_calibration`, and
`reserve_skill`. All five are computed and reported on every run, whatever the profile
decides.

## 3. How a bar is set

Three reference lines run the evidence pass: A design-based, B Bayesian, and C with its
own elder exposure, linkage, and mortality choices, listed as `reference_lines` in
`bars.json` and implemented under `meridia/methods/`. On six qualification worlds that is
eighteen final reference reports, `reference_report_count`. Each line and world pair
contributes seventeen deterministic replicates, `replicates_per_reference_line_and_world`,
for 306 replicate reports, `replicate_report_count`, and 102 paired resamples per line,
`paired_resample_count`. The control battery adds 132 reports, `control_report_count`, and
the pass leaves 480 run receipts, `run_receipt_count`.

Each component bar is the empirical p99 of that component's own values on the line whose
p99 for it is largest, never the best line and never pooled across lines. In the receipt
that is `order_statistic_rank_per_reference_line` 101 of `sample_count_per_reference_line`
102, which is the `ceil(0.99 * 102)` order statistic, at a `target_false_fail_rate` of
0.01 per component per line. `empirical_p99_by_reference_line` on every calibrated
component names the line the published `calibrated_value` came from.

The conservative achieved marginal product over nine components and three graded worlds is
`achieved_marginal_rate_product` 0.862617, against `target_marginal_product` 0.762343,
which is `0.99 ** 27`. Two caveats travel in the receipt's own `caveats` field: the
marginal products assume independent failures, and six qualification worlds do not
establish a one percent rate on a new world. A bar is never tightened after a world is
seen; if measured difficulty comes in above the registered band, the escalation ladder is
climbed and the bars are re-qualified on fresh worlds.

## 4. Profiles

A gate profile names which frozen composites decide. It never adds a gate, never adds a
component, and never moves a ceiling. Three are registered in `GATE_PROFILES` in
`meridia/verify.py`: `standard`, `full`, and `lite`, and all three read one body of
evidence. A receipt frozen under one profile cannot decide under another.

`standard` ships. It is the only version-four set recording `"frozen": true`, with an
empty `blockers` list. `full` and `lite` are the same freeze run read under the other two
profiles; both record `"frozen": false`, and `bars/history/national-v14-full/PROVENANCE.md`
records that the full profile refused before evidence was compiled, so it carries no bars.
Their reports are kept under `bars/`.

## 5. What decides

Four blocks decide on seven components, listed in `gate_profile_selection`: exposures and
rates at a p95 relative error of 15.667516; release accuracy at 1.314919; interval quality
at a coverage deviation of 0.900000 and a mean interval score of 10.250382; tail
calibration at a pooled exceedance deviation of 0.192106, a q95 width relative error of
18.100609, and an es95 width relative error of 20.680355. Every value is the
`calibrated_value` of its component in `gates` in `bars.json`. No final reference exceeds
any published bar on any of the six qualification worlds: `reference_failures` and
`ungated_reference_failures` are both empty.

Of the four, tail calibration is the only one that separates a registered wrong method
from the reference on every qualification world, and it separates two of the twenty-two
registered controls, `mean_only_tail` and `predictive_tails`, listed under
`separating_controls_by_gate` in `control_support` and in `freeze_report.txt`. The other
three act as validity gates: they reject an empty or broken submission and do not tell the
registered methods apart at this world size. `control_support.full_separation` is `false`
and `control_support.complete_gate_count` is 0.

## 6. What is reported and not gated

`reserve_skill` is measured whole, published with no bar, and decides nothing; it is the
single entry in `reported_only_gates`. Its two components are carried with a null value
and a stated reason under `reported_only_components`.

At the compiled rate of 3769 per person-year, `rate_per_person_year` in
`reserve_calibration_accepted.json`, the worst regional shortfall probability reads
1.000000 on all eighteen final reference reports, so its own p99 of 1.0 sits at the top of
its attainable range and no submission could exceed it. Skill loss has a finite p99 at
6.007926, but ten of the eighteen `reference_witnesses` on that component read above 1.0,
meaning the reference allocations lose to the proportional baseline on those reports, so a
ceiling taken from the reference spread separates nothing. The rate is not free: every
candidate the qualification set can produce lies between 3602 and 4140, the eighteen
entries of `identification.candidates` in `reserve_calibration_accepted.json`, and the
accepted rate is the largest of the six marked `identified`. The reserve figures are still
filed and still checked for arithmetic. A block that cannot discriminate is reported as
such rather than quietly gated, and saying so is the point.

## Where each number is re-read

`bars/national-v14-standard/bars.json` is the receipt and carries every value quoted here.
`PROVENANCE.md` beside it gives the per-bar provenance, world by world and witness by
witness. `freeze_report.txt` gives the block roles and the control separation matrix.
`docs/history/` holds the dated sequence the freeze came out of.
