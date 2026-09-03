# Version four: decisions on record

Every entry under the protocol's "Unresolved decisions" is either resolved here, with the
choice and the reason, or listed as still open with what it waits on. Nothing in this file
changes the benchmark's design; it records which of the protocol's stated options was
taken and why.

Scope of this file at the time of writing: the generator lane, meaning the mechanism
families, the development design, the identifiers, the local measurement scales, the
health selection rule, the historical experience file, and the health anchor. The
verifier, scoring, and reserve lanes record their own decisions.

## Resolved

### Historical experience file: annual, five years, long format

Fields, one row per (year, actuarial age band, sex, state):

    year, age_band, sex, state, exposure, deaths, qualifying_events, net_migration

`exposure` is person-years, `deaths` and `qualifying_events` are counts, `net_migration`
is arrivals minus departures across state boundaries among persons alive at both ends of
the year. `year` runs 1 to 5.

The series stops twelve months before the revised snapshot, the way published
demographic experience always lags collection, and `contract.json` states the lag. The
lag is not decoration. Without it the most recent year's exposure is a near-exact
contemporaneous population count by state and sex, and the scored state-level counts
would come free with the anchor that exists to identify the mortality trend. The ledger
is therefore 72 months before the revised snapshot: sixty for the five published years
and twelve for the lag.

Annual rather than monthly, because the protocol names annual first and because a monthly
file at age band by sex by state produces thin cells whose noise carries the ledger's own
month-level reporting pattern. Long format rather than one row per cell, because the
participant file guard rejects a wide table whose column names would have to encode a
band. Exposure comes from the same person-month reading pass the actuarial truth uses, so
a published rate and its denominator cannot disagree.

State rather than county, because the protocol's section 3 names age by sex by state.

### Health anchor: the survey item

A binary `recent_hospitalization` item on both survey snapshots, meaning an admission in
the twelve months before that snapshot's reference tick, reported with a declared
sensitivity of 0.82 and a declared specificity of 0.93. Both constants are in
`contract.json` under `health_anchor`.

The survey item rather than an audited inclusion sample, because the health source's
inclusion indicator is per encounter, not per person, so an audit sample would have to be
constructed rather than sliced, for no gain. The survey sample is drawn without reference
to health-source inclusion, so the item is a genuinely external anchor rather than a
restatement of the register whose selection it exists to identify. The true indicator is
read off the ledger, never off the health source.

### Which file carries the exposure and rate fields: the release table

As protocol section 4 item 1 states. The generator ships the experience file that
identifies the five-year trend; the release table carries the participant's own exposure
and rate estimates. Nothing in the generator lane depends on the choice, and moving it
would need a fifth submitted file, which the verifier's file-set gate refuses.

### Number of development worlds: twelve, on a committed Plackett-Burman design

Twelve worlds, one per row of `mechanisms.DEVELOPMENT_DESIGN`, at the top of the
protocol's stated range of eight to twelve. The design is the twelve-run Plackett-Burman
layout, first six columns, one column per axis of the regime family.

The design's measured properties, from `tests/test_mechanisms.py`:

- each column sums to zero, so every main effect is balanced,
- the six columns are mutually orthogonal, so main effects are estimable independently,
- no two-factor interaction column is fully aliased with any main effect; the largest
  partial alias is one third.

A regular resolution-IV fractional factorial in six factors at two levels needs sixteen
runs, above the protocol's ceiling. The twelve-run Plackett-Burman design is the
layout that fits the stated range while keeping every main effect clean.

### Hidden-world rule: drawn per world, not written in the module

A hidden world draws its level pattern from `HIDDEN_LEVEL_PATTERNS`, the fifty-two sign
patterns over the six axes that no row of the twelve-run design takes, and draws which two
of its intensities leave the development band. Both draws are keyed on the world's own
seed.

The first version-four pass wrote the pattern and the two axes as module constants, so all
nine hidden worlds landed on the same corner and moved the same pair of intensities.
Anyone holding the generator read the hidden configuration off a line of source instead of
estimating it, and proof obligation 7, that difficulty is stable across independent worlds,
was tested inside one corner rather than across the family. Measured after the change, over
the six qualification seeds and the three graded seeds, the level patterns and the outside
pairs are given in the identifiability report; they are no longer one value.

The two intensities that leave the band stay inside the public envelope, and the draw
checks that: `draw_mechanism_design` refuses a pattern that repeats a design row, and the
sealing script refuses a packet whose pattern is one the design spends.

### Mechanism intensities are continuous, and no rate is a world constant

All nineteen source rates are now a per-world continuous draw from a published
band and, on top of that, a per-record probability from a published family. Version three
froze fifteen of the nineteen across every world and both regimes, which let a rate be
measured once on development and carried unchanged to the hidden world.

### Covariate definitions

All four are rank statistics, which is what makes them recoverable from a noisy or
differently scaled estimate of the same underlying quantity:

- `urban_c`: rank of the county's persons per land cell, scaled to [0, 1]. The
  participant builds it from register persons per county over `land_cells`, now shipped in
  `geography.csv`. Measured rank agreement with the generator's value on three worlds:
  0.953, 0.983, 0.957.
- `econ_c`: rank of the county's establishment payroll per resident adult, payroll from
  the business source and adults from the population source. It is only partially
  recovered, measured at 0.25, 0.80 and 0.54 on three worlds, because the register that
  reports it is itself thinned by the completeness gradient `econ_c` drives. That is the
  protocol's "partially learnable" case, and the mechanism it carries has a second,
  direct handle: the register-to-benchmark gap per state measures the completeness
  gradient without going through the covariate at all.
- `elder_c`: rank of the county's share of persons 65 and over.
- `band_r`: the record's quintile of its own source's money distribution. A quantile rank
  is invariant to the unknown scale, which is what keeps the local money coefficients
  identifiable while the level is not.

The definitions are written into `contract.json` under `mechanisms.covariates`.

### Latent frailty and the person-table schema

Latent frailty is a per-person column on the microdata, carried through the ledger's
person state and its birth events so replay conserves it exactly. It drives mortality,
hospital incidence, and health-source inclusion, and the survey anchor is its only
observable handle.

Consequence, verified: the sealed `person_table` digest changes, and only that digest.
Running `verify_sealed_world(0, seals/meridia-reconstruction-v3.json)` now returns False,
with `elevation`, `flow_direction`, `population_grid` and `household_cells` unchanged and
`person_table` changed. Version four needs its own seal manifest, and the person-table
schema is frozen as of this change so that registration happens once.

### Newborn frailty

`log f_child = 0.35 * log f_mother + 0.42 * z - 0.5 * 0.42^2`. Health burden then clusters
by household instead of being white noise, which is what makes baseline health burden
observable structure rather than a nuisance term.

## Measured effect of the four changes

Three development worlds at 96 by 128, 120,000 persons, 18 counties, seeds 3, 7 and 11.
Version three's value is in brackets.

- Cross-vintage record key overlap: 0, 0, 0 rows [every row shared a key, so a
  preliminary-to-revised join was exact].
- Entity key persistence: 0.883, 0.851, 0.791, with 237, 329 and 324 surviving keys now
  pointing at a different entity [1.000, none reissued].
- County household growth against `urban_c`: correlation +0.706, +0.693, +0.781
  [growth was one national scalar, so county growth carried no signal and its dispersion
  measured 0.021 to 0.026 around the national number].
- Register money unit, median realized scale by county: spread 1.26, 1.26, and by income
  band 1.01 to 1.56 across worlds [one world-global float, so a single national ratio of
  register earnings to survey income recovered it exactly].
- Health-source inclusion, share included at high frailty minus low frailty: +0.023,
  +0.071, +0.019 on development and +0.342 on the hidden world, whose intensity is
  outside the development band [inclusion did not read morbidity at all].
- County death rate, dispersion over mean: 0.117, 0.106, 0.203, correlated with
  `urban_c` at -0.234, -0.484, -0.682 [mortality was a national Gompertz level].
- All twenty `SourceParams` fields differ across worlds [fifteen of
  nineteen were byte-identical in every world and both regimes].

## Cost

A full-size world at 288 by 384 with the default parameters builds in 580 seconds:
3,188,240 persons, 16,606,958 events, 34 counties, 132 ledger months. The added months
are the dominant term. The experience file's year-boundary replays sit early in the
ledger, so together they cost about 1.4 full replays.

The forecast packet is a separate clean surface and is unchanged: `forecast.build_*`
calls the ledger without a mechanism layer, which gives it a single neutral county.

## Still open

These need calibration worlds that do not exist yet, or belong to the verifier and reserve
lanes. They are listed so nothing is silently decided by default.

- Ensemble size M. The substream rule is settled and built:
  `build_event_history(..., continuation_member=m, branch_month=b)` runs the ledger's own
  stream through month `b` and the member's own stream after it, keyed
  `SeedSequence([seed, 0xC047, m, month])`, which is the tag
  `actuarial.continuation_member_key` uses. No member seed is arithmetic on the root
  seed. Measured on a 40,000-person world over twelve months branching at month six: the
  base ledger has 18,117 events and three members have 17,905, 18,247 and 18,244, with
  different survivor sets and a byte-identical prefix.

  M itself is open, and it has a hard cost wall. A member built this way re-runs the
  whole ledger, and a full-size world's ledger takes 580 seconds. Even a member that ran
  only the sixty-month suffix would cost about 260 seconds at full scale, so M = 2,048 is
  not reachable without either a much smaller grading world, a cheaper continuation than
  a full ledger replay, or extracting the monthly loop so a member starts from a replayed
  branch state and pays only the suffix. That extraction is the one structural change the
  ensemble still needs from the ledger, and it is deliberately not done here.
- Whether ES95 is required in version four or deferred.
- gamma, w_r, and the rounding rule for R. R is written into the participant contract by
  the packet builder once the reserve lane fixes it; the generator does not set it.
- The pooled and worst-region tau values, the score ceilings, and the frozen stabilizers
  c_x. Protocol section 12 requires these to come from generator-only calibration worlds
  before the hidden world exists.
- The frozen practical baseline allocation A_B.
- Three packet artifacts the verifier now reads and the packet builder does not yet
  write. `meridia/verify.py` reads `contract["reserve"]`, `retained/rate_truth_horizon.csv`
  and `retained/continuation_liabilities.npz`, and `methods/actuarial_reference.py` reads
  `contract["reserve"]["total"]`. `tests/test_actuarial.py` writes all three by hand into a
  temporary packet, so the shapes are settled:

  - `contract["reserve"]`: `obligation` (an `ObligationContract.as_public()` payload),
    `total` (protocol section 9 R), and optionally `gamma`, `regions` and `weights`.
  - `rate_truth_horizon.csv`: columns estimand, level, unit, sex, age_band, value, which
    is `exposure_and_rate_truth` over the sixty-month horizon.
  - `continuation_liabilities.npz`: `liability` of shape (members, regions) and
    `realized_member`.

  The first two are cheap. The third is the ensemble, and it waits on M and on the
  monthly-loop extraction described above. Wiring all three into `build_packet` is one
  step, and doing it before M is settled would freeze an ensemble size by accident.

## Open consequences of the generator changes

- The ledger is seventy-two months before the revised snapshot rather than twenty-four,
  because a five-year experience file with a twelve-month publication lag cannot be
  written from a twenty-four-month ledger. Every frozen bar under `bars/` and both seal
  manifests under `seals/` are derived from the shorter ledger and must be re-derived.
- `PacketParams` gained `design_cell`. The freeze script builds development packets
  without one, so it currently draws a design row from the seed rather than covering the
  twelve rows. Covering the design is a freeze-lane change.
- `geography.csv` gained a `land_cells` column and the survey files gained
  `recent_hospitalization`. Readers that select columns by name are unaffected.
- The contract tag moved from `meridia.packet.v0` to `meridia.packet.v4`, because the
  participant surface gained `experience_history.csv` and the `mechanisms`, `obligation`,
  `health_anchor` and `experience_history` blocks. Nothing in the tree reads the tag
  today, so anything that starts reading it will see the right value.
## Reference methods and controls (protocol sections 11 and 14)

Owner of this section: the methods lane, `meridia/methods/actuarial_reference.py` and the
version-four half of `meridia/methods/controls.py`. The two strong lines keep their own
reconstruction and their own uncertainty draws; the actuarial chain they share sits in
one module so a schema change is read in one place and the two lines cannot drift.

### Resolved here, with the reason

- **One shared actuarial layer, not one per line.** `actuarial_reference.actuarial_layer`
  takes population draws of shape (paths, counties, ages, sexes) and returns the rate
  block, the liability paths, and the reserve. The design-based line passes its bootstrap
  replicates of the county age cube and the Bayesian line its posterior draws, so the two
  submissions differ exactly where the two reconstructions differ. A copy of the actuarial
  arithmetic in each file would have let the comparison drift between releases.
- **Frozen vocabulary is imported, never restated.** Estimand names, band labels, the
  reserve columns and the obligation come from `meridia/actuarial.py`. A reference
  submission and the verifier cannot disagree about what a column is called.
- **The published rate block is the horizon window.** Truth is written to
  `rate_truth_horizon.csv` and read from the same person-month pass that prices the
  liabilities, so the released exposure, mortality and incidence are the projected
  sixty-month quantities, not current-period ones. The reference therefore emits them from
  the same simulation that produces the liability paths: one object, one set of
  assumptions, and no way for a submitted rate to disagree with the submitted tail. The
  experience file is what identifies the level and the drift that simulation starts from,
  which is the role protocol section 3 gives it.
- **Exposure additivity is built in, not repaired.** State exposure is the sum of the
  published county exposures and a broad band is the sum of the actuarial bands inside it,
  both on the point estimates, so `check_rate_additivity` has nothing to find. Rates are
  ratios of the summed numerator and the summed denominator at the level they are
  published at, never sums of rates.
- **Linkage carries its uncertainty downstream.** Version four keeps no record identifier
  across register vintages, so the two vintages are linked by a Fellegi and Sunter
  mixture fitted by expectation maximisation on blocked candidate pairs, reduced one to
  one by descending posterior. The disappearance count that identifies mortality is
  averaged over six Bernoulli imputations of the link set, and the spread across
  imputations is reported. Imputation rather than a threshold, because a threshold turns a
  probability into a decision and loses exactly the uncertainty the tails need.
- **The health anchor is used as a ratio of sums.** Inclusion probability of an admitted
  person is the archive's count over the anchor's expected count, pooled as a ratio of
  sums and shrunk per cell by its expected count. A mean of cell ratios is biased upward,
  because the denominator is small and random, and that bias runs straight into the
  incidence rates: on the development bench it read the pooled inclusion as 0.712 when
  the ratio of sums read 0.618 against a generator value of 0.633.
- **The drift is weighted by counts, not exposure.** The sampling variance of a log rate
  is one over the count, so weighting the five annual points by exposure hands the drift
  to the young bands, which have the most exposure and the least information. On clean
  five-year files the count-weighted estimator recovers drifts of -0.020, 0.000 and +0.030
  as -0.02000, -0.00041 and +0.02965.
- **All five experience years set the level, not the last one.** Each year's exposure is
  put on the last year's level through the estimated drift and the ratio is pooled, so the
  level carries five years of information instead of one.
- **Two thousand and forty-eight simulation paths**, matching the ensemble size the
  actuarial lane froze. The Monte Carlo error of an empirical 95th percentile is about
  0.16 standard deviations at 180 paths and about 0.05 at 2,048; a submitted quantile that
  noisy scores worse than a normal approximation to its own distribution, which would make
  ablation 6 pass for the wrong reason.
- **Quantiles are calibrated to the public reserve total.** R is published and the
  submission must satisfy sum_r A_r = R with A_r at or above q_hat_r, so a method whose own
  regional quantiles sum above R is making a claim the public total contradicts. The
  reference pulls every quantile toward its own mean by the one factor that closes the gap,
  which keeps the ordering across regions and leaves the means untouched. A proportional
  haircut is the fallback only when the means alone exceed R.
- **The reserve allocation is solved, not heuristic.** J is separable and convex, so at the
  optimum every region off its floor shares one marginal cost w_r P(L_r > A_r) = nu, and
  one bisection on nu solves the constrained problem on the empirical distribution. Checked
  against a four-thousand-point grid search on two regions: J agrees to six decimals.
- **Version-three fallback is automatic.** `MethodParams.actuarial` defaults to "auto":
  the version-four submission is written when the packet carries the obligation, the
  experience file and the reserve total, and the version-three submission is written
  unchanged when it does not. "on" raises with the missing keys named. This keeps the
  version-three tests meaningful while both surfaces exist.
- **The control battery is a switch table, not a fork.** `ACTUARIAL_SWITCHES` maps each of
  protocol section 11's ablations to one `LayerParams` field, so each control removes
  exactly one step of the strong line. A control that clears its gate then says the gate is
  loose, not that the control was subtle. `version_three_recipe` is the exception and rebuilds
  the release and projection tables as well, since proof obligation 2 is about a whole
  recipe rather than one step.
- **Controls are listed in a second tuple.** `CONTROLS` stays the version-three battery so
  the existing parametrised test keeps passing on a version-three packet;
  `ACTUARIAL_CONTROLS` holds the eight new ones and `ALL_CONTROLS` is the union the
  version-four freeze iterates.

### Left open, and why

- **The reserve total R is not published yet.** `meridia/packet.py` writes no
  `contract["reserve"]` block, so no method can satisfy sum_r A_r = R. The reader names the
  missing key rather than inventing a value. Needed keys: `total`, and optionally `gamma`,
  `weights` and a region map.
- **The rate window is inferred, not declared.** The reference reads the release rate block
  as the sixty-month horizon window because the truth file is named `rate_truth_horizon`
  and the truth pass starts at the revised tick. If the intended window is the historical
  one instead, the reference changes in one place, but the two lanes must agree in the
  contract rather than by inference.
- **Whether the reserve decision gate has attainable value.** Measured on the development
  bench: the sealed-information oracle beats the proportional baseline by 0.2 percent of
  J, and raising the ensemble dispersion does not move it. The gap comes from regional
  heterogeneity in tail shape that a size measure does not predict, not from dispersion:
  three regions of equal size and equal tail give a 0.0 percent gain, the same three with
  tail widths 0.02, 0.05 and 0.12 give 1.9 percent, and adding skew to two of six regions
  gives 7.1 percent. Proof obligation 4 is a question for the generator and actuarial
  lanes, not for the reference.
- **Which ablations separate is not yet settled.** On the bench, ignoring health
  selection, deterministic linkage with archive-only rates, and a mean-only tail each fail
  by a wide margin. A normal-approximation tail does not separate at all while the regional
  liability is close to normal, a padded tail separates only weakly because the public
  total bounds how far padding can travel, and a proportional reserve equals the optimiser
  whenever the submitted quantiles already sum to R. All three need qualification worlds to
  settle, and two of the three are properties of the ensemble rather than of the control.

## Integration pass: the surface the three lanes now share

Owner of this section: the integration pass that wired the generator, the actuarial
module and the methods lane into one surface and ran the freeze. It records the decisions
that no single lane could take because each of them changed what another lane reads.

### The three artifacts, and what settled them

`build_packet` now writes `contract["reserve"]`, `retained/rate_truth_horizon.csv` and
`retained/continuation_liabilities.npz`, in the shapes the verifier already read. Wiring
them needed three things settled first: how a continuation member is built, how many
there are, and what the reserve block publishes.

### Ensemble size M = 2048, and a member that pays only for its own future

The protocol names 2,048 or 4,096; 2,048 is taken, the lower of the two.

It became affordable by extracting the ledger's monthly loop. `_run_ledger_months` runs
one span of months over four carried quantities: the entity state, the event records, the
order counter, and each household's last move. `build_event_history(..., capture_month=m)`
keeps a copy of those four at the branch, and `continuation_events(branch, member,
months)` resumes from that copy, so a member costs the horizon window rather than the
whole ledger.

The extraction is exact, and `tests/test_events.py` proves it rather than asserting it: a
member resumed from the branch equals, event for event and identifier for identifier, the
member that re-ran every month before the branch. Measured on the version-three
continuation path: members 0, 1 and 7 over a twelve-month ledger branching at month six
agree on 2,464, 2,542 and 2,508 suffix rows with no differing column.

Cost, measured on a qualification world of 60,000 persons, 18 counties, 132 ledger months:
the world builds in 6 seconds, one member costs 0.96 seconds, and a 2,048-member ensemble
costs 310 seconds across fourteen processes. `workers` divides members between processes
and changes nothing: a packet built serially and one built on six processes have identical
manifests, including the ensemble digest
`20c92935a5d3d6b1bc1262e7a4307522c5957be47fa215baef839d6c9e1902a8`.

Member zero is the ledger's own future, which is the future the horizon truth tables are
read from, so the designated realized path and the tail truth are one world.

### The continuation carries systematic risk, and the shock family is published

This is the change without which nothing else in sections 6 to 9 works.

Version three's ensemble members shared the world's shock schedule, so they differed only
by demographic noise. Measured on a qualification world under the old arrangement: the
regional tail, q95 minus the mean over the mean, ran 0.011 to 0.020, and the strong
reference's regional liability estimates were off by 3 to 9 percent. The sealed exceedance
was then 0 or 1 in every region, the pooled criterion read 0.4973 against a tau of 0.02,
and no method could have done better: the gate was asking for a liability level known to
better than one percent.

A continuation now redraws every year after the branch from the declared family at a
published annual rate of 0.20, keyed `SeedSequence([seed, CONTINUATION_DOMAIN, member,
SHOCK_SUBSTREAM])`, and keeps the world's realized years before the branch. The family
itself is published in `contract.json` under `shock_family`, and the five-year experience
file carries roughly one realization of it, which is what makes the rate estimable rather
than assumed.

`mortality_spike` gained an `admission_multiplier` on the same draw as its mortality
multiplier: an epidemic year is one event, and a schedule that raised deaths while leaving
hospital admissions alone would put the liability's systematic risk outside the health
source, where no anchor could reach it.

Measured after the change, same world: the regional tail widens to 0.056 to 0.142 and the
reference's pooled exceedance deviation falls from 0.4973 to 0.1048, with regional
exceedances of 0.00, 0.27, 0.18, 0.01, 0.01 and 0.20 instead of zeros and ones.

### The obligation weights, and why they moved

`ObligationContract` now reads b = 150 a month, c = 15,000 at the first qualifying event,
d = 7,500 at death, against 120, 4,000 and 2,000. All three terms of section 5 stay in
force; what changed is their share: 23 percent annuity, 65 percent first-event cost, 11
percent death benefit.

The reason is the same one that put systematic risk in the continuation. A liability that
is mostly a monthly annuity over a large elderly stock is very nearly deterministic,
because the stock is a smooth function of a population the ensemble barely moves. The
first-event cost and the death benefit are counts, and counts carry the shock. Measured
across mixes on one qualification world at 256 continuations, regional tail width: 0.032
to 0.089 at the old weights, 0.056 to 0.142 at the new ones.

### Regions are the states, and the shortfall weights are published and not all one

`w_r` is the protocol's own unresolved item, and its stated default is one. The default
was measured and it does not work.

With unit weights the reserve decision carries nothing. On a qualification world at 1,024
continuations, the sealed oracle beat the frozen proportional baseline by 0.10 percent of
J at state regions and 0.55 percent at county regions, and an oracle fitted on half the
ensemble and paid on the other half did 3.7 and 2.5 percent worse than the baseline: the
whole apparent gain was fitted to Monte Carlo noise. Proof obligation 4 fails on the
default, so the default is not taken.

The weights are a published ladder over the regions ranked by their share of persons 85
and over in the revised population source, geometric from 0.5 to 2.0. Rank of a share
rather than of a count, so the ladder is not a restatement of region size, which is what
would make a size-proportional reserve optimal by construction. The register it is read
from is a participant file, so a method reproduces the ladder exactly, and the realized
numbers are published in the contract regardless. An uncovered obligation costing more
where the very old are concentrated is the reserving reason for weighting at all.

Measured with the ladder in place, same world and ensemble: the held-out oracle gain is
6.8 percent at state regions and 8.2 percent at county regions, against 11.5 and 11.6
percent for the sealed oracle. Regions stay the states, which is the level
`regions_from_admin` already returns and the level the experience file is published at.

### The baseline is public and size-proportional, and the oracle is free of the floors

A_B splits R in proportion to each region's share of persons at or above the eligibility
age in the revised population source, published in the contract as
`reserve.baseline_share` and reproducible by any participant. A* spends the same R under
non-negativity alone.

This replaces the first pass's pair, in which A_B spread the slack above the submitted
quantiles in proportion to those quantiles and A* stood on the same submitted quantiles as
floors. That pair was chosen to keep the oracle from measuring the distance between the
floors and the truth, which the tail gates already score. It had a worse defect: R sits
only a fraction of a percent above the sum of the true quantiles, so the slack the two
allocations disagreed about was a fraction of a percent of R, and the decision gate carried
almost no information. It was also a baseline that read the submission, which a frozen
practical baseline must not do.

Holding a reserve in proportion to how many people it covers is what a reserving office
does with no regional tail model, which is what the protocol asks a practical baseline to
be, and it is the version-three proportional heuristic in its new place. Under it, skill
zero means the submission did no better than that rule and skill one means perfect
information, so the `proportional_reserve` ablation lands at zero by construction and a
good forecast with a poor allocation is refused.

### gamma, the rounding rule, and the preregistered scale

gamma is 0.25, the midpoint of the protocol's 0.20 to 0.30. R is rounded up to the next
1,000, a public unit, and the rounding is verified to add less than one unit. ES95 is
required in version four rather than deferred: it is already produced by the same pass
that produces the quantile, the reserve total reads it, and deferring it would leave R
defined by a quantity the submission never files.

The preregistered regional scale for the quantile score and the shortfall error is the
ensemble's own 0.95 quantile. It is applied by the verifier and is deliberately not
published: a per-region sealed quantity in the participant contract would hand back the
truth the tail gates exist to ask for.

### The world set, and which worlds may set a bar

Committed in `scripts/build_v4_worlds.py`, one world size for all three families: 96 by
128 cells, 60,000 persons, 18 counties, 6 states, 72 observed months, a 60-month horizon,
2,048 continuations.

- development, seeds 1101 to 1112, one per row of the committed twelve-run design, under
  the development source regime. A method may tune on these.
- qualification, seeds 2101 to 2106, under the hidden source regime, minted before any
  graded world. Every threshold is frozen on these and on nothing else.
- graded, seeds 3101 to 3103, under the hidden source regime, minted after the freeze and
  never read back into it.

Qualification worlds are hidden-regime rather than development-regime because a bar frozen
on the development band and applied to a world outside it is not a bar; the external
review's "at least three, preferably five, independent hidden worlds" is the same point.
What makes the graded worlds graded is that they are minted after the bars exist, not that
they are drawn from a different law.

### The freeze, and what it found

`scripts/freeze_v4_bars.py` calibrates both strong lines on the twelve development worlds,
runs both on the six qualification worlds, sets every bar from the worse of the two
witnesses times a margin declared in the script before any world was scored, checks that
both witnesses then clear those bars, and runs the sixteen controls. Output is
`bars/national-v8/`.

Both witnesses pass on all six qualification worlds. Nine of the sixteen controls fail a
named gate on every qualification world:

- `register_only`, `survey_only`, `no_dedup`, `benchmark_only` and `version_three_recipe`
  fail on accuracy, coverage and the rate block. The version-three recipe, which is proof
  obligation 2, fails on nine families at once, with a national person count off by 10 to
  22 percent and pooled count coverage between 0.000 and 0.042.
- `inflated_intervals` fails the interval score, `static_projection` the projection
  accuracy and coverage, `uniform_allocation` the reserve feasibility.
- `deterministic_linkage` fails the mortality rate gate on every world, by a factor of 3
  to 9 against a ceiling that is itself loose: percentile errors of 16 to 39 against a
  ceiling of 4.50.

Six do not separate, and one separates on four worlds of six. The reason is one measured
fact, not six: the bars that would catch them are frozen from witnesses that are
themselves poor on those metrics, and the freeze takes the worse of twelve world by
witness pairs.

- `tau_mean` freezes at 0.8057 and `tau_worst` at 1.425, against a criterion that cannot
  exceed 0.95 by construction. `mean_only_tail`, `normal_tail` and `padded_tail` therefore
  clear the tail gates on five or six worlds.
- `skill_minimum` freezes at -0.9927, because one witness pair scored -0.8427.
  `proportional_reserve` clears it everywhere.
- `rate_coverage_floor` freezes at -0.06, because the witnesses' own rate intervals cover
  between 0.04 and 0.55 of the gated cells.
- `ignore_health_selection` and `development_average_regime` clear every gate, since the
  gates that would catch them are the rate and tail gates above.

The bars are not re-chosen to make the controls fail. A margin picked after seeing which
control clears it is not a bar. What the freeze reports instead is where the surface
stands: the gates that separate are the ones the witnesses are good at, and the gates that
do not separate are exactly the ones where the strong reference is currently no better
than the shortcut it is supposed to beat.

### What that leaves open, with the measurement behind it

- **Six ablations do not separate.** They need a reference whose regional liability is
  accurate enough to calibrate a tail, not a looser bar. Measured on a qualification
  world: the reference's regional liability means are 3 to 9 percent off, against a sealed
  regional tail width of 0.047 to 0.139, so its exceedance probabilities scatter from 0.00
  to 0.27 where they should sit at 0.05.
- **The rate intervals under-cover badly.** Witness coverage on gated rate cells runs from
  0.04 to 0.55 against a nominal 0.90. The gamma intervals around the partially pooled
  rates do not carry the linkage and selection uncertainty that moves the point estimate.
- **Skill is unstable across worlds**, from -0.84 to +0.31 for the same method, so proof
  obligation 7 is not yet met on the decision gate even though it is met on the counts.
- **The mortality trend is weakly identified at this world size.** The experience file's
  drift estimator has a standard error of 0.0106 against an axis spread of 0.0233, and its
  rank correlation with the realized intensity over eighteen worlds is 0.195. The world
  size was set by the ensemble's cost, not by identifiability, and this is the price.
- **The health-selection axis has a trace that does not carry across the regime shift.**
  The archive-to-anchor rate gap tracks the axis at 0.734 within the twelve development
  worlds and at -0.829 within the six hidden ones, because the level statistic also moves
  with the health source's own coverage. A county-level gradient statistic, which would
  not be confounded, is too thin at 3,000 survey respondents over 18 counties: it reads
  0.121.

### The sealing path is wired but not exercised

`scripts/build_sealed_reconstruction_packet.py` now builds at `packet.GRADING_WORLD`, the
same size as the development and qualification worlds, and takes `--workers`. It still
refuses any world that is not a hidden-regime draw with both declared intensities outside
the development band and inside the public envelope.

It was not run. Registering a version-four seal manifest needs the master key, and the
sealed `person_table` digest changed when latent frailty joined the person schema, so
`verify_sealed_world(0, seals/meridia-reconstruction-v3.json)` returns False on the
version-four generator with every other layer's digest unchanged. Version four needs its
own manifest, registered once, and that is the one step of this pass that is gated
outside the repository. The three graded worlds built here carry committed seeds instead,
which is enough to hold the surface but is not a sealed world.

## Integration pass two: what the external review changed

The review of the first version-four pass found ten defects that had to be fixed before
any bar could be frozen. Each is recorded here with the choice taken and the reason. The
worlds were rebuilt from scratch afterwards, because seven of the ten change what a world
contains.

### The survey instrument is drawn per world

`survey.SURVEY_BANDS` gives nine published bands, one per mechanism field of the survey:
the three coefficients of the unit-response logit, the two item-missingness rates and the
money-missingness dial, the money measurement error, and the age-heaping probability.
`draw_survey_params` draws one continuous value per field per world, and `build_world`
carries the draw through both snapshots.

Before this, `packet.py` called `draw_survey` with the dataclass defaults, so the whole
nonresponse and measurement model was the same number in all twenty-one worlds. The
development worlds ship truth, so it was estimable exactly there and transferred unchanged
to a world nobody had seen. That is the version-three failure in a layer version four had
not touched. The bands are written into `contract.json` under `survey_family`; the
realized values are retained, not published. The anchor's sensitivity and specificity stay
fixed and published, which is what makes the anchor an anchor.

### The mortality age slope and the frailty shape are drawn per world

`CHARACTER_RANGES` gains `gompertz_b`, `makeham`, `infant_extra`, `move_city_prob`,
`frailty_sigma`, `frailty_age_slope` and `frailty_urban_slope`. The age slope is the
sharpest of these: it is the first thing a Lee-Carter or CBD fit estimates, and the
five-year experience file identifies it well, so a constant slope handed a method the
hardest parameter of the mortality model for free. The three frailty shape fields are the
latent distribution the health-selection mechanism runs on, and a constant there let the
selection model be calibrated once on a world that ships truth.

### The target-dependence axis carries one mechanism

`missingness_target_dependence` is now only the frailty slope of health-source inclusion.
Item missingness on the money value has its own published slope, `item_missing_band_slope`.
One coefficient driving two mechanisms with different targets left neither identified, and
the anchor statistic's sign reversed between the development and hidden regimes, which is
what an unidentified axis looks like from outside.

### Three of the five declared interactions cross two axes

- migration by stale-address linkage: the recent-move term of identifier persistence is
  scaled by the world's migration intensity, so a world that moves people harder loses
  their identifiers faster. It is a product of `migration_age_pattern` and the linkage
  family that `linkage_urban_gradient` also enters.
- health completeness by latent frailty: the frailty slope of health inclusion is scaled
  by `administrative_completeness`, a product of two axes.
- age error by the age slope of mortality: age reporting error is scaled by the world's own
  Gompertz slope, which is now a per-world draw rather than the constant 0.105.

Before this every axis entered exactly one site, additively, so a method that fitted each
axis separately on the twelve development worlds transferred exactly to any recombination,
and the hidden world's new joint configuration carried no difficulty beyond its six
marginals.

### The mortality improvement band is wide enough for its anchor to resolve it

The development band moves from (0.002, 0.026) to (-0.010, 0.048) and the public envelope
from (-0.012, 0.042) to (-0.030, 0.075). The drift estimator's standard error on a world
this size is about 0.011, set by the roughly two and a half thousand deaths a five-year,
six-state experience file carries in the trend window. Against the old spread of 0.023 the
anchor could not resolve the axis it exists to identify. The band is the cheap side of that
inequality: lengthening the file or growing the world both cost build time that the
continuation ensemble already spends.

### Structural eligibility is per age band

`actuarial.EXPOSURE_ELIGIBILITY_BY_BAND` replaces one flat five thousand person-years. At
the committed world size a state by sex cell holds about 9,100 person-years at 18-44 and
about 720 at 85 and over, so the flat floor removed every cell at 65 and over from every
rate gate: the ages the obligation is made of, since the benefit starts at 65 and deaths
and first qualifying events concentrate there. A submission could be arbitrarily wrong
about old-age mortality and still clear every rate ceiling. Person-years is the wrong
invariant to hold constant across bands, because the same exposure buys two orders of
magnitude more deaths at 85 than at 8, so the floors fall with age.

Measured on qualification world qual-5, seed 2106, over the sixty-month window. State by
sex cells, the level the rate gates read:

    band     exposure range        gated cells, flat floor   gated cells, per band
    0-17     3,681 to 16,863 py    6                         8
    18-44    4,639 to 20,631 py    8                         12
    45-64    3,886 to 17,267 py    8                         11
    65-74      996 to  4,523 py    0                         9
    75-84      348 to  1,671 py    0                         3
    85+         47 to    178 py    0                         0

County by sex cells on the broad bands, the level the exposure gate reads: 0-17 goes from
two gated cells to four, 18-64 from eight to fifteen, and 65 and over from none to three.

The oldest band stays ungated at this world size and that is recorded rather than fixed by
a lower floor. A state by sex cell holds 47 to 178 person-years at 85 and over, which is
five to forty expected deaths; a floor that admitted it would put a cell with a forty
percent standard error into the gate percentile, and since one ceiling covers every gated
cell of an estimand, the noise would raise the ceiling for the bands that can be resolved.
Exposure at 85 and over is still scored inside the broad 65 and over band, and mortality
and incidence there are reported as diagnostics.

### The protected table has a utility requirement

`disclosure_audit` returns `utility`, the share of cells at or above the disclosure
threshold that the submission published, and `evaluate_gates` refuses a submission under
`bars["disclosure_utility_floor"]`. Protection is one-sided and a table of all-missing
cells meets it, which is the defect the protocol's own external review names. The
`suppress_all_detail` control is a strong submission whose detailed table publishes
nothing, and it exists to show the floor firing.

### A bar is clamped to what its criterion can take, and a bar file says whether it froze

`scripts/freeze_v4_bars.py` carries a declared attainability cap per ceiling and a declared
minimum per floor bar, both fixed before any world was scored. A bar that would be written
outside the range its own criterion can take is written at the limit instead, the freeze
report names it, and the witness then fails it rather than inheriting a number that can
never fire. The first pass emitted a `tau_worst` of 1.425 against a deviation whose maximum
is 0.95, a `regional_shortfall_ceiling` of 1.5 against a probability, and a
`rate_coverage_floor` of -0.06 against a coverage that cannot go below zero.

`bars.json` now records `frozen`, written after the control battery rather than before it,
and `verify_submission` refuses to gate a version-four submission with a bar set that does
not record a completed freeze. The first pass wrote its bars before the battery ran, the
run ended NOT FROZEN, and the file it left behind carried no trace of that.

### The tail calibration uses the published total in both directions

`calibrate_quantiles_to_total` may widen a tail as well as narrow it, up to a factor of
three. R is built from the sealed regional quantiles, so a method whose implied reserve
sits under R has been told by a published quantity that its tail is too low. Refusing to
use it left the reference filing quantiles about five percent under the truth on a tail
five to fourteen percent wide, which put its sealed exceedance rate at three times nominal
and made the tail gate unattainable by the reference that has to attain it. Measured on a
qualification-shaped world before and after, the reference's regional exceedance rates
moved from 0.13, 0.18, 0.13, 0.16 to 0.047, 0.070, 0.039, 0.055 against a nominal 0.05.

### The world set of this pass, and what is sealed

Twenty-one worlds at `packet.GRADING_WORLD`, all rebuilt after the changes above because
seven of them change what a world contains:

- twelve development worlds, seeds 1101 to 1112, one per row of the committed design,
  shipping truth, the only worlds a method may tune on;
- six qualification worlds, seeds 2101 to 2106, hidden regime, the only worlds a bar
  reads;
- three graded worlds, seeds 3101 to 3103, hidden regime, built and left closed.

The three graded worlds are independent hidden-regime worlds in the sense the protocol
asks for, and each now draws its own level pattern and its own pair of outside axes. They
are not sealed worlds: registering a version-four seal manifest reads the master key, which
is a step outside this pass. The path is wired and unchanged apart from the level-pattern
check, and the two commands are:

    python3 -c "from meridia.sealing import seal_worlds; seal_worlds(...)"
    python3 scripts/build_sealed_reconstruction_packet.py --seal-manifest seals/meridia-reconstruction-v4.json --index N --out OUT

A version-four manifest must register indices version three did not spend, which is five
and above: `sealed_seed` is a function of the index alone, so a repeated index hands a
version-four graded world the geography and population of a world version three already
exposed. The person-table digest changed with the frailty column and again with the
per-world frailty shape draw, so the version-three manifest cannot be reused.

### The reference reads the benchmark's state series, not only its national value

`benchmark_reconciliation` moved the county-up national count toward the published
benchmark by inverse variance and scaled every level by that one factor, which cannot
touch the composition. The composition is where a register-based reconstruction is
weakest: coverage rides the county economic gradient and the outpost penalty, both
declared in the public source ranges, so a state whose counties sit on one side of that
gradient is off by far more than the nation is.

Two steps were added, both on published inputs and both shared by the two strong lines:

- `benchmark_state_reconciliation` combines each state's register count with the
  benchmark's own state series by inverse variance, under the benchmark's declared state
  bias family and a state-level register model allowance of 0.10 taken from the published
  coverage ranges, then rescales the combined vector to the register's national total so
  the national step and this one compose instead of multiplying twice.
- `benchmark_age_scale` rakes the county age cube the liability is priced on to all three
  published count items at once: children under sixteen and people sixty-five and over
  take their own factors, and the middle ages take whatever factor makes the three blocks
  add to the reconciled headcount. Scaling by the persons factor alone puts the total right
  and leaves the shape wrong, and the obligation pays from sixty-five.

Measured on qualification world qual-5, the design-based line, one hundred bootstrap
replicates: state persons errors move from -20.6, -12.2, -8.4, -7.4, +3.0, +11.5 percent to
-6.8, -5.6, -5.0, -2.4, -1.4, -0.8 percent, and the reserve skill against the public
baseline moves from -0.139 to -0.090.

## What the rebuilt proof sequence found

Twenty-one worlds rebuilt, both witnesses and the seventeen-control battery run on the six
qualification worlds, bars written to `bars/national-v8`. The verdict is
`RESULT: NOT FROZEN`, and `bars.json` records `frozen: false`, so the verifier refuses to
grade with it. What blocks the freeze is proof obligation 2, not the ablations.

### Proof obligation 4 is met, with margin

`scripts/reserve_decision_value.py` on the six qualification worlds, with the public
size-proportional baseline and the perfect-information oracle on the sealed 2,048-member
ensemble:

    world    R              tail width      J(A_B)      J(A*)      gain     held out
    qual-0   243,050,000    0.031 to 0.146    875,276    329,472   62.36%   64.69%
    qual-1   225,872,000    0.040 to 0.121  6,788,227    117,975   98.26%   98.27%
    qual-2   250,701,000    0.114 to 0.134    586,569    418,029   28.73%   22.79%
    qual-3   248,214,000    0.029 to 0.152  4,237,394    524,224   87.63%   87.44%
    qual-4   244,738,000    0.085 to 0.134  1,488,037    370,413   75.11%   75.75%
    qual-5   234,376,000    0.052 to 0.108  2,677,771    173,000   93.54%   94.15%

The held-out column fits the oracle on half the ensemble and pays it on the other half, so
the gain is not an allocation fitted to Monte Carlo noise. Under the first pass's baseline,
which spread the slack above the submitted quantiles, the same worlds gave the decision
almost nothing to do.

### The ablations fail their intended gates

Each of the seventeen controls fails a named gate on at least four of the six worlds,
and the six that pass anywhere all pass on qual-4, the one world where both witnesses also
pass, which is what a bar frozen from the worst pair over six worlds does to the best
world. Named, with the gate and its worst value:

- `version_three_recipe`, the falsified strategy: fails on all six, on accuracy, coverage,
  interval score, projection, rate and tail; state persons error 0.21 to 0.30.
- `deterministic_linkage` with archive-only rates: fails on all six, on the mortality rate,
  percentile error 11.5 to 36.8 against a ceiling of 1.0.
- `ignore_health_selection`: fails on five, on the tail and the reserve.
- `development_average_regime`: fails on five, on the rate and the tail.
- `mean_only_tail`: fails on all six, pooled exceedance deviation 0.24 to 0.59.
- `normal_tail`: fails on five, pooled deviation 0.21 to 0.33.
- `padded_tail`: fails on five, on the tail and the reserve.
- `proportional_reserve`: fails on all six, skill -3.68 on qual-0 and the tail elsewhere.
- `uniform_allocation`: fails on all six, infeasible or on the rate.
- `suppress_all_detail`: fails on all six, disclosure utility 0.000 against a floor of
  0.878.
- `register_only`, `survey_only`, `no_dedup`, `benchmark_only`, `inflated_intervals`,
  `static_projection`, `exact_key_union`: fail on accuracy, coverage, interval score or
  projection on every world but one.

### What blocks the freeze: the reference itself

Both witnesses clear every bar on qual-4 and fail on the other five, and on the twelve
development worlds three of twenty-four runs pass. The failures are the tail and the
reserve, and behind them one measurement: the reference's regional liability means are off
by up to 17 percent while the sealed ensemble's regional process standard deviation is 2.8
to 5.5 percent of the mean. An exceedance rate cannot sit near 0.05 when the level is wrong
by three standard deviations of the whole distribution being scored.

The cause is not the headcount and not the anchor. On qual-5, after the benchmark steps
above, the state persons errors are within 6.8 percent and the state elder counts within 13
percent, and sharpening the benchmark's realized state bias by a factor of five changes the
regional liability bias by less than a percentage point. What remains is the regional
mortality and incidence estimates: the reference's own mortality rate percentile error at
state level is 2.26 to 3.37 against a ceiling of 1.0, and those rates are what price the
obligation.

That error was invisible in the first pass because the flat five thousand person-year floor
removed every band at 65 and over from every rate gate. The per-band floors put those bands
back, and the reference's weakest quantity is now scored.

### The next pass, in order

1. The regional mortality and incidence estimator is the binding constraint on three gates
   at once: the rate ceilings, the tail calibration and the reserve skill. Nothing else in
   the chain is worth touching before it.
2. `administrative_completeness` is still not identified by either statistic tried: the
   register against the survey gives a signed rank correlation of -0.150 over eighteen
   worlds, and the register against the published benchmark by state gives -0.057, with the
   sign reversing between regimes. The covariate it rides on, `econ_c`, is itself thinned by
   the mechanism, which is the likely reason. Either find a statistic that does not go
   through `econ_c` or record the axis as partially identified.
3. `missingness_target_dependence` now traces its realized coefficient consistently, at
   -0.350 within development and -0.486 within the hidden regime, where the sign declared
   before the measurement was positive. A stable sign in both regimes is identifiability;
   the direction the family implies has to be derived rather than read off, and until it is
   the axis is recorded as traced with an undetermined sign.

### Determinism receipt

`qual-0`, seed 2101, built twice at the committed size, once inside the four-process shard
that produced the world set and once alone across seven processes:

    manifest sha256 first  07e3d7c48e4f6c8648ed53004d4f2d35c53791bd87b670864b99f022cfb68bc1
    manifest sha256 second 07e3d7c48e4f6c8648ed53004d4f2d35c53791bd87b670864b99f022cfb68bc1
    manifests identical True
    files whose digest differs []
    files hashed 22

The worker count divides the continuation ensemble between processes and changes nothing in
the packet, which is what the second build checks. A small world built at one worker and at
three gives the same result in `tests/test_packet.py`.

## Phase two, 2026-09-03

### The verifier scoring surface

Owner of this section: the lane that holds `meridia/actuarial.py`, `meridia/verify.py`,
`meridia/scoring.py`, `scripts/freeze_v4_bars.py` and their tests. It closes the phase-one
audit findings on the scored surface. Every entry names what changed, the reason, and the
measurement behind it. Each change carries a test that fails on the previous tree and
passes on this one.

#### Eligibility comes from a published rule, not a floor per band

`EXPOSURE_ELIGIBILITY_BY_BAND` was a hand-written person-year floor for each band. It gated
the right ages, which the flat five thousand did not, but it stated no rule, so nothing
said why a band's floor was the number it was. The table is now derived from two published
constants and one published reference schedule, all in `meridia/actuarial.py`:

- every cell, whatever its estimand, stands on at least 120 expected persons over the
  scored window, which is 600 person-years at the sixty-month horizon;
- a rate cell also carries at least 25 expected events of its own kind, which is one over
  the square of the declared 0.20 relative standard error target, at the published
  reference rate for its band;
- `REFERENCE_EVENT_RATES` holds that schedule, rounded to two figures from the twelve
  development worlds, which ship truth. No qualification or graded world enters it.

The floor is therefore a minimum expected death count and a minimum expected event count,
which is the invariant the audit asked for. Person-years is the wrong one to hold fixed
across bands, because the same exposure buys two orders of magnitude more deaths at 85 than
at 8. Derived floors, in person-years:

    estimand              0-17    18-44   45-64  65-74  75-84   85+   18-64   65+
    exposure               600      600     600    600    600   600     600   600
    mortality rate      50,000   16,667   2,500    600    600   600   5,000   600
    qualifying rate      1,250      833     625    600    600   625     714   600

Measured on the six committed qualification worlds, gated state by sex cells for the
mortality rate, hand-set floors in brackets:

    world     45-64      65-74      75-84      85+     18-44
    qual-0    10 [10]    10 [10]     2 [2]     0 [0]    0 [10]
    qual-1    12  [8]    12  [8]     6 [6]     0 [0]    2  [8]
    qual-2    10 [10]    10 [10]     4 [4]     0 [0]    0 [10]
    qual-3    10  [8]     9  [6]     4 [4]     0 [0]    2  [8]
    qual-4    12 [10]    12  [8]     4 [4]     0 [0]    0 [10]
    qual-5    12 [11]    12  [9]     3 [3]     0 [0]    2 [12]

Both bands the obligation is priced on are gated at state level on all six worlds. Two
consequences are taken deliberately rather than repaired. The 85 and over band stays
outside every rate gate, now because a cell of at most 178 person-years is under 120
expected persons, not because a floor was chosen to keep it out. Mortality at 0-17 leaves
the gate on every world and at 18-44 keeps only the two largest cells on three of six,
since a state by sex cell there carries under 25 expected deaths; a gate on a cell that
thin raises the ceiling for the bands that can be resolved, which is the same argument the
oldest band was already excluded on.

At county level the exposure gate reads more of the table than it did: 0-17 goes from 1 to 8
gated cells to 30 to 36, 18-64 from 15 to 24 to 32 to 36, and 65 and over from 2 to 4 to 17
to 22, out of 70 cells per block. The exposure ceiling is frozen from the worse witness over
that set, so it will freeze looser than it did, and it will be read on cells that carry the
obligation instead of the dense young ones alone.

#### The detailed table is scored on what it publishes

`disclosure_audit` already reported `utility`, the share of releasable cells a submission
published, and a floor refused an all-missing table. A share alone is met by publishing
every releasable cell as any number at all, so the audit now also scores the released
cells: `detailed_error` is the median relative error over them, with the published
disclosure threshold as the stabilizer, and `evaluate_gates` refuses a submission over
`bars["detailed_accuracy_ceiling"]`. The 0.95 percentile and the worst cell are reported
and not gated.

The median is the gated statistic because the tail of that distribution is the thin cells
every reconstruction is poor on. Measured on qualification worlds qual-0, qual-1 and qual-5,
over the 167 to 180 releasable cells of each: the two strong references sit at 0.10, 0.10
and 0.18 at the median and at 0.28, 0.28 and 0.65 at the 0.95 percentile, while a table that
publishes every releasable cell as one number sits at 0.95 to 0.96 at the median. The
declared attainability cap for the bar is 0.50, which leaves the reference room and refuses
the constant table.

#### A bar that cannot fire stops the freeze

Clamping to the declared caps was already in place. What was missing is the verdict: a bar
written at zero on a criterion that is never negative was recorded as a note and the run
went on to report a freeze. `scripts/freeze_v4_bars.py` now carries `CRITERION_RANGE`, the
range each bounded criterion can take, and `_unattainable_bars` returns a hard list when a
ceiling sits at or above its criterion's maximum, when a floor sits at or below its minimum,
or when a bar is written with no declared range at all. The list goes at the top of the
freeze report, the result line names it, and the run does not complete. The three bars of
the first pass, a `tau_worst` of 1.425, a `regional_shortfall_ceiling` of 1.5 and a
`rate_coverage_floor` of -0.06, are the cases the check is written against: the first two
clamp back inside their range, and the coverage floor is the one that still cannot fire.

#### The tail is scored in units of the ensemble's own width

The tail bars that sat inside their range were still measured against the level, where a
tail that is out by its entire width reads as a fraction of a percent. `score_reserve` now
also reports `mean_q95_width_error` and `mean_es95_width_error`, the distance from the
sealed q95 and ES95 in units of the sealed tail's own distance above the sealed mean, with
the divisor held at a published one percent of the regional mean so a nearly degenerate
region cannot divide by zero. Both have a bar and a declared attainability cap of 0.50.

Measured on a 2,048-member ensemble whose regional width is 0.100 of the mean, which is the
0.099 the sealed worlds carry: a truthful tail reads 0.000, a mean-only tail reads 1.000 and
a doubled tail reads 1.000. Under the four tail bars of the first freeze the doubled tail
produced no reason at all, with a pooled exceedance deviation of 0.047 against 0.15, a
worst-region deviation of 0.048 against 0.30, and a quantile score of 0.0091 against 0.0098.
The width bar refuses it.

On the committed qualification worlds the same criterion reads 0.93 to 4.87 for the two
strong references. That is the regional liability level, in the units the tail gate cares
about: the references are out by one to five tail widths, which is why their exceedance
rates scatter instead of sitting near 0.05. The bar is written at its cap and the witness
fails it, which is what the freeze rule says to do, and the gate cannot freeze until the
regional mortality and incidence estimator improves. That was already the first item of the
next pass; the width criterion states it in one number per world instead of six exceedance
rates.

#### The practical baseline never reads the submission

`score_reserve` fell back to the submission's own q95 for the size-proportional baseline
whenever the contract published no `baseline_share`, which put the submission on both arms
of the skill denominator: a padded tail moved the baseline it was scored against. The
fallback is now an even split of R across regions, which is still a published rule, and the
report says which was used in `baseline_source`. The packet publishes the share, so the
committed worlds take the contract branch and nothing about them changes. The oracle stands
on the truth ensemble under non-negativity and the total alone, as it already did.

#### A gated block with no eligible cell is named

`score_rates` dropped a block whose cells all failed the floor, so a gated block that
decided nothing looked exactly like a block that passed. Every block is now recorded, with
`n_cells` zero, an empty cell list and a reason, and `evaluate_actuarial_gates` fails a
gated block that carries no eligible cell. The cells a block did read are recorded on the
metric as well, which is what the freeze provenance writes out.

#### The ratio-consistency check the docstring promised

`check_rate_additivity` claimed mortality and incidence were checked for ratio consistency
against their own exposures. They were not. The check is now implemented rather than the
claim withdrawn: a published rate times its own published exposure is an event count, and a
state's event count has to equal the sum of its counties'. A submission that files a state
rate its county rates and exposures contradict is stating two different numbers of deaths.
Measured on the twelve witness submissions of the qualification runs, the worst relative
inconsistency is 1.4e-13, so the existing tolerance of 1e-6 leaves the strong lines
untouched and refuses a real contradiction.

#### The freeze records which cells each rate bar read

A rate ceiling is only as strong as the cells its eligibility rule admits, and the first
pass froze three of them on a cell set that held no band at 65 and over without saying so
anywhere. `_rate_cell_record` collects, per gated block, the count per world and witness,
the distinct cells, and the bands they cover, and any world where the block had nothing
eligible. The freeze report prints it, `bars.json` carries it under `frozen_from`, and
`PROVENANCE.md` writes the derived eligibility floors next to it. A block with no eligible
cell also no longer enters the bar it would otherwise loosen.

#### What this leaves for the next freeze

The bars under `bars/national-v8` were frozen before these changes and are not edited by
hand here. `bars.json` records `frozen: false` and the verifier refuses to gate with it, so
nothing reads a stale set. A freeze run has to produce the new bars, and on the measurements
above it will end NOT FROZEN on the two width bars for the reason already recorded: the
regional liability level, not the tail shape, is what the strong references miss.

### The generator's remaining leaks

Owner of this section: the lane that holds `meridia/survey.py`, `meridia/demography.py`,
`meridia/character.py`, `meridia/mechanisms.py`, `meridia/sources.py`, `meridia/packet.py`,
`meridia/events.py`, `scripts/build_v4_worlds.py` and their tests. It closes the phase-one
audit findings on the generator side. Two of the eight were already closed on the baseline
commit and had no test that would catch a regression; they are recorded here with the test
that now pins them. The other six changed code, and each carries a test that fails on the
baseline tree and passes on this one.

#### The survey instrument has an envelope, and an evaluation world leaves the band

Nine survey fields were drawn per world already, which closed the version-three leak, but
every world was drawn from the one published band. Twelve development worlds ship truth and
cover that band densely, so a nonresponse and measurement model fitted there transferred
whole to a world nobody had seen: nothing about the instrument sat outside the region the
open worlds spanned.

`survey.SURVEY_ENVELOPE` is now the instrument's published plausibility envelope, wider than
the development band on both sides of all nine axes, and `draw_survey_instrument` places two
axes of an evaluation world between the band and the envelope edge, with the side drawn per
axis. That is the arrangement the mechanism layer already used for its six regime axes, and
the count is published as `survey_family.n_outside_axes` beside the envelope itself. Every
value in the envelope keeps the instrument well formed: the three rates stay strictly inside
zero and one, and the money error stays positive. Which two axes a world moved, and the
realized values, are retained rather than published, and the anchor's sensitivity and
specificity stay fixed and declared.

#### The survey stream is keyed on the world and the snapshot, not on their sum

`_survey_at` passed `seed + tick` to `draw_survey`. Development seeds run 1101 to 1112 and
the two snapshot ticks differ by the preliminary lag, so pairs of worlds landed on the same
key and drew the same households, the same nonresponse and the same reported error. The key
is now `SeedSequence([seed, domain tag, vintage])`, built in one place by `survey_stream`,
with the world's own seed and the snapshot index rather than any arithmetic on the two.

#### Four demography fields and the hidden level pattern: already drawn, now pinned

`gompertz_b`, `makeham`, `infant_extra` and `move_city_prob` sit in `CHARACTER_RANGES`, reach
`DemographyParams` through `draw_world_character`, and are read by the ledger at
`build_event_history`. `HIDDEN_LEVEL_PATTERNS` is the public candidate set rather than a
world's choice: `draw_mechanism_design` takes the pattern and the pair of outside axes from
the world's own seed under a domain tag no other stream uses. Neither finding needed a code
change, and neither had a test.

They have one each now. The four demography fields differ over twelve worlds, none sits at
its dataclass default, and the world with the steepest drawn slope has the steeper hazard
ratio between ages 90 and 60. Nine hidden seeds give eight distinct level patterns and four
distinct outside pairs, the module carries no `HIDDEN_LEVELS` or `HIDDEN_OUTSIDE_AXES`
constant, and the design draw is shown to be a separate stream from the coefficient draw:
the same seed under the two source rules gives the same non-axis coefficients.

#### Three declared interactions are now a product of two axes at one site

One of the five declared interactions crossed two axes. The rest crossed an axis with a
public covariate or with a character draw, so five of the six axes entered additively at a
single site and a method that fitted them one at a time on the twelve development worlds
transferred to any recombination. `PAIRWISE_AXIS_INTERACTIONS` now names three, published in
the contract next to the families that carry them:

- `linkage_gradient_by_migration`, drawn from (0.15, 0.75), scales the rural excess of the
  name, address and linkage error rates by `linkage_urban_gradient` times
  `migration_age_pattern`. Measured on one world at its own drawn coefficients, the rural
  minus urban gap in the linkage error rate runs 0.0112 at a migration intensity of 0.25,
  0.0150 at 1.00 and 0.0216 at 2.30.
- `health_inclusion_completeness_by_target` was already the product of
  `administrative_completeness` and `missingness_target_dependence` in the health inclusion
  logit, and stays where it is. Moving it to item missingness, as one reading of the finding
  suggested, would put the target-dependence axis back on two mechanisms, which is what the
  next finding exists to prevent.
- `death_report_by_age_error`, drawn from (6.0, 16.0), is `mortality_improvement` times
  `age_reporting_error` in the probability that a death reaches the register after the
  snapshot, so a register that reports ages coarsely closes its records at a speed that
  depends on which way the death count is moving. Realized late-report probability against a
  published base of 0.22: 0.191 to 0.252 over the twelve development seeds and 0.183 to
  0.329 over the nine hidden ones. Both axes are centred at the middle of the development
  band, so a world in the middle of the design gets the base rate and the product is a
  departure from it.

The trace is a quantity a method already computes. Deaths in the experience file are read
off the ledger's own event ticks, never off a recorded tick, so the file gives the true
count while the register gives the captured one, and their gap moves with the product.

#### Item missingness has its own county effect

Item missingness on the money value reused the coverage county effect verbatim, so the
county pattern of a thin register and the county pattern of a missing value were the same
vector and one estimate did for both. `item_missing` is a county effect family of its own,
with its own published spread `item_missing_county_sd` in (0.09, 0.36). On one world the two
vectors have standard deviations 0.19 and 0.34 and a correlation of 0.09, and the test
shows the item missingness rate reads its own effect and not the other.

#### The mortality trend needs a settled ledger, not a wider band

The improvement axis had a trace through the experience file and the trace did not work. The
reason is not the axis and not the file's format: a ledger's opening years carry a settling
term that the trend estimator reads as improvement. The frail die first, each band refills
from below, and the death rate inside a cell falls for reasons that have nothing to do with
the world's mortality axis. Mean latent frailty among the living on one world falls 2.4
percent in the first year, about 1 percent a year by year three, and 0.4 percent a year by
year ten.

Measured on twelve small worlds at 40,000 persons, the published count-weighted drift
estimator against the realized intensity:

    window                       mean bias    rmse     rank correlation
    ledger months 0 to 60          +0.084     0.095          0.21
    ledger months 48 to 108        -0.013     0.071          0.50

The band is 0.058 wide, so a bias of +0.084 is larger than the whole quantity being
estimated. The committed world therefore runs `EXPERIENCE_BURN_IN_MONTHS`, which is 48, of
ledger before the file's first year, and `GRADING_WORLD.observed_months` is that burn-in plus
the five published years plus the twelve-month publication lag, which is 120 rather than 72.
The contract publishes `first_year_starts_at_tick` beside the tick the file ends at.

What remains in the late window is one published family, not an unknown. Four of the twelve
worlds carry an epidemic year inside the window, and those four hold the whole remaining
error: without them the estimates track the intensity to about 0.01. A shock year multiplies
a whole year's deaths by 1.5 to 3.0 across every cell at once, `shock_family` publishes the
form and the annual rate, and the file's own national series shows the year. Leaving that in
is deliberate: the axis is now estimable by a method that models the published family, and
not by one that fits a straight line through five points.

#### Graded seeds live outside the repository

A world's whole configuration follows from its seed, so three graded seeds written into
`scripts/build_v4_worlds.py` put the graded configurations in the tree, and a build loop that
printed every seed put them in a terminal as well. `graded_seeds` reads them from a sealed
JSON file outside the repository, at `~/.config/meridia/v4_graded_seeds.json` or wherever
`MERIDIA_GRADED_SEED_FILE` points, and refuses a missing, repeating or malformed file by
name. `progress_line` prints a seed only for the development family, which is the family a
method may tune on, and the qualification and graded lines carry the world's name, its file
count and its build time.

#### Cost, and what has to be rebuilt

At the committed world size the ledger goes from 132 months to 180, and the world build goes
from 5.7 to 8.0 seconds. The continuation ensemble is unchanged: a member pays for the
horizon window, which is still sixty months, so the 2,048-member cost of about 310 seconds
across fourteen processes stands and the added months are under one percent of a packet.

Every world has to be rebuilt, and every bar re-frozen. Three of these changes move what a
world contains on their own: the coefficient vector is longer, so every draw after the first
new entry differs; the county effect families gained one, so every family after it differs;
and the ledger is longer, so the ticks move. The bars under `bars/national-v8` already record
`frozen: false`, and `bars/national-v8/identifiability.txt` is a reading of the previous
worlds. Rerunning `scripts/identifiability_v4.py` on the rebuilt development and
qualification worlds is the measurement that says whether the mortality axis now clears the
bar the other five already meet.

One guard is left for the freeze lane rather than taken here.
`scripts/build_sealed_reconstruction_packet.py` refuses a world whose mechanism levels
repeat a development design row and whose two outside intensities are not both inside the
public envelope. The survey instrument now has the same shape of invariant, and the retained
`world.json` carries `survey_outside`, so the same script could refuse an evaluation packet
whose instrument does not leave the development band on two axes. That file belongs to the
sealing and freeze lane and is not edited here.

One cross-lane consequence is recorded here because the rebuilt worlds are what surfaced it.
The world at seed 4711 in `tests/test_methods_b_and_controls.py` now makes the Bayesian line
publish a protected cell: it estimates zero for county 10, ages 16 to 24, male, whose true
count is 4. The methods lane has taken the general point in its own section, that a gate on
the truth of a suppressed cell is not a rule a method can keep. Underneath it sits one piece
of arithmetic that is a hole either way: `meridia/methods/common.py` writes a cell as
`"" if 0 < value < suppress_below else round(value, 3)`, so an estimate of exactly zero is
published rather than suppressed. Seed 4711 is the only one of eight at that world size with
any protected cell, which is how the lower bound sat unexercised.

### The reference methods and the control battery

Owner of this section: the methods lane, `meridia/methods/actuarial_reference.py`,
`meridia/methods/controls.py`, the two strong lines that share them, and their tests.
Nothing outside `meridia/methods/` is edited here, and no estimator reads a truth file;
truth is read only to measure what the estimators did, and every number below is such a
measurement.

The six worlds every measurement in this section is taken over are qual-0, qual-1, qual-3,
dev-00, dev-01 and dev-04, with the design-based witness at 2,048 simulation paths, unless
a line says otherwise.

#### Ablation 5 now fixes the regime at the development-world average it names

`development_average_regime` set the mortality drift to zero and its standard error to
0.002. Neither number is a development-world average. Zero drift is not the average regime,
it is no regime at all, and 0.002 is an eighth of the smallest standard error the estimator
reports on a world this size, so the control was also filing a false confidence that no
development world would have produced. Its passing was being read as evidence about the
regime axis of protocol section 10.

The override is now built at run time from one of two places, and both were measured:

- the average over the twelve development worlds, fitted by `fit_development_regime` with
  the same estimator the strong line runs and stored in calibration A beside the other
  development-world constants. It reads mortality drift -0.0242 with a mean reported
  standard error of 0.0216, and incidence drift +0.0083 at 0.0111;
- with no calibration to hand, the published development band of the mortality improvement
  axis. The design is balanced, so the average intensity over the twelve worlds is the
  band's midpoint and the spread a method carries from that average is the band's own
  standard deviation. The axis is a proportional decline, so the drift is the log of one
  minus it: -0.0192 at a spread of 0.0167.

The two routes agree to 0.005 on the drift and to 0.005 on the spread, which is the check
that the closed form is the same quantity the twelve worlds measure. The realized axis
spans -0.030 to 0.075 in the public envelope, so on a world outside the development band
the substituted drift is wrong by up to 0.05 a year, which is where the projection and the
tails are meant to fail.

`ACTUARIAL_SWITCHES["development_average_regime"]` is now empty and `LayerParams` carries no
default regime, so there is no constant left in the table that could drift out of step with
the worlds the average is measured on.

#### The continuation is the published law, and the level it starts from is estimated

Protocol section 6 gives the truth ensemble independent futures drawn from a family the
contract publishes. The reference simulated something else: one shock kind at a probability
of 0.10 that moved mortality alone, against a published annual rate of 0.20 over three kinds
whose fields move together on one draw.

- `read_shock_family` parses `contract["shock_family"]` and `draw_shock_year` draws it as
  published: one Bernoulli a year at the published rate, one kind uniform over the kinds,
  then a single uniform draw shared by that kind's multipliers. An epidemic year now raises
  deaths and hospital admissions together, which is what makes it one event.
- The leave-home multiplier of the migration kind is read and deliberately not applied. The
  simulation carries net migration per cell, and the published multiplier is on a gross
  leaving rate; multiplying a net flow by a gross-flow multiplier is not the published
  mechanism. This is recorded rather than fixed, since carrying gross flows would need a
  second migration quantity the experience file does not publish.
- A level a method estimates over several years already contains the family's expected
  loading, and the continuation adds it again. `expected_shock_loading` states that
  expectation from the published family, the year-level shock posteriors take out the part
  the file shows, and what the detector could not resolve is removed once. Neither step can
  remove it twice.

The level errors the continuation propagates are estimated rather than assumed. They were
0.06 and 0.10 in log units, the same two numbers on every world and every region:

- the sampling error of each level, which is one over the root of the counts behind it;
- the disagreement between two independent measurements of the same level, over root two:
  the experience file against the register vintages for mortality, and the experience file
  against the anchored archive for incidence;
- the spread across the imputations of the link set, for mortality.

They are drawn as one national and one per region, so a region's whole estimate moves
together, which is the correlation a regional tail is made of. The multiplier is centred to
average one; without the half-variance term an honest widening of the tail arrives as a
quiet loading on the mean, and at the widths now estimated that loading was 4 to 7 percent of
the national liability.

#### What the reference reads off the participant files, world by world

Version four draws the survey instrument, the demography and the interaction coefficients
per world, so anything a method holds constant is a constant carried into a world nobody
has seen. Five quantities that were constants or were not read at all are now fitted, and
none of them reads a truth file:

- **The unit-response model.** The survey names the households each sampling unit drew, so
  the response rate is observed per unit and its regression on the county's urbanity gives
  the intercept and the urban coefficient, and the tilt of the responders' mean age against
  the register's own age distribution gives the age coefficient. The fitted
  propensities divide the design weights inside each county, so the level the design fixed
  stays and only the composition moves. Measured: the urban coefficient reads -0.42 on
  dev-00 against a published band of -0.85 to -0.15, and +0.04 on qual-1, which is outside
  it. The estimator is unbiased in shape and thin in evidence at eighteen counties, and that
  is recorded rather than hidden. The money coefficient of the same logit is left unfitted
  and reported as not identified: recovering it needs the population's money distribution
  in the units the survey reports money in, the two money sources sit on scales no
  participant file states, and the county-level route to it regresses the response rate and
  the money level on the same urbanity.
- **The Gompertz age slope**, from five years of state by band by sex counts rather than
  from one snapshot. It sets the age gradient inside each band of the priced schedule.
  Measured: 0.0998 on dev-00 against a realized 0.1073, and 0.1042 on qual-1.
- **Age heaping in the survey.** The register carries no heaping at all, measured at 0.000
  excess mass on multiples of five; the survey carries 0.12 to 0.20. Two band boundaries the
  obligation is priced across sit on a multiple of five, so the anchor's weights and its item
  are spread back over the neighbouring ages before the bands are cut.
- **The migration by stale-address interaction**, as a fitted term. A record leaves the
  register between vintages because the person died, because they moved and the identifier
  did not survive the move, or because the reported birth date changed. The last two are
  churn and they have opposite age patterns, so one flat rate read off the young bands
  removes too much at the ages the obligation is priced on. Two shapes are fitted instead:
  the mobility profile from the experience file and the age-error profile from a
  cross-vintage probe on records whose names are unique in both files. That probe also
  measures the second declared interaction directly: the reported birth date moves for
  0.0006 of records at 0-17 and 0.0036 at 65 and over on dev-00, which is the age gradient
  the family declares.
- **The health completeness by frailty interaction**, as a fitted surface. The inclusion
  probability is regressed on the register's shortfall against the published benchmark by
  state and on the cell's own anchored prevalence, with their product as the declared
  interaction term, and each thin cell is shrunk toward that surface instead of toward one
  pooled number.

One outright error was found and fixed while wiring the anchor. The anchor asks about any
admission inside the window and the archive count it was compared against was filtered to
the qualifying diagnosis groups, so the ratio was reading the share of admissions that
qualify as if it were the share of patients the archive holds. Measured on dev-00: 39
percent of recently admitted patients are in a qualifying group, the pooled inclusion
probability read 0.268 against a generator value near 0.63, and it now reads 0.695. Putting
the numerator back cut the mean regional liability error over the six worlds from 0.168 to
0.104.

#### Where the county detail is, and where it is not

The two microdata sources measure the levels the obligation is priced on far worse than the
experience file does, and the previous pass took their county detail at close to face value.
Measured over the six worlds, as the median absolute log error of a state level against the
horizon truth:

    estimator                                    mortality   incidence
    experience file                                  0.241       0.115
    register vintages, churn removed flat            1.577
    register vintages, churn fitted                  2.497
    register vintages, raw disappearances            4.996
    health archive, inclusion adjusted                            0.379
    health archive, raw                                           0.246

The register loses records to identifier churn at many times the death rate, so a
disappearance count is a churn measurement with a death rate inside it. Three consequences
are taken:

- levels are combined by their own precisions, not by a chosen weight. `blend_levels` takes
  a variance from each side, and the vintage side reports the variance its churn correction
  carries, so it ends up with about 4 percent of the weight on dev-00 and the anchored
  archive with about 2 percent of the incidence weight. No constant anywhere says so.
- a county deviation is shrunk by how much between-county spread survives its own
  measurement error, by moments, so a source whose correction dominates it contributes a
  vector of ones rather than the correction's noise.
- the composition of the priced population is raked to the experience file's own state
  shares. The file's exposure is person-years read off the same pass the truth uses, so its
  composition is exact as of eighteen months before the snapshot, and what moves those
  shares afterwards is the difference in ageing between states rather than the growth they
  share. Measured against the retained truth: the state shares of the population at 65 and
  over sit within about 3 percent, where the register-based reconstruction of the same
  shares sits within about 13 percent.

The publication lag is what stops the file being a contemporaneous headcount, and it does
stop that: the un-aged elder count from the file's last year runs 11 percent under to 9
percent over the truth, world by world. It does not stop the file carrying the state
composition, because that error is common across states and cancels in a share. If the
generator lane intends the lag to protect the regional composition as well as the level,
the lag is not the instrument that does it; a coarser published geography in the file would
be. Nothing is asked for here, and the reference codes against the file as it stands.

#### A control that submits the experience file on its own

`experience_history_only` is the eighteenth control. It opens no microdata: the population
comes from the last published year's exposure, split across the counties of a state in
proportion to land area, spread evenly over the single years inside each band, and aged
forward to the snapshot under the file's own survival and net migration. Rates and the
liability come from the file's state levels alone, with no linkage, no archive and no
anchor. Households, money and education have no source in an aggregate demographic file and
are filed as zero rather than borrowed from a register the control says it does not need.

Measured on qual-0 against `bars/national-v8`, it fails 92 named gates, among them the
county persons accuracy at 1.16 against a ceiling of 0.40, the national elder count at 0.11
against 0.07, the pooled exceedance deviation at 0.4978 against 0.15, the q95 error at 1.07
tail widths against 0.5, and the reserve skill at -0.58 against 0.05. It fails on the count
gates because a land-area split is not a population, on the rate gates because the level is a
year and a half stale, and on the tail gates because a distribution with no reconstruction,
linkage or selection uncertainty in it is not a predictive distribution.

#### Measured effect of the whole pass

Design-based witness, six worlds, 2,048 paths, scored against `bars/national-v8` with the
verifier as it stands. Previous reference first, this one second:

    criterion                                   before    after
    pooled exceedance deviation                 0.1815   0.1641
    worst-region exceedance deviation           0.6687   0.5722
    q95 error, in ensemble tail widths          1.2581   1.0324
    ES95 error, in ensemble tail widths         1.0091   0.8972
    reserve skill against the public baseline  -0.5595  -0.3783
    mortality rate, state percentile error      0.8628   0.7240
    incidence rate, state percentile error      0.4509   0.4758
    mortality rate interval coverage            0.6718   0.8336
    mean regional liability error               0.0789   0.0878

Every tail and reserve criterion improves and the mortality rate gate improves with it. Two
do not: the incidence percentile error is 0.025 worse and the mean regional liability error
is 0.9 of a percentage point worse. The second is the honest cost of the first: the level a
region is priced at now carries the file's composition and a shock loading taken out of the
level, and both trade a little bias for the variance the tails are scored on.

Rate interval coverage is where the change is largest, from 0.67 to 0.83 against a nominal
0.90, because the intervals now carry the estimated level uncertainty rather than two fixed
numbers. The phase-one audit recorded coverage of 0.04 to 0.55 on gated cells; that finding
is closed to the extent the level uncertainty explains it.

#### What is still open

- **The binding constraint is unchanged in kind.** The q95 error is a little over one
  ensemble tail width, against a bar of half a width. The regional liability level is still
  the thing to fix, and after this pass the two largest remaining terms are the
  reconstruction's own state elder counts and the state incidence level, not the mortality
  level.
- **The bars must be re-frozen.** `bars/national-v8` was frozen from the previous reference
  and its file already records `frozen: false`. Both witnesses fail it on five of six worlds
  here, which is a statement about that bar set and not a verdict.
- **Ablation 4 now separates on a narrower margin.** The anchored archive takes about 2
  percent of the incidence level weight, because the anchor's own sampling error at three
  thousand respondents is what the variance says it is. Ignoring health selection still
  fails on every world tried and always by more than the strong line does, but the freeze
  should confirm the margin on the qualification set rather than assume it.
- **The response model is thin.** The urban coefficient is inside the published band on one
  world and outside it on another. Eighteen counties and one sampling unit each is what the
  world size affords, and the estimator is stated as fitted rather than identified.
- **The disclosure gate is not attainable as written.** Suppression reads the estimate and
  protection is defined on the truth, so a cell whose true count sits under the threshold
  can carry an estimate above any fixed cut. On the small hidden world in
  `tests/test_methods_b_and_controls.py` the Bayesian line publishes one such cell out of
  three protected, with 0.94 of the releasable table published and nothing recoverable from
  the published totals. The test now asserts what a method controls, which is that no
  published cell is recoverable and that the released share clears the floor. Whether
  `published_protected` stays a hard failure or is scored as a rate is a decision for the
  lane that owns the audit.
- **The age-error shape carries no weight in the churn fit.** The mix reads about 0.03 on the
  mobility shape and 0.00 on the age-error shape on both worlds inspected, which says the
  age-error mechanism is real but too small at this world size to separate from mobility
  inside the disappearance count.

## Phase two rebuild, 2026-09-03 12:42 (bars/national-v9, worlds-p3, code da01a42)

Twenty-one worlds rebuilt from the phase-two code. Reference and control runs on all six
qualification worlds; freeze RESULT: NOT FROZEN.

- Witnesses: line A passes qual-2 only; line B passes none. Every remaining failure is in
  the tail block (q95 error 0.56 to 2.02 tail widths against 0.5; worst-region exceedance
  deviation 0.95 on qual-1 for both lines, meaning every continuation exceeds the submitted
  95th percentile in one region; ES95 error 0.67 to 1.48 widths), plus one rate near miss
  (qualifying_event_rate at state, 1.05 and 1.06 against 1.0, qual-1) and one reserve
  shortfall (0.383 against 0.35, qual-5, line A). The rate block otherwise clears after the
  phase-two eligibility change.
- Controls: on qual-2, the one world the reference passes, six controls also pass under
  bars set from the witnesses (static_projection, ignore_health_selection, also on qual-4,
  development_average_regime, normal_tail, padded_tail, proportional_reserve). When the
  reference's own error is large, bars frozen from it admit shortcuts. Bars must both admit
  the reference at a stated false-fail rate and reject every control on every world; a gate
  that cannot do both at this world size is a deletion candidate.
- Identifiability from participant files (signed rank correlation with the realized
  intensity, pooled): mortality_improvement +0.50, migration_age_pattern +0.53,
  age_reporting_error +0.65, linkage_urban_gradient +0.75, administrative_completeness
  +0.07, missingness_target_dependence -0.02. The last two have no usable trace and cannot
  be hidden outside the development range under section 10 until they are anchored or
  held inside the range.
- Reserve decision value on qualification worlds: the perfect-information allocation cuts
  expected uncovered obligation by 71 to 98 percent against the proportional baseline;
  slack of R for the reference 0.003 to 0.010.
- Reading: the binding error is the regional liability level, set upstream by elder
  exposure and survival, not the tail model. Next pass: the elder cohort-component line on
  the experience file with the shock process redrawn in every continuation member, then
  refreeze with per-gate false-fail calibration and the control-separation requirement.

## Phase three, 2026-09-03

### The generator lane

This section belongs to the lane holding `meridia/mechanisms.py`, `meridia/sources.py`,
`meridia/survey.py`, `meridia/packet.py`, `meridia/events.py`, `meridia/demography.py`,
`meridia/character.py`, `meridia/microdata.py`, the two build scripts under `scripts/`
and their tests. It closes the two unidentified regime axes the phase-two rebuild left
open, adds the regional shock loadings the protocol's phase-three list names, re-mints the
graded seeds, and takes the world build off the cost wall it was on. Each entry below
carries a test that fails on the previous tree and passes here.

#### The measurement these entries are read against

Thirty worlds, built at the committed geometry: 96 by 128 cells, 60,000 persons, 18
counties, 6 states, 180 ledger months, one world per row of the twelve-run design plus
eighteen hidden-regime worlds. The continuation ensemble is cut to eight members, which is
the whole cost of a packet and enters none of the six statistics, so the worlds are the
committed object and the measurement runs in minutes rather than hours.

The thirty seeds are committed, because a correlation over thirty worlds is a claim about
thirty definite worlds and a set that lives only in one run is a claim no reader can
rebuild. The twelve development seeds are 1101 to 1112 and the eighteen hidden seeds are
4101 to 4118, both in `scripts/build_v4_worlds.py`. `--family identifiability` builds the
set, `scripts/identifiability_v4.py --record` writes the reading and the seed set it was
taken on to `measurements/identifiability_v4.json`, and `tests/test_identifiability_v4.py`
holds the figures below to that file. Nothing is graded on these worlds and no threshold is
frozen on them.

An earlier version of this section reported six pooled figures taken on eighteen hidden
worlds whose seeds were recorded nowhere. Its development-side figures rebuild exactly, and
every pooled figure moves with the hidden draw, so the pooled six could not be checked
against anything. The figures here replace them.

`scripts/identifiability_v4.py`, signed rank correlation of each axis's statistic with its
realized intensity, with the resampled interval in brackets beside it:

    mortality_improvement            +0.754   [+0.554, +0.886]
    migration_age_pattern            +0.599   [+0.331, +0.808]
    age_reporting_error              +0.664   [+0.425, +0.814]
    linkage_urban_gradient           +0.879   [+0.757, +0.920]
    administrative_completeness      +0.640   [+0.408, +0.795]
    missingness_target_dependence    -0.182   [-0.519, +0.166]

The interval is the fifth to the ninety-fifth percentile of the reading over two thousand
resamples of the thirty worlds, drawn with replacement. A reader needs it to compare a
figure against anything, because a point estimate over thirty worlds says nothing about how
far the next thirty would move it, and the widest of these six spans about seven tenths.

#### The completeness axis gets the anchor protocol section 3 already lists

Register coverage rides the county economic gradient, the covariate that reports that
gradient is itself thinned by the same mechanism, and the state series pools counties from
both ends of it. Neither statistic tried before worked: the register against the survey
read -0.150 and the register against the state benchmark read +0.067, with the sign
reversing between regimes.

The benchmark now publishes a count for a defined subgroup, which is one of the imperfect
aggregates section 3 names. The subgroup is a band of counties: the producer classifies
every county by its own establishment payroll per resident adult and publishes the resident
person count of each of four bands. `geography.csv` carries the band of every county, so the
grouping is reproducible rather than estimated, and `contract.json` carries the definition,
the reference tick, the number of bands and the bias family. The subgroup series has its own
declared bias spread, 0.004 to 0.015 against 0.03 to 0.08 for the state series, because it
is one national operation on a published classification rather than six separate
collections. That has to be the smaller of the two for the series to be an anchor at all:
coverage moves by about two percent per band across the gradient, so a per-band bias at the
state spread would be the whole signal.

Register persons over published persons, band by band, is then the gradient with nothing in
between. The axis reads +0.640 pooled on the committed set, +0.699 within the development
worlds and +0.618 within the hidden ones, with a resampled interval from 0.408 to 0.795.

Nothing scored is handed over. The subgroup cuts across the states and counties the release
is scored at, and four biased totals over eighteen counties do not place a county.

#### The target-dependence axis has a trace, no anchor, and stays inside the band

The axis is the slope of health-source inclusion in a person's latent burden. It carries
that one mechanism and the survey's admission item is its one anchor, which is the
arrangement phase two settled. What phase two did not check is whether the anchor can read
it.

Three things changed and one conclusion follows.

The mechanism was too faint to leave a mark. Latent burden's mean moves by about half a log
unit between a child and a person of eighty, so at the raw slope an axis in the middle of
its band moved the inclusion share by six points across the whole age range. The slope is
now scaled by two and the resulting shift is capped at plus or minus two logits, which keeps
the far tail of the burden distribution from driving an inclusion probability to zero or
one, a case protocol section 10 refuses. The mechanism does leave a mark in the retained
truth: the count-weighted slope of the log odds of the archive's retained share across the
six actuarial age bands, in `retained/health_inclusion_truth.csv`, ranks with the realized
coefficient at +0.59 over the committed set. Twenty-nine of the thirty worlds enter it; the
remaining one has no archived admission in its oldest band.

The statistic is a gradient rather than a level. The national gap between the archive's
admission rate and the anchor's reads how much the source retains overall, which is a
different axis, and that is why it read -0.02. The anchor and the archive are now compared
on three broad age groups, under 45, 45 to 64, and 65 and over, over the same window and the
same population definition, and the statistic is the oldest minus the youngest, averaged
over the two snapshots. `contract.json` states the comparison.

The anchor still cannot resolve it. At about four thousand respondents the sampling error of
one age group's admission rate is near seven percent and the whole gradient is near ten, so
the statistic reads -0.182 on the committed set, with a resampled interval from -0.519 to
+0.166. A two-year anchor window was tried for exactly this reason, since doubling the rate
cuts the error of the estimate, and it made the trace worse: nearly everyone frail enough to
be admitted at all is admitted inside two years, so the wider window thins the very contrast
being read. Raising the item's declared sensitivity and specificity was tried and reverted,
because it buys a sharper anchor by softening a declared difficulty and it did not close the
gap either.

The bar that decides this is now a number in the tree, `ELIGIBILITY_MIN_CORRELATION` at
`meridia/mechanisms.py`. Under no ordering at all a signed rank correlation over thirty
worlds has a standard deviation of one over the square root of twenty-nine, which is 0.186,
so a reading below about 0.30 is one a coin reproduces on this many worlds. An axis is
eligible when the low end of its resampled interval is at or above 0.30, refused when the
high end is below it, and held inside the band when the interval covers it, because listing
an axis is the direction that makes a world harder for a reason no anchor reaches. Five axes
clear the bar and the closest of them, `migration_age_pattern`, has a low end of 0.331. The
whole interval of `missingness_target_dependence` is below the bar, so it is refused rather
than undecided, and a decision that used to rest on one point estimate of -0.004 now rests
on where the interval sits. `tests/test_mechanisms.py` recomputes the verdict from
`measurements/identifiability_v4.json` and the bar, so the list cannot drift from the
measurement without a test failing.

So the axis is held inside the development band on evaluation worlds.
`mechanisms.OUTSIDE_ELIGIBLE_AXES` is the list of axes a hidden world may push past the
band, it holds the other five, and the contract publishes it beside the rule. The mechanism
still runs and still recombines into a joint configuration the design does not spend, and
the axis's realized coefficient still leaves the band, because the declared interaction with
`administrative_completeness` scales it and that axis is anchored: over the eighteen hidden
worlds the realized coefficient runs 0.157 to 1.738 against a band of 0.20 to 1.30.

Development packets ship `truth/health_inclusion_truth.csv`, the archive's realized share of
each age band's admissions. The worlds a method may tune on therefore show the quantity the
anchor is a noisy view of, which is the difference between an axis a method can learn to
model and one it is asked to read off nothing.

This is the world size talking, and it is the same wall the mortality trend hit in phase
two. Growing the survey would close it, at the price of a sample that is a tenth of the
country.

#### Regional shock loadings

The protocol's phase-three list asks for these in as many words. Version four's shocks were
national and every region took the whole multiplier, so the regional liabilities moved as
one and the aggregate tail was what the six marginals already said.

Each region now carries a loading on the family, drawn once per world from the published
band 0.35 to 1.80 on its own stream, held for every year and every continuation member. A
multiplier m lands in region r as 1 + L_r * (m - 1). A loading of one takes the whole
multiplier and a loading of zero takes none of it in either direction, so a fertility
multiplier under one thins births less where the loading is low. At m of one the factor is
exactly one everywhere, which is what keeps a shock-free month byte for byte what it was.

The loadings scale the mortality and the admission multipliers and nothing else. Those two
price the obligation and those two are what the experience file's deaths and qualifying
event counts carry, so a development world exposes the loadings in a file the agent
receives. Splitting fertility or internal migration would put loadings where no anchor
reaches them. For admissions the national target takes the mean of the per-candidate factors
and the selection weight takes the factor itself, so the level follows the published family
and the split between regions follows the loadings.

Visible in the experience file, measured on the twelve development worlds: four carry a
mortality spike inside the published window, and in those the state death rate lift ranks
with the loading at +0.886, +0.829, +0.657 and -0.314, median +0.743. The band and the rule
are in `contract.json` under `shock_family`; the realized vector is retained.

Replay and determinism are unchanged. The loadings are a function of the world's seed on a
stream nothing else uses, so no other draw moves; a continuation member inherits the world's
loadings and redraws only its own shock years; and a ledger run with no shock in it produces
the same events it produced before the loadings existed, which `tests/test_events.py` checks
by digest rather than by argument.

#### The graded seeds are re-minted

The values 3101, 3102 and 3103 appeared in transcripts and in tests, so the three worlds
they name are spent. Three new seeds were minted through the sealing module's keyed digest,
`sealed_seed(master, index)` at indices five, six and seven, which are indices version three
did not spend. They were written to the sealed file outside the repository at
`~/.config/meridia/v4_graded_seeds.json`, mode 600, and were not printed or logged.

Their digests, `sha256` of the decimal seed:

    index 5  271c38919fa7a7ae118b92c433be4ed9780a26ecd2af30ce9a1c6b923cb35732
    index 6  7b03878fc0426fb151f4753c2caa651d921a6fff6f9287b2513f5051c8976395
    index 7  5d4671046d118698c6b6e9dec3412432ba344c0bc198c6a5c89d3d547d924bdc

Anyone holding the master key can rederive the three seeds and check the digests. Nobody
without it can, and the three burned values are not among them.

#### What a world costs, and why a refreeze no longer pays for its futures

The twenty-one world set took about three and a half hours on six processes. Two things
made it that: the only division was inside one world's continuation ensemble, so five
sixths of the machine sat idle through every ledger, and each world waited for the one
before it.

`build_v4_worlds.py` now takes `--world-workers`, which builds whole worlds at once and is
separate from the `--workers` that divides one ensemble. Worlds share nothing, so the two
are independent and either can be one while the other is many.

The ensemble is also cached. A member is a function of the branch state the ledger kept at
the revised snapshot, the shock law it redraws its own future from, the horizon, the
obligation that prices it, the mechanism vector it runs under, and the source of the code
that runs it forward. A verifier, a bar and a scoring change sit downstream of all six.
`--cache` points at a directory of ensembles keyed on the digest of those six, so a rebuild
that moves nothing upstream of the ledger takes the futures off the shelf, and one that
moves anything upstream misses the key and pays. A cached ensemble with more members than
the packet asks for is used from the front, since a member is a function of its own index.

The sixth is there because the first five are recorded quantities and a generator edit is
not one of them. Retagging the substream a member draws its own future shock years on
reprices every member while the ledger, the mechanism record, the horizon and the
obligation stay byte for byte what they were: measured on one world at 48 by 64 cells, that
edit moved a regional liability by 0.635 relative and the mean over members and regions by
7.6 percent, and the key did not move. The key now carries a digest of the source of the
modules that price a member, walked from the import closure of `projection`, `actuarial`
and `events` rather than listed by hand, so a module that joins the pricing path joins the
key. `meridia/scoring.py` and `meridia/verify.py` are outside that closure and stay
outside the key.

The mechanism record went the same way for the same reason. It carried the standard
deviation of each county effect family, which a permutation across counties preserves
exactly, so two worlds that hand different counties different effects shared a key while
pricing different liabilities: reversing the mortality effect vector moved a regional
liability by 0.182 relative and left the key alone. The record now carries a digest of each
vector beside its spread, and of the county shock loadings as applied.

One world at the committed size, seed 1101, 2,048 members, six ensemble processes:

    cold, no cache        817.8 s
    cold, filling a cache 754.7 s
    warm, cache hit        19.0 s

All three write the same ensemble, digest `ce344979e02fe1b5...`. The nineteen seconds are
the terrain, the population, the ledger, the sources, the two surveys and the twenty-two
files; the eight hundred are the futures.

For the twenty-one world set that is the difference between rebuilding everything and
rebuilding what actually moved. A bar change or a verifier change now costs about seven
minutes of packet writing across the set rather than four and a half hours, and a
generator change still costs the full build, which is what a generator change should cost.
