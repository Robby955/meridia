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
