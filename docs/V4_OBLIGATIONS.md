# Proof obligations one to four, read against the P4 state

This file answers the first four of the seven proof obligations in the actuarial protocol,
section "Proof obligations before version four is called qualified", against this tree at
the close of P4. Each obligation is quoted verbatim, then given the artifact that
discharges it or fails to, then a verdict of PROVEN, PARTIAL or NOT PROVEN, then the one
next step that would move it. Obligations five to seven are out of scope here and are not
judged.

## What the P4 state is

- Bars. `bars/national-v10` records `RESULT: NOT FROZEN`, `bars.json` carries
  `"frozen": false` with an empty `gates` object, and the freeze report names one blocker,
  "replicate evidence missing; final verifier reports cannot be bootstrapped or resampled
  as replacement evidence". Behind it are zero final reference reports, zero paired
  replicates and zero control reports. No composite gate exists in this tree, so no
  submission of any kind can be called a pass or a fail against a frozen bar.
- Worlds. Eighteen packets at the committed 2,048-member ensemble, twelve development
  worlds at seeds 1101 to 1112 and six qualification worlds at the sealed qualification seeds, built
  into a scratch tree outside the repository as `worlds-p4`, with a second tree beside it
  at the candidate reserve rate 6321 as `worlds-p4-rate6321`. Neither tree is committed;
  the repository holds the builder, not the worlds.
- Reference submissions. Eighteen, from three fixed-seed lines A, B and C on the six
  qualification worlds, under `evidence-p4/phase_three/qualification` in the same scratch
  tree.
- The evidence runner halted in the reference stage, before `phase_three.measure`, so no
  control was scored in P4 and no verifier report was written to disk.

Every measurement quoted below was run fresh against those trees from this working copy,
with `PYTHONPATH` set to the repository root. Control submissions written for this file
went to a temporary directory, and nothing in the repository was changed to produce them.

One measurement is shared by three of the four obligations, so it is stated once. The
verifier was run on all eighteen reference reports. Every one returns `hard_pass=True`,
which is the deterministic half of the surface: file set, schema, additivity, contract
agreement and reserve feasibility. The reserve skill composite is the quantity that does
not survive. Its `skill_loss` is one minus skill, so a value above one is an allocation
worse than the published proportional baseline. Reading `skill_loss` by world and then by
line A, B, C: qual-0 0.1272, 2.0527, 0.3546; qual-1 undefined, undefined, undefined;
qual-2 0.5420, 0.8600, 0.6073; qual-3 9.7327, 4.9871, 8.8309; qual-4 0.5041, 1.1793,
0.3765; qual-5 2.1853, 11.7725, 1.3477. Three reports have no defined skill, eight of the
fifteen defined values are above one, and the worst is line B on qual-5 at a skill of minus
10.77. The three undefined values are the verbatim payload of the halt:

    qual-1 A: hard_pass=True skill={'skill_loss': nan, 'worst_regional_shortfall_probability': 0.44921875}

Pooled exceedance deviation on the same eighteen reports, world then line: qual-0 0.0387,
0.0488, 0.0817; qual-1 0.0500, 0.0500, 0.0500; qual-2 0.0425, 0.0500, 0.0389; qual-3
0.0865, 0.0454, 0.0787; qual-4 0.0443, 0.0499, 0.0376; qual-5 0.0500, 0.0500, 0.0500. A
reading of exactly 0.0500 is saturation: no continuation member exceeds the submitted q95
in any region, so observed exceedance is zero against a five percent target, and seven of
the eighteen reports sit there.

## Obligation 1

Verbatim: "A strong legal-information reference passes using only the files the agent
receives."

Artifacts.

- The legal-information half is discharged by the reference tests. `tests/test_methods.py`
  builds line A's submission by copying only the packet's `participant` directory into a
  fresh location and running the method against that copy, and
  `tests/test_methods_b_and_controls.py` does the same for lines B and C. A method that
  read anything on the truth side would fail on a missing file rather than pass. The suite
  was run in this working copy and its last line is:

      602 passed, 2 warnings in 237.43s (0:03:57)

- The passing half is not discharged, because there is no threshold in this tree for a
  submission to clear. `bars/national-v10/bars.json` is not frozen and its `gates` object
  is empty.
- What can be said about the eighteen reference reports is only the deterministic part, and
  it holds: `hard_pass=True` on all eighteen, from submissions built on participant files
  alone.
- What blocks a stochastic verdict is in the same measurement. On qual-1 all three lines
  report an undefined skill, because at the published total on that world both the
  proportional baseline and the perfect-information oracle leave nothing uncovered and the
  skill denominator is zero. A gate cannot be evaluated on a world where its statistic is
  undefined for every line, so even a frozen bar would not settle this obligation on that
  sixth of the world set.

Verdict: PARTIAL. The references are legal and structurally sound on all six qualification
worlds. Whether a strong reference passes is undecided, because nothing exists for it to
pass and one of the five composites is undefined on one world in six.

Next step: retarget the public reserve rate rule on submitted regional liability means
rather than submitted tails, require a candidate rate to keep the skill denominator
positive on all six qualification worlds, rebuild the eighteen packets at the accepted rate
from the ensemble cache, and rerun `scripts/build_v4_freeze_evidence.py` so that eighteen
final reference reports and their paired replicates exist. Obligation 1 cannot be answered
before that.

## Obligation 2

Verbatim: "The version-three Sol strategy (one growth factor, one global income scale,
additivity) fails for named reasons."

Artifacts.

- The strategy is registered as a control rather than described in prose. It is
  `version_three_recipe` in `meridia/methods/controls.py`, fitted by
  `fit_version_three_recipe` from the development worlds and run with the switches
  deterministic linkage, archive-only rates, ignored health selection, mean tail,
  proportional allocation, no reconstruction uncertainty, no raking to experience and no
  process or parameter noise. Its registered target composite is `release_accuracy`.
- P4 never ran it, so it was run here on all six qualification worlds of `worlds-p4`,
  against the calibration fit the P4 evidence tree already carries, and scored beside
  reference line A on the same world. One line of that measurement verbatim:

      qual-4 version_three_recipe hard_pass=True release_p95=2.1499 rates_p95=9.1644 cov_dev=0.9000 pooled_exc=0.9499 skill_loss=1.0000

- Exposure and rate error, control then reference line A, by world: qual-0 14.3329 against
  1.2574; qual-1 12.6134 against 1.4982; qual-2 11.5504 against 0.9314; qual-3 3.7095
  against 0.4132; qual-4 9.1644 against 8.2852; qual-5 17.3362 against 7.1264. Interval
  coverage deviation is pinned at 0.9000 on all six against 0.1632 to 0.7333 for the
  reference. Pooled exceedance deviation is 0.9500, 0.9500, 0.8152, 0.9500, 0.9499 and
  0.8001 against a five percent target, which is the point forecast failing the tail.
- So the named reasons hold on every world, in the quantities they name.
- The registered rule is stricter than that. `scripts/freeze_v4_bars.py` states the
  requirement as "every registered control hard-passes structure and fails its primary
  composite gate on every qualification world", and the primary composite here is release
  accuracy. Release accuracy for the recipe reads 0.4165, 0.6824, 1.5807, 1.0776, 2.1499
  and 1.7688 over the six worlds. On qual-0 the three reference lines read 0.8266, 0.7974
  and 0.8266, so the recipe is better there than every line a bar must admit, and no bar
  derived from reference attainment can fail it on that world. Measured against the worst
  of the three lines per world, the recipe is worse on its registered composite on five of
  six worlds and better on qual-0.

Verdict: PARTIAL. The strategy fails loudly for the reasons the protocol names, on every
world, in rates, intervals and tails. It does not fail its registered primary composite on
every qualification world, so the freeze rule as written cannot record it as a passing
destruction test.

Next step: rebind `version_three_recipe` to the composite its failure lives in. Its entry
in `CONTROL_TARGET_COMPOSITES` is `release_accuracy`; on this world set the separation is
in `exposures_and_rates` and `tail_calibration`, both by an order of magnitude on all six
worlds. Change the registration, or state and defend a second target, before the next
freeze attempt.

## Obligation 3

Verbatim: "Targeted ablations fail their intended gates: deterministic linkage,
archive-only incidence, no regime robustness, mean-only forecasting, normal-tail
approximation, proportional reserve allocation."

Artifacts.

- The battery is registered in `meridia/methods/controls.py`. Seventeen named controls plus
  five deletion controls make the twenty-two qualification controls, and two decomposition
  controls are development-only. An import-time check refuses to load the module unless
  every qualification control has exactly one target composite.
- P4 scored none of them, because the runner stopped in the reference stage.
- Seventeen were therefore run here on all six qualification worlds of `worlds-p4` and
  scored. The test applied is the only one a frozen bar could apply. A bar has to admit all
  three reference lines, so a control whose target-composite components all sit at or below
  the worst of the three lines on that world cannot be failed by any admissible bar.
- Counting worlds where a control is worse than the worst reference line on its own
  registered composite. On `exposures_and_rates`, deterministic_linkage 6 of 6 and
  ignore_health_selection 4 of 6. On `tail_calibration`, mean_only_tail 6 of 6, padded_tail
  6 of 6, normal_tail 1 of 6, development_average_regime 0 of 6. On `reserve_skill`,
  uniform_allocation 5 of 6 with qual-1 undefined, proportional_reserve 1 of 6 with qual-1
  undefined. On `interval_quality`, inflated_intervals 5 of 6. On `release_accuracy`,
  experience_history_only 5 of 6, version_three_recipe 5 of 6, survey_only 4 of 6,
  register_only 2 of 6, no_dedup 2 of 6, benchmark_only 2 of 6, exact_key_union 0 of 6,
  static_projection 0 of 6.
- Taking the protocol's six named ablations in order. Deterministic linkage with
  archive-only rates separates on all six, at an exposure and rate error of 4.15 to 18.89
  against a worst reference line of 0.57 to 8.29 on the same worlds. Mean-only forecasting
  separates on all six, at a pooled exceedance deviation of 0.2281 to 0.7146 against a five
  percent target. The other four do not. No regime robustness, the
  `development_average_regime` control, is inside the reference envelope on every component
  on all six worlds; on qual-0 its pooled exceedance deviation is 0.0390 and its q95 width
  error 1.5231, against reference-worst values of 0.0817 and 6.3714. The normal-tail
  approximation separates on qual-3 alone. The proportional reserve allocation separates on
  qual-2 alone.
- The proportional reserve result is the sharpest of these and is not a threshold question.
  That control is the published baseline, so its skill is exactly zero and its `skill_loss`
  exactly 1.0000 on the five worlds where the statistic is defined. The reference lines read
  0.1272 to 11.7725 on the same statistic, eight of the fifteen defined reports being worse
  than the baseline they are supposed to beat. A bar loose enough to admit the references
  admits the baseline, which is the blocker `tests/test_freeze_v4_composites.py` already
  anticipates by name.
- The five deletion controls and the two decomposition controls were not run here. They
  take a different entry point, and no P4 evidence exists for them either.

Verdict: NOT PROVEN. Two of the six named ablations separate on all six worlds. Three do
not separate on their intended composite on five or six of the six, and one of those three
cannot be separated by any bar the references pass, for a reason that lies in the reference
lines rather than in the control.

Next step: fix the reserve total first, since it is what leaves the skill statistic
undefined or uninformative, then rerun the full twenty-four control battery through
`scripts/build_v4_freeze_evidence.py` on the rebuilt packets and read the separation matrix
that `scripts/freeze_v4_bars.py` already computes. Any control still inside the reference
envelope afterwards is either misregistered, as the version-three recipe is, or is a layer
the deletion test says to remove.

## Obligation 4

Verbatim: "The decision has positive attainable value: a sealed-information reserve oracle
materially beats a strong frozen practical baseline."

Artifacts.

- `scripts/reserve_decision_value.py` asks exactly this question with truth in hand. The
  published total split by the public size-proportional rule is set against the same total
  spent by a perfect-information allocation, both paid on the sealed 2,048-member
  continuation ensemble, with a held-out variant that fits the oracle on half the ensemble
  and pays it on the other half. One line of its output on `worlds-p4` verbatim:

      - qual-2: R 257,625,000, 6 regions, 2048 continuations; slack -0.0045; regional tail width 0.056 to 0.188; J(A_B) 1,793,975, J(A*) 555,180, gain 69.05%, held out 64.11%

- The whole run at the compiled rate 4600, as published total, J at the baseline, J at the
  oracle, gain, and held-out gain:

```
world   published total   J baseline   J oracle    gain   held-out share
qual-0      263,053,000      610,763      3,497   99.43%     99.90%
qual-1      290,353,000            0          0    0.00%          .
qual-2      257,625,000    1,793,975    555,180   69.05%     64.11%
qual-3      272,129,000      657,659    134,501   79.55%     77.81%
qual-4      268,880,000    1,519,119    147,020   90.32%     90.03%
qual-5      280,027,000       11,587          0  100.00%     88.26%
```

- The same script on `worlds-p4-rate6321`, the tree built at the rate the published rule
  actually selects:

```
world   published total   J baseline   J oracle    gain   held-out share
qual-0      361,469,000            0          0    0.00%          .
qual-1      398,983,000            0          0    0.00%          .
qual-2      354,010,000        6,247          0  100.00%          .
qual-3      373,941,000            0          0    0.00%          .
qual-4      369,476,000            0          0    0.00%          .
qual-5      384,793,000            0          0    0.00%          .
```

  Held-out figures are shown as absent where both losses are zero, because the script's
  printed percentage there is a division by zero and carries no information. On qual-2 at
  this rate the held-out oracle is worse than the baseline by 18.90 percent, which is what
  fitting an allocation to 1,024 members of a 6,247-unit loss looks like.
- The value falls as the published total rises, and the crossing point differs by world.
  Sweeping the skill denominator, J at the baseline minus J at the oracle, over candidate
  rates on the same six worlds:

```
rate      qual-0       qual-1       qual-2       qual-3       qual-4       qual-5
3500    12,280,931   11,999,434    7,947,997   10,864,549    8,542,825    2,252,415
4000     7,808,026      751,752    1,913,525    1,582,765    4,590,601    1,000,871
4566       857,523            1    1,297,854      581,037    1,470,238       18,722
5000         2,795            0      675,162      136,655      312,093            0
6321             0            0        6,247            0            0            0
```

- Scale for materiality. Sealed mean total liability is 222,426,576 on qual-0, 221,559,250
  on qual-1, 227,891,941 on qual-2, 233,849,952 on qual-3, 231,662,867 on qual-4 and
  227,771,227 on qual-5. At the compiled rate the baseline's expected uncovered obligation
  is between 0.005 and 0.8 percent of that. The oracle's advantage is large as a fraction of
  the baseline's loss and small as a fraction of the liability being reserved against.

Verdict: PARTIAL. At the compiled rate the oracle beats the baseline by 69 to 100 percent of
the baseline's expected uncovered obligation on five of six worlds, and the held-out oracle
keeps 64 to 100 percent of that, so the gain is not fitted to Monte Carlo noise. On qual-1
both losses are zero and the decision has no value at all. At the rate the published rule
selects, five of six worlds have no value. The obligation holds at a rate the rule does not
choose and fails at the rate it does.

Next step: replace the reserve rate rule. Target the rate on submitted regional liability
means rather than on the submitted q95 sum plus a quarter of the ES95 gap, and accept a
candidate rate only when the skill denominator is positive on every qualification world at
that rate. The sweep above bounds the search: every world is positive at 4000, qual-1 is
down to one currency unit at 4566, and the current rule selects 6321. This changes how the
public total is derived, not any estimator, and no world has to be rebuilt from scratch,
only recompiled from the ensemble cache.

## What the four verdicts have in common

Three of the four turn on one quantity. The reserve total is set from the widest reference
line's submitted tail; that line's tails are wide enough that seven of the eighteen reports
show zero exceedance against a five percent target; and the resulting total is large enough
that the baseline and the oracle both cover almost everything. That leaves the skill
statistic undefined on one world, uninformative on most, and unable to separate the
proportional control from the reference lines a bar has to be looser than. Changing the
rate rule reopens obligations 1, 3 and 4 together, and obligation 2 needs a registration
change beside it.

## Reproducing the measurements

- Full suite: `PYTHONPATH=$PWD python3 -m pytest tests -q`.
- Decision value: `PYTHONPATH=$PWD python3 scripts/reserve_decision_value.py --packets`
  followed by the six qualification packet paths.
- Report scoring, control runs and the rate sweep call
  `meridia.verify.verify_actuarial_submission`, `meridia.methods.controls.run` and
  `meridia.actuarial.reserve_total` directly against the packet trees named at the top of
  this file.
