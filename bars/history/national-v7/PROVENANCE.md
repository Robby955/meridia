# Frozen bars, national v7 (version-three sources)

Status 2026-09-02: FROZEN on the nine qualification worlds. Both strong methods pass
every bar on all nine and all eight controls, including the `exact_key_union` replay of
the version-two count recipe, fail a named gate on all nine (`freeze_report.txt`,
`RESULT: bars frozen; every control fails a named gate`, `scripts/freeze_bars.py` exit
0). Digests: bars SHA-256
`f53983161245bb1eca4cb36a6a35325aba3c58262d9d46ef2cd97047d4f6aae9`, calibration A
`61cf3367661bee14155934be70e1afddf0ae4cf04c57494cd92537f6e2c0b107`, calibration B
`5fb058218393b113a2d2d16d7939aa0e69ca2a8f1041dac8954a9ab813dc6790`. The registered
sealed world index 4 was confirmed against this set; see
`seals/meridia-reconstruction-v3-confirmation.md`.

Derived by `scripts/freeze_bars.py` on the version-three sources (branch
`v3-hardening`), under the rule with per-kind, per-level floors and interval-score
ceilings that bind on pooled keys; margins unchanged from v6 (accuracy 1.25, coverage
slack 0.10, score 1.5, regret 2.0). Development worlds (calibration only, development
source regime): `dev-v3-1`, `dev-v3-2`, `dev-v3-3` (seeds 20260915, 20260917,
20260918). Qualification worlds (bars, hidden source regime): `qual-v3-1` through
`qual-v3-7` (seeds 20260916, 20260919 through 20260924) and `qual-v3-8`, `qual-v3-9`
(keyed worlds index 1 and 2 of `seals/meridia-reconstruction-v3.json`, retired after
their confirmations stopped). Generation is byte-deterministic: a packet rebuilt from
its seed reproduces every participant and retained file digest.

## What changed against v6

- The national count floor is restated from 0.03 to 0.05. Under version three the
  population source's coverage sits 0.02 to 0.08 below the development band, so the
  national count is read through the survey and the benchmark rather than the
  register, and the survey's own nonresponse total moves by a few percent between
  worlds while the benchmark carries a log-bias of magnitude 0.02 to 0.07. The v6 floor
  assumed the hidden world's coverage was exchangeable with the development worlds.
  Other floors are unchanged.
- An eighth control, `exact_key_union`, replays the count recipe of the version-two
  trajectory on the version-three surface: one row per exact name, birth-tick, and sex
  key in each source, the nation as the union of keys across the population, income,
  and health sources times a constant fitted on the development worlds, counties as the
  nation times the population source's reported county shares, widths from the
  between-world spread of the constant. It fails county persons on seven of the nine
  qualification worlds (0.41 to 1.40 against 0.3385, the exact key no longer
  identifying a person). On `qual-v3-5` and `qual-v3-7` the county gates hold for it
  and national elders separates it, at 0.0570 and 0.0626 against a bar of 0.05. How
  that bar reached its floor is in "How the set was closed".

## The fresh-world loop

Eight freezes were run. Each miss was closed by a general improvement to the reference
methods or by adding the missed world to the qualification set, never by moving a bar.
The last one is open.

1. Seven qualification worlds: NOT FROZEN. Both strong methods missed the pooled
   coverage floor on mean adult income on three worlds (design-based 0.000, 0.683,
   0.293; Bayesian 0.293, 0.683) with national misses of 5 to 29 percent. Cause: the
   income nonresponse correction is a line in the survey's income dispersion fitted on
   three development worlds whose dispersions (0.891 to 0.940) sat above every hidden
   world's (0.626 to 0.869); the line extrapolated to a factor of 0.76 where a factor
   near 1.1 was needed. Fix: both lines read the correction inside the development
   range of the dispersion only and hold the nearer edge beyond it (`dispersion_range`
   in the calibration receipts).
2. Seven worlds: frozen. Out-of-sample check on two fresh hidden-regime worlds
   (`fresh-v3-1`, seed 20260925; `fresh-v3-2`, seed 20260926): world 1 passed both
   methods; on world 2 the design-based line missed projected national children (0.161
   against 0.121). Cause: the projection drew the fertility rate uniformly from the
   public range (0.055 to 0.115). Fix: both lines estimate the fertility rate from the
   infant years of the deduplicated population source (births in the twenty-four ticks
   before the snapshot, halved, over women aged 18 to 45) and draw around it; on six
   inspected worlds the estimate sits within 3.5 percent of the realized rate.
3. Seven worlds: frozen; both fresh worlds passed. The one permitted confirmation on
   keyed world index 1 STOPPED: both lines missed national mean adult income (0.106 and
   0.116 against the 0.08 floor; the Bayesian line also the national median, 0.094
   against 0.09). That world's survey needed a nonresponse factor of 1.31 at a
   dispersion inside the development range, where the fitted line gives 1.19; nothing
   on the development side predicts it. The v6 record already showed a 0.093 worst
   miss on this key. Index 1 was retired to the qualification set as `qual-v3-8`.
4. Eight worlds: frozen; both fresh worlds passed. Confirmation on keyed world index 2
   STOPPED: both lines missed projected national households (0.097 and 0.109 against
   0.091; that world carries two migration waves inside the horizon, which no
   projection from the snapshot can foresee), and the design-based line missed pooled
   median-income coverage (0.659 against 0.70). Fix for the second miss: the
   design-based calibration receipt now records the correction's residual spread and
   widens the income intervals by it, the allowance the Bayesian line already carried.
   Index 2 was retired as `qual-v3-9`.
5. Nine worlds: NOT FROZEN. The design-based line still missed pooled median-income
   coverage on `qual-v3-9` (0.659): the fixed ten percent county model allowance for
   synthetic income items was too small under dated addresses and a 5 percent
   miscoding rate (county median bias spread 0.17). Fix: the design-based line now
   combines each county median and mean with the direct survey estimate where the
   county has four or more sampling units (as it already did for persons) and measures
   the synthetic model error from the residuals net of a delete-one-unit jackknife
   variance, never below ten percent.
6. Nine worlds: frozen; both strong methods passed all nine and all eight controls
   failed a named gate on all nine. Bars SHA-256
   `ce5a8cd8de3100fa12cc9f02e0a06fffe7f3588ddf36083fb09d8cd2402df73e`. The reference
   lines were reworked after this freeze, so this tree no longer reproduces that set,
   and the set is not in this directory. Its margin on the binding gate was 0.0044.

Out-of-sample check of the entry-6 bars on the two fresh date-derived worlds (not part
of the freeze): both strong methods PASS on both; all eight controls FAIL on both.

## Keyed confirmations: three stops, then index 4 closed

The one permitted confirmation on keyed world index 3 stopped: both lines missed the
county count bars by a wide margin (design-based persons/county worst 1.73, Bayesian
0.73, against the county persons bar of that freeze; households, children, elders
likewise) and both missed
median-income and low-income-share gates. That world drew a national total of 719,435
persons (the public size family is lognormal in density, sigma 0.55) with a smallest
county of 695 persons; at a 6.4 percent county miscoding rate, records misfiled from
large counties outnumber such a county's own population, and the misfiled-record
deconvolution in `register_counts` does not hold there. Every qualification and fresh
world so far had 1.6 to 4.1 million persons and a smallest county of 2,700 or more.
This is a small-county gap in the reference lines, not a bar-derivation matter: adding
index 3 to the qualification set would lift the county bars past the point where
the `exact_key_union` control passes.

The three stopped confirmations are kept with the packets, outside this repository, and
their receipt digests are listed above.

The one permitted confirmation on keyed world index 4, run against the frozen set in
this directory, PASSED: both strong witnesses cleared every bar and all eight controls
failed a named gate. Receipt SHA-256
`d8c246d9df4d6281de04507017aa1dde2056a5f94b3d3259f0bf9f85770ba58e`; the per-gate
numbers are in `seals/meridia-reconstruction-v3-confirmation.md`.

7. Ten worlds, index 3 added as `qual-v3-10`: NOT FROZEN, and abandoned. The freeze was
   run three times while the reference lines were reworked for small counties. Every run
   ended the way the entry above predicted: the county count bars widened far enough
   that `exact_key_union` cleared every bar on three qualification worlds. Index 3 is
   retired from grading and stays out of the qualification set; the small-county gap
   stays open. The qualification set is nine worlds.
8. Nine worlds: NOT FROZEN. Both strong methods passed every bar on all nine;
   `exact_key_union` cleared every bar on `qual-v3-5` and `qual-v3-7`. The gate that
   was left to separate it there was national elders: the control reached 0.0570 and
   0.0626, and the bar sat at 0.073711 because the Bayesian line reached 0.05897 on
   `qual-v3-2` while the design-based line stayed at 0.03574. The bar was set by the
   Bayesian line alone, and above the control. See "How the set was closed".
9. Nine worlds, current tree (this directory): FROZEN. Both strong methods pass every
   bar on all nine and all eight controls fail a named gate on all nine. Bars SHA-256
   `f53983161245bb1eca4cb36a6a35325aba3c58262d9d46ef2cd97047d4f6aae9`.

Out-of-sample check of these bars on the two fresh date-derived worlds (not part of the
freeze): both strong methods PASS on `fresh-v3-1` and `fresh-v3-2`; all eight controls
FAIL a named gate on both.

## How the set was closed

The cause of entry 8 was measured, not guessed. The Bayesian line's national count
errors tracked the benchmark's own bias almost one for one, because its national
posterior was wide enough that `benchmark_reconciliation` gave the county-up register
almost no weight:

- `qual-v3-1`: benchmark elders log-bias +0.0606; design-based error -0.0065, Bayesian
  +0.0455. Interval half-width over estimate: design-based 0.0567, Bayesian 0.1620.
- `qual-v3-2`: bias +0.0646; design-based +0.0204, Bayesian +0.0590. Half-widths 0.0588
  and 0.2238.
- `qual-v3-4`: bias +0.0653; design-based +0.0208, Bayesian +0.0563. Half-widths 0.0588
  and 0.1520.
- `qual-v3-5`: bias +0.0656; design-based +0.0150, Bayesian +0.0512. Half-widths 0.0630
  and 0.1389.
- `qual-v3-7`: bias +0.0288; design-based +0.0001, Bayesian +0.0264. Half-widths 0.0616
  and 0.1820.

Traced on `qual-v3-7`, the Bayesian line's relative posterior spread at the nation was
0.133 against the design-based line's bootstrap 0.035, and the inverse-variance step
put weight 0.113 on the register and 0.887 on the benchmark.

The closure is the second of the two candidates this file recorded, and it is a change
to the reference lines, not to a bar: `benchmark_reconciliation` now caps the variance
it assumes for the register at the coverage-model allowance
(`REGISTER_MODEL_RELATIVE_SD`, 0.025 relative), so no method can hand the benchmark
more than about a fifth of the weight. The register's error at the nation after
coverage correction is a model error rather than a sampling error, so the allowance is
the right ceiling on the variance the step may assume for it. On the three worlds that
were traced, the Bayesian line's national elders error fell from 0.05897, 0.05125 and
0.02639 to 0.03357, 0.02050 and 0.00931, and the design-based line's from 0.02042,
0.01501 and 0.00015 to 0.01069, 0.00185 and 0.00751.

With the reference lines closer to truth at the nation, the same freeze rule returns a
tighter set. The national elders bar falls from 0.073711 to its floor of 0.05, and
`exact_key_union` fails it at 0.0570 on `qual-v3-7` and 0.0626 on `qual-v3-5`. Every
national count bar and every county count bar except one tightened: persons/county from
0.345517 to 0.338514, children/county from 0.347663 to 0.339383, elders/county from
0.424038 to 0.383891, persons/nation from 0.071351 to 0.061475, children/nation from
0.065958 to 0.058858, households/nation from 0.071695 to 0.055525. The one that moved
the other way is households/county, from 0.272567 to 0.274994, which is the freeze rule
reading a slightly worse worst-county household error from the reference lines, not a
bar loosened by hand; `exact_key_union` fails on county persons, county children and
national elders regardless.

## Open: small-county register deconvolution

Index 3's stop is not closed. On a world whose smallest county holds a few hundred
persons, records misfiled from large counties outnumber that county's own population at
a 6.4 percent miscoding rate, and the misfiled-record deconvolution in `register_counts`
does not hold there. Every qualification and fresh world has 1.6 to 4.1 million persons
and a smallest county of 2,700 or more, so the freeze does not see it. The world stays
out of the qualification set, because a freeze including it lifts the county count bars
past the point where `exact_key_union` clears them.

## Version-two margin beside each version-three bar

`achieved_v2.json` is the re-grade of the version-two trajectory's three files against
the version-two hidden packet under the version-two verifier image and the v6 bars
(reward 1; file digests as in the banked trial). Each line: the v3 bar, the v6 bar, the
value that trajectory achieved, and achieved over the v3 bar. A small ratio at the
national counts records what the closed leaks were worth: the version-two nation was
the union count times a development-fitted constant with exact additivity, so its
national persons error was 0.0007 against a bar of 0.039; under version three the
national count has to be read through a coverage-shifted register, a survey, and a
biased benchmark, and the two reference lines need 0.05 to 0.075 at the nation. Where
a bar equals a floor (0.05, 0.08, 0.01, 0.02, 0.03 and their projection multiples) the
floor, not the reference lines, set it.

Release worst-unit error bars (achieved = worst error of the version-two trajectory on the version-two world)

- children_under_16/all: v3 bar 0.3394, v2 bar 0.3115, v2 achieved 0.0391, ratio 0.115
- children_under_16/county: v3 bar 0.3394, v2 bar 0.3115, v2 achieved 0.0391, ratio 0.115
- children_under_16/nation: v3 bar 0.0589, v2 bar 0.0392, v2 achieved 0.0045, ratio 0.076
- children_under_16/state: v3 bar 0.2286, v2 bar 0.3115, v2 achieved 0.0084, ratio 0.037
- elders_65_plus/all: v3 bar 0.3839, v2 bar 0.3266, v2 achieved 0.1354, ratio 0.353
- elders_65_plus/county: v3 bar 0.3839, v2 bar 0.3266, v2 achieved 0.1354, ratio 0.353
- elders_65_plus/nation: v3 bar 0.0500, v2 bar 0.0418, v2 achieved 0.0016, ratio 0.032
- elders_65_plus/state: v3 bar 0.2406, v2 bar 0.3266, v2 achieved 0.0205, ratio 0.085
- households/all: v3 bar 0.2750, v2 bar 0.3955, v2 achieved 0.1111, ratio 0.404
- households/county: v3 bar 0.2750, v2 bar 0.3955, v2 achieved 0.1111, ratio 0.404
- households/nation: v3 bar 0.0555, v2 bar 0.1013, v2 achieved 0.0009, ratio 0.016
- households/state: v3 bar 0.2138, v2 bar 0.3955, v2 achieved 0.0220, ratio 0.103
- low_income_household_share/all: v3 bar 0.2056, v2 bar 0.1705, v2 achieved 0.0796, ratio 0.387
- low_income_household_share/county: v3 bar 0.2056, v2 bar 0.1705, v2 achieved 0.0796, ratio 0.387
- low_income_household_share/nation: v3 bar 0.0638, v2 bar 0.0993, v2 achieved 0.0032, ratio 0.050
- low_income_household_share/state: v3 bar 0.2033, v2 bar 0.1425, v2 achieved 0.0134, ratio 0.066
- mean_income_adults/all: v3 bar 0.4007, v2 bar 0.3336, v2 achieved 0.1032, ratio 0.258
- mean_income_adults/county: v3 bar 0.4007, v2 bar 0.3336, v2 achieved 0.1032, ratio 0.258
- mean_income_adults/nation: v3 bar 0.1381, v2 bar 0.1167, v2 achieved 0.0221, ratio 0.160
- mean_income_adults/state: v3 bar 0.2800, v2 bar 0.2060, v2 achieved 0.0448, ratio 0.160
- median_household_income/all: v3 bar 0.4178, v2 bar 1.4969, v2 achieved 0.1779, ratio 0.426
- median_household_income/county: v3 bar 0.4178, v2 bar 1.4969, v2 achieved 0.1779, ratio 0.426
- median_household_income/nation: v3 bar 0.1111, v2 bar 0.0800, v2 achieved 0.0373, ratio 0.335
- median_household_income/state: v3 bar 0.3000, v2 bar 0.5490, v2 achieved 0.0665, ratio 0.222
- persons/all: v3 bar 0.3385, v2 bar 0.3199, v2 achieved 0.0634, ratio 0.187
- persons/county: v3 bar 0.3385, v2 bar 0.3199, v2 achieved 0.0634, ratio 0.187
- persons/nation: v3 bar 0.0615, v2 bar 0.0391, v2 achieved 0.0007, ratio 0.011
- persons/state: v3 bar 0.2359, v2 bar 0.3199, v2 achieved 0.0088, ratio 0.037
- tertiary_share_25_plus/all: v3 bar 0.0324, v2 bar 0.0300, v2 achieved 0.0065, ratio 0.200
- tertiary_share_25_plus/county: v3 bar 0.0324, v2 bar 0.0300, v2 achieved 0.0065, ratio 0.200
- tertiary_share_25_plus/nation: v3 bar 0.0100, v2 bar 0.0100, v2 achieved 0.0003, ratio 0.026
- tertiary_share_25_plus/state: v3 bar 0.0200, v2 bar 0.0200, v2 achieved 0.0012, ratio 0.061

Release interval-score ceilings (bind on pooled keys)

- children_under_16/all: v3 bar 1.6246, v2 bar 0.7997, v2 achieved 0.2737, ratio 0.168
- elders_65_plus/all: v3 bar 1.6888, v2 bar 0.8434, v2 achieved 0.4220, ratio 0.250
- households/all: v3 bar 1.7122, v2 bar 0.8195, v2 achieved 0.3437, ratio 0.201
- low_income_household_share/all: v3 bar 0.4844, v2 bar 0.4693, v2 achieved 0.0944, ratio 0.195
- mean_income_adults/all: v3 bar 0.9871, v2 bar 0.7500, v2 achieved 0.2892, ratio 0.293
- median_household_income/all: v3 bar 0.8858, v2 bar 3.0784, v2 achieved 1.2558, ratio 1.418
- persons/all: v3 bar 1.6411, v2 bar 0.8247, v2 achieved 0.2790, ratio 0.170
- tertiary_share_25_plus/all: v3 bar 0.0900, v2 bar 0.0900, v2 achieved 0.0108, ratio 0.120

Projection worst-unit error bars

- children_under_16/all: v3 bar 0.4650, v2 bar 0.4737, v2 achieved 0.2037, ratio 0.438
- children_under_16/county: v3 bar 0.4650, v2 bar 0.4737, v2 achieved 0.2037, ratio 0.438
- children_under_16/nation: v3 bar 0.0971, v2 bar 0.1543, v2 achieved 0.0118, ratio 0.121
- children_under_16/state: v3 bar 0.2898, v2 bar 0.2375, v2 achieved 0.0230, ratio 0.079
- elders_65_plus/all: v3 bar 0.4119, v2 bar 0.3779, v2 achieved 0.1478, ratio 0.359
- elders_65_plus/county: v3 bar 0.4119, v2 bar 0.3779, v2 achieved 0.1478, ratio 0.359
- elders_65_plus/nation: v3 bar 0.1279, v2 bar 0.1341, v2 achieved 0.0610, ratio 0.477
- elders_65_plus/state: v3 bar 0.3242, v2 bar 0.2387, v2 achieved 0.0653, ratio 0.201
- households/all: v3 bar 0.3185, v2 bar 0.3286, v2 achieved 0.0905, ratio 0.284
- households/county: v3 bar 0.3185, v2 bar 0.3286, v2 achieved 0.0905, ratio 0.284
- households/nation: v3 bar 0.1145, v2 bar 0.0559, v2 achieved 0.0023, ratio 0.020
- households/state: v3 bar 0.2543, v2 bar 0.3286, v2 achieved 0.0126, ratio 0.050
- low_income_household_share/all: v3 bar 0.1734, v2 bar 0.1582, v2 achieved 0.0568, ratio 0.328
- low_income_household_share/county: v3 bar 0.1734, v2 bar 0.1582, v2 achieved 0.0568, ratio 0.328
- low_income_household_share/nation: v3 bar 0.0378, v2 bar 0.0635, v2 achieved 0.0001, ratio 0.003
- low_income_household_share/state: v3 bar 0.1684, v2 bar 0.1234, v2 achieved 0.0047, ratio 0.028
- mean_income_adults/all: v3 bar 0.4626, v2 bar 0.4143, v2 achieved 0.0884, ratio 0.191
- mean_income_adults/county: v3 bar 0.4626, v2 bar 0.4143, v2 achieved 0.0884, ratio 0.191
- mean_income_adults/nation: v3 bar 0.1952, v2 bar 0.2046, v2 achieved 0.0178, ratio 0.091
- mean_income_adults/state: v3 bar 0.3909, v2 bar 0.3285, v2 achieved 0.0309, ratio 0.079
- median_household_income/all: v3 bar 0.5477, v2 bar 1.6988, v2 achieved 0.1611, ratio 0.294
- median_household_income/county: v3 bar 0.5477, v2 bar 1.6988, v2 achieved 0.1611, ratio 0.294
- median_household_income/nation: v3 bar 0.3173, v2 bar 0.2923, v2 achieved 0.0398, ratio 0.126
- median_household_income/state: v3 bar 0.5040, v2 bar 0.8202, v2 achieved 0.0601, ratio 0.119
- persons/all: v3 bar 0.3827, v2 bar 0.3789, v2 achieved 0.0664, ratio 0.174
- persons/county: v3 bar 0.3827, v2 bar 0.3789, v2 achieved 0.0664, ratio 0.174
- persons/nation: v3 bar 0.0823, v2 bar 0.0847, v2 achieved 0.0166, ratio 0.202
- persons/state: v3 bar 0.2574, v2 bar 0.2630, v2 achieved 0.0234, ratio 0.091
- tertiary_share_25_plus/all: v3 bar 0.1037, v2 bar 0.1000, v2 achieved 0.0132, ratio 0.127
- tertiary_share_25_plus/county: v3 bar 0.1037, v2 bar 0.1000, v2 achieved 0.0132, ratio 0.127
- tertiary_share_25_plus/nation: v3 bar 0.0454, v2 bar 0.0393, v2 achieved 0.0004, ratio 0.009
- tertiary_share_25_plus/state: v3 bar 0.0899, v2 bar 0.0713, v2 achieved 0.0018, ratio 0.021

Projection interval-score ceilings (pooled keys)

- children_under_16/all: v3 bar 1.6006, v2 bar 1.2017, v2 achieved 0.3970, ratio 0.248
- elders_65_plus/all: v3 bar 1.6606, v2 bar 1.0079, v2 achieved 0.3356, ratio 0.202
- households/all: v3 bar 1.6387, v2 bar 0.9759, v2 achieved 0.2868, ratio 0.175
- low_income_household_share/all: v3 bar 0.5682, v2 bar 0.6090, v2 achieved 0.0750, ratio 0.132
- mean_income_adults/all: v3 bar 1.0498, v2 bar 1.1309, v2 achieved 0.3665, ratio 0.349
- median_household_income/all: v3 bar 1.4040, v2 bar 3.1927, v2 achieved 1.4625, ratio 1.042
- persons/all: v3 bar 1.6307, v2 bar 0.9865, v2 achieved 0.3254, ratio 0.200
- tertiary_share_25_plus/all: v3 bar 0.1862, v2 bar 0.1759, v2 achieved 0.0301, ratio 0.161

Scalar gates

- coverage floor: v3 0.7, v2 0.7; v2 achieved pooled coverage minimum 0.927
- projection coverage floor: v3 0.7, v2 0.7; v2 achieved minimum 0.951
- allocation regret ceiling: v3 0.02380, v2 0.04346; v2 achieved 0.00110, ratio 0.046
