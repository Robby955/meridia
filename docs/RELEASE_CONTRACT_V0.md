# Release contract v0

What a published estimate table for a Meridia world must contain, how it is laid out, and how
it is scored against the retained population. This is the participant-facing half of the
capstone task; the scoring code in `meridia/scoring.py` implements exactly what is written
here, and `meridia/release.py` computes the truth it is scored against.

## Geography

Every nation is partitioned into states and counties (`meridia/admin.py`). Counties are the
catchments of county seats: the settlements in rank order, then the resource outposts, so
some counties are small and remote. States group counties around the largest settlements.
Every person and household belongs to exactly one county and one state; counts by county
add to their state and states add to the nation, exactly.

Units are integers. The nation is unit 0 at level `nation`; states are `0..S-1`; counties
are `0..C-1`. The participant learns `S` and `C` and the county-to-state map from the
release schema shipped with the packet, never from a map.

## Estimands

Each is published at all three levels: nation, state, county.

- `persons`: resident persons (count).
- `households`: households (count).
- `children_under_16`: persons aged 0 to 15 (count).
- `elders_65_plus`: persons aged 65 and over (count).
- `median_household_income`: median of household income, where household income is the
  sum of member incomes.
- `mean_income_adults`: mean income of persons aged 16 and over.
- `tertiary_share_25_plus`: share of persons aged 25 and over with tertiary or advanced
  education (education codes 2 and 3).
- `low_income_household_share`: share of households whose income is below 0.6 times the
  national median household income. The threshold is national at every level.

Counts are additive across the hierarchy. Means, medians, and shares are not.

## Release file

One flat table with exactly these columns:

```
estimand, level, unit, estimate, lower, upper
```

Rules the schema check enforces:

- exactly one row for every (estimand, level, unit); no extra rows, no duplicates;
- `estimate`, `lower`, `upper` finite numbers with `lower <= estimate <= upper`;
- counts, means, and medians have `lower >= 0`; shares lie in `[0, 1]`;
- intervals are 90 percent intervals;
- for the four counts, county estimates sum to their state estimate and state estimates
  sum to the national estimate (relative tolerance 1e-6). Intervals need not add.

A unit with no members must still have a row. Any values are accepted there; such units
are never scored.

## Detailed table and disclosure

The release also publishes one detailed table of person counts by county, age band
(`0-15`, `16-24`, `25-44`, `45-64`, `65+`), and sex, optionally with its totals: county by
age band, county by sex, county, and the national age band by sex. A suppressed cell or
total is published as missing.

A protected cell is one whose true count is above zero and below the threshold stated in
the packet. The audit fails, with no tolerance, if

- a protected cell is published;
- a suppressed protected cell is determined exactly by the published cells and totals,
  through any linear combination of them, including subtraction within one row; or
- published cells and totals disagree.

Complementary suppression is the participant's responsibility. Withholding a total is
allowed; publishing one that lets a protected cell be solved for is not.

## Scoring

All scoring uses the exact finite-population values.

- Error. For counts, means, and medians: `|estimate - truth| / max(|truth|, 1)`. For shares:
  `|estimate - truth|`. Reported for each estimand at each level as the worst unit and the
  mean; gates bind on the worst unit.
- Coverage. The share of a level's units whose interval contains the truth.
- Interval score. `(upper - lower) + (2 / 0.10) * distance of the truth outside the
  interval`, on the same scale as the error, averaged over units. Wide intervals raise it;
  missed truth raises it more. Gates bind on it so inflated intervals cannot buy coverage.

## Gates

A release passes only if every gate holds:

1. schema exact and additive;
2. worst-unit error at or below the frozen bar, for each estimand at each level;
3. coverage at or above the frozen floor and interval score at or below the frozen
   ceiling, everywhere;
4. disclosure audit clean.

Bars, floors, and ceilings are frozen from two executed strong pipelines before any trial
and shipped with the packet. Nothing in a free-text narrative is score bearing.

## Stage ten: projection and allocation

The release closes with a look ahead. The future is the world's own monthly ledger
replayed through the horizon, the same engine the observed sources were cut from. The participant publishes the same estimand table
for the next vintage, in the same schema, and commits an allocation: one non-negative
number per county whose sum does not exceed the stated budget. The world then runs forward
under its own dynamics, shocks included, and the future table is exact truth.

- The projection is scored exactly like the release: worst-unit error, coverage, and
  interval score against the future truth.
- The allocation is scored on realized loss: the share of true future demand left unmet.
  Demand in v0 is the future count of persons aged 65 and over by county. The oracle loss
  is what a perfect forecast could reach with the same budget, so regret, the difference,
  is what the participant's forecast cost. An allocation that is negative, non-finite, or
  over budget is infeasible and fails outright. A point commitment cannot be hedged with a
  wide interval.

## Not yet in v0

Movement classification between vintages and the executable audit trail are scored by
later versions of this contract; the same error, coverage, and interval-score definitions
apply to them.
