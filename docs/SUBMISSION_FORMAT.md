# Version-four submission format

The version-four submission directory contains exactly three regular CSV files. Symbolic
links, subdirectories, and any additional entry fail the deterministic file check. Column
names and their order are part of the contract.

## `release.csv`

Exact header:

```text
estimand,level,unit,sex,age_band,estimate,lower,upper
```

The file combines the population release with the exposure and rate release. Population
rows use the estimands and geographic levels listed in `contract.json`; `sex` and
`age_band` are empty on those rows. The three actuarial estimands are
`person_years_exposure`, `mortality_rate`, and `qualifying_event_rate`.

Actuarial rows are present for each state and county, for both published sexes. Exposure
rows contain the six attained-age bands `0-17`, `18-44`, `45-64`, `65-74`, `75-84`, and
`85+`, plus the broad bands `18-64` and `65+`. Rate rows contain the six attained-age
bands. The verifier reconstructs broad state rates by weighting the submitted fine rates
with the submitted fine exposures.

`estimate` is the submitted point value. `lower` and `upper` are the endpoints of the
submitted 90 percent interval. All three values must be finite, the lower endpoint must
not exceed the estimate, and the estimate must not exceed the upper endpoint. Exposure
and rate lower endpoints are nonnegative.

Counts and exposures obey the published geographic and age-band additivity rules. Rates
do not add. A state rate times its state exposure must agree with the sum of county rates
times county exposures.

Rate eligibility depends only on retained person-years exposure and is fixed before a
submitted value is read. The scored state-by-sex rate bands are `0-17`, `18-64`, and
`65+`. Their exposure floors are 600, 600, and 500 person-years respectively. The
`65-74`, `75-84`, and `85+` rates remain in the file as reported diagnostics and do not
create pass events. A freeze may complete only if all 72 state-by-sex `65+` cells across
the six qualification worlds clear the 500 person-year floor. The freeze report must
enumerate the 12 state-by-sex exposure values in each qualification world for every fine
and broad band, for 72 values per band. No P4 qualification values have yet been claimed.

## `projection.csv`

Exact header:

```text
estimand,level,unit,estimate,lower,upper
```

This file contains the horizon values of the population estimands listed in
`contract.json`. It has one row for every required estimand, geographic level, and unit.
Exposure and rate rows are not filed here. The future liability distribution is filed in
`reserve.csv`.

## `reserve.csv`

Exact header:

```text
region,liability_mean,q95,es95,allocation
```

There is one row for every state region. `liability_mean` is the submitted mean discounted
liability over the continuation distribution. `q95` is the submitted 0.95 quantile.
`es95` is the submitted mean liability at or above `q95`. `allocation` is the reserve
assigned to the region.

Every value is finite and nonnegative. The ordering is
`liability_mean <= q95 <= es95`. Allocations are finite and nonnegative, and they sum to
`contract.json` `reserve.total` within the published numerical tolerance. Submitted q95
and ES95 values are tail forecasts, not allocation floors. The tail gate scores them
directly against the continuation ensemble.

The `contract.json` `reserve.allocation_rule` object publishes the minimum allocation,
the required total, and the numerical tolerance used by the deterministic feasibility
check.

The reserve total is reproducible from participant-visible quantities. The
`contract.json` `reserve.total_rule` object contains these fields:

- `file`: `experience_history.csv`;
- `year`: the selection rule, `maximum published year`;
- `year_column`: `year`;
- `selected_year`: the resulting public year;
- `exposure_column`: `exposure`;
- `aggregation`: sum that column over every row in the selected year;
- `exposure_person_years`: the serialized public sum;
- `rate_per_person_year`: the registered public calibration rate accepted by the freeze;
- `rounding`: `up`;
- `rounding_unit`: the public monetary unit.

`reserve.total` is `exposure_person_years * rate_per_person_year`, rounded upward to
`rounding_unit`. A participant can recompute the exposure sum from the named file and
check it against the contract before allocating the total.

The 0.95 truth quantile is the order statistic at one-based rank
`ceiling(0.95 * M)` among the `M` continuation liabilities. Truth ES95 is the mean of all
members at or above that order statistic, including every tie.

## Scoring surface

Schema, additivity, rate consistency, and reserve feasibility are deterministic hard
checks. A structurally invalid submission cannot support a scientific control result.

Five stochastic composite gates decide a structurally valid submission:

- exposures and rates;
- release accuracy;
- interval quality;
- tail calibration;
- reserve skill.

The frozen bar document gives every component ceiling. Missing, incomplete, or old-format
bars stop scoring rather than supplying a default threshold.

The same document names the gate profile it froze under, and a profile selects which of
those five decide. Under the default `full` profile all five decide. Under `lite` the tail
block is measured and reported while exposures and rates, release accuracy, interval
quality, and reserve skill decide. Every verdict names the profile it came from, and a bar
document frozen under one profile cannot be scored under another.

## Participant input columns

`contract.json` publishes `participant_csv_schemas`, an exact map from every core input
CSV path to its ordered header. Packet construction checks the map against the written
files. Development-only truth tables sit outside this core map.

The two vintages of a source share one header. Every source file sits under `sources`
with a `_preliminary` or `_revised` suffix on the source name, and the survey pair sits at
the top of the participant tree instead. The core headers are:

- `experience_history.csv`: `year`, `age_band`, `sex`, `state`, `exposure`, `deaths`,
  `qualifying_events`, `net_migration`;
- `geography.csv`: `county`, `state`, `land_cells`, `economic_band`;
- the survey pair: `household`, `stratum`, `design_weight`, `age`, `sex`, `education`,
  `income`, `recent_hospitalization`, `county`, `psu`, `psu_sampled_households`;
- the benchmark series: `item`, `level`, `unit`, `value`;
- the business register: `record_id`, `business_id`, `enterprise_id`, `industry`,
  `county`, `employee_count`, `annual_payroll_cents`;
- the health encounters: `record_id`, `encounter_id`, `patient_id`, `facility_id`,
  `given_code`, `family_code`, `birth_tick`, `sex`, `patient_county`, `facility_county`,
  `admission_tick`, `discharge_tick`, `service`, `diagnosis_group`, `outcome`,
  `cost_cents`;
- the income records: `record_id`, `taxpayer_id`, `household_id`, `given_code`,
  `family_code`, `birth_tick`, `sex`, `county`, `employment_income_cents`, `employer_id`;
- the population register: `record_id`, `person_id`, `household_id`, `given_code`,
  `family_code`, `birth_tick`, `sex`, `education`, `county`.
