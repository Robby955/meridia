# Provenance of the version-four composite bars

Each component bar is the empirical p99 of that component's own values on the
reference line whose p99 for it is largest. Each line contributes 102
independent deterministic replicate reports; lines are never pooled for the
one-percent claim, and no component is carried to a ceiling another component
set. The one-percent false-fail target is per component and line.
Final witness reports are checked against the bars but are not resampled.
Scientific controls must pass the deterministic hard checks before a failure
can support a gate.

Schema: `meridia.v4.composite-bars.v1`.
Frozen: `false`.
Gate profile: `full`.


## Gate profile

- profile: full
- gates that decide: exposures_and_rates (p95_relative_error), release_accuracy (p95_relative_error), interval_quality (coverage_deviation, mean_interval_score), tail_calibration (pooled_exceedance_deviation, q95_width_relative_error, es95_width_relative_error), reserve_skill (skill_loss, worst_regional_shortfall_probability)
- measured and reported, deciding nothing: none
- reported-only components: none
- reference results above a reported bar:
  - none
- components published with no bar: none

## Authenticated evidence design

- measurement contract digest: `missing`
- common runner digest: `missing`
- final reference reports: 0
- paired replicate reports: 0; 0 resamples per world
- qualification control reports: 0
- development diagnostic reports: 0; these do not count as qualification controls
- unique run receipts: 0


## Empirical tail definition

q95 is order statistic ceil(0.95 * M) of the M continuations.
ES95 is the mean of all continuations tied at or above q95.

## False-fail accounting

- the rates below cover all 9 calibrated components of all 5 gates; 5 of those gates decide under the full profile
- target marginal product over nine components and three graded worlds: 0.762343
- conservative achieved conditional marginal-rate product: unavailable

The marginal products assume independent gate and world failures. They are arithmetic summaries, not empirical pass probabilities; failures can be correlated.

Only six qualification worlds support this freeze. Replicate false-fail rates are conditional on those worlds and do not establish a one-percent rate on new worlds.
