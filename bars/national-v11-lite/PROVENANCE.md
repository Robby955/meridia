# Provenance of the version-four composite bars

Each gate ceiling is the maximum of the three reference-line empirical p99
order statistics of per-row max severity under fixed component normalizers.
Each line contributes 102 independent deterministic replicate reports; lines
are never pooled for the one-percent claim. Component records preserve their
own empirical diagnostics but do not make marginal false-fail claims.
Final witness reports are checked against the bars but are not resampled.
Scientific controls must pass the deterministic hard checks before a failure
can support a gate.

Schema: `meridia.v4.composite-bars.v1`.
Frozen: `false`.
Gate profile: `lite`.


## Gate profile

- profile: lite
- gates that decide: exposures_and_rates (p95_relative_error), release_accuracy (p95_relative_error), interval_quality (coverage_deviation, mean_interval_score), reserve_skill (skill_loss)
- measured and reported, deciding nothing: tail_calibration, reserve_skill (worst_regional_shortfall_probability)
- reference results above a reported bar:
  - none

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

- the rates below cover all 5 calibrated gates; 4 of them decide under the lite profile
- target marginal product over five gates and three graded worlds: 0.860058
- conservative achieved conditional marginal-rate product: unavailable

The marginal products assume independent gate and world failures. They are arithmetic summaries, not empirical pass probabilities; failures can be correlated.

Only six qualification worlds support this freeze. Replicate false-fail rates are conditional on those worlds and do not establish a one-percent rate on new worlds.
