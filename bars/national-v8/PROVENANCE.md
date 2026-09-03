# Provenance of the version-four bars

Every threshold here was set by `scripts/freeze_v4_bars.py` from the worse of
two methodologically different witnesses on the qualification worlds named
below, times a margin declared in that script before any world was scored. No
graded world was opened, and no submission was looked at.

## Worlds

- development, used only to calibrate the two witnesses: dev-00, dev-01, dev-02, dev-03, dev-04, dev-05, dev-06, dev-07, dev-08, dev-09, dev-10, dev-11
- qualification, hidden source regime, the only worlds any bar reads: qual-0, qual-1, qual-2, qual-3, qual-4, qual-5
- graded: not opened by this script.

## Witnesses

- A, the design-based line, 100 bootstrap replicates, with the shared actuarial layer at its reference settings.
- B, the Bayesian line, 400 sweeps with a quarter burned in, through the same actuarial layer on its own posterior draws.

The two share the actuarial chain and differ in their reconstruction, which is
recorded in docs/INDEPENDENCE.md. A bar set from the worse of two lines that
agree on the chain is a weaker guarantee than one set from two independent
chains, and is read as such.

## Rule

- a ceiling is the worst witness value times its margin, never under its own
  floor, which only ever loosens it;
- a coverage or skill bar is the worst witness value minus its slack, and its
  constant is a cap on the bar rather than a floor under it: a constant above
  what both witnesses attain would not be a bar but a world neither can pass;
- margins: {"accuracy": 1.25, "coverage_slack": 0.1, "rate": 1.5, "score": 1.5, "shortfall": 1.5, "skill_slack": 0.15, "tail": 1.5};
- floors: {"accuracy": {"all": {"count": 0.15, "mean": 0.15, "median": 0.15, "proportion": 0.03}, "county": {"count": 0.15, "mean": 0.15, "median": 0.15, "proportion": 0.03}, "nation": {"count": 0.05, "mean": 0.08, "median": 0.08, "proportion": 0.01}, "state": {"count": 0.08, "mean": 0.12, "median": 0.12, "proportion": 0.02}}, "catastrophic": 0.5, "es_error": 0.1, "quantile_score": 0.05, "rate": {"mortality_rate": 0.25, "person_years_exposure": 0.1, "qualifying_event_rate": 0.25}, "rate_coverage_cap": 0.6, "shortfall": 0.15, "skill_cap": 0.6, "tau_mean": 0.02, "tau_worst": 0.05};
- a bar is then held inside the range its own criterion can take, at the
  declared attainability caps {"catastrophic_tail_ceiling": 0.5, "es_error_ceiling": 0.1, "exposure_error_ceiling": 0.5, "incidence_error_ceiling": 1.0, "mortality_error_ceiling": 1.0, "regional_shortfall_ceiling": 0.35, "tau_mean": 0.15, "tau_worst": 0.3} and the declared
  floor minima {"disclosure_utility_floor": 0.5, "skill_minimum": 0.05}. A bar that would be written outside that range is written at the limit
  and the witness then fails it, rather than inheriting a number that can never
  fire. The quantile score cap is 1.75 times the score the sealed truth itself
  pays on the same ensemble, which is 0.007679 here.

## Frozen actuarial bars

- catastrophic_tail_ceiling: 0.5
- es_error_ceiling: 0.1
- exposure_error_ceiling: 0.372072
- incidence_error_ceiling: 1.0
- mortality_error_ceiling: 1.0
- quantile_score_ceiling: 0.013438
- rate_coverage_floor: 0.0
- regional_shortfall_ceiling: 0.35
- skill_minimum: 0.05
- tau_mean: 0.15
- tau_worst: 0.3

## Whether this set froze

`bars.json` records `frozen`: false. It is written
after the control battery, not before it, and `verify_submission` refuses to
gate a version-four submission with a set that does not record a completed
freeze. A freeze completes when both witnesses clear every bar on every
qualification world and every control fails at least one named gate. The freeze
report names what blocked it when it does not.

## Reading them

`verify_submission(packet, submission, bars)` reads `bars["actuarial"]` for the
version-four gates and falls back to the placeholders in `ActuarialThresholds`
for any key that is absent. A metric with no threshold is reported and never
gates.
