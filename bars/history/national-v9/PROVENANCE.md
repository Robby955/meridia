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
  declared attainability caps {"catastrophic_tail_ceiling": 0.5, "detailed_accuracy_ceiling": 0.5, "es95_width_error_ceiling": 0.5, "es_error_ceiling": 0.1, "exposure_error_ceiling": 0.5, "incidence_error_ceiling": 1.0, "mortality_error_ceiling": 1.0, "q95_width_error_ceiling": 0.5, "regional_shortfall_ceiling": 0.35, "tau_mean": 0.15, "tau_worst": 0.3} and the declared
  floor minima {"disclosure_utility_floor": 0.5, "skill_minimum": 0.05}. A bar that would be written outside that range is written at the limit
  and the witness then fails it, rather than inheriting a number that can never
  fire. A bar that still cannot fire once clamped, or that carries no declared
  range at all, stops the freeze: the report names it and the run does not
  complete. The quantile score cap is 1.75 times the score the sealed truth itself
  pays on the same ensemble, which is 0.007372 here.

## Frozen actuarial bars

- catastrophic_tail_ceiling: 0.5
- es95_width_error_ceiling: 0.5
- es_error_ceiling: 0.1
- exposure_error_ceiling: 0.342492
- incidence_error_ceiling: 1.0
- mortality_error_ceiling: 0.698499
- q95_width_error_ceiling: 0.5
- quantile_score_ceiling: 0.0129
- rate_coverage_floor: 0.449
- regional_shortfall_ceiling: 0.35
- skill_minimum: 0.05
- tau_mean: 0.15
- tau_worst: 0.3
- disclosure_utility_floor: 0.889
- detailed_accuracy_ceiling: 0.116612

The two width bars score a submitted q95 and ES95 against the sealed ones in
units of the ensemble's own regional tail width. A tail that is out by its whole
width reads 1.0 there and a fraction of a percent on the level, which is how a
mean-only tail and a doubled one both cleared the first pass's tail bars.

## Which cells the rate bars read

A rate ceiling is only as strong as the cells its eligibility rule admits. The
floors are derived from a published rule rather than set per band: every cell
stands on at least 120 expected persons over the scored
window, and a rate cell also carries at least 25 expected events of its
own kind at the published reference rate for its band. The derived floors, in
person-years:

- mortality_rate: 0-17 50,000, 18-44 16,667, 18-64 5,000, 45-64 2,500, 65+ 600, 65-74 600, 75-84 600, 85+ 600
- person_years_exposure: 0-17 600, 18-44 600, 18-64 600, 45-64 600, 65+ 600, 65-74 600, 75-84 600, 85+ 600
- qualifying_event_rate: 0-17 1,250, 18-44 833, 18-64 714, 45-64 625, 65+ 600, 65-74 600, 75-84 600, 85+ 625

The cells each gated block actually read, per world and witness:

- mortality_rate/state: bands 18-44, 45-64, 65-74, 75-84, 40 distinct cells, counts {"qual-0/A": 24, "qual-0/B": 24, "qual-1/A": 32, "qual-1/B": 32, "qual-2/A": 26, "qual-2/B": 26, "qual-3/A": 27, "qual-3/B": 27, "qual-4/A": 29, "qual-4/B": 29, "qual-5/A": 31, "qual-5/B": 31}
- person_years_exposure/county: bands 0-17, 18-64, 65+, 108 distinct cells, counts {"qual-0/A": 88, "qual-0/B": 88, "qual-1/A": 100, "qual-1/B": 100, "qual-2/A": 93, "qual-2/B": 93, "qual-3/A": 83, "qual-3/B": 83, "qual-4/A": 92, "qual-4/B": 92, "qual-5/A": 91, "qual-5/B": 91}
- qualifying_event_rate/state: bands 0-17, 18-44, 45-64, 65-74, 75-84, 58 distinct cells, counts {"qual-0/A": 50, "qual-0/B": 50, "qual-1/A": 54, "qual-1/B": 54, "qual-2/A": 50, "qual-2/B": 50, "qual-3/A": 51, "qual-3/B": 51, "qual-4/A": 53, "qual-4/B": 53, "qual-5/A": 53, "qual-5/B": 53}

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
